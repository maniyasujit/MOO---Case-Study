#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II + SMS-EMOA on WFG4 (case study) -- built on the Exercise 4 handout.

Shared block (dominates .. SMSEMOA) is identical to the DTLZ files.
WFG4-specific (our non-DTLZ problem): the Pareto front is a SCALED sphere
octant, sum_i (f_i / (2i))^2 = 1, so objective i lives in [0, 2i] -- the
objectives are deliberately given DIFFERENT ranges (f1 in [0,2], f2 in [0,4],
f3 in [0,6]).  The decision variables are also scaled, z_i in [0, 2i].  The
shape function is highly MULTIMODAL, so unlike DTLZ2/DTLZ4 (unimodal g) this
problem tests convergence through many local optima, like DTLZ1 does -- but
with the added twist of the non-uniform objective scaling, which stresses any
selection mechanism that implicitly assumes equal objective ranges.
pymoo is used ONLY for the benchmark problem and the HV / IGD+ indicators.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import time


# ---- from the Ex4 handout  dominates and non dominated ones --------------------------------------

def dominates(p, q):
    return np.all(p <= q) and np.any(p < q)


def fast_non_dominated_sort(F):
    F = np.asarray(F, float)
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] <  F[None, :, :]).any(2)
    D = le & lt                       # D[i, j] = i dominates j
    ndom = D.sum(0).astype(int)       # how many dominate i
    fronts, cur = [], list(np.where(ndom == 0)[0])
    while cur:
        fronts.append(np.array(cur))
        for i in cur:
            ndom[np.where(D[i])[0]] -= 1
        ndom[cur] = -1
        cur = list(np.where(ndom == 0)[0])
    return fronts

# ---- crowding distance --------------------------

def crowding_distance(F):
    """Crowding distance within one front (lecture formula; boundary -> inf).
    Note the per-objective span normalisation: on WFG4 this is what keeps the
    f2 (range 4) / f3 (range 6) axes from drowning out f1 (range 2)."""
    k, m = F.shape
    if k <= 2:
        return np.full(k, np.inf)
    cd = np.zeros(k)
    for j in range(m):
        order = np.argsort(F[:, j])
        cd[order[0]] = cd[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span > 0:
            cd[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return cd


# ---- real-valued operators (SBX crossover + polynomial mutation) -----------

def sbx(p1, p2, xl, xu, rng, eta=15.0, p_c=0.9):
    """Bounded SBX: the spread factor beta is derived from each variable's
    distance to its bounds (not naive clipping). On WFG4 the bounds differ per
    variable (z_i in [0, 2i]) -- the operator handles that naturally because
    everything is expressed relative to xl[i] / xu[i]."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > p_c:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = p1[i], p2[i]
        lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
        u = rng.random()

        def betaq(beta_bound):                      # inverse-CDF with bound
            alpha = 2.0 - beta_bound ** (-(eta + 1.0))
            return (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha \
                else (1 / (2 - u * alpha)) ** (1 / (eta + 1))

        v1 = 0.5 * ((lo + hi) - betaq(1 + 2 * (lo - xl[i]) / (hi - lo)) * (hi - lo))
        v2 = 0.5 * ((lo + hi) + betaq(1 + 2 * (xu[i] - hi) / (hi - lo)) * (hi - lo))
        c1[i], c2[i] = np.clip(v1, xl[i], xu[i]), np.clip(v2, xl[i], xu[i])
    return c1, c2


def poly_mutation(x, xl, xu, rng, eta=20.0):
    """Bounded polynomial mutation: perturbation derived from the variable's
    distance to its bounds (textbook form); per-variable span again makes the
    step size follow each z_i's own [0, 2i] range."""
    y, p_m = x.copy(), 1.0 / len(x)
    for i in range(len(x)):
        if rng.random() >= p_m:
            continue
        span = xu[i] - xl[i]
        u = rng.random()
        d1, d2 = (y[i] - xl[i]) / span, (xu[i] - y[i]) / span
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        y[i] = np.clip(y[i] + dq * span, xl[i], xu[i])
    return y


# ---- NSGA-II (same class types as the handout's EMOA) ---------------------

class NSGA2:
    def __init__(self, problem, mu=100, seed=None):
        self.problem = problem
        self.mu = mu                       # population size  (lambda = mu)
        self.xl, self.xu = problem.xl, problem.xu
        self._rng = np.random.default_rng(seed)

    def _evaluate(self, X):
        return np.atleast_2d(self.problem.evaluate(np.array(X)))

    def _tournament(self, rank, cd):
        a, b = self._rng.integers(0, self.mu, size=2)
        if rank[a] != rank[b]:
            return a if rank[a] < rank[b] else b
        return a if cd[a] >= cd[b] else b      # tie -> larger crowding wins

    def _survive(self, X, F):
        keep = []
        for front in fast_non_dominated_sort(F):
            if len(keep) + len(front) <= self.mu:
                keep += list(front)
            else:
                cd = crowding_distance(F[front])
                order = front[np.argsort(-cd)]
                keep += list(order[: self.mu - len(keep)])
                break
        keep = np.array(keep)
        return X[keep], F[keep]

    def run(self, n_evals=20000):
        X = self._rng.uniform(self.xl, self.xu, size=(self.mu, self.problem.n_var))
        F = self._evaluate(X)
        evals = self.mu
        while evals + self.mu <= n_evals:
            rank = np.empty(self.mu, int); cd = np.empty(self.mu)
            for r, fr in enumerate(fast_non_dominated_sort(F)):
                rank[fr] = r; cd[fr] = crowding_distance(F[fr])
            # ---- create mu offspring ----
            Q = []
            while len(Q) < self.mu:
                a, b = self._tournament(rank, cd), self._tournament(rank, cd)
                c1, c2 = sbx(X[a], X[b], self.xl, self.xu, self._rng)
                Q.append(poly_mutation(c1, self.xl, self.xu, self._rng))
                if len(Q) < self.mu:
                    Q.append(poly_mutation(c2, self.xl, self.xu, self._rng))
            Q = np.array(Q); FQ = self._evaluate(Q); evals += self.mu
            # ---- (mu + mu) survival via NDS + crowding ----
            X, F = self._survive(np.vstack([X, Q]), np.vstack([F, FQ]))
        return X, F


# ---- hypervolume contributions (for SMS-EMOA) ------------------

def hv_contributions(F, ref):
    """Exclusive hypervolume contribution of each point of an anti-chain F.
    p = 2 : exact O(k log k) neighbour-rectangle method (the geometric picture
            from Exercise Sheet 3).  p >= 3: moocore's C routine (ships w/ pymoo)."""
    F = np.asarray(F, float)
    k, m = F.shape
    if k == 1:
        return np.array([float(np.prod(ref - F[0]))])
    if m == 2:
        order = np.argsort(F[:, 0])
        S = F[order]
        right_f1 = np.append(S[1:, 0], ref[0])            # right neighbour in f1
        upper_f2 = np.concatenate(([ref[1]], S[:-1, 1]))  # left neighbour in f2
        contrib = (right_f1 - S[:, 0]) * (upper_f2 - S[:, 1])
        out = np.empty(k); out[order] = contrib
        return out
    import moocore
    return moocore.hv_contributions(F, ref=ref)


# ---- SMS-EMOA (same operators + NDS as NSGA-II; (mu + 1) steady state) ------

class SMSEMOA:
    """SMS-EMOA (Beume, Naujoks & Emmerich 2007), "variant 2".
    Reuses the SAME sbx, poly_mutation and fast_non_dominated_sort as NSGA-II.
    Differs only in: (mu + 1) steady state (one offspring per iteration), and
    survival drops the point with the lowest hypervolume contribution in the
    last front (instead of NSGA-II's crowding-distance truncation)."""

    def __init__(self, problem, mu=100, seed=None):
        self.problem = problem
        self.mu = mu
        self.xl, self.xu = problem.xl, problem.xu
        self._rng = np.random.default_rng(seed)

    def _evaluate(self, X):
        return np.atleast_2d(self.problem.evaluate(np.array(X)))

    def run(self, n_evals=20000):
        X = self._rng.uniform(self.xl, self.xu, size=(self.mu, self.problem.n_var))
        F = self._evaluate(X)
        evals = self.mu
        while evals < n_evals:
            # ---- one offspring from two random parents ----
            i, j = self._rng.choice(self.mu, size=2, replace=False)
            c1, _ = sbx(X[i], X[j], self.xl, self.xu, self._rng)
            o = poly_mutation(c1, self.xl, self.xu, self._rng)
            fo = self._evaluate(o[None, :])[0]
            evals += 1
            Xall = np.vstack([X, o[None, :]]); Fall = np.vstack([F, fo[None, :]])
            # ---- (mu + 1) survival: remove worst point of the last front ----
            fronts = fast_non_dominated_sort(Fall)
            last = fronts[-1]
            if len(last) == 1:                          # singleton last front
                remove = last[0]
            else:                                       # lowest HV contribution
                F_last = Fall[last]
                ref = F_last.max(axis=0) + 1.0          # dynamic reference point
                remove = last[int(np.argmin(hv_contributions(F_last, ref)))]
            keep = np.ones(self.mu + 1, bool); keep[remove] = False
            X, F = Xall[keep], Fall[keep]
        return X, F


# ---- running the algorithms on WFG4 ----------------------------------------

if __name__ == "__main__":
    fig = plt.figure(figsize=(11, 4.5))

    # WFG4 front: scaled sphere octant  sum_i (f_i / (2i))^2 = 1, i.e.
    # nadir = (2, 4) for p=2 and (2, 4, 6) for p=3 -- objectives have
    # DIFFERENT ranges, so unlike the DTLZ files the reference point must be
    # a vector: 1.1 * nadir per objective (same 1.1-rule as everywhere else).
    NADIR2, NADIR3 = np.array([2.0, 4.0]), np.array([2.0, 4.0, 6.0])
    REF2, REF3 = 1.1 * NADIR2, 1.1 * NADIR3
    BUDGET = 20000
    N_SEEDS = 10

    # ---- IGD+ : inverted generational distance (modified).  For every point on
    #      the TRUE front, measure the distance to the nearest solution we found
    #      -- but only the "wrong side" (where our point is worse) is counted --
    #      then average over the front.  Lower = better (0 = front fully covered).
    #      WFG4's multimodal shape function makes full convergence hard, so like
    #      on DTLZ1, IGD+ stays informative even when HV differences look small.
    #      true (reference) front for WFG4 : the DTLZ2 sphere octant scaled by
    #      (2, 4[, 6]) per objective.
    t = np.linspace(0, np.pi / 2, 200)
    pf2 = np.column_stack([2 * np.cos(t), 4 * np.sin(t)])       # scaled quarter circle
    u = np.linspace(0, np.pi / 2, 50); v = np.linspace(0, np.pi / 2, 50)
    U, V = np.meshgrid(u, v)
    pf3 = np.column_stack([2 * (np.cos(U) * np.cos(V)).ravel(),
                           4 * (np.cos(U) * np.sin(V)).ravel(),
                           6 * np.sin(U).ravel()])              # scaled sphere octant
    igd2, igd3 = IGDPlus(pf2), IGDPlus(pf3)

    # --- p = 2: front is a curve -> 2D scatter ---
    prob2 = get_problem("wfg4", n_var=10, n_obj=2)
    print(f"Setup: n=10 decision vars, mu=100 population, budget={BUDGET} evals/run, {N_SEEDS} seeds")
    print(f"WFG4 p=2  (true-front HV ~ {np.prod(REF2) - np.pi/4 * 2 * 4:.4f}, scaled sphere, multimodal)")

    # keep every front and later PLOT the run with the MEDIAN hypervolume,
    # i.e. a typical run, not an arbitrary one (same convention as DTLZ4.py).
    nsga_hv2, nsga_igd2, nsga_rt2, nsga_fr2 = [], [], [], []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()   #checking time before run
        _, F2 = NSGA2(prob2, mu=100, seed=seed).run(n_evals=BUDGET)
        nsga_rt2.append(time.perf_counter() - t0) #checking the time after the run
        front2 = F2[fast_non_dominated_sort(F2)[0]]
        nsga_fr2.append(front2)
        nsga_hv2.append(HV(ref_point=REF2)(front2))
        nsga_igd2.append(igd2(front2))
    print(f"  NSGA-II : HV median={np.median(nsga_hv2):.4f}  IGD+ median={np.median(nsga_igd2):.4f}  min={np.min(nsga_hv2):.4f}  max={np.max(nsga_hv2):.4f}  rt median={np.median(nsga_rt2):.2f}s")

    sms_hv2, sms_igd2, sms_rt2, sms_fr2 = [], [], [], []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()   #checking time before run
        _, S2 = SMSEMOA(prob2, mu=100, seed=seed).run(n_evals=BUDGET)
        sms_rt2.append(time.perf_counter() - t0) #checking the time after the run
        sfront2 = S2[fast_non_dominated_sort(S2)[0]]
        sms_fr2.append(sfront2)
        sms_hv2.append(HV(ref_point=REF2)(sfront2))
        sms_igd2.append(igd2(sfront2))
    print(f"  SMS-EMOA: HV median={np.median(sms_hv2):.4f}  IGD+ median={np.median(sms_igd2):.4f}  min={np.min(sms_hv2):.4f}  max={np.max(sms_hv2):.4f}  rt median={np.median(sms_rt2):.2f}s")

    # median-HV run of each algorithm (representative, not best or worst)
    front2 = nsga_fr2[int(np.argsort(nsga_hv2)[N_SEEDS // 2])]
    sfront2 = sms_fr2[int(np.argsort(sms_hv2)[N_SEEDS // 2])]

    ax1 = fig.add_subplot(1, 2, 1)
    t = np.linspace(0, np.pi / 2, 200)
    ax1.plot(2 * np.cos(t), 4 * np.sin(t), "lightgrey", label="true front")
    ax1.scatter(front2[:, 0], front2[:, 1], s=12, color="tab:blue", label="NSGA-II")
    ax1.scatter(sfront2[:, 0], sfront2[:, 1], s=12, marker="^",
                color="tab:green", label="SMS-EMOA")
    ax1.set_xlabel("$f_1$"); ax1.set_ylabel("$f_2$")
    ax1.set_title("WFG4, $p=2$"); ax1.legend()

    # --- p = 3: 3D scatter
    prob3 = get_problem("wfg4", n_var=10, n_obj=3)
    print(f"WFG4 p=3  (true-front HV ~ {np.prod(REF3) - np.pi/6 * 2 * 4 * 6:.4f}, scaled sphere, multimodal)")

    nsga_hv3, nsga_igd3, nsga_rt3, nsga_fr3 = [], [], [], []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()   #checking time before run
        _, F3 = NSGA2(prob3, mu=100, seed=seed).run(n_evals=BUDGET)
        nsga_rt3.append(time.perf_counter() - t0) #checking the time after the run
        front3 = F3[fast_non_dominated_sort(F3)[0]]
        nsga_fr3.append(front3)
        nsga_hv3.append(HV(ref_point=REF3)(front3))
        nsga_igd3.append(igd3(front3))
    print(f"  NSGA-II : HV median={np.median(nsga_hv3):.4f}  IGD+ median={np.median(nsga_igd3):.4f}  min={np.min(nsga_hv3):.4f}  max={np.max(nsga_hv3):.4f}  rt median={np.median(nsga_rt3):.2f}s")

    sms_hv3, sms_igd3, sms_rt3, sms_fr3 = [], [], [], []
    for seed in range(N_SEEDS):
        t0 = time.perf_counter()   #checking time before run
        _, S3 = SMSEMOA(prob3, mu=100, seed=seed).run(n_evals=BUDGET)
        sms_rt3.append(time.perf_counter() - t0) #checking the time after the run
        sfront3 = S3[fast_non_dominated_sort(S3)[0]]
        sms_fr3.append(sfront3)
        sms_hv3.append(HV(ref_point=REF3)(sfront3))
        sms_igd3.append(igd3(sfront3))
    print(f"  SMS-EMOA: HV median={np.median(sms_hv3):.4f}  IGD+ median={np.median(sms_igd3):.4f}  min={np.min(sms_hv3):.4f}  max={np.max(sms_hv3):.4f}  rt median={np.median(sms_rt3):.2f}s")

    # median-HV run of each algorithm (representative, not best or worst)
    front3 = nsga_fr3[int(np.argsort(nsga_hv3)[N_SEEDS // 2])]
    sfront3 = sms_fr3[int(np.argsort(sms_hv3)[N_SEEDS // 2])]

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    u = np.linspace(0, np.pi / 2, 30); v = np.linspace(0, np.pi / 2, 30)
    U, V = np.meshgrid(u, v)
    ax2.plot_surface(2 * np.cos(U) * np.cos(V), 4 * np.cos(U) * np.sin(V),
                     6 * np.sin(U), color="lightgrey", alpha=0.2, linewidth=0)
    ax2.scatter(front3[:, 0], front3[:, 1], front3[:, 2], s=12,
                color="tab:orange", label="NSGA-II")
    ax2.scatter(sfront3[:, 0], sfront3[:, 1], sfront3[:, 2], s=12, marker="^",
                color="tab:green", label="SMS-EMOA")
    ax2.set_xlabel("$f_1$"); ax2.set_ylabel("$f_2$"); ax2.set_zlabel("$f_3$")
    ax2.set_title("WFG4, $p=3$"); ax2.legend()

    fig.suptitle("NSGA-II vs SMS-EMOA on WFG4 ($n=10$)")
    fig.tight_layout()
    fig.savefig("nsga2_wfg4.png", dpi=140)
    print("saved nsga2_wfg4.png")
    # ---- boxplots: HV and IGD+ distributions across seeds ----
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(10, 4))
    tick = ["NSGA\np=2", "SMS\np=2", "NSGA\np=3", "SMS\np=3"]
    bx1.boxplot([nsga_hv2, sms_hv2, nsga_hv3, sms_hv3])
    bx1.set_xticklabels(tick)   # works on old and new matplotlib alike
    bx1.set_title("WFG4 - Hypervolume"); bx1.set_ylabel("HV")
    bx2.boxplot([nsga_igd2, sms_igd2, nsga_igd3, sms_igd3])
    bx2.set_xticklabels(tick)
    bx2.set_title("WFG4 - IGD+"); bx2.set_ylabel("IGD+")
    fig2.tight_layout()
    fig2.savefig("boxplots_wfg4.png", dpi=140)
    plt.show()
