"""Shared building blocks for the from-scratch NSGA-II and SMS-EMOA implementations.

Fast non-dominated sorting and crowding distance follow Deb et al. (2002).
SBX crossover and polynomial mutation follow the real-coded operators used in the
original NSGA-II paper and pymoo's reference implementation.
"""
import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if objective vector a dominates b (minimisation)."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Fast non-dominated sorting (Deb et al., 2002).

    The pairwise dominance relation (the expensive O(n^2 * p) part) is
    vectorised with numpy; the O(n^2) front-peeling bookkeeping that is left
    follows the original algorithm exactly.

    Parameters
    ----------
    F : (n, p) array of objective values, minimisation assumed.

    Returns
    -------
    fronts : list of index arrays, fronts[0] is the non-dominated front.
    """
    n = len(F)
    leq = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    less = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = leq & less  # dom[i, j] == True iff i dominates j
    np.fill_diagonal(dom, False)

    domination_count = dom.sum(axis=0)
    dominated_sets = [np.where(dom[i])[0] for i in range(n)]

    fronts = []
    current_front = list(np.where(domination_count == 0)[0])
    while current_front:
        fronts.append(np.array(current_front))
        next_front = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance assignment (Deb et al., 2002) for a single front."""
    n, p = F.shape
    cd = np.zeros(n)
    if n <= 2:
        cd[:] = np.inf
        return cd

    for j in range(p):
        order = np.argsort(F[:, j])
        f_min, f_max = F[order[0], j], F[order[-1], j]
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        denom = f_max - f_min
        if denom == 0:
            continue
        for i in range(1, n - 1):
            cd[order[i]] += (F[order[i + 1], j] - F[order[i - 1], j]) / denom
    return cd


def crowded_comparison_key(rank: np.ndarray, cd: np.ndarray):
    """Return keys such that sorting ascending reproduces the NSGA-II
    crowded-comparison order (lower rank wins, then higher crowding distance)."""
    return list(zip(rank, -cd))


def sbx_crossover(x1: np.ndarray, x2: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                   eta: float, prob: float, rng: np.random.Generator):
    """Simulated Binary Crossover (SBX), producing two children from two parents."""
    n = len(x1)
    y1, y2 = x1.copy(), x2.copy()

    if rng.random() > prob:
        return y1, y2

    for i in range(n):
        if rng.random() > 0.5:
            continue
        if abs(x1[i] - x2[i]) < 1e-14:
            continue

        xi1, xi2 = min(x1[i], x2[i]), max(x1[i], x2[i])
        lb, ub = xl[i], xu[i]
        u = rng.random()

        def beta_q(beta):
            p = 1.0 / (eta + 1.0)
            if u <= 1.0 / beta:
                return (u * beta) ** p
            return (1.0 / (2.0 - u * beta)) ** p

        beta1 = 1.0 + (2.0 * (xi1 - lb) / (xi2 - xi1))
        beta2 = 1.0 + (2.0 * (ub - xi2) / (xi2 - xi1))
        alpha1 = 2.0 - beta1 ** (-(eta + 1.0))
        alpha2 = 2.0 - beta2 ** (-(eta + 1.0))

        bq1 = beta_q(alpha1)
        bq2 = beta_q(alpha2)

        c1 = 0.5 * (xi1 + xi2 - bq1 * (xi2 - xi1))
        c2 = 0.5 * (xi1 + xi2 + bq2 * (xi2 - xi1))

        c1 = min(max(c1, lb), ub)
        c2 = min(max(c2, lb), ub)

        if rng.random() <= 0.5:
            y1[i], y2[i] = c2, c1
        else:
            y1[i], y2[i] = c1, c2

    return y1, y2


def polynomial_mutation(x: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                         eta: float, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Polynomial mutation applied independently per gene with probability `prob`."""
    y = x.copy()
    n = len(x)
    for i in range(n):
        if rng.random() > prob:
            continue
        lb, ub = xl[i], xu[i]
        if ub - lb < 1e-14:
            continue
        u = rng.random()
        delta1 = (y[i] - lb) / (ub - lb)
        delta2 = (ub - y[i]) / (ub - lb)
        mut_pow = 1.0 / (eta + 1.0)

        if u <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
            delta_q = 1.0 - val ** mut_pow

        y[i] = y[i] + delta_q * (ub - lb)
        y[i] = min(max(y[i], lb), ub)
    return y


def binary_tournament_selection(rank: np.ndarray, cd: np.ndarray, n_select: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Binary tournament using the crowded-comparison operator, with replacement."""
    n = len(rank)
    selected = np.empty(n_select, dtype=int)
    for k in range(n_select):
        i, j = rng.integers(0, n, size=2)
        if rank[i] < rank[j] or (rank[i] == rank[j] and cd[i] > cd[j]):
            selected[k] = i
        elif rank[j] < rank[i] or (rank[j] == rank[i] and cd[j] > cd[i]):
            selected[k] = j
        else:
            selected[k] = i if rng.random() < 0.5 else j
    return selected
