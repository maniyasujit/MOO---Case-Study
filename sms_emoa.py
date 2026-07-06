"""From-scratch steady-state SMS-EMOA (Beume, Naujoks & Emmerich, 2007).

Representation / variation operators are identical to our NSGA-II (real-valued,
SBX + polynomial mutation) so that differences in the results can be attributed to
the survivor-selection strategy rather than to the variation operators.

Parent selection: binary tournament based directly on pairwise Pareto dominance
(ties broken uniformly at random) -- SMS-EMOA does not use crowding distance.

Survivor selection: one offspring is created per iteration (mu + 1). Non-dominated
sorting splits the merged population into fronts; the individual with the smallest
hypervolume contribution within the worst front is dropped so that the population
returns to size mu. The worst front is normalised by the current ideal/nadir point
before computing contributions, following the original paper's S-metric selection.
The dominance sort and the decision of *which* individual to drop are implemented
here from scratch; the O(n log n) hypervolume-contribution computation itself is
delegated to moocore (the C backend used by pymoo's own HV indicator), since
re-implementing an exact hypervolume algorithm is outside the scope of this
case study.
"""
import numpy as np
from moocore import hv_contributions

from operators import (
    fast_non_dominated_sort,
    sbx_crossover,
    polynomial_mutation,
)


class SMSEMOA:
    def __init__(self, problem, pop_size=100, sbx_eta=15, sbx_prob=0.9,
                 mut_eta=20, mut_prob=None, eps=1.0, seed=None):
        self.problem = problem
        self.n_var = problem.n_var
        self.xl = problem.xl
        self.xu = problem.xu
        self.pop_size = pop_size
        self.sbx_eta = sbx_eta
        self.sbx_prob = sbx_prob
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / self.n_var
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, X):
        out = {}
        self.problem._evaluate(X, out)
        return out["F"]

    def _dominance_tournament(self, F):
        """Binary tournament based on pairwise dominance (no crowding distance),
        as used in SMS-EMOA parent selection."""
        n = len(F)
        i, j = self.rng.integers(0, n, size=2)
        fi, fj = F[i], F[j]
        if np.all(fi <= fj) and np.any(fi < fj):
            return i
        if np.all(fj <= fi) and np.any(fj < fi):
            return j
        return i if self.rng.random() < 0.5 else j

    def _survivor_selection(self, X, F):
        n_survive = self.pop_size
        ideal = F.min(axis=0)
        nadir = F.max(axis=0) + 1e-32

        fronts = fast_non_dominated_sort(F)
        survivors = []

        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front.tolist())
                continue

            # worst front must be trimmed by dropping the least HV-contributing point(s)
            remaining = list(front)
            while len(survivors) + len(remaining) > n_survive:
                F_norm = (F[remaining] - ideal) / (nadir - ideal)
                ref_point = np.full(F.shape[1], 1.0 + self.eps)
                contrib = hv_contributions(F_norm, ref=ref_point)
                worst = int(np.argmin(contrib))
                del remaining[worst]

            survivors.extend(remaining)
            break

        survivors = np.array(survivors, dtype=int)
        return X[survivors], F[survivors]

    def run(self, n_gen=200):
        X = self.rng.uniform(self.xl, self.xu, size=(self.pop_size, self.n_var))
        F = self._evaluate(X)

        n_iterations = n_gen * self.pop_size  # 1 offspring per iteration -> match budget
        for _ in range(n_iterations):
            i1 = self._dominance_tournament(F)
            i2 = self._dominance_tournament(F)
            p1, p2 = X[i1], X[i2]

            c1, _ = sbx_crossover(p1, p2, self.xl, self.xu, self.sbx_eta, self.sbx_prob, self.rng)
            c1 = polynomial_mutation(c1, self.xl, self.xu, self.mut_eta, self.mut_prob, self.rng)
            f1 = self._evaluate(c1.reshape(1, -1))

            X = np.vstack([X, c1])
            F = np.vstack([F, f1])
            X, F = self._survivor_selection(X, F)

        return X, F
