"""From-scratch real-valued NSGA-II (Deb et al., 2002).

Representation: real-valued vectors bounded by the problem's box constraints.
Variation: SBX crossover + polynomial mutation (as used in the original NSGA-II paper).
Parent selection: binary tournament on the crowded-comparison operator.
Survival selection: non-dominated sorting + crowding distance.
"""
import numpy as np

from operators import (
    fast_non_dominated_sort,
    crowding_distance,
    sbx_crossover,
    polynomial_mutation,
    binary_tournament_selection,
)


class NSGA2:
    def __init__(self, problem, pop_size=100, sbx_eta=15, sbx_prob=0.9,
                 mut_eta=20, mut_prob=None, seed=None):
        self.problem = problem
        self.n_var = problem.n_var
        self.xl = problem.xl
        self.xu = problem.xu
        self.pop_size = pop_size
        self.sbx_eta = sbx_eta
        self.sbx_prob = sbx_prob
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / self.n_var
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, X):
        out = {}
        self.problem._evaluate(X, out)
        return out["F"]

    def _rank_and_crowd(self, F):
        n = len(F)
        rank = np.empty(n, dtype=int)
        cd = np.empty(n)
        fronts = fast_non_dominated_sort(F)
        for r, front in enumerate(fronts):
            rank[front] = r
            cd[front] = crowding_distance(F[front])
        return rank, cd, fronts

    def _make_offspring(self, X, rank, cd):
        parents_idx = binary_tournament_selection(rank, cd, self.pop_size, self.rng)
        children = []
        for k in range(0, self.pop_size, 2):
            p1 = X[parents_idx[k]]
            p2 = X[parents_idx[(k + 1) % self.pop_size]]
            c1, c2 = sbx_crossover(p1, p2, self.xl, self.xu, self.sbx_eta, self.sbx_prob, self.rng)
            c1 = polynomial_mutation(c1, self.xl, self.xu, self.mut_eta, self.mut_prob, self.rng)
            c2 = polynomial_mutation(c2, self.xl, self.xu, self.mut_eta, self.mut_prob, self.rng)
            children.append(c1)
            children.append(c2)
        return np.array(children[:self.pop_size])

    def run(self, n_gen=200):
        X = self.rng.uniform(self.xl, self.xu, size=(self.pop_size, self.n_var))
        F = self._evaluate(X)

        for _ in range(n_gen):
            rank, cd, _ = self._rank_and_crowd(F)
            Xc = self._make_offspring(X, rank, cd)
            Fc = self._evaluate(Xc)

            R_X = np.vstack([X, Xc])
            R_F = np.vstack([F, Fc])
            fronts = fast_non_dominated_sort(R_F)

            new_X, new_F = [], []
            for front in fronts:
                if len(new_X) + len(front) <= self.pop_size:
                    new_X.extend(R_X[front])
                    new_F.extend(R_F[front])
                else:
                    remaining = self.pop_size - len(new_X)
                    front_cd = crowding_distance(R_F[front])
                    order = np.argsort(-front_cd)[:remaining]
                    chosen = front[order]
                    new_X.extend(R_X[chosen])
                    new_F.extend(R_F[chosen])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X, F
