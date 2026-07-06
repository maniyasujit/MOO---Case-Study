"""Supplementary figure: final Pareto-front approximation vs. the true front,
for one representative run per algorithm, for the p=2 case of each problem.
Purely illustrative -- complements the quantitative boxplots in analyze.py.
"""
import matplotlib.pyplot as plt
from pymoo.problems import get_problem

from nsga2 import NSGA2
from sms_emoa import SMSEMOA

PROBLEMS = ["dtlz1", "dtlz2", "dtlz4"]
N_VAR = 10
POP_SIZE = 100
N_GEN = 200
SEED = 1


def main():
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(13, 4.2))

    for col, problem_name in enumerate(PROBLEMS):
        ax = axes[col]
        problem = get_problem(problem_name, n_var=N_VAR, n_obj=2)
        pf = problem.pareto_front()

        nsga2 = NSGA2(problem, pop_size=POP_SIZE, seed=SEED)
        _, F_nsga2 = nsga2.run(n_gen=N_GEN)

        sms = SMSEMOA(problem, pop_size=POP_SIZE, seed=SEED)
        _, F_sms = sms.run(n_gen=N_GEN)

        ax.plot(pf[:, 0], pf[:, 1], c="black", lw=1.5, label="True Pareto front", zorder=1)
        ax.scatter(F_nsga2[:, 0], F_nsga2[:, 1], s=14, c="#1f77b4", alpha=0.7, label="NSGA-II", zorder=2)
        ax.scatter(F_sms[:, 0], F_sms[:, 1], s=14, c="#ff7f0e", alpha=0.7, label="SMS-EMOA", zorder=2)
        ax.set_title(problem_name.upper())
        ax.set_xlabel("$f_1$")
        if problem_name == "dtlz1":
            # zoom in on the region around the true front; DTLZ1 populations often
            # retain a few individuals stuck far away in a local optimum, which would
            # otherwise dominate the axis scale and hide the converged cluster.
            pad = pf.max() * 0.4
            ax.set_xlim(-pad, pf.max() + pad)
            ax.set_ylim(-pad, pf.max() + pad)
        if col == 0:
            ax.set_ylabel("$f_2$")
            ax.legend(fontsize=8)

    fig.suptitle("Final population vs. true Pareto front (p=2, single representative run)")
    fig.tight_layout()
    fig.savefig("figures/front_comparison_p2.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
