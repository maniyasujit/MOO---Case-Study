"""Generates boxplots and summary tables from results/results.csv."""
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "results/results.csv"
FIG_DIR = "figures"

PROBLEMS = ["dtlz1", "dtlz2", "dtlz4"]
OBJ_COUNTS = [2, 3]


def boxplot_grid(df, metric, ylabel, filename, logscale=False):
    fig, axes = plt.subplots(len(OBJ_COUNTS), len(PROBLEMS), figsize=(13, 7), sharex=True)
    algorithms = sorted(df["algorithm"].unique())
    colors = {"NSGA-II": "#1f77b4", "SMS-EMOA": "#ff7f0e"}

    for row, n_obj in enumerate(OBJ_COUNTS):
        for col, problem in enumerate(PROBLEMS):
            ax = axes[row, col]
            sub = df[(df.problem == problem) & (df.n_obj == n_obj)]
            data = [sub[sub.algorithm == a][metric].values for a in algorithms]
            bp = ax.boxplot(data, labels=algorithms, patch_artist=True, widths=0.6)
            for patch, algo in zip(bp["boxes"], algorithms):
                patch.set_facecolor(colors.get(algo, "gray"))
                patch.set_alpha(0.6)
            if logscale:
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(problem.upper())
            if col == 0:
                ax.set_ylabel(f"p={n_obj}\n{ylabel}")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{ylabel} by problem, objective count and algorithm")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{filename}", dpi=150)
    plt.close(fig)


def summary_table(df):
    agg = df.groupby(["problem", "n_obj", "algorithm"]).agg(
        hv_mean=("hv", "mean"), hv_std=("hv", "std"),
        igd_mean=("igd_plus", "mean"), igd_std=("igd_plus", "std"),
        runtime_mean=("runtime_s", "mean"), runtime_std=("runtime_s", "std"),
    ).round(4)
    return agg


def main():
    df = pd.read_csv(IN_PATH)

    boxplot_grid(df, "hv", "Hypervolume", "hv_boxplots.png")
    boxplot_grid(df, "igd_plus", "IGD+", "igd_boxplots.png", logscale=True)
    boxplot_grid(df, "runtime_s", "Runtime (s)", "runtime_boxplots.png", logscale=True)

    table = summary_table(df)
    table.to_csv("results/summary_table.csv")
    print(table)


if __name__ == "__main__":
    main()
