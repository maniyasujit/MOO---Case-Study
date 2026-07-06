"""Experiment runner for the MOO case study.

Benchmarks our from-scratch NSGA-II and SMS-EMOA on DTLZ1, DTLZ2 and DTLZ4
(pymoo problem implementations, n_var=10) for n_obj in {2, 3}, over multiple
independent runs. Records the Hypervolume and IGD+ indicator values of the
final population plus wall-clock runtime, and stores everything as a single
CSV file for later analysis/plotting.
"""
import argparse
import time

import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from nsga2 import NSGA2
from sms_emoa import SMSEMOA

PROBLEMS = ["dtlz1", "dtlz2", "dtlz4"]
N_VAR = 10
OBJ_COUNTS = [2, 3]
ALGORITHMS = {"NSGA-II": NSGA2, "SMS-EMOA": SMSEMOA}


def get_reference_set(problem_name, n_obj):
    problem = get_problem(problem_name, n_var=N_VAR, n_obj=n_obj)
    if n_obj == 2:
        pf = problem.pareto_front()
    else:
        pf = problem.pareto_front(ref_dirs=_ref_dirs(n_obj))
    return problem, pf


def _ref_dirs(n_obj, n_partitions=12):
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def run_once(problem, algo_name, pop_size, n_gen, seed):
    algo_cls = ALGORITHMS[algo_name]
    algo = algo_cls(problem, pop_size=pop_size, seed=seed)
    t0 = time.time()
    X, F = algo.run(n_gen=n_gen)
    runtime = time.time() - t0
    return F, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=15)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--out", type=str, default="results/results.csv")
    args = parser.parse_args()

    rows = []

    for problem_name in PROBLEMS:
        for n_obj in OBJ_COUNTS:
            problem, pf = get_reference_set(problem_name, n_obj)
            nadir = pf.max(axis=0)
            ref_point = 1.1 * nadir
            hv_indicator = HV(ref_point=ref_point)
            igd_indicator = IGDPlus(pf)

            for algo_name in ALGORITHMS:
                for run in range(args.n_runs):
                    seed = 1000 * run + 1
                    F, runtime = run_once(problem, algo_name, args.pop_size, args.n_gen, seed)

                    hv_val = hv_indicator(F)
                    igd_val = igd_indicator(F)

                    rows.append({
                        "problem": problem_name,
                        "n_obj": n_obj,
                        "algorithm": algo_name,
                        "run": run,
                        "seed": seed,
                        "hv": hv_val,
                        "igd_plus": igd_val,
                        "runtime_s": runtime,
                        "n_var": N_VAR,
                        "pop_size": args.pop_size,
                        "n_gen": args.n_gen,
                    })
                    print(f"{problem_name} p={n_obj} {algo_name} run={run} "
                          f"HV={hv_val:.4f} IGD+={igd_val:.4f} t={runtime:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
