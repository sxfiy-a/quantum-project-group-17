"""
experiments.py — QIS-NET Project 2 sensitivity experiment (group 17).

One-at-a-time (OAT) sensitivity: change ONLY the memory time T2, keep every
other setting fixed, and see how the delivered RATE changes. For each T2 in
params['experiment']['t2_grid_ms'] we run 5 seeds and report the mean rate
with a 95% t-confidence interval.

Output contract (autograded): write experiment_results.json containing
{
  "parameter": "t2_ms",
  "grid": [...],
  "seeds_per_point": int,
  "points": [
     {"t2_ms": x, "rate_mean": m, "rate_ci95": h, "n_seeds": k}, ...
  ]
}

Run:  python experiments.py      (about 2 s)
"""

# copy.deepcopy makes a fully separate copy of params (so we never edit the original)
import copy
# json reads variant_params.json and writes experiment_results.json
import json
# math.sqrt for the confidence interval
import math

# numpy for mean and standard deviation
import numpy as np

# make_policy builds our final policy; run_variant_scenario runs one simulation
from policy import make_policy
from simulator import run_variant_scenario


def mean_ci95(values):
    """Return (mean, 95% CI half-width) using the t-distribution.
    PROVIDED.

    Takes:  values - list of numbers (one rate per seed)
    Gives:  (mean, half_width): the true mean is probably in mean +/- half_width
    Does:   half_width = t * sample_std / sqrt(n), t from a small table
            (n = 5 seeds -> 4 degrees of freedom -> t = 2.776).
    """
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, 0.0
    s = float(np.std(values, ddof=1))
    # t_{0.975, df} for small df (table lookup, no scipy needed)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t = t_table.get(n - 1, 1.96)
    return m, t * s / math.sqrt(n)


def run_one_point(params, t2, seeds):
    """Run all seeds for ONE value of T2.

    Takes:  params - variant_params.json as a dict (not changed)
            t2     - memory time to test, in ms
            seeds  - list of seeds to run
    Gives:  {"t2_ms", "rate_mean", "rate_ci95", "n_seeds"}
    Does:   copies params, sets T2, runs each seed with a NEW policy object,
            then takes the mean and 95% CI of the rates.
    """
    # separate copy, so changing T2 here never touches the caller's params
    params_copy = copy.deepcopy(params)
    # the ONE thing we change in this experiment
    params_copy["physics"]["t2_ms"] = t2
    # simulation length from the JSON (30000 ms)
    sim_time = params_copy["grading"]["sim_time_ms"]
    # one rate per seed; fresh policy each run so no state leaks between runs
    rates = [run_variant_scenario(params_copy, sim_time, seed,
                                  make_policy(params_copy))["rate"]
             for seed in seeds]
    # mean and 95% half-width over the seeds
    mean, half_width = mean_ci95(rates)
    # one row of the results table
    return {"t2_ms": t2, "rate_mean": mean,
            "rate_ci95": half_width, "n_seeds": len(rates)}


def run_sensitivity(params, seeds) -> dict:
    """The OAT sweep over T2 (graded entry point).

    Takes:  params - variant_params.json as a dict
            seeds  - list of seeds (we use the first seeds_per_point)
    Gives:  the results dict (format at the top of this file)
    Does:   calls run_one_point for each T2 in the grid, saves the dict
            to experiment_results.json and returns it.

    Our result (PurifyFreshPolicy), mean +/- 95% CI:
        T2  147.7 -> 21.45 +/- 0.76      T2 1181.6 -> 20.99 +/- 1.21
        T2  295.4 -> 21.48 +/- 0.95      T2 2363.2 -> 21.11 +/- 0.42
        T2  590.8 -> 21.30 +/- 0.83
    All CIs overlap, so the rate is NOT sensitive to T2. Rate is set by link 1 and q_swap, not by memory,
    because we always swap the freshest pairs. (T2 DOES change fidelity:
    a good extra plot for the report.)
    """
    # the experiment block of the JSON
    exp = params["experiment"]
    # how many seeds per grid point (5)
    n_seeds = exp["seeds_per_point"]
    # one result row per T2 value, in grid order
    points = [run_one_point(params, t2, seeds[:n_seeds])
              for t2 in exp["t2_grid_ms"]]
    # assemble the exact output format the grader expects
    results = {"parameter": "t2_ms",
               "grid": exp["t2_grid_ms"],
               "seeds_per_point": n_seeds,
               "points": points}
    # save to disk (graded file)
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    params = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "variant_params.json"))
    exp = params["experiment"]
    seeds = exp["seeds"]
    out = run_sensitivity(params, seeds)
    print(json.dumps(out, indent=2))
