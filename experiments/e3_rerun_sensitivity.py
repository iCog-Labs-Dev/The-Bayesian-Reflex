"""E3 sensitivity reruns on fresh seeds (the "additional E3 reruns" referred to in the PR description).

Why: in the primary E3 run (seeds 0..9) two statistics at sigma_y = 0.1 (mean of z and lag-1 autocorrelation)
sat close to their 3-SE limits. To see whether that is chance, the same checks are rerun, unchanged, on two
FRESH sets of 10 seeds that the primary run never used:

    rerun A: seeds 10..19      rerun B: seeds 20..29

for BOTH sigma_y = 0.1 and sigma_y = 1.0. Each rerun uses the same sample sizes (1000 trials x 10 seeds x T=50)
and the same pre-fixed criteria as E3: |mean z| < 3/sqrt(M), |var z - 1| < 3*sqrt(2/M), |lag-1| < 3/sqrt(M).
The primary seeds 0..9 are included as a reference row. Nothing is tuned: this is a report, not a new test.

Writes results/e3_rerun_sensitivity.json and prints a summary table. Exit status 0 only if every statistic
in every rerun is inside its limit, otherwise 1.
Run: python experiments/e3_rerun_sensitivity.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e3_prediction_error import C, OUT, gaussian_summary  # noqa: E402

SETS = {"primary (seeds 0-9)": list(range(0, 10)),
        "rerun A (seeds 10-19)": list(range(10, 20)),
        "rerun B (seeds 20-29)": list(range(20, 30))}


def main():
    results = []
    print(f"{'sigma_y':>7s}  {'seed set':22s}{'mean z':>9s}{'(limit)':>9s}{'var z':>9s}{'(limit)':>9s}"
          f"{'lag-1':>9s}{'(limit)':>9s}  status")
    for sy in C["sys"]:
        for name, seeds in SETS.items():
            s = gaussian_summary(sy, seeds)
            s["set"] = name
            s["all_in_limit"] = bool(s["mean_z_in_limit"] and s["var_z_in_limit"] and s["lag1_in_limit"])
            results.append(s)
            print(f"{sy:7.1f}  {name:22s}{s['mean_z']:9.4f}{s['limit_mean']:9.4f}{s['var_z']:9.4f}"
                  f"{s['limit_var']:9.4f}{s['lag1']:9.4f}{s['limit_lag1']:9.4f}  "
                  f"{'OK' if s['all_in_limit'] else 'OUT OF LIMIT'}")
    reruns = [r for r in results if r["set"] != "primary (seeds 0-9)"]
    ok = all(r["all_in_limit"] for r in reruns)
    print("\nE3 sensitivity reruns (fresh seeds, both sigma_y):", "PASS" if ok else "FAIL")
    (OUT / "e3_rerun_sensitivity.json").write_text(json.dumps(dict(
        purpose="rerun the E3 mean/variance/lag-1 checks on fresh seed sets to see whether near-limit values at "
                "sigma_y=0.1 in the primary run were chance",
        criteria="|mean z|<3/sqrt(M); |var z-1|<3*sqrt(2/M); |lag-1|<3/sqrt(M), M = trials*seeds*T",
        sets=SETS, overall_pass=bool(ok), results=results), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
