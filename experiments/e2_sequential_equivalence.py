"""E2a - Sequential updating vs batch oracle (scalar Gaussian module).

Scope: validates SEQUENTIAL UPDATING of the scalar Gaussian inference module (paper Thm 2.1,
conjugate case). It does not test prediction error, function space, action or the closed loop.

Question : Does the recursive update match the batch posterior over a long horizon, and stay numerically stable?
Oracle   : batch posterior from y_{1:t} in closed form, computed in extended precision (longdouble),
           so it is independent of the float64 recursion being tested.
Seeds    : 10 independent seeds (0..9), per the ten-seed rule. Every criterion below is evaluated on the
           WORST case over all seeds and problems (max error), so one bad seed fails the check.
Primary  : T=100, 100 random problems per seed (random mu0, s0, sy, theta) -> 1000 problems in total.
           Compare mu_t, sigma_t^2, KL(seq||batch) at every t.
Criteria : max|dmu| < 1e-6, max|dvar| < 1e-6, max KL < 1e-12 (all t, all problems, all seeds).
Stability: T=100000, 20 problems per seed (200 in total): same criteria, finite values, error growth reported.
Extra    : permutation invariance (<1e-10); NEGATIVE CONTROL: dropping every 10th observation must change mu
           (>1e-3) on EVERY seed (the minimum over seeds is tested).
Exit     : status 0 only if every required check passes, otherwise 1.
Run: python experiments/e2_sequential_equivalence.py
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models import ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
LD = np.longdouble
SEEDS = list(range(10))
N_PRIMARY, T_PRIMARY = 100, 100
N_STRESS, T_STRESS = 20, 100_000


def make_problems(rng, n, T):
    mu0 = rng.uniform(-3, 3, n)
    s0 = rng.uniform(0.3, 3.0, n)
    sy = rng.uniform(0.1, 2.0, n)
    theta = mu0 + s0 * rng.standard_normal(n)
    y = theta[:, None] + sy[:, None] * rng.standard_normal((n, T))
    return mu0, s0, sy, y


def sequential(mu0, s0, sy, y):
    n, T = y.shape
    b = ScalarGaussianBelief(mu0, s0, sy, n)                        # per-problem parameters via the constructor
    mu, var = np.empty((n, T)), np.empty((n, T))
    for t in range(T):
        b.update(y[:, t])
        mu[:, t], var[:, t] = b.mu, b.var
    return mu, var


def batch_oracle(mu0, s0, sy, y):
    """Posterior from y_{1:t}: precision = 1/s0^2 + t/sy^2; mean = (mu0/s0^2 + sum y / sy^2)/precision."""
    mu0, s0, sy, y = (a.astype(LD) for a in (mu0, s0, sy, y))
    t = np.arange(1, y.shape[1] + 1, dtype=LD)[None, :]
    prec = 1 / s0[:, None] ** 2 + t / sy[:, None] ** 2
    mean = (mu0[:, None] / s0[:, None] ** 2 + np.cumsum(y, axis=1) / sy[:, None] ** 2) / prec
    return mean, 1 / prec


def kl_gauss(mu_s, var_s, mu_b, var_b):
    """KL(N(mu_s,var_s) || N(mu_b,var_b)); log1p form avoids cancellation when the two are nearly equal."""
    d = (var_s - var_b) / var_b
    return 0.5 * (d - np.log1p(d)) + (mu_s - mu_b) ** 2 / (2 * var_b)


def compare(seq, orc):
    (mu_s, var_s), (mu_b, var_b) = seq, orc
    dmu = np.abs(mu_s.astype(LD) - mu_b)
    dvar = np.abs(var_s.astype(LD) - var_b)
    kl = kl_gauss(mu_s.astype(LD), var_s.astype(LD), mu_b, var_b)
    return dmu.astype(float), dvar.astype(float), (dvar / var_b).astype(float), kl.astype(float)


def run_seed(seed):
    """All E2a measurements for one seed. Returns per-step worst-case curves and scalar summaries."""
    rng = np.random.default_rng(seed)
    r = {}

    # ---- primary: T = 100 ----
    mu0, s0, sy, y = make_problems(rng, N_PRIMARY, T_PRIMARY)
    seq, orc = sequential(mu0, s0, sy, y), batch_oracle(mu0, s0, sy, y)
    dmu, dvar, rvar, kl = compare(seq, orc)
    r["p_dmu_t"], r["p_dvar_t"], r["p_kl_t"] = dmu.max(0), dvar.max(0), kl.max(0)
    r["p_rvar"] = float(rvar.max())

    # ---- permutation invariance ----
    perm = rng.permutation(T_PRIMARY)
    mu_p, _ = sequential(mu0, s0, sy, y[:, perm])
    r["perm"] = float(np.abs(mu_p[:, -1] - seq[0][:, -1]).max())

    # ---- negative control: drop every 10th observation ----
    keep = np.array([i for i in range(T_PRIMARY) if (i + 1) % 10 != 0])
    mu_d, _ = sequential(mu0, s0, sy, y[:, keep])
    r["control"] = float(np.abs(mu_d[:, -1].astype(LD) - orc[0][:, -1]).astype(float).mean())

    # ---- stability stress: T = 100000 ----
    mu0s, s0s, sys_, ys = make_problems(rng, N_STRESS, T_STRESS)
    seq_s, orc_s = sequential(mu0s, s0s, sys_, ys), batch_oracle(mu0s, s0s, sys_, ys)
    dmu_s, dvar_s, rvar_s, kl_s = compare(seq_s, orc_s)
    r["finite"] = bool(np.isfinite(seq_s[0]).all() and np.isfinite(seq_s[1]).all())
    r["s_dmu_t"], r["s_dvar_t"], r["s_kl_t"] = dmu_s.max(0), dvar_s.max(0), kl_s.max(0)
    r["s_rvar"] = float(rvar_s.max())
    return r


def main():
    runs = {s: run_seed(s) for s in SEEDS}
    worst = lambda key: np.max([runs[s][key] for s in SEEDS], axis=0)   # elementwise max over seeds
    worst_scalar = lambda key: float(np.max([np.max(runs[s][key]) for s in SEEDS]))

    p_dmu, p_dvar, p_kl = (worst(k) for k in ("p_dmu_t", "p_dvar_t", "p_kl_t"))
    s_dmu, s_dvar, s_kl = (worst(k) for k in ("s_dmu_t", "s_dvar_t", "s_kl_t"))
    finite = all(runs[s]["finite"] for s in SEEDS)
    perm_w = worst_scalar("perm")
    control_min = float(min(runs[s]["control"] for s in SEEDS))        # control must differ on EVERY seed

    rows = [("T=100 max|dmu|", p_dmu.max(), "<1e-6", p_dmu.max() < 1e-6),
            ("T=100 max|dvar|", p_dvar.max(), "<1e-6", p_dvar.max() < 1e-6),
            ("T=100 max KL", p_kl.max(), "<1e-12", p_kl.max() < 1e-12),
            ("permutation |dmu| (final)", perm_w, "<1e-10", perm_w < 1e-10),
            ("CONTROL mean|dmu| (dropped data)", control_min, ">1e-3 (must differ)", control_min > 1e-3),
            ("T=1e5 finite values", float(finite), "True", finite),
            ("T=1e5 max|dmu|", s_dmu.max(), "<1e-6", s_dmu.max() < 1e-6),
            ("T=1e5 max rel dvar", worst_scalar("s_rvar"), "<1e-6", worst_scalar("s_rvar") < 1e-6),
            ("T=1e5 max KL", s_kl.max(), "<1e-12", s_kl.max() < 1e-12)]

    t_ax = np.arange(1, T_STRESS + 1)
    sel = t_ax >= 10
    diag = dict(max_rel_dvar_T100=worst_scalar("p_rvar"),
                dmu_growth_loglog_slope_T1e5=float(np.polyfit(
                    np.log(t_ax[sel]), np.log(np.maximum(s_dmu[sel], 1e-300)), 1)[0]),
                per_seed_max_dmu_T100={str(s): float(runs[s]["p_dmu_t"].max()) for s in SEEDS},
                per_seed_max_dmu_T1e5={str(s): float(runs[s]["s_dmu_t"].max()) for s in SEEDS},
                per_seed_control_mean_dmu={str(s): runs[s]["control"] for s in SEEDS})

    print(f"seeds {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)} seeds): {len(SEEDS)*N_PRIMARY} problems at T={T_PRIMARY}, "
          f"{len(SEEDS)*N_STRESS} problems at T={T_STRESS}; criteria evaluated on the worst case over seeds")
    print(f"{'check':36s}{'value':>12s}  criterion")
    for n, v, c, p in rows:
        print(f"{n:36s}{v:12.3e}  {c:22s}{'PASS' if p else 'FAIL'}")
    print("diagnostics:", {k: v for k, v in diag.items() if not k.startswith("per_seed")})
    ok = all(r[3] for r in rows)
    print("\nE2a (scalar Gaussian sequential updating vs batch oracle):", "PASS" if ok else "FAIL")

    t100 = np.arange(1, T_PRIMARY + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(t100, np.maximum(p_dmu, 1e-20), label="max |dmu|")
    ax[0].semilogy(t100, np.maximum(p_dvar, 1e-20), label="max |dvar|")
    ax[0].semilogy(t100, np.maximum(p_kl, 1e-20), label="max KL")
    ax[0].axhline(1e-6, color="k", ls=":", label="1e-6 criterion")
    ax[0].set_title(f"T=100 ({len(SEEDS)*N_PRIMARY} problems, worst over {len(SEEDS)} seeds)")
    ax[1].loglog(t_ax, np.maximum(s_dmu, 1e-20), label="max |dmu|")
    ax[1].loglog(t_ax, np.maximum(s_dvar, 1e-20), label="max |dvar|")
    ax[1].loglog(t_ax, np.maximum(s_kl, 1e-20), label="max KL")
    ax[1].axhline(1e-6, color="k", ls=":")
    ax[1].set_title(f"stability stress T=1e5 ({len(SEEDS)*N_STRESS} problems, {len(SEEDS)} seeds)")
    for a in ax:
        a.set_xlabel("observations t"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e2_sequential_vs_batch.png", dpi=140); plt.close(fig)

    (OUT / "e2_results.json").write_text(json.dumps(dict(
        scope="scalar Gaussian sequential updating vs batch oracle (longdouble)", overall_pass=bool(ok),
        seeds=SEEDS, n_problems_T100=len(SEEDS) * N_PRIMARY, n_problems_T1e5=len(SEEDS) * N_STRESS,
        aggregation="worst case (max error) over seeds and problems; control uses the minimum over seeds",
        checks=[dict(name=n, value=float(v), criterion=c, passed=bool(p)) for n, v, c, p in rows],
        diagnostics=diag), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
