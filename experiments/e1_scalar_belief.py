"""E1 - Scalar Bayesian belief maintenance + simulation-based calibration (SBC).

Primary : correctly specified model -> ranks uniform, coverage nominal, RMSE == mean posterior sd.
Control : deliberately wrong noise (sy assumed 0.25, true 0.5) -> tests MUST fail (test has power).
Secondary: prior mismatch (theta fixed = 2.0) -> how fast data overrides a bad prior. Descriptive only
           (not pass/fail, not part of the ten-seed rule): a single run of 500 trials with seed 123.
Exit    : status 0 only if every required check passes (including the closed-form check), otherwise 1.
Run: python experiments/e1_scalar_belief.py
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models import ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
C = dict(mu0=0.0, s0=1.0, sy=0.5, N=1000, T=50, L=99, seeds=list(range(10)),
         cps=[1, 5, 10, 25, 50], levels=[0.5, 0.8, 0.95], bins=10, sy_wrong=0.25)


def run_sbc(seed, sy_assumed):
    rng = np.random.default_rng(seed)
    N, T, L = C["N"], C["T"], C["L"]
    theta = C["mu0"] + C["s0"] * rng.standard_normal(N)             # theta_i ~ prior
    y = theta[:, None] + C["sy"] * rng.standard_normal((N, T))      # D_i ~ p(D|theta_i)
    b = ScalarGaussianBelief(C["mu0"], C["s0"], sy_assumed, N)
    mu, sd, ranks = np.empty((N, T)), np.empty((N, T)), {}
    for t in range(T):                                              # strictly sequential
        b.update(y[:, t])
        mu[:, t], sd[:, t] = b.mu, b.sd
        if t + 1 in C["cps"]:
            ranks[t + 1] = (b.sample(L, rng) < theta[:, None]).sum(1)  # rank in 0..L
    return theta, mu, sd, ranks


def pooled(sy_assumed):
    runs = [run_sbc(s, sy_assumed) for s in C["seeds"]]
    theta = np.concatenate([r[0] for r in runs])
    mu = np.concatenate([r[1] for r in runs])
    sd = np.concatenate([r[2] for r in runs])
    ranks = {t: np.concatenate([r[3][t] for r in runs]) for t in C["cps"]}
    return theta, mu, sd, ranks


def metrics(theta, mu, sd, ranks):
    err = mu - theta[:, None]
    m = dict(rmse=np.sqrt((err ** 2).mean(0)), rms_sd=np.sqrt((sd ** 2).mean(0)))
    m["cov"] = {p: (np.abs(err) <= stats.norm.ppf((1 + p) / 2) * sd).mean(0) for p in C["levels"]}
    m["ks_p"], m["chi_p"], m["hist"] = {}, {}, {}
    for t in C["cps"]:
        u = stats.norm.cdf(-err[:, t - 1] / sd[:, t - 1])           # PIT of true theta
        m["ks_p"][t] = stats.kstest(u, "uniform").pvalue
        h = np.bincount(ranks[t] * C["bins"] // (C["L"] + 1), minlength=C["bins"])
        m["hist"][t] = h
        m["chi_p"][t] = stats.chisquare(h).pvalue
    return m


def prior_mismatch(theta=2.0, n=500, seed=123):
    rng = np.random.default_rng(seed)
    y = theta + C["sy"] * rng.standard_normal((n, C["T"]))
    out = {}
    for mu0 in (0.0, 10.0):
        b = ScalarGaussianBelief(mu0, C["s0"], C["sy"], n)
        mean_mu, cov95 = [], []
        for t in range(C["T"]):
            b.update(y[:, t])
            mean_mu.append(b.mu.mean())
            cov95.append((np.abs(b.mu - theta) <= 1.96 * b.sd).mean())
        mean_mu = np.array(mean_mu)
        hit = np.where(np.abs(mean_mu - theta) < 0.1)[0]
        out[mu0] = dict(mean_mu=mean_mu, cov95=np.array(cov95),
                        t_bias_lt_0p1=int(hit[0] + 1) if len(hit) else None)
    return out


def main():
    ok = pooled(C["sy"])
    wrong = pooled(C["sy_wrong"])
    M, W = metrics(*ok), metrics(*wrong)
    # analytic sanity: posterior sd must equal closed form (a required check, reported like the others)
    t_ax = np.arange(1, C["T"] + 1)
    sd_exact = 1 / np.sqrt(1 / C["s0"] ** 2 + t_ax / C["sy"] ** 2)
    sd_err = float(np.abs(ok[2] - sd_exact[None, :]).max())          # every trial, every t

    N_tot = C["N"] * len(C["seeds"])
    rows = []                                                       # (name, value, criterion, pass)
    rows.append(("closed-form posterior sd max err", sd_err, "<1e-12", sd_err < 1e-12))
    for t in C["cps"]:
        for p in C["levels"]:
            v, tol = M["cov"][p][t - 1], 3 * np.sqrt(p * (1 - p) / N_tot)
            rows.append((f"coverage{int(p*100)} t={t}", v, f"|v-{p}|<={tol:.4f}", abs(v - p) <= tol))
        rows.append((f"KS p t={t}", M["ks_p"][t], ">0.001", M["ks_p"][t] > 1e-3))
        rows.append((f"chi2 rank p t={t}", M["chi_p"][t], ">0.001", M["chi_p"][t] > 1e-3))
        r = M["rmse"][t - 1] / M["rms_sd"][t - 1]
        rows.append((f"RMSE/sd t={t}", r, "within 3% of 1", abs(r - 1) <= 0.03))
    v = W["cov"][0.95][-1]
    rows.append(("CONTROL cov95 t=50 (wrong sy)", v, "<0.90 (must fail)", v < 0.90))
    rows.append(("CONTROL KS p t=50 (wrong sy)", W["ks_p"][50], "<0.001 (must fail)", W["ks_p"][50] < 1e-3))

    print(f"{'check':34s}{'value':>10s}  criterion")
    for n, v, c, p in rows:
        print(f"{n:34s}{v:10.4f}  {c:24s}{'PASS' if p else 'FAIL'}")
    ok = all(r[3] for r in rows)
    print("\nE1 (scalar Gaussian inference module, foundation only):", "PASS" if ok else "FAIL")

    PM = prior_mismatch()
    for mu0, d in PM.items():
        print(f"prior mu0={mu0:>4}: first t with |mean bias|<0.1 = {d['t_bias_lt_0p1']}; "
              f"cov95 t=1/10/50 = {d['cov95'][0]:.2f}/{d['cov95'][9]:.2f}/{d['cov95'][49]:.2f}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
    exp = N_tot / C["bins"]
    band = stats.binom.ppf([0.005, 0.995], N_tot, 1 / C["bins"])
    for a, (lab, mm, t) in zip(ax, [("correct t=1", M, 1), ("correct t=10", M, 10),
                                    ("correct t=50", M, 50), ("CONTROL wrong sy, t=50", W, 50)]):
        a.bar(range(C["bins"]), mm["hist"][t], color="tab:red" if "CONTROL" in lab else "tab:blue")
        a.axhline(exp, color="k", ls="--")
        a.axhspan(*band, color="gray", alpha=0.2)
        a.set_title(lab, fontsize=9)
        a.set_xlabel("rank bin")
    ax[0].set_ylabel("count")
    fig.suptitle("E1 SBC rank histograms (dashed = uniform, band = 99% interval)")
    fig.tight_layout(); fig.savefig(OUT / "e1_sbc_ranks.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for p, col in zip(C["levels"], ["tab:green", "tab:orange", "tab:blue"]):
        ax.plot(t_ax, M["cov"][p], color=col, label=f"{int(p*100)}% interval")
        ax.axhline(p, color=col, ls=":")
        ax.plot(t_ax, W["cov"][p], color=col, ls="--", alpha=0.6)
    ax.set_xlabel("observations t"); ax.set_ylabel("empirical coverage")
    ax.set_title("E1 coverage (solid correct, dashed wrong sy)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "e1_coverage.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(t_ax, M["rmse"], label="RMSE of posterior mean")
    ax.loglog(t_ax, M["rms_sd"], "--", label="RMS posterior sd")
    ax.loglog(t_ax, sd_exact, ":", label="closed-form sd")
    ax.set_xlabel("observations t"); ax.set_title("E1 accuracy and uncertainty"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "e1_accuracy.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for mu0, d in PM.items():
        ax.plot(t_ax, d["mean_mu"], label=f"prior mean {mu0}")
    ax.axhline(2.0, color="k", ls="--", label="true theta = 2")
    ax.set_xlabel("observations t"); ax.set_ylabel("mean posterior mean")
    ax.set_title("E1 secondary: prior mismatch"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "e1_prior_mismatch.png", dpi=140); plt.close(fig)

    res = dict(scope="scalar Gaussian inference module under the specified generative model; does not test the reflex loop", config={k: v for k, v in C.items() if k != "seeds"}, seeds=C["seeds"],
               checks=[dict(name=n, value=float(v), criterion=c, passed=bool(p)) for n, v, c, p in rows],
               overall_pass=bool(ok),
               prior_mismatch={str(k): dict(t_bias_lt_0p1=d["t_bias_lt_0p1"],
                                            cov95_t1=float(d["cov95"][0]), cov95_t10=float(d["cov95"][9]),
                                            cov95_t50=float(d["cov95"][49])) for k, d in PM.items()})
    (OUT / "e1_results.json").write_text(json.dumps(res, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
