"""E3 - One-step-ahead prediction error, standardized by the Bayesian predictive (scalar Gaussian model).

Scope: validates that one-step-ahead prediction errors are correctly standardized by the Bayesian predictive
uncertainty under the scalar Gaussian model. It does NOT establish that a predictive-coding layer itself
performs or drives Bayesian learning (that interface decision is still open: E7). In this PR prediction
error is a diagnostic; the belief update never uses it.

What is exercised, and by which code
  Part A (shared framework): ScalarGaussianBelief + PredictiveCodingLayer. Covers standardized_error(),
        prediction_error() and precision_weighted_error().
  Part B (standalone, NOT the shared framework): the Normal-Inverse-Gamma / Student-t extension below is
        self-contained NumPy/SciPy code (nig_run). It does not call ScalarGaussianBelief, PredictiveCodingLayer
        or GenerativeModel, so it validates the closed-form NIG predictive derivation and its PIT, not the
        implementation classes. Its rows are tagged "[standalone]".

Three quantities (e = y - yhat, v = sigma_{t-1}^2 + sigma_y^2):
  prediction_error          e              raw error
  precision_weighted_error  e / v          variance 1/v when calibrated, i.e. NOT unit variance
  standardized_error        e / sqrt(v)    z_t, N(0,1) and independent over t when calibrated

z_t = (y_t - yhat_t) / sqrt(sigma_{t-1}^2 + sigma_y^2), yhat_t and sigma_{t-1}^2 from the belief BEFORE seeing y_t.
Setup  : theta ~ prior N(0,1); 1000 trials x 10 seeds x T=50; sigma_y in {0.1, 1.0}.
Checks (correct model): mean z ~ 0, var z ~ 1 (3 SE), KS vs N(0,1) at t=1,5,25,50 (p>0.001), lag-1 autocorr ~ 0 (3 SE),
        Kalman identity  mu_t = mu_{t-1} + K_t e_t, K_t = s2_{t-1}/(s2_{t-1}+sy^2)  to <1e-12,
        precision weighting: raw error scale follows sigma_y (ratio > 5) while z scale does not (ratio 1 +/- 0.01),
        precision_weighted_error(): equals z/sqrt(v) to <1e-12; Var(pw)*v ~ 1 at t=1 and t=50 (3 SE);
        and it drives the update exactly: mu_t = mu_{t-1} + s2_{t-1} * pw_t to <1e-12.
Negative controls (must FAIL): wrong sigma_y (0.5x), stale predictive variance (posterior after update),
        unweighted raw error used in the update, pw error treated as if it were unit variance.
Extension (paper 5.2, standalone): unknown noise variance (Normal-Inverse-Gamma) -> z ~ Student-t(2a_{t-1});
        control: N(0,1) CDF must fail.
Exit   : status 0 only if every required check passes, otherwise 1.
Run: python experiments/e3_prediction_error.py
(Sensitivity reruns on fresh seeds: python experiments/e3_rerun_sensitivity.py)
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
from bayesian_reflex import PredictiveCodingLayer  # noqa: E402
from bayesian_reflex.models import ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
C = dict(n=1000, seeds=list(range(10)), T=50, sys=[0.1, 1.0], cps=[1, 5, 25, 50],
         wrong_factor=0.5, a0=1.5, b0=1.0, k0=1.0, m0=0.0)


def gaussian_run(sy, seed):
    rng = np.random.default_rng(seed)
    n, T = C["n"], C["T"]
    theta = rng.standard_normal(n)                                   # theta ~ prior N(0,1)
    y = theta[:, None] + sy * rng.standard_normal((n, T))
    b = ScalarGaussianBelief(0.0, 1.0, sy, n)
    bw = ScalarGaussianBelief(0.0, 1.0, C["wrong_factor"] * sy, n)   # control: wrong noise level
    pc = PredictiveCodingLayer()
    Z, E, PW, Zs, Zw = (np.empty((n, T)) for _ in range(5))
    K, VP = np.empty(T), np.empty(T)
    kal = kal_pw = pw_id = kal_unw = 0.0
    for t in range(T):
        yt = y[:, t]
        mu_p, v_pred = b.predict()                                   # BEFORE the update
        var_prev = b.var
        E[:, t] = pc.prediction_error(yt, mu_p)
        Z[:, t] = pc.standardized_error(yt, mu_p, v_pred)
        PW[:, t] = pc.precision_weighted_error(yt, mu_p, v_pred)     # layer output, tested directly below
        mw, vw = bw.predict()
        Zw[:, t] = pc.standardized_error(yt, mw, vw)
        gain = var_prev / (var_prev + sy ** 2)
        b.update(yt)
        kal = max(kal, float(np.abs(b.mu - (mu_p + gain * E[:, t])).max()))
        kal_pw = max(kal_pw, float(np.abs(b.mu - (mu_p + var_prev * PW[:, t])).max()))
        kal_unw = max(kal_unw, float(np.abs(b.mu - (mu_p + var_prev * E[:, t])).max()))   # control
        pw_id = max(pw_id, float(np.abs(PW[:, t] * np.sqrt(v_pred) - Z[:, t]).max()))
        Zs[:, t] = pc.standardized_error(yt, mu_p, b.var + sy ** 2)  # control: stale (post-update) variance
        bw.update(yt)
        K[t], VP[t] = gain[0], v_pred[0]
    return dict(Z=Z, E=E, PW=PW, Zs=Zs, Zw=Zw, kal=kal, kal_pw=kal_pw, kal_unw=kal_unw, pw_id=pw_id, K=K, VP=VP)


def gaussian_summary(sy, seeds):
    """Mean z, var z, lag-1 autocorrelation for the correct model, with their 3-SE limits (used by the reruns)."""
    runs = [gaussian_run(sy, s) for s in seeds]
    Z = np.concatenate([r["Z"] for r in runs])
    M = Z.size
    r1 = float(np.corrcoef(Z[:, :-1].ravel(), Z[:, 1:].ravel())[0, 1])
    lim_mean, lim_var = 3 / np.sqrt(M), 3 * np.sqrt(2 / M)
    return dict(sy=sy, seeds=list(seeds), n_z=int(M), mean_z=float(Z.mean()), var_z=float(Z.var()), lag1=r1,
                limit_mean=float(lim_mean), limit_var=float(lim_var), limit_lag1=float(lim_mean),
                mean_z_in_limit=bool(abs(Z.mean()) < lim_mean), var_z_in_limit=bool(abs(Z.var() - 1) < lim_var),
                lag1_in_limit=bool(abs(r1) < lim_mean))


def nig_run(seed):
    """[standalone] Unknown variance: y~N(theta,s2), theta|s2~N(m,s2/k), s2~IG(a,b). Predictive = Student-t(2a, m, b(k+1)/(a k)).
    Self-contained: does not use the shared classes (ScalarGaussianBelief / PredictiveCodingLayer)."""
    rng = np.random.default_rng(1000 + seed)
    n, T, a, b, k, m = C["n"], C["T"], C["a0"], np.full(C["n"], C["b0"]), C["k0"], np.full(C["n"], C["m0"])
    s2 = C["b0"] / rng.gamma(C["a0"], 1.0, n)
    theta = C["m0"] + np.sqrt(s2 / C["k0"]) * rng.standard_normal(n)
    y = theta[:, None] + np.sqrt(s2)[:, None] * rng.standard_normal((n, T))
    Z, DF = np.empty((n, T)), np.empty(T)
    for t in range(T):
        scale = np.sqrt(b * (k + 1) / (a * k))
        Z[:, t], DF[t] = (y[:, t] - m) / scale, 2 * a
        k_new = k + 1
        b = b + k * (y[:, t] - m) ** 2 / (2 * k_new)
        m = (k * m + y[:, t]) / k_new
        k, a = k_new, a + 0.5
    return Z, DF


def main():
    rows, store = [], {}
    for sy in C["sys"]:
        runs = [gaussian_run(sy, s) for s in C["seeds"]]
        cat = lambda key: np.concatenate([r[key] for r in runs])
        Z, E, PW, Zs, Zw = (cat(k) for k in ("Z", "E", "PW", "Zs", "Zw"))
        kal, kal_pw = max(r["kal"] for r in runs), max(r["kal_pw"] for r in runs)
        kal_unw, pw_id = max(r["kal_unw"] for r in runs), max(r["pw_id"] for r in runs)
        K, VP = runs[0]["K"], runs[0]["VP"]
        store[sy] = dict(Z=Z, E=E, Zs=Zs, Zw=Zw, K=K)
        M = Z.size
        r1 = np.corrcoef(Z[:, :-1].ravel(), Z[:, 1:].ravel())[0, 1]
        tag = f"sy={sy}"
        rows += [(f"{tag} mean z", Z.mean(), f"|v|<{3/np.sqrt(M):.4f}", abs(Z.mean()) < 3 / np.sqrt(M)),
                 (f"{tag} var z", Z.var(), f"|v-1|<{3*np.sqrt(2/M):.4f}", abs(Z.var() - 1) < 3 * np.sqrt(2 / M)),
                 (f"{tag} lag-1 autocorr", r1, f"|v|<{3/np.sqrt(M):.4f}", abs(r1) < 3 / np.sqrt(M))]
        for t in C["cps"]:
            p = stats.kstest(Z[:, t - 1], "norm").pvalue
            rows.append((f"{tag} KS p t={t}", p, ">0.001", p > 1e-3))
        rows.append((f"{tag} Kalman identity max err", kal, "<1e-12", kal < 1e-12))
        vw = Zw.var()
        rows.append((f"CONTROL {tag} wrong sy: var z", vw, "|v-1|>0.1 (must fail)", abs(vw - 1) > 0.1))
        vs = Zs[:, :5].var()
        rows.append((f"CONTROL {tag} stale var: var z t<=5", vs, "|v-1|>0.05 (must fail)", abs(vs - 1) > 0.05))
        ps = stats.kstest(Zs[:, 0], "norm").pvalue
        rows.append((f"CONTROL {tag} stale var: KS p t=1", ps, "<0.001 (must fail)", ps < 1e-3))

        # ---- precision_weighted_error(): tested directly, and kept distinct from z ----
        N = PW.shape[0]
        tol_var = 3 * np.sqrt(2 / N)
        rows.append((f"{tag} pw*sqrt(v) == z max err", pw_id, "<1e-12", pw_id < 1e-12))
        for t in (1, 50):
            vv = PW[:, t - 1].var() * VP[t - 1]                      # Var(pw_t) * v_t, should be 1
            rows.append((f"{tag} Var(pw)*v_pred t={t}", vv, f"|v-1|<{tol_var:.4f}", abs(vv - 1) < tol_var))
        rows.append((f"{tag} update = mu + s2_prev*pw max err", kal_pw, "<1e-12", kal_pw < 1e-12))
        rows.append((f"CONTROL {tag} update with raw error", kal_unw, ">1e-3 (must fail)", kal_unw > 1e-3))
        dev = max(abs(PW[:, t - 1].var() - 1) for t in (1, 50))
        rows.append((f"CONTROL {tag} pw is not unit variance", dev, "max_t|Var(pw)-1|>0.1 (must fail)", dev > 0.1))

    lo, hi = store[0.1], store[1.0]
    raw_ratio = hi["E"][:, -1].std() / lo["E"][:, -1].std()
    z_ratio = hi["Z"].std() / lo["Z"].std()
    rows += [("precision wt: raw err std ratio (1.0/0.1)", raw_ratio, ">5 (raw follows sy)", raw_ratio > 5),
             ("precision wt: z std ratio (1.0/0.1)", z_ratio, "|v-1|<0.01", abs(z_ratio - 1) < 0.01)]

    # ---- extension: unknown variance, Student-t  [standalone: not the shared framework] ----
    nr = [nig_run(s) for s in C["seeds"]]
    Zn, DF = np.concatenate([r[0] for r in nr]), nr[0][1]
    for t in C["cps"]:
        p = stats.kstest(stats.t.cdf(Zn[:, t - 1], DF[t - 1]), "uniform").pvalue
        rows.append((f"[standalone] NIG Student-t PIT KS p t={t}", p, ">0.001", p > 1e-3))
    pc_ = stats.kstest(stats.norm.cdf(Zn[:, 0]), "uniform").pvalue
    rows.append(("CONTROL [standalone] NIG normal CDF KS p t=1", pc_, "<0.001 (must fail)", pc_ < 1e-3))

    print(f"{'check':50s}{'value':>11s}  criterion")
    for n_, v, c, p in rows:
        print(f"{n_:50s}{v:11.3e}  {c:34s}{'PASS' if p else 'FAIL'}")
    ok = all(r[3] for r in rows)
    print("\nE3 (scalar Gaussian one-step-ahead standardized prediction error):", "PASS" if ok else "FAIL")

    # ---- figure ----
    q = np.linspace(0.001, 0.999, 200)
    th = stats.norm.ppf(q)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, sy in zip(ax[0, :2], C["sys"]):
        s = store[sy]
        a.plot(th, np.quantile(s["Z"], q), "b-", label="correct")
        a.plot(th, np.quantile(s["Zw"], q), "r--", label="wrong sy (0.5x)")
        a.plot(th, np.quantile(s["Zs"][:, :5], q), color="orange", ls="-.", label="stale var (t<=5)")
        a.plot(th, th, "k:"); a.set_title(f"QQ of z vs N(0,1), sy={sy}")
        a.set_xlabel("N(0,1) quantile"); a.legend(fontsize=7)
    t_ax = np.arange(1, C["T"] + 1)
    for sy, col in zip(C["sys"], ["tab:green", "tab:purple"]):
        ax[0, 2].plot(t_ax, store[sy]["E"].std(0), color=col, ls="--", label=f"raw error std, sy={sy}")
        ax[0, 2].plot(t_ax, store[sy]["Z"].std(0), color=col, label=f"z std, sy={sy}")
        ax[1, 0].plot(t_ax, store[sy]["K"], color=col, label=f"gain K_t, sy={sy}")
    ax[0, 2].set_title("precision weighting: raw error vs z"); ax[0, 2].set_yscale("log"); ax[0, 2].legend(fontsize=7)
    ax[1, 0].set_title("Kalman gain K_t = s2_{t-1}/(s2_{t-1}+sy^2)"); ax[1, 0].set_xlabel("t"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].hist(stats.t.cdf(Zn[:, 0], DF[0]), bins=20, density=True, alpha=0.8)
    ax[1, 1].axhline(1, color="k", ls=":"); ax[1, 1].set_title("[standalone] NIG t=1 PIT, Student-t CDF (correct)")
    ax[1, 2].hist(stats.norm.cdf(Zn[:, 0]), bins=20, density=True, color="tab:red", alpha=0.8)
    ax[1, 2].axhline(1, color="k", ls=":"); ax[1, 2].set_title("[standalone] NIG t=1 PIT, N(0,1) CDF (control, must fail)")
    fig.suptitle("E3: standardized one-step-ahead prediction error (scalar Gaussian model)")
    fig.tight_layout(); fig.savefig(OUT / "e3_prediction_error.png", dpi=140); plt.close(fig)

    (OUT / "e3_results.json").write_text(json.dumps(dict(
        scope="scalar Gaussian one-step-ahead prediction errors standardized by the Bayesian predictive; "
              "does not establish that a PC layer performs or drives learning",
        scope_by_part=dict(
            gaussian="shared framework: ScalarGaussianBelief + PredictiveCodingLayer (standardized, raw and precision-weighted error)",
            nig_student_t="standalone closed-form NIG/Student-t check; does not call the shared framework classes"),
        seeds=C["seeds"],
        overall_pass=bool(ok),
        checks=[dict(name=n_, value=float(v), criterion=c, passed=bool(p)) for n_, v, c, p in rows]), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
