"""E4 - Ellipsoidal decomposition i.i.d. sampler (paper sec. 4; algorithm details from arXiv:2107.05956).

Scope: validates the implementation of the paper's ellipsoidal decomposition sampler (src/bayesian_reflex/sampling) on targets
with exact references, and checks it against the PR 1 scalar Gaussian belief. It does NOT test the reflex loop, Thompson
sampling, recursive Gaussian processes, derivative-aware optimisation, scaling to high dimension, or any comparison with
other inference methods. The sampler draws from pi_hat (Monte Carlo shell weights), not from pi exactly; see the docs.

Protocol (as in PR 1): criteria fixed before running, 10 seeds (0..9; pooled draws), every positive check has a negative
control that must fail, exit status 1 if any required check (positive or control) does not hold. alpha = 0.001.

Part A  one shell, 1-D N(0,1), shell {1.0 <= |x| <= 1.6}, rigorous bounds (p = 0.458); 20,000 draws x 10 seeds
        KS vs exact CDF p > alpha | mean|x| within 3 SE | mean coalescence time within 3 SE of (1-p)/p | 0 minorization violations
        CONTROLS (must fail): uniform draws without residual steps; the coalescence-time form stated in the source paper
Part B  Gaussian N(nu, S), d in {1,2,3,5}, partition matched to the target, Monte Carlo bounds as in the paper (N_i = 10,000,
        eta = 1e-5), 4,000 draws x 10 seeds; d in {2,5} also with rigorous bounds
        KS of Mahalanobis^2 vs chi2(d) p > alpha | every mean |z| < 4 | every covariance entry |z| < 4 | violations < 0.1% of steps
        CONTROLS (d=2, must fail): shells chosen uniformly instead of by pi_hat; tail truncated (fixed M = 4 shells)
Part C  non-Gaussian 1-D: bimodal 0.5 N(-2,1) + 0.5 N(2,1) and Student-t with 5 d.o.f.; 4,000 draws x 10 seeds
        KS vs exact CDF p > alpha.   CONTROL (t5, must fail): central ellipsoid only (fixed M = 1; drops ~5% of the mass. A first version used M = 4, which drops
        only ~0.13% and is below the resolution of a KS test at 40,000 draws, so it could not fail and was replaced)
Part D  continuity with PR 1 (ScalarGaussianBelief: mu0=0, s0=1, sigma_y=0.5)
        D1  sampler for the posterior after T=8 observations, partition = (mu_t, sigma_t^2) of the PR 1 belief:
            standardised draws KS vs N(0,1) p > alpha | mean and variance within 3 SE
        D2  simulation-based calibration (as PR 1 E1) with the sampler as posterior sampler, 10 seeds x 300 trials, L = 99 draws:
            chi2 on 10 rank bins p > alpha | 50/80/95% rank-interval coverage within 3 binomial SE
        CONTROL (D2, must fail): belief and sampler are both given the wrong noise level (0.25 vs 0.5), as in PR 1 E1
Exit status 0 only if every required check holds (positive checks and controls), otherwise 1.
Run: python experiments/e4_ellipsoidal_sampler.py     (about 3 minutes)
"""
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex import ScalarGaussianBelief  # noqa: E402
from bayesian_reflex.sampling import (EllipsoidalPartition, EllipsoidalSampler, LinearRadii,  # noqa: E402
                                      gaussian_log_density, gaussian_shell_log_bounds)

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEEDS = list(range(10))
ALPHA = 1e-3
rows, diag = [], {}


def add(name, value, criterion, passed, control=False):
    rows.append((("CONTROL " if control else "") + name, float(value), criterion, bool(passed)))


def summarize(sampler):
    d = sampler.diagnostics
    return dict(n_shells=d["n_shells"], mean_T=d["mean_coalescence_steps"], max_T=d["max_coalescence_steps"],
                min_p_selected=float(min(d["p_hat"][i - 1] for i in d["shell_counts"])),
                last_shell_weight=d["last_shell_weight"], violations=d["n_minorization_violations"],
                residual_steps=d["n_residual_steps"])


def merge(ds):
    return dict(n_shells=max(d["n_shells"] for d in ds), mean_T=float(np.mean([d["mean_T"] for d in ds])),
                max_T=max(d["max_T"] for d in ds), min_p_selected=min(d["min_p_selected"] for d in ds),
                max_last_shell_weight=max(d["last_shell_weight"] for d in ds),
                violations=sum(d["violations"] for d in ds), residual_steps=sum(d["residual_steps"] for d in ds))


# ============================================================================================== Part A
def part_a():
    lo, hi, K = 1.0, 1.6, 20_000
    z = 2 * (stats.norm.cdf(hi) - stats.norm.cdf(lo))

    def cdf(x):
        x = np.asarray(x)
        pos, neg = np.clip(x, lo, hi), np.clip(x, -hi, -lo)
        return (stats.norm.cdf(pos) - stats.norm.cdf(lo) + stats.norm.cdf(neg) - stats.norm.cdf(-hi)) / z

    e1 = 2 * integrate.quad(lambda t: t * stats.norm.pdf(t), lo, hi)[0] / z
    e2 = 2 * integrate.quad(lambda t: t * t * stats.norm.pdf(t), lo, hi)[0] / z
    sd = np.sqrt(e2 - e1 ** 2)
    p = np.exp(-0.5 * hi ** 2 + 0.5 * lo ** 2)
    xs, xs1, xs_u, ds, Ts = [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mk = lambda support: EllipsoidalSampler(  # noqa: E731
            gaussian_log_density([0.0], [[1.0]]),
            EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.6)), geometric_support=support, n_mc=100,
            log_bounds=gaussian_shell_log_bounds(EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.6)), [0.0], [[1.0]]))
        s0, s1 = mk(0), mk(1)
        xs.append(s0.sample_from_shells(np.full(K, 2), rng)[:, 0])
        xs1.append(s1.sample_from_shells(np.full(K, 2), rng)[:, 0])
        xs_u.append(s0.partition.sample_uniform(2, K, rng)[:, 0])
        ds.append(summarize(s0))
    x, x1, xu = (np.concatenate(v) for v in (xs, xs1, xs_u))
    n = len(x)
    ks = stats.kstest(x, cdf).pvalue
    add("A shell: KS p vs exact CDF", ks, f">{ALPHA}", ks > ALPHA)
    zm = (np.abs(x).mean() - e1) / (sd / np.sqrt(n))
    add("A shell: mean|x| z-score", zm, "|z|<3", abs(zm) < 3)
    mt = np.mean([d["mean_T"] for d in ds])
    zt = (mt - (1 - p) / p) / (np.sqrt(1 - p) / p / np.sqrt(n))
    add("A shell: mean coalescence time z-score vs (1-p)/p", zt, "|z|<3", abs(zt) < 3)
    add("A shell: minorization violations", sum(d["violations"] for d in ds), "0", sum(d["violations"] for d in ds) == 0)
    ku, k1 = stats.kstest(xu, cdf).pvalue, stats.kstest(x1, cdf).pvalue
    add("A shell: uniform draws, no residual steps: KS p", ku, f"<{ALPHA} (must fail)", ku < ALPHA, control=True)
    add("A shell: coalescence support {1,2,...} (source paper): KS p", k1, f"<{ALPHA} (must fail)", k1 < ALPHA, control=True)
    diag["A"] = dict(exact_p=float(p), mean_T=mt, theory_mean_T=float((1 - p) / p), mean_abs_x_exact=float(e1),
                     mean_abs_x_sampled=float(np.abs(x).mean()), mean_abs_x_support1=float(np.abs(x1).mean()),
                     draws=int(n))
    return x, cdf


# ============================================================================================== Part B
def target_params(d):
    nu = np.arange(1, d + 1, dtype=float)
    S = 2.0 * 0.6 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    return nu, S


def run_gaussian(d, bounds, n=4000, **kw):
    nu, S = target_params(d)
    xs, ds = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        part = EllipsoidalPartition(nu, S, LinearRadii(1.5, 0.3))
        lb = gaussian_shell_log_bounds(part, nu, S) if bounds == "exact" else None
        s = EllipsoidalSampler(gaussian_log_density(nu, S), part, log_bounds=lb, n_mc=10_000, **kw)
        shells_override = kw.pop("_uniform_shells", None)
        xs.append(s.sample(n, rng))
        ds.append(summarize(s))
    return np.concatenate(xs), merge(ds), nu, S


def check_gaussian(label, x, nu, S, dg, required_violations=True):
    d, n = len(nu), len(x)
    m2 = np.sum(np.linalg.solve(np.linalg.cholesky(S), (x - nu).T) ** 2, axis=0)
    ks = stats.kstest(m2, stats.chi2(d).cdf).pvalue
    add(f"{label}: KS p of Mahalanobis^2 vs chi2({d})", ks, f">{ALPHA}", ks > ALPHA)
    zmean = np.max(np.abs((x.mean(0) - nu) / np.sqrt(np.diag(S) / n)))
    add(f"{label}: max |z| of means", zmean, "<4", zmean < 4)
    se = np.sqrt((np.outer(np.diag(S), np.diag(S)) + S ** 2) / n)
    zcov = np.max(np.abs((np.cov(x.T).reshape(d, d) - S) / se))
    add(f"{label}: max |z| of covariance entries", zcov, "<4", zcov < 4)
    rate = dg["violations"] / max(dg["residual_steps"], 1)
    add(f"{label}: minorization violation rate", rate, "<0.001", rate < 1e-3)
    return ks


def part_b():
    out = {}
    for d in (1, 2, 3, 5):
        x, dg, nu, S = run_gaussian(d, "mc")
        check_gaussian(f"B d={d} MC bounds", x, nu, S, dg)
        diag[f"B_d{d}_mc"] = dg
        out[d] = (x, nu, S)
    for d in (2, 5):
        x, dg, nu, S = run_gaussian(d, "exact")
        ks = check_gaussian(f"B d={d} rigorous bounds", x, nu, S, dg)
        add(f"B d={d} rigorous bounds: violations", dg["violations"], "0", dg["violations"] == 0)
        diag[f"B_d{d}_exact"] = dg
    # controls on d = 2
    nu, S = target_params(2)
    xu, xt = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        part = EllipsoidalPartition(nu, S, LinearRadii(1.5, 0.3))
        s = EllipsoidalSampler(gaussian_log_density(nu, S), part, n_mc=10_000)
        M = s._M
        s._ensure_shell(M, rng)
        xu.append(s.sample_from_shells(rng.integers(1, M + 1, 4000), rng))           # shells chosen uniformly
        s2 = EllipsoidalSampler(gaussian_log_density(nu, S), part, n_mc=10_000, initial_shells=4, adaptive_shells=False)
        xt.append(s2.sample(4000, rng))                                               # tail truncated at M = 4
    for name, x in (("shells chosen uniformly (weights ignored)", np.concatenate(xu)),
                    ("tail truncated at M = 4 shells", np.concatenate(xt))):
        m2 = np.sum(np.linalg.solve(np.linalg.cholesky(S), (x - nu).T) ** 2, axis=0)
        ks = stats.kstest(m2, stats.chi2(2).cdf).pvalue
        add(f"B d=2: {name}: KS p", ks, f"<{ALPHA} (must fail)", ks < ALPHA, control=True)
    return out


# ============================================================================================== Part C
def part_c():
    comps = [(-2.0, 0.5), (2.0, 0.5)]
    pdf = lambda t: sum(w * np.exp(-0.5 * (t - m) ** 2) for m, w in comps)                 # noqa: E731
    cdf_mix = lambda t: sum(w * stats.norm.cdf(t, loc=m) for m, w in comps)                # noqa: E731
    t5 = stats.t(5)
    xm, xt, xc, dm, dt = [], [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        s = EllipsoidalSampler(lambda th: np.log(pdf(th[:, 0])),
                               EllipsoidalPartition([0.0], [[5.0]], LinearRadii(1.5, 0.5)), n_mc=10_000)
        xm.append(s.sample(4000, rng)[:, 0]); dm.append(summarize(s))
        part_t = EllipsoidalPartition([0.0], [[5.0 / 3.0]], LinearRadii(2.0, 1.0))
        ld_t = lambda th: -3.0 * np.log1p(th[:, 0] ** 2 / 5.0)                              # noqa: E731
        s = EllipsoidalSampler(ld_t, part_t, n_mc=10_000, initial_shells=8)
        xt.append(s.sample(4000, rng)[:, 0]); dt.append(summarize(s))
        s = EllipsoidalSampler(ld_t, part_t, n_mc=10_000, initial_shells=1, adaptive_shells=False)
        xc.append(s.sample(4000, rng)[:, 0])
    xm, xt, xc = (np.concatenate(v) for v in (xm, xt, xc))
    pm, pt, pc = stats.kstest(xm, cdf_mix).pvalue, stats.kstest(xt, t5.cdf).pvalue, stats.kstest(xc, t5.cdf).pvalue
    add("C bimodal normal mixture: KS p vs exact CDF", pm, f">{ALPHA}", pm > ALPHA)
    add("C Student-t(5): KS p vs exact CDF", pt, f">{ALPHA}", pt > ALPHA)
    add("C Student-t(5): central ellipsoid only (M = 1 shell): KS p", pc, f"<{ALPHA} (must fail)", pc < ALPHA, control=True)
    diag["C_mixture"], diag["C_t5"] = merge(dm), merge(dt)
    return xm, xt, cdf_mix, t5


# ============================================================================================== Part D
def posterior_log_density(y, s0, sy):
    return lambda th: (-0.5 * th[:, 0] ** 2 / s0 ** 2
                       - 0.5 * np.sum((y[None, :] - th[:, :1]) ** 2, axis=1) / sy ** 2)


def part_d():
    s0, sy, T = 1.0, 0.5, 8
    # ---- D1
    zs, ds = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(400 + seed)
        theta = rng.standard_normal()
        y = theta + sy * rng.standard_normal(T)
        belief = ScalarGaussianBelief(0.0, s0, sy, 1)
        for t in range(T):
            belief.update(y[t:t + 1])
        mu, sd = belief.mu[0], belief.sd[0]
        s = EllipsoidalSampler(posterior_log_density(y, s0, sy),
                               EllipsoidalPartition([mu], [[sd ** 2]], LinearRadii(1.5, 0.5)), n_mc=5000)
        zs.append((s.sample(4000, rng)[:, 0] - mu) / sd)
        ds.append(summarize(s))
    z = np.concatenate(zs)
    n = len(z)
    ks = stats.kstest(z, "norm").pvalue
    add("D1 PR1 posterior: KS p of standardised draws", ks, f">{ALPHA}", ks > ALPHA)
    add("D1 PR1 posterior: |mean z|", abs(z.mean()), f"<{3 / np.sqrt(n):.4f}", abs(z.mean()) < 3 / np.sqrt(n))
    add("D1 PR1 posterior: |var z - 1|", abs(z.var() - 1), f"<{3 * np.sqrt(2 / n):.4f}", abs(z.var() - 1) < 3 * np.sqrt(2 / n))
    diag["D1"] = merge(ds)

    # ---- D2: SBC with the sampler as posterior sampler (L = 99 draws -> 100 equally likely ranks)
    L, n_trials = 99, 300

    def sbc(wrong_noise=None):
        ranks = []
        for seed in SEEDS:
            rng = np.random.default_rng(500 + seed)
            theta = rng.standard_normal(n_trials)
            y = theta[:, None] + sy * rng.standard_normal((n_trials, T))
            belief = ScalarGaussianBelief(0.0, s0, wrong_noise or sy, n_trials)  # PR 1 belief gives (mu_t, sigma_t) per trial
            for t in range(T):
                belief.update(y[:, t])
            for k in range(n_trials):
                mu, sd = belief.mu[k], belief.sd[k]
                s = EllipsoidalSampler(posterior_log_density(y[k], s0, wrong_noise or sy),
                                       EllipsoidalPartition([mu], [[sd ** 2]], LinearRadii(1.5, 0.5)),
                                       n_mc=2000, initial_shells=8)
                ranks.append(int(np.sum(s.sample(L, rng)[:, 0] < theta[k])))
        return np.array(ranks)

    def sbc_checks(ranks, label, control=False):
        n = len(ranks)
        counts = np.bincount(ranks // 10, minlength=10)
        chi = stats.chisquare(counts).pvalue
        if control:
            add(f"D2 SBC {label}: chi2 rank p", chi, f"<{ALPHA} (must fail)", chi < ALPHA, control=True)
            return
        add(f"D2 SBC {label}: chi2 rank p", chi, f">{ALPHA}", chi > ALPHA)
        for nom, (a, b) in ((0.5, (25, 74)), (0.8, (10, 89)), (0.95, (3, 97))):
            cov = np.mean((ranks >= a) & (ranks <= b))
            tol = 3 * np.sqrt(nom * (1 - nom) / n)
            add(f"D2 SBC {label}: coverage {int(nom * 100)}%", cov, f"|v-{nom}|<={tol:.4f}", abs(cov - nom) <= tol)

    ranks = sbc()
    sbc_checks(ranks, "(sampler vs PR1-belief partition)")
    sbc_checks(sbc(wrong_noise=0.25), "wrong noise level 0.25", control=True)
    diag["D2"] = dict(n_trials=len(ranks), rank_counts_10_bins=np.bincount(ranks // 10, minlength=10).tolist())
    return z, ranks


# ============================================================================================== main
def main():
    t0 = time.time()
    xa, cdf_a = part_a()
    bres = part_b()
    xm, xt, cdf_mix, t5 = part_c()
    z, ranks = part_d()

    print(f"{'check':74s}{'value':>11s}  criterion")
    for n_, v, c, p in rows:
        print(f"{n_:74s}{v:11.3e}  {c:30s}{'PASS' if p else 'FAIL'}")
    ok = all(r[3] for r in rows)
    print(f"\nE4 (ellipsoidal decomposition sampler; {len(rows)} checks, {time.time() - t0:.0f} s):", "PASS" if ok else "FAIL")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    g = np.linspace(1.0, 1.6, 200)
    ax[0, 0].hist(np.abs(xa), bins=60, density=True, alpha=0.6, label="sampler")
    ax[0, 0].plot(g, 2 * stats.norm.pdf(g) / (2 * (stats.norm.cdf(1.6) - stats.norm.cdf(1.0))), "k-", label="exact |x| density")
    ax[0, 0].set_title("A: one shell, perfect sampler"); ax[0, 0].legend(fontsize=7)
    x2, nu2, S2 = bres[2]
    ax[0, 1].scatter(x2[:3000, 0], x2[:3000, 1], s=2, alpha=0.4)
    ev, U = np.linalg.eigh(S2)
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (1, 2, 3):
        e = (U * np.sqrt(ev)) @ (r * np.vstack([np.cos(th), np.sin(th)]))
        ax[0, 1].plot(nu2[0] + e[0], nu2[1] + e[1], "k-", lw=0.7)
    ax[0, 1].set_title("B: d=2 draws and 1,2,3-sigma contours")
    gx = np.linspace(-12, 12, 400)
    ax[0, 2].hist(xm, bins=120, density=True, alpha=0.6, label="sampler (bimodal)")
    ax[0, 2].plot(gx, 0.5 * stats.norm.pdf(gx, -2) + 0.5 * stats.norm.pdf(gx, 2), "k-")
    ax[0, 2].set_xlim(-8, 8); ax[0, 2].set_title("C: bimodal mixture"); ax[0, 2].legend(fontsize=7)
    ax[1, 0].hist(xt, bins=np.linspace(-12, 12, 160), density=True, alpha=0.6, label="sampler")
    ax[1, 0].plot(gx, t5.pdf(gx), "k-", label="Student-t(5)")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("C: Student-t(5), log scale"); ax[1, 0].legend(fontsize=7)
    q = np.linspace(0.001, 0.999, 200)
    ax[1, 1].plot(stats.norm.ppf(q), np.quantile(z, q), "b-"); ax[1, 1].plot(stats.norm.ppf(q), stats.norm.ppf(q), "k:")
    ax[1, 1].set_title("D1: PR1 posterior, standardised draws QQ")
    ax[1, 2].bar(np.arange(10), np.bincount(ranks // 10, minlength=10)); ax[1, 2].axhline(len(ranks) / 10, color="k", ls=":")
    ax[1, 2].set_title("D2: SBC rank histogram (10 bins)")
    fig.suptitle("E4: ellipsoidal decomposition i.i.d. sampler")
    fig.tight_layout(); fig.savefig(OUT / "e4_ellipsoidal_sampler.png", dpi=140); plt.close(fig)

    (OUT / "e4_ellipsoidal_results.json").write_text(json.dumps(dict(
        scope="implementation of the paper's ellipsoidal decomposition sampler (sec. 4); draws are from pi_hat "
              "(Monte Carlo shell weights); no comparison with other inference methods",
        seeds=SEEDS, alpha=ALPHA, overall_pass=bool(ok),
        checks=[dict(name=n_, value=v, criterion=c, passed=p) for n_, v, c, p in rows], diagnostics=diag), indent=2, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
