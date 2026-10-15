"""Ellipsoidal decomposition sampler: geometry, input validation, exactness against closed forms, PR 1 continuity."""
import numpy as np
import pytest
from scipy import stats
from scipy.special import gammaln

from bayesian_reflex import ScalarGaussianBelief
from bayesian_reflex.sampling import (EllipsoidalPartition, EllipsoidalSampler, LinearRadii, MinorizationViolationError,
                                      SamplerInfeasibleError, gaussian_log_density, gaussian_log_tail_bound,
                                      gaussian_shell_log_bounds)

ALPHA = 1e-3                                           # same significance level as the PR 1 experiments


def _gauss_sampler(mean, cov, p_mean, p_cov, r1, a, bounds="exact", **kw):
    """bounds='exact': rigorous mode (Gaussian shell and tail bounds); 'mc': the paper's Monte Carlo mode."""
    part = EllipsoidalPartition(p_mean, p_cov, LinearRadii(r1, a))
    if bounds == "exact":
        return EllipsoidalSampler(gaussian_log_density(mean, cov), part, mode="rigorous",
                                  log_bounds=gaussian_shell_log_bounds(part, mean, cov),
                                  log_tail_bound=gaussian_log_tail_bound(part, mean, cov), **kw)
    return EllipsoidalSampler(gaussian_log_density(mean, cov), part, **kw)


# ------------------------------------------------------------------------------------------------ geometry
def test_log_volume_matches_closed_form_in_2d():
    part = EllipsoidalPartition([1.0, -2.0], np.diag([4.0, 1.0]), LinearRadii(1.0, 1.0))   # |B| = 2, radii 1, 2, 3
    assert np.isclose(np.exp(part.log_volume(1)), 2 * np.pi * 1.0)
    assert np.isclose(np.exp(part.log_volume(2)), 2 * np.pi * (4 - 1))
    assert np.isclose(np.exp(part.log_volume(3)), 2 * np.pi * (9 - 4))


@pytest.mark.parametrize("d", [1, 3, 5])
def test_shell_volumes_add_up_to_the_ball(d):
    rng = np.random.default_rng(d)
    A = rng.standard_normal((d, d))
    cov = A @ A.T + d * np.eye(d)
    part = EllipsoidalPartition(np.zeros(d), cov, LinearRadii(0.7, 0.4))
    k = 6
    total = sum(np.exp(part.log_volume(i)) for i in range(1, k + 1))
    ball = np.exp(0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1) + np.sum(np.log(np.diag(part.chol))) + d * np.log(part.radius(k)))
    assert np.isclose(total, ball)


def test_uniform_sampling_stays_in_the_shell_and_is_uniform():
    rng = np.random.default_rng(0)
    cov = np.array([[2.0, 0.6], [0.6, 1.0]])
    part = EllipsoidalPartition([0.5, -1.0], cov, LinearRadii(1.0, 0.8))
    x = part.sample_uniform(3, 60_000, rng)                              # radii 1.0, 1.8, 2.6 -> shell 3 = [1.8, 2.6]
    m = np.sqrt(part.mahalanobis_sq(x))
    assert m.min() >= 1.8 - 1e-9 and m.max() <= 2.6 + 1e-9
    u = (m ** 2 - 1.8 ** 2) / (2.6 ** 2 - 1.8 ** 2)                      # |y|^d uniform for d = 2
    assert stats.kstest(u, "uniform").pvalue > ALPHA
    y = np.linalg.solve(part.chol, (x - part.mean).T).T                  # whitened: isotropic second moments
    assert np.allclose(np.cov(y.T), np.eye(2) * np.mean(m ** 2) / 2, atol=0.03)
    assert np.allclose(x.mean(0), part.mean, atol=0.03)


def test_uniform_sampling_works_for_extremely_thin_shells_and_high_dimension():
    rng = np.random.default_rng(1)
    thin = EllipsoidalPartition([0.0, 0.0], np.eye(2), lambda i: 0.0 if i == 0 else 2.0 + 1e-9 * (i - 1))
    m = np.sqrt(thin.mahalanobis_sq(thin.sample_uniform(2, 1000, rng)))
    assert m.min() >= 2.0 - 1e-8 and m.max() <= 2.0 + 2e-9
    big = EllipsoidalPartition(np.zeros(200), np.eye(200), LinearRadii(14.0, 0.5))
    m = np.sqrt(big.mahalanobis_sq(big.sample_uniform(3, 500, rng)))
    assert np.all(np.isfinite(m)) and m.min() >= 14.5 - 1e-6 and m.max() <= 15.0 + 1e-6


@pytest.mark.parametrize("args", [
    dict(mean=np.zeros((2, 2)), cov=np.eye(2)),
    dict(mean=[0.0, 0.0], cov=np.eye(3)),
    dict(mean=[0.0, 0.0], cov=[[1.0, 0.5], [0.0, 1.0]]),                 # not symmetric
    dict(mean=[0.0, 0.0], cov=[[1.0, 2.0], [2.0, 1.0]]),                 # not positive definite
    dict(mean=[0.0, np.nan], cov=np.eye(2)),
    dict(mean=[0.0, 0.0], cov=[[np.inf, 0.0], [0.0, 1.0]]),
    dict(mean=[0.0, 0.0], cov=np.eye(2), radii=3.0),                     # radii must be callable
])
def test_partition_rejects_invalid_arguments(args):
    kw = dict(mean=[0.0], cov=[[1.0]], radii=LinearRadii(1.0, 0.5))
    kw.update(args)
    with pytest.raises(ValueError):
        EllipsoidalPartition(**kw)


@pytest.mark.parametrize("r1,a", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (np.nan, 1.0), (1.0, np.inf), ("x", 1.0)])
def test_linear_radii_rejects_invalid_arguments(r1, a):
    with pytest.raises(ValueError):
        LinearRadii(r1, a)


def test_non_increasing_radii_are_rejected():
    part = EllipsoidalPartition([0.0], [[1.0]], lambda i: 0.0 if i == 0 else 1.0)    # constant radius
    with pytest.raises(ValueError, match="increasing"):
        part.log_volume(2)


# ------------------------------------------------------------------------------------------------ sampler input checks
def test_sampler_rejects_invalid_arguments():
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.5))
    ld = gaussian_log_density([0.0], [[1.0]])
    for kw in (dict(n_mc=1), dict(eta=-0.1), dict(eta=1.0), dict(geometric_support=2), dict(log_bounds=3),
               dict(initial_shells=0), dict(max_shells=0), dict(min_p=0.0), dict(mode="exact"), dict(on_violation="ignore"),
               dict(tail_tol=0.0), dict(tail_tol=1.0), dict(patience=0), dict(n_shells=0), dict(log_tail_bound="x"),
               dict(log_bounds=lambda i: (0.0, 0.0)),               # bounds imply mode='rigorous'
               dict(mode="rigorous"),                               # rigorous needs both bounds
               dict(mode="rigorous", log_bounds=lambda i: (0.0, 0.0)),
               dict(mode="rigorous", log_bounds=lambda i: (0.0, 0.0), log_tail_bound=lambda M: 0.0, on_violation="count"),
               dict(violation_tol=-0.1),
               dict(mode="rigorous", log_bounds=lambda i: (0.0, 0.0), log_tail_bound=lambda M: 0.0, violation_tol=0.1)):
        with pytest.raises(ValueError):
            EllipsoidalSampler(ld, part, **kw)
    with pytest.raises(ValueError):
        EllipsoidalSampler("not callable", part)
    with pytest.raises(ValueError):
        EllipsoidalSampler(ld, "not a partition")
    s = EllipsoidalSampler(ld, part, n_mc=500)
    rng = np.random.default_rng(0)
    for n in (0, -1, 2.5, True, None):
        with pytest.raises(ValueError):
            s.sample(n, rng)
    with pytest.raises(ValueError):
        s.sample(5, 123)                                                 # rng must be a Generator
    for bad in (np.array([0.0]), np.array([[1]]), np.array([0, 1.5]), np.array([0, -1]), "ab"):
        with pytest.raises(ValueError):
            s.sample_from_shells(bad, rng)


@pytest.mark.parametrize("bad", [lambda th: np.zeros(len(th) + 1), lambda th: np.full(len(th), np.nan),
                                 lambda th: np.full(len(th), np.inf), lambda th: np.zeros((len(th), 1))])
def test_sampler_rejects_a_misbehaving_log_density(bad):
    s = EllipsoidalSampler(bad, EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.5)), n_mc=100)
    with pytest.raises(ValueError):
        s.sample(3, np.random.default_rng(0))


def test_infeasible_radii_raise_instead_of_returning_garbage():
    s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=12.0, a=0.5, bounds="mc", n_mc=2000)   # inf/sup ~ e^-72
    with pytest.raises(SamplerInfeasibleError, match="p_hat"):
        s.sample(5, np.random.default_rng(0))


def test_coalescence_cap_raises_instead_of_truncating():
    s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=3.0, a=0.5, n_mc=500, max_coalescence=3)   # p of shell 1 ~ e^-4.5
    with pytest.raises(SamplerInfeasibleError, match="coalescence"):
        s.sample(2000, np.random.default_rng(0))


def test_max_shells_is_enforced_for_heavy_tails():
    cauchy = lambda th: -np.log1p(th[:, 0] ** 2)
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.5))
    s = EllipsoidalSampler(cauchy, part, n_mc=500, initial_shells=2, max_shells=4)
    with pytest.raises(SamplerInfeasibleError, match="max_shells"):
        s.sample(20_000, np.random.default_rng(0))


# ------------------------------------------------------------------------------------------------ exactness
def _exact_shell_cdf(lo, hi):
    """CDF of N(0,1) restricted to {lo <= |x| <= hi}."""
    z = 2 * (stats.norm.cdf(hi) - stats.norm.cdf(lo))

    def cdf(x):
        x = np.asarray(x)
        pos, neg = np.clip(x, lo, hi), np.clip(x, -hi, -lo)
        return (stats.norm.cdf(pos) - stats.norm.cdf(lo) + stats.norm.cdf(neg) - stats.norm.cdf(-hi)) / z

    return cdf


def test_perfect_sampler_on_one_shell_is_exact_and_coalescence_time_matches_theory():
    rng = np.random.default_rng(0)
    s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=1.0, a=0.6)             # shell 2 = |x| in [1.0, 1.6]
    K = 200_000
    x = s.sample_from_shells(np.full(K, 2), rng)[:, 0]
    assert stats.kstest(x, _exact_shell_cdf(1.0, 1.6)).pvalue > ALPHA
    p = np.exp(-0.5 * 1.6 ** 2 + 0.5 * 1.0 ** 2)                                  # exact inf/sup on the shell
    assert np.isclose(s.diagnostics["p_hat"][1], p)
    d = s.diagnostics
    se = np.sqrt(1 - p) / p / np.sqrt(K)                                          # T ~ Geometric on {0,1,..}
    assert abs(d["mean_coalescence_steps"] - (1 - p) / p) < 4 * se
    assert d["n_minorization_violations"] == 0


def test_the_coalescence_form_stated_in_the_source_is_biased():
    """Documented finding: T supported on {1,2,...} returns (pi_i - p Q_i)/(1 - p), not pi_i (see docs)."""
    rng = np.random.default_rng(0)
    s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=1.0, a=0.6, geometric_support=1)
    x = s.sample_from_shells(np.full(200_000, 2), rng)[:, 0]
    assert stats.kstest(x, _exact_shell_cdf(1.0, 1.6)).pvalue < 1e-12
    # the analytic prediction for E|x| under (pi_i - p Q_i)/(1 - p):
    p = np.exp(-0.5 * 1.6 ** 2 + 0.5)
    e_pi = (stats.norm.pdf(1.0) - stats.norm.pdf(1.6)) / (stats.norm.cdf(1.6) - stats.norm.cdf(1.0))
    predicted = (e_pi - p * 1.3) / (1 - p)                                        # E_Q|x| = (1.0 + 1.6)/2
    assert abs(np.abs(x).mean() - predicted) < 0.01


def test_overall_sampler_1d_normal_with_exact_bounds():
    s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=1.5, a=0.5, initial_shells=4)
    x = s.sample(60_000, np.random.default_rng(1))[:, 0]
    assert stats.kstest(x, "norm").pvalue > ALPHA
    assert abs(x.mean()) < 4 / np.sqrt(len(x)) and abs(x.var() - 1) < 4 * np.sqrt(2 / len(x))
    assert s.diagnostics["n_shells"] > 4                                          # the doubling rule was exercised
    assert s.diagnostics["n_minorization_violations"] == 0


def test_overall_sampler_2d_correlated_gaussian_matches_the_chi_square_law():
    mean, cov = np.array([1.0, -2.0]), np.array([[2.0, 0.9], [0.9, 1.0]])
    s = _gauss_sampler(mean, cov, mean, cov, r1=1.5, a=0.4)
    x = s.sample(40_000, np.random.default_rng(2))
    m2 = np.sum(np.linalg.solve(np.linalg.cholesky(cov), (x - mean).T) ** 2, axis=0)
    assert stats.kstest(m2, stats.chi2(2).cdf).pvalue > ALPHA
    n = len(x)
    z_mean = (x.mean(0) - mean) / np.sqrt(np.diag(cov) / n)
    assert np.all(np.abs(z_mean) < 4)
    emp = np.cov(x.T)
    se = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov ** 2) / n)
    assert np.all(np.abs(emp - cov) / se < 4)


def test_shifted_and_rescaled_partition_with_the_same_shape_is_valid_and_bounds_are_exact():
    """SOURCE sec. 2.2: any (mu, Sigma) is valid. Same-shape partitions make the Gaussian shell bounds exact."""
    mean, cov = np.array([0.0, 0.0]), np.array([[1.0, 0.5], [0.5, 1.5]])
    s = _gauss_sampler(mean, cov, [0.25, -0.15], 1.3 * cov, r1=1.2, a=0.4)
    x = s.sample(20_000, np.random.default_rng(3))
    m2 = np.sum(np.linalg.solve(np.linalg.cholesky(cov), x.T) ** 2, axis=0)
    assert stats.kstest(m2, stats.chi2(2).cdf).pvalue > ALPHA
    assert s.diagnostics["n_minorization_violations"] == 0


@pytest.mark.parametrize("bounds", ["mc", "exact"])
def test_mildly_mismatched_anisotropic_partition_is_still_exact(bounds):
    mean, cov = np.array([0.0, 0.0]), np.array([[1.0, 0.5], [0.5, 1.5]])
    s = _gauss_sampler(mean, cov, [0.2, -0.1], np.array([[1.15, 0.35], [0.35, 1.7]]), r1=1.2, a=0.4, bounds=bounds)
    x = s.sample(20_000, np.random.default_rng(3))
    m2 = np.sum(np.linalg.solve(np.linalg.cholesky(cov), x.T) ** 2, axis=0)
    assert stats.kstest(m2, stats.chi2(2).cdf).pvalue > ALPHA
    assert s.diagnostics["n_minorization_violations"] == 0


def test_strongly_mismatched_shape_is_reported_as_infeasible_not_silently_biased():
    """Across a shell the density varies by ~exp(-(beta_max - beta_min) rho^2 / 2) whatever the shell width, so a
    partition whose shape is far from the target's makes p_hat collapse in the tails: the sampler must raise."""
    mean, cov = np.array([0.0, 0.0]), np.array([[1.0, 0.5], [0.5, 1.5]])
    s = _gauss_sampler(mean, cov, [0.6, -0.4], np.diag([2.0, 0.7]), r1=1.2, a=0.4, bounds="mc", max_coalescence=2000)
    with pytest.raises(SamplerInfeasibleError):
        s.sample(5000, np.random.default_rng(0))


def test_monte_carlo_bounds_as_in_the_paper_3d():
    mean, cov = np.zeros(3), np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]])
    s = _gauss_sampler(mean, cov, mean, cov, r1=1.5, a=0.4, bounds="mc", n_mc=10_000)
    x = s.sample(30_000, np.random.default_rng(4))
    m2 = np.sum(np.linalg.solve(np.linalg.cholesky(cov), x.T) ** 2, axis=0)
    assert stats.kstest(m2, stats.chi2(3).cdf).pvalue > ALPHA
    assert s.diagnostics["n_minorization_violations"] < 0.001 * s.diagnostics["n_residual_steps"]


def test_bimodal_target_is_sampled_exactly():
    comps = [(-2.0, 0.5), (2.0, 0.5)]
    pdf = lambda t: sum(w * np.exp(-0.5 * ((t - m) / 1.0) ** 2) for m, w in comps)
    ld = lambda th: np.log(pdf(th[:, 0]))
    part = EllipsoidalPartition([0.0], [[5.0]], LinearRadii(1.5, 0.5))                  # mean/variance of the mixture
    s = EllipsoidalSampler(ld, part, n_mc=10_000)
    x = s.sample(40_000, np.random.default_rng(5))[:, 0]
    cdf = lambda t: sum(w * stats.norm.cdf(t, loc=m) for m, w in comps)
    assert stats.kstest(x, cdf).pvalue > ALPHA


def test_same_seed_gives_identical_draws():
    def draw():
        s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=1.5, a=0.5, n_mc=500)
        return s.sample(300, np.random.default_rng(7))
    assert np.array_equal(draw(), draw())


def test_draws_from_a_chosen_shell_lie_in_that_shell():
    s = _gauss_sampler([0.0, 0.0], np.eye(2), [0.0, 0.0], np.eye(2), r1=1.0, a=0.5)
    x = s.sample_from_shells(np.array([1, 2, 3, 3, 5]), np.random.default_rng(0))
    m = np.sqrt(s.partition.mahalanobis_sq(x))
    lo = np.array([0.0, 1.0, 1.5, 1.5, 2.5])
    hi = np.array([1.0, 1.5, 2.0, 2.0, 3.0])
    assert np.all(m >= lo - 1e-9) and np.all(m <= hi + 1e-9)


# ------------------------------------------------------------------------------------------------ continuity with PR 1
def _posterior_log_density(y, s0, sy):
    """log prior + log likelihood of theta for observations y: the generic Bayes-theorem form (paper eq. 9)."""
    def ld(theta):
        th = theta[:, 0]
        return -0.5 * th ** 2 / s0 ** 2 - 0.5 * np.sum((y[None, :] - th[:, None]) ** 2, axis=1) / sy ** 2
    return ld


@pytest.mark.parametrize("shift,scale", [(0.0, 1.0), (0.3, 1.3)])
def test_sampler_reproduces_the_pr1_conjugate_posterior(shift, scale):
    rng = np.random.default_rng(11)
    s0, sy, T = 1.0, 0.5, 8
    theta = rng.standard_normal()
    y = theta + sy * rng.standard_normal(T)
    belief = ScalarGaussianBelief(0.0, s0, sy, 1)
    for t in range(T):
        belief.update(y[t:t + 1])
    mu, sd = belief.mu[0], belief.sd[0]
    part = EllipsoidalPartition([mu + shift * sd], [[(scale * sd) ** 2]], LinearRadii(1.5, 0.5))
    s = EllipsoidalSampler(_posterior_log_density(y, s0, sy), part, n_mc=5000)
    x = s.sample(40_000, rng)[:, 0]
    assert stats.kstest(x, stats.norm(mu, sd).cdf).pvalue > ALPHA
    assert abs(x.mean() - mu) < 4 * sd / np.sqrt(len(x))
    assert abs(x.std() / sd - 1) < 4 / np.sqrt(2 * len(x))


# ------------------------------------------------------------------------------------------------ review: shell coverage is fixed
# before, and independent of, the returned draws; sample(1), repeated calls and bulk sampling agree
def _fresh_2d(seed=0, **kw):
    mean, cov = np.array([0.5, -0.5]), np.array([[1.5, 0.4], [0.4, 1.0]])
    return _gauss_sampler(mean, cov, mean, cov, r1=1.5, a=0.4, bounds="mc", n_mc=3000, **kw), mean, cov


def _maha2(x, mean, cov):
    return np.sum(np.linalg.solve(np.linalg.cholesky(cov), (x - mean).T) ** 2, axis=0)


def test_shells_are_fixed_by_prepare_and_never_change_with_the_draws():
    s, _, _ = _fresh_2d()
    rng = np.random.default_rng(0)
    s.prepare(rng)
    M, cum = s._M, s._cum.copy()
    s.sample(1, rng)
    s.sample(30_000, rng)
    assert s._M == M and np.array_equal(s._cum, cum)                      # nothing moved after 30,001 draws
    assert s.prepare(rng) is s and s._M == M                              # idempotent


def test_shell_set_does_not_depend_on_how_many_draws_are_requested():
    """Under the paper's Algorithm 1 the shell count grows when a returned draw lands in the last shell, so it depends on the
    batch size; here the same seed gives the same shells whether the first call asks for 1 draw or for 50,000."""
    shells = []
    for n in (1, 50_000):
        s = EllipsoidalSampler(gaussian_log_density([0.0], [[1.0]]),
                               EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.5, 0.5)), n_mc=2000, initial_shells=4)
        s.sample(n, np.random.default_rng(3))
        shells.append((s._M, s._cum.copy()))
    assert shells[0][0] == shells[1][0] and np.array_equal(shells[0][1], shells[1][1])


def test_sample_one_repeated_calls_and_bulk_sampling_draw_from_the_same_distribution():
    s, mean, cov = _fresh_2d()
    rng = np.random.default_rng(1)
    s.prepare(rng)
    singles = np.vstack([s.sample(1, rng) for _ in range(2500)])
    batches = np.vstack([s.sample(100, rng) for _ in range(25)])
    bulk = s.sample(2500, rng)
    m = {k: _maha2(v, mean, cov) for k, v in (("single", singles), ("batches", batches), ("bulk", bulk))}
    for k, v in m.items():
        assert stats.kstest(v, stats.chi2(2).cdf).pvalue > ALPHA, k
    assert stats.ks_2samp(m["single"], m["bulk"]).pvalue > ALPHA
    assert stats.ks_2samp(m["batches"], m["bulk"]).pvalue > ALPHA
    assert stats.ks_2samp(m["single"], m["batches"]).pvalue > ALPHA
    # the shell chosen for each draw has the same law whatever the call size
    shells = [np.searchsorted(s.partition.radius(1) + 0.4 * np.arange(s._M), np.sqrt(v)) for v in m.values()]
    assert all(np.array_equal(np.bincount(sh, minlength=s._M + 1)[:4] > 0, np.bincount(shells[0], minlength=s._M + 1)[:4] > 0)
               for sh in shells)


def test_coverage_stops_only_when_the_tail_criterion_holds():
    mean, cov = np.zeros(2), np.eye(2)
    s = _gauss_sampler(mean, cov, mean, cov, r1=1.0, a=0.5, initial_shells=4, tail_tol=1e-6)      # rigorous: bound
    s.prepare(np.random.default_rng(0))
    d = s.diagnostics
    assert d["tail_bound_rigorous"] and d["omitted_mass_bound"] <= 1e-6 and d["n_shells"] >= 8
    assert np.exp(gaussian_log_tail_bound(s.partition, mean, cov)(d["n_shells"] // 2)) / (2 * np.pi) > 1e-6 * 0.0  # defined
    t = _gauss_sampler(mean, cov, mean, cov, r1=1.0, a=0.5, bounds="mc", initial_shells=4, tail_tol=1e-6, n_mc=3000)
    t.prepare(np.random.default_rng(0))
    e = t.diagnostics
    assert not e["tail_bound_rigorous"] and e["omitted_mass_bound"] is None and e["omitted_mass_estimate"] <= 1e-6
    f = _gauss_sampler(mean, cov, mean, cov, r1=1.0, a=0.5, bounds="mc", n_shells=3, n_mc=500)
    f.prepare(np.random.default_rng(0))
    g = f.diagnostics
    assert g["shells_fixed"] and g["n_shells"] == 3 and g["omitted_mass_bound"] is None and g["omitted_mass_estimate"] is None


@pytest.mark.parametrize("p_mean,p_cov", [([0.0, 0.0], np.eye(2)), ([0.3, -0.2], np.diag([1.3, 1.1]))])
def test_gaussian_tail_bound_is_a_valid_upper_bound(p_mean, p_cov):
    rng = np.random.default_rng(0)
    mean, cov = np.array([0.0, 0.0]), np.eye(2)
    part = EllipsoidalPartition(p_mean, p_cov, LinearRadii(0.8, 0.5))
    bound = gaussian_log_tail_bound(part, mean, cov)
    x = rng.standard_normal((400_000, 2))                                  # draws from the target N(0, I)
    outside = lambda M: np.mean(part.mahalanobis_sq(x) > part.radius(M) ** 2)
    for M in (1, 2, 3, 4):
        true_mass = outside(M)                                             # P(outside), the bound is on P * (2 pi)^(d/2)|cov|^(1/2)
        assert np.exp(bound(M)) / (2 * np.pi) >= true_mass - 4 * np.sqrt(max(true_mass, 1e-9) / len(x))


# ------------------------------------------------------------------------------------------------ review: modes and violations
def test_modes_and_the_perfect_sampling_flag():
    mc, _, _ = _fresh_2d()
    mc.sample(5, np.random.default_rng(0))
    assert mc.diagnostics["mode"] == "monte_carlo" and mc.diagnostics["perfect_sampling_guaranteed"] is False
    ri = _gauss_sampler(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), r1=1.5, a=0.4)
    ri.sample(5, np.random.default_rng(0))
    assert ri.diagnostics["mode"] == "rigorous" and ri.diagnostics["perfect_sampling_guaranteed"] is True
    assert 0.0 <= ri.diagnostics["tv_weight_error_estimate"] < 0.05


def test_the_weight_error_estimate_shrinks_with_the_monte_carlo_size():
    est = {}
    for n_mc in (300, 20_000):
        s = _gauss_sampler([0.0], [[1.0]], [0.0], [[1.0]], r1=1.5, a=0.5, bounds="mc", n_mc=n_mc)
        s.prepare(np.random.default_rng(0))
        est[n_mc] = s.diagnostics["tv_weight_error_estimate"]
    assert est[300] > 3 * est[20_000] > 0


def test_rigorous_mode_raises_when_the_supplied_bounds_are_wrong():
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(1.0, 0.6))
    wrong = lambda i: (-0.05, 0.0)                                         # claims p = 0.95 on shells where it is ~0.46
    s = EllipsoidalSampler(gaussian_log_density([0.0], [[1.0]]), part, mode="rigorous", log_bounds=wrong,
                           log_tail_bound=gaussian_log_tail_bound(part, [0.0], [[1.0]]), n_mc=500)
    with pytest.raises(MinorizationViolationError, match="outside the bounds"):
        s.sample_from_shells(np.full(200, 2), np.random.default_rng(0))


def _spike_target(center=0.5, width=0.02, height=1000.0):
    """log of exp(-|x|) plus a narrow spike, on the shell |x| <= 2; a Monte Carlo min/max with few draws misses the spike."""
    return lambda th: np.log(np.exp(-np.abs(th[:, 0])) + height * (np.abs(th[:, 0] - center) < width / 2))


def _first_seed_where_mc_misses_the_spike(n_mc):
    ld = _spike_target()
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(2.0, 1.0))
    for seed in range(200):
        th = part.sample_uniform(1, n_mc, np.random.default_rng(seed))
        if ld(th).max() < np.log(100.0):                                   # no draw landed in the spike
            return seed
    raise AssertionError("no seed found")


def test_a_detected_violation_raises_by_default_and_is_only_counted_on_request():
    n_mc = 100
    seed = _first_seed_where_mc_misses_the_spike(n_mc)
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(2.0, 1.0))
    ld = _spike_target()
    strict = EllipsoidalSampler(ld, part, n_mc=n_mc, n_shells=1)
    with pytest.raises(MinorizationViolationError, match="acceptance ratio"):
        strict.sample(20_000, np.random.default_rng(seed))
    lenient = EllipsoidalSampler(ld, part, n_mc=n_mc, n_shells=1, on_violation="count")
    lenient.sample(20_000, np.random.default_rng(seed))
    d = lenient.diagnostics
    assert d["n_minorization_violations"] > 0 and d["perfect_sampling_guaranteed"] is False


def test_monte_carlo_mode_can_be_silently_wrong_while_rigorous_mode_is_not():
    """Documents why the guarantee is reserved for rigorous bounds: when the Monte Carlo min/max miss a spike and p_hat is
    close to 1, no residual steps are taken, no violation is ever detected, and the draws are badly wrong."""
    n_mc = 100
    seed = _first_seed_where_mc_misses_the_spike(n_mc)
    part = EllipsoidalPartition([0.0], [[1.0]], LinearRadii(2.0, 1.0))
    flat_spike = lambda th: np.log(1.0 + 1000.0 * (np.abs(th[:, 0] - 0.5) < 0.01))      # flat background plus the spike
    true_spike_share = 1000.0 * 0.02 / (1000.0 * 0.02 + 4.0)                              # ~0.83 of the mass
    mc = EllipsoidalSampler(flat_spike, part, n_mc=n_mc, n_shells=1)
    x = mc.sample(20_000, np.random.default_rng(seed))[:, 0]
    assert mc.diagnostics["n_minorization_violations"] == 0                              # nothing detected ...
    assert np.mean(np.abs(x - 0.5) < 0.01) < 0.05 < true_spike_share                     # ... yet the spike is missed
    rig = EllipsoidalSampler(flat_spike, part, mode="rigorous", n_mc=n_mc, n_shells=1,
                             log_bounds=lambda i: (0.0, np.log(1001.0)), log_tail_bound=lambda M: -np.inf)
    y = rig.sample(2000, np.random.default_rng(1))[:, 0]
    share = np.mean(np.abs(y - 0.5) < 0.01)
    assert abs(share - true_spike_share) < 4 * np.sqrt(true_spike_share * (1 - true_spike_share) / 2000)


def test_violation_tolerance_in_monte_carlo_mode_is_explicit_and_reported():
    """Estimated bounds are optimistic by about the Monte Carlo spacing, so rare tiny shortfalls are expected. They are counted
    and the worst is reported; with violation_tol = 0 the same run raises."""
    t5 = lambda th: -3.0 * np.log1p(th[:, 0] ** 2 / 5.0)
    part = EllipsoidalPartition([0.0], [[5.0 / 3.0]], LinearRadii(2.0, 1.0))
    kw = dict(n_mc=10_000, initial_shells=8)
    tolerant = EllipsoidalSampler(t5, part, **kw)                          # default violation_tol = 1e-2
    tolerant.sample(40_000, np.random.default_rng(0))
    d = tolerant.diagnostics
    assert 0 < d["n_minorization_violations"] < 1e-3 * d["n_residual_steps"] and 0 < d["max_violation_shortfall"] < 1e-2
    strict = EllipsoidalSampler(t5, part, violation_tol=0.0, **kw)
    with pytest.raises(MinorizationViolationError, match="relative shortfall"):
        strict.sample(40_000, np.random.default_rng(0))
