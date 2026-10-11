import numpy as np
import pytest

from bayesian_reflex import BayesianReflex, PredictiveCodingLayer, ScalarGaussianBelief


def test_posterior_matches_closed_form_and_batch():
    rng = np.random.default_rng(0)
    mu0, s0, sy, T = 0.3, 1.2, 0.5, 40
    y = 1.5 + sy * rng.standard_normal(T)
    b = ScalarGaussianBelief(mu0, s0, sy, 1)
    for t in range(T):
        b.update(y[t:t + 1])
    prec = 1 / s0**2 + T / sy**2
    mean = (mu0 / s0**2 + y.sum() / sy**2) / prec
    assert np.allclose(b.mu, mean) and np.allclose(b.var, 1 / prec)


def test_standardized_errors_are_standard_normal():
    rng = np.random.default_rng(1)
    n, T, sy = 2000, 30, 0.5
    theta = rng.standard_normal(n)
    b = ScalarGaussianBelief(0.0, 1.0, sy, n)
    pc, z = PredictiveCodingLayer(), []
    for _ in range(T):
        y = theta + sy * rng.standard_normal(n)
        y_hat, v = b.predict()
        z.append(pc.standardized_error(y, y_hat, v))
        b.update(y)
    z = np.concatenate(z)
    assert abs(z.mean()) < 0.02 and abs(z.var() - 1) < 0.03


def test_reflex_step_keys_and_predict_before_update():
    b = ScalarGaussianBelief(0.0, 1.0, 0.5, 1)
    r = BayesianReflex(b, PredictiveCodingLayer(), policy=lambda m: float(m.sd[0]))
    out = r.step(np.array([1.0]))
    assert set(out) == {"y_hat", "var_pred", "z", "action"}
    assert out["y_hat"][0] == 0.0 and b.mu[0] != 0.0


def test_prediction_error_is_diagnostic_only():
    """The belief after reflex.step equals the belief after a bare model.update on the same y."""
    y = np.array([1.0, -2.0, 0.3])
    a = ScalarGaussianBelief(0.1, 1.3, 0.5, 3)
    b = ScalarGaussianBelief(0.1, 1.3, 0.5, 3)
    BayesianReflex(a, PredictiveCodingLayer()).step(y)
    b.update(y)
    assert np.array_equal(a.mu, b.mu) and np.array_equal(a.prec, b.prec)


def test_precision_weighted_vs_standardized_error():
    pc = PredictiveCodingLayer()
    y, y_hat, v = np.array([1.0, -0.4]), np.array([0.2, 0.1]), np.array([0.5, 2.0])
    e, pw, z = pc.prediction_error(y, y_hat), pc.precision_weighted_error(y, y_hat, v), \
        pc.standardized_error(y, y_hat, v)
    assert np.allclose(pw, e / v) and np.allclose(z, e / np.sqrt(v))
    assert np.allclose(pw, z / np.sqrt(v))              # the two differ by exactly 1/sqrt(v)


def test_update_accepts_scalar_and_matching_vector():
    a, b = ScalarGaussianBelief(0.0, 1.0, 0.5, 3), ScalarGaussianBelief(0.0, 1.0, 0.5, 3)
    a.update(0.7)
    b.update(np.full(3, 0.7))
    assert np.array_equal(a.mu, b.mu)


@pytest.mark.parametrize("bad", [
    np.zeros((3, 1)), np.zeros((1, 3)), np.zeros(2), np.zeros(4), np.zeros((3, 3)),
    np.zeros(0), [[1.0, 2.0, 3.0]], "abc", None,
])
def test_update_rejects_incompatible_shapes_without_changing_belief(bad):
    b = ScalarGaussianBelief(0.2, 1.0, 0.5, 3)
    mu, prec = b.mu.copy(), b.prec.copy()
    with pytest.raises(ValueError):
        b.update(bad)
    assert np.array_equal(b.mu, mu) and np.array_equal(b.prec, prec)


@pytest.mark.parametrize("bad", [
    np.array([1.0, np.nan, 0.0]), np.array([np.inf, 0.0, 0.0]), np.array([0.0, 0.0, -np.inf]),
    np.nan, np.inf,
])
def test_update_rejects_nonfinite_without_changing_belief(bad):
    b = ScalarGaussianBelief(0.2, 1.0, 0.5, 3)
    mu, prec = b.mu.copy(), b.prec.copy()
    with pytest.raises(ValueError):
        b.update(bad)
    assert np.array_equal(b.mu, mu) and np.array_equal(b.prec, prec)


@pytest.mark.parametrize("kw", [dict(s0=0.0), dict(sy=-1.0), dict(sy=np.nan), dict(mu0=np.inf),
                                dict(n_trials=0), dict(n_trials=2.5), dict(sy=np.ones(2))])
def test_constructor_rejects_invalid_parameters(kw):
    with pytest.raises(ValueError):
        ScalarGaussianBelief(**{**dict(mu0=0.0, s0=1.0, sy=0.5, n_trials=3), **kw})
