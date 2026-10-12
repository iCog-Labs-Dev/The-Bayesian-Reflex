"""Ellipsoidal decomposition i.i.d. sampler (The Bayesian Reflex, arXiv:2608.00492, sec. 4).

The paper states the method in its sec. 4 and attributes it to Bhattacharya, "IID Sampling from Intractable
Distributions" (arXiv:2107.05956, referred to below as the SOURCE). Where sec. 4 is silent the SOURCE is used, and
every place where either text is ambiguous or where this code deviates is listed in docs/ellipsoidal_sampler.md.

Mathematics -> code
-------------------
Shells (paper eq. 11)      A_i = {theta : c_{i-1} <= (theta-mu)^T Sigma^{-1} (theta-mu) <= c_i}, 0 = c_0 < c_1 < ...
                           -> EllipsoidalPartition (radii rho_i = sqrt(c_i) given by LinearRadii or any callable)
Mixture (eq. 12)           pi = sum_i pi(A_i) pi_i,  pi_i = pi 1_{A_i} / pi(A_i)
                           -> weights pi_hat(A_i) = L(A_i) * mean_U[pi_tilde], U uniform on A_i   (SOURCE eq. 3)
Minorization (eq. 13)      P_i(theta, .) >= p_i Q_i(.),  p_i = inf pi / sup pi on A_i, Q_i uniform on A_i
                           -> p_hat_i = s_hat_i / S_hat_i - eta  (min and max over the Monte Carlo draws, SOURCE sec. 3.1)
Split chain / perfect draw P_i = p Q_i + (1-p) R_i;  draw T, theta^(-T) ~ Q_i, T steps of R_i, return theta^(0)
                           -> EllipsoidalSampler._perfect_sample_shell
Overall sampler (sec. 4.3) pick shell i with probability proportional to pi_hat(A_i), then a perfect draw from pi_i;
                           if the last shell M is picked, double M (SOURCE Algorithm 1, step 6(ii))
                           -> EllipsoidalSampler.sample

Residual step. With a = min(1, pi(theta')/pi(theta)) for a uniform proposal theta', the MH step is "accept if U < a".
Because a >= p, conditioning on U >= p (an event of probability 1-p that is independent of theta') gives exactly the
residual kernel R: propose theta' ~ Q_i, draw U ~ Uniform(p, 1), move iff U < a. This equals the SOURCE's rejection
scheme (its eqs. 20-23) without needing a Monte Carlo estimate of the rejection probability.

Coalescence time. T counts the residual steps after the regeneration draw from Q_i, and must be supported on
{0, 1, 2, ...} with P(T=k) = p (1-p)^k. The SOURCE (eq. 24) states {1, 2, ...}; that form returns (pi_i - p Q_i)/(1-p),
not pi_i (verified analytically and numerically, see docs). ``geometric_support=0`` is the default and ``1`` reproduces
the stated form for comparison.
"""
import numbers

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp


class SamplerInfeasibleError(ValueError):
    """The sampler cannot run with these settings (e.g. p_hat_i <= 0 or a coalescence time beyond the safety cap)."""


def _check_int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (numbers.Integral, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_float(value, name, low=None, high=None, strict_low=False):
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if low is not None and (v <= low if strict_low else v < low):
        raise ValueError(f"{name} must be {'>' if strict_low else '>='} {low}, got {v}")
    if high is not None and v >= high:
        raise ValueError(f"{name} must be < {high}, got {v}")
    return v


class LinearRadii:
    """sqrt(c_i) = r1 + a (i - 1) for i >= 1, sqrt(c_0) = 0 (the radius rule used throughout the SOURCE, sec. 5)."""

    def __init__(self, r1, a):
        self.r1 = _check_float(r1, "r1", 0.0, strict_low=True)
        self.a = _check_float(a, "a", 0.0, strict_low=True)

    def __call__(self, i):
        i = _check_int(i, "shell index", 0)
        return 0.0 if i == 0 else self.r1 + self.a * (i - 1)


class EllipsoidalPartition:
    """Concentric ellipsoids {theta: (theta-mean)^T cov^{-1} (theta-mean) <= c_i} and the shells between them.

    Any mean and positive-definite cov give a valid partition (SOURCE sec. 2.2); good choices only affect efficiency.
    ``radii`` is a callable i -> sqrt(c_i) with radii(0) = 0, strictly increasing.
    """

    def __init__(self, mean, cov, radii):
        try:
            mean = np.asarray(mean, dtype=float)
            cov = np.asarray(cov, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("mean and cov must be real arrays") from exc
        if mean.ndim == 0:
            mean = mean.reshape(1)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError(f"mean must be a non-empty 1-D array, got shape {mean.shape}")
        d = mean.size
        if cov.ndim == 0 and d == 1:
            cov = cov.reshape(1, 1)
        if cov.shape != (d, d):
            raise ValueError(f"cov must have shape ({d}, {d}), got {cov.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            raise ValueError("mean and cov must be finite")
        if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-12):
            raise ValueError("cov must be symmetric")
        try:
            self.chol = np.linalg.cholesky(0.5 * (cov + cov.T))
        except np.linalg.LinAlgError as exc:
            raise ValueError("cov must be positive definite") from exc
        if not callable(radii):
            raise ValueError("radii must be callable: i -> sqrt(c_i)")
        self.mean, self.cov, self.dim, self.radii = mean, cov, d, radii
        self._log_det_chol = float(np.sum(np.log(np.diag(self.chol))))
        self._log_unit_ball = 0.5 * d * np.log(np.pi) - float(gammaln(0.5 * d + 1.0))

    def radius(self, i):
        i = _check_int(i, "shell index", 0)
        rho = _check_float(self.radii(i), f"radius({i})", 0.0)
        if i == 0 and rho != 0.0:
            raise ValueError("radii(0) must be 0")
        if i >= 1:
            prev = float(self.radii(i - 1))
            if not rho > prev:
                raise ValueError(f"radii must be strictly increasing: radius({i}) = {rho} <= radius({i - 1}) = {prev}")
        return rho

    def log_volume(self, i):
        """log Lebesgue measure of shell i (SOURCE sec. 2.3.2): |B| V_d (rho_i^d - rho_{i-1}^d)."""
        i = _check_int(i, "shell index", 1)
        d, rho, prev = self.dim, self.radius(i), self.radius(i - 1)
        base = self._log_det_chol + self._log_unit_ball + d * np.log(rho)
        return float(base if prev == 0.0 else base + np.log1p(-((prev / rho) ** d)))

    def sample_uniform(self, i, n, rng):
        """n i.i.d. uniform points on shell i, exactly (no rejection, valid for thin shells).

        In whitened coordinates y = B^{-1}(theta - mean) the shell is {rho_{i-1} <= |y| <= rho_i}; the direction is
        uniform on the sphere and |y|^d is uniform on [rho_{i-1}^d, rho_i^d] (radial inverse CDF, in log space).
        """
        i = _check_int(i, "shell index", 1)
        n = _check_int(n, "n", 0)
        d, rho, prev = self.dim, self.radius(i), self.radius(i - 1)
        z = rng.standard_normal((n, d))
        dirs = z / np.linalg.norm(z, axis=1, keepdims=True)
        u = rng.uniform(size=n)
        with np.errstate(divide="ignore"):
            term_out = d * np.log(rho) + np.log(u)
            term_in = (d * np.log(prev) + np.log1p(-u)) if prev > 0.0 else np.full(n, -np.inf)
        radius = np.exp(np.logaddexp(term_in, term_out) / d)
        return self.mean + (radius[:, None] * dirs) @ self.chol.T

    def mahalanobis_sq(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        y = solve_triangular(self.chol, (theta - self.mean).T, lower=True)
        return np.sum(y ** 2, axis=0)


class EllipsoidalSampler:
    """I.i.d. sampler for an (unnormalised) density on R^d via the ellipsoidal decomposition.

    Parameters
    ----------
    log_density : callable, (n, d) array -> (n,) array of log pi_tilde (unnormalised log density, -inf allowed).
    partition   : EllipsoidalPartition.
    n_mc        : Monte Carlo size N_i per shell for the weights and for s_hat, S_hat (SOURCE: 10,000).
    eta         : the SOURCE's eta_i in p_hat_i = s_hat_i/S_hat_i - eta_i (SOURCE: 1e-5; its Challenger example uses 0).
    geometric_support : 0 -> T in {0,1,...} (default, exact); 1 -> {1,2,...} as written in the SOURCE eq. (24).
    log_bounds  : optional callable i -> (log s_i, log S_i), rigorous bounds on log pi_tilde over shell i. When given
                  they replace the Monte Carlo min/max and no eta is subtracted (minorization is then guaranteed).
    initial_shells, max_shells : M starts at ``initial_shells`` and doubles when the last shell is selected.
    min_p       : a selected shell needs p_hat > min_p, otherwise SamplerInfeasibleError is raised (shells that are never
                  selected may have any p_hat, as in the SOURCE where p_hat_i is only used when pi_i is drawn).
    adaptive_shells : False keeps M fixed (the tail beyond shell M is silently dropped; used only as a negative control).
    max_coalescence : safety cap on the drawn coalescence time; exceeding it raises instead of truncating.
    """

    def __init__(self, log_density, partition, n_mc=10_000, eta=1e-5, geometric_support=0, log_bounds=None,
                 initial_shells=16, max_shells=1_000_000, max_coalescence=1_000_000, adaptive_shells=True,
                 min_p=1e-9):
        if not callable(log_density):
            raise ValueError("log_density must be callable")
        if not isinstance(partition, EllipsoidalPartition):
            raise ValueError("partition must be an EllipsoidalPartition")
        if log_bounds is not None and not callable(log_bounds):
            raise ValueError("log_bounds must be callable or None")
        if geometric_support not in (0, 1):
            raise ValueError("geometric_support must be 0 or 1")
        self.log_density, self.partition, self.log_bounds = log_density, partition, log_bounds
        self.n_mc = _check_int(n_mc, "n_mc", 2)
        self.eta = _check_float(eta, "eta", 0.0, 1.0)
        self.geometric_support = int(geometric_support)
        self.max_shells = _check_int(max_shells, "max_shells", 1)
        self._M = min(_check_int(initial_shells, "initial_shells", 1), self.max_shells)
        self.max_coalescence = _check_int(max_coalescence, "max_coalescence", 1)
        self.adaptive_shells = bool(adaptive_shells)
        self.min_p = _check_float(min_p, "min_p", 0.0, strict_low=True)
        self._shells = []                       # per shell: dict(log_weight, log_s, log_S, p_hat)
        self._steps_total = self._steps_max = self._draws = self._violations = self._residual_steps = 0
        self._shell_counts = {}

    # ---------------------------------------------------------------- density evaluation
    def _eval(self, theta):
        lp = np.asarray(self.log_density(theta), dtype=float)
        if lp.shape != (len(theta),):
            raise ValueError(f"log_density must return shape ({len(theta)},), got {lp.shape}")
        if np.any(np.isnan(lp)) or np.any(lp == np.inf):
            raise ValueError("log_density returned NaN or +inf")
        return lp

    # ---------------------------------------------------------------- per-shell Monte Carlo estimates
    def _ensure_shell(self, i, rng):
        while len(self._shells) < i:
            k = len(self._shells) + 1
            theta = self.partition.sample_uniform(k, self.n_mc, rng)
            lp = self._eval(theta)
            log_weight = self.partition.log_volume(k) + float(logsumexp(lp)) - np.log(self.n_mc)
            if self.log_bounds is None:
                log_s, log_S = float(lp.min()), float(lp.max())
                p_hat = np.exp(log_s - log_S) - self.eta if np.isfinite(log_S) else 0.0
            else:
                log_s, log_S = (float(v) for v in self.log_bounds(k))
                if not (log_s <= log_S and np.isfinite(log_S) and log_s < np.inf):
                    raise ValueError(f"log_bounds({k}) must satisfy log_s <= log_S < inf, got ({log_s}, {log_S})")
                p_hat = float(np.exp(log_s - log_S))
            # p_hat is only REQUIRED for shells that are actually drawn from (checked in _perfect_sample_shell); far-tail
            # shells with negligible weight may have p_hat <= 0 without affecting the sampler.
            self._shells.append(dict(log_weight=log_weight, log_s=log_s, log_S=log_S, p_hat=min(p_hat, 1.0 - 1e-12)))

    def _weights(self, M):
        lw = np.array([s["log_weight"] for s in self._shells[:M]])
        w = np.exp(lw - logsumexp(lw))
        return w / w.sum()

    # ---------------------------------------------------------------- shell selection (Algorithm 1, step 6(i)-(ii))
    def _select_shells(self, u, rng):
        M = self._M
        while True:
            self._ensure_shell(M, rng)
            cum = np.cumsum(self._weights(M))
            cum[-1] = 1.0
            idx = np.minimum(np.searchsorted(cum, u, side="right"), M - 1)
            if not self.adaptive_shells or idx.max() < M - 1:
                break
            if 2 * M > self.max_shells:
                raise SamplerInfeasibleError(
                    f"the last shell was selected with M = {M} and max_shells = {self.max_shells} is reached")
            M *= 2
        self._M = M
        return idx + 1                                    # shell numbers are 1-based, as in the paper

    # ---------------------------------------------------------------- perfect sampling on one shell
    def _perfect_sample_shell(self, i, n, rng):
        self._ensure_shell(i, rng)
        p = self._shells[i - 1]["p_hat"]
        if n and not p > self.min_p:
            raise SamplerInfeasibleError(
                f"p_hat_{i} = {p:.3g} <= {self.min_p:g} but shell {i} was selected: the density varies too much on "
                f"shell {i} for perfect sampling; use smaller radii (or a flatter target)")
        T = rng.geometric(p, n) - (1 - self.geometric_support)       # numpy: support {1,2,...}; shift for {0,1,...}
        if n and T.max() > self.max_coalescence:
            raise SamplerInfeasibleError(
                f"coalescence time {int(T.max())} exceeds max_coalescence = {self.max_coalescence} on shell {i} "
                f"(p_hat = {p:.3g})")
        theta = self.partition.sample_uniform(i, n, rng)             # theta^(-T) ~ Q_i
        lp = self._eval(theta)
        order = np.argsort(-T, kind="stable")
        neg_T_sorted = -T[order]
        for step in range(int(T.max()) if n else 0):                 # T residual steps, only for rows that need them
            n_act = int(np.searchsorted(neg_T_sorted, -step, side="left"))
            rows = order[:n_act]
            prop = self.partition.sample_uniform(i, n_act, rng)
            lp_prop = self._eval(prop)
            with np.errstate(invalid="ignore"):
                log_a = np.minimum(0.0, lp_prop - lp[rows])
            if not np.all(np.isfinite(lp[rows])):
                raise SamplerInfeasibleError("a chain state has zero density; check log_density and the bounds")
            a = np.exp(log_a)
            self._violations += int(np.sum(a < p - 1e-12))
            u = rng.uniform(p, 1.0, n_act)                           # conditioning on "no regeneration": U >= p
            move = u < a
            theta[rows[move]], lp[rows[move]] = prop[move], lp_prop[move]
            self._residual_steps += n_act
        self._draws += n
        self._steps_total += int(T.sum())
        self._steps_max = max(self._steps_max, int(T.max()) if n else 0)
        self._shell_counts[i] = self._shell_counts.get(i, 0) + n
        return theta

    # ---------------------------------------------------------------- public API
    def sample_from_shells(self, shells, rng):
        """One perfect draw from pi_i for each (1-based) shell number in ``shells``; returns an (n, d) array."""
        shells = np.asarray(shells)
        if shells.ndim != 1 or not np.issubdtype(shells.dtype, np.integer) or np.any(shells < 1):
            raise ValueError("shells must be a 1-D integer array of shell numbers >= 1")
        out = np.empty((len(shells), self.partition.dim))
        for i in np.unique(shells):
            rows = np.flatnonzero(shells == i)
            out[rows] = self._perfect_sample_shell(int(i), len(rows), rng)
        return out

    def sample(self, n, rng):
        """n i.i.d. draws from pi_hat (paper sec. 4.3); returns an (n, d) array."""
        n = _check_int(n, "n", 1)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        shells = self._select_shells(rng.uniform(size=n), rng)
        return self.sample_from_shells(shells, rng)

    @property
    def diagnostics(self):
        M = min(self._M, len(self._shells))
        return dict(
            n_shells=self._M,
            p_hat=np.array([s["p_hat"] for s in self._shells[:M]]),
            last_shell_weight=float(self._weights(M)[-1]) if M else float("nan"),
            n_draws=self._draws,
            mean_coalescence_steps=self._steps_total / self._draws if self._draws else float("nan"),
            max_coalescence_steps=self._steps_max,
            n_residual_steps=self._residual_steps,
            n_minorization_violations=self._violations,
            shell_counts=dict(sorted(self._shell_counts.items())),
        )
