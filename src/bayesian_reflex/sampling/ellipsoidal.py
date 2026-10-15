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
                           -> mode "monte_carlo": p_hat_i = s_hat_i / S_hat_i - eta (min and max over the Monte Carlo draws,
                              SOURCE sec. 3.1); mode "rigorous": caller-supplied bounds
Split chain / perfect draw P_i = p Q_i + (1-p) R_i;  draw T, theta^(-T) ~ Q_i, T steps of R_i, return theta^(0)
                           -> EllipsoidalSampler._perfect_sample_shell
Overall sampler (sec. 4.3) pick shell i with probability proportional to pi_hat(A_i), then a perfect draw from pi_i
                           -> EllipsoidalSampler.prepare (fix the shells) and EllipsoidalSampler.sample

What is exact and what is approximate
-------------------------------------
Conditional on the prepared state (the shells 1..M, the estimated weights and the p_hat_i), the draws returned by
``sample`` are i.i.d. from the mixture pi_hat_M = sum_{i<=M} w_i pi_i, and the draw inside a shell is EXACT from pi_i if and
only if the minorization holds for the p_hat_i that is used. The result differs from the target pi by
    TV(pi_hat_M, pi) = 1/2 sum_{i<=M} |pi(A_i) - w_i| + tau/2,    tau = pi(outside shell M)   (identity: shells are disjoint),
i.e. by the weight estimation error (Monte Carlo, O(N_i^{-1/2})) and the omitted tail mass. Therefore:
  * mode "rigorous" (caller supplies log_bounds AND log_tail_bound): the perfect-sampling guarantee holds (a violation raises),
    the omitted mass is bounded by tail_tol relative to the retained mass; only the weight error remains approximate and is
    reported as an estimate in ``diagnostics['tv_weight_error_estimate']``.
  * mode "monte_carlo" (the paper's practical mode): p_hat comes from the sampled min and max, which are optimistic, so NO
    guarantee is claimed; a detected violation raises (``on_violation='count'`` only counts them), and the omitted mass is an
    ESTIMATE (geometric extrapolation of the last shell weights) unless a tail bound is supplied.
The shells are fixed by ``prepare`` BEFORE any returned draw is made and never change afterwards, so ``sample(1)``, repeated
calls and one bulk call draw from the same distribution (the paper's Algorithm 1 doubles M when a returned draw lands in the last
shell, which couples M to the returned uniforms; that coupling is removed here, see docs/ellipsoidal_sampler.md).

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


class MinorizationViolationError(SamplerInfeasibleError):
    """A step violated the minorization P >= p_hat Q (an acceptance ratio below p_hat, or a density outside the supplied
    bounds): the bounds are invalid, so the perfect-sampling guarantee does not hold."""


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
    mode        : "monte_carlo" (default; the paper's practical mode, NO perfect-sampling guarantee) or "rigorous"
                  (needs ``log_bounds`` and ``log_tail_bound``; a violation raises).
    n_mc        : Monte Carlo size N_i per shell for the weights (and, in monte_carlo mode, for s_hat, S_hat). SOURCE: 10,000.
    eta         : monte_carlo mode only: p_hat_i = s_hat_i/S_hat_i - eta (SOURCE: 1e-5).
    geometric_support : 0 -> T in {0,1,...} (default, exact); 1 -> {1,2,...} as written in the SOURCE eq. (24).
    log_bounds  : rigorous mode: callable i -> (log s_i, log S_i), valid bounds on log pi_tilde over shell i.
    log_tail_bound : callable M -> log of an upper bound on the integral of pi_tilde OUTSIDE the first M shells (same units as
                  the weights). Required in rigorous mode; optional in monte_carlo mode.
    tail_tol    : shell expansion stops when (bound or estimate of the omitted mass) / (retained mass) <= tail_tol.
    n_shells    : fix the number of shells and skip the coverage criterion (truncation; for controls and experts only).
    initial_shells, max_shells, patience : coverage search starts at ``initial_shells`` shells, doubles up to ``max_shells``;
                  without a tail bound the estimate rule also needs the last ``patience`` weight ratios to be below 1.
    on_violation : "raise" (default) or "count" (monte_carlo mode only): what to do when an acceptance ratio a falls below p_hat.
    violation_tol : monte_carlo mode: a violation raises only if its relative shortfall (p_hat - a)/p_hat exceeds this (default
                  1e-2); smaller ones are counted and the worst is reported as ``max_violation_shortfall`` (estimated bounds are
                  optimistic by about the Monte Carlo spacing, so tiny shortfalls are expected). 0 = raise on any. In rigorous
                  mode it is 0: any violation, or any density value outside the bounds, raises.
    max_coalescence : safety cap on the drawn coalescence time; exceeding it raises instead of truncating.
    min_p       : a selected shell needs p_hat > min_p, otherwise SamplerInfeasibleError is raised.
    """

    def __init__(self, log_density, partition, mode="monte_carlo", n_mc=10_000, eta=1e-5, geometric_support=0,
                 log_bounds=None, log_tail_bound=None, tail_tol=1e-6, n_shells=None, initial_shells=16,
                 max_shells=1_000_000, max_coalescence=1_000_000, on_violation="raise", violation_tol=None, min_p=1e-9, patience=3):
        if not callable(log_density):
            raise ValueError("log_density must be callable")
        if not isinstance(partition, EllipsoidalPartition):
            raise ValueError("partition must be an EllipsoidalPartition")
        if mode not in ("monte_carlo", "rigorous"):
            raise ValueError("mode must be 'monte_carlo' or 'rigorous'")
        for name, fn in (("log_bounds", log_bounds), ("log_tail_bound", log_tail_bound)):
            if fn is not None and not callable(fn):
                raise ValueError(f"{name} must be callable or None")
        if mode == "rigorous" and (log_bounds is None or log_tail_bound is None):
            raise ValueError("mode='rigorous' requires both log_bounds and log_tail_bound")
        if mode == "monte_carlo" and log_bounds is not None:
            raise ValueError("log_bounds are rigorous bounds: pass mode='rigorous' (monte_carlo mode estimates them)")
        if on_violation not in ("raise", "count"):
            raise ValueError("on_violation must be 'raise' or 'count'")
        if mode == "rigorous" and on_violation != "raise":
            raise ValueError("mode='rigorous' always raises on a violation")
        if mode == "rigorous" and violation_tol not in (None, 0, 0.0):
            raise ValueError("mode='rigorous' tolerates no violation (violation_tol must be 0)")
        if geometric_support not in (0, 1):
            raise ValueError("geometric_support must be 0 or 1")
        self.log_density, self.partition, self.mode = log_density, partition, mode
        self.log_bounds, self.log_tail_bound, self.on_violation = log_bounds, log_tail_bound, on_violation
        self.violation_tol = 0.0 if mode == "rigorous" else _check_float(1e-2 if violation_tol is None else violation_tol,
                                                                         "violation_tol", 0.0)
        self.n_mc = _check_int(n_mc, "n_mc", 2)
        self.eta = _check_float(eta, "eta", 0.0, 1.0)
        self.geometric_support = int(geometric_support)
        self.tail_tol = _check_float(tail_tol, "tail_tol", 0.0, 1.0, strict_low=True)
        self.max_shells = _check_int(max_shells, "max_shells", 1)
        self.patience = _check_int(patience, "patience", 1)
        self.n_shells_fixed = None if n_shells is None else min(_check_int(n_shells, "n_shells", 1), self.max_shells)
        self._M = max(_check_int(initial_shells, "initial_shells", 1), self.patience + 1)
        self._M = min(self._M, self.max_shells)
        self.max_coalescence = _check_int(max_coalescence, "max_coalescence", 1)
        self.min_p = _check_float(min_p, "min_p", 0.0, strict_low=True)
        self._shells = []                       # per shell: dict(log_weight, cv, log_s, log_S, p_hat)
        self._prepared, self._cum, self._coverage = False, None, {}
        self._steps_total = self._steps_max = self._draws = self._violations = self._residual_steps = 0
        self._max_shortfall = 0.0
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
            lse1 = float(logsumexp(lp))
            log_weight = self.partition.log_volume(k) + lse1 - np.log(self.n_mc)
            # relative standard deviation of pi_tilde over the shell: sqrt(E[pi^2]/E[pi]^2 - 1), in log space
            cv = float(np.sqrt(max(np.exp(float(logsumexp(2.0 * lp)) - 2.0 * lse1 + np.log(self.n_mc)) - 1.0, 0.0))) \
                if np.isfinite(lse1) else 0.0
            if self.mode == "monte_carlo":
                log_s, log_S = float(lp.min()), float(lp.max())
                p_hat = np.exp(log_s - log_S) - self.eta if np.isfinite(log_S) else 0.0
            else:
                log_s, log_S = (float(v) for v in self.log_bounds(k))
                if not (log_s <= log_S and np.isfinite(log_S) and log_s < np.inf):
                    raise ValueError(f"log_bounds({k}) must satisfy log_s <= log_S < inf, got ({log_s}, {log_S})")
                p_hat = float(np.exp(log_s - log_S))
            # p_hat is only REQUIRED for shells that are actually drawn from (checked in _perfect_sample_shell); far-tail
            # shells with negligible weight may have p_hat <= 0 without affecting the sampler.
            self._shells.append(dict(log_weight=log_weight, cv=cv, log_s=log_s, log_S=log_S,
                                     p_hat=min(p_hat, 1.0 - 1e-12)))

    def _weights(self, M):
        lw = np.array([s["log_weight"] for s in self._shells[:M]])
        w = np.exp(lw - logsumexp(lw))
        return w / w.sum()

    # ---------------------------------------------------------------- shell coverage (fixed before any draw is returned)
    def _coverage_check(self, M):
        """Return (stop, info): is the omitted mass beyond shell M small enough, by bound or by estimate?"""
        lw = np.array([s["log_weight"] for s in self._shells[:M]])
        log_retained = float(logsumexp(lw))
        info = dict(omitted_mass_bound=None, omitted_mass_estimate=None)
        ratios_ok = False
        if M >= 2 and np.isfinite(lw[-1]) and np.isfinite(lw[-2]):             # geometric extrapolation of the tail
            last = lw[-(self.patience + 1):]
            r = np.exp(np.diff(last))
            ratios_ok = bool(len(r) == self.patience and np.all(r < 1.0))
            if r[-1] < 1.0:
                log_tau = lw[-1] + np.log(r[-1]) - np.log1p(-r[-1])
                info["omitted_mass_estimate"] = float(np.exp(min(log_tau - log_retained, 700.0)))
        elif not np.isfinite(lw[-1]):                                          # no density at all beyond this point
            info["omitted_mass_estimate"], ratios_ok = 0.0, True
        if self.log_tail_bound is not None:
            frac = float(np.exp(min(float(self.log_tail_bound(M)) - log_retained, 700.0)))
            info["omitted_mass_bound"] = frac
            return frac <= self.tail_tol, info
        est = info["omitted_mass_estimate"]
        return bool(ratios_ok and est is not None and est <= self.tail_tol), info

    def prepare(self, rng):
        """Fix the shells (and so the mixture weights) once, independently of any returned draw. Idempotent.

        With ``n_shells`` the number of shells is given and no coverage criterion is applied. Otherwise M starts at
        ``initial_shells`` and doubles until the omitted-mass bound (``log_tail_bound``) or, without a bound, the
        omitted-mass ESTIMATE is at most ``tail_tol`` of the retained mass.
        """
        if self._prepared:
            return self
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if self.n_shells_fixed is not None:
            M = self.n_shells_fixed
            self._ensure_shell(M, rng)
            info = dict(omitted_mass_bound=None, omitted_mass_estimate=None)
        else:
            M = self._M
            while True:
                self._ensure_shell(M, rng)
                stop, info = self._coverage_check(M)
                if stop:
                    break
                if 2 * M > self.max_shells:
                    raise SamplerInfeasibleError(
                        f"the omitted mass is not yet below tail_tol = {self.tail_tol:g} with M = {M} shells and "
                        f"max_shells = {self.max_shells} is reached")
                M *= 2
        self._M, self._coverage, self._prepared = M, info, True
        cum = np.cumsum(self._weights(M))
        cum[-1] = 1.0
        self._cum = cum
        return self

    # ---------------------------------------------------------------- shell selection (the shells are already fixed)
    def _select_shells(self, u):
        idx = np.minimum(np.searchsorted(self._cum, u, side="right"), self._M - 1)
        return idx + 1                                    # shell numbers are 1-based, as in the paper

    # ---------------------------------------------------------------- violation handling
    def _violation(self, i, detail, shortfall):
        """Record a violation of P >= p_hat Q; ``shortfall`` = (p_hat - a) / p_hat >= 0 (or inf for a bound violation).

        Raises when on_violation == 'raise' and the shortfall exceeds ``violation_tol`` (0 in rigorous mode)."""
        self._violations += 1
        self._max_shortfall = max(self._max_shortfall, float(shortfall))
        if self.on_violation == "raise" and shortfall > self.violation_tol:
            raise MinorizationViolationError(
                f"minorization violated on shell {i} ({detail}; relative shortfall {shortfall:.3g} > violation_tol = "
                f"{self.violation_tol:g}); mode = {self.mode!r}: the bounds are invalid, so the perfect-sampling guarantee "
                "does not hold. Supply valid bounds, increase n_mc or eta (monte_carlo mode), or use "
                "on_violation='count' to only count violations (no guarantee)")

    def _check_range(self, i, lp):
        """rigorous mode: every density value must lie inside the supplied bounds."""
        sh = self._shells[i - 1]
        tol = 1e-9 * (1.0 + abs(sh["log_S"]))
        if lp.min() < sh["log_s"] - tol or lp.max() > sh["log_S"] + tol:
            self._violation(i, f"log density {lp.min():.6g}..{lp.max():.6g} outside the bounds "
                               f"[{sh['log_s']:.6g}, {sh['log_S']:.6g}]", np.inf)

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
        if self.mode == "rigorous" and n:
            self._check_range(i, lp)
        order = np.argsort(-T, kind="stable")
        neg_T_sorted = -T[order]
        for step in range(int(T.max()) if n else 0):                 # T residual steps, only for rows that need them
            n_act = int(np.searchsorted(neg_T_sorted, -step, side="left"))
            rows = order[:n_act]
            prop = self.partition.sample_uniform(i, n_act, rng)
            lp_prop = self._eval(prop)
            if self.mode == "rigorous":
                self._check_range(i, lp_prop)
            if not np.all(np.isfinite(lp[rows])):
                raise SamplerInfeasibleError("a chain state has zero density; check log_density and the bounds")
            with np.errstate(invalid="ignore"):
                a = np.exp(np.minimum(0.0, lp_prop - lp[rows]))
            bad = a < p - 1e-12
            if bad.any():
                worst = float(a[bad].min())
                self._violation(i, f"acceptance ratio {worst:.6g} < p_hat = {p:.6g}", (p - worst) / p)
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
        """n draws, i.i.d. conditional on the prepared state, from pi_hat_M (paper sec. 4.3); returns an (n, d) array.

        The first call prepares the shells (``prepare``) before any uniform is used for the returned draws.
        """
        n = _check_int(n, "n", 1)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        self.prepare(rng)
        return self.sample_from_shells(self._select_shells(rng.uniform(size=n)), rng)

    @property
    def diagnostics(self):
        M = min(self._M, len(self._shells))
        w = self._weights(M) if M else np.array([])
        cv = np.array([s["cv"] for s in self._shells[:M]])
        return dict(
            mode=self.mode,
            perfect_sampling_guaranteed=self.mode == "rigorous",     # needs valid bounds; a violation raises
            n_shells=self._M,
            shells_fixed=self.n_shells_fixed is not None,
            p_hat=np.array([s["p_hat"] for s in self._shells[:M]]),
            last_shell_weight=float(w[-1]) if M else float("nan"),
            omitted_mass_bound=self._coverage.get("omitted_mass_bound"),
            omitted_mass_estimate=self._coverage.get("omitted_mass_estimate"),
            tail_bound_rigorous=self.log_tail_bound is not None,
            tv_weight_error_estimate=float(0.5 * np.sum(w * cv) / np.sqrt(self.n_mc)) if M else float("nan"),
            n_draws=self._draws,
            mean_coalescence_steps=self._steps_total / self._draws if self._draws else float("nan"),
            max_coalescence_steps=self._steps_max,
            n_residual_steps=self._residual_steps,
            n_minorization_violations=self._violations,
            max_violation_shortfall=self._max_shortfall,
            shell_counts=dict(sorted(self._shell_counts.items())),
        )
