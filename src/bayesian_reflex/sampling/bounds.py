"""Gaussian helpers: an unnormalised Gaussian log density and RIGOROUS shell bounds for it.

The paper's p_i = inf pi / sup pi needs the infimum and supremum of the density on each shell; in general they are not
available and are estimated by Monte Carlo (SOURCE sec. 3.1). For a Gaussian target they can be bounded exactly, which
lets tests separate "the algorithm is right" from "the Monte Carlo bounds are slightly optimistic".
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import chi2


def gaussian_log_density(mean, cov):
    """theta (n, d) -> -0.5 (theta - mean)^T cov^{-1} (theta - mean), i.e. log of the UNNORMALISED density."""
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    chol = np.linalg.cholesky(np.atleast_2d(np.asarray(cov, dtype=float)))

    def log_density(theta):
        y = solve_triangular(chol, (np.atleast_2d(theta) - mean).T, lower=True)
        return -0.5 * np.sum(y ** 2, axis=0)

    return log_density


def gaussian_shell_log_bounds(partition, mean, cov):
    """Callable i -> (log s_i, log S_i) bounding log of exp(-q/2), q = (theta-mean)^T cov^{-1} (theta-mean), on shell i.

    In the partition's whitened coordinates u (theta = mu + L u, shell: rho_{i-1} <= |u| <= rho_i) q = |B^{1/2}(u - w)|^2
    with B = L^T cov^{-1} L and w = L^{-1}(mean - mu). With beta_min/max the eigenvalues of B and
    delta = |B^{1/2} w|, the triangle inequality gives the rigorous bounds
        q <= (sqrt(beta_max) rho_i + delta)^2,
        q >= max(0, sqrt(beta_min) rho_{i-1} - delta, delta - sqrt(beta_max) rho_i)^2.
    They are exact when B is a multiple of the identity (target covariance proportional to the partition's), and
    otherwise valid but loose (a smaller p_hat, hence longer coalescence times, never a biased sampler).
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    L = partition.chol
    precision = cho_solve(cho_factor(cov, lower=True), np.eye(cov.shape[0]))
    B = L.T @ precision @ L
    beta = np.linalg.eigvalsh(0.5 * (B + B.T))
    w = solve_triangular(L, mean - partition.mean, lower=True)
    delta = float(np.sqrt(max(w @ B @ w, 0.0)))
    root_min, root_max = np.sqrt(beta[0]), np.sqrt(beta[-1])

    def log_bounds(i):
        r_in, r_out = partition.radius(i - 1), partition.radius(i)
        q_max = (root_max * r_out + delta) ** 2
        q_min = max(0.0, root_min * r_in - delta, delta - root_max * r_out) ** 2
        return -0.5 * q_max, -0.5 * q_min

    return log_bounds


def gaussian_log_tail_bound(partition, mean, cov):
    """Callable M -> log of a RIGOROUS upper bound on the integral of exp(-q/2) outside the first M shells.

    Outside the ellipsoid of radius rho_M the target's Mahalanobis distance satisfies sqrt(q) >= sqrt(beta_min) rho_M - delta
    (same notation as gaussian_shell_log_bounds), so the omitted mass is at most (2 pi)^(d/2) |cov|^(1/2) P(chi2_d >= t^2)
    with t = max(0, sqrt(beta_min) rho_M - delta). Exact up to the triangle inequality, which is tight when the target's
    covariance is proportional to the partition's.
    """
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = cov.shape[0]
    L = partition.chol
    precision = cho_solve(cho_factor(cov, lower=True), np.eye(d))
    B = L.T @ precision @ L
    beta_min = float(np.linalg.eigvalsh(0.5 * (B + B.T))[0])
    w = solve_triangular(L, mean - partition.mean, lower=True)
    delta = float(np.sqrt(max(w @ B @ w, 0.0)))
    log_norm = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(cov)[1]

    def log_tail_bound(M):
        t = max(0.0, np.sqrt(beta_min) * partition.radius(M) - delta)
        return float(log_norm + chi2.logsf(t * t, d))

    return log_tail_bound
