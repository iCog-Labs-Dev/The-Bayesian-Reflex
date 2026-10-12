"""Gaussian helpers: an unnormalised Gaussian log density and RIGOROUS shell bounds for it.

The paper's p_i = inf pi / sup pi needs the infimum and supremum of the density on each shell; in general they are not
available and are estimated by Monte Carlo (SOURCE sec. 3.1). For a Gaussian target they can be bounded exactly, which
lets tests separate "the algorithm is right" from "the Monte Carlo bounds are slightly optimistic".
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


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
