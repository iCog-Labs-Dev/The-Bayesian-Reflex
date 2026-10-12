# Ellipsoidal decomposition sampler (paper sec. 4)

Scope: this PR implements the one algorithmic component of the paper that is missing after the foundations PR and that the
paper's later sections depend on: **exact i.i.d. sampling from a posterior by ellipsoidal decomposition** (sec. 4, eqs. 11-13).
It does not implement the reflex loop, Thompson sampling (sec. 3.3, 6.2), recursive Gaussian processes (sec. 5),
derivative-aware optimisation (sec. 6), or anything from the function-space / PC-FSVI line of work, and it makes no comparison
with other inference methods. Nothing in the merged foundations code was modified.

Sources. Sec. 4 of the paper states the method in a page and attributes it to Bhattacharya, "IID Sampling from Intractable
Distributions" (arXiv:2107.05956, called the SOURCE below). Where sec. 4 leaves a detail open, the SOURCE was read and used; each such
place is listed in "Ambiguities and discrepancies" so it can be approved or changed.

## 1. Continuity with the merged foundations PR (#1)

| Merged in PR #1 | Tests | Used here |
|---|---|---|
| `ScalarGaussianBelief`: exact conjugate update (paper eq. 9, Gaussian case) | E1 33/33, E2a 9/9 | exact oracle for the posterior the sampler must reproduce, and source of the partition's (mu, sigma^2) |
| `GenerativeModel` interface (`update`, `predict`, `sample(n, rng)`); its docstring names the ellipsoidal sampler as a later stage | `tests/test_core.py` (27 tests) | the sampler's `sample(n, rng)` follows the same call shape; returns `(n, d)` |
| `PredictiveCodingLayer`, `BayesianReflex` (prediction error is diagnostic only) | E3 41/41 | untouched |
| Method rules: criteria fixed before running, negative controls, 10 seeds, simulation-based calibration, exit status | `experiments/e1`-`e3` | E4 below follows them; SBC is reused with the sampler as posterior sampler |

The next missing component is the sampler: the paper says belief updates and Thompson sampling are exact *because* the posterior
can be sampled exactly (sec. 4.4, 5.5, 6.2), and the merged code only has a conjugate case where sampling is trivial.

## 2. The method and why it works

Let pi_tilde be an unnormalised density on R^d. Choose mu, Sigma and radii 0 = rho_0 < rho_1 < ... (rho_i = sqrt(c_i)). The shells are
A_i = {theta : rho_{i-1}^2 <= (theta - mu)^T Sigma^{-1} (theta - mu) <= rho_i^2} (eq. 11) and pi = sum_i pi(A_i) pi_i (eq. 12).

**Weights.** pi(A_i) is proportional to L(A_i) E_U[pi_tilde], U uniform on A_i, with L(A_i) = |B| V_d (rho_i^d - rho_{i-1}^d), B = chol(Sigma)
(SOURCE eq. 3). Only relative weights are needed, so the unnormalised density suffices. They are estimated once per shell from N_i uniform draws.

**Perfect draw from pi_i.** Run Metropolis-Hastings on A_i with uniform independence proposals Q_i. The acceptance a = min(1, pi(theta')/pi(theta))
is at least p_i = inf pi / sup pi, so P_i >= p_i Q_i (eq. 13) and P_i = p_i Q_i + (1 - p_i) R_i with a residual kernel R_i.
In a stationary chain the last regeneration (a Q_i draw) before time 0 happened T steps earlier, with P(T = k) = p (1 - p)^k, k = 0, 1, 2, ...;
after it the chain used R_i only. Hence theta^(0) = (Q_i draw, then T residual steps) has law sum_k p (1 - p)^k Q_i R_i^k = pi_i.

**Residual step without rejection loops.** Because a >= p, "accept iff U < a" with U uniform on (0,1) can be split into the event U < p (probability p,
always accepted, new state ~ Q_i: the regeneration) and its complement. Conditioning on U >= p gives exactly R_i:
propose theta' ~ Q_i, draw U ~ Uniform(p, 1), move iff U < a. The SOURCE reaches the same kernel by a rejection scheme (its eqs. 20-23) that needs a Monte Carlo
estimate of the rejection probability; the two agree (the stay probability is r/(1-p) in both), and the form used here needs no such estimate.

**Overall sampler (sec. 4.3).** Pick shell i with probability proportional to the estimated weights, then draw perfectly from pi_i. If the last of the M shells is picked,
M is doubled and the check repeated with the same uniforms (SOURCE Algorithm 1, step 6(ii)); this keeps the omitted tail negligible.

## 3. Implementation map

| Paper / SOURCE | Code (`src/bayesian_reflex/sampling/`) |
|---|---|
| shells, eq. 11; radii sqrt(c_1) = r, sqrt(c_i) = r + a(i-1) (SOURCE sec. 5) | `EllipsoidalPartition`, `LinearRadii` |
| shell volume (SOURCE 2.3.2) and uniform sampling on a shell (SOURCE 2.3.1) | `log_volume`, `sample_uniform` (exact radial inverse CDF in log space) |
| mixture weights, eq. 12, SOURCE eq. 3; p_hat_i = s_hat_i / S_hat_i - eta (SOURCE 3.1) | `EllipsoidalSampler._ensure_shell` |
| split chain, perfect draw (eq. 13, SOURCE 3.2-3.3) | `EllipsoidalSampler._perfect_sample_shell` |
| overall i.i.d. sampler (sec. 4.3, SOURCE Algorithm 1) | `EllipsoidalSampler.sample`, `sample_from_shells` |
| exact shell bounds for Gaussian targets (not in the paper; used only to separate "algorithm" from "Monte Carlo bounds") | `gaussian_log_density`, `gaussian_shell_log_bounds` |

## 4. Ambiguities and discrepancies, and what was done

Each item is either a place where the paper is silent or where the paper (or its SOURCE) disagrees with the maths or with itself. None was resolved silently;
items 1 and 2 change results and need a decision.

1. **Support of the geometric coalescence time (discrepancy).** Sec. 4.2 says only "T_i ~ Geometric(p_i)". The SOURCE (eq. 24) states P(T = t) = p (1-p)^(t-1) for t = 1, 2, ... and says the form with t = 0, 1, ... "failed".
   With T >= 1 the output law is pi_i R = (pi_i - p Q_i)/(1 - p), not pi_i, because the regeneration-at-once event (probability p) is dropped. Checked numerically on a shell with known answer (p = 0.458, 200,000 draws per run):
   support {0,1,...}: KS p = 0.29, E|x| = 1.2621 (exact 1.2618); support {1,2,...}: KS p ~ 0, E|x| = 1.2292, matching the predicted (1.2618 - 0.458 x 1.3)/(1 - 0.458) = 1.2292. A uniform draw with no residual steps also fails.
   **Default here: {0,1,...}**; `geometric_support=1` reproduces the SOURCE's stated form. It is possible the SOURCE's code indexes time differently from its text; its text could not be tested against its code.
2. **p_i = inf pi / sup pi is estimated, not bounded (exactness is conditional).** The SOURCE replaces inf and sup by the min and max of pi_tilde over the Monte Carlo draws and subtracts eta (1e-5; 0 in one example).
   Those estimates are optimistic (min too large, max too small), so the minorization P >= p Q can fail near the extremes and the residual step is then clamped (biased by the violated mass). Handling here: default as the SOURCE;
   violations are **counted** (`n_minorization_violations`; in E4: 0 in all Gaussian runs, 2 in 450,779 residual steps for Student-t(5)); and rigorous bounds can be supplied through `log_bounds` (provided for Gaussian targets, exact when the partition has the target's shape).
   For general targets no rigorous bound is known; the paper does not give one.
3. **Radii and M.** Only the family sqrt(c_i) = r + a(i-1) with experiment-specific (r, a, M) is given; there is no rule. They are required arguments (no defaults). The doubling rule for M is implemented as in Algorithm 1.
4. **mu and Sigma.** The paper says "e.g. posterior mode or mean" and "an estimate of the posterior covariance"; the SOURCE obtains them from TMCMC runs. TMCMC is a different algorithm and is **not implemented**; mu and Sigma are arguments. Any choice is valid, but the shape matters for feasibility (see Limitations).
5. **Weight estimation error and the "optional rejection step" (sec. 4.3).** Draws are exact from pi_hat (the mixture with estimated weights), not from pi. Sec. 4.3 mentions an optional rejection correction with acceptance >= (1-eps)/(1+eps) but does not say what is accepted or rejected;
   the SOURCE argues (its sec. 2.4) that it accepts with probability ~1 and omits it from Algorithm 1. It is **not implemented**. The size of the error is O(N_i^(-1/2)) in the weights and is not corrected; the SBC and KS checks below are the evidence it is small here. The TV bound eps/2 in sec. 4.3 rests on the SOURCE's assumptions (4)-(5) about Monte Carlo accuracy, which are not verified here.
6. **Thin shells.** The SOURCE samples uniformly on a shell by rejection from the ball and, for very thin shells, replaces the dimension by 10^5 (an approximation). Sec. 4.1 mentions "radial transformations"; the exact radial inverse-CDF method is used, which is exact for any width.
7. **Infinite mixture.** Eq. 12 has infinitely many shells; only M are used. The doubling rule makes the dropped tail negligible in practice (in E4 the last shell's weight is <= 1e-4 for Student-t(5) and ~0 otherwise), but there is no a-priori bound.
8. **Scale of the method.** The paper says the method "scales to dimensions of several thousand", using diffeomorphic transformations for the 160-dimensional example. Tested here: d <= 5. The diffeomorphism and the MPI parallel implementation are **not implemented**.
9. **Outside this PR, noted while reading sec. 2 (not changed):** the level indices in eqs. (2)-(7) are inconsistent (eq. 3 uses Sigma_0 for p(x | v_1) while eq. 2 with i = 1 would use Sigma_1; eq. 6 sums from i = 0 although eq. 5 defines the error only for i >= 1; and by my derivation the gradient of eq. 6 with eq. 5 is -dF/dv_i = J_i^T Sigma_i^{-1} eps_i - Sigma_{i+1}^{-1} eps_{i+1}, which is not eq. 7 as printed).
   This matters when a hierarchical predictive-coding layer is implemented; it is raised here for discussion and nothing depends on it in this PR.

## 5. Validation

Unit and integration tests (`tests/test_ellipsoidal.py`, 42 tests; full suite 69 passed in about 3 s): shell volumes against closed forms and against the ball volume; uniform sampling (membership, radial law, isotropy, extremely thin shells, d = 200);
input validation (partition, radii, sampler arguments, misbehaving densities); exactness against closed forms (one shell, 1-D normal, correlated 2-D normal, 3-D with Monte Carlo bounds, bimodal mixture, mismatched partitions);
the coalescence-time law; infeasible settings raise instead of returning biased draws; reproducibility; reproduction of the PR #1 conjugate posterior.

Experiment E4 (`experiments/e4_ellipsoidal_sampler.py`, 10 seeds, pooled draws, alpha = 0.001, about 30 s): **45 of 45 checks pass (39 positive, 6 negative controls that fail as required)**; exit status 0.

| Part | What is checked | Key observed values |
|---|---|---|
| A one shell | KS vs exact CDF; mean abs x; mean coalescence time; violations | KS p 0.29; E T 1.179 vs theory 1.181; 0 violations. Controls fail: uniform-only (p ~ 0); support {1,2,...} (p ~ 1e-298) |
| B Gaussian, d = 1,2,3,5 | KS of Mahalanobis^2 vs chi^2(d); all mean and covariance z-scores < 4; violation rate | KS p 0.53 / 0.20 / 0.30 / 0.34 (MC bounds), all z <= 2.3, 0 violations. d = 2, 5 also with rigorous bounds. Controls fail: uniform shell choice; tail truncated at M = 4 |
| C non-Gaussian, 1-D | bimodal mixture and Student-t(5) vs exact CDFs | KS p > 0.001 for both; mean coalescence time 3.3 (mixture), 11.3 (t5); 2 violations in 450,779 steps for t5. Control fails: central ellipsoid only |
| D1 PR #1 posterior | draws vs the exact conjugate posterior of `ScalarGaussianBelief` | KS p 0.67; |mean z| 7e-4, |var z - 1| 3e-3 |
| D2 SBC with the sampler | chi^2 on 10 rank bins; 50/80/95% coverage (3000 trials, L = 99) | p 0.62; coverage 0.498 / 0.784 / 0.944 within 3 binomial SE. Control fails: wrong noise level 0.25 (p ~ 0) |

Two things about the experiment's own design, stated openly: (a) one negative control was changed after the first run: truncating Student-t(5) at 4 shells drops only ~0.13% of the mass, below what a KS test can detect with 40,000 draws, so it could not fail; it was replaced by keeping only the central ellipsoid (~5% of the mass). The d = 2 truncation control (M = 4) is unchanged and fails as required.
(b) During development the SBC control was first run with the correct belief but the wrong likelihood, which made the sampler refuse (infeasible partition); a wrong model must be wrong end to end (as in PR #1 E1), which is how it is run now.

Exit status: `e4_ellipsoidal_sampler.py` exits 0 only if every required check holds. Verified by injecting the SOURCE's stated coalescence form into every sampler: the run printed FAIL (20 failing rows) and exited 1.

## 6. Limitations

- Exactness is conditional: draws are from pi_hat, and the minorization uses estimated bounds unless rigorous ones are supplied (items 2 and 5). Violations are counted, not corrected.
- **Shape sensitivity.** Across a shell the density varies by about exp(-(beta_max - beta_min) rho^2 / 2) however thin the shell is (beta = eigenvalues of L^T Sigma_target^{-1} L). If the partition's shape is far from the target's, p_hat collapses in the tails; the sampler then raises `SamplerInfeasibleError` rather than returning biased draws (tested). Mild mismatch works (tested); strong anisotropic mismatch does not.
- Cost per draw is O(1/p_hat) residual steps (measured: mean 1.2-2 steps for the Gaussian runs, 11 for t5); per shell the weights cost N_i evaluations. Pure numpy, no parallelism; tested only to d = 5.
- A selected shell must have p_hat > `min_p`; coalescence times above `max_coalescence` raise instead of truncating.
- Single partition (single mu, Sigma): the bimodal target is handled only because the partition covers both modes. The multimodal extension (arXiv:2109.12633) is not implemented.
- Heavier tails than Student-t(5) (the paper's Cauchy example needs M of order 2000-3000) were not run.
- Posterior beliefs as `GenerativeModel` objects (an `update` that adds a likelihood term and a `sample` that calls this sampler) are not part of this PR; the sampler is a standalone building block with a `sample(n, rng)` of the same call shape.

## 7. Remaining work (needs approval before starting)

Thompson sampling with exact draws (paper sec. 3.3, 6.2) as the natural next consumer of this sampler; a belief class wrapping the sampler (sec. 3.1, 5.5); the recursive GP with look-up table (sec. 5); the derivative-aware optimisation (sec. 6), which the paper only cites (Roy and Bhattacharya 2026) and so cannot be implemented from this paper alone.

## Reproduce

    pip install -r requirements.txt
    pytest                                          # 69 passed
    python experiments/e4_ellipsoidal_sampler.py    # exit 0, ends with "...sampler; 45 checks, ...): PASS"
