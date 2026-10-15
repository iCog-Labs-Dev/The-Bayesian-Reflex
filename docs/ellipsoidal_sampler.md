# Ellipsoidal decomposition sampler (paper sec. 4)

Scope: this PR implements the one algorithmic component of the paper that is missing after the foundations PR and that the
paper's later sections depend on: **i.i.d. sampling from a posterior by ellipsoidal decomposition** (sec. 4, eqs. 11-13). The draws are
exact only under stated conditions and approximate otherwise; section 3 says exactly which.
It does not implement the reflex loop, Thompson sampling (sec. 3.3, 6.2), recursive Gaussian processes (sec. 5),
derivative-aware optimisation (sec. 6), or anything from the function-space / PC-FSVI line of work, and it makes no comparison
with other inference methods. Nothing in the merged foundations code was modified.

Sources. Sec. 4 of the paper states the method in a page and attributes it to Bhattacharya, "IID Sampling from Intractable
Distributions" (arXiv:2107.05956, called the SOURCE below). Where sec. 4 leaves a detail open, the SOURCE was read and used; each such
place is listed in section 4 so it can be approved or changed.

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

**Overall sampler (sec. 4.3).** `prepare` first fixes the shells 1..M and their estimated weights, before any returned draw is made. Each draw then picks shell i with
probability proportional to the estimated weights and draws perfectly from pi_i. M is doubled until the omitted mass is at most `tail_tol` of the retained mass, judged by a
supplied bound or, without one, by an estimate. This differs from the SOURCE's Algorithm 1, step 6(ii), which doubles M when a *returned* draw lands in the last shell
(item 10 of section 4).

## 3. What is exact and what is approximate

Write pi_hat_M = sum_{i<=M} w_i pi_i for the distribution the sampler draws from, with w_i the estimated, normalised shell weights. Because the shells are disjoint,
TV(pi_hat_M, pi) = 1/2 sum_{i<=M} |pi(A_i) - w_i| + tau/2, where tau = pi(outside shell M). Everything approximate sits in those two terms and in the minorization condition.

| Part | Exact? | Condition |
|---|---|---|
| Draw inside a selected shell, from pi_i | exact | the minorization P >= p_hat Q holds for the p_hat used, T is geometric on {0,1,2,...}, and the coalescence time is finite. **Guaranteed in mode `rigorous`** (valid bounds supplied; any violation or out-of-bounds density raises). **Not guaranteed in mode `monte_carlo`.** |
| Choice of shell, given the weights | exact from w | the weights and M are fixed by `prepare` before the returned draws and never change, so `sample(1)`, repeated calls and one bulk call have the same distribution (tested) |
| Weights w_i versus pi(A_i) | **approximate** | Monte Carlo estimate with relative error about CV_i / sqrt(N_i); contributes the first TV term. Reported as `tv_weight_error_estimate` (at most 3.2e-3 in E4). Not corrected: the paper's optional rejection step is not specified |
| Omitted tail beyond shell M | **approximate** | contributes tau/2. Bounded by `tail_tol` (relative to the retained mass) only when a valid `log_tail_bound` is supplied (mode `rigorous` requires it; a Gaussian bound is provided). Otherwise `omitted_mass_estimate` is a geometric extrapolation of the last weights: an estimate, with no guarantee |
| Minorization constant p_hat | rigorous mode: valid by construction. monte_carlo mode: **estimated** | the sampled min and max are optimistic: a fresh uniform point falls below the sampled minimum with probability 1/(N+1), so an estimated bound is never rigorous |
| Independence of draws | i.i.d. **conditional on the prepared state** | within and across calls draws are i.i.d. given (shells, w, p_hat); unconditionally they share the estimated weights |

**When is the sampling exact?** Draws are exactly i.i.d. from pi only if the minorization is valid (mode `rigorous`), the weights are exact (w_i = pi(A_i); this implementation does not provide exact weights)
and no mass is omitted (tau = 0). In practice, in mode `rigorous` the result is within TV <= (weight error, estimated) + tail_tol / 2 of pi, where the second term is bounded and the first is only estimated.
In mode `monte_carlo`, the paper's practical mode, no guarantee is made: the draws are approximate, the omitted mass is an estimate, and a violation of the minorization is detected only if the chain visits it.

That detection can fail silently. `test_monte_carlo_mode_can_be_silently_wrong_while_rigorous_mode_is_not` builds a target with a narrow spike holding 83% of the mass: the Monte Carlo bounds miss it, p_hat is close to 1,
no residual step is taken, no violation is ever seen, and the spike is missed (under 5% of the draws land in it). A reported count of zero is therefore not evidence that the bounds are valid; the guarantee is reserved for mode `rigorous`.

**Violation policy.** In mode `rigorous` any violation, or any density value outside the supplied bounds, raises `MinorizationViolationError`. In mode `monte_carlo` a violation raises if its relative shortfall (p_hat - a)/p_hat
exceeds `violation_tol` (default 1e-2); smaller ones are counted and the worst is reported as `max_violation_shortfall`. Measured on Student-t(5), 40,000 draws: 5 violations in 451,962 steps, worst shortfall 1.1e-3 (4 seeds: 0 to 5 violations);
the Gaussian runs and the mixture had none. `violation_tol=0` raises on any; `on_violation="count"` never raises and makes no guarantee.

## 4. Ambiguities and discrepancies, and what was done

Each item is either a place where the paper is silent or where the paper (or its SOURCE) disagrees with the maths or with itself. None was resolved silently;
items 1, 2 and 10 change results and need a decision.

1. **Support of the geometric coalescence time (discrepancy).** Sec. 4.2 says only "T_i ~ Geometric(p_i)". The SOURCE (eq. 24) states P(T = t) = p (1-p)^(t-1) for t = 1, 2, ... and says the form with t = 0, 1, ... "failed".
   With T >= 1 the output law is pi_i R = (pi_i - p Q_i)/(1 - p), not pi_i, because the regeneration-at-once event (probability p) is dropped. Checked numerically on a shell with known answer (p = 0.458, 200,000 draws per run):
   support {0,1,...}: KS p = 0.29, E|x| = 1.2621 (exact 1.2618); support {1,2,...}: KS p ~ 0, E|x| = 1.2292, matching the predicted (1.2618 - 0.458 x 1.3)/(1 - 0.458) = 1.2292. A uniform draw with no residual steps also fails.
   **Default here: {0,1,...}**; `geometric_support=1` reproduces the SOURCE's stated form. It is possible the SOURCE's code indexes time differently from its text; its text could not be tested against its code.
2. **p_i = inf pi / sup pi is estimated, not bounded (exactness is conditional).** The SOURCE replaces inf and sup by the min and max of pi_tilde over the Monte Carlo draws and subtracts eta (1e-5; 0 in one example).
   Those estimates are optimistic, so the minorization can fail and the residual step is then clamped. Handling: two explicit modes. `monte_carlo` follows the SOURCE and makes **no** perfect-sampling guarantee; `rigorous` takes caller-supplied
   valid bounds and a valid tail bound, and raises on any violation (section 3). A Gaussian helper provides such bounds (exact when the partition has the target's shape); for general targets the paper gives none.
3. **Radii and M.** Only the family sqrt(c_i) = r + a(i-1) with experiment-specific (r, a, M) is given; there is no rule. (r, a) are required arguments. M is chosen by the coverage criterion of item 10.
4. **mu and Sigma.** The paper says "e.g. posterior mode or mean" and "an estimate of the posterior covariance"; the SOURCE obtains them from TMCMC runs. TMCMC is a different algorithm and is **not implemented**; mu and Sigma are arguments. Any choice is valid, but the shape matters for feasibility (see Limitations).
5. **Weight estimation error and the "optional rejection step" (sec. 4.3).** Draws are exact from pi_hat_M (estimated weights), not from pi. Sec. 4.3 mentions an optional rejection correction with acceptance >= (1-eps)/(1+eps) but does not say what is accepted or rejected;
   the SOURCE argues (its sec. 2.4) that it accepts with probability ~1 and omits it from Algorithm 1. It is **not implemented**. The error is O(N_i^(-1/2)) in the weights; it is estimated (`tv_weight_error_estimate`), not bounded and not corrected. The TV bound eps/2 in sec. 4.3 rests on the SOURCE's assumptions (4)-(5) about Monte Carlo accuracy, which are not verified here.
6. **Thin shells.** The SOURCE samples uniformly on a shell by rejection from the ball and, for very thin shells, replaces the dimension by 10^5 (an approximation). Sec. 4.1 mentions "radial transformations"; the exact radial inverse-CDF method is used, which is exact for any width.
7. **Infinite mixture.** Eq. 12 has infinitely many shells; only M are used. The omitted mass is controlled by item 10; there is no a-priori bound for a general target.
8. **Scale of the method.** The paper says the method "scales to dimensions of several thousand", using diffeomorphic transformations for the 160-dimensional example. Tested here: d <= 5. The diffeomorphism and the MPI parallel implementation are **not implemented**.
9. **Outside this PR, noted while reading sec. 2 (not changed):** the level indices in eqs. (2)-(7) are inconsistent (eq. 3 uses Sigma_0 for p(x | v_1) while eq. 2 with i = 1 would use Sigma_1; eq. 6 sums from i = 0 although eq. 5 defines the error only for i >= 1; and by my derivation the gradient of eq. 6 with eq. 5 is -dF/dv_i = J_i^T Sigma_i^{-1} eps_i - Sigma_{i+1}^{-1} eps_{i+1}, which is not eq. 7 as printed).
   This matters when a hierarchical predictive-coding layer is implemented; it is raised here for discussion and nothing depends on it in this PR.
10. **Shell coverage is fixed independently of the returned draws (deliberate deviation from Algorithm 1, step 6(ii)).** The SOURCE doubles M, with the same uniforms, when a returned draw lands in the last shell. That makes the shell set, and so the normalisation of the weights, depend on the
    random numbers that produce the returned draws: draws within a call are then not identically distributed given M, and `sample(1)`, repeated calls and one bulk call do not share a distribution. Here `prepare` fixes M before any returned draw, from a criterion that does not look at them:
    expand until (omitted mass) / (retained mass) <= `tail_tol` (default 1e-6). With a supplied `log_tail_bound` this is a **bound** (rigorous mode requires one). Without it, the omitted mass is **estimated** by geometric extrapolation of the last weight ratios
    (which requires the last `patience` ratios to be below 1); for a power-law tail this extrapolation is close to the true tail (within a factor k/(k-1) for w_i ~ i^-k) but it is an estimate, and it can underestimate tails that thin out in steps. In general no bound on the omitted
    mass can be derived from the density on the prepared shells alone, which is why the bound must come from the caller. Tested: the shell set is identical whatever the size of the first call (1 versus 50,000 draws), unchanged by later calls, and `sample(1)` x 2500, `sample(100)` x 25 and
    `sample(2500)` agree with each other and with the exact law (also E4, part E).

## 5. Implementation map

| Paper / SOURCE | Code (`src/bayesian_reflex/sampling/`) |
|---|---|
| shells, eq. 11; radii sqrt(c_1) = r, sqrt(c_i) = r + a(i-1) (SOURCE sec. 5) | `EllipsoidalPartition`, `LinearRadii` |
| shell volume (SOURCE 2.3.2) and uniform sampling on a shell (SOURCE 2.3.1) | `log_volume`, `sample_uniform` (exact radial inverse CDF in log space) |
| mixture weights, eq. 12, SOURCE eq. 3; p_hat_i = s_hat_i / S_hat_i - eta (SOURCE 3.1) | `EllipsoidalSampler._ensure_shell` |
| split chain, perfect draw (eq. 13, SOURCE 3.2-3.3) | `EllipsoidalSampler._perfect_sample_shell` |
| shell coverage (item 10), weights fixed before drawing | `EllipsoidalSampler.prepare` |
| overall sampler (sec. 4.3, SOURCE Algorithm 1) | `EllipsoidalSampler.sample`, `sample_from_shells` |
| valid shell and tail bounds for Gaussian targets (not in the paper; they make mode `rigorous` available and separate "the algorithm" from "the Monte Carlo bounds") | `gaussian_log_density`, `gaussian_shell_log_bounds`, `gaussian_log_tail_bound` |

## 6. Validation

Unit and integration tests (`tests/test_ellipsoidal.py`, 54 tests; full suite 81 passed in about 5 s): shell volumes against closed forms and against the ball volume; uniform sampling (membership, radial law, isotropy, extremely thin shells, d = 200);
input validation (partition, radii, sampler arguments, modes, misbehaving densities); exactness against closed forms (one shell, 1-D normal, correlated 2-D normal, 3-D with Monte Carlo bounds, bimodal mixture, mismatched partitions);
the coalescence-time law; infeasible settings raise instead of returning biased draws; reproducibility; reproduction of the PR #1 conjugate posterior; and, added after review,
**shells fixed by `prepare` and independent of batch size; `sample(1)`, repeated calls and bulk sampling agree; the Gaussian tail bound is a valid upper bound; coverage stops only when the criterion holds; rigorous mode raises on wrong bounds;
a detected violation raises by default and is only counted on request; the tolerance is explicit and reported; Monte Carlo mode can be silently wrong while rigorous mode is not.**

Experiment E4 (`experiments/e4_ellipsoidal_sampler.py`, 10 seeds, pooled draws, alpha = 0.001, about 35 s on the reference machine): **59 of 59 checks pass (52 positive, 7 negative controls that fail as required)**; exit status 0.

| Part | What is checked | Key observed values |
|---|---|---|
| A one shell (rigorous mode) | KS vs exact CDF; mean abs x; mean coalescence time; violations | KS p 0.29; E T 1.179 vs theory 1.181; 0 violations. Controls fail: uniform-only (p ~ 0); support {1,2,...} (p ~ 1e-298) |
| B Gaussian, d = 1,2,3,5 (monte_carlo mode) | KS of Mahalanobis^2 vs chi^2(d); all mean and covariance z-scores < 4; violation rate and worst shortfall | KS p 0.53 / 0.20 / 0.30 / 0.34, all z <= 2.3, 0 violations (worst shortfall 0). d = 2, 5 also in rigorous mode. Controls fail: uniform shell choice; tail truncated at M = 4 |
| C non-Gaussian, 1-D (monte_carlo mode) | bimodal mixture and Student-t(5) vs exact CDFs | KS p > 0.001 for both; mean coalescence time 3.3 (mixture), 11.2 (t5); 0 violations in this run. Control fails: central ellipsoid only |
| D1 PR #1 posterior | draws vs the exact conjugate posterior of `ScalarGaussianBelief` | KS p 0.67; abs mean z 7e-4, abs var z - 1 3e-3 |
| D2 SBC with the sampler | chi^2 on 10 rank bins; 50/80/95% coverage (3000 trials, L = 99) | p 0.62; coverage 0.498 / 0.784 / 0.944 within 3 binomial SE. Control fails: wrong noise level 0.25 (p ~ 0) |
| E coverage independent of the draws | sample(1) x 500, sample(100) x 5, sample(500) per seed: each vs chi^2(2); three pairwise two-sample KS; shell set unchanged | all p > 0.1 (smallest 0.12); 0 of 10 seeds changed the shell set. Control fails: target vs covariance x 1.2 (p 3e-7) |

Estimated weight error (total variation, first-order estimate) was at most 3.2e-3 in every configuration; the estimated omitted mass was below 1e-6 (rigorous mode: bound at most 1e-6).

Two things about the experiment's own design, stated openly: (a) one negative control was changed after the first run: truncating Student-t(5) at 4 shells drops only ~0.13% of the mass, below what a KS test can detect with 40,000 draws, so it could not fail; it was replaced by keeping only the central ellipsoid (~5% of the mass). The d = 2 truncation control (M = 4) is unchanged and fails as required.
(b) During development the SBC control was first run with the correct belief but the wrong likelihood, which made the sampler refuse (infeasible partition); a wrong model must be wrong end to end (as in PR #1 E1), which is how it is run now.
Part E was added in response to review. Its control shows the test can detect a 20% difference in scale; a deviation as small as the one the old coupled rule could cause is below its resolution, which is why the property is also covered by the deterministic unit tests (identical shell set for any batch size).

Exit status: `e4_ellipsoidal_sampler.py` exits 0 only if every required check holds. Verified by injecting the SOURCE's stated coalescence form into every sampler: the run printed FAIL (24 failing rows) and exited 1.

## 7. Limitations

- **Exactness is conditional** (section 3): the weights are estimated, the tail is bounded only with a supplied bound, and mode `monte_carlo` has no perfect-sampling guarantee; a zero violation count does not prove valid bounds.
- **Shape sensitivity.** Across a shell the density varies by about exp(-(beta_max - beta_min) rho^2 / 2) however thin the shell is (beta = eigenvalues of L^T Sigma_target^{-1} L). If the partition's shape is far from the target's, p_hat collapses in the tails; the sampler then raises `SamplerInfeasibleError` rather than returning biased draws (tested). Mild mismatch works (tested); strong anisotropic mismatch does not.
- Cost per draw is O(1/p_hat) residual steps (measured: mean 1.2-2 steps for the Gaussian runs, 11 for t5); per shell the weights cost N_i evaluations. Pure numpy, no parallelism; tested only to d = 5.
- A selected shell must have p_hat > `min_p`; coalescence times above `max_coalescence` raise instead of truncating.
- Single partition (single mu, Sigma): the bimodal target is handled only because the partition covers both modes. The multimodal extension (arXiv:2109.12633) is not implemented.
- Heavier tails than Student-t(5) (the paper's Cauchy example needs M of order 2000-3000) were not run.
- Posterior beliefs as `GenerativeModel` objects (an `update` that adds a likelihood term and a `sample` that calls this sampler) are not part of this PR; the sampler is a standalone building block with a `sample(n, rng)` of the same call shape.

## 8. Remaining work (needs approval before starting)

Thompson sampling with draws from this sampler (paper sec. 3.3, 6.2) as the natural next consumer; a belief class wrapping the sampler (sec. 3.1, 5.5); the recursive GP with look-up table (sec. 5); the derivative-aware optimisation (sec. 6), which the paper only cites (Roy and Bhattacharya 2026) and so cannot be implemented from this paper alone.

## Reproduce

    pip install -r requirements.txt
    pytest                                          # 81 passed
    python experiments/e4_ellipsoidal_sampler.py    # exit 0, ends with "...sampler; 59 checks, ...): PASS"
