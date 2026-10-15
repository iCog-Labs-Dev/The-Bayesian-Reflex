# Foundations validation: E1-E3

Scope: these three experiments validate the scalar Gaussian inference module of the Bayesian-reflex implementation (belief maintenance, sequential updating, standardized one-step-ahead prediction error). They do NOT test the reflex loop, function-space inference, nonstationarity, action, or the paper-specific machinery; those come in later PRs.

| Stage | Question | Reference | Seeds | Result |
|---|---|---|---|---|
| E1 | Is the scalar belief accurate and calibrated (simulation-based calibration, 1000 trials x 10 seeds)? | analytic Gaussian posterior | 0-9 | 33/33 checks |
| E2a | Does the recursive update equal the batch posterior (T=100, stress T=1e5)? | batch posterior in extended precision | 0-9 (100 problems per seed at T=100, 20 per seed at T=1e5) | 9/9 checks |
| E3 | Are one-step-ahead errors correctly standardized by the predictive variance? Kalman identity, precision-weighted error, controls, unknown-variance (Student-t) variant | exact predictive | 0-9 (+ reruns on 10-19 and 20-29) | 41/41 checks |

## Role of prediction error in this PR

Prediction error is a **diagnostic only** in this PR. `BayesianReflex.step` computes the error and returns it, but the belief update uses the raw observation and never sees the error (`tests/test_core.py::test_prediction_error_is_diagnostic_only` checks that the belief after `step` equals the belief after a bare `update`). Whether the error should drive learning is an open design decision, deferred to E7. The same wording applies everywhere in this document: the predictive-coding layer *reports*; the belief model *updates*.

## Three different error quantities

With `e = y - y_hat` and predictive variance `v = sigma_{t-1}^2 + sigma_y^2`:

| Quantity | Code | Formula | Calibrated behaviour | Tested by |
|---|---|---|---|---|
| raw error | `prediction_error` | `e` | scale follows `sigma_y` | E3 precision-weighting ratio (10.0) |
| precision-weighted error | `precision_weighted_error` | `e / v` | variance `1/v`, NOT 1 | E3: identity `pw*sqrt(v) = z`, `Var(pw)*v ~ 1` at t=1 and 50, and `mu_t = mu_{t-1} + sigma_{t-1}^2 * pw_t`; controls: raw error in the update fails, "pw is unit variance" fails |
| standardized error | `standardized_error` | `e / sqrt(v)` | N(0,1), independent over t | E3 mean / variance / lag-1 / KS checks |

E3's calibration claim (N(0,1), independent) is about the **standardized** error `z`. The **precision-weighted** error is a different quantity: it is not unit variance, and it is checked as an identity and as the quantity that scales the prior variance in the Kalman update.

## Seeds

The ten-seed rule applies to every pass/fail result. E1 and E3 use seeds 0-9. E2a now also uses seeds 0-9: each seed draws its own 100 random problems (T=100) and 20 stress problems (T=1e5); every E2a criterion is evaluated on the worst case over all seeds and problems, and the negative control must differ on every seed (minimum over seeds is tested). The E1 prior-mismatch result is descriptive (not pass/fail) and is a single run (500 trials, seed 123); it is not covered by the ten-seed rule and is reported as such.

## Scope of the Student-t (unknown variance) experiment

The Normal-Inverse-Gamma / Student-t part of E3 is a **standalone** check. `nig_run` is self-contained NumPy/SciPy code; it does not call `ScalarGaussianBelief`, `PredictiveCodingLayer` or `GenerativeModel`. It validates the closed-form NIG predictive derivation and its probability-integral transform, not the shared framework. Its rows are tagged `[standalone]` in the output and in `results/e3_results.json`. Validating the framework in this setting would need a NIG belief that implements `GenerativeModel`; that is future work.

## E3 sensitivity reruns

`experiments/e3_rerun_sensitivity.py` reruns the E3 mean / variance / lag-1 checks, with unchanged criteria, on two fresh sets of 10 seeds (10-19 and 20-29) for both sigma_y = 0.1 and 1.0. Seeds and summaries are in `results/e3_rerun_sensitivity.json`:

| sigma_y | seed set | mean z (limit 0.0042) | var z (limit 0.0060 from 1) | lag-1 (limit 0.0042) |
|---|---|---|---|---|
| 0.1 | 0-9 (primary) | 0.0028 | 0.9969 | -0.0037 |
| 0.1 | 10-19 | -0.0036 | 0.9996 | -0.0001 |
| 0.1 | 20-29 | -0.0018 | 0.9966 | -0.0008 |
| 1.0 | 0-9 (primary) | 0.0013 | 0.9969 | -0.0037 |
| 1.0 | 10-19 | -0.0039 | 0.9996 | -0.0003 |
| 1.0 | 20-29 | -0.0015 | 0.9967 | -0.0009 |

All statistics are inside their pre-fixed limits. Two caveats: sigma_y = 0.1 and 1.0 share seeds, so the two rows per seed set are not independent evidence; and several mean-z values lie at roughly 2-2.8 standard errors, inside the 3-SE rule but not far inside. This is consistent with chance, not proof of it.

## Exit status

All experiment commands exit 0 only if every required check passes (positive checks and negative controls) and exit 1 otherwise: `e1_scalar_belief.py`, `e2_sequential_equivalence.py`, `e3_prediction_error.py`, and `e3_rerun_sensitivity.py`. E1's closed-form posterior check is now a reported row rather than a bare `assert`.

## Input validation

`ScalarGaussianBelief.update` accepts a scalar or a 1-D array of shape `(n_trials,)`. Any other shape, and any NaN/inf, raises `ValueError` before the belief is modified. Constructor parameters (`mu0`, `s0`, `sy`, `n_trials`) are validated the same way.

Method rules used throughout: pass/fail criteria are fixed before running; every positive test has a negative control that must fail (wrong noise level, stale variance, dropped data, wrong CDF, raw error in the update); >= 10 seeds with tolerances from confidence intervals; claims are limited to what a stage tests.

Run (about 1 minute in total):

    pip install -r requirements.txt
    pytest                                          # 27 passed
    python experiments/e1_scalar_belief.py          # exit 0, ends with "...foundation only): PASS"
    python experiments/e2_sequential_equivalence.py # exit 0, ends with "...vs batch oracle): PASS"
    python experiments/e3_prediction_error.py       # exit 0, ends with "...prediction error): PASS"
    python experiments/e3_rerun_sensitivity.py      # exit 0, ends with "...both sigma_y): PASS"

Layout: src/bayesian_reflex/ (GenerativeModel, ScalarGaussianBelief, PredictiveCodingLayer, BayesianReflex), experiments/, tests/, docs/, results/ (figures and json written by the scripts).