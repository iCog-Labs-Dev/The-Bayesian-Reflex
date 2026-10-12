## Bayesian Reflex: A Predictive Coding Engine

Unofficial implementation of [*The Bayesian Reflex: A Predictive Coding Engine for Artificial Intelligence*](https://arxiv.org/abs/2608.00492).

### Overview

The paper reinterprets the "Bayesian reflex" (online Bayesian learning) as a computational form of predictive coding. Its three mechanisms map onto free-energy minimisation:

1. **Belief maintenance:** a posterior over unknown parameters, held as particles, a GP posterior, an ellipsoidal decomposition, or a variational density.
2. **Sequential updating:** beliefs are revised with Bayes' theorem as new observations arrive.
3. **Uncertainty-driven action:** decisions use the full posterior to balance exploration and exploitation.

### Reference

- Paper: https://arxiv.org/abs/2608.00492

### Validation (PR 1: foundations E1-E3)

See [`docs/validation_foundations.md`](docs/validation_foundations.md) for the summary and
[docs/PR1_Description.pdf](docs/PR1_Description.pdf)
Prediction error is a diagnostic only in this PR; it does not drive the belief update.

### Ellipsoidal decomposition sampler (PR 2)

Exact i.i.d. sampling by ellipsoidal decomposition (paper sec. 4): see
[`docs/ellipsoidal_sampler.md`](docs/ellipsoidal_sampler.md) for the method, the ambiguities found in the paper, validation results and limitations,
and [`docs/PR2_Description.pdf`](docs/PR2_Description.pdf) for the full description. Run `python experiments/e4_ellipsoidal_sampler.py`.
