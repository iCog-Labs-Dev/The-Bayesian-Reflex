"""E1 model: conjugate Gaussian belief over a scalar theta.

Prior theta ~ N(mu0, s0^2); likelihood y_t = theta + eps, eps ~ N(0, sy^2).
Array-valued so many independent trials run in parallel (one belief each).

Input contract
--------------
``n_trials`` independent beliefs are held, one per trial. ``mu0``, ``s0`` and ``sy`` may each be a
scalar (shared by all trials) or a 1-D array of length ``n_trials``. ``update`` accepts either

* a 0-d scalar (the same observation is broadcast to every trial), or
* a 1-D array of shape ``(n_trials,)`` (one observation per trial).

Every other shape (e.g. ``(n_trials, 1)``, ``(n_trials, 2)``, ``(1, n_trials)``, an empty array) is
rejected with ``ValueError``, as are NaN and +/-inf. Validation happens before any state is touched,
so a rejected call leaves the belief exactly as it was.
"""
import numpy as np

from .base import GenerativeModel


def _as_trial_vector(value, n_trials, name, positive=False):
    """Return a float64 array of shape (n_trials,) or raise ValueError."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be real-valued, got {value!r}") from exc
    if arr.ndim == 0:
        arr = np.full(n_trials, float(arr))
    elif arr.shape != (n_trials,):
        raise ValueError(f"{name} must be a scalar or have shape ({n_trials},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and not np.all(arr > 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr.copy()


class ScalarGaussianBelief(GenerativeModel):
    def __init__(self, mu0=0.0, s0=1.0, sy=0.5, n_trials=1):
        if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
            raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}")
        self.n_trials = int(n_trials)
        s0 = _as_trial_vector(s0, self.n_trials, "s0", positive=True)
        sy = _as_trial_vector(sy, self.n_trials, "sy", positive=True)
        self.sy2 = sy ** 2
        self.mu = _as_trial_vector(mu0, self.n_trials, "mu0")
        self.prec = 1.0 / s0 ** 2

    @property
    def var(self):
        return 1.0 / self.prec

    @property
    def sd(self):
        return np.sqrt(self.var)

    def _check_observation(self, observation):
        """Validate shape and finiteness; return a float64 array of shape (n_trials,)."""
        try:
            y = np.asarray(observation, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"observation must be real-valued, got {observation!r}") from exc
        if y.ndim == 0:
            y = np.full(self.n_trials, float(y))
        elif y.shape != (self.n_trials,):
            raise ValueError(
                f"observation must be a scalar or have shape ({self.n_trials},), got {y.shape}")
        if not np.all(np.isfinite(y)):
            raise ValueError("observation contains NaN or infinite values")
        return y

    def update(self, observation, context=None):
        y = self._check_observation(observation)            # raises BEFORE the belief is modified
        new_prec = self.prec + 1.0 / self.sy2
        new_mu = (self.prec * self.mu + y / self.sy2) / new_prec
        self.mu, self.prec = new_mu, new_prec               # commit only after everything succeeded

    def predict(self, query=None):
        return self.mu.copy(), self.var + self.sy2

    def sample(self, n, rng):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        return self.mu[:, None] + self.sd[:, None] * rng.standard_normal((self.n_trials, n))
