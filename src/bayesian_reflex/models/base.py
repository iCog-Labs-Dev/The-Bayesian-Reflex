"""Belief-maintenance interface (pillars 1 and 2). Later stages (GP, FSVI, look-up table,
ellipsoidal sampler, RGP) implement this same interface so the reflex loop never changes."""
from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    @abstractmethod
    def update(self, observation, context=None):
        """Sequential Bayes update: pi_t ∝ p(y_t | theta) pi_{t-1}."""

    @abstractmethod
    def predict(self, query=None):
        """One-step-ahead predictive distribution -> (mean, variance) for the next observation."""

    @abstractmethod
    def sample(self, n, rng):
        """Draw n samples from the current belief (needed for SBC and Thompson sampling)."""
