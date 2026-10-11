"""Orchestrator for one reflex step: predict -> report error -> update belief -> (optionally) act.

Role of prediction error in this PR: DIAGNOSTIC ONLY.
The error is computed and returned so it can be inspected and validated (E3), but it is NOT fed
back into the belief. ``model.update`` receives the raw observation ``y`` and applies the exact
Bayes update on its own; ``z`` never influences it. Whether the error should drive learning is an
explicit open design decision, deferred to a later experiment (E7). Acting (``policy``) is likewise
a placeholder until E6+.
"""


class BayesianReflex:
    def __init__(self, model, pc_layer, policy=None):
        self.model, self.pc, self.policy = model, pc_layer, policy

    def step(self, y, context=None):
        y_hat, var_pred = self.model.predict(context)      # prediction from the belief BEFORE the update
        z = self.pc.standardized_error(y, y_hat, var_pred)  # diagnostic only: not used by the update below
        self.model.update(y, context)                      # exact Bayes update from the raw observation
        action = self.policy(self.model) if self.policy else None   # E6+: uncertainty-driven
        return dict(y_hat=y_hat, var_pred=var_pred, z=z, action=action)
