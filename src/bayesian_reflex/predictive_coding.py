"""Predictive-coding layer: REPORTS the discrepancy between belief-generated prediction and
observation. It does not perform the Bayes update and its outputs are diagnostics in this at this stage.
Whether the error should drive learning is an open design decision, deferred to experiment E7.

Three related quantities (e = y - y_hat, v = predictive variance, pi = 1 / v):

* ``prediction_error``          e              raw error, in the units of y
* ``precision_weighted_error``  pi * e         raw error scaled by predictive precision; variance 1/v
* ``standardized_error``        e / sqrt(v)    unit-variance if the belief is calibrated (E3 claim)

They are related by  standardized = sqrt(pi) * e  and  precision_weighted = standardized / sqrt(v).
"""
import numpy as np


class PredictiveCodingLayer:
    @staticmethod
    def prediction_error(y, y_hat):
        return y - y_hat

    @staticmethod
    def predictive_precision(var_pred):
        return 1.0 / var_pred

    def precision_weighted_error(self, y, y_hat, var_pred):
        """pi * e with pi = 1 / var_pred. NOT unit-variance: Var = 1 / var_pred when calibrated."""
        return self.predictive_precision(var_pred) * self.prediction_error(y, y_hat)

    def standardized_error(self, y, y_hat, var_pred):
        """z_t = (y - y_hat) / sqrt(var_pred); var_pred = sigma_{t-1}^2 + sigma_y^2 (one-step-ahead)."""
        return self.prediction_error(y, y_hat) / np.sqrt(var_pred)
