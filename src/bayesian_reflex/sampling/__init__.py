"""Ellipsoidal decomposition i.i.d. sampler (paper sec. 4)."""
from .bounds import gaussian_log_density, gaussian_log_tail_bound, gaussian_shell_log_bounds
from .ellipsoidal import (EllipsoidalPartition, EllipsoidalSampler, LinearRadii, MinorizationViolationError,
                          SamplerInfeasibleError)

__all__ = ["EllipsoidalPartition", "EllipsoidalSampler", "LinearRadii", "MinorizationViolationError",
           "SamplerInfeasibleError", "gaussian_log_density", "gaussian_log_tail_bound", "gaussian_shell_log_bounds"]
