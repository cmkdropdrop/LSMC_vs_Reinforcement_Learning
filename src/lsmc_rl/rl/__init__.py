"""Reinforcement-learning baselines for path-based valuation."""

from lsmc_rl.rl.fitted_q import (
    AmericanFittedQPolicy,
    AmericanFittedQResult,
    FittedQConfig,
    american_fitted_q_features,
    train_american_fitted_q,
    value_american_option_fitted_q,
)
from lsmc_rl.rl.kernel_fitted_q import (
    AmericanKernelFittedQPolicy,
    AmericanKernelFittedQResult,
    KernelFittedQConfig,
    RandomFourierFeatureMap,
    american_kernel_base_features,
    train_american_kernel_fitted_q,
    value_american_option_kernel_fitted_q,
)

__all__ = [
    "AmericanFittedQPolicy",
    "AmericanFittedQResult",
    "AmericanKernelFittedQPolicy",
    "AmericanKernelFittedQResult",
    "FittedQConfig",
    "KernelFittedQConfig",
    "RandomFourierFeatureMap",
    "american_fitted_q_features",
    "american_kernel_base_features",
    "train_american_kernel_fitted_q",
    "train_american_fitted_q",
    "value_american_option_kernel_fitted_q",
    "value_american_option_fitted_q",
]
