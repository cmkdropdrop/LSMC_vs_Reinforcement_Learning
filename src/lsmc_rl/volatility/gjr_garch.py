"""GJR-GARCH(1,1) volatility model.

Model equation, fitted by Gaussian quasi maximum likelihood on scaled returns:

    r_t = mu + eps_t
    sigma_t^2 = omega
        + alpha * eps_{t-1}^2
        + gamma * I(eps_{t-1} < 0) * eps_{t-1}^2
        + beta * sigma_{t-1}^2

The optimizer constrains ``omega > 0`` and
``alpha + 0.5 * gamma + beta < 1`` to keep variance paths positive and
covariance-stationary under symmetric innovations.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class GJRGARCHParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    return_scale: float

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta


class GJRGARCHModel:
    """Small, dependency-light GJR-GARCH(1,1) model for path simulation."""

    def __init__(self, return_scale: float = 100.0, min_variance: float = 1e-12) -> None:
        self.return_scale = float(return_scale)
        self.min_variance = float(min_variance)
        self.params: GJRGARCHParams | None = None
        self.last_residual: float | None = None
        self.last_variance: float | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> "GJRGARCHModel":
        clean = _clean_returns(returns) * self.return_scale
        if clean.size < 30:
            raise ValueError("GJR-GARCH fit requires at least 30 finite returns")

        sample_var = float(np.var(clean, ddof=1))
        sample_var = max(sample_var, self.min_variance)
        mean_return = float(np.mean(clean))
        initial_guesses = [
            np.array([mean_return, sample_var * 0.03, 0.05, 0.05, 0.90]),
            np.array([mean_return, sample_var * 0.05, 0.08, 0.02, 0.85]),
            np.array([mean_return, sample_var * 0.10, 0.10, 0.10, 0.75]),
            np.array([mean_return, sample_var * 0.01, 0.03, 0.00, 0.94]),
        ]
        bounds = [
            (None, None),
            (self.min_variance, None),
            (0.0, 0.999),
            (0.0, 0.999),
            (0.0, 0.999),
        ]
        constraints = (
            {"type": "ineq", "fun": lambda p: 0.999 - p[2] - 0.5 * p[3] - p[4]},
        )

        best_result = None
        for initial in initial_guesses:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = minimize(
                    _negative_log_likelihood,
                    initial,
                    args=(clean, self.min_variance),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-8, "disp": False},
                )
            if result.success and np.isfinite(result.fun):
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
        if best_result is None:
            raise RuntimeError("GJR-GARCH optimization failed for all initial guesses")

        mu, omega, alpha, gamma, beta = [float(value) for value in best_result.x]
        params = GJRGARCHParams(
            mu=mu,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            return_scale=self.return_scale,
        )
        variances = _conditional_variances(clean, params, self.min_variance)
        residuals = clean - params.mu
        self.params = params
        self.last_residual = float(residuals[-1])
        self.last_variance = float(variances[-1])
        return self

    def forecast_variance(self, horizon_steps: int) -> np.ndarray:
        """Forecast raw log-return variance for future steps."""

        self._require_fitted()
        assert self.params is not None
        assert self.last_residual is not None
        assert self.last_variance is not None

        variances = np.empty(horizon_steps, dtype=float)
        residual_sq = self.last_residual**2
        variance = self.last_variance
        for step in range(horizon_steps):
            variance = (
                self.params.omega
                + (self.params.alpha + 0.5 * self.params.gamma) * residual_sq
                + self.params.beta * variance
            )
            variance = max(float(variance), self.min_variance)
            variances[step] = variance / self.return_scale**2
            residual_sq = variance
        return variances

    def simulate(
        self,
        start_price: float,
        horizon_steps: int,
        n_paths: int,
        seed: int | None = None,
        innovations: str = "normal",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate lognormal price paths from the last fitted state.

        Returns ``(prices, returns, variances)`` with shapes
        ``(n_paths, horizon_steps + 1)``, where step 0 is the start state.
        Innovations currently support ``normal``; the argument is part of the
        stable interface for future Student-t or bootstrap innovations.
        """

        if innovations != "normal":
            raise NotImplementedError("Only normal innovations are implemented for now")
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        if horizon_steps <= 0 or n_paths <= 0:
            raise ValueError("horizon_steps and n_paths must be positive")

        self._require_fitted()
        assert self.params is not None
        assert self.last_residual is not None
        assert self.last_variance is not None

        rng = np.random.default_rng(seed)
        prices = np.empty((n_paths, horizon_steps + 1), dtype=float)
        returns = np.zeros((n_paths, horizon_steps + 1), dtype=float)
        variances = np.full((n_paths, horizon_steps + 1), np.nan, dtype=float)
        prices[:, 0] = float(start_price)

        residual = np.full(n_paths, self.last_residual, dtype=float)
        variance = np.full(n_paths, self.last_variance, dtype=float)

        for step in range(1, horizon_steps + 1):
            leverage = (residual < 0.0).astype(float)
            variance = (
                self.params.omega
                + self.params.alpha * residual**2
                + self.params.gamma * leverage * residual**2
                + self.params.beta * variance
            )
            variance = np.maximum(variance, self.min_variance)
            residual = np.sqrt(variance) * rng.normal(size=n_paths)
            scaled_return = self.params.mu + residual
            raw_return = scaled_return / self.return_scale
            returns[:, step] = raw_return
            variances[:, step] = variance / self.return_scale**2
            prices[:, step] = prices[:, step - 1] * np.exp(raw_return)

        return prices, returns, variances

    def _require_fitted(self) -> None:
        if self.params is None or self.last_residual is None or self.last_variance is None:
            raise RuntimeError("GJRGARCHModel must be fitted before forecasting or simulation")


def _clean_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(returns, dtype=float)
    return array[np.isfinite(array)]


def _conditional_variances(
    scaled_returns: np.ndarray,
    params: GJRGARCHParams,
    min_variance: float,
) -> np.ndarray:
    residuals = scaled_returns - params.mu
    variances = np.empty_like(scaled_returns, dtype=float)
    denom = max(1.0 - params.persistence, 1e-6)
    variances[0] = max(params.omega / denom, np.var(scaled_returns, ddof=1), min_variance)
    for idx in range(1, len(scaled_returns)):
        leverage = 1.0 if residuals[idx - 1] < 0.0 else 0.0
        variances[idx] = (
            params.omega
            + params.alpha * residuals[idx - 1] ** 2
            + params.gamma * leverage * residuals[idx - 1] ** 2
            + params.beta * variances[idx - 1]
        )
        variances[idx] = max(float(variances[idx]), min_variance)
    return variances


def _negative_log_likelihood(
    parameter_values: np.ndarray,
    scaled_returns: np.ndarray,
    min_variance: float,
) -> float:
    mu, omega, alpha, gamma, beta = [float(value) for value in parameter_values]
    if omega <= 0.0 or alpha < 0.0 or gamma < 0.0 or beta < 0.0:
        return float("inf")
    if alpha + 0.5 * gamma + beta >= 0.999:
        return float("inf")

    params = GJRGARCHParams(
        mu=mu,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        return_scale=1.0,
    )
    variances = _conditional_variances(scaled_returns, params, min_variance)
    residuals = scaled_returns - mu
    return float(0.5 * np.sum(np.log(variances) + residuals**2 / variances))
