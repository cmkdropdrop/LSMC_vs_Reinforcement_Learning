"""HAR-RV volatility model with a linear regression core.

Daily realized variance is computed from intraday log returns. Forecast
features for day t are lagged values only: previous-day RV, previous 5-day
average RV, and previous 22-day average RV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HARRVParams:
    intercept: float
    beta_daily: float
    beta_weekly: float
    beta_monthly: float
    min_variance: float
    step_mean_return: float

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([self.beta_daily, self.beta_weekly, self.beta_monthly], dtype=float)


class HARRVModel:
    """Heterogeneous autoregressive model for realized variance."""

    feature_columns = ["rv_daily", "rv_weekly", "rv_monthly"]

    def __init__(
        self,
        weekly_window: int = 5,
        monthly_window: int = 22,
        min_variance: float = 1e-12,
    ) -> None:
        self.weekly_window = int(weekly_window)
        self.monthly_window = int(monthly_window)
        self.min_variance = float(min_variance)
        self.params: HARRVParams | None = None
        self.history_: pd.Series | None = None

    @staticmethod
    def daily_realized_variance(
        intraday_frame: pd.DataFrame,
        return_col: str = "log_return",
        time_col: str = "open_datetime",
    ) -> pd.DataFrame:
        """Aggregate intraday returns into UTC daily realized variance."""

        frame = intraday_frame[[time_col, return_col]].dropna().copy()
        if frame.empty:
            raise ValueError("No finite intraday returns available for realized variance")
        frame["date"] = pd.to_datetime(frame[time_col], utc=True).dt.floor("D")
        daily = (
            frame.groupby("date", sort=True)[return_col]
            .agg(observations="count", realized_variance=lambda values: float(np.sum(values**2)))
            .reset_index()
        )
        daily["realized_variance"] = daily["realized_variance"].clip(lower=1e-12)
        daily["realized_volatility"] = np.sqrt(daily["realized_variance"])
        return daily

    def make_features(self, daily_rv: pd.DataFrame) -> pd.DataFrame:
        """Create HAR-RV features without using same-day or future RV."""

        frame = daily_rv[["date", "realized_variance"]].sort_values("date").reset_index(drop=True)
        lagged = frame["realized_variance"].shift(1)
        features = pd.DataFrame(
            {
                "date": frame["date"],
                "rv_daily": lagged,
                "rv_weekly": lagged.rolling(self.weekly_window, min_periods=1).mean(),
                "rv_monthly": lagged.rolling(self.monthly_window, min_periods=1).mean(),
                "target_rv": frame["realized_variance"],
            }
        )
        return features.dropna().reset_index(drop=True)

    def fit(
        self,
        daily_rv: pd.DataFrame,
        step_mean_return: float = 0.0,
    ) -> "HARRVModel":
        """Fit OLS HAR-RV coefficients to daily realized variance."""

        features = self.make_features(daily_rv)
        if len(features) < 3:
            raise ValueError("HAR-RV fit requires at least 3 feature rows")

        x = features[self.feature_columns].to_numpy(dtype=float)
        y = features["target_rv"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        positive_targets = y[y > 0.0]
        effective_min_variance = self.min_variance
        if positive_targets.size:
            effective_min_variance = max(
                self.min_variance,
                float(np.quantile(positive_targets, 0.01)) * 0.1,
            )

        self.params = HARRVParams(
            intercept=float(coefficients[0]),
            beta_daily=float(coefficients[1]),
            beta_weekly=float(coefficients[2]),
            beta_monthly=float(coefficients[3]),
            min_variance=effective_min_variance,
            step_mean_return=float(step_mean_return),
        )
        history = (
            daily_rv.sort_values("date")["realized_variance"]
            .astype(float)
            .clip(lower=effective_min_variance)
        )
        self.history_ = history.reset_index(drop=True)
        return self

    def forecast_daily_variance(self, horizon_days: int) -> np.ndarray:
        """Forecast future daily realized variance recursively."""

        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        self._require_fitted()
        assert self.params is not None
        assert self.history_ is not None

        history = list(self.history_.to_numpy(dtype=float))
        forecasts: list[float] = []
        for _ in range(horizon_days):
            recent = np.asarray(history, dtype=float)
            daily = recent[-1]
            weekly = float(np.mean(recent[-self.weekly_window :]))
            monthly = float(np.mean(recent[-self.monthly_window :]))
            forecast = (
                self.params.intercept
                + self.params.beta_daily * daily
                + self.params.beta_weekly * weekly
                + self.params.beta_monthly * monthly
            )
            forecast = max(float(forecast), self.params.min_variance)
            forecasts.append(forecast)
            history.append(forecast)
        return np.asarray(forecasts, dtype=float)

    def simulate(
        self,
        start_price: float,
        horizon_steps: int,
        n_paths: int,
        steps_per_day: int,
        seed: int | None = None,
        innovations: str = "normal",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate lognormal paths using HAR daily variance forecasts.

        The daily variance forecast is spread evenly across intraday steps.
        This is a pragmatic first interface; richer intraday seasonality can be
        added behind the same return signature later.
        """

        if innovations != "normal":
            raise NotImplementedError("Only normal innovations are implemented for now")
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        if horizon_steps <= 0 or n_paths <= 0 or steps_per_day <= 0:
            raise ValueError("horizon_steps, n_paths, and steps_per_day must be positive")

        self._require_fitted()
        assert self.params is not None

        horizon_days = int(np.ceil(horizon_steps / steps_per_day))
        daily_variance = self.forecast_daily_variance(horizon_days)
        step_variance = np.repeat(daily_variance / steps_per_day, steps_per_day)[:horizon_steps]
        step_variance = np.maximum(step_variance, self.params.min_variance)

        rng = np.random.default_rng(seed)
        prices = np.empty((n_paths, horizon_steps + 1), dtype=float)
        returns = np.zeros((n_paths, horizon_steps + 1), dtype=float)
        variances = np.full((n_paths, horizon_steps + 1), np.nan, dtype=float)
        prices[:, 0] = float(start_price)

        for step in range(1, horizon_steps + 1):
            raw_return = self.params.step_mean_return + np.sqrt(step_variance[step - 1]) * rng.normal(size=n_paths)
            returns[:, step] = raw_return
            variances[:, step] = step_variance[step - 1]
            prices[:, step] = prices[:, step - 1] * np.exp(raw_return)

        return prices, returns, variances

    def _require_fitted(self) -> None:
        if self.params is None or self.history_ is None:
            raise RuntimeError("HARRVModel must be fitted before forecasting or simulation")
