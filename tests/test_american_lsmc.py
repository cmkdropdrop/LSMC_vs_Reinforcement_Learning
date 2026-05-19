from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.valuation import AmericanOptionContract, RegressionConfig, value_american_option_lsmc, value_european_option
from lsmc_rl.valuation.common import paths_frame_to_matrices


def test_paths_frame_to_matrices_validates_complete_grid() -> None:
    frame = pd.DataFrame(
        {
            "path": [1, 1, 0, 0],
            "step": [0, 1, 0, 1],
            "price": [10.0, 11.0, 10.0, 9.5],
            "variance": [np.nan, 0.01, np.nan, 0.01],
        }
    )

    matrices = paths_frame_to_matrices(frame)

    assert matrices.prices.shape == (2, 2)
    np.testing.assert_allclose(matrices.prices[0], [10.0, 9.5])
    assert matrices.variances is not None


def test_american_put_is_at_least_matching_european_on_same_paths() -> None:
    prices = np.array(
        [
            [100.0, 90.0, 85.0, 80.0],
            [100.0, 92.0, 88.0, 70.0],
            [100.0, 105.0, 95.0, 90.0],
            [100.0, 110.0, 120.0, 130.0],
            [100.0, 98.0, 99.0, 97.0],
            [100.0, 80.0, 82.0, 84.0],
        ]
    )
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.01, time_step_years=1 / 252)
    regression = RegressionConfig(min_regression_paths=2, ridge_alpha=1e-8)

    result = value_american_option_lsmc(prices, contract, regression)
    european_value, _, _ = value_european_option(prices, contract)

    assert result.price >= european_value
    assert result.european_value == european_value
    assert set(result.exercise_profile.columns) >= {"step", "exercise_count", "exercise_probability"}
    assert np.all((result.exercise_steps >= 1) & (result.exercise_steps <= 3))


def test_american_lsmc_is_deterministic_for_fixed_paths() -> None:
    rng = np.random.default_rng(123)
    returns = 0.001 + 0.02 * rng.normal(size=(80, 6))
    prices = np.column_stack([np.full(80, 100.0), 100.0 * np.exp(np.cumsum(returns, axis=1))])
    contract = AmericanOptionContract(strike=101.0, option_type="call", risk_free_rate=0.0, time_step_years=1 / 252)

    first = value_american_option_lsmc(prices, contract)
    second = value_american_option_lsmc(prices, contract)

    assert first.price == second.price
    np.testing.assert_array_equal(first.exercise_steps, second.exercise_steps)
    np.testing.assert_allclose(first.path_values, second.path_values)
