from __future__ import annotations

import numpy as np

from lsmc_rl.evaluation import evaluate_american_policy
from lsmc_rl.rl import FittedQConfig, train_american_fitted_q, value_american_option_fitted_q
from lsmc_rl.valuation import AmericanOptionContract


def test_fitted_q_exercises_profitable_american_put_early() -> None:
    prices = np.array([[100.0, 80.0, 90.0]])
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
    )
    config = FittedQConfig(min_regression_paths=10)

    result = train_american_fitted_q(prices, contract, config)

    assert result.price == 20.0
    assert result.policy.name == "american_fitted_q"
    assert result.exercise_steps.tolist() == [1]
    assert result.exercise_payoffs.tolist() == [20.0]


def test_fitted_q_policy_can_be_replayed_on_new_paths() -> None:
    train_prices = np.array([[100.0, 80.0, 90.0]])
    test_prices = np.array([[100.0, 70.0, 95.0], [100.0, 99.0, 60.0]])
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
    )
    result = train_american_fitted_q(train_prices, contract, FittedQConfig(min_regression_paths=10))

    replay = evaluate_american_policy(test_prices, contract, result.policy)

    np.testing.assert_allclose(replay.path_values, [30.0, 40.0])
    assert replay.exercise_steps.tolist() == [1, 2]


def test_fitted_q_training_is_deterministic_for_fixed_paths() -> None:
    rng = np.random.default_rng(123)
    returns = 0.001 + 0.02 * rng.normal(size=(80, 5))
    prices = np.column_stack([np.full(80, 100.0), 100.0 * np.exp(np.cumsum(returns, axis=1))])
    contract = AmericanOptionContract(strike=101.0, option_type="call", risk_free_rate=0.0, maturity_step=5)
    config = FittedQConfig(min_regression_paths=8, ridge_alpha=1e-8)

    first = value_american_option_fitted_q(prices, contract, config)
    second = value_american_option_fitted_q(prices, contract, config)

    assert first.price == second.price
    np.testing.assert_array_equal(first.exercise_steps, second.exercise_steps)
    np.testing.assert_allclose(first.path_values, second.path_values)
    assert set(first.training_diagnostics.columns) >= {"step", "status", "regression_rows", "exercised_paths"}
    assert set(first.training_diagnostics.columns) >= {
        "mean_abs_bellman_residual",
        "negative_continuation_share_before_clip",
    }
