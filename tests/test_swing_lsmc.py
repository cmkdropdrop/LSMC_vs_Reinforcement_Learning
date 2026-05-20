from __future__ import annotations

import numpy as np

from lsmc_rl.evaluation import evaluate_swing_policy
from lsmc_rl.valuation import RegressionConfig, SwingOptionContract, value_swing_option_lsmc


def test_swing_exercises_highest_margins_on_deterministic_path() -> None:
    prices = np.array([[100.0, 110.0, 120.0, 130.0]])
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.0,
        maturity_step=3,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )
    regression = RegressionConfig(min_regression_paths=100, clip_negative_continuation=False)

    result = value_swing_option_lsmc(prices, contract, regression)
    exercised = result.policy.loc[result.policy["action_volume"] > 0.0, ["step", "action_volume"]]

    assert result.price == 50.0
    assert exercised["step"].tolist() == [2, 3]
    assert exercised["action_volume"].tolist() == [1.0, 1.0]
    assert result.volume_summary["mean_total_volume"] == 2.0


def test_swing_has_zero_value_when_all_margins_negative_and_no_min_volume() -> None:
    prices = np.array(
        [
            [100.0, 95.0, 94.0],
            [100.0, 90.0, 91.0],
        ]
    )
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )

    result = value_swing_option_lsmc(prices, contract)

    assert result.price == 0.0
    assert result.policy["action_volume"].sum() == 0.0
    np.testing.assert_allclose(result.path_values, np.zeros(2))


def test_swing_min_total_volume_can_force_exercise() -> None:
    prices = np.array([[100.0, 95.0, 96.0]])
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        min_total_volume=1.0,
        max_total_volume=1.0,
        volume_step=1.0,
        shortfall_penalty_per_unit=100.0,
        enforce_min_total_volume=True,
    )
    regression = RegressionConfig(min_regression_paths=100, clip_negative_continuation=False)

    result = value_swing_option_lsmc(prices, contract, regression)

    assert result.price == -4.0
    assert result.policy.loc[result.policy["action_volume"] > 0.0, "step"].tolist() == [2]


def test_swing_lsmc_policy_model_can_be_replayed_on_new_paths() -> None:
    train_prices = np.array([[100.0, 110.0, 120.0, 130.0]])
    test_prices = np.array([[100.0, 108.0, 121.0, 129.0]])
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.0,
        maturity_step=3,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )
    result = value_swing_option_lsmc(
        train_prices,
        contract,
        RegressionConfig(min_regression_paths=100, clip_negative_continuation=False),
    )

    replay = evaluate_swing_policy(test_prices, contract, result.policy_model)
    exercised = replay.decision_trace.loc[replay.decision_trace["action_volume"] > 0.0, "step"]

    assert result.policy_model.name == "swing_lsmc"
    assert exercised.tolist() == [2, 3]
    np.testing.assert_allclose(replay.path_values, [50.0])


def test_swing_lsmc_result_values_are_frozen_policy_replay_values() -> None:
    rng = np.random.default_rng(42)
    returns = 0.01 * rng.normal(size=(80, 5))
    prices = np.column_stack([np.full(80, 100.0), 100.0 * np.exp(np.cumsum(returns, axis=1))])
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.01,
        time_step_years=1 / 252,
        maturity_step=5,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )
    result = value_swing_option_lsmc(
        prices,
        contract,
        RegressionConfig(min_regression_paths=5, itm_only=False, clip_negative_continuation=False),
    )

    replay = evaluate_swing_policy(prices, contract, result.policy_model)

    np.testing.assert_allclose(result.path_values, replay.path_values)
    assert result.price == float(np.mean(replay.path_values))


def test_swing_lsmc_replay_first_decision_does_not_depend_on_future_prices() -> None:
    train_prices = np.array([[100.0, 110.0, 120.0, 130.0]])
    test_prices = np.array(
        [
            [100.0, 108.0, 80.0, 80.0],
            [100.0, 108.0, 140.0, 140.0],
        ]
    )
    contract = SwingOptionContract(
        strike=100.0,
        risk_free_rate=0.0,
        maturity_step=3,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )
    result = value_swing_option_lsmc(
        train_prices,
        contract,
        RegressionConfig(min_regression_paths=100, clip_negative_continuation=False),
    )

    replay = evaluate_swing_policy(test_prices, contract, result.policy_model)
    first_step = replay.decision_trace.loc[replay.decision_trace["step"] == 1]

    assert first_step["price"].tolist() == [108.0, 108.0]
    assert first_step["action_volume"].nunique() == 1
