from __future__ import annotations

import numpy as np

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
