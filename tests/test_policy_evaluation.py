from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd

from lsmc_rl.evaluation import (
    AmericanPolicyState,
    ImmediateIntrinsicExercisePolicy,
    NeverEarlyExercisePolicy,
    NeverExerciseSwingPolicy,
    PositiveMarginSwingPolicy,
    QuotaAwareSwingPolicy,
    SwingPolicyState,
    ValidationSelectedAmericanPolicy,
    evaluate_american_policy,
    evaluate_swing_policy,
    paired_policy_metrics,
    path_value_summary,
    select_american_policy_by_validation,
)
from lsmc_rl.valuation import AmericanOptionContract, SwingOptionContract, value_european_option


def test_policy_interface_smoke() -> None:
    american_contract = AmericanOptionContract(strike=100.0, option_type="put", maturity_step=2)
    american_state = AmericanPolicyState(
        step=1,
        time=None,
        current_price=90.0,
        variance=None,
        volatility=None,
        maturity_step=2,
        remaining_steps=1,
        intrinsic_value=10.0,
        contract=american_contract,
    )
    assert not NeverEarlyExercisePolicy().decide_exercise(american_state)
    assert ImmediateIntrinsicExercisePolicy().decide_exercise(american_state)

    swing_contract = SwingOptionContract(strike=100.0, maturity_step=2, max_total_volume=2.0)
    swing_state = SwingPolicyState(
        step=1,
        time=None,
        current_price=105.0,
        variance=None,
        volatility=None,
        maturity_step=2,
        remaining_steps=1,
        remaining_exercise_dates=2,
        remaining_volume=2.0,
        exercised_volume=0.0,
        margin=5.0,
        contract=swing_contract,
    )
    assert NeverExerciseSwingPolicy().nominate(swing_state) == 0.0
    assert PositiveMarginSwingPolicy().nominate(swing_state) == 1.0


def test_american_never_exercise_matches_european_payoff() -> None:
    prices = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 105.0, 115.0],
            [100.0, 99.0, 80.0],
        ]
    )
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.02,
        time_step_years=1 / 252,
        maturity_step=2,
    )

    result = evaluate_american_policy(prices, contract, NeverEarlyExercisePolicy())
    _, _, european_path_values = value_european_option(prices, contract)

    np.testing.assert_allclose(result.path_values, european_path_values)
    np.testing.assert_array_equal(result.exercise_steps, np.full(3, 2))


def test_validation_selected_american_policy_uses_fallback_when_candidate_fails_validation() -> None:
    class BadPolicy:
        name = "bad_policy"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            return bool(state.intrinsic_value > 0.0)

    validation_prices = np.array(
        [
            [100.0, 90.0, 80.0],
            [100.0, 95.0, 70.0],
            [100.0, 105.0, 60.0],
            [100.0, 99.0, 50.0],
        ]
    )
    test_prices = np.array([[100.0, 90.0, 50.0]])
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)

    selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        BadPolicy(),
        bootstrap_seed=7,
        n_bootstrap=100,
    )
    replay = evaluate_american_policy(test_prices, contract, selection.policy)

    assert isinstance(selection.policy, ValidationSelectedAmericanPolicy)
    assert not selection.used_candidate
    assert selection.policy.name == "american_lsmc_validation_selected_deployment"
    assert selection.selected_policy_name == "never_early_exercise"
    assert replay.policy_name == "american_lsmc_validation_selected_deployment"
    np.testing.assert_allclose(replay.path_values, [50.0])
    assert replay.exercise_steps.tolist() == [2]


def test_validation_selection_uses_validation_paths_not_test_paths() -> None:
    class BadOnValidationGoodOnTest:
        name = "bad_on_validation_good_on_test"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            return bool(state.intrinsic_value > 0.0)

    validation_prices = np.array(
        [
            [100.0, 90.0, 70.0],
            [100.0, 95.0, 60.0],
            [100.0, 99.0, 50.0],
        ]
    )
    test_prices = np.array([[100.0, 90.0, 100.0]])
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)

    selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        BadOnValidationGoodOnTest(),
        bootstrap_seed=11,
        n_bootstrap=100,
    )
    replay = evaluate_american_policy(test_prices, contract, selection.policy)

    assert not selection.used_candidate
    assert selection.selected_policy_name == "never_early_exercise"
    np.testing.assert_allclose(replay.path_values, [0.0])
    assert replay.exercise_steps.tolist() == [2]


def test_validation_selection_is_deterministic_with_fixed_seed() -> None:
    class MixedPolicy:
        name = "mixed_policy"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            return bool(state.intrinsic_value > 0.0)

    validation_prices = np.array(
        [
            [100.0, 90.0, 80.0],
            [100.0, 95.0, 100.0],
            [100.0, 110.0, 90.0],
            [100.0, 99.0, 70.0],
        ]
    )
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)

    first = select_american_policy_by_validation(
        validation_prices,
        contract,
        MixedPolicy(),
        bootstrap_seed=123,
        n_bootstrap=250,
    )
    second = select_american_policy_by_validation(
        validation_prices,
        contract,
        MixedPolicy(),
        bootstrap_seed=123,
        n_bootstrap=250,
    )

    assert first.used_candidate == second.used_candidate
    assert first.selected_policy_name == second.selected_policy_name
    assert first.validation_metrics["bootstrap_mean_delta_ci95"] == second.validation_metrics[
        "bootstrap_mean_delta_ci95"
    ]


def test_validation_selection_can_use_variance_paths_for_candidate_state() -> None:
    class VarianceAwarePolicy:
        name = "variance_aware"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            return bool(state.variance is not None and state.variance > 0.0 and state.intrinsic_value > 0.0)

    validation_prices = np.array([[100.0, 90.0, 100.0]])
    validation_variances = np.array([[np.nan, 0.04, 0.04]])
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)

    selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        VarianceAwarePolicy(),
        variance_paths=validation_variances,
        n_bootstrap=0,
        selected_name="variance_selected",
    )

    assert selection.used_candidate
    assert selection.policy.name == "variance_selected"
    assert selection.selected_policy_name == "variance_aware"


def test_validation_selection_can_reject_positive_mean_with_bad_tail() -> None:
    class ImmediatePolicy:
        name = "immediate_test"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            return bool(state.intrinsic_value > 0.0)

    good_paths = np.tile(np.array([[100.0, 80.0, 100.0]]), (10, 1))
    bad_path = np.array([[100.0, 99.0, 0.01]])
    validation_prices = np.vstack([good_paths, bad_path])
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)

    selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        ImmediatePolicy(),
        n_bootstrap=0,
        min_ci_low=0.0,
        min_cvar_5_delta=0.0,
        selected_name="tail_gated",
    )

    assert selection.validation_metrics["mean_delta"] > 0.0
    assert selection.validation_metrics["cvar_5_delta"] < 0.0
    assert not selection.used_candidate
    assert not selection.validation_metrics["deployment_gate"]["cvar_gate_passed"]


def test_validation_selected_policy_does_not_apply_pathwise_european_floor() -> None:
    prices = np.array(
        [
            [100.0, 90.0, 50.0],
            [100.0, 90.0, 150.0],
        ]
    )
    contract = AmericanOptionContract(strike=100.0, option_type="put", risk_free_rate=0.0, maturity_step=2)
    selected_candidate = ValidationSelectedAmericanPolicy(
        candidate=ImmediateIntrinsicExercisePolicy(),
        use_candidate=True,
    )

    replay = evaluate_american_policy(prices, contract, selected_candidate)
    _, _, european_values = value_european_option(prices, contract)

    np.testing.assert_allclose(replay.path_values, [10.0, 10.0])
    assert np.any(replay.path_values < european_values)
    assert not np.allclose(replay.path_values, np.maximum(replay.path_values, european_values))


def test_swing_baselines_respect_volume_constraints_in_simple_cases() -> None:
    prices = np.array(
        [
            [100.0, 105.0, 106.0],
            [100.0, 104.0, 103.0],
        ]
    )
    relaxed_contract = SwingOptionContract(
        strike=100.0,
        maturity_step=2,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        min_total_volume=0.0,
        max_total_volume=2.0,
        volume_step=1.0,
    )

    for policy in (NeverExerciseSwingPolicy(), PositiveMarginSwingPolicy()):
        result = evaluate_swing_policy(prices, relaxed_contract, policy)
        assert result.path_results["constraint_violations"].sum() == 0
        assert (result.path_results["total_volume"] <= relaxed_contract.max_total_volume).all()

    quota_contract = SwingOptionContract(
        strike=100.0,
        maturity_step=2,
        exercise_start_step=1,
        max_exercise_volume=1.0,
        min_total_volume=1.0,
        max_total_volume=2.0,
        volume_step=1.0,
        enforce_min_total_volume=True,
    )
    quota_result = evaluate_swing_policy(prices, quota_contract, QuotaAwareSwingPolicy())
    assert quota_result.path_results["constraint_violations"].sum() == 0
    assert (quota_result.path_results["total_volume"] >= quota_contract.min_total_volume).all()


def test_paired_metrics_compute_expected_deltas() -> None:
    metrics = paired_policy_metrics(
        np.array([2.0, 4.0, 6.0]),
        np.array([1.0, 5.0, 3.0]),
        bootstrap_seed=7,
        n_bootstrap=100,
    )

    assert metrics["mean_delta"] == 1.0
    assert metrics["median_delta"] == 1.0
    assert metrics["share_delta_positive"] == 2 / 3
    assert metrics["q50_delta"] == 1.0
    assert metrics["n_paths"] == 3


def test_path_value_summary_includes_tail_and_risk_metrics() -> None:
    summary = path_value_summary(np.array([-2.0, -1.0, 1.0, 4.0]))

    assert summary["std"] > 0.0
    assert summary["cvar_5"] == -2.0
    assert summary["probability_positive"] == 0.5
    assert summary["probability_loss"] == 0.5
    assert summary["mean_loss"] == -1.5
    assert "sharpe_like" in summary
    assert "sortino_like" in summary


def test_bootstrap_ci_is_deterministic_with_fixed_seed() -> None:
    values_a = np.array([1.0, 2.0, 3.0, 4.0])
    values_b = np.array([0.5, 1.5, 3.5, 2.5])

    first = paired_policy_metrics(values_a, values_b, bootstrap_seed=123, n_bootstrap=250)
    second = paired_policy_metrics(values_a, values_b, bootstrap_seed=123, n_bootstrap=250)

    assert first["bootstrap_mean_delta_ci95"] == second["bootstrap_mean_delta_ci95"]


def test_evaluator_state_exposes_no_future_prices() -> None:
    allowed_fields = {field.name for field in fields(AmericanPolicyState)}
    seen_states: list[AmericanPolicyState] = []

    class RecordingPolicy:
        name = "recording"

        def decide_exercise(self, state: AmericanPolicyState) -> bool:
            seen_states.append(state)
            assert set(state.__dict__) == allowed_fields
            return False

    prices = np.array([[100.0, 90.0, 80.0], [100.0, 110.0, 120.0]])
    times = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "path": np.repeat([0, 1], 3),
            "step": np.tile([0, 1, 2], 2),
            "time": np.tile(times.to_numpy(), 2),
            "price": prices.reshape(-1),
        }
    )
    contract = AmericanOptionContract(strike=100.0, option_type="put", maturity_step=2)

    evaluate_american_policy(frame, contract, RecordingPolicy())

    assert [state.current_price for state in seen_states] == [90.0, 110.0]
    assert all(not hasattr(state, "future_prices") for state in seen_states)
    assert all(not hasattr(state, "path_prices") for state in seen_states)
