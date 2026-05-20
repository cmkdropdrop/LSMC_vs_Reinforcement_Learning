from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lsmc_rl.analysis.option_lsmc_report import (
    OptionLSMCReportConfig,
    _aggregate_american_bound_status,
    _evaluation_periods,
    _summarize_american,
    render_report,
)
from lsmc_rl.evaluation import evaluate_american_policy
from lsmc_rl.valuation import AmericanOptionContract, RegressionConfig, value_american_option_lsmc, value_european_option
from lsmc_rl.valuation.common import paths_frame_to_matrices

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "ttf_klines_5m_from_1m.sqlite"


def test_option_report_builds_multiple_evaluation_periods() -> None:
    config = OptionLSMCReportConfig(
        database_path=DB_PATH,
        period_count=3,
        period_lookback_days=120,
    )

    periods = _evaluation_periods(config)

    assert len(periods) == 3
    assert all(period.start < period.end for period in periods)
    assert len({period.name for period in periods}) == 3


def test_option_report_aggregate_tracks_call_candidates_only() -> None:
    valuations = {
        "p1_gjr_garch": {
            "period": "p1",
            "model_type": "gjr_garch",
            "american_call_candidates": {
                "lsmc": {
                    "candidate_label": "LSMC",
                    "raw_replay_below_european": True,
                    "raw_mean_delta_vs_european": -0.1,
                },
                "linear_fitted_q": {
                    "candidate_label": "linear Fitted-Q",
                    "raw_replay_below_european": False,
                    "raw_mean_delta_vs_european": 0.2,
                },
            },
        }
    }

    aggregate = _aggregate_american_bound_status(valuations)

    assert aggregate["american_contract_checks"] == 2
    assert aggregate["raw_replay_below_european_count"] == 1
    assert {row["candidate"] for row in aggregate["bound_checks"]} == {"lsmc", "linear_fitted_q"}
    assert {row["option"] for row in aggregate["bound_checks"]} == {"call"}


def test_option_report_rendering_labels_call_only_rl_candidates() -> None:
    def candidate(label: str) -> dict[str, object]:
        return {
            "candidate_label": label,
            "raw_candidate_value": 1.0,
            "european_value": 1.1,
            "raw_mean_delta_vs_european": -0.1,
            "selected_policy_name": "never_early_exercise",
            "deployment_value": 1.1,
            "selected_mean_delta_vs_european": 0.0,
            "validation_delta_ci95": [-0.2, 0.1],
            "used_candidate": False,
            "raw_early_exercise_probability": 0.4,
            "selected_early_exercise_probability": 0.0,
            "raw_delta_vs_european_ci95": [-0.2, 0.0],
            "selected_delta_vs_european_ci95": [0.0, 0.0],
            "median_regression_r2": 0.5,
        }

    metrics = {
        "config": {
            "symbol": "FRONT",
            "interval": "5m",
            "risk_free_rate": 0.03,
            "strike_moneyness": 1.0,
            "period_count": 1,
            "period_lookback_days": 120,
            "n_paths_if_simulated": 2048,
            "training_paths": 2048,
            "validation_paths": 2048,
            "rl_training_paths": 2048,
            "rl_validation_paths": 2048,
            "kernel_rff_features": 16,
            "kernel_length_scale": 1.0,
            "kernel_feature_seed": 123,
        },
        "valuation_assumption": "Call-only GJR-GARCH test assumption.",
        "aggregate": {
            "raw_replay_below_european_count": 1,
            "american_contract_checks": 3,
            "raw_replay_below_european_share": 1 / 3,
        },
        "plots": {
            "valuation_bars": "valuation_bars.png",
            "american_call_exercise": "american_call_exercise.png",
            "swing_nomination_profile": "swing_nomination_profile.png",
            "swing_value_distribution": "swing_value_distribution.png",
        },
        "models": {
            "p1_gjr_garch": {
                "period": "p1",
                "model_type": "gjr_garch",
                "path_count": 2048,
                "training_path_count": 2048,
                "validation_path_count": 2048,
                "rl_training_path_count": 2048,
                "rl_validation_path_count": 2048,
                "start_price": 50.0,
                "strike": 50.0,
                "american_call_candidates": {
                    "lsmc": candidate("LSMC"),
                    "linear_fitted_q": candidate("linear Fitted-Q"),
                    "kernel_fitted_q": candidate("kernel Fitted-Q"),
                },
                "swing_call": {
                    "price": 2.0,
                    "mean_delta_vs_quota_aware": 0.1,
                    "mean_total_volume": 3.0,
                    "q05_total_volume": 2.0,
                    "q50_total_volume": 3.0,
                    "q95_total_volume": 4.0,
                    "median_regression_r2": 0.2,
                },
            }
        },
    }

    text = render_report(metrics)

    assert "American call" in text
    assert "linear Fitted-Q" in text
    assert "kernel Fitted-Q" in text
    assert "American put" not in text
    assert "HAR-RV simulations are not part of this report" in text


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


def test_american_lsmc_policy_can_be_replayed_on_new_paths() -> None:
    train_prices = np.array([[100.0, 80.0, 90.0]])
    test_prices = np.array([[100.0, 70.0, 95.0], [100.0, 99.0, 60.0]])
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
    )
    result = value_american_option_lsmc(
        train_prices,
        contract,
        RegressionConfig(min_regression_paths=10),
    )

    replay = evaluate_american_policy(test_prices, contract, result.policy)

    assert result.policy.name == "american_lsmc"
    np.testing.assert_allclose(replay.path_values, [30.0, 40.0])
    assert replay.exercise_steps.tolist() == [1, 2]


def test_american_lsmc_replay_decision_does_not_depend_on_future_prices() -> None:
    train_prices = np.array([[100.0, 80.0, 120.0]])
    test_prices = np.array(
        [
            [100.0, 90.0, 50.0],
            [100.0, 90.0, 150.0],
        ]
    )
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.0,
        maturity_step=2,
        exercise_start_step=1,
    )
    result = value_american_option_lsmc(train_prices, contract, RegressionConfig(min_regression_paths=10))

    replay = evaluate_american_policy(test_prices, contract, result.policy)
    step_one = replay.decision_trace.loc[replay.decision_trace["step"] == 1]

    assert step_one["price"].tolist() == [90.0, 90.0]
    assert step_one["exercised"].tolist() == [True, True]
    np.testing.assert_allclose(replay.path_values, [10.0, 10.0])


def test_option_lsmc_summary_keeps_raw_lsmc_failure_visible() -> None:
    contract = AmericanOptionContract(strike=100.0, option_type="put", maturity_step=2)
    training_result = SimpleNamespace(
        price=1.0,
        exercise_steps=np.array([2, 2]),
        regression_diagnostics=pd.DataFrame({"regression_r2": [0.1, 0.2]}),
    )
    raw_result = SimpleNamespace(
        policy_name="american_lsmc",
        path_values=np.array([1.0, 1.0]),
        exercise_steps=np.array([1, 1]),
        contract=contract,
    )
    selected_result = SimpleNamespace(
        policy_name="american_lsmc_validation_selected_deployment",
        path_values=np.array([2.0, 3.0]),
        exercise_steps=np.array([2, 2]),
        contract=contract,
    )
    european_result = SimpleNamespace(
        policy_name="never_early_exercise",
        path_values=np.array([2.0, 3.0]),
    )

    summary = _summarize_american(
        training_result=training_result,
        raw_candidate_result=raw_result,
        selected_result=selected_result,
        european_result=european_result,
        validation_selection={
            "mean_delta": -1.0,
            "bootstrap_mean_delta_ci95": [-2.0, 0.0],
            "selected_policy_name": "never_early_exercise",
            "used_candidate": False,
            "decision_rule": "test rule",
        },
        bootstrap_seed=123,
        n_bootstrap=25,
    )

    assert summary["price"] == 1.0
    assert summary["raw_lsmc_value"] == 1.0
    assert summary["deployment_value"] == 2.5
    assert summary["european_value"] == 2.5
    assert summary["raw_mean_delta_vs_european"] == -1.5
    assert summary["mean_delta_vs_european"] == -1.5
    assert summary["selected_mean_delta_vs_european"] == 0.0
    assert summary["raw_replay_below_european"]
    assert not summary["selected_replay_below_european"]
    assert summary["american_value_bound_status"] == "violated_by_raw_policy_replay"
    assert summary["selected_policy_name"] == "never_early_exercise"
