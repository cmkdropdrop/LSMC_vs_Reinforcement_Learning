from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.analysis.trader_choice_backtest import (
    TraderChoiceBacktestConfig,
    _cost_stress_summary,
    _exercise_execution_cost,
    _historical_paired_diagnostics,
    _max_drawdown,
    _model_price_diagnostics,
    _oracle_american_exercise,
    _realized_garch_variance_path,
    _summarize_results,
    _validation_gate_audit,
)
from lsmc_rl.valuation import AmericanOptionContract
from lsmc_rl.volatility import GJRGARCHModel


def test_oracle_american_put_selects_best_realized_exercise_step() -> None:
    prices = np.array([100.0, 95.0, 80.0, 90.0])
    contract = AmericanOptionContract(
        strike=100.0,
        option_type="put",
        risk_free_rate=0.0,
        maturity_step=3,
        exercise_start_step=1,
    )

    oracle = _oracle_american_exercise(prices, contract)

    assert oracle["exercise_step"] == 2
    assert oracle["exercise_payoff"] == 20.0
    assert oracle["gross_pnl"] == 20.0


def test_max_drawdown_uses_running_cumulative_peak() -> None:
    cumulative = np.array([2.0, 5.0, 3.0, 4.0, -1.0, 6.0])

    assert _max_drawdown(cumulative) == -6.0


def test_realized_garch_variance_path_uses_fitted_state_without_future_prices() -> None:
    returns = np.array([0.002, -0.001, 0.003, -0.002] * 20)
    model = GJRGARCHModel().fit(returns)
    prices = np.array([100.0, 101.0, 99.0, 100.5])

    variances = _realized_garch_variance_path(model, prices)

    assert variances.shape == prices.shape
    assert np.isnan(variances[0])
    assert np.all(np.isfinite(variances[1:]))
    assert np.all(variances[1:] > 0.0)


def test_backtest_summary_reports_validation_deploy_rate() -> None:
    rows = [
        {
            "method": "Selected Linear Fitted-Q",
            "date": "2026-03-01",
            "gross_pnl": 1.0,
            "execution_cost": 0.1,
            "cost_adjusted_pnl": 0.9,
            "model_net_pnl": np.nan,
            "pnl_vs_maturity": 0.5,
            "regret_vs_oracle": 2.0,
            "exercised_early": False,
            "model_price": np.nan,
            "validation_used_candidate": True,
        },
        {
            "method": "Selected Linear Fitted-Q",
            "date": "2026-03-02",
            "gross_pnl": 0.0,
            "execution_cost": 0.0,
            "cost_adjusted_pnl": 0.0,
            "model_net_pnl": np.nan,
            "pnl_vs_maturity": -1.0,
            "regret_vs_oracle": 3.0,
            "exercised_early": True,
            "model_price": np.nan,
            "validation_used_candidate": False,
        },
    ]

    summary = _summarize_results(pd.DataFrame(rows))

    assert summary.loc[0, "validation_deploy_rate"] == 0.5
    assert summary.loc[0, "share_days_beating_maturity"] == 0.5
    assert summary.loc[0, "worst_day_gross_pnl"] == 0.0
    assert summary.loc[0, "total_execution_cost"] == 0.1
    assert summary.loc[0, "total_cost_adjusted_pnl"] == 0.9


def test_exercise_execution_cost_applies_only_to_positive_exercise_payoff() -> None:
    meta = {"exercise_fee_per_unit": 0.25, "slippage_bps": 10.0}

    assert _exercise_execution_cost(0.0, 100.0, meta) == 0.0
    assert np.isclose(_exercise_execution_cost(5.0, 100.0, meta), 0.35)


def test_historical_paired_diagnostics_reports_realized_edge_vs_maturity() -> None:
    rows = [
        {"date": "2026-03-01", "method": "Maturity", "gross_pnl": 1.0},
        {"date": "2026-03-01", "method": "Oracle", "gross_pnl": 2.0},
        {"date": "2026-03-01", "method": "Raw LSMC", "gross_pnl": 1.5},
        {"date": "2026-03-02", "method": "Maturity", "gross_pnl": 3.0},
        {"date": "2026-03-02", "method": "Oracle", "gross_pnl": 4.0},
        {"date": "2026-03-02", "method": "Raw LSMC", "gross_pnl": 2.0},
    ]

    diagnostics = _historical_paired_diagnostics(
        pd.DataFrame(rows),
        TraderChoiceBacktestConfig(n_bootstrap=0),
    )

    raw = diagnostics.loc[diagnostics["method"] == "Raw LSMC"].iloc[0]
    assert raw["total_delta_vs_maturity"] == -0.5
    assert raw["share_days_beating_maturity"] == 0.5
    assert np.isclose(raw["oracle_capture_rate"], 3.5 / 6.0)


def test_validation_gate_audit_compares_rejections_with_realized_raw_delta() -> None:
    rows = [
        {
            "date": "2026-03-01",
            "method": "Raw LSMC",
            "pnl_vs_maturity": -2.0,
        },
        {
            "date": "2026-03-01",
            "method": "Selected LSMC",
            "pnl_vs_maturity": 0.0,
            "validation_used_candidate": False,
            "validation_mean_delta": -0.1,
            "validation_ci_low": -0.3,
            "validation_cvar_5_delta": -1.0,
        },
        {
            "date": "2026-03-02",
            "method": "Raw LSMC",
            "pnl_vs_maturity": 0.5,
        },
        {
            "date": "2026-03-02",
            "method": "Selected LSMC",
            "pnl_vs_maturity": 0.0,
            "validation_used_candidate": False,
            "validation_mean_delta": -0.2,
            "validation_ci_low": -0.4,
            "validation_cvar_5_delta": -1.2,
        },
    ]

    audit = _validation_gate_audit(pd.DataFrame(rows))

    lsmc = audit.loc[audit["candidate_method"] == "Raw LSMC"].iloc[0]
    assert lsmc["deployed_days"] == 0
    assert lsmc["rejected_good_days"] == 1
    assert lsmc["rejected_bad_days"] == 1
    assert lsmc["validation_saved_pnl_vs_raw"] == 1.5


def test_model_price_diagnostics_treats_model_value_as_hypothetical_premium() -> None:
    rows = [
        {"date": "2026-03-01", "method": "Maturity", "gross_pnl": 1.0, "model_price": np.nan},
        {"date": "2026-03-01", "method": "Raw LSMC", "gross_pnl": 1.0, "model_price": 2.0},
        {"date": "2026-03-02", "method": "Maturity", "gross_pnl": 2.0, "model_price": np.nan},
        {"date": "2026-03-02", "method": "Raw LSMC", "gross_pnl": 1.5, "model_price": 0.5},
    ]

    diagnostics = _model_price_diagnostics(pd.DataFrame(rows))

    raw = diagnostics.loc[diagnostics["method"] == "Raw LSMC"].iloc[0]
    assert raw["total_model_price"] == 2.5
    assert raw["total_realized_gross_pnl"] == 2.5
    assert raw["total_model_net_pnl"] == 0.0
    assert raw["overvalued_day_rate"] == 0.5
    assert raw["total_model_shortfall"] == 1.0


def test_cost_stress_summary_reprices_cashflows_against_stressed_maturity() -> None:
    rows = [
        {
            "date": "2026-03-01",
            "method": "Maturity",
            "gross_pnl": 1.0,
            "exercise_payoff": 1.0,
            "exercise_price": 100.0,
        },
        {
            "date": "2026-03-01",
            "method": "Raw LSMC",
            "gross_pnl": 1.5,
            "exercise_payoff": 1.5,
            "exercise_price": 100.0,
        },
    ]
    config = TraderChoiceBacktestConfig(
        cost_stress_slippage_bps=(10.0,),
        cost_stress_exercise_fee_per_unit=(0.0,),
    )

    stress = _cost_stress_summary(pd.DataFrame(rows), config)

    raw = stress.loc[(stress["method"] == "Raw LSMC") & (stress["slippage_bps"] == 10.0)].iloc[0]
    assert np.isclose(raw["total_execution_cost"], 0.1)
    assert np.isclose(raw["total_cost_adjusted_pnl"], 1.4)
    assert np.isclose(raw["total_pnl_vs_maturity"], 0.5)
