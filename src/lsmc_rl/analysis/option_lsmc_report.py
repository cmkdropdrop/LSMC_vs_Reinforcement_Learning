"""Run American-call and swing valuation diagnostics on GJR-GARCH paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc_rl.data import DataQualityReport, load_market_data
from lsmc_rl.evaluation import (
    NeverEarlyExercisePolicy,
    PositiveMarginSwingPolicy,
    QuotaAwareSwingPolicy,
    evaluate_american_policy,
    evaluate_swing_policy,
    paired_policy_metrics,
    path_value_summary,
    select_american_policy_by_validation,
)
from lsmc_rl.rl import (
    FittedQConfig,
    KernelFittedQConfig,
    train_american_fitted_q,
    train_american_kernel_fitted_q,
)
from lsmc_rl.simulation.paths import PathSimulationConfig, paths_to_frame, run_simulation
from lsmc_rl.volatility import GJRGARCHModel
from lsmc_rl.valuation import (
    AmericanLSMCResult,
    AmericanOptionContract,
    RegressionConfig,
    SwingLSMCResult,
    SwingOptionContract,
    value_american_option_lsmc,
    value_swing_option_lsmc,
)
from lsmc_rl.valuation.common import json_ready, paths_frame_to_matrices


DEFAULT_PATH_FILES = {
    "gjr_garch": Path("outputs/mc_paths_front_gjr_garch.csv"),
}


@dataclass(frozen=True)
class OptionLSMCReportConfig:
    database_path: Path = Path("ttf_klines_5m_from_1m.sqlite")
    symbol: str = "FRONT"
    interval: str = "5m"
    output_dir: Path = Path("outputs/option_lsmc_report_front")
    risk_free_rate: float = 0.03
    strike_moneyness: float = 1.0
    n_paths_if_simulated: int = 2048
    training_paths: int = 2048
    validation_paths: int = 2048
    rl_training_paths: int = 2048
    rl_validation_paths: int = 2048
    horizon_steps: int = 288
    seed: int = 20260519
    training_seed: int = 20260522
    validation_seed: int = 20260523
    rl_training_seed: int = 20260521
    rl_validation_seed: int = 20260524
    bootstrap_seed: int = 20260520
    n_bootstrap: int = 2000
    kernel_rff_features: int = 96
    kernel_length_scale: float = 1.0
    kernel_feature_seed: int = 20260522
    steps_per_day: int = 288
    period_count: int = 4
    period_lookback_days: int = 120
    use_existing_paths: bool = False


@dataclass(frozen=True)
class EvaluationPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def run_report(config: OptionLSMCReportConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    periods = _evaluation_periods(config)

    regression = RegressionConfig(
        degree=3,
        ridge_alpha=1e-5,
        itm_only=True,
        min_regression_paths=15,
        include_log_moneyness=True,
        include_intrinsic=True,
        include_variance=True,
    )
    swing_regression = RegressionConfig(
        degree=3,
        ridge_alpha=1e-5,
        itm_only=False,
        min_regression_paths=20,
        include_log_moneyness=True,
        include_intrinsic=True,
        include_variance=True,
        clip_negative_continuation=False,
    )
    fitted_q_config = FittedQConfig(
        degree=3,
        ridge_alpha=1e-5,
        min_regression_paths=20,
        fit_itm_only=False,
        include_log_moneyness=True,
        include_intrinsic=True,
        include_variance=True,
        include_time_features=True,
        clip_negative_continuation=True,
        exercise_only_itm=True,
    )
    kernel_fitted_q_config = KernelFittedQConfig(
        n_rff_features=config.kernel_rff_features,
        length_scale=config.kernel_length_scale,
        ridge_alpha=1e-5,
        min_regression_paths=30,
        fit_itm_only=False,
        include_linear_features=True,
        include_variance=True,
        clip_negative_continuation=True,
        exercise_only_itm=True,
        feature_seed=config.kernel_feature_seed,
    )

    valuations: dict[str, Any] = {}
    american_plot_results: dict[str, dict[str, Any]] = {}
    swing_plot_results: dict[str, dict[str, Any]] = {}

    for period_index, period in enumerate(periods):
        for model in ("gjr_garch",):
            path_sets = _simulate_period_path_sets(model, period, period_index, config)
            frame = path_sets["evaluation"]
            training_frame = path_sets["training"]
            validation_frame = path_sets["validation"]
            rl_training_frame = path_sets["rl_training"]
            rl_validation_frame = path_sets["rl_validation"]
            data_quality = path_sets["data_quality"]
            scenario_key = f"{period.name}_{model}"

            matrices = paths_frame_to_matrices(frame)
            start_price = float(np.median(matrices.prices[:, 0]))
            strike = start_price * config.strike_moneyness
            maturity = int(matrices.prices.shape[1] - 1)
            time_step_years = _infer_time_step_years(frame)
            training_matrices = paths_frame_to_matrices(training_frame)
            validation_matrices = paths_frame_to_matrices(validation_frame)
            rl_training_matrices = paths_frame_to_matrices(rl_training_frame)
            rl_validation_matrices = paths_frame_to_matrices(rl_validation_frame)

            call_contract = AmericanOptionContract(
                strike=strike,
                option_type="call",
                risk_free_rate=config.risk_free_rate,
                time_step_years=time_step_years,
                maturity_step=maturity,
                exercise_start_step=1,
                exercise_step_interval=1,
            )
            swing_contract = SwingOptionContract(
                strike=strike,
                payoff_type="call",
                risk_free_rate=config.risk_free_rate,
                time_step_years=time_step_years,
                maturity_step=maturity,
                exercise_start_step=1,
                exercise_step_interval=12,
                min_exercise_volume=0.0,
                max_exercise_volume=1.0,
                min_total_volume=2.0,
                max_total_volume=6.0,
                volume_step=1.0,
                variable_cost_per_unit=0.0,
                shortfall_penalty_per_unit=2.0 * strike,
                enforce_min_total_volume=True,
            )

            call_train = value_american_option_lsmc(training_frame, call_contract, regression)
            call_fitted_q_train = train_american_fitted_q(rl_training_frame, call_contract, fitted_q_config)
            call_kernel_q_train = train_american_kernel_fitted_q(
                rl_training_frame,
                call_contract,
                kernel_fitted_q_config,
            )
            swing_train = value_swing_option_lsmc(training_frame, swing_contract, swing_regression)

            european_policy = NeverEarlyExercisePolicy()
            call_selection = select_american_policy_by_validation(
                validation_frame,
                call_contract,
                call_train.policy,
                baseline_policy=european_policy,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
                min_ci_low=0.0,
            )
            call_fitted_q_selection = select_american_policy_by_validation(
                rl_validation_frame,
                call_contract,
                call_fitted_q_train.policy,
                baseline_policy=european_policy,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
                min_ci_low=0.0,
                selected_name="american_fitted_q_validation_selected_deployment",
            )
            call_kernel_q_selection = select_american_policy_by_validation(
                rl_validation_frame,
                call_contract,
                call_kernel_q_train.policy,
                baseline_policy=european_policy,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
                min_ci_low=0.0,
                selected_name="american_kernel_fitted_q_validation_selected_deployment",
            )
            call_european = evaluate_american_policy(frame, call_contract, european_policy)
            call_raw_lsmc = evaluate_american_policy(frame, call_contract, call_train.policy)
            call_selected_lsmc = evaluate_american_policy(frame, call_contract, call_selection.policy)
            call_fitted_q = evaluate_american_policy(frame, call_contract, call_fitted_q_train.policy)
            call_selected_fitted_q = evaluate_american_policy(frame, call_contract, call_fitted_q_selection.policy)
            call_kernel_q = evaluate_american_policy(frame, call_contract, call_kernel_q_train.policy)
            call_selected_kernel_q = evaluate_american_policy(frame, call_contract, call_kernel_q_selection.policy)

            swing_lsmc = evaluate_swing_policy(frame, swing_contract, swing_train.policy_model)
            swing_positive = evaluate_swing_policy(frame, swing_contract, PositiveMarginSwingPolicy())
            swing_quota = evaluate_swing_policy(frame, swing_contract, QuotaAwareSwingPolicy())

            american_call_candidates = {
                "lsmc": _summarize_american(
                    training_result=call_train,
                    raw_candidate_result=call_raw_lsmc,
                    selected_result=call_selected_lsmc,
                    european_result=call_european,
                    validation_selection=call_selection.validation_metrics,
                    bootstrap_seed=config.bootstrap_seed,
                    n_bootstrap=config.n_bootstrap,
                    candidate_label="LSMC",
                ),
                "linear_fitted_q": _summarize_american(
                    training_result=call_fitted_q_train,
                    raw_candidate_result=call_fitted_q,
                    selected_result=call_selected_fitted_q,
                    european_result=call_european,
                    validation_selection=call_fitted_q_selection.validation_metrics,
                    bootstrap_seed=config.bootstrap_seed,
                    n_bootstrap=config.n_bootstrap,
                    candidate_label="linear Fitted-Q",
                ),
                "kernel_fitted_q": _summarize_american(
                    training_result=call_kernel_q_train,
                    raw_candidate_result=call_kernel_q,
                    selected_result=call_selected_kernel_q,
                    european_result=call_european,
                    validation_selection=call_kernel_q_selection.validation_metrics,
                    bootstrap_seed=config.bootstrap_seed,
                    n_bootstrap=config.n_bootstrap,
                    candidate_label="kernel Fitted-Q",
                ),
            }
            american_plot_results[scenario_key] = {
                "call_european": call_european,
                "call_raw_lsmc": call_raw_lsmc,
                "call_selected_lsmc": call_selected_lsmc,
                "call_fitted_q": call_fitted_q,
                "call_selected_fitted_q": call_selected_fitted_q,
                "call_kernel_q": call_kernel_q,
                "call_selected_kernel_q": call_selected_kernel_q,
            }
            swing_plot_results[scenario_key] = {
                "lsmc": swing_lsmc,
                "positive_margin": swing_positive,
                "quota_aware": swing_quota,
            }
            valuations[scenario_key] = {
                "period": period.name,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "model_type": model,
                "path_source": "generated_in_memory",
                "training_path_source": "generated_in_memory",
                "validation_path_source": "generated_in_memory",
                "path_count": int(matrices.prices.shape[0]),
                "training_path_count": int(training_matrices.prices.shape[0]),
                "validation_path_count": int(validation_matrices.prices.shape[0]),
                "rl_training_path_count": int(rl_training_matrices.prices.shape[0]),
                "rl_validation_path_count": int(rl_validation_matrices.prices.shape[0]),
                "horizon_steps": maturity,
                "historical_observations": int(data_quality.rows),
                "start_price": start_price,
                "strike": strike,
                "time_step_years": time_step_years,
                "american_call": american_call_candidates["lsmc"],
                "american_call_candidates": american_call_candidates,
                "swing_call": _summarize_swing(
                    training_result=swing_train,
                    evaluation_result=swing_lsmc,
                    positive_margin_result=swing_positive,
                    quota_result=swing_quota,
                    bootstrap_seed=config.bootstrap_seed,
                    n_bootstrap=config.n_bootstrap,
                ),
            }

    plot_paths = create_plots(config.output_dir, american_plot_results, swing_plot_results)
    aggregate = _aggregate_american_bound_status(valuations)
    metrics = {
        "config": {
            "database_path": str(config.database_path),
            "symbol": config.symbol,
            "interval": config.interval,
            "risk_free_rate": config.risk_free_rate,
            "strike_moneyness": config.strike_moneyness,
            "use_existing_paths": config.use_existing_paths,
            "n_paths_if_simulated": config.n_paths_if_simulated,
            "training_paths": config.training_paths,
            "validation_paths": config.validation_paths,
            "rl_training_paths": config.rl_training_paths,
            "rl_validation_paths": config.rl_validation_paths,
            "horizon_steps": config.horizon_steps,
            "seed": config.seed,
            "training_seed": config.training_seed,
            "validation_seed": config.validation_seed,
            "rl_training_seed": config.rl_training_seed,
            "rl_validation_seed": config.rl_validation_seed,
            "bootstrap_seed": config.bootstrap_seed,
            "n_bootstrap": config.n_bootstrap,
            "kernel_rff_features": config.kernel_rff_features,
            "kernel_length_scale": config.kernel_length_scale,
            "kernel_feature_seed": config.kernel_feature_seed,
            "steps_per_day": config.steps_per_day,
            "period_count": config.period_count,
            "period_lookback_days": config.period_lookback_days,
        },
        "valuation_assumption": (
            "Values are frozen-policy diagnostics under GJR-GARCH simulated path measures. "
            "Each row uses an independent historical calibration window and at least 2048 "
            "evaluation, training, and validation paths by default. Raw LSMC and RL/ADP "
            "American-call candidates are trained on independent generated paths and replayed "
            "on separate evaluation paths; selected American deployment policies are "
            "validation-chosen wrappers around those candidates or the never-exercise baseline. "
            "Put options and HAR-RV simulations are outside the current report scope. The "
            "GJR-GARCH paths are not yet risk-neutral calibrated."
        ),
        "aggregate": aggregate,
        "models": valuations,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    report_path = config.output_dir / "README.md"
    report_path.write_text(render_report(metrics), encoding="utf-8")
    return metrics


def create_plots(
    output_dir: Path,
    american_results: dict[str, dict[str, Any]],
    swing_results: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    rows = []
    for model, results in american_results.items():
        rows.append(_plot_row(model, "European call", results["call_european"].path_values))
        rows.append(_plot_row(model, "Call LSMC replay", results["call_raw_lsmc"].path_values))
        rows.append(_plot_row(model, "Call LSMC selected", results["call_selected_lsmc"].path_values))
        rows.append(_plot_row(model, "Call linear Fitted-Q replay", results["call_fitted_q"].path_values))
        rows.append(_plot_row(model, "Call linear Fitted-Q selected", results["call_selected_fitted_q"].path_values))
        rows.append(_plot_row(model, "Call kernel Fitted-Q replay", results["call_kernel_q"].path_values))
        rows.append(_plot_row(model, "Call kernel Fitted-Q selected", results["call_selected_kernel_q"].path_values))
        rows.append(_plot_row(model, "Swing LSMC", swing_results[model]["lsmc"].path_values))
        rows.append(_plot_row(model, "Swing quota-aware", swing_results[model]["quota_aware"].path_values))
    valuation_frame = pd.DataFrame(rows)
    pivot = valuation_frame.pivot(index="instrument", columns="model", values="price")
    error = valuation_frame.pivot(index="instrument", columns="model", values="stderr").reindex_like(pivot)
    ax = pivot.plot(kind="bar", figsize=(10.5, 5.2), yerr=error, capsize=3)
    ax.set_title("Frozen-policy valuation by path generator")
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=18)
    plt.tight_layout()
    paths["valuation_bars"] = output_dir / "valuation_bars.png"
    plt.savefig(paths["valuation_bars"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    for model, results in american_results.items():
        for key, style, label_suffix in (
            ("call_raw_lsmc", "-", "raw LSMC"),
            ("call_selected_lsmc", "-.", "selected LSMC"),
            ("call_fitted_q", "--", "raw linear Fitted-Q"),
            ("call_selected_fitted_q", (0, (3, 1, 1, 1)), "selected linear Fitted-Q"),
            ("call_kernel_q", ":", "raw kernel Fitted-Q"),
            ("call_selected_kernel_q", (0, (1, 2)), "selected kernel Fitted-Q"),
        ):
            profile = _american_policy_exercise_profile(results[key])
            maturity = int(results[key].contract.maturity_step or results[key].exercise_steps.max())
            profile = profile.loc[profile["step"] < maturity]
            if profile.empty:
                continue
            plt.plot(
                profile["step"],
                profile["exercise_probability"],
                linewidth=1.2,
                linestyle=style,
                label=f"{model} {label_suffix}",
            )
    plt.title("American call early-exercise profile: raw candidates vs selected deployments")
    plt.xlabel("Step")
    plt.ylabel("Exercise probability")
    plt.legend()
    plt.tight_layout()
    paths["american_call_exercise"] = output_dir / "american_call_exercise.png"
    plt.savefig(paths["american_call_exercise"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    for model, result in swing_results.items():
        profile = result["lsmc"].nomination_profile
        plt.plot(profile["step"], profile["mean_volume"], linewidth=1.2, label=model)
    plt.title("Frozen LSMC swing mean nomination profile")
    plt.xlabel("Step")
    plt.ylabel("Mean nominated volume")
    plt.legend()
    plt.tight_layout()
    paths["swing_nomination_profile"] = output_dir / "swing_nomination_profile.png"
    plt.savefig(paths["swing_nomination_profile"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.0, 4.8))
    for model, result in swing_results.items():
        plt.hist(result["lsmc"].path_values, bins=30, alpha=0.42, density=True, label=f"{model} LSMC")
        plt.hist(result["quota_aware"].path_values, bins=30, alpha=0.28, density=True, label=f"{model} quota")
    plt.title("Swing path-value distribution")
    plt.xlabel("Discounted path value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    paths["swing_value_distribution"] = output_dir / "swing_value_distribution.png"
    plt.savefig(paths["swing_value_distribution"], dpi=140)
    plt.close()

    return paths


def _plot_row(model: str, instrument: str, path_values: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(path_values, dtype=float)
    stderr = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {
        "model": model,
        "instrument": instrument,
        "price": float(np.mean(values)),
        "stderr": stderr,
    }


def _american_policy_exercise_profile(result: Any) -> pd.DataFrame:
    if result.path_results.empty:
        return pd.DataFrame(columns=["step", "exercise_probability"])
    profile = (
        result.path_results.groupby("exercise_step", sort=True)
        .agg(exercise_count=("exercise_step", "size"))
        .reset_index()
        .rename(columns={"exercise_step": "step"})
    )
    profile["exercise_probability"] = profile["exercise_count"] / len(result.path_results)
    return profile


def render_report(metrics: dict[str, Any]) -> str:
    models = metrics["models"]
    plots = metrics["plots"]
    lines = [
        f"# American Call LSMC/RL Valuation Report: {metrics['config']['symbol']}",
        "",
        metrics["valuation_assumption"],
        "",
        "## Setup",
        "",
        f"- Symbol: `{metrics['config']['symbol']}`",
        f"- Interval: `{metrics['config']['interval']}`",
        f"- Risk-free rate: `{_fmt(metrics['config']['risk_free_rate'])}`",
        f"- Strike moneyness: `{_fmt(metrics['config']['strike_moneyness'])}`",
        f"- Historical calibration windows: `{metrics['config']['period_count']}` rolling windows of `{metrics['config']['period_lookback_days']}` calendar days.",
        f"- Evaluation paths per scenario: `{metrics['config']['n_paths_if_simulated']}`.",
        f"- LSMC continuation models trained on `{metrics['config']['training_paths']}` generated paths per model.",
        f"- LSMC validation wrappers are chosen on `{metrics['config']['validation_paths']}` independent validation paths per model before replay on report paths.",
        f"- RL/ADP Fitted-Q models trained on `{metrics['config']['rl_training_paths']}` generated paths and validated on `{metrics['config']['rl_validation_paths']}` independent generated paths per model.",
        f"- Kernel Fitted-Q uses `{metrics['config']['kernel_rff_features']}` random Fourier features, length scale `{_fmt(metrics['config']['kernel_length_scale'])}`, feature seed `{metrics['config']['kernel_feature_seed']}`.",
        "- Path generator: GJR-GARCH only. HAR-RV simulations are not part of this report.",
        "- American contract: ATM call only, exercise possible at every simulated step. Put options are no longer part of this analysis.",
        "- Each American replay below is the raw frozen candidate policy replay, not a no-arbitrage American fair value.",
        "- A true American optimal-stopping value should not be below the European never-exercise value because never-exercise is feasible. Negative replay-minus-European deltas therefore flag LSMC policy failure, not a lower American option value.",
        "- Validation-selected deployment wrappers are reported separately because they may delegate to never-exercise and therefore equal the European baseline by construction.",
        "- No American replay result is post-processed with a European floor or pathwise maximum.",
        "- Swing contract: call-style gas nomination, hourly exercise grid (`12` x 5-minute steps), max `1` unit per exercise, total band `2` to `6` units.",
        "",
        f"![Valuation bars]({Path(plots['valuation_bars']).name})",
        "",
        "## Values",
        "",
        _markdown_table(
            [
                "period",
                "model",
                "candidate",
                "eval paths",
                "train paths",
                "valid paths",
                "start",
                "strike",
                "raw replay",
                "European",
                "raw-Euro",
                "bound check",
                "selected policy",
                "deployment",
                "deployment-Euro",
                "swing LSMC",
                "swing vs quota",
            ],
            [
                [
                    str(data["period"]),
                    str(data["model_type"]),
                    str(candidate["candidate_label"]),
                    str(data["path_count"]),
                    str(data["training_path_count"] if candidate_key == "lsmc" else data["rl_training_path_count"]),
                    str(data["validation_path_count"] if candidate_key == "lsmc" else data["rl_validation_path_count"]),
                    _fmt(data["start_price"]),
                    _fmt(data["strike"]),
                    _fmt(candidate["raw_candidate_value"]),
                    _fmt(candidate["european_value"]),
                    _fmt(candidate["raw_mean_delta_vs_european"]),
                    _bound_label(candidate["raw_mean_delta_vs_european"]),
                    str(candidate["selected_policy_name"]),
                    _fmt(candidate["deployment_value"]),
                    _fmt(candidate["selected_mean_delta_vs_european"]),
                    _fmt(data["swing_call"]["price"]),
                    _fmt(data["swing_call"]["mean_delta_vs_quota_aware"]),
                ]
                for _, data in models.items()
                for candidate_key, candidate in data["american_call_candidates"].items()
            ],
        ),
        "",
        "## Aggregate Bound Checks",
        "",
        (
            f"Raw American call candidate replay is below European in "
            f"`{metrics['aggregate']['raw_replay_below_european_count']}` of "
            f"`{metrics['aggregate']['american_contract_checks']}` call-policy checks "
            f"(`{_fmt(metrics['aggregate']['raw_replay_below_european_share'])}` share). "
            "Each violation is a policy-replay failure, not a valid American value."
        ),
        "",
        "## Validation-Selected Deployment",
        "",
        _markdown_table(
            [
                "period",
                "model",
                "candidate",
                "selected policy",
                "deployment",
                "deployment delta",
                "validation delta CI95",
                "gate",
            ],
            [
                [
                    str(data["period"]),
                    str(data["model_type"]),
                    str(candidate["candidate_label"]),
                    str(candidate["selected_policy_name"]),
                    _fmt(candidate["deployment_value"]),
                    _fmt(candidate["selected_mean_delta_vs_european"]),
                    _ci(candidate["validation_delta_ci95"]),
                    "candidate" if candidate["used_candidate"] else "never-exercise fallback",
                ]
                for _, data in models.items()
                for candidate in data["american_call_candidates"].values()
            ],
        ),
        "",
        "## Exercise Diagnostics",
        "",
        _markdown_table(
            [
                "period",
                "model",
                "candidate",
                "raw early ex.",
                "selected early ex.",
                "selected",
                "raw delta CI95",
                "selected delta CI95",
                "median train R2",
                "swing mean volume",
                "swing q05/q50/q95 volume",
                "median swing train R2",
            ],
            [
                [
                    str(data["period"]),
                    str(data["model_type"]),
                    str(candidate["candidate_label"]),
                    _fmt(candidate["raw_early_exercise_probability"]),
                    _fmt(candidate["selected_early_exercise_probability"]),
                    str(candidate["selected_policy_name"]),
                    _ci(candidate["raw_delta_vs_european_ci95"]),
                    _ci(candidate["selected_delta_vs_european_ci95"]),
                    _fmt(candidate["median_regression_r2"]),
                    _fmt(data["swing_call"]["mean_total_volume"]),
                    f"{_fmt(data['swing_call']['q05_total_volume'])} / {_fmt(data['swing_call']['q50_total_volume'])} / {_fmt(data['swing_call']['q95_total_volume'])}",
                    _fmt(data["swing_call"]["median_regression_r2"]),
                ]
                for _, data in models.items()
                for candidate in data["american_call_candidates"].values()
            ],
        ),
        "",
        f"![American call exercise]({Path(plots['american_call_exercise']).name})",
        "",
        f"![Swing nomination profile]({Path(plots['swing_nomination_profile']).name})",
        "",
        f"![Swing value distribution]({Path(plots['swing_value_distribution']).name})",
        "",
        "## Interpretation",
        "",
        _bottom_line(metrics),
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m lsmc_rl.analysis.option_lsmc_report --output-dir outputs/option_lsmc_report_front",
        "```",
        "",
        "Detailed numeric output is stored in `metrics.json` next to this report.",
        "",
    ]
    return "\n".join(lines)


def _evaluation_periods(config: OptionLSMCReportConfig) -> list[EvaluationPeriod]:
    market, _ = load_market_data(
        db_path=config.database_path,
        symbol=config.symbol,
        interval=config.interval,
    )
    first = pd.Timestamp(market["open_datetime"].iloc[0])
    last = pd.Timestamp(market["open_datetime"].iloc[-1])
    lookback = pd.Timedelta(days=max(1, int(config.period_lookback_days)))
    period_count = max(1, int(config.period_count))
    first_end = min(max(first + lookback, first), last)
    if period_count == 1 or first_end >= last:
        end_times = [last]
    else:
        end_times = list(pd.date_range(start=first_end, end=last, periods=period_count))

    periods: list[EvaluationPeriod] = []
    seen_names: set[str] = set()
    for index, end_time in enumerate(end_times, start=1):
        end = pd.Timestamp(end_time).tz_convert("UTC") if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time).tz_localize("UTC")
        start = max(first, end - lookback)
        name = f"p{index}_{end.strftime('%Y%m%d')}"
        if name in seen_names:
            name = f"{name}_{index}"
        seen_names.add(name)
        periods.append(EvaluationPeriod(name=name, start=start, end=end))
    return periods


def _simulate_period_path_sets(
    model_type: str,
    period: EvaluationPeriod,
    period_index: int,
    config: OptionLSMCReportConfig,
) -> dict[str, Any]:
    market, report = load_market_data(
        db_path=config.database_path,
        symbol=config.symbol,
        interval=config.interval,
        start=period.start,
        end=period.end,
    )
    start_price = float(market["close"].iloc[-1])
    if start_price <= 0.0:
        raise ValueError("period start price must be positive")
    returns = market["log_return"].dropna()
    model_key = model_type.lower()
    if model_key == "gjr_garch":
        model = GJRGARCHModel().fit(returns)

        def simulate(n_paths: int, seed: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return model.simulate(
                start_price=start_price,
                horizon_steps=config.horizon_steps,
                n_paths=n_paths,
                seed=seed,
            )

    else:
        raise ValueError("model_type must be gjr_garch")

    time_grid = pd.date_range(
        start=pd.Timestamp(market["open_datetime"].iloc[-1]),
        periods=config.horizon_steps + 1,
        freq="5min",
        tz="UTC",
    )
    eval_prices, eval_returns, eval_variances = simulate(
        config.n_paths_if_simulated,
        _period_seed(config.seed, period_index, model_key, 0),
    )
    train_prices, train_returns, train_variances = simulate(
        config.training_paths,
        _period_seed(config.training_seed, period_index, model_key, 101),
    )
    validation_prices, validation_returns, validation_variances = simulate(
        config.validation_paths,
        _period_seed(config.validation_seed, period_index, model_key, 202),
    )
    rl_train_prices, rl_train_returns, rl_train_variances = simulate(
        config.rl_training_paths,
        _period_seed(config.rl_training_seed, period_index, model_key, 303),
    )
    rl_validation_prices, rl_validation_returns, rl_validation_variances = simulate(
        config.rl_validation_paths,
        _period_seed(config.rl_validation_seed, period_index, model_key, 404),
    )
    return {
        "evaluation": paths_to_frame(eval_prices, eval_returns, eval_variances, time_grid, model_key),
        "training": paths_to_frame(train_prices, train_returns, train_variances, time_grid, model_key),
        "validation": paths_to_frame(validation_prices, validation_returns, validation_variances, time_grid, model_key),
        "rl_training": paths_to_frame(rl_train_prices, rl_train_returns, rl_train_variances, time_grid, model_key),
        "rl_validation": paths_to_frame(
            rl_validation_prices,
            rl_validation_returns,
            rl_validation_variances,
            time_grid,
            model_key,
        ),
        "data_quality": report,
    }


def _period_seed(base_seed: int | None, period_index: int, model_type: str, offset: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(period_index) * 100_000 + int(offset)


def _aggregate_american_bound_status(valuations: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for scenario, data in valuations.items():
        for candidate_key, option in data.get("american_call_candidates", {}).items():
            rows.append(
                {
                    "scenario": scenario,
                    "period": data.get("period"),
                    "model_type": data.get("model_type"),
                    "option": "call",
                    "candidate": candidate_key,
                    "candidate_label": option.get("candidate_label"),
                    "raw_replay_below_european": bool(option["raw_replay_below_european"]),
                    "raw_mean_delta_vs_european": float(option["raw_mean_delta_vs_european"]),
                }
            )
    violation_count = sum(1 for row in rows if row["raw_replay_below_european"])
    return {
        "american_contract_checks": len(rows),
        "raw_replay_below_european_count": int(violation_count),
        "raw_replay_below_european_share": float(violation_count / len(rows)) if rows else 0.0,
        "bound_checks": rows,
    }


def _load_or_simulate_paths(model_type: str, config: OptionLSMCReportConfig) -> pd.DataFrame:
    existing = DEFAULT_PATH_FILES[model_type]
    if config.use_existing_paths and existing.exists():
        return pd.read_csv(existing)
    simulation = run_simulation(
        PathSimulationConfig(
            database_path=config.database_path,
            symbol=config.symbol,
            interval=config.interval,
            model_type=model_type,
            horizon_steps=config.horizon_steps,
            n_paths=config.n_paths_if_simulated,
            seed=config.seed,
            output_path=None,
            steps_per_day=config.steps_per_day,
        )
    )
    return simulation.paths


def _load_or_simulate_training_paths(
    model_type: str,
    config: OptionLSMCReportConfig,
    maturity: int,
    cache: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    key = (model_type, maturity)
    if key not in cache:
        simulation = run_simulation(
            PathSimulationConfig(
                database_path=config.database_path,
                symbol=config.symbol,
                interval=config.interval,
                model_type=model_type,
                horizon_steps=maturity,
                n_paths=config.training_paths,
                seed=config.training_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _load_or_simulate_validation_paths(
    model_type: str,
    config: OptionLSMCReportConfig,
    maturity: int,
    cache: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    key = (model_type, maturity)
    if key not in cache:
        simulation = run_simulation(
            PathSimulationConfig(
                database_path=config.database_path,
                symbol=config.symbol,
                interval=config.interval,
                model_type=model_type,
                horizon_steps=maturity,
                n_paths=config.validation_paths,
                seed=config.validation_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _path_source_for_model(model_type: str, config: OptionLSMCReportConfig) -> str:
    existing = DEFAULT_PATH_FILES[model_type]
    if config.use_existing_paths and existing.exists():
        return str(existing)
    return "generated_in_memory"


def _infer_time_step_years(frame: pd.DataFrame) -> float:
    if "time" not in frame.columns:
        return 5.0 / (365.0 * 24.0 * 60.0)
    step_times = frame.drop_duplicates("step").sort_values("step")["time"]
    times = pd.to_datetime(step_times, utc=True, errors="coerce").dropna()
    if len(times) < 2:
        return 5.0 / (365.0 * 24.0 * 60.0)
    seconds = np.diff(times.astype("int64").to_numpy()) / 1_000_000_000
    seconds = seconds[np.isfinite(seconds) & (seconds > 0.0)]
    if seconds.size == 0:
        return 5.0 / (365.0 * 24.0 * 60.0)
    return float(np.median(seconds) / (365.0 * 24.0 * 60.0 * 60.0))


def _training_regression_r2(training_result: Any) -> pd.Series:
    diagnostics = getattr(training_result, "regression_diagnostics", None)
    if diagnostics is None:
        diagnostics = getattr(training_result, "training_diagnostics", None)
    if isinstance(diagnostics, pd.DataFrame) and "regression_r2" in diagnostics:
        return diagnostics["regression_r2"].dropna()
    return pd.Series(dtype=float)


def _summarize_american(
    training_result: Any,
    raw_candidate_result: Any,
    selected_result: Any,
    european_result: Any,
    validation_selection: dict[str, Any],
    bootstrap_seed: int,
    n_bootstrap: int,
    candidate_label: str = "LSMC",
) -> dict[str, Any]:
    maturity = int(selected_result.contract.maturity_step or selected_result.exercise_steps.max())
    regression_r2 = _training_regression_r2(training_result)
    selected_summary = path_value_summary(selected_result.path_values)
    raw_summary = path_value_summary(raw_candidate_result.path_values)
    european_summary = path_value_summary(european_result.path_values)
    selected_pair = paired_policy_metrics(
        selected_result.path_values,
        european_result.path_values,
        name_a=selected_result.policy_name,
        name_b=european_result.policy_name,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    raw_pair = paired_policy_metrics(
        raw_candidate_result.path_values,
        european_result.path_values,
        name_a=raw_candidate_result.policy_name,
        name_b=european_result.policy_name,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "candidate_label": candidate_label,
        "price": raw_summary["mean"],
        "lsmc_value": raw_summary["mean"],
        "deployment_value": selected_summary["mean"],
        "selected_policy_value": selected_summary["mean"],
        "raw_candidate_value": raw_summary["mean"],
        "raw_lsmc_value": raw_summary["mean"],
        "training_in_sample_value": training_result.price,
        "stderr": raw_summary["standard_error"],
        "selected_stderr": selected_summary["standard_error"],
        "raw_lsmc_stderr": raw_summary["standard_error"],
        "ci95_low": raw_summary["mean"] - 1.96 * raw_summary["standard_error"],
        "ci95_high": raw_summary["mean"] + 1.96 * raw_summary["standard_error"],
        "selected_ci95_low": selected_summary["mean"] - 1.96 * selected_summary["standard_error"],
        "selected_ci95_high": selected_summary["mean"] + 1.96 * selected_summary["standard_error"],
        "raw_lsmc_ci95_low": raw_summary["mean"] - 1.96 * raw_summary["standard_error"],
        "raw_lsmc_ci95_high": raw_summary["mean"] + 1.96 * raw_summary["standard_error"],
        "european_value": european_summary["mean"],
        "european_stderr": european_summary["standard_error"],
        "mean_delta_vs_european": raw_pair["mean_delta"],
        "selected_mean_delta_vs_european": selected_pair["mean_delta"],
        "raw_mean_delta_vs_european": raw_pair["mean_delta"],
        "raw_replay_below_european": bool(raw_pair["mean_delta"] < -1e-12),
        "selected_replay_below_european": bool(selected_pair["mean_delta"] < -1e-12),
        "american_value_bound_status": (
            "violated_by_raw_policy_replay" if raw_pair["mean_delta"] < -1e-12 else "not_violated_by_raw_policy_replay"
        ),
        "delta_vs_european_ci95": raw_pair["bootstrap_mean_delta_ci95"],
        "selected_delta_vs_european_ci95": selected_pair["bootstrap_mean_delta_ci95"],
        "raw_delta_vs_european_ci95": raw_pair["bootstrap_mean_delta_ci95"],
        "share_delta_vs_european_positive": raw_pair["share_delta_positive"],
        "selected_share_delta_vs_european_positive": selected_pair["share_delta_positive"],
        "raw_share_delta_vs_european_positive": raw_pair["share_delta_positive"],
        "early_exercise_probability": float(np.mean(raw_candidate_result.exercise_steps < maturity)),
        "selected_early_exercise_probability": float(np.mean(selected_result.exercise_steps < maturity)),
        "raw_early_exercise_probability": float(np.mean(raw_candidate_result.exercise_steps < maturity)),
        "training_early_exercise_probability": float(np.mean(training_result.exercise_steps < maturity)),
        "mean_exercise_step": float(np.mean(raw_candidate_result.exercise_steps)),
        "selected_mean_exercise_step": float(np.mean(selected_result.exercise_steps)),
        "raw_mean_exercise_step": float(np.mean(raw_candidate_result.exercise_steps)),
        "q05_path_value": raw_summary["q05"],
        "q50_path_value": raw_summary["q50"],
        "q95_path_value": raw_summary["q95"],
        "selected_q05_path_value": selected_summary["q05"],
        "selected_q50_path_value": selected_summary["q50"],
        "selected_q95_path_value": selected_summary["q95"],
        "raw_q05_path_value": raw_summary["q05"],
        "raw_q50_path_value": raw_summary["q50"],
        "raw_q95_path_value": raw_summary["q95"],
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
        "validation_mean_delta": validation_selection["mean_delta"],
        "validation_delta_ci95": validation_selection["bootstrap_mean_delta_ci95"],
        "selected_wrapper_name": selected_result.policy_name,
        "raw_policy_name": raw_candidate_result.policy_name,
        "selected_policy_name": validation_selection["selected_policy_name"],
        "used_candidate": validation_selection["used_candidate"],
        "used_lsmc_candidate": validation_selection["used_candidate"],
        "selection_rule": validation_selection["decision_rule"],
    }


def _summarize_swing(
    training_result: SwingLSMCResult,
    evaluation_result: Any,
    positive_margin_result: Any,
    quota_result: Any,
    bootstrap_seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    regression_r2 = training_result.regression_diagnostics["regression_r2"].dropna()
    policy_summary = path_value_summary(evaluation_result.path_values)
    positive_pair = paired_policy_metrics(
        evaluation_result.path_values,
        positive_margin_result.path_values,
        name_a=evaluation_result.policy_name,
        name_b=positive_margin_result.policy_name,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    quota_pair = paired_policy_metrics(
        evaluation_result.path_values,
        quota_result.path_values,
        name_a=evaluation_result.policy_name,
        name_b=quota_result.policy_name,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    summary = {
        "price": policy_summary["mean"],
        "training_in_sample_value": training_result.price,
        "stderr": policy_summary["standard_error"],
        "ci95_low": policy_summary["mean"] - 1.96 * policy_summary["standard_error"],
        "ci95_high": policy_summary["mean"] + 1.96 * policy_summary["standard_error"],
        "q05_path_value": policy_summary["q05"],
        "q50_path_value": policy_summary["q50"],
        "q95_path_value": policy_summary["q95"],
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
        "mean_delta_vs_positive_margin": positive_pair["mean_delta"],
        "delta_vs_positive_margin_ci95": positive_pair["bootstrap_mean_delta_ci95"],
        "mean_delta_vs_quota_aware": quota_pair["mean_delta"],
        "delta_vs_quota_aware_ci95": quota_pair["bootstrap_mean_delta_ci95"],
    }
    summary.update(_swing_evaluation_summary(evaluation_result))
    return summary


def _swing_evaluation_summary(result: Any) -> dict[str, float]:
    total = result.path_results["total_volume"].to_numpy(dtype=float)
    return {
        "mean_total_volume": float(np.mean(total)),
        "q05_total_volume": float(np.quantile(total, 0.05)),
        "q50_total_volume": float(np.quantile(total, 0.50)),
        "q95_total_volume": float(np.quantile(total, 0.95)),
        "mean_positive_value": float(np.mean(result.path_values > 0.0)),
        "mean_path_value": float(np.mean(result.path_values)),
        "mean_shortfall_volume": float(result.path_results["shortfall_volume"].mean()),
        "mean_constraint_violations": float(result.path_results["constraint_violations"].mean()),
        "mean_costs": float(result.path_results["costs"].mean()),
    }


def _bottom_line(metrics: dict[str, Any]) -> str:
    model_lines = []
    for _, data in metrics["models"].items():
        candidate_bits = []
        for candidate in data["american_call_candidates"].values():
            candidate_bits.append(
                f"{candidate['candidate_label']} raw-Euro `{_fmt(candidate['raw_mean_delta_vs_european'])}`, "
                f"selected-Euro `{_fmt(candidate['selected_mean_delta_vs_european'])}`"
            )
        model_lines.append(
            f"- `{data['period']}` `{data['model_type']}`: "
            + "; ".join(candidate_bits)
            + f"; swing LSMC value `{_fmt(data['swing_call']['price'])}` with mean total volume "
            f"`{_fmt(data['swing_call']['mean_total_volume'])}` and delta versus quota-aware "
            f"`{_fmt(data['swing_call']['mean_delta_vs_quota_aware'])}`."
        )
    aggregate = metrics.get("aggregate", {})
    aggregate_line = (
        f"Across windows and American-call candidates, raw replay violates the European lower bound in "
        f"`{aggregate.get('raw_replay_below_european_count', 'n/a')}` of "
        f"`{aggregate.get('american_contract_checks', 'n/a')}` checks."
    )
    caveat = (
        "Each raw American call line is a learned-policy replay diagnostic. If it is below European, it "
        "violates the American no-arbitrage lower bound and should be read as a failed exercise policy, "
        "not as an American option value. The selected deployment line is a validation-chosen wrapper, "
        "not a pathwise maximum with the European payoff. When validation cannot establish a robust "
        "improvement, the wrapper falls back to never exercising early. Put options and HAR-RV simulations "
        "are intentionally excluded from this report. These numbers are still engineering "
        "diagnostics, not final fair values, until drift/risk-neutral assumptions, transaction costs, and "
        "richer sensitivity analysis are locked down."
    )
    return "\n".join(model_lines + ["", aggregate_line, "", caveat])


def _ci(values: Any) -> str:
    try:
        low, high = values
    except (TypeError, ValueError):
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _bound_label(delta: Any) -> str:
    try:
        numeric = float(delta)
    except (TypeError, ValueError):
        return "n/a"
    if numeric < -1e-12:
        return "policy failure: below European"
    return "ok"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.6e}"
    return f"{numeric:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Value American calls and swing options on GJR-GARCH MC paths.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output-dir", default="outputs/option_lsmc_report_front")
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--strike-moneyness", type=float, default=1.0)
    parser.add_argument("--n-paths-if-simulated", type=int, default=2048)
    parser.add_argument("--training-paths", type=int, default=2048)
    parser.add_argument("--validation-paths", type=int, default=2048)
    parser.add_argument("--rl-training-paths", type=int, default=2048)
    parser.add_argument("--rl-validation-paths", type=int, default=2048)
    parser.add_argument("--horizon-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--training-seed", type=int, default=20260522)
    parser.add_argument("--validation-seed", type=int, default=20260523)
    parser.add_argument("--rl-training-seed", type=int, default=20260521)
    parser.add_argument("--rl-validation-seed", type=int, default=20260524)
    parser.add_argument("--bootstrap-seed", type=int, default=20260520)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--kernel-rff-features", type=int, default=96)
    parser.add_argument("--kernel-length-scale", type=float, default=1.0)
    parser.add_argument("--kernel-feature-seed", type=int, default=20260522)
    parser.add_argument("--period-count", type=int, default=4)
    parser.add_argument("--period-lookback-days", type=int, default=120)
    parser.add_argument("--use-existing-paths", action="store_true")
    parser.add_argument("--ignore-existing-paths", action="store_true")
    args = parser.parse_args(argv)

    config = OptionLSMCReportConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        output_dir=Path(args.output_dir).resolve(),
        risk_free_rate=args.risk_free_rate,
        strike_moneyness=args.strike_moneyness,
        n_paths_if_simulated=args.n_paths_if_simulated,
        training_paths=args.training_paths,
        validation_paths=args.validation_paths,
        rl_training_paths=args.rl_training_paths,
        rl_validation_paths=args.rl_validation_paths,
        horizon_steps=args.horizon_steps,
        seed=args.seed,
        training_seed=args.training_seed,
        validation_seed=args.validation_seed,
        rl_training_seed=args.rl_training_seed,
        rl_validation_seed=args.rl_validation_seed,
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
        kernel_rff_features=args.kernel_rff_features,
        kernel_length_scale=args.kernel_length_scale,
        kernel_feature_seed=args.kernel_feature_seed,
        period_count=args.period_count,
        period_lookback_days=args.period_lookback_days,
        use_existing_paths=bool(args.use_existing_paths and not args.ignore_existing_paths),
    )
    metrics = run_report(config)
    print(f"Wrote report: {config.output_dir / 'README.md'}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    for scenario, data in metrics["models"].items():
        candidate_text = ", ".join(
            f"{candidate_key}={_fmt(candidate['raw_candidate_value'])}/selected={_fmt(candidate['deployment_value'])}"
            for candidate_key, candidate in data["american_call_candidates"].items()
        )
        print(f"{scenario}: {candidate_text}, swing_lsmc={_fmt(data['swing_call']['price'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
