"""Generate a frozen-policy evaluation report on common test paths."""

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

from lsmc_rl.evaluation import (
    ImmediateIntrinsicExercisePolicy,
    NeverEarlyExercisePolicy,
    NeverExerciseSwingPolicy,
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
from lsmc_rl.simulation.paths import PathSimulationConfig, run_simulation
from lsmc_rl.valuation import (
    AmericanOptionContract,
    RegressionConfig,
    SwingOptionContract,
    value_american_option_lsmc,
    value_swing_option_lsmc,
)
from lsmc_rl.valuation.common import json_ready, paths_frame_to_matrices


DEFAULT_PATH_FILES = {
    "gjr_garch": Path("outputs/mc_paths_front_gjr_garch.csv"),
}


@dataclass(frozen=True)
class PolicyEvaluationReportConfig:
    database_path: Path = Path("ttf_klines_5m_from_1m.sqlite")
    symbol: str = "FRONT"
    interval: str = "5m"
    output_dir: Path = Path("outputs/policy_evaluation_front")
    risk_free_rate: float = 0.03
    strike_moneyness: float = 1.0
    n_paths_if_simulated: int = 2048
    horizon_steps: int = 288
    seed: int = 20260519
    bootstrap_seed: int = 20260520
    n_bootstrap: int = 2000
    rl_training_paths: int = 2048
    rl_training_seed: int = 20260521
    rl_validation_paths: int = 2048
    rl_validation_seed: int = 20260524
    lsmc_training_paths: int = 2048
    lsmc_training_seed: int = 20260522
    lsmc_validation_paths: int = 2048
    lsmc_validation_seed: int = 20260523
    deployment_min_ci_low: float = 0.0
    deployment_min_cvar_5_delta: float | None = 0.0
    kernel_rff_features: int = 96
    kernel_length_scale: float = 1.0
    kernel_feature_seed: int = 20260522
    steps_per_day: int = 288
    use_existing_paths: bool = False


def run_report(config: PolicyEvaluationReportConfig) -> dict[str, Any]:
    _ensure_allowed_artifact_dir(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path_frames = {model: _load_or_simulate_paths(model, config) for model in ("gjr_garch",)}
    path_quality = _load_path_quality_summary(Path("outputs/path_quality_report_front/metrics.json"))
    rl_training_cache: dict[tuple[str, int], pd.DataFrame] = {}
    rl_validation_cache: dict[tuple[str, int], pd.DataFrame] = {}
    lsmc_training_cache: dict[tuple[str, int], pd.DataFrame] = {}
    lsmc_validation_cache: dict[tuple[str, int], pd.DataFrame] = {}
    lsmc_regression = RegressionConfig(
        degree=3,
        ridge_alpha=1e-5,
        itm_only=True,
        min_regression_paths=20,
        include_log_moneyness=True,
        include_intrinsic=True,
        include_variance=True,
        clip_negative_continuation=True,
    )
    swing_lsmc_regression = RegressionConfig(
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

    model_metrics: dict[str, Any] = {}
    plot_inputs: dict[str, Any] = {}
    for model, frame in path_frames.items():
        matrices = paths_frame_to_matrices(frame)
        start_price = float(np.median(matrices.prices[:, 0]))
        strike = start_price * config.strike_moneyness
        maturity = int(matrices.prices.shape[1] - 1)
        time_step_years = _infer_time_step_years(frame)

        american_call = AmericanOptionContract(
            strike=strike,
            option_type="call",
            risk_free_rate=config.risk_free_rate,
            time_step_years=time_step_years,
            maturity_step=maturity,
            exercise_start_step=1,
            exercise_step_interval=1,
        )
        swing_call = SwingOptionContract(
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

        never = NeverEarlyExercisePolicy()
        immediate = ImmediateIntrinsicExercisePolicy()
        swing_never = NeverExerciseSwingPolicy()
        swing_positive = PositiveMarginSwingPolicy()
        swing_quota = QuotaAwareSwingPolicy()

        rl_training_frame = _load_or_simulate_rl_training_paths(model, config, maturity, rl_training_cache)
        rl_training_matrices = paths_frame_to_matrices(rl_training_frame)
        american_call_fitted_q_train = train_american_fitted_q(rl_training_frame, american_call, fitted_q_config)
        american_call_kernel_q_train = train_american_kernel_fitted_q(
            rl_training_frame,
            american_call,
            kernel_fitted_q_config,
        )
        rl_validation_frame = _load_or_simulate_rl_validation_paths(model, config, maturity, rl_validation_cache)
        rl_validation_matrices = paths_frame_to_matrices(rl_validation_frame)
        american_call_fitted_q_selection = select_american_policy_by_validation(
            rl_validation_frame,
            american_call,
            american_call_fitted_q_train.policy,
            baseline_policy=never,
            bootstrap_seed=config.bootstrap_seed,
            n_bootstrap=config.n_bootstrap,
            min_ci_low=config.deployment_min_ci_low,
            min_cvar_5_delta=config.deployment_min_cvar_5_delta,
            selected_name="american_fitted_q_validation_selected_deployment",
        )
        american_call_kernel_q_selection = select_american_policy_by_validation(
            rl_validation_frame,
            american_call,
            american_call_kernel_q_train.policy,
            baseline_policy=never,
            bootstrap_seed=config.bootstrap_seed,
            n_bootstrap=config.n_bootstrap,
            min_ci_low=config.deployment_min_ci_low,
            min_cvar_5_delta=config.deployment_min_cvar_5_delta,
            selected_name="american_kernel_fitted_q_validation_selected_deployment",
        )
        lsmc_training_frame = _load_or_simulate_lsmc_training_paths(model, config, maturity, lsmc_training_cache)
        lsmc_training_matrices = paths_frame_to_matrices(lsmc_training_frame)
        american_call_lsmc_train = value_american_option_lsmc(lsmc_training_frame, american_call, lsmc_regression)
        swing_lsmc_train = value_swing_option_lsmc(lsmc_training_frame, swing_call, swing_lsmc_regression)
        lsmc_validation_frame = _load_or_simulate_lsmc_validation_paths(
            model,
            config,
            maturity,
            lsmc_validation_cache,
        )
        lsmc_validation_matrices = paths_frame_to_matrices(lsmc_validation_frame)
        american_call_lsmc_selection = select_american_policy_by_validation(
            lsmc_validation_frame,
            american_call,
            american_call_lsmc_train.policy,
            baseline_policy=never,
            bootstrap_seed=config.bootstrap_seed,
            n_bootstrap=config.n_bootstrap,
            min_ci_low=config.deployment_min_ci_low,
            min_cvar_5_delta=config.deployment_min_cvar_5_delta,
        )

        american_call_never = evaluate_american_policy(frame, american_call, never)
        american_call_immediate = evaluate_american_policy(frame, american_call, immediate)
        american_call_raw_lsmc = evaluate_american_policy(frame, american_call, american_call_lsmc_train.policy)
        american_call_selected_lsmc = evaluate_american_policy(frame, american_call, american_call_lsmc_selection.policy)
        american_call_fitted_q = evaluate_american_policy(frame, american_call, american_call_fitted_q_train.policy)
        american_call_selected_fitted_q = evaluate_american_policy(
            frame,
            american_call,
            american_call_fitted_q_selection.policy,
        )
        american_call_kernel_q = evaluate_american_policy(frame, american_call, american_call_kernel_q_train.policy)
        american_call_selected_kernel_q = evaluate_american_policy(
            frame,
            american_call,
            american_call_kernel_q_selection.policy,
        )
        swing_never_result = evaluate_swing_policy(frame, swing_call, swing_never)
        swing_positive_result = evaluate_swing_policy(frame, swing_call, swing_positive)
        swing_quota_result = evaluate_swing_policy(frame, swing_call, swing_quota)
        swing_lsmc_result = evaluate_swing_policy(frame, swing_call, swing_lsmc_train.policy_model)

        pairs = {
            "american_call_immediate_vs_never": paired_policy_metrics(
                american_call_immediate.path_values,
                american_call_never.path_values,
                name_a=american_call_immediate.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_fitted_q_vs_never": paired_policy_metrics(
                american_call_fitted_q.path_values,
                american_call_never.path_values,
                name_a=american_call_fitted_q.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_fitted_q_vs_never": paired_policy_metrics(
                american_call_selected_fitted_q.path_values,
                american_call_never.path_values,
                name_a=american_call_selected_fitted_q.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_raw_lsmc_vs_never": paired_policy_metrics(
                american_call_raw_lsmc.path_values,
                american_call_never.path_values,
                name_a=american_call_raw_lsmc.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_lsmc_vs_never": paired_policy_metrics(
                american_call_selected_lsmc.path_values,
                american_call_never.path_values,
                name_a=american_call_selected_lsmc.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_raw_lsmc_vs_fitted_q": paired_policy_metrics(
                american_call_raw_lsmc.path_values,
                american_call_fitted_q.path_values,
                name_a=american_call_raw_lsmc.policy_name,
                name_b=american_call_fitted_q.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_lsmc_vs_fitted_q": paired_policy_metrics(
                american_call_selected_lsmc.path_values,
                american_call_fitted_q.path_values,
                name_a=american_call_selected_lsmc.policy_name,
                name_b=american_call_fitted_q.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_lsmc_vs_selected_fitted_q": paired_policy_metrics(
                american_call_selected_lsmc.path_values,
                american_call_selected_fitted_q.path_values,
                name_a=american_call_selected_lsmc.policy_name,
                name_b=american_call_selected_fitted_q.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_kernel_fitted_q_vs_never": paired_policy_metrics(
                american_call_kernel_q.path_values,
                american_call_never.path_values,
                name_a=american_call_kernel_q.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_kernel_fitted_q_vs_never": paired_policy_metrics(
                american_call_selected_kernel_q.path_values,
                american_call_never.path_values,
                name_a=american_call_selected_kernel_q.policy_name,
                name_b=american_call_never.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_kernel_fitted_q_vs_linear_fitted_q": paired_policy_metrics(
                american_call_kernel_q.path_values,
                american_call_fitted_q.path_values,
                name_a=american_call_kernel_q.policy_name,
                name_b=american_call_fitted_q.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "american_call_selected_kernel_fitted_q_vs_selected_linear_fitted_q": paired_policy_metrics(
                american_call_selected_kernel_q.path_values,
                american_call_selected_fitted_q.path_values,
                name_a=american_call_selected_kernel_q.policy_name,
                name_b=american_call_selected_fitted_q.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "swing_quota_vs_positive_margin": paired_policy_metrics(
                swing_quota_result.path_values,
                swing_positive_result.path_values,
                name_a=swing_quota_result.policy_name,
                name_b=swing_positive_result.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "swing_positive_margin_vs_never": paired_policy_metrics(
                swing_positive_result.path_values,
                swing_never_result.path_values,
                name_a=swing_positive_result.policy_name,
                name_b=swing_never_result.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "swing_lsmc_vs_quota_aware": paired_policy_metrics(
                swing_lsmc_result.path_values,
                swing_quota_result.path_values,
                name_a=swing_lsmc_result.policy_name,
                name_b=swing_quota_result.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
            "swing_lsmc_vs_positive_margin": paired_policy_metrics(
                swing_lsmc_result.path_values,
                swing_positive_result.path_values,
                name_a=swing_lsmc_result.policy_name,
                name_b=swing_positive_result.policy_name,
                bootstrap_seed=config.bootstrap_seed,
                n_bootstrap=config.n_bootstrap,
            ),
        }
        pairs["american_call_lsmc_vs_never"] = pairs["american_call_raw_lsmc_vs_never"]
        pairs["american_call_lsmc_vs_fitted_q"] = pairs["american_call_raw_lsmc_vs_fitted_q"]

        model_metrics[model] = {
            "path_source": _path_source_for_model(model, config),
            "path_count": int(matrices.prices.shape[0]),
            "horizon_steps": maturity,
            "time_step_years": time_step_years,
            "start_price": start_price,
            "strike": strike,
            "contracts": {
                "american_call": _contract_dict(american_call),
                "swing_call": _contract_dict(swing_call),
            },
            "policy_values": {
                "american_call_never": path_value_summary(american_call_never.path_values),
                "american_call_immediate": path_value_summary(american_call_immediate.path_values),
                "american_call_raw_lsmc": path_value_summary(american_call_raw_lsmc.path_values),
                "american_call_selected_lsmc": path_value_summary(american_call_selected_lsmc.path_values),
                "american_call_lsmc": path_value_summary(american_call_raw_lsmc.path_values),
                "american_call_fitted_q": path_value_summary(american_call_fitted_q.path_values),
                "american_call_selected_fitted_q": path_value_summary(american_call_selected_fitted_q.path_values),
                "american_call_kernel_fitted_q": path_value_summary(american_call_kernel_q.path_values),
                "american_call_selected_kernel_fitted_q": path_value_summary(american_call_selected_kernel_q.path_values),
                "swing_never": _swing_summary(swing_never_result),
                "swing_positive_margin": _swing_summary(swing_positive_result),
                "swing_quota_aware": _swing_summary(swing_quota_result),
                "swing_lsmc": _swing_summary(swing_lsmc_result),
            },
            "paired_metrics": pairs,
            "rl_training": {
                "model_type": model,
                "path_source": "generated_in_memory",
                "path_count": int(rl_training_matrices.prices.shape[0]),
                "horizon_steps": int(rl_training_matrices.prices.shape[1] - 1),
                "seed": config.rl_training_seed,
                "american_call_fitted_q": _fitted_q_training_summary(american_call_fitted_q_train),
                "american_call_kernel_fitted_q": _fitted_q_training_summary(american_call_kernel_q_train),
                "kernel_config": {
                    "n_rff_features": config.kernel_rff_features,
                    "length_scale": config.kernel_length_scale,
                    "feature_seed": config.kernel_feature_seed,
                },
            },
            "rl_validation": {
                "model_type": model,
                "path_source": "generated_in_memory",
                "path_count": int(rl_validation_matrices.prices.shape[0]),
                "horizon_steps": int(rl_validation_matrices.prices.shape[1] - 1),
                "seed": config.rl_validation_seed,
                "american_call_fitted_q": american_call_fitted_q_selection.validation_metrics,
                "american_call_kernel_fitted_q": american_call_kernel_q_selection.validation_metrics,
            },
            "lsmc_training": {
                "model_type": model,
                "path_source": "generated_in_memory",
                "path_count": int(lsmc_training_matrices.prices.shape[0]),
                "horizon_steps": int(lsmc_training_matrices.prices.shape[1] - 1),
                "seed": config.lsmc_training_seed,
                "american_call_lsmc": _lsmc_training_summary(american_call_lsmc_train),
                "swing_lsmc": _lsmc_training_summary(swing_lsmc_train),
            },
            "lsmc_validation": {
                "model_type": model,
                "path_source": "generated_in_memory",
                "path_count": int(lsmc_validation_matrices.prices.shape[0]),
                "horizon_steps": int(lsmc_validation_matrices.prices.shape[1] - 1),
                "seed": config.lsmc_validation_seed,
                "american_call_lsmc": american_call_lsmc_selection.validation_metrics,
            },
            "swing_rl_status": (
                "Not evaluated in this report. The current RL implementation covers American optimal stopping. "
                "Swing requires a separate remaining-volume state and action-grid Fitted-Q/ADP policy."
            ),
        }
        plot_inputs[model] = {
            "american_call_never": american_call_never,
            "american_call_immediate": american_call_immediate,
            "american_call_raw_lsmc": american_call_raw_lsmc,
            "american_call_selected_lsmc": american_call_selected_lsmc,
            "american_call_fitted_q": american_call_fitted_q,
            "american_call_selected_fitted_q": american_call_selected_fitted_q,
            "american_call_kernel_q": american_call_kernel_q,
            "american_call_selected_kernel_q": american_call_selected_kernel_q,
            "swing_positive": swing_positive_result,
            "swing_quota": swing_quota_result,
            "swing_lsmc": swing_lsmc_result,
        }

    plots = create_plots(config.output_dir, plot_inputs)
    metrics = {
        "config": {
            "database_path": str(config.database_path),
            "symbol": config.symbol,
            "interval": config.interval,
            "output_dir": str(config.output_dir),
            "risk_free_rate": config.risk_free_rate,
            "strike_moneyness": config.strike_moneyness,
            "n_paths_if_simulated": config.n_paths_if_simulated,
            "horizon_steps": config.horizon_steps,
            "seed": config.seed,
            "bootstrap_seed": config.bootstrap_seed,
            "n_bootstrap": config.n_bootstrap,
            "rl_training_model_type": "model_specific",
            "rl_training_paths": config.rl_training_paths,
            "rl_training_seed": config.rl_training_seed,
            "rl_validation_paths": config.rl_validation_paths,
            "rl_validation_seed": config.rl_validation_seed,
            "lsmc_training_paths": config.lsmc_training_paths,
            "lsmc_training_seed": config.lsmc_training_seed,
            "lsmc_validation_paths": config.lsmc_validation_paths,
            "lsmc_validation_seed": config.lsmc_validation_seed,
            "deployment_min_ci_low": config.deployment_min_ci_low,
            "deployment_min_cvar_5_delta": config.deployment_min_cvar_5_delta,
            "kernel_rff_features": config.kernel_rff_features,
            "kernel_length_scale": config.kernel_length_scale,
            "kernel_feature_seed": config.kernel_feature_seed,
            "use_existing_paths": config.use_existing_paths,
        },
        "evaluation_scope": (
            "Frozen-policy smoke evaluation on identical path rows per model. The evaluator does not fit "
            "or tune policies on these paths. LSMC policies are trained on independent model-specific "
            "Monte-Carlo paths; American Fitted-Q and kernel Fitted-Q policies are trained on independent "
            "model-specific paths. Learned American policies are deployment-gated on independent validation paths "
            "before selected policies are replayed on the evaluation set. These artifacts are not final fair values."
        ),
        "path_quality": path_quality,
        "models": model_metrics,
        "plots": {name: str(path) for name, path in plots.items()},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    (config.output_dir / "README.md").write_text(render_report(metrics), encoding="utf-8")
    for model, data in model_metrics.items():
        report_name = f"{model}_README.md"
        (config.output_dir / report_name).write_text(render_model_report(metrics, model, data), encoding="utf-8")
    return metrics


def create_plots(output_dir: Path, plot_inputs: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    plt.figure(figsize=(10.5, 5.2))
    for model, data in plot_inputs.items():
        plt.hist(
            data["american_call_never"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call never",
        )
        plt.hist(
            data["american_call_immediate"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call immediate",
        )
        plt.hist(
            data["american_call_fitted_q"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call fitted-Q",
        )
        plt.hist(
            data["american_call_raw_lsmc"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call raw LSMC",
        )
        plt.hist(
            data["american_call_selected_lsmc"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call selected",
        )
        plt.hist(
            data["american_call_kernel_q"].path_values,
            bins=28,
            alpha=0.32,
            density=True,
            label=f"{model} call kernel-Q",
        )
    plt.title("Pathwise American call value distributions")
    plt.xlabel("Discounted path value")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    paths["value_distributions"] = output_dir / "value_distributions.png"
    plt.savefig(paths["value_distributions"], dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for model, data in plot_inputs.items():
        immediate_delta = data["american_call_immediate"].path_values - data["american_call_never"].path_values
        raw_lsmc_delta = data["american_call_raw_lsmc"].path_values - data["american_call_never"].path_values
        selected_lsmc_delta = data["american_call_selected_lsmc"].path_values - data["american_call_never"].path_values
        fitted_q_delta = data["american_call_fitted_q"].path_values - data["american_call_never"].path_values
        selected_fitted_q_delta = (
            data["american_call_selected_fitted_q"].path_values - data["american_call_never"].path_values
        )
        kernel_q_delta = data["american_call_kernel_q"].path_values - data["american_call_never"].path_values
        selected_kernel_q_delta = (
            data["american_call_selected_kernel_q"].path_values - data["american_call_never"].path_values
        )
        swing_delta = data["swing_lsmc"].path_values - data["swing_quota"].path_values
        axes[0].hist(immediate_delta, bins=28, alpha=0.28, density=True, label=f"{model} immediate")
        axes[0].hist(raw_lsmc_delta, bins=28, alpha=0.28, density=True, label=f"{model} raw LSMC")
        axes[0].hist(selected_lsmc_delta, bins=28, alpha=0.28, density=True, label=f"{model} selected")
        axes[0].hist(fitted_q_delta, bins=28, alpha=0.28, density=True, label=f"{model} fitted-Q")
        axes[0].hist(
            selected_fitted_q_delta,
            bins=28,
            alpha=0.28,
            density=True,
            label=f"{model} selected fitted-Q",
        )
        axes[0].hist(kernel_q_delta, bins=28, alpha=0.28, density=True, label=f"{model} kernel-Q")
        axes[0].hist(
            selected_kernel_q_delta,
            bins=28,
            alpha=0.28,
            density=True,
            label=f"{model} selected kernel-Q",
        )
        axes[1].hist(swing_delta, bins=28, alpha=0.42, density=True, label=model)
    axes[0].set_title("American call: policy minus never")
    axes[0].set_xlabel("Paired delta")
    axes[0].set_ylabel("Density")
    axes[1].set_title("Swing: LSMC minus quota-aware")
    axes[1].set_xlabel("Paired delta")
    for ax in axes:
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    paths["paired_delta_distribution"] = output_dir / "paired_delta_distribution.png"
    plt.savefig(paths["paired_delta_distribution"], dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for model, data in plot_inputs.items():
        profile = _american_exercise_profile(data["american_call_immediate"])
        if not profile.empty:
            axes[0].plot(profile["step"], profile["exercise_probability"], linewidth=1.2, label=model)
        raw_lsmc_profile = _american_exercise_profile(data["american_call_raw_lsmc"])
        if not raw_lsmc_profile.empty:
            axes[0].plot(
                raw_lsmc_profile["step"],
                raw_lsmc_profile["exercise_probability"],
                linewidth=1.2,
                linestyle="-.",
                label=f"{model} raw LSMC",
            )
        selected_lsmc_profile = _american_exercise_profile(data["american_call_selected_lsmc"])
        if not selected_lsmc_profile.empty:
            axes[0].plot(
                selected_lsmc_profile["step"],
                selected_lsmc_profile["exercise_probability"],
                linewidth=1.2,
                linestyle=(0, (3, 1, 1, 1)),
                label=f"{model} selected",
            )
        fitted_q_profile = _american_exercise_profile(data["american_call_fitted_q"])
        if not fitted_q_profile.empty:
            axes[0].plot(
                fitted_q_profile["step"],
                fitted_q_profile["exercise_probability"],
                linewidth=1.2,
                linestyle="--",
                label=f"{model} fitted-Q",
            )
        selected_fitted_q_profile = _american_exercise_profile(data["american_call_selected_fitted_q"])
        if not selected_fitted_q_profile.empty:
            axes[0].plot(
                selected_fitted_q_profile["step"],
                selected_fitted_q_profile["exercise_probability"],
                linewidth=1.2,
                linestyle=(0, (2, 2)),
                label=f"{model} selected fitted-Q",
            )
        kernel_q_profile = _american_exercise_profile(data["american_call_kernel_q"])
        if not kernel_q_profile.empty:
            axes[0].plot(
                kernel_q_profile["step"],
                kernel_q_profile["exercise_probability"],
                linewidth=1.2,
                linestyle=":",
                label=f"{model} kernel-Q",
            )
        selected_kernel_q_profile = _american_exercise_profile(data["american_call_selected_kernel_q"])
        if not selected_kernel_q_profile.empty:
            axes[0].plot(
                selected_kernel_q_profile["step"],
                selected_kernel_q_profile["exercise_probability"],
                linewidth=1.2,
                linestyle=(0, (1, 2)),
                label=f"{model} selected kernel-Q",
            )
        swing_profile = data["swing_lsmc"].nomination_profile
        if not swing_profile.empty:
            axes[1].plot(swing_profile["step"], swing_profile["mean_volume"], linewidth=1.2, label=model)
    axes[0].set_title("American call exercise profile")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Exercise probability")
    axes[1].set_title("LSMC swing nomination profile")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean nominated volume")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()
    paths["exercise_nomination_profile"] = output_dir / "exercise_nomination_profile.png"
    plt.savefig(paths["exercise_nomination_profile"], dpi=140)
    plt.close()

    return paths


def render_report(metrics: dict[str, Any]) -> str:
    plots = metrics["plots"]
    evaluation_setup = "; ".join(
        f"{model}: `{data['path_count']}` paths, `{data['horizon_steps']}` steps, strike `{_fmt(data['strike'])}`"
        for model, data in metrics["models"].items()
    )
    lines = [
        f"# LSMC Policy-Evaluation Test: {metrics['config']['symbol']}",
        "",
        "## What Is Being Tested?",
        "",
        (
            "Frozen decision policies are evaluated on identical out-of-sample "
            "paths. Each policy receives only current state information, creates "
            "one discounted path value per path, and is compared pairwise:"
        ),
        "",
        "```text",
        "delta_i = value_policy_A_i - value_policy_B_i",
        "```",
        "",
        (
            "American and swing LSMC policies are trained on independent "
            "model-specific Monte-Carlo paths. The raw American LSMC candidate "
            "is reported separately. A validation-selected deployment wrapper "
            "then uses a separate validation path set to decide whether to deploy "
            "the learned exercise policy or the never-exercise fallback before "
            "replay on evaluation paths. American linear Fitted-Q and kernel "
            "Fitted-Q are trained on independent model-specific Monte-Carlo paths, "
            "validated on another independent model-specific set, and replayed "
            "both as raw candidates and as selected deployments."
        ),
        "",
        "## Why This Test Matters",
        "",
        (
            "LSMC and RL/ADP methods should not be ranked by in-sample training "
            "values. The fair comparison is to freeze a policy, choose deployment "
            "rules on validation data only, replay on identical unseen paths, and "
            "inspect paired pathwise deltas."
        ),
        "",
        "## What The Metrics Mean",
        "",
        (
            "The report checks whether a policy creates higher value under the "
            "same market, cost, and contract assumptions. Mean delta, bootstrap "
            "confidence intervals, tail deltas, and constraint diagnostics matter "
            "more than a single average value."
        ),
        "",
        "## Results",
        "",
        (
            f"Setup: `{metrics['config']['symbol']}`, `{metrics['config']['interval']}`, "
            f"evaluation paths by generator: {evaluation_setup}; "
            f"bootstrap seed `{metrics['config']['bootstrap_seed']}`. "
            f"American Fitted-Q training: `{metrics['config']['rl_training_paths']}` "
            "model-specific paths, "
            f"seed `{metrics['config']['rl_training_seed']}`. "
            f"RL validation: `{metrics['config']['rl_validation_paths']}` "
            "model-specific paths, "
            f"seed `{metrics['config']['rl_validation_seed']}`. "
            f"LSMC training: `{metrics['config']['lsmc_training_paths']}` "
            f"model-specific paths, seed `{metrics['config']['lsmc_training_seed']}`. "
            f"LSMC validation: `{metrics['config']['lsmc_validation_paths']}` "
            f"model-specific paths, seed `{metrics['config']['lsmc_validation_seed']}`. "
            f"Kernel Fitted-Q: `{metrics['config']['kernel_rff_features']}` RFF features, "
            f"length scale `{metrics['config']['kernel_length_scale']}`. "
            f"Deployment gate: validation CI-low > `{metrics['config']['deployment_min_ci_low']}`"
            f"{_deployment_cvar_text(metrics['config']['deployment_min_cvar_5_delta'])}."
        ),
        "",
        "Only GJR-GARCH paths are used in this report. HAR-RV simulations and American puts are outside the current evaluation scope.",
        "",
        _model_result_section("GJR-GARCH", "gjr_garch", metrics["models"]["gjr_garch"], metrics),
        "",
        "## Bottom Line",
        "",
        (
            "American LSMC, swing LSMC, linear Fitted-Q, and kernel Fitted-Q "
            "are now all evaluated as frozen replay policies where applicable. "
            "Raw learned American candidates remain visible. Separate "
            "selected-deployment lines use independent validation gates against "
            "never-exercise before test evaluation, so weak learned exercise rules "
            "are not deployed. This is policy selection, not a European payoff "
            "floor and not a pure model value."
        ),
        "",
        (
            "A final LSMC-vs-RL ranking still requires risk-neutral or explicitly "
            "documented drift assumptions, larger sensitivity runs, and the same "
            "frozen-policy protocol across American and swing instruments."
        ),
        "",
        "![Paired deltas]({})".format(Path(plots["paired_delta_distribution"]).name),
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m lsmc_rl.analysis.policy_evaluation_report --output-dir outputs/policy_evaluation_front",
        "```",
        "",
        "Detailed metrics are stored in `metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def render_model_report(metrics: dict[str, Any], model_key: str, data: dict[str, Any]) -> str:
    title = "GJR-GARCH" if model_key == "gjr_garch" else "HAR-RV"
    lines = [
        f"# LSMC Policy-Evaluation Test: {title}",
        "",
        "## What Is Being Tested?",
        "",
        (
            "A frozen policy is evaluated using only current state information. "
            "Each path produces one discounted value, and values are compared "
            "pairwise against baselines."
        ),
        "",
        "## Why This Test Matters",
        "",
        (
            "This model report keeps interpretation separated by path generator. "
            "A policy evaluation is only as credible as the paths on which it is "
            "measured."
        ),
        "",
        "## What The Metrics Mean",
        "",
        (
            "The section shows how the frozen policies perform on this specific "
            "path generator. These results should not be merged with another "
            "generator's results."
        ),
        "",
        "## Results",
        "",
        _model_result_section(title, model_key, data, metrics),
        "",
        "Detailed metrics are stored in `metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def _load_or_simulate_paths(model_type: str, config: PolicyEvaluationReportConfig) -> pd.DataFrame:
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


def _load_or_simulate_rl_training_paths(
    model_type: str,
    config: PolicyEvaluationReportConfig,
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
                n_paths=config.rl_training_paths,
                seed=config.rl_training_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _load_or_simulate_rl_validation_paths(
    model_type: str,
    config: PolicyEvaluationReportConfig,
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
                n_paths=config.rl_validation_paths,
                seed=config.rl_validation_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _load_or_simulate_lsmc_training_paths(
    model_type: str,
    config: PolicyEvaluationReportConfig,
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
                n_paths=config.lsmc_training_paths,
                seed=config.lsmc_training_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _load_or_simulate_lsmc_validation_paths(
    model_type: str,
    config: PolicyEvaluationReportConfig,
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
                n_paths=config.lsmc_validation_paths,
                seed=config.lsmc_validation_seed,
                output_path=None,
                steps_per_day=config.steps_per_day,
            )
        )
        cache[key] = simulation.paths
    return cache[key]


def _path_source_for_model(model_type: str, config: PolicyEvaluationReportConfig) -> str:
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


def _contract_dict(contract: Any) -> dict[str, Any]:
    return dict(contract.__dict__)


def _swing_summary(result: Any) -> dict[str, float]:
    summary = path_value_summary(result.path_values)
    path_results = result.path_results
    summary.update(
        {
            "mean_total_volume": float(path_results["total_volume"].mean()),
            "mean_remaining_volume": float(path_results["remaining_volume"].mean()),
            "mean_shortfall_volume": float(path_results["shortfall_volume"].mean()),
            "mean_constraint_violations": float(path_results["constraint_violations"].mean()),
            "mean_costs": float(path_results["costs"].mean()),
        }
    )
    return summary


def _fitted_q_training_summary(result: Any) -> dict[str, Any]:
    diagnostics = result.training_diagnostics
    regression_r2 = diagnostics["regression_r2"].dropna() if "regression_r2" in diagnostics else pd.Series(dtype=float)
    fitted_statuses = {"ridge", "kernel_ridge"}
    exercise_rows = diagnostics.loc[diagnostics["exercise_date"]] if "exercise_date" in diagnostics else diagnostics
    return {
        "in_sample_value": path_value_summary(result.path_values),
        "fitted_steps": int(diagnostics["status"].isin(fitted_statuses).sum()) if "status" in diagnostics else 0,
        "fallback_steps": int((diagnostics["status"] == "constant_fallback").sum()) if "status" in diagnostics else 0,
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
        "mean_abs_bellman_residual": (
            float(exercise_rows["mean_abs_bellman_residual"].mean())
            if "mean_abs_bellman_residual" in exercise_rows
            else float("nan")
        ),
        "max_negative_continuation_share_before_clip": (
            float(exercise_rows["negative_continuation_share_before_clip"].max())
            if "negative_continuation_share_before_clip" in exercise_rows
            else float("nan")
        ),
        "mean_exercised_paths_per_step": (
            float(diagnostics.loc[diagnostics["exercise_date"], "exercised_paths"].mean())
            if {"exercise_date", "exercised_paths"}.issubset(diagnostics.columns)
            else 0.0
        ),
    }


def _lsmc_training_summary(result: Any) -> dict[str, Any]:
    diagnostics = result.regression_diagnostics
    regression_r2 = diagnostics["regression_r2"].dropna() if "regression_r2" in diagnostics else pd.Series(dtype=float)
    summary: dict[str, Any] = {
        "in_sample_value": path_value_summary(result.path_values),
        "fitted_steps": int((diagnostics["status"] == "ridge").sum()) if "status" in diagnostics else 0,
        "fallback_steps": int((diagnostics["status"] == "constant_fallback").sum()) if "status" in diagnostics else 0,
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
    }
    if "exercised_paths" in diagnostics:
        summary["mean_exercised_paths_per_step"] = float(diagnostics["exercised_paths"].mean())
    if "mean_start_state_value" in diagnostics:
        summary["mean_start_state_value"] = float(diagnostics["mean_start_state_value"].mean())
    return summary


def _american_exercise_profile(result: Any) -> pd.DataFrame:
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


def _paired_table(pairs: dict[str, Any]) -> str:
    rows = []
    for name, values in pairs.items():
        ci_low, ci_high = values["bootstrap_mean_delta_ci95"]
        rows.append(
            [
                name,
                _fmt(values["mean_delta"]),
                _fmt(values["median_delta"]),
                _fmt(values["share_delta_positive"]),
                f"{_fmt(values['q05_delta'])} / {_fmt(values['q50_delta'])} / {_fmt(values['q95_delta'])}",
                f"{_fmt(ci_low)} / {_fmt(ci_high)}",
                _fmt(values["cvar_5_delta"]),
                _fmt(values["standard_error"]),
            ]
        )
    return _markdown_table(
        ["comparison", "mean delta", "median", "share > 0", "q05/q50/q95", "boot CI95 mean", "CVaR 5%", "stderr"],
        rows,
    )


def _swing_row(name: str, summary: dict[str, Any]) -> list[str]:
    return [
        name,
        _fmt(summary["mean"]),
        _fmt(summary["mean_total_volume"]),
        _fmt(summary["mean_remaining_volume"]),
        _fmt(summary["mean_constraint_violations"]),
        _fmt(summary["mean_costs"]),
    ]


def _model_result_section(label: str, model_key: str, data: dict[str, Any], metrics: dict[str, Any]) -> str:
    call = data["paired_metrics"]["american_call_immediate_vs_never"]
    raw_lsmc = data["paired_metrics"]["american_call_raw_lsmc_vs_never"]
    selected_lsmc = data["paired_metrics"]["american_call_selected_lsmc_vs_never"]
    raw_lsmc_vs_fitted = data["paired_metrics"]["american_call_raw_lsmc_vs_fitted_q"]
    selected_lsmc_vs_fitted = data["paired_metrics"]["american_call_selected_lsmc_vs_fitted_q"]
    selected_lsmc_vs_selected_fitted = data["paired_metrics"]["american_call_selected_lsmc_vs_selected_fitted_q"]
    fitted_q = data["paired_metrics"]["american_call_fitted_q_vs_never"]
    selected_fitted_q = data["paired_metrics"]["american_call_selected_fitted_q_vs_never"]
    kernel_q = data["paired_metrics"]["american_call_kernel_fitted_q_vs_never"]
    selected_kernel_q = data["paired_metrics"]["american_call_selected_kernel_fitted_q_vs_never"]
    kernel_vs_linear = data["paired_metrics"]["american_call_kernel_fitted_q_vs_linear_fitted_q"]
    selected_kernel_vs_linear = data["paired_metrics"]["american_call_selected_kernel_fitted_q_vs_selected_linear_fitted_q"]
    swing = data["paired_metrics"]["swing_quota_vs_positive_margin"]
    swing_lsmc = data["paired_metrics"]["swing_lsmc_vs_quota_aware"]
    quality = _path_quality_text(metrics.get("path_quality", {}), model_key)
    rl_training = data.get("rl_training", {})
    rl_validation = data.get("rl_validation", {})
    lsmc_training = data.get("lsmc_training", {})
    lsmc_validation = data.get("lsmc_validation", {})
    call_validation = (lsmc_validation.get("american_call_lsmc") or {})
    call_fitted_q_validation = (rl_validation.get("american_call_fitted_q") or {})
    call_kernel_q_validation = (rl_validation.get("american_call_kernel_fitted_q") or {})
    rl_text = (
        f"American Fitted-Q training: `{rl_training.get('path_count', 'n/a')}` "
        f"`{rl_training.get('model_type', model_key)}` paths, seed `{rl_training.get('seed', 'n/a')}`. "
        f"RL validation: `{rl_validation.get('path_count', 'n/a')}` "
        f"`{rl_validation.get('model_type', model_key)}` paths, seed `{rl_validation.get('seed', 'n/a')}`; "
        f"selected call linear `{call_fitted_q_validation.get('selected_policy_name', 'n/a')}`, "
        f"call kernel `{call_kernel_q_validation.get('selected_policy_name', 'n/a')}`. "
        f"Kernel RFF features: `{(rl_training.get('kernel_config') or {}).get('n_rff_features', 'n/a')}`. "
        f"LSMC training: `{lsmc_training.get('path_count', 'n/a')}` `{model_key}` paths, "
        f"seed `{lsmc_training.get('seed', 'n/a')}`. "
        f"LSMC validation selected call `{call_validation.get('selected_policy_name', 'n/a')}`."
    )
    table = _markdown_table(
        ["Comparison", "Mean delta", "Bootstrap 95% CI", "Median", "CVaR 5%", "Short assessment"],
        [
            [
                "American call: immediate vs. never",
                _fmt(call["mean_delta"]),
                _ci(call),
                _fmt(call["median_delta"]),
                _fmt(call["cvar_5_delta"]),
                "Immediate worse",
            ],
            [
                "American call: raw LSMC candidate vs. never",
                _fmt(raw_lsmc["mean_delta"]),
                _ci(raw_lsmc),
                _fmt(raw_lsmc["median_delta"]),
                _fmt(raw_lsmc["cvar_5_delta"]),
                _delta_verdict(raw_lsmc, "Raw LSMC"),
            ],
            [
                "American call: selected deployment vs. never",
                _fmt(selected_lsmc["mean_delta"]),
                _ci(selected_lsmc),
                _fmt(selected_lsmc["median_delta"]),
                _fmt(selected_lsmc["cvar_5_delta"]),
                _delta_verdict(selected_lsmc, "Selected deployment"),
            ],
            [
                "American call: raw LSMC candidate vs. fitted-Q",
                _fmt(raw_lsmc_vs_fitted["mean_delta"]),
                _ci(raw_lsmc_vs_fitted),
                _fmt(raw_lsmc_vs_fitted["median_delta"]),
                _fmt(raw_lsmc_vs_fitted["cvar_5_delta"]),
                _delta_verdict(raw_lsmc_vs_fitted, "Raw LSMC"),
            ],
            [
                "American call: selected LSMC deployment vs. raw fitted-Q",
                _fmt(selected_lsmc_vs_fitted["mean_delta"]),
                _ci(selected_lsmc_vs_fitted),
                _fmt(selected_lsmc_vs_fitted["median_delta"]),
                _fmt(selected_lsmc_vs_fitted["cvar_5_delta"]),
                _delta_verdict(selected_lsmc_vs_fitted, "Selected deployment"),
            ],
            [
                "American call: selected LSMC deployment vs. selected fitted-Q",
                _fmt(selected_lsmc_vs_selected_fitted["mean_delta"]),
                _ci(selected_lsmc_vs_selected_fitted),
                _fmt(selected_lsmc_vs_selected_fitted["median_delta"]),
                _fmt(selected_lsmc_vs_selected_fitted["cvar_5_delta"]),
                _delta_verdict(selected_lsmc_vs_selected_fitted, "Selected LSMC"),
            ],
            [
                "American call: raw fitted-Q vs. never",
                _fmt(fitted_q["mean_delta"]),
                _ci(fitted_q),
                _fmt(fitted_q["median_delta"]),
                _fmt(fitted_q["cvar_5_delta"]),
                _delta_verdict(fitted_q, "Fitted-Q"),
            ],
            [
                "American call: selected fitted-Q deployment vs. never",
                _fmt(selected_fitted_q["mean_delta"]),
                _ci(selected_fitted_q),
                _fmt(selected_fitted_q["median_delta"]),
                _fmt(selected_fitted_q["cvar_5_delta"]),
                _delta_verdict(selected_fitted_q, "Selected Fitted-Q"),
            ],
            [
                "American call: raw kernel fitted-Q vs. never",
                _fmt(kernel_q["mean_delta"]),
                _ci(kernel_q),
                _fmt(kernel_q["median_delta"]),
                _fmt(kernel_q["cvar_5_delta"]),
                _delta_verdict(kernel_q, "Kernel-Q"),
            ],
            [
                "American call: selected kernel fitted-Q deployment vs. never",
                _fmt(selected_kernel_q["mean_delta"]),
                _ci(selected_kernel_q),
                _fmt(selected_kernel_q["median_delta"]),
                _fmt(selected_kernel_q["cvar_5_delta"]),
                _delta_verdict(selected_kernel_q, "Selected Kernel-Q"),
            ],
            [
                "American call: raw kernel fitted-Q vs. raw linear fitted-Q",
                _fmt(kernel_vs_linear["mean_delta"]),
                _ci(kernel_vs_linear),
                _fmt(kernel_vs_linear["median_delta"]),
                _fmt(kernel_vs_linear["cvar_5_delta"]),
                _delta_verdict(kernel_vs_linear, "Kernel-Q"),
            ],
            [
                "American call: selected kernel fitted-Q vs. selected linear Fitted-Q",
                _fmt(selected_kernel_vs_linear["mean_delta"]),
                _ci(selected_kernel_vs_linear),
                _fmt(selected_kernel_vs_linear["median_delta"]),
                _fmt(selected_kernel_vs_linear["cvar_5_delta"]),
                _delta_verdict(selected_kernel_vs_linear, "Selected Kernel-Q"),
            ],
            [
                "Swing: LSMC vs. quota-aware",
                _fmt(swing_lsmc["mean_delta"]),
                _ci(swing_lsmc),
                _fmt(swing_lsmc["median_delta"]),
                _fmt(swing_lsmc["cvar_5_delta"]),
                _delta_verdict(swing_lsmc, "Swing LSMC"),
            ],
            [
                "Swing: quota-aware vs. positive-margin",
                _fmt(swing["mean_delta"]),
                _ci(swing),
                _fmt(swing["median_delta"]),
                _fmt(swing["cvar_5_delta"]),
                "Quota-aware better",
            ],
        ],
    )
    return "\n".join(
        [
            f"### {label}",
            "",
            quality,
            "",
            rl_text,
            "",
            _validation_gate_table(data),
            "",
            table,
            "",
            _model_interpretation(
                model_key,
                call,
                raw_lsmc,
                selected_lsmc,
                fitted_q,
                selected_fitted_q,
                kernel_q,
                selected_kernel_q,
                swing,
                swing_lsmc,
            ),
        ]
    )


def _validation_gate_table(data: dict[str, Any]) -> str:
    lsmc_validation = data.get("lsmc_validation", {})
    rl_validation = data.get("rl_validation", {})
    rows = []
    candidates = [
        ("American call LSMC", lsmc_validation.get("american_call_lsmc") or {}),
        ("American call linear Fitted-Q", rl_validation.get("american_call_fitted_q") or {}),
        ("American call kernel Fitted-Q", rl_validation.get("american_call_kernel_fitted_q") or {}),
    ]
    for label, metrics in candidates:
        if not metrics:
            continue
        gate = metrics.get("deployment_gate") or {}
        rows.append(
            [
                label,
                _fmt(metrics.get("mean_delta")),
                _ci(metrics),
                _fmt(metrics.get("cvar_5_delta")),
                _pass_fail(gate.get("ci_gate_passed")),
                _pass_fail(gate.get("cvar_gate_passed")),
                str(metrics.get("selected_policy_name", "n/a")),
            ]
        )
    if not rows:
        return "Validation gate diagnostics: n/a"
    return "\n".join(
        [
            "Validation gate diagnostics:",
            "",
            _markdown_table(
                ["Candidate", "Validation mean", "Validation CI95", "Validation CVaR5", "CI gate", "Tail gate", "Selected policy"],
                rows,
            ),
        ]
    )


def _pass_fail(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "n/a"


def _model_interpretation(
    model_key: str,
    call: dict[str, Any],
    raw_lsmc: dict[str, Any],
    selected_lsmc: dict[str, Any],
    fitted_q: dict[str, Any],
    selected_fitted_q: dict[str, Any],
    kernel_q: dict[str, Any],
    selected_kernel_q: dict[str, Any],
    swing: dict[str, Any],
    swing_lsmc: dict[str, Any],
) -> str:
    prefix = "On the GJR-GARCH paths" if model_key == "gjr_garch" else f"On {model_key} paths"
    return (
        f"{prefix}, immediate intrinsic exercise has mean delta `{_fmt(call['mean_delta'])}` "
        f"for the American call versus never exercising early. The raw LSMC candidate has mean delta "
        f"`{_fmt(raw_lsmc['mean_delta'])}` against Never-Early-Exercise, while the "
        f"validation-selected deployment has `{_fmt(selected_lsmc['mean_delta'])}`. The "
        f"model-specific raw linear Fitted-Q policy has mean delta `{_fmt(fitted_q['mean_delta'])}` against "
        f"Never-Early-Exercise and selected linear Fitted-Q has `{_fmt(selected_fitted_q['mean_delta'])}`. "
        f"Raw kernel Fitted-Q has `{_fmt(kernel_q['mean_delta'])}`, while selected kernel Fitted-Q has "
        f"`{_fmt(selected_kernel_q['mean_delta'])}`. "
        f"For swing, LSMC minus quota-aware has mean delta `{_fmt(swing_lsmc['mean_delta'])}`; "
        f"quota-aware minus positive-margin is `{_fmt(swing['mean_delta'])}`."
    )


def _delta_verdict(metrics: dict[str, Any], label: str) -> str:
    low, high = metrics.get("bootstrap_mean_delta_ci95", [float("nan"), float("nan")])
    if low > 0.0:
        return f"{label} better"
    if high < 0.0:
        return f"{label} worse"
    if metrics["mean_delta"] > 0.0:
        return f"{label} mean positive, uncertain"
    if metrics["mean_delta"] < 0.0:
        return f"{label} mean negative, uncertain"
    return f"{label} neutral"


def _load_path_quality_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = raw.get("models", {})
    summary: dict[str, Any] = {}
    for model, data in models.items():
        summary[model] = {
            "qlike": data.get("qlike"),
            "normal_nll": data.get("normal_nll"),
            "energy_score": data.get("energy_score"),
            "aggregate_mmd2": (data.get("multiband_mmd") or {}).get("aggregate_mmd2"),
        }
    summary["ranking"] = raw.get("ranking", {})
    return summary


def _path_quality_text(path_quality: dict[str, Any], model_key: str) -> str:
    model = path_quality.get(model_key, {})
    if not model:
        return "Path quality: no path-quality metrics were found for this policy report."
    label = "GJR-GARCH" if model_key == "gjr_garch" else "HAR-RV"
    ranking = path_quality.get("ranking", {})
    best = all(ranking.get(score) == model_key for score in ("qlike", "normal_nll", "energy_score", "aggregate_mmd2"))
    assessment = (
        "currently the stronger path generator"
        if best
        else "currently a weaker comparison and stress case"
    )
    return (
        f"Path quality {label}: QLIKE `{_fmt(model.get('qlike'))}`, "
        f"Energy Score `{_fmt(model.get('energy_score'))}`, "
        f"MMD^2 `{_fmt(model.get('aggregate_mmd2'))}`; {assessment}."
    )


def _ci(metrics: dict[str, Any]) -> str:
    low, high = metrics["bootstrap_mean_delta_ci95"]
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _deployment_cvar_text(value: Any) -> str:
    if value is None:
        return ""
    return f" and validation CVaR5 >= `{_fmt(value)}`"


def _interpretation(metrics: dict[str, Any]) -> str:
    lines = [
        "This run is an engineering check of the paired evaluator, not a final model ranking. "
        "A policy should only be called better when the paired mean delta is economically relevant, "
        "the bootstrap interval is plausibly away from zero, and tail deltas and constraint diagnostics do not deteriorate.",
        "",
        "The current report evaluates frozen LSMC and American RL/ADP policies, but it still does not claim final "
        "LSMC-vs-RL superiority because the paths are not risk-neutral calibrated and sensitivity runs are limited.",
    ]
    for model, data in metrics["models"].items():
        swing = data["paired_metrics"]["swing_lsmc_vs_quota_aware"]
        ci_low, ci_high = swing["bootstrap_mean_delta_ci95"]
        lines.append(
            f"- `{model}` swing LSMC minus quota-aware mean delta `{_fmt(swing['mean_delta'])}` "
            f"with CI `{_fmt(ci_low)} / {_fmt(ci_high)}` and CVaR5 `{_fmt(swing['cvar_5_delta'])}`."
        )
    return "\n".join(lines)


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


def _ensure_allowed_artifact_dir(path: Path) -> None:
    normalized = path.resolve()
    allowed_roots = [(Path.cwd() / "outputs").resolve(), (Path.cwd() / "runs").resolve()]
    if not any(normalized == root or root in normalized.parents for root in allowed_roots):
        raise ValueError("Policy-evaluation artifacts must be written under outputs/ or runs/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen baseline policies on identical test paths.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output-dir", default="outputs/policy_evaluation_front")
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--strike-moneyness", type=float, default=1.0)
    parser.add_argument("--n-paths-if-simulated", type=int, default=2048)
    parser.add_argument("--horizon-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--bootstrap-seed", type=int, default=20260520)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rl-training-paths", type=int, default=2048)
    parser.add_argument("--rl-training-seed", type=int, default=20260521)
    parser.add_argument("--rl-validation-paths", type=int, default=2048)
    parser.add_argument("--rl-validation-seed", type=int, default=20260524)
    parser.add_argument("--lsmc-training-paths", type=int, default=2048)
    parser.add_argument("--lsmc-training-seed", type=int, default=20260522)
    parser.add_argument("--lsmc-validation-paths", type=int, default=2048)
    parser.add_argument("--lsmc-validation-seed", type=int, default=20260523)
    parser.add_argument("--deployment-min-ci-low", type=float, default=0.0)
    parser.add_argument("--deployment-min-cvar-5-delta", type=float, default=0.0)
    parser.add_argument("--kernel-rff-features", type=int, default=96)
    parser.add_argument("--kernel-length-scale", type=float, default=1.0)
    parser.add_argument("--kernel-feature-seed", type=int, default=20260522)
    parser.add_argument("--ignore-existing-paths", action="store_true")
    args = parser.parse_args(argv)

    config = PolicyEvaluationReportConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        output_dir=Path(args.output_dir).resolve(),
        risk_free_rate=args.risk_free_rate,
        strike_moneyness=args.strike_moneyness,
        n_paths_if_simulated=args.n_paths_if_simulated,
        horizon_steps=args.horizon_steps,
        seed=args.seed,
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
        rl_training_paths=args.rl_training_paths,
        rl_training_seed=args.rl_training_seed,
        rl_validation_paths=args.rl_validation_paths,
        rl_validation_seed=args.rl_validation_seed,
        lsmc_training_paths=args.lsmc_training_paths,
        lsmc_training_seed=args.lsmc_training_seed,
        lsmc_validation_paths=args.lsmc_validation_paths,
        lsmc_validation_seed=args.lsmc_validation_seed,
        deployment_min_ci_low=args.deployment_min_ci_low,
        deployment_min_cvar_5_delta=args.deployment_min_cvar_5_delta,
        kernel_rff_features=args.kernel_rff_features,
        kernel_length_scale=args.kernel_length_scale,
        kernel_feature_seed=args.kernel_feature_seed,
        use_existing_paths=False,
    )
    metrics = run_report(config)
    print(f"Wrote report: {config.output_dir / 'README.md'}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    for model, data in metrics["models"].items():
        call = data["paired_metrics"]["american_call_immediate_vs_never"]
        raw_lsmc = data["paired_metrics"]["american_call_raw_lsmc_vs_never"]
        selected_lsmc = data["paired_metrics"]["american_call_selected_lsmc_vs_never"]
        fitted_q = data["paired_metrics"]["american_call_fitted_q_vs_never"]
        selected_fitted_q = data["paired_metrics"]["american_call_selected_fitted_q_vs_never"]
        kernel_q = data["paired_metrics"]["american_call_kernel_fitted_q_vs_never"]
        selected_kernel_q = data["paired_metrics"]["american_call_selected_kernel_fitted_q_vs_never"]
        swing = data["paired_metrics"]["swing_lsmc_vs_quota_aware"]
        print(
            f"{model}: call_immediate_delta={_fmt(call['mean_delta'])}, "
            f"call_raw_lsmc_delta={_fmt(raw_lsmc['mean_delta'])}, "
            f"call_selected_lsmc_delta={_fmt(selected_lsmc['mean_delta'])}, "
            f"call_fitted_q_delta={_fmt(fitted_q['mean_delta'])}, "
            f"call_selected_fitted_q_delta={_fmt(selected_fitted_q['mean_delta'])}, "
            f"call_kernel_q_delta={_fmt(kernel_q['mean_delta'])}, "
            f"call_selected_kernel_q_delta={_fmt(selected_kernel_q['mean_delta'])}, "
            f"swing_lsmc_delta={_fmt(swing['mean_delta'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
