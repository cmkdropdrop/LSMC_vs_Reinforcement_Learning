"""Run LSMC option valuation diagnostics on GJR-GARCH and HAR-RV paths."""

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

from lsmc_rl.simulation.paths import PathSimulationConfig, run_simulation
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
    "har_rv": Path("outputs/mc_paths_front_har_rv.csv"),
}


@dataclass(frozen=True)
class OptionLSMCReportConfig:
    database_path: Path = Path("ttf_klines_5m_from_1m.sqlite")
    symbol: str = "FRONT"
    interval: str = "5m"
    output_dir: Path = Path("outputs/option_lsmc_report_front")
    risk_free_rate: float = 0.03
    strike_moneyness: float = 1.0
    n_paths_if_simulated: int = 1000
    horizon_steps: int = 288
    seed: int = 20260519
    steps_per_day: int = 288
    use_existing_paths: bool = True


def run_report(config: OptionLSMCReportConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path_frames = {
        model: _load_or_simulate_paths(model, config)
        for model in ("gjr_garch", "har_rv")
    }

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

    valuations: dict[str, Any] = {}
    american_results: dict[str, dict[str, AmericanLSMCResult]] = {}
    swing_results: dict[str, SwingLSMCResult] = {}

    for model, frame in path_frames.items():
        matrices = paths_frame_to_matrices(frame)
        start_price = float(np.median(matrices.prices[:, 0]))
        strike = start_price * config.strike_moneyness
        maturity = int(matrices.prices.shape[1] - 1)
        time_step_years = _infer_time_step_years(frame)

        call_contract = AmericanOptionContract(
            strike=strike,
            option_type="call",
            risk_free_rate=config.risk_free_rate,
            time_step_years=time_step_years,
            maturity_step=maturity,
            exercise_start_step=1,
            exercise_step_interval=1,
        )
        put_contract = AmericanOptionContract(
            strike=strike,
            option_type="put",
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

        call = value_american_option_lsmc(frame, call_contract, regression)
        put = value_american_option_lsmc(frame, put_contract, regression)
        swing = value_swing_option_lsmc(frame, swing_contract, swing_regression)

        american_results[model] = {"call": call, "put": put}
        swing_results[model] = swing
        valuations[model] = {
            "path_source": _path_source_for_model(model, config),
            "path_count": int(matrices.prices.shape[0]),
            "horizon_steps": maturity,
            "start_price": start_price,
            "strike": strike,
            "time_step_years": time_step_years,
            "american_call": _summarize_american(call),
            "american_put": _summarize_american(put),
            "swing_call": _summarize_swing(swing),
        }

    plot_paths = create_plots(config.output_dir, american_results, swing_results)
    metrics = {
        "config": {
            "database_path": str(config.database_path),
            "symbol": config.symbol,
            "interval": config.interval,
            "risk_free_rate": config.risk_free_rate,
            "strike_moneyness": config.strike_moneyness,
            "use_existing_paths": config.use_existing_paths,
            "n_paths_if_simulated": config.n_paths_if_simulated,
            "horizon_steps": config.horizon_steps,
            "seed": config.seed,
            "steps_per_day": config.steps_per_day,
        },
        "valuation_assumption": (
            "Values are diagnostic LSMC estimates under the supplied simulated path measure. "
            "The GJR-GARCH and HAR-RV paths are not yet risk-neutral calibrated."
        ),
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
    american_results: dict[str, dict[str, AmericanLSMCResult]],
    swing_results: dict[str, SwingLSMCResult],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    rows = []
    for model, results in american_results.items():
        rows.append({"model": model, "instrument": "American call", "price": results["call"].price, "stderr": results["call"].stderr})
        rows.append({"model": model, "instrument": "American put", "price": results["put"].price, "stderr": results["put"].stderr})
        rows.append({"model": model, "instrument": "Swing call", "price": swing_results[model].price, "stderr": swing_results[model].stderr})
    valuation_frame = pd.DataFrame(rows)
    pivot = valuation_frame.pivot(index="instrument", columns="model", values="price")
    error = valuation_frame.pivot(index="instrument", columns="model", values="stderr").reindex_like(pivot)
    ax = pivot.plot(kind="bar", figsize=(9.5, 4.8), yerr=error, capsize=3)
    ax.set_title("LSMC valuation by volatility path generator")
    ax.set_ylabel("Value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    paths["valuation_bars"] = output_dir / "valuation_bars.png"
    plt.savefig(paths["valuation_bars"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    for model, results in american_results.items():
        profile = results["put"].exercise_profile.copy()
        profile = profile.loc[~profile["is_maturity"]]
        if profile.empty:
            continue
        plt.plot(profile["step"], profile["exercise_probability"], linewidth=1.2, label=model)
    plt.title("American put early-exercise profile")
    plt.xlabel("Step")
    plt.ylabel("Exercise probability")
    plt.legend()
    plt.tight_layout()
    paths["american_put_exercise"] = output_dir / "american_put_exercise.png"
    plt.savefig(paths["american_put_exercise"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    for model, result in swing_results.items():
        profile = result.exercise_profile
        plt.plot(profile["step"], profile["mean_volume"], linewidth=1.2, label=model)
    plt.title("Swing option mean nomination profile")
    plt.xlabel("Step")
    plt.ylabel("Mean nominated volume")
    plt.legend()
    plt.tight_layout()
    paths["swing_nomination_profile"] = output_dir / "swing_nomination_profile.png"
    plt.savefig(paths["swing_nomination_profile"], dpi=140)
    plt.close()

    plt.figure(figsize=(10.0, 4.8))
    for model, result in swing_results.items():
        plt.hist(result.path_values, bins=30, alpha=0.42, density=True, label=f"{model} swing")
    plt.title("Swing path-value distribution")
    plt.xlabel("Discounted path value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    paths["swing_value_distribution"] = output_dir / "swing_value_distribution.png"
    plt.savefig(paths["swing_value_distribution"], dpi=140)
    plt.close()

    return paths


def render_report(metrics: dict[str, Any]) -> str:
    models = metrics["models"]
    plots = metrics["plots"]
    lines = [
        f"# LSMC Option Valuation Report: {metrics['config']['symbol']}",
        "",
        metrics["valuation_assumption"],
        "",
        "## Setup",
        "",
        f"- Symbol: `{metrics['config']['symbol']}`",
        f"- Interval: `{metrics['config']['interval']}`",
        f"- Risk-free rate: `{_fmt(metrics['config']['risk_free_rate'])}`",
        f"- Strike moneyness: `{_fmt(metrics['config']['strike_moneyness'])}`",
        "- American contracts: ATM call and put, exercise possible at every simulated step.",
        "- Reported American values use the better of the fitted LSMC exercise policy and the European never-exercise baseline.",
        "- Swing contract: call-style gas nomination, hourly exercise grid (`12` x 5-minute steps), max `1` unit per exercise, total band `2` to `6` units.",
        "",
        f"![Valuation bars]({Path(plots['valuation_bars']).name})",
        "",
        "## Values",
        "",
        _markdown_table(
            ["model", "paths", "start", "strike", "American call", "European call", "American put", "European put", "Swing call"],
            [
                [
                    model,
                    str(data["path_count"]),
                    _fmt(data["start_price"]),
                    _fmt(data["strike"]),
                    _fmt(data["american_call"]["price"]),
                    _fmt(data["american_call"]["european_value"]),
                    _fmt(data["american_put"]["price"]),
                    _fmt(data["american_put"]["european_value"]),
                    _fmt(data["swing_call"]["price"]),
                ]
                for model, data in models.items()
            ],
        ),
        "",
        "## Exercise Diagnostics",
        "",
        _markdown_table(
            ["model", "call early ex.", "put early ex.", "swing mean volume", "swing q05/q50/q95 volume", "median swing R2"],
            [
                [
                    model,
                    _fmt(data["american_call"]["early_exercise_probability"]),
                    _fmt(data["american_put"]["early_exercise_probability"]),
                    _fmt(data["swing_call"]["mean_total_volume"]),
                    f"{_fmt(data['swing_call']['q05_total_volume'])} / {_fmt(data['swing_call']['q50_total_volume'])} / {_fmt(data['swing_call']['q95_total_volume'])}",
                    _fmt(data["swing_call"]["median_regression_r2"]),
                ]
                for model, data in models.items()
            ],
        ),
        "",
        f"![American put exercise]({Path(plots['american_put_exercise']).name})",
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


def _summarize_american(result: AmericanLSMCResult) -> dict[str, float]:
    maturity = int(result.exercise_profile["step"].max())
    regression_r2 = result.regression_diagnostics["regression_r2"].dropna()
    reported_value = max(result.price, result.european_value)
    reported_stderr = result.stderr if result.price >= result.european_value else result.european_stderr
    return {
        "price": reported_value,
        "lsmc_policy_value": result.price,
        "stderr": reported_stderr,
        "ci95_low": reported_value - 1.96 * reported_stderr,
        "ci95_high": reported_value + 1.96 * reported_stderr,
        "european_value": result.european_value,
        "european_stderr": result.european_stderr,
        "used_european_baseline": float(result.price < result.european_value),
        "early_exercise_probability": float(np.mean(result.exercise_steps < maturity)),
        "mean_exercise_step": float(np.mean(result.exercise_steps)),
        "q05_path_value": float(np.quantile(result.path_values, 0.05)),
        "q50_path_value": float(np.quantile(result.path_values, 0.50)),
        "q95_path_value": float(np.quantile(result.path_values, 0.95)),
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
    }


def _summarize_swing(result: SwingLSMCResult) -> dict[str, float]:
    regression_r2 = result.regression_diagnostics["regression_r2"].dropna()
    summary = {
        "price": result.price,
        "stderr": result.stderr,
        "ci95_low": result.confidence_interval_95[0],
        "ci95_high": result.confidence_interval_95[1],
        "q05_path_value": float(np.quantile(result.path_values, 0.05)),
        "q50_path_value": float(np.quantile(result.path_values, 0.50)),
        "q95_path_value": float(np.quantile(result.path_values, 0.95)),
        "median_regression_r2": float(np.median(regression_r2)) if len(regression_r2) else float("nan"),
    }
    summary.update(result.volume_summary)
    return summary


def _bottom_line(metrics: dict[str, Any]) -> str:
    model_lines = []
    for model, data in metrics["models"].items():
        call_premium = data["american_call"]["price"] - data["american_call"]["european_value"]
        put_premium = data["american_put"]["price"] - data["american_put"]["european_value"]
        model_lines.append(
            f"- `{model}`: American call early-exercise premium `{_fmt(call_premium)}`, "
            f"American put early-exercise premium `{_fmt(put_premium)}`, "
            f"swing value `{_fmt(data['swing_call']['price'])}` with mean total volume `{_fmt(data['swing_call']['mean_total_volume'])}`."
        )
    caveat = (
        "These numbers are useful for engineering comparison of exercise logic and path sensitivity. "
        "They are not final fair values until drift/risk-neutral assumptions, independent policy evaluation, "
        "transaction costs and richer path validation are locked down."
    )
    return "\n".join(model_lines + ["", caveat])


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
    parser = argparse.ArgumentParser(description="Value American and swing options on GJR-GARCH/HAR-RV MC paths.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output-dir", default="outputs/option_lsmc_report_front")
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--strike-moneyness", type=float, default=1.0)
    parser.add_argument("--n-paths-if-simulated", type=int, default=1000)
    parser.add_argument("--horizon-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20260519)
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
        horizon_steps=args.horizon_steps,
        seed=args.seed,
        use_existing_paths=not args.ignore_existing_paths,
    )
    metrics = run_report(config)
    print(f"Wrote report: {config.output_dir / 'README.md'}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    for model, data in metrics["models"].items():
        print(
            f"{model}: call={_fmt(data['american_call']['price'])}, "
            f"put={_fmt(data['american_put']['price'])}, swing={_fmt(data['swing_call']['price'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
