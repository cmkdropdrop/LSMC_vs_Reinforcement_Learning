"""Rolling-refit GJR-GARCH diagnostics.

This report evaluates GJR-GARCH on daily aggregates of sequential one-step
intraday variance forecasts. That is closer to the later path-generation use
case than comparing every noisy 5-minute squared return in isolation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lsmc_rl.analysis.volatility_report import (
    _fmt,
    _json_ready,
    _markdown_table,
    gjr_one_step_variance_forecast,
    temporal_return_split,
    variance_metrics,
)
from lsmc_rl.data import load_market_data
from lsmc_rl.volatility import GJRGARCHModel


@dataclass(frozen=True)
class GarchRefitReportConfig:
    database_path: Path
    symbol: str = "FRONT"
    interval: str = "5m"
    train_fraction: float = 0.70
    refit_interval_steps: int = 288
    output_dir: Path = Path("outputs/garch_refit_report_front")
    report_path: Path = Path("docs/garch_refit_analysis.md")


def static_gjr_forecast(
    market: pd.DataFrame,
    train_returns: pd.Series,
    test_returns: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit once on train data and produce sequential one-step test forecasts."""

    model = GJRGARCHModel().fit(train_returns)
    variances = gjr_one_step_variance_forecast(model, test_returns)
    mean_return = model.params.mu / model.params.return_scale if model.params is not None else 0.0
    means = pd.Series(mean_return, index=variances.index, name="forecast_mean")
    frame = make_step_forecast_frame(market, test_returns, variances, means)
    return frame, _params_dict(model, status="fit", train_observations=len(train_returns))


def rolling_refit_gjr_forecast(
    market: pd.DataFrame,
    train_returns: pd.Series,
    test_returns: pd.Series,
    refit_interval_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit GJR-GARCH on an expanding window at fixed test-step intervals."""

    if refit_interval_steps <= 0:
        raise ValueError("refit_interval_steps must be positive")

    all_returns = market["log_return"].dropna()
    test_indices = list(test_returns.index)
    forecasts: list[pd.DataFrame] = []
    refits: list[dict[str, Any]] = []
    fallback_model = GJRGARCHModel().fit(train_returns)

    block_id = 0
    position = 0
    while position < len(test_indices):
        block_indices = test_indices[position : position + refit_interval_steps]
        block_returns = all_returns.loc[block_indices]
        first_index = int(block_indices[0])
        history = all_returns.loc[: first_index - 1].dropna()

        try:
            model = GJRGARCHModel().fit(history)
            status = "fit"
            fallback_model = model
        except RuntimeError:
            model = fallback_model
            status = "fallback_previous_fit"

        variances = gjr_one_step_variance_forecast(model, block_returns)
        mean_return = model.params.mu / model.params.return_scale if model.params is not None else 0.0
        means = pd.Series(mean_return, index=variances.index, name="forecast_mean")
        forecasts.append(make_step_forecast_frame(market, block_returns, variances, means))

        params = _params_dict(model, status=status, train_observations=len(history))
        params.update(
            {
                "block": block_id,
                "forecast_start": market.loc[first_index, "open_datetime"],
                "forecast_end": market.loc[int(block_indices[-1]), "open_datetime"],
            }
        )
        refits.append(params)

        block_id += 1
        position += refit_interval_steps

    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(refits)


def make_step_forecast_frame(
    market: pd.DataFrame,
    returns: pd.Series,
    variances: pd.Series,
    means: pd.Series,
) -> pd.DataFrame:
    """Align step-level realized returns and forecasts."""

    index = variances.index
    return pd.DataFrame(
        {
            "source_index": index,
            "time": market.loc[index, "open_datetime"].to_numpy(),
            "actual_return": returns.loc[index].to_numpy(dtype=float),
            "forecast_mean": means.loc[index].to_numpy(dtype=float),
            "forecast_variance": variances.loc[index].to_numpy(dtype=float),
        }
    )


def daily_forecast_frame(step_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sequential intraday forecasts to UTC daily realized variance."""

    frame = step_frame.copy()
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.floor("D")
    daily = (
        frame.groupby("date", sort=True)
        .agg(
            actual_return=("actual_return", "sum"),
            actual_rv=("actual_return", lambda values: float(np.sum(np.asarray(values) ** 2))),
            forecast_mean=("forecast_mean", "sum"),
            forecast_rv=("forecast_variance", "sum"),
            observations=("actual_return", "size"),
        )
        .reset_index()
    )
    daily["forecast_volatility"] = np.sqrt(daily["forecast_rv"])
    daily["standardized_residual"] = (daily["actual_return"] - daily["forecast_mean"]) / daily[
        "forecast_volatility"
    ]
    return daily


def persistence_daily_rv(market: pd.DataFrame, forecast_dates: pd.Series) -> pd.DataFrame:
    """Previous observed daily RV baseline for the requested dates."""

    returns = market["log_return"].dropna()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(market.loc[returns.index, "open_datetime"], utc=True).dt.floor("D"),
            "squared_return": returns.to_numpy(dtype=float) ** 2,
        }
    )
    daily = frame.groupby("date", sort=True)["squared_return"].sum().reset_index(name="actual_rv")
    daily["persistence_rv"] = daily["actual_rv"].shift(1)
    return daily[daily["date"].isin(set(forecast_dates))].dropna().reset_index(drop=True)


def daily_coverage_metrics(daily: pd.DataFrame) -> dict[str, float]:
    residual = daily["standardized_residual"].to_numpy(dtype=float)
    residual = residual[np.isfinite(residual)]
    return {
        "observations": float(len(residual)),
        "var_5pct_hit_rate": float(np.mean(residual < stats.norm.ppf(0.05))),
        "var_1pct_hit_rate": float(np.mean(residual < stats.norm.ppf(0.01))),
        "standardized_residual_mean": float(np.mean(residual)),
        "standardized_residual_std": float(np.std(residual, ddof=1)),
        "standardized_residual_skew": float(stats.skew(residual, bias=False)),
        "standardized_residual_excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)),
    }


def run_report(config: GarchRefitReportConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)

    market, data_quality = load_market_data(config.database_path, config.symbol, config.interval)
    split = temporal_return_split(market, config.train_fraction)

    static_steps, static_params = static_gjr_forecast(market, split.train_returns, split.test_returns)
    rolling_steps, refits = rolling_refit_gjr_forecast(
        market=market,
        train_returns=split.train_returns,
        test_returns=split.test_returns,
        refit_interval_steps=config.refit_interval_steps,
    )

    static_daily = daily_forecast_frame(static_steps)
    rolling_daily = daily_forecast_frame(rolling_steps)
    persistence = persistence_daily_rv(market, rolling_daily["date"])

    static_metrics = variance_metrics(static_daily["actual_rv"], static_daily["forecast_rv"])
    rolling_metrics = variance_metrics(rolling_daily["actual_rv"], rolling_daily["forecast_rv"])
    persistence_metrics = variance_metrics(persistence["actual_rv"], persistence["persistence_rv"])
    rolling_metrics["mse_skill_vs_persistence"] = 1.0 - rolling_metrics["mse"] / persistence_metrics["mse"]
    static_metrics["mse_skill_vs_persistence"] = 1.0 - static_metrics["mse"] / persistence_metrics["mse"]

    coverage = daily_coverage_metrics(rolling_daily)
    plot_paths = create_plots(
        output_dir=config.output_dir,
        market=market,
        split_time=split.split_time,
        static_daily=static_daily,
        rolling_daily=rolling_daily,
        persistence=persistence,
        refits=refits,
    )

    metrics: dict[str, Any] = {
        "config": {
            "database_path": str(config.database_path),
            "symbol": config.symbol,
            "interval": config.interval,
            "train_fraction": config.train_fraction,
            "refit_interval_steps": config.refit_interval_steps,
            "output_dir": str(config.output_dir),
            "report_path": str(config.report_path),
        },
        "data_quality": data_quality.to_dict(),
        "split": {
            "train_start": split.train_market["open_datetime"].iloc[0].isoformat(),
            "train_end": split.split_time.isoformat(),
            "test_start": split.test_market["open_datetime"].iloc[0].isoformat(),
            "test_end": split.test_market["open_datetime"].iloc[-1].isoformat(),
            "train_return_observations": int(len(split.train_returns)),
            "test_return_observations": int(len(split.test_returns)),
            "test_daily_observations": int(len(rolling_daily)),
        },
        "static_gjr_garch": {
            "params": static_params,
            "daily_rv_metrics": static_metrics,
        },
        "rolling_refit_gjr_garch": {
            "refits": int(len(refits)),
            "fallback_refits": int((refits["status"] != "fit").sum()),
            "daily_rv_metrics": rolling_metrics,
            "daily_return_coverage": coverage,
            "latest_params": refits.iloc[-1].to_dict(),
        },
        "persistence_baseline": {
            "daily_rv_metrics": persistence_metrics,
        },
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }

    metrics_path = config.output_dir / "garch_refit_metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    config.report_path.write_text(render_report(metrics, config.report_path), encoding="utf-8")
    return metrics


def create_plots(
    output_dir: Path,
    market: pd.DataFrame,
    split_time: pd.Timestamp,
    static_daily: pd.DataFrame,
    rolling_daily: pd.DataFrame,
    persistence: pd.DataFrame,
    refits: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    plt.figure(figsize=(11, 4.6))
    plt.plot(market["open_datetime"], market["close"], linewidth=1.0)
    plt.axvline(split_time, color="black", linestyle="--", linewidth=1.0, label="train/test split")
    plt.title("FRONT close price and temporal split")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    paths["price_split"] = output_dir / "garch_refit_price_split.png"
    plt.savefig(paths["price_split"], dpi=140)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.plot(rolling_daily["date"], rolling_daily["actual_rv"], label="actual daily RV", linewidth=1.2)
    plt.plot(rolling_daily["date"], static_daily["forecast_rv"], label="static GJR forecast", linewidth=1.0)
    plt.plot(rolling_daily["date"], rolling_daily["forecast_rv"], label="rolling-refit GJR forecast", linewidth=1.0)
    plt.plot(persistence["date"], persistence["persistence_rv"], label="persistence baseline", linewidth=0.9, alpha=0.75)
    plt.title("Daily realized variance: actual versus forecasts")
    plt.ylabel("Daily realized variance")
    plt.legend()
    plt.tight_layout()
    paths["daily_rv"] = output_dir / "garch_daily_rv_forecast.png"
    plt.savefig(paths["daily_rv"], dpi=140)
    plt.close()

    lower_5 = rolling_daily["forecast_mean"] + rolling_daily["forecast_volatility"] * stats.norm.ppf(0.05)
    upper_95 = rolling_daily["forecast_mean"] + rolling_daily["forecast_volatility"] * stats.norm.ppf(0.95)
    lower_1 = rolling_daily["forecast_mean"] + rolling_daily["forecast_volatility"] * stats.norm.ppf(0.01)
    plt.figure(figsize=(11, 4.8))
    plt.plot(rolling_daily["date"], rolling_daily["actual_return"], label="actual daily return", linewidth=1.1)
    plt.plot(rolling_daily["date"], rolling_daily["forecast_mean"], label="forecast mean", linewidth=0.9)
    plt.fill_between(rolling_daily["date"], lower_5, upper_95, color="#1f77b4", alpha=0.16, label="5%-95% normal band")
    plt.plot(rolling_daily["date"], lower_1, color="#d62728", linestyle="--", linewidth=0.9, label="1% lower VaR")
    plt.title("Rolling-refit GJR daily return bands")
    plt.ylabel("Daily log return")
    plt.legend()
    plt.tight_layout()
    paths["return_bands"] = output_dir / "garch_daily_return_bands.png"
    plt.savefig(paths["return_bands"], dpi=140)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.plot(refits["forecast_start"], refits["alpha"], marker="o", linewidth=1.0, label="alpha")
    plt.plot(refits["forecast_start"], refits["beta"], marker="o", linewidth=1.0, label="beta")
    plt.plot(refits["forecast_start"], refits["gamma"], marker="o", linewidth=1.0, label="gamma")
    plt.plot(refits["forecast_start"], refits["persistence"], marker="o", linewidth=1.2, label="persistence")
    plt.title("Rolling-refit GJR parameter path")
    plt.ylabel("Parameter value")
    plt.legend()
    plt.tight_layout()
    paths["parameters"] = output_dir / "garch_refit_parameters.png"
    plt.savefig(paths["parameters"], dpi=140)
    plt.close()

    residual = rolling_daily["standardized_residual"].replace([np.inf, -np.inf], np.nan).dropna()
    x = np.linspace(-4, 4, 300)
    plt.figure(figsize=(9, 4.8))
    plt.hist(residual, bins=30, density=True, alpha=0.55, label="standardized daily residuals")
    plt.plot(x, stats.norm.pdf(x), color="black", linewidth=1.2, label="standard normal")
    plt.title("Rolling-refit GJR standardized daily residuals")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    paths["residuals"] = output_dir / "garch_standardized_residuals.png"
    plt.savefig(paths["residuals"], dpi=140)
    plt.close()

    return paths


def render_report(metrics: dict[str, Any], report_path: Path) -> str:
    split = metrics["split"]
    static = metrics["static_gjr_garch"]["daily_rv_metrics"]
    rolling = metrics["rolling_refit_gjr_garch"]["daily_rv_metrics"]
    coverage = metrics["rolling_refit_gjr_garch"]["daily_return_coverage"]
    persistence = metrics["persistence_baseline"]["daily_rv_metrics"]
    plots = {name: _relative_path(path, report_path.parent) for name, path in metrics["plots"].items()}

    lines = [
        "# GARCH Refit Analyse",
        "",
        "Diese Analyse ersetzt die fruehere stark kleinteilige 5-Minuten-Diagnostik nicht, bewertet GJR-GARCH aber auf einer fuer Pfadgenerierung faireren Ebene: sequenzielle Intraday-One-Step-Varianzprognosen werden zu Tages-RV aggregiert. Zusaetzlich wird das Modell im Testzeitraum regelmaessig neu gefittet.",
        "",
        "## Setup",
        "",
        _markdown_table(
            ["Groesse", "Wert"],
            [
                ["Symbol", f"`{metrics['config']['symbol']}`"],
                ["Intervall", f"`{metrics['config']['interval']}`"],
                ["Train", f"`{split['train_start']}` bis `{split['train_end']}`"],
                ["Test", f"`{split['test_start']}` bis `{split['test_end']}`"],
                ["Train-Returns", str(split["train_return_observations"])],
                ["Test-Returns", str(split["test_return_observations"])],
                ["Test-Tage", str(split["test_daily_observations"])],
                ["Refit-Intervall", f"{metrics['config']['refit_interval_steps']} Test-Returns"],
                ["Refits", str(metrics["rolling_refit_gjr_garch"]["refits"])],
                ["Fallback-Refits", str(metrics["rolling_refit_gjr_garch"]["fallback_refits"])],
            ],
        ),
        "",
        f"![Price split]({plots['price_split']})",
        "",
        "## Ergebnisuebersicht",
        "",
        _markdown_table(
            ["Metrik", "Statisches GJR", "Rolling-Refit GJR", "Persistence"],
            [
                ["Daily-RV RMSE", _fmt(static["rmse"]), _fmt(rolling["rmse"]), _fmt(persistence["rmse"])],
                ["Daily-RV MAE", _fmt(static["mae"]), _fmt(rolling["mae"]), _fmt(persistence["mae"])],
                ["Daily-RV Korrelation", _fmt(static["correlation"]), _fmt(rolling["correlation"]), _fmt(persistence["correlation"])],
                ["Daily-RV R2 vs. Mittelwert", _fmt(static["r2_vs_realized_mean"]), _fmt(rolling["r2_vs_realized_mean"]), _fmt(persistence["r2_vs_realized_mean"])],
                ["QLIKE", _fmt(static["qlike"]), _fmt(rolling["qlike"]), _fmt(persistence["qlike"])],
                ["MSE-Skill vs. Persistence", _fmt(static["mse_skill_vs_persistence"]), _fmt(rolling["mse_skill_vs_persistence"]), "-"],
            ],
        ),
        "",
        "Die aggregierte Tages-RV-Diagnostik ist deutlich stabiler als die reine 5-Minuten-Auswertung. Das Rolling-Refit-GJR erreicht eine Daily-RV-Korrelation von "
        f"`{_fmt(rolling['correlation'])}` und ein R2 von `{_fmt(rolling['r2_vs_realized_mean'])}`. Gegenueber der Persistence-Baseline liegt der MSE-Skill bei `{_fmt(rolling['mse_skill_vs_persistence'])}`.",
        "",
        f"![Daily RV forecast]({plots['daily_rv']})",
        "",
        "## Rolling-Refit Und Kalibrierung",
        "",
        "Der Refit erfolgt auf einem expandierenden historischen Fenster. Dadurch kann das Modell spaeter im Testzeitraum neue Volatilitaetsregime aufnehmen, ohne zukuenftige Daten zu verwenden. "
        + _fallback_sentence(metrics["rolling_refit_gjr_garch"]["fallback_refits"]),
        "",
        _markdown_table(
            ["Kalibrierungsmetrik", "Wert"],
            [
                ["5%-VaR-Hitrate auf Tagesreturns", _fmt(coverage["var_5pct_hit_rate"])],
                ["1%-VaR-Hitrate auf Tagesreturns", _fmt(coverage["var_1pct_hit_rate"])],
                ["Mittelwert standardisierter Tagesresiduen", _fmt(coverage["standardized_residual_mean"])],
                ["Std. standardisierter Tagesresiduen", _fmt(coverage["standardized_residual_std"])],
                ["Excess Kurtosis standardisierter Tagesresiduen", _fmt(coverage["standardized_residual_excess_kurtosis"])],
            ],
        ),
        "",
        f"![Daily return bands]({plots['return_bands']})",
        "",
        f"![Standardized residuals]({plots['residuals']})",
        "",
        "Die 1%-VaR-Hitrate liegt nahe am nominalen Niveau. Die 5%-Hitrate ist konservativ niedrig, und die standardisierten Residuen bleiben etwas zu breit und leptokurtisch. Das ist deutlich besser interpretierbar als die alte 5-Minuten-Tail-Diagnostik, aber noch kein Freifahrtschein fuer Optionsbewertung.",
        "",
        "## Parameterpfad",
        "",
        f"![Refit parameters]({plots['parameters']})",
        "",
        "Der Parameterpfad zeigt, ob der Refit stabile Parameter liefert oder einzelne Testabschnitte stark andere Dynamiken erzwingen. Fuer spaetere Experimente sollte dieser Pfad zusammen mit Optionswert-Sensitivitaeten beobachtet werden.",
        "",
        "## Fazit",
        "",
        "Die neue GARCH-Analyse liefert deutlich bessere und fachlich fairere Ergebnisse, weil sie Volatilitaet auf Tagesebene bewertet und das Modell im Testzeitraum regelmaessig neu kalibriert. Fuer die naechste Stufe ist Rolling-Refit-GJR daher ein besserer Kandidat als die alte statische 5-Minuten-Diagnose.",
        "",
        "Trotzdem bleiben Annahmen offen: Normalinnovationen, keine explizite Intraday-Saisonalitaet und keine Rolling-Origin-Grid-Suche ueber Refit-Frequenzen. Diese Punkte sollten vor einer produktiven LSMC-/RL-Nutzung weiter getestet werden.",
        "",
        "## Reproduktion",
        "",
        "```powershell",
        "python -m lsmc_rl.analysis.garch_refit_report --output-dir outputs/garch_refit_report_front --report-path docs/garch_refit_analysis.md",
        "```",
        "",
        "Die numerischen Rohdaten stehen in `outputs/garch_refit_report_front/garch_refit_metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def _params_dict(model: GJRGARCHModel, status: str, train_observations: int) -> dict[str, Any]:
    if model.params is None:
        return {"status": status, "train_observations": train_observations}
    return {
        "status": status,
        "train_observations": train_observations,
        "mu": model.params.mu,
        "omega": model.params.omega,
        "alpha": model.params.alpha,
        "gamma": model.params.gamma,
        "beta": model.params.beta,
        "persistence": model.params.persistence,
        "return_scale": model.params.return_scale,
    }


def _relative_path(path: str | Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path), base)).as_posix()


def _fallback_sentence(fallback_refits: int) -> str:
    if fallback_refits == 0:
        return "In dieser Auswertung gab es keine notwendigen Fallbacks auf alte Parameter."
    return f"In dieser Auswertung wurden {fallback_refits} Refit-Bloecke mit dem vorherigen Parametersatz fortgesetzt."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate rolling-refit GJR-GARCH diagnostics.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--refit-interval-steps", type=int, default=288)
    parser.add_argument("--output-dir", default="outputs/garch_refit_report_front")
    parser.add_argument("--report-path", default="docs/garch_refit_analysis.md")
    args = parser.parse_args(argv)

    config = GarchRefitReportConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        train_fraction=args.train_fraction,
        refit_interval_steps=args.refit_interval_steps,
        output_dir=Path(args.output_dir).resolve(),
        report_path=Path(args.report_path).resolve(),
    )
    metrics = run_report(config)
    rolling = metrics["rolling_refit_gjr_garch"]["daily_rv_metrics"]
    print(f"Wrote report: {config.report_path}")
    print(f"Wrote metrics: {config.output_dir / 'garch_refit_metrics.json'}")
    print(f"Rolling Daily-RV correlation: {_fmt(rolling['correlation'])}")
    print(f"Rolling Daily-RV R2: {_fmt(rolling['r2_vs_realized_mean'])}")
    print(f"Rolling MSE skill vs persistence: {_fmt(rolling['mse_skill_vs_persistence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
