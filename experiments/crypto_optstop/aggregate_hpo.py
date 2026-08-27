from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FOLDS = {"F1", "F2", "F3", "F4"}
EXTRA_SEEDS = (20260828, 20260829)


def encode_json_b64(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def load_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("trial_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_result_path"] = str(path)
        results.append(row)
    if not results:
        raise RuntimeError(f"No trial_result.json files found below {root}")
    return results


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def fold_utility(metrics: dict[str, Any]) -> float:
    sharpe = float(np.clip(safe_float(metrics.get("sharpe_ratio_hourly_annualized")), -5.0, 5.0))
    total_return = float(np.clip(safe_float(metrics.get("total_return_pct")), -100.0, 100.0))
    max_drawdown = float(np.clip(safe_float(metrics.get("maximum_drawdown_pct")), -100.0, 0.0))
    profit_factor = float(np.clip(safe_float(metrics.get("profit_factor"), 0.05), 0.05, 20.0))
    trades = safe_float(metrics.get("number_of_trades"))
    trade_penalty = 0.10 * max(0.0, 8.0 - trades)
    return sharpe + 0.04 * total_return + 0.02 * max_drawdown + 0.20 * math.log(profit_factor) - trade_penalty


def rows_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result["development_metrics"]
        rows.append({
            "config_id": result["config_id"],
            "fold_id": result["fold_id"],
            "seed": int(result["seed"]),
            "selected_action_margin_pp": result["selected_action_margin_pp"],
            "selected_model_iteration": result["selected_model_iteration"],
            "selected_validation_policy_huber": result["selected_validation_policy_huber"],
            "elapsed_seconds": result["elapsed_seconds"],
            "fold_utility": fold_utility(metrics),
            "_result_path": result["_result_path"],
            **{k: v for k, v in metrics.items()},
        })
    return pd.DataFrame(rows)


def summarize_stage1(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, group in df.groupby("config_id"):
        fold_ids = set(group["fold_id"].astype(str))
        complete = fold_ids == EXPECTED_FOLDS and len(group) == len(EXPECTED_FOLDS)
        utilities = group["fold_utility"].astype(float).to_numpy()
        returns = group["total_return_pct"].astype(float).to_numpy()
        sharpes = group["sharpe_ratio_hourly_annualized"].astype(float).to_numpy()
        mdds = group["maximum_drawdown_pct"].astype(float).to_numpy()
        pfs = group["profit_factor"].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        neg_fraction = float(np.mean(returns < 0))
        robust = (
            0.65 * float(np.median(utilities))
            + 0.35 * float(np.min(utilities))
            - 0.25 * float(np.std(utilities, ddof=0))
            - 0.50 * neg_fraction
        ) if complete else -1e9
        rows.append({
            "config_id": config_id,
            "complete_four_folds": complete,
            "robust_score": robust,
            "median_fold_utility": float(np.median(utilities)),
            "minimum_fold_utility": float(np.min(utilities)),
            "std_fold_utility": float(np.std(utilities, ddof=0)),
            "median_return_pct": float(np.median(returns)),
            "minimum_return_pct": float(np.min(returns)),
            "median_sharpe": float(np.median(sharpes)),
            "minimum_sharpe": float(np.min(sharpes)),
            "median_max_drawdown_pct": float(np.median(mdds)),
            "median_profit_factor": float(np.nanmedian(pfs)),
            "negative_return_fold_fraction": neg_fraction,
            "minimum_trades": float(group["number_of_trades"].min()),
            "mean_elapsed_seconds": float(group["elapsed_seconds"].mean()),
        })
    return pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)


def plot_stage1(summary: pd.DataFrame, folds: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = summary.sort_values("robust_score", ascending=True)
    ax.barh(ordered["config_id"], ordered["robust_score"])
    ax.set_xlabel("Robust walk-forward HPO score")
    ax.set_ylabel("Configuration")
    ax.set_title("Pre-OOS hyperparameter optimization")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "hpo_robust_scores.png", dpi=160)
    plt.close(fig)

    pivot = folds.pivot(index="fold_id", columns="config_id", values="total_return_pct")
    fig, ax = plt.subplots(figsize=(12, 7))
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column], marker="o", label=column)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Six-month fold return (%)")
    ax.set_xlabel("Walk-forward fold")
    ax.set_title("Return stability across pre-OOS folds")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "hpo_fold_returns.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = df[columns].copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def stage1(args: argparse.Namespace) -> None:
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.trials_dir)
    df = rows_dataframe(results)
    summary = summarize_stage1(df)
    df.to_csv(output / "hpo_fold_results.csv", index=False)
    summary.to_csv(output / "hpo_stage1_summary.csv", index=False)
    plot_stage1(summary, df, output)

    lookup = {(r["config_id"], r["fold_id"]): r for r in results}
    top3 = summary.head(3)["config_id"].tolist()
    matrix: list[dict[str, Any]] = []
    for config_id in top3:
        base = lookup[(config_id, "F4")]
        for seed in EXTRA_SEEDS:
            config = dict(base["config"])
            config["seed"] = seed
            matrix.append({
                "config_id": config_id,
                "seed": seed,
                "config_b64": encode_json_b64(config),
                "fold_b64": encode_json_b64(base["fold"]),
            })
    (output / "confirmation_matrix.json").write_text(json.dumps(matrix, separators=(",", ":")), encoding="utf-8")

    report = [
        "# HPO stage 1: pre-OOS walk-forward search",
        "",
        "The consumed 2026-02-01 to 2026-08-01 OOS window was not loaded. Eight fixed configurations were evaluated on four anchored six-month development folds.",
        "",
        markdown_table(summary, [
            "config_id", "robust_score", "median_return_pct", "minimum_return_pct",
            "median_sharpe", "minimum_sharpe", "median_max_drawdown_pct",
            "median_profit_factor", "negative_return_fold_fraction", "minimum_trades"
        ]),
        "",
        "The top three configurations proceed to fixed-seed confirmation on the latest pre-OOS fold.",
    ]
    (output / "HPO_STAGE1_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print((output / "confirmation_matrix.json").read_text(), flush=True)


def final(args: argparse.Namespace) -> None:
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stage1_results = load_results(args.trials_dir)
    confirm_results = load_results(args.confirm_dir)
    fold_df = rows_dataframe(stage1_results)
    stage1_summary = summarize_stage1(fold_df)
    confirm_df = rows_dataframe(confirm_results)

    top3 = stage1_summary.head(3)["config_id"].tolist()
    confirmation_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for config_id in top3:
        original = fold_df[(fold_df["config_id"] == config_id) & (fold_df["fold_id"] == "F4")]
        extra = confirm_df[confirm_df["config_id"] == config_id]
        seeds = pd.concat([original, extra], ignore_index=True)
        seed_std = float(seeds["fold_utility"].std(ddof=0))
        seed_return_std = float(seeds["total_return_pct"].std(ddof=0))
        seed_sharpe_std = float(seeds["sharpe_ratio_hourly_annualized"].std(ddof=0))
        base_score = float(stage1_summary.loc[stage1_summary["config_id"] == config_id, "robust_score"].iloc[0])
        final_score = base_score - 0.30 * seed_std
        final_rows.append({
            "config_id": config_id,
            "stage1_robust_score": base_score,
            "seed_fold_utility_std": seed_std,
            "seed_return_std_pct": seed_return_std,
            "seed_sharpe_std": seed_sharpe_std,
            "seed_median_return_pct": float(seeds["total_return_pct"].median()),
            "seed_minimum_return_pct": float(seeds["total_return_pct"].min()),
            "seed_median_sharpe": float(seeds["sharpe_ratio_hourly_annualized"].median()),
            "final_score": final_score,
        })
        confirmation_rows.extend(seeds.to_dict("records"))

    final_summary = pd.DataFrame(final_rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    seed_df = pd.DataFrame(confirmation_rows)
    selected_id = str(final_summary.iloc[0]["config_id"])
    selected_result = next(r for r in stage1_results if r["config_id"] == selected_id and r["fold_id"] == "F4")
    selected_result_path = Path(selected_result["_result_path"])
    selected_bundle = selected_result_path.parent / "bundle"
    if not selected_bundle.exists():
        raise RuntimeError(f"Selected bundle not found: {selected_bundle}")
    shutil.copytree(selected_bundle, output / "selected_model_bundle")

    selected_fold_rows = fold_df[fold_df["config_id"] == selected_id].sort_values("fold_id")
    selected_seed_rows = seed_df[seed_df["config_id"] == selected_id].sort_values("seed")
    selected_margins = pd.concat([
        selected_fold_rows["selected_action_margin_pp"],
        selected_seed_rows["selected_action_margin_pp"],
    ]).astype(float)
    recommended_margin = float(selected_margins.median())

    selected_config = dict(selected_result["config"])
    selection = {
        "selected_config_id": selected_id,
        "selected_config": selected_config,
        "recommended_action_margin_pp_median_across_folds_and_seed_checks": recommended_margin,
        "frozen_bundle_action_margin_pp": selected_result["selected_action_margin_pp"],
        "selection_method": "Four-fold pre-OOS robust score, then fixed-seed instability penalty on F4",
        "oos_reused": False,
        "warning": "The original six-month OOS window was already consumed before this HPO and was not rerun. A new future holdout is required for an unbiased final estimate."
    }
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    fold_df.to_csv(output / "hpo_all_fold_results.csv", index=False)
    stage1_summary.to_csv(output / "hpo_stage1_summary.csv", index=False)
    seed_df.to_csv(output / "hpo_seed_confirmation.csv", index=False)
    final_summary.to_csv(output / "hpo_final_ranking.csv", index=False)
    plot_stage1(stage1_summary, fold_df, output)

    report = [
        "# Deep-Q optimal-stopping HPO report",
        "",
        "## Data isolation",
        "",
        "The HPO used only bars earlier than 2026-02-01 00:00 UTC. The previously consumed 2026-02-01 to 2026-08-01 OOS window was neither downloaded nor evaluated by this workflow.",
        "",
        "## Final ranking",
        "",
        markdown_table(final_summary, [
            "config_id", "stage1_robust_score", "seed_fold_utility_std",
            "seed_return_std_pct", "seed_sharpe_std", "seed_median_return_pct",
            "seed_minimum_return_pct", "seed_median_sharpe", "final_score"
        ]),
        "",
        f"## Selected configuration: {selected_id}",
        "",
        "```json",
        json.dumps(selection, indent=2, sort_keys=True),
        "```",
        "",
        "### Selected configuration across walk-forward folds",
        "",
        markdown_table(selected_fold_rows, [
            "fold_id", "total_return_pct", "sharpe_ratio_hourly_annualized",
            "maximum_drawdown_pct", "profit_factor", "number_of_trades",
            "selected_action_margin_pp", "selected_model_iteration", "fold_utility"
        ]),
        "",
        "### Fixed-seed confirmation on F4",
        "",
        markdown_table(selected_seed_rows, [
            "seed", "total_return_pct", "sharpe_ratio_hourly_annualized",
            "maximum_drawdown_pct", "profit_factor", "number_of_trades",
            "selected_action_margin_pp", "selected_model_iteration", "fold_utility"
        ]),
        "",
        "## Interpretation",
        "",
        "This is hyperparameter selection evidence, not a new out-of-sample performance estimate. Because the earlier six-month OOS result was already observed, it cannot become untouched again. The selected configuration must be assessed on a newly accruing future holdout before any production claim."
    ]
    (output / "HPO_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(final_summary.to_string(index=False), flush=True)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("stage1")
    p1.add_argument("--trials-dir", type=Path, required=True)
    p1.add_argument("--output-dir", type=Path, required=True)

    p2 = sub.add_parser("final")
    p2.add_argument("--trials-dir", type=Path, required=True)
    p2.add_argument("--confirm-dir", type=Path, required=True)
    p2.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "stage1":
        stage1(args)
    else:
        final(args)


if __name__ == "__main__":
    main()
