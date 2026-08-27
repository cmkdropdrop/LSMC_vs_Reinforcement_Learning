from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


def decode_json_b64(value: str) -> dict[str, Any]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(value + padding)
    return json.loads(raw.decode("utf-8"))


def python_literal(field: str, value: Any) -> str:
    if field in {"hidden", "margin_candidates_pp"}:
        return repr(tuple(value))
    return repr(value)


def patch_config(source: str, overrides: dict[str, Any]) -> str:
    tree = ast.parse(source)
    config = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"),
        None,
    )
    if config is None:
        raise RuntimeError("Config class was not found in the base runner")

    nodes: dict[str, ast.AnnAssign] = {}
    for node in config.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node

    missing = sorted(set(overrides) - set(nodes))
    if missing:
        raise KeyError(f"Unknown Config fields: {missing}")

    lines = source.splitlines(keepends=True)
    for field, value in overrides.items():
        node = nodes[field]
        if node.lineno != node.end_lineno:
            raise RuntimeError(f"Config assignment for {field} spans multiple lines")
        index = node.lineno - 1
        old = lines[index]
        if "=" not in old:
            raise RuntimeError(f"Config assignment for {field} has no equals sign")
        prefix, _ = old.split("=", 1)
        newline = "\n" if old.endswith("\n") else ""
        lines[index] = f"{prefix}= {python_literal(field, value)}{newline}"

    patched = "".join(lines)
    ast.parse(patched)
    return patched


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, tuple):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--config-b64", required=True)
    parser.add_argument("--fold-b64", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    config = decode_json_b64(args.config_b64)
    fold = decode_json_b64(args.fold_b64)
    config_id = str(config["config_id"])
    fold_id = str(fold["fold_id"])

    seed = int(config.get("seed", 20260827))
    margin_candidates = config.get(
        "margin_candidates_pp", [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    )

    overrides: dict[str, Any] = {
        "train_end": fold["train_end"],
        "model_val_end": fold["model_val_end"],
        "margin_val_end": fold["margin_val_end"],
        "dev_test_end": fold["dev_test_end"],
        "oos_start": fold["dev_test_end"],
        "develop_last_month": fold["develop_last_month"],
        "seed": seed,
        "hidden": config["hidden"],
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "initial_fit_epochs": config["initial_fit_epochs"],
        "policy_iterations": config["policy_iterations"],
        "epochs_per_iteration": config["epochs_per_iteration"],
        "max_hold_hours": config["max_hold_hours"],
        "min_hold_hours": config["min_hold_hours"],
        "margin_candidates_pp": margin_candidates,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = args.output_dir / "bundle"
    source = args.base_source.read_text(encoding="utf-8")
    patched = patch_config(source, overrides)
    patched_path = args.output_dir / "patched_run_experiment.py"
    patched_path.write_text(patched, encoding="utf-8")
    source_hash = hashlib.sha256(patched.encode("utf-8")).hexdigest()

    started = time.time()
    command = [
        sys.executable,
        str(patched_path),
        "develop",
        "--output-dir",
        str(bundle_dir),
        "--cache-dir",
        str(args.cache_dir),
    ]
    print(json.dumps({"config_id": config_id, "fold_id": fold_id, "seed": seed, "command": command}), flush=True)
    subprocess.run(command, check=True)
    elapsed = time.time() - started

    manifest = json.loads((bundle_dir / "frozen_manifest.json").read_text(encoding="utf-8"))
    training = pd.read_csv(bundle_dir / "training_iterations.csv")
    margin = pd.read_csv(bundle_dir / "margin_validation.csv")

    best_index = int(training["validation_policy_huber"].astype(float).idxmin())
    best_row = training.loc[best_index]
    selected_margin = float(manifest["selected_action_margin_pp"])
    selected_margin_rows = margin.loc[
        (margin["action_margin_pp"].astype(float) - selected_margin).abs() < 1e-12
    ]
    selected_margin_metrics = (
        selected_margin_rows.iloc[0].to_dict() if not selected_margin_rows.empty else {}
    )

    result = {
        "status": "success",
        "config_id": config_id,
        "fold_id": fold_id,
        "seed": seed,
        "config": config,
        "fold": fold,
        "selected_action_margin_pp": selected_margin,
        "selected_model_iteration": int(best_row["iteration"]),
        "selected_validation_policy_huber": float(best_row["validation_policy_huber"]),
        "selected_margin_validation_metrics": selected_margin_metrics,
        "development_metrics": manifest["development_test_metrics"],
        "patched_source_sha256": source_hash,
        "model_bundle_sha256": manifest["bundle_sha256"],
        "elapsed_seconds": elapsed,
        "data_assertion": manifest["development_data_assertion"],
    }
    result = finite_json(result)
    (args.output_dir / "trial_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
