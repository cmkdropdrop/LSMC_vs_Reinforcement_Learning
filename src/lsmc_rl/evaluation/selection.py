"""Validation-based policy selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lsmc_rl.evaluation.evaluator import evaluate_american_policy
from lsmc_rl.evaluation.metrics import paired_policy_metrics, path_value_summary
from lsmc_rl.evaluation.policies import (
    AmericanExercisePolicy,
    NeverEarlyExercisePolicy,
    ValidationSelectedAmericanPolicy,
)
from lsmc_rl.valuation.american import AmericanOptionContract


@dataclass(frozen=True)
class AmericanPolicySelectionResult:
    """Frozen policy chosen by independent validation diagnostics."""

    policy: ValidationSelectedAmericanPolicy
    validation_metrics: dict[str, Any]
    selected_policy_name: str
    used_candidate: bool
    decision_rule: str


def select_american_policy_by_validation(
    validation_paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    candidate_policy: AmericanExercisePolicy,
    *,
    variance_paths: np.ndarray | None = None,
    baseline_policy: AmericanExercisePolicy | None = None,
    bootstrap_seed: int = 12345,
    n_bootstrap: int = 2000,
    min_ci_low: float = 0.0,
    min_mean_delta: float | None = None,
    min_cvar_5_delta: float | None = None,
    selected_name: str = "american_lsmc_validation_selected_deployment",
) -> AmericanPolicySelectionResult:
    """Select an American exercise policy using validation paths only.

    The candidate is deployed only when its paired value delta versus the
    baseline passes the configured validation gates. The default gate requires
    the bootstrap lower confidence bound for the mean delta to exceed
    ``min_ci_low``. Optional mean and CVaR gates make the same wrapper usable
    for stricter production-style deployment checks.

    The returned policy delegates globally either to the candidate or to the
    baseline for every evaluation path. This is an ex-ante deployment rule, not
    a raw model value estimate and not a pathwise value floor.
    """

    baseline = baseline_policy or NeverEarlyExercisePolicy()
    candidate_eval = evaluate_american_policy(
        validation_paths,
        contract,
        candidate_policy,
        variance_paths=variance_paths,
    )
    baseline_eval = evaluate_american_policy(
        validation_paths,
        contract,
        baseline,
        variance_paths=variance_paths,
    )
    metrics = paired_policy_metrics(
        candidate_eval.path_values,
        baseline_eval.path_values,
        name_a=candidate_eval.policy_name,
        name_b=baseline_eval.policy_name,
        bootstrap_seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    ci_low, _ = metrics["bootstrap_mean_delta_ci95"]
    ci_gate_passed = bool(ci_low > min_ci_low)
    mean_gate_passed = bool(min_mean_delta is None or metrics["mean_delta"] > min_mean_delta)
    cvar_gate_passed = bool(min_cvar_5_delta is None or metrics["cvar_5_delta"] >= min_cvar_5_delta)
    use_candidate = bool(ci_gate_passed and mean_gate_passed and cvar_gate_passed)
    selected_policy = ValidationSelectedAmericanPolicy(
        candidate=candidate_policy,
        fallback=baseline,
        use_candidate=use_candidate,
        name=selected_name,
    )
    rule_parts = [f"bootstrap CI low > {min_ci_low:.6g}"]
    if min_mean_delta is not None:
        rule_parts.append(f"mean delta > {min_mean_delta:.6g}")
    if min_cvar_5_delta is not None:
        rule_parts.append(f"CVaR5 delta >= {min_cvar_5_delta:.6g}")
    decision_rule = "deploy candidate only if validation " + " and ".join(rule_parts)
    metrics = {
        **metrics,
        "candidate_value": path_value_summary(candidate_eval.path_values),
        "baseline_value": path_value_summary(baseline_eval.path_values),
        "selected_policy_name": selected_policy.selected_policy_name,
        "used_candidate": use_candidate,
        "decision_rule": decision_rule,
        "deployment_gate": {
            "min_ci_low": float(min_ci_low),
            "min_mean_delta": None if min_mean_delta is None else float(min_mean_delta),
            "min_cvar_5_delta": None if min_cvar_5_delta is None else float(min_cvar_5_delta),
            "ci_gate_passed": ci_gate_passed,
            "mean_gate_passed": mean_gate_passed,
            "cvar_gate_passed": cvar_gate_passed,
        },
    }
    return AmericanPolicySelectionResult(
        policy=selected_policy,
        validation_metrics=metrics,
        selected_policy_name=selected_policy.selected_policy_name,
        used_candidate=use_candidate,
        decision_rule=decision_rule,
    )
