"""Small frozen-policy interfaces and baseline policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from lsmc_rl.valuation.american import AmericanOptionContract
from lsmc_rl.valuation.swing import SwingOptionContract


@dataclass(frozen=True)
class AmericanPolicyState:
    """Information available to an American exercise policy at one step."""

    step: int
    time: pd.Timestamp | None
    current_price: float
    variance: float | None
    volatility: float | None
    maturity_step: int
    remaining_steps: int
    intrinsic_value: float
    contract: AmericanOptionContract


class AmericanExercisePolicy(Protocol):
    """Frozen American-option exercise rule."""

    name: str

    def decide_exercise(self, state: AmericanPolicyState) -> bool:
        """Return True when the policy exercises at the current step."""


@dataclass(frozen=True)
class SwingPolicyState:
    """Information available to a swing nomination policy at one step."""

    step: int
    time: pd.Timestamp | None
    current_price: float
    variance: float | None
    volatility: float | None
    maturity_step: int
    remaining_steps: int
    remaining_exercise_dates: int
    remaining_volume: float
    exercised_volume: float
    margin: float
    contract: SwingOptionContract


class SwingNominationPolicy(Protocol):
    """Frozen swing-option nomination rule."""

    name: str

    def nominate(self, state: SwingPolicyState) -> float:
        """Return the requested action volume at the current exercise date."""


@dataclass(frozen=True)
class NeverEarlyExercisePolicy:
    """American baseline that only receives terminal payoff."""

    name: str = "never_early_exercise"

    def decide_exercise(self, state: AmericanPolicyState) -> bool:
        return False


@dataclass(frozen=True)
class ValidationSelectedAmericanPolicy:
    """Global validation-selected deployment wrapper.

    This policy delegates either to the candidate or to the fallback for all
    evaluation paths. It is not a raw learned policy and it does not apply a
    pathwise maximum against the European payoff.
    """

    candidate: AmericanExercisePolicy
    use_candidate: bool
    fallback: AmericanExercisePolicy = field(default_factory=NeverEarlyExercisePolicy)
    name: str = "american_lsmc_validation_selected_deployment"

    def decide_exercise(self, state: AmericanPolicyState) -> bool:
        selected = self.candidate if self.use_candidate else self.fallback
        return bool(selected.decide_exercise(state))

    @property
    def selected_policy_name(self) -> str:
        selected = self.candidate if self.use_candidate else self.fallback
        return str(getattr(selected, "name", selected.__class__.__name__))


@dataclass(frozen=True)
class ImmediateIntrinsicExercisePolicy:
    """Exercise as soon as intrinsic value is strictly positive."""

    name: str = "immediate_intrinsic"
    tolerance: float = 1e-12

    def decide_exercise(self, state: AmericanPolicyState) -> bool:
        return bool(state.intrinsic_value > self.tolerance)


@dataclass(frozen=True)
class NeverExerciseSwingPolicy:
    """Swing baseline that nominates zero volume at every exercise date."""

    name: str = "never_exercise"

    def nominate(self, state: SwingPolicyState) -> float:
        return 0.0


@dataclass(frozen=True)
class PositiveMarginSwingPolicy:
    """Nominate max period volume when current spot margin is positive."""

    name: str = "positive_margin"
    tolerance: float = 1e-12

    def nominate(self, state: SwingPolicyState) -> float:
        if state.margin <= self.tolerance or state.remaining_volume <= 0.0:
            return 0.0
        return float(min(state.contract.max_exercise_volume, state.remaining_volume))


@dataclass(frozen=True)
class QuotaAwareSwingPolicy:
    """Positive-margin rule with minimum-total-volume catch-up logic."""

    name: str = "quota_aware_positive_margin"
    tolerance: float = 1e-12

    def nominate(self, state: SwingPolicyState) -> float:
        if state.remaining_volume <= 0.0:
            return 0.0

        contract = state.contract
        forced_volume = self._forced_minimum_volume(state)
        desired = contract.max_exercise_volume if state.margin > self.tolerance else forced_volume
        desired = max(desired, forced_volume)
        if desired <= self.tolerance:
            return 0.0

        if desired < contract.min_exercise_volume:
            desired = contract.min_exercise_volume
        desired = min(desired, contract.max_exercise_volume, state.remaining_volume)
        return float(_ceil_to_volume_step(desired, contract.volume_step))

    def _forced_minimum_volume(self, state: SwingPolicyState) -> float:
        contract = state.contract
        if not contract.enforce_min_total_volume or contract.min_total_volume <= 0.0:
            return 0.0

        still_required = max(contract.min_total_volume - state.exercised_volume, 0.0)
        future_dates_after_current = max(state.remaining_exercise_dates - 1, 0)
        future_capacity = future_dates_after_current * contract.max_exercise_volume
        forced = max(still_required - future_capacity, 0.0)
        return min(forced, contract.max_exercise_volume, state.remaining_volume)


def _ceil_to_volume_step(value: float, volume_step: float) -> float:
    units = int(np.ceil((float(value) - 1e-12) / float(volume_step)))
    return max(units, 0) * float(volume_step)
