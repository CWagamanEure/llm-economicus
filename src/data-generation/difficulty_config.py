"""
Shared difficulty configuration for dataset generators.

This module keeps static defaults separate from generator logic so they can be
reused and calibrated later from empirical results.
"""

from typing import Final, get_args

from schema import BayesianSignalTaskSubtype, RiskLossTimeTaskSubtype

DIFFICULTY_LEVEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "easy": "Low arithmetic and comparison burden; clear utility separation.",
    "medium": "Moderate arithmetic/comparison burden or closer utility tradeoffs.",
    "hard": "High arithmetic burden, tighter utility gaps, or additional constraints.",
}


RISK_LOSS_TIME_SUBTYPES: Final[tuple[str, ...]] = get_args(RiskLossTimeTaskSubtype)

RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE: Final[
    dict[RiskLossTimeTaskSubtype, str]
] = {
    "lottery_choice": "easy",
    "ce_offer_comparison": "medium",
    "mixed_gain_loss_choice": "medium",
    "time_discounting": "medium",
}

BAYESIAN_SIGNAL_SUBTYPES: Final[tuple[str, ...]] = get_args(BayesianSignalTaskSubtype)

BAYESIAN_SIGNAL_DEFAULT_DIFFICULTY_BY_SUBTYPE: Final[
    dict[BayesianSignalTaskSubtype, str]
] = {
    "basic_bayes_update": "easy",
    "binary_signal_decision": "medium",
    "information_cascade_step": "hard",
    "noisy_signal_asset_update": "hard",
}
