"""
Belief-bias dataset generator for normative judgment tasks.

These samples target classic behavioral belief distortions while preserving a
fully deterministic normative answer and solver-verifiable target.
The `base_rate_neglect` subtype intentionally uses standard Bayes updating
mathematics with behaviorally salient framing; differences vs
`BayesianSignalGenerator` are framing-oriented, not solver-family changes.
"""

import hashlib
import json
import logging
import random
import re
from collections import Counter
from math import ceil, comb, erf, floor, sqrt
from typing import Any, Callable, Literal

from base_generator import BaseGenerator
from difficulty_config import BELIEF_BIAS_DEFAULT_DIFFICULTY_BY_SUBTYPE

from schema import (
    ActionScalars,
    BaseRateNeglectProblemSpec,
    BayesianAssumptions,
    BeliefBiasTaskSubtype,
    ComparisonPair,
    ConjunctionFallacyAssumptions,
    ConjunctionFallacyProblemSpec,
    DataPoint,
    GamblerFallacyAssumptions,
    GamblerFallacyProblemSpec,
    Metadata,
    OverprecisionCalibrationAssumptions,
    OverprecisionCalibrationProblemSpec,
    ProblemSpec,
    SampleSizeNeglectAssumptions,
    SampleSizeNeglectProblemSpec,
    SolverTrace,
    Target,
)

TaskSubtype = BeliefBiasTaskSubtype
DifficultyMetrics = dict[str, float | int | bool | str]
ActionValueSemantics = Literal[
    "posterior_probability_comparison",
    "probability_comparison",
    "claim_correctness",
    "binomial_tail_probability_comparison",
    "interval_coverage_comparison",
]
PosteriorBeliefs = dict[str, float]
PromptStyle = Literal[
    "default",
    "formal",
    "plain_english",
    "compact",
    "finance_framed",
    "unlabeled",
    "random",
]
PromptStyleRegime = Literal[
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
]
ConfiguredPromptStyleRegime = Literal[
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
    "random",
]
PromptFrameVariant = Literal[
    "auto",
    "medical_screening",
    "fraud_detection",
    "hiring_screen",
    "security_alert",
    "trading_signal",
    "neutral_statistical",
    "sports_streak",
    "market_streak",
    "roulette_streak",
    "neutral_coin",
    "vivid_description",
    "plain_probability",
    "representative_profile",
    "hospital_births",
    "fund_returns",
    "quality_control_batches",
    "analyst_forecast",
    "weather_forecast",
    "startup_projection",
]
SUPPORTED_PROMPT_STYLES: tuple[PromptStyle, ...] = (
    "default",
    "formal",
    "plain_english",
    "compact",
    "finance_framed",
    "unlabeled",
    "random",
)
NON_RANDOM_PROMPT_STYLES: tuple[PromptStyle, ...] = (
    "default",
    "formal",
    "plain_english",
    "compact",
    "finance_framed",
    "unlabeled",
)
SUPPORTED_PROMPT_STYLE_REGIMES: tuple[ConfiguredPromptStyleRegime, ...] = (
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
    "random",
)
NON_RANDOM_PROMPT_STYLE_REGIMES: tuple[PromptStyleRegime, ...] = (
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
)
SUPPORTED_PROMPT_FRAME_VARIANTS: tuple[PromptFrameVariant, ...] = (
    "auto",
    "medical_screening",
    "fraud_detection",
    "hiring_screen",
    "security_alert",
    "trading_signal",
    "neutral_statistical",
    "sports_streak",
    "market_streak",
    "roulette_streak",
    "neutral_coin",
    "vivid_description",
    "plain_probability",
    "representative_profile",
    "hospital_births",
    "fund_returns",
    "quality_control_batches",
    "analyst_forecast",
    "weather_forecast",
    "startup_projection",
)
COMPARISON_PAIR_BY_SUBTYPE: dict[TaskSubtype, ComparisonPair] = {
    "base_rate_neglect": {
        "left_action": "choose_state_high",
        "right_action": "choose_state_low",
    },
    "conjunction_fallacy": {
        "left_action": "choose_A",
        "right_action": "choose_B",
    },
    "gambler_fallacy": {
        "left_action": "choose_A",
        "right_action": "choose_B",
    },
    "sample_size_neglect": {
        "left_action": "choose_A",
        "right_action": "choose_B",
    },
    "overprecision_calibration": {
        "left_action": "choose_A",
        "right_action": "choose_B",
    },
}
PROMPT_RENDERER_METHOD_BY_SUBTYPE: dict[TaskSubtype, str] = {
    "base_rate_neglect": "_render_base_rate_neglect_prompt",
    "conjunction_fallacy": "_render_conjunction_fallacy_prompt",
    "gambler_fallacy": "_render_gambler_fallacy_prompt",
    "sample_size_neglect": "_render_sample_size_neglect_prompt",
    "overprecision_calibration": "_render_overprecision_calibration_prompt",
}
ACTION_VALUE_SEMANTICS_BY_SUBTYPE: dict[TaskSubtype, ActionValueSemantics] = {
    "base_rate_neglect": "posterior_probability_comparison",
    "conjunction_fallacy": "probability_comparison",
    "gambler_fallacy": "claim_correctness",
    "sample_size_neglect": "binomial_tail_probability_comparison",
    "overprecision_calibration": "interval_coverage_comparison",
}
PROBABILITY_FIELD_KEYS = {
    "prior_high",
    "p_signal_high_given_high",
    "p_signal_high_given_low",
    "posterior_high",
    "posterior_low",
    "p_heads",
    "baseline_rate",
    "extreme_threshold",
    "probability",
}
logger = logging.getLogger(__name__)


class BeliefBiasGenerator(BaseGenerator):
    """Generate normative belief-judgment tasks with behaviorally tempting framing."""

    EXPECTED_UTILITY_PRECISION = 6
    CHOICE_TIE_EPSILON = 1e-6

    def __init__(
        self,
        seed: int | None = None,
        version: str = "v1",
        prompt_style: PromptStyle = "default",
        prompt_style_regime: ConfiguredPromptStyleRegime = "neutral_realistic",
        prompt_frame_variant: PromptFrameVariant = "auto",
    ):
        if prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(
                f"Unsupported prompt_style: {prompt_style}. "
                f"Expected one of {SUPPORTED_PROMPT_STYLES}."
            )
        if prompt_frame_variant not in SUPPORTED_PROMPT_FRAME_VARIANTS:
            raise ValueError(
                f"Unsupported prompt_frame_variant: {prompt_frame_variant}. "
                f"Expected one of {SUPPORTED_PROMPT_FRAME_VARIANTS}."
            )
        if prompt_style_regime not in SUPPORTED_PROMPT_STYLE_REGIMES:
            raise ValueError(
                f"Unsupported prompt_style_regime: {prompt_style_regime}. "
                f"Expected one of {SUPPORTED_PROMPT_STYLE_REGIMES}."
            )
        self.rng = random.Random(seed)
        self.base_seed = seed if seed is not None else 0
        self.version = version
        self.prompt_style = prompt_style
        self.prompt_style_regime = prompt_style_regime
        self.prompt_frame_variant = prompt_frame_variant
        self._last_prompt_frame_variant: PromptFrameVariant = "neutral_statistical"
        self._last_prompt_style_regime: PromptStyleRegime = "neutral_realistic"
        self._last_conjunction_render_mode: str | None = None
        self._last_representativeness_strength: str | None = None
        self._last_streak_domain: str | None = None
        self._difficulty_counts: Counter[str] = Counter()
        self._prompt_style_counts: Counter[str] = Counter()
        self._task_subtype_counts: Counter[str] = Counter()
        self.sample_index = 0

    def _resolve_prompt_style(self) -> PromptStyle:
        if self.prompt_style == "random":
            return self.rng.choice(list(NON_RANDOM_PROMPT_STYLES))
        return self.prompt_style

    def _resolve_prompt_style_regime(self) -> PromptStyleRegime:
        if self.prompt_style_regime == "random":
            return self.rng.choice(list(NON_RANDOM_PROMPT_STYLE_REGIMES))
        return self.prompt_style_regime

    def _prompt_style_tier(self, style: PromptStyle, regime: PromptStyleRegime) -> str:
        if regime == "normative_explicit":
            return "formal"
        if regime == "bias_eliciting" or style in {"finance_framed", "unlabeled"}:
            return "naturalistic"
        return "neutral_natural"

    def _select_template_variant(
        self,
        *,
        task_subtype: TaskSubtype,
        frame_variant: PromptFrameVariant,
        tier: str,
        problem_spec: ProblemSpec,
        templates: list[str] | tuple[str, ...],
    ) -> int:
        return self.select_template_index_balanced(
            task_subtype=task_subtype,
            frame_variant=frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            templates=templates,
        )

    def _resolve_prompt_frame_variant(
        self,
        *,
        task_subtype: TaskSubtype,
        problem_spec: ProblemSpec,
        prompt_style: PromptStyle,
        prompt_style_regime: PromptStyleRegime,
    ) -> PromptFrameVariant:
        if self.prompt_frame_variant != "auto":
            return self.prompt_frame_variant
        payload = {
            "task_subtype": task_subtype,
            "problem_spec": problem_spec,
            "prompt_style": prompt_style,
            "prompt_style_regime": prompt_style_regime,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        frame_candidates = self._frame_candidates_for_subtype(task_subtype)
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        return frame_candidates[digest[0] % len(frame_candidates)]

    def _frame_candidates_for_subtype(
        self, task_subtype: TaskSubtype
    ) -> tuple[PromptFrameVariant, ...]:
        if task_subtype == "gambler_fallacy":
            return ("sports_streak", "market_streak", "roulette_streak", "neutral_coin")
        if task_subtype == "conjunction_fallacy":
            return ("vivid_description", "plain_probability", "representative_profile")
        if task_subtype == "sample_size_neglect":
            return ("hospital_births", "fund_returns", "quality_control_batches")
        if task_subtype == "overprecision_calibration":
            return ("analyst_forecast", "weather_forecast", "startup_projection")
        return (
            "medical_screening",
            "fraud_detection",
            "hiring_screen",
            "security_alert",
            "trading_signal",
            "neutral_statistical",
        )

    def _apply_prompt_frame_variant(
        self,
        *,
        prompt: str,
        frame_variant: PromptFrameVariant,
        task_subtype: TaskSubtype,
    ) -> str:
        if task_subtype != "base_rate_neglect":
            # Subtype renderers handle substantive framing directly for
            # conjunction/gambler/sample-size/overprecision.
            return prompt
        lead_by_variant = {
            "medical_screening": (
                "Interpret high as condition-present and low as condition-absent."
            ),
            "fraud_detection": ("Interpret high as fraudulent and low as legitimate."),
            "hiring_screen": ("Interpret high as strong-fit and low as weak-fit."),
            "security_alert": ("Interpret high as active threat and low as no threat."),
            "trading_signal": (
                "Interpret high as favorable market regime and low as unfavorable regime."
            ),
            "neutral_statistical": ("Interpret this as a generic noisy classification decision."),
        }
        lead = lead_by_variant.get(frame_variant, "")
        if not lead:
            return prompt
        return f"{lead} {prompt}"

    def generate(self) -> DataPoint:
        current_index = self.sample_index
        self.sample_index += 1
        subtype: TaskSubtype = self.rng.choice(
            [
                "base_rate_neglect",
                "conjunction_fallacy",
                "gambler_fallacy",
                "sample_size_neglect",
                "overprecision_calibration",
            ]
        )
        if subtype == "base_rate_neglect":
            return self._generate_base_rate_neglect(sample_index=current_index)
        if subtype == "conjunction_fallacy":
            return self._generate_conjunction_fallacy(sample_index=current_index)
        if subtype == "gambler_fallacy":
            return self._generate_gambler_fallacy(sample_index=current_index)
        if subtype == "sample_size_neglect":
            return self._generate_sample_size_neglect(sample_index=current_index)
        if subtype == "overprecision_calibration":
            return self._generate_overprecision_calibration(sample_index=current_index)
        raise ValueError(f"Unknown subtype: {subtype}")

    def _metadata(
        self,
        sample_index: int,
        difficulty_metrics: DifficultyMetrics,
        resolved_prompt_style: PromptStyle,
        prompt_frame_variant: PromptFrameVariant,
        example_fingerprint: str,
        tie_threshold: float,
        prompt_style_regime: PromptStyleRegime | None = None,
    ) -> Metadata:
        return Metadata(
            generator_name=self.__class__.__name__,
            version=self.version,
            seed=self.base_seed,
            dataset_role="normative_training",
            requested_prompt_style=self.prompt_style,
            resolved_prompt_style=resolved_prompt_style,
            prompt_style_regime=prompt_style_regime or self._last_prompt_style_regime,
            prompt_frame_variant=prompt_frame_variant,
            prompt_has_action_labels=resolved_prompt_style != "unlabeled",
            example_fingerprint=example_fingerprint,
            tie_threshold=tie_threshold,
            sample_index=sample_index,
            conjunction_render_mode=self._last_conjunction_render_mode,
            representativeness_strength=self._last_representativeness_strength,
            streak_domain=self._last_streak_domain,
            difficulty_metrics=difficulty_metrics,
        )

    def _task_id(self, prefix: str, sample_index: int) -> str:
        return f"{prefix}_{self.version}_{sample_index:06d}"

    def _compute_example_fingerprint(
        self,
        *,
        task_subtype: TaskSubtype,
        problem_spec: ProblemSpec,
        optimal_decision: str,
    ) -> str:
        payload = {
            "task_subtype": task_subtype,
            "problem_spec": problem_spec,
            "optimal_decision": optimal_decision,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _assert_probabilities_in_unit_interval(
        self, value: Any, *, path: str = "problem_spec"
    ) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                nested_path = f"{path}.{key}"
                if key in PROBABILITY_FIELD_KEYS or key.startswith("p_"):
                    if isinstance(nested, (int, float)) and not (0.0 <= float(nested) <= 1.0):
                        raise ValueError(f"{nested_path} must be in [0, 1], got {nested}.")
                self._assert_probabilities_in_unit_interval(nested, path=nested_path)
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                self._assert_probabilities_in_unit_interval(nested, path=f"{path}[{index}]")

    def _assert_prompt_option_distinction(self, *, prompt: str, target: Target) -> None:
        lower_prompt = prompt.lower()
        left_action = target.comparison_pair["left_action"].lower()
        right_action = target.comparison_pair["right_action"].lower()
        has_left = left_action in lower_prompt
        has_right = right_action in lower_prompt
        if has_left != has_right:
            raise ValueError(
                "Prompt mentions only one comparison action label, "
                "which can create option ambiguity."
            )
        if not has_left and {left_action, right_action} == {"choose_a", "choose_b"}:
            has_option_a = "option a" in lower_prompt or "a='" in lower_prompt
            has_option_b = "option b" in lower_prompt or "b='" in lower_prompt
            if not (has_option_a and has_option_b):
                raise ValueError(
                    "Prompt must clearly distinguish both options for choose_A/choose_B tasks."
                )

    def _validate_conjunction_option_structure(
        self,
        *,
        option_a: dict[str, object],
        option_b: dict[str, object],
    ) -> None:
        role_a = option_a.get("event_role")
        role_b = option_b.get("event_role")
        if {role_a, role_b} != {"conjunction", "constituent"}:
            raise ValueError(
                "conjunction_fallacy options must include one conjunction and one constituent."
            )
        conjunction_option = option_a if role_a == "conjunction" else option_b
        constituent_option = option_a if role_a == "constituent" else option_b
        conjunction_label = str(conjunction_option.get("event_label", ""))
        constituent_label = str(constituent_option.get("event_label", ""))
        if not conjunction_label or not constituent_label:
            raise ValueError(
                "conjunction_fallacy options must include non-empty event_label fields."
            )
        if " and " not in conjunction_label.lower():
            raise ValueError(
                "conjunction_fallacy conjunction label must include an explicit conjunction."
            )
        if constituent_label.lower() not in conjunction_label.lower():
            raise ValueError(
                "conjunction_fallacy conjunction label must be stricter than its constituent label."
            )

    def _record_distribution_counts(
        self, *, task_subtype: TaskSubtype, prompt_style: PromptStyle, difficulty: str
    ) -> None:
        self._task_subtype_counts[task_subtype] += 1
        self._prompt_style_counts[prompt_style] += 1
        self._difficulty_counts[difficulty] += 1
        sample_total = sum(self._task_subtype_counts.values())
        if sample_total == 1 or sample_total % 50 == 0:
            logger.info(
                (
                    "Generation distributions after %s samples | difficulty=%s | "
                    "prompt_style=%s | task_subtype=%s"
                ),
                sample_total,
                dict(self._difficulty_counts),
                dict(self._prompt_style_counts),
                dict(self._task_subtype_counts),
            )

    def _comparison_pair_for_subtype(self, subtype: TaskSubtype) -> ComparisonPair:
        pair = COMPARISON_PAIR_BY_SUBTYPE[subtype]
        return {
            "left_action": pair["left_action"],
            "right_action": pair["right_action"],
        }

    def _prompt_renderer_for_subtype(self, subtype: TaskSubtype) -> Callable[..., str]:
        renderer_method_name = PROMPT_RENDERER_METHOD_BY_SUBTYPE[subtype]
        renderer = getattr(self, renderer_method_name)
        return renderer

    def _choose_optimal_action(
        self,
        *,
        left_label: str,
        left_value: float,
        right_label: str,
        right_value: float,
        epsilon: float | None = None,
    ) -> str:
        tie_epsilon = self.CHOICE_TIE_EPSILON if epsilon is None else epsilon
        if abs(left_value - right_value) <= tie_epsilon:
            return "indifferent"
        if left_value > right_value:
            return left_label
        if right_value > left_value:
            return right_label
        return "indifferent"

    def _build_target(
        self,
        *,
        objective: str,
        state: dict,
        beliefs: dict,
        constraints: dict,
        actions: list[str],
        comparison_pair: ComparisonPair,
        outcome_model: dict,
        action_values: ActionScalars | dict[str, float],
        decision_values: ActionScalars | dict[str, float],
        optimal_decision: str,
        solver_trace: SolverTrace,
        brief_rationale: str,
    ) -> Target:
        return Target(
            objective=objective,
            state=state,
            beliefs=beliefs,
            constraints=constraints,
            actions=actions,
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal_decision,
            solver_trace=solver_trace,
            brief_rationale=brief_rationale,
        )

    def _build_solver_trace(
        self,
        *,
        comparison_pair: ComparisonPair,
        decision_values: ActionScalars,
        tie_epsilon: float,
        comparison_result: str,
    ) -> SolverTrace:
        left_action = comparison_pair["left_action"]
        right_action = comparison_pair["right_action"]
        return {
            "left_action": left_action,
            "right_action": right_action,
            "left_value": float(decision_values[left_action]),
            "right_value": float(decision_values[right_action]),
            "tie_epsilon": float(tie_epsilon),
            "comparison_result": comparison_result,
        }

    def _normalize_actions_for_comparison_pair(
        self, *, actions: list[str], comparison_pair: ComparisonPair
    ) -> list[str]:
        left_action = comparison_pair["left_action"]
        right_action = comparison_pair["right_action"]
        ordered_actions: list[str] = [left_action, right_action]
        for action in actions:
            if action not in ordered_actions:
                ordered_actions.append(action)
        return ordered_actions

    def _normalize_scalars_for_comparison_pair(
        self,
        *,
        values: ActionScalars | dict[str, float],
        comparison_pair: ComparisonPair,
    ) -> ActionScalars:
        left_action = comparison_pair["left_action"]
        right_action = comparison_pair["right_action"]
        normalized_values = ActionScalars(values)
        if left_action not in normalized_values or right_action not in normalized_values:
            raise ValueError(
                "Action scalar map must include comparison_pair left_action and right_action."
            )
        ordered_values = ActionScalars(
            {
                left_action: normalized_values[left_action],
                right_action: normalized_values[right_action],
            }
        )
        for action, value in normalized_values.items():
            if action not in ordered_values:
                ordered_values[action] = value
        return ordered_values

    def _require_mapping(
        self, value: object, *, field_path: str, subtype: str
    ) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError(f"{field_path} must be a mapping for subtype '{subtype}'.")
        return value

    def _require_numeric_field(
        self,
        mapping: dict[str, object],
        *,
        key: str,
        field_path: str,
        subtype: str,
    ) -> float:
        if key not in mapping:
            raise ValueError(f"{field_path}.{key} is required for subtype '{subtype}'.")
        value = mapping[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_path}.{key} must be numeric for subtype '{subtype}'.")
        return float(value)

    def _require_probability_field(
        self,
        mapping: dict[str, object],
        *,
        key: str,
        field_path: str,
        subtype: str,
    ) -> float:
        value = self._require_numeric_field(
            mapping, key=key, field_path=field_path, subtype=subtype
        )
        if value < 0 or value > 1:
            raise ValueError(f"{field_path}.{key} must be between 0 and 1 for subtype '{subtype}'.")
        return value

    def _require_non_negative_field(
        self,
        mapping: dict[str, object],
        *,
        key: str,
        field_path: str,
        subtype: str,
    ) -> float:
        value = self._require_numeric_field(
            mapping, key=key, field_path=field_path, subtype=subtype
        )
        if value < 0:
            raise ValueError(f"{field_path}.{key} must be non-negative for subtype '{subtype}'.")
        return value

    def _require_signal(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> Literal["high", "low"]:
        if key not in mapping:
            raise ValueError(f"{field_path}.{key} is required for subtype '{subtype}'.")
        value = mapping[key]
        if value not in ("high", "low"):
            raise ValueError(f"{field_path}.{key} must be 'high' or 'low' for subtype '{subtype}'.")
        return value

    def _round_scalar(self, value: int | float) -> float:
        return round(float(value), self.EXPECTED_UTILITY_PRECISION)

    def _round_problem_decimal(self, value: int | float, *, digits: int = 2) -> float:
        return round(float(value), digits)

    def _format_number(self, value: int | float) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1 + erf(x / sqrt(2)))

    def _normal_interval_coverage(
        self, *, lower: float, upper: float, mu: float, sd: float
    ) -> float:
        if sd <= 0:
            raise ValueError("Standard deviation must be positive for interval coverage.")
        return self._normal_cdf((upper - mu) / sd) - self._normal_cdf((lower - mu) / sd)

    def _posterior_from_signal(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
    ) -> float:
        if observed_signal == "high":
            like_high = p_signal_high_given_high
            like_low = p_signal_high_given_low
        else:
            like_high = 1 - p_signal_high_given_high
            like_low = 1 - p_signal_high_given_low
        numerator = prior_high * like_high
        denominator = numerator + (1 - prior_high) * like_low
        if denominator <= 0:
            raise ValueError("Bayesian posterior denominator must be positive.")
        return numerator / denominator

    def _binomial_extreme_probability(
        self,
        *,
        sample_size: int,
        baseline_rate: float,
        extreme_threshold: float,
        extreme_direction: Literal["at_or_above", "at_or_below"],
    ) -> float:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        threshold_count = self._extreme_count_cutoff(
            sample_size=sample_size,
            extreme_threshold=extreme_threshold,
            extreme_direction=extreme_direction,
        )
        if extreme_direction == "at_or_above":
            start = max(0, threshold_count)
            end = sample_size
        else:
            start = 0
            end = min(sample_size, threshold_count)
        probability = 0.0
        for k in range(start, end + 1):
            probability += (
                comb(sample_size, k)
                * (baseline_rate**k)
                * ((1 - baseline_rate) ** (sample_size - k))
            )
        return probability

    def _extreme_count_cutoff(
        self,
        *,
        sample_size: int,
        extreme_threshold: float,
        extreme_direction: Literal["at_or_above", "at_or_below"],
    ) -> int:
        raw_cutoff = sample_size * extreme_threshold
        if extreme_direction == "at_or_above":
            return int(ceil(raw_cutoff))
        return int(floor(raw_cutoff))

    def _solve_from_problem_spec(
        self, *, problem_spec: ProblemSpec
    ) -> tuple[ActionScalars, ActionScalars, str, PosteriorBeliefs]:
        subtype_raw = problem_spec.get("task_subtype")
        if subtype_raw not in COMPARISON_PAIR_BY_SUBTYPE:
            raise ValueError(
                "problem_spec.task_subtype must be one of "
                f"{sorted(COMPARISON_PAIR_BY_SUBTYPE.keys())}; got {subtype_raw!r}."
            )
        subtype = subtype_raw
        assumptions = self._require_mapping(
            problem_spec.get("assumptions"),
            field_path="problem_spec.assumptions",
            subtype=subtype,
        )
        options = self._require_mapping(
            problem_spec.get("options"),
            field_path="problem_spec.options",
            subtype=subtype,
        )

        if subtype == "base_rate_neglect":
            prior_high = self._require_probability_field(
                assumptions,
                key="prior_high",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            p_signal_high_given_high = self._require_probability_field(
                assumptions,
                key="p_signal_high_given_high",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            p_signal_high_given_low = self._require_probability_field(
                assumptions,
                key="p_signal_high_given_low",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            observed_signal = self._require_signal(
                assumptions,
                key="observed_signal",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            posterior_high = self._posterior_from_signal(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
            )
            posterior_low = 1 - posterior_high
            choose_state_high = self._round_scalar(posterior_high)
            choose_state_low = self._round_scalar(posterior_low)
            action_values = ActionScalars(
                {
                    "choose_state_high": choose_state_high,
                    "choose_state_low": choose_state_low,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_state_high",
                left_value=choose_state_high,
                right_label="choose_state_low",
                right_value=choose_state_low,
                epsilon=tie_epsilon,
            )
            posterior_beliefs: PosteriorBeliefs = {
                "prior_high": self._round_scalar(prior_high),
                "prior_low": self._round_scalar(1 - prior_high),
                "posterior_high": choose_state_high,
                "posterior_low": choose_state_low,
            }
            return action_values, decision_values, optimal_decision, posterior_beliefs

        if subtype == "conjunction_fallacy":
            # This is an event-statement comparison task grounded in the
            # conjunction axiom (conjunction <= constituent), not just a numeric
            # max operation. Explicit probabilities are attached for deterministic
            # grading and verifier reproducibility.
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            p_a = self._require_probability_field(
                option_a,
                key="probability",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_b = self._require_probability_field(
                option_b,
                key="probability",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            self._validate_conjunction_option_structure(
                option_a=option_a,
                option_b=option_b,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_a = self._round_scalar(p_a)
            choose_b = self._round_scalar(p_b)
            action_values = ActionScalars({"choose_A": choose_a, "choose_B": choose_b})
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_A",
                left_value=choose_a,
                right_label="choose_B",
                right_value=choose_b,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, {}

        if subtype == "gambler_fallacy":
            # This subtype scores claim correctness under independence, not the
            # probability of next outcomes. Action values are binary truth scores.
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            p_heads = self._require_probability_field(
                assumptions,
                key="p_heads",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            if abs(p_heads - 0.5) > 1e-9:
                raise ValueError(
                    "problem_spec.assumptions.p_heads must equal 0.5 for subtype 'gambler_fallacy'."
                )
            queried_outcome = assumptions.get("queried_outcome")
            if queried_outcome not in ("heads", "tails"):
                raise ValueError(
                    "problem_spec.assumptions.queried_outcome must be heads or tails "
                    "for subtype 'gambler_fallacy'."
                )
            recent_sequence = assumptions.get("recent_sequence")
            if not isinstance(recent_sequence, str) or not recent_sequence:
                raise ValueError(
                    "problem_spec.assumptions.recent_sequence must be a non-empty "
                    "string for subtype 'gambler_fallacy'."
                )
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )

            def _claim_value(claim_option: dict[str, object], path: str) -> float:
                claim_type = claim_option.get("claim_type")
                if claim_type not in ("more_likely", "not_more_likely"):
                    raise ValueError(
                        f"{path}.claim_type must be more_likely or not_more_likely "
                        f"for subtype '{subtype}'."
                    )
                return 1.0 if claim_type == "not_more_likely" else 0.0

            choose_a = self._round_scalar(_claim_value(option_a, "problem_spec.options.A"))
            choose_b = self._round_scalar(_claim_value(option_b, "problem_spec.options.B"))
            action_values = ActionScalars({"choose_A": choose_a, "choose_B": choose_b})
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_A",
                left_value=choose_a,
                right_label="choose_B",
                right_value=choose_b,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, {}

        if subtype == "sample_size_neglect":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )

            def _extract_process(
                process_option: dict[str, object], path: str
            ) -> tuple[int, float, float, Literal["at_or_above", "at_or_below"]]:
                n = int(
                    self._require_non_negative_field(
                        process_option,
                        key="sample_size",
                        field_path=path,
                        subtype=subtype,
                    )
                )
                if n <= 0:
                    raise ValueError(
                        f"{path}.sample_size must be positive for subtype '{subtype}'."
                    )
                baseline_rate = self._require_probability_field(
                    process_option,
                    key="baseline_rate",
                    field_path=path,
                    subtype=subtype,
                )
                extreme_threshold = self._require_probability_field(
                    process_option,
                    key="extreme_threshold",
                    field_path=path,
                    subtype=subtype,
                )
                direction_raw = process_option.get("extreme_direction")
                if direction_raw not in ("at_or_above", "at_or_below"):
                    raise ValueError(
                        f"{path}.extreme_direction must be at_or_above or at_or_below "
                        f"for subtype '{subtype}'."
                    )
                return n, baseline_rate, extreme_threshold, direction_raw

            n_a, baseline_a, threshold_a, direction_a = _extract_process(
                option_a, "problem_spec.options.A"
            )
            n_b, baseline_b, threshold_b, direction_b = _extract_process(
                option_b, "problem_spec.options.B"
            )

            p_a = self._round_scalar(
                self._binomial_extreme_probability(
                    sample_size=n_a,
                    baseline_rate=baseline_a,
                    extreme_threshold=threshold_a,
                    extreme_direction=direction_a,
                )
            )
            p_b = self._round_scalar(
                self._binomial_extreme_probability(
                    sample_size=n_b,
                    baseline_rate=baseline_b,
                    extreme_threshold=threshold_b,
                    extreme_direction=direction_b,
                )
            )
            action_values = ActionScalars({"choose_A": p_a, "choose_B": p_b})
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_A",
                left_value=p_a,
                right_label="choose_B",
                right_value=p_b,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, {}

        if subtype == "overprecision_calibration":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            true_value_sd = self._require_non_negative_field(
                assumptions,
                key="true_value_sd",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            true_value_mean = self._require_numeric_field(
                assumptions,
                key="true_value_mean",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            if true_value_sd <= 0:
                raise ValueError(
                    "problem_spec.assumptions.true_value_sd must be positive for "
                    "subtype 'overprecision_calibration'."
                )

            def _coverage(interval_option: dict[str, object], path: str) -> float:
                center = self._require_numeric_field(
                    interval_option,
                    key="center",
                    field_path=path,
                    subtype=subtype,
                )
                lower = self._require_numeric_field(
                    interval_option,
                    key="lower",
                    field_path=path,
                    subtype=subtype,
                )
                upper = self._require_numeric_field(
                    interval_option,
                    key="upper",
                    field_path=path,
                    subtype=subtype,
                )
                if not (lower <= center <= upper):
                    raise ValueError(
                        f"{path}.center must lie within [lower, upper] for subtype '{subtype}'."
                    )
                # Interval center is part of framing/coherence; scoring is still
                # coverage of [lower, upper] under the assumed true-value distribution.
                return self._normal_interval_coverage(
                    lower=lower,
                    upper=upper,
                    mu=true_value_mean,
                    sd=true_value_sd,
                )

            coverage_a = self._round_scalar(_coverage(option_a, "problem_spec.options.A"))
            coverage_b = self._round_scalar(_coverage(option_b, "problem_spec.options.B"))
            action_values = ActionScalars({"choose_A": coverage_a, "choose_B": coverage_b})
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_A",
                left_value=coverage_a,
                right_label="choose_B",
                right_value=coverage_b,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, {}

        raise ValueError(f"Unsupported subtype: {subtype}")

    def _verify_target_solution(
        self, *, problem_spec: ProblemSpec, target: Target, prompt: str = ""
    ) -> None:
        self._assert_probabilities_in_unit_interval(problem_spec)
        (
            solved_action_values,
            solved_decision_values,
            solved_optimal_decision,
            solved_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        normalized_solved_action_values = self._normalize_scalars_for_comparison_pair(
            values=solved_action_values,
            comparison_pair=target.comparison_pair,
        )
        normalized_solved_decision_values = self._normalize_scalars_for_comparison_pair(
            values=solved_decision_values,
            comparison_pair=target.comparison_pair,
        )
        tolerance = 1e-9

        def _assert_scalar_map_matches(
            field_name: str,
            expected: ActionScalars,
            actual: ActionScalars,
        ) -> None:
            if list(expected.keys()) != list(actual.keys()):
                raise ValueError(
                    f"{field_name} action ordering does not match solver output: "
                    f"expected {list(expected.keys())}, got {list(actual.keys())}."
                )
            for action, expected_value in expected.items():
                actual_value = float(actual[action])
                if abs(actual_value - expected_value) > tolerance:
                    raise ValueError(
                        f"{field_name}.{action} mismatch: expected {expected_value}, "
                        f"got {actual_value}."
                    )

        _assert_scalar_map_matches(
            "target.action_values",
            normalized_solved_action_values,
            target.action_values,
        )
        _assert_scalar_map_matches(
            "target.decision_values",
            normalized_solved_decision_values,
            target.decision_values,
        )
        if target.optimal_decision != solved_optimal_decision:
            raise ValueError(
                "target.optimal_decision mismatch: "
                f"expected {solved_optimal_decision}, got {target.optimal_decision}."
            )
        if solved_beliefs:
            for key, solved_value in solved_beliefs.items():
                actual_value = float(target.beliefs.get(key, -1))
                if abs(actual_value - solved_value) > tolerance:
                    raise ValueError(
                        f"target.beliefs.{key} mismatch: expected {solved_value}, "
                        f"got {actual_value}."
                    )
            posterior_high = float(target.beliefs.get("posterior_high", 0.0))
            posterior_low = float(target.beliefs.get("posterior_low", 0.0))
            if abs((posterior_high + posterior_low) - 1.0) > tolerance:
                raise ValueError("Posterior probabilities in target.beliefs must sum to 1.")
        if prompt:
            self._assert_prompt_option_distinction(prompt=prompt, target=target)

    def _assemble_normative_datapoint(
        self,
        *,
        sample_index: int,
        task_subtype: TaskSubtype,
        task_id_prefix: str,
        problem_spec: ProblemSpec,
        prompt: str,
        state: dict,
        beliefs: dict,
        actions: list[str],
        comparison_pair: ComparisonPair,
        outcome_model: dict[str, str],
        action_values: ActionScalars | dict[str, float],
        decision_values: ActionScalars | dict[str, float],
        optimal_decision: str,
        brief_rationale: str,
        difficulty_metrics: DifficultyMetrics,
        prompt_style: PromptStyle,
        tie_threshold: float,
        constraints: dict | None = None,
    ) -> DataPoint:
        normalized_actions = self._normalize_actions_for_comparison_pair(
            actions=actions,
            comparison_pair=comparison_pair,
        )
        normalized_action_values = self._normalize_scalars_for_comparison_pair(
            values=action_values,
            comparison_pair=comparison_pair,
        )
        normalized_decision_values = self._normalize_scalars_for_comparison_pair(
            values=decision_values,
            comparison_pair=comparison_pair,
        )
        solver_trace = self._build_solver_trace(
            comparison_pair=comparison_pair,
            decision_values=normalized_decision_values,
            tie_epsilon=tie_threshold,
            comparison_result=optimal_decision,
        )
        target = self._build_target(
            objective=problem_spec["objective"],
            state=state,
            beliefs=beliefs,
            constraints={} if constraints is None else constraints,
            actions=normalized_actions,
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=normalized_action_values,
            decision_values=normalized_decision_values,
            optimal_decision=optimal_decision,
            solver_trace=solver_trace,
            brief_rationale=brief_rationale,
        )
        self._verify_target_solution(problem_spec=problem_spec, target=target, prompt=prompt)
        example_fingerprint = self._compute_example_fingerprint(
            task_subtype=task_subtype,
            problem_spec=problem_spec,
            optimal_decision=target.optimal_decision,
        )
        difficulty_metrics_with_semantics = dict(difficulty_metrics)
        difficulty_metrics_with_semantics["action_value_semantics"] = (
            ACTION_VALUE_SEMANTICS_BY_SUBTYPE[task_subtype]
        )
        datapoint = DataPoint(
            task_family="belief_bias_judgment",
            task_subtype=task_subtype,
            task_id=self._task_id(task_id_prefix, sample_index),
            difficulty=self._difficulty_for(task_subtype, difficulty_metrics_with_semantics),
            problem_spec=problem_spec,
            input=prompt,
            target=target,
            metadata=self._metadata(
                sample_index,
                difficulty_metrics_with_semantics,
                prompt_style,
                self._last_prompt_frame_variant,
                example_fingerprint,
                tie_threshold,
            ),
        )
        self._record_distribution_counts(
            task_subtype=task_subtype,
            prompt_style=prompt_style,
            difficulty=datapoint.difficulty,
        )
        return datapoint

    def _difficulty_for(
        self,
        subtype: TaskSubtype,
        difficulty_metrics: DifficultyMetrics | None = None,
    ) -> str:
        _ = difficulty_metrics
        return BELIEF_BIAS_DEFAULT_DIFFICULTY_BY_SUBTYPE[subtype]

    def _difficulty_metrics(
        self,
        *,
        left_value: float,
        right_value: float,
        numeric_complexity: int,
        evidence_count: int = 1,
        prompt_complexity_features: DifficultyMetrics | None = None,
    ) -> DifficultyMetrics:
        metrics: DifficultyMetrics = {
            "value_gap": round(abs(left_value - right_value), 6),
            "numeric_complexity": numeric_complexity,
            "evidence_count": evidence_count,
        }
        if prompt_complexity_features is not None:
            metrics.update(prompt_complexity_features)
        return metrics

    def _compute_prompt_complexity_features(
        self,
        *,
        prompt: str,
        prompt_style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        numeric_values: list[int | float],
        comparison_pair: ComparisonPair,
    ) -> DifficultyMetrics:
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        lower_prompt = prompt.lower()
        clause_count = 1 + sum(
            lower_prompt.count(token)
            for token in (",", ";", " or ", " and ", " otherwise ", " then ")
        )
        has_decimal_in_prompt = bool(re.search(r"\d+\.\d+", prompt))
        left_action = comparison_pair["left_action"].lower()
        right_action = comparison_pair["right_action"].lower()
        action_token_patterns = [r"\bchoose_[a-z0-9_]+\b"]
        for action_label in (left_action, right_action):
            if action_label.startswith("choose_"):
                continue
            action_token_patterns.append(rf"\b{re.escape(action_label)}\b")

        action_tokens: list[str] = []
        for pattern in action_token_patterns:
            action_tokens.extend(re.findall(pattern, lower_prompt))
        prompt_action_token_count = len(action_tokens)
        prompt_contains_action_tokens = prompt_action_token_count > 0
        asymmetric_choice_framing = left_action.split("_", 1)[0] != right_action.split("_", 1)[0]
        prompt_complexity = (
            clause_count
            + int(has_decimal_in_prompt)
            + int(asymmetric_choice_framing)
            + int(prompt_contains_action_tokens)
        )
        numeric_span = 0.0
        if numeric_values:
            min_value = min(float(v) for v in numeric_values)
            max_value = max(float(v) for v in numeric_values)
            numeric_span = max_value - min_value
        return {
            "prompt_style_variant": prompt_style,
            "prompt_style_regime": resolved_regime,
            "prompt_clause_count": clause_count,
            "prompt_has_decimal": has_decimal_in_prompt,
            "prompt_asymmetric_choice_framing": asymmetric_choice_framing,
            "prompt_contains_action_tokens": prompt_contains_action_tokens,
            "prompt_action_token_count": prompt_action_token_count,
            "prompt_complexity": prompt_complexity,
            "prompt_numeric_span": round(numeric_span, 6),
        }

    def _build_prompt_and_complexity(
        self,
        *,
        task_subtype: TaskSubtype,
        problem_spec: ProblemSpec,
        numeric_values: list[int | float],
        comparison_pair: ComparisonPair,
    ) -> tuple[str, PromptStyle, DifficultyMetrics]:
        renderer = self._prompt_renderer_for_subtype(task_subtype)
        prompt_style = self._resolve_prompt_style()
        prompt_style_regime = self._resolve_prompt_style_regime()
        frame_variant = self._resolve_prompt_frame_variant(
            task_subtype=task_subtype,
            problem_spec=problem_spec,
            prompt_style=prompt_style,
            prompt_style_regime=prompt_style_regime,
        )
        prompt = ""
        qa_failures: list[dict[str, str]] = []
        max_attempts = 4
        for _ in range(max_attempts):
            prompt = renderer(
                problem_spec=problem_spec,
                style=prompt_style,
                prompt_style_regime=prompt_style_regime,
                prompt_frame_variant=frame_variant,
            )
            prompt = self._apply_prompt_frame_variant(
                prompt=prompt,
                frame_variant=frame_variant,
                task_subtype=task_subtype,
            )
            prompt = self._collapse_stacked_prompt_wrappers(prompt=prompt)
            self.assert_prompt_regime_no_leakage(
                prompt=prompt,
                prompt_style_regime=prompt_style_regime,
            )
            qa_failures = self._qa_validate_rendered_prompt(
                task_subtype=task_subtype,
                prompt=prompt,
                problem_spec=problem_spec,
                frame_variant=frame_variant,
                prompt_style_regime=prompt_style_regime,
            )
            if not qa_failures:
                break
        if qa_failures:
            raise ValueError(
                "Rendered prompt failed QA after retries: "
                f"{json.dumps(qa_failures, sort_keys=True)}"
            )
        self._last_prompt_frame_variant = frame_variant
        self._last_prompt_style_regime = prompt_style_regime
        self._last_conjunction_render_mode = None
        self._last_representativeness_strength = None
        self._last_streak_domain = None
        if task_subtype == "conjunction_fallacy":
            self._last_conjunction_render_mode = prompt_style_regime
            strength_by_variant = {
                "plain_probability": "low",
                "representative_profile": "medium",
                "vivid_description": "high",
            }
            self._last_representativeness_strength = strength_by_variant.get(
                frame_variant, "medium"
            )
        if task_subtype == "gambler_fallacy":
            domain_by_variant = {
                "neutral_coin": "coin",
                "roulette_streak": "roulette",
                "sports_streak": "basketball",
                "market_streak": "market",
            }
            self._last_streak_domain = domain_by_variant.get(frame_variant, "coin")
        prompt_complexity_features = self._compute_prompt_complexity_features(
            prompt=prompt,
            prompt_style=prompt_style,
            prompt_style_regime=prompt_style_regime,
            numeric_values=numeric_values,
            comparison_pair=comparison_pair,
        )
        prompt_complexity_features["prompt_frame_variant"] = frame_variant
        return prompt, prompt_style, prompt_complexity_features

    def _compute_numeric_complexity(
        self,
        *,
        numeric_values: list[int | float],
        arithmetic_operations: int,
    ) -> int:
        distinct_values = {round(float(value), 6) for value in numeric_values}
        has_decimal = any(not float(value).is_integer() for value in numeric_values)
        complexity = len(distinct_values) + arithmetic_operations
        if has_decimal:
            complexity += 1
        return complexity

    def _count_arithmetic_operations(self, expression: str) -> int:
        token_types: list[str] = []
        index = 0
        expression_length = len(expression)
        while index < expression_length:
            char = expression[index]
            if char.isspace():
                index += 1
                continue
            if char in "+-*/":
                token_types.append(char)
                index += 1
                continue
            if char == "(":
                token_types.append("LPAREN")
                index += 1
                continue
            if char == ")":
                token_types.append("RPAREN")
                index += 1
                continue
            if char.isdigit() or char == ".":
                index += 1
                while index < expression_length and (
                    expression[index].isdigit() or expression[index] == "."
                ):
                    index += 1
                token_types.append("NUMBER")
                continue
            if char.isalpha() or char == "_":
                index += 1
                while index < expression_length and (
                    expression[index].isalnum() or expression[index] == "_"
                ):
                    index += 1
                token_types.append("IDENT")
                continue
            index += 1
        operation_count = 0
        operand_end_tokens = {"NUMBER", "IDENT", "RPAREN"}
        operand_start_tokens = {"NUMBER", "IDENT", "LPAREN"}

        def _starts_operand(start_index: int) -> bool:
            if start_index >= len(token_types):
                return False
            token_type = token_types[start_index]
            if token_type in operand_start_tokens:
                return True
            if token_type in ("+", "-"):
                return _starts_operand(start_index + 1)
            return False

        for token_index, token_type in enumerate(token_types):
            if token_type in ("+", "*", "/"):
                operation_count += 1
                continue
            if token_type != "-":
                continue
            previous_token_type = token_types[token_index - 1] if token_index > 0 else None
            is_binary_minus = previous_token_type in operand_end_tokens and _starts_operand(
                token_index + 1
            )
            if is_binary_minus:
                operation_count += 1
        return operation_count

    def _count_operations_in_outcome_model(self, outcome_model: dict[str, str]) -> int:
        return sum(self._count_arithmetic_operations(formula) for formula in outcome_model.values())

    def _build_base_rate_neglect_assumptions(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
    ) -> BayesianAssumptions:
        return {
            "prior_high": prior_high,
            "p_signal_high_given_high": p_signal_high_given_high,
            "p_signal_high_given_low": p_signal_high_given_low,
            "observed_signal": observed_signal,
            "signal_model": "binary_conditional_likelihood",
            "decision_rule": "bayes_update_then_expected_value",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_base_rate_neglect_problem_spec(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
    ) -> BaseRateNeglectProblemSpec:
        return {
            "task_subtype": "base_rate_neglect",
            "objective": "infer the more likely class using Bayes rule",
            "options": {
                "A": {"type": "state_hypothesis", "state": "high"},
                "B": {"type": "state_hypothesis", "state": "low"},
            },
            "assumptions": self._build_base_rate_neglect_assumptions(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
            ),
        }

    def _build_conjunction_fallacy_assumptions(
        self,
        *,
        profile_description: str,
        semantic_domain: str,
    ) -> ConjunctionFallacyAssumptions:
        return {
            "decision_rule": "probability_axiom_comparison",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
            "profile_description": profile_description,
            "semantic_domain": semantic_domain,
        }

    def _build_conjunction_fallacy_problem_spec(
        self,
        *,
        profile_description: str,
        semantic_domain: str,
        event_a_label: str,
        event_b_detail_label: str,
        p_event_a: float,
        p_event_a_and_b: float,
        conjunction_in_option_a: bool,
    ) -> ConjunctionFallacyProblemSpec:
        # Store event-statement text plus explicit probabilities so scoring is
        # deterministic/verifier-friendly while the behavioral target remains the
        # conjunction-axiom relation between constituent and conjunction events.
        constituent = {
            "type": "event_probability_claim",
            "event_role": "constituent",
            "event_label": event_a_label,
            "probability": p_event_a,
        }
        conjunction_label = (
            event_b_detail_label
            if event_a_label.lower() in event_b_detail_label.lower()
            else f"{event_a_label} and {event_b_detail_label}"
        )
        conjunction = {
            "type": "event_probability_claim",
            "event_role": "conjunction",
            "event_label": conjunction_label,
            "probability": p_event_a_and_b,
        }
        if conjunction_in_option_a:
            option_a = conjunction
            option_b = constituent
        else:
            option_a = constituent
            option_b = conjunction
        return {
            "task_subtype": "conjunction_fallacy",
            "objective": "select the more probable event statement",
            "options": {
                "A": option_a,
                "B": option_b,
            },
            "assumptions": self._build_conjunction_fallacy_assumptions(
                profile_description=profile_description,
                semantic_domain=semantic_domain,
            ),
        }

    def _build_gambler_fallacy_assumptions(
        self,
        *,
        p_heads: float,
        queried_outcome: Literal["heads", "tails"],
        recent_sequence: str,
    ) -> GamblerFallacyAssumptions:
        return {
            "p_heads": p_heads,
            "queried_outcome": queried_outcome,
            "recent_sequence": recent_sequence,
            "independence_assumption": True,
            # Scored as claim correctness under independence, not direct
            # next-outcome probability comparison.
            "decision_rule": "independence_claim_validation",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_gambler_fallacy_problem_spec(
        self,
        *,
        p_heads: float,
        queried_outcome: Literal["heads", "tails"],
        recent_sequence: str,
        correct_claim_in_option_a: bool,
    ) -> GamblerFallacyProblemSpec:
        # Represent one shared streak context plus two competing claims about it.
        # The solver evaluates which claim is normatively correct under independence.
        claim_true = {
            "type": "independence_claim",
            "claim_type": "not_more_likely",
        }
        claim_false = {
            "type": "independence_claim",
            "claim_type": "more_likely",
        }
        option_a = claim_true if correct_claim_in_option_a else claim_false
        option_b = claim_false if correct_claim_in_option_a else claim_true
        return {
            "task_subtype": "gambler_fallacy",
            "objective": "identify which claim is normatively correct under independence",
            "options": {
                "A": option_a,
                "B": option_b,
            },
            "assumptions": self._build_gambler_fallacy_assumptions(
                p_heads=p_heads,
                queried_outcome=queried_outcome,
                recent_sequence=recent_sequence,
            ),
        }

    def _build_sample_size_neglect_assumptions(
        self,
        *,
        baseline_rate: float,
        extreme_threshold: float,
        extreme_direction: Literal["at_or_above", "at_or_below"],
    ) -> SampleSizeNeglectAssumptions:
        return {
            "baseline_rate": baseline_rate,
            "extreme_threshold": extreme_threshold,
            "extreme_direction": extreme_direction,
            # Compare binomial tail probabilities induced by a proportion threshold
            # (mapped to integer count cutoffs via the rounding rule below).
            "decision_rule": "compare_extreme_frequency_probability",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_sample_size_neglect_problem_spec(
        self,
        *,
        sample_size_a: int,
        sample_size_b: int,
        baseline_rate: float,
        extreme_threshold: float,
        extreme_direction: Literal["at_or_above", "at_or_below"],
    ) -> SampleSizeNeglectProblemSpec:
        return {
            "task_subtype": "sample_size_neglect",
            "objective": "choose the process more likely to show an extreme sample frequency",
            "options": {
                "A": {
                    "type": "sample_process",
                    "sample_size": sample_size_a,
                    "baseline_rate": baseline_rate,
                    "extreme_threshold": extreme_threshold,
                    "extreme_direction": extreme_direction,
                },
                "B": {
                    "type": "sample_process",
                    "sample_size": sample_size_b,
                    "baseline_rate": baseline_rate,
                    "extreme_threshold": extreme_threshold,
                    "extreme_direction": extreme_direction,
                },
            },
            "assumptions": self._build_sample_size_neglect_assumptions(
                baseline_rate=baseline_rate,
                extreme_threshold=extreme_threshold,
                extreme_direction=extreme_direction,
            ),
        }

    def _build_overprecision_calibration_assumptions(
        self, *, true_value_mean: float, true_value_sd: float
    ) -> OverprecisionCalibrationAssumptions:
        return {
            "error_model": "normal",
            "true_value_mean": true_value_mean,
            "true_value_sd": true_value_sd,
            "decision_rule": "maximize_interval_coverage",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_overprecision_calibration_problem_spec(
        self,
        *,
        center_a: float,
        lower_a: float,
        upper_a: float,
        center_b: float,
        lower_b: float,
        upper_b: float,
        true_value_mean: float,
        true_value_sd: float,
    ) -> OverprecisionCalibrationProblemSpec:
        normalized_center_a = self._round_problem_decimal(center_a, digits=2)
        normalized_lower_a = self._round_problem_decimal(lower_a, digits=2)
        normalized_upper_a = self._round_problem_decimal(upper_a, digits=2)
        normalized_center_b = self._round_problem_decimal(center_b, digits=2)
        normalized_lower_b = self._round_problem_decimal(lower_b, digits=2)
        normalized_upper_b = self._round_problem_decimal(upper_b, digits=2)
        # Centers are explicit for interval framing; normative scoring is based on
        # distributional coverage of [lower, upper] around true_value_mean/sd.
        return {
            "task_subtype": "overprecision_calibration",
            "objective": "select the interval with higher probability of containing the true value",
            "options": {
                "A": {
                    "type": "prediction_interval",
                    "center": normalized_center_a,
                    "lower": normalized_lower_a,
                    "upper": normalized_upper_a,
                },
                "B": {
                    "type": "prediction_interval",
                    "center": normalized_center_b,
                    "lower": normalized_lower_b,
                    "upper": normalized_upper_b,
                },
            },
            "assumptions": self._build_overprecision_calibration_assumptions(
                true_value_mean=true_value_mean, true_value_sd=true_value_sd
            ),
        }

    # Style goal: present screening judgments as realistic decisions. Non-formal
    # variants intentionally use behaviorally tempting framing; reserve explicit
    # method instruction (e.g., Bayes rule) for formal style.
    def _render_base_rate_neglect_prompt(
        self,
        *,
        problem_spec: BaseRateNeglectProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        assumptions = problem_spec["assumptions"]
        prior_high = assumptions["prior_high"]
        p_pos_given_high = assumptions["p_signal_high_given_high"]
        p_pos_given_low = assumptions["p_signal_high_given_low"]
        observed_signal = assumptions["observed_signal"]
        test_result = "positive" if observed_signal == "high" else "negative"
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                "A screening model has base rate P(class_high)="
                f"{self._format_number(prior_high)}. "
                "Signal reliability is P(positive|class_high)="
                f"{self._format_number(p_pos_given_high)} and P(positive|class_low)="
                f"{self._format_number(p_pos_given_low)}. "
                f"The observed result is {test_result}. "
                "Given those rates and this result, which class now seems more likely?"
            )
        elif tier == "neutral_natural":
            body = (
                "A screening team reviews cases where only a fraction are truly high-risk: "
                f"{self._format_number(prior_high)}. "
                "The alert appears with chance "
                f"{self._format_number(p_pos_given_high)} for high-risk and "
                f"{self._format_number(p_pos_given_low)} for low-risk. "
                f"This case came back {test_result}. Which class is more likely?"
            )
        else:
            body = (
                "A credit screen has default base rate "
                f"{self._format_number(prior_high)}. "
                "It flags true defaulters with probability "
                f"{self._format_number(p_pos_given_high)} and flags non-defaulters "
                f"with probability {self._format_number(p_pos_given_low)}. "
                f"This applicant was flagged {test_result}. Which borrower state is more likely?"
            )

        if style == "unlabeled":
            return body
        return f"{body}\n- choose_state_high\n- choose_state_low"

    # Style goal: make this feel like choosing between two plausible statements.
    # Non-formal variants intentionally lean on vivid wording that can tempt
    # representativeness; formal style carries explicit logical terminology.
    def _render_conjunction_fallacy_prompt(
        self,
        *,
        problem_spec: ConjunctionFallacyProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        assumptions = problem_spec["assumptions"]
        frame_variant = prompt_frame_variant or "plain_probability"
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        profile_text = assumptions.get("profile_description", "You receive a short profile.")
        profile_text_clean = profile_text.rstrip(" .")
        domain_text = str(assumptions.get("semantic_domain", "general")).replace("_", " ")
        domain = assumptions.get("semantic_domain", "general")

        def _vivid_scene_text() -> str:
            if domain == "startup":
                return (
                    "The founder is rotating through customer calls, the product team is "
                    "shipping fast, launch momentum is building, and investor chatter is active."
                )
            if domain == "nonprofit":
                return (
                    "The organizer is juggling donor calls, volunteer rosters, and community-site "
                    "logistics before the next public event."
                )
            if domain == "scientist":
                return (
                    "The lab runs late with replication checks, draft figures, and back-to-back "
                    "method meetings before submission."
                )
            if domain == "campaign":
                return (
                    "Volunteers are knocking doors, turnout lists are updating in real time, and "
                    "field leads are coordinating last-minute coverage."
                )
            if domain == "sales":
                return (
                    "The rep is in consecutive discovery calls, updating deal notes, and preparing "
                    "a final proposal review with procurement."
                )
            if domain == "product_manager":
                return (
                    "The PM is moving between user interviews, roadmap tradeoff meetings, and "
                    "release-readiness check-ins."
                )
            if domain == "community_organizer":
                return (
                    "Neighborhood leaders are comparing attendance sheets, volunteer sign-ups, and "
                    "follow-up plans before a civic meeting."
                )
            if domain == "teacher":
                return (
                    "The classroom day includes quick assessments, one-on-one feedback, and lesson "
                    "adjustments based on live student progress."
                )
            if domain == "market":
                return (
                    "Screens are green into the close, desks are adding risk in spots, "
                    "and volatility chatter is loud across the floor."
                )
            if domain == "investor":
                return (
                    "The investor keeps a disciplined routine: regular rebalancing, long "
                    "research notes, and values screens on every allocation meeting."
                )
            if domain == "borrower":
                return (
                    "Bills are paid on schedule, utilization stays controlled, and account "
                    "activity shows steady month-to-month discipline."
                )
            if domain == "candidate":
                return (
                    "The candidate arrives with organized work samples, crisp follow-through, "
                    "and a pattern of reliable execution under deadlines."
                )
            if domain == "resident":
                return (
                    "Neighbors recognize this resident from planning meetings, block updates, "
                    "and recurring turnout at local community events."
                )
            return "Concrete details in the case line up tightly and make the situation feel vivid."

        def _representative_profile_text() -> str:
            if domain == "startup":
                return (
                    "an execution-focused startup operator with strong customer signal and "
                    "high follow-through"
                )
            if domain == "nonprofit":
                return (
                    "a mission-driven nonprofit organizer with reliable donor and volunteer "
                    "coordination"
                )
            if domain == "scientist":
                return (
                    "a methodical researcher with high procedural discipline and replication focus"
                )
            if domain == "campaign":
                return (
                    "a field-operations volunteer with strong turnout and outreach habits"
                )
            if domain == "sales":
                return (
                    "a structured enterprise seller with disciplined pipeline execution"
                )
            if domain == "product_manager":
                return (
                    "a cross-functional product operator with evidence-driven prioritization"
                )
            if domain == "community_organizer":
                return "a civic organizer with durable neighborhood coalition behavior"
            if domain == "teacher":
                return "a high-structure educator with consistent progress monitoring"
            if domain == "market":
                return "a risk-on tape with improving breadth and softer fear signals"
            if domain == "investor":
                return "a methodical long-horizon allocator with a values-aware tilt"
            if domain == "borrower":
                return (
                    "a low-volatility credit behavior pattern with consistently prudent habits"
                )
            if domain == "candidate":
                return (
                    "a conscientious high-structure contributor with dependable follow-through"
                )
            if domain == "resident":
                return (
                    "a civically engaged neighborhood participant with steady local involvement"
                )
            return f"a recognizable {domain} pattern"

        def _plain_summary_text() -> str:
            return f"In this {domain_text} case, {profile_text_clean}."

        option_pair = f"A='{option_a['event_label']}' | B='{option_b['event_label']}'"
        # Template banks vary discourse form (memo/snapshot/committee) to reduce
        # structural repetition across frame variants while preserving semantics.
        if tier == "formal":
            if frame_variant == "vivid_description":
                templates = [
                    (
                        f"{_vivid_scene_text()} Compare probability for {option_pair}."
                    ),
                    (
                        f"{_vivid_scene_text()} Determine which statement is more probable: "
                        f"{option_pair}."
                    ),
                    (
                        f"{_vivid_scene_text()} Rank {option_pair} by probability."
                    ),
                    (
                        f"{_vivid_scene_text()} Which statement has higher probability: "
                        f"{option_pair}?"
                    ),
                ]
            elif frame_variant == "representative_profile":
                templates = [
                    (
                        "Category-match framing: this looks like "
                        f"{_representative_profile_text()}. "
                        f"Compare event probabilities for "
                        f"{option_pair}."
                    ),
                    (
                        f"Using a category-fit lens, this resembles "
                        f"{_representative_profile_text()}. Rank {option_pair} by probability."
                    ),
                    (
                        f"Treat this as {_representative_profile_text()}. Select the more "
                        f"probable statement in "
                        f"{option_pair}."
                    ),
                    (
                        f"This case maps to {_representative_profile_text()}. Which statement is "
                        "more probable: "
                        f"{option_pair}?"
                    ),
                ]
            else:
                templates = [
                    (
                        f"{_plain_summary_text()} "
                        f"Which statement has higher probability: {option_pair}?"
                    ),
                    (
                        f"{_plain_summary_text()} "
                        f"Evaluate {option_pair} and select the more probable statement."
                    ),
                    (
                        f"{_plain_summary_text()} "
                        f"Pick the higher-probability line from {option_pair}."
                    ),
                    (
                        f"{_plain_summary_text()} "
                        f"Which statement is more probable in {option_pair}?"
                    ),
                ]
        elif tier == "neutral_natural":
            if frame_variant == "representative_profile":
                representative_core = _representative_profile_text().rstrip(".")
                templates = [
                    (
                        f"This case reads like {representative_core}. "
                        f"Which statement is more likely: {option_pair}?"
                    ),
                    (
                        f"Given this category match ({representative_core}), "
                        f"which statement is more likely: "
                        f"{option_pair}?"
                    ),
                    (
                        f"The case resembles {representative_core}. "
                        "If you sort it by category fit, which statement is more "
                        f"likely: {option_pair}?"
                    ),
                    (
                        f"Category-read from this case: {representative_core}. "
                        f"Choose the more likely line in {option_pair}."
                    ),
                    (
                        f"Category-fit view: {representative_core}. "
                        f"Which statement is likelier for this case: "
                        f"{option_pair}?"
                    ),
                ]
            elif frame_variant == "vivid_description":
                templates = [
                    (
                        f"{_vivid_scene_text()} With those details in mind, which statement "
                        f"is more likely: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} Given those details, which statement is more "
                        f"likely: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} From these details, which statement sounds more "
                        f"likely in that "
                        f"situation: {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} Which line is more likely given these details: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} From the scene details, choose the statement "
                        f"that is more likely: "
                        f"{option_pair}."
                    ),
                ]
            else:
                templates = [
                    (
                        f"Facts: {profile_text_clean}. Which statement is more likely: "
                        f"{option_pair}?"
                    ),
                    (
                        f"Facts only: {profile_text_clean}. Which line is more likely, "
                        f"{option_pair}?"
                    ),
                    (
                        f"Using these details only: {profile_text_clean}. "
                        f"Which statement is more likely: {option_pair}?"
                    ),
                    (
                        f"{profile_text_clean}. Compare A and B directly: {option_pair}. "
                        "Which line is likelier?"
                    ),
                    (
                        f"Just the facts: {profile_text_clean}. Which statement is more likely, "
                        f"{option_pair}?"
                    ),
                ]
        else:
            if frame_variant == "vivid_description":
                templates = [
                    (
                        f"{_vivid_scene_text()} "
                        "Scene momentum is obvious. "
                        f"At first glance, which line feels more fitting: {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} "
                        "The story momentum is building. "
                        f"Instant read: which statement feels like the better fit, {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} "
                        "Narrative fit check. "
                        f"First impression only: which line seems to match best, {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} "
                        "Scene-first instinct. "
                        f"Gut-check: which statement feels more fitting right away, {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} "
                        "Story cues are strong. "
                        f"Snap judgment: which line seems like the natural fit, {option_pair}?"
                    ),
                    (
                        f"{_vivid_scene_text()} "
                        "Narrative pressure is high. "
                        "If you must choose immediately, which statement feels more fitting: "
                        f"{option_pair}?"
                    ),
                ]
            elif frame_variant == "representative_profile":
                templates = [
                    (
                        f"Category-match read: this feels like {_representative_profile_text()}. "
                        "First-glance fit read: "
                        "which statement feels more fitting, "
                        f"{option_pair}?"
                    ),
                    (
                        f"This case reads like {_representative_profile_text()}. "
                        "Quick fit intuition: "
                        f"which line feels like the better fit, {option_pair}?"
                    ),
                    (
                        f"Category pull is strong: {_representative_profile_text()}. "
                        f"Immediate fit read: which statement feels more representative, "
                        f"{option_pair}?"
                    ),
                    (
                        f"Category-fit cue: {_representative_profile_text()}. "
                        f"First-impression call: which line seems to match best, {option_pair}?"
                    ),
                    (
                        f"Fit instinct: {_representative_profile_text()}. "
                        f"On-the-spot read: which statement feels more fitting, "
                        f"{option_pair}?"
                    ),
                    (
                        f"Match cue: {_representative_profile_text()}. "
                        "Under time pressure, which line looks like the better match first: "
                        f"{option_pair}?"
                    ),
                ]
            else:
                templates = [
                    (
                        f"{profile_text_clean}. At first glance, which line seems likelier: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{profile_text_clean}. On a first-pass read, which statement do you pick: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{profile_text_clean}. Which statement do you lean toward first: "
                        f"{option_pair}?"
                    ),
                    (
                        f"{_plain_summary_text()} First-pass choice: which line do you take, "
                        f"{option_pair}?"
                    ),
                    (
                        f"{profile_text_clean}. On an immediate read, which statement seems more "
                        f"likely, "
                        f"{option_pair}?"
                    ),
                    (
                        f"{_plain_summary_text()} If you had to decide now, which statement do "
                        "you lean toward at first glance: "
                        f"{option_pair}?"
                    ),
                ]
        template_idx = self._select_template_variant(
            task_subtype="conjunction_fallacy",
            frame_variant=frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            templates=templates,
        )
        body = templates[template_idx]

        if style == "unlabeled":
            return body
        return f"{body}\n- choose_A\n- choose_B"

    # Style goal: present streak judgments in everyday language.
    # Neutral prompts stay plain; bias-eliciting prompts amplify streak salience
    # without explicitly teaching or naming the fallacy.
    def _indefinite_article_for(self, term: str) -> str:
        token = term.strip().split()[0].lower() if term.strip() else ""
        return "an" if token[:1] in {"a", "e", "i", "o", "u"} else "a"

    def _comparative_outcome_phrase(
        self,
        *,
        frame_variant: PromptFrameVariant,
        queried_term: str,
        opposite_term: str,
        more_likely: bool,
        phrase_style_idx: int = 0,
    ) -> str:
        polarity_terms = (
            ("more likely", "not more likely")
            if more_likely
            else ("not more likely", "no more likely")
        )
        if frame_variant == "market_streak":
            queried_dir = queried_term.replace(" day", "")
            opposite_dir = opposite_term.replace(" day", "")
            level_word = {"up": "higher", "down": "lower"}
            queried_level = level_word.get(queried_dir, queried_dir)
            opposite_level = level_word.get(opposite_dir, opposite_dir)
            variants = [
                f"{polarity_terms[0]} to end {queried_dir} than {opposite_dir}",
                f"{polarity_terms[0]} to close {queried_dir} than {opposite_dir}",
                f"{polarity_terms[0]} to finish {queried_dir} than {opposite_dir}",
                f"{polarity_terms[1]} to close {queried_level} than {opposite_level}",
            ]
            return variants[phrase_style_idx % len(variants)]
        if frame_variant == "sports_streak":
            variants = [
                f"{polarity_terms[0]} to be a {queried_term} than a {opposite_term}",
                f"{polarity_terms[0]} to {queried_term} than {opposite_term}",
                f"{polarity_terms[0]} to end as a {queried_term} than a {opposite_term}",
                f"{polarity_terms[1]} to come out as a {queried_term} than a {opposite_term}",
            ]
            return variants[phrase_style_idx % len(variants)]
        if frame_variant == "roulette_streak":
            variants = [
                f"{polarity_terms[0]} to land {queried_term} than {opposite_term}",
                f"{polarity_terms[0]} to hit {queried_term} than {opposite_term}",
                f"{polarity_terms[0]} to come up {queried_term} than {opposite_term}",
                f"{polarity_terms[1]} to land {queried_term} than {opposite_term}",
            ]
            return variants[phrase_style_idx % len(variants)]
        queried_article = self._indefinite_article_for(queried_term)
        opposite_article = self._indefinite_article_for(opposite_term)
        polarity = "more likely" if more_likely else "not more likely"
        return (
            f"{polarity} than {opposite_term}"
            if queried_term in {"heads", "tails"}
            else (
                f"{polarity} to be {queried_article} {queried_term} "
                f"rather than {opposite_article} {opposite_term}"
            )
        )

    def _gambler_claim_text(
        self,
        *,
        claim_type: str,
        queried_outcome: Literal["heads", "tails"],
        frame_variant: PromptFrameVariant,
        phrase_style_idx: int = 0,
    ) -> str:
        outcome_terms_by_variant: dict[str, tuple[str, str]] = {
            "sports_streak": ("make", "miss"),
            "market_streak": ("up day", "down day"),
            "roulette_streak": ("red", "black"),
            "neutral_coin": ("heads", "tails"),
        }
        heads_term, tails_term = outcome_terms_by_variant.get(
            frame_variant, outcome_terms_by_variant["neutral_coin"]
        )
        queried_term = heads_term if queried_outcome == "heads" else tails_term
        opposite_term = tails_term if queried_outcome == "heads" else heads_term
        leadins_more = (
            "because of what just happened,",
            "after that streak,",
            "off that run,",
            "from this run,",
        )
        leadins_not_more = (
            "given this streak,",
            "from this run alone,",
            "after that run,",
            "on this evidence,",
        )
        leadin = (
            leadins_more[phrase_style_idx % len(leadins_more)]
            if claim_type == "more_likely"
            else leadins_not_more[phrase_style_idx % len(leadins_not_more)]
        )
        if frame_variant == "sports_streak":
            comparison = self._comparative_outcome_phrase(
                frame_variant=frame_variant,
                queried_term=queried_term,
                opposite_term=opposite_term,
                more_likely=(claim_type == "more_likely"),
                phrase_style_idx=phrase_style_idx,
            )
            return f"{leadin} the next shot is {comparison}"
        if frame_variant == "market_streak":
            comparison = self._comparative_outcome_phrase(
                frame_variant=frame_variant,
                queried_term=queried_term,
                opposite_term=opposite_term,
                more_likely=(claim_type == "more_likely"),
                phrase_style_idx=phrase_style_idx,
            )
            return f"{leadin} the next session is {comparison}"
        if frame_variant == "roulette_streak":
            comparison = self._comparative_outcome_phrase(
                frame_variant=frame_variant,
                queried_term=queried_term,
                opposite_term=opposite_term,
                more_likely=(claim_type == "more_likely"),
                phrase_style_idx=phrase_style_idx,
            )
            return f"{leadin} the next spin is {comparison}"
        if claim_type == "more_likely":
            return (
                f"{leadin} {queried_term} should be more likely than "
                f"{opposite_term} on the very next flip"
            )
        return (
            f"{leadin} {queried_term} is not more likely than {opposite_term} on the very next flip"
        )

    def _render_gambler_fallacy_prompt(
        self,
        *,
        problem_spec: GamblerFallacyProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        streak = assumptions["recent_sequence"]
        queried_outcome = assumptions["queried_outcome"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        frame_variant = prompt_frame_variant or "neutral_coin"
        neutral_context_by_variant = {
            "neutral_coin": (f"Sequence on the board: {streak}. Interpret H=heads and T=tails."),
            "roulette_streak": (
                f"Wheel history panel shows: {streak}. At this table, H=red and T=black."
            ),
            "sports_streak": (
                f"Scoreboard of the last shots reads: {streak}. For this drill, H=make and T=miss."
            ),
            "market_streak": (
                f"Market tape for recent sessions is: {streak}. "
                "In this toy setup, H=up day and T=down day."
            ),
        }
        bias_context_by_variant = {
            "neutral_coin": (
                f"The run jumps out on the chart: {streak}. Treat H as heads and T as tails."
            ),
            "roulette_streak": (
                f"The wheel board is lit up with: {streak}. "
                "At this table, H means red and T means black."
            ),
            "sports_streak": (
                f"The gym is staring at the shot chart: {streak}. Here, H=make and T=miss."
            ),
            "market_streak": (
                f"Traders are fixated on this tape run: {streak}. "
                "In this toy tape, H=up day and T=down day."
            ),
        }
        neutral_context = neutral_context_by_variant.get(
            frame_variant, neutral_context_by_variant["neutral_coin"]
        )
        bias_context = bias_context_by_variant.get(
            frame_variant, bias_context_by_variant["neutral_coin"]
        )

        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        claim_phrase_style_idx = self._select_template_variant(
            task_subtype="gambler_fallacy",
            frame_variant=frame_variant,
            tier=f"{tier}_claim_text",
            problem_spec=problem_spec,
            templates=["v1", "v2", "v3"],
        )
        claim_a = self._gambler_claim_text(
            claim_type=option_a["claim_type"],
            queried_outcome=queried_outcome,
            frame_variant=frame_variant,
            phrase_style_idx=claim_phrase_style_idx,
        )
        claim_b = self._gambler_claim_text(
            claim_type=option_b["claim_type"],
            queried_outcome=queried_outcome,
            frame_variant=frame_variant,
            phrase_style_idx=claim_phrase_style_idx,
        )
        if tier == "formal":
            if frame_variant == "sports_streak":
                body = (
                    f"In a shooting drill, the recent make/miss sequence is {streak} "
                    "(H=make, T=miss). Assume each shot remains 50/50 and independent of "
                    "earlier shots. Evaluate the two claims and select the correct one "
                    "for the very next attempt."
                )
            elif frame_variant == "roulette_streak":
                body = (
                    f"Roulette wheel history shows {streak} (H=red, T=black). Assume each "
                    "spin remains 50/50 and independent of prior spins. Which claim about "
                    "the next spin is normatively correct?"
                )
            elif frame_variant == "market_streak":
                body = (
                    f"A toy market log shows {streak} (H=up session, T=down session). "
                    "Assume unchanged 50/50 odds and independence across sessions. Which "
                    "claim is correct for the next session under those assumptions?"
                )
            else:
                body = (
                    f"A fair process produced this streak: {streak}. "
                    "Assume unchanged 50/50 odds and trial independence. "
                    "Determine which claim is correct about the very next outcome."
                )
        elif tier == "neutral_natural":
            if frame_variant == "roulette_streak":
                body = f"{neutral_context} Which statement is better about the next spin?"
            elif frame_variant == "sports_streak":
                body = f"{neutral_context} For the very next attempt, which statement is right?"
            elif frame_variant == "market_streak":
                body = f"{neutral_context} For the next session, which statement is right?"
            else:
                body = f"{neutral_context} Which statement is right about the next outcome?"
        else:
            if frame_variant == "roulette_streak":
                templates = [
                    (
                        f"{bias_context} A lot of eyes are on the wheel now. "
                        "What is your first call on the next spin?"
                    ),
                    (
                        f"{bias_context} Chips are already down and the run is hard to ignore. "
                        "Which claim do you lean toward right now?"
                    ),
                    (
                        f"{bias_context} The board keeps flashing the streak as bets lock in. "
                        "Which claim feels more convincing at a glance?"
                    ),
                ]
            elif frame_variant == "sports_streak":
                templates = [
                    (
                        f"{bias_context} The crowd is locked on the run before the next release. "
                        "What is your first read?"
                    ),
                    (
                        f"{bias_context} Teammates are already saying the streak is due to flip. "
                        "Which claim do you reach for first?"
                    ),
                    (
                        f"{bias_context} The run is all anyone is talking about at the line. "
                        "Which claim feels likelier right now?"
                    ),
                ]
            elif frame_variant == "market_streak":
                templates = [
                    (
                        f"{bias_context} The streak is dominating desk chatter into the close. "
                        "What is your first take for the next print?"
                    ),
                    (
                        f"{bias_context} Screens keep replaying the run as positions are set. "
                        "Which claim do you lean toward?"
                    ),
                    (
                        f"{bias_context} Attention is fixed on whether the streak finally breaks. "
                        "Which claim feels more plausible right now?"
                    ),
                ]
            else:
                templates = [
                    (
                        f"{bias_context} The run stands out and people expect a turn. "
                        "Which claim feels more plausible right now?"
                    ),
                    (
                        f"{bias_context} The streak is hard to unsee as the next flip approaches. "
                        "What is your first read?"
                    ),
                    (
                        f"{bias_context} The sequence is the only thing people are focused on. "
                        "Which claim do you lean toward right away?"
                    ),
                ]
            template_idx = self._select_template_variant(
                task_subtype="gambler_fallacy",
                frame_variant=frame_variant,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]

        if style == "unlabeled":
            return f"{body} Option A: {claim_a}. Option B: {claim_b}."
        return f"{body}\n- choose_A: {claim_a}\n- choose_B: {claim_b}"

    # Style goal: highlight intuitive small-vs-large group judgments. Non-formal
    # variants intentionally make this feel like ordinary comparative judgment;
    # keep heavy statistical phrasing for formal style only.
    def _render_sample_size_neglect_prompt(
        self,
        *,
        problem_spec: SampleSizeNeglectProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        threshold = option_a["extreme_threshold"]
        direction = option_a["extreme_direction"]
        baseline_rate = option_a["baseline_rate"]
        direction_text = "at least" if direction == "at_or_above" else "at most"
        frame_variant = prompt_frame_variant or "hospital_births"
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            if frame_variant == "fund_returns":
                templates = [
                    (
                        f"Compute then compare fund-tail probabilities: baseline up-share "
                        f"{self._format_number(baseline_rate)}, threshold {direction_text} "
                        f"{self._format_number(threshold)}. "
                        f"A uses n={option_a['sample_size']}, B uses n={option_b['sample_size']}."
                    ),
                    (
                        f"Fund review sheet: with baseline {self._format_number(baseline_rate)} "
                        f"and criterion {direction_text} {self._format_number(threshold)}, "
                        "determine whether "
                        f"n={option_a['sample_size']} or n={option_b['sample_size']} "
                        "has the larger tail probability."
                    ),
                    (
                        "Probability ranking task (funds): "
                        f"base rate={self._format_number(baseline_rate)}, "
                        f"event={direction_text} {self._format_number(threshold)} up-share. "
                        f"Compare A(n={option_a['sample_size']}) vs B(n={option_b['sample_size']})."
                    ),
                    (
                        f"Analytic brief: two funds share baseline up-share "
                        f"{self._format_number(baseline_rate)}. For the event "
                        f"'{direction_text} {self._format_number(threshold)}', "
                        f"select the process with higher realization probability: "
                        f"A n={option_a['sample_size']} or B n={option_b['sample_size']}."
                    ),
                    (
                        f"Evaluate tail-likelihood choice (fund returns): start with "
                        f"baseline up-share {self._format_number(baseline_rate)}, apply event "
                        f"{direction_text} {self._format_number(threshold)}, then compare "
                        f"A(n={option_a['sample_size']}) against B(n={option_b['sample_size']})."
                    ),
                    (
                        f"Using the same baseline up-share "
                        f"{self._format_number(baseline_rate)}, compare which fund is more likely "
                        f"to realize {direction_text} {self._format_number(threshold)} "
                        f"up-share in one day. Evaluate Fund A "
                        f"(n={option_a['sample_size']}) versus Fund B "
                        f"(n={option_b['sample_size']})."
                    ),
                    (
                        f"Determination prompt: with the same long-run up-share "
                        f"{self._format_number(baseline_rate)}, compute which fund sample size "
                        "makes the stated extreme more probable. "
                        f"Compare A={option_a['sample_size']} and B={option_b['sample_size']} "
                        f"for event {direction_text} {self._format_number(threshold)}."
                    ),
                ]
            elif frame_variant == "quality_control_batches":
                templates = [
                    (
                        f"Compute then compare batch-tail probabilities: baseline pass rate "
                        f"{self._format_number(baseline_rate)}, threshold {direction_text} "
                        f"{self._format_number(threshold)}. "
                        f"A inspects n={option_a['sample_size']}, "
                        f"B inspects n={option_b['sample_size']}."
                    ),
                    (
                        f"QC comparison sheet: with baseline {self._format_number(baseline_rate)} "
                        f"and event {direction_text} {self._format_number(threshold)} pass-share, "
                        f"determine whether line A(n={option_a['sample_size']}) or "
                        f"line B(n={option_b['sample_size']}) is more likely."
                    ),
                    (
                        f"Under the same pass-rate baseline "
                        f"{self._format_number(baseline_rate)}, identify which production line is "
                        f"more likely to show a batch with {direction_text} "
                        f"{self._format_number(threshold)} pass-share. Compare line A "
                        f"(n={option_a['sample_size']}) versus line B "
                        f"(n={option_b['sample_size']})."
                    ),
                    (
                        f"Analytic QC brief: two lines share baseline pass rate "
                        f"{self._format_number(baseline_rate)}. For the event "
                        f"{direction_text} {self._format_number(threshold)} pass-share, "
                        f"pick the higher-probability process: "
                        f"A n={option_a['sample_size']} or B n={option_b['sample_size']}."
                    ),
                    (
                        f"Evaluate tail-probability choice (QC): baseline pass rate="
                        f"{self._format_number(baseline_rate)}, event={direction_text} "
                        f"{self._format_number(threshold)} pass-share. Compare "
                        f"A(n={option_a['sample_size']}) with B(n={option_b['sample_size']})."
                    ),
                    (
                        f"Compare which QC line has the larger binomial-tail probability, given "
                        f"shared pass rate {self._format_number(baseline_rate)} and criterion "
                        f"{direction_text} {self._format_number(threshold)} pass-share. "
                        f"Line A inspects n={option_a['sample_size']}; "
                        f"line B inspects n={option_b['sample_size']}."
                    ),
                    (
                        f"Decision worksheet (quality control): quantify the event "
                        f"{direction_text} {self._format_number(threshold)} using baseline "
                        f"{self._format_number(baseline_rate)}, then select the higher "
                        f"probability process between A={option_a['sample_size']} and "
                        f"B={option_b['sample_size']} inspections."
                    ),
                ]
            else:
                templates = [
                    (
                        f"Compute then compare birth-tail probabilities: baseline girl-share "
                        f"{self._format_number(baseline_rate)}, event {direction_text} "
                        f"{self._format_number(threshold)}. "
                        f"Hospital A has n={option_a['sample_size']} births/day; "
                        f"Hospital B has n={option_b['sample_size']} births/day."
                    ),
                    (
                        f"Hospital comparison sheet: with baseline "
                        f"{self._format_number(baseline_rate)}, determine whether "
                        f"Hospital A (n={option_a['sample_size']}) or "
                        f"Hospital B (n={option_b['sample_size']}) "
                        f"is more likely to produce {direction_text} "
                        f"{self._format_number(threshold)} girl-share in one day."
                    ),
                    (
                        f"With the same baseline girl-share "
                        f"{self._format_number(baseline_rate)}, compare which hospital is more "
                        f"likely to record {direction_text} {self._format_number(threshold)} "
                        f"girl-share in a day. Compare Hospital A "
                        f"(n={option_a['sample_size']}) versus Hospital B "
                        f"(n={option_b['sample_size']})."
                    ),
                    (
                        f"Analytic births brief: two hospitals share baseline "
                        f"girl-share {self._format_number(baseline_rate)}. "
                        f"For event {direction_text} {self._format_number(threshold)}, "
                        f"select the higher-probability hospital process: "
                        f"Hospital A n={option_a['sample_size']} or "
                        f"Hospital B n={option_b['sample_size']}."
                    ),
                    (
                        f"Evaluate extreme-day likelihood (births): baseline girl-share "
                        f"{self._format_number(baseline_rate)}, threshold "
                        f"{direction_text} {self._format_number(threshold)}. "
                        f"Compare hospital sample sizes A={option_a['sample_size']} and "
                        f"B={option_b['sample_size']}."
                    ),
                    (
                        f"Determine which hospital has the larger binomial-tail probability for "
                        f"event {direction_text} {self._format_number(threshold)} girl-share "
                        f"under baseline {self._format_number(baseline_rate)}. "
                        f"Hospital A has n={option_a['sample_size']}; "
                        f"Hospital B has n={option_b['sample_size']}."
                    ),
                    (
                        f"Hospital probability comparison sheet: use the same baseline "
                        f"girl-share {self._format_number(baseline_rate)} and compute which "
                        "hospital sample size more often produces the specified extreme. "
                        f"Compare Hospital A n={option_a['sample_size']} vs Hospital B "
                        f"n={option_b['sample_size']} "
                        f"for event {direction_text} {self._format_number(threshold)}."
                    ),
                ]
            template_idx = self._select_template_variant(
                task_subtype="sample_size_neglect",
                frame_variant=frame_variant,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        elif tier == "neutral_natural":
            if frame_variant == "fund_returns":
                templates = [
                    (
                        "Two funds share the same long-run up-share "
                        f"({self._format_number(baseline_rate)}). For the event "
                        f"{direction_text} {self._format_number(threshold)} up-share in one day, "
                        f"which fund is likelier: A with {option_a['sample_size']} stocks or "
                        f"B with {option_b['sample_size']}?"
                    ),
                    (
                        "Daily breadth note: both funds run at the same background up-share "
                        f"({self._format_number(baseline_rate)}). Which fund is more likely to "
                        f"show {direction_text} {self._format_number(threshold)} up-share "
                        "in a day? "
                        f"A={option_a['sample_size']} stocks, B={option_b['sample_size']}."
                    ),
                    (
                        f"Observed-day framing: suppose you check one day for a move of "
                        f"{direction_text} {self._format_number(threshold)} up-share. "
                        "Both funds have the same long-run up-share "
                        f"({self._format_number(baseline_rate)}). Which fund would hit that day "
                        "more often: "
                        f"A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                    (
                        "Manager review: baseline up-share is the same "
                        f"({self._format_number(baseline_rate)}). Rank fund A "
                        f"(n={option_a['sample_size']}) and fund B "
                        f"(n={option_b['sample_size']}) by likelihood of "
                        f"{direction_text} {self._format_number(threshold)} up-share in one day."
                    ),
                    (
                        f"Likelihood ranking task for funds: event is {direction_text} "
                        f"{self._format_number(threshold)} up-share, baseline is "
                        f"{self._format_number(baseline_rate)}, sample sizes are "
                        f"A={option_a['sample_size']} and B={option_b['sample_size']}. "
                        "Which fund ranks higher?"
                    ),
                ]
            elif frame_variant == "quality_control_batches":
                templates = [
                    (
                        "Two factories share the same long-run pass rate "
                        f"({self._format_number(baseline_rate)}). For a batch with "
                        f"{direction_text} {self._format_number(threshold)} passing share, "
                        f"which line is likelier: A with {option_a['sample_size']} inspected items "
                        f"or B with {option_b['sample_size']}?"
                    ),
                    (
                        "QC report view: both lines have baseline pass share "
                        f"{self._format_number(baseline_rate)}. Which line is more likely to post "
                        f"a batch at {direction_text} {self._format_number(threshold)} "
                        "passing share? "
                        f"A={option_a['sample_size']}, B={option_b['sample_size']}."
                    ),
                    (
                        f"Observed-batch framing: for one checked batch, event is "
                        f"{direction_text} {self._format_number(threshold)} passing share. "
                        "Lines A and B have the same long-run pass rate "
                        f"({self._format_number(baseline_rate)}). Which line would show this event "
                        "more often, "
                        f"A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                    (
                        "Production manager review: with shared pass-rate baseline "
                        f"{self._format_number(baseline_rate)}, compare line A "
                        f"(n={option_a['sample_size']}) vs line B "
                        f"(n={option_b['sample_size']}) for likelihood of "
                        f"{direction_text} {self._format_number(threshold)} passing share."
                    ),
                    (
                        f"Likelihood ranking for QC lines: event={direction_text} "
                        f"{self._format_number(threshold)} passing share, baseline="
                        f"{self._format_number(baseline_rate)}, A={option_a['sample_size']}, "
                        f"B={option_b['sample_size']}. Which line ranks higher?"
                    ),
                ]
            else:
                templates = [
                    (
                        "Two hospitals have the same long-run girl-share "
                        f"({self._format_number(baseline_rate)}). For a day with "
                        f"{direction_text} {self._format_number(threshold)} girls, which hospital "
                        f"is likelier: A with {option_a['sample_size']} births/day or "
                        f"B with {option_b['sample_size']}?"
                    ),
                    (
                        "Daily births report: both hospitals share baseline girl-share "
                        f"{self._format_number(baseline_rate)}. Which hospital is more likely to "
                        f"record {direction_text} {self._format_number(threshold)} girls in a day? "
                        f"A={option_a['sample_size']} births, B={option_b['sample_size']}."
                    ),
                    (
                        f"Observed-day framing: check one day for event "
                        f"{direction_text} {self._format_number(threshold)} girl-share. "
                        "Both hospitals have the same long-run share "
                        f"({self._format_number(baseline_rate)}). Which one would show this event "
                        "more often: "
                        f"A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                    (
                        "Hospital manager review: with shared girl-share baseline "
                        f"{self._format_number(baseline_rate)}, compare Hospital A "
                        f"(n={option_a['sample_size']}) and Hospital B "
                        f"(n={option_b['sample_size']}) for likelihood of "
                        f"{direction_text} {self._format_number(threshold)} girls."
                    ),
                    (
                        f"Likelihood ranking for hospitals: event is {direction_text} "
                        f"{self._format_number(threshold)} girls, baseline is "
                        f"{self._format_number(baseline_rate)}, A={option_a['sample_size']}, "
                        f"B={option_b['sample_size']}. Which hospital ranks higher?"
                    ),
                ]
            template_idx = self._select_template_variant(
                task_subtype="sample_size_neglect",
                frame_variant=frame_variant,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        else:
            if frame_variant == "fund_returns":
                templates = [
                    (
                        "Two portfolio managers report daily breadth with the same long-run "
                        f"up-share ({self._format_number(baseline_rate)}). Which manager would "
                        f"you expect to show {direction_text} "
                        f"{self._format_number(threshold)} up-share more often? "
                        f"A tracks {option_a['sample_size']} stocks; "
                        f"B tracks {option_b['sample_size']}."
                    ),
                    (
                        f"Tape impression: both managers run near "
                        f"{self._format_number(baseline_rate)} up-share over time. "
                        f"Which desk looks more prone to a day at {direction_text} "
                        f"{self._format_number(threshold)} up-share, A "
                        f"(n={option_a['sample_size']}) or B (n={option_b['sample_size']})?"
                    ),
                    (
                        "Quick manager review: same long-run up-share "
                        f"({self._format_number(baseline_rate)}), but different basket sizes. "
                        f"Which one feels more likely to print {direction_text} "
                        f"{self._format_number(threshold)} up-share in one day? "
                        f"A={option_a['sample_size']}, B={option_b['sample_size']}."
                    ),
                    (
                        "Observed-day instinct: if you only saw one day, which manager would "
                        f"you pick as more likely to hit {direction_text} "
                        f"{self._format_number(threshold)} up-share? "
                        f"A uses {option_a['sample_size']} names, B uses {option_b['sample_size']}."
                    ),
                    (
                        f"Likelihood ranking under pressure: baseline="
                        f"{self._format_number(baseline_rate)}, event={direction_text} "
                        f"{self._format_number(threshold)} up-share. Which manager gets your first "
                        f"pick, A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                ]
            elif frame_variant == "quality_control_batches":
                templates = [
                    (
                        "Two production lines share the same long-run pass share "
                        f"({self._format_number(baseline_rate)}). Which line seems more likely "
                        f"to show a batch at {direction_text} "
                        f"{self._format_number(threshold)} passing share? "
                        f"Line A inspects {option_a['sample_size']} items; "
                        f"line B inspects {option_b['sample_size']}."
                    ),
                    (
                        f"Floor impression: both lines run near "
                        f"{self._format_number(baseline_rate)} pass share overall. "
                        f"Which line looks more prone to a batch with {direction_text} "
                        f"{self._format_number(threshold)} passing share, A "
                        f"(n={option_a['sample_size']}) or B (n={option_b['sample_size']})?"
                    ),
                    (
                        "Supervisor review: same baseline pass share "
                        f"({self._format_number(baseline_rate)}), different inspected counts. "
                        f"Which line feels more likely to post {direction_text} "
                        f"{self._format_number(threshold)} passing share in one batch? "
                        f"A={option_a['sample_size']}, B={option_b['sample_size']}."
                    ),
                    (
                        "Observed-batch instinct: on a single batch read, which line would you "
                        f"tag as more likely to hit {direction_text} "
                        f"{self._format_number(threshold)} passing share? "
                        f"A uses {option_a['sample_size']} checks, B uses "
                        f"{option_b['sample_size']}."
                    ),
                    (
                        f"Likelihood ranking under pressure: baseline="
                        f"{self._format_number(baseline_rate)}, event={direction_text} "
                        f"{self._format_number(threshold)} passing share. Which line gets your "
                        "first pick, "
                        f"A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                ]
            else:
                templates = [
                    (
                        "Two city hospitals report daily births with the same long-run "
                        f"girl-share ({self._format_number(baseline_rate)}). Which hospital "
                        f"seems more likely to post {direction_text} "
                        f"{self._format_number(threshold)} girls in a day? "
                        f"Hospital A: {option_a['sample_size']} births/day; "
                        f"Hospital B: {option_b['sample_size']} births/day."
                    ),
                    (
                        f"Ward summary view: both hospitals run near "
                        f"{self._format_number(baseline_rate)} girl-share over time. "
                        f"Which hospital looks more prone to a day with {direction_text} "
                        f"{self._format_number(threshold)} girls, A "
                        f"(n={option_a['sample_size']}) or B (n={option_b['sample_size']})?"
                    ),
                    (
                        "Administrator review: same baseline girl-share "
                        f"({self._format_number(baseline_rate)}), different births/day. "
                        f"Which hospital feels more likely to log {direction_text} "
                        f"{self._format_number(threshold)} girls in one day? "
                        f"A={option_a['sample_size']}, B={option_b['sample_size']}."
                    ),
                    (
                        "Observed-day instinct: if you checked one day only, which hospital "
                        f"would you expect to hit {direction_text} "
                        f"{self._format_number(threshold)} girls? "
                        f"A has {option_a['sample_size']} births/day, "
                        f"B has {option_b['sample_size']}."
                    ),
                    (
                        f"Likelihood ranking under pressure: baseline="
                        f"{self._format_number(baseline_rate)}, event={direction_text} "
                        f"{self._format_number(threshold)} girls. Which hospital gets your first "
                        f"pick, A ({option_a['sample_size']}) or B ({option_b['sample_size']})?"
                    ),
                ]
            template_idx = self._select_template_variant(
                task_subtype="sample_size_neglect",
                frame_variant=frame_variant,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]

        if style == "unlabeled":
            return body
        return f"{body}\n- choose_A\n- choose_B"

    # Style goal: frame interval choices as practical forecast judgments.
    # Non-formal variants intentionally use everyday forecasting language;
    # formal style carries explicit distribution/coverage language.
    def _render_overprecision_calibration_prompt(
        self,
        *,
        problem_spec: OverprecisionCalibrationProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        mu = assumptions["true_value_mean"]
        sd = assumptions["true_value_sd"]
        frame_variant = prompt_frame_variant or "analyst_forecast"
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        interval_a = (
            f"[{self._format_number(option_a['lower'])}, {self._format_number(option_a['upper'])}]"
        )
        interval_b = (
            f"[{self._format_number(option_b['lower'])}, {self._format_number(option_b['upper'])}]"
        )
        # Template banks intentionally span committee/dashboard/memo debate
        # structures so interval prompts differ in discourse form, not nouns.
        if tier == "formal":
            if frame_variant == "weather_forecast":
                templates = [
                    (
                        f"Forecast committee review (weather): "
                        f"model center={self._format_number(mu)}, "
                        f"error scale={self._format_number(sd)}. Team A={interval_a}, "
                        f"Team B={interval_b}. Which interval is more likely to contain "
                        "the realized weather value?"
                    ),
                    (
                        f"Weather dashboard note: baseline level {self._format_number(mu)}, "
                        f"typical error {self._format_number(sd)}. Intervals are A={interval_a} "
                        f"and B={interval_b}. Select the interval with higher containment chance."
                    ),
                    (
                        f"Comparison sheet (weather): center={self._format_number(mu)}, "
                        f"error={self._format_number(sd)}; A={interval_a}, B={interval_b}. "
                        "Which interval has higher containment probability?"
                    ),
                    (
                        f"Forecast audit line: weather level around {self._format_number(mu)} with "
                        f"error scale {self._format_number(sd)}. Compare A={interval_a} and "
                        f"B={interval_b}; choose the more likely interval to contain the outcome."
                    ),
                    (
                        f"Use center {self._format_number(mu)} and error scale "
                        f"{self._format_number(sd)} to compare A={interval_a} and "
                        f"B={interval_b}. Which interval has the higher containment probability?"
                    ),
                    (
                        f"With mean {self._format_number(mu)} and sd "
                        f"{self._format_number(sd)}, evaluate interval A={interval_a} against "
                        f"B={interval_b}. Which one is more likely to contain the realized "
                        "weather value?"
                    ),
                    (
                        f"Evaluate interval choice: weather value is modeled around "
                        f"{self._format_number(mu)} with spread {self._format_number(sd)}. "
                        f"Choose between A={interval_a} and B={interval_b} by larger "
                        "coverage probability."
                    ),
                ]
            elif frame_variant == "startup_projection":
                templates = [
                    (
                        f"Startup projection review: KPI mean reference={self._format_number(mu)}, "
                        f"error scale={self._format_number(sd)}; A={interval_a}, B={interval_b}. "
                        "Which interval is more likely to contain realized KPI?"
                    ),
                    (
                        f"Centered at {self._format_number(mu)} with "
                        f"error scale {self._format_number(sd)}. Candidate intervals are "
                        f"A={interval_a}, "
                        f"B={interval_b}. Choose the higher-coverage interval."
                    ),
                    (
                        f"Projection comparison sheet: KPI center={self._format_number(mu)}, "
                        f"error scale={self._format_number(sd)}. Intervals A={interval_a} and "
                        f"B={interval_b}; select the interval with higher containment probability."
                    ),
                    (
                        f"Analytic startup brief: expected level {self._format_number(mu)}, "
                        f"error {self._format_number(sd)}. Candidate intervals are {interval_a} "
                        f"and {interval_b}. Which interval is more likely to include realized KPI?"
                    ),
                    (
                        f"Using center {self._format_number(mu)} and error scale "
                        f"{self._format_number(sd)}, compare A={interval_a} with "
                        f"B={interval_b}. Which interval is more likely to contain "
                        "the realized KPI?"
                    ),
                    (
                        f"Given mean {self._format_number(mu)} and sd "
                        f"{self._format_number(sd)}, evaluate whether A={interval_a} or "
                        f"B={interval_b} has the larger chance to contain realized KPI."
                    ),
                    (
                        f"Determine the higher-probability interval under the stated model: "
                        f"KPI center {self._format_number(mu)}, spread "
                        f"{self._format_number(sd)}, ranges A={interval_a}, B={interval_b}."
                    ),
                ]
            else:
                templates = [
                    (
                        f"Analyst disagreement brief: center={self._format_number(mu)}, "
                        f"error scale={self._format_number(sd)}. Analyst A={interval_a}; "
                        f"Analyst B={interval_b}. Which interval is more likely to include outcome?"
                    ),
                    (
                        f"Forecast comparison sheet: expected level {self._format_number(mu)}, "
                        f"typical error {self._format_number(sd)}; intervals A={interval_a}, "
                        f"B={interval_b}. Select the interval with higher containment probability."
                    ),
                    (
                        f"Analyst review form: baseline={self._format_number(mu)}, "
                        f"error scale={self._format_number(sd)}, A={interval_a}, B={interval_b}. "
                        "Choose the interval with higher containment probability."
                    ),
                    (
                        f"Model check card: center {self._format_number(mu)}, error "
                        f"{self._format_number(sd)}. Candidate intervals: A={interval_a}, "
                        f"B={interval_b}. Which interval is more likely to include the outcome?"
                    ),
                    (
                        f"Expected level is {self._format_number(mu)} with error scale "
                        f"{self._format_number(sd)}. Compare intervals A={interval_a} and "
                        f"B={interval_b}; select the one with higher containment probability."
                    ),
                    (
                        f"With mean {self._format_number(mu)} and sd "
                        f"{self._format_number(sd)}, determine whether A={interval_a} or "
                        f"B={interval_b} has higher containment probability."
                    ),
                    (
                        f"Evaluate interval choice analytically: use the stated distribution "
                        f"(center {self._format_number(mu)}, scale {self._format_number(sd)}) "
                        f"to compare A={interval_a} and B={interval_b}, then choose the higher "
                        "coverage interval."
                    ),
                ]
        elif tier == "neutral_natural":
            if frame_variant == "weather_forecast":
                templates = [
                    (
                        "Tomorrow's weather plan uses two forecast "
                        f"ranges: A={interval_a} "
                        f"and B={interval_b}. Past realizations sit near "
                        f"{self._format_number(mu)} with typical miss "
                        f"{self._format_number(sd)}. Which range gives better odds of "
                        "covering the realized value?"
                    ),
                    (
                        "For tomorrow's weather call, which range is better for covering "
                        "the realized value: "
                        f"A={interval_a} or B={interval_b}? "
                        f"The variable usually centers near {self._format_number(mu)} with miss "
                        f"around {self._format_number(sd)}."
                    ),
                    (
                        f"Weather outcomes cluster around "
                        f"{self._format_number(mu)} with typical miss "
                        f"{self._format_number(sd)}. Compare A={interval_a} with "
                        f"B={interval_b}. Which range is likelier to include the outcome?"
                    ),
                    (
                        f"A weather desk review sees two competing calls, A={interval_a} and "
                        f"B={interval_b}. Typical level is {self._format_number(mu)} and "
                        f"typical miss is {self._format_number(sd)}. Which range is more likely "
                        "to include the realized weather value?"
                    ),
                    (
                        f"Case summary: A={interval_a}; B={interval_b}; center "
                        f"{self._format_number(mu)}; miss {self._format_number(sd)}. "
                        "For planning coverage, which range is likelier to cover "
                        "the realized value?"
                    ),
                    (
                        f"A={interval_a}, B={interval_b}. "
                        f"Reference level is {self._format_number(mu)} with typical error "
                        f"{self._format_number(sd)}. Which interval gives better odds of "
                        "containing the realized weather value?"
                    ),
                ]
            elif frame_variant == "startup_projection":
                templates = [
                    (
                        "A startup planning pass has two KPI ranges on the table: "
                        f"A={interval_a} and "
                        f"B={interval_b}. Past realizations are around "
                        f"{self._format_number(mu)} with miss about "
                        f"{self._format_number(sd)}. Which range gives better coverage for KPI?"
                    ),
                    (
                        "For this startup planning choice, which range is better for containing "
                        "realized KPI: "
                        f"A={interval_a} or "
                        f"B={interval_b}? Typical level is {self._format_number(mu)} and "
                        f"typical miss is {self._format_number(sd)}."
                    ),
                    (
                        f"History first: KPI outcomes center near {self._format_number(mu)} with "
                        f"error around {self._format_number(sd)}. Candidate ranges are "
                        f"A={interval_a} and B={interval_b}. Which is more likely to include "
                        "the realized result?"
                    ),
                    (
                        f"Planning review has to choose between A={interval_a} and B={interval_b}. "
                        f"Typical KPI level is {self._format_number(mu)} with miss around "
                        f"{self._format_number(sd)}. Which range is more likely to include "
                        "the realized KPI?"
                    ),
                    (
                        f"Case summary: A={interval_a}; B={interval_b}; historical center "
                        f"{self._format_number(mu)}; miss {self._format_number(sd)}. "
                        "For planning coverage, which range is likelier to cover the realized KPI?"
                    ),
                    (
                        f"A={interval_a} and B={interval_b}. "
                        f"Reference level is {self._format_number(mu)} with error about "
                        f"{self._format_number(sd)}. Which range has better odds of containing "
                        "the realized KPI?"
                    ),
                ]
            else:
                templates = [
                    (
                        "Two forecasts are on the desk, "
                        f"A={interval_a} and B={interval_b}. "
                        f"Past outcomes cluster near {self._format_number(mu)} with miss about "
                        f"{self._format_number(sd)}. Which interval is likelier to contain value?"
                    ),
                    (
                        "For this planning decision, which option is better at containing the "
                        f"outcome, A={interval_a} or "
                        f"B={interval_b}? Center is {self._format_number(mu)} and "
                        f"typical error is {self._format_number(sd)}."
                    ),
                    (
                        f"History first: typical level is {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. The candidate ranges are A={interval_a} and "
                        f"B={interval_b}. Which interval is likelier to include value?"
                    ),
                    (
                        f"A team is deciding now between interval A={interval_a} "
                        f"and B={interval_b}. "
                        f"Usual level sits near {self._format_number(mu)} with miss around "
                        f"{self._format_number(sd)}. Which option is better for "
                        "covering the realized value?"
                    ),
                    (
                        f"Baseline {self._format_number(mu)}, error "
                        f"{self._format_number(sd)}, ranges A={interval_a} and B={interval_b}. "
                        "Which interval seems likelier to cover the outcome?"
                    ),
                    (
                        f"A={interval_a}, B={interval_b}. "
                        f"Reference level {self._format_number(mu)} with typical miss "
                        f"{self._format_number(sd)}. Which interval is more likely to include "
                        "the realized value?"
                    ),
                ]
        else:
            if frame_variant == "weather_forecast":
                templates = [
                    (
                        f"The weather board shows range A={interval_a} and range B={interval_b}. "
                        f"The variable usually tracks near {self._format_number(mu)} with miss "
                        f"around {self._format_number(sd)}. Which range would you trust first "
                        "to catch the realized value?"
                    ),
                    (
                        f"Competing weather calls are up, A={interval_a} versus B={interval_b}. "
                        f"Typical level is {self._format_number(mu)}, miss size "
                        f"{self._format_number(sd)}. Which range feels safer to stand behind "
                        "right now?"
                    ),
                    (
                        f"A confidence debate is live: Team A gives {interval_a}, Team B gives "
                        f"{interval_b}. Given the usual level {self._format_number(mu)} and miss "
                        f"{self._format_number(sd)}, which range would you back under time "
                        "pressure if you want coverage?"
                    ),
                    (
                        f"Live dashboard readout: A={interval_a}, B={interval_b}. "
                        f"Typical weather level is {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. Which range is more likely to contain "
                        "the realized weather value?"
                    ),
                    (
                        f"Posted weather ranges are {interval_a} and "
                        f"{interval_b}. Usual level {self._format_number(mu)}, miss "
                        f"{self._format_number(sd)}. Quick call: which range do you trust to "
                        "include the realized value?"
                    ),
                    (
                        f"Forecast confidence call: weather options are A={interval_a} and "
                        f"B={interval_b}. Typical level {self._format_number(mu)}, miss "
                        f"{self._format_number(sd)}. Which range is more likely to include "
                        "the realized weather value?"
                    ),
                ]
            elif frame_variant == "startup_projection":
                templates = [
                    (
                        f"Investor call asks for a quick read: KPI range A={interval_a}, range "
                        f"B={interval_b}. "
                        f"Past levels center near {self._format_number(mu)} with miss around "
                        f"{self._format_number(sd)}. Which range would you trust more to contain "
                        "the realized KPI?"
                    ),
                    (
                        f"Boardroom confidence check: A={interval_a} and B={interval_b}. "
                        f"Typical KPI level is {self._format_number(mu)} with error "
                        f"{self._format_number(sd)}. "
                        "Which range sounds more convincing to cover the realized result?"
                    ),
                    (
                        f"Two KPI windows are on screen, A={interval_a} "
                        f"and B={interval_b}. Usual center is {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. Which window do you trust first to include "
                        "the realized KPI?"
                    ),
                    (
                        f"Investor update note: KPI ranges are A={interval_a} versus "
                        f"B={interval_b}. "
                        f"Typical level is {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. Confidence call: which interval "
                        "is more likely "
                        "to contain the realized KPI?"
                    ),
                    (
                        f"Two startup windows are listed, {interval_a} and "
                        f"{interval_b}. Center tends to {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. Under pressure, which window seems more "
                        "credible to include the realized KPI?"
                    ),
                    (
                        f"Pitch-ready summary: A={interval_a}, B={interval_b}. "
                        f"KPI usually sits near {self._format_number(mu)} with miss "
                        f"{self._format_number(sd)}. Which range is more likely to contain "
                        "the realized KPI?"
                    ),
                ]
            else:
                templates = [
                    (
                        f"A lists {interval_a}, B lists {interval_b}. "
                        f"With outcomes usually near {self._format_number(mu)} and miss "
                        f"{self._format_number(sd)}, which range would you trust first to "
                        "contain the realized value?"
                    ),
                    (
                        f"Forecast confidence check: A={interval_a}, B={interval_b}. "
                        f"Typical level {self._format_number(mu)}, miss scale "
                        f"{self._format_number(sd)}. Which range sounds more convincing to "
                        "include the realized value?"
                    ),
                    (
                        f"Dashboard split-view: interval A {interval_a} vs "
                        f"interval B {interval_b}. "
                        f"Historical center is {self._format_number(mu)} with miss around "
                        f"{self._format_number(sd)}. Which range seems likelier to contain value?"
                    ),
                    (
                        f"Quick analyst pulse: A={interval_a}, B={interval_b}. "
                        f"Typical level {self._format_number(mu)}, miss scale "
                        f"{self._format_number(sd)}. Confidence read: which interval would you "
                        "back to contain the realized value?"
                    ),
                    (
                        f"Comparison dashboard: ranges are {interval_a} and {interval_b}; "
                        f"center {self._format_number(mu)}, miss {self._format_number(sd)}. "
                        "Which range is more likely to include the realized value?"
                    ),
                    (
                        f"First-impression summary: intervals A={interval_a} and B={interval_b}; "
                        f"typical level {self._format_number(mu)}, miss "
                        f"{self._format_number(sd)}. Which range feels most trustworthy for "
                        "containing the realized value?"
                    ),
                ]

        template_idx = self._select_template_variant(
            task_subtype="overprecision_calibration",
            frame_variant=frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            templates=templates,
        )
        body = templates[template_idx]
        body = self._enrich_overprecision_frame_realism(
            prompt=body,
            frame_variant=frame_variant,
            mu_text=self._format_number(mu),
            sd_text=self._format_number(sd),
        )
        body = self._compact_overprecision_opening(
            prompt=body,
            frame_variant=frame_variant,
            tier=tier,
        )
        body = self._ensure_overprecision_containment_wording(body)

        if style == "unlabeled":
            return body
        return f"{body}\n- choose_A\n- choose_B"

    def _ensure_overprecision_containment_wording(self, prompt: str) -> str:
        """Ensure overprecision prompts explicitly ask containment probability."""
        lower = prompt.lower()
        containment_markers = ("contain", "include", "cover", "coverage", "catch")
        if any(marker in lower for marker in containment_markers):
            return prompt
        rewrites = (
            (
                r"Which range gets your first call\?\s*$",
                "On first call, which range is more likely to contain the realized value?",
            ),
            (
                r"Gut-check: which range do you pick\?\s*$",
                "Gut-check: which range is more likely to contain the realized value?",
            ),
            (
                r"Which range do you choose\?\s*$",
                "Which range is more likely to contain the realized value?",
            ),
            (
                r"Which range would you back\?\s*$",
                "Which range is more likely to contain the realized value?",
            ),
        )
        for pattern, replacement in rewrites:
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                return re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        trimmed = prompt.rstrip()
        if not trimmed.endswith((".", "?", "!")):
            trimmed = f"{trimmed}."
        return f"{trimmed} Which interval is more likely to contain the realized value?"

    def _enrich_overprecision_frame_realism(
        self,
        *,
        prompt: str,
        frame_variant: PromptFrameVariant,
        mu_text: str,
        sd_text: str,
    ) -> str:
        _ = mu_text
        _ = sd_text
        lower = prompt.lower()
        if frame_variant == "weather_forecast":
            markers = ("weather", "forecast", "temperature", "day")
            anchor = "Weather forecast:"
        elif frame_variant == "startup_projection":
            markers = ("startup", "kpi", "projection", "planning", "board")
            anchor = "Startup KPI projection:"
        else:
            markers = ("analyst", "projection", "desk", "range", "note")
            anchor = "Analyst projection note:"
        if any(marker in lower for marker in markers):
            return prompt
        return f"{anchor} {prompt}"

    def _compact_overprecision_opening(
        self,
        *,
        prompt: str,
        frame_variant: PromptFrameVariant,
        tier: str = "",
    ) -> str:
        """Reduce overprecision prompt preamble bloat to one short frame anchor."""
        collapsed = " ".join(prompt.split())
        noisy_prefixes = (
            "forecast committee review",
            "weather dashboard note",
            "comparison sheet",
            "forecast audit line",
            "coverage ranking prompt",
            "startup projection review",
            "planning memo",
            "projection comparison sheet",
            "analytic startup brief",
            "analyst disagreement brief",
            "forecast comparison sheet",
            "analyst review form",
            "model check card",
            "coverage ranking worksheet",
            "case review",
            "case summary",
            "history first",
            "range-first view",
            "range-first read",
            "alert desk summary",
            "forecast confidence debate",
            "live dashboard readout",
            "operator pulse note",
            "investor call snapshot",
            "boardroom projection debate",
            "launch planning handoff",
            "investor update note",
            "projection pulse",
            "quick triage note",
            "forecast confidence check",
            "dashboard split-view",
            "quick analyst pulse",
            "comparison dashboard",
            "first-impression summary",
        )
        for prefix in noisy_prefixes:
            collapsed = re.sub(
                rf"^\s*{re.escape(prefix)}\s*:\s*",
                "",
                collapsed,
                flags=re.IGNORECASE,
            )

        # Remove fused meta-instruction labels so only one shell remains.
        instruction_prefixes = (
            "compute then compare",
            "coverage ranking prompt",
            "coverage ranking task",
            "evaluate interval choice analytically",
            "determine the higher-probability interval under the stated model",
            "in this weather case, which option is better for containing the realized value",
            "in this startup case, which option is better for containing realized kpi",
            "in this review, which option is better at containing the outcome",
        )
        for prefix in instruction_prefixes:
            collapsed = re.sub(
                rf"^\s*{re.escape(prefix)}\s*:\s*",
                "",
                collapsed,
                flags=re.IGNORECASE,
            )

        opening_options_by_frame = {
            "analyst_forecast": (
                "Analyst desk view. {body}",
                "Desk forecasts diverge today. {body}",
                "{body}",
            ),
            "weather_forecast": (
                "Weather desk view. {body}",
                "Tomorrow's forecast has two ranges. {body}",
                "{body}",
            ),
            "startup_projection": (
                "Startup planning view. {body}",
                "The board deck lists two KPI ranges. {body}",
                "{body}",
            ),
        }
        opening_options = opening_options_by_frame.get(
            frame_variant,
            (
                "Forecast view. {body}",
                "Two ranges are on the table. {body}",
                "{body}",
            ),
        )
        # Remove any previously attached anchor variants first.
        collapsed = re.sub(
            (
                r"^\s*(?:analyst projection note|weather forecast|"
                r"startup kpi projection|forecast note)\s*:\s*"
            ),
            "",
            collapsed,
            flags=re.IGNORECASE,
        )
        generic_header_pattern = re.compile(
            r"^\s*([A-Za-z][A-Za-z ()/\-]{0,80})\s*:\s*",
            flags=re.IGNORECASE,
        )
        header_match = generic_header_pattern.match(collapsed)
        if header_match:
            header_text = header_match.group(1).lower()
            meta_tokens = (
                "review",
                "note",
                "sheet",
                "card",
                "summary",
                "brief",
                "pulse",
                "debate",
                "check",
                "handoff",
                "snapshot",
                "comparison",
                "dashboard",
                "committee",
                "audit",
                "compute",
                "evaluate",
                "rank",
                "task",
                "prompt",
                "framing",
            )
            if any(token in header_text for token in meta_tokens):
                collapsed = collapsed[header_match.end() :].lstrip()
        body = collapsed.strip()
        # Deterministic opener diversification by prompt content and frame.
        bucket = int(
            hashlib.sha256(f"{frame_variant}|{tier}|{body}".encode("utf-8")).hexdigest(),
            16,
        ) % len(opening_options)
        opener = opening_options[bucket]
        return opener.format(body=body)

    def _qa_validate_rendered_prompt(
        self,
        *,
        task_subtype: TaskSubtype,
        prompt: str,
        problem_spec: ProblemSpec,
        frame_variant: PromptFrameVariant,
        prompt_style_regime: PromptStyleRegime | None = None,
    ) -> list[dict[str, str]]:
        failures = self._prompt_qa_generic_failures(prompt=prompt)
        lower = prompt.lower()
        active_regime = prompt_style_regime or self._resolve_prompt_style_regime()
        if task_subtype == "conjunction_fallacy":
            option_a = problem_spec["options"]["A"]["event_label"].lower()
            option_b = problem_spec["options"]["B"]["event_label"].lower()
            if option_a not in lower or option_b not in lower:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_conjunction_event_statement",
                        detail=(
                            "Conjunction prompt must include both event statements from "
                            "problem_spec options."
                        ),
                    )
                )
            if active_regime in {"neutral_realistic", "bias_eliciting"}:
                leaked_rule_tokens = (
                    "conjunction axiom",
                    "probability axiom",
                    "probability logic",
                    "constituent",
                    "conjunction event",
                )
                if any(token in lower for token in leaked_rule_tokens):
                    failures.append(
                        self._prompt_qa_failure(
                            code="conjunction_rule_leakage",
                            detail=(
                                "Non-normative conjunction prompt contains explicit "
                                "rule/axiom terminology."
                            ),
                        )
                    )
            if active_regime == "neutral_realistic":
                neutral_markers = (
                    "case",
                    "summary",
                    "record",
                    "details",
                    "profile",
                    "facts",
                    "comparison",
                    "snapshot",
                )
                if not any(marker in lower for marker in neutral_markers):
                    failures.append(
                        self._prompt_qa_failure(
                            code="conjunction_neutral_style_under_specified",
                            detail=(
                                "Neutral conjunction prompt should read as plain case "
                                "interpretation with observational framing."
                            ),
                        )
                    )
            if active_regime == "bias_eliciting":
                bias_markers = (
                    "first glance",
                    "first-glance",
                    "first impression",
                    "first-pass",
                    "instant read",
                    "gut-check",
                    "feels more fitting",
                    "better fit",
                    "representative",
                )
                if not any(marker in lower for marker in bias_markers):
                    failures.append(
                        self._prompt_qa_failure(
                            code="conjunction_bias_style_under_specified",
                            detail=(
                                "Bias-eliciting conjunction prompt should include "
                                "impression/type-intuition markers."
                            ),
                        )
                    )
        if task_subtype == "gambler_fallacy":
            required_terms_by_frame = {
                "sports_streak": ("make", "miss"),
                "market_streak": ("up", "down"),
                "roulette_streak": ("red", "black"),
            }
            required_terms = required_terms_by_frame.get(frame_variant, tuple())
            if required_terms and not all(term in lower for term in required_terms):
                failures.append(
                    self._prompt_qa_failure(
                        code="non_native_gambler_choice_text",
                        detail=(
                            f"Gambler prompt for frame '{frame_variant}' is missing required "
                            f"context-native terms {required_terms}."
                        ),
                    )
                )
            if frame_variant != "neutral_coin":
                forbidden_coin_phrases = (
                    "heads on the very next flip",
                    "tails on the very next flip",
                )
                if any(phrase in lower for phrase in forbidden_coin_phrases):
                    failures.append(
                        self._prompt_qa_failure(
                            code="non_native_coin_phrase_leak",
                            detail=(
                                "Non-coin gambler frame contains literal coin-flip answer phrase."
                            ),
                        )
                    )
            option_lines = [
                line.strip().lower()
                for line in prompt.splitlines()
                if line.strip().lower().startswith("- choose_a:")
                or line.strip().lower().startswith("- choose_b:")
            ]
            if len(option_lines) == 2 and option_lines[0] == option_lines[1]:
                failures.append(
                    self._prompt_qa_failure(
                        code="gambler_repetitive_option_text",
                        detail="Gambler options are text-identical; require explicit contrast.",
                    )
                )
        if task_subtype == "overprecision_calibration":
            assumptions = problem_spec["assumptions"]
            option_a = problem_spec["options"]["A"]
            option_b = problem_spec["options"]["B"]
            center_text = self._format_number(float(assumptions["true_value_mean"]))
            error_scale_text = self._format_number(float(assumptions["true_value_sd"]))
            interval_a = (
                f"[{self._format_number(option_a['lower'])}, "
                f"{self._format_number(option_a['upper'])}]"
            ).lower()
            interval_b = (
                f"[{self._format_number(option_b['lower'])}, "
                f"{self._format_number(option_b['upper'])}]"
            ).lower()
            if center_text not in prompt:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_overprecision_center",
                        detail=(
                            "Overprecision prompt missing stated center/baseline "
                            f"value '{center_text}'."
                        ),
                    )
                )
            if error_scale_text not in prompt:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_overprecision_error_scale",
                        detail=(
                            "Overprecision prompt missing stated error scale "
                            f"value '{error_scale_text}'."
                        ),
                    )
                )
            if interval_a not in lower or interval_b not in lower:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_overprecision_intervals",
                        detail="Overprecision prompt must include both interval strings.",
                    )
                )
            prompt_body = lower.split("\n- choose_a")[0].split("\n- choose_b")[0].strip()
            containment_markers = ("contain", "include", "cover", "coverage", "catch")
            has_containment_wording = any(marker in prompt_body for marker in containment_markers)
            if not has_containment_wording:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_overprecision_containment_wording",
                        detail=(
                            "Overprecision prompt must explicitly ask about "
                            "containment/inclusion/coverage probability."
                        ),
                    )
                )
            ambiguous_endings = (
                "which do you choose?",
                "which would you pick?",
                "which do you mark first?",
            )
            if (
                any(prompt_body.endswith(ending) for ending in ambiguous_endings)
                and not has_containment_wording
            ):
                failures.append(
                    self._prompt_qa_failure(
                        code="overprecision_ambiguous_choice_wording",
                        detail=(
                            "Overprecision prompt ends with vague choice wording "
                            "without explicit containment/inclusion/coverage language."
                        ),
                    )
                )
            preference_markers = (
                "feels safer",
                "would you choose",
                "would you pick",
                "do you mark first",
                "first pick",
            )
            if (
                any(marker in prompt_body for marker in preference_markers)
                and not has_containment_wording
            ):
                failures.append(
                    self._prompt_qa_failure(
                        code="overprecision_preference_only_wording",
                        detail=(
                            "Overprecision prompt uses preference-only wording "
                            "without explicit containment language."
                        ),
                    )
                )
        return failures

    def _build_base_rate_neglect_outcome_model(
        self,
        *,
        problem_spec: BaseRateNeglectProblemSpec,
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        prior_high = assumptions["prior_high"]
        p_signal_high_given_high = assumptions["p_signal_high_given_high"]
        p_signal_high_given_low = assumptions["p_signal_high_given_low"]
        observed_signal = assumptions["observed_signal"]
        if observed_signal == "high":
            like_high = p_signal_high_given_high
            like_low = p_signal_high_given_low
        else:
            like_high = 1 - p_signal_high_given_high
            like_low = 1 - p_signal_high_given_low
        return {
            "choose_state_high": (
                "posterior_high = "
                f"({self._format_number(prior_high)}*{self._format_number(like_high)})/"
                f"(({self._format_number(prior_high)}*{self._format_number(like_high)}) + "
                f"(({self._format_number(1 - prior_high)}*{self._format_number(like_low)})))"
            ),
            "choose_state_low": "posterior_low = 1 - posterior_high",
        }

    def _build_conjunction_fallacy_outcome_model(
        self,
        *,
        problem_spec: ConjunctionFallacyProblemSpec,
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        return {
            "choose_A": f"P(A_statement) = {self._format_number(option_a['probability'])}",
            "choose_B": f"P(B_statement) = {self._format_number(option_b['probability'])}",
        }

    def _build_gambler_fallacy_outcome_model(
        self,
        *,
        problem_spec: GamblerFallacyProblemSpec,
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        queried_outcome = assumptions["queried_outcome"]
        opposite_outcome = "tails" if queried_outcome == "heads" else "heads"

        def _claim_formula(claim_type: str) -> str:
            if claim_type == "more_likely":
                return (
                    f"I({queried_outcome} more likely than {opposite_outcome} "
                    "under independence) = 0"
                )
            return (
                f"I({queried_outcome} not more likely than {opposite_outcome} "
                "under independence) = 1"
            )

        return {
            "choose_A": _claim_formula(option_a["claim_type"]),
            "choose_B": _claim_formula(option_b["claim_type"]),
        }

    def _build_sample_size_neglect_outcome_model(
        self,
        *,
        problem_spec: SampleSizeNeglectProblemSpec,
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        direction = option_a["extreme_direction"]
        comparator = ">=" if direction == "at_or_above" else "<="
        cutoff_a = self._extreme_count_cutoff(
            sample_size=option_a["sample_size"],
            extreme_threshold=option_a["extreme_threshold"],
            extreme_direction=direction,
        )
        cutoff_b = self._extreme_count_cutoff(
            sample_size=option_b["sample_size"],
            extreme_threshold=option_b["extreme_threshold"],
            extreme_direction=direction,
        )
        return {
            "choose_A": (
                "P(extreme_A) = P(X_A "
                f"{comparator} {cutoff_a}), X_A~Binomial(n_A={option_a['sample_size']}, "
                f"p={self._format_number(option_a['baseline_rate'])})"
            ),
            "choose_B": (
                "P(extreme_B) = P(X_B "
                f"{comparator} {cutoff_b}), X_B~Binomial(n_B={option_b['sample_size']}, "
                f"p={self._format_number(option_b['baseline_rate'])})"
            ),
        }

    def _build_overprecision_calibration_outcome_model(
        self,
        *,
        problem_spec: OverprecisionCalibrationProblemSpec,
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        mu = problem_spec["assumptions"]["true_value_mean"]
        sd = problem_spec["assumptions"]["true_value_sd"]
        return {
            "choose_A": (
                "coverage_A = Phi(("
                f"{self._format_number(option_a['upper'])}-{self._format_number(mu)})/"
                f"{self._format_number(sd)}) - Phi(("
                f"{self._format_number(option_a['lower'])}-{self._format_number(mu)})/"
                f"{self._format_number(sd)})"
            ),
            "choose_B": (
                "coverage_B = Phi(("
                f"{self._format_number(option_b['upper'])}-{self._format_number(mu)})/"
                f"{self._format_number(sd)}) - Phi(("
                f"{self._format_number(option_b['lower'])}-{self._format_number(mu)})/"
                f"{self._format_number(sd)})"
            ),
        }

    def _generate_base_rate_neglect(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        # This subtype is intentionally Bayes-equivalent to signal-update tasks.
        # The benchmark target is base-rate-neglect susceptibility from framing,
        # not a different posterior-update rule.
        regime = self.rng.choice(["choose_state_high", "choose_state_low", "near_indifferent"])
        prior_high = 0.05
        p_signal_high_given_high = 0.9
        p_signal_high_given_low = 0.2
        observed_signal: Literal["high", "low"] = "high"
        posterior_high = self._posterior_from_signal(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
        )
        found_candidate = False
        for _ in range(200):
            candidate_prior_high = round(self.rng.uniform(0.01, 0.45), 2)
            candidate_p_signal_high_given_high = round(self.rng.uniform(0.65, 0.95), 2)
            candidate_p_signal_high_given_low = round(self.rng.uniform(0.05, 0.35), 2)
            candidate_observed_signal: Literal["high", "low"] = self.rng.choice(["high", "low"])
            candidate_posterior_high = self._posterior_from_signal(
                prior_high=candidate_prior_high,
                p_signal_high_given_high=candidate_p_signal_high_given_high,
                p_signal_high_given_low=candidate_p_signal_high_given_low,
                observed_signal=candidate_observed_signal,
            )
            gap = candidate_posterior_high - 0.5
            if regime == "choose_state_high" and gap < 0.08:
                continue
            if regime == "choose_state_low" and gap > -0.08:
                continue
            if regime == "near_indifferent" and abs(gap) > 0.02:
                continue

            prior_high = candidate_prior_high
            p_signal_high_given_high = candidate_p_signal_high_given_high
            p_signal_high_given_low = candidate_p_signal_high_given_low
            observed_signal = candidate_observed_signal
            posterior_high = candidate_posterior_high
            found_candidate = True
            break

        if not found_candidate:
            (
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
                observed_signal,
                posterior_high,
            ) = self._base_rate_neglect_fallback(regime=regime)

        posterior_low = self._round_scalar(1 - posterior_high)
        choose_state_high = self._round_scalar(posterior_high)
        choose_state_low = posterior_low
        optimal = self._choose_optimal_action(
            left_label="choose_state_high",
            left_value=choose_state_high,
            right_label="choose_state_low",
            right_value=choose_state_low,
        )
        decision_values = ActionScalars(
            {
                "choose_state_high": choose_state_high,
                "choose_state_low": choose_state_low,
            }
        )
        comparison_pair = self._comparison_pair_for_subtype("base_rate_neglect")
        problem_spec = self._build_base_rate_neglect_problem_spec(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
        )
        outcome_model = self._build_base_rate_neglect_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="base_rate_neglect",
            problem_spec=problem_spec,
            numeric_values=[
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
                choose_state_high,
                choose_state_low,
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=choose_state_high,
            right_value=choose_state_low,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=2,
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "solver_family": "bayes_update",
                "framing_intent": "base_rate_neglect_behavioral_prompt",
                "base_rate_small": prior_high <= 0.1,
                "likelihood_ratio": round(
                    p_signal_high_given_high / max(p_signal_high_given_low, 1e-6),
                    6,
                ),
                "posterior_minus_half_signed": round(choose_state_high - 0.5, 6),
                "base_rate_regime_intended": regime,
                "base_rate_regime_realized": self._base_rate_realized_regime(
                    choose_state_high - 0.5
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="base_rate_neglect",
            task_id_prefix="belief_base_rate",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            beliefs={
                "prior_high": self._round_scalar(prior_high),
                "prior_low": self._round_scalar(1 - prior_high),
                "posterior_high": choose_state_high,
                "posterior_low": choose_state_low,
            },
            actions=["choose_state_high", "choose_state_low", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=decision_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Posterior probabilities are class_high={choose_state_high} "
                f"and class_low={choose_state_low}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _base_rate_neglect_fallback(
        self, *, regime: str
    ) -> tuple[float, float, float, Literal["high", "low"], float]:
        if regime == "choose_state_high":
            prior_high = 0.35
            p_signal_high_given_high = 0.9
            p_signal_high_given_low = 0.2
            observed_signal: Literal["high", "low"] = "high"
        elif regime == "choose_state_low":
            prior_high = 0.05
            p_signal_high_given_high = 0.9
            p_signal_high_given_low = 0.2
            observed_signal = "high"
        else:
            prior_high = 0.2
            p_signal_high_given_high = 0.8
            p_signal_high_given_low = 0.2
            observed_signal = "high"
        posterior_high = self._posterior_from_signal(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
        )
        gap = posterior_high - 0.5
        if regime == "choose_state_high" and gap < 0.08:
            raise RuntimeError("Fallback failed for base_rate_neglect choose_state_high")
        if regime == "choose_state_low" and gap > -0.08:
            raise RuntimeError("Fallback failed for base_rate_neglect choose_state_low")
        if regime == "near_indifferent" and abs(gap) > 0.02:
            raise RuntimeError("Fallback failed for base_rate_neglect near_indifferent")
        return (
            prior_high,
            p_signal_high_given_high,
            p_signal_high_given_low,
            observed_signal,
            posterior_high,
        )

    def _base_rate_realized_regime(self, gap: float) -> str:
        if gap >= 0.08:
            return "choose_state_high"
        if gap <= -0.08:
            return "choose_state_low"
        if abs(gap) <= 0.02:
            return "near_indifferent"
        return "ambiguous_band"

    def _conjunction_semantic_pools(
        self,
    ) -> dict[str, dict[str, tuple[str, ...] | tuple[tuple[str, str], ...]]]:
        return {
            "startup": {
                "profiles": (
                    "The founder runs daily customer calls and tracks funnel conversion.",
                    "The operating lead reviews retention dashboards each morning.",
                    "The team ships rapid updates and stays close to design partners.",
                ),
                "event_pairs": (
                    ("startup grows revenue", "startup signs an enterprise partnership"),
                    (
                        "startup reduces churn",
                        "startup reduces churn and expands multi-year contracts",
                    ),
                ),
            },
            "nonprofit": {
                "profiles": (
                    "The organizer coordinates volunteers, donor outreach, and service events.",
                    "The nonprofit lead keeps steady grant follow-up and partner ties.",
                    "Program staff track attendance and recurring donor engagement.",
                ),
                "event_pairs": (
                    (
                        "nonprofit expands outreach",
                        "nonprofit expands outreach and secures a major grant",
                    ),
                    (
                        "nonprofit increases volunteers",
                        "nonprofit increases volunteers and launches a new food drive",
                    ),
                ),
            },
            "scientist": {
                "profiles": (
                    "The researcher emphasizes replication checks and clear lab records.",
                    "The lab lead stresses careful measurement and weekly data audits.",
                    "The scientist is methodical in experiment design and interpretation.",
                ),
                "event_pairs": (
                    (
                        "researcher publishes findings",
                        "researcher publishes findings and wins a top conference award",
                    ),
                    (
                        "study produces significant result",
                        "study produces significant result and is replicated externally",
                    ),
                ),
            },
            "campaign": {
                "profiles": (
                    "The volunteer manages canvassing routes and turnout reminders.",
                    "The field team runs phone banks and tracks voter response notes.",
                    "The organizer handles outreach and regularly recruits volunteers.",
                ),
                "event_pairs": (
                    (
                        "campaign increases turnout",
                        "campaign increases turnout and flips a swing precinct",
                    ),
                    (
                        "campaign boosts volunteer hours",
                        "campaign boosts volunteer hours and raises small-dollar donations",
                    ),
                ),
            },
            "sales": {
                "profiles": (
                    "The sales lead runs disciplined pipeline reviews and account plans.",
                    "The rep keeps discovery calls active and follows up on proposals.",
                    "The team tracks deal stages and prioritizes strategic accounts.",
                ),
                "event_pairs": (
                    (
                        "sales lead closes deals",
                        "sales lead closes deals and lands a Fortune 500 account",
                    ),
                    (
                        "team hits quarterly target",
                        "team hits quarterly target and expands an enterprise contract",
                    ),
                ),
            },
            "product_manager": {
                "profiles": (
                    "The PM balances roadmap tradeoffs with user research signals.",
                    "The PM tracks activation metrics and coordinates release planning.",
                    "The lead turns customer feedback into clear product priorities.",
                ),
                "event_pairs": (
                    ("product launch succeeds", "product launch succeeds and lifts retention"),
                    (
                        "feature adoption rises",
                        "feature adoption rises and reduces support tickets",
                    ),
                ),
            },
            "community_organizer": {
                "profiles": (
                    "The organizer builds coalitions and tracks resident follow-up.",
                    "The neighborhood lead coordinates cleanups and meeting turnout.",
                    "The community team maintains contact lists and event participation.",
                ),
                "event_pairs": (
                    (
                        "community turnout increases",
                        "community turnout increases and a local policy proposal passes",
                    ),
                    (
                        "organizer recruits volunteers",
                        "organizer recruits volunteers and launches a housing campaign",
                    ),
                ),
            },
            "teacher": {
                "profiles": (
                    "The teacher plans structured lessons and tracks student progress.",
                    "The educator keeps careful classroom routines and parent updates.",
                    "The teaching team uses weekly assessments and interventions.",
                ),
                "event_pairs": (
                    ("class scores improve", "class scores improve and absenteeism declines"),
                    (
                        "teacher boosts engagement",
                        "teacher boosts engagement and launches an after-school program",
                    ),
                ),
            },
        }

    def _sample_conjunction_profile(self) -> tuple[str, str]:
        pools = self._conjunction_semantic_pools()
        pool_name = self.rng.choice(list(pools.keys()))
        profiles = pools[pool_name]["profiles"]
        if not isinstance(profiles, tuple):
            raise RuntimeError("Invalid conjunction semantic pool profiles.")
        return pool_name, self.rng.choice(list(profiles))

    def _sample_conjunction_event_pair(self, *, pool_name: str) -> tuple[str, str]:
        pools = self._conjunction_semantic_pools()
        if pool_name not in pools:
            raise RuntimeError(f"Unknown conjunction semantic pool: {pool_name}")
        event_pairs = pools[pool_name]["event_pairs"]
        if not isinstance(event_pairs, tuple):
            raise RuntimeError("Invalid conjunction semantic pool event_pairs.")
        return self.rng.choice(list(event_pairs))

    def _generate_conjunction_fallacy(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        semantic_domain, profile_description = self._sample_conjunction_profile()
        event_a_label, event_b_detail_label = self._sample_conjunction_event_pair(
            pool_name=semantic_domain
        )
        p_event_a = round(self.rng.uniform(0.25, 0.9), 2)
        p_event_a_and_b = round(self.rng.uniform(0.05, max(0.06, p_event_a - 0.02)), 2)
        if p_event_a_and_b >= p_event_a:
            p_event_a_and_b = round(max(0.01, p_event_a - 0.03), 2)
        conjunction_in_option_a = self.rng.random() < 0.5

        comparison_pair = self._comparison_pair_for_subtype("conjunction_fallacy")
        problem_spec = self._build_conjunction_fallacy_problem_spec(
            profile_description=profile_description,
            semantic_domain=semantic_domain,
            event_a_label=event_a_label,
            event_b_detail_label=event_b_detail_label,
            p_event_a=p_event_a,
            p_event_a_and_b=p_event_a_and_b,
            conjunction_in_option_a=conjunction_in_option_a,
        )
        outcome_model = self._build_conjunction_fallacy_outcome_model(problem_spec=problem_spec)
        action_values, decision_values, optimal, _ = self._solve_from_problem_spec(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="conjunction_fallacy",
            problem_spec=problem_spec,
            numeric_values=[p_event_a, p_event_a_and_b],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_A"],
            right_value=decision_values["choose_B"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[p_event_a, p_event_a_and_b],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=1,
            prompt_complexity_features=prompt_complexity_features,
        )
        conjunction_option = "A" if conjunction_in_option_a else "B"
        constituent_option = "B" if conjunction_in_option_a else "A"
        difficulty_metrics.update(
            {
                "conjunction_subset_constituent_event": event_a_label,
                "conjunction_is_subset_of": event_a_label,
                "conjunction_option": conjunction_option,
                "constituent_option": constituent_option,
                "conjunction_event_label": problem_spec["options"][conjunction_option][
                    "event_label"
                ],
                "constituent_event_label": problem_spec["options"][constituent_option][
                    "event_label"
                ],
                "conjunction_gap": round(abs(p_event_a - p_event_a_and_b), 6),
                "conjunction_render_mode": self._last_prompt_style_regime,
                "representativeness_strength": self._last_representativeness_strength,
                "conjunction_semantic_pool": semantic_domain,
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="conjunction_fallacy",
            task_id_prefix="belief_conjunction",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            beliefs={},
            actions=["choose_A", "choose_B", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                "A conjunction event cannot exceed its constituent event probability; "
                "the higher-probability option is selected."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_gambler_fallacy(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        p_heads = 0.5
        streak_length = self.rng.randint(4, 9)
        streak_symbol = self.rng.choice(["H", "T"])
        recent_sequence = streak_symbol * streak_length
        queried_outcome: Literal["heads", "tails"] = self.rng.choice(["heads", "tails"])
        correct_claim_in_option_a = self.rng.random() < 0.5

        comparison_pair = self._comparison_pair_for_subtype("gambler_fallacy")
        problem_spec = self._build_gambler_fallacy_problem_spec(
            p_heads=p_heads,
            queried_outcome=queried_outcome,
            recent_sequence=recent_sequence,
            correct_claim_in_option_a=correct_claim_in_option_a,
        )
        outcome_model = self._build_gambler_fallacy_outcome_model(problem_spec=problem_spec)
        action_values, decision_values, optimal, _ = self._solve_from_problem_spec(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="gambler_fallacy",
            problem_spec=problem_spec,
            numeric_values=[p_heads, streak_length],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_A"],
            right_value=decision_values["choose_B"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[p_heads, streak_length],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=1,
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "streak_length": streak_length,
                "recent_streak_symbol": streak_symbol,
                "independence_assumption": True,
                "fair_process": p_heads == 0.5,
                "queried_outcome": queried_outcome,
                "correct_claim_option": "A" if correct_claim_in_option_a else "B",
                "independence_claim_correct_realized": optimal,
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="gambler_fallacy",
            task_id_prefix="belief_gambler",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "recent_sequence": recent_sequence,
                "queried_outcome": queried_outcome,
                "options": problem_spec["options"],
            },
            beliefs={},
            actions=["choose_A", "choose_B", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                "With unchanged 50/50 odds, a streak does not make the queried outcome "
                "more likely; the matching claim is selected."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_sample_size_neglect(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        small_n = self.rng.choice([20, 30, 40, 50])
        large_n = self.rng.choice([120, 150, 180, 220])
        baseline_rate = self.rng.choice([0.5, 0.45, 0.55])
        if baseline_rate >= 0.5:
            extreme_direction: Literal["at_or_above", "at_or_below"] = "at_or_above"
            extreme_threshold = self.rng.choice([0.6, 0.65])
        else:
            extreme_direction = "at_or_below"
            extreme_threshold = self.rng.choice([0.35, 0.4])

        small_first = self.rng.random() < 0.5
        sample_size_a = small_n if small_first else large_n
        sample_size_b = large_n if small_first else small_n

        comparison_pair = self._comparison_pair_for_subtype("sample_size_neglect")
        problem_spec = self._build_sample_size_neglect_problem_spec(
            sample_size_a=sample_size_a,
            sample_size_b=sample_size_b,
            baseline_rate=baseline_rate,
            extreme_threshold=extreme_threshold,
            extreme_direction=extreme_direction,
        )
        outcome_model = self._build_sample_size_neglect_outcome_model(problem_spec=problem_spec)
        action_values, decision_values, optimal, _ = self._solve_from_problem_spec(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="sample_size_neglect",
            problem_spec=problem_spec,
            numeric_values=[sample_size_a, sample_size_b, baseline_rate, extreme_threshold],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_A"],
            right_value=decision_values["choose_B"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[sample_size_a, sample_size_b, baseline_rate, extreme_threshold],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=2,
            prompt_complexity_features=prompt_complexity_features,
        )
        smaller_option = "A" if sample_size_a < sample_size_b else "B"
        cutoff_a = self._extreme_count_cutoff(
            sample_size=sample_size_a,
            extreme_threshold=extreme_threshold,
            extreme_direction=extreme_direction,
        )
        cutoff_b = self._extreme_count_cutoff(
            sample_size=sample_size_b,
            extreme_threshold=extreme_threshold,
            extreme_direction=extreme_direction,
        )
        difficulty_metrics.update(
            {
                "sample_size_small": min(sample_size_a, sample_size_b),
                "sample_size_large": max(sample_size_a, sample_size_b),
                "extreme_event_basis": "sample_proportion_threshold",
                "extreme_threshold": extreme_threshold,
                "extreme_direction": extreme_direction,
                "extreme_count_cutoff_A": cutoff_a,
                "extreme_count_cutoff_B": cutoff_b,
                "extreme_threshold_rounding_rule": ("ceil_for_at_or_above_floor_for_at_or_below"),
                "smaller_sample_option": smaller_option,
                "small_sample_more_extreme_realized": (
                    (
                        smaller_option == "A"
                        and decision_values["choose_A"] > decision_values["choose_B"]
                    )
                    or (
                        smaller_option == "B"
                        and decision_values["choose_B"] > decision_values["choose_A"]
                    )
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="sample_size_neglect",
            task_id_prefix="belief_sample_size",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            beliefs={},
            actions=["choose_A", "choose_B", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                "Extreme sample-frequency probability is compared via binomial tails; "
                "the higher-probability process is selected."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_overprecision_calibration(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        true_value_mean = float(self.rng.randint(60, 180))
        true_value_sd = round(self.rng.uniform(4.0, 15.0), 2)
        center_a = true_value_mean
        center_b = true_value_mean
        lower_a = true_value_mean - 1
        upper_a = true_value_mean + 1
        lower_b = true_value_mean - 1
        upper_b = true_value_mean + 1
        variant = "width_only"
        found_candidate = False
        for _ in range(200):
            candidate_variant = self.rng.choice(["width_only", "miscentered_tradeoff"])
            if candidate_variant == "width_only":
                common_center = round(
                    true_value_mean + self.rng.uniform(-0.25, 0.25) * true_value_sd,
                    2,
                )
                narrow_half_width = round(
                    self.rng.uniform(0.4 * true_value_sd, 1.0 * true_value_sd),
                    2,
                )
                wide_half_width = round(
                    self.rng.uniform(1.3 * true_value_sd, 2.4 * true_value_sd),
                    2,
                )
                if wide_half_width <= narrow_half_width:
                    wide_half_width = round(
                        narrow_half_width + max(0.5, 0.4 * true_value_sd),
                        2,
                    )
                candidate_center_1 = common_center
                candidate_lower_1 = self._round_problem_decimal(
                    common_center - wide_half_width, digits=2
                )
                candidate_upper_1 = self._round_problem_decimal(
                    common_center + wide_half_width, digits=2
                )
                candidate_center_2 = common_center
                candidate_lower_2 = self._round_problem_decimal(
                    common_center - narrow_half_width, digits=2
                )
                candidate_upper_2 = self._round_problem_decimal(
                    common_center + narrow_half_width, digits=2
                )
            else:
                centered_half_width = round(
                    self.rng.uniform(0.8 * true_value_sd, 1.4 * true_value_sd),
                    2,
                )
                miscentered_half_width = round(
                    self.rng.uniform(1.4 * true_value_sd, 2.3 * true_value_sd),
                    2,
                )
                if miscentered_half_width <= centered_half_width:
                    miscentered_half_width = round(
                        centered_half_width + max(0.5, 0.4 * true_value_sd),
                        2,
                    )
                centered_center = round(
                    true_value_mean + self.rng.uniform(-0.2, 0.2) * true_value_sd,
                    2,
                )
                shift_sign = self.rng.choice([-1, 1])
                miscentered_center = round(
                    true_value_mean + shift_sign * self.rng.uniform(1.8, 2.8) * true_value_sd,
                    2,
                )
                candidate_center_1 = centered_center
                candidate_lower_1 = self._round_problem_decimal(
                    centered_center - centered_half_width, digits=2
                )
                candidate_upper_1 = self._round_problem_decimal(
                    centered_center + centered_half_width, digits=2
                )
                candidate_center_2 = miscentered_center
                candidate_lower_2 = self._round_problem_decimal(
                    miscentered_center - miscentered_half_width, digits=2
                )
                candidate_upper_2 = self._round_problem_decimal(
                    miscentered_center + miscentered_half_width, digits=2
                )

            coverage_1 = self._normal_interval_coverage(
                lower=candidate_lower_1,
                upper=candidate_upper_1,
                mu=true_value_mean,
                sd=true_value_sd,
            )
            coverage_2 = self._normal_interval_coverage(
                lower=candidate_lower_2,
                upper=candidate_upper_2,
                mu=true_value_mean,
                sd=true_value_sd,
            )
            if abs(coverage_1 - coverage_2) <= 0.02:
                continue

            if self.rng.random() < 0.5:
                center_a = candidate_center_1
                lower_a = candidate_lower_1
                upper_a = candidate_upper_1
                center_b = candidate_center_2
                lower_b = candidate_lower_2
                upper_b = candidate_upper_2
            else:
                center_a = candidate_center_2
                lower_a = candidate_lower_2
                upper_a = candidate_upper_2
                center_b = candidate_center_1
                lower_b = candidate_lower_1
                upper_b = candidate_upper_1
            variant = candidate_variant
            found_candidate = True
            break

        if not found_candidate:
            centered_half_width = round(1.0 * true_value_sd, 2)
            miscentered_half_width = round(1.9 * true_value_sd, 2)
            centered_center = true_value_mean
            miscentered_center = round(true_value_mean + 2.2 * true_value_sd, 2)
            center_a = centered_center
            lower_a = self._round_problem_decimal(centered_center - centered_half_width, digits=2)
            upper_a = self._round_problem_decimal(centered_center + centered_half_width, digits=2)
            center_b = miscentered_center
            lower_b = self._round_problem_decimal(
                miscentered_center - miscentered_half_width, digits=2
            )
            upper_b = self._round_problem_decimal(
                miscentered_center + miscentered_half_width, digits=2
            )
            variant = "miscentered_tradeoff_fallback"

        comparison_pair = self._comparison_pair_for_subtype("overprecision_calibration")
        problem_spec = self._build_overprecision_calibration_problem_spec(
            center_a=center_a,
            lower_a=lower_a,
            upper_a=upper_a,
            center_b=center_b,
            lower_b=lower_b,
            upper_b=upper_b,
            true_value_mean=true_value_mean,
            true_value_sd=true_value_sd,
        )
        outcome_model = self._build_overprecision_calibration_outcome_model(
            problem_spec=problem_spec
        )
        action_values, decision_values, optimal, _ = self._solve_from_problem_spec(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="overprecision_calibration",
            problem_spec=problem_spec,
            numeric_values=[
                true_value_mean,
                true_value_sd,
                center_a,
                lower_a,
                upper_a,
                center_b,
                lower_b,
                upper_b,
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_A"],
            right_value=decision_values["choose_B"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    true_value_mean,
                    true_value_sd,
                    center_a,
                    lower_a,
                    upper_a,
                    center_b,
                    lower_b,
                    upper_b,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=1,
            prompt_complexity_features=prompt_complexity_features,
        )
        width_a = upper_a - lower_a
        width_b = upper_b - lower_b
        wider_option = "A" if width_a > width_b else "B"
        offset_a = abs(center_a - true_value_mean)
        offset_b = abs(center_b - true_value_mean)
        difficulty_metrics.update(
            {
                "interval_width_ratio": round(
                    max(width_a, width_b) / max(min(width_a, width_b), 1e-6),
                    6,
                ),
                "overprecision_gap": round(
                    abs(decision_values["choose_A"] - decision_values["choose_B"]),
                    6,
                ),
                "true_value_mean": true_value_mean,
                "interval_center_offset_A": round(offset_a, 6),
                "interval_center_offset_B": round(offset_b, 6),
                "overprecision_variant_intended": variant,
                "wider_interval_option": wider_option,
                "wider_interval_superior_realized": (
                    (
                        wider_option == "A"
                        and decision_values["choose_A"] > decision_values["choose_B"]
                    )
                    or (
                        wider_option == "B"
                        and decision_values["choose_B"] > decision_values["choose_A"]
                    )
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="overprecision_calibration",
            task_id_prefix="belief_overprecision",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            beliefs={},
            actions=["choose_A", "choose_B", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                "Each interval is scored by its implied probability of containing the "
                "true value under the stated distribution; the higher score is selected."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )
