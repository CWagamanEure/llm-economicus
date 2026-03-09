"""
Bayesian updating and signal extraction dataset generator.

These samples are normative Bayesian decision problems where posterior beliefs
are updated from binary signals and mapped to binary actions using
expected-value-maximizing decision rules.
"""

import hashlib
import json
import logging
import random
import re
from collections import Counter
from typing import Any, Callable, Literal

from base_generator import BaseGenerator
from difficulty_config import BAYESIAN_SIGNAL_DEFAULT_DIFFICULTY_BY_SUBTYPE

from schema import (
    ActionScalars,
    BasicBayesUpdateProblemSpec,
    BayesianAssumptions,
    BayesianSignalTaskSubtype,
    BinarySignalDecisionProblemSpec,
    ComparisonPair,
    DataPoint,
    InformationCascadeProblemSpec,
    Metadata,
    NoisySignalAssetUpdateProblemSpec,
    ProblemSpec,
    SolverTrace,
    Target,
)

TaskSubtype = BayesianSignalTaskSubtype
DifficultyMetrics = dict[str, float | int | bool | str]
ActionValueSemantics = Literal[
    "posterior_probability_comparison",
    "posterior_expected_payoff_comparison",
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
    "manufacturing_defect",
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
    "manufacturing_defect",
)
COMPARISON_PAIR_BY_SUBTYPE: dict[TaskSubtype, ComparisonPair] = {
    "basic_bayes_update": {
        "left_action": "choose_state_high",
        "right_action": "choose_state_low",
    },
    "binary_signal_decision": {
        "left_action": "act",
        "right_action": "do_not_act",
    },
    "information_cascade_step": {
        "left_action": "choose_high",
        "right_action": "choose_low",
    },
    "noisy_signal_asset_update": {
        "left_action": "buy",
        "right_action": "do_not_buy",
    },
}
PROMPT_RENDERER_METHOD_BY_SUBTYPE: dict[TaskSubtype, str] = {
    "basic_bayes_update": "_render_basic_bayes_update_prompt",
    "binary_signal_decision": "_render_binary_signal_decision_prompt",
    "information_cascade_step": "_render_information_cascade_prompt",
    "noisy_signal_asset_update": "_render_noisy_signal_asset_update_prompt",
}
ACTION_VALUE_SEMANTICS_BY_SUBTYPE: dict[TaskSubtype, ActionValueSemantics] = {
    "basic_bayes_update": "posterior_probability_comparison",
    "binary_signal_decision": "posterior_expected_payoff_comparison",
    "information_cascade_step": "posterior_probability_comparison",
    "noisy_signal_asset_update": "posterior_expected_payoff_comparison",
}
PROBABILITY_FIELD_KEYS = {
    "prior_high",
    "p_signal_high_given_high",
    "p_signal_high_given_low",
    "posterior_high",
    "posterior_low",
    "posterior_threshold",
    "probability",
}
logger = logging.getLogger(__name__)


class BayesianSignalGenerator(BaseGenerator):
    """Generate normative Bayesian update tasks with pairwise decisions."""

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
        self._last_prompt_frame_variant: PromptFrameVariant = "medical_screening"
        self._last_prompt_style_regime: PromptStyleRegime = "neutral_realistic"
        self._last_reliability_phrase_pattern: str | None = None
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
        _ = task_subtype
        return (
            "medical_screening",
            "fraud_detection",
            "hiring_screen",
            "security_alert",
            "trading_signal",
            "manufacturing_defect",
        )

    def _apply_prompt_frame_variant(
        self,
        *,
        prompt: str,
        frame_variant: PromptFrameVariant,
        task_subtype: TaskSubtype,
    ) -> str:
        _ = frame_variant
        _ = task_subtype
        return prompt

    def generate(self) -> DataPoint:
        current_index = self.sample_index
        self.sample_index += 1
        subtype: TaskSubtype = self.rng.choice(
            [
                "basic_bayes_update",
                "binary_signal_decision",
                "information_cascade_step",
                "noisy_signal_asset_update",
            ]
        )
        if subtype == "basic_bayes_update":
            return self._generate_basic_bayes_update(sample_index=current_index)
        if subtype == "binary_signal_decision":
            return self._generate_binary_signal_decision(sample_index=current_index)
        if subtype == "information_cascade_step":
            return self._generate_information_cascade_step(sample_index=current_index)
        if subtype == "noisy_signal_asset_update":
            return self._generate_noisy_signal_asset_update(sample_index=current_index)
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
            semantic_context=prompt_frame_variant,
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
        self, *, values: ActionScalars | dict[str, float], comparison_pair: ComparisonPair
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
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> float:
        if key not in mapping:
            raise ValueError(f"{field_path}.{key} is required for subtype '{subtype}'.")
        value = mapping[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_path}.{key} must be numeric for subtype '{subtype}'.")
        return float(value)

    def _require_probability_field(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> float:
        value = self._require_numeric_field(
            mapping, key=key, field_path=field_path, subtype=subtype
        )
        if value < 0 or value > 1:
            raise ValueError(f"{field_path}.{key} must be between 0 and 1 for subtype '{subtype}'.")
        return value

    def _require_non_negative_field(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
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

    def _format_number(self, value: int | float) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")

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

    def _posterior_from_signal_sequence(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signals: list[Literal["high", "low"]],
    ) -> float:
        posterior = prior_high
        for signal in observed_signals:
            posterior = self._posterior_from_signal(
                prior_high=posterior,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=signal,
            )
        return posterior

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
        posterior_beliefs: PosteriorBeliefs = {
            "prior_high": self._round_scalar(prior_high),
            "prior_low": self._round_scalar(1 - prior_high),
            "posterior_high": self._round_scalar(posterior_high),
            "posterior_low": self._round_scalar(posterior_low),
        }

        if subtype == "basic_bayes_update":
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
            return action_values, decision_values, optimal_decision, posterior_beliefs

        if subtype == "binary_signal_decision":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            payoff_if_high = self._require_numeric_field(
                option_a,
                key="payoff_if_high",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            payoff_if_low = self._require_numeric_field(
                option_a,
                key="payoff_if_low",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            do_not_act_payoff = self._require_numeric_field(
                option_b,
                key="payoff",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            act = self._round_scalar(
                posterior_high * payoff_if_high + posterior_low * payoff_if_low
            )
            do_not_act = self._round_scalar(do_not_act_payoff)
            action_values = ActionScalars(
                {
                    "act": act,
                    "do_not_act": do_not_act,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="act",
                left_value=act,
                right_label="do_not_act",
                right_value=do_not_act,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, posterior_beliefs

        if subtype == "information_cascade_step":
            public_actions_raw = assumptions.get("public_actions", [])
            if not isinstance(public_actions_raw, list):
                raise ValueError(
                    "problem_spec.assumptions.public_actions must be a list for "
                    "subtype 'information_cascade_step'."
                )
            observed_signals: list[Literal["high", "low"]] = []
            for index, action in enumerate(public_actions_raw):
                if action == "choose_high":
                    observed_signals.append("high")
                    continue
                if action == "choose_low":
                    observed_signals.append("low")
                    continue
                raise ValueError(
                    "problem_spec.assumptions.public_actions[%d] must be choose_high "
                    "or choose_low for subtype 'information_cascade_step'." % index
                )
            observed_signals.append(observed_signal)
            posterior_high = self._posterior_from_signal_sequence(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signals=observed_signals,
            )
            posterior_low = 1 - posterior_high
            posterior_beliefs = {
                "prior_high": self._round_scalar(prior_high),
                "prior_low": self._round_scalar(1 - prior_high),
                "posterior_high": self._round_scalar(posterior_high),
                "posterior_low": self._round_scalar(posterior_low),
            }
            choose_high = self._round_scalar(posterior_high)
            choose_low = self._round_scalar(posterior_low)
            action_values = ActionScalars(
                {
                    "choose_high": choose_high,
                    "choose_low": choose_low,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_high",
                left_value=choose_high,
                right_label="choose_low",
                right_value=choose_low,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, posterior_beliefs

        if subtype == "noisy_signal_asset_update":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            value_if_high = self._require_numeric_field(
                option_a,
                key="value_if_high",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            value_if_low = self._require_numeric_field(
                option_a,
                key="value_if_low",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            market_price = self._require_numeric_field(
                option_a,
                key="market_price",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            do_not_buy = self._require_numeric_field(
                option_b,
                key="payoff",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            transaction_cost = float(assumptions.get("transaction_cost", 0.0))
            buy = self._round_scalar(
                posterior_high * value_if_high
                + posterior_low * value_if_low
                - market_price
                - transaction_cost
            )
            no_buy = self._round_scalar(do_not_buy)
            action_values = ActionScalars({"buy": buy, "do_not_buy": no_buy})
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="buy",
                left_value=buy,
                right_label="do_not_buy",
                right_value=no_buy,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision, posterior_beliefs

        raise ValueError(f"Unsupported subtype: {subtype}")

    def _verify_target_solution(
        self, *, problem_spec: ProblemSpec, target: Target, prompt: str = ""
    ) -> None:
        self._assert_probabilities_in_unit_interval(problem_spec)
        (
            solved_action_values,
            solved_decision_values,
            solved_optimal_decision,
            solved_posterior_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        normalized_solved_action_values = self._normalize_scalars_for_comparison_pair(
            values=solved_action_values, comparison_pair=target.comparison_pair
        )
        normalized_solved_decision_values = self._normalize_scalars_for_comparison_pair(
            values=solved_decision_values, comparison_pair=target.comparison_pair
        )
        tolerance = 1e-9

        def _assert_scalar_map_matches(
            field_name: str, expected: ActionScalars, actual: ActionScalars
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
            "target.action_values", normalized_solved_action_values, target.action_values
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
        posterior_high = float(target.beliefs.get("posterior_high", -1))
        posterior_low = float(target.beliefs.get("posterior_low", -1))
        if abs(posterior_high - solved_posterior_beliefs["posterior_high"]) > tolerance:
            raise ValueError(
                "target.beliefs.posterior_high mismatch: "
                f"expected {solved_posterior_beliefs['posterior_high']}, got {posterior_high}."
            )
        if abs(posterior_low - solved_posterior_beliefs["posterior_low"]) > tolerance:
            raise ValueError(
                "target.beliefs.posterior_low mismatch: "
                f"expected {solved_posterior_beliefs['posterior_low']}, got {posterior_low}."
            )
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
            actions=actions, comparison_pair=comparison_pair
        )
        normalized_action_values = self._normalize_scalars_for_comparison_pair(
            values=action_values, comparison_pair=comparison_pair
        )
        normalized_decision_values = self._normalize_scalars_for_comparison_pair(
            values=decision_values, comparison_pair=comparison_pair
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
            task_family="bayesian_signal_extraction",
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
        return BAYESIAN_SIGNAL_DEFAULT_DIFFICULTY_BY_SUBTYPE[subtype]

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
        action_tokens = re.findall(
            r"\b(?:choose_[a-z0-9_]+|act|do_not_act|buy|do_not_buy)\b",
            lower_prompt,
        )
        prompt_action_token_count = len(action_tokens)
        prompt_contains_action_tokens = prompt_action_token_count > 0
        left_action = comparison_pair["left_action"]
        right_action = comparison_pair["right_action"]
        asymmetric_choice_framing = left_action.split("_", 1)[0] != right_action.split("_", 1)[0]
        prompt_complexity = (
            clause_count
            + int(has_decimal_in_prompt)
            + int(asymmetric_choice_framing)
            + int(prompt_contains_action_tokens)
        )
        return {
            "prompt_style_variant": prompt_style,
            "prompt_style_regime": resolved_regime,
            "prompt_clause_count": clause_count,
            "prompt_has_decimal": has_decimal_in_prompt,
            "prompt_asymmetric_choice_framing": asymmetric_choice_framing,
            "prompt_contains_action_tokens": prompt_contains_action_tokens,
            "prompt_action_token_count": prompt_action_token_count,
            "prompt_complexity": prompt_complexity,
            "prompt_mentions_bayes_rule": "bayes" in lower_prompt,
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

    def _build_bayesian_assumptions(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
        public_actions: list[Literal["choose_high", "choose_low"]] | None = None,
        transaction_cost: float | None = None,
    ) -> BayesianAssumptions:
        assumptions: BayesianAssumptions = {
            "prior_high": prior_high,
            "p_signal_high_given_high": p_signal_high_given_high,
            "p_signal_high_given_low": p_signal_high_given_low,
            "observed_signal": observed_signal,
            "signal_model": "binary_conditional_likelihood",
            "decision_rule": "bayes_update_then_expected_value",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }
        if public_actions is not None:
            assumptions["public_actions"] = public_actions
        if transaction_cost is not None:
            assumptions["transaction_cost"] = transaction_cost
        return assumptions

    def _build_basic_bayes_update_problem_spec(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
    ) -> BasicBayesUpdateProblemSpec:
        return {
            "task_subtype": "basic_bayes_update",
            "objective": "infer the more likely state using Bayes rule",
            "options": {
                "A": {"type": "state_hypothesis", "state": "high"},
                "B": {"type": "state_hypothesis", "state": "low"},
            },
            "assumptions": self._build_bayesian_assumptions(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
            ),
        }

    def _build_binary_signal_decision_problem_spec(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
        payoff_if_high: float,
        payoff_if_low: float,
        do_not_act_payoff: float,
    ) -> BinarySignalDecisionProblemSpec:
        return {
            "task_subtype": "binary_signal_decision",
            "objective": "maximize posterior expected payoff after Bayesian updating",
            "options": {
                "A": {
                    "type": "act",
                    "payoff_if_high": payoff_if_high,
                    "payoff_if_low": payoff_if_low,
                },
                "B": {
                    "type": "do_not_act",
                    "payoff": do_not_act_payoff,
                },
            },
            "assumptions": self._build_bayesian_assumptions(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
            ),
        }

    def _build_information_cascade_problem_spec(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
        public_actions: list[Literal["choose_high", "choose_low"]],
    ) -> InformationCascadeProblemSpec:
        return {
            "task_subtype": "information_cascade_step",
            "objective": "maximize posterior probability of choosing the true state",
            "options": {
                "A": {"type": "cascade_action", "implied_state": "high"},
                "B": {"type": "cascade_action", "implied_state": "low"},
            },
            "assumptions": self._build_bayesian_assumptions(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
                public_actions=public_actions,
            ),
        }

    def _build_noisy_signal_asset_update_problem_spec(
        self,
        *,
        prior_high: float,
        p_signal_high_given_high: float,
        p_signal_high_given_low: float,
        observed_signal: Literal["high", "low"],
        value_if_high: float,
        value_if_low: float,
        market_price: float,
        transaction_cost: float,
    ) -> NoisySignalAssetUpdateProblemSpec:
        return {
            "task_subtype": "noisy_signal_asset_update",
            "objective": "maximize posterior expected trading payoff",
            "options": {
                "A": {
                    "type": "buy_asset",
                    "value_if_high": value_if_high,
                    "value_if_low": value_if_low,
                    "market_price": market_price,
                },
                "B": {"type": "do_not_buy", "payoff": 0.0},
            },
            "assumptions": self._build_bayesian_assumptions(
                prior_high=prior_high,
                p_signal_high_given_high=p_signal_high_given_high,
                p_signal_high_given_low=p_signal_high_given_low,
                observed_signal=observed_signal,
                transaction_cost=transaction_cost,
            ),
        }

    def _context_labels(self, context: PromptFrameVariant) -> dict[str, str]:
        labels = {
            "medical_screening": {
                "entity": "patient case",
                "high_short": "the condition is present",
                "low_short": "the condition is absent",
                "high": "the condition is truly present",
                "low": "the condition is absent",
                "when_high": "when the condition is present",
                "when_low": "when the condition is absent",
                "among_high_cases": "among condition-present cases",
                "among_low_cases": "among condition-absent cases",
                "signal_high": "screening test returns positive",
                "signal_low": "screening test returns negative",
                "observed_high": "The screening test came back positive.",
                "observed_low": "The screening test came back negative.",
                "history_high": "called positive",
                "history_low": "called negative",
            },
            "fraud_detection": {
                "entity": "payment review",
                "high_short": "the payment is fraudulent",
                "low_short": "the payment is legitimate",
                "high": "the payment is fraudulent",
                "low": "the payment is legitimate",
                "when_high": "when the payment is fraudulent",
                "when_low": "when the payment is legitimate",
                "among_high_cases": "among fraudulent payments",
                "among_low_cases": "among legitimate payments",
                "signal_high": "fraud system flags the payment",
                "signal_low": "fraud system does not flag the payment",
                "observed_high": "The payment was flagged by the fraud system.",
                "observed_low": "The payment was not flagged by the fraud system.",
                "history_high": "flagged",
                "history_low": "cleared",
            },
            "security_alert": {
                "entity": "security incident review",
                "high_short": "there is an active threat",
                "low_short": "there is no active threat",
                "high": "there is an active threat",
                "low": "there is no active threat",
                "when_high": "when an active threat is present",
                "when_low": "when no active threat is present",
                "among_high_cases": "among active-threat situations",
                "among_low_cases": "among no-threat situations",
                "signal_high": "monitoring system triggers an alert",
                "signal_low": "monitoring system stays quiet",
                "observed_high": "The monitoring system triggered an alert.",
                "observed_low": "The monitoring system did not trigger an alert.",
                "history_high": "escalated",
                "history_low": "dismissed",
            },
            "hiring_screen": {
                "entity": "candidate review",
                "high_short": "the candidate is a strong fit",
                "low_short": "the candidate is not a strong fit",
                "high": "the candidate is a strong fit",
                "low": "the candidate is not a strong fit",
                "when_high": "when the candidate is a strong fit",
                "when_low": "when the candidate is not a strong fit",
                "among_high_cases": "among strong-fit candidates",
                "among_low_cases": "among weak-fit candidates",
                "signal_high": "automated screen rates the candidate strong",
                "signal_low": "automated screen rates the candidate weak",
                "observed_high": "The candidate received a strong automated screen result.",
                "observed_low": "The candidate received a weak automated screen result.",
                "history_high": "recommended advance",
                "history_low": "recommended reject",
            },
            "trading_signal": {
                "entity": "market regime call",
                "high_short": "the market is in a bullish regime",
                "low_short": "the market is in a bearish regime",
                "high": "the market is in a bullish regime",
                "low": "the market is in a bearish regime",
                "when_high": "when the market is in a bullish regime",
                "when_low": "when the market is in a bearish regime",
                "among_high_cases": "among bullish-regime days",
                "among_low_cases": "among bearish-regime days",
                "signal_high": "model flashes a bullish signal",
                "signal_low": "model flashes a bearish signal",
                "observed_high": "The model flashed a bullish signal.",
                "observed_low": "The model flashed a bearish signal.",
                "history_high": "called bullish",
                "history_low": "called bearish",
            },
            "manufacturing_defect": {
                "entity": "production batch check",
                "high_short": "the batch is defect-heavy",
                "low_short": "the batch is within normal defect levels",
                "high": "the batch is defect-heavy",
                "low": "the batch is within normal defect levels",
                "when_high": "when the batch is defect-heavy",
                "when_low": "when the batch is within normal defect levels",
                "among_high_cases": "among defect-heavy batches",
                "among_low_cases": "among normal-quality batches",
                "signal_high": "inspection system raises a defect alarm",
                "signal_low": "inspection system does not raise an alarm",
                "observed_high": "The inspection system raised a defect alarm.",
                "observed_low": "The inspection system did not raise an alarm.",
                "history_high": "flagged defect-heavy",
                "history_low": "passed",
            },
        }
        return labels.get(context, labels["medical_screening"])

    def _observed_signal_sentence(self, *, labels: dict[str, str], signal: str) -> str:
        return labels["observed_high"] if signal == "high" else labels["observed_low"]

    def _binary_prompt_has_observed_signal(
        self,
        *,
        prompt: str,
        signal: str,
        observed_sentence: str,
    ) -> bool:
        lower = prompt.lower()
        observed_lower = observed_sentence.lower()
        signal_tokens = (
            f"signal={signal}",
            f"observed={signal}",
            f"observed {signal}",
            f"observed signal={signal}",
            f"observation={signal}",
            f"observation first ({signal})",
            f"private signal={signal}",
            f"signal is {signal}",
            f"signal {signal}",
        )
        if observed_lower in lower:
            return True
        return any(token in lower for token in signal_tokens)

    def _validate_binary_signal_decision_prompt_completeness(
        self,
        *,
        prompt: str,
        prior_text: str,
        observed_signal: str,
        observed_sentence: str,
        likelihood_high_text: str,
        likelihood_low_text: str,
        payoff_high_text: str,
        payoff_low_text: str,
        do_not_act_payoff_text: str,
    ) -> None:
        """Ensure the rendered prompt contains all decision-relevant visible inputs."""
        lower = prompt.lower()
        missing: list[str] = []
        if prior_text not in prompt:
            missing.append("prior/base_rate")
        if not self._binary_prompt_has_observed_signal(
            prompt=prompt, signal=observed_signal, observed_sentence=observed_sentence
        ):
            missing.append("observed_signal")
        if likelihood_high_text not in prompt or likelihood_low_text not in prompt:
            missing.append("likelihood_pair")
        if (
            payoff_high_text not in prompt
            or payoff_low_text not in prompt
            or do_not_act_payoff_text not in prompt
        ):
            missing.append("action_payoffs")
        if missing:
            raise ValueError(
                "binary_signal_decision prompt missing required elements: "
                f"{', '.join(missing)}. Prompt='{lower}'"
            )

    def _explicit_reliability_clause(
        self,
        *,
        p_high_text: str,
        p_low_text: str,
        state_high: str,
        state_low: str,
    ) -> str:
        return (
            f"this cue appears with probability {p_high_text} when {state_high} "
            f"and {p_low_text} when {state_low}"
        )

    def _reliability_clause_templates(self, *, tier: str) -> tuple[tuple[str, str], ...]:
        if tier == "formal":
            return (
                (
                    "formal_when_true_false",
                    "If {state_high}, this signal appears with probability "
                    "{p_high}; if {state_low}, it appears with probability {p_low}.",
                ),
                (
                    "formal_under_cases",
                    "A result like this shows up with probability {p_high} in cases where "
                    "{state_high} and {p_low} in cases where {state_low}.",
                ),
                (
                    "formal_conditional_statement",
                    "The conditional signal probability is {p_high} {when_high} and "
                    "{p_low} {when_low}.",
                ),
                (
                    "formal_seen_with_probability",
                    "This result is seen with probability {p_high} when {state_high} "
                    "and {p_low} when {state_low}.",
                ),
                (
                    "formal_returned_rate",
                    "This result is returned at probability {p_high} when {state_high} "
                    "and {p_low} when {state_low}.",
                ),
            )
        if tier == "neutral_natural":
            return (
                (
                    "neutral_seen_with_probability",
                    "This result is seen with probability {p_high} when {state_high} "
                    "and {p_low} when {state_low}.",
                ),
                (
                    "neutral_signal_occurs",
                    "The signal occurs with probability {p_high} if {state_high} and "
                    "{p_low} if {state_low}.",
                ),
                (
                    "neutral_shows_up_under",
                    "A result like this shows up with probability {p_high} in cases where "
                    "{state_high} and {p_low} in cases where {state_low}.",
                ),
                (
                    "neutral_when_true_false",
                    "If {state_high}, this signal appears with probability "
                    "{p_high}; if {state_low}, it appears with probability {p_low}.",
                ),
                (
                    "neutral_chance_in_cases",
                    "The chance of this signal is {p_high} in cases where {state_high} "
                    "and {p_low} in cases where {state_low}.",
                ),
                (
                    "neutral_returned_rate",
                    "This result is returned at probability {p_high} when {state_high} "
                    "and {p_low} when {state_low}.",
                ),
                (
                    "neutral_rate_under",
                    "The rate for this signal is {p_high} under {state_high} and "
                    "{p_low} under {state_low}.",
                ),
            )
        return (
            (
                "bias_seen_with_probability",
                "This result is seen with probability {p_high} when {state_high} and "
                "{p_low} when {state_low}.",
            ),
            (
                "bias_signal_occurs_if",
                "The signal occurs with probability {p_high} if {state_high} and "
                "{p_low} if {state_low}.",
            ),
            (
                "bias_shows_up_under",
                "A result like this shows up with probability {p_high} in cases where "
                "{state_high} and {p_low} in cases where {state_low}.",
            ),
            (
                "bias_when_true_false",
                "If {state_high}, this signal appears with probability "
                "{p_high}; if {state_low}, it appears with probability {p_low}.",
            ),
            (
                "bias_chance_in_cases",
                "The chance of this signal is {p_high} in cases where {state_high} "
                "and {p_low} in cases where {state_low}.",
            ),
            (
                "bias_returned_rate",
                "This result is returned at probability {p_high} when {state_high} and "
                "{p_low} when {state_low}.",
            ),
            (
                "bias_rate_under",
                "The rate for this signal is {p_high} under {state_high} and "
                "{p_low} under {state_low}.",
            ),
        )

    def _select_explicit_reliability_clause(
        self,
        *,
        task_subtype: str,
        frame_variant: PromptFrameVariant,
        tier: str,
        problem_spec: ProblemSpec,
        p_high_text: str,
        p_low_text: str,
        state_high: str,
        state_low: str,
    ) -> str:
        templates = self._reliability_clause_templates(tier=tier)
        rendered = [
            template.format(
                p_high=p_high_text,
                p_low=p_low_text,
                state_high=state_high,
                state_low=state_low,
                when_high=f"when {state_high}",
                when_low=f"when {state_low}",
            )
            for _, template in templates
        ]
        idx = self._select_template_variant(
            task_subtype=f"{task_subtype}_reliability_clause",
            frame_variant=frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            templates=rendered,
        )
        self._last_reliability_phrase_pattern = templates[idx][0]
        return rendered[idx]

    def _normalize_reliability_wording(
        self,
        *,
        prompt: str,
        p_high_text: str,
        p_low_text: str,
        state_high: str,
        state_low: str,
    ) -> str:
        clause = self._explicit_reliability_clause(
            p_high_text=p_high_text,
            p_low_text=p_low_text,
            state_high=state_high,
            state_low=state_low,
        )
        normalized = prompt
        replacements = (
            f"signal performance is {p_high_text} vs {p_low_text}",
            f"signal split={p_high_text}/{p_low_text}",
            f"signal split {p_high_text}/{p_low_text}",
            f"signal reliability split: {p_high_text} vs {p_low_text}",
            f"signal reliability split {p_high_text} vs {p_low_text}",
            f"signal reliability split={p_high_text}/{p_low_text}",
            f"signal reliability split {p_high_text}/{p_low_text}",
            f"cue profile={p_high_text}/{p_low_text}",
            f"cue profile {p_high_text}/{p_low_text}",
            f"likelihoods={p_high_text}/{p_low_text}",
            f"likelihoods {p_high_text}/{p_low_text}",
            f"signal rates run {p_high_text} vs {p_low_text}",
            f"Signal rates run {p_high_text} vs {p_low_text}",
            f"signal rates are {p_high_text} and {p_low_text}",
            f"Signal rates are {p_high_text} and {p_low_text}",
        )
        for src in replacements:
            if src in normalized:
                normalized = normalized.replace(src, clause)
        p1 = re.escape(p_high_text)
        p2 = re.escape(p_low_text)
        regex_patterns = (
            rf"signal split(?:\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"signal reliability split(?:\s*:|\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"cue profile(?:\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"signal profile(?:\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"likelihood split(?:\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"signal performance(?:\s+is|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"likelihoods(?:\s+are|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
            rf"signal rates(?:\s+are|=)?\s*{p1}\s*(?:/|vs)\s*{p2}",
        )
        for pattern in regex_patterns:
            normalized = re.sub(pattern, clause, normalized, flags=re.IGNORECASE)
        normalized = normalized.replace(
            f"signal model={p_high_text}/{p_low_text}",
            f"{clause}",
        )
        normalized = normalized.replace(
            f"likelihood model={p_high_text} vs {p_low_text}",
            f"{clause}",
        )
        normalized = normalized.replace(
            f"signal rates={p_high_text}/{p_low_text}",
            f"{clause}",
        )
        return normalized

    def _validate_reliability_wording_lint(
        self,
        *,
        prompt: str,
        state_high: str,
        state_low: str,
    ) -> None:
        lower = prompt.lower()
        shorthand_markers = (
            "signal split",
            "cue profile",
            "signal performance",
        )
        has_vs_number = bool(re.search(r"\b\d+(?:\.\d+)?\s*vs\.?\s*\d+(?:\.\d+)?\b", lower))
        has_shorthand = has_vs_number or any(marker in lower for marker in shorthand_markers)
        if not has_shorthand:
            return
        has_conditional_anchor = state_high.lower() in lower and state_low.lower() in lower
        has_probability_semantics = "probability" in lower or "chance" in lower
        has_event_verb = any(
            marker in lower
            for marker in (
                "appears",
                "occurs",
                "shows up",
                "is seen",
            )
        )
        if not (has_conditional_anchor and has_probability_semantics and has_event_verb):
            raise ValueError(
                "Prompt contains compressed reliability shorthand without explicit "
                f"conditional interpretation. Prompt='{lower}'"
            )

    def _validate_bayes_bias_wording(self, *, prompt: str) -> None:
        """Reject bias-mode prompts that explicitly name cognitive mechanisms."""
        lower = prompt.lower()
        forbidden_markers = (
            "anchoring",
            "cognitive risk",
            "cognitive bias",
            "first-impression bias",
            "first impression bias",
            "bias risk",
        )
        hit = next((marker for marker in forbidden_markers if marker in lower), None)
        if hit is None:
            return
        raise ValueError(
            "Bias-eliciting Bayes prompt explicitly names a cognitive mechanism "
            f"('{hit}'). Prompt='{lower}'"
        )

    def _qa_validate_rendered_prompt(
        self,
        *,
        task_subtype: TaskSubtype,
        prompt: str,
        problem_spec: ProblemSpec,
        frame_variant: PromptFrameVariant,
    ) -> list[dict[str, str]]:
        failures = self._prompt_qa_generic_failures(prompt=prompt)
        lower_prompt = prompt.lower()
        assumptions = problem_spec["assumptions"]
        labels = self._context_labels(frame_variant)
        observed_signal = assumptions.get("observed_signal")
        if observed_signal not in {"high", "low"}:
            observed_signal = "high"
        observed_sentence = self._observed_signal_sentence(
            labels=labels,
            signal=observed_signal,
        )
        prior_text = self._format_number(float(assumptions["prior_high"]))
        p_high_text = self._format_number(float(assumptions["p_signal_high_given_high"]))
        p_low_text = self._format_number(float(assumptions["p_signal_high_given_low"]))
        when_high = self._state_phrase(labels=labels, state="high", mode="when")
        when_low = self._state_phrase(labels=labels, state="low", mode="when")
        among_high = self._state_phrase(labels=labels, state="high", mode="among")
        among_low = self._state_phrase(labels=labels, state="low", mode="among")

        if task_subtype in {"basic_bayes_update", "binary_signal_decision"}:
            if prior_text not in prompt:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_prior_base_rate",
                        detail=f"Prior/base rate value '{prior_text}' not found in prompt.",
                    )
                )
            if not self._binary_prompt_has_observed_signal(
                prompt=prompt,
                signal=observed_signal,
                observed_sentence=observed_sentence,
            ):
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_observed_cue",
                        detail="Observed cue/signal not found in rendered prompt.",
                    )
                )
            if p_high_text not in prompt or p_low_text not in prompt:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_cue_rate_pair",
                        detail="One or both conditional cue rates are missing.",
                    )
                )
            has_when_pair = (
                when_high.lower() in prompt.lower() and when_low.lower() in prompt.lower()
            )
            has_among_pair = (
                among_high.lower() in prompt.lower() and among_low.lower() in prompt.lower()
            )
            has_state_short_pair = (
                labels["high_short"].lower() in prompt.lower()
                and labels["low_short"].lower() in prompt.lower()
            )
            if not has_when_pair and not has_among_pair and not has_state_short_pair:
                failures.append(
                    self._prompt_qa_failure(
                        code="uninterpretable_conditional_rates",
                        detail=(
                            "Cue rates are present but not clearly anchored to both "
                            "high and low conditional contexts."
                        ),
                    )
                )
            try:
                self._validate_reliability_wording_lint(
                    prompt=prompt,
                    state_high=labels["high_short"],
                    state_low=labels["low_short"],
                )
            except ValueError as exc:
                failures.append(
                    self._prompt_qa_failure(
                        code="compressed_reliability_shorthand",
                        detail=str(exc),
                    )
                )

        if task_subtype == "binary_signal_decision":
            option_a = problem_spec["options"]["A"]
            option_b = problem_spec["options"]["B"]
            payoff_high = self._format_number(float(option_a["payoff_if_high"]))
            payoff_low = self._format_number(float(option_a["payoff_if_low"]))
            wait_payoff = self._format_number(float(option_b["payoff"]))
            if payoff_high not in prompt or payoff_low not in prompt or wait_payoff not in prompt:
                failures.append(
                    self._prompt_qa_failure(
                        code="missing_action_payoffs",
                        detail=(
                            "binary_signal_decision prompt must contain act high/low payoffs "
                            "and do_not_act payoff."
                        ),
                    )
                )
            unnatural_fragment_patterns = (
                r"\bact\s*[:=]\s*-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?",
                r"\b(?:hold|wait)\s*[:=]\s*-?\d+(?:\.\d+)?",
                r"\bdecision row\b",
                r"\bpayoff pair=",
                r"\bresult rate=",
                r"\bdo_not_act=",
            )
            for pattern in unnatural_fragment_patterns:
                if re.search(pattern, prompt.lower()):
                    failures.append(
                        self._prompt_qa_failure(
                            code="binary_telegraphic_fragment_notation",
                            detail=(
                                "binary_signal_decision prompt includes fragment-heavy "
                                f"notation matching '{pattern}'."
                            ),
                        )
                    )
            compressed_payoff_pair = re.compile(r"\b-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?\b")
            if compressed_payoff_pair.search(lower_prompt):
                failures.append(
                    self._prompt_qa_failure(
                        code="binary_compressed_payoff_shorthand",
                        detail=(
                            "binary_signal_decision prompt contains compressed payoff shorthand "
                            "(e.g., '136/-26') instead of explicit state-conditioned outcomes."
                        ),
                    )
                )
        return failures

    def _state_phrase(
        self, *, labels: dict[str, str], state: Literal["high", "low"], mode: str
    ) -> str:
        """Return grammar-safe state phrasing for interpolation in templates."""
        if mode == "when":
            key = f"when_{state}"
        elif mode == "among":
            key = f"among_{state}_cases"
        else:
            key = state
        if key in labels:
            return labels[key]
        return labels[state]

    def _render_cascade_history(self, *, public_actions: list[str], labels: dict[str, str]) -> str:
        if not public_actions:
            return "no prior calls"
        mapped: list[str] = []
        for action in public_actions:
            if action == "choose_high":
                mapped.append(labels["history_high"])
            elif action == "choose_low":
                mapped.append(labels["history_low"])
            else:
                mapped.append(action)
        return ", ".join(mapped)

    def _render_basic_bayes_update_prompt(
        self,
        *,
        problem_spec: BasicBayesUpdateProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        context = prompt_frame_variant or "medical_screening"
        labels = self._context_labels(context)
        observed_sentence = self._observed_signal_sentence(labels=labels, signal=signal)
        when_high = self._state_phrase(labels=labels, state="high", mode="when")
        when_low = self._state_phrase(labels=labels, state="low", mode="when")
        among_high = self._state_phrase(labels=labels, state="high", mode="among")
        among_low = self._state_phrase(labels=labels, state="low", mode="among")
        prevalence_text = f"About {prior} of cases are ones where {labels['high_short']}."
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        explicit_reliability_sentence = self._select_explicit_reliability_clause(
            task_subtype="basic_bayes_update",
            frame_variant=context,
            tier=tier,
            problem_spec=problem_spec,
            p_high_text=p_sh_h,
            p_low_text=p_sh_l,
            state_high=labels["high_short"],
            state_low=labels["low_short"],
        ).rstrip(".")
        explicit_reliability = (
            explicit_reliability_sentence[0].lower() + explicit_reliability_sentence[1:]
        )
        if tier == "formal":
            templates = [
                (
                    f"Compute then compare: prior={prior}, "
                    f"P(signal=high|high)={p_sh_h}, P(signal=high|low)={p_sh_l}, "
                    f"observed signal={signal}. Apply Bayes updating and select the "
                    "higher-probability state."
                ),
                (
                    f"Evaluate posterior choice for this {context} case. "
                    f"Observation={signal}; prior={prior}; {explicit_reliability}. "
                    "Choose the state with larger posterior probability."
                ),
                (
                    f"Decision sheet: base rate is {prior} and observed signal={signal}. "
                    f"Here, {explicit_reliability}. "
                    "Evaluate posterior probabilities and determine the more likely state."
                ),
                (
                    f"Analytic prompt ({context}): observed signal is {signal}. "
                    f"Given prior {prior} and {explicit_reliability}, "
                    "compute the posterior distribution and select the higher-probability state."
                ),
                (
                    f"Posterior ranking task: prior high={prior}; "
                    f"P(signal=high|high)={p_sh_h}; P(signal=high|low)={p_sh_l}; "
                    f"signal observed={signal}. Determine which state is more probable."
                ),
                (
                    f"Evaluate posterior choice ({context}): start from prior={prior}, "
                    f"apply likelihoods {p_sh_h} and {p_sh_l}, and condition on "
                    f"observed signal={signal}. Select the higher-probability state."
                ),
                (
                    f"Rank by posterior probability. Observed signal={signal}; "
                    f"base rate is {prior}; {explicit_reliability}. "
                    "Which state should rank first?"
                ),
                (
                    f"Bayesian update worksheet: first record observation ({signal}), then use "
                    f"prior {prior} with the conditionals where {explicit_reliability}. "
                    "Compute posteriors and choose the more likely state."
                ),
                (
                    f"Model-selection card ({labels['entity']}): the observed signal is "
                    f"{signal}. With base rate {prior} and {explicit_reliability}, "
                    "determine which state has larger posterior mass."
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="basic_bayes_update",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        elif tier == "neutral_natural":
            # Neutral prompts rotate discourse order (observation/base-rate/
            # likelihood/question placement) instead of wrapper labels.
            templates = [
                (
                    f"In this {labels['entity']}, {observed_sentence.lower()} "
                    f"About {prior} of cases are "
                    f"ones where {labels['high_short']}. This result appears with probability "
                    f"{p_sh_h} {when_high} and {p_sh_l} {when_low}. "
                    "Which state now seems more likely?"
                ),
                (
                    f"{observed_sentence} {prevalence_text} "
                    f"The chance of this result is {p_sh_h} {when_high} and "
                    f"{p_sh_l} {when_low}. Which state should you treat as likelier?"
                ),
                (
                    f"{observed_sentence} The signal occurs with probability "
                    f"{p_sh_h} {when_high} and {p_sh_l} {when_low}. "
                    f"{prevalence_text} Deciding now, which state should you select?"
                ),
                (
                    f"Case review: about {prior} of these {labels['entity']} cases are ones where "
                    f"{labels['high_short']}. {observed_sentence} "
                    f"You see this result with probability {p_sh_h} {among_high} and "
                    f"{p_sh_l} {among_low}. Which state now looks likelier?"
                ),
                (
                    f"Deciding now on this {labels['entity']} case: the base rate is {prior} "
                    f"for {labels['high_short']}, and {observed_sentence.lower()} "
                    f"(probability {p_sh_h} {when_high}; {p_sh_l} {when_low}), "
                    "which state is more likely?"
                ),
                (
                    f"Start from what you received: {observed_sentence} "
                    f"Among cases where {labels['high_short']}, "
                    f"this result is returned at {p_sh_h}; "
                    f"among the alternative cases, it is returned at {p_sh_l}. "
                    f"The background share is {prior} across all cases. "
                    "Which state is the stronger current read?"
                ),
                (
                    f"{observed_sentence} {prevalence_text} "
                    f"This result is returned at rate {p_sh_h} "
                    f"{when_high} and {p_sh_l} {when_low}. "
                    "Which state should you mark as more likely?"
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="basic_bayes_update",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        else:
            templates = [
                (
                    f"The visible cue is hard to ignore: {observed_sentence} "
                    "Which state seems likelier right now? "
                    f"Base rate is {prior}. {explicit_reliability_sentence}."
                ),
                (
                    f"This signal can look decisive. {observed_sentence} "
                    f"{prevalence_text} {explicit_reliability_sentence}. "
                    "Without a full review, which state do you mark as more likely?"
                ),
                (
                    f"Queue pressure is building. Start with the visible cue: "
                    f"{observed_sentence} "
                    f"Base rate is {prior}. {explicit_reliability_sentence}. "
                    "Which state do you mark in triage?"
                ),
                (
                    f"Operational triage note: {observed_sentence} "
                    f"{explicit_reliability_sentence}. {prevalence_text} "
                    "Given limited time, which state looks likelier?"
                ),
                (
                    f"Attention is split across cases. {observed_sentence} "
                    f"Base rate is {prior}. {explicit_reliability_sentence}. "
                    "With a quick pass, which state appears more likely?"
                ),
                (
                    "A strong signal can look one-sided. "
                    f"{observed_sentence} {explicit_reliability_sentence}. "
                    f"The base rate is {prior}. "
                    "Before deeper analysis, which state do you pick?"
                ),
                (
                    f"The desk needs a provisional call now. "
                    f"{observed_sentence} {prevalence_text} "
                    f"{explicit_reliability_sentence}. "
                    "Which state do you select for now?"
                ),
                (
                    f"This cue can look decisive in this case. {observed_sentence} "
                    f"The base rate is {prior}. {explicit_reliability_sentence}. "
                    "What state does your current read favor?"
                ),
                (
                    f"You have only partial attention on this case. "
                    f"Before a full breakdown, {observed_sentence} "
                    f"The base rate is {prior}, and {explicit_reliability}. "
                    "Which state looks more likely at this point?"
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="basic_bayes_update",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
            self._validate_bayes_bias_wording(prompt=body)
        body = self._normalize_reliability_wording(
            prompt=body,
            p_high_text=p_sh_h,
            p_low_text=p_sh_l,
            state_high=labels["high_short"],
            state_low=labels["low_short"],
        )
        self._validate_reliability_wording_lint(
            prompt=body,
            state_high=labels["high_short"],
            state_low=labels["low_short"],
        )
        if style == "unlabeled":
            return body
        return f"{body}\n- choose_state_high\n- choose_state_low"

    def _render_binary_signal_decision_prompt(
        self,
        *,
        problem_spec: BinarySignalDecisionProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        context = prompt_frame_variant or "medical_screening"
        labels = self._context_labels(context)
        observed_sentence = self._observed_signal_sentence(labels=labels, signal=signal)
        among_high = self._state_phrase(labels=labels, state="high", mode="among")
        among_low = self._state_phrase(labels=labels, state="low", mode="among")
        payoff_high = self._format_number(option_a["payoff_if_high"])
        payoff_low = self._format_number(option_a["payoff_if_low"])
        wait_payoff = self._format_number(option_b["payoff"])
        prevalence_text = f"The prior probability is {prior}."
        payoff_state_terms_by_context = {
            "medical_screening": ("condition-present", "condition-absent"),
            "fraud_detection": ("fraudulent", "legitimate"),
            "hiring_screen": ("strong-fit", "not-strong-fit"),
            "security_alert": ("active-threat", "no-threat"),
            "trading_signal": ("bullish-regime", "bearish-regime"),
            "manufacturing_defect": ("defect-heavy", "normal-quality"),
        }
        payoff_high_term, payoff_low_term = payoff_state_terms_by_context.get(
            context, ("high-state", "low-state")
        )
        payoff_sentence_compact = (
            f"Act pays {payoff_high} if {payoff_high_term}, "
            f"{payoff_low} if {payoff_low_term}; do_not_act pays {wait_payoff}."
        )
        payoff_sentence_formal = (
            f"Act pays {payoff_high} if {payoff_high_term}, "
            f"{payoff_low} if {payoff_low_term}; do_not_act pays {wait_payoff}."
        )
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        reliability_templates = [
            (
                f"This signal is seen with probability {p_sh_h} {among_high} "
                f"and {p_sh_l} {among_low}."
            ),
            (
                f"The signal occurs with probability {p_sh_h} {among_high} "
                f"and {p_sh_l} {among_low}."
            ),
            (
                f"A result like this shows up with probability {p_sh_h} {among_high} "
                f"and {p_sh_l} {among_low}."
            ),
            f"The chance of this signal is {p_sh_h} {among_high} and {p_sh_l} {among_low}.",
            (
                f"This result is returned at probability {p_sh_h} {among_high} "
                f"and {p_sh_l} {among_low}."
            ),
            (
                f"The rate for this signal is {p_sh_h} {among_high} "
                f"and {p_sh_l} {among_low}."
            ),
        ]
        reliability_idx = self._select_template_variant(
            task_subtype="binary_signal_decision",
            frame_variant=context,
            tier=f"{tier}_reliability",
            problem_spec=problem_spec,
            templates=reliability_templates,
        )
        reliability_sentence_compact = reliability_templates[reliability_idx]
        if tier == "formal":
            templates = [
                (
                    f"Prior={prior}; observed signal={signal}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_formal} "
                    "Apply Bayes updating and choose the higher posterior expected payoff action."
                ),
                (
                    f"Compute then compare: prior {prior}, signal {signal}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_formal} "
                    "Evaluate posterior expected payoff for each action and select the larger."
                ),
                (
                    f"Formal evaluation: prior {prior}, observed {signal}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_formal} "
                    "Determine which action has larger posterior expected payoff."
                ),
                (
                    f"Posterior choice: prior={prior}, observed={signal}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_formal} "
                    "Compare posterior expected payoffs and pick the better action."
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="binary_signal_decision",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        elif tier == "neutral_natural":
            templates = [
                (
                    f"{observed_sentence} Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} Deciding now, what do you do?"
                ),
                (
                    f"Base rate is {prior}. {observed_sentence} {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} Which option is better now?"
                ),
                (
                    f"{observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} Should you act?"
                ),
                (
                    f"{payoff_sentence_compact} "
                    f"{observed_sentence} Base rate is {prior}. {reliability_sentence_compact} "
                    "Deciding now, which option is better?"
                ),
                (
                    f"{observed_sentence} In this {labels['entity']} case, base rate is {prior}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_compact} Which action should you take now?"
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="binary_signal_decision",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
        else:
            templates = [
                (
                    f"The signal stands out on this case. {observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Which action do you lean toward?"
                ),
                (
                    f"This may look decisive from the cue alone. {observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Which action seems right on this pass?"
                ),
                (
                    f"Queue pressure is high before deeper analysis. {observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Which action do you put forward?"
                ),
                (
                    f"Operational handoff for this {labels['entity']} case: {observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Which action do you mark for now?"
                ),
                (
                    f"Limited-attention pass: {observed_sentence} "
                    f"{prevalence_text} {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Which move do you choose on this pass?"
                ),
                (
                    f"Provisional call before full analysis: {observed_sentence} "
                    f"Base rate is {prior}. {reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "What action does this point you toward?"
                ),
                (
                    f"The cue can look strongly one-sided. {observed_sentence} "
                    f"Base rate is {prior}. "
                    f"{reliability_sentence_compact} "
                    f"{payoff_sentence_compact} "
                    "Without a full breakdown, which option seems likelier to be right?"
                ),
            ]
            template_idx = self._select_template_variant(
                task_subtype="binary_signal_decision",
                frame_variant=context,
                tier=tier,
                problem_spec=problem_spec,
                templates=templates,
            )
            body = templates[template_idx]
            self._validate_bayes_bias_wording(prompt=body)
        body = self._normalize_reliability_wording(
            prompt=body,
            p_high_text=p_sh_h,
            p_low_text=p_sh_l,
            state_high=labels["high_short"],
            state_low=labels["low_short"],
        )
        self._validate_reliability_wording_lint(
            prompt=body,
            state_high=labels["high_short"],
            state_low=labels["low_short"],
        )

        if style == "unlabeled":
            self._validate_binary_signal_decision_prompt_completeness(
                prompt=body,
                prior_text=prior,
                observed_signal=signal,
                observed_sentence=observed_sentence,
                likelihood_high_text=p_sh_h,
                likelihood_low_text=p_sh_l,
                payoff_high_text=payoff_high,
                payoff_low_text=payoff_low,
                do_not_act_payoff_text=wait_payoff,
            )
            return body
        rendered = f"{body}\n- act\n- do_not_act"
        self._validate_binary_signal_decision_prompt_completeness(
            prompt=rendered,
            prior_text=prior,
            observed_signal=signal,
            observed_sentence=observed_sentence,
            likelihood_high_text=p_sh_h,
            likelihood_low_text=p_sh_l,
            payoff_high_text=payoff_high,
            payoff_low_text=payoff_low,
            do_not_act_payoff_text=wait_payoff,
        )
        return rendered

    def _render_information_cascade_prompt(
        self,
        *,
        problem_spec: InformationCascadeProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        context = prompt_frame_variant or "medical_screening"
        labels = self._context_labels(context)
        when_high = self._state_phrase(labels=labels, state="high", mode="when")
        when_low = self._state_phrase(labels=labels, state="low", mode="when")
        public_actions = assumptions.get("public_actions", [])
        history_text = self._render_cascade_history(public_actions=public_actions, labels=labels)
        observed_sentence = self._observed_signal_sentence(labels=labels, signal=signal)
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Prior P(high)={prior}. Signal model: P(signal=high|high)={p_sh_h}, "
                f"P(signal=high|low)={p_sh_l}. Public actions={history_text}. "
                f"Private signal={signal}. Apply Bayes updating over all evidence "
                "and choose choose_high or choose_low."
            )
        elif tier == "neutral_natural":
            body = (
                f"In this {labels['entity']}, earlier calls were: {history_text}. "
                f"{observed_sentence} "
                f"The signal rate is {p_sh_h} {when_high} and {p_sh_l} "
                f"{when_low} (baseline {prior}). "
                "Which state seems more likely now?"
            )
        else:
            body = (
                f"Recent calls are noticeable: {history_text}. {observed_sentence} "
                f"(Signal rates: {p_sh_h} {when_high}, {p_sh_l} "
                f"{when_low}; baseline {prior}.) "
                "Which state would you choose?"
            )

        if style == "unlabeled":
            return body
        return f"{body}\n- choose_high\n- choose_low"

    def _render_noisy_signal_asset_update_prompt(
        self,
        *,
        problem_spec: NoisySignalAssetUpdateProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        context = prompt_frame_variant or "medical_screening"
        labels = self._context_labels(context)
        observed_sentence = self._observed_signal_sentence(labels=labels, signal=signal)
        when_high = self._state_phrase(labels=labels, state="high", mode="when")
        when_low = self._state_phrase(labels=labels, state="low", mode="when")
        value_high = self._format_number(option_a["value_if_high"])
        value_low = self._format_number(option_a["value_if_low"])
        market_price = self._format_number(option_a["market_price"])
        transaction_cost = self._format_number(float(assumptions.get("transaction_cost", 0.0)))
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Prior P(high)={prior}. Signal model: P(signal=high|high)={p_sh_h}, "
                f"P(signal=high|low)={p_sh_l}, observed signal={signal}. "
                f"Contract value is {value_high} in high and {value_low} in low. "
                f"Entry price is {market_price} and transaction cost is {transaction_cost}. "
                "Apply Bayes updating and choose buy or do_not_buy."
            )
        elif tier == "neutral_natural":
            body = (
                f"You can buy a contingent contract tied to this {labels['entity']}. "
                f"Baseline chance that {labels['high_short']} is {prior}. "
                f"{observed_sentence} "
                f"If high, contract value is {value_high}; if low, value is {value_low}. "
                f"Price is {market_price} and transaction cost is {transaction_cost}. "
                "Should you buy?"
            )
        else:
            body = (
                f"{observed_sentence} "
                f"This signal appears with chance {p_sh_h} {when_high} and "
                f"{p_sh_l} {when_low}. "
                f"The contract pays {value_high} in high and {value_low} in low; "
                f"entry price {market_price}, transaction cost {transaction_cost}. "
                f"(Baseline rate: {prior}.) Buy?"
            )

        if style == "unlabeled":
            return body
        return f"{body}\n- buy\n- do_not_buy"

    def _build_basic_bayes_outcome_model(
        self, *, problem_spec: BasicBayesUpdateProblemSpec
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        prior = assumptions["prior_high"]
        p_sh_h = assumptions["p_signal_high_given_high"]
        p_sh_l = assumptions["p_signal_high_given_low"]
        signal = assumptions["observed_signal"]
        like_h = p_sh_h if signal == "high" else (1 - p_sh_h)
        like_l = p_sh_l if signal == "high" else (1 - p_sh_l)
        return {
            "choose_state_high": (
                f"({self._format_number(prior)} * {self._format_number(like_h)})"
                f" / ({self._format_number(prior)} * {self._format_number(like_h)}"
                f" + (1 - {self._format_number(prior)}) * {self._format_number(like_l)})"
            ),
            "choose_state_low": "1 - choose_state_high",
        }

    def _build_binary_signal_decision_outcome_model(
        self, *, problem_spec: BinarySignalDecisionProblemSpec
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        prior = assumptions["prior_high"]
        p_sh_h = assumptions["p_signal_high_given_high"]
        p_sh_l = assumptions["p_signal_high_given_low"]
        signal = assumptions["observed_signal"]
        like_h = p_sh_h if signal == "high" else (1 - p_sh_h)
        like_l = p_sh_l if signal == "high" else (1 - p_sh_l)
        posterior_formula = (
            f"({self._format_number(prior)} * {self._format_number(like_h)})"
            f" / ({self._format_number(prior)} * {self._format_number(like_h)}"
            f" + (1 - {self._format_number(prior)}) * {self._format_number(like_l)})"
        )
        return {
            "act": (
                f"({posterior_formula}) * {self._format_number(option_a['payoff_if_high'])}"
                f" + (1 - ({posterior_formula})) * {self._format_number(option_a['payoff_if_low'])}"
            ),
            "do_not_act": self._format_number(option_b["payoff"]),
        }

    def _build_information_cascade_outcome_model(
        self, *, problem_spec: InformationCascadeProblemSpec
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        prior = self._format_number(assumptions["prior_high"])
        history = assumptions.get("public_actions", [])
        private_signal = assumptions["observed_signal"]
        signal_terms: list[str] = []
        for action in history:
            signal_terms.append("high" if action == "choose_high" else "low")
        signal_terms.append(private_signal)
        return {
            "choose_high": (
                "BayesPosterior("
                f"prior={prior},"
                f"signals={signal_terms},"
                f"p_sh_h={self._format_number(assumptions['p_signal_high_given_high'])},"
                f"p_sh_l={self._format_number(assumptions['p_signal_high_given_low'])}"
                ")"
            ),
            "choose_low": "1 - choose_high",
        }

    def _build_noisy_asset_outcome_model(
        self, *, problem_spec: NoisySignalAssetUpdateProblemSpec
    ) -> dict[str, str]:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        prior = assumptions["prior_high"]
        p_sh_h = assumptions["p_signal_high_given_high"]
        p_sh_l = assumptions["p_signal_high_given_low"]
        signal = assumptions["observed_signal"]
        like_h = p_sh_h if signal == "high" else (1 - p_sh_h)
        like_l = p_sh_l if signal == "high" else (1 - p_sh_l)
        posterior_formula = (
            f"({self._format_number(prior)} * {self._format_number(like_h)})"
            f" / ({self._format_number(prior)} * {self._format_number(like_h)}"
            f" + (1 - {self._format_number(prior)}) * {self._format_number(like_l)})"
        )
        transaction_cost = self._format_number(float(assumptions.get("transaction_cost", 0.0)))
        return {
            "buy": (
                f"({posterior_formula}) * {self._format_number(option_a['value_if_high'])}"
                f" + (1 - ({posterior_formula})) * {self._format_number(option_a['value_if_low'])}"
                f" - {self._format_number(option_a['market_price'])}"
                f" - {transaction_cost}"
            ),
            "do_not_buy": "0",
        }

    def _generate_basic_bayes_update(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        prior_high = round(self.rng.uniform(0.15, 0.85), 2)
        p_signal_high_given_high = round(self.rng.uniform(0.65, 0.95), 2)
        p_signal_high_given_low = round(self.rng.uniform(0.05, 0.35), 2)
        observed_signal: Literal["high", "low"] = self.rng.choice(["high", "low"])
        problem_spec = self._build_basic_bayes_update_problem_spec(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
        )
        (
            action_values,
            decision_values,
            optimal,
            posterior_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        comparison_pair = self._comparison_pair_for_subtype("basic_bayes_update")
        outcome_model = self._build_basic_bayes_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="basic_bayes_update",
            problem_spec=problem_spec,
            numeric_values=[
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_state_high"],
            right_value=decision_values["choose_state_low"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="basic_bayes_update",
            task_id_prefix="bayes_basic",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "observed_signal": observed_signal,
                "options": problem_spec["options"],
            },
            beliefs=posterior_beliefs,
            actions=["choose_state_high", "choose_state_low", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Updated probabilities are high={posterior_beliefs['posterior_high']} "
                f"and low={posterior_beliefs['posterior_low']}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_binary_signal_decision(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        prior_high = round(self.rng.uniform(0.15, 0.85), 2)
        p_signal_high_given_high = round(self.rng.uniform(0.6, 0.95), 2)
        p_signal_high_given_low = round(self.rng.uniform(0.05, 0.45), 2)
        observed_signal: Literal["high", "low"] = self.rng.choice(["high", "low"])
        payoff_if_high = self.rng.randint(40, 180)
        payoff_if_low = -self.rng.randint(10, 140)
        do_not_act_payoff = self.rng.choice([0, 5, 10, 15, 20])
        problem_spec = self._build_binary_signal_decision_problem_spec(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
            payoff_if_high=payoff_if_high,
            payoff_if_low=payoff_if_low,
            do_not_act_payoff=do_not_act_payoff,
        )
        (
            action_values,
            decision_values,
            optimal,
            posterior_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        comparison_pair = self._comparison_pair_for_subtype("binary_signal_decision")
        outcome_model = self._build_binary_signal_decision_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="binary_signal_decision",
            problem_spec=problem_spec,
            numeric_values=[
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
                payoff_if_high,
                payoff_if_low,
                do_not_act_payoff,
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["act"],
            right_value=decision_values["do_not_act"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                    payoff_if_high,
                    payoff_if_low,
                    do_not_act_payoff,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="binary_signal_decision",
            task_id_prefix="signal_decision",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "observed_signal": observed_signal,
                "options": problem_spec["options"],
            },
            beliefs=posterior_beliefs,
            actions=["act", "do_not_act", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Posterior expected payoffs are act={decision_values['act']} "
                f"and do_not_act={decision_values['do_not_act']}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_information_cascade_step(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        prior_high = round(self.rng.uniform(0.35, 0.65), 2)
        p_signal_high_given_high = round(self.rng.uniform(0.6, 0.9), 2)
        p_signal_high_given_low = round(self.rng.uniform(0.1, 0.4), 2)
        public_actions: list[Literal["choose_high", "choose_low"]] = []
        for _ in range(self.rng.randint(2, 4)):
            public_actions.append(self.rng.choice(["choose_high", "choose_low"]))
        observed_signal: Literal["high", "low"] = self.rng.choice(["high", "low"])
        problem_spec = self._build_information_cascade_problem_spec(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
            public_actions=public_actions,
        )
        (
            action_values,
            decision_values,
            optimal,
            posterior_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        comparison_pair = self._comparison_pair_for_subtype("information_cascade_step")
        outcome_model = self._build_information_cascade_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="information_cascade_step",
            problem_spec=problem_spec,
            numeric_values=[
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
                len(public_actions),
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["choose_high"],
            right_value=decision_values["choose_low"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                    len(public_actions),
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            evidence_count=len(public_actions) + 1,
            prompt_complexity_features=prompt_complexity_features,
        )
        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="information_cascade_step",
            task_id_prefix="cascade",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "public_actions": public_actions,
                "private_signal": observed_signal,
                "options": problem_spec["options"],
            },
            beliefs=posterior_beliefs,
            actions=["choose_high", "choose_low", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"After combining public and private evidence, probabilities are "
                f"high={posterior_beliefs['posterior_high']} and "
                f"low={posterior_beliefs['posterior_low']}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_noisy_signal_asset_update(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        prior_high = round(self.rng.uniform(0.15, 0.85), 2)
        p_signal_high_given_high = round(self.rng.uniform(0.6, 0.95), 2)
        p_signal_high_given_low = round(self.rng.uniform(0.05, 0.45), 2)
        observed_signal: Literal["high", "low"] = self.rng.choice(["high", "low"])
        value_if_high = self.rng.randint(130, 280)
        value_if_low = self.rng.randint(20, 120)
        market_price = round(self.rng.uniform(50, 230), 2)
        transaction_cost = float(self.rng.choice([0, 1, 2, 3, 5]))
        problem_spec = self._build_noisy_signal_asset_update_problem_spec(
            prior_high=prior_high,
            p_signal_high_given_high=p_signal_high_given_high,
            p_signal_high_given_low=p_signal_high_given_low,
            observed_signal=observed_signal,
            value_if_high=value_if_high,
            value_if_low=value_if_low,
            market_price=market_price,
            transaction_cost=transaction_cost,
        )
        (
            action_values,
            decision_values,
            optimal,
            posterior_beliefs,
        ) = self._solve_from_problem_spec(problem_spec=problem_spec)
        comparison_pair = self._comparison_pair_for_subtype("noisy_signal_asset_update")
        outcome_model = self._build_noisy_asset_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = self._build_prompt_and_complexity(
            task_subtype="noisy_signal_asset_update",
            problem_spec=problem_spec,
            numeric_values=[
                prior_high,
                p_signal_high_given_high,
                p_signal_high_given_low,
                value_if_high,
                value_if_low,
                market_price,
                transaction_cost,
            ],
            comparison_pair=comparison_pair,
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=decision_values["buy"],
            right_value=decision_values["do_not_buy"],
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                    value_if_high,
                    value_if_low,
                    market_price,
                    transaction_cost,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(outcome_model),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="noisy_signal_asset_update",
            task_id_prefix="asset_update",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "observed_signal": observed_signal,
                "market_price": market_price,
                "options": problem_spec["options"],
            },
            beliefs=posterior_beliefs,
            actions=["buy", "do_not_buy", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values=action_values,
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Posterior expected payoffs are buy={decision_values['buy']} "
                f"and do_not_buy={decision_values['do_not_buy']}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )
