"""
Risk/loss/time dataset generator for normative decision tasks.

These samples are constructed as normative decision problems where the optimal
action follows expected-value maximization under linear utility; time tasks use
simple annual discounting to compare present values.
"""

import hashlib
import json
import logging
import random
import re
from collections import Counter
from typing import Any, Callable, Literal

from base_generator import BaseGenerator
from difficulty_config import RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE

from schema import (
    ActionScalars,
    AmbiguityAversionAssumptions,
    AmbiguityAversionChoiceProblemSpec,
    CeOfferComparisonProblemSpec,
    ComparisonPair,
    DataPoint,
    ExpectedValueAssumptions,
    HyperbolicDiscountingCounterexampleProblemSpec,
    LossAversionCounterexampleProblemSpec,
    LotteryProblemSpec,
    Metadata,
    MixedGainLossProblemSpec,
    ProbabilityWeightingCounterexampleProblemSpec,
    ProblemSpec,
    RiskLossTimeTaskSubtype,
    SolverTrace,
    Target,
    TimeDiscountingAssumptions,
    TimeDiscountingProblemSpec,
)

TaskSubtype = RiskLossTimeTaskSubtype
DifficultyMetrics = dict[str, float | int | bool | str]
ActionValueSemantics = Literal[
    "expected_value_comparison",
    "discounted_value_comparison",
]
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
    "gain_focus",
    "loss_focus",
    "safety_focus",
    "upside_focus",
    "single_shot",
    "repeated_play",
    "investing_context",
    "everyday_money_context",
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
SUPPORTED_PROMPT_FRAME_VARIANTS: tuple[PromptFrameVariant, ...] = (
    "auto",
    "gain_focus",
    "loss_focus",
    "safety_focus",
    "upside_focus",
    "single_shot",
    "repeated_play",
    "investing_context",
    "everyday_money_context",
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
COMPARISON_PAIR_BY_SUBTYPE: dict[TaskSubtype, ComparisonPair] = {
    "lottery_choice": {
        "left_action": "choose_lottery",
        "right_action": "choose_sure",
    },
    "ce_offer_comparison": {
        "left_action": "accept_offer",
        "right_action": "reject_offer",
    },
    "mixed_gain_loss_choice": {
        "left_action": "choose_risky",
        "right_action": "choose_sure",
    },
    "time_discounting": {
        "left_action": "choose_later",
        "right_action": "choose_now",
    },
    "ambiguity_aversion_choice": {
        "left_action": "choose_ambiguous",
        "right_action": "choose_known_risk",
    },
    "probability_weighting_counterexample": {
        "left_action": "choose_longshot",
        "right_action": "choose_sure",
    },
    "loss_aversion_counterexample": {
        "left_action": "accept_gamble",
        "right_action": "choose_status_quo",
    },
    "hyperbolic_discounting_counterexample": {
        "left_action": "choose_later",
        "right_action": "choose_earlier",
    },
}
PROMPT_RENDERER_METHOD_BY_SUBTYPE: dict[TaskSubtype, str] = {
    "lottery_choice": "_render_lottery_choice_prompt",
    "ce_offer_comparison": "_render_ce_offer_comparison_prompt",
    "mixed_gain_loss_choice": "_render_mixed_gain_loss_choice_prompt",
    "time_discounting": "_render_time_discounting_prompt",
    "ambiguity_aversion_choice": "_render_ambiguity_aversion_choice_prompt",
    "probability_weighting_counterexample": "_render_probability_weighting_counterexample_prompt",
    "loss_aversion_counterexample": "_render_loss_aversion_counterexample_prompt",
    "hyperbolic_discounting_counterexample": "_render_hyperbolic_discounting_counterexample_prompt",
}
ACTION_VALUE_SEMANTICS_BY_SUBTYPE: dict[TaskSubtype, ActionValueSemantics] = {
    "lottery_choice": "expected_value_comparison",
    "ce_offer_comparison": "expected_value_comparison",
    "mixed_gain_loss_choice": "expected_value_comparison",
    "time_discounting": "discounted_value_comparison",
    "ambiguity_aversion_choice": "expected_value_comparison",
    "probability_weighting_counterexample": "expected_value_comparison",
    "loss_aversion_counterexample": "expected_value_comparison",
    "hyperbolic_discounting_counterexample": "discounted_value_comparison",
}
PROBABILITY_FIELD_KEYS = {
    "p_win",
    "p_gain",
    "known_probability",
    "subjective_ambiguous_win_probability",
    "baseline_rate",
    "extreme_threshold",
    "probability",
}
logger = logging.getLogger(__name__)


class RiskLossTimeGenerator(BaseGenerator):
    """Generate normative expected-value decision tasks for risk/loss/time choices."""
    EXPECTED_UTILITY_PRECISION = 4
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
        self._last_prompt_frame_variant: PromptFrameVariant = "gain_focus"
        self._last_prompt_style_regime: PromptStyleRegime = "neutral_realistic"
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

    def _diversify_body_template(
        self,
        *,
        task_subtype: TaskSubtype,
        frame_variant: PromptFrameVariant | None,
        tier: str,
        problem_spec: ProblemSpec,
        body: str,
    ) -> str:
        frame_key = frame_variant or "auto"
        variants: list[str] = [body]
        sentence_parts = body.split(". ", 1)
        if len(sentence_parts) == 2:
            first = sentence_parts[0].strip()
            rest = sentence_parts[1].strip()
            if first and rest and "?" not in first and "?" not in rest:
                rest = rest[:-1] if rest.endswith(".") else rest
                variants.append(f"{rest}. {first}.")
        variants.append(f"Decision brief: {body}")
        # preserve order while deduplicating
        deduped = list(dict.fromkeys(variants))
        idx = self._select_template_variant(
            task_subtype=task_subtype,
            frame_variant=frame_key,
            tier=tier,
            problem_spec=problem_spec,
            templates=deduped,
        )
        return deduped[idx]

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
            "gain_focus",
            "loss_focus",
            "safety_focus",
            "upside_focus",
            "single_shot",
            "repeated_play",
            "investing_context",
            "everyday_money_context",
        )

    def _apply_prompt_frame_variant(
        self,
        *,
        prompt: str,
        frame_variant: PromptFrameVariant,
        task_subtype: TaskSubtype,
    ) -> str:
        _ = task_subtype
        lead_by_variant = {
            "gain_focus": (
                "You are choosing between ways to come out ahead financially."
            ),
            "loss_focus": (
                "You are trying to avoid ending up with less money."
            ),
            "safety_focus": (
                "You care most about keeping outcomes stable and avoiding unpleasant surprises."
            ),
            "upside_focus": (
                "You are focused on the chance of a bigger upside payoff."
            ),
            "single_shot": (
                "This is a one-time decision that will be settled once."
            ),
            "repeated_play": (
                "Think of this as a choice you could face repeatedly in similar conditions."
            ),
            "investing_context": (
                "Frame this as choosing between a safer allocation and a riskier position."
            ),
            "everyday_money_context": (
                "Frame this as an everyday money decision from a household budget perspective."
            ),
        }
        lead = lead_by_variant.get(frame_variant, "")
        if not lead:
            return prompt
        return f"{lead} {prompt}"

    def generate(self) -> DataPoint:
        current_index = self.sample_index
        self.sample_index += 1
        subtype: TaskSubtype = self.rng.choice([
            "lottery_choice",
            "ce_offer_comparison",
            "mixed_gain_loss_choice",
            "time_discounting",
            "ambiguity_aversion_choice",
            "probability_weighting_counterexample",
            "loss_aversion_counterexample",
            "hyperbolic_discounting_counterexample",
        ])

        if subtype == "lottery_choice":
            return self._generate_lottery_choice(sample_index=current_index)
        elif subtype == "ce_offer_comparison":
            return self._generate_ce_offer_comparison(sample_index=current_index)
        elif subtype == "mixed_gain_loss_choice":
            return self._generate_mixed_gain_loss_choice(sample_index=current_index)
        elif subtype == "time_discounting":
            return self._generate_time_discounting(sample_index=current_index)
        elif subtype == "ambiguity_aversion_choice":
            return self._generate_ambiguity_aversion_choice(sample_index=current_index)
        elif subtype == "probability_weighting_counterexample":
            return self._generate_probability_weighting_counterexample(
                sample_index=current_index
            )
        elif subtype == "loss_aversion_counterexample":
            return self._generate_loss_aversion_counterexample(sample_index=current_index)
        elif subtype == "hyperbolic_discounting_counterexample":
            return self._generate_hyperbolic_discounting_counterexample(
                sample_index=current_index
            )
        else:
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
        prompt_has_action_labels = resolved_prompt_style != "unlabeled"
        return Metadata(
            generator_name=self.__class__.__name__,
            version=self.version,
            seed=self.base_seed,
            dataset_role="normative_training",
            requested_prompt_style=self.prompt_style,
            resolved_prompt_style=resolved_prompt_style,
            prompt_style_regime=prompt_style_regime or self._last_prompt_style_regime,
            prompt_frame_variant=prompt_frame_variant,
            prompt_has_action_labels=prompt_has_action_labels,
            example_fingerprint=example_fingerprint,
            tie_threshold=tie_threshold,
            sample_index=sample_index,
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
                        raise ValueError(
                            f"{nested_path} must be in [0, 1], got {nested}."
                        )
                self._assert_probabilities_in_unit_interval(nested, path=nested_path)
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                self._assert_probabilities_in_unit_interval(
                    nested, path=f"{path}[{index}]"
                )

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

    def _prompt_renderer_for_subtype(
        self, subtype: TaskSubtype
    ) -> Callable[..., str]:
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
            raise ValueError(
                f"{field_path} must be a mapping for subtype '{subtype}'."
            )
        return value

    def _require_numeric_field(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> float:
        if key not in mapping:
            raise ValueError(f"{field_path}.{key} is required for subtype '{subtype}'.")
        value = mapping[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{field_path}.{key} must be numeric for subtype '{subtype}'."
            )
        return float(value)

    def _require_probability_field(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> float:
        value = self._require_numeric_field(
            mapping, key=key, field_path=field_path, subtype=subtype
        )
        if value < 0 or value > 1:
            raise ValueError(
                f"{field_path}.{key} must be between 0 and 1 for subtype '{subtype}'."
            )
        return value

    def _require_non_negative_field(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> float:
        value = self._require_numeric_field(
            mapping, key=key, field_path=field_path, subtype=subtype
        )
        if value < 0:
            raise ValueError(
                f"{field_path}.{key} must be non-negative for subtype '{subtype}'."
            )
        return value

    def _solve_from_problem_spec(
        self, *, problem_spec: ProblemSpec
    ) -> tuple[ActionScalars, ActionScalars, str]:
        subtype_raw = problem_spec.get("task_subtype")
        if subtype_raw not in COMPARISON_PAIR_BY_SUBTYPE:
            raise ValueError(
                "problem_spec.task_subtype must be one of "
                f"{sorted(COMPARISON_PAIR_BY_SUBTYPE.keys())}; got {subtype_raw!r}."
            )
        subtype = subtype_raw
        options = self._require_mapping(
            problem_spec.get("options"),
            field_path="problem_spec.options",
            subtype=subtype,
        )
        assumptions = self._require_mapping(
            problem_spec.get("assumptions"),
            field_path="problem_spec.assumptions",
            subtype=subtype,
        )

        if subtype == "lottery_choice":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            sure_amount = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_win = self._require_probability_field(
                option_b, key="p_win", field_path="problem_spec.options.B", subtype=subtype
            )
            win_amount = self._require_numeric_field(
                option_b,
                key="win_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            lose_amount = self._require_numeric_field(
                option_b,
                key="lose_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_sure = self._round_expected_utility(sure_amount)
            choose_lottery = self._round_expected_utility(
                p_win * win_amount + (1 - p_win) * lose_amount
            )
            action_values = ActionScalars(
                {
                    "choose_sure": choose_sure,
                    "choose_lottery": choose_lottery,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_lottery",
                left_value=choose_lottery,
                right_label="choose_sure",
                right_value=choose_sure,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "ce_offer_comparison":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            offered = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_win = self._require_probability_field(
                option_b, key="p_win", field_path="problem_spec.options.B", subtype=subtype
            )
            high = self._require_numeric_field(
                option_b,
                key="high",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            low = self._require_numeric_field(
                option_b,
                key="low",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            accept_offer = self._round_expected_utility(offered)
            reject_offer = self._round_expected_utility(
                p_win * high + (1 - p_win) * low
            )
            action_values = ActionScalars(
                {
                    "accept_offer": accept_offer,
                    "reject_offer": reject_offer,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="accept_offer",
                left_value=accept_offer,
                right_label="reject_offer",
                right_value=reject_offer,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "mixed_gain_loss_choice":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            sure = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_gain = self._require_probability_field(
                option_b, key="p_gain", field_path="problem_spec.options.B", subtype=subtype
            )
            gain = self._require_numeric_field(
                option_b,
                key="gain",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            loss = self._require_numeric_field(
                option_b,
                key="loss",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_sure = self._round_expected_utility(sure)
            choose_risky = self._round_expected_utility(
                p_gain * gain + (1 - p_gain) * loss
            )
            action_values = ActionScalars(
                {
                    "choose_sure": choose_sure,
                    "choose_risky": choose_risky,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_risky",
                left_value=choose_risky,
                right_label="choose_sure",
                right_value=choose_sure,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "time_discounting":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            now = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            later_offer = self._require_numeric_field(
                option_b,
                key="amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            delay_days = self._require_non_negative_field(
                option_b,
                key="delay_days",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            annual_rate = self._require_non_negative_field(
                assumptions,
                key="annual_discount_rate",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_now = self._round_expected_utility(now)
            choose_later = self._round_expected_utility(
                later_offer / (1 + annual_rate * (delay_days / 365))
            )
            action_values = ActionScalars(
                {
                    "choose_now": choose_now,
                    "choose_later": choose_later,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_later",
                left_value=choose_later,
                right_label="choose_now",
                right_value=choose_now,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "ambiguity_aversion_choice":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            known_probability = self._require_probability_field(
                assumptions,
                key="known_probability",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            subjective_ambiguous_win_probability = self._require_probability_field(
                assumptions,
                key="subjective_ambiguous_win_probability",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            known_p_win = self._require_probability_field(
                option_a, key="p_win", field_path="problem_spec.options.A", subtype=subtype
            )
            known_win_amount = self._require_numeric_field(
                option_a,
                key="win_amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            known_lose_amount = self._require_numeric_field(
                option_a,
                key="lose_amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            ambiguous_win_amount = self._require_numeric_field(
                option_b,
                key="win_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            ambiguous_lose_amount = self._require_numeric_field(
                option_b,
                key="lose_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_known_risk = self._round_expected_utility(
                known_probability * known_win_amount
                + (1 - known_probability) * known_lose_amount
            )
            choose_ambiguous = self._round_expected_utility(
                subjective_ambiguous_win_probability * ambiguous_win_amount
                + (1 - subjective_ambiguous_win_probability) * ambiguous_lose_amount
            )
            # Intentional duplication guard: known probability is stored in both
            # assumptions and Option A so standalone option consumers and solver
            # assumptions stay aligned.
            if abs(known_p_win - known_probability) > tie_epsilon:
                raise ValueError(
                    "problem_spec.options.A.p_win must match "
                    "problem_spec.assumptions.known_probability for subtype "
                    "'ambiguity_aversion_choice' (intentional duplicated field)."
                )
            action_values = ActionScalars(
                {
                    "choose_known_risk": choose_known_risk,
                    "choose_ambiguous": choose_ambiguous,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_ambiguous",
                left_value=choose_ambiguous,
                right_label="choose_known_risk",
                right_value=choose_known_risk,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "probability_weighting_counterexample":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            sure_amount = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_win = self._require_probability_field(
                option_b, key="p_win", field_path="problem_spec.options.B", subtype=subtype
            )
            win_amount = self._require_numeric_field(
                option_b,
                key="win_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            lose_amount = self._require_numeric_field(
                option_b,
                key="lose_amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_sure = self._round_expected_utility(sure_amount)
            choose_longshot = self._round_expected_utility(
                p_win * win_amount + (1 - p_win) * lose_amount
            )
            action_values = ActionScalars(
                {
                    "choose_sure": choose_sure,
                    "choose_longshot": choose_longshot,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_longshot",
                left_value=choose_longshot,
                right_label="choose_sure",
                right_value=choose_sure,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "loss_aversion_counterexample":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            status_quo_amount = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            p_gain = self._require_probability_field(
                option_b, key="p_gain", field_path="problem_spec.options.B", subtype=subtype
            )
            gain = self._require_numeric_field(
                option_b,
                key="gain",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            loss = self._require_numeric_field(
                option_b,
                key="loss",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_status_quo = self._round_expected_utility(status_quo_amount)
            accept_gamble = self._round_expected_utility(
                p_gain * gain + (1 - p_gain) * loss
            )
            action_values = ActionScalars(
                {
                    "choose_status_quo": choose_status_quo,
                    "accept_gamble": accept_gamble,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="accept_gamble",
                left_value=accept_gamble,
                right_label="choose_status_quo",
                right_value=choose_status_quo,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        if subtype == "hyperbolic_discounting_counterexample":
            option_a = self._require_mapping(
                options.get("A"), field_path="problem_spec.options.A", subtype=subtype
            )
            option_b = self._require_mapping(
                options.get("B"), field_path="problem_spec.options.B", subtype=subtype
            )
            earlier_amount = self._require_numeric_field(
                option_a,
                key="amount",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            earlier_delay_days = self._require_non_negative_field(
                option_a,
                key="delay_days",
                field_path="problem_spec.options.A",
                subtype=subtype,
            )
            later_amount = self._require_numeric_field(
                option_b,
                key="amount",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            later_delay_days = self._require_non_negative_field(
                option_b,
                key="delay_days",
                field_path="problem_spec.options.B",
                subtype=subtype,
            )
            annual_rate = self._require_non_negative_field(
                assumptions,
                key="annual_discount_rate",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            tie_epsilon = self._require_non_negative_field(
                assumptions,
                key="tie_epsilon",
                field_path="problem_spec.assumptions",
                subtype=subtype,
            )
            choose_earlier = self._round_expected_utility(
                earlier_amount / (1 + annual_rate * (earlier_delay_days / 365))
            )
            choose_later = self._round_expected_utility(
                later_amount / (1 + annual_rate * (later_delay_days / 365))
            )
            action_values = ActionScalars(
                {
                    "choose_earlier": choose_earlier,
                    "choose_later": choose_later,
                }
            )
            decision_values = ActionScalars(dict(action_values))
            optimal_decision = self._choose_optimal_action(
                left_label="choose_later",
                left_value=choose_later,
                right_label="choose_earlier",
                right_value=choose_earlier,
                epsilon=tie_epsilon,
            )
            return action_values, decision_values, optimal_decision

        raise ValueError(f"Unhandled subtype in solver: {subtype}")

    def _verify_target_solution(
        self, *, problem_spec: ProblemSpec, target: Target, prompt: str = ""
    ) -> None:
        self._assert_probabilities_in_unit_interval(problem_spec)
        solved_action_values, solved_decision_values, solved_optimal_decision = (
            self._solve_from_problem_spec(problem_spec=problem_spec)
        )
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
            expected_keys = list(expected.keys())
            actual_keys = list(actual.keys())
            if expected_keys != actual_keys:
                raise ValueError(
                    f"{field_name} action ordering does not match solver output: "
                    f"expected {expected_keys}, got {actual_keys}."
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
            beliefs=dict(problem_spec["assumptions"]),
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
            task_family="risk_loss_time_choice",
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
        # TODO(difficulty): derive label from metrics such as ev_gap and
        # numeric_complexity while preserving stable thresholds.
        _ = difficulty_metrics
        return RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE[subtype]

    def _difficulty_metrics(
        self,
        *,
        left_value: float,
        right_value: float,
        numeric_complexity: int,
        time_horizon_days: int | None = None,
        prompt_complexity_features: DifficultyMetrics | None = None,
    ) -> DifficultyMetrics:
        metrics: DifficultyMetrics = {
            "ev_gap": round(abs(left_value - right_value), 4),
            "numeric_complexity": numeric_complexity,
        }
        if time_horizon_days is not None:
            metrics["time_horizon_days"] = time_horizon_days
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
        includes_signed_outcomes: bool = False,
    ) -> DifficultyMetrics:
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        lower_prompt = prompt.lower()
        clause_count = 1 + sum(
            lower_prompt.count(token)
            for token in (",", ";", " or ", " and ", " otherwise ", " instead ")
        )
        has_decimal_in_prompt = bool(re.search(r"\d+\.\d+", prompt))
        has_positive = any(float(value) > 0 for value in numeric_values)
        has_negative = any(float(value) < 0 for value in numeric_values)
        mixed_signed_outcomes = includes_signed_outcomes or (has_positive and has_negative)
        # Detect generic snake_case action labels so prompt complexity stays
        # robust if future subtypes introduce new action verbs.
        action_tokens = re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", lower_prompt)
        prompt_action_token_count = len(action_tokens)
        prompt_contains_action_tokens = prompt_action_token_count > 0

        left_action = comparison_pair["left_action"]
        right_action = comparison_pair["right_action"]
        left_prefix = left_action.split("_", 1)[0]
        right_prefix = right_action.split("_", 1)[0]
        asymmetric_choice_framing = left_prefix != right_prefix

        prompt_complexity = (
            clause_count
            + int(has_decimal_in_prompt)
            + int(mixed_signed_outcomes)
            + int(asymmetric_choice_framing)
            + int(prompt_contains_action_tokens)
        )
        return {
            "prompt_style_variant": prompt_style,
            "prompt_style_regime": resolved_regime,
            "prompt_clause_count": clause_count,
            "prompt_has_decimal": has_decimal_in_prompt,
            "prompt_mixed_signed_outcomes": mixed_signed_outcomes,
            "prompt_asymmetric_choice_framing": asymmetric_choice_framing,
            "prompt_contains_action_tokens": prompt_contains_action_tokens,
            "prompt_action_token_count": prompt_action_token_count,
            "prompt_complexity": prompt_complexity,
        }

    def _build_prompt_and_complexity(
        self,
        *,
        task_subtype: TaskSubtype,
        problem_spec: ProblemSpec,
        numeric_values: list[int | float],
        comparison_pair: ComparisonPair,
        includes_signed_outcomes: bool = False,
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
        self._last_prompt_frame_variant = frame_variant
        self._last_prompt_style_regime = prompt_style_regime
        prompt_complexity_features = self._compute_prompt_complexity_features(
            prompt=prompt,
            prompt_style=prompt_style,
            prompt_style_regime=prompt_style_regime,
            numeric_values=numeric_values,
            comparison_pair=comparison_pair,
            includes_signed_outcomes=includes_signed_outcomes,
        )
        prompt_complexity_features["prompt_frame_variant"] = frame_variant
        return prompt, prompt_style, prompt_complexity_features

    def _compute_numeric_complexity(
        self,
        *,
        numeric_values: list[int | float],
        arithmetic_operations: int,
        includes_signed_outcomes: bool = False,
    ) -> int:
        distinct_values = {round(float(value), 6) for value in numeric_values}
        has_decimal = any(not float(value).is_integer() for value in numeric_values)
        has_positive = any(float(value) > 0 for value in numeric_values)
        has_negative = any(float(value) < 0 for value in numeric_values)
        has_signed_outcomes = includes_signed_outcomes or (has_positive and has_negative)

        complexity = len(distinct_values) + arithmetic_operations
        if has_decimal:
            complexity += 1
        if has_signed_outcomes:
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
                number_start = index
                has_dot = char == "."
                index += 1
                while index < expression_length:
                    current = expression[index]
                    if current.isdigit():
                        index += 1
                        continue
                    if current == "." and not has_dot:
                        has_dot = True
                        index += 1
                        continue
                    break

                if index < expression_length and expression[index] in ("e", "E"):
                    exponent_index = index + 1
                    if (
                        exponent_index < expression_length
                        and expression[exponent_index] in ("+", "-")
                    ):
                        exponent_index += 1
                    exponent_start = exponent_index
                    while (
                        exponent_index < expression_length
                        and expression[exponent_index].isdigit()
                    ):
                        exponent_index += 1
                    if exponent_index > exponent_start:
                        index = exponent_index

                if index < expression_length and expression[index] == "%":
                    index += 1

                if index > number_start:
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

            if char == "%":
                token_types.append("PERCENT")
                index += 1
                continue

            token_types.append("OTHER")
            index += 1

        operation_count = 0
        operand_end_tokens = {"NUMBER", "IDENT", "RPAREN", "PERCENT"}
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
            is_binary_minus = (
                previous_token_type in operand_end_tokens
                and _starts_operand(token_index + 1)
            )
            if is_binary_minus:
                operation_count += 1

        return operation_count

    def _count_operations_in_outcome_model(self, outcome_model: dict[str, str]) -> int:
        return sum(
            self._count_arithmetic_operations(formula)
            for formula in outcome_model.values()
        )

    def _format_number(self, value: int | float) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def _round_expected_utility(self, value: int | float) -> float:
        return round(float(value), self.EXPECTED_UTILITY_PRECISION)

    def _build_expected_value_assumptions(self) -> ExpectedValueAssumptions:
        return {
            "probabilities_are_known": True,
            "utility_model": "linear",
            "decision_rule": "expected_value_maximization",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_lottery_problem_spec(
        self,
        *,
        sure_amount: int,
        p_win: float,
        win_amount: int,
        lose_amount: int,
    ) -> LotteryProblemSpec:
        return {
            "task_subtype": "lottery_choice",
            "objective": "maximize expected monetary value",
            "options": {
                "A": {"type": "certain", "amount": sure_amount},
                "B": {
                    "type": "lottery",
                    "p_win": p_win,
                    "win_amount": win_amount,
                    "lose_amount": lose_amount,
                },
            },
            "assumptions": self._build_expected_value_assumptions(),
        }

    def _build_ce_offer_comparison_problem_spec(
        self,
        *,
        offered: float,
        p_win: float,
        high: int,
        low: int,
    ) -> CeOfferComparisonProblemSpec:
        return {
            "task_subtype": "ce_offer_comparison",
            "objective": "maximize expected monetary value",
            "options": {
                "A": {"type": "certain", "amount": offered},
                "B": {
                    "type": "lottery",
                    "p_win": p_win,
                    "high": high,
                    "low": low,
                },
            },
            "assumptions": self._build_expected_value_assumptions(),
        }

    def _build_mixed_gain_loss_problem_spec(
        self,
        *,
        sure: int,
        p_gain: float,
        gain: int,
        loss: int,
    ) -> MixedGainLossProblemSpec:
        return {
            "task_subtype": "mixed_gain_loss_choice",
            "objective": "maximize expected monetary value",
            "options": {
                "A": {"type": "certain", "amount": sure},
                "B": {
                    "type": "mixed_lottery",
                    "p_gain": p_gain,
                    "gain": gain,
                    "loss": loss,
                },
            },
            "assumptions": self._build_expected_value_assumptions(),
        }

    def _build_time_discounting_problem_spec(
        self,
        *,
        now: int,
        later_offer: float,
        days: int,
        annual_rate: float,
    ) -> TimeDiscountingProblemSpec:
        assumptions: TimeDiscountingAssumptions = {
            "discount_model": "simple",
            "annual_discount_rate": annual_rate,
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }
        return {
            "task_subtype": "time_discounting",
            "objective": "maximize discounted monetary value",
            "options": {
                "A": {"type": "immediate", "amount": now, "delay_days": 0},
                "B": {
                    "type": "delayed",
                    "amount": later_offer,
                    "delay_days": days,
                },
            },
            "assumptions": assumptions,
        }

    def _build_ambiguity_aversion_assumptions(
        self,
        *,
        known_probability: float,
        subjective_ambiguous_win_probability: float,
    ) -> AmbiguityAversionAssumptions:
        return {
            "known_probability": known_probability,
            "subjective_ambiguous_win_probability": subjective_ambiguous_win_probability,
            "utility_model": "linear",
            "decision_rule": "expected_value_maximization",
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }

    def _build_ambiguity_aversion_problem_spec(
        self,
        *,
        known_probability: float,
        known_win_amount: int,
        known_lose_amount: int,
        ambiguous_win_amount: int,
        ambiguous_lose_amount: int,
        subjective_ambiguous_win_probability: float,
    ) -> AmbiguityAversionChoiceProblemSpec:
        return {
            "task_subtype": "ambiguity_aversion_choice",
            "objective": (
                "maximize expected monetary value under stated subjective beliefs "
                "for the ambiguous option (ambiguity-themed EV comparison, not "
                "unresolved ambiguity)"
            ),
            "options": {
                "A": {
                    "type": "known_risk_lottery",
                    # Intentionally mirrored in assumptions.known_probability;
                    # solver validates equality to catch inconsistent transforms.
                    "p_win": known_probability,
                    "win_amount": known_win_amount,
                    "lose_amount": known_lose_amount,
                },
                "B": {
                    "type": "ambiguous_lottery",
                    "win_amount": ambiguous_win_amount,
                    "lose_amount": ambiguous_lose_amount,
                },
            },
            "assumptions": self._build_ambiguity_aversion_assumptions(
                known_probability=known_probability,
                subjective_ambiguous_win_probability=subjective_ambiguous_win_probability,
            ),
        }

    def _build_probability_weighting_counterexample_problem_spec(
        self,
        *,
        sure_amount: int,
        p_win: float,
        win_amount: int,
        lose_amount: int,
    ) -> ProbabilityWeightingCounterexampleProblemSpec:
        return {
            "task_subtype": "probability_weighting_counterexample",
            "objective": "maximize expected monetary value",
            "options": {
                "A": {"type": "certain", "amount": sure_amount},
                "B": {
                    "type": "lottery",
                    "p_win": p_win,
                    "win_amount": win_amount,
                    "lose_amount": lose_amount,
                },
            },
            "assumptions": self._build_expected_value_assumptions(),
        }

    def _build_loss_aversion_counterexample_problem_spec(
        self,
        *,
        status_quo_amount: int,
        p_gain: float,
        gain: int,
        loss: int,
    ) -> LossAversionCounterexampleProblemSpec:
        return {
            "task_subtype": "loss_aversion_counterexample",
            "objective": "maximize expected monetary value over total outcomes",
            "options": {
                "A": {"type": "status_quo", "amount": status_quo_amount},
                "B": {
                    "type": "mixed_lottery",
                    "p_gain": p_gain,
                    "gain": gain,
                    "loss": loss,
                },
            },
            "assumptions": self._build_expected_value_assumptions(),
        }

    def _build_hyperbolic_discounting_counterexample_problem_spec(
        self,
        *,
        earlier_amount: int,
        earlier_delay_days: int,
        later_amount: float,
        later_delay_days: int,
        annual_rate: float,
    ) -> HyperbolicDiscountingCounterexampleProblemSpec:
        assumptions: TimeDiscountingAssumptions = {
            "discount_model": "simple",
            "annual_discount_rate": annual_rate,
            "tie_epsilon": self.CHOICE_TIE_EPSILON,
        }
        return {
            "task_subtype": "hyperbolic_discounting_counterexample",
            "objective": "maximize discounted monetary value",
            "options": {
                "A": {
                    "type": "earlier",
                    "amount": earlier_amount,
                    "delay_days": earlier_delay_days,
                },
                "B": {
                    "type": "later",
                    "amount": later_amount,
                    "delay_days": later_delay_days,
                },
            },
            "assumptions": assumptions,
        }

    def _render_lottery_choice_prompt(
        self,
        *,
        problem_spec: LotteryProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        sure_amount = option_a["amount"]
        p_win = option_b["p_win"]
        win_amount = option_b["win_amount"]
        lose_amount = option_b["lose_amount"]
        sure_text = self._format_number(sure_amount)
        win_probability_text = self._format_number(p_win * 100)
        win_text = self._format_number(win_amount)
        lose_text = self._format_number(lose_amount)
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_sure) yields a certain payoff of ${sure_text}. "
                f"Option B (choose_lottery) yields ${win_text} with probability "
                f"{win_probability_text}% and ${lose_text} otherwise. "
                "Choose the action that would leave you with more money over many similar rounds."
            )
        elif tier == "neutral_natural":
            body = (
                f"You can take ${sure_text} for sure (choose_sure), or gamble "
                f"(choose_lottery): {win_probability_text}% chance of ${win_text}, "
                f"else ${lose_text}. Which option would you choose?"
            )
        else:
            body = (
                f"Allocate to a risk-free payoff (choose_sure): ${sure_text}, "
                f"or a risky ticket (choose_lottery): {win_probability_text}% of "
                f"${win_text}, otherwise ${lose_text}. "
                "Which option feels more compelling in this case?"
            )
        if style == "unlabeled":
            return (
                f"Option A gives ${sure_text} for certain. Option B gives ${win_text} "
                f"with probability {win_probability_text}% and ${lose_text} otherwise. "
                "Which option would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="lottery_choice",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_ce_offer_comparison_prompt(
        self,
        *,
        problem_spec: CeOfferComparisonProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        p_win = option_b["p_win"]
        high = option_b["high"]
        low = option_b["low"]
        offered = option_a["amount"]
        win_probability_text = self._format_number(p_win * 100)
        high_text = self._format_number(high)
        low_text = self._format_number(low)
        offered_text = self._format_number(offered)
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"A lottery offers ${high_text} with probability {win_probability_text}% "
                f"and ${low_text} otherwise. A sure payment of ${offered_text} is "
                "available as an alternative. Choose "
                "accept_offer or reject_offer based on which leaves you with more "
                "money over repeats."
            )
        elif tier == "neutral_natural":
            body = (
                f"A lottery pays ${high_text} with probability {win_probability_text}% "
                f"and ${low_text} otherwise. You are offered a certain ${offered_text} "
                "instead. Choose whether to accept the certain offer (accept_offer) "
                "or keep the lottery (reject_offer)."
            )
        else:
            body = (
                f"A contingent payout contract returns ${high_text} with "
                f"{win_probability_text}% probability and ${low_text} otherwise. "
                f"Counterparty offers a cash-out at ${offered_text}. "
                "Choose accept_offer or reject_offer."
            )
        if style == "unlabeled":
            return (
                f"A lottery pays ${high_text} with probability {win_probability_text}% "
                f"and ${low_text} otherwise. You can instead take a certain "
                f"${offered_text}. Which option would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="ce_offer_comparison",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_mixed_gain_loss_choice_prompt(
        self,
        *,
        problem_spec: MixedGainLossProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        sure = option_a["amount"]
        p_gain = option_b["p_gain"]
        gain = option_b["gain"]
        loss = option_b["loss"]
        sure_text = self._format_number(sure)
        gain_probability_text = self._format_number(p_gain * 100)
        gain_text = self._format_number(gain)
        loss_text = self._format_number(loss)
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_sure) yields a certain payoff of ${sure_text}. "
                f"Option B (choose_risky) yields ${gain_text} with probability "
                f"{gain_probability_text}% and ${loss_text} otherwise. "
                "Choose the option that would leave you with more money over many similar rounds."
            )
        elif tier == "neutral_natural":
            body = (
                f"Take ${sure_text} for sure (choose_sure), or take the risky option "
                f"(choose_risky): {gain_probability_text}% chance of ${gain_text}, "
                f"otherwise ${loss_text}. Which option would you choose?"
            )
        else:
            body = (
                f"Compare a guaranteed cashflow (choose_sure): ${sure_text} versus "
                f"a risky exposure (choose_risky): {gain_probability_text}% of "
                f"${gain_text}, otherwise ${loss_text}. "
                "Which option feels more compelling here?"
            )
        if style == "unlabeled":
            return (
                f"Option A gives ${sure_text} for sure. Option B gives ${gain_text} "
                f"with probability {gain_probability_text}% and ${loss_text} otherwise. "
                "Which option would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="mixed_gain_loss_choice",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_time_discounting_prompt(
        self,
        *,
        problem_spec: TimeDiscountingProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        assumptions = problem_spec["assumptions"]
        now = option_a["amount"]
        later_offer = option_b["amount"]
        days = option_b["delay_days"]
        annual_rate = assumptions["annual_discount_rate"]
        now_text = self._format_number(now)
        later_text = self._format_number(later_offer)
        days_text = self._format_number(days)
        rate_text = self._format_number(annual_rate)
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_now) pays ${now_text} immediately. "
                f"Option B (choose_later) pays ${later_text} after {days_text} days. "
                f"Evaluate using simple annual discounting at r={rate_text}."
            )
        elif tier == "neutral_natural":
            body = (
                f"Choose now (choose_now): ${now_text} today, or choose later "
                f"(choose_later): ${later_text} in {days_text} days. "
                f"Use annual simple discount rate r={rate_text}."
            )
        else:
            body = (
                f"Immediate settlement (choose_now): ${now_text}. Deferred settlement "
                f"(choose_later): ${later_text} at T={days_text} days. "
                f"With annual discount rate r={rate_text}, which choice is worth more now?"
            )
        if style == "unlabeled":
            return (
                f"Option A pays ${now_text} today. Option B pays ${later_text} "
                f"in {days_text} days. Compare them using simple annual discounting "
                f"at rate r={rate_text}."
            )
        body = self._diversify_body_template(
            task_subtype="time_discounting",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_ambiguity_aversion_choice_prompt(
        self,
        *,
        problem_spec: AmbiguityAversionChoiceProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        assumptions = problem_spec["assumptions"]
        known_probability = assumptions["known_probability"]
        subjective_ambiguous_win_probability = assumptions[
            "subjective_ambiguous_win_probability"
        ]
        known_probability_text = self._format_number(known_probability * 100)
        subjective_probability_text = self._format_number(
            subjective_ambiguous_win_probability * 100
        )
        known_win_text = self._format_number(option_a["win_amount"])
        known_lose_text = self._format_number(option_a["lose_amount"])
        ambiguous_win_text = self._format_number(option_b["win_amount"])
        ambiguous_lose_text = self._format_number(option_b["lose_amount"])
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                "Option A (choose_known_risk) is a known-risk lottery with "
                f"{known_probability_text}% chance of ${known_win_text} and "
                f"${known_lose_text} otherwise. Option B (choose_ambiguous) is an "
                f"ambiguous lottery paying ${ambiguous_win_text} or ${ambiguous_lose_text}; "
                f"use subjective win probability {subjective_probability_text}%. "
                "Choose the option that would leave you with more money over many repeats."
            )
        elif tier == "neutral_natural":
            body = (
                f"Known urn option (choose_known_risk): {known_probability_text}% chance "
                f"to win ${known_win_text}, otherwise ${known_lose_text}. "
                f"Ambiguous urn option (choose_ambiguous): outcomes are ${ambiguous_win_text} "
                f"or ${ambiguous_lose_text}, and you should evaluate it using subjective "
                f"win chance {subjective_probability_text}%. "
                "Which option would you choose?"
            )
        else:
            body = (
                f"Known-risk ticket (choose_known_risk): {known_probability_text}% of "
                f"${known_win_text}, else ${known_lose_text}. Ambiguous ticket "
                f"(choose_ambiguous): ${ambiguous_win_text} or ${ambiguous_lose_text}, "
                f"priced using subjective win probability {subjective_probability_text}%. "
                "Which position would you take?"
            )
        if style == "unlabeled":
            return (
                f"Option A has known probability {known_probability_text}% for "
                f"${known_win_text}, otherwise ${known_lose_text}. Option B is ambiguous, "
                f"with outcomes ${ambiguous_win_text} or ${ambiguous_lose_text}; evaluate "
                f"Option B using subjective win probability {subjective_probability_text}%. "
                "Which option would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="ambiguity_aversion_choice",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_probability_weighting_counterexample_prompt(
        self,
        *,
        problem_spec: ProbabilityWeightingCounterexampleProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        sure_text = self._format_number(option_a["amount"])
        p_win_text = self._format_number(option_b["p_win"] * 100)
        win_text = self._format_number(option_b["win_amount"])
        lose_text = self._format_number(option_b["lose_amount"])
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_sure) yields a certain ${sure_text}. "
                f"Option B (choose_longshot) yields ${win_text} with probability "
                f"{p_win_text}% and ${lose_text} otherwise. "
                "Choose the one that would pay more financially over many repeats."
            )
        elif tier == "neutral_natural":
            body = (
                f"Take ${sure_text} for sure (choose_sure), or take the longshot "
                f"(choose_longshot): {p_win_text}% chance of ${win_text}, else "
                f"${lose_text}. Which option would you choose?"
            )
        else:
            body = (
                f"Guaranteed payoff (choose_sure): ${sure_text}. Speculative tail payoff "
                f"(choose_longshot): {p_win_text}% of ${win_text}, otherwise ${lose_text}. "
                "Which choice would you make here?"
            )
        if style == "unlabeled":
            return (
                f"Option A pays ${sure_text} for sure. Option B pays ${win_text} with "
                f"probability {p_win_text}% and ${lose_text} otherwise. Which option "
                "would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="probability_weighting_counterexample",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_loss_aversion_counterexample_prompt(
        self,
        *,
        problem_spec: LossAversionCounterexampleProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        status_quo_text = self._format_number(option_a["amount"])
        p_gain_text = self._format_number(option_b["p_gain"] * 100)
        gain_text = self._format_number(option_b["gain"])
        loss_text = self._format_number(option_b["loss"])
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_status_quo) yields ${status_quo_text} for certain. "
                f"Option B (accept_gamble) yields ${gain_text} with probability "
                f"{p_gain_text}% and ${loss_text} otherwise. Choose by expected "
                "financial outcome over many similar rounds."
            )
        elif tier == "neutral_natural":
            body = (
                f"Keep the status quo (choose_status_quo): ${status_quo_text}, or accept "
                f"the mixed gamble (accept_gamble): {p_gain_text}% chance of ${gain_text}, "
                f"otherwise ${loss_text}. Which option would you choose?"
            )
        else:
            body = (
                f"Flat position (choose_status_quo): ${status_quo_text}. Mixed exposure "
                f"(accept_gamble): {p_gain_text}% of ${gain_text}, otherwise ${loss_text}. "
                "Which option feels more compelling?"
            )
        if style == "unlabeled":
            return (
                f"Option A keeps a certain ${status_quo_text}. Option B gives ${gain_text} "
                f"with probability {p_gain_text}% and ${loss_text} otherwise. Which option "
                "would you choose?"
            )
        body = self._diversify_body_template(
            task_subtype="loss_aversion_counterexample",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _render_hyperbolic_discounting_counterexample_prompt(
        self,
        *,
        problem_spec: HyperbolicDiscountingCounterexampleProblemSpec,
        style: PromptStyle,
        prompt_style_regime: PromptStyleRegime | None = None,
        prompt_frame_variant: PromptFrameVariant | None = None,
    ) -> str:
        _ = prompt_frame_variant
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        assumptions = problem_spec["assumptions"]
        earlier_amount_text = self._format_number(option_a["amount"])
        earlier_days_text = self._format_number(option_a["delay_days"])
        later_amount_text = self._format_number(option_b["amount"])
        later_days_text = self._format_number(option_b["delay_days"])
        annual_rate_text = self._format_number(assumptions["annual_discount_rate"])
        resolved_regime = (
            self._resolve_prompt_style_regime()
            if prompt_style_regime is None
            else prompt_style_regime
        )
        tier = self._prompt_style_tier(style, resolved_regime)
        if tier == "formal":
            body = (
                f"Option A (choose_earlier) pays ${earlier_amount_text} in "
                f"{earlier_days_text} days. Option B (choose_later) pays "
                f"${later_amount_text} in {later_days_text} days. Compare both by "
                f"simple annual discounting at r={annual_rate_text}."
            )
        elif tier == "neutral_natural":
            body = (
                f"You can take ${earlier_amount_text} in {earlier_days_text} days "
                f"(choose_earlier), or ${later_amount_text} in {later_days_text} days "
                f"(choose_later). Value both using simple discount rate r={annual_rate_text}."
            )
        else:
            body = (
                f"Earlier settlement (choose_earlier): ${earlier_amount_text} at "
                f"T={earlier_days_text}d. Later settlement (choose_later): "
                f"${later_amount_text} at T={later_days_text}d. Discount both using "
                f"simple annual rate r={annual_rate_text}."
            )
        if style == "unlabeled":
            return (
                f"Option A pays ${earlier_amount_text} in {earlier_days_text} days. "
                f"Option B pays ${later_amount_text} in {later_days_text} days. "
                f"Compare using simple annual discounting at r={annual_rate_text}."
            )
        body = self._diversify_body_template(
            task_subtype="hyperbolic_discounting_counterexample",
            frame_variant=prompt_frame_variant,
            tier=tier,
            problem_spec=problem_spec,
            body=body,
        )
        return body

    def _build_lottery_outcome_model(
        self, *, problem_spec: LotteryProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        sure_amount = option_a["amount"]
        p_win = option_b["p_win"]
        win_amount = option_b["win_amount"]
        lose_amount = option_b["lose_amount"]
        return {
            "choose_sure": self._format_number(sure_amount),
            "choose_lottery": (
                f"{self._format_number(p_win)} * {self._format_number(win_amount)}"
                f" + (1 - {self._format_number(p_win)})"
                f" * {self._format_number(lose_amount)}"
            ),
        }

    def _build_ce_offer_outcome_model(
        self, *, problem_spec: CeOfferComparisonProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        offered = option_a["amount"]
        p_win = option_b["p_win"]
        high = option_b["high"]
        low = option_b["low"]
        return {
            "accept_offer": self._format_number(offered),
            "reject_offer": (
                f"{self._format_number(p_win)} * {self._format_number(high)}"
                f" + (1 - {self._format_number(p_win)}) * {self._format_number(low)}"
            ),
        }

    def _build_mixed_gain_loss_outcome_model(
        self, *, problem_spec: MixedGainLossProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        sure = option_a["amount"]
        p_gain = option_b["p_gain"]
        gain = option_b["gain"]
        loss = option_b["loss"]
        return {
            "choose_sure": self._format_number(sure),
            "choose_risky": (
                f"{self._format_number(p_gain)} * {self._format_number(gain)}"
                f" + (1 - {self._format_number(p_gain)}) * {self._format_number(loss)}"
            ),
        }

    def _build_time_discounting_outcome_model(
        self, *, problem_spec: TimeDiscountingProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        now = option_a["amount"]
        later_offer = option_b["amount"]
        days = option_b["delay_days"]
        annual_rate = problem_spec["assumptions"]["annual_discount_rate"]
        return {
            "choose_now": self._format_number(now),
            "choose_later": (
                f"{self._format_number(later_offer)}"
                f" / (1 + {self._format_number(annual_rate)}"
                f" * ({self._format_number(days)} / 365))"
            ),
        }

    def _build_ambiguity_aversion_outcome_model(
        self, *, problem_spec: AmbiguityAversionChoiceProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        assumptions = problem_spec["assumptions"]
        known_probability = assumptions["known_probability"]
        subjective_ambiguous_win_probability = assumptions[
            "subjective_ambiguous_win_probability"
        ]
        return {
            "choose_known_risk": (
                f"{self._format_number(known_probability)} * "
                f"{self._format_number(option_a['win_amount'])} + (1 - "
                f"{self._format_number(known_probability)}) * "
                f"{self._format_number(option_a['lose_amount'])}"
            ),
            "choose_ambiguous": (
                f"{self._format_number(subjective_ambiguous_win_probability)} * "
                f"{self._format_number(option_b['win_amount'])} + (1 - "
                f"{self._format_number(subjective_ambiguous_win_probability)}) * "
                f"{self._format_number(option_b['lose_amount'])}"
            ),
        }

    def _build_probability_weighting_counterexample_outcome_model(
        self, *, problem_spec: ProbabilityWeightingCounterexampleProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        p_win = option_b["p_win"]
        return {
            "choose_sure": self._format_number(option_a["amount"]),
            "choose_longshot": (
                f"{self._format_number(p_win)} * "
                f"{self._format_number(option_b['win_amount'])} + "
                f"(1 - {self._format_number(p_win)}) * "
                f"{self._format_number(option_b['lose_amount'])}"
            ),
        }

    def _build_loss_aversion_counterexample_outcome_model(
        self, *, problem_spec: LossAversionCounterexampleProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        p_gain = option_b["p_gain"]
        return {
            "choose_status_quo": self._format_number(option_a["amount"]),
            "accept_gamble": (
                f"{self._format_number(p_gain)} * {self._format_number(option_b['gain'])}"
                f" + (1 - {self._format_number(p_gain)}) * "
                f"{self._format_number(option_b['loss'])}"
            ),
        }

    def _build_hyperbolic_discounting_counterexample_outcome_model(
        self, *, problem_spec: HyperbolicDiscountingCounterexampleProblemSpec
    ) -> dict[str, str]:
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        annual_rate = problem_spec["assumptions"]["annual_discount_rate"]
        return {
            "choose_earlier": (
                f"{self._format_number(option_a['amount'])}"
                f" / (1 + {self._format_number(annual_rate)}"
                f" * ({self._format_number(option_a['delay_days'])} / 365))"
            ),
            "choose_later": (
                f"{self._format_number(option_b['amount'])}"
                f" / (1 + {self._format_number(annual_rate)}"
                f" * ({self._format_number(option_b['delay_days'])} / 365))"
            ),
        }

    def _generate_lottery_choice(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        sure_amount = self.rng.randint(20, 150)
        p_win = round(self.rng.uniform(0.1, 0.9), 2)
        win_amount = self.rng.randint(50, 300)
        lose_amount = 0

        ev_sure = self._round_expected_utility(sure_amount)
        ev_lottery = self._round_expected_utility(
            p_win * win_amount + (1 - p_win) * lose_amount
        )

        optimal = self._choose_optimal_action(
            left_label="choose_lottery",
            left_value=ev_lottery,
            right_label="choose_sure",
            right_value=ev_sure,
        )
        decision_values = {"choose_lottery": ev_lottery, "choose_sure": ev_sure}
        comparison_pair = self._comparison_pair_for_subtype("lottery_choice")
        problem_spec = self._build_lottery_problem_spec(
            sure_amount=sure_amount,
            p_win=p_win,
            win_amount=win_amount,
            lose_amount=lose_amount,
        )
        outcome_model = self._build_lottery_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="lottery_choice",
                problem_spec=problem_spec,
                numeric_values=[sure_amount, p_win, win_amount, lose_amount],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_lottery,
            right_value=ev_sure,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[sure_amount, p_win, win_amount, lose_amount],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="lottery_choice",
            task_id_prefix="lottery",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["choose_sure", "choose_lottery", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={"choose_sure": ev_sure, "choose_lottery": ev_lottery},
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Average dollar value is {ev_lottery} for the lottery "
                f"versus {ev_sure} for the sure option."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_ce_offer_comparison(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        # Offer-comparison task under linear utility: compare the sure offer to
        # the lottery expected value (equal to the certainty equivalent here).
        p_win = round(self.rng.uniform(0.2, 0.9), 2)
        high = self.rng.randint(60, 250)
        low = self.rng.randint(0, max(0, high - 40))
        ce = self._round_expected_utility(p_win * high + (1 - p_win) * low)
        offered = round(ce + self.rng.uniform(-25, 25), 2)
        ev_offer = self._round_expected_utility(offered)
        optimal = self._choose_optimal_action(
            left_label="accept_offer",
            left_value=ev_offer,
            right_label="reject_offer",
            right_value=ce,
        )
        decision_values = {"accept_offer": ev_offer, "reject_offer": ce}
        comparison_pair = self._comparison_pair_for_subtype("ce_offer_comparison")
        problem_spec = self._build_ce_offer_comparison_problem_spec(
            offered=offered,
            p_win=p_win,
            high=high,
            low=low,
        )
        outcome_model = self._build_ce_offer_outcome_model(problem_spec=problem_spec)
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="ce_offer_comparison",
                problem_spec=problem_spec,
                numeric_values=[p_win, high, low, offered],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_offer,
            right_value=ce,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[p_win, high, low, offered],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="ce_offer_comparison",
            task_id_prefix="ce",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["accept_offer", "reject_offer", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "accept_offer": ev_offer,
                "reject_offer": ce,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"The lottery's average dollar value is {ce}, compared with the "
                f"sure offer of {ev_offer}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_mixed_gain_loss_choice(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        sure = self.rng.randint(-120, 120)
        p_gain = round(self.rng.uniform(0.15, 0.85), 2)
        gain = self.rng.randint(40, 220)
        loss = -self.rng.randint(20, 180)
        ev_risky = self._round_expected_utility(p_gain * gain + (1 - p_gain) * loss)
        ev_sure = self._round_expected_utility(sure)
        optimal = self._choose_optimal_action(
            left_label="choose_risky",
            left_value=ev_risky,
            right_label="choose_sure",
            right_value=ev_sure,
        )
        decision_values = {"choose_risky": ev_risky, "choose_sure": ev_sure}
        comparison_pair = self._comparison_pair_for_subtype("mixed_gain_loss_choice")
        problem_spec = self._build_mixed_gain_loss_problem_spec(
            sure=sure,
            p_gain=p_gain,
            gain=gain,
            loss=loss,
        )
        outcome_model = self._build_mixed_gain_loss_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="mixed_gain_loss_choice",
                problem_spec=problem_spec,
                numeric_values=[sure, p_gain, gain, loss],
                comparison_pair=comparison_pair,
                includes_signed_outcomes=True,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_risky,
            right_value=ev_sure,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[sure, p_gain, gain, loss],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
                includes_signed_outcomes=True,
            ),
            prompt_complexity_features=prompt_complexity_features,
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="mixed_gain_loss_choice",
            task_id_prefix="mixed_gain_loss",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["choose_sure", "choose_risky", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_sure": ev_sure,
                "choose_risky": ev_risky,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Average dollar value is {ev_risky} for the risky option "
                f"and {ev_sure} for the sure option."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_time_discounting(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        now = self.rng.randint(20, 180)
        days = self.rng.choice([7, 14, 30, 60, 90, 180, 365])
        annual_rate = round(self.rng.uniform(0.02, 0.25), 4)
        discount_factor = 1 / (1 + annual_rate * (days / 365))
        later_break_even = round(now / discount_factor, 2)
        later_offer = round(later_break_even + self.rng.uniform(-20, 20), 2)
        pv_later = self._round_expected_utility(later_offer * discount_factor)
        ev_now = self._round_expected_utility(now)
        optimal = self._choose_optimal_action(
            left_label="choose_later",
            left_value=pv_later,
            right_label="choose_now",
            right_value=ev_now,
        )
        decision_values = {"choose_later": pv_later, "choose_now": ev_now}
        comparison_pair = self._comparison_pair_for_subtype("time_discounting")
        problem_spec = self._build_time_discounting_problem_spec(
            now=now,
            later_offer=later_offer,
            days=days,
            annual_rate=annual_rate,
        )
        outcome_model = self._build_time_discounting_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="time_discounting",
                problem_spec=problem_spec,
                numeric_values=[now, later_offer, days, annual_rate],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=pv_later,
            right_value=ev_now,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[now, later_offer, days, annual_rate],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            time_horizon_days=days,
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "earlier_delay_days": 0,
                "later_delay_days": days,
                "delay_gap_days": days,
                "later_minus_now_signed": round(pv_later - ev_now, 4),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="time_discounting",
            task_id_prefix="time",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "time_horizon_days": problem_spec["options"]["B"]["delay_days"],
                "options": problem_spec["options"],
            },
            actions=["choose_now", "choose_later", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_now": ev_now,
                "choose_later": pv_later,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Today's value is {pv_later} for the later payment versus {ev_now} now."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_ambiguity_aversion_choice(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        regime = self.rng.choice(
            [
                "choose_ambiguous",
                "choose_known_risk",
                "near_indifferent",
            ]
        )
        known_probability = 0.5
        known_win_amount = 100
        known_lose_amount = 0
        ambiguous_win_amount = 120
        ambiguous_lose_amount = 0
        subjective_ambiguous_win_probability = 0.5
        found_candidate = False
        for _ in range(200):
            candidate_known_probability = round(self.rng.uniform(0.3, 0.7), 2)
            candidate_known_lose_amount = self.rng.randint(0, 40)
            candidate_ambiguous_win_amount = self.rng.randint(80, 280)
            candidate_ambiguous_lose_amount = self.rng.randint(0, 40)
            candidate_subjective_ambiguous_win_probability = round(
                self.rng.uniform(0.2, 0.8), 2
            )
            if regime == "choose_ambiguous":
                target_gap = self.rng.uniform(6.0, 35.0)
            elif regime == "choose_known_risk":
                target_gap = self.rng.uniform(-35.0, -6.0)
            else:
                target_gap = self.rng.uniform(-3.0, 3.0)

            candidate_ev_ambiguous = self._round_expected_utility(
                candidate_subjective_ambiguous_win_probability
                * candidate_ambiguous_win_amount
                + (1 - candidate_subjective_ambiguous_win_probability)
                * candidate_ambiguous_lose_amount
            )
            target_ev_known = candidate_ev_ambiguous - target_gap
            candidate_known_win_amount = int(
                round(
                    (
                        target_ev_known
                        - (1 - candidate_known_probability) * candidate_known_lose_amount
                    )
                    / max(candidate_known_probability, 1e-6)
                )
            )
            if candidate_known_win_amount < 60 or candidate_known_win_amount > 220:
                continue

            candidate_ev_known_risk = self._round_expected_utility(
                candidate_known_probability * candidate_known_win_amount
                + (1 - candidate_known_probability) * candidate_known_lose_amount
            )
            candidate_gap = candidate_ev_ambiguous - candidate_ev_known_risk
            if not self._ambiguity_gap_matches_regime(regime=regime, gap=candidate_gap):
                continue

            known_probability = candidate_known_probability
            known_win_amount = candidate_known_win_amount
            known_lose_amount = candidate_known_lose_amount
            ambiguous_win_amount = candidate_ambiguous_win_amount
            ambiguous_lose_amount = candidate_ambiguous_lose_amount
            subjective_ambiguous_win_probability = (
                candidate_subjective_ambiguous_win_probability
            )
            found_candidate = True
            break

        if not found_candidate:
            (
                known_probability,
                known_win_amount,
                known_lose_amount,
                ambiguous_win_amount,
                ambiguous_lose_amount,
                subjective_ambiguous_win_probability,
            ) = self._ambiguity_aversion_regime_fallback(regime=regime)

        # This subtype remains normative EV under an explicit subjective belief;
        # it is ambiguity-themed rather than an unresolved-ambiguity benchmark.
        ev_known_risk = self._round_expected_utility(
            known_probability * known_win_amount
            + (1 - known_probability) * known_lose_amount
        )
        ev_ambiguous = self._round_expected_utility(
            subjective_ambiguous_win_probability * ambiguous_win_amount
            + (1 - subjective_ambiguous_win_probability) * ambiguous_lose_amount
        )
        optimal = self._choose_optimal_action(
            left_label="choose_ambiguous",
            left_value=ev_ambiguous,
            right_label="choose_known_risk",
            right_value=ev_known_risk,
        )
        decision_values = {
            "choose_ambiguous": ev_ambiguous,
            "choose_known_risk": ev_known_risk,
        }
        comparison_pair = self._comparison_pair_for_subtype("ambiguity_aversion_choice")
        problem_spec = self._build_ambiguity_aversion_problem_spec(
            known_probability=known_probability,
            known_win_amount=known_win_amount,
            known_lose_amount=known_lose_amount,
            ambiguous_win_amount=ambiguous_win_amount,
            ambiguous_lose_amount=ambiguous_lose_amount,
            subjective_ambiguous_win_probability=subjective_ambiguous_win_probability,
        )
        outcome_model = self._build_ambiguity_aversion_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="ambiguity_aversion_choice",
                problem_spec=problem_spec,
                numeric_values=[
                    known_probability,
                    known_win_amount,
                    known_lose_amount,
                    subjective_ambiguous_win_probability,
                    ambiguous_win_amount,
                    ambiguous_lose_amount,
                ],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_ambiguous,
            right_value=ev_known_risk,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    known_probability,
                    known_win_amount,
                    known_lose_amount,
                    subjective_ambiguous_win_probability,
                    ambiguous_win_amount,
                    ambiguous_lose_amount,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "ambiguous_minus_known_ev_signed": round(
                    ev_ambiguous - ev_known_risk, 4
                ),
                "ambiguity_regime_intended": regime,
                "ambiguity_regime_realized": self._ambiguity_realized_regime_from_gap(
                    ev_ambiguous - ev_known_risk
                ),
                "known_probability": known_probability,
                "subjective_ambiguous_win_probability": (
                    subjective_ambiguous_win_probability
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="ambiguity_aversion_choice",
            task_id_prefix="ambiguity",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["choose_known_risk", "choose_ambiguous", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_known_risk": ev_known_risk,
                "choose_ambiguous": ev_ambiguous,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Known-risk EV is {ev_known_risk}; ambiguous EV under the provided "
                f"subjective probability is {ev_ambiguous}."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _ambiguity_gap_matches_regime(self, *, regime: str, gap: float) -> bool:
        if regime == "choose_ambiguous":
            return gap >= 6.0
        if regime == "choose_known_risk":
            return gap <= -6.0
        return abs(gap) <= 3.0

    def _ambiguity_realized_regime_from_gap(self, gap: float) -> str:
        if gap >= 6.0:
            return "choose_ambiguous"
        if gap <= -6.0:
            return "choose_known_risk"
        if abs(gap) <= 3.0:
            return "near_indifferent"
        return "ambiguous_band"

    def _ambiguity_aversion_regime_fallback(
        self, *, regime: str
    ) -> tuple[float, int, int, int, int, float]:
        if regime == "choose_ambiguous":
            return (0.5, 100, 0, 120, 0, 0.6)
        if regime == "choose_known_risk":
            return (0.5, 140, 0, 120, 0, 0.45)
        return (0.5, 120, 0, 120, 0, 0.5)

    def _generate_probability_weighting_counterexample(
        self, sample_index: int
    ) -> DataPoint:
        current_index = sample_index
        p_win = round(self.rng.uniform(0.03, 0.18), 2)
        win_amount = self.rng.randint(300, 1200)
        lose_amount = 0

        ev_longshot = self._round_expected_utility(
            p_win * win_amount + (1 - p_win) * lose_amount
        )
        margin = self.rng.randint(5, 30)
        longshot_ev_superior = self.rng.random() < 0.5
        if longshot_ev_superior:
            sure_amount = max(1, int(ev_longshot) - margin)
            if sure_amount >= ev_longshot:
                sure_amount = max(1, int(ev_longshot) - 1)
        else:
            sure_amount = int(ev_longshot) + margin
            if sure_amount <= ev_longshot:
                sure_amount = int(ev_longshot) + 1

        ev_sure = self._round_expected_utility(sure_amount)

        optimal = self._choose_optimal_action(
            left_label="choose_longshot",
            left_value=ev_longshot,
            right_label="choose_sure",
            right_value=ev_sure,
        )
        decision_values = {
            "choose_longshot": ev_longshot,
            "choose_sure": ev_sure,
        }
        comparison_pair = self._comparison_pair_for_subtype(
            "probability_weighting_counterexample"
        )
        problem_spec = self._build_probability_weighting_counterexample_problem_spec(
            sure_amount=sure_amount,
            p_win=p_win,
            win_amount=win_amount,
            lose_amount=lose_amount,
        )
        outcome_model = self._build_probability_weighting_counterexample_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="probability_weighting_counterexample",
                problem_spec=problem_spec,
                numeric_values=[sure_amount, p_win, win_amount, lose_amount],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_longshot,
            right_value=ev_sure,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[sure_amount, p_win, win_amount, lose_amount],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "is_longshot_setup": p_win <= 0.2 and win_amount >= 250,
                "longshot_probability": p_win,
                "longshot_ev_superior": ev_longshot > ev_sure,
                "longshot_near_tie": abs(ev_longshot - ev_sure)
                <= self.CHOICE_TIE_EPSILON,
                "longshot_minus_sure_ev_signed": round(ev_longshot - ev_sure, 4),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="probability_weighting_counterexample",
            task_id_prefix="prob_weight",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["choose_sure", "choose_longshot", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_sure": ev_sure,
                "choose_longshot": ev_longshot,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Average dollar value is {ev_longshot} for the longshot "
                f"and {ev_sure} for the sure option."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_loss_aversion_counterexample(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        status_quo_amount = self.rng.randint(-30, 30)
        ev_status_quo = self._round_expected_utility(status_quo_amount)
        regime = self.rng.choice(
            [
                "accept_gamble",
                "choose_status_quo",
                "near_indifferent",
            ]
        )
        p_gain = 0.55
        gain = 120
        loss = -80
        ev_gamble = self._round_expected_utility(p_gain * gain + (1 - p_gain) * loss)
        found_candidate = False
        for _ in range(200):
            candidate_p_gain = round(self.rng.uniform(0.45, 0.7), 2)
            candidate_loss = -self.rng.randint(40, 160)
            if regime == "accept_gamble":
                target_gap = self.rng.uniform(6.0, 35.0)
            elif regime == "choose_status_quo":
                target_gap = self.rng.uniform(-35.0, -6.0)
            else:
                target_gap = self.rng.uniform(-3.0, 3.0)

            target_ev_gamble = ev_status_quo + target_gap
            candidate_gain = int(
                round(
                    (
                        target_ev_gamble - (1 - candidate_p_gain) * candidate_loss
                    )
                    / max(candidate_p_gain, 1e-6)
                )
            )
            if candidate_gain < 40 or candidate_gain > 320:
                continue

            candidate_ev_gamble = self._round_expected_utility(
                candidate_p_gain * candidate_gain
                + (1 - candidate_p_gain) * candidate_loss
            )
            gap = candidate_ev_gamble - ev_status_quo
            if not self._loss_aversion_gap_matches_regime(regime=regime, gap=gap):
                continue

            p_gain = candidate_p_gain
            gain = candidate_gain
            loss = candidate_loss
            ev_gamble = candidate_ev_gamble
            found_candidate = True
            break

        if not found_candidate:
            p_gain, gain, loss, ev_gamble = self._loss_aversion_counterexample_fallback(
                regime=regime, ev_status_quo=ev_status_quo
            )

        optimal = self._choose_optimal_action(
            left_label="accept_gamble",
            left_value=ev_gamble,
            right_label="choose_status_quo",
            right_value=ev_status_quo,
        )
        decision_values = {
            "accept_gamble": ev_gamble,
            "choose_status_quo": ev_status_quo,
        }
        comparison_pair = self._comparison_pair_for_subtype(
            "loss_aversion_counterexample"
        )
        problem_spec = self._build_loss_aversion_counterexample_problem_spec(
            status_quo_amount=status_quo_amount,
            p_gain=p_gain,
            gain=gain,
            loss=loss,
        )
        outcome_model = self._build_loss_aversion_counterexample_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="loss_aversion_counterexample",
                problem_spec=problem_spec,
                numeric_values=[status_quo_amount, p_gain, gain, loss],
                comparison_pair=comparison_pair,
                includes_signed_outcomes=True,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=ev_gamble,
            right_value=ev_status_quo,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[status_quo_amount, p_gain, gain, loss],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
                includes_signed_outcomes=True,
            ),
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "mixed_gamble_ev_superior_to_status_quo": ev_gamble > ev_status_quo,
                "mixed_gamble_positive_ev": ev_gamble > 0,
                "mixed_gamble_minus_status_quo_ev_signed": round(
                    ev_gamble - ev_status_quo, 4
                ),
                "mixed_gamble_regime_intended": regime,
                "mixed_gamble_regime_realized": self._loss_aversion_realized_regime_from_gap(
                    ev_gamble - ev_status_quo
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="loss_aversion_counterexample",
            task_id_prefix="loss_averse",
            problem_spec=problem_spec,
            prompt=prompt,
            state={"options": problem_spec["options"]},
            actions=["choose_status_quo", "accept_gamble", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_status_quo": ev_status_quo,
                "accept_gamble": ev_gamble,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Average dollar value is {ev_gamble} for the gamble "
                f"and {ev_status_quo} for the status quo."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _loss_aversion_gap_matches_regime(self, *, regime: str, gap: float) -> bool:
        if regime == "accept_gamble":
            return gap >= 6.0
        if regime == "choose_status_quo":
            return gap <= -6.0
        return abs(gap) <= 3.0

    def _loss_aversion_realized_regime_from_gap(self, gap: float) -> str:
        if gap >= 6.0:
            return "accept_gamble"
        if gap <= -6.0:
            return "choose_status_quo"
        if abs(gap) <= 3.0:
            return "near_indifferent"
        return "ambiguous_band"

    def _loss_aversion_counterexample_fallback(
        self, *, regime: str, ev_status_quo: float
    ) -> tuple[float, int, int, float]:
        if regime == "accept_gamble":
            p_gain = 0.7
            loss = -160
            target_gap = 10.0
        elif regime == "choose_status_quo":
            p_gain = 0.45
            loss = -160
            target_gap = -10.0
        else:
            p_gain = 0.45
            loss = -160
            target_gap = 0.0

        target_ev_gamble = ev_status_quo + target_gap
        gain = int(round((target_ev_gamble - (1 - p_gain) * loss) / p_gain))
        gain = min(max(gain, 40), 320)
        ev_gamble = self._round_expected_utility(p_gain * gain + (1 - p_gain) * loss)
        gap = ev_gamble - ev_status_quo
        if not self._loss_aversion_gap_matches_regime(regime=regime, gap=gap):
            raise RuntimeError(
                "Unable to construct loss_aversion_counterexample matching requested regime."
            )
        return p_gain, gain, loss, ev_gamble

    def _generate_hyperbolic_discounting_counterexample(
        self, sample_index: int
    ) -> DataPoint:
        current_index = sample_index
        earlier_delay_days = self.rng.choice([0, 7, 14, 21])
        later_delay_days = earlier_delay_days + self.rng.choice([7, 14, 21, 30, 45])
        annual_rate = round(self.rng.uniform(0.03, 0.2), 4)
        earlier_amount = self.rng.randint(40, 180)
        earlier_discount_factor = 1 / (1 + annual_rate * (earlier_delay_days / 365))
        later_discount_factor = 1 / (1 + annual_rate * (later_delay_days / 365))
        later_break_even = round(
            earlier_amount * (earlier_discount_factor / later_discount_factor), 2
        )
        regime = self.rng.choice(
            [
                "choose_later",
                "choose_earlier",
                "near_indifferent",
            ]
        )
        if regime == "choose_later":
            later_amount = round(later_break_even + self.rng.uniform(4, 25), 2)
        elif regime == "choose_earlier":
            later_amount = round(max(1, later_break_even - self.rng.uniform(4, 25)), 2)
        else:
            later_amount = round(max(1, later_break_even + self.rng.uniform(-2, 2)), 2)

        pv_earlier = self._round_expected_utility(earlier_amount * earlier_discount_factor)
        pv_later = self._round_expected_utility(later_amount * later_discount_factor)
        optimal = self._choose_optimal_action(
            left_label="choose_later",
            left_value=pv_later,
            right_label="choose_earlier",
            right_value=pv_earlier,
        )
        decision_values = {
            "choose_later": pv_later,
            "choose_earlier": pv_earlier,
        }
        comparison_pair = self._comparison_pair_for_subtype(
            "hyperbolic_discounting_counterexample"
        )
        problem_spec = self._build_hyperbolic_discounting_counterexample_problem_spec(
            earlier_amount=earlier_amount,
            earlier_delay_days=earlier_delay_days,
            later_amount=later_amount,
            later_delay_days=later_delay_days,
            annual_rate=annual_rate,
        )
        outcome_model = self._build_hyperbolic_discounting_counterexample_outcome_model(
            problem_spec=problem_spec
        )
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="hyperbolic_discounting_counterexample",
                problem_spec=problem_spec,
                numeric_values=[
                    earlier_amount,
                    earlier_delay_days,
                    later_amount,
                    later_delay_days,
                    annual_rate,
                ],
                comparison_pair=comparison_pair,
            )
        )
        difficulty_metrics = self._difficulty_metrics(
            left_value=pv_later,
            right_value=pv_earlier,
            numeric_complexity=self._compute_numeric_complexity(
                numeric_values=[
                    earlier_amount,
                    earlier_delay_days,
                    later_amount,
                    later_delay_days,
                    annual_rate,
                ],
                arithmetic_operations=self._count_operations_in_outcome_model(
                    outcome_model
                ),
            ),
            time_horizon_days=later_delay_days,
            prompt_complexity_features=prompt_complexity_features,
        )
        difficulty_metrics.update(
            {
                "earlier_delay_days": earlier_delay_days,
                "later_delay_days": later_delay_days,
                "delay_gap_days": later_delay_days - earlier_delay_days,
                "later_break_even_amount": later_break_even,
                "later_minus_earlier_pv_signed": round(pv_later - pv_earlier, 4),
                "timing_regime_intended": regime,
                "timing_regime_realized": self._hyperbolic_realized_regime_from_gap(
                    pv_later - pv_earlier
                ),
            }
        )

        return self._assemble_normative_datapoint(
            sample_index=current_index,
            task_subtype="hyperbolic_discounting_counterexample",
            task_id_prefix="hyperbolic",
            problem_spec=problem_spec,
            prompt=prompt,
            state={
                "time_horizon_days": problem_spec["options"]["B"]["delay_days"],
                "options": problem_spec["options"],
            },
            actions=["choose_earlier", "choose_later", "indifferent"],
            comparison_pair=comparison_pair,
            outcome_model=outcome_model,
            action_values={
                "choose_earlier": pv_earlier,
                "choose_later": pv_later,
            },
            decision_values=decision_values,
            optimal_decision=optimal,
            brief_rationale=(
                f"Discounted value of later payment is {pv_later}, compared with "
                f"{pv_earlier} for the earlier payment."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _hyperbolic_realized_regime_from_gap(self, gap: float) -> str:
        if gap >= 4.0:
            return "choose_later"
        if gap <= -4.0:
            return "choose_earlier"
        if abs(gap) <= 2.0:
            return "near_indifferent"
        return "ambiguous_band"
