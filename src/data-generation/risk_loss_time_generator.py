"""
Risk/loss/time dataset generator for normative decision tasks.

These samples are constructed as normative decision problems where the optimal
action follows expected-value maximization under linear utility; time tasks use
simple annual discounting to compare present values.
"""

import hashlib
import json
import random
import re
from typing import Callable, Literal

from base_generator import BaseGenerator
from difficulty_config import RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE

from schema import (
    ActionScalars,
    CeOfferComparisonProblemSpec,
    ComparisonPair,
    DataPoint,
    ExpectedValueAssumptions,
    LotteryProblemSpec,
    Metadata,
    MixedGainLossProblemSpec,
    ProblemSpec,
    RiskLossTimeTaskSubtype,
    SolverTrace,
    Target,
    TimeDiscountingAssumptions,
    TimeDiscountingProblemSpec,
)

TaskSubtype = RiskLossTimeTaskSubtype
DifficultyMetrics = dict[str, float | int | bool | str]
PromptStyle = Literal[
    "default",
    "formal",
    "plain_english",
    "compact",
    "finance_framed",
    "unlabeled",
    "random",
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
}
PROMPT_RENDERER_METHOD_BY_SUBTYPE: dict[TaskSubtype, str] = {
    "lottery_choice": "_render_lottery_choice_prompt",
    "ce_offer_comparison": "_render_ce_offer_comparison_prompt",
    "mixed_gain_loss_choice": "_render_mixed_gain_loss_choice_prompt",
    "time_discounting": "_render_time_discounting_prompt",
}


class RiskLossTimeGenerator(BaseGenerator):
    """Generate normative expected-value decision tasks for risk/loss/time choices."""
    EXPECTED_UTILITY_PRECISION = 4
    CHOICE_TIE_EPSILON = 1e-6

    def __init__(
        self,
        seed: int | None = None,
        version: str = "v1",
        prompt_style: PromptStyle = "default",
    ):
        if prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(
                f"Unsupported prompt_style: {prompt_style}. "
                f"Expected one of {SUPPORTED_PROMPT_STYLES}."
            )
        self.rng = random.Random(seed)
        self.base_seed = seed if seed is not None else 0
        self.version = version
        self.prompt_style = prompt_style
        self.sample_index = 0

    def _resolve_prompt_style(self) -> PromptStyle:
        if self.prompt_style == "random":
            return self.rng.choice(list(NON_RANDOM_PROMPT_STYLES))
        return self.prompt_style

    def generate(self) -> DataPoint:
        current_index = self.sample_index
        self.sample_index += 1
        subtype: TaskSubtype = self.rng.choice([
            "lottery_choice",
            "ce_offer_comparison",
            "mixed_gain_loss_choice",
            "time_discounting",
        ])

        if subtype == "lottery_choice":
            return self._generate_lottery_choice(sample_index=current_index)
        elif subtype == "ce_offer_comparison":
            return self._generate_ce_offer_comparison(sample_index=current_index)
        elif subtype == "mixed_gain_loss_choice":
            return self._generate_mixed_gain_loss_choice(sample_index=current_index)
        elif subtype == "time_discounting":
            return self._generate_time_discounting(sample_index=current_index)
        else:
            raise ValueError(f"Unknown subtype: {subtype}")

    def _metadata(
        self,
        sample_index: int,
        difficulty_metrics: DifficultyMetrics,
        resolved_prompt_style: PromptStyle,
        example_fingerprint: str,
        tie_threshold: float,
    ) -> Metadata:
        prompt_has_action_labels = resolved_prompt_style != "unlabeled"
        return Metadata(
            generator_name=self.__class__.__name__,
            version=self.version,
            seed=self.base_seed,
            dataset_role="normative_training",
            requested_prompt_style=self.prompt_style,
            resolved_prompt_style=resolved_prompt_style,
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

    def _comparison_pair_for_subtype(self, subtype: TaskSubtype) -> ComparisonPair:
        pair = COMPARISON_PAIR_BY_SUBTYPE[subtype]
        return {
            "left_action": pair["left_action"],
            "right_action": pair["right_action"],
        }

    def _prompt_renderer_for_subtype(
        self, subtype: TaskSubtype
    ) -> Callable[[ProblemSpec, PromptStyle], str]:
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

    def _verify_target_solution(self, *, problem_spec: ProblemSpec, target: Target) -> None:
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
        self._verify_target_solution(problem_spec=problem_spec, target=target)
        example_fingerprint = self._compute_example_fingerprint(
            task_subtype=task_subtype,
            problem_spec=problem_spec,
            optimal_decision=target.optimal_decision,
        )
        return DataPoint(
            task_family="risk_loss_time_choice",
            task_subtype=task_subtype,
            task_id=self._task_id(task_id_prefix, sample_index),
            difficulty=self._difficulty_for(task_subtype, difficulty_metrics),
            problem_spec=problem_spec,
            input=prompt,
            target=target,
            metadata=self._metadata(
                sample_index,
                difficulty_metrics,
                prompt_style,
                example_fingerprint,
                tie_threshold,
            ),
        )

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
        numeric_values: list[int | float],
        comparison_pair: ComparisonPair,
        includes_signed_outcomes: bool = False,
    ) -> DifficultyMetrics:
        lower_prompt = prompt.lower()
        clause_count = 1 + sum(
            lower_prompt.count(token)
            for token in (",", ";", " or ", " and ", " otherwise ", " instead ")
        )
        has_decimal_in_prompt = bool(re.search(r"\d+\.\d+", prompt))
        has_positive = any(float(value) > 0 for value in numeric_values)
        has_negative = any(float(value) < 0 for value in numeric_values)
        mixed_signed_outcomes = includes_signed_outcomes or (has_positive and has_negative)
        action_tokens = re.findall(
            r"\b(?:choose_[a-z0-9_]+|accept_[a-z0-9_]+|reject_[a-z0-9_]+)\b",
            lower_prompt,
        )
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
        prompt = renderer(problem_spec=problem_spec, style=prompt_style)
        prompt_complexity_features = self._compute_prompt_complexity_features(
            prompt=prompt,
            prompt_style=prompt_style,
            numeric_values=numeric_values,
            comparison_pair=comparison_pair,
            includes_signed_outcomes=includes_signed_outcomes,
        )
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

    def _render_lottery_choice_prompt(
        self, *, problem_spec: LotteryProblemSpec, style: PromptStyle
    ) -> str:
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
        if style == "formal":
            return (
                f"Option A (choose_sure) yields a certain payoff of ${sure_text}. "
                f"Option B (choose_lottery) yields ${win_text} with probability "
                f"{win_probability_text}% and ${lose_text} otherwise. "
                "Select the action with the higher expected monetary value."
            )
        if style == "plain_english":
            return (
                f"You can take ${sure_text} for sure (choose_sure), or gamble "
                f"(choose_lottery): {win_probability_text}% chance of ${win_text}, "
                f"else ${lose_text}. Pick the option worth more on average."
            )
        if style == "compact":
            return (
                f"choose_sure: ${sure_text} certain; choose_lottery: "
                f"{win_probability_text}%*${win_text} else ${lose_text}. "
                "Objective: maximize expected value."
            )
        if style == "finance_framed":
            return (
                f"Allocate to a risk-free payoff (choose_sure): ${sure_text}, "
                f"or a risky ticket (choose_lottery): {win_probability_text}% of "
                f"${win_text}, otherwise ${lose_text}. "
                "Choose the higher expected dollar return."
            )
        if style == "unlabeled":
            return (
                f"Option A gives ${sure_text} for certain. Option B gives ${win_text} "
                f"with probability {win_probability_text}% and ${lose_text} otherwise. "
                "Which option has the higher expected monetary value?"
            )
        return (
            f"Choose the sure option (choose_sure): receive ${sure_text} for certain, "
            f"or choose the lottery (choose_lottery): a {win_probability_text}% chance "
            f"of receiving ${win_text}, otherwise ${lose_text}. "
            "Assume your objective is to maximize monetary value."
        )

    def _render_ce_offer_comparison_prompt(
        self, *, problem_spec: CeOfferComparisonProblemSpec, style: PromptStyle
    ) -> str:
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
        if style == "formal":
            return (
                f"A lottery offers ${high_text} with probability {win_probability_text}% "
                f"and ${low_text} otherwise. A sure payment of ${offered_text} is "
                "available as an alternative. Select accept_offer or reject_offer "
                "according to higher expected monetary value."
            )
        if style == "plain_english":
            return (
                f"The lottery pays ${high_text} {win_probability_text}% of the time "
                f"and ${low_text} otherwise. You can take a sure ${offered_text} now. "
                "Choose accept_offer for the sure amount or reject_offer to keep the lottery."
            )
        if style == "compact":
            return (
                f"Lottery: {win_probability_text}%*${high_text}, else ${low_text}. "
                f"Offer: ${offered_text} certain. "
                "Action: accept_offer vs reject_offer by expected value."
            )
        if style == "finance_framed":
            return (
                f"A contingent payout contract returns ${high_text} with "
                f"{win_probability_text}% probability and ${low_text} otherwise. "
                f"Counterparty offers a cash-out at ${offered_text}. "
                "Choose accept_offer or reject_offer based on higher EV."
            )
        if style == "unlabeled":
            return (
                f"A lottery pays ${high_text} with probability {win_probability_text}% "
                f"and ${low_text} otherwise. You can instead take a certain "
                f"${offered_text}. Which option has the higher expected monetary value?"
            )
        return (
            f"A lottery pays ${high_text} with probability {win_probability_text}% "
            f"and ${low_text} otherwise. You are offered a certain ${offered_text} "
            "instead. This is an offer-comparison task using the lottery's "
            "certainty-equivalent benchmark under linear utility. "
            "Choose whether to accept the certain offer (accept_offer) "
            "or keep the lottery (reject_offer), based on higher expected value."
        )

    def _render_mixed_gain_loss_choice_prompt(
        self, *, problem_spec: MixedGainLossProblemSpec, style: PromptStyle
    ) -> str:
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
        if style == "formal":
            return (
                f"Option A (choose_sure) yields a certain payoff of ${sure_text}. "
                f"Option B (choose_risky) yields ${gain_text} with probability "
                f"{gain_probability_text}% and ${loss_text} otherwise. "
                "Select the option with higher expected monetary value."
            )
        if style == "plain_english":
            return (
                f"Take ${sure_text} for sure (choose_sure), or take the risky option "
                f"(choose_risky): {gain_probability_text}% chance of ${gain_text}, "
                f"otherwise ${loss_text}. Pick the better average-value choice."
            )
        if style == "compact":
            return (
                f"choose_sure: ${sure_text}; choose_risky: "
                f"{gain_probability_text}%*${gain_text} else ${loss_text}. "
                "Use expected value."
            )
        if style == "finance_framed":
            return (
                f"Compare a guaranteed cashflow (choose_sure): ${sure_text} versus "
                f"a risky exposure (choose_risky): {gain_probability_text}% of "
                f"${gain_text}, otherwise ${loss_text}. "
                "Choose the position with higher expected dollar return."
            )
        if style == "unlabeled":
            return (
                f"Option A gives ${sure_text} for sure. Option B gives ${gain_text} "
                f"with probability {gain_probability_text}% and ${loss_text} otherwise. "
                "Which option has the higher expected monetary value?"
            )
        return (
            f"Choose the sure payoff (choose_sure): ${sure_text}. "
            f"Or choose the risky option (choose_risky): ${gain_text} with "
            f"probability {gain_probability_text}% and ${loss_text} otherwise. "
            "Assume risk-neutral expected value maximization."
        )

    def _render_time_discounting_prompt(
        self, *, problem_spec: TimeDiscountingProblemSpec, style: PromptStyle
    ) -> str:
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
        if style == "formal":
            return (
                f"Option A (choose_now) pays ${now_text} immediately. "
                f"Option B (choose_later) pays ${later_text} after {days_text} days. "
                f"Evaluate using simple annual discounting at r={rate_text}."
            )
        if style == "plain_english":
            return (
                f"You can get ${now_text} today (choose_now), or ${later_text} in "
                f"{days_text} days (choose_later). Use discount rate r={rate_text} "
                "to compare what each is worth today."
            )
        if style == "compact":
            return (
                f"choose_now: ${now_text} today; choose_later: ${later_text} in "
                f"{days_text} days; discount rate r={rate_text} (simple annual)."
            )
        if style == "finance_framed":
            return (
                f"Immediate settlement (choose_now): ${now_text}. Deferred settlement "
                f"(choose_later): ${later_text} at T={days_text} days. "
                f"Discount cashflows at simple annual rate r={rate_text}."
            )
        if style == "unlabeled":
            return (
                f"Option A pays ${now_text} today. Option B pays ${later_text} "
                f"in {days_text} days. Compare them using simple annual discounting "
                f"at rate r={rate_text}."
            )
        return (
            f"Choose now (choose_now): ${now_text} today, or choose later "
            f"(choose_later): ${later_text} in {days_text} days. "
            f"Use annual simple discount rate r={rate_text}."
        )

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
                f"Expected value of the lottery is {ev_lottery}, "
                f"compared with {ev_sure} for the certain option."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )

    def _generate_ce_offer_comparison(self, sample_index: int) -> DataPoint:
        current_index = sample_index
        # CE-based offer comparison task: choose between a lottery and a
        # certain offer positioned around the lottery CE under linear utility.
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
                "This CE-based comparison uses the lottery's certainty-equivalent "
                f"benchmark ({ce}) under linear utility versus the offered certain amount."
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
                f"Option B expected value is {ev_risky}, Option A expected value is {ev_sure}."
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
                f"Present value of later option is {pv_later}, compared with {ev_now} now."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )
