"""
Bayesian updating and signal extraction dataset generator.

These samples are normative Bayesian decision problems where posterior beliefs
are updated from binary signals and mapped to binary actions using
expected-value-maximizing decision rules.
"""

import hashlib
import json
import random
import re
from typing import Callable, Literal

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


class BayesianSignalGenerator(BaseGenerator):
    """Generate normative Bayesian update tasks with pairwise decisions."""

    EXPECTED_UTILITY_PRECISION = 6
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
            "basic_bayes_update",
            "binary_signal_decision",
            "information_cascade_step",
            "noisy_signal_asset_update",
        ])
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
        example_fingerprint: str,
        tie_threshold: float,
    ) -> Metadata:
        return Metadata(
            generator_name=self.__class__.__name__,
            version=self.version,
            seed=self.base_seed,
            dataset_role="normative_training",
            requested_prompt_style=self.prompt_style,
            resolved_prompt_style=resolved_prompt_style,
            prompt_has_action_labels=resolved_prompt_style != "unlabeled",
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

    def _require_signal(
        self, mapping: dict[str, object], *, key: str, field_path: str, subtype: str
    ) -> Literal["high", "low"]:
        if key not in mapping:
            raise ValueError(f"{field_path}.{key} is required for subtype '{subtype}'.")
        value = mapping[key]
        if value not in ("high", "low"):
            raise ValueError(
                f"{field_path}.{key} must be 'high' or 'low' for subtype '{subtype}'."
            )
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

    def _verify_target_solution(self, *, problem_spec: ProblemSpec, target: Target) -> None:
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
        self._verify_target_solution(problem_spec=problem_spec, target=target)
        example_fingerprint = self._compute_example_fingerprint(
            task_subtype=task_subtype,
            problem_spec=problem_spec,
            optimal_decision=target.optimal_decision,
        )
        return DataPoint(
            task_family="bayesian_signal_extraction",
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
        numeric_values: list[int | float],
        comparison_pair: ComparisonPair,
    ) -> DifficultyMetrics:
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
        asymmetric_choice_framing = left_action.split("_", 1)[0] != right_action.split(
            "_", 1
        )[0]
        prompt_complexity = (
            clause_count
            + int(has_decimal_in_prompt)
            + int(asymmetric_choice_framing)
            + int(prompt_contains_action_tokens)
        )
        return {
            "prompt_style_variant": prompt_style,
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
        prompt = renderer(problem_spec=problem_spec, style=prompt_style)
        prompt_complexity_features = self._compute_prompt_complexity_features(
            prompt=prompt,
            prompt_style=prompt_style,
            numeric_values=numeric_values,
            comparison_pair=comparison_pair,
        )
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

    def _render_basic_bayes_update_prompt(
        self, *, problem_spec: BasicBayesUpdateProblemSpec, style: PromptStyle
    ) -> str:
        assumptions = problem_spec["assumptions"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        if style == "formal":
            return (
                f"Let P(H)= {prior}, P(s=high|H)= {p_sh_h}, and "
                f"P(s=high|L)= {p_sh_l}. You observe s={signal}. "
                "Update beliefs using Bayes rule and select the more likely state: "
                "choose_state_high or choose_state_low."
            )
        if style == "plain_english":
            return (
                f"Start with a {prior} chance the state is high. "
                f"The signal says 'high' with chance {p_sh_h} when high is true, "
                f"and {p_sh_l} when low is true. You observed a {signal} signal. "
                "Which state is now more likely?"
            )
        if style == "compact":
            return (
                f"P(H)={prior}; P(s=high|H)={p_sh_h}; P(s=high|L)={p_sh_l}; s={signal}. "
                "Action: choose_state_high vs choose_state_low."
            )
        if style == "finance_framed":
            return (
                f"Prior fundamental-high probability is {prior}. "
                f"Signal model: P(pos|high)={p_sh_h}, P(pos|low)={p_sh_l}. "
                f"Observed signal={signal}. Update the implied fundamental probability "
                "and pick the more likely regime."
            )
        if style == "unlabeled":
            return (
                f"Prior probability of the high state is {prior}. "
                f"P(signal=high | high)={p_sh_h}, P(signal=high | low)={p_sh_l}. "
                f"Observed signal is {signal}. Which state is now more likely?"
            )
        return (
            f"Bayesian update task: prior high-state probability is {prior}. "
            f"Signal reliability is P(high-signal|high-state)={p_sh_h} and "
            f"P(high-signal|low-state)={p_sh_l}. Observed signal={signal}. "
            "Choose state_high (choose_state_high) or state_low (choose_state_low)."
        )

    def _render_binary_signal_decision_prompt(
        self, *, problem_spec: BinarySignalDecisionProblemSpec, style: PromptStyle
    ) -> str:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        option_b = problem_spec["options"]["B"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        payoff_high = self._format_number(option_a["payoff_if_high"])
        payoff_low = self._format_number(option_a["payoff_if_low"])
        wait_payoff = self._format_number(option_b["payoff"])
        if style == "formal":
            return (
                f"P(H)={prior}, P(s=high|H)={p_sh_h}, P(s=high|L)={p_sh_l}, observed s={signal}. "
                f"If act: payoff is {payoff_high} in H and {payoff_low} in L. "
                f"If do_not_act: payoff is {wait_payoff}. "
                "Choose the action with higher posterior expected payoff."
            )
        if style == "plain_english":
            return (
                f"Chance of high state starts at {prior}. The signal model is {p_sh_h} vs "
                f"{p_sh_l}, and you saw signal={signal}. Acting pays {payoff_high} if high "
                f"and {payoff_low} if low. Not acting pays {wait_payoff}. "
                "Should you act?"
            )
        if style == "compact":
            return (
                f"P(H)={prior}; signal model=({p_sh_h},{p_sh_l}); s={signal}; "
                f"act=[H:{payoff_high},L:{payoff_low}], do_not_act={wait_payoff}."
            )
        if style == "finance_framed":
            return (
                f"Prior bullish-state probability={prior}. Signal calibration: "
                f"P(pos|bull)={p_sh_h}, P(pos|bear)={p_sh_l}. Signal observed={signal}. "
                f"Trade now (act): payoff {payoff_high} in bull, {payoff_low} in bear. "
                f"No trade: {wait_payoff}. Choose max posterior EV."
            )
        if style == "unlabeled":
            return (
                f"Prior high-state probability is {prior}. Signal model: {p_sh_h} and {p_sh_l}. "
                f"Observed signal is {signal}. Taking the action yields {payoff_high} in high "
                f"state and {payoff_low} in low state; skipping yields {wait_payoff}. "
                "Which choice has higher posterior expected payoff?"
            )
        return (
            f"Update beliefs with Bayes rule using prior={prior}, "
            f"P(signal=high|high)={p_sh_h}, P(signal=high|low)={p_sh_l}, "
            f"observed signal={signal}. Then compare act (H:{payoff_high}, "
            f"L:{payoff_low}) against do_not_act ({wait_payoff})."
        )

    def _render_information_cascade_prompt(
        self, *, problem_spec: InformationCascadeProblemSpec, style: PromptStyle
    ) -> str:
        assumptions = problem_spec["assumptions"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        public_actions = assumptions.get("public_actions", [])
        history_text = ", ".join(public_actions) if public_actions else "none"
        if style == "formal":
            return (
                f"Prior P(H)={prior}. Signal model: P(s=high|H)={p_sh_h}, "
                f"P(s=high|L)={p_sh_l}. Public history={history_text}. "
                f"Private signal={signal}. Infer posterior and choose_high or choose_low."
            )
        if style == "plain_english":
            return (
                f"Start with {prior} chance the state is high. Earlier public actions were: "
                f"{history_text}. Your private signal is {signal}. "
                "Use Bayesian updating with the given signal reliability and pick high or low."
            )
        if style == "compact":
            return (
                f"P(H)={prior}; reliabilities=({p_sh_h},{p_sh_l}); public={history_text}; "
                f"private={signal}; action: choose_high vs choose_low."
            )
        if style == "finance_framed":
            return (
                f"Prior high-fundamental probability={prior}. "
                f"Public order-flow choices={history_text}. "
                f"Private signal={signal}. Signal precision: "
                f"P(pos|high)={p_sh_h}, P(pos|low)={p_sh_l}. "
                "Update posterior and pick the higher-probability state."
            )
        if style == "unlabeled":
            return (
                f"Prior high-state probability is {prior}. Public choices so far: {history_text}. "
                f"Private signal: {signal}. Signal model parameters are {p_sh_h} and {p_sh_l}. "
                "Which state is now more likely?"
            )
        return (
            f"Information-cascade step: prior={prior}, public actions={history_text}, "
            f"private signal={signal}, with P(s=high|H)={p_sh_h} and P(s=high|L)={p_sh_l}. "
            "Choose high (choose_high) or low (choose_low)."
        )

    def _render_noisy_signal_asset_update_prompt(
        self, *, problem_spec: NoisySignalAssetUpdateProblemSpec, style: PromptStyle
    ) -> str:
        assumptions = problem_spec["assumptions"]
        option_a = problem_spec["options"]["A"]
        prior = self._format_number(assumptions["prior_high"])
        p_sh_h = self._format_number(assumptions["p_signal_high_given_high"])
        p_sh_l = self._format_number(assumptions["p_signal_high_given_low"])
        signal = assumptions["observed_signal"]
        value_high = self._format_number(option_a["value_if_high"])
        value_low = self._format_number(option_a["value_if_low"])
        market_price = self._format_number(option_a["market_price"])
        transaction_cost = self._format_number(float(assumptions.get("transaction_cost", 0.0)))
        if style == "formal":
            return (
                f"Prior P(H)={prior}. Signal model: P(s=high|H)={p_sh_h}, P(s=high|L)={p_sh_l}, "
                f"observed s={signal}. Asset value is {value_high} in H and {value_low} in L. "
                f"Market price is {market_price} with transaction cost {transaction_cost}. "
                "Choose buy or do_not_buy by posterior expected payoff."
            )
        if style == "plain_english":
            return (
                f"Start with {prior} chance the asset is high value. You saw signal={signal}. "
                f"If high, value is {value_high}; if low, value is {value_low}. "
                f"Price is {market_price} and trading cost is {transaction_cost}. "
                "Should you buy?"
            )
        if style == "compact":
            return (
                f"P(H)={prior}; rel=({p_sh_h},{p_sh_l}); s={signal}; "
                f"V(H)={value_high}, V(L)={value_low}, Px={market_price}, tc={transaction_cost}; "
                "buy vs do_not_buy."
            )
        if style == "finance_framed":
            return (
                f"Implied prior high-fundamental probability={prior}. "
                f"Signal calibration: P(pos|high)={p_sh_h}, P(pos|low)={p_sh_l}. "
                f"Observed signal={signal}. Fundamental values: high={value_high}, "
                f"low={value_low}. "
                f"Quoted price={market_price}, transaction cost={transaction_cost}. "
                "Take the trade only if posterior EV exceeds cost."
            )
        if style == "unlabeled":
            return (
                f"Prior probability of high value is {prior}. "
                f"Signal model is {p_sh_h} and {p_sh_l}; observed signal is {signal}. "
                f"The asset would be worth {value_high} if high and {value_low} "
                f"if low, while market price is {market_price} and cost is {transaction_cost}. "
                "Which choice has higher posterior expected payoff?"
            )
        return (
            f"Bayesian asset update: prior={prior}, observed signal={signal}, "
            f"P(signal=high|high)={p_sh_h}, P(signal=high|low)={p_sh_l}. "
            f"Value(high)={value_high}, Value(low)={value_low}, price={market_price}, "
            f"transaction_cost={transaction_cost}. Choose buy or do_not_buy."
        )

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
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
                task_subtype="basic_bayes_update",
                problem_spec=problem_spec,
                numeric_values=[
                    prior_high,
                    p_signal_high_given_high,
                    p_signal_high_given_low,
                ],
                comparison_pair=comparison_pair,
            )
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
                f"Posterior high-state probability is {posterior_beliefs['posterior_high']}, "
                f"low-state probability is {posterior_beliefs['posterior_low']}."
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
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
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
                f"Posterior high-state probability is {posterior_beliefs['posterior_high']}; "
                f"act EV is {decision_values['act']} versus {decision_values['do_not_act']}."
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
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
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
                f"Posterior high-state probability after public and private evidence is "
                f"{posterior_beliefs['posterior_high']}."
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
        prompt, prompt_style, prompt_complexity_features = (
            self._build_prompt_and_complexity(
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
                f"Posterior high-state probability is {posterior_beliefs['posterior_high']}; "
                f"buy EV is {decision_values['buy']} after price and costs."
            ),
            difficulty_metrics=difficulty_metrics,
            prompt_style=prompt_style,
            tie_threshold=problem_spec["assumptions"]["tie_epsilon"],
        )
