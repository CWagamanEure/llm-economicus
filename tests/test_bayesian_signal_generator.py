import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from base_generator import PROMPT_FORBIDDEN_PHRASES_BY_REGIME  # noqa: E402
from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402
from difficulty_config import (  # noqa: E402
    BAYESIAN_SIGNAL_DEFAULT_DIFFICULTY_BY_SUBTYPE,
    BAYESIAN_SIGNAL_SUBTYPES,
)


def test_generate_dispatches_all_subtypes_with_expected_task_id_prefixes():
    expected_prefix = {
        "basic_bayes_update": "bayes_basic_",
        "binary_signal_decision": "signal_decision_",
        "information_cascade_step": "cascade_",
        "noisy_signal_asset_update": "asset_update_",
    }
    for subtype, prefix in expected_prefix.items():
        gen = BayesianSignalGenerator(seed=7)

        def fake_choice(options):
            if "basic_bayes_update" in options:
                return subtype
            return options[0]

        gen.rng.choice = fake_choice
        dp = gen.generate()
        assert dp.task_subtype == subtype
        assert dp.task_id.startswith(prefix)


def test_subtype_names_and_difficulty_config_keys_are_synchronized():
    config_keys = set(BAYESIAN_SIGNAL_DEFAULT_DIFFICULTY_BY_SUBTYPE.keys())
    subtype_keys = set(BAYESIAN_SIGNAL_SUBTYPES)
    assert config_keys == subtype_keys


def test_metadata_uses_generator_seed_and_monotonic_sample_index():
    gen = BayesianSignalGenerator(seed=99)
    first = gen._generate_basic_bayes_update(0)
    second = gen._generate_binary_signal_decision(1)

    assert first.metadata.generator_name == "BayesianSignalGenerator"
    assert first.metadata.dataset_role == "normative_training"
    assert first.metadata.seed == 99
    assert first.metadata.sample_index == 0
    assert second.metadata.sample_index == 1
    assert first.metadata.tie_threshold == gen.CHOICE_TIE_EPSILON
    assert isinstance(first.metadata.example_fingerprint, str)
    assert len(first.metadata.example_fingerprint) == 64


def test_basic_bayes_update_solver_matches_closed_form():
    gen = BayesianSignalGenerator(seed=1)
    problem_spec = gen._build_basic_bayes_update_problem_spec(
        prior_high=0.4,
        p_signal_high_given_high=0.8,
        p_signal_high_given_low=0.3,
        observed_signal="high",
    )
    action_values, decision_values, optimal_decision, posterior = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    expected_posterior = (0.4 * 0.8) / ((0.4 * 0.8) + (0.6 * 0.3))
    assert action_values["choose_state_high"] == pytest.approx(expected_posterior, abs=1e-6)
    assert action_values["choose_state_low"] == pytest.approx(1 - expected_posterior, abs=1e-6)
    assert decision_values["choose_state_high"] == action_values["choose_state_high"]
    assert posterior["posterior_high"] == pytest.approx(expected_posterior, abs=1e-6)
    assert optimal_decision == "choose_state_high"


def test_prompt_qa_reports_structured_failures_for_incomplete_basic_bayes_prompt():
    gen = BayesianSignalGenerator(seed=21)
    problem_spec = gen._build_basic_bayes_update_problem_spec(
        prior_high=0.4,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
    )
    bad_prompt = "Base rate is 0.4. Which state is more likely?"
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="basic_bayes_update",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    assert failures
    assert all("code" in failure and "detail" in failure for failure in failures)
    codes = {failure["code"] for failure in failures}
    assert "missing_observed_cue" in codes
    assert "missing_cue_rate_pair" in codes


def test_prompt_qa_flags_unanchored_vs_shorthand():
    gen = BayesianSignalGenerator(seed=22)
    problem_spec = gen._build_basic_bayes_update_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="low",
    )
    bad_prompt = "Prior is 0.41. Reliability is 0.81 vs 0.25. Pick a state."
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="basic_bayes_update",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="fraud_detection",
    )
    codes = {failure["code"] for failure in failures}
    assert "unanchored_vs_shorthand" in codes or "compressed_reliability_shorthand" in codes


def test_prompt_qa_flags_raw_slash_numeric_shorthand():
    gen = BayesianSignalGenerator(seed=220)
    problem_spec = gen._build_basic_bayes_update_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
    )
    bad_prompt = (
        "The base rate is 0.41. The test is positive. "
        "Reliability is 0.81/0.25. Which state is more likely?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="basic_bayes_update",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    assert any(failure["code"] == "raw_slash_numeric_shorthand" for failure in failures)


def test_prompt_qa_reports_missing_payoffs_for_binary_signal_decision():
    gen = BayesianSignalGenerator(seed=23)
    problem_spec = gen._build_binary_signal_decision_problem_spec(
        prior_high=0.55,
        p_signal_high_given_high=0.76,
        p_signal_high_given_low=0.28,
        observed_signal="high",
        payoff_if_high=110,
        payoff_if_low=-40,
        do_not_act_payoff=10,
    )
    bad_prompt = (
        "The screening test came back positive. Base rate is 0.55 among condition-present cases. "
        "This cue appears with probability 0.76 when the condition is present and "
        "0.28 when the condition is absent. Should you act?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="binary_signal_decision",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    codes = {failure["code"] for failure in failures}
    assert "missing_action_payoffs" in codes


def test_prompt_qa_flags_invalid_base_rate_among_positive_class_wording():
    gen = BayesianSignalGenerator(seed=24)
    problem_spec = gen._build_basic_bayes_update_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
    )
    bad_prompt = (
        "The screening test came back positive. "
        "The base rate is 0.41 among condition-present cases. "
        "This result appears with probability 0.81 when the condition is present and "
        "0.25 when the condition is absent."
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="basic_bayes_update",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    assert any(
        failure["code"] == "invalid_base_rate_among_positive_class" for failure in failures
    )


def test_binary_prompt_naturalness_lint_flags_telegraphic_fragments():
    gen = BayesianSignalGenerator(seed=25)
    problem_spec = gen._build_binary_signal_decision_problem_spec(
        prior_high=0.69,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
        payoff_if_high=136,
        payoff_if_low=-26,
        do_not_act_payoff=0,
    )
    bad_prompt = (
        "The screening test came back positive. The base rate is 0.69. "
        "This result appears with probability 0.81 when the condition is present and 0.25 when "
        "the condition is absent. Decision row: act 136/-26, hold 0. Which action?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="binary_signal_decision",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    assert any(
        failure["code"] == "binary_telegraphic_fragment_notation" for failure in failures
    )


def test_binary_prompt_qa_flags_compressed_payoff_pair():
    gen = BayesianSignalGenerator(seed=26)
    problem_spec = gen._build_binary_signal_decision_problem_spec(
        prior_high=0.69,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
        payoff_if_high=136,
        payoff_if_low=-26,
        do_not_act_payoff=0,
    )
    bad_prompt = (
        "The screening test came back positive. The base rate is 0.69. "
        "This result appears with probability 0.81 when the condition is present and "
        "0.25 when the condition is absent. Acting payoff pair is 136/-26 and waiting pays 0. "
        "What action now?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="binary_signal_decision",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
    )
    assert any(
        failure["code"] == "binary_compressed_payoff_shorthand" for failure in failures
    )


def test_binary_signal_decision_rendered_prompts_are_complete_across_regimes_and_frames():
    gen = BayesianSignalGenerator(seed=26)
    problem_spec = gen._build_binary_signal_decision_problem_spec(
        prior_high=0.69,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
        payoff_if_high=136,
        payoff_if_low=-26,
        do_not_act_payoff=0,
    )
    regimes = ("normative_explicit", "neutral_realistic", "bias_eliciting")
    frames = ("medical_screening", "fraud_detection", "hiring_screen")
    for regime in regimes:
        for frame in frames:
            prompt = gen._render_binary_signal_decision_prompt(
                problem_spec=problem_spec,
                style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=frame,
            )
            failures = gen._qa_validate_rendered_prompt(
                task_subtype="binary_signal_decision",
                prompt=prompt,
                problem_spec=problem_spec,
                frame_variant=frame,
            )
            assert not failures, (
                f"prompt QA failures for regime={regime}, frame={frame}: {failures}; "
                f"prompt={prompt}"
            )


def test_binary_signal_decision_solver_uses_posterior_expected_value():
    gen = BayesianSignalGenerator(seed=2)
    problem_spec = gen._build_binary_signal_decision_problem_spec(
        prior_high=0.5,
        p_signal_high_given_high=0.9,
        p_signal_high_given_low=0.2,
        observed_signal="high",
        payoff_if_high=100,
        payoff_if_low=-40,
        do_not_act_payoff=0,
    )
    action_values, _, optimal_decision, posterior = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    posterior_high_exact = (0.5 * 0.9) / ((0.5 * 0.9) + (0.5 * 0.2))
    expected_act = round(posterior_high_exact * 100 + (1 - posterior_high_exact) * -40, 6)
    assert action_values["act"] == pytest.approx(expected_act, abs=1e-6)
    assert posterior["posterior_high"] == pytest.approx(round(posterior_high_exact, 6), abs=1e-6)
    assert optimal_decision == "act"


def test_unlabeled_prompt_style_disables_action_labels_in_metadata():
    dp = BayesianSignalGenerator(seed=15, prompt_style="unlabeled")._generate_basic_bayes_update(
        0
    )
    assert dp.metadata.requested_prompt_style == "unlabeled"
    assert dp.metadata.resolved_prompt_style == "unlabeled"
    assert dp.metadata.prompt_has_action_labels is False


def test_random_prompt_style_resolution_is_seed_deterministic():
    first = BayesianSignalGenerator(seed=41, prompt_style="random")._generate_basic_bayes_update(
        0
    )
    second = BayesianSignalGenerator(
        seed=41, prompt_style="random"
    )._generate_basic_bayes_update(0)
    assert first.metadata.resolved_prompt_style == second.metadata.resolved_prompt_style
    assert first.input == second.input


def test_prompt_frame_variant_is_recorded_in_metadata():
    dp = BayesianSignalGenerator(seed=42, prompt_style="default")._generate_basic_bayes_update(
        0
    )
    assert dp.metadata.prompt_frame_variant in {
        "medical_screening",
        "fraud_detection",
        "hiring_screen",
        "security_alert",
        "trading_signal",
        "manufacturing_defect",
    }
    assert (
        dp.metadata.prompt_frame_variant
        == dp.metadata.difficulty_metrics["prompt_frame_variant"]
    )
    assert dp.metadata.semantic_context == dp.metadata.prompt_frame_variant


def test_bayes_prompts_have_no_forbidden_normative_leakage_in_non_normative_regimes():
    generators = [
        "_generate_basic_bayes_update",
        "_generate_binary_signal_decision",
        "_generate_information_cascade_step",
        "_generate_noisy_signal_asset_update",
    ]
    for regime in ("neutral_realistic", "bias_eliciting"):
        gen = BayesianSignalGenerator(seed=23, prompt_style_regime=regime)
        forbidden = PROMPT_FORBIDDEN_PHRASES_BY_REGIME[regime]
        for idx, method_name in enumerate(generators):
            dp = getattr(gen, method_name)(idx)
            lower_prompt = dp.input.lower()
            for phrase in forbidden:
                assert phrase not in lower_prompt


def test_each_bayes_subtype_supports_all_semantic_context_templates():
    contexts = [
        "medical_screening",
        "fraud_detection",
        "security_alert",
        "hiring_screen",
        "trading_signal",
        "manufacturing_defect",
    ]
    subtype_generators = [
        "_generate_basic_bayes_update",
        "_generate_binary_signal_decision",
        "_generate_information_cascade_step",
        "_generate_noisy_signal_asset_update",
    ]
    for context in contexts:
        gen = BayesianSignalGenerator(
            seed=99,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=context,
        )
        for idx, method_name in enumerate(subtype_generators):
            dp = getattr(gen, method_name)(idx)
            assert dp.metadata.semantic_context == context
            assert dp.metadata.prompt_frame_variant == context


def test_noisy_signal_asset_update_can_choose_do_not_buy():
    gen = BayesianSignalGenerator(seed=3)
    problem_spec = gen._build_noisy_signal_asset_update_problem_spec(
        prior_high=0.2,
        p_signal_high_given_high=0.6,
        p_signal_high_given_low=0.4,
        observed_signal="low",
        value_if_high=120,
        value_if_low=60,
        market_price=130,
        transaction_cost=5,
    )
    action_values, _, optimal_decision, _ = gen._solve_from_problem_spec(problem_spec=problem_spec)
    assert action_values["buy"] < action_values["do_not_buy"]
    assert optimal_decision == "do_not_buy"


def test_subtypes_expose_action_value_semantics_metadata():
    gen = BayesianSignalGenerator(seed=23)
    datapoints = [
        gen._generate_basic_bayes_update(0),
        gen._generate_binary_signal_decision(1),
        gen._generate_information_cascade_step(2),
        gen._generate_noisy_signal_asset_update(3),
    ]
    expected = {
        "basic_bayes_update": "posterior_probability_comparison",
        "binary_signal_decision": "posterior_expected_payoff_comparison",
        "information_cascade_step": "posterior_probability_comparison",
        "noisy_signal_asset_update": "posterior_expected_payoff_comparison",
    }

    for dp in datapoints:
        metrics = dp.metadata.difficulty_metrics
        assert metrics is not None
        assert metrics["action_value_semantics"] == expected[dp.task_subtype]


def test_bayes_prompts_avoid_awkward_state_case_interpolation_patterns():
    bad_literals = (
        "in the payment is fraudulent cases",
        "in the condition is present cases",
    )
    bad_regex = re.compile(r"in the [^\\n]* is [^\\n]* cases")
    contexts = ("medical_screening", "fraud_detection", "hiring_screen")
    subtype_generators = (
        "_generate_basic_bayes_update",
        "_generate_binary_signal_decision",
        "_generate_information_cascade_step",
        "_generate_noisy_signal_asset_update",
    )
    for context in contexts:
        gen = BayesianSignalGenerator(
            seed=111,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=context,
        )
        for idx, method_name in enumerate(subtype_generators):
            prompt = getattr(gen, method_name)(idx).input.lower()
            for literal in bad_literals:
                assert literal not in prompt
            assert bad_regex.search(prompt) is None


def test_binary_signal_decision_prompts_are_complete_across_regimes_and_frames():
    contexts = (
        "medical_screening",
        "fraud_detection",
        "hiring_screen",
        "security_alert",
        "trading_signal",
        "manufacturing_defect",
    )
    regimes = ("normative_explicit", "neutral_realistic", "bias_eliciting")
    for regime in regimes:
        for context in contexts:
            gen = BayesianSignalGenerator(
                seed=314,
                prompt_style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=context,
            )
            dp = gen._generate_binary_signal_decision(0)
            assumptions = dp.problem_spec["assumptions"]
            option_a = dp.problem_spec["options"]["A"]
            option_b = dp.problem_spec["options"]["B"]
            labels = gen._context_labels(context)
            observed_sentence = gen._observed_signal_sentence(
                labels=labels, signal=assumptions["observed_signal"]
            )
            gen._validate_binary_signal_decision_prompt_completeness(
                prompt=dp.input,
                prior_text=gen._format_number(assumptions["prior_high"]),
                observed_signal=assumptions["observed_signal"],
                observed_sentence=observed_sentence,
                likelihood_high_text=gen._format_number(
                    assumptions["p_signal_high_given_high"]
                ),
                likelihood_low_text=gen._format_number(
                    assumptions["p_signal_high_given_low"]
                ),
                payoff_high_text=gen._format_number(option_a["payoff_if_high"]),
                payoff_low_text=gen._format_number(option_a["payoff_if_low"]),
                do_not_act_payoff_text=gen._format_number(option_b["payoff"]),
            )


def test_bayes_reliability_wording_lint_rejects_unqualified_shorthand():
    contexts = ("medical_screening", "fraud_detection", "hiring_screen")
    regimes = ("neutral_realistic", "bias_eliciting")
    shorthand_markers = ("signal split", "cue profile", "signal performance")
    shorthand_vs = re.compile(r"\b\d+(?:\.\d+)?\s*vs\.?\s*\d+(?:\.\d+)?\b")

    for regime in regimes:
        for context in contexts:
            gen = BayesianSignalGenerator(
                seed=501,
                prompt_style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=context,
            )
            for idx, method_name in enumerate(
                ("_generate_basic_bayes_update", "_generate_binary_signal_decision")
            ):
                dp = getattr(gen, method_name)(idx)
                prompt = dp.input.lower()
                labels = gen._context_labels(context)
                when_high = labels["when_high"].lower()
                when_low = labels["when_low"].lower()
                high_short = labels["high_short"].lower()
                low_short = labels["low_short"].lower()
                if shorthand_vs.search(prompt) or any(m in prompt for m in shorthand_markers):
                    assert ("probability" in prompt) or ("chance" in prompt)
                    assert any(
                        token in prompt
                        for token in (
                            "appears",
                            "occurs",
                            "shows up",
                            "is seen",
                        )
                    )
                    assert when_high in prompt or high_short in prompt
                    assert when_low in prompt or low_short in prompt


def _extract_reliability_phrase_bucket(prompt: str) -> str:
    lower = prompt.lower()
    if "is seen with probability" in lower:
        return "seen_with_probability"
    if "signal occurs with probability" in lower:
        return "signal_occurs_with_probability"
    if "shows up with probability" in lower:
        return "shows_up_with_probability"
    if "is true, this signal appears with probability" in lower:
        return "when_true_appears_probability"
    if "chance of this signal is" in lower:
        return "chance_in_cases"
    if "conditional signal probability is" in lower:
        return "conditional_signal_probability"
    if "this cue appears with probability" in lower:
        return "cue_appears_with_probability"
    return "other"


def test_reliability_wording_rotation_prevents_single_phrase_dominance():
    gen = BayesianSignalGenerator(
        seed=222,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="auto",
    )
    buckets = []
    for i in range(40):
        if i % 2 == 0:
            dp = gen._generate_basic_bayes_update(i)
        else:
            dp = gen._generate_binary_signal_decision(i)
        buckets.append(_extract_reliability_phrase_bucket(dp.input))
    counts = Counter(buckets)
    assert len(counts) >= 3, (
        "Expected at least 3 reliability-phrasing variants in a batch, got "
        f"{counts}."
    )
    max_share = max(counts.values()) / sum(counts.values())
    assert max_share <= 0.8, (
        "One explicit probability phrase dominates the batch too heavily: "
        f"{counts}."
    )


def test_bayes_bias_prompts_do_not_name_cognitive_mechanisms():
    contexts = ("medical_screening", "fraud_detection", "hiring_screen")
    forbidden = (
        "anchoring",
        "cognitive risk",
        "cognitive bias",
        "first-impression bias",
        "first impression bias",
        "bias risk",
    )
    for context in contexts:
        gen = BayesianSignalGenerator(
            seed=612,
            prompt_style="default",
            prompt_style_regime="bias_eliciting",
            prompt_frame_variant=context,
        )
        for idx, method_name in enumerate(
            ("_generate_basic_bayes_update", "_generate_binary_signal_decision")
        ):
            prompt = getattr(gen, method_name)(idx).input.lower()
            for marker in forbidden:
                assert marker not in prompt
