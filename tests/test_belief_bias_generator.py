import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from base_generator import PROMPT_FORBIDDEN_PHRASES_BY_REGIME  # noqa: E402
from belief_bias_generator import BeliefBiasGenerator  # noqa: E402


def _normalize_prompt(text: str) -> str:
    compact = " ".join(text.lower().split())
    compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
    compact = re.sub(r"'[^']*'", "<quoted>", compact)
    compact = re.sub(r"\b[ht]{4,}\b", "<seq>", compact)
    return compact


def _conjunction_structure_fingerprint(text: str) -> str:
    lower = text.lower()
    if any(
        marker in lower
        for marker in (
            "with that scene in mind",
            "from these details",
            "from the scene details",
            "in that situation",
            "scene momentum",
            "story momentum",
            "founder is rotating",
            "launch momentum",
        )
    ):
        return "scene_based"
    if (
        "facts only:" in lower
        or "just the facts:" in lower
        or "using these details only" in lower
        or "compare a and b directly" in lower
    ):
        return "plain_literal"
    if (
        "profile type:" in lower
        or "profile type" in lower
        or "archetype" in lower
        or "category-fit" in lower
        or "category match" in lower
        or "category-read" in lower
        or "category match" in lower
        or "sort this into a type" in lower
        or "given this profile and its details" in lower
        or "for this profile" in lower
        or "type-match" in lower
        or "profile-match" in lower
    ):
        return "type_profile"
    return "other"


def test_gambler_fallacy_problem_spec_moves_shared_context_to_assumptions():
    gen = BeliefBiasGenerator(seed=1)
    problem_spec = gen._build_gambler_fallacy_problem_spec(
        p_heads=0.5,
        queried_outcome="heads",
        recent_sequence="HHHH",
        correct_claim_in_option_a=True,
    )

    assumptions = problem_spec["assumptions"]
    assert assumptions["queried_outcome"] == "heads"
    assert assumptions["recent_sequence"] == "HHHH"
    assert set(problem_spec["options"]["A"].keys()) == {"type", "claim_type"}
    assert set(problem_spec["options"]["B"].keys()) == {"type", "claim_type"}


def test_gambler_fallacy_prompt_and_solver_read_context_from_assumptions():
    gen = BeliefBiasGenerator(seed=1)
    problem_spec = gen._build_gambler_fallacy_problem_spec(
        p_heads=0.5,
        queried_outcome="tails",
        recent_sequence="TTTTT",
        correct_claim_in_option_a=False,
    )

    prompt = gen._render_gambler_fallacy_prompt(problem_spec=problem_spec, style="default")
    assert "TTTTT" in prompt
    assert "tails" in prompt

    action_values, _, optimal, _ = gen._solve_from_problem_spec(problem_spec=problem_spec)
    assert action_values["choose_A"] == 0.0
    assert action_values["choose_B"] == 1.0
    assert optimal == "choose_B"


def test_generated_gambler_fallacy_state_and_options_are_non_redundant():
    gen = BeliefBiasGenerator(seed=7)
    dp = gen._generate_gambler_fallacy(0)
    assumptions = dp.problem_spec["assumptions"]

    assert dp.problem_spec["task_subtype"] == "gambler_fallacy"
    assert dp.target.state["recent_sequence"] == assumptions["recent_sequence"]
    assert dp.target.state["queried_outcome"] == assumptions["queried_outcome"]
    for option in dp.problem_spec["options"].values():
        assert "recent_sequence" not in option
        assert "queried_outcome" not in option


def test_overprecision_solver_requires_center_inside_interval_bounds():
    gen = BeliefBiasGenerator(seed=9)
    problem_spec = gen._build_overprecision_calibration_problem_spec(
        center_a=11.0,
        lower_a=12.0,
        upper_a=14.0,
        center_b=10.0,
        lower_b=8.0,
        upper_b=12.0,
        true_value_mean=10.0,
        true_value_sd=2.0,
    )

    with pytest.raises(ValueError, match="center must lie within \\[lower, upper\\]"):
        gen._solve_from_problem_spec(problem_spec=problem_spec)


def test_sample_size_neglect_metadata_marks_proportion_threshold_basis():
    gen = BeliefBiasGenerator(seed=13)
    dp = gen._generate_sample_size_neglect(0)
    metrics = dp.metadata.difficulty_metrics

    assert metrics is not None
    assert metrics["extreme_event_basis"] == "sample_proportion_threshold"
    assert "extreme_count_cutoff_A" in metrics
    assert "extreme_count_cutoff_B" in metrics


def test_belief_bias_subtypes_expose_action_value_semantics_metadata():
    gen = BeliefBiasGenerator(seed=21)
    datapoints = [
        gen._generate_base_rate_neglect(0),
        gen._generate_conjunction_fallacy(1),
        gen._generate_gambler_fallacy(2),
        gen._generate_sample_size_neglect(3),
        gen._generate_overprecision_calibration(4),
    ]

    expected = {
        "base_rate_neglect": "posterior_probability_comparison",
        "conjunction_fallacy": "probability_comparison",
        "gambler_fallacy": "claim_correctness",
        "sample_size_neglect": "binomial_tail_probability_comparison",
        "overprecision_calibration": "interval_coverage_comparison",
    }

    for dp in datapoints:
        metrics = dp.metadata.difficulty_metrics
        assert metrics is not None
        assert metrics["action_value_semantics"] == expected[dp.task_subtype]


def test_base_rate_neglect_frame_native_rendering_has_no_interpret_prefix():
    gen = BeliefBiasGenerator(
        seed=1510,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="hiring_screen",
    )
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.7,
        p_signal_high_given_low=0.2,
        observed_signal="high",
    )
    prompt = gen._render_base_rate_neglect_prompt(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="hiring_screen",
    )
    assert "interpret high as" not in prompt.lower()
    lower = prompt.lower()
    assert "candidate" in lower
    assert "screen" in lower
    assert "high-risk" not in lower
    assert "credit screen" not in lower


@pytest.mark.parametrize(
    ("frame_variant", "required_markers"),
    (
        ("medical_screening", ("condition", "screening test")),
        ("fraud_detection", ("payment", "fraud")),
        ("hiring_screen", ("candidate", "screen")),
        ("security_alert", ("threat", "alert")),
        ("trading_signal", ("market", "signal")),
        ("neutral_statistical", ("class", "signal")),
    ),
)
def test_base_rate_neglect_prompt_contains_frame_markers(
    frame_variant: str, required_markers: tuple[str, str]
):
    gen = BeliefBiasGenerator(
        seed=1511,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant=frame_variant,
    )
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.35,
        p_signal_high_given_high=0.8,
        p_signal_high_given_low=0.3,
        observed_signal="low",
    )
    prompt = gen._render_base_rate_neglect_prompt(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant=frame_variant,
    ).lower()
    assert all(marker in prompt for marker in required_markers)


def test_prompt_qa_flags_base_rate_frame_mismatch():
    gen = BeliefBiasGenerator(seed=1512)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.31,
        p_signal_high_given_high=0.9,
        p_signal_high_given_low=0.18,
        observed_signal="high",
    )
    bad_prompt = (
        "Interpret high as strong-fit and low as weak-fit. "
        "A credit screen has default base rate 0.31. "
        "The alert appears with chance 0.9 for high-risk and 0.18 for low-risk. "
        "This case came back positive. Which class is more likely?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_prefix_interpretation_leak" in codes
    assert "base_rate_cross_frame_wording" in codes


def test_prompt_qa_flags_base_rate_missing_core_update_fields():
    gen = BeliefBiasGenerator(seed=1513)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    bad_prompt = (
        "The candidate received a weak automated screen result. "
        "Which class is more likely now?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "missing_base_rate_prior" in codes
    assert "missing_base_rate_cue_rate_pair" in codes
    assert "missing_base_rate_state_anchors" in codes


def test_prompt_qa_flags_low_signal_natural_language_raw_rate_usage():
    gen = BeliefBiasGenerator(seed=1516)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    bad_prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "This result appears with probability 0.68 when the candidate is a strong fit and "
        "0.20 when the candidate is not a strong fit. "
        "Which state is more likely?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_missing_observed_cue_pair" in codes
    assert "base_rate_low_cue_uses_raw_rates" in codes


def test_prompt_qa_allows_formal_explicit_parameterization_with_raw_positive_rates():
    gen = BeliefBiasGenerator(seed=1517)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    ok_prompt = (
        "In this hiring screen, the strong-fit base rate is 0.41. "
        "Use P(positive|the candidate is a strong fit)=0.68 and "
        "P(positive|the candidate is not a strong fit)=0.20. "
        "The candidate received a weak automated screen result. "
        "Determine whether high or low is now more probable."
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=ok_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="normative_explicit",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_prompt_qa_allows_low_signal_natural_language_with_complement_rates():
    gen = BeliefBiasGenerator(seed=1522)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    good_prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "A negative result appears with probability 0.32 when the candidate is a strong fit and "
        "0.8 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=good_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_missing_observed_cue_pair" not in codes
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_prompt_qa_flags_ambiguous_base_rate_parameterization_mixing():
    gen = BeliefBiasGenerator(seed=1518)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    bad_prompt = (
        "In this hiring screen, the strong-fit base rate is 0.41. "
        "Use P(positive|the candidate is a strong fit)=0.68 and "
        "P(positive|the candidate is not a strong fit)=0.20. "
        "This result appears with probability 0.68 when the candidate is a strong fit and "
        "0.20 when the candidate is not a strong fit. "
        "The candidate received a weak automated screen result. Which state is more likely?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="normative_explicit",
    )
    assert any(
        failure["code"] == "base_rate_ambiguous_parameterization_mix"
        for failure in failures
    )


def test_prompt_qa_allows_high_signal_explicit_formal_prompt():
    gen = BeliefBiasGenerator(seed=1523)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.31,
        p_signal_high_given_high=0.9,
        p_signal_high_given_low=0.18,
        observed_signal="high",
    )
    prompt = (
        "For this decision setting, the condition-present prevalence is 0.31. "
        "Use P(positive|the condition is present)=0.9 and "
        "P(positive|the condition is absent)=0.18. "
        "The screening test came back positive. "
        "Determine whether the condition is more likely present or absent."
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=prompt,
        problem_spec=problem_spec,
        frame_variant="medical_screening",
        prompt_style_regime="normative_explicit",
    )
    assert not failures


def _base_rate_prompt_fail_codes(
    *,
    gen: BeliefBiasGenerator,
    prompt: str,
    observed_signal: str,
    frame_variant: str,
) -> set[str]:
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41 if observed_signal == "low" else 0.31,
        p_signal_high_given_high=0.68 if observed_signal == "low" else 0.9,
        p_signal_high_given_low=0.2 if observed_signal == "low" else 0.18,
        observed_signal=observed_signal,
    )
    failures = gen._validate_base_rate_neglect_prompt_completeness(
        prompt=prompt,
        problem_spec=problem_spec,
        frame_variant=frame_variant,
    )
    return {failure["code"] for failure in failures}


def test_base_rate_focused_low_signal_formal_raw_positive_parameterization_passes():
    gen = BeliefBiasGenerator(seed=1601)
    prompt = (
        "For this decision setting, the condition-present prevalence is 0.41. "
        "The screening test is positive with probability 0.68 when the condition is present "
        "and 0.2 when the condition is absent. "
        "The screening test came back negative. "
        "Determine whether the condition is more likely present or absent."
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="medical_screening",
    )
    assert "base_rate_missing_observed_cue_pair" not in codes
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_base_rate_focused_low_signal_natural_complement_wording_passes():
    gen = BeliefBiasGenerator(seed=1602)
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "A negative result appears with probability 0.32 when the candidate is a strong fit "
        "and 0.8 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_missing_observed_cue_pair" not in codes
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_base_rate_focused_low_signal_natural_raw_positive_only_fails():
    gen = BeliefBiasGenerator(seed=1603)
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "This result appears with probability 0.68 when the candidate is a strong fit and "
        "0.2 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_missing_observed_cue_pair" in codes
    assert "base_rate_low_cue_uses_raw_rates" in codes


def test_base_rate_low_signal_natural_complements_with_swapped_state_mapping_fails():
    gen = BeliefBiasGenerator(seed=1608)
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "A negative result appears with probability 0.8 when the candidate is a strong fit and "
        "0.32 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_missing_observed_cue_pair" in codes


def test_base_rate_low_signal_natural_wording_rejects_raw_pair_even_if_complements_present():
    gen = BeliefBiasGenerator(seed=1609)
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.41. "
        "A negative result appears with probability 0.32 when the candidate is a strong fit and "
        "0.8 when the candidate is not a strong fit. "
        "This result appears with probability 0.68 when the candidate is a strong fit and "
        "0.2 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_low_cue_uses_raw_rates" in codes


def test_base_rate_low_signal_overlap_pair_correct_mapping_passes():
    gen = BeliefBiasGenerator(seed=1610)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.4,
        p_signal_high_given_high=0.8,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.4. "
        "A negative result appears with probability 0.2 when the candidate is a strong fit "
        "and 0.8 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_missing_observed_cue_pair" not in codes
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_base_rate_low_signal_overlap_pair_incorrect_mapping_fails():
    gen = BeliefBiasGenerator(seed=1611)
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.4,
        p_signal_high_given_high=0.8,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    prompt = (
        "The candidate received a weak automated screen result. "
        "The strong-fit base rate is 0.4. "
        "A negative result appears with probability 0.8 when the candidate is a strong fit "
        "and 0.2 when the candidate is not a strong fit. "
        "Which is more likely for this candidate: strong fit or not a strong fit?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="base_rate_neglect",
        prompt=prompt,
        problem_spec=problem_spec,
        frame_variant="hiring_screen",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "base_rate_missing_observed_cue_pair" in codes
    assert "base_rate_low_cue_uses_raw_rates" in codes


def test_base_rate_state_mapped_cue_pair_helper_requires_correct_attachment():
    gen = BeliefBiasGenerator(seed=16115)
    prompt_good = (
        "A negative result appears with probability 0.2 when the condition is present "
        "and 0.8 when the condition is absent."
    )
    prompt_bad = (
        "A negative result appears with probability 0.8 when the condition is present "
        "and 0.2 when the condition is absent."
    )
    assert gen._base_rate_prompt_has_state_mapped_cue_pair(
        prompt=prompt_good,
        high_value_text="0.2",
        low_value_text="0.8",
        state_high_text="the condition is present",
        state_low_text="the condition is absent",
    )
    assert not gen._base_rate_prompt_has_state_mapped_cue_pair(
        prompt=prompt_bad,
        high_value_text="0.2",
        low_value_text="0.8",
        state_high_text="the condition is present",
        state_low_text="the condition is absent",
    )


def test_base_rate_numeric_substring_safety_for_state_attachment():
    gen = BeliefBiasGenerator(seed=1612)
    prompt = (
        "A negative result appears with probability 0.25 when the candidate is a strong fit "
        "and 0.75 when the candidate is not a strong fit."
    )
    assert not gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.2",
        state_text="the candidate is a strong fit",
    )
    assert gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.25",
        state_text="the candidate is a strong fit",
    )


def test_base_rate_state_attachment_does_not_jump_across_later_mapping_clause():
    gen = BeliefBiasGenerator(seed=1613)
    prompt = (
        "A negative result appears with probability 0.2 when the condition is absent "
        "and 0.8 when the condition is present."
    )
    assert gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.2",
        state_text="the condition is absent",
    )
    assert not gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.2",
        state_text="the condition is present",
    )


def test_base_rate_state_attachment_respects_punctuation_clause_boundaries():
    gen = BeliefBiasGenerator(seed=16131)
    prompt = (
        "A negative result appears with probability 0.2 when the condition is absent, "
        "and 0.8 when the condition is present."
    )
    assert gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.2",
        state_text="the condition is absent",
    )
    assert not gen._base_rate_value_attached_to_state(
        prompt=prompt,
        value_text="0.2",
        state_text="the condition is present",
    )


def test_base_rate_focused_low_signal_mixed_without_complements_fails_ambiguity():
    gen = BeliefBiasGenerator(seed=1604)
    prompt = (
        "For this hiring screen, use P(positive|the candidate is a strong fit)=0.68 and "
        "P(positive|the candidate is not a strong fit)=0.2. "
        "This result appears with probability 0.68 when the candidate is a strong fit and "
        "0.2 when the candidate is not a strong fit. "
        "The candidate received a weak automated screen result."
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_ambiguous_parameterization_mix" in codes


def test_base_rate_focused_low_signal_mixed_with_complements_passes():
    gen = BeliefBiasGenerator(seed=16041)
    prompt = (
        "For this hiring screen, use P(positive|the candidate is a strong fit)=0.68 and "
        "P(positive|the candidate is not a strong fit)=0.2. "
        "A negative result appears with probability 0.32 when the candidate is a strong fit and "
        "0.8 when the candidate is not a strong fit. "
        "The candidate received a weak automated screen result."
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="low",
        frame_variant="hiring_screen",
    )
    assert "base_rate_missing_observed_cue_pair" not in codes
    assert "base_rate_low_cue_uses_raw_rates" not in codes
    assert "base_rate_ambiguous_parameterization_mix" not in codes


def test_base_rate_focused_high_signal_formal_raw_positive_parameterization_passes():
    gen = BeliefBiasGenerator(seed=1605)
    prompt = (
        "For this decision setting, the condition-present prevalence is 0.31. "
        "Use P(positive|the condition is present)=0.9 and "
        "P(positive|the condition is absent)=0.18. "
        "The screening test came back positive. "
        "Determine whether the condition is more likely present or absent."
    )
    codes = _base_rate_prompt_fail_codes(
        gen=gen,
        prompt=prompt,
        observed_signal="high",
        frame_variant="medical_screening",
    )
    assert not codes


def test_base_rate_observed_cue_natural_wording_detector_positive_cases():
    gen = BeliefBiasGenerator(seed=1606)
    prompts = (
        "A negative result appears with probability 0.1 when the condition is present.",
        "A positive signal appears with probability 0.9 when the payment is fraudulent.",
        "This result is seen with probability 0.32 when the candidate is a strong fit.",
        "This signal shows up with probability 0.8 when the condition is absent.",
        "A result like this shows up with probability 0.2 when class low is true.",
        "The chance of this result is 0.32 when the condition is present.",
        "The chance of a negative result is 0.8 when the condition is absent.",
        "The chance of a positive signal is 0.9 when the condition is present.",
    )
    for prompt in prompts:
        assert gen._base_rate_prompt_uses_observed_cue_natural_wording(prompt=prompt)


def test_base_rate_observed_cue_natural_wording_detector_excludes_formal_parameterization():
    gen = BeliefBiasGenerator(seed=1607)
    prompts = (
        "The screening test is positive with probability 0.9 when the condition is present.",
        "P(positive | high)=0.9 and P(positive | low)=0.2.",
        "P(signal = high | class high)=0.9 and P(signal = high | class low)=0.2.",
    )
    for prompt in prompts:
        assert not gen._base_rate_prompt_uses_observed_cue_natural_wording(prompt=prompt)


def test_base_rate_renderer_low_signal_neutral_uses_observed_cue_complements():
    gen = BeliefBiasGenerator(
        seed=1519,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="hiring_screen",
    )
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    prompt = gen._render_base_rate_neglect_prompt(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="hiring_screen",
    )
    lower = prompt.lower()
    assert "a negative result appears with probability 0.32" in lower
    assert " and 0.8 when the candidate is not a strong fit" in lower
    assert not re.search(
        r"negative result appears with probability\s+0\.68[^\n]{0,80}0\.2",
        lower,
    )


def test_base_rate_renderer_low_signal_formal_allows_explicit_positive_parameterization():
    gen = BeliefBiasGenerator(
        seed=1520,
        prompt_style="default",
        prompt_style_regime="normative_explicit",
        prompt_frame_variant="hiring_screen",
    )
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.41,
        p_signal_high_given_high=0.68,
        p_signal_high_given_low=0.2,
        observed_signal="low",
    )
    prompt = gen._render_base_rate_neglect_prompt(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime="normative_explicit",
        prompt_frame_variant="hiring_screen",
    ).lower()
    assert "is positive with probability 0.68" in prompt
    assert "and 0.2 when the candidate is not a strong fit" in prompt
    assert "the candidate received a weak automated screen result" in prompt


@pytest.mark.parametrize(
    ("frame_variant", "expected_tail"),
    (
        ("hiring_screen", "strong fit or not a strong fit?"),
        ("fraud_detection", "fraudulent or legitimate?"),
        ("medical_screening", "condition present or absent?"),
    ),
)
def test_base_rate_frame_native_endings_avoid_generic_high_low_phrase(
    frame_variant: str, expected_tail: str
):
    gen = BeliefBiasGenerator(
        seed=1521,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant=frame_variant,
    )
    problem_spec = gen._build_base_rate_neglect_problem_spec(
        prior_high=0.37,
        p_signal_high_given_high=0.81,
        p_signal_high_given_low=0.25,
        observed_signal="high",
    )
    prompt = gen._render_base_rate_neglect_prompt(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant=frame_variant,
    ).lower()
    assert expected_tail in prompt
    assert "high or low?" not in prompt


def test_prompt_qa_flags_sample_size_missing_threshold_or_sizes():
    gen = BeliefBiasGenerator(seed=1514)
    problem_spec = gen._build_sample_size_neglect_problem_spec(
        sample_size_a=20,
        sample_size_b=150,
        baseline_rate=0.45,
        extreme_threshold=0.35,
        extreme_direction="at_or_below",
    )
    bad_prompt = (
        "Two hospitals have the same long-run girl-share of 0.45. "
        "Which hospital seems better?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="sample_size_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="hospital_births",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "missing_sample_size_threshold" in codes
    assert "missing_sample_size_values" in codes
    assert "missing_sample_size_likelihood_comparison" in codes


def test_prompt_qa_flags_sample_size_prompt_without_ab_distinction():
    gen = BeliefBiasGenerator(seed=1515)
    problem_spec = gen._build_sample_size_neglect_problem_spec(
        sample_size_a=30,
        sample_size_b=150,
        baseline_rate=0.5,
        extreme_threshold=0.65,
        extreme_direction="at_or_above",
    )
    bad_prompt = (
        "Two funds have the same long-run up-share of 0.5. "
        "Event is at least 0.65 up-share. "
        "One sample size is 30 and another is 150. "
        "Which is more likely?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="sample_size_neglect",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="fund_returns",
        prompt_style_regime="neutral_realistic",
    )
    assert any(
        failure["code"] == "missing_sample_size_ab_distinction"
        for failure in failures
    )


def test_overprecision_problem_spec_serialization_uses_clean_interval_decimals():
    gen = BeliefBiasGenerator(seed=31)
    dp = gen._generate_overprecision_calibration(0)
    payload = json.dumps(dp.problem_spec, sort_keys=True)

    assert "00000000000001" not in payload
    assert "99999999999999" not in payload


def test_prompt_frame_variant_is_recorded_in_metadata():
    dp = BeliefBiasGenerator(seed=55, prompt_style="default")._generate_conjunction_fallacy(0)
    assert dp.metadata.prompt_frame_variant in {
        "vivid_description",
        "plain_probability",
        "representative_profile",
    }
    assert (
        dp.metadata.prompt_frame_variant
        == dp.metadata.difficulty_metrics["prompt_frame_variant"]
    )
    assert dp.metadata.conjunction_render_mode in {
        "normative_explicit",
        "neutral_realistic",
        "bias_eliciting",
    }
    assert dp.metadata.representativeness_strength in {"low", "medium", "high"}


def test_belief_bias_prompts_have_no_forbidden_normative_leakage_in_non_normative_regimes():
    generators = [
        "_generate_base_rate_neglect",
        "_generate_conjunction_fallacy",
        "_generate_gambler_fallacy",
        "_generate_sample_size_neglect",
        "_generate_overprecision_calibration",
    ]
    for regime in ("neutral_realistic", "bias_eliciting"):
        gen = BeliefBiasGenerator(seed=24, prompt_style_regime=regime)
        forbidden = PROMPT_FORBIDDEN_PHRASES_BY_REGIME[regime]
        for idx, method_name in enumerate(generators):
            dp = getattr(gen, method_name)(idx)
            lower_prompt = dp.input.lower()
            for phrase in forbidden:
                assert phrase not in lower_prompt


def test_gambler_normative_explicit_contains_independence_cue():
    gen = BeliefBiasGenerator(seed=91, prompt_style_regime="normative_explicit")
    dp = gen._generate_gambler_fallacy(0)
    lower_prompt = dp.input.lower()
    assert ("independence" in lower_prompt or "independent" in lower_prompt)
    assert "50/50" in lower_prompt or "fair" in lower_prompt


def test_gambler_non_normative_prompts_include_fairness_and_independence_cues():
    for regime in ("neutral_realistic", "bias_eliciting"):
        gen = BeliefBiasGenerator(seed=92, prompt_style_regime=regime)
        dp = gen._generate_gambler_fallacy(0)
        lower_prompt = dp.input.lower()
        assert "50/50" in lower_prompt or "fair" in lower_prompt
        assert "independent" in lower_prompt or "independence" in lower_prompt


def test_gambler_bias_prompts_do_not_explain_the_bias_directly():
    gen = BeliefBiasGenerator(
        seed=94,
        prompt_style="default",
        prompt_style_regime="bias_eliciting",
        prompt_frame_variant="roulette_streak",
    )
    dp = gen._generate_gambler_fallacy(0)
    lower_prompt = dp.input.lower()
    assert "people often feel" not in lower_prompt
    assert "reversal is due" not in lower_prompt


def test_gambler_metadata_records_streak_domain():
    for frame, expected in (
        ("neutral_coin", "coin"),
        ("roulette_streak", "roulette"),
        ("sports_streak", "basketball"),
        ("market_streak", "market"),
    ):
        gen = BeliefBiasGenerator(
            seed=93,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame,
        )
        dp = gen._generate_gambler_fallacy(0)
        assert dp.metadata.streak_domain == expected
        assert dp.metadata.prompt_style_regime == "neutral_realistic"


def test_gambler_normative_explicit_variants_are_not_identical_after_normalization():
    prompts = []
    for frame in ("sports_streak", "roulette_streak", "market_streak"):
        gen = BeliefBiasGenerator(
            seed=95,
            prompt_style="default",
            prompt_style_regime="normative_explicit",
            prompt_frame_variant=frame,
        )
        prompts.append(gen._generate_gambler_fallacy(0).input)

    normalized = [_normalize_prompt(p) for p in prompts]
    assert len(set(normalized)) == len(normalized)


def test_gambler_context_prompts_use_domain_native_outcome_terms():
    checks = (
        ("sports_streak", ("make", "miss")),
        ("market_streak", ("up day", "down day")),
        ("roulette_streak", ("red", "black")),
    )
    for frame_variant, required_terms in checks:
        gen = BeliefBiasGenerator(
            seed=97,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame_variant,
        )
        dp = gen._generate_gambler_fallacy(0)
        lower_prompt = dp.input.lower()
        for term in required_terms:
            assert term in lower_prompt


def test_gambler_non_coin_frames_avoid_generic_heads_tails_flip_phrase():
    forbidden_phrases = (
        "heads on the very next flip",
        "tails on the very next flip",
    )
    for regime in ("normative_explicit", "neutral_realistic", "bias_eliciting"):
        for frame_variant in ("sports_streak", "market_streak", "roulette_streak"):
            gen = BeliefBiasGenerator(
                seed=98,
                prompt_style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=frame_variant,
            )
            dp = gen._generate_gambler_fallacy(0)
            lower_prompt = dp.input.lower()
            for phrase in forbidden_phrases:
                assert phrase not in lower_prompt


def test_gambler_market_article_handling_has_no_malformed_phrasing():
    malformed_patterns = (
        r"\ban down\b",
        r"\ba up\b",
        r"\ban down day\b",
        r"\ba up day\b",
    )
    for regime in ("normative_explicit", "neutral_realistic", "bias_eliciting"):
        for queried_outcome in ("heads", "tails"):
            for correct_in_a in (True, False):
                gen = BeliefBiasGenerator(
                    seed=140,
                    prompt_style="default",
                    prompt_style_regime=regime,
                    prompt_frame_variant="market_streak",
                )
                problem_spec = gen._build_gambler_fallacy_problem_spec(
                    p_heads=0.5,
                    queried_outcome=queried_outcome,
                    recent_sequence="HHHHTTTT",
                    correct_claim_in_option_a=correct_in_a,
                )
                prompt = gen._render_gambler_fallacy_prompt(
                    problem_spec=problem_spec,
                    style="default",
                    prompt_style_regime=regime,
                    prompt_frame_variant="market_streak",
                ).lower()
                for bad in malformed_patterns:
                    assert re.search(bad, prompt) is None


def test_gambler_non_coin_contexts_use_natural_comparative_phrasing():
    for frame_variant in ("sports_streak", "market_streak", "roulette_streak"):
        gen = BeliefBiasGenerator(
            seed=141,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame_variant,
        )
        prompt = gen._generate_gambler_fallacy(0).input.lower()
        assert " than " in prompt


def test_gambler_option_wording_has_multiple_phrase_variants_per_frame():
    frames = ("sports_streak", "market_streak", "roulette_streak")
    for frame_variant in frames:
        gen = BeliefBiasGenerator(
            seed=145,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame_variant,
        )
        option_phrases: set[str] = set()
        for i in range(12):
            prompt = gen._generate_gambler_fallacy(i).input
            for line in prompt.splitlines():
                if line.startswith("- choose_A:") or line.startswith("- choose_B:"):
                    option_phrases.add(line.strip())
        assert len(option_phrases) >= 4


def test_prompt_qa_flags_non_native_coin_phrase_in_market_frame():
    gen = BeliefBiasGenerator(seed=142)
    problem_spec = gen._build_gambler_fallacy_problem_spec(
        p_heads=0.5,
        queried_outcome="heads",
        recent_sequence="HHHHTT",
        correct_claim_in_option_a=True,
    )
    bad_prompt = (
        "Market tape shows HHHHTT. "
        "choose_A: heads on the very next flip is more likely.\n"
        "choose_B: heads on the very next flip is not more likely."
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="gambler_fallacy",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="market_streak",
    )
    assert any(failure["code"] == "non_native_coin_phrase_leak" for failure in failures)


def test_prompt_qa_flags_missing_gambler_fairness_and_independence_assumptions():
    gen = BeliefBiasGenerator(seed=1430)
    problem_spec = gen._build_gambler_fallacy_problem_spec(
        p_heads=0.5,
        queried_outcome="heads",
        recent_sequence="HHHHTT",
        correct_claim_in_option_a=True,
    )
    bad_prompt = (
        "Wheel history panel shows HHHHTT. At this table, H=red and T=black. "
        "Which statement is better about the next spin?\n"
        "- choose_A: after that streak, the next spin is more likely to land red than black\n"
        "- choose_B: given this streak, the next spin is not more likely to land red than black"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="gambler_fallacy",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="roulette_streak",
        prompt_style_regime="neutral_realistic",
    )
    codes = {failure["code"] for failure in failures}
    assert "missing_gambler_fairness_assumption" in codes
    assert "missing_gambler_independence_assumption" in codes


def test_rendered_gambler_prompts_pass_new_assumption_qa_checks():
    for regime in ("normative_explicit", "neutral_realistic", "bias_eliciting"):
        for frame_variant in (
            "neutral_coin",
            "roulette_streak",
            "sports_streak",
            "market_streak",
        ):
            gen = BeliefBiasGenerator(
                seed=1431,
                prompt_style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=frame_variant,
            )
            dp = gen._generate_gambler_fallacy(0)
            failures = gen._qa_validate_rendered_prompt(
                task_subtype="gambler_fallacy",
                prompt=dp.input,
                problem_spec=dp.problem_spec,
                frame_variant=frame_variant,
                prompt_style_regime=regime,
            )
            codes = {failure["code"] for failure in failures}
            assert "missing_gambler_fairness_assumption" not in codes
            assert "missing_gambler_independence_assumption" not in codes


def test_prompt_option_distinction_accepts_ab_equals_markers():
    gen = BeliefBiasGenerator(seed=1501)
    target = gen._generate_sample_size_neglect(0).target
    prompt = "Case note: A='Hospital A', B='Hospital B'. Which one is more likely?"
    gen._assert_prompt_option_distinction(prompt=prompt, target=target)


def test_prompt_option_distinction_accepts_option_a_option_b_markers():
    gen = BeliefBiasGenerator(seed=1502)
    target = gen._generate_overprecision_calibration(0).target
    prompt = (
        "Forecast comparison: Option A: [71.2, 88.5]. Option B: [66.0, 94.0]. "
        "Which interval is more likely to contain the realized value?"
    )
    gen._assert_prompt_option_distinction(prompt=prompt, target=target)


def test_prompt_option_distinction_accepts_frame_native_ab_markers():
    gen = BeliefBiasGenerator(seed=1503)
    target = gen._generate_sample_size_neglect(0).target
    prompt = (
        "Hospital A sees 20 births per day. Hospital B sees 150 births per day. "
        "Which hospital is more likely to show an extreme day?"
    )
    gen._assert_prompt_option_distinction(prompt=prompt, target=target)


def test_prompt_option_distinction_requires_both_ab_markers():
    gen = BeliefBiasGenerator(seed=1504)
    target = gen._generate_overprecision_calibration(0).target
    prompt = "Comparison sheet: interval A is [70, 90]. Which one should we use?"
    with pytest.raises(
        ValueError,
        match="Prompt must clearly distinguish both options for choose_A/choose_B tasks.",
    ):
        gen._assert_prompt_option_distinction(prompt=prompt, target=target)


def test_prompt_option_distinction_does_not_pass_on_stray_letters():
    gen = BeliefBiasGenerator(seed=1505)
    target = gen._generate_sample_size_neglect(0).target
    prompt = "A manager wrote a brief note before a baseline review."
    with pytest.raises(
        ValueError,
        match="Prompt must clearly distinguish both options for choose_A/choose_B tasks.",
    ):
        gen._assert_prompt_option_distinction(prompt=prompt, target=target)


def test_prompt_qa_flags_missing_overprecision_intervals_or_center():
    gen = BeliefBiasGenerator(seed=143)
    problem_spec = gen._build_overprecision_calibration_problem_spec(
        center_a=100.0,
        lower_a=95.0,
        upper_a=105.0,
        center_b=100.0,
        lower_b=90.0,
        upper_b=110.0,
        true_value_mean=100.0,
        true_value_sd=5.0,
    )
    bad_prompt = "Two forecast ranges are compared. Which one seems more likely to contain value?"
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="overprecision_calibration",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="analyst_forecast",
    )
    codes = {failure["code"] for failure in failures}
    assert "missing_overprecision_center" in codes
    assert "missing_overprecision_intervals" in codes
    assert "missing_overprecision_error_scale" in codes


def test_prompt_qa_flags_ambiguous_overprecision_choice_wording_without_containment():
    gen = BeliefBiasGenerator(seed=144)
    problem_spec = gen._build_overprecision_calibration_problem_spec(
        center_a=100.0,
        lower_a=95.0,
        upper_a=105.0,
        center_b=100.0,
        lower_b=90.0,
        upper_b=110.0,
        true_value_mean=100.0,
        true_value_sd=5.0,
    )
    bad_prompt = (
        "Two forecast ranges are compared: A=[95, 105] and B=[90, 110]. "
        "Typical level is 100 with typical miss 5. Which would you pick?"
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="overprecision_calibration",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="analyst_forecast",
    )
    assert any(
        failure["code"] == "overprecision_ambiguous_choice_wording"
        for failure in failures
    )


def test_prompt_qa_flags_conjunction_rule_leakage_in_non_normative_regime():
    gen = BeliefBiasGenerator(seed=146)
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description="A short startup profile with concrete operating details.",
        semantic_domain="startup",
        event_a_label="startup grows revenue",
        event_b_detail_label="startup grows revenue and signs an enterprise partnership",
        p_event_a=0.62,
        p_event_a_and_b=0.18,
        conjunction_in_option_a=False,
    )
    bad_prompt = (
        "Case summary: founder update notes strong execution. "
        "Under the conjunction axiom, compare A='startup grows revenue' and "
        "B='startup grows revenue and signs an enterprise partnership'."
    )
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="conjunction_fallacy",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="vivid_description",
        prompt_style_regime="bias_eliciting",
    )
    assert any(failure["code"] == "conjunction_rule_leakage" for failure in failures)


def test_prompt_qa_flags_missing_conjunction_event_statement_text():
    gen = BeliefBiasGenerator(seed=147)
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description="Profile note about a researcher and lab habits.",
        semantic_domain="scientist",
        event_a_label="researcher publishes a study",
        event_b_detail_label="researcher publishes a study and presents at a conference",
        p_event_a=0.6,
        p_event_a_and_b=0.2,
        conjunction_in_option_a=False,
    )
    bad_prompt = "Case summary: lab activity is busy. Which statement is more likely?"
    failures = gen._qa_validate_rendered_prompt(
        task_subtype="conjunction_fallacy",
        prompt=bad_prompt,
        problem_spec=problem_spec,
        frame_variant="plain_probability",
        prompt_style_regime="neutral_realistic",
    )
    assert any(
        failure["code"] == "missing_conjunction_event_statement" for failure in failures
    )


def test_conjunction_structure_validation_rejects_non_subset_label():
    gen = BeliefBiasGenerator(seed=77)
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description="A short market profile suggests improving sentiment.",
        semantic_domain="market",
        event_a_label="market closes up",
        event_b_detail_label="volatility index falls",
        p_event_a=0.6,
        p_event_a_and_b=0.2,
        conjunction_in_option_a=True,
    )
    # Break subset relation intentionally while keeping conjunction role label.
    problem_spec["options"]["A"]["event_label"] = "volatility index falls and bond yields rise"

    with pytest.raises(ValueError, match="stricter than its constituent"):
        gen._solve_from_problem_spec(problem_spec=problem_spec)


def test_conjunction_target_consistency_across_render_modes():
    for regime in ("normative_explicit", "neutral_realistic", "bias_eliciting"):
        gen = BeliefBiasGenerator(seed=88, prompt_style_regime=regime)
        dp = gen._generate_conjunction_fallacy(0)
        values = dp.target.action_values
        if values["choose_A"] > values["choose_B"]:
            assert dp.target.optimal_decision == "choose_A"
        elif values["choose_B"] > values["choose_A"]:
            assert dp.target.optimal_decision == "choose_B"
        else:
            assert dp.target.optimal_decision == "indifferent"


def test_conjunction_variants_are_not_wrapper_only_startup_rewrites():
    gen = BeliefBiasGenerator(seed=96, prompt_style="default", prompt_style_regime="bias_eliciting")
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description=(
            "A startup profile highlights disciplined execution, strong customer feedback, "
            "and fast iteration."
        ),
        semantic_domain="startup",
        event_a_label="startup grows revenue",
        event_b_detail_label="startup signs an enterprise partnership",
        p_event_a=0.63,
        p_event_a_and_b=0.19,
        conjunction_in_option_a=False,
    )
    variants = {}
    for frame in ("plain_probability", "representative_profile", "vivid_description"):
        variants[frame] = gen._render_conjunction_fallacy_prompt(
            problem_spec=problem_spec,
            style="default",
            prompt_style_regime="bias_eliciting",
            prompt_frame_variant=frame,
        )

    normalized = {k: _normalize_prompt(v) for k, v in variants.items()}
    assert len(set(normalized.values())) == 3
    plain_lower = variants["plain_probability"].lower()
    assert "startup profile highlights disciplined execution" in plain_lower
    assert any(
        marker in plain_lower
        for marker in (
            "facts:",
            "facts only:",
            "just the facts:",
            "using these details only",
            "compare a and b directly",
            "which statement is more likely",
            "lean toward first",
            "without overthinking it",
        )
    )
    representative_lower = variants["representative_profile"].lower()
    assert any(
        marker in representative_lower
        for marker in (
            "archetype",
            "type-match",
            "profile-match",
            "profile type",
            "category-fit",
            "category match",
            "category-read",
            "fit instinct",
            "category pull",
            "match cue",
            "category-match read",
            "quick fit intuition",
        )
    )
    vivid_lower = variants["vivid_description"].lower()
    assert any(token in vivid_lower for token in ("founder", "launch", "investor", "customer"))


def test_conjunction_neutral_vs_bias_style_separation_markers():
    gen = BeliefBiasGenerator(seed=118, prompt_style="default")
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description=(
            "A founder profile highlights disciplined execution, frequent customer contact, "
            "and reliable follow-through across launches."
        ),
        semantic_domain="startup",
        event_a_label="startup grows revenue",
        event_b_detail_label="startup signs an enterprise partnership",
        p_event_a=0.62,
        p_event_a_and_b=0.18,
        conjunction_in_option_a=False,
    )
    frames = ("plain_probability", "representative_profile", "vivid_description")
    vivid_markers = {
        "founder",
        "launch",
        "scene",
        "story",
        "momentum",
        "buzzing",
        "archetype",
        "category-fit",
        "profile type",
        "narrative",
    }
    intuitive_markers = {
        "first impression",
        "first-glance",
        "first-pass",
        "gut-check",
        "snap",
        "instinct",
        "on-the-spot",
        "immediate",
        "first glance",
        "lean toward",
        "feels more fitting",
        "better fit",
        "representative",
    }
    neutral_discourse_markers = {
        "given this profile",
        "which statement is more likely",
        "which line is more likely",
        "using these details only",
        "choose the statement that is more likely",
        "select the more likely",
    }

    neutral_vivid = 0
    bias_vivid = 0
    neutral_intuitive = 0
    bias_intuitive = 0
    neutral_discourse_hits = 0
    bias_discourse_hits = 0

    for frame in frames:
        neutral_prompt = gen._render_conjunction_fallacy_prompt(
            problem_spec=problem_spec,
            style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame,
        ).lower()
        bias_prompt = gen._render_conjunction_fallacy_prompt(
            problem_spec=problem_spec,
            style="default",
            prompt_style_regime="bias_eliciting",
            prompt_frame_variant=frame,
        ).lower()
        neutral_vivid += sum(1 for marker in vivid_markers if marker in neutral_prompt)
        bias_vivid += sum(1 for marker in vivid_markers if marker in bias_prompt)
        neutral_intuitive += sum(1 for marker in intuitive_markers if marker in neutral_prompt)
        bias_intuitive += sum(1 for marker in intuitive_markers if marker in bias_prompt)
        assert any(marker in bias_prompt for marker in intuitive_markers)
        assert all(
            marker not in neutral_prompt
            for marker in (
                "first-glance",
                "feels more fitting",
                "better fit",
                "instant read",
                "first-pass instinct",
            )
        )
        if any(marker in neutral_prompt for marker in neutral_discourse_markers):
            neutral_discourse_hits += 1
        if any(marker in bias_prompt for marker in intuitive_markers):
            bias_discourse_hits += 1

    assert bias_vivid > neutral_vivid
    assert bias_intuitive > neutral_intuitive
    assert neutral_discourse_hits >= 2
    assert bias_discourse_hits >= 2


def test_conjunction_frame_variants_have_distinct_structural_fingerprints():
    gen = BeliefBiasGenerator(seed=130, prompt_style="default")
    problem_spec = gen._build_conjunction_fallacy_problem_spec(
        profile_description=(
            "A founder is in constant customer contact, runs tight release cycles, "
            "and coordinates cross-functional execution reliably."
        ),
        semantic_domain="startup",
        event_a_label="startup grows revenue",
        event_b_detail_label="startup signs an enterprise partnership",
        p_event_a=0.64,
        p_event_a_and_b=0.18,
        conjunction_in_option_a=False,
    )
    variants = {}
    for frame in ("vivid_description", "plain_probability", "representative_profile"):
        variants[frame] = gen._render_conjunction_fallacy_prompt(
            problem_spec=problem_spec,
            style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant=frame,
        )

    fingerprints = {
        frame: _conjunction_structure_fingerprint(text)
        for frame, text in variants.items()
    }
    assert fingerprints["vivid_description"] == "scene_based"
    assert fingerprints["plain_probability"] == "plain_literal"
    assert fingerprints["representative_profile"] == "type_profile"

    normalized = {k: _normalize_prompt(v) for k, v in variants.items()}
    for left, right in (
        ("vivid_description", "plain_probability"),
        ("vivid_description", "representative_profile"),
        ("plain_probability", "representative_profile"),
    ):
        sim = SequenceMatcher(None, normalized[left], normalized[right]).ratio()
        assert sim < 0.9, (
            f"Conjunction frame variants are too structurally similar: "
            f"{left} vs {right} similarity={sim:.4f}"
        )


def test_conjunction_semantic_pools_include_required_archetypes():
    gen = BeliefBiasGenerator(seed=101)
    pools = set(gen._conjunction_semantic_pools().keys())
    required = {
        "startup",
        "nonprofit",
        "scientist",
        "campaign",
        "sales",
        "product_manager",
        "community_organizer",
        "teacher",
    }
    assert required.issubset(pools)


def test_conjunction_generation_spans_multiple_semantic_pools():
    gen = BeliefBiasGenerator(
        seed=102,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
    )
    pool_counts: dict[str, int] = {}
    n = 120
    for i in range(n):
        dp = gen._generate_conjunction_fallacy(i)
        pool = dp.problem_spec["assumptions"]["semantic_domain"]
        pool_counts[pool] = pool_counts.get(pool, 0) + 1

    assert len(pool_counts) >= 6
    max_share = max(pool_counts.values()) / n
    assert max_share <= 0.35
