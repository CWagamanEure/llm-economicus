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
    assert "independence" in lower_prompt or "50/50" in lower_prompt


def test_gambler_non_normative_prompts_avoid_explicit_independence_cues():
    forbidden = (
        "same process",
        "same chance each trial",
        "setup stays the same",
        "independence",
        "trial independence",
        "50/50 odds",
        "unchanged 50/50",
    )
    for regime in ("neutral_realistic", "bias_eliciting"):
        gen = BeliefBiasGenerator(seed=92, prompt_style_regime=regime)
        dp = gen._generate_gambler_fallacy(0)
        lower_prompt = dp.input.lower()
        for phrase in forbidden:
            assert phrase not in lower_prompt


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
