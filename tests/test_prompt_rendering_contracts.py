import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from base_generator import (  # noqa: E402
    PROMPT_FORBIDDEN_PHRASES_BY_REGIME,
    STACKED_WRAPPER_PREFIX_LABELS,
)
from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402
from belief_bias_generator import BeliefBiasGenerator  # noqa: E402
from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402

REGIMES = ("normative_explicit", "neutral_realistic", "bias_eliciting")


@dataclass(frozen=True)
class PromptFixture:
    generator_cls: type
    method_name: str
    frame_variants: tuple[str, ...]
    sample_index: int = 0


RISK_FIXTURES = (
    PromptFixture(
        RiskLossTimeGenerator,
        "_generate_lottery_choice",
        ("gain_focus", "loss_focus", "investing_context"),
    ),
    PromptFixture(
        RiskLossTimeGenerator,
        "_generate_mixed_gain_loss_choice",
        ("safety_focus", "upside_focus", "everyday_money_context"),
    ),
)

BAYES_FIXTURES = (
    PromptFixture(
        BayesianSignalGenerator,
        "_generate_basic_bayes_update",
        ("medical_screening", "fraud_detection", "security_alert"),
    ),
    PromptFixture(
        BayesianSignalGenerator,
        "_generate_binary_signal_decision",
        ("hiring_screen", "trading_signal", "manufacturing_defect"),
    ),
)

BELIEF_FIXTURES = (
    PromptFixture(
        BeliefBiasGenerator,
        "_generate_gambler_fallacy",
        ("neutral_coin", "roulette_streak", "market_streak"),
    ),
    PromptFixture(
        BeliefBiasGenerator,
        "_generate_conjunction_fallacy",
        ("plain_probability", "representative_profile", "vivid_description"),
    ),
    PromptFixture(
        BeliefBiasGenerator,
        "_generate_sample_size_neglect",
        ("hospital_births", "fund_returns", "quality_control_batches"),
    ),
    PromptFixture(
        BeliefBiasGenerator,
        "_generate_overprecision_calibration",
        ("analyst_forecast", "weather_forecast", "startup_projection"),
    ),
)

ALL_FIXTURES = RISK_FIXTURES + BAYES_FIXTURES + BELIEF_FIXTURES


def _generate_datapoint(
    fixture: PromptFixture,
    *,
    seed: int,
    regime: str,
    frame_variant: str,
):
    generator = fixture.generator_cls(
        seed=seed,
        prompt_style="default",
        prompt_style_regime=regime,
        prompt_frame_variant=frame_variant,
    )
    return getattr(generator, fixture.method_name)(fixture.sample_index)


def _assert_target_contract_equal(left, right) -> None:
    assert left.target.objective == right.target.objective
    assert left.target.outcome_model == right.target.outcome_model
    assert dict(left.target.action_values) == dict(right.target.action_values)
    assert dict(left.target.decision_values) == dict(right.target.decision_values)
    assert left.target.optimal_decision == right.target.optimal_decision
    assert dict(left.target.solver_trace) == dict(right.target.solver_trace)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _is_cosmetic_prefix_variant(a: str, b: str) -> bool:
    a_norm = " ".join(a.split())
    b_norm = " ".join(b.split())
    if a_norm == b_norm:
        return True
    for first, second in ((a_norm, b_norm), (b_norm, a_norm)):
        if first.endswith(second):
            prefix = first[: len(first) - len(second)].strip()
            if prefix.endswith(":") or prefix in {
                "scenario",
                "decision context",
                "financial context",
            }:
                return True
    return False


def _normalize_prompt(text: str) -> str:
    compact = " ".join(text.lower().split())
    compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
    compact = re.sub(r"'[^']*'", "<quoted>", compact)
    compact = re.sub(r"\b[ht]{4,}\b", "<seq>", compact)
    return compact


def _prompt_skeleton(text: str) -> str:
    skeleton = _normalize_prompt(text)
    skeleton = re.sub(
        (
            r"\b("
            r"medical|patient|condition|test|fraud|transaction|candidate|hiring|"
            r"security|threat|market|trading|hospital|births|girls|fund|stocks|"
            r"factory|batch|analyst|weather|startup|kpi|roulette|basketball|"
            r"coin|wheel|tape|scoreboard"
            r")\b"
        ),
        "<ctx>",
        skeleton,
    )
    return " ".join(skeleton.split())


def _style_distance_failures(generator_cls: type, normative: str, neutral: str):
    gen = generator_cls(seed=0, prompt_style="default")
    return gen.style_distance_lint(
        normative_prompt=normative,
        neutral_prompt=neutral,
    )


def test_stacked_wrapper_cleanup_and_lint():
    gen = BeliefBiasGenerator(seed=1, prompt_style="default")
    prompt = (
        "The analyst desk is circulating fresh numbers. "
        "Decision brief: Case review: Interval A is [10, 20]. "
        "Which interval is more likely to contain the realized value?"
    )
    assert gen._count_stacked_framing_prefixes(prompt) >= 2
    cleaned = gen._collapse_stacked_prompt_wrappers(prompt=prompt)
    assert gen._count_stacked_framing_prefixes(cleaned) == 0
    failures = gen._prompt_qa_generic_failures(prompt=cleaned)
    assert not any(f["code"] == "stacked_framing_prefixes" for f in failures)


@pytest.mark.parametrize(
    ("generator_cls", "method_name", "frame_variants"),
    (
        (
            BayesianSignalGenerator,
            "_generate_binary_signal_decision",
            ("medical_screening", "fraud_detection", "hiring_screen"),
        ),
        (
            BeliefBiasGenerator,
            "_generate_conjunction_fallacy",
            ("vivid_description", "plain_probability", "representative_profile"),
        ),
        (
            BeliefBiasGenerator,
            "_generate_overprecision_calibration",
            ("analyst_forecast", "weather_forecast", "startup_projection"),
        ),
    ),
)
def test_targeted_subtypes_do_not_emit_stacked_wrappers(
    generator_cls: type,
    method_name: str,
    frame_variants: tuple[str, ...],
):
    for regime in REGIMES:
        for frame_variant in frame_variants:
            gen = generator_cls(
                seed=1234,
                prompt_style="default",
                prompt_style_regime=regime,
                prompt_frame_variant=frame_variant,
            )
            dp = getattr(gen, method_name)(0)
            labels = gen._extract_framing_prefix_labels(dp.input)
            assert len(labels) <= 1, (
                f"Expected <=1 wrapper label, found {labels} in prompt: {dp.input}"
            )
            for label in STACKED_WRAPPER_PREFIX_LABELS:
                assert f"{label}:" not in dp.input.lower(), (
                    f"Wrapper label '{label}:' should have been collapsed: {dp.input}"
                )


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_target_invariance_across_prompt_style_regimes(fixture: PromptFixture):
    baseline = _generate_datapoint(
        fixture,
        seed=123,
        regime="neutral_realistic",
        frame_variant=fixture.frame_variants[0],
    )
    for regime in REGIMES:
        candidate = _generate_datapoint(
            fixture,
            seed=123,
            regime=regime,
            frame_variant=fixture.frame_variants[0],
        )
        _assert_target_contract_equal(candidate, baseline)


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_target_invariance_across_frame_variants(fixture: PromptFixture):
    baseline = _generate_datapoint(
        fixture,
        seed=456,
        regime="neutral_realistic",
        frame_variant=fixture.frame_variants[0],
    )
    for frame_variant in fixture.frame_variants[1:]:
        candidate = _generate_datapoint(
            fixture,
            seed=456,
            regime="neutral_realistic",
            frame_variant=frame_variant,
        )
        _assert_target_contract_equal(candidate, baseline)


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
@pytest.mark.parametrize("regime", ("neutral_realistic", "bias_eliciting"))
def test_non_normative_regimes_do_not_leak_normative_cues(fixture: PromptFixture, regime: str):
    forbidden = PROMPT_FORBIDDEN_PHRASES_BY_REGIME[regime]
    dp = _generate_datapoint(
        fixture,
        seed=789,
        regime=regime,
        frame_variant=fixture.frame_variants[0],
    )
    lower_prompt = dp.input.lower()
    leaks = [phrase for phrase in forbidden if phrase in lower_prompt]
    assert not leaks, (
        f"Found forbidden normative cues for {fixture.method_name} in regime {regime}: {leaks}"
    )


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_frame_variants_have_substantive_diversity(fixture: PromptFixture):
    prompts = []
    for frame_variant in fixture.frame_variants:
        dp = _generate_datapoint(
            fixture,
            seed=901,
            regime="neutral_realistic",
            frame_variant=frame_variant,
        )
        prompts.append((frame_variant, dp.input))

    max_pairwise_distance = 0.0
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            variant_i, prompt_i = prompts[i]
            variant_j, prompt_j = prompts[j]
            assert not _is_cosmetic_prefix_variant(prompt_i, prompt_j), (
                f"Cosmetic-only frame change detected for {fixture.method_name}: "
                f"{variant_i} vs {variant_j}"
            )
            tokens_i = _tokenize(prompt_i)
            tokens_j = _tokenize(prompt_j)
            jaccard = len(tokens_i & tokens_j) / max(1, len(tokens_i | tokens_j))
            similarity = SequenceMatcher(None, prompt_i, prompt_j).ratio()
            max_pairwise_distance = max(max_pairwise_distance, 1 - similarity)
            assert jaccard < 0.99, (
                f"Lexical overlap too high for {fixture.method_name}: "
                f"{variant_i} vs {variant_j} (jaccard={jaccard:.4f})"
            )
            assert similarity < 0.995, (
                f"Structural similarity too high for {fixture.method_name}: "
                f"{variant_i} vs {variant_j} (similarity={similarity:.4f})"
            )

    assert max_pairwise_distance >= 0.03, (
        f"Frame variants may be too cosmetic for {fixture.method_name}; "
        f"max pairwise distance={max_pairwise_distance:.4f}"
    )


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_metadata_consistency_for_prompt_fields(fixture: PromptFixture):
    dp = _generate_datapoint(
        fixture,
        seed=1001,
        regime="neutral_realistic",
        frame_variant=fixture.frame_variants[0],
    )
    assert dp.metadata.prompt_style_regime in REGIMES
    assert dp.metadata.prompt_frame_variant == fixture.frame_variants[0]
    if fixture.generator_cls is BayesianSignalGenerator:
        assert dp.metadata.semantic_context == fixture.frame_variants[0]
    else:
        assert dp.metadata.semantic_context in (None, fixture.frame_variants[0])


@pytest.mark.parametrize(
    "fixture",
    (
        PromptFixture(
            BayesianSignalGenerator,
            "_generate_binary_signal_decision",
            ("medical_screening", "fraud_detection", "hiring_screen"),
        ),
        PromptFixture(
            BeliefBiasGenerator,
            "_generate_sample_size_neglect",
            ("hospital_births", "fund_returns", "quality_control_batches"),
        ),
        PromptFixture(
            BeliefBiasGenerator,
            "_generate_overprecision_calibration",
            ("analyst_forecast", "weather_forecast", "startup_projection"),
        ),
    ),
)
def test_normative_explicit_frame_variants_are_not_text_identical_after_normalization(
    fixture: PromptFixture,
):
    prompts = [
        _generate_datapoint(
            fixture,
            seed=1201,
            regime="normative_explicit",
            frame_variant=frame_variant,
        ).input
        for frame_variant in fixture.frame_variants
    ]
    normalized = [_normalize_prompt(p) for p in prompts]
    assert len(set(normalized)) == len(normalized), (
        f"Normative-explicit frame variants collapsed after normalization for "
        f"{fixture.method_name}."
    )


def test_conjunction_frame_variants_have_low_enough_lexical_overlap():
    fixture = PromptFixture(
        BeliefBiasGenerator,
        "_generate_conjunction_fallacy",
        ("plain_probability", "representative_profile", "vivid_description"),
    )
    prompts = [
        _generate_datapoint(
            fixture,
            seed=1301,
            regime="bias_eliciting",
            frame_variant=frame_variant,
        ).input
        for frame_variant in fixture.frame_variants
    ]
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a_tokens = _tokenize(prompts[i])
            b_tokens = _tokenize(prompts[j])
            jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
            assert jaccard < 0.88, (
                "Conjunction frame variants are too lexically similar "
                f"(pair={i}-{j}, jaccard={jaccard:.3f})."
            )


@pytest.mark.parametrize(
    "fixture",
    (
        PromptFixture(
            BayesianSignalGenerator,
            "_generate_binary_signal_decision",
            ("medical_screening", "fraud_detection", "hiring_screen"),
        ),
        PromptFixture(
            BeliefBiasGenerator,
            "_generate_sample_size_neglect",
            ("hospital_births", "fund_returns", "quality_control_batches"),
        ),
        PromptFixture(
            BeliefBiasGenerator,
            "_generate_overprecision_calibration",
            ("analyst_forecast", "weather_forecast", "startup_projection"),
        ),
        PromptFixture(
            BeliefBiasGenerator,
            "_generate_conjunction_fallacy",
            ("plain_probability", "representative_profile", "vivid_description"),
        ),
    ),
)
def test_frame_variants_do_not_share_the_same_sentence_skeleton(fixture: PromptFixture):
    prompts = [
        _generate_datapoint(
            fixture,
            seed=1401,
            regime="normative_explicit",
            frame_variant=frame_variant,
        ).input
        for frame_variant in fixture.frame_variants
    ]
    skeletons = [_prompt_skeleton(p) for p in prompts]
    assert len(set(skeletons)) > 1, (
        f"All frame variants share the same skeleton for {fixture.method_name}."
    )
    for i in range(len(skeletons)):
        for j in range(i + 1, len(skeletons)):
            similarity = SequenceMatcher(None, skeletons[i], skeletons[j]).ratio()
            assert similarity < 0.99, (
                f"Frame variants look like noun-swaps for {fixture.method_name} "
                f"(pair={i}-{j}, skeleton_similarity={similarity:.4f})."
            )


@pytest.mark.parametrize(
    "fixture,frame_variant",
    (
        (
            PromptFixture(
                RiskLossTimeGenerator,
                "_generate_lottery_choice",
                ("gain_focus", "loss_focus", "safety_focus"),
            ),
            "gain_focus",
        ),
        (
            PromptFixture(
                BayesianSignalGenerator,
                "_generate_basic_bayes_update",
                ("medical_screening", "fraud_detection", "security_alert"),
            ),
            "medical_screening",
        ),
        (
            PromptFixture(
                BayesianSignalGenerator,
                "_generate_binary_signal_decision",
                ("hiring_screen", "trading_signal", "manufacturing_defect"),
            ),
            "hiring_screen",
        ),
        (
            PromptFixture(
                BeliefBiasGenerator,
                "_generate_overprecision_calibration",
                ("analyst_forecast", "weather_forecast", "startup_projection"),
            ),
            "analyst_forecast",
        ),
    ),
)
def test_normative_vs_neutral_style_distance_not_header_only(
    fixture: PromptFixture, frame_variant: str
):
    normative = _generate_datapoint(
        fixture,
        seed=1601,
        regime="normative_explicit",
        frame_variant=frame_variant,
    ).input
    neutral = _generate_datapoint(
        fixture,
        seed=1601,
        regime="neutral_realistic",
        frame_variant=frame_variant,
    ).input
    failures = _style_distance_failures(fixture.generator_cls, normative, neutral)
    assert not failures, (
        f"Normative vs neutral style distance is too small for {fixture.method_name}: {failures}"
    )
