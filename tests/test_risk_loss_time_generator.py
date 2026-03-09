import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from base_generator import PROMPT_FORBIDDEN_PHRASES_BY_REGIME  # noqa: E402
from difficulty_config import (  # noqa: E402
    RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE,
    RISK_LOSS_TIME_SUBTYPES,
)
from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402

from schema import ActionScalars, Target  # noqa: E402


def test_generate_dispatches_all_subtypes_with_expected_task_id_prefixes():
    expected_prefix = {
        "lottery_choice": "lottery_",
        "ce_offer_comparison": "ce_",
        "mixed_gain_loss_choice": "mixed_gain_loss_",
        "time_discounting": "time_",
        "ambiguity_aversion_choice": "ambiguity_",
        "probability_weighting_counterexample": "prob_weight_",
        "loss_aversion_counterexample": "loss_averse_",
        "hyperbolic_discounting_counterexample": "hyperbolic_",
    }

    for subtype, prefix in expected_prefix.items():
        gen = RiskLossTimeGenerator(seed=1)

        def fake_choice(options):
            if "lottery_choice" in options:
                return subtype
            return options[0]

        gen.rng.choice = fake_choice
        dp = gen.generate()
        assert dp.task_subtype == subtype
        assert dp.task_id.startswith(prefix)


def test_subtype_names_and_difficulty_config_keys_are_synchronized():
    config_keys = set(RISK_LOSS_TIME_DEFAULT_DIFFICULTY_BY_SUBTYPE.keys())
    subtype_keys = set(RISK_LOSS_TIME_SUBTYPES)
    assert config_keys == subtype_keys
    assert "certainty_equivalent" not in config_keys


def test_metadata_uses_generator_seed_and_monotonic_sample_index():
    gen = RiskLossTimeGenerator(seed=11)
    first = gen._generate_lottery_choice(0)
    second = gen._generate_lottery_choice(1)

    assert first.metadata.generator_name == "RiskLossTimeGenerator"
    assert first.metadata.dataset_role == "normative_training"
    assert second.metadata.dataset_role == "normative_training"
    assert first.metadata.prompt_has_action_labels is True
    assert second.metadata.prompt_has_action_labels is True
    assert isinstance(first.metadata.example_fingerprint, str)
    assert isinstance(second.metadata.example_fingerprint, str)
    assert len(first.metadata.example_fingerprint) == 64
    assert len(second.metadata.example_fingerprint) == 64
    assert first.metadata.example_fingerprint != second.metadata.example_fingerprint
    assert first.metadata.tie_threshold == gen.CHOICE_TIE_EPSILON
    assert second.metadata.tie_threshold == gen.CHOICE_TIE_EPSILON
    assert first.metadata.seed == 11
    assert second.metadata.seed == 11
    assert first.metadata.sample_index == 0
    assert second.metadata.sample_index == 1
    assert first.metadata.difficulty_metrics is not None
    assert "ev_gap" in first.metadata.difficulty_metrics
    assert "numeric_complexity" in first.metadata.difficulty_metrics


def test_task_ids_are_deterministic_and_derived_from_version_and_sample_index():
    gen = RiskLossTimeGenerator(seed=11, version="v1")
    first = gen._generate_lottery_choice(0)
    second = gen._generate_lottery_choice(1)

    assert first.task_id == "lottery_v1_000000"
    assert second.task_id == "lottery_v1_000001"
    assert first.metadata.sample_index == 0
    assert second.metadata.sample_index == 1


def test_task_id_suffix_matches_metadata_sample_index():
    gen = RiskLossTimeGenerator(seed=7, version="v2")
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]

    for dp in datapoints:
        suffix = int(dp.task_id.split("_")[-1])
        assert suffix == dp.metadata.sample_index


def test_all_subtypes_use_normalized_options_state():
    gen = RiskLossTimeGenerator(seed=15)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    for dp in datapoints:
        assert "options" in dp.target.state
        assert set(dp.target.state["options"].keys()) == {"A", "B"}


def test_datapoints_include_problem_spec_with_matching_subtype():
    gen = RiskLossTimeGenerator(seed=21)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    for dp in datapoints:
        assert dp.problem_spec["task_subtype"] == dp.task_subtype
        assert set(dp.problem_spec["options"].keys()) == {"A", "B"}


def test_targets_include_explicit_comparison_pair_for_evaluation_tooling():
    gen = RiskLossTimeGenerator(seed=17)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    expected_pairs = {
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

    for dp in datapoints:
        assert dp.target.comparison_pair == expected_pairs[dp.task_subtype]


def test_comparison_pair_helper_returns_subtype_specific_mapping():
    gen = RiskLossTimeGenerator(seed=17)
    assert gen._comparison_pair_for_subtype("lottery_choice") == {
        "left_action": "choose_lottery",
        "right_action": "choose_sure",
    }
    assert gen._comparison_pair_for_subtype("ce_offer_comparison") == {
        "left_action": "accept_offer",
        "right_action": "reject_offer",
    }
    assert gen._comparison_pair_for_subtype("mixed_gain_loss_choice") == {
        "left_action": "choose_risky",
        "right_action": "choose_sure",
    }
    assert gen._comparison_pair_for_subtype("time_discounting") == {
        "left_action": "choose_later",
        "right_action": "choose_now",
    }
    assert gen._comparison_pair_for_subtype("ambiguity_aversion_choice") == {
        "left_action": "choose_ambiguous",
        "right_action": "choose_known_risk",
    }
    assert gen._comparison_pair_for_subtype("probability_weighting_counterexample") == {
        "left_action": "choose_longshot",
        "right_action": "choose_sure",
    }
    assert gen._comparison_pair_for_subtype("loss_aversion_counterexample") == {
        "left_action": "accept_gamble",
        "right_action": "choose_status_quo",
    }
    assert gen._comparison_pair_for_subtype("hyperbolic_discounting_counterexample") == {
        "left_action": "choose_later",
        "right_action": "choose_earlier",
    }


def test_prompt_renderer_helper_returns_subtype_specific_renderer():
    gen = RiskLossTimeGenerator(seed=17)
    assert (
        gen._prompt_renderer_for_subtype("lottery_choice").__name__
        == "_render_lottery_choice_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype("ce_offer_comparison").__name__
        == "_render_ce_offer_comparison_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype("mixed_gain_loss_choice").__name__
        == "_render_mixed_gain_loss_choice_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype("time_discounting").__name__
        == "_render_time_discounting_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype("ambiguity_aversion_choice").__name__
        == "_render_ambiguity_aversion_choice_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype(
            "probability_weighting_counterexample"
        ).__name__
        == "_render_probability_weighting_counterexample_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype("loss_aversion_counterexample").__name__
        == "_render_loss_aversion_counterexample_prompt"
    )
    assert (
        gen._prompt_renderer_for_subtype(
            "hyperbolic_discounting_counterexample"
        ).__name__
        == "_render_hyperbolic_discounting_counterexample_prompt"
    )


def test_target_fields_follow_comparison_pair_action_ordering():
    gen = RiskLossTimeGenerator(seed=117)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]

    for dp in datapoints:
        left_action = dp.target.comparison_pair["left_action"]
        right_action = dp.target.comparison_pair["right_action"]
        assert dp.target.actions[0] == left_action
        assert dp.target.actions[1] == right_action
        assert list(dp.target.action_values.keys())[:2] == [left_action, right_action]
        assert list(dp.target.decision_values.keys())[:2] == [left_action, right_action]


def test_solver_trace_records_normalized_comparison_values():
    gen = RiskLossTimeGenerator(seed=117)
    dp = gen._generate_lottery_choice(0)
    trace = dp.target.solver_trace
    left_action = dp.target.comparison_pair["left_action"]
    right_action = dp.target.comparison_pair["right_action"]

    assert trace["left_action"] == left_action
    assert trace["right_action"] == right_action
    assert trace["left_value"] == dp.target.decision_values[left_action]
    assert trace["right_value"] == dp.target.decision_values[right_action]
    assert trace["tie_epsilon"] == dp.problem_spec["assumptions"]["tie_epsilon"]
    assert trace["comparison_result"] == dp.target.optimal_decision


def test_target_uses_typed_action_scalar_maps():
    gen = RiskLossTimeGenerator(seed=6)
    dp = gen._generate_lottery_choice(0)

    assert isinstance(dp.target.action_values, ActionScalars)
    assert isinstance(dp.target.decision_values, ActionScalars)


def test_target_validation_rejects_non_numeric_decision_value():
    with pytest.raises(TypeError):
        Target(
            objective="maximize value",
            state={},
            beliefs={},
            constraints={},
            actions=["a", "b", "indifferent"],
            comparison_pair={"left_action": "a", "right_action": "b"},
            outcome_model={},
            action_values={"a": 1.0, "b": 2.0},
            decision_values={"a": "bad", "b": 2.0},
            optimal_decision="a",
            solver_trace={
                "left_action": "a",
                "right_action": "b",
                "left_value": 1.0,
                "right_value": 2.0,
                "tie_epsilon": 1e-6,
                "comparison_result": "b",
            },
            brief_rationale="test",
        )


def test_difficulty_metrics_include_prompt_complexity_features():
    gen = RiskLossTimeGenerator(seed=19)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]

    for dp in datapoints:
        metrics = dp.metadata.difficulty_metrics
        assert metrics is not None
        assert "prompt_clause_count" in metrics
        assert "prompt_style_variant" in metrics
        assert "prompt_has_decimal" in metrics
        assert "prompt_mixed_signed_outcomes" in metrics
        assert "prompt_asymmetric_choice_framing" in metrics
        assert "prompt_contains_action_tokens" in metrics
        assert "prompt_action_token_count" in metrics
        assert "prompt_complexity" in metrics
        assert metrics["prompt_clause_count"] >= 1
        assert metrics["prompt_action_token_count"] >= 0
        assert metrics["prompt_complexity"] >= metrics["prompt_clause_count"]
        assert metrics["prompt_contains_action_tokens"] == (
            metrics["prompt_action_token_count"] > 0
        )

    ce_dp = next(dp for dp in datapoints if dp.task_subtype == "ce_offer_comparison")
    mixed_dp = next(
        dp for dp in datapoints if dp.task_subtype == "mixed_gain_loss_choice"
    )
    assert ce_dp.metadata.difficulty_metrics["prompt_asymmetric_choice_framing"] is True
    assert mixed_dp.metadata.difficulty_metrics["prompt_mixed_signed_outcomes"] is True


def test_behavioral_subtypes_include_targeted_difficulty_metadata():
    gen = RiskLossTimeGenerator(seed=4242)
    ambiguity_dp = gen._generate_ambiguity_aversion_choice(0)
    probability_dp = gen._generate_probability_weighting_counterexample(1)
    loss_dp = gen._generate_loss_aversion_counterexample(2)
    time_dp = gen._generate_time_discounting(3)
    hyperbolic_dp = gen._generate_hyperbolic_discounting_counterexample(4)

    ambiguity_metrics = ambiguity_dp.metadata.difficulty_metrics
    assert "ambiguous_minus_known_ev_signed" in ambiguity_metrics
    assert "ambiguity_regime_intended" in ambiguity_metrics
    assert "ambiguity_regime_realized" in ambiguity_metrics
    assert "known_probability" in ambiguity_metrics
    assert "subjective_ambiguous_win_probability" in ambiguity_metrics
    assert ambiguity_metrics["ambiguous_minus_known_ev_signed"] == round(
        ambiguity_dp.target.action_values["choose_ambiguous"]
        - ambiguity_dp.target.action_values["choose_known_risk"],
        4,
    )
    ambiguity_gap = (
        ambiguity_dp.target.action_values["choose_ambiguous"]
        - ambiguity_dp.target.action_values["choose_known_risk"]
    )
    if ambiguity_gap >= 6.0:
        expected_ambiguity_regime = "choose_ambiguous"
    elif ambiguity_gap <= -6.0:
        expected_ambiguity_regime = "choose_known_risk"
    elif abs(ambiguity_gap) <= 3.0:
        expected_ambiguity_regime = "near_indifferent"
    else:
        expected_ambiguity_regime = "ambiguous_band"
    assert ambiguity_metrics["ambiguity_regime_realized"] == expected_ambiguity_regime

    probability_metrics = probability_dp.metadata.difficulty_metrics
    assert probability_metrics["is_longshot_setup"] is True
    assert "longshot_probability" in probability_metrics
    assert "longshot_ev_superior" in probability_metrics
    assert "longshot_near_tie" in probability_metrics
    assert "longshot_minus_sure_ev_signed" in probability_metrics
    assert probability_metrics["longshot_ev_superior"] == (
        probability_dp.target.action_values["choose_longshot"]
        > probability_dp.target.action_values["choose_sure"]
    )
    probability_gap = (
        probability_dp.target.action_values["choose_longshot"]
        - probability_dp.target.action_values["choose_sure"]
    )
    assert probability_metrics["longshot_near_tie"] == (
        abs(probability_gap) <= gen.CHOICE_TIE_EPSILON
    )
    if probability_metrics["longshot_near_tie"]:
        assert probability_dp.target.optimal_decision == "indifferent"

    loss_metrics = loss_dp.metadata.difficulty_metrics
    assert "mixed_gamble_ev_superior_to_status_quo" in loss_metrics
    assert "mixed_gamble_positive_ev" in loss_metrics
    assert "mixed_gamble_minus_status_quo_ev_signed" in loss_metrics
    assert "mixed_gamble_regime_intended" in loss_metrics
    assert "mixed_gamble_regime_realized" in loss_metrics
    assert loss_metrics["mixed_gamble_ev_superior_to_status_quo"] == (
        loss_dp.target.action_values["accept_gamble"]
        > loss_dp.target.action_values["choose_status_quo"]
    )
    assert loss_metrics["mixed_gamble_positive_ev"] == (
        loss_dp.target.action_values["accept_gamble"] > 0
    )

    time_metrics = time_dp.metadata.difficulty_metrics
    assert time_metrics["earlier_delay_days"] == 0
    assert time_metrics["later_delay_days"] == time_metrics["time_horizon_days"]
    assert time_metrics["delay_gap_days"] == time_metrics["later_delay_days"]
    assert "later_minus_now_signed" in time_metrics

    hyperbolic_metrics = hyperbolic_dp.metadata.difficulty_metrics
    assert hyperbolic_metrics["later_delay_days"] > hyperbolic_metrics["earlier_delay_days"]
    assert hyperbolic_metrics["delay_gap_days"] == (
        hyperbolic_metrics["later_delay_days"] - hyperbolic_metrics["earlier_delay_days"]
    )
    assert "later_break_even_amount" in hyperbolic_metrics
    assert "later_minus_earlier_pv_signed" in hyperbolic_metrics
    assert "timing_regime_intended" in hyperbolic_metrics
    assert "timing_regime_realized" in hyperbolic_metrics


def test_prompt_action_token_metrics_distinguish_labeled_vs_unlabeled_styles():
    labeled_dp = RiskLossTimeGenerator(seed=801, prompt_style="default")._generate_lottery_choice(
        0
    )
    unlabeled_dp = RiskLossTimeGenerator(
        seed=801, prompt_style="unlabeled"
    )._generate_lottery_choice(0)

    labeled_metrics = labeled_dp.metadata.difficulty_metrics
    unlabeled_metrics = unlabeled_dp.metadata.difficulty_metrics
    assert labeled_metrics is not None
    assert unlabeled_metrics is not None
    assert labeled_metrics["prompt_contains_action_tokens"] is True
    assert int(labeled_metrics["prompt_action_token_count"]) > 0
    assert unlabeled_metrics["prompt_contains_action_tokens"] is False
    assert int(unlabeled_metrics["prompt_action_token_count"]) == 0


def test_prompt_action_token_metrics_support_future_action_verbs():
    gen = RiskLossTimeGenerator(seed=0)
    features = gen._compute_prompt_complexity_features(
        prompt=(
            "Option A (defer_payment) pays $100 in 30 days. "
            "Option B (take_cash_now) pays $95 now."
        ),
        prompt_style="default",
        numeric_values=[100, 95, 30],
        comparison_pair={
            "left_action": "defer_payment",
            "right_action": "take_cash_now",
        },
    )
    assert features["prompt_contains_action_tokens"] is True
    assert int(features["prompt_action_token_count"]) >= 2


def test_problem_spec_builder_helpers_construct_expected_shape():
    gen = RiskLossTimeGenerator(seed=0)

    lottery_spec = gen._build_lottery_problem_spec(
        sure_amount=50,
        p_win=0.4,
        win_amount=120,
        lose_amount=0,
    )
    ce_spec = gen._build_ce_offer_comparison_problem_spec(
        offered=88.5,
        p_win=0.6,
        high=150,
        low=20,
    )
    mixed_spec = gen._build_mixed_gain_loss_problem_spec(
        sure=-10,
        p_gain=0.55,
        gain=100,
        loss=-60,
    )
    time_spec = gen._build_time_discounting_problem_spec(
        now=40,
        later_offer=44.5,
        days=30,
        annual_rate=0.08,
    )
    ambiguity_spec = gen._build_ambiguity_aversion_problem_spec(
        known_probability=0.5,
        known_win_amount=100,
        known_lose_amount=10,
        ambiguous_win_amount=140,
        ambiguous_lose_amount=0,
        subjective_ambiguous_win_probability=0.35,
    )
    probability_weighting_spec = gen._build_probability_weighting_counterexample_problem_spec(
        sure_amount=70,
        p_win=0.1,
        win_amount=500,
        lose_amount=0,
    )
    loss_aversion_spec = gen._build_loss_aversion_counterexample_problem_spec(
        status_quo_amount=0,
        p_gain=0.6,
        gain=120,
        loss=-80,
    )
    hyperbolic_spec = gen._build_hyperbolic_discounting_counterexample_problem_spec(
        earlier_amount=100,
        earlier_delay_days=7,
        later_amount=108.0,
        later_delay_days=30,
        annual_rate=0.08,
    )

    specs = [
        lottery_spec,
        ce_spec,
        mixed_spec,
        time_spec,
        ambiguity_spec,
        probability_weighting_spec,
        loss_aversion_spec,
        hyperbolic_spec,
    ]
    for spec in specs:
        assert "objective" in spec
        assert "assumptions" in spec
        assert set(spec["options"].keys()) == {"A", "B"}
        assert spec["assumptions"]["tie_epsilon"] == gen.CHOICE_TIE_EPSILON


def test_ambiguity_problem_spec_objective_explicitly_scopes_interpretation():
    gen = RiskLossTimeGenerator(seed=0)
    spec = gen._build_ambiguity_aversion_problem_spec(
        known_probability=0.5,
        known_win_amount=100,
        known_lose_amount=10,
        ambiguous_win_amount=140,
        ambiguous_lose_amount=0,
        subjective_ambiguous_win_probability=0.35,
    )
    objective = spec["objective"]
    assert "subjective beliefs" in objective
    assert "not unresolved ambiguity" in objective


def test_ambiguity_problem_spec_intentionally_mirrors_known_probability():
    gen = RiskLossTimeGenerator(seed=0)
    spec = gen._build_ambiguity_aversion_problem_spec(
        known_probability=0.42,
        known_win_amount=100,
        known_lose_amount=10,
        ambiguous_win_amount=140,
        ambiguous_lose_amount=0,
        subjective_ambiguous_win_probability=0.35,
    )
    assert spec["options"]["A"]["p_win"] == spec["assumptions"]["known_probability"]


def test_numeric_complexity_operator_count_ignores_unary_minus():
    gen = RiskLossTimeGenerator(seed=0)
    operations = gen._count_operations_in_outcome_model(
        {
            "choose_sure": "10",
            "choose_risky": "0.4 * 100 + (1 - 0.4) * -50",
        }
    )
    assert operations == 4


def test_operator_count_distinguishes_binary_and_unary_minus_tokens():
    gen = RiskLossTimeGenerator(seed=0)
    operations = gen._count_operations_in_outcome_model(
        {"choice": "0.5 * 100 - -20 + (30 - 10) / 2"}
    )
    assert operations == 5


def test_operator_count_ignores_percentage_sign_variants():
    gen = RiskLossTimeGenerator(seed=0)
    operations = gen._count_operations_in_outcome_model(
        {
            "a": "20% * 100",
            "b": "50 - 10% + 2",
            "c": "30 % * 4",
        }
    )
    assert operations == 4


def test_choose_optimal_action_returns_indifferent_on_tie():
    gen = RiskLossTimeGenerator(seed=0)
    choice = gen._choose_optimal_action(
        left_label="left",
        left_value=1.5,
        right_label="right",
        right_value=1.5,
    )
    assert choice == "indifferent"


def test_choose_optimal_action_returns_indifferent_within_epsilon():
    gen = RiskLossTimeGenerator(seed=0)
    choice = gen._choose_optimal_action(
        left_label="left",
        left_value=10.0000004,
        right_label="right",
        right_value=10.0,
    )
    assert choice == "indifferent"


def test_choose_optimal_action_uses_generator_tie_epsilon_by_default():
    gen = RiskLossTimeGenerator(seed=0)
    epsilon = gen.CHOICE_TIE_EPSILON

    near_tie = gen._choose_optimal_action(
        left_label="left",
        left_value=10.0 + epsilon / 2,
        right_label="right",
        right_value=10.0,
    )
    clear_gap = gen._choose_optimal_action(
        left_label="left",
        left_value=10.0 + (epsilon * 2),
        right_label="right",
        right_value=10.0,
    )

    assert near_tie == "indifferent"
    assert clear_gap == "left"


def test_lottery_tie_yields_indifferent_optimal_decision():
    gen = RiskLossTimeGenerator(seed=99)
    randint_values = iter([50, 100])  # sure_amount, win_amount
    uniform_values = iter([0.5])  # p_win
    gen.rng.randint = lambda _a, _b: next(randint_values)
    gen.rng.uniform = lambda _a, _b: next(uniform_values)

    dp = gen._generate_lottery_choice(0)
    assert dp.target.action_values["choose_sure"] == 50.0
    assert dp.target.action_values["choose_lottery"] == 50.0
    assert dp.metadata.difficulty_metrics is not None
    assert dp.metadata.difficulty_metrics["ev_gap"] == 0.0
    assert dp.metadata.difficulty_metrics["numeric_complexity"] == 9
    assert dp.target.decision_values == {
        "choose_lottery": 50.0,
        "choose_sure": 50.0,
    }
    assert dp.target.optimal_decision == "indifferent"
    assert "indifferent" in dp.target.actions


def test_time_discounting_has_explicit_choose_later_formula():
    gen = RiskLossTimeGenerator(seed=1234)
    dp = gen._generate_time_discounting(0)

    now_amount = dp.target.state["options"]["A"]["amount"]
    later_offer = dp.target.state["options"]["B"]["amount"]
    days = dp.target.state["options"]["B"]["delay_days"]
    assert dp.target.state["time_horizon_days"] == days
    assert dp.target.beliefs["discount_model"] == "simple"
    assert dp.target.beliefs["tie_epsilon"] == gen.CHOICE_TIE_EPSILON
    annual_rate = dp.target.beliefs["annual_discount_rate"]
    expected_now_formula = f"{now_amount}"
    expected_formula = f"{later_offer} / (1 + {annual_rate} * ({days} / 365))"

    assert dp.target.outcome_model["choose_now"] == expected_now_formula
    assert dp.target.outcome_model["choose_later"] == expected_formula
    assert dp.metadata.difficulty_metrics is not None
    assert dp.metadata.difficulty_metrics["time_horizon_days"] == days
    expected_prompt_core = (
        f"Choose now (choose_now): ${gen._format_number(now_amount)} today, "
        f"or choose later (choose_later): ${gen._format_number(later_offer)} "
        f"in {gen._format_number(days)} days. "
        f"Use annual simple discount rate r={gen._format_number(annual_rate)}."
    )
    assert dp.input.endswith(expected_prompt_core)


def test_expected_value_tasks_include_explicit_evaluation_model_in_beliefs():
    gen = RiskLossTimeGenerator(seed=42)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
    ]

    for dp in datapoints:
        beliefs = dp.target.beliefs
        assert beliefs["probabilities_are_known"] is True
        assert beliefs["utility_model"] == "linear"
        assert beliefs["decision_rule"] == "expected_value_maximization"
        assert beliefs["tie_epsilon"] == gen.CHOICE_TIE_EPSILON
        assert dp.target.decision_values == dp.target.action_values


def test_target_objective_and_beliefs_are_sourced_from_problem_spec():
    gen = RiskLossTimeGenerator(seed=88)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]

    for dp in datapoints:
        assert dp.target.objective == dp.problem_spec["objective"]
        assert dp.target.beliefs == dp.problem_spec["assumptions"]


def test_metadata_includes_resolved_prompt_style():
    gen = RiskLossTimeGenerator(seed=90, prompt_style="formal")
    dp = gen._generate_lottery_choice(0)
    assert dp.metadata.dataset_role == "normative_training"
    assert dp.metadata.requested_prompt_style == "formal"
    assert dp.metadata.resolved_prompt_style == "formal"
    assert dp.metadata.prompt_has_action_labels is True
    assert dp.metadata.tie_threshold == gen.CHOICE_TIE_EPSILON


def test_random_prompt_style_is_recorded_explicitly_in_metadata():
    dp = RiskLossTimeGenerator(seed=321, prompt_style="random")._generate_lottery_choice(
        0
    )
    assert dp.metadata.requested_prompt_style == "random"
    assert dp.metadata.resolved_prompt_style in {
        "default",
        "formal",
        "plain_english",
        "compact",
        "finance_framed",
        "unlabeled",
    }
    assert (
        dp.metadata.resolved_prompt_style
        == dp.metadata.difficulty_metrics["prompt_style_variant"]
    )
    assert dp.metadata.prompt_has_action_labels == (
        dp.metadata.resolved_prompt_style != "unlabeled"
    )
    assert isinstance(dp.metadata.example_fingerprint, str)
    assert dp.metadata.tie_threshold == dp.problem_spec["assumptions"]["tie_epsilon"]


def test_prompt_frame_variant_is_recorded_in_metadata():
    dp = RiskLossTimeGenerator(seed=322, prompt_style="default")._generate_lottery_choice(0)
    assert dp.metadata.prompt_frame_variant in {
        "gain_focus",
        "loss_focus",
        "safety_focus",
        "upside_focus",
        "single_shot",
        "repeated_play",
        "investing_context",
        "everyday_money_context",
    }
    assert (
        dp.metadata.prompt_frame_variant
        == dp.metadata.difficulty_metrics["prompt_frame_variant"]
    )


def test_prompt_leakage_validator_rejects_forbidden_phrases_for_non_normative_regimes():
    gen = RiskLossTimeGenerator(seed=1)
    with pytest.raises(ValueError, match="normative leakage phrase"):
        gen.assert_prompt_regime_no_leakage(
            prompt="Choose the option with higher expected value.",
            prompt_style_regime="neutral_realistic",
        )

    gen.assert_prompt_regime_no_leakage(
        prompt="Choose the option with higher expected value.",
        prompt_style_regime="normative_explicit",
    )


def test_risk_prompts_have_no_forbidden_normative_leakage_in_non_normative_regimes():
    generators = [
        "_generate_lottery_choice",
        "_generate_ce_offer_comparison",
        "_generate_mixed_gain_loss_choice",
        "_generate_time_discounting",
        "_generate_ambiguity_aversion_choice",
        "_generate_probability_weighting_counterexample",
        "_generate_loss_aversion_counterexample",
        "_generate_hyperbolic_discounting_counterexample",
    ]
    for regime in ("neutral_realistic", "bias_eliciting"):
        gen = RiskLossTimeGenerator(seed=22, prompt_style_regime=regime)
        forbidden = PROMPT_FORBIDDEN_PHRASES_BY_REGIME[regime]
        for idx, method_name in enumerate(generators):
            dp = getattr(gen, method_name)(idx)
            lower_prompt = dp.input.lower()
            for phrase in forbidden:
                assert phrase not in lower_prompt


def test_example_fingerprint_matches_stable_helper_payload():
    gen = RiskLossTimeGenerator(seed=9090)
    dp = gen._generate_mixed_gain_loss_choice(0)
    expected = gen._compute_example_fingerprint(
        task_subtype=dp.task_subtype,
        problem_spec=dp.problem_spec,
        optimal_decision=dp.target.optimal_decision,
    )
    assert dp.metadata.example_fingerprint == expected


def test_example_fingerprint_is_deterministic_for_same_seed_and_subtype():
    first = RiskLossTimeGenerator(seed=1717)._generate_time_discounting(0)
    second = RiskLossTimeGenerator(seed=1717)._generate_time_discounting(0)
    assert first.metadata.example_fingerprint == second.metadata.example_fingerprint


def test_ce_offer_comparison_has_formula_based_outcome_model():
    gen = RiskLossTimeGenerator(seed=202)
    dp = gen._generate_ce_offer_comparison(0)

    offered = dp.target.state["options"]["A"]["amount"]
    lottery = dp.target.state["options"]["B"]
    p_win = lottery["p_win"]
    high = lottery["high"]
    low = lottery["low"]

    assert dp.target.outcome_model["accept_offer"] == f"{offered}"
    assert dp.target.outcome_model["reject_offer"] == (
        f"{p_win} * {high} + (1 - {p_win}) * {low}"
    )


def test_outcome_model_helpers_match_generated_target_formulas():
    gen = RiskLossTimeGenerator(seed=2024)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    helper_by_subtype = {
        "lottery_choice": gen._build_lottery_outcome_model,
        "ce_offer_comparison": gen._build_ce_offer_outcome_model,
        "mixed_gain_loss_choice": gen._build_mixed_gain_loss_outcome_model,
        "time_discounting": gen._build_time_discounting_outcome_model,
        "ambiguity_aversion_choice": gen._build_ambiguity_aversion_outcome_model,
        "probability_weighting_counterexample": (
            gen._build_probability_weighting_counterexample_outcome_model
        ),
        "loss_aversion_counterexample": gen._build_loss_aversion_counterexample_outcome_model,
        "hyperbolic_discounting_counterexample": (
            gen._build_hyperbolic_discounting_counterexample_outcome_model
        ),
    }

    for dp in datapoints:
        helper = helper_by_subtype[dp.task_subtype]
        assert helper(problem_spec=dp.problem_spec) == dp.target.outcome_model


def test_ce_offer_comparison_prompt_uses_action_aligned_wording():
    gen = RiskLossTimeGenerator(seed=303)
    dp = gen._generate_ce_offer_comparison(0)

    assert "accept the certain offer (accept_offer)" in dp.input
    assert "keep the lottery (reject_offer)" in dp.input


def test_renderers_support_multiple_styles_per_problem_spec():
    gen = RiskLossTimeGenerator(seed=0)
    styles = [
        "default",
        "formal",
        "plain_english",
        "compact",
        "finance_framed",
        "unlabeled",
    ]

    lottery_spec = gen._build_lottery_problem_spec(
        sure_amount=50,
        p_win=0.4,
        win_amount=120,
        lose_amount=0,
    )
    ce_spec = gen._build_ce_offer_comparison_problem_spec(
        offered=88.5,
        p_win=0.6,
        high=150,
        low=20,
    )
    mixed_spec = gen._build_mixed_gain_loss_problem_spec(
        sure=-10,
        p_gain=0.55,
        gain=100,
        loss=-60,
    )
    time_spec = gen._build_time_discounting_problem_spec(
        now=40,
        later_offer=44.5,
        days=30,
        annual_rate=0.08,
    )
    ambiguity_spec = gen._build_ambiguity_aversion_problem_spec(
        known_probability=0.5,
        known_win_amount=100,
        known_lose_amount=0,
        ambiguous_win_amount=120,
        ambiguous_lose_amount=0,
        subjective_ambiguous_win_probability=0.45,
    )
    probability_weighting_spec = gen._build_probability_weighting_counterexample_problem_spec(
        sure_amount=52,
        p_win=0.08,
        win_amount=500,
        lose_amount=0,
    )
    loss_aversion_spec = gen._build_loss_aversion_counterexample_problem_spec(
        status_quo_amount=0,
        p_gain=0.55,
        gain=140,
        loss=-90,
    )
    hyperbolic_spec = gen._build_hyperbolic_discounting_counterexample_problem_spec(
        earlier_amount=100,
        earlier_delay_days=7,
        later_amount=108,
        later_delay_days=30,
        annual_rate=0.08,
    )

    lottery_prompts = [
        gen._render_lottery_choice_prompt(problem_spec=lottery_spec, style=style)
        for style in styles
    ]
    ce_prompts = [
        gen._render_ce_offer_comparison_prompt(problem_spec=ce_spec, style=style)
        for style in styles
    ]
    mixed_prompts = [
        gen._render_mixed_gain_loss_choice_prompt(problem_spec=mixed_spec, style=style)
        for style in styles
    ]
    time_prompts = [
        gen._render_time_discounting_prompt(problem_spec=time_spec, style=style)
        for style in styles
    ]
    ambiguity_prompts = [
        gen._render_ambiguity_aversion_choice_prompt(
            problem_spec=ambiguity_spec, style=style
        )
        for style in styles
    ]
    probability_weighting_prompts = [
        gen._render_probability_weighting_counterexample_prompt(
            problem_spec=probability_weighting_spec, style=style
        )
        for style in styles
    ]
    loss_aversion_prompts = [
        gen._render_loss_aversion_counterexample_prompt(
            problem_spec=loss_aversion_spec, style=style
        )
        for style in styles
    ]
    hyperbolic_prompts = [
        gen._render_hyperbolic_discounting_counterexample_prompt(
            problem_spec=hyperbolic_spec, style=style
        )
        for style in styles
    ]

    for prompt in lottery_prompts[:-1]:
        assert "choose_sure" in prompt
        assert "choose_lottery" in prompt
    for prompt in ce_prompts[:-1]:
        assert "accept_offer" in prompt
        assert "reject_offer" in prompt
    for prompt in mixed_prompts[:-1]:
        assert "choose_sure" in prompt
        assert "choose_risky" in prompt
    for prompt in time_prompts[:-1]:
        assert "choose_now" in prompt
        assert "choose_later" in prompt
    for prompt in ambiguity_prompts[:-1]:
        assert "choose_known_risk" in prompt
        assert "choose_ambiguous" in prompt
    for prompt in probability_weighting_prompts[:-1]:
        assert "choose_sure" in prompt
        assert "choose_longshot" in prompt
    for prompt in loss_aversion_prompts[:-1]:
        assert "choose_status_quo" in prompt
        assert "accept_gamble" in prompt
    for prompt in hyperbolic_prompts[:-1]:
        assert "choose_earlier" in prompt
        assert "choose_later" in prompt

    assert len(set(lottery_prompts)) > 1
    assert len(set(ce_prompts)) > 1
    assert len(set(mixed_prompts)) > 1
    assert len(set(time_prompts)) > 1
    assert len(set(ambiguity_prompts)) > 1
    assert len(set(probability_weighting_prompts)) > 1
    assert len(set(loss_aversion_prompts)) > 1
    assert len(set(hyperbolic_prompts)) > 1


def test_unlabeled_prompt_style_omits_action_tokens_but_keeps_structured_targets():
    gen = RiskLossTimeGenerator(seed=909, prompt_style="unlabeled")
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    subtype_action_tokens = {
        "lottery_choice": {"choose_sure", "choose_lottery"},
        "ce_offer_comparison": {"accept_offer", "reject_offer"},
        "mixed_gain_loss_choice": {"choose_sure", "choose_risky"},
        "time_discounting": {"choose_now", "choose_later"},
        "ambiguity_aversion_choice": {"choose_known_risk", "choose_ambiguous"},
        "probability_weighting_counterexample": {"choose_sure", "choose_longshot"},
        "loss_aversion_counterexample": {"choose_status_quo", "accept_gamble"},
        "hyperbolic_discounting_counterexample": {"choose_earlier", "choose_later"},
    }

    for dp in datapoints:
        assert dp.metadata.requested_prompt_style == "unlabeled"
        assert dp.metadata.resolved_prompt_style == "unlabeled"
        assert dp.metadata.prompt_has_action_labels is False
        assert dp.target.comparison_pair["left_action"] in dp.target.actions
        assert dp.target.comparison_pair["right_action"] in dp.target.actions
        for token in subtype_action_tokens[dp.task_subtype]:
            assert token not in dp.input


def test_random_prompt_style_is_deterministic_under_fixed_seed():
    first = RiskLossTimeGenerator(seed=77, prompt_style="random")._generate_lottery_choice(
        0
    )
    second = RiskLossTimeGenerator(
        seed=77, prompt_style="random"
    )._generate_lottery_choice(0)
    assert first.input == second.input


def test_solver_verification_recomputes_target_fields_from_problem_spec():
    gen = RiskLossTimeGenerator(seed=404)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]

    for dp in datapoints:
        expected_action_values, expected_decision_values, expected_optimal_decision = (
            gen._solve_from_problem_spec(problem_spec=dp.problem_spec)
        )
        normalized_expected_action_values = gen._normalize_scalars_for_comparison_pair(
            values=expected_action_values, comparison_pair=dp.target.comparison_pair
        )
        normalized_expected_decision_values = gen._normalize_scalars_for_comparison_pair(
            values=expected_decision_values, comparison_pair=dp.target.comparison_pair
        )
        assert dp.target.action_values == normalized_expected_action_values
        assert list(dp.target.action_values.keys()) == list(
            normalized_expected_action_values.keys()
        )
        assert dp.target.decision_values == normalized_expected_decision_values
        assert list(dp.target.decision_values.keys()) == list(
            normalized_expected_decision_values.keys()
        )
        assert dp.target.optimal_decision == expected_optimal_decision


def test_solver_verification_raises_on_mismatched_target_fields():
    gen = RiskLossTimeGenerator(seed=505)
    dp = gen._generate_lottery_choice(0)
    tampered_target = Target(
        objective=dp.target.objective,
        state=dp.target.state,
        beliefs=dp.target.beliefs,
        constraints=dp.target.constraints,
        actions=dp.target.actions,
        comparison_pair=dp.target.comparison_pair,
        outcome_model=dp.target.outcome_model,
        action_values={
            "choose_lottery": dp.target.action_values["choose_lottery"],
            "choose_sure": -1.0,
        },
        decision_values=dp.target.decision_values,
        optimal_decision=dp.target.optimal_decision,
        solver_trace=dp.target.solver_trace,
        brief_rationale=dp.target.brief_rationale,
    )

    with pytest.raises(ValueError, match="target.action_values.choose_sure mismatch"):
        gen._verify_target_solution(problem_spec=dp.problem_spec, target=tampered_target)


def test_solver_verification_raises_on_mismatched_action_value_ordering():
    gen = RiskLossTimeGenerator(seed=606)
    dp = gen._generate_lottery_choice(0)
    tampered_target = Target(
        objective=dp.target.objective,
        state=dp.target.state,
        beliefs=dp.target.beliefs,
        constraints=dp.target.constraints,
        actions=dp.target.actions,
        comparison_pair=dp.target.comparison_pair,
        outcome_model=dp.target.outcome_model,
        action_values={
            "choose_sure": dp.target.action_values["choose_sure"],
            "choose_lottery": dp.target.action_values["choose_lottery"],
        },
        decision_values=dp.target.decision_values,
        optimal_decision=dp.target.optimal_decision,
        solver_trace=dp.target.solver_trace,
        brief_rationale=dp.target.brief_rationale,
    )

    with pytest.raises(
        ValueError, match="target.action_values action ordering does not match solver output"
    ):
        gen._verify_target_solution(problem_spec=dp.problem_spec, target=tampered_target)


def test_solver_validation_rejects_missing_required_lottery_field():
    gen = RiskLossTimeGenerator(seed=707)
    dp = gen._generate_lottery_choice(0)
    malformed = deepcopy(dp.problem_spec)
    del malformed["options"]["B"]["p_win"]

    with pytest.raises(
        ValueError, match=r"problem_spec\.options\.B\.p_win is required"
    ):
        gen._solve_from_problem_spec(problem_spec=malformed)


def test_solver_validation_rejects_invalid_ce_field_type():
    gen = RiskLossTimeGenerator(seed=708)
    dp = gen._generate_ce_offer_comparison(0)
    malformed = deepcopy(dp.problem_spec)
    malformed["options"]["B"]["high"] = "bad"

    with pytest.raises(
        ValueError, match=r"problem_spec\.options\.B\.high must be numeric"
    ):
        gen._solve_from_problem_spec(problem_spec=malformed)


def test_solver_validation_rejects_out_of_range_mixed_probability():
    gen = RiskLossTimeGenerator(seed=709)
    dp = gen._generate_mixed_gain_loss_choice(0)
    malformed = deepcopy(dp.problem_spec)
    malformed["options"]["B"]["p_gain"] = 1.2

    with pytest.raises(
        ValueError, match=r"problem_spec\.options\.B\.p_gain must be between 0 and 1"
    ):
        gen._solve_from_problem_spec(problem_spec=malformed)


def test_solver_validation_rejects_invalid_time_discounting_constraints():
    gen = RiskLossTimeGenerator(seed=710)
    dp = gen._generate_time_discounting(0)

    malformed_delay = deepcopy(dp.problem_spec)
    malformed_delay["options"]["B"]["delay_days"] = -30
    with pytest.raises(
        ValueError, match=r"problem_spec\.options\.B\.delay_days must be non-negative"
    ):
        gen._solve_from_problem_spec(problem_spec=malformed_delay)

    malformed_rate = deepcopy(dp.problem_spec)
    malformed_rate["assumptions"]["annual_discount_rate"] = -0.01
    with pytest.raises(
        ValueError,
        match=r"problem_spec\.assumptions\.annual_discount_rate must be non-negative",
    ):
        gen._solve_from_problem_spec(problem_spec=malformed_rate)


def test_ambiguity_aversion_solver_uses_subjective_ambiguous_probability():
    gen = RiskLossTimeGenerator(seed=0)
    problem_spec = gen._build_ambiguity_aversion_problem_spec(
        known_probability=0.5,
        known_win_amount=100,
        known_lose_amount=0,
        ambiguous_win_amount=120,
        ambiguous_lose_amount=0,
        subjective_ambiguous_win_probability=0.45,
    )

    action_values, decision_values, optimal = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    assert action_values == {
        "choose_known_risk": 50.0,
        "choose_ambiguous": 54.0,
    }
    assert decision_values == action_values
    assert optimal == "choose_ambiguous"


def test_probability_weighting_counterexample_solver_remains_plain_ev():
    gen = RiskLossTimeGenerator(seed=0)
    problem_spec = gen._build_probability_weighting_counterexample_problem_spec(
        sure_amount=60,
        p_win=0.05,
        win_amount=500,
        lose_amount=0,
    )

    action_values, decision_values, optimal = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    assert action_values == {
        "choose_sure": 60.0,
        "choose_longshot": 25.0,
    }
    assert decision_values == action_values
    assert optimal == "choose_sure"


def test_ambiguity_generator_balances_ev_regimes():
    gen = RiskLossTimeGenerator(seed=2026)
    saw_choose_ambiguous_region = False
    saw_choose_known_risk_region = False
    saw_near_indifferent_region = False

    for i in range(60):
        dp = gen._generate_ambiguity_aversion_choice(i)
        ev_ambiguous = dp.target.action_values["choose_ambiguous"]
        ev_known_risk = dp.target.action_values["choose_known_risk"]
        gap = ev_ambiguous - ev_known_risk

        if gap >= 6.0:
            saw_choose_ambiguous_region = True
            assert dp.target.optimal_decision == "choose_ambiguous"
        elif gap <= -6.0:
            saw_choose_known_risk_region = True
            assert dp.target.optimal_decision == "choose_known_risk"
        elif abs(gap) <= 3.0:
            saw_near_indifferent_region = True

    assert saw_choose_ambiguous_region
    assert saw_choose_known_risk_region
    assert saw_near_indifferent_region


def test_probability_weighting_generator_balances_ev_orderings():
    gen = RiskLossTimeGenerator(seed=2026)
    saw_longshot_ev_superior = False
    saw_sure_ev_superior = False
    saw_choose_longshot_optimal = False
    saw_choose_sure_optimal = False

    for i in range(40):
        dp = gen._generate_probability_weighting_counterexample(i)
        ev_longshot = dp.target.action_values["choose_longshot"]
        ev_sure = dp.target.action_values["choose_sure"]

        if ev_longshot > ev_sure:
            saw_longshot_ev_superior = True
            assert dp.target.optimal_decision == "choose_longshot"
        elif ev_sure > ev_longshot:
            saw_sure_ev_superior = True
            assert dp.target.optimal_decision == "choose_sure"

        if dp.target.optimal_decision == "choose_longshot":
            saw_choose_longshot_optimal = True
        if dp.target.optimal_decision == "choose_sure":
            saw_choose_sure_optimal = True

    assert saw_longshot_ev_superior
    assert saw_sure_ev_superior
    assert saw_choose_longshot_optimal
    assert saw_choose_sure_optimal


def test_loss_aversion_counterexample_solver_accepts_positive_ev_gamble():
    gen = RiskLossTimeGenerator(seed=0)
    problem_spec = gen._build_loss_aversion_counterexample_problem_spec(
        status_quo_amount=0,
        p_gain=0.6,
        gain=120,
        loss=-80,
    )

    action_values, decision_values, optimal = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    assert action_values == {
        "choose_status_quo": 0.0,
        "accept_gamble": 40.0,
    }
    assert decision_values == action_values
    assert optimal == "accept_gamble"


def test_loss_aversion_counterexample_generator_varies_status_quo_amount():
    gen = RiskLossTimeGenerator(seed=123)
    status_quo_amounts: list[int] = []

    for i in range(30):
        dp = gen._generate_loss_aversion_counterexample(i)
        status_quo = int(dp.problem_spec["options"]["A"]["amount"])
        status_quo_amounts.append(status_quo)
        assert -30 <= status_quo <= 30

    assert any(amount != 0 for amount in status_quo_amounts)


def test_loss_aversion_generator_balances_ev_regimes():
    gen = RiskLossTimeGenerator(seed=2027)
    saw_accept_gamble_region = False
    saw_choose_status_quo_region = False
    saw_near_indifferent_region = False

    for i in range(60):
        dp = gen._generate_loss_aversion_counterexample(i)
        ev_gamble = dp.target.action_values["accept_gamble"]
        ev_status_quo = dp.target.action_values["choose_status_quo"]
        gap = ev_gamble - ev_status_quo

        if gap >= 6.0:
            saw_accept_gamble_region = True
            assert dp.target.optimal_decision == "accept_gamble"
        elif gap <= -6.0:
            saw_choose_status_quo_region = True
            assert dp.target.optimal_decision == "choose_status_quo"
        elif abs(gap) <= 3.0:
            saw_near_indifferent_region = True

    assert saw_accept_gamble_region
    assert saw_choose_status_quo_region
    assert saw_near_indifferent_region


def test_loss_aversion_counterexample_uses_regime_safe_fallback_after_failed_search():
    gen = RiskLossTimeGenerator(seed=7)
    original_choice = gen.rng.choice
    original_uniform = gen.rng.uniform

    def fake_choice(options):
        if "choose_status_quo" in options:
            return "choose_status_quo"
        return original_choice(options)

    def fake_uniform(a, b):
        if a == 0.45 and b == 0.7:
            return 0.45
        return 1000.0

    gen.rng.choice = fake_choice
    gen.rng.uniform = fake_uniform
    try:
        dp = gen._generate_loss_aversion_counterexample(0)
    finally:
        gen.rng.choice = original_choice
        gen.rng.uniform = original_uniform

    assert dp.metadata.difficulty_metrics is not None
    assert (
        dp.metadata.difficulty_metrics["mixed_gamble_regime_intended"]
        == "choose_status_quo"
    )
    gap = (
        dp.target.action_values["accept_gamble"]
        - dp.target.action_values["choose_status_quo"]
    )
    assert gap <= -6.0
    assert (
        dp.metadata.difficulty_metrics["mixed_gamble_regime_realized"]
        == "choose_status_quo"
    )


def test_hyperbolic_discounting_counterexample_solver_uses_stated_discount_model():
    gen = RiskLossTimeGenerator(seed=0)
    problem_spec = gen._build_hyperbolic_discounting_counterexample_problem_spec(
        earlier_amount=100,
        earlier_delay_days=7,
        later_amount=108.0,
        later_delay_days=30,
        annual_rate=0.08,
    )

    action_values, decision_values, optimal = gen._solve_from_problem_spec(
        problem_spec=problem_spec
    )
    assert action_values == {
        "choose_earlier": gen._round_expected_utility(100 / (1 + 0.08 * (7 / 365))),
        "choose_later": gen._round_expected_utility(108 / (1 + 0.08 * (30 / 365))),
    }
    assert decision_values == action_values
    assert optimal == "choose_later"


def test_hyperbolic_generator_balances_discount_value_regimes():
    gen = RiskLossTimeGenerator(seed=2028)
    saw_choose_later_region = False
    saw_choose_earlier_region = False
    saw_near_indifferent_region = False

    for i in range(60):
        dp = gen._generate_hyperbolic_discounting_counterexample(i)
        pv_later = dp.target.action_values["choose_later"]
        pv_earlier = dp.target.action_values["choose_earlier"]
        gap = pv_later - pv_earlier

        if gap >= 4.0:
            saw_choose_later_region = True
            assert dp.target.optimal_decision == "choose_later"
        elif gap <= -4.0:
            saw_choose_earlier_region = True
            assert dp.target.optimal_decision == "choose_earlier"
        elif abs(gap) <= 2.0:
            saw_near_indifferent_region = True

    assert saw_choose_later_region
    assert saw_choose_earlier_region
    assert saw_near_indifferent_region


def test_subtypes_expose_action_value_semantics_metadata():
    gen = RiskLossTimeGenerator(seed=29)
    datapoints = [
        gen._generate_lottery_choice(0),
        gen._generate_ce_offer_comparison(1),
        gen._generate_mixed_gain_loss_choice(2),
        gen._generate_time_discounting(3),
        gen._generate_ambiguity_aversion_choice(4),
        gen._generate_probability_weighting_counterexample(5),
        gen._generate_loss_aversion_counterexample(6),
        gen._generate_hyperbolic_discounting_counterexample(7),
    ]
    expected = {
        "lottery_choice": "expected_value_comparison",
        "ce_offer_comparison": "expected_value_comparison",
        "mixed_gain_loss_choice": "expected_value_comparison",
        "time_discounting": "discounted_value_comparison",
        "ambiguity_aversion_choice": "expected_value_comparison",
        "probability_weighting_counterexample": "expected_value_comparison",
        "loss_aversion_counterexample": "expected_value_comparison",
        "hyperbolic_discounting_counterexample": "discounted_value_comparison",
    }

    for dp in datapoints:
        metrics = dp.metadata.difficulty_metrics
        assert metrics is not None
        assert metrics["action_value_semantics"] == expected[dp.task_subtype]
