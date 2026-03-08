import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

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
