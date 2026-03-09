import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_GENERATION_DIR = SRC_DIR / "data-generation"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402


def test_template_balancing_spreads_selection_on_repeated_calls():
    generator = BayesianSignalGenerator(seed=7)
    problem_spec = {
        "task_subtype": "basic_bayes_update",
        "options": {},
        "assumptions": {
            "prior_high": 0.4,
            "p_signal_high_given_high": 0.8,
            "p_signal_high_given_low": 0.3,
            "observed_signal": "high",
        },
    }
    templates = [
        "case memo alpha",
        "review note beta",
        "dashboard alert gamma",
        "quick brief delta",
    ]

    generator._enable_prompt_diversity_balancing = False  # noqa: SLF001
    generator.reset_prompt_diversity_state()
    baseline = [
        generator.select_template_index_balanced(
            task_subtype="basic_bayes_update",
            frame_variant="medical_screening",
            tier="neutral_natural",
            problem_spec=problem_spec,
            templates=templates,
        )
        for _ in range(8)
    ]
    assert len(set(baseline)) == 1

    generator._enable_prompt_diversity_balancing = True  # noqa: SLF001
    generator.reset_prompt_diversity_state()
    balanced = [
        generator.select_template_index_balanced(
            task_subtype="basic_bayes_update",
            frame_variant="medical_screening",
            tier="neutral_natural",
            problem_spec=problem_spec,
            templates=templates,
        )
        for _ in range(8)
    ]
    assert len(set(balanced)) > 1
