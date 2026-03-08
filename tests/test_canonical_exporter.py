import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_GENERATION_DIR = ROOT / "src" / "data-generation"
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402

from canonical_exporter import (  # noqa: E402
    datapoint_to_canonical_dict,
    datapoint_to_canonical_json,
    write_canonical_jsonl,
)


def test_canonical_dict_has_stable_top_level_and_nested_key_order():
    dp = RiskLossTimeGenerator(seed=10)._generate_lottery_choice(0)
    canonical = datapoint_to_canonical_dict(dp)

    assert list(canonical.keys()) == [
        "task_family",
        "task_subtype",
        "task_id",
        "difficulty",
        "problem_spec",
        "input",
        "target",
        "metadata",
    ]
    assert list(canonical["problem_spec"].keys()) == [
        "task_subtype",
        "objective",
        "options",
        "assumptions",
    ]
    assert list(canonical["target"].keys()) == [
        "objective",
        "state",
        "beliefs",
        "constraints",
        "actions",
        "comparison_pair",
        "outcome_model",
        "action_values",
        "decision_values",
        "optimal_decision",
        "solver_trace",
        "brief_rationale",
    ]
    assert list(canonical["metadata"].keys()) == [
        "generator_name",
        "version",
        "seed",
        "dataset_role",
        "requested_prompt_style",
        "resolved_prompt_style",
        "prompt_has_action_labels",
        "example_fingerprint",
        "tie_threshold",
        "sample_index",
        "difficulty_metrics",
    ]

    left_action = dp.target.comparison_pair["left_action"]
    right_action = dp.target.comparison_pair["right_action"]
    assert list(canonical["target"]["action_values"].keys())[:2] == [
        left_action,
        right_action,
    ]
    assert list(canonical["target"]["decision_values"].keys())[:2] == [
        left_action,
        right_action,
    ]


def test_canonical_json_serialization_is_deterministic():
    dp = RiskLossTimeGenerator(seed=11)._generate_time_discounting(0)
    first = datapoint_to_canonical_json(dp)
    second = datapoint_to_canonical_json(dp)
    assert first == second
    assert json.loads(first) == json.loads(second)


def test_write_canonical_jsonl_writes_one_canonical_datapoint_per_line(tmp_path):
    gen = RiskLossTimeGenerator(seed=12)
    datapoints = [gen._generate_lottery_choice(0), gen._generate_ce_offer_comparison(1)]
    output_path = tmp_path / "dataset.jsonl"

    write_canonical_jsonl(datapoints, output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == datapoint_to_canonical_dict(datapoints[0])
    assert json.loads(lines[1]) == datapoint_to_canonical_dict(datapoints[1])
