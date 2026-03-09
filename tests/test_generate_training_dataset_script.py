import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_training_dataset.py"


def test_script_generates_canonical_dataset_from_all_generators(tmp_path: Path):
    output_path = tmp_path / "training_canonical.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count-per-generator",
            "2",
            "--seed",
            "19",
            "--format",
            "canonical",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6

    generator_names = {row["metadata"]["generator_name"] for row in rows}
    assert generator_names == {
        "RiskLossTimeGenerator",
        "BayesianSignalGenerator",
        "BeliefBiasGenerator",
    }

    risk_rows = [
        row for row in rows if row["metadata"]["generator_name"] == "RiskLossTimeGenerator"
    ]
    bayes_rows = [
        row for row in rows if row["metadata"]["generator_name"] == "BayesianSignalGenerator"
    ]
    belief_rows = [
        row for row in rows if row["metadata"]["generator_name"] == "BeliefBiasGenerator"
    ]

    assert [row["metadata"]["sample_index"] for row in risk_rows] == [0, 1]
    assert [row["metadata"]["sample_index"] for row in bayes_rows] == [0, 1]
    assert [row["metadata"]["sample_index"] for row in belief_rows] == [0, 1]



def test_script_generates_sft_dataset(tmp_path: Path):
    output_path = tmp_path / "training_sft.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count-per-generator",
            "1",
            "--seed",
            "13",
            "--format",
            "sft",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3

    first = rows[0]
    assert set(first.keys()) == {
        "task_id",
        "task_family",
        "task_subtype",
        "input",
        "output",
        "metadata",
    }
    assert set(first["output"].keys()) == {
        "optimal_decision",
        "action_values",
        "decision_values",
        "brief_rationale",
    }



def test_script_rejects_non_positive_count(tmp_path: Path):
    output_path = tmp_path / "invalid.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--count-per-generator",
            "0",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--count-per-generator must be a positive integer" in result.stderr
