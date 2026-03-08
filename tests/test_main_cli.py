import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_run_writes_expected_number_of_jsonl_rows(tmp_path: Path):
    output_path = tmp_path / "mock.jsonl"

    exit_code = main.run(
        [
            "--count",
            "5",
            "--seed",
            "123",
            "--prompt-style",
            "default",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    first_row = json.loads(lines[0])
    last_row = json.loads(lines[-1])
    assert first_row["metadata"]["seed"] == 123
    assert first_row["metadata"]["sample_index"] == 0
    assert last_row["metadata"]["sample_index"] == 4


def test_run_rejects_non_positive_count():
    try:
        main.run(["--count", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject --count=0")


def test_run_supports_bayesian_signal_generator(tmp_path: Path):
    output_path = tmp_path / "bayesian.jsonl"

    exit_code = main.run(
        [
            "--generator",
            "bayesian_signal",
            "--count",
            "3",
            "--seed",
            "17",
            "--prompt-style",
            "default",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    first_row = json.loads(rows[0])
    assert first_row["metadata"]["generator_name"] == "BayesianSignalGenerator"
    assert first_row["metadata"]["seed"] == 17
    assert first_row["metadata"]["sample_index"] == 0
