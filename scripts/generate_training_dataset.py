import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_GENERATION_DIR = SRC_DIR / "data-generation"


def _bootstrap_import_paths() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(DATA_GENERATION_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_GENERATION_DIR))


_bootstrap_import_paths()

from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402
from belief_bias_generator import BeliefBiasGenerator  # noqa: E402
from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402

from canonical_exporter import (  # noqa: E402
    datapoint_to_canonical_dict,
    write_canonical_jsonl,
)


def _datapoint_to_sft_record(datapoint: Any) -> dict[str, Any]:
    canonical = datapoint_to_canonical_dict(datapoint)
    target = canonical["target"]
    return {
        "task_id": canonical["task_id"],
        "task_family": canonical["task_family"],
        "task_subtype": canonical["task_subtype"],
        "input": canonical["input"],
        "output": {
            "optimal_decision": target["optimal_decision"],
            "action_values": target["action_values"],
            "decision_values": target["decision_values"],
            "brief_rationale": target["brief_rationale"],
        },
        "metadata": canonical["metadata"],
    }


def _write_sft_jsonl(datapoints: Iterable[Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for datapoint in datapoints:
            record = _datapoint_to_sft_record(datapoint)
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a mixed training dataset from all current generator families."
    )
    parser.add_argument(
        "--count-per-generator",
        type=int,
        default=100,
        help="Number of samples to generate per generator family (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic output (default: 42).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Metadata version tag to attach to each sample (default: v1).",
    )
    parser.add_argument(
        "--prompt-style",
        default="default",
        help="Prompt rendering style for all generators (default: default).",
    )
    parser.add_argument(
        "--format",
        choices=("canonical", "sft"),
        default="canonical",
        help=(
            "Output format: canonical schema rows or instruction-output SFT rows "
            "(default: canonical)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Deterministically shuffle merged rows using the base seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "training_all_generators.jsonl",
        help="Output JSONL path (default: data/training_all_generators.jsonl).",
    )
    return parser


def _build_generators(seed: int, version: str, prompt_style: str) -> list[Any]:
    return [
        RiskLossTimeGenerator(seed=seed, version=version, prompt_style=prompt_style),
        BayesianSignalGenerator(seed=seed + 1, version=version, prompt_style=prompt_style),
        BeliefBiasGenerator(seed=seed + 2, version=version, prompt_style=prompt_style),
    ]


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count_per_generator <= 0:
        parser.error("--count-per-generator must be a positive integer.")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generators = _build_generators(
        seed=args.seed,
        version=args.version,
        prompt_style=args.prompt_style,
    )

    datapoints = [
        datapoint
        for generator in generators
        for datapoint in (generator.generate() for _ in range(args.count_per_generator))
    ]

    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(datapoints)

    if args.format == "canonical":
        write_canonical_jsonl(datapoints, output_path)
    else:
        _write_sft_jsonl(datapoints, output_path)

    print(
        "Wrote %s samples to %s (generators=%s, count_per_generator=%s, format=%s, shuffle=%s)"
        % (
            len(datapoints),
            output_path,
            len(generators),
            args.count_per_generator,
            args.format,
            args.shuffle,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
