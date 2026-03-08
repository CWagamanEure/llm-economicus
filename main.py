import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_GENERATION_DIR = SRC_DIR / "data-generation"


def _bootstrap_import_paths() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if str(DATA_GENERATION_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_GENERATION_DIR))


_bootstrap_import_paths()

from risk_loss_time_generator import (  # noqa: E402
    SUPPORTED_PROMPT_STYLES,
    RiskLossTimeGenerator,
)

from canonical_exporter import write_canonical_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate mock risk/loss/time datapoints as canonical JSONL."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generator seed for deterministic output (default: 42).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Metadata version tag to attach to each sample (default: v1).",
    )
    parser.add_argument(
        "--prompt-style",
        choices=SUPPORTED_PROMPT_STYLES,
        default="default",
        help="Prompt rendering style (default: default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "mock_risk_loss_time.jsonl",
        help="Output JSONL path (default: data/mock_risk_loss_time.jsonl).",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count <= 0:
        parser.error("--count must be a positive integer.")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = RiskLossTimeGenerator(
        seed=args.seed,
        version=args.version,
        prompt_style=args.prompt_style,
    )
    datapoints = [generator.generate() for _ in range(args.count)]
    write_canonical_jsonl(datapoints, output_path)

    print(f"Wrote {len(datapoints)} samples to {output_path}")
    print(
        "Seed=%s, version=%s, prompt_style=%s"
        % (args.seed, args.version, args.prompt_style)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
