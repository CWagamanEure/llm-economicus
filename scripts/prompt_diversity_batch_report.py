#!/usr/bin/env python3
"""Batch prompt diversity diagnostics with baseline vs balanced template selection."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_GENERATION_DIR = SRC_DIR / "data-generation"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DATA_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_DIR))

from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402
from belief_bias_generator import BeliefBiasGenerator  # noqa: E402
from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    generator_cls: type
    seed: int


FAMILIES: tuple[FamilyConfig, ...] = (
    FamilyConfig("risk_loss_time_choice", RiskLossTimeGenerator, 41),
    FamilyConfig("bayesian_signal_extraction", BayesianSignalGenerator, 43),
    FamilyConfig("belief_bias_judgment", BeliefBiasGenerator, 44),
)


def _normalize_prompt(text: str) -> str:
    compact = " ".join(text.lower().split())
    compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
    return compact


def _opening_signature(text: str) -> str:
    tokens = re.findall(r"[a-z0-9_]+", _normalize_prompt(text))
    return " ".join(tokens[:3])


def _jaccard_mean(prompts: list[str]) -> float:
    if len(prompts) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a = set(re.findall(r"[a-z0-9_]+", prompts[i].lower()))
            b = set(re.findall(r"[a-z0-9_]+", prompts[j].lower()))
            scores.append(len(a & b) / max(1, len(a | b)))
    return sum(scores) / len(scores)


def _similarity_mean(prompts: list[str]) -> float:
    if len(prompts) < 2:
        return 0.0
    scores: list[float] = []
    normalized = [_normalize_prompt(p) for p in prompts]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            scores.append(SequenceMatcher(None, normalized[i], normalized[j]).ratio())
    return sum(scores) / len(scores)


def _repeated_opening_rate(prompts: list[str]) -> float:
    if not prompts:
        return 0.0
    openings = [_opening_signature(p) for p in prompts if _opening_signature(p)]
    if not openings:
        return 0.0
    return 1.0 - (len(set(openings)) / len(openings))


def _run_batch(
    family: FamilyConfig,
    *,
    count: int,
    balancing_enabled: bool,
) -> dict[str, Any]:
    generator = family.generator_cls(
        seed=family.seed,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="auto",
    )
    generator._enable_prompt_diversity_balancing = balancing_enabled  # noqa: SLF001
    generator.reset_prompt_diversity_state()

    prompts: list[str] = []
    for _ in range(count):
        prompts.append(generator.generate().input)

    internal = generator.render_diversity_diagnostics(top_k=8)
    return {
        "prompts": prompts,
        "metrics": {
            "lexical_overlap_jaccard_mean": round(_jaccard_mean(prompts), 4),
            "normalized_similarity_mean": round(_similarity_mean(prompts), 4),
            "repeated_opening_rate": round(_repeated_opening_rate(prompts), 4),
        },
        "template_usage_frequency": internal.get("template_usage_frequency", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prompt diversity metrics before/after anti-repetition balancing."
    )
    parser.add_argument(
        "--count-per-family",
        type=int,
        default=30,
        help="Number of samples to generate per family for each mode.",
    )
    args = parser.parse_args()

    for family in FAMILIES:
        before = _run_batch(
            family,
            count=args.count_per_family,
            balancing_enabled=False,
        )
        after = _run_batch(
            family,
            count=args.count_per_family,
            balancing_enabled=True,
        )
        print("\n" + "=" * 96)
        print(f"FAMILY: {family.name}")
        print("=" * 96)
        print("before (baseline hash selection):")
        print(before["metrics"])
        print("after (balanced selection):")
        print(after["metrics"])
        print("most-used templates before:")
        print(before["template_usage_frequency"][:5])
        print("most-used templates after:")
        print(after["template_usage_frequency"][:5])


if __name__ == "__main__":
    main()
