#!/usr/bin/env python3
"""Qualitative prompt inspection for dataset generators.

Samples underlying problems per task family, then re-renders each in all
prompt_style_regime variants and multiple prompt_frame_variant settings.
Outputs grouped comparisons with leakage and lexical-overlap diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from base_generator import PROMPT_FORBIDDEN_PHRASES_BY_REGIME  # noqa: E402
from bayesian_signal_generator import BayesianSignalGenerator  # noqa: E402
from belief_bias_generator import BeliefBiasGenerator  # noqa: E402
from risk_loss_time_generator import RiskLossTimeGenerator  # noqa: E402

REGIMES: tuple[str, ...] = (
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
)


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _normalize_prompt(text: str) -> str:
    compact = " ".join(text.lower().split())
    compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
    compact = re.sub(r"'[^']*'", "<quoted>", compact)
    return compact


def _pairwise_overlap(prompts: list[str]) -> dict[str, float]:
    if len(prompts) < 2:
        return {"min_jaccard": 1.0, "max_jaccard": 1.0, "mean_jaccard": 1.0}
    scores: list[float] = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a = _tokenize(prompts[i])
            b = _tokenize(prompts[j])
            score = len(a & b) / max(1, len(a | b))
            scores.append(score)
    return {
        "min_jaccard": min(scores),
        "max_jaccard": max(scores),
        "mean_jaccard": sum(scores) / len(scores),
    }


def _pairwise_normalized_similarity(prompts: list[str]) -> dict[str, float]:
    normalized = [_normalize_prompt(p) for p in prompts]
    if len(normalized) < 2:
        return {"min_sim": 1.0, "max_sim": 1.0, "mean_sim": 1.0}
    scores: list[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            scores.append(SequenceMatcher(None, normalized[i], normalized[j]).ratio())
    return {
        "min_sim": min(scores),
        "max_sim": max(scores),
        "mean_sim": sum(scores) / len(scores),
    }


def _problem_fingerprint(problem_spec: dict[str, Any]) -> str:
    canonical = json.dumps(problem_spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _sample_underlying_problems(
    family: FamilyConfig, count: int
) -> list[dict[str, Any]]:
    generator = family.generator_cls(
        seed=family.seed,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="auto",
    )
    sampled: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    max_draws = count * 20
    draws = 0
    while len(sampled) < count and draws < max_draws:
        draws += 1
        dp = generator.generate()
        fp = _problem_fingerprint(dp.problem_spec)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        sampled.append(
            {
                "task_family": dp.task_family,
                "task_subtype": dp.task_subtype,
                "problem_spec": dp.problem_spec,
                "target": dp.target,
                "fingerprint": fp,
            }
        )
    return sampled


def _frame_candidates(
    generator: Any, task_subtype: str, max_frames: int | None
) -> list[str]:
    candidates = list(generator._frame_candidates_for_subtype(task_subtype))
    if max_frames is not None:
        candidates = candidates[:max_frames]
    return candidates


def _render_prompt(
    generator: Any,
    *,
    task_subtype: str,
    problem_spec: dict[str, Any],
    regime: str,
    frame_variant: str,
    style: str = "default",
) -> dict[str, Any]:
    renderer = generator._prompt_renderer_for_subtype(task_subtype)
    prompt = renderer(
        problem_spec=problem_spec,
        style=style,
        prompt_style_regime=regime,
        prompt_frame_variant=frame_variant,
    )
    prompt = generator._apply_prompt_frame_variant(
        prompt=prompt,
        frame_variant=frame_variant,
        task_subtype=task_subtype,
    )
    forbidden = list(PROMPT_FORBIDDEN_PHRASES_BY_REGIME.get(regime, ()))
    lower = prompt.lower()
    leakage_hits = [phrase for phrase in forbidden if phrase in lower]
    semantic_context = frame_variant if isinstance(generator, BayesianSignalGenerator) else None
    return {
        "task_subtype": task_subtype,
        "prompt_style_regime": regime,
        "prompt_frame_variant": frame_variant,
        "semantic_context": semantic_context,
        "input": prompt,
        "prompt_length": len(prompt),
        "forbidden_hits": leakage_hits,
    }


def _print_group_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_underlying_problem(problem: dict[str, Any]) -> None:
    target = problem["target"]
    print(f"task_family      : {problem['task_family']}")
    print(f"task_subtype     : {problem['task_subtype']}")
    print(f"problem_fp       : {problem['fingerprint']}")
    print(f"optimal_decision : {target.optimal_decision}")
    print(f"target_actions   : {list(target.action_values.keys())}")
    print(
        "action_values    : "
        + ", ".join(f"{k}={v}" for k, v in target.action_values.items())
    )


def _print_variant_render(problem: dict[str, Any], rendered: dict[str, Any]) -> None:
    print("-" * 100)
    print(f"task_family          : {problem['task_family']}")
    print(f"task_subtype         : {rendered['task_subtype']}")
    print(f"prompt_style_regime  : {rendered['prompt_style_regime']}")
    print(f"prompt_frame_variant : {rendered['prompt_frame_variant']}")
    if rendered["semantic_context"] is not None:
        print(f"semantic_context     : {rendered['semantic_context']}")
    print(f"optimal_decision     : {problem['target'].optimal_decision}")
    print(f"prompt_length        : {rendered['prompt_length']}")
    print(
        "forbidden_leakage    : "
        + ("none" if not rendered["forbidden_hits"] else ", ".join(rendered["forbidden_hits"]))
    )
    print("input:")
    print(rendered["input"])


def inspect_family(
    family: FamilyConfig,
    *,
    num_problems: int,
    max_frames: int | None,
) -> None:
    _print_group_header(f"FAMILY: {family.name}")
    underlying = _sample_underlying_problems(family, num_problems)
    if not underlying:
        print("No problems sampled.")
        return

    for index, problem in enumerate(underlying, start=1):
        _print_group_header(f"{family.name} | sample {index}/{len(underlying)}")
        _print_underlying_problem(problem)
        task_subtype = problem["task_subtype"]
        base_generator = family.generator_cls(
            seed=family.seed,
            prompt_style="default",
            prompt_style_regime="neutral_realistic",
            prompt_frame_variant="auto",
        )
        frames = _frame_candidates(base_generator, task_subtype, max_frames=max_frames)

        rendered_variants: list[dict[str, Any]] = []
        for regime in REGIMES:
            for frame in frames:
                generator = family.generator_cls(
                    seed=family.seed,
                    prompt_style="default",
                    prompt_style_regime=regime,
                    prompt_frame_variant=frame,
                )
                rendered = _render_prompt(
                    generator,
                    task_subtype=task_subtype,
                    problem_spec=problem["problem_spec"],
                    regime=regime,
                    frame_variant=frame,
                )
                rendered_variants.append(rendered)
                _print_variant_render(problem, rendered)

        overlaps = _pairwise_overlap([entry["input"] for entry in rendered_variants])
        print("-" * 100)
        print(
            "lexical_overlap_jaccard : "
            f"min={overlaps['min_jaccard']:.4f} "
            f"mean={overlaps['mean_jaccard']:.4f} "
            f"max={overlaps['max_jaccard']:.4f}"
        )
        norm_sim = _pairwise_normalized_similarity(
            [entry["input"] for entry in rendered_variants]
        )
        print(
            "normalized_similarity   : "
            f"min={norm_sim['min_sim']:.4f} "
            f"mean={norm_sim['mean_sim']:.4f} "
            f"max={norm_sim['max_sim']:.4f}"
        )
        signal = SequenceMatcher(
            None,
            rendered_variants[0]["input"],
            rendered_variants[-1]["input"],
        ).ratio()
        print(f"structural_similarity   : first_vs_last={signal:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitative inspection of prompt regime/frame renderings."
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Underlying problems to sample per task family (default: 5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help=(
            "Max frame variants per subtype to render. "
            "Use --max-frames 0 to render all."
        ),
    )
    parser.add_argument(
        "--family",
        choices=[cfg.name for cfg in FAMILIES] + ["all"],
        default="all",
        help="Inspect one family or all families.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames: int | None = None if args.max_frames == 0 else args.max_frames
    families = FAMILIES if args.family == "all" else [f for f in FAMILIES if f.name == args.family]
    for family in families:
        inspect_family(family, num_problems=args.num_problems, max_frames=max_frames)


if __name__ == "__main__":
    main()
