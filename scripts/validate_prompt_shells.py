#!/usr/bin/env python3
"""One-shell prompt validator.

Detects prompts that combine multiple rhetorical shells, repeated framing labels,
or duplicated framing nouns. Prints flagged prompts and shell counts by family/subtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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

REGIMES: tuple[str, ...] = ("normative_explicit", "neutral_realistic", "bias_eliciting")


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

SHELL_PATTERNS: dict[str, tuple[str, ...]] = {
    "scene_narrative": (
        r"\bscene\b",
        r"\bnarrative\b",
        r"\bfield snapshot\b",
        r"\breported scene\b",
        r"\bstory\b",
        r"\bmomentum\b",
    ),
    "memo_brief_note": (
        r"\bmemo\b",
        r"\bbrief\b",
        r"\bnote\b",
        r"\breview\b",
        r"\bcard\b",
        r"\bsummary\b",
        r"\bsnapshot\b",
    ),
    "worksheet_evaluation": (
        r"\bworksheet\b",
        r"\bevaluation\b",
        r"\bevaluate\b",
        r"\baudit\b",
        r"\branking prompt\b",
        r"\bformal evaluation\b",
    ),
    "profile_archetype": (
        r"\bprofile\b",
        r"\barchetype\b",
        r"\btype[- ]fit\b",
        r"\bcategory[- ]fit\b",
    ),
    "quick_call_first_impression": (
        r"\bfirst glance\b",
        r"\bfirst impression\b",
        r"\bfirst-pass\b",
        r"\bgut-check\b",
        r"\bsnap\b",
        r"\bimmediate\b",
        r"\bon-the-spot\b",
        r"\bquick call\b",
        r"\btime pressure\b",
    ),
    "compute_compare_directive": (
        r"\bcompute then compare\b",
        r"\bcompare\b",
        r"\brank\b",
        r"\bdetermine\b",
        r"\bselect\b",
        r"\bevaluate\b",
    ),
}

FRAMING_NOUNS: tuple[str, ...] = (
    "note",
    "brief",
    "memo",
    "review",
    "card",
    "summary",
    "description",
    "snapshot",
)


def _problem_fingerprint(problem_spec: dict[str, Any]) -> str:
    canonical = json.dumps(problem_spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _sample_underlying_problems(family: FamilyConfig, count: int) -> list[dict[str, Any]]:
    generator = family.generator_cls(
        seed=family.seed,
        prompt_style="default",
        prompt_style_regime="neutral_realistic",
        prompt_frame_variant="auto",
    )
    sampled: list[dict[str, Any]] = []
    seen: set[str] = set()
    draws = 0
    while len(sampled) < count and draws < count * 30:
        draws += 1
        dp = generator.generate()
        fp = _problem_fingerprint(dp.problem_spec)
        if fp in seen:
            continue
        seen.add(fp)
        sampled.append(
            {
                "task_family": dp.task_family,
                "task_subtype": dp.task_subtype,
                "problem_spec": dp.problem_spec,
                "fingerprint": fp,
            }
        )
    return sampled


def _frame_candidates(generator: Any, subtype: str, max_frames: int | None) -> list[str]:
    frames = list(generator._frame_candidates_for_subtype(subtype))
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def _render_prompt(generator: Any, *, subtype: str, problem_spec: dict[str, Any]) -> str:
    renderer = generator._prompt_renderer_for_subtype(subtype)
    prompt = renderer(
        problem_spec=problem_spec,
        style="default",
        prompt_style_regime=generator.prompt_style_regime,
        prompt_frame_variant=generator.prompt_frame_variant,
    )
    prompt = generator._apply_prompt_frame_variant(
        prompt=prompt,
        frame_variant=generator.prompt_frame_variant,
        task_subtype=subtype,
    )
    if hasattr(generator, "_collapse_stacked_prompt_wrappers"):
        prompt = generator._collapse_stacked_prompt_wrappers(prompt=prompt)
    return prompt


def _detect_shells(prompt: str) -> dict[str, int]:
    lower = prompt.lower()
    hits: dict[str, int] = {}
    for shell, patterns in SHELL_PATTERNS.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, lower))
        if count > 0:
            hits[shell] = count
    return hits


def _detect_repeated_meta_start(prompt: str) -> tuple[int, list[str]]:
    head = " ".join(prompt.strip().split())[:120]
    raw_labels = re.findall(r"(?:^|[.]\s+)([A-Za-z][A-Za-z ()/\-]{2,40}):", head)
    labels: list[str] = []
    for label in raw_labels:
        lower = label.lower()
        if any(token in lower for token in ("choose_", "act", "do_not_", "buy", "sell")):
            continue
        labels.append(label.strip())
    return len(labels), labels


def _detect_duplicated_framing_nouns(prompt: str) -> dict[str, int]:
    lower = prompt.lower()
    counts: dict[str, int] = {}
    for noun in FRAMING_NOUNS:
        hits = len(re.findall(rf"\b{re.escape(noun)}\b", lower))
        if hits:
            counts[noun] = hits
    return counts


def _suggest_shell(
    *,
    shells: dict[str, int],
    regime: str,
    frame: str,
) -> tuple[str, list[str]]:
    if not shells:
        return "none_detected", []
    preferred: list[str] = []
    if "vivid" in frame:
        preferred.append("scene_narrative")
    if "plain" in frame:
        preferred.append("memo_brief_note")
    if "profile" in frame or "archetype" in frame:
        preferred.append("profile_archetype")
    if regime == "normative_explicit":
        preferred.extend(["worksheet_evaluation", "compute_compare_directive"])
    elif regime == "bias_eliciting":
        preferred.extend(["quick_call_first_impression", "scene_narrative"])
    else:
        preferred.extend(["memo_brief_note", "profile_archetype"])

    keep = next((shell for shell in preferred if shell in shells), max(shells, key=shells.get))
    remove = [shell for shell in shells if shell != keep]
    return keep, remove


def run_validator(
    *,
    num_problems: int,
    max_frames: int | None,
    max_flagged: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        problems = _sample_underlying_problems(family, num_problems)
        if not problems:
            continue
        probe = family.generator_cls(seed=family.seed, prompt_style="default")
        for problem in problems:
            subtype = problem["task_subtype"]
            frames = _frame_candidates(probe, subtype, max_frames)
            for regime in REGIMES:
                for frame in frames:
                    gen = family.generator_cls(
                        seed=family.seed,
                        prompt_style="default",
                        prompt_style_regime=regime,
                        prompt_frame_variant=frame,
                    )
                    prompt = _render_prompt(
                        gen,
                        subtype=subtype,
                        problem_spec=problem["problem_spec"],
                    )
                    shells = _detect_shells(prompt)
                    meta_count, meta_labels = _detect_repeated_meta_start(prompt)
                    noun_counts = _detect_duplicated_framing_nouns(prompt)
                    duplicated_noun_total = sum(noun_counts.values())
                    duplicated_noun_unique = len(noun_counts)
                    issues: list[str] = []
                    if len(shells) > 1:
                        issues.append("multiple_primary_shells")
                    if meta_count >= 2:
                        issues.append("repeated_meta_labels_at_start")
                    if duplicated_noun_total >= 3 or duplicated_noun_unique >= 2:
                        issues.append("duplicated_framing_nouns")
                    keep, remove = _suggest_shell(shells=shells, regime=regime, frame=frame)

                    rows.append(
                        {
                            "family": family.name,
                            "subtype": subtype,
                            "regime": regime,
                            "frame": frame,
                            "fingerprint": problem["fingerprint"],
                            "prompt": prompt,
                            "shells": shells,
                            "meta_count": meta_count,
                            "meta_labels": meta_labels,
                            "noun_counts": noun_counts,
                            "issues": issues,
                            "keep_shell": keep,
                            "remove_shells": remove,
                        }
                    )

    flagged = [row for row in rows if row["issues"]]
    print("=" * 100)
    print("ONE-SHELL VALIDATOR RESULTS")
    print("=" * 100)
    print(f"total_prompts={len(rows)} flagged_prompts={len(flagged)}")

    if flagged:
        print("\nFlagged prompts:")
        for idx, row in enumerate(flagged[:max_flagged], start=1):
            print("-" * 100)
            print(
                f"{idx}. {row['family']} | {row['subtype']} | "
                f"{row['regime']} | {row['frame']} | fp={row['fingerprint']}"
            )
            print(f"issues={row['issues']}")
            print(f"shells={row['shells']}")
            print(f"meta_labels={row['meta_labels']}")
            print(f"framing_nouns={row['noun_counts']}")
            print(
                f"suggestion=retain '{row['keep_shell']}' "
                f"remove {row['remove_shells']}"
            )
            print("prompt:")
            print(row["prompt"])
        if len(flagged) > max_flagged:
            print(f"... ({len(flagged) - max_flagged} more flagged prompts not shown)")

    shell_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["family"], row["subtype"])
        if row["shells"]:
            for shell in row["shells"]:
                shell_counts[key][shell] += 1
        else:
            shell_counts[key]["none_detected"] += 1

    print("\n" + "=" * 100)
    print("SHELL COUNTS BY FAMILY/SUBTYPE")
    print("=" * 100)
    for (family, subtype), counter in sorted(shell_counts.items()):
        summary = ", ".join(f"{k}={v}" for k, v in counter.most_common())
        print(f"{family} | {subtype} -> {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rhetorical shell mixing in prompts.")
    parser.add_argument(
        "--num-problems",
        type=int,
        default=8,
        help="Number of underlying problems sampled per family.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Maximum number of frame variants per subtype.",
    )
    parser.add_argument(
        "--max-flagged",
        type=int,
        default=25,
        help="Maximum number of flagged prompts to print.",
    )
    args = parser.parse_args()
    run_validator(
        num_problems=max(1, args.num_problems),
        max_frames=max(1, args.max_frames) if args.max_frames is not None else None,
        max_flagged=max(1, args.max_flagged),
    )


if __name__ == "__main__":
    main()
