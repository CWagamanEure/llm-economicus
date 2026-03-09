#!/usr/bin/env python3
"""Batch-level prompt similarity validator.

Flags clusters where prompts in the same family/subtype/style/frame are still
too close in wording/structure or dominated by one phrasing shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
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


def _normalize_prompt(text: str) -> str:
    compact = " ".join(text.lower().split())
    compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
    compact = re.sub(r"'[^']*'", "<quoted>", compact)
    compact = re.sub(r"\b[ht]{4,}\b", "<seq>", compact)
    return compact


def _prompt_skeleton(text: str) -> str:
    skeleton = _normalize_prompt(text)
    skeleton = re.sub(
        (
            r"\b("
            r"medical|patient|condition|test|fraud|transaction|candidate|hiring|"
            r"security|threat|market|trading|hospital|births|girls|fund|stocks|"
            r"factory|batch|analyst|weather|startup|kpi|roulette|basketball|"
            r"coin|wheel|tape|scoreboard|forecast|projection|board|deck|meteorology"
            r")\b"
        ),
        "<ctx>",
        skeleton,
    )
    return " ".join(skeleton.split())


def _opening_signature(text: str) -> str:
    tokens = re.findall(r"[a-z0-9_]+", _normalize_prompt(text))
    return " ".join(tokens[:4]) if tokens else ""


def _pairwise_scores(values: list[str], normalizer: callable) -> list[float]:
    normalized = [normalizer(v) for v in values]
    scores: list[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a = normalized[i]
            b = normalized[j]
            if isinstance(a, set):
                scores.append(len(a & b) / max(1, len(a | b)))
            else:
                scores.append(SequenceMatcher(None, a, b).ratio())
    return scores


def _lexical_scores(prompts: list[str]) -> list[float]:
    return _pairwise_scores(prompts, lambda v: set(re.findall(r"[a-z0-9_]+", v.lower())))


def _normalized_scores(prompts: list[str]) -> list[float]:
    return _pairwise_scores(prompts, _normalize_prompt)


def _structural_scores(prompts: list[str]) -> list[float]:
    return _pairwise_scores(prompts, _prompt_skeleton)


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


def _bayes_reliability_pattern(prompt: str) -> str:
    lower = prompt.lower()
    checks = (
        ("conditional_signal_probability", "conditional signal probability is"),
        ("seen_with_probability", "is seen with probability"),
        ("signal_occurs_with_probability", "signal occurs with probability"),
        ("shows_up_with_probability", "shows up with probability"),
        ("if_then_appears_probability", "if " and "appears with probability"),
        ("chance_of_signal", "chance of this signal is"),
        ("result_occurs_probability", "this result occurs with probability"),
        ("result_appears_probability", "this result appears with probability"),
    )
    for key, marker in checks:
        if marker in lower:
            return key
    return "other"


def _conjunction_shell_pattern(prompt: str) -> str:
    lower = prompt.lower()
    shells = (
        "which statement is more likely",
        "choose the more likely statement",
        "select the more likely statement",
        "which line is likelier",
        "which line feels more fitting",
        "which statement feels more representative",
        "compare directly",
        "mark the one that is more likely",
    )
    for shell in shells:
        if shell in lower:
            return shell
    return "other"


def _gambler_option_pattern(prompt: str) -> str:
    option_lines = [
        line.strip().lower()
        for line in prompt.splitlines()
        if line.strip().lower().startswith("- choose_")
    ]
    if not option_lines:
        return "none"
    joined = " | ".join(option_lines)
    joined = re.sub(r"\b(make|miss|up|down|red|black|heads|tails)\b", "<outcome>", joined)
    joined = re.sub(r"\bshot|session|spin|flip\b", "<trial>", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def _overprecision_shell_pattern(prompt: str) -> str:
    lower = prompt.lower()
    shells = (
        "which interval is more likely to contain",
        "which range is more likely to contain",
        "which range is more likely to include",
        "which interval has higher containment probability",
        "which interval gives better odds of containing",
        "which range seems likelier to contain",
    )
    for shell in shells:
        if shell in lower:
            return shell
    return "other"


def _core_pattern(task_family: str, subtype: str, prompt: str) -> str:
    if task_family == "bayesian_signal_extraction":
        return _bayes_reliability_pattern(prompt)
    if task_family == "belief_bias_judgment" and subtype == "conjunction_fallacy":
        return _conjunction_shell_pattern(prompt)
    if task_family == "belief_bias_judgment" and subtype == "gambler_fallacy":
        return _gambler_option_pattern(prompt)
    if task_family == "belief_bias_judgment" and subtype == "overprecision_calibration":
        return _overprecision_shell_pattern(prompt)
    return "generic"


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
    return prompt


def _renderer_hint(family: FamilyConfig, subtype: str) -> str:
    gen = family.generator_cls(seed=family.seed, prompt_style="default")
    method = gen._prompt_renderer_for_subtype(subtype)
    method_name = getattr(method, "__name__", str(method))
    module = family.generator_cls.__module__
    return f"{module}.{method_name}"


def run_validator(
    *,
    num_problems: int,
    max_frames: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        problems = _sample_underlying_problems(family, num_problems)
        if not problems:
            continue
        probe_gen = family.generator_cls(seed=family.seed, prompt_style="default")
        for problem in problems:
            subtype = problem["task_subtype"]
            frames = _frame_candidates(probe_gen, subtype, max_frames)
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
                    rows.append(
                        {
                            "family": family.name,
                            "subtype": subtype,
                            "regime": regime,
                            "frame": frame,
                            "fingerprint": problem["fingerprint"],
                            "prompt": prompt,
                            "opening": _opening_signature(prompt),
                            "pattern": _core_pattern(family.name, subtype, prompt),
                            "renderer_hint": _renderer_hint(family, subtype),
                        }
                    )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["subtype"], row["regime"], row["frame"])].append(row)

    flagged: list[dict[str, Any]] = []
    for key, items in grouped.items():
        if len(items) < 3:
            continue
        prompts = [item["prompt"] for item in items]
        lexical = _lexical_scores(prompts)
        norm_sim = _normalized_scores(prompts)
        struct = _structural_scores(prompts)
        opening_counts = Counter(item["opening"] for item in items if item["opening"])
        pattern_counts = Counter(item["pattern"] for item in items if item["pattern"] != "generic")
        dominant_opening = opening_counts.most_common(1)[0] if opening_counts else ("", 0)
        dominant_pattern = pattern_counts.most_common(1)[0] if pattern_counts else ("", 0)

        issues: list[str] = []
        if lexical and norm_sim and struct:
            if (
                (sum(struct) / len(struct) >= 0.93 and sum(lexical) / len(lexical) >= 0.58)
                or max(struct) >= 0.97
            ):
                issues.append("noun-substitution-like structural similarity")
        if dominant_opening[1] >= max(3, int(0.55 * len(items))):
            issues.append("repeated leading phrase dominates cluster")
        if dominant_pattern[1] >= max(3, int(0.65 * len(items))):
            issues.append("single core wording pattern dominates cluster")

        if issues:
            flagged.append(
                {
                    "key": key,
                    "size": len(items),
                    "mean_lexical_jaccard": round(sum(lexical) / len(lexical), 4)
                    if lexical
                    else 0.0,
                    "mean_normalized_similarity": round(sum(norm_sim) / len(norm_sim), 4)
                    if norm_sim
                    else 0.0,
                    "mean_structural_similarity": round(sum(struct) / len(struct), 4)
                    if struct
                    else 0.0,
                    "dominant_opening": dominant_opening,
                    "dominant_pattern": dominant_pattern,
                    "issues": issues,
                    "renderer_hint": items[0]["renderer_hint"],
                }
            )

    # Regime-cosmetic check: same family/subtype/frame/fingerprint across regimes.
    by_problem_key: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        by_problem_key[(row["family"], row["subtype"], row["frame"], row["fingerprint"])][
            row["regime"]
        ] = row["prompt"]

    regime_flagged: list[dict[str, Any]] = []
    by_cluster: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (family, subtype, frame, _fp), prompt_map in by_problem_key.items():
        norm = prompt_map.get("normative_explicit")
        neutral = prompt_map.get("neutral_realistic")
        if not norm or not neutral:
            continue
        n1 = " ".join(norm.lower().split())
        n2 = " ".join(neutral.lower().split())
        n1 = re.sub(r"^(decision memo|case brief|quick brief|review note)\s*:?\s*", "", n1)
        n2 = re.sub(r"^(decision memo|case brief|quick brief|review note)\s*:?\s*", "", n2)
        sim = SequenceMatcher(None, _prompt_skeleton(n1), _prompt_skeleton(n2)).ratio()
        by_cluster[(family, subtype, frame)].append(sim)
    for key, sims in by_cluster.items():
        if not sims:
            continue
        mean_sim = sum(sims) / len(sims)
        if mean_sim >= 0.9:
            family_name, subtype, _frame = key
            family = next(f for f in FAMILIES if f.name == family_name)
            regime_flagged.append(
                {
                    "key": key,
                    "mean_structural_similarity_normative_vs_neutral": round(mean_sim, 4),
                    "issue": "regime differences mostly cosmetic",
                    "renderer_hint": _renderer_hint(family, subtype),
                }
            )

    return flagged, regime_flagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate batch-level prompt similarity.")
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Max frame variants per subtype; 0 means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_frames == 0 else args.max_frames
    flagged, regime_flagged = run_validator(
        num_problems=args.num_problems,
        max_frames=max_frames,
    )

    print("=" * 96)
    print("FLAGGED CLUSTERS")
    print("=" * 96)
    if not flagged:
        print("No cluster-level similarity flags.")
    for item in flagged:
        family, subtype, regime, frame = item["key"]
        print(
            f"- {family} | {subtype} | {regime} | {frame} "
            f"(n={item['size']}, lex={item['mean_lexical_jaccard']}, "
            f"norm={item['mean_normalized_similarity']}, "
            f"struct={item['mean_structural_similarity']})"
        )
        print(f"  issues: {', '.join(item['issues'])}")
        print(
            f"  dominant opening: {item['dominant_opening'][0]!r} "
            f"({item['dominant_opening'][1]}/{item['size']})"
        )
        print(
            f"  dominant core pattern: {item['dominant_pattern'][0]!r} "
            f"({item['dominant_pattern'][1]}/{item['size']})"
        )
        print(f"  likely renderer: {item['renderer_hint']}")

    print("\n" + "=" * 96)
    print("REGIME COSMETIC FLAGS")
    print("=" * 96)
    if not regime_flagged:
        print("No regime-cosmetic clusters flagged.")
    for item in regime_flagged:
        family, subtype, frame = item["key"]
        print(
            f"- {family} | {subtype} | frame={frame} "
            f"(mean structural sim norm-vs-neutral="
            f"{item['mean_structural_similarity_normative_vs_neutral']})"
        )
        print(f"  issue: {item['issue']}")
        print(f"  likely renderer: {item['renderer_hint']}")


if __name__ == "__main__":
    main()
