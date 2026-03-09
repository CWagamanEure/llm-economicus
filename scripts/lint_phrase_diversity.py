#!/usr/bin/env python3
"""Phrase-diversity lint for prompt generation batches.

Detects overused rhetorical stems within each (task_family, task_subtype,
prompt_style_regime) cluster.
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

REGIMES: tuple[str, ...] = (
    "normative_explicit",
    "neutral_realistic",
    "bias_eliciting",
)

CONTEXT_NOUNS_RE = re.compile(
    (
        r"\b("
        r"medical|patient|condition|test|fraud|transaction|candidate|hiring|"
        r"security|threat|market|trading|hospital|births|girls|fund|stocks|"
        r"factory|batch|analyst|weather|startup|kpi|roulette|basketball|"
        r"coin|wheel|tape|scoreboard|forecast|projection|board|deck|meteorology|"
        r"community|teacher|organizer|campaign|researcher|scientist|sales"
        r")\b"
    ),
    flags=re.IGNORECASE,
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


@dataclass(frozen=True)
class LintConfig:
    num_problems: int
    max_frames: int | None
    stem_min_count: int
    stem_min_rate: float
    prefix_min_count: int
    opening_skeleton_min_count: int
    top_k: int


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
    while len(sampled) < count and draws < count * 40:
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
        return frames[:max_frames]
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
    prompt = generator._collapse_stacked_prompt_wrappers(prompt=prompt)
    return " ".join(prompt.split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_']+", text.lower())


def _opening_ngram(text: str, n: int) -> str:
    toks = _tokens(text)
    if len(toks) < n:
        return ""
    return " ".join(toks[:n])


def _prefix_before_colon(text: str) -> str:
    head = text.split("\n", 1)[0]
    m = re.match(r"^\s*([A-Za-z][A-Za-z ()/\-]{1,80})\s*:\s*", head)
    if not m:
        return ""
    prefix = " ".join(m.group(1).lower().split())
    words = prefix.split()
    if len(words) > 10:
        return ""
    return prefix


def _opening_text(text: str) -> str:
    # Opening rhetorical chunk: before first sentence boundary, capped length.
    first = re.split(r"[.!?]", text, maxsplit=1)[0]
    return " ".join(first.split())[:180]


def _normalize_opening_for_skeleton(text: str) -> str:
    s = _opening_text(text).lower()
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", s)
    s = re.sub(r"'[^']*'", "<quoted>", s)
    s = CONTEXT_NOUNS_RE.sub("<ctx>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _collect_rows(cfg: LintConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        problems = _sample_underlying_problems(family, cfg.num_problems)
        if not problems:
            continue
        probe = family.generator_cls(seed=family.seed, prompt_style="default")
        for problem in problems:
            subtype = problem["task_subtype"]
            frames = _frame_candidates(probe, subtype, cfg.max_frames)
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
                            "opening_bigram": _opening_ngram(prompt, 2),
                            "opening_trigram": _opening_ngram(prompt, 3),
                            "meta_prefix": _prefix_before_colon(prompt),
                            "opening_skeleton": _normalize_opening_for_skeleton(prompt),
                        }
                    )
    return rows


def _flag_counter(
    counter: Counter[str],
    *,
    total: int,
    min_count: int,
    min_rate: float,
) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for stem, count in counter.most_common():
        if not stem:
            continue
        rate = count / max(1, total)
        if count >= min_count and rate >= min_rate:
            flagged.append({"stem": stem, "count": count, "rate": round(rate, 4)})
    return flagged


def run_lint(cfg: LintConfig) -> dict[str, Any]:
    rows = _collect_rows(cfg)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["subtype"], row["regime"])].append(row)

    clusters: list[dict[str, Any]] = []
    global_bigram = Counter(row["opening_bigram"] for row in rows if row["opening_bigram"])
    global_trigram = Counter(row["opening_trigram"] for row in rows if row["opening_trigram"])
    global_prefix = Counter(row["meta_prefix"] for row in rows if row["meta_prefix"])
    global_skeleton = Counter(row["opening_skeleton"] for row in rows if row["opening_skeleton"])

    for key, items in grouped.items():
        family, subtype, regime = key
        n = len(items)
        if n == 0:
            continue
        bigrams = Counter(x["opening_bigram"] for x in items if x["opening_bigram"])
        trigrams = Counter(x["opening_trigram"] for x in items if x["opening_trigram"])
        prefixes = Counter(x["meta_prefix"] for x in items if x["meta_prefix"])
        skeletons = Counter(x["opening_skeleton"] for x in items if x["opening_skeleton"])

        flagged_bi = _flag_counter(
            bigrams,
            total=n,
            min_count=cfg.stem_min_count,
            min_rate=cfg.stem_min_rate,
        )
        flagged_tri = _flag_counter(
            trigrams,
            total=n,
            min_count=cfg.stem_min_count,
            min_rate=cfg.stem_min_rate,
        )
        flagged_prefix = _flag_counter(
            prefixes,
            total=n,
            min_count=cfg.prefix_min_count,
            min_rate=0.0,
        )
        flagged_skeleton = _flag_counter(
            skeletons,
            total=n,
            min_count=cfg.opening_skeleton_min_count,
            min_rate=cfg.stem_min_rate,
        )

        if flagged_bi or flagged_tri or flagged_prefix or flagged_skeleton:
            severity = 0.0
            severity += sum(item["rate"] for item in flagged_bi[:3])
            severity += sum(item["rate"] for item in flagged_tri[:3])
            severity += sum(item["rate"] for item in flagged_skeleton[:3])
            severity += 0.1 * sum(item["count"] for item in flagged_prefix[:3])
            clusters.append(
                {
                    "key": key,
                    "size": n,
                    "flagged_opening_bigrams": flagged_bi[: cfg.top_k],
                    "flagged_opening_trigrams": flagged_tri[: cfg.top_k],
                    "flagged_meta_prefixes": flagged_prefix[: cfg.top_k],
                    "flagged_opening_skeletons": flagged_skeleton[: cfg.top_k],
                    "severity": round(severity, 4),
                }
            )

    clusters.sort(key=lambda x: (-x["severity"], -x["size"], x["key"]))

    return {
        "rows": rows,
        "clusters": clusters,
        "global_top_opening_bigrams": global_bigram.most_common(cfg.top_k),
        "global_top_opening_trigrams": global_trigram.most_common(cfg.top_k),
        "global_top_meta_prefixes": global_prefix.most_common(cfg.top_k),
        "global_top_opening_skeletons": global_skeleton.most_common(cfg.top_k),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint repeated opening phrase stems in prompts.")
    parser.add_argument("--num-problems", type=int, default=12)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3,
        help="Max frame variants per subtype; 0 means all.",
    )
    parser.add_argument("--stem-min-count", type=int, default=3)
    parser.add_argument("--stem-min-rate", type=float, default=0.34)
    parser.add_argument("--prefix-min-count", type=int, default=3)
    parser.add_argument("--opening-skeleton-min-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LintConfig(
        num_problems=args.num_problems,
        max_frames=None if args.max_frames == 0 else args.max_frames,
        stem_min_count=args.stem_min_count,
        stem_min_rate=args.stem_min_rate,
        prefix_min_count=args.prefix_min_count,
        opening_skeleton_min_count=args.opening_skeleton_min_count,
        top_k=args.top_k,
    )
    report = run_lint(cfg)

    print("=" * 100)
    print("PHRASE DIVERSITY LINT")
    print("=" * 100)
    print(f"total rendered prompts: {len(report['rows'])}")

    print("\nGLOBAL TOP OPENING BIGRAMS")
    for stem, count in report["global_top_opening_bigrams"]:
        print(f"- {stem!r}: {count}")

    print("\nGLOBAL TOP OPENING TRIGRAMS")
    for stem, count in report["global_top_opening_trigrams"]:
        print(f"- {stem!r}: {count}")

    print("\nGLOBAL TOP META PREFIXES")
    if report["global_top_meta_prefixes"]:
        for stem, count in report["global_top_meta_prefixes"]:
            print(f"- {stem!r}: {count}")
    else:
        print("- none")

    print("\nGLOBAL TOP OPENING SKELETONS")
    for stem, count in report["global_top_opening_skeletons"]:
        print(f"- {stem!r}: {count}")

    print("\nWORST OFFENDING CLUSTERS")
    clusters = report["clusters"][: args.max_clusters]
    if not clusters:
        print("- none")
        return
    for cluster in clusters:
        family, subtype, regime = cluster["key"]
        print("-" * 100)
        print(
            f"{family} | {subtype} | {regime} | n={cluster['size']} | "
            f"severity={cluster['severity']}"
        )
        if cluster["flagged_opening_bigrams"]:
            print("  opening bigrams:")
            for item in cluster["flagged_opening_bigrams"]:
                print(f"    - {item['stem']!r}: {item['count']} ({item['rate']})")
        if cluster["flagged_opening_trigrams"]:
            print("  opening trigrams:")
            for item in cluster["flagged_opening_trigrams"]:
                print(f"    - {item['stem']!r}: {item['count']} ({item['rate']})")
        if cluster["flagged_meta_prefixes"]:
            print("  meta prefixes before colon:")
            for item in cluster["flagged_meta_prefixes"]:
                print(f"    - {item['stem']!r}: {item['count']}")
        if cluster["flagged_opening_skeletons"]:
            print("  near-duplicate opening skeletons:")
            for item in cluster["flagged_opening_skeletons"]:
                print(f"    - {item['stem']!r}: {item['count']} ({item['rate']})")


if __name__ == "__main__":
    main()
