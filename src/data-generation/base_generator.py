import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections import Counter, deque
from difflib import SequenceMatcher
from typing import Any

from schema import DataPoint, Metadata

PROMPT_FORBIDDEN_PHRASES_BY_REGIME: dict[str, tuple[str, ...]] = {
    "neutral_realistic": (
        "expected value",
        "average payoff",
        "over many repeats",
        "many repeats",
        "bayes rule",
        "use bayes",
        "independent process",
        "probability axioms",
        "same process on the next trial",
        "same process on the very next trial",
        "financially better",
    ),
    "bias_eliciting": (
        "expected value",
        "average payoff",
        "over many repeats",
        "many repeats",
        "bayes rule",
        "use bayes",
        "independent process",
        "probability axioms",
        "same process on the next trial",
        "same process on the very next trial",
        "financially better",
    ),
}

STYLE_MARKERS_BY_REGIME: dict[str, tuple[str, ...]] = {
    "normative_explicit": (
        "compare",
        "evaluate",
        "determine",
        "probability",
        "posterior",
        "apply",
        "over many similar rounds",
        "discount",
    ),
    "neutral_realistic": (
        "case",
        "review",
        "received",
        "saw",
        "deciding now",
        "which option",
        "which action",
        "which state",
        "which range",
        "which interval",
        "you can",
        "which option is better",
        "which state now",
    ),
}

STACKED_WRAPPER_PREFIX_LABELS: tuple[str, ...] = (
    "decision brief",
    "case review",
    "scene description",
    "narrative snapshot",
    "case summary",
    "quick brief",
    "review note",
    "operator note",
    "decision memo",
    "case memo",
    "scenario card",
    "dashboard alert",
    "comparison sheet",
    "field report",
    "analyst summary",
    "analyst disagreement brief",
    "compute then compare",
    "model-selection card",
    "decision row",
)


class BaseGenerator(ABC):
    """
    This is the base data generator
    """

    @abstractmethod
    def generate(self) -> DataPoint:
        raise NotImplementedError()

    def build_metadata(
        self,
        seed: int,
        difficulty_metrics: dict[str, Any],
        version: str = "v1",
        sample_index: int | None = None,
        dataset_role: str = "normative_training",
        requested_prompt_style: str | None = None,
        resolved_prompt_style: str | None = None,
        prompt_style_regime: str | None = None,
        prompt_frame_variant: str | None = None,
        prompt_has_action_labels: bool = True,
        example_fingerprint: str | None = None,
        tie_threshold: float | None = None,
        semantic_context: str | None = None,
        conjunction_render_mode: str | None = None,
        representativeness_strength: str | None = None,
        streak_domain: str | None = None,
    ) -> Metadata:
        return Metadata(
            generator_name=self.__class__.__name__,
            version=version,
            seed=seed,
            dataset_role=dataset_role,
            requested_prompt_style=requested_prompt_style,
            resolved_prompt_style=resolved_prompt_style,
            prompt_style_regime=prompt_style_regime,
            prompt_frame_variant=prompt_frame_variant,
            prompt_has_action_labels=prompt_has_action_labels,
            example_fingerprint=example_fingerprint,
            tie_threshold=tie_threshold,
            sample_index=sample_index,
            semantic_context=semantic_context,
            conjunction_render_mode=conjunction_render_mode,
            representativeness_strength=representativeness_strength,
            streak_domain=streak_domain,
            difficulty_metrics=difficulty_metrics,
        )

    def assert_prompt_regime_no_leakage(
        self,
        *,
        prompt: str,
        prompt_style_regime: str,
    ) -> None:
        forbidden = PROMPT_FORBIDDEN_PHRASES_BY_REGIME.get(prompt_style_regime, ())
        lower_prompt = prompt.lower()
        for phrase in forbidden:
            if phrase in lower_prompt:
                raise ValueError(
                    "Prompt contains normative leakage phrase "
                    f"'{phrase}' for regime '{prompt_style_regime}'."
                )

    def _prompt_qa_failure(self, *, code: str, detail: str) -> dict[str, str]:
        return {"code": code, "detail": detail}

    def _prompt_qa_sentence_windows(self, prompt: str) -> list[str]:
        windows = re.split(r"[.!?;:\n]+", prompt)
        return [window.strip().lower() for window in windows if window.strip()]

    def _prompt_qa_vs_pair_is_anchored(self, sentence: str) -> bool:
        anchor_tokens = (
            "when ",
            "among ",
            "given ",
            "probability",
            "chance",
            "if ",
            "p(",
            "observed",
            "signal",
            "baseline",
            "base rate",
            "prior",
            "payoff",
            "returns",
        )
        return any(token in sentence for token in anchor_tokens)

    def _prompt_qa_generic_failures(self, *, prompt: str) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        lower = prompt.lower()
        malformed_patterns = (
            (r"\bin the [^,.!?;\n]{1,60} is [^,.!?;\n]{1,60} cases\b", "broken_case_phrase"),
            (r"\bfor among\b", "broken_preposition_phrase"),
            (r"\ban down\b", "malformed_article"),
            (r"\ba up\b", "malformed_article"),
        )
        for pattern, code in malformed_patterns:
            if re.search(pattern, lower):
                failures.append(
                    self._prompt_qa_failure(
                        code=code,
                        detail=f"Matched malformed pattern '{pattern}'.",
                    )
                )
        vs_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*vs\s*\d+(?:\.\d+)?\b")
        windows = self._prompt_qa_sentence_windows(prompt)
        for match in vs_pattern.finditer(lower):
            snippet = match.group(0)
            containing_sentence = next(
                (window for window in windows if snippet in window),
                lower,
            )
            if not self._prompt_qa_vs_pair_is_anchored(containing_sentence):
                failures.append(
                    self._prompt_qa_failure(
                        code="unanchored_vs_shorthand",
                        detail=(
                            "Found compressed 'X vs Y' numeric shorthand without explicit "
                            f"conditional anchor: '{snippet}'."
                        ),
                    )
                )
        interpolation_fragments = (
            "signal split=",
            "signal profile=",
            "cue profile=",
            "signal performance is",
        )
        if any(fragment in lower for fragment in interpolation_fragments):
            failures.append(
                self._prompt_qa_failure(
                    code="compressed_interpretability_shorthand",
                    detail="Prompt contains compressed reliability shorthand fragment.",
                )
            )
        slash_pair_pattern = re.compile(r"\b(?:0?\.\d+)\s*/\s*(?:0?\.\d+)\b")
        if slash_pair_pattern.search(lower):
            failures.append(
                self._prompt_qa_failure(
                    code="raw_slash_numeric_shorthand",
                    detail=(
                        "Prompt contains raw slash shorthand (e.g., '0.81/0.25') "
                        "instead of explicit conditional interpretation."
                    ),
                )
            )
        invalid_prior_phrasing = re.compile(
            r"\b(?:base rate|background share|background rate|baseline share)\b"
            r"[^;\n]{0,120}\bamong\b",
            flags=re.IGNORECASE,
        )
        invalid_prior_match = invalid_prior_phrasing.search(lower)
        # Ignore cross-sentence matches caused by nearby "among ..." in a later sentence.
        if invalid_prior_match and ". " not in invalid_prior_match.group(0):
            failures.append(
                self._prompt_qa_failure(
                    code="invalid_base_rate_among_positive_class",
                    detail=(
                        "Prior/base-rate phrasing is semantically invalid: base/background "
                        "rate should describe prevalence in the overall case population."
                    ),
                )
            )
        framing_prefixes = self._extract_framing_prefix_labels(prompt)
        if len(framing_prefixes) >= 2:
            failures.append(
                self._prompt_qa_failure(
                    code="stacked_framing_prefixes",
                    detail=(
                        "Prompt appears to contain stacked framing wrappers: "
                        f"{framing_prefixes}."
                    ),
                )
            )
        return failures

    def _extract_framing_prefix_labels(self, prompt: str) -> list[str]:
        lower = prompt.lower()
        found: list[str] = []
        for label in STACKED_WRAPPER_PREFIX_LABELS:
            pattern = re.compile(rf"\b{re.escape(label)}\b\s*:", flags=re.IGNORECASE)
            if pattern.search(lower):
                found.append(label)
        return found

    def _count_stacked_framing_prefixes(self, prompt: str) -> int:
        return len(self._extract_framing_prefix_labels(prompt))

    def _collapse_stacked_prompt_wrappers(self, *, prompt: str) -> str:
        """
        Remove repeated wrapper labels such as "Decision brief:" / "Case review:"
        so each prompt has a single coherent rhetorical shell.
        """
        collapsed = prompt
        for label in STACKED_WRAPPER_PREFIX_LABELS:
            pattern = re.compile(rf"\b{re.escape(label)}\b\s*:\s*", flags=re.IGNORECASE)
            collapsed = pattern.sub("", collapsed)

        # Normalize whitespace introduced by wrapper removal.
        collapsed = re.sub(r"\s{2,}", " ", collapsed)
        collapsed = re.sub(r"\s+([,.!?;:])", r"\1", collapsed)
        return collapsed.strip()

    def _ensure_prompt_diversity_state(self) -> None:
        if hasattr(self, "_prompt_diversity_state"):
            return
        if not hasattr(self, "_enable_prompt_diversity_balancing"):
            self._enable_prompt_diversity_balancing = True
        self._prompt_diversity_state = {
            "template_usage": Counter(),
            "recent_openings": deque(maxlen=24),
            "recent_prompts": deque(maxlen=64),
            "selection_events": deque(maxlen=256),
        }

    def _normalize_prompt_text(self, text: str) -> str:
        compact = " ".join(text.lower().split())
        compact = re.sub(r"\b\d+(\.\d+)?\b", "<n>", compact)
        return compact

    def normalize_prompt_for_style_distance(self, text: str) -> str:
        """Normalize prompt text for style-distance checks.

        This strips common leading wrapper headers so checks fail when two
        prompts only differ by lightweight framing labels.
        """
        normalized = " ".join(text.lower().split())
        normalized = re.sub(
            r"^(?:quick brief|review note|decision brief|operator note|"
            r"decision memo|case memo|scenario|case brief|dashboard alert)"
            r"\s*:?\s*",
            "",
            normalized,
        )
        normalized = re.sub(r"\b\d+(\.\d+)?\b", "<n>", normalized)
        return normalized

    def style_distance_lint(
        self,
        *,
        normative_prompt: str,
        neutral_prompt: str,
    ) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        norm_markers = STYLE_MARKERS_BY_REGIME["normative_explicit"]
        neutral_markers = STYLE_MARKERS_BY_REGIME["neutral_realistic"]
        normative_lower = normative_prompt.lower()
        neutral_lower = neutral_prompt.lower()

        if not any(marker in normative_lower for marker in norm_markers):
            failures.append(
                self._prompt_qa_failure(
                    code="style_marker_missing_normative",
                    detail=(
                        "Normative prompt is missing explicit analytic style markers "
                        "(e.g., compare/evaluate/determine/probability/posterior)."
                    ),
                )
            )
        if not any(marker in neutral_lower for marker in neutral_markers):
            failures.append(
                self._prompt_qa_failure(
                    code="style_marker_missing_neutral",
                    detail=(
                        "Neutral prompt is missing practical case-style markers "
                        "(e.g., case/review/received/saw/deciding now)."
                    ),
                )
            )

        norm_norm = self.normalize_prompt_for_style_distance(normative_prompt)
        neut_norm = self.normalize_prompt_for_style_distance(neutral_prompt)
        similarity = SequenceMatcher(None, norm_norm, neut_norm).ratio()
        norm_tokens = set(re.findall(r"[a-z0-9_]+", norm_norm))
        neut_tokens = set(re.findall(r"[a-z0-9_]+", neut_norm))
        jaccard = len(norm_tokens & neut_tokens) / max(1, len(norm_tokens | neut_tokens))
        if similarity >= 0.95 and jaccard >= 0.88:
            failures.append(
                self._prompt_qa_failure(
                    code="style_distance_too_small",
                    detail=(
                        "Normative and neutral prompts are too similar after header "
                        f"normalization (similarity={similarity:.4f}, jaccard={jaccard:.4f})."
                    ),
                )
            )
        return failures

    def _opening_signature(self, text: str) -> str:
        normalized = self._normalize_prompt_text(text)
        tokens = re.findall(r"[a-z0-9_]+", normalized)
        return " ".join(tokens[:3]) if tokens else ""

    def _discourse_marker(self, text: str) -> str:
        normalized = self._normalize_prompt_text(text)
        markers = (
            "case memo",
            "decision memo",
            "decision note",
            "review note",
            "quick brief",
            "analyst summary",
            "dashboard alert",
            "operator note",
            "handoff note",
            "scenario card",
            "comparison sheet",
            "field report",
        )
        for marker in markers:
            if marker in normalized:
                return marker
        return ""

    def _hash_template_rank(
        self,
        *,
        task_subtype: str,
        frame_variant: str,
        tier: str,
        problem_spec: dict[str, Any],
        template_idx: int,
    ) -> float:
        payload = {
            "task_subtype": task_subtype,
            "frame_variant": frame_variant,
            "tier": tier,
            "problem_spec": problem_spec,
            "template_idx": template_idx,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") / float(2**64)

    def select_template_index_balanced(
        self,
        *,
        task_subtype: str,
        frame_variant: str,
        tier: str,
        problem_spec: dict[str, Any],
        templates: list[str] | tuple[str, ...],
    ) -> int:
        """Pick a template with deterministic baseline ranking plus anti-repetition penalties."""
        self._ensure_prompt_diversity_state()
        state = self._prompt_diversity_state
        template_usage: Counter[str] = state["template_usage"]
        recent_openings: deque[str] = state["recent_openings"]
        recent_prompts: deque[str] = state["recent_prompts"]

        if not templates:
            return 0

        family_key = f"{self.__class__.__name__}:{task_subtype}:{frame_variant}:{tier}"

        if not self._enable_prompt_diversity_balancing:
            payload = {
                "task_subtype": task_subtype,
                "frame_variant": frame_variant,
                "tier": tier,
                "problem_spec": problem_spec,
                "num_templates": len(templates),
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode("utf-8")).digest()
            chosen_idx = digest[0] % max(1, len(templates))
            opening = self._opening_signature(str(templates[chosen_idx]))
            marker = self._discourse_marker(str(templates[chosen_idx]))
            template_key = f"{family_key}:{chosen_idx}"
            template_usage[template_key] += 1
            if opening:
                recent_openings.append(opening)
            recent_prompts.append(str(templates[chosen_idx]))
            state["selection_events"].append(
                {
                    "family_key": family_key,
                    "template_idx": chosen_idx,
                    "opening": opening,
                    "marker": marker,
                }
            )
            return chosen_idx

        usage_values = [template_usage[f"{family_key}:{i}"] for i in range(len(templates))]
        mean_usage = sum(usage_values) / max(1, len(usage_values))

        candidates: list[tuple[float, int, str, str]] = []
        for idx, template in enumerate(templates):
            opening = self._opening_signature(template)
            marker = self._discourse_marker(template)
            base_rank = self._hash_template_rank(
                task_subtype=task_subtype,
                frame_variant=frame_variant,
                tier=tier,
                problem_spec=problem_spec,
                template_idx=idx,
            )
            usage_penalty = 0.18 * float(template_usage[f"{family_key}:{idx}"])
            overuse_penalty = 0.0
            if template_usage[f"{family_key}:{idx}"] > mean_usage + 0.5:
                overuse_penalty = 0.45
            opening_penalty = 0.0
            if opening and opening in recent_openings:
                opening_penalty = 0.35
            marker_penalty = 0.0
            if marker:
                marker_hits = sum(1 for p in recent_prompts if self._discourse_marker(p) == marker)
                marker_penalty = 0.08 * marker_hits
            total_score = (
                base_rank + usage_penalty + overuse_penalty + opening_penalty + marker_penalty
            )
            candidates.append((total_score, idx, opening, marker))

        # If alternatives exist, strongly avoid reusing a recent opening.
        non_repeated = [item for item in candidates if item[2] and item[2] not in recent_openings]
        if non_repeated:
            chosen = min(non_repeated, key=lambda x: x[0])
        else:
            chosen = min(candidates, key=lambda x: x[0])

        _, chosen_idx, chosen_opening, chosen_marker = chosen
        template_key = f"{family_key}:{chosen_idx}"
        template_usage[template_key] += 1
        if chosen_opening:
            recent_openings.append(chosen_opening)
        recent_prompts.append(str(templates[chosen_idx]))
        state["selection_events"].append(
            {
                "family_key": family_key,
                "template_idx": chosen_idx,
                "opening": chosen_opening,
                "marker": chosen_marker,
            }
        )
        return chosen_idx

    def render_diversity_diagnostics(self, *, top_k: int = 5) -> dict[str, Any]:
        self._ensure_prompt_diversity_state()
        state = self._prompt_diversity_state
        prompts = list(state["recent_prompts"])
        openings = list(state["recent_openings"])
        template_usage: Counter[str] = state["template_usage"]

        def _pairwise_jaccard(values: list[str]) -> float:
            if len(values) < 2:
                return 0.0
            scores: list[float] = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a = set(re.findall(r"[a-z0-9_]+", values[i].lower()))
                    b = set(re.findall(r"[a-z0-9_]+", values[j].lower()))
                    scores.append(len(a & b) / max(1, len(a | b)))
            return sum(scores) / len(scores)

        def _pairwise_similarity(values: list[str]) -> float:
            if len(values) < 2:
                return 0.0
            scores: list[float] = []
            normalized = [self._normalize_prompt_text(v) for v in values]
            for i in range(len(normalized)):
                for j in range(i + 1, len(normalized)):
                    scores.append(SequenceMatcher(None, normalized[i], normalized[j]).ratio())
            return sum(scores) / len(scores)

        repeated_opening_rate = 0.0
        if openings:
            unique_openings = len(set(openings))
            repeated_opening_rate = 1.0 - (unique_openings / len(openings))

        most_used = template_usage.most_common(top_k)
        return {
            "lexical_overlap_jaccard_mean": round(_pairwise_jaccard(prompts), 4),
            "normalized_similarity_mean": round(_pairwise_similarity(prompts), 4),
            "repeated_opening_rate": round(repeated_opening_rate, 4),
            "template_usage_frequency": most_used,
            "num_recent_prompts": len(prompts),
        }

    def reset_prompt_diversity_state(self) -> None:
        self._prompt_diversity_state = {
            "template_usage": Counter(),
            "recent_openings": deque(maxlen=24),
            "recent_prompts": deque(maxlen=64),
            "selection_events": deque(maxlen=256),
        }
