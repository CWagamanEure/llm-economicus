import json
from pathlib import Path
from typing import Any, Iterable

from schema import DataPoint


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def datapoint_to_canonical_dict(datapoint: DataPoint) -> dict[str, Any]:
    problem_spec = datapoint.problem_spec
    target = datapoint.target
    metadata = datapoint.metadata

    return {
        "task_family": datapoint.task_family,
        "task_subtype": datapoint.task_subtype,
        "task_id": datapoint.task_id,
        "difficulty": datapoint.difficulty,
        "problem_spec": {
            "task_subtype": problem_spec["task_subtype"],
            "objective": problem_spec["objective"],
            "options": _canonicalize(problem_spec["options"]),
            "assumptions": _canonicalize(problem_spec["assumptions"]),
        },
        "input": datapoint.input,
        "target": {
            "objective": target.objective,
            "state": _canonicalize(target.state),
            "beliefs": _canonicalize(target.beliefs),
            "constraints": _canonicalize(target.constraints),
            "actions": list(target.actions),
            "comparison_pair": {
                "left_action": target.comparison_pair["left_action"],
                "right_action": target.comparison_pair["right_action"],
            },
            "outcome_model": _canonicalize(target.outcome_model),
            "action_values": {k: float(v) for k, v in target.action_values.items()},
            "decision_values": {k: float(v) for k, v in target.decision_values.items()},
            "optimal_decision": target.optimal_decision,
            "solver_trace": _canonicalize(target.solver_trace),
            "brief_rationale": target.brief_rationale,
        },
        "metadata": {
            "generator_name": metadata.generator_name,
            "version": metadata.version,
            "seed": metadata.seed,
            "dataset_role": metadata.dataset_role,
            "requested_prompt_style": metadata.requested_prompt_style,
            "resolved_prompt_style": metadata.resolved_prompt_style,
            "prompt_has_action_labels": metadata.prompt_has_action_labels,
            "example_fingerprint": metadata.example_fingerprint,
            "tie_threshold": metadata.tie_threshold,
            "sample_index": metadata.sample_index,
            "difficulty_metrics": _canonicalize(metadata.difficulty_metrics),
        },
    }


def datapoint_to_canonical_json(
    datapoint: DataPoint, *, indent: int | None = None
) -> str:
    canonical = datapoint_to_canonical_dict(datapoint)
    return json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), indent=indent)


def write_canonical_jsonl(datapoints: Iterable[DataPoint], output_path: str | Path) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        for datapoint in datapoints:
            handle.write(datapoint_to_canonical_json(datapoint))
            handle.write("\n")
