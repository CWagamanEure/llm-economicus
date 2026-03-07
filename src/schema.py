from dataclasses import dataclass
from typing import Any


@dataclass
class Target:
    objective: str
    state: dict[str, Any]
    beliefs: dict[str, Any]
    constraints: dict[str, Any]
    actions: list[str]
    outcome_model: dict[str, Any]
    expected_utilities: dict[str, float]
    optimal_decision: str
    brief_rationale: str


@dataclass
class Metadata:
    generator_name: str
    version: str
    seed: int


@dataclass
class DataPoint:
    task_family: str
    task_id: str
    difficulty: str
    input: str
    target: Target
    metadata: Metadata
