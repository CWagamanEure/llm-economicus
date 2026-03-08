from abc import ABC, abstractmethod
from typing import Any

from schema import DataPoint, Metadata


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
        prompt_has_action_labels: bool = True,
        example_fingerprint: str | None = None,
        tie_threshold: float | None = None,
    ) -> Metadata:
        return Metadata(
            generator_name=self.__class__.__name__,
            version=version,
            seed=seed,
            dataset_role=dataset_role,
            requested_prompt_style=requested_prompt_style,
            resolved_prompt_style=resolved_prompt_style,
            prompt_has_action_labels=prompt_has_action_labels,
            example_fingerprint=example_fingerprint,
            tie_threshold=tie_threshold,
            sample_index=sample_index,
            difficulty_metrics=difficulty_metrics,
        )
