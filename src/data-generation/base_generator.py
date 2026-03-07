from abc import ABC, abstractmethod
from schema import DataPoint, Target, Metadata 


class BaseGenerator(ABC):
    """
    This is the base data generator
    """

    @abstractmethod
    def generate(self) -> DataPoint:
        raise NotImplementedError()

    def build_metadata(self, seed: int, version: str = "v1") -> Metadata:
        return Metadata(
            generator_name=self.__class__.__name__,
            version=version,
            seed=seed,
        )
