from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from elastic_surv.dataset import ESDataset
from elastic_surv.models.params import Params


class ModelSkeleton(metaclass=ABCMeta):
    """Base class for all models."""

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @abstractmethod
    def train(self, dataset: ESDataset, **kwargs: Any) -> "ModelSkeleton":
        ...
