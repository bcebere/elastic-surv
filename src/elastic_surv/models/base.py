from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

import numpy as np
from pycox.evaluation import EvalSurv

from elastic_surv.dataset import BasicDataset
from elastic_surv.models.params import Params


class ModelSkeleton(metaclass=ABCMeta):
    """Base class for all models."""

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        """
        Return the list of hyperparameters for the current model.
        """
        ...

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Sample from the list of hyperparameters for the current model.
        """
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return the name of the current model.
        """
        ...

    @abstractmethod
    def train(self, dataset: BasicDataset, **kwargs: Any) -> "ModelSkeleton":
        """
        Train the current model
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the survival function for the input.
        """
        ...

    def score(self, dataset: BasicDataset, **kwargs: Any) -> dict:
        """
        Return the C-Index and Brier score for the input.
        """
        test_ds = dataset.copy().test()
        test_dl = test_ds.dataloader(batch_size=len(test_ds))
        x_test, y_test = next(iter(test_dl))

        surv = self.predict(x_test)

        t_test, e_test = y_test
        t_test = t_test.cpu().numpy()
        e_test = e_test.cpu().numpy()

        ev = EvalSurv(surv, t_test, e_test, censor_surv="km")

        time_grid = np.linspace(t_test.min(), t_test.max(), 100)

        return {
            "c_index": ev.concordance_td(),
            "brier_score": ev.integrated_brier_score(time_grid),
        }
