from typing import Any, List

import numpy as np
import torchtuples as tt
from pycox.models import CoxPH

from elastic_surv.dataset import BasicDataset
from elastic_surv.models.base import ModelSkeleton
from elastic_surv.models.params import Categorical, Integer, Params


class CoxPHModel(ModelSkeleton):
    def __init__(
        self,
        in_features: int,
        hidden_nodes: list = [32, 32],
        batch_norm: bool = True,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 200,
        patience: int = 10,
        batch_size: int = 128,
        verbose: bool = False,
    ) -> None:
        self.in_features = in_features

        self.num_nodes = hidden_nodes
        self.out_features = 1
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_bias = False

        self.net = tt.practical.MLPVanilla(
            self.in_features,
            self.num_nodes,
            self.out_features,
            bool(self.batch_norm),
            self.dropout,
            output_bias=self.output_bias,
        )

        self.model = CoxPH(self.net, tt.optim.Adam)
        self.model.optimizer.set_lr(lr)

        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return [
            Categorical("batch_norm", [1, 0]),
            Categorical("dropout", [0, 0.1, 0.2]),
            Categorical("lr", [1e-2, 1e-3, 1e-4]),
            Integer("patience", 10, 50, 10),
        ]

    @staticmethod
    def name() -> str:
        return "cox_ph"

    def train(self, dataset: BasicDataset, **kwargs: Any) -> "CoxPHModel":
        if not isinstance(dataset, BasicDataset):
            raise ValueError(f"Invalid dataset {type(dataset)}")

        dl_train = dataset.copy().train().dataloader(batch_size=self.batch_size)
        dl_test = dataset.copy().test().dataloader(batch_size=self.batch_size)

        log = self.model.fit_dataloader(
            dl_train, self.epochs, self.callbacks, self.verbose, val_dataloader=dl_test
        )
        if self.verbose:
            log.plot()

        for idx, batch in enumerate(dl_train):
            x_train, y_train = batch
            _ = self.model.compute_baseline_hazards(x_train, y_train)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_surv_df(X)
