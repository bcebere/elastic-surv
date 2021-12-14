from typing import Any, List

import numpy as np
import torchtuples as tt
from pycox.models import DeepHitSingle

from elastic_surv.dataset import BasicDataset
from elastic_surv.models.base import ModelSkeleton
from elastic_surv.models.params import Categorical, Float, Integer, Params


class DeepHitModel(ModelSkeleton):
    def __init__(
        self,
        in_features: int,
        hidden_nodes: list = [32, 32],
        batch_norm: bool = True,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 200,
        patience: int = 25,
        batch_size: int = 128,
        verbose: bool = False,
        num_durations: int = 10,
        alpha: float = 0.2,
        sigma: float = 0.1,
    ) -> None:
        self.in_features = in_features

        self.num_nodes = hidden_nodes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_bias = False

        self.num_durations = num_durations

        self.net = tt.practical.MLPVanilla(
            self.in_features,
            self.num_nodes,
            self.num_durations,  # output
            bool(self.batch_norm),
            self.dropout,
            output_bias=self.output_bias,
        )
        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return [
            Categorical("batch_norm", [1, 0]),
            Categorical("dropout", [0, 0.1, 0.2]),
            Categorical("lr", [1e-2, 1e-3, 1e-4]),
            Integer("patience", 10, 50, 10),
            Float("alpha", 0, 0.5),
            Float("sigma", 0, 0.5),
        ]

    @staticmethod
    def name() -> str:
        return "deephit"

    def train(self, dataset: BasicDataset, **kwargs: Any) -> "DeepHitModel":
        assert isinstance(dataset, BasicDataset), f"Invalid dataset {type(dataset)}"

        labtrans = dataset.discrete_outcome(
            DeepHitSingle.label_transform, self.num_durations
        )

        self.model = DeepHitSingle(
            self.net,
            tt.optim.Adam,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=labtrans.cuts,
        )
        self.model.optimizer.set_lr(self.lr)

        dl_train = (
            dataset.copy()
            .train()
            .pair_rank_mat(state=True)
            .dataloader(batch_size=self.batch_size)
        )
        dl_test = (
            dataset.copy()
            .test()
            .pair_rank_mat(state=True)
            .dataloader(batch_size=self.batch_size)
        )

        log = self.model.fit_dataloader(
            dl_train, self.epochs, self.callbacks, self.verbose, val_dataloader=dl_test
        )
        if self.verbose:
            log.plot()

        dataset.pair_rank_mat(state=False)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.interpolate(self.num_durations).predict_surv_df(X)
