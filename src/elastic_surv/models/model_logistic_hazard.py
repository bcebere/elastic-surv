from typing import Any, List

import torchtuples as tt
from pycox.models import LogisticHazard

from elastic_surv.dataset import ESDataset
from elastic_surv.models.base import ModelSkeleton
from elastic_surv.models.params import Params


class LogisticHazardModel(ModelSkeleton):
    def __init__(
        self,
        in_features: int,
        hidden_nodes: list = [32, 32],
        batch_norm: bool = True,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 512,
        verbose: bool = False,
        num_durations: int = 10,
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
            self.batch_norm,
            self.dropout,
            output_bias=self.output_bias,
        )
        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return []

    @staticmethod
    def name() -> str:
        return "logistic_hazard"

    def train(self, dataset: ESDataset, **kwargs: Any) -> "LogisticHazard":
        assert isinstance(dataset, ESDataset), f"Invalid dataset {type(dataset)}"

        labtrans = dataset.discrete_outcome(
            LogisticHazard.label_transform, self.num_durations
        )

        self.model = LogisticHazard(
            self.net, tt.optim.Adam, duration_index=labtrans.cuts
        )
        self.model.optimizer.set_lr(self.lr)

        dl_train = dataset.train().dataloader(batch_size=self.batch_size)
        dl_test = dataset.copy().test().dataloader(batch_size=self.batch_size)

        log = self.model.fit_dataloader(
            dl_train, self.epochs, self.callbacks, self.verbose, val_dataloader=dl_test
        )
        if self.verbose:
            log.plot()

        return self
