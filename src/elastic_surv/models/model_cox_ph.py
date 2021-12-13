from typing import Any, List

import torchtuples as tt
from pycox.models import CoxPH

from elastic_surv.dataset import ESDataset
from elastic_surv.models.base import ModelSkeleton
from elastic_surv.models.params import Params


class CoxPHModel(ModelSkeleton):
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
            self.batch_norm,
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
        return []

    @staticmethod
    def name() -> str:
        return "cox_ph"

    def train(self, dataset: ESDataset, **kwargs: Any) -> "CoxPHModel":
        assert isinstance(dataset, ESDataset), f"Invalid dataset {type(dataset)}"
        dl_train = dataset.train().dataloader(batch_size=self.batch_size)
        dl_test = dataset.copy().test().dataloader(batch_size=self.batch_size)

        log = self.model.fit_dataloader(
            dl_train, self.epochs, self.callbacks, self.verbose, val_dataloader=dl_test
        )
        if self.verbose:
            log.plot()

        return self
