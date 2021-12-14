import numpy as np
from pysurvival.datasets import Dataset

from elastic_surv.dataset import PandasDataset
from elastic_surv.models.base import ModelSkeleton
from elastic_surv.optimizer import HyperbandOptimizer


def load_data() -> tuple:
    raw_dataset = Dataset("churn").load()

    time_column = "months_active"
    event_column = "churned"

    features = np.setdiff1d(raw_dataset.columns, [time_column, event_column]).tolist()

    return raw_dataset, time_column, event_column, features


def test_sanity() -> None:
    opt = HyperbandOptimizer(max_iter=27, eta=2)

    assert opt.max_iter == 27
    assert len(opt.seeds) == 3
    assert opt.eta == 2


def test_model_search() -> None:
    opt = HyperbandOptimizer(max_iter=1)
    df, tcol, ecol, feat = load_data()

    dataset = PandasDataset(
        df,
        time_column=tcol,
        event_column=ecol,
        features=feat,
    )

    selected_model = opt.select_model(dataset)

    assert isinstance(selected_model, ModelSkeleton)
