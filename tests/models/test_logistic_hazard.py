import numpy as np
from lifelines.datasets import load_gbsg2

from elastic_surv.dataset import PandasDataset
from elastic_surv.models import LogisticHazardModel


def load_data() -> tuple:
    raw_dataset = load_gbsg2()

    time_column = "time"
    event_column = "cens"

    features = np.setdiff1d(raw_dataset.columns, [time_column, event_column]).tolist()

    return raw_dataset, time_column, event_column, features


def test_sanity() -> None:
    df, tcol, ecol, feat = load_data()

    dataset = PandasDataset(
        df,
        time_column=tcol,
        event_column=ecol,
        features=feat,
    )

    model = LogisticHazardModel(
        in_features=dataset.features(),
        hidden_nodes=[32],
        batch_norm=False,
        dropout=0.2,
        lr=1e-3,
        epochs=3,
        patience=30,
        batch_size=100,
        verbose=True,
        num_durations=11,
    )

    assert model.in_features == dataset.features()
    assert model.num_durations == 11
    assert model.batch_norm is False
    assert model.dropout == 0.2
    assert model.epochs == 3
    assert model.verbose is True
    assert model.batch_size == 100


def test_train() -> None:
    df, tcol, ecol, feat = load_data()

    dataset = PandasDataset(
        df,
        time_column=tcol,
        event_column=ecol,
        features=feat,
    )

    model = LogisticHazardModel(
        in_features=dataset.features(),
        epochs=5,
    )

    model.train(dataset)

    score = model.score(dataset)

    assert "c_index" in score
    assert "brier_score" in score


def test_hyperparams() -> None:
    assert len(LogisticHazardModel.hyperparameter_space()) > 0

    params = LogisticHazardModel.sample_hyperparameters()

    assert len(params.keys()) == len(LogisticHazardModel.hyperparameter_space())
