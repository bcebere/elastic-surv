import pandas as pd

from elastic_surv.dataset import PandasDataset


def generate_dummy(n: int = 3) -> PandasDataset:
    dummy = pd.DataFrame(
        [[1 * i, 2 * i, 3 * i] for i in range(1, n + 1)],
        columns=["A", "B", "C"],
    )

    return PandasDataset(
        dummy,
        time_column="A",
        event_column="B",
        pair_rank=False,
    )


def test_pandas_dataset_sanity() -> None:
    df = generate_dummy()

    assert df._features == ["C"]
    assert df._time_column == "A"
    assert df._event_column == "B"
    assert df._pair_rank is False


def test_pandas_dataset_outcome() -> None:
    df = generate_dummy()

    assert list(df.train().outcome()[0]) == [1, 2]
    assert list(df.train().outcome()[1]) == [2, 4]

    assert list(df.test().outcome()[0]) == [3]
    assert list(df.test().outcome()[1]) == [6]


def test_pandas_dataset_train_test() -> None:
    df = generate_dummy()

    df.train()
    assert df._train is True
    dup = df.copy().test()
    assert df._train is True
    assert dup._train is False

    assert len(df.train()) == 2
    assert df.train().features() == 1
    assert len(df.test()) == 1
    assert df.test().features() == 1


def test_pandas_dataset_rank_mat() -> None:
    df = generate_dummy()

    df.pair_rank_mat(True)
    assert df._pair_rank is True
    df.pair_rank_mat(False)
    assert df._pair_rank is False


def test_pandas_dataloader() -> None:
    df = generate_dummy(n=10)

    dl = df.dataloader(batch_size=5)

    it = iter(dl)

    x, y = next(it)
    assert list(x.shape) == [5]

    t, e = y
    assert len(t) == 5
    assert len(e) == 5

    x, y = next(it)
    assert list(x.shape) == [4]

    t, e = y
    assert len(t) == 4
    assert len(e) == 4
