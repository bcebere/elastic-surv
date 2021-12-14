from typing import Any, Callable, Optional

import eland as ed
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models.data import pair_rank_mat
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicDataset(Dataset):
    def __init__(
        self,
        df: Any,
        time_column: str,
        event_column: str,
        features: Optional[list] = None,
        train: bool = True,
        train_ratio: float = 0.9,
        pair_rank: bool = False,
        label_transformer: Optional[Callable] = None,
    ) -> None:
        self._df = df

        if features is None:
            features = np.setdiff1d(
                self._df.columns, [time_column, event_column]
            ).tolist()

        features = list(features)

        self._columns = features + [time_column, event_column]
        self._features = features
        self._time_column = time_column
        self._event_column = event_column
        self._pair_rank = pair_rank

        train_len = int(len(self._df) * train_ratio)
        test_len = len(self._df) - train_len

        self._train_df = self._df.head(train_len)
        self._test_df = self._df.tail(test_len)

        self._train = train
        self._train_ratio = train_ratio
        self._iter = 0

        self._label_transformer: Optional[Callable] = label_transformer

    def outcome(self) -> tuple:
        ...

    def pair_rank_mat(self, state: bool) -> "BasicDataset":
        self._pair_rank = state

        return self

    def discrete_outcome(self, transformer: Any, num_durations: int) -> Any:
        labtrans = transformer(num_durations)

        labtrans.fit(*self.train().outcome())

        self._label_transformer = labtrans.transform

        return labtrans

    def train(self) -> "BasicDataset":
        self._train = True
        return self

    def test(self) -> "BasicDataset":
        self._train = False
        return self

    def dataloader(self, batch_size: int = 512) -> tt.data.DataLoaderBatch:
        return tt.data.DataLoaderBatch(self, batch_size)

    def __len__(self) -> int:
        return len(self._train_df) if self._train else len(self._test_df)

    def features(self) -> int:
        return len(self._columns) - 2

    def to_pandas(self, df: Any) -> pd.DataFrame:
        ...

    def __getitem__(self, index: list) -> Any:
        if not hasattr(index, "__iter__"):
            index = [index]

        batch_size = len(index)

        working_df = self._train_df if self._train else self._test_df

        if self._iter >= len(working_df):
            self._iter = 0

        data = working_df.head(self._iter + batch_size).tail(batch_size)
        data = self.to_pandas(data)
        self._iter += batch_size

        X = data[self._features].values.squeeze()
        T = data[self._time_column].values.squeeze()
        Y = data[self._event_column].values.squeeze()

        if self._label_transformer:
            T, Y = self._label_transformer(T, Y)

        target = tt.tuplefy(T, Y).to_tensor()

        if self._pair_rank:
            target = target.to_numpy()
            rank_mat = pair_rank_mat(*target)
            target = tt.tuplefy(*target, rank_mat).to_tensor()

        X = torch.from_numpy(X).float().to(DEVICE)

        return tt.tuplefy(X, target)


class ESDataset(BasicDataset):
    def __init__(
        self,
        es_index_pattern: str,
        time_column: str,
        event_column: str,
        es_client: Any = "localhost",
        features: Optional[list] = None,
        train: bool = True,
        train_ratio: float = 0.9,
        pair_rank: bool = False,
        label_transformer: Optional[Callable] = None,
    ) -> None:
        self.es_index_pattern = es_index_pattern
        self.es_client = es_client

        df = ed.DataFrame(es_client, es_index_pattern)
        super().__init__(
            df=df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            train=train,
            train_ratio=train_ratio,
            pair_rank=pair_rank,
            label_transformer=label_transformer,
        )

    def outcome(self) -> tuple:
        working_df = self._train_df if self._train else self._test_df
        working_df = ed.eland_to_pandas(working_df)

        return (working_df[self._time_column], working_df[self._event_column])

    def copy(self) -> "ESDataset":
        return ESDataset(
            es_index_pattern=self.es_index_pattern,
            time_column=self._time_column,
            event_column=self._event_column,
            features=self._features,
            train=self._train,
            train_ratio=self._train_ratio,
            es_client=self.es_client,
            pair_rank=self._pair_rank,
            label_transformer=self._label_transformer,
        )

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return ed.eland_to_pandas(df)


class PandasDataset(BasicDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        time_column: str,
        event_column: str,
        features: Optional[list] = None,
        train: bool = True,
        train_ratio: float = 0.9,
        pair_rank: bool = False,
        label_transformer: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            df=df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            train=train,
            train_ratio=train_ratio,
            pair_rank=pair_rank,
            label_transformer=label_transformer,
        )

    def outcome(self) -> tuple:
        working_df = self._train_df if self._train else self._test_df

        return (working_df[self._time_column], working_df[self._event_column])

    def copy(self) -> "PandasDataset":
        return PandasDataset(
            df=self._df,
            time_column=self._time_column,
            event_column=self._event_column,
            features=self._features,
            train=self._train,
            train_ratio=self._train_ratio,
            pair_rank=self._pair_rank,
            label_transformer=self._label_transformer,
        )

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df
