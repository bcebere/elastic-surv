from typing import Any, Optional

import eland as ed
import torch
import torchtuples as tt
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESDataset(Dataset):
    def __init__(
        self,
        es_index_pattern: str,
        time_column: str,
        event_column: str,
        es_client: Any = "localhost",
        features: Optional[list] = None,
        train: bool = True,
        train_ratio: float = 0.9,
    ) -> None:
        self._params = {
            "es_index_pattern": es_index_pattern,
            "time_column": time_column,
            "event_column": event_column,
            "es_client": es_client,
            "features": features,
            "train": train,
            "train_ratio": train_ratio,
        }

        self._df = ed.DataFrame(es_client, es_index_pattern)

        if features is None:
            features = self._df.columns

        features = list(features)

        self._columns = features + [time_column, event_column]
        self._features = features
        self._time_column = time_column
        self._event_column = event_column

        # self._df = self._df[self._columns]

        train_len = int(len(self._df) * train_ratio)
        test_len = len(self._df) - train_len

        self._train_df = self._df.head(train_len)
        self._test_df = self._df.tail(test_len)

        self._train = train
        self._iter = 0

    def train(self) -> "ESDataset":
        self._train = True
        return self

    def test(self) -> "ESDataset":
        self._train = False
        return self

    def copy(self) -> "ESDataset":
        return ESDataset(**self._params)

    def dataloader(self, batch_size: int = 512) -> tt.data.DataLoaderBatch:
        return tt.data.DataLoaderBatch(self, batch_size)

    def __len__(self) -> int:
        return len(self._train_df) if self._train else len(self._test_df)

    def features(self) -> int:
        return len(self._columns) - 2

    def __getitem__(self, index: list) -> Any:
        if not hasattr(index, "__iter__"):
            index = [index]

        batch_size = len(index)

        working_df = self._train_df if self._train else self._test_df

        if self._iter >= len(working_df):
            self._iter = 0

        data = working_df.head(self._iter + batch_size).tail(batch_size)
        data = ed.eland_to_pandas(data)
        self._iter += batch_size

        X = data[self._features].values.squeeze()
        T = data[self._time_column].values.squeeze()
        Y = data[self._event_column].values.squeeze()

        X, T, Y = (
            torch.from_numpy(X).float().to(DEVICE),
            torch.from_numpy(T).float().to(DEVICE),
            torch.from_numpy(Y).long().to(DEVICE),
        )

        return tt.tuplefy(X, (T, Y))
