from typing import Any, Callable, Optional

import eland as ed
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models.data import pair_rank_mat
from sklearn.preprocessing import OneHotEncoder
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
        verbose: bool = False,
    ) -> None:
        """
        Base class for the data backends.

        Args:
            df: ed.DataFrame or pd.DataFrame. The data source
            time_column: str. Which column in the index contains the time-to-event data.
            event_column: str. Which column in the index contains the outcome data.
            features: list, optional. Which features to include.
            train: bool. If the dataset is used for training or testing.
            train_ratio: float. The ratio for the training data.
            pair_rank: bool. Whether to compute the pair rank matrix. Used for DeepHit.
            label_transformer: Callable. Callback to convert the time horizons to discrete values. Used by DeepHit or LogisticHazard.
            verbose: bool. Print debug information.
        """
        self._df = df

        if features is None:
            features = np.setdiff1d(
                self._df.columns, [time_column, event_column]
            ).tolist()

        features = list(features)

        self._columns = features + [time_column, event_column]
        self._raw_features = features
        self._features = features
        self._time_column = time_column
        self._event_column = event_column
        self._pair_rank = pair_rank

        if df[self._time_column].dtype == "object":
            raise ValueError(f"Invalid time data type {df[self._time_column].dtype}")

        if df[self._event_column].dtype == "object":
            raise ValueError(f"Invalid tevent data type {df[self._event_column].dtype}")

        train_len = int(len(self._df) * train_ratio)
        test_len = len(self._df) - train_len

        self._train_df = self._df.head(train_len)
        self._test_df = self._df.tail(test_len)

        self._train = train
        self._train_ratio = train_ratio
        self._iter = 0

        self._verbose = verbose

        self._label_transformer: Optional[Callable] = label_transformer

        self._encoders = {}

        for col in self._columns:
            if self._df[col].dtype != "object":
                continue
            unique_cnt = self._df[col].nunique()

            unique_vals = set()
            for idx, row in self._df.iterrows():
                val = row[col]
                unique_vals.add(val)

                if len(unique_vals) == unique_cnt:
                    break

            if self._verbose:
                print(f"preprocess: dataset one-hot encoding for {col}", unique_vals)
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            target = np.asarray(list(unique_vals)).reshape(-1, 1)
            ohe.fit(target)
            self._encoders[col] = ohe

        for col in self._columns:
            if col not in self._encoders:
                continue
            new_cols = self._encoders[col].get_feature_names([col])

            self._features = np.setdiff1d(self._features, [col]).tolist()
            self._features.extend(new_cols)

    def outcome(self) -> tuple:
        """
        Get (time_to_event, outcome) values for the current dataset.
        """
        ...

    def pair_rank_mat(self, state: bool) -> "BasicDataset":
        """
        Enable/disable pair rank matrix generation.
        """
        self._pair_rank = state

        return self

    def discrete_outcome(self, transformer: Any, num_durations: int) -> Any:
        """
        Convert outcomes to discrete points.
        """
        labtrans = transformer(num_durations)

        labtrans.fit(*self.train().outcome())

        self._label_transformer = labtrans.transform

        return labtrans

    def train(self) -> "BasicDataset":
        """
        Enable train mode.
        """
        self._train = True
        return self

    def test(self) -> "BasicDataset":
        """
        Enable test mode.
        """
        self._train = False
        return self

    def dataloader(self, batch_size: int = 512) -> tt.data.DataLoaderBatch:
        """
        Return the dataloader for the current dataset.
        """
        return tt.data.DataLoaderBatch(self, batch_size)

    def __len__(self) -> int:
        """
        Return the length of the current dataset.
        """
        return len(self._train_df) if self._train else len(self._test_df)

    def features(self) -> int:
        """
        Return the number of features.
        """
        return len(self._features)

    def to_pandas(self, df: Any) -> pd.DataFrame:
        """
        Convert the dataset to pandas DataFrame
        """
        ...

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encoding for non-numeric columns
        """
        df = df.copy()
        for col in self._encoders:
            ohe = self._encoders[col]

            encoded = pd.DataFrame(
                ohe.transform(df[col].values.reshape(-1, 1)),
                columns=ohe.get_feature_names([col]),
                index=df.index.copy(),
            )
            df = pd.concat([df, encoded], axis=1)
            df.drop(columns=[col], inplace=True)
        return df

    def __getitem__(self, index: list) -> Any:
        """
        Retrive batches from the backend.
        """
        if not hasattr(index, "__iter__"):
            index = [index]

        batch_size = len(index)

        working_df = self._train_df if self._train else self._test_df

        if self._iter >= len(working_df):
            self._iter = 0

        data = working_df.head(self._iter + batch_size).tail(batch_size)
        data = self.to_pandas(data)
        self._iter += batch_size

        data = self.encode(data)

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
        verbose: bool = False,
    ) -> None:
        """
        Class for the ElasticSearch data backends.

        Args:
            es_index_pattern: str. The index pattern to retrieve from ElasticSearch
            time_column: str. Which column in the index contains the time-to-event data.
            event_column: str. Which column in the index contains the outcome data.
            es_client: str or elasticsearch.Elasticsearch object. Default is "localhost".
            features: list, optional. Which features to include.
            train: bool. If the dataset is used for training or testing.
            train_ratio: float. The ratio for the training data.
            pair_rank: bool. Whether to compute the pair rank matrix. Used for DeepHit.
            label_transformer: Callable. Callback to convert the time horizons to discrete values. Used by DeepHit or LogisticHazard.
            verbose: bool. Print debug information.
        """
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
            verbose=verbose,
        )

    def outcome(self) -> tuple:
        """
        Get (time_to_event, outcome) values for the current dataset.
        """

        working_df = self._train_df if self._train else self._test_df
        working_df = ed.eland_to_pandas(working_df)

        return (working_df[self._time_column], working_df[self._event_column])

    def copy(self) -> "ESDataset":
        """
        Duplicate the ESDataset object.
        """
        return ESDataset(
            es_index_pattern=self.es_index_pattern,
            time_column=self._time_column,
            event_column=self._event_column,
            features=self._raw_features,
            train=self._train,
            train_ratio=self._train_ratio,
            es_client=self.es_client,
            pair_rank=self._pair_rank,
            label_transformer=self._label_transformer,
        )

    def to_pandas(self, df: Any) -> pd.DataFrame:
        """
        Convert the eland.DataFrame to pandas dataframe
        """
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
        verbose: bool = False,
    ) -> None:
        """
        Class for the Pandas data backends.

        Args:
            df: pd.DataFrame.
            time_column: str. Which column in the index contains the time-to-event data.
            event_column: str. Which column in the index contains the outcome data.
            features: list, optional. Which features to include.
            train: bool. If the dataset is used for training or testing.
            train_ratio: float. The ratio for the training data.
            pair_rank: bool. Whether to compute the pair rank matrix. Used for DeepHit.
            label_transformer: Callable. Callback to convert the time horizons to discrete values. Used by DeepHit or LogisticHazard.
            verbose: bool. Print debug information.
        """

        super().__init__(
            df=df,
            time_column=time_column,
            event_column=event_column,
            features=features,
            train=train,
            train_ratio=train_ratio,
            pair_rank=pair_rank,
            label_transformer=label_transformer,
            verbose=verbose,
        )

    def outcome(self) -> tuple:
        """
        Get (time_to_event, outcome) values for the current dataset.
        """
        working_df = self._train_df if self._train else self._test_df

        return (working_df[self._time_column], working_df[self._event_column])

    def copy(self) -> "PandasDataset":
        """
        Duplicate the dataset.
        """
        return PandasDataset(
            df=self._df,
            time_column=self._time_column,
            event_column=self._event_column,
            features=self._raw_features,
            train=self._train,
            train_ratio=self._train_ratio,
            pair_rank=self._pair_rank,
            label_transformer=self._label_transformer,
        )

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df
