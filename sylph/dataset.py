from typing import Optional

import numpy as np

from sylph.dataframe import DataFrame


class DataSet(DataFrame):
    """
    A dataset comprises the union of its training and testing subsets, which are themselves
    datasets. So, e.g.
    dataset.n == len(dataset.observations) == len(dataset.labels)
    dataset.n == dataset.training_dataset.n + dataset.testing_dataset.n
    """

    def __init__(
        self,
        columns,
        training_rows: Optional[np.ndarray] = None,
        training_proportion: Optional[float] = None,
    ):
        self._validate_columns(columns)
        super().__init__(columns)
        self._training_rows = training_rows
        self.training_proportion = training_proportion

    @property
    def observations(self):
        return self._columns["observations"]

    @property
    def ids(self):
        return self._columns["ids"]

    @property
    def labels(self):
        return self._columns["labels"]

    def __repr__(self):
        _type = type(self).__name__
        return f"{_type}(n={self.n}, labels=[{repr(self.labels[0])}, ...])"

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def training_rows(self) -> np.ndarray:
        if self._training_rows is not None:
            return self._training_rows
        elif self.training_proportion is not None:
            n_training = int(self.training_proportion * self.n)
            return np.array([i < n_training for i in range(self.n)])
        else:
            raise ValueError("DataSet requires either `training_rows` or `training_proprtion`")

    @property
    def training_dataset(self) -> "DataSet":
        return self.get_rows(self.training_rows)  # type: ignore

    @property
    def testing_dataset(self) -> "DataSet":
        return self.get_rows(~self.training_rows)  # type: ignore

    @staticmethod
    def _validate_columns(columns):
        try:
            observations = columns["observations"]
        except KeyError:
            raise ValueError('columns dict must include a key named "observations".')
        try:
            ids = columns["ids"]
        except KeyError:
            raise ValueError('columns dict must include a key named "ids".')
        try:
            labels = columns["labels"]
        except KeyError:
            raise ValueError('columns dict must include a key named "labels".')

        # The array of observations is either a 1D array of python objects (each of which wrap an
        # underlying array of data) or a multi-dimensional array (i.e. each observation is itself
        # an array).
        if observations.dtype == np.object:
            assert len(observations.shape) == 1
        else:
            assert len(observations.shape) > 1
        # The arrays of labels and ids are both 1D
        assert len(labels.shape) == len(ids.shape) == 1
