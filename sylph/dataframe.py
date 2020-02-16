from typing import Dict

import numpy as np


class DataFrame:
    """
    A minimal dataframe class.

    Unlike pandas.DataFrame, columns can be multidimensional arrays; the size of the first
    dimension must match the number of rows in the dataframe. The dataframe can be indexed in two
    ways:

    df[key]                   # returns a column
    df.get_rows(row_indices)  # returns a DataFrame with a subset of rows and all columns

    The dataframe does not share memory with the column data supplied as arguments. Similarly
    dataframes created by `get_rows` do not share memory with their parent dataframe.
    """

    def __init__(self, columns: Dict[str, np.ndarray]):
        self._validate_columns(columns)
        self._columns = {k: np.array(v) for k, v in columns.items()}  # deep copy

    def __getitem__(self, key):
        return self._columns[key]

    def __setitem__(self, key, value):
        nrows, _ = self.shape
        if len(value) != nrows:
            raise ValueError(
                f"Length of new column ({len(value)}) is not the same as "
                f"current number of rows ({nrows})."
            )
        self._columns[key] = value

    def __contains__(self, key):
        return key in self._columns

    def __delitem__(self, key):
        del self._columns[key]

    def __len__(self):
        [nrows] = set(map(len, self._columns.values()))
        return nrows

    def get_rows(self, row_indices: np.ndarray) -> "DataFrame":
        """
        Return a DataFrame containing the specified subset of rows.

        Indexing semantics are those of numpy. I.e. the return value contains
        `column[row_indices]`,for each column, where both `column` and `row_indices` are numpy
        arrays.

        One feature of numpy indexing semantics is that the result does not share memory with the
        indexed array.
        """
        return type(self)({k: v[row_indices] for k, v in self._columns.items()})

    @property
    def shape(self):
        nrows = len(self)
        ncols = len(self._columns)
        return (nrows, ncols)

    @staticmethod
    def _validate_columns(columns):
        column_lengths = {k: len(v) for k, v in columns.items()}
        if len(set(column_lengths.values())) > 1:
            raise ValueError(f"Columns have different lengths: {column_lengths}")
        if not all(isinstance(v, np.ndarray) for v in columns.values()):
            raise TypeError("Columns must be numpy arrays")
