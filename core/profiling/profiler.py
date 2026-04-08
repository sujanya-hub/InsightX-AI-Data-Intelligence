"""
core/profiling/profiler.py

Structural profiling of a pandas DataFrame.
Strictly core-layer: no UI, no print statements, no side effects.
"""

from __future__ import annotations

import pandas as pd


class DataProfiler:
    """
    Computes structural and statistical metadata for a fixed DataFrame.

    The DataFrame is bound at construction time and treated as immutable
    throughout the object's lifetime — no method modifies it in place or
    reassigns it.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.  Must be a concrete, non-empty DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a ``pd.DataFrame``.
    ValueError
        If ``df`` has no columns or no rows.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._validate(df)
        # Store a *view* — avoids copying large datasets while still
        # preventing accidental mutation via the private reference.
        self._df: pd.DataFrame = df

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_structure_summary(self) -> dict:
        """
        Return high-level structural metadata about the dataset.

        All counts are derived from vectorised dtype inspection; no
        Python-level iteration over columns is performed.

        Returns
        -------
        dict
            Keys and value types:

            ``rows`` : int
                Total number of rows.
            ``columns`` : int
                Total number of columns.
            ``numeric_columns`` : int
                Count of columns with a numeric dtype
                (integers, floats, and their nullable variants).
            ``categorical_columns`` : int
                Count of columns with an ``object``, ``category``, or
                ``StringDtype`` dtype.
            ``datetime_columns`` : int
                Count of columns with any ``datetime64`` dtype.
            ``memory_usage_mb`` : float
                Deep memory footprint of the DataFrame in megabytes,
                rounded to four decimal places.  "Deep" mode measures
                the true size of object/string columns.
        """
        dtypes = self._df.dtypes

        numeric_count: int = int(dtypes.map(pd.api.types.is_numeric_dtype).sum())
        datetime_count: int = int(dtypes.map(pd.api.types.is_datetime64_any_dtype).sum())
        categorical_count: int = int(
            dtypes.map(
                lambda dt: (
                    pd.api.types.is_object_dtype(dt)
                    or pd.api.types.is_categorical_dtype(dt)
                    # pd.StringDtype (pandas >= 1.0) is string-typed but
                    # reports False for is_object_dtype.
                    or (
                        pd.api.types.is_string_dtype(dt)
                        and not pd.api.types.is_numeric_dtype(dt)
                    )
                )
            ).sum()
        )

        memory_bytes: int = int(self._df.memory_usage(deep=True).sum())

        return {
            "rows":                int(self._df.shape[0]),
            "columns":             int(self._df.shape[1]),
            "numeric_columns":     numeric_count,
            "categorical_columns": categorical_count,
            "datetime_columns":    datetime_count,
            "memory_usage_mb":     round(memory_bytes / (1024 ** 2), 4),
        }

    def get_missing_percentage(self) -> pd.Series:
        """
        Compute the proportion of missing values for every column.

        Uses a single vectorised pass: ``isnull()`` produces a boolean
        DataFrame; ``.mean()`` collapses it to per-column ratios in one
        operation, then the result is scaled and rounded.

        Returns
        -------
        pd.Series
            Index — column names (same order as the DataFrame).
            Values — missing percentage as ``float``, rounded to two
            decimal places.  Range: ``0.00`` – ``100.00``.
            Name — ``"missing_pct"``.

        Notes
        -----
        Returns an all-zero Series (not an empty one) when the DataFrame
        contains no missing values, preserving a consistent index for
        callers regardless of data quality.
        """
        missing_pct: pd.Series = (
            self._df.isnull().mean().mul(100).round(2)
        )
        missing_pct.name = "missing_pct"
        return missing_pct

    def get_cardinality(self) -> pd.Series:
        """
        Count the number of distinct values in every column.

        ``nunique()`` operates column-wise in a single vectorised call
        with ``dropna=False`` so that ``NaN`` itself is counted as a
        distinct value where present, giving a more faithful picture of
        true cardinality.

        Returns
        -------
        pd.Series
            Index — column names (same order as the DataFrame).
            Values — unique-value count as ``int`` (dtype ``int64``).
            Name — ``"cardinality"``.
        """
        cardinality: pd.Series = self._df.nunique(dropna=False)
        cardinality.name = "cardinality"
        return cardinality

    def get_dtype_summary(self) -> pd.Series:
        """
        Return the dtype of every column as a human-readable string.

        Delegates directly to ``DataFrame.dtypes``; the dtype objects
        are converted to strings so that callers receive a uniform,
        JSON-serialisable representation rather than numpy/pandas dtype
        objects.

        Returns
        -------
        pd.Series
            Index — column names (same order as the DataFrame).
            Values — dtype label string, e.g. ``"int64"``,
            ``"float64"``, ``"object"``, ``"datetime64[ns]"``,
            ``"category"``, ``"string"``.
            Name — ``"dtype"``.
        """
        dtype_summary: pd.Series = self._df.dtypes.astype(str)
        dtype_summary.name = "dtype"
        return dtype_summary

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """
        Assert structural prerequisites before binding the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Candidate DataFrame to validate.

        Raises
        ------
        TypeError
            If ``df`` is not a ``pd.DataFrame``.
        ValueError
            If ``df`` has zero columns or zero rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"DataProfiler requires a pd.DataFrame; "
                f"received {type(df).__name__!r}."
            )
        if df.shape[1] == 0:
            raise ValueError(
                "DataProfiler requires at least one column; "
                "the supplied DataFrame has none."
            )
        if df.shape[0] == 0:
            raise ValueError(
                "DataProfiler requires at least one row; "
                "the supplied DataFrame is empty."
            )