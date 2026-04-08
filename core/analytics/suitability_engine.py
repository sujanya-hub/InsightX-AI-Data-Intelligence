"""
core/analytics/suitability_engine.py

Structural suitability assessment for common analytical use cases.
Strictly core-layer: no UI, no print statements, no ML logic, no model
training.  Evaluation is based entirely on DataFrame shape and column
dtype composition.  The original DataFrame is never mutated.
"""

from __future__ import annotations

import pandas as pd


class SuitabilityEngine:
    """
    Evaluates whether a bound DataFrame meets the structural prerequisites
    for four common analytical use cases: classification, regression,
    clustering, and time-series forecasting.

    Assessment is purely structural — dtype counts, row counts, and
    column presence.  No statistical models are fit, no data is modified,
    and no heavy computation is performed.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to assess.  Must be a non-empty DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a ``pd.DataFrame``.
    ValueError
        If ``df`` has no columns or no rows.
    """

    # Minimum row thresholds per use case.
    _MIN_ROWS_CLASSIFICATION: int = 50
    _MIN_ROWS_REGRESSION:     int = 50
    _MIN_ROWS_CLUSTERING:     int = 50
    _MIN_ROWS_TIME_SERIES:    int = 30

    # Minimum feature-column counts.
    _MIN_NUMERIC_FOR_CLUSTERING: int = 2

    def __init__(self, df: pd.DataFrame) -> None:
        self._validate(df)
        self._df: pd.DataFrame = df

        # ── pre-compute dtype signals once (vectorised) ──────────────────
        # All four evaluate() sub-checks share the same signals, so we
        # compute them a single time in the constructor rather than
        # re-running dtype detection on every call.
        dtypes: pd.Series = df.dtypes

        numeric_mask: pd.Series = dtypes.map(pd.api.types.is_numeric_dtype)
        datetime_mask: pd.Series = dtypes.map(pd.api.types.is_datetime64_any_dtype)
        categorical_mask: pd.Series = dtypes.map(
            lambda dt: (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                or (
                    pd.api.types.is_string_dtype(dt)
                    and not pd.api.types.is_numeric_dtype(dt)
                )
            )
        )

        self._numeric_count:     int = int(numeric_mask.sum())
        self._categorical_count: int = int(categorical_mask.sum())
        self._datetime_count:    int = int(datetime_mask.sum())
        self._row_count:         int = int(df.shape[0])

        # Candidate target columns: categorical columns are typical
        # classification targets; numeric columns are typical regression targets.
        self._numeric_cols:     list[str] = dtypes.index[numeric_mask].tolist()
        self._categorical_cols: list[str] = dtypes.index[categorical_mask].tolist()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self) -> dict:
        """
        Assess the dataset's structural suitability for four use cases.

        Each use case is evaluated against a fixed set of structural rules
        based solely on column dtype composition and row count.  No data
        values are inspected, no statistical tests are run, and no models
        are trained.

        **Suitability rules:**

        *Classification*
            - At least 1 categorical column (potential label column).
            - At least 1 numeric column (potential feature column).
            - At least 50 rows.

        *Regression*
            - At least 1 numeric column (potential target or feature).
            - At least 50 rows.

        *Clustering*
            - At least 2 numeric columns (distance computation requires
              a multi-dimensional feature space).
            - At least 50 rows.

        *Time-series*
            - At least 1 datetime column (temporal ordering axis).
            - At least 30 rows (lower threshold; shorter series are still
              meaningful for trend/seasonal decomposition).

        Returns
        -------
        dict
            Keys and value types:

            ``classification_ready`` : bool
                ``True`` when all classification prerequisites are met.
            ``regression_ready`` : bool
                ``True`` when all regression prerequisites are met.
            ``clustering_ready`` : bool
                ``True`` when all clustering prerequisites are met.
            ``time_series_ready`` : bool
                ``True`` when all time-series prerequisites are met.
            ``details`` : dict
                Supporting signals used by the readiness flags:

                ``has_numeric_features`` : bool
                    At least 1 numeric column is present.
                ``has_categorical_features`` : bool
                    At least 1 categorical column is present
                    (``object``, ``category``, or ``StringDtype``).
                ``has_datetime_column`` : bool
                    At least 1 ``datetime64`` column is present.
                ``sufficient_rows`` : bool
                    Row count meets the *most demanding* threshold
                    (``>= 50``).  Individual use-case checks use their
                    own thresholds internally.
                ``target_candidate_columns`` : list[str]
                    Ordered list of column names that are plausible
                    modelling targets: categorical columns first
                    (classification labels), followed by numeric columns
                    (regression targets).
        """
        has_numeric:     bool = self._numeric_count >= 1
        has_categorical: bool = self._categorical_count >= 1
        has_datetime:    bool = self._datetime_count >= 1
        sufficient_rows: bool = self._row_count >= max(
            self._MIN_ROWS_CLASSIFICATION,
            self._MIN_ROWS_REGRESSION,
            self._MIN_ROWS_CLUSTERING,
        )

        classification_ready: bool = (
            has_categorical
            and has_numeric
            and self._row_count >= self._MIN_ROWS_CLASSIFICATION
        )

        regression_ready: bool = (
            has_numeric
            and self._row_count >= self._MIN_ROWS_REGRESSION
        )

        clustering_ready: bool = (
            self._numeric_count >= self._MIN_NUMERIC_FOR_CLUSTERING
            and self._row_count >= self._MIN_ROWS_CLUSTERING
        )

        time_series_ready: bool = (
            has_datetime
            and self._row_count >= self._MIN_ROWS_TIME_SERIES
        )

        # Target candidates: categorical cols (classification labels)
        # listed before numeric cols (regression targets).
        target_candidates: list[str] = self._categorical_cols + self._numeric_cols

        return {
            "classification_ready": classification_ready,
            "regression_ready":     regression_ready,
            "clustering_ready":     clustering_ready,
            "time_series_ready":    time_series_ready,
            "details": {
                "has_numeric_features":      has_numeric,
                "has_categorical_features":  has_categorical,
                "has_datetime_column":       has_datetime,
                "sufficient_rows":           sufficient_rows,
                "target_candidate_columns":  target_candidates,
            },
        }

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
                f"SuitabilityEngine requires a pd.DataFrame; "
                f"received {type(df).__name__!r}."
            )
        if df.shape[1] == 0:
            raise ValueError(
                "SuitabilityEngine requires at least one column; "
                "the supplied DataFrame has none."
            )
        if df.shape[0] == 0:
            raise ValueError(
                "SuitabilityEngine requires at least one row; "
                "the supplied DataFrame is empty."
            )