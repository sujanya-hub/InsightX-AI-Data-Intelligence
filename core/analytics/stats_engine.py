"""
core/analytics/stats_engine.py

Descriptive statistical analysis for numeric and categorical columns.
Strictly core-layer: no UI, no print statements, no side effects.
The original DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StatisticalEngine:
    """
    Computes column-level descriptive statistics for a bound DataFrame.

    The DataFrame is bound at construction time and treated as immutable
    throughout the object's lifetime.  Both public methods return a
    freshly constructed ``pd.DataFrame`` whose index is the column names
    of the relevant column subset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.  Must be a non-empty DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a ``pd.DataFrame``.
    ValueError
        If ``df`` has no columns or no rows.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._validate(df)
        self._df: pd.DataFrame = df

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def numerical_summary(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for every numeric column.

        All aggregations are performed with vectorised pandas/NumPy
        calls across the full numeric sub-frame simultaneously — no
        Python-level loops over columns or rows.

        Returns
        -------
        pd.DataFrame
            One row per numeric column; columns are:

            ``count`` : int
                Number of non-null observations.
            ``mean`` : float
                Arithmetic mean of non-null values.
            ``median`` : float
                50th percentile (Q2) of non-null values.
            ``std`` : float
                Sample standard deviation (ddof=1).
            ``variance`` : float
                Sample variance (ddof=1).
            ``min`` : float
                Minimum observed value.
            ``max`` : float
                Maximum observed value.
            ``skewness`` : float
                Fisher-Pearson coefficient of skewness (ddof=0 bias
                correction applied by pandas).  Positive values indicate
                a right-skewed distribution; negative values left-skewed.
            ``kurtosis`` : float
                Excess kurtosis (Fisher definition, normal → 0).
                Computed with pandas' default bias correction (ddof=0).
            ``missing_percentage`` : float
                Percentage of null cells in the column, rounded to 2
                decimal places.  Range: ``[0.00, 100.00]``.

            Returns an **empty** ``pd.DataFrame`` (with the above
            columns defined) when the bound DataFrame contains no
            numeric columns.

        Notes
        -----
        Every aggregation is computed on the numeric sub-frame in a
        single vectorised call (e.g. ``.mean()``, ``.median()``) that
        returns a ``pd.Series`` of per-column results.  These Series are
        then assembled into the output DataFrame via ``pd.concat``, which
        performs a single alignment operation rather than building the
        result row by row.
        """
        _COLUMNS = [
            "count", "mean", "median", "std", "variance",
            "min", "max", "skewness", "kurtosis", "missing_percentage",
        ]

        numeric: pd.DataFrame = self._df.select_dtypes(include="number")

        if numeric.empty:
            return pd.DataFrame(columns=_COLUMNS)

        # ── one vectorised call per statistic ────────────────────────────
        count:    pd.Series = numeric.count()
        mean:     pd.Series = numeric.mean()
        median:   pd.Series = numeric.median()
        std:      pd.Series = numeric.std(ddof=1)
        variance: pd.Series = numeric.var(ddof=1)
        minimum:  pd.Series = numeric.min()
        maximum:  pd.Series = numeric.max()
        skewness: pd.Series = numeric.skew()
        kurtosis: pd.Series = numeric.kurt()

        # missing_pct: isnull() produces a boolean frame; mean() reduces
        # it to per-column null ratios in one C-level pass.
        missing_pct: pd.Series = numeric.isnull().mean().mul(100).round(2)

        # ── assemble into a single DataFrame via concat ──────────────────
        result: pd.DataFrame = pd.concat(
            [
                count.rename("count"),
                mean.rename("mean"),
                median.rename("median"),
                std.rename("std"),
                variance.rename("variance"),
                minimum.rename("min"),
                maximum.rename("max"),
                skewness.rename("skewness"),
                kurtosis.rename("kurtosis"),
                missing_pct.rename("missing_percentage"),
            ],
            axis=1,
        )

        # count is integer; all others float — enforce clean dtypes.
        result["count"] = result["count"].astype(int)
        result[result.columns.difference(["count"])] = (
            result[result.columns.difference(["count"])].astype(float)
        )

        return result

    def categorical_summary(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for every categorical column.

        "Categorical" is defined broadly to include ``object``,
        ``category``, and ``pd.StringDtype`` columns, matching the
        dtype-detection convention used elsewhere in the InsightX pipeline.

        Internally, ``value_counts()`` is called once per column to
        derive mode-related metrics.  No Python loops iterate over
        individual rows; the per-column loop iterates over *column names*
        only (O(c) iterations, where c is the number of categorical
        columns — not the number of rows).

        Returns
        -------
        pd.DataFrame
            One row per categorical column; columns are:

            ``count`` : int
                Number of non-null observations.
            ``unique_count`` : int
                Number of distinct non-null values (``nunique``).
            ``top_value`` : object
                The most frequent non-null value.  ``None`` when the
                column is entirely null.
            ``top_frequency`` : int
                Absolute count of the most frequent value.  ``0`` when
                the column is entirely null.
            ``top_percentage`` : float
                ``top_frequency`` expressed as a percentage of
                *non-null* observations, rounded to 2 decimal places.
                Range: ``[0.00, 100.00]``.
            ``missing_percentage`` : float
                Percentage of null cells in the column, rounded to 2
                decimal places.  Range: ``[0.00, 100.00]``.

            Returns an **empty** ``pd.DataFrame`` (with the above
            columns defined) when the bound DataFrame contains no
            categorical columns.

        Notes
        -----
        ``top_percentage`` is relative to *non-null* observations rather
        than total rows.  This gives a more informative frequency signal:
        a value that appears in 90 % of filled cells is more dominant
        than raw row-count percentages would suggest in high-missingness
        columns.
        """
        _COLUMNS = [
            "count", "unique_count", "top_value",
            "top_frequency", "top_percentage", "missing_percentage",
        ]

        # Identify categorical columns via the same three-way dtype mask
        # used across the InsightX pipeline (object | category | StringDtype).
        cat_mask: pd.Series = self._df.dtypes.map(
            lambda dt: (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                or (
                    pd.api.types.is_string_dtype(dt)
                    and not pd.api.types.is_numeric_dtype(dt)
                )
            )
        )
        cat_cols: pd.Index = self._df.dtypes.index[cat_mask]

        if cat_cols.empty:
            return pd.DataFrame(columns=_COLUMNS)

        cat_df: pd.DataFrame = self._df[cat_cols]

        # ── vectorised aggregates (one call covers all cat columns) ───────
        counts:       pd.Series = cat_df.count()                           # non-null
        unique_count: pd.Series = cat_df.nunique(dropna=True)              # distinct non-null
        missing_pct:  pd.Series = cat_df.isnull().mean().mul(100).round(2)

        # ── mode-related metrics (one value_counts call per column) ───────
        # These cannot be fully vectorised across heterogeneous string
        # columns in a single pandas call; the column-level loop is O(c)
        # over column names, never over rows.
        top_values:      list = []
        top_frequencies: list = []
        top_percentages: list = []

        for col in cat_cols:
            vc: pd.Series = cat_df[col].value_counts(dropna=True, sort=True)
            if vc.empty:
                top_values.append(None)
                top_frequencies.append(0)
                top_percentages.append(0.0)
            else:
                top_freq: int = int(vc.iloc[0])
                non_null_count: int = int(counts[col])
                top_pct: float = (
                    round(top_freq / non_null_count * 100, 2)
                    if non_null_count > 0
                    else 0.0
                )
                top_values.append(vc.index[0])
                top_frequencies.append(top_freq)
                top_percentages.append(top_pct)

        # ── assemble ─────────────────────────────────────────────────────
        result: pd.DataFrame = pd.DataFrame(
            {
                "count":              counts.astype(int).values,
                "unique_count":       unique_count.astype(int).values,
                "top_value":          top_values,
                "top_frequency":      top_frequencies,
                "top_percentage":     top_percentages,
                "missing_percentage": missing_pct.values,
            },
            index=cat_cols,
        )

        return result

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
                f"StatisticalEngine requires a pd.DataFrame; "
                f"received {type(df).__name__!r}."
            )
        if df.shape[1] == 0:
            raise ValueError(
                "StatisticalEngine requires at least one column; "
                "the supplied DataFrame has none."
            )
        if df.shape[0] == 0:
            raise ValueError(
                "StatisticalEngine requires at least one row; "
                "the supplied DataFrame is empty."
            )