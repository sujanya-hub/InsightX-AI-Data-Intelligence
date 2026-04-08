"""
core/cleaning/cleaning_engine.py

Data cleaning pipeline for a bound DataFrame.
Strictly core-layer: no UI, no print statements, no side effects.
The original DataFrame is never mutated — every method returns a new copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CleaningEngine:
    """
    Applies a suite of data cleaning transformations to a fixed DataFrame.

    The source DataFrame is bound at construction time and treated as
    immutable throughout the object's lifetime.  Every public method
    operates on an internal copy and returns a *new* DataFrame; the
    bound ``_df`` attribute is never written to.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset to clean.  Must be a non-empty DataFrame.

    Raises
    ------
    TypeError
        If ``df`` is not a ``pd.DataFrame``.
    ValueError
        If ``df`` has no columns or no rows.
    """

    # Minimum non-null observations required to compute IQR on a column.
    _MIN_VALID_FOR_IQR: int = 4

    def __init__(self, df: pd.DataFrame) -> None:
        self._validate(df)
        self._df: pd.DataFrame = df

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def standardize_types(self) -> pd.DataFrame:
        """
        Coerce columns to their most appropriate dtype.

        **Numeric coercion** — every column that is not already numeric
        is passed through ``pd.to_numeric(errors='coerce')``.  A column
        is promoted to numeric only when *all* of its non-null values
        parse successfully (i.e. the number of NaNs does not increase
        after coercion).  This prevents silently converting mixed-type
        or genuinely textual columns.

        **Datetime coercion** — columns that survive the numeric filter
        (still non-numeric) are tested with ``pd.to_datetime(errors='coerce')``.
        The same zero-new-NaN guard is applied, so only unambiguous
        date/time strings are promoted.

        All other columns retain their original dtype.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with dtype-coerced columns.  The original
            DataFrame bound to this instance is not modified.

        Notes
        -----
        The method uses ``infer_datetime_format=False`` (pandas ≥ 2.0
        default) and relies on pandas' built-in parser, which handles
        ISO-8601 and many locale-aware formats without looping.
        """
        df: pd.DataFrame = self._df.copy()
        non_numeric_cols: pd.Index = df.select_dtypes(exclude="number").columns

        for col in non_numeric_cols:
            original_null_count: int = int(df[col].isnull().sum())

            # ── attempt numeric promotion ────────────────────────────────
            numeric_candidate: pd.Series = pd.to_numeric(df[col], errors="coerce")
            new_null_count: int = int(numeric_candidate.isnull().sum())
            if new_null_count == original_null_count:
                # No new NaNs introduced → safe to promote
                df[col] = numeric_candidate
                continue

            # ── attempt datetime promotion ───────────────────────────────
            # Only attempt on object/string columns, not category or bool.
            if not pd.api.types.is_object_dtype(df[col]) and \
               not pd.api.types.is_string_dtype(df[col]):
                continue

            datetime_candidate: pd.Series = pd.to_datetime(
                df[col], errors="coerce", utc=False
            )
            dt_null_count: int = int(datetime_candidate.isnull().sum())
            if dt_null_count == original_null_count:
                df[col] = datetime_candidate

        return df

    def handle_missing(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Impute missing values using the specified strategy.

        Parameters
        ----------
        strategy : str, optional
            Imputation strategy.  Currently supported values:

            ``"auto"`` *(default)*
                Fill numeric columns with their column-wise **median**
                (robust to outliers).  Fill categorical columns
                (``object``, ``category``, ``StringDtype``) with the
                string literal ``"Unknown"``.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with missing values filled.  Columns with
            no missing values are returned unchanged.

        Raises
        ------
        ValueError
            If ``strategy`` is not a recognised value.

        Notes
        -----
        *Numeric median* is computed on the non-null values of each
        column independently, so columns with different missing patterns
        receive correct per-column medians.

        The fill is applied via ``DataFrame.fillna`` on the entire
        numeric sub-frame at once using a per-column median Series,
        which is a single vectorised operation rather than a per-column
        loop.
        """
        if strategy != "auto":
            raise ValueError(
                f"Unknown strategy {strategy!r}.  Supported values: 'auto'."
            )

        df: pd.DataFrame = self._df.copy()

        # ── numeric imputation (vectorised) ──────────────────────────────
        numeric_cols: pd.Index = df.select_dtypes(include="number").columns
        if numeric_cols.size > 0:
            medians: pd.Series = df[numeric_cols].median()   # one pass
            df[numeric_cols] = df[numeric_cols].fillna(medians)

        # ── categorical imputation ────────────────────────────────────────
        # Covers object, category, and pd.StringDtype.
        cat_mask: pd.Series = df.dtypes.map(
            lambda dt: (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                or (
                    pd.api.types.is_string_dtype(dt)
                    and not pd.api.types.is_numeric_dtype(dt)
                )
            )
        )
        cat_cols: pd.Index = df.dtypes.index[cat_mask]
        if cat_cols.size > 0:
            df[cat_cols] = df[cat_cols].fillna("Unknown")

        return df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Drop fully duplicated rows, keeping the first occurrence.

        Uses ``DataFrame.drop_duplicates(keep='first')`` — a single
        vectorised hash-based pass over all columns.  No row-level
        Python iteration is performed.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with duplicate rows removed and the index
            reset to a contiguous integer range.
        """
        return (
            self._df
            .copy()
            .drop_duplicates(keep="first")
            .reset_index(drop=True)
        )

    def cap_outliers(self, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Winsorise extreme values in numeric columns to the IQR fences.

        For each numeric column with sufficient data, values below the
        lower fence or above the upper fence are *capped* (clipped) to
        the fence value.  Rows are never removed.

        **Fence definition:**

        .. code-block:: text

            lower = Q1 − iqr_multiplier × IQR
            upper = Q3 + iqr_multiplier × IQR

        Columns with fewer than ``_MIN_VALID_FOR_IQR`` non-null values
        or a zero IQR (constant columns) are skipped to avoid
        meaningless capping.

        Parameters
        ----------
        iqr_multiplier : float, optional
            Fence multiplier applied to the IQR.
            ``1.5`` *(default)* is the standard Tukey fence;
            ``3.0`` targets only extreme outliers.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with outlier values capped.  Non-numeric
            columns are preserved unchanged.

        Notes
        -----
        Capping is performed with ``DataFrame.clip(lower=..., upper=...)``
        using per-column bounds broadcast as a Series — one vectorised
        C-level call across the entire numeric sub-frame simultaneously,
        rather than iterating column by column.
        """
        df: pd.DataFrame = self._df.copy()
        numeric_cols: pd.Index = df.select_dtypes(include="number").columns

        if numeric_cols.empty:
            return df

        numeric_df: pd.DataFrame = df[numeric_cols]

        q1: pd.Series = numeric_df.quantile(0.25)
        q3: pd.Series = numeric_df.quantile(0.75)
        iqr: pd.Series = q3 - q1

        # Identify eligible columns (non-zero IQR, enough observations).
        obs: pd.Series = numeric_df.notna().sum()
        eligible: pd.Index = iqr.index[
            (iqr > 0) & (obs >= self._MIN_VALID_FOR_IQR)
        ]

        if eligible.empty:
            return df

        lower: pd.Series = q1[eligible] - iqr_multiplier * iqr[eligible]
        upper: pd.Series = q3[eligible] + iqr_multiplier * iqr[eligible]

        # clip() broadcasts lower/upper Series across the sub-frame in
        # a single vectorised operation.
        df[eligible] = numeric_df[eligible].clip(lower=lower, upper=upper, axis=1)

        return df

    def simulate_cleaning(self) -> dict:
        """
        Dry-run the cleaning pipeline and report what would change.

        No data is modified — the method reads from the bound DataFrame
        and returns counts that describe the impact of the full pipeline.

        Returns
        -------
        dict
            Keys and value types:

            ``rows_before`` : int
                Row count in the original DataFrame.
            ``rows_after_duplicates_removed`` : int
                Row count after duplicate removal
                (equivalent to ``remove_duplicates()`` output length).
            ``missing_values_to_fill`` : int
                Total number of null cells that ``handle_missing()``
                would impute (numeric *and* categorical columns).
            ``outlier_values_to_cap`` : int
                Total number of individual cell values that
                ``cap_outliers()`` would winsorise.
        """
        rows_before: int = len(self._df)

        # ── duplicates ────────────────────────────────────────────────────
        rows_after_dedup: int = int(
            (~self._df.duplicated(keep="first")).sum()
        )

        # ── missing values ────────────────────────────────────────────────
        missing_to_fill: int = int(self._df.isnull().sum().sum())

        # ── outliers (IQR) ───────────────────────────────────────────────
        numeric_df: pd.DataFrame = self._df.select_dtypes(include="number")
        outlier_cell_count: int = 0

        if not numeric_df.empty:
            q1: pd.Series = numeric_df.quantile(0.25)
            q3: pd.Series = numeric_df.quantile(0.75)
            iqr: pd.Series = q3 - q1
            obs: pd.Series = numeric_df.notna().sum()

            eligible: pd.Index = iqr.index[
                (iqr > 0) & (obs >= self._MIN_VALID_FOR_IQR)
            ]

            if not eligible.empty:
                lower: pd.Series = q1[eligible] - 1.5 * iqr[eligible]
                upper: pd.Series = q3[eligible] + 1.5 * iqr[eligible]
                below: pd.DataFrame = numeric_df[eligible].lt(lower)
                above: pd.DataFrame = numeric_df[eligible].gt(upper)
                # NaN cells produce False in lt/gt, so they are not counted.
                outlier_cell_count = int((below | above).sum().sum())

        return {
            "rows_before":                   rows_before,
            "rows_after_duplicates_removed": rows_after_dedup,
            "missing_values_to_fill":        missing_to_fill,
            "outlier_values_to_cap":         outlier_cell_count,
        }

    def execute_cleaning(self) -> pd.DataFrame:
        """
        Run the full cleaning pipeline in a fixed, deterministic order.

        **Pipeline order:**

        1. :meth:`standardize_types` — coerce columns to best-fit dtypes.
        2. :meth:`handle_missing` — impute nulls (auto strategy).
        3. :meth:`remove_duplicates` — drop exact duplicate rows.
        4. :meth:`cap_outliers` — winsorise numeric extremes (IQR × 1.5).

        Each stage receives the output of the previous stage, so type
        coercion happens before imputation (ensuring medians are computed
        on properly typed numerics) and imputation happens before
        duplicate removal (so rows are compared in their filled state).

        Returns
        -------
        pd.DataFrame
            The fully cleaned DataFrame.  The original DataFrame bound
            to this instance remains unchanged.
        """
        # Build a temporary CleaningEngine at each stage so we can chain
        # on intermediate results without mutating self._df.
        stage1: pd.DataFrame = self.standardize_types()
        stage2: pd.DataFrame = CleaningEngine(stage1).handle_missing()
        stage3: pd.DataFrame = CleaningEngine(stage2).remove_duplicates()
        stage4: pd.DataFrame = CleaningEngine(stage3).cap_outliers()
        return stage4

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
                f"CleaningEngine requires a pd.DataFrame; "
                f"received {type(df).__name__!r}."
            )
        if df.shape[1] == 0:
            raise ValueError(
                "CleaningEngine requires at least one column; "
                "the supplied DataFrame has none."
            )
        if df.shape[0] == 0:
            raise ValueError(
                "CleaningEngine requires at least one row; "
                "the supplied DataFrame is empty."
            )