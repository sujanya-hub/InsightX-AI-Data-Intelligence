from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Payloads
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ColumnSchema:
    """Classified profile for a single column."""
    name: str
    dtype_raw: str
    role: str               # "numeric" | "categorical" | "datetime" | "text" | "id" | "boolean"
    cardinality: int
    missing_pct: float
    sample_values: list     # first 5 non-null values


@dataclass(frozen=True)
class DetectionResult:
    """Full schema inference result for a DataFrame."""
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    boolean_cols: list[str]
    id_cols: list[str]
    text_cols: list[str]
    primary_metric: Optional[str]           # best numeric target column
    segment_col: Optional[str]              # best categorical grouping column
    time_col: Optional[str]                 # best datetime column
    has_time_series: bool
    schema: list[ColumnSchema]
    column_map: dict[str, str]              # col → role


# ══════════════════════════════════════════════════════════════════════════════
# Heuristics
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYWORDS = re.compile(
    r"(revenue|sales|amount|price|cost|profit|income|value|total|sum|"
    r"score|rate|count|qty|quantity|volume|spend|gross|net|margin|units)",
    re.IGNORECASE,
)

_SEGMENT_KEYWORDS = re.compile(
    r"(category|segment|region|country|state|city|product|brand|channel|"
    r"department|team|group|type|class|tier|status|label|name)",
    re.IGNORECASE,
)

_TIME_KEYWORDS = re.compile(
    r"(date|time|year|month|week|day|period|quarter|timestamp|created|updated)",
    re.IGNORECASE,
)

_ID_KEYWORDS = re.compile(
    r"(^id$|_id$|^uuid|^key$|^index$|^code$|^no$|^num$|^number$)",
    re.IGNORECASE,
)

_HIGH_CARDINALITY_RATIO = 0.85      # col is "id-like" if unique ratio exceeds this
_LOW_CARDINALITY_MAX = 50           # categorical if unique count ≤ this
_TEXT_MIN_AVG_LEN = 20              # object col is "text" if avg str len exceeds this


# ══════════════════════════════════════════════════════════════════════════════
# Detector
# ══════════════════════════════════════════════════════════════════════════════

class UniversalDetector:
    """
    Stateless schema inference engine.

    Usage
    -----
    result = UniversalDetector().detect(df)
    """

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Infer column roles, primary metric, segment column, and time column.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe. Not mutated.

        Returns
        -------
        DetectionResult
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None.")

        df = self._coerce_datetimes(df.copy())
        n = len(df)

        schemas: list[ColumnSchema] = []
        col_map: dict[str, str] = {}

        for col in df.columns:
            role = self._classify(df[col], col, n)
            schemas.append(ColumnSchema(
                name=col,
                dtype_raw=str(df[col].dtype),
                role=role,
                cardinality=int(df[col].nunique(dropna=True)),
                missing_pct=round(df[col].isnull().mean() * 100, 2),
                sample_values=df[col].dropna().head(5).tolist(),
            ))
            col_map[col] = role

        numeric_cols    = [s.name for s in schemas if s.role == "numeric"]
        categorical_cols = [s.name for s in schemas if s.role == "categorical"]
        datetime_cols   = [s.name for s in schemas if s.role == "datetime"]
        boolean_cols    = [s.name for s in schemas if s.role == "boolean"]
        id_cols         = [s.name for s in schemas if s.role == "id"]
        text_cols       = [s.name for s in schemas if s.role == "text"]

        primary_metric = self._infer_primary_metric(df, numeric_cols)
        segment_col    = self._infer_segment(df, categorical_cols)
        time_col       = datetime_cols[0] if datetime_cols else None

        return DetectionResult(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            boolean_cols=boolean_cols,
            id_cols=id_cols,
            text_cols=text_cols,
            primary_metric=primary_metric,
            segment_col=segment_col,
            time_col=time_col,
            has_time_series=bool(time_col and primary_metric),
            schema=schemas,
            column_map=col_map,
        )

    # ── coercion ──────────────────────────────────────────────────────────────

    def _coerce_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to parse object columns that look like dates."""
        for col in df.select_dtypes(include=["object"]).columns:
            if _TIME_KEYWORDS.search(col):
                try:
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    if parsed.notnull().mean() >= 0.7:
                        df[col] = parsed
                except Exception:
                    pass
        return df

    # ── column classification ─────────────────────────────────────────────────

    def _classify(self, series: pd.Series, col: str, n: int) -> str:
        dtype = series.dtype

        # datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # boolean
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1, True, False, "0", "1", "yes", "no", "true", "false",
                           "Yes", "No", "True", "False"}:
            if len(unique_vals) <= 2:
                return "boolean"

        # numeric
        if pd.api.types.is_numeric_dtype(dtype):
            unique_ratio = series.nunique(dropna=True) / max(n, 1)
            # high-cardinality int → probably id
            if pd.api.types.is_integer_dtype(dtype) and _ID_KEYWORDS.search(col):
                return "id"
            if unique_ratio >= _HIGH_CARDINALITY_RATIO and pd.api.types.is_integer_dtype(dtype):
                return "id"
            return "numeric"

        # object / string
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / max(n, 1)

            # id-like
            if _ID_KEYWORDS.search(col) or unique_ratio >= _HIGH_CARDINALITY_RATIO:
                return "id"

            # long free text
            sample = series.dropna().astype(str)
            if len(sample) and sample.str.len().mean() > _TEXT_MIN_AVG_LEN and unique_count > 100:
                return "text"

            # categorical
            if unique_count <= _LOW_CARDINALITY_MAX or _SEGMENT_KEYWORDS.search(col):
                return "categorical"

            return "text"

        return "text"

    # ── primary metric inference ──────────────────────────────────────────────

    def _infer_primary_metric(self, df: pd.DataFrame, numeric_cols: list[str]) -> Optional[str]:
        if not numeric_cols:
            return None

        scored: list[tuple[str, float]] = []
        for col in numeric_cols:
            score = 0.0
            if _METRIC_KEYWORDS.search(col):
                score += 3.0
            series = df[col].dropna()
            if len(series) == 0:
                continue
            # prefer higher variance (more signal)
            cv = series.std() / (series.mean() + 1e-9)
            score += min(cv, 2.0)
            # penalise missing
            score -= df[col].isnull().mean() * 2.0
            scored.append((col, score))

        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    # ── segment column inference ──────────────────────────────────────────────

    def _infer_segment(self, df: pd.DataFrame, categorical_cols: list[str]) -> Optional[str]:
        if not categorical_cols:
            return None

        scored: list[tuple[str, float]] = []
        for col in categorical_cols:
            score = 0.0
            if _SEGMENT_KEYWORDS.search(col):
                score += 3.0
            n_unique = df[col].nunique(dropna=True)
            # prefer 2–30 distinct segments
            if 2 <= n_unique <= 30:
                score += 2.0
            elif n_unique > 30:
                score -= 1.0
            score -= df[col].isnull().mean() * 2.0
            scored.append((col, score))

        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]