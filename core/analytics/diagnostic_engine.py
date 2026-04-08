"""
DiagnosticEngine — Core analytical intelligence for InsightX.
Pure computation layer. Zero UI dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

_MAX_CORR_COLUMNS = 50


@dataclass
class ColumnDiagnostic:
    name: str
    dtype: str
    skewness: Optional[float] = None
    outlier_pct: Optional[float] = None
    distribution: Optional[str] = None
    dominance_pct: Optional[float] = None
    cardinality: Optional[int] = None
    missing_pct: float = 0.0
    outlier_risk: Optional[str] = None
    skew_risk: Optional[str] = None


@dataclass
class DiagnosticReport:
    column_diagnostics: dict[str, ColumnDiagnostic] = field(default_factory=dict)
    top_correlations: list[dict] = field(default_factory=list)
    strongest_correlation_pair: Optional[tuple[str, str, float]] = None
    most_skewed_column: Optional[tuple[str, float]] = None
    top_risk_columns: list[ColumnDiagnostic] = field(default_factory=list)


class DiagnosticEngine:
    """Encapsulates all analytical diagnostic logic. No Streamlit dependencies."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        self._report: Optional[DiagnosticReport] = None

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def get_row_count(self) -> int:
        """Return number of rows in working dataset."""
        return self._df.shape[0]

    def get_column_names(self) -> list[str]:
        return self._df.columns.tolist()

    def get_column_diagnostic(self, column: str) -> Optional[ColumnDiagnostic]:
        return self.build_report().column_diagnostics.get(column)

    def get_column_series(self, column: str) -> pd.Series:
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found.")
        return self._df[column].dropna()

    def get_numeric_distribution(self, column: str) -> dict:
        series = self._df[column].dropna()
        counts, bin_edges = np.histogram(series, bins="auto")
        return {
            "x": (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist(),
            "counts": counts.tolist(),
        }

    def get_boxplot_series(self, column: str) -> pd.Series:
        return self._df[column].dropna()

    def get_frequency_table(self, column: str) -> pd.Series:
        return self._df[column].value_counts(dropna=False)

    def get_time_series(self, column: str) -> pd.DataFrame:
        ts = self._df[column].value_counts().sort_index().reset_index()
        ts.columns = [column, "count"]
        ts["rolling_mean"] = ts["count"].rolling(7, min_periods=1).mean()
        return ts

    def build_report(self) -> DiagnosticReport:
        if self._report is not None:
            return self._report

        report = DiagnosticReport()
        report.column_diagnostics = self._diagnose_all_columns()
        report.top_correlations = self._compute_top_correlations()
        report.strongest_correlation_pair = self._extract_strongest_pair(report.top_correlations)
        report.most_skewed_column = self._find_most_skewed(report.column_diagnostics)
        report.top_risk_columns = self._compute_top_risk_columns(report.column_diagnostics)

        self._report = report
        return report

    # ─────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────

    def _diagnose_all_columns(self) -> dict[str, ColumnDiagnostic]:
        diagnostics = {}

        for col in self._numeric_cols:
            series = self._df[col].dropna()
            skew = float(series.skew()) if len(series) > 2 else 0.0
            outlier_pct = self._compute_outlier_pct(series)

            diagnostics[col] = ColumnDiagnostic(
                name=col,
                dtype="numeric",
                skewness=round(skew, 4),
                outlier_pct=round(outlier_pct, 2),
                distribution=self._classify_distribution(skew),
                missing_pct=round(self._df[col].isna().mean() * 100, 2),
                outlier_risk=self._outlier_risk(outlier_pct),
                skew_risk=self._skew_risk(skew),
            )

        for col in self._cat_cols:
            vc = self._df[col].value_counts(dropna=False)
            dominance = (vc.iloc[0] / len(self._df) * 100) if len(vc) > 0 else 0.0

            diagnostics[col] = ColumnDiagnostic(
                name=col,
                dtype="categorical",
                cardinality=int(self._df[col].nunique()),
                dominance_pct=round(dominance, 2),
                missing_pct=round(self._df[col].isna().mean() * 100, 2),
            )

        for col in self._datetime_cols:
            diagnostics[col] = ColumnDiagnostic(
                name=col,
                dtype="datetime",
                missing_pct=round(self._df[col].isna().mean() * 100, 2),
            )

        return diagnostics

    def _compute_top_correlations(self, top_n: int = 5) -> list[dict]:
        if len(self._numeric_cols) < 2:
            return []

        cols = self._select_high_variance_columns()
        corr = self._df[cols].corr()

        pairs = [
            {
                "col_a": a,
                "col_b": b,
                "correlation": round(float(corr.loc[a, b]), 4),
            }
            for i, a in enumerate(cols)
            for b in cols[i + 1 :]
            if not np.isnan(corr.loc[a, b])
        ]

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs[:top_n]

    def _select_high_variance_columns(self) -> list[str]:
        if len(self._numeric_cols) <= _MAX_CORR_COLUMNS:
            return self._numeric_cols

        return (
            self._df[self._numeric_cols]
            .var()
            .sort_values(ascending=False)
            .head(_MAX_CORR_COLUMNS)
            .index.tolist()
        )

    def _compute_top_risk_columns(
        self,
        diagnostics: dict[str, ColumnDiagnostic],
        top_n: int = 3,
    ) -> list[ColumnDiagnostic]:

        _rank = {"High": 2, "Moderate": 1, "Low": 0}
        numeric = [d for d in diagnostics.values() if d.dtype == "numeric"]

        return sorted(
            numeric,
            key=lambda d: _rank.get(d.outlier_risk or "Low", 0)
            + _rank.get(d.skew_risk or "Low", 0),
            reverse=True,
        )[:top_n]

    @staticmethod
    def _extract_strongest_pair(correlations: list[dict]) -> Optional[tuple]:
        if not correlations:
            return None
        top = correlations[0]
        return (top["col_a"], top["col_b"], top["correlation"])

    @staticmethod
    def _find_most_skewed(diagnostics: dict) -> Optional[tuple]:
        candidates = [
            (n, d.skewness)
            for n, d in diagnostics.items()
            if d.skewness is not None
        ]
        return max(candidates, key=lambda x: abs(x[1])) if candidates else None

    @staticmethod
    def _compute_outlier_pct(series: pd.Series) -> float:
        if len(series) < 4:
            return 0.0
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        return len(series[mask]) / len(series) * 100

    @staticmethod
    def _classify_distribution(skewness: float) -> str:
        a = abs(skewness)
        if a > 2:
            return f"Highly skewed ({'right' if skewness > 0 else 'left'})"
        if a > 1:
            return f"Moderately skewed ({'right' if skewness > 0 else 'left'})"
        if a > 0.5:
            return "Slightly skewed"
        return "Approximately normal"

    @staticmethod
    def _outlier_risk(pct: float) -> str:
        return "High" if pct > 5 else "Moderate" if pct >= 1 else "Low"

    @staticmethod
    def _skew_risk(skewness: float) -> str:
        a = abs(skewness)
        return "High" if a > 1 else "Moderate" if a > 0.5 else "Low"