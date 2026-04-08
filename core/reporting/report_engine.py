"""
InsightX · Report Engine
========================
Converts a cleaned DataFrame into a structured, JSON-safe analytics
payload consumed by NarrativeEngine and PDFBuilder.

Zero Streamlit code. Pure Python 3.11.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(val: Any) -> Any:
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.floating):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, np.ndarray):  return [_safe(x) for x in val.tolist()]
    if isinstance(val, pd.Timestamp): return val.isoformat()
    if isinstance(val, dict):         return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, list):         return [_safe(x) for x in val]
    return val


def _sd(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


# ── ReportEngine ─────────────────────────────────────────────────────────────

class ReportEngine:
    """
    Build a structured analytics payload from a cleaned DataFrame
    and optional outputs from Tabs 1-4.

    Parameters
    ----------
    df           : cleaned pd.DataFrame (required)
    tab_outputs  : merged dict of any outputs stored by previous tabs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tab_outputs: dict[str, Any] | None = None,
    ) -> None:
        if df is None or df.empty:
            raise ValueError("ReportEngine requires a non-empty DataFrame.")
        self.df   = df
        self.tabs = tab_outputs or {}

    # ── public ────────────────────────────────────────────────────────────────

    def build_report_payload(self) -> dict[str, Any]:
        logger.info("Building report payload …")
        payload = {
            "dataset_summary":  self.extract_dataset_summary(),
            "kpis":             self.extract_kpis(),
            "correlations":     self.extract_correlations(),
            "risk_factors":     self.detect_risk_signals(),
            "forecast_summary": self.summarize_forecast(),
        }
        logger.info("Payload built — %d top-level keys.", len(payload))
        return payload

    def extract_dataset_summary(self) -> dict[str, Any]:
        df   = self.df
        num  = df.select_dtypes(include="number")
        cat  = df.select_dtypes(exclude="number")
        miss = df.isnull().sum()
        pct  = (miss / len(df) * 100).round(2)

        missing_profile = {
            col: {"count": int(miss[col]), "pct": float(pct[col])}
            for col in df.columns if miss[col] > 0
        }

        desc: dict[str, Any] = {}
        if not num.empty:
            raw  = num.describe().to_dict()
            desc = {
                col: {
                    k: _safe(v)
                    for k, v in stats.items()
                    if k in ("mean", "std", "min", "max", "50%")
                }
                for col, stats in raw.items()
            }

        qs = self.tabs.get("quality_score") or self.tabs.get("overall_quality_score")

        return {
            "rows":                  int(len(df)),
            "columns":               int(df.shape[1]),
            "numeric_columns":       num.columns.tolist(),
            "categorical_columns":   cat.columns.tolist(),
            "duplicate_rows":        int(df.duplicated().sum()),
            "missing_value_profile": missing_profile,
            "quality_score":         _safe(qs),
            "descriptive_stats":     desc,
        }

    def extract_kpis(self) -> dict[str, Any]:
        num  = self.df.select_dtypes(include="number")
        kpis: dict[str, Any] = {}

        for col in num.columns:
            s = num[col].dropna()
            if s.empty:
                continue
            pct_last = float(s.pct_change().iloc[-1] * 100) if len(s) > 1 else None
            kpis[col] = {
                "mean":            _safe(s.mean()),
                "median":          _safe(s.median()),
                "std":             _safe(s.std()),
                "min":             _safe(s.min()),
                "max":             _safe(s.max()),
                "pct_change_last": _safe(pct_last),
            }

        tab2_kpis = self.tabs.get("kpis") or self.tabs.get("key_metrics") or {}
        return {**kpis, **_sd(tab2_kpis)}

    def extract_correlations(self) -> list[dict[str, Any]]:
        raw = self.tabs.get("correlations")
        if isinstance(raw, list) and raw:
            return [_sd(r) for r in raw][:15]

        num = self.df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return []

        corr = num.corr(method="pearson")
        pairs: list[dict[str, Any]] = []
        cols = corr.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = corr.loc[a, b]
                if math.isnan(v) or math.isinf(v):
                    continue
                av = abs(float(v))
                strength = (
                    "Very Strong" if av >= 0.8 else
                    "Strong"      if av >= 0.6 else
                    "Moderate"    if av >= 0.4 else
                    "Weak"        if av >= 0.2 else "Negligible"
                )
                pairs.append({"feature_a": a, "feature_b": b,
                               "correlation": round(float(v), 4),
                               "strength": strength})

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs[:15]

    def detect_risk_signals(self) -> list[dict[str, Any]]:
        risks: list[dict[str, Any]] = []

        # missing data risks
        ds = self.extract_dataset_summary()
        for col, info in ds["missing_value_profile"].items():
            p = info["pct"]
            if p >= 20:
                risks.append({
                    "category":    "Data Quality",
                    "description": f"'{col}' has {p}% missing values.",
                    "severity":    "High" if p >= 50 else "Medium",
                    "source":      "Dataset",
                })

        # quality score
        qs = ds.get("quality_score")
        if qs is not None and float(qs) < 70:
            risks.append({
                "category":    "Data Quality",
                "description": f"Overall quality score is low: {round(float(qs), 1)}%.",
                "severity":    "High",
                "source":      "Tab 1",
            })

        # high skew / outlier columns
        num = self.df.select_dtypes(include="number")
        for col in num.columns:
            s = num[col].dropna()
            if s.empty:
                continue
            skew = float(abs(s.skew()))
            if skew > 2:
                risks.append({
                    "category":    "Statistical",
                    "description": f"'{col}' is highly skewed (skew={skew:.2f}).",
                    "severity":    "Low",
                    "source":      "EDA",
                })

        # pass-through risks from prior tabs
        for key in ("risk_factors", "risks", "anomalies"):
            for r in (self.tabs.get(key) or []):
                if isinstance(r, dict):
                    risks.append({
                        "category":    r.get("category", "General"),
                        "description": r.get("description", str(r)),
                        "severity":    r.get("severity", "Low"),
                        "source":      r.get("source", "Pipeline"),
                    })
                elif isinstance(r, str):
                    risks.append({"category": "General", "description": r,
                                  "severity": "Low", "source": "Pipeline"})

        order = {"High": 0, "Medium": 1, "Low": 2}
        risks.sort(key=lambda x: order.get(x["severity"], 3))
        return risks

    def summarize_forecast(self) -> dict[str, Any]:
        tab4 = self.tabs.get("tab4_outputs") or {}
        if not tab4:
            return {"available": False}

        series = tab4.get("forecast_series") or tab4.get("forecast") or []
        if isinstance(series, pd.DataFrame):
            series = series.to_dict(orient="records")

        values = [r.get("value") or r.get("yhat") for r in series if isinstance(r, dict)]
        values = [v for v in values if v is not None]
        trend  = "Insufficient Data"
        if len(values) >= 2:
            pct = (values[-1] - values[0]) / abs(values[0]) * 100 if values[0] != 0 else 0
            trend = "Upward" if pct > 3 else "Downward" if pct < -3 else "Flat"

        return {
            "available":        True,
            "trend_direction":  trend,
            "model_used":       tab4.get("model_used") or tab4.get("forecast_model") or "Unknown",
            "horizon":          tab4.get("forecast_horizon") or len(series),
            "confidence_level": tab4.get("confidence_level") or 0.95,
            "target_column":    tab4.get("target_column") or "Unknown",
            "accuracy_metrics": _safe(tab4.get("accuracy_metrics") or {}),
        }