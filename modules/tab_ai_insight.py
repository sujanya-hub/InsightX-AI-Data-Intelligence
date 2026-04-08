"""
InsightX · AI Insight Lab  (Tab 4)
modules/tab_ai_insight.py

Diagnostic Intelligence Briefing.
Purpose: explain WHY patterns exist in the dataset.
Forecasting / scenario modelling lives in the Decision Lab (Tab 3).

Six sections
────────────
1. Dataset Pulse            — executive characterisation of the data
2. Metric Driver Analysis   — ranked variable influence on the primary metric
3. Hidden Pattern Detection — structural / cyclical / noise deviations
4. Segment Intelligence     — top/bottom performers with narrative
5. Risk Watchlist           — severity-ranked data risks
6. Strategic Opportunities  — actionable recommendations from pattern evidence

Architecture rules
──────────────────
• Zero modification to analytics engines.
• All computation via the pure helpers already present in this file
  (ported verbatim from the original tab_ai_insight.py).
• All LLM calls via  ai_engine.generate(prompt, max_tokens=…).
• All outputs persisted to st.session_state for the ReportEngine.

Session-state keys populated
─────────────────────────────
ai_dataset_pulse  ·  ai_driver_analysis  ·  ai_hidden_patterns
ai_segment_intelligence  ·  ai_risk_watchlist  ·  ai_opportunities
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine


# ═══════════════════════════════════════════════════════════════════════════════
# 1 ·  DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

T: dict[str, str] = dict(
    bg     = "#0B0F19",
    card   = "#141A2A",
    card2  = "#0F1523",
    border = "#1F2937",
    a1     = "#7C5CFF",   # violet   — primary / dataset pulse
    a2     = "#06B6D4",   # cyan     — driver analysis
    a3     = "#10B981",   # emerald  — opportunities / positive
    a4     = "#F59E0B",   # amber    — patterns / warning
    a5     = "#EF4444",   # red      — risk / critical
    a6     = "#8B5CF6",   # purple   — hidden patterns
    text   = "#F8FAFC",
    muted  = "#94A3B8",
)

_SEV: dict[str, str] = {
    "critical": "#EF4444",
    "high":     "#F59E0B",
    "medium":   "#06B6D4",
    "low":      "#10B981",
}

_PAT: dict[str, str] = {
    "Structural": "#EF4444",
    "Cyclical":   "#8B5CF6",
    "Noise":      "#F59E0B",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ·  SERIALISATION  (ported verbatim from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _default_encoder(obj: Any) -> Any:
    try:
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
    except Exception:
        pass
    try:
        if isinstance(obj, pd.Series):    return obj.tolist()
        if isinstance(obj, pd.DataFrame): return obj.to_dict(orient="list")
    except Exception:
        pass
    return str(obj)


def _safe_json(obj: Any, max_chars: int = 3000) -> str:
    try:
        raw = json.dumps(obj, indent=2, default=_default_encoder)
    except Exception:
        raw = str(obj)
    return raw[:max_chars] + "\n... [truncated]" if len(raw) > max_chars else raw


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ·  ANALYTICS HELPERS  (ported verbatim from original, nothing modified)
# ═══════════════════════════════════════════════════════════════════════════════

def _top_correlations(df: pd.DataFrame, n: int = 5) -> list[dict]:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["col_a", "col_b", "abs_r"]
    top = pairs.sort_values("abs_r", ascending=False).head(n)
    return [
        {"col_a": str(r["col_a"]), "col_b": str(r["col_b"]),
         "abs_r": round(float(r["abs_r"]), 4)}
        for _, r in top.iterrows()
    ]


def _variance_summary(df: pd.DataFrame) -> dict[str, float]:
    num = df.select_dtypes(include=[np.number])
    return {col: round(float(num[col].var()), 4) for col in num.columns}


def _skewness_summary(df: pd.DataFrame) -> dict[str, float]:
    num = df.select_dtypes(include=[np.number])
    return {col: round(float(num[col].skew()), 4) for col in num.columns}


def _linear_regression_coef(x: pd.Series, y: pd.Series) -> float:
    idx = x.dropna().index.intersection(y.dropna().index)
    if len(idx) < 2:
        return 0.0
    xv = x.loc[idx].values.astype(float)
    yv = y.loc[idx].values.astype(float)
    xm, ym = xv.mean(), yv.mean()
    denom = float(np.sum((xv - xm) ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum((xv - xm) * (yv - ym)) / denom)


def _dataset_mini_summary(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=[np.number])
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": list(num.columns),
        "numeric_means_top5": {
            col: round(float(num[col].mean()), 4)
            for col in list(num.columns)[:5]
        },
    }


def _compute_key_drivers(df: pd.DataFrame, target_col: str) -> list[dict]:
    num = df.select_dtypes(include=[np.number])
    drivers = []
    for col in num.columns:
        if col == target_col:
            continue
        idx = df[col].dropna().index.intersection(df[target_col].dropna().index)
        if len(idx) < 2:
            continue
        corr_val = float(df[col].loc[idx].corr(df[target_col].loc[idx]))
        coef = _linear_regression_coef(df[col], df[target_col])
        drivers.append({
            "column":          col,
            "correlation":     round(corr_val, 4),
            "abs_correlation": round(abs(corr_val), 4),
            "regression_coef": round(coef, 6),
            "direction":       "positive" if corr_val >= 0 else "negative",
        })
    return sorted(drivers, key=lambda d: d["abs_correlation"], reverse=True)


def _compute_segment_intelligence(
    df: pd.DataFrame, cat_col: str, metric_col: str, top_n: int = 5
) -> dict:
    grouped = (
        df.groupby(cat_col, observed=True)[metric_col]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "avg", "count": "n", "std": "std_dev"})
    )
    grouped["avg"]     = grouped["avg"].round(4)
    grouped["std_dev"] = grouped["std_dev"].round(4)
    grouped = grouped.sort_values("avg", ascending=False)
    overall_mean = round(float(df[metric_col].mean()), 4)
    overall_std  = round(float(df[metric_col].std()),  4)
    return {
        "category_column":  cat_col,
        "metric_column":    metric_col,
        "total_segments":   int(len(grouped)),
        "overall_mean":     overall_mean,
        "overall_std":      overall_std,
        "top_performers":   grouped.head(top_n).to_dict(orient="records"),
        "bottom_performers":grouped.tail(top_n).to_dict(orient="records"),
    }


def _compute_anomalies(df: pd.DataFrame, method: str = "zscore") -> dict:
    num = df.select_dtypes(include=[np.number])
    report: dict[str, Any] = {}
    for col in num.columns:
        series = num[col].dropna()
        if len(series) < 4:
            continue
        if method == "zscore":
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            z     = ((series - mean) / std).abs()
            flagged = series[z > 3]
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = series[(series < lo) | (series > hi)]
        if not flagged.empty:
            report[col] = {
                "count":          int(len(flagged)),
                "sample_values":  [round(float(v), 4) for v in flagged.values[:5]],
                "column_mean":    round(float(series.mean()), 4),
                "column_std":     round(float(series.std()),  4),
                "pct_anomalous":  round(len(flagged) / len(series) * 100, 2),
            }
    total = sum(v["count"] for v in report.values())
    return {
        "method":                 method,
        "total_anomalous_values": total,
        "columns_with_anomalies": len(report),
        "details":                report,
    }


def _compute_trend_intelligence(
    df: pd.DataFrame, date_col: str, metric_col: str, freq: str = "ME"
) -> dict:
    tmp = df[[date_col, metric_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, metric_col]).set_index(date_col).sort_index()
    agg = tmp[metric_col].resample(freq).mean().dropna()
    if len(agg) < 2:
        return {"error": "Insufficient time-series data after resampling."}
    periods = list(range(len(agg)))
    coef    = _linear_regression_coef(pd.Series(periods, index=agg.index), agg)
    pct_chg = round(
        float((agg.iloc[-1] - agg.iloc[0]) / abs(agg.iloc[0]) * 100)
        if agg.iloc[0] != 0 else 0.0, 2,
    )
    peak    = str(agg.idxmax().date() if hasattr(agg.idxmax(), "date") else agg.idxmax())
    trough  = str(agg.idxmin().date() if hasattr(agg.idxmin(), "date") else agg.idxmin())
    series_sample = {
        str(k.date() if hasattr(k, "date") else k): round(float(v), 4)
        for k, v in agg.tail(12).items()
    }
    return {
        "date_column":       date_col,
        "metric_column":     metric_col,
        "frequency":         freq,
        "periods_analysed":  int(len(agg)),
        "trend_slope":       round(float(coef), 6),
        "trend_direction":   "upward" if coef > 0 else "downward",
        "total_pct_change":  pct_chg,
        "peak_period":       peak,
        "trough_period":     trough,
        "first_value":       round(float(agg.iloc[0]),  4),
        "last_value":        round(float(agg.iloc[-1]), 4),
        "recent_12_periods": series_sample,
    }


def _compute_risk_profile(df: pd.DataFrame, anomaly_data: dict) -> dict:
    num      = df.select_dtypes(include=[np.number])
    skewness = _skewness_summary(df)
    volatility: dict[str, float] = {}
    for col in num.columns:
        m = float(num[col].mean())
        s = float(num[col].std())
        volatility[col] = round(abs(s / m) if m != 0 else 0.0, 4)
    missing = {
        col: round(float(df[col].isna().mean() * 100), 2)
        for col in df.columns if df[col].isna().any()
    }
    high_skew   = [c for c, v in skewness.items() if abs(v) > 2]
    high_vol    = sorted(volatility, key=lambda c: volatility[c], reverse=True)[:3]
    return {
        "high_skew_columns":       high_skew,
        "skewness_values":         {c: skewness[c] for c in high_skew},
        "high_volatility_columns": high_vol,
        "volatility_scores":       {c: volatility[c] for c in high_vol},
        "columns_with_missing_data": missing,
        "total_missing_columns":   len(missing),
        "anomaly_summary": {
            "total_anomalous_values": anomaly_data.get("total_anomalous_values", 0),
            "columns_affected":       anomaly_data.get("columns_with_anomalies",  0),
        },
        "overall_risk_indicators": (
            len(high_skew)
            + len(missing)
            + anomaly_data.get("columns_with_anomalies", 0)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4 ·  ADDITIONAL PURE HELPERS  (new, no engine dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _primary_metric(df: pd.DataFrame) -> str | None:
    """Heuristic: highest-variance numeric column."""
    num = df.select_dtypes(include=[np.number])
    return str(num.var().idxmax()) if not num.empty else None


def _dataset_full_profile(df: pd.DataFrame) -> dict:
    num  = df.select_dtypes(include=[np.number])
    cats = df.select_dtypes(include=["object", "category"])
    pm   = _primary_metric(df)
    desc: dict = {}
    for col in num.columns[:8]:
        s = num[col].dropna()
        desc[col] = {
            "mean": round(float(s.mean()), 4),
            "std":  round(float(s.std()),  4),
            "min":  round(float(s.min()),  4),
            "max":  round(float(s.max()),  4),
            "skew": round(float(s.skew()), 4),
        }
    return {
        "rows":                  int(len(df)),
        "columns":               int(df.shape[1]),
        "numeric_columns":       list(num.columns),
        "categorical_columns":   list(cats.columns),
        "primary_metric":        pm,
        "missing_pct":           round(float(df.isna().mean().mean() * 100), 2),
        "duplicate_pct":         round(float(df.duplicated().mean() * 100), 2),
        "descriptive_stats":     desc,
        "segment_cardinalities": {c: int(df[c].nunique()) for c in list(cats.columns)[:5]},
    }


def _detect_hidden_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Pure-statistics hidden pattern detector.
    Returns a list of findings categorised as Structural / Cyclical / Noise.
    """
    patterns: list[dict] = []
    num = df.select_dtypes(include=[np.number])

    for col in num.columns:
        s = num[col].dropna()
        if len(s) < 4:
            continue
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        cv   = abs(float(s.std() / s.mean())) if s.mean() != 0 else 0.0

        # Heavy-tailed / leptokurtic → Structural
        if abs(kurt) > 5:
            patterns.append({
                "type":     "Structural",
                "column":   col,
                "metric":   f"Kurtosis = {kurt:.2f}",
                "detail":   (
                    f"'{col}' shows heavy-tailed behaviour (kurtosis {kurt:.2f}), "
                    "indicating infrequent extreme values consistent with structural "
                    "business events rather than random noise."
                ),
                "severity": "high" if abs(kurt) > 10 else "medium",
            })

        # Strong skew → Structural (severe) or Noise (moderate)
        if abs(skew) > 2:
            pat_type = "Structural" if abs(skew) > 4 else "Noise"
            patterns.append({
                "type":     pat_type,
                "column":   col,
                "metric":   f"Skewness = {skew:.2f}",
                "detail":   (
                    f"'{col}' is {'strongly' if abs(skew) > 4 else 'moderately'} "
                    f"{'right' if skew > 0 else 'left'}-skewed ({skew:.2f}), "
                    "suggesting a natural floor or ceiling in the underlying process."
                ),
                "severity": "high" if abs(skew) > 4 else "medium",
            })

        # High coefficient of variation → Noise
        if cv > 1.5:
            patterns.append({
                "type":     "Noise",
                "column":   col,
                "metric":   f"CV = {cv:.2f}",
                "detail":   (
                    f"'{col}' has a coefficient of variation of {cv:.2f}. "
                    "Standard deviation exceeds the mean, indicating high "
                    "operational variability or data quality concerns."
                ),
                "severity": "high" if cv > 3 else "medium",
            })

        # Compressed IQR relative to range → periodic spikes (Cyclical)
        iqr        = float(s.quantile(0.75) - s.quantile(0.25))
        full_range = float(s.max() - s.min())
        if full_range > 0 and (iqr / full_range) < 0.10 and len(s) > 20:
            patterns.append({
                "type":     "Cyclical",
                "column":   col,
                "metric":   f"IQR/Range = {iqr / full_range:.3f}",
                "detail":   (
                    f"'{col}' has a compressed IQR relative to its full range "
                    f"(ratio {iqr / full_range:.3f}), consistent with periodic "
                    "spikes or cyclical demand patterns."
                ),
                "severity": "medium",
            })

    # Near-perfect cross-column correlations → multicollinearity (Structural)
    if num.shape[1] >= 2:
        corr_mat = num.corr().abs()
        for i in range(len(corr_mat.columns)):
            for j in range(i + 1, len(corr_mat.columns)):
                r = float(corr_mat.iloc[i, j])
                if r > 0.92:
                    ca, cb = corr_mat.columns[i], corr_mat.columns[j]
                    patterns.append({
                        "type":     "Structural",
                        "column":   f"{ca} × {cb}",
                        "metric":   f"r = {r:.4f}",
                        "detail":   (
                            f"'{ca}' and '{cb}' are near-perfectly correlated (r={r:.4f}). "
                            "This redundancy may indicate a derived relationship and will "
                            "distort driver analysis coefficients."
                        ),
                        "severity": "high",
                    })

    sev_order = {"high": 0, "medium": 1, "low": 2}
    patterns.sort(key=lambda p: (sev_order.get(p["severity"], 2), p["type"]))
    return patterns[:12]


def _build_risk_watchlist(df: pd.DataFrame) -> list[dict]:
    """
    Enumerate data-level risks with severity classification.
    Returns a flat list sorted critical → low.
    """
    risks: list[dict] = []
    num = df.select_dtypes(include=[np.number])

    # Concentration risk (categorical dominance)
    for col in df.select_dtypes(include=["object", "category"]).columns[:5]:
        top_share = float(df[col].value_counts(normalize=True).iloc[0])
        if top_share > 0.50:
            sev = "critical" if top_share > 0.80 else "high" if top_share > 0.65 else "medium"
            risks.append({
                "risk":     "Concentration Risk",
                "column":   col,
                "value":    f"{top_share:.1%} in top segment",
                "severity": sev,
                "detail":   (
                    f"The dominant category in '{col}' holds {top_share:.1%} of "
                    "observations, reducing representativeness and amplifying "
                    "sensitivity to that segment's behaviour."
                ),
            })

    # High volatility (CV > 1)
    for col in num.columns:
        s = num[col].dropna()
        if s.mean() == 0 or len(s) < 4:
            continue
        cv = abs(float(s.std() / s.mean()))
        if cv > 1.0:
            sev = "critical" if cv > 3 else "high" if cv > 2 else "medium"
            risks.append({
                "risk":     "High Volatility",
                "column":   col,
                "value":    f"CV = {cv:.2f}",
                "severity": sev,
                "detail":   (
                    f"'{col}' has CV={cv:.2f}. High-volatility metrics produce wide "
                    "prediction intervals and are unreliable as primary KPI numerators."
                ),
            })

    # Anomaly pressure (z > 3)
    for col in num.columns:
        s = num[col].dropna()
        if s.std() == 0 or len(s) < 4:
            continue
        n_anom = int(((s - s.mean()).abs() > 3 * s.std()).sum())
        if n_anom > 0:
            pct  = round(n_anom / len(s) * 100, 2)
            z_mx = round(float(((s - s.mean()) / s.std()).abs().max()), 1)
            risks.append({
                "risk":     "Anomaly Pressure",
                "column":   col,
                "value":    f"{n_anom} values ({pct}%), max-z={z_mx}",
                "severity": "high" if pct > 5 else "medium",
                "detail":   (
                    f"'{col}' contains {n_anom} outliers beyond z=3 ({pct}% of values, "
                    f"max z={z_mx}). These distort mean-based aggregations and "
                    "inflate model error during training."
                ),
            })

    # Skewed distributions
    for col in num.columns:
        s    = num[col].dropna()
        skew = float(s.skew())
        if abs(skew) > 3:
            sev = "high" if abs(skew) > 6 else "medium"
            risks.append({
                "risk":     "Skewed Distribution",
                "column":   col,
                "value":    f"Skew = {skew:.2f}",
                "severity": sev,
                "detail":   (
                    f"'{col}' is heavily {'right' if skew > 0 else 'left'}-skewed "
                    f"(skew={skew:.2f}). Mean is unreliable; consider median-based "
                    "statistics or a log transform before analysis."
                ),
            })

    # Missing data
    for col in df.columns:
        miss = float(df[col].isna().mean() * 100)
        if miss > 5:
            sev = "critical" if miss > 30 else "high" if miss > 15 else "medium"
            risks.append({
                "risk":     "Missing Data",
                "column":   col,
                "value":    f"{miss:.1f}% null",
                "severity": sev,
                "detail":   (
                    f"'{col}' is {miss:.1f}% missing. Validate imputation before "
                    "using this column in segmentation or predictive models."
                ),
            })

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    risks.sort(key=lambda r: sev_order.get(r["severity"], 3))
    return risks[:15]


# ═══════════════════════════════════════════════════════════════════════════════
# 5 ·  CSS
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown(
        f"""<style>
/* ── Card wrapper ──────────────────────────────────────────────────── */
.dil-card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}}
.dil-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: var(--dil-accent, {T['a1']});
    border-radius: 16px 16px 0 0;
    opacity: .8;
}}

/* ── Section label ─────────────────────────────────────────────────── */
.dil-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--dil-accent, {T['a1']});
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 7px;
}}
.dil-label::before {{
    content: '';
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--dil-accent, {T['a1']});
    flex-shrink: 0;
}}

/* ── Title ─────────────────────────────────────────────────────────── */
.dil-title {{
    font-size: 17px;
    font-weight: 700;
    color: {T['text']};
    margin-bottom: 20px;
    letter-spacing: -.01em;
    line-height: 1.3;
}}

/* ── Narrative block ───────────────────────────────────────────────── */
.dil-narrative {{
    font-size: 13px;
    line-height: 1.88;
    color: #CBD5E1;
    margin-top: 16px;
    padding: 18px 22px;
    background: {T['bg']};
    border: 1px solid {T['border']};
    border-left: 3px solid var(--dil-accent, {T['a1']});
    border-radius: 0 10px 10px 0;
    white-space: pre-wrap;
}}

/* ── Stat tile ─────────────────────────────────────────────────────── */
.dil-stat {{
    text-align: center;
    padding: 14px 6px;
}}
.dil-stat-val {{
    font-size: 22px;
    font-weight: 700;
    color: {T['text']};
    line-height: 1;
}}
.dil-stat-lbl {{
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: {T['muted']};
    margin-top: 5px;
}}

/* ── Key-value row ─────────────────────────────────────────────────── */
.dil-kv {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 12px;
    border-bottom: 1px solid {T['border']}55;
}}
.dil-kv:last-child {{ border-bottom: none; }}

/* ── Driver bar row ────────────────────────────────────────────────── */
.dil-bar-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid {T['border']}44;
    font-size: 12px;
}}
.dil-bar-row:last-child {{ border-bottom: none; }}
.dil-bar-track {{
    flex: 1;
    height: 6px;
    background: {T['border']};
    border-radius: 3px;
    overflow: hidden;
}}
.dil-bar-fill {{
    height: 100%;
    border-radius: 3px;
}}

/* ── Pill badge ────────────────────────────────────────────────────── */
.dil-pill {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .06em;
    margin: 2px 3px 2px 0;
}}

/* ── Pattern finding card ──────────────────────────────────────────── */
.dil-pat-card {{
    background: {T['card2']};
    border: 1px solid {T['border']};
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}}

/* ── Risk row ──────────────────────────────────────────────────────── */
.dil-risk-row {{
    display: grid;
    grid-template-columns: 145px 1fr auto;
    align-items: start;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid {T['border']}55;
    font-size: 12px;
}}
.dil-risk-row:last-child {{ border-bottom: none; }}

/* ── Opportunity item ──────────────────────────────────────────────── */
.dil-opp-item {{
    display: flex;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid {T['border']}44;
    align-items: flex-start;
}}
.dil-opp-item:last-child {{ border-bottom: none; }}
.dil-opp-num {{
    font-size: 18px;
    font-weight: 700;
    color: {T['a3']};
    min-width: 24px;
    flex-shrink: 0;
    line-height: 1.3;
}}
</style>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6 ·  UI PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def _card_open(accent: str) -> None:
    st.markdown(
        f'<div class="dil-card" style="--dil-accent:{accent}">',
        unsafe_allow_html=True,
    )


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _sec_label(text: str, accent: str) -> None:
    st.markdown(
        f'<div class="dil-label" style="--dil-accent:{accent};">{text}</div>',
        unsafe_allow_html=True,
    )


def _title(text: str) -> None:
    st.markdown(f'<div class="dil-title">{text}</div>', unsafe_allow_html=True)


def _narrative(text: str, accent: str) -> None:
    st.markdown(
        f'<div class="dil-narrative" style="--dil-accent:{accent};">{text}</div>',
        unsafe_allow_html=True,
    )


def _pill(label: str, color: str) -> str:
    return (
        f'<span class="dil-pill" '
        f'style="background:{color}1A;border:1px solid {color}44;color:{color};">'
        f'{label}</span>'
    )


def _kv(label: str, value: str, value_color: str = "") -> None:
    vc = f"color:{value_color};font-weight:600;" if value_color else f"color:{T['text']};font-weight:600;"
    st.markdown(
        f'<div class="dil-kv">'
        f'<span style="color:{T["muted"]}">{label}</span>'
        f'<span style="{vc}">{value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _no_engine() -> None:
    st.markdown(
        f'<div style="color:{T["muted"]};font-size:12px;padding:6px 0;">'
        "AI engine not configured — structured analysis stored, narrative unavailable."
        "</div>",
        unsafe_allow_html=True,
    )


def _divider() -> None:
    st.markdown(
        f'<div style="height:1px;background:{T["border"]};margin:16px 0;"></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7 ·  SECTION 1 — DATASET PULSE
# ═══════════════════════════════════════════════════════════════════════════════

def _render_dataset_pulse(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a1"])
    _sec_label("01 — Dataset Pulse", T["a1"])
    _title("Executive Dataset Characterisation")

    profile = _dataset_full_profile(df)

    # ── Stat tiles ──────────────────────────────────────────────────────────
    cols = st.columns(5)
    for c, (val, lbl) in zip(cols, [
        (f"{profile['rows']:,}",                           "Rows"),
        (str(profile["columns"]),                          "Columns"),
        (str(len(profile["numeric_columns"])),             "Numeric"),
        (str(len(profile["categorical_columns"])),         "Categorical"),
        (f"{profile['missing_pct']:.1f}%",                 "Missing"),
    ]):
        with c:
            st.markdown(
                f'<div class="dil-stat">'
                f'<div class="dil-stat-val">{val}</div>'
                f'<div class="dil-stat-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _divider()

    # ── Key facts ───────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        _kv("Primary Metric",
            profile["primary_metric"] or "—")
        _kv("Numeric Columns",
            ", ".join(profile["numeric_columns"][:4]) or "—")
        _kv("Missing Data",
            f"{profile['missing_pct']:.2f}%",
            T["a4"] if profile["missing_pct"] > 5 else T["a3"])
    with c2:
        _kv("Segment Variables",
            ", ".join(profile["categorical_columns"][:3]) or "—")
        _kv("Duplicate Rows",
            f"{profile['duplicate_pct']:.2f}%",
            T["a4"] if profile["duplicate_pct"] > 2 else T["a3"])
        if profile["segment_cardinalities"]:
            top_cat, top_n = next(iter(profile["segment_cardinalities"].items()))
            _kv("Largest Segment Var", f"{top_cat} ({top_n} unique values)")

    # ── AI narrative ────────────────────────────────────────────────────────
    if ai_engine is not None:
        if st.button("Generate Dataset Pulse", key="btn_pulse"):
            prompt = (
                "You are a senior data analyst preparing a briefing for a C-suite audience.\n\n"
                "Dataset profile:\n"
                f"{_safe_json(profile, max_chars=2800)}\n\n"
                "Write a concise dataset pulse in 4-6 sentences covering:\n"
                "1. What this dataset most likely represents, inferred from its column names and structure.\n"
                "2. The primary metric and what it measures in commercial context.\n"
                "3. The main segmentation variables and their analytical value.\n"
                "4. Any immediate quality concerns visible in the statistics.\n\n"
                "Write as a single executive briefing paragraph. "
                "No bullet points. No headers. Address the reader directly."
            )
            with st.spinner("Characterising dataset…"):
                result = ai_engine.generate(prompt, max_tokens=320)
            st.session_state["ai_dataset_pulse"] = {
                "structured":  profile,
                "explanation": result,
            }
    else:
        _no_engine()
        if "ai_dataset_pulse" not in st.session_state:
            st.session_state["ai_dataset_pulse"] = {
                "structured": profile, "explanation": None,
            }

    if "ai_dataset_pulse" in st.session_state:
        exp = st.session_state["ai_dataset_pulse"].get("explanation")
        if exp:
            _narrative(exp, T["a1"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 8 ·  SECTION 2 — METRIC DRIVER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_driver_analysis(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a2"])
    _sec_label("02 — Metric Driver Analysis", T["a2"])
    _title("Variable Influence on Primary Metric")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.info("At least two numeric columns are required for driver analysis.")
        _card_close()
        return

    pm     = _primary_metric(df)
    target = st.selectbox(
        "Primary metric (target)",
        num_cols,
        index=num_cols.index(pm) if pm in num_cols else 0,
        key="dil_driver_target",
    )

    if st.button("Run Driver Analysis", key="btn_driver"):
        drivers = _compute_key_drivers(df, target)

        if not drivers:
            st.info("No sufficient correlations found for driver analysis.")
            _card_close()
            return

        # ── Influence bar chart ──────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.12em;'
            f'text-transform:uppercase;color:{T["muted"]};margin-bottom:10px;">'
            f'RANKED INFLUENCE — TARGET: {target}</div>',
            unsafe_allow_html=True,
        )

        top_corr  = drivers[0]["abs_correlation"] if drivers else 1.0

        for d in drivers[:8]:
            bar_w   = int(d["abs_correlation"] / top_corr * 100) if top_corr > 0 else 0
            d_color = T["a3"] if d["direction"] == "positive" else T["a5"]
            coef_s  = f"{d['regression_coef']:+.4f}"
            st.markdown(
                f'<div class="dil-bar-row">'
                f'<div style="width:150px;color:{T["text"]};overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{d["column"]}</div>'
                f'<div class="dil-bar-track">'
                f'<div class="dil-bar-fill" style="width:{bar_w}%;background:{d_color};"></div>'
                f'</div>'
                f'<div style="width:62px;color:{d_color};font-weight:700;'
                f'text-align:right;">r={d["correlation"]:+.3f}</div>'
                f'<div style="width:72px;color:{T["muted"]};'
                f'text-align:right;font-size:10px;">β={coef_s}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Legend ───────────────────────────────────────────────────────
        st.markdown(
            f'<div style="margin-top:8px;font-size:10px;color:{T["muted"]};">'
            f'{_pill("▲ positive", T["a3"])}'
            f'{_pill("▼ negative", T["a5"])}'
            f'&nbsp; r = Pearson correlation &nbsp;|&nbsp; β = regression coefficient'
            f'</div>',
            unsafe_allow_html=True,
        )

        structured = {
            "target":          target,
            "total_evaluated": len(drivers),
            "top_drivers":     drivers[:6],
        }

        if ai_engine is not None:
            prompt = (
                "You are a senior quantitative analyst writing for a C-suite audience.\n\n"
                "Driver analysis results:\n"
                f"{_safe_json(structured, max_chars=2500)}\n\n"
                "In 4-5 sentences explain:\n"
                "1. Which 2-3 variables most strongly drive the target and in what direction.\n"
                "2. What the regression coefficients reveal about practical influence magnitude.\n"
                "3. A specific caveat about correlation vs causation relevant to these columns.\n\n"
                "Reference actual column names and values. "
                "No bullet points. One diagnostic paragraph."
            )
            with st.spinner("Analysing drivers…"):
                explanation = ai_engine.generate(prompt, max_tokens=320)
        else:
            explanation = None
            _no_engine()

        st.session_state["ai_driver_analysis"] = {
            "structured":  structured,
            "explanation": explanation,
        }

    if "ai_driver_analysis" in st.session_state:
        exp = st.session_state["ai_driver_analysis"].get("explanation")
        if exp:
            _narrative(exp, T["a2"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 9 ·  SECTION 3 — HIDDEN PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _render_hidden_patterns(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a6"])
    _sec_label("03 — Hidden Pattern Detection", T["a6"])
    _title("Structural, Cyclical and Noise Deviations")

    if st.button("Detect Hidden Patterns", key="btn_patterns"):
        patterns = _detect_hidden_patterns(df)

        if not patterns:
            st.success("No significant statistical deviations detected in this dataset.")
            st.session_state["ai_hidden_patterns"] = {
                "structured": {"pattern_count": 0, "type_summary": {}},
                "patterns":   [],
                "explanation": None,
            }
            _card_close()
            return

        # ── Type summary pills ───────────────────────────────────────────
        counts: dict[str, int] = {}
        for p in patterns:
            counts[p["type"]] = counts.get(p["type"], 0) + 1

        pills = "".join(
            _pill(f"{t} ({n})", _PAT.get(t, T["muted"]))
            for t, n in counts.items()
        )
        st.markdown(f'<div style="margin-bottom:16px;">{pills}</div>', unsafe_allow_html=True)

        # ── Finding cards ────────────────────────────────────────────────
        for p in patterns:
            p_color  = _PAT.get(p["type"], T["muted"])
            sv_color = _SEV.get(p["severity"], T["muted"])
            st.markdown(
                f'<div class="dil-pat-card" '
                f'style="border-left:3px solid {p_color};">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:5px;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="background:{p_color}1A;border:1px solid {p_color}44;'
                f'color:{p_color};border-radius:4px;padding:2px 8px;'
                f'font-size:10px;font-weight:700;">{p["type"]}</span>'
                f'<span style="font-size:12px;font-weight:600;color:{T["text"]};">'
                f'{p["column"]}</span>'
                f'</div>'
                f'<span style="font-size:10px;font-weight:700;color:{sv_color};'
                f'text-transform:uppercase;letter-spacing:.08em;">'
                f'{p["severity"]}</span>'
                f'</div>'
                f'<div style="font-size:11px;color:{T["muted"]};margin-bottom:3px;">'
                f'{p["metric"]}</div>'
                f'<div style="font-size:12px;color:#CBD5E1;line-height:1.6;">'
                f'{p["detail"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        structured = {
            "pattern_count": len(patterns),
            "type_summary":  counts,
            "findings": [
                {"type": p["type"], "column": p["column"],
                 "metric": p["metric"], "severity": p["severity"]}
                for p in patterns[:8]
            ],
        }

        if ai_engine is not None:
            prompt = (
                "You are a senior statistical analyst writing for a C-suite audience.\n\n"
                "Hidden pattern detection results:\n"
                f"{_safe_json(structured, max_chars=2500)}\n\n"
                "In 4-5 sentences explain:\n"
                "1. The dominant pattern type and what it implies about the data-generating process.\n"
                "2. Which finding poses the greatest risk to analytical conclusions and why.\n"
                "3. What the structural vs noise split reveals about operational consistency.\n\n"
                "Be diagnostic. Reference specific column names. "
                "No bullet points. One cohesive paragraph."
            )
            with st.spinner("Analysing patterns…"):
                explanation = ai_engine.generate(prompt, max_tokens=320)
        else:
            explanation = None
            _no_engine()

        st.session_state["ai_hidden_patterns"] = {
            "structured":  structured,
            "patterns":    patterns,
            "explanation": explanation,
        }

    if "ai_hidden_patterns" in st.session_state:
        exp = st.session_state["ai_hidden_patterns"].get("explanation")
        if exp:
            _narrative(exp, T["a6"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 10 · SECTION 4 — SEGMENT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _render_segment_intelligence(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a3"])
    _sec_label("04 — Segment Intelligence", T["a3"])
    _title("Performance Across Categories")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        st.info("No categorical columns found. Segment intelligence requires at least one category variable.")
        _card_close()
        return
    if not num_cols:
        st.info("No numeric columns found for segment measurement.")
        _card_close()
        return

    pm   = _primary_metric(df)
    c1, c2 = st.columns(2)
    with c1:
        metric_col = st.selectbox(
            "Metric", num_cols,
            index=num_cols.index(pm) if pm in num_cols else 0,
            key="dil_seg_metric",
        )
    with c2:
        cat_col = st.selectbox("Segment by", cat_cols, key="dil_seg_cat")

    if st.button("Analyse Segments", key="btn_seg"):
        if df[cat_col].nunique() > 120:
            st.warning(
                f"'{cat_col}' has {df[cat_col].nunique()} unique values. "
                "Consider a lower-cardinality column for cleaner insights."
            )

        seg = _compute_segment_intelligence(df, cat_col, metric_col)

        # ── Summary metrics ──────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Segments",    seg["total_segments"])
        m2.metric("Overall Mean",      f"{seg['overall_mean']:.4f}")
        m3.metric(
            "Performance Gap",
            f"{seg['top_performers'][0]['avg'] - seg['bottom_performers'][-1]['avg']:+.4f}"
            if seg["top_performers"] and seg["bottom_performers"] else "—",
            delta_color="off",
        )

        _divider()

        # ── Top / bottom tables ──────────────────────────────────────────
        c_top, c_bot = st.columns(2)

        with c_top:
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;'
                f'color:{T["a3"]};text-transform:uppercase;margin-bottom:8px;">'
                f'TOP PERFORMERS</div>',
                unsafe_allow_html=True,
            )
            for row in seg["top_performers"][:5]:
                name = str(row.get(cat_col, "—"))
                val  = float(row.get("avg", 0))
                dev  = round(
                    (val - seg["overall_mean"]) / seg["overall_mean"] * 100, 1
                ) if seg["overall_mean"] != 0 else 0.0
                _kv(
                    name[:30],
                    f"{val:.4f}  ({dev:+.1f}%)",
                    T["a3"] if dev >= 0 else T["a5"],
                )

        with c_bot:
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;'
                f'color:{T["a5"]};text-transform:uppercase;margin-bottom:8px;">'
                f'BOTTOM PERFORMERS</div>',
                unsafe_allow_html=True,
            )
            for row in seg["bottom_performers"][:3]:
                name = str(row.get(cat_col, "—"))
                val  = float(row.get("avg", 0))
                dev  = round(
                    (val - seg["overall_mean"]) / seg["overall_mean"] * 100, 1
                ) if seg["overall_mean"] != 0 else 0.0
                _kv(
                    name[:30],
                    f"{val:.4f}  ({dev:+.1f}%)",
                    T["a3"] if dev >= 0 else T["a5"],
                )

        structured = {
            "category_column":  cat_col,
            "metric_column":    metric_col,
            "total_segments":   seg["total_segments"],
            "overall_mean":     seg["overall_mean"],
            "overall_std":      seg["overall_std"],
            "top_performers":   seg["top_performers"][:5],
            "bottom_performers":seg["bottom_performers"][:3],
        }

        if ai_engine is not None:
            prompt = (
                "You are a senior business intelligence analyst.\n\n"
                "Segment performance data:\n"
                f"{_safe_json(structured, max_chars=2500)}\n\n"
                "In 4-5 sentences explain:\n"
                "1. Whether the performance gap between top and bottom segments is commercially significant.\n"
                "2. What the numbers reveal about why top performers outperform.\n"
                "3. Whether within-segment standard deviation suggests consistency or volatility.\n"
                "4. One specific action the business should take based on this segmentation.\n\n"
                "Use actual segment names and metric values. "
                "No bullet points. One diagnostic paragraph."
            )
            with st.spinner("Analysing segments…"):
                explanation = ai_engine.generate(prompt, max_tokens=320)
        else:
            explanation = None
            _no_engine()

        st.session_state["ai_segment_intelligence"] = {
            "structured":  structured,
            "explanation": explanation,
        }

    if "ai_segment_intelligence" in st.session_state:
        exp = st.session_state["ai_segment_intelligence"].get("explanation")
        if exp:
            _narrative(exp, T["a3"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 11 · SECTION 5 — RISK WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

def _render_risk_watchlist(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a5"])
    _sec_label("05 — Risk Watchlist", T["a5"])
    _title("Data Risk Assessment")

    if st.button("Generate Risk Watchlist", key="btn_risk_wl"):
        risks = _build_risk_watchlist(df)

        if not risks:
            st.success("No significant data risks detected in this dataset.")
        else:
            # ── Severity summary pills ───────────────────────────────────
            sev_counts: dict[str, int] = {}
            for r in risks:
                sev_counts[r["severity"]] = sev_counts.get(r["severity"], 0) + 1

            pills = "".join(
                _pill(f"{s.upper()} ({n})", _SEV.get(s, T["muted"]))
                for s, n in sev_counts.items()
            )
            st.markdown(f'<div style="margin-bottom:16px;">{pills}</div>', unsafe_allow_html=True)

            # ── Risk rows ────────────────────────────────────────────────
            for r in risks:
                sc = _SEV.get(r["severity"], T["muted"])
                st.markdown(
                    f'<div class="dil-risk-row">'
                    # Left: risk name + column
                    f'<div>'
                    f'<div style="font-size:11px;font-weight:700;color:{sc};">'
                    f'{r["risk"]}</div>'
                    f'<div style="font-size:10px;color:{T["muted"]};margin-top:2px;">'
                    f'{r["column"]}</div>'
                    f'</div>'
                    # Middle: detail
                    f'<div style="font-size:12px;color:#CBD5E1;line-height:1.55;">'
                    f'{r["detail"]}</div>'
                    # Right: value badge
                    f'<div style="text-align:right;white-space:nowrap;">'
                    f'<span style="font-size:10px;font-weight:700;'
                    f'background:{sc}1A;border:1px solid {sc}44;'
                    f'color:{sc};padding:2px 8px;border-radius:4px;">'
                    f'{r["value"]}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        structured = {
            "total_risks":      len(risks),
            "severity_summary": sev_counts if risks else {},
            "risks": [
                {"risk": r["risk"], "column": r["column"],
                 "value": r["value"], "severity": r["severity"]}
                for r in risks[:10]
            ],
        }

        if ai_engine is not None and risks:
            prompt = (
                "You are a chief data officer writing a risk assessment for a leadership team.\n\n"
                "Data risk watchlist:\n"
                f"{_safe_json(structured, max_chars=2500)}\n\n"
                "In 4-5 sentences:\n"
                "1. State the overall risk posture (low / moderate / high) and justify it.\n"
                "2. Identify the single highest-priority risk and explain why it must be addressed first.\n"
                "3. Describe the downstream analytical impact if these risks are left unaddressed.\n\n"
                "Be direct. Use specific column names and statistics. "
                "No bullet points. One formal risk paragraph."
            )
            with st.spinner("Assessing risks…"):
                explanation = ai_engine.generate(prompt, max_tokens=320)
        else:
            explanation = None
            if ai_engine is None:
                _no_engine()

        st.session_state["ai_risk_watchlist"] = {
            "structured":  structured,
            "risks":       risks if risks else [],
            "explanation": explanation,
        }

    if "ai_risk_watchlist" in st.session_state:
        exp = st.session_state["ai_risk_watchlist"].get("explanation")
        if exp:
            _narrative(exp, T["a5"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 12 · SECTION 6 — STRATEGIC OPPORTUNITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _render_strategic_opportunities(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    _card_open(T["a3"])
    _sec_label("06 — Strategic Opportunities", T["a3"])
    _title("Actionable Intelligence from Pattern Evidence")

    if st.button("Generate Opportunities", key="btn_opps"):
        pm = _primary_metric(df)

        # ── Pull cached artefacts, fall back to fresh computation ────────
        raw_drivers = (
            st.session_state.get("ai_driver_analysis", {})
            .get("structured", {})
            .get("top_drivers", [])
        )
        raw_risks = (
            st.session_state.get("ai_risk_watchlist", {})
            .get("risks", [])
        )
        raw_seg = (
            st.session_state.get("ai_segment_intelligence", {})
            .get("structured", {})
        )

        drivers = raw_drivers if raw_drivers else _compute_key_drivers(df, pm)[:5] if pm else []
        risks   = raw_risks   if raw_risks   else _build_risk_watchlist(df)

        seg_ctx: dict = raw_seg
        if not seg_ctx:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols and pm:
                seg_ctx = _compute_segment_intelligence(df, cat_cols[0], pm)

        # ── Build opportunity context ────────────────────────────────────
        ctx = {
            "rows":           int(len(df)),
            "columns":        int(df.shape[1]),
            "primary_metric": pm,
            "top_drivers": [
                {"col": d["column"], "correlation": d["correlation"],
                 "direction": d["direction"]}
                for d in (drivers if drivers and isinstance(drivers[0], dict) else [])[:5]
            ],
            "segment_context": {
                "category":       seg_ctx.get("category_column"),
                "metric":         seg_ctx.get("metric_column"),
                "gap":            (
                    round(
                        seg_ctx["top_performers"][0]["avg"]
                        - seg_ctx["bottom_performers"][-1]["avg"], 4
                    )
                    if seg_ctx.get("top_performers") and seg_ctx.get("bottom_performers")
                    else None
                ),
                "top_segment": (
                    seg_ctx["top_performers"][0].get(seg_ctx.get("category_column", ""))
                    if seg_ctx.get("top_performers") else None
                ),
            } if seg_ctx else {},
            "key_risks": [
                {"risk": r["risk"], "column": r["column"], "severity": r["severity"]}
                for r in risks
                if r["severity"] in ("critical", "high")
            ][:4],
        }

        if ai_engine is not None:
            prompt = (
                "You are a strategic business intelligence consultant preparing an "
                "executive opportunity brief.\n\n"
                "Analytics context:\n"
                f"{_safe_json(ctx, max_chars=3000)}\n\n"
                "Identify exactly 5 concrete business opportunities supported by this data.\n"
                "Format: numbered list 1-5.\n"
                "Each item must follow this exact structure:\n"
                "SHORT NAME — Evidence referencing a specific statistic or column. "
                "One specific action the business should take.\n\n"
                "Do not invent facts not present in the context. "
                "Be commercially specific. "
                "Start directly with '1.' — no preamble, no closing sentence."
            )
            with st.spinner("Identifying opportunities…"):
                explanation = ai_engine.generate(prompt, max_tokens=500)
        else:
            explanation = None
            _no_engine()

        st.session_state["ai_opportunities"] = {
            "structured":  {"primary_metric": pm, "context": ctx},
            "explanation": explanation,
        }

    # ── Render stored opportunities ──────────────────────────────────────────
    if "ai_opportunities" in st.session_state:
        exp = st.session_state["ai_opportunities"].get("explanation", "")
        if exp:
            # Parse numbered list → styled opportunity cards
            items = re.split(r"\n(?=\d+\.)", exp.strip())
            items = [i.strip() for i in items if i.strip()]

            if len(items) >= 2:
                for item in items:
                    m = re.match(r"^(\d+)\.\s*(.*)", item, re.DOTALL)
                    if not m:
                        continue
                    num     = m.group(1)
                    content = m.group(2).strip()
                    parts   = re.split(r"\s*[—–-]\s*", content, maxsplit=1)
                    name = parts[0].replace("**", "").strip()
                    body = parts[1].replace("**", "").strip() if len(parts) > 1 else content.replace("**", "")
                    st.markdown(
                        f'<div class="dil-opp-item">'
                        f'<div class="dil-opp-num">{num}</div>'
                        f'<div>'
                        f'<div style="font-size:13px;font-weight:700;'
                        f'color:{T["text"]};margin-bottom:5px;">{name}</div>'
                        f'<div style="font-size:12px;color:#CBD5E1;'
                        f'line-height:1.65;">{body}</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                # Fallback: raw narrative block
                _narrative(exp, T["a3"])

    _card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# 13 · PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def render_ai_insight(df: pd.DataFrame, ai_engine: "AIEngine | None") -> None:
    """
    Main entry point for the AI Insight Lab (Tab 4).

    Parameters
    ----------
    df         : Active dataset (cleaned preferred, raw acceptable).
    ai_engine  : Injected AIEngine instance from app.py.
                 Pass None to render structured data without AI narratives.
    """
    _inject_css()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:28px;padding-bottom:18px;
                    border-bottom:1px solid {T['border']};">
            <div style="font-size:10px;font-weight:700;letter-spacing:.18em;
                        text-transform:uppercase;color:{T['a1']};margin-bottom:8px;">
                InsightX · Diagnostic Intelligence
            </div>
            <div style="font-size:26px;font-weight:700;color:{T['text']};
                        letter-spacing:-.02em;line-height:1.2;">
                AI Insight Lab
            </div>
            <div style="font-size:13px;color:{T['muted']};margin-top:8px;
                        line-height:1.7;max-width:620px;">
                A consulting-style diagnostic briefing that explains
                <em>why</em> patterns exist in your dataset.
                Forecasting and scenario modelling are handled by the Decision Lab.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("Upload and clean a dataset to activate the Diagnostic Intelligence module.")
        return

    # ── Six diagnostic sections ───────────────────────────────────────────────
    _render_dataset_pulse(df, ai_engine)
    _render_driver_analysis(df, ai_engine)
    _render_hidden_patterns(df, ai_engine)
    _render_segment_intelligence(df, ai_engine)
    _render_risk_watchlist(df, ai_engine)
    _render_strategic_opportunities(df, ai_engine)
    
    # ── NEXT NAVIGATION ──────────────────────────────────────
    st.divider()
    col_left, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Next: Report Studio →", type="primary", use_container_width=True):
           st.session_state["active_tab"] = "Report Studio"
           st.rerun()    