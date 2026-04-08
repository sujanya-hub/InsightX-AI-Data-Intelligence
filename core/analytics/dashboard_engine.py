"""
InsightX · Dashboard Engine
core/analytics/dashboard_engine.py

Executive-level KPIs, risk indices, growth signals, and auto-generated alerts.
Zero Streamlit. Zero LLM. Consumes EDAPayload + DetectionResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.analytics.eda_engine import EDAPayload
from core.analytics.universal_detector import DetectionResult


# ══════════════════════════════════════════════════════════════════════════════
# Payloads
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutiveKPIs:
    health_score: float             # 0–100 composite
    concentration_risk: float       # 0–1  (HHI-derived)
    volatility_score: float         # 0–1  (CV of primary metric)
    growth_signal: float            # % change over time (or 0)
    segment_dominance: float        # top-segment share %
    anomaly_pressure: float         # 0–1 derived from anomaly counts
    data_completeness: float        # 0–100


@dataclass
class Alert:
    severity: str       # "critical" | "warning" | "info"
    category: str       # "quality" | "risk" | "trend" | "anomaly"
    message: str
    metric: str
    value: float


@dataclass
class PerformanceMatrix:
    """Pivot of segment × time period for a heatmap."""
    pivot: pd.DataFrame     # index=segment, columns=period, values=metric
    segment_col: str
    metric_col: str
    time_col: str


@dataclass
class DashboardPayload:
    kpis: ExecutiveKPIs
    alerts: list[Alert]
    performance_matrix: Optional[PerformanceMatrix]
    top_performers: pd.DataFrame        # top 10 segments by metric
    bottom_performers: pd.DataFrame     # bottom 10 segments by metric
    ai_context: dict                    # safe structured context for LLM


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

class DashboardEngine:
    """
    Stateless executive dashboard analytics engine.

    Usage
    -----
    payload = DashboardEngine().compute(df, eda, detection)
    """

    # ── entry point ───────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        eda: EDAPayload,
        detection: DetectionResult,
    ) -> DashboardPayload:
        """
        Derive executive KPIs, alerts, and performance tables from EDA results.

        Parameters
        ----------
        df        : pd.DataFrame
        eda       : EDAPayload from EDAEngine
        detection : DetectionResult from UniversalDetector

        Returns
        -------
        DashboardPayload
        """
        kpis      = self._kpis(df, eda, detection)
        alerts    = self._alerts(eda, kpis, detection)
        perf_mat  = self._performance_matrix(df, detection)
        top, bot  = self._performer_tables(eda)
        ctx       = self._ai_context(kpis, eda, detection, alerts)

        return DashboardPayload(
            kpis=kpis,
            alerts=alerts,
            performance_matrix=perf_mat,
            top_performers=top,
            bottom_performers=bot,
            ai_context=ctx,
        )

    # ── KPIs ──────────────────────────────────────────────────────────────────

    def _kpis(
        self, df: pd.DataFrame, eda: EDAPayload, d: DetectionResult
    ) -> ExecutiveKPIs:
        q = eda.quality

        # health score: composite penalty
        penalty = (
            q.missing_pct   * 0.40
            + q.duplicate_pct * 0.25
            + q.outlier_pct   * 0.20
            + self._anomaly_pressure(eda) * 100 * 0.15
        )
        health = round(max(0.0, 100.0 - penalty), 1)

        conc = self._concentration_risk(eda)

        vol = 0.0
        if d.primary_metric and d.primary_metric in df.columns:
            s = df[d.primary_metric].dropna()
            if len(s) > 1:
                cv = s.std() / (abs(s.mean()) + 1e-9)
                vol = round(min(float(cv), 1.0), 4)

        growth = 0.0
        if eda.time_trend:
            growth = eda.time_trend.pct_change_overall

        seg_dom = 0.0
        if eda.segment_ranking and not eda.segment_ranking.top_10.empty:
            seg_dom = round(float(eda.segment_ranking.top_10["share_pct"].iloc[0]), 2)

        return ExecutiveKPIs(
            health_score=health,
            concentration_risk=round(conc, 4),
            volatility_score=vol,
            growth_signal=round(growth, 2),
            segment_dominance=seg_dom,
            anomaly_pressure=round(self._anomaly_pressure(eda), 4),
            data_completeness=q.completeness_score,
        )

    def _concentration_risk(self, eda: EDAPayload) -> float:
        if not eda.segment_ranking or eda.segment_ranking.top_10.empty:
            return 0.0
        shares = eda.segment_ranking.top_10["share_pct"].values / 100.0
        hhi = float(np.sum(shares ** 2))
        n = max(len(shares), 1)
        hhi_min = 1.0 / n
        return round((hhi - hhi_min) / max(1.0 - hhi_min, 1e-9), 4)

    def _anomaly_pressure(self, eda: EDAPayload) -> float:
        if not eda.anomalies or not eda.quality.row_count:
            return 0.0
        total = sum(a.count for a in eda.anomalies)
        return min(total / max(eda.quality.row_count, 1), 1.0)

    # ── alerts ────────────────────────────────────────────────────────────────

    def _alerts(
        self, eda: EDAPayload, kpis: ExecutiveKPIs, d: DetectionResult
    ) -> list[Alert]:
        alerts: list[Alert] = []
        q = eda.quality

        if q.missing_pct >= 20:
            alerts.append(Alert("critical", "quality", f"Missing data exceeds 20% ({q.missing_pct:.1f}%)", "missing_pct", q.missing_pct))
        elif q.missing_pct >= 10:
            alerts.append(Alert("warning", "quality", f"Missing data at {q.missing_pct:.1f}% — review imputation strategy", "missing_pct", q.missing_pct))

        if q.duplicate_pct >= 5:
            alerts.append(Alert("warning", "quality", f"Duplicate rows detected: {q.duplicate_pct:.1f}%", "duplicate_pct", q.duplicate_pct))

        if q.outlier_pct >= 15:
            alerts.append(Alert("critical", "quality", f"High outlier exposure: {q.outlier_pct:.1f}% of numeric cells", "outlier_pct", q.outlier_pct))

        if kpis.concentration_risk >= 0.6:
            alerts.append(Alert("critical", "risk", f"Concentration risk {kpis.concentration_risk:.2f} — top segment dominates portfolio", "concentration_risk", kpis.concentration_risk))
        elif kpis.concentration_risk >= 0.35:
            alerts.append(Alert("warning", "risk", f"Moderate concentration risk: {kpis.concentration_risk:.2f}", "concentration_risk", kpis.concentration_risk))

        if kpis.volatility_score >= 0.6:
            alerts.append(Alert("warning", "risk", f"High metric volatility (CV = {kpis.volatility_score:.2f})", "volatility_score", kpis.volatility_score))

        if eda.correlation.high_corr_pairs:
            n = len(eda.correlation.high_corr_pairs)
            alerts.append(Alert("info", "risk", f"{n} high-correlation feature pair(s) detected (|r| ≥ 0.75)", "high_corr_pairs", float(n)))

        if eda.time_trend:
            tt = eda.time_trend
            if tt.direction == "down" and abs(tt.pct_change_overall) >= 10:
                alerts.append(Alert("critical", "trend", f"Primary metric declining {tt.pct_change_overall:.1f}% over full period", "growth_signal", tt.pct_change_overall))
            elif tt.direction == "up":
                alerts.append(Alert("info", "trend", f"Positive growth signal: +{tt.pct_change_overall:.1f}%", "growth_signal", tt.pct_change_overall))

        if kpis.anomaly_pressure >= 0.05:
            alerts.append(Alert("warning", "anomaly", f"Anomaly pressure index {kpis.anomaly_pressure:.3f} — statistical outliers detected", "anomaly_pressure", kpis.anomaly_pressure))

        severe_skews = [(c, s) for c, s in eda.skew_ranking if abs(s) > 2]
        if len(severe_skews) >= 3:
            alerts.append(Alert("warning", "quality", f"{len(severe_skews)} columns with severe skew (|skew| > 2)", "skewed_cols", float(len(severe_skews))))

        alerts.sort(key=lambda a: {"critical": 0, "warning": 1, "info": 2}[a.severity])
        return alerts

    # ── performance matrix ────────────────────────────────────────────────────

    def _performance_matrix(
        self, df: pd.DataFrame, d: DetectionResult
    ) -> Optional[PerformanceMatrix]:
        seg    = d.segment_col
        metric = d.primary_metric
        t_col  = d.time_col
        if not seg or not metric or not t_col:
            return None
        if not all(c in df.columns for c in [seg, metric, t_col]):
            return None

        tmp = df[[seg, t_col, metric]].dropna().copy()
        tmp[t_col] = pd.to_datetime(tmp[t_col], errors="coerce")
        tmp = tmp.dropna(subset=[t_col])
        if tmp.empty:
            return None

        span = (tmp[t_col].max() - tmp[t_col].min()).days
        freq_char = "W" if span <= 365 else "Q"
        tmp["_period"] = tmp[t_col].dt.to_period(freq_char).astype(str)

        try:
            pivot = tmp.pivot_table(index=seg, columns="_period", values=metric, aggfunc="sum")
            pivot = pivot.dropna(how="all")
            if pivot.empty:
                return None
        except Exception:
            return None

        return PerformanceMatrix(pivot=pivot, segment_col=seg, metric_col=metric, time_col=t_col)

    # ── performer tables ──────────────────────────────────────────────────────

    def _performer_tables(self, eda: EDAPayload) -> tuple[pd.DataFrame, pd.DataFrame]:
        if eda.segment_ranking is None:
            return pd.DataFrame(), pd.DataFrame()
        return eda.segment_ranking.top_10.copy(), eda.segment_ranking.bottom_10.copy()

    # ── AI context ────────────────────────────────────────────────────────────

    def _ai_context(
        self,
        kpis: ExecutiveKPIs,
        eda: EDAPayload,
        d: DetectionResult,
        alerts: list[Alert],
    ) -> dict:
        """Build a safe, structured context dictionary for LLM consumption. No raw data."""
        return {
            "dataset_shape": {
                "rows": eda.quality.row_count,
                "columns": eda.quality.col_count,
                "numeric": len(eda.numeric_cols),
                "categorical": len(eda.categorical_cols),
            },
            "kpis": {
                "health_score": kpis.health_score,
                "data_completeness": kpis.data_completeness,
                "concentration_risk": kpis.concentration_risk,
                "volatility_score": kpis.volatility_score,
                "growth_signal": kpis.growth_signal,
                "anomaly_pressure": kpis.anomaly_pressure,
                "segment_dominance": kpis.segment_dominance,
            },
            "quality": {
                "missing_pct": eda.quality.missing_pct,
                "duplicate_pct": eda.quality.duplicate_pct,
                "outlier_pct": eda.quality.outlier_pct,
                "top_missing_cols": list(eda.quality.per_column_missing.items())[:5],
            },
            "correlation": {
                "strongest_pairs": eda.correlation.strongest_pairs[:5],
                "high_corr_count": len(eda.correlation.high_corr_pairs),
            },
            "skew": {"top_skewed": eda.skew_ranking[:5]},
            "time_trend": {
                "direction": eda.time_trend.direction if eda.time_trend else None,
                "pct_change": eda.time_trend.pct_change_overall if eda.time_trend else None,
                "metric": eda.time_trend.metric_col if eda.time_trend else None,
            },
            "alerts": [
                {"severity": a.severity, "message": a.message, "metric": a.metric}
                for a in alerts[:8]
            ],
            "primary_metric": d.primary_metric,
            "segment_col": d.segment_col,
        }
    