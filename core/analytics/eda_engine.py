from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from core.analytics.universal_detector import DetectionResult


# ══════════════════════════════════════════════════════════════════════════════
# Payloads
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    row_count: int
    col_count: int
    missing_pct: float
    duplicate_pct: float
    outlier_pct: float
    completeness_score: float       # 0-100
    per_column_missing: dict[str, float]    # col → %


@dataclass
class DistributionSummary:
    column: str
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    p5: float
    p95: float
    outlier_count: int


@dataclass
class CorrelationResult:
    matrix: pd.DataFrame
    strongest_pairs: list[tuple[str, str, float]]   # top-10 by abs(r)
    high_corr_pairs: list[tuple[str, str, float]]   # abs(r) >= 0.75


@dataclass
class SegmentRanking:
    segment_col: str
    metric_col: str
    top_10: pd.DataFrame        # columns: segment, value, share_pct
    bottom_10: pd.DataFrame


@dataclass
class TimeTrend:
    time_col: str
    metric_col: str
    trend_df: pd.DataFrame      # columns: period, value, rolling_avg
    freq: str                   # inferred pandas offset alias
    direction: str              # "up" | "down" | "flat"
    pct_change_overall: float


@dataclass
class AnomalyReport:
    column: str
    anomalies: pd.DataFrame     # columns: index_label, value, z_score
    threshold: float
    count: int


@dataclass
class EDAPayload:
    quality: QualityReport
    distributions: list[DistributionSummary]
    skew_ranking: list[tuple[str, float]]           # sorted by abs(skew) desc
    correlation: CorrelationResult
    segment_ranking: Optional[SegmentRanking]
    time_trend: Optional[TimeTrend]
    anomalies: list[AnomalyReport]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

class EDAEngine:
    """
    Stateless EDA computation engine.

    Usage
    -----
    payload = EDAEngine().compute(df, detection)
    """

    CORR_THRESHOLD: float = 0.75
    Z_THRESHOLD: float = 3.0
    IQR_MULT: float = 1.5
    TOP_CORR_FEATURES: int = 25

    # ── entry point ───────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame, detection: DetectionResult) -> EDAPayload:
        """
        Run full EDA on the provided DataFrame using pre-computed schema.

        Parameters
        ----------
        df : pd.DataFrame
        detection : DetectionResult  from UniversalDetector

        Returns
        -------
        EDAPayload
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty.")

        df = df.copy()
        num_cols = detection.numeric_cols
        cat_cols = detection.categorical_cols
        dt_cols  = detection.datetime_cols

        quality      = self._quality(df, num_cols)
        distributions = [self._distribution(df[c], c) for c in num_cols if df[c].dropna().shape[0] > 3]
        skew_ranking  = sorted(
            [(d.column, d.skew) for d in distributions],
            key=lambda x: abs(x[1]), reverse=True
        )
        correlation   = self._correlation(df, num_cols)
        segment_rank  = self._segment_ranking(df, detection)
        time_trend    = self._time_trend(df, detection)
        anomalies     = [self._anomalies(df[c], c) for c in num_cols if df[c].dropna().shape[0] > 10]

        return EDAPayload(
            quality=quality,
            distributions=distributions,
            skew_ranking=skew_ranking,
            correlation=correlation,
            segment_ranking=segment_rank,
            time_trend=time_trend,
            anomalies=anomalies,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            datetime_cols=dt_cols,
        )

    # ── quality ───────────────────────────────────────────────────────────────

    def _quality(self, df: pd.DataFrame, num_cols: list[str]) -> QualityReport:
        n, c = df.shape
        dup_pct   = round(df.duplicated().sum() / max(n, 1) * 100, 2)
        miss_pct  = round(df.isnull().mean().mean() * 100, 2)
        per_col   = {col: round(df[col].isnull().mean() * 100, 2) for col in df.columns}

        outlier_cells = 0
        total_num_cells = 0
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            total_num_cells += len(s)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_cells += int(((s < q1 - self.IQR_MULT * iqr) | (s > q3 + self.IQR_MULT * iqr)).sum())

        outlier_pct = round(outlier_cells / max(total_num_cells, 1) * 100, 2)
        completeness = round(100 - miss_pct, 2)

        return QualityReport(
            row_count=n,
            col_count=c,
            missing_pct=miss_pct,
            duplicate_pct=dup_pct,
            outlier_pct=outlier_pct,
            completeness_score=completeness,
            per_column_missing={k: v for k, v in per_col.items() if v > 0},
        )

    # ── distribution ──────────────────────────────────────────────────────────

    def _distribution(self, series: pd.Series, col: str) -> DistributionSummary:
        s = series.dropna().astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_count = 0
        if iqr > 0:
            outlier_count = int(((s < q1 - self.IQR_MULT * iqr) | (s > q3 + self.IQR_MULT * iqr)).sum())

        return DistributionSummary(
            column=col,
            mean=round(float(s.mean()), 6),
            median=round(float(s.median()), 6),
            std=round(float(s.std()), 6),
            skew=round(float(s.skew()), 4),
            kurtosis=round(float(s.kurtosis()), 4),
            p5=round(float(s.quantile(0.05)), 6),
            p95=round(float(s.quantile(0.95)), 6),
            outlier_count=outlier_count,
        )

    # ── correlation ───────────────────────────────────────────────────────────

    def _correlation(self, df: pd.DataFrame, num_cols: list[str]) -> CorrelationResult:
        if len(num_cols) < 2:
            return CorrelationResult(matrix=pd.DataFrame(), strongest_pairs=[], high_corr_pairs=[])

        # limit features for performance
        if len(num_cols) > self.TOP_CORR_FEATURES:
            variances = df[num_cols].var().nlargest(self.TOP_CORR_FEATURES)
            num_cols = variances.index.tolist()

        corr = df[num_cols].corr(method="pearson")
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "a", "level_1": "b", 0: "r"})
        )
        pairs["abs_r"] = pairs["r"].abs()
        pairs = pairs.sort_values("abs_r", ascending=False)

        strongest = [
            (row["a"], row["b"], round(float(row["r"]), 4))
            for _, row in pairs.head(10).iterrows()
        ]
        high_corr = [
            (row["a"], row["b"], round(float(row["r"]), 4))
            for _, row in pairs[pairs["abs_r"] >= self.CORR_THRESHOLD].iterrows()
        ]
        return CorrelationResult(matrix=corr, strongest_pairs=strongest, high_corr_pairs=high_corr)

    # ── segment ranking ───────────────────────────────────────────────────────

    def _segment_ranking(self, df: pd.DataFrame, d: DetectionResult) -> Optional[SegmentRanking]:
        seg = d.segment_col
        metric = d.primary_metric
        if not seg or not metric or seg not in df.columns or metric not in df.columns:
            return None

        grp = (
            df.groupby(seg, observed=True)[metric]
            .agg("sum")
            .reset_index()
            .rename(columns={seg: "segment", metric: "value"})
        )
        grp = grp.dropna(subset=["value"])
        total = grp["value"].sum()
        grp["share_pct"] = round(grp["value"] / max(total, 1e-9) * 100, 2)
        grp = grp.sort_values("value", ascending=False).reset_index(drop=True)

        return SegmentRanking(
            segment_col=seg,
            metric_col=metric,
            top_10=grp.head(10).copy(),
            bottom_10=grp.tail(10).iloc[::-1].reset_index(drop=True).copy(),
        )

    # ── time trend ────────────────────────────────────────────────────────────

    def _time_trend(self, df: pd.DataFrame, d: DetectionResult) -> Optional[TimeTrend]:
        t_col  = d.time_col
        metric = d.primary_metric
        if not t_col or not metric:
            return None
        if t_col not in df.columns or metric not in df.columns:
            return None

        tmp = df[[t_col, metric]].dropna()
        if len(tmp) < 5:
            return None

        tmp = tmp.copy()
        tmp[t_col] = pd.to_datetime(tmp[t_col], errors="coerce")
        tmp = tmp.dropna(subset=[t_col]).sort_values(t_col)

        span_days = (tmp[t_col].max() - tmp[t_col].min()).days
        freq = "D" if span_days <= 90 else "W" if span_days <= 730 else "ME"

        try:
            trend = (
                tmp.set_index(t_col)[metric]
                .resample(freq)
                .sum()
                .reset_index()
                .rename(columns={t_col: "period", metric: "value"})
            )
        except Exception:
            return None

        trend["rolling_avg"] = trend["value"].rolling(3, min_periods=1).mean().round(4)

        first = trend["value"].iloc[0]
        last  = trend["value"].iloc[-1]
        pct_change = round((last - first) / max(abs(first), 1e-9) * 100, 2)
        direction = "up" if pct_change > 2 else "down" if pct_change < -2 else "flat"

        return TimeTrend(
            time_col=t_col,
            metric_col=metric,
            trend_df=trend,
            freq=freq,
            direction=direction,
            pct_change_overall=pct_change,
        )

    # ── anomaly detection ─────────────────────────────────────────────────────

    def _anomalies(self, series: pd.Series, col: str) -> AnomalyReport:
        s = series.dropna().astype(float)
        if len(s) < 10:
            return AnomalyReport(column=col, anomalies=pd.DataFrame(), threshold=self.Z_THRESHOLD, count=0)

        z = np.abs(stats.zscore(s))
        mask = z > self.Z_THRESHOLD
        anomaly_df = pd.DataFrame({
            "index_label": s.index[mask],
            "value": s.values[mask],
            "z_score": z[mask],
        }).reset_index(drop=True)

        return AnomalyReport(
            column=col,
            anomalies=anomaly_df,
            threshold=self.Z_THRESHOLD,
            count=int(mask.sum()),
        )