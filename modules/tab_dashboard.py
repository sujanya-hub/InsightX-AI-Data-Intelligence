"""
InsightX · Dashboard Tab
modules/tab_dashboard.py

Pure UI layer. Zero analytics logic. Zero raw API calls.
Renders Section A: EDA Intelligence + Section B: Executive Dashboard
with Adaptive Intelligence Layer (Smart / Advanced Mode).

Upgrade: Enterprise multi-chart grid, scatter plots, donuts, box plots,
scatter matrix, KPI sparklines, chart recommendation engine, dashboard narrative.
"""
from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from core.analytics.dashboard_engine import DashboardEngine, DashboardPayload
from core.analytics.eda_engine import EDAEngine, EDAPayload
from core.analytics.universal_detector import UniversalDetector

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine

T = dict(
    bg="#0B0F19", card="#141A2A", card2="#181F30", border="#1F2937",
    a1="#7C5CFF", a2="#00D4FF", a3="#00E096", a4="#FF8A3D",
    success="#00E096", warn="#FF8A3D", crit="#FF4D6D",
    text="#F8FAFC", muted="#94A3B8",
    grad1="rgba(124,92,255,0.15)", grad2="rgba(0,212,255,0.10)",
)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans',sans-serif", color=T["text"], size=11),
    margin=dict(l=16, r=16, t=36, b=16),
    xaxis=dict(gridcolor=T["border"], linecolor=T["border"], zerolinecolor=T["border"]),
    yaxis=dict(gridcolor=T["border"], linecolor=T["border"], zerolinecolor=T["border"]),
)

_PALETTE = [
    T["a1"], T["a2"], T["success"], T["warn"], T["crit"],
    "#E879F9", "#34D399", "#FBBF24", "#60A5FA", "#F472B6",
    "#A78BFA", "#2DD4BF", "#FB923C", "#818CF8",
]


def _pl(**overrides) -> dict:
    base = copy.deepcopy(PL)
    for key, val in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(val, dict) and key in base:
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    return base


def _rgba(hex6: str, alpha: float) -> str:
    """
    Convert a 6-digit hex colour + an alpha float (0–1) to an rgba() string.
    Plotly does NOT accept 8-digit hex (#RRGGBBAA) for most colour properties.

    Examples
    --------
    _rgba("#7C5CFF", 0.20)  ->  "rgba(124,92,255,0.20)"
    _rgba("#FF8A3D", 0.094) ->  "rgba(255,138,61,0.09)"
    """
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


_CFG = {"displayModeBar": False}


def _pchart(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=_CFG, key=key)


# ── CSS ───────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{{background:{T["bg"]}!important;font-family:'DM Sans',sans-serif;color:{T["text"]};}}
[data-testid="stHeader"]{{background:transparent!important;}}
section[data-testid="stSidebar"]{{background:{T["card"]}!important;border-right:1px solid {T["border"]};}}
.block-container{{padding-top:1.2rem!important;}}
.ix-card{{background:{T["card"]};border:1px solid {T["border"]};border-radius:12px;padding:18px 22px;position:relative;overflow:hidden;margin-bottom:0;}}
.ix-card::before{{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(124,92,255,.05) 0%,transparent 60%);pointer-events:none;}}
.kpi-card{{background:{T["card"]};border:1px solid {T["border"]};border-radius:10px;padding:14px 18px;text-align:center;position:relative;overflow:hidden;}}
.kpi-card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--kpi-accent,{T["a1"]}),transparent);}}
.kpi-label{{font-size:9px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:{T["muted"]};margin-bottom:5px;}}
.kpi-value{{font-size:24px;font-weight:700;line-height:1;}}
.kpi-sub{{font-size:10px;color:{T["muted"]};margin-top:3px;}}
.ix-sec{{font-size:10px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:{T["muted"]};border-left:3px solid {T["a1"]};padding-left:9px;margin-bottom:12px;}}
.ix-tab-header{{font-size:15px;font-weight:700;color:{T["text"]};margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid {T["border"]};}}
.alert-row{{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid {T["border"]}22;font-size:12px;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;}}
.b-crit{{background:{T["crit"]}22;color:{T["crit"]};border:1px solid {T["crit"]}44;}}
.b-warn{{background:{T["warn"]}22;color:{T["warn"]};border:1px solid {T["warn"]}44;}}
.b-info{{background:{T["a2"]}22;color:{T["a2"]};border:1px solid {T["a2"]}44;}}
.b-ok  {{background:{T["success"]}22;color:{T["success"]};border:1px solid {T["success"]}44;}}
.ai-block{{background:linear-gradient(135deg,{T["card"]} 0%,#1a1f35 100%);border:1px solid {T["a1"]}44;border-radius:12px;padding:22px 26px;margin-top:10px;font-size:13px;line-height:1.7;white-space:pre-wrap;}}
.ai-label{{font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:{T["a1"]};margin-bottom:10px;}}
.adv-panel{{background:{T["card"]};border:1px solid {T["a1"]}44;border-radius:12px;padding:16px 20px;margin-bottom:18px;}}
.adv-suggest{{background:rgba(124,92,255,.08);border:1px solid {T["a1"]}33;border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:11px;}}
.adv-suggest-label{{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:{T["a1"]};margin-bottom:6px;}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
.chart-label{{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:{T["muted"]};margin-bottom:8px;padding-left:2px;}}
.narrative-block{{background:linear-gradient(135deg,{T["card"]} 0%,{T["card2"]} 100%);border:1px solid {T["a1"]}33;border-radius:14px;padding:28px 32px;margin-top:8px;}}
.narrative-title{{font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:{T["a1"]};margin-bottom:18px;display:flex;align-items:center;gap:8px;}}
.narrative-title::before{{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:{T["a1"]};}}
.narrative-section{{margin-bottom:16px;}}
.narrative-section-label{{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{T["a2"]};margin-bottom:6px;}}
.narrative-section-text{{font-size:13px;line-height:1.75;color:#CBD5E1;}}
.rec-chip{{display:inline-block;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);border-radius:20px;padding:3px 12px;font-size:11px;color:{T["a2"]};margin:3px 3px 3px 0;}}
.stButton>button{{background:linear-gradient(135deg,{T["a1"]}cc,{T["a1"]})!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:12px!important;padding:9px 20px!important;}}
.stButton>button:hover{{opacity:.85!important;}}
[data-testid="stTabs"] [data-baseweb="tab"]{{font-size:12px;font-weight:600;color:{T["muted"]};letter-spacing:.06em;}}
[data-testid="stTabs"] [aria-selected="true"]{{color:{T["a1"]}!important;}}
</style>""", unsafe_allow_html=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sec(title: str) -> None:
    st.markdown(f'<div class="ix-sec">{title}</div>', unsafe_allow_html=True)

def _card_open() -> None:
    st.markdown('<div class="ix-card">', unsafe_allow_html=True)

def _card_close() -> None:
    st.markdown('</div>', unsafe_allow_html=True)

def _div() -> None:
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {T["border"]};margin:20px 0">',
        unsafe_allow_html=True,
    )

def _badge(sev: str) -> str:
    cls = {"critical": "b-crit", "warning": "b-warn", "info": "b-info"}.get(sev, "b-info")
    return f'<span class="badge {cls}">{sev.upper()}</span>'

def _kpi(label: str, value: str, sub: str = "", color: str = "") -> str:
    col_style = f"color:{color}" if color else ""
    accent    = color or T["a1"]
    return (
        f'<div class="kpi-card" style="--kpi-accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="{col_style}">{value}</div>'
        + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
        + '</div>'
    )

def _health_color(score: float) -> str:
    return T["success"] if score >= 80 else T["warn"] if score >= 60 else T["crit"]

def _badge_html(score: float) -> str:
    label = "HEALTHY" if score >= 80 else "MODERATE" if score >= 60 else "CRITICAL"
    cls   = "b-ok" if score >= 80 else "b-warn" if score >= 60 else "b-crit"
    return f'<span class="badge {cls}">{label}</span>'


# ── Chart helpers — base ──────────────────────────────────────────────────────

def _chart_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    n  = len(corr_matrix)
    cs = [[0, T["crit"]], [0.25, "#6B2FBD"], [0.5, T["card"]], [0.75, "#2B6CB0"], [1, T["a2"]]]
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=cs, zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2) if n <= 15 else None,
        texttemplate="%{text}" if n <= 15 else None,
        textfont=dict(size=8),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=8, color=T["muted"]), outlinewidth=0),
    ))
    fig.update_layout(**_pl(
        height=max(300, min(n * 24, 560)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def _chart_bar(data: list[tuple], title: str, color: str, unit: str = "") -> go.Figure:
    if not data:
        return go.Figure()
    labels, values = zip(*data)
    fig = go.Figure(go.Bar(
        y=list(labels), x=list(values), orientation="h",
        marker=dict(
            color=list(values),
            colorscale=[[0, _rgba(color, 0.33)], [1, color]],
            showscale=False,
        ),
        text=[f"{v}{unit}" for v in values], textposition="outside",
        textfont=dict(size=9, color=T["muted"]),
        hovertemplate=f"<b>%{{y}}</b><br>{title}: %{{x}}{unit}<extra></extra>",
    ))
    fig.update_layout(**_pl(
        title=dict(text=title, font=dict(size=12), x=0.01),
        height=max(200, len(data) * 32 + 60),
        yaxis=dict(autorange="reversed"), showlegend=False,
    ))
    return fig


def _chart_histogram(series: pd.Series, col: str) -> go.Figure:
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=35,
        marker=dict(color=T["a1"], opacity=0.85,
                    line=dict(color=_rgba(T["a1"], 0.20), width=0.5)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    if len(s) > 10:
        try:
            from scipy.stats import gaussian_kde
            kde_x   = np.linspace(float(s.min()), float(s.max()), 200)
            scale   = len(s) * (float(s.max()) - float(s.min())) / 35
            fig.add_trace(go.Scatter(
                x=kde_x, y=gaussian_kde(s)(kde_x) * scale,
                mode="lines", line=dict(color=T["a2"], width=2), name="KDE",
            ))
        except Exception:
            pass
    fig.update_layout(**_pl(
        title=dict(text=f"Distribution · {col}", font=dict(size=12), x=0.01),
        height=260, showlegend=False, bargap=0.02,
    ))
    return fig


def _chart_time_trend(trend_df: pd.DataFrame, metric: str, direction: str) -> go.Figure:
    color = T["success"] if direction == "up" else T["crit"] if direction == "down" else T["a2"]
    fig   = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=trend_df["period"].astype(str), y=trend_df["value"],
        mode="lines", line=dict(color=color, width=0),
        fill="tozeroy", fillcolor=_rgba(color, 0.09), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["period"].astype(str), y=trend_df["value"],
        mode="lines+markers", line=dict(color=color, width=2),
        marker=dict(size=4, color=color), name=metric,
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
    ))
    if "rolling_avg" in trend_df.columns:
        fig.add_trace(go.Scatter(
            x=trend_df["period"].astype(str), y=trend_df["rolling_avg"],
            mode="lines", line=dict(color=T["muted"], width=1.5, dash="dot"), name="3-period MA",
        ))
    fig.update_layout(**_pl(
        title=dict(text=f"Time Trend · {metric}", font=dict(size=12), x=0.01),
        height=270, showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def _chart_performance_heatmap(pivot: pd.DataFrame) -> go.Figure:
    cs = [[0, T["crit"]], [0.5, T["card"]], [1, T["success"]]]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=cs,
        hovertemplate="<b>%{y}</b> · %{x}<br>Value: %{z:,.2f}<extra></extra>",
        colorbar=dict(thickness=8, tickfont=dict(size=8, color=T["muted"]), outlinewidth=0),
    ))
    fig.update_layout(**_pl(
        height=max(280, len(pivot) * 22 + 60),
        xaxis=dict(tickangle=-30, tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def _chart_anomaly(series: pd.Series, col: str, threshold: float) -> go.Figure:
    s = series.dropna().reset_index(drop=True)
    try:
        from scipy import stats as sc
        z = np.abs(sc.zscore(s))
    except Exception:
        mean, std = s.mean(), s.std()
        z = np.abs((s - mean) / std) if std > 0 else pd.Series(np.zeros(len(s)))
    is_anomaly = z > threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index[~is_anomaly], y=s.values[~is_anomaly],
        mode="markers", marker=dict(color=T["a1"], size=4, opacity=0.6), name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=s.index[is_anomaly], y=s.values[is_anomaly],
        mode="markers", marker=dict(color=T["crit"], size=7, symbol="x"), name="Anomaly",
    ))
    fig.add_hline(y=float(s.mean()) + threshold * float(s.std()),
                  line_dash="dot", line_color=T["warn"], line_width=1)
    fig.add_hline(y=float(s.mean()) - threshold * float(s.std()),
                  line_dash="dot", line_color=T["warn"], line_width=1)
    fig.update_layout(**_pl(
        title=dict(text=f"Anomaly Map · {col}", font=dict(size=12), x=0.01),
        height=250, showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ── NEW chart builders ────────────────────────────────────────────────────────

def _chart_box_plot(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Box plot for one or more numeric columns."""
    fig = go.Figure()
    for i, col in enumerate(cols[:8]):
        color = _PALETTE[i % len(_PALETTE)]
        s     = df[col].dropna()
        fig.add_trace(go.Box(
            y=s, name=col,
            marker_color=color, line_color=color,
            fillcolor=_rgba(color, 0.20),
            boxmean="sd",
            hovertemplate=f"<b>{col}</b><br>%{{y:,.3f}}<extra></extra>",
        ))
    fig.update_layout(**_pl(
        title=dict(text="Box Plot — Variance Overview", font=dict(size=12), x=0.01),
        height=300, showlegend=False,
        xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
    ))
    return fig


def _chart_donut(df: pd.DataFrame, cat_col: str, metric_col: str, top_n: int = 10) -> go.Figure:
    """Donut chart: segment shares by metric."""
    grp = (
        df.groupby(cat_col, observed=True)[metric_col]
        .sum()
        .reset_index()
        .sort_values(metric_col, ascending=False)
    )
    # Collapse tail into 'Other'
    if len(grp) > top_n:
        top  = grp.head(top_n)
        rest = grp.tail(len(grp) - top_n)[metric_col].sum()
        other_row = pd.DataFrame({cat_col: ["Other"], metric_col: [rest]})
        grp = pd.concat([top, other_row], ignore_index=True)

    colors = _PALETTE[:len(grp)]
    fig = go.Figure(go.Pie(
        labels=grp[cat_col].astype(str),
        values=grp[metric_col],
        hole=0.58,
        marker=dict(colors=colors, line=dict(color=T["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=9, color=T["text"]),
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} (%{percent})<extra></extra>",
        rotation=90,
    ))
    fig.update_layout(**_pl(
        title=dict(text=f"Segment Share · {metric_col}", font=dict(size=12), x=0.01),
        height=300,
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)", orientation="v",
                    x=1.01, y=0.5),
        margin=dict(l=0, r=80, t=36, b=0),
    ))
    return fig


def _chart_scatter(df: pd.DataFrame, col_x: str, col_y: str,
                   color_col: str = None) -> go.Figure:
    """Scatter plot for a correlated pair, optional colour by category."""
    sub = df[[col_x, col_y]].dropna()
    if sub.empty:
        return go.Figure()

    if color_col and color_col in df.columns and df[color_col].nunique() <= 20:
        sub = sub.join(df[color_col])
        cats  = sub[color_col].unique()
        traces = []
        for i, cat in enumerate(cats):
            mask = sub[color_col] == cat
            traces.append(go.Scatter(
                x=sub.loc[mask, col_x], y=sub.loc[mask, col_y],
                mode="markers",
                name=str(cat),
                marker=dict(color=_PALETTE[i % len(_PALETTE)], size=5, opacity=0.7),
                hovertemplate=f"<b>{cat}</b><br>{col_x}: %{{x:,.3f}}<br>{col_y}: %{{y:,.3f}}<extra></extra>",
            ))
        fig = go.Figure(traces)
    else:
        # Colour by density proxy using index
        fig = go.Figure(go.Scatter(
            x=sub[col_x], y=sub[col_y],
            mode="markers",
            marker=dict(
                size=5, opacity=0.65,
                color=np.arange(len(sub)),
                colorscale=[[0, _rgba(T["a1"], 0.53)], [1, T["a2"]]],
                showscale=False,
            ),
            hovertemplate=f"{col_x}: %{{x:,.3f}}<br>{col_y}: %{{y:,.3f}}<extra></extra>",
        ))

    # OLS trend line
    try:
        x_v = sub[col_x].values.astype(float)
        y_v = sub[col_y].values.astype(float)
        m   = np.polyfit(x_v, y_v, 1)
        x_r = np.linspace(x_v.min(), x_v.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_r, y=np.polyval(m, x_r),
            mode="lines", name="OLS trend",
            line=dict(color=T["warn"], width=1.5, dash="dash"),
            showlegend=False,
        ))
    except Exception:
        pass

    fig.update_layout(**_pl(
        title=dict(text=f"{col_x} vs {col_y}", font=dict(size=12), x=0.01),
        height=280,
       xaxis=dict(title=dict(text=col_x, font=dict(size=10))),
       yaxis=dict(title=dict(text=col_y, font=dict(size=10))),
        showlegend=bool(color_col),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def _chart_scatter_matrix(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Scatter matrix (SPLOM) for up to 5 numeric columns."""
    cols = cols[:5]
    n    = len(cols)
    if n < 2:
        return go.Figure()

    fig = go.Figure(go.Splom(
        dimensions=[
            dict(label=col, values=df[col].dropna().reindex(df.index))
            for col in cols
        ],
        marker=dict(
            color=T["a1"],
            size=3,
            opacity=0.55,
            line=dict(width=0),
        ),
        diagonal_visible=True,
        showupperhalf=False,
        hovertemplate="%{xaxis.title.text}: %{x:,.3f}<br>%{yaxis.title.text}: %{y:,.3f}<extra></extra>",
    ))
    fig.update_layout(**_pl(
        title=dict(text="Scatter Matrix — Numeric Relationships", font=dict(size=12), x=0.01),
        height=max(380, n * 90),
        dragmode="select",
    ))
    # Style sub-axes
    for i in range(1, n + 1):
        ax_key = f"xaxis{i if i > 1 else ''}"
        ay_key = f"yaxis{i if i > 1 else ''}"
        for k in (ax_key, ay_key):
            if k in fig.layout:
                fig.layout[k].update(
                    gridcolor=T["border"],
                    linecolor=T["border"],
                    tickfont=dict(size=7),
                )
    return fig


def _chart_sparkline(values: list[float], color: str = None) -> go.Figure:
    """Compact sparkline for KPI cards."""
    color = color or T["a1"]
    fig   = go.Figure(go.Scatter(
        x=list(range(len(values))), y=values,
        mode="lines",
        line=dict(color=color, width=1.5, shape="spline"),
        fill="tozeroy", fillcolor=_rgba(color, 0.13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=40,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _chart_forecast_simple(trend_df: pd.DataFrame, metric: str, direction: str,
                            horizon: int = 6) -> go.Figure:
    """
    Simple linear extrapolation forecast appended to the trend series,
    rendered alongside the historical line.
    """
    color = T["success"] if direction == "up" else T["crit"] if direction == "down" else T["a2"]
    vals  = trend_df["value"].values.astype(float)
    n     = len(vals)

    # Linear regression for forecast
    x      = np.arange(n)
    coeffs = np.polyfit(x, vals, 1)
    x_fc   = np.arange(n, n + horizon)
    y_fc   = np.polyval(coeffs, x_fc)

    # CI ± 1.5 std of residuals
    resid = vals - np.polyval(coeffs, x)
    std   = float(np.std(resid))
    labels_hist = trend_df["period"].astype(str).tolist()
    labels_fc   = [f"F+{i + 1}" for i in range(horizon)]

    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=labels_hist, y=vals.tolist(),
        mode="lines+markers", name="Historical",
        line=dict(color=color, width=2),
        marker=dict(size=3),
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=labels_fc, y=y_fc.tolist(),
        mode="lines+markers", name="Forecast",
        line=dict(color=T["warn"], width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
    ))
    # CI band
    fig.add_trace(go.Scatter(
        x=labels_fc + labels_fc[::-1],
        y=(y_fc + 1.5 * std).tolist() + (y_fc - 1.5 * std).tolist()[::-1],
        fill="toself", fillcolor=_rgba(T["warn"], 0.09),
        line=dict(color="rgba(0,0,0,0)"), name="CI ±1.5σ",
    ))
    fig.update_layout(**_pl(
        title=dict(text=f"Trend + Forecast · {metric}", font=dict(size=12), x=0.01),
        height=270, showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ── Chart Recommendation Engine ───────────────────────────────────────────────

def _recommend_charts(eda: EDAPayload, df: pd.DataFrame) -> dict[str, bool]:
    """
    Analyse dataset structure and return a feature-flag dict
    indicating which chart types should be rendered.
    """
    num_cols = eda.numeric_cols  or []
    cat_cols = eda.categorical_cols or []
    dt_cols  = eda.datetime_cols or []

    return {
        "time_trend":      bool(dt_cols and eda.time_trend),
        "forecast":        bool(dt_cols and eda.time_trend and len(eda.time_trend.trend_df) >= 6),
        "segment_bar":     bool(cat_cols),
        "segment_donut":   bool(cat_cols and num_cols),
        "correlation":     len(num_cols) >= 3,
        "scatter_pairs":   bool(eda.correlation and eda.correlation.strongest_pairs),
        "scatter_matrix":  len(num_cols) >= 3,
        "box_plot":        len(num_cols) >= 2,
        "distribution":    bool(num_cols),
        "anomaly":         bool([a for a in eda.anomalies if a.count > 0]),
        "perf_heatmap":    bool(cat_cols and dt_cols),
    }


# ── KPI Sparklines ────────────────────────────────────────────────────────────

def _build_sparkline_data(df: pd.DataFrame, num_cols: list[str],
                          window: int = 20) -> dict[str, list[float]]:
    """
    Extract the last `window` non-null values for each numeric column
    to serve as sparkline data.
    """
    result = {}
    for col in num_cols[:8]:
        vals = df[col].dropna().tail(window).tolist()
        if len(vals) >= 3:
            result[col] = [float(v) for v in vals]
    return result


def _render_kpi_sparklines(df: pd.DataFrame, eda: EDAPayload,
                           dash: DashboardPayload) -> None:
    """KPI summary row with embedded sparkline charts."""
    _sec("KPI Sparklines")
    spark_data = _build_sparkline_data(df, eda.numeric_cols or [])

    if not spark_data:
        st.info("Insufficient numeric data for sparklines.")
        return

    cols_per_row = 4
    items        = list(spark_data.items())
    for row_start in range(0, min(len(items), 8), cols_per_row):
        row_items = items[row_start: row_start + cols_per_row]
        cols_ui   = st.columns(len(row_items))
        for col_ui, (col_name, values) in zip(cols_ui, row_items):
            with col_ui:
                mean_v  = float(np.mean(values))
                last_v  = values[-1]
                delta   = ((last_v - values[0]) / abs(values[0]) * 100) if values[0] != 0 else 0.0
                d_color = T["success"] if delta >= 0 else T["crit"]

                st.markdown(
                    f'<div class="kpi-card" style="--kpi-accent:{d_color};padding-bottom:4px">'
                    f'<div class="kpi-label">{col_name[:20]}</div>'
                    f'<div class="kpi-value" style="color:{d_color}">{last_v:,.2f}</div>'
                    f'<div class="kpi-sub" style="color:{d_color}">{delta:+.1f}% vs start</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _pchart(
                    _chart_sparkline(values, d_color),
                    key=f"sparkline_{col_name}",
                )


# ── Adaptive Intelligence Layer ───────────────────────────────────────────────

def _render_adaptive_controls(eda: EDAPayload, detection) -> dict:
    suggested_metric = detection.primary_metric
    suggested_group  = detection.segment_col
    suggested_time   = detection.time_col

    adv = st.toggle("Advanced Analysis Mode", key="adv_mode_toggle", value=False)

    overrides = dict(
        metric=suggested_metric,
        group_by=suggested_group,
        time_col=suggested_time,
        top_n=10,
        comparison_type="Segment Analysis",
        advanced_mode=adv,
    )

    if not adv:
        return overrides

    st.markdown('<div class="adv-panel">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="adv-suggest">'
        f'<div class="adv-suggest-label">Suggested Configuration</div>'
        f'<span style="color:{T["muted"]};font-size:11px">'
        f'Metric: <b style="color:{T["a2"]}">{suggested_metric or "—"}</b>'
        f' &nbsp;·&nbsp; '
        f'Group: <b style="color:{T["a2"]}">{suggested_group or "—"}</b>'
        f' &nbsp;·&nbsp; '
        f'Time: <b style="color:{T["a2"]}">{suggested_time or "—"}</b>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    num_cols = eda.numeric_cols     or []
    cat_cols = eda.categorical_cols or []
    dt_cols  = eda.datetime_cols    or []

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 2])
    with c1:
        metric_opts = num_cols if num_cols else ["—"]
        default_m   = metric_opts.index(suggested_metric) if suggested_metric in metric_opts else 0
        sel_metric  = st.selectbox("Metric", metric_opts, index=default_m, key="adv_metric")
    with c2:
        group_opts = cat_cols if cat_cols else ["—"]
        default_g  = group_opts.index(suggested_group) if suggested_group in group_opts else 0
        sel_group  = st.selectbox("Group By", group_opts, index=default_g, key="adv_group")
    with c3:
        time_opts = dt_cols if dt_cols else ["—"]
        default_t = time_opts.index(suggested_time) if suggested_time in time_opts else 0
        sel_time  = st.selectbox("Time Column", time_opts, index=default_t, key="adv_time")
    with c4:
        sel_top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1, key="adv_topn")
    with c5:
        comp_types = ["Segment Analysis", "Time Trend", "Distribution", "Correlation"]
        sel_comp   = st.selectbox("Comparison Type", comp_types, key="adv_comp")

    st.markdown('</div>', unsafe_allow_html=True)

    overrides.update(dict(
        metric=sel_metric  if sel_metric  != "—" else suggested_metric,
        group_by=sel_group if sel_group   != "—" else suggested_group,
        time_col=sel_time  if sel_time    != "—" else suggested_time,
        top_n=int(sel_top_n),
        comparison_type=sel_comp,
    ))
    return overrides


# ── Section A: EDA Intelligence ───────────────────────────────────────────────

def _eda_quality(eda: EDAPayload) -> None:
    q = eda.quality
    _sec("Data Quality Panel")
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "Completeness",  f"{q.completeness_score:.1f}%", "",
         _health_color(q.completeness_score)),
        (c2, "Missing",       f"{q.missing_pct:.1f}%",        f"{q.row_count:,} rows",
         T["crit"] if q.missing_pct >= 10 else T["text"]),
        (c3, "Duplicates",    f"{q.duplicate_pct:.1f}%",      "",
         T["warn"] if q.duplicate_pct >= 3 else T["text"]),
        (c4, "Outlier Cells", f"{q.outlier_pct:.1f}%",        "IQR method",
         T["warn"] if q.outlier_pct >= 10 else T["text"]),
    ]
    for col_ui, label, val, sub, color in kpis:
        with col_ui:
            st.markdown(_kpi(label, val, sub, color), unsafe_allow_html=True)

    if q.per_column_missing:
        st.markdown("<br>", unsafe_allow_html=True)
        items = sorted(q.per_column_missing.items(), key=lambda x: x[1], reverse=True)[:12]
        _pchart(_chart_bar(items, "Missing %", T["crit"], "%"), key="eda_missing_bar")


def _eda_distribution(eda: EDAPayload, df: pd.DataFrame, charts: dict) -> None:
    _sec("Distribution Explorer")
    if not eda.numeric_cols:
        st.info("No numeric columns detected.")
        return

    col_sel = st.selectbox(
        "Select column", eda.numeric_cols,
        key="eda_dist_col", label_visibility="collapsed",
    )

    # Multi-chart grid: histogram LEFT, box plot RIGHT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-label">Histogram + KDE</div>', unsafe_allow_html=True)
        _pchart(_chart_histogram(df[col_sel], col_sel), key="eda_hist")
    with c2:
        st.markdown('<div class="chart-label">Box Plot — Variance</div>', unsafe_allow_html=True)
        box_cols = eda.numeric_cols[:6]
        _pchart(_chart_box_plot(df, box_cols), key="eda_box_dist")

    # Stats panel
    dist = next((d for d in eda.distributions if d.column == col_sel), None)
    if dist:
        stat_cols = st.columns(8)
        stat_items = [
            ("Mean",     f"{dist.mean:,.4f}"),
            ("Median",   f"{dist.median:,.4f}"),
            ("Std Dev",  f"{dist.std:,.4f}"),
            ("Skew",     f"{dist.skew:.4f}"),
            ("Kurtosis", f"{dist.kurtosis:.4f}"),
            ("P5",       f"{dist.p5:,.4f}"),
            ("P95",      f"{dist.p95:,.4f}"),
            ("Outliers", str(dist.outlier_count)),
        ]
        for sc_ui, (lbl, val) in zip(stat_cols, stat_items):
            with sc_ui:
                sc = T["warn"] if lbl == "Skew" and abs(dist.skew) > 2 else T["text"]
                st.markdown(
                    f'<div style="text-align:center;padding:8px 4px;">'
                    f'<div style="font-size:9px;color:{T["muted"]};text-transform:uppercase;'
                    f'letter-spacing:.1em">{lbl}</div>'
                    f'<div style="font-size:13px;font-weight:600;color:{sc};margin-top:3px">'
                    f'{val}</div></div>',
                    unsafe_allow_html=True,
                )

    if eda.skew_ranking:
        st.markdown("<br>", unsafe_allow_html=True)
        _sec("Skewness Ranking")
        items = [(c, round(abs(s), 4)) for c, s in eda.skew_ranking[:12]]
        _pchart(_chart_bar(items, "Abs Skew", T["warn"]), key="eda_skew_bar")


def _eda_correlation(eda: EDAPayload, df: pd.DataFrame, charts: dict) -> None:
    _sec("Correlation Analysis")
    corr = eda.correlation
    if corr.matrix.empty:
        st.info("Insufficient numeric columns for correlation analysis.")
        return

    # Multi-chart grid: heatmap LEFT, scatter matrix RIGHT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-label">Correlation Heatmap</div>', unsafe_allow_html=True)
        _pchart(_chart_heatmap(corr.matrix), key="eda_corr_heatmap")
    with c2:
        st.markdown('<div class="chart-label">Scatter Matrix</div>', unsafe_allow_html=True)
        if charts["scatter_matrix"] and len(eda.numeric_cols) >= 3:
            _pchart(_chart_scatter_matrix(df, eda.numeric_cols[:5]), key="eda_scatter_matrix")
        else:
            st.info("Need 3+ numeric columns for scatter matrix.")

    # Scatter plots for top correlated pairs
    if charts["scatter_pairs"] and corr.strongest_pairs:
        _sec("Top Correlated Pairs — Scatter Plots")
        pairs   = corr.strongest_pairs[:4]
        n_pairs = len(pairs)
        pair_cols = st.columns(min(n_pairs, 2))
        for i, (col_a, col_b, r_val) in enumerate(pairs):
            with pair_cols[i % 2]:
                r_color = T["crit"] if abs(r_val) >= 0.75 else T["warn"] if abs(r_val) >= 0.5 else T["success"]
                st.markdown(
                    f'<div style="font-size:10px;color:{T["muted"]};margin-bottom:4px">'
                    f'{col_a} <span style="color:{T["a1"]}">↔</span> {col_b} '
                    f'<span style="color:{r_color};font-weight:700">r={r_val:.3f}</span></div>',
                    unsafe_allow_html=True,
                )
                cat_col = eda.categorical_cols[0] if eda.categorical_cols else None
                _pchart(
                    _chart_scatter(df, col_a, col_b, color_col=cat_col),
                    key=f"eda_scatter_{i}",
                )

    # Strongest pairs table
    if corr.strongest_pairs:
        _sec("Correlation Pairs Table")
        c_left, _ = st.columns(2)
        with c_left:
            for a, b, r in corr.strongest_pairs[:6]:
                r_color = T["crit"] if abs(r) >= 0.75 else T["warn"] if abs(r) >= 0.5 else T["success"]
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                    f'border-bottom:1px solid {T["border"]};font-size:12px;">'
                    f'<span style="color:{T["a1"]}">{a}</span>'
                    f'<span style="color:{T["muted"]};margin:0 6px">↔</span>'
                    f'<span style="color:{T["a2"]}">{b}</span>'
                    f'<span style="color:{r_color};font-weight:700;margin-left:auto">{r:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _eda_segments(eda: EDAPayload, df: pd.DataFrame, charts: dict) -> None:
    _sec("Segment Explorer")
    sr = eda.segment_ranking
    if not sr:
        st.info("No segment column detected.")
        return

    # Multi-chart grid: bar chart LEFT, donut chart RIGHT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-label">Top Performers</div>', unsafe_allow_html=True)
        _pchart(
            _chart_bar(
                list(zip(sr.top_10["segment"].astype(str), sr.top_10["value"])),
                sr.metric_col, T["success"],
            ),
            key="eda_seg_top",
        )
    with c2:
        st.markdown('<div class="chart-label">Segment Distribution</div>', unsafe_allow_html=True)
        if charts["segment_donut"] and eda.numeric_cols:
            _pchart(
                _chart_donut(df, sr.segment_col, sr.metric_col),
                key="eda_seg_donut",
            )
        else:
            _pchart(
                _chart_bar(
                    list(zip(sr.bottom_10["segment"].astype(str), sr.bottom_10["value"])),
                    sr.metric_col, T["crit"],
                ),
                key="eda_seg_bot",
            )

    # Bottom performers bar below
    if charts["segment_donut"]:
        _sec("Bottom Performers")
        _pchart(
            _chart_bar(
                list(zip(sr.bottom_10["segment"].astype(str), sr.bottom_10["value"])),
                sr.metric_col, T["crit"],
            ),
            key="eda_seg_bot_2",
        )


def _eda_time_trend(eda: EDAPayload, charts: dict) -> None:
    _sec("Time Trend Analysis")
    tt = eda.time_trend
    if not tt:
        st.info("No datetime + primary metric combination detected.")
        return

    dir_color = (
        T["success"] if tt.direction == "up"
        else T["crit"] if tt.direction == "down"
        else T["muted"]
    )
    st.markdown(
        f'<div style="display:flex;gap:20px;margin-bottom:12px;font-size:12px;">'
        f'<span style="color:{T["muted"]}">Metric: '
        f'<b style="color:{T["text"]}">{tt.metric_col}</b></span>'
        f'<span style="color:{T["muted"]}">Direction: '
        f'<b style="color:{dir_color}">{tt.direction.upper()}</b></span>'
        f'<span style="color:{T["muted"]}">Overall Delta: '
        f'<b style="color:{dir_color}">{tt.pct_change_overall:+.1f}%</b></span>'
        f'<span style="color:{T["muted"]}">Freq: '
        f'<b style="color:{T["text"]}">{tt.freq}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Multi-chart grid: trend LEFT, forecast RIGHT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-label">Historical Trend</div>', unsafe_allow_html=True)
        _pchart(_chart_time_trend(tt.trend_df, tt.metric_col, tt.direction), key="eda_time_trend")
    with c2:
        st.markdown('<div class="chart-label">Trend + Forecast</div>', unsafe_allow_html=True)
        if charts["forecast"]:
            _pchart(
                _chart_forecast_simple(tt.trend_df, tt.metric_col, tt.direction),
                key="eda_forecast",
            )
        else:
            st.info("Insufficient data points for forecast projection (need 6+).")


def _eda_anomalies(eda: EDAPayload, df: pd.DataFrame) -> None:
    _sec("Anomaly Panel")
    significant = [a for a in eda.anomalies if a.count > 0]
    if not significant:
        st.success("No statistically significant anomalies detected (Z-score threshold = 3.0).")
        return

    col_sel  = st.selectbox(
        "Select column", [a.column for a in significant],
        key="anomaly_col_sel", label_visibility="collapsed",
    )
    a_report = next(a for a in significant if a.column == col_sel)
    c1, c2   = st.columns([2, 1])
    with c1:
        _pchart(_chart_anomaly(df[col_sel], col_sel, a_report.threshold), key="eda_anomaly")
    with c2:
        st.markdown(
            f'<div style="padding:8px 0">'
            f'<div style="font-size:10px;color:{T["muted"]};text-transform:uppercase;'
            f'letter-spacing:.1em">Anomaly Count</div>'
            f'<div style="font-size:28px;font-weight:700;color:{T["crit"]}">'
            f'{a_report.count}</div>'
            f'<div style="font-size:11px;color:{T["muted"]};margin-top:4px">'
            f'Z-threshold: {a_report.threshold}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if not a_report.anomalies.empty:
            for _, row in a_report.anomalies.head(5).iterrows():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                    f'border-bottom:1px solid {T["border"]};font-size:11px;">'
                    f'<span style="color:{T["muted"]}">idx {row["index_label"]}</span>'
                    f'<span style="color:{T["crit"]};font-weight:600">{row["value"]:,.3f}</span>'
                    f'<span style="color:{T["warn"]}">z={row["z_score"]:.2f}</span></div>',
                    unsafe_allow_html=True,
                )


# ── Section B: Executive Dashboard ───────────────────────────────────────────

def _dash_header(dash: DashboardPayload, overrides: dict) -> None:
    kpis        = dash.kpis
    badge       = _badge_html(kpis.health_score)
    health_c    = _health_color(kpis.health_score)
    mode_label  = (
        f'<span style="font-size:10px;color:{T["a1"]};background:rgba(124,92,255,.12);'
        f'border:1px solid {T["a1"]}44;border-radius:20px;padding:2px 10px;margin-left:10px">'
        f'ADVANCED · {overrides["comparison_type"]}</span>'
        if overrides["advanced_mode"] else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">'
        f'<div><div style="font-size:20px;font-weight:700">Executive Dashboard{mode_label}</div>'
        f'<div style="font-size:11px;color:{T["muted"]};margin-top:3px">'
        f'Analytics Intelligence · InsightX'
        + (f' &nbsp;·&nbsp; Metric: <b style="color:{T["a2"]}">{overrides["metric"]}</b>'
           f' · Group: <b style="color:{T["a2"]}">{overrides["group_by"] or "auto"}</b>'
           if overrides["advanced_mode"] else "")
        + f'</div></div><div style="margin-left:auto">{badge}</div></div>',
        unsafe_allow_html=True,
    )
    kpi_data = [
        ("Health Score",       f"{kpis.health_score:.0f}",        "/100",        health_c),
        ("Growth Signal",      f"{kpis.growth_signal:+.1f}%",     "overall",     T["success"] if kpis.growth_signal >= 0 else T["crit"]),
        ("Concentration Risk", f"{kpis.concentration_risk:.2f}",  "HHI-derived", T["warn"] if kpis.concentration_risk >= 0.35 else T["success"]),
        ("Volatility",         f"{kpis.volatility_score:.2f}",    "CV ratio",    T["warn"] if kpis.volatility_score >= 0.5 else T["text"]),
        ("Seg. Dominance",     f"{kpis.segment_dominance:.1f}%",  "top segment", T["text"]),
        ("Anomaly Pressure",   f"{kpis.anomaly_pressure:.3f}",    "0–1 index",   T["crit"] if kpis.anomaly_pressure >= 0.05 else T["success"]),
    ]
    cols = st.columns(6)
    for col_ui, (label, val, sub, color) in zip(cols, kpi_data):
        with col_ui:
            st.markdown(_kpi(label, val, sub, color), unsafe_allow_html=True)


def _dash_performers(dash: DashboardPayload, df: pd.DataFrame,
                     eda: EDAPayload, charts: dict, top_n: int = 10) -> None:
    _sec("Top & Bottom Performers")
    if dash.top_performers.empty:
        st.info("No segment data available.")
        return

    top = dash.top_performers.head(top_n)
    bot = dash.bottom_performers.head(top_n)

    # Multi-chart grid: bar LEFT, donut RIGHT
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-label">Top Performers — Bar</div>', unsafe_allow_html=True)
        fig_top = _chart_bar(list(zip(top["segment"].astype(str), top["value"])), "Value", T["success"])
        fig_top.update_layout(title=dict(text="Top Performers", font=dict(size=12), x=0.01))
        _pchart(fig_top, key="dash_top_perf")
    with c2:
        st.markdown('<div class="chart-label">Segment Distribution — Donut</div>', unsafe_allow_html=True)
        if charts["segment_donut"] and eda.segment_ranking:
            sr = eda.segment_ranking
            _pchart(
                _chart_donut(df, sr.segment_col, sr.metric_col, top_n=top_n),
                key="dash_donut",
            )
        else:
            fig_bot = _chart_bar(list(zip(bot["segment"].astype(str), bot["value"])), "Value", T["crit"])
            fig_bot.update_layout(title=dict(text="Bottom Performers", font=dict(size=12), x=0.01))
            _pchart(fig_bot, key="dash_bot_perf_alt")

    # Bottom performers below
    _sec("Bottom Performers")
    c3, c4 = st.columns(2)
    with c3:
        fig_bot2 = _chart_bar(list(zip(bot["segment"].astype(str), bot["value"])), "Value", T["crit"])
        fig_bot2.update_layout(title=dict(text="Bottom Performers", font=dict(size=12), x=0.01))
        _pchart(fig_bot2, key="dash_bot_perf")
    with c4:
        # Box plot for numeric variance overview
        if charts["box_plot"] and eda.numeric_cols:
            st.markdown('<div class="chart-label">Numeric Variance</div>', unsafe_allow_html=True)
            _pchart(_chart_box_plot(df, eda.numeric_cols[:6]), key="dash_box")


def _dash_advanced_view(df: pd.DataFrame, overrides: dict, eda: EDAPayload, charts: dict) -> None:
    comp     = overrides["comparison_type"]
    metric   = overrides["metric"]
    group_by = overrides["group_by"]
    top_n    = overrides["top_n"]

    _sec(f"Advanced View · {comp}")

    if comp == "Distribution":
        if metric and metric in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                _pchart(_chart_histogram(df[metric], metric), key="adv_dist")
            with c2:
                _pchart(_chart_box_plot(df, [metric] + eda.numeric_cols[:3]), key="adv_box")
        else:
            st.info("Select a valid numeric metric.")

    elif comp == "Correlation":
        corr = eda.correlation
        if not corr.matrix.empty:
            c1, c2 = st.columns(2)
            with c1:
                _pchart(_chart_heatmap(corr.matrix), key="adv_corr")
            with c2:
                if charts["scatter_matrix"]:
                    _pchart(_chart_scatter_matrix(df, eda.numeric_cols[:5]), key="adv_smat")
                else:
                    st.info("Need 3+ numeric columns for scatter matrix.")
        else:
            st.info("Insufficient numeric columns for correlation.")

    elif comp == "Time Trend":
        tt = eda.time_trend
        if tt:
            c1, c2 = st.columns(2)
            with c1:
                _pchart(_chart_time_trend(tt.trend_df, tt.metric_col, tt.direction), key="adv_trend")
            with c2:
                if charts["forecast"]:
                    _pchart(
                        _chart_forecast_simple(tt.trend_df, tt.metric_col, tt.direction),
                        key="adv_forecast",
                    )
                else:
                    st.info("Insufficient data for forecast.")
        else:
            st.info("No time trend data available.")

    else:  # Segment Analysis
        if metric and group_by and metric in df.columns and group_by in df.columns:
            grp = (
                df.groupby(group_by, observed=True)[metric]
                .sum().reset_index()
                .rename(columns={group_by: "segment", metric: "value"})
                .sort_values("value", ascending=False)
            )
            top  = list(zip(grp.head(top_n)["segment"].astype(str), grp.head(top_n)["value"]))
            bot  = list(zip(grp.tail(top_n)["segment"].astype(str), grp.tail(top_n)["value"]))
            c1, c2 = st.columns(2)
            with c1:
                _pchart(_chart_bar(top, f"Top {top_n} · {metric}", T["success"]), key="adv_seg_top")
            with c2:
                _pchart(_chart_donut(df, group_by, metric, top_n), key="adv_seg_donut")
            _pchart(_chart_bar(bot, f"Bottom {top_n} · {metric}", T["crit"]), key="adv_seg_bot")
        else:
            st.info("Select valid Metric and Group By columns for Segment Analysis.")


def _dash_perf_heatmap(dash: DashboardPayload) -> None:
    pm = dash.performance_matrix
    if pm is None or pm.pivot.empty:
        return
    _sec(f"Performance Heatmap · {pm.segment_col} x Time")
    _pchart(_chart_performance_heatmap(pm.pivot), key="dash_perf_heatmap")


def _dash_alerts(dash: DashboardPayload) -> None:
    _sec("Alert Feed")
    if not dash.alerts:
        st.success("No alerts generated.")
        return
    for alert in dash.alerts:
        badge     = _badge(alert.severity)
        cat_color = {
            "quality": T["warn"], "risk": T["crit"],
            "trend": T["a2"],     "anomaly": T["warn"],
        }.get(alert.category, T["muted"])
        st.markdown(
            f'<div class="alert-row">{badge}'
            f'<span style="color:{cat_color};font-weight:600;min-width:60px">'
            f'{alert.category}</span>'
            f'<span style="color:{T["text"]}">{alert.message}</span></div>',
            unsafe_allow_html=True,
        )


def _dash_ai(dash: DashboardPayload, ai_engine: "AIEngine | None") -> None:
    _sec("AI Executive Intelligence")
    st.markdown(
        f'<div style="font-size:12px;color:{T["muted"]};margin-bottom:16px;line-height:1.6">'
        "Structured context derived from analytics — no raw data transmitted.</div>",
        unsafe_allow_html=True,
    )
    if ai_engine is None:
        st.info("AI engine not available. Add GROQ_API_KEY to .streamlit/secrets.toml to enable.")
        return
    ctx  = dash.ai_context
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    with b1: exec_btn  = st.button("Executive Brief", key="ai_exec")
    with b2: risk_btn  = st.button("Risk Exposure",   key="ai_risk")
    with b3: board_btn = st.button("Board Summary",   key="ai_board")
    for btn, label, method in [
        (exec_btn,  "Executive Brief", "generate_executive_summary"),
        (risk_btn,  "Risk Exposure",   "explain_quality"),
        (board_btn, "Board Summary",   "explain_suitability"),
    ]:
        if btn:
            with st.spinner(f"Generating {label}…"):
                try:
                    response = getattr(ai_engine, method)(ctx)
                    st.markdown(
                        f'<div class="ai-block"><div class="ai-label">'
                        f'AI INTELLIGENCE · {label.upper()}</div>{response}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"AI Engine error: {exc}")


# ── Dashboard Narrative ───────────────────────────────────────────────────────

def _build_narrative_context(
    eda: EDAPayload,
    dash: DashboardPayload,
    df: pd.DataFrame,
    charts: dict,
) -> dict:
    """Assemble a compact analytics context dict for the narrative prompt."""
    ctx: dict = {
        "rows":    int(len(df)),
        "columns": int(df.shape[1]),
        "health_score":       round(float(dash.kpis.health_score), 1),
        "growth_signal":      round(float(dash.kpis.growth_signal), 2),
        "concentration_risk": round(float(dash.kpis.concentration_risk), 4),
        "volatility_score":   round(float(dash.kpis.volatility_score), 4),
        "anomaly_pressure":   round(float(dash.kpis.anomaly_pressure), 4),
        "missing_pct":        round(float(eda.quality.missing_pct), 2),
        "duplicate_pct":      round(float(eda.quality.duplicate_pct), 2),
    }

    # Correlation top pairs
    if eda.correlation and eda.correlation.strongest_pairs:
        ctx["top_correlations"] = [
            {"a": a, "b": b, "r": round(r, 4)}
            for a, b, r in eda.correlation.strongest_pairs[:5]
        ]

    # Segment data
    sr = eda.segment_ranking
    if sr is not None:
        ctx["segment_col"]      = sr.segment_col
        ctx["metric_col"]       = sr.metric_col
        ctx["top_segment"]      = str(sr.top_10.iloc[0]["segment"]) if not sr.top_10.empty else None
        ctx["bottom_segment"]   = str(sr.bottom_10.iloc[0]["segment"]) if not sr.bottom_10.empty else None
        ctx["top_segment_val"]  = round(float(sr.top_10.iloc[0]["value"]), 4) if not sr.top_10.empty else None
        ctx["bot_segment_val"]  = round(float(sr.bottom_10.iloc[0]["value"]), 4) if not sr.bottom_10.empty else None

    # Time trend
    tt = eda.time_trend
    if tt:
        ctx["trend_direction"]    = tt.direction
        ctx["trend_pct_change"]   = round(float(tt.pct_change_overall), 2)
        ctx["trend_metric"]       = tt.metric_col
        ctx["trend_freq"]         = tt.freq

    # Anomalies
    sig_anomalies = [a for a in eda.anomalies if a.count > 0]
    if sig_anomalies:
        ctx["anomaly_columns"] = [
            {"col": a.column, "count": a.count}
            for a in sig_anomalies[:5]
        ]

    # Alerts
    if dash.alerts:
        ctx["alerts"] = [
            {"severity": al.severity, "category": al.category, "message": al.message}
            for al in dash.alerts[:6]
        ]

    # Skewness highlights
    if eda.skew_ranking:
        ctx["high_skew"] = [
            {"col": c, "skew": round(s, 3)}
            for c, s in eda.skew_ranking[:5]
            if abs(s) > 1.0
        ]

    return ctx


def _render_dashboard_narrative(
    eda: EDAPayload,
    dash: DashboardPayload,
    df: pd.DataFrame,
    charts: dict,
    ai_engine: "AIEngine | None",
) -> None:
    """
    Dashboard Narrative section — AI-generated summary of key chart observations.
    Uses a single generate() call with the full analytics context.
    """
    _sec("Dashboard Narrative")

    if ai_engine is None:
        st.info("Connect an AI engine to generate the dashboard narrative.")
        return

    if st.button("Generate Dashboard Narrative", key="btn_dash_narrative"):
        ctx     = _build_narrative_context(eda, dash, df, charts)
        import json
        ctx_str = json.dumps(ctx, indent=2, default=str)[:3200]

        prompt = (
            "You are a senior data analyst writing a concise dashboard narrative "
            "for an executive audience.\n\n"
            "Analytics context:\n"
            f"{ctx_str}\n\n"
            "Write a structured narrative with exactly these four sections. "
            "Return only plain text — no markdown, no JSON, no headers with # symbols.\n\n"
            "KEY OBSERVATIONS\n"
            "2-3 sentences on the most important patterns visible across the charts.\n\n"
            "SEGMENT PERFORMANCE\n"
            "2-3 sentences comparing top and bottom performing segments. "
            "Name segments and quantify gaps where data is available.\n\n"
            "RISK SIGNALS\n"
            "2-3 sentences on anomalies, volatility, missing data, or concentration risks "
            "visible in the data.\n\n"
            "RECOMMENDED ACTIONS\n"
            "3 specific, numbered actions the business should take based on these observations. "
            "Each action must reference a specific data point."
        )

        with st.spinner("Generating narrative…"):
            try:
                raw = ai_engine.generate(prompt, max_tokens=600)
                st.session_state["dashboard_narrative"] = raw
            except Exception as exc:
                st.error(f"Narrative generation failed: {exc}")
                return

    if "dashboard_narrative" not in st.session_state:
        return

    raw_narrative = st.session_state["dashboard_narrative"]

    # Parse the four sections from plain-text response
    section_map = {
        "KEY OBSERVATIONS":   ("key_observations",   "Key Observations"),
        "SEGMENT PERFORMANCE": ("segment_performance", "Segment Performance"),
        "RISK SIGNALS":        ("risk_signals",        "Risk Signals"),
        "RECOMMENDED ACTIONS": ("recommended_actions", "Recommended Actions"),
    }

    def _extract_section(text: str, heading: str, next_headings: list[str]) -> str:
        import re
        pattern = rf"{re.escape(heading)}\s*\n(.*?)(?={'|'.join(re.escape(h) for h in next_headings) + '|$'})"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    headings = list(section_map.keys())
    sections = {}
    for i, heading in enumerate(headings):
        next_h = headings[i + 1:] if i < len(headings) - 1 else ["ZZEND"]
        sections[heading] = _extract_section(raw_narrative, heading, next_h)

    # Fallback: show raw if parsing found nothing
    any_found = any(v for v in sections.values())

    st.markdown('<div class="narrative-block">', unsafe_allow_html=True)
    st.markdown(
        '<div class="narrative-title">Dashboard Narrative</div>',
        unsafe_allow_html=True,
    )

    if any_found:
        for heading, (key, title) in section_map.items():
            content = sections.get(heading, "").strip()
            if not content:
                continue
            if key == "recommended_actions":
                # Render numbered items as chips
                import re
                items = re.split(r"\n?\d+\.\s+", content)
                items = [it.strip() for it in items if it.strip()]
                st.markdown(
                    f'<div class="narrative-section">'
                    f'<div class="narrative-section-label">{title}</div>'
                    f'<div>'
                    + "".join(f'<span class="rec-chip">{it}</span>' for it in items)
                    + '</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="narrative-section">'
                    f'<div class="narrative-section-label">{title}</div>'
                    f'<div class="narrative-section-text">{content}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        # Fallback: display raw response
        st.markdown(
            f'<div class="narrative-section-text" style="white-space:pre-wrap">'
            f'{raw_narrative}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ── Public entry point ────────────────────────────────────────────────────────

def render_dashboard(df: pd.DataFrame, ai_engine: "AIEngine | None" = None) -> None:
    """Main entry point. df = validated dataset, ai_engine = injected AIEngine."""
    _css()
    try:
        detection = UniversalDetector().detect(df)
        eda       = EDAEngine().compute(df, detection)
        dash      = DashboardEngine().compute(df, eda, detection)
    except Exception as exc:
        st.error(f"Analytics pipeline error: {exc}")
        return

    # Chart recommendation engine — determines which chart types to render
    charts = _recommend_charts(eda, df)

    tab_a, tab_b = st.tabs(["  EDA Intelligence", "  Executive Dashboard"])

    # ── Tab A: EDA Intelligence ───────────────────────────────────────────────
    with tab_a:
        st.markdown('<div class="ix-tab-header">EDA Intelligence</div>', unsafe_allow_html=True)

        _card_open(); _eda_quality(eda); _card_close(); _div()
        _card_open(); _eda_distribution(eda, df, charts); _card_close(); _div()
        _card_open(); _eda_correlation(eda, df, charts); _card_close(); _div()

        if charts["segment_bar"]:
            _card_open(); _eda_segments(eda, df, charts); _card_close(); _div()

        if charts["time_trend"]:
            _card_open(); _eda_time_trend(eda, charts); _card_close(); _div()

        if charts["anomaly"]:
            _card_open(); _eda_anomalies(eda, df); _card_close(); _div()

        # KPI Sparklines at the bottom of EDA
        _card_open(); _render_kpi_sparklines(df, eda, dash); _card_close()

    # ── Tab B: Executive Dashboard ────────────────────────────────────────────
    with tab_b:
        overrides = _render_adaptive_controls(eda, detection)
        _div()
        _dash_header(dash, overrides)
        _div()

        if overrides["advanced_mode"]:
            _card_open(); _dash_advanced_view(df, overrides, eda, charts); _card_close(); _div()

        _card_open()
        _dash_performers(dash, df, eda, charts, top_n=overrides["top_n"])
        _card_close(); _div()

        if charts["scatter_pairs"] and eda.correlation and eda.correlation.strongest_pairs:
            _card_open()
            _sec("Relationship Charts — Top Correlated Pairs")
            pairs     = eda.correlation.strongest_pairs[:4]
            pair_cols = st.columns(2)
            cat_col   = eda.categorical_cols[0] if eda.categorical_cols else None
            for i, (col_a, col_b, r_val) in enumerate(pairs):
                with pair_cols[i % 2]:
                    r_color = T["crit"] if abs(r_val) >= 0.75 else T["warn"] if abs(r_val) >= 0.5 else T["success"]
                    st.markdown(
                        f'<div style="font-size:10px;color:{T["muted"]};margin-bottom:4px">'
                        f'{col_a} <span style="color:{T["a1"]}">↔</span> {col_b} '
                        f'<span style="color:{r_color};font-weight:700">r={r_val:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _pchart(
                        _chart_scatter(df, col_a, col_b, color_col=cat_col),
                        key=f"dash_scatter_{i}",
                    )
            _card_close(); _div()

        if dash.performance_matrix:
            _card_open(); _dash_perf_heatmap(dash); _card_close(); _div()

        _card_open(); _dash_alerts(dash); _card_close(); _div()

        _card_open()
        _render_dashboard_narrative(eda, dash, df, charts, ai_engine)
        _card_close(); _div()

        _card_open(); _dash_ai(dash, ai_engine); _card_close()

    # ── NEXT NAVIGATION ──────────────────────────────────────
    st.divider()
    col_left, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["active_tab"] = "Decision Lab"
            st.rerun()    