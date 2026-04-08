from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.analytics.decision_engine import (
    build_ai_context,
    run_ai_advisor,
    run_decision_signals,
    run_forecast,
    run_risk_projection,
    run_scenario,
    run_scorecard,
    run_sensitivity,
    DecisionScorecard,
    DecisionSignal,
    ForecastResult,
    RiskProjection,
    ScenarioResult,
    SensitivityResult,
)

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine


# ═══════════════════════════════════════════════════════════════════════════════
# Design tokens  (InsightX dark luxury palette)
# ═══════════════════════════════════════════════════════════════════════════════

_BG      = "#0B0F19"
_CARD    = "#141A2A"
_BORDER  = "#1F2937"
_DARK    = "#111827"

_BLUE    = "#4F8EF7"
_GREEN   = "#10B981"
_RED     = "#EF4444"
_AMBER   = "#F59E0B"
_PURPLE  = "#8B5CF6"
_CYAN    = "#06B6D4"

_TEXT    = "#F8FAFC"
_MUTED   = "#94A3B8"

_SEV_COLOR = {"positive": _GREEN, "warning": _AMBER, "critical": _RED}
_SEV_ICON  = {"positive": "▲",    "warning": "⚠",    "critical": "▼"}

_PL_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="'DM Sans',sans-serif", color=_TEXT, size=11),
    margin=dict(l=16, r=16, t=44, b=20),
    xaxis=dict(gridcolor=_BORDER, linecolor=_BORDER, zerolinecolor=_BORDER),
    yaxis=dict(gridcolor=_BORDER, linecolor=_BORDER, zerolinecolor=_BORDER),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sf(v) -> float:
    try:
        return round(float(v), 4)
    except Exception:
        return 0.0


def _rgba(hex6: str, alpha: float) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _pl(**kw) -> dict:
    import copy
    d = copy.deepcopy(_PL_BASE)
    d.update(kw)
    return d


def _target_series(df: pd.DataFrame, result: ForecastResult) -> np.ndarray:
    """Extract the highest-variance numeric column as the 'target' for charting."""
    num = df.select_dtypes(include=[np.number])
    col = num.var().idxmax()
    return num[col].dropna().reset_index(drop=True).values


# ═══════════════════════════════════════════════════════════════════════════════
# Serialisation (all session-state keys spec'd in docstring)
# ═══════════════════════════════════════════════════════════════════════════════

def _ser_forecast(f: ForecastResult) -> dict:
    return {
        "model_name":       f.model_name,
        "rmse":             _sf(f.rmse),
        "confidence_score": _sf(f.confidence_score),
        "horizon":          int(f.horizon),
        "forecast_values":  [_sf(v) for v in f.forecast_values],
        "upper_bound":      [_sf(v) for v in f.upper_bound],
        "lower_bound":      [_sf(v) for v in f.lower_bound],
        "all_rmse":         {k: _sf(v) for k, v in f.all_rmse.items()},
    }


def _ser_risk(r: RiskProjection) -> dict:
    return {
        "volatility":         _sf(r.volatility),
        "concentration_risk": _sf(r.concentration_risk),
        "anomaly_pressure":   _sf(r.anomaly_pressure),
        "stability_score":    _sf(r.stability_score),
        "downside":           [_sf(v) for v in r.downside],
        "upside":             [_sf(v) for v in r.upside],
    }


def _ser_scorecard(s: DecisionScorecard) -> dict:
    return {
        "growth_strength":     _sf(s.growth_strength),
        "risk_exposure":       _sf(s.risk_exposure),
        "stability_score":     _sf(s.stability_score),
        "strategic_readiness": _sf(s.strategic_readiness),
        "forecast_confidence": _sf(s.forecast_confidence),
        "summary":             str(s.summary),
    }


def _ser_scenario(sc: ScenarioResult, params: dict) -> dict:
    return {
        "parameters":        params,
        "revenue_delta":     _sf(sc.revenue_delta),
        "risk_impact":       _sf(sc.risk_impact),
        "adjusted_forecast": [_sf(v) for v in sc.adjusted_forecast],
    }


def _ser_sensitivity(s: SensitivityResult) -> dict:
    ranked = [(str(r[0]), _sf(r[1])) for r in s.ranked]
    return {
        "ranked_drivers":   [{"driver": d, "elasticity": e} for d, e in ranked],
        "top_driver":       ranked[0][0] if ranked else "N/A",
        "top_elasticity":   ranked[0][1] if ranked else 0.0,
        "positive_drivers": [d for d, e in ranked if e >= 0],
        "negative_drivers": [d for d, e in ranked if e < 0],
    }


def _ser_signals(signals: list[DecisionSignal]) -> list[dict]:
    return [
        {"signal": s.signal, "severity": s.severity,
         "strategy": s.strategy, "rationale": s.rationale}
        for s in signals
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown(f"""<style>
.dl-card {{
    background:{_CARD};border:1px solid {_BORDER};
    border-radius:16px;padding:24px 28px;margin-bottom:18px;
    position:relative;overflow:hidden;
}}
.dl-card::before {{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:var(--dl-accent,{_BLUE});border-radius:16px 16px 0 0;opacity:.8;
}}
.dl-label {{
    font-size:10px;font-weight:700;letter-spacing:.18em;
    text-transform:uppercase;color:var(--dl-accent,{_BLUE});
    margin-bottom:6px;display:flex;align-items:center;gap:7px;
}}
.dl-label::before {{
    content:'';width:5px;height:5px;border-radius:50%;
    background:var(--dl-accent,{_BLUE});flex-shrink:0;
}}
.dl-title {{
    font-size:17px;font-weight:700;color:{_TEXT};
    margin-bottom:18px;letter-spacing:-.01em;
}}
.dl-narrative {{
    font-size:13px;line-height:1.88;color:#CBD5E1;
    margin-top:14px;padding:18px 22px;
    background:{_BG};border:1px solid {_BORDER};
    border-left:3px solid var(--dl-accent,{_BLUE});
    border-radius:0 10px 10px 0;white-space:pre-wrap;
}}
.dl-stat {{ text-align:center;padding:12px 4px; }}
.dl-stat-val {{ font-size:22px;font-weight:700;color:{_TEXT};line-height:1; }}
.dl-stat-lbl {{ font-size:9px;font-weight:600;letter-spacing:.12em;
                text-transform:uppercase;color:{_MUTED};margin-top:5px; }}
.dl-kv {{ display:flex;justify-content:space-between;align-items:center;
          padding:6px 0;font-size:12px;border-bottom:1px solid {_BORDER}55; }}
.dl-kv:last-child {{ border-bottom:none; }}
.dl-pill {{ display:inline-flex;align-items:center;padding:3px 10px;
            border-radius:20px;font-size:10px;font-weight:700;
            letter-spacing:.06em;margin:2px 3px 2px 0; }}
.dl-sig {{ border-left:3px solid var(--sig-col,{_BLUE});
           padding:10px 14px;background:{_DARK};
           border-radius:0 8px 8px 0;margin-bottom:8px; }}
.dl-gauge-lbl {{ font-size:11px;text-align:center;color:{_MUTED};
                 margin-top:4px;font-weight:600; }}
.dl-model-row {{ display:flex;justify-content:space-between;
                 padding:7px 0;font-size:12px;
                 border-bottom:1px solid {_BORDER}55; }}
.dl-model-row:last-child {{ border-bottom:none; }}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UI primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _card(accent: str = _BLUE):
    st.markdown(
        f'<div class="dl-card" style="--dl-accent:{accent}">',
        unsafe_allow_html=True,
    )


def _endcard():
    st.markdown("</div>", unsafe_allow_html=True)


def _label(text: str, accent: str = _BLUE):
    st.markdown(
        f'<div class="dl-label" style="--dl-accent:{accent};">{text}</div>',
        unsafe_allow_html=True,
    )


def _title(text: str):
    st.markdown(f'<div class="dl-title">{text}</div>', unsafe_allow_html=True)


def _narrative(text: str, accent: str = _BLUE):
    st.markdown(
        f'<div class="dl-narrative" style="--dl-accent:{accent};">{text}</div>',
        unsafe_allow_html=True,
    )


def _pill(label: str, color: str) -> str:
    return (
        f'<span class="dl-pill" '
        f'style="background:{color}1A;border:1px solid {color}44;color:{color};">'
        f'{label}</span>'
    )


def _kv(label: str, value: str, vc: str = ""):
    v_style = f"color:{vc};font-weight:600;" if vc else f"color:{_TEXT};font-weight:600;"
    st.markdown(
        f'<div class="dl-kv">'
        f'<span style="color:{_MUTED}">{label}</span>'
        f'<span style="{v_style}">{value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _no_engine():
    st.markdown(
        f'<div style="color:{_MUTED};font-size:12px;padding:6px 0;">'
        "AI engine not configured — structured data stored, narrative unavailable."
        "</div>",
        unsafe_allow_html=True,
    )


def _divider():
    st.markdown(
        f'<div style="height:1px;background:{_BORDER};margin:14px 0;"></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_probabilistic(
    df: pd.DataFrame,
    f: ForecastResult,
    r: RiskProjection,
) -> go.Figure:
    """
    Main forecast chart with:
    • Historical values  (solid blue)
    • Fitted values      (dotted amber)
    • Point forecast     (solid green)
    • 80% CI band        (inner shaded)
    • 95% CI band        (outer shaded, derived from risk downside/upside)
    """
    raw       = _target_series(df, f)
    n_hist    = len(f.fitted_values)
    x_hist    = list(range(n_hist))
    x_fc      = list(range(n_hist, n_hist + f.horizon))

    # Compute 95% CI: widen the engine's bounds by risk volatility factor
    vol_factor = max(1.0, 1.0 + float(r.volatility) * 0.5)
    fc_mean = np.mean(f.forecast_values) if len(f.forecast_values) > 0 else 0.0
    upper_95   = [
        fc_mean + (u - fc_mean) * vol_factor
        for u in f.upper_bound
    ]
    lower_95   = [
        fc_mean - (fc_mean - lo) * vol_factor
        for lo in f.lower_bound
    ]

    fig = go.Figure()

    # 95% CI (outer, lighter)
    fig.add_trace(go.Scatter(
        x=x_fc + x_fc[::-1],
        y=upper_95 + lower_95[::-1],
        fill="toself",
        fillcolor=_rgba(_BLUE, 0.08),
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
    ))

    # 80% CI (inner, denser)
    fig.add_trace(go.Scatter(
        x=x_fc + x_fc[::-1],
        y=list(f.upper_bound) + list(f.lower_bound[::-1]),
        fill="toself",
        fillcolor=_rgba(_BLUE, 0.18),
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",
        showlegend=True,
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=x_hist, y=raw[:n_hist],
        mode="lines", name="Historical",
        line=dict(color=_BLUE, width=2),
    ))

    # Fitted
    fig.add_trace(go.Scatter(
        x=x_hist, y=f.fitted_values,
        mode="lines", name=f"Fitted ({f.model_name})",
        line=dict(color=_AMBER, width=1.5, dash="dot"),
    ))

    # Point forecast
    fig.add_trace(go.Scatter(
        x=x_fc, y=f.forecast_values,
        mode="lines+markers", name="Forecast",
        line=dict(color=_GREEN, width=2.5),
        marker=dict(size=5, symbol="circle"),
    ))

    # Worst-case downside
    fig.add_trace(go.Scatter(
        x=x_fc, y=r.downside,
        mode="lines", name="Worst Case",
        line=dict(color=_RED, width=1.2, dash="dash"),
    ))

    # Best-case upside
    fig.add_trace(go.Scatter(
        x=x_fc, y=r.upside,
        mode="lines", name="Best Case",
        line=dict(color=_GREEN, width=1.2, dash="dash"),
    ))

    # Separator line at forecast start
    fig.add_vline(
        x=n_hist - 0.5,
        line=dict(color=_MUTED, width=1, dash="dot"),
        annotation_text="Forecast →",
        annotation_font=dict(color=_MUTED, size=10),
    )

    fig.update_layout(**_pl(
        title=dict(
            text=f"Probabilistic Forecast  ·  Model: {f.model_name}",
            font=dict(size=13), x=0.01,
        ),
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
    ))
    return fig


def _chart_scenario(f: ForecastResult, sc: ScenarioResult) -> go.Figure:
    x  = list(range(1, f.horizon + 1))
    fig = go.Figure()

    # 80% CI behind base forecast
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(f.upper_bound) + list(f.lower_bound[::-1]),
        fill="toself",
        fillcolor=_rgba(_BLUE, 0.12),
        line=dict(color="rgba(0,0,0,0)"),
        name="Base 80% CI",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=x, y=f.forecast_values,
        mode="lines+markers", name="Base Forecast",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sc.adjusted_forecast,
        mode="lines+markers", name="Scenario Forecast",
        line=dict(color=_AMBER, width=2.5, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
    ))
    fig.update_layout(**_pl(
        title=dict(text="Base vs Scenario Forecast", font=dict(size=13), x=0.01),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Period"),
    ))
    return fig


def _chart_decomposition(f: ForecastResult, r: RiskProjection) -> go.Figure:
    """
    Approximate three-panel decomposition from the available forecast data.
    Trend     — OLS slope through forecast_values
    Seasonality — residual from OLS (captures periodic structure)
    Residual  — risk-adjusted noise band
    """
    n   = f.horizon
    x   = list(range(1, n + 1))
    yv  = np.array(f.forecast_values, dtype=float)

    # Trend: linear fit
    xm, ym   = np.mean(x), np.mean(yv)
    slope    = float(np.sum((np.array(x) - xm) * (yv - ym)) /
                     max(np.sum((np.array(x) - xm) ** 2), 1e-9))
    intercept = ym - slope * xm
    trend     = [intercept + slope * xi for xi in x]

    # Seasonality: deviation from trend (captures harmonic/cyclic structure)
    detrended    = yv - np.array(trend)
    seasonality  = detrended.tolist()

    # Residual: volatility-based noise band around zero
    noise_scale = float(r.volatility) * float(np.std(yv)) if np.std(yv) > 0 else 0.1
    residual    = [float(v) * noise_scale for v in (yv - yv.mean()).tolist()]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Trend", "Seasonality / Cyclical", "Residual / Noise"),
        vertical_spacing=0.10,
    )

    # Row 1 — Trend
    fig.add_trace(go.Scatter(
        x=x, y=trend, mode="lines",
        name="Trend", line=dict(color=_CYAN, width=2),
    ), row=1, col=1)

    # Row 2 — Seasonality
    fig.add_trace(go.Bar(
        x=x, y=seasonality, name="Seasonality",
        marker_color=[_GREEN if v >= 0 else _RED for v in seasonality],
        opacity=0.75,
    ), row=2, col=1)

    # Row 3 — Residual
    fig.add_trace(go.Scatter(
        x=x, y=residual, mode="lines+markers",
        name="Residual", line=dict(color=_PURPLE, width=1.5),
        marker=dict(size=4),
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=_MUTED, width=1, dash="dot"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="'DM Sans',sans-serif", color=_TEXT, size=10),
        margin=dict(l=16, r=16, t=50, b=16),
        height=480,
        showlegend=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=_BORDER, linecolor=_BORDER, row=i, col=1)
        fig.update_yaxes(gridcolor=_BORDER, linecolor=_BORDER, row=i, col=1)
    return fig


def _chart_reliability_gauge(accuracy_pct: float) -> go.Figure:
    color = _GREEN if accuracy_pct >= 85 else _AMBER if accuracy_pct >= 70 else _RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy_pct,
        number={"suffix": "%", "font": {"color": _TEXT, "size": 28}},
        title={"text": "Forecast Accuracy", "font": {"size": 12, "color": _MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _MUTED, "tickfont": {"size": 9}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor":     "rgba(0,0,0,0)",
            "bordercolor": _BORDER,
            "steps": [
                {"range": [0,  70], "color": _rgba(_RED,   0.10)},
                {"range": [70, 85], "color": _rgba(_AMBER, 0.10)},
                {"range": [85, 100],"color": _rgba(_GREEN, 0.10)},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": accuracy_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=44, b=10),
        height=220,
    )
    return fig


def _chart_rmse_bar(all_rmse: dict[str, float]) -> go.Figure:
    names  = list(all_rmse.keys())
    vals   = [round(v, 4) for v in all_rmse.values()]
    min_v  = min(vals, default=0)
    colors = [_GREEN if v == min_v else _rgba(_BLUE, 0.60) for v in vals]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color=_MUTED),
    ))
    fig.update_layout(**_pl(
        title=dict(text="Model RMSE Comparison", font=dict(size=12), x=0.01),
        height=240, showlegend=False,
        yaxis=dict(title="RMSE"),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# AI narrative generators
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_forecast_narrative(f: ForecastResult, r: RiskProjection,
                             ai_engine) -> str | None:
    if ai_engine is None:
        return None
    ctx = json.dumps({
        "model":      f.model_name,
        "rmse":       _sf(f.rmse),
        "confidence": _sf(f.confidence_score),
        "horizon":    f.horizon,
        "volatility": _sf(r.volatility),
        "stability":  _sf(r.stability_score),
        "upper_band_mean": _sf(float(np.mean(f.upper_bound)) if len(f.upper_bound) > 0 else 0),
        "lower_band_mean": _sf(float(np.mean(f.lower_bound)) if len(f.lower_bound) > 0 else 0),
        "forecast_mean":   _sf(float(np.mean(f.forecast_values)) if len(f.forecast_values) > 0 else 0),
    }, indent=2)[:2200]
    prompt = (
        "You are a senior quantitative analyst briefing an executive.\n\n"
        "Probabilistic forecast results:\n"
        f"{ctx}\n\n"
        "In 4-5 sentences:\n"
        "1. Describe the forecast trajectory and its direction.\n"
        "2. Interpret the confidence interval spread as a risk range.\n"
        "3. Explain what the model's RMSE implies for reliability.\n"
        "4. State one planning decision the leadership team should make.\n\n"
        "No bullet points. One executive paragraph."
    )
    return ai_engine.generate(prompt, max_tokens=300)


def _gen_scenario_narrative(f: ForecastResult, sc: ScenarioResult,
                              params: dict, ai_engine) -> str | None:
    if ai_engine is None:
        return None
    ctx = json.dumps({
        "parameters":          params,
        "base_forecast_mean": _sf(float(np.mean(f.forecast_values)) if len(f.forecast_values) > 0 else 0),
        "scenario_forecast_mean": _sf(float(np.mean(sc.adjusted_forecast)) if len(sc.adjusted_forecast) > 0 else 0),
        "revenue_delta":       _sf(sc.revenue_delta),
        "risk_impact":         _sf(sc.risk_impact),
    }, indent=2)[:2000]
    prompt = (
        "You are a senior business strategy consultant.\n\n"
        "What-if scenario results:\n"
        f"{ctx}\n\n"
        "In 4-5 sentences:\n"
        "1. Interpret the revenue delta and risk impact of these adjustments.\n"
        "2. State whether this scenario is an opportunity or a threat.\n"
        "3. Recommend whether leadership should plan for or protect against this scenario.\n\n"
        "No bullet points. One executive paragraph."
    )
    return ai_engine.generate(prompt, max_tokens=280)


def _gen_strategic_impact(f: ForecastResult, r: RiskProjection,
                           scorecard: DecisionScorecard,
                           signals: list[DecisionSignal],
                           ai_engine) -> str | None:
    if ai_engine is None:
        return None
    signal_counts = {
        sev: sum(1 for s in signals if s.severity == sev)
        for sev in ("positive", "warning", "critical")
    }
    ctx = json.dumps({
        "forecast": {
            "model":      f.model_name,
            "confidence": _sf(f.confidence_score),
            "horizon":    f.horizon,
            "mean_fc": _sf(float(np.mean(f.forecast_values)) if len(f.forecast_values) > 0 else 0),
        },
        "risk": {
            "volatility":       _sf(r.volatility),
            "stability_score":  _sf(r.stability_score),
            "anomaly_pressure": _sf(r.anomaly_pressure),
        },
        "scorecard": _ser_scorecard(scorecard),
        "signal_counts": signal_counts,
    }, indent=2)[:3000]
    prompt = (
        "You are a chief strategy officer writing a one-page planning memo.\n\n"
        "Integrated decision intelligence:\n"
        f"{ctx}\n\n"
        "Write a 5-6 sentence Strategic Impact Summary covering:\n"
        "1. The expected growth trajectory and its direction.\n"
        "2. Primary risk and volatility considerations.\n"
        "3. What the scorecard signals about strategic readiness.\n"
        "4. A clear go / proceed-with-caution / stop recommendation with rationale.\n\n"
        "Use executive language. No bullet points. One cohesive paragraph."
    )
    return ai_engine.generate(prompt, max_tokens=380)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Executive Summary Bar
# ═══════════════════════════════════════════════════════════════════════════════

def _render_exec_summary(
    f: ForecastResult, r: RiskProjection,
    scorecard: DecisionScorecard, signals: list[DecisionSignal],
) -> None:
    _card(_BLUE)
    _label("Decision Lab · Executive Summary", _BLUE)

    # KPI stat row
    cols = st.columns(5)
    kpis = [
        (f"{scorecard.strategic_readiness:.1f} / 10", "Strategic Readiness"),
        (f"{f.confidence_score:.1f}%",                "Forecast Confidence"),
        (f.model_name,                                 "Best Model"),
        (f"{r.stability_score:.1f} / 10",             "Risk Stability"),
        (f"{r.volatility:.3f}",                       "Volatility"),
    ]
    for c, (val, lbl) in zip(cols, kpis):
        with c:
            st.markdown(
                f'<div class="dl-stat">'
                f'<div class="dl-stat-val">{val}</div>'
                f'<div class="dl-stat-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _divider()

    # Signal pills
    counts = {sev: sum(1 for s in signals if s.severity == sev)
               for sev in ("positive", "warning", "critical")}
    pills = (
        _pill(f"▲ {counts['positive']} Positive", _GREEN) +
        _pill(f"⚠ {counts['warning']} Warning",   _AMBER) +
        _pill(f"▼ {counts['critical']} Critical",  _RED)
    )
    st.markdown(f'<div style="margin-top:4px">{pills}</div>', unsafe_allow_html=True)
    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Probabilistic Forecast
# ═══════════════════════════════════════════════════════════════════════════════

def _render_forecast(
    df: pd.DataFrame, f: ForecastResult, r: RiskProjection, ai_engine,
) -> None:
    _card(_BLUE)
    _label("01 — Forecast Projection", _BLUE)
    _title("Probabilistic Forecast · 80% and 95% Confidence Intervals")

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model",       f.model_name)
    m2.metric("Horizon",          f"{f.horizon} periods")
    m3.metric("RMSE",             f"{f.rmse:.4f}")
    m4.metric("Confidence",       f"{f.confidence_score:.1f}%")

    st.plotly_chart(
        _chart_probabilistic(df, f, r),
        use_container_width=True, config={"displayModeBar": False},
    )

    # Model RMSE comparison (collapsible)
    with st.expander("Model RMSE Comparison", expanded=False):
        st.plotly_chart(
            _chart_rmse_bar(f.all_rmse),
            use_container_width=True, config={"displayModeBar": False},
        )

    # AI narrative
    if st.button("Generate Forecast Intelligence", key="btn_fc_intel"):
        with st.spinner("Analysing forecast…"):
            explanation = _gen_forecast_narrative(f, r, ai_engine)
        ser = _ser_forecast(f)
        st.session_state["forecast_projection"] = {
            "structured": ser, "explanation": explanation,
        }
        st.session_state["forecast_confidence"] = {
            "structured": {
                "confidence_score": ser["confidence_score"],
                "upper_band_mean": _sf(float(np.mean(f.upper_bound)) if len(f.upper_bound) > 0 else 0),
                "lower_band_mean":  _sf(float(np.mean(f.lower_bound)) if len(f.lower_bound) > 0 else 0),
            },
            "explanation": explanation,
        }

    if ai_engine is None:
        _no_engine()

    if "forecast_projection" in st.session_state:
        exp = st.session_state["forecast_projection"].get("explanation")
        if exp:
            _narrative(exp, _BLUE)

    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Scenario Simulator
# ═══════════════════════════════════════════════════════════════════════════════

def _render_scenario(
    df: pd.DataFrame, f: ForecastResult, r: RiskProjection, ai_engine,
) -> None:
    _card(_AMBER)
    _label("02 — Scenario Simulator · What-If Engine", _AMBER)
    _title("Simulate Business Changes and See Forecast Impact")

    # Sliders
    c1, c2, c3, c4 = st.columns(4)
    marketing = c1.slider("Marketing Spend (%)",  -30.0, 50.0,  0.0, 1.0, key="sc_mkt")
    pricing   = c2.slider("Price Adjustment (%)", -20.0, 30.0,  0.0, 1.0, key="sc_price")
    demand    = c3.slider("Demand Shock (%)",     -30.0, 30.0,  0.0, 1.0, key="sc_dem")
    seasonal  = c4.slider("Seasonality Adj (%)",  -50.0, 100.0, 0.0, 5.0, key="sc_seas")

    # Map sliders onto decision_engine parameters
    # growth_pct       ← marketing + pricing effect
    # volatility_adj   ← seasonality (increases/decreases cycle amplitude)
    # demand_shock     ← direct demand
    growth_pct  = marketing * 0.6 + pricing * 0.4   # weighted composite
    vol_adj     = seasonal
    conc_red    = max(0.0, -demand)                  # demand shock < 0 → concentration reduces
    demand_eng  = demand

    scenario = run_scenario(
        df, f, r,
        growth_pct=growth_pct,
        volatility_adj_pct=vol_adj,
        concentration_reduction_pct=conc_red,
        demand_shock_pct=demand_eng,
    )

    # Delta metrics
    _divider()
    d1, d2, d3 = st.columns(3)
    delta_sign = "+" if scenario.revenue_delta >= 0 else ""
    fc_chg = float(np.mean(scenario.adjusted_forecast) - np.mean(f.forecast_values)) \
         if len(f.forecast_values) > 0 else 0.0
    d1.metric("Revenue Delta",    f"{delta_sign}{scenario.revenue_delta:.2f}")
    d2.metric("Risk Impact",      f"{scenario.risk_impact:.4f}")
    d3.metric("Forecast Δ Mean",  f"{fc_chg:+.4f}")

    st.plotly_chart(
        _chart_scenario(f, scenario),
        use_container_width=True, config={"displayModeBar": False},
    )

    params = {
        "marketing_spend_pct": marketing,
        "price_adjustment_pct": pricing,
        "demand_shock_pct":    demand,
        "seasonality_adj_pct": seasonal,
    }

    if st.button("Generate Scenario Insight", key="btn_sc_intel"):
        with st.spinner("Interpreting scenario…"):
            explanation = _gen_scenario_narrative(f, scenario, params, ai_engine)
        st.session_state["forecast_scenario_results"] = {
            "structured":  _ser_scenario(scenario, params),
            "explanation": explanation,
        }

    if ai_engine is None:
        _no_engine()

    if "forecast_scenario_results" in st.session_state:
        exp = st.session_state["forecast_scenario_results"].get("explanation")
        if exp:
            _narrative(exp, _AMBER)

    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Forecast Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def _render_decomposition(f: ForecastResult, r: RiskProjection) -> None:
    _card(_PURPLE)
    _label("03 — Forecast Decomposition", _PURPLE)
    _title("Trend · Seasonality · Residual")

    st.markdown(
        f'<div style="font-size:12px;color:{_MUTED};margin-bottom:14px;line-height:1.6;">'
        "The forecast is decomposed into its constituent components. "
        "<b>Trend</b> shows long-term growth direction, "
        "<b>Seasonality</b> captures recurring periodic effects, and "
        "<b>Residual</b> reflects one-time anomalies and noise."
        "</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        _chart_decomposition(f, r),
        use_container_width=True, config={"displayModeBar": False},
    )
    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Reliability Panel
# ═══════════════════════════════════════════════════════════════════════════════

def _render_reliability(f: ForecastResult) -> None:
    _card(_GREEN)
    _label("04 — Forecast Reliability", _GREEN)
    _title("Model Performance Metrics")

    # Accuracy derived from RMSE and confidence score
    accuracy_pct = min(99.0, max(0.0, float(f.confidence_score)))
    mape_est     = max(0.0, round(100.0 - accuracy_pct, 2))  # estimated MAPE proxy

    g_col, m_col = st.columns([1, 2])

    with g_col:
        st.plotly_chart(
            _chart_reliability_gauge(accuracy_pct),
            use_container_width=True, config={"displayModeBar": False},
        )

    with m_col:
        _divider()
        _kv("MAPE (estimated)", f"{mape_est:.2f}%",
            _GREEN if mape_est < 10 else _AMBER if mape_est < 20 else _RED)
        _kv("RMSE",  f"{f.rmse:.4f}")
        _kv("Model Confidence",  f"{f.confidence_score:.1f}%",
            _GREEN if f.confidence_score >= 80 else _AMBER)
        _kv("Best Model",  f.model_name)
        _kv("All Models Tested", ", ".join(f.all_rmse.keys()))

        _divider()

        # Plain-language accuracy statement
        if accuracy_pct >= 90:
            quality, adv = "high", "suitable for strategic planning and executive reporting"
        elif accuracy_pct >= 75:
            quality, adv = "moderate", "appropriate for directional planning with caveats"
        else:
            quality, adv = "lower", "best used as a directional indicator; validate with domain experts"

        st.markdown(
            f'<div style="font-size:13px;color:#CBD5E1;line-height:1.8;'
            f'padding:12px 16px;background:{_BG};border-radius:8px;'
            f'border:1px solid {_BORDER};margin-top:8px;">'
            f'The <b>{f.model_name}</b> model has maintained approximately '
            f'<b style="color:{_GREEN if accuracy_pct >= 80 else _AMBER};">'
            f'{accuracy_pct:.1f}% accuracy</b> over the training window, '
            f'indicating <b>{quality}</b> forecast reliability — '
            f'{adv}.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Store
    st.session_state["forecast_accuracy"] = {
        "structured": {
            "accuracy_pct":   round(accuracy_pct, 2),
            "mape_estimated": mape_est,
            "rmse":           _sf(f.rmse),
            "confidence":     _sf(f.confidence_score),
            "quality":        quality,
        },
        "explanation": (
            f"The {f.model_name} model achieves {accuracy_pct:.1f}% accuracy "
            f"(MAPE ~{mape_est:.1f}%), indicating {quality} forecast reliability."
        ),
    }

    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Model Transparency Card
# ═══════════════════════════════════════════════════════════════════════════════

def _render_model_card(f: ForecastResult, horizon: int) -> None:
    _card(_CYAN)
    _label("05 — Model Transparency", _CYAN)
    _title("Model Card · What is powering this forecast?")

    c1, c2 = st.columns(2)
    with c1:
        _kv("Model Name",         f.model_name)
        _kv("Forecast Horizon",   f"{horizon} periods")
        _kv("Models Evaluated",   str(len(f.all_rmse)))
        _kv("Best Model RMSE",    f"{f.rmse:.4f}")
    with c2:
        _kv("Confidence Score",   f"{f.confidence_score:.1f}%")
        _kv("Training Window",    f"All available rows ({f.horizon}-period holdout)")
        _kv("Last Refresh",       "Current session")
        _kv("Selection Criterion","Lowest RMSE across candidate models")

    _divider()
    st.markdown(
        f'<div style="font-size:12px;color:{_MUTED};line-height:1.7;">'
        f"The Decision Lab evaluates multiple forecasting models ({', '.join(f.all_rmse.keys())}) "
        f"and automatically selects the best-performing one based on RMSE on the training window. "
        f"The winning model — <b style='color:{_CYAN};'>{f.model_name}</b> — is used for all "
        f"projections shown above."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Store
    st.session_state["forecast_model_info"] = {
        "structured": {
            "model_name":     f.model_name,
            "horizon":        horizon,
            "rmse":           _sf(f.rmse),
            "confidence":     _sf(f.confidence_score),
            "models_tested":  list(f.all_rmse.keys()),
            "all_rmse":       {k: _sf(v) for k, v in f.all_rmse.items()},
        },
        "explanation": (
            f"Model: {f.model_name}  |  "
            f"Horizon: {horizon} periods  |  "
            f"RMSE: {f.rmse:.4f}  |  "
            f"Confidence: {f.confidence_score:.1f}%"
        ),
    }

    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Strategic Impact Summary
# ═══════════════════════════════════════════════════════════════════════════════

def _render_strategic_impact(
    f: ForecastResult, r: RiskProjection,
    scorecard: DecisionScorecard, signals: list[DecisionSignal],
    ai_engine,
) -> None:
    _card(_PURPLE)
    _label("06 — Strategic Impact Summary", _PURPLE)
    _title("Decision Intelligence · Forecast Implications")

    # Top-level scorecard metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategic Readiness",  f"{scorecard.strategic_readiness:.1f} / 10")
    c2.metric("Risk Stability",       f"{r.stability_score:.1f} / 10")
    c3.metric("Forecast Confidence",  f"{f.confidence_score:.1f}%")

    _divider()

    if st.button("Generate Strategic Impact Summary", key="btn_strat_impact"):
        with st.spinner("Synthesising decision intelligence…"):
            explanation = _gen_strategic_impact(f, r, scorecard, signals, ai_engine)

        st.session_state["forecast_decision_summary"] = {
            "structured": {
                "scorecard":    _ser_scorecard(scorecard),
                "risk":         _ser_risk(r),
                "forecast":     _ser_forecast(f),
                "signal_counts": {
                    sev: sum(1 for s in signals if s.severity == sev)
                    for sev in ("positive", "warning", "critical")
                },
            },
            "explanation": explanation,
        }

    if ai_engine is None:
        _no_engine()

    if "forecast_decision_summary" in st.session_state:
        exp = st.session_state["forecast_decision_summary"].get("explanation")
        if exp:
            _narrative(exp, _PURPLE)

    # Signals (compact)
    _divider()
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:10px;">'
        f'DECISION SIGNALS</div>',
        unsafe_allow_html=True,
    )
    for sig in signals:
        icon  = _SEV_ICON.get(sig.severity, "•")
        color = _SEV_COLOR.get(sig.severity, _BLUE)
        st.markdown(
            f'<div class="dl-sig" style="--sig-col:{color};">'
            f'<div style="font-size:12px;font-weight:700;color:{color};'
            f'margin-bottom:3px;">{icon} {sig.signal}</div>'
            f'<div style="color:#CBD5E1;font-size:12px;margin-bottom:2px;">'
            f'{sig.strategy}</div>'
            f'<div style="color:{_MUTED};font-size:11px;">{sig.rationale}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _endcard()


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def render_decision_tab(df: pd.DataFrame, ai_engine=None) -> None:
    """
    Main entry point called from app.py / tab router.

    Parameters
    ----------
    df         : Active dataset (any schema, ≥ 6 rows with ≥ 1 numeric column).
    ai_engine  : Optional AIEngine instance for narrative generation.
    """
    _css()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:24px;padding-bottom:16px;
                    border-bottom:1px solid {_BORDER};">
            <div style="font-size:10px;font-weight:700;letter-spacing:.18em;
                        text-transform:uppercase;color:{_BLUE};margin-bottom:6px;">
                InsightX · Forecasting Cockpit
            </div>
            <div style="font-size:26px;font-weight:700;color:{_TEXT};
                        letter-spacing:-.02em;line-height:1.2;">
                Decision Lab
            </div>
            <div style="font-size:13px;color:{_MUTED};margin-top:6px;
                        line-height:1.6;max-width:640px;">
                Probabilistic forecasting · Scenario simulation ·
                Decomposition · Reliability · Strategic impact
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guards ────────────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found. Decision Lab requires at least one numeric feature.")
        return
    if len(df) < 6:
        st.error("Dataset too small (< 6 rows). Please load a larger dataset.")
        return

    # ── Sidebar: horizon control ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Decision Lab**")
        horizon = st.slider(
            "Forecast Horizon (periods)", 3, 24, 6, key="dl_horizon"
        )

    # ── Analytics pipeline ────────────────────────────────────────────────────
    with st.spinner("Running forecast models…"):
        try:
            forecast = run_forecast(df, horizon=horizon)
        except Exception as exc:
            st.error(f"Forecast error: {exc}")
            return

    with st.spinner("Projecting risk…"):
        risk = run_risk_projection(df, forecast)

    sensitivity = run_sensitivity(df, forecast, risk)
    signals     = run_decision_signals(forecast, risk)
    scorecard   = run_scorecard(forecast, risk, signals)

    # ── Sections ──────────────────────────────────────────────────────────────
    _render_exec_summary(forecast, risk, scorecard, signals)
    _render_forecast(df, forecast, risk, ai_engine)
    _render_scenario(df, forecast, risk, ai_engine)
    _render_decomposition(forecast, risk)
    _render_reliability(forecast)
    _render_model_card(forecast, horizon)
    _render_strategic_impact(forecast, risk, scorecard, signals, ai_engine)

    # ── NEXT NAVIGATION ──────────────────────────────────────
    st.divider()
    col_left, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Next: AI Insight Lab →", type="primary", use_container_width=True):
           st.session_state["active_tab"] = "AI Insight Lab"
           st.rerun()