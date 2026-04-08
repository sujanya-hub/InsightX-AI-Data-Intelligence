from __future__ import annotations

import io
import logging
import traceback
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from core.reporting.narrative_engine import (
    InvalidNarrativeError, NarrativeEngine, NarrativeEngineError,
)
from core.reporting.pdf_builder import PDFBuilder
from core.reporting.report_engine import ReportEngine

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine

logger = logging.getLogger(__name__)

# ── palette ───────────────────────────────────────────────────────────────────
T = dict(
    bg="#0B0F19", card="#141A2A", border="#1F2937",
    a1="#7C5CFF", a2="#00D4FF",
    success="#00E096", warn="#FF8A3D", crit="#FF4D6D",
    text="#F8FAFC", muted="#94A3B8",
)

_K_PAYLOAD   = "rep_payload"
_K_NARRATIVE = "rep_narrative"
_K_PDF       = "rep_pdf_bytes"


# ── micro-components ──────────────────────────────────────────────────────────

def _card(content_md: str, *, accent: str = "") -> None:
    border = f"border-left:3px solid {accent};" if accent else ""
    st.markdown(
        f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
        f'border-radius:14px;padding:20px 24px;margin-bottom:16px;{border}">'
        f'{content_md}</div>',
        unsafe_allow_html=True,
    )


def _label(txt: str) -> str:
    return (
        f'<div style="font-size:11px;font-weight:700;letter-spacing:.12em;'
        f'text-transform:uppercase;color:{T["a1"]};margin-bottom:10px">{txt}</div>'
    )


def _ai_text(txt: str) -> None:
    st.markdown(
        f'<div style="background:{T["bg"]};border:1px solid {T["border"]};'
        f'border-radius:10px;padding:16px 20px;font-size:13px;line-height:1.8;'
        f'color:{T["text"]};white-space:pre-wrap;margin-top:6px">{txt}</div>',
        unsafe_allow_html=True,
    )


def _divider() -> None:
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {T["border"]};margin:20px 0">',
        unsafe_allow_html=True,
    )


def _sev_colour(sev: str) -> str:
    return {"High": T["crit"], "Medium": T["warn"], "Low": T["success"]}.get(sev, T["muted"])


def _fmt(v: Any, d: int = 2) -> str:
    if v is None: return "—"
    try:    return f"{float(v):,.{d}f}"
    except: return str(v)


def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<span style="font-size:16px">{icon}</span>'
        f'<span style="font-size:15px;font-weight:700;color:{T["text"]};'
        f'margin-left:8px">{title}</span>'
        + (f'<div style="font-size:11px;color:{T["muted"]};margin-top:3px;'
           f'margin-left:26px">{subtitle}</div>' if subtitle else "")
        + '</div>',
        unsafe_allow_html=True,
    )

# ── section renderers ─────────────────────────────────────────────────────────

def _kpi_bar(payload: dict[str, Any]) -> None:
    ds    = payload.get("dataset_summary", {})
    qs    = ds.get("quality_score")
    risks = payload.get("risk_factors", [])
    high  = sum(1 for r in risks if r.get("severity") == "High")
    corrs = payload.get("correlations", [])
    strong = sum(1 for c in corrs if c.get("strength") in ("Strong", "Very Strong"))

    cols = st.columns(5)
    cols[0].metric("Rows",            f"{ds.get('rows', 0):,}")
    cols[1].metric("Columns",         str(ds.get("columns", "—")))
    cols[2].metric("Quality Score",   f"{round(float(qs), 1)}%" if qs else "—")
    cols[3].metric("Risk Signals",    str(len(risks)),
                   delta="High" if len(risks) >= 3 else "Low",
                   delta_color="inverse")
    cols[4].metric("Strong Correlations", str(strong))


def _section_exec_summary(narrative: dict[str, str]) -> None:
    _section_header("Executive Summary",
                    "AI-generated boardroom brief grounded in your data")
    _ai_text(narrative.get("executive_summary", "Not available."))


def _section_key_insights(narrative: dict[str, str], payload: dict[str, Any]) -> None:
    _section_header("Key Insights",
                    "Data-driven observations extracted by the AI analyst")
    raw   = narrative.get("key_insights", "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cols  = st.columns(2)
    for i, line in enumerate(lines):
        for pre in ("- ", "* ", "• "):
            if line.startswith(pre): line = line[len(pre):]
        with cols[i % 2]:
            _card(
                f'<div style="font-size:13px;color:{T["text"]};line-height:1.7">'
                f'<span style="color:{T["a2"]};font-weight:700;margin-right:6px">◆</span>'
                f'{line}</div>',
                accent=T["a1"],
            )
    corr = payload.get("correlations", [])
    if corr:
        _divider()
        st.markdown(_label("Top Variable Correlations"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(corr[:8])[["feature_a", "feature_b",
                                     "correlation", "strength"]],
            use_container_width=True, hide_index=True,
        )


def _section_business_impact(narrative: dict[str, str]) -> None:
    _section_header("Business Impact",
                    "Operational and financial consequences of the data signals")
    _ai_text(narrative.get("business_impact", "Not available."))


def _section_market_opportunities(narrative: dict[str, str]) -> None:
    _section_header("Market Opportunities",
                    "Growth levers and untapped potential identified from the data")
    text = narrative.get(
        "market_opportunities",
        narrative.get("strategic_recommendations", "Not available."),
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Detect list vs prose
    is_list = any(
        ln.startswith(("- ", "* ", "• ")) or (len(ln) > 2 and ln[0].isdigit())
        for ln in lines
    )
    if is_list:
        for i, line in enumerate(lines):
            for pre in ("- ", "* ", "• "):
                if line.startswith(pre): line = line[len(pre):]
            _card(
                f'<div style="display:flex;align-items:flex-start;gap:14px">'
                f'<div style="font-size:20px;font-weight:700;color:{T["a2"]};'
                f'min-width:26px;line-height:1.3">{i + 1}</div>'
                f'<div style="font-size:13px;color:{T["text"]};line-height:1.7">'
                f'{line}</div></div>',
                accent=T["a2"],
            )
    else:
        _ai_text(text)


def _section_operational_recommendations(narrative: dict[str, str]) -> None:
    _section_header("Operational Recommendations",
                    "Prioritised actions for immediate implementation")
    text  = narrative.get(
        "operational_recommendations",
        narrative.get("strategic_recommendations", "Not available."),
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines, 1):
        for pre in ("- ", "* ", "• ", f"{i}. ", f"{i}) "):
            if line.startswith(pre): line = line[len(pre):]
        _card(
            f'<div style="display:flex;align-items:flex-start;gap:14px">'
            f'<div style="font-size:22px;font-weight:700;color:{T["a1"]};'
            f'min-width:28px;line-height:1.3">{i}</div>'
            f'<div style="font-size:13px;color:{T["text"]};line-height:1.7">'
            f'{line}</div></div>',
            accent=T["a1"],
        )


def _section_data_limitations(narrative: dict[str, str]) -> None:
    _section_header("Data Limitations",
                    "Caveats and constraints to consider before acting on findings")
    text = narrative.get(
        "data_limitations",
        narrative.get("risks_and_limitations", "Not available."),
    )
    _ai_text(text)


def _section_forecast(payload: dict[str, Any]) -> None:
    _section_header("Forecast Outlook",
                    "Forward-looking projections from the Decision Intelligence pipeline")
    fc = payload.get("forecast_summary", {})
    if not fc.get("available"):
        _card(
            f'<span style="color:{T["muted"]};font-size:13px">'
            "Forecasting data not available. Complete Tab 4 first.</span>"
        )
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Target",     str(fc.get("target_column", "—")))
    c2.metric("Trend",      str(fc.get("trend_direction", "—")))
    c3.metric("Confidence", f"{float(fc.get('confidence_level', 0.95)) * 100:.0f}%")


def _section_risks(payload: dict[str, Any], narrative: dict[str, str]) -> None:
    _section_header("Risk Analysis",
                    "Severity-ranked signals requiring executive attention")
    risk_text = narrative.get("risks_and_limitations", "")
    if risk_text:
        _ai_text(risk_text)

    risks = payload.get("risk_factors", [])
    if not risks:
        _card(
            f'<span style="color:{T["success"]}">✓ No significant risk signals detected.</span>'
        )
        return

    _divider()
    # Summary counts
    high = sum(1 for r in risks if r.get("severity") == "High")
    med  = sum(1 for r in risks if r.get("severity") == "Medium")
    low  = sum(1 for r in risks if r.get("severity") == "Low")
    c1, c2, c3 = st.columns(3)
    c1.metric("High",   high,  delta="Critical" if high else None, delta_color="inverse")
    c2.metric("Medium", med)
    c3.metric("Low",    low)

    st.markdown("<br>", unsafe_allow_html=True)
    for r in risks[:10]:
        sev    = r.get("severity", "Low")
        colour = _sev_colour(sev)
        _card(
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<div style="background:{colour}22;border:1px solid {colour};'
            f'border-radius:6px;padding:2px 10px;font-size:11px;'
            f'font-weight:700;color:{colour};white-space:nowrap">{sev}</div>'
            f'<div style="flex:1">'
            f'<div style="font-size:11px;color:{T["muted"]};margin-bottom:2px">'
            f'{r.get("category", "—")} · {r.get("source", "—")}</div>'
            f'<div style="font-size:13px;color:{T["text"]}">'
            f'{r.get("description", "—")}</div>'
            f'</div></div>',
            accent=colour,
        )


def _section_charts_preview(payload: dict[str, Any],
                             df: pd.DataFrame | None) -> None:
    """Inline Matplotlib chart previews — same charts embedded in the PDF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.reporting.pdf_builder import (
            _chart_correlation_heatmap,
            _chart_distribution,
            _chart_segment_bar,
            _chart_time_trend,
        )
    except ImportError:
        st.info("Matplotlib not available — chart previews skipped.")
        return

    _section_header("Visual Analytics",
                    "Chart previews embedded in the PDF report")

    col_a, col_b = st.columns(2)

    # Distribution
    with col_a:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{T["muted"]};margin-bottom:6px">'
            f'Distribution</div>',
            unsafe_allow_html=True,
        )
        fig = _chart_distribution(df) if df is not None else None
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Insufficient numeric data.")

    # Segment bar
    with col_b:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{T["muted"]};margin-bottom:6px">'
            f'Column Performance</div>',
            unsafe_allow_html=True,
        )
        fig = _chart_segment_bar(payload)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Insufficient data.")

    col_c, col_d = st.columns(2)

    # Correlation heatmap
    with col_c:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{T["muted"]};margin-bottom:6px;'
            f'margin-top:12px">Correlation Heatmap</div>',
            unsafe_allow_html=True,
        )
        fig = _chart_correlation_heatmap(payload)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Need ≥ 2 numeric columns.")

    # Trend
    with col_d:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{T["muted"]};margin-bottom:6px;'
            f'margin-top:12px">Metric Trends</div>',
            unsafe_allow_html=True,
        )
        fig = _chart_time_trend(payload)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Insufficient KPI trend data.")


# ── pre-generate state ────────────────────────────────────────────────────────

def _section_pre_generate(cleaned_df: pd.DataFrame) -> None:
    _divider()
    _section_header("◈", "Dataset Ready for Reporting")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",             f"{len(cleaned_df):,}")
    c2.metric("Columns",          str(cleaned_df.shape[1]))
    c3.metric("Numeric Features", str(len(
        cleaned_df.select_dtypes(include="number").columns)))

    with st.expander("Preview (first 5 rows)"):
        st.dataframe(cleaned_df.head(5), use_container_width=True)

    st.markdown(_label("Pipeline Outputs Detected"), unsafe_allow_html=True)
    checks = {
        "Tab 1 – Data Readiness":        "tab1_outputs",
        "Tab 2 – Executive Dashboard":   "tab2_outputs",
        "Tab 3 – Decision Intelligence": "tab3_outputs",
        "Tab 4 – Forecasting":           "tab4_outputs",
    }
    for lbl, key in checks.items():
        ok     = bool(st.session_state.get(key))
        colour = T["success"] if ok else T["muted"]
        icon   = "✓" if ok else "○"
        st.markdown(
            f'<div style="font-size:12px;color:{colour};line-height:2">'
            f'{icon} {lbl}</div>',
            unsafe_allow_html=True,
        )

    _divider()
    st.markdown(
        f'<div style="background:{T["card"]};border:1px solid {T["a1"]}44;'
        f'border-radius:12px;padding:20px 24px;">'
        f'<div style="font-size:13px;color:{T["muted"]};line-height:1.8">'
        f'The report will include <b style="color:{T["text"]}">8 pages</b>:'
        f'<br>Cover · Executive Summary · Key Insights · '
        f'Distribution &amp; Segment Charts · Correlation &amp; Trend Charts · '
        f'Business Impact &amp; Opportunities · Operational Recommendations &amp; '
        f'Data Limitations · Risk Analysis &amp; Technical Appendix.'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ── generation pipeline ───────────────────────────────────────────────────────

def _collect_tab_outputs() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("tab1_outputs", "tab2_outputs", "tab3_outputs", "tab4_outputs"):
        val = st.session_state.get(key) or {}
        if isinstance(val, dict):
            merged.update(val)
        merged[key] = val
    return merged


def _run_pipeline(cleaned_df: pd.DataFrame, ai_engine: "AIEngine") -> None:
    prog = st.progress(0, text="Analysing dataset …")

    tab_outputs = _collect_tab_outputs()

    prog.progress(15, text="Building analytics payload …")
    engine  = ReportEngine(df=cleaned_df, tab_outputs=tab_outputs)
    payload = engine.build_report_payload()
    st.session_state[_K_PAYLOAD] = payload

    prog.progress(40, text="Generating AI narrative (this may take 20–30 s) …")
    narrative = NarrativeEngine(
        payload=payload, ai_engine=ai_engine
    ).generate_narrative()
    st.session_state[_K_NARRATIVE] = narrative

    prog.progress(75, text="Rendering charts and building PDF …")
    org   = st.session_state.get("report_organisation", "InsightX Analytics")
    title = st.session_state.get("report_title", "Executive Intelligence Report")
    pdf   = PDFBuilder(
        payload=payload,
        narrative=narrative,
        title=title,
        organisation=org,
        df=cleaned_df,           
    ).build()
    st.session_state[_K_PDF] = pdf

    prog.progress(100, text="Done.")
    prog.empty()


# ── main entry point ──────────────────────────────────────────────────────────

def render(ai_engine: "AIEngine | None" = None) -> None:
    """
    Entry point for Tab 5.
    Called from app.py as: render_report(ai_engine=ai_engine)
    """
    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:24px">'
        f'<div style="font-size:23px;font-weight:700;color:{T["text"]};'
        f'letter-spacing:.01em">⬟ Executive Intelligence Report</div>'
        f'<div style="font-size:12px;color:{T["muted"]};margin-top:4px">'
        f'8-page boardroom PDF · AI narrative · embedded charts · '
        f'auto-generated from the analytics pipeline</div></div>',
        unsafe_allow_html=True,
    )

    # ── guards ────────────────────────────────────────────────────────────────
    cleaned_df: pd.DataFrame | None = st.session_state.get("cleaned_df")
    if cleaned_df is None or (hasattr(cleaned_df, "empty") and cleaned_df.empty):
        st.error("No cleaned dataset. Complete **Tab 1 – Data Readiness** first.")
        return
    if ai_engine is None:
        st.error("AI engine unavailable. Ensure **GROQ_API_KEY** is set in secrets.")
        return

    # ── report settings ───────────────────────────────────────────────────────
    with st.expander("Report settings", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            org = st.text_input(
                "Organisation",
                value=st.session_state.get("report_organisation", "InsightX Analytics"),
            )
            st.session_state["report_organisation"] = org
        with cb:
            ttl = st.text_input(
                "Report Title",
                value=st.session_state.get("report_title", "Executive Intelligence Report"),
            )
            st.session_state["report_title"] = ttl

    _divider()

    # ── action row ────────────────────────────────────────────────────────────
    col_gen, col_dl, col_rst = st.columns([3, 3, 1])

    with col_gen:
        gen_clicked = st.button(
            "Generate Executive Report",
            type="primary", use_container_width=True,
        )

    pdf_bytes: bytes | None = st.session_state.get(_K_PDF)
    with col_dl:
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes or b"",
            file_name="InsightX_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=pdf_bytes is None,
        )

    with col_rst:
        if st.button("🗑️", use_container_width=True, help="Clear report cache"):
            for k in (_K_PAYLOAD, _K_NARRATIVE, _K_PDF):
                st.session_state.pop(k, None)
            st.rerun()

    # ── run pipeline ──────────────────────────────────────────────────────────
    if gen_clicked:
        for k in (_K_PAYLOAD, _K_NARRATIVE, _K_PDF):
            st.session_state.pop(k, None)
        try:
            _run_pipeline(cleaned_df, ai_engine)
            st.success("Report ready — download the PDF above.")
            st.rerun()
        except (NarrativeEngineError, InvalidNarrativeError) as exc:
            st.error(f"AI narrative error: {exc}")
        except ValueError as exc:
            st.error(f"Data error: {exc}")
        except Exception as exc:                                    # noqa: BLE001
            st.error(f"Unexpected error: {exc}")
            with st.expander("Debug traceback"):
                st.code(traceback.format_exc())
            logger.error("tab_report render error", exc_info=True)

    # ── report display ────────────────────────────────────────────────────────
    payload:   dict[str, Any] = st.session_state.get(_K_PAYLOAD, {})
    narrative: dict[str, str] = st.session_state.get(_K_NARRATIVE, {})

    if not payload or not narrative:
        _section_pre_generate(cleaned_df)
        return

    _divider()
    _kpi_bar(payload)
    _divider()

    tabs = st.tabs([
        "Summary",
        "Insights",
        "Business Impact",
        "Opportunities",
        "Recommendations",
        "Data Limitations",
        "Forecast",
        "Risks",
        "Charts",
    ])

    with tabs[0]: _section_exec_summary(narrative)
    with tabs[1]: _section_key_insights(narrative, payload)
    with tabs[2]: _section_business_impact(narrative)
    with tabs[3]: _section_market_opportunities(narrative)
    with tabs[4]: _section_operational_recommendations(narrative)
    with tabs[5]: _section_data_limitations(narrative)
    with tabs[6]: _section_forecast(payload)
    with tabs[7]: _section_risks(payload, narrative)
    with tabs[8]: _section_charts_preview(payload, cleaned_df)

    # ── NEXT NAVIGATION ──────────────────────────────────────
    st.divider()
    col_left, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Start New Analysis", type="primary", use_container_width=True):
           st.session_state["active_tab"] = "Data Readiness"
           st.rerun()    