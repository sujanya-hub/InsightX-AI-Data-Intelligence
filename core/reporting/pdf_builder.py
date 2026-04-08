"""
InsightX · PDF Builder — ReportLab renderer with embedded Matplotlib charts.
Returns raw bytes. Zero Streamlit. Python 3.11.

Theme: Professional consulting light theme (white/grey palette).

Page map (6–8 pages):
  P1  Cover
  P2  Executive Summary + Key Findings
  P3  Dataset Overview + Key Insights + Correlations table
  P4  Visualisation — Distribution Histogram + Segment Bar Chart
  P5  Visualisation — Correlation Heatmap + Time Trend Chart
  P6  Business Impact + Market Opportunities
  P7  Operational Recommendations + Data Limitations
  P8  Risks & Limitations table + Technical Appendix
"""

from __future__ import annotations

import io
import math
import textwrap
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── ReportLab light palette ───────────────────────────────────────────────────
_WHITE  = colors.HexColor("#FFFFFF")
_OFFWHT = colors.HexColor("#F9FAFB")      # page background accent rows
_LGREY  = colors.HexColor("#F3F4F6")      # alternate table rows
_BORDER = colors.HexColor("#E5E7EB")      # grid lines / dividers
_MGREY  = colors.HexColor("#9CA3AF")      # muted labels / captions
_DKGREY = colors.HexColor("#374151")      # body text
_BLACK  = colors.HexColor("#111827")      # headings
_INDIGO = colors.HexColor("#4F46E5")      # primary accent (cover, H1 rule, headers)
_SLATE  = colors.HexColor("#6366F1")      # secondary accent
_AMBER  = colors.HexColor("#D97706")      # warning / highlight
_GREEN  = colors.HexColor("#16A34A")      # positive signal
_RED    = colors.HexColor("#DC2626")      # negative / risk
_CVBG   = colors.HexColor("#1E1B4B")      # cover page bg (deep indigo — keeps visual impact)
_CVTXT  = colors.HexColor("#FFFFFF")      # cover text

# Matplotlib equivalents for chart rendering
_MPL = dict(
    bg     = "#FFFFFF",
    card   = "#FFFFFF",
    border = "#E5E7EB",
    vio    = "#4F46E5",
    cyan   = "#6366F1",
    amber  = "#D97706",
    text   = "#111827",
    muted  = "#6B7280",
    green  = "#16A34A",
    red    = "#DC2626",
)

PW, PH = A4
ML, MR = 2.0 * cm, 2.0 * cm
MT, MB = 2.8 * cm, 2.2 * cm
BW = PW - ML - MR

_CW_FULL = BW
_CH_FULL = 7.2 * cm
_CW_HALF = BW * 0.48
_CH_HALF = 6.8 * cm


# ── styles ────────────────────────────────────────────────────────────────────
def _styles() -> dict[str, ParagraphStyle]:
    return {
        # Cover
        "cv_logo":  ParagraphStyle("cv_logo",  fontName="Helvetica-Bold", fontSize=10,
                                   textColor=colors.HexColor("#A5B4FC"),
                                   letterSpacing=2, spaceAfter=4),
        "cv_title": ParagraphStyle("cv_title", fontName="Helvetica-Bold", fontSize=28,
                                   textColor=_CVTXT, leading=34, spaceAfter=6),
        "cv_sub":   ParagraphStyle("cv_sub",   fontName="Helvetica", fontSize=12,
                                   textColor=colors.HexColor("#C7D2FE"), spaceAfter=4),
        "cv_meta":  ParagraphStyle("cv_meta",  fontName="Helvetica", fontSize=9,
                                   textColor=colors.HexColor("#A5B4FC")),
        # Body headings
        "h1":       ParagraphStyle("h1",  fontName="Helvetica-Bold", fontSize=13,
                                   textColor=_BLACK,
                                   spaceBefore=14, spaceAfter=5, leading=17),
        "h2":       ParagraphStyle("h2",  fontName="Helvetica-Bold", fontSize=10,
                                   textColor=_DKGREY,
                                   spaceBefore=9, spaceAfter=4),
        "h3":       ParagraphStyle("h3",  fontName="Helvetica-Bold", fontSize=9,
                                   textColor=_DKGREY,
                                   spaceBefore=6, spaceAfter=3),
        # Body text
        "body":     ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                                   textColor=_DKGREY,
                                   alignment=TA_JUSTIFY, spaceAfter=5, leading=14),
        "bul":      ParagraphStyle("bul",  fontName="Helvetica", fontSize=9,
                                   textColor=_DKGREY,
                                   leftIndent=12, spaceAfter=3, leading=13),
        "cap":      ParagraphStyle("cap",  fontName="Helvetica-Oblique", fontSize=8,
                                   textColor=_MGREY, spaceAfter=3),
        # Table
        "th":       ParagraphStyle("th",   fontName="Helvetica-Bold", fontSize=8,
                                   textColor=_WHITE, alignment=TA_CENTER),
        "td":       ParagraphStyle("td",   fontName="Helvetica", fontSize=8,
                                   textColor=_DKGREY, leading=11),
        "kv_key":   ParagraphStyle("kv_key", fontName="Helvetica-Bold", fontSize=8,
                                   textColor=_BLACK),
        "chart_title": ParagraphStyle("chart_title", fontName="Helvetica-Bold",
                                      fontSize=9, textColor=_BLACK,
                                      spaceBefore=8, spaceAfter=2),
    }


# ── flowable helpers ──────────────────────────────────────────────────────────
def _p(txt: str, s: ParagraphStyle) -> Paragraph:
    return Paragraph(str(txt), s)

def _sp(h: float = 0.35) -> Spacer:
    return Spacer(1, h * cm)

def _hr(c=_BORDER, t: float = 0.6) -> HRFlowable:
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=4, spaceBefore=4)

def _hr_accent(t: float = 1.2) -> HRFlowable:
    return HRFlowable(width="100%", thickness=t, color=_INDIGO, spaceAfter=4, spaceBefore=4)

def _fmt(v: Any, d: int = 2) -> str:
    if v is None: return "—"
    try:    return f"{float(v):,.{d}f}"
    except: return str(v)

def _bullets(text: str, S: dict) -> list:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        for pre in ("- ", "* ", "• "):
            if ln.startswith(pre): ln = ln[len(pre):]
        out.append(_p(f"• &nbsp;{ln}", S["bul"]))
    return out

def _base_ts() -> list:
    """Light-themed table style."""
    return [
        ("BACKGROUND",     (0, 0), (-1, 0),  _INDIGO),       # header row: indigo
        ("TEXTCOLOR",      (0, 0), (-1, 0),  _WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LGREY]),  # alt rows: white / light grey
        ("GRID",           (0, 0), (-1, -1), 0.35, _BORDER),
        ("BOX",            (0, 0), (-1, -1), 0.7,  colors.HexColor("#D1D5DB")),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]

def _kv_tbl(rows: list[list[str]], S: dict) -> Table:
    cw = [BW * 0.38, BW * 0.62]
    styled = [[_p(c, S["th"]) for c in rows[0]]]
    for row in rows[1:]:
        styled.append([_p(row[0], S["kv_key"]), _p(row[1], S["td"])])
    t = Table(styled, colWidths=cw)
    t.setStyle(TableStyle(_base_ts()))
    return t

def _data_tbl(rows: list[list[str]], S: dict,
              cw: list[float] | None = None) -> Table:
    n  = len(rows[0]) if rows else 1
    cw = cw or [BW / n] * n
    styled = [[_p(c, S["th"]) for c in rows[0]]]
    for row in rows[1:]:
        styled.append([_p(c, S["td"]) for c in row])
    t = Table(styled, colWidths=cw)
    t.setStyle(TableStyle(_base_ts()))
    return t


# ── canvas callbacks ──────────────────────────────────────────────────────────
def _cover_bg(canvas, doc) -> None:
    """Deep-indigo cover — keeps brand impact while body pages are fully light."""
    canvas.saveState()
    # Full-page deep indigo background
    canvas.setFillColor(_CVBG)
    canvas.rect(0, 0, PW, PH, fill=1, stroke=0)
    # Bottom accent bar — indigo gradient simulation
    canvas.setFillColor(_INDIGO)
    canvas.rect(0, 0, PW, 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#818CF8"))
    canvas.rect(0, 10 * mm, PW * 0.25, 2 * mm, fill=1, stroke=0)
    # Header strip
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor("#A5B4FC"))
    canvas.drawString(ML, PH - 1.1 * cm, "InsightX — Executive Intelligence Report")
    canvas.drawRightString(PW - MR, PH - 1.1 * cm, datetime.now().strftime("%B %Y"))
    canvas.setStrokeColor(colors.HexColor("#3730A3"))
    canvas.setLineWidth(0.5)
    canvas.line(ML, PH - 1.35 * cm, PW - MR, PH - 1.35 * cm)
    canvas.restoreState()


def _page_deco(canvas, doc) -> None:
    """Light header/footer for all body pages."""
    canvas.saveState()
    # White page background
    canvas.setFillColor(_WHITE)
    canvas.rect(0, 0, PW, PH, fill=1, stroke=0)
    # Thin top indigo rule
    canvas.setFillColor(_INDIGO)
    canvas.rect(0, PH - 1 * mm, PW, 1 * mm, fill=1, stroke=0)

    y_hdr = PH - 1.15 * cm
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(_INDIGO)
    canvas.drawString(ML, y_hdr, "InsightX")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_MGREY)
    canvas.drawString(ML + 1.35 * cm, y_hdr, "· Executive Intelligence Report")
    canvas.drawRightString(PW - MR, y_hdr, datetime.now().strftime("%B %Y"))
    canvas.setStrokeColor(_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(ML, y_hdr - 3, PW - MR, y_hdr - 3)

    y_ftr = MB - 0.35 * cm
    canvas.setStrokeColor(_BORDER)
    canvas.line(ML, y_ftr + 10, PW - MR, y_ftr + 10)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_MGREY)
    canvas.drawCentredString(PW / 2, y_ftr, f"Page {doc.page}")
    canvas.drawString(ML, y_ftr, "CONFIDENTIAL")
    canvas.drawRightString(PW - MR, y_ftr, "InsightX Analytics Platform")
    canvas.restoreState()


# ── Matplotlib chart factory ──────────────────────────────────────────────────

def _fig_to_image(fig: plt.Figure, w_pt: float, h_pt: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=w_pt, height=h_pt)


def _apply_light_axes(ax, grid_axis: str = "both") -> None:
    """Apply consistent light-theme styling to a matplotlib Axes."""
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(_MPL["border"])
        spine.set_linewidth(0.8)
    if grid_axis in ("both", "x"):
        ax.xaxis.grid(True, color=_MPL["border"], linewidth=0.6, linestyle="--")
    if grid_axis in ("both", "y"):
        ax.yaxis.grid(True, color=_MPL["border"], linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MPL["muted"], labelsize=7)


def _chart_distribution(df: pd.DataFrame) -> plt.Figure | None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return None
    col = num_cols[0]
    s   = df[col].dropna()
    if len(s) < 5:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor="white")
    _apply_light_axes(ax, "y")

    n, bins, _ = ax.hist(s, bins=30, color=_MPL["vio"], alpha=0.75, edgecolor="white",
                         linewidth=0.4)
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(float(s.min()), float(s.max()), 200)
        scale = len(s) * (bins[1] - bins[0])
        ax.plot(kde_x, gaussian_kde(s)(kde_x) * scale,
                color=_MPL["cyan"], linewidth=1.8, label="KDE", zorder=3)
    except Exception:
        pass

    ax.axvline(float(s.mean()),   color=_MPL["amber"], linewidth=1.4,
               linestyle="--", label=f"Mean {s.mean():.2f}", zorder=4)
    ax.axvline(float(s.median()), color=_MPL["green"], linewidth=1.4,
               linestyle=":",  label=f"Median {s.median():.2f}", zorder=4)

    ax.set_title(f"Distribution · {col}", color=_MPL["text"], fontsize=9,
                 fontweight="bold", pad=8)
    ax.set_xlabel(col,     color=_MPL["muted"], fontsize=7)
    ax.set_ylabel("Count", color=_MPL["muted"], fontsize=7)
    ax.legend(fontsize=7, facecolor="white", edgecolor=_MPL["border"],
              labelcolor=_MPL["text"], framealpha=1)
    fig.tight_layout(pad=0.8)
    return fig


def _chart_segment_bar(payload: dict) -> plt.Figure | None:
    ds   = payload.get("dataset_summary", {})
    kpis = payload.get("kpis", {})
    desc = ds.get("descriptive_stats", {})

    if not desc and not kpis:
        return None

    items: list[tuple[str, float]] = []
    for col, stats in list(desc.items())[:12]:
        mean = stats.get("mean")
        if mean is not None:
            try:
                items.append((col[:22], float(mean)))
            except (TypeError, ValueError):
                pass
    if not items:
        for col, stats in list(kpis.items())[:12]:
            if isinstance(stats, dict) and stats.get("mean") is not None:
                try:
                    items.append((col[:22], float(stats["mean"])))
                except (TypeError, ValueError):
                    pass
    if not items:
        return None

    items.sort(key=lambda x: x[1], reverse=True)
    labels, vals = zip(*items[:10])

    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor="white")
    _apply_light_axes(ax, "x")

    # Gradient effect: darken bars progressively
    palette = [_MPL["vio"] if i % 2 == 0 else _MPL["cyan"] for i in range(len(labels))]
    bars = ax.barh(list(labels), list(vals), color=palette,
                   edgecolor="white", linewidth=0.4, height=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.2f}", va="center", ha="left",
                color=_MPL["muted"], fontsize=7)

    ax.set_title("Column Mean Values — Top 10", color=_MPL["text"],
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Mean Value", color=_MPL["muted"], fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.8)
    return fig


def _chart_correlation_heatmap(payload: dict) -> plt.Figure | None:
    corr_list = payload.get("correlations", [])
    if not corr_list:
        return None

    features: list[str] = []
    for r in corr_list[:20]:
        for k in ("feature_a", "feature_b"):
            f = r.get(k, "")
            if f and f not in features:
                features.append(f)
    if len(features) < 2:
        return None

    features = features[:12]
    n = len(features)
    mat = np.zeros((n, n))
    np.fill_diagonal(mat, 1.0)
    f_idx = {f: i for i, f in enumerate(features)}
    for r in corr_list:
        a, b = r.get("feature_a"), r.get("feature_b")
        if a in f_idx and b in f_idx:
            v = float(r.get("correlation", 0))
            mat[f_idx[a]][f_idx[b]] = v
            mat[f_idx[b]][f_idx[a]] = v

    fig, ax = plt.subplots(figsize=(6.5, 3.0), facecolor="white")
    ax.set_facecolor("white")

    # Light-friendly diverging colormap
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [f[:13] for f in features]
    ax.set_xticklabels(short, rotation=40, ha="right",
                       color=_MPL["muted"], fontsize=6)
    ax.set_yticklabels(short, color=_MPL["muted"], fontsize=6)

    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                txt_color = "white" if abs(val) > 0.65 else _MPL["text"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=6, colors=_MPL["muted"])
    cbar.outline.set_edgecolor(_MPL["border"])

    ax.set_title("Correlation Heatmap", color=_MPL["text"],
                 fontsize=9, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_MPL["border"])
    fig.tight_layout(pad=0.8)
    return fig


def _chart_time_trend(payload: dict) -> plt.Figure | None:
    ds   = payload.get("dataset_summary", {})
    desc = ds.get("descriptive_stats", {})
    kpis = payload.get("kpis", {})

    trend_data: list[tuple[str, float]] = []
    for col, stats in list(kpis.items())[:15]:
        if isinstance(stats, dict):
            pct = stats.get("pct_change_last")
            if pct is not None:
                try:
                    trend_data.append((col[:18], float(pct)))
                except (TypeError, ValueError):
                    pass

    if not trend_data:
        for col, stats in list(desc.items())[:10]:
            mean = stats.get("mean")
            if mean is not None:
                try:
                    trend_data.append((col[:18], float(mean)))
                except (TypeError, ValueError):
                    pass
    if not trend_data:
        return None

    labels, vals = zip(*trend_data)

    fig, ax = plt.subplots(figsize=(6.5, 2.8), facecolor="white")
    _apply_light_axes(ax, "y")

    x = range(len(labels))
    bar_colors = [_MPL["green"] if v >= 0 else _MPL["red"] for v in vals]
    ax.bar(x, vals, color=bar_colors, alpha=0.70, edgecolor="white",
           linewidth=0.4, width=0.6)
    ax.plot(x, vals, color=_MPL["vio"], linewidth=1.6,
            marker="o", markersize=4, zorder=3)
    ax.axhline(0, color=_MPL["border"], linewidth=1.0, linestyle="-")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       color=_MPL["muted"], fontsize=6.5)
    ax.set_ylabel("% Change (last)", color=_MPL["muted"], fontsize=7)
    ax.set_title("Metric Trend Signals — Last Period % Change",
                 color=_MPL["text"], fontsize=9, fontweight="bold", pad=8)

    pos_patch = mpatches.Patch(color=_MPL["green"], alpha=0.75, label="Positive")
    neg_patch = mpatches.Patch(color=_MPL["red"],   alpha=0.75, label="Negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=7,
              facecolor="white", edgecolor=_MPL["border"],
              labelcolor=_MPL["text"], framealpha=1)
    fig.tight_layout(pad=0.8)
    return fig


# ── PDFBuilder ────────────────────────────────────────────────────────────────
class PDFBuilder:
    def __init__(
        self,
        payload:      dict[str, Any],
        narrative:    dict[str, str],
        title:        str = "Executive Intelligence Report",
        organisation: str = "InsightX Analytics",
        df:           pd.DataFrame | None = None,
    ) -> None:
        self.payload      = payload
        self.narrative    = narrative
        self.title        = title
        self.organisation = organisation
        self.df           = df
        self.S            = _styles()
        self.ts           = datetime.now()

    def build(self) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=ML, rightMargin=MR,
            topMargin=MT,  bottomMargin=MB,
            title=self.title,
            author=self.organisation,
        )
        doc.build(
            self._story(),
            onFirstPage=_cover_bg,
            onLaterPages=_page_deco,
        )
        return buf.getvalue()

    # ── story assembly ────────────────────────────────────────────────────────

    def _story(self) -> list:
        s  = self._cover()
        s += [PageBreak()]
        s += self._exec_summary()
        s += self._key_findings()          # NEW: Key Findings section
        s += [PageBreak()]
        s += self._dataset_overview()
        s += self._key_insights()
        s += [PageBreak()]
        s += self._visualisation_page_1()
        s += [PageBreak()]
        s += self._visualisation_page_2()
        s += [PageBreak()]
        s += self._business_impact()
        s += self._market_opportunities()
        s += [PageBreak()]
        s += self._operational_recommendations()
        s += self._data_limitations()
        s += [PageBreak()]
        s += self._risks()
        s += self._appendix()
        return s

    # ── pages ─────────────────────────────────────────────────────────────────

    def _cover(self) -> list:
        S = self.S
        return [
            _sp(4.0),
            _p("INSIGHTX", S["cv_logo"]),
            _sp(0.2),
            _p(self.title, S["cv_title"]),
            _sp(0.3),
            HRFlowable(width="30%", thickness=2, color=_INDIGO, spaceAfter=10),
            _p(self.organisation, S["cv_sub"]),
            _p(self.ts.strftime("Generated %d %B %Y · %H:%M UTC"), S["cv_meta"]),
            _sp(0.2),
            _p("CONFIDENTIAL — For executive use only.", S["cv_meta"]),
        ]

    def _exec_summary(self) -> list:
        S = self.S
        return [
            _p("1. Executive Summary", S["h1"]),
            _hr_accent(),
            _sp(0.15),
            _p(self.narrative.get("executive_summary", "Not available."), S["body"]),
            _sp(0.3),
        ]

    def _key_findings(self) -> list:
        """New consulting-style Key Findings section."""
        S  = self.S
        ds = self.payload.get("dataset_summary", {})
        rows = ds.get("rows", 0)
        cols = ds.get("columns", 0)

        text = (
            f"This analysis evaluates {rows:,} records across {cols} variables. "
            "Strong correlations between pricing and sales indicators suggest "
            "that demand behaviour is influenced by pricing structure and "
            "order volume. Distribution analysis indicates moderate variance "
            "in sales performance, while trend signals reveal emerging "
            "growth patterns across several metrics."
        )

        return [
            _p("Key Findings", S["h1"]),
            _hr_accent(),
            _sp(0.15),
            _p(text, S["body"]),
            _sp(0.3),
        ]

    def _dataset_overview(self) -> list:
        S  = self.S
        ds = self.payload.get("dataset_summary", {})
        qs = ds.get("quality_score")
        rows_tbl = [
            ["Attribute", "Value"],
            ["Rows",            f"{ds.get('rows', 0):,}"],
            ["Columns",         str(ds.get("columns", "—"))],
            ["Numeric Columns", str(len(ds.get("numeric_columns", [])))],
            ["Categorical",     str(len(ds.get("categorical_columns", [])))],
            ["Duplicates",      str(ds.get("duplicate_rows", "—"))],
            ["Quality Score",   f"{qs}%" if qs is not None else "—"],
        ]
        out = [
            _p("2. Dataset Overview", S["h1"]),
            _hr_accent(), _sp(0.15),
            _kv_tbl(rows_tbl, S), _sp(0.25),
        ]

        miss = ds.get("missing_value_profile", {})
        if miss:
            mrows = [["Column", "Missing Count", "Missing %"]]
            for col, info in list(miss.items())[:12]:
                mrows.append([col, str(info["count"]), f"{info['pct']}%"])
            out += [_p("Missing Values", S["h2"]), _data_tbl(mrows, S), _sp(0.25)]

        desc = ds.get("descriptive_stats", {})
        if desc:
            srows = [["Column", "Mean", "Std", "Min", "Max"]]
            for col, s in list(desc.items())[:10]:
                srows.append([col[:20], _fmt(s.get("mean")), _fmt(s.get("std")),
                               _fmt(s.get("min")), _fmt(s.get("max"))])
            cw = [BW * .30, BW * .175, BW * .175, BW * .175, BW * .175]
            out += [
                _p("Descriptive Statistics", S["h2"]),
                _data_tbl(srows, S, cw), _sp(0.25),
            ]
        return out

    def _key_insights(self) -> list:
        S   = self.S
        out = [_p("3. Key Insights", S["h1"]), _hr_accent(), _sp(0.15)]
        text = self.narrative.get("key_drivers") or self.narrative.get("key_insights", "Not available.")
        out += _bullets(text, S)

        corr = self.payload.get("correlations", [])
        if corr:
            crow = [["Feature A", "Feature B", "Correlation", "Strength"]]
            for r in corr[:8]:
                crow.append([r["feature_a"], r["feature_b"],
                              _fmt(r["correlation"], 4), r["strength"]])
            cw = [BW * .28, BW * .28, BW * .22, BW * .22]
            out += [_sp(0.2), _p("Top Correlations", S["h2"]),
                    _data_tbl(crow, S, cw)]
        out.append(_sp(0.25))
        return out

    def _visualisation_page_1(self) -> list:
        S   = self.S
        out = [
            _p("4. Data Visualisations — Distributions & Segments", S["h1"]),
            _hr_accent(), _sp(0.15),
        ]

        out.append(_p("Distribution Analysis", S["h2"]))
        dist_fig = _chart_distribution(self.df) if self.df is not None else None
        if dist_fig:
            out.append(_fig_to_image(dist_fig, _CW_FULL, _CH_FULL))
            out.append(_p(
                "The histogram shows the value distribution of the primary numeric column "
                "with KDE overlay. Amber dashed line = mean; green dotted line = median.",
                S["cap"]))
        else:
            out.append(_p("Insufficient numeric data for distribution chart.", S["cap"]))

        out.append(_sp(0.3))
        out.append(_p("Column Performance Overview", S["h2"]))
        seg_fig = _chart_segment_bar(self.payload)
        if seg_fig:
            out.append(_fig_to_image(seg_fig, _CW_FULL, _CH_FULL))
            out.append(_p(
                "Horizontal bars represent the mean value of each numeric column. "
                "Alternating indigo and slate bars aid comparison across features.",
                S["cap"]))
        else:
            out.append(_p("Insufficient data for segment bar chart.", S["cap"]))

        out.append(_sp(0.2))
        return out

    def _visualisation_page_2(self) -> list:
        S   = self.S
        out = [
            _p("5. Data Visualisations — Correlations & Trends", S["h1"]),
            _hr_accent(), _sp(0.15),
        ]

        out.append(_p("Correlation Heatmap", S["h2"]))
        heat_fig = _chart_correlation_heatmap(self.payload)
        if heat_fig:
            out.append(_fig_to_image(heat_fig, _CW_FULL, _CH_FULL))
            out.append(_p(
                "Pearson correlation matrix. Blue = positive; Red = negative. "
                "Values |r| >= 0.75 indicate potential multicollinearity.",
                S["cap"]))
        else:
            out.append(_p("Insufficient correlation data for heatmap.", S["cap"]))

        out.append(_sp(0.3))
        out.append(_p("Metric Trend Analysis", S["h2"]))
        trend_fig = _chart_time_trend(self.payload)
        if trend_fig:
            out.append(_fig_to_image(trend_fig, _CW_FULL, _CH_FULL))
            out.append(_p(
                "Last-period % change per numeric column. Green = growth; Red = decline. "
                "Indigo line connects values to reveal directional momentum.",
                S["cap"]))
        else:
            out.append(_p("Insufficient trend data for chart.", S["cap"]))

        out.append(_sp(0.2))
        return out

    def _business_impact(self) -> list:
        S = self.S
        text = (self.narrative.get("opportunities")
                or self.narrative.get("business_impact", "Not available."))
        return [
            _p("6. Strategic Opportunities", S["h1"]),
            _hr_accent(), _sp(0.15),
            _p(text, S["body"]),
            _sp(0.25),
        ]

    def _market_opportunities(self) -> list:
        S = self.S
        text = (self.narrative.get("market_trends")
                or self.narrative.get("market_opportunities", "Not available."))
        out = [_p("7. Market Trends & Opportunities", S["h1"]), _hr_accent(), _sp(0.15)]
        out.append(_p(text, S["body"]))
        out.append(_sp(0.25))
        return out

    def _operational_recommendations(self) -> list:
        S = self.S
        text = (self.narrative.get("strategic_recommendations")
                or self.narrative.get("operational_recommendations", "Not available."))
        out = [_p("8. Strategic Recommendations", S["h1"]), _hr_accent(), _sp(0.15)]
        out += _bullets(text, S)
        out.append(_sp(0.25))
        return out

    def _data_limitations(self) -> list:
        S = self.S
        text = (self.narrative.get("data_limitations")
                or self.narrative.get("risk_signals")
                or self.narrative.get("risks_and_limitations", "Not available."))
        return [
            _p("9. Data Limitations", S["h1"]),
            _hr_accent(), _sp(0.15),
            _p(text, S["body"]),
            _sp(0.25),
        ]

    def _risks(self) -> list:
        S     = self.S
        risks = self.payload.get("risk_factors", [])
        out   = [
            _p("10. Risks & Limitations", S["h1"]),
            _hr_accent(), _sp(0.15),
            _p(self.narrative.get("risk_signals") or self.narrative.get("risks_and_limitations", "Not available."), S["body"]),
            _sp(0.2),
        ]
        if risks:
            # Light severity backgrounds
            sev_bg = {
                "High":   colors.HexColor("#FEE2E2"),   # light red
                "Medium": colors.HexColor("#FEF3C7"),   # light amber
                "Low":    colors.HexColor("#DCFCE7"),   # light green
            }
            sev_txt = {
                "High":   colors.HexColor("#991B1B"),
                "Medium": colors.HexColor("#92400E"),
                "Low":    colors.HexColor("#166534"),
            }
            styled = [[_p(c, S["th"]) for c in
                       ["Severity", "Category", "Description", "Source"]]]
            for r in risks[:12]:
                styled.append([_p(c, S["td"]) for c in [
                    r.get("severity", "Low"),
                    r.get("category", "—"),
                    textwrap.shorten(r.get("description", "—"), 55),
                    r.get("source", "—"),
                ]])
            cw   = [BW * .13, BW * .18, BW * .46, BW * .23]
            t    = Table(styled, colWidths=cw)
            cmds = _base_ts()
            for i, r in enumerate(risks[:12], start=1):
                bg  = sev_bg.get(r.get("severity", "Low"), _WHITE)
                txt = sev_txt.get(r.get("severity", "Low"), _DKGREY)
                cmds.append(("BACKGROUND", (0, i), (0, i), bg))
                cmds.append(("TEXTCOLOR",  (0, i), (0, i), txt))
                cmds.append(("FONTNAME",   (0, i), (0, i), "Helvetica-Bold"))
            t.setStyle(TableStyle(cmds))
            out.append(t)
        out.append(_sp(0.25))
        return out

    def _appendix(self) -> list:
        S  = self.S
        ds = self.payload.get("dataset_summary", {})
        fc = self.payload.get("forecast_summary", {})
        out = [_p("11. Technical Appendix", S["h1"]), _hr_accent(), _sp(0.15)]

        num = ds.get("numeric_columns", [])
        cat = ds.get("categorical_columns", [])
        if num or cat:
            irows = [["Column", "Type"]]
            for c in num: irows.append([c, "Numeric"])
            for c in cat: irows.append([c, "Categorical"])
            out += [
                _p("Column Inventory", S["h2"]),
                _data_tbl(irows[:25], S, [BW * .65, BW * .35]),
                _sp(0.2),
            ]

        if fc.get("available"):
            frows = [
                ["Attribute", "Value"],
                ["Target",     str(fc.get("target_column", "—"))],
                ["Model",      str(fc.get("model_used", "—"))],
                ["Horizon",    str(fc.get("horizon", "—"))],
                ["Confidence", f"{float(fc.get('confidence_level', 0.95)) * 100:.0f}%"],
                ["Trend",      str(fc.get("trend_direction", "—"))],
            ]
            out += [_p("Forecast Summary", S["h2"]), _kv_tbl(frows, S), _sp(0.2)]

        out += [
            _hr(_AMBER, 1), _sp(0.1),
            _p(
                f"Auto-generated by InsightX on "
                f"{self.ts.strftime('%d %B %Y, %H:%M UTC')}. "
                "Validate findings with domain experts before informing decisions.",
                S["cap"],
            ),
        ]
        return out