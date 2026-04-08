"""
modules/tab_data_readiness.py

AI-driven Data Readiness pipeline for the InsightX analytics platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback

try:
    from core.ingestion.data_loader import DataLoader
except ImportError:
    DataLoader = None

try:
    from core.profiling.profiler import DataProfiler
except ImportError:
    DataProfiler = None

try:
    from core.profiling.quality_engine import QualityEngine
except ImportError:
    QualityEngine = None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>

/* ── Design tokens ───────────────────────────────────────── */
:root {
    --bg-page:        #0d1117;
    --bg-card:        #111827;
    --bg-card-alt:    #161f2e;
    --bg-ai:          #0e1520;
    --border:         #1f2937;
    --border-focus:   #2d4a6e;
    --accent:         #3b82f6;
    --accent-dim:     rgba(59, 130, 246, 0.10);
    --success:        #22c55e;
    --success-dim:    rgba(34, 197, 94, 0.10);
    --danger:         #ef4444;
    --danger-dim:     rgba(239, 68, 68, 0.10);
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #64748b;
    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      14px;
    --shadow-sm:      0 1px 3px rgba(0,0,0,0.4);
    --shadow-md:      0 4px 12px rgba(0,0,0,0.4);
}

/* ── Section card ────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    margin-bottom: 6px;
    box-shadow: var(--shadow-md);
}

/* ── Section heading ─────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.section-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.section-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1;
}

/* ── Inline metric row ───────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 4px;
}
.metric-cell {
    flex: 1;
    min-width: 130px;
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    box-shadow: var(--shadow-sm);
}
.metric-cell .m-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 5px;
}
.metric-cell .m-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

/* ── Upload zone ─────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--border-focus);
    border-radius: var(--radius-lg);
    padding: 36px 24px;
    text-align: center;
    background: var(--bg-card);
    transition: border-color 0.2s, background 0.2s;
    margin-bottom: 10px;
}
.upload-zone:hover {
    border-color: var(--accent);
    background: var(--bg-card-alt);
}
.upload-zone .uz-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.upload-zone .uz-sub {
    font-size: 12px;
    color: var(--text-muted);
}

/* ── Quality comparison table ────────────────────────────── */
.q-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    font-size: 13px;
    margin-top: 6px;
}
.q-table th {
    background: var(--bg-card-alt);
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.q-table td {
    padding: 11px 16px;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
.q-table tr:last-child td {
    border-bottom: none;
}
.q-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.015);
}
.q-metric-name { font-weight: 600; color: var(--text-secondary); }
.q-val         { font-weight: 700; color: var(--text-primary);   }
.q-val-good    { font-weight: 700; color: var(--success);        }
.q-val-bad     { font-weight: 700; color: var(--danger);         }
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
}
.badge-good    { background: var(--success-dim); color: var(--success); }
.badge-bad     { background: var(--danger-dim);  color: var(--danger);  }
.badge-neutral { background: rgba(255,255,255,0.06); color: var(--text-muted); }

/* ── Pipeline stepper ────────────────────────────────────── */
.stepper {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 0;
    padding: 18px 0 10px;
}
.step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    min-width: 68px;
    max-width: 80px;
}
.step-line {
    position: absolute;
    top: 13px;
    left: calc(50% + 13px);
    width: calc(100% - 13px);
    height: 2px;
    background: var(--border);
}
.step:last-child .step-line { display: none; }
.step-dot {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    border: 2px solid var(--border-focus);
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 1;
}
.step.active .step-dot {
    border-color: var(--accent);
    background: var(--accent-dim);
    color: var(--accent);
}
.step.done .step-dot {
    border-color: var(--success);
    background: var(--success-dim);
    color: var(--success);
}
.step.active .step-line,
.step.done   .step-line {
    background: linear-gradient(90deg, var(--accent), var(--border));
}
.step-label {
    margin-top: 5px;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    line-height: 1.3;
}
.step.active .step-label { color: var(--accent);  }
.step.done   .step-label { color: var(--success); }

/* ── Feature engineering ─────────────────────────────────── */
.fe-group { margin-bottom: 12px; }
.fe-type-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.fe-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.fe-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: border-color 0.15s, background 0.15s;
    cursor: default;
}
.fe-pill:hover {
    border-color: var(--accent);
    background: var(--accent-dim);
    color: var(--text-primary);
}
.fe-arrow { color: var(--accent); font-weight: 700; }

/* ── AI analysis box ─────────────────────────────────────── */
.ai-box {
    background: var(--bg-ai);
    border: 1px solid var(--border-focus);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    line-height: 1.75;
    font-size: 13.5px;
    color: var(--text-secondary);
}
.ai-box-header {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}

/* ── Cleaning description ────────────────────────────────── */
.clean-desc {
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 12px;
}

/* ── Success banner ──────────────────────────────────────── */
.success-banner {
    background: var(--success-dim);
    border: 1px solid var(--success);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    text-align: center;
    box-shadow: var(--shadow-md);
}
.success-banner .sb-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--success);
    margin-bottom: 4px;
}
.success-banner .sb-sub {
    font-size: 13px;
    color: var(--text-muted);
}

/* ── Dataframe overrides ─────────────────────────────────── */
div[data-testid="stDataFrame"] > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Pipeline step labels
# ---------------------------------------------------------------------------

_STEPS = [
    "Upload",
    "Intelligence",
    "Metadata",
    "Quality",
    "Cleaning",
    "Comparison",
    "Preview",
    "Features",
    "Statistics",
    "Suitability",
]

# Step index mapping (0-based, matches _STEPS list):
#   0  Upload         — no data yet
#   1  Intelligence   — raw_df loaded
#   2  Metadata       — (rendered inline after intelligence)
#   3  Quality        — (rendered inline)
#   4  Cleaning       — (rendered inline, awaiting run)
#   5  Comparison     — cleaned_df exists (cleaning done)
#   6  Preview        — (rendered inline)
#   7  Features       — (rendered inline, awaiting run)
#   8  Statistics     — (rendered inline)
#   9  Suitability    — eng_df exists (features done) → pipeline complete


def _compute_stepper_index() -> int:
    """
    FIX (Problem 1): derive the active stepper step exclusively from
    session state flags.  The previous code hard-coded active_step = 5
    whenever cleaned_df existed, preventing the stepper from advancing
    past the Cleaning stage.
    """
    eng_df     = st.session_state.get("eng_df")
    cleaned_df = st.session_state.get("cleaned_df")
    raw_df     = st.session_state.get("raw_df")

    if eng_df is not None:
        # Feature engineering complete → return len(_STEPS) so every step
        # is evaluated as i < active_index in _stepper(), marking all done.
        return len(_STEPS)
    if cleaned_df is not None:
        # Cleaning done → Comparison is the active step (index 5)
        return 5
    if raw_df is not None:
        # File uploaded → Intelligence is active (index 1)
        return 1
    # Nothing loaded yet
    return 0


# ---------------------------------------------------------------------------
# UI component helpers
# ---------------------------------------------------------------------------

def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def _card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def _section_header(num: str, title: str):
    st.markdown(
        f'<div class="section-header">'
        f'  <div class="section-num">{num}</div>'
        f'  <p class="section-title">{title}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _metric_row(items: list):
    """items: list of {"label": str, "value": str}"""
    cells = ""
    for it in items:
        cells += (
            f'<div class="metric-cell">'
            f'  <div class="m-label">{it["label"]}</div>'
            f'  <div class="m-value">{it["value"]}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="metric-row">{cells}</div>', unsafe_allow_html=True)


def _stepper(steps: list, active_index: int):
    html = '<div class="stepper">'
    for i, label in enumerate(steps):
        if i < active_index:
            cls = "done"
            dot = "v"
        elif i == active_index:
            cls = "active"
            dot = str(i + 1)
        else:
            cls = ""
            dot = str(i + 1)
        html += (
            f'<div class="step {cls}">'
            f'  <div class="step-line"></div>'
            f'  <div class="step-dot">{dot}</div>'
            f'  <div class="step-label">{label}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _ai_box(content: str, header: str = "AI Analysis"):
    clean_content = str(content).replace("**", "")

    st.markdown(
        f'<div class="ai-box">'
        f'<div class="ai-box-header">{header}</div>'
        f'<div style="white-space:pre-line;">{clean_content}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _quality_comparison_table(initial_metrics: dict, after_metrics: dict):
    def _delta_badge(delta: float, lower_is_better: bool):
        if abs(delta) < 0.005:
            return '<span class="badge badge-neutral">no change</span>'
        improved = (delta < 0) if lower_is_better else (delta > 0)
        cls   = "badge-good" if improved else "badge-bad"
        arrow = "down " if delta < 0 else "up "
        return f'<span class="badge {cls}">{arrow}{abs(delta):.2f}</span>'

    def _val_cls(delta: float, lower_is_better: bool):
        if abs(delta) < 0.005:
            return "q-val"
        improved = (delta < 0) if lower_is_better else (delta > 0)
        return "q-val-good" if improved else "q-val-bad"

    rows_def = [
        ("Missing %",     "missing_pct",   True,  "%"),
        ("Duplicates %",  "duplicate_pct", True,  "%"),
        ("Outliers %",    "outlier_pct",   True,  "%"),
        ("Quality Score", "quality_score", False, ""),
    ]

    html = (
        '<table class="q-table">'
        "<thead><tr>"
        "<th>Metric</th><th>Before</th><th>After</th><th>Change</th>"
        "</tr></thead><tbody>"
    )
    for label, key, lib, unit in rows_def:
        before = initial_metrics.get(key, 0)
        after  = after_metrics.get(key,  0)
        delta  = round(after - before, 2)
        vcls   = _val_cls(delta, lib)
        html += (
            f"<tr>"
            f'  <td><span class="q-metric-name">{label}</span></td>'
            f'  <td class="q-val">{before}{unit}</td>'
            f'  <td class="{vcls}">{after}{unit}</td>'
            f'  <td>{_delta_badge(delta, lib)}</td>'
            f"</tr>"
        )
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def _feature_pills(datetime_fe: list, numeric_fe: list, categorical_fe: list):
    html = ""

    if datetime_fe:
        transforms   = ["year", "month", "weekday", "quarter"]
        cols_preview = ", ".join(datetime_fe[:3])
        if len(datetime_fe) > 3:
            cols_preview += f" +{len(datetime_fe) - 3} more"
        pills = "".join(
            f'<span class="fe-pill"><span class="fe-arrow">-&gt;</span> {t}</span>'
            for t in transforms
        )
        html += (
            f'<div class="fe-group">'
            f'  <div class="fe-type-label">Datetime &nbsp;·&nbsp; {cols_preview}</div>'
            f'  <div class="fe-pills">{pills}</div>'
            f'</div>'
        )

    if numeric_fe:
        transforms   = ["z-score", "log transform"]
        cols_preview = ", ".join(numeric_fe[:5])
        if len(numeric_fe) > 5:
            cols_preview += f" +{len(numeric_fe) - 5} more"
        pills = "".join(
            f'<span class="fe-pill"><span class="fe-arrow">-&gt;</span> {t}</span>'
            for t in transforms
        )
        html += (
            f'<div class="fe-group">'
            f'  <div class="fe-type-label">Numeric &nbsp;·&nbsp; {cols_preview}</div>'
            f'  <div class="fe-pills">{pills}</div>'
            f'</div>'
        )

    if categorical_fe:
        cols_preview = ", ".join(categorical_fe[:5])
        if len(categorical_fe) > 5:
            cols_preview += f" +{len(categorical_fe) - 5} more"
        pills = '<span class="fe-pill"><span class="fe-arrow">-&gt;</span> frequency encoding</span>'
        html += (
            f'<div class="fe-group">'
            f'  <div class="fe-type-label">Categorical &nbsp;·&nbsp; {cols_preview}</div>'
            f'  <div class="fe-pills">{pills}</div>'
            f'</div>'
        )

    if not html:
        st.info("No transformations applicable to the current dataset.")
        return

    st.markdown(html, unsafe_allow_html=True)


def _success_banner():
    st.markdown(
        '<div class="success-banner">'
        '  <div class="sb-title">Pipeline Complete</div>'
        '  <div class="sb-sub">All stages passed. The next tab is now unlocked.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _load_dataset(uploaded_file):
    """Load CSV using DataLoader with pandas fallback."""
    if DataLoader is not None:
        try:
            loader = DataLoader()
            df = loader.load_csv(uploaded_file)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def _classify_columns(df):
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols    = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number, "datetime64", "datetimetz"]).columns.tolist()
    return numeric_cols, categorical_cols, datetime_cols


def _compute_quality_metrics(df):
    metrics = {}
    if QualityEngine is not None:
        try:
            qe = QualityEngine(df)
            metrics["missing_pct"]   = qe.missing_percentage()
            metrics["duplicate_pct"] = qe.duplicate_percentage()
            metrics["outlier_pct"]   = qe.outlier_percentage()
            metrics["quality_score"] = qe.quality_score()
            return metrics
        except Exception:
            pass

    total_cells   = df.shape[0] * df.shape[1]
    missing_pct   = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0
    duplicate_pct = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0.0

    numeric_cols       = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_count      = 0
    total_numeric_vals = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_count      += ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        total_numeric_vals += len(series)

    outlier_pct   = (outlier_count / total_numeric_vals * 100) if total_numeric_vals > 0 else 0.0
    quality_score = max(0.0, 100.0 - missing_pct - duplicate_pct * 0.5 - outlier_pct * 0.3)

    metrics["missing_pct"]   = round(missing_pct,   2)
    metrics["duplicate_pct"] = round(duplicate_pct, 2)
    metrics["outlier_pct"]   = round(outlier_pct,   2)
    metrics["quality_score"] = round(quality_score, 2)
    return metrics


def _clean_dataframe(df):
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            try:
                converted = pd.to_numeric(cleaned[col], errors="coerce")
                if converted.notna().sum() / max(len(converted), 1) > 0.7:
                    cleaned[col] = converted
                    continue
            except Exception:
                pass
            try:
                converted = pd.to_datetime(cleaned[col], errors="coerce", infer_datetime_format=True)
                if converted.notna().sum() / max(len(converted), 1) > 0.7:
                    cleaned[col] = converted
                    continue
            except Exception:
                pass

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = cleaned.select_dtypes(exclude=[np.number, "datetime64", "datetimetz"]).columns.tolist()
    for col in categorical_cols:
        if cleaned[col].isnull().any():
            mode_vals = cleaned[col].mode()
            if not mode_vals.empty:
                cleaned[col] = cleaned[col].fillna(mode_vals[0])

    return cleaned


def _engineer_features(df):
    eng = df.copy()

    datetime_cols = eng.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in datetime_cols:
        try:
            eng[f"{col}_year"]    = eng[col].dt.year
            eng[f"{col}_month"]   = eng[col].dt.month
            eng[f"{col}_weekday"] = eng[col].dt.weekday
            eng[f"{col}_quarter"] = eng[col].dt.quarter
        except Exception:
            pass

    numeric_cols = eng.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        try:
            mean = eng[col].mean()
            std  = eng[col].std()
            if std and std > 0:
                eng[f"{col}_zscore"] = (eng[col] - mean) / std
        except Exception:
            pass
        try:
            if (eng[col] > 0).all():
                eng[f"{col}_log"] = np.log1p(eng[col])
        except Exception:
            pass

    categorical_cols = eng.select_dtypes(exclude=[np.number, "datetime64", "datetimetz"]).columns.tolist()
    for col in categorical_cols:
        try:
            freq_map = eng[col].value_counts(normalize=True).to_dict()
            eng[f"{col}_freq_enc"] = eng[col].map(freq_map)
        except Exception:
            pass

    return eng


def _compute_statistics(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        stats[col] = {
            "mean":     round(series.mean(),     4),
            "std":      round(series.std(),      4),
            "skew":     round(series.skew(),     4),
            "kurtosis": round(series.kurtosis(), 4),
        }
    return pd.DataFrame(stats).T.reset_index().rename(columns={"index": "column"})


def _detect_dataset_type(df):
    col_str = " ".join(c.lower() for c in df.columns)
    if any(k in col_str for k in ["revenue", "sales", "price", "amount", "profit"]):
        return "financial / sales"
    if any(k in col_str for k in ["customer", "user", "client", "churn", "segment"]):
        return "customer"
    if any(k in col_str for k in ["date", "time", "timestamp", "period"]):
        return "time-series / operational"
    if any(k in col_str for k in ["product", "inventory", "order", "sku"]):
        return "operational"
    return "general"


def _ai_dataset_intelligence(df, ai_engine):
    col_info = {col: str(df[col].dtype) for col in df.columns}
    context = (
        f"Column names and types: {col_info}\n"
        f"Row count: {len(df)}\n"
        "Analyze this dataset schema and describe what type of dataset it appears to be, "
        "its potential domain (sales, financial, customer, operational), complexity, "
        "and possible business context. Be concise and professional."
    )
    if ai_engine is not None:
        try:
            result = ai_engine.generate(context)
            if result:
                return str(result)
        except Exception:
            pass
    dataset_type = _detect_dataset_type(df)
    numeric_cols, categorical_cols, datetime_cols = _classify_columns(df)
    return (
        f"Detected dataset type: {dataset_type}. "
        f"The dataset contains {len(df)} rows and {len(df.columns)} columns "
        f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(datetime_cols)} datetime). "
        f"Complexity is {'high' if len(df.columns) > 20 else 'moderate' if len(df.columns) > 10 else 'low'}."
    )


def _ai_suitability(df, metrics, dataset_type, ai_engine):
    numeric_cols, categorical_cols, _ = _classify_columns(df)
    context = (
        f"Dataset rows: {len(df)}\n"
        f"Dataset columns: {len(df.columns)}\n"
        f"Quality score: {metrics.get('quality_score', 'N/A')}\n"
        f"Numeric columns: {numeric_cols}\n"
        f"Categorical columns: {categorical_cols}\n"
        f"Detected dataset type: {dataset_type}\n"
        "Based on the above, list the analytics use cases this dataset is suitable for. "
        "Return bullet-point recommendations only."
    )
    if ai_engine is not None:
        try:
            result = ai_engine.explain_suitability(context)
            if result:
                return str(result)
        except Exception:
            try:
                result = ai_engine.generate(context)
                if result:
                    return str(result)
            except Exception:
                pass

    recommendations = []
    if len(numeric_cols) >= 3:
        recommendations.append("- Suitable for regression and forecasting")
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 2:
        recommendations.append("- Suitable for classification models")
    if len(df) > 1000:
        recommendations.append("- Suitable for customer or behavioral segmentation")
    if metrics.get("outlier_pct", 0) > 2:
        recommendations.append("- Suitable for anomaly detection")
    recommendations.append("- Suitable for business intelligence dashboards")
    return "\n".join(recommendations)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(ai_engine=None):
    _inject_css()

    st.title("Data Readiness Pipeline")
    st.caption("AI-enhanced enterprise data preparation powered by InsightX.")

    # ── Stepper (FIX Problem 1) ──────────────────────────────────────────────
    # Progress is derived entirely from session state; no stale flag involved.
    active_step = _compute_stepper_index()
    _stepper(_STEPS, active_step)

    # ── 1 · DATA UPLOAD ─────────────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("1", "Data Upload")

        st.markdown(
            '<div class="upload-zone">'
            '  <div class="uz-title">Drag and drop your CSV file here</div>'
            '  <div class="uz-sub">or click Browse files below</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload a CSV dataset",
            type=["csv"],
            label_visibility="collapsed",
        )
        _card_close()

    # Upload guard: load only when either no dataset exists yet, or the user
    # has uploaded a different file (detected by filename change).  This
    # prevents the infinite-rerun loop that occurs when Streamlit keeps the
    # UploadedFile object alive across reruns while raw_df is already set.
    _new_file = (
        uploaded_file is not None
        and (
            st.session_state.get("raw_df") is None
            or st.session_state.get("uploaded_filename") != uploaded_file.name
        )
    )
    if _new_file:
        try:
            df = _load_dataset(uploaded_file)
            st.session_state["raw_df"]           = df
            st.session_state["cleaned_df"]       = None
            st.session_state["eng_df"]           = None
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.rerun()  # advance stepper to step 1 immediately
        except Exception as exc:
            st.error(f"Failed to load dataset: {exc}")
            return

    raw_df = st.session_state.get("raw_df")

    if raw_df is None:
        st.info("Upload a CSV file to begin the data readiness pipeline.")
        return

    df = raw_df

    _metric_row([
        {"label": "Rows",    "value": f"{len(df):,}"},
        {"label": "Columns", "value": str(len(df.columns))},
        {"label": "Memory",  "value": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"},
    ])

    # ── 2 · DATASET INTELLIGENCE ─────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("2", "Dataset Intelligence")

        dataset_type = _detect_dataset_type(df)
        numeric_cols, categorical_cols, datetime_cols = _classify_columns(df)
        complexity = "High" if len(df.columns) > 20 else "Moderate" if len(df.columns) > 10 else "Low"

        _metric_row([
            {"label": "Detected Type",    "value": dataset_type.title()},
            {"label": "Complexity",       "value": complexity},
            {"label": "Column Diversity", "value": f"{len(numeric_cols)}N / {len(categorical_cols)}C / {len(datetime_cols)}D"},
        ])

        with st.expander("AI Dataset Analysis", expanded=True):
            with st.spinner("Analyzing dataset..."):
                intelligence_text = _ai_dataset_intelligence(df, ai_engine)
            _ai_box(intelligence_text, header="Dataset Intelligence")

        _card_close()

    # ── 3 · METADATA SUMMARY ────────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("3", "Metadata Summary")

        avg_cardinality = (
            round(np.mean([df[c].nunique() for c in df.columns]), 1)
            if len(df.columns) > 0 else 0
        )
        total_cells     = len(df) * len(df.columns)
        missing_density = (
            round(df.isnull().sum().sum() / total_cells * 100, 2)
            if total_cells > 0 else 0.0
        )

        _metric_row([
            {"label": "Numeric Cols",     "value": str(len(numeric_cols))},
            {"label": "Categorical Cols", "value": str(len(categorical_cols))},
            {"label": "Datetime Cols",    "value": str(len(datetime_cols))},
            {"label": "Avg Cardinality",  "value": str(avg_cardinality)},
            {"label": "Missing Density",  "value": f"{missing_density}%"},
        ])

        _card_close()

    # ── 4 · DATA QUALITY SCORE ──────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("4", "Data Quality Score")

        initial_metrics = _compute_quality_metrics(df)

        _metric_row([
            {"label": "Missing %",     "value": f"{initial_metrics['missing_pct']}%"},
            {"label": "Duplicates %",  "value": f"{initial_metrics['duplicate_pct']}%"},
            {"label": "Outliers %",    "value": f"{initial_metrics['outlier_pct']}%"},
            {"label": "Quality Score", "value": str(initial_metrics["quality_score"])},
        ])

        _card_close()

    # ── 5 · AI CLEANING PIPELINE ────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("5", "AI Cleaning Pipeline")

        st.markdown(
            '<div class="clean-desc">'
            "The cleaning pipeline removes duplicates, converts mistyped columns, "
            "and imputes missing values using <strong>median</strong> (numeric) "
            "and <strong>mode</strong> (categorical)."
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("Run AI Cleaning Pipeline", use_container_width=True):
            with st.spinner("Executing cleaning pipeline..."):
                try:
                    cleaned = _clean_dataframe(df)
                    st.session_state["cleaned_df"] = cleaned
                    st.session_state["eng_df"]     = None  # reset downstream
                    st.success("Cleaning pipeline completed successfully.")
                    st.rerun()  # rerun immediately so stepper advances to step 5
                except Exception as exc:
                    st.error(f"Cleaning failed: {exc}")
                    st.code(traceback.format_exc())

        _card_close()

    cleaned_df = st.session_state.get("cleaned_df")

    if cleaned_df is None:
        return

    # ── 6 · QUALITY COMPARISON ──────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("6", "Quality Comparison")

        after_metrics = _compute_quality_metrics(cleaned_df)
        _quality_comparison_table(initial_metrics, after_metrics)

        _card_close()

    # ── 7 · CLEANED DATA PREVIEW ────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("7", "Cleaned Data Preview")

        st.dataframe(cleaned_df.head(20), use_container_width=True, height=300)

        _card_close()

    # ── 8 · FEATURE ENGINEERING ─────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("8", "Feature Engineering")

        datetime_fe    = cleaned_df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        numeric_fe     = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_fe = cleaned_df.select_dtypes(
            exclude=[np.number, "datetime64", "datetimetz"]
        ).columns.tolist()

        with st.expander("Planned Transformations", expanded=True):
            _feature_pills(datetime_fe, numeric_fe, categorical_fe)

        if st.button("Generate AI Features", use_container_width=True):
            with st.spinner("Engineering features..."):
                try:
                    eng = _engineer_features(cleaned_df)
                    st.session_state["eng_df"] = eng
                    new_cols = len(eng.columns) - len(cleaned_df.columns)
                    st.success(f"Feature engineering complete. {new_cols} new columns added.")
                    st.dataframe(eng.head(5), use_container_width=True, height=240)
                    st.rerun()  # rerun so stepper advances to step 9 (done)
                except Exception as exc:
                    st.error(f"Feature engineering failed: {exc}")
                    st.code(traceback.format_exc())

        _card_close()

    # ── 9 · STATISTICAL SUMMARY ─────────────────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("9", "Statistical Summary")

        stats_df = _compute_statistics(cleaned_df)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, height=300)
        else:
            st.info("No numeric columns available for statistical analysis.")

        _card_close()

    # ── 10 · AI DATA SUITABILITY ANALYSIS ───────────────────────────────────
    st.divider()
    with st.container():
        _card_open()
        _section_header("10", "AI Data Suitability Analysis")

        with st.expander("Suitability Recommendations", expanded=True):
            with st.spinner("Running suitability analysis..."):
                suitability_text = _ai_suitability(
                    cleaned_df, after_metrics, dataset_type, ai_engine
                )
            _ai_box(suitability_text, header="Suitability Recommendations")

        _card_close()

    # ── NEXT NAVIGATION ──────────────────────────────────────────────────────
    st.divider()
    col_left, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state["active_tab"] = "Executive Dashboard"
            st.rerun()