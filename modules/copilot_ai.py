"""
modules/copilot_ai.py
─────────────────────
InsightX – Global AI Copilot

A sidebar-based conversational AI assistant grounded in the current
cleaned dataset.  Drop-in: call render_ai_copilot(df, ai_engine) after
the sidebar is opened in app.py.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from core.ai.ai_engine import AIEngine

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_HISTORY_KEY  = "copilot_chat_history"
_MAX_TOKENS   = 480
_MAX_CORR_PAIRS = 4
_MAX_NUMERIC_PREVIEW = 6

_SUGGESTED_QUESTIONS: list[str] = [
    "What are the key trends in this dataset?",
    "Which columns have the most missing values?",
    "What correlations should I investigate further?",
    "Summarise the numeric distributions in plain English.",
    "Are there any anomalies or outliers I should know about?",
    "What business decisions can I draw from this data?",
]

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior business intelligence analyst with deep expertise in
    data exploration, statistical reasoning, and translating numbers into
    actionable business recommendations.

    Always be concise, direct, and evidence-based. When you cite a number,
    reference the dataset context provided. Avoid generic statements —
    ground every insight in the actual data summary below.

    Dataset context:
    {dataset_summary}
""").strip()

_USER_TURN_TEMPLATE = "{user_question}"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — injected once per page load
# ─────────────────────────────────────────────────────────────────────────────

_COPILOT_CSS = """
<style>
/* ── Copilot header ───────────────────────────────── */
.ix-cop-header {
    background: linear-gradient(135deg, #0E1B2E 0%, #0D1320 100%);
    border: 1px solid #1E2D42;
    border-radius: 10px;
    padding: 14px 16px 12px;
    margin-bottom: 14px;
}
.ix-cop-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #F0F4FF;
}
.ix-cop-header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4A6080;
    margin-top: 3px;
    letter-spacing: 0.08em;
}

/* ── Chat bubbles ─────────────────────────────────── */
.ix-bubble-user {
    background: #0F2240;
    border-left: 2px solid #4A90D9;
    border-radius: 0 8px 8px 0;
    padding: 10px 12px;
    margin: 8px 0 4px;
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    color: #C8DCF0;
    line-height: 1.6;
    position: relative;
}
.ix-bubble-assistant {
    background: #0C2018;
    border-left: 2px solid #2E7D52;
    border-radius: 0 8px 8px 0;
    padding: 10px 12px;
    margin: 4px 0 8px;
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    color: #C8E6D0;
    line-height: 1.6;
    position: relative;
}
.ix-bubble-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.55;
    margin-bottom: 5px;
}
.ix-bubble-ts {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #3A5068;
    float: right;
    margin-top: -2px;
}

/* ── Empty state ──────────────────────────────────── */
.ix-cop-empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #3A4A60;
    letter-spacing: 0.06em;
    text-align: center;
    padding: 20px 0 12px;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataset_summary(df: pd.DataFrame) -> str:
    """
    Return a compact text summary of *df* suitable for LLM context.
    Uses numpy directly — no pd.np dependency.
    """
    lines: list[str] = []

    # Shape
    lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Column breakdown
    numeric_cols: list[str]     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols: list[str] = df.select_dtypes(exclude="number").columns.tolist()
    lines.append(
        f"Numeric columns ({len(numeric_cols)}): "
        + (", ".join(numeric_cols) if numeric_cols else "none")
    )
    lines.append(
        f"Categorical columns ({len(categorical_cols)}): "
        + (", ".join(categorical_cols) if categorical_cols else "none")
    )

    # Missing values (top offenders only)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        top_missing = missing.head(4)
        missing_str = "  |  ".join(
            f"{col}: {cnt} ({cnt / len(df) * 100:.1f}%)"
            for col, cnt in top_missing.items()
        )
        lines.append(f"Columns with missing values: {missing_str}")
    else:
        lines.append("Missing values: none")

    # Column means for first N numeric columns
    if numeric_cols:
        preview = numeric_cols[:_MAX_NUMERIC_PREVIEW]
        means   = df[preview].mean()
        stds    = df[preview].std()
        stat_parts = [
            f"{c}: mean={means[c]:.2f}, std={stds[c]:.2f}"
            for c in preview
        ]
        lines.append("Numeric stats: " + "  |  ".join(stat_parts))

    # Top correlations — uses numpy, not pd.np
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper = corr_matrix.where(mask)
        top_corr = (
            upper.stack()
            .sort_values(ascending=False)
            .head(_MAX_CORR_PAIRS)
        )
        if not top_corr.empty:
            corr_parts = [
                f"{a} ↔ {b}: {v:.2f}" for (a, b), v in top_corr.items()
            ]
            lines.append("Top correlations: " + "  |  ".join(corr_parts))

    # Categorical value counts (top category per column, first 3 cols)
    if categorical_cols:
        cat_preview = categorical_cols[:3]
        cat_parts: list[str] = []
        for col in cat_preview:
            top_val   = df[col].value_counts().idxmax()
            top_count = df[col].value_counts().max()
            cat_parts.append(
                f"{col}: most common = '{top_val}' ({top_count:,})"
            )
        lines.append("Categorical top values: " + "  |  ".join(cat_parts))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chat history helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_history() -> None:
    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []


def _append_message(role: str, content: str) -> None:
    st.session_state[_HISTORY_KEY].append(
        {
            "role":    role,
            "content": content,
            "ts":      datetime.now().strftime("%H:%M"),
        }
    )


def _clear_history() -> None:
    st.session_state[_HISTORY_KEY] = []


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(question: str, df: pd.DataFrame) -> str:
    summary = _build_dataset_summary(df)
    system  = _SYSTEM_PROMPT.format(dataset_summary=summary)
    user    = _USER_TURN_TEMPLATE.format(user_question=question)
    # Single-turn prompt: system context + user question
    return f"{system}\n\nUser question: {user}"


# ─────────────────────────────────────────────────────────────────────────────
# Core rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_ai_copilot(
    df: Optional[pd.DataFrame],
    ai_engine: Optional[AIEngine],
) -> None:
    """
    Render the InsightX AI Copilot inside ``st.sidebar``.

    Parameters
    ----------
    df :
        The cleaned DataFrame (``st.session_state.get("cleaned_df")``).
        May be *None* if no dataset has been loaded yet.
    ai_engine :
        Object exposing ``generate(prompt: str, max_tokens: int) -> str``.
        May be *None* if the engine is unavailable.
    """
    _init_history()

    with st.sidebar:
        st.markdown(_COPILOT_CSS, unsafe_allow_html=True)

        # ── Header ──────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="ix-cop-header">
                <div class="ix-cop-header-title">AI Copilot</div>
                <div class="ix-cop-header-sub">grounded in your dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Guard: engine missing ────────────────────────────────────────────
        if ai_engine is None:
            st.warning(
                "AI engine not configured. "
                "Set GROQ_API_KEY in your Streamlit secrets to activate the Copilot.",
                icon="⚠️",
            )
            return

        # ── Guard: no data ────────────────────────────────────────────────────
        if df is None or df.empty:
            st.info(
                "Load and clean a dataset to activate the Copilot.",
                icon="📂",
            )
            return

        # ── Dataset snapshot ─────────────────────────────────────────────────
        with st.expander("Dataset snapshot", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Rows",    f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            numeric_n = len(df.select_dtypes(include="number").columns)
            st.caption(
                f"Numeric: {numeric_n}  ·  "
                f"Categorical: {df.shape[1] - numeric_n}  ·  "
                f"Nulls: {df.isnull().sum().sum():,}"
            )

        # ── Suggested questions ──────────────────────────────────────────────
        with st.expander("Suggested questions", expanded=False):
            for suggestion in _SUGGESTED_QUESTIONS:
                if st.button(
                    suggestion,
                    key=f"sug_{suggestion[:32]}",
                    use_container_width=True,
                ):
                    _handle_question(suggestion, df, ai_engine)
                    st.rerun()

        # ── Conversation history ─────────────────────────────────────────────
        history: list[dict] = st.session_state[_HISTORY_KEY]

        if history:
            for msg in history:
                _render_bubble(msg["role"], msg["content"], msg.get("ts", ""))
        else:
            st.markdown(
                '<div class="ix-cop-empty">No messages yet — ask anything below.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1E2D42;margin:10px 0">',
            unsafe_allow_html=True,
        )

        # ── Input area ───────────────────────────────────────────────────────
        user_question: str = st.text_area(
            "Ask a question:",
            placeholder="e.g. Which features correlate most with revenue?",
            height=80,
            key="copilot_input",
            label_visibility="collapsed",
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_clicked = st.button(
                "Ask Copilot",
                type="primary",
                use_container_width=True,
                disabled=not (user_question or "").strip(),
            )
        with col_clear:
            if st.button("Clear", use_container_width=True, help="Clear conversation"):
                _clear_history()
                st.rerun()

        if send_clicked and (user_question or "").strip():
            _handle_question(user_question.strip(), df, ai_engine)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_question(
    question: str,
    df: pd.DataFrame,
    ai_engine: AIEngine,
) -> None:
    """Build the prompt, call the AI engine, and persist the exchange."""
    _append_message("user", question)

    prompt = _build_prompt(question, df)

    with st.sidebar:
        with st.spinner("Analysing…"):
            try:
                response: str = ai_engine.generate(prompt, max_tokens=_MAX_TOKENS)
                if not isinstance(response, str) or not response.strip():
                    response = (
                        "The AI engine returned an empty response. "
                        "Please try rephrasing your question."
                    )
            except Exception as exc:  # noqa: BLE001
                response = f"Error from AI engine: {exc}"

    _append_message("assistant", response)


def _render_bubble(role: str, content: str, ts: str) -> None:
    """Render a single chat bubble with consistent styling."""
    if role == "user":
        st.markdown(
            f"""
            <div class="ix-bubble-user">
                <div class="ix-bubble-label">You
                    <span class="ix-bubble-ts">{ts}</span>
                </div>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ix-bubble-assistant">
                <div class="ix-bubble-label">Copilot
                    <span class="ix-bubble-ts">{ts}</span>
                </div>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )