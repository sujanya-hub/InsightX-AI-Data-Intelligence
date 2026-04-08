"""
InsightX — AI Powered Data Intelligence Platform
================================================
Entry point: app.py

Responsibilities
----------------
* Page configuration
* Session state bootstrap
* Cover page gate
* AI engine initialisation (cached)
* Sidebar navigation
* Dataset access guard
* Tab routing
* Header / footer

All business logic lives in modules/.  This file is a pure controller.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# Must be the very first Streamlit call in the script.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightX",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Module Imports
# ---------------------------------------------------------------------------

from modules.cover_page   import cover_page, init_cover_state
from modules.copilot_ai   import render_ai_copilot

import modules.tab_data_readiness as tab_data_readiness
import modules.tab_dashboard      as tab_dashboard
import modules.tab_decision        as tab_decision
import modules.tab_ai_insight      as tab_ai_insight
import modules.tab_report          as tab_report

from core.ai.ai_engine import AIEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE:    str = "InsightX"
APP_SUBTITLE: str = "AI Powered Data Intelligence Platform"

# Tab name constants — single source of truth used by navigation,
# routing, and the dataset guard.
TAB_DATA_READINESS: str = "Data Readiness"
TAB_EXEC_DASHBOARD: str = "Executive Dashboard"
TAB_DECISION_LAB:   str = "Decision Lab"
TAB_AI_INSIGHT_LAB: str = "AI Insight Lab"
TAB_REPORT_STUDIO:  str = "Report Studio"

TAB_LABELS: list[str] = [
    TAB_DATA_READINESS,
    TAB_EXEC_DASHBOARD,
    TAB_DECISION_LAB,
    TAB_AI_INSIGHT_LAB,
    TAB_REPORT_STUDIO,
]

# Tabs that require a cleaned dataset before they can render.
GATED_TABS: set[str] = {
    TAB_EXEC_DASHBOARD,
    TAB_DECISION_LAB,
    TAB_AI_INSIGHT_LAB,
    TAB_REPORT_STUDIO,
}

# Sidebar display labels (numbered for enterprise style).
_NAV_ITEMS: list[tuple[str, str]] = [
    (TAB_DATA_READINESS, "01  Data Readiness"),
    (TAB_EXEC_DASHBOARD, "02  Executive Dashboard"),
    (TAB_DECISION_LAB,   "03  Decision Lab"),
    (TAB_AI_INSIGHT_LAB, "04  AI Insight Lab"),
    (TAB_REPORT_STUDIO,  "05  Report Studio"),
]

# ---------------------------------------------------------------------------
# Sidebar CSS
# Injected once per run; scoped entirely to the sidebar.
# ---------------------------------------------------------------------------

_SIDEBAR_CSS: str = """
<style>
/* Sidebar shell */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2937 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem !important;
}

/* Brand block */
.ix-brand {
    padding: 0 1rem 1rem 1rem;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 0.5rem;
}
.ix-brand-title {
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 2px;
}
.ix-brand-sub {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3b82f6;
}

/* Nav section label */
.ix-nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    padding: 0.8rem 1rem 0.3rem 1rem;
}

/* Nav buttons — reset all Streamlit default chrome */
div[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    margin-bottom: 2px !important;
    text-align: left !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, color 0.15s !important;
    box-shadow: none !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.08) !important;
    color: #f1f5f9 !important;
}

/* Active nav item — applied via a wrapper div */
div[data-testid="stSidebar"] .ix-nav-active > div > button {
    background: rgba(59,130,246,0.12) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    border-left: 3px solid #3b82f6 !important;
    padding-left: calc(1rem - 3px) !important;
}

/* Divider above AI copilot */
.ix-sidebar-divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 0.75rem 1rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """
    Safely bootstrap every session state key the application needs.

    Uses a defaults dict so new keys can be added in one place without
    scattering `if key not in st.session_state` guards across the codebase.
    """
    defaults: dict[str, object] = {
        # Cover page gate
        "app_started":       False,
        # Active navigation tab
        "active_tab":        TAB_DATA_READINESS,
        # Data pipeline artefacts
        "raw_df":            None,
        "cleaned_df":        None,
        "eng_df":            None,
        # Tracks the last loaded filename to detect file replacement
        # without triggering the infinite-rerun loop.
        "uploaded_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------------------------------------------------------------------
# AI Engine
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _init_ai_engine() -> AIEngine | None:
    """
    Construct the AIEngine exactly once and cache it for the server lifetime.

    Reads GROQ_API_KEY from Streamlit secrets.  Returns None gracefully if
    the key is absent or construction fails, so AI-dependent features degrade
    rather than crash.
    """
    try:
        api_key: str | None = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.warning("GROQ_API_KEY not set in secrets.toml — AI features disabled.")
            return None
        return AIEngine(api_key=api_key)
    except Exception as exc:
        st.warning(f"AI Engine could not be initialised: {exc}")
        return None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header() -> None:
    """Render the platform masthead below the sidebar."""
    st.markdown(
        f"<h1 style='margin-bottom:0;'>{APP_TITLE}</h1>"
        f"<p style='opacity:0.6;margin-top:0.2rem;font-size:1rem;'>{APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )
    st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(ai_engine: AIEngine | None) -> None:
    """
    Render the enterprise sidebar navigation and the AI Copilot widget.

    Navigation pattern
    ------------------
    Each tab is a plain st.button with a unique key.  On click the button
    sets active_tab in session state and calls st.rerun().  This is
    identical to the "Next →" pattern used inside tab modules, so there is
    no widget-state caching conflict and no override race.

    The active item is wrapped in a <div class="ix-nav-active"> so the CSS
    rule above can style it without JavaScript.
    """
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="ix-brand">'
            '  <div class="ix-brand-title">InsightX</div>'
            '  <div class="ix-brand-sub">Analytical Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ix-nav-label">Platform</div>', unsafe_allow_html=True)

        current_tab: str = st.session_state.get("active_tab", TAB_DATA_READINESS)
        if current_tab not in TAB_LABELS:
            current_tab = TAB_DATA_READINESS

        for tab_key, tab_display in _NAV_ITEMS:
            is_active: bool = tab_key == current_tab

            # Wrap active button so CSS can target it by class name.
            if is_active:
                st.markdown('<div class="ix-nav-active">', unsafe_allow_html=True)

            if st.button(tab_display, key=f"sidenav_{tab_key}", use_container_width=True):
                # Only navigate when the user selects a different tab.
                # Clicking the already-active tab is a no-op — prevents a
                # redundant rerun that would interrupt any in-progress widget.
                if tab_key != current_tab:
                    st.session_state.active_tab = tab_key
                    st.rerun()

            if is_active:
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="ix-sidebar-divider">', unsafe_allow_html=True)

        # AI Copilot — always visible; receives None gracefully when no data.
        render_ai_copilot(
            df=st.session_state.get("cleaned_df"),
            ai_engine=ai_engine,
        )

# ---------------------------------------------------------------------------
# Dataset Guard
# ---------------------------------------------------------------------------

def _dataset_guard(tab: str) -> bool:
    """
    Prevent gated tabs from rendering without a cleaned dataset.

    Returns True  → execution may continue.
    Returns False → a warning has been shown; caller must not render the tab.
    """
    if tab not in GATED_TABS:
        return True

    if st.session_state.get("cleaned_df") is not None:
        return True

    st.warning("Run Data Readiness first — no dataset is loaded.")
    if st.button("Go to Data Readiness", type="primary"):
        st.session_state.active_tab = TAB_DATA_READINESS
        st.rerun()

    return False

# ---------------------------------------------------------------------------
# Tab Router
# ---------------------------------------------------------------------------

def _route_tab(tab: str, ai_engine: AIEngine | None) -> None:
    """
    Dispatch to the correct tab module.

    Each branch is wrapped in a try/except so a bug in one module cannot
    bring down the entire application.
    """
    df = st.session_state.get("cleaned_df")

    try:
        if tab == TAB_DATA_READINESS:
            tab_data_readiness.render(ai_engine)

        elif tab == TAB_EXEC_DASHBOARD:
            tab_dashboard.render_dashboard(df, ai_engine)

        elif tab == TAB_DECISION_LAB:
            tab_decision.render_decision_tab(df, ai_engine)

        elif tab == TAB_AI_INSIGHT_LAB:
            tab_ai_insight.render_ai_insight(df, ai_engine)

        elif tab == TAB_REPORT_STUDIO:
            tab_report.render(ai_engine)

        else:
            st.error(f"Unknown tab: '{tab}'. Please select a valid section from the sidebar.")

    except Exception as exc:
        st.error(f"An error occurred while rendering '{tab}': {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _render_footer() -> None:
    st.markdown(
        """
        <style>
        .ix-footer {
            text-align: center;
            padding: 2.5rem 0 1rem 0;
            font-size: 0.78rem;
            opacity: 0.38;
            letter-spacing: 0.05em;
        }
        </style>
        <div class="ix-footer">
            InsightX &nbsp;|&nbsp; AI Powered Data Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main Controller
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Application entry point.

    Execution order
    ---------------
    1. Session state bootstrap (safe, idempotent)
    2. Cover page gate          (st.stop() if not started)
    3. AI engine init           (cached; None on failure)
    4. Sidebar                  (navigation + copilot)
    5. Read active tab          (session state is authoritative)
    6. Header
    7. Dataset guard            (blocks gated tabs without data)
    8. Tab routing
    9. Footer
    """

    # 1. Bootstrap session state and cover-page state.
    _init_session_state()
    init_cover_state()

    # 2. Cover page gate — stop here until the user clicks through.
    if not st.session_state.get("app_started", False):
        cover_page()
        st.stop()

    # 3. AI engine — constructed once, cached for the server lifetime.
    ai_engine: AIEngine | None = _init_ai_engine()

    # 4. Sidebar navigation + AI Copilot.
    #    Each nav button sets active_tab and calls st.rerun() directly.
    #    There is no return value; session state is the single source of truth.
    _render_sidebar(ai_engine)

    # 5. Read the active tab.  Guard against any invalid stored value that
    #    could arise after a hot-reload or session migration.
    active_tab: str = st.session_state.get("active_tab", TAB_DATA_READINESS)
    if active_tab not in TAB_LABELS:
        active_tab = TAB_DATA_READINESS
        st.session_state.active_tab = active_tab

    # 6. Platform header.
    _render_header()

    # 7. Dataset guard — show warning and return early for gated tabs.
    if not _dataset_guard(active_tab):
        _render_footer()
        return

    # 8. Route to the selected tab module.
    _route_tab(active_tab, ai_engine)

    # 9. Footer.
    _render_footer()


if __name__ == "__main__":
    main()