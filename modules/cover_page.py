import streamlit as st
# ── cover_page.py — InsightX splash screen ──────────────────────────────────
# The "Get Started" button is a real Streamlit widget pinned via CSS to sit
# visually inside the glass card.  pointer-events:none on the decorative layers
# ensures the button remains fully clickable.


def init_cover_state():
    if "app_started" not in st.session_state:
        st.session_state["app_started"] = False
    if "active_tab_index" not in st.session_state:
        st.session_state["active_tab_index"] = 0


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@300;400;500;600&display=swap');

/* ── 1. Hide all Streamlit chrome ── */
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer,
section[data-testid="stSidebar"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── 2. Full-screen animated canvas (pointer-events off so button works) ── */
.main-wrapper {
    height: 100vh; width: 100vw;
    position: fixed; top: 0; left: 0; z-index: 1;
    background: #050508;
    background-image:
        radial-gradient(ellipse 65% 55% at 50% 50%, rgba(90,75,210,0.50) 0%, transparent 68%),
        linear-gradient(rgba(100,120,255,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(100,120,255,0.07) 1px, transparent 1px);
    background-size: 100% 100%, 52px 52px, 52px 52px;
    animation: gridDrift 22s linear infinite;
    pointer-events: none;
}
@keyframes gridDrift {
    from { background-position: 0 0, 0 0, 0 0; }
    to   { background-position: 0 0, 52px 52px, 52px 52px; }
}
.main-wrapper::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 55% 48% at 50% 50%, rgba(80,65,200,0.45), transparent 70%);
    animation: breathe 7s ease-in-out infinite;
    pointer-events: none;
}
@keyframes breathe {
    0%,100% { opacity: 0.8; transform: scale(1); }
    50%      { opacity: 1;   transform: scale(1.07); }
}

/* ── 3. Glass card — fixed-centred, pointer-events off (decorative) ── */
.glass-card {
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10;
    width: min(540px, 88vw);
    background: rgba(22, 20, 60, 0.55);
    backdrop-filter: blur(30px) saturate(1.5);
    -webkit-backdrop-filter: blur(30px) saturate(1.5);
    border: 1px solid rgba(120,130,255,0.24);
    border-radius: 32px;
    /* extra bottom padding makes room for the button inside */
    padding: 64px 72px 100px;
    text-align: center;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.045) inset,
        0 40px 100px rgba(0,0,0,0.55),
        0 0 70px rgba(80,70,200,0.20);
    animation: cardIn 0.9s cubic-bezier(0.22,1,0.36,1) both;
    pointer-events: none;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 18%; right: 18%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    border-radius: 50%;
}
@keyframes cardIn {
    from { opacity: 0; transform: translate(-50%, calc(-50% + 28px)) scale(0.97); }
    to   { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}

/* ── 4. Card typography ── */
.ix-tag {
    font-family: 'Rajdhani', sans-serif;
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.7em; text-transform: uppercase;
    color: #00D4FF; opacity: 0.8;
    margin-bottom: 18px;
    animation: fadeUp 0.8s 0.20s ease both;
}
.ix-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: clamp(2.8rem, 6.5vw, 4.4rem);
    letter-spacing: 0.04em; line-height: 1;
    margin: 0 0 10px;
    animation: fadeUp 0.8s 0.30s ease both;
}
.ix-insight {
    background: linear-gradient(135deg, #c8d0ff 0%, #7b8cff 55%, #4a52cc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ix-x {
    background: linear-gradient(135deg, #00d4ff 0%, #00aaee 55%, #0075cc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 16px rgba(0,212,255,0.65));
}
.ix-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(0.75rem, 1.8vw, 0.9rem); font-weight: 400;
    letter-spacing: 0.32em; text-transform: uppercase;
    color: rgba(200,210,255,0.58);
    animation: fadeUp 0.8s 0.40s ease both;
}
.ix-divider {
    width: 52%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(120,130,255,0.38), transparent);
    margin: 26px auto 0;
    animation: fadeUp 0.8s 0.50s ease both;
}

/* ── 5. "Get Started" Streamlit button — pinned INSIDE the card ──
   Card bottom edge ≈  50vh + (card_height / 2).
   Card height with padding ≈ 400px  →  bottom edge ≈ 50vh + 200px.
   We place the button at  50vh + 130px  so it floats inside the card.
   Adjust the `top` value if your card text grows / shrinks.            ── */
div[data-testid="stButton"] {
    position: fixed !important;
    top: calc(50% + 118px) !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 200 !important;
    display: flex !important;
    justify-content: center !important;
    animation: fadeUp 0.8s 0.65s ease both;
}
div[data-testid="stButton"] > button {
    background: transparent !important;
    color: rgba(210, 220, 255, 0.85) !important;
    border: 1px solid rgba(150, 165, 255, 0.40) !important;
    padding: 8px 28px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.20em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    width: fit-content !important;
    min-width: unset !important;
    box-shadow: inset 0 0 14px rgba(110,130,255,0.07) !important;
    transition: border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(160, 180, 255, 0.85) !important;
    color: #ffffff !important;
    background: rgba(110, 130, 255, 0.12) !important;
    box-shadow: 0 0 22px rgba(110,130,255,0.28), inset 0 0 14px rgba(110,130,255,0.10) !important;
    transform: none !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.98) !important;
    background: rgba(110, 130, 255, 0.20) !important;
}

/* ── 6. Corner sparkle ── */
.sparkle {
    position: fixed; bottom: 36px; right: 42px;
    z-index: 300; width: 20px; height: 20px;
    animation: sparklePulse 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes sparklePulse {
    0%,100% { opacity: 0.45; transform: scale(1)    rotate(0deg); }
    50%      { opacity: 0.90; transform: scale(1.35) rotate(18deg); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""

SPARKLE_SVG = """
<div class="sparkle">
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2 L13.6 10.4 L22 12 L13.6 13.6 L12 22 L10.4 13.6 L2 12 L10.4 10.4 Z"
          fill="white"/>
  </svg>
</div>
"""


def cover_page():
    st.markdown(CSS, unsafe_allow_html=True)

    # Animated background canvas (pointer-events: none)
    st.markdown('<div class="main-wrapper"></div>', unsafe_allow_html=True)

    # Glass card with all text — sits above background, below button
    st.markdown(f"""
        <div class="glass-card">
            <div class="ix-tag">System Initialization</div>
            <div class="ix-title">
                <span class="ix-insight">INSIGHT</span><span class="ix-x">X</span>
            </div>
            <div class="ix-sub">Automated Intelligence Core</div>
            <div class="ix-divider"></div>
        </div>
        {SPARKLE_SVG}
    """, unsafe_allow_html=True)

    if st.button("Get Started", key="cover_start_btn"):
       st.session_state.app_started = True
       st.session_state.active_tab = "Data Readiness"

       st.rerun()


# ── Standalone preview ───────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="InsightX",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_cover_state()
    if not st.session_state["app_started"]:
        cover_page()
    else:
        st.title("InsightX — replace with your main app content")