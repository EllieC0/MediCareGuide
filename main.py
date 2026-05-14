from pathlib import Path
import streamlit as st
from ui.backend import get_lookup, _warm_tts
from ui.state import init_session_state
from ui.screens.welcome import render_welcome
from ui.screens.intake import render_intake
from ui.screens.select import render_select

# ======================================================================== #
#  Page config — MUST be the very first Streamlit call                      #
# ======================================================================== #

st.set_page_config(
    page_title="MediGuide",
    page_icon="🧓",
    layout="centered",
)

# ── CSS injection ──────────────────────────────────────────────────────────
try:
    with open(Path(__file__).parent / "ui" / "style.css") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown("""
<style>
/* Tab labels — large enough for elderly users */
button[data-baseweb="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    font-family: Georgia, serif !important;
}
button[data-baseweb="tab"] > div,
button[data-baseweb="tab"] p {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    font-family: Georgia, serif !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================================== #
#  Initialize and Warm Cache                                                #
# ======================================================================== #

lookup = get_lookup()
init_session_state(lookup)

# ======================================================================== #
#  Main routing                                                             #
# ======================================================================== #

_screen = st.session_state.screen
if   _screen == "WELCOME": render_welcome()
elif _screen == "INTAKE":  render_intake()
elif _screen == "SELECT":  render_select()
else:
    st.error(f"Unknown screen state: {_screen!r}. Resetting.")
    st.session_state.clear()
    st.rerun()
