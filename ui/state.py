import json
from pathlib import Path
import streamlit as st
from core.session import MediCareGuideSession

_SAVED_SESSION_PATH = Path.home() / ".medicareguide_session.json"

def _save_session_to_disk() -> None:
    """
    Persist the intake profile, context, step, screen, sort key, and
    language choice to a local JSON file so a browser refresh restores
    the user's progress without re-entering everything.

    Only profile + context are saved — DataFrames and analysis text are
    regenerated automatically when the user returns to SELECT mode.
    """
    session = st.session_state.get("session")
    if session is None:
        return
    payload = {
        "intake_step": st.session_state.intake_step,
        "screen":      st.session_state.screen,
        "sort_key":    st.session_state.sort_key,
        "language":    st.session_state.get("language", "English"),
        "profile":     session.state["profile"],
        "context":     session.state["context"],
    }
    try:
        _SAVED_SESSION_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass   # non-fatal — user just loses persistence


def _restore_session_from_disk() -> bool:
    """
    Load the saved session file (if any) and restore profile, context,
    sort key, and language to the current Streamlit session state.

    Always sets screen=WELCOME and intake_step=0 so the user starts fresh
    from the front page on every refresh. Returns True if a valid saved
    session was found (profile data restored), False otherwise.
    """
    if not _SAVED_SESSION_PATH.exists():
        return False
    try:
        payload = json.loads(_SAVED_SESSION_PATH.read_text())
        step    = int(payload.get("intake_step", 0))
        if step == 0:
            return False   # nothing useful was saved

        session = st.session_state.session
        session.state["profile"].update(payload.get("profile", {}))
        session.state["context"].update(payload.get("context", {}))

        # Always land on WELCOME with a clean intake so the user starts fresh.
        st.session_state.intake_step    = 0
        session.state["intake_step"]    = 0
        st.session_state.screen         = "WELCOME"

        st.session_state.sort_key  = payload.get("sort_key")
        st.session_state.language  = payload.get("language", "English")
        return True
    except Exception:
        return False


def _clear_saved_session() -> None:
    """Delete the saved session file. Called when the user clicks Start over."""
    try:
        _SAVED_SESSION_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def init_session_state(lookup) -> None:
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized     = True
    st.session_state.screen          = "WELCOME"
    st.session_state.intake_step     = 0
    st.session_state.explain_text    = ""
    st.session_state.explain_step    = -1
    st.session_state.session         = MediCareGuideSession(lookup)
    st.session_state.chat_history    = []
    st.session_state.filtered_df     = None
    st.session_state.sorted_df       = None
    st.session_state.sort_key        = None
    st.session_state.sort_label      = ""
    st.session_state.sort_reasoning  = ""
    st.session_state.filter_summary       = ""
    st.session_state.filter_explanation   = ""
    st.session_state.select_analysis      = ""
    st.session_state.audio_enabled        = False
    st.session_state.inference_mode       = "cloud"
    st.session_state.language             = "English"

    # Attempt to restore a previous session (profile/language only; intake resets to 0)
    _restore_session_from_disk()
