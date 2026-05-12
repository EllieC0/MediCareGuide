"""
MediGuide — Streamlit UI
=========================
Elderly-friendly Medicare plan advisor powered by Gemma 4 (Ollama).

Run:
    streamlit run app.py

Inference modes (toggle on WELCOME screen):
  ⚡ Fast mode  — gemma4:31b-cloud (Ollama-hosted, zero-retention)
  🔒 Private mode — gemma4:e4b (fully local, nothing leaves your machine)

Place CY2026_Landscape_202603.csv in the same directory, or edit CSV_PATH.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from core.lookup    import MediCareGuideLookup, sort_plans, derive_sort_key, SORT_LABELS
from core.inference import (
    build_prompt_welcome_mode,
    build_prompt_select_mode,
    build_prompt_filter_explanation,
    build_prompt_sort_reasoning,
)
from core.ollama    import call_ollama, split_system_and_user
from core.session   import MediCareGuideSession
from ui.international      import t as _ti18n


def _t(key: str, **kwargs) -> str:
    """Shorthand: translate *key* using the current session language."""
    lang = st.session_state.get("language", "English")
    return _ti18n(key, lang, **kwargs)

try:
    from core.tts import generate_audio_bytes, TTS_AVAILABLE
except ImportError:
    TTS_AVAILABLE = False

    def generate_audio_bytes(text, voice="af_heart", speed=1.0, ui_language="English"):   # type: ignore[misc]
        return None

try:
    from core.stt import transcribe_streamlit_audio, WHISPER_AVAILABLE as STT_AVAILABLE
except ImportError:
    STT_AVAILABLE = False

    def transcribe_streamlit_audio(audio_bytes):   # type: ignore[misc]
        return None

try:
    from core.rag import build_or_load_index, retrieve, format_context, RAG_AVAILABLE
except ImportError:
    RAG_AVAILABLE = False

    def build_or_load_index(*a, **kw):             # type: ignore[misc]
        return None

    def retrieve(*a, **kw):                        # type: ignore[misc]
        return []

    def format_context(*a, **kw):                  # type: ignore[misc]
        return ""


# ======================================================================== #
#  Constants                                                                #
# ======================================================================== #

CSV_PATH           = Path(__file__).parent / "data" / "CY2026_Landscape_202603.csv"
_SAVED_SESSION_PATH = Path.home() / ".medicareguide_session.json"

MEDIGAP_REFERRAL = (
    "Medigap (Medicare Supplement) plans aren't in my local database — "
    "they're sold directly by private insurers and aren't part of the "
    "CMS Landscape file I use.\n\n"
    "To compare Medigap options in your area, visit: **medicare.gov/find-a-plan**\n\n"
    "I can still answer general questions about how Medigap works, "
    "what the different plan letters mean (G, N, etc.), or how it "
    "compares to Medicare Advantage — just ask."
)

_EXPLAIN_TRACK = """
<strong>How the three coverage types differ</strong>
<table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.93rem;">
  <thead>
    <tr style="background:#003366; color:#fff;">
      <th style="padding:8px 10px; text-align:left;"></th>
      <th style="padding:8px 10px; text-align:left;">Medicare Advantage</th>
      <th style="padding:8px 10px; text-align:left;">Part D Only</th>
      <th style="padding:8px 10px; text-align:left;">Original + Medigap</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Doctor choice</td>
      <td style="padding:7px 10px;">Restricted network (HMO or PPO)</td>
      <td style="padding:7px 10px;">Any doctor that accepts Medicare</td>
      <td style="padding:7px 10px;">Any doctor that accepts Medicare</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Prescriptions included</td>
      <td style="padding:7px 10px;">Usually yes</td>
      <td style="padding:7px 10px;">Yes — that is its only purpose</td>
      <td style="padding:7px 10px;">No — you must add a separate Part D plan</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Dental, vision, hearing</td>
      <td style="padding:7px 10px;">Often included</td>
      <td style="padding:7px 10px;">Not included</td>
      <td style="padding:7px 10px;">Not included</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Out-of-pocket cap (MOOP)</td>
      <td style="padding:7px 10px;">Yes — limits your yearly costs</td>
      <td style="padding:7px 10px;">No cap on medical costs</td>
      <td style="padding:7px 10px;">Medigap covers gaps; no single cap</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Monthly premium</td>
      <td style="padding:7px 10px;">Often $0, but has copays per visit</td>
      <td style="padding:7px 10px;">Low to moderate</td>
      <td style="padding:7px 10px;">Higher, but costs are more predictable</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Referrals needed</td>
      <td style="padding:7px 10px;">Often yes (HMO plans)</td>
      <td style="padding:7px 10px;">No</td>
      <td style="padding:7px 10px;">No</td>
    </tr>
  </tbody>
</table>
"""

_EXPLAIN_SNP = """
<strong>What these three items mean — and why they matter</strong>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Dual Eligible (D-SNP)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>You qualify if you receive both <strong>Medicare</strong> (federal) <strong>and Medicaid</strong> (state) benefits</li>
    <li>Special D-SNP plans coordinate both programs — less paperwork, less confusion</li>
    <li>Typically $0 or very low out-of-pocket costs for most services</li>
    <li>Often adds extra benefits: rides to appointments, meal delivery, over-the-counter allowances</li>
  </ul>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Chronic Condition (C-SNP)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>Plans built around managing one serious ongoing condition (diabetes, heart failure, COPD, etc.)</li>
    <li>Your care team specializes in your condition — more targeted than a general plan</li>
    <li>May include disease management programs, dedicated nurse support lines, and easier specialist access</li>
  </ul>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Extra Help / Low Income Subsidy (LIS)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>A federal program that reduces what you pay for prescription drugs</li>
    <li>Lowers Part D costs — premiums, deductibles, and copays can drop significantly or go to $0</li>
    <li>Based on income and assets — many people qualify without realizing it</li>
    <li>If you are not sure, check — it costs nothing to apply</li>
  </ul>
</div>
"""

_EXPLAIN_BUDGET = """
<strong>Understanding your plan costs</strong>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">What a premium is</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>The fixed monthly amount you pay just to have the plan — whether you use it or not</li>
    <li>This is <strong>separate from and on top of</strong> your Part B premium (<strong>$185/month in 2026</strong>)</li>
  </ul>
</div>

<div style="margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin:12px 0 8px;">The four costs to know</div>
  <table style="width:100%; border-collapse:collapse; font-size:0.93rem;">
    <thead>
      <tr style="background:#003366; color:#fff;">
        <th style="padding:8px 10px; text-align:left;">Cost type</th>
        <th style="padding:8px 10px; text-align:left;">What it means</th>
        <th style="padding:8px 10px; text-align:left;">When you pay it</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#fff;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Premium</td>
        <td style="padding:7px 10px;">Monthly fee to hold the plan</td>
        <td style="padding:7px 10px;">Every month, always</td>
      </tr>
      <tr style="background:#f7f9fc;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Deductible</td>
        <td style="padding:7px 10px;">Amount you pay before the plan starts covering costs</td>
        <td style="padding:7px 10px;">First uses each year</td>
      </tr>
      <tr style="background:#fff;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Copay</td>
        <td style="padding:7px 10px;">Fixed fee per doctor visit or service</td>
        <td style="padding:7px 10px;">Each time you use care</td>
      </tr>
      <tr style="background:#f7f9fc;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">MOOP</td>
        <td style="padding:7px 10px;">The most you would ever pay in a year — plan covers 100% after this</td>
        <td style="padding:7px 10px;">Stops your costs at a cap</td>
      </tr>
    </tbody>
  </table>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">The $0 premium trade-off</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>Many Medicare Advantage plans charge $0/month — but you pay copays each time you use care</li>
    <li>A $0 premium plan can cost more overall if you visit doctors frequently</li>
    <li>A higher premium plan often means lower copays — more predictable total costs</li>
  </ul>
</div>
"""

_EXPLAIN_PREFS = """
<strong>What each preference does for your results</strong>

<table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.93rem;">
  <thead>
    <tr style="background:#003366; color:#fff;">
      <th style="padding:8px 10px; text-align:left;">Preference</th>
      <th style="padding:8px 10px; text-align:left;">How it affects your recommendations</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Regular prescriptions</td>
      <td style="padding:7px 10px;">Plans are sorted by lowest drug deductible or total annual cost — drug costs are prioritised</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Keep my doctors</td>
      <td style="padding:7px 10px;">Gemma flags which plan types restrict your network and reminds you to verify your doctors are in-network before enrolling</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Dental / vision / hearing</td>
      <td style="padding:7px 10px;">Gemma highlights which plans include these benefits and adds a checklist item to confirm what is actually covered</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Prefer PPO</td>
      <td style="padding:7px 10px;">PPO plans are shown before HMO plans in your results</td>
    </tr>
  </tbody>
</table>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">HMO vs PPO — what is the difference?</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li><strong>HMO:</strong> Lower premiums, but you must use doctors in the plan's network and usually need a referral to see a specialist</li>
    <li><strong>PPO:</strong> Higher premiums, but you can see any doctor (in or out of network) without a referral — more flexibility</li>
  </ul>
</div>
"""

# Short labels for the 6 sort override buttons
SORT_BUTTON_LABELS: dict[str, str] = {
    "lowest_premium":    "Lowest Premium",
    "total_cost":        "Total Annual Cost",
    "star_rating":       "Star Rating",
    "lowest_moop":       "Lowest MOOP",
    "lowest_deductible": "Lowest Deductible",
    "ppo_first":         "PPO First",
}

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
#  Cached backend                                                           #
# ======================================================================== #

@st.cache_resource(show_spinner="Loading Medicare plan database…")
def get_lookup() -> MediCareGuideLookup:
    return MediCareGuideLookup(CSV_PATH)


@st.cache_resource(show_spinner="Loading voice model… (first time only)")
def _warm_tts() -> None:
    """Pre-load the Kokoro ONNX model at startup so audio plays instantly."""
    if TTS_AVAILABLE:
        try:
            from core.tts import _get_kokoro
            _get_kokoro()
        except Exception:
            pass


_warm_tts()


@st.cache_resource(show_spinner="Loading Medicare handbook index… (first time only)")
def get_rag_index():
    """
    Build or load the FAISS index from the CMS Medicare & You 2026 PDF.
    Cached by Streamlit — runs once per server process.
    First run takes ~30 s on CPU to embed ~350 chunks; subsequent startups
    load the saved index from ~/.core.rag/ in < 1 s.
    """
    return build_or_load_index()


# ── Kick off index load at startup (non-blocking cache warm) ──────────────
if RAG_AVAILABLE:
    get_rag_index()


# ======================================================================== #
#  Session persistence helpers                                              #
# ======================================================================== #

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


def init_session_state() -> None:
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized     = True
    st.session_state.screen          = "WELCOME"
    st.session_state.intake_step     = 0
    st.session_state.explain_text    = ""
    st.session_state.explain_step    = -1
    st.session_state.session         = MediCareGuideSession(get_lookup())
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


# ======================================================================== #
#  Shared helpers                                                           #
# ======================================================================== #

def render_language_selector(key_suffix: str = "") -> None:
    """Language selector rendered as three pill toggle buttons."""
    current = st.session_state.get("language", "English")
    labels  = [("English", "EN"), ("中文", "中文"), ("Español", "ES")]
    st.markdown('<div class="lang-pill">', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (lang, label) in zip(cols, labels):
        with col:
            if st.button(
                label,
                key=f"lang_{lang}_{key_suffix}",
                type="primary" if lang == current else "secondary",
                use_container_width=True,
            ):
                if lang != current:
                    st.session_state.language = lang
                    _save_session_to_disk()
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def maybe_play_audio(text: str) -> None:
    """Play audio via st.audio() if audio is enabled and TTS available."""
    if st.session_state.audio_enabled and TTS_AVAILABLE:
        wav = generate_audio_bytes(
            text,
            ui_language=st.session_state.get("language", "English"),
        )
        if wav:
            st.audio(wav, format="audio/wav", autoplay=True)


def safe_md(text: str) -> None:
    """Render markdown while preventing Streamlit from interpreting $ as LaTeX."""
    st.markdown(text.replace("$", r"\$"))


def render_chat_history() -> None:
    """Render all display-layer chat messages."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                safe_md(msg["content"])
                if msg.get("sources"):
                    st.caption(f"📖 *Medicare & You 2026 — {msg['sources']}*")
            else:
                st.markdown(msg["content"])



def render_voice_input(widget_key: str) -> "str | None":
    """
    Render a microphone input widget and return the transcription if the
    user just recorded audio, otherwise return None.

    Only shown when STT_AVAILABLE is True (sounddevice installed).
    Uses st.audio_input() which records directly in the browser — no
    extra browser permissions dialog beyond the standard mic prompt.

    Args:
        widget_key: Unique Streamlit key for this widget instance.

    Returns:
        Transcribed text string, or None if no audio / transcription failed.
    """
    if not STT_AVAILABLE:
        return None

    st.caption(_t("voice_hint"))
    # After each successful transcription we bump this counter so the widget
    # renders with a new key on the next rerun, clearing the browser's stale
    # MediaRecorder state that causes "An error has occurred, please try again."
    gen_key = f"_mic_gen_{widget_key}"
    actual_key = f"{widget_key}_{st.session_state.get(gen_key, 0)}"
    audio_val = st.audio_input(
        "🎤 Or speak your question",
        key=actual_key,
        label_visibility="collapsed",
    )
    if audio_val is None:
        return None

    # Avoid re-processing the same recording on reruns.
    # id(audio_val) is unreliable across reruns (Streamlit creates new objects),
    # so hash the raw audio bytes instead for stable dedup.
    audio_bytes = audio_val.read()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    seen_key = f"_stt_seen_{widget_key}"
    if st.session_state.get(seen_key) == audio_hash:
        return None
    st.session_state[seen_key] = audio_hash

    with st.spinner(_t("transcribing")):
        text = transcribe_streamlit_audio(
            audio_bytes,
            ui_language=st.session_state.get("language", "English"),
        )

    if text:
        # Bump generation counter — next rerun gets a fresh widget with no error state
        st.session_state[gen_key] = st.session_state.get(gen_key, 0) + 1
        return text

    st.warning(_t("stt_failed"))
    return None


def ask_welcome(user_input: str) -> str:
    """Ask a WELCOME-mode question. Returns Gemma's answer string.

    Saves and restores intake_step and mode so that keywords in a
    casual welcome question (e.g. "what is PPO?") cannot accidentally
    advance the intake wizard or jump straight to SELECT mode.

    If the RAG index is available, retrieves the top-3 relevant passages
    from Medicare & You 2026 and injects them into the system prompt so
    Gemma can cite specific handbook content rather than relying solely on
    its training weights.
    """
    session = st.session_state.session
    saved_step = session.state["intake_step"]
    saved_mode = session.state["mode"]

    # ── RAG: retrieve relevant handbook passages ──────────────────────────
    rag_context = ""
    if RAG_AVAILABLE:
        rag_index = get_rag_index()
        if rag_index is not None:
            results     = retrieve(user_input, rag_index, k=3)
            rag_context = format_context(results, max_words=400)

    state  = session.process_turn(user_input)
    prompt = build_prompt_welcome_mode(
        user_input,
        state,
        language    = st.session_state.get("language", "English"),
        rag_context = rag_context,
    )
    with st.spinner("Thinking…"):
        answer = call_ollama("", prompt, [], mode=st.session_state.get("inference_mode", "cloud"))
    session.close_turn(answer)

    # Restore — welcome chat must not advance the intake wizard
    session.state["intake_step"] = saved_step
    session.state["mode"]        = saved_mode
    return answer


def handle_select_turn(user_input: str) -> str:
    """
    Handle one SELECT-mode follow-up turn.
    Filters, sorts, builds prompt, calls Ollama.
    Updates sort metadata in session state.
    """
    session = st.session_state.session
    lookup  = get_lookup()
    state   = session.process_turn(user_input)
    profile = state["profile"]

    if profile.get("track") == "MEDIGAP":
        session.close_turn(MEDIGAP_REFERRAL)
        return MEDIGAP_REFERRAL

    filtered_df, _ = lookup.get_plans_filtered(profile["zip"], profile)

    manual_key = st.session_state.sort_key
    if manual_key:
        sort_key      = manual_key
        sort_reasoning = f"Sorted by: {SORT_LABELS.get(sort_key, sort_key)}"
    else:
        sort_key, sort_reasoning = derive_sort_key(profile, state["context"])

    sorted_df, sort_label = sort_plans(filtered_df, sort_key, profile)

    prompt = build_prompt_select_mode(
        user_question  = user_input,
        plans          = sorted_df.head(5),
        state          = state,
        sort_label     = sort_label,
        sort_reasoning = sort_reasoning,
        language       = st.session_state.get("language", "English"),
    )
    system_part, user_part = split_system_and_user(prompt)
    with st.spinner("Thinking…"):
        answer = call_ollama(system_part, user_part, [], mode=st.session_state.get("inference_mode", "cloud"))
    session.close_turn(answer)

    # Keep sorted data in sync with UI
    st.session_state.sorted_df      = sorted_df
    st.session_state.sort_key       = sort_key
    st.session_state.sort_label     = sort_label
    st.session_state.sort_reasoning  = sort_reasoning

    return answer


def _render_inline_explain(step: int) -> None:
    """If an explanation for this step is ready, render it inline below the buttons."""
    if (
        st.session_state.get("explain_step") == step
        and st.session_state.get("explain_text")
    ):
        st.write("")
        st.markdown(
            f"<div style='background:#f0f4fa; border-left:4px solid #003366; "
            f"border-radius:8px; padding:16px 20px; font-size:0.95rem;'>"
            f"{st.session_state.explain_text}</div>",
            unsafe_allow_html=True,
        )


# ======================================================================== #
#  WELCOME screen                                                           #
# ======================================================================== #

def render_welcome() -> None:
    # ── Language selector ─────────────────────────────────────────────────
    _, top_r = st.columns([3, 2])
    with top_r:
        render_language_selector("welcome")
    # ── Hero + CTA — unified visual block ────────────────────────────────
    # Wrapped in st.container() so CSS can target the inner stVerticalBlock
    # and set gap:0, visually connecting the hero banner and the CTA button.
    # st.button fires over WebSocket — no page reload, no new tab.
    with st.container():
        st.markdown(f"""
        <div class="hero-banner">
            <div class="hero-badge">{_t("hero_badge")}</div>
            <div class="hero-title">🧓 Medicare Guide</div>
            <div class="hero-tagline">{_t("hero_tagline")}</div>
        </div>
        <div class="hero-cta-section"></div>
        """, unsafe_allow_html=True)
        if st.button(
            _t("cta_button"),
            type="secondary",
            use_container_width=True,
            key="cta_btn",
        ):
            st.session_state.screen = "INTAKE"
            st.session_state.intake_step = 0
            st.rerun()

    # Style the CTA button via JS — only targets the button inside
    # the .hero-banner container, leaving all other buttons untouched.
    # Kept outside st.container() so the iframe wrapper is not a child of the
    # hero stVerticalBlock (which would trigger the white-panel CSS on the iframe).
    st.html("""
    <script>
    (function() {
        function styleBtn() {
            var doc = window.parent.document;
            var banner = doc.querySelector('.hero-banner');
            if (!banner) return false;
            var container = banner.closest('[data-testid="stVerticalBlock"]');
            if (!container) return false;
            var btns = container.querySelectorAll(
                'button:not([data-testid="baseButton-primary"])'
            );
            btns.forEach(function(btn) {
                btn.style.setProperty('min-height', '80px', 'important');
                btn.style.setProperty('padding-bottom', '0', 'important');
                var p = btn.querySelector('p');
                if (p) {
                    p.style.setProperty('font-size', '1.8rem', 'important');
                    p.style.setProperty('font-weight', '700', 'important');
                    p.style.setProperty('letter-spacing', '0.02em', 'important');
                }
            });
            return btns.length > 0;
        }
        // Retry until React has rendered the button
        var tries = 0;
        var iv = setInterval(function() {
            if (styleBtn() || ++tries > 30) clearInterval(iv);
        }, 100);
    })();
    </script>
    """)

    st.divider()

    # ── Chat section ──────────────────────────────────────────────────────
    st.markdown(f"#### {_t('welcome_chat_heading')}")

    if st.session_state.chat_history:
        render_chat_history()

    # Play TTS deferred from the previous rerun so answer text renders first.
    if st.session_state.get("_pending_tts"):
        st.audio(st.session_state._pending_tts, format="audio/wav", autoplay=True)
        del st.session_state["_pending_tts"]
    elif st.session_state.get("_pending_tts_text"):
        _tts_text = st.session_state.pop("_pending_tts_text")
        with st.spinner(_t("transcribing") or "Generating audio…"):
            wav = generate_audio_bytes(
                _tts_text,
                ui_language=st.session_state.get("language", "English"),
            )
        if wav:
            st.audio(wav, format="audio/wav", autoplay=True)

    # Inline chat form — taller input via CSS
    st.markdown("""
    <style>
    div[data-testid="stForm"] [data-baseweb="input"] {
        min-height: 120px !important;
        align-items: center !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input {
        font-size: 1.4rem !important;
    }
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        min-height: 120px !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("welcome_chat", clear_on_submit=True):
        col_inp, col_btn = st.columns([6, 1])
        with col_inp:
            user_input = st.text_input(
                "question",
                placeholder=_t("welcome_chat_placeholder"),
                label_visibility="collapsed",
                key="welcome_chat_input",
            )
        with col_btn:
            submitted = st.form_submit_button(_t("ask_button"), use_container_width=True)

    # Voice input — outside form (st.audio_input cannot live inside st.form)
    voiced = render_voice_input("welcome_mic")
    if voiced:
        submitted = True
        user_input = voiced

    if submitted and user_input.strip():
        q = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": q})
        answer = ask_welcome(q)
        sources = _extract_sources(answer)
        clean_answer = _strip_sources(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": clean_answer, "sources": sources})
        # Always rerun so chat_history renders first (answer text is visible
        # immediately), then TTS generates on the next pass without blocking display.
        if st.session_state.audio_enabled and TTS_AVAILABLE:
            st.session_state._pending_tts_text = clean_answer
        st.rerun()

    st.divider()

    # ── How it works ──────────────────────────────────────────────────────
    st.markdown(f'<div class="how-it-works-title">{_t("how_it_works")}</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""<div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">{_t("step1_title")}</div>
            <div class="step-desc">{_t("step1_desc")}</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""<div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">{_t("step2_title")}</div>
            <div class="step-desc">{_t("step2_desc")}</div>
        </div>""", unsafe_allow_html=True)
    with h3:
        st.markdown(f"""<div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">{_t("step3_title")}</div>
            <div class="step-desc">{_t("step3_desc")}</div>
        </div>""", unsafe_allow_html=True)

    # ── Inference mode toggle ──────────────────────────────────────────────
    st.divider()
    _is_cloud = st.session_state.get("inference_mode", "cloud") == "cloud"
    infer_label = "⚡ Fast mode (Ollama Cloud)" if _is_cloud else "🔒 Private mode (Fully local)"
    infer_help  = (
        "Currently using Ollama Cloud — fast responses, no data stored after your session. "
        "Click to switch to fully local inference (slower, nothing leaves your machine)."
        if _is_cloud else
        "Currently running fully locally — private, but slower. "
        "Click to switch to Ollama Cloud (fast, zero-retention)."
    )
    col_itxt, col_ibtn = st.columns([5, 1])
    with col_itxt:
        st.markdown(
            f"<p style='font-size:1.15rem; font-weight:600; color:#333; margin:0.4rem 0;'>"
            f"{infer_label}</p>",
            unsafe_allow_html=True,
        )
    with col_ibtn:
        if st.button("Switch", key="infer_toggle_welcome", help=infer_help,
                     use_container_width=True):
            st.session_state.inference_mode = "local" if _is_cloud else "cloud"
            st.rerun()

    # ── Audio toggle — bottom of welcome page ─────────────────────────────
    if TTS_AVAILABLE:
        st.divider()
        audio_label = _t("audio_on") if st.session_state.audio_enabled else _t("audio_off")
        col_txt, col_btn = st.columns([5, 1])
        with col_txt:
            st.markdown(
                f"<p style='font-size:1.15rem; font-weight:600; color:#333; margin:0.4rem 0;'>"
                f"{_t('audio_accessibility')}</p>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(audio_label, key="audio_welcome", help=_t("audio_toggle_help"),
                         use_container_width=True):
                st.session_state.audio_enabled = not st.session_state.audio_enabled
                st.rerun()


# ======================================================================== #
#  INTAKE screen                                                            #
# ======================================================================== #

def render_intake() -> None:
    session = st.session_state.session
    profile = session.state["profile"]

    # Persist progress on every intake render (cheap JSON write)
    _save_session_to_disk()

    # MEDIGAP short-circuit — show referral immediately
    if profile.get("track") == "MEDIGAP":
        render_medigap_referral()
        return

    title_col, lang_col = st.columns([3, 2])
    with title_col:
        st.title(_t("intake_title"))
    with lang_col:
        render_language_selector("intake")

    step = st.session_state.intake_step
    st.progress(min(step / 5, 1.0), text=_t("intake_progress", step=step))
    st.divider()

    # Route to current intake step
    if   step == 0: render_step0_zip()
    elif step == 1: render_step1_track()
    elif step == 2: render_step2_snp()
    elif step == 3: render_step3_budget()
    elif step == 4: render_step4_prefs()
    elif step >= 5: trigger_select_mode()

    # ── Chat persists during intake ───────────────────────────────────────
    st.divider()
    st.markdown(
        f"<p style='font-size:1.05rem; font-weight:700; color:#003366; margin-bottom:4px;'>"
        f"{_t('intake_chat_heading')}</p>"
        f"<p style='font-size:0.9rem; color:#666; margin-top:0;'>"
        f"{_t('intake_chat_sub')}</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.chat_history:
        with st.expander(_t("view_history"), expanded=False):
            render_chat_history()

    with st.form("intake_chat", clear_on_submit=True):
        col_inp, col_ask = st.columns([6, 1])
        with col_inp:
            user_input = st.text_input(
                "question",
                placeholder=_t("intake_chat_placeholder"),
                label_visibility="collapsed",
                key="intake_chat_input",
            )
        with col_ask:
            asked = st.form_submit_button(_t("ask_button"), use_container_width=True)

    # Voice input — outside form
    voiced = render_voice_input("intake_mic")
    if voiced:
        asked = True
        user_input = voiced

    if asked and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.chat_message("user"):
            st.markdown(user_input.strip())
        answer = ask_welcome(user_input.strip())
        sources = _extract_sources(answer)
        clean_answer = _strip_sources(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": clean_answer, "sources": sources})
        with st.chat_message("assistant"):
            safe_md(clean_answer)
            if sources:
                st.caption(f"📖 *Medicare & You 2026 — {sources}*")
        maybe_play_audio(clean_answer)

    # ── Back button — below the question box ──────────────────────────────
    st.write("")
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("←", key="intake_back", help="Go back", use_container_width=False):
            if step == 0:
                st.session_state.screen = "WELCOME"
            else:
                st.session_state.intake_step = step - 1
                session.state["intake_step"] = step - 1
            st.rerun()


# ── Step 0 — ZIP code ──────────────────────────────────────────────────────

def render_step0_zip() -> None:
    st.subheader(_t("zip_prompt"))
    st.caption(_t("zip_hint"))
    st.write("")

    zip_val = st.text_input(
        _t("zip_label"),
        max_chars=5,
        placeholder=_t("zip_placeholder"),
        key="zip_text_input",
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(_t("continue_button"), type="primary", use_container_width=True, key="zip_continue"):
            v = zip_val.strip()
            if not v.isdigit() or len(v) != 5:
                st.error(_t("zip_error_invalid"))
            else:
                state = st.session_state.session.set_intake_field(0, "zip", v)
                if state["profile"]["zip"]:
                    p = state["profile"]
                    st.session_state.intake_step = state["intake_step"]
                    location = v
                    if p.get("county") and p.get("state"):
                        location += f" ({p['county']}, {p['state']})"
                    st.success(_t("zip_found", location=location))
                    st.rerun()
                else:
                    st.error(_t("zip_error_not_found", zip=v))


# ── Step 1 — Coverage track ────────────────────────────────────────────────

def render_step1_track() -> None:
    st.subheader(_t("track_heading"))
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(_t("track_ma_heading"))
        st.caption(_t("track_ma_desc"))
        if st.button(
            _t("track_ma_button"),
            use_container_width=True,
            type="primary",
            key="track_ma",
        ):
            st.session_state.session.set_intake_field(1, "track", "MA_D", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col2:
        st.markdown(_t("track_pdp_heading"))
        st.caption(_t("track_pdp_desc"))
        if st.button(
            _t("track_pdp_button"),
            use_container_width=True,
            key="track_pdp",
        ):
            st.session_state.session.set_intake_field(1, "track", "PDP", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col3:
        st.markdown(_t("track_medigap_heading"))
        st.caption(_t("track_medigap_desc"))
        if st.button(
            _t("track_medigap_button"),
            use_container_width=True,
            key="track_medigap",
        ):
            st.session_state.session.set_intake_field(1, "track", "MEDIGAP", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    st.write("")
    col_skip, col_explain = st.columns(2)
    with col_skip:
        if st.button(
            _t("track_skip"),
            use_container_width=True,
            key="track_skip",
        ):
            st.session_state.session.set_intake_field(1, "track", None, "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()
    with col_explain:
        if st.button(
            _t("explain_difference"),
            use_container_width=True,
            key="track_explain",
        ):
            st.session_state.explain_text = _EXPLAIN_TRACK
            st.session_state.explain_step = 1

    _render_inline_explain(1)


# ── Step 2 — SNP / situation flags ────────────────────────────────────────

def render_step2_snp() -> None:
    st.subheader(_t("snp_heading"))
    st.caption(_t("snp_caption"))
    st.write("")

    d_snp = st.checkbox(_t("snp_d_snp"), key="snp_d_snp")
    c_snp = st.checkbox(_t("snp_c_snp"), key="snp_c_snp")
    lis   = st.checkbox(_t("snp_lis"),   key="snp_lis")

    st.write("")
    col_cont, col_none, col_exp = st.columns(3)

    with col_cont:
        if st.button(
            _t("continue_button"),
            type="primary",
            use_container_width=True,
            key="snp_continue",
        ):
            flags = []
            if d_snp: flags.append("D_SNP")
            if c_snp: flags.append("C_SNP")
            if lis:   flags.append("LIS")
            st.session_state.session.set_intake_field(2, "snp_flags", flags, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_none:
        if st.button(
            _t("snp_none"),
            use_container_width=True,
            key="snp_none",
        ):
            st.session_state.session.set_intake_field(2, "snp_flags", [], "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(
            _t("explain_items"),
            use_container_width=True,
            key="snp_explain",
        ):
            st.session_state.explain_text = _EXPLAIN_SNP
            st.session_state.explain_step = 2

    _render_inline_explain(2)


# ── Step 3 — Budget ────────────────────────────────────────────────────────

def render_step3_budget() -> None:
    st.subheader(_t("budget_heading"))
    st.markdown(
        f"<p style='font-size:1rem; color:#444;'>{_t('budget_desc')}</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    no_limit = st.checkbox(_t("budget_no_limit"), key="budget_no_limit")

    budget_val = None
    if not no_limit:
        budget_val = st.slider(
            _t("budget_slider_label"),
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            format="$%d",
            key="budget_slider",
        )
        st.markdown(
            f"<p style='font-size:1rem; color:#444;'>{_t('budget_display', val=budget_val)}</p>",
            unsafe_allow_html=True,
        )

    st.write("")
    col_cont, col_exp = st.columns([2, 1])

    with col_cont:
        if st.button(
            _t("continue_button"),
            type="primary",
            use_container_width=True,
            key="budget_continue",
        ):
            if no_limit:
                st.session_state.session.set_intake_field(3, "budget_max", None, "skip")
            else:
                st.session_state.session.set_intake_field(3, "budget_max", budget_val, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(
            _t("explain_premiums"),
            use_container_width=True,
            key="budget_explain",
        ):
            st.session_state.explain_text = _EXPLAIN_BUDGET
            st.session_state.explain_step = 3

    _render_inline_explain(3)


# ── Step 4 — Preferences ──────────────────────────────────────────────────

def render_step4_prefs() -> None:
    st.subheader(_t("prefs_heading"))
    st.caption(_t("prefs_caption"))
    st.write("")

    has_rx    = st.checkbox(_t("pref_has_rx"),    key="pref_has_rx")
    keep_docs = st.checkbox(_t("pref_keep_docs"), key="pref_keep_docs")
    dental    = st.checkbox(_t("pref_dental"),    key="pref_dental")
    ppo       = st.checkbox(_t("pref_ppo"),       key="pref_ppo")

    st.write("")
    col_find, col_none, col_exp = st.columns(3)

    with col_find:
        if st.button(
            _t("find_plans_button"),
            type="primary",
            use_container_width=True,
            key="prefs_find",
        ):
            flags = []
            if has_rx:    flags.append("has_rx")
            if keep_docs: flags.append("keep_doctors")
            if dental:    flags.append("wants_dental")
            if ppo:       flags.append("prefers_ppo")
            st.session_state.session.set_intake_field(4, "context", flags, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_none:
        if st.button(
            _t("prefs_none"),
            use_container_width=True,
            key="prefs_none",
        ):
            st.session_state.session.set_intake_field(4, "context", [], "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(
            _t("explain_prefs"),
            use_container_width=True,
            key="prefs_explain",
        ):
            st.session_state.explain_text = _EXPLAIN_PREFS
            st.session_state.explain_step = 4

    _render_inline_explain(4)


# ── MEDIGAP referral ──────────────────────────────────────────────────────

def render_medigap_referral() -> None:
    st.title("🧓 MediGuide")
    st.warning(
        "**Medigap (Medicare Supplement) plans** are not included in the "
        "CMS Landscape database. Medigap policies are sold directly by "
        "private insurers — they are not part of the plan data I have access to."
    )
    st.markdown(
        "**To compare Medigap plans in your area, visit:**\n\n"
        "🔗 [medicare.gov/find-a-plan](https://www.medicare.gov/find-a-plan/)\n\n"
        "I can still answer general questions about how Medigap works, "
        "what the plan letters mean (G, N, K, L, etc.), or how "
        "Medigap compares to Medicare Advantage — just ask below."
    )
    st.divider()

    render_chat_history()

    user_input = st.chat_input(_t("medigap_chat_placeholder"))
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        answer = ask_welcome(user_input)
        sources = _extract_sources(answer)
        clean_answer = _strip_sources(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": clean_answer, "sources": sources})
        with st.chat_message("assistant"):
            safe_md(clean_answer)
            if sources:
                st.caption(f"📖 *Medicare & You 2026 — {sources}*")
        maybe_play_audio(clean_answer)

    st.divider()
    col_restart, _ = st.columns([2, 1])
    with col_restart:
        if st.button(
            _t("medigap_restart"),
            use_container_width=True,
        ):
            st.session_state.clear()
            st.rerun()


# ======================================================================== #
#  trigger_select_mode — transition INTAKE → SELECT                        #
# ======================================================================== #

def _generate_summary_html() -> str:
    """
    Build a self-contained, printable HTML page from the current SELECT
    session state. No new Gemma call — uses data already in session state.

    Included sections:
      - User profile (ZIP, track, budget, preferences)
      - Why these plans appear first (sort_reasoning)
      - Top 5 plan cards (premium, MOOP, deductible, stars, Gemma WHY sentence)
      - Before you enroll checklist (same items as the Gemma prompt)

    Returns an HTML string suitable for st.download_button(data=...).
    """
    from datetime import date

    session  = st.session_state.session
    profile  = session.state["profile"]
    context  = session.state["context"]
    sdf      = st.session_state.sorted_df
    sort_reasoning = st.session_state.sort_reasoning or st.session_state.sort_label
    gemma_whys     = _parse_plan_whys(st.session_state.get("select_analysis", ""))

    # ── Profile text ──────────────────────────────────────────────────
    loc = profile.get("zip", "")
    if profile.get("county") and profile.get("state"):
        loc += f" ({profile['county']}, {profile['state']})"
    track_map = {
        "MA_D":    "Medicare Advantage (Part C + D)",
        "PDP":     "Part D Drug Plan Only",
        "MEDIGAP": "Original Medicare + Medigap",
    }
    track  = track_map.get(profile.get("track", ""), profile.get("track", "—"))
    budget = f"${profile['budget_max']}/month" if profile.get("budget_max") is not None else "No limit"

    prefs = []
    if context.get("has_rx"):        prefs.append("Takes regular prescriptions")
    if context.get("keep_doctors"):  prefs.append("Wants to keep current doctors")
    if context.get("wants_dental"):  prefs.append("Dental / vision / hearing benefits important")
    if context.get("prefers_ppo"):   prefs.append("Prefers PPO flexibility")
    prefs_li = "".join(f"<li>{p}</li>" for p in prefs) if prefs else "<li>None stated</li>"

    # ── Checklist (mirrors build_prompt_select_mode logic) ────────────
    checklist = [
        "Confirm your preferred doctors and specialists are in-network",
        "Check that your pharmacy is in the plan's network",
    ]
    if context.get("has_rx"):
        checklist.insert(0,
            "Verify your specific prescriptions are on this plan's formulary "
            "(drug list) — call the plan or check Medicare.gov's drug cost tool")
    if context.get("wants_dental"):
        checklist.append(
            "Confirm what the dental benefit actually covers "
            "(cleanings only vs. fillings and major work)")
    checklist_li = "".join(f"<li>{item}</li>" for item in checklist)

    # ── Plan cards ────────────────────────────────────────────────────
    plan_cards_html = ""
    for i, (_, row) in enumerate(sdf.head(5).iterrows()):
        name      = row.get("Plan Name", f"Plan {i + 1}")
        premium   = _plan_premium(row)
        moop      = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
        stars     = row.get("Overall Star Rating", "—")
        ptype     = row.get("Plan Type", "—")
        deduct    = row.get("Annual Part D Deductible Amount", "—")
        insurer   = row.get("Organization Marketing Name", "—")
        why       = (gemma_whys[i] if i < len(gemma_whys) and gemma_whys[i]
                     else _why_recommended(row, i + 1, st.session_state.sort_key))
        bg        = "#f7f9fc" if i % 2 == 0 else "#ffffff"

        plan_cards_html += f"""
        <div style="background:{bg};border:1px solid #dde4ee;border-radius:8px;
                    padding:16px 20px;margin-bottom:14px;">
          <div style="font-size:1.05rem;font-weight:700;color:#003366;margin-bottom:10px;">
            #{i + 1} &nbsp; {name}
          </div>
          <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">
            <tr>
              <td style="padding:3px 10px;color:#555;width:22%;">Insurer</td>
              <td style="padding:3px 10px;font-weight:600;width:28%;">{insurer}</td>
              <td style="padding:3px 10px;color:#555;width:22%;">Plan type</td>
              <td style="padding:3px 10px;font-weight:600;">{ptype}</td>
            </tr>
            <tr>
              <td style="padding:3px 10px;color:#555;">Monthly premium</td>
              <td style="padding:3px 10px;font-weight:600;color:#0055aa;">{premium}</td>
              <td style="padding:3px 10px;color:#555;">Star rating</td>
              <td style="padding:3px 10px;font-weight:600;">{stars}</td>
            </tr>
            <tr>
              <td style="padding:3px 10px;color:#555;">Max out-of-pocket</td>
              <td style="padding:3px 10px;font-weight:600;">{moop}</td>
              <td style="padding:3px 10px;color:#555;">Drug deductible</td>
              <td style="padding:3px 10px;font-weight:600;">{deduct}</td>
            </tr>
          </table>
          <div style="margin-top:10px;padding:8px 12px;background:#eef4ff;
                      border-left:3px solid #0055aa;border-radius:4px;
                      font-size:0.85rem;color:#1a1a1a;">
            <strong>Why it fits you:</strong> {why}
          </div>
        </div>"""

    today = date.today().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Medicare Plan Summary — MediGuide</title>
  <style>
    body  {{ font-family: Georgia, serif; font-size: 16px; color: #1a1a1a;
             max-width: 820px; margin: 0 auto; padding: 32px 24px; }}
    h1   {{ color: #003366; font-size: 1.7rem; margin-bottom: 2px; }}
    h2   {{ color: #003366; font-size: 1.1rem; margin: 22px 0 10px;
             border-bottom: 2px solid #003366; padding-bottom: 4px; }}
    .meta {{ font-size: 0.82rem; color: #666; margin-bottom: 22px; }}
    .profile-grid {{ display:grid; grid-template-columns:160px 1fr;
                     gap:6px 0; background:#f4f7fb; border-radius:8px;
                     padding:14px 18px; font-size:0.9rem; }}
    .label {{ color:#555; }} .value {{ font-weight:600; }}
    .sort-note {{ background:#e8f0fb; border-left:4px solid #003366;
                  border-radius:6px; padding:10px 14px; font-size:0.9rem;
                  margin-bottom:16px; }}
    .checklist {{ background:#fff8e6; border:1px solid #f0d080;
                  border-radius:8px; padding:14px 20px; }}
    .checklist li {{ margin-bottom:6px; font-size:0.9rem; }}
    .footer {{ margin-top:30px; font-size:0.76rem; color:#888;
               border-top:1px solid #dde4ee; padding-top:12px; }}
    @media print {{ body {{ padding:16px; }} }}
  </style>
</head>
<body>

<h1>🧓 My Medicare Plan Summary</h1>
<div class="meta">
  Generated by MediGuide on {today} &nbsp;·&nbsp;
  Powered by Gemma 4 AI &nbsp;·&nbsp;
  Data: CMS 2026 Landscape
</div>

<h2>Your Profile</h2>
<div class="profile-grid">
  <span class="label">Location</span>      <span class="value">{loc}</span>
  <span class="label">Coverage type</span> <span class="value">{track}</span>
  <span class="label">Monthly budget</span><span class="value">{budget}</span>
  <span class="label">Preferences</span>
  <span class="value"><ul style="margin:0;padding-left:16px;">{prefs_li}</ul></span>
</div>

<h2>Why These Plans Appear First</h2>
<div class="sort-note">{sort_reasoning}</div>

<h2>Your Top 5 Plans</h2>
{plan_cards_html}

<h2>⚠ Before You Enroll — Verify These Directly with the Plan</h2>
<div class="checklist"><ol>{checklist_li}</ol></div>

<div class="footer">
  This summary was generated by MediGuide, a free offline Medicare plan advisor
  powered by Gemma 4 AI running locally on your computer. Plan data is from the
  CMS 2026 Medicare Landscape file. This is not medical or legal advice. Always
  verify details directly with the plan before enrolling.
  Visit <strong>medicare.gov</strong> for official enrollment information.
</div>

</body>
</html>"""


def trigger_select_mode() -> None:
    """
    Called when intake_step reaches 5.
    Filters plans, auto-sorts, calls Gemma for the initial top-5 analysis,
    then switches the screen to SELECT.
    """
    session = st.session_state.session
    lookup  = get_lookup()
    profile = session.state["profile"]
    context = session.state["context"]

    filtered_df, decision = lookup.get_plans_filtered(profile["zip"], profile)
    sort_key, reasoning   = derive_sort_key(profile, context)
    sorted_df, sort_label = sort_plans(filtered_df, sort_key, profile)

    lang = st.session_state.get("language", "English")
    prompt = build_prompt_select_mode(
        user_question  = "Please introduce and analyze my top plan options.",
        plans          = sorted_df.head(5),
        state          = session.state,
        sort_label     = sort_label,
        sort_reasoning = reasoning,
        language       = lang,
    )

    _imode = st.session_state.get("inference_mode", "cloud")
    system_part, user_part = split_system_and_user(prompt)
    st.info(_t("gemma_analyzing"))
    with st.spinner(_t("gemma_spinner")):
        analysis = call_ollama(system_part, user_part, [], mode=_imode)
    session.close_turn(analysis)

    # Short secondary calls: translate filter log and sort reasoning if needed.
    filter_summary_raw = decision.user_summary()
    with st.spinner(_t("preparing_results")):
        filter_expl_prompt = build_prompt_filter_explanation(filter_summary_raw, language=lang)
        filter_explanation = call_ollama("", filter_expl_prompt, [], mode=_imode)
        if lang != "English" and reasoning:
            sort_reasoning_prompt = build_prompt_sort_reasoning(reasoning, language=lang)
            reasoning = call_ollama("", sort_reasoning_prompt, [], mode=_imode)

    st.session_state.filtered_df      = filtered_df
    st.session_state.sorted_df        = sorted_df
    st.session_state.sort_key         = sort_key
    st.session_state.sort_label       = sort_label
    st.session_state.sort_reasoning   = reasoning
    st.session_state.filter_summary   = filter_summary_raw
    st.session_state.filter_explanation = filter_explanation
    clean_analysis = _strip_why_lines(analysis)
    st.session_state.select_analysis = analysis          # keep WHY lines for card parsing
    st.session_state.chat_history.append({"role": "assistant", "content": clean_analysis})
    st.session_state.screen          = "SELECT"
    _save_session_to_disk()
    st.rerun()


# ======================================================================== #
#  SELECT screen                                                            #
# ======================================================================== #

def render_sort_controls() -> None:
    """Row of 6 sort-override buttons. Active key highlighted as primary."""
    st.markdown(_t("resort_label"))
    cols = st.columns(len(SORT_BUTTON_LABELS))
    for col, (key, _) in zip(cols, SORT_BUTTON_LABELS.items()):
        with col:
            is_active = (key == st.session_state.sort_key)
            if st.button(
                _t(f"sort_{key}"),
                key=f"sort_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    new_df, new_label = sort_plans(
                        st.session_state.filtered_df,
                        key,
                        st.session_state.session.state["profile"],
                    )
                    st.session_state.sorted_df      = new_df
                    st.session_state.sort_key       = key
                    st.session_state.sort_label     = new_label
                    st.session_state.sort_reasoning = f"You selected: {new_label}"
                    st.rerun()


def _plan_premium(row: "pd.Series") -> str:
    """Return the best available premium string for a plan row.
    PDP standalone plans have N/A in the consolidated column — fall back to
    'Part D Total Premium'."""
    consolidated = row.get("Monthly Consolidated Premium (Part C + D)", "")
    if str(consolidated).strip().lower() in ("", "nan", "n/a", "not applicable"):
        pdp = row.get("Part D Total Premium", "")
        if str(pdp).strip().lower() not in ("", "nan", "n/a", "not applicable"):
            return str(pdp)
    return str(consolidated) if str(consolidated).strip() else "—"


def _is_pdp(row: "pd.Series") -> bool:
    plan_type = str(row.get("Plan Type", "")).upper()
    cat       = str(row.get("Contract Category Type", "")).upper()
    return "PDP" in plan_type or cat.startswith("PDP")


_PDP_MOOP_NOTE  = (
    "Part D drug plans do not have a MOOP — they cover only prescriptions, "
    "not medical or hospital costs. MOOP applies to Medicare Advantage plans only."
)
_PDP_STARS_NOTE = (
    "CMS star ratings are not published for Part D drug-only plans in this dataset."
)


def _parse_plan_whys(analysis: str, n: int = 5) -> list[str]:
    """
    Extract WHY_1: … WHY_N: … lines appended by Gemma at the end of select_analysis.
    Returns a list of n strings (empty string = not found, caller falls back).
    """
    if not analysis:
        return [""] * n
    result = [""] * n
    for m in re.finditer(r'WHY_(\d):[ \t]*(.+)', analysis):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            result[idx] = m.group(2).strip()
    return result


def _strip_why_lines(analysis: str) -> str:
    """Remove the WHY_N: marker lines from Gemma's analysis before display."""
    return re.sub(r'\nWHY_\d:[^\n]*', '', analysis).strip()


def _extract_sources(text: str) -> "str | None":
    """Extract the SOURCES: line appended by Gemma when RAG passages were provided.

    Returns a page string like 'p.42, p.67', or None if absent or 'none'.
    """
    m = re.search(r'\nSOURCES:\s*(.+)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return None if val.lower() == "none" else val
    return None


def _strip_sources(text: str) -> str:
    """Remove the SOURCES: line from Gemma's response before storing or displaying."""
    return re.sub(r'\nSOURCES:[^\n]*', '', text, flags=re.IGNORECASE).strip()


def render_top5_cards(df: pd.DataFrame, sort_key: str | None = None) -> None:
    """Render top 5 plans as foldable cards with per-plan 'why' from Gemma. First card open."""
    gemma_whys = _parse_plan_whys(st.session_state.get("select_analysis", ""))
    top5 = df.head(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        plan_name = row.get("Plan Name", f"Plan {i + 1}")
        premium   = _plan_premium(row)
        pdp       = _is_pdp(row)
        with st.expander(
            f"#{i + 1}  {plan_name}  —  {premium}/month",
            expanded=(i == 0),
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(_t("card_monthly_premium"), premium)
                moop_val = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
                st.metric(_t("card_moop"), moop_val)
                if pdp and str(moop_val).strip().lower() in ("not applicable", "n/a", ""):
                    st.caption(f"ℹ️ {_PDP_MOOP_NOTE}")
            with col_b:
                st.metric(_t("card_plan_type"), row.get("Plan Type", "—"))
                stars_val = row.get("Overall Star Rating", "—")
                st.metric(_t("card_star_rating"), stars_val)
                if str(stars_val).strip().lower() in ("not applicable", "n/a", ""):
                    st.caption(f"ℹ️ {_PDP_STARS_NOTE}")

            # Extra detail line
            extras = []
            org = row.get("Organization Marketing Name", "")
            if org:
                extras.append(f"**{_t('card_offered_by')}** {org}")
            drug_ded = row.get("Annual Part D Deductible Amount", "")
            if drug_ded and drug_ded not in ("", "nan", "N/A"):
                extras.append(f"**{_t('card_drug_deductible')}** {drug_ded}")
            snp = row.get("SNP Type", "")
            if snp and snp.upper() not in ("", "NAN", "N/A", "NOT APPLICABLE"):
                extras.append(f"**{_t('card_snp')}** {snp}")
            sanctioned = row.get("Sanctioned Plan", "")
            if sanctioned and "YES" in str(sanctioned).upper():
                extras.append("⚠️ **CMS-sanctioned plan**")
            if extras:
                st.markdown("  ·  ".join(extras))

            # Why it's near the top of your list
            why = gemma_whys[i] if i < len(gemma_whys) and gemma_whys[i] else \
                  _why_recommended(row, i + 1, sort_key)
            if why:
                st.markdown(
                    f"<div style='margin-top:10px; padding:10px 14px; "
                    f"background:#f0f4fa; border-left:4px solid #003366; "
                    f"border-radius:6px; font-size:0.92rem; color:#1a1a1a;'>"
                    f"<strong>{_t('why_near_top')}</strong> {why}</div>",
                    unsafe_allow_html=True,
                )


def _why_recommended(row: "pd.Series", rank: int, sort_key: str | None) -> str:
    """Compute a short 'why this plan' label based on sort key and plan data."""
    prem  = _plan_premium(row)
    pdp   = _is_pdp(row)
    parts = []

    key = sort_key or "lowest_premium"

    if key == "lowest_premium":
        if prem.replace("$", "").replace(".00", "").strip() == "0":
            parts.append("$0 monthly premium")
        else:
            parts.append(f"Premium: {prem}/mo")

    elif key == "total_cost":
        parts.append("Lowest est. annual cost (premium + MOOP)")

    elif key == "star_rating":
        stars = row.get("Overall Star Rating", "")
        if stars and str(stars).lower() not in ("not applicable", "n/a", "nan", ""):
            parts.append(f"{stars}-star CMS rating")
        else:
            parts.append("Sorted by star rating")

    elif key == "lowest_moop":
        moop = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "")
        if not pdp and moop and str(moop).lower() not in ("not applicable", "n/a"):
            parts.append(f"MOOP: {moop}")
        elif pdp:
            parts.append("PDP — no MOOP applies")

    elif key == "lowest_deductible":
        ded = row.get("Annual Part D Deductible Amount", "")
        if ded and str(ded).lower() not in ("not applicable", "n/a", "nan"):
            if ded.replace("$", "").replace(".00", "").strip() == "0":
                parts.append("$0 drug deductible")
            else:
                parts.append(f"Drug deductible: {ded}")

    elif key == "ppo_first":
        ptype = row.get("Plan Type", "")
        parts.append(f"Plan type: {ptype}")

    if rank == 1:
        parts.insert(0, "Top pick")

    return " · ".join(parts) if parts else f"#{rank} overall match"


def render_plan_table(df: pd.DataFrame, sort_key: str | None) -> None:
    """Render all matching plans as a styled table with a Why Recommended column."""
    rows = []
    has_pdp = False
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        pdp  = _is_pdp(row)
        if pdp:
            has_pdp = True
        moop  = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
        stars = row.get("Overall Star Rating", "—")
        rows.append({
            "#":               rank,
            "Plan Name":       row.get("Plan Name", "—"),
            "Type":            row.get("Plan Type", "—"),
            "Premium":         _plan_premium(row),
            "MOOP ¹" if pdp else "MOOP":
                               "N/A ¹" if (pdp and str(moop).lower() in ("not applicable","n/a","")) else moop,
            "Stars ²" if str(stars).lower() in ("not applicable","n/a","nan","") else "Stars":
                               "N/A ²" if str(stars).lower() in ("not applicable","n/a","nan","") else stars,
            "Drug Deductible": row.get("Annual Part D Deductible Amount", "—"),
            "Insurer":         row.get("Organization Marketing Name", "—"),
            "Why Recommended": _why_recommended(row, rank, sort_key),
        })
    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=420)

    if has_pdp:
        st.caption(
            "¹ **MOOP (Maximum Out-of-Pocket)** shows N/A for Part D drug plans. "
            "PDP plans cover prescriptions only — they have no MOOP because they don't cover "
            "medical or hospital costs. MOOP limits apply only to Medicare Advantage (Part C) plans.  \n"
            "² **Star Rating** is not published by CMS for standalone Part D drug plans in this dataset."
        )


def render_select() -> None:
    session = st.session_state.session
    profile = session.state["profile"]

    # ── Header ──────────────────────────────────────────────────────────
    st.title(_t("select_title"))

    loc = profile.get("zip", "")
    if profile.get("county") and profile.get("state"):
        loc += f" · {profile['county']}, {profile['state']}"
    track_map   = {"MA_D": "Medicare Advantage", "PDP": "Part D Drug Plan"}
    track_label = track_map.get(profile.get("track", ""), profile.get("track", ""))
    st.markdown(
        f"<p style='font-size:1.05rem; color:#555; margin-top:-0.5rem;'>"
        f"Coverage type: <strong>{track_label}</strong> &nbsp;·&nbsp; "
        f"Location: <strong>{loc}</strong></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    sdf = st.session_state.sorted_df
    no_plans = sdf is None or sdf.empty

    # ── Two-tab layout ──────────────────────────────────────────────────
    tab1, tab2 = st.tabs([_t("tab_recommendations"), _t("tab_explore")])

    # ══════════════════════════════════════════════════════════════════ #
    #  Tab 1 — Recommendations                                          #
    # ══════════════════════════════════════════════════════════════════ #
    with tab1:
        # ── Filter explanation (Gemma plain-English) + raw detail ─────────
        if st.session_state.filter_summary:
            with st.expander(_t("filter_expander"), expanded=False):
                if st.session_state.filter_explanation:
                    st.markdown(st.session_state.filter_explanation)
                    with st.expander(_t("filter_details_expander"), expanded=False):
                        st.text(st.session_state.filter_summary)
                else:
                    safe_md(st.session_state.filter_summary)

        if st.session_state.sort_reasoning:
            st.info(f"{_t('why_first')} {st.session_state.sort_reasoning}")

        # ── Download summary button ───────────────────────────────────────
        if not no_plans:
            html_bytes = _generate_summary_html().encode("utf-8")
            st.download_button(
                label=_t("download_button"),
                data=html_bytes,
                file_name="medicareguide_plan_summary.html",
                mime="text/html",
                use_container_width=True,
                help=_t("download_help"),
            )

        if no_plans:
            st.warning(_t("no_plans_warning"))
        else:
            total = len(sdf)
            st.subheader(_t("top5_subheader", n=min(5, total), total=total))
            render_top5_cards(sdf, st.session_state.sort_key)

        # Follow-up chat — new Q&A only, no initial Gemma table
        st.divider()
        follow_ups = [m for m in st.session_state.chat_history
                      if m != st.session_state.chat_history[0]]  # skip initial analysis
        if len(follow_ups) > 1:
            for msg in follow_ups[1:]:   # skip the first assistant analysis message
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        safe_md(msg["content"])
                    else:
                        st.markdown(msg["content"])

        # Voice input — fires immediately when the user stops recording
        voiced = render_voice_input("select_mic")
        if voiced:
            st.session_state._select_voice_pending = voiced

        user_input = st.chat_input(_t("select_chat_placeholder"))

        # Consume a pending voice transcription (set above on the previous rerun)
        if not user_input and st.session_state.get("_select_voice_pending"):
            user_input = st.session_state.pop("_select_voice_pending")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            answer = handle_select_turn(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                safe_md(answer)
            maybe_play_audio(answer)

    # ══════════════════════════════════════════════════════════════════ #
    #  Tab 2 — Explore All Plans                                        #
    # ══════════════════════════════════════════════════════════════════ #
    with tab2:
        if no_plans:
            st.warning("No plans to display.")
        else:
            total = len(sdf)
            st.subheader(_t("all_plans_subheader", total=total))
            st.caption(_t("explore_caption"))
            render_sort_controls()
            st.write("")
            render_plan_table(sdf, st.session_state.sort_key)

    st.divider()
    col_back, _, col_reset = st.columns([1, 4, 1])
    with col_back:
        if st.button("←", key="select_back", help="Back to setup"):
            st.session_state.screen = "INTAKE"
            st.session_state.intake_step = 4
            session.state["intake_step"] = 4
            session.state["mode"] = "WELCOME"
            st.rerun()
    with col_reset:
        if st.button(_t("start_over"), key="select_reset"):
            _clear_saved_session()
            st.session_state.clear()
            st.rerun()


# ======================================================================== #
#  Main routing                                                             #
# ======================================================================== #

init_session_state()

_screen = st.session_state.screen
if   _screen == "WELCOME": render_welcome()
elif _screen == "INTAKE":  render_intake()
elif _screen == "SELECT":  render_select()
else:
    st.error(f"Unknown screen state: {_screen!r}. Resetting.")
    st.session_state.clear()
    st.rerun()
