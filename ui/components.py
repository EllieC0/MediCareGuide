import hashlib
import streamlit as st
from ui.international import t as _ti18n
from ui.state import _save_session_to_disk

try:
    from core.tts import generate_audio_bytes, TTS_AVAILABLE
except ImportError:
    TTS_AVAILABLE = False
    def generate_audio_bytes(text, voice="af_heart", speed=1.0, ui_language="English"):
        return None

try:
    from core.stt import transcribe_streamlit_audio, WHISPER_AVAILABLE as STT_AVAILABLE
except ImportError:
    STT_AVAILABLE = False
    def transcribe_streamlit_audio(audio_bytes, ui_language="English"):
        return None

def _t(key: str, **kwargs) -> str:
    """Shorthand: translate *key* using the current session language."""
    lang = st.session_state.get("language", "English")
    return _ti18n(key, lang, **kwargs)

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
    """
    if not STT_AVAILABLE:
        return None

    st.caption(_t("voice_hint"))
    gen_key = f"_mic_gen_{widget_key}"
    actual_key = f"{widget_key}_{st.session_state.get(gen_key, 0)}"
    audio_val = st.audio_input(
        "🎤 Or speak your question",
        key=actual_key,
        label_visibility="collapsed",
    )
    if audio_val is None:
        return None

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
        st.session_state[gen_key] = st.session_state.get(gen_key, 0) + 1
        return text

    st.warning(_t("stt_failed"))
    return None
