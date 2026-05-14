import streamlit as st
from ui.components import (
    _t, render_language_selector, render_chat_history,
    render_voice_input, maybe_play_audio, safe_md
)
from ui.utils import _extract_sources, _strip_sources
from ui.backend import get_rag_index
from core.ollama import call_ollama
from core.inference import build_prompt_welcome_mode

try:
    from core.tts import generate_audio_bytes, TTS_AVAILABLE
except ImportError:
    TTS_AVAILABLE = False
    def generate_audio_bytes(*args, **kwargs): return None

try:
    from core.rag import retrieve, format_context, RAG_AVAILABLE
except ImportError:
    RAG_AVAILABLE = False
    def retrieve(*args, **kwargs): return []
    def format_context(*args, **kwargs): return ""

def ask_welcome(user_input: str) -> str:
    """Ask a WELCOME-mode question. Returns Gemma's answer string."""
    session = st.session_state.session
    saved_step = session.state["intake_step"]
    saved_mode = session.state["mode"]

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

    session.state["intake_step"] = saved_step
    session.state["mode"]        = saved_mode
    return answer

def render_welcome() -> None:
    _, top_r = st.columns([3, 2])
    with top_r:
        render_language_selector("welcome")

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

    st.divider()
    
    # ── Chat & Voice Section ────────────────────────────────────────────────
    st.markdown('<div class="chat-section-compact">', unsafe_allow_html=True)
    
    st.markdown(f"#### {_t('welcome_chat_heading')}")
    
    if TTS_AVAILABLE:
        v_label = _t("voice_mode_on") if st.session_state.audio_enabled else _t("voice_mode_off")
        if st.button(v_label, key="voice_mode_toggle", help=_t("voice_mode_hint"), use_container_width=True):
            st.session_state.audio_enabled = not st.session_state.audio_enabled
            if st.session_state.audio_enabled:
                from ui.backend import _warm_tts
                _warm_tts()
            st.rerun()

    if st.session_state.chat_history:
        render_chat_history()
    
    # Microphone input
    voiced = None
    if st.session_state.audio_enabled and TTS_AVAILABLE:
        voiced = render_voice_input("welcome_mic")

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

    st.markdown("""
    <style>
    div[data-testid="stForm"] [data-baseweb="input"] {
        min-height: 100px !important;
        align-items: center !important;
    }
    div[data-testid="stForm"] [data-baseweb="input"] input {
        font-size: 1.4rem !important;
    }
    div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        min-height: 100px !important;
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
        if st.session_state.audio_enabled and TTS_AVAILABLE:
            st.session_state._pending_tts_text = clean_answer
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div class="how-it-works-title">{_t("how_it_works")}</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""<div class="step-card step-card-1">
            <div class="step-num">1</div>
            <div class="step-title">{_t("step1_title")}</div>
            <div class="step-desc">{_t("step1_desc")}</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""<div class="step-card step-card-2">
            <div class="step-num">2</div>
            <div class="step-title">{_t("step2_title")}</div>
            <div class="step-desc">{_t("step2_desc")}</div>
        </div>""", unsafe_allow_html=True)
    with h3:
        st.markdown(f"""<div class="step-card step-card-3">
            <div class="step-num">3</div>
            <div class="step-title">{_t("step3_title")}</div>
            <div class="step-desc">{_t("step3_desc")}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    with st.container(border=True):
        _is_cloud = st.session_state.get("inference_mode", "cloud") == "cloud"
        infer_label = "⚡ Fast mode (Ollama Cloud)" if _is_cloud else "🔒 Private mode (Fully local)"
        infer_help  = (
            "Use Ollama Cloud for fast responses (zero retention). "
            "Turn off to run fully locally (private, but slower)."
        )
        
        st.markdown('<div class="inference-toggle-box">', unsafe_allow_html=True)
        new_is_cloud = st.toggle(infer_label, value=_is_cloud, help=infer_help, key="infer_toggle_welcome")
        if new_is_cloud != _is_cloud:
            st.session_state.inference_mode = "cloud" if new_is_cloud else "local"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Inference toggle logic ends here.
