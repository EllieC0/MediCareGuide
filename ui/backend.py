import streamlit as st
from pathlib import Path
from core.lookup import MediCareGuideLookup

CSV_PATH = Path(__file__).parent.parent / "data" / "CY2026_Landscape_202603.csv"

@st.cache_resource(show_spinner="Loading Medicare plan database…")
def get_lookup() -> MediCareGuideLookup:
    return MediCareGuideLookup(CSV_PATH)

@st.cache_resource(show_spinner="Loading voice model… (first time only)")
def _warm_tts() -> None:
    """Pre-load the Kokoro ONNX model at startup so audio plays instantly."""
    try:
        from core.tts import TTS_AVAILABLE
        if TTS_AVAILABLE:
            from core.tts import _get_kokoro
            _get_kokoro()
    except Exception:
        pass

@st.cache_resource(show_spinner="Loading Medicare handbook index… (first time only)")
def get_rag_index():
    """Build or load the FAISS index from the CMS Medicare & You 2026 PDF."""
    try:
        from core.rag import build_or_load_index, RAG_AVAILABLE
        if RAG_AVAILABLE:
            return build_or_load_index()
    except ImportError:
        pass
    return None
