"""
MediCareGuide Speech-to-Text (STT) Module
=======================================
Captures microphone input and transcribes it via faster-whisper (OpenAI Whisper).

  record_audio()    — capture mic until user presses Enter (15s cap)
  transcribe_audio() — transcribe WAV bytes with faster-whisper
  voice_input()     — full flow: record → transcribe → confirm, with retry
  smart_input()     — drop-in for input(); adds [or 'v' for voice] hint

Degrades gracefully: if sounddevice is not installed, VOICE_AVAILABLE is False
and smart_input() behaves like a plain input() call.

Dependencies:
    pip3 install sounddevice faster-whisper

Usage:
    from medicareguide_stt import smart_input, VOICE_AVAILABLE

    zipcode = smart_input("Enter ZIP code").strip()
"""

import io
import sys
import time
import wave
from pathlib import Path

try:
    import sounddevice as sd
    import numpy as np
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as _WhisperModel  # noqa: F401
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

AUDIO_SAMPLE_RATE = 16000       # 16kHz mono — standard for speech
AUDIO_DURATION_SECONDS = 15     # max cap; user stops early by pressing Enter

# Domain-specific prompt fed to Whisper as prior context.
# Biases recognition toward Medicare acronyms so they are kept as English
# letters even when the surrounding speech is Chinese or Spanish.
_WHISPER_PROMPT = (
    "Medicare PPO HMO PDP SNP MOOP LIS D-SNP C-SNP "
    "Medicare Advantage Part D deductible premium copay"
)



def record_audio(max_duration: int = AUDIO_DURATION_SECONDS,
                 sample_rate: int = AUDIO_SAMPLE_RATE) -> "bytes | None":
    """
    Record from the default microphone until the user presses Enter,
    capped at max_duration seconds.

    Returns raw WAV bytes (with header), or None on failure.
    """
    if not VOICE_AVAILABLE:
        return None
    try:
        import threading
        import select

        chunks = []
        stop_event = threading.Event()

        def _capture():
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                while not stop_event.is_set():
                    chunk, _ = stream.read(sample_rate // 10)  # 100ms chunks
                    chunks.append(chunk.copy())

        thread = threading.Thread(target=_capture, daemon=True)
        thread.start()

        deadline = time.time() + max_duration
        try:
            while time.time() < deadline:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    sys.stdin.readline()  # consume the Enter
                    break
        except Exception:
            time.sleep(max_duration)    # fallback on platforms without select

        stop_event.set()
        thread.join(timeout=1)

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()
    except Exception as e:
        print(f"[Audio capture error] {e}")
        return None


def transcribe_audio(wav_bytes: bytes, ui_language: str = "English") -> "str | None":
    """
    Transcribe WAV bytes using faster-whisper (base model, runs on CPU).

    Audio from record_audio() is already 16kHz mono WAV — written to a temp
    file so faster-whisper can read it, then cleaned up.
    Language is auto-detected by Whisper (supports English, Chinese, Spanish).
    Returns the transcribed text string, or None if transcription failed.
    """
    if not wav_bytes:
        return None
    import os, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name
    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(wav_path, initial_prompt=_WHISPER_PROMPT)
        text = " ".join(seg.text for seg in segments).strip()
        return text if len(text) >= 2 else None
    except Exception as e:
        print(f"[Transcription error] {e}")
        return None
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


def voice_input(prompt: str, max_retries: int = 2) -> "str | None":
    """
    Full voice input flow: record → transcribe → show result → user confirms.

    If the transcription confidence is low (silent/noisy audio), asks the user
    to repeat up to max_retries times before falling back to text input.

    Returns the confirmed transcription string, or None if unavailable,
    transcription failed, or the user rejected the result.
    """
    if not VOICE_AVAILABLE:
        print("Voice input unavailable. Run: pip3 install sounddevice")
        return None

    for attempt in range(max_retries):
        print("Recording... speak now, then press Enter when done.")
        wav_bytes = record_audio()
        if wav_bytes is None:
            return None

        print("\nTranscribing...")
        text = transcribe_audio(wav_bytes)

        if text is None:
            if attempt < max_retries - 1:
                print("I didn't quite catch that — could you repeat it?")
                continue
            print("[Could not understand audio — type your answer instead]")
            return None

        # Collapse spaces in digit-only results (e.g. "240 60" → "24060")
        if text.replace(" ", "").isdigit():
            text = text.replace(" ", "")

        print(f'\n  Transcription: "{text}"')
        confirm = input("Use this? [Y/n]: ").strip().lower()
        if confirm in ("", "y"):
            return text
        # User rejected — offer one more try if retries remain
        if attempt < max_retries - 1:
            print("Let's try again.")
        else:
            return None

    return None


_whisper_model = None


def _get_whisper_model():
    """Lazy-load faster-whisper base model (downloaded once, ~150 MB)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    import os
    # ctranslate2 and numpy both ship an OpenMP runtime; allow both to coexist.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from faster_whisper import WhisperModel
    print("[STT] Loading Whisper base model…")
    _whisper_model = WhisperModel("small", device="cpu", compute_type="float32")
    return _whisper_model


def transcribe_streamlit_audio(audio_bytes: bytes, ui_language: str = "English") -> "str | None":
    """
    Transcribe audio recorded by st.audio_input() in the Streamlit UI.

    The browser records in WebM/Opus; ffmpeg converts it to 16kHz mono WAV,
    then faster-whisper transcribes it.
    Language is auto-detected by Whisper (supports English, Chinese, Spanish).
    ui_language is accepted for API compatibility but not used for STT.

    Returns the transcribed text string, or None if transcription failed.
    """
    if not audio_bytes:
        return None
    import os, subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        webm_path = f.name
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        # Convert browser WebM/Opus to 16kHz mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True,
        )
        model = _get_whisper_model()
        segments, _ = model.transcribe(wav_path, initial_prompt=_WHISPER_PROMPT)
        text = " ".join(seg.text for seg in segments).strip()
        return text if len(text) >= 2 else None
    except Exception as e:
        print(f"[STT error] {e}")
        return None
    finally:
        for p in (webm_path, wav_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def smart_input(prompt: str) -> str:
    """
    Drop-in replacement for input().

    Shows a voice hint when sounddevice is available. If the user types
    'v' or 'voice', triggers voice_input() and returns the transcription.
    Falls back to a plain text prompt if voice fails or is rejected.
    """
    hint = " [or 'v' for voice]" if VOICE_AVAILABLE else ""
    raw = input(f"{prompt}{hint}: ").strip()
    if raw.lower() in ("v", "voice"):
        result = voice_input(prompt)
        if result is not None:
            return result
        return input(f"{prompt}: ").strip()
    return raw
