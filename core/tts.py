"""
MediCareGuide Text-to-Speech Module
=================================
Converts LLM answer text to spoken audio using Kokoro ONNX TTS.
Apache 2.0 licensed, fully offline after first model download (~350MB).

Requires Python 3.10+ and:
    pip install kokoro-onnx onnxruntime sounddevice soundfile

Model files are downloaded automatically from HuggingFace on first use
and cached in ~/.cache/core.tts/.

Usage:
    from core.tts import speak, TTS_AVAILABLE
    speak("Hello, here are your Medicare plan options.")

Changes from original _clean_text():
    1. Markdown tables converted to spoken sentences before any other
       processing — pipe characters and separator rows are handled
       structurally, not stripped blindly.
    2. Slash-unit suffixes ($0.00/month, $50/year) converted to spoken
       form ("zero dollars per month", "50 dollars per year") BEFORE
       the dollar regex runs, so the slash is never left in the output.
    3. Bullet points now get a period+space replacement (not just a space)
       so Kokoro pauses naturally between list items.
    4. Processing order made explicit and numbered with no duplicate labels.
"""

import io
import re
import time
from pathlib import Path

import requests

try:
    from kokoro_onnx import Kokoro
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Streamlit might not be available if called from CLI tests
try:
    import streamlit as st
except ImportError:
    st = None

_CACHE_DIR = Path.home() / ".cache" / "core.tts"
_MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

_kokoro = None   # lazy-loaded on first call to speak()

# Chinese and Spanish use the macOS `say` command (kokoro-onnx only supports English).
# Maps UI language → macOS say voice name for non-English languages.
_SAY_VOICES = {
    "中文":    "Lili (Premium)",          # zh_CN Mandarin — premium neural voice
    "Español": "Sandy (Spanish (Mexico))", # es_MX Spanish  — premium neural voice
}


_CHINESE_SAY_VOICES = {"Lili (Premium)", "Tingting", "Meijia", "Sinji"}


def _normalise_for_say(text: str, voice: str) -> str:
    """
    Fix punctuation before passing text to macOS `say`.

    Chinese `say`: reads Latin periods as "点" (dot) instead of pausing.
      - Replace mid-sentence ". " with Chinese comma "，" so speech flows.
      - Strip any trailing period.

    Spanish `say`: handles "." as a pause correctly, but a trailing period
      can still be vocalised — strip it.
    """
    if voice in _CHINESE_SAY_VOICES:
        text = re.sub(r'\.\s+', '，', text)   # mid-sentence period → Chinese comma
        text = text.rstrip('. ')
    else:
        text = text.rstrip('. ')              # strip trailing period for all say voices
    return text.strip()


def _generate_audio_bytes_say(text: str, voice: str) -> "bytes | None":
    """
    Generate WAV bytes using the macOS `say` command.
    Used for Chinese and Spanish where kokoro-onnx lacks phonemizer support.
    """
    text = _normalise_for_say(text, voice)
    import os, subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        aiff_path = f.name
    wav_path = aiff_path.replace(".aiff", ".wav")
    try:
        subprocess.run(["say", "-v", voice, text, "-o", aiff_path], check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, "-ar", "24000", "-ac", "1", "-f", "wav", wav_path],
            check=True, capture_output=True,
        )
        with open(wav_path, "rb") as wf:
            return wf.read()
    except Exception as e:
        print(f"[TTS say error] {e}")
        return None
    finally:
        for p in (aiff_path, wav_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ====================================================================== #
#  Model download + init                                                  #
# ====================================================================== #

def _download_if_missing(url: str, dest: Path) -> None:
    """Download a file with a Streamlit progress indicator and ETA if it doesn't exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = dest.name
    
    print(f"[TTS] Downloading {filename} (first use only)...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    # Initialize Streamlit progress bar if running in a Streamlit context
    progress_bar = None
    status_text = None
    if st and hasattr(st, "progress"):
        import logging
        logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        logger.setLevel(old_level)
        if ctx is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
    
    downloaded = 0
    start_time = time.time()
    
    tmp_dest = dest.with_suffix(".tmp")
    with open(tmp_dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if progress_bar:
                    done = downloaded / total_size
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    remaining = (total_size - downloaded) / speed if speed > 0 else 0
                    
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    
                    eta_min = int(remaining // 60)
                    eta_sec = int(remaining % 60)
                    eta_str = f"~{eta_min}m {eta_sec}s remaining" if eta_min > 0 else f"~{eta_sec}s remaining"
                    
                    progress_bar.progress(min(done, 1.0))
                    status_text.markdown(
                        f"**Downloading {filename}**: {mb_done:.1f} / {mb_total:.1f} MB) — {eta_str}"
                    )

    tmp_dest.rename(dest)

    if progress_bar:
        progress_bar.empty()
        status_text.empty()
        
    print(f"[TTS] {filename} ready.")


def _get_kokoro() -> "Kokoro":
    """Initialise Kokoro once, downloading model files if needed."""
    global _kokoro
    if _kokoro is None:
        model_path  = _CACHE_DIR / "kokoro-v1.0.onnx"
        voices_path = _CACHE_DIR / "voices-v1.0.bin"
        if _kokoro is None:
            _download_if_missing(_MODEL_URL,  model_path)
            _download_if_missing(_VOICES_URL, voices_path)
            try:
                _kokoro = Kokoro(str(model_path), str(voices_path))
            except Exception as e:
                if "Protobuf parsing failed" in str(e) or "INVALID_PROTOBUF" in str(e):
                    print(f"[TTS] Corrupted model detected. Deleting cache: {model_path}")
                    model_path.unlink(missing_ok=True)
                    voices_path.unlink(missing_ok=True)
                    raise RuntimeError("TTS model was corrupted and has been deleted. Please try again to re-download.") from e
                raise
        return _kokoro


# ====================================================================== #
#  Table → speech conversion                                             #
# ====================================================================== #

# Maps Landscape column header fragments to natural spoken labels.
# Checked case-insensitively against the header cell text.
# Empty string means "use as row identifier, not a label".
_SPOKEN_COL_NAMES: list[tuple[str, str]] = [
    ("plan name",                        ""),           # row identifier
    ("organization marketing name",      "offered by"),
    ("parent organization",              "parent company"),
    ("monthly consolidated premium",     "monthly premium"),
    ("part c premium",                   "Part C premium"),
    ("part d total premium",             "Part D premium"),
    ("part d basic premium",             "Part D basic premium"),
    ("in-network maximum out-of-pocket", "out-of-pocket maximum"),
    ("moop",                             "out-of-pocket maximum"),
    ("annual part d deductible",         "drug deductible"),
    ("overall star rating",              "star rating"),
    ("part c summary star rating",       "Part C star rating"),
    ("part d summary star rating",       "Part D star rating"),
    ("plan type",                        "plan type"),
    ("special needs plan (snp) indicator", "special needs plan"),
    ("snp type",                         "special needs plan type"),
    ("part d coverage indicator",        "includes drug coverage"),
    ("low income subsidy",               "low income subsidy"),
    ("sanctioned plan",                  "sanctioned"),
    ("contract category type",           "coverage category"),
    ("county name",                      "county"),
    ("state territory name",             "state"),
]


def _spoken_col(header: str) -> str:
    """
    Map a table column header to a natural spoken label.
    Returns "" for columns used as the row identifier (plan name).
    Returns title-cased raw header as fallback for unknown columns.
    """
    h = header.strip().lower()
    for fragment, spoken in _SPOKEN_COL_NAMES:
        if fragment in h:
            return spoken
    return header.strip().title()


def _parse_table_row(line: str) -> list[str]:
    """Strip outer pipes, split on |, strip whitespace from each cell."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """True if every non-empty cell contains only dashes, colons, spaces."""
    return all(re.match(r"^[-: ]+$", c) for c in cells if c)


def _strip_markdown_inline(text: str) -> str:
    """Strip bold (**), italic (*), normalise dashes from a cell value."""
    text = re.sub(r'\*+', '', text)
    text = text.replace('\u2011', '-')       # non-breaking hyphen → regular hyphen
    text = re.sub(r'\s*\u2013\s*', ' to ',  text)  # en-dash in ranges → " to "
    text = re.sub(r'\s*\u2014\s*', ', ',    text)  # em-dash → comma pause
    return text.strip()


def _is_feature_value_table(headers: list[str]) -> bool:
    """
    Return True if this is a 2-column feature/value table.

    Gemma uses this layout for per-plan detail:
        | Feature | Value |
        |--------|-------|
        | Monthly premium | $0.00/month |

    The distinguishing signals:
      - Exactly 2 non-empty columns
      - First header is a label-like word (Feature, Metric, Detail,
        Attribute, Item, Category, Field) OR is a generic short word
        that does not look like a plan-data column name.

    This is intentionally conservative — unknown 2-column tables are
    treated as feature/value because that is far more common in Gemma's
    Medicare output than a 2-column comparison table.
    """
    non_empty = [h for h in headers if h.strip()]
    if len(non_empty) != 2:
        return False
    first = non_empty[0].strip().lower()
    # Explicit feature/value header words
    feature_words = {"feature", "metric", "detail", "attribute",
                     "item", "category", "field", "property",
                     "description", "criteria"}
    if first in feature_words:
        return True
    # If first column looks like a data-column name from the Landscape,
    # this is probably a comparison table — not feature/value.
    data_col_fragments = {"plan name", "plan type", "premium",
                          "deductible", "moop", "star", "rating",
                          "snp", "organization", "county"}
    if any(frag in first for frag in data_col_fragments):
        return False
    # Default: treat unknown 2-column tables as feature/value
    return True


def _table_to_speech(table_lines: list[str], heading: str = "") -> str:
    """
    Convert a list of markdown table lines into natural spoken sentences.

    Two layouts are handled:

    Layout A — Feature/value (per-plan detail, 2 columns):
        Gemma produces one table per plan, preceded by a bold heading.
        The heading is already the plan name — the table just lists
        attributes. Each row is spoken as "Feature: value."

        Input heading: "Plan 1 – Aetna Medicare Carilion Health Prime (HMO-POS)"
        Input table:
            | Feature                    | Value        |
            |---------------------------|--------------|
            | Monthly premium            | $0.00/month  |
            | Star rating                | 3.0          |

        Output:
            "Monthly premium: $0.00/month. Star rating: 3.0."

        Note: the heading itself is NOT repeated here — it was already
        added to output_lines by Step 1 of _clean_text() before this
        function is called.

    Layout B — Multi-plan comparison (N columns, first is plan name):
        A single table comparing multiple plans side by side.
        Each row becomes "Plan N: name. col: value. col: value."

        Input:
            | Plan Name        | Premium     | Stars |
            |-----------------|-------------|-------|
            | Humana Gold HMO  | $0.00/month | 4.0   |
            | Aetna PPO        | $23.50/month| 4.0   |

        Output:
            "Plan 1: Humana Gold HMO. Monthly premium: $0.00/month. Star rating: 4.0."
            "Plan 2: Aetna PPO. Monthly premium: $23.50/month. Star rating: 4.0."

    General rules for both layouts:
        - Separator rows (---|---) are skipped.
        - Cells containing N/A, -, —, or empty are skipped silently.
        - Inline markdown (**bold**) is stripped from cell values.
        - Dollar values are left as-is — _clean_text() converts them later.
    """
    if not table_lines:
        return ""

    rows = [_parse_table_row(l) for l in table_lines]
    if not rows:
        return ""

    headers = [_strip_markdown_inline(h) for h in rows[0]]

    # ── Detect layout ─────────────────────────────────────────────────
    if _is_feature_value_table(headers):
        return _feature_value_to_speech(rows)
    else:
        return _comparison_to_speech(rows, headers)


def _feature_value_to_speech(rows: list[list[str]]) -> str:
    """
    Convert a 2-column feature/value table to spoken "feature: value." sentences.

    Each data row → one sentence.
    The feature name (left column) is mapped through _spoken_col() for
    natural labels. The value (right column) is used as-is.
    Skips separator rows and empty/placeholder values.
    """
    sentences = []
    for row in rows[1:]:          # skip header row
        if _is_separator_row(row):
            continue
        if len(row) < 2:
            continue
        feature = _strip_markdown_inline(row[0])
        value   = _strip_markdown_inline(row[1])

        if not feature or not value:
            continue
        if value in ("N/A", "n/a", "-", "—", "nan"):
            continue

        # Map the feature name to a natural spoken label
        spoken_feature = _spoken_col(feature)
        if not spoken_feature:
            # _spoken_col returns "" for "plan name" — unlikely in a feature
            # column, but fall back to the raw feature text title-cased
            spoken_feature = feature.title()

        sentences.append(f"{spoken_feature}: {value}.")

    return " ".join(sentences)


def _comparison_to_speech(rows: list[list[str]], headers: list[str]) -> str:
    """
    Convert a multi-column comparison table to spoken plan sentences.

    Each data row → "Plan N: <name>. <col>: <value>. ..."
    The first column whose _spoken_col() label is "" is used as the
    plan name / row identifier.
    """
    spoken_labels = [_spoken_col(h) for h in headers]
    sentences     = []
    plan_n        = 0

    for row in rows[1:]:
        if _is_separator_row(row):
            continue
        if not any(c for c in row):
            continue

        padded     = row + [""] * max(0, len(headers) - len(row))
        plan_n    += 1
        identifier = None
        parts      = []

        for cell, label in zip(padded, spoken_labels):
            cell = _strip_markdown_inline(cell)
            if not cell or cell in ("N/A", "n/a", "-", "—", "nan"):
                continue
            if label == "" and identifier is None:
                identifier = cell
            else:
                col_label = label if label else headers[spoken_labels.index(label)].strip().title()
                parts.append(f"{col_label}: {cell}")

        row_id   = identifier if identifier else f"Option {plan_n}"
        detail   = ". ".join(parts)
        sentence = f"Plan {plan_n}: {row_id}."
        if detail:
            sentence += f" {detail}."
        sentences.append(sentence)

    return "\n".join(sentences)


# ====================================================================== #
#  Main text cleaning                                                    #
# ====================================================================== #

def _clean_text(text: str) -> str:
    """
    Convert Gemma's markdown output to natural speech-friendly plain text.

    Processing order (order is load-bearing — do not rearrange):

    Step 1  Table conversion
            Must run first while pipe characters still identify table rows.
            Converts markdown tables to "Plan N: name. label: value." sentences.

    Step 2  Slash-unit suffixes
            Converts $0.00/month → $0.00 per month BEFORE the dollar regex
            so the slash is never passed to Kokoro as a literal character.
            Also handles /year, /day, /week.

    Step 3  Dollar amounts
            $0.00 → "zero dollars", $26.50 → "26 dollars and 50 cents".
            Runs after slash-unit fix so "/month" is already gone.

    Step 4  Decimal numbers
            4.5 → "4 point 5". Runs after dollar conversion so $3.50
            is already replaced before this pattern would match it.

    Step 5  Bullet points
            Each bullet gets a period+space prefix so Kokoro pauses
            between items. Original code used a plain space, causing
            consecutive items to be read as one continuous phrase.

    Step 6  Markdown symbols
            Strip **, ##, ` — these have no spoken equivalent.

    Step 7  Whitespace collapse
            Paragraph breaks → sentence pause (". ").
            Remaining newlines → space.
            Multiple spaces → single space.
    """

    # ── Step 1: Table conversion ──────────────────────────────────────
    # Scan line by line. Buffer consecutive table rows (start with "|").
    # Also track the most recent non-empty non-table line as a potential
    # heading — Gemma often places a bold plan name immediately before
    # each per-plan feature/value table. That heading is kept in
    # output_lines as a spoken sentence; the table rows that follow
    # are converted to "feature: value." sentences without repeating
    # the plan name.
    lines        = text.split("\n")
    output_lines = []
    table_buffer = []
    last_heading = ""   # most recent non-empty non-table line

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_buffer.append(line)
        else:
            # Flush any buffered table first
            if table_buffer:
                spoken = _table_to_speech(table_buffer, heading=last_heading)
                if spoken:
                    output_lines.append(spoken)
                table_buffer = []

            output_lines.append(line)

            # Track the last non-empty line as a potential table heading.
            # Strip markdown bold markers and normalise dashes so the
            # heading reads naturally: "Plan 1 – Aetna..." → "Plan 1, Aetna..."
            if stripped:
                heading_clean = re.sub(r'\*+', '', stripped)
                heading_clean = re.sub(r'\s*[–—]\s*', ', ', heading_clean)  # em/en dash → comma
                last_heading  = heading_clean.strip()
                # Also replace the last output_lines entry with the cleaned heading
                # so the spoken heading itself has no bold markers or raw dashes.
                output_lines[-1] = heading_clean

    # Flush a table that ends at the last line of the text
    if table_buffer:
        spoken = _table_to_speech(table_buffer, heading=last_heading)
        if spoken:
            output_lines.append(spoken)

    text = "\n".join(output_lines)

    # ── Step 2: Slash-unit suffixes ───────────────────────────────────
    # Fix: "$0.00/month" was passing the slash to Kokoro as a literal
    # character, producing "zero dollars slash month".
    # Must run BEFORE the dollar regex (step 3) consumes the number.
    _slash_units = {
        "/month": " per month",
        "/mo":    " per month",
        "/year":  " per year",
        "/yr":    " per year",
        "/day":   " per day",
        "/week":  " per week",
    }
    for slash_form, spoken_form in _slash_units.items():
        # Match the slash unit immediately after a digit or closing paren
        text = re.sub(
            re.escape(slash_form),
            spoken_form,
            text,
            flags=re.IGNORECASE,
        )

    # ── Step 3: Dollar amounts ────────────────────────────────────────
    def _dollar(m: re.Match) -> str:
        amount = m.group(1).replace(",", "")
        try:
            val = float(amount)
        except ValueError:
            return m.group(0)
        dollars = int(val)
        cents   = round((val - dollars) * 100)
        if dollars == 0 and cents == 0:
            return "zero dollars"
        if cents == 0:
            return f"{dollars} dollars"
        return f"{dollars} dollars and {cents} cents"

    text = re.sub(r'\$([0-9,]+(?:\.[0-9]{1,2})?)', _dollar, text)

    # ── Step 4: Decimal numbers ───────────────────────────────────────
    # Runs after dollar conversion so "$3.50" → "3 dollars and 50 cents"
    # is already done and won't be touched by this pattern.
    text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)

    # ── Step 5: Bullet points ─────────────────────────────────────────
    # Fix: original replaced "- item" with " item" (space only).
    # This caused consecutive bullet items to run together with no pause.
    # Replace with ". " so Kokoro treats each item as a new sentence.
    #
    # Pattern covers:
    #   "  - item"    (dash bullet, any indentation)
    #   "  * item"    (asterisk bullet, any indentation)
    #   "  • item"    (unicode bullet)
    #   "  1. item"   (numbered list)
    #   "  1) item"   (numbered list with parenthesis)
    text = re.sub(r'^\s*[-*•]\s+',      '. ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[.)]\s+',   '. ', text, flags=re.MULTILINE)

    # ── Step 6: Markdown symbols ──────────────────────────────────────
    text = re.sub(r'\*+',   '', text)   # bold / italic
    text = re.sub(r'#+\s*', '', text)   # headers
    text = text.replace('`', '')        # inline code

    # ── Step 7: Whitespace collapse ───────────────────────────────────
    text = re.sub(r'\n{2,}', '. ', text)  # paragraph breaks → sentence pause
    text = re.sub(r'\n',     ' ', text)   # remaining newlines → space
    text = re.sub(r'\s{2,}', ' ', text)   # multiple spaces → single space

    # Clean up any double periods introduced by the bullet replacement
    # e.g. ". . item" or ".. item" → ". item"
    text = re.sub(r'\.\s*\.+', '.', text)
    text = re.sub(r'\.\s+\.', '.', text)

    # Remove leading ". " artifact when the very first line was a bullet.
    # Replacement prepends ". " with nothing before it, producing
    # ". First item ..." instead of "First item ...".
    text = re.sub(r'^\.\s+', '', text)

    return text.strip()


# ====================================================================== #
#  Chunk splitter — unchanged from original                              #
# ====================================================================== #

def _split_chunks(text: str, max_chars: int = 200) -> list[str]:
    """
    Split text into speakable chunks at sentence boundaries.
    Kokoro silently fails on very long inputs, so we chunk at ~200 chars.
    """
    sentences = re.split(r'(?<=[.!?:])\s+', text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current) + len(sentence) <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                parts   = sentence.split(", ")
                current = ""
                for part in parts:
                    if len(current) + len(part) <= max_chars:
                        current = (current + ", " + part).lstrip(", ")
                    else:
                        if current:
                            chunks.append(current)
                        current = part
            else:
                current = sentence
    if current:
        chunks.append(current)
    return chunks


# ====================================================================== #
#  Public API — unchanged from original                                  #
# ====================================================================== #

def speak(text: str, voice: str = "af_heart", speed: float = 1.0,
          ui_language: str = "English") -> None:
    """
    Convert text to speech and play through the default audio output.

    Long responses are split into chunks (Kokoro has a length limit), but all
    chunks are concatenated into one audio array before playback so there are
    no inter-chunk gaps or unnatural pauses.

    Args:
        text:        The text to speak.
        voice:       Kokoro voice ID (English only).
        speed:       Playback speed multiplier (1.0 = normal).
        ui_language: UI language name ("English", "中文", "Español"). Chinese and
                     Spanish route through macOS `say`; English uses Kokoro.
    """
    if not TTS_AVAILABLE:
        print("[TTS] Kokoro not available. Run: pip install kokoro-onnx onnxruntime sounddevice soundfile")
        return
    if ui_language in _SAY_VOICES:
        wav = _generate_audio_bytes_say(_clean_text(text), _SAY_VOICES[ui_language])
        if wav:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav); tmp = f.name
            try:
                data, sr = sf.read(tmp)
                sd.play(data, sr); sd.wait()
            finally:
                os.unlink(tmp)
        return
    try:
        clean = _clean_text(text)
        kokoro = _get_kokoro()
        all_samples = []
        sample_rate = 24000
        for chunk in _split_chunks(clean):
            samples, sample_rate = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
            all_samples.append(samples)
        if all_samples:
            combined = np.concatenate(all_samples)
            sd.play(combined, sample_rate)
            sd.wait()
    except Exception as e:
        print(f"[TTS error] {e}")


def generate_audio_bytes(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    ui_language: str = "English",
) -> "bytes | None":
    """
    Same pipeline as speak() but returns WAV bytes instead of playing.

    Used by the Streamlit UI to pass audio to st.audio().
    Returns None if kokoro-onnx is not installed or on any error.

    Args:
        text:        Text to convert to speech.
        voice:       Kokoro voice ID (English only).
        speed:       Playback speed multiplier (1.0 = normal).
        ui_language: UI language name ("English", "中文", "Español"). Chinese and
                     Spanish route through macOS `say`; English uses Kokoro.

    Returns:
        WAV-encoded bytes, or None on failure.
    """
    if not TTS_AVAILABLE:
        return None
    clean = _clean_text(text)
    # Detect Chinese characters in the actual text — Kokoro phonemizes them to
    # silence even with lang="en-us", so always route Chinese content to `say`.
    if any('\u4e00' <= c <= '\u9fff' for c in clean) or ui_language in _SAY_VOICES:
        say_voice = _SAY_VOICES.get(ui_language, "Lili (Premium)" if any('\u4e00' <= c <= '\u9fff' for c in clean) else "Sandy (Spanish (Mexico))")
        return _generate_audio_bytes_say(clean, say_voice)
    try:
        kokoro = _get_kokoro()
        all_samples = []
        sample_rate = 24000
        for chunk in _split_chunks(clean):
            samples, sample_rate = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
            all_samples.append(samples)
        if not all_samples:
            return None
        combined = np.concatenate(all_samples)
        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS generate_audio_bytes error] {e}")
        return None