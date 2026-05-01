# MediCareGuide
### Democratizing Medicare Knowledge for America's Elderly

Every year, roughly 11,000 Americans turn 65 and face a Medicare enrollment decision that can cost them thousands of dollars if they get it wrong. The official CMS Plan Finder is confusing, recommendations are opaque, and the "help" available is often a broker on commission. MediCareGuide is a conversational, multilingual Medicare plan advisor that gives elderly users plain-language, personalized guidance — grounded in the official CMS handbook and 2026 plan database — with no account, no subscription, and no broker commission.


---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires **Ollama** running locally:

```bash
ollama pull gemma4:31b-cloud    # Fast mode (default) — Ollama-hosted, zero-retention
ollama pull gemma4:e4b          # Private mode (optional) — fully local, air-gapped
ollama serve                     # starts automatically on most installs
```

### Required Data Files

Place both files in the same directory as `app.py`:

| File | How to get it |
|---|---|
| `CY2026_Landscape_202603.csv` | Download the [CY2026 Landscape ZIP](https://www.cms.gov/files/zip/cy2026-landscape-202603.zip) from CMS and extract the CSV |
| `10050-medicare-and-you.pdf` | Search "Medicare & You 2026" at [Medicare.gov publications](https://www.medicare.gov/publications) |

Both files are free U.S. government public data. Not included due to file size. The RAG index builds automatically on first run (~30s) and caches to `~/.medicareguide_rag/` for subsequent startups (~2.5s).

### macOS Audio Setup (Chinese and Spanish TTS)

Chinese and Spanish read-aloud uses macOS premium neural voices. Download them once:
**System Settings → Accessibility → Spoken Content → Manage Voices**
- Chinese: `Lili (Premium)`
- Spanish: `Sandy (Spanish (Mexico))`

English TTS (Kokoro ONNX) downloads automatically on first use.

---

## Two Inference Modes

| Mode | Model | Speed | Privacy |
|---|---|---|---|
| ⚡ Fast (default) | `gemma4:31b-cloud` | ~10s responses | Ollama zero-retention cloud |
| 🔒 Private | `gemma4:e4b` | Several minutes on CPU | Fully air-gapped, nothing leaves the machine |

The mode toggle is on the opening screen. All other components (STT, TTS, embeddings, filtering, session state) run on-device in both modes.

---

## Known Limitations

- **Local inference latency:** Private mode can take several minutes per response on CPU-only hardware.
- **Multilingual audio requires macOS:** Chinese and Spanish TTS uses macOS `say`. On Windows/Linux, text responses work but audio falls back to English only.
- **No provider network or formulary data:** Cannot confirm whether a specific doctor or drug is covered. Users must verify with each plan before enrolling.

---

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Primary entry point — Streamlit web UI |
| `medicareguide_inference.py` | Prompt builders for WELCOME and SELECT modes |
| `medicareguide_ollama.py` | Ollama HTTP transport, two-mode model routing |
| `medicareguide_lookup.py` | CMS plan filtering and sorting (pandas) |
| `medicareguide_rag.py` | RAG pipeline — PDF chunking, FAISS index, retrieval |
| `medicareguide_session.py` | Session state machine, intent routing |
| `medicareguide_stt.py` | Speech-to-text (faster-whisper, offline) |
| `medicareguide_tts.py` | Text-to-speech (Kokoro ONNX + macOS `say`) |
| `medicareguide_international.py` | UI i18n — English, 中文, Español |
| `style.css` | Elderly-friendly stylesheet (large text, high contrast) |
| `test_medicareguide.py` | Legacy CLI interface |

For architecture details, pipeline diagrams, and design decisions see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Matches Gemma 4's license terms.
