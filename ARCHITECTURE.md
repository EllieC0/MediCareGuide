# MediCareGuide — Architecture Reference

Technical reference for the MediCareGuide codebase. For project overview, setup, and quick start see [README.md](README.md). For motivation, design rationale, and engineering decisions in narrative form see the [Kaggle write-up](Write_up_draft.md).

---

## Entry Points

| File | Purpose |
|---|---|
| `main.py` | **Primary entry point** — Streamlit web UI |
| `tests/test_medicareguide.py` | Legacy CLI interface (text only, no UI) |

---

## UI Layer — `main.py`

Streamlit single-page app. All UI logic lives here.

**Three screens** controlled by `st.session_state.screen`:

| Screen | Trigger | Description |
|---|---|---|
| `WELCOME` | App start | Hero banner, Get Started CTA, RAG-grounded chat Q&A |
| `INTAKE` | "Get Started" click | 5-step guided wizard (ZIP → track → SNP → budget → prefs) |
| `SELECT` | Step 5 complete | Filtered plan results with Gemma analysis |

**Session state keys:**

```python
{
    "screen":              "WELCOME" | "INTAKE" | "SELECT",
    "intake_step":         int,           # 0–5
    "explain_text":        str,           # hardcoded explanation HTML for current step
    "explain_step":        int,           # which step the explanation belongs to (-1 = none)
    "session":             MediCareGuideSession,
    "chat_history":        list[dict],    # {"role", "content"} display-only
    "filtered_df":         pd.DataFrame,
    "sorted_df":           pd.DataFrame,
    "sort_key":            str | None,
    "sort_label":          str,
    "sort_reasoning":      str,
    "filter_summary":      str,           # raw filter log from FilterDecision
    "filter_explanation":  str,           # Gemma plain-English rewrite of filter log
    "select_analysis":     str,           # raw Gemma output incl. WHY_N: markers
    "audio_enabled":       bool,
    "language":            str,           # "English" | "中文" | "Español"
    "inference_mode":      str,           # "cloud" | "local"
}
```

**Intake "Explain" buttons — hardcoded HTML, no Gemma call:**

Each intake step has an "Explain" button. All four are hardcoded constants in `main.py`
— instant, consistent, and informative (highlights differences, not definitions):

| Step | Constant | Content |
|---|---|---|
| 1 — Coverage track | `_EXPLAIN_TRACK` | 6-row comparison table: MA vs Part D vs Medigap |
| 2 — SNP flags | `_EXPLAIN_SNP` | Three cards: D-SNP, C-SNP, Extra Help/LIS |
| 3 — Budget | `_EXPLAIN_BUDGET` | What a premium is + 4-cost table + $0 premium trade-off |
| 4 — Preferences | `_EXPLAIN_PREFS` | Table showing how each checkbox affects sorting + HMO vs PPO card |

**SELECT screen — two tabs:**

- **📋 Recommendations** — Top 5 plan cards (foldable expanders) with per-plan "Why it's near the top of your list" from Gemma's `WHY_N:` structured output, filter explanation, sort reasoning, downloadable HTML summary, and follow-up chat.
- **🔀 Explore All Plans** — Six sort controls + full plan table.

**WHY_N: structured output:**
`build_prompt_select_mode()` appends `WHY_1: … WHY_5:` lines to Gemma's prompt.
`_parse_plan_whys()` extracts them for card display.
`_strip_why_lines()` removes them before storing in `chat_history`.

**Dollar sign rendering:**
All Gemma text and filter summaries rendered via `safe_md()` which escapes `$` → `\$`
before `st.markdown()` to prevent Streamlit's LaTeX renderer from mangling prices.
Hardcoded HTML blocks (explain cards, plan cards) use raw `$` — HTML is unaffected.

---

## Backend Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  WELCOME / INTAKE chat (general Q&A)                            │
│                                                                  │
│  User question                                                   │
│       │                                                          │
│       ▼                                                          │
│  core/rag.py  ── FAISS search ──► top-3 handbook passages │
│       │              (Medicare & You 2026, 213 chunks)          │
│       ▼                                                          │
│  core/inference.py  build_prompt_welcome_mode()           │
│       │  (injects RAG context as ## REFERENCE block)           │
│       ▼                                                          │
│  core/ollama.py  ──► Gemma 4  ──► page-cited answer       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SELECT mode (plan recommendation)                              │
│                                                                  │
│  5-step intake wizard  +  session restore (JSON)               │
│       │                                                          │
│       ▼                                                          │
│  core/lookup.py  get_plans_filtered()                     │
│       │  ZIP → track → SNP → budget → sanctions                │
│       ▼                                                          │
│  Gemma 4  ──► filter explanation  (plain-English rewrite)      │
│       │                                                          │
│       ▼                                                          │
│  core/lookup.py  sort_plans()  (6 deterministic strategies)│
│       │                                                          │
│       ▼                                                          │
│  core/inference.py  _derive_user_priorities()             │
│       │  (deterministic CoT — no extra Gemma call)             │
│       ▼                                                          │
│  core/inference.py  build_prompt_select_mode()            │
│       │  (profile + plan table + CoT priorities + WHY_N:)      │
│       │  (<|think|> token enables Gemma 4 thinking mode)       │
│       ▼                                                          │
│  core/ollama.py  ──► Gemma 4  ──► structured analysis    │
│       │                                                          │
│       ├──► plan cards + downloadable HTML summary              │
│       ├──► core/tts.py  Kokoro ONNX  ──► spoken output   │
│       └──► voice follow-up chat  (core/stt.py  STT)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Reference

### `core/session.py` — Session State & Intent Router

Tracks all conversation state. Uses **rule-based classification** (no LLM) for speed.

**State dict:**

```python
{
    "mode":        "WELCOME" | "SELECT",
    "intake_step": 0–5,
    "sort_key":    str,
    "profile": {
        "zip":        str | None,
        "county":     str | None,
        "state":      str | None,
        "track":      "MA_D" | "PDP" | "MEDIGAP" | None,
        "snp_flags":  list,      # "D_SNP", "C_SNP", "LIS"
        "budget_max": int | None,
    },
    "context": {
        "has_rx":        bool,
        "keep_doctors":  bool,
        "wants_dental":  bool,
        "prefers_ppo":   bool,
    },
    "history": list[dict],       # last 12 turns
}
```

**Intake methods:**

| Method | Purpose |
|---|---|
| `set_intake_field(step, field, value, action)` | Writes profile/context values from structured UI selections (steps 0–4); step 0 resolves ZIP to county + state via `_resolve_zip()` |
| `process_turn(text)` | Pure Q&A turn handler — appends to history, resolves mode, never writes to the profile |

---

### `core/lookup.py` — Data Lookup & Filtering

Loads the CMS Landscape CSV once at startup (`@st.cache_resource` in `main.py`).

**Filter stages applied by `get_plans_filtered(zipcode, profile)`:**

1. Geography filter (county/state match)
2. Coverage track filter (MA/PDP by plan type and Part D indicator)
3. SNP flag inclusion (D-SNP, C-SNP plans added when profile flags match)
4. Budget ceiling filter (drops plans above `budget_max`)
5. Sanctioned plan removal

Each stage is recorded in a `FilterDecision` dataclass (rows before → rows after)
whose `user_summary()` is passed to Gemma for plain-English rewriting.

**`sort_plans(df, sort_key, profile)`** — sort criteria:

| Key | Logic |
|---|---|
| `lowest_premium` | Consolidated premium; falls back to Part D Total Premium for PDP plans |
| `total_cost` | Premium × 12 + MOOP |
| `star_rating` | Overall star rating descending |
| `lowest_moop` | MOOP ascending (PDP plans sorted to bottom — no MOOP applies) |
| `lowest_deductible` | Annual Part D deductible ascending |
| `ppo_first` | PPO plan types first, then by lowest premium |

**`derive_sort_key(profile, context)`** — auto-selects sort key from profile signals
(priority order: PDP track → has_rx+C_SNP → has_rx+budget → has_rx → C_SNP →
budget_max → D_SNP → prefers_ppo → star_rating default).

---

### `core/inference.py` — Prompt Builder

Builds Gemma-ready prompts. Does **not** classify questions.

**Key exports:**

| Name | Description |
|---|---|
| `INTAKE_QUESTIONS` | Steps 0–4: prompt and hint text |
| `SUPPORTED_LANGUAGES` | `{"English": "", "Español": "...", "中文": "..."}` — 3 languages |
| `SYSTEM_PROMPT` | MediCareGuide persona and tone for Gemma |
| `build_prompt_welcome_mode(question, state, language, rag_context)` | WELCOME: open Q&A; injects RAG passages as `## REFERENCE` block when provided |
| `build_prompt_select_mode(question, plans, state, sort_label, sort_reasoning, language)` | SELECT: prepends `<\|think\|>`, injects profile + CoT priorities + plan table + `WHY_N:` structured output request |
| `build_prompt_filter_explanation(filter_summary, language)` | Secondary prompt: rewrites raw filter log as ≤60-word plain-English summary |
| `_derive_user_priorities(profile, context)` | Deterministic CoT — produces 3 ranked priorities from intake answers without a Gemma call |

**Chain-of-Thought (CoT) priority derivation:**
`_derive_user_priorities()` analyses intake answers in pure Python and injects a
ranked `USER PRIORITIES` block before plan data in every SELECT prompt. This anchors
Gemma's WHY_N explanations to the user's stated situation rather than generic plan
attributes, without the latency or variability of a second Gemma call.

**Thinking mode:**
SELECT prompts prepend `<|think|>` to the system prompt, enabling Gemma 4's internal
reasoning before producing the user-visible explanation. WELCOME prompts omit the token
— speed matters more than depth for handbook Q&A.

---

### `core/ollama.py` — Ollama Transport

HTTP client for the Ollama inference server. Supports both cloud and local models via a `mode` parameter.

| Constant / Function | Description |
|---|---|
| `MODEL_CLOUD` | `gemma4:31b-cloud` — Fast mode, Ollama-hosted, zero-retention |
| `MODEL_LOCAL` | `gemma4:e4b` — Private mode, fully local, air-gapped |
| `TIMEOUT_CLOUD` / `TIMEOUT_LOCAL` | 120s / 600s — per-mode request timeouts |
| `call_ollama(system_prompt, user_message, history, mode="cloud")` | POST to `/api/chat`, return answer string; `mode` selects model and timeout |
| `strip_thinking_block(text)` | Removes Gemma 4 `<\|channel>thought…<channel\|>` blocks before display |
| `split_system_and_user(full_prompt)` | Splits a combined prompt string on `"User question:"` into `(system_part, user_part)` for correct role assignment |

**Endpoint:** `http://localhost:11434/api/chat`

---

### `core/rag.py` — Retrieval-Augmented Generation

Offline RAG pipeline using the CMS *Medicare & You 2026* handbook (`10050-medicare-and-you.pdf`, 128 pages).

**Index build (once, ~30s on CPU):**
- Extracts text from PDF pages 9–128 via `pypdf` (skips cover and topic index)
- Splits each page into overlapping 300-word chunks (60-word overlap) → 213 chunks
- Embeds every chunk with `all-MiniLM-L6-v2` via `fastembed` + `onnxruntime`
- Stores L2-normalised vectors in a FAISS `IndexFlatIP` index
- Persists index + metadata to `~/.medicareguide_rag/` — subsequent startups load in ~2.5s

**Retrieval (per question, ~0.2–3s on CPU):**
- Embeds the user query with the same model
- FAISS cosine search returns top-3 most relevant chunks
- `format_context()` formats chunks as `[Medicare & You 2026, p.N] …` citation blocks
- Injected into `build_prompt_welcome_mode()` as a `## REFERENCE` block

| Function | Description |
|---|---|
| `build_or_load_index(pdf_path)` | Build from PDF or load from cache. Called once via `@st.cache_resource` |
| `retrieve(query, rag, k=3)` | Return top-k relevant chunks with page numbers and scores |
| `format_context(results, max_words=400)` | Format chunks into a prompt-ready citation block |

**Why `fastembed` instead of `sentence-transformers`:**
`sentence-transformers` requires PyTorch ≥ 2.4 via `transformers`, which conflicts with `kokoro-onnx` (numpy ≥ 2.0.2 incompatible with torch 2.2) and `ctranslate2`/`faster-whisper`. `fastembed` runs the same model via `onnxruntime` — already a dependency of `kokoro-onnx` — with no torch requirement.

---

### `core/stt.py` — Speech-to-Text

Microphone input transcribed **fully offline** by **faster-whisper** (`small` multilingual model, ~500 MB). Language is **auto-detected** on every transcription — no language code is pinned. `_WHISPER_PROMPT` seeds Medicare acronyms (`PPO HMO PDP SNP MOOP LIS D-SNP C-SNP`) as `initial_prompt` to keep them as English tokens even inside Chinese or Spanish speech.

| Function | Description |
|---|---|
| `transcribe_streamlit_audio(audio_bytes, ui_language)` | WebM/Opus → WAV via ffmpeg, transcribes with faster-whisper |
| `transcribe_audio(wav_bytes, ui_language)` | CLI path: transcribes 16kHz mono WAV directly |
| `voice_input(prompt)` | CLI flow: record → transcribe → confirm |
| `smart_input(prompt)` | Drop-in `input()` replacement with voice hint (CLI only) |

**Runtime flags required on macOS:**
- `compute_type="float32"` — `int8` triggers SIGABRT from CTranslate2 on macOS CPU
- `KMP_DUPLICATE_LIB_OK=TRUE` — prevents OpenMP abort from duplicate `libiomp5.dylib` in ctranslate2 and numpy

**Graceful degradation:** `VOICE_AVAILABLE = False` if `sounddevice` not installed; `WHISPER_AVAILABLE = False` if `faster-whisper` not installed.

---

### `core/tts.py` — Text-to-Speech

Multilingual TTS with two backends, selected automatically per language:

| Language | Backend | Voice |
|---|---|---|
| English | **Kokoro ONNX** (~350MB, fully offline) | `af_heart` (default) |
| 中文 (Chinese) | **macOS `say`** | `Lili (Premium)` — neural, must be downloaded |
| Español (Spanish) | **macOS `say`** | `Sandy (Spanish (Mexico))` — neural, must be downloaded |

**Language routing:** Routes to `say` when `ui_language` is Chinese/Spanish OR when the response text contains Unicode range `\u4e00–\u9fff` — catching the case where the UI is English but Gemma replied in Chinese (Kokoro would return silent WAV with no exception).

**Graceful degradation:** `TTS_AVAILABLE = False` if `kokoro-onnx` not installed; `say` requires no installation on macOS.

---

### `ui/international.py` — UI Internationalisation

Full UI translation for **English**, **中文 (Simplified Chinese)**, and **Español (Spanish)** via a translation dict with ~50 string keys. `t(key, language, **kwargs)` returns the translated string with English fallback. `main.py` wraps it as `_t(key)` reading language from session state automatically.

---

### `ui/style.css` — Elderly-Friendly Stylesheet

| Rule | Value |
|---|---|
| Base font | 18px+ Georgia serif |
| Heading colour | Navy `#003366` (WCAG AAA contrast) |
| Tap targets | Min 56px on all buttons |
| Layout | 860px max-width single-column |
| Hero banner | Multi-colour gradient (navy → royal blue → sky blue → teal) |

---

## Session Persistence — `~/.medicareguide_session.json`

Intake answers saved to disk on every render. On browser refresh, the app restores profile, context, sort key, and language preference, then starts at the WELCOME screen. DataFrames and Gemma analysis are not saved — regenerated when the user navigates back to SELECT.

Saved fields: `intake_step`, `sort_key`, `language`, `profile{}`, `context{}`.

---

## HTML Plan Summary Export

`_generate_summary_html()` builds a self-contained printable HTML page from session state (no new Gemma call): user profile, sort reasoning, top-5 plan cards with WHY explanations, and a pre-enrollment checklist. Available as a download button on the SELECT Recommendations tab.

---

## Module Dependency Graph

```
main.py  (Streamlit UI — primary entry point)
    ├── core/session.py        (state machine, rule-based classifier)
    │       └── core/lookup.py       [_resolve_zip()]
    ├── core/lookup.py         (pandas, zipcodes)
    ├── core/inference.py      (pandas)
    ├── core/ollama.py         (requests)
    ├── core/rag.py            (pypdf, fastembed, faiss-cpu)
    │       └── 10050-medicare-and-you.pdf
    ├── ui/international.py  (i18n translations dict)
    ├── core/tts.py            (kokoro-onnx, onnxruntime, soundfile, numpy)
    └── core/stt.py            (sounddevice, numpy, faster-whisper, ffmpeg)

tests/test_medicareguide.py  (legacy CLI)
    ├── core/session.py
    ├── core/lookup.py
    ├── core/inference.py
    ├── core/ollama.py
    ├── core/stt.py
    └── core/tts.py
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Hardcoded explain content (steps 1–4) | Gemma explanations repeated definitions already visible on screen; hardcoded content is instant and highlights actual differences |
| `safe_md()` for all Gemma output | Streamlit's `st.markdown()` interprets `$…$` as LaTeX math — prices rendered as garbled green monospace without escaping |
| `WHY_N:` structured output | Parsing free-form Gemma narrative for per-plan reasons is fragile; structured markers are reliable and can be stripped before display |
| `select_analysis` stores raw text | `WHY_N:` lines kept in `select_analysis` for `_parse_plan_whys()`; `chat_history` stores cleaned version via `_strip_why_lines()` |
| No `st.chat_input` on intake screen | `st.chat_input` is sticky at viewport bottom — nothing can render below it; replaced with `st.form` so the ← back button can sit below the input |
| PDP premium fallback | PDP plans have `"Not Applicable"` in consolidated premium column; `_plan_premium()` falls back to `"Part D Total Premium"` for display and sort |
| TTS pre-warm at startup | Kokoro model (~350MB) loads lazily on first call causing a multi-minute delay; `@st.cache_resource` warms it at startup |
| Chat messages never write to profile | Profile values come exclusively from `set_intake_field()`. `process_turn()` is a pure Q&A handler — asking "What is a PPO?" in chat cannot accidentally set `prefers_ppo=True` |
| Deterministic CoT, no second Gemma call | A second Gemma call to derive priorities would add 15–30s latency; `_derive_user_priorities()` produces the same structured anchoring purely in Python |
| `<|think|>` only on SELECT prompts | Thinking mode adds latency unsuitable for quick handbook Q&A; per-prompt token control means one model serves both use cases |
| RAG index cached to disk | Embedding 213 chunks takes ~30s on first run; FAISS index persisted to `~/.medicareguide_rag/` so all subsequent startups load in ~2.5s |
| Filter explanation as secondary Gemma call | Raw `FilterDecision.user_summary()` contains technical column names; a capped-at-60-words Gemma rewrite produces friendly prose |
| STT stored in `_select_voice_pending` | `st.chat_input()` cannot be pre-filled programmatically; transcription stored in session state and consumed on next rerun |
| `_mic_gen_{key}` counter rotates widget key | After transcription `st.rerun()` leaves browser MediaRecorder in stale state; bumping the key forces a fresh widget instance |
| `_pending_tts_text` defers TTS on WELCOME | Kokoro synthesis is synchronous; storing the answer and calling `st.rerun()` lets text appear first, TTS generates on the next pass |
| `compute_type="float32"` + `KMP_DUPLICATE_LIB_OK` | `int8` triggers SIGABRT on macOS CPU; duplicate `libiomp5.dylib` in ctranslate2 and numpy triggers a second abort without the env flag |
| Whisper `small` model, not `base` | `base` (~150MB) frequently micro-recognises Medicare acronyms; `small` (~500MB) resolves these with multilingual auto-detection intact |
| Whisper language auto-detection | Pinning language to UI setting breaks Chinese speech on English UI; auto-detection handles any mismatch transparently |
| Chinese/Spanish TTS via macOS `say` | Kokoro v0.5.0 returns silent WAV (no exception) for Chinese input; `say` with premium neural voices produces correct audio |
| Unicode scan in `generate_audio_bytes()` | `ui_language` alone misses English-UI + Chinese-response case; scanning for `\u4e00–\u9fff` forces `say` routing regardless |
| `fastembed` not `sentence-transformers` | Avoids torch dependency that creates mutually exclusive numpy requirements with kokoro-onnx |
| `SOURCES: p.X` structured output | Soft citation instructions ignored by Gemma; hard structured line (same pattern as `WHY_N:`) produces reliable page references |
| Session as JSON, not full state | DataFrames non-serialisable; lightweight profile + context persisted; SELECT analysis regenerated on re-entry |
| i18n via translation dict, not gettext | No build step; `t(key, lang)` with English fallback sufficient for 3 languages in one auditable file |
