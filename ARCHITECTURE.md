# MediCareGuide — Architecture Reference

Technical reference for the MediCareGuide codebase. For project overview, setup, and quick start see [README.md](README.md). For motivation, design rationale, and engineering decisions in narrative form see the [Kaggle write-up](Write_up_draft.md).

---

## Entry Points

| File | Purpose |
|---|---|
| `main.py` | **Primary entry point** — Streamlit web UI |
| `tests/test_medicareguide.py` | Legacy CLI interface (text only, no UI) |

---

## UI Layer — `main.py` & `ui/`

The UI is a Streamlit single-page app, refactored into a modular structure. `main.py` serves as a lightweight router, while the core UI logic resides in the `ui/` directory.

**Three screens** (defined in `ui/screens/`) controlled by `st.session_state.screen`:

| Screen | File | Description |
|---|---|---|
| `WELCOME` | `ui/screens/welcome.py` | Hero banner, Get Started CTA, RAG-grounded chat Q&A |
| `INTAKE` | `ui/screens/intake.py` | 5-step guided wizard (ZIP → track → SNP → budget → prefs) |
| `SELECT` | `ui/screens/select.py` | Filtered plan results with Gemma analysis |

**Session state keys:**
(Managed via `ui/state.py` and `init_session_state()`)
...
**Intake "Explain" buttons — hardcoded HTML, no Gemma call:**

Each intake step has an "Explain" button. All four are hardcoded constants in `ui/screens/intake.py`
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

### `ui/` — UI Layer Modules

| File | Purpose |
|---|---|
| `main.py` | **Lightweight Router**: Injects CSS, warms caches, and routes to screens based on session state. |
| `ui/screens/` | **Screen Modules**: Individual files for `welcome.py`, `intake.py`, and `select.py`. |
| `ui/components.py` | **UI Components**: Shared widgets like voice input, chat history, and i18n helpers. |
| `ui/state.py` | **Session State**: Initialization and JSON-based disk persistence. |
| `ui/backend.py` | **Backend Integration**: Streamlit-cached resource loaders for CSV, RAG, and TTS. |
| `ui/international.py` | **i18n**: Multi-language translation dictionary. |
| `ui/utils.py` | **Utilities**: Regex parsers for Gemma output and shared UI constants. |

### `core/session.py` — Session State & Intent Router
...
**`derive_sort_key(profile, context)`** — auto-selects sort key from profile signals
(priority order: PDP track → has_rx+C_SNP → has_rx+budget → has_rx → C_SNP →
budget_max → D_SNP → prefers_ppo → star_rating default).

---

### `core/inference.py` — Prompt Builder
...
**Unicode scan in `generate_audio_bytes()`:** `ui_language` alone misses English-UI + Chinese-response case; scanning for `\u4e00–\u9fff` forces `say` routing regardless.

---

## Module Dependency Graph

```
main.py  (Streamlit UI — lightweight router)
    ├── ui/backend.py          (resource loaders)
    ├── ui/state.py            (session initialization & persistence)
    ├── ui/screens/welcome.py  (Welcome screen logic)
    ├── ui/screens/intake.py   (Intake wizard logic)
    └── ui/screens/select.py   (Select screen logic)

ui/screens/ modules
    ├── ui/components.py       (shared widgets: voice, chat, i18n)
    ├── ui/utils.py            (regex parsers, constants)
    ├── ui/international.py    (translation dict)
    ├── core/session.py        (state machine, rule-based classifier)
    ├── core/lookup.py         (pandas, plan filtering/sorting)
    ├── core/inference.py      (prompt builders)
    ├── core/ollama.py         (Ollama transport)
    ├── core/rag.py            (FAISS retrieval)
    ├── core/tts.py            (Kokoro/macOS speech)
    └── core/stt.py            (Whisper transcription)

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
