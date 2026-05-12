"""
MediCareGuide — Gemma 4 via Ollama
================================
CLI harness that wires all MediCareGuide modules together.

  session state → MediCareGuideLookup → prompt builder → Gemma 4

Usage:
    python test_medicareguide.py
    python test_medicareguide.py --zip 24073
    python test_medicareguide.py --zip 24073 --question "What are my cheapest options?"
    python test_medicareguide.py --csv /path/to/landscape.csv --zip 24073

Runtime commands (type at any input prompt):
    audio on / audio off     Toggle spoken responses
    sort by <criterion>      Override automatic sort order
                               sort by lowest premium
                               sort by total cost
                               sort by star rating
                               sort by lowest moop
                               sort by lowest deductible
                               sort by ppo first
                               sort by auto  (reset to automatic)
    reset                    Clear session and start over
    q / quit / exit          Exit

Requires Ollama running locally:
    ollama pull gemma4:31b-cloud
    ollama serve
"""

import sys
import argparse
import re
from pathlib import Path

# Add project root to sys.path so 'core' package can be found
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# ── Core modules ──────────────────────────────────────────────────────
from core.lookup    import MediCareGuideLookup, sort_plans, derive_sort_key, SORT_LABELS
from core.inference import (
    INTAKE_QUESTIONS,
    build_prompt_welcome_mode,
    build_prompt_select_mode,
)
from core.ollama    import call_ollama
from core.session   import MediCareGuideSession

# ── Optional modules (degrade gracefully if not installed) ────────────
try:
    from core.stt import smart_input, VOICE_AVAILABLE
except ImportError:
    VOICE_AVAILABLE = False
    def smart_input(prompt: str) -> str:      # type: ignore[misc]
        return input(prompt)

try:
    from core.tts import speak, TTS_AVAILABLE
except ImportError:
    TTS_AVAILABLE = False
    def speak(text: str, **kwargs) -> None:   # type: ignore[misc]
        pass

CSV_PATH = Path(__file__).parent.parent / "data" / "CY2026_Landscape_202603.csv"

# Medigap referral — no Landscape data exists for these plans
MEDIGAP_REFERRAL = (
    "Medigap (Medicare Supplement) plans aren't in my local database — "
    "they're sold directly by private insurers and aren't part of the "
    "CMS Landscape file I use.\n\n"
    "To compare Medigap options in your area, visit:\n"
    "  medicare.gov/find-a-plan\n\n"
    "I can still answer general questions about how Medigap works, "
    "what the different plan letters mean (G, N, etc.), or how it "
    "compares to Medicare Advantage — just ask."
)


# ====================================================================== #
#  Display helpers                                                        #
# ====================================================================== #

def show_intake_prompt(state: dict) -> None:
    """
    Replaces the original show_guided_menu().

    For intake steps 0–4: shows the structured INTAKE_QUESTIONS entry
    for the current step plus a compact summary of what's been collected.

    For step 5+ (intake complete): shows the free-form hint and
    available sort options.
    """
    step    = state["intake_step"]
    profile = state["profile"]
    context = state["context"]

    print()
    print("-" * 60)

    if step in INTAKE_QUESTIONS:
        q = INTAKE_QUESTIONS[step]
        print(f"  {q['prompt']}")
        print()
        for line in q["hint"].splitlines():
            print(f"  {line}" if line.strip() else "")

        # Compact progress line — only shown once at least one field is set
        collected = []
        if profile.get("zip"):
            loc = profile["zip"]
            if profile.get("county"):
                loc += f" ({profile['county']})"
            collected.append(f"ZIP: {loc}")
        if profile.get("track"):
            collected.append(f"Track: {profile['track']}")
        if profile.get("snp_flags"):
            collected.append(f"Flags: {', '.join(profile['snp_flags'])}")
        if profile.get("budget_max") is not None:
            collected.append(f"Budget: ${profile['budget_max']}/mo")
        active_prefs = [k for k, v in context.items() if v]
        if active_prefs:
            collected.append(f"Prefs: {', '.join(active_prefs)}")

        if collected:
            print()
            print(f"  Progress ({step}/5): {' · '.join(collected)}")

    else:
        # Intake complete — free-form mode
        print("  What would you like to know?")
        print()
        print("  Re-sort results anytime:")
        print("    sort by lowest premium  (default)")
        print("    sort by total cost")
        print("    sort by star rating")
        print("    sort by lowest moop")
        print("    sort by lowest deductible")

    print("-" * 60)


# ====================================================================== #
#  Intake step parser                                                     #
# ====================================================================== #

def _run_intake_step(step: int, user_input: str, session: MediCareGuideSession) -> None:
    """
    Parse user_input for the given intake step and call set_intake_field().
    Prints an error and leaves intake_step unchanged on invalid input.
    """
    text = user_input.strip().lower()

    if step == 0:
        session.set_intake_field(0, "zip", user_input.strip())
        p = session.state["profile"]
        if p.get("zip"):
            print(f"\n  ZIP {user_input.strip()} → {p['county']}, {p['state']}")
        else:
            session.state["intake_step"] = 0   # undo advance on failed ZIP lookup
            print(f"\n  ZIP '{user_input.strip()}' not recognised. Try a 5-digit US ZIP code.")

    elif step == 1:
        _TRACK_MAP = {
            "1": "MA_D",    "ma": "MA_D",  "medicare advantage": "MA_D",
            "2": "PDP",     "pdp": "PDP",  "part d": "PDP",
            "3": "MEDIGAP", "medigap": "MEDIGAP", "supplement": "MEDIGAP",
        }
        track = _TRACK_MAP.get(text)
        if track:
            session.set_intake_field(1, "track", track)
        else:
            print("\n  Not recognised. Type 1 (Medicare Advantage), 2 (Part D), or 3 (Medigap).")

    elif step == 2:
        _FLAG_MAP = {"1": "D_SNP", "2": "C_SNP", "3": "LIS"}
        if text in ("none", "n", "0", ""):
            session.set_intake_field(2, "snp_flags", [])
            return
        flags, valid = [], True
        for part in text.replace(" ", "").split(","):
            if part in _FLAG_MAP:
                flags.append(_FLAG_MAP[part])
            else:
                print(f"\n  '{part}' not recognised. Enter numbers 1–3, comma-separated, or 'none'.")
                valid = False
                break
        if valid:
            session.set_intake_field(2, "snp_flags", flags)

    elif step == 3:
        if text in ("none", "n", "no limit", ""):
            session.set_intake_field(3, "budget_max", None)
            return
        try:
            budget = int(text.replace("$", "").replace(",", "").strip())
            session.set_intake_field(3, "budget_max", budget)
        except ValueError:
            print("\n  Enter a whole dollar amount (e.g. '50') or 'none' for no limit.")

    elif step == 4:
        _CTX_MAP = {
            "1": "has_rx", "2": "keep_doctors",
            "3": "wants_dental", "4": "prefers_ppo",
        }
        if text in ("none", "n", "0", ""):
            session.set_intake_field(4, "context", [])
            return
        flags, valid = [], True
        for part in text.replace(" ", "").split(","):
            if part in _CTX_MAP:
                flags.append(_CTX_MAP[part])
            else:
                print(f"\n  '{part}' not recognised. Enter numbers 1–4, comma-separated, or 'none'.")
                valid = False
                break
        if valid:
            session.set_intake_field(4, "context", flags)


# ====================================================================== #
#  Core turn handler                                                      #
# ====================================================================== #

def handle_turn(
    user_input: str,
    session:    MediCareGuideSession,
    lookup:     MediCareGuideLookup,
) -> str:
    """
    Runs one full conversation turn:
        1. session.process_turn() — classify input, update state, resolve mode
        2. Route: MEDIGAP referral / EDUCATE prompt / SELECT prompt
        3. call_ollama()
        4. session.close_turn() — append answer to history

    Returns the answer string.

    Replaces the original classify_question → build_prompt → call_ollama block.

    Decision — call_ollama receives system_prompt="" and history=[]:
        Both the system prompt and conversation history are already embedded
        inside the strings returned by build_prompt_educate_mode() and
        build_prompt_select_mode(). Passing them again via call_ollama's
        parameters would cause Gemma to see them twice, degrading responses.
        split_system_and_user() is NOT used here — the new prompt builders
        are self-contained and do not need splitting.
    """
    state   = session.process_turn(user_input)
    mode    = state["mode"]
    profile = state["profile"]

    # ── MEDIGAP: no data available, issue referral ────────────────────
    if profile.get("track") == "MEDIGAP":
        answer = MEDIGAP_REFERRAL
        session.close_turn(answer)
        return answer

    # ── WELCOME mode ──────────────────────────────────────────────────
    if mode == "WELCOME":
        prompt = build_prompt_welcome_mode(user_input, state)
        answer = call_ollama(
            system_prompt="",
            user_message=prompt,
            history=[],
        )
        session.close_turn(answer)
        return answer

    # ── SELECT mode ───────────────────────────────────────────────────
    filtered_df, decision = lookup.get_plans_filtered(
        profile["zip"],
        profile,
    )

    # Print filter transparency summary before Gemma's answer
    summary = decision.user_summary()
    if summary:
        print()
        print("·" * 60)
        for line in summary.splitlines():
            print(f"  {line}")
        print("·" * 60)

    # ── Derive or use manual sort key ────────────────────────────────
    manual_key = state.get("sort_key")
    if manual_key:
        sort_key      = manual_key
        sort_reasoning = f"You selected: {SORT_LABELS.get(sort_key, sort_key)}"
    else:
        sort_key, sort_reasoning = derive_sort_key(profile, state["context"])

    sorted_df, sort_label = sort_plans(filtered_df, sort_key, profile)
    top_plans             = sorted_df.head(5)

    # Print sort reasoning + transparency header
    print()
    print("·" * 60)
    print(f"  {sort_reasoning}")
    if sort_label:
        print(f"  ({sort_label})")
    if not top_plans.empty:
        print(f"  Discussing top {len(top_plans)} of {len(sorted_df)} plan(s).")
    print("·" * 60)

    prompt = build_prompt_select_mode(
        user_question=user_input,
        plans=top_plans,
        state=state,
        sort_label=sort_label,
        sort_reasoning=sort_reasoning,
    )
    answer = call_ollama(
        system_prompt="",
        user_message=prompt,
        history=[],
    )
    session.close_turn(answer)
    return answer


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="MediCareGuide — Medicare plan advisor")
    parser.add_argument(
        "--zip",
        default=None,
        help="ZIP code (skips the ZIP prompt on the first turn)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Single question to answer then exit (single-shot mode)",
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"Path to CMS Landscape CSV (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--debug-filters",
        action="store_true",
        default=False,
        dest="debug_filters",
        help="Print full filter decision trace on every SELECT turn",
    )
    args = parser.parse_args()

    # ── Welcome banner ────────────────────────────────────────────────
    print("=" * 60)
    print("  MediCareGuide — Powered by Gemma 4 (Ollama)")
    if VOICE_AVAILABLE:
        print("  Voice input available — type 'v' at any prompt.")
    if TTS_AVAILABLE:
        print("  Type 'audio on' at any prompt to enable voice responses.")
    print("=" * 60)

    # ── Load Landscape CSV ────────────────────────────────────────────
    print(f"\nLoading: {args.csv}")
    try:
        lookup = MediCareGuideLookup(args.csv)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {args.csv}")
        print("  Download from cms.gov and pass --csv <path>.")
        sys.exit(1)

    # ── Initialise session ────────────────────────────────────────────
    session       = MediCareGuideSession(lookup)
    audio_enabled = False

    # Pre-load ZIP from CLI flag via set_intake_field() — resolves
    # county + state and advances intake_step exactly as the UI does.
    if args.zip:
        session.set_intake_field(0, "zip", args.zip.strip())
        p = session.state["profile"]
        if p.get("county"):
            print(f"  ZIP {args.zip} → {p['county']}, {p['state']}")
        else:
            session.state["intake_step"] = 0   # reset so user is re-prompted
            print(f"  ZIP {args.zip} not recognised — you'll be prompted again.")

    # ── Single-shot mode ──────────────────────────────────────────────
    if args.question:
        print("\nMediCareGuide is thinking...\n")
        answer = handle_turn(args.question, session, lookup)
        print("-" * 60)
        print(answer)
        print("-" * 60)
        sys.exit(0)

    # ── Conversation loop ─────────────────────────────────────────────
    pending_question: str = ""

    _SORT_CMD_MAP = {
        "lowest premium":    "lowest_premium",
        "total cost":        "total_cost",
        "star rating":       "star_rating",
        "lowest moop":       "lowest_moop",
        "lowest deductible": "lowest_deductible",
        "ppo first":         "ppo_first",
    }

    while True:
        step = session.state["intake_step"]

        # ── Structured intake steps 0–4 ───────────────────────────────
        if step in INTAKE_QUESTIONS:
            show_intake_prompt(session.state)
            choice = smart_input("\n  You: ").strip()

            if choice.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break
            if choice.lower() in ("reset", "start over", "restart"):
                session = MediCareGuideSession(lookup)
                print("\n  Session reset. Starting over.")
                continue

            _run_intake_step(step, choice, session)
            continue

        # ── Free-form Q&A (all intake steps complete) ─────────────────
        if pending_question:
            user_question    = pending_question
            pending_question = ""
        else:
            show_intake_prompt(session.state)
            choice = smart_input("\n  You: ").strip()

            # Sort override
            _sort_match = re.match(r"sort\s+by\s+(.+)", choice.lower())
            if _sort_match:
                criterion = _sort_match.group(1).strip()
                if criterion == "auto":
                    session.state["sort_key"] = None
                    print("  Sort reset to automatic (based on your profile).")
                elif criterion in _SORT_CMD_MAP:
                    session.state["sort_key"] = _SORT_CMD_MAP[criterion]
                    print(f"  Sort updated: {SORT_LABELS[_SORT_CMD_MAP[criterion]]}")
                else:
                    print(f"  Unknown sort criterion '{criterion}'. Options:")
                    for cmd in _SORT_CMD_MAP:
                        print(f"    sort by {cmd}")
                    print("    sort by auto  (reset to automatic)")
                continue

            # Audio toggle
            if choice.lower() in ("audio on", "audio off"):
                audio_enabled = choice.lower() == "audio on"
                if audio_enabled and not TTS_AVAILABLE:
                    print("  TTS not available. Install kokoro-onnx to enable audio.")
                    audio_enabled = False
                else:
                    print(f"  [Audio {'ON' if audio_enabled else 'OFF'}]")
                continue

            # Reset
            if choice.lower() in ("reset", "start over", "restart"):
                session = MediCareGuideSession(lookup)
                print("\n  Session reset. Starting over.")
                continue

            # Quit
            if choice.lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break

            if not choice:
                continue

            user_question = choice

        # ── Debug filter trace (SELECT mode only) ────────────────────
        if args.debug_filters and session.state["mode"] == "SELECT":
            _, decision = lookup.get_plans_filtered(
                session.state["profile"]["zip"],
                session.state["profile"],
            )
            decision.print_summary()

        # ── Run the turn ─────────────────────────────────────────────
        print("\nMediCareGuide is thinking...\n")
        answer = handle_turn(user_question, session, lookup)

        print("-" * 60)
        print(answer)
        print("-" * 60)

        if audio_enabled:
            speak(answer)

        # ── Follow-up prompt (mirrors original behaviour) ─────────────
        again = smart_input(
            "\n  Follow-up question, or press Enter for menu (q to quit): "
        ).strip()

        if again.lower() in ("audio on", "audio off"):
            audio_enabled = again.lower() == "audio on"
            if audio_enabled and not TTS_AVAILABLE:
                print("  TTS not available.")
                audio_enabled = False
            else:
                print(f"  [Audio {'ON' if audio_enabled else 'OFF'}]")
            continue

        if again.lower() in ("reset", "start over", "restart"):
            session = MediCareGuideSession(lookup)
            print("\n  Session reset.")
            if args.zip:
                session.set_intake_field(0, "zip", args.zip.strip())
            continue

        if again.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        if again:
            pending_question = again


if __name__ == "__main__":
    main()