"""
MediCareGuide Inference Module
==============================
Builds Gemma-ready prompts from plan data + session state + conversation
history. Routing between welcome and selection mode is handled upstream
by MediCareGuideSession — this module only builds prompts.

Usage:
    from medicareguide_inference import (
        INTAKE_QUESTIONS,
        build_prompt_welcome_mode,
        build_prompt_select_mode,
    )
"""

from __future__ import annotations

import pandas as pd


# ====================================================================== #
#  Guided intake questions #
# ====================================================================== #
#
#  One entry per intake_step (0–4). Each entry has:
#    prompt  — the question shown to the user verbatim
#    hint    — options or guidance printed below the prompt
#
#  intake_step 0 → ask for ZIP
#  intake_step 1 → ask for coverage track
#  intake_step 2 → ask for situation flags (SNP / LIS)
#  intake_step 3 → ask for budget ceiling
#  intake_step 4 → ask for unverifiable preferences (context{})
#
#  Once intake_step reaches 5, the guided menu is no longer shown.
#  The user may ask free-form questions at any point.

INTAKE_QUESTIONS = {
    0: {
        "prompt": "What's your ZIP code?",
        "hint": "Enter your 5-digit ZIP code so I can find plans in your area.",
    },
    1: {
        "prompt": "Which type of Medicare coverage are you looking for?",
        "hint": (
            "  1  Medicare Advantage (MA) — replaces Original Medicare; usually includes\n"
            "       Part D drug coverage, dental, and vision in one plan.\n"
            "  2  Part D only (PDP) — adds drug coverage alongside Original Medicare.\n"
            "  3  Medigap — supplement to Original Medicare (I'll refer you to Medicare.gov).\n"
            "\n"
            "Type 1, 2, or 3 (or 'ma', 'pdp', 'medigap'):"
        ),
    },
    2: {
        "prompt": "Do any of the following apply to you? (comma-separated numbers, or 'none')",
        "hint": (
            "  1  D-SNP — I have both Medicare and Medicaid.\n"
            "  2  C-SNP — I have a chronic condition (heart disease, diabetes, COPD, etc.).\n"
            "  3  Extra Help / LIS — I qualify for Medicare's Low Income Subsidy.\n"
            "\n"
            "Example: '1,3' for D-SNP + Extra Help, or 'none' if none apply:"
        ),
    },
    3: {
        "prompt": "What is the most you can pay for your monthly plan premium?",
        "hint": (
            "Enter a dollar amount (e.g. '50' for $50/month).\n"
            "Press Enter or type 'none' to skip (no budget ceiling):"
        ),
    },
    4: {
        "prompt": "Which of these are important to you? (comma-separated numbers, or 'none')",
        "hint": (
            "  1  I take regular prescription medications.\n"
            "  2  I want to keep my current doctors.\n"
            "  3  Dental and vision coverage matter to me.\n"
            "  4  I prefer a PPO (more flexibility to see any doctor).\n"
            "\n"
            "Example: '1,3' for prescriptions + dental, or 'none':"
        ),
    },
}


# ====================================================================== #
#  Columns included in plan data sent to Gemma                           #
# ====================================================================== #
#
#  42 of 52 Landscape columns. Dropped: 9 internal CMS identifiers
#  (Contract ID, Plan ID, Segment ID, ContractPlanID,
#  ContractPlanSegmentID, MA Region Code, PDP Region Code,
#  US Territory, State Territory Abbreviation)
#  and the _rank column added by sort_plans().

DISPLAY_COLUMNS = [
    # Identity & organisation
    "Contract Year",
    "Contract Category Type",
    "State Territory Name",
    "County Name",
    "Parent Organization Name",
    "Contract Name",
    "Organization Marketing Name",
    "Organization Type",
    "Plan Name",
    "Plan Type",

    # Quality
    "Part C Summary Star Rating",
    "Part D Summary Star Rating",
    "Overall Star Rating",

    # Cost — premiums
    "Part C Premium",
    "Part D Basic Premium",
    "Part D Supplemental Premium",
    "Part D Total Premium",
    "Monthly Consolidated Premium (Part C + D)",

    # Cost — deductibles and out-of-pocket
    "Annual Part D Deductible Amount",
    "In-Network Maximum Out-of-Pocket (MOOP) Amount",
    "Part D Out-of-Pocket (OOP) Threshold",

    # Drug coverage
    "Part D Coverage Indicator",
    "National PDP",
    "Drug Benefit Category",
    "Drug Benefit Type",
    "Offers Drug Tier with No Part D Deductible",

    # Special needs plans
    "Special Needs Plan (SNP) Indicator",
    "SNP Type",
    "SNP Institutional Type",
    "SNP Institutional Category",
    "Dual Eligible SNP (D-SNP) Integration Status",
    "D-SNP Applicable Integrated Plan (AIP) Identifier",
    "Chronic or Disabling Condition SNP (C-SNP) Condition Type",
    "Medicare Zero-Dollar Cost Sharing D-SNP Plan",

    # Low income / Extra Help
    "Low Income Subsidy (LIS) Auto Enrollment",
    "Part D Basic Premium At or Below Regional Benchmark",
    "Voluntary De Minimis Program Participant",
    "Low Income Premium Subsidy (LIPS) Amount",
    "Part D LIPS (CMS Pays)",
    "Part D Low Income Beneficiary Premium Amount",

    # Safety
    "Sanctioned Plan",

    # Region (display names, not codes)
    "MA Region",
    "PDP Region",
]


# ====================================================================== #
#  System prompt                                                         #
# ====================================================================== #
#
#  Session handles WELCOME vs SELECT.
#  Retained: persona, tone, what Gemma knows, what it must not do.

# ====================================================================== #
#  Language support                                                      #
# ====================================================================== #

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "",
    "Español": "Respond entirely in Spanish (Español).",
    "中文":     "Respond entirely in Simplified Chinese (简体中文).",
}

_SELECT_LANGUAGE_SUFFIX = (
    " EXCEPTION: the structured output lines at the end of your response "
    "(WHY_1:, WHY_2:, WHY_3:, WHY_4:, WHY_5:) must keep their exact English labels — "
    "only translate the sentence content that follows the colon."
)


def _language_instruction(language: str, select_mode: bool = False) -> str:
    """Return the language instruction line for a given language key.
    Returns an empty string for English (default — no extra instruction needed).
    In select_mode, appends an exception to preserve WHY_N: English markers."""
    base = SUPPORTED_LANGUAGES.get(language, "")
    if base and select_mode:
        return base + _SELECT_LANGUAGE_SUFFIX
    return base


SYSTEM_PROMPT = """You are MediCareGuide, a friendly and knowledgeable Medicare plan advisor.

YOUR ROLE:
- Help Medicare beneficiaries understand and compare the plans available in their area.
- Explain plan details in plain, simple language. Avoid jargon.
- When comparing plans, organise your answer clearly so differences are easy to see.

HOW TO HANDLE DIFFERENT QUESTION TYPES:

1. GENERAL MEDICARE EDUCATION (like "what is an HMO?" or "what does star rating mean?"):
   - Answer from your general Medicare knowledge.
   - After explaining the concept, connect it to the user's situation when relevant.
   - Example: "An HMO requires you to use in-network providers. In your area there
     are several HMO plans — once you choose a coverage track I can show you exactly
     which ones are available."

2. FOLLOW-UP QUESTIONS (like "tell me more about that plan" or "what about #2?"):
   - Use the conversation history to resolve references like "that plan" or "the
     second one". Always anchor to a specific plan name before answering.

3. OUT-OF-SCOPE QUESTIONS (like "is my doctor in-network?" or "does this cover
   knee surgery?" or "what's my copay for a specialist?"):
   - Be honest: this tool covers plan costs, ratings, types, and drug coverage
     availability. It does not have provider networks or drug formularies.
   - Suggest they call the plan directly or visit Medicare.gov for those details.
   - Do NOT guess or make up coverage details.

WHAT YOU KNOW:
- Plan name, organisation, plan type, monthly premiums (Part C + D combined),
  drug deductible, in-network out-of-pocket maximum, star ratings, SNP status,
  drug coverage indicators, and low-income subsidy details.
- Star ratings run 1–5. Higher is better. "Plan Too New To Be Measured" means
  the plan launched recently and has not yet been rated.
- Special Needs Plans (SNPs) are designed for specific populations: dual-eligible
  (Medicare + Medicaid), chronic conditions, or institutional care.
- Low-income subsidy (LIS / Extra Help) fields show what a beneficiary with
  limited income would pay after premium subsidies are applied.

WHAT YOU MUST NOT DO:
- Do not make up plan details not in the data provided to you.
- Do not recommend one plan over another as "the best". Present facts and trade-offs
  and let the user decide.
- Do not give medical advice or diagnose conditions.
- Do not reference internal identifiers like Contract ID, Plan ID, or Segment ID.

TONE:
- Warm, patient, and clear — as if explaining to a family member.
- Keep answers concise but thorough enough to be useful.
- If the user seems confused, offer to explain terms or suggest they try one of
  the guided questions.
"""


# ====================================================================== #
#  Internal helpers                                                      #
# ====================================================================== #

def _format_plans_as_text(plans: pd.DataFrame) -> str:
    """
    Convert a DataFrame of plans into structured text blocks
    that Gemma can parse and reason about.

    Skips the internal _rank column if present (added by sort_plans()).
    """
    available_cols = [c for c in DISPLAY_COLUMNS if c in plans.columns]
    subset = plans[available_cols]

    blocks = []
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        # Use _rank if present, otherwise sequential number
        rank = int(plans.iloc[i - 1]["_rank"]) if "_rank" in plans.columns else i
        lines = [f"--- Plan {rank} ---"]
        for col in available_cols:
            value = str(row[col]).strip()
            if pd.isna(row[col]) or value == "" or value == "nan":
                continue
            lines.append(f"  {col}: {value}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _format_history(history: list[dict]) -> str:
    """
    Convert conversation history into a text block for the prompt.

    Accepts entries in either format:
      {"role": "user",      "content": "..."}   ← session format
      {"role": "assistant", "content": "..."}
    """
    if not history:
        return ""

    lines = ["CONVERSATION SO FAR:"]
    for turn in history:
        role = "User" if turn["role"] == "user" else "MediCareGuide"
        content = turn.get("content", turn.get("text", ""))
        lines.append(f"  {role}: {content}")

    return "\n".join(lines) + "\n\n"


def _format_profile_text(profile: dict) -> str:
    """Render the session profile as a readable bullet list."""
    lines = []
    if profile.get("zip"):
        loc = profile["zip"]
        if profile.get("county") and profile.get("state"):
            loc += f" ({profile['county']}, {profile['state']})"
        lines.append(f"ZIP: {loc}")
    if profile.get("track"):
        lines.append(f"Coverage track: {profile['track']}")
    if profile.get("snp_flags"):
        lines.append(f"Situation flags: {', '.join(profile['snp_flags'])}")
    if profile.get("budget_max") is not None:
        lines.append(f"Budget ceiling: ${profile['budget_max']}/month")
    return "\n".join(f"  - {l}" for l in lines) if lines else "  - (none collected yet)"


def _format_context_text(context: dict) -> str:
    """Render the session context flags as a readable bullet list."""
    active = []
    if context.get("has_rx"):
        active.append("Takes regular prescriptions")
    if context.get("keep_doctors"):
        active.append("Wants to keep current doctors / providers")
    if context.get("wants_dental"):
        active.append("Dental, vision, or hearing benefits important")
    if context.get("prefers_ppo"):
        active.append("Prefers PPO (flexibility to see any doctor)")
    return "\n".join(f"  - {a}" for a in active) if active else "  - (none stated)"


def _context_reasoning_notes(context: dict) -> str:
    """
    Generate specific reasoning instructions for Gemma based on
    which context flags are set. These appear inside SELECT prompts.
    """
    notes = []
    if context.get("has_rx"):
        notes.append(
            "- The user takes regular prescriptions. After your plan summary, "
            "remind them that formulary coverage (which specific drugs are covered) "
            "cannot be confirmed here — they must verify their medications with "
            "each plan directly before enrolling."
        )
    if context.get("keep_doctors"):
        notes.append(
            "- The user wants to keep their current doctors. Remind them to verify "
            "that their providers are in-network before enrolling, especially for "
            "HMO plans which restrict access to plan networks."
        )
    if context.get("wants_dental"):
        notes.append(
            "- The user cares about dental, vision, or hearing benefits. Note for "
            "each plan whether it is an HMO (more likely to bundle supplemental "
            "benefits) or PPO (less commonly includes them). Be specific if the "
            "plan data indicates supplemental benefit availability."
        )
    if context.get("prefers_ppo"):
        notes.append(
            "- The user prefers PPO flexibility (no referrals, wider network). "
            "When discussing plans, flag which are HMO vs PPO and explain the "
            "practical difference for this user."
        )
    return "\n".join(notes) if notes else ""


def _derive_user_priorities(profile: dict, context: dict) -> str:
    """
    Derive three ranked plain-English evaluation priorities from the intake
    answers and inject them into the SELECT prompt before the plan data.

    Purpose: give Gemma an explicit, pre-reasoned anchor for its WHY_N
    sentences so they reference what actually matters to this user rather
    than generic plan attributes.

    Priority 1 — cost shape (budget + Rx interaction)
    Priority 2 — network / flexibility
    Priority 3 — special situation or quality fallback

    Fully deterministic — no extra Gemma call, no added latency.
    """
    priorities = []
    budget_max = profile.get("budget_max")
    snp_flags  = profile.get("snp_flags", [])
    has_rx        = context.get("has_rx", False)
    keep_doctors  = context.get("keep_doctors", False)
    wants_dental  = context.get("wants_dental", False)
    prefers_ppo   = context.get("prefers_ppo", False)

    # ── Priority 1: cost shape ────────────────────────────────────────
    if budget_max is not None and has_rx:
        priorities.append(
            f"1. Total annual cost — this user has a ${budget_max}/month premium "
            f"ceiling AND takes regular prescriptions. Drug deductible and total "
            f"out-of-pocket matter more than headline premium alone. Cite both "
            f"the premium and drug deductible when explaining each plan."
        )
    elif budget_max is not None:
        priorities.append(
            f"1. Monthly premium — this user set a strict ${budget_max}/month "
            f"ceiling. Call out exactly how each plan's premium compares to that "
            f"limit and flag any plan that is close to or at the ceiling."
        )
    elif has_rx:
        priorities.append(
            "1. Drug costs — this user takes regular prescriptions. "
            "Part D deductible and whether the plan offers a $0-deductible drug "
            "tier are the primary cost drivers. Cite the drug deductible value "
            "for each plan."
        )
    else:
        priorities.append(
            "1. Overall value — no specific cost constraint was stated. "
            "Balance monthly premium, MOOP, and star rating. Prefer plans where "
            "all three are favourable rather than optimising one at the expense "
            "of the others."
        )

    # ── Priority 2: network / flexibility ────────────────────────────
    if prefers_ppo:
        priorities.append(
            "2. Plan type flexibility — this user explicitly prefers PPO. "
            "For every plan, state whether it is HMO or PPO. Explain that HMO "
            "plans require referrals and restrict to in-network providers, while "
            "PPO plans allow out-of-network visits without referrals."
        )
    elif keep_doctors:
        priorities.append(
            "2. Provider continuity — this user wants to keep their current "
            "doctors. Flag any HMO plans prominently: HMO networks are closed "
            "and require referrals. PPO or HMO-POS plans give more flexibility. "
            "Remind the user to verify their specific providers before enrolling."
        )
    else:
        priorities.append(
            "2. Financial protection — with no network preference stated, "
            "the in-network MOOP is the key differentiator between plans at "
            "similar premiums. A lower MOOP caps worst-case costs for the year. "
            "Cite the MOOP value explicitly for each plan."
        )

    # ── Priority 3: special situation or quality fallback ────────────
    if "D_SNP" in snp_flags:
        priorities.append(
            "3. Dual-eligible coordination — this user has both Medicare and "
            "Medicaid. D-SNP plans are specifically designed to reduce paperwork "
            "and coordinate benefits between both programs. Highlight any D-SNP "
            "plan and explain why that coordination matters for this user."
        )
    elif "C_SNP" in snp_flags:
        priorities.append(
            "3. Chronic condition support — this user has a qualifying chronic "
            "condition. C-SNP plans provide specialised care coordination for "
            "that condition. Highlight any C-SNP plan and explain the care-team "
            "benefit over a general Medicare Advantage plan."
        )
    elif wants_dental:
        priorities.append(
            "3. Supplemental benefits — dental, vision, or hearing benefits "
            "matter to this user. For each plan, note whether it is an HMO "
            "(more likely to bundle supplemental benefits) or PPO. Caution that "
            "benefit scope varies widely — 'includes dental' may mean cleanings "
            "only; the user must confirm coverage details directly with the plan."
        )
    else:
        priorities.append(
            "3. Quality rating — as a proxy for care quality, favour plans "
            "with 4 or more CMS stars when premiums and MOOP are otherwise "
            "similar. Note the star rating for every plan and flag any plan "
            "rated below 3.5 stars or listed as 'Plan Too New To Be Measured'."
        )

    return "\n".join(priorities)


# ====================================================================== #
#  WELCOME mode prompt                                                   #
# ====================================================================== #
#
#  Active before the user clicks "Start finding my plan".
#  No intake questions, no steering. Pure open conversation.

def build_prompt_welcome_mode(
    user_question: str,
    state: dict,
    language: str    = "English",
    rag_context: str = "",
) -> str:
    """
    Prompt for WELCOME mode.

    The user has not yet started the guided intake. They may be asking
    general Medicare questions, checking if this tool is right for them,
    or simply exploring. Gemma should answer naturally with no steering
    toward intake questions.

    If rag_context is provided (passages retrieved from Medicare & You 2026),
    it is injected as authoritative reference material before the question.
    """
    history_text  = _format_history(state["history"][:-1])
    lang_instr    = _language_instruction(language)
    lang_block    = f"\n\nLANGUAGE INSTRUCTION: {lang_instr}" if lang_instr else ""
    rag_block     = (
        f"\n\n## REFERENCE — Medicare & You 2026 (official CMS handbook)\n"
        f"The following passages were retrieved from the official handbook. "
        f"Use them to ground your answer in authoritative CMS language.\n\n"
        f"{rag_context}"
    ) if rag_context else ""

    # Hard structured-output requirement — only when handbook passages are present.
    # Placed after the user question so it is the last instruction Gemma sees.
    sources_instr = (
        f"\n\nAfter your answer, on a new line write exactly:\n"
        f"SOURCES: p.X, p.Y\n"
        f"listing every handbook page number you drew on. "
        f"If you did not use the handbook passages, write: SOURCES: none"
    ) if rag_context else ""

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## MODE: WELCOME\n"
        f"The user has not yet started the plan-finding intake. "
        f"Answer their question naturally from your general Medicare knowledge. "
        f"Do not reference specific plans or local coverage data. "
        f"Do not steer toward or mention any intake questions. "
        f"Do not prompt the user to provide their ZIP code or coverage preferences. "
        f"If they seem ready to find a plan, you may mention that clicking "
        f"'Start finding my plan' will begin the guided process — but only if "
        f"it is natural to do so."
        f"{rag_block}\n\n"
        f"{history_text}"
        f"User question: {user_question}"
        f"{lang_block}"
        f"{sources_instr}"
    )


# ====================================================================== #
#  SELECT mode prompt                                                    #
# ====================================================================== #

def build_prompt_select_mode(
    user_question: str,
    plans: pd.DataFrame,
    state: dict,
    sort_label: str = "",
    sort_reasoning: str = "",
    language: str = "English",
) -> str:
    """
    Prompt for SELECT mode.

    Used when both zip and track are set in the session profile.
    Receives a pre-filtered, pre-sorted DataFrame from get_plans_filtered()
    + sort_plans(). Gemma's job is explanation and trade-off narration only
    — it must NOT reorder the plans.

    Args:
        user_question:  The user's current input.
        plans:          Pre-filtered and pre-sorted DataFrame (top 5 rows).
                        Must already have _rank column set by sort_plans().
        state:          The full session state dict from MediCareGuideSession.
        sort_label:     Transparency string from sort_plans() describing the
                        active sort order, e.g. "Sorted by lowest monthly premium".
        sort_reasoning: Plain-English explanation of why this sort was chosen,
                        derived from the user's intake. Shown to the user and
                        used by Gemma to open its response.

    Returns:
        A complete prompt string ready to send to Gemma via call_ollama().
    """
    profile = state["profile"]
    context = state["context"]

    history_text  = _format_history(state["history"][:-1])
    profile_text  = _format_profile_text(profile)
    context_text  = _format_context_text(context)
    reasoning     = _context_reasoning_notes(context)
    priorities    = _derive_user_priorities(profile, context)
    lang_instr    = _language_instruction(language, select_mode=True)
    lang_block    = f"\n\nLANGUAGE INSTRUCTION: {lang_instr}" if lang_instr else ""

    plan_count = len(plans)
    if plans.empty:
        plans_text = (
            "No plans matched the user's filters. Let the user know and suggest "
            "they broaden their criteria — for example by removing the budget "
            "ceiling or choosing a different coverage track."
        )
    else:
        plans_text = _format_plans_as_text(plans)

    # ── Sort transparency instruction ─────────────────────────────────
    sort_instruction = ""
    if sort_label:
        sort_instruction = (
            f"\nSORT ORDER: {sort_label}\n"
            f"Plans are already sorted by Python code using the exact column values "
            f"in the data. Present them in the order given. Do NOT reorder them. "
            f"When explaining why Plan 1 appears first, cite the specific value "
            f"from its data (e.g. 'Plan 1 has the lowest premium at $0.00/month') "
            f"rather than making a general statement.\n"
        )

    # ── Sort reasoning block ──────────────────────────────────────────
    # sort_reasoning explains WHY this sort was chosen from the user's intake.
    # Gemma opens its response with one sentence citing this, then lists the
    # override options so the user knows they can change it.
    sort_reasoning_block = ""
    if sort_reasoning:
        sort_reasoning_block = (
            f"\nSORT REASONING:\n{sort_reasoning}\n"
            f"Open your response with one sentence explaining why plans are "
            f"sorted this way (paraphrase, do not repeat verbatim). Then tell "
            f"the user they can change the sort order by choosing: lowest "
            f"premium, total cost, star rating, lowest MOOP, lowest deductible"
            f"{', or PPO first' if context.get('prefers_ppo') else ''}.\n"
        )

    # ── Context reasoning instructions ────────────────────────────────
    reasoning_block = ""
    if reasoning:
        reasoning_block = f"\nREASONING INSTRUCTIONS (from user preferences):\n{reasoning}\n"

    # ── Verify checklist ──────────────────────────────────────────────
    # Always included. Specific items are expanded based on context flags.
    checklist_items = [
        "Confirm your preferred doctors and specialists are in-network",
        "Check that your pharmacy is in the plan's network",
    ]
    if context.get("has_rx"):
        checklist_items.insert(
            0,
            "Verify your specific prescriptions are on this plan's formulary "
            "(drug list) — call the plan or check Medicare.gov's drug cost tool",
        )
    if context.get("wants_dental"):
        checklist_items.append(
            "Confirm what the dental benefit actually covers "
            "(cleanings only vs. fillings and major work)"
        )
    checklist_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(checklist_items))

    prompt = (
        f"<|think|>{SYSTEM_PROMPT}\n\n"
        f"## MODE: SELECT\n"
        f"The user has completed intake. Use the plan data below to answer "
        f"their question. Discuss the top 5 plans. For each, state "
        f"its monthly premium, MOOP, star rating, plan type, and one sentence "
        f"on why it fits this user's profile.\n\n"
        f"USER PROFILE:\n{profile_text}\n\n"
        f"USER PREFERENCES:\n{context_text}\n"
        f"{sort_instruction}"
        f"{sort_reasoning_block}"
        f"{reasoning_block}\n"
        f"USER PRIORITIES (ranked — these are derived from the user's intake "
        f"answers; anchor every plan explanation and every WHY line to these "
        f"priorities in order, citing specific values from the plan data):\n"
        f"{priorities}\n\n"
        f"AVAILABLE PLANS ({plan_count} shown after filtering and sorting):\n\n"
        f"{plans_text}\n\n"
        f"{history_text}"
        f"{lang_block}\n\n"
        f"User question: {user_question}\n\n"
        f"End your response with a short 'Before you enroll' section listing "
        f"these items the user must verify directly with the plan:\n"
        f"{checklist_text}\n\n"
        f"After the 'Before you enroll' section, output exactly {plan_count} lines "
        f"in this format (no extra text, no bullets, no markdown).\n"
        f"IMPORTANT: Keep the WHY_N: labels exactly as shown in English — "
        f"only the sentence after the colon should be in the response language:\n"
        f"WHY_1: <one sentence anchored to USER PRIORITIES above — "
        f"cite a specific value from Plan 1's data>\n"
        f"WHY_2: <one sentence anchored to USER PRIORITIES above — "
        f"cite a specific value from Plan 2's data>\n"
        f"WHY_3: <one sentence anchored to USER PRIORITIES above — "
        f"cite a specific value from Plan 3's data>\n"
        f"WHY_4: <one sentence anchored to USER PRIORITIES above — "
        f"cite a specific value from Plan 4's data>\n"
        f"WHY_5: <one sentence anchored to USER PRIORITIES above — "
        f"cite a specific value from Plan 5's data>"
    )
    return prompt


# ====================================================================== #
#  Sort reasoning translation prompt                                    #
# ====================================================================== #

def build_prompt_sort_reasoning(reasoning: str, language: str = "English") -> str:
    """
    Translate the English sort_reasoning string into the user's language.
    Only called when language != English. Returns a prompt for call_ollama().
    """
    lang_instr = _language_instruction(language)
    return (
        f"Translate the following sentence into {language}. "
        f"Output only the translated sentence, nothing else:\n\n"
        f"{reasoning}"
        + (f"\n\nLANGUAGE INSTRUCTION: {lang_instr}" if lang_instr else "")
    )


# ====================================================================== #
#  Filter explanation prompt                                            #
# ====================================================================== #

def build_prompt_filter_explanation(filter_summary: str, language: str = "English") -> str:
    """
    Ask Gemma to rewrite the raw FilterDecision.user_summary() log as a
    warm, plain-English paragraph for an elderly user.

    The raw summary looks like:
        Showing 23 of 156 plans in your area after applying your filters:
          • Coverage type filter: MA+D plans only (133 plans removed)
          • Budget ceiling $30/month applied (12 plans removed)
          ⚠ Budget filter removed all plans in your area...

    Gemma rewrites this as 2-3 reassuring sentences that explain the
    benefit of each filter step to the user in plain language.

    Kept under 60 words so the call is fast and the result fits cleanly
    in the filter expander without overwhelming the plan cards.

    Args:
        filter_summary: Output of FilterDecision.user_summary().

    Returns:
        A complete prompt string ready to pass to call_ollama().
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"The plan filtering system produced this log after narrowing the "
        f"CMS plan database to match a user's intake answers:\n\n"
        f"{filter_summary}\n\n"
        f"Rewrite this as 2-3 friendly sentences speaking directly to the user "
        f"(use 'Based on your answers, I...' or similar). Rules:\n"
        f"- Warm and reassuring — the user should feel the filtering helped them, "
        f"not restricted them.\n"
        f"- If plans were removed, explain the benefit "
        f"(e.g. 'This keeps your results within your stated budget').\n"
        f"- If SNP or LIS plans were added, explain why that matters for them.\n"
        f"- No bullet points. No headers. No technical terms like 'filter', "
        f"'DataFrame', 'flag', or 'CSV'.\n"
        f"- Maximum 60 words total."
        + (f"\n\nLANGUAGE INSTRUCTION: {_language_instruction(language)}"
           if _language_instruction(language) else "")
    )
