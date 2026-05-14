import re

MEDIGAP_REFERRAL = (
    "Medigap (Medicare Supplement) plans aren't in my local database — "
    "they're sold directly by private insurers and aren't part of the "
    "CMS Landscape file I use.\n\n"
    "To compare Medigap options in your area, visit: **medicare.gov/find-a-plan**\n\n"
    "I can still answer general questions about how Medigap works, "
    "what the different plan letters mean (G, N, etc.), or how it "
    "compares to Medicare Advantage — just ask."
)

SORT_BUTTON_LABELS: dict[str, str] = {
    "lowest_premium":    "Lowest Premium",
    "total_cost":        "Total Annual Cost",
    "star_rating":       "Star Rating",
    "lowest_moop":       "Lowest MOOP",
    "lowest_deductible": "Lowest Deductible",
    "ppo_first":         "PPO First",
}

def _extract_sources(text: str) -> "str | None":
    """Extract the SOURCES: line appended by Gemma when RAG passages were provided."""
    m = re.search(r'\nSOURCES:\s*(.+)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return None if val.lower() == "none" else val
    return None

def _strip_sources(text: str) -> str:
    """Remove the SOURCES: line from Gemma's response before storing or displaying."""
    return re.sub(r'\nSOURCES:[^\n]*', '', text, flags=re.IGNORECASE).strip()

def _parse_plan_whys(analysis: str, n: int = 5) -> list[str]:
    """
    Extract WHY_1: … WHY_N: … lines appended by Gemma at the end of select_analysis.
    Returns a list of n strings (empty string = not found, caller falls back).
    """
    if not analysis:
        return [""] * n
    result = [""] * n
    for m in re.finditer(r'WHY_(\d):[ \t]*(.+)', analysis):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            result[idx] = m.group(2).strip()
    return result

def _strip_why_lines(analysis: str) -> str:
    """Remove the WHY_N: marker lines from Gemma's analysis before display."""
    return re.sub(r'\nWHY_\d:[^\n]*', '', analysis).strip()
