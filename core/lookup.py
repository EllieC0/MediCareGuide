"""
MediCareGuide Data Lookup Module
=============================
Loads the CMS Landscape CSV, resolves ZIP codes to counties, and filters
plans by the session profile collected during intake.

Architecture position:
    core.session  →  core.lookup  →  core.inference
    (profile{})            (filtered df)         (prompt builder)

Call sequence every SELECT-mode turn:
    1. get_plans(zip)               — geography filter only
    2. get_plans_filtered(zip, profile) — calls get_plans(), then applies
                                         track / SNP / budget filters
    3. sort_plans(df, sort_key, profile) — deterministic sort + transparency label
    4. df.head(5) passed to build_prompt_select_mode()

Decision tracker:
    get_plans_filtered() returns a FilterDecision dataclass alongside the
    DataFrame. It records every filter step — what was applied, how many
    rows were kept or added, and why — so the caller can show the user
    exactly why they are seeing what they are seeing.

Dependencies:
    pip install zipcodes pandas

Usage:
    lookup = MediCareGuideLookup("CY2026_Landscape_202603.csv")

    filtered_df, decision = lookup.get_plans_filtered(
        "24502",
        state["profile"],
    )

    sorted_df, label = sort_plans(filtered_df, state["sort_key"], state["profile"])

    build_prompt_select_mode(question, sorted_df.head(5), state, label)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd
import zipcodes


# ====================================================================== #
#  Decision tracker                                                       #
# ====================================================================== #

@dataclass
class FilterStep:
    """
    Records a single filter step applied inside get_plans_filtered().

    Attributes:
        name        Short identifier, e.g. "track_filter".
        applied     True if this step actually ran (condition was met).
        reason      Why this step ran or was skipped.
        rows_before Row count entering this step.
        rows_after  Row count leaving this step. Equal to rows_before
                    if the step was skipped.
        note        Optional extra detail, e.g. a warning string.
    """
    name:        str
    applied:     bool
    reason:      str
    rows_before: int
    rows_after:  int
    note:        str = ""

    def summary(self) -> str:
        """One-line human-readable summary for printing or logging."""
        status = "applied" if self.applied else "skipped"
        delta  = self.rows_after - self.rows_before
        change = (
            f"  ({delta:+d} rows → {self.rows_after} remaining)"
            if self.applied else
            f"  ({self.rows_before} rows unchanged)"
        )
        note_part = f"  NOTE: {self.note}" if self.note else ""
        return f"[{status:7s}] {self.name}: {self.reason}{change}{note_part}"


@dataclass
class FilterDecision:
    """
    Complete record of every filter decision made by get_plans_filtered().

    Passed back to the caller alongside the filtered DataFrame so that
    test_medicareguide.py can show the user (and the developer) exactly why
    the returned plans look the way they do.

    Attributes:
        zip         ZIP code used for geography lookup.
        steps       Ordered list of FilterStep, one per filter stage.
        total_in    Row count before any profile filtering (after geography).
        total_out   Row count in the returned DataFrame.
        warnings    Any user-visible warnings accumulated across steps.
    """
    zip:       str
    steps:     list[FilterStep] = field(default_factory=list)
    total_in:  int = 0
    total_out: int = 0
    warnings:  list[str] = field(default_factory=list)

    def add_step(self, step: FilterStep) -> None:
        self.steps.append(step)
        if step.note:
            self.warnings.append(step.note)

    def print_summary(self) -> None:
        """Print the full decision trace to stdout — useful for debugging."""
        print(f"\n{'='*60}")
        print(f"  Filter decision for ZIP {self.zip}")
        print(f"  Geography pool: {self.total_in} plans")
        print(f"  Final result:   {self.total_out} plans")
        print(f"{'='*60}")
        for step in self.steps:
            print(f"  {step.summary()}")
        if self.warnings:
            print(f"\n  Warnings:")
            for w in self.warnings:
                print(f"    ! {w}")
        print()

    def user_summary(self) -> str:
        """
        Short paragraph shown to the user explaining what was filtered and why.
        Used by test_medicareguide.py to print before the plan list.
        """
        lines = [
            f"Showing {self.total_out} of {self.total_in} plans in your area "
            f"after applying your filters:"
        ]
        for step in self.steps:
            if step.applied:
                lines.append(f"  • {step.reason} "
                              f"({abs(step.rows_after - step.rows_before)} plans "
                              f"{'added' if step.rows_after > step.rows_before else 'removed'})")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ====================================================================== #
#  Numeric helpers shared by get_plans_filtered and sort_plans           #
# ====================================================================== #

def _to_numeric(series: pd.Series, fallback: float = 9999.0) -> pd.Series:
    """
    Strip dollar signs and commas from a string Series, then cast to float.

    Used for: Monthly Consolidated Premium, MOOP, deductible columns.
    All of these arrive from the CSV as strings like "$1,234.00".

    Args:
        series:   A pandas Series of strings.
        fallback: Value substituted for unparseable entries (NaN after coerce).
                  Default 9999 sorts unknown values to the bottom for
                  ascending sorts (lowest first), and to the bottom for
                  descending sorts after fillna(0) — callers must choose
                  the right fallback for their sort direction.

    Returns:
        Float Series with no NaN values.
    """
    return (
        pd.to_numeric(
            series.str.replace(r"[\$,]", "", regex=True).str.strip(),
            errors="coerce",
        ).fillna(fallback)
    )


# ====================================================================== #
#  Sort labels — single source of truth                                  #
# ====================================================================== #
#
#  Keys match state["sort_key"] values set by core.session.
#  Values are shown to the user verbatim as the sort transparency string.

SORT_LABELS: dict[str, str] = {
    "lowest_premium": (
        "Sorted by lowest monthly premium"
    ),
    "total_cost": (
        "Sorted by estimated annual cost "
        "(monthly premium × 12 + out-of-pocket maximum)"
    ),
    "star_rating": (
        "Sorted by star rating — highest quality first"
    ),
    "lowest_moop": (
        "Sorted by lowest out-of-pocket maximum — "
        "best financial protection if you need a lot of care"
    ),
    "lowest_deductible": (
        "Sorted by lowest Part D drug deductible"
    ),
    "ppo_first": (
        "Sorted by plan type — PPO plans first, then by lowest monthly premium"
    ),
}


# ====================================================================== #
#  Sort function                                                          #
# ====================================================================== #

def sort_plans(
    df: pd.DataFrame,
    sort_key: str,
    profile: dict,
) -> tuple[pd.DataFrame, str]:
    """
    Sort a filtered plan DataFrame deterministically.

    Sort order is always fully determined by Python — Gemma must never
    reorder the plans it receives. The returned label is injected into
    the SELECT-mode prompt so Gemma can cite the sort criteria when
    explaining why Plan 1 appears first.

    Sort hierarchy (applied simultaneously via sort_values):
        1. Sanctioned plans last  (always, regardless of sort_key)
        2. SNP-matching plans first  (only when profile["snp_flags"] set)
        3. Primary sort key  (the user's chosen or default criterion)

    Tie-breaking within identical primary values:
        lowest_premium  → secondary: lowest MOOP
        total_cost      → no secondary needed (computed value, rarely ties)
        star_rating     → secondary: lowest premium
        lowest_moop     → secondary: lowest premium
        lowest_deductible → secondary: lowest premium

    A _rank column (1-based integer) is added to the returned DataFrame.
    _format_plans_as_text() in core.inference.py reads this column
    to label plan blocks as "Plan 1", "Plan 2", etc. in the prompt.

    Args:
        df:       Filtered DataFrame from get_plans_filtered().
        sort_key: Key from SORT_LABELS dict. Defaults to "lowest_premium"
                  for any unrecognised value.
        profile:  state["profile"] dict — used to check snp_flags.

    Returns:
        (sorted_df, label) where label is the SORT_LABELS string for
        the active sort key, optionally appended with an SNP note.
    """
    if df.empty:
        return df, SORT_LABELS.get(sort_key, SORT_LABELS["lowest_premium"])

    df = df.copy()

    # ── Sanctioned flag (always sort last) ───────────────────────────
    # Decision: sanctioned plans may still be legally available but CMS
    # has identified compliance issues. We never hide them (that would
    # be deceptive) but we always rank them below non-sanctioned plans.
    if "Sanctioned Plan" in df.columns:
        df["_sanctioned"] = (
            df["Sanctioned Plan"].str.upper()
            .str.contains("YES", na=False)
            .astype(int)   # 0 = clean, 1 = sanctioned → ascending puts clean first
        )
    else:
        df["_sanctioned"] = 0

    # ── SNP match flag (secondary sort when user has flags) ───────────
    # Decision: if the user flagged a chronic condition or dual eligibility,
    # plans specifically designed for that situation are surfaced first
    # within each sort tier. This is a secondary sort, not a filter —
    # non-SNP plans are still shown below.
    snp_flags = profile.get("snp_flags", [])
    if snp_flags and "SNP Type" in df.columns:
        snp_col  = df["SNP Type"].str.upper()
        is_match = pd.Series(False, index=df.index)
        if "D_SNP" in snp_flags:
            is_match |= snp_col.str.contains("D-SNP", na=False)
        if "C_SNP" in snp_flags:
            is_match |= snp_col.str.contains("C-SNP", na=False)
        # 0 = SNP match (sort first), 1 = no match (sort after)
        df["_snp_first"] = (~is_match).astype(int)
        snp_note = " · Plans designed for your situation shown first"
    else:
        df["_snp_first"] = 0
        snp_note = ""

    # ── Primary sort columns ──────────────────────────────────────────
    prem  = _to_numeric(df["Monthly Consolidated Premium (Part C + D)"],
                        fallback=9999.0)
    # For PDP-only plans the consolidated premium is "Not Applicable".
    # Fall back to "Part D Total Premium" for any row where the
    # consolidated column resolved to the 9999 sentinel.
    if "Part D Total Premium" in df.columns:
        pdp_prem = _to_numeric(df["Part D Total Premium"], fallback=9999.0)
        prem = prem.where(prem < 9999.0, pdp_prem)
    moop  = _to_numeric(df["In-Network Maximum Out-of-Pocket (MOOP) Amount"],
                        fallback=9999.0)

    if sort_key == "lowest_premium":
        # Decision: default sort. Monthly premium is the single most
        # cited factor in CMS beneficiary research. When premiums tie
        # (common with $0 plans), break by MOOP so the plan with better
        # financial protection ranks above its identical-premium peer.
        df["_primary"]   = prem
        df["_secondary"] = moop
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["lowest_premium"]

    elif sort_key == "total_cost":
        # Decision: estimated annual cost = (premium × 12) + MOOP.
        # This is the worst-case cost if the user reaches their MOOP.
        # Most users spend less, but this metric is the most honest
        # apples-to-apples comparison between a $0 premium / high MOOP
        # plan vs a moderate premium / low MOOP plan.
        df["_primary"]   = prem * 12 + moop
        df["_secondary"] = prem   # break ties by premium
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["total_cost"]

    elif sort_key == "star_rating":
        # Decision: CMS Overall Star Rating (1–5). Missing / unrated plans
        # get fillna(0) so they sort below all rated plans when descending.
        # Break ties by premium so the cheaper plan ranks above its same-
        # rated peer.
        stars = pd.to_numeric(
            df["Overall Star Rating"].str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)
        df["_primary"]   = stars
        df["_secondary"] = prem
        ascending_primary   = False  # highest stars first
        ascending_secondary = True   # then lowest premium
        label = SORT_LABELS["star_rating"]

    elif sort_key == "lowest_moop":
        # Decision: lowest out-of-pocket maximum. Best for users who
        # expect heavy utilisation (chronic conditions, planned surgery).
        # Break ties by premium.
        df["_primary"]   = moop
        df["_secondary"] = prem
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["lowest_moop"]

    elif sort_key == "lowest_deductible":
        # Decision: lowest Part D drug deductible. Best for users who
        # take regular prescriptions and want drug costs to start being
        # covered as early as possible in the year.
        deduct = _to_numeric(
            df["Annual Part D Deductible Amount"], fallback=9999.0
        )
        df["_primary"]   = deduct
        df["_secondary"] = prem
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["lowest_deductible"]

    elif sort_key == "ppo_first":
        # Decision: user prefers PPO flexibility. Assign a numeric priority
        # to each plan type so PPO plans sort above HMO plans.
        # Within each plan-type tier, break ties by lowest premium.
        if "Plan Type" in df.columns:
            def _ppo_priority(pt: str) -> int:
                pt = pt.upper()
                if "PPO" in pt and "POS" not in pt:
                    return 0   # pure PPO first
                if "PPO" in pt and "POS" in pt:
                    return 1   # PPO-POS second
                if "HMO" in pt and "POS" not in pt:
                    return 2   # pure HMO third
                if "HMO" in pt and "POS" in pt:
                    return 3   # HMO-POS fourth
                return 4       # all other types last
            df["_primary"] = df["Plan Type"].apply(_ppo_priority)
        else:
            df["_primary"] = 0
        df["_secondary"]    = prem
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["ppo_first"]

    else:
        # Unknown sort key — fall back to default without raising.
        # Decision: safe fallback prevents a bad state["sort_key"] value
        # from crashing the pipeline. The label tells the user what
        # actually happened.
        df["_primary"]   = prem
        df["_secondary"] = moop
        ascending_primary   = True
        ascending_secondary = True
        label = SORT_LABELS["lowest_premium"]

    # ── Apply the sort ────────────────────────────────────────────────
    df = df.sort_values(
        by=["_sanctioned", "_snp_first", "_primary", "_secondary"],
        ascending=[True, True, ascending_primary, ascending_secondary],
        kind="stable",   # stable sort preserves CSV row order for equal values
    )

    # ── Add visible rank column ───────────────────────────────────────
    # _rank is read by _format_plans_as_text() to label blocks in the prompt.
    # It is dropped from DISPLAY_COLUMNS so Gemma never sees the raw integer.
    df["_rank"] = range(1, len(df) + 1)

    # ── Drop internal sort columns ────────────────────────────────────
    df = df.drop(columns=["_sanctioned", "_snp_first", "_primary", "_secondary"])

    return df, label + snp_note


# ====================================================================== #
#  Auto sort derivation                                                   #
# ====================================================================== #

def derive_sort_key(profile: dict, context: dict) -> tuple[str, str]:
    """
    Derive the best sort key and a plain-English reasoning string from
    the completed session profile and context flags.

    Called by handle_turn() when state["sort_key"] is None (no manual
    override). Returns (sort_key, reasoning) where reasoning is shown
    to the user and passed to Gemma to open its response.

    Priority order (first match wins):
        1. PDP track          → lowest_deductible  (drug plan, deductible is key)
        2. has_rx + C_SNP     → lowest_moop        (chronic + Rx = heavy utilisation)
        3. has_rx + budget    → total_cost         (within budget, honest annual picture)
        4. has_rx             → lowest_deductible  (minimise drug cost barrier)
        5. C_SNP              → lowest_moop        (chronic = high utilisation)
        6. budget_max         → lowest_premium     (cheapest within budget)
        7. D_SNP              → lowest_premium     (SNP match handles prioritisation)
        8. prefers_ppo        → ppo_first          (PPO plans surfaced above HMO)
        9. no signals         → star_rating        (quality-first default)
    """
    has_rx      = context.get("has_rx", False)
    prefers_ppo = context.get("prefers_ppo", False)
    snp_flags   = profile.get("snp_flags", [])
    budget_max  = profile.get("budget_max")
    track       = profile.get("track")
    c_snp       = "C_SNP" in snp_flags
    d_snp       = "D_SNP" in snp_flags

    if track == "PDP":
        return (
            "lowest_deductible",
            "You chose a standalone Part D drug plan, so plans are sorted by "
            "lowest drug deductible — minimising what you pay before coverage "
            "kicks in for your prescriptions.",
        )

    if has_rx and c_snp:
        return (
            "lowest_moop",
            "You have a chronic condition and take regular prescriptions. "
            "Plans are sorted by lowest out-of-pocket maximum to protect you "
            "against high costs from frequent care and medications.",
        )

    if has_rx and budget_max is not None:
        return (
            "total_cost",
            f"You take regular prescriptions and set a budget of "
            f"\${budget_max}/month. Plans are sorted by estimated annual cost "
            f"(premium \u00d7 12 + out-of-pocket maximum) for the most honest "
            f"picture of what you will spend.",
        )

    if has_rx:
        return (
            "lowest_deductible",
            "You take regular prescriptions. Plans are sorted by lowest drug "
            "deductible so your medications start being covered as early as "
            "possible in the year.",
        )

    if c_snp:
        return (
            "lowest_moop",
            "You have a chronic condition. Plans are sorted by lowest "
            "out-of-pocket maximum to protect you against high costs from "
            "frequent care.",
        )

    if budget_max is not None:
        return (
            "lowest_premium",
            f"You set a budget of \${budget_max}/month. Plans are sorted by "
            f"lowest monthly premium within your budget.",
        )

    if d_snp:
        return (
            "lowest_premium",
            "Plans designed for dual-eligible beneficiaries are shown first, "
            "then sorted by lowest monthly premium.",
        )

    if prefers_ppo:
        return (
            "ppo_first",
            "You prefer PPO flexibility. PPO plans are shown before HMO plans, "
            "then sorted by lowest premium within each plan type.",
        )

    return (
        "star_rating",
        "No specific cost or coverage needs were identified, so plans are "
        "sorted by star rating — highest quality first — then by lowest "
        "premium within the same rating.",
    )


# ====================================================================== #
#  Main class                                                             #
# ====================================================================== #

class MediCareGuideLookup:
    """
    Medicare plan lookup engine.

    Loads the CMS Landscape CSV once at startup, keeps everything in
    memory, and answers queries by ZIP code.

    Call sequence (SELECT mode):

        filtered_df, decision = lookup.get_plans_filtered(zip, profile)
        sorted_df,   label    = sort_plans(filtered_df, sort_key, profile)
        prompt = build_prompt_select_mode(q, sorted_df.head(5), state, label)

    get_plans() is called internally by get_plans_filtered() and is also
    available directly for debugging or for future modules that need the
    unfiltered geography pool.
    """

    # ------------------------------------------------------------------ #
    #  County suffix normalisation                                        #
    # ------------------------------------------------------------------ #
    #
    #  The zipcodes library returns county names like "Campbell County".
    #  The CMS Landscape CSV uses bare names like "Campbell".
    #  We strip these suffixes so the two sources can be matched.
    #
    #  Decision: kept as a class attribute (not a local variable) so that
    #  subclasses or tests can override it without touching the method.

    COUNTY_SUFFIXES: list[str] = [
        " County",
        " Parish",        # Louisiana
        " Borough",       # Alaska
        " Municipality",  # Alaska
        " city",          # Virginia independent cities (lowercase 'c')
        " City",          # Virginia independent cities (uppercase 'C')
    ]

    def __init__(self, landscape_path: str) -> None:
        """
        Load the Landscape CSV once into memory.

        Decision: dtype=str on all columns prevents pandas from silently
        coercing Plan ID "001" → 1, or a premium "$0.00" → float before
        we have a chance to handle the dollar sign ourselves.

        Decision: strip() on all string columns at load time. The CMS CSV
        has trailing spaces on several fields (notably premium columns).
        Stripping once at load is cheaper than stripping per-query.

        Args:
            landscape_path: Path to the CY2026_Landscape CSV file.
        """
        self.landscape = pd.read_csv(landscape_path, dtype=str)
        self.landscape = self.landscape.apply(
            lambda col: col.str.strip() if col.dtype == object else col
        )
        print(f"Loaded {len(self.landscape)} rows from {landscape_path}.")

    # ------------------------------------------------------------------ #
    #  STEP 1 — ZIP → county/state                                        #
    # ------------------------------------------------------------------ #

    def _resolve_zip(self, zipcode: str) -> list[dict] | None:
        """
        Convert a ZIP code string to all matching state + county pairs.

        A single ZIP can legally span multiple counties (e.g. a ZIP on a
        county border). All matches are returned so get_plans() can build
        an OR-mask across every county the ZIP touches.

        Decision: return None (not []) for invalid input so callers can
        distinguish "invalid ZIP" from "valid ZIP with no CMS plans".

        Decision: deduplicate by (state.upper(), county.upper()) key.
        The zipcodes library can return the same county multiple times
        with different city names within it.

        Args:
            zipcode: A 5-digit string, e.g. "24502".

        Returns:
            List of {"state": "VA", "county": "Campbell", "city": "Lynchburg"}
            dicts, or None if the ZIP is not found / not numeric.
        """
        try:
            results = zipcodes.matching(zipcode)
        except (ValueError, TypeError):
            return None

        if not results:
            return None

        locations: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for r in results:
            county = r.get("county", "")
            for suffix in self.COUNTY_SUFFIXES:
                county = county.replace(suffix, "")
            county = county.strip()

            state = r.get("state", "")
            key   = (state.upper(), county.upper())
            if key not in seen:
                seen.add(key)
                locations.append({
                    "state":  state,
                    "county": county,
                    "city":   r.get("city", ""),
                })

        return locations

    # ------------------------------------------------------------------ #
    #  STEP 2 — Geography filter                                          #
    # ------------------------------------------------------------------ #

    def get_plans(self, zipcode: str) -> pd.DataFrame:
        """
        Return all Medicare plans available in a ZIP code's geography.

        Applies geography filtering only — no profile, no track, no budget.
        This is the base pool that get_plans_filtered() narrows further.

        Decision: include both county-specific rows AND statewide
        "All Counties" rows. CMS uses "All Counties" for plans that serve
        an entire state (common for PDPs). Without this, PDP plans would
        be invisible to users in any county.

        Decision: deduplicate by Contract ID + Plan ID + Segment ID.
        When a ZIP spans multiple counties, a statewide plan matched from
        each county would otherwise appear multiple times.

        Args:
            zipcode: 5-digit ZIP string.

        Returns:
            DataFrame of eligible plans, or empty DataFrame if ZIP is
            invalid or no plans exist for that geography.
        """
        locations = self._resolve_zip(zipcode)
        if not locations:
            return pd.DataFrame()

        state_col  = self.landscape["State Territory Abbreviation"].str.upper()
        county_col = self.landscape["County Name"].str.upper()

        mask = pd.Series(False, index=self.landscape.index)
        for loc in locations:
            state  = loc["state"].upper()
            county = loc["county"].upper()
            loc_mask = state_col.eq(state) & (
                county_col.eq(county) | county_col.eq("ALL COUNTIES")
            )
            mask = mask | loc_mask

        plans = self.landscape[mask].copy()

        id_cols = [
            c for c in ("Contract ID", "Plan ID", "Segment ID")
            if c in plans.columns
        ]
        plans = (
            plans.drop_duplicates(subset=id_cols) if id_cols
            else plans.drop_duplicates()
        )

        return plans

    # ------------------------------------------------------------------ #
    #  STEP 3 — Profile filter                                            #
    # ------------------------------------------------------------------ #

    def get_plans_filtered(
        self,
        zipcode: str,
        profile: dict,
    ) -> tuple[pd.DataFrame, FilterDecision]:
        """
        Return plans filtered by the session profile collected during intake.

        Applies three filter stages in sequence on top of get_plans():
            1. Track filter   — narrows by coverage type
            2. SNP filter     — adds back plans for the user's situation
            3. Budget filter  — removes plans above the premium ceiling

        Each stage is recorded in the FilterDecision object so callers
        can explain to the user exactly what was applied and why.

        IMPORTANT — SNP filter adds rows, does not restrict them:
            A D-SNP plan has Contract Category Type = "SNP", not "MA".
            If we only applied the track filter, dual-eligible users
            would never see the plans designed specifically for them.
            So after the track filter narrows the pool, the SNP filter
            pulls matching SNP rows from the full geography pool and
            unions them back in.

        Args:
            zipcode: 5-digit ZIP string.
            profile: state["profile"] dict from MediCareGuideSession.
                     Relevant keys: track, snp_flags, budget_max.

        Returns:
            (filtered_df, decision) where decision is a FilterDecision
            recording every step taken and any warnings generated.
        """
        decision = FilterDecision(zip=zipcode)

        # ── Geography base pool ───────────────────────────────────────
        all_plans = self.get_plans(zipcode)

        if all_plans.empty:
            decision.total_in  = 0
            decision.total_out = 0
            decision.add_step(FilterStep(
                name        = "geography",
                applied     = False,
                reason      = f"No plans found for ZIP {zipcode}",
                rows_before = 0,
                rows_after  = 0,
                note        = "Verify the ZIP code is correct.",
            ))
            return all_plans, decision

        decision.total_in = len(all_plans)
        df = all_plans.copy()

        # ── Filter 1: Track ───────────────────────────────────────────
        #
        #  Decision: the session track value maps to CMS column combinations
        #  as follows:
        #
        #    "MA_D"  → Contract Category Type contains "MA" or "MAPD"
        #              AND Part D Coverage Indicator = "Yes"
        #              Rationale: user wants a bundled MA plan that includes
        #              drug coverage. Plans labelled "MAPD" explicitly bundle
        #              both; plans labelled "MA" with Part D = "Yes" are
        #              equivalent. Both are included.
        #
        #    "PDP"   → Contract Category Type contains "PDP"
        #              Rationale: user wants a standalone drug plan to pair
        #              with Original Medicare. PDP plans only.
        #
        #    "MEDIGAP" → never reaches this method (core.session keeps
        #              mode = EDUCATE for MEDIGAP and issues a referral).
        #              Included as a guard in case the caller bypasses session.
        #
        #    None / unknown → no track filter applied. Return full pool.
        #              Rationale: if the user hasn't chosen a track yet,
        #              we should not be in SELECT mode. But if get_plans_filtered
        #              is called anyway, showing everything is safer than
        #              showing nothing.

        track       = profile.get("track")
        rows_before = len(df)

        if track == "MA_D":
            cat_col    = df["Contract Category Type"].str.upper()
            part_d_col = df["Part D Coverage Indicator"].str.upper()
            # "MA" and "MAPD" both satisfy the track — see decision note above
            df = df[
                cat_col.str.contains("MA", na=False) &
                part_d_col.str.contains("YES", na=False)
            ]
            decision.add_step(FilterStep(
                name        = "track_filter",
                applied     = True,
                reason      = (
                    "Track = Medicare Advantage with drug coverage (MA_D). "
                    "Kept plans where Contract Category Type contains 'MA' "
                    "and Part D Coverage Indicator = 'Yes'."
                ),
                rows_before = rows_before,
                rows_after  = len(df),
            ))

        elif track == "PDP":
            cat_col = df["Contract Category Type"].str.upper()
            df = df[cat_col.str.contains("PDP", na=False)]
            decision.add_step(FilterStep(
                name        = "track_filter",
                applied     = True,
                reason      = (
                    "Track = standalone Part D drug plan (PDP). "
                    "Kept plans where Contract Category Type contains 'PDP'."
                ),
                rows_before = rows_before,
                rows_after  = len(df),
            ))

        elif track == "MEDIGAP":
            # Should never arrive here — session blocks SELECT for MEDIGAP.
            # Return empty with an explanatory step.
            decision.add_step(FilterStep(
                name        = "track_filter",
                applied     = False,
                reason      = (
                    "Track = MEDIGAP. Medigap plans are not in the CMS "
                    "Landscape file. No Landscape filtering is possible."
                ),
                rows_before = rows_before,
                rows_after  = 0,
                note        = (
                    "Refer user to medicare.gov/find-a-plan for Medigap comparison."
                ),
            ))
            decision.total_out = 0
            return pd.DataFrame(), decision

        else:
            # No track set or unrecognised value — skip filter, log it.
            decision.add_step(FilterStep(
                name        = "track_filter",
                applied     = False,
                reason      = (
                    f"Track = {track!r}. No track filter applied — "
                    "returning full geography pool."
                ),
                rows_before = rows_before,
                rows_after  = rows_before,
            ))

        # ── Filter 2: SNP flags ───────────────────────────────────────
        #
        #  Decision: SNP plans have Contract Category Type = "SNP", which
        #  is DIFFERENT from "MA". This means the track filter above will
        #  have excluded them from the pool entirely if track = "MA_D".
        #
        #  To correct this, we pull matching SNP rows from the FULL
        #  geography pool (all_plans, before any track filtering) and
        #  union them back into df. This ensures dual-eligible users see
        #  D-SNP plans, and users with chronic conditions see C-SNP plans,
        #  regardless of which track they chose.
        #
        #  This is an ADDITIVE step (rows can only increase), not a
        #  restrictive one. Non-SNP plans are preserved below SNP plans
        #  in the sort order.
        #
        #  LIS flag: Low Income Subsidy (Extra Help) plans are not a
        #  separate plan type — they are standard MA or PDP plans that
        #  auto-enroll LIS-eligible beneficiaries. We surface them by
        #  checking the LIS Auto Enrollment column and ensuring those rows
        #  are present in the result.

        snp_flags   = profile.get("snp_flags", [])
        rows_before = len(df)

        if snp_flags:
            snp_col = all_plans["SNP Type"].str.upper()
            lis_col = all_plans["Low Income Subsidy (LIS) Auto Enrollment"].str.upper()

            extra_mask = pd.Series(False, index=all_plans.index)
            flag_notes: list[str] = []

            if "D_SNP" in snp_flags:
                extra_mask |= snp_col.str.contains("D-SNP", na=False)
                flag_notes.append("D-SNP (dual eligible)")

            if "C_SNP" in snp_flags:
                extra_mask |= snp_col.str.contains("C-SNP", na=False)
                flag_notes.append("C-SNP (chronic condition)")

            if "LIS" in snp_flags:
                extra_mask |= lis_col.str.contains("YES", na=False)
                flag_notes.append("LIS auto-enrollment")

            extra_df = all_plans[extra_mask].copy()

            # Union with track-filtered pool; deduplicate by plan identity
            id_cols = [
                c for c in ("Contract ID", "Plan ID", "Segment ID")
                if c in df.columns
            ]
            df = (
                pd.concat([df, extra_df], ignore_index=True)
                .drop_duplicates(subset=id_cols) if id_cols
                else pd.concat([df, extra_df], ignore_index=True).drop_duplicates()
            )

            added = len(df) - rows_before
            decision.add_step(FilterStep(
                name        = "snp_filter",
                applied     = True,
                reason      = (
                    f"SNP flags detected: {', '.join(flag_notes)}. "
                    f"Added matching SNP/LIS plans from geography pool "
                    f"(pulled from full pool before track filter to avoid "
                    f"missing SNP-typed plans)."
                ),
                rows_before = rows_before,
                rows_after  = len(df),
                note        = (
                    f"{added} SNP/LIS plan(s) added back to results."
                    if added > 0 else ""
                ),
            ))
        else:
            decision.add_step(FilterStep(
                name        = "snp_filter",
                applied     = False,
                reason      = "No SNP or LIS flags in profile. Step skipped.",
                rows_before = rows_before,
                rows_after  = rows_before,
            ))

        # ── Filter 3: Budget ceiling ──────────────────────────────────
        #
        #  Decision: filter on "Monthly Consolidated Premium (Part C + D)"
        #  which is the single combined premium figure from the Landscape.
        #  This is the most meaningful cost signal at the plan-comparison
        #  stage — Part B premium is fixed for everyone and excluded.
        #
        #  Decision: if the budget filter would eliminate ALL remaining
        #  plans, skip it and warn instead of returning an empty result.
        #  An empty result gives the user nothing to work with; a full
        #  result with a clear warning lets them see what's available and
        #  decide whether to raise their budget.
        #
        #  Decision: plans with unparseable premium values (e.g. missing
        #  or non-numeric) are KEPT when budget filtering. Dropping them
        #  would silently hide plans the user might want — the sort will
        #  push them to the bottom anyway via fallback=9999.

        budget_max  = profile.get("budget_max")
        rows_before = len(df)

        if budget_max is not None:
            prem_col = "Monthly Consolidated Premium (Part C + D)"
            numeric_prem = _to_numeric(df[prem_col], fallback=9999.0)
            budget_mask  = numeric_prem <= float(budget_max)
            filtered     = df[budget_mask]

            if filtered.empty:
                # Skip filter; warn instead of returning nothing
                decision.add_step(FilterStep(
                    name        = "budget_filter",
                    applied     = False,
                    reason      = (
                        f"Budget ceiling = \${budget_max}/month. "
                        f"No plans found at or below this premium."
                    ),
                    rows_before = rows_before,
                    rows_after  = rows_before,
                    note        = (
                        f"No plans found under \${budget_max}/month after other "
                        f"filters. Showing all {rows_before} plan(s) without the "
                        f"budget ceiling so you can see what's available."
                    ),
                ))
            else:
                df = filtered
                decision.add_step(FilterStep(
                    name        = "budget_filter",
                    applied     = True,
                    reason      = (
                        f"Budget ceiling = \${budget_max}/month. "
                        f"Kept plans with Monthly Consolidated Premium "
                        f"≤ \${budget_max}."
                    ),
                    rows_before = rows_before,
                    rows_after  = len(df),
                ))
        else:
            decision.add_step(FilterStep(
                name        = "budget_filter",
                applied     = False,
                reason      = "No budget ceiling in profile. Step skipped.",
                rows_before = rows_before,
                rows_after  = rows_before,
            ))

        decision.total_out = len(df)
        return df, decision


# ====================================================================== #
#  Quick test — run this file directly                                    #
# ====================================================================== #

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/CY2026_Landscape_202603.csv"
    test_zip = sys.argv[2] if len(sys.argv) > 2 else "24502"

    print(f"\n{'='*60}")
    print(f"  MediCareGuide Lookup — Quick Test")
    print(f"{'='*60}\n")

    lookup = MediCareGuideLookup(csv_path)

    # ── Test 1: ZIP resolution ─────────────────────────────────────────
    print("--- ZIP resolution ---")
    locs = lookup._resolve_zip(test_zip)
    if locs:
        for loc in locs:
            print(f"  {test_zip} → state={loc['state']}, "
                  f"county={loc['county']}, city={loc['city']}")
    else:
        print(f"  ZIP {test_zip} not found.")

    # ── Test 2: Geography pool ─────────────────────────────────────────
    print("\n--- Geography pool (get_plans) ---")
    all_plans = lookup.get_plans(test_zip)
    print(f"  Found {len(all_plans)} plans.")
    if not all_plans.empty and "Contract Category Type" in all_plans.columns:
        print("  By contract category:")
        for cat, count in all_plans["Contract Category Type"].value_counts().items():
            print(f"    {cat}: {count}")

    # ── Test 3: Profile-filtered (MA+D, budget $50) ────────────────────
    print("\n--- Profile filter: MA_D + budget $50 ---")
    profile_a = {
        "track":      "MA_D",
        "snp_flags":  [],
        "budget_max": 50,
    }
    df_a, dec_a = lookup.get_plans_filtered(test_zip, profile_a)
    dec_a.print_summary()
    print(f"  Returned {len(df_a)} plans.")

    # ── Test 4: Profile-filtered (MA+D + C_SNP, no budget) ────────────
    print("\n--- Profile filter: MA_D + C_SNP flag, no budget ---")
    profile_b = {
        "track":      "MA_D",
        "snp_flags":  ["C_SNP"],
        "budget_max": None,
    }
    df_b, dec_b = lookup.get_plans_filtered(test_zip, profile_b)
    dec_b.print_summary()
    print(f"  Returned {len(df_b)} plans.")

    # ── Test 5: Sort ───────────────────────────────────────────────────
    if not df_a.empty:
        print("\n--- Sort test: lowest_premium on filtered set ---")
        sorted_df, label = sort_plans(df_a, "lowest_premium", profile_a)
        print(f"  Label: {label}")
        cols = [
            "Plan Name",
            "Monthly Consolidated Premium (Part C + D)",
            "In-Network Maximum Out-of-Pocket (MOOP) Amount",
            "Overall Star Rating",
            "_rank",
        ]
        show_cols = [c for c in cols if c in sorted_df.columns]
        print(sorted_df[show_cols].head(5).to_string(index=False))