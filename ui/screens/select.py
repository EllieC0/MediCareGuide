import streamlit as st
import pandas as pd
from ui.components import (
    _t, render_language_selector, render_chat_history,
    render_voice_input, maybe_play_audio, safe_md
)
from ui.utils import (
    _extract_sources, _strip_sources, _parse_plan_whys, _strip_why_lines,
    MEDIGAP_REFERRAL, SORT_BUTTON_LABELS
)
from ui.state import _save_session_to_disk, _clear_saved_session
from ui.backend import get_lookup
from core.lookup import sort_plans, derive_sort_key, SORT_LABELS
from core.inference import (
    build_prompt_select_mode, build_prompt_filter_explanation,
    build_prompt_sort_reasoning
)
from core.ollama import call_ollama, split_system_and_user

_PDP_MOOP_NOTE  = (
    "Part D drug plans do not have a MOOP — they cover only prescriptions, "
    "not medical or hospital costs. MOOP applies to Medicare Advantage plans only."
)
_PDP_STARS_NOTE = (
    "CMS star ratings are not published for Part D drug-only plans in this dataset."
)

def _plan_premium(row: "pd.Series") -> str:
    consolidated = row.get("Monthly Consolidated Premium (Part C + D)", "")
    if str(consolidated).strip().lower() in ("", "nan", "n/a", "not applicable"):
        pdp = row.get("Part D Total Premium", "")
        if str(pdp).strip().lower() not in ("", "nan", "n/a", "not applicable"):
            return str(pdp)
    return str(consolidated) if str(consolidated).strip() else "—"

def _is_pdp(row: "pd.Series") -> bool:
    plan_type = str(row.get("Plan Type", "")).upper()
    cat       = str(row.get("Contract Category Type", "")).upper()
    return "PDP" in plan_type or cat.startswith("PDP")

def _why_recommended(row: "pd.Series", rank: int, sort_key: str | None) -> str:
    prem  = _plan_premium(row)
    pdp   = _is_pdp(row)
    parts = []
    key = sort_key or "lowest_premium"

    if key == "lowest_premium":
        if prem.replace("$", "").replace(".00", "").strip() == "0":
            parts.append("$0 monthly premium")
        else:
            parts.append(f"Premium: {prem}/mo")
    elif key == "total_cost":
        parts.append("Lowest est. annual cost (premium + MOOP)")
    elif key == "star_rating":
        stars = row.get("Overall Star Rating", "")
        if stars and str(stars).lower() not in ("not applicable", "n/a", "nan", ""):
            parts.append(f"{stars}-star CMS rating")
        else:
            parts.append("Sorted by star rating")
    elif key == "lowest_moop":
        moop = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "")
        if not pdp and moop and str(moop).lower() not in ("not applicable", "n/a"):
            parts.append(f"MOOP: {moop}")
        elif pdp:
            parts.append("PDP — no MOOP applies")
    elif key == "lowest_deductible":
        ded = row.get("Annual Part D Deductible Amount", "")
        if ded and str(ded).lower() not in ("not applicable", "n/a", "nan"):
            if ded.replace("$", "").replace(".00", "").strip() == "0":
                parts.append("$0 drug deductible")
            else:
                parts.append(f"Drug deductible: {ded}")
    elif key == "ppo_first":
        ptype = row.get("Plan Type", "")
        parts.append(f"Plan type: {ptype}")

    if rank == 1:
        parts.insert(0, "Top pick")

    return " · ".join(parts) if parts else f"#{rank} overall match"

def _generate_summary_html() -> str:
    from datetime import date
    session  = st.session_state.session
    profile  = session.state["profile"]
    context  = session.state["context"]
    sdf      = st.session_state.sorted_df
    sort_reasoning = st.session_state.sort_reasoning or st.session_state.sort_label
    gemma_whys     = _parse_plan_whys(st.session_state.get("select_analysis", ""))

    loc = profile.get("zip", "")
    if profile.get("county") and profile.get("state"):
        loc += f" ({profile['county']}, {profile['state']})"
    track_map = {
        "MA_D":    "Medicare Advantage (Part C + D)",
        "PDP":     "Part D Drug Plan Only",
        "MEDIGAP": "Original Medicare + Medigap",
    }
    track  = track_map.get(profile.get("track", ""), profile.get("track", "—"))
    budget = f"${profile['budget_max']}/month" if profile.get("budget_max") is not None else "No limit"

    prefs = []
    if context.get("has_rx"):        prefs.append("Takes regular prescriptions")
    if context.get("keep_doctors"):  prefs.append("Wants to keep current doctors")
    if context.get("wants_dental"):  prefs.append("Dental / vision / hearing benefits important")
    if context.get("prefers_ppo"):   prefs.append("Prefers PPO flexibility")
    prefs_li = "".join(f"<li>{p}</li>" for p in prefs) if prefs else "<li>None stated</li>"

    checklist = ["Confirm your preferred doctors and specialists are in-network", "Check that your pharmacy is in the plan's network"]
    if context.get("has_rx"):
        checklist.insert(0, "Verify your specific prescriptions are on this plan's formulary (drug list) — call the plan or check Medicare.gov's drug cost tool")
    if context.get("wants_dental"):
        checklist.append("Confirm what the dental benefit actually covers (cleanings only vs. fillings and major work)")
    checklist_li = "".join(f"<li>{item}</li>" for item in checklist)

    plan_cards_html = ""
    for i, (_, row) in enumerate(sdf.head(5).iterrows()):
        name      = row.get("Plan Name", f"Plan {i + 1}")
        premium   = _plan_premium(row)
        moop      = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
        stars     = row.get("Overall Star Rating", "—")
        ptype     = row.get("Plan Type", "—")
        deduct    = row.get("Annual Part D Deductible Amount", "—")
        insurer   = row.get("Organization Marketing Name", "—")
        why       = (gemma_whys[i] if i < len(gemma_whys) and gemma_whys[i] else _why_recommended(row, i + 1, st.session_state.sort_key))
        bg        = "#f7f9fc" if i % 2 == 0 else "#ffffff"

        plan_cards_html += f"""
        <div style="background:{bg};border:1px solid #dde4ee;border-radius:8px; padding:16px 20px;margin-bottom:14px;">
          <div style="font-size:1.05rem;font-weight:700;color:#003366;margin-bottom:10px;">#{i + 1} &nbsp; {name}</div>
          <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">
            <tr><td style="padding:3px 10px;color:#555;width:22%;">Insurer</td><td style="padding:3px 10px;font-weight:600;width:28%;">{insurer}</td><td style="padding:3px 10px;color:#555;width:22%;">Plan type</td><td style="padding:3px 10px;font-weight:600;">{ptype}</td></tr>
            <tr><td style="padding:3px 10px;color:#555;">Monthly premium</td><td style="padding:3px 10px;font-weight:600;color:#0055aa;">{premium}</td><td style="padding:3px 10px;color:#555;">Star rating</td><td style="padding:3px 10px;font-weight:600;">{stars}</td></tr>
            <tr><td style="padding:3px 10px;color:#555;">Max out-of-pocket</td><td style="padding:3px 10px;font-weight:600;">{moop}</td><td style="padding:3px 10px;color:#555;">Drug deductible</td><td style="padding:3px 10px;font-weight:600;">{deduct}</td></tr>
          </table>
          <div style="margin-top:10px;padding:8px 12px;background:#eef4ff; border-left:3px solid #0055aa;border-radius:4px; font-size:0.85rem;color:#1a1a1a;"><strong>Why it fits you:</strong> {why}</div>
        </div>"""

    today = date.today().strftime("%B %d, %Y")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>My Medicare Plan Summary — MediGuide</title><style>body {{ font-family: Georgia, serif; font-size: 16px; color: #1a1a1a; max-width: 820px; margin: 0 auto; padding: 32px 24px; }} h1 {{ color: #003366; font-size: 1.7rem; margin-bottom: 2px; }} h2 {{ color: #003366; font-size: 1.1rem; margin: 22px 0 10px; border-bottom: 2px solid #003366; padding-bottom: 4px; }} .meta {{ font-size: 0.82rem; color: #666; margin-bottom: 22px; }} .profile-grid {{ display:grid; grid-template-columns:160px 1fr; gap:6px 0; background:#f4f7fb; border-radius:8px; padding:14px 18px; font-size:0.9rem; }} .label {{ color:#555; }} .value {{ font-weight:600; }} .sort-note {{ background:#e8f0fb; border-left:4px solid #003366; border-radius:6px; padding:10px 14px; font-size:0.9rem; margin-bottom:16px; }} .checklist {{ background:#fff8e6; border:1px solid #f0d080; border-radius:8px; padding:14px 20px; }} .checklist li {{ margin-bottom:6px; font-size:0.9rem; }} .footer {{ margin-top:30px; font-size:0.76rem; color:#888; border-top:1px solid #dde4ee; padding-top:12px; }} @media print {{ body {{ padding:16px; }} }}</style></head><body><h1>🧓 My Medicare Plan Summary</h1><div class="meta">Generated by MediGuide on {today} · Powered by Gemma 4 AI · Data: CMS 2026 Landscape</div><h2>Your Profile</h2><div class="profile-grid"><span class="label">Location</span><span class="value">{loc}</span><span class="label">Coverage type</span><span class="value">{track}</span><span class="label">Monthly budget</span><span class="value">{budget}</span><span class="label">Preferences</span><span class="value"><ul style="margin:0;padding-left:16px;">{prefs_li}</ul></span></div><h2>Why These Plans Appear First</h2><div class="sort-note">{sort_reasoning}</div><h2>Your Top 5 Plans</h2>{plan_cards_html}<h2>⚠ Before You Enroll — Verify These Directly with the Plan</h2><div class="checklist"><ol>{checklist_li}</ol></div><div class="footer">This summary was generated by MediGuide, a free offline Medicare plan advisor powered by Gemma 4 AI running locally on your computer. Plan data is from the CMS 2026 Medicare Landscape file. This is not medical or legal advice. Always verify details directly with the plan before enrolling. Visit <strong>medicare.gov</strong> for official enrollment information.</div></body></html>"""

def handle_select_turn(user_input: str) -> str:
    session = st.session_state.session
    lookup  = get_lookup()
    state   = session.process_turn(user_input)
    profile = state["profile"]

    if profile.get("track") == "MEDIGAP":
        session.close_turn(MEDIGAP_REFERRAL)
        return MEDIGAP_REFERRAL

    filtered_df, _ = lookup.get_plans_filtered(profile["zip"], profile)
    manual_key = st.session_state.sort_key
    if manual_key:
        sort_key      = manual_key
        sort_reasoning = f"Sorted by: {SORT_LABELS.get(sort_key, sort_key)}"
    else:
        sort_key, sort_reasoning = derive_sort_key(profile, state["context"])

    sorted_df, sort_label = sort_plans(filtered_df, sort_key, profile)
    prompt = build_prompt_select_mode(
        user_question  = user_input,
        plans          = sorted_df.head(5),
        state          = state,
        sort_label     = sort_label,
        sort_reasoning = sort_reasoning,
        language       = st.session_state.get("language", "English"),
    )
    system_part, user_part = split_system_and_user(prompt)
    with st.spinner("Thinking…"):
        answer = call_ollama(system_part, user_part, [], mode=st.session_state.get("inference_mode", "cloud"))
    session.close_turn(answer)

    st.session_state.sorted_df      = sorted_df
    st.session_state.sort_key       = sort_key
    st.session_state.sort_label     = sort_label
    st.session_state.sort_reasoning  = sort_reasoning
    return answer

def trigger_select_mode() -> None:
    session = st.session_state.session
    lookup  = get_lookup()
    profile = session.state["profile"]
    context = session.state["context"]

    filtered_df, decision = lookup.get_plans_filtered(profile["zip"], profile)
    sort_key, reasoning   = derive_sort_key(profile, context)
    sorted_df, sort_label = sort_plans(filtered_df, sort_key, profile)

    lang = st.session_state.get("language", "English")
    prompt = build_prompt_select_mode(
        user_question  = "Please introduce and analyze my top plan options.",
        plans          = sorted_df.head(5),
        state          = session.state,
        sort_label     = sort_label,
        sort_reasoning = reasoning,
        language       = lang,
    )

    _imode = st.session_state.get("inference_mode", "cloud")
    system_part, user_part = split_system_and_user(prompt)
    st.info(_t("gemma_analyzing"))
    with st.spinner(_t("gemma_spinner")):
        analysis = call_ollama(system_part, user_part, [], mode=_imode)
    session.close_turn(analysis)

    filter_summary_raw = decision.user_summary()
    with st.spinner(_t("preparing_results")):
        filter_expl_prompt = build_prompt_filter_explanation(filter_summary_raw, language=lang)
        filter_explanation = call_ollama("", filter_expl_prompt, [], mode=_imode)
        if lang != "English" and reasoning:
            sort_reasoning_prompt = build_prompt_sort_reasoning(reasoning, language=lang)
            reasoning = call_ollama("", sort_reasoning_prompt, [], mode=_imode)

    st.session_state.filtered_df      = filtered_df
    st.session_state.sorted_df        = sorted_df
    st.session_state.sort_key         = sort_key
    st.session_state.sort_label       = sort_label
    st.session_state.sort_reasoning   = reasoning
    st.session_state.filter_summary   = filter_summary_raw
    st.session_state.filter_explanation = filter_explanation
    clean_analysis = _strip_why_lines(analysis)
    st.session_state.select_analysis = analysis
    st.session_state.chat_history.append({"role": "assistant", "content": clean_analysis})
    st.session_state.screen          = "SELECT"
    _save_session_to_disk()
    st.rerun()

def render_sort_controls() -> None:
    st.markdown(_t("resort_label"))
    cols = st.columns(len(SORT_BUTTON_LABELS))
    for col, (key, _) in zip(cols, SORT_BUTTON_LABELS.items()):
        with col:
            is_active = (key == st.session_state.sort_key)
            if st.button(_t(f"sort_{key}"), key=f"sort_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
                if not is_active:
                    new_df, new_label = sort_plans(st.session_state.filtered_df, key, st.session_state.session.state["profile"])
                    st.session_state.sorted_df      = new_df
                    st.session_state.sort_key       = key
                    st.session_state.sort_label     = new_label
                    st.session_state.sort_reasoning = f"You selected: {new_label}"
                    st.rerun()

def render_top5_cards(df: pd.DataFrame, sort_key: str | None = None) -> None:
    gemma_whys = _parse_plan_whys(st.session_state.get("select_analysis", ""))
    top5 = df.head(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        plan_name = row.get("Plan Name", f"Plan {i + 1}")
        premium   = _plan_premium(row)
        pdp       = _is_pdp(row)
        with st.expander(f"#{i + 1}  {plan_name}  —  {premium}/month", expanded=(i == 0)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(_t("card_monthly_premium"), premium)
                moop_val = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
                st.metric(_t("card_moop"), moop_val)
                if pdp and str(moop_val).strip().lower() in ("not applicable", "n/a", ""): st.caption(f"ℹ️ {_PDP_MOOP_NOTE}")
            with col_b:
                st.metric(_t("card_plan_type"), row.get("Plan Type", "—"))
                stars_val = row.get("Overall Star Rating", "—")
                st.metric(_t("card_star_rating"), stars_val)
                if str(stars_val).strip().lower() in ("not applicable", "n/a", ""): st.caption(f"ℹ️ {_PDP_STARS_NOTE}")
            extras = []
            org = row.get("Organization Marketing Name", "")
            if org: extras.append(f"**{_t('card_offered_by')}** {org}")
            drug_ded = row.get("Annual Part D Deductible Amount", "")
            if drug_ded and drug_ded not in ("", "nan", "N/A"): extras.append(f"**{_t('card_drug_deductible')}** {drug_ded}")
            snp = row.get("SNP Type", "")
            if snp and snp.upper() not in ("", "NAN", "N/A", "NOT APPLICABLE"): extras.append(f"**{_t('card_snp')}** {snp}")
            sanctioned = row.get("Sanctioned Plan", "")
            if sanctioned and "YES" in str(sanctioned).upper(): extras.append("⚠️ **CMS-sanctioned plan**")
            if extras: st.markdown("  ·  ".join(extras))
            why = gemma_whys[i] if i < len(gemma_whys) and gemma_whys[i] else _why_recommended(row, i + 1, sort_key)
            if why: st.markdown(f"<div style='margin-top:10px; padding:10px 14px; background:#f0f4fa; border-left:4px solid #003366; border-radius:6px; font-size:0.92rem; color:#1a1a1a;'><strong>{_t('why_near_top')}</strong> {why}</div>", unsafe_allow_html=True)

def render_plan_table(df: pd.DataFrame, sort_key: str | None) -> None:
    rows = []
    has_pdp = False
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        pdp  = _is_pdp(row)
        if pdp: has_pdp = True
        moop  = row.get("In-Network Maximum Out-of-Pocket (MOOP) Amount", "—")
        stars = row.get("Overall Star Rating", "—")
        rows.append({
            "#": rank, "Plan Name": row.get("Plan Name", "—"), "Type": row.get("Plan Type", "—"), "Premium": _plan_premium(row),
            "MOOP ¹" if pdp else "MOOP": "N/A ¹" if (pdp and str(moop).lower() in ("not applicable","n/a","")) else moop,
            "Stars ²" if str(stars).lower() in ("not applicable","n/a","nan","") else "Stars": "N/A ²" if str(stars).lower() in ("not applicable","n/a","nan","") else stars,
            "Drug Deductible": row.get("Annual Part D Deductible Amount", "—"), "Insurer": row.get("Organization Marketing Name", "—"), "Why Recommended": _why_recommended(row, rank, sort_key),
        })
    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=420)
    if has_pdp: st.caption("¹ **MOOP (Maximum Out-of-Pocket)** shows N/A for Part D drug plans. PDP plans cover prescriptions only — they have no MOOP because they don't cover medical or hospital costs. MOOP limits apply only to Medicare Advantage (Part C) plans.  \n² **Star Rating** is not published by CMS for standalone Part D drug plans in this dataset.")

def render_select() -> None:
    session = st.session_state.session
    profile = session.state["profile"]
    st.title(_t("select_title"))
    loc = profile.get("zip", "")
    if profile.get("county") and profile.get("state"): loc += f" · {profile['county']}, {profile['state']}"
    track_map   = {"MA_D": "Medicare Advantage", "PDP": "Part D Drug Plan"}
    track_label = track_map.get(profile.get("track", ""), profile.get("track", ""))
    st.markdown(f"<p style='font-size:1.05rem; color:#555; margin-top:-0.5rem;'>Coverage type: <strong>{track_label}</strong> &nbsp;·&nbsp; Location: <strong>{loc}</strong></p>", unsafe_allow_html=True)
    st.divider()
    sdf = st.session_state.sorted_df
    no_plans = sdf is None or sdf.empty
    tab1, tab2 = st.tabs([_t("tab_recommendations"), _t("tab_explore")])
    with tab1:
        if st.session_state.filter_summary:
            with st.expander(_t("filter_expander"), expanded=False):
                if st.session_state.filter_explanation:
                    st.markdown(st.session_state.filter_explanation)
                    with st.expander(_t("filter_details_expander"), expanded=False): st.text(st.session_state.filter_summary)
                else: safe_md(st.session_state.filter_summary)
        if st.session_state.sort_reasoning: st.info(f"{_t('why_first')} {st.session_state.sort_reasoning}")
        if not no_plans:
            html_bytes = _generate_summary_html().encode("utf-8")
            st.download_button(label=_t("download_button"), data=html_bytes, file_name="medicareguide_plan_summary.html", mime="text/html", use_container_width=True, help=_t("download_help"))
        if no_plans: st.warning(_t("no_plans_warning"))
        else:
            total = len(sdf)
            st.subheader(_t("top5_subheader", n=min(5, total), total=total))
            render_top5_cards(sdf, st.session_state.sort_key)
        st.divider()
        follow_ups = [m for m in st.session_state.chat_history if m != st.session_state.chat_history[0]]
        if len(follow_ups) > 1:
            for msg in follow_ups[1:]:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant": safe_md(msg["content"])
                    else: st.markdown(msg["content"])
        voiced = render_voice_input("select_mic")
        if voiced: st.session_state._select_voice_pending = voiced
        user_input = st.chat_input(_t("select_chat_placeholder"))
        if not user_input and st.session_state.get("_select_voice_pending"): user_input = st.session_state.pop("_select_voice_pending")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)
            answer = handle_select_turn(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"): safe_md(answer)
            maybe_play_audio(answer)
    with tab2:
        if no_plans: st.warning("No plans to display.")
        else:
            total = len(sdf)
            st.subheader(_t("all_plans_subheader", total=total))
            st.caption(_t("explore_caption"))
            render_sort_controls()
            st.write("")
            render_plan_table(sdf, st.session_state.sort_key)
    st.divider()
    col_back, _, col_reset = st.columns([1, 4, 1])
    with col_back:
        if st.button("←", key="select_back", help="Back to setup"):
            st.session_state.screen = "INTAKE"
            st.session_state.intake_step = 4
            session.state["intake_step"] = 4
            session.state["mode"] = "WELCOME"
            st.rerun()
    with col_reset:
        if st.button(_t("start_over"), key="select_reset"):
            _clear_saved_session()
            st.session_state.clear()
            st.rerun()
