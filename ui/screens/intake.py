import streamlit as st
from ui.components import (
    _t, render_language_selector, render_chat_history,
    render_voice_input, maybe_play_audio, safe_md
)
from ui.utils import _extract_sources, _strip_sources
from ui.state import _save_session_to_disk
from ui.screens.welcome import ask_welcome

# ======================================================================== #
#  Constants                                                                #
# ======================================================================== #

_EXPLAIN_TRACK = """
<strong>How the three coverage types differ</strong>
<table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.93rem;">
  <thead>
    <tr style="background:#003366; color:#fff;">
      <th style="padding:8px 10px; text-align:left;"></th>
      <th style="padding:8px 10px; text-align:left;">Medicare Advantage</th>
      <th style="padding:8px 10px; text-align:left;">Part D Only</th>
      <th style="padding:8px 10px; text-align:left;">Original + Medigap</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Doctor choice</td>
      <td style="padding:7px 10px;">Restricted network (HMO or PPO)</td>
      <td style="padding:7px 10px;">Any doctor that accepts Medicare</td>
      <td style="padding:7px 10px;">Any doctor that accepts Medicare</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Prescriptions included</td>
      <td style="padding:7px 10px;">Usually yes</td>
      <td style="padding:7px 10px;">Yes — that is its only purpose</td>
      <td style="padding:7px 10px;">No — you must add a separate Part D plan</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Dental, vision, hearing</td>
      <td style="padding:7px 10px;">Often included</td>
      <td style="padding:7px 10px;">Not included</td>
      <td style="padding:7px 10px;">Not included</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Out-of-pocket cap (MOOP)</td>
      <td style="padding:7px 10px;">Yes — limits your yearly costs</td>
      <td style="padding:7px 10px;">No cap on medical costs</td>
      <td style="padding:7px 10px;">Medigap covers gaps; no single cap</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Monthly premium</td>
      <td style="padding:7px 10px;">Often $0, but has copays per visit</td>
      <td style="padding:7px 10px;">Low to moderate</td>
      <td style="padding:7px 10px;">Higher, but costs are more predictable</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Referrals needed</td>
      <td style="padding:7px 10px;">Often yes (HMO plans)</td>
      <td style="padding:7px 10px;">No</td>
      <td style="padding:7px 10px;">No</td>
    </tr>
  </tbody>
</table>
"""

_EXPLAIN_SNP = """
<strong>What these three items mean — and why they matter</strong>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Dual Eligible (D-SNP)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>You qualify if you receive both <strong>Medicare</strong> (federal) <strong>and Medicaid</strong> (state) benefits</li>
    <li>Special D-SNP plans coordinate both programs — less paperwork, less confusion</li>
    <li>Typically $0 or very low out-of-pocket costs for most services</li>
    <li>Often adds extra benefits: rides to appointments, meal delivery, over-the-counter allowances</li>
  </ul>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Chronic Condition (C-SNP)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>Plans built around managing one serious ongoing condition (diabetes, heart failure, COPD, etc.)</li>
    <li>Your care team specializes in your condition — more targeted than a general plan</li>
    <li>May include disease management programs, dedicated nurse support lines, and easier specialist access</li>
  </ul>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">Extra Help / Low Income Subsidy (LIS)</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>A federal program that reduces what you pay for prescription drugs</li>
    <li>Lowers Part D costs — premiums, deductibles, and copays can drop significantly or go to $0</li>
    <li>Based on income and assets — many people qualify without realizing it</li>
    <li>If you are not sure, check — it costs nothing to apply</li>
  </ul>
</div>
"""

_EXPLAIN_BUDGET = """
<strong>Understanding your plan costs</strong>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px; margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">What a premium is</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>The fixed monthly amount you pay just to have the plan — whether you use it or not</li>
    <li>This is <strong>separate from and on top of</strong> your Part B premium (<strong>$185/month in 2026</strong>)</li>
  </ul>
</div>

<div style="margin-bottom:10px;">
  <div style="font-weight:700; color:#003366; margin:12px 0 8px;">The four costs to know</div>
  <table style="width:100%; border-collapse:collapse; font-size:0.93rem;">
    <thead>
      <tr style="background:#003366; color:#fff;">
        <th style="padding:8px 10px; text-align:left;">Cost type</th>
        <th style="padding:8px 10px; text-align:left;">What it means</th>
        <th style="padding:8px 10px; text-align:left;">When you pay it</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#fff;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Premium</td>
        <td style="padding:7px 10px;">Monthly fee to hold the plan</td>
        <td style="padding:7px 10px;">Every month, always</td>
      </tr>
      <tr style="background:#f7f9fc;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Deductible</td>
        <td style="padding:7px 10px;">Amount you pay before the plan starts covering costs</td>
        <td style="padding:7px 10px;">First uses each year</td>
      </tr>
      <tr style="background:#fff;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">Copay</td>
        <td style="padding:7px 10px;">Fixed fee per doctor visit or service</td>
        <td style="padding:7px 10px;">Each time you use care</td>
      </tr>
      <tr style="background:#f7f9fc;">
        <td style="padding:7px 10px; font-weight:600; color:#003366;">MOOP</td>
        <td style="padding:7px 10px;">The most you would ever pay in a year — plan covers 100% after this</td>
        <td style="padding:7px 10px;">Stops your costs at a cap</td>
      </tr>
    </tbody>
  </table>
</div>

<div style="padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">The $0 premium trade-off</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li>Many Medicare Advantage plans charge $0/month — but you pay copays each time you use care</li>
    <li>A $0 premium plan can cost more overall if you visit doctors frequently</li>
    <li>A higher premium plan often means lower copays — more predictable total costs</li>
  </ul>
</div>
"""

_EXPLAIN_PREFS = """
<strong>What each preference does for your results</strong>

<table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.93rem;">
  <thead>
    <tr style="background:#003366; color:#fff;">
      <th style="padding:8px 10px; text-align:left;">Preference</th>
      <th style="padding:8px 10px; text-align:left;">How it affects your recommendations</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Regular prescriptions</td>
      <td style="padding:7px 10px;">Plans are sorted by lowest drug deductible or total annual cost — drug costs are prioritised</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Keep my doctors</td>
      <td style="padding:7px 10px;">Gemma flags which plan types restrict your network and reminds you to verify your doctors are in-network before enrolling</td>
    </tr>
    <tr style="background:#fff;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Dental / vision / hearing</td>
      <td style="padding:7px 10px;">Gemma highlights which plans include these benefits and adds a checklist item to confirm what is actually covered</td>
    </tr>
    <tr style="background:#f7f9fc;">
      <td style="padding:7px 10px; font-weight:600; color:#003366;">Prefer PPO</td>
      <td style="padding:7px 10px;">PPO plans are shown before HMO plans in your results</td>
    </tr>
  </tbody>
</table>

<div style="margin-top:14px; padding:12px 14px; background:#fff; border-left:3px solid #003366; border-radius:6px;">
  <div style="font-weight:700; color:#003366; margin-bottom:6px;">HMO vs PPO — what is the difference?</div>
  <ul style="margin:0; padding-left:18px; line-height:1.7;">
    <li><strong>HMO:</strong> Lower premiums, but you must use doctors in the plan's network and usually need a referral to see a specialist</li>
    <li><strong>PPO:</strong> Higher premiums, but you can see any doctor (in or out of network) without a referral — more flexibility</li>
  </ul>
</div>
"""

# ======================================================================== #
#  INTAKE screen                                                            #
# ======================================================================== #

def _render_inline_explain(step: int) -> None:
    """If an explanation for this step is ready, render it inline below the buttons."""
    if (
        st.session_state.get("explain_step") == step
        and st.session_state.get("explain_text")
    ):
        st.write("")
        st.markdown(
            f"<div style='background:#f0f4fa; border-left:4px solid #003366; "
            f"border-radius:8px; padding:16px 20px; font-size:0.95rem;'>"
            f"{st.session_state.explain_text}</div>",
            unsafe_allow_html=True,
        )

def render_step0_zip() -> None:
    st.subheader(_t("zip_prompt"))
    st.caption(_t("zip_hint"))
    st.write("")

    zip_val = st.text_input(
        _t("zip_label"),
        max_chars=5,
        placeholder=_t("zip_placeholder"),
        key="zip_text_input",
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(_t("continue_button"), type="primary", use_container_width=True, key="zip_continue"):
            v = zip_val.strip()
            if not v.isdigit() or len(v) != 5:
                st.error(_t("zip_error_invalid"))
            else:
                state = st.session_state.session.set_intake_field(0, "zip", v)
                if state["profile"]["zip"]:
                    p = state["profile"]
                    st.session_state.intake_step = state["intake_step"]
                    location = v
                    if p.get("county") and p.get("state"):
                        location += f" ({p['county']}, {p['state']})"
                    st.success(_t("zip_found", location=location))
                    st.rerun()
                else:
                    st.error(_t("zip_error_not_found", zip=v))

def render_step1_track() -> None:
    st.subheader(_t("track_heading"))
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(_t("track_ma_heading"))
        st.caption(_t("track_ma_desc"))
        if st.button(_t("track_ma_button"), use_container_width=True, type="primary", key="track_ma"):
            st.session_state.session.set_intake_field(1, "track", "MA_D", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col2:
        st.markdown(_t("track_pdp_heading"))
        st.caption(_t("track_pdp_desc"))
        if st.button(_t("track_pdp_button"), use_container_width=True, key="track_pdp"):
            st.session_state.session.set_intake_field(1, "track", "PDP", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col3:
        st.markdown(_t("track_medigap_heading"))
        st.caption(_t("track_medigap_desc"))
        if st.button(_t("track_medigap_button"), use_container_width=True, key="track_medigap"):
            st.session_state.session.set_intake_field(1, "track", "MEDIGAP", "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    st.write("")
    col_skip, col_explain = st.columns(2)
    with col_skip:
        if st.button(_t("track_skip"), use_container_width=True, key="track_skip"):
            st.session_state.session.set_intake_field(1, "track", None, "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()
    with col_explain:
        if st.button(_t("explain_difference"), use_container_width=True, key="track_explain"):
            st.session_state.explain_text = _EXPLAIN_TRACK
            st.session_state.explain_step = 1

    _render_inline_explain(1)

def render_step2_snp() -> None:
    st.subheader(_t("snp_heading"))
    st.caption(_t("snp_caption"))
    st.write("")

    d_snp = st.checkbox(_t("snp_d_snp"), key="snp_d_snp")
    c_snp = st.checkbox(_t("snp_c_snp"), key="snp_c_snp")
    lis   = st.checkbox(_t("snp_lis"),   key="snp_lis")

    st.write("")
    col_cont, col_none, col_exp = st.columns(3)

    with col_cont:
        if st.button(_t("continue_button"), type="primary", use_container_width=True, key="snp_continue"):
            flags = []
            if d_snp: flags.append("D_SNP")
            if c_snp: flags.append("C_SNP")
            if lis:   flags.append("LIS")
            st.session_state.session.set_intake_field(2, "snp_flags", flags, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_none:
        if st.button(_t("snp_none"), use_container_width=True, key="snp_none"):
            st.session_state.session.set_intake_field(2, "snp_flags", [], "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(_t("explain_items"), use_container_width=True, key="snp_explain"):
            st.session_state.explain_text = _EXPLAIN_SNP
            st.session_state.explain_step = 2

    _render_inline_explain(2)

def render_step3_budget() -> None:
    st.subheader(_t("budget_heading"))
    st.markdown(f"<p style='font-size:1rem; color:#444;'>{_t('budget_desc')}</p>", unsafe_allow_html=True)
    st.write("")

    no_limit = st.checkbox(_t("budget_no_limit"), key="budget_no_limit")

    budget_val = None
    if not no_limit:
        budget_val = st.slider(
            _t("budget_slider_label"),
            min_value=0, max_value=500, value=50, step=10, format="$%d", key="budget_slider",
        )
        st.markdown(f"<p style='font-size:1rem; color:#444;'>{_t('budget_display', val=budget_val)}</p>", unsafe_allow_html=True)

    st.write("")
    col_cont, col_exp = st.columns([2, 1])

    with col_cont:
        if st.button(_t("continue_button"), type="primary", use_container_width=True, key="budget_continue"):
            if no_limit:
                st.session_state.session.set_intake_field(3, "budget_max", None, "skip")
            else:
                st.session_state.session.set_intake_field(3, "budget_max", budget_val, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(_t("explain_premiums"), use_container_width=True, key="budget_explain"):
            st.session_state.explain_text = _EXPLAIN_BUDGET
            st.session_state.explain_step = 3

    _render_inline_explain(3)

def render_step4_prefs() -> None:
    st.subheader(_t("prefs_heading"))
    st.caption(_t("prefs_caption"))
    st.write("")

    has_rx    = st.checkbox(_t("pref_has_rx"),    key="pref_has_rx")
    keep_docs = st.checkbox(_t("pref_keep_docs"), key="pref_keep_docs")
    dental    = st.checkbox(_t("pref_dental"),    key="pref_dental")
    ppo       = st.checkbox(_t("pref_ppo"),       key="pref_ppo")

    st.write("")
    col_find, col_none, col_exp = st.columns(3)

    with col_find:
        if st.button(_t("find_plans_button"), type="primary", use_container_width=True, key="prefs_find"):
            flags = []
            if has_rx:    flags.append("has_rx")
            if keep_docs: flags.append("keep_doctors")
            if dental:    flags.append("wants_dental")
            if ppo:       flags.append("prefers_ppo")
            st.session_state.session.set_intake_field(4, "context", flags, "set")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_none:
        if st.button(_t("prefs_none"), use_container_width=True, key="prefs_none"):
            st.session_state.session.set_intake_field(4, "context", [], "skip")
            st.session_state.intake_step = st.session_state.session.state["intake_step"]
            st.rerun()

    with col_exp:
        if st.button(_t("explain_prefs"), use_container_width=True, key="prefs_explain"):
            st.session_state.explain_text = _EXPLAIN_PREFS
            st.session_state.explain_step = 4

    _render_inline_explain(4)

def render_medigap_referral() -> None:
    st.title("🧓 MediGuide")
    st.warning(
        "**Medigap (Medicare Supplement) plans** are not included in the "
        "CMS Landscape database. Medigap policies are sold directly by "
        "private insurers — they are not part of the plan data I have access to."
    )
    st.markdown(
        "**To compare Medigap plans in your area, visit:**\n\n"
        "🔗 [medicare.gov/find-a-plan](https://www.medicare.gov/find-a-plan/)\n\n"
        "I can still answer general questions about how Medigap works, "
        "what the plan letters mean (G, N, K, L, etc.), or how "
        "Medigap compares to Medicare Advantage — just ask below."
    )
    st.divider()

    render_chat_history()

    user_input = st.chat_input(_t("medigap_chat_placeholder"))
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        answer = ask_welcome(user_input)
        sources = _extract_sources(answer)
        clean_answer = _strip_sources(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": clean_answer, "sources": sources})
        with st.chat_message("assistant"):
            safe_md(clean_answer)
            if sources:
                st.caption(f"📖 *Medicare & You 2026 — {sources}*")
        maybe_play_audio(clean_answer)

    st.divider()
    col_restart, _ = st.columns([2, 1])
    with col_restart:
        if st.button(_t("medigap_restart"), use_container_width=True):
            st.session_state.clear()
            st.rerun()

def render_intake() -> None:
    from ui.screens.select import trigger_select_mode
    session = st.session_state.session
    profile = session.state["profile"]

    _save_session_to_disk()

    if profile.get("track") == "MEDIGAP":
        render_medigap_referral()
        return

    title_col, lang_col = st.columns([3, 2])
    with title_col:
        st.title(_t("intake_title"))
    with lang_col:
        render_language_selector("intake")

    step = st.session_state.intake_step
    st.progress(min(step / 5, 1.0), text=_t("intake_progress", step=step))
    st.divider()

    if   step == 0: render_step0_zip()
    elif step == 1: render_step1_track()
    elif step == 2: render_step2_snp()
    elif step == 3: render_step3_budget()
    elif step == 4: render_step4_prefs()
    elif step >= 5: trigger_select_mode()

    st.divider()
    st.markdown(
        f"<p style='font-size:1.05rem; font-weight:700; color:#003366; margin-bottom:4px;'>"
        f"{_t('intake_chat_heading')}</p>"
        f"<p style='font-size:0.9rem; color:#666; margin-top:0;'>"
        f"{_t('intake_chat_sub')}</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.chat_history:
        with st.expander(_t("view_history"), expanded=False):
            render_chat_history()

    with st.form("intake_chat", clear_on_submit=True):
        col_inp, col_ask = st.columns([6, 1])
        with col_inp:
            user_input = st.text_input(
                "question",
                placeholder=_t("intake_chat_placeholder"),
                label_visibility="collapsed",
                key="intake_chat_input",
            )
        with col_ask:
            asked = st.form_submit_button(_t("ask_button"), use_container_width=True)

    voiced = render_voice_input("intake_mic")
    if voiced:
        asked = True
        user_input = voiced

    if asked and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.chat_message("user"):
            st.markdown(user_input.strip())
        answer = ask_welcome(user_input.strip())
        sources = _extract_sources(answer)
        clean_answer = _strip_sources(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": clean_answer, "sources": sources})
        with st.chat_message("assistant"):
            safe_md(clean_answer)
            if sources:
                st.caption(f"📖 *Medicare & You 2026 — {sources}*")
        maybe_play_audio(clean_answer)

    st.write("")
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("←", key="intake_back", help="Go back", use_container_width=False):
            if step == 0:
                st.session_state.screen = "WELCOME"
            else:
                st.session_state.intake_step = step - 1
                session.state["intake_step"] = step - 1
            st.rerun()
