"""
MediCareGuide Internationalization (i18n)
======================================
Full UI translation for three languages: English, 中文 (Simplified Chinese),
and Español (Spanish).

Usage:
    from medicareguide_international import t

    t("cta_button", "中文")          # returns "开始 — 查找我的医保计划 →"
    t("budget_display", "Español", val=50)  # "Verá planes de $50/mes o menos."
"""

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Translation table
#  Keys are stable identifiers; values are per-language strings.
#  Use {placeholder} syntax for interpolated values.
# ─────────────────────────────────────────────────────────────────────────────

TRANSLATIONS: dict[str, dict[str, str]] = {

    # ── Welcome / Hero ───────────────────────────────────────────────────────
    "hero_badge": {
        "English": "AI-Powered · 100% Private · Free · CMS 2026 Plan Database",
        "中文":     "AI 驱动 · 100% 私密 · 免费 · CMS 2026 计划数据库",
        "Español":  "Con IA · 100% Privado · Gratis · Base de Datos CMS 2026",
    },
    "hero_tagline": {
        "English": "Find the right Medicare plan — explained in plain English",
        "中文":     "找到适合您的医保计划 — 用简单易懂的语言为您解释",
        "Español":  "Encuentre el plan de Medicare adecuado — explicado en términos sencillos",
    },
    "hero_sub": {
        "English": "Gemma 4 AI runs entirely on your computer &nbsp;·&nbsp; Your data never leaves your device &nbsp;·&nbsp; Official CMS 2026 plan database",
        "中文":     "Gemma 4 AI 完全在您的计算机上运行 &nbsp;·&nbsp; 您的数据永远不会离开您的设备 &nbsp;·&nbsp; 官方 CMS 2026 计划数据库",
        "Español":  "Gemma 4 IA se ejecuta completamente en su computadora &nbsp;·&nbsp; Sus datos nunca salen de su dispositivo &nbsp;·&nbsp; Base de datos oficial CMS 2026",
    },
    "cta_button": {
        "English": "Get started — Find my Medicare plan  →",
        "中文":     "开始 — 查找我的医保计划  →",
        "Español":  "Comenzar — Buscar mi plan de Medicare  →",
    },
    "cta_sub": {
        "English": "Free · No account required · Works offline after setup",
        "中文":     "免费 · 无需注册 · 设置后可离线使用",
        "Español":  "Gratis · Sin cuenta requerida · Funciona sin conexión tras la configuración",
    },
    "how_it_works": {
        "English": "How it works",
        "中文":     "如何使用",
        "Español":  "Cómo funciona",
    },
    "step1_title": {
        "English": "Tell us about you",
        "中文":     "告诉我们您的情况",
        "Español":  "Cuéntenos sobre usted",
    },
    "step1_desc": {
        "English": "Enter your ZIP code, coverage type, budget, and any health situation — takes 2 minutes.",
        "中文":     "输入您的邮政编码、保障类型、预算和健康状况 — 只需 2 分钟。",
        "Español":  "Ingrese su código postal, tipo de cobertura, presupuesto y situación de salud — toma 2 minutos.",
    },
    "step2_title": {
        "English": "We filter &amp; sort",
        "中文":     "我们为您筛选和排序",
        "Español":  "Filtramos y ordenamos",
    },
    "step2_desc": {
        "English": "MediCareGuide filters the CMS database to plans that match your profile and sorts them to your needs.",
        "中文":     "MediCareGuide 从 CMS 数据库中筛选符合您个人情况的计划，并按您的需求排序。",
        "Español":  "MediCareGuide filtra la base de datos de CMS para encontrar los planes que coinciden con su perfil y los ordena según sus necesidades.",
    },
    "step3_title": {
        "English": "Gemma explains",
        "中文":     "Gemma 为您解释",
        "Español":  "Gemma explica",
    },
    "step3_desc": {
        "English": "Gemma 4 walks you through your top 5 options — costs, trade-offs, and what to verify before enrolling.",
        "中文":     "Gemma 4 为您详细介绍前 5 个方案 — 费用、权衡利弊，以及在加入前需要核实的事项。",
        "Español":  "Gemma 4 le guía por sus 5 mejores opciones — costos, ventajas y desventajas, y qué verificar antes de inscribirse.",
    },
    "welcome_chat_heading": {
        "English": "💬 Have a question before getting started?",
        "中文":     "💬 开始前有问题吗？",
        "Español":  "💬 ¿Tiene alguna pregunta antes de comenzar?",
    },
    "welcome_chat_placeholder": {
        "English": "e.g. What is the difference between HMO and PPO?",
        "中文":     "例如：HMO 和 PPO 有什么区别？",
        "Español":  "p. ej. ¿Cuál es la diferencia entre HMO y PPO?",
    },
    "ask_button": {
        "English": "Ask →",
        "中文":     "提问 →",
        "Español":  "Preguntar →",
    },
    "audio_accessibility": {
        "English": "♿ <strong>Accessibility:</strong> MediCareGuide can read every answer aloud.",
        "中文":     "♿ <strong>无障碍功能：</strong>MediCareGuide 可以朗读每个答案。",
        "Español":  "♿ <strong>Accesibilidad:</strong> MediCareGuide puede leer en voz alta cada respuesta.",
    },
    "audio_on": {
        "English": "🔊 Read aloud: ON",
        "中文":     "🔊 朗读：开启",
        "Español":  "🔊 Leer en voz alta: ACTIVADO",
    },
    "audio_off": {
        "English": "🔇 Read aloud: OFF",
        "中文":     "🔇 朗读：关闭",
        "Español":  "🔇 Leer en voz alta: DESACTIVADO",
    },
    "welcome_back_banner": {
        "English": "Welcome back! Your previous setup has been restored. Click **Get started** to continue where you left off.",
        "中文":     "欢迎回来！您之前的设置已恢复。点击**开始**继续上次的进度。",
        "Español":  "¡Bienvenido de nuevo! Su configuración anterior ha sido restaurada. Haga clic en **Comenzar** para continuar donde lo dejó.",
    },
    "lang_selector_help": {
        "English": "Choose your preferred language for AI responses",
        "中文":     "选择 AI 回复的语言",
        "Español":  "Elija su idioma preferido para las respuestas de IA",
    },

    # ── Intake screen ────────────────────────────────────────────────────────
    "intake_title": {
        "English": "🧓 MediCareGuide — Setup",
        "中文":     "🧓 MediCareGuide — 设置",
        "Español":  "🧓 MediCareGuide — Configuración",
    },
    "intake_progress": {
        "English": "Step {step} of 5",
        "中文":     "第 {step} 步，共 5 步",
        "Español":  "Paso {step} de 5",
    },
    "intake_chat_heading": {
        "English": "💬 Have a question? Ask me anything about Medicare.",
        "中文":     "💬 有问题吗？问我任何关于医保的问题。",
        "Español":  "💬 ¿Tiene alguna pregunta? Pregúnteme sobre Medicare.",
    },
    "intake_chat_sub": {
        "English": "Type below — I'll answer while you complete the setup.",
        "中文":     "请在下方输入 — 我会在您完成设置的同时回答您的问题。",
        "Español":  "Escriba abajo — responderé mientras completa la configuración.",
    },
    "intake_chat_placeholder": {
        "English": "Ask a Medicare question while setting up…",
        "中文":     "在设置期间询问医保问题……",
        "Español":  "Haga una pregunta sobre Medicare mientras configura…",
    },
    "view_history": {
        "English": "View conversation history",
        "中文":     "查看对话记录",
        "Español":  "Ver historial de conversación",
    },
    "back_button": {
        "English": "←",
        "中文":     "←",
        "Español":  "←",
    },

    # ── Step 0 — ZIP ─────────────────────────────────────────────────────────
    "zip_prompt": {
        "English": "What's your ZIP code?",
        "中文":     "您的邮政编码是什么？",
        "Español":  "¿Cuál es su código postal?",
    },
    "zip_hint": {
        "English": "Enter your 5-digit ZIP code so I can find plans in your area.",
        "中文":     "请输入您的 5 位邮政编码，以便我查找您所在地区的计划。",
        "Español":  "Ingrese su código postal de 5 dígitos para encontrar planes en su área.",
    },
    "zip_label": {
        "English": "5-digit ZIP code",
        "中文":     "5 位邮政编码",
        "Español":  "Código postal de 5 dígitos",
    },
    "zip_placeholder": {
        "English": "e.g. 24073",
        "中文":     "例如：24073",
        "Español":  "p. ej. 24073",
    },
    "continue_button": {
        "English": "Continue →",
        "中文":     "继续 →",
        "Español":  "Continuar →",
    },
    "zip_error_invalid": {
        "English": "Please enter a valid 5-digit ZIP code.",
        "中文":     "请输入有效的 5 位邮政编码。",
        "Español":  "Por favor ingrese un código postal válido de 5 dígitos.",
    },
    "zip_spinner": {
        "English": "Looking up your area…",
        "中文":     "正在查找您所在地区……",
        "Español":  "Buscando su área…",
    },
    "zip_found": {
        "English": "✓ Found: {location}",
        "中文":     "✓ 已找到：{location}",
        "Español":  "✓ Encontrado: {location}",
    },
    "zip_error_not_found": {
        "English": "ZIP code {zip} was not recognised. Please check and try again.",
        "中文":     "未识别邮政编码 {zip}，请检查后重试。",
        "Español":  "El código postal {zip} no fue reconocido. Por favor verifique e intente de nuevo.",
    },

    # ── Step 1 — Track ───────────────────────────────────────────────────────
    "track_heading": {
        "English": "What kind of Medicare coverage are you looking for?",
        "中文":     "您在寻找哪种类型的医保？",
        "Español":  "¿Qué tipo de cobertura de Medicare está buscando?",
    },
    "track_ma_heading": {
        "English": "**Medicare Advantage (Part C)**",
        "中文":     "**医疗优势计划（C 部分）**",
        "Español":  "**Medicare Advantage (Parte C)**",
    },
    "track_ma_desc": {
        "English": "One plan that bundles hospital, medical, and usually prescription drug coverage. Often includes dental, vision, and hearing benefits.",
        "中文":     "一个捆绑住院、医疗以及通常包括处方药保障的综合计划，通常还包括牙科、视力和听力保障。",
        "Español":  "Un plan que incluye cobertura hospitalaria, médica y generalmente de medicamentos. A menudo incluye beneficios dentales, de visión y audición.",
    },
    "track_ma_button": {
        "English": "Choose Medicare Advantage",
        "中文":     "选择医疗优势计划",
        "Español":  "Elegir Medicare Advantage",
    },
    "track_pdp_heading": {
        "English": "**Part D Drug Plan Only**",
        "中文":     "**仅 D 部分药物计划**",
        "Español":  "**Solo Plan de Medicamentos Parte D**",
    },
    "track_pdp_desc": {
        "English": "A standalone drug plan that adds prescription coverage on top of Original Medicare (Parts A and B). You keep Original Medicare for hospital and medical.",
        "中文":     "一种独立的药物计划，在原始医保（A 部分和 B 部分）基础上增加处方药保障。您保留原始医保的住院和医疗保障。",
        "Español":  "Un plan de medicamentos independiente que agrega cobertura de recetas sobre el Medicare Original (Partes A y B). Usted mantiene el Medicare Original para hospital y médico.",
    },
    "track_pdp_button": {
        "English": "Choose Part D Only",
        "中文":     "选择仅 D 部分计划",
        "Español":  "Elegir solo Parte D",
    },
    "track_medigap_heading": {
        "English": "**Original Medicare + Supplement**",
        "中文":     "**原始医保 + 补充保险**",
        "Español":  "**Medicare Original + Suplemento**",
    },
    "track_medigap_desc": {
        "English": "Keep Original Medicare and add a Medigap (supplement) policy to help cover out-of-pocket costs like copays and deductibles.",
        "中文":     "保留原始医保，并添加 Medigap（补充）保险，以帮助支付自付费用，如共付额和自付额。",
        "Español":  "Conserve el Medicare Original y agregue una póliza Medigap (suplemento) para ayudar a cubrir los costos de bolsillo como copagos y deducibles.",
    },
    "track_medigap_button": {
        "English": "Choose Medigap",
        "中文":     "选择 Medigap 补充保险",
        "Español":  "Elegir Medigap",
    },
    "track_skip": {
        "English": "I'm not sure yet — skip this step",
        "中文":     "我还不确定 — 跳过此步骤",
        "Español":  "No estoy seguro todavía — omitir este paso",
    },
    "explain_difference": {
        "English": "📖 Explain the difference",
        "中文":     "📖 解释区别",
        "Español":  "📖 Explicar la diferencia",
    },

    # ── Step 2 — SNP ─────────────────────────────────────────────────────────
    "snp_heading": {
        "English": "Do any of these apply to you?",
        "中文":     "以下哪项适用于您？",
        "Español":  "¿Alguna de estas opciones se aplica a usted?",
    },
    "snp_caption": {
        "English": "These help us find plans designed specifically for your situation.",
        "中文":     "这些信息帮助我们找到专为您的情况设计的计划。",
        "Español":  "Esto nos ayuda a encontrar planes diseñados específicamente para su situación.",
    },
    "snp_d_snp": {
        "English": "I have both Medicare and Medicaid (dual eligible)",
        "中文":     "我同时拥有 Medicare 和 Medicaid（双重资格）",
        "Español":  "Tengo tanto Medicare como Medicaid (doble elegibilidad)",
    },
    "snp_c_snp": {
        "English": "I have a chronic condition (such as diabetes, heart failure, or COPD)",
        "中文":     "我患有慢性病（如糖尿病、心力衰竭或慢阻肺）",
        "Español":  "Tengo una condición crónica (como diabetes, insuficiencia cardíaca o EPOC)",
    },
    "snp_lis": {
        "English": "I may qualify for Extra Help (Low Income Subsidy) on prescription costs",
        "中文":     "我可能有资格获得处方药费用的额外帮助（低收入补贴）",
        "Español":  "Puedo calificar para Ayuda Adicional (Subsidio de Bajos Ingresos) en costos de medicamentos",
    },
    "snp_none": {
        "English": "None of these apply",
        "中文":     "以上均不适用",
        "Español":  "Ninguna de estas aplica",
    },
    "explain_items": {
        "English": "📖 Explain these items",
        "中文":     "📖 解释这些选项",
        "Español":  "📖 Explicar estos elementos",
    },

    # ── Step 3 — Budget ──────────────────────────────────────────────────────
    "budget_heading": {
        "English": "What's the most you'd want to pay per month in plan premiums?",
        "中文":     "您每月最多愿意支付多少保费？",
        "Español":  "¿Cuánto es lo máximo que desea pagar por mes en primas del plan?",
    },
    "budget_desc": {
        "English": "This is on top of your Part B premium (<strong>$185/month in 2026</strong>). Many Medicare Advantage plans have a <strong>$0</strong> premium.",
        "中文":     "这是在您的 B 部分保费（<strong>2026 年为每月 185 美元</strong>）之外。许多医疗优势计划的保费为 <strong>$0</strong>。",
        "Español":  "Esto es adicional a su prima de la Parte B (<strong>$185/mes en 2026</strong>). Muchos planes de Medicare Advantage tienen una prima de <strong>$0</strong>.",
    },
    "budget_no_limit": {
        "English": "No limit — show me all plans regardless of premium cost",
        "中文":     "无限制 — 显示所有计划，不考虑保费金额",
        "Español":  "Sin límite — mostrarme todos los planes independientemente del costo de la prima",
    },
    "budget_slider_label": {
        "English": "Maximum monthly plan premium",
        "中文":     "每月最高计划保费",
        "Español":  "Prima máxima mensual del plan",
    },
    "budget_display": {
        "English": "You'll see plans at <strong>${val}/month</strong> or less.",
        "中文":     "您将看到每月保费 <strong>${val}</strong> 或以下的计划。",
        "Español":  "Verá planes de <strong>${val}/mes</strong> o menos.",
    },
    "explain_premiums": {
        "English": "📖 Explain premiums",
        "中文":     "📖 解释保费",
        "Español":  "📖 Explicar las primas",
    },

    # ── Step 4 — Preferences ─────────────────────────────────────────────────
    "prefs_heading": {
        "English": "Any preferences I should keep in mind?",
        "中文":     "有什么偏好需要我注意吗？",
        "Español":  "¿Alguna preferencia que deba tener en cuenta?",
    },
    "prefs_caption": {
        "English": "Check all that apply. These help sort and tailor plan recommendations.",
        "中文":     "请勾选所有适用项。这些有助于排序和定制计划推荐。",
        "Español":  "Marque todo lo que aplique. Esto ayuda a ordenar y personalizar las recomendaciones.",
    },
    "pref_has_rx": {
        "English": "I take regular prescription medications",
        "中文":     "我定期服用处方药",
        "Español":  "Tomo medicamentos recetados con regularidad",
    },
    "pref_keep_docs": {
        "English": "I want to keep my current doctors and specialists",
        "中文":     "我想保留我现有的医生和专科医生",
        "Español":  "Quiero conservar mis médicos y especialistas actuales",
    },
    "pref_dental": {
        "English": "Dental, vision, or hearing benefits are important to me",
        "中文":     "牙科、视力或听力保障对我很重要",
        "Español":  "Los beneficios dentales, de visión o audición son importantes para mí",
    },
    "pref_ppo": {
        "English": "I prefer a PPO plan — flexibility to see any doctor without a referral",
        "中文":     "我更喜欢 PPO 计划 — 可以灵活就诊任何医生，无需转诊",
        "Español":  "Prefiero un plan PPO — flexibilidad para ver a cualquier médico sin referido",
    },
    "find_plans_button": {
        "English": "Find My Plans  →",
        "中文":     "查找我的计划  →",
        "Español":  "Buscar Mis Planes  →",
    },
    "prefs_none": {
        "English": "None of these",
        "中文":     "以上均无",
        "Español":  "Ninguna de estas",
    },
    "explain_prefs": {
        "English": "📖 How do these affect my results?",
        "中文":     "📖 这些如何影响我的结果？",
        "Español":  "📖 ¿Cómo afectan mis resultados?",
    },

    # ── SELECT screen ────────────────────────────────────────────────────────
    "select_title": {
        "English": "🧓 MediCareGuide — Your Plans",
        "中文":     "🧓 MediCareGuide — 您的计划",
        "Español":  "🧓 MediCareGuide — Sus Planes",
    },
    "tab_recommendations": {
        "English": "📋 Recommendations",
        "中文":     "📋 推荐计划",
        "Español":  "📋 Recomendaciones",
    },
    "tab_explore": {
        "English": "🔀 Explore All Plans",
        "中文":     "🔀 浏览所有计划",
        "Español":  "🔀 Explorar Todos los Planes",
    },
    "filter_expander": {
        "English": "🔍 How your plans were filtered",
        "中文":     "🔍 您的计划是如何筛选的",
        "Español":  "🔍 Cómo se filtraron sus planes",
    },
    "filter_details_expander": {
        "English": "See full filter details",
        "中文":     "查看完整筛选详情",
        "Español":  "Ver detalles completos del filtro",
    },
    "why_first": {
        "English": "**Why these plans appear first:**",
        "中文":     "**为什么这些计划排在最前面：**",
        "Español":  "**Por qué estos planes aparecen primero:**",
    },
    "why_near_top": {
        "English": "Why it's near the top of your list:",
        "中文":     "为什么它排在您的列表前列：",
        "Español":  "Por qué está cerca del tope de su lista:",
    },
    "card_monthly_premium": {
        "English": "Monthly Premium",
        "中文":     "月保费",
        "Español":  "Prima mensual",
    },
    "card_moop": {
        "English": "MOOP",
        "中文":     "年度自付上限",
        "Español":  "MOOP",
    },
    "card_plan_type": {
        "English": "Plan Type",
        "中文":     "计划类型",
        "Español":  "Tipo de plan",
    },
    "card_star_rating": {
        "English": "Star Rating",
        "中文":     "星级评分",
        "Español":  "Calificación de estrellas",
    },
    "card_offered_by": {
        "English": "Offered by:",
        "中文":     "提供方：",
        "Español":  "Ofrecido por:",
    },
    "card_drug_deductible": {
        "English": "Drug deductible:",
        "中文":     "药品免赔额：",
        "Español":  "Deducible de medicamentos:",
    },
    "card_snp": {
        "English": "Special needs plan:",
        "中文":     "特殊需求计划：",
        "Español":  "Plan de necesidades especiales:",
    },
    "download_button": {
        "English": "⬇ Save my plan summary (printable HTML)",
        "中文":     "⬇ 保存我的计划摘要（可打印 HTML）",
        "Español":  "⬇ Guardar mi resumen de planes (HTML imprimible)",
    },
    "download_help": {
        "English": "Opens in any browser — print or share with family or your doctor.",
        "中文":     "在任何浏览器中打开 — 可以打印或与家人或医生分享。",
        "Español":  "Se abre en cualquier navegador — imprima o comparta con su familia o médico.",
    },
    "no_plans_warning": {
        "English": "No plans matched your criteria. Try removing the budget ceiling or choosing a different coverage type.",
        "中文":     "没有计划符合您的条件。尝试删除预算上限或选择不同的保障类型。",
        "Español":  "Ningún plan coincidió con sus criterios. Intente eliminar el límite de presupuesto o elegir un tipo de cobertura diferente.",
    },
    "top5_subheader": {
        "English": "Top {n} Plans  (out of {total} matching your filters)",
        "中文":     "前 {n} 个计划（共 {total} 个符合您筛选条件的计划）",
        "Español":  "Los {n} mejores planes (de {total} que coinciden con sus filtros)",
    },
    "select_chat_placeholder": {
        "English": "Ask a follow-up question about these plans…",
        "中文":     "询问关于这些计划的后续问题……",
        "Español":  "Haga una pregunta de seguimiento sobre estos planes…",
    },
    "resort_label": {
        "English": "**Re-sort plans by:**",
        "中文":     "**按以下方式重新排序计划：**",
        "Español":  "**Reordenar planes por:**",
    },
    "sort_lowest_premium": {
        "English": "Lowest Premium",
        "中文":     "最低保费",
        "Español":  "Prima más baja",
    },
    "sort_total_cost": {
        "English": "Total Annual Cost",
        "中文":     "年度总费用",
        "Español":  "Costo anual total",
    },
    "sort_star_rating": {
        "English": "Star Rating",
        "中文":     "星级评分",
        "Español":  "Calificación de estrellas",
    },
    "sort_lowest_moop": {
        "English": "Lowest MOOP",
        "中文":     "最低自付上限",
        "Español":  "MOOP más bajo",
    },
    "sort_lowest_deductible": {
        "English": "Lowest Deductible",
        "中文":     "最低免赔额",
        "Español":  "Deducible más bajo",
    },
    "sort_ppo_first": {
        "English": "PPO First",
        "中文":     "PPO 优先",
        "Español":  "PPO primero",
    },
    "all_plans_subheader": {
        "English": "All {total} Matching Plans",
        "中文":     "所有 {total} 个符合条件的计划",
        "Español":  "Todos los {total} planes que coinciden",
    },
    "explore_caption": {
        "English": "Re-sort the table below, then switch back to Recommendations to see an updated AI analysis.",
        "中文":     "重新排序下表，然后切换回推荐计划以查看更新的 AI 分析。",
        "Español":  "Reordene la tabla siguiente, luego vuelva a Recomendaciones para ver un análisis actualizado de IA.",
    },
    "start_over": {
        "English": "↺ Start over",
        "中文":     "↺ 重新开始",
        "Español":  "↺ Volver a empezar",
    },
    "gemma_analyzing": {
        "English": "Finding the best plans for you — Gemma is analyzing your options…",
        "中文":     "正在为您找到最佳计划 — Gemma 正在分析您的选项……",
        "Español":  "Encontrando los mejores planes para usted — Gemma está analizando sus opciones…",
    },
    "gemma_spinner": {
        "English": "This may take 15–30 seconds…",
        "中文":     "这可能需要 15–30 秒……",
        "Español":  "Esto puede tardar 15–30 segundos…",
    },
    "preparing_results": {
        "English": "Preparing your results…",
        "中文":     "正在准备您的结果……",
        "Español":  "Preparando sus resultados…",
    },

    # ── Medigap referral ─────────────────────────────────────────────────────
    "medigap_chat_placeholder": {
        "English": "Ask about Medigap or other Medicare questions…",
        "中文":     "询问关于 Medigap 或其他医保问题……",
        "Español":  "Pregunte sobre Medigap u otras preguntas de Medicare…",
    },
    "medigap_restart": {
        "English": "← Start over with a different coverage type",
        "中文":     "← 重新选择其他保障类型",
        "Español":  "← Volver a empezar con un tipo de cobertura diferente",
    },

    # ── STT / voice ──────────────────────────────────────────────────────────
    "transcribing": {
        "English": "Transcribing…",
        "中文":     "正在转录……",
        "Español":  "Transcribiendo…",
    },
    "stt_failed": {
        "English": "Couldn't make out the audio — please try again or type your question.",
        "中文":     "无法识别音频 — 请重试或直接输入您的问题。",
        "Español":  "No se pudo entender el audio — por favor intente de nuevo o escriba su pregunta.",
    },
    "voice_hint": {
        "English": "🎤 Click the microphone button to record your question, then click it again to stop.",
        "中文":     "🎤 点击话筒按钮开始录音，录完后再次点击停止。",
        "Español":  "🎤 Haga clic en el botón del micrófono para grabar su pregunta y vuelva a hacer clic para detener.",
    },

    # ── Audio toggle (inline) ────────────────────────────────────────────────
    "audio_toggle_on": {
        "English": "🔊 Audio ON",
        "中文":     "🔊 音频：开启",
        "Español":  "🔊 Audio: ACTIVADO",
    },
    "audio_toggle_off": {
        "English": "🔇 Audio OFF",
        "中文":     "🔇 音频：关闭",
        "Español":  "🔇 Audio: DESACTIVADO",
    },
    "audio_toggle_help": {
        "English": "Toggle voice responses",
        "中文":     "切换语音回复",
        "Español":  "Activar/desactivar respuestas de voz",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK = "English"


def t(key: str, language: str = "English", **kwargs: Any) -> str:
    """
    Return the translated string for *key* in *language*.

    Falls back to English if the key or language is not found.
    Interpolates any keyword arguments using str.format().

    Examples:
        t("cta_button", "中文")
        t("intake_progress", "Español", step=3)
        t("top5_subheader", "English", n=5, total=42)
    """
    lang_dict = TRANSLATIONS.get(key)
    if lang_dict is None:
        return key   # unknown key — return as-is so nothing breaks

    text = lang_dict.get(language) or lang_dict.get(_FALLBACK) or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass   # return unformatted if interpolation fails
    return text
