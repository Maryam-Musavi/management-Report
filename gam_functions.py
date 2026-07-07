"""
**************************************************
Gam Chatbot Functions
**************************************************
"""

from typing import Optional
import pandas as pd
import streamlit as st
import gam_constants as constants

# ── Name-matching helpers ─────────────────────────
_HONORIFICS  = ['آقای', 'خانم', 'جناب', 'سرکار']
_SUFFIX_ONLY = {'راد', 'زاده', 'نیا', 'پور', 'نژاد'}  # ambiguous alone


def _name_sig_parts(full_name: str) -> list[str]:
    """Return significant (non-honorific, non-ambiguous) parts of a name."""
    clean = full_name
    for h in _HONORIFICS:
        clean = clean.replace(h, '')
    parts = [p.strip() for p in clean.split() if len(p.strip()) > 1]
    sig   = [p for p in parts if p not in _SUFFIX_ONLY]
    return sig if sig else parts  # fallback: keep all if filtering leaves nothing


def _find_person(user_prompt: str, df: pd.DataFrame) -> tuple[Optional[pd.Series], str]:
    """
    Robust name matching: score each row by overlap with the user's query.
    Returns (best_row, matched_name) or (None, '').
    """
    name_col = 'نام متقاضی'
    best_score, best_row, best_name = 0, None, ''

    for _, row in df.iterrows():
        raw  = str(row.get(name_col, '')).strip()
        if not raw:
            continue
        sig  = _name_sig_parts(raw)
        hits = sum(1 for p in sig if p in user_prompt)
        # small bonus when the last name part appears verbatim
        if sig and sig[-1] in user_prompt:
            hits += 0.5
        if hits > best_score:
            best_score, best_row, best_name = hits, row, raw

    return (best_row, best_name) if best_score > 0 else (None, '')


# ══════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════

def load_data_from_upload(uploaded_file) -> None:
    """Load DataFrame from UI file upload. Keeps chat history intact."""
    try:
        df = pd.read_excel(uploaded_file)
        df = df.fillna('')
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        st.session_state.dataframe = df
    except Exception as e:
        st.sidebar.error(body=f'⛔ خطا در خواندن فایل: {e}')


def load_data() -> Optional[pd.DataFrame]:
    return st.session_state.get('dataframe', None)


# ══════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════

def _rows_to_context(subset: pd.DataFrame) -> str:
    """Compact text representation of a filtered DataFrame for the LLM."""
    if subset.empty:
        return 'No matching records found.'
    lines = [f'Relevant records ({len(subset)} total):']
    for _, row in subset.iterrows():
        lines.append(
            f"- Row {row.get('ردیف','?')} | "
            f"Province: {row.get('واحد درخواست کننده','')} | "
            f"Name: {row.get('نام متقاضی','')} | "
            f"Type: {row.get('نوع درخواست کننده','')} | "
            f"Goods: {row.get('نوع کالای درخواستی','')} | "
            f"Supplier: {row.get('وضعیت تامین کننده','')} | "
            f"Date: {row.get('تاریخ','')} | "
            f"Intro: {row.get('نحوه معرفی','')} | "
            f"Actions: {row.get('اقدامات انجام شده','(empty)')}"
        )
    return '\n'.join(lines)


# Goods normalisation map  (canonical → aliases found in data)
_GOODS_MAP: dict[str, list[str]] = {
    'لوازم خانگی':       ['لوازم خانگی', 'اوازم خانگی', 'لوازم خانگی - دیپوینت',
                           'لوازم الکترونیکی', 'اسپیلیت', 'الماس بلورین ایرانیان'],
    'کالای دیجیتال':     ['کالای دیجیتال'],
    'مبل و مبلمان':      ['کارخانه تولید مبل'],
    'تجهیزات کشاورزی':  ['تجهیزات کشاورزی'],
    'موتورسیکلت':        ['شرکت موتورسیکلت'],
    'دیپوینت':           ['دیپوینت'],
}


def _compute_goods_stats(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Normalise the goods column and return {category: [names]}.
    Handles combined values like 'کالای دیجیتال، لوازم خانگی'.
    """
    result: dict[str, list[str]] = {k: [] for k in _GOODS_MAP}
    for _, row in df.iterrows():
        raw  = str(row.get('نوع کالای درخواستی', ''))
        name = str(row.get('نام متقاضی', ''))
        for canonical, aliases in _GOODS_MAP.items():
            if any(a.lower() in raw.lower() for a in aliases):
                if name not in result[canonical]:
                    result[canonical].append(name)
    return result


def _compute_actions_summary(df: pd.DataFrame) -> str:
    """
    Produce a structured English summary of the actions column for the LLM.
    Includes: who has actions, who doesn't, and the full action text for each.
    """
    action_col = 'اقدامات انجام شده'
    name_col   = 'نام متقاضی'

    done, empty = [], []
    for _, row in df.iterrows():
        name   = str(row.get(name_col, '')).strip()
        action = str(row.get(action_col, '')).strip()
        (done if action else empty).append({'name': name, 'action': action})

    lines = [
        f'Total records: {len(df)}',
        f'Records WITH action logged: {len(done)}',
        f'Records WITHOUT any action: {len(empty)}',
        f'No-action names: {", ".join(x["name"] for x in empty)}',
        '',
        '=== Full action log ===',
    ]
    for item in done:
        lines.append(f'  [{item["name"]}]: {item["action"]}')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════
# Smart Query Pre-processing
# ══════════════════════════════════════════════════

def preprocess_query(user_prompt: str, df: pd.DataFrame) -> tuple[str, str]:
    """
    Python-first query resolution.

    Detects intent → computes exact facts in Python →
    injects [Python pre-computation] block into user message.
    The LLM then only writes natural Persian prose around those facts.
    """
    province_col = 'واحد درخواست کننده'
    type_col     = 'نوع درخواست کننده'
    name_col     = 'نام متقاضی'
    goods_col    = 'نوع کالای درخواستی'
    action_col   = 'اقدامات انجام شده'
    date_col     = 'تاریخ'
    supplier_col = 'وضعیت تامین کننده'

    # ── 1. Person-specific lookup (highest priority) ──
    # Check this BEFORE province, so 'همدان' in a name doesn't match province
    row, matched_name = _find_person(user_prompt, df)
    if row is not None:
        action = str(row.get(action_col, '')).strip()
        date   = str(row.get(date_col, '')).strip()
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] Record for '{matched_name}':\n"
            f"  Date: {date or '(تاریخ ثبت نشده)'}\n"
            f"  Actions: {action or '(هیچ اقدامی ثبت نشده است)'}\n"
            f"  Province: {row.get(province_col,'')}\n"
            f"  Type: {row.get(type_col,'')}\n"
            f"  Goods: {row.get(goods_col,'')}\n"
            f"  Supplier: {row.get(supplier_col,'')}\n"
            f"  Introduction: {row.get('نحوه معرفی','')}"
        )
        return enriched, _rows_to_context(df[df[name_col] == matched_name])

    # ── 2. Province lookup ────────────────────────────
    for prov in df[province_col].unique():
        if prov and prov in user_prompt:
            subset = df[df[province_col] == prov]
            names  = subset[name_col].tolist()
            enriched = (
                f'{user_prompt}\n\n'
                f"[Python pre-computation] Province '{prov}': "
                f"exactly {len(subset)} records. "
                f"Names: {', '.join(names)}"
            )
            return enriched, _rows_to_context(subset)

    # ── 3. Applicant type ────────────────────────────
    if any(k in user_prompt for k in ['حقیقی', 'حقوقی', 'نوع متقاضی']):
        real_df  = df[df[type_col] == 'متقاضی حقیقی']
        legal_df = df[df[type_col] == 'متقاضی حقوقی']
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] "
            f"'متقاضی حقیقی': exactly {len(real_df)} — {', '.join(real_df[name_col].tolist())}. "
            f"'متقاضی حقوقی': exactly {len(legal_df)} — {', '.join(legal_df[name_col].tolist())}."
        )
        return enriched, _rows_to_context(pd.concat([real_df, legal_df]))

    # ── 4. Province count ────────────────────────────
    if any(k in user_prompt for k in ['چند استان', 'تعداد استان', 'کدام استان', 'همه استان']):
        prov_counts = df.groupby(province_col)[name_col].apply(list).to_dict()
        lines = [f'{p}: {len(ns)} نفر — {", ".join(ns)}'
                 for p, ns in sorted(prov_counts.items(), key=lambda x: -len(x[1]))]
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] Unique provinces: exactly {df[province_col].nunique()}.\n"
            + '\n'.join(lines)
        )
        return enriched, '\n'.join(lines)

    # ── 5. Bank ──────────────────────────────────────
    if any(k in user_prompt for k in ['بانک', 'شعبه']):
        subset = df[df[type_col] == 'شعبه بانک']
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] 'شعبه بانک': exactly {len(subset)}. "
            f"Names: {', '.join(subset[name_col].tolist())}"
        )
        return enriched, _rows_to_context(subset)

    # ── 6. Supplier ──────────────────────────────────
    if any(k in user_prompt for k in ['تامین کننده', 'تامین‌کننده']):
        subset = df[df[supplier_col].str.contains('تامین کننده دارد', na=False)]
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] Records with supplier: exactly {len(subset)}. "
            f"Names: {', '.join(subset[name_col].tolist())}"
        )
        return enriched, _rows_to_context(subset)

    # ── 7. Goods analysis ────────────────────────────
    if any(k in user_prompt for k in ['کالا', 'کالای', 'لوازم', 'دیجیتال',
                                       'چه کالا', 'کدام کالا', 'بیشتر', 'مبل']):
        stats    = _compute_goods_stats(df)
        no_goods = df[df[goods_col] == ''][name_col].tolist()
        lines    = [f'  {cat}: {len(names)} نفر — {", ".join(names)}'
                    for cat, names in stats.items() if names]
        enriched = (
            f'{user_prompt}\n\n'
            f'[Python pre-computation] Goods breakdown:\n'
            + '\n'.join(lines)
            + f'\n  (بدون کالای مشخص: {len(no_goods)} نفر — {", ".join(no_goods)})'
        )
        return enriched, '\n'.join(lines)

    # ── 8. Actions / general analysis ───────────────
    if any(k in user_prompt for k in ['اقدام', 'تحلیل', 'گزارش', 'بررسی', 'وضعیت', 'کلی']):
        actions_txt  = _compute_actions_summary(df)
        type_counts  = df[type_col].value_counts().to_dict()
        prov_counts  = df[province_col].value_counts().to_dict()
        enriched = (
            f'{user_prompt}\n\n'
            f'[Python pre-computation] Full table analysis:\n'
            f'  Total records: {len(df)}\n'
            f'  By type: {type_counts}\n'
            f'  By province: {prov_counts}\n\n'
            f'Actions analysis:\n{actions_txt}'
        )
        return enriched, actions_txt

    # ── 9. Fallback ──────────────────────────────────
    type_counts = df[type_col].value_counts().to_dict()
    prov_counts = df[province_col].value_counts().to_dict()
    enriched = (
        f'{user_prompt}\n\n'
        f'[Python pre-computation] Total: {len(df)}. '
        f'By type: {type_counts}. By province: {prov_counts}.'
    )
    return enriched, _rows_to_context(df)


# ══════════════════════════════════════════════════
# Prompt Building
# ══════════════════════════════════════════════════

def build_messages_with_context(
    user_prompt: str,
    history: list[dict],
) -> list[dict]:
    """Build the full Ollama message list."""
    df: Optional[pd.DataFrame] = load_data()

    if df is not None:
        enriched_prompt, data_context = preprocess_query(user_prompt, df)
    else:
        enriched_prompt = user_prompt
        data_context    = 'No data loaded.'

    return [
        {'role': 'system',    'content': constants.SYSTEM_PROMPT.format(data_context=data_context)},
        *[m for m in history if m['role'] in ('user', 'assistant')],
        {'role': 'user',      'content': enriched_prompt},
    ]


# ══════════════════════════════════════════════════
# Sidebar Stats
# ══════════════════════════════════════════════════

def show_sidebar_stats() -> None:
    df: Optional[pd.DataFrame] = load_data()
    if df is None:
        return
    st.subheader(body=constants.STATS_HEADER)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='📌 کل', value=len(df))
        st.metric(label='👤 حقیقی',
                  value=len(df[df['نوع درخواست کننده'] == 'متقاضی حقیقی']))
    with col2:
        st.metric(label='🗺️ استان', value=df['واحد درخواست کننده'].nunique())
        st.metric(label='🏢 حقوقی',
                  value=len(df[df['نوع درخواست کننده'] == 'متقاضی حقوقی']))


# ══════════════════════════════════════════════════
# Streamlit Setup
# ══════════════════════════════════════════════════

def set_page_config() -> None:
    st.set_page_config(
        page_title='دستیار هوشمند اوراق گام',
        page_icon='📊',
        layout='wide',
    )


def initial_session_state() -> None:
    if 'messages' not in st.session_state:
        st.session_state.messages = []
