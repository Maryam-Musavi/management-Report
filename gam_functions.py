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


import gam_database as db

# ══════════════════════════════════════════════════
# Data Loading  (SQLite-backed — no more Excel)
# ══════════════════════════════════════════════════

def load_data_from_upload(uploaded_file) -> None:
    """
    Import a new .sqlite/.db file uploaded from the UI, replacing the
    current database content. Keeps chat history intact.
    """
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        import sqlite3
        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )['name'].tolist()
        if not tables:
            raise ValueError('هیچ جدولی در فایل یافت نشد.')
        new_df = pd.read_sql_query(f'SELECT * FROM {tables[0]}', conn)
        conn.close()

        if 'شماره تماس' not in new_df.columns:
            new_df['شماره تماس'] = None

        db.import_dataframe(new_df, replace=True)
        st.session_state.pop('dataframe', None)  # force reload from DB on next access
    except Exception as e:
        st.sidebar.error(body=f'⛔ خطا در خواندن فایل: {e}')


def load_data() -> Optional[pd.DataFrame]:
    """
    Return the customers DataFrame, cached in session_state.
    Loaded from the SQLite database on first access (or after an upload).
    """
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = db.load_dataframe()
    return st.session_state.dataframe


def reload_data() -> None:
    """Force a fresh reload from the database (e.g. after external updates)."""
    st.session_state.dataframe = db.load_dataframe()


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
            f"Phone: {row.get('شماره تماس') or '(not recorded)'} | "
            f"Intro: {row.get('نحوه معرفی','')} | "
            f"Actions: {row.get('اقدامات انجام شده','(empty)')}"
        )
    return '\n'.join(lines)


import re as _re


def _compute_goods_stats(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Split the raw goods column into individual categories WITHOUT assuming
    any grouping beyond what the data itself expresses.

    Only two safe transformations are applied:
      1. Fix the obvious typo 'اوازم خانگی' → 'لوازم خانگی'
      2. Split combined values that list multiple goods for one customer,
         e.g. 'کالای دیجیتال، لوازم خانگی' or 'لوازم خانگی - دیپوینت'
         (split on '،' or '-')

    Distinct product/brand names (e.g. 'اسپیلیت', 'الماس بلورین ایرانیان',
    'لوازم الکترونیکی') are kept as their OWN separate category — they are
    NOT folded into 'لوازم خانگی', since that would be an unverified
    assumption about what those specific goods are.

    Returns:
        {category_name: [customer names]}, ordered by frequency (desc)
        when iterated by the caller.
    """
    name_col, goods_col = 'نام متقاضی', 'نوع کالای درخواستی'
    result: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        raw  = str(row.get(goods_col, '')).strip()
        name = str(row.get(name_col, '')).strip()
        if not raw:
            continue
        raw_norm = raw.replace('اوازم', 'لوازم')  # fix typo only
        parts = [p.strip() for p in _re.split(r'[،\-]', raw_norm) if p.strip()]
        for p in parts:
            result.setdefault(p, [])
            if name not in result[p]:
                result[p].append(name)

    # Sort categories by popularity (most requested first)
    return dict(sorted(result.items(), key=lambda kv: -len(kv[1])))



# Pattern groups for deeper action-column analysis.
# Each pattern maps a Persian label -> list of substrings, ALL of which
# (mode='and') or ANY of which (mode='or') must appear in the action text.
_ACTION_PATTERNS: list[tuple[str, list[str], str]] = [
    ('اوراق گام منتشر شده (دارای مبلغ اوراق)',        ['منتشر شده', 'منتشر شد'], 'or'),
    ('ابراز نارضایتی نسبت به نرخ تنزیل',                ['نرخ تنزیل'],             'or'),
    ('ابراز نارضایتی به\u200cطور کلی',                   ['نارضایتی', 'ناراضی'],   'or'),
    ('منصرف شده یا معامله را لغو کرده\u200cاند',         ['منصرف', 'لغو'],          'or'),
    # AND-pattern: both "تبدیل" and "نکرده" must appear (handles wording
    # variations like "تبدیل به اوراق گام نکرده اند")
    ('هنوز مبالغ را به اوراق تبدیل نکرده\u200cاند',      ['تبدیل', 'نکرده'],        'and'),
    ('مشکل دستگاه پوز / کارت رفاهی',                     ['پوز'],                   'or'),
    ('آمادگی برای انجام معامله',                          ['آمادگی دارند', 'آمادگی دارد'], 'or'),
    ('در انتظار تثبیت قیمت / نوسانات ارز',               ['نوسانات نرخ ارز', 'تثبیت قیمت'], 'or'),
]


def _compute_pattern_matches(df: pd.DataFrame) -> str:
    """
    Scan the action column for recurring themes (bond publication,
    discount-rate complaints, general dissatisfaction, etc.) and
    return exact counts + names + full quoted text for each theme.

    Each pattern is Python-verified (exact substring/AND matching),
    so the LLM never has to detect these itself — it only phrases
    the already-computed findings in Persian.
    """
    action_col = 'اقدامات انجام شده'
    name_col   = 'نام متقاضی'

    lines = ['=== Thematic pattern analysis (exact counts + quotes from Python) ===']
    for label, keywords, mode in _ACTION_PATTERNS:
        if mode == 'and':
            mask = df[action_col].apply(lambda x, kws=keywords: all(k in x for k in kws))
        else:
            mask = df[action_col].apply(lambda x, kws=keywords: any(k in x for k in kws))
        matched = df[mask]
        lines.append(f'\n{label}: exactly {len(matched)} نفر')
        for _, row in matched.iterrows():
            lines.append(f'  - {row[name_col]}: {row[action_col]}')
        if matched.empty:
            lines.append('  (هیچکس)')
    return '\n'.join(lines)


def _compute_dominant_conclusions(df: pd.DataFrame) -> str:
    """
    Produce PERCENTAGE-based 'dominant pattern' conclusions — the kind of
    high-level narrative summary a manager wants ("اکثر مشتریان..."),
    as opposed to a case-by-case itemised list.

    Every number here is Python-verified (exact counts / exact percentages),
    so the LLM's only job is to turn them into flowing Persian prose that
    leads with the most prevalent (dominant) patterns first.
    """
    action_col = 'اقدامات انجام شده'
    goods_col  = 'نوع کالای درخواستی'
    type_col   = 'نوع درخواست کننده'
    total      = len(df)

    with_action    = int((df[action_col] != '').sum())
    without_action = total - with_action

    real_cnt  = int((df[type_col] == 'متقاضی حقیقی').sum())
    legal_cnt = int((df[type_col] == 'متقاضی حقوقی').sum())
    bank_cnt  = total - real_cnt - legal_cnt

    # ── Dominant goods category ───────────────────────
    goods_stats      = _compute_goods_stats(df)              # already sorted desc
    non_empty_goods  = int((df[goods_col] != '').sum())
    top_goods        = list(goods_stats.items())[:3]

    # ── Dominant action themes (ranked by frequency) ──
    theme_stats = []
    for label, keywords, mode in _ACTION_PATTERNS:
        if mode == 'and':
            mask = df[action_col].apply(lambda x, kws=keywords: all(k in x for k in kws))
        else:
            mask = df[action_col].apply(lambda x, kws=keywords: any(k in x for k in kws))
        cnt = int(mask.sum())
        pct_of_total  = (cnt / total * 100) if total else 0
        pct_of_active = (cnt / with_action * 100) if with_action else 0
        theme_stats.append((label, cnt, pct_of_total, pct_of_active))
    theme_stats.sort(key=lambda x: -x[1])

    lines = [
        'OVERALL SNAPSHOT (use these percentages to write majority-style '
        'conclusions, e.g. "اکثر مشتریان..." or "X از Y نفر (Z٪)..."):',
        f'  Total customers: {total}',
        f'  With an action logged: {with_action} ({with_action/total*100:.0f}% of total)' if total else '',
        f'  Without any action: {without_action} ({without_action/total*100:.0f}% of total)' if total else '',
        f'  Real applicants (متقاضی حقیقی): {real_cnt} ({real_cnt/total*100:.0f}%)' if total else '',
        f'  Legal applicants (متقاضی حقوقی): {legal_cnt} ({legal_cnt/total*100:.0f}%)' if total else '',
        f'  Bank-side / internal (غیر متقاضی واقعی): {bank_cnt} ({bank_cnt/total*100:.0f}%)' if total else '',
        '',
        f'MOST-REQUESTED GOODS (base: {non_empty_goods} customers who specified any goods):',
    ]
    for cat, names in top_goods:
        pct = (len(names) / non_empty_goods * 100) if non_empty_goods else 0
        lines.append(f'  - "{cat}": {len(names)} نفر ({pct:.0f}٪ از کسانی که کالا مشخص کرده‌اند)')

    lines.append('')
    lines.append(
        f'DOMINANT THEMES IN CUSTOMER ACTIONS, ranked by frequency '
        f'(base: {with_action} customers who have a logged action):'
    )
    any_theme = False
    for label, cnt, pct_t, pct_a in theme_stats:
        if cnt == 0:
            continue
        any_theme = True
        lines.append(
            f'  - {label}: {cnt} نفر ({pct_a:.0f}٪ از کسانی که اقدام دارند، '
            f'{pct_t:.0f}٪ از کل مشتریان)'
        )
    if not any_theme:
        lines.append('  (هیچ الگوی مشخصی یافت نشد)')

    return '\n'.join(l for l in lines if l != '')


def _compute_actions_summary(df: pd.DataFrame) -> str:
    """
    Produce a structured summary of the actions column for the LLM:
    who has actions, who doesn't, thematic patterns (bond releases,
    dissatisfaction, cancellations, etc.), and the full action text
    for each record (so the LLM can quote specifics accurately).
    """
    action_col = 'اقدامات انجام شده'
    name_col   = 'نام متقاضی'

    done, empty = [], []
    for _, row in df.iterrows():
        name   = str(row.get(name_col, '')).strip()
        action = str(row.get(action_col, '')).strip()
        (done if action else empty).append({'name': name, 'action': action})

    no_action_block = '\n'.join(f'  - {n}' for n in (x['name'] for x in empty))

    lines = [
        _compute_dominant_conclusions(df),
        '',
        f'Records WITH action logged: {len(done)} | WITHOUT any action: {len(empty)}',
        f'No-action names:\n{no_action_block}',
        '',
        _compute_pattern_matches(df),
        '',
        '=== Full action log (use this to quote specific customer situations if asked) ===',
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
        phone  = str(row.get('شماره تماس', '')).strip()
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] Record for '{matched_name}':\n"
            f"  Date: {date or '(تاریخ ثبت نشده)'}\n"
            f"  Actions: {action or '(هیچ اقدامی ثبت نشده است)'}\n"
            f"  Phone: {phone or '(شماره تماس برای این شخص ثبت نشده است)'}\n"
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
            names_block = '\n'.join(f'  - {n}' for n in names)
            enriched = (
                f'{user_prompt}\n\n'
                f"[Python pre-computation] Province '{prov}': exactly {len(subset)} records.\n"
                f"Names (already one per line — keep this format in your answer):\n{names_block}"
            )
            return enriched, _rows_to_context(subset)

    # ── 3. Applicant type ────────────────────────────
    if any(k in user_prompt for k in ['حقیقی', 'حقوقی', 'نوع متقاضی']):
        real_df  = df[df[type_col] == 'متقاضی حقیقی']
        legal_df = df[df[type_col] == 'متقاضی حقوقی']
        real_block  = '\n'.join(f'  - {n}' for n in real_df[name_col].tolist())
        legal_block = '\n'.join(f'  - {n}' for n in legal_df[name_col].tolist())
        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] "
            f"'متقاضی حقیقی': exactly {len(real_df)}:\n{real_block}\n"
            f"'متقاضی حقوقی': exactly {len(legal_df)}:\n{legal_block}"
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

    # ── 5. Bank ("بانک" = every type EXCEPT متقاضی حقیقی/حقوقی) ──
    # Per the user's explicit business rule: real end-customers are only
    # 'متقاضی حقیقی' and 'متقاضی حقوقی'. Every other applicant type —
    # بازاریابی و فروش, شعبه بانک, اعتبارات, سرپرستی — is internally
    # classified as "بانک" (bank-side / non-customer contact).
    if any(k in user_prompt for k in ['بانک', 'شعبه']):
        bank_subset = df[~df[type_col].isin(['متقاضی حقیقی', 'متقاضی حقوقی'])]
        by_subtype  = bank_subset.groupby(type_col)[name_col].apply(list).to_dict()

        lines = []
        for subtype, names in sorted(by_subtype.items(), key=lambda x: -len(x[1])):
            lines.append(f'{subtype}: exactly {len(names)} نفر')
            lines.extend(f'  - {n}' for n in names)

        enriched = (
            f'{user_prompt}\n\n'
            f"[Python pre-computation] 'بانک' = every applicant type EXCEPT "
            f"'متقاضی حقیقی' and 'متقاضی حقوقی' (i.e. بازاریابی و فروش + شعبه بانک + "
            f"اعتبارات + سرپرستی, per business definition).\n"
            f"TOTAL: exactly {len(bank_subset)} نفر.\n"
            f"Breakdown by sub-type:\n" + '\n'.join(lines)
        )
        return enriched, '\n'.join(lines)

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
        lines = []
        for cat, names in stats.items():
            if not names:
                continue
            lines.append(f'{cat}: exactly {len(names)} نفر')
            lines.extend(f'  - {n}' for n in names)
        lines.append(f'بدون کالای مشخص: exactly {len(no_goods)} نفر')
        lines.extend(f'  - {n}' for n in no_goods)
        enriched = (
            f'{user_prompt}\n\n'
            f'[Python pre-computation] Goods breakdown (each name is already on its own line — '
            f'preserve this one-name-per-line format in your answer, do NOT merge into a single line):\n'
            + '\n'.join(lines)
        )
        return enriched, '\n'.join(lines)

    # ── 7.5. Phone number overview (who HAS a phone recorded) ──
    if any(k in user_prompt for k in ['شماره تماس', 'شماره تلفن', 'تلفن']):
        has_phone   = df[df['شماره تماس'] != '']
        phone_lines = [f'{n}: {p}' for n, p in
                       zip(has_phone[name_col], has_phone['شماره تماس'])]

        if phone_lines:
            body = (
                f"[Python pre-computation] Phone numbers are recorded for ONLY "
                f"{len(has_phone)} out of {len(df)} customers (most customers have "
                f"no phone number on file). Available phone numbers:\n"
                + '\n'.join(f'  - {l}' for l in phone_lines)
            )
        else:
            body = '[Python pre-computation] No phone numbers are recorded for any customer.'

        enriched = f'{user_prompt}\n\n{body}'
        return enriched, '\n'.join(phone_lines)

    # ── 8. Actions / general analysis / overall summary ──
    if any(k in user_prompt for k in ['اقدام', 'تحلیل', 'گزارش', 'بررسی', 'وضعیت',
                                       'کلی', 'خلاصه', 'نظر کلی', 'جمع بندی',
                                       'جمع‌بندی', 'دیدگاه', 'نتیجه']):
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
