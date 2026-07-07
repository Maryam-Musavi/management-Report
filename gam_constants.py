"""
**************************************************
Gam Chatbot Constants
**************************************************
"""

from typing import Final
import glob
import os

# ══════════════════════════════════════════════════
# Model & File Settings
# ══════════════════════════════════════════════════

MODEL_NAME: Final[str] = "llama3.1:8b"

_BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR: str = os.path.join(_BASE_DIR, "data")


def _find_excel_file() -> str:
    for ext in ("*.xlsx", "*.xls"):
        files = glob.glob(os.path.join(_DATA_DIR, ext))
        if files:
            return files[0]
    return os.path.join(_DATA_DIR, "درخواست_کالای_مشتریان_1405_04_09.xlsx")


EXCEL_FILE_PATH: Final[str] = _find_excel_file()
SHOW_TOKEN_INFO: Final[bool] = True

# ══════════════════════════════════════════════════
# UI Roles & Texts
# ══════════════════════════════════════════════════

AI:   Final[str] = "AI"
USER: Final[str] = "USER"

PAGE_HEADER:             Final[str] = "📊 دستیار هوشمند اوراق گام"
SETTINGS_HEADER:         Final[str] = "⚙️ تنظیمات"
SIDEBAR_INFO:            Final[str] = f"مدل: **{MODEL_NAME}**\n\nاین دستیار اطلاعات جدول مشتریان اوراق گام را تحلیل می‌کند."
SIDEBAR_FOOTER:          Final[str] = "ساخته‌شده با Streamlit + Ollama"
SHOW_TABLE_LABEL:        Final[str] = "📋 مشاهده جدول مشتریان"
CLEAR_HISTORY_BUTTON:    Final[str] = "🗑️ پاک کردن تاریخچه گفتگو"
USER_PROMPT_PLACEHOLDER: Final[str] = "سوال خود را اینجا بپرسید..."
THINKING_MESSAGE:        Final[str] = "در حال تحلیل..."
ERROR_NO_ANSWER:         Final[str] = "متأسفانه مدل پاسخی برنگرداند. لطفاً دوباره تلاش کنید."
STATS_HEADER:            Final[str] = "📈 آمار کلی جدول"

# ══════════════════════════════════════════════════
# System Prompt  (English for better LLM accuracy)
# ══════════════════════════════════════════════════
#
# Architecture:
#   Python pre-computes exact facts → injects [Python pre-computation] block
#   LLM reads those facts → writes a clear Persian answer
#   LLM must NEVER re-count or re-filter on its own.
#
# {data_context} = filtered rows relevant to this query
# ══════════════════════════════════════════════════

SYSTEM_PROMPT: Final[str] = """You are a Persian-language data assistant for the "Gam Bonds" (اوراق گام) customer table.

## Your only job
Read the [Python pre-computation] block in the user message — it contains EXACT numbers and data computed directly from the database — then write a clear, natural PERSIAN answer based on those facts.

## Critical rules
1. ALWAYS reply in Persian (Farsi).
2. The [Python pre-computation] block is ground truth. Use its numbers and names EXACTLY.
   NEVER re-count, re-filter, or second-guess it.
3. If the pre-computation says a field is "(empty)" or "(هیچ اقدامی ثبت نشده است)",
   report that clearly: e.g. "برای این شخص هیچ اقدامی در سیستم ثبت نشده است."
4. If pre-computation provides an action text, quote it in full.
5. Phone numbers are NOT stored in this table.
6. For actions analysis: summarise patterns (who has actions, who doesn't, key outcomes).
7. For goods analysis: report each category with exact count and names.
8. Be concise and direct. No preamble like "based on the data...".

## Table columns (reference)
- ردیف              : row number
- واحد درخواست کننده : province
- نام متقاضی         : applicant name
- نوع درخواست کننده  : متقاضی حقیقی / متقاضی حقوقی / بازاریابی و فروش / شعبه بانک / اعتبارات / سرپرستی
- نوع کالای درخواستی : requested goods (may be empty)
- وضعیت تامین کننده  : supplier status
- تاریخ              : registration date
- نحوه معرفی         : introduction channel
- اقدامات انجام شده  : actions taken (may be empty)

## Data for this query
{data_context}
"""

# ══════════════════════════════════════════════════
# Persian UI Style
# ══════════════════════════════════════════════════

STREAMLIT_STYLE: Final[str] = """
<style>
    @import url('https://fonts.cdnfonts.com/css/iransansx');

    html, body, p, h1, h2, h3, h4, h5, h6,
    input, textarea, li, span, div, button, label {
        font-family: 'IRANSansX', Tahoma, sans-serif !important;
    }

    .block-container, section, input, textarea,
    div.stMarkdown, div.stAlert { direction: rtl; text-align: right; }

    div[data-testid="stChatMessageContent"] { direction: rtl; text-align: right; }
    div[data-testid="stChatInput"] textarea  { direction: rtl; text-align: right; }
    div[data-testid="stDataFrame"]           { direction: rtl; }
    section[data-testid="stSidebar"]         { direction: rtl; text-align: right; }
    div[data-testid="stMetric"]              { direction: rtl; text-align: right; }

    .token-info {
        font-size: 0.72rem;
        color: #999;
        direction: ltr;
        text-align: left;
        margin-top: 4px;
        padding: 2px 8px;
        background: #f5f5f5;
        border-radius: 4px;
        display: inline-block;
    }
</style>
"""
