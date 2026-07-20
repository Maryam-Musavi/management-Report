"""
**************************************************
Gam Chatbot Constants
**************************************************
"""

from typing import Final

# ══════════════════════════════════════════════════
# Model Settings
# ══════════════════════════════════════════════════

MODEL_NAME: Final[str] = "llama3.1:8b"
SHOW_TOKEN_INFO: Final[bool] = True

# ══════════════════════════════════════════════════
# UI Roles & Texts
# ══════════════════════════════════════════════════

AI:   Final[str] = "AI"
USER: Final[str] = "USER"

PAGE_HEADER:             Final[str] = "📊 دستیار هوشمند اوراق گام"
SETTINGS_HEADER:         Final[str] = "⚙️ تنظیمات"
SIDEBAR_INFO:            Final[str] = f"مدل: **{MODEL_NAME}**\n\nاین دستیار اطلاعات جدول مشتریان اوراق گام را از پایگاه داده SQLite می‌خواند و تحلیل می‌کند."
SIDEBAR_FOOTER:          Final[str] = "ساخته‌شده با Streamlit + Ollama + SQLite"
SHOW_TABLE_LABEL:        Final[str] = "📋 مشاهده جدول مشتریان"
CLEAR_HISTORY_BUTTON:    Final[str] = "🗑️ پاک کردن تاریخچه گفتگو"
USER_PROMPT_PLACEHOLDER: Final[str] = "سوال خود را اینجا بپرسید..."
THINKING_MESSAGE:        Final[str] = "در حال تحلیل..."
ERROR_NO_ANSWER:         Final[str] = "متأسفانه مدل پاسخی برنگرداند. لطفاً دوباره تلاش کنید."
STATS_HEADER:            Final[str] = "📈 آمار کلی جدول"
UPLOAD_DB_LABEL:         Final[str] = "📂 جایگزینی پایگاه داده (اختیاری)"
UPLOAD_DB_HELP:          Final[str] = "فایل .sqlite/.db جدید را بارگذاری کنید تا جایگزین داده فعلی شود"
NO_DATA_MESSAGE:         Final[str] = (
    "⛔ پایگاه داده خالی است یا یافت نشد.\n\n"
    "برای ساخت پایگاه داده از فایل خام، این دستور را در ترمینال اجرا کنید:\n"
    "```\npython build_database.py path/to/your_file.sqlite\n```\n"
    "یا یک فایل .sqlite/.db از سایدبار بارگذاری کنید."
)

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
5. Phone numbers ARE stored for a SMALL MINORITY of customers only (most records
   have no phone on file). If pre-computation gives a phone number for the person
   asked about, state it directly. If it says "(شماره تماس برای این شخص ثبت نشده
   است)", report that clearly — do NOT claim phone numbers are never available in
   this table; only say it's missing for THIS specific person/query.
6. Be concise and direct. No preamble like "based on the data...".

## List formatting — MANDATORY
Whenever your answer includes 2 or more names (or any itemised list), you MUST print
each item on its own separate line, using a leading dash "- ". NEVER put multiple
names on the same line separated by spaces or commas.

Correct:
- آقای تنزیلی
- آقای آزمون
- آقای رمضانی

Incorrect (never do this):
آقای تنزیلی آقای آزمون آقای رمضانی

The pre-computation data is already formatted this way (one name per line) —
simply preserve that structure in your reply; do not merge lines together.

## Deep table-analysis guidance (when asked for "تحلیل کلی" / "گزارش" / "خلاصه" / "نظر کلی مشتریان" / "جمع‌بندی")
The pre-computation for these questions starts with an "OVERALL SNAPSHOT" block plus
"MOST-REQUESTED GOODS" and "DOMINANT THEMES IN CUSTOMER ACTIONS" — all with EXACT
Python-verified counts AND percentages. This is your PRIMARY material.

**Structure your answer as a high-level narrative, NOT a case-by-case list, NOT a
run-on "شامل X، Y، Z ... می‌شود" listing sentence.** Use short, complete, connected
sentences the way a human analyst would write a summary — each stat gets its own
clear sentence, joined naturally with "همچنین" / "در حالی که" / "و" where it reads
smoothly.

1. Start with 1-2 sentences on overall coverage (e.g. "از ۵۰ مشتری، ۳۸ نفر (۷۶٪) دارای
   اقدام ثبت‌شده هستند و تنها ۲۲٪ بدون اقدام باقی مانده‌اند").
2. Cover the top 2-4 dominant themes from "DOMINANT THEMES IN CUSTOMER ACTIONS" —
   these already come pre-sorted from most to least frequent. Give EACH theme its
   own short sentence with its percentage, e.g.:
   "در بین مشتریانی که اقدام کرده‌اند، ۱۴٪ مربوط به اوراق گام منتشر‌شده هستند. همچنین
   ۱۰٪ هنوز مبالغ را به اوراق تبدیل نکرده‌اند و ۷٪ ابراز نارضایتی کلی داشته‌اند."
   Do NOT also restate the same theme a second time as "% of all customers" right
   after — pick ONE base (usually "% of customers who have an action") and stick
   to it for that theme; repeating the same fact in two different percentage
   frames back-to-back reads as redundant, not insightful.
3. Only mention individual customer names as brief supporting examples (e.g. "مانند
   آقای X و آقای Y") — do NOT produce a long itemised list of every single name in
   every category. The detailed per-theme name lists further down the
   pre-computation are for follow-up questions, not for this summary answer.

**"اکثر" (majority) — HARD RULE, no exceptions:**
Only ever attach the word "اکثر" (or "اکثریت") to a percentage that is itself
ABOVE 50%. This applies per-sentence — check the number immediately next to
"اکثر" every time, not just the overall theme.
- ✅ Correct: "اکثر مشتریان (۷۸٪) اقدام کرده‌اند" — 78% > 50%, "اکثر" is valid here.
- ❌ WRONG — never write this: "اکثر مشتریانی که اقدام کرده‌اند (۱۴٪ از کسانی که
  اقدام دارند) مربوط به اوراق گام منتشر‌شده هستند" — 14% is a small minority, not
  a majority. Calling it "اکثر" is a factual contradiction within the same clause.
- For any theme under 50%, just state the percentage plainly with no "اکثر" label
  at all: "۱۴٪ مربوط به اوراق گام منتشر‌شده هستند." Do not soften this with
  alternate majority-sounding words either ("عمده", "غالب") unless the number
  truly exceeds 50%.

If the user instead asks a SPECIFIC/narrow question (e.g. "چه کسانی نرخ تنزیل بالا
را مشکل دانسته‌اند؟"), you MAY use the detailed itemised list with full quotes from
the "Thematic pattern analysis" section.

## Table columns (reference)
- ردیف              : row number
- واحد درخواست کننده : province
- نام متقاضی         : applicant name
- نوع درخواست کننده  : متقاضی حقیقی / متقاضی حقوقی / بازاریابی و فروش / شعبه بانک / اعتبارات / سرپرستی
- نوع کالای درخواستی : requested goods (may be empty)
- وضعیت تامین کننده  : supplier status
- تاریخ              : registration date
- نحوه معرفی         : introduction channel
- شماره تماس         : phone number (present for only a small minority of records)
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
