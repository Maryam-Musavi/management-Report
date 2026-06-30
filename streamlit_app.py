"""
Streamlit Excel Q&A Chatbot
----------------------------
Refactored version of the original script:
- Uses the `ollama` package (via ollama_utility.chat) instead of raw `requests` calls
  to the Ollama HTTP API — following the pattern used in the instructor's module.
- Wrapped in a Streamlit chat UI instead of a console input() loop.

Run with:
    streamlit run streamlit_app.py
"""

import json
import streamlit as st
import pandas as pd

import ollama_utility

# =========================
# Config (Editable)
# =========================
MODEL_NAME = "qwen2.5:7b"
BASE_URL = "http://127.0.0.1:11434"
TEMPERATURE = 0.1

POSSIBLE_PROVINCE_COLUMNS = [
    "استان", "شهر", "محل", "آدرس", "موقعیت", "منطقه",
    "واحد درخواست کننده", "نام متقاضی",
]

SYSTEM_PROMPT_TEMPLATE = """شما یک دستیار هوشمند هستید که به سوالات کاربر درباره داده‌های یک فایل اکسل پاسخ می‌دهید.

اطلاعات داده‌ها:

تعداد کل رکوردها: {total_records}
تعداد استان/شهرهای مختلف: {unique_provinces}

لیست استان/شهرها و تعداد تماس‌های هرکدام:
{provinces_json}

نام ستونی که اطلاعات استان/شهر در آن قرار دارد: {province_column}

همه ستون‌های موجود در فایل:
{all_columns}

لطفاً به سوالات کاربر به زبان فارسی و با دقت پاسخ دهید. اگر سوال درباره تعداد استان‌ها یا تعداد تماس‌های هر استان است، از اطلاعات بالا استفاده کنید.
اگر سوال نیاز به اطلاعاتی دارد که در خلاصه بالا موجود نیست، صادقانه بگویید که آن اطلاعات در دسترس نیست.
"""


# =========================
# Data loading / analysis
# =========================
@st.cache_data(show_spinner=False)
def load_and_analyze_data(file_bytes: bytes):
    """Load Excel file (from bytes, so it works with Streamlit's uploader) and extract key info."""
    df = pd.read_excel(pd.io.common.BytesIO(file_bytes), sheet_name=0, engine="openpyxl")

    province_col = None
    for col in POSSIBLE_PROVINCE_COLUMNS:
        if col in df.columns:
            province_col = col
            break

    if province_col is None:
        for col in df.columns:
            if df[col].dtype == "object":
                province_col = col
                break

    provinces: dict[str, int] = {}
    if province_col:
        for value in df[province_col].dropna():
            val_str = str(value).strip()
            if val_str and val_str != "nan":
                provinces[val_str] = provinces.get(val_str, 0) + 1

    summary = {
        "total_records": len(df),
        "unique_provinces": len(provinces),
        "provinces_count": provinces,
        "province_column": province_col,
        "all_columns": df.columns.tolist(),
    }
    return summary, df


def build_system_prompt(data_summary: dict) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        total_records=data_summary["total_records"],
        unique_provinces=data_summary["unique_provinces"],
        provinces_json=json.dumps(data_summary["provinces_count"], ensure_ascii=False, indent=2),
        province_column=data_summary["province_column"],
        all_columns=data_summary["all_columns"],
    )


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="چت بات اکسل با Ollama", page_icon="🤖", layout="wide")
st.title("🤖 چت بات هوشمند برای سوالات از داده‌های اکسل")
st.caption(f"مدل: `{MODEL_NAME}` — از طریق سرور محلی Ollama")

with st.sidebar:
    st.header("⚙️ تنظیمات")
    uploaded_file = st.file_uploader("فایل اکسل را بارگذاری کنید", type=["xlsx"])
    model_name = st.text_input("نام مدل", value=MODEL_NAME)
    base_url = st.text_input("آدرس سرور Ollama", value=BASE_URL)
    temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.05)

    if st.button("🗑️ پاک کردن مکالمه"):
        st.session_state.pop("messages", None)
        st.rerun()

if uploaded_file is None:
    st.info("⬅️ لطفاً ابتدا یک فایل اکسل (.xlsx) از نوار کناری بارگذاری کنید.")
    st.stop()

data_summary, df = load_and_analyze_data(uploaded_file.getvalue())

# Stats panel
with st.expander("📊 آمار سریع داده‌ها", expanded=True):
    col1, col2 = st.columns(2)
    col1.metric("تعداد کل رکوردها", data_summary["total_records"])
    col2.metric("تعداد استان/شهر یکتا", data_summary["unique_provinces"])

    sorted_provinces = sorted(
        data_summary["provinces_count"].items(), key=lambda x: x[1], reverse=True
    )
    if sorted_provinces:
        stats_df = pd.DataFrame(sorted_provinces, columns=["استان/شهر", "تعداد تماس"])
        st.dataframe(stats_df, use_container_width=True)

    st.dataframe(df.head(10), use_container_width=True)

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": build_system_prompt(data_summary)}
    ]

# Render history (skip system message)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("سوال خود را درباره داده‌ها بپرسید...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            try:
                answer, prompt_tokens, completion_tokens = ollama_utility.chat(
                    messages=st.session_state.messages,
                    model_name=model_name,
                    temperature=temperature,
                    base_url=base_url,
                )
            except Exception as e:
                answer = f"❌ خطا در ارتباط با Ollama: {e}"
                prompt_tokens = completion_tokens = 0

        st.markdown(answer)
        st.caption(f"🔢 توکن‌ها — ورودی: {prompt_tokens} | خروجی: {completion_tokens}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
