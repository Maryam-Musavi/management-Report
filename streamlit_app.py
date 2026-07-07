"""
**************************************************
گزارش‌گیری هوشمند از جدول مشتریان اوراق گام
با استفاده از هوش مصنوعی Llama 3.1:8b و Streamlit
**************************************************
For Running:
> streamlit run ./streamlit_app.py
**************************************************
"""

import streamlit as st
import dt_ollama as ollama
import gam_constants as constants
import gam_functions as functions

# **************************************************
# مقداردهی اولیه session state
# **************************************************
functions.initial_session_state()

# **************************************************
# تنظیمات صفحه (باید اولین دستور Streamlit باشد)
# **************************************************
functions.set_page_config()

# **************************************************
# اعمال استایل فارسی
# **************************************************
st.markdown(body=constants.STREAMLIT_STYLE, unsafe_allow_html=True)

# **************************************************
# Sidebar
# **************************************************
with st.sidebar:
    st.subheader(body=constants.SETTINGS_HEADER, divider="rainbow")
    st.info(body=constants.SIDEBAR_INFO, icon="🤖")
    st.divider()

    # ─── آپلود فایل اکسل ─────────────────────────
    st.markdown("**📂 بارگذاری فایل اکسل**")
    uploaded_file = st.file_uploader(
        label="فایل اکسل مشتریان را انتخاب کنید",
        type=["xlsx", "xls"],
        help="فایل درخواست_کالای_مشتریان را اینجا بارگذاری کنید",
    )

    if uploaded_file is not None:
        functions.load_data_from_upload(uploaded_file=uploaded_file)
        st.success(body="✅ فایل با موفقیت بارگذاری شد!", icon="✅")

    st.divider()

    # ─── آمار کلی ────────────────────────────────
    functions.show_sidebar_stats()

    if "dataframe" in st.session_state:
        st.divider()

    # ─── پاک کردن تاریخچه ────────────────────────
    if st.button(label=constants.CLEAR_HISTORY_BUTTON, use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(body=constants.SIDEBAR_FOOTER)

# **************************************************
# هدر اصلی
# **************************************************
st.header(body=constants.PAGE_HEADER, divider="rainbow")

# **************************************************
# بررسی وجود داده و نمایش جدول
# **************************************************
df = functions.load_data()

if df is None:
    st.info(
        body=(
            "👆 برای شروع، فایل اکسل مشتریان را از **منوی سمت راست** بارگذاری کنید.\n\n"
            "فایل مورد نظر: `درخواست_کالای_مشتریان_1405_04_09.xlsx`"
        ),
        icon="📂",
    )
else:
    with st.expander(label=constants.SHOW_TABLE_LABEL, expanded=False):
        st.dataframe(data=df, use_container_width=True, hide_index=True)

# **************************************************
# نمایش تاریخچه گفتگو
# **************************************************
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(name=constants.USER):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(name=constants.AI):
            st.write(message["content"])
            # نمایش توکن‌های ذخیره‌شده همراه هر پیام
            if constants.SHOW_TOKEN_INFO and "tokens" in message:
                st.markdown(
                    body=(
                        f'<div class="token-info">'
                        f'🔢 ورودی: {message["tokens"]["prompt"]} | '
                        f'خروجی: {message["tokens"]["completion"]}'
                        f'</div>'
                    ),
                    unsafe_allow_html=True,
                )

# **************************************************
# دریافت سوال کاربر
# **************************************************
user_prompt: str | None = st.chat_input(
    placeholder=constants.USER_PROMPT_PLACEHOLDER,
    disabled=(df is None),
)

if user_prompt:
    user_prompt = user_prompt.strip()

if user_prompt:
    # ۱. نمایش فوری سوال کاربر
    with st.chat_message(name=constants.USER):
        st.write(user_prompt)

    # ۲. ذخیره سوال در تاریخچه
    user_message: dict = {"role": "user", "content": user_prompt}
    st.session_state.messages.append(user_message)

    # ۳. ساخت پیام‌های کامل برای ارسال به مدل
    messages_to_send: list[dict] = functions.build_messages_with_context(
        user_prompt=user_prompt,
        history=st.session_state.messages,
    )

    # ۴. دریافت پاسخ از Ollama
    with st.chat_message(name=constants.AI):
        with st.spinner(text=constants.THINKING_MESSAGE):
            assistant_answer, prompt_tokens, completion_tokens = ollama.chat(
                messages=messages_to_send,
                model_name=constants.MODEL_NAME,
            )

        if not assistant_answer:
            # برگرداندن سوال کاربر از تاریخچه در صورت خطا
            st.session_state.messages.pop()
            st.error(body=constants.ERROR_NO_ANSWER)
        else:
            # ۵. نمایش پاسخ
            st.write(assistant_answer)

            # ۶. نمایش اطلاعات توکن
            if constants.SHOW_TOKEN_INFO:
                st.markdown(
                    body=(
                        f'<div class="token-info">'
                        f'🔢 ورودی: {prompt_tokens} | '
                        f'خروجی: {completion_tokens}'
                        f'</div>'
                    ),
                    unsafe_allow_html=True,
                )

            # ۷. ذخیره پاسخ + اطلاعات توکن در تاریخچه
            assistant_message: dict = {
                "role": "assistant",
                "content": assistant_answer,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                },
            }
            st.session_state.messages.append(assistant_message)
