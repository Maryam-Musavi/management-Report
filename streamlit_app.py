"""
**************************************************
گزارش‌گیری هوشمند از جدول مشتریان اوراق گام
با استفاده از هوش مصنوعی 3:1b، Streamlit و SQLite
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

    # ─── جایگزینی پایگاه داده (اختیاری) ───────────
    # داده به‌طور خودکار از data/gam_customers.db خوانده می‌شود.
    # این آپلودر فقط برای جایگزین کردن آن با فایل sqlite دیگری است.
    st.markdown(f"**{constants.UPLOAD_DB_LABEL}**")
    uploaded_file = st.file_uploader(
        label="فایل sqlite/db را انتخاب کنید",
        type=["sqlite", "db"],
        help=constants.UPLOAD_DB_HELP,
    )
    if uploaded_file is not None:
        functions.load_data_from_upload(uploaded_file=uploaded_file)
        st.success(body="✅ پایگاه داده جایگزین شد!", icon="✅")

    st.divider()

    # ─── آمار کلی ────────────────────────────────
    functions.show_sidebar_stats()
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
# بارگذاری داده از SQLite (خودکار، بدون نیاز به آپلود)
# **************************************************
df = functions.load_data()

if df is None:
    st.info(body=constants.NO_DATA_MESSAGE, icon="📂")
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
    with st.chat_message(name=constants.USER):
        st.write(user_prompt)

    user_message: dict = {"role": "user", "content": user_prompt}
    st.session_state.messages.append(user_message)

    messages_to_send: list[dict] = functions.build_messages_with_context(
        user_prompt=user_prompt,
        history=st.session_state.messages,
    )

    with st.chat_message(name=constants.AI):
        with st.spinner(text=constants.THINKING_MESSAGE):
            assistant_answer, prompt_tokens, completion_tokens, error_detail = ollama.chat(
                messages=messages_to_send,
                model_name=constants.MODEL_NAME,
            )

        if not assistant_answer:
            st.session_state.messages.pop()
            st.error(body=constants.ERROR_NO_ANSWER)
            if error_detail:
                # جزئیات فنی خطا (مثلاً Ollama در دسترس نیست، مدل نصب نشده،
                # اتصال برقرار نشد و ...) — برای عیب‌یابی سریع‌تر کاربر
                st.caption(body=f"جزئیات فنی: `{error_detail}`")
        else:
            st.write(assistant_answer)

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

            assistant_message: dict = {
                "role": "assistant",
                "content": assistant_answer,
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                },
            }
            st.session_state.messages.append(assistant_message)
