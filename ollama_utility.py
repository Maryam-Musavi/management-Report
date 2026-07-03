"""
Ollama client utility — based on the pattern used by your instructor.
Uses the official `ollama` Python package (Client) instead of raw HTTP requests.
"""

from typing import Final, Optional
from ollama import Client, ChatResponse

import logging

VERSION: Final[str] = "1.0"
TEMPERATURE: Final[float] = 0.1

MODEL_NAME: Final[str] = "llama3.1:8b"
BASE_URL_OFFLINE: Final[str] = "http://127.0.0.1:11434"

logger = logging.getLogger(name=__name__)
logger.addHandler(hdlr=logging.NullHandler())


def get_offline_client(base_url: str = BASE_URL_OFFLINE) -> Client:
    """Get a client pointing to a local Ollama server."""
    return Client(host=base_url)


def chat(
    messages: list[dict],
    think: bool = False,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    base_url: str = BASE_URL_OFFLINE,
) -> tuple[Optional[str], int, int]:
    """
    Chat with the local Ollama service.

    messages: list of {"role": "user"/"assistant"/"system", "content": "..."}
    Returns: (assistant_answer, prompt_tokens, completion_tokens)
    """

    client = get_offline_client(base_url=base_url)

    logger.debug(msg=f"Ollama '{model_name}' chat started...")

    response: ChatResponse = client.chat(
    think=think,
    stream=False,
    model=model_name,
    messages=messages,
    options={
        "temperature": temperature,
        "num_ctx": 16384  # <--- اضافه کردن این خط، پنجره دید مدل را به ۱۶ هزار توکن افزایش می‌دهد
    },
)

    logger.debug(msg=f"Ollama '{model_name}' chat finished.")

    assistant_answer: Optional[str] = response.message.content

    prompt_tokens: int = 0
    completion_tokens: int = 0

    if assistant_answer:
        if response.eval_count:
            completion_tokens = response.eval_count
        if response.prompt_eval_count:
            prompt_tokens = response.prompt_eval_count

    return assistant_answer, prompt_tokens, completion_tokens

def generate_pandas_code(question: str, columns: list, sample_data_json: str, base_url: str, model_name: str) -> str:
    """
    تبدیل سوال کاربر به کد پانداس با استفاده از ال‌ال‌ام
    """
    client = get_offline_client(base_url=base_url)
    
    system_prompt = f"""You are an expert Python data analyst. Your ONLY job is to convert a Persian user question into a valid single-line or multi-line pandas expression that answers the question based on a DataFrame named 'df'.

Available Columns in 'df': {columns}
Sample Data Structure (JSON):
{sample_data_json}

STRICT RULES:
1. Return ONLY the executable Python code block. No explanations, no markdown (do NOT use ```python), no extra text.
2. Assume the DataFrame is already loaded and named 'df'.
3. Clean user names if needed (handle both string exact match or .str.contains).
4. For text searches (like names or statuses), always use `.str.contains('keyword', na=False, case=False)` to be safe.
5. If the result is a list of people, make sure the expression extracts the name column.

EXAMPLES:
Q: از استان بوشهر چند نفر تماس گرفته اند؟
A: len(df[df['استان'].astype(str).str.contains('بوشهر', na=False)])

Q: چه اشخاصی از استان فارس تماس گرفته اند؟
A: df[df['استان'].astype(str).str.contains('فارس', na=False)]['نام متقاضی'].dropna().unique().tolist()

Q: شماره تماس آقای علی علوی را بهم بده.
A: df[df['نام متقاضی'].astype(str).str.contains('علی علوی', na=False)]['شماره تماس'].dropna().tolist()

Q: اوراق گام چه اشخاصی منتشر شده است؟
A: df[df['اقدامات انجام شده'].astype(str).str.contains('منتشر', na=False)]['نام متقاضی'].dropna().unique().tolist()

Q: چه اشخاصی ناراضی هستند؟
A: df[df['اقدامات انجام شده'].astype(str).str.contains('ناراضی|نارضایتی', na=False)]['نام متقاضی'].dropna().unique().tolist()

Q: جمعا از چند استان تماس گیرنده داریم؟
A: df['استان'].dropna().nunique()

Q: چند متقاضی حقیقی و حقوقی تماس گرفته اند؟
A: df['نوع متقاضی'].value_counts().to_dict()
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nPandas Code:"}
    ]

    response = client.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.0, "num_ctx": 4096} # دما صفر برای دقت بالا در کدنویسی
    )
    
    # تمیز کردن خروجی جی‌پی‌تی از مارک‌داون احتمالی
    code = response.message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code