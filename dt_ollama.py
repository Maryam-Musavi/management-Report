"""
**************************************************
Dariush Tasdighi Custom 'ollama' Package Module
**************************************************
"""

from typing import Final
from typing import Optional

import logging

VERSION: Final[str] = "2.5"
TEMPERATURE: Final[float]  = 0.1     # دقت حداکثری برای تحلیل داده
NUM_CTX: Final[int]        = 16384   # پنجره context — بدترین حالت واقعی پروژه (خلاصه گزارش کامل) حدود ۶۰۰۰ توکن است؛ ۱۶۳۸۴ حاشیه امنیت کافی می‌دهد بدون فشار غیرضروری به RAM
NUM_PREDICT: Final[int]    = 4096    # حداکثر توکن خروجی — پیش‌فرض Ollama (~128) باعث قطع پاسخ‌های تحلیلی می‌شد

MODEL_NAME: Final[str] = "llama3.1:8b"
BASE_URL_OFFLINE: Final[str] = "http://127.0.0.1:11434"

logger = logging.getLogger(name=__name__)
logger.addHandler(hdlr=logging.NullHandler())


def get_offline_client(base_url: str = BASE_URL_OFFLINE):
    """Get offline Ollama client"""
    from ollama import Client
    return Client(host=base_url)


def chat(
    messages: list[dict],
    think: bool = False,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    base_url: str = BASE_URL_OFFLINE,
) -> tuple[Optional[str], int, int, Optional[str]]:
    """
    Chat with Ollama service.

    Returns:
        tuple: (assistant_answer, prompt_tokens, completion_tokens, error_message)
        error_message is None on success, or a human-readable string on failure.
    """
    from ollama import ChatResponse

    try:
        client = get_offline_client(base_url=base_url)
        logger.debug(msg=f"Ollama '{model_name}' chat started...")

        response: ChatResponse = client.chat(
            think=think,
            stream=False,
            model=model_name,
            messages=messages,
            options={
                "temperature": temperature,
                "num_ctx": NUM_CTX,          # افزایش context window
                "num_predict": NUM_PREDICT,  # جلوگیری از قطع‌شدن پاسخ‌های طولانی
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

        return assistant_answer, prompt_tokens, completion_tokens, None

    except Exception as e:
        # مهم: قبلاً این خطا فقط به یک logger با NullHandler می‌رفت و
        # هیچ‌جا (نه در ترمینال، نه در UI) دیده نمی‌شد. حالا هم در کنسول
        # چاپ می‌شود و هم متن آن به caller برگردانده می‌شود تا در صورت
        # نیاز در رابط کاربری Streamlit نمایش داده شود.
        error_text = f"Ollama chat error ({type(e).__name__}): {e}"
        print(f"⛔ {error_text}")   # در ترمینالی که streamlit run اجرا شده دیده می‌شود
        logger.error(msg=error_text)
        return None, 0, 0, error_text


if __name__ == "__main__":
    print("این ماژول نباید مستقیم اجرا شود.")
