"""
**************************************************
Dariush Tasdighi Custom 'ollama' Package Module
**************************************************
"""

from typing import Final
from typing import Optional

import logging

VERSION: Final[str] = "2.4"
TEMPERATURE: Final[float] = 0.1   # دقت حداکثری برای تحلیل داده
NUM_CTX: Final[int] = 32768        # پنجره context — باید از 2048 پیش‌فرض بزرگتر باشد

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
) -> tuple[Optional[str], int, int]:
    """
    Chat with Ollama service.

    Returns:
        tuple: (assistant_answer, prompt_tokens, completion_tokens)
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
                "num_ctx": NUM_CTX,      # افزایش context window
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

    except Exception as e:
        logger.error(msg=f"Ollama chat error: {e}")
        return None, 0, 0


if __name__ == "__main__":
    print("این ماژول نباید مستقیم اجرا شود.")
