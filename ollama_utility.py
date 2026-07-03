"""
Ollama client utility — based on the pattern used by your instructor.
Uses the official `ollama` Python package (Client) instead of raw HTTP requests.
"""

from typing import Final, Optional
from ollama import Client, ChatResponse

import logging

VERSION: Final[str] = "1.0"
TEMPERATURE: Final[float] = 0.1

MODEL_NAME: Final[str] = "qwen2.5:7b"
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
