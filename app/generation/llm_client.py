"""
app/generation/llm_client.py
------------------------------
Shared LLM client for Phase 4 synthesis and validation tasks.

Uses SUMMARIZER_LLM_MODEL (Qwen-2.5-32B-Instruct) pointed at the local
inference endpoint defined by LLM_BASE_URL (e.g. Ollama, vLLM).
No OpenAI API key is required.

Consumed by:
  - app/synthesis/synthesizer.py     (summary generation)
  - app/validation/validator.py      (LLM-as-judge, JSON mode)
"""

import logging

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception_type,
)

from app import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton client
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    Returns a cached client pointed at LLM_BASE_URL using the
    OpenAI-compatible REST interface (Ollama / vLLM expose this by default).
    No API key is sent — the server is local and unauthenticated.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key="not-required",        # local server ignores this
            base_url=config.LLM_BASE_URL,  # e.g. http://localhost:11434/v1
        )
        logger.info(
            "llm_client: initialised — model=%s  base_url=%s",
            config.SUMMARIZER_LLM_MODEL,
            config.LLM_BASE_URL,
        )
    return _client


# ---------------------------------------------------------------------------
# Retry policy  (3 attempts, exponential back-off 1 s → 10 s)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_with_retry(
    client: OpenAI,
    messages: list,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    kwargs: dict = {
        "model":       config.SUMMARIZER_LLM_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_call(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_tokens: int = 1000,
) -> str:
    """
    Make a single LLM call using SUMMARIZER_LLM_MODEL.

    Parameters
    ----------
    system_prompt : str
        System-role instruction for the model.
    user_prompt : str
        User-role content (source material, query, etc.).
    temperature : float
        0.1 for synthesis, 0.0 for deterministic judge calls.
    json_mode : bool
        True → response_format=json_object (required by validator.py).
    max_tokens : int
        Max tokens in the response.

    Returns
    -------
    str
        Stripped text content of the model response.

    Raises
    ------
    Exception
        Re-raised after 3 failed retry attempts.
    """
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    logger.debug(
        "llm_client: call | model=%s | json_mode=%s | temp=%.2f | max_tokens=%d",
        config.SUMMARIZER_LLM_MODEL,
        json_mode,
        temperature,
        max_tokens,
    )

    return _call_with_retry(
        client=client,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )
