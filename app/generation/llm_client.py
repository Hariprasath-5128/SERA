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
_GLOBAL_USE_FALLBACK = False

def _get_client(use_fallback: bool = False) -> OpenAI:
    """
    Returns a fresh OpenAI-compatible client on every call.
    Reads API key and base_url live from config so .env changes
    are always picked up without restarting the process.
    """
    api_key = config.OPENAI_API_KEY_FALLBACK if use_fallback and config.OPENAI_API_KEY_FALLBACK else config.OPENAI_API_KEY
    client = OpenAI(
        api_key=api_key or "not-required",
        base_url=config.LLM_BASE_URL,
    )
    logger.debug(
        "llm_client: using model=%s  base_url=%s",
        config.SUMMARIZER_LLM_MODEL,
        config.LLM_BASE_URL,
    )
    return client


# ---------------------------------------------------------------------------
# Retry policy  (3 attempts, exponential back-off 1 s → 10 s)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=30),
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
    # if json_mode:
    #     kwargs["response_format"] = {"type": "json_object"}

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

    global _GLOBAL_USE_FALLBACK
    
    try:
        # If we previously globally fell back, don't even try the primary key
        client = _get_client(use_fallback=_GLOBAL_USE_FALLBACK)
        return _call_with_retry(
            client=client,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    except Exception as e:
        # If we haven't globally fallen back yet, do it now
        if not _GLOBAL_USE_FALLBACK and config.OPENAI_API_KEY_FALLBACK:
            logger.warning("llm_client: Primary API key failed. Switching to secondary API key globally. Error: %s", e)
            _GLOBAL_USE_FALLBACK = True
            client_fallback = _get_client(use_fallback=True)
            return _call_with_retry(
                client=client_fallback,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        raise

def check_token_limits(model: str, use_fallback: bool) -> dict:
    """
    Make a dummy request to fetch token limit headers or parse rate limit exceptions.
    """
    import re
    client = _get_client(use_fallback=use_fallback)
    try:
        response = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        headers = response.headers
        return {
            "status": "active",
            "limit": headers.get("x-ratelimit-limit-tokens", "Unknown"),
            "remaining": headers.get("x-ratelimit-remaining-tokens", "Unknown"),
            "reset_time": headers.get("x-ratelimit-reset-tokens", "Unknown")
        }
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            # Groq format: Limit 100000, Used 98854, Requested 1628. Please try again in 6m56.448s
            match_limit = re.search(r"Limit\s+(\d+)", error_msg)
            match_time = re.search(r"try again in ([0-9a-zA-Z\.]+)", error_msg)
            
            limit = match_limit.group(1) if match_limit else "Unknown"
            reset = match_time.group(1) if match_time else "Unknown"
            
            return {
                "status": "rate_limited",
                "limit": limit,
                "remaining": 0,
                "reset_time": reset
            }
        else:
            return {
                "status": "error",
                "message": error_msg
            }
