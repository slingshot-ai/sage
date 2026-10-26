"""OpenRouter LLM client with retry logic."""

import os
import random
import time

import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_client() -> openai.OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def llm_call(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    use_json_mode: bool = False,
    retries: int = 100,
    retry_delay: float = 15.0,
) -> str:
    """Make an LLM call via OpenRouter with retry logic.

    Returns the response content string. Retries on transient errors
    (timeouts, rate limits, server errors) with exponential backoff + jitter.
    """
    for attempt in range(1, retries + 1):
        try:
            params: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_json_mode:
                params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**params)

            if not response or not response.choices:
                raise ValueError("Empty response from API")

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")

            return content

        except Exception as e:
            retryable_keywords = [
                "timeout",
                "timed out",
                "rate limit",
                "429",
                "500",
                "502",
                "503",
                "internal server error",
                "empty response",
                "none content",
                "connection",
            ]
            is_retryable = any(k in str(e).lower() for k in retryable_keywords)

            if is_retryable and attempt < retries:
                sleep = retry_delay * (1.5 ** (attempt - 1)) * random.uniform(0.5, 1.5)
                sleep = min(sleep, 120.0)
                print(f"  LLM call failed (attempt {attempt}/{retries}): {e}. Retrying in {sleep:.1f}s...")
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("Unreachable")
