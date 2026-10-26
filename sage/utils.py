"""Shared utilities: answer extraction, token counting."""

import json
import re

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _extract_boxed_content(text: str) -> str | None:
    pattern = r"\\boxed\{"
    match = re.search(pattern, text)
    if not match:
        return None

    start = match.end() - 1
    brace_count = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1 : i]
        i += 1
    return None


def extract_answer(response: str) -> str:
    """Extract final answer from model response (JSON preferred, then fallbacks)."""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", "No final answer found"))
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # Finish[...] format
    matches = re.findall(r"Finish\[(.*?)\]", response)
    if matches:
        return matches[-1]

    # "final_answer": "..." (double quotes)
    matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
    if matches:
        return matches[-1]

    # 'final_answer': '...' (single quotes)
    matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
    if matches:
        return matches[-1]

    # final_answer: <value> (unquoted)
    matches = re.findall(r"['\"]final_answer['\"]\s*:\s*([^,}]+)", response)
    if matches:
        answer = re.sub(r"[,}]*$", "", matches[-1].strip())
        return answer

    # "The final answer is: \boxed{...}"
    fa_pattern = r"[Tt]he final answer is:?\s*\$?\\boxed\{"
    match = re.search(fa_pattern, response)
    if match:
        boxed = _extract_boxed_content(response[match.start() :])
        if boxed:
            return boxed

    # "The final answer is X"
    matches = re.findall(r"[Tt]he final answer is:?\s*([^\n.]+)", response)
    if matches:
        answer = matches[-1].strip()
        answer = re.sub(r"^\$?\\boxed\{([^}]+)\}\$?$", r"\1", answer)
        answer = answer.replace("$", "").strip()
        if answer:
            return answer

    return "No final answer found"
