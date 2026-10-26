"""AppWorld evaluation: a ReAct code agent runs each task with the optimised playbook, scored by AppWorld's evaluator.

The agent and its few-shot prompt (``prompts/appworld_react.txt``) follow ACE's AppWorld agent
(https://github.com/ace-agent/ace, Apache-2.0). Results use the same format as the other tasks.
"""

from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "appworld_react.txt"

APPWORLD_AVAILABLE = False
try:
    from appworld import (  # type: ignore[import-untyped]
        AppWorld,
        evaluate_task,
        load_task_ids,
    )

    APPWORLD_AVAILABLE = True
except ImportError:
    pass


def _require_appworld() -> None:
    if not APPWORLD_AVAILABLE:
        raise ImportError(
            "AppWorld tasks need the appworld package: "
            "pip install appworld && appworld install && appworld download data"
        )


_FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)


def _extract_code(text: str) -> tuple[str, str]:
    """Extract first python code block; return (code, fixed_text)."""
    if text is None:
        return "", ""
    for m in _FULL_CODE_RE.finditer(text):
        return m.group(1).strip(), text[: m.end()]
    partial = _PARTIAL_CODE_RE.match(text, re.DOTALL)
    if partial:
        code = partial.group(1).strip()
        fixed = text if text.endswith("\n") else text + "\n"
        return code, fixed + "```"
    return "", text


def _text_to_messages(text: str) -> list[dict[str, str]]:
    """Parse USER:/ASSISTANT:/A: delimited prompt into message dicts."""
    messages: list[dict[str, str]] = []
    last_start = 0
    for m in re.finditer(r"(USER|ASSISTANT|SYSTEM|A):\n", text, re.IGNORECASE):
        last_end = m.span()[0]
        if not messages:
            if last_end != 0:
                raise ValueError(f"Prompt starts without a role: {text[:last_end]}")
        else:
            messages[-1]["content"] = text[last_start:last_end]
        role = m.group(1).lower()
        if role == "a":
            role = "assistant"
        messages.append({"role": role, "content": ""})
        last_start = m.span()[1]
    if messages:
        messages[-1]["content"] = text[last_start:]
    return messages


def _truncate_output(content: str, max_len: int = 20_000) -> str:
    if len(content) > max_len:
        return content[:max_len] + "\n[REST NOT SHOWN FOR BREVITY]"
    return content


def _chat_completion(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    retries: int = 50,
    retry_delay: float = 10.0,
) -> str:
    """Multi-turn chat completion call with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not response or not response.choices:
                raise ValueError("Empty response from API")
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            return content
        except Exception as e:
            retryable = [
                "timeout",
                "rate limit",
                "429",
                "500",
                "502",
                "503",
                "internal server error",
                "connection",
                "empty response",
            ]
            if any(k in str(e).lower() for k in retryable) and attempt < retries:
                sleep = min(retry_delay * (1.5 ** (attempt - 1)), 120.0)
                print(f"  [AppWorld] LLM retry {attempt}/{retries}: {e}. Sleeping {sleep:.0f}s...")
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("Unreachable")


def run_appworld_task(
    client: openai.OpenAI | None,
    model: str,
    playbook: str,
    task_id: str,
    experiment_name: str = "sage_eval",
    max_steps: int = 40,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    random_seed: int = 123,
) -> dict[str, Any]:
    """Run the ReAct agent on one AppWorld task with the given playbook.

    Returns a dict with task_id, is_correct (TGC=1), num_passes, num_failures,
    failure_messages, and the task instruction.

    If *client* is ``None`` a fresh OpenRouter client is created (used by
    ``ProcessPoolExecutor`` workers that cannot receive a pickled client).
    """
    _require_appworld()
    if client is None:
        from .llm import create_client

        client = create_client()
    from jinja2 import Template  # type: ignore[import-untyped]

    prompt_template = _PROMPT_TEMPLATE_PATH.read_text()
    template = Template(prompt_template)

    with AppWorld(
        task_id=task_id,
        experiment_name=experiment_name,
        random_seed=random_seed,
    ) as world:
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for k, v in world.task.app_descriptions.items()],
            indent=1,
        )
        rendered = template.render(
            input_str=world.task.instruction,
            main_user=world.task.supervisor,
            app_descriptions=app_descriptions,
            playbook=playbook,
        )
        rendered = rendered + "\n\n"
        instruction = world.task.instruction

        messages = _text_to_messages(rendered)
        num_instruction_messages = len(messages)

        for step in range(max_steps):
            trimmed = _trim_messages(messages, num_instruction_messages)
            output_text = _chat_completion(
                client,
                model,
                trimmed,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            code, fixed_text = _extract_code(output_text)
            messages.append({"role": "assistant", "content": fixed_text + "\n\n"})

            if not code:
                messages.append(
                    {
                        "role": "user",
                        "content": "Output:\n```\nNo code was provided. Please provide a python code block.\n```\n\n",
                    }
                )
                continue

            execution_output = world.execute(code)
            maybe_nl = "\n" if not execution_output.endswith("\n") else ""
            messages.append(
                {
                    "role": "user",
                    "content": f"Output:\n```\n{_truncate_output(execution_output)}{maybe_nl}```\n\n",
                }
            )

            if world.task_completed():
                break

    # Full conversation (template + ReAct turns).
    transcript = [{"role": m["role"], "content": m["content"]} for m in messages]

    tracker = evaluate_task(task_id=task_id, experiment_name=experiment_name)
    num_passes = len(tracker.passes)
    num_failures = len(tracker.failures)
    failure_msgs = [f.get("requirement", str(f)) if isinstance(f, dict) else str(f) for f in tracker.failures]

    return {
        "task_id": task_id,
        "instruction": instruction,
        "is_correct": num_failures == 0,
        "num_passes": num_passes,
        "num_failures": num_failures,
        "failure_messages": failure_msgs,
        "transcript": transcript,
    }


def _trim_messages(
    messages: list[dict[str, str]],
    num_instruction_messages: int,
    max_output_length: int = 400_000,
) -> list[dict[str, str]]:
    """Trim older observation blocks when context gets too long."""
    msgs = copy.deepcopy(messages)
    pre = msgs[: num_instruction_messages - 1] if num_instruction_messages > 1 else []
    post = msgs[max(num_instruction_messages - 1, 0) :]

    def _total_len() -> int:
        return sum(len(m.get("content", "")) for m in post)

    obs_idx = 0
    while _total_len() > max_output_length:
        found = False
        if obs_idx < len(post) - 5:
            for i, m in enumerate(post[obs_idx:], start=obs_idx):
                if m["role"] == "user" and m["content"].startswith("Output:"):
                    m["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                    found = True
                    obs_idx = i + 1
                    break
            if not found:
                obs_idx = len(post)
        if not found and len(post) > 2:
            if not post[0]["content"].endswith("[TRIMMED HISTORY]\n\n"):
                post[0]["content"] += "[TRIMMED HISTORY]\n\n"
            post = [post[0]] + post[2:]
            found = True
        if not found:
            break

    return pre + post


def _worker_run_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Entry point for each ``ProcessPoolExecutor`` child.

    Creates its own OpenRouter client (sockets can't cross process boundaries)
    and runs a single AppWorld task.
    """
    idx = kwargs.pop("index")
    try:
        r = run_appworld_task(client=None, **kwargs)
        r["index"] = idx
        return r
    except Exception as e:
        return {
            "index": idx,
            "task_id": kwargs["task_id"],
            "instruction": "",
            "is_correct": False,
            "num_passes": 0,
            "num_failures": 1,
            "failure_messages": [str(e)],
            "transcript": [],
            "error": str(e),
        }


def evaluate_appworld_playbook(
    client: openai.OpenAI,
    model: str,
    playbook: str,
    task_ids: list[str],
    experiment_name: str = "sage_eval",
    max_steps: int = 40,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_workers: int = 1,
    label: str = "eval",
    random_seed: int = 123,
) -> dict[str, Any]:
    """Evaluate a playbook on a list of AppWorld tasks.

    Returns a dict matching SAGE's standard format:
      accuracy, correct, total, detailed_results
    where accuracy = TGC (fraction of tasks with all tests passing).

    When *max_workers* > 1, tasks are run in parallel via
    ``ProcessPoolExecutor`` (separate processes avoid SQLite thread-safety
    issues).  Each worker creates its own OpenRouter client.
    """
    _require_appworld()
    print(f"  [{label}] Evaluating playbook on {len(task_ids)} AppWorld tasks (workers={max_workers})...")

    results: list[dict[str, Any]] = []
    exp_name = f"{experiment_name}_{label}"

    common_kwargs: dict[str, Any] = {
        "model": model,
        "playbook": playbook,
        "experiment_name": exp_name,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "random_seed": random_seed,
    }

    if max_workers <= 1:
        for idx, tid in enumerate(task_ids):
            try:
                r = run_appworld_task(client=client, task_id=tid, **common_kwargs)
                r["index"] = idx
                results.append(r)
                status = "PASS" if r["is_correct"] else "FAIL"
                print(f"  [{label}] Task {idx + 1}/{len(task_ids)} ({tid}): {status}")
            except Exception as e:
                print(f"  [{label}] Task {idx + 1}/{len(task_ids)} ({tid}): ERROR - {e}")
                results.append(
                    {
                        "index": idx,
                        "task_id": tid,
                        "instruction": "",
                        "is_correct": False,
                        "num_passes": 0,
                        "num_failures": 1,
                        "failure_messages": [str(e)],
                        "transcript": [],
                    }
                )
    else:
        work_items = [{"index": i, "task_id": tid, **common_kwargs} for i, tid in enumerate(task_ids)]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_worker_run_task, w): w for w in work_items}
            for future in as_completed(futures):
                r = future.result()  # errors already caught inside _worker_run_task
                idx, tid = r["index"], r["task_id"]
                if "error" in r:
                    print(f"  [{label}] Task ({tid}): ERROR - {r['error']}")
                else:
                    status = "PASS" if r["is_correct"] else "FAIL"
                    print(f"  [{label}] Task ({tid}): {status}")
                results.append(r)

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    total_passes = sum(r["num_passes"] for r in results)
    total_assertions = total_passes + sum(r["num_failures"] for r in results)
    sgc = total_passes / total_assertions if total_assertions > 0 else 0.0

    print(f"  [{label}] Done — TGC: {accuracy:.3f} ({correct}/{total}), SGC: {sgc:.3f}")

    # Map to SAGE's expected format
    detailed_results = []
    for r in sorted(results, key=lambda x: x["index"]):
        response_data = {
            "num_passes": r["num_passes"],
            "num_failures": r["num_failures"],
            "failure_messages": r.get("failure_messages", []),
            "transcript": r.get("transcript", []),
        }
        detailed_results.append(
            {
                "index": r["index"],
                "task_id": r["task_id"],
                "question": r.get("instruction", ""),
                "context": "",
                "target": "task_completed",
                "predicted": "completed" if r["is_correct"] else "failed",
                "is_correct": r["is_correct"],
                "response": json.dumps(response_data, ensure_ascii=False),
            }
        )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sgc": sgc,
        "detailed_results": detailed_results,
    }


def load_appworld_task_ids(split: str = "train") -> list[str]:
    """Load AppWorld task IDs for a dataset split.

    Valid splits: train, test_normal, test_challenge, dev
    """
    _require_appworld()
    return load_task_ids(dataset_name=split)


def appworld_task_ids_to_samples(task_ids: list[str]) -> list[dict[str, Any]]:
    """Convert task IDs to SAGE's expected sample format.

    Each sample is a dict with 'task_id', 'question' (instruction), 'context', 'target'.
    The instruction is loaded lazily during evaluation, so we use task_id as the key.
    """
    return [
        {
            "task_id": tid,
            "question": tid,
            "context": "",
            "target": "task_completed",
        }
        for tid in task_ids
    ]
