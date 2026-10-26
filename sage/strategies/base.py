"""Strategy interface and the error summary shared by the meta-prompts."""

from __future__ import annotations

import json as _json
import random as _rng
from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """A proposal strategy: given the evaluated top-P population, return Q new candidate prompts."""

    name: str

    @abstractmethod
    def propose(
        self,
        population: list[dict[str, Any]],
        task_description: str,
        num_proposals: int,
        **kwargs,
    ) -> list[str]:
        """Return ``num_proposals`` new prompts.

        ``population`` is sorted by train accuracy (best first); each entry has ``prompt``,
        ``accuracy`` and ``eval_result`` (the output of :func:`sage.evaluate.evaluate_prompt`).
        The optimizer also passes ``iter_dir`` and ``task_name`` as keyword arguments.
        """


def format_error_summary(
    eval_result: dict[str, Any] | None,
    max_errors: int = 10,
    task_name: str | None = None,
) -> str:
    """Accuracy plus a random sample of failed examples (or AppWorld task traces) for a meta-prompt."""
    if eval_result is None:
        return "No detailed evaluation results available for error analysis."

    detailed = eval_result.get("detailed_results", [])
    errors = [r for r in detailed if not r["is_correct"]]
    sampled = _rng.sample(errors, min(max_errors, len(errors))) if errors else []

    lines = [
        f"Accuracy: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})",
        f"Total errors: {len(errors)}",
        "",
    ]

    if task_name == "appworld":
        sgc = eval_result.get("sgc")
        if sgc is not None:
            lines.insert(1, f"SGC (Scenario Goal Completion): {sgc:.1%}")
        if sampled:
            lines.append(f"Sample failed tasks ({len(sampled)} shown):")
            for e in sampled:
                tid = e.get("task_id", e.get("question", "?"))
                lines.append(f"  Task: {tid}")
                instr = e.get("question", "")[:300]
                if instr and instr != tid:
                    lines.append(f"  Instruction: {instr}")
                try:
                    info = _json.loads(e.get("response", "{}"))
                    failures = info.get("failure_messages", [])
                    passes = info.get("num_passes", 0)
                    fails = info.get("num_failures", 0)
                    lines.append(f"  Assertions: {passes} passed, {fails} failed")
                    for fm in failures[:3]:
                        lines.append(f"    - {fm[:200]}")
                    transcript = info.get("transcript", [])
                    if transcript:
                        last_msgs = transcript[-4:]
                        lines.append(f"  Transcript excerpt (last {len(last_msgs)} of {len(transcript)} messages):")
                        for msg in last_msgs:
                            role = msg.get("role", "?").upper()
                            content = msg.get("content", "").strip()
                            if len(content) > 300:
                                content = content[:300] + "..."
                            lines.append(f"    [{role}] {content}")
                except Exception:
                    pass
                lines.append("")
    elif sampled:
        lines.append(f"Sample errors ({len(sampled)} shown):")
        for e in sampled:
            q = e["question"][:300] + ("..." if len(e["question"]) > 300 else "")
            lines.append(f"  Q: {q}")
            lines.append(f"  Expected: {e['target']}")
            lines.append(f"  Got:      {e['predicted']}")
            reasoning = _extract_reasoning(e.get("response", ""))
            if reasoning:
                lines.append(f"  Model reasoning: {reasoning}")
            lines.append("")
    return "\n".join(lines)


def _extract_reasoning(response: str) -> str:
    """Pull the reasoning field from a JSON model response, or a truncated snippet."""
    try:
        parsed = _json.loads(response)
        text = parsed.get("reasoning", "")
        if not isinstance(text, str):
            text = str(text)
    except Exception:
        text = response
    text = text.strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text
