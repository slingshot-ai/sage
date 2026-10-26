"""SPO-RS (residual search): Q independent rewrites of the best prompt, conditioned on its errors, at high temperature."""

from __future__ import annotations

from typing import Any

import openai

from ..llm import llm_call
from ..prompts import SPO_RS_PROMPT
from .base import Strategy, format_error_summary


class ResidualSearch(Strategy):
    """SPO-RS: sample candidates from the meta-model, seeded with the best prompt and its errors."""

    name = "spo-rs"

    def __init__(
        self,
        client: openai.OpenAI,
        meta_model: str,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        max_error_examples: int = 10,
    ):
        self.client = client
        self.meta_model = meta_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_error_examples = max_error_examples

    def propose(
        self,
        population: list[dict[str, Any]],
        task_description: str,
        num_proposals: int,
        **kwargs,
    ) -> list[str]:
        task_name = kwargs.get("task_name")
        best = max(population, key=lambda e: e["accuracy"])
        error_summary = format_error_summary(
            best.get("eval_result"),
            max_errors=self.max_error_examples,
            task_name=task_name,
        )
        meta_prompt = SPO_RS_PROMPT.format(
            task_description=task_description,
            best_prompt=best["prompt"],
            best_accuracy=best["accuracy"],
            error_summary=error_summary,
        )
        print(f"  [SPO-RS] Seeding from best prompt (accuracy: {best['accuracy']:.3f})")

        new_prompts: list[str] = []
        for i in range(num_proposals):
            print(f"  [SPO-RS] Generating prompt {i + 1}/{num_proposals}...")
            text = llm_call(
                self.client,
                self.meta_model,
                meta_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ).strip()
            new_prompts.append(text)

        return new_prompts
