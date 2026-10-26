"""SPO-GA (genetic algorithm): LLM-driven crossover of two parents and error-informed mutation of one."""

from __future__ import annotations

import random as rng
from typing import Any

import openai

from ..llm import llm_call
from ..prompts import SPO_GA_CROSSOVER_PROMPT, SPO_GA_MUTATION_PROMPT, SPO_RS_PROMPT
from .base import Strategy, format_error_summary


class GeneticAlgorithm(Strategy):
    """SPO-GA: a share ``crossover_ratio`` of candidates by crossover, the rest by mutation."""

    name = "spo-ga"

    def __init__(
        self,
        client: openai.OpenAI,
        meta_model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        crossover_ratio: float = 0.3,
        max_error_examples: int = 10,
    ):
        self.client = client
        self.meta_model = meta_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.crossover_ratio = crossover_ratio
        self.max_error_examples = max_error_examples

    def propose(
        self,
        population: list[dict[str, Any]],
        task_description: str,
        num_proposals: int,
        **kwargs,
    ) -> list[str]:
        self._task_name = kwargs.get("task_name")
        if len(population) < 2:
            return self._bootstrap_proposals(task_description, num_proposals, population)

        num_crossover = max(1, round(num_proposals * self.crossover_ratio))
        num_mutation = num_proposals - num_crossover

        new_prompts: list[str] = []

        for i in range(num_crossover):
            print(f"  [SPO-GA] Crossover {i + 1}/{num_crossover}...")
            new_prompts.append(self._crossover(population, task_description))

        for i in range(num_mutation):
            print(f"  [SPO-GA] Mutation {i + 1}/{num_mutation}...")
            new_prompts.append(self._mutate(population, task_description))

        return new_prompts

    def _crossover(
        self,
        population: list[dict[str, Any]],
        task_description: str,
    ) -> str:
        """Merge two parents into one child prompt."""
        parent_a, parent_b = rng.sample(population, 2)
        if parent_a["accuracy"] < parent_b["accuracy"]:
            parent_a, parent_b = parent_b, parent_a
        meta_prompt = SPO_GA_CROSSOVER_PROMPT.format(
            task_description=task_description,
            accuracy_a=parent_a["accuracy"],
            accuracy_b=parent_b["accuracy"],
            prompt_a=parent_a["prompt"],
            prompt_b=parent_b["prompt"],
            error_summary_a=format_error_summary(
                parent_a.get("eval_result"),
                max_errors=5,
                task_name=getattr(self, "_task_name", None),
            ),
            error_summary_b=format_error_summary(
                parent_b.get("eval_result"),
                max_errors=5,
                task_name=getattr(self, "_task_name", None),
            ),
        )
        return llm_call(
            self.client,
            self.meta_model,
            meta_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ).strip()

    def _mutate(
        self,
        population: list[dict[str, Any]],
        task_description: str,
    ) -> str:
        """Improve one tournament-selected parent with an error-informed rewrite."""
        # Tournament selection: pick 2, mutate the better one
        candidates = rng.sample(population, min(2, len(population)))
        parent = max(candidates, key=lambda c: c["accuracy"])

        error_summary = format_error_summary(
            parent.get("eval_result"),
            task_name=getattr(self, "_task_name", None),
        )
        meta_prompt = SPO_GA_MUTATION_PROMPT.format(
            task_description=task_description,
            accuracy=parent["accuracy"],
            error_summary=error_summary,
            prompt=parent["prompt"],
        )
        return llm_call(
            self.client,
            self.meta_model,
            meta_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ).strip()

    def _bootstrap_proposals(
        self,
        task_description: str,
        num_proposals: int,
        population: list[dict[str, Any]],
    ) -> list[str]:
        """With fewer than two parents, half the candidates come from SPO-RS rewrites and half from mutation."""
        best = max(population, key=lambda e: e["accuracy"])
        error_summary = format_error_summary(
            best.get("eval_result"),
            task_name=getattr(self, "_task_name", None),
        )

        new_prompts: list[str] = []
        num_random = num_proposals // 2
        num_mut = num_proposals - num_random

        for i in range(num_random):
            print(f"  [SPO-GA] Bootstrap random {i + 1}/{num_random}...")
            meta_prompt = SPO_RS_PROMPT.format(
                task_description=task_description,
                best_prompt=best["prompt"],
                best_accuracy=best["accuracy"],
                error_summary=error_summary,
            )
            text = llm_call(
                self.client,
                self.meta_model,
                meta_prompt,
                max_tokens=self.max_tokens,
                temperature=0.9,
            ).strip()
            new_prompts.append(text)

        for i in range(num_mut):
            print(f"  [SPO-GA] Bootstrap mutation {i + 1}/{num_mut}...")
            new_prompts.append(self._mutate(population, task_description))

        return new_prompts
