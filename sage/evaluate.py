"""Parallel prompt evaluation on a set of samples (SAGE / SPO-RS / SPO-GA runs)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai

from .data import DataProcessor
from .llm import llm_call
from .prompts import EVAL_TEMPLATE
from .utils import extract_answer


def evaluate_single_sample(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    sample: dict[str, Any],
    index: int,
    data_processor: DataProcessor,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    full_prompt = EVAL_TEMPLATE.format(
        system_prompt=prompt,
        question=sample["question"],
        context=sample.get("context", ""),
    )
    response = llm_call(client, model, full_prompt, max_tokens=max_tokens, temperature=0.0)
    predicted = extract_answer(response)
    is_correct = data_processor.answer_is_correct(predicted, sample["target"])
    return {
        "index": index,
        "question": sample["question"],
        "context": sample.get("context", ""),
        "target": sample["target"],
        "predicted": predicted,
        "is_correct": is_correct,
        "response": response,
    }


def evaluate_prompt(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    samples: list[dict[str, Any]],
    data_processor: DataProcessor,
    max_tokens: int = 4096,
    max_workers: int = 20,
    label: str = "eval",
) -> dict[str, Any]:
    """Evaluate a prompt on a list of samples in parallel.

    Returns dict with keys: accuracy, correct, total, detailed_results.
    """
    print(f"  [{label}] Evaluating {len(samples)} samples ({max_workers} workers)...")

    results: list[dict[str, Any]] = []
    predictions: list[str] = []
    targets: list[str] = []

    def _worker(idx: int, sample: dict) -> dict[str, Any]:
        return evaluate_single_sample(
            client,
            model,
            prompt,
            sample,
            idx,
            data_processor,
            max_tokens,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i, s): i for i, s in enumerate(samples)}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result()
                results.append(result)
                predictions.append(result["predicted"])
                targets.append(result["target"])
            except Exception as e:
                idx = futures[future]
                print(f"  [{label}] Error on sample {idx}: {type(e).__name__}: {e}")

            if done_count % 50 == 0:
                acc_so_far = sum(r["is_correct"] for r in results) / len(results) if results else 0
                print(f"  [{label}] Progress: {done_count}/{len(samples)}, accuracy so far: {acc_so_far:.3f}")

    correct = sum(r["is_correct"] for r in results)
    total = len(results)
    accuracy = data_processor.evaluate_accuracy(predictions, targets) if predictions else 0.0

    print(f"  [{label}] Done — accuracy: {accuracy:.3f} ({correct}/{total})")
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "detailed_results": sorted(results, key=lambda r: r["index"]),
    }
