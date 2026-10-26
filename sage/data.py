"""Data loading and task-specific processing / scoring."""

from __future__ import annotations

import json
import os
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def _parse_finer(context: str) -> tuple[str, str]:
    """Parse FiNER context → (input_text, question)."""
    if "Input: " in context and "Instruction: " in context:
        instruction = context.split("Input: ")[0].strip().split("Instruction: ")[1].strip()
        remaining = context.split("Input: ")[1]
        input_text = remaining.split("Answer: ")[0].strip()
        return input_text, instruction
    return "", context


def _parse_formula(context: str) -> tuple[str, str]:
    """Parse Formula context → (input_text, question)."""
    if "Question: " in context and ". Answer:" in context:
        parts = context.split("Question: ", 1)
        question_part = parts[1]
        question_text = question_part.split(". Answer:")[0].strip()
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += (
            " Your answer should be a plain floating point number, "
            "round to the nearest hundredth if necessary. Do the necessary "
            "conversions, for example 5 million should be 5000000.0. "
        )
        return "", question_text
    return "", context


def _parse_appworld(context: str) -> tuple[str, str]:
    """AppWorld tasks use task_id as the key; no parsing needed."""
    return "", context


_PARSERS = {
    "finer": _parse_finer,
    "formula": _parse_formula,
    "appworld": _parse_appworld,
}


class DataProcessor:
    """Task-specific data processing, correctness checking, and accuracy computation.

    Mirrors the ACE DataProcessor to ensure identical evaluation.
    """

    def __init__(self, task_name: str):
        if task_name not in _PARSERS:
            raise ValueError(f"Unknown task: {task_name}. Must be one of {list(_PARSERS)}")
        self.task_name = task_name

    # -- preprocessing -------------------------------------------------------

    def process(self, raw_data: list[dict]) -> list[dict]:
        parse_fn = _PARSERS[self.task_name]
        processed = []
        for item in raw_data:
            context = item.get("context", "")
            target = item.get("target", "")
            input_text, question = parse_fn(context)
            processed.append(
                {
                    "context": input_text,
                    "question": question,
                    "target": target,
                }
            )
        return processed

    # -- correctness ---------------------------------------------------------

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        if self.task_name == "finer":
            return self._finer_correct(predicted, ground_truth)
        if self.task_name == "appworld":
            return predicted == "completed"
        return self._formula_correct(predicted, ground_truth)

    def _finer_correct(self, predicted: str, ground_truth: str, return_counts: bool = False):
        pred = [v.lower().strip() for v in predicted.split(",")]
        label = [v.lower().strip() for v in ground_truth.split(",")]
        if len(pred) != len(label):
            pred = pred[: len(label)] if len(pred) > len(label) else pred + [""] * (len(label) - len(pred))
        count = 0
        for p, g in zip(pred, label):
            try:
                g = eval(g)
                p = eval(p.replace(",", "").replace("$", ""))
            except Exception:
                pass
            if p == g:
                count += 1
        if return_counts:
            return count, len(pred)
        return count == len(pred) and len(pred) > 0

    @staticmethod
    def _formula_correct(predicted: str, ground_truth: str) -> bool:
        try:
            return float(predicted.replace(",", "")) == float(ground_truth.replace(",", ""))
        except Exception:
            return predicted == ground_truth

    # -- aggregate accuracy --------------------------------------------------

    def evaluate_accuracy(self, predictions: list[str], targets: list[str]) -> float:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length")
        if self.task_name == "finer":
            return self._finer_accuracy(predictions, targets)
        if self.task_name == "appworld":
            correct = sum(1 for p in predictions if p == "completed")
            return correct / len(predictions) if predictions else 0.0
        return self._formula_accuracy(predictions, targets)

    def _finer_accuracy(self, preds: list[str], targets: list[str]) -> float:
        correct_count = total_count = 0
        for p, t in zip(preds, targets):
            c, tot = self._finer_correct(p, t, return_counts=True)
            correct_count += c
            total_count += tot
        return correct_count / total_count if total_count else 0.0

    def _formula_accuracy(self, preds: list[str], targets: list[str]) -> float:
        correct = sum(1 for p, t in zip(preds, targets) if self._formula_correct(p, t))
        return correct / len(preds) if preds else 0.0
