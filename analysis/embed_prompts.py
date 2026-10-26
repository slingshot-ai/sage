#!/usr/bin/env python3
"""Collect unique prompts (with accuracy) from optimization runs and embed them.

Scans all run directories under ``results/``, extracts every prompt that appeared in
the top-P pool at any iteration (these are the ones with accuracy persisted to
disk via ``top_i_accX.XXXX.txt``), deduplicates by prompt text, and calls
OpenAI ``text-embedding-3-large`` to produce 3072-d embeddings.

If the same prompt text appears with different reported accuracies (e.g. across
runs or iterations), each distinct accuracy (to 4 decimals, matching filenames)
is counted once and the stored ``accuracy`` is the mean of those values.

Outputs per task:
  - ``landscape/<task>_prompts.json``  — metadata (prompt_id, strategy, run, iteration, accuracy, prompt_text)
  - ``landscape/<task>_embeddings.npy`` — (N, 3072) float32 array aligned with the JSON
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import openai
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "results")
DEFAULT_OUT_DIR = os.path.join(ROOT, "landscape")

STRATEGY_NAMES = {"spo-rs": "SPO-RS", "spo-ga": "SPO-GA", "sage": "SAGE"}

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
MAX_TOKENS = 8191
BATCH_SIZE = 16


def _parse_run_dir(run_dir: str) -> dict | None:
    """Extract strategy and task from a run directory name + config."""
    config_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        config = json.load(f)
    strategy_class = config.get("strategy", "")
    return {
        "task": config.get("task_name", ""),
        "strategy": STRATEGY_NAMES.get(strategy_class, strategy_class),
        "strategy_class": strategy_class,
    }


def _run_index(run_dir: str, all_runs_for_key: list[str]) -> int:
    """Assign a 1-based run index by timestamp order within the same (task, strategy) group."""
    return sorted(all_runs_for_key).index(run_dir) + 1


def collect_prompts(results_dir: str, task: str) -> list[dict]:
    """Collect unique prompts from top_*_acc*.txt; accuracy = mean of distinct values."""
    run_dirs = sorted(d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d))

    runs_by_key: dict[tuple, list[str]] = {}
    run_meta: dict[str, dict] = {}
    for rd in run_dirs:
        meta = _parse_run_dir(rd)
        if meta is None or meta["task"] != task:
            continue
        run_meta[rd] = meta
        key = (meta["task"], meta["strategy"])
        runs_by_key.setdefault(key, []).append(rd)

    seen: dict[str, dict] = {}

    for rd, meta in run_meta.items():
        key = (meta["task"], meta["strategy"])
        run_idx = _run_index(rd, runs_by_key[key])
        top_files = glob.glob(os.path.join(rd, "iterations", "iteration_*", "top_*_acc*.txt"))

        for tf in top_files:
            with open(tf) as f:
                prompt_text = f.read()

            basename = os.path.basename(tf)
            acc_str = basename.split("_acc")[1].replace(".txt", "")
            accuracy = float(acc_str)

            iter_match = re.search(r"iteration_(\d+)", tf)
            iteration = int(iter_match.group(1)) if iter_match else -1
            acc_q = round(accuracy, 4)

            if prompt_text not in seen:
                seen[prompt_text] = {
                    "strategy": meta["strategy"],
                    "run": run_idx,
                    "iteration": iteration,
                    "accuracy_set": {acc_q},
                }
            else:
                seen[prompt_text]["accuracy_set"].add(acc_q)

    prompts = []
    for prompt_text, row in seen.items():
        accs = row["accuracy_set"]
        prompts.append(
            {
                "strategy": row["strategy"],
                "run": row["run"],
                "iteration": row["iteration"],
                "accuracy": sum(accs) / len(accs),
                "prompt_text": prompt_text,
            }
        )
    for i, p in enumerate(prompts):
        p["prompt_id"] = i
    return prompts


def _truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to at most max_tokens for the embedding model."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def embed_prompts(prompts: list[dict], cache_path: str | None = None) -> np.ndarray:
    """Embed prompt texts using OpenAI text-embedding-3-large. Caches to disk."""
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(prompts):
            print(f"  Loaded cached embeddings from {cache_path} ({arr.shape})")
            return arr
        print(f"  Cache shape mismatch ({arr.shape[0]} vs {len(prompts)}), re-embedding")

    client = openai.OpenAI()
    texts = [_truncate_to_tokens(p["prompt_text"]) for p in prompts]
    n_truncated = sum(
        1 for p in prompts if len(tiktoken.get_encoding("cl100k_base").encode(p["prompt_text"])) > MAX_TOKENS
    )
    if n_truncated:
        print(f"  Truncated {n_truncated} prompts to {MAX_TOKENS} tokens")
    embeddings = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        batch = texts[start:end]
        print(f"  Embedding batch {start}–{end - 1} / {len(texts) - 1} ...")

        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                break
            except openai.RateLimitError:
                wait = 2**attempt
                print(f"    Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
        else:
            raise RuntimeError("Exceeded retries for embedding API")

        for item in resp.data:
            embeddings[start + item.index] = item.embedding

    if cache_path:
        np.save(cache_path, embeddings)
        print(f"  Saved embeddings to {cache_path}")

    return embeddings


def main():
    p = argparse.ArgumentParser(description="Collect and embed prompts from optimization runs.")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory of run directories")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for landscape data")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["formula", "finer", "appworld"],
        help="Tasks to process",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for task in args.tasks:
        print(f"\n{'=' * 50}")
        print(f"Task: {task}")
        print(f"{'=' * 50}")

        prompts = collect_prompts(args.results_dir, task)
        print(f"  Unique prompts with accuracy: {len(prompts)}")

        by_strategy = {}
        for p_item in prompts:
            by_strategy.setdefault(p_item["strategy"], []).append(p_item)
        for s, items in sorted(by_strategy.items()):
            print(f"    {s}: {len(items)}")

        json_path = os.path.join(args.out_dir, f"{task}_prompts.json")
        with open(json_path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"  Saved prompt metadata to {json_path}")

        npy_path = os.path.join(args.out_dir, f"{task}_embeddings.npy")
        embeddings = embed_prompts(prompts, cache_path=npy_path)
        print(f"  Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
