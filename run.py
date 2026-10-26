#!/usr/bin/env python3
"""Run SPO (SPO-RS, SPO-GA or SAGE) on a benchmark task, or evaluate a prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sage.data import DataProcessor, load_jsonl
from sage.llm import create_client
from sage.optimizer import Optimizer
from sage.prompts import TASKS
from sage.strategies import STRATEGIES

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--strategy", default="sage", choices=list(STRATEGIES))
    p.add_argument("--mode", default="optimize", choices=["optimize", "eval_only"])

    g = p.add_argument_group("models")
    g.add_argument("--model", default="deepseek/deepseek-chat-v3-0324", help="target model (OpenRouter)")
    g.add_argument(
        "--meta-model", default="anthropic/claude-sonnet-4-6", help="SPO-RS / SPO-GA meta-model (OpenRouter)"
    )
    g.add_argument("--agent-model", default="claude-sonnet-4-6", help="SAGE agent model (Claude Agent SDK)")

    g = p.add_argument_group("search")
    g.add_argument("-K", "--iterations", type=int, default=10, help="total iterations")
    g.add_argument("-P", "--population", type=int, default=2, help="prompts kept per iteration")
    g.add_argument("-Q", "--candidates", type=int, default=3, help="new candidates per iteration")
    g.add_argument("--temperature", type=float, help="meta-model temperature (default 0.9 SPO-RS, 0.7 SPO-GA)")
    g.add_argument("--crossover-ratio", type=float, default=0.3, help="SPO-GA share of crossover candidates")
    g.add_argument("--hypotheses", type=int, default=8, help="SAGE: hypotheses from the analyzer (H)")
    g.add_argument("--investigations", type=int, default=5, help="SAGE: hypotheses investigated (I)")
    g.add_argument("--agent-max-turns", type=int, default=200, help="SAGE: agent turns per iteration")
    g.add_argument("--agent-max-budget", type=float, default=50.0, help="SAGE: USD budget per iteration")
    g.add_argument("--agent-effort", default="high", choices=["low", "medium", "high"])

    g = p.add_argument_group("evaluation")
    g.add_argument("--max-workers", type=int, default=20)
    g.add_argument("--max-tokens", type=int, default=4096, help="target model output limit")
    g.add_argument("--train-limit", type=int, help="use only the first N train samples")
    g.add_argument("--no-val", action="store_true", help="skip per-iteration validation")
    g.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    g.add_argument("--appworld-max-steps", type=int, default=40)
    g.add_argument("--appworld-test-split", default="test_normal", choices=["test_normal", "test_challenge"])

    g = p.add_argument_group("input / output")
    g.add_argument("--prompt-path", help="initial prompt (default: the task's seed prompt)")
    g.add_argument("--config-path", default=str(ROOT / "config.json"), help="data paths for finer / formula")
    g.add_argument("--save-dir", help="output directory (default: ./results)")
    g.add_argument("--resume-from", help="run directory to continue")

    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = str(Path(args.resume_from).resolve().parent) if args.resume_from else "./results"
    return args


def build_strategy(args: argparse.Namespace, client):
    if args.strategy == "sage":
        strategy = STRATEGIES["sage"](
            task_name=args.task,
            agent_model=args.agent_model,
            num_hypotheses=args.hypotheses,
            num_investigations=args.investigations,
            agent_max_turns=args.agent_max_turns,
            agent_max_budget_usd=args.agent_max_budget,
            agent_effort=args.agent_effort,
        )
        return strategy, args.agent_model
    kwargs = {"client": client, "meta_model": args.meta_model}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.strategy == "spo-ga":
        kwargs["crossover_ratio"] = args.crossover_ratio
    return STRATEGIES[args.strategy](**kwargs), args.meta_model


def load_samples(args: argparse.Namespace) -> dict[str, list[dict] | None]:
    """Train / val / test samples for the task (``None`` where a split is skipped)."""
    if args.task == "appworld":
        from sage.appworld_runner import appworld_task_ids_to_samples, load_appworld_task_ids

        def load(split):
            return appworld_task_ids_to_samples(load_appworld_task_ids(split))

        splits = {"train": "train", "val": "dev", "test": args.appworld_test_split}
    else:
        config_path = Path(args.config_path)
        paths = json.loads(config_path.read_text())[args.task]
        processor = DataProcessor(args.task)

        def load(split):
            return processor.process(load_jsonl(str(config_path.parent / paths[f"{split}_data"])))

        splits = {"train": "train", "val": "val", "test": "test"}

    samples = {
        "train": load(splits["train"]) if args.mode == "optimize" else None,
        "val": None if args.no_val or args.mode != "optimize" else load(splits["val"]),
        "test": None if args.no_test else load(splits["test"]),
    }
    if samples["train"] and args.train_limit:
        samples["train"] = samples["train"][: args.train_limit]
    return samples


def main() -> None:
    args = parse_args()
    client = create_client()
    initial_prompt = Path(args.prompt_path).read_text().strip() if args.prompt_path else TASKS[args.task].SEED_PROMPT

    resume_from = None
    if args.resume_from:
        resume_from = str(Path(args.resume_from).resolve())
        config = json.loads((Path(resume_from) / "run_config.json").read_text())
        if config.get("task_name") != args.task:
            sys.exit(f"--task {args.task!r} does not match the resumed run ({config.get('task_name')!r})")

    strategy, strategy_model = build_strategy(args, client)
    optimizer = Optimizer(
        client=client,
        task_model=args.model,
        strategy=strategy,
        strategy_model=strategy_model,
        task_name=args.task,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        appworld_config={"max_steps": args.appworld_max_steps, "experiment_name": "sage", "random_seed": 123},
    )
    samples = load_samples(args)

    if args.mode == "eval_only":
        optimizer.eval_only(samples["test"], initial_prompt)
        return
    optimizer.optimize(
        train_samples=samples["train"],
        val_samples=samples["val"],
        test_samples=samples["test"],
        initial_prompt=initial_prompt,
        num_iterations=args.iterations,
        top_n=args.population,
        num_proposals=args.candidates,
        save_dir=args.save_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
