"""The SPO outer loop: stochastic hill climbing over prompts, shared by SPO-RS, SPO-GA and SAGE.

Each iteration evaluates new candidates on the train set, keeps the top P, and asks the strategy for Q
new candidates. The best prompt of each iteration is scored on the validation set; the prompt with the
best validation accuracy is scored once on the test set.
"""

from __future__ import annotations

import glob as globmod
import json
import os
from datetime import datetime
from typing import Any

import openai

from .data import DataProcessor
from .evaluate import evaluate_prompt
from .prompts import TASKS
from .strategies.base import Strategy
from .utils import count_tokens

_APPWORLD_RUNNER_LOADED = False
try:
    from .appworld_runner import evaluate_appworld_playbook

    _APPWORLD_RUNNER_LOADED = True
except ImportError:
    pass


class Optimizer:
    """Runs SPO with a given proposal strategy."""

    def __init__(
        self,
        client: openai.OpenAI,
        task_model: str,
        strategy: Strategy,
        task_name: str,
        strategy_model: str | None = None,
        max_tokens: int = 4096,
        max_workers: int = 20,
        appworld_config: dict[str, Any] | None = None,
    ):
        self.client = client
        self.task_model = task_model
        self.strategy = strategy
        self.strategy_model = strategy_model
        self.task_name = task_name
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.data_processor = DataProcessor(task_name)
        self.appworld_config = appworld_config or {}

    def _evaluate(
        self,
        prompt: str,
        samples: list[dict[str, Any]],
        label: str = "eval",
    ) -> dict[str, Any]:
        """Dispatch evaluation to the correct backend based on task type."""
        if self.task_name == "appworld":
            if not _APPWORLD_RUNNER_LOADED:
                raise ImportError("AppWorld runner not available. Install appworld package.")
            task_ids = [s["task_id"] for s in samples]
            return evaluate_appworld_playbook(
                client=self.client,
                model=self.task_model,
                playbook=prompt,
                task_ids=task_ids,
                max_steps=self.appworld_config.get("max_steps", 40),
                max_tokens=self.max_tokens,
                temperature=0.0,
                max_workers=self.max_workers,
                label=label,
                experiment_name=self.appworld_config.get("experiment_name", "sage_eval"),
                random_seed=self.appworld_config.get("random_seed", 123),
            )
        return evaluate_prompt(
            self.client,
            self.task_model,
            prompt,
            samples,
            self.data_processor,
            self.max_tokens,
            self.max_workers,
            label=label,
        )

    def optimize(
        self,
        train_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]] | None = None,
        val_samples: list[dict[str, Any]] | None = None,
        initial_prompt: str | None = None,
        num_iterations: int = 5,
        top_n: int = 3,
        num_proposals: int = 3,
        save_dir: str = "./results",
        resume_from: str | None = None,
    ) -> dict[str, Any]:
        """Run SPO for ``num_iterations`` iterations (a total, also when resuming) and return the results.

        ``top_n`` is the population size P and ``num_proposals`` the number of candidates Q. With
        ``resume_from``, the run continues in that directory from its last completed iteration.
        """

        prompt = initial_prompt or TASKS[self.task_name].SEED_PROMPT
        task_desc = TASKS[self.task_name].DESCRIPTION
        strategy_name = self.strategy.name

        # --- Setup run directory and initial state ---
        start_iteration = 1
        iteration_log: list[dict[str, Any]] = []
        population: list[dict[str, Any]] = [{"prompt": prompt, "accuracy": None, "eval_result": None}]

        if resume_from:
            run_dir = resume_from
            state = self._load_resume_state(run_dir)
            start_iteration = state["start_iteration"]
            iteration_log = state["iteration_log"]
            population = state["population"]
            best_acc = max(
                (e["accuracy"] for e in population if e["accuracy"] is not None),
                default=0.0,
            )
            print(f"\nResuming from {run_dir}")
            print(f"  Completed iterations: {start_iteration - 1}")
            print(f"  Population size: {len(population)}")
            print(f"  Best accuracy so far: {best_acc:.3f}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(save_dir, f"{strategy_name}_{self.task_name}_{timestamp}")

        os.makedirs(run_dir, exist_ok=True)

        self._save_run_config(
            run_dir,
            num_iterations,
            top_n,
            num_proposals,
            strategy_name,
            resume_from,
        )

        self._print_header(
            "OPTIMIZE",
            train_samples,
            test_samples,
            num_iterations,
            top_n,
            num_proposals,
            strategy_name,
            val_samples=val_samples,
        )

        results: dict[str, Any] = {}
        if resume_from:
            final_path = os.path.join(run_dir, "final_results.json")
            if os.path.exists(final_path):
                with open(final_path, "r") as f:
                    results = json.load(f)

        # --- Initial test (baseline — skip if already present) ---
        if test_samples and "initial_test" not in results:
            print(f"\n{'=' * 60}")
            print("INITIAL TEST (baseline before optimisation)")
            print(f"{'=' * 60}\n")
            initial_test = self._evaluate(prompt, test_samples, label="initial_test")
            results["initial_test"] = self._summarise(initial_test)
            print(f"  Initial test accuracy: {initial_test['accuracy']:.3f}\n")

        # Back-fill validation for completed iterations missing val results
        if val_samples and resume_from:
            self._backfill_validation(run_dir, iteration_log, val_samples)
            if iteration_log:
                best_train = max(e.get("best_accuracy", 0) for e in iteration_log)
                with open(os.path.join(run_dir, "train_results.json"), "w") as f:
                    json.dump(
                        {
                            "best_accuracy": best_train,
                            "iterations": iteration_log,
                        },
                        f,
                        indent=2,
                    )

        if start_iteration > num_iterations:
            print(f"\n  Already completed {start_iteration - 1} iterations (target: {num_iterations}). Nothing to do.")

        _cached_val_prompt: str | None = None
        _cached_val_summary: dict | None = None

        for iteration in range(start_iteration, num_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration}/{num_iterations}")
            print(f"{'=' * 60}")

            iter_dir = os.path.join(run_dir, "iterations", f"iteration_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            # 1. Evaluate unevaluated candidates
            for entry in population:
                if entry["accuracy"] is None:
                    ev = self._evaluate(
                        entry["prompt"],
                        train_samples,
                        label=f"iter{iteration}_train",
                    )
                    entry["accuracy"] = ev["accuracy"]
                    entry["eval_result"] = ev

            # 2. Sort and select top-P
            population.sort(key=lambda e: e["accuracy"], reverse=True)
            top = population[:top_n]

            best = top[0]
            print(f"\n  Top-{min(top_n, len(top))} accuracies: {[round(e['accuracy'], 4) for e in top]}")
            print(f"  Best prompt tokens: {count_tokens(best['prompt'])}")

            # Save iteration state (prompts + eval results for top entries)
            self._save_iteration(iter_dir, iteration, top)

            # Evaluate best prompt on validation set (cached if prompt unchanged)
            val_acc = None
            if val_samples:
                if best["prompt"] == _cached_val_prompt:
                    val_acc = _cached_val_summary["accuracy"]
                    with open(os.path.join(iter_dir, "val_result.json"), "w") as f:
                        json.dump(_cached_val_summary, f, indent=2)
                    print(f"  Validation accuracy: {val_acc:.3f} (cached)")
                else:
                    print(f"\n--- Validating best prompt (iteration {iteration}) ---")
                    val_eval = self._evaluate(
                        best["prompt"],
                        val_samples,
                        label=f"iter{iteration}_val",
                    )
                    val_acc = val_eval["accuracy"]
                    _cached_val_summary = self._summarise(val_eval)
                    _cached_val_prompt = best["prompt"]
                    with open(os.path.join(iter_dir, "val_result.json"), "w") as f:
                        json.dump(_cached_val_summary, f, indent=2)
                    print(f"  Validation accuracy: {val_acc:.3f}")

            # 3. Strategy proposes Q new prompts
            print(f"\n--- Proposing {num_proposals} new prompts via {strategy_name} ---")
            new_prompt_texts = self.strategy.propose(
                top,
                task_desc,
                num_proposals,
                iter_dir=iter_dir,
                task_name=self.task_name,
            )
            print(f"  Received {len(new_prompt_texts)} new prompts")

            for i, p in enumerate(new_prompt_texts):
                with open(os.path.join(iter_dir, f"proposed_{i}.txt"), "w") as f:
                    f.write(p)

            # 4. Build next population = top-P (evaluated) + Q new (unevaluated)
            population = list(top) + [{"prompt": p, "accuracy": None, "eval_result": None} for p in new_prompt_texts]

            iter_result = {
                "iteration": iteration,
                "best_accuracy": best["accuracy"],
                "top_accuracies": [round(e["accuracy"], 4) for e in top],
                "num_proposed": len(new_prompt_texts),
                "best_prompt_tokens": count_tokens(best["prompt"]),
            }
            if val_acc is not None:
                iter_result["val_accuracy"] = val_acc
            iteration_log.append(iter_result)
            with open(os.path.join(iter_dir, "iteration_result.json"), "w") as f:
                json.dump(iter_result, f, indent=2)

            # Checkpoint best prompt + train results after each iteration
            with open(os.path.join(run_dir, "best_prompt.txt"), "w") as f:
                f.write(best["prompt"])
            with open(os.path.join(run_dir, "train_results.json"), "w") as f:
                json.dump(
                    {
                        "best_accuracy": best["accuracy"],
                        "iterations": iteration_log,
                    },
                    f,
                    indent=2,
                )

        ran_new_iterations = start_iteration <= num_iterations

        # --- Evaluate final unevaluated candidates from last proposal ---
        if ran_new_iterations:
            unevaluated = [e for e in population if e["accuracy"] is None]
            if unevaluated:
                print(f"\n{'=' * 60}")
                print("FINAL EVALUATION OF REMAINING CANDIDATES")
                print(f"{'=' * 60}")
                for entry in unevaluated:
                    ev = self._evaluate(
                        entry["prompt"],
                        train_samples,
                        label="final_train",
                    )
                    entry["accuracy"] = ev["accuracy"]
                    entry["eval_result"] = ev

        evaluated = [e for e in population if e["accuracy"] is not None]
        evaluated.sort(key=lambda e: e["accuracy"], reverse=True)
        best_by_train = evaluated[0]
        print(f"\n  Overall best train accuracy: {best_by_train['accuracy']:.3f}")

        results["iteration_log"] = iteration_log
        results["best_train_accuracy"] = best_by_train["accuracy"]

        # --- Select final best prompt: by val accuracy when available ---
        val_entries = [e for e in iteration_log if "val_accuracy" in e]
        if val_entries:
            best_val_entry = max(val_entries, key=lambda e: e["val_accuracy"])
            best_val_iter = best_val_entry["iteration"]
            iter_dir = os.path.join(run_dir, "iterations", f"iteration_{best_val_iter}")
            best_prompt = self._load_best_prompt_for_iteration(iter_dir)
            print(f"  Best val accuracy:   {best_val_entry['val_accuracy']:.3f} (iteration {best_val_iter})")

            prev_val_iter = results.get("best_val_iteration")
            results["best_val_accuracy"] = best_val_entry["val_accuracy"]
            results["best_val_iteration"] = best_val_iter
            results["best_prompt_tokens"] = count_tokens(best_prompt)
            needs_final_test = "final_test" not in results or prev_val_iter != best_val_iter
        else:
            best_prompt = best_by_train["prompt"]
            results["best_prompt_tokens"] = count_tokens(best_prompt)
            needs_final_test = ran_new_iterations or "final_test" not in results

        with open(os.path.join(run_dir, "best_prompt.txt"), "w") as f:
            f.write(best_prompt)

        # --- Final test ---
        if test_samples and needs_final_test:
            print(f"\n{'=' * 60}")
            print("FINAL TEST (with best prompt)")
            print(f"{'=' * 60}\n")
            final_test = self._evaluate(best_prompt, test_samples, label="final_test")
            results["final_test"] = self._summarise(final_test)
            print(f"  Final test accuracy: {final_test['accuracy']:.3f}\n")

        # Save consolidated results
        with open(os.path.join(run_dir, "train_results.json"), "w") as f:
            json.dump(
                {
                    "best_accuracy": best_by_train["accuracy"],
                    "iterations": iteration_log,
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        self._print_summary(results, run_dir)
        return results

    def eval_only(
        self,
        test_samples: list[dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any]:
        """Evaluate a single prompt on the test set (no optimisation)."""
        self._print_header("EVAL_ONLY", None, test_samples)
        result = self._evaluate(prompt, test_samples, label="test")
        print(f"\n  Test accuracy: {result['accuracy']:.3f}")
        return self._summarise(result)

    def _load_resume_state(self, run_dir: str) -> dict[str, Any]:
        """Load state from a previous run directory for resumption.

        Returns a dict with keys: start_iteration, iteration_log, population.
        """
        train_results_path = os.path.join(run_dir, "train_results.json")
        if os.path.exists(train_results_path):
            with open(train_results_path, "r") as f:
                train_results = json.load(f)
            iteration_log = train_results.get("iterations", [])
        else:
            iteration_log = self._reconstruct_iterations_from_disk(run_dir)

        completed = len(iteration_log)
        if completed == 0:
            best_prompt_path = os.path.join(run_dir, "best_prompt.txt")
            prompt = ""
            if os.path.exists(best_prompt_path):
                with open(best_prompt_path, "r") as f:
                    prompt = f.read().strip()
            return {
                "start_iteration": 1,
                "iteration_log": [],
                "population": [{"prompt": prompt, "accuracy": None, "eval_result": None}],
            }

        population = self._load_population_from_iteration(run_dir, completed)

        return {
            "start_iteration": completed + 1,
            "iteration_log": iteration_log,
            "population": population,
        }

    def _reconstruct_iterations_from_disk(self, run_dir: str) -> list[dict[str, Any]]:
        """Rebuild iteration log from iteration_*/iteration_result.json when
        train_results.json is missing (e.g. mid-iteration crash)."""
        iterations_dir = os.path.join(run_dir, "iterations")
        if not os.path.isdir(iterations_dir):
            return []
        results: list[dict[str, Any]] = []
        n = 1
        while True:
            path = os.path.join(iterations_dir, f"iteration_{n}", "iteration_result.json")
            if not os.path.exists(path):
                break
            with open(path, "r") as f:
                results.append(json.load(f))
            n += 1
        return results

    def _load_population_from_iteration(
        self,
        run_dir: str,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Rebuild population from an iteration directory's saved files.

        Returns top entries (with accuracy and eval_result when available)
        plus proposed entries (unevaluated).
        """
        iter_dir = os.path.join(run_dir, "iterations", f"iteration_{iteration}")
        population: list[dict[str, Any]] = []

        # Load evaluated top prompts
        i = 0
        while True:
            matches = globmod.glob(os.path.join(iter_dir, f"top_{i}_acc*.txt"))
            if not matches:
                break
            prompt_path = matches[0]
            basename = os.path.basename(prompt_path)
            acc_str = basename.split("_acc")[1].replace(".txt", "")
            accuracy = float(acc_str)
            with open(prompt_path, "r") as f:
                prompt_text = f.read()

            result_path = os.path.join(iter_dir, f"top_{i}_result.json")
            eval_result = None
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    eval_result = json.load(f)

            population.append(
                {
                    "prompt": prompt_text,
                    "accuracy": accuracy,
                    "eval_result": eval_result,
                }
            )
            i += 1

        # Load unevaluated proposed prompts
        j = 0
        while True:
            proposed_path = os.path.join(iter_dir, f"proposed_{j}.txt")
            if not os.path.exists(proposed_path):
                break
            with open(proposed_path, "r") as f:
                prompt_text = f.read()
            population.append(
                {
                    "prompt": prompt_text,
                    "accuracy": None,
                    "eval_result": None,
                }
            )
            j += 1

        if not population:
            best_path = os.path.join(run_dir, "best_prompt.txt")
            if os.path.exists(best_path):
                with open(best_path, "r") as f:
                    population.append(
                        {
                            "prompt": f.read().strip(),
                            "accuracy": None,
                            "eval_result": None,
                        }
                    )

        return population

    def _backfill_validation(
        self,
        run_dir: str,
        iteration_log: list[dict[str, Any]],
        val_samples: list[dict[str, Any]],
    ) -> None:
        """Evaluate validation set for completed iterations missing val results.

        Updates *iteration_log* entries in-place and writes per-iteration files.
        """
        needs_backfill = any(
            not os.path.exists(
                os.path.join(run_dir, "iterations", f"iteration_{entry['iteration']}", "val_result.json")
            )
            for entry in iteration_log
        )
        if not needs_backfill:
            for entry in iteration_log:
                if "val_accuracy" not in entry:
                    iter_dir = os.path.join(run_dir, "iterations", f"iteration_{entry['iteration']}")
                    val_path = os.path.join(iter_dir, "val_result.json")
                    if os.path.exists(val_path):
                        with open(val_path, "r") as f:
                            entry["val_accuracy"] = json.load(f)["accuracy"]
            return

        print(f"\n{'=' * 60}")
        print("BACK-FILLING VALIDATION RESULTS")
        print(f"{'=' * 60}")

        val_cache: dict[str, dict] = {}

        for entry in iteration_log:
            iteration = entry["iteration"]
            iter_dir = os.path.join(run_dir, "iterations", f"iteration_{iteration}")
            val_result_path = os.path.join(iter_dir, "val_result.json")

            if os.path.exists(val_result_path):
                with open(val_result_path, "r") as f:
                    val_data = json.load(f)
                entry["val_accuracy"] = val_data["accuracy"]
                best_prompt = self._load_best_prompt_for_iteration(iter_dir)
                if best_prompt:
                    val_cache[best_prompt] = val_data
                print(f"  Iteration {iteration}: val result already exists (accuracy={entry['val_accuracy']:.3f})")
                continue

            best_prompt = self._load_best_prompt_for_iteration(iter_dir)
            if not best_prompt:
                print(f"  Iteration {iteration}: WARNING — could not find best prompt, skipping")
                continue

            if best_prompt in val_cache:
                cached = val_cache[best_prompt]
                entry["val_accuracy"] = cached["accuracy"]
                with open(val_result_path, "w") as f:
                    json.dump(cached, f, indent=2)
                print(f"  Iteration {iteration}: reusing cached val result (accuracy={cached['accuracy']:.3f})")
            else:
                print(f"\n--- Validating best prompt (iteration {iteration}) ---")
                val_eval = self._evaluate(
                    best_prompt,
                    val_samples,
                    label=f"iter{iteration}_val",
                )
                val_summary = self._summarise(val_eval)
                val_cache[best_prompt] = val_summary
                entry["val_accuracy"] = val_eval["accuracy"]
                with open(val_result_path, "w") as f:
                    json.dump(val_summary, f, indent=2)
                print(f"  Validation accuracy: {val_eval['accuracy']:.3f}")

            iter_result_path = os.path.join(iter_dir, "iteration_result.json")
            if os.path.exists(iter_result_path):
                with open(iter_result_path, "r") as f:
                    iter_result = json.load(f)
                iter_result["val_accuracy"] = entry["val_accuracy"]
                with open(iter_result_path, "w") as f:
                    json.dump(iter_result, f, indent=2)

    @staticmethod
    def _load_best_prompt_for_iteration(iter_dir: str) -> str | None:
        """Load the best (top_0) prompt text from an iteration directory."""
        matches = globmod.glob(os.path.join(iter_dir, "top_0_acc*.txt"))
        if not matches:
            return None
        with open(matches[0], "r") as f:
            return f.read()

    @staticmethod
    def _summarise(eval_result: dict) -> dict:
        return {
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
        }

    def _save_run_config(
        self,
        run_dir,
        iters,
        top_n,
        num_proposals,
        strategy,
        resume_from=None,
    ):
        config = {
            "task_name": self.task_name,
            "task_model": self.task_model,
            "strategy": strategy,
            "num_iterations": iters,
            "top_n": top_n,
            "num_proposals": num_proposals,
            "max_tokens": self.max_tokens,
            "max_workers": self.max_workers,
        }
        if self.strategy_model:
            config["strategy_model"] = self.strategy_model
        if resume_from:
            config["resumed_from"] = resume_from
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _save_iteration(self, iter_dir, iteration, top):
        for i, entry in enumerate(top):
            with open(os.path.join(iter_dir, f"top_{i}_acc{entry['accuracy']:.4f}.txt"), "w") as f:
                f.write(entry["prompt"])
            if entry.get("eval_result"):
                with open(os.path.join(iter_dir, f"top_{i}_result.json"), "w") as f:
                    json.dump(entry["eval_result"], f, indent=2)

        summary = {
            "iteration": iteration,
            "top_accuracies": [round(e["accuracy"], 4) for e in top],
        }
        with open(os.path.join(iter_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def _print_header(
        self, mode, train, test, iters=None, top_n=None, num_proposals=None, strategy=None, val_samples=None
    ):
        print(f"\n{'=' * 60}")
        print(f"SPO — {mode}")
        print(f"{'=' * 60}")
        print(f"  Task:       {self.task_name}")
        print(f"  Model:      {self.task_model}")
        if self.strategy_model:
            print(f"  Strat model:{self.strategy_model}")
        if strategy:
            print(f"  Strategy:   {strategy}")
        if train:
            print(f"  Train:      {len(train)} samples")
        if val_samples:
            print(f"  Val:        {len(val_samples)} samples")
        if test:
            print(f"  Test:       {len(test)} samples")
        if iters:
            print(f"  Iterations: {iters}")
            print(f"  Top-N:      {top_n}")
            print(f"  Proposals:  {num_proposals}")
        print(f"{'=' * 60}")

    @staticmethod
    def _print_summary(results, run_dir):
        print(f"\n{'=' * 60}")
        print("RUN COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Best train accuracy:   {results.get('best_train_accuracy', 'N/A')}")
        if "best_val_accuracy" in results:
            print(
                f"  Best val accuracy:     {results['best_val_accuracy']:.3f} "
                f"(iteration {results.get('best_val_iteration', '?')})"
            )
        if "initial_test" in results:
            print(f"  Initial test accuracy: {results['initial_test']['accuracy']:.3f}")
        if "final_test" in results:
            sel = "val-best" if "best_val_accuracy" in results else "train-best"
            print(f"  Final test accuracy:   {results['final_test']['accuracy']:.3f} ({sel} prompt)")
        print(f"  Results saved to: {run_dir}")
        print(f"{'=' * 60}\n")
