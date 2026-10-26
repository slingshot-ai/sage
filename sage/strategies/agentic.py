"""SAGE (agent-guided exploration): an orchestrated multi-agent pipeline run with the Claude Agent SDK.

Per iteration: one analyzer proposes H hypotheses from the top-P prompts' evaluation results, the
orchestrator triages them to I, I investigators verify them in parallel, and Q generators each write a
revised prompt from one investigation report.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, ResultMessage, query

from ..prompts.agents import ANALYZER_PROMPT, GENERATOR_PROMPT, INVESTIGATOR_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from ..prompts.tasks import TASKS
from .base import Strategy

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "agent_scripts"


class SAGE(Strategy):
    """SAGE: each ``propose`` call exports the population's results and runs the agent pipeline on them."""

    name = "sage"

    def __init__(
        self,
        task_name: str,
        agent_model: str = "claude-sonnet-4-6",
        num_hypotheses: int = 8,
        num_investigations: int = 5,
        agent_max_turns: int = 200,
        agent_max_budget_usd: float = 50.0,
        agent_effort: Literal["low", "medium", "high"] = "high",
    ):
        self.task_name = task_name
        self.agent_model = agent_model
        self.num_hypotheses = num_hypotheses
        self.num_investigations = num_investigations
        self.agent_max_turns = agent_max_turns
        self.agent_max_budget_usd = agent_max_budget_usd
        self.agent_effort = agent_effort

    def propose(
        self,
        population: list[dict[str, Any]],
        task_description: str,
        num_proposals: int,
        **kwargs,
    ) -> list[str]:
        iter_dir = kwargs.get("iter_dir")
        task_name = kwargs.get("task_name", self.task_name)

        if not iter_dir:
            raise ValueError("SAGE requires iter_dir")

        iter_path = Path(iter_dir)

        # Write dataset for ALL top-P prompts, one subdirectory each.
        # prompt_0 is the best (population is sorted by accuracy descending).
        dataset_dir = iter_path / "dataset"
        for i, entry in enumerate(population):
            prompt_subdir = dataset_dir / f"prompt_{i}_acc{entry['accuracy']:.4f}"
            self._write_agent_dataset(
                entry["eval_result"],
                entry["prompt"],
                prompt_subdir,
            )
        # Copy analysis scripts into iteration working directory
        scripts_dir = iter_path / "scripts"
        if AGENT_SCRIPTS_DIR.is_dir():
            shutil.copytree(AGENT_SCRIPTS_DIR, scripts_dir, dirs_exist_ok=True)

        scripts_help = self._get_scripts_help(scripts_dir)

        # Run the agent pipeline
        print(f"\n--- Running SAGE agent pipeline (targeting {num_proposals} prompts) ---")
        new_prompts = asyncio.run(
            self._run_agent_pipeline(
                population=population,
                task_name=task_name,
                num_prompts=num_proposals,
                iter_path=iter_path,
                dataset_dir=dataset_dir,
                scripts_dir=scripts_dir,
                scripts_help=scripts_help,
            )
        )

        print(f"  [SAGE] Agent pipeline produced {len(new_prompts)} new prompt candidates")
        return new_prompts

    @staticmethod
    def _write_agent_dataset(
        eval_result: dict[str, Any] | None,
        prompt: str,
        dataset_dir: Path | str,
    ) -> None:
        """Write ``prompt.txt``, ``results.jsonl``, ``errors.jsonl``, ``summary.json`` for SAGE agents."""
        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        (dataset_dir / "prompt.txt").write_text(prompt)

        if eval_result is None:
            return

        detailed = eval_result.get("detailed_results", [])
        detailed_sorted = sorted(detailed, key=lambda r: r["index"])

        with open(dataset_dir / "results.jsonl", "w") as f:
            for r in detailed_sorted:
                row = {
                    "index": r["index"],
                    "question": r["question"],
                    "context": r.get("context", ""),
                    "target": r["target"],
                    "predicted": r["predicted"],
                    "is_correct": r["is_correct"],
                    "response": r.get("response", ""),
                }
                if "task_id" in r:
                    row["task_id"] = r["task_id"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        errors = [r for r in detailed_sorted if not r["is_correct"]]
        with open(dataset_dir / "errors.jsonl", "w") as f:
            for r in errors:
                row = {
                    "index": r["index"],
                    "question": r["question"],
                    "context": r.get("context", ""),
                    "target": r["target"],
                    "predicted": r["predicted"],
                    "is_correct": False,
                    "response": r.get("response", ""),
                }
                if "task_id" in r:
                    row["task_id"] = r["task_id"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "total": eval_result["total"],
            "correct": eval_result["correct"],
            "incorrect": eval_result["total"] - eval_result["correct"],
            "accuracy": eval_result["accuracy"],
        }
        with open(dataset_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _get_scripts_help(scripts_dir: Path) -> str:
        """Collect ``--help`` text from copied agent helper scripts (SAGE iteration)."""
        parts: list[str] = []
        for script in sorted(scripts_dir.glob("*.py")):
            try:
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                help_text = (result.stdout or result.stderr).strip()
                parts.append(f"### {script.name}\n```\n{help_text}\n```")
            except Exception:
                parts.append(f"### {script.name}\n(help unavailable)")
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt_summary(population: list[dict[str, Any]]) -> str:
        """One-line summary per top-P prompt for orchestrator context (SAGE)."""
        lines = []
        for i, entry in enumerate(population):
            label = "★ BEST" if i == 0 else f"  #{i + 1}"
            acc = entry["accuracy"]
            ev = entry.get("eval_result") or {}
            correct = ev.get("correct", "?")
            total = ev.get("total", "?")
            dirname = f"prompt_{i}_acc{acc:.4f}"
            lines.append(f"  {label}  {dirname}  —  accuracy {acc:.1%} ({correct}/{total})")
        return "\n".join(lines)

    def _build_agent_definitions(
        self,
        dataset_dir: Path,
        scripts_dir: Path,
        scripts_help: str,
        task_description: str,
        task_analysis_guidance: str,
    ) -> dict[str, AgentDefinition]:
        """Build analyzer / investigator / generator :class:`AgentDefinition` dict for SAGE."""
        shared_fmt = dict(
            task_description=task_description,
            task_analysis_guidance=task_analysis_guidance,
            dataset_dir=str(dataset_dir),
            scripts_dir=str(scripts_dir),
            scripts_help=scripts_help,
        )
        return {
            "analyzer": AgentDefinition(
                description=(
                    "Explores evaluation results and proposes error-pattern hypotheses. "
                    "Use ONCE at the start of the pipeline (Step 1)."
                ),
                prompt=ANALYZER_PROMPT.format(
                    num_max_hypotheses=self.num_hypotheses,
                    **shared_fmt,
                ),
                tools=["Bash", "Read", "Grep", "Glob", "Write"],
            ),
            "investigator": AgentDefinition(
                description=(
                    "Deep-dives into a specific error pattern with quantitative and "
                    "qualitative analysis. Invoke once per hypothesis in Step 3, "
                    "all instances in parallel."
                ),
                prompt=INVESTIGATOR_PROMPT.format(**shared_fmt),
                tools=["Bash", "Read", "Grep", "Glob", "Write"],
            ),
            "generator": AgentDefinition(
                description=(
                    "Generates a complete updated system prompt for a specific "
                    "improvement direction. Invoke once per selected direction in "
                    "Step 5, all instances in parallel."
                ),
                prompt=GENERATOR_PROMPT.format(**shared_fmt),
                tools=["Read", "Write", "Grep", "Glob"],
            ),
        }

    async def _run_agent_pipeline(
        self,
        population: list[dict[str, Any]],
        task_name: str,
        num_prompts: int,
        iter_path: Path,
        dataset_dir: Path,
        scripts_dir: Path,
        scripts_help: str,
    ) -> list[str]:
        """Execute the Claude Agent SDK query and return parsed new prompts (SAGE)."""
        task_description = TASKS[task_name].AGENT_DESCRIPTION
        task_analysis_guidance = TASKS[task_name].ANALYSIS_GUIDANCE

        agent_outputs_dir = iter_path / "agent_outputs"
        for subdir in ["analyzer", "investigators", "generators"]:
            (agent_outputs_dir / subdir).mkdir(parents=True, exist_ok=True)

        agents = self._build_agent_definitions(
            dataset_dir=dataset_dir.resolve(),
            scripts_dir=scripts_dir.resolve(),
            scripts_help=scripts_help,
            task_description=task_description,
            task_analysis_guidance=task_analysis_guidance,
        )

        best = population[0]
        prompt_summary = self._build_prompt_summary(population)

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            num_prompts=num_prompts,
            num_hypotheses_to_investigate=self.num_investigations,
            num_max_hypotheses=self.num_hypotheses,
            task_description=task_description,
            dataset_dir=str(dataset_dir.resolve()),
            scripts_dir=str(scripts_dir.resolve()),
            agent_outputs_dir=str(agent_outputs_dir.resolve()),
            current_accuracy=best["accuracy"],
            num_correct=best["eval_result"]["correct"],
            num_total=best["eval_result"]["total"],
            num_top_prompts=len(population),
            prompt_summary=prompt_summary,
        )

        user_prompt = (
            f"The dataset at `{dataset_dir.resolve()}` contains evaluation results for "
            f"{len(population)} prompt(s):\n{prompt_summary}\n\n"
            f"Analyze the errors across ALL prompts, compare their strengths and weaknesses, "
            f"and propose exactly {num_prompts} prompt improvements. For each, select the "
            f"best parent prompt to start from. Begin with Step 1: invoke the analyzer."
        )

        # Save agent prompts for debugging/reproducibility
        prompts_save_dir = iter_path / "saved_prompts"
        prompts_save_dir.mkdir(exist_ok=True)
        (prompts_save_dir / "orchestrator.md").write_text(system_prompt)
        (prompts_save_dir / "user.md").write_text(user_prompt)
        for name, defn in agents.items():
            (prompts_save_dir / f"{name}.md").write_text(defn.prompt)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.agent_model,
            max_turns=self.agent_max_turns,
            max_budget_usd=self.agent_max_budget_usd,
            effort=self.agent_effort,
            permission_mode="bypassPermissions",
            allowed_tools=["Agent", "Read", "Grep", "Glob"],
            agents=agents,
            cwd=str(iter_path),
        )

        async def prompt_stream():
            yield {"type": "user", "message": {"role": "user", "content": user_prompt}}

        log_file = iter_path / "agent_logs.jsonl"
        result_message: ResultMessage | None = None

        with open(log_file, "w") as log_fh:
            async for message in query(prompt=prompt_stream(), options=options):
                _log_message(message, log_fh)
                _print_message(message)
                if isinstance(message, ResultMessage) and result_message is None:
                    result_message = message

        if result_message is None or result_message.is_error:
            print(f"  [SAGE] Agent pipeline failed: {result_message}")
            return []

        metadata = {
            "total_cost_usd": result_message.total_cost_usd,
            "duration_ms": result_message.duration_ms,
            "num_turns": result_message.num_turns,
        }
        (iter_path / "agent_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        return self._parse_generator_outputs(agent_outputs_dir / "generators")

    @staticmethod
    def _parse_generator_outputs(generators_dir: Path) -> list[str]:
        """Read ``prompt.txt`` / ``prompt.md`` from each generator output subdirectory."""
        if not generators_dir.exists():
            print(f"  [SAGE] Generators directory not found: {generators_dir}")
            return []

        prompts: list[str] = []
        for path in sorted(p for p in generators_dir.iterdir() if p.is_dir()):
            prompt_file = path / "prompt.txt"
            if not prompt_file.exists():
                prompt_file = path / "prompt.md"
            if prompt_file.exists():
                text = prompt_file.read_text().strip()
                if text:
                    prompts.append(text)
                    name_file = path / "name.txt"
                    name = name_file.read_text().strip() if name_file.exists() else path.name
                    print(f"  [SAGE] Parsed prompt from generator {path.name}: {name}")
                else:
                    print(f"  [SAGE] Empty prompt in {path.name}, skipping")
            else:
                print(f"  [SAGE] No prompt file in {path.name}, skipping")
        return prompts


def _log_message(message, log_fh) -> None:
    """Append a JSON line to the SAGE agent transcript log."""
    try:
        data = {"type": type(message).__name__, "timestamp": datetime.now().isoformat()}
        if hasattr(message, "content"):
            data["content"] = str(message.content)[:2000]
        if hasattr(message, "agent_name"):
            data["agent_name"] = message.agent_name
        log_fh.write(json.dumps(data, default=str) + "\n")
        log_fh.flush()
    except Exception:
        pass


def _print_message(message) -> None:
    """Print a concise line for agent/orchestrator messages during the SAGE pipeline."""
    try:
        agent = getattr(message, "agent_name", None)
        prefix = f"  [SAGE/{agent}] " if agent else "  [SAGE/orchestrator] "
        msg_type = type(message).__name__

        if msg_type == "ResultMessage":
            cost = getattr(message, "total_cost_usd", "?")
            turns = getattr(message, "num_turns", "?")
            print(f"{prefix}Pipeline complete (cost=${cost}, turns={turns})")
        elif hasattr(message, "content"):
            content = str(message.content)
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{prefix}{msg_type}: {content}")
    except Exception:
        pass
