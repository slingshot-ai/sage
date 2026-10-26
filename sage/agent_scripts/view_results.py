#!/usr/bin/env python3
"""
View and filter evaluation results from a SAGE iteration.

The dataset directory contains one subdirectory per evaluated prompt:
  dataset/prompt_0_acc0.8500/  (best)
  dataset/prompt_1_acc0.8200/
  ...

Usage:
    python view_results.py                           # Overview of all prompts
    python view_results.py --prompt 0 --incorrect    # Errors from prompt 0
    python view_results.py --prompt 1 --n 5          # First 5 results from prompt 1
    python view_results.py --prompt 0 --index 42     # Sample 42 from prompt 0
    python view_results.py --prompt 0 --show-response  # Include full model response
    python view_results.py --prompt 0 --show-transcript  # AppWorld: show full ReAct transcripts
    python view_results.py --prompt 0 --task-id abc123   # AppWorld: filter by task ID
"""

import argparse
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"


def discover_prompts(dataset_dir):
    """Find all prompt_* subdirectories, sorted by index."""
    dirs = sorted(
        (d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")),
        key=lambda d: d.name,
    )
    return dirs


def prompt_summary(prompt_dirs):
    """Print overview of all prompts with accuracy."""
    print("## Evaluated Prompts\n")
    for d in prompt_dirs:
        summary_file = d / "summary.json"
        if summary_file.exists():
            s = json.loads(summary_file.read_text())
            tag = "★ BEST" if d.name.startswith("prompt_0_") else "       "
            print(
                f"  {tag}  {d.name}  —  accuracy {s['accuracy']:.1%} "
                f"({s['correct']}/{s['total']}, {s['incorrect']} errors)"
            )
        else:
            print(f"          {d.name}  —  (no summary)")
    print()


def resolve_prompt_dir(dataset_dir, prompt_dirs, selector):
    """Resolve --prompt value to a directory. Accepts index (0,1,...) or dirname."""
    try:
        idx = int(selector)
        if 0 <= idx < len(prompt_dirs):
            return prompt_dirs[idx]
        print(f"Error: prompt index {idx} out of range (0-{len(prompt_dirs) - 1})")
        return None
    except ValueError:
        pass
    candidate = dataset_dir / selector
    if candidate.is_dir():
        return candidate
    for d in prompt_dirs:
        if selector in d.name:
            return d
    print(f"Error: no prompt matching '{selector}'")
    return None


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def detect_task(results):
    if not results:
        return "unknown"
    sample = results[0]
    q = sample.get("question", "")
    t = sample.get("target", "")
    if t == "task_completed" or sample.get("task_id"):
        return "appworld"
    if "Output US GAAP tags" in q or "US GAAP" in q:
        return "finer"
    try:
        float(t)
        return "formula"
    except (ValueError, TypeError):
        pass
    return "generic"


def extract_finer_questions(question_text):
    match = re.search(r"Answer the following \d+ independent questions.*?\n(.*)", question_text, re.DOTALL)
    if match:
        remainder = match.group(1).strip()
        parts = re.split(r"\n(?=\d+\.)", remainder)
        return [p.strip() for p in parts if p.strip()]
    return [question_text[-1000:].strip()]


def format_finer_comparison(predicted, target):
    pred_tags = [t.strip() for t in predicted.split(",")]
    tgt_tags = [t.strip() for t in target.split(",")]
    n = max(len(pred_tags), len(tgt_tags))
    lines = []
    for i in range(n):
        p = pred_tags[i] if i < len(pred_tags) else "(missing)"
        t = tgt_tags[i] if i < len(tgt_tags) else "(missing)"
        mark = "✓" if p == t else "✗"
        lines.append(f"  [{i + 1}] {mark}  predicted: {p}")
        if p != t:
            lines.append(f"        expected: {t}")
    return "\n".join(lines)


def detect_formula_metric(question):
    q_lower = question.lower()
    keywords = {
        "WACC": ["wacc", "weighted average cost of capital"],
        "EAR": ["ear", "effective annual rate", "effective interest rate"],
        "APR": ["apr", "annual percentage rate"],
        "ROI": ["roi", "return on investment"],
        "ROE": ["roe", "return on equity"],
        "ROA": ["roa", "return on assets"],
        "CAPM": ["capm", "expected return", "required return", "cost of equity"],
        "NPV": ["npv", "net present value"],
        "Sharpe Ratio": ["sharpe ratio"],
        "Beta": ["beta"],
        "Gross Profit Margin": ["gross profit margin", "gross margin"],
        "Net Profit Margin": ["net profit margin", "net margin"],
        "Operating Margin": ["operating margin"],
        "Current Ratio": ["current ratio"],
        "Black-Scholes": ["black-scholes", "black scholes", "option price"],
    }
    for metric, kws in keywords.items():
        for kw in kws:
            if kw in q_lower:
                return metric
    return None


def parse_appworld_response(response_str):
    """Parse the structured JSON response from an AppWorld evaluation."""
    try:
        return json.loads(response_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def format_transcript(transcript, max_turns=None, max_chars=None):
    """Format an AppWorld ReAct transcript for display."""
    if not transcript:
        return "(no transcript available)"
    turns = transcript if max_turns is None else transcript[: max_turns * 2]
    parts = []
    for msg in turns:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "..."
        parts.append(f"[{role}]\n{content}")
    text = "\n\n".join(parts)
    if max_turns and len(transcript) > max_turns * 2:
        text += f"\n\n... ({len(transcript) - max_turns * 2} more messages)"
    return text


def print_result(r, task, show_response=False):
    status = "CORRECT" if r["is_correct"] else "INCORRECT"
    print(f"### Sample {r['index']} — {status}")

    if task == "appworld":
        tid = r.get("task_id", "")
        if tid:
            print(f"**Task ID:** {tid}")
        instr = r.get("question", "")
        if instr and instr != tid:
            print(f"**Instruction:** {instr[:600]}{'...' if len(instr) > 600 else ''}")
        print(f"**Result:** {r['predicted']}")

        info = parse_appworld_response(r.get("response", ""))
        passes = info.get("num_passes", 0)
        failures = info.get("num_failures", 0)
        failure_msgs = info.get("failure_messages", [])
        if passes or failures:
            print(f"**Assertions:** {passes} passed, {failures} failed")
        if failure_msgs:
            print("**Failure reasons:**")
            for fm in failure_msgs:
                print(f"  - {fm[:300]}")

        if show_response:
            transcript = info.get("transcript", [])
            if transcript:
                print(f"**Transcript ({len(transcript)} messages):**")
                print(format_transcript(transcript))
            elif r.get("response"):
                print(f"**Raw Response:**\n{r['response'][:2000]}")

    elif task == "finer":
        sub_qs = extract_finer_questions(r["question"])
        print("**Sub-questions:**")
        for sq in sub_qs:
            print(f"  {sq[:300]}{'...' if len(sq) > 300 else ''}")
        print("**Answer comparison (per position):**")
        print(format_finer_comparison(r["predicted"], r["target"]))

        if show_response and r.get("response"):
            resp = r["response"]
            print(f"**Full Response:**\n{resp[:2000]}{'...' if len(resp) > 2000 else ''}")
    else:
        metric = detect_formula_metric(r["question"]) if task == "formula" else None
        if metric:
            print(f"**Metric:** {metric}")
        print(f"**Question:** {r['question'][:400]}{'...' if len(r['question']) > 400 else ''}")
        if r.get("context"):
            ctx = r["context"]
            print(f"**Context:** {ctx[:400]}{'...' if len(ctx) > 400 else ''}")
        print(f"**Expected:**  {r['target']}")
        print(f"**Predicted:** {r['predicted']}")

        if task == "formula":
            try:
                pred_f = float(r["predicted"])
                tgt_f = float(r["target"])
                if tgt_f != 0:
                    ratio = pred_f / tgt_f
                    if abs(ratio - 100) < 5:
                        print("  ⚠ Likely % vs decimal error (predicted is ~100× target)")
                    elif abs(ratio - 0.01) < 0.005:
                        print("  ⚠ Likely decimal vs % error (predicted is ~0.01× target)")
                    elif abs(ratio - 1000) < 50:
                        print("  ⚠ Likely unit scaling error (predicted is ~1000× target)")
                    elif abs(ratio - 1) < 0.01 and abs(pred_f - tgt_f) > 0.001:
                        print("  ⚠ Likely rounding/precision error (within 1%)")
            except (ValueError, TypeError):
                pass

        if show_response and r.get("response"):
            resp = r["response"]
            print(f"**Full Response:**\n{resp[:2000]}{'...' if len(resp) > 2000 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description="View and filter evaluation results from a SAGE iteration")
    parser.add_argument(
        "--prompt", default=None, help="Prompt to view: index (0,1,...) or dirname. Default: show overview only"
    )
    parser.add_argument("--incorrect", action="store_true", help="Show only incorrect results")
    parser.add_argument("--correct", action="store_true", help="Show only correct results")
    parser.add_argument("--index", type=int, help="Show specific sample by index")
    parser.add_argument("--task-id", type=str, help="AppWorld: filter by task ID (substring match)")
    parser.add_argument("--n", type=int, help="Maximum number of results to show")
    parser.add_argument("--show-response", action="store_true", help="Include full model response")
    parser.add_argument(
        "--show-transcript", action="store_true", help="AppWorld: show full ReAct transcript (implies --show-response)"
    )
    args = parser.parse_args()

    prompt_dirs = discover_prompts(DATASET_DIR)
    if not prompt_dirs:
        print(f"Error: no prompt directories found in {DATASET_DIR}")
        return

    prompt_summary(prompt_dirs)

    if args.prompt is None:
        print("Use --prompt <index> to view results for a specific prompt.")
        return

    target = resolve_prompt_dir(DATASET_DIR, prompt_dirs, args.prompt)
    if not target:
        return

    results_file = target / "results.jsonl"
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    results = load_results(results_file)
    task = detect_task(results)

    show_response = args.show_response or args.show_transcript

    print(f"# Results: {target.name}  (task: {task})\n")

    if args.index is not None:
        results = [r for r in results if r["index"] == args.index]
    elif args.task_id:
        results = [r for r in results if args.task_id in r.get("task_id", "")]
    elif args.incorrect:
        results = [r for r in results if not r["is_correct"]]
    elif args.correct:
        results = [r for r in results if r["is_correct"]]

    if args.n:
        results = results[: args.n]

    if not results:
        print("No matching results found.")
        return

    print(f"Showing {len(results)} result(s):\n")
    for r in results:
        print_result(r, task, show_response=show_response)


if __name__ == "__main__":
    main()
