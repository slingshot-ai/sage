#!/usr/bin/env python3
"""
Analyze error patterns in evaluation results from a SAGE iteration.

Works for FiNER (XBRL tag classification), Formula (financial calculation),
and AppWorld (agentic task completion) tasks.

The dataset directory contains one subdirectory per evaluated prompt:
  dataset/prompt_0_acc0.8500/  (best)
  dataset/prompt_1_acc0.8200/
  ...

Usage:
    python error_analysis.py                    # Overview of all prompts, then analyze best
    python error_analysis.py --prompt 0         # Analyze prompt 0
    python error_analysis.py --prompt 1         # Analyze prompt 1
    python error_analysis.py --prompt all       # Analyze all prompts
    python error_analysis.py --scale            # Formula: show scale/unit error breakdown
    python error_analysis.py --confusion        # FiNER: show top confused tag pairs
    python error_analysis.py --transcripts      # AppWorld: show transcript excerpts in errors
"""

import argparse
import json
import re
from collections import Counter
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


def analyze_basic(results, task, label=""):
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    incorrect = total - correct
    header = f"# Error Analysis: {label}" if label else f"# Error Analysis  (task: {task})"
    print(f"{header}\n")
    print(f"- **Total samples:** {total}")
    print(f"- **Correct:** {correct} ({correct / total * 100:.1f}%)")
    print(f"- **Incorrect:** {incorrect} ({incorrect / total * 100:.1f}%)")
    print()


def show_extraction_failures(errors):
    no_answer = [e for e in errors if e["predicted"] == "No final answer found"]
    if no_answer:
        print(f"### Failed to extract answer: {len(no_answer)} cases")
        for e in no_answer[:5]:
            print(f"  - Sample {e['index']}: expected `{e['target']}`")
        print()
    return [e for e in errors if e["predicted"] != "No final answer found"]


def extract_finer_questions(question_text):
    match = re.search(r"Answer the following \d+ independent questions.*?\n(.*)", question_text, re.DOTALL)
    if match:
        parts = re.split(r"\n(?=\d+\.)", match.group(1).strip())
        return [p.strip() for p in parts if p.strip()]
    return [question_text[-800:].strip()]


def compute_per_tag_accuracy(all_results):
    tag_correct = Counter()
    tag_total = Counter()
    for r in all_results:
        pred_tags = [t.strip() for t in r["predicted"].split(",")]
        tgt_tags = [t.strip() for t in r["target"].split(",")]
        for i in range(min(len(pred_tags), len(tgt_tags))):
            tag_total[tgt_tags[i]] += 1
            if pred_tags[i] == tgt_tags[i]:
                tag_correct[tgt_tags[i]] += 1
    return tag_correct, tag_total


def analyze_finer_errors(all_results, errors, top_n, show_confusion, show_per_tag):
    print(f"## FiNER Error Analysis ({len(errors)} incorrect samples)\n")

    position_errors = Counter()
    tag_confusion = Counter()
    num_tags_wrong = Counter()

    for e in errors:
        pred_tags = [t.strip() for t in e["predicted"].split(",")]
        tgt_tags = [t.strip() for t in e["target"].split(",")]
        n = max(len(pred_tags), len(tgt_tags))
        wrong_in_sample = 0
        for i in range(n):
            p = pred_tags[i] if i < len(pred_tags) else "(missing)"
            t = tgt_tags[i] if i < len(tgt_tags) else "(missing)"
            if p != t:
                position_errors[i + 1] += 1
                tag_confusion[(t, p)] += 1
                wrong_in_sample += 1
        num_tags_wrong[wrong_in_sample] += 1

    print("### Wrong sub-questions per sample\n")
    for n_wrong in sorted(num_tags_wrong):
        count = num_tags_wrong[n_wrong]
        print(f"  - {n_wrong} wrong: {count} samples ({count / len(errors) * 100:.1f}%)")
    print()

    print("### Errors by sub-question position\n")
    for pos in sorted(position_errors):
        cnt = position_errors[pos]
        print(f"  - Position {pos}: {cnt} errors ({cnt / len(errors) * 100:.1f}% of error cases)")
    print()

    if show_confusion and tag_confusion:
        print("### Top confused tag pairs (expected → predicted)\n")
        for (expected, predicted), count in tag_confusion.most_common(25):
            print(f"  - `{expected}` → `{predicted}`: {count}×")
        print()

    if show_per_tag:
        tag_correct, tag_total = compute_per_tag_accuracy(all_results)
        tags_with_errors = [
            (tag, tag_correct.get(tag, 0), tag_total[tag])
            for tag in tag_total
            if tag_correct.get(tag, 0) < tag_total[tag]
        ]
        tags_with_errors.sort(key=lambda x: (x[1] / x[2] if x[2] else 1, -x[2]))
        print("### Per-tag accuracy (tags with errors, sorted by accuracy ascending)\n")
        for tag, correct, total in tags_with_errors:
            acc = correct / total * 100 if total else 0
            print(f"  - `{tag}`: {correct}/{total} ({acc:.0f}%) — {total - correct} errors")
        print()

    print(f"### Sample error cases (first {top_n})\n")
    for i, e in enumerate(errors[:top_n]):
        sub_qs = extract_finer_questions(e["question"])
        pred_tags = [t.strip() for t in e["predicted"].split(",")]
        tgt_tags = [t.strip() for t in e["target"].split(",")]
        print(f"**Error {i + 1}** (Sample {e['index']})")
        for j, sq in enumerate(sub_qs):
            p = pred_tags[j] if j < len(pred_tags) else "(missing)"
            t = tgt_tags[j] if j < len(tgt_tags) else "(missing)"
            mark = "✓" if p == t else "✗"
            print(f"  [{j + 1}] {mark} {sq[:200]}{'...' if len(sq) > 200 else ''}")
            if p != t:
                print(f"       expected:  `{t}`")
                print(f"       predicted: `{p}`")
        print()


def classify_formula_error(predicted, target):
    try:
        p = float(predicted)
        t = float(target)
    except (ValueError, TypeError):
        return "parse_failure"

    if t == 0:
        return "zero_target"

    ratio = p / t
    if abs(ratio - 100) < 2:
        return "pct_vs_decimal"
    if abs(ratio - 0.01) < 0.002:
        return "decimal_vs_pct"
    if abs(ratio - 1000) < 20:
        return "unit_scaling_1000x"
    if abs(ratio - 0.001) < 0.00002:
        return "unit_scaling_div1000"
    if abs(ratio - 1000000) < 20000:
        return "unit_scaling_1m"
    if abs(p - t) < 0.01 * abs(t):
        return "rounding"
    if abs(p - t) < 0.05 * abs(t):
        return "near_miss"
    if abs(ratio - 1) < 0.20:
        return "moderate_miss"
    return "large_error"


_METRIC_KEYWORDS = {
    "WACC": ["wacc", "weighted average cost of capital"],
    "EAR": ["ear", "effective annual rate", "effective interest rate", "effective rate"],
    "APR": ["apr", "annual percentage rate"],
    "ROI": ["roi", "return on investment"],
    "ROE": ["roe", "return on equity"],
    "ROA": ["roa", "return on assets"],
    "CAPM/Expected Return": ["capm", "expected return", "required return", "cost of equity"],
    "NPV": ["npv", "net present value"],
    "Sharpe Ratio": ["sharpe ratio"],
    "Jensen's Alpha": ["jensen", "alpha"],
    "Beta": ["beta"],
    "Gross Profit Margin": ["gross profit margin", "gross margin"],
    "Net Profit Margin": ["net profit margin", "net margin"],
    "Operating Margin": ["operating margin", "operating profit margin"],
    "Current Ratio": ["current ratio"],
    "Dividend Yield": ["dividend yield"],
    "Dividend Payout": ["dividend payout", "payout ratio"],
    "Inflation Rate": ["inflation rate"],
    "Interest Coverage": ["interest coverage"],
    "P/E Ratio": ["p/e ratio", "price-to-earnings", "price to earnings"],
    "P/B Ratio": ["p/b ratio", "price-to-book", "price to book"],
    "EPS": ["earnings per share", "eps"],
    "Black-Scholes": ["black-scholes", "black scholes", "option price", "call option", "put option"],
    "Put-Call Parity": ["put-call parity", "put call parity"],
    "Annuity/NPV Payments": ["annuity", "periodic payment", "present value of"],
    "PPP": ["purchasing power parity", "ppp"],
    "Bond/Debt": ["bond price", "coupon", "yield to maturity", "ytm"],
}


def detect_financial_metric(question):
    q_lower = question.lower()
    for metric, keywords in _METRIC_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return metric
    return "Other"


def analyze_formula_errors(errors, top_n, show_scale):
    print(f"## Formula Error Analysis ({len(errors)} incorrect samples)\n")

    categories = Counter()
    for e in errors:
        cat = classify_formula_error(e["predicted"], e["target"])
        categories[cat] = categories.get(cat, 0) + 1

    cat_labels = {
        "pct_vs_decimal": "Percentage expressed as integer (×100 off)",
        "decimal_vs_pct": "Decimal when percentage expected (÷100 off)",
        "unit_scaling_1000x": "Unit scaling error (×1000 off, e.g. thousands vs units)",
        "unit_scaling_div1000": "Unit scaling error (÷1000 off)",
        "unit_scaling_1m": "Unit scaling error (×1M off)",
        "rounding": "Near-correct (within 1%, rounding/precision)",
        "near_miss": "Near miss (within 5%)",
        "moderate_miss": "Moderate miss (within 20%)",
        "large_error": "Large error (formula or conceptual mistake)",
        "zero_target": "Target is zero (special case)",
        "parse_failure": "Could not parse predicted or target as number",
    }

    print("### Error categories\n")
    for cat, count in categories.most_common():
        label = cat_labels.get(cat, cat)
        print(f"  - **{label}**: {count} ({count / len(errors) * 100:.1f}%)")
    print()

    metric_errors = Counter()
    metric_categories = {}
    for e in errors:
        metric = detect_financial_metric(e["question"])
        cat = classify_formula_error(e["predicted"], e["target"])
        metric_errors[metric] += 1
        metric_categories.setdefault(metric, Counter())[cat] += 1

    print("### Errors by financial metric\n")
    for metric, count in metric_errors.most_common():
        top_cats = metric_categories[metric].most_common(3)
        cats_str = ", ".join(f"{cat_labels.get(c, c).split('(')[0].strip()} ({n})" for c, n in top_cats)
        print(f"  - **{metric}**: {count} errors — {cats_str}")
    print()

    if show_scale:
        scale_errors = [
            e
            for e in errors
            if classify_formula_error(e["predicted"], e["target"])
            in ("pct_vs_decimal", "decimal_vs_pct", "unit_scaling_1000x", "unit_scaling_div1000", "unit_scaling_1m")
        ]
        if scale_errors:
            print("### Scale/unit error examples\n")
            for e in scale_errors[:10]:
                metric = detect_financial_metric(e["question"])
                cat = classify_formula_error(e["predicted"], e["target"])
                print(f"  - Sample {e['index']} ({metric}, {cat}): expected `{e['target']}`, got `{e['predicted']}`")
                q = e["question"][:200]
                print(f"    Question: {q}{'...' if len(e['question']) > 200 else ''}")
            print()

    print(f"### Sample error cases (first {top_n})\n")
    for i, e in enumerate(errors[:top_n]):
        cat = classify_formula_error(e["predicted"], e["target"])
        label = cat_labels.get(cat, cat)
        metric = detect_financial_metric(e["question"])
        print(f"**Error {i + 1}** (Sample {e['index']}) — {metric} — {label}")
        q = e["question"][:300]
        print(f"- Question: {q}{'...' if len(e['question']) > 300 else ''}")
        print(f"- Expected:  `{e['target']}`")
        print(f"- Predicted: `{e['predicted']}`")
        print()


def parse_appworld_response(response_str):
    """Parse the structured JSON response from an AppWorld evaluation."""
    try:
        return json.loads(response_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def classify_appworld_error(info):
    """Classify an AppWorld failure into a high-level category based on response data."""
    failure_msgs = info.get("failure_messages", [])
    transcript = info.get("transcript", [])
    num_passes = info.get("num_passes", 0)

    if not transcript:
        return "no_transcript"

    joined_failures = " ".join(fm.lower() for fm in failure_msgs)

    if not failure_msgs and num_passes == 0:
        return "no_assertions_ran"
    if any("complete_task" in fm.lower() for fm in failure_msgs):
        return "task_not_completed"
    if any(kw in joined_failures for kw in ["assertionerror", "assert"]):
        return "assertion_mismatch"
    if any(kw in joined_failures for kw in ["error", "exception", "traceback"]):
        return "runtime_error"
    if num_passes > 0:
        return "partial_success"
    return "other"


def compute_transcript_stats(info):
    """Compute basic statistics about an AppWorld ReAct transcript."""
    transcript = info.get("transcript", [])
    assistant_turns = [m for m in transcript if m.get("role") == "assistant"]
    user_turns = [m for m in transcript if m.get("role") == "user"]
    total_chars = sum(len(m.get("content", "")) for m in transcript)
    return {
        "num_messages": len(transcript),
        "num_assistant_turns": len(assistant_turns),
        "num_user_turns": len(user_turns),
        "total_chars": total_chars,
    }


def analyze_appworld_errors(all_results, errors, top_n, show_transcripts=False):
    print(f"## AppWorld Error Analysis ({len(errors)} failed tasks)\n")

    # Overall stats
    total_passes = sum(parse_appworld_response(r.get("response", "")).get("num_passes", 0) for r in all_results)
    total_failures = sum(parse_appworld_response(r.get("response", "")).get("num_failures", 0) for r in all_results)
    total_assertions = total_passes + total_failures
    sgc = total_passes / total_assertions if total_assertions > 0 else 0.0
    print("### Aggregate assertion stats\n")
    print(f"  - Total assertions: {total_assertions}")
    print(
        f"  - Passed: {total_passes} ({total_passes / total_assertions * 100:.1f}%)"
        if total_assertions
        else "  - Passed: 0"
    )
    print(f"  - Failed: {total_failures}")
    print(f"  - SGC (sub-goal completion): {sgc:.1%}")
    print()

    # Error classification
    categories = Counter()
    error_infos = []
    for e in errors:
        info = parse_appworld_response(e.get("response", ""))
        cat = classify_appworld_error(info)
        categories[cat] += 1
        error_infos.append((e, info, cat))

    cat_labels = {
        "no_transcript": "No transcript captured (crash or timeout)",
        "no_assertions_ran": "No assertions executed",
        "task_not_completed": "Task not marked as completed (complete_task not called)",
        "assertion_mismatch": "Assertion failures (wrong state/values)",
        "runtime_error": "Runtime errors during execution",
        "partial_success": "Partial success (some assertions passed)",
        "other": "Other failures",
    }

    print("### Error categories\n")
    for cat, count in categories.most_common():
        label = cat_labels.get(cat, cat)
        print(f"  - **{label}**: {count} ({count / len(errors) * 100:.1f}%)")
    print()

    # Transcript length stats for errors vs correct
    error_stats = [compute_transcript_stats(info) for _, info, _ in error_infos]
    correct_results = [r for r in all_results if r["is_correct"]]
    correct_stats = [compute_transcript_stats(parse_appworld_response(r.get("response", ""))) for r in correct_results]

    if error_stats:
        avg_err_turns = sum(s["num_assistant_turns"] for s in error_stats) / len(error_stats)
        avg_err_msgs = sum(s["num_messages"] for s in error_stats) / len(error_stats)
        print("### Transcript stats (failed tasks)\n")
        print(f"  - Avg messages: {avg_err_msgs:.1f}")
        print(f"  - Avg agent turns: {avg_err_turns:.1f}")
    if correct_stats:
        avg_ok_turns = sum(s["num_assistant_turns"] for s in correct_stats) / len(correct_stats)
        avg_ok_msgs = sum(s["num_messages"] for s in correct_stats) / len(correct_stats)
        print("### Transcript stats (passed tasks)\n")
        print(f"  - Avg messages: {avg_ok_msgs:.1f}")
        print(f"  - Avg agent turns: {avg_ok_turns:.1f}")
    print()

    # Common failure message patterns
    all_failure_msgs = []
    for e in errors:
        info = parse_appworld_response(e.get("response", ""))
        all_failure_msgs.extend(info.get("failure_messages", []))

    if all_failure_msgs:
        # Normalize failure messages for grouping
        failure_patterns = Counter()
        for fm in all_failure_msgs:
            # Truncate to first 120 chars for grouping
            pattern = fm[:120].strip()
            failure_patterns[pattern] += 1

        print("### Most common failure messages\n")
        for pattern, count in failure_patterns.most_common(15):
            print(f"  - ({count}×) {pattern}")
        print()

    # Detailed error cases
    print(f"### Sample error cases (first {top_n})\n")
    for i, (e, info, cat) in enumerate(error_infos[:top_n]):
        tid = e.get("task_id", "")
        label = cat_labels.get(cat, cat)
        passes = info.get("num_passes", 0)
        fails = info.get("num_failures", 0)
        print(f"**Error {i + 1}** (Sample {e['index']}) — {label}")
        if tid:
            print(f"- Task ID: {tid}")
        instr = e.get("question", "")
        if instr and instr != tid:
            print(f"- Instruction: {instr[:400]}{'...' if len(instr) > 400 else ''}")
        print(f"- Assertions: {passes} passed, {fails} failed")
        failure_msgs = info.get("failure_messages", [])
        if failure_msgs:
            print("- Failures:")
            for fm in failure_msgs[:5]:
                print(f"    - {fm[:200]}")

        if show_transcripts:
            transcript = info.get("transcript", [])
            if transcript:
                # Show last few turns which are most relevant to understanding failure
                last_turns = transcript[-6:]
                print(f"- Transcript (last {len(last_turns)} messages of {len(transcript)}):")
                for msg in last_turns:
                    role = msg.get("role", "unknown").upper()
                    content = msg.get("content", "").strip()
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"    [{role}] {content}")
        print()


def analyze_generic_errors(errors, top_n):
    print(f"## Error Cases ({len(errors)} total)\n")
    for i, e in enumerate(errors[:top_n]):
        print(f"**Error {i + 1}** (Sample {e['index']})")
        print(f"- Question: {e['question'][:300]}{'...' if len(e['question']) > 300 else ''}")
        print(f"- Expected:  `{e['target']}`")
        print(f"- Predicted: `{e['predicted']}`")
        print()


def analyze_one(prompt_dir, args):
    """Run full error analysis on a single prompt directory."""
    results_file = prompt_dir / "results.jsonl"
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    results = load_results(results_file)
    task = detect_task(results)
    analyze_basic(results, task, label=prompt_dir.name)

    errors = [r for r in results if not r["is_correct"]]
    errors = show_extraction_failures(errors)

    if task == "appworld":
        analyze_appworld_errors(results, errors, args.top, show_transcripts=args.transcripts)
    elif task == "finer":
        analyze_finer_errors(results, errors, args.top, show_confusion=args.confusion, show_per_tag=args.per_tag)
    elif task == "formula":
        analyze_formula_errors(errors, args.top, show_scale=args.scale)
    else:
        analyze_generic_errors(errors, args.top)


def main():
    parser = argparse.ArgumentParser(description="Analyze error patterns in evaluation results")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to analyze: index (0,1,...), dirname, or 'all'. Default: overview + best prompt analysis",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of detailed error cases to show (default: 20)")
    parser.add_argument("--scale", action="store_true", help="Formula: show examples of scale/unit errors")
    parser.add_argument("--confusion", action="store_true", help="FiNER: show top confused tag pairs")
    parser.add_argument("--per-tag", action="store_true", help="FiNER: show per-tag accuracy breakdown")
    parser.add_argument("--transcripts", action="store_true", help="AppWorld: show transcript excerpts for error cases")
    args = parser.parse_args()

    prompt_dirs = discover_prompts(DATASET_DIR)
    if not prompt_dirs:
        print(f"Error: no prompt directories found in {DATASET_DIR}")
        return

    prompt_summary(prompt_dirs)

    if args.prompt is None:
        analyze_one(prompt_dirs[0], args)
    elif args.prompt == "all":
        for d in prompt_dirs:
            analyze_one(d, args)
            print(f"{'=' * 60}\n")
    else:
        target = resolve_prompt_dir(DATASET_DIR, prompt_dirs, args.prompt)
        if target:
            analyze_one(target, args)


if __name__ == "__main__":
    main()
