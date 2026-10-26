#!/usr/bin/env python3
"""Plot accuracy & tokens (mean + IQR) and test accuracy (mean + IQR), grouped by strategy & N & M."""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(ROOT, "results")


def load_run(run_dir):
    """Load the iteration log and ``run_config.json`` of one run directory."""
    iterations = None
    for fname in ("train_results.json", "final_results.json"):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            iterations = data.get("iterations") or data.get("iteration_log", [])
            if iterations:
                break
    if not iterations:
        return None

    config = {}
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    initial_test = None
    final_test = None
    final_path = os.path.join(run_dir, "final_results.json")
    if os.path.exists(final_path):
        with open(final_path) as f:
            fr = json.load(f)
        initial_test = fr.get("initial_test")
        final_test = fr.get("final_test")

    return {
        "iterations": iterations,
        "config": config,
        "initial_test": initial_test,
        "final_test": final_test,
    }


STRATEGY_NAMES = {
    "spo-rs": "SPO-RS",
    "spo-ga": "SPO-GA",
    "sage": "SAGE",
}


def label(config, include_task=True):
    """Human-readable run label using SPO-RS / SPO-GA / SAGE naming."""
    raw = config.get("strategy", "?")
    name = STRATEGY_NAMES.get(raw, raw)
    parts = [name]
    if include_task:
        parts.append(config.get("task_name", ""))
    n, m = config.get("top_n"), config.get("num_proposals")
    if n is not None and m is not None:
        parts.append(f"N={n} M={m}")
    return " | ".join(p for p in parts if p)


def _build_results_dict(by_task, tasks) -> dict:
    """Build a structured dict with all numeric results for JSON serialisation."""
    results: dict = {}
    for task in tasks:
        runs = by_task[task]
        groups = defaultdict(list)
        for run in runs:
            groups[_aggregate_key(run["config"])].append(run)
        keys_sorted = sorted(groups.keys(), key=_sort_aggregate_key)

        task_settings: dict = {}
        for key in keys_sorted:
            grp = groups[key]
            cfg0 = grp[0]["config"]
            setting = label(cfg0, include_task=False)
            entry: dict = {"n_runs": len(grp)}

            xs, means, lows, highs = _aggregate_accuracy_series(grp)
            if xs.size:
                entry["best_train_accuracy"] = {
                    "iterations": xs.tolist(),
                    "mean": means.tolist(),
                    "q25": lows.tolist(),
                    "q75": highs.tolist(),
                }

            xs, means, lows, highs = _aggregate_val_accuracy_series(grp)
            if xs.size:
                entry["val_accuracy"] = {
                    "iterations": xs.tolist(),
                    "mean": means.tolist(),
                    "q25": lows.tolist(),
                    "q75": highs.tolist(),
                }

            xs, means, lows, highs = _aggregate_token_series(grp)
            if xs.size:
                entry["best_prompt_tokens"] = {
                    "iterations": xs.tolist(),
                    "mean": means.tolist(),
                    "q25": lows.tolist(),
                    "q75": highs.tolist(),
                }

            ia = [
                r["initial_test"]["accuracy"]
                for r in grp
                if r.get("initial_test") and r["initial_test"].get("accuracy") is not None
            ]
            fa = [
                r["final_test"]["accuracy"]
                for r in grp
                if r.get("final_test") and r["final_test"].get("accuracy") is not None
            ]
            im = _mean_and_iqr(ia)
            fm = _mean_and_iqr(fa)
            if im or fm or ia or fa:
                test_entry: dict = {}
                if im:
                    test_entry["initial"] = {"mean": im[0], "q25": im[1], "q75": im[2]}
                if ia:
                    test_entry["initial_best"] = float(max(ia))
                if fm:
                    test_entry["final"] = {"mean": fm[0], "q25": fm[1], "q75": fm[2]}
                if fa:
                    test_entry["final_best"] = float(max(fa))
                entry["test_accuracy"] = test_entry

            task_settings[setting] = entry
        results[task] = task_settings
    return results


def _save_results_json(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}")


def _print_summary_table(results: dict):
    """Print a concise summary table (one row per task × setting)."""

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    header = (
        f"{'Task':<12} {'Setting':<22} {'Runs':>4}  {'Train':>8} {'Val':>8} {'Test-I':>8} {'Test-F':>8} {'Best-F':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for task, settings in results.items():
        for setting, entry in settings.items():
            n = entry["n_runs"]
            train = entry.get("best_train_accuracy", {}).get("mean")
            train_final = train[-1] if train else None
            val = entry.get("val_accuracy", {}).get("mean")
            val_final = val[-1] if val else None
            ta = entry.get("test_accuracy", {})
            ti = ta.get("initial", {}).get("mean")
            tf = ta.get("final", {}).get("mean")
            bf = ta.get("final_best")
            print(
                f"{task:<12} {setting:<22} {n:>4}  "
                f"{fmt(train_final):>8} {fmt(val_final):>8} "
                f"{fmt(ti):>8} {fmt(tf):>8} {fmt(bf):>8}"
            )
    print(sep)


def _filename_suffix(args) -> str:
    """Suffix for output PNGs when filters are applied (avoids clobbering full plots)."""
    parts = []
    if getattr(args, "task", None):
        parts.append(f"task_{args.task}")
    if getattr(args, "strategy", None):
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", args.strategy.strip()).strip("_").lower()
        if safe:
            parts.append(f"strat_{safe}")
    return ("_" + "_".join(parts)) if parts else ""


def _any_token_data(by_task, tasks) -> bool:
    for task in tasks:
        for run in by_task[task]:
            for it in run["iterations"]:
                if it.get("best_prompt_tokens") is not None:
                    return True
    return False


def _any_test_data(by_task, tasks) -> bool:
    for task in tasks:
        for run in by_task[task]:
            if run.get("initial_test") or run.get("final_test"):
                return True
    return False


def _aggregate_key(config: dict) -> tuple:
    """Hashable key: same task (caller), strategy label, top_n, num_proposals."""
    raw = config.get("strategy", "?")
    strat = STRATEGY_NAMES.get(raw, raw)
    return (strat, config.get("top_n"), config.get("num_proposals"))


def _sort_aggregate_key(k: tuple) -> tuple:
    s, n, m = k
    return (s, n if n is not None else -1, m if m is not None else -1)


def _iter_accuracy_by_iteration(run) -> dict[int, float]:
    return {it["iteration"]: it["best_accuracy"] for it in run["iterations"]}


def _aggregate_accuracy_series(runs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-iteration mean and IQR (25th–75th percentile) across ``runs``."""
    maps = [_iter_accuracy_by_iteration(r) for r in runs]
    all_iters = sorted(set().union(*(m.keys() for m in maps)))
    xs: list[int] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for k in all_iters:
        vals = [m[k] for m in maps if k in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mu = float(np.mean(arr))
        lo, hi = (float(x) for x in np.percentile(arr, [25.0, 75.0]))
        lo = max(0.0, min(lo, 1.0))
        hi = max(0.0, min(hi, 1.0))
        xs.append(k)
        means.append(mu)
        lows.append(lo)
        highs.append(hi)
    return (
        np.asarray(xs, dtype=int),
        np.asarray(means, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def _any_val_data(by_task, tasks) -> bool:
    for task in tasks:
        for run in by_task[task]:
            for it in run["iterations"]:
                if it.get("val_accuracy") is not None:
                    return True
    return False


def _iter_val_accuracy_by_iteration(run) -> dict[int, float]:
    out: dict[int, float] = {}
    for it in run["iterations"]:
        v = it.get("val_accuracy")
        if v is not None:
            out[it["iteration"]] = float(v)
    return out


def _aggregate_val_accuracy_series(runs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-iteration mean and IQR for validation accuracy across ``runs``."""
    maps = [_iter_val_accuracy_by_iteration(r) for r in runs]
    all_iters = sorted(set().union(*(m.keys() for m in maps)))
    xs: list[int] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for k in all_iters:
        vals = [m[k] for m in maps if k in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mu = float(np.mean(arr))
        lo, hi = (float(x) for x in np.percentile(arr, [25.0, 75.0]))
        lo = max(0.0, min(lo, 1.0))
        hi = max(0.0, min(hi, 1.0))
        xs.append(k)
        means.append(mu)
        lows.append(lo)
        highs.append(hi)
    return (
        np.asarray(xs, dtype=int),
        np.asarray(means, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def _iter_tokens_by_iteration(run) -> dict[int, float]:
    out: dict[int, float] = {}
    for it in run["iterations"]:
        t = it.get("best_prompt_tokens")
        if t is not None:
            out[it["iteration"]] = float(t)
    return out


def _aggregate_token_series(runs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-iteration mean token count and IQR across ``runs``."""
    maps = [_iter_tokens_by_iteration(r) for r in runs]
    all_iters = sorted(set().union(*(m.keys() for m in maps)))
    xs: list[int] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for k in all_iters:
        vals = [m[k] for m in maps if k in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mu = float(np.mean(arr))
        lo, hi = (float(x) for x in np.percentile(arr, [25.0, 75.0]))
        lo = max(0.0, lo)
        xs.append(k)
        means.append(mu)
        lows.append(lo)
        highs.append(hi)
    return (
        np.asarray(xs, dtype=int),
        np.asarray(means, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def _mean_and_iqr(vals: list[float]) -> tuple[float, float, float] | None:
    """Mean and IQR (25th–75th percentile) across values; same rule as ``_aggregate_accuracy_series``.

    Returns ``(mean, q25, q75)`` with quartiles clipped to ``[0, 1]``, or ``None`` if empty.
    """
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    mu = float(np.mean(arr))
    lo, hi = (float(x) for x in np.percentile(arr, [25.0, 75.0]))
    lo = max(0.0, min(lo, 1.0))
    hi = max(0.0, min(hi, 1.0))
    return mu, lo, hi


def _plot_test_accuracy_figure(by_task, tasks):
    """Grouped markers by setting: mean test accuracy with IQR error bars (``final_results.json``)."""
    ncols = min(len(tasks), 3)
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        runs = by_task[task]
        groups = defaultdict(list)
        for run in runs:
            groups[_aggregate_key(run["config"])].append(run)
        keys_sorted = sorted(groups.keys(), key=_sort_aggregate_key)
        n_groups = len(keys_sorted)

        has_initial = any(r.get("initial_test") for r in runs)
        has_final = any(r.get("final_test") for r in runs)
        test_y_min = np.inf
        test_y_max = -np.inf

        if not has_initial and not has_final:
            ax.text(
                0.5,
                0.5,
                "No initial_test / final_test in final_results.json",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        elif n_groups == 0:
            ax.text(
                0.5,
                0.5,
                "No grouped runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        else:
            x = np.arange(n_groups, dtype=float)
            x_off = 0.18
            both = has_initial and has_final
            leg_init = True
            leg_fin = True
            for i, key in enumerate(keys_sorted):
                grp = groups[key]
                if has_initial:
                    ia = [
                        r["initial_test"]["accuracy"]
                        for r in grp
                        if r.get("initial_test") and r["initial_test"].get("accuracy") is not None
                    ]
                    mi = _mean_and_iqr(ia)
                    if mi is not None:
                        mu, lo, hi = mi
                        test_y_min = min(test_y_min, lo, mu)
                        test_y_max = max(test_y_max, hi, mu)
                        xpos = x[i] - x_off if both else x[i]
                        ax.errorbar(
                            xpos,
                            mu,
                            yerr=np.array([[mu - lo], [hi - mu]]),
                            fmt="o",
                            markersize=7,
                            capsize=4,
                            capthick=1.2,
                            elinewidth=1.2,
                            color="C0",
                            label="Initial test (baseline)" if leg_init else None,
                        )
                        leg_init = False
                if has_final:
                    fa = [
                        r["final_test"]["accuracy"]
                        for r in grp
                        if r.get("final_test") and r["final_test"].get("accuracy") is not None
                    ]
                    mi = _mean_and_iqr(fa)
                    if mi is not None:
                        mu, lo, hi = mi
                        test_y_min = min(test_y_min, lo, mu)
                        test_y_max = max(test_y_max, hi, mu)
                        xpos = x[i] + x_off if both else x[i]
                        ax.errorbar(
                            xpos,
                            mu,
                            yerr=np.array([[mu - lo], [hi - mu]]),
                            fmt="o",
                            markersize=7,
                            capsize=4,
                            capthick=1.2,
                            elinewidth=1.2,
                            color="C1",
                            label="Final test (best prompt)" if leg_fin else None,
                        )
                        leg_fin = False

            tick_labels = []
            for key in keys_sorted:
                g = groups[key]
                cfg0 = g[0]["config"]
                leg = label(cfg0, include_task=False)
                if len(g) > 1:
                    leg = f"{leg} (n={len(g)})"
                tick_labels.append(leg)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
            ax.legend(fontsize=8, loc="lower right")

        ax.set_ylabel("Test accuracy", fontsize=12)
        ax.set_title(f"Task: {task}", fontsize=14)
        ax.grid(True, axis="y", alpha=0.3)
        if np.isfinite(test_y_min) and np.isfinite(test_y_max):
            span = test_y_max - test_y_min
            ym = float(plt.rcParams["axes.ymargin"])
            if span > 0:
                pad = ym * span
            else:
                pad = max(ym * max(abs(test_y_min), 0.05), 0.02)
            ax.set_ylim(test_y_min - pad, test_y_max + pad)
        else:
            ax.set_ylim(0, 1.05)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "SPO — Test accuracy by setting (mean; IQR across runs with same strategy & P, Q)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _plot_accuracy_figure(by_task, tasks):
    """Mean best accuracy per iteration; runs with same strategy + N + M are averaged (IQR band)."""
    ncols = min(len(tasks), 3)
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        runs = by_task[task]
        groups = defaultdict(list)
        for run in runs:
            groups[_aggregate_key(run["config"])].append(run)

        for key in sorted(groups.keys(), key=_sort_aggregate_key):
            grp = groups[key]
            cfg0 = grp[0]["config"]
            xs, means, lows, highs = _aggregate_accuracy_series(grp)
            if xs.size == 0:
                continue
            leg = label(cfg0, include_task=False)
            if len(grp) > 1:
                leg = f"{leg} (n={len(grp)})"
            (line,) = ax.plot(
                xs,
                means,
                "o-",
                linewidth=2,
                markersize=5,
                label=leg,
            )
            color = line.get_color()
            ax.fill_between(xs, lows, highs, alpha=0.28, color=color, linewidth=0)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best accuracy", fontsize=12)
        ax.set_title(f"Task: {task}", fontsize=14)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "SPO — Best accuracy per iteration (mean; shaded IQR across same-setting runs)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _plot_val_accuracy_figure(by_task, tasks):
    """Mean validation accuracy per iteration; same-setting runs averaged (IQR band)."""
    ncols = min(len(tasks), 3)
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        runs = by_task[task]
        groups = defaultdict(list)
        for run in runs:
            groups[_aggregate_key(run["config"])].append(run)

        for key in sorted(groups.keys(), key=_sort_aggregate_key):
            grp = groups[key]
            cfg0 = grp[0]["config"]
            xs, means, lows, highs = _aggregate_val_accuracy_series(grp)
            if xs.size == 0:
                continue
            leg = label(cfg0, include_task=False)
            if len(grp) > 1:
                leg = f"{leg} (n={len(grp)})"
            (line,) = ax.plot(
                xs,
                means,
                "o-",
                linewidth=2,
                markersize=5,
                label=leg,
            )
            color = line.get_color()
            ax.fill_between(xs, lows, highs, alpha=0.28, color=color, linewidth=0)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Validation accuracy", fontsize=12)
        ax.set_title(f"Task: {task}", fontsize=14)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "SPO — Validation accuracy per iteration (mean; shaded IQR across same-setting runs)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _plot_tokens_figure(by_task, tasks):
    """Mean best-prompt tokens per iteration; same-setting runs averaged (IQR band)."""
    ncols = min(len(tasks), 3)
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        runs = by_task[task]
        groups = defaultdict(list)
        for run in runs:
            groups[_aggregate_key(run["config"])].append(run)

        any_line = False
        for key in sorted(groups.keys(), key=_sort_aggregate_key):
            grp = groups[key]
            cfg0 = grp[0]["config"]
            xs, means, lows, highs = _aggregate_token_series(grp)
            if xs.size == 0:
                continue
            any_line = True
            leg = label(cfg0, include_task=False)
            if len(grp) > 1:
                leg = f"{leg} (n={len(grp)})"
            (line,) = ax.plot(
                xs,
                means,
                "s-",
                linewidth=2,
                markersize=4,
                label=leg,
            )
            color = line.get_color()
            ax.fill_between(xs, lows, highs, alpha=0.28, color=color, linewidth=0)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best prompt (tokens)", fontsize=12)
        ax.set_title(f"Task: {task}", fontsize=14)
        if any_line:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.text(
                0.5,
                0.5,
                "No best_prompt_tokens in logs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
        ax.grid(True, alpha=0.3)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "SPO — Best prompt tokens per iteration (mean; shaded IQR across same-setting runs)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main():
    """Plot best train accuracy vs iteration for saved runs (SPO-RS, SPO-GA, SAGE)."""
    p = argparse.ArgumentParser(
        description="Plot metrics vs iteration for saved runs; writes PNGs under --dir.",
    )
    p.add_argument(
        "--dir",
        "-d",
        default=DEFAULT_RESULTS_DIR,
        help=f"Folder of run directories, where PNGs are also written (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument("--task", help="Filter by task name (e.g. formula, finer)")
    p.add_argument(
        "--strategy",
        help="Filter by strategy (spo-rs, spo-ga or sage)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display figures interactively (default: save only)",
    )
    args = p.parse_args()

    data_dir = os.path.abspath(args.dir)
    os.makedirs(data_dir, exist_ok=True)

    dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isfile(os.path.join(d, "run_config.json"))
    )
    runs = []
    for d in dirs:
        run = load_run(d)
        if not run:
            continue
        cfg = run["config"]
        if args.task and cfg.get("task_name") != args.task:
            continue
        if args.strategy and args.strategy.lower() not in cfg.get("strategy", "").lower():
            continue
        run["run_dir"] = d
        runs.append(run)

    if not runs:
        print("No matching runs found.", file=sys.stderr)
        sys.exit(1)

    by_task = defaultdict(list)
    for run in runs:
        task = run["config"].get("task_name", "unknown")
        by_task[task].append(run)
    tasks = sorted(by_task)

    results = _build_results_dict(by_task, tasks)

    suffix = _filename_suffix(args)
    json_path = os.path.join(data_dir, f"sage_accuracy_results{suffix}.json")
    _save_results_json(results, json_path)
    _print_summary_table(results)

    suffix = _filename_suffix(args)
    path_acc = os.path.join(data_dir, f"sage_plot_best_accuracy_per_iteration{suffix}.png")
    path_val = os.path.join(data_dir, f"sage_plot_val_accuracy_per_iteration{suffix}.png")
    path_test = os.path.join(data_dir, f"sage_plot_test_accuracy{suffix}.png")
    path_tok = os.path.join(data_dir, f"sage_plot_best_prompt_tokens_per_iteration{suffix}.png")

    fig_acc = _plot_accuracy_figure(by_task, tasks)
    fig_acc.savefig(path_acc, dpi=150)
    print(f"Saved {path_acc}")

    if _any_val_data(by_task, tasks):
        fig_val = _plot_val_accuracy_figure(by_task, tasks)
        fig_val.savefig(path_val, dpi=150)
        print(f"Saved {path_val}")
    else:
        print("Skipping val-accuracy plot (no val_accuracy in any run).", file=sys.stderr)

    if _any_test_data(by_task, tasks):
        fig_test = _plot_test_accuracy_figure(by_task, tasks)
        fig_test.savefig(path_test, dpi=150)
        print(f"Saved {path_test}")
    else:
        print("Skipping test-accuracy plot (no initial_test / final_test in any run).", file=sys.stderr)

    if _any_token_data(by_task, tasks):
        fig_tok = _plot_tokens_figure(by_task, tasks)
        fig_tok.savefig(path_tok, dpi=150)
        print(f"Saved {path_tok}")
    else:
        print("Skipping token plot (no best_prompt_tokens in any run).", file=sys.stderr)

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
