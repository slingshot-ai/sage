#!/usr/bin/env python3
"""Plot semivariogram and model-derived autocorrelation for prompt landscape.

Reads ``landscape/<task>_autocorrelation.json`` (produced by
``compute_autocorrelation.py``).

Outputs:
  - ``landscape/autocorrelation_overlay.png`` — two-panel figure:
      Left:  empirical semivariogram γ(d) with fitted exponential model
      Right: model-derived autocorrelation ρ(d) = exp(−d/a)
  - Summary table to stdout
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "landscape")

TASK_DISPLAY = {"formula": "Formula", "finer": "FiNER", "appworld": "AppWorld"}
TASK_COLORS = {"formula": "#2563eb", "finer": "#dc2626", "appworld": "#059669"}


def load_task_data(data_dir: str, task: str) -> dict | None:
    ac_path = os.path.join(data_dir, f"{task}_autocorrelation.json")
    if not os.path.exists(ac_path):
        return None
    with open(ac_path) as f:
        return json.load(f)


def plot_landscape(data_by_task: dict[str, dict], out_path: str):
    fig, (ax_var, ax_rho) = plt.subplots(1, 2, figsize=(13, 5))

    for task, data in data_by_task.items():
        x = np.array(data["bin_centres"])
        gamma = np.array(data["gamma"])
        ci_lo = np.array(data["ci_lo"])
        ci_hi = np.array(data["ci_hi"])
        color = TASK_COLORS.get(task, "gray")
        display = TASK_DISPLAY.get(task, task)
        sill = data["global_var_accuracy"]
        a = data.get("range")

        range_label = f" (a={a:.3f})" if a else ""
        label = f"{display}{range_label}"

        ax_var.plot(x, gamma, "o-", color=color, linewidth=2, markersize=5, label=label, zorder=3)
        ax_var.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color, linewidth=0, zorder=2)

        if data.get("fit_success") and a is not None:
            c0 = data["nugget"]
            c1 = data["partial_sill"]
            x_fit = np.linspace(0, x.max(), 200)
            y_model = c0 + c1 * (1.0 - np.exp(-x_fit / a))
            ax_var.plot(x_fit, y_model, "--", color=color, alpha=0.5, linewidth=1.5)

            rho_fit = np.exp(-x_fit / a)
            ax_rho.plot(x_fit, rho_fit, "-", color=color, linewidth=2.2, label=label, zorder=3)

    ax_var.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    for task, data in data_by_task.items():
        sill = data["global_var_accuracy"]
        color = TASK_COLORS.get(task, "gray")
        ax_var.axhline(sill, color=color, linestyle=":", linewidth=0.8, alpha=0.4)

    ax_var.set_xlabel("Cosine distance", fontsize=13)
    ax_var.set_ylabel("Semivariance γ(d)", fontsize=13)
    ax_var.set_title("Empirical Semivariogram", fontsize=13, fontweight="bold")
    ax_var.legend(fontsize=9, loc="lower right")
    ax_var.grid(True, alpha=0.25)
    ax_var.tick_params(labelsize=11)

    ax_rho.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    ax_rho.set_xlabel("Cosine distance", fontsize=13)
    ax_rho.set_ylabel("Autocorrelation ρ(d)", fontsize=13)
    ax_rho.set_title("Model Autocorrelation  ρ(d) = exp(−d/a)", fontsize=13, fontweight="bold")
    ax_rho.legend(fontsize=9, loc="upper right")
    ax_rho.grid(True, alpha=0.25)
    ax_rho.set_ylim(-0.05, 1.05)
    ax_rho.tick_params(labelsize=11)

    fig.suptitle("Prompt Landscape Analysis", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.close(fig)


def print_summary_table(data_by_task: dict[str, dict], tasks: list[str]):
    header = f"{'Task':<10} {'N':>5} {'Nugget':>10} {'Sill':>10} {'Range (a)':>10} {'ρ(0.1)':>10} {'ρ(0.3)':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    for task in tasks:
        data = data_by_task.get(task)
        if data is None:
            continue
        display = TASK_DISPLAY.get(task, task)
        n = data["n_prompts"]
        nugget = data.get("nugget")
        sill_fit = data.get("sill")
        a = data.get("range")
        rho_01 = data.get("rho_at_0.1")
        rho_03 = data.get("rho_at_0.3")

        print(
            f"{display:<10} {n:>5} {fmt(nugget):>10} {fmt(sill_fit):>10} "
            f"{fmt(a):>10} {fmt(rho_01):>10} {fmt(rho_03):>10}"
        )
    print(sep)


def main():
    p = argparse.ArgumentParser(description="Plot landscape variogram and autocorrelation.")
    p.add_argument("--dir", default=DEFAULT_DIR, help="Landscape data directory")
    p.add_argument("--tasks", nargs="+", default=["formula", "finer", "appworld"])
    args = p.parse_args()

    data_by_task = {}
    for task in args.tasks:
        d = load_task_data(args.dir, task)
        if d is not None:
            data_by_task[task] = d
        else:
            print(f"Warning: no autocorrelation data for {task}")

    if not data_by_task:
        print("No data found. Run embed_prompts.py and compute_autocorrelation.py first.")
        return

    out_path = os.path.join(args.dir, "autocorrelation_overlay.png")
    plot_landscape(data_by_task, out_path)
    print_summary_table(data_by_task, args.tasks)


if __name__ == "__main__":
    main()
