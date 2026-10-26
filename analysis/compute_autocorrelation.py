#!/usr/bin/env python3
"""Compute prompt landscape ruggedness via the empirical semivariogram.

Following the geostatistics literature (Cressie 1993; Journel & Huijbregts
1978) and the fitness-landscape autocorrelation framework (Weinberger 1990):

  1. Full pairwise cosine distance matrix  (N×N)
  2. Quantile-bin all pairs by distance  (B bins, equal pair counts)
  3. Per-bin semivariance:  γ(d) = (1/2|P_d|) Σ_{(i,j)∈P_d} (acc_i − acc_j)²
  4. Fit exponential variogram model:  γ(d) = c₀ + c₁(1 − exp(−d/a))
     - c₀ = nugget  (micro-scale variance / noise)
     - c₁ = partial sill
     - a  = range  (correlation length)
  5. Derive autocorrelation from model:  ρ(d) = exp(−d/a)
     This is always in [0, 1], matching the Weinberger AR(1) convention.
  6. Bootstrap 95 % CIs on γ(d) via within-bin pair resampling.

Saves ``landscape/<task>_autocorrelation.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "landscape")


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return full N×N cosine distance matrix (1 − cos similarity)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def compute_variogram(
    distances: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Quantile-bin pairs by distance; compute empirical semivariance per bin."""
    N = len(accuracies)
    mu = float(np.mean(accuracies))
    var = float(np.var(accuracies, ddof=0))
    if var < 1e-15:
        raise ValueError("Zero variance in accuracies — cannot compute variogram")

    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_dists = distances[triu_i, triu_j]
    sq_diffs = (accuracies[triu_i] - accuracies[triu_j]) ** 2

    d_min, d_max = float(pair_dists.min()), float(pair_dists.max())

    order = np.argsort(pair_dists)
    sorted_dists = pair_dists[order]
    sorted_sq = sq_diffs[order]

    n_pairs = len(sorted_dists)
    bin_size = n_pairs // n_bins

    bin_centres = []
    gamma_values = []
    bin_counts = []

    for b in range(n_bins):
        lo_idx = b * bin_size
        hi_idx = (b + 1) * bin_size if b < n_bins - 1 else n_pairs
        chunk_d = sorted_dists[lo_idx:hi_idx]
        chunk_sq = sorted_sq[lo_idx:hi_idx]

        gamma = float(np.mean(chunk_sq)) / 2.0
        bin_centres.append(float(np.median(chunk_d)))
        gamma_values.append(gamma)
        bin_counts.append(len(chunk_d))

    return {
        "bin_centres": bin_centres,
        "gamma": gamma_values,
        "bin_counts": bin_counts,
        "d_min": d_min,
        "d_max": d_max,
        "global_mean_accuracy": mu,
        "global_var_accuracy": var,
        "n_prompts": N,
        "n_pairs": n_pairs,
    }


def bootstrap_variogram(
    embeddings: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Empirical variogram with 95% CIs via within-bin pair bootstrap."""
    distances = pairwise_cosine_distances(embeddings)
    N = len(accuracies)

    base = compute_variogram(distances, accuracies, n_bins)

    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_dists = distances[triu_i, triu_j]
    sq_diffs = (accuracies[triu_i] - accuracies[triu_j]) ** 2

    order = np.argsort(pair_dists)
    sorted_sq = sq_diffs[order]

    n_pairs = len(sorted_sq)
    bin_size = n_pairs // n_bins
    n_actual_bins = len(base["bin_centres"])

    rng = np.random.RandomState(seed)
    boot_matrix = np.empty((n_bootstrap, n_actual_bins))

    for b in range(n_actual_bins):
        lo_idx = b * bin_size
        hi_idx = (b + 1) * bin_size if b < n_bins - 1 else n_pairs
        chunk = sorted_sq[lo_idx:hi_idx]
        n_chunk = len(chunk)
        samples = rng.choice(chunk, size=(n_bootstrap, n_chunk), replace=True)
        boot_matrix[:, b] = samples.mean(axis=1) / 2.0

    base["ci_lo"] = np.percentile(boot_matrix, 2.5, axis=0).tolist()
    base["ci_hi"] = np.percentile(boot_matrix, 97.5, axis=0).tolist()
    return base


def fit_variogram_model(bin_centres: list[float], gamma: list[float], sill: float) -> dict:
    """Fit exponential variogram model: γ(d) = c₀ + c₁(1 − exp(−d/a)).

    Returns nugget c₀, partial sill c₁, range a, and the model-derived
    autocorrelation ρ(d) = exp(−d/a) at key distances.
    """
    x = np.array(bin_centres)
    y = np.array(gamma)

    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 3:
        return {"nugget": None, "partial_sill": None, "range": None, "fit_success": False}

    xv, yv = x[valid], y[valid]

    def exp_variogram(d, c0, c1, a):
        return c0 + c1 * (1.0 - np.exp(-d / a))

    try:
        p0 = [yv[0], sill - yv[0], xv[-1] / 3.0]
        bounds = ([0, 0, 1e-6], [sill, sill * 3, np.inf])
        popt, _ = curve_fit(exp_variogram, xv, yv, p0=p0, bounds=bounds, maxfev=10000)
        c0, c1, a = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        return {"nugget": None, "partial_sill": None, "range": None, "fit_success": False}

    rho_model = [float(np.exp(-d / a)) for d in bin_centres]

    return {
        "nugget": c0,
        "partial_sill": c1,
        "range": a,
        "sill": c0 + c1,
        "fit_success": True,
        "rho_model": rho_model,
        "rho_at_0.1": float(np.exp(-0.1 / a)),
        "rho_at_0.3": float(np.exp(-0.3 / a)),
    }


def main():
    p = argparse.ArgumentParser(description="Compute landscape variogram and autocorrelation.")
    p.add_argument("--dir", default=DEFAULT_DIR, help="Landscape data directory")
    p.add_argument("--tasks", nargs="+", default=["formula", "finer", "appworld"])
    p.add_argument("--bins", type=int, default=15, help="Number of distance bins (quantile-based)")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    args = p.parse_args()

    for task in args.tasks:
        print(f"\n{'=' * 50}")
        print(f"Task: {task}")
        print(f"{'=' * 50}")

        json_path = os.path.join(args.dir, f"{task}_prompts.json")
        npy_path = os.path.join(args.dir, f"{task}_embeddings.npy")

        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            print("  Missing data files, skipping (run embed_prompts.py first)")
            continue

        with open(json_path) as f:
            prompts = json.load(f)
        embeddings = np.load(npy_path)
        accuracies = np.array([pr["accuracy"] for pr in prompts], dtype=np.float64)
        sill = float(np.var(accuracies, ddof=0))

        print(f"  Prompts: {len(prompts)}")
        print(f"  Accuracy range: [{accuracies.min():.4f}, {accuracies.max():.4f}]")
        print(f"  Accuracy mean:  {accuracies.mean():.4f}, std: {accuracies.std():.4f}")
        print(f"  Sill (σ²):      {sill:.6f}")

        print(f"  Computing variogram with {args.bins} bins and {args.bootstrap} bootstrap resamples ...")
        result = bootstrap_variogram(
            embeddings,
            accuracies,
            n_bins=args.bins,
            n_bootstrap=args.bootstrap,
        )

        fit = fit_variogram_model(result["bin_centres"], result["gamma"], sill)
        result.update(fit)

        print(f"  Distance range: [{result['d_min']:.4f}, {result['d_max']:.4f}]")
        if fit["fit_success"]:
            print(f"  Nugget (c₀):    {fit['nugget']:.6f}")
            print(f"  Partial sill:   {fit['partial_sill']:.6f}")
            print(f"  Range (a):      {fit['range']:.4f}")
            print(f"  ρ(0.1):         {fit['rho_at_0.1']:.4f}")
            print(f"  ρ(0.3):         {fit['rho_at_0.3']:.4f}")
        else:
            print("  Variogram model fit failed")

        out_path = os.path.join(args.dir, f"{task}_autocorrelation.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
