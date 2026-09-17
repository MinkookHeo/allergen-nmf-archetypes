# -*- coding: utf-8 -*-
"""
02_nmf_k_selection.py

Runs NMF consensus clustering on allergen_source_matrix.csv and computes the
K-selection metrics (cophenetic correlation / RSS / delta area) and consensus
matrices.

Output: manuscript Figure 1 (3x3: 3 metric panels + 6 consensus heatmaps).
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

from config import (
    MATRIX_PATH, RESULTS_DIR, CACHE_DIR,
    set_journal_font,
)

# ===========================================================================
# 0. Paths / setup
# ===========================================================================
set_journal_font()

IN_MATRIX = MATRIX_PATH
OUT_METRICS = RESULTS_DIR / "NMF_K_selection_metrics.csv"

# Manuscript Figure 1
OUT_FIG_COMBINED = RESULTS_DIR / "Figure_1.png"
OUT_FIG_PDF = RESULTS_DIR / "Figure_1.pdf"

K_RANGE = range(2, 16)
N_RUNS = 30

NMF_KWARGS = dict(
    init="random",
    solver="cd",
    beta_loss="frobenius",
    tol=1e-4,
    max_iter=2000,
)


# ===========================================================================
# 1. Load and preprocess
# ===========================================================================
def load_processed_data(file_path):
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path, index_col=0)
    df_log = np.log1p(df)
    matrix_norm = Normalizer(norm="l1").fit_transform(df_log)
    return matrix_norm, df.columns, df.index


# ===========================================================================
# 2. Consensus matrix (cached)
#    Cache key includes N_RUNS so a changed run count is not silently reused.
# ===========================================================================
def build_consensus(matrix, k, n_runs=N_RUNS, use_cache=True):
    cache_c = CACHE_DIR / f"consensus_K{k}_n{n_runs}.npy"
    cache_r = CACHE_DIR / f"rss_K{k}_n{n_runs}.npy"

    if use_cache and cache_c.exists() and cache_r.exists():
        return np.load(cache_c), float(np.load(cache_r))

    n = matrix.shape[0]
    consensus = np.zeros((n, n))
    errors = []

    for i in range(n_runs):
        model = NMF(n_components=k, random_state=i, **NMF_KWARGS)
        W = model.fit_transform(matrix)
        errors.append(model.reconstruction_err_ ** 2)

        labels = np.argmax(W, axis=1)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            consensus[np.ix_(idx, idx)] += 1

    consensus /= n_runs
    mean_rss = float(np.mean(errors))

    np.save(cache_c, consensus)
    np.save(cache_r, np.array(mean_rss))
    return consensus, mean_rss


def cdf_area(consensus):
    n = consensus.shape[0]
    values = consensus[np.tril_indices(n, k=-1)]
    hist, _ = np.histogram(values, bins=100, range=(0, 1))
    cdf = np.cumsum(hist) / len(values)
    return float(np.sum(cdf * (1.0 / 100)))


def cophenetic_corr(consensus):
    d = squareform(1 - consensus, checks=False)
    Z = linkage(d, method="average")
    coph, _ = cophenet(Z, d)
    return float(coph), Z


# ===========================================================================
# 3. Run
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()

    matrix_norm, feature_names, species_names = load_processed_data(IN_MATRIX)
    print(f"[OK] input matrix: {matrix_norm.shape[0]} species x {matrix_norm.shape[1]} features")
    print(f"[RUN] K={list(K_RANGE)}, runs={N_RUNS}\n")

    rows = []
    for k in K_RANGE:
        consensus, rss = build_consensus(matrix_norm, k)
        coph, _ = cophenetic_corr(consensus)
        auc = cdf_area(consensus)
        rows.append({"K": k, "cophenetic": coph, "RSS": rss, "CDF_area": auc})
        print(f"  K={k:2d} | Coph={coph:.4f} | RSS={rss:.4f} | AUC={auc:.4f}")

    met = pd.DataFrame(rows)

    # Delta area (relative change in CDF area vs the previous K).
    # The first K has no predecessor, so its delta is undefined (NaN).
    auc = met["CDF_area"].to_numpy()
    delta = np.empty_like(auc)
    delta[0] = np.nan
    delta[1:] = (auc[1:] - auc[:-1]) / auc[:-1]
    met["delta_area"] = delta

    met.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"\n[OK] metrics saved -> {OUT_METRICS}")

    # =======================================================================
    # Combined Figure 1 (3x3 layout)
    # =======================================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    k_list = met["K"]

    # --- Row 1: K-selection metrics (A, B, C) ---
    color_A, color_B, color_C = '#1B9E77', '#D95F02', '#7570B3'

    # (A) Cophenetic correlation
    axes[0, 0].plot(k_list, met["cophenetic"], "o-", color=color_A, lw=2, ms=6)
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Cophenetic correlation coefficient")
    axes[0, 0].text(-0.15, 1.05, "(A)", transform=axes[0, 0].transAxes,
                    fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(list(k_list))

    # (B) RSS
    axes[0, 1].plot(k_list, met["RSS"], "o-", color=color_B, lw=2, ms=6)
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Residual sum of squares (RSS)")
    axes[0, 1].text(-0.15, 1.05, "(B)", transform=axes[0, 1].transAxes,
                    fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(list(k_list))

    # (C) Delta area
    axes[0, 2].plot(k_list, met["delta_area"], "o-", color=color_C, lw=2, ms=6)
    axes[0, 2].set_xlabel("k")
    axes[0, 2].set_ylabel("Delta area under consensus CDF")
    axes[0, 2].text(-0.15, 1.05, "(C)", transform=axes[0, 2].transAxes,
                    fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_xticks(list(k_list))

    sns.despine(ax=axes[0, 0])
    sns.despine(ax=axes[0, 1])
    sns.despine(ax=axes[0, 2])

    # --- Rows 2-3: consensus heatmaps (D) ---
    # CLI argument or default [3, 4, 5, 6, 7, 8]
    if len(sys.argv) > 1:
        target_ks = [int(v) for v in sys.argv[1:]]
    else:
        target_ks = [3, 4, 5, 6, 7, 8]

    heatmap_axes = axes[1:, :].flatten()

    for i, k in enumerate(target_ks):
        if i >= len(heatmap_axes):
            break

        consensus, _ = build_consensus(matrix_norm, k)
        _, Z = cophenetic_corr(consensus)
        order = leaves_list(Z)

        sns.heatmap(
            consensus[np.ix_(order, order)],
            ax=heatmap_axes[i], cmap="YlGnBu", vmin=0, vmax=1,
            cbar=False, xticklabels=False, yticklabels=False,
        )

        if i == 0:
            heatmap_axes[i].text(
                -0.05, 1.05, "(D)", transform=heatmap_axes[i].transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

        heatmap_axes[i].set_title(f"k = {k}", pad=10)

    for j in range(len(target_ks), len(heatmap_axes)):
        heatmap_axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_FIG_COMBINED, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, bbox_inches="tight")
    print(f"\n[OK] combined 3x3 figure saved -> {OUT_FIG_COMBINED} / {OUT_FIG_PDF}")
    plt.close()

    print(f"\n[DONE] elapsed: {time.time() - t0:.1f}s")
