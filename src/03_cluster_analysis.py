# -*- coding: utf-8 -*-
"""
Fit NMF at a chosen K and export per-species archetype membership.

Produces an Excel report with two sheets: archetype signatures (top features
of each H component) and species membership (primary/secondary archetype,
weights, and Core/Ambiguous classification).

Classification follows the single manuscript criterion: a species is a Core
Member when the relative abundance of its primary archetype is >= 0.80.

The fitted W and H matrices are saved as .npy files so that downstream
scripts (04, 05) can load them without refitting.

Usage:
    python 03_cluster_analysis.py            # default K = 6
    python 03_cluster_analysis.py 6 7        # or pass specific K values
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. Paths / constants (config.py)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DB_PATH, MATRIX_PATH, RESULTS_DIR,
    NMF_W_PATH, NMF_H_PATH, NMF_MEMBERSHIP_PATH,
    K, RANDOM_STATE, MAX_ITER, CORE_RA_THRESHOLD,
    norm_name,
)

IN_MATRIX = MATRIX_PATH

_log = []


def log(m=""):
    print(m)
    _log.append(str(m))


# ---------------------------------------------------------------------------
# 1. Load and preprocess
# ---------------------------------------------------------------------------
def load_final_matrix(file_path):
    """log1p transform followed by per-species (row) L1 normalization."""
    path = Path(file_path)
    if not path.exists():
        log(f"[ERROR] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, index_col=0)

    zero_rows = df.index[df.sum(axis=1) == 0].tolist()
    if zero_rows:
        log(f"[WARN] {len(zero_rows)} species have no assignable conserved "
            f"domain: {zero_rows}")
        log("       Their W rows are zero and they fall into Ambiguous.")

    matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(df))
    log(f"[OK] Ready: {df.shape[0]} species x {df.shape[1]} families")
    return matrix_norm, df.columns, df.index


# ---------------------------------------------------------------------------
# 2. Taxonomy mapping (SQLite)
# ---------------------------------------------------------------------------
def get_taxonomy_info(db_path, species_list):
    import sqlite3

    base = pd.DataFrame({"Display_Name": list(species_list)})
    base["key"] = base["Display_Name"].apply(norm_name)

    try:
        with sqlite3.connect(db_path) as conn:
            tax = pd.read_sql_query(
                "SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy", conn)
    except Exception as e:
        log(f"[WARN] DB mapping error: {e}")
        base["Order"] = np.nan
        base["Family"] = np.nan
        return base.drop(columns=["key"])

    tax["key"] = tax["Species"].apply(norm_name)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
    map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

    base["Order"] = base["key"].map(map_order)
    base["Family"] = base["key"].map(map_family)

    n_miss = int(base["Order"].isna().sum())
    log(f"[OK] Taxonomy mapping: {n_miss} species without an Order match")
    for nm in base.loc[base["Order"].isna(), "Display_Name"]:
        log(f"      unmatched: {nm}")

    return base.drop(columns=["key"])


# ---------------------------------------------------------------------------
# 3. Analysis and report
# ---------------------------------------------------------------------------
def generate_report(matrix, feature_names, species_names, db_path, k_val):
    log(f"\n{'=' * 62}")
    log(f"[RUN] K={k_val} detailed analysis")
    log("=" * 62)

    model = NMF(n_components=k_val, init="nndsvda",
                max_iter=MAX_ITER, random_state=RANDOM_STATE)
    W = model.fit_transform(matrix)      # species x archetype
    H = model.components_                # archetype x feature
    log(f"[OK] NMF K={k_val}: W{W.shape}, H{H.shape}")

    # --- persist W, H for downstream scripts (04, 05) ---------------------
    if k_val == K:
        np.save(NMF_W_PATH, W)
        np.save(NMF_H_PATH, H)
        log(f"[OK] Saved W -> {NMF_W_PATH}")
        log(f"[OK] Saved H -> {NMF_H_PATH}")

    # --- primary / secondary archetype -------------------------------------
    sorted_idx = np.argsort(W, axis=1)[:, ::-1]
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1]
    top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).ravel()
    top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).ravel()

    # --- metrics -----------------------------------------------------------
    eps = 1e-9
    dominance_ratio = top1_w / (top2_w + eps)
    row_sums = W.sum(axis=1)
    relative_abundance = top1_w / np.where(row_sums == 0, 1, row_sums)

    # --- classification: single criterion ----------------------------------
    is_core = relative_abundance >= CORE_RA_THRESHOLD
    membership = np.where(is_core, "Core Member", "Ambiguous")

    # Sanity check: RA >= 0.80 gives the same partition as the alternative
    # dominance-based rule, because RA >= 0.80 forces a dominance ratio >= 4.
    alt = (dominance_ratio >= 2.0) & ((top1_w >= 0.3) | (relative_abundance >= 0.8))
    n_diff = int((is_core != alt).sum())
    log(f"[CHECK] Single criterion (RA>={CORE_RA_THRESHOLD}) vs alternative "
        f"dominance rule - disagreeing species: {n_diff}")
    if n_diff == 0 and is_core.any():
        log(f"        Identical partition. Minimum dominance ratio among Core "
            f"species = {dominance_ratio[is_core].min():.3f} (>= 4 as expected)")
    elif n_diff:
        log("[WARN] The two rules disagree; re-check the criterion.")

    log(f"[RESULT] {len(species_names)} species / "
        f"Core Member {int(is_core.sum())} / Ambiguous {int((~is_core).sum())}")
    per_arch = (pd.Series(top1_idx[is_core] + 1)
                .value_counts()
                .reindex(range(1, k_val + 1), fill_value=0)
                .tolist())
    log(f"       Core count per archetype 1..{k_val}: {per_arch}")

    # --- Sheet 1: archetype signatures (top features of H) -----------------
    sig_rows = []
    for k in range(k_val):
        for rank, idx in enumerate(np.argsort(H[k])[::-1][:10], 1):
            sig_rows.append({
                "Archetype": k + 1,
                "Cluster": k,                       # 0-based internal index
                "Rank": rank,
                "Feature (Proteins)": feature_names[idx],
                "Weight_H": H[k, idx],              # feature loadings are in H
            })
    df_sig = pd.DataFrame(sig_rows)

    # --- Sheet 2: membership and classification ----------------------------
    df_tax = get_taxonomy_info(db_path, species_names)
    df_tax["Primary_Cluster"] = top1_idx
    df_tax["Archetype"] = top1_idx + 1
    df_tax["Secondary_Cluster"] = top2_idx
    df_tax["Secondary_Archetype"] = top2_idx + 1
    df_tax["Top1_Weight"] = top1_w
    df_tax["Top2_Weight"] = top2_w
    df_tax["Relative_Abundance"] = relative_abundance
    df_tax["Dominance_Ratio"] = dominance_ratio
    df_tax["Membership_Status"] = membership

    for i in range(k_val):
        df_tax[f"Weight_A{i + 1}"] = W[:, i]        # 1-based archetype columns

    # Sort: archetype -> Core first -> dominance ratio descending
    df_tax["_order"] = np.where(df_tax["Membership_Status"] == "Core Member", 0, 1)
    df_tax = (df_tax
              .sort_values(by=["Archetype", "_order", "Dominance_Ratio"],
                           ascending=[True, True, False])
              .drop(columns=["_order"]))

    # --- persist membership CSV for downstream scripts (04, 05) -----------
    if k_val == K:
        df_tax.to_csv(NMF_MEMBERSHIP_PATH, index=False, encoding="utf-8-sig")
        log(f"[OK] Membership CSV -> {NMF_MEMBERSHIP_PATH}")

    out_xlsx = RESULTS_DIR / f"NMF_Final_Analysis_K{k_val}_Step3_Advanced.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_sig.to_excel(writer, sheet_name="1_Cluster_Signatures", index=False)
        df_tax.to_excel(writer, sheet_name="2_Species_Membership", index=False)
    log(f"[OK] Report saved -> {out_xlsx}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit NMF at the given K and export membership reports.")
    parser.add_argument("k", type=int, nargs="*", default=[6],
                        help="one or more K values (default: 6)")
    args = parser.parse_args()
    target_ks = args.k if args.k else [6]

    matrix_norm, features, species = load_final_matrix(IN_MATRIX)
    log(f"[RUN] target K: {target_ks}")

    for k in target_ks:
        generate_report(matrix_norm, features, species, DB_PATH, k)

    (RESULTS_DIR / "cluster_analysis_log.txt").write_text(
        "\n".join(_log), encoding="utf-8")
    log(f"\n[DONE] Log -> {RESULTS_DIR / 'cluster_analysis_log.txt'}")
    log("[NEXT] 04_figure_generation.py (figures) -> "
        "05_statistical_analysis.py (statistics)")
