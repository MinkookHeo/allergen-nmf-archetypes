# -*- coding: utf-8 -*-
"""
03_cluster_analysis.py

Applies NMF (K) to allergen_source_matrix.csv and writes the per-species
archetype membership as an Excel report.

The NMF results (W, H) are saved as .npy so that downstream scripts (04, 05)
can load them without recomputation.

Dependencies: pandas numpy scikit-learn openpyxl
"""

import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

from config import (
    DB_PATH, MATRIX_PATH, RESULTS_DIR,
    NMF_W_PATH, NMF_H_PATH, NMF_MEMBERSHIP_PATH,
    K, RANDOM_STATE, MAX_ITER, CORE_RA_THRESHOLD,
    TAXONOMY_CORRECTIONS, MANUAL_ORDERS,
    norm_name, Logger,
)

# ===========================================================================
# 0. Setup
# ===========================================================================
log = Logger()


# ===========================================================================
# 1. Load and preprocess
# ===========================================================================
def load_final_matrix(file_path):
    """log1p transform, then row-wise (species) L1 normalization."""
    if not file_path.exists():
        log(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path, index_col=0)

    zero_rows = df.index[df.sum(axis=1) == 0].tolist()
    if zero_rows:
        log(f"[WARN] {len(zero_rows)} species have no assignable conserved "
            f"domain: {zero_rows}")
        log("       Their W rows are 0 and are therefore classified as Distributed.")

    matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(df))
    log(f"[OK] ready: {df.shape[0]} species x {df.shape[1]} families")
    return matrix_norm, df.columns, df.index


# ===========================================================================
# 2. SQLite taxonomy mapping
# ===========================================================================
def get_taxonomy_info(species_list):
    base = pd.DataFrame({"Display_Name": list(species_list)})
    base["key"] = base["Display_Name"].apply(norm_name)
    base["key"] = base["key"].replace(TAXONOMY_CORRECTIONS)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            tax = pd.read_sql_query(
                "SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy", conn)
    except Exception as e:
        log(f"[WARN] DB mapping error: {e}")
        base["Order"] = base["key"].map(MANUAL_ORDERS)
        base["Family"] = np.nan
        return base.drop(columns=["key"])

    tax["key"] = tax["Species"].apply(norm_name)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
    map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

    base["Order"] = base["key"].map(map_order)
    base["Family"] = base["key"].map(map_family)

    # Fill species unresolved by the DB with the manual overrides.
    n_before = int(base["Order"].isna().sum())
    base["Order"] = base["Order"].fillna(base["key"].map(MANUAL_ORDERS))
    n_manual = n_before - int(base["Order"].isna().sum())
    if n_manual:
        log(f"[OK] filled by MANUAL_ORDERS: {n_manual}")

    n_miss = int(base["Order"].isna().sum())
    log(f"[OK] taxonomy mapping: {n_miss} species without Order")
    for nm in base.loc[base["Order"].isna(), "Display_Name"]:
        log(f"      unmatched: {nm}")

    return base.drop(columns=["key"])


# ===========================================================================
# 3. Analysis and report
# ===========================================================================
def generate_report(matrix, feature_names, species_names, k_val):
    log(f"\n{'=' * 62}")
    log(f"[RUN] K={k_val} analysis")
    log("=" * 62)

    model = NMF(n_components=k_val, init="nndsvda",
                max_iter=MAX_ITER, random_state=RANDOM_STATE)
    W = model.fit_transform(matrix)      # species x archetype
    H = model.components_                # archetype x feature
    log(f"[OK] NMF K={k_val}: W{W.shape}, H{H.shape}")

    # Save NMF results (loaded by downstream scripts)
    np.save(NMF_W_PATH, W)
    np.save(NMF_H_PATH, H)
    log(f"[OK] NMF results saved -> {NMF_W_PATH.name}, {NMF_H_PATH.name}")

    # --- primary / secondary archetype ------------------------------------
    sorted_idx = np.argsort(W, axis=1)[:, ::-1]
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1]
    top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).ravel()
    top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).ravel()

    # --- metrics ----------------------------------------------------------
    eps = 1e-9
    dominance_ratio = top1_w / (top2_w + eps)
    row_sums = W.sum(axis=1)
    relative_abundance = top1_w / np.where(row_sums == 0, 1, row_sums)

    # --- classification: single criterion (manuscript 2.4) ----------------
    is_core = relative_abundance >= CORE_RA_THRESHOLD
    membership = np.where(is_core, "Core Member", "Distributed")

    # Self-check against a previous hybrid rule
    hybrid = (dominance_ratio >= 2.0) & ((top1_w >= 0.3) | (relative_abundance >= 0.8))
    n_diff = int((is_core != hybrid).sum())
    log(f"[CHECK] single rule (RA>={CORE_RA_THRESHOLD}) vs previous hybrid rule "
        f"disagreements: {n_diff}")
    if n_diff == 0 and is_core.any():
        log(f"        The two rules give the identical split. "
            f"Min dominance ratio among Core = {dominance_ratio[is_core].min():.3f} "
            f"(>= 4 matches the theory)")
    elif n_diff:
        log("[WARN] The two rules disagree. Re-check manuscript 2.4.")

    log(f"[RESULT] total {len(species_names)} species / "
        f"Core Member {int(is_core.sum())} / Distributed {int((~is_core).sum())}")
    per_arch = (pd.Series(top1_idx[is_core] + 1)
                .value_counts()
                .reindex(range(1, k_val + 1), fill_value=0)
                .tolist())
    log(f"       Core per archetype 1..{k_val}: {per_arch}")

    # --- Sheet 1: archetype signatures (top H features) -------------------
    sig_rows = []
    for k in range(k_val):
        for rank, idx in enumerate(np.argsort(H[k])[::-1][:10], 1):
            sig_rows.append({
                "Archetype": k + 1,
                "Cluster": k,
                "Rank": rank,
                "Feature (Proteins)": feature_names[idx],
                "Weight_H": H[k, idx],
            })
    df_sig = pd.DataFrame(sig_rows)

    # --- Sheet 2: membership and classification ---------------------------
    df_tax = get_taxonomy_info(species_names)
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
        df_tax[f"Weight_A{i + 1}"] = W[:, i]

    # Sort: archetype -> Core first -> Dominance Ratio descending
    df_tax["_order"] = np.where(df_tax["Membership_Status"] == "Core Member", 0, 1)
    df_tax = (df_tax
              .sort_values(by=["Archetype", "_order", "Dominance_Ratio"],
                           ascending=[True, True, False])
              .drop(columns=["_order"]))

    out_xlsx = RESULTS_DIR / f"NMF_Final_Analysis_K{k_val}_Step3_Advanced.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_sig.to_excel(writer, sheet_name="1_Cluster_Signatures", index=False)
        df_tax.to_excel(writer, sheet_name="2_Species_Membership", index=False)
    log(f"[OK] report saved -> {out_xlsx}")

    # Membership CSV (nmf_membership.csv; fallback/reference for 08)
    df_tax.to_csv(NMF_MEMBERSHIP_PATH, index=False, encoding="utf-8-sig")
    log(f"[OK] membership CSV saved -> {NMF_MEMBERSHIP_PATH.name}")


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    matrix_norm, features, species = load_final_matrix(MATRIX_PATH)

    # K: CLI argument if given, else default K=6
    if len(sys.argv) > 1:
        target_ks = [int(v) for v in sys.argv[1:]]
    else:
        target_ks = [K]
    log(f"[RUN] target K: {target_ks}")

    for k in target_ks:
        generate_report(matrix_norm, features, species, k)

    log.save(RESULTS_DIR / "cluster_analysis_log.txt")
    log(f"\n[DONE] log -> {RESULTS_DIR / 'cluster_analysis_log.txt'}")
    log("[NEXT] 04_figure_generation.py -> 05_statistical_analysis.py")
