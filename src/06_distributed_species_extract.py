# -*- coding: utf-8 -*-
"""
06_distributed_species_extract.py

Extracts the Distributed species from the membership table produced by
03_cluster_analysis.py and ranks them by a clinical-characterization proxy
(the number of registered WHO/IUIS reference allergens per species), to
surface Distributed targets with sparse clinical data.

  * Does not re-run NMF; reads the existing Excel (2_Species_Membership).
  * Clinical-characterization proxy = registered WHO/IUIS reference allergens.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from config import (
    DB_PATH, RESULTS_DIR,
    K, norm_name, Logger,
)

# ===========================================================================
# 0. Constants
# ===========================================================================
K_DEFAULT = K
MEMBERSHIP_SHEET = "2_Species_Membership"
EXPECTED_DISTRIBUTED = 69

# Recognize both labels so older outputs remain readable.
DISTRIBUTED_LABELS = ("Distributed", "Ambiguous")

# Seed-storage archetype index (confirm against 1_Cluster_Signatures)
SEED_STORAGE_ARCHETYPE = 6

log = Logger()


# ===========================================================================
# 1. Load membership + filter Distributed
# ===========================================================================
def load_membership(k_val):
    xlsx = RESULTS_DIR / f"NMF_Final_Analysis_K{k_val}_Step3_Advanced.xlsx"
    if not xlsx.exists():
        log(f"[ERROR] membership file not found: {xlsx}")
        log("        Run 03_cluster_analysis.py first.")
        sys.exit(1)

    df = pd.read_excel(xlsx, sheet_name=MEMBERSHIP_SHEET)
    log(f"[OK] membership loaded: {xlsx.name}  (total {len(df)} species)")

    if "Membership_Status" not in df.columns:
        log(f"[ERROR] 'Membership_Status' column missing. "
            f"columns: {list(df.columns)}")
        sys.exit(1)

    dist = df[df["Membership_Status"].isin(DISTRIBUTED_LABELS)].copy()
    log(f"[OK] Distributed filter: {len(dist)} species")
    if len(dist) != EXPECTED_DISTRIBUTED:
        log(f"[WARN] Distributed count ({len(dist)}) differs from the "
            f"manuscript value ({EXPECTED_DISTRIBUTED}).")
    return dist


# ===========================================================================
# 2. Clinical-characterization proxy: reference allergens per species
# ===========================================================================
def get_ref_allergen_counts():
    if not DB_PATH.exists():
        log(f"[WARN] DB missing: {DB_PATH}  -> proceeding without the proxy.")
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            ref = pd.read_sql_query(
                "SELECT species AS Source_Species, "
                "COUNT(DISTINCT rowid) AS n_ref_allergens "
                "FROM Allergens GROUP BY species", conn)
    except Exception as e:
        log(f"[WARN] DB query failed: {e}  -> proceeding without the proxy")
        return None

    ref["key"] = ref["Source_Species"].apply(norm_name)
    log(f"[OK] per-species reference allergen counts: {len(ref)} species")
    return ref.drop_duplicates("key").set_index("key")["n_ref_allergens"]


# ===========================================================================
# 3. Merge / sort / save
# ===========================================================================
def main(k_val):
    dist = load_membership(k_val)
    dist["key"] = dist["Display_Name"].apply(norm_name)

    ref_counts = get_ref_allergen_counts()
    if ref_counts is not None:
        dist["n_ref_allergens"] = dist["key"].map(ref_counts)
        n_miss = int(dist["n_ref_allergens"].isna().sum())
        if n_miss:
            log(f"[WARN] reference-allergen count unmatched for {n_miss} species: "
                f"{dist.loc[dist['n_ref_allergens'].isna(), 'Display_Name'].tolist()}")
    else:
        dist["n_ref_allergens"] = np.nan

    # Flag species that blend with the seed-storage archetype.
    arch_cols = [c for c in ["Archetype", "Secondary_Archetype"]
                 if c in dist.columns]
    if arch_cols:
        dist["blends_seed_storage"] = (
            dist[arch_cols].eq(SEED_STORAGE_ARCHETYPE).any(axis=1))
    else:
        dist["blends_seed_storage"] = np.nan

    want = ["Display_Name", "Order", "Family", "Archetype",
            "Secondary_Archetype", "Relative_Abundance",
            "Dominance_Ratio", "n_ref_allergens", "blends_seed_storage"]
    cols = [c for c in want if c in dist.columns]
    out = dist[cols].copy()

    sort_keys, sort_asc = [], []
    if "n_ref_allergens" in out.columns:
        sort_keys.append("n_ref_allergens"); sort_asc.append(True)
    if "Dominance_Ratio" in out.columns:
        sort_keys.append("Dominance_Ratio"); sort_asc.append(True)
    if sort_keys:
        out = out.sort_values(by=sort_keys, ascending=sort_asc,
                              na_position="last")

    out_csv = RESULTS_DIR / f"distributed_species_K{k_val}_candidates.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"\n[OK] Distributed candidate list -> {out_csv}")

    log("\n[Target-1 candidates] top 12 Distributed species with sparse clinical data")
    preview = out.head(12)
    for _, r in preview.iterrows():
        nref = r.get("n_ref_allergens", np.nan)
        nref_s = "NA" if pd.isna(nref) else f"{int(nref)}"
        dr = r.get("Dominance_Ratio", np.nan)
        dr_s = "NA" if pd.isna(dr) else f"{dr:.2f}"
        ss = r.get("blends_seed_storage", np.nan)
        ss_s = "  <seed-storage blend>" if ss is True else ""
        log(f"    - {r['Display_Name']:<45} ref={nref_s:>3}  "
            f"DR={dr_s:>6}{ss_s}")

    if out["blends_seed_storage"].any(skipna=True):
        n_ss = int(out["blends_seed_storage"].sum())
        log(f"\n[note] Distributed species blending with the seed-storage "
            f"archetype (A{SEED_STORAGE_ARCHETYPE}): {n_ss}")

    log.save(RESULTS_DIR / f"distributed_extract_log_K{k_val}.txt")
    log(f"[OK] log -> distributed_extract_log_K{k_val}.txt")


# ===========================================================================
if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else K_DEFAULT
    log(f"[RUN] K={k} Distributed species extraction")
    main(k)
