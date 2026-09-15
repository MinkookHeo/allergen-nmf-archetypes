# -*- coding: utf-8 -*-
"""
06_ambiguous_species_extract.py

Extracts the Ambiguous species from the membership table written by 03 and
ranks them by how thinly they are characterized, to highlight candidates for
targeted experimental follow-up.

The proxy for characterization depth is the number of WHO/IUIS reference
allergens registered for each species. NMF is not refitted; the existing
membership workbook is read as is.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

# --- allow "from config import ..." when run from src/ -------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------
from config import (
    DB_PATH, RESULTS_DIR,
    K, norm_name, Logger,
)

# ===========================================================================
# 0. Constants
# ===========================================================================
K_DEFAULT = K
MEMBERSHIP_SHEET = "2_Species_Membership"
EXPECTED_AMBIGUOUS = 69

# Seed storage archetype index (see the 1_Cluster_Signatures sheet)
SEED_STORAGE_ARCHETYPE = 6

log = Logger()


# ===========================================================================
# 1. Load membership and filter Ambiguous
# ===========================================================================
def load_membership(k_val):
    xlsx = RESULTS_DIR / f"NMF_Final_Analysis_K{k_val}_Step3_Advanced.xlsx"
    if not xlsx.exists():
        log(f"[ERROR] Membership workbook not found: {xlsx}")
        log("        Run 03_cluster_analysis.py first.")
        sys.exit(1)

    df = pd.read_excel(xlsx, sheet_name=MEMBERSHIP_SHEET)
    log(f"[OK] Membership loaded: {xlsx.name}  ({len(df)} species)")

    if "Membership_Status" not in df.columns:
        log(f"[ERROR] Column 'Membership_Status' is missing. "
            f"Columns: {list(df.columns)}")
        sys.exit(1)

    amb = df[df["Membership_Status"] == "Ambiguous"].copy()
    log(f"[OK] Ambiguous species: {len(amb)}")
    if len(amb) != EXPECTED_AMBIGUOUS:
        log(f"[WARN] Ambiguous count ({len(amb)}) differs from the reported "
            f"value ({EXPECTED_AMBIGUOUS}).")
    return amb


# ===========================================================================
# 2. Characterization proxy: WHO/IUIS reference allergens per species
# ===========================================================================
def get_ref_allergen_counts():
    if not DB_PATH.exists():
        log(f"[WARN] Database not found: {DB_PATH}  -> listing without the proxy")
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            ref = pd.read_sql_query(
                "SELECT species AS Source_Species, "
                "COUNT(DISTINCT rowid) AS n_ref_allergens "
                "FROM Allergens GROUP BY species", conn)
    except Exception as e:
        log(f"[WARN] Database query failed: {e}  -> continuing without the proxy")
        return None

    ref["key"] = ref["Source_Species"].apply(norm_name)
    log(f"[OK] Reference allergen counts collected for {len(ref)} species")
    return ref.drop_duplicates("key").set_index("key")["n_ref_allergens"]


# ===========================================================================
# 3. Merge, sort, save
# ===========================================================================
def main(k_val):
    amb = load_membership(k_val)
    amb["key"] = amb["Display_Name"].apply(norm_name)

    ref_counts = get_ref_allergen_counts()
    if ref_counts is not None:
        amb["n_ref_allergens"] = amb["key"].map(ref_counts)
        n_miss = int(amb["n_ref_allergens"].isna().sum())
        if n_miss:
            log(f"[WARN] {n_miss} species without a reference allergen count: "
                f"{amb.loc[amb['n_ref_allergens'].isna(), 'Display_Name'].tolist()}")
    else:
        amb["n_ref_allergens"] = np.nan

    # Flag Ambiguous species that blend with the seed storage archetype
    arch_cols = [c for c in ["Archetype", "Secondary_Archetype"]
                 if c in amb.columns]
    if arch_cols:
        amb["blends_seed_storage"] = (
            amb[arch_cols].eq(SEED_STORAGE_ARCHETYPE).any(axis=1))
    else:
        amb["blends_seed_storage"] = np.nan

    want = ["Display_Name", "Order", "Family", "Archetype",
            "Secondary_Archetype", "Relative_Abundance",
            "Dominance_Ratio", "n_ref_allergens", "blends_seed_storage"]
    cols = [c for c in want if c in amb.columns]
    out = amb[cols].copy()

    sort_keys, sort_asc = [], []
    if "n_ref_allergens" in out.columns:
        sort_keys.append("n_ref_allergens"); sort_asc.append(True)
    if "Dominance_Ratio" in out.columns:
        sort_keys.append("Dominance_Ratio"); sort_asc.append(True)
    if sort_keys:
        out = out.sort_values(by=sort_keys, ascending=sort_asc,
                              na_position="last")

    out_csv = RESULTS_DIR / f"ambiguous_species_K{k_val}_candidates.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"\n[OK] Candidate list saved -> {out_csv}")

    log("\n[Top 12] Ambiguous species with the thinnest reference coverage")
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
        log(f"\n[NOTE] Ambiguous species blending with the seed storage "
            f"archetype (A{SEED_STORAGE_ARCHETYPE}): {n_ss}")

    log.save(RESULTS_DIR / f"ambiguous_extract_log_K{k_val}.txt")
    log(f"[OK] log -> ambiguous_extract_log_K{k_val}.txt")


# ===========================================================================
if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else K_DEFAULT
    log(f"[RUN] Extracting Ambiguous species at K={k}")
    main(k)
