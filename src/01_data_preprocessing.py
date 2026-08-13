# -*- coding: utf-8 -*-
"""
Build the source-species x conserved-domain feature matrix.

Reads the allergen SQLite database, joins reference allergens to their
cross-reactive homologs and CDD conserved-domain annotations, removes
non-food source species by exact genus matching, and writes the
species x domain matrix used by the downstream NMF pipeline.

Output: data/allergen_source_matrix.csv
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths (config.py)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MATRIX_PATH, RESULTS_DIR

OUT_MATRIX = MATRIX_PATH
OUT_REMOVED = RESULTS_DIR / "removed_species.csv"
OUT_SUMMARY = RESULTS_DIR / "matrix_build_summary.txt"

# Optional cross-check against a previous NMF species list (skipped if absent).
PREV_NMF_XLSX = RESULTS_DIR / "NMF_Final_Analysis_K6_Step3_Advanced.xlsx"
PREV_NMF_SHEET = "2_Species_Membership"

N_REFERENCE_ALLERGENS = 1149  # total WHO/IUIS records queried

if not DB_PATH.exists():
    print(f"[ERROR] Database not found. Place it in the data folder: {DB_PATH}")
    sys.exit(1)

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


log(f"[PATH] DB      : {DB_PATH}")
log(f"[PATH] Results : {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# 1. Load and assemble
# ---------------------------------------------------------------------------
# Query_ID is a GenBank accession. A single allergen may hold several
# accessions, so the number of unique Query_ID values is not the allergen
# count. Reference allergens are counted by A.rowid (Allergen_UID).
QUERY = """
SELECT
    A.rowid            AS Allergen_UID,
    A.species          AS Source_Species,
    C.Query_ID         AS Query_ID,
    C.Similar_Protein  AS Similar_Protein,
    CN.common_name     AS common_name,
    P.Superfamily      AS Superfamily,
    P."Short name"     AS Short_name,
    P.Bitscore         AS Bitscore
FROM Allergens A
JOIN Crossreactivity C
    ON ';' || A.genbank_ids || ';' LIKE '%;' || C.Query_ID || ';%'
LEFT JOIN Protein_families P
    ON C.Similar_Protein = P.Query
LEFT JOIN Common_names CN
    ON A.species = CN.scientific_name
"""

try:
    with sqlite3.connect(DB_PATH) as conn:
        merged = pd.read_sql_query(QUERY, conn)
except Exception as e:
    log(f"[ERROR] Failed to load data: {e}")
    sys.exit(1)

log(f"[OK] Data assembled ({len(merged):,} rows)")


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def composite_id(df):
    """CDD composite identifier: Superfamily + Short name."""
    return df["Superfamily"].fillna("-") + "|" + df["Short_name"].fillna("Unknown")


def collect_stats(df):
    """Return the descriptive counts reported in Methods."""
    return {
        "pairs": len(df),
        "homologs": df["Similar_Protein"].nunique(),
        "allergens": df["Allergen_UID"].nunique(),
        "accessions": df["Query_ID"].nunique(),
        "species": df["Source_Species"].nunique(),
        "features": composite_id(df).nunique(),
    }


def report(stats, label):
    log(f"\n--- [{label}] " + "-" * max(4, 40 - len(label)))
    log(f"  allergen-homolog pairs (rows) : {stats['pairs']:,}")
    log(f"  unique homologs               : {stats['homologs']:,}")
    log(f"  reference allergens (unique)  : {stats['allergens']:,}"
        f"  / queried {N_REFERENCE_ALLERGENS:,}")
    log(f"  GenBank accessions recovered  : {stats['accessions']:,}")
    log(f"  source species                : {stats['species']:,}")
    log(f"  conserved-domain features     : {stats['features']:,}")


pre_stats = collect_stats(merged)
report(pre_stats, "before filtering")

if pre_stats["allergens"] > N_REFERENCE_ALLERGENS:
    log(f"[WARN] Unique allergen count ({pre_stats['allergens']:,}) exceeds the "
        f"number queried ({N_REFERENCE_ALLERGENS:,}); check for duplicate rows "
        f"in the Allergens table.")


# ---------------------------------------------------------------------------
# 3. Remove non-food species (exact genus match)
# ---------------------------------------------------------------------------
# Matching on the genus (first token) rather than substrings avoids collateral
# removals, e.g. "Mus" would otherwise catch Musa acuminata (banana).
GENERA_TO_REMOVE = {
    # Bees / wasps / ants (Hymenoptera) - venom allergens
    "Apis", "Vespa", "Vespula", "Bombus", "Polistes", "Polybia", "Solenopsis",
    "Myrmecia", "Dolichovespula", "Anoplolepis", "Linepithema",
    "Pachycondyla",

    # Cockroaches and household pests
    "Blattella", "Periplaneta", "Coptotermes", "Shelfordella", "Supella",

    # Moths / silkworm (remove Bombyx from this set to include silkworm as food)
    "Bombyx", "Plodia", "Ephestia", "Galleria", "Thaumetopoea", "Tineola",

    # Biting / environmental / blood-feeding insects and flies
    "Aedes", "Anopheles", "Culex", "Glossina", "Tabanus", "Musca", "Chironomus",
    "Ctenocephalides", "Cimex", "Triatoma", "Lepisma", "Forcipomyia",
    "Arge", "Lucilia", "Sarcophaga", "Calliphora",

    # Mites and ticks
    "Dermatophagoides", "Tyrophagus", "Euroglyphus", "Glycyphagus",
    "Lepidoglyphus", "Blomia", "Tetranychus", "Argas", "Ixodes", "Acarus",
    "Chortoglyphus", "Sarcoptes",

    # Parasites and nematodes
    "Ascaris", "Anisakis", "Trichuris", "Enterobius", "Strongyloides",
    "Schistosoma",

    # Fungi and yeasts
    "Aspergillus", "Alternaria", "Cladosporium", "Penicillium", "Candida",
    "Cochliobolus", "Rhizopus", "Curvularia", "Stachybotrys", "Malassezia",
    "Fusarium", "Epicoccum", "Trichophyton", "Ulocladium", "Rhodotorula",
    "Saccharomyces",

    # Bacteria
    "Bacillus", "Staphylococcus", "Streptococcus", "Escherichia", "Salmonella",
    "Listeria",

    # Non-food mammals and human
    "Canis", "Felis", "Cavia", "Mesocricetus", "Phodopus", "Rattus", "Mus",
    "Homo",
}

merged["Genus"] = (
    merged["Source_Species"].fillna("").astype(str).str.strip().str.split().str[0]
)

mask_remove = merged["Genus"].isin(GENERA_TO_REMOVE)
removed_species = sorted(set(merged.loc[mask_remove, "Source_Species"].dropna()))
removed_genera = sorted(set(merged.loc[mask_remove, "Genus"].dropna()))

n_species_before = merged["Source_Species"].nunique()
merged = merged[~mask_remove].copy()
n_species_after = merged["Source_Species"].nunique()

log(f"\n[FILTER] species {n_species_before} -> {n_species_after} "
    f"({len(removed_species)} species / {len(removed_genera)} genera removed)")

# Audit 1: genera listed for removal but absent from the DB (typo / not present).
unmatched = sorted(GENERA_TO_REMOVE - set(removed_genera))
if unmatched:
    log(f"[WARN] {len(unmatched)} listed genera not matched in the DB "
        f"(check for typos / absence): {unmatched}")

# Audit 2: save the removed-species list for reproducibility.
pd.DataFrame({
    "removed_species": removed_species,
    "genus": [s.split()[0] if s else "" for s in removed_species],
}).to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")
log(f"[OK] Removed-species list -> {OUT_REMOVED}")

post_stats = collect_stats(merged)
report(post_stats, "after filtering")


# ---------------------------------------------------------------------------
# 4. Display name / composite key
# ---------------------------------------------------------------------------
has_common = merged["common_name"].notna() & (
    merged["common_name"].astype(str).str.strip() != ""
)
merged["Display_Name"] = merged["Source_Species"].where(
    ~has_common,
    merged["Source_Species"] + " (" + merged["common_name"].astype(str) + ")",
)

merged["Superfamily"] = merged["Superfamily"].fillna("-")
merged["Short_name"] = merged["Short_name"].fillna("Unknown")
merged["Composite_ID"] = merged["Superfamily"] + "|" + merged["Short_name"]


# ---------------------------------------------------------------------------
# 5. Build the matrix
# ---------------------------------------------------------------------------
matrix = (
    merged.groupby(["Display_Name", "Composite_ID"])["Bitscore"]
    .sum()
    .unstack()
    .fillna(0)
)
matrix.to_csv(OUT_MATRIX, encoding="utf-8-sig")

log("\n" + "=" * 62)
log(f"[DONE] Final matrix: {matrix.shape[0]} species x {matrix.shape[1]} families")
log(f"       saved -> {OUT_MATRIX}")
log("=" * 62)


# ---------------------------------------------------------------------------
# 6. Cross-check species composition against a previous NMF run
# ---------------------------------------------------------------------------
log("\n[CHECK] Compare species composition with a previous NMF run")
if not PREV_NMF_XLSX.exists():
    log(f"  skipped - file not found: {PREV_NMF_XLSX}")
else:
    try:
        prev = pd.read_excel(PREV_NMF_XLSX, sheet_name=PREV_NMF_SHEET)
        prev_set = set(prev["Display_Name"].dropna())
        new_set = set(matrix.index)

        added = sorted(new_set - prev_set)
        dropped = sorted(prev_set - new_set)

        log(f"  previous: {len(prev_set)} species  /  current: {len(new_set)} species")
        log(f"  > newly included ({len(added)}):")
        for s in added:
            log(f"      + {s}")
        log(f"  > dropped ({len(dropped)}):")
        for s in dropped:
            log(f"      - {s}")
        if not added and not dropped:
            log("      (no change)")
    except Exception as e:
        log(f"  [WARN] comparison failed: {e}")


# ---------------------------------------------------------------------------
# 7. Numbers for Methods
# ---------------------------------------------------------------------------
pct = post_stats["allergens"] / N_REFERENCE_ALLERGENS * 100

log("\n[Numbers for Methods]")
log(f"  reference allergens queried               : {N_REFERENCE_ALLERGENS:,}")
log(f"  reference allergens with >=1 hit (pre)    : {pre_stats['allergens']:,}")
log(f"  reference allergens retained (post)       : {post_stats['allergens']:,} "
    f"({pct:.1f}%)")
log(f"  GenBank accessions recovered (pre -> post): "
    f"{pre_stats['accessions']:,} -> {post_stats['accessions']:,}")
log(f"  allergen-homolog relationships (pre->post): "
    f"{pre_stats['pairs']:,} -> {post_stats['pairs']:,}")
log(f"  unique cross-reactive homologs (pre->post): "
    f"{pre_stats['homologs']:,} -> {post_stats['homologs']:,}")
log(f"  source species (pre -> post)              : "
    f"{pre_stats['species']:,} -> {post_stats['species']:,}")
log(f"  conserved-domain features (pre -> post)   : "
    f"{pre_stats['features']:,} -> {post_stats['features']:,}")

OUT_SUMMARY.write_text("\n".join(_log_lines), encoding="utf-8")
print(f"\n[OK] Run log -> {OUT_SUMMARY}")
