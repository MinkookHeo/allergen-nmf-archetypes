# -*- coding: utf-8 -*-
"""
07_phylogeny_itol.py

Builds the Core species list and the iTOL annotation files for Figure 3.

Outputs
  (A) Core species scientific-name list  -> phyloT / NCBI Common Tree input
  (B) iTOL multibar dataset (NMF archetype proportions) + leaf-label dataset
  (C) DATASET_STYLE that renders leaf labels in italic
  (D) A copy of the Newick tree for upload

Two-pass workflow
  Pass 1 (no tree yet): only the species list (A) is written; the tree must be
          generated externally (phyloT / NCBI) from that list and saved as
          data/phylogeny/phyliptree.phy.
  Pass 2 (tree present): the iTOL datasets (B, C, D) are written.
"""

import re
import shutil
import sys

import pandas as pd

from config import (
    FIGDIR, PHYLO_DIR,
    K, ARCH_HEX,
    BIOLOGICAL_MAP,
    norm_name,
)

# ===========================================================================
# 0. Paths
# ===========================================================================
IN_SOURCE = FIGDIR / "figure_source_data.csv"    # written by 04
IN_TREE = PHYLO_DIR / "phyliptree.phy"

OUT_SPECIES_TXT = PHYLO_DIR / "core_species.txt"
OUT_TREE_COPY = PHYLO_DIR / "iTOL_upload_tree.phy"
OUT_MULTIBAR = PHYLO_DIR / "iTOL_multibar_dataset.txt"
OUT_LABELS = PHYLO_DIR / "iTOL_label_dataset.txt"
OUT_STYLE = PHYLO_DIR / "iTOL_label_style_dataset.txt"


# ===========================================================================
# Helpers
# ===========================================================================
def _norm(s):
    """Normalize a tip/label for matching: unquote, underscores->spaces,
    lowercase, collapse whitespace."""
    s = str(s).strip().strip("'").replace("_", " ")
    return " ".join(s.lower().split())


def _binom(norm_s):
    """First two tokens (genus species) of a normalized name."""
    toks = norm_s.split()
    return " ".join(toks[:2]) if len(toks) >= 2 else norm_s


def parse_tree_tips(tree_text):
    """Return the tree tips as {normalized_full: token, normalized_binomial: token}.

    Handles both quoted labels ('Genus species ...') and unquoted labels
    (Genus_species:...). The token is the exact string to reference from the
    iTOL datasets.
    """
    by_full, by_binom = {}, {}
    tokens = []
    # quoted labels
    for m in re.finditer(r"'([^']+)'", tree_text):
        tokens.append((m.group(1), f"'{m.group(1)}'"))
    # unquoted labels (name immediately followed by ':')
    for m in re.finditer(r"[(,]\s*([A-Za-z_][^,():;']*?)\s*:", tree_text):
        tokens.append((m.group(1), m.group(1)))

    for clean, token in tokens:
        nf = _norm(clean)
        by_full.setdefault(nf, token)
        by_binom.setdefault(_binom(nf), token)
    return by_full, by_binom


# ===========================================================================
# 1. Core species list (scientific names for phyloT / NCBI)
# ===========================================================================
def extract_core_species(df):
    """Write the Core species scientific names, applying NCBI/phyloT synonyms."""
    core = df[df["Membership_Status"] == "Core Member"]["key"].copy()
    corrected = core.apply(lambda x: BIOLOGICAL_MAP.get(x.strip(), x.strip()))
    corrected.to_csv(OUT_SPECIES_TXT, index=False, header=False)
    print(f"[OK] Core {len(corrected)} species -> {OUT_SPECIES_TXT.name}")
    return corrected


# ===========================================================================
# 2. iTOL files (multibar + label + style)
# ===========================================================================
def generate_itol_files(df):
    """Generate the iTOL multibar, label and style datasets."""
    shutil.copy(IN_TREE, OUT_TREE_COPY)
    tree_text = IN_TREE.read_text(encoding="utf-8", errors="ignore")
    by_full, by_binom = parse_tree_tips(tree_text)

    # multibar header
    bar_lines = [
        "DATASET_MULTIBAR",
        "SEPARATOR COMMA",
        "DATASET_LABEL,NMF_Archetype_Composition",
        "COLOR,#000000",
        f"FIELD_COLORS,{','.join(ARCH_HEX)}",
        ("FIELD_LABELS,Archetype 1 (PR-10),Archetype 2 (Parvalbumin),"
         "Archetype 3 (nsLTP1),Archetype 4 (Profilin),"
         "Archetype 5 (Tropomyosin),Archetype 6 (Seed storage)"),
        "LEGEND_TITLE,Allergenic archetypes",
        "LEGEND_SHAPES,1,1,1,1,1,1",
        f"LEGEND_COLORS,{','.join(ARCH_HEX)}",
        ("LEGEND_LABELS,Archetype 1 (PR-10),Archetype 2 (Parvalbumin),"
         "Archetype 3 (nsLTP1),Archetype 4 (Profilin),"
         "Archetype 5 (Tropomyosin),Archetype 6 (Seed storage)"),
        "ALIGN_FIELDS,1",
        "WIDTH,100",
        "MARGIN,5",
        "DATA",
    ]

    label_lines = ["LABELS", "SEPARATOR TAB", "DATA"]

    # Whole leaf label rendered in italic.
    style_lines = [
        "DATASET_STYLE",
        "SEPARATOR TAB",
        "DATASET_LABEL\tLeaf label style (italic)",
        "COLOR\t#000000",
        "DATA",
    ]

    # archetype proportions
    wc = [f"Weight_A{i+1}" for i in range(K)]
    w_vals = df[wc].to_numpy(dtype=float)
    row_sums = w_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fracs = w_vals / row_sums

    missing = []
    for idx, row in df.iterrows():
        if row.get("Membership_Status") != "Core Member":
            continue

        key_name = str(row["key"]).strip()
        target_name = BIOLOGICAL_MAP.get(key_name, key_name)

        # Match against the tree by exact binomial, then genus+species.
        nt = _norm(target_name)
        node = by_full.get(nt) or by_binom.get(_binom(nt))
        if node is None:
            missing.append(f"{key_name} -> {target_name}")
            continue

        f_vals = [f"{fracs[idx, i]:.4f}" for i in range(K)]
        bar_lines.append(f"{node},{','.join(f_vals)}")

        # Leaf label: 'Genus species (common name)' on one line.
        disp = str(row.get("Display_Name", key_name)).strip()
        display_label = disp if disp else key_name

        label_lines.append(f"{node}\t{display_label}")
        style_lines.append(f"{node}\tlabel\tnode\t#000000\t1\titalic")

    OUT_MULTIBAR.write_text("\n".join(bar_lines), encoding="utf-8", newline="\n")
    OUT_LABELS.write_text("\n".join(label_lines), encoding="utf-8", newline="\n")
    OUT_STYLE.write_text("\n".join(style_lines), encoding="utf-8", newline="\n")

    print(f"[OK] iTOL multibar    -> {OUT_MULTIBAR.name}")
    print(f"[OK] iTOL labels      -> {OUT_LABELS.name}")
    print(f"[OK] iTOL label style -> {OUT_STYLE.name}")
    print(f"[OK] tree copy        -> {OUT_TREE_COPY.name}")

    if missing:
        print(f"\n[WARN] {len(missing)} Core species not found in the tree "
              f"(no dataset row written for them):")
        for m in missing:
            print(f"       - {m}")
        print("       -> add these tips to the tree (regenerate from "
              f"{OUT_SPECIES_TXT.name}) and re-run.")


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    if not IN_SOURCE.exists():
        print(f"[ERROR] source data not found: {IN_SOURCE}")
        print("        Run 04_figure_generation.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_SOURCE)
    print(f"[OK] source loaded: {len(df)} species")

    extract_core_species(df)

    if not IN_TREE.exists():
        print(f"\n[STOP] Newick tree not found: {IN_TREE}")
        print(f"       Pass 1 complete: generate the tree from "
              f"{OUT_SPECIES_TXT.name} (phyloT / NCBI Common Tree), save it as")
        print(f"       {IN_TREE}, then re-run this script to build the iTOL files.")
        sys.exit(0)

    generate_itol_files(df)
    print("\n[DONE] phylogeny files complete.")
