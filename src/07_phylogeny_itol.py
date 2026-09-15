# -*- coding: utf-8 -*-
"""
07_phylogeny_itol.py

Builds the inputs for Figure 3.

  (A) Writes the Core species list in phyloT / NCBI Common Tree format.
  (B) Writes an iTOL multibar dataset holding the archetype proportions of each
      Core species, together with a matching label file.
  (C) Copies the Newick tree for upload.

Bar segments use the continuous archetype weights, so a species that loads onto
several archetypes is shown as a stacked bar rather than a single colour.
"""

import re
import shutil
from pathlib import Path

import pandas as pd

# --- allow "from config import ..." when run from src/ -------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------
from config import (
    FIGDIR, PHYLO_DIR,
    K, ARCH_HEX,
    BIOLOGICAL_MAP, PHYLOT_CORRECTIONS,
    norm_name,
)

# ===========================================================================
# 0. Paths
# ===========================================================================
IN_SOURCE = FIGDIR / "figure_source_data.csv"    # written by 04
IN_TREE = PHYLO_DIR / "phyliptree.phy"

OUT_SPECIES_TXT = PHYLO_DIR / "core_135_species.txt"
OUT_TREE_COPY = PHYLO_DIR / "iTOL_upload_tree.phy"
OUT_MULTIBAR = PHYLO_DIR / "iTOL_multibar_dataset.txt"
OUT_LABELS = PHYLO_DIR / "iTOL_label_dataset.txt"


# ===========================================================================
# 1. Core species list (phyloT synonyms applied)
# ===========================================================================
def extract_core_species(df):
    """Write the Core species binomials in phyloT / NCBI compatible form."""
    core = df[df["Membership_Status"] == "Core Member"]["key"].copy()

    corrected = core.apply(
        lambda x: PHYLOT_CORRECTIONS.get(x.strip(), x.strip()))

    corrected.to_csv(OUT_SPECIES_TXT, index=False, header=False)
    print(f"[OK] {len(corrected)} Core species -> {OUT_SPECIES_TXT.name}")
    return corrected


# ===========================================================================
# 2. iTOL files (multibar + labels)
# ===========================================================================
def generate_itol_files(df):
    """Write the iTOL multibar dataset and the label file."""

    # --- Copy the Newick tree
    if IN_TREE.exists():
        shutil.copy(IN_TREE, OUT_TREE_COPY)
        tree_text = IN_TREE.read_text(encoding="utf-8", errors="ignore")
        tree_actual_nodes = set(re.findall(r"'[^']+'", tree_text))
        tree_clean_names = {node.strip("'"): node for node in tree_actual_nodes}
    else:
        print(f"[WARN] Newick tree not found: {IN_TREE}")
        tree_clean_names = {}
        tree_actual_nodes = set()

    # --- Locate the common-name column
    common_cols = [c for c in df.columns
                   if 'common' in c.lower()
                   or ('name' in c.lower() and c.lower() != 'key')]
    common_col = common_cols[0] if common_cols else None

    # --- Multibar header
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

    label_lines = [
        "LABELS",
        "SEPARATOR TAB",
        "DATA",
    ]

    # --- Archetype proportions
    wc = [f"Weight_A{i+1}" for i in range(K)]
    w_vals = df[wc].to_numpy(dtype=float)
    row_sums = w_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fracs = w_vals / row_sums

    # --- Iterate over Core species
    for idx, row in df.iterrows():
        if row.get("Membership_Status") != "Core Member":
            continue

        key_name = str(row["key"]).strip()
        target_name = BIOLOGICAL_MAP.get(key_name, key_name)

        # Match the tree node
        final_node_id = None
        if target_name in tree_clean_names:
            final_node_id = tree_clean_names[target_name]
        else:
            genus = target_name.split()[0]
            for act_node in tree_actual_nodes:
                if genus.lower() in act_node.lower():
                    final_node_id = act_node
                    break
            if final_node_id is None:
                import difflib
                matches = difflib.get_close_matches(
                    target_name, tree_clean_names.keys(), n=1, cutoff=0.5)
                final_node_id = (tree_clean_names[matches[0]]
                                 if matches else f"'{target_name}'")

        final_node_id = final_node_id.replace("\n", "").replace("\r", "")

        # Multibar row
        f_vals = [f"{fracs[idx, i]:.4f}" for i in range(K)]
        bar_lines.append(f"{final_node_id},{','.join(f_vals)}")

        # Label, avoiding doubled parentheses
        common_name = ""
        if common_col and pd.notna(row.get(common_col)):
            common_name = str(row[common_col]).strip()
            if common_name.lower() == "nan":
                common_name = ""

        if common_name:
            if key_name.lower() in common_name.lower() or "(" in common_name:
                display_label = common_name
            else:
                display_label = f"{key_name} ({common_name})"
        else:
            display_label = key_name

        label_lines.append(f"{final_node_id}\t{display_label}")

    # --- Write
    OUT_MULTIBAR.write_text("\n".join(bar_lines), encoding="utf-8",
                            newline="\n")
    OUT_LABELS.write_text("\n".join(label_lines), encoding="utf-8",
                          newline="\n")

    print(f"[OK] iTOL multibar -> {OUT_MULTIBAR.name}")
    print(f"[OK] iTOL labels   -> {OUT_LABELS.name}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    if not IN_SOURCE.exists():
        print(f"[ERROR] Source data not found: {IN_SOURCE}")
        print("        Run 04_figure_generation.py first.")
        import sys; sys.exit(1)

    df = pd.read_csv(IN_SOURCE)
    print(f"[OK] Source loaded: {len(df)} species")

    extract_core_species(df)
    generate_itol_files(df)

    print("\n[DONE] iTOL input files written.")
