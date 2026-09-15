# -*- coding: utf-8 -*-
"""
08_figure2d_audit.py

Verifies the aggregation behind the Figure 2D alluvial diagram.

Because the archetype weight matrix is row-normalized, the ribbons emanating
from a taxonomic order must sum to the number of species assigned to that
order. The script checks this identity and reports every source of loss.

  [1] Per species: the row sum of Frac_A1..AK equals 1 (zero for empty rows).
  [2] Per order:   the sum of ribbon widths equals the species count.
  [3] Loss:        species without an Order, species with all-zero weights,
                   and ribbons dropped by the display threshold.
  [4] Contrast:    soft-weight aggregation against hard-label counting.
  [5] Detail:      per-species archetype breakdown for one order.

Input  : figure_source_data.csv (written by 03 and 04)
         required columns - Display_Name, Order, Archetype, Weight_A1..Weight_A{K}
Output : results/audit_fig2d_order_totals.csv
         results/audit_fig2d_species_detail.csv
         results/audit_fig2d_log.txt

Usage  : python src/08_figure2d_audit.py             # detail for Rosales
         python src/08_figure2d_audit.py Fagales     # detail for another order
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# 0. Settings (falls back to local defaults if config.py is unavailable)
# ===========================================================================
# --- allow "from config import ..." when run from src/ -------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------
try:
    from config import FIGDIR, RESULTS_DIR, K, NMF_MEMBERSHIP_PATH
except Exception:
    RESULTS_DIR = Path(".")
    FIGDIR = Path(".")
    K = 6
    NMF_MEMBERSHIP_PATH = RESULTS_DIR / "nmf_membership.csv"

# Must match the list used in draw_panel_D
TARGET_ORDERS = [
    "Poales", "Rosales", "Fagales", "Fabales", "Asterales",
    "Malpighiales", "Lamiales", "Zingiberales",
]
RIBBON_MIN = 0.001          # display threshold used in 04
TOL = 1e-6                  # floating-point tolerance

IN_CANDIDATES = [
    FIGDIR / "figure_source_data.csv",
    NMF_MEMBERSHIP_PATH,
    RESULTS_DIR / "nmf_membership.csv",
]

OUT_ORDER = RESULTS_DIR / "audit_fig2d_order_totals.csv"
OUT_SPECIES = RESULTS_DIR / "audit_fig2d_species_detail.csv"
OUT_LOG = RESULTS_DIR / "audit_fig2d_log.txt"

_lines = []


def log(msg=""):
    print(msg)
    _lines.append(str(msg))


# ===========================================================================
# 1. Load input
# ===========================================================================
def load_source():
    for p in IN_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            log(f"[IN] {p}  ({len(df)} rows)")
            return df
    log("[ERROR] Input file not found. Candidates:")
    for p in IN_CANDIDATES:
        log(f"       - {p}")
    sys.exit(1)


df = load_source()

WCOLS = [f"Weight_A{i + 1}" for i in range(K)]
missing = [c for c in ["Display_Name", "Order"] + WCOLS if c not in df.columns]
if missing:
    log(f"[ERROR] Missing required columns: {missing}")
    log(f"        Columns present: {list(df.columns)}")
    sys.exit(1)

log(f"[OK] {len(df)} species / K={K}")

# ===========================================================================
# 2. Archetype proportions (same procedure as draw_panel_D)
# ===========================================================================
w = df[WCOLS].to_numpy(dtype=float)
row_sums = w.sum(axis=1, keepdims=True)
zero_mask = (row_sums.ravel() == 0)
row_sums[row_sums == 0] = 1.0
fracs = w / row_sums

FCOLS = [f"Frac_A{i + 1}" for i in range(K)]
for i, c in enumerate(FCOLS):
    df[c] = fracs[:, i]

df["Frac_rowsum"] = df[FCOLS].sum(axis=1)
df["is_zero_row"] = zero_mask

# ---- [1] Per-species row sums ---------------------------------------------
log("\n" + "=" * 70)
log("[1] Per species: row sum of proportions == 1")
log("=" * 70)

bad_rows = df[(~df["is_zero_row"]) & (np.abs(df["Frac_rowsum"] - 1.0) > TOL)]
if len(bad_rows):
    log(f"[FAIL] {len(bad_rows)} species whose row sum is not 1")
    for _, r in bad_rows.iterrows():
        log(f"       {r['Display_Name']}: {r['Frac_rowsum']:.6f}")
else:
    log(f"[PASS] all {int((~zero_mask).sum())} species with non-zero weights sum to 1")

if zero_mask.any():
    log(f"[NOTE] {int(zero_mask.sum())} species have all-zero archetype weights "
        f"and contribute nothing to the diagram")
    for nm in df.loc[zero_mask, "Display_Name"]:
        log(f"       - {nm}")

# ---- Species without an Order ---------------------------------------------
n_no_order = int(df["Order"].isna().sum())
if n_no_order:
    log(f"\n[NOTE] {n_no_order} species have no Order and are excluded entirely")
    for nm in df.loc[df["Order"].isna(), "Display_Name"]:
        log(f"       - {nm}")

# ===========================================================================
# 3. Per-order aggregation check
# ===========================================================================
log("\n" + "=" * 70)
log("[2] Per order: sum of ribbon widths == species count")
log("=" * 70)

rows = []
for o in TARGET_ORDERS:
    sub = df[df["Order"] == o]
    n_sp = len(sub)
    if n_sp == 0:
        log(f"[WARN] {o}: no species in the dataset.")
        continue

    per_arch = sub[FCOLS].sum()                   # contribution per archetype
    soft_total = float(per_arch.sum())            # should equal n_sp
    drawn = float(per_arch[per_arch > RIBBON_MIN].sum())   # width actually drawn
    dropped = soft_total - drawn
    n_zero = int(sub["is_zero_row"].sum())
    n_ribbon = int((per_arch > RIBBON_MIN).sum())

    expected = n_sp - n_zero                      # all-zero species contribute 0
    ok = abs(soft_total - expected) <= TOL * max(1, n_sp)

    # Contrast with hard-label counting
    hard = sub["Archetype"].value_counts().reindex(
        range(1, K + 1), fill_value=0) if "Archetype" in sub.columns else None
    n_arch_hard = int((hard > 0).sum()) if hard is not None else np.nan

    rows.append({
        "Order": o,
        "n_species": n_sp,
        "n_zero_weight_species": n_zero,
        "expected_total": expected,
        "soft_sum": round(soft_total, 6),
        "drawn_sum": round(drawn, 6),
        "dropped_by_threshold": round(dropped, 6),
        "n_ribbons_drawn": n_ribbon,
        "n_archetypes_hard_label": n_arch_hard,
        "check": "PASS" if ok else "FAIL",
    })

    flag = "PASS" if ok else "FAIL"
    log(f"  {o:<14} n {n_sp:>3} | sum {soft_total:8.4f} "
        f"| expected {expected:>3} | drawn {drawn:8.4f} "
        f"| dropped {dropped:.4f} | ribbons {n_ribbon}/{K} "
        f"| hard {n_arch_hard} | {flag}")

order_df = pd.DataFrame(rows)
order_df.to_csv(OUT_ORDER, index=False, encoding="utf-8-sig")

n_fail = int((order_df["check"] == "FAIL").sum())
tot_dropped = float(order_df["dropped_by_threshold"].sum())

log("")
if n_fail:
    log(f"[FAIL] {n_fail} orders disagree with their species-level totals.")
else:
    log("[PASS] every displayed order satisfies sum(ribbons) = species count")

log(f"[3] Display threshold RIBBON_MIN={RIBBON_MIN} dropped a total "
    f"contribution of {tot_dropped:.6f} (equivalent to {tot_dropped:.4f} species)")
if tot_dropped > 0:
    log("    -> state in the legend that ribbons below 0.1% are not drawn")

# ---- Soft weights vs hard labels ------------------------------------------
log("\n" + "=" * 70)
log("[4] Soft-weight aggregation vs hard-label counting")
log("=" * 70)
if "Archetype" in df.columns:
    for o in order_df["Order"]:
        sub = df[df["Order"] == o]
        soft = sub[FCOLS].sum().round(3).tolist()
        hard = sub["Archetype"].value_counts().reindex(
            range(1, K + 1), fill_value=0).tolist()
        log(f"  {o:<14} soft {soft}")
        log(f"  {'':<14} hard {hard}")
else:
    log("  Column 'Archetype' is absent; skipping the contrast.")

# ===========================================================================
# 4. Per-species breakdown for one order
# ===========================================================================
focus = sys.argv[1] if len(sys.argv) > 1 else "Rosales"

log("\n" + "=" * 70)
log(f"[5] {focus}: per-species archetype breakdown")
log("=" * 70)

sub = df[df["Order"] == focus].copy()
if len(sub) == 0:
    log(f"  No species found for {focus}.")
else:
    sub = sub.sort_values("Display_Name")
    cols = ["Display_Name"] + FCOLS + ["Frac_rowsum"]
    if "Archetype" in sub.columns:
        cols.insert(1, "Archetype")
    if "Membership_Status" in sub.columns:
        cols.insert(1, "Membership_Status")

    detail = sub[cols].copy()
    detail[FCOLS] = detail[FCOLS].round(4)
    detail.to_csv(OUT_SPECIES, index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 200,
                           "display.max_columns", 50):
        log(detail.to_string(index=False))

    log(f"\n  {focus} species count : {len(sub)}")
    log(f"  total proportion    : {sub[FCOLS].to_numpy().sum():.6f}")
    log(f"  sum per archetype   : "
        f"{dict(zip(range(1, K + 1), sub[FCOLS].sum().round(4).tolist()))}")
    log(f"  -> these must match the {focus} ribbon widths in Figure 2D")
    log(f"  [OK] per-species detail -> {OUT_SPECIES}")

# ===========================================================================
# 5. Save
# ===========================================================================
OUT_LOG.write_text("\n".join(_lines), encoding="utf-8")
log(f"\n[OK] order summary -> {OUT_ORDER}")
log(f"[OK] log           -> {OUT_LOG}")
