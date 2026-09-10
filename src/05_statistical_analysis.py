# -*- coding: utf-8 -*-
"""
Compute every statistic reported in the manuscript.

Loads precomputed NMF results (W, H matrices) from 03_cluster_analysis.py
rather than refitting.

Sections:
  [1] COUNTS      descriptive statistics (Core/Ambiguous, per-archetype counts)
  [2] BIAS        test whether repertoire entropy is an annotation artifact
  [3] RAREFACTION entropy rank stability after equalizing sequencing depth
  [4] AMI         archetype vs taxonomy agreement with a permutation null

Rationale for [2]
-----------------------------------------------------------------------------
The question is whether the multi-family profile of an Ambiguous species is
genuine biology or a by-product of the species being well studied and having
many registered allergens.

The difficulty is that every candidate proxy carries both components at once:
feature richness, homolog count, total bitscore, and reference-allergen count
all reflect annotation depth AND true breadth simultaneously. Peanut carrying
Ara h 1-17 is partly a matter of fame, but also because it genuinely spans
cupin / prolamin / PR-10 / profilin. A partial correlation controlling a
single proxy therefore cannot separate the two components.

This script instead decomposes the reference-allergen count into two parts:

      n_allergens = n_families  x  allergens_per_family
                    -----------    -------------------
                    breadth        redundant listing
                    (biology)      (annotation depth)

Two species may each have 10 allergens, yet one has all 10 as 2S albumin
(n_families=1, per_family=10) while the other spreads them across 6 families
(n_families=6, per_family=1.7). The former is deep only; the latter is
genuinely multi-family.

  Test A : entropy vs n_families                     -> breadth component
  Test B : entropy vs per_family | n_families held   -> pure depth component
  Test C : entropy vs n_allergens | n_families held  -> alternative form of B

If B and C are null, "listing depth alone does not create entropy" can be
stated quantitatively. If B is significant, it must be moved to Limitations.

Reporting principles
  - Do not accept the null from a p-value alone; report a bootstrap 95% CI.
  - Use TOST equivalence testing to affirm negligible magnitude.
  - Report every correlation that reaches significance, without hiding any.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. Paths / constants (config.py)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DB_PATH, MATRIX_PATH, RESULTS_DIR,
    NMF_W_PATH, NMF_H_PATH,
    K, RANDOM_STATE, CORE_RA_THRESHOLD,
    norm_name,
)

IN_MATRIX = MATRIX_PATH
OUT_TXT = RESULTS_DIR / "statistical_analysis.txt"
OUT_CSV = RESULTS_DIR / "statistics_per_species.csv"

N_BOOT = 10000          # bootstrap iterations
N_PERM = 5000           # AMI permutation iterations
RAREFY_REPS = 200       # rarefaction iterations
TOST_BOUNDS = (0.15, 0.20, 0.25, 0.30)

_log = []


def log(m=""):
    print(m)
    _log.append(str(m))


# ---------------------------------------------------------------------------
# 1. Statistics utilities
# ---------------------------------------------------------------------------
def _residualize(rank_y, rank_Z):
    design = np.column_stack([np.ones(len(rank_y)), rank_Z])
    beta, *_ = np.linalg.lstsq(design, rank_y, rcond=None)
    return rank_y - design @ beta


def partial_spearman(x, y, Z=None):
    """Partial Spearman correlation; simple Spearman when Z is None."""
    df = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})
    if Z is not None:
        Z = np.atleast_2d(np.asarray(Z, float))
        if Z.shape[0] != len(df):
            Z = Z.T
        for j in range(Z.shape[1]):
            df[f"z{j}"] = Z[:, j]
    df = df.dropna()
    n = len(df)
    if n < 5:
        return np.nan, np.nan, n, 0

    zcols = [c for c in df.columns if c.startswith("z")]
    if not zcols:
        rho, p = stats.spearmanr(df["x"], df["y"])
        return float(rho), float(p), n, 0

    rx = rankdata(df["x"].to_numpy())
    ry = rankdata(df["y"].to_numpy())
    RZ = np.column_stack([rankdata(df[c].to_numpy()) for c in zcols])
    ex, ey = _residualize(rx, RZ), _residualize(ry, RZ)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return np.nan, np.nan, n, len(zcols)
    rho, p = stats.pearsonr(ex, ey)
    return float(rho), float(p), n, len(zcols)


def boot_ci(x, y, Z=None, n_boot=N_BOOT, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Zc = None if Z is None else np.atleast_2d(np.asarray(Z, float))
    if Zc is not None and Zc.shape[0] != len(x):
        Zc = Zc.T

    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _, _, _ = partial_spearman(x[idx], y[idx],
                                      None if Zc is None else Zc[idx])
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 100:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def tost_equivalence(rho, n, n_ctrl=0, bound=0.25):
    """Fisher z-based TOST. H0: |rho| >= bound."""
    dof = n - 3 - n_ctrl
    if dof <= 0 or not np.isfinite(rho):
        return np.nan
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(dof)
    zb = np.arctanh(bound)
    return float(max(1 - stats.norm.cdf((z + zb) / se),
                     stats.norm.cdf((z - zb) / se)))


def report(name, rho, p, n, lo, hi, n_ctrl=0, note=""):
    log(f"  {name}")
    if not np.isfinite(rho):
        log(f"      not computable (zero residual variance or too few samples). "
            f"n = {n}")
        if note:
            log(f"      {note}")
        return
    log(f"      rho = {rho:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
        f"p = {p:.4f}   n = {n}")
    cells, smallest = [], None
    for b in TOST_BOUNDS:
        pt = tost_equivalence(rho, n, n_ctrl, bound=b)
        cells.append(f"±{b:.2f}:{pt:.3f}{'*' if pt < 0.05 else ' '}")
        if pt < 0.05 and smallest is None:
            smallest = b
    log("      TOST p (equivalence) " + "  ".join(cells) + "   (* = holds)")
    if p < 0.05:
        log("      -> a significant correlation is present")
    elif smallest is not None:
        log(f"      -> non-significant and |rho| < {smallest:.2f} equivalence "
            f"holds = negligible magnitude can be asserted")
    else:
        log("      -> non-significant but equivalence not established: "
            "underpowered; do not claim 'not an artifact'")
    if note:
        log(f"      {note}")


# ---------------------------------------------------------------------------
# 2. Per-species annotation profile (DB)
# ---------------------------------------------------------------------------
def species_annotation_profile(db_path):
    """
    Aggregate per species:
        N_Allergens : number of WHO/IUIS reference allergens  (depth x breadth)
        N_Families  : distinct families spanned by those allergens  (breadth)
        Per_Family  : N_Allergens / N_Families  (redundant listing = depth)
        N_Homologs  : number of unique recovered homologs
    Each allergen's family is defined by the CDD composite identifier
    (Superfamily|Short name) of its highest-bitscore homolog.
    """
    q = """
    SELECT A.rowid           AS Allergen_UID,
           A.species         AS Source_Species,
           C.Similar_Protein AS Similar_Protein,
           P.Superfamily     AS Superfamily,
           P."Short name"    AS Short_name,
           P.Bitscore        AS Bitscore
    FROM Allergens A
    JOIN Crossreactivity C
      ON ';' || A.genbank_ids || ';' LIKE '%;' || C.Query_ID || ';%'
    LEFT JOIN Protein_families P
      ON C.Similar_Protein = P.Query
    """
    with sqlite3.connect(db_path) as conn:
        rel = pd.read_sql_query(q, conn)

    rel["key"] = rel["Source_Species"].apply(norm_name)
    rel["Composite"] = (rel["Superfamily"].fillna("-") + "|"
                        + rel["Short_name"].fillna("Unknown"))
    rel["Bitscore"] = pd.to_numeric(rel["Bitscore"], errors="coerce").fillna(0)

    best = (rel.sort_values("Bitscore", ascending=False)
               .drop_duplicates("Allergen_UID")[["Allergen_UID", "key", "Composite"]])

    prof = best.groupby("key").agg(
        N_Allergens=("Allergen_UID", "nunique"),
        N_Families=("Composite", "nunique"),
    )
    prof["Per_Family"] = prof["N_Allergens"] / prof["N_Families"].clip(lower=1)
    prof["N_Homologs"] = rel.groupby("key")["Similar_Protein"].nunique()
    return prof


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    # --- data
    raw = pd.read_csv(IN_MATRIX, index_col=0)
    log(f"[OK] Input matrix: {raw.shape[0]} species x {raw.shape[1]} features")

    zero_rows = raw.index[raw.sum(axis=1) == 0].tolist()
    if zero_rows:
        log(f"[WARN] {len(zero_rows)} species have no assignable conserved "
            f"domain: {zero_rows}")

    matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(raw))

    # --- load precomputed NMF; fall back to fitting if .npy absent
    if NMF_W_PATH.exists() and NMF_H_PATH.exists():
        W = np.load(NMF_W_PATH)
        H = np.load(NMF_H_PATH)
        log(f"[OK] Loaded W{W.shape}, H{H.shape} from {NMF_W_PATH.parent}")
    else:
        from sklearn.decomposition import NMF
        model = NMF(n_components=K, init="nndsvda", max_iter=5000,
                    random_state=RANDOM_STATE)
        W = model.fit_transform(matrix_norm)
        H = model.components_
        log(f"[WARN] .npy not found; fitted NMF from scratch. "
            f"Run 03_cluster_analysis.py first for reproducibility.")

    o_idx = np.argsort(W, axis=1)[:, ::-1]
    t1 = np.take_along_axis(W, o_idx[:, :1], axis=1).ravel()
    t2 = np.take_along_axis(W, o_idx[:, 1:2], axis=1).ravel()
    rowsum = W.sum(axis=1)
    rel_ab = t1 / np.where(rowsum == 0, 1, rowsum)
    dom_ratio = t1 / (t2 + 1e-9)
    core = rel_ab >= CORE_RA_THRESHOLD

    Wn = W / rowsum.reshape(-1, 1).clip(min=1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(Wn > 0, Wn * np.log(Wn), 0.0), axis=1)

    # ================================================================
    # [1] COUNTS
    # ================================================================
    log("\n" + "=" * 68)
    log("[1] COUNTS  descriptive statistics for the manuscript")
    log("=" * 68)
    alt = (dom_ratio >= 2.0) & ((t1 >= 0.3) | (rel_ab >= 0.8))
    log(f"  total species                    : {len(raw)}")
    log(f"  Core Member (RA >= {CORE_RA_THRESHOLD})        : {int(core.sum())}")
    log(f"  Ambiguous                        : {int((~core).sum())}")
    log(f"  disagreement with alt. rule      : {int((core != alt).sum())} species")
    if core.any():
        log(f"  min dominance ratio among Core   : {dom_ratio[core].min():.3f}"
            f"   (>= 4 matches the manuscript)")
    per_arch = (pd.Series(o_idx[core, 0] + 1).value_counts()
                .reindex(range(1, K + 1), fill_value=0).tolist())
    log(f"  Core count per archetype 1..{K}    : {per_arch}  (sum {sum(per_arch)})")
    log(f"  species with second weight == 0  : {int((t2 == 0).sum())}"
        f"   (Figure 3 legend value)")
    log(f"  species with no assignable domain: {len(zero_rows)}")

    # --- per-species profile
    df = pd.DataFrame({
        "Display_Name": raw.index,
        "Entropy": entropy,
        "Relative_Abundance": rel_ab,
        "Dominance_Ratio": dom_ratio,
        "Membership_Status": np.where(core, "Core Member", "Ambiguous"),
        "Feature_Richness": (raw > 0).sum(axis=1).to_numpy(),
        "Total_Bitscore": raw.sum(axis=1).to_numpy(),
    })
    df["key"] = df["Display_Name"].apply(norm_name)
    df = df.join(species_annotation_profile(DB_PATH), on="key")
    n_miss = int(df["N_Allergens"].isna().sum())
    log(f"\n  Per-species DB profile mapped (unmatched: {n_miss} species)")
    for nm in df.loc[df["N_Allergens"].isna(), "Display_Name"]:
        log(f"      unmatched: {nm}")

    # ================================================================
    # [2] BIAS
    # ================================================================
    def bias_block(sub, tag):
        log("\n" + "=" * 68)
        log(f"[2] BIAS  annotation-artifact test  [{tag}]  n = {len(sub)}")
        log("=" * 68)
        e = sub["Entropy"].to_numpy()

        log("\n-- (a) Simple correlations: all reported "
            "(interpretation via the decomposition below) --")
        log("     Note: each variable below carries both annotation depth and")
        log("           true breadth, so none can be interpreted alone.")
        for col, label in [
                ("N_Allergens", "entropy vs number of reference allergens"),
                ("N_Families", "entropy vs number of allergen families"),
                ("N_Homologs", "entropy vs number of homologs"),
                ("Feature_Richness", "entropy vs feature richness"),
                ("Total_Bitscore", "entropy vs total bitscore")]:
            v = sub[col].to_numpy(float)
            r, p, n, _ = partial_spearman(e, v)
            lo, hi = boot_ci(e, v, seed=1)
            report(label, r, p, n, lo, hi)

        log("\n-- (b) Decomposition: breadth component vs depth component --")
        log("     n_allergens = n_families x per_family")
        log("                   (breadth)    (redundant listing = depth)")

        nf = sub[["N_Families"]].to_numpy(float)

        r, p, n, _ = partial_spearman(e, sub["N_Families"].to_numpy(float))
        lo, hi = boot_ci(e, sub["N_Families"].to_numpy(float), seed=2)
        report("[Test A] entropy vs family count  (breadth component)",
               r, p, n, lo, hi,
               note="expected significant = repertoire is genuinely multi-family")

        r, p, n, nc = partial_spearman(e, sub["Per_Family"].to_numpy(float), nf)
        lo, hi = boot_ci(e, sub["Per_Family"].to_numpy(float), nf, seed=3)
        report("[Test B] entropy vs allergens-per-family | family count  (depth)",
               r, p, n, lo, hi, n_ctrl=nc,
               note="should be null = at equal breadth, more listings do not "
                    "change entropy")

        r, p, n, nc = partial_spearman(e, sub["N_Allergens"].to_numpy(float), nf)
        lo, hi = boot_ci(e, sub["N_Allergens"].to_numpy(float), nf, seed=4)
        report("[Test C] entropy vs allergen count | family count  (alt. form of B)",
               r, p, n, lo, hi, n_ctrl=nc)

        log("\n-- (c) Verdict --")
        rB, pB, nB, ncB = partial_spearman(
            e, sub["Per_Family"].to_numpy(float), nf)
        eqB = min([b for b in TOST_BOUNDS
                   if tost_equivalence(rB, nB, ncB, b) < 0.05], default=None)
        rA, pA, _, _ = partial_spearman(e, sub["N_Families"].to_numpy(float))
        if not (np.isfinite(pA) and np.isfinite(pB)):
            log("     >> Test A or B not computable; check input distributions.")
        elif pA < 0.05 and pB >= 0.05 and eqB is not None:
            log("     >> Breadth explains entropy; annotation depth does not.")
            log("        The claim 'not reducible to annotation depth' is")
            log("        quantitatively supported.")
        elif pB < 0.05:
            log("     >> The depth component is also significantly associated")
            log("        with entropy. An artifact cannot be excluded; move to")
            log("        Limitations and defend with external clinical validity.")
        else:
            log("     >> The depth component is non-significant but equivalence")
            log("        is not established (underpowered); describe cautiously")
            log("        as 'cannot be excluded'.")

    bias_block(df, "all species")
    if zero_rows:
        bias_block(df[~df["Display_Name"].isin(zero_rows)].reset_index(drop=True),
                   "excluding species with no assignable domain")

    # ================================================================
    # [3] RAREFACTION
    # ================================================================
    log("\n" + "=" * 68)
    log("[3] RAREFACTION  entropy rank stability after depth equalization")
    log("=" * 68)
    log("  Goal: test whether entropy ranks arise from per-species total counts.")
    log("        High agreement with the original entropy means depth does not")
    log("        drive the ranks.")

    counts = raw.to_numpy(float)
    depth = counts.sum(axis=1)
    target = int(np.floor(np.percentile(depth[depth > 0], 10)))
    keep = depth >= target
    log(f"  target depth = {target:,} (10th percentile of positive depths), "
        f"{int(keep.sum())} / {len(depth)} species")

    # For rarefaction transform, need to project through the fitted NMF.
    # If W was loaded, reconstruct the model via H for transform().
    # Use a lightweight wrapper that applies the precomputed H.
    class _NMFProjector:
        """Project new data through a precomputed H matrix."""
        def __init__(self, H_matrix):
            self.components_ = H_matrix
        def transform(self, X):
            from sklearn.decomposition import non_negative_factorization
            W_out, _, _ = non_negative_factorization(
                X, H=self.components_, n_components=self.components_.shape[0],
                init="custom", update_H=False, max_iter=500, random_state=RANDOM_STATE)
            return W_out

    projector = _NMFProjector(H)

    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = np.where(keep)[0]
    probs = counts[sub_idx] / counts[sub_idx].sum(axis=1, keepdims=True)
    acc = np.zeros(len(sub_idx))
    for _ in range(RAREFY_REPS):
        sampled = np.array([rng.multinomial(target, p) for p in probs], float)
        Wr = projector.transform(
            Normalizer(norm="l1").fit_transform(np.log1p(sampled)))
        Wrn = Wr / Wr.sum(axis=1, keepdims=True).clip(min=1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc += -np.nansum(np.where(Wrn > 0, Wrn * np.log(Wrn), 0.0), axis=1)

    ent_rare = np.full(len(depth), np.nan)
    ent_rare[sub_idx] = acc / RAREFY_REPS
    df["Entropy_Rarefied"] = ent_rare

    ok = df["Entropy_Rarefied"].notna()
    r2, p2 = stats.spearmanr(df.loc[ok, "Entropy"], df.loc[ok, "Entropy_Rarefied"])
    log(f"  original vs rarefied entropy: rho = {r2:+.3f}, p = {p2:.3g}, "
        f"n = {int(ok.sum())}")
    if r2 > 0.9:
        log("  -> Ranks are essentially preserved; entropy reflects the shape")
        log("     of the archetype-weight distribution, not the total count.")

    rB2, pB2, nB2, ncB2 = partial_spearman(
        df.loc[ok, "Entropy_Rarefied"], df.loc[ok, "Per_Family"],
        df.loc[ok, ["N_Families"]].to_numpy(float))
    loB2, hiB2 = boot_ci(df.loc[ok, "Entropy_Rarefied"].to_numpy(),
                         df.loc[ok, "Per_Family"].to_numpy(),
                         df.loc[ok, ["N_Families"]].to_numpy(float), seed=5)
    report("[Test B re-check] rarefied entropy vs allergens-per-family | family count",
           rB2, pB2, nB2, loB2, hiB2, n_ctrl=ncB2)

    # ================================================================
    # [4] AMI
    # ================================================================
    log("\n" + "=" * 68)
    log("[4] AMI  archetype vs taxonomic order + permutation null")
    log("=" * 68)
    with sqlite3.connect(DB_PATH) as conn:
        tax = pd.read_sql_query(
            "SELECT Species, [Order] AS Ord FROM SpeciesTaxonomy", conn)
    tax["key"] = tax["Species"].apply(norm_name)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    df["Order"] = df["key"].map(
        tax[valid].drop_duplicates("key").set_index("key")["Ord"])
    df["Archetype"] = o_idx[:, 0] + 1

    sub = df[df["Order"].notna()]
    lab_tax = sub["Order"].astype("category").cat.codes.to_numpy()
    lab_arc = (sub["Archetype"] - 1).to_numpy()
    ami = adjusted_mutual_info_score(lab_tax, lab_arc, average_method="arithmetic")
    log(f"  observed AMI = {ami:.4f}  (n = {len(sub)}, "
        f"orders = {sub['Order'].nunique()})")

    rng2 = np.random.default_rng(RANDOM_STATE)
    null = np.array([
        adjusted_mutual_info_score(lab_tax, rng2.permutation(lab_arc),
                                   average_method="arithmetic")
        for _ in range(N_PERM)])
    p_emp = (np.sum(null >= ami) + 1) / (N_PERM + 1)
    log(f"  permutation null: mean {null.mean():+.4f}, SD {null.std():.4f}, "
        f"95th pct {np.percentile(null, 95):+.4f}")
    log(f"  empirical p = {p_emp:.4g}   z = {(ami - null.mean()) / null.std():.2f}")
    if p_emp < 0.05:
        log("  -> Clearly above chance but small in absolute terms; report as a")
        log("     'significant but weak phylogenetic signal'.")
    else:
        log("  -> Indistinguishable from chance; describe as independent of "
            "phylogeny.")

    # --- save
    df.drop(columns=["key"]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_TXT.write_text("\n".join(_log), encoding="utf-8")
    log(f"\n[DONE] per-species values -> {OUT_CSV}")
    log(f"[DONE] log -> {OUT_TXT}")


if __name__ == "__main__":
    main()
