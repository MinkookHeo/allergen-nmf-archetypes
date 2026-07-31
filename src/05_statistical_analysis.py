import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. 경로 / 상수 (config.py 연동)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MATRIX_PATH, RESULTS_DIR

IN_MATRIX = MATRIX_PATH
OUT_TXT = RESULTS_DIR / "statistical_analysis.txt"
OUT_CSV = RESULTS_DIR / "statistics_per_species.csv"

K = 6
RANDOM_STATE = 42
CORE_RA_THRESHOLD = 0.80

N_BOOT = 10000          
N_PERM = 5000           
RAREFY_REPS = 200       
TOST_BOUNDS = (0.15, 0.20, 0.25, 0.30)

_log = []
def log(m=""):
    print(m)
    _log.append(str(m))


def norm_name(s):
    """'Genus species (common name)' -> 'Genus species'"""
    return str(s).split(" (")[0].strip()

# ===========================================================================
# 1. 통계 유틸
# ===========================================================================
def _residualize(rank_y, rank_Z):
    design = np.column_stack([np.ones(len(rank_y)), rank_Z])
    beta, *_ = np.linalg.lstsq(design, rank_y, rcond=None)
    return rank_y - design @ beta


def partial_spearman(x, y, Z=None):
    """Spearman 편상관. Z=None 이면 단순 Spearman."""
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
    # 잔차가 상수이면 상관이 정의되지 않는다(예: 모든 종의 Per_Family 가 동일).
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
    """Fisher z 기반 TOST. H0: |rho| >= bound."""
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
        log(f"      계산 불가 (잔차 분산 0 또는 표본 부족).  n = {n}")
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
    log("      TOST p (등가성) " + "  ".join(cells) + "   (*=성립)")
    if p < 0.05:
        log("      -> 유의한 상관이 존재한다")
    elif smallest is not None:
        log(f"      -> 비유의 + |rho| < {smallest:.2f} 등가성 성립 "
            f"= '무시할 수준'이라고 적극 주장 가능")
    else:
        log("      -> 비유의하지만 등가성 미성립: 검정력 부족. "
            "'artifact 가 아니다' 라고 단정하지 말 것")
    if note:
        log(f"      {note}")


# ===========================================================================
# 2. 종별 annotation 지표 (DB)
# ===========================================================================
def species_annotation_profile(db_path):
    """
    종별로 다음을 집계한다.
        N_Allergens   : WHO/IUIS reference allergen 수      (깊이 x 다면성)
        N_Families    : 그 알레르겐들이 걸친 서로 다른 family 수  (다면성)
        Per_Family    : N_Allergens / N_Families            (중복 등재 = 깊이)
        N_Homologs    : 회수된 unique homolog 수
    각 reference allergen 의 family 는 최고 bitscore homolog 의
    CDD 복합 식별자(Superfamily|Short name)로 정의한다.
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

    # 알레르겐 1개 -> 최고 bitscore homolog 의 family 를 그 알레르겐의 family 로
    best = (rel.sort_values("Bitscore", ascending=False)
               .drop_duplicates("Allergen_UID")[["Allergen_UID", "key", "Composite"]])

    prof = best.groupby("key").agg(
        N_Allergens=("Allergen_UID", "nunique"),
        N_Families=("Composite", "nunique"),
    )
    prof["Per_Family"] = prof["N_Allergens"] / prof["N_Families"].clip(lower=1)
    prof["N_Homologs"] = rel.groupby("key")["Similar_Protein"].nunique()
    return prof


# ===========================================================================
# 3. 메인
# ===========================================================================
def main():
    # -------------------------------------------------- 데이터 + NMF
    raw = pd.read_csv(IN_MATRIX, index_col=0)
    log(f"[OK] 입력 행렬: {raw.shape[0]} 종 x {raw.shape[1]} feature")

    zero_rows = raw.index[raw.sum(axis=1) == 0].tolist()
    if zero_rows:
        log(f"[WARN] 할당 가능한 conserved domain 이 없는 종 "
            f"{len(zero_rows)}개: {zero_rows}")

    matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(raw))
    model = NMF(n_components=K, init="nndsvda", max_iter=5000,
                random_state=RANDOM_STATE)
    W = model.fit_transform(matrix_norm)
    H = model.components_
    log(f"[OK] NMF K={K}: W{W.shape}, H{H.shape}")

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
    log("[1] COUNTS  원고 기재용 기술통계")
    log("=" * 68)
    hybrid = (dom_ratio >= 2.0) & ((t1 >= 0.3) | (rel_ab >= 0.8))
    log(f"  총 종 수                        : {len(raw)}")
    log(f"  Core Member (RA >= {CORE_RA_THRESHOLD})       : {int(core.sum())}")
    log(f"  Ambiguous                       : {int((~core).sum())}")
    log(f"  구 하이브리드 기준과 불일치       : {int((core != hybrid).sum())} 종")
    if core.any():
        log(f"  Core 종의 최소 dominance ratio   : {dom_ratio[core].min():.3f}"
            f"   (>= 4 이면 원고 주장과 일치)")
    per_arch = (pd.Series(o_idx[core, 0] + 1).value_counts()
                .reindex(range(1, K + 1), fill_value=0).tolist())
    log(f"  아키타입별 Core 수 1..{K}         : {per_arch}  (합 {sum(per_arch)})")
    log(f"  second-ranked weight == 0 인 종   : {int((t2 == 0).sum())}"
        f"   <- Figure 3 legend 수치")
    log(f"  할당 도메인이 없는 종             : {len(zero_rows)}")

    # -------------------------------------------------- 종별 프로파일
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
    log(f"\n  DB 종별 프로파일 매핑 완료 (미매칭 {n_miss}종)")
    for nm in df.loc[df["N_Allergens"].isna(), "Display_Name"]:
        log(f"      미매칭: {nm}")

    # ================================================================
    # [2] BIAS
    # ================================================================
    def bias_block(sub, tag):
        log("\n" + "=" * 68)
        log(f"[2] BIAS  annotation artifact 검정  [{tag}]  n = {len(sub)}")
        log("=" * 68)
        e = sub["Entropy"].to_numpy()

        log("\n-- (a) 단순 상관 : 전부 보고 (해석은 아래 분해 검정에서) ------")
        log("     주의: 아래 변수들은 모두 '연구 깊이'와 '실제 다면성'을")
        log("           동시에 담고 있어 단독으로는 해석할 수 없다.")
        for col, label in [
                ("N_Allergens", "entropy vs reference allergen 수"),
                ("N_Families", "entropy vs allergen family 수"),
                ("N_Homologs", "entropy vs homolog 수"),
                ("Feature_Richness", "entropy vs feature richness"),
                ("Total_Bitscore", "entropy vs 총 bitscore")]:
            v = sub[col].to_numpy(float)
            r, p, n, _ = partial_spearman(e, v)
            lo, hi = boot_ci(e, v, seed=1)
            report(label, r, p, n, lo, hi)

        log("\n-- (b) 분해 검정 : 다면성 성분 vs 깊이 성분 -------------------")
        log("     n_allergens = n_families x per_family")
        log("                   (생물학)     (중복 등재 = 깊이)")

        nf = sub[["N_Families"]].to_numpy(float)

        r, p, n, _ = partial_spearman(e, sub["N_Families"].to_numpy(float))
        lo, hi = boot_ci(e, sub["N_Families"].to_numpy(float), seed=2)
        report("[검정 A] entropy vs family 수  (다면성 성분)", r, p, n, lo, hi,
               note="유의할 것으로 기대됨 = repertoire 가 실제로 다면적")

        r, p, n, nc = partial_spearman(e, sub["Per_Family"].to_numpy(float), nf)
        lo, hi = boot_ci(e, sub["Per_Family"].to_numpy(float), nf, seed=3)
        report("[검정 B] entropy vs family당 allergen 수 | family 수  (순수 깊이)",
               r, p, n, lo, hi, n_ctrl=nc,
               note="null 이어야 함 = 같은 다면성이면 더 많이 등재돼도 entropy 불변")

        r, p, n, nc = partial_spearman(e, sub["N_Allergens"].to_numpy(float), nf)
        lo, hi = boot_ci(e, sub["N_Allergens"].to_numpy(float), nf, seed=4)
        report("[검정 C] entropy vs allergen 수 | family 수  (B의 대안 표현)",
               r, p, n, lo, hi, n_ctrl=nc)

        log("\n-- (c) 판정 -------------------------------------------------")
        rB, pB, nB, ncB = partial_spearman(
            e, sub["Per_Family"].to_numpy(float), nf)
        eqB = min([b for b in TOST_BOUNDS
                   if tost_equivalence(rB, nB, ncB, b) < 0.05], default=None)
        rA, pA, _, _ = partial_spearman(e, sub["N_Families"].to_numpy(float))
        if not (np.isfinite(pA) and np.isfinite(pB)):
            log("     >> 검정 A 또는 B 를 계산할 수 없다. 입력 변수 분포를 확인할 것.")
        elif pA < 0.05 and pB >= 0.05 and eqB is not None:
            log("     >> 다면성은 entropy 를 설명하고, 등재 깊이는 설명하지 않는다.")
            log("        원고에 'annotation depth 로 환원되지 않는다' 라고")
            log("        정량적으로 주장 가능.")
        elif pB < 0.05:
            log("     >> 등재 깊이 성분도 entropy 와 유의하게 연관된다.")
            log("        artifact 가능성을 배제할 수 없으므로 Limitation 으로")
            log("        내리고, 임상적 외부 타당성으로 방어할 것.")
        else:
            log("     >> 깊이 성분이 비유의하나 등가성 미성립(검정력 부족).")
            log("        '배제할 수 없다' 수준으로 신중하게 기술할 것.")

    bias_block(df, "전체 종")
    if zero_rows:
        bias_block(df[~df["Display_Name"].isin(zero_rows)].reset_index(drop=True),
                   "할당 도메인 없는 종 제외")

    # ================================================================
    # [3] RAREFACTION
    # ================================================================
    log("\n" + "=" * 68)
    log("[3] RAREFACTION  깊이 균등화 후 entropy 순위 안정성")
    log("=" * 68)
    log("  목적: entropy 순위가 종별 데이터 총량 차이 때문에 생긴 것인지 확인.")
    log("        원 entropy 와의 일치도가 높으면 깊이는 순위를 좌우하지 않는다.")

    counts = raw.to_numpy(float)
    depth = counts.sum(axis=1)
    target = int(np.floor(np.percentile(depth[depth > 0], 10)))
    keep = depth >= target
    log(f"  목표 깊이 = {target:,} (양수 깊이 종의 10 분위수), "
        f"대상 {int(keep.sum())} / {len(depth)} 종")

    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = np.where(keep)[0]
    probs = counts[sub_idx] / counts[sub_idx].sum(axis=1, keepdims=True)
    acc = np.zeros(len(sub_idx))
    for _ in range(RAREFY_REPS):
        sampled = np.array([rng.multinomial(target, p) for p in probs], float)
        Wr = model.transform(
            Normalizer(norm="l1").fit_transform(np.log1p(sampled)))
        Wrn = Wr / Wr.sum(axis=1, keepdims=True).clip(min=1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc += -np.nansum(np.where(Wrn > 0, Wrn * np.log(Wrn), 0.0), axis=1)

    ent_rare = np.full(len(depth), np.nan)
    ent_rare[sub_idx] = acc / RAREFY_REPS
    df["Entropy_Rarefied"] = ent_rare

    ok = df["Entropy_Rarefied"].notna()
    r2, p2 = stats.spearmanr(df.loc[ok, "Entropy"], df.loc[ok, "Entropy_Rarefied"])
    log(f"  원 entropy vs rarefied entropy: rho = {r2:+.3f}, p = {p2:.3g}, "
        f"n = {int(ok.sum())}")
    if r2 > 0.9:
        log("  -> 순위가 거의 그대로 유지된다. entropy 는 데이터 총량이 아니라")
        log("     아키타입 가중치의 분포 형태를 반영한다.")

    rB2, pB2, nB2, ncB2 = partial_spearman(
        df.loc[ok, "Entropy_Rarefied"], df.loc[ok, "Per_Family"],
        df.loc[ok, ["N_Families"]].to_numpy(float))
    loB2, hiB2 = boot_ci(df.loc[ok, "Entropy_Rarefied"].to_numpy(),
                         df.loc[ok, "Per_Family"].to_numpy(),
                         df.loc[ok, ["N_Families"]].to_numpy(float), seed=5)
    report("[검정 B 재확인] rarefied entropy vs family당 allergen 수 | family 수",
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
    log(f"  관측 AMI = {ami:.4f}  (n = {len(sub)}, orders = {sub['Order'].nunique()})")

    rng2 = np.random.default_rng(RANDOM_STATE)
    null = np.array([
        adjusted_mutual_info_score(lab_tax, rng2.permutation(lab_arc),
                                   average_method="arithmetic")
        for _ in range(N_PERM)])
    p_emp = (np.sum(null >= ami) + 1) / (N_PERM + 1)
    log(f"  permutation null: 평균 {null.mean():+.4f}, SD {null.std():.4f}, "
        f"95 분위 {np.percentile(null, 95):+.4f}")
    log(f"  경험적 p = {p_emp:.4g}   z = {(ami - null.mean()) / null.std():.2f}")
    if p_emp < 0.05:
        log("  -> 우연 수준보다 확실히 높으나 절대값은 낮다. 원고에")
        log("     '유의하지만 약한 계통 신호' 라고 양쪽 다 정량 기술 가능.")
    else:
        log("  -> 우연 수준과 구별되지 않는다. '계통과 무관하다' 쪽으로 기술할 것.")

    # -------------------------------------------------- 저장
    df.drop(columns=["key"]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_TXT.write_text("\n".join(_log), encoding="utf-8")
    log(f"\n[DONE] 종별 값 -> {OUT_CSV}")
    log(f"[DONE] 로그     -> {OUT_TXT}")


if __name__ == "__main__":
    main()
