import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. 경로 / 상수 (config.py 연동)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MATRIX_PATH, RESULTS_DIR

IN_MATRIX = MATRIX_PATH

RANDOM_STATE = 42
MAX_ITER = 5000
CORE_RA_THRESHOLD = 0.80

_log = []
def log(m=""):
    print(m)
    _log.append(str(m))


def norm_name(s):
    """'Genus species (common name)' -> 'Genus species'"""
    return str(s).split(" (")[0].strip()

# ===========================================================================
# 1. 데이터 로드 및 전처리
# ===========================================================================
def load_final_matrix(file_path):
    """log1p 변환 후 종(행)별 L1 정규화."""
    path = Path(file_path)
    if not path.exists():
        log(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    df = pd.read_csv(path, index_col=0)

    zero_rows = df.index[df.sum(axis=1) == 0].tolist()
    if zero_rows:
        log(f"[WARN] 할당 가능한 conserved domain 이 없는 종 "
            f"{len(zero_rows)}개: {zero_rows}")
        log("       이 종들은 W 행이 0 이 되어 자동으로 Ambiguous 로 분류된다.")

    matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(df))
    log(f"[OK] 분석 준비 완료: {df.shape[0]} 종 x {df.shape[1]} 단백질군")
    return matrix_norm, df.columns, df.index


# ===========================================================================
# 2. SQLite 분류 정보 매핑  (04_Figure_Generation.py 와 동일 로직)
# ===========================================================================
def get_taxonomy_info(db_path, species_list):
    import sqlite3

    base = pd.DataFrame({"Display_Name": list(species_list)})
    base["key"] = base["Display_Name"].apply(norm_name)

    try:
        with sqlite3.connect(db_path) as conn:
            tax = pd.read_sql_query(
                "SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy", conn)
    except Exception as e:
        log(f"[WARN] DB 매핑 오류: {e}")
        base["Order"] = np.nan
        base["Family"] = np.nan
        return base.drop(columns=["key"])

    tax["key"] = tax["Species"].apply(norm_name)
    # 유효하지 않은 값은 '매핑 전에' 제외한다 (04 와 동일)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
    map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

    base["Order"] = base["key"].map(map_order)
    base["Family"] = base["key"].map(map_family)

    n_miss = int(base["Order"].isna().sum())
    log(f"[OK] taxonomy 매핑: Order 미매칭 {n_miss}종")
    for nm in base.loc[base["Order"].isna(), "Display_Name"]:
        log(f"      미매칭: {nm}")

    return base.drop(columns=["key"])


# ===========================================================================
# 3. 분석 및 리포트 생성
# ===========================================================================
def generate_report(matrix, feature_names, species_names, db_path, k_val):
    log(f"\n{'=' * 62}")
    log(f"[RUN] K={k_val} 상세 분석")
    log("=" * 62)

    model = NMF(n_components=k_val, init="nndsvda",
                max_iter=MAX_ITER, random_state=RANDOM_STATE)
    W = model.fit_transform(matrix)      # 종 x 아키타입
    H = model.components_                # 아키타입 x feature
    log(f"[OK] NMF K={k_val}: W{W.shape}, H{H.shape}")

    # --- 1등 / 2등 아키타입 -------------------------------------------------
    sorted_idx = np.argsort(W, axis=1)[:, ::-1]
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1]
    top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).ravel()
    top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).ravel()

    # --- 지표 ---------------------------------------------------------------
    eps = 1e-9
    dominance_ratio = top1_w / (top2_w + eps)
    row_sums = W.sum(axis=1)
    relative_abundance = top1_w / np.where(row_sums == 0, 1, row_sums)

    # --- 분류: 단일 기준 (원고 2.4) ----------------------------------------
    is_core = relative_abundance >= CORE_RA_THRESHOLD
    membership = np.where(is_core, "Core Member", "Ambiguous")

    # 이전 하이브리드 기준과 결과가 동일한지 자체 검증
    hybrid = (dominance_ratio >= 2.0) & ((top1_w >= 0.3) | (relative_abundance >= 0.8))
    n_diff = int((is_core != hybrid).sum())
    log(f"[CHECK] 단일 기준(RA>={CORE_RA_THRESHOLD}) vs 이전 하이브리드 기준 "
        f"불일치 종: {n_diff}")
    if n_diff == 0 and is_core.any():
        log(f"        두 기준이 동일한 분할을 준다. "
            f"Core 종의 최소 dominance ratio = {dominance_ratio[is_core].min():.3f} "
            f"(>= 4 이면 이론과 일치)")
    elif n_diff:
        log("[WARN] 두 기준이 다른 결과를 준다. 원고 2.4 기술을 재확인할 것.")

    log(f"[결과] 총 {len(species_names)}종 / "
        f"Core Member {int(is_core.sum())} / Ambiguous {int((~is_core).sum())}")
    per_arch = (pd.Series(top1_idx[is_core] + 1)
                .value_counts()
                .reindex(range(1, k_val + 1), fill_value=0)
                .tolist())
    log(f"       아키타입별 Core 수 1..{k_val}: {per_arch}")

    # --- Sheet 1: 아키타입 시그니처 (H 행렬 상위 feature) -------------------
    sig_rows = []
    for k in range(k_val):
        for rank, idx in enumerate(np.argsort(H[k])[::-1][:10], 1):
            sig_rows.append({
                "Archetype": k + 1,
                "Cluster": k,                       # 0-기반 내부 번호 (참고용)
                "Rank": rank,
                "Feature (Proteins)": feature_names[idx],
                "Weight_H": H[k, idx],              # 주: feature 로딩은 H 행렬
            })
    df_sig = pd.DataFrame(sig_rows)

    # --- Sheet 2: 소속 및 분류 ---------------------------------------------
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
        df_tax[f"Weight_A{i + 1}"] = W[:, i]        # 1-기반으로 통일

    # 정렬: 아키타입 -> Core 우선 -> Dominance Ratio 내림차순
    df_tax["_order"] = np.where(df_tax["Membership_Status"] == "Core Member", 0, 1)
    df_tax = (df_tax
              .sort_values(by=["Archetype", "_order", "Dominance_Ratio"],
                           ascending=[True, True, False])
              .drop(columns=["_order"]))

    out_xlsx = RESULTS_DIR / f"NMF_Final_Analysis_K{k_val}_Step3_Advanced.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_sig.to_excel(writer, sheet_name="1_Cluster_Signatures", index=False)
        df_tax.to_excel(writer, sheet_name="2_Species_Membership", index=False)
    log(f"[OK] 보고서 저장 -> {out_xlsx}")


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NMF(K) 종별 아키타입 소속 리포트 생성")
    parser.add_argument("k", nargs="*", type=int, default=[6],
                        help="분석할 K값 (공백 구분, 기본 6). 예: 03_cluster_analysis.py 6 7")
    args = parser.parse_args()
    target_ks = args.k if args.k else [6]
    log(f"[RUN] 대상 K: {target_ks}")

    matrix_norm, features, species = load_final_matrix(IN_MATRIX)

    for k in target_ks:
        generate_report(matrix_norm, features, species, DB_PATH, k)

    (RESULTS_DIR / "cluster_analysis_log.txt").write_text(
        "\n".join(_log), encoding="utf-8")
    log(f"\n[DONE] 로그 -> {RESULTS_DIR / 'cluster_analysis_log.txt'}")
    log("[NEXT] 04_figure_generation.py (그림) -> "
        "05_statistical_analysis.py (통계)")
