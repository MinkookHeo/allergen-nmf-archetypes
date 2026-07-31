import sqlite3
import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 경로 설정 (config.py 연동)
# ---------------------------------------------------------------------------
# src 폴더의 상위 폴더(프로젝트 루트)에 있는 config를 불러옵니다.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MATRIX_PATH, RESULTS_DIR

OUT_MATRIX = MATRIX_PATH
OUT_REMOVED = RESULTS_DIR / "removed_species.csv"
OUT_SUMMARY = RESULTS_DIR / "matrix_build_summary.txt"

# 대조용 이전 NMF 결과
PREV_NMF_XLSX = RESULTS_DIR / "NMF_Final_Analysis_K6_Step3_Advanced.xlsx"
PREV_NMF_SHEET = "2_Species_Membership"
N_REFERENCE_ALLERGENS = 1149

if not DB_PATH.exists():
    print(f"[ERROR] DB를 찾을 수 없습니다. data 폴더에 DB를 넣어주세요: {DB_PATH}")
    sys.exit(1)

# ===========================================================================
# 1. 데이터 로드
# ===========================================================================
# NOTE: Query_ID 는 GenBank accession 이다. 알레르겐 하나가 여러 accession 을
#       가지므로 Query_ID 의 unique 개수는 알레르겐 수가 아니다.
#       실제 알레르겐 수는 A.rowid(Allergen_UID) 로 센다.
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
    log(f"[ERROR] 데이터 로드 실패: {e}")
    sys.exit(1)

log(f"[OK] 데이터 조립 완료 (행 수: {len(merged):,})")


# ===========================================================================
# 2. 공통 유틸
# ===========================================================================
def composite_id(df):
    """CDD Superfamily + Short name 복합 식별자."""
    return df["Superfamily"].fillna("-") + "|" + df["Short_name"].fillna("Unknown")


def collect_stats(df):
    """논문 Methods 에 들어갈 지표를 dict 로 반환."""
    return {
        "pairs": len(df),
        "homologs": df["Similar_Protein"].nunique(),
        "allergens": df["Allergen_UID"].nunique(),
        "accessions": df["Query_ID"].nunique(),
        "species": df["Source_Species"].nunique(),
        "features": composite_id(df).nunique(),
    }


def report(stats, label):
    log(f"\n--- [{label}] " + "-" * max(4, 46 - len(label)))
    log(f"  allergen-homolog 쌍(행)      : {stats['pairs']:,}")
    log(f"  unique homolog               : {stats['homologs']:,}")
    log(f"  reference allergen (고유)     : {stats['allergens']:,}"
        f"  / 질의 {N_REFERENCE_ALLERGENS:,}")
    log(f"  GenBank accession (회수됨)    : {stats['accessions']:,}")
    log(f"  source species               : {stats['species']:,}")
    log(f"  conserved-domain feature     : {stats['features']:,}")


pre_stats = collect_stats(merged)
report(pre_stats, "필터 전")

if pre_stats["allergens"] > N_REFERENCE_ALLERGENS:
    log(f"[WARN] 고유 알레르겐 수({pre_stats['allergens']:,})가 질의 수"
        f"({N_REFERENCE_ALLERGENS:,})를 초과합니다. "
        f"Allergens 테이블에 중복 행이 있는지 확인하세요.")


# ===========================================================================
# 3. 비식용 종 제거 (속명 정확 매칭)
# ===========================================================================
GENERA_TO_REMOVE = {
    # 1. 벌 / 말벌 / 개미 (Hymenoptera) — 침독 알레르겐
    "Apis", "Vespa", "Vespula", "Bombus", "Polistes", "Polybia", "Solenopsis",
    "Myrmecia", "Dolichovespula", "Anoplolepis", "Linepithema",
    "Pachycondyla",

    # 2. 바퀴벌레 및 집안 해충
    "Blattella", "Periplaneta", "Coptotermes", "Shelfordella", "Supella",

    # 3. 나방 / 누에 등
    #    NOTE: Bombyx(누에)를 식용 곤충으로 포함하려면 아래 목록에서 제거할 것
    "Bombyx", "Plodia", "Ephestia", "Galleria", "Thaumetopoea", "Tineola",

    # 4. 자상 / 환경성 / 흡혈 곤충 및 파리
    "Aedes", "Anopheles", "Culex", "Glossina", "Tabanus", "Musca", "Chironomus",
    "Ctenocephalides", "Cimex", "Triatoma", "Lepisma", "Forcipomyia",
    "Arge", "Lucilia", "Sarcophaga", "Calliphora",

    # 5. 진드기류 (Mites / Ticks)
    "Dermatophagoides", "Tyrophagus", "Euroglyphus", "Glycyphagus",
    "Lepidoglyphus", "Blomia", "Tetranychus", "Argas", "Ixodes", "Acarus",
    "Chortoglyphus", "Sarcoptes",

    # 6. 기생충 및 선충
    "Ascaris", "Anisakis", "Trichuris", "Enterobius", "Strongyloides",
    "Schistosoma",

    # 7. 곰팡이 및 효모
    "Aspergillus", "Alternaria", "Cladosporium", "Penicillium", "Candida",
    "Cochliobolus", "Rhizopus", "Curvularia", "Stachybotrys", "Malassezia",
    "Fusarium", "Epicoccum", "Trichophyton", "Ulocladium", "Rhodotorula",
    "Saccharomyces",

    # 8. 세균
    "Bacillus", "Staphylococcus", "Streptococcus", "Escherichia", "Salmonella",
    "Listeria",

    # 9. 비식용 포유류 및 인간
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

log(f"\n[FILTER] 종 {n_species_before} -> {n_species_after} "
    f"({len(removed_species)}종 / {len(removed_genera)}속 제거)")

# 감사 1: 제거 목록에 있으나 DB 에 존재하지 않는 속명 (오타/부재 탐지)
unmatched = sorted(GENERA_TO_REMOVE - set(removed_genera))
if unmatched:
    log(f"[WARN] DB에서 매칭되지 않은 속명 {len(unmatched)}개 (오타/부재 확인): "
        f"{unmatched}")

# 감사 2: 제거된 종 목록 저장 (Supplementary 재현성용)
pd.DataFrame({
    "removed_species": removed_species,
    "genus": [s.split()[0] if s else "" for s in removed_species],
}).to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")
log(f"[OK] 제거된 종 목록 -> {OUT_REMOVED}")

post_stats = collect_stats(merged)
report(post_stats, "필터 후")


# ===========================================================================
# 4. 표시명 / 복합 키
# ===========================================================================
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


# ===========================================================================
# 5. 행렬 생성
# ===========================================================================
matrix = (
    merged.groupby(["Display_Name", "Composite_ID"])["Bitscore"]
    .sum()
    .unstack()
    .fillna(0)
)
matrix.to_csv(OUT_MATRIX, encoding="utf-8-sig")

log("\n" + "=" * 62)
log(f"[DONE] 최종 매트릭스: {matrix.shape[0]} (종) x {matrix.shape[1]} (단백질군)")
log(f"       저장 -> {OUT_MATRIX}")
log("=" * 62)


# ===========================================================================
# 6. 이전 NMF 결과와 종 목록 대조
# ===========================================================================
log("\n[대조] 이전 NMF 결과와 종 구성 비교")
if not PREV_NMF_XLSX.exists():
    log(f"  건너뜀 - 파일 없음: {PREV_NMF_XLSX}")
else:
    try:
        prev = pd.read_excel(PREV_NMF_XLSX, sheet_name=PREV_NMF_SHEET)
        prev_set = set(prev["Display_Name"].dropna())
        new_set = set(matrix.index)

        added = sorted(new_set - prev_set)
        dropped = sorted(prev_set - new_set)

        log(f"  이전: {len(prev_set)}종  /  현재: {len(new_set)}종")
        log(f"  > 새로 포함된 종 ({len(added)}):")
        for s in added:
            log(f"      + {s}")
        log(f"  > 제외된 종 ({len(dropped)}):")
        for s in dropped:
            log(f"      - {s}")
        if not added and not dropped:
            log("      (변화 없음)")
    except Exception as e:
        log(f"  [WARN] 대조 실패: {e}")


# ===========================================================================
# 7. 논문 Methods 용 요약
# ===========================================================================
pct = post_stats["allergens"] / N_REFERENCE_ALLERGENS * 100

log("\n[Methods에 그대로 넣을 숫자]")
log(f"  reference allergens queried              : {N_REFERENCE_ALLERGENS:,}")
log(f"  reference allergens with >=1 hit (전)     : {pre_stats['allergens']:,}")
log(f"  reference allergens retained (후)         : {post_stats['allergens']:,} "
    f"({pct:.1f}%)")
log(f"  GenBank accessions recovered (전 -> 후)   : "
    f"{pre_stats['accessions']:,} -> {post_stats['accessions']:,}")
log(f"  allergen-homolog relationships (전 -> 후) : "
    f"{pre_stats['pairs']:,} -> {post_stats['pairs']:,}")
log(f"  unique cross-reactive homologs (전 -> 후) : "
    f"{pre_stats['homologs']:,} -> {post_stats['homologs']:,}")
log(f"  source species (전 -> 후)                 : "
    f"{pre_stats['species']:,} -> {post_stats['species']:,}")
log(f"  conserved-domain features (전 -> 후)      : "
    f"{pre_stats['features']:,} -> {post_stats['features']:,}")

OUT_SUMMARY.write_text("\n".join(_log_lines), encoding="utf-8")
print(f"\n[OK] 실행 로그 -> {OUT_SUMMARY}")
