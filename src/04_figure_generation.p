import os
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. 경로 / 상수 (config.py 연동)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, MATRIX_PATH, FIG_DIR, RESULTS_DIR

IN_MATRIX = MATRIX_PATH

K = 6
RANDOM_STATE = 42

FIGNUM = {
    "barplot": 1,      
    "landscape": 3,    
    "fingerprint": 4,  
    "alluvial": 5,     
}

_log = []
def log(m=""):
    print(m)
    _log.append(str(m))

# ===========================================================================
# 1. 로드 + 전처리 + NMF
# ===========================================================================
raw = pd.read_csv(IN_MATRIX, index_col=0)
species_names = raw.index
feature_names = raw.columns

_zero = raw.index[raw.sum(axis=1) == 0].tolist()
if _zero:
    log(f"[WARN] 할당 가능한 conserved domain 이 없는 종 {len(_zero)}개: {_zero}")
    log("       이 종들은 W 행이 0 이 되어 자동으로 Ambiguous 로 분류된다.")
    log("       원고 Results/Limitations 에 명시할 것.")

matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(raw))

model = NMF(n_components=K, init="nndsvda", max_iter=5000,
            random_state=RANDOM_STATE)
W = model.fit_transform(matrix_norm)
H = model.components_
log(f"[OK] NMF K={K}: W{W.shape}, H{H.shape}")

order_idx = np.argsort(W, axis=1)[:, ::-1]
top1_idx = order_idx[:, 0]
top2_idx = order_idx[:, 1]
top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).squeeze()
top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).squeeze()

eps = 1e-9
dominance_ratio = top1_w / (top2_w + eps)
row_sum = W.sum(axis=1)
relative_abundance = top1_w / np.where(row_sum == 0, 1, row_sum)

# --- 단일 기준 (원고 2.4와 동일): 상대 비중 >= 0.80 ---
is_core = relative_abundance >= 0.80
status = np.where(is_core, "Core Member", "Ambiguous")

# 이전 하이브리드 기준과 결과가 동일한지 자체 검증
_hybrid = (dominance_ratio >= 2.0) & ((top1_w >= 0.3) | (relative_abundance >= 0.8))
_n_diff = int((is_core != _hybrid).sum())
log(f"[CHECK] 단일 기준 vs 이전 하이브리드 기준 불일치 종: {_n_diff}")
if _n_diff == 0:
    _min_dr = float(dominance_ratio[is_core].min())
    log(f"        두 기준이 동일한 분할을 준다. "
        f"Core 종의 최소 dominance ratio = {_min_dr:.3f} (>= 4 이면 이론과 일치)")
else:
    log("[WARN] 두 기준이 다른 결과를 준다. 원고 기술을 재확인할 것.")

df = pd.DataFrame({
    "Display_Name": species_names,
    "Primary_Cluster": top1_idx,
    "Archetype": top1_idx + 1,               # 1..6
    "Top1_Weight": top1_w,
    "Top2_Weight": top2_w,
    "Relative_Abundance": relative_abundance,
    "Dominance_Ratio": dominance_ratio,
    "Membership_Status": status,
})
for i in range(K):
    df[f"Weight_A{i+1}"] = W[:, i]

log(f"[OK] Core {int(is_core.sum())} / Ambiguous {int((~is_core).sum())}")
per_arch = df[df.Membership_Status == "Core Member"]["Archetype"].value_counts().sort_index()
log(f"     Core per Archetype 1..6: {per_arch.reindex(range(1,7), fill_value=0).tolist()}")


# ===========================================================================
# 2. Taxonomy 매핑 (괄호 앞 학명 기준, Not Found 최소화)
# ===========================================================================
with sqlite3.connect(DB_PATH) as conn:
    tax = pd.read_sql_query(
        "SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy", conn)

tax["key"] = tax["Species"].apply(norm_name)
valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

df["key"] = df["Display_Name"].apply(norm_name)
df["Order"] = df["key"].map(map_order)
df["Family"] = df["key"].map(map_family)

n_nf = df["Order"].isna().sum()
log(f"[OK] taxonomy 매핑: Order 미매칭 {n_nf}종")
for nm in df.loc[df["Order"].isna(), "Display_Name"]:
    log(f"      미매칭: {nm}")


# ===========================================================================
# 3. Figure 1 : Top10 막대 (order / family / protein-family)
# ===========================================================================
def fig_dataset_overview():
    """
    homolog 데이터 기반 Top10 막대.
    Crossreactivity(homolog) 각각의 출처 종 -> Order/Family, 그리고
    homolog의 conserved-domain family 분포.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cr = pd.read_sql_query(
            'SELECT Query_ID, Similar_Protein FROM Crossreactivity', conn)
        pf = pd.read_sql_query(
            'SELECT Query, Superfamily, "Short name" AS Short_name '
            'FROM Protein_families', conn)
        alg = pd.read_sql_query(
            'SELECT genbank_ids, species FROM Allergens', conn)
        taxa = pd.read_sql_query(
            'SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy', conn)

    # homolog의 conserved-domain family 분포
    pf["Superfamily"] = pf["Superfamily"].fillna("-")
    pf["Short_name"] = pf["Short_name"].fillna("Unknown")
    pf["combo"] = pf["Superfamily"] + "|" + pf["Short_name"]
    hom = cr.merge(pf, left_on="Similar_Protein", right_on="Query", how="left")
    top_protfam = hom["combo"].value_counts().head(10)

    # homolog 출처 종의 order/family (레퍼런스 알레르겐 종 기준)
    # genbank_ids 를 분해해 Query_ID 와 매칭
    alg2 = alg.assign(gid=alg["genbank_ids"].str.split(";")).explode("gid")
    alg2["gid"] = alg2["gid"].str.strip()
    cr2 = cr.merge(alg2, left_on="Query_ID", right_on="gid", how="left")
    taxa["key"] = taxa["Species"].apply(norm_name)
    cr2["key"] = cr2["species"].apply(lambda x: norm_name(x) if pd.notna(x) else x)
    cr2 = cr2.merge(taxa[["key", "Ord", "Family"]].drop_duplicates("key"),
                    on="key", how="left")
    valid_o = cr2[~cr2["Ord"].isin(["Not Found", "Unknown", None])]
    top_order = valid_o["Ord"].value_counts().head(10)
    valid_f = cr2[~cr2["Family"].isin(["Not Found", "Unknown", None])]
    top_family = valid_f["Family"].value_counts().head(10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ser, title, color in [
        (axes[0], top_order, "Top 10 taxonomic orders", "#4c78a8"),
        (axes[1], top_family, "Top 10 taxonomic families", "#54a24b"),
        (axes[2], top_protfam, "Top 10 conserved protein families", "#e45756"),
    ]:
        ser = ser[::-1]
        ax.barh(range(len(ser)), ser.values, color=color)
        ax.set_yticks(range(len(ser)))
        ax.set_yticklabels(ser.index, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Number of homologs")
    plt.tight_layout()
    out = FIGDIR / f"Figure{FIGNUM['barplot']}_family_order_barplot.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[OK] Figure {FIGNUM['barplot']} (barplot) -> {out}")

    # 캡션용 총 개수
    log(f"     homolog 총계로 본 order={cr2[~cr2['Ord'].isin(['Not Found','Unknown',None])]['Ord'].nunique()}, "
        f"family={cr2[~cr2['Family'].isin(['Not Found','Unknown',None])]['Family'].nunique()}, "
        f"protein-family(combo)={hom['combo'].nunique()}")


# ===========================================================================
# 4. Figure 4 : Dominance landscape (하이브리드 기준)
# ===========================================================================
def fig_dominance_landscape():
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=df, x="Relative_Abundance", y="Dominance_Ratio",
        hue="Membership_Status",
        palette={"Core Member": "#e74c3c", "Ambiguous": "#3498db"},
        alpha=0.75, edgecolor="k", s=60, ax=ax,
    )
    # 분류 기준은 RA >= 0.80 단 하나 (세로 파선).
    # DR >= 4 는 그로부터 수학적으로 강제되는 결과일 뿐이므로 점선으로 구분 표시.
    ax.axvline(0.80, color="gray", ls="--", lw=1.5, zorder=0,
               label="Core Member threshold (RA = 0.80)")
    ax.axhline(4.0, color="gray", ls=":", lw=1.2, zorder=0,
               label="Implied bound (DR = 4)")
    ax.set_yscale("log")
    ax.set_title("The Dominance Landscape")
    ax.set_xlabel("Relative abundance of primary archetype")
    ax.set_ylabel("Dominance ratio (Top 1 / Top 2 weight, log scale)")
    plt.tight_layout()
    out = FIGDIR / f"Figure{FIGNUM['landscape']}_dominance_landscape.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[OK] Figure {FIGNUM['landscape']} (dominance landscape) -> {out}")


# ===========================================================================
# 5. Figure 5 : Compositional fingerprint
# ===========================================================================
def fig_compositional_fingerprint():
    targets = [
        "Gadus morhua (Atlantic cod)",
        "Litopenaeus vannamei (Pacific white shrimp)",
        "Betula verrucosa (Betula pendula)",
        "Malus domestica (apple)",
        "Corylus avellana (European hazelnut)",
        "Prunus avium (Sweet cherry)",
        "Arachis hypogaea (Peanut)",
    ]
    d = df[df["Display_Name"].isin(targets)].copy()
    wc = [f"Weight_A{i+1}" for i in range(K)]
    d[wc] = d[wc].div(d[wc].sum(axis=1), axis=0)
    d = d.sort_values("Membership_Status", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(d))
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    for i, col in enumerate(wc):
        ax.barh(d["Display_Name"], d[col], left=bottom, color=colors[i],
                label=f"Archetype {i+1}")
        bottom += d[col].to_numpy()
    for idx, (_, row) in enumerate(d.iterrows()):
        ax.text(1.01, idx, f" D.R: {row['Dominance_Ratio']:.1f}",
                va="center", fontsize=10)
    ax.set_title("Compositional Fingerprint of Clinical Model Species", pad=20)
    ax.set_xlabel("Relative archetype contribution")
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
    plt.tight_layout()
    out = FIGDIR / f"Figure{FIGNUM['fingerprint']}_compositional_fingerprint.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"[OK] Figure {FIGNUM['fingerprint']} (fingerprint) -> {out}")


# ===========================================================================
# 6. Figure 6 : Sankey/alluvial
# ===========================================================================
def fig_alluvial_taxonomy():
    target_orders = ["Poales", "Rosales", "Fagales", "Fabales", "Asterales",
                     "Malpighiales", "Lamiales", "Malvales", "Zingiberales"]
    sub = df[df["Order"].isin(target_orders)]
    agg = sub.groupby(["Order", "Archetype"]).size().reset_index(name="Count")

    orders = list(agg["Order"].unique())
    arches = [f"Archetype {i}" for i in sorted(agg["Archetype"].unique())]
    nodes = orders + arches
    nidx = {n: i for i, n in enumerate(nodes)}

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20,
                  line=dict(color="black", width=0.5),
                  label=nodes, color="lightgrey"),
        link=dict(
            source=agg["Order"].map(nidx),
            target=agg["Archetype"].apply(lambda x: f"Archetype {x}").map(nidx),
            value=agg["Count"],
            color="rgba(100,149,237,0.4)"),
    )])
    fig.update_layout(
        title_text="Concordance and Divergence Between Biological Phylogeny "
                   "and Structural Allergenic Archetypes",
        font_size=18, width=1400, height=900)
    out = FIGDIR / f"Figure{FIGNUM['alluvial']}_sankey_taxonomy.html"
    fig.write_html(str(out))
    log(f"[OK] Figure {FIGNUM['alluvial']} (alluvial) -> {out}")


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == "__main__":
    # 그림만 생성한다. AMI / 편상관 등 통계 검정은
    # 05_Annotation_Bias_Check.py 에서 수행한다.
    fig_dataset_overview()
    fig_dominance_landscape()
    fig_compositional_fingerprint()
    fig_alluvial_taxonomy()

    df.to_csv(FIGDIR / "figure_source_data.csv", index=False,
              encoding="utf-8-sig")
    (RESULTS / "figure_generation_log.txt").write_text(
        "\n".join(_log), encoding="utf-8")
    log(f"\n[DONE] 그림 -> {FIGDIR}")
    log("[NEXT] 통계 검정은 05_Annotation_Bias_Check.py 를 실행할 것")
