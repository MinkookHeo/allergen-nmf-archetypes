# -*- coding: utf-8 -*-
"""
Generate the panels of Figure 2 and assemble them.

Loads precomputed NMF results (W matrix) from 03_cluster_analysis.py
rather than refitting.

Panels:
    A  Top-10 taxonomic orders / families / conserved protein families
    B  Compositional fingerprint of regulated representative species
    C  Order-to-archetype alluvial (native matplotlib and Plotly HTML)
    D  Dominance landscape (relative abundance vs dominance ratio)
"""

import io
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import Normalizer

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.colors import white, black

# ---------------------------------------------------------------------------
# 0. Paths / constants (config.py)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DB_PATH, MATRIX_PATH, RESULTS_DIR, A4DIR,
    NMF_W_PATH, NMF_MEMBERSHIP_PATH,
    K, RANDOM_STATE, CORE_RA_THRESHOLD,
    ARCH_HEX, ARCH_RGBA, CORE_COLOR, AMB_COLOR,
    cap_first, norm_name, set_journal_font,
)

IN_MATRIX = MATRIX_PATH
FIGDIR = RESULTS_DIR

set_journal_font()


# ---------------------------------------------------------------------------
# 1. Data pipeline (loads precomputed NMF from 03)
# ---------------------------------------------------------------------------
def prepare_data():
    raw = pd.read_csv(IN_MATRIX, index_col=0)

    # Load precomputed W matrix; fall back to fitting if .npy is absent.
    if NMF_W_PATH.exists():
        W = np.load(NMF_W_PATH)
        print(f"[OK] Loaded W from {NMF_W_PATH}")
    else:
        from sklearn.decomposition import NMF
        matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(raw))
        model = NMF(n_components=K, init="nndsvda", max_iter=5000,
                    random_state=RANDOM_STATE)
        W = model.fit_transform(matrix_norm)
        print("[WARN] W.npy not found; fitted NMF from scratch. "
              "Run 03_cluster_analysis.py first for reproducibility.")

    order_idx = np.argsort(W, axis=1)[:, ::-1]
    top1_idx, top2_idx = order_idx[:, 0], order_idx[:, 1]
    top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).squeeze()
    top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).squeeze()

    eps = 1e-9
    dominance_ratio = top1_w / (top2_w + eps)
    row_sum = W.sum(axis=1)
    relative_abundance = top1_w / np.where(row_sum == 0, 1, row_sum)

    is_core = relative_abundance >= CORE_RA_THRESHOLD
    status = np.where(is_core, "Core Member", "Ambiguous")

    df_nmf = pd.DataFrame({
        "Display_Name": raw.index,
        "Primary_Cluster": top1_idx,
        "Archetype": top1_idx + 1,
        "Relative_Abundance": relative_abundance,
        "Dominance_Ratio": dominance_ratio,
        "Membership_Status": status,
    })
    for i in range(K):
        df_nmf[f"Weight_A{i+1}"] = W[:, i]

    with sqlite3.connect(DB_PATH) as conn:
        tax = pd.read_sql_query(
            "SELECT Species, [Order] AS Ord, Family FROM SpeciesTaxonomy", conn)
        cr = pd.read_sql_query(
            "SELECT Query_ID, Similar_Protein FROM Crossreactivity", conn)
        pf = pd.read_sql_query(
            'SELECT Query, Superfamily, "Short name" AS Short_name '
            'FROM Protein_families', conn)
        alg = pd.read_sql_query("SELECT genbank_ids, species FROM Allergens", conn)

    tax["key"] = tax["Species"].apply(norm_name)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
    map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

    df_nmf["key"] = df_nmf["Display_Name"].apply(norm_name)
    df_nmf["Order"] = df_nmf["key"].map(map_order)
    df_nmf["Family"] = df_nmf["key"].map(map_family)

    pf["Superfamily"] = pf["Superfamily"].fillna("-")
    pf["Short_name"] = pf["Short_name"].fillna("Unknown")
    pf["combo"] = pf["Superfamily"] + "|" + pf["Short_name"]
    hom = cr.merge(pf, left_on="Similar_Protein", right_on="Query", how="left")
    top_protfam = hom["combo"].value_counts().head(10)

    alg2 = alg.assign(gid=alg["genbank_ids"].str.split(";")).explode("gid")
    alg2["gid"] = alg2["gid"].str.strip()
    cr2 = cr.merge(alg2, left_on="Query_ID", right_on="gid", how="left")
    cr2["key"] = cr2["species"].apply(lambda x: norm_name(x) if pd.notna(x) else x)
    cr2 = cr2.merge(tax[["key", "Ord", "Family"]].drop_duplicates("key"),
                    on="key", how="left")

    top_order = cr2[~cr2["Ord"].isin(["Not Found", "Unknown", None])]["Ord"].value_counts().head(10)
    top_family = cr2[~cr2["Family"].isin(["Not Found", "Unknown", None])]["Family"].value_counts().head(10)

    top_order.index = [cap_first(x) for x in top_order.index]
    top_family.index = [cap_first(x) for x in top_family.index]

    df_nmf.to_csv(FIGDIR / "figure_source_data.csv", index=False,
                  encoding="utf-8-sig")

    return df_nmf, top_order, top_family, top_protfam


# ---------------------------------------------------------------------------
# 2. Panel modules
# ---------------------------------------------------------------------------
def draw_panel_A(axes, top_order, top_family, top_protfam, is_individual=False):
    sns.barplot(x=top_order.values, y=top_order.index, color=ARCH_HEX[0], ax=axes[0])
    axes[0].set_xlabel('Number of species'); axes[0].set_ylabel('')
    for lbl in axes[0].get_yticklabels():
        lbl.set_fontstyle('italic')

    sns.barplot(x=top_family.values, y=top_family.index, color=ARCH_HEX[1], ax=axes[1])
    axes[1].set_xlabel('Number of species'); axes[1].set_ylabel('')
    for lbl in axes[1].get_yticklabels():
        lbl.set_fontstyle('italic')

    sns.barplot(x=top_protfam.values, y=top_protfam.index, color=ARCH_HEX[2], ax=axes[2])
    axes[2].set_xlabel('Number of homologs'); axes[2].set_ylabel('')

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    if is_individual:
        axes[0].set_title("Top 10 taxonomic orders")
        axes[1].set_title("Top 10 taxonomic families")
        axes[2].set_title("Top 10 conserved protein families")
    else:
        axes[0].text(-0.05, 1.05, '(A)', transform=axes[0].transAxes,
                     fontsize=18, va='bottom', ha='right')


def draw_panel_B(ax, df_nmf, is_individual=False):
    REGULATED_REPS = [
        ("Anacardium occidentale",      "Cashew"),
        ("Sesamum indicum",             "Sesame"),
        ("Brassica juncea",             "Brown mustard"),
        ("Litopenaeus vannamei",        "Pacific white shrimp"),
        ("Octopus vulgaris",            "Common octopus"),
        ("Lupinus angustifolius",       "Lupin"),
        ("Triticum turgidum ssp durum", "Durum wheat"),
        ("Castanea sativa",             "Chestnut"),
        ("Gadus morhua",                "Atlantic cod"),
    ]
    wc = [f"Weight_A{i+1}" for i in range(K)]
    rows_b = []
    for key, label in REGULATED_REPS:
        m = df_nmf[df_nmf["Display_Name"].str.contains(key, na=False, regex=False)]
        if m.empty:
            continue
        r = m.iloc[0]; w = r[wc].to_numpy(dtype=float)
        frac = w / w.sum()
        dr = r["Dominance_Ratio"]; dr_str = "> 10³" if dr > 1000 else f"{dr:.1f}"
        rows_b.append((label, frac, dr_str, float(frac.max())))

    rows_b.sort(key=lambda x: x[3])
    for i, (label, frac, dr_str, _) in enumerate(rows_b):
        left = 0.0
        for a in range(K):
            if frac[a] > 0:
                ax.barh(i, frac[a], left=left, color=ARCH_HEX[a],
                        edgecolor="white", linewidth=0.6, height=0.62)
                left += frac[a]
        ax.text(1.02, i, f"D.R: {dr_str}", va="center", ha="left", fontsize=11)

    ax.set_yticks(range(len(rows_b)))
    ax.set_yticklabels([r[0] for r in rows_b], fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Relative Archetype Contribution", fontsize=11)

    handles = [Patch(facecolor=ARCH_HEX[a], label=f"Archetype {a+1}") for a in range(K)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if not is_individual:
        ax.text(-0.05, 1.05, '(B)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


def _ribbon(ax, xL, xR, l_top, l_bot, r_top, r_bot, color, alpha=0.5):
    cx = (xL + xR) / 2.0
    verts = [(xL, l_top), (cx, l_top), (cx, r_top), (xR, r_top), (xR, r_bot),
             (cx, r_bot), (cx, l_bot), (xL, l_bot), (xL, l_top)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", alpha=alpha))


def draw_panel_C_native(ax, df_nmf, is_individual=False):
    target_orders = ["Poales", "Rosales", "Fagales", "Fabales", "Asterales",
                     "Malpighiales", "Lamiales", "Malvales", "Zingiberales"]
    sub = df_nmf[df_nmf["Order"].isin(target_orders)]
    agg = sub.groupby(["Order", "Archetype"]).size().reset_index(name="Count")
    if agg.empty:
        ax.axis("off"); return

    orders = [o for o in target_orders if o in set(agg["Order"])]
    arch_list = sorted(int(a) for a in agg["Archetype"].unique())
    total = float(agg["Count"].sum())
    order_tot = agg.groupby("Order")["Count"].sum()
    arch_tot = agg.groupby("Archetype")["Count"].sum()

    gap = 0.03
    nL, nR = len(orders), len(arch_list)
    scale = min((1 - gap * (nL - 1)) / total, (1 - gap * (nR - 1)) / total)

    def node_pos(items, totals, n):
        used = total * scale + gap * (n - 1)
        y = 1 - (1 - used) / 2.0
        pos = {}
        for it in items:
            h = float(totals[it]) * scale
            pos[it] = [y, y - h]
            y -= h + gap
        return pos

    posL = node_pos(orders, order_tot, nL)
    posR = node_pos(arch_list, arch_tot, nR)

    xL0, xL1, xR0, xR1 = 0.00, 0.03, 0.97, 1.00
    offL = {o: posL[o][0] for o in orders}
    offR = {a: posR[a][0] for a in arch_list}

    for o in orders:
        rows = agg[agg["Order"] == o].sort_values("Archetype")
        for _, r in rows.iterrows():
            a = int(r["Archetype"]); h = float(r["Count"]) * scale
            lt = offL[o]; lb = lt - h; offL[o] = lb
            rt = offR[a]; rb = rt - h; offR[a] = rb
            _ribbon(ax, xL1, xR0, lt, lb, rt, rb, ARCH_HEX[a - 1], alpha=0.5)

    for o in orders:
        t, b = posL[o]
        ax.add_patch(Rectangle((xL0, b), xL1 - xL0, t - b,
                               facecolor="lightgrey", edgecolor="none"))
        ax.text(xL0 - 0.02, (t + b) / 2, cap_first(o), ha="right", va="center",
                fontsize=11, fontstyle="italic")
    for a in arch_list:
        t, b = posR[a]
        ax.add_patch(Rectangle((xR0, b), xR1 - xR0, t - b,
                               facecolor=ARCH_HEX[a - 1], edgecolor="none"))
        ax.text(xR1 + 0.02, (t + b) / 2, f"Archetype {a}", ha="left",
                va="center", fontsize=11)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    if not is_individual:
        ax.text(-0.05, 1.05, '(C)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


def draw_panel_D(ax, df_nmf, is_individual=False):
    sns.scatterplot(data=df_nmf, x="Relative_Abundance", y="Dominance_Ratio",
                    hue="Membership_Status",
                    palette={"Core Member": CORE_COLOR, "Ambiguous": AMB_COLOR},
                    alpha=0.75, edgecolor="k", s=60, ax=ax)
    ax.axvline(0.80, color="gray", ls="--", lw=1.5, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Relative abundance of primary archetype\n(Core if ≥ 0.8)")
    ax.set_ylabel("Dominance ratio (log scale)")
    ax.get_legend().remove()
    ax.text(0.90, ax.get_ylim()[1] * 0.4, "Core\n(single-family dominant)",
            color=CORE_COLOR, fontsize=11, ha="center", va="top")
    ax.text(0.35, 10**3, "Ambiguous\n(multi-family)",
            color=AMB_COLOR, fontsize=11, ha="center", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)

    if is_individual:
        ax.set_title("The Dominance Landscape")
    else:
        ax.text(-0.05, 1.05, '(D)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


# ---------------------------------------------------------------------------
# 3. Figure assembly
# ---------------------------------------------------------------------------
def create_figure_2a_independent(top_order, top_family, top_protfam):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_panel_A(axes, top_order, top_family, top_protfam, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(a)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2d_independent(df_nmf):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_panel_D(ax, df_nmf, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(d)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2c_alluvial_html(df):
    target_orders = ["Poales", "Rosales", "Fagales", "Fabales", "Asterales",
                     "Malpighiales", "Lamiales", "Malvales", "Zingiberales"]
    sub = df[df["Order"].isin(target_orders)]
    agg = sub.groupby(["Order", "Archetype"]).size().reset_index(name="Count")
    orders = list(agg["Order"].unique())
    arches = [f"Archetype {i}" for i in sorted(agg["Archetype"].unique())]
    nodes = orders + arches
    nidx = {n: i for i, n in enumerate(nodes)}

    node_colors = [ARCH_HEX[int(n.split(" ")[1]) - 1] if "Archetype" in n
                   else "lightgrey" for n in nodes]
    link_colors = [ARCH_RGBA[int(t_arch) - 1] for t_arch in agg["Archetype"]]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=nodes, color=node_colors),
        link=dict(source=agg["Order"].map(nidx),
                  target=agg["Archetype"].apply(lambda x: f"Archetype {x}").map(nidx),
                  value=agg["Count"], color=link_colors)
    )])
    fig.update_layout(
        title_text="Concordance and divergence between biological phylogeny "
                   "and structural allergenic archetypes",
        font_size=18, width=1400, height=900)
    out_html = FIGDIR / "Figure_2(c)_alluvial.html"
    fig.write_html(str(out_html))
    return out_html


def create_figure_2_combined(df_nmf, top_order, top_family, top_protfam):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"width_ratios": [1, 1.4, 1]})

    draw_panel_A([axes[0, 0], axes[0, 1], axes[0, 2]],
                 top_order, top_family, top_protfam, is_individual=False)
    draw_panel_B(axes[1, 0], df_nmf, is_individual=False)
    draw_panel_C_native(axes[1, 1], df_nmf, is_individual=False)
    draw_panel_D(axes[1, 2], df_nmf, is_individual=False)

    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    return out_pdf


# ---------------------------------------------------------------------------
# 4. A4 formatting (post-processing)
# ---------------------------------------------------------------------------
def apply_a4_formatting(pdf_tasks):
    MM = 72.0 / 25.4
    A4_W, A4_H = 210 * MM, 297 * MM
    margin = 15 * MM
    legend_space = 45 * MM

    for src_pdf_path, fig_label in pdf_tasks:
        if not src_pdf_path.exists():
            continue

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(A4_W, A4_H))
        c.setFillColor(white)
        c.rect(0, 0, A4_W, A4_H, fill=1, stroke=0)
        c.setFillColor(black)
        c.setFont("Helvetica", 10)
        c.drawRightString(A4_W - margin, margin, f"{fig_label}_Heo and Rhee")
        c.showPage()
        c.save()
        buf.seek(0)

        src = PdfReader(src_pdf_path)
        sp = src.pages[0]

        left, bottom = float(sp.mediabox.left), float(sp.mediabox.bottom)
        sw, sh = float(sp.mediabox.width), float(sp.mediabox.height)

        avail_w = A4_W - 2 * margin
        avail_h = A4_H - margin - (margin + legend_space)
        scale = min(avail_w / sw, avail_h / sh)

        fw, fh = sw * scale, sh * scale
        tx = margin + (avail_w - fw) / 2.0
        ty = (A4_H - margin) - fh

        op = (Transformation()
              .translate(-left, -bottom)
              .scale(scale)
              .translate(tx, ty))

        base = PdfReader(buf)
        page = base.pages[0]
        page.merge_transformed_page(sp, op)

        final_pdf_path = A4DIR / f"{fig_label.replace(' ', '_')}.pdf"
        writer = PdfWriter()
        writer.add_page(page)
        with open(final_pdf_path, "wb") as f:
            writer.write(f)

        # Keep only the A4-formatted output.
        os.remove(src_pdf_path)
        print(f"[A4] {final_pdf_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("1. Loading data and NMF results...")
    df_nmf, top_order, top_family, top_protfam = prepare_data()

    print("2. Building individual panels...")
    fig2a_pdf = create_figure_2a_independent(top_order, top_family, top_protfam)
    fig2d_pdf = create_figure_2d_independent(df_nmf)
    fig2c_html = create_figure_2c_alluvial_html(df_nmf)
    print(f"[HTML] {fig2c_html.name}")

    print("3. Assembling the combined 2x3 Figure 2...")
    fig2_pdf = create_figure_2_combined(df_nmf, top_order, top_family, top_protfam)

    print("4. Applying A4 formatting and author caption...")
    pdf_tasks = [
        (fig2a_pdf, "Figure 2(a)"),
        (fig2d_pdf, "Figure 2(d)"),
        (fig2_pdf,  "Figure 2"),
    ]
    apply_a4_formatting(pdf_tasks)

    print(f"\n[DONE] Figures saved to {FIGDIR}")
