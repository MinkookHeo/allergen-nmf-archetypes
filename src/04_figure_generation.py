# -*- coding: utf-8 -*-
"""
04_figure_generation.py

Loads the NMF results (W/H saved by 03) and generates all Figure 2 panels.
Also performs the A4 post-processing (margin crop + composition) of the
Figure 3 tree PDF.

Figure 2 panels
  (A) Top-10 barplot x 3 (Order / Family / Protein family)
  (B) Core vs Distributed scatter (dominance landscape)
  (C) Regulated-allergen composition (stacked bar)
  (D) Alluvial diagram (soft-clustering)
"""

import io
import os
import sqlite3

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

from config import (
    DB_PATH, MATRIX_PATH, RESULTS_DIR, FIGDIR, A4DIR, PHYLO_DIR,
    NMF_W_PATH,
    K, RANDOM_STATE, CORE_RA_THRESHOLD,
    ARCH_HEX, ARCH_RGBA, CORE_COLOR, DIST_COLOR,
    TAXONOMY_CORRECTIONS, MANUAL_ORDERS,
    cap_first, norm_name, set_journal_font,
)

# ===========================================================================
# 0. Font setup
# ===========================================================================
set_journal_font()

# ===========================================================================
# 1. Data pipeline (loads NMF results)
# ===========================================================================
def prepare_data():
    raw = pd.read_csv(MATRIX_PATH, index_col=0)

    # Load W saved by 03 instead of recomputing NMF.
    if NMF_W_PATH.exists():
        W = np.load(NMF_W_PATH)
        print(f"[OK] NMF W loaded: {NMF_W_PATH.name}  shape={W.shape}")
    else:
        from sklearn.decomposition import NMF as _NMF
        print("[WARN] W.npy missing -> recomputing NMF (run 03 first)")
        matrix_norm = Normalizer(norm="l1").fit_transform(np.log1p(raw))
        model = _NMF(n_components=K, init="nndsvda",
                     max_iter=5000, random_state=RANDOM_STATE)
        W = model.fit_transform(matrix_norm)

    order_idx = np.argsort(W, axis=1)[:, ::-1]
    top1_idx, top2_idx = order_idx[:, 0], order_idx[:, 1]
    top1_w = np.take_along_axis(W, top1_idx[:, None], axis=1).ravel()
    top2_w = np.take_along_axis(W, top2_idx[:, None], axis=1).ravel()

    eps = 1e-9
    dominance_ratio = top1_w / (top2_w + eps)
    row_sum = W.sum(axis=1)
    relative_abundance = top1_w / np.where(row_sum == 0, 1, row_sum)

    is_core = relative_abundance >= CORE_RA_THRESHOLD
    status = np.where(is_core, "Core Member", "Distributed")

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
            'SELECT Query_ID, Similar_Protein FROM Crossreactivity', conn)
        pf = pd.read_sql_query(
            'SELECT Query, Superfamily, "Short name" AS Short_name '
            'FROM Protein_families', conn)
        alg = pd.read_sql_query(
            'SELECT genbank_ids, species FROM Allergens', conn)

    tax["key"] = tax["Species"].apply(norm_name)
    valid = ~tax["Ord"].isin(["Not Found", "Unknown", None])
    map_order = tax[valid].drop_duplicates("key").set_index("key")["Ord"]
    map_family = tax[valid].drop_duplicates("key").set_index("key")["Family"]

    df_nmf["key"] = df_nmf["Display_Name"].apply(norm_name)

    # taxonomy-only key (the "key" column used by 07 is left as-is)
    df_nmf["tax_key"] = df_nmf["key"].replace(TAXONOMY_CORRECTIONS)
    df_nmf["Order"] = df_nmf["tax_key"].map(map_order)
    df_nmf["Family"] = df_nmf["tax_key"].map(map_family)

    # Fill species unresolved by the DB with the manual overrides.
    n_before = int(df_nmf["Order"].isna().sum())
    df_nmf["Order"] = df_nmf["Order"].fillna(df_nmf["tax_key"].map(MANUAL_ORDERS))
    n_manual = n_before - int(df_nmf["Order"].isna().sum())
    if n_manual:
        print(f"[TAX] filled by MANUAL_ORDERS: {n_manual}")
    n_miss = int(df_nmf["Order"].isna().sum())
    print(f"[TAX] {n_miss} species without Order")
    for nm in df_nmf.loc[df_nmf["Order"].isna(), "Display_Name"]:
        print(f"      unmatched: {nm}")

    pf["Superfamily"] = pf["Superfamily"].fillna("-")
    pf["Short_name"] = pf["Short_name"].fillna("Unknown")
    pf["combo"] = pf["Superfamily"] + "|" + pf["Short_name"]
    hom = cr.merge(pf, left_on="Similar_Protein", right_on="Query", how="left")
    top_protfam = hom["combo"].value_counts().head(10)

    alg2 = alg.assign(gid=alg["genbank_ids"].str.split(";")).explode("gid")
    alg2["gid"] = alg2["gid"].str.strip()
    cr2 = cr.merge(alg2, left_on="Query_ID", right_on="gid", how="left")
    cr2["key"] = cr2["species"].apply(lambda x: norm_name(x) if pd.notna(x) else x)
    cr2 = cr2.merge(
        tax[["key", "Ord", "Family"]].drop_duplicates("key"), on="key", how="left")

    top_order = cr2[~cr2["Ord"].isin(
        ["Not Found", "Unknown", None])]["Ord"].value_counts().head(10)
    top_family = cr2[~cr2["Family"].isin(
        ["Not Found", "Unknown", None])]["Family"].value_counts().head(10)

    top_order.index = [cap_first(x) for x in top_order.index]
    top_family.index = [cap_first(x) for x in top_family.index]

    df_nmf.to_csv(FIGDIR / "figure_source_data.csv",
                  index=False, encoding="utf-8-sig")

    return df_nmf, top_order, top_family, top_protfam


# ===========================================================================
# 2. Individual panels
# ===========================================================================
def draw_panel_A(axes, top_order, top_family, top_protfam, is_individual=False):
    sns.barplot(x=top_order.values, y=top_order.index,
                color=ARCH_HEX[0], ax=axes[0])
    axes[0].set_xlabel('Number of homologs'); axes[0].set_ylabel('')
    for lbl in axes[0].get_yticklabels():
        lbl.set_fontstyle('italic')

    sns.barplot(x=top_family.values, y=top_family.index,
                color=ARCH_HEX[1], ax=axes[1])
    axes[1].set_xlabel('Number of homologs'); axes[1].set_ylabel('')
    for lbl in axes[1].get_yticklabels():
        lbl.set_fontstyle('italic')

    sns.barplot(x=top_protfam.values, y=top_protfam.index,
                color=ARCH_HEX[2], ax=axes[2])
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
    sns.scatterplot(
        data=df_nmf, x="Relative_Abundance", y="Dominance_Ratio",
        hue="Membership_Status",
        palette={"Core Member": CORE_COLOR, "Distributed": DIST_COLOR},
        alpha=0.75, edgecolor="k", s=60, ax=ax)
    ax.axvline(0.80, color="gray", ls="--", lw=1.5, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Relative abundance of primary archetype\n(Core if >= 0.8)")
    ax.set_ylabel("Dominance ratio (log scale)")
    ax.get_legend().remove()

    ax.text(0.58, 10**7.5, "Core\n(single-family dominant)",
            color=CORE_COLOR, fontsize=11, ha="center", va="center")
    ax.text(0.35, 10**3, "Distributed\n(multi-family)",
            color=DIST_COLOR, fontsize=11, ha="center", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)

    if is_individual:
        ax.set_title("The Dominance Landscape")
    else:
        ax.text(-0.05, 1.05, '(B)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


def draw_panel_C(ax, df_nmf, is_individual=False):
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
        r = m.iloc[0]
        w = r[wc].to_numpy(dtype=float)
        frac = w / w.sum()
        dr = r["Dominance_Ratio"]
        dr_str = "> 10^3" if dr > 1000 else f"{dr:.1f}"
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

    handles = [Patch(facecolor=ARCH_HEX[a], label=f"Archetype {a+1}")
               for a in range(K)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if is_individual:
        ax.set_title("Regulated Allergens Composition")
    else:
        ax.text(-0.05, 1.05, '(C)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


def _ribbon(ax, xL, xR, l_top, l_bot, r_top, r_bot, color, alpha=0.5):
    cx = (xL + xR) / 2.0
    verts = [
        (xL, l_top), (cx, l_top), (cx, r_top), (xR, r_top),
        (xR, r_bot), (cx, r_bot), (cx, l_bot), (xL, l_bot), (xL, l_top)]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(
        MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_panel_D(ax, df_nmf, is_individual=False):
    target_orders = [
        "Poales", "Rosales", "Fagales", "Fabales", "Asterales",
        "Malpighiales", "Lamiales", "Zingiberales"]

    sub = df_nmf[df_nmf["Order"].isin(target_orders)].copy()

    wc = [f"Weight_A{i+1}" for i in range(K)]
    w_vals = sub[wc].to_numpy(dtype=float)
    row_sums = w_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fracs = w_vals / row_sums

    for i, c in enumerate(wc):
        sub[f"Frac_A{i+1}"] = fracs[:, i]

    order_sums = sub.groupby("Order")[
        [f"Frac_A{i+1}" for i in range(K)]].sum()

    agg_list = []
    for order in order_sums.index:
        for i in range(K):
            val = order_sums.loc[order, f"Frac_A{i+1}"]
            if val > 0.001:
                agg_list.append(
                    {"Order": order, "Archetype": i + 1, "Count": val})

    agg = pd.DataFrame(agg_list)
    if agg.empty:
        ax.axis("off")
        return

    orders = [o for o in target_orders if o in set(agg["Order"])]
    arch_list = sorted(int(a) for a in agg["Archetype"].unique())
    total = float(agg["Count"].sum())
    order_tot = agg.groupby("Order")["Count"].sum()
    arch_tot = agg.groupby("Archetype")["Count"].sum()

    gap = 0.03
    nL, nR = len(orders), len(arch_list)
    scale = min(
        (1 - gap * (nL - 1)) / total, (1 - gap * (nR - 1)) / total)

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
            a = int(r["Archetype"])
            h = float(r["Count"]) * scale
            lt = offL[o]; lb = lt - h; offL[o] = lb
            rt = offR[a]; rb = rt - h; offR[a] = rb
            _ribbon(ax, xL1, xR0, lt, lb, rt, rb,
                    ARCH_HEX[a - 1], alpha=0.5)

    for o in orders:
        t, b = posL[o]
        ax.add_patch(Rectangle(
            (xL0, b), xL1 - xL0, t - b,
            facecolor="lightgrey", edgecolor="none"))
        ax.text(xL0 - 0.02, (t + b) / 2, cap_first(o),
                ha="right", va="center", fontsize=11, fontstyle="italic")
    for a in arch_list:
        t, b = posR[a]
        ax.add_patch(Rectangle(
            (xR0, b), xR1 - xR0, t - b,
            facecolor=ARCH_HEX[a - 1], edgecolor="none"))
        ax.text(xR1 + 0.02, (t + b) / 2, f"Archetype {a}",
                ha="left", va="center", fontsize=11)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    if is_individual:
        ax.set_title("Phylogeny vs Archetypes Alluvial")
    else:
        ax.text(-0.05, 1.05, '(D)', transform=ax.transAxes,
                fontsize=18, va='bottom', ha='right')


# ===========================================================================
# 3. Plotly HTML alluvial
# ===========================================================================
def create_figure_2c_alluvial_html(df):
    target_orders = [
        "Poales", "Rosales", "Fagales", "Fabales", "Asterales",
        "Malpighiales", "Lamiales", "Zingiberales"]
    sub = df[df["Order"].isin(target_orders)].copy()

    wc = [f"Weight_A{i+1}" for i in range(K)]
    w_vals = sub[wc].to_numpy(dtype=float)
    row_sums = w_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fracs = w_vals / row_sums

    for i in range(K):
        sub[f"Frac_A{i+1}"] = fracs[:, i]

    order_sums = sub.groupby("Order")[
        [f"Frac_A{i+1}" for i in range(K)]].sum()

    agg_list = []
    for order in order_sums.index:
        for i in range(K):
            val = order_sums.loc[order, f"Frac_A{i+1}"]
            if val > 0.001:
                agg_list.append(
                    {"Order": order, "Archetype": i + 1, "Count": val})
    agg = pd.DataFrame(agg_list)

    orders = list(agg["Order"].unique())
    arches = [f"Archetype {i}" for i in sorted(agg["Archetype"].unique())]
    nodes = orders + arches
    nidx = {n: i for i, n in enumerate(nodes)}

    node_colors = [
        ARCH_HEX[int(n.split(" ")[1]) - 1] if "Archetype" in n
        else "lightgrey" for n in nodes]
    link_colors = [ARCH_RGBA[int(t) - 1] for t in agg["Archetype"]]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes, color=node_colors),
        link=dict(
            source=agg["Order"].map(nidx),
            target=agg["Archetype"].apply(
                lambda x: f"Archetype {x}").map(nidx),
            value=agg["Count"], color=link_colors)
    )])
    fig.update_layout(
        title_text=("Concordance and Divergence Between Biological "
                    "Phylogeny and Structural Allergenic Archetypes"),
        font_size=18, width=1400, height=900)
    out_html = FIGDIR / "Figure_2(c)_alluvial.html"
    fig.write_html(str(out_html))
    return out_html


# ===========================================================================
# 4. Individual panel PDFs
# ===========================================================================
def create_figure_2a_independent(top_order, top_family, top_protfam):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_panel_A(axes, top_order, top_family, top_protfam, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(a)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2b_independent(df_nmf):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_panel_B(ax, df_nmf, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(b)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2c_independent(df_nmf):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_panel_C(ax, df_nmf, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(c)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2d_independent(df_nmf):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_panel_D(ax, df_nmf, is_individual=True)
    plt.tight_layout()
    out_pdf = FIGDIR / "Figure_2(d)_raw.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_pdf


def create_figure_2_combined(df_nmf, top_order, top_family, top_protfam):
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.1]})

    draw_panel_A(
        [axes[0, 0], axes[0, 1], axes[0, 2]],
        top_order, top_family, top_protfam, is_individual=False)
    draw_panel_B(axes[1, 0], df_nmf, is_individual=False)
    draw_panel_C(axes[1, 1], df_nmf, is_individual=False)
    draw_panel_D(axes[1, 2], df_nmf, is_individual=False)

    plt.tight_layout()

    out_pdf = FIGDIR / "Figure_2_raw.pdf"
    out_png = FIGDIR / "Figure_2_raw.png"

    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.25, dpi=300)

    plt.close()
    return out_pdf


# ===========================================================================
# 5. A4 layout for the panels (pypdf + reportlab)
# ===========================================================================
def apply_a4_formatting(pdf_tasks):
    """Compose the Figure 2 panels onto A4 pages (pypdf + reportlab)."""
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

        left = float(sp.mediabox.left)
        bottom = float(sp.mediabox.bottom)
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

        os.remove(src_pdf_path)
        print(f"[A4] {final_pdf_path.name}")


# ===========================================================================
# 6. Figure 3 A4 post-processing
# ===========================================================================
def format_figure3_a4():
    """Crop margins of the Figure 3 tree PDF and compose it onto A4 (pymupdf)."""
    try:
        import pymupdf
    except ImportError:
        print("[WARN] pymupdf not installed -> skipping Figure 3 A4 step")
        return

    input_pdf = PHYLO_DIR / "Figure_3.pdf"
    if not input_pdf.exists():
        print(f"[SKIP] Figure 3 source not found: {input_pdf}")
        return

    MM = 72.0 / 25.4
    A4_W, A4_H = 210 * MM, 297 * MM
    MARGIN = 15 * MM
    LEGEND_SPACE = 45 * MM

    # STEP 1. crop margins
    doc = pymupdf.open(input_pdf)
    page = doc[0]

    rects = []
    for block in page.get_text("blocks"):
        rects.append(pymupdf.Rect(block[:4]))
    for drawing in page.get_drawings():
        rects.append(drawing["rect"])

    if not rects:
        print("[WARN] Figure 3 content region not detected")
        doc.close()
        return

    content_rect = rects[0]
    for r in rects[1:]:
        content_rect |= r

    padding = 10
    crop_rect = pymupdf.Rect(
        content_rect.x0 - padding, content_rect.y0 - padding,
        content_rect.x1 + padding, content_rect.y1 + padding)
    page.set_cropbox(crop_rect)

    cropped_path = PHYLO_DIR / "Figure_3_CROPPED.pdf"
    doc.save(cropped_path)
    doc.close()

    # STEP 2. compose onto A4
    cropped_doc = pymupdf.open(cropped_path)
    src_page = cropped_doc[0]
    crop_box = src_page.cropbox
    sw, sh = crop_box.width, crop_box.height

    avail_w = A4_W - 2 * MARGIN
    avail_h = A4_H - MARGIN - (MARGIN + LEGEND_SPACE)
    scale = min(avail_w / sw, avail_h / sh)
    fw, fh = sw * scale, sh * scale

    tx = MARGIN + (avail_w - fw) / 2.0
    ty = MARGIN

    a4_doc = pymupdf.open()
    a4_page = a4_doc.new_page(width=A4_W, height=A4_H)

    target_rect = pymupdf.Rect(tx, ty, tx + fw, ty + fh)
    a4_page.show_pdf_page(target_rect, cropped_doc, 0, clip=crop_box)

    label_text = "Figure 3_Heo and Rhee"
    font_size = 10
    text_len = pymupdf.get_text_length(
        label_text, fontname="helv", fontsize=font_size)
    label_x = A4_W - MARGIN - text_len
    label_y = A4_H - MARGIN
    a4_page.insert_text(
        (label_x, label_y), label_text,
        fontname="helv", fontsize=font_size, color=(0, 0, 0))

    a4_output = A4DIR / "Figure_3.pdf"
    a4_doc.save(a4_output)
    a4_doc.close()
    cropped_doc.close()

    print(f"[A4] Figure 3 -> {a4_output.name}")


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    print("1. Load NMF results + prepare data...")
    df_nmf, top_order, top_family, top_protfam = prepare_data()

    print("2. Individual panels...")
    fig2a_pdf = create_figure_2a_independent(top_order, top_family, top_protfam)
    fig2b_pdf = create_figure_2b_independent(df_nmf)
    fig2c_pdf = create_figure_2c_independent(df_nmf)
    fig2d_pdf = create_figure_2d_independent(df_nmf)
    fig2c_html = create_figure_2c_alluvial_html(df_nmf)
    print(f"   [HTML] {fig2c_html.name}")

    print("3. Combined 2x3 Figure 2...")
    fig2_pdf = create_figure_2_combined(df_nmf, top_order, top_family, top_protfam)

    print("4. Figure 2 A4 layout...")
    pdf_tasks = [
        (fig2a_pdf, "Figure 2(a)"),
        (fig2b_pdf, "Figure 2(b)"),
        (fig2c_pdf, "Figure 2(c)"),
        (fig2d_pdf, "Figure 2(d)"),
        (fig2_pdf,  "Figure 2"),
    ]
    apply_a4_formatting(pdf_tasks)

    print("5. Figure 3 A4 post-processing...")
    format_figure3_a4()

    print("\n[DONE] Figure 2 + Figure 3 A4 complete.")
