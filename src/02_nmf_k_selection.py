import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 0. 설정 (config.py 연동 및 Allergy 가이드라인 폰트)
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MATRIX_PATH, RESULTS_DIR, FIG_DIR

CACHE_DIR = RESULTS_DIR / "consensus_cache"
IN_MATRIX = MATRIX_PATH
OUT_METRICS = RESULTS_DIR / "NMF_K_selection_metrics.csv"
OUT_FIG_COMBINED = FIG_DIR / "Figure_2.png"  # 통합 그림은 figures 폴더로

K_RANGE = range(2, 16)
N_RUNS = 30
NMF_KWARGS = dict(init="random", solver="cd", beta_loss="frobenius", tol=1e-4, max_iter=2000)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 1. 데이터 로드 및 전처리
# ===========================================================================
def load_processed_data(file_path):
    if not Path(file_path).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path, index_col=0)
    df_log = np.log1p(df)
    matrix_norm = Normalizer(norm="l1").fit_transform(df_log)
    return matrix_norm, df.columns, df.index


# ===========================================================================
# 2. Consensus matrix (캐시 사용)
# ===========================================================================
def build_consensus(matrix, k, n_runs=N_RUNS, use_cache=True):
    cache_c = CACHE_DIR / f"consensus_K{k}.npy"
    cache_r = CACHE_DIR / f"rss_K{k}.npy"

    if use_cache and cache_c.exists() and cache_r.exists():
        return np.load(cache_c), float(np.load(cache_r))

    n = matrix.shape[0]
    consensus = np.zeros((n, n))
    errors = []

    for i in range(n_runs):
        model = NMF(n_components=k, random_state=i, **NMF_KWARGS)
        W = model.fit_transform(matrix)
        errors.append(model.reconstruction_err_ ** 2)

        labels = np.argmax(W, axis=1)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            consensus[np.ix_(idx, idx)] += 1

    consensus /= n_runs
    mean_rss = float(np.mean(errors))

    np.save(cache_c, consensus)
    np.save(cache_r, np.array(mean_rss))
    return consensus, mean_rss


def cdf_area(consensus):
    n = consensus.shape[0]
    values = consensus[np.tril_indices(n, k=-1)]
    hist, _ = np.histogram(values, bins=100, range=(0, 1))
    cdf = np.cumsum(hist) / len(values)
    return float(np.sum(cdf * (1.0 / 100)))


def cophenetic_corr(consensus):
    d = squareform(1 - consensus, checks=False)
    Z = linkage(d, method="average")
    coph, _ = cophenet(Z, d)
    return float(coph), Z


# ===========================================================================
# 3. 실행
# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()

    matrix_norm, feature_names, species_names = load_processed_data(IN_MATRIX)
    print(f"[OK] 입력 행렬: {matrix_norm.shape[0]} 종 x {matrix_norm.shape[1]} feature")
    print(f"[RUN] K={list(K_RANGE)}, runs={N_RUNS}\n")

    rows = []
    for k in K_RANGE:
        consensus, rss = build_consensus(matrix_norm, k)
        coph, _ = cophenetic_corr(consensus)
        auc = cdf_area(consensus)
        rows.append({"K": k, "cophenetic": coph, "RSS": rss, "CDF_area": auc})
        print(f"  K={k:2d} | Coph={coph:.4f} | RSS={rss:.4f} | AUC={auc:.4f}")

    met = pd.DataFrame(rows)

    # Delta area 계산
    auc = met["CDF_area"].to_numpy()
    delta = np.empty_like(auc)
    delta[0] = auc[0]
    delta[1:] = (auc[1:] - auc[:-1]) / auc[:-1]
    met["delta_area"] = delta

    met.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 지표 저장 -> {OUT_METRICS}")

    # =======================================================================
    # 통합 Figure 2 생성 (3x3 레이아웃)
    # =======================================================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    k_list = met["K"]

    # --- 1행: K 선정 지표 (A, B, C) ---
    color_A, color_B, color_C = '#1B9E77', '#D95F02', '#7570B3'

    # (A) Cophenetic correlation
    axes[0, 0].plot(k_list, met["cophenetic"], "o-", color=color_A, lw=2, ms=6)
    axes[0, 0].set_xlabel("K"); axes[0, 0].set_ylabel("Cophenetic correlation coefficient")
    axes[0, 0].text(-0.15, 1.05, "(A)", transform=axes[0, 0].transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(list(k_list))

    # (B) RSS
    axes[0, 1].plot(k_list, met["RSS"], "o-", color=color_B, lw=2, ms=6)
    axes[0, 1].set_xlabel("K"); axes[0, 1].set_ylabel("Residual sum of squares (RSS)")
    axes[0, 1].text(-0.15, 1.05, "(B)", transform=axes[0, 1].transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(list(k_list))

    # (C) Delta area
    axes[0, 2].plot(k_list, met["delta_area"], "o-", color=color_C, lw=2, ms=6)
    axes[0, 2].set_xlabel("K"); axes[0, 2].set_ylabel("Delta area under consensus CDF")
    axes[0, 2].text(-0.15, 1.05, "(C)", transform=axes[0, 2].transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].set_xticks(list(k_list))

    sns.despine(ax=axes[0, 0])
    sns.despine(ax=axes[0, 1])
    sns.despine(ax=axes[0, 2])

    # --- 2행 및 3행: Consensus 히트맵 (D) ---
    # 비대화형(재현) 실행에서는 입력을 생략하고 논문 기본값 K=3~8을 쓴다.
    try:
        k_input = input(
            "\n히트맵으로 확인할 K값 입력 (공백 구분, 기본값 K=3~8은 그냥 엔터): "
        ).strip()
    except (EOFError, OSError):
        k_input = ""
    target_ks = [int(v) for v in k_input.split()] if k_input else [3, 4, 5, 6, 7, 8]

    # 히트맵을 그릴 6개의 축(axes) 평탄화 추출 (2행과 3행)
    heatmap_axes = axes[1:, :].flatten()

    for i, k in enumerate(target_ks):
        if i >= len(heatmap_axes):
            break # 3x3 배열을 초과하는 경우 방지

        consensus, _ = build_consensus(matrix_norm, k)   
        _, Z = cophenetic_corr(consensus)
        order = leaves_list(Z)
        
        sns.heatmap(
            consensus[np.ix_(order, order)],
            ax=heatmap_axes[i], cmap="YlGnBu", vmin=0, vmax=1,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        
        # 첫 번째 히트맵(K=3) 좌상단에만 대표 패널 라벨 (D) 삽입
        if i == 0:
            heatmap_axes[i].text(-0.05, 1.05, "(D)", transform=heatmap_axes[i].transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")
        
        heatmap_axes[i].set_title(f"K = {k}", pad=10)

    # 비어 있는 축이 있다면(target_ks가 6개 미만일 때) 숨김 처리
    for j in range(len(target_ks), len(heatmap_axes)):
        heatmap_axes[j].axis("off")

    plt.tight_layout()
    # 하나의 Figure로 600 dpi 저장
    plt.savefig(OUT_FIG_COMBINED, dpi=600, bbox_inches="tight")
    print(f"\n[OK] 3x3 통합 Figure 저장 완료 -> {OUT_FIG_COMBINED}")
    plt.close()

    print(f"\n[DONE] 총 소요: {time.time() - t0:.1f}초")
