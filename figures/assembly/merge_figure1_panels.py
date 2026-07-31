# -*- coding: utf-8 -*-
# =============================================================================
# 그림 배치(assembly) 보조 스크립트 - 분석 파이프라인이 아님
# -----------------------------------------------------------------------------
# 04_figure_generation.py 가 만든 개별 그림(PNG)을 논문 게재용 하나의
# 패널 그림으로 붙이는 용도입니다. 픽셀 좌표 기반으로 이미지를 잘라 배치하므로,
# 입력 그림의 크기/여백이 바뀌면 아래 경로와 좌표를 직접 조정해야 합니다.
#
# 사용법: 아래 입력/출력 경로를 본인 환경(figures/ 폴더)에 맞게 수정한 뒤 실행.
#         재현 파이프라인(01-05)과 무관하며, 최종 그림 편집용입니다.
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# 1. 파일 경로 설정
fig2_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure 2.png"
fig3_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure 3.png"
out_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure2_Merged_Final.png"

# 2. 이미지 불러오기
img2 = mpimg.imread(fig2_path)
img3 = mpimg.imread(fig3_path)

# 3. Figure 2 (선 그래프) 3등분
h2, w2, _ = img2.shape
w2_third = w2 // 3
panel_a = img2[:, :w2_third]
panel_b = img2[:, w2_third : w2_third*2]
panel_c = img2[:, w2_third*2 :]

# 4. Figure 3 (매트릭스) 6등분 및 기존 작은 글자 잘라내기
h3, w3, _ = img3.shape
w3_third = w3 // 3
h3_half = h3 // 2

top_crop = int(h3_half * 0.09) 

panel_k3 = img3[top_crop : h3_half, :w3_third]
panel_k4 = img3[top_crop : h3_half, w3_third:w3_third*2]
panel_k5 = img3[top_crop : h3_half, w3_third*2:]
panel_k6 = img3[h3_half + top_crop : h3_half*2, :w3_third]
panel_k7 = img3[h3_half + top_crop : h3_half*2, w3_third:w3_third*2]
panel_k8 = img3[h3_half + top_crop : h3_half*2, w3_third*2:]

# 5. 3행 3열 그리드 레이아웃 설정
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3, 3, height_ratios=[1, 1.1, 1.1], figure=fig)

# 각 패널 축 생성
axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),  # 1행 (a, b, c)
    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),  # 2행 (k3, k4, k5)
    fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])   # 3행 (k6, k7, k8)
]

panels = [panel_a, panel_b, panel_c, panel_k3, panel_k4, panel_k5, panel_k6, panel_k7, panel_k8]

# 💡 패널 기호(a,b,c,d)와 K-제목을 완전히 분리했습니다.
panel_labels = ['(a)', '(b)', '(c)', '(d)', '', '', '', '', '']
k_titles = ['', '', '', 'K = 3', 'K = 4', 'K = 5', 'K = 6', 'K = 7', 'K = 8']

for i, ax in enumerate(axes):
    ax.imshow(panels[i])
    ax.axis('off')
    
    # 1. (a), (b), (c), (d)는 좌측 상단에 굵고 큰 폰트(24)로 통일
    if panel_labels[i]:
        ax.set_title(panel_labels[i], loc='left', fontsize=24, fontweight='bold', pad=15)
        
    # 2. K = 3 ~ 8 은 패널 중앙에 일반 굵기, 적당한 폰트(20)로 100% 통일
    if k_titles[i]:
        ax.set_title(k_titles[i], loc='center', fontsize=20, fontweight='normal', pad=15)

# 패널 간 겹침 방지 여백 설정
plt.tight_layout(pad=2.0)

# 6. 최종 저장
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 글자 크기 불균형 수정 및 정렬 완료!\n저장 위치: {out_path}")