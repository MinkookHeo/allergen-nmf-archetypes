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
import matplotlib.transforms as transforms

# 1. 파일 경로 설정
fig_a_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure3_dominance_landscape.png"
fig_b_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure4_compositional_fingerprint.png"
fig_c_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\newplot.png"
out_path = r"D:\Allergyweb\Allergen_Hetero_Crosslinking\Results_Figures\Figure3_Merged_abc_3x1_Aligned.png"

# 2. 이미지 불러오기
img_a = mpimg.imread(fig_a_path)
img_b = mpimg.imread(fig_b_path)
img_c = mpimg.imread(fig_c_path)

# 3. 레이아웃 설정: 3행 1열 그리드
fig, axes = plt.subplots(3, 1, figsize=(14, 24))

images = [img_a, img_b, img_c]
labels = ['(a)', '(b)', '(c)']

# 4. 그림 삽입 및 라벨 절대 위치 고정
for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis('off')
    
    # 💡 핵심: X축은 '전체 도화지' 기준, Y축은 '각 그림' 기준으로 좌표계를 혼합합니다.
    trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    
    # x=0.08 (도화지 왼쪽 끝에서 8% 지점에 완벽히 일렬로 고정)
    # y=1.02 (각 그림의 바로 위쪽)
    ax.text(0.08, 1.02, labels[i], transform=trans, 
            fontsize=32, fontweight='bold', va='bottom', ha='left')

# 5. 패널 간격 확보 및 저장
plt.tight_layout(pad=4.0)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 라벨 수직 정렬 완료!\n저장 위치: {out_path}")