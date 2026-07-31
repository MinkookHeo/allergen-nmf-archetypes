# config.py
import os
from pathlib import Path

# 프로젝트 최상위 경로 (config.py가 위치한 곳)
PROJECT_DIR = Path(__file__).resolve().parent

# 하위 디렉토리 설정
DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

# 주요 데이터 파일 경로
# 주의: allergen_database.sqlite는 용량이 크므로 깃허브에 직접 올리지 않고,
# 실행하는 사람이 data 폴더 안에 넣어두도록 README에 안내합니다.
DB_PATH = DATA_DIR / "allergen_database.sqlite"
MATRIX_PATH = DATA_DIR / "allergen_source_matrix.csv"

# 결과를 저장할 폴더가 없으면 자동으로 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
