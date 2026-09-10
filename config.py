# config.py
"""
Centralized paths, constants, palettes and utility functions.

All scripts import from this module. Edit nothing if you keep the
default folder layout; only BASE_DIR / PROJECT_DIR need changing
for a relocated checkout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================================
# 1. Directory layout
# ============================================================================
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
A4DIR = RESULTS_DIR / "A4_formatted"
PHYLO_DIR = DATA_DIR / "phylogeny"
CACHE_DIR = RESULTS_DIR / "consensus_cache"

# Key data files
DB_PATH = DATA_DIR / "allergen_database.sqlite"
MATRIX_PATH = DATA_DIR / "allergen_source_matrix.csv"

# NMF artefacts produced by 03, consumed by 04 and 05
NMF_W_PATH = RESULTS_DIR / "nmf_W.npy"
NMF_H_PATH = RESULTS_DIR / "nmf_H.npy"
NMF_MEMBERSHIP_PATH = RESULTS_DIR / "figure_source_data.csv"

# Create output directories
for _d in [DATA_DIR, RESULTS_DIR, A4DIR, PHYLO_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. NMF / analysis constants
# ============================================================================
K = 6
RANDOM_STATE = 42
CORE_RA_THRESHOLD = 0.80
MAX_ITER = 5000
N_REFERENCE_ALLERGENS = 1149   # total WHO/IUIS records queried

# ============================================================================
# 3. Colour palette (Okabe-Ito CVD-safe, 6 colours)
#    Must stay identical to the iTOL annotation in Figure 3.
# ============================================================================
ARCH_HEX = ['#0072B2', '#D55E00', '#56B4E9', '#E69F00', '#009E73', '#F0E442']
ARCH_RGBA = [
    'rgba(0,114,178,0.5)', 'rgba(213,94,0,0.5)', 'rgba(86,180,233,0.5)',
    'rgba(230,159,0,0.5)', 'rgba(0,158,115,0.5)', 'rgba(240,228,66,0.5)',
]
CORE_COLOR = '#D55E00'
AMB_COLOR = '#0072B2'

# ============================================================================
# 4. Utility functions
# ============================================================================
def cap_first(s):
    """Capitalise only the first letter; leave the rest unchanged."""
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def norm_name(s):
    """'Genus species (common name)' -> 'Genus species'."""
    return str(s).split(" (")[0].strip()


def set_journal_font():
    """Set matplotlib font to Arial / Helvetica (Allergy journal guideline)."""
    preferred = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams["font.sans-serif"] = preferred
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["font.size"] = 12
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
