# config.py
"""
Centralized paths, constants, palettes and utility functions.

All scripts import from this module. Edit nothing if you keep the
default folder layout; only PROJECT_DIR needs changing for a
relocated checkout.
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
FIGDIR = RESULTS_DIR                       # figures are written to results/
A4DIR = RESULTS_DIR / "A4_formatted"
PHYLO_DIR = DATA_DIR / "phylogeny"
CACHE_DIR = RESULTS_DIR / "consensus_cache"

# Key data files
DB_PATH = DATA_DIR / "allergen_database.sqlite"
MATRIX_PATH = DATA_DIR / "allergen_source_matrix.csv"

# NMF artefacts produced by 03, consumed by 04/05 (.npy) and 07/08 (CSV).
# NMF_MEMBERSHIP_PATH is kept distinct from the figure_source_data.csv that
# 04 writes: the two files have different column schemas, so a shared name
# would overwrite one depending on run order.
NMF_W_PATH = RESULTS_DIR / "nmf_W.npy"
NMF_H_PATH = RESULTS_DIR / "nmf_H.npy"
NMF_MEMBERSHIP_PATH = RESULTS_DIR / "nmf_membership.csv"

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
DIST_COLOR = '#0072B2'          # Distributed species

# ============================================================================
# 4. Taxonomic synonym maps
# ============================================================================
# Display names whose binomials differ from the accepted NCBI name.
# Used by 07 to match tips of the phyloT / NCBI Common Tree.
BIOLOGICAL_MAP = {
    "Acacia farnesiana": "Vachellia farnesiana",
    "Bos domesticus": "Bos taurus",
    "Crassostrea angulata [Magallana angulata]": "Magallana angulata",
    "Crassostrea gigas [Magallana gigas]": "Magallana gigas",
    "Gadus callarias": "Gadus morhua",
    "Kali turgidum [Salsola kali]": "Salsola kali",
    "Kochia scoparia [Bassia scoparia]": "Bassia scoparia",
    "Litopenaeus vannamei": "Penaeus vannamei",
    "Melicertus latisulcatus": "Penaeus latisulcatus",
    "Prosopis juliflora": "Neltuma juliflora",
    "Rana esculenta": "Pelophylax ridibundus",
    "Sebastes marinus [S. norvegicus]": "Sebastes norvegicus",
    "Triticum turgidum ssp durum": "Triticum turgidum subsp. durum",
}

# phyloT accepts NCBI TaxIDs for a few tips that resolve ambiguously by name.
PHYLOT_CORRECTIONS = {
    **BIOLOGICAL_MAP,
    "Gadus callarias": "8053",
    "Rana esculenta": "8406",
}

# Display names whose binomials differ from the SpeciesTaxonomy entries.
# Applied before the Order / Family lookup in 03, 04 and 05.
TAXONOMY_CORRECTIONS = {
    "Bos domesticus": "Bos taurus",
    "Gallus domesticus": "Gallus gallus",
    "Rana esculenta": "Pelophylax ridibundus",
    "Triticum turgidum ssp durum": "Triticum turgidum subsp. durum",
}

# Species absent from SpeciesTaxonomy, or present without an Order value.
# Orders follow the same NCBI Taxonomy revision used elsewhere in the table.
MANUAL_ORDERS = {
    "Lates calcarifer": "Carangiformes",
    "Pelophylax ridibundus": "Anura",
    "Bos taurus": "Artiodactyla",
    "Gallus gallus": "Galliformes",
    "Triticum turgidum subsp. durum": "Poales",
}

# ============================================================================
# 5. Utility functions
# ============================================================================
def cap_first(s):
    """Capitalise only the first letter; leave the rest unchanged."""
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def norm_name(s):
    """'Genus species (common name)' -> 'Genus species'."""
    return str(s).split(" (")[0].strip()


def set_journal_font():
    """Journal font setup (Arial preferred)."""
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


# ============================================================================
# 6. Logging helper
# ============================================================================
class Logger:
    """Simple console + file logger."""
    def __init__(self):
        self._lines = []

    def __call__(self, msg=""):
        print(msg)
        self._lines.append(str(msg))

    def save(self, path):
        Path(path).write_text("\n".join(self._lines), encoding="utf-8")
