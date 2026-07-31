Six fundamental archetypes of food allergy revealed via non-negative matrix factorization

Code and data to reproduce the non-negative matrix factorization (NMF) pipeline, figures, and statistical analyses for the paper "Six fundamental archetypes of food allergy revealed via non-negative matrix factorization."

Starting from a 204 species × 840 conserved-domain feature matrix, the pipeline performs consensus NMF, selects the number of archetypes (K = 6), classifies each species as a Core Member or Ambiguous, generates the paper figures, and computes every statistic reported in the manuscript.

Repository structure
allergen-nmf-archetypes/
├── config.py              # Centralized paths (edit nothing if you keep the default layout)
├── requirements.txt
├── src/                   # Core pipeline — run in numerical order 01 → 05
│   ├── 01_data_preprocessing.py     # SQLite DB  -> allergen_source_matrix.csv
│   ├── 02_nmf_k_selection.py        # Consensus NMF, K selection (Figure 1)
│   ├── 03_cluster_analysis.py       # K=6 membership report (Tables 1–2 source data)
│   ├── 04_figure_generation.py      # Individual panels of Figure 2 (A–D) + Figure 3 inputs
│   └── 05_statistical_analysis.py   # All reported statistics (counts, bias, rarefaction, AMI)
├── figures/
│   └── assembly/          # Helper scripts that stitch individual panels into the final figures
│       ├── merge_figure1_panels.py
│       └── merge_figure2_panels.py
├── data/                  # Place the SQLite database here (see Data availability)
│   └── allergen_source_matrix.csv   # Processed 204 × 840 matrix (provided)
└── results/               # Auto-generated outputs (gitignored)
Installation
bash
git clone https://github.com/MinkookHeo/allergen-nmf-archetypes.git
cd allergen-nmf-archetypes
pip install -r requirements.txt

Python 3.9+ is recommended.

Data availability

The processed feature matrix (data/allergen_source_matrix.csv) is included, so scripts 02–05 run without the database for the parts that depend only on the matrix (NMF, K selection, membership classification, descriptive counts).

The large primary SQLite database (allergen_database.sqlite) is hosted separately:

Zenodo: https://doi.org/10.5281/zenodo.21714756

Download it and place it at data/allergen_database.sqlite. The database is required for:

01_data_preprocessing.py (rebuilds the matrix from scratch)
taxonomy mapping and the barplot / alluvial panels in 04_figure_generation.py
the partial-correlation and AMI blocks in 05_statistical_analysis.py

Without the database, scripts 04 and 05 print their matrix-only results and then stop at the first database query — this is expected, not a bug.

Reference allergen records were obtained from the WHO/IUIS Allergen Nomenclature Database (http://allergen.org); homolog sequences from the NCBI non-redundant protein database.

Usage

Run the core pipeline in order. All paths are resolved through config.py, so no edits are needed if you keep the default folder layout.

bash
# 1. (optional) Rebuild the matrix from the database
python src/01_data_preprocessing.py

# 2. Consensus NMF and K selection  ->  Figure 1
python src/02_nmf_k_selection.py

# 3. K=6 membership report  ->  results/NMF_Final_Analysis_K6_Step3_Advanced.xlsx
python src/03_cluster_analysis.py          # defaults to K=6
python src/03_cluster_analysis.py 6 7      # or pass specific K values

# 4. Individual figure panels  ->  figures/
python src/04_figure_generation.py

# 5. All manuscript statistics  ->  results/statistical_analysis.txt
python src/05_statistical_analysis.py
Figure assembly

04_figure_generation.py produces the individual panels (bar plots, compositional fingerprint, Sankey/alluvial, dominance landscape). The scripts in figures/assembly/ combine those panels into the composite figures used in the paper. They stitch images by pixel coordinates, so if panel sizes change you may need to adjust the paths and crop coordinates inside them. These are layout helpers, not part of the analytical pipeline.

Figure 3 (phylogenetic tree)

The circular phylogenetic tree in Figure 3 was rendered in iTOL from the species tree and the per-species archetype annotation produced by 03/04. The tree and annotation files are provided under data/phylogeny/; iTOL itself is not scripted here.

Reproducibility notes
NMF uses a fixed random seed (random_state=42) and deterministic NNDSVDa initialization, so archetype assignments are reproducible run to run.
Expected key numbers (K = 6): 135 Core Members / 69 Ambiguous, per-archetype Core counts [9, 17, 20, 37, 30, 22], minimum Core dominance ratio 4.303.
Two species (Beta vulgaris, Eriocheir sinensis) carry no assignable conserved domain and enter as all-zero rows; they are reported as Ambiguous and all analyses were repeated with them excluded.
Citation

If you use this code or data, please cite the paper (citation to be added upon publication) and the archived release (Zenodo DOI above).

License

Released under the MIT License. See LICENSE.
