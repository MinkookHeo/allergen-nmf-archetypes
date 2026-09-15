# Six fundamental archetypes of food allergy revealed via non-negative matrix factorization

Code and data to reproduce the non-negative matrix factorization (NMF) pipeline
and statistical analyses reported in the paper.

## Repository structure

- `config.py` — paths, constants, colour palette, taxonomic synonym maps.
- `src/` — analysis scripts.
- `data/` — processed matrix (`allergen_source_matrix.csv`) and tree files.
- `results/` — figures, statistical reports and tables (created at runtime).

## Scripts

| Script | Purpose |
| --- | --- |
| `01_data_preprocessing.py` | Species-by-domain bitscore matrix from the SQLite database. |
| `02_nmf_k_selection.py` | Consensus clustering across K; cophenetic, RSS and delta-area metrics (Figure 1). |
| `03_cluster_analysis.py` | NMF at K = 6; Core and Ambiguous assignment; membership table. |
| `04_figure_generation.py` | Figure 2 panels and A4 layout; A4 composition of the Figure 3 tree. |
| `05_statistical_analysis.py` | Descriptive counts, annotation-bias tests, rarefaction, AMI with a permutation null. |
| `06_ambiguous_species_extract.py` | Ambiguous species ranked by reference-allergen coverage. |
| `07_phylogeny_itol.py` | Core species list and iTOL annotation files for Figure 3. |
| `08_figure2d_audit.py` | Verification of the Figure 2D aggregation. |

Run `01` through `05` in order, then `07`. `06` and `08` read existing outputs
and can be run at any point afterwards.

`07` writes `iTOL_multibar_dataset.txt` and `iTOL_label_dataset.txt` into
`data/phylogeny/`. These are uploaded to iTOL with the Newick tree to render
Figure 3; the exported PDF is returned to `data/phylogeny/Figure_3.pdf` for the
A4 step in `04`.

## Archetype assignment

NMF is used as a soft clustering: every species carries a continuous weight for
all six archetypes. Figures that aggregate across taxa (Figure 2D, Figure 3)
therefore use these weights rather than a single primary label, so that species
with mixed repertoires contribute to each archetype they load onto.
`08_figure2d_audit.py` checks the resulting identity: since the weight matrix is
row-normalized, the ribbons emanating from a taxonomic order sum to the number
of species in that order.

## Data availability

The primary SQLite database (`allergen_database.sqlite`) is hosted separately
due to file size limits.

- Database: https://doi.org/10.5281/zenodo.21714756
- Place the downloaded file in `data/` before running the pipeline.

Reference allergen records were obtained from the WHO/IUIS Allergen
Nomenclature Database (http://allergen.org) and homolog sequences from the NCBI
non-redundant protein database.

## Usage

```bash
pip install -r requirements.txt

python src/01_data_preprocessing.py
python src/02_nmf_k_selection.py
python src/03_cluster_analysis.py
python src/04_figure_generation.py
python src/05_statistical_analysis.py
python src/07_phylogeny_itol.py
python src/08_figure2d_audit.py
```

## License

MIT
