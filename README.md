# Six fundamental archetypes of food allergy revealed via non-negative matrix factorization

This repository contains the code and data necessary to reproduce the Non-Negative Matrix Factorization (NMF) pipeline and statistical analyses for the paper: *"Six fundamental archetypes of food allergy revealed via non-negative matrix factorization"*.

## Repository Structure
* `config.py`: Centralized configuration for all file paths.
* `src/`: Python scripts for the core analytical pipeline (must be run sequentially from `01` to `05`).
* `data/`: Contains the processed matrix (`allergen_source_matrix.csv`) and phylogenetic tree files.
* `figures/`: Output directory for generated figures.
* `results/`: Output directory for statistical reports and tables.

## Data Availability
The large primary SQLite database (`allergen_database.sqlite`) is hosted separately due to file size limits. 
* **Database Download:** Available on Zenodo at [https://doi.org/10.5281/zenodo.21714756](https://doi.org/10.5281/zenodo.21714756)
* Please download the database and place it in the `data/` folder before running the pipeline.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the scripts in the `src/` directory in numerical order (01 to 05).
