# RCSB PDB Data Visualizer

RCSB PDB Data Visualizer downloads the structural data needed for the
accompanying article, builds its CSV datasets, and renders the article figures.
This README contains the steps required to reproduce those results.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Reproduce All Article Figures](#reproduce-all-article-figures)
- [Reproduce Individual Article Figures](#reproduce-individual-article-figures)
  - [Figure 1](#figure-1)
  - [Figure 2](#figure-2)
  - [Figure 3](#figure-3)
  - [Figure 4](#figure-4)
  - [Figure 5](#figure-5)
  - [Figure 6](#figure-6)
  - [Figure 7](#figure-7)
  - [Figure 8](#figure-8)
- [Long-Running Builds](#long-running-builds)
- [Technical Reference](#technical-reference)
- [License](#license)

## Requirements

- Python 3.10 or newer.
- Network access and at least 80 GiB of free disk space for a full build. The
  current coordinate cache occupies approximately 64 GiB; 100 GiB is recommended
  for temporary files and future updates.
- [STRIDE][stride-repository], GNU Make, and a C compiler for Figures 3–7.

Run every command below from the repository root.

## Setup

Clone the [source repository][repository] if it is not already available:

```bash
git clone https://github.com/akabynda/RCSB_PDB_Data_Visualizer.git
cd RCSB_PDB_Data_Visualizer
```

Create and activate a virtual environment, then install the pinned Python
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Figures 3–7 also require STRIDE. On macOS or Linux, it can be built at the
location detected automatically by the dataset builder:

```bash
git clone https://github.com/MDAnalysis/stride.git /tmp/stride_src
make -C /tmp/stride_src/src
```

If STRIDE is installed elsewhere, add the following option to dataset commands
that require it:

```text
--solution-nmr-monomer-stride-executable /path/to/stride
```

## Reproduce All Article Figures

Build every dataset in dependency order:

```bash
python src/pdb_dataset_builder.py --datasets all
```

This is a long-running operation. It downloads coordinate files, queries RCSB
PDB, runs STRIDE, searches for X-ray homologs, and calculates RMSD values.

After the datasets have been created in `data/`, render all figures:

```bash
python src/pdb_plot.py --plots all
```

Each figure is written to its own directory under `figures/`. The main article
image has the same name as its directory. Three additional PNG variants without
a title and/or with open top and right axes are generated alongside it.

## Reproduce Individual Article Figures

Run only the commands for the required figure to avoid building unrelated
datasets. Figures 3–7 require STRIDE. Figures 6 and 7 use X-ray homologs at 100%
sequence identity.

### Figure 1

Includes only entries that contain at least one protein polymer entity.

```bash
python src/pdb_dataset_builder.py --datasets method_counts
python src/pdb_plot.py --plots method_counts
```

Output:

- `figures/pdb_method_trends/pdb_method_trends.png`

### Figure 2

Includes exact single-method `SOLUTION NMR` entries that contain at least one
protein polymer entity.

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_weights
python src/pdb_plot.py \
  --plots solution_nmr_weight_stats,solution_nmr_period_area_share
```

Outputs:

- Figure 2A:
  `figures/solution_nmr_mean_weight_by_year/solution_nmr_mean_weight_by_year.png`
- Figure 2B:
  `figures/solution_nmr_area_share_by_weight_category/solution_nmr_area_share_by_weight_category.png`

### Figure 3

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model
python src/pdb_plot.py \
  --plots solution_nmr_monomer_stride_modeled_first_model
```

Output:

- `figures/solution_nmr_monomer_stride_modeled_first_model_by_year/solution_nmr_monomer_stride_modeled_first_model_by_year.png`

### Figure 4

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_precision_stride_modeled_first_model
python src/pdb_plot.py \
  --plots solution_nmr_monomer_precision_stride_modeled_first_model_median
```

Output:

- `figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.png`

### Figure 5

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs_historical
python src/pdb_plot.py --plots solution_nmr_monomer_xray_homolog_timing_share
```

Output:

- `figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.png`

### Figure 6

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd \
  --xray-rmsd-sequence-identity 100
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python src/pdb_plot.py --plots solution_nmr_monomer_xray_rmsd
```

Output:

- `figures/solution_nmr_monomer_xray_min_median_rmsd_by_year/solution_nmr_monomer_xray_min_median_rmsd_by_year.png`

### Figure 7

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_precision_stride_modeled_first_model

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python src/pdb_plot.py \
  --plots solution_nmr_monomer_xray_rmsd_precision_correlation
```

Output:

- `figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation/solution_nmr_monomer_xray_min_rmsd_precision_correlation.png`

### Figure 8

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_quality

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_program_clusters

python src/pdb_plot.py \
  --plots solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters
```

Outputs:

- Figure 8A:
  `figures/solution_nmr_monomer_quality_clashscore_by_year/solution_nmr_monomer_quality_clashscore_by_year.png`
- Figure 8B:
  `figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year/solution_nmr_monomer_quality_ramachandran_outliers_by_year.png`
- Figure 8C:
  `figures/solution_nmr_monomer_quality_sidechain_outliers_by_year/solution_nmr_monomer_quality_sidechain_outliers_by_year.png`
- Figure 8D:
  `figures/solution_nmr_monomer_program_cluster_share_by_year/solution_nmr_monomer_program_cluster_share_by_year.png`

## Long-Running Builds

- Downloaded coordinates and STRIDE results are cached in `data/`. Rerunning a
  command reuses available cache files.
- To continue an interrupted X-ray homolog search, rerun its dataset command
  with `--xray-homolog-resume`.
- Every generated CSV has a sibling `.log` file. An empty log means no warnings
  or errors were recorded for that dataset.
- Use `python src/pdb_dataset_builder.py --help` and
  `python src/pdb_plot.py --help` for optional path, worker, overwrite, and SVG
  settings.

## Technical Reference

Implementation details, filtering rules, dataset definitions, cache behavior,
and RMSD formulas are documented in [`src/README.md`](src/README.md).

## License

No open-source license is granted for this repository. See [`LICENSE`](LICENSE)
for the applicable terms. STRIDE and all Python dependencies remain subject to
their own licenses.

[stride-repository]: https://github.com/MDAnalysis/stride
[repository]: https://github.com/akabynda/RCSB_PDB_Data_Visualizer
