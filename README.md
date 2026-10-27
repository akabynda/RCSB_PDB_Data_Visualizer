# RCSB PDB Data Visualizer

Downloads structural data from RCSB PDB, builds CSV datasets, and generates the
figures for the accompanying article. Plots include deposition years through
2024; rebuilding queries the RCSB records available at the time of the run.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Reproduce All Figures](#reproduce-all-figures)
- [Reproduce Individual Figures](#reproduce-individual-figures)
  - [Figure 1](#figure-1)
  - [Figure 2](#figure-2)
  - [Figure 3](#figure-3)
  - [Figure 4](#figure-4)
  - [Figure 5](#figure-5)
  - [Figure 6](#figure-6)
  - [Figure 7](#figure-7)
  - [Figure 8](#figure-8)
  - [Figure 99](#figure-99)
- [Outputs](#outputs)
- [Reference](#reference)
- [License](#license)

## Requirements

- Python 3.10 or newer and network access.
- Sufficient disk space for downloaded coordinates and calculations; a full
  build can require about 100 GiB.
- Figures 3–7: STRIDE, or Git, GNU Make and a C compiler for its automatic
  installation on macOS/Linux. On Windows, use WSL or supply a prebuilt STRIDE.
- Figure 99: TmDet, or CMake and a C++20 compiler for its automatic installation.

## Setup

```bash
git clone https://github.com/akabynda/RCSB_PDB_Data_Visualizer.git
cd RCSB_PDB_Data_Visualizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the following commands from the repository root with this environment
active. The builder installs STRIDE and TmDet when needed. To use existing
executables, pass `--solution-nmr-monomer-stride-executable /path/to/stride` or
`--tmdet-executable /path/to/tmdet` to the builder.

## Reproduce All Figures

```bash
python src/pdb_dataset_builder.py --datasets all
python src/pdb_plot.py --plots all
```

This runs all supported analyses, including those beyond the article figures.
The full build downloads coordinates, searches for homologs and calculates
structural comparisons, so it can take a long time.

The builder writes CSV datasets to `data/`; the plotter reads them and writes
PNGs under `figures/`. See [Outputs](#outputs) to locate each figure. Input/output
path options are listed by `--help`; add `--svg` to export SVGs as well.
For individual figures, run the commands below. Figures 6 and 7 use homologs
selected at 100% sequence identity.

## Reproduce Individual Figures

### Figure 1

```bash
python src/pdb_dataset_builder.py --datasets method_counts
python src/pdb_plot.py --plots method_counts
```

### Figure 2

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_weights
python src/pdb_plot.py \
  --plots solution_nmr_weight_stats,solution_nmr_period_area_share
```

### Figure 3

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model
python src/pdb_plot.py \
  --plots solution_nmr_monomer_stride_modeled_first_model
```

### Figure 4

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_precision_stride_modeled_first_model
python src/pdb_plot.py \
  --plots solution_nmr_monomer_precision_stride_modeled_first_model_median
```

### Figure 5

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical
python src/pdb_plot.py --plots solution_nmr_monomer_xray_homolog_timing_share
```

### Figure 6

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_rmsd,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python src/pdb_plot.py --plots solution_nmr_monomer_xray_rmsd
```

### Figure 7

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_precision_stride_modeled_first_model,solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python src/pdb_plot.py \
  --plots solution_nmr_monomer_xray_rmsd_precision_correlation
```

### Figure 8

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters

python src/pdb_plot.py \
  --plots solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters
```

### Figure 99

Temporary figure number: the residue-count map of modeled membrane versus
other modeled protein residues in solution-NMR entries, with identical chains
averaged per copy. Generate it with the membrane plot group:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_membrane_proteins
python src/pdb_plot.py --plots solution_nmr_membrane_proteins
```

## Outputs

Dataset CSVs are saved in `data/`; their names and contents are listed in the
[dataset reference](src/README.md#dataset-reference).

Each figure has a directory `figures/<stem>/` containing `<stem>.png` and
variants with suffixes `_no_title`, `_open_axes` and `_no_title_open_axes`.
Figures 1–6 and 8 use the titled `_open_axes.png` version; Figure 7 uses the
base PNG with a closed frame.

| Figure | Main output stem(s) |
| --- | --- |
| 1 | `pdb_method_trends` |
| 2A | `solution_nmr_mean_weight_by_year` |
| 2B | `solution_nmr_area_share_by_weight_category` |
| 3 | `solution_nmr_monomer_stride_modeled_first_model_by_year` |
| 4 | `solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year` |
| 5 | `solution_nmr_monomer_xray_homologs_95_timing_share_by_year` |
| 6 | `solution_nmr_monomer_xray_min_median_rmsd_by_year` |
| 7 | `solution_nmr_monomer_xray_min_rmsd_precision_correlation` |
| 8A | `solution_nmr_monomer_quality_clashscore_by_year` |
| 8B | `solution_nmr_monomer_quality_ramachandran_outliers_by_year` |
| 8C | `solution_nmr_monomer_quality_sidechain_outliers_by_year` |
| 8D | `solution_nmr_monomer_program_cluster_share_by_year` |
| 99 | `solution_nmr_membrane_tmdet_residues` |

Figure 99 is only the residue-count map in
`figures/solution_nmr_membrane_tmdet_residues/`. The membrane plot group also
produces separate count, status and time-summary figures.

## Reference

- [Dataset definitions, filtering, RMSD, output variants and resume options](src/README.md).
- [Test and coverage commands](test/README.md).
- Full command-line options:

```bash
python src/pdb_dataset_builder.py --help
python src/pdb_plot.py --help
```

## License

See [`LICENSE`](LICENSE) for the repository terms. STRIDE, TmDet and Python
dependencies are subject to their own licenses.
