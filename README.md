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
  current coordinate cache occupies approximately 65 GiB; 100 GiB is recommended
  for temporary files and future updates.
- Git, GNU Make, and a C compiler (`gcc`, `cc`, or `clang`) for the first
  automatic STRIDE build used by Figures 3–7. These tools are not needed when a
  working STRIDE executable is already installed or supplied explicitly.

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

Figures 3–7 also require STRIDE. On macOS and Linux no separate STRIDE setup is
needed: the first STRIDE-dependent dataset command checks `PATH` and existing
local builds, then downloads the pinned [upstream source][stride-repository]
and builds it under
`data/stride/867a5eb0f2479cb16615512a53ee472c54649505/<system>-<architecture>/`.
Later commands on the same platform reuse that executable. The first build
therefore needs access to GitHub plus Git, GNU Make, and a C compiler.

If STRIDE is installed elsewhere, pass its executable explicitly. An invalid
explicit path is reported as an error and does not trigger an automatic
download:

```text
--solution-nmr-monomer-stride-executable /path/to/stride
```

Use `--stride-install-dir /another/directory` to change the root of the managed
installation. Native Windows users should run the builder in WSL or provide a
prebuilt executable explicitly.

## Reproduce All Article Figures

Build every dataset in dependency order:

```bash
python src/pdb_dataset_builder.py --datasets all
```

This is a long-running operation. It downloads coordinate files, queries RCSB
PDB, installs and runs STRIDE when necessary, searches for X-ray homologs, and
calculates RMSD values.

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

```bash
python src/pdb_dataset_builder.py --datasets method_counts
python src/pdb_plot.py --plots method_counts
```

Output:

- `figures/pdb_method_trends/pdb_method_trends.png`

### Figure 2

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

- Downloaded coordinates, the managed STRIDE source/binary, and STRIDE results
  are cached in `data/`. Rerunning a command reuses available files.
- Coordinate downloads and conversions are single-flight per PDB ID. Threads
  and concurrent POSIX builder processes cannot publish mixed PDB/mmCIF,
  chain-map, or metadata cache bundles; different PDB IDs still run in parallel.
- Every selected dataset is rebuilt by default. Pass `--resume` to continue an
  interrupted homolog, precision, or X-ray RMSD calculation.
- Every primary dataset CSV generated by the builder has a paired
  `name_filtered.csv`
  containing the `entry_id`, deposition `year`, and reason for each structure
  rejected while that dataset was built. A header-only file means that no
  structures were filtered out.
- The X-ray homolog build additionally writes `*_rejected.csv` beside each
  95%/100% homolog CSV. These reports contain one row for every RCSB sequence
  hit rejected by the modeled-core coordinate filter, including its source NMR
  entry, X-ray polymer entity, chains, cutoff, and reason.
- Every primary dataset CSV generated by the builder has a sibling `.log` file.
  An empty log means no warnings or errors were recorded for that dataset. CSVs
  written by the plotter, such as homolog timing counts, do not receive these
  companion files.
- Use `python src/pdb_dataset_builder.py --help` and
  `python src/pdb_plot.py --help` for optional input/output paths, worker
  counts, resume behavior, and SVG generation.

## Technical Reference

Implementation details, filtering rules, dataset definitions, cache behavior,
and RMSD formulas are documented in [`src/README.md`](src/README.md).

## License

No open-source license is granted for this repository. See [`LICENSE`](LICENSE)
for the applicable terms. STRIDE and all Python dependencies remain subject to
their own licenses. In particular, the [STRIDE license][stride-license] permits
academic use subject to its notice, citation, and bug-reporting conditions;
commercial use requires a separate written license. Automatic installation does
not change those terms.

[stride-repository]: https://github.com/MDAnalysis/stride
[stride-license]: https://github.com/MDAnalysis/stride/blob/867a5eb0f2479cb16615512a53ee472c54649505/LICENSE
[repository]: https://github.com/akabynda/RCSB_PDB_Data_Visualizer
