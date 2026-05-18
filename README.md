# RCSB PDB Data Visualizer

## Setup for First-Time Users

Create a Python virtual environment and install the required Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Figures 3-7 also require STRIDE. The dataset builder automatically checks `/tmp/stride_src/src/stride`, so the simplest setup is to clone and build STRIDE there:

```bash
git clone https://github.com/MDAnalysis/stride.git /tmp/stride_src
make -C /tmp/stride_src/src
```

After that, the commands below can be run exactly as written. If STRIDE is installed somewhere else, pass its path with `--solution-nmr-monomer-stride-executable /path/to/stride`.

## Reproducing the Article Figures

Run the commands below from the repository root. For each figure, the dataset-builder command creates the required CSV inputs, and the plotting command renders the article figure. Some plot groups also write companion plots that are not used in the article. Figures 3-7 require a STRIDE executable. Figures 6 and 7 use the 100% sequence-identity X-ray RMSD datasets.

### Figure 1

Build the dataset:

```bash
python src/pdb_dataset_builder.py --datasets method_counts
```

Render the figure:

```bash
python src/pdb_plot.py --plots method_counts
```

Article output:

- `figures/pdb_method_trends.png`

### Figure 2

Build the shared dataset for both panels:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_weights
```

Render both article panels:

```bash
python src/pdb_plot.py --plots solution_nmr_weight_stats,solution_nmr_period_area_share
```

Article outputs:

- Figure 2A: `figures/solution_nmr_mean_weight_by_year.png`
- Figure 2B: `figures/solution_nmr_area_share_by_weight_category.png`

### Figure 3

Build the dataset:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_stride_modeled_first_model
```

Render the figure:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_stride_modeled_first_model
```

Article output:

- `figures/solution_nmr_monomer_stride_modeled_first_model_by_year.png`

### Figure 4

Build the dataset:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_precision_stride_modeled_first_model
```

Render the figure:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_precision_stride_modeled_first_model_median
```

Article output:

- `figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.png`

### Figure 5

Build the required current and historical homolog datasets:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical
```

Render the figure:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_xray_homolog_timing_share
```

Article output:

- `figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.png`

### Figure 6

Build the 100% sequence-identity homolog input and the RMSD datasets used by the X-ray RMSD plot group:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100
```

Render the figure:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_xray_rmsd
```

Article output:

- `figures/solution_nmr_monomer_xray_min_median_rmsd_by_year.png`

### Figure 7

Build the precision dataset, the 100% sequence-identity homolog input, and the minimum-X-ray-RMSD dataset:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_precision_stride_modeled_first_model

python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100
```

Render the figure:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_xray_rmsd_precision_correlation
```

Article output:

- `figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation.png`

### Figure 8

Build the quality dataset for panels A-C, populate the PDB cache with refinement-program remarks, and then build the program-cluster dataset for panel D:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_program_counts,solution_nmr_monomer_quality

python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_program_clusters
```

Render all article panels:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters
```

Article outputs:

- Figure 8A: `figures/solution_nmr_monomer_quality_clashscore_by_year.png`
- Figure 8B: `figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year.png`
- Figure 8C: `figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.png`
- Figure 8D: `figures/solution_nmr_monomer_program_cluster_share_by_year.png`

## Technical Pipeline Reference

Detailed documentation for the dataset builder, filtering rules, dataset definitions, and general plot generation lives in [`src/README.md`](src/README.md).
