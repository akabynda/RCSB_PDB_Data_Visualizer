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

## Generating Datasets

Run dataset commands from the repository root. By default,
`src/pdb_dataset_builder.py` writes CSV files to `data/`, downloads reusable PDB
and mmCIF files to `data/pdb_cache/`, and stores STRIDE results in
`data/stride_cache/`.

Build all available datasets in dependency order:

```bash
python src/pdb_dataset_builder.py --datasets all
```

This is a long-running workflow: it uses the RCSB APIs, downloads coordinate
files, runs STRIDE, searches for X-ray homologs, and computes RMSD values. To
build only selected datasets, pass one name or a comma-separated list:

```bash
python src/pdb_dataset_builder.py \
  --datasets method_counts,solution_nmr_weights,solution_nmr_monomer_quality
```

Existing coordinate and STRIDE caches are reused. Some precision and X-ray RMSD
CSVs also support resuming an interrupted calculation. Use
`python src/pdb_dataset_builder.py --help` for dataset names, output-path
options, worker limits, and overwrite flags. The dependency-aware commands for
the article datasets are also given below in Figure 1–8 order.

If an X-ray homolog search finished with transient request failures, rerun the
same command with `--xray-homolog-resume`. Matching completed rows in the 95%
and 100% CSV files and intentional exclusions are retained; missing or failed
entries are searched again. Progress is stored next to the 95% CSV in a
`.resume.tsv` checkpoint file.

Every generated CSV has a sibling `.log` file with the same stem. The log is
recreated on each run and contains warnings and errors for that specific
dataset; an empty log means the build produced no warnings or errors.

For coordinate-level monomer datasets, positive-occupancy CA atoms from both
`ATOM` and `HETATM` records are included in the modeled part and STRIDE core.
However, an NMR structure whose STRIDE core contains any `HETATM` CA residue is
excluded from homology search and the downstream X-ray RMSD datasets.
Structures for which no valid homology query was built are omitted from the
homolog CSV files and from homolog-share denominators; `has_xray_homolog = 0`
is written only after a valid search was actually performed.

## Generating Figures

After the required CSV files exist in `data/`, render every plot group with:

```bash
python src/pdb_plot.py --plots all
```

Each logical figure is written to its own directory. By default, that directory
contains four PNG variants:

```text
figures/<figure_name>/
  <figure_name>.png
  <figure_name>_no_title.png
  <figure_name>_open_axes.png
  <figure_name>_no_title_open_axes.png
```

Render selected plot groups with a comma-separated list, for example:

```bash
python src/pdb_plot.py --plots method_counts,solution_nmr_weight_stats
```

PNG is the default and the article workflow does not require SVG. Run
`python src/pdb_plot.py --help` for all plot groups and input/output path
options.

## RMSD Definitions

The pipeline uses the following RMSD measures.

### NMR ensemble precision

For `solution_nmr_monomer_precision_stride_modeled_first_model`, let `N` be the
number of NMR models, `n` the number of common CA residues in the first-model
STRIDE core, and `r_ij(aligned)` the coordinate of residue `j` in model `i`
after aligning every model to the first NMR model. The per-residue mean is

```text
r_mean,j = (1/N) * sum_i r_ij(aligned),
```

and the reported ensemble precision is

```text
P = sqrt[(1 / (N*n)) * sum_i sum_j ||r_ij(aligned) - r_mean,j||^2].
```

If `RMSD_i` denotes the RMSD of aligned model `i` from the mean coordinates,
then

```text
P = sqrt[mean_i(RMSD_i^2)]
```

### NMR-to-X-ray RMSD

For NMR entry `e` and X-ray homolog candidate `h`, the current X-ray comparison
uses matched standard-`ATOM` CA coordinates from the first NMR model and the
first X-ray model:

```text
d_eh = RMSD_superposed(NMR_e,model1, Xray_h,model1)
```

`solution_nmr_monomer_xray_rmsd_extremes` stores
`d_e,min = min_h(d_eh)` and `d_e,max = max_h(d_eh)`. Figure 6,
`solution_nmr_monomer_xray_min_median_rmsd_by_year`, reports

```text
Y_y = median_{e: year(e)=y}(d_e,min)
```

Figure 7 correlates `d_e,min` with the separately calculated NMR ensemble
precision `P`.

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

- `figures/pdb_method_trends/pdb_method_trends.png`

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

- Figure 2A: `figures/solution_nmr_mean_weight_by_year/solution_nmr_mean_weight_by_year.png`
- Figure 2B: `figures/solution_nmr_area_share_by_weight_category/solution_nmr_area_share_by_weight_category.png`

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

- `figures/solution_nmr_monomer_stride_modeled_first_model_by_year/solution_nmr_monomer_stride_modeled_first_model_by_year.png`

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

- `figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.png`

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

- `figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.png`

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

- `figures/solution_nmr_monomer_xray_min_median_rmsd_by_year/solution_nmr_monomer_xray_min_median_rmsd_by_year.png`

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

- `figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation/solution_nmr_monomer_xray_min_rmsd_precision_correlation.png`

### Figure 8

Build the quality dataset for panels A-C, populate the PDB cache with refinement-program remarks, and then build the program-cluster dataset for panel D:

```bash
python src/pdb_dataset_builder.py --datasets solution_nmr_program_counts,solution_nmr_monomer_quality

python src/pdb_dataset_builder.py --datasets solution_nmr_monomer_program_clusters
```

Programs are collected from both `REMARK 3 PROGRAM` and `REMARK 210 SOFTWARE USED`, including wrapped `REMARK 210` lines. The clusters are `AMBER`, `ARIA`, `CNS`, `CYANA`, `DISCOVER`, `DIANA_DYANA` (`DIANA` or `DYANA`), `XPLOR` (without `NIH`), `XPLOR_NIH`, and `OTHER`. A structure with `n` distinct known clusters contributes `1/n` to each; if none are found, it contributes `1` to `OTHER`. Matching includes audited spelling/separator aliases and avoids the false positives `VARIAN` → `ARIA` and `DISCOVERY STUDIO` → `DISCOVER`.

Render all article panels:

```bash
python src/pdb_plot.py --plots solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters
```

Article outputs:

- Figure 8A: `figures/solution_nmr_monomer_quality_clashscore_by_year/solution_nmr_monomer_quality_clashscore_by_year.png`
- Figure 8B: `figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year/solution_nmr_monomer_quality_ramachandran_outliers_by_year.png`
- Figure 8C: `figures/solution_nmr_monomer_quality_sidechain_outliers_by_year/solution_nmr_monomer_quality_sidechain_outliers_by_year.png`
- Figure 8D: `figures/solution_nmr_monomer_program_cluster_share_by_year/solution_nmr_monomer_program_cluster_share_by_year.png`

## Technical Pipeline Reference

Detailed documentation for the dataset builder, filtering rules, dataset definitions, and general plot generation lives in [`src/README.md`](src/README.md).
