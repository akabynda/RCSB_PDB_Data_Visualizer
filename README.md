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
python pdb_dataset_builder.py --datasets method_counts
```

Render the figure:

```bash
python pdb_plot.py --plots method_counts
```

Article output:

- `figures/pdb_method_trends.png`

### Figure 2

Build the shared dataset for both panels:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_weights
```

Render both article panels:

```bash
python pdb_plot.py --plots solution_nmr_weight_stats,solution_nmr_period_area_share
```

Article outputs:

- Figure 2A: `figures/solution_nmr_mean_weight_by_year.png`
- Figure 2B: `figures/solution_nmr_area_share_by_weight_category.png`

### Figure 3

Build the dataset:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_monomer_stride_modeled_first_model
```

Render the figure:

```bash
python pdb_plot.py --plots solution_nmr_monomer_stride_modeled_first_model
```

Article output:

- `figures/solution_nmr_monomer_stride_modeled_first_model_by_year.png`

### Figure 4

Build the dataset:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_monomer_precision_stride_modeled_first_model
```

Render the figure:

```bash
python pdb_plot.py --plots solution_nmr_monomer_precision_stride_modeled_first_model_median
```

Article output:

- `figures/solution_nmr_monomer_precision_stride_modeled_first_model_median_rmsd_by_year.png`

### Figure 5

Build the required current and historical homolog datasets:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical
```

Render the figure:

```bash
python pdb_plot.py --plots solution_nmr_monomer_xray_homolog_timing_share
```

Article output:

- `figures/solution_nmr_monomer_xray_homologs_95_timing_share_by_year.png`

### Figure 6

Build the 100% sequence-identity homolog input and the RMSD datasets used by the X-ray RMSD plot group:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100
```

Render the figure:

```bash
python pdb_plot.py --plots solution_nmr_monomer_xray_rmsd
```

Article output:

- `figures/solution_nmr_monomer_xray_min_median_rmsd_by_year.png`

### Figure 7

Build the precision dataset, the 100% sequence-identity homolog input, and the minimum-X-ray-RMSD dataset:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_monomer_precision_stride_modeled_first_model

python pdb_dataset_builder.py --datasets solution_nmr_monomer_xray_homologs

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100
```

Render the figure:

```bash
python pdb_plot.py --plots solution_nmr_monomer_xray_rmsd_precision_correlation
```

Article output:

- `figures/solution_nmr_monomer_xray_min_rmsd_precision_correlation.png`

### Figure 8

Build the quality dataset for panels A-C, populate the PDB cache with refinement-program remarks, and then build the program-cluster dataset for panel D:

```bash
python pdb_dataset_builder.py --datasets solution_nmr_program_counts,solution_nmr_monomer_quality

python pdb_dataset_builder.py --datasets solution_nmr_monomer_program_clusters
```

Render all article panels:

```bash
python pdb_plot.py --plots solution_nmr_monomer_quality,solution_nmr_monomer_program_clusters
```

Article outputs:

- Figure 8A: `figures/solution_nmr_monomer_quality_clashscore_by_year.png`
- Figure 8B: `figures/solution_nmr_monomer_quality_ramachandran_outliers_by_year.png`
- Figure 8C: `figures/solution_nmr_monomer_quality_sidechain_outliers_by_year.png`
- Figure 8D: `figures/solution_nmr_monomer_program_cluster_share_by_year.png`

## Quick Start

The dataset builder writes CSV files to `data/` and uses `data/pdb_cache/` for cached PDB coordinate files. Large datasets need network access and enough disk space for the cache.

Show all available dataset-builder options:

```bash
python pdb_dataset_builder.py --help
```

Run every available dataset:

```bash
python pdb_dataset_builder.py --datasets all
```

`all` can take a long time. Some datasets download PDB files, run STRIDE, and compute RMSD values.

Run one dataset:

```bash
python pdb_dataset_builder.py --datasets method_counts
```

Run several datasets:

```bash
python pdb_dataset_builder.py \
  --datasets method_counts,membrane_protein_counts,solution_nmr_weights
```

## Dataset Selection

Use `--datasets` with one dataset kind, a comma-separated list, or `all`.

Available dataset kinds:

- `method_counts`
- `membrane_protein_counts`
- `solution_nmr_program_counts`
- `solution_nmr_monomer_program_clusters`
- `solution_nmr_weights`
- `solution_nmr_monomer_stride_modeled_first_model`
- `solution_nmr_monomer_precision_stride_modeled_first_model`
- `solution_nmr_monomer_quality`
- `solution_nmr_monomer_xray_homologs`
- `solution_nmr_monomer_xray_homologs_historical`
- `solution_nmr_monomer_xray_rmsd`
- `solution_nmr_monomer_xray_rmsd_historical`
- `solution_nmr_monomer_xray_rmsd_extremes`
- `solution_nmr_monomer_xray_rmsd_extremes_historical`

## Important Filtering Rules

All `solution_nmr_*` datasets start from entries whose experimental method is exactly `SOLUTION NMR`. Entries with multiple experimental methods are excluded. For example, an entry that lists both `SOLUTION NMR` and another method is not used by these datasets.

The monomer datasets (`monomer_*`) do not use all proteins. They keep only protein monomers that pass several structural filters:

- the entry has more than one deposited model;
- the entry has exactly one polymer entity;
- that polymer entity is a protein, with entity type `polypeptide(L)` or `polypeptide(D)`;
- the polymer entity has exactly one chain ID in `pdbx_strand_id`;

`method_counts` and `membrane_protein_counts` are broader summary datasets. They intentionally count method trends across X-ray, cryo-EM, and NMR categories.

## Modeled Part

For the coordinate-level monomer datasets, the modeled part is defined directly from the first deposited coordinate model in the PDB file. A residue is counted as modeled only when that first model contains an `ATOM` CA record for it with positive occupancy. Residues without a first-model CA atom, and residues whose first-model CA occupancy is zero, are excluded.

All downstream residue-level calculations use exactly this first-model modeled residue set and keep the author residue IDs from the PDB file instead of mapping label IDs from RCSB metadata. For STRIDE-based analyses, STRIDE is run on the first model and its assignments are then restricted to these modeled residues before secondary-structure fractions or core ranges are computed.

## Core Region

The core region is the residue span used for the most structure-sensitive comparisons. In the active STRIDE-based datasets, it is derived from the first model only.

The builder runs STRIDE on the first model, keeps only modeled residues, and identifies residues assigned to core secondary-structure states:

- `H`: alpha helix
- `G`: 3-10 helix
- `I`: pi helix
- `E`: beta strand
- `B`: isolated beta bridge

The STRIDE core region is the author-residue span from the first to the last modeled residue with one of those STRIDE core states. Author residue numbering does not have to be contiguous: downstream sequence and coordinate operations use the actually present CA residues inside that span. Entries are skipped when no usable core can be found. Homolog search also requires a usable core sequence, and the current implementation skips very short cores.

## STRIDE

The following datasets require a STRIDE executable:

- `solution_nmr_monomer_stride_modeled_first_model`
- `solution_nmr_monomer_precision_stride_modeled_first_model`
- `solution_nmr_monomer_xray_homologs`

If `stride` is available in `PATH`, the builder finds it automatically. It also checks `/tmp/stride_src/src/stride`. To pass an explicit path:

```bash
python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model \
  --solution-nmr-monomer-stride-executable /path/to/stride
```

First-model STRIDE state maps are cached by structure in `data/stride_cache/` by default and reused across STRIDE-based datasets. Use `--stride-cache-dir` to change that location.

## Useful Options

- `--datasets`: dataset kind list, or `all`.
- `--workers`: parallel workers for GraphQL/API calls.
- `--batch-size`: GraphQL batch size.
- `--page-size`: RCSB Search API page size.
- `--log-level`: logging level, for example `INFO` or `DEBUG`.

Long-running calculations:

- `--precision-max-entries`: limit the number of entries processed for precision calculations.
- `--precision-workers`: worker count for precision RMSD calculations.
- `--precision-overwrite`: recompute the precision CSV from scratch.
- `--xray-rmsd-max-entries`: limit the number of entries processed for X-ray RMSD calculations.
- `--xray-rmsd-workers`: worker count for X-ray RMSD calculations.
- `--xray-rmsd-overwrite`: recompute the X-ray RMSD CSV from scratch.
- `--xray-rmsd-sequence-identity {95,100}`: choose which homolog CSV is used by the X-ray RMSD datasets.

## Recommended Run Order

Some datasets read CSV files produced by earlier datasets. A practical run order is:

```bash
python pdb_dataset_builder.py \
  --datasets method_counts,membrane_protein_counts,solution_nmr_weights

python pdb_dataset_builder.py \
  --datasets solution_nmr_program_counts,solution_nmr_monomer_quality

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_program_clusters

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model,solution_nmr_monomer_precision_stride_modeled_first_model

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_rmsd_historical,solution_nmr_monomer_xray_rmsd_extremes_historical \
  --xray-rmsd-sequence-identity 100
```

To produce 95% sequence-identity RMSD datasets, repeat the RMSD commands with `--xray-rmsd-sequence-identity 95`. Use custom output paths if you need to keep both 95% and 100% RMSD CSV files at the same time.

## Dataset Reference

### `method_counts`

Counts PDB entries by deposition year and broad experimental-method category: X-ray, cryo-EM, and NMR.

The NMR category combines exact single-method `SOLUTION NMR` and exact single-method `SOLID-STATE NMR` entries under the `NMR` label.

Output:

- `data/pdb_method_counts_by_year.csv`

### `membrane_protein_counts`

Counts entries with membrane-protein annotations by deposition year. It also writes a method split for membrane entries.

Membrane annotations come from RCSB annotation types such as OPM, PDBTM, MemProtMD, and mpstruc.

Outputs:

- `data/membrane_protein_counts_by_year.csv`
- `data/membrane_protein_method_counts_by_year.csv`

### `solution_nmr_program_counts`

Collects exact single-method `SOLUTION NMR` entries, downloads PDB files into the cache, and extracts refinement program names from PDB remarks. The output shows refinement-program usage by year.

Output:

- `data/solution_nmr_program_counts_by_year.csv`

Useful options:

- `--solution-nmr-program-cache-dir`
- `--solution-nmr-program-cache-only`

### `solution_nmr_monomer_program_clusters`

Assigns SOLUTION NMR protein monomers to all unique refinement-program clusters mentioned in PDB `REMARK 3 PROGRAM` records. This is multi-label: one structure can contribute to multiple clusters. In the per-cluster CSVs, `structure_count` therefore means cluster mentions/assignments; yearly totals remain unique structure counts. `OTHER` is used only when no known program cluster is found.

This dataset requires an existing quality CSV and cached PDB files with refinement program remarks.

Requires:

- `data/solution_nmr_monomer_quality_metrics.csv`
- PDB files in `data/pdb_cache/`

Outputs:

- `data/solution_nmr_monomer_program_cluster_assignments.csv`
- `data/solution_nmr_monomer_program_cluster_quality_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total.csv`

### `solution_nmr_weights`

Collects exact single-method `SOLUTION NMR` entries and reads one total molecular-weight value per entry from RCSB entry metadata.

Output:

- `data/solution_nmr_structure_weights.csv`

### `solution_nmr_monomer_stride_modeled_first_model`

Runs STRIDE on the first model of each eligible SOLUTION NMR protein monomer, restricts the STRIDE assignments to the modeled residues from that same first model, and summarizes the resulting state fractions over the modeled part only.

The output stores the modeled residue span and STRIDE fractions for `H`, `G`, `I`, `E`, `B`, `T`, and `C`.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Output:

- `data/solution_nmr_monomer_stride_modeled_first_model.csv`
- cached first-model STRIDE state maps in `data/stride_cache/`

### `solution_nmr_monomer_precision_stride_modeled_first_model`

Computes NMR ensemble precision for eligible SOLUTION NMR protein monomers. Models are aligned to the first model, then precision is measured as `sqrt(1 / (N*n) * sum_i sum_j |r_ij - r_mean,j|^2)` across models and CA atoms. The residue range is the STRIDE core region from the first model.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Output:

- `data/solution_nmr_monomer_precision_stride_modeled_first_model.csv`

### `solution_nmr_monomer_quality`

Collects validation metrics for eligible SOLUTION NMR protein monomers: clashscore, Ramachandran outlier percentage, and sidechain rotamer outlier percentage.

Output:

- `data/solution_nmr_monomer_quality_metrics.csv`

### `solution_nmr_monomer_xray_homologs`

Builds a STRIDE-core query sequence for each eligible SOLUTION NMR protein monomer and searches RCSB for X-ray polymer-entity homologs. It writes separate CSV files for 95% and 100% sequence identity.

Candidates are checked against the modeled NMR core sequence so that downstream RMSD calculations compare a residue range that is actually modeled. When checking whether an X-ray candidate models the NMR core sequence, X-ray CA residues from both `ATOM` and `HETATM` records may be used for sequence matching.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

### `solution_nmr_monomer_xray_homologs_historical`

Filters the X-ray homolog CSV files to keep only X-ray structures released no later than the deposit date of the corresponding NMR entry.

This supports historical analysis: it answers which X-ray homologs were already available when the NMR structure was deposited.

Requires:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

### `solution_nmr_monomer_xray_rmsd`

Computes CA RMSD between the NMR STRIDE core region and the best matching X-ray homolog candidate. The homolog input is selected with `--xray-rmsd-sequence-identity`. X-ray `HETATM` CA residues may still be considered while finding candidate sequence matches, but the final RMSD is computed only from matched residue pairs whose NMR and X-ray coordinates both come from standard `ATOM` records.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd.csv`

### `solution_nmr_monomer_xray_rmsd_historical`

Same calculation as `solution_nmr_monomer_xray_rmsd`, but using only historical homologs released no later than the NMR deposit date.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_historical.csv`

### `solution_nmr_monomer_xray_rmsd_extremes`

Computes the minimum and maximum CA RMSD among suitable X-ray homolog candidates for each eligible NMR monomer. This captures the spread between the best and worst modeled homolog matches.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes.csv`

### `solution_nmr_monomer_xray_rmsd_extremes_historical`

Historical version of `solution_nmr_monomer_xray_rmsd_extremes`. It computes minimum and maximum CA RMSD using only X-ray homologs that were already released when the corresponding NMR entry was deposited.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv`

## Plot Generation

After the CSV files are ready, build figures with `pdb_plot.py`.

By default, `pdb_plot.py` reads the standard CSV paths in `data/` and writes PNG/SVG figures to `figures/`.

Build only selected plot groups with `--plots`:

```bash
python pdb_plot.py \
  --plots method_counts,solution_nmr_weight_stats,solution_nmr_monomer_quality
```

Use `all` to build every available plot group:

```bash
python pdb_plot.py --plots all
```

Use `--no-svg` if only PNG output is needed:

```bash
python pdb_plot.py --no-svg
```

Input and output paths can be overridden with the corresponding flags, for example:

```bash
python pdb_plot.py \
  --counts-input data/pdb_method_counts_by_year.csv \
  --annual-output-png figures/pdb_method_trends.png
```

Run `python pdb_plot.py --help` to see all plot groups and path options.
