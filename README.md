# RCSB PDB Data Visualizer

The collector writes CSV files to `data/` and uses `data/pdb_cache/` for cached
PDB coordinate files. Large datasets need network access and enough disk space
for the cache.

## Quick Start

Show all available collector options:

```bash
python pdb_data_collector.py --help
```

Run every available dataset:

```bash
python pdb_data_collector.py --datasets all
```

`all` can take a long time. Some datasets download PDB files, run STRIDE, and
compute RMSD values.

Run one dataset:

```bash
python pdb_data_collector.py --datasets method_counts
```

Run several datasets:

```bash
python pdb_data_collector.py \
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

All `solution_nmr_*` datasets start from entries whose experimental method is
exactly `SOLUTION NMR`. Entries with multiple experimental methods are excluded.
For example, an entry that lists both `SOLUTION NMR` and another method is not
used by these datasets.

The monomer datasets do not use all proteins. They keep only SOLUTION NMR
protein monomers that pass several structural filters:

- the entry has more than one deposited model;
- the entry has exactly one polymer entity;
- that polymer entity is a protein, with entity type `polypeptide(L)` or
  `polypeptide(D)`;
- the polymer entity has exactly one chain ID in `pdbx_strand_id`;
- datasets that need residue-level analysis also require exactly one polymer
  instance, a valid sequence length, modeled residues, and usable chain/residue
  mapping.

`method_counts` and `membrane_protein_counts` are broader summary datasets.
They intentionally count method trends across X-ray, cryo-EM, and NMR categories.
The strict `SOLUTION NMR` rule above applies to the datasets whose names begin
with `solution_nmr_`.

## Modeled Part

The modeled part is the subset of polymer residues that have usable coordinates
in the deposited structure.

The collector starts from the polymer sequence and removes residues marked by
RCSB instance features such as:

- `UNOBSERVED_RESIDUE_XYZ`
- `ZERO_OCCUPANCY_RESIDUE_XYZ`
- `UNMODELED_RESIDUE_XYZ`
- `MISSING_RESIDUE`

The code tracks both numbering systems when possible:

- label sequence IDs: canonical entity sequence numbering;
- author sequence IDs: residue numbering used in the coordinate file.

When an `auth_to_entity_poly_seq_mapping` is available, label IDs are mapped to
author IDs before coordinate-level calculations.

## Core Region

The core region is the residue span used for the most structure-sensitive
comparisons. In the active STRIDE-based datasets, it is derived from the first
model only.

The collector runs STRIDE on the first model, keeps only modeled residues, and
identifies residues assigned to core secondary-structure states:

- `H`: alpha helix
- `G`: 3-10 helix
- `I`: pi helix
- `E`: beta strand
- `B`: isolated beta bridge

The STRIDE core region is the continuous author-residue span from the first to
the last modeled residue with one of those STRIDE core states. Entries are
skipped when no usable core can be found. Homolog search also requires a usable
core sequence, and the current implementation skips very short cores.

## STRIDE

The following datasets require a STRIDE executable:

- `solution_nmr_monomer_stride_modeled_first_model`
- `solution_nmr_monomer_precision_stride_modeled_first_model`
- `solution_nmr_monomer_xray_homologs`

If `stride` is available in `PATH`, the collector finds it automatically. It
also checks `/tmp/stride_src/src/stride`. To pass an explicit path:

```bash
python pdb_data_collector.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model \
  --solution-nmr-monomer-stride-executable /path/to/stride
```

## Useful Options

- `--datasets`: dataset kind list, or `all`.
- `--workers`: parallel workers for GraphQL/API calls.
- `--batch-size`: GraphQL batch size.
- `--page-size`: RCSB Search API page size.
- `--log-level`: logging level, for example `INFO` or `DEBUG`.

Long-running calculations:

- `--precision-max-entries`: limit the number of entries processed for
  precision calculations.
- `--precision-workers`: worker count for precision RMSD calculations.
- `--precision-overwrite`: recompute the precision CSV from scratch.
- `--xray-rmsd-max-entries`: limit the number of entries processed for X-ray
  RMSD calculations.
- `--xray-rmsd-workers`: worker count for X-ray RMSD calculations.
- `--xray-rmsd-overwrite`: recompute the X-ray RMSD CSV from scratch.
- `--xray-rmsd-sequence-identity {95,100}`: choose which homolog CSV is used by
  the X-ray RMSD datasets.

## Recommended Run Order

Some datasets read CSV files produced by earlier datasets. A practical run order
is:

```bash
python pdb_data_collector.py \
  --datasets method_counts,membrane_protein_counts,solution_nmr_weights

python pdb_data_collector.py \
  --datasets solution_nmr_program_counts,solution_nmr_monomer_quality

python pdb_data_collector.py \
  --datasets solution_nmr_monomer_program_clusters

python pdb_data_collector.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model,solution_nmr_monomer_precision_stride_modeled_first_model

python pdb_data_collector.py \
  --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical

python pdb_data_collector.py \
  --datasets solution_nmr_monomer_xray_rmsd,solution_nmr_monomer_xray_rmsd_extremes \
  --xray-rmsd-sequence-identity 100

python pdb_data_collector.py \
  --datasets solution_nmr_monomer_xray_rmsd_historical,solution_nmr_monomer_xray_rmsd_extremes_historical \
  --xray-rmsd-sequence-identity 100
```

To produce 95% sequence-identity RMSD datasets, repeat the RMSD commands with
`--xray-rmsd-sequence-identity 95`. Use custom output paths if you need to keep
both 95% and 100% RMSD CSV files at the same time.

## Dataset Reference

### `method_counts`

Counts PDB entries by deposition year and broad experimental-method category:
X-ray, cryo-EM, and NMR.

The NMR category combines exact single-method `SOLUTION NMR` and exact
single-method `SOLID-STATE NMR` entries under the `NMR` label.

Output:

- `data/pdb_method_counts_by_year.csv`

### `membrane_protein_counts`

Counts entries with membrane-protein annotations by deposition year. It also
writes a method split for membrane entries.

Membrane annotations come from RCSB annotation types such as OPM, PDBTM,
MemProtMD, and mpstruc.

Outputs:

- `data/membrane_protein_counts_by_year.csv`
- `data/membrane_protein_method_counts_by_year.csv`

### `solution_nmr_program_counts`

Collects exact single-method `SOLUTION NMR` entries, downloads PDB files into
the cache, and extracts refinement program names from PDB remarks. The output
shows refinement-program usage by year.

Output:

- `data/solution_nmr_program_counts_by_year.csv`

Useful options:

- `--solution-nmr-program-cache-dir`
- `--solution-nmr-program-cache-only`

### `solution_nmr_monomer_program_clusters`

Assigns SOLUTION NMR protein monomers to refinement-program clusters and
summarizes validation quality by year and cluster.

This dataset requires an existing quality CSV and cached PDB files with
refinement program remarks.

Requires:

- `data/solution_nmr_monomer_quality_metrics.csv`
- PDB files in `data/pdb_cache/`

Outputs:

- `data/solution_nmr_monomer_program_cluster_assignments.csv`
- `data/solution_nmr_monomer_program_cluster_quality_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total.csv`

### `solution_nmr_weights`

Collects exact single-method `SOLUTION NMR` entries and calculates molecular
weights. The output includes the RCSB entry molecular weight, the maximum
polymer molecular weight, the total polymer weight, and the modeled-sequence
weight when available.

This dataset is not limited to protein monomers; it can include supported
protein and nucleic-acid polymer types.

Output:

- `data/solution_nmr_structure_weights.csv`

### `solution_nmr_monomer_stride_modeled_first_model`

Runs STRIDE on the first model of each eligible SOLUTION NMR protein monomer and
summarizes secondary-structure fractions over the modeled part only.

The output includes RCSB secondary-structure coverage and STRIDE state
fractions for helix, beta, turn, and coil categories.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Output:

- `data/solution_nmr_monomer_stride_modeled_first_model.csv`

### `solution_nmr_monomer_precision_stride_modeled_first_model`

Computes NMR ensemble precision for eligible SOLUTION NMR protein monomers.
Precision is measured as RMSD between deposited models after alignment to an
average structure. The residue range is the STRIDE core region from the first
model.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Output:

- `data/solution_nmr_monomer_precision_stride_modeled_first_model.csv`

### `solution_nmr_monomer_quality`

Collects validation metrics for eligible SOLUTION NMR protein monomers:
clashscore, Ramachandran outlier percentage, and sidechain rotamer outlier
percentage.

Output:

- `data/solution_nmr_monomer_quality_metrics.csv`

### `solution_nmr_monomer_xray_homologs`

Builds a STRIDE-core query sequence for each eligible SOLUTION NMR protein
monomer and searches RCSB for X-ray polymer-entity homologs. It writes separate
CSV files for 95% and 100% sequence identity.

Candidates are checked against the modeled NMR core sequence so that downstream
RMSD calculations compare a residue range that is actually modeled.

Requires:

- STRIDE executable
- PDB files in `data/pdb_cache/`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

### `solution_nmr_monomer_xray_homologs_historical`

Filters the X-ray homolog CSV files to keep only X-ray structures released no
later than the deposit date of the corresponding NMR entry.

This supports historical analysis: it answers which X-ray homologs were already
available when the NMR structure was deposited.

Requires:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

### `solution_nmr_monomer_xray_rmsd`

Computes CA RMSD between the NMR STRIDE core region and the best matching X-ray
homolog candidate. The homolog input is selected with
`--xray-rmsd-sequence-identity`.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd.csv`

### `solution_nmr_monomer_xray_rmsd_historical`

Same calculation as `solution_nmr_monomer_xray_rmsd`, but using only historical
homologs released no later than the NMR deposit date.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_historical.csv`

### `solution_nmr_monomer_xray_rmsd_extremes`

Computes the minimum and maximum CA RMSD among suitable X-ray homolog
candidates for each eligible NMR monomer. This captures the spread between the
best and worst modeled homolog matches.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes.csv`

### `solution_nmr_monomer_xray_rmsd_extremes_historical`

Historical version of `solution_nmr_monomer_xray_rmsd_extremes`. It computes
minimum and maximum CA RMSD using only X-ray homologs that were already released
when the corresponding NMR entry was deposited.

Requires one of:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv`

## Plot Generation

After the CSV files are ready, build figures with `pdb_plot.py`.

By default, `pdb_plot.py` reads the standard CSV paths in `data/` and writes
PNG/SVG figures to `figures/`.

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

Input and output paths can be overridden with the corresponding flags, for
example:

```bash
python pdb_plot.py \
  --counts-input data/pdb_method_counts_by_year.csv \
  --annual-output-png figures/pdb_method_trends.png
```

Run `python pdb_plot.py --help` to see all plot groups and path
options.
