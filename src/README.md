# Source Pipeline Reference

This document is the technical reference for the two executable modules in
`src/`. For installation and the article reproduction workflow, see the
[project README](../README.md). Unless noted otherwise, run every command below
from the repository root.

## Table of Contents

- [Dataset Builder](#dataset-builder)
- [Dataset Selection](#dataset-selection)
- [Important Filtering Rules](#important-filtering-rules)
- [Modeled Part](#modeled-part)
- [Core Region](#core-region)
- [STRIDE](#stride)
- [Useful Options](#useful-options)
- [Recommended Run Order](#recommended-run-order)
- [Dataset Reference](#dataset-reference)
  - [`method_counts`](#method_counts)
  - [`membrane_protein_counts`](#membrane_protein_counts)
  - [`solution_nmr_program_counts`](#solution_nmr_program_counts)
  - [`solution_nmr_monomer_program_clusters`](#solution_nmr_monomer_program_clusters)
  - [`solution_nmr_weights`](#solution_nmr_weights)
  - [`solution_nmr_monomer_stride_modeled_first_model`](#solution_nmr_monomer_stride_modeled_first_model)
  - [`solution_nmr_monomer_precision_stride_modeled_first_model`](#solution_nmr_monomer_precision_stride_modeled_first_model)
  - [`solution_nmr_monomer_quality`](#solution_nmr_monomer_quality)
  - [`solution_nmr_monomer_experiments`](#solution_nmr_monomer_experiments)
  - [`solution_nmr_monomer_xray_homologs`](#solution_nmr_monomer_xray_homologs)
  - [`solution_nmr_monomer_xray_homologs_historical`](#solution_nmr_monomer_xray_homologs_historical)
  - [`solution_nmr_monomer_xray_rmsd`](#solution_nmr_monomer_xray_rmsd)
  - [`solution_nmr_monomer_xray_rmsd_historical`](#solution_nmr_monomer_xray_rmsd_historical)
  - [`solution_nmr_monomer_xray_rmsd_extremes`](#solution_nmr_monomer_xray_rmsd_extremes)
  - [`solution_nmr_monomer_xray_rmsd_extremes_historical`](#solution_nmr_monomer_xray_rmsd_extremes_historical)
- [Plot Generation](#plot-generation)
- [License](#license)

## Dataset Builder

`src/pdb_dataset_builder.py` builds the CSV datasets used by the plots. It
writes tabular outputs to `data/`, stores downloaded PDB and mmCIF coordinate
files in `data/pdb_cache/`, and reuses first-model STRIDE assignments from
`data/stride_cache/` when needed. Large datasets require network access and
enough disk space for the coordinate cache. Metadata is obtained from the [RCSB
PDB Data and Search APIs][rcsb-apis].

Show all available options:

```bash
python src/pdb_dataset_builder.py --help
```

Build every available dataset:

```bash
python src/pdb_dataset_builder.py --datasets all
```

`all` can take a long time. Some datasets download PDB files, run STRIDE, and
compute RMSD values.

Coordinates are downloaded from RCSB, wwPDB, PDBe, or the EBI archive mirror.
Each cached `.pdb` or `.cif` file has a `.cache.json` sidecar with its checksum,
size, modification time, and available remote validators. Cache entries are
revalidated after 24 hours by default. Set `--pdb-cache-validation-hours 0` to
validate them on every access.

Structures that do not fit the legacy PDB format are downloaded as mmCIF and
converted to per-chain PDB subsets. The cache also stores the resulting chain-ID
mapping. Cache writes are atomic and safe for concurrent workers.

Each output CSV receives a sibling `.log` file containing warnings and errors
from that build. An empty log indicates a clean run.

Each output `name.csv` also receives a paired `name_filtered.csv`. The paired
file has three columns:

- `entry_id`: the RCSB PDB entry rejected from that output;
- `year`: the structure deposition year;
- `reason`: the recorded metadata, eligibility, coordinate, STRIDE, homology,
  historical-date, or RMSD exclusion reason.

The `year` cell is empty when no valid deposition date is available. One entry
can have several rows if it fails independent checks. Duplicate
`entry_id`/`reason` rows are suppressed. A header-only file means that nothing
was filtered. Resumed and derived builds preserve relevant earlier exclusions.

Build one dataset:

```bash
python src/pdb_dataset_builder.py --datasets method_counts
```

Build several datasets:

```bash
python src/pdb_dataset_builder.py \
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
- `solution_nmr_monomer_experiments`
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

The `solution_nmr_weights` dataset additionally requires at least one protein
polymer entity in each entry. This is the same protein-presence filter used by
`method_counts`.

The `solution_nmr_monomer_*` datasets do not use all proteins. They keep only
protein monomers that pass several structural filters:

- the entry has more than one deposited model;
- the entry has exactly one polymer entity;
- that polymer entity is a protein, with entity type `polypeptide(L)` or
  `polypeptide(D)`;
- the polymer entity has exactly one chain ID in `pdbx_strand_id`;
- every coordinate model has the same number of modeled CA positions, as
  defined below.

All `solution_nmr_monomer_*` datasets use this filter. A model-length difference
outside the STRIDE core therefore still excludes the entry.

`method_counts` and `membrane_protein_counts` are broader summary datasets. They
intentionally count method trends across X-ray, cryo-EM, and NMR categories.

## Modeled Part

For coordinate-level monomer datasets, a modeled CA position is an author
residue number with a positive-occupancy `ATOM` or `HETATM` CA record in a
coordinate model. Duplicate records, insertion-code variants, and alternate
locations with the same residue number are collapsed to one position.

The positions in the first model define the STRIDE summary, core endpoints, and
homology query. Precision uses positions shared by all models. NMR-to-X-ray
RMSD uses matched `ATOM` CA pairs from the first NMR and X-ray models.

## Core Region

The STRIDE core is the span from the first to the last modeled residue assigned
one of these states in the first model:

- `H`: alpha helix
- `G`: 3-10 helix
- `I`: pi helix
- `E`: beta strand
- `B`: isolated beta bridge

Both `ATOM` and `HETATM` CA records can define the endpoints. The STRIDE summary
can retain an entry with no core states, but precision and homology datasets
cannot. Homolog search also requires at least 11 modeled CA positions inside
the core.

Homolog search excludes an NMR entry if any modeled CA position in its core is
represented by a `HETATM` record. Entries without a valid query are absent from
the homolog datasets. Therefore, `has_xray_homolog = 0` means that a valid
search found no homolog.

## STRIDE

The following datasets require a STRIDE executable:

- `solution_nmr_monomer_stride_modeled_first_model`
- `solution_nmr_monomer_precision_stride_modeled_first_model`
- `solution_nmr_monomer_xray_homologs`

If `stride` is available in `PATH`, the builder finds it automatically. It also
checks `/tmp/stride_src/src/stride`. The upstream source and build instructions
are available in the [STRIDE repository][stride-repository]. To pass an explicit
executable path:

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model \
  --solution-nmr-monomer-stride-executable /path/to/stride
```

First-model STRIDE state maps are cached by structure in `data/stride_cache/` by
default and reused across STRIDE-based datasets. Use `--stride-cache-dir` to
change that location.

The cached STRIDE result covers the complete first model, including `ATOM` and
`HETATM` records.

## Useful Options

- `--datasets`: dataset kind list, or `all`.
- `--workers`: parallel workers for GraphQL/API calls.
- `--batch-size`: GraphQL batch size.
- `--page-size`: RCSB Search API page size.
- `--log-level`: logging level, for example `INFO` or `DEBUG`.
- `--solution-nmr-monomer-cache-dir`: PDB cache used by the base monomer
  model-length filter and coordinate-level monomer datasets. The legacy option
  name `--solution-nmr-monomer-stride-cache-dir` remains accepted as an alias.
- `--pdb-cache-validation-hours`: remote PDB/mmCIF validation interval; `0`
  validates every access.

Long-running calculations:

- `--xray-homolog-resume`: continue a homolog build from its checkpoint.
- `--precision-max-entries`: limit the number of entries processed for precision
  calculations.
- `--precision-workers`: worker count for precision RMSD calculations.
- `--precision-overwrite`: recompute the precision CSV from scratch.
- `--xray-rmsd-max-entries`: limit the number of entries processed for X-ray
  RMSD calculations.
- `--xray-rmsd-workers`: worker count for X-ray RMSD calculations.
- `--xray-rmsd-overwrite`: recompute the X-ray RMSD CSV from scratch.
- `--xray-rmsd-sequence-identity {95,100}`: choose which homolog CSV is used by
  the X-ray RMSD datasets.

The homolog completion checkpoint is written beside the 95% output as
`<95%-output-stem>.resume.tsv`. Run without `--xray-homolog-resume` to rebuild
the homolog CSV pair and checkpoint from scratch.

## Recommended Run Order

Some datasets read CSV files produced by earlier datasets. A practical run order
is:

```bash
python src/pdb_dataset_builder.py \
  --datasets method_counts,membrane_protein_counts,solution_nmr_weights

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_program_counts,solution_nmr_monomer_quality,solution_nmr_monomer_experiments

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_program_clusters

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model,solution_nmr_monomer_precision_stride_modeled_first_model

python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_xray_homologs,solution_nmr_monomer_xray_homologs_historical

rmsd_datasets=solution_nmr_monomer_xray_rmsd,\
solution_nmr_monomer_xray_rmsd_extremes
python src/pdb_dataset_builder.py \
  --datasets "$rmsd_datasets" \
  --xray-rmsd-sequence-identity 100

historical_rmsd_datasets=solution_nmr_monomer_xray_rmsd_historical,\
solution_nmr_monomer_xray_rmsd_extremes_historical
python src/pdb_dataset_builder.py \
  --datasets "$historical_rmsd_datasets" \
  --xray-rmsd-sequence-identity 100
```

Use `--xray-rmsd-sequence-identity 95` to read candidates from the 95% homolog
CSV. The RMSD calculation itself still requires an exact modeled-core match.
Use different output paths to keep both the 95% and 100% runs.

## Dataset Reference

### `method_counts`

Counts PDB entries by deposition year and broad experimental-method category:
X-ray, cryo-EM, and NMR. Every counted entry must contain at least one protein
polymer entity.

The NMR category includes entries with exactly `SOLUTION NMR`, exactly
`SOLID-STATE NMR`, or exactly the two-method combination `SOLID-STATE NMR` and
`SOLUTION NMR`. All three cases are counted under the `NMR` label.

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

Assigns SOLUTION NMR protein monomers to refinement-program clusters using PDB
`REMARK 3 PROGRAM` and `REMARK 210 SOFTWARE USED`. If a structure names `n`
known clusters, each receives a score of `1/n`, so the total score remains `1`.
`OTHER` is used only when no known cluster is found.

Matching is case-insensitive and accepts versions, separators, compound program
strings, and these aliases:

- `CLUSTER1` — `AMBER`
- `CLUSTER2` — `ARIA`
- `CLUSTER3` — `CNS`
- `CLUSTER4` — `CYANA`
- `CLUSTER5` — `DISCOVER` (standalone `INSIGHT II` rows are included)
- `CLUSTER6` — `DIANA` or `DYANA`
- `CLUSTER7` — `XPLOR` (also `X-PLOR`), only when the name does not contain
  `NIH`
- `CLUSTER8` — `XPLOR_NIH` (also underscore, reversed `NIH-XPLOR`, compact
  `NIHXPLOR`, parenthesized NIH, and the observed `NHI` typo)
- `CLUSTER9` — `OTHER`, used only when no known cluster matches

For example, `CNS ARIA` contributes `0.5` to both `CLUSTER3` and `CLUSTER2`.
This dataset requires the quality CSV and downloads missing PDB files itself.
`solution_nmr_program_counts` is not a prerequisite.

Requires:

- `data/solution_nmr_monomer_quality_metrics.csv`

By default, downloaded PDB files are cached in `data/pdb_cache/`.

Outputs:

- `data/solution_nmr_monomer_program_cluster_assignments.csv`
- `data/solution_nmr_monomer_program_cluster_quality_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total.csv`

### `solution_nmr_weights`

Collects exact single-method `SOLUTION NMR` entries that contain at least one
protein polymer entity and reads one total molecular-weight value per entry from
RCSB entry metadata.

Output:

- `data/solution_nmr_structure_weights.csv`

### `solution_nmr_monomer_stride_modeled_first_model`

Runs STRIDE on the first model of each eligible SOLUTION NMR protein monomer,
restricts the STRIDE assignments to the modeled `ATOM` and `HETATM` CA residues
from that same first model, and summarizes the resulting state fractions over
the modeled part only.

The output stores the modeled residue span and STRIDE fractions for `H`, `G`,
`I`, `E`, `B`, `T`, and `C`.

Unrecognized or missing assignments are counted as `C`. If STRIDE returns no
usable state map, the row is retained with state fractions of `-1.0` and a
secondary-structure value of `200.0`. Figure 3 excludes these sentinel rows. In
the article snapshot, it excludes 97 rows and plots 10,030 through 2024.

Requires:

- STRIDE executable
- PDB coordinates, downloaded automatically into `data/pdb_cache/` when missing

Output:

- `data/solution_nmr_monomer_stride_modeled_first_model.csv`
- cached first-model STRIDE state maps in `data/stride_cache/`

### `solution_nmr_monomer_precision_stride_modeled_first_model`

Computes NMR ensemble precision over the first-model STRIDE core. It uses
positive-occupancy `ATOM` and `HETATM` CA residues found in every coordinate
model. At least three common residues are required.

Every NMR model is first rigidly aligned to the first NMR model. Let `N` be the
number of models, `n` the number of common CA residues, `r_ij(aligned)` the
aligned coordinate of residue `j` in model `i`, and

```text
r_mean,j = (1/N) * sum_i r_ij(aligned).
```

The current ensemble precision is

```text
P = sqrt[(1 / (N*n)) * sum_i sum_j ||r_ij(aligned) - r_mean,j||^2].
```

Requires:

- STRIDE executable
- PDB coordinates, downloaded automatically into `data/pdb_cache/` when missing

Output:

- `data/solution_nmr_monomer_precision_stride_modeled_first_model.csv`

### `solution_nmr_monomer_quality`

Collects validation metrics for eligible SOLUTION NMR protein monomers:
clashscore, Ramachandran outlier percentage, and sidechain rotamer outlier
percentage. The values are read from the first RCSB
`pdbx_vrpt_summary_geometry` record and are not recomputed locally. An entry is
omitted unless all three values are present and numeric.

Output:

- `data/solution_nmr_monomer_quality_metrics.csv`

### `solution_nmr_monomer_experiments`

Collects the `_pdbx_nmr_exptl.type` values (the PDB field
`NMR EXPERIMENTS CONDUCTED`) for every eligible SOLUTION NMR protein monomer.
The CSV contains one row per PDB entry; multiple experiment descriptions are
joined in one cell with a semicolon followed by a space. Entries with no
reported experiments are retained with an empty field.

Output:

- `data/solution_nmr_monomer_experiments.csv`

### `solution_nmr_monomer_xray_homologs`

Builds a modeled STRIDE-core sequence for each eligible NMR monomer and searches
RCSB for X-ray polymer-entity homologs. The query must contain at least 11
residues. Each candidate chain is then checked against its first-model
coordinates.

- The 95% output requires at least `ceil(0.95 * query length)` modeled pairs and
  the same number of identities.
- The 100% output requires a complete one-to-one identity match.
- An NMR core containing a `HETATM` CA is excluded.
- X-ray `ATOM` and `HETATM` CA records can be used for sequence matching.
- Missing resolution does not exclude a candidate and is written as `nan`.

HTTP 5xx failures are retried up to three times. Other failures are not retried
by this mechanism.

Requires:

- STRIDE executable
- PDB coordinates, downloaded automatically into `data/pdb_cache/` when missing

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

### `solution_nmr_monomer_xray_homologs_historical`

Keeps only X-ray homologs released by the NMR deposition date. The NMR entry is
omitted if its deposition date or any candidate release date is missing.

Requires:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

### `solution_nmr_monomer_xray_rmsd`

Computes CA RMSD between an NMR STRIDE core and a matching X-ray homolog. The
input homolog CSV is selected with `--xray-rmsd-sequence-identity`, but the RMSD
stage always requires an exact modeled-core sequence match.

The sequence match may use X-ray `ATOM` and `HETATM` CA records. The RMSD uses
only matched `ATOM` CA pairs from the first NMR and X-ray models and requires at
least three pairs. An NMR core containing a `HETATM` CA is excluded.

```text
d_eh = RMSD_superposed(NMR_e,model1, Xray_h,model1)
```

The coordinates are centered and optimally superposed before RMSD is calculated.

The ordinary RMSD CSV keeps the first usable candidate after sorting by
resolution, entry ID, and entity ID. Missing resolutions sort last. Within an
entity, it chooses the chain with the most CA pairs, using lower RMSD to break a
tie. The `*_extremes` dataset provides minima and maxima across candidates.

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

Computes the minimum and maximum `d_eh` across usable X-ray candidates:

```text
d_e,min = min_h(d_eh)
d_e,max = max_h(d_eh).
```

The yearly minimum-RMSD plot reports:

```text
Y_y = median_{e: year(e)=y}(d_e,min).
```

The precision-correlation plot compares `d_e,min` with ensemble precision `P`.
An entry is written only if at least one candidate produces a usable RMSD.

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

After the CSV files are ready, build figures with `src/pdb_plot.py`.

By default, `src/pdb_plot.py` reads the standard CSV paths in `data/` and writes
four PNG variants of each figure to `figures/<figure_name>/`.

The variants combine title/no title with closed/open top and right axes. PNG is
the default image format. The homolog-timing plot also writes two yearly count
CSVs.

Datasets may contain later records, but plots use only years through 2024.

Build only selected plot groups with `--plots`:

```bash
python src/pdb_plot.py \
  --plots method_counts,solution_nmr_weight_stats,solution_nmr_monomer_quality
```

Use `all` to build every available plot group:

```bash
python src/pdb_plot.py --plots all
```

Use `--svg` to additionally generate SVG files:

```bash
python src/pdb_plot.py --svg
```

Input and output paths can be overridden with the corresponding flags, for
example:

```bash
python src/pdb_plot.py \
  --plots method_counts \
  --counts-input data/pdb_method_counts_by_year.csv \
  --annual-output-png figures/pdb_method_trends.png
```

The output is still grouped automatically, so this example writes
`figures/pdb_method_trends/pdb_method_trends.png` and its three PNG variants.

Run `python src/pdb_plot.py --help` to see all plot groups and path options.

## License

No open-source license is granted for this repository. See
[`../LICENSE`](../LICENSE) for the applicable terms. STRIDE and all Python
dependencies remain subject to their own licenses.

[rcsb-apis]: https://www.rcsb.org/docs/programmatic-access/web-apis-overview
[stride-repository]: https://github.com/MDAnalysis/stride
