# Source Pipeline Reference

This document is the technical reference for the two executable modules in
`src/`. For installation and the article reproduction workflow, see the
[project README](../README.md). Unless noted otherwise, run every command below
from the repository root.

## Table of Contents

- [Dataset Builder](#dataset-builder)
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

Use `--datasets` with one dataset kind, a comma-separated list, or `all`. Valid
kinds are the entries in the [Dataset Reference](#dataset-reference).

```bash
python src/pdb_dataset_builder.py --datasets method_counts
python src/pdb_dataset_builder.py --datasets all
```

Building `all` can take a long time because some datasets download coordinates,
run STRIDE, and compute RMSD values.

Coordinates are downloaded from RCSB, wwPDB, PDBe, or the EBI archive mirror.
Validated coordinate-cache entries have a `.cache.json` sidecar with their
checksum, size, modification time, and available remote validators. An
intermediate `.cif` retained after the legacy-PDB conversion fallback does not
have its own sidecar; validation metadata is stored for the converted `.pdb`.
Cache entries are revalidated after 24 hours by default. Set
`--pdb-cache-validation-hours 0` to validate them on every access.

Structures that do not fit the legacy PDB format are downloaded as mmCIF and
converted to per-chain PDB subsets. The cache also stores the resulting chain-ID
mapping. Per-chain subset PDBs and chain-ID mappings are installed through
unique temporary files and atomic replacement.

A per-chain subset is reused only when its requested chain set and source mmCIF
SHA-256 still match. Its chain-ID mapping is also embedded in the metadata
transaction, so an absent or truncated mapping cannot be combined with an
otherwise valid subset. Remote revalidation uses `ETag` and `Last-Modified`
when available.

All coordinate artifacts for one PDB ID—legacy PDB, mmCIF, converted subsets,
chain maps, and metadata—share one keyed lock. Cache state is checked again
after acquiring the lock. A transaction revision lets concurrent waiters reuse
the result even with `--pdb-cache-validation-hours 0`; a later sequential call
still performs the requested validation. Locks are per entry, so unrelated PDB
downloads remain parallel. On POSIX, persistent lock files in
`data/pdb_cache/.locks/` also protect the cache between builder processes.

First-model X-ray CA residues and coordinates are cached beside each parsed PDB
as `*.pdb.first_model_ca.v1.npz`. One cold pass parses every chain and records
residue order, identity, `ATOM`/`HETATM` flags, and selected coordinates. The
pickle-free payload contains both a schema version and parser revision and is
accepted only when the source PDB SHA-256 matches. A changed source, old
revision, or damaged NPZ is reparsed and atomically replaced. The cache is
shared by 95%/100% homology checks and all X-ray RMSD views.

Each primary dataset CSV receives a sibling `.log` file containing warnings and
errors from that build. Logs are recreated at the start of a run; multi-output
datasets send shared warnings and errors to each affected log. An empty log
indicates a clean run.

Each primary dataset `name.csv` also receives a paired `name_filtered.csv`. The
paired file has three columns:

- `entry_id`: the RCSB PDB entry rejected from that output;
- `year`: the structure deposition year;
- `reason`: the recorded metadata, eligibility, coordinate, STRIDE, homology,
  historical-date, or RMSD exclusion reason.

The `year` cell is empty when no valid deposition date is available. One entry
can have several rows if it fails independent checks. Duplicate
`entry_id`/`reason` rows are suppressed. A header-only file means that nothing
was filtered. Fresh builds recreate the report. A resumed
homolog build preserves its earlier exclusions, while a derived dataset imports
upstream exclusions and then appends its own. Shared multi-output exclusions are
written to every affected report; output-specific exclusions stay in their own
report.

The 95% and 100% X-ray homolog outputs additionally receive sibling
`*_rejected.csv` reports. These are candidate-level audit files, distinct from
the NMR-structure-level `*_filtered.csv` reports. Their schema and rejection
rules are described in the dataset reference below.

## Important Filtering Rules

All `solution_nmr_*` datasets start from entries whose experimental method is
exactly `SOLUTION NMR`. Entries with multiple experimental methods are excluded.

The `solution_nmr_weights` dataset additionally requires at least one protein
polymer entity in each entry. This is the same protein-presence filter used by
`method_counts`.

The `solution_nmr_monomer_*` datasets keep only protein monomers that pass these
structural filters:

- the entry metadata reports more than one deposited model, and the coordinate
  file contains at least two parsed models;
- the entry has a valid deposition year;
- the entry has exactly one polymer entity;
- that polymer entity is a protein, with entity type `polypeptide(L)` or
  `polypeptide(D)`;
- the polymer entity has exactly one chain ID in `pdbx_strand_id`;
- every coordinate model has the same number of modeled CA positions, as
  defined below.

A model-length difference outside the STRIDE core therefore still excludes the
entry. The equal-length test compares the number of selected CA positions, not
their residue-number sets; precision later uses the positions shared by all
models. Direct monomer datasets apply the filter themselves, while
program-cluster, historical, and RMSD datasets inherit it through their input
CSVs.

The STRIDE summary additionally requires exactly one polymer entity instance.

## Modeled Part

For coordinate-level monomer datasets, a modeled CA position is an author
residue number with a positive-occupancy `ATOM` or `HETATM` CA record in a
coordinate model. Duplicate records, insertion-code variants, and alternate
locations with the same residue number are collapsed to one position.

The selection is deterministic: `ATOM` is preferred to `HETATM`; insertion codes
are ordered blank first and then lexically; higher occupancy is preferred next;
alternate locations are ordered blank, `A`, `1`, then lexically. Residue-level
operations retain PDB author residue numbers rather than replacing them with
RCSB label IDs. The parser also remembers that a positive-occupancy `HETATM` CA
was present when an `ATOM` CA wins this collapsing step at the same author
residue number. This prevents the stricter homology filters from masking the
`HETATM` record.

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
cannot. The core states define only the endpoints; downstream steps start from
the observed CA positions between them, not only residues in those states.

## STRIDE

The following datasets require a STRIDE executable:

- `solution_nmr_monomer_stride_modeled_first_model`
- `solution_nmr_monomer_precision_stride_modeled_first_model`
- `solution_nmr_monomer_xray_homologs`

The builder downloads missing PDB coordinates into `data/pdb_cache/`.

The builder resolves STRIDE in this order:

1. the path passed with `--solution-nmr-monomer-stride-executable`;
2. an executable named `stride` in `PATH`;
3. the versioned, platform-specific managed build under `data/stride/`;
4. the legacy local build at `/tmp/stride_src/src/stride`;
5. a new automatic managed installation.

The automatic path is used only when no explicit value was supplied. A bad
explicit path fails instead of silently selecting or downloading another
binary. To pass an explicit executable path:

```bash
python src/pdb_dataset_builder.py \
  --datasets solution_nmr_monomer_stride_modeled_first_model \
  --solution-nmr-monomer-stride-executable /path/to/stride
```

For a first-time managed installation on macOS or Linux, the builder clones the
[STRIDE repository][stride-repository], checks out the fixed revision
`867a5eb0f2479cb16615512a53ee472c54649505`, runs
`make -C <checkout>/src stride`, validates the resulting executable, and
publishes the completed checkout at
`data/stride/867a5eb0f2479cb16615512a53ee472c54649505/<system>-<architecture>/`.
Clone/build commands use argument lists rather than a shell, each have a
five-minute timeout, and are serialized across threads and POSIX processes. A
failed first installation is not published; the next run can retry cleanly. If
an existing versioned checkout has lost its binary, its Git revision and tracked
files are verified before its Makefile is run again.

Automatic setup requires Git, GNU Make, a C compiler (`gcc`, `cc`, or `clang`),
and GitHub access. Use `--stride-install-dir` to change the managed installation
root. Native Windows users should use WSL or pass a prebuilt executable. The
versioned directory means that changing the pinned revision creates a separate
installation rather than silently replacing an older one.

First-model STRIDE state maps are cached by structure in `data/stride_cache/` by
default and reused across STRIDE-based datasets. A cache entry is accepted only
when the SHA-1 of the complete first-model coordinate text still matches. The
STRIDE path and version are not part of that key, so clear this cache after
changing STRIDE. Use `--stride-cache-dir` to change the cache location.

## Useful Options

- `--workers`: parallel workers for API, coordinate, STRIDE, and homology work.
- `--batch-size`: GraphQL batch size.
- `--page-size`: RCSB Search API page size.
- `--log-level`: logging level, for example `INFO` or `DEBUG`.
- `--solution-nmr-monomer-cache-dir`: PDB cache used by the base monomer
  model-length filter and coordinate-level monomer datasets.
- `--pdb-cache-validation-hours`: remote PDB/mmCIF validation interval; `0`
  validates every access.
- `--stride-install-dir`: root for the pinned, automatically built STRIDE
  checkout (default: `data/stride`).
- `--stride-cache-dir`: cache for per-structure STRIDE assignments; this is
  separate from the managed source and executable.

Long-running calculations:

- `--resume`: retain existing homolog, precision, and selected X-ray RMSD
  results and process unfinished entries.
- `--precision-workers`: worker count for precision RMSD calculations.
- `--xray-rmsd-workers`: worker count for X-ray RMSD calculations.
- `--xray-rmsd-sequence-identity {95,100}`: choose which homolog CSV is used by
  the X-ray RMSD datasets (default: `100`).

The homolog completion checkpoint is written beside the 95% output as
`<95%-output-stem>.resume.tsv`. With `--resume`, valid paired 95%/100% rows are
retained only after their rejected-candidate audit was checkpointed; entries
checkpointed as `ineligible` are also retained. Unfinished, failed, or legacy
unaudited pairs are retried. Without the flag, the homolog CSV pair, rejected
reports, and checkpoint are rebuilt.

Every selected dataset is rebuilt by default. With `--resume`, precision and
each selected RMSD output reuse valid existing rows independently, so a
completed ordinary CSV does not suppress missing extremes or historical rows.
Resume should be used only with the same inputs and calculation settings as the
interrupted run.

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
solution_nmr_monomer_xray_rmsd_extremes,\
solution_nmr_monomer_xray_rmsd_historical,\
solution_nmr_monomer_xray_rmsd_extremes_historical
python src/pdb_dataset_builder.py \
  --datasets "$rmsd_datasets" \
  --xray-rmsd-sequence-identity 100
```

Selecting related RMSD kinds together lets the builder reuse candidate
calculations. Input selection and historical behavior are described in the
[`solution_nmr_monomer_xray_rmsd`](#solution_nmr_monomer_xray_rmsd) reference.

## Dataset Reference

### `method_counts`

Counts PDB entries by deposition year and broad experimental-method category:
X-ray, cryo-EM, and NMR. Every counted entry must contain at least one protein
polymer entity.

The categories use exact method sets: `X-RAY DIFFRACTION` for X-ray,
`ELECTRON MICROSCOPY` for cryo-EM, and either `SOLUTION NMR`, `SOLID-STATE NMR`,
or their two-method combination for NMR. Other hybrids are excluded.

Output:

- `data/pdb_method_counts_by_year.csv`

### `membrane_protein_counts`

Counts entries with membrane-protein annotations by deposition year. It also
writes a method split for protein-containing membrane entries whose experimental
method set exactly matches one of the supported X-ray, cryo-EM, or NMR sets.
Therefore, the method-split counts need not sum to the overall membrane count.

Membrane annotations come from RCSB annotation types such as OPM, PDBTM,
MemProtMD, and mpstruc.

Outputs:

- `data/membrane_protein_counts_by_year.csv`
- `data/membrane_protein_method_counts_by_year.csv`

### `solution_nmr_program_counts`

Collects exact single-method `SOLUTION NMR` entries, downloads PDB files into
the cache, and extracts refinement program names from PDB remarks. The output
shows refinement-program usage by year.

Program names from `REMARK 3 PROGRAM` and wrapped `REMARK 210 SOFTWARE USED`
fields are normalized before counting. Each entry contributes once to each
distinct normalized name it reports.

Output:

- `data/solution_nmr_program_counts_by_year.csv`

Useful options:

- `--solution-nmr-program-cache-dir`

### `solution_nmr_monomer_program_clusters`

Assigns SOLUTION NMR protein monomers to refinement-program clusters using PDB
`REMARK 3 PROGRAM` and `REMARK 210 SOFTWARE USED`. If a structure names `n`
known clusters, each receives a score of `1/n`, so the total score remains `1`.
`OTHER` is used only when no known cluster is found.

Wrapped `REMARK 210` fields are joined before matching. Program-aware boundaries
prevent unrelated names such as `VARIAN` and `DISCOVERY STUDIO` from matching
`ARIA` or `DISCOVER`.

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
`cluster_score`, per-cluster `structure_count`, and per-cluster quality means use
these fractional weights. Overall yearly totals count unique structures.
The dataset downloads missing PDB files itself; `solution_nmr_program_counts` is
not a prerequisite.

Requires:

- `data/solution_nmr_monomer_quality_metrics.csv`

Outputs:

- `data/solution_nmr_monomer_program_cluster_assignments.csv`
- `data/solution_nmr_monomer_program_cluster_quality_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total_by_year.csv`
- `data/solution_nmr_monomer_program_cluster_quality_total.csv`

### `solution_nmr_weights`

Reads one total molecular-weight value from RCSB metadata for each entry that
passes the shared SOLUTION NMR and protein-presence filters above.

Weight-category plots use `<10 kDa`, `10–20 kDa` inclusive, and `>20 kDa`.

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
usable state map, the row is retained with state fractions of `-1.0` and
`stride_secondary_structure_percent = 200.0`.

That stored percentage is `100 * (1 - C)` and includes turns (`T`). Figure 3
does not use it: the plot computes `100 * (H + G + I + E + B)`, removes values
outside 0–100%, and takes the arithmetic mean for each year. In the article
snapshot, this excludes 97 rows and plots 10,030 through 2024.

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

The CSV field `n_ca_core_used` is the size of the collapsed-CA intersection.
`n_ca_core_raw` is the smallest raw positive-occupancy CA-record count across
models at those positions.

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

Builds a query from every modeled first-model CA position between the STRIDE
core endpoints. The query must contain at least 11 residues. RCSB Search uses a
protein-sequence identity cutoff of 0.95 or 1.0, an E-value cutoff of 0.1, and an
`X-RAY DIFFRACTION` condition. Every returned entry is then checked against its
complete experimental-method list, and only entries whose sole method is
`X-RAY DIFFRACTION` are retained. X-ray hybrids with any additional method are
excluded before coordinate evaluation.

For candidates that pass the method check, every polymer-entity chain is tested
against its first-model coordinates. A candidate remains eligible if at least one
entity chain contains at least one matching HETATM-free region.

- The 95% coordinate check runs a local gapped alignment independently in each
  HETATM-free X-ray region; a `HETATM` CA splits the sequence, so an alignment
  cannot cross it. At least `ceil(0.95 * query length)` modeled pairs and
  identities are required.
- The 100% check requires the complete query to match one consecutive window;
  the X-ray chain may have additional residues outside that window.
- An NMR core is excluded if any modeled CA position is represented by, or also
  contains, a positive-occupancy `HETATM` record.
- A matching X-ray region containing any positive-occupancy `HETATM` CA is
  excluded at every identity cutoff. If several regions match, clean regions
  remain eligible and dirty regions are ignored; a `HETATM` outside the selected
  region does not exclude it.
- Missing resolution does not exclude a candidate. Resolution is not stored in
  the homolog CSV; downstream RMSD outputs write a missing value as `nan`.

Entries without a valid NMR query are absent from the homolog datasets.
`has_xray_homolog = 0` therefore means that a valid search found no eligible
homolog.

Every hit rejected by the method or modeled-core check is written to the
cutoff-specific `*_rejected.csv` report. Each row records the NMR chain and core,
X-ray entry/entity and chains, cutoff, and reason. Rows are deduplicated by NMR
entry, cutoff, and X-ray entity. Metadata or coordinate errors remain
inconclusive and fail/retry the complete NMR entry rather than creating a normal
rejection row.

Within one NMR seed, the 95% and 100% passes share candidate metadata and the
first-model X-ray CA cache described above. This avoids duplicate downloads and
parses while preserving separate sequence searches and cutoff checks.

After ordinary request retries, a remaining HTTP 5xx requeues the entry for at
most three entry-level attempts. Other failures are not requeued. A final error
removes the entry rather than writing `has_xray_homolog = 0`.

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`
- `data/solution_nmr_monomer_xray_homologs_95_rejected.csv`
- `data/solution_nmr_monomer_xray_homologs_100_rejected.csv`

### `solution_nmr_monomer_xray_homologs_historical`

Keeps only X-ray homologs released by the NMR deposition date. Records with no
candidates remain with `has_xray_homolog = 0`. For a record with candidates, a
missing NMR deposition date or any missing X-ray initial-release date removes
the complete record. The release date is the RCSB `initial_release_date` field.

Requires:

- `data/solution_nmr_monomer_xray_homologs_95.csv`
- `data/solution_nmr_monomer_xray_homologs_100.csv`

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

### `solution_nmr_monomer_xray_rmsd`

The four RMSD dataset kinds select their cutoff with
`--xray-rmsd-sequence-identity`. For each selected variant, the value replaces
`<cutoff>` in its input path:

- current variants: `data/solution_nmr_monomer_xray_homologs_<cutoff>.csv`;
- historical variants:
  `data/solution_nmr_monomer_xray_homologs_<cutoff>_historical.csv`.

Historical-only runs do not require a current homolog CSV. The calculation still
requires an exact modeled-core match at either cutoff. Use different output paths
to retain both 95% and 100% runs.

Computes CA RMSD between an NMR STRIDE core and a matching X-ray homolog.

The exact sequence match is repeated over HETATM-free X-ray regions, so RMSD
cannot silently select a dirty repeat after the homolog stage accepted a clean
one. The RMSD uses matched `ATOM` CA pairs from the first NMR and X-ray models
and requires at least three pairs. The NMR-core positive-occupancy `HETATM` rule
above is revalidated. Invariant NMR residues and first-model coordinates are
parsed once per NMR entry and reused across its X-ray candidates. X-ray
first-model residues and coordinates come from the CA cache described above.

```text
d_eh = RMSD_superposed(NMR_e,model1, Xray_h,model1)
```

The coordinates are centered and optimally superposed before RMSD is calculated.

If one chain has several exact matching windows, the window with the lowest RMSD
is used. Within an entity, the chain with the most CA pairs is selected, using
lower RMSD to break a tie. The ordinary CSV then keeps the first usable candidate
after sorting known resolutions from lowest to highest, followed by entry and
entity ID; missing resolutions sort last. It is not replaced by the minimum-RMSD
candidate used by `*_extremes`.

When several RMSD variants are selected together, all usable candidate pairs are
computed once per NMR entry. Historical outputs filter that shared set to
historical entity IDs and recompute their own total homolog counts; historical
extremes also recomputes its successful homolog counts. When both current and
historical variants are selected, their core metadata and the historical-subset
invariant are validated before the shared run.

Output:

- `data/solution_nmr_monomer_xray_rmsd.csv`

### `solution_nmr_monomer_xray_rmsd_historical`

Uses the [shared RMSD input selection](#solution_nmr_monomer_xray_rmsd) and the
same calculation as `solution_nmr_monomer_xray_rmsd`, but keeps only homologs
released no later than the NMR deposit date.

Output:

- `data/solution_nmr_monomer_xray_rmsd_historical.csv`

### `solution_nmr_monomer_xray_rmsd_extremes`

Using the [shared RMSD input selection](#solution_nmr_monomer_xray_rmsd),
computes the minimum and maximum `d_eh` across usable X-ray candidates:

```text
d_e,min = min_h(d_eh)
d_e,max = max_h(d_eh).
```

The yearly minimum-RMSD plot reports:

```text
Y_y = median_{e: year(e)=y}(d_e,min).
```

The precision-correlation plot compares `d_e,min` with ensemble precision `P`.
An entry is written only if at least one candidate produces a usable RMSD. The
minimum and maximum are computed after the per-entity chain selection described
above.

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes.csv`

### `solution_nmr_monomer_xray_rmsd_extremes_historical`

This historical variant follows the [same input-selection
rules](#solution_nmr_monomer_xray_rmsd) and computes minimum and maximum CA RMSD
only from X-ray homologs released by the NMR deposition date.

Output:

- `data/solution_nmr_monomer_xray_rmsd_extremes_historical.csv`

## Plot Generation

After the CSV files are ready, build figures with `src/pdb_plot.py`.

By default, `src/pdb_plot.py` reads the standard CSV paths in `data/` and writes
four PNG variants of each figure to `figures/<figure_name>/`. Omit `--plots` or
pass `--plots all` to select every group; all standard input CSVs must then be
present.

The variants combine title/no title with closed/open top and right axes. PNG is
the default image format. The homolog-timing plot also writes two yearly count
CSVs.

Datasets may contain later records, but plots use only years through 2024.

Build only selected plot groups with `--plots`:

```bash
python src/pdb_plot.py \
  --plots method_counts,solution_nmr_weight_stats,solution_nmr_monomer_quality
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

See the project [license summary](../README.md#license) and
[`LICENSE`](../LICENSE) for the applicable terms.

[rcsb-apis]: https://www.rcsb.org/docs/programmatic-access/web-apis-overview
[stride-repository]: https://github.com/MDAnalysis/stride
