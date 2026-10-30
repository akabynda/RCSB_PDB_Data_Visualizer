# Source Pipeline Reference

Implementation and methodology for the dataset builder and plotter.
See the [main README](../README.md) for installation and figure commands,
and the [test README](../test/README.md) for checks.

## Table of Contents

- [Source Organization](#source-organization)
- [Dataset Builder](#dataset-builder)
- [Important Filtering Rules](#important-filtering-rules)
- [Modeled Part](#modeled-part)
- [Core Region](#core-region)
- [STRIDE](#stride)
- [Calculation Options](#calculation-options)
- [Dataset Reference](#dataset-reference)
  - [`method_counts`](#method_counts)
  - [`membrane_protein_counts`](#membrane_protein_counts)
  - [`solution_nmr_membrane_proteins`](#solution_nmr_membrane_proteins)
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

## Source Organization

Paths below are relative to `src/`; grouped filenames share the first path's
directory.

| Modules | Responsibility |
| --- | --- |
| `pdb_dataset_builder.py`, `pdb_plot.py` | CLI entry points and public Python exports. |
| `dataset/config.py`, `records.py`, `errors.py` | Dataset kinds, settings, records and exceptions. |
| `dataset/client/` | RCSB search, metadata, NMR and homology requests. |
| `dataset/structures.py`, `coordinates.py`, `pdb_normalization.py` | Coordinate conversion, parsing and conformer selection. |
| `dataset/matching.py`, `geometry.py` | Sequence matching, superposition and RMSD. |
| `dataset/cache.py`, `downloads.py`, `ca_cache.py`, `api_cache.py` | Coordinate/API retrieval and cache validation. |
| `dataset/stride.py`, `stride_install.py` | STRIDE installation and assignments. |
| `dataset/tmdet.py`, `tmdet_install.py`, `tmdet_annotations.py` | TmDet installation, execution and residue mapping. |
| `dataset/programs.py`, `program_statistics.py`, `history.py` | Software-name parsing, weighted summaries and homolog date filtering. |
| `dataset/builders/`, `io/`, `reporting.py` | Calculations, CSV schemas, checkpoints and exclusions. |
| `dataset/arguments.py`, `cli.py`, `workflows/` | CLI options, execution order and dataset orchestration. |
| `plotting/config.py`, `constants.py`, `tables.py` | Plot settings and table preparation. |
| `plotting/style.py`, `rendering.py` | Styling and output variants. |
| Other plot-family modules in `plotting/` | Counts, weights, quality, software, homolog, RMSD and membrane plots. |
| `plotting/plotter.py`, `cli.py`, `cli_options/` | Public plotter API and CLI dispatch. |

## Dataset Builder

`--datasets` accepts one kind, a comma-separated list, or `all`. Metadata comes
from the RCSB Search and Data APIs. Selecting several kinds runs their workflows
in dependency order; unselected prerequisites must already be available.
CSV paths in the dataset reference are defaults for `core` residue selection.

Required inputs:

| Derived dataset | Required dataset |
| --- | --- |
| `solution_nmr_monomer_program_clusters` | `solution_nmr_monomer_quality` |
| `solution_nmr_monomer_xray_homologs_historical` | Both 95% and 100% current homolog datasets |
| Current X-ray RMSD variants | Current homolog dataset at the selected cutoff |
| Historical X-ray RMSD variants | Historical homolog dataset at the selected cutoff |

Coordinate downloads try RCSB, wwPDB, PDBe and the EBI archive. A failed native
PDB download with at least one HTTP 404 permits an RCSB mmCIF conversion fallback.
X-ray calculations can instead convert only requested mmCIF chains, retaining
an author-chain mapping. Converted coordinates preserve NMR software metadata.
All coordinate consumers use the same conformer normalization described below.

Coordinate validation checks checksums and remote validators, with a default
24-hour interval. Chain subsets additionally require matching source hashes and
requested chains. First-model X-ray CA records are reused only while the source
PDB hash matches. Updates are atomic and locked per PDB entry, including between
processes on POSIX.

Each primary `name.csv` has a warning/error log `name.log` and an exclusion
table `name_filtered.csv` containing `entry_id`, deposition `year` and `reason`.
Missing years remain empty; duplicate
entry/reason pairs are suppressed. Exclusions cover entries considered after
the initial API search. Derived datasets import upstream exclusions. Homolog
outputs also record individual rejected X-ray candidates separately from
excluded NMR entries.

Default storage paths can be changed with the corresponding CLI options:

| Contents | Default path | Option |
| --- | --- | --- |
| Coordinates for monomer eligibility and membrane calculations | `data/pdb_cache/` | `--solution-nmr-monomer-cache-dir` |
| First-model STRIDE assignments | `data/stride_cache/` | `--stride-cache-dir` |
| Managed STRIDE build | `data/stride/` | `--stride-install-dir` |
| Membrane API responses | `data/rcsb_cache/solution_nmr_membrane_proteins/` | `--solution-nmr-membrane-cache-dir` |
| Managed TmDet build | `data/tmdet/` | `--tmdet-install-dir` |
| TmDet inputs and results | `data/tmdet_cache/` | `--tmdet-cache-dir` |

Precision, X-ray RMSD and software workflows have separate coordinate-cache
options listed by `--help`; each also defaults to `data/pdb_cache/`.

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
- every coordinate model has at least one usable modeled CA position in that
  chain;
- no positive-occupancy protein `HETATM` CA records remain after conformer
  selection in any model, including outside the STRIDE core;
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

For coordinate-level monomer datasets, a modeled CA position is a full author
residue identifier (integer number plus insertion code) with a positive-occupancy
carbon `CA` record in a coordinate model. Calcium ions, explicit noncarbon
atoms, and ligand atoms named `CA` are excluded. For native PDB files, a
`HETATM` residue needs polypeptide evidence from `MODRES` or its chain's `SEQRES`.
When SEQRES is present, membership in Bio.PDB's amino-acid dictionary alone is
insufficient: a known component can also be a free ligand (for example, `8SP`
in `2LYB`). The dictionary is a fallback only for files without SEQRES.
mmCIF conversions preserve exact polypeptide membership
from `_entity_poly` and `_atom_site` in `REMARK 999 POLYPEPTIDE V1` records;
this exact membership takes precedence over component-name inference.
`56`, `56A`, ..., `56E` are separate positions, independent of atom altLoc.

Alternate conformers are resolved upstream for the entire residue, separately
for each model, chain, and TER segment. When CA has alternatives, its highest
positive occupancy selects the conformer; otherwise the highest mean occupancy
of labeled atoms selects it. Ties use blank, `A`, `1`, then lexical order.
Shared blank-altLoc atoms are retained, minor conformers are removed, and the
selected atoms' altLoc fields are cleared. Coordinates and residue IDs are not
renumbered. Duplicate CA records still sharing a full residue ID prefer `ATOM`
to `HETATM`, then higher occupancy. A retained positive-occupancy `HETATM` CA
also sets the position's HETATM flag even when its `ATOM` record wins.

Author IDs are retained rather than replaced with RCSB label IDs. Negative and
zero numbers, gaps, nonmonotonic numbers, reverse insertion-code order, and
letter or numeric insertion codes remain distinct in coordinate order. CSV
endpoints use labels such as `56E`; a numeric insertion code is written `56^1`
to distinguish it from residue `561`. STRIDE receives reversible temporary
numbering when an insertion code would make its output ambiguous; returned
assignments are mapped back to the original full IDs. mmCIF-to-PDB conversion
rejects author numbers outside the PDB field range `-999` through `9999` instead
of shifting their digits into the insertion-code column. Nonnumeric mmCIF
author numbers are not supported by the current Bio.PDB conversion. Reuse of
the same full CA identifier in another TER segment of the same model is
rejected explicitly instead of merging the segments.

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

The STRIDE summary can retain an entry with no core states, but precision and
homology datasets cannot. By default, core calculations include all modeled CA
positions between the endpoints. With `--rmsd-residue-selection
secondary-structure`, precision and X-ray RMSD use only the `H/G/I/E/B` positions
within that same span; turns (`T`) and coils (`C`) are excluded. The selection
comes from the first NMR model, and its matching X-ray positions are used
without requiring a separate X-ray STRIDE assignment. Both rigid-body fitting
and RMSD use the selected atoms. Precision additionally intersects these
positions across all ensemble models. At least three selected CA atoms must
remain. X-ray sequence matching and homolog eligibility still use the full core,
and all saved candidates are evaluated before choosing the minimum RMSD.
The span follows coordinate-record order, including insertion codes, rather
than numeric or lexical sorting. The first model defines its full set of IDs;
a missing endpoint in another model cannot change that span.

## STRIDE

STRIDE is required by the modeled-state, ensemble-precision and homolog-search
builders. Executable lookup order is the explicit
`--solution-nmr-monomer-stride-executable`, `stride` in `PATH`, the managed build,
`/tmp/stride_src/src/stride`, then automatic installation. An invalid explicit
path fails without falling back.

Automatic installation on macOS/Linux uses STRIDE revision
`867a5eb0f2479cb16615512a53ee472c54649505` and requires Git, GNU Make, a C compiler
and GitHub access. Installation is serialized; each revision/platform has its
own build. `--stride-install-dir` sets the installation root.

First-model assignments are reused when the SHA-1 of the first-model coordinate
text matches. The executable version is not part of this key: changing STRIDE
requires clearing its assignment cache or selecting another location with
`--stride-cache-dir`.

## Calculation Options

Run `python src/pdb_dataset_builder.py --help` for all paths and defaults.

| Option | Effect |
| --- | --- |
| `--workers`, `--precision-workers`, `--xray-rmsd-workers` | Concurrency for collection, precision and X-ray RMSD. |
| `--batch-size`, `--page-size` | GraphQL batch and Search API page sizes. |
| `--log-level` | Logging verbosity. |
| `--pdb-cache-validation-hours` | Coordinate revalidation interval; `0` validates every access. |
| `--xray-rmsd-sequence-identity {95,100}` | Homolog input cutoff; default `100`. |
| `--rmsd-residue-selection {core,secondary-structure}` | Atom selection; default `core`. The plotter accepts the same switch. |
| `--precision-input PATH` | Reuse entries, chains and core ranges from a precision CSV and recompute RMSD. Input and output must differ. |
| `--resume` | Retain completed homolog, precision and X-ray RMSD results. |

Secondary-structure selection adds `_secondary_structure` to default
precision/RMSD CSV and figure stems; explicit paths override those defaults.
Resume requires the same inputs and calculation settings. Precision and RMSD
rows record `residue_selection`; an
absent field means `core`. Resume and the precision-correlation plot reject
incompatible selections.

Without resume, datasets are rebuilt. Homolog resume retains completed paired
95%/100% results only when their candidate-rejection reports are checkpointed,
and also retains checkpointed ineligible entries. Failed or unfinished entries
are retried. The homolog checkpoint is `<95%-output-stem>.resume.tsv` beside
the 95% output. Precision and each selected RMSD variant resume independently.

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

### `solution_nmr_membrane_proteins`

Selects PDB entries with at least one `polypeptide(L)` or `polypeptide(D)` chain,
exactly `SOLUTION NMR` as the experimental-method set, more than one deposited model,
and a protein-entity annotation of type `OPM`, `PDBTM`, `MemProtMD` or `mpstruc`.
Missing/invalid model counts are excluded. UniProt membrane annotation alone
does not qualify. There are no monomer, HETATM, length, core or year filters.
The unit is a PDB entry; repeated proteins are not deduplicated.

**TmDet calculation.** The coordinate file must also contain multiple models.
The normalized first model, with all ATOM/HETATM coordinates and no core crop
or assembly expansion, is passed to TmDet. The program applies its own atom
and chain eligibility rules. The managed build uses revision
`311af38c09b1ec8636810b6f9d7ab532adaa846e` (XML version 4.1.4), Gemmi 0.7.0,
Eigen 3.4.0, nlohmann/json 3.11.3 and pugixml 1.14. Archives are hash-checked;
CMake and a C++20 compiler are required. Build adjustments are defined in
`dataset/tmdet_install.py`. `--tmdet-executable` selects an existing binary.

Arguments are `-pi input.pdb -x result.xml -po transformed.cif -m 0 -nc`, with
an explicit environment file. Model index `0` is the extracted first model;
planar-membrane defaults apply, and `-nc` disables the sequence-only surface
cache. `--tmdet-timeout-seconds` defaults to 600. Required chemical components
and the `ZZZ` installation marker are retrieved from RCSB CCD.

**Residue counts.** XML boundaries map onto observed coordinate order using
PDB chain IDs, author numbers and insertion codes. Numbering gaps add no
residues. Each modeled polypeptide residue is counted once across all protein
chains, including partner chains. Polymer HETATM membership follows the shared
coordinate rules. All XML region codes and attributes are retained. Membrane
residues are those with the codes TmDet 4 classes as annotated membrane types
(`Region::isAnnotatedMembraneType`): `H` (TM helix), `B` (TM β-strand), `L`
(re-entrant loop), `F` (interfacial helix) and `N` (β-barrel inside element).
A chain with any unassigned residue (`partial_annotation`) is excluded whole.
Over the remaining chains, `N_membrane = N_H + N_B + N_L + N_F + N_N`,
`f_membrane = N_membrane / N_counted`, and the other-residue count is
`N_counted - N_membrane`. `tmdet_counted_residue_count` is `N_counted`;
`tmdet_excluded_chain_count` and `tmdet_excluded_residue_count` record the
exclusions. `tmdet_modeled_residue_count`, assigned/unassigned counts and
`tmdet_region_counts_json` cover all modeled residues. RCSB
`modeled_residue_count` is separate from the measured
`tmdet_modeled_residue_count`.

**Copy counts.** `max_protein_entity_chain_count` is the largest number of
chains of one protein entity; values above 1 mark identical chains. TmDet still
runs on the whole first model; only aggregation differs. For entity `e` with
chains `c`, `N_copy = Σ_e (1/n_e) Σ_c N_c`, computed separately for modeled,
membrane and other residues (`tmdet_copy_mean_*`), over counted chains only;
an entity without counted chains drops out. `tmdet_copy_min_membrane_residue_count`
and `tmdet_copy_max_membrane_residue_count` sum each entity's smallest and
largest chain membrane count. Copy counts equal the entry totals when every
entity has one counted chain. They are blank unless every protein chain other
than the excluded ones has a complete result.

**Symmetry flag.** The flag uses the one RCSB assembly that lists every
deposited protein chain and has the same protein-instance count. Its
`Global Symmetry` symbol, stoichiometry and oligomeric state are stored;
`assembly_symmetry_json` keeps all its `rcsb_struct_symmetry` records. Values are
blank without exactly one such assembly or global record. The flag is
descriptive: it does not select entries or change counts.

| Status | Meaning |
| --- | --- |
| `membrane_detected` | Every modeled protein residue has an XML region assignment. |
| `partial_annotation` | Some chains have unassigned residues. Entry: counts over the other chains, blank if none remain. Chain: membrane count of assigned residues; fraction and other-residue count are blank. |
| `no_membrane_detected` | Valid negative result; membrane count/fraction are zero, without inferred region labels. |
| `no_analyzable_protein` | TmDet reports `Noprotein`; membrane count/fraction are blank. |
| `not_in_model` | Chain absent from the first model; chain membrane count/fraction are blank. |
| `error` | Download, execution, timeout or mapping failed; reason is retained and membrane count/fraction are blank. |

The main CSV is `data/solution_nmr_membrane_proteins.csv`. Companions share its stem:

| Suffix | Unit and contents |
| --- | --- |
| `_entities.csv` | Protein entity: annotations, deposited sequence/hash, UniProt IDs. |
| `_chains.csv` | Protein chain: label/auth IDs, PDBTM availability and TmDet counts. |
| `_segments.csv` | Original RCSB PDBTM interval, raw positions and validity. |
| `_tmdet_regions.csv` | Every XML region, author boundaries and modeled residue count. |
| `_tmdet_residues.csv` | Every modeled protein residue, all region codes and assignment status. |
| `_duplicate_candidates.csv` | One-model/ensemble audit; normally empty under the multiple-model requirement. |
| `_filtered.csv` | Excluded entry ID, deposition year and reason. |
| `_manifest.json` | Source/run provenance, parameters, filter counts, status/category statistics, hashes, command and software versions. |

PDBTM segment bounds are one-based `label_seq_id` positions in the deposited
sequence. Invalid intervals are retained with reasons. PDBTM
`complete/partial/missing` describes annotation availability; it does not assign
biological completeness or replace the TmDet calculation.

**Reuse and failures.** Raw API responses are hash-checked and revalidated after
24 hours by default; `--solution-nmr-membrane-cache-validation-hours 0` forces
revalidation. TmDet reuse requires matching coordinate, first-model, binary
and CCD hashes, schema and arguments. Raw inputs, XML, available transformed
coordinates, logs and hashes are retained. Failed calculations are retried on
rerun. Tables are written atomically, with the manifest published last; a run
with calculation errors saves error rows and exits unsuccessfully.

**Plots.** Entries with known deposition years through 2024 enter the annual
entry count (`solution_nmr_membrane_proteins_by_year`) and the residue map
(`solution_nmr_membrane_tmdet_residues`). Usable results are complete positive
and valid negative results, plus partial ones with counted chains. The residue
map uses usable results with copy counts: x is the copy-mean membrane count,
y the copy-mean other count. Entries with identical chains are pink circles; all others are
blue circles; the map has no legend. Usable entries without copy counts are
omitted with a warning. This map alone is Figure 9. Both axes use one symlog
scale on a square panel without marginal histograms: linear from 0 to 10
residues, logarithmic above, with `linscale` 0.9 so that 0–10 is drawn one
decade wide. Both run from zero to the first major tick (0, 5, 10, 20, 50, 100,
200, 500, …) of at least 10 above the largest x or y value. Major and minor
grid lines are drawn. The table
`figures/solution_nmr_membrane_tmdet_residues/solution_nmr_membrane_tmdet_residues.csv`
retains excluded entries and marks selection with `included_in_residue_plot`.
It keeps entry totals, exclusion counts, copy means and ranges, chain counts,
the symmetry flag and `repeated_protein_chains`. No full-length/fragment class
is assigned. The annual count includes every entry regardless of TmDet status.

### `solution_nmr_program_counts`

Collects exact single-method `SOLUTION NMR` entries, downloads PDB files into
the cache, and extracts refinement program names from PDB remarks. The output
shows refinement-program usage by year.

Program names from `REMARK 3 PROGRAM` and wrapped `REMARK 210 SOFTWARE USED`
fields are normalized before counting. During mmCIF-to-PDB conversion,
`_pdbx_nmr_software.name` and optional `.version` are transferred to
`REMARK 210 SOFTWARE USED`. Missing mmCIF values (`?` and `.`) are ignored, and
repeated name/version pairs are deduplicated. All reported NMR software roles
are included, matching the scope of `SOFTWARE USED`. Long fields are wrapped
at word boundaries into continuation lines. Each entry contributes once to
each distinct normalized name it reports.

Output:

- `data/solution_nmr_program_counts_by_year.csv`

### `solution_nmr_monomer_program_clusters`

Assigns SOLUTION NMR protein monomers to refinement-program clusters using PDB
`REMARK 3 PROGRAM` and `REMARK 210 SOFTWARE USED`, including software carried
over during mmCIF conversion. If a structure names `n` known clusters, each
receives a score of `1/n`, so the total score remains `1`. `OTHER` is used only
when no known cluster is found.

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
- `CLUSTER7` — `XPLOR` (also `X-PLOR`), for occurrences outside a recognized
  XPLOR-NIH name; a compound software field can contain both clusters
- `CLUSTER8` — `XPLOR_NIH` (also underscore, reversed `NIH-XPLOR`, compact
  `NIHXPLOR`, parenthesized NIH, and the observed `NHI` typo)
- `CLUSTER9` — `OTHER`, used only when no known cluster matches

For example, `CNS ARIA` contributes `0.5` to both `CLUSTER3` and `CLUSTER2`.
`cluster_score`, per-cluster `structure_count`, and per-cluster quality means use
these fractional weights. Overall yearly totals count unique structures.
The dataset downloads missing PDB files itself; `solution_nmr_program_counts` is
not a prerequisite.

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

Runs STRIDE on the first model of each eligible SOLUTION NMR protein monomer
and computes state fractions over its modeled CA positions.

The output stores the modeled residue span and STRIDE fractions for `H`, `G`,
`I`, `E`, `B`, `T`, and `C`.

Every modeled CA position must have a recognized STRIDE assignment. `C` means
an explicit coil assignment; a missing or unrecognized assignment is not coil.
STRIDE's `B` and lowercase `b` both mean an isolated bridge and are counted in
the dataset's `B` class.
Incomplete or failed STRIDE results are excluded and recorded in the filtered
structure report, including when STRIDE exits successfully. No sentinel state
fractions or percentages are written as a dataset row.
The report preserves the distinction between a nonzero STRIDE exit (including
its diagnostic), a missing requested chain, and incomplete residue assignments
(including the missing author IDs). Assignments from another chain never
substitute for the requested chain. STRIDE's `-` output label is mapped to a
blank author chain only when the input actually contains a blank chain.

The stored `stride_secondary_structure_percent` is `100 * (1 - C)` and includes
turns (`T`). Figure 3 does not use it: the plot computes
`100 * (H + G + I + E + B)`, removes values outside 0–100%, and takes the
arithmetic mean for each year.

Output:

- `data/solution_nmr_monomer_stride_modeled_first_model.csv`
- cached first-model STRIDE state maps in `data/stride_cache/`

### `solution_nmr_monomer_precision_stride_modeled_first_model`

Computes NMR ensemble precision over the first-model STRIDE core. It uses
positive-occupancy `ATOM` CA residues found in every coordinate
model. At least three common residues are required.

Every NMR model is first rigidly aligned to the first NMR model. Let `N` be the
number of models, `n` the number of common CA residues, `r_ij(aligned)` the
aligned coordinate of residue `j` in model `i`, and

```text
r_mean,j = (1/N) * sum_i r_ij(aligned).
```

Ensemble precision in Å is

```text
P = sqrt[(1 / (N*n)) * sum_i sum_j ||r_ij(aligned) - r_mean,j||^2].
```

The CSV field `n_ca_core_used` is the size of the intersection of full residue
IDs, including insertion codes. `n_ca_core_raw` is the smallest retained
positive-occupancy CA-record count across models at those positions, after the
shared altLoc cleanup and before duplicate-record selection.

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

For candidates that pass the method check, polymer-entity chains are tested
against their first-model coordinates until a matching HETATM-free region is
found. One such region in any entity chain makes the candidate eligible.

- The 95% coordinate check runs a local gapped alignment independently in each
  HETATM-free X-ray region; a `HETATM` CA splits the sequence, so an alignment
  cannot cross it. At least `ceil(0.95 * query length)` modeled pairs and
  identities are required.
- The 100% check requires the complete query to match one consecutive window;
  the X-ray chain may have additional residues outside that window.
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
entry, cutoff, and X-ray entity. Missing candidate metadata, or coordinate errors
that leave a candidate without either a confirmed match or a complete rejection,
fail the complete NMR entry rather than creating a normal rejection row. A
confirmed match in another chain is sufficient even if one chain failed.

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

Outputs:

- `data/solution_nmr_monomer_xray_homologs_95_historical.csv`
- `data/solution_nmr_monomer_xray_homologs_100_historical.csv`

### `solution_nmr_monomer_xray_rmsd`

The four RMSD dataset kinds read the current or historical homolog output
selected by `--xray-rmsd-sequence-identity`. Historical-only runs require only
the historical input. The coordinate calculation requires an exact core match
at either cutoff. Default inputs are
`data/solution_nmr_monomer_xray_homologs_<cutoff>.csv` or
`data/solution_nmr_monomer_xray_homologs_<cutoff>_historical.csv`.
Use separate output paths to retain both cutoff runs.

Computes CA RMSD between an NMR STRIDE core and an exactly matching HETATM-free
X-ray region. It uses at least three matched `ATOM` CA pairs from the first NMR
and X-ray models. NMR residues and coordinates are parsed once per entry and
reused across its X-ray candidates; X-ray data comes from the CA cache.

```text
d_eh = RMSD_superposed(NMR_e,model1, Xray_h,model1)
```

The coordinates are centered and optimally superposed before RMSD is calculated
in Å.

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

The yearly minimum-RMSD median plot reports:

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

`python src/pdb_plot.py --plots all` selects every plot group; selected groups
require their input datasets. `--help` lists input/output flags and defaults.
Plots use deposition years through 2024. `--svg` adds SVG export to PNG output.

The plotter reads the standard CSVs in `data/`. An output path such as
`figures/pdb_method_trends.png` is grouped as
`figures/pdb_method_trends/pdb_method_trends.png`. Article figure names and
title/frame variants are listed in the [output guide](../README.md#outputs).

The homolog-timing plot also exports yearly counts. Membrane exports are
specified in the [dataset reference](#solution_nmr_membrane_proteins).

For the median precision plot,
`--nmr-monomer-precision-secondary-structure-input PATH` adds core/secondary-
structure comparisons while `--rmsd-residue-selection` remains `core`. The
second input is built with the builder's `secondary-structure` selection.
Comparisons join entries by `entry_id`; the yearly curve uses the original core
sample. Histograms use 0.1 Å bins below 3 Å and one `≥3 Å` category. Their shared
table keeps unrounded RMSDs in Å, deposition year, chain/core boundaries and
model/CA counts, sorted by decreasing core RMSD. Both RMSDs use the ensemble
precision formula above. The comparison outputs use the suffixes
`_with_histogram`, `_comparison_panels` and `_histogram`; the shared CSV is saved
beside the standalone histogram.
