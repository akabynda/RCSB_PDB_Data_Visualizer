"""Choose one residue conformer before any PDB coordinate consumer runs."""

from __future__ import annotations

import math
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from Bio.PDB.Polypeptide import is_aa


POLYPEPTIDE_REMARK_PREFIX = "REMARK 999 POLYPEPTIDE V1 "


@dataclass
class PDBPolymerMetadata:
    """Identify polypeptide residues without treating a ligand's CA as Cα."""

    seqres_by_chain: dict[str, set[str]] = field(default_factory=dict)
    modres_keys: set[tuple[str, int, str, str]] = field(default_factory=set)
    exact_keys: set[tuple[str, int, str, str]] = field(default_factory=set)
    exact_membership_complete: bool = False

    def includes(self, chain: str, number: int, insertion: str, name: str) -> bool:
        """Prefer exact converted-mmCIF membership, then native PDB evidence."""
        key = chain, number, insertion, name
        if self.exact_membership_complete:
            return key in self.exact_keys
        if key in self.modres_keys:
            return True
        if self.seqres_by_chain:
            # A known amino-acid-like component can be a free ligand: 2LYB's
            # 8SP is in Bio.PDB's dictionary but absent from polymer SEQRES.
            return name in self.seqres_by_chain.get(chain, set())
        return is_aa(name)


def parse_pdb_polymer_metadata(lines: Iterable[str]) -> PDBPolymerMetadata:
    """Read native SEQRES/MODRES and exact membership saved during conversion."""
    metadata = PDBPolymerMetadata()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM", "MODEL ")):
            break
        if line.startswith("SEQRES") and len(line) > 19:
            metadata.seqres_by_chain.setdefault(line[11].strip(), set()).update(
                line[19:70].split()
            )
        elif line.startswith("MODRES") and len(line) >= 27:
            try:
                key = (
                    line[16].strip(),
                    int(line[18:22]),
                    line[22].strip(),
                    line[12:15].strip(),
                )
            except ValueError:
                continue
            if key[3] and line[24:27].strip():
                metadata.modres_keys.add(key)
        elif line.startswith(POLYPEPTIDE_REMARK_PREFIX):
            value = line[len(POLYPEPTIDE_REMARK_PREFIX) :].strip()
            if value == "COMPLETE":
                metadata.exact_membership_complete = True
            else:
                residue = json.loads(value)
                if (
                    not isinstance(residue, list)
                    or len(residue) != 4
                    or not isinstance(residue[0], str)
                    or not isinstance(residue[1], int)
                    or not all(isinstance(item, str) for item in residue[2:])
                ):
                    raise ValueError("Invalid polypeptide residue metadata")
                metadata.exact_keys.add(tuple(residue))
    return metadata


def load_pdb_polymer_metadata(pdb_path: str | Path) -> PDBPolymerMetadata:
    """Load only the header metadata needed by all coordinate parsers."""
    with Path(pdb_path).open(encoding="latin-1") as handle:
        return parse_pdb_polymer_metadata(handle)


def pdb_atom_field_offset(line: str) -> int:
    """Account for legacy long component names without splitting residue IDs."""
    fields = line[17:].split(maxsplit=1)
    return max(0, len(fields[0]) - 3) if fields else 0


@dataclass(frozen=True)
class _Atom:
    index: int
    line: str
    model: int
    segment: int
    polymer_metadata: PDBPolymerMetadata

    @property
    def altloc(self) -> str:
        return self.line[16].strip()

    @property
    def occupancy(self) -> float:
        offset = pdb_atom_field_offset(self.line)
        try:
            value = float(self.line[54 + offset : 60 + offset])
        except ValueError:
            return 0.0
        return value if math.isfinite(value) else 0.0

    @property
    def is_carbon_ca(self) -> bool:
        offset = pdb_atom_field_offset(self.line)
        return (
            self.line[12:16].strip() == "CA"
            and self.line[17 : 20 + offset].strip().upper() != "CA"
            and self.line[76 + offset : 78 + offset].strip().upper() in {"", "C"}
            and (
                not self.line.startswith("HETATM")
                or (
                    isinstance(self.residue_key[1], int)
                    and self.polymer_metadata.includes(
                        *self.residue_key, self.component_name
                    )
                )
            )
        )

    @property
    def companion_key(self) -> tuple[int, int, str]:
        return (
            self.model,
            self.segment,
            self.line[6 : 27 + pdb_atom_field_offset(self.line)],
        )

    @property
    def component_name(self) -> str:
        return self.line[17 : 20 + pdb_atom_field_offset(self.line)].strip()

    @property
    def residue_key(self) -> tuple[str, int | str, str]:
        offset = pdb_atom_field_offset(self.line)
        number_text = self.line[22 + offset : 26 + offset].strip()
        try:
            number: int | str = int(number_text)
        except ValueError:
            # Unsupported author-number encodings are rejected by the PDB
            # residue parser; normalization must not guess another numbering.
            number = number_text
        return (
            self.line[21 + offset].strip(),
            number,
            self.line[26 + offset].strip(),
        )


def _altloc_priority(label: str) -> tuple[int, str]:
    """Resolve occupancy ties consistently, including numeric altLoc labels."""
    return {"": 0, "A": 1, "1": 2}.get(label, 3), label


def _select_residue_altloc(atoms: list[_Atom]) -> str:
    """Use Cα occupancy, or mean labeled-atom occupancy if Cα is shared."""
    ca_atoms = [atom for atom in atoms if atom.is_carbon_ca and atom.occupancy > 0]
    if any(atom.altloc for atom in ca_atoms):
        return min(
            ca_atoms,
            key=lambda atom: (-atom.occupancy, _altloc_priority(atom.altloc)),
        ).altloc
    occupancies: dict[str, list[float]] = defaultdict(list)
    for atom in atoms:
        if atom.altloc and atom.occupancy > 0:
            occupancies[atom.altloc].append(atom.occupancy)
    if not occupancies:
        return ""
    return min(
        occupancies,
        key=lambda label: (
            -sum(occupancies[label]) / len(occupancies[label]),
            _altloc_priority(label),
        ),
    )


def normalize_pdb_altloc_text(text: str) -> str:
    """Remove minor conformers and clear altLoc on the selected residue atoms.

    Residue groups retain the full author number and insertion-code columns,
    chain, model, and TER segment. Shared blank-altLoc atoms stay in the chosen
    conformer; atoms from another labeled conformer never fill its missing atoms.
    No coordinate, residue-number, insertion-code, or software REMARK is rewritten.
    """
    lines = text.splitlines(keepends=True)
    polymer_metadata = parse_pdb_polymer_metadata(lines)
    has_alternates = any(
        line[:6] in {"ATOM  ", "HETATM"} and len(line) >= 27 and line[16].strip()
        for line in lines
    )

    groups: dict[tuple[int, int, tuple[str, int | str, str], str], list[_Atom]] = (
        defaultdict(list)
    )
    ca_segments: dict[tuple[int, tuple[str, int | str, str]], int] = {}
    contexts: list[tuple[int, int]] = []
    model = segment = 0
    for index, line in enumerate(lines):
        record = line[:6]
        if record == "MODEL ":
            model += 1
            segment = 0
        contexts.append((model, segment))
        if record in {"ATOM  ", "HETATM"} and len(line) >= 27:
            if not has_alternates and line[12:16].strip() != "CA":
                continue
            atom = _Atom(index, line, model, segment, polymer_metadata)
            author_key = atom.residue_key
            if atom.is_carbon_ca and atom.occupancy > 0:
                residue_key = model, author_key
                previous_segment = ca_segments.setdefault(residue_key, segment)
                if previous_segment != segment:
                    raise ValueError(
                        "Repeated Cα residue identifier across TER segments: "
                        f"model {model or 1}, chain {author_key[0]!r}, "
                        f"author residue {author_key[1]}{author_key[2]}"
                    )
            if not has_alternates:
                continue
            # A ligand/water can reuse a polymer's author residue number. Those
            # separate chemical residues are not alternate conformers. Known
            # modified amino acids may still alternate with an ATOM residue.
            hetero_component = (
                atom.component_name
                if record == "HETATM"
                and not (
                    isinstance(author_key[1], int)
                    and polymer_metadata.includes(*author_key, atom.component_name)
                )
                else ""
            )
            groups[(model, segment, author_key, hetero_component)].append(atom)
        elif record == "TER   " or line.strip() == "TER":
            segment += 1

    if not has_alternates:
        return text

    removed_indices: set[int] = set()
    rewritten_indices: set[int] = set()
    atoms = [atom for group in groups.values() for atom in group]
    for group in groups.values():
        if not any(atom.altloc for atom in group):
            continue
        chosen_altloc = _select_residue_altloc(group)
        chosen_atoms: dict[str, _Atom] = {}
        for atom in group:
            if atom.altloc and (atom.altloc != chosen_altloc or atom.occupancy <= 0):
                removed_indices.add(atom.index)
                continue
            atom_name = atom.line[12:16]
            previous = chosen_atoms.get(atom_name)
            if previous is None:
                chosen_atoms[atom_name] = atom
                continue
            # A shared atom and a labeled atom must not become duplicate atoms
            # when their altLoc fields are cleared.
            winner = min(
                (previous, atom),
                key=lambda candidate: (
                    -candidate.occupancy,
                    _altloc_priority(candidate.altloc),
                    candidate.index,
                ),
            )
            loser = atom if winner is previous else previous
            chosen_atoms[atom_name] = winner
            removed_indices.add(loser.index)
        rewritten_indices.update(
            atom.index for atom in chosen_atoms.values() if atom.altloc
        )

    removed_companions = {
        atom.companion_key for atom in atoms if atom.index in removed_indices
    }
    kept_companions = {
        atom.companion_key for atom in atoms if atom.index not in removed_indices
    }
    discarded_companions = removed_companions - kept_companions
    removed_serials = {
        atom.line[6:11].strip() for atom in atoms if atom.index in removed_indices
    } - {atom.line[6:11].strip() for atom in atoms if atom.index not in removed_indices}
    output: list[str] = []
    for index, line in enumerate(lines):
        if index in removed_indices:
            continue
        record = line[:6]
        if index in rewritten_indices:
            line = line[:16] + " " + line[17:]
        elif record in {"ANISOU", "SIGATM", "SIGUIJ"} and len(line) >= 27:
            companion_key = (
                *contexts[index],
                line[6 : 27 + pdb_atom_field_offset(line)],
            )
            if companion_key in discarded_companions:
                continue
            if companion_key in kept_companions:
                line = line[:16] + " " + line[17:]
        elif record == "CONECT" and removed_serials:
            ending = (
                "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            )
            body = line.rstrip("\r\n")
            serials = [body[start : start + 5] for start in range(6, len(body), 5)]
            if serials and serials[0].strip() in removed_serials:
                continue
            if any(serial.strip() in removed_serials for serial in serials[1:]):
                serials = [
                    serial
                    for serial in serials
                    if serial.strip() not in removed_serials
                ]
                if len([serial for serial in serials if serial.strip()]) < 2:
                    continue
                line = "CONECT" + "".join(serials) + ending
        output.append(line)
    return "".join(output)


def iter_normalized_pdb_lines(pdb_path: str | Path) -> Iterator[str]:
    """Read the shared conformer selection without modifying the input file."""
    text = Path(pdb_path).read_bytes().decode("latin-1")
    yield from normalize_pdb_altloc_text(text).splitlines(keepends=True)


def ensure_normalized_pdb(pdb_path: str | Path) -> Path:
    """Atomically normalize one PDB in place, retaining its mtime if unchanged.

    Cache owners must compute file hashes after this step. The read-only iterator
    is preferable for callers that do not own the input file and its sidecars.
    """
    path = Path(pdb_path)
    original = path.read_bytes()
    normalized = normalize_pdb_altloc_text(original.decode("latin-1")).encode("latin-1")
    if normalized == original:
        return path
    mode = path.stat().st_mode & 0o777
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            prefix=f".{path.name}.",
            suffix=".altloc.tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(mode)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
