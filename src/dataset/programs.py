"""Parse and normalize refinement software from PDB and mmCIF metadata."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from src.dataset.config import (
    LOGGER,
)

PROGRAM_REMARK_PATTERN = re.compile(r"^REMARK\s+3\s+PROGRAM\s*:\s*(.*)$")


NMR_SOFTWARE_REMARK_PATTERN = re.compile(
    r"^REMARK\s+210\s+SOFTWARE\s+USED\s*:\s*(.*)$",
    re.IGNORECASE,
)


NMR_REMARK_PATTERN = re.compile(r"^REMARK\s+210(?P<payload>.*)$")


NMR_REMARK_FIELD_PATTERN = re.compile(r"^\s*[A-Z][A-Z0-9 ,()/_-]*\s*:")


PROGRAM_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|;|/|\+|\|\||\bAND\b)\s*",
    re.IGNORECASE,
)


PROGRAM_TRAILING_VERSION_PATTERN = re.compile(
    r"\s+(?:V(?:ERSION)?\.?\s*)?\d[\w.\-_:]*$",
    re.IGNORECASE,
)


PROGRAM_PARENTHESIS_PATTERN = re.compile(r"\([^)]*\)")


PROGRAM_HAS_LETTER_PATTERN = re.compile(r"[A-Z]")


PROGRAM_EMPTY_VALUES: frozenset[str] = frozenset(
    {
        "",
        "NULL",
        "NONE",
        "N/A",
        "NA",
        "NOT APPLICABLE",
        "NOT PROVIDED",
        "UNKNOWN",
        "?",
    }
)


PROGRAM_CLUSTER_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("CLUSTER1", "AMBER"),
    ("CLUSTER2", "ARIA"),
    ("CLUSTER3", "CNS"),
    ("CLUSTER4", "CYANA"),
    ("CLUSTER5", "DISCOVER"),
    ("CLUSTER6", "DIANA_DYANA"),
    ("CLUSTER7", "XPLOR"),
    ("CLUSTER8", "XPLOR_NIH"),
    ("CLUSTER9", "OTHER"),
)


PROGRAM_CLUSTER_NAME_BY_ID: dict[str, str] = dict(PROGRAM_CLUSTER_DEFINITIONS)


PROGRAM_CLUSTER_MATCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CLUSTER1", re.compile(r"(?<![A-Z])AMBER")),
    # The left boundary prevents VARIAN from being classified as ARIA.
    ("CLUSTER2", re.compile(r"(?<![A-Z])ARIA")),
    ("CLUSTER3", re.compile(r"(?<![A-Z])CNS")),
    ("CLUSTER4", re.compile(r"CYANA")),
    # DISCOVERY STUDIO is a separate program, while the standalone INSIGHT II
    # rows in the audited token table refer to the DISCOVER/Insight II suite.
    (
        "CLUSTER5",
        re.compile(
            r"(?<![A-Z])DISCOVER(?![A-Z])|"
            r"(?<!MODULE OF )INSIGHT II(?: II)?(?! VER)"
        ),
    ),
    ("CLUSTER6", re.compile(r"DIANA|DYANA")),
)


XPLOR_PATTERN = re.compile(r"X[-_ ]?PLOR")


XPLOR_NIH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"X[-_ ]?PLOR[-_ ]*\(?N(?:IH|HI)\)?"),
    re.compile(r"N(?:IH|HI)[-_ ]*X[-_ ]?PLOR"),
)


def _normalize_refinement_program_name(raw_value: str) -> str | None:
    """Normalize raw refinement program text to a canonical program label."""
    token = raw_value.strip().upper()
    if not token:
        return None
    token = PROGRAM_PARENTHESIS_PATTERN.sub(" ", token)
    token = " ".join(token.split()).strip(".,:; ")
    if not token or token in PROGRAM_EMPTY_VALUES:
        return None
    if ":" in token:
        token = token.split(":", 1)[0].strip(".,:; ")
    while token:
        trimmed = PROGRAM_TRAILING_VERSION_PATTERN.sub("", token).strip(".,:; ")
        if trimmed == token:
            break
        token = trimmed
    if token.endswith(" VERSION"):
        token = token[: -len(" VERSION")].strip(".,:; ")
    token = " ".join(token.split()).strip(".,:; ")
    if not token or token in PROGRAM_EMPTY_VALUES:
        return None
    if PROGRAM_HAS_LETTER_PATTERN.search(token) is None:
        return None
    return token


def extract_raw_refinement_program_text_from_mmcif(cif_path: Path) -> str:
    """Read NMR software names and versions, retaining their deposited order."""
    try:
        metadata = MMCIF2Dict(str(cif_path))
    except (OSError, ValueError) as exc:
        LOGGER.warning(
            "Failed to read NMR software metadata from %s: %s", cif_path, exc
        )
        return ""

    names = metadata.get("_pdbx_nmr_software.name", [])
    versions = metadata.get("_pdbx_nmr_software.version", [])
    values: list[str] = []
    for index, raw_name in enumerate(names):
        name = " ".join(raw_name.split())
        if name == "." or name.upper() in PROGRAM_EMPTY_VALUES:
            continue
        version = " ".join(versions[index].split()) if index < len(versions) else ""
        if version == "." or version.upper() in PROGRAM_EMPTY_VALUES:
            version = ""
        values.append(f"{name} {version}" if version else name)

    # One program can have several roles (e.g. calculation and refinement).
    return " || ".join(dict.fromkeys(values))


def _prepend_mmcif_software_remarks(pdb_path: Path, cif_path: Path) -> None:
    """Atomically add NMR software remarks while preserving coordinate bytes."""
    if pdb_path.stat().st_size == 0:
        return
    if extract_raw_refinement_program_text_from_pdb(pdb_path):
        return
    program_text = extract_raw_refinement_program_text_from_mmcif(cif_path)
    if not program_text:
        return
    prefix = "REMARK 210   SOFTWARE USED                 : "
    remarks = textwrap.fill(
        program_text,
        width=80,
        initial_indent=prefix,
        subsequent_indent="REMARK 210".ljust(len(prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            prefix=f".{pdb_path.name}.",
            suffix=".software.tmp",
            dir=str(pdb_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write((remarks + "\n").encode("utf-8"))
            with pdb_path.open("rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(pdb_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def extract_raw_refinement_program_text_from_pdb(pdb_path: Path) -> str:
    """Read software text stored directly in REMARK 3 and REMARK 210."""
    values: list[str] = []
    nmr_software_parts: list[str] = []
    collecting_nmr_software = False

    def flush_nmr_software() -> None:
        """Append and clear the currently buffered NMR software text."""
        nonlocal nmr_software_parts
        if nmr_software_parts:
            value = " ".join(nmr_software_parts).strip()
            if value:
                values.append(value)
            nmr_software_parts = []

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line[:6].strip() in {"ATOM", "HETATM", "MODEL"}:
                break
            match = PROGRAM_REMARK_PATTERN.match(line)
            if match:
                value = match.group(1).strip()
                if value:
                    values.append(value)
                continue

            software_match = NMR_SOFTWARE_REMARK_PATTERN.match(line)
            if software_match:
                flush_nmr_software()
                collecting_nmr_software = True
                value = software_match.group(1).strip()
                if value:
                    nmr_software_parts.append(value)
                continue

            if not collecting_nmr_software:
                continue
            remark_match = NMR_REMARK_PATTERN.match(line)
            if not remark_match:
                flush_nmr_software()
                collecting_nmr_software = False
                continue
            payload = remark_match.group("payload")
            stripped_payload = payload.strip()
            if (
                not stripped_payload
                or NMR_REMARK_FIELD_PATTERN.match(payload) is not None
            ):
                flush_nmr_software()
                collecting_nmr_software = False
                continue
            nmr_software_parts.append(stripped_payload)

    flush_nmr_software()
    return " || ".join(values)


def extract_refinement_programs_from_pdb(pdb_path: Path) -> set[str]:
    """Extract canonical refinement program names from PDB remarks."""
    programs: set[str] = set()
    raw_text = extract_raw_refinement_program_text_from_pdb(pdb_path)
    for raw_token in PROGRAM_SPLIT_PATTERN.split(raw_text):
        normalized = _normalize_refinement_program_name(raw_token)
        if normalized is not None:
            programs.add(normalized)
    return programs


def _extract_program_cluster_matches(
    program_text: str,
) -> list[tuple[int, str, str]]:
    """Return every known cluster match as (position, id, name)."""
    text = program_text.upper()
    matches: list[tuple[int, str, str]] = []
    xplor_nih_spans: list[tuple[int, int]] = []

    for cluster_id, pattern in PROGRAM_CLUSTER_MATCH_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                (match.start(), cluster_id, PROGRAM_CLUSTER_NAME_BY_ID[cluster_id])
            )

    for pattern in XPLOR_NIH_PATTERNS:
        for match in pattern.finditer(text):
            xplor_nih_spans.append(match.span())
            matches.append(
                (
                    match.start(),
                    "CLUSTER8",
                    PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER8"],
                )
            )

    for match in XPLOR_PATTERN.finditer(text):
        if any(
            match.start() < nih_end and match.end() > nih_start
            for nih_start, nih_end in xplor_nih_spans
        ):
            continue
        matches.append(
            (
                match.start(),
                "CLUSTER7",
                PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER7"],
            )
        )
    return sorted(matches, key=lambda item: (item[0], item[1]))


def extract_solution_nmr_program_clusters(
    program_text: str | None,
) -> list[tuple[str, str]]:
    """Extract unique program clusters from refinement program text in text order."""
    clusters: list[tuple[str, str]] = []
    seen_cluster_ids: set[str] = set()
    text = (program_text or "").strip()
    for _, cluster_id, cluster_name in _extract_program_cluster_matches(text):
        if cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)
        clusters.append((cluster_id, cluster_name))

    if clusters:
        return clusters
    return [("CLUSTER9", PROGRAM_CLUSTER_NAME_BY_ID["CLUSTER9"])]
