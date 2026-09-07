"""Additional network-free coverage for coordinate download caching."""

from __future__ import annotations

import src.dataset.downloads as dataset_downloads

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src import pdb_dataset_builder as builder


class _Response:
    """Minimal streaming response double with observable cleanup."""

    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.close_calls = 0

    def iter_content(self, chunk_size: int):
        """Yield the configured body as a requests-compatible stream."""
        assert chunk_size == 1024 * 1024
        yield b""
        if self.body:
            midpoint = max(1, len(self.body) // 2)
            yield self.body[:midpoint]
            yield self.body[midpoint:]

    def raise_for_status(self) -> None:
        """Raise the same exception family used by requests."""
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,  # type: ignore[arg-type]
            )

    def close(self) -> None:
        """Record that the downloader released this response."""
        self.close_calls += 1


class _Session:
    """Queue deterministic responses and exceptions without network access."""

    def __init__(self, outcomes: list[_Response | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Record a request and return the next queued outcome."""
        self.calls.append({"url": url, **kwargs})
        if not self.outcomes:
            raise AssertionError(f"Unexpected HTTP request for {url}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_LockedDownload = Callable[
    [requests.Session, builder.DatasetBuildConfig, Path, str, str | None], Path
]


def _cache_path(cache_dir: Path, suffix: str) -> Path:
    return cache_dir / f"1ABC{suffix}"


def _write_valid_cache(
    path: Path,
    *,
    revision: str = "revision-1",
    source_url: str | None = None,
    validated_at: str = "2000-01-01T00:00:00Z",
    etag: str | None = None,
    last_modified: str | None = None,
) -> dict[str, Any]:
    """Write a coordinate file and matching versioned cache metadata."""
    path.write_text("cached coordinates\n", encoding="utf-8")
    stat = path.stat()
    metadata = {
        "schema_version": builder.PDB_CACHE_METADATA_SCHEMA_VERSION,
        "cache_revision": revision,
        "entry_id": "1ABC",
        "source_url": source_url,
        "etag": etag,
        "last_modified": last_modified,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "validated_at": validated_at,
    }
    builder._pdb_cache_metadata_path(path).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return metadata


def _compressed_coordinate(suffix: str, marker: str) -> bytes:
    if suffix == ".pdb":
        content = f"HEADER    {marker}\nEND\n"
    else:
        content = f"data_{marker}\n#\n_entry.id {marker}\n#\n"
    return gzip.compress(content.encode("utf-8"))


def _minimal_mmcif(chain_id: str = "LONG") -> bytes:
    """Return one parseable, uncompressed mmCIF structure."""
    return (
        "data_1ABC\n"
        "#\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_formal_charge\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
        f"ATOM 1 C CA . ALA {chain_id} 1 1 ? 0 0 0 1.00 10 ? "
        f"1 ALA {chain_id} CA 1\n"
        "#\n"
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("suffix", "download"),
    [
        (".pdb", builder._download_pdb_if_needed_locked),
        (".cif", builder._download_mmcif_if_needed_locked),
    ],
)
def test_locked_download_reuses_cache_updated_while_waiting_for_lock(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
) -> None:
    """A changed transaction revision proves another worker did the work."""
    path = _cache_path(tmp_path, suffix)
    _write_valid_cache(path, revision="installed-by-other-worker")
    session = MagicMock()

    result = download(
        session=session,
        config=builder.DatasetBuildConfig(pdb_cache_validation_hours=0),
        cache_dir=tmp_path,
        entry_id="1abc",
        observed_revision="revision-before-lock",
    )

    assert result == path
    session.get.assert_not_called()


@pytest.mark.parametrize(
    ("suffix", "download"),
    [
        (".pdb", builder._download_pdb_if_needed_locked),
        (".cif", builder._download_mmcif_if_needed_locked),
    ],
)
def test_locked_download_reuses_fresh_cache_at_same_revision(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
) -> None:
    """Fresh validation metadata avoids even a conditional request."""
    path = _cache_path(tmp_path, suffix)
    _write_valid_cache(
        path,
        revision="same-revision",
        validated_at=datetime.now(timezone.utc).isoformat(),
    )
    session = MagicMock()

    result = download(
        session=session,
        config=builder.DatasetBuildConfig(pdb_cache_validation_hours=1),
        cache_dir=tmp_path,
        entry_id="1ABC",
        observed_revision="same-revision",
    )

    assert result == path
    session.get.assert_not_called()


@pytest.mark.parametrize(
    ("suffix", "download", "sources"),
    [
        (
            ".pdb",
            builder._download_pdb_if_needed_locked,
            builder._pdb_download_sources,
        ),
        (
            ".cif",
            builder._download_mmcif_if_needed_locked,
            builder._mmcif_download_sources,
        ),
    ],
)
def test_locked_download_conditionally_revalidates_cached_source_first(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
    sources: Callable[[str], tuple[tuple[str, bool], ...]],
) -> None:
    """Send both validators only to the remembered source and accept its 304."""
    path = _cache_path(tmp_path, suffix)
    cached_source = sources("1ABC")[2][0]
    old_metadata = _write_valid_cache(
        path,
        source_url=cached_source,
        etag='"old-etag"',
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
    )
    response = _Response(304)
    session = _Session([response])

    result = download(
        session=session,  # type: ignore[arg-type]
        config=builder.DatasetBuildConfig(pdb_cache_validation_hours=0),
        cache_dir=tmp_path,
        entry_id="1abc",
        observed_revision="revision-1",
    )

    assert result == path
    assert session.calls == [
        {
            "url": cached_source,
            "headers": {
                "If-None-Match": '"old-etag"',
                "If-Modified-Since": "Wed, 01 Jan 2025 00:00:00 GMT",
            },
            "stream": True,
            "timeout": 60,
        }
    ]
    refreshed = json.loads(
        builder._pdb_cache_metadata_path(path).read_text(encoding="utf-8")
    )
    assert refreshed["cache_revision"] != old_metadata["cache_revision"]
    assert refreshed["validated_at"] != old_metadata["validated_at"]
    assert refreshed["sha256"] == old_metadata["sha256"]
    assert response.close_calls == 1


@pytest.mark.parametrize(
    ("suffix", "download", "sources"),
    [
        (
            ".pdb",
            builder._download_pdb_if_needed_locked,
            builder._pdb_download_sources,
        ),
        (
            ".cif",
            builder._download_mmcif_if_needed_locked,
            builder._mmcif_download_sources,
        ),
    ],
)
def test_locked_download_accepts_304_without_optional_validators(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
    sources: Callable[[str], tuple[tuple[str, bool], ...]],
) -> None:
    """A remembered source can return 304 even without ETag or Last-Modified."""
    path = _cache_path(tmp_path, suffix)
    cached_source = sources("1ABC")[0][0]
    _write_valid_cache(path, source_url=cached_source)
    response = _Response(304)
    session = _Session([response])

    assert (
        download(
            session=session,  # type: ignore[arg-type]
            config=builder.DatasetBuildConfig(pdb_cache_validation_hours=0),
            cache_dir=tmp_path,
            entry_id="1ABC",
            observed_revision="revision-1",
        )
        == path
    )

    assert session.calls[0]["headers"] == {}
    assert response.close_calls == 1


@pytest.mark.parametrize(
    ("suffix", "download", "sources"),
    [
        (
            ".pdb",
            builder._download_pdb_if_needed_locked,
            builder._pdb_download_sources,
        ),
        (
            ".cif",
            builder._download_mmcif_if_needed_locked,
            builder._mmcif_download_sources,
        ),
    ],
)
def test_locked_download_does_not_prioritize_unknown_cached_source(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
    sources: Callable[[str], tuple[tuple[str, bool], ...]],
) -> None:
    """Ignore a retired source URL and avoid leaking its validators elsewhere."""
    path = _cache_path(tmp_path, suffix)
    _write_valid_cache(
        path,
        source_url="https://retired.invalid/1ABC",
        etag='"private-etag"',
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
    )
    response = _Response(200, _compressed_coordinate(suffix, "CURRENT"))
    session = _Session([response])

    assert (
        download(
            session=session,  # type: ignore[arg-type]
            config=builder.DatasetBuildConfig(
                retries=1,
                pdb_cache_validation_hours=0,
            ),
            cache_dir=tmp_path,
            entry_id="1ABC",
            observed_revision="revision-1",
        )
        == path
    )

    assert session.calls[0]["url"] == sources("1ABC")[0][0]
    assert session.calls[0]["headers"] == {}
    assert response.close_calls == 1


@pytest.mark.parametrize(
    ("suffix", "download", "sources"),
    [
        (
            ".pdb",
            builder._download_pdb_if_needed_locked,
            builder._pdb_download_sources,
        ),
        (
            ".cif",
            builder._download_mmcif_if_needed_locked,
            builder._mmcif_download_sources,
        ),
    ],
)
def test_locked_download_does_not_trust_validators_from_mismatched_file(
    tmp_path: Path,
    suffix: str,
    download: _LockedDownload,
    sources: Callable[[str], tuple[tuple[str, bool], ...]],
) -> None:
    """Do not send conditional headers when the file no longer matches metadata."""
    path = _cache_path(tmp_path, suffix)
    _write_valid_cache(
        path,
        source_url=sources("1ABC")[2][0],
        etag='"stale-etag"',
        last_modified="Wed, 01 Jan 2025 00:00:00 GMT",
    )
    path.write_text("locally modified\n", encoding="utf-8")
    response = _Response(
        200,
        _compressed_coordinate(suffix, "REPLACEMENT"),
        {"ETag": '"new-etag"'},
    )
    session = _Session([response])

    result = download(
        session=session,  # type: ignore[arg-type]
        config=builder.DatasetBuildConfig(retries=1),
        cache_dir=tmp_path,
        entry_id="1abc",
        observed_revision="revision-1",
    )

    assert result == path
    assert session.calls[0]["url"] == sources("1ABC")[0][0]
    assert session.calls[0]["headers"] == {}
    assert "REPLACEMENT" in path.read_text(encoding="utf-8")
    assert response.close_calls == 1


def test_mmcif_404_routes_are_skipped_on_retry_before_clear_error(
    tmp_path: Path,
) -> None:
    """Permanent 404 routes are not queried again on later retry rounds."""
    responses = [_Response(404) for _ in builder._mmcif_download_sources("1ABC")]
    session = _Session(responses)

    with patch.object(builder.time, "sleep") as sleep:
        with pytest.raises(RuntimeError, match=r"Failed to download 1ABC mmCIF: 404"):
            builder._download_mmcif_if_needed_locked(
                session=session,  # type: ignore[arg-type]
                config=builder.DatasetBuildConfig(
                    retries=2,
                    backoff_seconds=0.25,
                ),
                cache_dir=tmp_path,
                entry_id="1abc",
                observed_revision=None,
            )

    assert [call["url"] for call in session.calls] == [
        source[0] for source in builder._mmcif_download_sources("1ABC")
    ]
    assert all(response.close_calls == 1 for response in responses)
    sleep.assert_called_once_with(0.25)


@pytest.mark.parametrize(
    ("download", "sources", "message"),
    [
        (
            builder._download_pdb_if_needed_locked,
            builder._pdb_download_sources,
            "Failed to download 1ABC: HTTP 503",
        ),
        (
            builder._download_mmcif_if_needed_locked,
            builder._mmcif_download_sources,
            "Failed to download 1ABC mmCIF: HTTP 503",
        ),
    ],
)
def test_locked_download_reports_http_errors_and_closes_every_response(
    tmp_path: Path,
    download: _LockedDownload,
    sources: Callable[[str], tuple[tuple[str, bool], ...]],
    message: str,
) -> None:
    """Exhaust independent routes, release their responses, and report the cause."""
    responses = [_Response(503) for _ in sources("1ABC")]
    session = _Session(responses)

    with pytest.raises(RuntimeError, match=message):
        download(
            session=session,  # type: ignore[arg-type]
            config=builder.DatasetBuildConfig(retries=1),
            cache_dir=tmp_path,
            entry_id="1abc",
            observed_revision=None,
        )

    assert len(session.calls) == len(responses)
    assert all(response.close_calls == 1 for response in responses)


def test_pdb_404_routes_fall_back_to_mmcif_and_convert_real_structure(
    tmp_path: Path,
) -> None:
    """Convert an mmCIF fallback and publish its chain map and metadata."""
    not_found = [_Response(404) for _ in builder._pdb_download_sources("1ABC")]
    cif_response = _Response(
        200,
        _minimal_mmcif(),
        {
            "ETag": '"cif-etag"',
            "Last-Modified": "Thu, 02 Jan 2025 00:00:00 GMT",
        },
    )
    session = _Session([*not_found, cif_response])

    path = builder._download_pdb_if_needed_locked(
        session=session,  # type: ignore[arg-type]
        config=builder.DatasetBuildConfig(retries=1),
        cache_dir=tmp_path,
        entry_id="1abc",
        observed_revision=None,
    )

    assert path == tmp_path / "1ABC.pdb"
    assert " ALA L   1" in path.read_text(encoding="utf-8")
    assert (tmp_path / "1ABC.cif").read_bytes() == _minimal_mmcif()
    assert (tmp_path / "1ABC.chain_map.csv").read_text(encoding="utf-8") == (
        "original_chain_id,mapped_chain_id\nLONG,L\n"
    )
    metadata = json.loads(
        builder._pdb_cache_metadata_path(path).read_text(encoding="utf-8")
    )
    assert metadata["source_url"] == "https://files.rcsb.org/download/1ABC.cif"
    assert metadata["chain_id_map"] == {"LONG": "L"}
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [response.close_calls for response in [*not_found, cif_response]] == [
        1,
        1,
        1,
        1,
        1,
    ]


def test_converted_mmcif_keeps_software_available_to_cached_cluster_builder(
    tmp_path: Path,
) -> None:
    """PDBIO conversion must not turn deposited NMR software into OTHER."""
    not_found = [_Response(404) for _ in builder._pdb_download_sources("1ABC")]
    cif_bytes = _minimal_mmcif() + (
        "loop_\n"
        "_pdbx_nmr_software.name\n"
        "_pdbx_nmr_software.version\n"
        "_pdbx_nmr_software.classification\n"
        "CYANA 3.98 'structure calculation'\n"
        "Rosetta ? refinement\n"
        "#\n"
    ).encode("utf-8")
    session = _Session([*not_found, _Response(200, cif_bytes)])
    config = builder.DatasetBuildConfig(retries=1)
    path = builder.download_pdb_if_needed(
        session=session,  # type: ignore[arg-type]
        config=config,
        cache_dir=tmp_path,
        entry_id="1ABC",
    )

    pdb_text = path.read_text(encoding="utf-8")
    assert "REMARK 210" in pdb_text
    assert "SOFTWARE USED" in pdb_text
    assert pdb_text.index("SOFTWARE USED") < pdb_text.index("ATOM")
    # The converted PDB is a standalone artifact, including when its source
    # mmCIF is no longer available to the program extractor's fallback.
    (tmp_path / "1ABC.cif").unlink()
    assert builder.extract_raw_refinement_program_text_from_pdb(path) == (
        "CYANA 3.98 || Rosetta"
    )
    assert builder.extract_refinement_programs_from_pdb(path) == {"CYANA", "ROSETTA"}
    quality = builder.SolutionNMRMonomerQualityRecord(
        entry_id="1ABC",
        year=2023,
        clashscore=3.0,
        ramachandran_outliers_percent=1.0,
        sidechain_outliers_percent=2.0,
    )
    client = MagicMock(session=session)
    request_count = len(session.calls)

    assignments, _ = builder.SolutionNMRMonomerProgramClusterBuilder(
        quality_records=[quality],
        cache_dir=tmp_path,
        max_workers=1,
        client=client,
        config=config,
    ).build()

    assert len(session.calls) == request_count
    assert len(assignments) == 1
    assert assignments[0].cluster_name == "CYANA"
    assert assignments[0].cluster_score == 1.0
    assert assignments[0].has_program_text
    assert assignments[0].program_text == "CYANA 3.98 || Rosetta"


@pytest.mark.parametrize("conversion", ["full", "subset"])
def test_mmcif_conversion_embeds_wrapped_software_without_changing_coordinates(
    tmp_path: Path,
    conversion: str,
) -> None:
    """Both conversion routes retain software in a portable multi-model PDB."""
    cif_bytes = _minimal_mmcif().replace(
        b"ATOM 1 C CA . ALA LONG 1 1 ? 0 0 0 1.00 10 ? 1 ALA LONG CA 1\n",
        (
            "ATOM 1 C CA . ALA LONG 1 1 ? 1 2 3 1.00 10 ? 1 ALA LONG CA 1\n"
            "ATOM 2 C CA . GLY OTHER 2 1 ? 4 5 6 1.00 10 ? 7 GLY OTHER CA 1\n"
            "ATOM 3 C CA . ALA LONG 1 1 ? 7 8 9 1.00 10 ? 1 ALA LONG CA 2\n"
            "ATOM 4 C CA . GLY OTHER 2 1 ? 10 11 12 1.00 10 ? 7 GLY OTHER CA 2\n"
        ).encode("utf-8"),
    ) + (
        "loop_\n"
        "_pdbx_nmr_software.name\n"
        "_pdbx_nmr_software.version\n"
        "_pdbx_nmr_software.classification\n"
        "CYANA 3.98 'structure calculation'\n"
        "'X-PLOR NIH' 2.9 refinement\n"
        "ARIA 2.3 'structure calculation'\n"
        "NMRFAM-SPARKY 1.414 processing\n"
        "Rosetta 2026.17 refinement\n"
        "#\n"
    ).encode("utf-8")
    expected_program_text = (
        "CYANA 3.98 || X-PLOR NIH 2.9 || ARIA 2.3 || "
        "NMRFAM-SPARKY 1.414 || Rosetta 2026.17"
    )
    config = builder.DatasetBuildConfig(retries=1)
    if conversion == "full":
        session = _Session(
            [
                *[_Response(404) for _ in builder._pdb_download_sources("1ABC")],
                _Response(200, cif_bytes),
            ]
        )
        path = builder.download_pdb_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=tmp_path,
            entry_id="1ABC",
        )
        expected_chain_map = {"LONG": "L", "OTHER": "O"}
    else:
        session = _Session([_Response(200, gzip.compress(cif_bytes))])
        path, chain_map = builder.download_pdb_chain_subset_if_needed(
            session=session,  # type: ignore[arg-type]
            config=config,
            cache_dir=tmp_path,
            entry_id="1ABC",
            chain_ids=["LONG"],
        )
        expected_chain_map = {"LONG": "L"}
        assert chain_map == expected_chain_map

    pdb_text = path.read_text(encoding="utf-8")
    remarks = [line for line in pdb_text.splitlines() if line.startswith("REMARK")]
    assert remarks
    assert all(line.startswith("REMARK 210") and len(line) <= 80 for line in remarks)
    assert any("SOFTWARE USED" in line for line in remarks)
    assert any("SOFTWARE USED" not in line for line in remarks)
    assert pdb_text.rindex("REMARK") < pdb_text.index("MODEL")

    metadata = json.loads(
        builder._pdb_cache_metadata_path(path).read_text(encoding="utf-8")
    )
    assert metadata["chain_id_map"] == expected_chain_map
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["size_bytes"] == path.stat().st_size
    assert metadata["mtime_ns"] == path.stat().st_mtime_ns

    # Moving just the output rules out all sibling-mmCIF recovery paths.
    portable_dir = tmp_path / "portable"
    portable_dir.mkdir()
    portable_pdb_path = portable_dir / "renamed.pdb"
    portable_pdb_path.write_bytes(path.read_bytes())
    (tmp_path / "1ABC.cif").unlink()
    assert builder.extract_raw_refinement_program_text_from_pdb(portable_pdb_path) == (
        expected_program_text
    )
    assert builder.extract_refinement_programs_from_pdb(portable_pdb_path) == {
        "CYANA",
        "X-PLOR NIH",
        "ARIA",
        "NMRFAM-SPARKY",
        "ROSETTA",
    }
    assert builder.extract_solution_nmr_program_clusters(expected_program_text) == [
        ("CLUSTER4", "CYANA"),
        ("CLUSTER8", "XPLOR_NIH"),
        ("CLUSTER2", "ARIA"),
    ]
    structure = builder.parse_pdb_structure("1ABC", portable_pdb_path)
    atoms = [
        (
            model.serial_num,
            chain.id,
            residue.id[1],
            atom.name,
            atom.element,
            tuple(atom.coord),
        )
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    expected_atoms = [
        (1, "L", 1, "CA", "C", (1.0, 2.0, 3.0)),
        (2, "L", 1, "CA", "C", (7.0, 8.0, 9.0)),
    ]
    if conversion == "full":
        expected_atoms = [
            expected_atoms[0],
            (1, "O", 7, "CA", "C", (4.0, 5.0, 6.0)),
            expected_atoms[1],
            (2, "O", 7, "CA", "C", (10.0, 11.0, 12.0)),
        ]
    assert atoms == expected_atoms


def test_pdb_fallback_without_chains_removes_obsolete_chain_map(
    tmp_path: Path,
) -> None:
    """An empty fallback mapping must remove a map left by older content."""
    map_path = tmp_path / "1ABC.chain_map.csv"
    map_path.write_text(
        "original_chain_id,mapped_chain_id\nold,A\n",
        encoding="utf-8",
    )
    responses = [
        *[_Response(404) for _ in builder._pdb_download_sources("1ABC")],
        _Response(200, b"data_1ABC\n#\n"),
    ]
    session = _Session(responses)

    class _PDBIO:
        def set_structure(self, structure: object) -> None:
            assert structure is sentinel

        def save(self, path: str) -> None:
            Path(path).write_text("END\n", encoding="utf-8")

    sentinel = object()
    with (
        patch.object(dataset_downloads, "parse_mmcif_structure", return_value=sentinel),
        patch.object(
            dataset_downloads, "_coerce_structure_chain_ids_for_pdbio", return_value={}
        ),
        patch.object(dataset_downloads, "PDBIO", _PDBIO),
    ):
        path = builder._download_pdb_if_needed_locked(
            session=session,  # type: ignore[arg-type]
            config=builder.DatasetBuildConfig(retries=1),
            cache_dir=tmp_path,
            entry_id="1ABC",
            observed_revision=None,
        )

    assert path.read_text(encoding="utf-8") == "END\n"
    assert not map_path.exists()
    metadata = json.loads(
        builder._pdb_cache_metadata_path(path).read_text(encoding="utf-8")
    )
    assert metadata["chain_id_map"] == {}


def test_pdb_fallback_rejects_an_empty_converted_file(tmp_path: Path) -> None:
    """Do not publish metadata when PDBIO produces an empty fallback output."""
    responses = [
        *[_Response(404) for _ in builder._pdb_download_sources("1ABC")],
        _Response(200, b"data_1ABC\n#\n"),
    ]
    session = _Session(responses)

    class _EmptyPDBIO:
        def set_structure(self, structure: object) -> None:
            assert structure is sentinel

        def save(self, path: str) -> None:
            assert Path(path).exists()

    sentinel = object()
    with (
        patch.object(dataset_downloads, "parse_mmcif_structure", return_value=sentinel),
        patch.object(
            dataset_downloads, "_coerce_structure_chain_ids_for_pdbio", return_value={}
        ),
        patch.object(dataset_downloads, "PDBIO", _EmptyPDBIO),
    ):
        with pytest.raises(RuntimeError, match="Failed to download 1ABC"):
            builder._download_pdb_if_needed_locked(
                session=session,  # type: ignore[arg-type]
                config=builder.DatasetBuildConfig(retries=1),
                cache_dir=tmp_path,
                entry_id="1ABC",
                observed_revision=None,
            )

    assert not builder._pdb_cache_metadata_path(tmp_path / "1ABC.pdb").exists()
    assert responses[-1].close_calls == 1


def test_pdb_fallback_retries_parser_errors_without_damaging_cached_pdb(
    tmp_path: Path,
) -> None:
    """A failed conversion retries, closes responses, and preserves old PDB bytes."""
    pdb_path = tmp_path / "1ABC.pdb"
    _write_valid_cache(
        pdb_path,
        source_url=builder._pdb_download_sources("1ABC")[0][0],
        etag='"old"',
    )
    old_bytes = pdb_path.read_bytes()
    pdb_responses = [_Response(404) for _ in builder._pdb_download_sources("1ABC")]
    cif_responses = [_Response(200, _minimal_mmcif()) for _ in range(2)]
    session = _Session([*pdb_responses, *cif_responses])

    with (
        patch.object(
            dataset_downloads,
            "parse_mmcif_structure",
            side_effect=[ValueError("bad cif one"), ValueError("bad cif two")],
        ),
        patch.object(builder.time, "sleep") as sleep,
    ):
        with pytest.raises(RuntimeError, match="bad cif two"):
            builder._download_pdb_if_needed_locked(
                session=session,  # type: ignore[arg-type]
                config=builder.DatasetBuildConfig(
                    retries=2,
                    backoff_seconds=0.5,
                    pdb_cache_validation_hours=0,
                ),
                cache_dir=tmp_path,
                entry_id="1ABC",
                observed_revision="revision-1",
            )

    assert pdb_path.read_bytes() == old_bytes
    assert len(session.calls) == 6
    assert all(response.close_calls == 1 for response in pdb_responses)
    assert all(response.close_calls == 1 for response in cif_responses)
    assert sleep.call_args_list == [call(0.5), call(0.5)]
