"""Tests for the durable first-model X-ray alpha-carbon cache."""

import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import numpy as np

import src.pdb_dataset_builder as builder


def _ca_line(
    record: str,
    serial: int,
    resname: str,
    chain_id: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
) -> str:
    """Return one fixed-column PDB alpha-carbon record."""
    return (
        f"{record:<6}{serial:5d}  CA  {resname:>3} {chain_id}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _write_fixture(path: Path, first_x: float = 1.0) -> None:
    """Write a two-chain, two-model PDB fixture with a mixed ATOM/HETATM site."""
    path.write_text(
        "".join(
            [
                "MODEL        1\n",
                _ca_line("ATOM", 1, "ALA", "A", 1, first_x, 2.0, 3.0),
                # ATOM wins coordinate/identity selection, while the HETATM
                # marker at the same residue must still survive in the cache.
                _ca_line("HETATM", 2, "MSE", "A", 1, 8.0, 8.0, 8.0),
                _ca_line("HETATM", 3, "MSE", "A", 2, 4.0, 5.0, 6.0),
                _ca_line("ATOM", 4, "GLY", "B", 7, 7.0, 8.0, 9.0),
                "ENDMDL\n",
                "MODEL        2\n",
                _ca_line("ATOM", 5, "ALA", "A", 1, 99.0, 99.0, 99.0),
                "ENDMDL\n",
            ]
        ),
        encoding="utf-8",
    )


class XrayCaCacheTests(unittest.TestCase):
    """Verify cache fidelity, versioning, invalidation, and thread safety."""

    def setUp(self) -> None:
        """Create one isolated source PDB for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.pdb_path = Path(self.temp_dir.name) / "1ABC.pdb"
        _write_fixture(self.pdb_path)

    def test_round_trips_residues_coordinates_and_hetatm_flags(self) -> None:
        """Preserve first-model selection and HETATM evidence on a disk hit."""
        first_records, first_coords = builder.load_cached_first_model_ca_data(
            self.pdb_path, "A"
        )
        second_records, second_coords = builder.load_cached_first_model_ca_data(
            self.pdb_path, "A"
        )

        self.assertEqual(first_records, second_records)
        self.assertEqual([record.resid for record in second_records], [1, 2])
        self.assertEqual(second_records[0].identity, "A")
        self.assertTrue(second_records[0].is_standard_atom)
        self.assertTrue(second_records[0].has_hetatm_ca)
        self.assertFalse(second_records[1].is_standard_atom)
        self.assertTrue(second_records[1].has_hetatm_ca)
        np.testing.assert_allclose(second_coords[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(second_coords[2], [4.0, 5.0, 6.0])
        self.assertEqual(set(first_coords), set(second_coords))

    def test_second_chain_uses_same_whole_file_parse(self) -> None:
        """Parse all first-model chains once and serve each chain from one cache."""
        real_parser = builder._parse_first_model_ca_data_by_chain
        with patch.object(
            builder,
            "_parse_first_model_ca_data_by_chain",
            wraps=real_parser,
        ) as parse_source:
            builder.load_cached_first_model_ca_data(self.pdb_path, "A")
            records_b, coords_b = builder.load_cached_first_model_ca_data(
                self.pdb_path, "B"
            )

        self.assertEqual(parse_source.call_count, 1)
        self.assertEqual([record.resid for record in records_b], [7])
        np.testing.assert_allclose(coords_b[7], [7.0, 8.0, 9.0])
        cache_path = builder._first_model_ca_cache_path(self.pdb_path)
        self.assertTrue(cache_path.is_file())
        self.assertIn(".v", cache_path.name)

    def test_source_sha256_invalidates_cache_even_if_size_and_mtime_match(
        self,
    ) -> None:
        """Never reuse parsed coordinates for different same-stat source bytes."""
        _, initial_coords = builder.load_cached_first_model_ca_data(self.pdb_path, "A")
        source_stat = self.pdb_path.stat()
        original_size = source_stat.st_size

        _write_fixture(self.pdb_path, first_x=9.0)
        self.assertEqual(self.pdb_path.stat().st_size, original_size)
        os.utime(
            self.pdb_path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )

        real_parser = builder._parse_first_model_ca_data_by_chain
        with patch.object(
            builder,
            "_parse_first_model_ca_data_by_chain",
            wraps=real_parser,
        ) as parse_source:
            _, refreshed_coords = builder.load_cached_first_model_ca_data(
                self.pdb_path, "A"
            )

        np.testing.assert_allclose(initial_coords[1], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(refreshed_coords[1], [9.0, 2.0, 3.0])
        self.assertEqual(parse_source.call_count, 1)

    def test_corrupt_cache_is_treated_as_a_miss_and_replaced(self) -> None:
        """Recover from a truncated cache without leaking the parse error."""
        builder.load_cached_first_model_ca_data(self.pdb_path, "A")
        cache_path = builder._first_model_ca_cache_path(self.pdb_path)
        cache_path.write_bytes(b"truncated-npz")

        real_parser = builder._parse_first_model_ca_data_by_chain
        with patch.object(
            builder,
            "_parse_first_model_ca_data_by_chain",
            wraps=real_parser,
        ) as parse_source:
            records, coords = builder.load_cached_first_model_ca_data(
                self.pdb_path, "A"
            )

        self.assertEqual(parse_source.call_count, 1)
        self.assertEqual([record.resid for record in records], [1, 2])
        np.testing.assert_allclose(coords[1], [1.0, 2.0, 3.0])
        self.assertGreater(cache_path.stat().st_size, len(b"truncated-npz"))

    def test_concurrent_cold_load_parses_and_writes_once(self) -> None:
        """Serialize cache creation for one source while retaining shared results."""
        real_parser = builder._parse_first_model_ca_data_by_chain
        count_lock = Lock()
        parse_count = 0

        def counted_parser(path: Path):
            """Delay a cold parse so competing workers exercise the same lock."""
            nonlocal parse_count
            with count_lock:
                parse_count += 1
            time.sleep(0.05)
            return real_parser(path)

        with patch.object(
            builder,
            "_parse_first_model_ca_data_by_chain",
            side_effect=counted_parser,
        ):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _: builder.load_cached_first_model_ca_data(
                            self.pdb_path, "A"
                        ),
                        range(16),
                    )
                )

        self.assertEqual(parse_count, 1)
        self.assertEqual(len(results), 16)
        for records, coords in results:
            self.assertEqual([record.resid for record in records], [1, 2])
            np.testing.assert_allclose(coords[1], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
