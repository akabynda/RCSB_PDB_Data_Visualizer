"""Tests for parsing modeled alpha-carbon residues from PDB files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.pdb_dataset_builder as builder

from src.pdb_dataset_builder import (
    parse_first_model_ca_residues,
    parse_first_model_modeled_ca_auth_seq_ids,
    parse_models_ca_coords_with_stats,
)


def _ca_line(serial: int, resid: int, occupancy: float, x: float = 0.0) -> str:
    """Return a formatted ATOM alpha-carbon record for test fixtures."""
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _hetatm_ca_line(serial: int, resname: str, resid: int, occupancy: float) -> str:
    """Return a formatted HETATM alpha-carbon record for test fixtures."""
    return (
        f"HETATM{serial:5d}  CA  {resname} A{resid:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
        "           C\n"
    )


def _calcium_line(serial: int, resid: int, element: str = "CA", x: float = 99.0) -> str:
    """Return a calcium ion with distinct atom-name and element columns."""
    return (
        f"HETATM{serial:5d} CA    CA A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2}\n"
    )


class PdbModeledCaTests(unittest.TestCase):
    """Verify modeled-residue and coordinate parsing edge cases."""

    def test_calcium_does_not_inflate_modeled_length_or_stride_coil(self) -> None:
        """Use only the 75 protein positions as the STRIDE denominator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "calcium.pdb"
            pdb_path.write_text(
                "".join(_ca_line(resid, resid, 1.0) for resid in range(1, 76))
                + _calcium_line(76, 101)
                + _calcium_line(77, 102),
                encoding="utf-8",
            )
            modeled_ids = parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A")

            with (
                patch.object(builder, "download_pdb_if_needed", return_value=pdb_path),
                patch.object(builder, "load_cached_chain_id_map", return_value={}),
                patch.object(
                    builder,
                    "load_first_model_stride_state_by_chain",
                    return_value=({"A": dict.fromkeys(range(1, 62), "H")}, 1),
                ),
            ):
                coverages, model_count, succeeded = (
                    builder.compute_stride_state_coverages_for_chain_modeled_first_model(
                        session=MagicMock(),
                        config=builder.DatasetBuildConfig(),
                        cache_dir=root,
                        stride_cache_dir=root / "stride",
                        entry_id="1B1G",
                        chain_id="A",
                        modeled_sequence_length=len(modeled_ids),
                        modeled_auth_seq_ids=modeled_ids,
                        stride_executable="stride",
                    )
                )

            self.assertEqual(len(modeled_ids), 75)
            self.assertEqual((model_count, succeeded), (1, 1))
            self.assertAlmostEqual(coverages["H"], 61 / 75)
            self.assertAlmostEqual(coverages["C"], 14 / 75)

    def test_calcium_at_protein_residue_id_does_not_mark_hetatm(self) -> None:
        """Ignore calcium before residue collapsing and HETATM flag collection."""
        for element in ("CA", ""):
            for calcium_first in (True, False):
                with self.subTest(element=element, calcium_first=calcium_first):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdb_path = Path(tmpdir) / "same_resid.pdb"
                        records = [
                            _calcium_line(1, 5, element=element),
                            _ca_line(2, 5, 1.0),
                        ]
                        if not calcium_first:
                            records.reverse()
                        pdb_path.write_text("".join(records), encoding="utf-8")

                        residues = parse_first_model_ca_residues(pdb_path, "A")

                        self.assertEqual(len(residues), 1)
                        self.assertEqual(residues[0].identity, "A")
                        self.assertTrue(residues[0].is_standard_atom)
                        self.assertFalse(residues[0].has_hetatm_ca)

    def test_ensemble_ca_maps_and_counts_exclude_calcium_inside_core(self) -> None:
        """Exclude unique and overlapping calcium IDs in every model and range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "calcium_ensemble.pdb"
            pdb_path.write_text(
                "".join(
                    f"MODEL     {model:4d}\n"
                    + _calcium_line(1, 10)
                    + _ca_line(2, 10, 1.0, x=float(model))
                    + _calcium_line(3, 11)
                    + _ca_line(4, 12, 1.0, x=float(model + 2))
                    + "ENDMDL\n"
                    for model in (1, 2)
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(
                pdb_path, "A", start_seq_id=10, end_seq_id=12
            )

            self.assertEqual([set(coords) for coords in model_maps], [{10, 12}] * 2)
            self.assertEqual(raw_counts, [{10: 1, 12: 1}] * 2)
            self.assertEqual([coords[10][0] for coords in model_maps], [1.0, 2.0])

    def test_missing_element_keeps_protein_and_modified_amino_acid_ca(self) -> None:
        """Retain legacy carbon records, including long component IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "blank_elements.pdb"
            records = [
                _ca_line(1, 1, 1.0),
                _hetatm_ca_line(2, "MSE", 2, 1.0),
                _hetatm_ca_line(3, "A1BEB", 3, 1.0),
            ]
            pdb_path.write_text(
                "".join(record.removesuffix("C\n") + " \n" for record in records)
                + _calcium_line(4, 4, element=""),
                encoding="utf-8",
            )

            residues = parse_first_model_ca_residues(pdb_path, "A")
            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([record.resid for record in residues], [1, 2, 3])
            self.assertEqual(
                [record.identity for record in residues], ["A", "HET:MSE", "HET:A1BEB"]
            )
            self.assertEqual([set(coords) for coords in model_maps], [{1, 2, 3}])
            self.assertEqual(raw_counts, [{1: 1, 2: 1, 3: 1}])

    def test_explicit_noncarbon_element_is_rejected_for_protein_component(self) -> None:
        """Check chemistry for both fixed columns and long-component fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "noncarbon.pdb"
            pdb_path.write_text(
                _ca_line(1, 1, 1.0).removesuffix(" C\n")
                + "CA\n"
                + _hetatm_ca_line(2, "A1BEB", 2, 1.0).removesuffix(" C\n")
                + "CA\n"
                + _ca_line(3, 3, 1.0)
                + _hetatm_ca_line(4, "A1BEB", 4, 1.0),
                encoding="utf-8",
            )

            residues = parse_first_model_ca_residues(pdb_path, "A")
            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([record.resid for record in residues], [3, 4])
            self.assertEqual([set(coords) for coords in model_maps], [{3, 4}])
            self.assertEqual(raw_counts, [{3: 1, 4: 1}])

    def test_calcium_inside_stride_core_keeps_homology_query_eligible(self) -> None:
        """Build the protein query despite distinct and shared calcium IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "core_calcium.pdb"
            protein_ids = list(range(1, 24, 2))
            pdb_path.write_text(
                "".join(_ca_line(resid, resid, 1.0) for resid in protein_ids)
                + _calcium_line(24, 12)
                + _calcium_line(25, 11),
                encoding="utf-8",
            )
            config = builder.DatasetBuildConfig()
            homolog_builder = builder.SolutionNMRMonomerXrayHomologBuilder(
                client=builder.RCSBClient(config),
                config=config,
                stride_executable="stride",
                cache_dir=root,
                stride_cache_dir=root / "stride",
            )
            seed = builder.SolutionNMRMonomerXrayHomologSeedRecord("1ABC", 2000, "A")
            with (
                patch.object(builder, "download_pdb_if_needed", return_value=pdb_path),
                patch.object(builder, "load_cached_chain_id_map", return_value={}),
                patch.object(
                    builder,
                    "load_first_model_stride_state_by_chain",
                    return_value=({"A": {1: "H", 23: "E"}}, 1),
                ),
            ):
                sequence, start, end, residues = (
                    homolog_builder._build_stride_core_query_sequence(seed)
                )

            self.assertEqual((sequence, start, end), ("A" * 12, 1, 23))
            self.assertEqual([record.resid for record in residues], protein_ids)
            self.assertFalse(any(record.has_hetatm_ca for record in residues))

    def test_first_model_modeled_ids_skip_zero_occupancy_and_preserve_gaps(
        self,
    ) -> None:
        """Skip zero-occupancy residues without collapsing sequence gaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "modeled.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 10, 1.0),
                        _ca_line(2, 11, 0.0),
                        _ca_line(3, 13, 0.5),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, 10, 1.0, x=1.0),
                        _ca_line(5, 11, 1.0, x=1.0),
                        _ca_line(6, 13, 1.0, x=1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {10, 13},
            )

    def test_model_coordinate_maps_skip_zero_occupancy_ca_atoms(self) -> None:
        """Exclude zero-occupancy alpha carbons from per-model maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _ca_line(2, 2, 0.0),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(3, 1, 1.0),
                        _ca_line(4, 2, 0.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([set(model_map) for model_map in model_maps], [{1}, {1}])
            self.assertEqual(raw_counts, [{1: 1}, {1: 1}])

    def test_first_model_modeled_ids_ignore_seqadv_labels(self) -> None:
        """Ignore SEQADV annotations when collecting first-model residue IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "artifact.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "SEQADV 1CWW GLY A   -5  UNP  O14727              CLONING ARTIFACT\n",
                        "SEQADV 2M2E GLN A   -4  UNP  Q99543              EXPRESSION TAG\n",
                        "MODEL        1\n",
                        _ca_line(1, -5, 1.0),
                        _ca_line(2, -4, 1.0),
                        _ca_line(3, -3, 1.0),
                        _ca_line(4, 1, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {-5, -4, -3, 1},
            )

    def test_model_coordinate_maps_ignore_seqadv_labels(self) -> None:
        """Ignore SEQADV annotations when parsing model coordinates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "artifact_coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "SEQADV 1CWW GLY A   -5  UNP  O14727              CLONING ARTIFACT\n",
                        "SEQADV 2M2E GLN A   -4  UNP  Q99543              EXPRESSION TAG\n",
                        "MODEL        1\n",
                        _ca_line(1, -5, 1.0),
                        _ca_line(2, -4, 1.0),
                        _ca_line(3, 1, 1.0),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(4, -5, 1.0),
                        _ca_line(5, -4, 1.0),
                        _ca_line(6, 1, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual(
                [set(model_map) for model_map in model_maps],
                [{-5, -4, 1}, {-5, -4, 1}],
            )
            self.assertEqual(
                raw_counts,
                [{-5: 1, -4: 1, 1: 1}, {-5: 1, -4: 1, 1: 1}],
            )

    def test_nmr_modeled_ids_include_hetatm_records(self) -> None:
        """Include HETATM alpha carbons in modeled NMR residue identifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "hetatm.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "NLE", 2, 1.0),
                        _hetatm_ca_line(3, "A1BEB", 3, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                parse_first_model_modeled_ca_auth_seq_ids(pdb_path, "A"),
                {1, 2, 3},
            )

    def test_model_coordinate_maps_include_hetatm_ca_atoms(self) -> None:
        """Include HETATM alpha carbons in per-model coordinate maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "hetatm_coords.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "A1BEB", 2, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            model_maps, raw_counts = parse_models_ca_coords_with_stats(pdb_path, "A")

            self.assertEqual([set(model_map) for model_map in model_maps], [{1, 2}])
            self.assertEqual(raw_counts, [{1: 1, 2: 1}])

    def test_xray_parser_keeps_long_name_hetatm_ca_records(self) -> None:
        """Preserve HETATM alpha carbons with nonstandard long residue names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "long_hetatm.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1, 1.0),
                        _hetatm_ca_line(2, "A1BEB", 30, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            residues = parse_first_model_ca_residues(
                pdb_path=pdb_path,
                chain_id="A",
                include_hetatm=True,
            )

            self.assertEqual([record.resid for record in residues], [1, 30])
            self.assertEqual(
                [(record.identity, record.is_standard_atom) for record in residues],
                [("A", True), ("HET:A1BEB", False)],
            )

    def test_marks_hetatm_when_atom_is_selected_at_the_same_residue_id(self) -> None:
        """Retain raw HETATM presence when ATOM wins residue collapsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "atom_and_hetatm.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _hetatm_ca_line(1, "MSE", 5, 1.0),
                        _ca_line(2, 5, 1.0),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            residues = parse_first_model_ca_residues(pdb_path, "A")

            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_standard_atom)
            self.assertTrue(residues[0].has_hetatm_ca)


if __name__ == "__main__":
    unittest.main()
