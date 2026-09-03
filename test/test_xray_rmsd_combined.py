"""Tests for projecting shared X-ray RMSD candidate calculations."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pdb_dataset_builder import (
    DatasetBuildConfig,
    RCSBClient,
    SolutionNMRMonomerXrayHomologRecord,
    SolutionNMRMonomerXrayRmsdBuilder,
    SolutionNMRMonomerXrayRmsdRecord,
    _build_xray_rmsd_extremes_record_from_candidates,
    _select_ordinary_xray_rmsd_record,
    _set_active_dataset_filtered_csvs,
    build_solution_nmr_monomer_xray_rmsd_outputs_to_csv,
    read_solution_nmr_monomer_xray_rmsd_csv,
    read_solution_nmr_monomer_xray_rmsd_extremes_csv,
    write_solution_nmr_monomer_xray_homolog_csv,
    write_solution_nmr_monomer_xray_rmsd_csv,
)


def _homolog_record(
    entity_ids: tuple[str, ...],
) -> SolutionNMRMonomerXrayHomologRecord:
    """Build one homolog view over a shared set of RMSD calculations."""
    return SolutionNMRMonomerXrayHomologRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=100,
        nmr_core_start_seq_id=10,
        nmr_core_end_seq_id=20,
        nmr_query_sequence_length=11,
        xray_homolog_entry_ids=tuple(
            dict.fromkeys(entity_id.split("_", 1)[0] for entity_id in entity_ids)
        ),
        xray_homolog_entity_ids=entity_ids,
        has_xray_homolog=bool(entity_ids),
    )


def _candidate_record(
    entity_id: str,
    *,
    rmsd: float,
    resolution: float,
    n_common_ca: int = 11,
    source_homolog_count: int = 99,
) -> SolutionNMRMonomerXrayRmsdRecord:
    """Build one successful, view-neutral candidate calculation."""
    entry_id = entity_id.split("_", 1)[0]
    return SolutionNMRMonomerXrayRmsdRecord(
        entry_id="1NMR",
        year=2001,
        sequence_identity_percent=100,
        nmr_chain_id="A",
        nmr_core_start_seq_id=10,
        nmr_core_end_seq_id=20,
        nmr_query_sequence_length=11,
        xray_homolog_entity_id=entity_id,
        xray_homolog_count=source_homolog_count,
        xray_entry_id=entry_id,
        xray_chain_id="B",
        xray_core_start_seq_id=30,
        xray_core_end_seq_id=40,
        xray_resolution_angstrom=resolution,
        n_common_ca=n_common_ca,
        rmsd_ca_angstrom=rmsd,
    )


class CombinedXrayRmsdProjectionTests(unittest.TestCase):
    """Verify ordinary and extremes outputs derived from one candidate pass."""

    def tearDown(self) -> None:
        """Prevent coordinator output routing from leaking into another test."""
        _set_active_dataset_filtered_csvs(())

    @staticmethod
    def _run_outputs_coordinator(
        *,
        root: Path,
        current_input: Path | None,
        historical_input: Path | None,
        ordinary_output: Path | None,
        historical_ordinary_output: Path | None,
        extremes_output: Path | None,
        historical_extremes_output: Path | None,
        resume: bool = False,
    ) -> None:
        """Run the lightweight unified coordinator with standard test settings."""
        config = DatasetBuildConfig()
        build_solution_nmr_monomer_xray_rmsd_outputs_to_csv(
            client=RCSBClient(config),
            config=config,
            current_homolog_input_path=current_input,
            historical_homolog_input_path=historical_input,
            ordinary_output_path=ordinary_output,
            historical_ordinary_output_path=historical_ordinary_output,
            extremes_output_path=extremes_output,
            historical_extremes_output_path=historical_extremes_output,
            cache_dir=root / "cache",
            rmsd_workers=1,
            sequence_identity_percent=100,
            resume=resume,
        )

    @staticmethod
    def _candidate_set_side_effect(
        candidate_records: tuple[SolutionNMRMonomerXrayRmsdRecord, ...],
    ):
        """Return a mocked builder method that emits one set per work record."""

        def emit(
            builder: SolutionNMRMonomerXrayRmsdBuilder,
            skip_entry_ids=None,
            on_candidate_set=None,
        ):
            del skip_entry_ids
            results = []
            for homolog in builder.homolog_records:
                results.append((homolog, candidate_records))
                if on_candidate_set is not None:
                    on_candidate_set(homolog, candidate_records)
            return results

        return emit

    def test_ordinary_preserves_resolution_sorted_candidate_order(self) -> None:
        """Choose the first usable candidate, not the candidate with minimum RMSD."""
        homolog = _homolog_record(("1AAA_1", "2BBB_1"))
        candidates = (
            _candidate_record("1AAA_1", rmsd=4.0, resolution=1.0),
            _candidate_record("2BBB_1", rmsd=0.5, resolution=2.0),
        )

        ordinary = _select_ordinary_xray_rmsd_record(
            homolog=homolog,
            candidate_records=candidates,
        )

        self.assertIsNotNone(ordinary)
        assert ordinary is not None
        self.assertEqual(ordinary.xray_homolog_entity_id, "1AAA_1")
        self.assertEqual(ordinary.rmsd_ca_angstrom, 4.0)
        self.assertEqual(ordinary.xray_homolog_count, 2)

    def test_extremes_select_minimum_and_maximum_rmsd(self) -> None:
        """Select RMSD extrema independently of candidate iteration order."""
        homolog = _homolog_record(("1AAA_1", "2BBB_1", "3CCC_1", "4DDD_1"))
        candidates = (
            _candidate_record("1AAA_1", rmsd=2.0, resolution=1.0),
            _candidate_record("2BBB_1", rmsd=4.5, resolution=1.5),
            _candidate_record("3CCC_1", rmsd=0.75, resolution=2.0),
        )

        extremes = _build_xray_rmsd_extremes_record_from_candidates(
            homolog=homolog,
            candidate_records=candidates,
        )

        self.assertIsNotNone(extremes)
        assert extremes is not None
        self.assertEqual(extremes.best_xray_homolog_entity_id, "3CCC_1")
        self.assertEqual(extremes.best_rmsd_ca_angstrom, 0.75)
        self.assertEqual(extremes.worst_xray_homolog_entity_id, "2BBB_1")
        self.assertEqual(extremes.worst_rmsd_ca_angstrom, 4.5)
        self.assertEqual(extremes.rmsd_delta_angstrom, 3.75)
        self.assertEqual(extremes.xray_homolog_count, 4)
        self.assertEqual(extremes.successful_xray_homolog_count, 3)

    def test_historical_projection_filters_candidates_and_recomputes_counts(
        self,
    ) -> None:
        """Restrict shared current calculations to the historical homolog view."""
        historical = _homolog_record(("2BBB_1", "3CCC_1", "4DDD_1"))
        shared_candidates = (
            _candidate_record("1AAA_1", rmsd=0.1, resolution=0.8),
            _candidate_record("2BBB_1", rmsd=3.0, resolution=1.0),
            _candidate_record("3CCC_1", rmsd=1.0, resolution=1.5),
        )

        ordinary = _select_ordinary_xray_rmsd_record(
            homolog=historical,
            candidate_records=shared_candidates,
        )
        extremes = _build_xray_rmsd_extremes_record_from_candidates(
            homolog=historical,
            candidate_records=shared_candidates,
        )

        self.assertIsNotNone(ordinary)
        self.assertIsNotNone(extremes)
        assert ordinary is not None
        assert extremes is not None
        self.assertEqual(ordinary.xray_homolog_entity_id, "2BBB_1")
        self.assertEqual(ordinary.xray_homolog_count, 3)
        self.assertEqual(extremes.best_xray_homolog_entity_id, "3CCC_1")
        self.assertEqual(extremes.worst_xray_homolog_entity_id, "2BBB_1")
        self.assertEqual(extremes.xray_homolog_count, 3)
        self.assertEqual(extremes.successful_xray_homolog_count, 2)

    def test_projection_returns_none_without_a_successful_view_candidate(self) -> None:
        """Omit both rows when every successful calculation belongs to another view."""
        historical = _homolog_record(("2BBB_1",))
        shared_candidates = (_candidate_record("1AAA_1", rmsd=1.0, resolution=1.0),)

        self.assertIsNone(
            _select_ordinary_xray_rmsd_record(
                homolog=historical,
                candidate_records=shared_candidates,
            )
        )
        self.assertIsNone(
            _build_xray_rmsd_extremes_record_from_candidates(
                homolog=historical,
                candidate_records=shared_candidates,
            )
        )

    def test_four_outputs_share_one_candidate_set_calculation(self) -> None:
        """Project all current and historical CSV rows from one builder pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_input = root / "current_homologs.csv"
            historical_input = root / "historical_homologs.csv"
            ordinary_output = root / "ordinary.csv"
            historical_ordinary_output = root / "ordinary_historical.csv"
            extremes_output = root / "extremes.csv"
            historical_extremes_output = root / "extremes_historical.csv"
            current = _homolog_record(("1AAA_1", "2BBB_1", "3CCC_1"))
            historical = _homolog_record(("2BBB_1", "3CCC_1"))
            write_solution_nmr_monomer_xray_homolog_csv([current], current_input)
            write_solution_nmr_monomer_xray_homolog_csv([historical], historical_input)
            candidates = (
                _candidate_record("1AAA_1", rmsd=2.0, resolution=0.8),
                _candidate_record("2BBB_1", rmsd=3.0, resolution=1.0),
                _candidate_record("3CCC_1", rmsd=0.5, resolution=1.5),
            )

            with patch.object(
                SolutionNMRMonomerXrayRmsdBuilder,
                "build_candidate_sets",
                autospec=True,
                side_effect=self._candidate_set_side_effect(candidates),
            ) as build_candidate_sets:
                self._run_outputs_coordinator(
                    root=root,
                    current_input=current_input,
                    historical_input=historical_input,
                    ordinary_output=ordinary_output,
                    historical_ordinary_output=historical_ordinary_output,
                    extremes_output=extremes_output,
                    historical_extremes_output=historical_extremes_output,
                )

            build_candidate_sets.assert_called_once()
            current_ordinary = read_solution_nmr_monomer_xray_rmsd_csv(ordinary_output)[
                0
            ]
            historical_ordinary = read_solution_nmr_monomer_xray_rmsd_csv(
                historical_ordinary_output
            )[0]
            current_extremes = read_solution_nmr_monomer_xray_rmsd_extremes_csv(
                extremes_output
            )[0]
            historical_extremes = read_solution_nmr_monomer_xray_rmsd_extremes_csv(
                historical_extremes_output
            )[0]
            self.assertEqual(current_ordinary.xray_homolog_entity_id, "1AAA_1")
            self.assertEqual(historical_ordinary.xray_homolog_entity_id, "2BBB_1")
            self.assertEqual(current_extremes.best_xray_homolog_entity_id, "3CCC_1")
            self.assertEqual(current_extremes.worst_xray_homolog_entity_id, "2BBB_1")
            self.assertEqual(current_extremes.xray_homolog_count, 3)
            self.assertEqual(historical_extremes.best_xray_homolog_entity_id, "3CCC_1")
            self.assertEqual(historical_extremes.worst_xray_homolog_entity_id, "2BBB_1")
            self.assertEqual(historical_extremes.xray_homolog_count, 2)

    def test_historical_only_does_not_require_current_homolog_path(self) -> None:
        """Build a historical output when no current homolog input is supplied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            historical_input = root / "historical_homologs.csv"
            historical_output = root / "ordinary_historical.csv"
            historical = _homolog_record(("2BBB_1",))
            write_solution_nmr_monomer_xray_homolog_csv([historical], historical_input)
            candidates = (_candidate_record("2BBB_1", rmsd=1.25, resolution=1.5),)

            with patch.object(
                SolutionNMRMonomerXrayRmsdBuilder,
                "build_candidate_sets",
                autospec=True,
                side_effect=self._candidate_set_side_effect(candidates),
            ) as build_candidate_sets:
                self._run_outputs_coordinator(
                    root=root,
                    current_input=None,
                    historical_input=historical_input,
                    ordinary_output=None,
                    historical_ordinary_output=historical_output,
                    extremes_output=None,
                    historical_extremes_output=None,
                )

            build_candidate_sets.assert_called_once()
            records = read_solution_nmr_monomer_xray_rmsd_csv(historical_output)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].xray_homolog_entity_id, "2BBB_1")

    def test_asymmetric_resume_keeps_completed_output_and_writes_missing_one(
        self,
    ) -> None:
        """Retain an ordinary row while deriving its missing extremes row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_input = root / "current_homologs.csv"
            ordinary_output = root / "ordinary.csv"
            extremes_output = root / "extremes.csv"
            current = _homolog_record(("1AAA_1", "2BBB_1"))
            write_solution_nmr_monomer_xray_homolog_csv([current], current_input)
            completed_ordinary = _candidate_record(
                "1AAA_1",
                rmsd=9.0,
                resolution=0.8,
                source_homolog_count=2,
            )
            write_solution_nmr_monomer_xray_rmsd_csv(
                [completed_ordinary], ordinary_output
            )
            candidates = (
                _candidate_record("1AAA_1", rmsd=2.0, resolution=0.8),
                _candidate_record("2BBB_1", rmsd=0.5, resolution=1.5),
            )

            with patch.object(
                SolutionNMRMonomerXrayRmsdBuilder,
                "build_candidate_sets",
                autospec=True,
                side_effect=self._candidate_set_side_effect(candidates),
            ) as build_candidate_sets:
                self._run_outputs_coordinator(
                    root=root,
                    current_input=current_input,
                    historical_input=None,
                    ordinary_output=ordinary_output,
                    historical_ordinary_output=None,
                    extremes_output=extremes_output,
                    historical_extremes_output=None,
                    resume=True,
                )

            build_candidate_sets.assert_called_once()
            ordinary_records = read_solution_nmr_monomer_xray_rmsd_csv(ordinary_output)
            extremes_records = read_solution_nmr_monomer_xray_rmsd_extremes_csv(
                extremes_output
            )
            self.assertEqual(len(ordinary_records), 1)
            self.assertEqual(ordinary_records[0].rmsd_ca_angstrom, 9.0)
            self.assertEqual(len(extremes_records), 1)
            self.assertEqual(extremes_records[0].best_xray_homolog_entity_id, "2BBB_1")

    def test_default_rebuild_replaces_existing_output(self) -> None:
        """Discard a completed row during the default fresh RMSD rebuild."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_input = root / "current_homologs.csv"
            ordinary_output = root / "ordinary.csv"
            current = _homolog_record(("1AAA_1",))
            write_solution_nmr_monomer_xray_homolog_csv([current], current_input)
            write_solution_nmr_monomer_xray_rmsd_csv(
                [
                    _candidate_record(
                        "1AAA_1",
                        rmsd=9.0,
                        resolution=0.8,
                        source_homolog_count=1,
                    )
                ],
                ordinary_output,
            )
            candidates = (
                _candidate_record(
                    "1AAA_1",
                    rmsd=2.0,
                    resolution=0.8,
                    source_homolog_count=1,
                ),
            )

            with patch.object(
                SolutionNMRMonomerXrayRmsdBuilder,
                "build_candidate_sets",
                autospec=True,
                side_effect=self._candidate_set_side_effect(candidates),
            ):
                self._run_outputs_coordinator(
                    root=root,
                    current_input=current_input,
                    historical_input=None,
                    ordinary_output=ordinary_output,
                    historical_ordinary_output=None,
                    extremes_output=None,
                    historical_extremes_output=None,
                    resume=False,
                )

            records = read_solution_nmr_monomer_xray_rmsd_csv(ordinary_output)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].rmsd_ca_angstrom, 2.0)


if __name__ == "__main__":
    unittest.main()
