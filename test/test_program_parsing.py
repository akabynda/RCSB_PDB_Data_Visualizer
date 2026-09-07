"""Tests for parsing and clustering NMR refinement-program metadata."""

import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    SolutionNMRMonomerProgramClusterBuilder,
    SolutionNMRMonomerQualityRecord,
    _normalize_refinement_program_name,
    _prepend_mmcif_software_remarks,
    extract_raw_refinement_program_text_from_mmcif,
    extract_raw_refinement_program_text_from_pdb,
    extract_refinement_programs_from_pdb,
    extract_solution_nmr_program_clusters,
)


class NormalizeRefinementProgramNameTests(unittest.TestCase):
    """Verify canonical normalization of raw program labels."""

    def test_removes_versions_and_parenthetical_text(self) -> None:
        """Strip version suffixes and explanatory parenthetical text."""
        self.assertEqual(
            _normalize_refinement_program_name(" AMBER 3.0 "),
            "AMBER",
        )
        self.assertEqual(
            _normalize_refinement_program_name("CNS VERSION 1.3"),
            "CNS",
        )
        self.assertEqual(
            _normalize_refinement_program_name("X-PLOR (NIH) 2.9"),
            "X-PLOR",
        )

    def test_ignores_empty_unknown_and_numeric_values(self) -> None:
        """Reject labels that contain no meaningful program name."""
        self.assertIsNone(_normalize_refinement_program_name(""))
        self.assertIsNone(_normalize_refinement_program_name("UNKNOWN"))
        self.assertIsNone(_normalize_refinement_program_name("3.0"))


class ExtractSolutionNMRProgramClustersTests(unittest.TestCase):
    """Verify mapping of raw software text to known program clusters."""

    def test_extracts_all_unique_clusters_in_program_order(self) -> None:
        """Return unique cluster matches in their first textual order."""
        self.assertEqual(
            extract_solution_nmr_program_clusters("DIANA, FANTOM, AMBER 3.0"),
            [
                ("CLUSTER6", "DIANA_DYANA"),
                ("CLUSTER1", "AMBER"),
            ],
        )

    def test_uses_other_only_when_no_known_cluster_is_present(self) -> None:
        """Use the OTHER cluster only when no recognized program is present."""
        self.assertEqual(
            extract_solution_nmr_program_clusters("FANTOM, AMBER 3.0"),
            [("CLUSTER1", "AMBER")],
        )

    def test_deduplicates_repeated_programs_and_cluster_aliases(self) -> None:
        """Deduplicate repeated software names and aliases for one cluster."""
        self.assertEqual(
            extract_solution_nmr_program_clusters(
                "CNS 1.0, CNS MODIFIED CNS WITH CONFORMATIONAL, CNS"
            ),
            [("CLUSTER3", "CNS")],
        )

    def test_extracts_every_cluster_from_compound_strings(self) -> None:
        """Extract every recognized cluster from compound program text."""
        self.assertEqual(
            extract_solution_nmr_program_clusters(
                "DYANA AMBER, CNS ARIA, CYANA-DYANA, X-PLOR XPLOR-NIH"
            ),
            [
                ("CLUSTER6", "DIANA_DYANA"),
                ("CLUSTER1", "AMBER"),
                ("CLUSTER3", "CNS"),
                ("CLUSTER2", "ARIA"),
                ("CLUSTER4", "CYANA"),
                ("CLUSTER7", "XPLOR"),
                ("CLUSTER8", "XPLOR_NIH"),
            ],
        )

    def test_handles_xplor_nih_aliases_and_typo(self) -> None:
        """Recognize X-PLOR NIH aliases and the known NIH transposition typo."""
        for value in (
            "XPLOR_NIH",
            "X-PLOR_NIH",
            "NIH-XPLOR",
            "NIHXPLOR",
            "XPLOR-NHI",
            "X-PLOR (NIH) 2.9",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    extract_solution_nmr_program_clusters(value),
                    [("CLUSTER8", "XPLOR_NIH")],
                )

    def test_avoids_known_false_positive_substrings(self) -> None:
        """Avoid matching program names embedded in unrelated words."""
        self.assertEqual(
            extract_solution_nmr_program_clusters("VNMR VARIAN INC"),
            [("CLUSTER9", "OTHER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters("DISCOVERY STUDIO"),
            [("CLUSTER9", "OTHER")],
        )

    def test_handles_insight_ii_as_discover_but_not_dgii_description(self) -> None:
        """Map standalone Insight II while excluding descriptive DGII text."""
        self.assertEqual(
            extract_solution_nmr_program_clusters("INSIGHT II"),
            [("CLUSTER5", "DISCOVER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters("INSIGHT II II"),
            [("CLUSTER5", "DISCOVER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters("NMRPIPE, INSIGHT II, VNMR"),
            [("CLUSTER5", "DISCOVER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters("DGII MODULE OF INSIGHT II VER"),
            [("CLUSTER9", "OTHER")],
        )

    def test_returns_other_when_no_program_cluster_is_parsed(self) -> None:
        """Return OTHER when program text contains no known cluster."""
        self.assertEqual(
            extract_solution_nmr_program_clusters("UNKNOWN"),
            [("CLUSTER9", "OTHER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters(None),
            [("CLUSTER9", "OTHER")],
        )


class ExtractRefinementProgramsFromPDBTests(unittest.TestCase):
    """Verify extraction of program metadata from PDB remark records."""

    def test_extracts_raw_program_lines_in_pdb_order(self) -> None:
        """Preserve raw REMARK program lines in their original file order."""
        pdb_path = self._write_pdb(
            "\n".join(
                [
                    "HEADER    TEST",
                    "REMARK   3   PROGRAM     : DIANA, AMBER 3.0",
                    "REMARK   3   PROGRAM     : CNS VERSION 1.3",
                    "REMARK   3   OTHER FIELD : IGNORED",
                    "END",
                ]
            )
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_pdb(pdb_path),
            "DIANA, AMBER 3.0 || CNS VERSION 1.3",
        )

    def test_extracts_normalized_program_set(self) -> None:
        """Return a normalized set of programs extracted from PDB remarks."""
        pdb_path = self._write_pdb(
            "\n".join(
                [
                    "HEADER    TEST",
                    "REMARK   3   PROGRAM     : DIANA, AMBER 3.0",
                    "REMARK   3   PROGRAM     : CNS VERSION 1.3; UNKNOWN",
                    "REMARK   3   PROGRAM     : X-PLOR NIH + CYANA 2.1",
                    "END",
                ]
            )
        )

        self.assertEqual(
            extract_refinement_programs_from_pdb(pdb_path),
            {"DIANA", "AMBER", "CNS", "X-PLOR NIH", "CYANA"},
        )

    def test_extracts_remark_210_software_and_continuation_lines(self) -> None:
        """Join REMARK 210 software fields with their continuation lines."""
        pdb_path = self._write_pdb(
            "\n".join(
                [
                    "HEADER    TEST",
                    "REMARK   3   PROGRAM     : CNS 1.2",
                    "REMARK 210   SOFTWARE USED                 : AMBER, X-PLOR",
                    "REMARK 210                                   NIH, CYANA 3.0",
                    "REMARK 210   METHOD USED                   : SIMULATED ANNEALING",
                    "REMARK 210                                   SHOULD NOT BE INCLUDED",
                    "END",
                ]
            )
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_pdb(pdb_path),
            "CNS 1.2 || AMBER, X-PLOR NIH, CYANA 3.0",
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters(
                extract_raw_refinement_program_text_from_pdb(pdb_path)
            ),
            [
                ("CLUSTER3", "CNS"),
                ("CLUSTER1", "AMBER"),
                ("CLUSTER8", "XPLOR_NIH"),
                ("CLUSTER4", "CYANA"),
            ],
        )

    def _write_pdb(self, text: str) -> Path:
        """Write ``text`` to the temporary PDB fixture and return its path."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        pdb_path = Path(temp_dir.name) / "test.pdb"
        pdb_path.write_text(text + "\n", encoding="utf-8")
        return pdb_path


class ExtractRefinementProgramsFromMMCIFTests(unittest.TestCase):
    """Read deposited mmCIF software for inclusion in converted PDB records."""

    def test_reads_quoted_loop_names_and_versions_in_source_order(self) -> None:
        """Keep names and versions while accepting normal mmCIF quoting."""
        cif_path = self._write_mmcif(
            """data_TEST
loop_
_pdbx_nmr_software.name
_pdbx_nmr_software.version
_pdbx_nmr_software.classification
'X-PLOR NIH' 2.9 'structure calculation'
CYANA 3.98 'structure calculation'
Rosetta ? refinement
'X-PLOR NIH' 2.9 refinement
#
"""
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_mmcif(cif_path),
            "X-PLOR NIH 2.9 || CYANA 3.98 || Rosetta",
        )

    def test_reads_scalar_category_and_semicolon_multiline_name(self) -> None:
        """Parse semicolon text fields and collapse embedded whitespace."""
        cif_path = self._write_mmcif(
            """data_TEST
_pdbx_nmr_software.name
;
X-PLOR
 NIH
;
_pdbx_nmr_software.version '2.9'
_pdbx_nmr_software.classification 'structure calculation'
#
"""
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_mmcif(cif_path),
            "X-PLOR NIH 2.9",
        )

    def test_ignores_unknown_names_and_missing_versions(self) -> None:
        """mmCIF placeholders must not become software names or versions."""
        cif_path = self._write_mmcif(
            """data_TEST
loop_
_pdbx_nmr_software.name
_pdbx_nmr_software.version
? 3.0
. 4.0
UNKNOWN ?
NULL .
CYANA ?
ARIA .
#
"""
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_mmcif(cif_path),
            "CYANA || ARIA",
        )

    def test_accepts_software_category_without_version_or_classification(self) -> None:
        """Do not require optional category fields to recover the program."""
        cif_path = self._write_mmcif("data_TEST\n_pdbx_nmr_software.name ARIA\n#\n")

        self.assertEqual(
            extract_raw_refinement_program_text_from_mmcif(cif_path), "ARIA"
        )

    def test_includes_processing_software_like_remark_210(self) -> None:
        """Conversion retains software regardless of its NMR classification."""
        cif_path = self._write_mmcif(
            """data_TEST
loop_
_pdbx_nmr_software.name
_pdbx_nmr_software.classification
NMRPipe processing
CYANA 'structure calculation'
#
"""
        )

        self.assertEqual(
            extract_raw_refinement_program_text_from_mmcif(cif_path),
            "NMRPipe || CYANA",
        )

    def test_missing_or_unusable_mmcif_preserves_empty_result(self) -> None:
        """Absent, unrelated, unknown, and malformed metadata remain empty."""
        for cif_text in (
            None,
            "data_TEST\n_entry.id TEST\n#\n",
            "data_TEST\n_pdbx_nmr_software.name ?\n#\n",
            "data_TEST\n_pdbx_nmr_software.name .\n#\n",
            "data_TEST\n_pdbx_nmr_software.name 'unterminated\n",
        ):
            with self.subTest(cif_text=cif_text):
                cif_path = self._write_mmcif(cif_text)
                self.assertEqual(
                    extract_raw_refinement_program_text_from_mmcif(cif_path), ""
                )

    def _write_mmcif(self, cif_text: str | None) -> Path:
        """Create an mmCIF fixture, or a missing-file path when text is absent."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cif_path = Path(temp_dir.name) / "TEST.cif"
        if cif_text is not None:
            cif_path.write_text(cif_text, encoding="utf-8")
        return cif_path


class ProgramClusterScoringTests(unittest.TestCase):
    """Verify fractional weighting across software-cluster assignments."""

    def test_distributes_one_structure_equally_between_unique_clusters(self) -> None:
        """Split one structure equally among its unique cluster assignments."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cache_dir = Path(temp_dir.name)
        (cache_dir / "TEST.pdb").write_text(
            "\n".join(
                [
                    "REMARK   3   PROGRAM     : CNS",
                    "REMARK 210   SOFTWARE USED                 : ARIA, CYANA",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        quality = SolutionNMRMonomerQualityRecord(
            entry_id="TEST",
            year=2020,
            clashscore=3.0,
            ramachandran_outliers_percent=1.0,
            sidechain_outliers_percent=2.0,
        )

        assignments, summaries = SolutionNMRMonomerProgramClusterBuilder(
            quality_records=[quality], cache_dir=cache_dir, max_workers=1
        ).build()

        self.assertEqual(len(assignments), 3)
        self.assertTrue(
            all(
                abs(record.cluster_score - (1.0 / 3.0)) < 1e-12
                for record in assignments
            )
        )
        self.assertAlmostEqual(sum(record.cluster_score for record in assignments), 1.0)
        nonzero = [record for record in summaries if record.structure_count > 0]
        self.assertEqual(len(nonzero), 3)
        self.assertTrue(
            all(abs(record.structure_count - (1.0 / 3.0)) < 1e-12 for record in nonzero)
        )

    def test_supports_one_eighth_scores_for_all_known_clusters(self) -> None:
        """Assign one-eighth weight when all eight known clusters occur."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cache_dir = Path(temp_dir.name)
        (cache_dir / "ALL8.pdb").write_text(
            "REMARK   3   PROGRAM     : AMBER, ARIA, CNS, CYANA, DISCOVER, "
            "DYANA, XPLOR, XPLOR-NIH\nEND\n",
            encoding="utf-8",
        )
        quality = SolutionNMRMonomerQualityRecord(
            entry_id="ALL8",
            year=2021,
            clashscore=3.0,
            ramachandran_outliers_percent=1.0,
            sidechain_outliers_percent=2.0,
        )

        assignments, _ = SolutionNMRMonomerProgramClusterBuilder(
            quality_records=[quality], cache_dir=cache_dir, max_workers=1
        ).build()

        self.assertEqual(len(assignments), 8)
        self.assertTrue(
            all(abs(record.cluster_score - 0.125) < 1e-12 for record in assignments)
        )

    def test_converts_reported_software_into_clustered_pdb_metadata(self) -> None:
        """Reported CYANA and ARIA cases must not acquire a false OTHER label."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cache_dir = Path(temp_dir.name)
        entries = (
            ("8URV", 2023, "CYANA\nRosetta", "CYANA", "CYANA || Rosetta"),
            ("9D36", 2024, "CYANA\nYASARA", "CYANA", "CYANA || YASARA"),
            ("9YM6", 2025, "ARIA", "ARIA", "ARIA"),
            ("9YM7", 2025, "ARIA", "ARIA", "ARIA"),
        )
        quality_records = []
        for entry_id, year, software_rows, _, _ in entries:
            (cache_dir / f"{entry_id}.pdb").write_text("END\n", encoding="utf-8")
            (cache_dir / f"{entry_id}.cif").write_text(
                f"data_{entry_id}\nloop_\n_pdbx_nmr_software.name\n"
                f"{software_rows}\n#\n",
                encoding="utf-8",
            )
            _prepend_mmcif_software_remarks(
                cache_dir / f"{entry_id}.pdb", cache_dir / f"{entry_id}.cif"
            )
            (cache_dir / f"{entry_id}.cif").unlink()
            quality_records.append(
                SolutionNMRMonomerQualityRecord(
                    entry_id=entry_id,
                    year=year,
                    clashscore=3.0,
                    ramachandran_outliers_percent=1.0,
                    sidechain_outliers_percent=2.0,
                )
            )

        assignments, summaries = SolutionNMRMonomerProgramClusterBuilder(
            quality_records=quality_records, cache_dir=cache_dir, max_workers=2
        ).build()

        self.assertEqual(len(assignments), len(entries))
        assignments_by_entry = {record.entry_id: record for record in assignments}
        for entry_id, year, _, cluster_name, program_text in entries:
            with self.subTest(entry_id=entry_id):
                assignment = assignments_by_entry[entry_id]
                self.assertEqual(assignment.year, year)
                self.assertEqual(assignment.cluster_name, cluster_name)
                self.assertEqual(assignment.program_text, program_text)
                self.assertTrue(assignment.has_program_text)
                self.assertEqual(assignment.cluster_score, 1.0)
        self.assertEqual(
            sum(record.structure_count for record in summaries), float(len(entries))
        )

    def test_weights_unique_mmcif_clusters_equally(self) -> None:
        """Repeated mmCIF rows and unknown software do not dilute cluster weights."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cache_dir = Path(temp_dir.name)
        (cache_dir / "TEST.pdb").write_text("END\n", encoding="utf-8")
        (cache_dir / "TEST.cif").write_text(
            "data_TEST\nloop_\n_pdbx_nmr_software.name\n"
            "CYANA\nARIA\nCYANA\nRosetta\n#\n",
            encoding="utf-8",
        )
        _prepend_mmcif_software_remarks(cache_dir / "TEST.pdb", cache_dir / "TEST.cif")
        (cache_dir / "TEST.cif").unlink()
        quality = SolutionNMRMonomerQualityRecord(
            entry_id="TEST",
            year=2023,
            clashscore=3.0,
            ramachandran_outliers_percent=1.0,
            sidechain_outliers_percent=2.0,
        )

        assignments, summaries = SolutionNMRMonomerProgramClusterBuilder(
            quality_records=[quality], cache_dir=cache_dir, max_workers=1
        ).build()

        self.assertEqual(
            {record.cluster_name: record.cluster_score for record in assignments},
            {"CYANA": 0.5, "ARIA": 0.5},
        )
        self.assertAlmostEqual(sum(record.cluster_score for record in assignments), 1.0)
        nonzero = [record for record in summaries if record.structure_count > 0]
        self.assertEqual(len(nonzero), 2)
        self.assertTrue(all(record.structure_count == 0.5 for record in nonzero))


if __name__ == "__main__":
    unittest.main()
