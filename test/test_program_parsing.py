import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import (
    _normalize_refinement_program_name,
    extract_raw_refinement_program_text_from_pdb,
    extract_refinement_programs_from_pdb,
    extract_solution_nmr_program_clusters,
)


class NormalizeRefinementProgramNameTests(unittest.TestCase):
    def test_removes_versions_and_parenthetical_text(self) -> None:
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
        self.assertIsNone(_normalize_refinement_program_name(""))
        self.assertIsNone(_normalize_refinement_program_name("UNKNOWN"))
        self.assertIsNone(_normalize_refinement_program_name("3.0"))


class ExtractSolutionNMRProgramClustersTests(unittest.TestCase):
    def test_extracts_all_unique_clusters_in_program_order(self) -> None:
        self.assertEqual(
            extract_solution_nmr_program_clusters("DIANA, FANTOM, AMBER 3.0"),
            [
                ("CLUSTER6", "DIANA_DYANA"),
                ("CLUSTER1", "AMBER"),
            ],
        )

    def test_uses_other_only_when_no_known_cluster_is_present(self) -> None:
        self.assertEqual(
            extract_solution_nmr_program_clusters("FANTOM, AMBER 3.0"),
            [("CLUSTER1", "AMBER")],
        )

    def test_deduplicates_repeated_programs_and_cluster_aliases(self) -> None:
        self.assertEqual(
            extract_solution_nmr_program_clusters(
                "CNS 1.0, CNS MODIFIED CNS WITH CONFORMATIONAL, CNS"
            ),
            [("CLUSTER3", "CNS")],
        )

    def test_returns_other_when_no_program_cluster_is_parsed(self) -> None:
        self.assertEqual(
            extract_solution_nmr_program_clusters("UNKNOWN"),
            [("CLUSTER9", "OTHER")],
        )
        self.assertEqual(
            extract_solution_nmr_program_clusters(None),
            [("CLUSTER9", "OTHER")],
        )


class ExtractRefinementProgramsFromPDBTests(unittest.TestCase):
    def test_extracts_raw_program_lines_in_pdb_order(self) -> None:
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

    def _write_pdb(self, text: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        pdb_path = Path(temp_dir.name) / "test.pdb"
        pdb_path.write_text(text + "\n", encoding="utf-8")
        return pdb_path


if __name__ == "__main__":
    unittest.main()
