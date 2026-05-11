import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import (
    _normalize_refinement_program_name,
    classify_solution_nmr_program_cluster,
    extract_raw_refinement_program_text_from_pdb,
    extract_refinement_programs_from_pdb,
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


class ClassifySolutionNMRProgramClusterTests(unittest.TestCase):
    def test_uses_first_recognized_program_not_cluster_priority(self) -> None:
        self.assertEqual(
            classify_solution_nmr_program_cluster("DIANA, AMBER 3.0"),
            ("CLUSTER6", "DIANA_DYANA"),
        )
        self.assertEqual(
            classify_solution_nmr_program_cluster("AMBER 3.0, DIANA"),
            ("CLUSTER1", "AMBER"),
        )

    def test_skips_unknown_tokens_before_first_recognized_program(self) -> None:
        self.assertEqual(
            classify_solution_nmr_program_cluster("FANTOM, AMBER 3.0"),
            ("CLUSTER1", "AMBER"),
        )

    def test_supports_all_named_program_cluster_families(self) -> None:
        cases = {
            "ARIA": ("CLUSTER2", "ARIA"),
            "CNS": ("CLUSTER3", "CNS"),
            "CYANA": ("CLUSTER4", "CYANA"),
            "DISCOVER": ("CLUSTER5", "DISCOVER"),
            "DYANA": ("CLUSTER6", "DIANA_DYANA"),
            "X-PLOR": ("CLUSTER7", "XPLOR"),
            "X-PLOR NIH": ("CLUSTER8", "XPLOR_NIH"),
        }

        for program_text, expected in cases.items():
            with self.subTest(program_text=program_text):
                self.assertEqual(
                    classify_solution_nmr_program_cluster(program_text),
                    expected,
                )

    def test_supports_multiple_remark_separator(self) -> None:
        self.assertEqual(
            classify_solution_nmr_program_cluster("DIANA || AMBER 3.0"),
            ("CLUSTER6", "DIANA_DYANA"),
        )

    def test_falls_back_to_other_without_recognized_cluster(self) -> None:
        self.assertEqual(
            classify_solution_nmr_program_cluster("FANTOM"),
            ("CLUSTER9", "OTHER"),
        )
        self.assertEqual(
            classify_solution_nmr_program_cluster(None),
            ("CLUSTER9", "OTHER"),
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
