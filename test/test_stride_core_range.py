import unittest

from pdb_data_collector import _extract_stride_core_range_for_modeled_auth_seq_ids


class ExtractStrideCoreRangeForModeledAuthSeqIdsTests(unittest.TestCase):
    def test_uses_only_hgieb_states_inside_modeled_residues(self) -> None:
        result = _extract_stride_core_range_for_modeled_auth_seq_ids(
            chain_states={
                10: "C",
                11: "H",
                12: "T",
                13: "E",
                14: "B",
                15: "G",
                16: "I",
                17: "C",
                18: "H",
            },
            modeled_auth_seq_ids={10, 11, 12, 13, 14, 15, 16, 17},
        )

        self.assertEqual(result, (11, 16))

    def test_returns_none_when_modeled_residues_have_no_structured_states(self) -> None:
        result = _extract_stride_core_range_for_modeled_auth_seq_ids(
            chain_states={5: "T", 6: "C", 7: "H"},
            modeled_auth_seq_ids={5, 6},
        )

        self.assertIsNone(result)

    def test_keeps_outer_structured_core_range_across_numbering_gap(self) -> None:
        result = _extract_stride_core_range_for_modeled_auth_seq_ids(
            chain_states={10: "H", 11: "E", 12: "G", 13: "I"},
            modeled_auth_seq_ids={10, 11, 13},
        )

        self.assertEqual(result, (10, 13))


if __name__ == "__main__":
    unittest.main()
