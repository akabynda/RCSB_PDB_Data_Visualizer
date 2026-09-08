"""Tests for deriving structured core ranges from STRIDE states."""

import unittest

from src.pdb_dataset_builder import _extract_stride_core_range_for_modeled_auth_seq_ids
from src.dataset.records import ResidueId


class ExtractStrideCoreRangeForModeledAuthSeqIdsTests(unittest.TestCase):
    """Verify structured-core selection within modeled residue identifiers."""

    def test_uses_only_hgieb_states_inside_modeled_residues(self) -> None:
        """Use only structured HGIEB states attached to modeled residues."""
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
        """Return no range when modeled residues contain no structured state."""
        result = _extract_stride_core_range_for_modeled_auth_seq_ids(
            chain_states={5: "T", 6: "C", 7: "H"},
            modeled_auth_seq_ids={5, 6},
        )

        self.assertIsNone(result)

    def test_keeps_outer_structured_core_range_across_numbering_gap(self) -> None:
        """Keep the outer structured bounds across an authorization-number gap."""
        result = _extract_stride_core_range_for_modeled_auth_seq_ids(
            chain_states={10: "H", 11: "E", 12: "G", 13: "I"},
            modeled_auth_seq_ids={10, 11, 13},
        )

        self.assertEqual(result, (10, 13))


if __name__ == "__main__":
    unittest.main()


def test_core_endpoint_includes_the_last_structured_insertion() -> None:
    ids = [ResidueId(56), *(ResidueId(56, code) for code in "ABCDE"), ResidueId(57)]
    states = dict(zip(ids, ["H", "T", "T", "E", "E", "E", "C"]))
    assert _extract_stride_core_range_for_modeled_auth_seq_ids(states, ids) == (
        ResidueId(56),
        ResidueId(56, "E"),
    )


def test_core_endpoints_follow_reverse_insertion_and_nonmonotonic_sequence_order() -> (
    None
):
    ids = [ResidueId(1, "R"), ResidueId(1, "Q"), ResidueId(1), ResidueId(-3)]
    states = dict(zip(ids, ["E", "H", "H", "C"]))
    assert _extract_stride_core_range_for_modeled_auth_seq_ids(states, ids) == (
        ResidueId(1, "R"),
        ResidueId(1),
    )
