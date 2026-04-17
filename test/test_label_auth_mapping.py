import unittest

from pdb_data_collector import _map_label_seq_ids_to_auth_seq_ids


def _string_mapping_range(start: int, end: int) -> list[str]:
    return [str(value) for value in range(start, end + 1)]


class MapLabelSeqIdsToAuthSeqIdsTests(unittest.TestCase):
    def test_maps_label_positions_to_auth_sequence_ids(self) -> None:
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={1, 3, 4},
            auth_mapping_raw=[101, 102, 205, 300],
        )

        self.assertEqual(result, {101, 205, 300})

    def test_skips_invalid_and_out_of_range_mapping_values(self) -> None:
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={1, 2, 4, 6},
            auth_mapping_raw=["10", "bad", None, " 40 "],
        )

        self.assertEqual(result, {10, 40})

    def test_falls_back_to_label_ids_when_mapping_has_no_valid_values(self) -> None:
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={2, 5},
            auth_mapping_raw=["bad", None],
        )

        self.assertEqual(result, {2, 5})

    def test_returns_empty_set_for_empty_label_ids(self) -> None:
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids=set(),
            auth_mapping_raw=[1, 2, 3],
        )

        self.assertEqual(result, set())

    def test_maps_real_pdb_example_7aqt_chain_a(self) -> None:
        # RCSB GraphQL, 2026-04-17:
        # 7AQT.A auth_to_entity_poly_seq_mapping = ["4", ..., "112"]
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={1, 10, 109},
            auth_mapping_raw=_string_mapping_range(4, 112),
        )

        self.assertEqual(result, {4, 13, 112})

    def test_maps_real_pdb_example_2flj_chain_a(self) -> None:
        # RCSB GraphQL, 2026-04-17:
        # 2FLJ.A auth_to_entity_poly_seq_mapping = ["0", ..., "133"]
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={1, 2, 134},
            auth_mapping_raw=_string_mapping_range(0, 133),
        )

        self.assertEqual(result, {0, 1, 133})

    def test_maps_real_pdb_example_2lsb_chain_a(self) -> None:
        # RCSB GraphQL, 2026-04-17:
        # 2LSB.A auth_to_entity_poly_seq_mapping = ["90", ..., "231"]
        result = _map_label_seq_ids_to_auth_seq_ids(
            label_seq_ids={1, 53, 142},
            auth_mapping_raw=_string_mapping_range(90, 231),
        )

        self.assertEqual(result, {90, 142, 231})


if __name__ == "__main__":
    unittest.main()
