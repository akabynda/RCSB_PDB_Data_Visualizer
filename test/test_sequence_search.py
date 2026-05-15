import unittest

from pdb_data_collector import CollectorConfig, RCSBClient


class _NullResultSetResponse:
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {"total_count": 0, "result_set": None}

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def post(self, *args, **kwargs) -> _NullResultSetResponse:
        return _NullResultSetResponse()


class SequenceSearchTests(unittest.TestCase):
    def test_treats_null_result_set_as_empty_search_result(self) -> None:
        client = RCSBClient(CollectorConfig(retries=1))
        client.session = _FakeSession()

        result = client.fetch_xray_polymer_entity_ids_by_sequence(
            sequence="ACDEFGHIKLMNPQRSTVWY",
            sequence_identity_percent=100,
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
