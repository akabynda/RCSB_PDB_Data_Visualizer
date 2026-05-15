import tempfile
import unittest
from pathlib import Path

from pdb_data_collector import (
    CollectorConfig,
    RCSBClient,
    SolutionNMRMonomerPrecisionCollector,
)


def _ca_line(serial: int, resid: int) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{float(resid):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class PrecisionSkipLoggingTests(unittest.TestCase):
    def test_logs_when_core_has_fewer_than_two_coordinate_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "one_model.pdb"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        _ca_line(2, 2),
                        _ca_line(3, 3),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )
            collector = SolutionNMRMonomerPrecisionCollector(
                client=RCSBClient(CollectorConfig()),
                config=CollectorConfig(),
                cache_dir=Path(tmpdir),
                precision_workers=1,
            )

            with self.assertLogs("pdb_data_collector", level="INFO") as logs:
                record = collector._build_record_from_core_range(
                    pdb_path=pdb_path,
                    entry_id="TEST",
                    year=2000,
                    chain_id="A",
                    core_start_seq_id=1,
                    core_end_seq_id=3,
                )

            self.assertIsNone(record)
            self.assertTrue(
                any(
                    "Skipping precision entry TEST chain A: fewer than 2 coordinate models"
                    in message
                    for message in logs.output
                )
            )


if __name__ == "__main__":
    unittest.main()
