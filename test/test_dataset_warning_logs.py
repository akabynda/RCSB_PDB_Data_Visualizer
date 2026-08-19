"""Tests for routing dataset-build warnings to per-output log files."""

import logging
import tempfile
import unittest
from pathlib import Path

from src.pdb_dataset_builder import (
    DatasetKind,
    _configure_dataset_warning_logs,
    _set_active_dataset_warning_logs,
)


class DatasetWarningLogTests(unittest.TestCase):
    """Verify warning-log selection and file lifecycle behavior."""

    def test_routes_only_warnings_and_errors_to_active_csv_log(self) -> None:
        """Route only warning-level records to the currently active dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            handlers = _configure_dataset_warning_logs(
                {
                    DatasetKind.METHOD_COUNTS: (first_csv,),
                    DatasetKind.SOLUTION_NMR_WEIGHTS: (second_csv,),
                }
            )
            logger = logging.getLogger("test.dataset.warning.logs")
            try:
                _set_active_dataset_warning_logs((first_csv,))
                logger.info("not written")
                logger.warning("first warning")
                _set_active_dataset_warning_logs((second_csv,))
                logger.error("second error")
            finally:
                _set_active_dataset_warning_logs(())
                root_logger = logging.getLogger()
                for handler in handlers:
                    root_logger.removeHandler(handler)
                    handler.close()

            first_log = (root / "first.log").read_text(encoding="utf-8")
            second_log = (root / "second.log").read_text(encoding="utf-8")
            self.assertIn("WARNING | first warning", first_log)
            self.assertNotIn("not written", first_log)
            self.assertNotIn("second error", first_log)
            self.assertIn("ERROR | second error", second_log)
            self.assertNotIn("first warning", second_log)

    def test_recreates_existing_log_file(self) -> None:
        """Replace stale log contents when handlers are configured again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "dataset.csv"
            output_log = output_csv.with_suffix(".log")
            output_log.write_text("stale warning\n", encoding="utf-8")

            handlers = _configure_dataset_warning_logs(
                {DatasetKind.METHOD_COUNTS: (output_csv,)}
            )
            try:
                self.assertEqual(output_log.read_text(encoding="utf-8"), "")
            finally:
                root_logger = logging.getLogger()
                for handler in handlers:
                    root_logger.removeHandler(handler)
                    handler.close()


if __name__ == "__main__":
    unittest.main()
