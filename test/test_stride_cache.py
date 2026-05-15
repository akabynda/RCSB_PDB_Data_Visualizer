import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdb_data_collector import load_first_model_stride_state_by_chain


def _ca_line(serial: int, resid: int) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{resid:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        "           C\n"
    )


class StrideCacheTests(unittest.TestCase):
    def test_reuses_cached_first_model_stride_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "1ABC.pdb"
            cache_dir = root / "stride_cache"
            pdb_path.write_text(
                "".join(
                    [
                        "MODEL        1\n",
                        _ca_line(1, 1),
                        "ENDMDL\n",
                        "MODEL        2\n",
                        _ca_line(2, 1),
                        "ENDMDL\n",
                    ]
                ),
                encoding="utf-8",
            )

            expected_states = {"A": {1: "H"}}
            with patch(
                "pdb_data_collector._run_stride_for_model_text",
                return_value=expected_states,
            ) as run_stride:
                first_states, first_model_count = load_first_model_stride_state_by_chain(
                    pdb_path=pdb_path,
                    entry_id="1ABC",
                    stride_executable="stride",
                    stride_cache_dir=cache_dir,
                )
                second_states, second_model_count = load_first_model_stride_state_by_chain(
                    pdb_path=pdb_path,
                    entry_id="1ABC",
                    stride_executable="stride",
                    stride_cache_dir=cache_dir,
                )

            self.assertEqual(first_states, expected_states)
            self.assertEqual(second_states, expected_states)
            self.assertEqual(first_model_count, 2)
            self.assertEqual(second_model_count, 2)
            self.assertEqual(run_stride.call_count, 1)
            self.assertTrue((cache_dir / "1ABC.json").exists())


if __name__ == "__main__":
    unittest.main()
