import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.training_env import check_torchao_compatibility


class TrainingEnvTests(unittest.TestCase):
    def test_check_torchao_compatibility_rejects_old_torchao(self):
        with patch("mop_divpo.training_env.metadata.version", return_value="0.10.0"):
            with self.assertRaisesRegex(RuntimeError, "pip uninstall -y torchao"):
                check_torchao_compatibility()

    def test_check_torchao_compatibility_rejects_boundary_torchao(self):
        with patch("mop_divpo.training_env.metadata.version", return_value="0.16.0"):
            with self.assertRaisesRegex(RuntimeError, "torchao>0.16.0"):
                check_torchao_compatibility()

    def test_check_torchao_compatibility_accepts_supported_torchao(self):
        with patch("mop_divpo.training_env.metadata.version", return_value="0.16.1"):
            check_torchao_compatibility()

    def test_check_torchao_compatibility_accepts_missing_torchao(self):
        from importlib import metadata

        with patch(
            "mop_divpo.training_env.metadata.version",
            side_effect=metadata.PackageNotFoundError,
        ):
            check_torchao_compatibility()


if __name__ == "__main__":
    unittest.main()
