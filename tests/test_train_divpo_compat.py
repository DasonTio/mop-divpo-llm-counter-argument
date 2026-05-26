import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_divpo import add_supported_trainer_kwargs, split_supported_kwargs


class TrainDivPOCompatibilityTests(unittest.TestCase):
    def test_split_supported_kwargs_keeps_supported_config_args(self):
        class ModernDPOConfig:
            def __init__(self, output_dir, beta, max_prompt_length, max_length):
                pass

        supported, unsupported = split_supported_kwargs(
            ModernDPOConfig,
            {
                "output_dir": "out",
                "beta": 0.1,
                "max_prompt_length": 256,
                "max_length": 512,
            },
        )

        self.assertEqual(
            supported,
            {
                "output_dir": "out",
                "beta": 0.1,
                "max_prompt_length": 256,
                "max_length": 512,
            },
        )
        self.assertEqual(unsupported, {})

    def test_split_supported_kwargs_moves_legacy_length_args_out_of_config(self):
        class LegacyDPOConfig:
            def __init__(self, output_dir, beta):
                pass

        supported, unsupported = split_supported_kwargs(
            LegacyDPOConfig,
            {
                "output_dir": "out",
                "beta": 0.1,
                "max_prompt_length": 256,
                "max_length": 512,
            },
        )

        self.assertEqual(supported, {"output_dir": "out", "beta": 0.1})
        self.assertEqual(
            unsupported,
            {
                "max_prompt_length": 256,
                "max_length": 512,
            },
        )

    def test_split_supported_kwargs_allows_var_keyword_signatures(self):
        class FlexibleDPOConfig:
            def __init__(self, **kwargs):
                pass

        supported, unsupported = split_supported_kwargs(
            FlexibleDPOConfig,
            {
                "output_dir": "out",
                "max_prompt_length": 256,
            },
        )

        self.assertEqual(supported, {"output_dir": "out", "max_prompt_length": 256})
        self.assertEqual(unsupported, {})

    def test_add_supported_trainer_kwargs_forwards_legacy_length_args(self):
        class LegacyDPOTrainer:
            def __init__(self, model, args, max_prompt_length, max_length):
                pass

        trainer_kwargs = {"model": object(), "args": object()}

        with self.assertWarns(RuntimeWarning):
            add_supported_trainer_kwargs(
                trainer_kwargs,
                LegacyDPOTrainer,
                {
                    "max_prompt_length": 256,
                    "max_length": 512,
                    "unknown_future_arg": "ignored",
                },
            )

        self.assertEqual(trainer_kwargs["max_prompt_length"], 256)
        self.assertEqual(trainer_kwargs["max_length"], 512)
        self.assertNotIn("unknown_future_arg", trainer_kwargs)


if __name__ == "__main__":
    unittest.main()
