import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_divpo import add_supported_trainer_kwargs, build_parser, split_supported_kwargs


class TrainDivPOCompatibilityTests(unittest.TestCase):
    def test_default_training_args_are_t4_safe_for_dpo_reference_forward(self):
        args = build_parser().parse_args(["--persona", "systems_thinker"])

        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.grad_accum, 32)
        self.assertEqual(args.max_length, 384)
        self.assertIsNone(args.dataset_dir)
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(args.sft_stage, "sft")
        self.assertIsNone(args.sft_adapter_dir)
        self.assertEqual(args.output_stage, "divpo")

    def test_can_target_local_dataset_and_new_output_stage(self):
        args = build_parser().parse_args(
            [
                "--persona",
                "systems_thinker",
                "--dataset-dir",
                "data/processed/divpo_v2",
                "--base-model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--sft-stage",
                "sft_1p5b",
                "--sft-adapter-dir",
                "outputs/adapters",
                "--output-stage",
                "divpo_v2_1p5b",
            ]
        )

        self.assertEqual(args.dataset_dir, "data/processed/divpo_v2")
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(args.sft_stage, "sft_1p5b")
        self.assertEqual(args.sft_adapter_dir, "outputs/adapters")
        self.assertEqual(args.output_stage, "divpo_v2_1p5b")

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

    def test_add_supported_trainer_kwargs_ignores_known_optional_unsupported_args(self):
        class CurrentDPOTrainer:
            def __init__(self, model, args):
                pass

        trainer_kwargs = {"model": object(), "args": object()}
        add_supported_trainer_kwargs(
            trainer_kwargs,
            CurrentDPOTrainer,
            {
                "group_by_length": True,
                "max_prompt_length": 256,
            },
        )

        self.assertEqual(set(trainer_kwargs), {"model", "args"})


if __name__ == "__main__":
    unittest.main()
