import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_divpo_inference import (
    build_messages,
    build_sft_adapter_kwargs,
    configure_warning_filters,
    resolve_adapter,
    score_generation,
)


class ValidateDivPOInferenceTests(unittest.TestCase):
    def test_warning_filter_suppresses_expected_nested_peft_warning_only(self):
        import warnings

        configure_warning_filters()

        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("Already found a `peft_config` attribute in the model.", UserWarning)
            warnings.warn("Found missing adapter keys while loading the checkpoint: x", UserWarning)

        self.assertEqual(len(caught), 1)
        self.assertIn("Found missing adapter keys", str(caught[0].message))

    def test_build_messages_uses_persona_system_prompt_and_counter_argument_task(self):
        personas = {
            "contrarian": {
                "system_prompt": "You are a Contrarian Analyst.",
            }
        }

        messages = build_messages("contrarian", "Remote work is strictly better.", personas)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a Contrarian Analyst.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Generate a counter-argument", messages[1]["content"])
        self.assertIn("Remote work is strictly better.", messages[1]["content"])

    def test_score_generation_flags_supportive_restatement_as_invalid(self):
        result = score_generation(
            "Remote work is strictly better for productivity.",
            "Remote work can be more efficient than in-office work. "
            "Remote work can lead to increased efficiency by eliminating commute time.",
        )

        self.assertFalse(result["passes_gate"])
        self.assertFalse(result["has_counter_signal"])
        self.assertIn("missing counter-argument signal", result["issues"])

    def test_score_generation_accepts_clear_counter_argument(self):
        result = score_generation(
            "Remote work is strictly better for productivity.",
            "However, the claim ignores hidden coordination costs: onboarding, trust, "
            "documentation, and delayed feedback can reduce productivity.",
        )

        self.assertTrue(result["passes_gate"])
        self.assertTrue(result["has_counter_signal"])
        self.assertGreaterEqual(result["word_count"], 10)

    def test_resolve_adapter_prefers_local_then_hub(self):
        local = {
            "minimalist": Path("outputs/adapters/divpo/minimalist"),
        }
        hub_files = {
            "divpo/contrarian/adapter_config.json",
            "divpo/minimalist/adapter_config.json",
        }

        resolved = resolve_adapter(
            "minimalist",
            local_adapters=local,
            hub_files=hub_files,
            model_repo="repo",
        )

        self.assertEqual(resolved["source"], "outputs/adapters/divpo/minimalist")
        self.assertEqual(resolved["kwargs"], {})

        resolved = resolve_adapter(
            "contrarian",
            local_adapters=local,
            hub_files=hub_files,
            model_repo="repo",
        )

        self.assertEqual(resolved["source"], "repo")
        self.assertEqual(resolved["kwargs"], {"subfolder": "divpo/contrarian"})

    def test_build_sft_adapter_kwargs_points_to_matching_persona(self):
        kwargs = build_sft_adapter_kwargs("systems_thinker", "repo", "hf_token")

        self.assertEqual(kwargs["source"], "repo")
        self.assertEqual(kwargs["kwargs"], {"subfolder": "sft/systems_thinker", "token": "hf_token"})


if __name__ == "__main__":
    unittest.main()
