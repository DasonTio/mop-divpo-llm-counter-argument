import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mop_divpo.inference.generate import adapter_chain, build_messages
from mop_divpo.eval.aggregate import (
    bootstrap_paired_pvalue,
    build_csv,
    build_markdown_table,
    per_prompt_automated_metrics,
    per_prompt_judge_metrics,
    summarize,
)
from scripts.experiment_persona_distinctness import compute_prompt_conditioned_matrix
from scripts.run_baseline_evaluation import (
    build_method_configs,
    filter_records_to_prompts,
)


class AdapterChainTests(unittest.TestCase):
    def test_base_loads_nothing(self):
        self.assertEqual(adapter_chain("base", None), [])

    def test_sft_loads_one(self):
        self.assertEqual(adapter_chain("sft", "contrarian"), ["sft/contrarian"])

    def test_divpo_stacks_sft_then_divpo(self):
        self.assertEqual(
            adapter_chain("divpo", "minimalist"),
            ["sft/minimalist", "divpo/minimalist"],
        )

    def test_divpo_v2_stacks_sft_then_divpo_v2(self):
        self.assertEqual(
            adapter_chain("divpo_v2", "minimalist"),
            ["sft/minimalist", "divpo_v2/minimalist"],
        )

    def test_custom_sft_stage_loads_one(self):
        self.assertEqual(
            adapter_chain("sft_1p5b", "contrarian"),
            ["sft_1p5b/contrarian"],
        )

    def test_custom_divpo_stage_uses_matching_sft_stage(self):
        self.assertEqual(
            adapter_chain("divpo_v2_1p5b", "minimalist", sft_stage="sft_1p5b"),
            ["sft_1p5b/minimalist", "divpo_v2_1p5b/minimalist"],
        )

    def test_single_uses_single_name(self):
        self.assertEqual(adapter_chain("single", None, single_name="all"), ["single/all"])

    def test_sft_without_persona_raises(self):
        with self.assertRaises(ValueError):
            adapter_chain("sft", None)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            adapter_chain("bogus", "contrarian")


class BuildMessagesTests(unittest.TestCase):
    def test_no_system_prompt_for_base(self):
        messages = build_messages("topic", system_prompt=None)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "topic")

    def test_counter_argument_wrapping(self):
        messages = build_messages("Remote work is best.", as_counter_argument=True)
        self.assertIn("counter-argument", messages[0]["content"])
        self.assertIn("Remote work is best.", messages[0]["content"])

    def test_system_prompt_added_first(self):
        messages = build_messages("topic", system_prompt="You are X.")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")


class BaselineEvaluationConfigTests(unittest.TestCase):
    def test_method_overrides_keep_scaling_stages_isolated(self):
        configs = build_method_configs(
            stage_overrides={
                "mop_sft": "sft_1p5b",
                "mop_divpo_v2": "divpo_v2_1p5b",
            },
            sft_stage_overrides={"mop_divpo_v2": "sft_1p5b"},
        )

        self.assertEqual(configs["mop_sft"]["stage"], "sft_1p5b")
        self.assertEqual(configs["mop_divpo_v2"]["stage"], "divpo_v2_1p5b")
        self.assertEqual(configs["mop_divpo_v2"]["sft_stage"], "sft_1p5b")
        self.assertEqual(configs["prompt_only"]["stage"], "base")


class AggregateTests(unittest.TestCase):
    def _fake_embed(self, texts):
        # deterministic embedding: length + first-char code, normalized later
        return np.array([[len(t), float(ord(t[0]) if t else 0)] for t in texts], dtype=np.float32)

    def test_automated_metrics_grouped_per_method(self):
        grouped = {
            ("mop_divpo", "p1"): ["alpha beta gamma", "delta epsilon zeta"],
            ("mop_divpo", "p2"): ["one two three", "four five six"],
            ("base", "p1"): ["same words here", "same words here"],
        }
        obs = per_prompt_automated_metrics(grouped, self._fake_embed)
        self.assertEqual(len(obs["mop_divpo"]["self_bleu"]), 2)
        self.assertEqual(len(obs["base"]["self_bleu"]), 1)
        # identical base outputs -> high self-bleu vs diverse mop
        self.assertGreater(obs["base"]["self_bleu"][0], min(obs["mop_divpo"]["self_bleu"]))

    def test_judge_metrics_average_per_prompt(self):
        scored = [
            {"method": "mop_divpo", "prompt": "p1", "persona": "contrarian",
             "quality": {"relevance": 4, "coherence": 4, "substance": 4},
             "novelty": {"novelty": 5}, "utility": {"prewriting_utility": 3},
             "persona_fidelity": {"persona_fidelity": 5}},
            {"method": "mop_divpo", "prompt": "p1", "persona": "minimalist",
             "quality": {"relevance": 2, "coherence": 2, "substance": 2},
             "novelty": {"novelty": 1}, "utility": {"prewriting_utility": 1},
             "persona_fidelity": {"persona_fidelity": 1}},
        ]
        obs = per_prompt_judge_metrics(scored)
        # quality per output: 4 and 2 -> mean 3
        self.assertAlmostEqual(obs["mop_divpo"]["quality"][0], 3.0)
        self.assertAlmostEqual(obs["mop_divpo"]["novelty"][0], 3.0)

    def test_judge_metrics_skip_none_fidelity(self):
        scored = [
            {"method": "base", "prompt": "p1", "persona": None,
             "quality": {"relevance": 3, "coherence": 3, "substance": 3},
             "novelty": {"novelty": 2}, "utility": {"prewriting_utility": 2},
             "persona_fidelity": None},
        ]
        obs = per_prompt_judge_metrics(scored)
        self.assertNotIn("persona_fidelity", obs["base"])

    def test_summarize_reports_mean_std_n(self):
        obs = {"m": {"x": [1.0, 2.0, 3.0]}}
        summary = summarize(obs)
        self.assertAlmostEqual(summary["m"]["x"]["mean"], 2.0)
        self.assertEqual(summary["m"]["x"]["n"], 3)

    def test_bootstrap_detects_clear_winner_lower_better(self):
        # a is clearly lower (better) than b on a lower-is-better metric
        a = [0.1, 0.1, 0.1, 0.1, 0.1]
        b = [0.9, 0.9, 0.9, 0.9, 0.9]
        result = bootstrap_paired_pvalue(a, b, lower_is_better=True, seed=1)
        self.assertGreater(result["mean_diff"], 0)  # positive = a better
        self.assertLess(result["p_value"], 0.05)

    def test_bootstrap_higher_better_direction(self):
        a = [5.0, 5.0, 5.0, 5.0]
        b = [1.0, 1.0, 1.0, 1.0]
        result = bootstrap_paired_pvalue(a, b, lower_is_better=False, seed=1)
        self.assertGreater(result["mean_diff"], 0)

    def test_bootstrap_empty_is_safe(self):
        result = bootstrap_paired_pvalue([], [], lower_is_better=True)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_markdown_table_has_method_rows_and_arrows(self):
        summary = {
            "base": {"self_bleu": {"mean": 0.5, "std": 0.1, "n": 3}},
            "mop_divpo": {"self_bleu": {"mean": 0.2, "std": 0.1, "n": 3}},
        }
        table = build_markdown_table(summary, ["base", "mop_divpo"])
        self.assertIn("Self-BLEU↓", table)
        self.assertIn("base", table)
        self.assertIn("mop_divpo", table)
        self.assertIn("0.200", table)

    def test_csv_header_and_rows(self):
        summary = {"base": {"self_bleu": {"mean": 0.5, "std": 0.0, "n": 1}}}
        csv = build_csv(summary, ["base"])
        self.assertIn("method,self_bleu", csv)
        self.assertIn("base,0.5000", csv)


class PersonaDistinctnessTests(unittest.TestCase):
    def test_matrix_compares_personas_within_same_prompt_before_averaging(self):
        records = [
            {"prompt_id": 0, "persona": "a", "text": "p0 a one"},
            {"prompt_id": 0, "persona": "a", "text": "p0 a two"},
            {"prompt_id": 0, "persona": "b", "text": "p0 b one"},
            {"prompt_id": 0, "persona": "b", "text": "p0 b two"},
            {"prompt_id": 1, "persona": "a", "text": "p1 a one"},
            {"prompt_id": 1, "persona": "a", "text": "p1 a two"},
            {"prompt_id": 1, "persona": "b", "text": "p1 b one"},
            {"prompt_id": 1, "persona": "b", "text": "p1 b two"},
        ]

        def fake_embed(texts):
            mapping = {
                "p0 a one": [1.0, 0.0],
                "p0 a two": [1.0, 0.0],
                "p0 b one": [1.0, 0.0],
                "p0 b two": [1.0, 0.0],
                "p1 a one": [0.0, 1.0],
                "p1 a two": [0.0, 1.0],
                "p1 b one": [0.0, 1.0],
                "p1 b two": [0.0, 1.0],
            }
            return np.array([mapping[text] for text in texts], dtype=np.float32)

        matrix = compute_prompt_conditioned_matrix(records, ["a", "b"], fake_embed)

        self.assertEqual(matrix["a"]["a"], 1.0)
        self.assertEqual(matrix["b"]["b"], 1.0)
        self.assertAlmostEqual(matrix["a"]["b"], 1.0)
        self.assertAlmostEqual(matrix["b"]["a"], 1.0)


class BaselineRunnerTests(unittest.TestCase):
    def test_filter_records_to_prompts_keeps_only_requested_prompt_set(self):
        records = [
            {"method": "base", "prompt": "p1", "output": "a"},
            {"method": "mop_sft", "prompt": "p1", "output": "b"},
            {"method": "base", "prompt": "p2", "output": "c"},
            {"method": "mop_sft", "prompt": "p3", "output": "d"},
        ]

        filtered = filter_records_to_prompts(records, ["p1", "p3"])

        self.assertEqual([record["prompt"] for record in filtered], ["p1", "p1", "p3"])


if __name__ == "__main__":
    unittest.main()
