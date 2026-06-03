import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.eval.persona_diversity import build_persona_diversity_report


class PersonaDiversityTests(unittest.TestCase):
    def test_prompt_controlled_inter_and_intra_metrics(self):
        records = [
            {"method": "base", "prompt": "p1", "persona": None, "output": "b1"},
            {"method": "base", "prompt": "p1", "persona": None, "output": "b2"},
            {"method": "prompt_only", "prompt": "p1", "persona": "contrarian", "output": "c1"},
            {"method": "prompt_only", "prompt": "p1", "persona": "systems_thinker", "output": "s1"},
            {"method": "prompt_only", "prompt": "p2", "persona": "contrarian", "output": "c2"},
            {"method": "prompt_only", "prompt": "p2", "persona": "systems_thinker", "output": "s2"},
            {"method": "repeat", "prompt": "p1", "persona": "contrarian", "output": "r1"},
            {"method": "repeat", "prompt": "p1", "persona": "contrarian", "output": "r2"},
            {"method": "repeat", "prompt": "p1", "persona": "systems_thinker", "output": "r3"},
            {"method": "repeat", "prompt": "p1", "persona": "systems_thinker", "output": "r4"},
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],      # base p1 sample 1
                [0.0, 1.0],      # base p1 sample 2 => distance 1
                [1.0, 0.0],      # prompt_only p1 contrarian
                [0.0, 1.0],      # prompt_only p1 systems => distance 1
                [1.0, 0.0],      # prompt_only p2 contrarian
                [1.0, 0.0],      # prompt_only p2 systems => distance 0
                [1.0, 0.0],      # repeat contrarian sample 1
                [1.0, 0.0],      # repeat contrarian sample 2 => intra distance 0
                [0.0, 1.0],      # repeat systems sample 1
                [0.0, 1.0],      # repeat systems sample 2 => intra distance 0
            ],
            dtype=np.float32,
        )

        report = build_persona_diversity_report(records, embeddings, run_label="toy")

        base_intra = report["metrics"]["base"]["same_prompt_intra_persona"]["summary"]
        self.assertAlmostEqual(base_intra["mean"], 1.0)
        self.assertEqual(base_intra["n"], 1)

        prompt_inter = report["metrics"]["prompt_only"]["prompt_controlled_inter_persona"]["summary"]
        self.assertAlmostEqual(prompt_inter["mean"], 0.5)
        self.assertEqual(prompt_inter["n"], 2)

        ratio = report["metrics"]["prompt_only"]["persona_over_base_sampling"]["ratio"]
        self.assertAlmostEqual(ratio["mean"], 1.0)
        self.assertEqual(ratio["n"], 1)

        repeat_intra = report["metrics"]["repeat"]["same_prompt_intra_persona"]["summary"]
        self.assertAlmostEqual(repeat_intra["mean"], 0.0)
        self.assertEqual(repeat_intra["n"], 2)

    def test_no_true_intra_when_only_one_sample_per_prompt_persona(self):
        records = [
            {"method": "mop_sft", "prompt": "p1", "persona": "contrarian", "output": "a"},
            {"method": "mop_sft", "prompt": "p1", "persona": "systems_thinker", "output": "b"},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        report = build_persona_diversity_report(records, embeddings, run_label="toy")

        intra = report["metrics"]["mop_sft"]["same_prompt_intra_persona"]
        self.assertFalse(intra["available"])
        self.assertIsNone(intra["summary"]["mean"])
        self.assertEqual(intra["summary"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
