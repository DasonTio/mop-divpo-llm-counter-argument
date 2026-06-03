import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_persona_breakdown import judge_scalar, stat


class PersonaBreakdownTests(unittest.TestCase):
    def test_judge_scalar_averages_quality_subscores(self):
        record = {
            "quality": {
                "relevance": 5,
                "coherence": 4,
                "substance": 3,
            }
        }

        self.assertEqual(judge_scalar(record, "quality"), 4.0)

    def test_judge_scalar_extracts_single_axis_scores(self):
        record = {
            "novelty": {"novelty": 5},
            "utility": {"prewriting_utility": 4},
            "persona_fidelity": {"persona_fidelity": 3},
        }

        self.assertEqual(judge_scalar(record, "novelty"), 5.0)
        self.assertEqual(judge_scalar(record, "utility"), 4.0)
        self.assertEqual(judge_scalar(record, "persona_fidelity"), 3.0)

    def test_stat_reports_empty_inputs_explicitly(self):
        self.assertEqual(stat([]), {"mean": None, "std": None, "n": 0})


if __name__ == "__main__":
    unittest.main()
