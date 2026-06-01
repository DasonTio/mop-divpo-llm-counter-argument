import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.eval.llm_judge import (
    JudgeCache,
    extract_json,
    judge_output,
    score_all_outputs,
    score_keys_for,
)
from mop_divpo.eval.llm_judge_rubrics import RUBRICS, build_rubric_prompt


class ExtractJsonTests(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(extract_json('{"novelty": 4}'), {"novelty": 4})

    def test_strips_markdown_fence(self):
        self.assertEqual(extract_json('```json\n{"novelty": 5}\n```'), {"novelty": 5})

    def test_extracts_json_embedded_in_prose(self):
        text = 'Here is my score: {"relevance": 3, "coherence": 4} done.'
        self.assertEqual(extract_json(text), {"relevance": 3, "coherence": 4})

    def test_raises_on_no_json(self):
        with self.assertRaises(ValueError):
            extract_json("no json here")


class RubricTests(unittest.TestCase):
    def test_all_four_rubrics_present(self):
        self.assertEqual(set(RUBRICS), {"quality", "persona_fidelity", "novelty", "utility"})

    def test_quality_score_keys(self):
        self.assertEqual(score_keys_for("quality"), ["relevance", "coherence", "substance"])

    def test_build_prompt_injects_fields(self):
        prompt = build_rubric_prompt("quality", output="OUT", prompt="TOPIC")
        self.assertIn("OUT", prompt)
        self.assertIn("TOPIC", prompt)

    def test_build_prompt_missing_required_field_raises(self):
        with self.assertRaises(ValueError):
            build_rubric_prompt("novelty", output="OUT")  # needs prompt + peers

    def test_unknown_rubric_raises(self):
        with self.assertRaises(KeyError):
            build_rubric_prompt("bogus", output="OUT")

    def test_peers_are_numbered_in_novelty_prompt(self):
        prompt = build_rubric_prompt(
            "novelty", output="OUT", prompt="T", peers=["first", "second"]
        )
        self.assertIn("1. first", prompt)
        self.assertIn("2. second", prompt)


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_persists_to_disk(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            cache = JudgeCache(path)
            cache.set("quality", "rendered prompt text", {"relevance": 5})
            cache.save()

            reloaded = JudgeCache(path)
            self.assertEqual(reloaded.get("quality", "rendered prompt text"), {"relevance": 5})

    def test_judge_output_uses_cache_and_skips_call(self):
        calls = []

        def call_fn(prompt: str) -> str:
            calls.append(prompt)
            return '{"prewriting_utility": 4, "rationale": "x"}'

        cache = JudgeCache(None)
        first = judge_output("utility", output="O", prompt="P", call_fn=call_fn, cache=cache)
        second = judge_output("utility", output="O", prompt="P", call_fn=call_fn, cache=cache)

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)  # second call served from cache


class ScoreAllTests(unittest.TestCase):
    def _fake_caller(self):
        def call_fn(prompt: str) -> str:
            if "RELEVANCE" in prompt:
                return '{"relevance": 4, "coherence": 4, "substance": 3, "rationale": "x"}'
            if "cognitive style" in prompt:
                return '{"persona_fidelity": 5, "evidence": "phrase"}'
            if "novel" in prompt:
                return '{"novelty": 3, "most_similar_peer_index": 1, "rationale": "x"}'
            return '{"prewriting_utility": 4, "rationale": "x"}'

        return call_fn

    def test_score_all_skips_fidelity_for_none_persona(self):
        outputs = [
            {"output_id": "base__t__0", "method": "base", "prompt": "Topic?",
             "persona": None, "output": "an argument"},
            {"output_id": "base__t__1", "method": "base", "prompt": "Topic?",
             "persona": None, "output": "another argument"},
        ]
        scored = score_all_outputs(outputs, call_fn=self._fake_caller())
        self.assertIsNone(scored[0]["persona_fidelity"])
        self.assertEqual(scored[0]["quality"]["relevance"], 4)

    def test_score_all_runs_fidelity_for_persona(self):
        outputs = [
            {"output_id": "mop__t__0", "method": "mop_divpo", "prompt": "Topic?",
             "persona": "contrarian", "output": "a contrarian argument"},
            {"output_id": "mop__t__1", "method": "mop_divpo", "prompt": "Topic?",
             "persona": "minimalist", "output": "a minimal argument"},
        ]
        scored = score_all_outputs(outputs, call_fn=self._fake_caller())
        self.assertEqual(scored[0]["persona_fidelity"]["persona_fidelity"], 5)


if __name__ == "__main__":
    unittest.main()
