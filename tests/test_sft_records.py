import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.data.sft_records import load_sft_jsonl_records, normalize_sft_record


class SFTRecordTests(unittest.TestCase):
    def test_load_sft_jsonl_records_preserves_chat_messages(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
            {"role": "assistant", "content": "assistant response"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contrarian.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": messages,
                        "metadata": {
                            "persona": "contrarian",
                            "source": "unit",
                            "source_id": "1",
                        },
                        "prompt": "legacy extra column should be ignored",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_sft_jsonl_records(path, persona="contrarian")

        self.assertEqual(
            records,
            [
                {
                    "messages": messages,
                    "metadata": {
                        "persona": "contrarian",
                        "source": "unit",
                        "source_id": "1",
                    },
                }
            ],
        )

    def test_normalize_sft_record_converts_legacy_flat_prompt_response(self):
        record = normalize_sft_record(
            {
                "persona": "minimalist",
                "source": "legacy",
                "id": "abc",
                "prompt": "We should abandon marriage",
                "response": "The premise treats one institution as the whole problem.",
                "metadata": "{\"quality\": 0.9}",
            },
            persona="minimalist",
        )

        self.assertEqual(
            record["metadata"],
            {
                "persona": "minimalist",
                "source": "legacy",
                "source_id": "abc",
            },
        )
        self.assertEqual(
            [m["role"] for m in record["messages"]],
            ["system", "user", "assistant"],
        )
        self.assertIn("Minimalist Designer", record["messages"][0]["content"])
        self.assertEqual(record["messages"][1]["content"], "We should abandon marriage")
        self.assertEqual(
            record["messages"][2]["content"],
            "The premise treats one institution as the whole problem.",
        )


if __name__ == "__main__":
    unittest.main()
