import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "KAGGLE_DIVPO.ipynb"


def notebook_sources() -> list[str]:
    with NOTEBOOK.open(encoding="utf-8") as f:
        nb = json.load(f)
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


class KaggleNotebookTests(unittest.TestCase):
    def test_verify_cell_loads_local_divpo_adapter_before_hub_fallback(self):
        verify_cells = [
            source
            for source in notebook_sources()
            if "Verify — Load DivPO adapter and generate" in source
            or "PeftModel.from_pretrained" in source
        ]
        self.assertTrue(verify_cells)
        verify_source = "\n".join(verify_cells)

        self.assertIn("outputs/adapters/divpo", verify_source)
        self.assertIn("adapter_config.json", verify_source)
        self.assertIn("adapter_source", verify_source)
        self.assertIn("PeftModel.from_pretrained(base, adapter_source", verify_source)


if __name__ == "__main__":
    unittest.main()
