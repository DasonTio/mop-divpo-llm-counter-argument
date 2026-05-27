import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "KAGGLE_DIVPO.ipynb"


def notebook_sources() -> list[str]:
    with NOTEBOOK.open(encoding="utf-8") as f:
        nb = json.load(f)
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


class KaggleNotebookTests(unittest.TestCase):
    def test_training_cells_use_explicit_accelerate_config_and_safe_batch(self):
        sources = "\n".join(notebook_sources())

        # Notebook builds cmd as a Python list; check for the quoted list-item form.
        self.assertIn("/tmp/accel_config.yaml", sources)
        self.assertIn('"--num_processes"', sources)
        self.assertIn('"--mixed_precision"', sources)
        self.assertIn('"fp16"', sources)
        self.assertIn('"--batch-size"', sources)
        self.assertIn('"--grad-accum"', sources)
        self.assertIn('"--max-length"', sources)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", sources)

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
        self.assertIn("sft_subfolder", verify_source)
        self.assertIn("PeftModel.from_pretrained(base, MODEL_REPO", verify_source)
        self.assertIn("PeftModel.from_pretrained(model, adapter_source", verify_source)
        self.assertIn("warnings.filterwarnings", verify_source)
        self.assertIn("torch.cuda.empty_cache()", verify_source)

    def test_verify_cell_auto_selects_available_persona_when_unset(self):
        verify_source = "\n".join(
            source
            for source in notebook_sources()
            if "PeftModel.from_pretrained" in source
        )

        self.assertIn('PERSONA_TO_VERIFY = None', verify_source)
        self.assertIn("available_hub_personas", verify_source)
        self.assertIn("resolved_persona", verify_source)
        self.assertIn("Auto-selected", verify_source)
        self.assertNotIn('PERSONA_TO_VERIFY = "contrarian"', verify_source)

    def test_notebook_runs_pre_eval_validation_script(self):
        sources = "\n".join(notebook_sources())

        self.assertIn("Pre-Evaluation Inference Validation", sources)
        self.assertIn("scripts/validate_divpo_inference.py", sources)
        self.assertIn("pre_eval_validation.jsonl", sources)


if __name__ == "__main__":
    unittest.main()
