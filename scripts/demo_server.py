#!/usr/bin/env python3
"""MoP+DivPO interactive demo server.

Usage (from project root):
    source .venv/bin/activate
    PYTHONPATH=src python scripts/demo_server.py [--port 5001] [--host 127.0.0.1]

Then open http://localhost:5001
"""
from __future__ import annotations

import threading
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

app = Flask(__name__)

# ── static config ─────────────────────────────────────────────────────────────

EXAMPLES = [
    "What is the future of remote work?",
    "How should cities adapt to climate change?",
    "What's wrong with current education systems?",
    "Why is housing unaffordable in major cities?",
    "CMV: Social media has done more harm than good for society.",
    "CMV: Universal basic income would discourage people from working.",
    "CMV: College degrees are no longer worth the cost for most people.",
    "CMV: Working from home makes employees less productive.",
    "Why is loneliness rising in connected societies?",
    "What does meaningful work look like in 2035?",
]

METHODS = {
    "base": {
        "label": "Base",
        "description": "Raw Qwen2.5-0.5B-Instruct. No persona prompt, no adapter.",
        "stage": "base",
        "use_persona_prompt": False,
        "multi_persona": False,
        "metrics": {"Self-BLEU↓": 0.086, "Distinct-2↑": 0.894, "SBERT↓": 0.675, "Quality↑": 3.417},
    },
    "prompt_only": {
        "label": "Prompt-only",
        "description": "Persona system prompts injected at inference. No LoRA adapter.",
        "stage": "base",
        "use_persona_prompt": True,
        "multi_persona": True,
        "metrics": {"Self-BLEU↓": 0.062, "Distinct-2↑": 0.923, "SBERT↓": 0.708, "Quality↑": 3.975},
    },
    "single_lora": {
        "label": "Single LoRA",
        "description": "One LoRA trained on all personas combined; persona prompt at inference.",
        "stage": "single",
        "use_persona_prompt": True,
        "multi_persona": True,
        "metrics": {"Self-BLEU↓": 0.040, "Distinct-2↑": 0.945, "SBERT↓": 0.481, "Quality↑": 2.589},
    },
    "mop_sft": {
        "label": "MoP SFT",
        "description": "Mixture of Personas — 4 dedicated SFT LoRA adapters, one per persona.",
        "stage": "sft",
        "use_persona_prompt": True,
        "multi_persona": True,
        "metrics": {"Self-BLEU↓": 0.026, "Distinct-2↑": 0.943, "SBERT↓": 0.488, "Quality↑": 2.633},
    },
    "mop_divpo": {
        "label": "MoP + DivPO",
        "description": "MoP SFT + DivPO alignment (v1, within-persona rarity scoring).",
        "stage": "divpo",
        "use_persona_prompt": True,
        "multi_persona": True,
        "metrics": {"Self-BLEU↓": 0.056, "Distinct-2↑": 0.920, "SBERT↓": 0.705, "Quality↑": 3.931},
    },
    "mop_divpo_v2": {
        "label": "MoP + DivPO v2",
        "description": "DivPO v2 with cross-persona rarity scoring (improved production model).",
        "stage": "divpo_v2",
        "use_persona_prompt": True,
        "multi_persona": True,
        "metrics": {"Self-BLEU↓": 0.051, "Distinct-2↑": 0.931, "SBERT↓": 0.690, "Quality↑": 3.956},
    },
}

# Best value per metric (used by frontend to highlight best-in-class)
BEST_METRICS = {
    "Self-BLEU↓": 0.026,
    "Distinct-2↑": 0.945,
    "SBERT↓": 0.481,
    "Quality↑": 3.975,
}

PERSONA_INFO = {
    "contrarian": {
        "label": "Contrarian",
        "tagline": "Challenges hidden assumptions",
        "color": "#e85d75",
    },
    "systems_thinker": {
        "label": "Systems Thinker",
        "tagline": "Maps feedback loops & side effects",
        "color": "#5b9bd5",
    },
    "cross_domain_analogist": {
        "label": "Cross-Domain Analogist",
        "tagline": "Borrows mechanisms from other domains",
        "color": "#4caf78",
    },
    "minimalist": {
        "label": "Minimalist",
        "tagline": "Exposes the core tension",
        "color": "#f0b429",
    },
}

PERSONA_ORDER = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]

# ── model cache (one generator in memory at a time) ───────────────────────────

_lock = threading.Lock()
_generator = None
_generator_key: str | None = None


def _get_generator(stage: str, use_persona_prompt: bool):
    global _generator, _generator_key
    key = f"{stage}:{use_persona_prompt}"
    if _generator_key != key:
        if _generator is not None:
            _generator.unload()
        from mop_divpo.inference.generate import MoPGenerator
        _generator = MoPGenerator(
            adapter_stage=stage,
            use_persona_prompt=use_persona_prompt,
        )
        _generator_key = key
    return _generator


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(Path(__file__).parent / "demo.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "methods": METHODS,
        "examples": EXAMPLES,
        "best_metrics": BEST_METRICS,
        "persona_info": PERSONA_INFO,
        "persona_order": PERSONA_ORDER,
    })


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True)
    method_id = data.get("method", "mop_divpo_v2")
    prompt = data.get("prompt", "").strip()
    n = max(1, min(int(data.get("n", 1)), 5))

    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    if method_id not in METHODS:
        return jsonify({"error": f"unknown method {method_id!r}"}), 400

    cfg = METHODS[method_id]
    personas = PERSONA_ORDER if cfg["multi_persona"] else [None]

    try:
        with _lock:
            gen = _get_generator(cfg["stage"], cfg["use_persona_prompt"])
            outputs = {
                str(p): gen.generate(
                    prompt,
                    persona=p,
                    n=n,
                    as_counter_argument=True,
                )
                for p in personas
            }
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"outputs": outputs})


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoP+DivPO demo server")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  MoP+DivPO Demo → http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
