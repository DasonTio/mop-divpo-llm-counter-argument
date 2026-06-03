"""Unified persona inference for every evaluation method.

One code path produces outputs for all five baselines in the research plan:

| Method        | stage    | persona prompt | adapters loaded                    |
|---------------|----------|----------------|------------------------------------|
| Base          | base     | no             | none                               |
| Prompt-only   | base     | yes            | none                               |
| Single LoRA   | single   | yes            | single/{single_name}               |
| MoP SFT       | sft      | yes            | sft/{persona}                      |
| MoP + DivPO   | divpo    | yes            | sft/{persona} + divpo/{persona}    |

DivPO is trained as a second LoRA stacked on the frozen SFT adapter
(see scripts/train_divpo.py), so the divpo stage loads both, in order.

`MoPGenerator` caches each loaded model so a long sweep (hundreds of prompts)
only pays the load cost once per persona. `generate(...)` is a thin convenience
wrapper matching the smoke-test command in docs/research-plan.md §10.
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

# personas.py lives at src/personas.py — ensure src is importable when this
# package is used via PYTHONPATH=src or an editable install.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REPO = "DasonTio/mop-divpo-coauthor"
PERSONA_IDS = ["contrarian", "systems_thinker", "cross_domain_analogist", "minimalist"]

VALID_STAGES = {"base", "sft", "divpo", "divpo_v2", "single"}

COUNTER_ARGUMENT_TEMPLATE = "Generate a counter-argument to this claim:\n\n{prompt}"


def configure_warning_filters() -> None:
    """Silence the benign nested-PEFT warnings emitted when stacking adapters."""
    warnings.filterwarnings(
        "ignore",
        message=r"Already found a `peft_config` attribute in the model\.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are trying to modify a model with PEFT for a second time\.",
        category=UserWarning,
    )


def _is_custom_stage(stage: str) -> bool:
    return stage.startswith("sft_") or stage.startswith("divpo_")


def adapter_chain(
    stage: str,
    persona: str | None,
    single_name: str = "all",
    sft_stage: str = "sft",
) -> list[str]:
    """Return the ordered list of repo subfolders to stack for a method.

    Pure function — no model loading — so it is unit-testable.
    """
    if stage not in VALID_STAGES and not _is_custom_stage(stage):
        raise ValueError(
            f"Unknown stage {stage!r}. Expected one of {sorted(VALID_STAGES)} "
            "or a custom stage starting with 'sft_' or 'divpo_'."
        )
    if stage == "base":
        return []
    if stage == "single":
        return [f"single/{single_name}"]
    if persona is None:
        raise ValueError(f"stage={stage!r} requires a persona.")
    if stage == "sft" or stage.startswith("sft_"):
        return [f"{stage}/{persona}"]
    # divpo variants: SFT adapter first, DivPO adapter stacked on top.
    return [f"{sft_stage}/{persona}", f"{stage}/{persona}"]


def build_messages(
    prompt: str,
    *,
    system_prompt: str | None = None,
    as_counter_argument: bool = False,
) -> list[dict[str, str]]:
    """Build the chat message list.

    system_prompt=None -> no system turn (Base method).
    as_counter_argument -> wrap the prompt in the counter-argument instruction
    (Phase 3 eval domain). Otherwise the prompt is sent verbatim (Phase 1
    distinctness, where prompts are open questions).
    """
    user_content = (
        COUNTER_ARGUMENT_TEMPLATE.format(prompt=prompt) if as_counter_argument else prompt
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _persona_system_prompt(persona: str | None) -> str | None:
    if persona is None:
        return None
    from personas import PERSONAS

    return PERSONAS[persona]["system_prompt"]


class MoPGenerator:
    """Loads and caches persona models, then samples outputs.

    Typical use:
        gen = MoPGenerator(adapter_stage="divpo")
        outs = gen.generate("What is the future of remote work?", persona="contrarian", n=5)
        gen.unload()
    """

    def __init__(
        self,
        *,
        adapter_prefix: str = MODEL_REPO,
        adapter_stage: str = "sft",
        single_name: str = "all",
        sft_stage: str = "sft",
        base_model: str = BASE_MODEL,
        token: str | None = None,
        use_persona_prompt: bool = True,
    ) -> None:
        if adapter_stage not in VALID_STAGES and not _is_custom_stage(adapter_stage):
            raise ValueError(f"Unknown adapter_stage {adapter_stage!r}.")
        self.adapter_prefix = adapter_prefix
        self.adapter_stage = adapter_stage
        self.single_name = single_name
        self.sft_stage = sft_stage
        self.base_model = base_model
        self.token = token
        self.use_persona_prompt = use_persona_prompt
        self._tokenizer = None
        self._models: dict[str | None, object] = {}
        configure_warning_filters()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.base_model, token=self.token)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tokenizer = tok
        return self._tokenizer

    def _load_model(self, persona: str | None):
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
            token=self.token,
        )
        chain = adapter_chain(
            self.adapter_stage,
            persona,
            single_name=self.single_name,
            sft_stage=self.sft_stage,
        )
        if chain:
            from peft import PeftModel

            for subfolder in chain:
                model = PeftModel.from_pretrained(
                    model, self.adapter_prefix, subfolder=subfolder, token=self.token
                )
        model.eval()
        return model

    def _get_model(self, persona: str | None):
        # Base / single methods do not vary by persona, so they share one cache key.
        key = None if self.adapter_stage in {"base", "single"} else persona
        if key not in self._models:
            self._models[key] = self._load_model(persona)
        return self._models[key]

    def generate(
        self,
        prompt: str,
        *,
        persona: str | None = None,
        n: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        as_counter_argument: bool = False,
    ) -> list[str]:
        """Sample `n` outputs for one (prompt, persona). Returns decoded strings."""
        import torch

        tokenizer = self._get_tokenizer()
        model = self._get_model(persona)

        system_prompt = _persona_system_prompt(persona) if self.use_persona_prompt else None
        messages = build_messages(
            prompt, system_prompt=system_prompt, as_counter_argument=as_counter_argument
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in output
        ]

    def unload(self) -> None:
        """Free all cached models and clear the CUDA cache."""
        import torch

        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate(
    prompt: str,
    personas: list[str] | None = None,
    *,
    adapter_prefix: str = MODEL_REPO,
    adapter_stage: str = "sft",
    use_persona_prompt: bool = True,
    n: int = 1,
    temperature: float = 0.9,
    max_new_tokens: int = 256,
    token: str | None = None,
    as_counter_argument: bool = False,
) -> dict[str | None, list[str]]:
    """Convenience one-shot generation. Returns {persona: [outputs]}.

    Matches docs/research-plan.md §10:
        generate(prompt='test', personas=['contrarian'],
                 adapter_prefix='DasonTio/mop-divpo-coauthor', adapter_stage='sft')
    """
    gen = MoPGenerator(
        adapter_prefix=adapter_prefix,
        adapter_stage=adapter_stage,
        base_model=BASE_MODEL,
        token=token,
        use_persona_prompt=use_persona_prompt,
    )
    targets = personas if personas else ([None] if adapter_stage == "base" else PERSONA_IDS)
    try:
        return {
            persona: gen.generate(
                prompt,
                persona=persona,
                n=n,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                as_counter_argument=as_counter_argument,
            )
            for persona in targets
        }
    finally:
        gen.unload()
