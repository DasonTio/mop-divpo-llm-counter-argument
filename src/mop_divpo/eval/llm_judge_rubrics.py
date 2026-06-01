"""The four LLM-as-judge rubric prompts (docs/research-plan.md §2.2).

Each rubric is a self-contained template plus metadata describing which JSON
keys it returns and which of those keys carry the 1-5 score(s) used for
aggregation in the baseline table.

A shared instruction tells the judge to ignore output length — a documented
bias mitigation (plan §2.6).
"""
from __future__ import annotations

from typing import Any

_IGNORE_LENGTH = (
    "Judge only the substance of the argument. Do not reward or penalise an "
    "output for being longer or shorter than the others.\n\n"
)

QUALITY_PROMPT = (
    "You are an expert evaluator scoring the quality of a counter-argument "
    "generated for a writer who is brainstorming.\n\n"
    + _IGNORE_LENGTH
    + "Original topic: {prompt}\n"
    "Generated counter-argument: {output}\n\n"
    "Score on three sub-criteria, each 1-5 (1 = poor, 5 = excellent):\n\n"
    "1. RELEVANCE - does the output address the topic?\n"
    "2. COHERENCE - is the argument internally consistent and well-formed?\n"
    "3. SUBSTANCE - does the output offer a real point, not platitudes?\n\n"
    "Return JSON only:\n"
    '{{"relevance": int, "coherence": int, "substance": int, "rationale": "one sentence"}}'
)

PERSONA_FIDELITY_PROMPT = (
    "You are an expert evaluator checking whether a generated response "
    "demonstrates a specific cognitive style.\n\n"
    + _IGNORE_LENGTH
    + "Cognitive style: {persona_description}\n"
    "Generated response: {output}\n\n"
    "Score 1-5 on how strongly the response exhibits the cognitive style:\n"
    "1 = does not exhibit the style at all\n"
    "3 = somewhat exhibits the style\n"
    "5 = clearly and strongly exhibits the style\n\n"
    "Return JSON only:\n"
    '{{"persona_fidelity": int, "evidence": "quote one phrase from the output '
    "that demonstrates the style, or 'none'\"}}"
)

NOVELTY_PROMPT = (
    "You are an expert evaluator measuring how novel a counter-argument is "
    "compared to other counter-arguments on the same topic.\n\n"
    + _IGNORE_LENGTH
    + "Topic: {prompt}\n"
    "Target argument: {output}\n"
    "Other arguments on the same topic:\n"
    "{peers}\n\n"
    "Score the target argument 1-5 on novelty relative to the others:\n"
    "1 = essentially repeats one of the others\n"
    "3 = different surface words but similar conceptual point\n"
    "5 = makes a substantively different argument\n\n"
    "Return JSON only:\n"
    '{{"novelty": int, "most_similar_peer_index": int_or_null, "rationale": "one sentence"}}'
)

UTILITY_PROMPT = (
    "You are an expert writing coach evaluating whether a brainstormed "
    "counter-argument would be useful to a writer at the pre-writing stage.\n\n"
    + _IGNORE_LENGTH
    + "Writer's topic: {prompt}\n"
    "Generated counter-argument: {output}\n\n"
    "Score 1-5 on pre-writing usefulness:\n"
    "1 = the writer would discard this immediately\n"
    "3 = the writer might use this as a starting point\n"
    "5 = the writer would clearly benefit from following this direction\n\n"
    "Return JSON only:\n"
    '{{"prewriting_utility": int, "rationale": "one sentence"}}'
)


# Each rubric declares the score keys it produces (the 1-5 integers) and the
# template fields it consumes beyond {output}.
RUBRICS: dict[str, dict[str, Any]] = {
    "quality": {
        "template": QUALITY_PROMPT,
        "score_keys": ["relevance", "coherence", "substance"],
        "needs": ["prompt"],
    },
    "persona_fidelity": {
        "template": PERSONA_FIDELITY_PROMPT,
        "score_keys": ["persona_fidelity"],
        "needs": ["persona_description"],
    },
    "novelty": {
        "template": NOVELTY_PROMPT,
        "score_keys": ["novelty"],
        "needs": ["prompt", "peers"],
    },
    "utility": {
        "template": UTILITY_PROMPT,
        "score_keys": ["prewriting_utility"],
        "needs": ["prompt"],
    },
}


def _format_peers(peers: list[str]) -> str:
    if not peers:
        return "(no other arguments available)"
    return "\n".join(f"{i + 1}. {peer}" for i, peer in enumerate(peers))


def build_rubric_prompt(
    rubric_name: str,
    *,
    output: str,
    prompt: str | None = None,
    persona_description: str | None = None,
    peers: list[str] | None = None,
) -> str:
    """Render a rubric template into the final judge prompt string.

    Raises KeyError for an unknown rubric and ValueError if a required field is
    missing, so a miswired call fails loudly instead of producing a garbled prompt.
    """
    if rubric_name not in RUBRICS:
        raise KeyError(f"Unknown rubric {rubric_name!r}. Known: {sorted(RUBRICS)}")
    spec = RUBRICS[rubric_name]
    fields: dict[str, str] = {"output": output}
    for need in spec["needs"]:
        if need == "prompt":
            if prompt is None:
                raise ValueError(f"Rubric {rubric_name!r} needs `prompt`.")
            fields["prompt"] = prompt
        elif need == "persona_description":
            if persona_description is None:
                raise ValueError(f"Rubric {rubric_name!r} needs `persona_description`.")
            fields["persona_description"] = persona_description
        elif need == "peers":
            fields["peers"] = _format_peers(peers or [])
    return spec["template"].format(**fields)
