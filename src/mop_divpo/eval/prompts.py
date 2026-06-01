"""Evaluation prompt sets (docs/research-plan.md §1.2 and §3.4).

PERSONA_DISTINCTNESS_PROMPTS — 20 persona-neutral open questions for Phase 1.
No persona-trigger keywords (no "challenge", "system", "feedback", "minimal",
"analogy"), topic-diverse so a one-topic persona is exposed.

EVALUATION_PROMPTS — the 20 distinctness prompts plus 10 hand-curated
CGA-CMV-style claim titles (written as Reddit r/ChangeMyView thread titles,
not drawn from the training set) for Phase 3.
"""
from __future__ import annotations

PERSONA_DISTINCTNESS_PROMPTS = [
    "What is the future of remote work?",
    "How should cities adapt to climate change?",
    "What's wrong with current education systems?",
    "Why do startups fail in the first three years?",
    "How can governments handle aging populations?",
    "What makes a city worth living in?",
    "Why are mental health rates worsening globally?",
    "What's the right way to evaluate AI safety?",
    "How should universities change in the next decade?",
    "Why is public transit underfunded?",
    "What does meaningful work look like in 2035?",
    "Why is housing unaffordable in major cities?",
    "How should social media platforms be regulated?",
    "Why do diets fail for most people?",
    "What is the role of religion in modern society?",
    "How do we fix scientific peer review?",
    "Why is creative writing declining in schools?",
    "What's wrong with how we measure economic growth?",
    "How should we think about generational wealth?",
    "Why is loneliness rising in connected societies?",
]

# 10 CMV-style claim titles (held out of training), phrased as assertions so the
# counter-argument task is well-posed.
CMV_STYLE_PROMPTS = [
    "CMV: Standardized testing should be abolished entirely.",
    "CMV: Working from home makes employees less productive.",
    "CMV: Social media has done more harm than good for society.",
    "CMV: Universal basic income would discourage people from working.",
    "CMV: Nuclear energy is the only realistic path to decarbonization.",
    "CMV: College degrees are no longer worth the cost for most people.",
    "CMV: Voting should be mandatory for all eligible citizens.",
    "CMV: Cancel culture has gone too far and chills free expression.",
    "CMV: Self-driving cars will never be safe enough to fully trust.",
    "CMV: Tipping culture should be replaced with higher fixed wages.",
]

EVALUATION_PROMPTS = PERSONA_DISTINCTNESS_PROMPTS + CMV_STYLE_PROMPTS

assert len(PERSONA_DISTINCTNESS_PROMPTS) == 20
assert len(EVALUATION_PROMPTS) == 30
