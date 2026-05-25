#!/usr/bin/env python
"""Prepare SFT training datasets for each MoP persona.

Usage:
    python scripts/prepare_sft_datasets.py --persona contrarian --limit 5000
    python scripts/prepare_sft_datasets.py --all --limit 5000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mop_divpo.data.prepare_sft import (
    prepare_contrarian,
    prepare_cross_domain_analogist,
    prepare_minimalist,
    prepare_systems_thinker,
)

PERSONA_FUNCS = {
    "contrarian": prepare_contrarian,
    "systems_thinker": prepare_systems_thinker,
    "cross_domain_analogist": prepare_cross_domain_analogist,
    "minimalist": prepare_minimalist,
}


def run_persona(persona: str, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    checks_dir = Path(args.checks_dir)
    examples_path = (
        Path(args.write_examples) if args.write_examples else checks_dir / "sft_examples.md"
    )

    print(f"\n=== {persona} ===")
    _, stats = PERSONA_FUNCS[persona](
        output_path=output_dir / f"{persona}.jsonl",
        limit=args.limit,
        stats_path=checks_dir / f"sft_summary_{persona}.json",
        examples_path=examples_path,
    )
    s = stats.to_dict()
    print(
        f"  kept={s['total_kept']}  filtered={s['total_filtered']}  "
        f"reasons={dict(s['filter_counts'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for MoP personas.")
    parser.add_argument("--persona", choices=list(PERSONA_FUNCS), help="Single persona.")
    parser.add_argument("--all", action="store_true", help="Prepare all personas.")
    parser.add_argument("--limit", type=int, default=5000, help="Max records per persona.")
    parser.add_argument("--output-dir", default="data/processed/sft")
    parser.add_argument("--checks-dir", default="outputs/data_checks")
    parser.add_argument("--write-examples", default=None, help="Custom path for examples .md file.")
    args = parser.parse_args()

    if not args.persona and not args.all:
        parser.error("Provide --persona <name> or --all.")

    personas = list(PERSONA_FUNCS) if args.all else [args.persona]
    for p in personas:
        run_persona(p, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
