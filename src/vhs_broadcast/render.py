"""Headless re-render of a recorded take. No GUI required.

    uv run vhs-render --session take.json --output out.mp4
    uv run vhs-render --input clip.mp4 --session take.json -o out.mp4 --full
"""
from __future__ import annotations

import argparse
import sys

from .config import PREVIEW_MAX_WIDTH
from .core.automation import Session, Timeline
from .core.parameters import ParameterSet
from .core.pipeline import render_to_file
from .effects.vhs import default_effects


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Re-render a VHS Broadcast take.")
    ap.add_argument("--session", "-s", required=True, help="take .json")
    ap.add_argument("--input", "-i", help="override source video path")
    ap.add_argument("--output", "-o", required=True, help="output .mp4")
    ap.add_argument("--full", action="store_true",
                    help="render at full resolution (default: preview width)")
    args = ap.parse_args(argv)

    params = ParameterSet()
    effects = default_effects(params)
    timeline = Timeline()
    session = Session(params, timeline, effects)
    session.load(args.session)
    if session.order:
        by_id = {e.id: e for e in effects}
        ordered = list(dict.fromkeys(session.order))
        effects = [by_id[k] for k in ordered if k in by_id] + [
            e for e in effects if e.id not in ordered]

    source = args.input or session.source_path
    if not source:
        print("error: no source video (use --input)", file=sys.stderr)
        return 2

    max_w = None if args.full else PREVIEW_MAX_WIDTH

    def prog(f):
        sys.stdout.write(f"\rrendering {f*100:5.1f}%")
        sys.stdout.flush()

    render_to_file(source, args.output, params, timeline, effects,
                   max_width=max_w, progress=prog)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
