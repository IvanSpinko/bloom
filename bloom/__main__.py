"""Command line entry point: ``python -m bloom``."""

from __future__ import annotations

import argparse
import sys

from .palettes import PALETTES
from .scene import DEFAULT_SEED, Scene


def _size(text: str) -> tuple[int, int]:
    width, height = (int(v) for v in text.lower().split("x"))
    return width, height


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bloom",
        description="A golden ASCII bouquet that grows, blooms and shimmers.",
    )
    parser.add_argument("seed", nargs="?", default=DEFAULT_SEED,
                        help="any word or phrase; the same seed always arranges the same bouquet")
    parser.add_argument("-p", "--palette", choices=list(PALETTES), default="starlight")
    parser.add_argument("-t", "--terminal", action="store_true", help="render in the terminal instead of a window")
    parser.add_argument("-w", "--windowed", action="store_true", help="open a window instead of full screen")
    parser.add_argument("--size", default="1280x800", help="window size in windowed mode, WIDTHxHEIGHT")
    parser.add_argument("--font-size", type=int, help="smaller font = more detail (default: follows screen height)")
    parser.add_argument("--detail", type=float, help="geometry density (default: 0.6 window, 0.4 terminal)")
    parser.add_argument("--speed", type=float, default=1.0, help="animation speed multiplier")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float, help="terminal mode: stop after N seconds")
    # Unknown arguments (e.g. from an outdated run configuration) are ignored, not fatal
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"bloom: ignoring unknown arguments: {' '.join(unknown)}", file=sys.stderr)

    detail = args.detail or (0.4 if args.terminal else 0.6)
    scene = Scene(args.seed, args.palette, args.speed, detail)

    if args.terminal:
        from . import terminal

        terminal.run(scene, min(args.fps, 30), args.duration)
    else:
        from . import window

        width, height = _size(args.size)
        window.run(scene, width, height, args.font_size, args.fps, fullscreen=not args.windowed)


if __name__ == "__main__":
    main()
