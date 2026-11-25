"""Terminal front end: draws the same frames with 24-bit ANSI colors."""

from __future__ import annotations

import os
import shutil
import sys
import time

from .scene import Scene


def _enable_ansi() -> None:
    if os.name == "nt":
        os.system("")  # turns on VT escape sequence processing in the Windows console


def _encode(frame) -> str:
    out = ["\x1b[H"]
    last = None
    for r in range(frame.rows):
        chars = frame.chars[r]
        colors = frame.rgb[r].tolist()
        line = []
        for c in range(frame.cols):
            ch = chars[c]
            if ch == " ":
                line.append(" ")
                continue
            color = colors[c]
            if color != last:
                line.append(f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m")
                last = color
            line.append(ch)
        out.append("".join(line))
        out.append("\n" if r < frame.rows - 1 else "")
    return "".join(out)


def run(scene: Scene, fps: int, duration: float | None) -> None:
    _enable_ansi()
    sys.stdout.write("\x1b[?25l\x1b[2J")  # hide cursor, clear screen
    start = last = time.perf_counter()
    try:
        while duration is None or last - start < duration:
            now = time.perf_counter()
            scene.step(min(now - last, 0.1))
            last = now
            size = shutil.get_terminal_size((100, 40))
            frame = scene.frame(size.columns, size.lines - 1, 2.0)
            if scene.reveal > 0 and scene.lettering.text:
                text = scene.lettering.text
                shown = text[: int(len(text) * scene.reveal)]
                frame.put_text(frame.rows - 2, shown, [(235, 205, 150)] * len(shown))
            sys.stdout.write(_encode(frame))
            sys.stdout.flush()
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")
        sys.stdout.flush()
