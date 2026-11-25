"""Drawing ASCII frames onto pygame surfaces (shared by the window and the exporter)."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .raster import Frame  # noqa: E402


FONT_NAMES = "consolas,cascadiamono,menlo,dejavusansmono,liberationmono,couriernew"
BG = (7, 6, 5)


def load_font(size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.SysFont(FONT_NAMES, size)


def cell_size(font: pygame.font.Font) -> tuple[int, int]:
    return font.size("@")[0], font.get_linesize()


class GlyphCache:
    """Pre-rendered glyph surfaces keyed by character and quantized color."""

    def __init__(self, font: pygame.font.Font, backdrop: float = 0.36):
        self.font = font
        self.backdrop = backdrop  # strength of the soft glow under the glyphs
        self.cw, self.ch = cell_size(font)
        self.cache: dict[int, pygame.Surface] = {}

    def get(self, key: int) -> pygame.Surface:
        surf = self.cache.get(key)
        if surf is None:
            ch = chr(key >> 15)
            r, g, b = ((key >> 10) & 31) * 255 // 31, ((key >> 5) & 31) * 255 // 31, (key & 31) * 255 // 31
            surf = self.font.render(ch, True, (r, g, b))
            if len(self.cache) > 60000:
                self.cache.clear()
            self.cache[key] = surf
        return surf

    def grid_for(self, width: int, height: int) -> tuple[int, int, float]:
        """Columns, rows and cell aspect ratio that fit a surface of this size."""
        return max(10, width // self.cw), max(5, height // self.ch), self.ch / self.cw

    def draw(self, surface: pygame.Surface, frame: Frame) -> None:
        """Fill the surface with the background and draw the frame centered on it.

        Under the glyphs goes a soft glow in the color of each cell: thin
        characters alone cover too little of a cell for shapes to read well.
        """
        surface.fill(BG)
        w, h = surface.get_size()
        ox, oy = (w - frame.cols * self.cw) // 2, (h - frame.rows * self.ch) // 2
        if self.backdrop > 0:
            glow = (frame.rgb.astype(np.float32) * self.backdrop).astype(np.uint8)
            small = pygame.surfarray.make_surface(np.ascontiguousarray(glow.transpose(1, 0, 2)))
            big = pygame.transform.smoothscale(small, (frame.cols * self.cw, frame.rows * self.ch))
            surface.blit(big, (ox, oy), special_flags=pygame.BLEND_RGB_ADD)
        ys, xs = np.nonzero(frame.chars != " ")
        if not len(ys):
            return
        codes = np.frombuffer(frame.chars[ys, xs].astype("<U1").tobytes(), dtype=np.uint32)
        q = frame.rgb[ys, xs].astype(np.uint32) >> 3
        keys = (codes << 15) | (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
        px = (xs * self.cw + ox).tolist()
        py = (ys * self.ch + oy).tolist()
        get = self.get
        surface.fblits([(get(k), (x, y)) for k, x, y in zip(keys.tolist(), px, py)])
