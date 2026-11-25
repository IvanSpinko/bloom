"""A small line of text under the bouquet, set in a thin font and lit like gold.

Unlike the bouquet, the words are not built from ASCII cells: thin strokes do
not survive a character grid. They are drawn as smooth anti-aliased text with
wide letter spacing, filled with the palette's gold, a soft halo behind them
and a glint that slides across every few seconds.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from . import palettes  # noqa: E402


# Thin fonts, best first: Windows font files, then system font names (macOS, Linux)
FONT_FILES = ("segoeuil.ttf", "calibril.ttf", "segoeuisl.ttf")
FONT_NAMES = "helveticaneuelight,avenirnextultralight,avenirlight,sourcesanspro-light,dejavusans-extralight,arial"


def _thin_font(size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for name in FONT_FILES:
        path = os.path.join(fonts_dir, name)
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(FONT_NAMES, size)


class Lettering:
    def __init__(self, text: str, size: float = 0.026, center: float = 0.9, tracking: float = 0.22):
        self.text = text
        self.size = size          # font size as a fraction of the screen height
        self.center = center      # vertical position, fraction of the screen height
        self.tracking = tracking  # extra space between letters, in font sizes
        self.glint_period = 8.0
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _masks(self, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Alpha masks (w, h) of the letters and of their blurred halo."""
        if height not in self._cache:
            size = max(10, round(height * self.size))
            font = _thin_font(size)
            glyphs = [font.render(ch, True, (255, 255, 255)) for ch in self.text]
            gap = round(size * self.tracking)
            pad = size  # room for the halo
            width = sum(g.get_width() for g in glyphs) + gap * (len(glyphs) - 1) + 2 * pad
            line = pygame.Surface((width, font.get_height() + 2 * pad), pygame.SRCALPHA)
            x = pad
            for g in glyphs:
                line.blit(g, (x, pad))
                x += g.get_width() + gap
            alpha = pygame.surfarray.array_alpha(line).astype(np.float32) / 255.0
            w, h = line.get_size()
            k = 6
            small = pygame.transform.smoothscale(line, (max(1, w // k), max(1, h // k)))
            halo = pygame.surfarray.array_alpha(pygame.transform.smoothscale(small, (w, h))).astype(np.float32)
            self._cache[height] = (alpha, halo / 255.0)
        return self._cache[height]

    def draw(self, surface: pygame.Surface, palette: str, time: float, reveal: float) -> None:
        """Draw onto the surface; `reveal` goes 0..1 and uncovers the words left to right."""
        if not self.text or reveal <= 0:
            return
        sw, sh = surface.get_size()
        alpha, halo = self._masks(sh)
        w, h = alpha.shape

        x = np.linspace(0.0, 1.0, w, dtype=np.float32)
        # gold that drifts slowly along the words
        rgb = palettes.sample(palette, time * 0.04 + x * 0.5).astype(np.float32)
        rgb = rgb * 0.75 + np.array([255, 245, 225], np.float32) * 0.25
        glint_pos = (time % self.glint_period) / self.glint_period * 1.8 - 0.4
        glint = np.exp(-((x - glint_pos) * 7.0) ** 2)
        rgb += (255.0 - rgb) * (0.7 * glint)[:, None]

        fade = np.clip((reveal * 1.4 - x) / 0.4, 0.0, 1.0)  # appears from left to right
        a = (alpha * 0.95 + halo * 0.35) * fade[:, None]
        if a.max() <= 0:
            return

        x0 = (sw - w) // 2
        y0 = int(sh * self.center - h / 2)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(sw, x0 + w), min(sh, y0 + h)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        a = np.clip(a[sx0 - x0: sx1 - x0, sy0 - y0: sy1 - y0, None], 0.0, 1.0)
        col = rgb[sx0 - x0: sx1 - x0, None, :]

        target = pygame.surfarray.pixels3d(surface)
        region = target[sx0:sx1, sy0:sy1].astype(np.float32)
        target[sx0:sx1, sy0:sy1] = (region * (1.0 - a) + col * a).astype(np.uint8)
        del target  # unlock the surface
