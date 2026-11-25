"""Gold glitter twinkling in the dark background behind the bouquet.

Each speck drifts very slowly and flashes now and then: most of the time it is
a faint dot, and at the peak of its twinkle it opens into a small four-pointed
star. Specks that fall behind the bouquet or the words are hidden.
"""

from __future__ import annotations

import numpy as np

from .draw import pygame
from .raster import Frame


LEVELS = 12  # brightness steps of the pre-rendered sprites
COLORS = np.array([(255, 236, 190), (250, 214, 150), (255, 246, 225), (240, 200, 150)], np.float32)


def _star(radius: int, color: np.ndarray, brightness: float) -> pygame.Surface:
    """A soft dot with four thin rays, for additive blending."""
    size = 2 * radius + 1
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1].astype(np.float32)
    r = np.hypot(x, y) / max(radius, 1)
    core = np.exp(-(r * 3.2) ** 2)
    rays = (np.exp(-(x * 1.6) ** 2) + np.exp(-(y * 1.6) ** 2)) * np.clip(1.0 - r, 0, 1) ** 1.5
    light = np.clip(core + 0.55 * rays, 0, 1) * brightness
    rgb = (light[..., None] * color[None, None, :]).astype(np.uint8)
    surf = pygame.Surface((size, size))
    pygame.surfarray.blit_array(surf, np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    return surf


class Glitter:
    def __init__(self, density: float = 1.5, seed: int = 7):
        self.density = density
        self.rng = np.random.default_rng(seed)
        self.size: tuple[int, int] | None = None
        self._sprites: dict[tuple[int, int, int], pygame.Surface] = {}

    def _scatter(self, w: int, h: int) -> None:
        n = int(w * h / 9000 * self.density)
        rng = self.rng
        self.size = (w, h)
        self.pos = rng.uniform(0, 1, (n, 2)) * [w, h]
        self.vel = rng.normal(0, 1, (n, 2)) * [2.0, 1.5] + [0.0, -2.5]  # px/s, a slow upward drift
        self.freq = rng.uniform(0.3, 1.1, n)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.sharp = rng.uniform(4.0, 12.0, n)       # how short the flash is
        self.radius = rng.choice([3, 5, 7, 9], n, p=[0.5, 0.28, 0.15, 0.07])
        self.color = rng.integers(0, len(COLORS), n)
        self.base = rng.uniform(0.04, 0.16, n)       # faint glow between flashes

    def _sprite(self, radius: int, color: int, level: int) -> pygame.Surface:
        key = (radius, color, level)
        if key not in self._sprites:
            self._sprites[key] = _star(radius, COLORS[color], level / (LEVELS - 1))
        return self._sprites[key]

    def draw(self, surface: pygame.Surface, time: float, dt: float, frame: Frame, cell: tuple[int, int],
             origin: tuple[int, int]) -> None:
        w, h = surface.get_size()
        if self.size != (w, h):
            self._scatter(w, h)
        if not len(self.pos):
            return

        self.pos += self.vel * dt
        self.pos %= [w, h]  # wrap around the screen edges

        twinkle = np.maximum(0.0, np.sin(time * self.freq * 2 * np.pi / 3 + self.phase)) ** self.sharp
        bright = self.base + (1.0 - self.base) * twinkle
        level = np.clip((bright * (LEVELS - 1)).round().astype(int), 0, LEVELS - 1)

        # hide specks behind the bouquet: skip the ones whose cell holds a glyph
        cw, ch = cell
        col = ((self.pos[:, 0] - origin[0]) // cw).astype(int)
        row = ((self.pos[:, 1] - origin[1]) // ch).astype(int)
        inside = (col >= 0) & (col < frame.cols) & (row >= 0) & (row < frame.rows)
        covered = np.zeros(len(self.pos), bool)
        covered[inside] = frame.chars[row[inside], col[inside]] != " "
        show = ~covered & (level > 0)
        # big rays only during a flash, so the background stays calm
        radius = np.where(twinkle > 0.35, self.radius, 2)

        blits = []
        for i in np.nonzero(show)[0]:
            r = int(radius[i])
            blits.append((self._sprite(r, int(self.color[i]), int(level[i])),
                          (int(self.pos[i, 0]) - r, int(self.pos[i, 1]) - r)))
        surface.fblits(blits, pygame.BLEND_RGB_ADD)
