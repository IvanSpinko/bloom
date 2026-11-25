"""Life around the flowers: rising pollen, falling petals and drifting sparkles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import palettes


@dataclass
class Sprites:
    """Points that are drawn with a fixed character and color."""

    pos: np.ndarray    # (N, 3)
    chars: np.ndarray  # (N,) '<U1'
    rgb: np.ndarray    # (N, 3) float

    @staticmethod
    def empty() -> "Sprites":
        return Sprites(np.zeros((0, 3), np.float32), np.zeros(0, "<U1"), np.zeros((0, 3), np.float32))


POLLEN_CHARS = np.array(list(".'`*+"))
PETAL_CHARS = np.array(list("~=-~o"))


class Life:
    def __init__(self, ground: float, seed: int = 0, sources: np.ndarray | None = None, sparkles: int = 7):
        self.rng = np.random.default_rng(seed)
        self.ground = ground
        # points pollen rises from (the hearts of the flowers)
        self.sources = np.array([[0.0, 0.0, 0.55]]) if sources is None else np.asarray(sources, float)
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.kind = np.zeros(0, np.uint8)  # 0 pollen, 1 petal
        self.phase = np.zeros(0)
        self._pollen_debt = 0.0
        self._petal_debt = 0.0
        self.fireflies = self.rng.uniform(0, 1, (sparkles, 4))

    def burst(self, n: int = 60) -> None:
        """A puff of pollen, e.g. when the user clicks the flower."""
        self._spawn_pollen(n, speed=1.8)

    def _spawn(self, pos, vel, life, kind):
        n = len(pos)
        self.pos = np.concatenate([self.pos, pos])
        self.vel = np.concatenate([self.vel, vel])
        self.age = np.concatenate([self.age, np.zeros(n)])
        self.life = np.concatenate([self.life, life])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.uint8)])
        self.phase = np.concatenate([self.phase, self.rng.uniform(0, 6.3, n)])

    def _spawn_pollen(self, n: int, speed: float = 1.0) -> None:
        a = self.rng.uniform(0, 2 * np.pi, n)
        r = self.rng.uniform(0, 0.08, n)
        origin = self.sources[self.rng.integers(0, len(self.sources), n)]
        pos = origin + np.stack([r * np.cos(a), r * np.sin(a), self.rng.uniform(0, 0.1, n)], 1)
        out = self.rng.uniform(0.05, 0.25, n) * speed
        vel = np.stack([out * np.cos(a), out * np.sin(a), self.rng.uniform(0.15, 0.4, n) * speed], 1)
        self._spawn(pos, vel, self.rng.uniform(2.5, 5.0, n), 0)

    def _spawn_petal(self) -> None:
        a = self.rng.uniform(0, 2 * np.pi)
        r = self.rng.uniform(0.1, 0.25)
        origin = self.sources[self.rng.integers(0, len(self.sources))]
        pos = (origin + np.array([r * np.cos(a), r * np.sin(a), 0.0]))[None, :]
        vel = np.array([[0.1 * np.cos(a), 0.1 * np.sin(a), -0.16]])
        self._spawn(pos, vel, np.array([30.0]), 1)

    def update(self, dt: float, bloom: float, time: float) -> None:
        if bloom > 0.85:
            self._pollen_debt += dt * 7.0
            self._petal_debt += dt * 0.35
            n = int(self._pollen_debt)
            if n:
                self._spawn_pollen(n)
                self._pollen_debt -= n
            while self._petal_debt >= 1.0:
                self._spawn_petal()
                self._petal_debt -= 1.0

        if len(self.pos):
            sway = np.stack(
                [np.sin(time * 1.7 + self.phase), np.cos(time * 1.3 + self.phase), np.zeros_like(self.phase)], 1
            )
            amp = np.where(self.kind == 1, 0.35, 0.08)[:, None]
            self.pos = self.pos + (self.vel + sway * amp) * dt
            self.vel[:, 2] -= np.where(self.kind == 0, 0.05, 0.0) * dt  # pollen slowly sinks back
            self.age += dt
            alive = (self.age < self.life) & (self.pos[:, 2] > self.ground)
            self.pos, self.vel = self.pos[alive], self.vel[alive]
            self.age, self.life = self.age[alive], self.life[alive]
            self.kind, self.phase = self.kind[alive], self.phase[alive]

    def sprites(self, time: float, palette: str, bloom: float) -> Sprites:
        parts = []
        if len(self.pos):
            fade = np.clip(np.minimum(self.age * 3.0, (self.life - self.age) * 1.5), 0, 1)
            twinkle = 0.55 + 0.45 * np.sin(time * 9.0 + self.phase * 3.0)
            pollen_rgb = np.array([255, 214, 120]) * (fade * twinkle)[:, None]
            petal_rgb = palettes.sample(palette, self.phase / 6.3 + time * 0.1) * fade[:, None]
            is_petal = (self.kind == 1)[:, None]
            rgb = np.where(is_petal, petal_rgb, pollen_rgb)
            flutter = (time * 6.0 + self.phase * 5).astype(int)
            chars = np.where(
                self.kind == 1,
                PETAL_CHARS[flutter % len(PETAL_CHARS)],
                POLLEN_CHARS[(flutter // 2) % len(POLLEN_CHARS)],
            )
            parts.append(Sprites(self.pos.astype(np.float32), chars, rgb.astype(np.float32)))

        if bloom > 0.6:
            f = self.fireflies
            w = time * (0.25 + 0.2 * f[:, 3])
            pos = np.stack(
                [
                    1.6 * np.sin(w + f[:, 0] * 6.3) * (0.7 + 0.3 * f[:, 1]),
                    1.2 * np.cos(w * 0.8 + f[:, 1] * 6.3),
                    -0.35 + 1.0 * np.sin(w * 0.6 + f[:, 2] * 6.3),
                ],
                1,
            )
            blink = np.clip(np.sin(time * (1.5 + f[:, 3]) + f[:, 0] * 9) * 1.6, 0, 1)
            blink *= np.clip((bloom - 0.6) / 0.4, 0, 1)
            rgb = np.array([255, 224, 155]) * blink[:, None]
            keep = blink > 0.08
            parts.append(Sprites(pos[keep].astype(np.float32), np.full(keep.sum(), "*"), rgb[keep].astype(np.float32)))

        if not parts:
            return Sprites.empty()
        return Sprites(
            np.concatenate([p.pos for p in parts]),
            np.concatenate([p.chars for p in parts]),
            np.concatenate([p.rgb for p in parts]),
        )
