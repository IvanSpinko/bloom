"""Ties the bouquet, its surroundings, the lettering and the camera together."""

from __future__ import annotations

import math
import zlib

import numpy as np

from . import palettes
from .bouquet import BOTTOM_Z, Bouquet
from .lettering import Lettering
from .particles import Life
from .raster import Camera, Frame, render


DEFAULT_SEED = "starlight"
MESSAGE = "capable & loved"


class Scene:
    def __init__(
        self,
        seed: str = DEFAULT_SEED,
        palette: str = "starlight",
        speed: float = 1.0,
        detail: float = 1.0,
    ):
        self.palette = palette
        self.speed = speed
        self.detail = detail
        self.lettering = Lettering(MESSAGE)
        self.camera = Camera(pitch=0.3, zoom=1.12, target_z=-0.6, center=0.45)
        # the bouquet slowly turns a little to one side and back
        self.sway_amp = 0.38
        self.sway_period = 30.0
        self.yaw_offset = 0.0
        self.shimmer_speed = 0.05
        self.breathe_freq = 1.3
        self.time = 0.0
        self.plant(seed)

    # ------------------------------------------------------------ lifecycle
    def plant(self, seed: str) -> None:
        """Build a new bouquet from a seed and let it grow from the start."""
        self.seed = seed
        self.bouquet = Bouquet(seed, self.detail)
        self.life = Life(
            ground=BOTTOM_Z - 0.3,
            seed=zlib.crc32(seed.encode()),
            sources=self.bouquet.sources(),
            sparkles=18,
        )
        self.tau = 0.0
        self.gust = 0.0

    def replay(self) -> None:
        self.plant(self.seed)

    def set_loop(self, period: float) -> None:
        """Make every slow motion (sway, color drift, breathing) repeat exactly every `period` s."""
        self.sway_period = period
        self.shimmer_speed = max(1, round(self.shimmer_speed * period)) / period
        self.breathe_freq = max(1, round(self.breathe_freq * period / math.tau)) * math.tau / period
        self.lettering.glint_period = period / max(1, round(period / 8.0))

    def next_palette(self) -> str:
        names = list(palettes.PALETTES)
        self.palette = names[(names.index(self.palette) + 1) % len(names)]
        return self.palette

    def blow(self) -> None:
        """A gust of wind: petals ripple and pollen flies off."""
        self.gust = 1.0
        if self.bloom > 0.5:
            self.life.burst()

    # ---------------------------------------------------------------- state
    @property
    def settled_at(self) -> float:
        return self.bouquet.settled_at

    @property
    def bloom(self) -> float:
        """Overall progress of the bouquet, 0 = bare stems, 1 = every rose open."""
        return float(np.clip((self.tau - 3.0) / (self.settled_at - 3.0), 0, 1))

    def step(self, dt: float) -> None:
        dt *= self.speed
        self.time += dt
        self.tau += dt
        self.gust *= math.exp(-1.2 * dt)
        self.camera.yaw = self.yaw_offset + self.sway_amp * math.sin(math.tau * self.time / self.sway_period)
        self.life.update(dt, self.bloom, self.time)

    def frame(self, cols: int, rows: int, cell_aspect: float) -> Frame:
        cloud = self.bouquet.build(self.tau, self.gust, self.time, self.breathe_freq)
        sprites = self.life.sprites(self.time, self.palette, self.bloom)
        frame = render(
            cloud, sprites, self.camera, cols, rows, cell_aspect,
            self.palette, self.time, self.shimmer_speed,
        )
        return frame

    @property
    def reveal(self) -> float:
        """0..1: how far the words under the bouquet have appeared."""
        return float(np.clip((self.tau - (self.settled_at - 3.0)) / 3.0, 0, 1))

    def draw_overlay(self, surface, time: float | None = None, reveal: float | None = None) -> None:
        """Draw the words onto a finished pygame surface."""
        self.lettering.draw(
            surface, self.palette, self.time if time is None else time, self.reveal if reveal is None else reveal
        )
