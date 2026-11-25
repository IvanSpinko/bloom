"""3D geometry of a single rose head and of leaf-like sheets.

The rose head is a spiral surface: the parameter ``t`` winds around the
center (inner petals have large ``t``), ``x`` runs from the base of a petal
to its tip. The opening angle of every petal depends on the bloom progress,
so the same surface goes from a closed bud to a fully open flower.

Z is "up"; the base of the head sits at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import Genome


KIND_PETAL = 0
KIND_GREEN = 1

TAU = 2.0 * np.pi


@dataclass
class Cloud:
    """A point cloud with normals: everything the rasterizer needs."""

    pos: np.ndarray   # (N, 3)
    nrm: np.ndarray   # (N, 3), unit length
    kind: np.ndarray  # (N,) uint8, KIND_*
    tone: np.ndarray  # (N,) float, per-kind color parameter
    ao: np.ndarray    # (N,) float, ambient occlusion: 0 = deep shadow, 1 = open
    part: np.ndarray  # (N,) int32, id of the petal / leaf a point belongs to

    @staticmethod
    def concat(parts: list["Cloud"]) -> "Cloud":
        parts = [p for p in parts if len(p.pos)]
        if not parts:
            empty = np.zeros((0, 3), np.float32)
            none = np.zeros(0, np.float32)
            return Cloud(empty, empty, np.zeros(0, np.uint8), none, none, np.zeros(0, np.int32))
        return Cloud(
            np.concatenate([p.pos for p in parts]),
            np.concatenate([p.nrm for p in parts]),
            np.concatenate([p.kind for p in parts]),
            np.concatenate([p.tone for p in parts]),
            np.concatenate([p.ao for p in parts]),
            np.concatenate([p.part for p in parts]),
        )


@dataclass
class Growth:
    """How far the flower has grown at a moment of the animation."""

    stem: float     # 0..1, stem height
    head: float     # 0..1, size of the flower head
    bloom: float    # 0..1, how open the petals are
    breathe: float  # ~1, slow open/close pulse once fully bloomed

    @staticmethod
    def at(tau: float, breathe_freq: float = 1.3, phase: float = 0.0) -> "Growth":
        stem = _ease(np.clip(tau / 2.2, 0, 1))
        head = _ease(np.clip((tau - 1.7) / 1.4, 0, 1))
        bloom = _ease_in_out(np.clip((tau - 2.8) / 6.5, 0, 1))
        settle = np.clip(tau - 9.3, 0, 1)
        breathe = 1.0 + 0.04 * np.sin(breathe_freq * tau + phase) * settle
        return Growth(float(stem), float(head), float(bloom), float(breathe))

    @property
    def finished(self) -> bool:
        return self.bloom >= 0.999


def _ease(v):
    return 1.0 - (1.0 - v) ** 3


def _ease_in_out(v):
    return v * v * (3.0 - 2.0 * v)


def _surface_normals(px, py, pz):
    """Normals of a gridded surface via finite differences along both axes."""
    du = [np.gradient(a, axis=0) for a in (px, py, pz)]
    dv = [np.gradient(a, axis=1) for a in (px, py, pz)]
    nx = du[1] * dv[2] - du[2] * dv[1]
    ny = du[2] * dv[0] - du[0] * dv[2]
    nz = du[0] * dv[1] - du[1] * dv[0]
    n = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return n


def _pack(px, py, pz, kind, tone, ao=None, part=-1) -> Cloud:
    pos = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=1).astype(np.float32)
    nrm = _surface_normals(px, py, pz).astype(np.float32)
    ao = np.ones(len(pos), np.float32) if ao is None else ao.ravel().astype(np.float32)
    part = np.full(len(pos), part, np.int32) if np.isscalar(part) else part.ravel().astype(np.int32)
    return Cloud(pos, nrm, np.full(len(pos), kind, np.uint8), tone.ravel().astype(np.float32), ao, part)


class Flower:
    def __init__(self, genome: Genome, nx: int = 34, nt: int = 1800):
        self.g = genome
        t0 = 4.0 * np.pi
        t1 = t0 + genome.turns * TAU
        self.x, self.t = np.meshgrid(
            np.linspace(0.0, 1.0, nx, dtype=np.float64),
            np.linspace(t0, t1, nt, dtype=np.float64),
        )
        # 0 = outermost petal layer, 1 = innermost
        self.tn = (self.t - t0) / (t1 - t0)
        # Fully open angle of each petal measured from the vertical
        self.p_open = 0.9 * (np.pi / 2.0) * np.exp(-0.5 - 2.5 * self.tn)
        # Closed bud: a teardrop, petals wrapped around each other, tips meeting
        self.bud_r = (0.36 - 0.2 * self.tn) * np.sin(np.pi * self.x**0.8) * (1.0 - 0.15 * self.x)
        self.bud_z = (0.85 + 0.1 * self.tn) * self.x
        # Petal outline: u dips to 0.5 between neighbouring petals
        m = np.mod(genome.petals_per_turn * self.t, TAU)
        self.u_base = 1.0 - (1.0 - m / np.pi) ** 4 / 2.0
        self.petal_id = np.floor(genome.petals_per_turn * self.t / TAU).astype(np.int32)
        self.petal_shade = 0.82 + 0.18 * np.sin(self.petal_id * 2.39996) ** 2
        self.ruffle = genome.ruffle_amp * np.sin(genome.ruffle_freq * self.t)

    # ------------------------------------------------------------------ head
    def head(self, growth: Growth, gust: float, time: float) -> Cloud:
        x, t, tn = self.x, self.t, self.tn

        # Outer petals open first, inner ones follow.
        local = np.clip(growth.bloom * 1.35 - 0.35 * tn, 0.0, 1.0)
        local = local * local * (3.0 - 2.0 * local)

        p = self.p_open + self.g.curl * x**3 * (1.0 - tn)  # outer tips curl outwards
        p = p * growth.breathe

        change = self.ruffle
        if gust > 1e-3:
            # A ripple running through the petals from the outside in
            change = change + gust * 0.06 * x * np.sin(9.0 * tn * TAU - 7.0 * time)

        u = self.u_base + change
        y = 2.0 * (x * x - x) ** 2 * np.sin(p)
        r_open = u * (x * np.sin(p) + y * np.cos(p))
        z_open = u * (x * np.cos(p) - y * np.sin(p))
        r_bud = self.bud_r * (0.8 + 0.2 * u)
        z_bud = self.bud_z * u

        r = r_bud + (r_open - r_bud) * local
        z = z_bud + (z_open - z_bud) * local

        s = growth.head
        px = s * r * np.cos(t)
        py = s * r * np.sin(t)
        pz = s * z

        # tone: depth of the petal layer mixed with the position along the petal
        tone = 1.2 * tn - 0.35 * x
        # Shadows: petal bases sit deep inside the flower, the gaps between
        # neighbouring petals are dark seams, the heart of the rose is darkest.
        seam = np.clip((self.u_base - 0.5) / 0.12, 0.0, 1.0)
        ao = (0.3 + 0.7 * x**0.6) * (0.35 + 0.65 * seam) * (1.0 - 0.35 * tn * (1.0 - x))
        ao = ao * self.petal_shade  # every petal catches the light a little differently
        return _pack(px, py, pz, KIND_PETAL, tone, ao, self.petal_id)

    # --------------------------------------------------------------- leaves
    @staticmethod
    def leaf(base, angle, length, rise, droop, width=0.3, nu=26, nv=11, part=-2) -> Cloud:
        if length <= 1e-3:
            return Cloud.concat([])
        u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(-1, 1, nv), indexing="ij")
        d = np.array([np.cos(angle), np.sin(angle), 0.0])
        side = np.array([-np.sin(angle), np.cos(angle), 0.0])
        w = width * length * np.sin(np.pi * u) ** 0.8
        lift = length * (rise * u - droop * u * u) + 0.25 * w * v * v  # V-shaped fold
        px = base[0] + d[0] * u * length + side[0] * v * w
        py = base[1] + d[1] * u * length + side[1] * v * w
        pz = base[2] + lift
        return _pack(px, py, pz, KIND_GREEN, 0.5 + 0.5 * u, part=part)
