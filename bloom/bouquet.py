"""A bouquet: several roses on a dome, stems gathered and tied with a satin bow,
wrapped in pearl paper, with leaves tucked between the flowers.

Each rose reuses the single-flower geometry (:class:`Flower`), then gets its own
size, tilt and position. Stems run from every flower head down to the tie point
and fan out slightly below it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .genome import Genome
from .geometry import KIND_GREEN, KIND_PETAL, TAU, Cloud, Flower, Growth, _ease, _pack


KIND_RIBBON = 2
KIND_PAPER = 3

TIE_Z = -0.95       # where the stems are tied together
BOTTOM_Z = -1.85    # cut ends of the stems


def _rotation_to(d: np.ndarray, spin: float) -> np.ndarray:
    """Rotation matrix that turns +Z into direction d, after spinning by `spin` around Z."""
    d = d / np.linalg.norm(d)
    c, s = np.cos(spin), np.sin(spin)
    spin_m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    sin_a, cos_a = np.linalg.norm(v), float(np.dot(z, d))
    if sin_a < 1e-8:
        return spin_m
    k = v / sin_a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + sin_a * kx + (1 - cos_a) * kx @ kx
    return rot @ spin_m


def _transform(cloud: Cloud, rot: np.ndarray, scale: float, offset: np.ndarray) -> Cloud:
    pos = (cloud.pos @ rot.T) * scale + offset
    nrm = cloud.nrm @ rot.T
    return Cloud(pos.astype(np.float32), nrm.astype(np.float32), cloud.kind, cloud.tone, cloud.ao, cloud.part)


@dataclass
class Bloom:
    """One rose in the bouquet."""

    flower: Flower
    position: np.ndarray
    direction: np.ndarray
    rot: np.ndarray
    scale: float
    delay: float        # seconds after the bouquet starts before this rose wakes up
    shade: float        # offset into the palette, so roses differ in tone
    bottom: np.ndarray  # where its stem ends


def _tube(path: np.ndarray, radius: float, kind: int, tone: float, part: int, na: int = 10) -> Cloud:
    """A thin tube along a polyline (n, 3)."""
    if len(path) < 2:
        return Cloud.concat([])
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    helper = np.where(np.abs(tangent[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    n1 = np.cross(tangent, helper)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9
    n2 = np.cross(tangent, n1)
    a = np.linspace(0, TAU, na, endpoint=False)
    nrm = n1[:, None, :] * np.cos(a)[None, :, None] + n2[:, None, :] * np.sin(a)[None, :, None]
    pos = path[:, None, :] + radius * nrm
    n = pos.shape[0] * pos.shape[1]
    return Cloud(
        pos.reshape(-1, 3).astype(np.float32),
        nrm.reshape(-1, 3).astype(np.float32),
        np.full(n, kind, np.uint8),
        np.full(n, tone, np.float32),
        np.ones(n, np.float32),
        np.full(n, part, np.int32),
    )


class Bouquet:
    def __init__(self, seed: str, detail: float = 1.0):
        rng = random.Random(seed)
        self.seed = seed
        self.blooms: list[Bloom] = []

        nx = max(16, int(30 * detail))
        nt = max(500, int(1500 * detail))

        layout = [(0.0, 0.0, 0.2, 0.0, 0.56)]  # (ring radius, angle, height, tilt, scale)
        ring = 6
        for i in range(ring):
            angle = i * TAU / ring + rng.uniform(-0.15, 0.15) + 0.3
            layout.append((0.56, angle, -0.02, 0.62, rng.uniform(0.46, 0.53)))
        outer = 5
        for i in range(outer):
            angle = (i + 0.5) * TAU / outer + rng.uniform(-0.2, 0.2)
            layout.append((0.86, angle, -0.3, 1.05, rng.uniform(0.3, 0.36)))

        for i, (radius, angle, height, tilt, scale) in enumerate(layout):
            genome = Genome.from_seed(f"{seed}/{i}")
            k = 1.0 if i < 1 + ring else 0.55  # small outer roses need fewer points
            flower = Flower(genome, nx=max(14, int(nx * k)), nt=max(300, int(nt * k)))
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
            direction = np.array([np.sin(tilt) * np.cos(angle), np.sin(tilt) * np.sin(angle), np.cos(tilt)])
            spread = 0.06 + 0.1 * radius
            bottom = np.array([spread * np.cos(angle + 0.4), spread * np.sin(angle + 0.4), BOTTOM_Z])
            self.blooms.append(
                Bloom(
                    flower=flower,
                    position=pos,
                    direction=direction,
                    rot=_rotation_to(direction, rng.uniform(0, TAU)),
                    scale=scale,
                    delay=0.35 * i + rng.uniform(0, 0.3),
                    shade=rng.uniform(0, 1),
                    bottom=bottom,
                )
            )

        self._static: Cloud | None = None              # stems, leaves, paper, ribbon once grown
        self._open: list[Cloud | None] = [None] * len(self.blooms)  # fully open roses

        # Leaves tucked around the rim, pointing outwards between the roses
        self.leaves = []
        for i in range(9):
            angle = (i + 0.3) * TAU / 9 + rng.uniform(-0.15, 0.15)
            base = np.array([0.55 * np.cos(angle), 0.55 * np.sin(angle), -0.42])
            self.leaves.append((base, angle, rng.uniform(0.42, 0.55)))

    @property
    def settled_at(self) -> float:
        """Time when the last rose has fully opened."""
        return max(b.delay for b in self.blooms) + 9.3

    # ------------------------------------------------------------- pieces
    def _stems(self, grow: float) -> list[Cloud]:
        parts = []
        s = np.linspace(0, 1, 90)[:, None]
        for i, b in enumerate(self.blooms):
            head = b.position - b.direction * 0.02
            tie = np.array([b.position[0] * 0.08, b.position[1] * 0.08, TIE_Z])
            ctrl = head - b.direction * 0.35
            upper = (1 - s) ** 2 * head + 2 * (1 - s) * s * ctrl + s * s * tie  # head -> tie
            lower = tie + (b.bottom - tie) * s                                 # tie -> cut end
            path = np.concatenate([upper, lower[1:]])
            visible = np.linspace(1, 0, len(path)) <= grow  # grows from the cut end upwards
            parts.append(_tube(path[visible], 0.02, KIND_GREEN, 0.25, -100 - i, na=8))
        return parts

    def _leaves(self, unfurl: float) -> list[Cloud]:
        if unfurl <= 0:
            return []
        return [
            Flower.leaf(base, angle, size * _ease(unfurl), rise=0.35, droop=0.75, width=0.34, part=-200 - i)
            for i, (base, angle, size) in enumerate(self.leaves)
        ]

    def _paper(self, unfold: float) -> list[Cloud]:
        """Pearl wrapping paper: a ruffled cone around the back and the sides."""
        if unfold <= 0:
            return []
        h = np.linspace(0, 1, 40)
        a = np.linspace(-0.12 * np.pi, 1.12 * np.pi, 150)  # leaves the front open
        H, A = np.meshgrid(h, a, indexing="ij")
        top = -0.05 + 0.08 * np.sin(7 * A) * H
        z = TIE_Z + (top - TIE_Z) * H * unfold
        r = (0.1 + 0.95 * H**1.3) * unfold + 0.08 * (1 - unfold)
        r = r * (1 + 0.04 * np.sin(11 * A) * H)  # soft folds
        px, py = r * np.cos(A), r * np.sin(A)
        cloud = _pack(px, py, z, KIND_PAPER, 0.5 + 0.5 * H, ao=0.55 + 0.45 * H, part=-300)
        return [cloud]

    def _ribbon(self, tie: float) -> list[Cloud]:
        if tie <= 0:
            return []
        parts = []
        # the band around the stems
        a = np.linspace(0, TAU, 60)
        z = np.linspace(-0.035, 0.035, 5)
        A, Z = np.meshgrid(a, z, indexing="ij")
        r = 0.07
        parts.append(_pack(r * np.cos(A), r * np.sin(A), TIE_Z + Z, KIND_RIBBON, 0.5 + 0.0 * A, part=-400))
        # bow loops and tails, facing the viewer (-y)
        knot = np.array([0.0, -0.1, TIE_Z])
        s = _ease(tie)
        for i, (angle, rise, droop, size) in enumerate(
            [(0.15, 0.7, 1.1, 0.2), (np.pi - 0.15, 0.7, 1.1, 0.2), (-0.5 * np.pi + 0.5, -1.4, 0.2, 0.26),
             (-0.5 * np.pi - 0.5, -1.5, 0.2, 0.24)]
        ):
            leaf = Flower.leaf(knot, angle, size * s, rise=rise, droop=droop, width=0.45 if i < 2 else 0.16,
                               nu=22, nv=9, part=-410 - i)
            leaf.kind[:] = KIND_RIBBON
            parts.append(leaf)
        return parts

    # ---------------------------------------------------------------- build
    def build(self, tau: float, gust: float, time: float, breathe_freq: float = 1.3) -> Cloud:
        parts: list[Cloud] = []
        if tau >= 3.5 and self._static is not None:
            parts.append(self._static)
        else:
            static = Cloud.concat(
                self._stems(_ease(np.clip(tau / 2.0, 0, 1)))
                + self._paper(_ease(np.clip((tau - 0.6) / 2.2, 0, 1)))
                + self._leaves(np.clip((tau - 1.4) / 2.0, 0, 1))
                + self._ribbon(np.clip((tau - 2.0) / 1.2, 0, 1))
            )
            self._static = static if tau >= 3.5 else None
            parts.append(static)

        for i, b in enumerate(self.blooms):
            g = Growth.at(tau - b.delay, breathe_freq, phase=b.shade * TAU)
            if g.head <= 0:
                continue
            if g.finished and gust < 1e-3:
                # Fully open and calm: reuse the cached shape and only let it
                # breathe by opening a touch wider or closing a touch.
                if self._open[i] is None:
                    calm = Growth(g.stem, g.head, 1.0, 1.0)
                    self._open[i] = self._tint(b.flower.head(calm, 0.0, time), i, b)
                head = self._open[i]
                k = 1.0 + 1.4 * (g.breathe - 1.0)
                rot = b.rot @ np.diag([k, k, 1.0 / k])
                parts.append(_transform(head, rot, b.scale, b.position))
            else:
                head = self._tint(b.flower.head(g, gust, time), i, b)
                parts.append(_transform(head, b.rot, b.scale, b.position))
        return Cloud.concat(parts)

    @staticmethod
    def _tint(head: Cloud, i: int, b: Bloom) -> Cloud:
        head.tone = head.tone + 2.6 * b.shade  # each rose starts at its own shade of gold
        head.part = head.part + 1000 * (i + 1)
        return head

    def sources(self) -> np.ndarray:
        """Hearts of the roses: where pollen comes from."""
        return np.array([b.position + b.direction * 0.3 * b.scale for b in self.blooms])
