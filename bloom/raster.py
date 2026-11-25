"""Turns 3D point clouds into a colored grid of ASCII characters.

Points are projected with perspective, a z-buffer keeps the nearest point per
character cell, and only the winners get lit and colored. Brightness picks
the glyph from a density ramp; color comes from the palette and shimmers
with time and with the viewing angle (like the shell of a beetle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import palettes
from .bouquet import KIND_PAPER, KIND_RIBBON
from .geometry import KIND_GREEN, KIND_PETAL, Cloud
from .particles import Sprites


RAMP = np.array(list(".,:;-=+*#%@"))

LIGHT = np.array([-0.3, -0.85, 0.45])
LIGHT /= np.linalg.norm(LIGHT)


@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = 0.7   # radians, how much we look down at the flower
    zoom: float = 1.15
    target_z: float = -0.45
    distance: float = 6.0
    center: float = 0.5  # where the target sits on screen, as a fraction of the height


@dataclass
class Frame:
    chars: np.ndarray  # (rows, cols) '<U1', ' ' where empty
    rgb: np.ndarray    # (rows, cols, 3) uint8

    @property
    def rows(self) -> int:
        return self.chars.shape[0]

    @property
    def cols(self) -> int:
        return self.chars.shape[1]

    def put_text(self, row: int, text: str, rgb: np.ndarray) -> None:
        """Write text centered on a row; rgb is (len(text), 3)."""
        if not 0 <= row < self.rows:
            return
        start = max(0, (self.cols - len(text)) // 2)
        for i, ch in enumerate(text[: self.cols - start]):
            if ch != " ":
                self.chars[row, start + i] = ch
                self.rgb[row, start + i] = rgb[i]


def _rotate(v: np.ndarray, cam: Camera) -> np.ndarray:
    """World -> view space. Returns columns (right, depth, up); depth < 0 is closer."""
    cy, sy = np.cos(cam.yaw), np.sin(cam.yaw)
    cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)
    x1 = v[:, 0] * cy - v[:, 1] * sy
    y1 = v[:, 0] * sy + v[:, 1] * cy
    z1 = v[:, 2]
    return np.stack([x1, y1 * cp - z1 * sp, y1 * sp + z1 * cp], axis=1)


def render(
    cloud: Cloud,
    sprites: Sprites,
    cam: Camera,
    cols: int,
    rows: int,
    cell_aspect: float,
    palette: str,
    time: float,
    shimmer_speed: float = 0.1,
) -> Frame:
    chars = np.full((rows, cols), " ", dtype="<U1")
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    n_cloud = len(cloud.pos)
    pos = np.concatenate([cloud.pos, sprites.pos]) if len(sprites.pos) else cloud.pos
    if len(pos) == 0:
        return Frame(chars, rgb)

    view = _rotate(pos - np.array([0.0, 0.0, cam.target_z]), cam)
    f = cam.distance / (cam.distance + view[:, 1])
    scale = cam.zoom * min(cols, rows * cell_aspect) / 4.0
    col = np.round(cols / 2 + view[:, 0] * f * scale).astype(np.int64)
    row = np.round(rows * cam.center - view[:, 2] * f * scale / cell_aspect).astype(np.int64)

    ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows) & (f > 0)
    idx_all = np.nonzero(ok)[0]
    if len(idx_all) == 0:
        return Frame(chars, rgb)
    cell = row[idx_all] * cols + col[idx_all]
    depth = view[idx_all, 1]

    # z-buffer: write points from the farthest to the nearest, so the nearest
    # point of every cell is the one that stays
    # (16-bit depth keys let numpy use a radix sort, which is much faster)
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    key = ((1.0 - d) * 65535).astype(np.uint16)
    order = np.argsort(key, kind="stable")
    owner = np.full(rows * cols, -1, np.int64)
    owner[cell[order]] = order
    win_cells = np.nonzero(owner >= 0)[0]
    win_local = owner[win_cells]
    winners = idx_all[win_local]

    # Outlines: where one part overlaps another, the cell that lies behind
    # becomes a dark seam. This is what makes separate petals readable.
    part = np.full(rows * cols, -(2**30), np.int32)
    zbuf = np.full(rows * cols, np.inf, np.float32)
    surface_win = winners < n_cloud
    part[win_cells[surface_win]] = cloud.part[winners[surface_win]]
    zbuf[win_cells[surface_win]] = depth[win_local][surface_win]
    part, zbuf = part.reshape(rows, cols), zbuf.reshape(rows, cols)
    seam = np.zeros((rows, cols), bool)
    gap = 0.04
    for a_sl, b_sl in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        pa, pb, za, zb = part[a_sl], part[b_sl], zbuf[a_sl], zbuf[b_sl]
        differ = (pa != pb) & np.isfinite(za) & np.isfinite(zb)
        seam[a_sl] |= differ & (za > zb + gap)
        seam[b_sl] |= differ & (zb > za + gap)
    seam = seam.ravel()

    is_surface = winners < n_cloud
    surf = winners[is_surface]
    surf_cells = win_cells[is_surface]

    if len(surf):
        n = _rotate(cloud.nrm[surf], cam)
        facing = -n[:, 1]
        back = facing < 0
        n[back] *= -1.0  # petals are two-sided
        facing = np.abs(facing)

        diffuse = np.clip(n @ LIGHT, 0, 1)
        half = LIGHT + np.array([0.0, -1.0, 0.0])
        half /= np.linalg.norm(half)
        spec = np.clip(n @ half, 0, 1) ** 24
        rim = (1.0 - facing) ** 3
        lum = 0.28 + 0.68 * diffuse**0.8 + 0.3 * spec + 0.3 * rim
        lum = np.where(back, lum * 0.7, lum)
        lum = lum * (0.4 + 0.6 * cloud.ao[surf])
        lum = np.where(seam[surf_cells], lum * 0.25, lum)
        lum = np.clip(lum, 0, 1)

        kind = cloud.kind[surf]
        tone = cloud.tone[surf]
        base = np.empty((len(surf), 3), np.float32)

        petal = kind == KIND_PETAL
        if petal.any():
            # Iridescence: the hue drifts with time, with depth inside the
            # flower and with the angle at which we look at each petal.
            phase = time * shimmer_speed + 0.35 * tone[petal] + 0.3 * (1.0 - facing[petal])
            base[petal] = palettes.sample(palette, phase)

        green = kind == KIND_GREEN
        lum = np.where(green, lum * 0.8, lum)  # foliage stays a step behind the bloom
        if green.any():
            # warm olive and sage, so the leaves sit well next to gold
            t = tone[green][:, None]
            shimmer = 0.5 + 0.5 * np.sin(time * 0.8 + tone[green] * 8.0)[:, None]
            base[green] = np.array([70, 95, 45]) * (1 - t) + np.array([160, 175, 105]) * t
            base[green] += np.array([20, 15, 0]) * shimmer

        ribbon = kind == KIND_RIBBON
        if ribbon.any():
            # satin: pale champagne with a soft sheen that slides along it
            sheen = 0.5 + 0.5 * np.sin(time * 0.6 + tone[ribbon] * 6.0)[:, None]
            base[ribbon] = np.array([235, 215, 170]) + np.array([20, 25, 40]) * sheen
            lum = np.where(ribbon, lum * 0.85, lum)

        paper = kind == KIND_PAPER
        if paper.any():
            t = tone[paper][:, None]
            base[paper] = np.array([150, 120, 80]) * (1 - t) + np.array([225, 200, 160]) * t
            lum = np.where(paper, lum * 0.7, lum)  # the wrap frames the roses, never outshines them

        color = base * (0.3 + 0.9 * lum)[:, None]
        if petal.any() and palette == "starlight":
            # Metallic gold: shadows sink into warm amber instead of going grey,
            # lit parts climb to champagne.
            l = lum[petal][:, None]
            warm = np.array([0.55, 0.4, 0.2], np.float32)
            color[petal] = base[petal] * (warm * (1.0 - l) ** 1.5 + 1.3 * l ** 0.8)
        glint = np.where(kind == KIND_PETAL, 0.7, 0.25) * spec
        color += (255.0 - color) * glint[:, None]  # glints go toward white

        level = np.clip((lum**0.85 * len(RAMP)).astype(int), 0, len(RAMP) - 1)
        chars.flat[surf_cells] = RAMP[level]
        rgb.reshape(-1, 3)[surf_cells] = np.clip(color, 0, 255).astype(np.uint8)

    spr = winners[~is_surface] - n_cloud
    if len(spr):
        spr_cells = win_cells[~is_surface]
        chars.flat[spr_cells] = sprites.chars[spr]
        rgb.reshape(-1, 3)[spr_cells] = np.clip(sprites.rgb[spr], 0, 255).astype(np.uint8)

    return Frame(chars, rgb)
