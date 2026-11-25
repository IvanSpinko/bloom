"""Cyclic color palettes. Colors flow through the stops over time."""

from __future__ import annotations

import numpy as np


PALETTES: dict[str, np.ndarray] = {
    name: np.array(stops, dtype=np.float32)
    for name, stops in {
        # champagne, honey, rose gold and pale gold: made for a Starlight MacBook
        "starlight": [(255, 238, 196), (248, 210, 140), (232, 182, 100), (244, 196, 156), (255, 224, 160),
                      (222, 172, 92)],
        "rose": [(220, 30, 70), (255, 105, 160), (200, 40, 160), (255, 120, 110)],
        "aurora": [(40, 230, 160), (40, 150, 255), (170, 80, 255), (255, 90, 200)],
        "ember": [(255, 50, 30), (255, 140, 20), (255, 210, 70), (220, 20, 70)],
        "ocean": [(20, 220, 220), (40, 120, 255), (90, 60, 230), (60, 200, 255)],
        "sakura": [(255, 190, 220), (255, 120, 180), (210, 150, 255), (255, 230, 240)],
        "prism": [(255, 60, 60), (255, 220, 50), (60, 255, 120), (60, 140, 255), (200, 70, 255)],
    }.items()
}


def sample(palette: str, phase: np.ndarray) -> np.ndarray:
    """Return an (N, 3) float array of RGB colors for phases in [0, 1) (wrapping)."""
    stops = PALETTES[palette]
    k = len(stops)
    f = np.mod(phase, 1.0) * k
    i = np.floor(f).astype(np.int32) % k
    frac = f - np.floor(f)
    frac = frac * frac * (3.0 - 2.0 * frac)  # smoothstep for softer blending
    return stops[i] * (1.0 - frac)[:, None] + stops[(i + 1) % k] * frac[:, None]
