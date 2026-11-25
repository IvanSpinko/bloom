"""A flower's "DNA": every visual trait is derived deterministically from a seed.

The same seed always grows the same flower, so a word (a name, a date, a
wish) becomes a flower that belongs only to it.
"""

from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass

from .palettes import PALETTES


WORDS = (
    "amber", "dawn", "ember", "fern", "glow", "haze", "iris", "jade", "lumen",
    "moss", "nova", "opal", "petal", "quill", "rain", "sage", "tide", "umbra",
    "velvet", "willow", "zephyr", "lily", "meadow", "aurora", "breeze",
)


def random_seed() -> str:
    """Two random words plus a number, e.g. ``opal-breeze-42``."""
    rng = random.Random(secrets.randbits(64))
    return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1, 99)}"


@dataclass(frozen=True)
class Leaf:
    position: float  # 0 = top of the stem, 1 = ground
    angle: float     # direction around the stem, radians
    size: float


@dataclass(frozen=True)
class Genome:
    seed: str
    petals_per_turn: float
    turns: float
    ruffle_freq: float
    ruffle_amp: float
    curl: float
    stem_length: float
    stem_bend: float
    leaves: tuple[Leaf, ...]
    sepals: int
    palette: str
    shimmer_speed: float

    @classmethod
    def from_seed(cls, seed: str) -> "Genome":
        digest = hashlib.sha256(seed.strip().lower().encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest, "big"))

        leaf_count = rng.randint(1, 3)
        leaves = tuple(
            Leaf(
                position=0.35 + 0.5 * (i + rng.random() * 0.6) / leaf_count,
                angle=rng.uniform(0, 6.283) + i * 2.4,
                size=rng.uniform(0.55, 0.85),
            )
            for i in range(leaf_count)
        )

        return cls(
            seed=seed,
            petals_per_turn=rng.uniform(3.25, 4.3),
            turns=rng.uniform(4.0, 6.0),
            ruffle_freq=rng.uniform(11.0, 19.0),
            ruffle_amp=rng.uniform(0.004, 0.012),
            curl=rng.uniform(0.1, 0.35),
            stem_length=rng.uniform(1.8, 2.3),
            stem_bend=rng.uniform(-0.35, 0.35),
            leaves=leaves,
            sepals=rng.randint(4, 6),
            palette=rng.choice(list(PALETTES)),
            shimmer_speed=rng.uniform(0.07, 0.14),
        )
