"""Every knob the IBL bake has, in one place.

The baker used to hardcode its map sizes and sample counts as module
constants, which meant a saved ``.npz`` was only readable by a build that
happened to agree with it. A ``BakeSettings`` travels with the bake instead:
the GUI builds one, the bake obeys it, and it is written into the file's
``meta`` block so a loader can size its textures from the file rather than
from an import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

_SIZE_FIELDS = ("env_size", "irradiance_size", "prefilter_size", "lut_size")


def prefilter_key(mip: int) -> str:
    """Name of the prefilter array for roughness mip ``mip`` (0 = mirror)."""
    return f"prefilter_{mip}"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class BakeSettings:
    """Sizes and sample counts for one bake. Defaults reproduce the v1 maps."""

    env_size: int = 512
    irradiance_size: int = 32
    prefilter_size: int = 128
    prefilter_mips: int = 5
    lut_size: int = 512
    prefilter_samples: int = 1024
    brdf_samples: int = 1024
    irradiance_sample_delta: float = 0.025

    def validate(self) -> None:
        """Raise ``ValueError`` if any field would produce a broken bake."""
        for name in _SIZE_FIELDS:
            value = getattr(self, name)
            if not _is_power_of_two(value):
                raise ValueError(f"{name} must be a power of two, got {value}")
        # roughness_for_mip divides by (mips - 1), and one mip cannot span
        # a roughness range anyway.
        if self.prefilter_mips < 2:
            raise ValueError(f"prefilter_mips must be >= 2, got {self.prefilter_mips}")
        if self.prefilter_size >> (self.prefilter_mips - 1) < 1:
            raise ValueError(
                f"prefilter_size {self.prefilter_size} is too small for "
                f"{self.prefilter_mips} mips (the last level would be under a texel)"
            )
        for name in ("prefilter_samples", "brdf_samples"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not 0.0 < self.irradiance_sample_delta <= 1.0:
            raise ValueError(
                "irradiance_sample_delta must be in (0, 1], got "
                f"{self.irradiance_sample_delta}"
            )

    def roughness_for_mip(self, mip: int) -> float:
        """Roughness baked into prefilter mip ``mip``: 0 at the top, 1 at the last."""
        return mip / (self.prefilter_mips - 1)

    def roughness_levels(self) -> list[float]:
        """Roughness at every mip in the chain, top to bottom."""
        return [self.roughness_for_mip(m) for m in range(self.prefilter_mips)]

    def to_meta(self) -> dict:
        """A JSON-safe dict of every field, for the ``.npz`` meta block."""
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: dict) -> "BakeSettings":
        """Rebuild settings from a file's meta block.

        A schema v1 file has no ``settings`` block; it can only ever have been
        baked at the constants of the day, so fall back to those.
        """
        block = meta.get("settings")
        if block is None:
            return cls.legacy_v1()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in known})

    @classmethod
    def legacy_v1(cls) -> "BakeSettings":
        """The one fixed shape every schema v1 file was baked at. Frozen
        history -- it must not track the defaults, which are free to move."""
        return cls(
            env_size=512,
            irradiance_size=32,
            prefilter_size=128,
            prefilter_mips=5,
            lut_size=512,
        )


def expected_shapes(settings: BakeSettings) -> dict[str, tuple[int, ...]]:
    """Array name -> required shape for a map set baked at ``settings``."""
    shapes: dict[str, tuple[int, ...]] = {
        "env": (6, settings.env_size, settings.env_size, 4),
        "irradiance": (6, settings.irradiance_size, settings.irradiance_size, 4),
        "brdf_lut": (settings.lut_size, settings.lut_size, 2),
    }
    for mip in range(settings.prefilter_mips):
        size = settings.prefilter_size >> mip
        shapes[prefilter_key(mip)] = (6, size, size, 4)
    return shapes
