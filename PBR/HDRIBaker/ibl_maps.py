"""Save and load the baked split-sum IBL maps as a single ``.npz``.

One file holds every map the shader needs: the environment cube (for the
skybox), the irradiance cube (diffuse ambient), the prefiltered specular
cube's mip chain, and the BRDF lookup table. Arrays are float16 to match the
GPU bake format.

The ``meta`` block is what makes a file self-describing: it carries the
:class:`~bake_settings.BakeSettings` the maps were baked at, so a loader sizes
its textures from the file it was handed rather than from whatever constants
its own build happens to hold. Files written before that block existed
(schema 1) are assumed to be the one fixed shape the baker could produce then.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from bake_settings import BakeSettings, expected_shapes, prefilter_key

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_ENV_SIZE",
    "DEFAULT_IRRADIANCE_SIZE",
    "DEFAULT_PREFILTER_SIZE",
    "DEFAULT_PREFILTER_MIPS",
    "DEFAULT_LUT_SIZE",
    "prefilter_key",
    "save_maps",
    "load_maps",
]

SCHEMA_VERSION = 2

# Seeds for the baker's GUI only. Nothing may derive a map's shape from these
# -- that is the settings block's job.
DEFAULT_ENV_SIZE = 512
DEFAULT_IRRADIANCE_SIZE = 32
DEFAULT_PREFILTER_SIZE = 128
DEFAULT_PREFILTER_MIPS = 5
DEFAULT_LUT_SIZE = 512


def _array_keys(settings: BakeSettings) -> list[str]:
    return ["env", "irradiance", "brdf_lut"] + [
        prefilter_key(m) for m in range(settings.prefilter_mips)
    ]


def save_maps(maps: dict, path: str | Path) -> None:
    """Write ``maps`` (arrays + a ``meta`` dict carrying ``settings``) to a ``.npz``."""
    meta = dict(maps["meta"])
    settings = BakeSettings.from_meta(meta)
    settings.validate()
    meta["schema_version"] = SCHEMA_VERSION
    meta["settings"] = settings.to_meta()
    meta["prefilter_roughness"] = settings.roughness_levels()

    arrays = {k: np.asarray(maps[k], dtype=np.float16) for k in _array_keys(settings)}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load_maps(path: str | Path) -> dict:
    """Load a ``.npz`` written by :func:`save_maps`; validate and return it.

    The returned dict holds each array, the raw ``meta`` dict, and a
    ``settings`` :class:`BakeSettings` rebuilt from the file.
    """
    try:
        npz = np.load(path, allow_pickle=False)
    except ValueError as err:
        # A non-.npz file (e.g. a stray .py passed to --maps) makes np.load
        # fall back to pickle and raise a confusing "pickled data" message;
        # translate it into something a user can act on.
        raise ValueError(
            f"{path} is not a NumPy .npz IBL maps file "
            f"(expected one written by the baker's Save)"
        ) from err
    with npz:
        if "meta" not in npz:
            raise ValueError(f"maps file {path} is missing arrays: ['meta']")
        meta = json.loads(str(npz["meta"]))
        settings = BakeSettings.from_meta(meta)
        settings.validate()

        missing = [k for k in _array_keys(settings) if k not in npz]
        if missing:
            raise ValueError(f"maps file {path} is missing arrays: {missing}")
        out = {k: npz[k] for k in _array_keys(settings)}

    out["meta"] = meta
    out["settings"] = settings
    for key, shape in expected_shapes(settings).items():
        if out[key].shape != shape:
            raise ValueError(
                f"maps file {path}: {key} has shape {out[key].shape}, expected {shape}"
            )
    return out
