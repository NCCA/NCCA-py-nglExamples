"""Save and load the baked split-sum IBL maps as a single ``.npz``.

One file holds every map the shader needs: the environment cube (for the
skybox), the irradiance cube (diffuse ambient), the prefiltered specular
cube's mip chain, and the BRDF lookup table. Arrays are float16 to match the
GPU bake format; a small JSON ``meta`` block records where they came from and
how they are shaped so the demo configures itself from the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ENV_SIZE = 512
IRRADIANCE_SIZE = 32
PREFILTER_SIZE = 128
PREFILTER_MIPS = 5
LUT_SIZE = 512


def prefilter_key(mip: int) -> str:
    """Name of the prefilter array for roughness mip ``mip`` (0 = mirror)."""
    return f"prefilter_{mip}"


def _array_keys() -> list[str]:
    return ["env", "irradiance", "brdf_lut"] + [
        prefilter_key(m) for m in range(PREFILTER_MIPS)
    ]


def save_maps(maps: dict, path: str | Path) -> None:
    """Write ``maps`` (arrays + a ``meta`` dict) to a compressed ``.npz``."""
    meta = dict(maps["meta"])
    meta.setdefault("schema_version", SCHEMA_VERSION)
    arrays = {k: np.asarray(maps[k], dtype=np.float16) for k in _array_keys()}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load_maps(path: str | Path) -> dict:
    """Load a ``.npz`` written by :func:`save_maps`; validate and return it."""
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
        missing = [k for k in _array_keys() if k not in npz]
        if "meta" not in npz:
            missing.append("meta")
        if missing:
            raise ValueError(f"maps file {path} is missing arrays: {missing}")

        out = {k: npz[k] for k in _array_keys()}
        out["meta"] = json.loads(str(npz["meta"]))

    expected = {
        "env": (6, ENV_SIZE, ENV_SIZE, 4),
        "irradiance": (6, IRRADIANCE_SIZE, IRRADIANCE_SIZE, 4),
        "brdf_lut": (LUT_SIZE, LUT_SIZE, 2),
    }
    for mip in range(PREFILTER_MIPS):
        size = PREFILTER_SIZE >> mip
        expected[prefilter_key(mip)] = (6, size, size, 4)
    for key, shape in expected.items():
        if out[key].shape != shape:
            raise ValueError(
                f"maps file {path}: {key} has shape {out[key].shape}, expected {shape}"
            )
    return out
