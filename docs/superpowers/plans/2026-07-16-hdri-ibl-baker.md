# HDRI IBL Baker Tool + Map-Loading Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PySide6 tool that loads/previews an HDRI, bakes the split-sum IBL maps on the GPU, and saves them to a single `.npz`; plus a WebGPU demo that loads that `.npz` and lights a teapot grid with zero runtime bake.

**Architecture:** A new self-contained demo folder `PBR/HDRIBaker/`. Pure-Python, headless, unit-tested pieces (HDRI `.exr`/`.hdr` loader, `.npz` save/load) sit under an offscreen wgpu bake core (ported from `PBR/HDRI/HDRIWebGPU.py` with texture readback added). The Qt tool drives them; the demo reuses the existing PBR + skybox WGSL pipeline but uploads pre-baked textures instead of baking.

**Tech Stack:** Python, uv, numpy, wgpu-py 0.29, PySide6, OpenEXR/Imath, ruff, pytest.

## Global Constraints

- Package manager is **uv** exclusively; run everything via `uv run`.
- Each demo folder is **self-contained** — copy the WGSL shaders it needs into `PBR/HDRIBaker/shaders/`, never reference another folder's shaders.
- No new hard dependency: `.hdr` support is a built-in RGBE decoder (`imageio` is not installed).
- Bake format is `rgba16float` (cube/env/irradiance/prefilter) and `rg16float` (BRDF LUT); saved arrays are **float16**.
- Map schema shapes: `env (6,512,512,4)`, `irradiance (6,32,32,4)`, `prefilter_m (6, 128>>m, 128>>m, 4)` for `m` in `0..4`, `brdf_lut (512,512,2)`.
- WebGPU projection uses `PerspMode.WebGPU`; cube capture views and dtypes are copied **verbatim** from `PBR/HDRI/HDRIWebGPU.py`.
- Demos accept `--smoketest MS` (default 200), print `SMOKETEST OK`, and exit — matching repo convention.
- Prose (README, docstrings, comments) follows the jon-writing-style skill.
- Work happens in a git worktree off a clean base; never commit to `main`/`master`; conventional commit messages; run ruff + pytest before finishing.

---

### Task 1: HDRI loader (`.exr` + `.hdr`)

**Files:**
- Create: `PBR/HDRIBaker/hdri_input.py`
- Create: `PBR/HDRIBaker/tests/__init__.py` (empty)
- Test: `PBR/HDRIBaker/tests/test_hdri_input.py`

**Interfaces:**
- Produces: `load_equirect_hdr(path: str | Path) -> np.ndarray` returning `(H, W, 3)` float32 RGB; `decode_rgbe(data: bytes) -> np.ndarray` returning `(H, W, 3)` float32 (exposed for testing).

- [ ] **Step 1: Write the failing tests**

Create `PBR/HDRIBaker/tests/__init__.py` (empty) and `PBR/HDRIBaker/tests/test_hdri_input.py`:

```python
import struct

import numpy as np
import pytest

from hdri_input import decode_rgbe, load_equirect_hdr


def _make_rle_hdr() -> bytes:
    """An 8x1 Radiance .hdr, new-style RLE, every pixel RGBE=(128,128,128,128).

    scale = 2**(128-136) = 1/256, so every channel decodes to 128/256 = 0.5.
    """
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n"
    # new-RLE scanline header: 2, 2, width_hi, width_lo
    scan = bytes([2, 2, 0, 8])
    # four channels R,G,B,E: run code 128+8=136 means "run of 8 of the next byte"
    for _ in range(4):
        scan += bytes([136, 128])
    return header + scan


def test_decode_rgbe_rle_scanline():
    img = decode_rgbe(_make_rle_hdr())
    assert img.shape == (1, 8, 3)
    assert img.dtype == np.float32
    np.testing.assert_allclose(img, 0.5, atol=1e-6)


def test_decode_rgbe_zero_exponent_is_black():
    # width 4 (<8) forces old flat format: 4 bytes RGBE per pixel, E=0 -> black
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n"
    body = bytes([200, 200, 200, 0]) * 4
    img = decode_rgbe(header + body)
    assert img.shape == (1, 4, 3)
    np.testing.assert_allclose(img, 0.0)


def test_load_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "panorama.png"
    bad.write_bytes(b"not hdr")
    with pytest.raises(ValueError, match="unsupported"):
        load_equirect_hdr(bad)


def test_load_hdr_dispatch(tmp_path):
    p = tmp_path / "flat.hdr"
    p.write_bytes(_make_rle_hdr())
    img = load_equirect_hdr(p)
    assert img.shape == (1, 8, 3)
    np.testing.assert_allclose(img, 0.5, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest PBR/HDRIBaker/tests/test_hdri_input.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hdri_input'`.
(pytest's rootdir insertion puts the test's folder on `sys.path`, so `import hdri_input` resolves to the sibling module.)

- [ ] **Step 3: Write the implementation**

Create `PBR/HDRIBaker/hdri_input.py`:

```python
"""Read an equirectangular HDRI (.exr or .hdr) into a numpy float32 RGB array.

This is the headless, GPU-free front of the baker, so it is the part that
gets real unit tests. The .exr path reuses the OpenEXR reader the HDRI demo
already relies on; the .hdr path is a small built-in Radiance RGBE decoder so
we do not pull in another image library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_equirect_hdr(path: str | Path) -> np.ndarray:
    """Load an equirectangular ``.exr`` or ``.hdr`` as ``(H, W, 3)`` float32 RGB.

    HDR range is preserved; the suffix picks the decoder.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".exr":
        return _load_exr(path)
    if suffix == ".hdr":
        return decode_rgbe(path.read_bytes())
    raise ValueError(f"unsupported HDRI format {suffix!r} (use .exr or .hdr)")


def _load_exr(path: Path) -> np.ndarray:
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(str(path))
    window = exr.header()["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channels(["R", "G", "B"], pixel_type)
    channels = [np.frombuffer(c, dtype=np.float32).reshape(height, width) for c in raw]
    return np.ascontiguousarray(np.stack(channels, axis=-1))


def _rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 4)`` uint8 RGBE array to ``(..., 3)`` float32 RGB."""
    rgb = rgbe[..., :3].astype(np.float32)
    exponent = rgbe[..., 3].astype(np.int32)
    # value = mantissa * 2**(E - 128 - 8); E == 0 is the encoded black.
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0)
    return (rgb * scale[..., None].astype(np.float32)).astype(np.float32)


def decode_rgbe(data: bytes) -> np.ndarray:
    """Decode Radiance ``.hdr`` (RGBE) bytes to an ``(H, W, 3)`` float32 array."""
    pos = data.find(b"\n\n")
    if pos < 0 or not data.startswith(b"#?"):
        raise ValueError("not a Radiance .hdr file")
    header, body = data[:pos], data[pos + 2 :]

    # Resolution line, e.g. "-Y 512 +X 1024"; we only handle the common
    # top-down, left-to-right orientation used by every panorama we bake.
    eol = body.find(b"\n")
    res = body[:eol].split()
    body = body[eol + 1 :]
    if res[0] != b"-Y" or res[2] != b"+X":
        raise ValueError(f"unsupported HDR orientation {res!r}")
    height, width = int(res[1]), int(res[3])

    rgbe = np.zeros((height, width, 4), dtype=np.uint8)
    offset = 0
    for y in range(height):
        offset = _decode_scanline(body, offset, rgbe[y], width)
    return _rgbe_to_float(rgbe)


def _decode_scanline(body: bytes, offset: int, row: np.ndarray, width: int) -> int:
    """Fill one ``(width, 4)`` scanline from ``body`` at ``offset``; return the
    new offset. Handles both new-style adaptive RLE and old flat RGBE."""
    new_rle = (
        8 <= width < 0x8000
        and body[offset] == 2
        and body[offset + 1] == 2
        and (body[offset + 2] << 8 | body[offset + 3]) == width
    )
    if not new_rle:
        flat = np.frombuffer(body, dtype=np.uint8, count=width * 4, offset=offset)
        row[:] = flat.reshape(width, 4)
        return offset + width * 4

    offset += 4
    for channel in range(4):  # R, G, B, E stored as four separate RLE streams
        x = 0
        while x < width:
            count = body[offset]
            offset += 1
            if count > 128:  # a run: (count - 128) copies of the next byte
                run = count - 128
                row[x : x + run, channel] = body[offset]
                offset += 1
                x += run
            else:  # a literal span of `count` bytes
                row[x : x + count, channel] = np.frombuffer(
                    body, dtype=np.uint8, count=count, offset=offset
                )
                offset += count
                x += count
    return offset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest PBR/HDRIBaker/tests/test_hdri_input.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add PBR/HDRIBaker/hdri_input.py PBR/HDRIBaker/tests/__init__.py PBR/HDRIBaker/tests/test_hdri_input.py
git commit -m "feat(hdri-baker): add .exr/.hdr equirect loader with RGBE decoder"
```

---

### Task 2: Map schema — save/load `.npz`

**Files:**
- Create: `PBR/HDRIBaker/ibl_maps.py`
- Test: `PBR/HDRIBaker/tests/test_ibl_maps.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SCHEMA_VERSION: int`
  - `PREFILTER_MIPS: int` (= 5), `ENV_SIZE=512`, `IRRADIANCE_SIZE=32`, `PREFILTER_SIZE=128`, `LUT_SIZE=512`
  - `prefilter_key(mip: int) -> str` returning `f"prefilter_{mip}"`
  - `save_maps(maps: dict, path: str | Path) -> None` — `maps` holds `env`, `irradiance`, `prefilter_0..4`, `brdf_lut` (all np.float16) and `meta` (a dict).
  - `load_maps(path: str | Path) -> dict` — inverse; `meta` comes back as a dict; raises `ValueError` on missing arrays or shape mismatch.

- [ ] **Step 1: Write the failing tests**

Create `PBR/HDRIBaker/tests/test_ibl_maps.py`:

```python
import numpy as np
import pytest

import ibl_maps
from ibl_maps import load_maps, prefilter_key, save_maps


def _fake_maps() -> dict:
    maps = {
        "env": np.zeros((6, 512, 512, 4), np.float16),
        "irradiance": np.full((6, 32, 32, 4), 0.25, np.float16),
        "brdf_lut": np.zeros((512, 512, 2), np.float16),
        "meta": {"source": "test.exr", "prefilter_mips": ibl_maps.PREFILTER_MIPS},
    }
    for mip in range(ibl_maps.PREFILTER_MIPS):
        size = ibl_maps.PREFILTER_SIZE >> mip
        maps[prefilter_key(mip)] = np.zeros((6, size, size, 4), np.float16)
    return maps


def test_round_trip_preserves_arrays_and_meta(tmp_path):
    path = tmp_path / "maps.npz"
    save_maps(_fake_maps(), path)
    loaded = load_maps(path)

    assert loaded["irradiance"].shape == (6, 32, 32, 4)
    assert loaded["irradiance"].dtype == np.float16
    np.testing.assert_allclose(loaded["irradiance"], 0.25)
    assert loaded[prefilter_key(4)].shape == (6, 8, 8, 4)
    assert loaded["meta"]["source"] == "test.exr"
    assert loaded["meta"]["prefilter_mips"] == ibl_maps.PREFILTER_MIPS


def test_load_rejects_missing_array(tmp_path):
    path = tmp_path / "broken.npz"
    maps = _fake_maps()
    del maps["irradiance"]
    # save raw so the file exists but is incomplete
    np.savez_compressed(path, **{k: v for k, v in maps.items() if k != "meta"})
    with pytest.raises(ValueError, match="irradiance"):
        load_maps(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest PBR/HDRIBaker/tests/test_ibl_maps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ibl_maps'`.

- [ ] **Step 3: Write the implementation**

Create `PBR/HDRIBaker/ibl_maps.py`:

```python
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
    with np.load(path, allow_pickle=False) as npz:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest PBR/HDRIBaker/tests/test_ibl_maps.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add PBR/HDRIBaker/ibl_maps.py PBR/HDRIBaker/tests/test_ibl_maps.py
git commit -m "feat(hdri-baker): add .npz save/load schema for baked IBL maps"
```

---

### Task 3: Offscreen bake core with readback

**Files:**
- Create: `PBR/HDRIBaker/shaders/Equirect2Cube.wgsl`, `Irradiance.wgsl`, `Prefilter.wgsl`, `BRDF.wgsl` (copies)
- Create: `PBR/HDRIBaker/bake_ibl.py`

**Interfaces:**
- Consumes: `ibl_maps.PREFILTER_MIPS`, sizes, `prefilter_key`.
- Produces: `bake_maps(image: np.ndarray, source: str = "") -> dict` — takes an `(H, W, 3)` float32 equirect image, returns a dict with `env`, `irradiance`, `prefilter_0..4`, `brdf_lut` (all np.float16, shapes per the schema) and `meta`.

- [ ] **Step 1: Copy the four bake shaders into the folder**

```bash
mkdir -p PBR/HDRIBaker/shaders
cp PBR/HDRI/Equirect2Cube.wgsl PBR/HDRI/Irradiance.wgsl PBR/HDRI/Prefilter.wgsl PBR/HDRI/BRDF.wgsl PBR/HDRIBaker/shaders/
```

- [ ] **Step 2: Write `bake_ibl.py`**

Create `PBR/HDRIBaker/bake_ibl.py`. This ports the four bake stages from `PBR/HDRI/HDRIWebGPU.py` (methods `_upload_2d`, `_make_cube_pipeline`, `_source_bind_group`, `_bake_cube`, `_render_face`, `_bake_prefilter`, `_bake_brdf`, and the module constants `_CAPTURE_*`, `_PREFILTER_CAPTURE_DTYPE`, `ENV_SIZE`…`LUT_SIZE`, `BAKE_FORMAT`, `LUT_FORMAT`, `_VERTEX_STRIDE`) into free functions on an explicit `device`, with **no Qt widget**, and adds texture readback.

```python
"""Bake the split-sum IBL maps offscreen and read them back to numpy.

The GPU work is the same as ``PBR/HDRI/HDRIWebGPU.py`` — reproject the
equirect panorama to a cube, convolve it to an irradiance cube, GGX-prefilter
a roughness mip chain, and integrate the BRDF lookup table — but here it runs
on a headless device with no window, and every result is copied back off the
GPU into a numpy array so it can be saved to a file.
"""

from __future__ import annotations

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import PerspMode, PrimData, Prims, Vec3, look_at, perspective

import ibl_maps

_FLOAT = np.dtype(np.float32).itemsize
_VERTEX_STRIDE = 8 * _FLOAT
BAKE_FORMAT = wgpu.TextureFormat.rgba16float
LUT_FORMAT = wgpu.TextureFormat.rg16float
_SHADER_DIR = __import__("pathlib").Path(__file__).resolve().parent / "shaders"

_CAPTURE_PROJECTION = perspective(90.0, 1.0, 0.1, 10.0, PerspMode.WebGPU)
_CAPTURE_VIEWS = [
    look_at(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(-1, 0, 0), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)),
    look_at(Vec3(0, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, -1)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0)),
    look_at(Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, -1, 0)),
]
_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4))],
        "offsets": [0, 64],
        "itemsize": 128,
    }
)
_PREFILTER_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "roughness"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), np.float32],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)


def bake_maps(image: np.ndarray, source: str = "") -> dict:
    """Bake every IBL map from an ``(H, W, 3)`` float32 equirect image."""
    device = wgpu.utils.get_default_device()
    baker = _Baker(device)
    rgba = np.dstack([image, np.ones(image.shape[:2], np.float32)]).astype(np.float32)
    equirect = baker.upload_2d(rgba, BAKE_FORMAT)

    env = baker.bake_cube("Equirect2Cube.wgsl", ibl_maps.ENV_SIZE, "2d", equirect)
    irradiance = baker.bake_cube(
        "Irradiance.wgsl", ibl_maps.IRRADIANCE_SIZE, "cube", env
    )
    prefilter = baker.bake_prefilter(env)
    lut = baker.bake_brdf()

    out = {
        "env": baker.read_cube(env, ibl_maps.ENV_SIZE, 0),
        "irradiance": baker.read_cube(irradiance, ibl_maps.IRRADIANCE_SIZE, 0),
        "brdf_lut": baker.read_2d(lut, ibl_maps.LUT_SIZE, ibl_maps.LUT_SIZE, 2, 0),
    }
    for mip in range(ibl_maps.PREFILTER_MIPS):
        size = ibl_maps.PREFILTER_SIZE >> mip
        out[ibl_maps.prefilter_key(mip)] = baker.read_cube(prefilter, size, mip)
    out["meta"] = {
        "source": source,
        "prefilter_mips": ibl_maps.PREFILTER_MIPS,
        "prefilter_roughness": [
            m / (ibl_maps.PREFILTER_MIPS - 1) for m in range(ibl_maps.PREFILTER_MIPS)
        ],
        "format": "rgba16float / rg16float",
    }
    return out


class _Baker:
    def __init__(self, device: "wgpu.GPUDevice") -> None:
        self.device = device
        cube = PrimData.primitive(Prims.CUBE.value).astype(np.float32)
        self.cube_buffer = device.create_buffer_with_data(
            data=cube, usage=wgpu.BufferUsage.VERTEX
        )
        self.cube_count = cube.size // 8
        self.sampler = device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
        )

    # ---- IO / upload (ported from HDRIWebGPU._upload_2d) --------------------
    def upload_2d(self, data: np.ndarray, fmt: str) -> "wgpu.GPUTexture":
        height, width = data.shape[:2]
        half = data.astype(np.float16)
        tex = self.device.create_texture(
            size=(width, height, 1),
            format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": tex},
            half.tobytes(),
            {"bytes_per_row": width * 4 * 2, "rows_per_image": height},
            (width, height, 1),
        )
        return tex

    # ---- pipelines / bake (ported verbatim from HDRIWebGPU) ----------------
    # NOTE FOR IMPLEMENTER: copy the bodies of _make_cube_pipeline,
    # _source_bind_group, _bake_cube, _render_face, _bake_prefilter and
    # _bake_brdf from PBR/HDRI/HDRIWebGPU.py, making these mechanical edits:
    #   * `self.device`         -> `self.device`   (unchanged)
    #   * `self.linear_sampler` -> `self.sampler`
    #   * `self.cube_geometry["buffer"]` -> `self.cube_buffer`
    #   * `self.cube_geometry["count"]`  -> `self.cube_count`
    #   * `HDRI_DIR / shader_name`       -> `_SHADER_DIR / shader_name`
    #   * texture `usage=` on every baked cube/lut gains `| wgpu.TextureUsage.COPY_SRC`
    #     so it can be read back (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC)
    #   * ENV_SIZE/IRRADIANCE_SIZE/PREFILTER_SIZE/PREFILTER_MIPS/LUT_SIZE come
    #     from the ibl_maps module (e.g. ibl_maps.PREFILTER_MIPS)
    # Keep _make_cube_pipeline / _bake_cube / _bake_prefilter / _bake_brdf and
    # _render_face otherwise identical.

    # ---- readback (NEW) ----------------------------------------------------
    def read_2d(
        self, texture, width: int, height: int, channels: int, mip: int, layer: int = 0
    ) -> np.ndarray:
        """Copy one mip/layer of a float16 texture back to an (H,W,C) array."""
        bytes_per_pixel = channels * 2  # float16
        # copy_texture_to_buffer requires bytes_per_row to be a multiple of 256
        row_bytes = width * bytes_per_pixel
        padded = (row_bytes + 255) & ~255
        buffer = self.device.create_buffer(
            size=padded * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": mip,
                "origin": (0, 0, layer),
            },
            {"buffer": buffer, "bytes_per_row": padded, "rows_per_image": height},
            (width, height, 1),
        )
        self.device.queue.submit([encoder.finish()])

        buffer.map_sync(wgpu.MapMode.READ)
        raw = buffer.read_mapped()
        buffer.unmap()
        flat = np.frombuffer(bytes(raw), dtype=np.float16)
        rows = flat.reshape(height, padded // 2)
        return np.ascontiguousarray(
            rows[:, : width * channels].reshape(height, width, channels)
        )

    def read_cube(self, texture, size: int, mip: int) -> np.ndarray:
        """Read all six faces of a cube mip into a (6, size, size, 4) array."""
        faces = [self.read_2d(texture, size, size, 4, mip, layer) for layer in range(6)]
        return np.stack(faces, axis=0)
```

Fill in the ported method bodies exactly as instructed in the `NOTE FOR IMPLEMENTER` comment above; do not invent new logic.

- [ ] **Step 3: Verify the bake runs and shapes match the schema**

Run:
```bash
uv run python -c "
import numpy as np, sys; sys.path.insert(0, 'PBR/HDRIBaker')
from bake_ibl import bake_maps
img = np.random.rand(64, 128, 3).astype(np.float32)
m = bake_maps(img, source='rand')
import ibl_maps
assert m['env'].shape == (6,512,512,4), m['env'].shape
assert m['irradiance'].shape == (6,32,32,4)
assert m['prefilter_4'].shape == (6,8,8,4), m['prefilter_4'].shape
assert m['brdf_lut'].shape == (512,512,2)
assert m['env'].dtype == np.float16
print('BAKE OK')
"
```
Expected: prints `BAKE OK` (requires a working GPU/EGL; on a headless CI box wgpu falls back to a software adapter).

- [ ] **Step 4: Commit**

```bash
git add PBR/HDRIBaker/shaders/*.wgsl PBR/HDRIBaker/bake_ibl.py
git commit -m "feat(hdri-baker): offscreen IBL bake core with GPU texture readback"
```

---

### Task 4: PySide6 baker tool + bundled `.npz`

**Files:**
- Create: `PBR/HDRIBaker/hdri_baker.py`
- Create: `PBR/HDRIBaker/ibl_maps.npz` (generated, committed)
- Modify: copy the source EXR so the tool has a default — `cp PBR/HDRI/images/historic_cloister_passage_1k.exr PBR/HDRIBaker/images/`

**Interfaces:**
- Consumes: `load_equirect_hdr`, `bake_maps`, `save_maps`.
- Produces: `HDRIBakerWindow(QMainWindow)`; CLI with `--smoketest MS`.

- [ ] **Step 1: Write `hdri_baker.py`**

Create `PBR/HDRIBaker/hdri_baker.py`:

```python
#!/usr/bin/env -S uv run --script
"""Bake IBL maps from an HDRI and save them for reuse. See README.md.

Load an equirectangular ``.exr`` or ``.hdr`` panorama, preview it, bake the
split-sum image-based-lighting maps on the GPU, eyeball the results as
thumbnails, then save the whole set to a single ``.npz`` a demo can load
without baking anything itself.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from bake_ibl import bake_maps
from hdri_input import load_equirect_hdr
from ibl_maps import prefilter_key, save_maps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
DEFAULT_HDRI = HERE / "images" / "historic_cloister_passage_1k.exr"


def _tonemap_to_qimage(rgb: np.ndarray) -> QImage:
    """Reinhard tonemap + gamma an (H,W,3+) float array to an 8-bit QImage."""
    rgb = np.asarray(rgb[..., :3], dtype=np.float32)
    mapped = rgb / (rgb + 1.0)
    srgb = np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)
    buf = np.ascontiguousarray((srgb * 255).astype(np.uint8))
    h, w, _ = buf.shape
    return QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class HDRIBakerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HDRI IBL Baker")
        self.image: np.ndarray | None = None
        self.maps: dict | None = None

        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.open_btn = QPushButton("Open HDRI…")
        self.bake_btn = QPushButton("Bake")
        self.save_btn = QPushButton("Save .npz…")
        self.bake_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.on_open)
        self.bake_btn.clicked.connect(self.on_bake)
        self.save_btn.clicked.connect(self.on_save)
        for b in (self.open_btn, self.bake_btn, self.save_btn):
            toolbar.addWidget(b)

        self.preview = QLabel("Open an HDRI to begin")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 320)

        self.thumbs = [QLabel() for _ in range(3)]
        thumb_row = QHBoxLayout()
        for label, caption in zip(self.thumbs, ("irradiance", "prefilter", "brdf")):
            col = QVBoxLayout()
            label.setFixedSize(128, 128)
            label.setAlignment(Qt.AlignCenter)
            col.addWidget(label)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            col.addWidget(cap)
            thumb_row.addLayout(col)

        layout = QVBoxLayout()
        layout.addWidget(self.preview, 1)
        layout.addLayout(thumb_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _show_error(self, title: str, err: Exception) -> None:
        traceback.print_exc()
        QMessageBox.critical(self, title, str(err))

    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HDRI", str(DEFAULT_HDRI.parent), "HDRI (*.exr *.hdr)"
        )
        if not path:
            return
        try:
            self.image = load_equirect_hdr(path)
        except Exception as err:  # noqa: BLE001 - surfaced to the user
            self._show_error("Could not load HDRI", err)
            return
        self._source = path
        pix = QPixmap.fromImage(_tonemap_to_qimage(self.image))
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.bake_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

    def on_bake(self) -> None:
        if self.image is None:
            return
        try:
            self.maps = bake_maps(self.image, source=Path(self._source).name)
        except Exception as err:  # noqa: BLE001
            self._show_error("Bake failed", err)
            return
        previews = (
            self.maps["irradiance"][0],
            self.maps[prefilter_key(2)][0],
            # BRDF LUT is 2-channel; pad a zero blue so it tonemaps as RGB
            np.dstack(
                [
                    self.maps["brdf_lut"],
                    np.zeros(self.maps["brdf_lut"].shape[:2], np.float16),
                ]
            ),
        )
        for label, data in zip(self.thumbs, previews):
            img = _tonemap_to_qimage(np.asarray(data, np.float32))
            label.setPixmap(QPixmap.fromImage(img).scaled(128, 128, Qt.KeepAspectRatio))
        self.save_btn.setEnabled(True)

    def on_save(self) -> None:
        if self.maps is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save maps", str(HERE / "ibl_maps.npz"), "NumPy (*.npz)"
        )
        if not path:
            return
        try:
            save_maps(self.maps, path)
        except Exception as err:  # noqa: BLE001
            self._show_error("Save failed", err)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = HDRIBakerWindow()
    win.resize(760, 620)
    win.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Copy the default EXR and smoketest the tool**

```bash
mkdir -p PBR/HDRIBaker/images
cp PBR/HDRI/images/historic_cloister_passage_1k.exr PBR/HDRIBaker/images/
uv run PBR/HDRIBaker/hdri_baker.py --smoketest 300
```
Expected: a window flashes up and the process prints `SMOKETEST OK`.

- [ ] **Step 3: Generate the bundled `ibl_maps.npz` from the cloister EXR**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'PBR/HDRIBaker')
from hdri_input import load_equirect_hdr
from bake_ibl import bake_maps
from ibl_maps import save_maps
img = load_equirect_hdr('PBR/HDRIBaker/images/historic_cloister_passage_1k.exr')
save_maps(bake_maps(img, source='historic_cloister_passage_1k.exr'), 'PBR/HDRIBaker/ibl_maps.npz')
print('WROTE ibl_maps.npz')
"
ls -la PBR/HDRIBaker/ibl_maps.npz
```
Expected: prints `WROTE ibl_maps.npz`; file is a few MB.

- [ ] **Step 4: Commit**

```bash
git add PBR/HDRIBaker/hdri_baker.py PBR/HDRIBaker/images/historic_cloister_passage_1k.exr PBR/HDRIBaker/ibl_maps.npz
git commit -m "feat(hdri-baker): add PySide6 baker tool and bundle a baked .npz"
```

---

### Task 5: WebGPU demo that loads the maps

**Files:**
- Create: `PBR/HDRIBaker/hdri_demo.py`
- Create: `PBR/HDRIBaker/shaders/PBR.wgsl`, `PBR/HDRIBaker/shaders/Skybox.wgsl` (copies)

**Interfaces:**
- Consumes: `load_maps`, `prefilter_key`, and the same WGSL PBR/Skybox pipeline as `HDRIWebGPU`.
- Produces: `HDRIDemo(WebGPUWidget)`; CLI `--maps PATH` (default bundled `ibl_maps.npz`) and `--smoketest MS`.

- [ ] **Step 1: Copy the render shaders**

```bash
cp PBR/HDRI/PBR.wgsl PBR/HDRI/Skybox.wgsl PBR/HDRIBaker/shaders/
```

- [ ] **Step 2: Write `hdri_demo.py`**

Create `PBR/HDRIBaker/hdri_demo.py` by adapting `PBR/HDRI/HDRIWebGPU.py`. Copy that file's body and make exactly these changes:

1. Update the module docstring to say it loads pre-baked maps from a `.npz` instead of baking (jon-writing-style).
2. Point shader loads at the local `shaders/` folder: change `HDRI_DIR / "PBR.wgsl"` → `SHADER_DIR / "PBR.wgsl"` and `HDRI_DIR / "Skybox.wgsl"` → `SHADER_DIR / "Skybox.wgsl"`, adding `SHADER_DIR = HDRI_DIR / "shaders"`.
3. Add the maps path to `__init__(self, maps_path)` and store it; `main()` passes `args.maps`.
4. **Replace the whole bake section** of `_initialize_web_gpu` — every line from `img = load_equirect_hdr(...)` through `self.brdf_lut = self._bake_brdf()` — with a call to `self._load_baked_maps()` (below). Keep the surrounding calls (`_create_render_buffer`, `_load_geometry`, `_create_sampler`, `_create_skybox_pipeline`, `_create_pbr_pipeline`, `_build_grid`) unchanged.
5. Delete the now-unused bake methods and their imports: `_upload_2d`, `_make_cube_pipeline`, `_source_bind_group`, `_bake_cube`, `_render_face`, `_bake_prefilter`, `_bake_brdf`, and the `from exr_loader import load_equirect_hdr` line plus the `_CAPTURE_*` / `_PREFILTER_CAPTURE_DTYPE` constants and `perspective`-based capture projection (they are only used by the deleted bake).
6. Add the new upload + load methods:

```python
def _upload_cube(self, faces: np.ndarray, size: int, mips: list | None = None):
    """Create a cube texture and fill it from (6,size,size,4) float16 arrays.

    `faces` is mip 0. `mips`, when given, is a list of extra
    (6, size>>m, size>>m, 4) arrays for mips 1..n.
    """
    levels = [faces] + (mips or [])
    tex = self.device.create_texture(
        size=(size, size, 6),
        mip_level_count=len(levels),
        format=BAKE_FORMAT,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    for mip, level in enumerate(levels):
        level_size = size >> mip
        data = np.ascontiguousarray(level.astype(np.float16))
        for face in range(6):
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": mip, "origin": (0, 0, face)},
                data[face].tobytes(),
                {"bytes_per_row": level_size * 4 * 2, "rows_per_image": level_size},
                (level_size, level_size, 1),
            )
    return tex


def _upload_lut(self, lut: np.ndarray):
    h, w = lut.shape[:2]
    data = np.ascontiguousarray(lut.astype(np.float16))
    tex = self.device.create_texture(
        size=(w, h, 1),
        format=LUT_FORMAT,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    self.device.queue.write_texture(
        {"texture": tex},
        data.tobytes(),
        {"bytes_per_row": w * 2 * 2, "rows_per_image": h},
        (w, h, 1),
    )
    return tex


def _load_baked_maps(self) -> None:
    maps = load_maps(self.maps_path)
    self.env_cube = self._upload_cube(maps["env"], ENV_SIZE)
    self.irradiance_cube = self._upload_cube(maps["irradiance"], IRRADIANCE_SIZE)
    prefilter_mips = [maps[prefilter_key(m)] for m in range(1, PREFILTER_MIPS)]
    self.prefilter_cube = self._upload_cube(
        maps[prefilter_key(0)], PREFILTER_SIZE, prefilter_mips
    )
    self.brdf_lut = self._upload_lut(maps["brdf_lut"])
```

7. Add imports at the top: `from ibl_maps import load_maps, prefilter_key, PREFILTER_MIPS` and `from pathlib import Path` (already present). Keep the `ENV_SIZE, IRRADIANCE_SIZE, PREFILTER_SIZE, PREFILTER_MIPS, LUT_SIZE` constants (they can stay as literals or import from `ibl_maps`; keep the existing literals to minimise churn) and `BAKE_FORMAT`/`LUT_FORMAT`.
8. In `main()`, add `parser.add_argument("--maps", default=str(HDRI_DIR / "ibl_maps.npz"))` and construct `HDRIScene(args.maps)`; update `_initialize_web_gpu` to be called after `self.maps_path = maps_path` is set in `__init__`.

- [ ] **Step 3: Smoketest the demo**

Run: `uv run PBR/HDRIBaker/hdri_demo.py --smoketest 400`
Expected: the teapot grid + skybox window flashes up, then prints `SMOKETEST OK`. Try `--maps PBR/HDRIBaker/ibl_maps.npz` explicitly too.

- [ ] **Step 4: Verify a missing maps file fails cleanly**

Run: `uv run PBR/HDRIBaker/hdri_demo.py --maps /no/such.npz --smoketest 200; echo "exit=$?"`
Expected: a clear error (file not found / load error), non-zero exit.

- [ ] **Step 5: Commit**

```bash
git add PBR/HDRIBaker/hdri_demo.py PBR/HDRIBaker/shaders/PBR.wgsl PBR/HDRIBaker/shaders/Skybox.wgsl
git commit -m "feat(hdri-baker): WebGPU demo that lights a scene from saved IBL maps"
```

---

### Task 6: README, preview image, root link, full check

**Files:**
- Create: `PBR/HDRIBaker/README.md`
- Create: `PBR/HDRIBaker/HDRIBaker.png` (screenshot)
- Modify: root `README.md` (add a link to the new demo)

- [ ] **Step 1: Capture a preview screenshot**

Run the demo, arrange the view, and save a screenshot to `PBR/HDRIBaker/HDRIBaker.png` (macOS: `Cmd-Shift-4`, drag over the window). This is manual; the file must exist before committing.

- [ ] **Step 2: Write `PBR/HDRIBaker/README.md`**

Use the jon-writing-style skill. Cover: what the tool does (load/preview HDRI → bake → save `.npz`), the four maps and the schema table, how to run the tool (`uv run PBR/HDRIBaker/hdri_baker.py`) and the demo (`uv run PBR/HDRIBaker/hdri_demo.py --maps ibl_maps.npz`), `.exr`/`.hdr` support, and a pointer back to `PBR/HDRI` as the bake-at-startup version. Embed `![](HDRIBaker.png)`.

- [ ] **Step 3: Add the demo to the root README**

Add a link to `PBR/HDRIBaker/` in the root `README.md` alongside the other PBR demos (match the existing list/table format — read the file first to follow its convention).

- [ ] **Step 4: Run the full check**

```bash
uv run ruff check PBR/HDRIBaker/ && uv run ruff format --check PBR/HDRIBaker/
uv run pytest PBR/HDRIBaker/tests/ -v
uv run PBR/HDRIBaker/hdri_baker.py --smoketest 300
uv run PBR/HDRIBaker/hdri_demo.py --smoketest 400
```
Expected: ruff clean, all pytest passing, both demos print `SMOKETEST OK`. If `ruff format --check` complains, run `uv run ruff format PBR/HDRIBaker/` and re-check.

- [ ] **Step 5: Commit**

```bash
git add PBR/HDRIBaker/README.md PBR/HDRIBaker/HDRIBaker.png README.md
git commit -m "docs(hdri-baker): add README, preview image and root README link"
```

---

## Self-Review notes

- **Spec coverage:** loader `.exr`+`.hdr` (Task 1), `.npz` schema (Task 2), offscreen bake + readback (Task 3), Qt tool with HDRI preview + thumbnails (Task 4), bundled `.npz` (Task 4), map-loading demo (Task 5), README/preview/root link + tests (Tasks 1,2,6). All spec sections map to a task.
- **Type consistency:** `prefilter_key`, `PREFILTER_MIPS`, `ENV_SIZE`… are defined in `ibl_maps` (Task 2) and consumed unchanged in Tasks 3–5; `bake_maps(image, source)` produced in Task 3 is consumed in Task 4; `load_maps` produced in Task 2 is consumed in Task 5.
- **Readback alignment:** `read_2d` pads `bytes_per_row` to 256 and crops — covers the small prefilter mips (16px→128B, 8px→64B rows) that are not already 256-aligned.
```