# HDRI IBL Baker Tool + Map-Loading Demo — Design

Date: 2026-07-16

## Goal

The `PBR/HDRI` demo bakes its image-based-lighting (IBL) maps on the GPU
every time it starts. This project adds:

1. A **PySide6 tool** that loads and previews an HDRI panorama, bakes the
   split-sum IBL maps, and saves them to a file so they can be reused.
2. A **WebGPU demo** that loads those saved maps and lights a scene with
   them — no bake at runtime.

Maps are saved as a single compressed `.npz` (float16 arrays + JSON metadata).

## Background

`PBR/HDRI/HDRIWebGPU.py` bakes four things on the GPU from an equirectangular
`.exr`:

| Map              | Shape (per file schema)      | Source shader           |
| ---------------- | ---------------------------- | ----------------------- |
| Environment cube | `(6, 512, 512, 4)` f16       | `Equirect2Cube.wgsl`    |
| Irradiance cube  | `(6, 32, 32, 4)` f16         | `Irradiance.wgsl`       |
| Prefilter cube   | 5 mips, `(6, 128>>m, …, 4)`  | `Prefilter.wgsl`        |
| BRDF LUT         | `(512, 512, 2)` f16          | `BRDF.wgsl`             |

`exr_loader.py` reads the `.exr` into `(H, W, 3)` float32. Everything
downstream currently lives on the GPU and is never read back to the CPU.

## Decisions (from brainstorming)

- **Save format:** all four maps → one `.npz` with named arrays + a `meta`
  JSON blob.
- **Backend:** WebGPU for both the baker and the demo (maximum reuse of the
  existing `HDRIWebGPU` bake path and WGSL shaders).
- **Tool UI:** HDRI equirect preview + baked-map thumbnails (irradiance,
  a prefilter mip, BRDF LUT), then Bake & Save.
- **Input formats:** `.exr` (existing OpenEXR loader) and `.hdr` (Radiance
  RGBE), the latter via a small built-in decoder — no new hard dependency
  (`imageio` is not installed).

## Folder layout

New self-contained demo folder, auto-discovered by `RunDemos.py`:

```
PBR/HDRIBaker/
  bake_ibl.py        # headless offscreen wgpu bake -> dict of numpy arrays
  hdri_input.py      # .exr / .hdr loader -> (H,W,3) float32
  ibl_maps.py        # save_maps / load_maps + .npz schema + metadata
  hdri_baker.py      # PySide6 tool GUI (Open / Bake / Save + previews)
  hdri_demo.py       # WebGPU demo: load .npz -> upload -> teapot grid + skybox
  shaders/           # copies of the WGSL bake + PBR/Skybox shaders it reuses
  ibl_maps.npz       # bundled pre-baked maps so hdri_demo.py runs out of the box
  README.md
  HDRIBaker.png      # preview image for RunDemos.py + root README
  tests/test_ibl_maps.py
```

Per repo convention each demo folder is self-contained, so the WGSL shaders
from `PBR/HDRI` are **copied** into `PBR/HDRIBaker/shaders/`, not referenced
across folders.

## Components

### 1. HDRI loader — `hdri_input.py`

`load_equirect_hdr(path) -> np.ndarray` returning `(H, W, 3)` float32, RGB,
HDR range preserved. Dispatches on suffix:

- `.exr` → reuse the OpenEXR reader (ported from `PBR/HDRI/exr_loader.py`).
- `.hdr` → a small built-in Radiance RGBE decoder (parse `#?RADIANCE`
  header, resolution line, RLE-decoded scanlines, RGBE → float via the
  shared exponent). Pure Python, headless, unit-testable.
- other suffix → raise `ValueError` with a clear message.

### 2. Headless bake core — `bake_ibl.py`

Lifts the four bake stages out of `HDRIWebGPU.py`
(`_upload_2d`, `_bake_cube`, `_bake_prefilter`, `_bake_brdf`) into a
standalone **offscreen** baker with no Qt window — uses
`wgpu.utils.get_default_device()` directly. The bake logic (capture views,
projection, dtypes, per-face render) is unchanged.

New piece the demo never needed: **texture readback**. For each cube face /
mip level and the LUT, copy the texture to a mappable buffer
(`copy_texture_to_buffer`, respecting the 256-byte `bytes_per_row`
alignment), map it, and assemble a numpy array of the documented shape/dtype.

`bake_maps(image: np.ndarray) -> dict` returns
`{env, irradiance, prefilter_0..4, brdf_lut, meta}`.

### 3. Map schema — `ibl_maps.py`

- `save_maps(maps: dict, path)` — `np.savez_compressed` of the float16
  arrays plus `meta` (a JSON string in a 0-d array): source filename, env /
  irradiance / prefilter / LUT sizes, prefilter mip count and per-mip
  roughness, texture format tag, schema version.
- `load_maps(path) -> dict` — inverse; validates array presence and parses
  `meta`. Raises a clear error on missing keys or shape mismatch.

### 4. Qt tool — `hdri_baker.py`

`QMainWindow`:

- Toolbar: **Open HDRI…**, **Bake**, **Save .npz…** (Bake disabled until an
  HDRI is loaded; Save disabled until a bake exists).
- Central widget: tonemapped equirect preview (Reinhard tonemap + gamma to a
  `QImage`/`QLabel`, scaled to fit).
- Thumbnail strip: baked irradiance (one face), a prefilter mip face, and the
  BRDF LUT, each tonemapped to a small `QImage`.
- Open → `load_equirect_hdr` → tonemap → preview.
- Bake → `bake_maps` (offscreen wgpu) → thumbnails from the readback arrays.
- Save → `save_maps` to a chosen path.

### 5. Demo — `hdri_demo.py`

A trimmed `HDRIWebGPU`:

- `_initialize_web_gpu` calls `load_maps(npz)` and uploads each array into the
  same cube / LUT textures the existing pipeline expects — an `_upload_cube`
  helper mirroring `_upload_2d`, writing all 6 faces (and, for the prefilter
  cube, all mips). No bake stages, no `.exr` read, no bake shaders needed at
  runtime.
- The **identical** PBR split-sum + skybox draw code, camera, controls, and
  `E`/`I`/Space keys carry over unchanged.
- CLI: `--maps PATH` (defaults to the bundled `ibl_maps.npz`). Env cube drives
  the skybox; irradiance + prefilter + BRDF LUT drive the ambient term —
  demonstrating the saved maps light the scene with zero runtime bake.

## Data flow

```
Open   : HDRI file -> load_equirect_hdr -> (H,W,3) f32 -> tonemap -> preview
Bake   : image -> offscreen wgpu bake (4 textures) -> readback -> numpy dict -> thumbnails
Save   : numpy dict -> save_maps -> ibl_maps.npz
Run    : hdri_demo.py --maps file.npz -> load_maps -> upload textures -> render grid + skybox
```

## Error handling

- Loader raises `ValueError` on unknown suffix / malformed RGBE.
- Tool catches load/bake/save errors and shows a `QMessageBox`, staying open.
- Demo: missing / corrupt `.npz` or shape mismatch → clear message, exit
  non-zero.
- Both `hdri_baker.py` and `hdri_demo.py` accept `--smoketest MS` (repo
  convention) to run briefly headless and print `SMOKETEST OK` for CI.

## Testing

`tests/test_ibl_maps.py` — headless, no GPU:

- `save_maps` / `load_maps` round-trip: array shapes, float16 dtype, and
  `meta` fields preserved.
- RGBE `.hdr` decoder against a tiny synthetic hand-built file (flat RLE and a
  known RGBE → float conversion).
- Loader suffix dispatch and rejection of an unsupported extension.

The GPU bake itself stays smoketest-only, consistent with the rest of the
repo (headless unit tests cover pure-Python maths/IO only).

## Reuse notes

- WGSL shaders (`Equirect2Cube`, `Irradiance`, `Prefilter`, `BRDF`, `PBR`,
  `Skybox`) are copied from `PBR/HDRI` into `PBR/HDRIBaker/shaders/`.
- Root `README.md` gets a link to the new demo folder; the folder gets a
  `README.md` and a preview screenshot per repo convention.
```