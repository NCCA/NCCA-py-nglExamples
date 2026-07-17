# HDRI IBL Baker

![](HDRIBaker.png)

[`PBR/HDRI`](../HDRI) bakes its split-sum IBL maps on the GPU every time it
starts up. That's fine for a demo you run once and look at, but it means
every launch pays for the same convolution work again, and the bake is
tangled up with the render loop it feeds. This factors the bake out into its
own tool: a PySide6 app that loads an HDRI, lets you preview it and the
baked results, then writes everything to a single `.npz` a separate demo can
load and light from with no GPU bake at all.

## The tool — `hdri_baker.py`

```bash
uv run PBR/HDRIBaker/hdri_baker.py
```

Open an `.exr` or `.hdr` equirectangular panorama (there's a default —
`images/historic_cloister_passage_1k.exr` — loaded on startup), look at the
tonemapped preview, hit bake, and thumbnails of the four maps it
produced before saving them out. The `.exr` path goes through
[OpenEXR](https://pypi.org/project/OpenEXR/); `.hdr` is decoded by a small
Radiance RGBE reader in `hdri_input.py` so the tool doesn't need another
image library just for that one format.

The bake itself (`bake_ibl.py`) is the same split-sum pipeline as
`PBR/HDRI`: reproject the panorama to a cube, convolve it for irradiance,
importance-sample a GGX prefiltered specular chain across the roughness
mips, and compute the BRDF split-sum LUT. What's different here is that it
runs once, offscreen, and the results are read back to numpy and saved
rather than staying GPU-resident for an immediate render.

### Bake settings

Everything the bake can be tuned with lives in `BakeSettings`
(`bake_settings.py`) and the "Bake settings" panel just edits one of those.
It falls into two groups:

- **Sizes** (environment cube, irradiance cube, prefilter cube + mip count,
  BRDF LUT) — these are the resolution of each map. Bigger costs you file
  size and bake time; the LUT and irradiance map barely need it (they're
  already blurry by construction) but a small prefilter cube shows up as
  blocky reflections at low roughness.
- **Sample counts** (prefilter samples, BRDF samples, irradiance sample
  delta) — these buy you less noise per texel for the same size, at the
  cost of bake time. The prefilter is Monte-Carlo GGX importance sampling,
  so it's the one where sample count actually matters visually.

Worth trying: bake once at 8 prefilter samples, then again at 2048, and
compare the prefilter thumbnail and the `baked in …s` readout under the
panel. The low-sample bake is fast and speckled; the high-sample one is
smooth and much slower. That trade-off is the entire point of Monte-Carlo
convolution, and it's otherwise invisible unless you can see both ends of
it side by side.

## The `.npz` schema

`ibl_maps.py` defines the layout and validates it on load. This is schema
v2: the file carries the `BakeSettings` it was baked at in its `meta`
block, so `hdri_demo.py` reads that block and sizes its GPU textures from
the file rather than from constants in the code — bake at a 1024 env cube
and 3 prefilter mips and the demo just goes and does that. A v1 file (no
`settings` block) still loads; it's assumed baked at the old fixed shape
below, which is also `BakeSettings`'s defaults.

| Array               | Shape                       | dtype     |
| -------------------- | --------------------------- | --------- |
| `env`                | 6 × 512 × 512 × 4            | float16   |
| `irradiance`          | 6 × 32 × 32 × 4               | float16   |
| `prefilter_0..4`      | 6 × (128 >> mip) × (128 >> mip) × 4 | float16 |
| `brdf_lut`            | 512 × 512 × 2                | float16   |
| `meta`                | JSON string (schema version, bake settings, source path, …) | — |

Everything is `float16` to match the GPU bake format and keep the file
small, and `np.savez_compressed` packs it all into one `ibl_maps.npz`. A
copy baked from the bundled cloister HDRI at the default settings ships in
this folder so `hdri_demo.py` runs out of the box without anyone having to
bake first.

## The demo — `hdri_demo.py`

![](HDRIDemo.png)

```bash
uv run PBR/HDRIBaker/hdri_demo.py --maps ibl_maps.npz
```

WebGPU only. Loads the `.npz`, uploads the four maps as GPU textures, and
lights a single teapot with the split-sum IBL plus a skybox from the
environment cube. No convolution happens at runtime; it's the same
split-sum ambient term as `PBR/HDRI`, just sampling textures that were
baked ahead of time instead of ones this process just rendered. `--maps`
defaults to the bundled `ibl_maps.npz`, so leaving it off also works.

Where `PBR/HDRI` fixes the material and sweeps it across a 7×7 grid, this
one hands the material to you. A floating control panel — a transparent
QML overlay borrowed from
[`GUIDemos/QMLWebGPUOverlay`](../../GUIDemos/QMLWebGPUOverlay) — drives the
teapot's PBR inputs live: metallic, roughness and ambient-occlusion
sliders, an albedo colour picker, an IBL on/off toggle, and a selector that
puts the environment, irradiance or a prefilter mip in the skybox so you
can see the maps the shader actually samples. There's also an orbit toggle
that circles the camera around the teapot so you can see every side, with a
speed slider to set how fast — while it orbits, the up/down arrow keys raise
and lower the camera and left/right widen or tighten the orbit radius. Drag
a panel anywhere; click off a panel and the drag rotates the camera (wheel
zooms). Turn metallic down and the
albedo colour shows through; push roughness up and the reflection walks up
the prefiltered mip chain — the split sum made tangible.

The `File` menu swaps things out at runtime. `Load IBL Maps…` opens a
different `.npz` — bake another HDRI with `hdri_baker.py` and load it here
to relight the mesh in a new environment without restarting. `Load Mesh…`
opens a triangulated `.obj` (I centre it and scale it to fit, so any model
lands in front of the camera at a sensible size) — point it at one of the
meshes under `NormalMapping/models` or `ObjViewer/models` to swap the
teapot for a troll or a helix.

## Notes

If you regenerate `ibl_maps.npz` from a different HDRI, the demo will use
whatever it finds at the given path — there's no dependency on which HDRI
produced it, only on the arrays matching the settings block the file
carries.

## References

- [`PBR/HDRI`](../HDRI) — the bake-at-startup version this tool separates
  the baking from.
- [historic_cloister_passage_1k.exr](https://polyhaven.com/a/historic_cloister_passage)
