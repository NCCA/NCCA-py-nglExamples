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
tonemapped preview, hit bake, and eyeball thumbnails of the four maps it
produced before saving them out. The `.exr` path goes through
[OpenEXR](https://pypi.org/project/OpenEXR/); `.hdr` is decoded by a small
Radiance RGBE reader in `hdri_input.py` so the tool doesn't need another
image library just for that one format.

The bake itself (`bake_ibl.py`) is the same split-sum pipeline as
`PBR/HDRI`: reproject the panorama to a cube, convolve it for irradiance,
importance-sample a GGX prefiltered specular chain across five roughness
mips, and compute the BRDF split-sum LUT. What's different here is that it
runs once, offscreen, and the results are read back to numpy and saved
rather than staying GPU-resident for an immediate render.

## The `.npz` schema

`ibl_maps.py` defines the layout and validates it on load:

| Array               | Shape                       | dtype     |
| -------------------- | --------------------------- | --------- |
| `env`                | 6 × 512 × 512 × 4            | float16   |
| `irradiance`          | 6 × 32 × 32 × 4               | float16   |
| `prefilter_0..4`      | 6 × (128 >> mip) × (128 >> mip) × 4 | float16 |
| `brdf_lut`            | 512 × 512 × 2                | float16   |
| `meta`                | JSON string (schema version, source path, …) | — |

Everything is `float16` to match the GPU bake format and keep the file
small, and `np.savez_compressed` packs it all into one `ibl_maps.npz`. A
copy baked from the bundled cloister HDRI ships in this folder so
`hdri_demo.py` runs out of the box without anyone having to bake first.

## The demo — `hdri_demo.py`

```bash
uv run PBR/HDRIBaker/hdri_demo.py --maps ibl_maps.npz
```

WebGPU only. Loads the `.npz`, uploads the four maps as GPU textures, and
lights the same 7×7 teapot grid as `PBR/HDRI` — metallic down the rows,
roughness across the columns — plus a skybox from the environment cube. No
convolution happens at runtime; it's the same split-sum ambient term as the
sibling demo, just sampling textures that were baked ahead of time instead
of ones this process just rendered. `--maps` defaults to the bundled
`ibl_maps.npz`, so leaving it off also works.

## Notes

If you regenerate `ibl_maps.npz` from a different HDRI, the demo will use
whatever it finds at the given path — there's no dependency on which HDRI
produced it, only on the array shapes matching `ibl_maps.py`'s schema.

## References

- [`PBR/HDRI`](../HDRI) — the bake-at-startup version this tool separates
  the baking from.
- [historic_cloister_passage_1k.exr](https://polyhaven.com/a/historic_cloister_passage)
