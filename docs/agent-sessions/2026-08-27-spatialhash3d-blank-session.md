# 2026-08-27 -- SpatialHash3D drew nothing

## Goal

`WebGPUCompute/SpatialHash3D` had stopped drawing. The window opened, the
yellow HUD line ticked over at 60fps, and the viewport stayed empty.

## What was wrong

The demo never read its rendered frame back to the CPU.

`WebGPUWidget` renders to its own textures and presents through `QPainter`, so
somebody has to copy the resolved colour target into `self.frame_buffer` each
frame. Up to PyNGL commit `430596b` (the Version1.0 merge) `paintEvent` did
that itself; from there on the call moved into the subclass, and the docstring
on `paintWebGPU` says as much — "implementations should end by calling
`_update_colour_buffer()`".

Every WebGPU demo in this repo was updated to do that. This one was missed. It
sat unnoticed because the pinned `ncca-ngl` was still the old library; commit
`efabe15` bumped `uv.lock` to the new one and the demo went blank the same day.

So the compute pass ran, the render pass drew grid and spheres into the MSAA
target, everything resolved correctly — and `frame_buffer` stayed the array of
zeros it was initialised with. Zeros are transparent black, `_present_image`
painted nothing, and the widget background showed through. The HUD text still
appeared because that is drawn by `QPainter` after the image, straight onto the
widget, and never goes near WebGPU.

No exception anywhere, which is why the log was clean.

The three-line `hasattr` guard in `_render_pass` that falls back to
`_init_textures()` is dead code, by the way: the base class's
`_create_render_buffer` sets all three attributes it checks for, on the first
resize. Left alone — it isn't what broke, and the demo's own textures are
`bgra8unorm` without `COPY_SRC`, so if it ever did fire the readback would fail
validation. Worth a look another day.

## The fix

One call at the end of `_render_pass`, after `submit`, matching
`SpatialHash2D/WebGPU2D.py:654`:

```python
self._update_colour_buffer()
```

## Files changed

- `WebGPUCompute/SpatialHash3D/WebGPU3D.py`

## Commands run

```bash
uv run ./WebGPU3D.py            # before: blank; after: 1000 spheres, ~59fps
uv run ./WebGPU3DGui.py         # the GUI host reuses WebGPUScene3D, also fixed
uv run ruff check WebGPUCompute/SpatialHash3D/
uv run ruff format --check WebGPUCompute/SpatialHash3D/
uv run pytest                   # 834 passed
```

Verified by screenshotting the running window either side of the change, since
there is nothing headless to assert on here.

## Not changed

`_create_compute_pipeline` opens `CollisionCompute3D.wgsl` by a bare relative
path, so `uv run WebGPUCompute/SpatialHash3D/WebGPU3D.py` from the repo root
dies with `FileNotFoundError` and gives you a blank window for an entirely
different reason. `RunDemos.py` sets `cwd` per demo so the launcher is fine, and
the 2D sibling and `SimpleComputeWebGPU` load shaders the same way — it is the
house style here, not a regression, so it stays as it is.
