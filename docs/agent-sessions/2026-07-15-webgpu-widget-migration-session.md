# 2026-07-15 — Migrate WebGPU demos to the PyNGL WebGPUWidget

## Goal

`SimpleWebGPU` had already been switched to use `ncca.ngl.webgpu.WebGPUWidget`
instead of its own local copy. The job was to do the same for every other
WebGPU demo — use the library widget and drop the per-folder `WebGPUWidget.py`.

## What I found

Eighteen demos still imported a local `from WebGPUWidget import WebGPUWidget`.
The local copies came in a few generations:

- Most (16 demos) already matched the current widget API — same `paintWebGPU`
  / `resizeWebGPU` contract, same `_create_render_buffer` / `_update_colour_buffer`
  / `start_update_timer` methods. Their local file differed from the PyNGL one
  only cosmetically (docstrings, `devicePixelRatio` vs `devicePixelRatioF`, type
  hints). For these the migration is just the import swap and deleting the file.
- `Particles/ParticleQuads` used the older contract: the base created the device
  itself (`initializeWebGPU` / `_init_default_context`), rendered into a fresh
  per-frame texture, and read it back via `update_colour_buffer(texture)`. That
  one needed a real rewrite.

Demos that drive their own animation loop with Qt's `startTimer` + `timerEvent`
were left alone — that is built-in `QObject` behaviour, nothing to do with the
widget.

## Changes

- Swapped the import to `from ncca.ngl.webgpu import WebGPUWidget` in all 18
  demos and deleted the 18 local `WebGPUWidget.py` files.
- Rewrote `Particles/ParticleQuads/main.py` to the current pattern: create the
  device with `get_default_device()`, call `_create_render_buffer()` once, render
  into the base-owned `colour_buffer_texture_view`, and read back with
  `_update_colour_buffer()`. The particle pipeline is single-sampled, so I set
  `self.msaa_sample_count = 1` before the buffers are built.

Files changed: the 18 demo entry scripts plus 18 deleted widget files.

## Commands

```bash
# per-demo verification (each self-exits)
uv run <demo>.py --smoketest 300
uv run pytest -q          # 270 passed
uv run ruff check / format
```

All 18 demos print `SMOKETEST OK` with no tracebacks; tests and linters pass.

## Note

`WebGPUMultiGeo/WebGPUMultiGeo_updated.py` is an untracked scratch file that still
imports the local widget. It was not part of the worktree, so it is untouched —
if it is kept, its import needs the same swap.

## Follow-up: read-back ring bug (same day)

After the merge, `OITransparency` and `SimpleComputeWebGPU` came up grey. The
PyNGL widget's `_update_colour_buffer` reads from a pipelined ring of buffers
(`readback_buffers` / `_readback_index` / `_readback_pending`) that only its own
`_create_render_buffer` sets up. Four demos override `_create_render_buffer` and
built a single `readback_buffer` instead, so the inherited `_update_colour_buffer`
raised `AttributeError` every frame — swallowed by its try/except, which fills the
frame grey. My first smoketest check only looked for `SMOKETEST OK` and missed it;
the stricter check greps for `Failed to update colour buffer` too.

Fix (commit `30f6486` on `Version1.0`):

- `2DDrawingOpenGL/WebGPU2D`, `SimpleComputeWebGPU/WebGPU2D` and `RayMarchingSDF`
  only duplicated (or were a subset of) the base targets, so their overrides were
  removed and they inherit the base, ring included.
- `OITransparency` keeps its custom accum/reveal/composite targets but now calls
  `super()._create_render_buffer()` first to get the ring.

Verified all 18 demos with the stricter check (no readback failures); 275 tests pass.

The silent-failure class of bug was also fixed at the library level — see the PyNGL
repo session note for 2026-07-15 (fail loudly on a missing read-back ring).
