# 2026-08-27 -- DefferedLighting did nothing

## Goal

`DefferedLighting` was the one entry in `TODO.md` marked "not working at all",
and its README told you not to expect much. Find out why and make it draw.

## What was wrong

Four faults, each of which would have killed the demo on its own. Only the
first one announced itself.

**The lighting pass could not be called.** `LightingPipeline.paint()` was
declared as `paint(self, texture_view)` but `paintWebGPU` called it as
`paint(command_encoder, texture_view)`, so every paint event raised

```
TypeError: LightingPipeline.paint() takes 2 positional arguments but 3 were given
```

That is the whole of the visible symptom — a window that never paints and a
console full of the same traceback.

**The passes were queued in the wrong order.** Behind that signature mismatch,
`paint()` created its own command encoder and submitted it immediately, whilst
the caller was still holding an unfinished encoder with the geometry pass
recorded into it. Fixing the signature alone would have given a lighting pass
that ran before anything had written to the G-buffer. `paint()` now records
into the encoder it is handed and leaves the submit to the caller, which is the
point of passing an encoder around in the first place.

**The lighting bind group outlived the textures it pointed at.** `__init__`
built the G-buffer at the widget's pre-show size and handed those textures to
`LightingPipeline`, which made views of them for its bind group. Then
`win.resize(1024, 720)` fired `resizeEvent`, `_create_g_buffer()` threw the
textures away and built new ones, and the geometry pass started writing into
the new set — whilst the lighting pass carried on sampling the old set, which
nothing was writing to any more. Black frame, no error. `set_g_buffer()` now
rebuilds the bind group, and `_create_g_buffer()` calls it once the pipelines
exist.

**The floor lied about where it was.** `GBufferChecker.wgsl` wrote
`input.position` straight into the position target with a comment saying the
`M` matrix wasn't needed "as the floor is not transformed". It is — by the
mouse rotation and by a -0.45 offset in `y` — so the lighting pass was shading
the floor at points it wasn't at. The clip-space position used `MVP` correctly,
which is why the geometry looked right whilst the lighting didn't.

A fifth thing, less serious: pixels the geometry pass never touched keep the
cleared normal of `vec4(0)`, and the lighting pass fed that to `normalize`.
`normalize(vec3(0.0))` is NaN, so the horizon was whatever the tonemap made of
NaN — black, as it happens, which is why nobody noticed. There is now an
explicit test for a zero-length normal and a background colour written instead.

## The fix

- `LightingPipeline.paint(command_encoder, texture_view)` records into the
  caller's encoder and no longer submits.
- `LightingPipeline.set_g_buffer(views)` rebuilds the bind group;
  `_create_g_buffer()` calls it after a resize.
- `LightingPipeline` takes the widget's G-buffer *views* rather than the
  textures, so there is one set of views and no chance of a mismatch.
- Shader paths resolve against `__file__`, so the command in the README works
  from the repo root and not just from inside the folder. (`RunDemos.py` sets
  `cwd` to the demo folder, which is why this never showed up there.)
- `GBufferChecker.wgsl` writes `transforms.M * position` as the world position.
- `lighting.wgsl` masks out untouched pixels before `normalize`.
- `set_projection()` on both geometry pipelines, called from `resizeWebGPU` —
  they cached the projection built before the window was shown, so the aspect
  ratio never followed the window.
- `wheelEvent` reads `angleDelta().y()`; it was reading `.x()`, which is the
  tilt axis, so zoom did nothing.
- Dropped the duplicate `_create_render_buffer()` and frame-buffer allocation
  in `resizeEvent` — `resizeWebGPU` already does both.

## Files changed

- `DefferedLighting/LightingPipeline.py`
- `DefferedLighting/TeapotPipeline.py`
- `DefferedLighting/FloorPipeline.py`
- `DefferedLighting/SimpleWebGPU.py`
- `DefferedLighting/GBufferChecker.wgsl`
- `DefferedLighting/lighting.wgsl`
- `DefferedLighting/README.md`, `DefferedLighting/WebGPUNGL.png`
- `TODO.md` — the "not working at all" line is gone

## Commands run

```bash
git worktree add .worktrees/deferred-lighting-fix -b agent/deferred-lighting-fix
uv run DefferedLighting/SimpleWebGPU.py --smoketest 1500 --debug
uv run ruff check --select I --fix DefferedLighting/
uv run ruff format DefferedLighting/
uv run ruff check DefferedLighting/
uv run pytest          # 834 passed
```

Verified by driving the widget from a throwaway script that shows it, resizes
it to a different size, sets `spin_x_face` / `spin_y_face`, and dumps
`frame_buffer` to a PNG — that covers the resize path, which is the one that
fails silently. Before: teapot and floor with a black horizon. After the shader
fixes: the same scene with the floor lit at its actual position and a grey
background. The screenshot in the folder is a straight dump of `frame_buffer`.

## Left alone

`SimpleWebGPU/` has the same cwd-relative shader paths and the same cached
projection matrix. It works when launched the way `RunDemos.py` launches it, so
it isn't broken as such — but it will bite in the same way, and the same two
fixes apply.
