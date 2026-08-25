# 2026-07-15 — Fix initial camera jump on arrow-key movement

## Goal

The user noticed that moving the `FirstPersonCamera` with the arrow keys in
`PBR/PBRTexture/main.py` gave a large jump on the first press. Find the cause,
fix it, and apply the same fix across every demo sharing the pattern.

## Diagnosis

Movement is scaled by `delta_time = current_frame - last_frame`. In
`PBR/PBRTexture/main.py`, `last_frame` was only updated *inside* the block that
runs when a key is held, so between presses it went stale. The first press
after startup (or any idle gap) produced a `delta_time` spanning the whole gap,
which `camera.move()` multiplied into a big jump.

Survey of the other demos using this scheme:

- `Voxels/main.py` — advanced `last_frame` every frame but never clamped, so the
  first press after an idle gap (no frames drawn while idle) still jumped.
- `WebGPUMultiGeo/WebGPUMultiGeo.py`, `WebGPUMultiGeo/WebGPUMultiGeo_updated.py`,
  `WebGPUShadows/PCFShadows.py` — already advanced every frame and clamped
  `min(dt, 0.05)`. No change needed.
- `Particles/ParticleQuads/Emitter.py` — uses `time.time()` for the particle
  sim, unrelated to camera movement. No change needed.

## Fix

Advance the frame clock on every call and clamp the delta to `0.05s`, matching
the WebGPU demos.

## Files changed

- `PBR/PBRTexture/main.py` — moved the timing update out of the movement block
  and clamped `delta_time`.
- `Voxels/main.py` — clamped `delta_time`.

## Commands run

```
git worktree add .worktrees/camera-timing-jump -b agent/camera-timing-jump
uv run PBR/PBRTexture/main.py --smoketest      # SMOKETEST OK
uv run Voxels/main.py --smoketest              # SMOKETEST OK
uv run ruff check PBR/PBRTexture/main.py Voxels/main.py    # All checks passed
uv run ruff format --check PBR/PBRTexture/main.py Voxels/main.py   # already formatted
```

## Notes

Both smoketests print a pre-existing first-frame `AttributeError`
(`light_positions` / `render_fbo` accessed before `initializeGL` completes).
Confirmed present on the committed baseline too — unrelated to this change.
