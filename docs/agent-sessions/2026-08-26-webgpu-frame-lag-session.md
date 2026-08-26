# 2026-08-26 -- ViewToWorldTransform WebGPU click lag

## Goal

Shift-click in `ViewToWorldTransform/main_webgpu.py` places a cube at the
unprojected point, and the cube did not turn up until the next click. Find out
why. The OpenGL `main.py` was fine.

## What was wrong

Nothing in this repo, as it turned out. `view_to_world.unproject_point` was
placing the cube in exactly the right spot — it just wasn't being shown.

`WebGPUWidget._update_colour_buffer` in PyNGL read the rendered frame back
through a pipelined ring of two buffers: it copies the current frame into one
buffer and maps the *other*, which holds the previous frame. Mapping a copy the
GPU has had a whole frame to finish returns straight away, so the CPU never
stalls — but what reaches the widget is always frame N-1.

Under continuous repaints you never notice. This demo repaints only in response
to events, so the click's own paint presented the pre-click image, and the cube
appeared on whatever event happened to trigger the next paint. Same story for
`Space` to clear, and for the last frame of an orbit or pan.

A headless probe that paints, counts non-background pixels and paints again:

```
paint 3 (1st after click)    non-background pixels = 0
paint 4 (2nd after click)    non-background pixels = 10816
```

About forty of the forty-six WebGPU demos here are event-driven and were all
quietly a frame behind. It only became visible in this one because the whole
point of the demo is that a single discrete click changes the picture.

## Changes

None in this repo. The fix is in PyNGL on branch `agent/webgpu-sync-readback`
(`fix: present the frame just drawn, not the previous one`) — a new
`pipelined_readback` flag on `WebGPUWidget`, defaulting to off, which maps the
buffer just written instead of the other one. See that repo's
`docs/agent-sessions/2026-08-26-webgpu-sync-readback-session.md`.

**This demo only behaves once that branch is merged into PyNGL's main**, since
`[tool.uv.sources]` pins `ncca-ngl` to the main checkout at
`/Users/jmacey/teaching/Code/PyNGL`. To try it before then:

```bash
PYTHONPATH=/Users/jmacey/teaching/Code/PyNGL/.worktrees/webgpu-sync-readback/src \
  uv run ViewToWorldTransform/main_webgpu.py
```

I looked at fixing it demo-side instead — either an update timer, or repainting
twice per event — and decided against both. The lag affects nearly every
event-driven demo here, so fixing it in forty demos to leave the library wrong
is the wrong way round.

No demo needed to opt back into pipelining either. The timer-driven demos render
trivial scenes far above 60fps, the synchronous path costs about 1.3ms a frame,
and several of them can pause their animation from the keyboard — which would
put them straight back to showing a stale frame.

## Commands

```bash
# probes and benchmarks were throwaway scripts against a headless WebGPUScene
PYTHONPATH=.../.worktrees/webgpu-sync-readback/src uv run <demo> --smoketest 400
```

`ViewToWorldTransform`, `SimpleWebGPU`, `WebGPUShadows`, `Instancing`,
`ShadedGrid` and `Collisions/SphereSphere` all smoketest green against the
branch. `SimpleWebGPU` and `WebGPUShadows` have to be run from inside their own
folders — they open their `.wgsl` by relative path, which is pre-existing and
nothing to do with this.
