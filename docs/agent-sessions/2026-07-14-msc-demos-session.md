# Agent session — 2026-07-14

## Goal

Implement the MSc demos plan
(`docs/superpowers/plans/2026-07-11-msc-demos-plan.md`): six new teaching demos
covering skeletal animation, GPU flocking, ray marching, marching cubes, gimbal
lock, and image-based lighting. Run subagent-driven-development — a fresh
implementer per task, a spec+quality review after each, and a whole-branch
review at the end.

## Approach

Worked in a dedicated worktree `.worktrees/msc-demos` on branch
`agent/msc-demos`, branched from `Version1.0` (581eeef). Each task went out to a
fresh implementer subagent (Sonnet) with a task brief extracted from the plan,
then to a separate reviewer subagent; the final whole-branch review ran on Opus.
Progress tracked in a ledger so the run could survive a compaction. All six
tasks passed spec and quality review; the whole-branch review returned
ready-to-merge with no Critical or Important findings.

The pure maths for every demo lives in a numpy-only module with pytest tests, so
the parts that matter (skinning, boid rules, SDFs, the marching-cubes tables,
Euler/quaternion composition, the BRDF LUT and irradiance integral) are verified
headless. The GLSL/WGSL shaders are deliberate transcriptions of those modules —
the cross-language mirroring is the teaching point.

## Demos added (one commit each)

- **SkeletalAnimation** (OpenGL, `3c9e7e3`) — linear blend skinning vs
  dual-quaternion skinning, with the candy-wrapper twist that collapses the LBS
  ring while DQS keeps it. 13 tests.
- **BoidsCompute** (WebGPU, `bc412d1`) — Reynolds flocking on the GPU: compute
  pass with ping-pong storage buffers plus instanced rendering that reads the
  boid state in the vertex shader. 6 tests.
- **RayMarchingSDF** (OpenGL + WebGPU, `917d2c7`) — sphere-traced SDFs in a
  fullscreen fragment shader, the same scene in both backends. 21 tests.
- **MarchingCubes** (OpenGL, `d0a5ebe`) — metaballs polygonised with vectorised
  numpy and uploaded per frame through the ChangingVAO pattern; 48³ grid runs at
  ~16 ms. 9 tests.
- **GimbalLock** (OpenGL, `46ebd58`) — split-screen Euler rig vs quaternion,
  with the scripted lock animation. 16 tests.
- **PBR/IBL** (OpenGL, stretch, `576d4fd`) — irradiance map and split-sum BRDF
  LUT precomputed in numpy and cached to `.npy`. The prefiltered specular mip
  chain was cut for time (the plan allowed shipping a subset); the shader
  approximates it with an unblurred sample and the README says so. 7 tests.

## Files changed

49 files, ~7800 insertions across the six demo folders plus the root
`README.md` (new Animation, Compute Shaders, Ray Marching, Geometry & Meshes,
and Curves & Interpolation rows, and the IBL row under PBR). Each demo folder is
self-contained — its own `README.md`, preview `.png`, numpy maths module,
`tests/`, shaders, and (for the WebGPU demos) its own copy of `WebGPUWidget.py`.
`PBR/IBL/cubemap_gen.py` is a copy of the SkyBoxEnvMap one with provenance noted.

Jon added the six preview screenshots by hand (agents can't capture them
headless) and made small tweaks to BoidsCompute, RayMarchingSDF and
SkeletalAnimation — commits `221b667`, `841112e`, `9253be2`, `dc04920`.

## Commands run

- `uv run pytest` across all six `tests/` folders — 72 passed.
- `uv run ruff check` / `ruff format --check` — clean (23 files).
- WebGPU smoketests ran headless and passed:
  `QT_QPA_PLATFORM=offscreen uv run BoidsCompute/main.py --smoketest` and the
  RayMarchingSDF WebGPU one both print `SMOKETEST OK`.
- OpenGL smoketests could not run: `QT_QPA_PLATFORM=offscreen` segfaults for
  every GL demo in this sandbox, confirmed identical on the pre-existing
  `Blending` reference — an environment limit, not a demo bug. The `--smoketest`
  code paths were verified by reading them.
- Merged into `Version1.0` with `git merge --no-ff agent/msc-demos` (`6d035f6`),
  re-ran the suite on the merged tree (72 passed), then removed the worktree and
  deleted the branch.

## Follow-ups (non-blocking)

- BoidsCompute: the "nose" comments in `main.py` and `BoidsRender.wgsl` are stale
  after Jon's z-flip in `221b667` — worth reconciling against the live render.
- SkeletalAnimation: skeleton overlay uses `GL_LINE_STRIP` not `GL_LINES`
  (identical for a linear chain); bone-palette uniform locations are looked up
  each frame with no `-1` guard. All cosmetic.
