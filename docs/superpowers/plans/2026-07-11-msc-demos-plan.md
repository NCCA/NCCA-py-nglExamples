# MSc Demos Implementation Plan — Skeletal Animation, Boids Compute, Marching Cubes, Ray Marching, Gimbal Lock, IBL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Read first:** `docs/superpowers/specs/2026-07-11-new-teaching-demos-design.md`.
Reference implementations: `Blending/`, `OITransparency/`, plus per-task references
named below. Tasks are ordered by (teaching value ÷ risk); IBL is last and explicitly
a stretch goal.

---

## Task 1: SkeletalAnimation (OpenGL)

Linear blend skinning with a dual-quaternion comparison — completely missing from the
repo and reuses the Quaternion/QuatSlerp maths students already know.

**Files:**
- `SkeletalAnimation/main.py`, `SkeletalAnimation/skinning_maths.py` (numpy-only, the
  heart of the demo), `SkeletalAnimation/tests/test_skinning_maths.py`,
  `SkeletalAnimation/shaders/SkinVertex.glsl` + frag, `README.md`.

**No asset files:** build everything procedurally so there is no loader to write:
- Skeleton: a 4-bone chain along +y (`bind_pose(n_bones, bone_length)` → list of bind
  matrices + inverse binds).
- Mesh: a tube (cylinder shell) around the chain, ~16 segments/ring, rings every 0.25 —
  generated in numpy with positions + normals.
- Weights: per-vertex from the two nearest bones, linear falloff, normalised
  (`compute_weights(verts, bone_origins)` → (indices (V,2) int32, weights (V,2) f32)).

**skinning_maths.py API (all tested):**
```python
bind_pose(n, length) -> list[Mat4-as-numpy], list[inverse]
pose_matrices(angles_deg) -> per-bone world matrices (chain FK, explicit Mat4 compose)
skin_lbs(verts, normals, idx, wts, palette) -> skinned verts/normals   # CPU reference
quat_from_mat / dual_quat_from_mat / dlb_blend(dqs, wts) / dq_to_mat    # for DQS
skin_dqs(...) -> skinned verts                                          # CPU reference
```
**Tests:** identity pose = bind mesh; 90° single-joint FK positions; LBS with weight
(1,0) equals rigid transform; the candy-wrapper case — 180° twist of the end bone with
50/50 weights collapses the LBS ring to (near) zero radius while DQS preserves radius
(this asymmetry IS the demo, pin it numerically).

**GPU side:** vertex shader with `uniform mat4 bones[8]` palette + attributes for
indices/weights (locations 3/4, float attribs are fine at this size); `D` toggles an
alternative `uniform vec4 boneDQs[16]` dual-quat path in the same shader (bool uniform
selects). Animate joint angles with sines on a timer; `T` snaps to the candy-wrapper
twist pose; `S` overlays the skeleton as `GL_LINES` (COLOUR shader, depth test off).

**Pitfalls:** normal skinning uses the same palette (fine for rigid-ish bones — note
in README); DQ blending must hemisphere-align (dot < 0 → negate) before summing —
tested; keep vertex count small (~2k) so the CPU reference cross-check in tests stays
instant.

- [ ] skinning_maths + tests green (including candy-wrapper asymmetry)
- [ ] GL demo: LBS/DQS toggle, twist pose, skeleton overlay
- [ ] README (LBS artefact theory, DLB reference: Kavan et al. 2007), root README row (new "Animation" section), smoketest, ruff, commit

---

## Task 2: BoidsCompute (WebGPU)

Classic Reynolds flocking on the GPU: compute shaders + instanced rendering combined.
References: `WebGPUCompute/` demos for compute-pass structure, `VertexArrayObject/Boid`
for the boid shape, the Instancing demo (UG2 plan Task 1) for instanced drawing.

**Files:**
- `BoidsCompute/main.py` (WebGPU is the only backend — name it main.py),
  `BoidsCompute/WebGPUWidget.py`, `BoidsCompute/BoidsCompute.wgsl` (compute) +
  `BoidsRender.wgsl`, `BoidsCompute/boid_maths.py` + `tests/test_boid_maths.py`, `README.md`.

**Simulation:** N boids (default 2048) in a bounded box (soft wall-avoid force, reuse
the approach in `WebGPUCompute/SpatialHash3D` walls). State = two storage buffers
(pos vec4 + vel vec4 each, ping-pong A→B, swap bind groups each frame). Compute pass:
one thread per boid, O(N²) neighbour loop — at 2048 that is 4M pairs, fine on GPU, and
the README points at SpatialHash3D as the optimisation follow-on. Rules: separation
(radius rs), alignment (ra), cohesion (rc), weights as uniforms; clamp speed to
[v_min, v_max]; dt from a uniform.

**boid_maths.py:** a CPU reference `step(positions, velocities, params) -> new state`
implementing the identical rules, vectorised numpy. **Tests:** two converging boids
separate; a lone boid keeps velocity (no rules fire); alignment matches hand-computed
2-boid case; speed clamp respected; symmetric pair produces mirror-image forces.
(The GPU shader is a transcription of this file — say so in comments on both sides.)

**Rendering:** instanced draw of the classic boid arrow (hardcode the small vertex
array from `VertexArrayObject/Boid`), oriented along velocity: build the basis in the
vertex shader from `normalize(vel)` + world-up cross products (guard the parallel
case). Storage buffer read in the *vertex* shader via `@builtin(instance_index)` —
no per-instance vertex buffer, which is itself the teaching point vs the Instancing demo.

**Controls:** `1/2/3` + `+`/`-` adjust separation/alignment/cohesion weights (HUD),
`R` re-seed, `Space` pause simulation (skip compute pass, keep rendering).

**Pitfalls:** storage-buffer usage flags need STORAGE|COPY_DST (+COPY_SRC only if
debugging readback); workgroup size 64 with `ceil(N/64)` dispatch and an `if id >= N`
guard; vec3 in storage buffers has vec4 stride — use vec4 and ignore w (put THIS in the
README, it is the #1 WGSL data bug).

- [ ] boid_maths + tests green
- [ ] Compute + render pipelines working, ping-pong verified (motion coherent, no flicker)
- [ ] Controls + HUD
- [ ] README (rule maths, O(N²)→spatial-hash pointer, vec3-stride warning), root README row under Compute Shaders, smoketest, ruff, commit

---

## Task 3: RayMarchingSDF (OpenGL + WebGPU)

Sphere-traced signed distance fields in a single fragment shader — huge student appeal,
minimal machinery. Reference: `ScreenTri/` for the no-VBO fullscreen setup.

**Files:**
- `RayMarchingSDF/main.py` (GL), `RayMarchingSDF/RayMarchingWebGPU.py` + widget copy,
  `RayMarchingSDF/sdf_maths.py` + `tests/test_sdf_maths.py`,
  `RayMarchingSDF/shaders/RayMarchFragment.glsl` (+ CompositeVertex-style vert),
  `RayMarchingSDF/RayMarch.wgsl`, `README.md`.

**sdf_maths.py:** numpy mirrors of every SDF used in the shaders:
`sd_sphere, sd_box, sd_torus, sd_plane, smooth_min(a,b,k)`, `scene(p)` composition, and
`estimate_normal(p)` central differences. **Tests:** exact distances (sphere surface=0,
inside<0), smooth_min ≤ min and equals min when |a−b|≫k, normal of sphere ≈ radial.
The GLSL/WGSL scenes are line-for-line transcriptions — keep the same function names.

**Shader scene:** plane + sphere smooth-blended with a box + torus, one moving sphere
(time uniform) melting through the others (smooth_min k as uniform). Ray march loop:
100 steps max, epsilon 1e-3, far 40. Shading: normal via gradient, one directional
light, soft shadow (march toward light accumulating penumbra factor), 5-tap AO along
the normal, simple fog by distance. Camera: reuse the standard orbit — pass
`camPos`/`camTarget` derived from the existing mouse state as uniforms and build the
ray basis in-shader (forward/right/up + fov).

**Controls:** `S` shadows, `O` AO, `+`/`-` smooth-min k, `N` visualise normals,
`I` visualise iteration count as heat map (superb for explaining cost), pause `Space`.

**Pitfalls:** iteration heat-map needs the loop counter exported — write it to a
varying-free debug colour branch; WGSL has no `#define` — use `const`; keep both
shaders literally parallel so the README can show them side by side.

- [ ] sdf_maths + tests green
- [ ] GL demo (all toggles), WebGPU demo (same scene)
- [ ] README (sphere tracing explanation, GLSL vs WGSL diff highlights), root README row (new "Ray Marching" or Shading section), smoketests, ruff, commit

---

## Task 4: MarchingCubes (OpenGL)

Scalar fields → meshes: metaballs polygonised on the CPU with vectorised numpy,
uploaded through the ChangingVAO pattern. Reference: `VertexArrayObject/ChangingVAO`
for per-frame VBO updates, `Voxels/` for field/terrain flavour.

**Files:**
- `MarchingCubes/main.py`, `MarchingCubes/marching_cubes.py` (numpy-only algorithm),
  `MarchingCubes/mc_tables.py` (the classic 256-entry edge/tri tables — generate or
  transcribe from Bourke; cite the source in the header),
  `MarchingCubes/tests/test_marching_cubes.py`, plain diffuse shaders, `README.md`.

**marching_cubes.py:**
```python
def sample_metaballs(centres (M,3), radii (M,), grid_n, extent) -> field (n,n,n) f32
def polygonise(field, iso, extent) -> verts (T*3,3) f32, normals (T*3,3) f32
```
Vectorised: classify all cells at once (corner>iso bitmask → case index), gather
edge intersections with linear interpolation, normals from the field gradient
(precompute `np.gradient(field)` and trilinear-sample it at vertex positions — much
better shading than face normals). Target: 48³ grid + 4 metaballs well under 50 ms/frame;
if profiling misses, drop to 32³ — measure, don't guess.

**Tests:** single centred metaball at iso → mesh is closed (every edge shared by
exactly 2 triangles), vertex distances from centre ≈ expected iso radius (tolerance
~cell size), triangle winding consistent (all face-normal · gradient-normal dots > 0);
empty field → zero triangles; case-index sanity for one hand-built corner configuration.

**Demo:** 4 metaballs on lissajous paths (time-driven), re-polygonised per frame,
uploaded with `glBufferData(GL_DYNAMIC_DRAW)` re-specification (the ChangingVAO
lesson); HUD shows grid size, triangle count, polygonise ms. Controls: `+`/`-` grid
resolution (16/32/48/64), `I`/`Shift+I` iso level, `W` wireframe, `Space` pause.

**Pitfalls:** the edge/tri tables are the bug farm — the closed-mesh test catches
transcription errors; keep the tables in their own file so the algorithm file stays
readable; do NOT try the GPU compute version here (note it as a follow-on WebGPU demo).

- [ ] mc_tables + marching_cubes + tests green (closed-mesh test especially)
- [ ] Demo animating at interactive rate, HUD stats
- [ ] README (algorithm walk-through, table provenance), root README row under Geometry & Meshes, smoketest, ruff, commit

---

## Task 5: GimbalLock (OpenGL)

Euler vs quaternion orientation, side by side — the maths-lecture companion to
QuatSlerp. Reference: `QuatSlerp/` for quaternion API usage.

**Files:**
- `GimbalLock/main.py`, `GimbalLock/rotation_maths.py` + `tests/test_rotation_maths.py`,
  `README.md`. `DefaultShader.DIFFUSE`/`COLOUR` only.

**rotation_maths.py:** `euler_to_mat(rx, ry, rz)` (explicit `Rz @ Ry @ Rx` compose —
document the order in the docstring, it IS the lesson), `is_gimbal_locked(ry)` (|cos ry|
< eps), `lost_dof_axis(...)`, and quaternion helpers wrapping `ncca.ngl.Quaternion`
where possible (check the PyNGL source for from-euler/to-matrix; fall back to numpy
implementations mirrored from QuatSlerp if the API lacks them).
**Tests:** euler round-trips for non-degenerate angles; at ry=90° the x and z rotations
produce the same world effect (rank of the combined jacobian drops — test simply: two
different (rx, rz) pairs with rx+rz equal give the same matrix within tolerance);
quaternion path unaffected at the same pose.

**Demo:** split screen via two `glViewport` halves (single window, draw scene twice
with different projection aspect): LEFT = Euler rig — three nested torus rings
(`Prims.TORUS` if present, else numpy torus) coloured RGB for the x/y/z gimbals, each
ring's matrix built by the *nested* composition so students see rings carry rings;
a small aeroplane-ish arrow (cubes) innermost. RIGHT = the same arrow driven by a
quaternion built from the same target angles. Labels via `Text`.

**Controls:** `X/Y/Z` (+Shift) drive the three angles ±5° (HUD shows all three + lock
warning when `is_gimbal_locked`), `G` animate pitch → 90° then try yawing (the canned
"watch the DOF vanish" moment; scripted sequence on a timer), `Space` reset.

**Pitfalls:** the two viewports need scissor as well as viewport for per-half clears
(`glEnable(GL_SCISSOR_TEST)` around the clear); ring nesting order must match
`euler_to_mat` composition order or the visual lies.

- [ ] rotation_maths + tests green (DOF-collapse test)
- [ ] Split-screen demo + scripted lock animation
- [ ] README (gimbal maths, when Euler is still fine), root README row (Maths/Interpolation section), smoketest, ruff, commit

---

## Task 6 (STRETCH): IBL — Image Based Lighting (OpenGL first)

Completes the PBR story (`PBR/SimplePBR`, `PBR/PBRTexture`) with ambient environment
lighting. Depends on SkyBoxEnvMap (intermediate plan Task 1) — do not start before it.
This is the largest task; if time-boxed, ship stages 1–2 only and record the cut.

**Files:**
- `PBR/IBL/main.py`, `PBR/IBL/ibl_precompute.py` + `tests/`, shaders, `README.md`
  (lives inside the existing `PBR/` group).

**Approach — precompute on the CPU in numpy at startup (seconds, cached to .npy next
to the script), avoiding the render-to-cubemap FBO machinery entirely:**
1. Irradiance map: 16² per face cosine-convolved from the SkyBoxEnvMap procedural
   cubemap (reuse `cubemap_gen.py` — import by path or copy, folders stay standalone:
   copy it and note the provenance).
2. Prefiltered specular chain: 64² base, 5 mips, GGX importance-sampled per mip
   roughness (numpy Hammersley + GGX half-vector sampling — port the standard
   LearnOpenGL derivation; cite it).
3. BRDF LUT: 256² RG16F `A/B` split-sum table — pure numpy integral, testable
   (monotonicity in roughness, known corner values, energy ≤ 1).
4. Shader: extend SimplePBR's direct lighting with
   `ambient = kD * irradiance(N) * albedo + prefiltered(R, rough) * (F * A + B)`;
   upload the mip chain with `glTexImage2D` per level + `GL_TEXTURE_CUBE_MAP` trilinear.

**Controls:** roughness/metallic sweep keys (reuse SimplePBR's), `I` toggle IBL vs
direct-only (the money shot), `E` show irradiance/prefilter/LUT debug views.

**Tests:** LUT properties as above; irradiance of a uniform-colour environment equals
that colour (energy sanity); prefilter mip0 ≈ source for roughness 0.

- [ ] SkyBoxEnvMap merged first (hard dependency)
- [ ] ibl_precompute + tests green, .npy caching working
- [ ] Demo with IBL toggle + debug views
- [ ] README (split-sum approximation, Karis 2013 reference), root README row under Textures & Materials/PBR, smoketest, ruff, commit
