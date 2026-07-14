# BoidsCompute

![](BoidsCompute.png)

Classic Reynolds flocking — separation, alignment, cohesion — run entirely on
the GPU. 2048 boids by default, one compute thread each, ping-ponging their
position and velocity through a pair of storage buffers every frame. The
rendering half is instanced drawing taken to its logical extreme: there is no
per-instance vertex buffer at all, the vertex shader just reads the same
storage buffers the compute pass wrote straight out of `@builtin(instance_index)`.

- `boid_maths.py` — the rules, in plain numpy, developed test-first
- `BoidsCompute.wgsl` — the same rules, transcribed onto the GPU, one thread per boid
- `BoidsRender.wgsl` — instanced arrow rendering, storage-buffer read in the vertex shader
- `main.py` — WebGPU only for this one; wires the two shaders together and owns the ping-pong bookkeeping

## The rules

For boid *i*, every other boid *j* is tested against three radii and folded
into an acceleration:

- **Separation** (radius `rs`): average of `(pos_i - pos_j) / |pos_i - pos_j|`
  over every `j` closer than `rs` — steer directly away from anyone too close.
- **Alignment** (radius `ra`): average velocity of every `j` within `ra`,
  minus boid *i*'s own velocity — steer towards the local average heading.
- **Cohesion** (radius `rc`): average position of every `j` within `rc`, minus
  boid *i*'s own position — steer towards the local centre of mass.

Each term is weighted (`w_sep`, `w_align`, `w_coh`), summed with a soft wall
push (see below), and integrated: `vel += accel * dt`, then speed is clamped
to `[v_min, v_max]` before `pos += vel * dt`. That is the entirety of
`step()` in `boid_maths.py`, and `BoidsCompute.wgsl` follows it almost line
for line — each side's comments point at the other, so if you change one,
change both.

This is the naive O(N²) version: every boid scans every other boid, which at
N=2048 is around four million pairs a frame — trivial for a GPU, hopeless for
a CPU `for` loop. It does not scale forever, though; once you want tens of
thousands of boids, bucket them into a grid first so each boid only scans its
own neighbourhood. `WebGPUCompute/SpatialHash3D` builds exactly that grid
(for particle collisions rather than flocking, but the bucketing is the same
idea) — start there if you want to push this demo further.

## The #1 WGSL data bug

Storage buffers pad a `vec3<f32>` to 16 bytes (the stride of a `vec4`), not
12. If you declare `array<vec3<f32>>` in the shader and upload a tightly
packed numpy array of 3-float positions, the second boid's data lands 4 bytes
short of where the shader expects it, and every boid after that reads a
scrambled mix of its neighbour's fields. Both position and velocity buffers
here are `array<vec4<f32>>` on the GPU and `(N, 4)` float32 arrays on the
Python side, with `.w` set to zero and ignored everywhere. Match strides
first, worry about elegance never.

## Ping-pong state

There are two storage buffers each for position and velocity (call them set 0
and set 1). Every frame the compute pass reads whichever set currently holds
valid state and writes the *other* set — a boid must never read a value some
other invocation in the same dispatch has already overwritten, which is
exactly what would happen if a single buffer were read and written in place.
`main.py` tracks this with one integer, `self.current`, and two
pre-built bind groups per pipeline (`compute_bind_groups`, `render_bind_groups`)
so nothing is rebuilt per frame — only which bind group gets bound changes.

## Controls

| Key | Action |
| :--- | :--- |
| `1` / `2` / `3` | select separation / alignment / cohesion as the weight to edit |
| `+` / `-` | increase / decrease the selected weight (shown on the HUD) |
| `R` | re-seed: new random positions and velocities |
| `Space` | pause the simulation (rendering keeps running on the last computed state) |
| LMB / RMB / wheel | rotate / pan / zoom |
| `Esc` | quit |

## Tests

```bash
uv run pytest BoidsCompute/tests
```

Covers: a lone boid with no neighbours keeps its velocity unchanged; two
boids converging head-on end up steered apart; a hand-computed two-boid
alignment case (with `dt=1` the two boids simply swap headings); speed
clamping at both `v_min` and `v_max`; and a symmetric pair producing
mirror-image forces (no hidden left/right bias in the maths).

## References

- C. Reynolds, "Flocks, Herds, and Schools: A Distributed Behavioral Model",
  SIGGRAPH 1987 — [PDF](https://www.red3d.com/cwr/papers/1987/boids.html) — the original paper.
- [Craig Reynolds — Boids (background and pseudocode)](https://www.red3d.com/cwr/boids/) — the canonical reference for the three rules.
- `WebGPUCompute/SpatialHash3D` — the grid-bucketed alternative to this demo's O(N²) neighbour search.
- `Instancing` — per-instance *vertex buffer* instancing, the pattern this demo deliberately does not use for its render pass.
