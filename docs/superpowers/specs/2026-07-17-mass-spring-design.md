# Mass spring chain (PyNGL / PySide6) — design

A PyNGL + PySide6 port of the C++ `NGL9Demos/MassSpring` demo, extended so the
single spring becomes a chain of N masses joined by N-1 springs. Lives in a new
top-level demo folder, `MassSpring/`.

## Why the physics has to be restructured

The C++ `RK4Spring` owns its two endpoints and integrates the *relative*
displacement `b - a`, then splits the result across the ends:

```cpp
if(m_aFixed != true) { m_a -= getState().m_velocity; }
if(m_bFixed != true) { m_b += getState().m_velocity; }
```

That is fine for one isolated spring, but it does not chain. In `A—B—C`, mass
`B` belongs to two springs, and each would integrate its own copy of `B` and
write it independently — the two updates fight and the result is not a
simulation of anything.

So the state moves off the spring and onto the masses: the system holds
`positions` and `velocities` for all N masses, and springs become pure force
producers between two mass indices. Every spring touching a mass accumulates
into the same acceleration slot, so shared masses just work.

At N=2 this degenerates to exactly the original demo — two masses, one spring,
either end pinnable. The chain is a generalisation, not a different demo.

## Components

Four source files, split so the physics never imports Qt or OpenGL and can be
tested headlessly.

### `rk4.py` — the integrator (no Qt, no GL)

- `State` — `position` and `velocity`, both `(N, 3)` float arrays. The C++
  `State` with a mass dimension added.
- `AbstractRK4Integrator` — ABC with an abstract
  `motion_function(state, t) -> (N, 3)` returning accelerations, plus
  `evaluate()` and `integrate(t, dt)`. Keeps the C++ a/b/c/d Runge-Kutta
  structure so it still reads like the lecture slide, and stays reusable for
  any motion function.

### `mass_spring.py` — the chain

`MassSpringChain(AbstractRK4Integrator)`: N masses, N-1 springs between
consecutive masses.

`motion_function` walks the springs applying the demo's own force law

```
F = -k(|x| - d)(x / |x|) - b v
```

equal and opposite to each pair, adds gravity when enabled, then zeroes the
acceleration on pinned masses. All masses are 1.0, so force is acceleration.

Owns: `k`, `damping`, `rest_length`, `gravity` / `gravity_strength`,
`num_masses`, `start`, `end`, `fix_first`, `fix_last`, and `reset()` — which
re-lerps the masses between `start` and `end` and zeroes the velocities.

`rest_length` is the rest length of a *single* spring, shared by every spring
in the chain (not the rest length of the whole chain). So a chain of N masses
at rest spans `(N - 1) * rest_length`.

### `MassSpringScene.py` — drawing

A `QOpenGLWidget` (so it embeds in the main layout), following
`GUIDemos/NGLWidgetsOpenGL/PyNGLScene.py`. Per frame:

- a `GL_LINE_STRIP` through the mass positions, via a `VAOFactory` simple VAO
  rebuilt each frame (the `CurveDemos/main.py` pattern)
- a cube per mass, red when pinned, green when free
- ghost spheres at the initial positions, as the original draws

Arcball rotate (`PySideEventHandlingMixin`) and the sim timer live here.

Note: `Prims.CUBE` is a *mesh* default — it must come from
`Primitives.load_default_primitives()`, not `Primitives.create()`, which raises
`ValueError` for it and would abort `initializeGL`.

### `main.py` — the GUI

`QMainWindow`, controls built in code (not a `.ui` file — it avoids the
`QUiLoader` widget-promotion dance).

PyNGL widgets from `ncca.ngl.widgets`:

- `Vec3Widget` **Start (A)** and `Vec3Widget` **End (B)** — the original's
  `m_aX/Y/Z` and `m_bX/Y/Z`. Intermediate masses lerp between them on reset.

Scalars use plain `QDoubleSpinBox`, since `ncca.ngl.widgets` has no scalar
widget: k, damping, rest length, dt, timer interval, gravity strength.

Plus: **Masses** spinbox (2 to 32, default 2 — so the demo opens as the
original single spring), **Fix Start** / **Fix End** checkboxes
(generalising `m_aFixed` / `m_bFixed`), a **Gravity** toggle, **Reset**, and a
**Sim** toggle button.

## Data flow

```
GUI widget signal
  -> MassSpringChain setter (k / damping / positions / masses / gravity ...)
Sim timer tick (in the scene)
  -> chain.update()          # integrate(t, dt), advance t
  -> scene.update()          # repaint from chain.positions
```

The chain is created by `main.py` and shared with the scene, mirroring the C++
`MainWindow` / `NGLScene` split.

## Error handling

The chain validates its own inputs rather than trusting the GUI: `num_masses`
below 2 and a non-positive `dt` are rejected. Changing the mass count rebuilds
the state arrays and resets, so the sim can never index a stale array. A spring
at zero length would divide by zero in `x/|x|`, so the force law guards the
degenerate case and returns no spring force for coincident masses.

## Testing

The physics core is Qt-free and GL-free, so it is all headless unit tests in
`MassSpring/tests/`:

- the RK4 integrator against the analytic simple-harmonic-oscillator solution
  (proves the integration, not merely that it runs)
- pinned masses never move
- a chain already at rest length, gravity off, stays at rest
- damping monotonically reduces total energy
- a top-pinned chain sags under gravity; a free chain falls
- N=2 reproduces the single-spring oscillation of the original
- coincident masses do not produce NaN

Rendering and GUI wiring are covered by `--smoketest`, as elsewhere in this
repo, not by unit tests.

## Picking and dragging a mass

Added after the first version shipped; this section supersedes the original
"no dragging" exclusion below.

Left-press runs a colour-ID pick pass. If it hits a mass you drag that mass;
if it hits the background the camera rotates exactly as before. That resolves
the binding conflict without needing a modifier key.

### Kinematic while held

A held mass is kinematic -- the mouse owns its position and the integrator must
not fight it. That is mechanically the pin path the chain already has, so
rather than invent a second concept the fixed mask grows to include the dragged
index: `motion_function` already zeroes acceleration for masses in the mask and
`_apply_fixed` already clamps them.

Two consequences follow from that reuse:

- **Dragging a pinned mass moves its anchor.** `_apply_fixed` clamps pinned
  masses back to `_initial_positions`, so without this the mass snaps back and
  the drag looks broken.
- **A held free mass has its velocity zeroed every frame.** It is being
  teleported, so otherwise it accumulates momentum and is flung on release.

On release the mass goes free again from a standstill (it does not keep the
mouse's velocity, and it is not left pinned).

### Drag plane

The mass follows the cursor in a plane parallel to the screen passing through
the mass, so grabbing feels direct at any camera angle.

The scene rotates everything by the arcball `global_tx`, so the ray is computed
in *chain space* by folding `global_tx` into the MVP, and the plane normal is
the camera forward pushed through `global_tx` inverted (it is a pure rotation,
so a transpose).

### Components

- `picking.py` -- pure maths, no Qt or GL: `encode_id`/`decode_id` (black is
  reserved for the background), `ray_from_screen` (the tested pattern from
  `RayPickingSelection/picking_maths.py`, copied rather than imported since
  demos here are self-contained) and `intersect_plane`.
- `mass_spring.py` -- `set_dragged(index | None)` and `move_dragged(position)`.
- `MassSpringScene.py` -- the ID pass and the press/move/release routing. The
  held mass draws yellow so it is obvious what is grabbed.

**MSAA turned out to matter more than expected** (corrected after building it;
the original guess here was that blended edge pixels merely decode to garbage,
and that reading a block would be enough).

A `QOpenGLWidget` renders into a *multisampled* framebuffer, and `glReadPixels`
on a multisampled buffer is `GL_INVALID_OPERATION` -- so the ID pass cannot be
drawn over the widget's own target at all. `SelectionManipulator` gets away
with that only because it is a `QOpenGLWindow` drawing to a real surface, and
this demo has to be a widget to sit in the layout next to the panel.

So the ID pass renders into its own single-sample framebuffer, which is the
right target regardless: with no antialiasing, no pixel is ever a blend of two
IDs decoding to a third.

The 9x9 block is kept, but for a different reason than first assumed: the
masses are small on screen, so the pick prefers the pixel directly under the
cursor and falls back to the block only when that is background. That keeps
grabbing forgiving while leaving two nearby masses separable.

### Testing

Headless, since the maths has no Qt in it: ID encode/decode round-trips over
the full index range and rejects black; a ray hits an expected point on a
plane; a ray parallel to the plane returns nothing. Chain side: a dragged mass
does not move under gravity, `move_dragged` puts it exactly where asked,
releasing restores it to free, and dragging a pinned mass carries its anchor.
The pick-pass rendering itself is smoke-tested.

## Out of scope

Per-mass mass values (all 1.0), per-spring parameters, and arbitrary spring
topology (the chain order is fixed).
