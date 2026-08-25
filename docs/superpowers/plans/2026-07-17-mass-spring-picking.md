# Mass Spring Picking and Dragging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user grab a mass with the mouse and move it, picked by colour ID.

**Architecture:** A colour-ID pass renders each mass flat in a unique colour; `glReadPixels` around the cursor decodes the index. A held mass is kinematic, reusing the chain's existing fixed mask rather than adding a second concept. The drag happens in world space, where the plane normal is a constant because the camera is fixed.

**Tech Stack:** Python 3, `uv`, PySide6, `ncca.ngl`, `ncca.ngl.opengl`, numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-17-mass-spring-design.md` (section "Picking and dragging a mass")

## Global Constraints

- Work in the existing worktree `.worktrees/mass-spring` on branch `agent/mass-spring`. Never commit to `Version1.0`.
- All commands via `uv`. Lint with `uv run ruff check MassSpring/` and `uv run ruff format MassSpring/` before every commit; both must pass.
- `picking.py` and `mass_spring.py` MUST NOT import PySide6 or OpenGL — their tests are headless.
- Tests live in `MassSpring/tests/` and start with `sys.path.insert(0, str(Path(__file__).parent.parent))`. **Do NOT add `MassSpring/tests/__init__.py`** — it creates a second Python package named `tests` that collides with `PBR/HDRIBaker/tests` and breaks collection of that whole folder (fixed in `fbf35c9`; do not reintroduce).
- Verify with the FULL suite (`uv run pytest` from the worktree root), not just `MassSpring/tests` — the collision above passed the demo's own tests and only the full run caught it. Expect **334 passed** before this work.
- Prose uses the jon-writing-style skill: first person, British English, no marketing adjectives, no emoji.
- Run GL demos from their own folder (`cd MassSpring && uv run main.py`).

### Traps, verified

1. **MSAA is on** (`glEnable(GL_MULTISAMPLE)` plus a 4-sample surface format). Blended edge pixels in the ID pass decode to garbage indices. Read a 9x9 block and accept only exact colour matches, like `SelectionManipulator/main.py`.
2. **Matrix convention.** PyNGL `Mat4.to_numpy()` works with row-vector multiplies: `ray_from_screen` in `RayPickingSelection/picking_maths.py` does `np.array([ndc_x, ndc_y, -1, 1]) @ inverse`. Follow that exactly; do not transpose.
3. **The mixin grabs LMB.** `PySideEventHandlingMixin.mousePressEvent` sets `self.rotate = True` on left press. The override must pick first and `return` without calling `super()` when a mass is hit, or the camera rotates while you drag.
4. `Prims.CUBE` is a mesh default — `Primitives.load_default_primitives()`, never `Primitives.create()`.

---

### Task 1: Picking maths

**Files:**
- Create: `MassSpring/picking.py`
- Test: `MassSpring/tests/test_picking.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `encode_id(index: int) -> tuple[int, int, int]` — index 0 becomes colour (1,0,0); black is reserved for background.
  - `decode_id(pixel) -> int | None` — inverse; returns `None` for black.
  - `ray_from_screen(x, y, width, height, mvp: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(origin, direction)`, direction normalised, in whatever space `mvp` transforms from.
  - `intersect_plane(origin, direction, point, normal) -> np.ndarray | None` — `None` when the ray is parallel to the plane.
  - `transform_point(p: np.ndarray, mat: np.ndarray) -> np.ndarray` — row-vector convention, perspective divide applied.

- [ ] **Step 1: Write the failing test**

Create `MassSpring/tests/test_picking.py`:

```python
"""Tests for the colour-picking and drag maths.

None of this touches Qt or OpenGL, so it can all be checked headlessly --
which matters, because getting an unproject subtly wrong produces a drag
that merely feels a bit off rather than something that crashes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Vec3, look_at, perspective  # noqa: E402
from picking import (  # noqa: E402
    decode_id,
    encode_id,
    intersect_plane,
    ray_from_screen,
    transform_point,
)


def test_ids_round_trip_over_the_whole_range():
    for index in range(0, 64):
        assert decode_id(encode_id(index)) == index


def test_black_is_the_background_not_an_index():
    assert decode_id((0, 0, 0)) is None
    # so no index may ever encode to black
    assert encode_id(0) != (0, 0, 0)


def test_ids_stay_inside_a_byte_per_channel():
    for index in range(0, 64):
        assert all(0 <= c <= 255 for c in encode_id(index))


def test_a_ray_down_the_screen_centre_points_along_the_view():
    view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
    project = perspective(45.0, 1.0, 0.5, 150.0)
    mvp = (project @ view).to_numpy()
    origin, direction = ray_from_screen(50.0, 50.0, 100, 100, mvp)
    # camera sits at +7z looking at the origin, so the centre ray runs -z
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0], atol=1e-5)
    assert origin[2] < 7.0


def test_the_centre_ray_hits_the_origin_plane_at_the_origin():
    view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
    project = perspective(45.0, 1.0, 0.5, 150.0)
    mvp = (project @ view).to_numpy()
    origin, direction = ray_from_screen(50.0, 50.0, 100, 100, mvp)
    hit = intersect_plane(origin, direction, np.zeros(3), np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(hit, [0.0, 0.0, 0.0], atol=1e-4)


def test_a_ray_above_centre_hits_the_plane_higher_up():
    view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
    project = perspective(45.0, 1.0, 0.5, 150.0)
    mvp = (project @ view).to_numpy()
    # y=25 is above the centre in Qt's top-left pixel origin
    origin, direction = ray_from_screen(50.0, 25.0, 100, 100, mvp)
    hit = intersect_plane(origin, direction, np.zeros(3), np.array([0.0, 0.0, -1.0]))
    assert hit[1] > 0.1
    assert abs(hit[0]) < 1e-4


def test_a_ray_parallel_to_the_plane_misses():
    hit = intersect_plane(
        np.array([0.0, 0.0, 5.0]),
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -1.0]),
    )
    assert hit is None


def test_transform_point_round_trips_through_a_matrix_and_its_inverse():
    from ncca.ngl import Mat4

    m = Mat4().rotate_y(35.0) @ Mat4().rotate_x(20.0)
    mat = m.to_numpy()
    p = np.array([1.0, -2.0, 0.5])
    there = transform_point(p, mat)
    back = transform_point(there, np.linalg.inv(mat))
    np.testing.assert_allclose(back, p, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_picking.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'picking'`

- [ ] **Step 3: Write minimal implementation**

Create `MassSpring/picking.py`:

```python
"""Colour-ID picking and the maths for dragging a mass.

Nothing here touches Qt or OpenGL so it can be tested headlessly. The
unproject follows RayPickingSelection/picking_maths.py -- copied rather than
imported, because the demos in this repo are meant to stand alone.
"""

import numpy as np

# The ID pass clears to black, so black has to mean "nothing here". Shifting
# every index up by one keeps index 0 pickable.
_ID_OFFSET = 1

# A ray whose direction is this close to perpendicular to the plane normal is
# parallel to the plane for our purposes, and never meets it.
_PARALLEL_EPSILON = 1e-8


def encode_id(index: int) -> tuple[int, int, int]:
    """Turn a mass index into a unique RGB colour for the ID pass."""
    value = index + _ID_OFFSET
    return (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF)


def decode_id(pixel) -> int | None:
    """Turn a pixel from the ID pass back into a mass index, or None for the
    background."""
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    value = r | (g << 8) | (b << 16)
    if value < _ID_OFFSET:
        return None
    return value - _ID_OFFSET


def ray_from_screen(
    x: float, y: float, width: int, height: int, mvp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a pixel position into a ray in the space `mvp` transforms from.

    x, y are pixel coordinates with Qt's top-left origin. Returns
    (origin, direction) with direction normalised.
    """
    ndc_x = 2.0 * x / width - 1.0
    ndc_y = 1.0 - 2.0 * y / height  # flip: NDC y is up, pixel y is down
    inverse = np.linalg.inv(mvp.astype(np.float64))
    # OpenGL NDC z runs -1 (near) to +1 (far)
    near = np.array([ndc_x, ndc_y, -1.0, 1.0]) @ inverse
    far = np.array([ndc_x, ndc_y, 1.0, 1.0]) @ inverse
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    direction = far - near
    direction /= np.linalg.norm(direction)
    return near, direction


def intersect_plane(
    origin: np.ndarray, direction: np.ndarray, point: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """Where a ray meets a plane, or None if it runs parallel to it."""
    denominator = float(np.dot(direction, normal))
    if abs(denominator) < _PARALLEL_EPSILON:
        return None
    t = float(np.dot(point - origin, normal)) / denominator
    return origin + direction * t


def transform_point(p: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Transform a point by a 4x4, row-vector convention, dividing through by w."""
    v = np.array([p[0], p[1], p[2], 1.0]) @ mat
    return v[:3] / v[3]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_picking.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/picking.py MassSpring/tests/test_picking.py
git commit -m "feat(mass-spring): add the colour-picking and drag maths"
```

---

### Task 2: Chain drag support

**Files:**
- Modify: `MassSpring/mass_spring.py`
- Test: `MassSpring/tests/test_mass_spring.py` (append)

**Interfaces:**
- Consumes: the existing `MassSpringChain`.
- Produces:
  - `set_dragged(index: int | None) -> None` — mark a mass kinematic, or release it. Raises `ValueError` for an out-of-range index.
  - `move_dragged(position: np.ndarray) -> None` — put the held mass exactly there. No-op if nothing is held.
  - `dragged -> int | None` property.

- [ ] **Step 1: Write the failing test**

Append to `MassSpring/tests/test_mass_spring.py`:

```python
def test_a_dragged_mass_does_not_fall():
    chain = MassSpringChain(start=Vec3(0, 2, 0), end=Vec3(0, -2, 0), num_masses=3)
    chain.set_gravity(True)
    chain.set_dragged(1)
    held = chain.positions[1].copy()
    _settled(chain, 50)
    np.testing.assert_allclose(chain.positions[1], held, atol=1e-9)


def test_move_dragged_puts_the_mass_exactly_where_asked():
    chain = MassSpringChain(num_masses=3)
    chain.set_dragged(1)
    chain.move_dragged(np.array([1.5, -0.5, 0.25]))
    np.testing.assert_allclose(chain.positions[1], [1.5, -0.5, 0.25], atol=1e-9)
    chain.update()
    # still exactly there after a step, because it is kinematic while held
    np.testing.assert_allclose(chain.positions[1], [1.5, -0.5, 0.25], atol=1e-9)


def test_releasing_a_dragged_mass_lets_it_move_again():
    chain = MassSpringChain(start=Vec3(0, 2, 0), end=Vec3(0, -2, 0), num_masses=3)
    chain.set_gravity(True)
    chain.set_dragged(1)
    _settled(chain, 10)
    chain.set_dragged(None)
    assert chain.dragged is None
    held = chain.positions[1].copy()
    _settled(chain, 20)
    assert chain.positions[1][1] < held[1]


def test_dragging_a_pinned_mass_carries_its_anchor():
    # without this the pinned mass snaps straight back to its old anchor
    chain = MassSpringChain(start=Vec3(0, 2, 0), end=Vec3(0, -2, 0), num_masses=3)
    chain.set_fix_first(True)
    chain.set_dragged(0)
    chain.move_dragged(np.array([1.0, 2.0, 0.0]))
    _settled(chain, 20)
    np.testing.assert_allclose(chain.positions[0], [1.0, 2.0, 0.0], atol=1e-9)
    chain.set_dragged(None)
    _settled(chain, 20)
    # still pinned, and pinned at the new spot
    np.testing.assert_allclose(chain.positions[0], [1.0, 2.0, 0.0], atol=1e-9)


def test_a_held_free_mass_does_not_bank_momentum():
    # it is being teleported, so it must not accumulate velocity and fling
    # itself the moment it is released
    chain = MassSpringChain(start=Vec3(0, 2, 0), end=Vec3(0, -2, 0), num_masses=3)
    chain.set_gravity(True)
    chain.set_dragged(1)
    for step in range(20):
        chain.move_dragged(np.array([step * 0.05, 1.0, 0.0]))
        chain.update()
    np.testing.assert_allclose(chain.state.velocity[1], [0.0, 0.0, 0.0], atol=1e-9)


def test_rejects_dragging_a_mass_that_does_not_exist():
    chain = MassSpringChain(num_masses=3)
    with pytest.raises(ValueError):
        chain.set_dragged(3)


def test_changing_the_mass_count_drops_the_drag():
    # the state arrays are rebuilt, so a held index could dangle
    chain = MassSpringChain(num_masses=3)
    chain.set_dragged(2)
    chain.set_num_masses(2)
    assert chain.dragged is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/test_mass_spring.py -q`
Expected: FAIL — `AttributeError: 'MassSpringChain' object has no attribute 'set_dragged'`

- [ ] **Step 3: Write the implementation**

In `MassSpring/mass_spring.py`, add `self._dragged = None` in `__init__` immediately before the `self.reset()` call:

```python
        self._fix_first = False
        self._fix_last = False
        self._dragged = None
        self._t = 0.0
        self.reset()
```

Add the accessor next to `num_masses`:

```python
    @property
    def dragged(self) -> int | None:
        """Index of the mass the mouse is holding, or None."""
        return self._dragged
```

Add the mutators after `set_fix_last`:

```python
def set_dragged(self, index: int | None) -> None:
    """Mark a mass as held by the mouse, or pass None to let go.

    A held mass is kinematic: it goes into the fixed mask so the
    integrator leaves it alone and the mouse can place it freely.
    """
    if index is not None and not 0 <= index < self._num_masses:
        raise ValueError(f"no mass {index} in a chain of {self._num_masses}")
    self._dragged = index
    self._rebuild_fixed_mask()


def move_dragged(self, position: np.ndarray) -> None:
    """Put the held mass exactly at `position`. Does nothing if nothing is held.

    The anchor moves too: a pinned mass is clamped back to its anchor by
    _apply_fixed, so without this dragging one would snap straight back.
    """
    if self._dragged is None:
        return
    self.state.position[self._dragged] = position
    self.state.velocity[self._dragged] = 0.0
    self._initial_positions[self._dragged] = position
```

Replace `_rebuild_fixed_mask` so the held mass joins the mask:

```python
    def _rebuild_fixed_mask(self) -> None:
        mask = np.zeros(self._num_masses, dtype=bool)
        mask[0] = self._fix_first
        mask[-1] = self._fix_last
        # a held mass is kinematic, which is the same thing as pinned as far
        # as the integrator is concerned
        if self._dragged is not None:
            mask[self._dragged] = True
        self._fixed_mask = mask
```

In `reset()`, drop any drag before rebuilding the mask, since the arrays change shape. Replace the tail of `reset()`:

```python
        self._initial_positions = positions.copy()
        self._t = 0.0
        self._dragged = None
        self._rebuild_fixed_mask()
```

- [ ] **Step 4: Run the full suite**

Run: `cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring && uv run pytest MassSpring/tests/ -q`
Expected: PASS, 31 passed (17 + 7 picking + 7 here)

- [ ] **Step 5: Lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/mass_spring.py MassSpring/tests/test_mass_spring.py
git commit -m "feat(mass-spring): let a mass be held kinematic by the mouse"
```

---

### Task 3: The pick pass and drag routing

**Files:**
- Modify: `MassSpring/MassSpringScene.py`

**Interfaces:**
- Consumes: `picking.encode_id/decode_id/ray_from_screen/intersect_plane/transform_point` (Task 1); `chain.set_dragged/move_dragged/dragged` (Task 2).
- Produces: no new public API — mouse behaviour only.

No unit test: this is GL and event routing, smoke-tested as per repo convention.

- [ ] **Step 1: Add the imports and the held colour**

In `MassSpringScene.py` add to the imports:

```python
from picking import (
    decode_id,
    encode_id,
    intersect_plane,
    ray_from_screen,
    transform_point,
)
from PySide6.QtCore import Qt, Slot
```

(the existing line is `from PySide6.QtCore import Slot` — replace it with the above)

and next to the other colours:

```python
_HELD_COLOUR = (1.0, 1.0, 0.0, 1.0)
# The ID pass is read back as a block rather than a single pixel: MSAA blends
# edge pixels into colours that decode to nonsense, so we sample around the
# cursor and take the first exact match.
_PICK_BLOCK = 9
```

- [ ] **Step 2: Add the drag state**

In `__init__`, after `self._timer_id = None`:

```python
        # index of the mass the mouse is holding, plus the world-space plane
        # it is being dragged in (set on press)
        self._drag_plane_point = None
```

- [ ] **Step 3: Add the pick pass and drag helpers**

Add these methods to `MassSpringScene`, after `_draw_chain_line`:

```python
# -------------------------------------------------------------- picking
def _world_mvp(self) -> Mat4:
    """Projection @ view, with no arcball -- the space the drag happens in."""
    return self.project @ self.view


def _pick(self, x: float, y: float) -> int | None:
    """Render every mass flat in its ID colour and read back which one is
    under the cursor. x, y are device pixels, Qt's top-left origin."""
    self.makeCurrent()
    gl.glViewport(0, 0, self.window_width, self.window_height)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # black means nothing
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    global_tx = self._mouse_global_tx()
    ShaderLib.use(DefaultShader.COLOUR)
    for i in range(self.chain.num_masses):
        r, g, b = encode_id(i)
        self.transform.reset()
        self.transform.set_scale(_MASS_SCALE, _MASS_SCALE, _MASS_SCALE)
        p = self.chain.positions[i]
        self.transform.set_position(float(p[0]), float(p[1]), float(p[2]))
        mvp = self.project @ self.view @ global_tx @ self.transform.matrix()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("Colour", r / 255.0, g / 255.0, b / 255.0, 1.0)
        Primitives.draw("cube")
    gl.glClearColor(0.4, 0.4, 0.4, 1.0)

    half = _PICK_BLOCK // 2
    read_x = max(0, int(x) - half)
    read_y = max(0, self.window_height - int(y) - half)
    data = gl.glReadPixels(
        read_x, read_y, _PICK_BLOCK, _PICK_BLOCK, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
    )
    pixels = [tuple(data[i : i + 3]) for i in range(0, len(data), 3)]
    for pixel in pixels:
        index = decode_id(pixel)
        if index is not None and index < self.chain.num_masses:
            return index
    return None


def _drag_to(self, x: float, y: float) -> None:
    """Slide the held mass along its drag plane to follow the cursor."""
    if self.chain.dragged is None or self._drag_plane_point is None:
        return
    origin, direction = ray_from_screen(
        x, y, self.window_width, self.window_height, self._world_mvp().to_numpy()
    )
    # the camera is fixed looking down -z, so the screen-parallel plane
    # has this normal in world space whatever the arcball is doing
    hit = intersect_plane(
        origin, direction, self._drag_plane_point, np.array([0.0, 0.0, -1.0])
    )
    if hit is None:
        return
    # back out of the arcball into the chain's own space
    global_np = self._mouse_global_tx().to_numpy()
    self.chain.move_dragged(transform_point(hit, np.linalg.inv(global_np)))
    self.update()
```

- [ ] **Step 4: Route the mouse**

Add these overrides at the end of the class:

```python
# ---------------------------------------------------------------- mouse
def mousePressEvent(self, event) -> None:
    """Left-press grabs a mass if one is under the cursor, otherwise it
    falls through to the mixin and rotates the camera."""
    if event.button() == Qt.LeftButton:
        dpr = self.devicePixelRatio()
        position = event.position()
        index = self._pick(position.x() * dpr, position.y() * dpr)
        if index is not None:
            self.chain.set_dragged(index)
            # the drag plane passes through the mass, in world space
            self._drag_plane_point = transform_point(
                self.chain.positions[index], self._mouse_global_tx().to_numpy()
            )
            self.update()
            return  # do NOT let the mixin start a camera rotate too
    super().mousePressEvent(event)


def mouseMoveEvent(self, event) -> None:
    if self.chain.dragged is not None:
        dpr = self.devicePixelRatio()
        position = event.position()
        self._drag_to(position.x() * dpr, position.y() * dpr)
        return
    super().mouseMoveEvent(event)


def mouseReleaseEvent(self, event) -> None:
    if event.button() == Qt.LeftButton and self.chain.dragged is not None:
        self.chain.set_dragged(None)
        self._drag_plane_point = None
        self.update()
        return
    super().mouseReleaseEvent(event)
```

- [ ] **Step 5: Draw the held mass in its own colour**

In `paintGL`, replace the mass-drawing loop with one that highlights the held mass:

```python
        for i in range(self.chain.num_masses):
            if i == self.chain.dragged:
                colour = _HELD_COLOUR
            elif self.chain.is_fixed(i):
                colour = _FIXED_COLOUR
            else:
                colour = _FREE_COLOUR
            self._draw_mass(self.chain.positions[i], colour, "cube", global_tx)
```

Note `is_fixed` is True for a held mass too (it is kinematic), so the `dragged` check has to come first or the held mass shows red.

- [ ] **Step 6: Smoke-test**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring
uv run main.py --smoketest 600
```
Expected: `SMOKETEST OK`, no traceback.

- [ ] **Step 7: Drive a real pick and drag offscreen**

Rendering is smoke-tested here, but the pick pass is the one bit that can silently return `None` forever, so exercise it once for real. Write `/tmp/drag_check.py`:

```python
"""Throwaway: prove the ID pass finds a mass and the drag moves it."""

import sys

from main import MainWindow
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QSurfaceFormat
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
fmt = QSurfaceFormat()
fmt.setSamples(4)
fmt.setMajorVersion(4)
fmt.setMinorVersion(1)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)

win = MainWindow()
win.sim_button.setChecked(False)  # freeze so nothing moves under us
win.show()


def check():
    scene = win.scene
    dpr = scene.devicePixelRatio()
    # project the free end (mass 1) to a pixel and pick exactly there
    from picking import transform_point

    mvp = (scene.project @ scene.view @ scene._mouse_global_tx()).to_numpy()
    ndc = transform_point(scene.chain.positions[1], mvp)
    px = (ndc[0] * 0.5 + 0.5) * scene.window_width
    py = (1.0 - (ndc[1] * 0.5 + 0.5)) * scene.window_height
    index = scene._pick(px, py)
    print("picked index:", index, "(expected 1)")
    assert index == 1, "colour pick failed to find the mass"

    # now press there, drag right, release
    pos = QPointF(px / dpr, py / dpr)
    scene.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert scene.chain.dragged == 1, "press did not grab the mass"
    assert not scene.rotate, "press also started a camera rotate"
    before = scene.chain.positions[1].copy()
    moved = QPointF(pos.x() + 120, pos.y())
    scene.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            moved,
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    after = scene.chain.positions[1].copy()
    print("mass moved", before, "->", after)
    assert after[0] > before[0] + 0.1, "drag did not move the mass right"
    scene.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            moved,
            Qt.LeftButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
    )
    assert scene.chain.dragged is None, "release did not let go"

    # and a press on empty background must rotate instead of grabbing
    scene.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )
    assert scene.chain.dragged is None, "background press grabbed a mass"
    assert scene.rotate, "background press did not start a camera rotate"
    print("DRAG CHECK OK")
    app.quit()


QTimer.singleShot(800, check)
sys.exit(app.exec())
```

Run it from the demo folder (needs a real display, do NOT use offscreen):

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring/MassSpring
uv run python /tmp/drag_check.py
```
Expected: `picked index: 1 (expected 1)`, the mass moving right, then `DRAG CHECK OK`.

- [ ] **Step 8: Full suite, lint and commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
uv run pytest -q          # expect 348 passed
uv run ruff check MassSpring/ && uv run ruff format MassSpring/
git add MassSpring/MassSpringScene.py
git commit -m "feat(mass-spring): pick and drag a mass with the mouse"
```

Expected full-suite count: 334 before this plan + 7 (Task 1) + 7 (Task 2) = **348 passed**.

---

### Task 4: Document it

**Files:**
- Modify: `MassSpring/README.md`

- [ ] **Step 1: Update the controls paragraph**

In `MassSpring/README.md`, replace the controls paragraph:

```markdown
Left mouse rotates; everything else is on the panel. Pin either end, turn on
gravity and watch it swing. The cubes are the masses (red when pinned, green
when free) and the small blue spheres are ghosts showing where each mass
started.
```

with:

```markdown
Left mouse drags a mass if you grab one and rotates the camera if you miss.
Everything else is on the panel. Pin either end, turn on gravity and watch it
swing. The cubes are the masses -- red when pinned, yellow while you are
holding one, green otherwise -- and the small blue spheres are ghosts showing
where each mass started.

Picking is done by colour: the masses are rendered flat, each in a unique
colour keyed to its index, and the pixel under the cursor is read back and
decoded. It is worth knowing that multisampling blends the colours along an
edge into ones that decode to nothing, so the pick reads a small block around
the cursor and takes the first exact match rather than trusting one pixel.

A mass you are holding is kinematic -- it goes into the same fixed set as a
pinned mass, so the integrator leaves it alone rather than fighting the mouse.
Let go and it drops from a standstill. Drag a pinned mass and you move where
it is pinned to.
```

- [ ] **Step 2: Add picking.py to the file list**

In the "How it is put together" list, after the `mass_spring.py` bullet:

```markdown
- `picking.py` -- the colour ID encoding, the unproject and the ray/plane
  intersection used to drag a mass. No Qt, no OpenGL.
```

- [ ] **Step 3: Commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mass-spring
git add MassSpring/README.md
git commit -m "docs(mass-spring): document picking and dragging"
```
