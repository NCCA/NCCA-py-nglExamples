# Core Demos Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port NGL9Demos/Collisions (4 sub-demos: RaySphere, RayTriangle, SpherePlane, SphereSphere) to a single PyNGLDemos demo, `Collisions/`, Tab-toggled between the 4 modes, with an OpenGL and a WebGPU entry point.

**Architecture:** One pure-numpy maths module (`Collisions/collision_maths.py`, no GL/Qt/wgpu imports) exports the four collision-test functions, unit-tested headless (mirroring `RayPickingSelection/picking_maths.py`'s pattern — the explicit precedent named in the design spec). Both entry points import this module unchanged. Each entry point is a single `QOpenGLWindow`/`WebGPUWidget` subclass with 4 independent scene states (one per mode), switched by Tab, mirroring `LookAtDemos`'s precedent of combining multiple NGL9Demos sub-demos into one Tab-toggled Python demo rather than 4 separate folders.

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, `uv run --script`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md`

## Source mapping and deliberate deviations from NGL9Demos

Read directly during planning: `/Users/jmacey/teaching/NGL9Demos/Collisions/{RaySphere,RayTriangle,SpherePlane,SphereSphere}`.

- **RaySphere**: quadratic discriminant test (`raySphere()`). C++ treats a tangent hit (discriminant == 0) as a miss (`discrim <= 0.0` → false) — this plan matches that exactly (`discriminant > 0.0`).
- **RayTriangle**: Möller–Trumbore (`rayTriangleIntersect()`). The C++ computes the ray-parameter `t` via Möller–Trumbore (misleadingly stored in a variable called `m_w`), then **redundantly recomputes the same hit point a second way** via a separate plane-intersection step using the triangle's face normal. This plan uses the already-computed `t` directly (`hit_point = origin + t * direction`) — mathematically identical result, simpler code, no behavioural change. Documented here, not a silent simplification.
- **SpherePlane**: signed-distance-to-plane test (`spherePlaneCollide()`). The C++ computes distance as `plane_normal · sphere_pos` (dotting the raw position against the normal, with no subtraction of a point on the plane) — this only gives the correct signed distance when the plane passes through the world origin along its normal. This plan uses the general, correct point-to-plane formula, `plane_normal · (sphere_pos - plane_center)`, which works regardless of where the plane sits — a deliberate correctness improvement over the C++'s narrower assumption, not a behavioural regression for any placement this demo actually uses.
- **SphereSphere**: squared-distance test against the sum of radii (`sphereSphereCollision()`) — faithful, direct port, no changes.

All four are pure analytic geometry — no physics engine, no collision response beyond "reverse velocity" / "reset position", matching the NGL9Demos originals' teaching-level simplicity.

## Global Constraints

- Work happens in branch `agent/core-demos-phase3`, worktree at `.worktrees/core-demos-phase3` (create with `git worktree add .worktrees/core-demos-phase3 -b agent/core-demos-phase3` before starting Task 1 — not yet created).
- No edits to `/Users/jmacey/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder.
- No importing code from other demo folders (e.g. `RayPickingSelection/picking_maths.py`) even though its `intersect_sphere`/`intersect_triangles` functions are conceptually similar — this repo's convention is no shared application code between demos, only the shared `ncca.ngl` dependency. `Collisions/collision_maths.py` is self-contained, written fresh (informed by, but not importing, `RayPickingSelection`'s style).
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) which runs one paint pass via `QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))` then exits 0 — copy the pattern verbatim from `Spotlight/main.py`'s `__main__` block (OpenGL) or `Spotlight/main_webgpu.py`'s `main()` (WebGPU).
- OpenGL entry point: `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)`, calling `self.setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01, zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` in `__init__`. GL 4.1 core profile via the standard `QSurfaceFormat` block (copy from `Spotlight/main.py`'s `__main__`).
- WebGPU entry point: `class WebGPUScene(WebGPUWidget)` importing `from ncca.ngl.webgpu import WebGPUWidget` directly. Set `self.msaa_sample_count = 4`, call `get_default_device()`, build pipelines/scene, then `self._create_render_buffer()`. Mouse/keyboard handlers hand-copied from `Blending/BlendingWebGPU.py` (no mixin for `QWidget`).
- **Mandatory from the start, not discovered later**: any entry point using a repeating `QTimer` for animation MUST override `closeEvent` to stop that timer before `super().closeEvent(event)`. This is a real, user-reported crash fix from Phase 2 (`OpenGL.error.Error: Attempt to retrieve context when no valid context` when a `QTimer` fired GPU calls after window-close teardown) — copy the exact pattern from `Spotlight/main.py:152-154` (`self.animation_timer.stop()` then `super().closeEvent(event)`). Apply this to BOTH the OpenGL and WebGPU entry points in this plan — the Phase 2 fix only covered OpenGL entry points and left `Spotlight/main_webgpu.py` without this protection; do not repeat that gap here.
- Maths convention: numpy/PyNGL row-vector convention — points transform as `row_vec @ M`, translation lives in row 3 (`mat[3, 0..2]`). The collision-test functions in this plan work directly on world-space points/vectors and involve no matrix transforms.
- Smoketest verification for OpenGL demos in this environment: `QT_QPA_PLATFORM=offscreen` segfaults on this machine for every `QOpenGLWindow` demo (confirmed repo-wide since Phase 1). Verify `main.py` WITHOUT that env var: `cd Collisions && uv run --script main.py --smoketest` (real display, briefly flashes a window, auto-quits). WebGPU entry points work fine WITH `QT_QPA_PLATFORM=offscreen` (confirmed repeatedly in Phases 1-2).
- `ruff check` and `ruff format --check` must pass.
- README.md (description, controls, teaching points, `![](Collisions.png)` image reference — screenshot itself expected missing, deferred to Jon); add a row to the root `README.md` under an appropriate section (likely near `RayPickingSelection` or `FrustumCull`, both collision/geometry-test-adjacent — create the section if none fits).
- One commit per task.
- Test module lives at `Collisions/tests/test_collision_maths.py`, following `RayPickingSelection/tests/test_picking_maths.py`'s exact structure: `sys.path.insert(0, str(Path(__file__).parent.parent))` then `from collision_maths import (...)  # noqa: E402`, class-per-function test grouping.

---

## Task 1: Collisions maths + tests + OpenGL

**Files:**
- Create: `Collisions/collision_maths.py`
- Create: `Collisions/tests/test_collision_maths.py`
- Create: `Collisions/main.py`
- Create: `Collisions/README.md`

**Interfaces:**
- Produces (in `collision_maths.py`, no GL/Qt/wgpu imports — Task 2 imports these unchanged):
  - `ray_sphere_intersect(ray_start: np.ndarray, ray_dir: np.ndarray, sphere_pos: np.ndarray, radius: float) -> bool`
  - `ray_triangle_intersect(ray_start: np.ndarray, ray_end: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple[bool, np.ndarray | None]`
  - `sphere_plane_collide(sphere_pos: np.ndarray, radius: float, plane_center: np.ndarray, plane_normal: np.ndarray, plane_width: float, plane_depth: float) -> bool`
  - `sphere_sphere_collide(pos1: np.ndarray, radius1: float, pos2: np.ndarray, radius2: float) -> bool`

- [ ] **Step 1: Write the collision-maths module**

Create `Collisions/collision_maths.py`:

```python
"""Pure-maths collision-test helpers, ported from NGL9Demos/Collisions
(RaySphere, RayTriangle, SpherePlane, SphereSphere).

Deliberately numpy-only (no GL/Qt/wgpu) so the collision maths is
unit-testable headless, mirroring RayPickingSelection/picking_maths.py's
pattern. These functions work directly on world-space points/vectors --
no matrix transforms are needed for analytic collision tests.
"""

from __future__ import annotations

import numpy as np

_DET_EPSILON = 1e-5


def ray_sphere_intersect(
    ray_start: np.ndarray,
    ray_dir: np.ndarray,
    sphere_pos: np.ndarray,
    radius: float,
) -> bool:
    """True if the ray (direction need not be normalised) hits the sphere.

    Ported from NGL9Demos/Collisions/RaySphere's raySphere(): quadratic
    discriminant test. A tangent hit (discriminant == 0) counts as a miss,
    matching the C++'s `discrim <= 0.0 -> false`.
    """
    d = np.asarray(ray_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    p = np.asarray(ray_start, dtype=np.float64) - np.asarray(
        sphere_pos, dtype=np.float64
    )
    a = float(d @ d)
    b = 2.0 * float(d @ p)
    c = float(p @ p) - radius * radius
    discriminant = b * b - 4.0 * a * c
    return discriminant > 0.0


def ray_triangle_intersect(
    ray_start: np.ndarray,
    ray_end: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """Moller-Trumbore ray/triangle intersection, ported from
    NGL9Demos/Collisions/RayTriangle's rayTriangleIntersect().

    ray_start/ray_end define a finite ray segment (direction = end - start,
    not normalised, matching the C++). Returns (hit, hit_point); hit_point
    is None when hit is False.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    origin = np.asarray(ray_start, dtype=np.float64)
    direction = np.asarray(ray_end, dtype=np.float64) - origin

    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(edge1 @ pvec)
    if -_DET_EPSILON < det < _DET_EPSILON:
        return False, None
    inv_det = 1.0 / det

    tvec = origin - v0
    u = float(tvec @ pvec) * inv_det
    if u < -0.001 or u > 1.001:
        return False, None

    qvec = np.cross(tvec, edge1)
    v = float(direction @ qvec) * inv_det
    if v < -0.001 or u + v > 1.001:
        return False, None

    t = float(edge2 @ qvec) * inv_det
    if t <= 0.0:
        return False, None

    hit_point = origin + t * direction
    return True, hit_point.astype(np.float32)


def sphere_plane_collide(
    sphere_pos: np.ndarray,
    radius: float,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
    plane_width: float,
    plane_depth: float,
) -> bool:
    """True if the sphere touches or has crossed the plane, within the
    plane's rectangular extent (width along x, depth along z, centred on
    plane_center). Ported from NGL9Demos/Collisions/SpherePlane's
    spherePlaneCollide(), generalised to a plane_center not at the world
    origin (the C++ only handles a plane through the origin -- see the
    plan's "deliberate deviations" note)."""
    pos = np.asarray(sphere_pos, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    center = np.asarray(plane_center, dtype=np.float64)

    signed_dist = float(normal @ (pos - center)) - radius
    if signed_dist > 0.0:
        return False

    half_w = plane_width / 2.0
    half_d = plane_depth / 2.0
    return (
        center[0] - half_w < pos[0] < center[0] + half_w
        and center[2] - half_d < pos[2] < center[2] + half_d
    )


def sphere_sphere_collide(
    pos1: np.ndarray, radius1: float, pos2: np.ndarray, radius2: float
) -> bool:
    """True if two spheres overlap or touch. Ported from
    NGL9Demos/Collisions/SphereSphere's sphereSphereCollision(): squared-
    distance test against the sum of radii."""
    p1 = np.asarray(pos1, dtype=np.float64)
    p2 = np.asarray(pos2, dtype=np.float64)
    rel = p1 - p2
    dist_sq = float(rel @ rel)
    min_dist = radius1 + radius2
    return dist_sq <= min_dist * min_dist
```

- [ ] **Step 2: Write the failing tests**

Create `Collisions/tests/test_collision_maths.py`:

```python
"""Headless tests for the collision-detection maths."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from collision_maths import (  # noqa: E402
    ray_sphere_intersect,
    ray_triangle_intersect,
    sphere_plane_collide,
    sphere_sphere_collide,
)


class TestRaySphereIntersect:
    def test_ray_through_centre_hits(self):
        assert ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_ray_missing_sphere(self):
        assert not ray_sphere_intersect(
            np.array([0.0, 5.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_ray_pointing_away_misses(self):
        # sphere is behind the ray origin -- no hit even though the
        # infinite *line* would pass through it
        assert not ray_sphere_intersect(
            np.array([0.0, 0.0, 10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )

    def test_unnormalised_direction_gives_same_result(self):
        hit_a = ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )
        hit_b = ray_sphere_intersect(
            np.array([0.0, 0.0, -10.0]),
            np.array([0.0, 0.0, 50.0]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
        )
        assert hit_a == hit_b


class TestRayTriangleIntersect:
    def setup_method(self):
        self.v0 = np.array([-1.0, -1.0, 0.0])
        self.v1 = np.array([1.0, -1.0, 0.0])
        self.v2 = np.array([0.0, 1.0, 0.0])

    def test_ray_through_centre_hits(self):
        hit, point = ray_triangle_intersect(
            np.array([0.0, -0.3, -5.0]),
            np.array([0.0, -0.3, 5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert hit
        np.testing.assert_allclose(point, [0.0, -0.3, 0.0], atol=1e-5)

    def test_ray_missing_triangle(self):
        hit, point = ray_triangle_intersect(
            np.array([5.0, 5.0, -5.0]),
            np.array([5.0, 5.0, 5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit
        assert point is None

    def test_ray_parallel_to_triangle_misses(self):
        hit, point = ray_triangle_intersect(
            np.array([0.0, 0.0, -5.0]),
            np.array([1.0, 0.0, -5.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit
        assert point is None

    def test_ray_stopping_before_triangle_misses(self):
        # segment ends at z=-1, triangle is at z=0 -- t would be > 1 (this
        # port treats the segment as a ray though, so this actually still
        # hits since t is unbounded above; use a ray pointing away instead
        hit, _ = ray_triangle_intersect(
            np.array([0.0, -0.3, 5.0]),
            np.array([0.0, -0.3, 10.0]),
            self.v0,
            self.v1,
            self.v2,
        )
        assert not hit


class TestSpherePlaneCollide:
    def setup_method(self):
        self.center = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        self.width = 10.0
        self.depth = 10.0

    def test_sphere_touching_plane_collides(self):
        assert sphere_plane_collide(
            np.array([0.0, 1.0, 0.0]), 1.0, self.center, self.normal, self.width, self.depth
        )

    def test_sphere_above_plane_no_collision(self):
        assert not sphere_plane_collide(
            np.array([0.0, 5.0, 0.0]), 1.0, self.center, self.normal, self.width, self.depth
        )

    def test_sphere_below_plane_within_bounds_collides(self):
        assert sphere_plane_collide(
            np.array([0.0, -2.0, 0.0]), 1.0, self.center, self.normal, self.width, self.depth
        )

    def test_sphere_below_plane_outside_bounds_no_collision(self):
        assert not sphere_plane_collide(
            np.array([20.0, -2.0, 0.0]), 1.0, self.center, self.normal, self.width, self.depth
        )

    def test_offset_plane_centre_still_correct(self):
        # regression test for the deliberate generalisation over the C++
        # source, which only handles a plane through the world origin
        offset_center = np.array([0.0, 5.0, 0.0])
        assert sphere_plane_collide(
            np.array([0.0, 6.0, 0.0]), 1.0, offset_center, self.normal, self.width, self.depth
        )
        assert not sphere_plane_collide(
            np.array([0.0, 9.0, 0.0]), 1.0, offset_center, self.normal, self.width, self.depth
        )


class TestSphereSphereCollide:
    def test_touching_spheres_collide(self):
        assert sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([2.0, 0.0, 0.0]), 1.0
        )

    def test_overlapping_spheres_collide(self):
        assert sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([1.0, 0.0, 0.0]), 1.0
        )

    def test_separated_spheres_no_collision(self):
        assert not sphere_sphere_collide(
            np.array([0.0, 0.0, 0.0]), 1.0, np.array([5.0, 0.0, 0.0]), 1.0
        )
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `cd Collisions && uv run pytest tests/ -v`
Expected: 16 tests pass (4 RaySphere + 4 RayTriangle + 5 SpherePlane + 3 SphereSphere = 16). There is no "RED" step here since `collision_maths.py` is written in Step 1 alongside the tests, not test-first, because the maths is a direct, well-understood port rather than a novel algorithm being designed -- this mirrors how `ViewToWorldTransform`'s `unproject_point` was planned.

- [ ] **Step 4: Verify the `ncca.ngl` API used in `main.py` below**

Run: `grep -n "def perspective\|def look_at\|class Transform\|class Vec3\|def rotate_x\|def rotate_y\|def rotate_z" /Users/jmacey/teaching/Code/PyNGL/src/ncca/ngl/*.py`
Expected: confirms these exist with the signatures already used throughout Phases 1-2 (`perspective(fov, aspect, near, far)`, `look_at(eye, look, up)`, `Transform` with `set_position`/`set_scale`/`matrix`, `Vec3`). If any differ, adjust Step 5's code to match before proceeding.

- [ ] **Step 5: Write main.py**

Create `Collisions/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""Collisions demo: Ray-Sphere, Ray-Triangle, Sphere-Plane, Sphere-Sphere
collision detection, Tab-toggled between the 4 modes (OpenGL)."""

import argparse
import sys
import traceback
from enum import Enum, auto
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from collision_maths import (
    ray_sphere_intersect,
    ray_triangle_intersect,
    sphere_plane_collide,
    sphere_sphere_collide,
)
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class Mode(Enum):
    RAY_SPHERE = auto()
    RAY_TRIANGLE = auto()
    SPHERE_PLANE = auto()
    SPHERE_SPHERE = auto()


_MODE_ORDER = [Mode.RAY_SPHERE, Mode.RAY_TRIANGLE, Mode.SPHERE_PLANE, Mode.SPHERE_SPHERE]
_MODE_LABELS = {
    Mode.RAY_SPHERE: "Ray - Sphere",
    Mode.RAY_TRIANGLE: "Ray - Triangle",
    Mode.SPHERE_PLANE: "Sphere - Plane",
    Mode.SPHERE_SPHERE: "Sphere - Sphere",
}


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Collisions")
        self.mode_index = 0
        self.rng = np.random.default_rng(42)
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()

    @property
    def mode(self) -> Mode:
        return _MODE_ORDER[self.mode_index]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 5.0, 8.0, 5.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        self.view = look_at(Vec3(0, 4, 16), Vec3(0, 0, 0), Vec3(0, 1, 0))
        Primitives.load_default_primitives()

        self._init_ray_sphere()
        self._init_ray_triangle()
        self._init_sphere_plane()
        self._init_sphere_sphere()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    # ---- Ray-Sphere ---------------------------------------------------
    def _init_ray_sphere(self) -> None:
        self.rs_spheres = []
        for _ in range(8):
            x = float(self.rng.uniform(-5, 5))
            y = float(self.rng.uniform(-4, 4))
            radius = float(self.rng.uniform(0.3, 1.0))
            self.rs_spheres.append({"pos": np.array([x, y, 0.0]), "radius": radius, "hit": False})
        self.rs_ray1_x = -8.0
        self.rs_ray2_x = 8.0
        self.rs_direction = 1.0

    def _tick_ray_sphere(self) -> None:
        self.rs_ray1_x += 0.15 * self.rs_direction
        self.rs_ray2_x -= 0.15 * self.rs_direction
        if self.rs_ray1_x > 8.0 or self.rs_ray1_x < -8.0:
            self.rs_direction *= -1.0
        ray1_start = np.array([self.rs_ray1_x, 6.0, 0.0])
        ray1_dir = np.array([0.0, -1.0, 0.0])
        ray2_start = np.array([self.rs_ray2_x, -6.0, 0.0])
        ray2_dir = np.array([0.0, 1.0, 0.0])
        for sphere in self.rs_spheres:
            hit1 = ray_sphere_intersect(ray1_start, ray1_dir, sphere["pos"], sphere["radius"])
            hit2 = ray_sphere_intersect(ray2_start, ray2_dir, sphere["pos"], sphere["radius"])
            sphere["hit"] = hit1 or hit2
        self.rs_ray1 = (ray1_start, ray1_start + ray1_dir * 12.0)
        self.rs_ray2 = (ray2_start, ray2_start + ray2_dir * 12.0)

    # ---- Ray-Triangle ---------------------------------------------------
    def _init_ray_triangle(self) -> None:
        self.rt_v0 = np.array([-2.5, -1.5, 0.0])
        self.rt_v1 = np.array([2.5, -1.5, 0.0])
        self.rt_v2 = np.array([0.0, 2.5, 0.0])
        self.rt_ray_x = -6.0
        self.rt_direction = 1.0
        self.rt_hit = False
        self.rt_hit_point: np.ndarray | None = None

    def _tick_ray_triangle(self) -> None:
        self.rt_ray_x += 0.1 * self.rt_direction
        if self.rt_ray_x > 6.0 or self.rt_ray_x < -6.0:
            self.rt_direction *= -1.0
        ray_start = np.array([self.rt_ray_x, 0.0, 6.0])
        ray_end = np.array([self.rt_ray_x, 0.0, -6.0])
        self.rt_hit, self.rt_hit_point = ray_triangle_intersect(
            ray_start, ray_end, self.rt_v0, self.rt_v1, self.rt_v2
        )
        self.rt_ray = (ray_start, ray_end)

    # ---- Sphere-Plane ---------------------------------------------------
    def _init_sphere_plane(self) -> None:
        self.sp_plane_center = np.array([0.0, -3.0, 0.0])
        self.sp_plane_normal = np.array([0.0, 1.0, 0.0])
        self.sp_plane_width = 10.0
        self.sp_plane_depth = 10.0
        self.sp_spheres = []
        for _ in range(6):
            self.sp_spheres.append(self._new_falling_sphere())

    def _new_falling_sphere(self) -> dict:
        x = float(self.rng.uniform(-4, 4))
        z = float(self.rng.uniform(-4, 4))
        return {
            "pos": np.array([x, 8.0, z]),
            "vel": np.array([0.0, 0.0, 0.0]),
            "radius": 0.4,
            "hit": False,
        }

    def _tick_sphere_plane(self) -> None:
        gravity = 0.01
        for sphere in self.sp_spheres:
            sphere["vel"][1] -= gravity
            sphere["pos"] = sphere["pos"] + sphere["vel"]
            hit = sphere_plane_collide(
                sphere["pos"],
                sphere["radius"],
                self.sp_plane_center,
                self.sp_plane_normal,
                self.sp_plane_width,
                self.sp_plane_depth,
            )
            sphere["hit"] = hit
            if hit or sphere["pos"][1] < -10.0:
                new = self._new_falling_sphere()
                sphere["pos"] = new["pos"]
                sphere["vel"] = new["vel"]

    # ---- Sphere-Sphere ---------------------------------------------------
    def _init_sphere_sphere(self) -> None:
        self.ss_spheres = [
            {"pos": np.array([-3.0, 0.0, 0.0]), "vel": np.array([0.04, 0.03, 0.0]), "radius": 0.6, "hit": False},
            {"pos": np.array([3.0, 0.0, 0.0]), "vel": np.array([-0.03, 0.04, 0.0]), "radius": 0.6, "hit": False},
            {"pos": np.array([0.0, 3.0, 0.0]), "vel": np.array([0.03, -0.03, 0.0]), "radius": 1.0, "hit": False},
            {"pos": np.array([0.0, -3.0, 0.0]), "vel": np.array([-0.04, 0.02, 0.0]), "radius": 1.0, "hit": False},
        ]
        self.ss_bounds = 6.0

    def _tick_sphere_sphere(self) -> None:
        for sphere in self.ss_spheres:
            sphere["pos"] = sphere["pos"] + sphere["vel"]
            for axis in (0, 1):
                if abs(sphere["pos"][axis]) > self.ss_bounds:
                    sphere["vel"][axis] *= -1.0
            sphere["hit"] = False

        pairs = [(0, 1), (0, 2), (1, 3)]
        for i, j in pairs:
            a, b = self.ss_spheres[i], self.ss_spheres[j]
            if sphere_sphere_collide(a["pos"], a["radius"], b["pos"], b["radius"]):
                a["vel"] = a["vel"] * -1.0
                b["vel"] = b["vel"] * -1.0
                a["hit"] = True
                b["hit"] = True

    def _on_tick(self) -> None:
        if self.mode == Mode.RAY_SPHERE:
            self._tick_ray_sphere()
        elif self.mode == Mode.RAY_TRIANGLE:
            self._tick_ray_triangle()
        elif self.mode == Mode.SPHERE_PLANE:
            self._tick_sphere_plane()
        elif self.mode == Mode.SPHERE_SPHERE:
            self._tick_sphere_sphere()
        self.update()

    def _draw_sphere(self, pos: np.ndarray, radius: float, hit: bool) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        colour = (1.0, 0.2, 0.2, 1.0) if hit else (1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("Colour", *colour)
        tx = Transform()
        tx.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        tx.set_scale(radius, radius, radius)
        self._load_matrices(tx)
        Primitives.draw("sphere")

    def _draw_line(self, start: np.ndarray, end: np.ndarray) -> None:
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(float(start[0]), float(start[1]), float(start[2]))
        gl.glVertex3f(float(end[0]), float(end[1]), float(end[2]))
        gl.glEnd()

    def _load_matrices(self, tx: Transform) -> None:
        m = self.global_tx() @ tx.matrix()
        mv = self.view @ m
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(m).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        m = rot_y @ rot_x
        m[3, 0] = self.model_position.x
        m[3, 1] = self.model_position.y
        m[3, 2] = self.model_position.z
        return m

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self.mode == Mode.RAY_SPHERE:
            for sphere in self.rs_spheres:
                self._draw_sphere(sphere["pos"], sphere["radius"], sphere["hit"])
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("MVP", self.project @ self.view @ self.global_tx())
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            self._draw_line(*self.rs_ray1)
            self._draw_line(*self.rs_ray2)

        elif self.mode == Mode.RAY_TRIANGLE:
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("MVP", self.project @ self.view @ self.global_tx())
            colour = (1.0, 0.2, 0.2, 1.0) if self.rt_hit else (0.2, 0.6, 1.0, 1.0)
            ShaderLib.set_uniform("Colour", *colour)
            gl.glBegin(gl.GL_TRIANGLES)
            for v in (self.rt_v0, self.rt_v1, self.rt_v2):
                gl.glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            gl.glEnd()
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            self._draw_line(*self.rt_ray)
            if self.rt_hit and self.rt_hit_point is not None:
                self._draw_sphere(self.rt_hit_point, 0.15, True)

        elif self.mode == Mode.SPHERE_PLANE:
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("MVP", self.project @ self.view @ self.global_tx())
            ShaderLib.set_uniform("Colour", 0.5, 0.5, 0.5, 1.0)
            hw = self.sp_plane_width / 2.0
            hd = self.sp_plane_depth / 2.0
            cx, cy, cz = self.sp_plane_center
            gl.glBegin(gl.GL_LINE_LOOP)
            for dx, dz in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
                gl.glVertex3f(cx + dx, cy, cz + dz)
            gl.glEnd()
            for sphere in self.sp_spheres:
                self._draw_sphere(sphere["pos"], sphere["radius"], sphere["hit"])

        elif self.mode == Mode.SPHERE_SPHERE:
            for sphere in self.ss_spheres:
                self._draw_sphere(sphere["pos"], sphere["radius"], sphere["hit"])

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 100.0)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Tab:
            self.mode_index = (self.mode_index + 1) % len(_MODE_ORDER)
            self.setTitle(f"Collisions - {_MODE_LABELS[self.mode]}")
            self.update()
        self.update()
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

Note: `gl.glBegin`/`gl.glEnd`/`gl.glVertex3f` immediate-mode calls are used here for the ray/triangle/plane-outline line drawing (matching `CurveDemos/main.py`'s established fallback pattern in this repo for simple line drawing under a Core Profile context). **Verify this actually works** by running the smoke test (Step 6). If it fails (core profile rejects `glBegin`), replace with a small ad-hoc `VAOFactory.create_vao("simple", mode=gl.GL_LINES)` VAO the same way `CurveDemos/main.py` or `MatrixStack/main.py`'s reference grid does, with a single `vec3` position attribute at location 0.

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x Collisions/main.py
cd Collisions && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback (real display, not offscreen -- see Global Constraints).

- [ ] **Step 7: Write README.md**

Create `Collisions/README.md`:

```markdown
# Collisions

Four analytic-geometry collision tests -- Ray-Sphere, Ray-Triangle,
Sphere-Plane, Sphere-Sphere -- ported from NGL9Demos, Tab-toggled between
modes. The collision maths lives in `collision_maths.py`, a pure-numpy
module with no GL/Qt/wgpu imports (see `tests/`), shared unchanged between
this OpenGL entry point and the WebGPU one (`main_webgpu.py`).

- **Ray-Sphere**: two rays sweep back and forth through a field of random
  spheres; a sphere turns red when either ray intersects it.
- **Ray-Triangle**: a single ray sweeps across a fixed triangle
  (Moller-Trumbore intersection); the hit point is marked with a small
  sphere when it lands inside the triangle.
- **Sphere-Plane**: spheres fall under simple gravity and reset to a random
  position above the plane when they cross it.
- **Sphere-Sphere**: four spheres bounce inside a bounding box and reverse
  direction pairwise on contact.

## Controls
`Tab` : cycle through the 4 collision modes
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 8: Commit**

```bash
git add Collisions/collision_maths.py Collisions/tests/ Collisions/main.py Collisions/README.md
git commit -m "feat(collisions): add collision maths, tests, and OpenGL demo"
```

---

## Task 2: Collisions (WebGPU)

**Files:**
- Create: `Collisions/CollisionsShader.wgsl`
- Create: `Collisions/main_webgpu.py`
- Modify: `Collisions/README.md` (append WebGPU section)

**Interfaces:**
- Consumes: `ray_sphere_intersect`, `ray_triangle_intersect`, `sphere_plane_collide`, `sphere_sphere_collide` from `collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:**

Independent of Task 1's `main.py` -- does not import it, mirrors the same 4 modes, same scene parameters (sphere counts, positions, radii, ray sweep bounds), same Tab-toggle behaviour, in WGSL/wgpu-py instead. WebGPU has no runtime primitive generator here, so spheres use the baked `octahedron` mesh (`PrimData.primitive(Prims.OCTAHEDRON.value)`), matching `MatrixStack/main_webgpu.py`'s established precedent for a baked-mesh sphere stand-in (Phase 1 confirmed `"sphere"` is not in the baked set -- only `troll`, `teapot`, `cube`, `bunny`, `buddah`, `dragon`, `football`, `octahedron`, `dodecahedron`, `icosahedron`, `tetrahedron`). Rays, the plane outline, and the Ray-Triangle fill are drawn via two small extra pipelines using position-only vertex buffers rebuilt each frame.

**Draw-count / buffer-pool sizing (read this carefully before Step 3):** the worst case is Ray-Sphere mode, which draws 8 spheres **plus 2 ray lines** = 10 draw calls inside one render pass. Every one of those 10 draws -- spheres and lines alike -- needs its OWN uniform-buffer/bind-group slot from the pool; reusing a slot between two draws in the same frame (e.g. giving a line draw the same slot a sphere draw already used) reintroduces exactly the queue-timeline aliasing bug this pool exists to prevent, just as certainly as never pooling at all. `_DRAW_POOL_SIZE = 10`, and `paintWebGPU` must maintain a single incrementing `draw_index` counter shared across every draw call in the frame (sphere or line), passed explicitly into whichever draw helper is called -- mirroring `Spotlight/main_webgpu.py`'s pattern where the ground plane and every teapot share one counter, just extended here to cover line draws too, since `Spotlight` never had that case.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/CollisionsShader.wgsl`:

```wgsl
struct Uniforms {
    m: mat4x4<f32>,
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) in_vert: vec3<f32>, @location(1) in_normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_position = (u.m * vec4<f32>(in_vert, 1.0)).xyz;
    out.normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

const light_pos = vec3<f32>(5.0, 8.0, 5.0);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(light_pos - in.world_position);
    let n_dot_l = max(0.15, dot(n, l));
    return vec4<f32>(u.colour.rgb * n_dot_l, u.colour.a);
}

struct LineOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_line(@location(0) in_vert: vec3<f32>) -> LineOutput {
    var out: LineOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    return out;
}

@fragment
fn fs_line(in: LineOutput) -> @location(0) vec4<f32> {
    return u.colour;
}
```

`vs_main`/`fs_main` shade the lit octahedron (sphere stand-in) draws; `vs_line`/`fs_line` are an unlit pass-through for line-list and flat-fill triangle geometry, sharing the same `Uniforms` struct and bind-group layout (the `normal_matrix` field is simply unused/ignored by `fs_line`).

- [ ] **Step 2: Verify `PerspMode` and API names before writing main_webgpu.py**

Run: `grep -n "PerspMode\|def primitive\|class PrimData" /Users/jmacey/teaching/Code/PyNGL/src/ncca/ngl/__init__.py /Users/jmacey/teaching/Code/PyNGL/src/ncca/ngl/webgpu/*.py`
Expected: confirms `perspective(fov, aspect, near, far, PerspMode.WebGPU)` is still the correct call shape (used in every Phase 1-2 WebGPU demo) and `PrimData.primitive(name)` / `Prims.OCTAHEDRON.value` are unchanged since `Spotlight/main_webgpu.py` was written. Adjust Step 3 if anything differs.

- [ ] **Step 3: Write main_webgpu.py**

Create `Collisions/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""
Collisions: Ray-Sphere, Ray-Triangle, Sphere-Plane, Sphere-Sphere collision
detection, Tab-toggled between the 4 modes (WebGPU).

Same 4 modes and collision maths as the OpenGL version (main.py) -- both
import collision_maths.py unchanged -- but built on the WebGPU stack.
Spheres use the baked octahedron mesh (WebGPU has no procedural sphere
generator here); rays/plane outline/triangle fill are drawn as small
hand-built line-list/triangle-list geometry, rebuilt once per frame.

This demo draws up to 10 objects (Ray-Sphere mode's 8 spheres + 2 rays)
inside a single render pass, each with its own uniform buffer pulled from
a pool pre-allocated at init and indexed by a per-frame draw counter that
covers every draw call in the frame, spheres and lines alike -- see
_create_draw_buffer_pool for why a single shared buffer, or reusing a slot
between a sphere draw and a line draw, doesn't work here.

Controls:
    Tab : cycle through the 4 collision modes
    LMB rotate  RMB pan  wheel zoom  Space reset
"""

import argparse
import sys
import traceback
from enum import Enum, auto
from pathlib import Path

import numpy as np
import wgpu
from collision_maths import (
    ray_sphere_intersect,
    ray_triangle_intersect,
    sphere_plane_collide,
    sphere_sphere_collide,
)
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device


class Mode(Enum):
    RAY_SPHERE = auto()
    RAY_TRIANGLE = auto()
    SPHERE_PLANE = auto()
    SPHERE_SPHERE = auto()


_MODE_ORDER = [Mode.RAY_SPHERE, Mode.RAY_TRIANGLE, Mode.SPHERE_PLANE, Mode.SPHERE_SPHERE]
_MODE_LABELS = {
    Mode.RAY_SPHERE: "Ray - Sphere",
    Mode.RAY_TRIANGLE: "Ray - Triangle",
    Mode.SPHERE_PLANE: "Sphere - Plane",
    Mode.SPHERE_SPHERE: "Sphere - Sphere",
}
# Worst case is Ray-Sphere mode: 8 spheres + 2 ray lines = 10 draws in one
# render pass. Every draw, sphere or line, needs its own pool slot.
_DRAW_POOL_SIZE = 10


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.setWindowTitle("Collisions (WebGPU)")
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.original_x = 0
        self.original_y = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.model_position = Vec3(0, 0, 0)
        self.mode_index = 0
        self.rng = np.random.default_rng(42)
        self.eye = Vec3(0, 4, 16)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.5, 100.0, PerspMode.WebGPU)

        self.device = get_default_device()
        self._create_pipelines()
        self._create_octahedron_geometry()
        self._create_draw_buffer_pool()

        self._init_ray_sphere()
        self._init_ray_triangle()
        self._init_sphere_plane()
        self._init_sphere_sphere()

        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    @property
    def mode(self) -> Mode:
        return _MODE_ORDER[self.mode_index]

    def _create_pipelines(self) -> None:
        shader_src = (Path(__file__).parent / "CollisionsShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        lit_vertex_layout = {
            "array_stride": 8 * 4,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32x3, "offset": 3 * 4, "shader_location": 1},
            ],
        }
        self.sphere_pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader_module, "entry_point": "vs_main", "buffers": [lit_vertex_layout]},
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        line_vertex_layout = {
            "array_stride": 3 * 4,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
            ],
        }
        self.line_pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader_module, "entry_point": "vs_line", "buffers": [line_vertex_layout]},
            fragment={
                "module": shader_module,
                "entry_point": "fs_line",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )
        self.tri_pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": shader_module, "entry_point": "vs_line", "buffers": [line_vertex_layout]},
            fragment={
                "module": shader_module,
                "entry_point": "fs_line",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_octahedron_geometry(self) -> None:
        octahedron = PrimData.primitive(Prims.OCTAHEDRON.value)
        self.octahedron_buffer = self.device.create_buffer_with_data(
            data=octahedron.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.octahedron_count = octahedron.size // 8

    def _create_draw_buffer_pool(self) -> None:
        """Pre-allocate one uniform buffer + bind group per draw call.

        Ray-Sphere mode draws 8 spheres + 2 ray lines = 10 draws inside one
        render pass. WebGPU's queue-timeline ordering only guarantees a
        submitted command buffer sees a resource's state as of immediately
        before that submit -- rewriting one shared uniform buffer multiple
        times before a single submit() would leave every draw seeing only
        the last write (the exact bug fixed in MatrixStack/main_webgpu.py
        and LookAtDemos/main_webgpu.py). A pool sidesteps that: EVERY draw
        this frame -- sphere or line -- gets its own buffer/bind-group
        slot, indexed by one counter shared across the whole frame and
        reset at the start of paintWebGPU.
        """
        # 208-byte Uniforms struct: 3 mat4x4 @ 64 bytes each + 1 vec4 @ 16 bytes.
        self._uniform_size = 4 * 4 * 4 * 3 + 16
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for index in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=self._uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"collisions_draw_{index}",
            )
            self.draw_uniform_buffers.append(buf)
            self.draw_bind_groups.append(
                self.device.create_bind_group(
                    layout=self.bind_group_layout,
                    entries=[
                        {
                            "binding": 0,
                            "resource": {"buffer": buf, "offset": 0, "size": self._uniform_size},
                        }
                    ],
                )
            )

    def _global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _write_uniforms(self, draw_index: int, model: Mat4, colour: tuple) -> None:
        mvp = self.project @ self.view @ model
        normal_matrix = model.inverse().transposed()
        data = np.zeros(self._uniform_size // 4, dtype=np.float32)
        data[0:16] = model.to_numpy().flatten()
        data[16:32] = mvp.to_numpy().flatten()
        data[32:48] = normal_matrix.to_numpy().flatten()
        data[48:52] = colour
        self.device.queue.write_buffer(self.draw_uniform_buffers[draw_index], 0, data.tobytes())

    def _draw_sphere(
        self, render_pass, draw_index: int, pos: np.ndarray, radius: float, hit: bool
    ) -> None:
        model = (
            self._global_tx()
            @ Mat4().translate(float(pos[0]), float(pos[1]), float(pos[2]))
            @ Mat4().scale(radius, radius, radius)
        )
        colour = (1.0, 0.2, 0.2, 1.0) if hit else (1.0, 1.0, 0.0, 1.0)
        self._write_uniforms(draw_index, model, colour)
        render_pass.set_pipeline(self.sphere_pipeline)
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.octahedron_buffer)
        render_pass.draw(self.octahedron_count)

    def _draw_lines(
        self, render_pass, draw_index: int, points: list, colour: tuple, filled: bool = False
    ) -> None:
        data = np.array(points, dtype=np.float32).flatten()
        buf = self.device.create_buffer_with_data(
            data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self._write_uniforms(draw_index, self._global_tx(), colour)
        render_pass.set_pipeline(self.tri_pipeline if filled else self.line_pipeline)
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, buf)
        render_pass.draw(len(points))

    # ---- Ray-Sphere ---------------------------------------------------
    def _init_ray_sphere(self) -> None:
        self.rs_spheres = []
        for _ in range(8):
            x = float(self.rng.uniform(-5, 5))
            y = float(self.rng.uniform(-4, 4))
            radius = float(self.rng.uniform(0.3, 1.0))
            self.rs_spheres.append({"pos": np.array([x, y, 0.0]), "radius": radius, "hit": False})
        self.rs_ray1_x = -8.0
        self.rs_ray2_x = 8.0
        self.rs_direction = 1.0
        self.rs_ray1 = (np.zeros(3), np.zeros(3))
        self.rs_ray2 = (np.zeros(3), np.zeros(3))

    def _tick_ray_sphere(self) -> None:
        self.rs_ray1_x += 0.15 * self.rs_direction
        self.rs_ray2_x -= 0.15 * self.rs_direction
        if self.rs_ray1_x > 8.0 or self.rs_ray1_x < -8.0:
            self.rs_direction *= -1.0
        ray1_start = np.array([self.rs_ray1_x, 6.0, 0.0])
        ray1_dir = np.array([0.0, -1.0, 0.0])
        ray2_start = np.array([self.rs_ray2_x, -6.0, 0.0])
        ray2_dir = np.array([0.0, 1.0, 0.0])
        for sphere in self.rs_spheres:
            hit1 = ray_sphere_intersect(ray1_start, ray1_dir, sphere["pos"], sphere["radius"])
            hit2 = ray_sphere_intersect(ray2_start, ray2_dir, sphere["pos"], sphere["radius"])
            sphere["hit"] = hit1 or hit2
        self.rs_ray1 = (ray1_start, ray1_start + ray1_dir * 12.0)
        self.rs_ray2 = (ray2_start, ray2_start + ray2_dir * 12.0)

    # ---- Ray-Triangle ---------------------------------------------------
    def _init_ray_triangle(self) -> None:
        self.rt_v0 = np.array([-2.5, -1.5, 0.0])
        self.rt_v1 = np.array([2.5, -1.5, 0.0])
        self.rt_v2 = np.array([0.0, 2.5, 0.0])
        self.rt_ray_x = -6.0
        self.rt_direction = 1.0
        self.rt_hit = False
        self.rt_hit_point: np.ndarray | None = None
        self.rt_ray = (np.zeros(3), np.zeros(3))

    def _tick_ray_triangle(self) -> None:
        self.rt_ray_x += 0.1 * self.rt_direction
        if self.rt_ray_x > 6.0 or self.rt_ray_x < -6.0:
            self.rt_direction *= -1.0
        ray_start = np.array([self.rt_ray_x, 0.0, 6.0])
        ray_end = np.array([self.rt_ray_x, 0.0, -6.0])
        self.rt_hit, self.rt_hit_point = ray_triangle_intersect(
            ray_start, ray_end, self.rt_v0, self.rt_v1, self.rt_v2
        )
        self.rt_ray = (ray_start, ray_end)

    # ---- Sphere-Plane ---------------------------------------------------
    def _init_sphere_plane(self) -> None:
        self.sp_plane_center = np.array([0.0, -3.0, 0.0])
        self.sp_plane_normal = np.array([0.0, 1.0, 0.0])
        self.sp_plane_width = 10.0
        self.sp_plane_depth = 10.0
        self.sp_spheres = [self._new_falling_sphere() for _ in range(6)]

    def _new_falling_sphere(self) -> dict:
        x = float(self.rng.uniform(-4, 4))
        z = float(self.rng.uniform(-4, 4))
        return {
            "pos": np.array([x, 8.0, z]),
            "vel": np.array([0.0, 0.0, 0.0]),
            "radius": 0.4,
            "hit": False,
        }

    def _tick_sphere_plane(self) -> None:
        gravity = 0.01
        for sphere in self.sp_spheres:
            sphere["vel"][1] -= gravity
            sphere["pos"] = sphere["pos"] + sphere["vel"]
            hit = sphere_plane_collide(
                sphere["pos"],
                sphere["radius"],
                self.sp_plane_center,
                self.sp_plane_normal,
                self.sp_plane_width,
                self.sp_plane_depth,
            )
            sphere["hit"] = hit
            if hit or sphere["pos"][1] < -10.0:
                new = self._new_falling_sphere()
                sphere["pos"] = new["pos"]
                sphere["vel"] = new["vel"]

    # ---- Sphere-Sphere ---------------------------------------------------
    def _init_sphere_sphere(self) -> None:
        self.ss_spheres = [
            {"pos": np.array([-3.0, 0.0, 0.0]), "vel": np.array([0.04, 0.03, 0.0]), "radius": 0.6, "hit": False},
            {"pos": np.array([3.0, 0.0, 0.0]), "vel": np.array([-0.03, 0.04, 0.0]), "radius": 0.6, "hit": False},
            {"pos": np.array([0.0, 3.0, 0.0]), "vel": np.array([0.03, -0.03, 0.0]), "radius": 1.0, "hit": False},
            {"pos": np.array([0.0, -3.0, 0.0]), "vel": np.array([-0.04, 0.02, 0.0]), "radius": 1.0, "hit": False},
        ]
        self.ss_bounds = 6.0

    def _tick_sphere_sphere(self) -> None:
        for sphere in self.ss_spheres:
            sphere["pos"] = sphere["pos"] + sphere["vel"]
            for axis in (0, 1):
                if abs(sphere["pos"][axis]) > self.ss_bounds:
                    sphere["vel"][axis] *= -1.0
            sphere["hit"] = False
        pairs = [(0, 1), (0, 2), (1, 3)]
        for i, j in pairs:
            a, b = self.ss_spheres[i], self.ss_spheres[j]
            if sphere_sphere_collide(a["pos"], a["radius"], b["pos"], b["radius"]):
                a["vel"] = a["vel"] * -1.0
                b["vel"] = b["vel"] * -1.0
                a["hit"] = True
                b["hit"] = True

    def _on_tick(self) -> None:
        if self.mode == Mode.RAY_SPHERE:
            self._tick_ray_sphere()
        elif self.mode == Mode.RAY_TRIANGLE:
            self._tick_ray_triangle()
        elif self.mode == Mode.SPHERE_PLANE:
            self._tick_sphere_plane()
        elif self.mode == Mode.SPHERE_SPHERE:
            self._tick_sphere_sphere()
        self.update()

    def paintWebGPU(self) -> None:
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "clear_value": (0.15, 0.15, 0.15, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )

        draw_index = 0  # one counter for every draw this frame -- sphere or line

        if self.mode == Mode.RAY_SPHERE:
            for sphere in self.rs_spheres:
                self._draw_sphere(render_pass, draw_index, sphere["pos"], sphere["radius"], sphere["hit"])
                draw_index += 1
            self._draw_lines(render_pass, draw_index, [self.rs_ray1[0], self.rs_ray1[1]], (1.0, 1.0, 1.0, 1.0))
            draw_index += 1
            self._draw_lines(render_pass, draw_index, [self.rs_ray2[0], self.rs_ray2[1]], (1.0, 1.0, 1.0, 1.0))
            draw_index += 1

        elif self.mode == Mode.RAY_TRIANGLE:
            colour = (1.0, 0.2, 0.2, 1.0) if self.rt_hit else (0.2, 0.6, 1.0, 1.0)
            self._draw_lines(
                render_pass, draw_index, [self.rt_v0, self.rt_v1, self.rt_v2], colour, filled=True
            )
            draw_index += 1
            self._draw_lines(render_pass, draw_index, [self.rt_ray[0], self.rt_ray[1]], (1.0, 1.0, 1.0, 1.0))
            draw_index += 1
            if self.rt_hit and self.rt_hit_point is not None:
                self._draw_sphere(render_pass, draw_index, self.rt_hit_point, 0.15, True)
                draw_index += 1

        elif self.mode == Mode.SPHERE_PLANE:
            hw = self.sp_plane_width / 2.0
            hd = self.sp_plane_depth / 2.0
            cx, cy, cz = self.sp_plane_center
            outline = [
                np.array([cx - hw, cy, cz - hd]), np.array([cx + hw, cy, cz - hd]),
                np.array([cx + hw, cy, cz - hd]), np.array([cx + hw, cy, cz + hd]),
                np.array([cx + hw, cy, cz + hd]), np.array([cx - hw, cy, cz + hd]),
                np.array([cx - hw, cy, cz + hd]), np.array([cx - hw, cy, cz - hd]),
            ]
            self._draw_lines(render_pass, draw_index, outline, (0.7, 0.7, 0.7, 1.0))
            draw_index += 1
            for sphere in self.sp_spheres:
                self._draw_sphere(render_pass, draw_index, sphere["pos"], sphere["radius"], sphere["hit"])
                draw_index += 1

        elif self.mode == Mode.SPHERE_SPHERE:
            for sphere in self.ss_spheres:
                self._draw_sphere(render_pass, draw_index, sphere["pos"], sphere["radius"], sphere["hit"])
                draw_index += 1

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.5, 100.0, PerspMode.WebGPU)
        self.update()

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x = position.x()
            self.original_y = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        if self.rotate and event.buttons() == Qt.LeftButton:
            diff_x = position.x() - self.original_x
            diff_y = position.y() - self.original_y
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x = position.x()
            self.original_y = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            diff_x = position.x() - self.original_x_pos
            diff_y = position.y() - self.original_y_pos
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += 0.01 * diff_x
            self.model_position.y -= 0.01 * diff_y
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.model_position.z += 0.1
        elif delta < 0:
            self.model_position.z -= 0.1
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Tab:
            self.mode_index = (self.mode_index + 1) % len(_MODE_ORDER)
            self.setWindowTitle(f"Collisions (WebGPU) - {_MODE_LABELS[self.mode]}")
        elif event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position = Vec3(0, 0, 0)
        self.update()

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x Collisions/main_webgpu.py
cd Collisions && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 5: Verify the buffer pool actually prevents aliasing**

The same empirical check used in Phase 2 (`Spotlight/main_webgpu.py`'s task report): render a frame in Ray-Sphere mode (the worst-case 10-draw mode), read back the frame buffer, and confirm the 8 spheres appear at 8 visually distinct positions/colours and both rays are visible as separate lines -- not everything collapsed onto one draw's transform. If they collapse, a slot is being shared between two draws in the same frame -- re-check that `draw_index` genuinely increments once per draw call with no reuse, and fix before proceeding. Do not ship this unverified given the project's history with this exact bug class (including the one caught and fixed during this plan's own self-review, see the "Draw-count / buffer-pool sizing" note above).

- [ ] **Step 6: Add the WebGPU note to README.md**

Append to `Collisions/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same 4 Tab-toggled modes using the baked
`octahedron` mesh in place of a runtime-generated sphere (WebGPU has no
procedural primitive generator here), and a per-draw uniform-buffer pool
sized to the worst case (Ray-Sphere mode's 8 spheres + 2 rays = 10 draws)
so every simultaneously-drawn object -- sphere or line -- gets its own GPU
buffer. A single shared buffer, or two draws sharing one pool slot, would
alias onto the last-written transform.
```

- [ ] **Step 7: Commit**

```bash
git add Collisions/CollisionsShader.wgsl Collisions/main_webgpu.py Collisions/README.md
git commit -m "feat(collisions): add WebGPU entry point"
```

---

## Final steps (after both tasks)

- [ ] **Add root README.md entry**

Add a row for `Collisions` to the root `README.md`, under whichever existing section fits best (likely alongside `RayPickingSelection`/`FrustumCull` in a geometry-test-themed grouping, or a new section if none fits) -- follow the existing row format exactly (name, link, thumbnail).

- [ ] **Run full verification**

```bash
uv run ruff check Collisions/
uv run ruff format --check Collisions/
uv run pytest Collisions/tests/ -v
```
Expected: ruff clean, all 16 tests pass.

- [ ] **Report to Jon**

List the `Collisions.png` screenshot that still needs capturing, and note that exercising Tab through all 4 modes (not just the default Ray-Sphere mode) is the only way to visually confirm all 4 collision tests work correctly -- no automated check exercises Tab.
