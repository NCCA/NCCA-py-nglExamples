# Core Demos Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all 5 NGL9Demos `Collisions` sub-demos (RaySphere, RayTriangle, SpherePlane, SphereSphere, BoundingBox) to PyNGLDemos as 5 self-contained sub-demo folders under one `Collisions/` parent folder, each faithful to its C++ original's object counts and controls, each with an OpenGL and a WebGPU entry point.

**Architecture:** One shared `Collisions/collision_maths.py` (pure numpy, no GL/Qt/wgpu imports) holds the 5 analytic collision-test functions, tested headless in `Collisions/tests/test_collision_maths.py`. Every sub-demo folder (`Collisions/RaySphere/`, `Collisions/RayTriangle/`, `Collisions/SpherePlane/`, `Collisions/SphereSphere/`, `Collisions/BoundingBox/`) is otherwise self-contained (own `main.py`, own `main_webgpu.py`, own README, own shaders) and imports the one shared maths module unchanged via `sys.path.insert(0, str(Path(__file__).parent.parent))`. This mirrors the existing `PBR/` and `WebGPUCompute/` precedent of one parent topic-folder containing several independent sub-demo folders, plus this repo's established pattern (`MatrixStack`, `ViewToWorldTransform`) of a GL/WebGPU sibling pair sharing one pure-maths module unchanged.

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, `uv run --script`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md`

**Note on scope revision:** the design spec's Phase 3 entry originally described "Collisions" as one demo combining 4 sub-demos with a Tab toggle (mirroring `LookAtDemos`). That approach was built, then explicitly rejected by the user in favour of the structure this plan implements: 5 separate sub-demo folders (a 5th, `BoundingBox`, was also discovered in the source and added to scope), each faithful to its C++ original's exact object counts and controls rather than a simplified/combined teaching version. This plan supersedes the spec's brief Phase-3 description; the spec text is not being edited to match, so treat this plan as authoritative for Phase 3's actual scope.

## Global Constraints

- No edits to `/Users/jmacey/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder (except the one intentional in-topic sharing described above).
- Work happens in branch `agent/core-demos-phase3`, worktree at `.worktrees/core-demos-phase3` (already exists, do not recreate).
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) using the `QTimer.singleShot(...)` pattern from `VAOPrimitives/main.py` (OpenGL) or `Blending/BlendingWebGPU.py`'s `main()` (WebGPU).
- OpenGL entry points: `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)`, calling `self.setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01, zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` in `__init__`. GL 4.1 core profile via the standard `QSurfaceFormat` block (copy from `VAOPrimitives/main.py`'s `__main__`).
- WebGPU entry points: `class WebGPUScene(WebGPUWidget)` importing `from ncca.ngl.webgpu import WebGPUWidget` directly. `self.msaa_sample_count = 4`, call `get_default_device()`, build pipelines/scene, then `self._create_render_buffer()`. Mouse/keyboard handlers hand-copied from `Blending/BlendingWebGPU.py` (no mixin for `QWidget`).
- Maths convention: numpy/PyNGL row-vector convention — points transform as `row_vec @ M`, translation lives in row 3. Matrix composition order matches the C++ source (`A @ B @ C` applies like `A * B * C` did in NGL9Demos) — do not reorder.
- **Smoketest verification command differs by backend, established repeatedly across Phases 1-2:** OpenGL entry points (`main.py`) must be verified WITHOUT `QT_QPA_PLATFORM=offscreen` on this machine (that mode segfaults for every `QOpenGLWindow` demo here — pre-existing environment limitation). WebGPU entry points (`main_webgpu.py`) must be verified WITH it (works fine).
- **`closeEvent`-stops-the-animation-timer requirement, mandatory from the start:** any entry point that drives its scene with a repeating `QTimer` (i.e. calls `.start(ms)`, not `QTimer.singleShot`) MUST override `closeEvent` to call `<timer_attr>.stop()` before `super().closeEvent(event)` — a real, twice-fixed crash in this repo (Phase 2's `Spotlight`/`ShadedGrid`) came from a queued timer tick firing a GL/GPU call after window-close teardown. Of the 5 sub-demos here, `RaySphere`, `SpherePlane`, `SphereSphere`, and `BoundingBox` all animate via a repeating timer and need this; `RayTriangle` has no repeating timer (its scene only changes on user key/mouse input) and does not need it.
- **Dynamic-marker draw pattern** (referenced by name in later tasks): for small, variable-count, per-frame-recomputed visual markers that don't fit a fixed per-object WebGPU buffer pool cleanly (e.g. ray/sphere hit points, a tilting plane's normal-indicator line) — build ONE position(+colour)-only vertex buffer per marker *kind* each frame (rebuilt in Python/numpy, uploaded via `queue.write_buffer`), and issue exactly one draw call for the whole batch (a line-list or point-list topology pipeline), rather than growing the object buffer pool per marker. This mirrors `ShadedGrid/main_webgpu.py`'s established per-frame full-geometry-rebuild pattern.
- **WebGPU per-draw uniform buffer pool** (the queue-timeline aliasing bug, found and fixed twice already — `MatrixStack`, `LookAtDemos`): any WebGPU task issuing more than one draw call against per-object uniform data (M/MVP/colour, etc.) within a single render pass MUST pre-allocate a pool of that many uniform buffers + bind groups once at init, index them with a per-frame counter reset at the start of `paintWebGPU`, and never let two draws in the same frame share a slot. Each task below states its exact worst-case pool size — do not deviate without recomputing the true worst case.
- WebGPU has no runtime primitive generator — baked meshes only, via `PrimData.primitive(Prims.<NAME>.value)`. Confirmed available: `troll`, `teapot`, `cube`, `bunny`, `buddah`, `dragon`, `football`, `octahedron`, `dodecahedron`, `icosahedron`, `tetrahedron` — no `sphere`. Every WebGPU task below that needs a sphere stand-in uses the baked `octahedron` mesh, matching this repo's established precedent (`MatrixStack/main_webgpu.py`'s original ring, `Spotlight/main_webgpu.py`).
- `ncca.ngl.Random` exists but its exact method surface wasn't grepped for this plan (not needed — Python's own `random` module is used throughout instead; the exact RNG algorithm/distribution shape only needs to match the C++'s described ranges, not its bit-level PRNG, since these are decorative scene-population values, not something a test asserts bit-for-bit).
- `ruff check` and `ruff format --check` must pass.
- README.md per sub-demo folder (description, controls, teaching points, `![](<SubDemo>.png)` reference — screenshot itself expected missing, deferred to Jon). No README.md at the `Collisions/` parent-folder level (matches the `PBR/`/`WebGPUCompute/` precedent — only subfolders get READMEs).
- Root `README.md` gets one row per sub-demo, named `Collisions/<SubDemo>` for both link text and image path (exact format confirmed against `PBR/PBRTexture` and `WebGPUCompute/SpatialHash3D`'s existing rows: `| <a href="Collisions/RaySphere"><img src="Collisions/RaySphere/RaySphere.png" width="220"></a> | [Collisions/RaySphere](Collisions/RaySphere) | <description> |`). Add all 5 rows as part of Task 1 is NOT required — add each sub-demo's row in its own OpenGL task (first entry point built for that sub-demo), matching how root-README rows landed inside feature commits in Phase 2 (a pattern the Phase 2 final review confirmed as harmless/acceptable, not something to avoid this time).
- One commit per task.

## Ported-vs-original deviations (apply consistently, don't re-derive per task)

- **RaySphere/RayTriangle hit-point/marker rendering on WebGPU** uses the Dynamic-marker pattern above (one pooled draw for the objects, one extra draw for hit-point markers), not a wireframe-polygon-mode toggle per hit (wgpu doesn't expose per-draw polygon-mode the way `glPolygonMode` does with a shared pooled pipeline) — hit objects are shown in a distinct tint colour instead. This is a documented, deliberate WebGPU-only rendering adaptation; the OpenGL siblings use the C++'s real wireframe-on-hit via `glPolygonMode`.
- **SpherePlane's plane** is rendered as a static local-space quad (`Primitives.create(Prims.TRIANGLE_PLANE, "plane", 5, 5, 1, 1, Vec3(0, 1, 0))` on OpenGL; a hand-built numpy quad on WebGPU) with the accumulated tilt applied as a rotation transform at draw time (`Mat4.rotate_z(zrot) @ Mat4.rotate_x(xrot)`), rather than porting the C++'s bespoke `MultiBufferIndexVAO`/regenerated-vertex approach — same visual result (a plane that tilts about world X and Z), simpler and idiomatic for this repo's `Transform`-based rendering. This does not change any user-facing control or count.
- **BoundingBox's variable sphere count** (`+`/`-` keys) is uncapped in the C++. The WebGPU sibling's fixed-size buffer pool needs a hard ceiling; Task 11 documents the exact cap chosen and why — this is a WebGPU-architecture necessity, not a simplification of the demo's default behaviour (50 spheres, add/remove by 1), which is fully faithful.

---

## Task 1: Collisions maths + tests (finalize)

**Files:**
- Modify: `Collisions/collision_maths.py` (already exists, uncommitted, in this worktree — 4 of 5 functions already written and passing 16 tests; this task adds the 5th and commits the whole module)
- Modify: `Collisions/tests/test_collision_maths.py` (already exists, uncommitted — add tests for the 5th function)

**Interfaces:**
- Produces (no GL/Qt/wgpu imports — every later task imports these unchanged via `sys.path.insert(0, str(Path(__file__).parent.parent)); from collision_maths import <name>`):
  - `ray_sphere_intersect(ray_start, ray_dir, sphere_pos, radius) -> bool` (already written)
  - `ray_triangle_intersect(ray_start, ray_end, v0, v1, v2) -> tuple[bool, np.ndarray | None]` (already written)
  - `sphere_plane_collide(sphere_pos, radius, plane_center, plane_normal, plane_width, plane_depth) -> bool` (already written)
  - `sphere_sphere_collide(pos1, radius1, pos2, radius2) -> bool` (already written)
  - `sphere_bbox_reflect(position: np.ndarray, direction: np.ndarray, radius: float, half_extent: float) -> tuple[bool, np.ndarray]` — **new in this task**

The existing 4 functions in `Collisions/collision_maths.py` are correct, already-TDD'd ports (verified against the actual NGL9Demos C++ source during planning) — do not rewrite them, only add the 5th.

- [ ] **Step 1: Write the failing tests for `sphere_bbox_reflect`**

Append to `Collisions/tests/test_collision_maths.py` (add the import to the existing `from collision_maths import (...)` block, and add this new test class at the end of the file):

```python
from collision_maths import (  # noqa: E402
    ray_sphere_intersect,
    ray_triangle_intersect,
    sphere_bbox_reflect,
    sphere_plane_collide,
    sphere_sphere_collide,
)


class TestSphereBboxReflect:
    def test_no_wall_hit_direction_unchanged(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            40.0,
        )
        assert not hit
        np.testing.assert_allclose(new_dir, [1.0, 0.0, 0.0])

    def test_hitting_positive_x_wall_reflects_x_component(self):
        # position + radius crosses the +X wall at half_extent=40
        hit, new_dir = sphere_bbox_reflect(
            np.array([39.5, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_hitting_positive_y_wall_reflects_y_component_only(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 39.5, 0.0]),
            np.array([0.5, 1.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [0.5, -1.0, 0.0], atol=1e-10)

    def test_hitting_negative_z_wall_reflects_z_component(self):
        hit, new_dir = sphere_bbox_reflect(
            np.array([0.0, 0.0, -39.5]),
            np.array([0.0, 0.0, -1.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [0.0, 0.0, 1.0], atol=1e-10)

    def test_corner_reflects_off_both_walls(self):
        # position + radius crosses both +X and +Y walls simultaneously
        hit, new_dir = sphere_bbox_reflect(
            np.array([39.5, 39.5, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            1.0,
            40.0,
        )
        assert hit
        np.testing.assert_allclose(new_dir, [-1.0, -1.0, 0.0], atol=1e-10)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd Collisions && uv run pytest tests/test_collision_maths.py -v -k BboxReflect`
Expected: FAIL — `ImportError: cannot import name 'sphere_bbox_reflect'` (function doesn't exist yet).

- [ ] **Step 3: Implement `sphere_bbox_reflect`**

Append to `Collisions/collision_maths.py` (after `sphere_sphere_collide`), and update the module docstring's first line to read `"""Pure-maths collision-test helpers, ported from NGL9Demos/Collisions` `(RaySphere, RayTriangle, SpherePlane, SphereSphere, BoundingBox).` (just the parenthetical list — the rest of the docstring is unchanged):

```python
_BBOX_FACE_NORMALS = (
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
)


def sphere_bbox_reflect(
    position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    half_extent: float,
) -> tuple[bool, np.ndarray]:
    """True/new-direction if a sphere moving inside a cube centred on the
    origin (half_extent along every axis) has crossed one of the cube's 6
    axis-aligned walls. Ported from NGL9Demos/Collisions/BoundingBox's
    BBoxCollision(): for each face normal, if position . normal + radius
    >= half_extent, reflect direction across that normal (in place, so a
    corner can reflect off two -- or three -- walls in one call, matching
    the C++'s unconditional loop over all 6 faces every frame)."""
    pos = np.asarray(position, dtype=np.float64)
    new_dir = np.asarray(direction, dtype=np.float64).copy()
    hit = False
    for normal in _BBOX_FACE_NORMALS:
        d = float(normal @ pos) + radius
        if d >= half_extent:
            new_dir = new_dir - 2.0 * float(new_dir @ normal) * normal
            hit = True
    return hit, new_dir
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd Collisions && uv run pytest tests/test_collision_maths.py -v`
Expected: all tests pass (16 existing + 5 new = 21 passed).

- [ ] **Step 5: Lint and commit**

```bash
cd Collisions && uv run ruff check collision_maths.py tests/test_collision_maths.py && uv run ruff format --check collision_maths.py tests/test_collision_maths.py
cd ..
git add Collisions/collision_maths.py Collisions/tests/test_collision_maths.py
git commit -m "feat(collisions): add collision-maths module (5 analytic tests, 21 pytest cases)"
```

---

## Task 2: SphereSphere (OpenGL)

**Files:**
- Create: `Collisions/SphereSphere/main.py`
- Create: `Collisions/SphereSphere/README.md`

**Interfaces:**
- Consumes: `sphere_sphere_collide` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:** the simplest of the 5 sub-demos — exactly 4 fixed spheres, no spawning/respawning, no wireframe-on-hit. Ported faithfully from `NGL9Demos/Collisions/SphereSphere/src/NGLScene.cpp`: sphere `[0]` at `(-10,0,0)` radius 2, static, yellow; sphere `[1]` at `(10,0,0)` radius 2, static, yellow; sphere `[2]` at `(-7,0,0)` radius 1, direction `(0.5,0,0)`, red, moving; sphere `[3]` at `(7,0,0)` radius 1, direction `(-0.5,0,0)`, blue, moving. Every tick (20ms): move spheres 2 and 3 by their direction; if 2 and 3 collide, reverse both directions; if 0 and 2 collide, reverse 2; if 1 and 3 collide, reverse 3 (checked independently, matching the C++ — not an early-exit chain). The C++'s `glClearColor(1.4, 1.4, 1.4, 1)` clamps to pure white in OpenGL — use `(1.0, 1.0, 1.0, 1.0)` directly. Only `Escape` is a real key handler in the C++ (no F/N/Space here) — the mixin's own Escape/Space/W/S handling still applies via `super()`.

- [ ] **Step 1: Write main.py**

Create `Collisions/SphereSphere/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""SphereSphere: 4 fixed spheres, 2 moving and bouncing off 2 static ones.

Ported from NGL9Demos/Collisions/SphereSphere -- exact object count (4
spheres, fixed positions/radii/directions/colours) and collision rules,
no simplification.
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_sphere_collide  # noqa: E402

_SPHERES = [
    {
        "pos": Vec3(-10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(-7.0, 0.0, 0.0),
        "dir": Vec3(0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (1.0, 0.0, 0.0),
    },
    {
        "pos": Vec3(7.0, 0.0, 0.0),
        "dir": Vec3(-0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (0.0, 0.0, 1.0),
    },
]


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
        self.setTitle("SphereSphere")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [dict(s) for s in _SPHERES]
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        self.animation_timer.start(20)

    def _on_tick(self) -> None:
        self.spheres[2]["pos"] = self.spheres[2]["pos"] + self.spheres[2]["dir"]
        self.spheres[3]["pos"] = self.spheres[3]["pos"] + self.spheres[3]["dir"]
        self._check_collisions()
        self.update()

    def _check_collisions(self) -> None:
        s2, s3, s0, s1 = (
            self.spheres[2],
            self.spheres[3],
            self.spheres[0],
            self.spheres[1],
        )
        if sphere_sphere_collide(
            _v3(s2["pos"]), s2["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
            s3["dir"] = s3["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s0["pos"]), s0["radius"], _v3(s2["pos"]), s2["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s1["pos"]), s1["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s3["dir"] = s3["dir"] * -1.0

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.DIFFUSE)
        for s in self.spheres:
            ShaderLib.set_uniform("Colour", *s["colour"], 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("sphere")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)


def _v3(v: Vec3):
    import numpy as np

    return np.array([v.x, v.y, v.z])


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


def main() -> None:
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


if __name__ == "__main__":
    main()
```

Note: `Mat3.from_mat4(m)` (model-only, not model-view) matches this repo's established world-space-normal-matrix convention. Confirm `Vec3.__mul__(float)` and `Vec3.__add__(Vec3)` exist (used throughout every prior phase, already proven).

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x Collisions/SphereSphere/main.py
cd Collisions/SphereSphere && uv run --script main.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback (real display, not offscreen).

- [ ] **Step 3: Write README.md**

Create `Collisions/SphereSphere/README.md`:

```markdown
# SphereSphere

![](SphereSphere.png)

Two large, static spheres (yellow) and two small spheres (red, blue) that
move toward each other and bounce apart on collision -- with each other
and with the static spheres -- using an analytic sphere/sphere overlap
test (`collision_maths.sphere_sphere_collide`).

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 4: Add the root README row**

Add to the root `README.md`, in whichever section fits best (check the existing sections — likely alongside other physics/collision-flavoured demos, or create a small "Collision Detection" grouping if none fits; match the exact row format from `PBR/PBRTexture`'s row):

```markdown
| <a href="Collisions/SphereSphere"><img src="Collisions/SphereSphere/SphereSphere.png" width="220"></a> | [Collisions/SphereSphere](Collisions/SphereSphere) | Two moving spheres bounce off two static ones (analytic sphere/sphere test) |
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/SphereSphere/
git commit -m "feat(sphere-sphere): add OpenGL sphere/sphere collision demo"
```

---

## Task 3: SphereSphere (WebGPU)

**Files:**
- Create: `Collisions/SphereSphere/main_webgpu.py`
- Create: `Collisions/SphereSphere/SphereSphereShader.wgsl`

**Interfaces:**
- Consumes: `sphere_sphere_collide` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks. Independent of Task 2 -- does not import `main.py`, mirrors the same 4-sphere setup and collision rules.

**Design notes:** only 4 draw calls per frame (one octahedron per sphere) -- well below any real risk threshold, but the WebGPU per-draw buffer pool convention still applies for consistency: `_DRAW_POOL_SIZE = 4`, one `draw_index` counter reset at the top of `paintWebGPU`.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/SphereSphere/SphereSphereShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = (u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let lambert = max(dot(n, light_dir), 0.0);
    let colour = u.colour.rgb * (0.3 + 0.7 * lambert);
    return vec4<f32>(colour, 1.0);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `Collisions/SphereSphere/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""SphereSphere (WebGPU): 4 fixed spheres, 2 moving and bouncing off 2
static ones -- independent WebGPU port of Collisions/SphereSphere/main.py,
same object count/positions/radii/colours/collision rules.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import PrimData, Prims, WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_sphere_collide  # noqa: E402

_SPHERES = [
    {
        "pos": Vec3(-10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(10.0, 0.0, 0.0),
        "dir": Vec3(0.0, 0.0, 0.0),
        "radius": 2.0,
        "colour": (1.0, 1.0, 0.0),
    },
    {
        "pos": Vec3(-7.0, 0.0, 0.0),
        "dir": Vec3(0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (1.0, 0.0, 0.0),
    },
    {
        "pos": Vec3(7.0, 0.0, 0.0),
        "dir": Vec3(-0.5, 0.0, 0.0),
        "radius": 1.0,
        "colour": (0.0, 0.0, 1.0),
    },
]
_DRAW_POOL_SIZE = 4


def _v3(v: Vec3) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


class WebGPUScene(WebGPUWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.msaa_sample_count = 4
        self.spheres = [dict(s) for s in _SPHERES]
        self.view = look_at(Vec3(0, 0, -20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.05, 350.0, PerspMode.WebGPU)
        self.mouse_global_tx = Mat4()
        self.model_position = Vec3(0, 0, 0)
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.orig_x = 0
        self.orig_y = 0
        self.orig_x_pos = 0
        self.orig_y_pos = 0
        self.device = get_default_device()
        self._create_pipeline()
        self._create_geometry()
        self._create_draw_buffer_pool()
        self._create_render_buffer()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(20)

    def _create_pipeline(self) -> None:
        shader_path = Path(__file__).parent / "SphereSphereShader.wgsl"
        shader_module = self.device.create_shader_module(code=shader_path.read_text())
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
        vertex_buffer_layout = {
            "array_stride": 8 * 4,
            "attributes": [
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
                {
                    "format": wgpu.VertexFormat.float32x2,
                    "offset": 6 * 4,
                    "shader_location": 2,
                },
            ],
        }
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    def _create_geometry(self) -> None:
        data = PrimData.primitive(Prims.OCTAHEDRON.value)
        vertex_count = data.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = vertex_count

    def _create_draw_buffer_pool(self) -> None:
        uniform_size = (16 + 16 + 4) * 4  # mvp mat4 + normal_matrix mat4 + colour vec4
        self.draw_uniform_buffers = []
        self.draw_bind_groups = []
        for _ in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": buf, "offset": 0, "size": uniform_size},
                    }
                ],
            )
            self.draw_uniform_buffers.append(buf)
            self.draw_bind_groups.append(bind_group)

    def _on_tick(self) -> None:
        self.spheres[2]["pos"] = self.spheres[2]["pos"] + self.spheres[2]["dir"]
        self.spheres[3]["pos"] = self.spheres[3]["pos"] + self.spheres[3]["dir"]
        self._check_collisions()
        self.update()

    def _check_collisions(self) -> None:
        s2, s3, s0, s1 = (
            self.spheres[2],
            self.spheres[3],
            self.spheres[0],
            self.spheres[1],
        )
        if sphere_sphere_collide(
            _v3(s2["pos"]), s2["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
            s3["dir"] = s3["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s0["pos"]), s0["radius"], _v3(s2["pos"]), s2["radius"]
        ):
            s2["dir"] = s2["dir"] * -1.0
        if sphere_sphere_collide(
            _v3(s1["pos"]), s1["radius"], _v3(s3["pos"]), s3["radius"]
        ):
            s3["dir"] = s3["dir"] * -1.0

    def _draw_sphere(
        self, render_pass, draw_index: int, s: dict, global_tx: Mat4
    ) -> None:
        m = Mat4().translate(s["pos"].x, s["pos"].y, s["pos"].z) @ Mat4().scale(
            s["radius"], s["radius"], s["radius"]
        )
        m = global_tx @ m
        mv = self.view @ m
        mvp = self.project @ mv
        normal_matrix = m.inverse().transposed()
        data = np.zeros(16 + 16 + 4, dtype=np.float32)
        data[0:16] = mvp.to_numpy().flatten()
        data[16:32] = normal_matrix.to_numpy().flatten()
        data[32:36] = np.array([*s["colour"], 1.0], dtype=np.float32)
        self.device.queue.write_buffer(
            self.draw_uniform_buffers[draw_index], 0, data.tobytes()
        )
        render_pass.set_bind_group(0, self.draw_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)

    def paintWebGPU(self) -> None:
        if not hasattr(self, "device"):
            return
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
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
        render_pass.set_pipeline(self.pipeline)
        draw_index = 0
        for s in self.spheres:
            self._draw_sphere(render_pass, draw_index, s, self.mouse_global_tx)
            draw_index += 1
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, w: int, h: int) -> None:
        self.project = perspective(
            45.0, float(w) / max(h, 1), 0.05, 350.0, PerspMode.WebGPU
        )

    def closeEvent(self, event) -> None:
        self.animation_timer.stop()
        super().closeEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.orig_x, self.orig_y = position.x(), position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.orig_x_pos, self.orig_y_pos = position.x(), position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.orig_x
            diff_y = position.y() - self.orig_y
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.orig_x, self.orig_y = position.x(), position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.orig_x_pos)
            diff_y = int(position.y() - self.orig_y_pos)
            self.orig_x_pos, self.orig_y_pos = position.x(), position.y()
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
            self.model_position.z += 0.5
        elif delta < 0:
            self.model_position.z -= 0.5
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position = Vec3(0, 0, 0)
        self.update()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.setWindowTitle("SphereSphere (WebGPU)")
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

Note: verify `PrimData.primitive`, `wgpu.utils`/`wgpu` import surface, `PerspMode.WebGPU`, and the `colour_buffer_texture_view`/`depth_buffer_view`/`multisample_texture_view`/`_update_colour_buffer`/`_create_render_buffer` attribute names against `Collisions/SphereSphere`'s nearest already-shipped sibling (`Spotlight/main_webgpu.py` or `ShadedGrid/main_webgpu.py`) before running -- these names have been stable across Phase 1-2 but confirm against the actual current `ncca.ngl.webgpu` source if the smoketest errors on an unrecognised attribute.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Collisions/SphereSphere/main_webgpu.py
cd Collisions/SphereSphere && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Collisions/SphereSphere/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same 4-sphere setup and collision rules
independently. Spheres are drawn as the baked `octahedron` mesh (WebGPU
has no runtime sphere primitive here).
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/SphereSphere/main_webgpu.py Collisions/SphereSphere/SphereSphereShader.wgsl Collisions/SphereSphere/README.md
git commit -m "feat(sphere-sphere): add WebGPU entry point"
```

---

## Task 4: RaySphere (OpenGL)

**Files:**
- Create: `Collisions/RaySphere/main.py`
- Create: `Collisions/RaySphere/README.md`

**Interfaces:**
- Consumes: `ray_sphere_intersect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:** ported from `NGL9Demos/Collisions/RaySphere/src/NGLScene.cpp`. **50 spheres by default** (the C++ takes this as `argv[1]`; this port exposes it as `--spheres N`, default 50, preserving the "configurable count, default 50" behaviour). Sphere spawn: `x = uniform(0, 10)`, `y = uniform(0, 8)`, `z = 0`, `radius = uniform(0, 1) + 0.2` (range `[0.2, 1.2)`), colour fixed yellow. Two rays, both animated every tick (50ms): ray 1 start `(0, 10, 0)` fixed, end `(0, -5, 0)` with `end.x` sweeping `±22` in steps of `0.5`; ray 2 start `(0, 0, 20)` fixed, end `(0, 0, -5)` with `end.x` sweeping the *opposite* direction in lock-step (one shared forward/backward flag drives both). Each tick, every sphere is tested against both rays via `ray_sphere_intersect`; a sphere hit by either ray draws wireframe instead of filled, and each ray/sphere hit pair draws two small marker spheres at the near (red) and far (green) intersection points along that ray -- this needs the actual hit-point positions (not just the boolean), so it is computed locally in this file via the plain quadratic-root formula (drawing-only geometry, not a "collision test" per se, so it stays local rather than growing `collision_maths.py`). Camera: `look_at((0,0,-25), origin, (0,1,0))`, `perspective(45, aspect, 0.05, 350)` (near/far match the C++'s `resizeGL`, which always overrides the constructor's transient projection). Background grey `(0.4,0.4,0.4,1)`. Keys: `Escape` (mixin default), `F` = `showFullScreen()`, `N` = `showNormal()`, `Space` = toggle animation. Timer 50ms -- needs the `closeEvent` fix.

- [ ] **Step 1: Write main.py**

Create `Collisions/RaySphere/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""RaySphere: N spheres tested each tick against 2 sweeping rays.

Ported from NGL9Demos/Collisions/RaySphere -- default 50 spheres
(configurable via --spheres), 2 animated rays sweeping in opposite x
directions, wireframe-on-hit, near/far hit-point markers.
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    Prims,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import ray_sphere_intersect  # noqa: E402


def _hit_points(ray_start: Vec3, ray_dir: Vec3, sphere_pos: Vec3, radius: float):
    """Quadratic-root near/far hit points, for drawing only -- ported from
    NGL9Demos's drawHitPoints(). Returns (near, far) Vec3 or (None, None)."""
    d = np.array([ray_dir.x, ray_dir.y, ray_dir.z])
    d = d / np.linalg.norm(d)
    p = np.array([ray_start.x, ray_start.y, ray_start.z]) - np.array(
        [sphere_pos.x, sphere_pos.y, sphere_pos.z]
    )
    a = float(d @ d)
    b = 2.0 * float(d @ p)
    c = float(p @ p) - radius * radius
    discrim = b * b - 4.0 * a * c
    if discrim < 0.0:
        return None, None
    root = discrim**0.5
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    o = np.array([ray_start.x, ray_start.y, ray_start.z])
    h1 = o + d * t1
    h2 = o + d * t2
    return Vec3(*h1), Vec3(*h2)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_spheres: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("RaySphere")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()

        self.spheres = [
            {
                "pos": Vec3(random.uniform(0, 10), random.uniform(0, 8), 0.0),
                "radius": random.uniform(0, 1) + 0.2,
                "hit": False,
            }
            for _ in range(num_spheres)
        ]
        self.ray1_start = Vec3(0, 10, 0)
        self.ray1_end = Vec3(0, -5, 0)
        self.ray2_start = Vec3(0, 0, 20)
        self.ray2_end = Vec3(0, 0, -5)
        self._sweep_forward = True
        self.animate = True
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, -25), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "smallSphere", 0.2, 10)
        self.animation_timer.start(50)

    def _on_tick(self) -> None:
        if not self.animate:
            return
        for s in self.spheres:
            hit1 = ray_sphere_intersect(
                np.array([self.ray1_start.x, self.ray1_start.y, self.ray1_start.z]),
                np.array(
                    [
                        self.ray1_end.x - self.ray1_start.x,
                        self.ray1_end.y - self.ray1_start.y,
                        self.ray1_end.z - self.ray1_start.z,
                    ]
                ),
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
            )
            hit2 = ray_sphere_intersect(
                np.array([self.ray2_start.x, self.ray2_start.y, self.ray2_start.z]),
                np.array(
                    [
                        self.ray2_end.x - self.ray2_start.x,
                        self.ray2_end.y - self.ray2_start.y,
                        self.ray2_end.z - self.ray2_start.z,
                    ]
                ),
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
            )
            s["hit"] = hit1 or hit2

        step = 0.5 if self._sweep_forward else -0.5
        self.ray1_end.x += step
        self.ray2_end.x -= step
        if self.ray1_end.x > 22.0:
            self._sweep_forward = False
        elif self.ray1_end.x <= -22.0:
            self._sweep_forward = True
        self.update()

    def _draw_line(self, p0: Vec3, p1: Vec3, mvp: Mat4) -> None:
        from ncca.ngl.opengl import VAOFactory, VAOType
        from ncca.ngl.opengl.vertex_data import VertexData

        vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        data = np.array([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], dtype=np.float32)
        with vao:
            vao.set_data(VertexData(data, 2))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(2)
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("MVP", mvp)
            vao.draw()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        for start in (self.ray1_start, self.ray2_start):
            tx = Transform()
            tx.set_position(start.x, start.y, start.z)
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("cube")

        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
            ShaderLib.use(DefaultShader.DIFFUSE)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("sphere")
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if s["hit"]:
                for ray_start, ray_end in (
                    (self.ray1_start, self.ray1_end),
                    (self.ray2_start, self.ray2_end),
                ):
                    ray_dir = Vec3(
                        ray_end.x - ray_start.x,
                        ray_end.y - ray_start.y,
                        ray_end.z - ray_start.z,
                    )
                    near, far = _hit_points(ray_start, ray_dir, s["pos"], s["radius"])
                    if near is None:
                        continue
                    for point, colour in (
                        (near, (1.0, 0.0, 0.0)),
                        (far, (0.0, 1.0, 0.0)),
                    ):
                        ShaderLib.use(DefaultShader.DIFFUSE)
                        ShaderLib.set_uniform("Colour", *colour, 1.0)
                        tx2 = Transform()
                        tx2.set_position(point.x, point.y, point.z)
                        m2 = global_tx @ tx2.matrix()
                        mv2 = self.view @ m2
                        ShaderLib.set_uniform("MVP", self.project @ mv2)
                        ShaderLib.set_uniform(
                            "normalMatrix", Mat3.from_mat4(m2).inverse().transposed()
                        )
                        Primitives.draw("smallSphere")

        for ray_start, ray_end in (
            (self.ray1_start, self.ray1_end),
            (self.ray2_start, self.ray2_end),
        ):
            mvp = self.project @ self.view @ global_tx
            self._draw_line(ray_start, ray_end, mvp)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Space:
            self.animate = not self.animate
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spheres", type=int, default=50)
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
    window = MainWindow(num_spheres=args.spheres)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

Note: verify `VAOFactory`/`VAOType`/`VertexData` import paths (`ncca.ngl.opengl.vao_factory`/`vertex_data`) and `Prims.SPHERE` accepting `(name, radius, precision)` via `Primitives.create` against `FrustumCull/main.py`'s working usage before relying on the exact import lines above -- adjust the import path if it differs.

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x Collisions/RaySphere/main.py
cd Collisions/RaySphere && uv run --script main.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 3: Write README.md**

Create `Collisions/RaySphere/README.md`:

```markdown
# RaySphere

![](RaySphere.png)

50 randomly-placed spheres (configurable via `--spheres N`), tested each
tick against 2 animated rays sweeping in opposite `x` directions. A hit
sphere draws wireframe instead of filled, with red/green markers at the
ray's near/far intersection points.

## Controls
`space` : pause/resume ray animation, `f` : fullscreen, `n` : windowed
Left-drag : orbit, Right-drag : pan, Wheel : zoom
```

- [ ] **Step 4: Add the root README row**

```markdown
| <a href="Collisions/RaySphere"><img src="Collisions/RaySphere/RaySphere.png" width="220"></a> | [Collisions/RaySphere](Collisions/RaySphere) | N spheres tested each tick against 2 animated sweeping rays |
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/RaySphere/
git commit -m "feat(ray-sphere): add OpenGL ray/sphere collision demo"
```

---

## Task 5: RaySphere (WebGPU)

**Files:**
- Create: `Collisions/RaySphere/main_webgpu.py`
- Create: `Collisions/RaySphere/RaySphereShader.wgsl`

**Interfaces:**
- Consumes: `ray_sphere_intersect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks. Independent of Task 4.

**Design notes:** spheres use the baked `octahedron` mesh. **Draw-count sizing:** the object pool covers 50 spheres + 2 ray-start marker cubes (also `octahedron`-substituted, or reuse `Prims.CUBE.value` which IS in the baked set) = `_DRAW_POOL_SIZE = 52`. Per the Global Constraints' Dynamic-marker pattern: the 2 ray lines are one combined 4-point line-list draw (rebuilt each frame from the current animated endpoints, not pooled); the near/far hit-point markers (up to `2 rays x 2 points x N hit spheres` in the worst case where every sphere is hit by both rays) are rendered as ONE point-list draw with a per-vertex colour attribute (red for near, green for far), rebuilt each frame from whichever spheres are currently hit -- NOT as individual pooled object draws, since the true worst-case count (up to 200) would make the object pool impractically large for no benefit. Hit visualisation on WebGPU uses a tint colour instead of `glPolygonMode` wireframe (wgpu doesn't expose per-draw polygon-mode against a shared pipeline the way OpenGL does) -- a documented, deliberate adaptation, not a simplification of scene content.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/RaySphere/RaySphereShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = (u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let lambert = max(dot(n, light_dir), 0.0);
    let colour = u.colour.rgb * (0.3 + 0.7 * lambert);
    return vec4<f32>(colour, 1.0);
}

struct LineVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

struct LineUniforms {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> lu: LineUniforms;

@vertex
fn vs_line(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_colour: vec3<f32>,
) -> LineVertexOutput {
    var out: LineVertexOutput;
    out.position = lu.mvp * vec4<f32>(in_vert, 1.0);
    out.colour = in_colour;
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `Collisions/RaySphere/main_webgpu.py`. This mirrors Task 4's scene/sweep logic and Task 3's per-draw buffer-pool + mouse/keyboard boilerplate exactly (`WebGPUScene(WebGPUWidget)`, `self.msaa_sample_count = 4`, `get_default_device()`, `_create_render_buffer()`, hand-copied handlers) -- write it following that established shape, with these RaySphere-specific pieces:

- `_DRAW_POOL_SIZE = 52` (50 spheres + 2 ray-start cube markers), one shared `draw_index` counter incremented for every sphere AND every ray-start marker in `paintWebGPU`.
- A second pipeline (`vs_line`/`fs_line` entry points from the shader above) using a `float32x3 position + float32x3 colour` vertex layout, `primitive.topology = wgpu.PrimitiveTopology.line_list`, its own single small uniform buffer (`mvp` only) and bind group (not pooled -- one line-list draw per frame for the 2 rays: 4 vertices, all white, rebuilt each frame from `ray1_start/end`/`ray2_start/end` since the ends animate).
- A third pipeline (same `vs_line`/`fs_line` shader entry points, `primitive.topology = wgpu.PrimitiveTopology.point_list`) for the hit-point markers -- one draw per frame, vertex buffer rebuilt from every currently-hit sphere's near/far points (red/green colour per vertex, matching Task 4's near=red/far=green), sized generously (e.g. a fixed-capacity buffer for up to `4 * num_spheres` points, only the first `written_count` drawn via `render_pass.draw(written_count)`).
- Sphere hit tint: multiply `colour` by e.g. `(1.6, 0.6, 0.6)` when `s["hit"]` is true (a simple red-shift) instead of a wireframe toggle.
- Same tick-driven sweep/hit-test logic as Task 4's `_on_tick`, same `_hit_points` quadratic-root helper (duplicate the small pure-numpy helper locally in this file too -- it's rendering-only geometry, not part of the shared `collision_maths.py` API, matching Task 4's own local placement).
- `closeEvent` stops `self.animation_timer` (same repeating-timer requirement as Task 4).

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Collisions/RaySphere/main_webgpu.py
cd Collisions/RaySphere && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Collisions/RaySphere/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same 50-sphere/2-ray setup independently.
Spheres use the baked `octahedron` mesh; a hit sphere is tinted red
instead of drawn wireframe (wgpu has no practical per-draw polygon-mode
toggle against a pooled pipeline).
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/RaySphere/main_webgpu.py Collisions/RaySphere/RaySphereShader.wgsl Collisions/RaySphere/README.md
git commit -m "feat(ray-sphere): add WebGPU entry point"
```

---

## Task 6: RayTriangle (OpenGL)

**Files:**
- Create: `Collisions/RayTriangle/main.py`
- Create: `Collisions/RayTriangle/README.md`

**Interfaces:**
- Consumes: `ray_triangle_intersect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:** ported from `NGL9Demos/Collisions/RayTriangle/src/NGLScene.cpp`. **50 triangles by default** (`--triangles N`, default 50, same configurability as RaySphere). Triangle spawn: `c = random_unit_vec3 * 10`, then `v_k = c + Vec3(uniform(-2,2) + 0.1, uniform(-2,2) + 0.1, -uniform(0,2) + 0.1)` for each of 3 verts independently. One ray: `start = (0, 0, 0.2)`, `end = (0, 0, -20)` -- **no animation timer at all** (the C++ has none for this sub-demo; the ray only moves via direct key input) -- so **no `closeEvent` override needed here**. Keys: `Up`/`Down` move `end.y` by `±0.5`, `Left`/`Right` move `end.x` by `±0.5`, `W`/`Z` move `start.y` by `±0.5`, `A`/`S` move `start.x` by `±0.5`. Every `paintGL` call (not timer-gated), every triangle is re-tested against the current ray via `ray_triangle_intersect`; a hit triangle draws wireframe and shows a small marker sphere at the hit point, both still yellow (the C++ never changes colour for a hit, only polygon mode). A small cube marks each triangle's `v0`. Camera: `look_at((0,1,15), origin, (0,1,0))`, same `perspective(45, aspect, 0.05, 350)`.

- [ ] **Step 1: Write main.py**

Create `Collisions/RayTriangle/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""RayTriangle: N random triangles tested every frame against one
interactively-movable ray.

Ported from NGL9Demos/Collisions/RayTriangle -- default 50 triangles
(configurable via --triangles), keyboard-moved ray endpoints, no
animation timer (matches the C++, which has none for this sub-demo).
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    Prims,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import ray_triangle_intersect  # noqa: E402

_STEP = 0.5


def _random_triangle() -> tuple[Vec3, Vec3, Vec3]:
    axis = np.random.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    c = Vec3(*(axis * 10.0))
    verts = []
    for _ in range(3):
        verts.append(
            Vec3(
                c.x + random.uniform(-2, 2) + 0.1,
                c.y + random.uniform(-2, 2) + 0.1,
                c.z - random.uniform(0, 2) + 0.1,
            )
        )
    return tuple(verts)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_triangles: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("RayTriangle")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.triangles = [_random_triangle() for _ in range(num_triangles)]
        self.ray_start = Vec3(0, 0, 0.2)
        self.ray_end = Vec3(0, 0, -20)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "smallSphere", 0.05, 10)

    def _draw_line(self, p0: Vec3, p1: Vec3, mvp: Mat4) -> None:
        from ncca.ngl.opengl import VAOFactory, VAOType
        from ncca.ngl.opengl.vertex_data import VertexData

        vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        data = np.array([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], dtype=np.float32)
        with vao:
            vao.set_data(VertexData(data, 2))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(2)
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("MVP", mvp)
            vao.draw()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        tx = Transform()
        tx.set_position(self.ray_start.x, self.ray_start.y, self.ray_start.z)
        m = global_tx @ tx.matrix()
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        Primitives.draw("cube")

        mvp_line = self.project @ self.view @ global_tx
        self._draw_line(self.ray_start, self.ray_end, mvp_line)

        ray_start_np = np.array([self.ray_start.x, self.ray_start.y, self.ray_start.z])
        ray_end_np = np.array([self.ray_end.x, self.ray_end.y, self.ray_end.z])

        ShaderLib.use(DefaultShader.DIFFUSE)
        for v0, v1, v2 in self.triangles:
            hit, hit_point = ray_triangle_intersect(
                ray_start_np,
                ray_end_np,
                np.array([v0.x, v0.y, v0.z]),
                np.array([v1.x, v1.y, v1.z]),
                np.array([v2.x, v2.y, v2.z]),
            )
            self._draw_triangle(v0, v1, v2, hit, hit_point, global_tx)

    def _draw_triangle(self, v0, v1, v2, hit, hit_point, global_tx) -> None:
        from ncca.ngl.opengl import VAOFactory, VAOType
        from ncca.ngl.opengl.vertex_data import VertexData

        normal = _calc_normal(v0, v1, v2)
        data = np.array(
            [
                v0.x,
                v0.y,
                v0.z,
                normal.x,
                normal.y,
                normal.z,
                0.0,
                0.0,
                v1.x,
                v1.y,
                v1.z,
                normal.x,
                normal.y,
                normal.z,
                0.0,
                0.0,
                v2.x,
                v2.y,
                v2.z,
                normal.x,
                normal.y,
                normal.z,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if hit else gl.GL_FILL)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        m = global_tx
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        with vao:
            vao.set_data(VertexData(data, 3))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 8 * 4, 0)
            vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 8 * 4, 3 * 4)
            vao.set_num_indices(3)
            vao.draw()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        tx = Transform()
        tx.set_position(v0.x, v0.y, v0.z)
        tx.set_scale(0.06, 0.06, 0.06)
        m2 = global_tx @ tx.matrix()
        mv2 = self.view @ m2
        ShaderLib.set_uniform("MVP", self.project @ mv2)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m2).inverse().transposed())
        Primitives.draw("cube")

        if hit and hit_point is not None:
            tx3 = Transform()
            tx3.set_position(*hit_point)
            m3 = global_tx @ tx3.matrix()
            mv3 = self.view @ m3
            ShaderLib.set_uniform("MVP", self.project @ mv3)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m3).inverse().transposed()
            )
            Primitives.draw("smallSphere")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.ray_end.y += _STEP
        elif key == Qt.Key_Down:
            self.ray_end.y -= _STEP
        elif key == Qt.Key_Left:
            self.ray_end.x -= _STEP
        elif key == Qt.Key_Right:
            self.ray_end.x += _STEP
        elif key == Qt.Key_W:
            self.ray_start.y += _STEP
        elif key == Qt.Key_Z:
            self.ray_start.y -= _STEP
        elif key == Qt.Key_A:
            self.ray_start.x -= _STEP
        elif key == Qt.Key_S:
            self.ray_start.x += _STEP
        self.update()
        super().keyPressEvent(event)


def _calc_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    e1 = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
    e2 = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
    n = np.cross(e1, e2)
    length = np.linalg.norm(n)
    if length > 0:
        n = n / length
    return Vec3(*n)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangles", type=int, default=50)
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
    window = MainWindow(num_triangles=args.triangles)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

Note: same `VAOFactory`/`VAOType`/`VertexData` import-path verification as Task 4. `Primitives.create(Prims.SPHERE, "smallSphere", 0.05, 10)` mirrors the C++'s tiny hit-point sphere radius (`0.05`, vs `0.2` in RaySphere).

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x Collisions/RayTriangle/main.py
cd Collisions/RayTriangle && uv run --script main.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 3: Write README.md**

Create `Collisions/RayTriangle/README.md`:

```markdown
# RayTriangle

![](RayTriangle.png)

50 randomly-placed triangles (configurable via `--triangles N`),
re-tested every frame against one ray you move with the keyboard
(Moller-Trumbore intersection). A hit triangle draws wireframe with a
small marker sphere at the exact hit point.

## Controls
`up`/`down`/`left`/`right` : move the ray's end point
`w`/`z` : move the ray's start point up/down, `a`/`s` : left/right
Left-drag : orbit, Right-drag : pan, Wheel : zoom
```

- [ ] **Step 4: Add the root README row**

```markdown
| <a href="Collisions/RayTriangle"><img src="Collisions/RayTriangle/RayTriangle.png" width="220"></a> | [Collisions/RayTriangle](Collisions/RayTriangle) | N triangles tested every frame against a keyboard-moved ray (Moller-Trumbore) |
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/RayTriangle/
git commit -m "feat(ray-triangle): add OpenGL ray/triangle collision demo"
```

---

## Task 7: RayTriangle (WebGPU)

**Files:**
- Create: `Collisions/RayTriangle/main_webgpu.py`
- Create: `Collisions/RayTriangle/RayTriangleShader.wgsl`

**Interfaces:**
- Consumes: `ray_triangle_intersect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks. Independent of Task 6.

**Design notes:** no animation timer here either (same as Task 6) -- **no `closeEvent` override needed**. Triangles are real 3-vertex geometry (no baked-mesh stand-in needed, unlike spheres) -- each triangle gets its own small vertex buffer (3 verts, position+normal), drawn via the object pool. `v0` markers use the baked `cube` mesh (confirmed in the baked set). **Draw-count sizing:** `_DRAW_POOL_SIZE = 100` (50 triangles + 50 `v0` cube markers -- both drawn through the same pipeline/pool since both are ordinary triangle-list meshes with position+normal attributes, just different vertex data per draw). The ray line (2 points, rebuilt each frame since the endpoints move) and the single hit-point marker (0 or 1 point, rebuilt each frame) both use the Dynamic-marker pattern -- one line-list draw, one point-list draw, neither pooled. Hit triangles are tinted (same red-shift approach as Task 5) instead of wireframe.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/RayTriangle/RayTriangleShader.wgsl` -- reuse the exact same 3-part structure as `Collisions/RaySphere/RaySphereShader.wgsl` (Task 5: `vs_main`/`fs_main` for lit triangle-mesh objects, `vs_line`/`fs_line` for the line/point markers) -- copy that file's content verbatim into this one (the shader logic is backend-generic, not RaySphere-specific).

- [ ] **Step 2: Write main_webgpu.py**

Create `Collisions/RayTriangle/main_webgpu.py`, following the same `WebGPUScene(WebGPUWidget)` shape as Task 3/5 (mouse/keyboard handlers, `_create_render_buffer`, etc.), with these RayTriangle-specific pieces:

- `_DRAW_POOL_SIZE = 100` (50 triangle draws + 50 `v0`-cube-marker draws), one shared `draw_index` counter.
- Each triangle's own 3-vertex position+normal buffer built once at scene-construction time (triangles are static once spawned -- only the ray moves) and stored per-triangle; `_v0_marker` reuses `PrimData.primitive(Prims.CUBE.value)`.
- Line pipeline (`vs_line`/`fs_line`, `line_list` topology): one draw per frame for the ray (`ray_start`→`ray_end`, white, rebuilt from current key-driven endpoint state).
- Point pipeline (`vs_line`/`fs_line`, `point_list` topology): one draw per frame for the current hit point, if any (0 or 1 point -- when no triangle is hit this frame, `render_pass.draw(0)` is a valid no-op draw, or simply skip the draw call that frame).
- Re-test every triangle against the current ray every `paintWebGPU` call (matching Task 6's every-frame-not-timer-gated behaviour) using `ray_triangle_intersect` imported unchanged.
- Same `Up`/`Down`/`Left`/`Right`/`W`/`Z`/`A`/`S` key handling as Task 6, moving `ray_end`/`ray_start`, calling `self.update()`.
- No `closeEvent` override (no repeating timer in this demo).

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Collisions/RayTriangle/main_webgpu.py
cd Collisions/RayTriangle && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Collisions/RayTriangle/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same 50-triangle setup independently,
with real per-triangle geometry (no baked-mesh stand-in needed). A hit
triangle is tinted red instead of drawn wireframe.
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/RayTriangle/main_webgpu.py Collisions/RayTriangle/RayTriangleShader.wgsl Collisions/RayTriangle/README.md
git commit -m "feat(ray-triangle): add WebGPU entry point"
```

---

## Task 8: SpherePlane (OpenGL)

**Files:**
- Create: `Collisions/SpherePlane/main.py`
- Create: `Collisions/SpherePlane/README.md`

**Interfaces:**
- Consumes: `sphere_plane_collide` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:** ported from `NGL9Demos/Collisions/SpherePlane/src/NGLScene.cpp`. **50 spheres by default** (`--spheres N`). Spawn/respawn: `pos = (uniform(-6,6), 8, uniform(-6,6))`, `dir = (0,-1,0)` (falling), `radius = 0.2` (fixed, not randomised -- matches the C++ exactly). Plane: `center = (0,0,0)`, `width = depth = 5`, tiltable via `Up`/`Down` (rotate about world X by `±1°` per press, accumulated) and `Left`/`Right` (rotate about world Z by `±1°` per press, accumulated) -- see the Global Constraints' plane-rendering deviation note (rendered as a static local quad with the accumulated tilt applied as a draw-time rotation, not regenerated vertices). Timer 130ms -- **needs the `closeEvent` fix**. Every tick: move every sphere (`pos += dir`), test each against the plane via `sphere_plane_collide`; a colliding sphere's direction is set TO the plane's current normal (an intentional, slightly odd "bounce" rule straight from the C++ -- do not "fix" it) and its hit flag is set (wireframe). Every 20 ticks (~2.6s), ALL spheres respawn to a fresh random position regardless of hit state. **The C++ never wires up a `Space` toggle for this sub-demo despite declaring an `m_animate` flag** -- faithfully leave animation always-on with no pause control, matching the source exactly (don't "fix" this either). Camera: `look_at((0,0,15), origin, (0,1,0))`, background grey `(0.4,0.4,0.4,1)`.

- [ ] **Step 1: Write main.py**

Create `Collisions/SpherePlane/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""SpherePlane: N falling spheres collide with a tiltable plane.

Ported from NGL9Demos/Collisions/SpherePlane -- default 50 spheres
(configurable via --spheres), fixed 0.2 radius, respawn every 20 ticks,
plane tilt via Up/Down (X axis) and Left/Right (Z axis). No pause
control (the C++ declares but never wires up an animate toggle here --
faithfully left as always-on).
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_plane_collide  # noqa: E402

_PLANE_WIDTH = 5.0
_PLANE_DEPTH = 5.0
_RESPAWN_EVERY = 20


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(random.uniform(-6, 6), 8.0, random.uniform(-6, 6)),
        "dir": Vec3(0.0, -1.0, 0.0),
        "radius": 0.2,
        "hit": False,
    }


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_spheres: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("SpherePlane")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.plane_xrot = 0.0
        self.plane_zrot = 0.0
        self.tick_count = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(
            Prims.TRIANGLE_PLANE,
            "plane",
            _PLANE_WIDTH,
            _PLANE_DEPTH,
            1,
            1,
            Vec3(0, 1, 0),
        )
        self.animation_timer.start(130)

    def _plane_normal(self) -> Vec3:
        rot = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        n = (
            rot.mult_vec3(Vec3(0, 1, 0))
            if hasattr(rot, "mult_vec3")
            else rot @ Vec3(0, 1, 0)
        )
        return n

    def _on_tick(self) -> None:
        normal = self._plane_normal()
        normal_np = np.array([normal.x, normal.y, normal.z])
        for s in self.spheres:
            s["pos"] = s["pos"] + s["dir"]
            hit = sphere_plane_collide(
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                s["radius"],
                np.array([0.0, 0.0, 0.0]),
                normal_np,
                _PLANE_WIDTH,
                _PLANE_DEPTH,
            )
            if hit:
                s["dir"] = normal
                s["hit"] = True

        self.tick_count += 1
        if self.tick_count >= _RESPAWN_EVERY:
            self.tick_count = 0
            for s in self.spheres:
                fresh = _spawn_sphere()
                s["pos"], s["dir"], s["radius"], s["hit"] = (
                    fresh["pos"],
                    fresh["dir"],
                    fresh["radius"],
                    fresh["hit"],
                )
        self.update()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        plane_tilt = Mat4().rotate_z(self.plane_zrot) @ Mat4().rotate_x(self.plane_xrot)
        m = global_tx @ plane_tilt
        mv = self.view @ m
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(m).inverse().transposed())
        Primitives.draw("plane")

        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m2 = global_tx @ tx.matrix()
            mv2 = self.view @ m2
            ShaderLib.set_uniform("MVP", self.project @ mv2)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m2).inverse().transposed()
            )
            Primitives.draw("sphere")
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Up:
            self.plane_xrot += 1.0
        elif key == Qt.Key_Down:
            self.plane_xrot -= 1.0
        elif key == Qt.Key_Left:
            self.plane_zrot -= 1.0
        elif key == Qt.Key_Right:
            self.plane_zrot += 1.0
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spheres", type=int, default=50)
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
    window = MainWindow(num_spheres=args.spheres)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

Note: verify `Mat4 @ Vec3` (or a `.mult_vec3`/equivalent method) actually rotates a `Vec3` by a `Mat4` and returns a `Vec3` -- `_plane_normal`'s fallback (`hasattr` check) covers either API shape, but confirm which one this repo's `Mat4` actually exposes (grep `def __matmul__\|def mult_vec3\|def mult_point` in `/Users/jmacey/teaching/Code/PyNGL/src/ncca/ngl/mat4.py`) and simplify to the real one before relying on the try-both fallback. Also verify `Primitives.create(Prims.TRIANGLE_PLANE, name, width, depth, wp, dp, normal)`'s exact parameter order against an existing user (e.g. grep `TRIANGLE_PLANE` across this repo).

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x Collisions/SpherePlane/main.py
cd Collisions/SpherePlane && uv run --script main.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 3: Write README.md**

Create `Collisions/SpherePlane/README.md`:

```markdown
# SpherePlane

![](SpherePlane.png)

50 spheres (configurable via `--spheres N`) continuously fall and
collide with a 5x5 plane you can tilt. A colliding sphere switches to
moving along the plane's current normal and draws wireframe; every 20
ticks all spheres respawn at a fresh random position above the plane.

## Controls
`up`/`down` : tilt the plane about world X, `left`/`right` : tilt about world Z
Left-drag : orbit, Right-drag : pan, Wheel : zoom
```

- [ ] **Step 4: Add the root README row**

```markdown
| <a href="Collisions/SpherePlane"><img src="Collisions/SpherePlane/SpherePlane.png" width="220"></a> | [Collisions/SpherePlane](Collisions/SpherePlane) | N falling spheres collide with a tiltable plane |
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/SpherePlane/
git commit -m "feat(sphere-plane): add OpenGL sphere/plane collision demo"
```

---

## Task 9: SpherePlane (WebGPU)

**Files:**
- Create: `Collisions/SpherePlane/main_webgpu.py`
- Create: `Collisions/SpherePlane/SpherePlaneShader.wgsl`

**Interfaces:**
- Consumes: `sphere_plane_collide` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks. Independent of Task 8.

**Design notes:** the plane is a small hand-built numpy quad (5x5 in local XZ, same pattern as `Blending/BlendingWebGPU.py`'s `quad()` / `MatrixStack/main_webgpu.py`'s `quad_floor()`), tilted via the same draw-time rotation as Task 8's OpenGL version. **Draw-count sizing:** `_DRAW_POOL_SIZE = 51` (50 spheres, using the baked `octahedron` mesh, + 1 plane draw sharing the same pooled pipeline since both need per-draw `M`/`MVP`/`normal_matrix`/colour -- the plane is just object index 0 every frame). No line/point Dynamic-marker draws needed here (no rays, no hit-point markers in this sub-demo -- only wireframe-vs-tint on hit). Timer 130ms -- **needs `closeEvent`**.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/SpherePlane/SpherePlaneShader.wgsl` -- reuse `Collisions/SphereSphere/SphereSphereShader.wgsl`'s `vs_main`/`fs_main` content verbatim (Task 3) -- the lit-mesh-with-per-draw-colour shader is identical here.

- [ ] **Step 2: Write main_webgpu.py**

Create `Collisions/SpherePlane/main_webgpu.py`, following the same `WebGPUScene(WebGPUWidget)` + per-draw buffer-pool shape as Task 3, with these SpherePlane-specific pieces:

- `_DRAW_POOL_SIZE = 51`; `draw_index = 0` draws the plane (quad geometry, tilt rotation `Mat4().rotate_z(zrot) @ Mat4().rotate_x(xrot)` folded into its model matrix, fixed yellow colour), `draw_index = 1..50` draw the spheres (`octahedron` mesh, yellow, tinted red when hit -- same tint approach as Tasks 5/7).
- `animation_timer` at 130ms driving `_on_tick`: same fall/collide/respawn-every-20-ticks logic as Task 8, importing `sphere_plane_collide` unchanged, computing the plane normal the same way (rotate `(0,1,0)` by the current tilt matrix).
- `Up`/`Down`/`Left`/`Right` key handling adjusts `plane_xrot`/`plane_zrot` exactly as Task 8.
- `closeEvent` stops `self.animation_timer`.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Collisions/SpherePlane/main_webgpu.py
cd Collisions/SpherePlane && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Collisions/SpherePlane/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same 50-sphere/tiltable-plane setup
independently. Spheres use the baked `octahedron` mesh, the plane is a
small hand-built quad; a colliding sphere is tinted red instead of drawn
wireframe.
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/SpherePlane/main_webgpu.py Collisions/SpherePlane/SpherePlaneShader.wgsl Collisions/SpherePlane/README.md
git commit -m "feat(sphere-plane): add WebGPU entry point"
```

---

## Task 10: BoundingBox (OpenGL)

**Files:**
- Create: `Collisions/BoundingBox/main.py`
- Create: `Collisions/BoundingBox/README.md`

**Interfaces:**
- Consumes: `sphere_sphere_collide` and `sphere_bbox_reflect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Design notes:** ported from `NGL9Demos/Collisions/BoundingBox/src/NGLScene.cpp` -- the most involved of the 5. **50 spheres by default** (`--spheres N`), **variable count at runtime** via `+`/`-` keys (never drops below 1). Spawn/reset: `pos = uniform(-20,20)` per axis, `dir` = a random unit vector, `radius = uniform(0.5, 2.5)`. Bounding cube: centred on the origin, half-extent `40` (i.e. `height=width=depth=80` in the C++'s terms), drawn as a 12-edge wireframe box. Timer 40ms -- **needs `closeEvent`**. Every tick: move every sphere (`pos += dir`); if the `S`-toggled `check_sphere_sphere` flag is on (default off), run the full `O(n^2)` all-pairs test via `sphere_sphere_collide` (on a collision, only the sphere in the outer/"Current" loop position reverses direction and gets flagged hit -- an asymmetric rule straight from the C++, keep it as-is); ALWAYS run the 6-wall reflection via `sphere_bbox_reflect` (any wall hit also sets the hit flag). Keys: `Escape`, `F` = fullscreen, `N` = windowed, `Space` = toggle animation, `S` = toggle sphere-sphere checking, `R` = reset all spheres to fresh random state, `Minus` = remove the last sphere (clamped to a minimum of 1), `Plus` = append one freshly-spawned sphere. Camera: `look_at((0,80,80), origin, (0,1,0))` -- a much larger, elevated view than the other 4 sub-demos, matching the much bigger `±40` play area. Background grey `(0.4,0.4,0.4,1)`.

- [ ] **Step 1: Write main.py**

Create `Collisions/BoundingBox/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""BoundingBox: N spheres bounce inside a cubic bounding box, with an
optional all-pairs sphere/sphere check.

Ported from NGL9Demos/Collisions/BoundingBox -- default 50 spheres
(configurable via --spheres), variable count at runtime (+/- keys,
minimum 1), half-extent-40 cube, optional sphere-sphere checking (S key,
default off).
"""

import argparse
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, logger, look_at, perspective
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from collision_maths import sphere_bbox_reflect, sphere_sphere_collide  # noqa: E402

_HALF_EXTENT = 40.0


def _random_unit_vec3() -> Vec3:
    v = np.random.normal(size=3)
    v = v / np.linalg.norm(v)
    return Vec3(*v)


def _spawn_sphere() -> dict:
    return {
        "pos": Vec3(
            random.uniform(-20, 20), random.uniform(-20, 20), random.uniform(-20, 20)
        ),
        "dir": _random_unit_vec3(),
        "radius": random.uniform(0.5, 2.5),
        "hit": False,
    }


_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom face (y = -h)
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top face (y = +h)
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # connecting edges
)


def _box_corners(h: float) -> list[Vec3]:
    return [
        Vec3(-h, -h, -h),
        Vec3(h, -h, -h),
        Vec3(h, -h, h),
        Vec3(-h, -h, h),
        Vec3(-h, h, -h),
        Vec3(h, h, -h),
        Vec3(h, h, h),
        Vec3(-h, h, h),
    ]


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, num_spheres: int = 50, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("BoundingBox")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.spheres = [_spawn_sphere() for _ in range(num_spheres)]
        self.animate = True
        self.check_sphere_sphere = False
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.box_vao = None

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 80, 80), Vec3(0, 0, 0), Vec3(0, 1, 0))
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        from ncca.ngl.opengl import VAOFactory, VAOType

        self.box_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        self.animation_timer.start(40)

    def _on_tick(self) -> None:
        if not self.animate:
            return
        for s in self.spheres:
            s["pos"] = s["pos"] + s["dir"]

        if self.check_sphere_sphere:
            for current in self.spheres:
                for other in self.spheres:
                    if current is other:
                        continue
                    if sphere_sphere_collide(
                        np.array([other["pos"].x, other["pos"].y, other["pos"].z]),
                        other["radius"],
                        np.array(
                            [current["pos"].x, current["pos"].y, current["pos"].z]
                        ),
                        current["radius"],
                    ):
                        current["dir"] = current["dir"] * -1.0
                        current["hit"] = True

        for s in self.spheres:
            hit, new_dir = sphere_bbox_reflect(
                np.array([s["pos"].x, s["pos"].y, s["pos"].z]),
                np.array([s["dir"].x, s["dir"].y, s["dir"].z]),
                s["radius"],
                _HALF_EXTENT,
            )
            if hit:
                s["dir"] = Vec3(*new_dir)
                s["hit"] = True
        self.update()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.COLOUR)
        from ncca.ngl.opengl.vertex_data import VertexData

        corners = _box_corners(_HALF_EXTENT)
        verts: list[float] = []
        for a, b in _BOX_EDGES:
            verts.extend((corners[a].x, corners[a].y, corners[a].z))
            verts.extend((corners[b].x, corners[b].y, corners[b].z))
        data = np.array(verts, dtype=np.float32)
        with self.box_vao as vao:
            vao.set_data(VertexData(data, len(_BOX_EDGES) * 2))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            vao.set_num_indices(len(_BOX_EDGES) * 2)
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
            vao.draw()

        ShaderLib.use(DefaultShader.DIFFUSE)
        for s in self.spheres:
            gl.glPolygonMode(
                gl.GL_FRONT_AND_BACK, gl.GL_LINE if s["hit"] else gl.GL_FILL
            )
            ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
            tx = Transform()
            tx.set_position(s["pos"].x, s["pos"].y, s["pos"].z)
            tx.set_scale(s["radius"], s["radius"], s["radius"])
            m = global_tx @ tx.matrix()
            mv = self.view @ m
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform(
                "normalMatrix", Mat3.from_mat4(m).inverse().transposed()
            )
            Primitives.draw("sphere")
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 350.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_F:
            self.showFullScreen()
        elif key == Qt.Key_N:
            self.showNormal()
        elif key == Qt.Key_Space:
            self.animate = not self.animate
        elif key == Qt.Key_S:
            self.check_sphere_sphere = not self.check_sphere_sphere
        elif key == Qt.Key_R:
            self.spheres = [_spawn_sphere() for _ in range(len(self.spheres))]
        elif key == Qt.Key_Minus:
            if len(self.spheres) > 1:
                self.spheres.pop()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.spheres.append(_spawn_sphere())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spheres", type=int, default=50)
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
    window = MainWindow(num_spheres=args.spheres)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

Note: same `VAOFactory`/`VertexData` import-path verification as prior tasks. The O(n^2) sphere-sphere check (`Key_S`, off by default) is intentionally quadratic, matching the C++ exactly -- do not optimise it, and do not enable it by default.

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x Collisions/BoundingBox/main.py
cd Collisions/BoundingBox && uv run --script main.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 3: Write README.md**

Create `Collisions/BoundingBox/README.md`:

```markdown
# BoundingBox

![](BoundingBox.png)

50 spheres (configurable via `--spheres N`) bounce around inside an
80-unit cubic bounding box, reflecting off any of its 6 walls. An
optional, off-by-default all-pairs sphere/sphere check can be toggled on
top of the wall collisions.

## Controls
`space` : pause/resume, `s` : toggle sphere/sphere checking (off by default)
`r` : reset all spheres, `+`/`-` : add/remove a sphere (minimum 1)
`f` : fullscreen, `n` : windowed
Left-drag : orbit, Right-drag : pan, Wheel : zoom
```

- [ ] **Step 4: Add the root README row**

```markdown
| <a href="Collisions/BoundingBox"><img src="Collisions/BoundingBox/BoundingBox.png" width="220"></a> | [Collisions/BoundingBox](Collisions/BoundingBox) | N spheres bounce inside a cubic bounding box, optional sphere/sphere checking |
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/BoundingBox/
git commit -m "feat(bounding-box): add OpenGL sphere/bbox collision demo"
```

---

## Task 11: BoundingBox (WebGPU)

**Files:**
- Create: `Collisions/BoundingBox/main_webgpu.py`
- Create: `Collisions/BoundingBox/BoundingBoxShader.wgsl`

**Interfaces:**
- Consumes: `sphere_sphere_collide` and `sphere_bbox_reflect` from `Collisions/collision_maths.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks. Independent of Task 10.

**Design notes:** the trickiest WebGPU task in this phase, because the C++'s sphere count is genuinely unbounded (`+` has no upper limit). A fixed-size WebGPU buffer pool needs a hard ceiling; this plan picks **`_POOL_CAP = 200`** (4x the default 50, generous headroom for interactive experimentation) and clamps the `Plus` key to a no-op once the pool is full -- a WebGPU-architecture necessity, not a simplification of the demo's actual default behaviour (50 spheres, add/remove by 1, which is fully faithful and exercised long before the cap matters). **Draw-count sizing:** the object pool covers spheres only, sized to `_POOL_CAP = 200` (the box wireframe is a separate single line-list draw via the Dynamic-marker pattern, not pooled, since its 24 vertices are static and don't need per-draw uniform isolation the way per-sphere transforms do -- one shared uniform buffer for the box's `MVP` is enough). Timer 40ms -- **needs `closeEvent`**.

- [ ] **Step 1: Write the WGSL shader**

Create `Collisions/BoundingBox/BoundingBoxShader.wgsl` -- reuse `Collisions/RaySphere/RaySphereShader.wgsl`'s full content verbatim (Task 5: `vs_main`/`fs_main` for lit sphere objects, `vs_line`/`fs_line` for the box wireframe) -- both pipelines are needed here too.

- [ ] **Step 2: Write main_webgpu.py**

Create `Collisions/BoundingBox/main_webgpu.py`, following the same `WebGPUScene(WebGPUWidget)` + per-draw buffer-pool shape as prior WebGPU tasks, with these BoundingBox-specific pieces:

- `_POOL_CAP = 200`; sphere buffer pool pre-allocated to this size once at init (mirroring Task 5's pool-of-52 pattern, just bigger); `paintWebGPU` draws only `len(self.spheres)` of the `_POOL_CAP` slots each frame (`draw_index` runs `0..len(self.spheres)-1`, the rest of the pool sits idle that frame -- this is safe, since idle slots are simply never bound/drawn, not aliased).
- Box wireframe: one static `line_list` draw per frame (24 vertices, box never moves -- only the camera/mouse orbit changes its `MVP`, recomputed each frame like every other demo's global transform), using the same 12-edge/8-corner geometry as Task 10's `_BOX_EDGES`/`_box_corners`.
- `animation_timer` at 40ms driving `_on_tick`: same move/optional-sphere-sphere/always-bbox-reflect logic as Task 10, importing both `sphere_sphere_collide` and `sphere_bbox_reflect` unchanged.
- `Space`/`S`/`R`/`Minus`/`Plus`(clamped to `_POOL_CAP`) key handling, same semantics as Task 10 (no `F`/`N` fullscreen toggle needed on the WebGPU `QWidget` side -- `showFullScreen`/`showNormal` are a `QOpenGLWindow`-specific convenience the WebGPU sibling doesn't have an equivalent hook for; omit them here, matching how other WebGPU siblings in this repo already drop GL-only window-chrome keys).
- `closeEvent` stops `self.animation_timer`.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Collisions/BoundingBox/main_webgpu.py
cd Collisions/BoundingBox && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ../..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Collisions/BoundingBox/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` reproduces the same default 50-sphere setup and
controls independently (minus `f`/`n` fullscreen, which has no WebGPU
`QWidget` equivalent here). The sphere count is capped at 200 -- a
fixed-size GPU buffer pool needs a ceiling, unlike the C++'s unbounded
array; the default behaviour (50 spheres, add/remove by 1) is unaffected.
```

- [ ] **Step 5: Commit**

```bash
git add Collisions/BoundingBox/main_webgpu.py Collisions/BoundingBox/BoundingBoxShader.wgsl Collisions/BoundingBox/README.md
git commit -m "feat(bounding-box): add WebGPU entry point"
```

---

## Final steps (after all 11 tasks)

- [ ] **Run full verification**

```bash
cd Collisions
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
cd ..
```
Expected: ruff clean, all 21 collision-maths tests pass.

- [ ] **Confirm all 5 root README rows are present and correctly formatted**

Each demo task above adds its own row inline; this step is just a final visual check that all 5 rows exist, in a sensible section, with no duplicates.

- [ ] **Report to Jon**

List the 5 `.png` screenshots that still need capturing (`RaySphere.png`, `RayTriangle.png`, `SpherePlane.png`, `SphereSphere.png`, `BoundingBox.png`) — all under their respective `Collisions/<SubDemo>/` folders. Flag the known risk areas for a human/reviewer to double-check interactively, since no automated smoketest exercises them: BoundingBox's `+`/`-`/`R`/`S` keys and the 200-sphere cap; SpherePlane's tilt controls and 20-tick respawn cycle; RaySphere/RayTriangle's hit-point marker rendering on WebGPU (the Dynamic-marker point/line pipelines); and the `closeEvent` fix on all 4 timer-driven sub-demos (RaySphere, SpherePlane, SphereSphere, BoundingBox) x 2 backends each.
