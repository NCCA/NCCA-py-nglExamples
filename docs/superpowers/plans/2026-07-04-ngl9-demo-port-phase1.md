# NGL9Demos Port — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 10 "straightforward reuse" C++ NGL9Demos (Camera, ColourObj, CurveDemos, QuatSlerp, KleinBottle, FrustumCull, PointCloud, AnimatedTextures, Interpolation, ImageHeightMap) to Python PyNGLDemos, each a self-contained PySide6/QOpenGLWindow demo following the VAOPrimitives convention.

**Architecture:** Each demo is `<DemoName>/main.py` (a `QOpenGLWindow` subclass, standard mouse orbit/pan/zoom copied from `VAOPrimitives/main.py`) + `README.md` + `shaders/*.glsl` + any needed `models/`/`textures/`/`data/`. No changes to the `ncca.ngl` library.

**Tech Stack:** Python 3.13, `ncca.ngl` (local package at `/Volumes/teaching/Code/PyNGL`), PySide6, PyOpenGL, `uv run --script`.

## Global Constraints

- No edits to `/Volumes/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder.
- Every `main.py` starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and is a `QOpenGLWindow` subclass — copy the skeleton (imports, `MainWindow.__init__` camera/mouse state, `keyPressEvent`/`mousePressEvent`/`mouseMoveEvent`/`mouseReleaseEvent`/`wheelEvent`, `DebugApplication`, `if __name__ == "__main__":` block) verbatim from `/Volumes/teaching/Code/PyNGLDemos/VAOPrimitives/main.py`, only customizing `initializeGL`/`paintGL`/`resizeGL` and window title per demo.
- Verification for every demo: run `QT_QPA_PLATFORM=offscreen uv run --script <DemoName>/main.py --smoketest` from the repo root; the demo must exit 0 within 5 seconds printing `SMOKETEST OK` and no traceback. Each `main.py` supports this via a `--smoketest` CLI flag (checked in `__main__`) that runs the app for one paint pass via a `QTimer.singleShot(200, app.quit)` then exits — added once per demo, shown explicitly in each task.
- Every demo has a `README.md` with a one-paragraph description (adapted from the corresponding NGL9Demos README) and controls list.
- Commit after each task with `git add <DemoName>/ && git commit -m "feat: add <DemoName> demo"`.
- Work happens in the `agent/ngl9-demo-port` branch, worktree at `.worktrees/ngl9-demo-port` (already created).

---

## Task 1: Camera

**Files:**
- Create: `Camera/main.py`
- Create: `Camera/README.md`
- Create: `Camera/shaders/PBRVertex.glsl`, `Camera/shaders/PBRFragment.glsl` (copy from `ShadingModels/shaders/`)
- Create: `Camera/uvn_camera.py` (hand-ported UVN camera class — the C++ demo's own `Camera` class, not part of `ncca.ngl`)

**Interfaces:**
- Produces: `UVNCamera` class in `Camera/uvn_camera.py` with constructor `UVNCamera(eye: Vec3, look: Vec3, up: Vec3, fov: float, aspect: float, near: float, far: float)`, methods `set_shape(fov, aspect, near, far)`, `move_eye(dx, dy, dz)`, `move_look(dx, dy, dz)`, `move_both(dx, dy, dz)`, `slide(dx, dy, dz)`, `roll(deg)`, `pitch(deg)`, `yaw(deg)`, properties `view: Mat4`, `project: Mat4`.

- [ ] **Step 1: Copy PBR shaders**

```bash
mkdir -p Camera/shaders
cp ShadingModels/shaders/PBRVertex.glsl Camera/shaders/PBRVertex.glsl
cp ShadingModels/shaders/PBRFragment.glsl Camera/shaders/PBRFragment.glsl
```

- [ ] **Step 2: Write the UVN camera class**

Create `Camera/uvn_camera.py`:

```python
"""A hand-rolled UVN camera, ported from NGL9Demos/Camera (not part of ncca.ngl)."""

from __future__ import annotations

from ncca.ngl import Mat4, Vec3, look_at, perspective


class UVNCamera:
    def __init__(
        self,
        eye: Vec3,
        look: Vec3,
        up: Vec3,
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ) -> None:
        self.eye = Vec3(eye.x, eye.y, eye.z)
        self.look = Vec3(look.x, look.y, look.z)
        self.up = Vec3(up.x, up.y, up.z)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self._update_view()
        self._update_projection()

    def _update_view(self) -> None:
        self.view = look_at(self.eye, self.look, self.up)
        n = (self.look - self.eye).normalized()
        u = n.cross(self.up).normalized()
        v = u.cross(n).normalized()
        self.n = n
        self.u = u
        self.v = v

    def _update_projection(self) -> None:
        self.project = perspective(self.fov, self.aspect, self.near, self.far)

    def set_shape(self, fov: float, aspect: float, near: float, far: float) -> None:
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self._update_projection()

    def move_eye(self, dx: float, dy: float, dz: float) -> None:
        self.eye = self.eye + self.u * dx + self.v * dy + self.n * dz
        self._update_view()

    def move_look(self, dx: float, dy: float, dz: float) -> None:
        self.look = self.look + self.u * dx + self.v * dy + self.n * dz
        self._update_view()

    def move_both(self, dx: float, dy: float, dz: float) -> None:
        offset = self.u * dx + self.v * dy + self.n * dz
        self.eye = self.eye + offset
        self.look = self.look + offset
        self._update_view()

    def slide(self, dx: float, dy: float, dz: float) -> None:
        self.move_both(dx, dy, dz)

    def roll(self, degrees: float) -> None:
        r = Mat4().rotate_z(degrees)
        self.up = (r.mult_vec3(self.up)).normalized()
        self._update_view()

    def pitch(self, degrees: float) -> None:
        r = Mat4().rotate_x(degrees)
        self.look = self.eye + r.mult_vec3(self.look - self.eye)
        self._update_view()

    def yaw(self, degrees: float) -> None:
        r = Mat4().rotate_y(degrees)
        self.look = self.eye + r.mult_vec3(self.look - self.eye)
        self._update_view()
```

Note: `Mat4.mult_vec3` may not exist under that exact name — if `Mat4 @ Vec3` (via `__matmul__`) or `Mat4.mult_point`/`mult_vector` is the actual API, use that instead. Confirm with:

```bash
grep -n "def mult\|def __matmul__" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/mat4.py
```

and adjust `roll`/`pitch`/`yaw` to use whichever method multiplies a `Mat4` by a `Vec3` and returns a `Vec3`.

- [ ] **Step 3: Verify the Mat4 API used above**

Run: `grep -n "def mult\|def __matmul__\|def rotate_x\|def rotate_y\|def rotate_z" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/mat4.py`
Expected: shows the exact method names; update Step 2's code to match before proceeding if names differ.

- [ ] **Step 4: Write main.py**

Copy `VAOPrimitives/main.py` to `Camera/main.py`, then replace `initializeGL`, `paintGL`, `resizeGL`, and add a `--smoketest` flag, as follows (full file):

```python
#!/usr/bin/env -S uv run --script
"""Camera demo: 4 selectable UVN cameras viewing a lit PBR scene."""

import sys
import traceback

import OpenGL.GL as gl
from ncca.ngl import (
    Mat3,
    Mat4,
    Primitives,
    Prims,
    ShaderLib,
    Transform,
    Vec3,
    logger,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

from uvn_camera import UVNCamera


class MainWindow(QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.mouse_global_tx: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("Camera")

        self.rotate: bool = False
        self.translate: bool = False
        self.spin_x_face: int = 0
        self.spin_y_face: int = 0
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.1

        self.cameras: list[UVNCamera] = []
        self.camera_index: int = 3
        self.rotation: float = 0.0
        self.light_on: list[bool] = [True, True, True]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        aspect = self.window_width / self.window_height
        self.cameras = [
            UVNCamera(Vec3(0, 0, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0),
            UVNCamera(Vec3(0, 20, 0.001), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0),
            UVNCamera(Vec3(20, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0),
            UVNCamera(Vec3(8, 6, 20), Vec3(0, 0, 0), Vec3(0, 1, 0), 65.0, aspect, 0.2, 150.0),
        ]

        ShaderLib.load_shader(
            "PBR", "shaders/PBRVertex.glsl", "shaders/PBRFragment.glsl"
        )
        ShaderLib.use("PBR")
        ShaderLib.set_uniform("lightPositions[0]", -10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[0]", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("lightPositions[1]", 10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[1]", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("lightPositions[2]", -10.0, 4.0, 10.0)
        ShaderLib.set_uniform("lightColours[2]", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("albedo", 0.8, 0.1, 0.1)
        ShaderLib.set_uniform("metallic", 0.6)
        ShaderLib.set_uniform("roughness", 0.3)
        ShaderLib.set_uniform("ao", 1.0)

        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "ground", 30, 30, 20, 20, Vec3(0, 1, 0))

    def _update_lights(self) -> None:
        for i in range(3):
            c = 400.0 if self.light_on[i] else 0.0
            ShaderLib.set_uniform(f"lightColours[{i}]", c, c, c)

    def load_matrices(self, camera: UVNCamera, tx: Transform) -> None:
        m = tx.matrix()
        mv = camera.view @ self.mouse_global_tx @ m
        mvp = camera.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("camPos", camera.eye.x, camera.eye.y, camera.eye.z)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        camera = self.cameras[self.camera_index]
        ShaderLib.use("PBR")
        self._update_lights()

        tx = Transform()
        tx.set_position(0, -1, 0)
        tx.set_scale(15, 1, 15)
        self.load_matrices(camera, tx)
        Primitives.draw("ground")

        tx.reset()
        tx.set_position(0, 0, 0)
        self.load_matrices(camera, tx)
        Primitives.draw("teapot")

        tx.reset()
        tx.set_position(-3, 0.5, 0)
        tx.set_rotation(0, self.rotation, 0)
        self.load_matrices(camera, tx)
        Primitives.draw("cube")

        tx.reset()
        tx.set_position(3, 0, 0)
        tx.set_scale(0.05, 0.05, 0.05)
        self.load_matrices(camera, tx)
        Primitives.draw("football")

        self.rotation = (self.rotation + 1.0) % 360.0
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        aspect = float(w) / h
        for camera in self.cameras:
            camera.set_shape(camera.fov, aspect, camera.near, camera.far)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        camera = self.cameras[self.camera_index]
        if key == Qt.Key_Escape:
            self.close()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            self.camera_index = key - Qt.Key_1
        elif key == Qt.Key_Left:
            camera.move_eye(-0.5, 0, 0)
        elif key == Qt.Key_Right:
            camera.move_eye(0.5, 0, 0)
        elif key == Qt.Key_Up:
            camera.move_eye(0, 0, -0.5)
        elif key == Qt.Key_Down:
            camera.move_eye(0, 0, 0.5)
        elif key == Qt.Key_R:
            camera.roll(3.0)
        elif key == Qt.Key_Y:
            camera.yaw(3.0)
        elif key == Qt.Key_P:
            camera.pitch(3.0)
        elif key == Qt.Key_Z:
            self.light_on[0] = not self.light_on[0]
        elif key == Qt.Key_X:
            self.light_on[1] = not self.light_on[1]
        elif key == Qt.Key_C:
            self.light_on[2] = not self.light_on[2]
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            camera.set_shape(camera.fov + 1.0, camera.aspect, camera.near, camera.far)
        elif key == Qt.Key_Minus:
            camera.set_shape(camera.fov - 1.0, camera.aspect, camera.near, camera.far)
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        num_pixels = event.angleDelta()
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
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


if __name__ == "__main__":
    format: QSurfaceFormat = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    smoketest = "--smoketest" in sys.argv
    if "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if smoketest:
        QTimer.singleShot(200, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

- [ ] **Step 5: Make executable and smoke-test**

```bash
chmod +x Camera/main.py
cd Camera && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: prints `SMOKETEST OK` and exits 0, no traceback.

- [ ] **Step 6: Write README.md**

Create `Camera/README.md`:

```markdown
# Camera

Demonstrates a hand-rolled UVN camera (`uvn_camera.py`) with 4 selectable views (front,
top, side, perspective) looking at a PBR-lit teapot, spinning cube, and football.

## Controls
- `1`-`4` : switch active camera
- Arrow keys : move the active camera's eye
- `r` / `y` / `p` : roll / yaw / pitch the active camera
- `z` / `x` / `c` : toggle the 3 scene lights
- `+` / `-` : adjust field of view
- Left-drag : orbit scene, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 7: Commit**

```bash
git add Camera/
git commit -m "feat: add Camera demo"
```

---

## Task 2: ColourObj

**Files:**
- Create: `ColourObj/main.py`
- Create: `ColourObj/README.md`
- Create: `ColourObj/shaders/ColourVertex.glsl`, `ColourObj/shaders/ColourFragment.glsl`
- Create: `ColourObj/models/face_mesh_neutral.obj` (copied from NGL9Demos)
- Create: `ColourObj/colour_obj.py` (custom `Obj` subclass + VAO builder)

**Interfaces:**
- Produces: `ColourObj` class in `ColourObj/colour_obj.py` with `ColourObj.from_file(path) -> ColourObj`, `.create_colour_vao() -> SimpleVAO`-like object with `.draw()`, `.bbox`.

- [ ] **Step 1: Copy the model asset**

```bash
mkdir -p ColourObj/models ColourObj/shaders
cp /Volumes/teaching/NGL9Demos/ColourObj/models/face_mesh_neutral.obj ColourObj/models/
```

- [ ] **Step 2: Write the shaders**

Create `ColourObj/shaders/ColourVertex.glsl`:

```glsl
#version 410 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColour;

out vec3 vertColour;

uniform mat4 MVP;

void main()
{
    vertColour = inColour;
    gl_Position = MVP * vec4(inVert, 1.0);
}
```

Create `ColourObj/shaders/ColourFragment.glsl`:

```glsl
#version 410 core
in vec3 vertColour;
layout(location = 0) out vec4 fragColour;

void main()
{
    fragColour = vec4(vertColour, 1.0);
}
```

- [ ] **Step 3: Confirm the `Obj`/VAO low-level API before writing the subclass**

Run:
```bash
grep -n "_parse_vertex\|def create_vao\|class SimpleVAO\|def set_data\|def set_vertex_attribute" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/obj.py /Volumes/teaching/Code/PyNGL/src/ncca/ngl/base_mesh.py /Volumes/teaching/Code/PyNGL/src/ncca/ngl/simple_vao.py
```
Expected: shows the exact private method name for vertex-line parsing (`_parse_vertex`) and the `SimpleVAO`/`AbstractVAO` construction API (e.g. `set_data`, `set_vertex_attribute_pointer`, `draw`). Use these exact names in Step 4 — if they differ from what's assumed below, adjust accordingly.

- [ ] **Step 4: Write the ColourObj mesh class**

Create `ColourObj/colour_obj.py`:

```python
"""Loads the MeshLab-style 6-float-per-vertex OBJ variant used by NGL9Demos/ColourObj,
where each `v` line is `x y z r g b` (position + baked vertex colour)."""

from __future__ import annotations

import numpy as np
from ncca.ngl import Obj, Vec3, VAOFactory


class ColourObj(Obj):
    def __init__(self) -> None:
        super().__init__()
        self.colours: list[Vec3] = []
        self.vao = None

    @classmethod
    def from_file(cls, path: str) -> "ColourObj":
        obj = cls()
        obj.load(path)
        return obj

    def _parse_vertex(self, tokens: list[str]) -> None:
        values = [float(t) for t in tokens]
        self.add_vertex(Vec3(values[0], values[1], values[2]))
        if len(values) >= 6:
            self.colours.append(Vec3(values[3], values[4], values[5]))
        else:
            self.colours.append(Vec3(1.0, 1.0, 1.0))

    def create_colour_vao(self) -> None:
        verts: list[float] = []
        for face in self.faces:
            for i in range(3):
                vert_idx = face.vertex[i] - 1
                norm_idx = face.normal[i] - 1 if face.normal else -1
                uv_idx = face.uv[i] - 1 if face.uv else -1
                p = self.vertices[vert_idx]
                n = self.normals[norm_idx] if norm_idx >= 0 else Vec3(0, 1, 0)
                uv = self.uvs[uv_idx] if uv_idx >= 0 else (0.0, 0.0)
                c = self.colours[vert_idx]
                verts.extend([p.x, p.y, p.z, n.x, n.y, n.z, uv[0], uv[1], c.x, c.y, c.z])

        data = np.array(verts, dtype=np.float32)
        self.vao = VAOFactory.create_vao("simple", mode=4)  # GL_TRIANGLES
        self.vao.bind()
        self.vao.set_data(data)
        stride = 11 * 4
        self.vao.set_vertex_attribute_pointer(0, 3, "float", stride, 0)
        self.vao.set_vertex_attribute_pointer(1, 3, "float", stride, 3 * 4)
        self.vao.set_vertex_attribute_pointer(2, 2, "float", stride, 6 * 4)
        self.vao.set_vertex_attribute_pointer(3, 3, "float", stride, 8 * 4)
        self.vao.set_num_indices(len(verts) // 11)
        self.vao.unbind()

    def draw(self) -> None:
        if self.vao:
            self.vao.bind()
            self.vao.draw()
            self.vao.unbind()
```

Note: `Obj.faces`/`.vertices`/`.normals`/`.uvs` attribute names and `VAOFactory.create_vao`/`SimpleVAO` method names (`set_data`, `set_vertex_attribute_pointer`, `set_num_indices`, `bind`/`unbind`/`draw`) must be confirmed against Step 3's grep output — this is the highest-risk task in this phase since it relies on internals not exercised by any existing demo. If a method doesn't exist under this name, find its actual equivalent in `simple_vao.py`/`abstract_vao.py`/`vao_factory.py` and adjust.

- [ ] **Step 5: Write main.py**

Copy the `VAOPrimitives/main.py` skeleton to `ColourObj/main.py`, replacing `initializeGL`/`paintGL`/`resizeGL`:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Colour", "shaders/ColourVertex.glsl", "shaders/ColourFragment.glsl"
        )

        self.mesh = ColourObj.from_file("models/face_mesh_neutral.obj")
        self.mesh.create_colour_vao()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Colour")
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)
        self.mesh.draw()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

Add `from colour_obj import ColourObj` to the imports, set `self.setTitle("ColourObj")`, and add the `--smoketest` handling identical to Task 1 Step 4's `__main__` block.

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x ColourObj/main.py
cd ColourObj && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 7: Write README.md**

```markdown
# ColourObj

Loads an OBJ variant where each vertex line carries a baked RGB colour
(`x y z r g b`) and renders it with per-vertex colour, no lighting.

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 8: Commit**

```bash
git add ColourObj/
git commit -m "feat: add ColourObj demo"
```

---

## Task 3: CurveDemos

**Files:**
- Create: `CurveDemos/main.py`
- Create: `CurveDemos/README.md`

**Interfaces:**
- Consumes: `ncca.ngl.BezierCurve` (`add_point`, `get_point_on_curve(u)`, `.control_points`).

- [ ] **Step 1: Confirm BezierCurve control-point attribute name**

Run: `grep -n "control_points\|def add_point\|def get_point_on_curve" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/bezier_curve.py`
Expected: confirms `control_points` is the property name holding the list of `Vec3` points, and `get_point_on_curve(u: float) -> Vec3` signature.

- [ ] **Step 2: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `CurveDemos/main.py`:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))

        self.curve = BezierCurve()
        self.curve.add_point(Vec3(-5, 0, -5))
        self.curve.add_point(Vec3(-2, 2, 1))
        self.curve.add_point(Vec3(3, -3, -3))
        self.curve.add_point(Vec3(2, -6, 2))

        lod = 200
        self.curve_points = [self.curve.get_point_on_curve(i / (lod - 1)) for i in range(lod)]

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.COLOUR)
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

        gl.glBegin(gl.GL_LINE_STRIP)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        for p in self.curve_points:
            gl.glVertex3f(p.x, p.y, p.z)
        gl.glEnd()

        gl.glBegin(gl.GL_LINE_STRIP)
        ShaderLib.set_uniform("Colour", 1.0, 0.0, 0.0, 1.0)
        for p in self.curve.control_points:
            gl.glVertex3f(p.x, p.y, p.z)
        gl.glEnd()

        gl.glPointSize(4)
        gl.glBegin(gl.GL_POINTS)
        ShaderLib.set_uniform("Colour", 0.0, 1.0, 0.0, 1.0)
        for p in self.curve.control_points:
            gl.glVertex3f(p.x, p.y, p.z)
        gl.glEnd()
        gl.glPointSize(1)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

Note: `glBegin`/`glEnd`/`glVertex3f` immediate-mode calls are deprecated under a Core Profile context (the `__main__` block requests `QSurfaceFormat.CoreProfile`). **Before relying on immediate mode, verify it actually works** by running the smoke test (Step 4). If it fails (core profile rejects `glBegin`), replace with a small ad-hoc VAO: build a `numpy.float32` array of the polyline/points and use `ncca.ngl.SimpleVAO`/`VAOFactory.create_vao("simple", mode=gl.GL_LINE_STRIP)` the same way as `ColourObj`'s `create_colour_vao` (Task 2 Step 4), with a single `vec3` position attribute at location 0, paired with `nglColourShader`'s `Colour` uniform (no per-vertex colour needed — draw 3 separate VAOs: curve polyline, hull polyline, control points).

Add `from ncca.ngl import BezierCurve, DefaultShader` to imports, `self.setTitle("CurveDemos")`, and the `--smoketest` block from Task 1.

- [ ] **Step 3: Make executable**

```bash
chmod +x CurveDemos/main.py
```

- [ ] **Step 4: Smoke-test and fix immediate-mode fallback if needed**

```bash
cd CurveDemos && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0. If it crashes with a GL error mentioning deprecated/invalid operation for `glBegin`, apply the VAO-based fallback described in Step 2's note and re-run this command until it passes.

- [ ] **Step 5: Write README.md**

```markdown
# CurveDemos

Builds a 4-point Bezier/B-spline curve (`ncca.ngl.BezierCurve`) and draws the
sampled curve (white), its control polygon "hull" (red), and control points
(green dots).

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 6: Commit**

```bash
git add CurveDemos/
git commit -m "feat: add CurveDemos demo"
```

---

## Task 4: QuatSlerp

**Files:**
- Create: `QuatSlerp/main.py`
- Create: `QuatSlerp/README.md`

**Interfaces:**
- Consumes: `ncca.ngl.Quaternion.from_mat4(Mat4) -> Quaternion`, `q1.slerp(q2, t) -> Quaternion`, `q.to_mat4() -> Mat4`.

- [ ] **Step 1: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `QuatSlerp/main.py`. This demo drops the original's Qt-Designer spin-box UI in favour of keyboard controls, matching this repo's plain-`QOpenGLWindow` convention (per design spec):

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 8), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()

        self.start_rotation = Vec3(45, 90, 80)
        self.end_rotation = Vec3(-300, 270, 360)
        start_mat = Mat4().rotate_z(self.start_rotation.z) @ Mat4().rotate_y(
            self.start_rotation.y
        ) @ Mat4().rotate_x(self.start_rotation.x)
        end_mat = Mat4().rotate_z(self.end_rotation.z) @ Mat4().rotate_y(
            self.end_rotation.y
        ) @ Mat4().rotate_x(self.end_rotation.x)
        self.start_quat = Quaternion.from_mat4(start_mat)
        self.end_quat = Quaternion.from_mat4(end_mat)
        self.interp: float = 0.0

    def load_matrices_to_shader(self, transform) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        mv = self.view @ self.mouse_global_tx @ transform.matrix()
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        interp_quat = self.start_quat.slerp(self.end_quat, self.interp)

        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)

        tx = Transform()
        tx.set_position(0, 0, 0)
        tx.set_rotation_matrix(interp_quat.to_mat4())
        self.load_matrices_to_shader(tx)
        Primitives.draw("teapot")

        tx2 = Transform()
        tx2.set_position(-2, 0, 0)
        tx2.set_rotation_matrix(self.start_quat.to_mat4())
        self.load_matrices_to_shader(tx2)
        Primitives.draw("teapot")

        tx3 = Transform()
        tx3.set_position(2, 0, 0)
        tx3.set_rotation_matrix(self.end_quat.to_mat4())
        self.load_matrices_to_shader(tx3)
        Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

Add `from ncca.ngl import DefaultShader, Primitives, Prims, Quaternion, ShaderLib, Transform` to imports, `self.setTitle("QuatSlerp")`.

In `keyPressEvent`, add before the `self.update()`/`super()` call:
```python
        elif key == Qt.Key_Up:
            self.interp = min(1.0, self.interp + 0.05)
        elif key == Qt.Key_Down:
            self.interp = max(0.0, self.interp - 0.05)
```

- [ ] **Step 2: Confirm `Transform.set_rotation_matrix` exists**

Run: `grep -n "def set_rotation\|def set_position\|def matrix" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/transform.py`
Expected: confirms a method that sets rotation from a `Mat4` (e.g. `set_rotation_matrix` or `set_matrix`). If it's named differently, use the actual name in Step 1's code — if no such method exists, apply the interpolated rotation by multiplying matrices directly instead of going through `Transform`: build `mv = self.view @ self.mouse_global_tx @ Mat4().translate(...) @ interp_quat.to_mat4()` manually and skip `Transform` for the 3 teapots.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x QuatSlerp/main.py
cd QuatSlerp && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 4: Write README.md**

```markdown
# QuatSlerp

Slerps between two quaternion orientations (built from Euler-angle rotation
matrices) and shows the interpolated teapot (centre) alongside the start
(left) and end (right) orientations.

## Controls
`Up`/`Down` : increase/decrease interpolation factor
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 5: Commit**

```bash
git add QuatSlerp/
git commit -m "feat: add QuatSlerp demo"
```

---

## Task 5: KleinBottle

**Files:**
- Create: `KleinBottle/main.py`
- Create: `KleinBottle/README.md`
- Create: `KleinBottle/shaders/PhongVertex.glsl`, `KleinBottle/shaders/PhongFragment.glsl`
- Create: `KleinBottle/klein_bottle.py`

**Interfaces:**
- Produces: `build_klein_bottle(resolution: int = 40) -> numpy.ndarray` in `KleinBottle/klein_bottle.py`, returning interleaved `x,y,z,nx,ny,nz` float32 vertex data (no UVs needed), `GL_TRIANGLES` layout.

- [ ] **Step 1: Write the mesh generator**

Create `KleinBottle/klein_bottle.py`:

```python
"""Procedural Klein bottle mesh, parametric equations from Paul Bourke,
ported from NGL9Demos/KleinBottle."""

import math

import numpy as np
from ncca.ngl import Vec3, calc_normal


def _eval(u: float, v: float) -> Vec3:
    r = 4.0 * (1.0 - math.cos(u) / 2.0)
    if u < math.pi:
        x = 6.0 * math.cos(u) * (1.0 + math.sin(u)) + r * math.cos(u) * math.cos(v)
        y = 16.0 * math.sin(u) + r * math.sin(u) * math.cos(v)
    else:
        x = 6.0 * math.cos(u) * (1.0 + math.sin(u)) + r * math.cos(v + math.pi)
        y = 16.0 * math.sin(u)
    z = r * math.sin(v)
    return Vec3(x, y, z)


def build_klein_bottle(resolution: int = 40) -> np.ndarray:
    du = (2.0 * math.pi) / resolution
    dv = (2.0 * math.pi) / resolution
    eps = 0.01
    verts: list[float] = []

    for i in range(resolution):
        u = i * du
        for j in range(resolution):
            v = j * dv

            def quad_vertex(uu: float, vv: float) -> tuple[Vec3, Vec3]:
                p = _eval(uu, vv)
                p_u = _eval(uu + eps, vv)
                p_v = _eval(uu, vv + eps)
                n = calc_normal(p, p_u, p_v)
                n = Vec3(-n.x, -n.y, -n.z)
                return p, n

            corners = [
                (u, v),
                (u + du, v),
                (u + du, v + dv),
                (u, v),
                (u + du, v + dv),
                (u, v + dv),
            ]
            for uu, vv in corners:
                p, n = quad_vertex(uu, vv)
                verts.extend([p.x * 0.05, p.y * 0.05, p.z * 0.05, n.x, n.y, n.z])

    return np.array(verts, dtype=np.float32)
```

Note: the C++ scales the bottle to roughly world-scale 20-30 units (camera sits at `(0,12,50)`); the `* 0.05` factor above rescales it to fit alongside this repo's other demos (typically viewed from a few units away). Verify visually and adjust the scale factor or the camera `look_at` distance in Step 3 together, whichever reads better once running.

Confirm `calc_normal` is the exact exported name:
```bash
grep -n "def calc_normal\|calc_normal" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/util.py /Volumes/teaching/Code/PyNGL/src/ncca/ngl/__init__.py
```

- [ ] **Step 2: Write the Phong shaders**

Create `KleinBottle/shaders/PhongVertex.glsl`:

```glsl
#version 330 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;

out vec3 fragPos;
out vec3 fragNormal;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat3 normalMatrix;

void main()
{
    fragPos = vec3(MV * vec4(inVert, 1.0));
    fragNormal = normalMatrix * inNormal;
    gl_Position = MVP * vec4(inVert, 1.0);
}
```

Create `KleinBottle/shaders/PhongFragment.glsl`:

```glsl
#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
out vec4 fragColour;

uniform vec3 lightPos;
uniform vec3 viewerPos;

const vec3 ambient = vec3(0.274725, 0.1995, 0.0745);
const vec3 diffuseColour = vec3(0.75164, 0.60648, 0.22648);
const vec3 specularColour = vec3(0.628281, 0.555802, 0.3666065);
const float shininess = 51.2;

void main()
{
    vec3 n = gl_FrontFacing ? normalize(fragNormal) : normalize(-fragNormal);
    vec3 l = normalize(lightPos - fragPos);
    vec3 v = normalize(viewerPos - fragPos);
    vec3 r = reflect(-l, n);

    vec3 diffuse = diffuseColour * max(dot(n, l), 0.0);
    vec3 specular = specularColour * pow(max(dot(r, v), 0.0), shininess);

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
```

- [ ] **Step 3: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `KleinBottle/main.py`:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        )

        data = build_klein_bottle(40)
        self.vertex_count = len(data) // 6
        self.vao = VAOFactory.create_vao("simple", mode=gl.GL_TRIANGLES)
        self.vao.bind()
        self.vao.set_data(data)
        self.vao.set_vertex_attribute_pointer(0, 3, "float", 6 * 4, 0)
        self.vao.set_vertex_attribute_pointer(1, 3, "float", 6 * 4, 3 * 4)
        self.vao.set_num_indices(self.vertex_count)
        self.vao.unbind()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Phong")
        mv = self.view @ self.mouse_global_tx
        mvp = self.project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("lightPos", 2.0, 2.0, 2.0)
        ShaderLib.set_uniform("viewerPos", 0.0, 1.0, 4.0)

        self.vao.bind()
        self.vao.draw()
        self.vao.unbind()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

Add `from ncca.ngl import Mat3, ShaderLib, VAOFactory` and `from klein_bottle import build_klein_bottle` to imports, `self.setTitle("KleinBottle")`. Add a `w`/`s` polygon-mode toggle in `keyPressEvent` matching `VAOPrimitives`.

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x KleinBottle/main.py
cd KleinBottle && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0. If the `VAOFactory`/`SimpleVAO` method names in Step 3 don't match the real API, fix per the same grep approach as Task 2 Step 3 (`grep -n "def " /Volumes/teaching/Code/PyNGL/src/ncca/ngl/simple_vao.py /Volumes/teaching/Code/PyNGL/src/ncca/ngl/vao_factory.py`).

- [ ] **Step 5: Write README.md**

```markdown
# KleinBottle

Procedurally generates a Klein bottle mesh from Paul Bourke's parametric
equations (40x40 resolution) and shades it with a gold Phong material,
starting in wireframe mode.

## Controls
`w` : wireframe, `s` : solid fill
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 6: Commit**

```bash
git add KleinBottle/
git commit -m "feat: add KleinBottle demo"
```

---

## Task 6: FrustumCull

**Files:**
- Create: `FrustumCull/main.py`
- Create: `FrustumCull/README.md`
- Create: `FrustumCull/shaders/PhongVertex.glsl`, `FrustumCull/shaders/PhongFragment.glsl` (reuse KleinBottle's, copy)
- Create: `FrustumCull/uvn_camera.py` (extends Task 1's `UVNCamera` with frustum extraction)

**Interfaces:**
- Consumes: `ncca.ngl.Plane(p1, p2, p3)`, `.distance(p: Vec3) -> float`.
- Produces: `FrustumCamera` class with `.calculate_frustum() -> None`, `.is_sphere_in_frustum(center: Vec3, radius: float) -> str` (returns `"OUTSIDE"`, `"INTERSECT"`, or `"INSIDE"`).

- [ ] **Step 1: Copy the Phong shaders and Camera 1's UVN camera**

```bash
mkdir -p FrustumCull/shaders
cp KleinBottle/shaders/PhongVertex.glsl FrustumCull/shaders/
cp KleinBottle/shaders/PhongFragment.glsl FrustumCull/shaders/
cp Camera/uvn_camera.py FrustumCull/uvn_camera.py
```

- [ ] **Step 2: Confirm the Plane API**

Run: `grep -n "class Plane\|def __init__\|def distance\|def set_points" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/plane.py`
Expected: confirms `Plane(p1: Vec3, p2: Vec3, p3: Vec3)` constructor and `.distance(point: Vec3) -> float` signature (positive = in front of plane normal).

- [ ] **Step 3: Extend the camera with frustum extraction**

Append to `FrustumCull/uvn_camera.py`:

```python
import math

from ncca.ngl import Plane


class FrustumCamera(UVNCamera):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.planes: dict[str, Plane] = {}
        self.calculate_frustum()

    def calculate_frustum(self) -> None:
        near_height = 2.0 * math.tan(math.radians(self.fov) / 2.0) * self.near
        near_width = near_height * self.aspect
        far_height = 2.0 * math.tan(math.radians(self.fov) / 2.0) * self.far
        far_width = far_height * self.aspect

        near_center = self.eye + self.n * self.near
        far_center = self.eye + self.n * self.far

        ntl = near_center + self.v * (near_height / 2) - self.u * (near_width / 2)
        ntr = near_center + self.v * (near_height / 2) + self.u * (near_width / 2)
        nbl = near_center - self.v * (near_height / 2) - self.u * (near_width / 2)
        nbr = near_center - self.v * (near_height / 2) + self.u * (near_width / 2)

        ftl = far_center + self.v * (far_height / 2) - self.u * (far_width / 2)
        ftr = far_center + self.v * (far_height / 2) + self.u * (far_width / 2)
        fbl = far_center - self.v * (far_height / 2) - self.u * (far_width / 2)
        fbr = far_center - self.v * (far_height / 2) + self.u * (far_width / 2)

        self.planes["NEAR"] = Plane(ntl, ntr, nbr)
        self.planes["FAR"] = Plane(ftr, ftl, fbl)
        self.planes["LEFT"] = Plane(ntl, nbl, fbl)
        self.planes["RIGHT"] = Plane(ntr, ftr, fbr)
        self.planes["TOP"] = Plane(ntl, ftl, ftr)
        self.planes["BOTTOM"] = Plane(nbl, nbr, fbr)
        self.corners = [ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr]

    def is_sphere_in_frustum(self, center, radius: float) -> str:
        result = "INSIDE"
        for plane in self.planes.values():
            d = plane.distance(center)
            if d < -radius:
                return "OUTSIDE"
            if d < radius:
                result = "INTERSECT"
        return result

    def set_shape(self, fov: float, aspect: float, near: float, far: float) -> None:
        super().set_shape(fov, aspect, near, far)
        self.calculate_frustum()

    def move_eye(self, dx, dy, dz) -> None:
        super().move_eye(dx, dy, dz)
        self.calculate_frustum()
```

Note: `Plane.distance` sign convention must be confirmed with a quick manual check (Step 2) — if positive means "behind" rather than "in front", flip the `< -radius`/`< radius` comparisons in `is_sphere_in_frustum` accordingly, and verify by observing that spheres visibly outside the view cone stop being drawn once running (Step 5).

- [ ] **Step 4: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `FrustumCull/main.py`. Use a **reduced grid** (step 4, extent ±20, i.e. 11 positions per axis = 1331 candidate spheres) rather than the C++'s huge ±150 step-2 grid, since this is an interactive Python demo, not a benchmark:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        aspect = self.window_width / self.window_height
        self.test_camera = FrustumCamera(
            Vec3(0, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0), 45.0, aspect, 2.0, 15.0
        )
        self.observer_camera = FrustumCamera(
            Vec3(0, 40, 0.001), Vec3(0, 0, 0), Vec3(0, 1, 0), 60.0, aspect, 0.5, 100.0
        )
        self.active_camera_index = 1

        ShaderLib.load_shader(
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        )
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 12)

        self.grid_positions = [
            Vec3(x, y, z)
            for x in range(-20, 21, 4)
            for y in range(-8, 9, 4)
            for z in range(-20, 21, 4)
        ]

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        camera = (
            self.test_camera if self.active_camera_index == 0 else self.observer_camera
        )

        ShaderLib.use("Phong")
        drawn = 0
        for pos in self.grid_positions:
            state = self.test_camera.is_sphere_in_frustum(pos, 1.0)
            if state == "OUTSIDE":
                continue
            drawn += 1
            mv = camera.view @ self.mouse_global_tx @ Mat4().translate(pos.x, pos.y, pos.z)
            mvp = camera.project @ mv
            normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            ShaderLib.set_uniform("lightPos", 10.0, 10.0, 10.0)
            ShaderLib.set_uniform("viewerPos", camera.eye.x, camera.eye.y, camera.eye.z)
            Primitives.draw("sphere")

        self.last_drawn = drawn
        self.setTitle(f"FrustumCull - drawn {drawn}/{len(self.grid_positions)}")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        aspect = float(w) / h
        self.test_camera.set_shape(self.test_camera.fov, aspect, self.test_camera.near, self.test_camera.far)
        self.observer_camera.set_shape(self.observer_camera.fov, aspect, self.observer_camera.near, self.observer_camera.far)
```

Add `from uvn_camera import FrustumCamera`, `from ncca.ngl import Mat3, Primitives, Prims, ShaderLib`, `self.setTitle("FrustumCull")`. Add key `1`/`2` to switch `self.active_camera_index`, matching Task 1.

- [ ] **Step 5: Make executable and smoke-test**

```bash
chmod +x FrustumCull/main.py
cd FrustumCull && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 6: Write README.md**

```markdown
# FrustumCull

Culls a 3D grid of spheres against a test camera's view frustum (6-plane
extraction + sphere/plane distance test) and only draws the spheres that are
inside or intersecting. View from the observer camera (top-down) to see the
effect; the title bar shows drawn/total sphere counts.

## Controls
`1` : view from test camera, `2` : view from observer camera
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 7: Commit**

```bash
git add FrustumCull/
git commit -m "feat: add FrustumCull demo"
```

---

## Task 7: PointCloud

**Files:**
- Create: `PointCloud/main.py`
- Create: `PointCloud/README.md`
- Create: `PointCloud/data/test.xyz` (copied from NGL9Demos)
- Create: `PointCloud/point_cloud.py`

**Interfaces:**
- Produces: `PointCloud` class in `PointCloud/point_cloud.py` with `PointCloud.from_file(path) -> PointCloud`, `.points: list[Vec3]`, `.bbox_center: Vec3`, `.bbox_max_dim: float`, `.sphere_center: Vec3`, `.sphere_radius: float`.

- [ ] **Step 1: Copy the data file**

```bash
mkdir -p PointCloud/data
cp /Volumes/teaching/NGL9Demos/PointCloud/data/test.xyz PointCloud/data/
```

- [ ] **Step 2: Write the point-cloud loader**

Create `PointCloud/point_cloud.py`:

```python
"""Loads a plain XYZ point cloud and computes bounding box / Ritter bounding sphere,
ported from NGL9Demos/PointCloud."""

from __future__ import annotations

from ncca.ngl import Vec3


class PointCloud:
    def __init__(self) -> None:
        self.points: list[Vec3] = []
        self.bbox_center = Vec3()
        self.bbox_max_dim = 1.0
        self.sphere_center = Vec3()
        self.sphere_radius = 1.0

    @classmethod
    def from_file(cls, path: str) -> "PointCloud":
        cloud = cls()
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                cloud.points.append(Vec3(float(parts[0]), float(parts[1]), float(parts[2])))
        cloud._calculate_bounding_box()
        cloud._calculate_bounding_sphere()
        cloud._unitize()
        return cloud

    def _calculate_bounding_box(self) -> None:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]
        min_p = Vec3(min(xs), min(ys), min(zs))
        max_p = Vec3(max(xs), max(ys), max(zs))
        self.bbox_center = Vec3(
            (min_p.x + max_p.x) / 2, (min_p.y + max_p.y) / 2, (min_p.z + max_p.z) / 2
        )
        self.bbox_max_dim = max(max_p.x - min_p.x, max_p.y - min_p.y, max_p.z - min_p.z)
        self._min_p = min_p
        self._max_p = max_p

    def _calculate_bounding_sphere(self) -> None:
        # Ritter's approximate bounding sphere
        p0 = self.points[0]
        farthest_from_p0 = max(self.points, key=lambda p: (p - p0).length_squared())
        farthest_from_that = max(
            self.points, key=lambda p: (p - farthest_from_p0).length_squared()
        )
        center = Vec3(
            (farthest_from_p0.x + farthest_from_that.x) / 2,
            (farthest_from_p0.y + farthest_from_that.y) / 2,
            (farthest_from_p0.z + farthest_from_that.z) / 2,
        )
        radius = (farthest_from_that - farthest_from_p0).length() / 2

        for p in self.points:
            d = (p - center).length()
            if d > radius:
                new_radius = (radius + d) / 2
                k = (new_radius - radius) / d
                center = center + (p - center) * k
                radius = new_radius

        self.sphere_center = center
        self.sphere_radius = radius

    def _unitize(self) -> None:
        scale = 1.0 / self.bbox_max_dim if self.bbox_max_dim > 0 else 1.0
        self.points = [(p - self.bbox_center) * scale for p in self.points]
        self.sphere_center = (self.sphere_center - self.bbox_center) * scale
        self.sphere_radius *= scale
        self.bbox_center = Vec3()
```

Note: confirm `Vec3.length_squared()`/`.length()` are the actual method names (`grep -n "def length" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/vec3.py`) — if named e.g. `length_sq`, adjust.

- [ ] **Step 3: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `PointCloud/main.py`. Since the source shader's per-point "normal" attribute has no real data (flagged in research — the C++ demo never uploads a normal buffer either), render with a flat uniform colour instead of a fabricated per-point colour:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.05, 0.05, 0.05, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self.cloud = PointCloud.from_file("data/test.xyz")
        cam_z = 1.25 * 1.0
        self.view = look_at(
            Vec3(self.cloud.sphere_center.x, self.cloud.sphere_center.y, cam_z),
            self.cloud.bbox_center,
            Vec3(0, 1, 1),
        )
        self.point_size = 5

        data = []
        for p in self.cloud.points:
            data.extend([p.x, p.y, p.z])
        import numpy as np

        arr = np.array(data, dtype=np.float32)
        self.vao = VAOFactory.create_vao("simple", mode=gl.GL_POINTS)
        self.vao.bind()
        self.vao.set_data(arr)
        self.vao.set_vertex_attribute_pointer(0, 3, "float", 3 * 4, 0)
        self.vao.set_num_indices(len(self.cloud.points))
        self.vao.unbind()

        ShaderLib.use(DefaultShader.COLOUR)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPointSize(self.point_size)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use(DefaultShader.COLOUR)
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("Colour", 0.2, 0.8, 1.0, 1.0)

        self.vao.bind()
        self.vao.draw()
        self.vao.unbind()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.001, 20.0)
```

Add `from ncca.ngl import DefaultShader, ShaderLib, VAOFactory`, `from point_cloud import PointCloud`, `self.setTitle("PointCloud")`. Add `+`/`-` (`Qt.Key_Equal`/`Qt.Key_Minus`) key handling to grow/shrink `self.point_size` (min 1).

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x PointCloud/main.py
cd PointCloud && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 5: Write README.md**

```markdown
# PointCloud

Loads a plain XYZ point cloud (1000 random points), computes its bounding
box and a Ritter approximate bounding sphere, unitizes it to fit the view,
and renders it as `GL_POINTS`.

## Controls
`+`/`-` : grow/shrink point size
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 6: Commit**

```bash
git add PointCloud/
git commit -m "feat: add PointCloud demo"
```

---

## Task 8: AnimatedTextures

**Files:**
- Create: `AnimatedTextures/main.py`
- Create: `AnimatedTextures/README.md`
- Create: `AnimatedTextures/shaders/BillboardVert.glsl`, `BillboardGeo.glsl`, `BillboardFrag.glsl`
- Create: `AnimatedTextures/textures/map1.png`, `map2.png`, `map3.png` (copied)

**Interfaces:**
- Consumes: `ShaderLib.load_shader(name, vert, frag, geo=...)`, `ncca.ngl.Texture`.

- [ ] **Step 1: Copy the sprite-sheet textures**

```bash
mkdir -p AnimatedTextures/shaders AnimatedTextures/textures
cp /Volumes/teaching/NGL9Demos/AnimatedTextures/textures/map1.png AnimatedTextures/textures/
cp /Volumes/teaching/NGL9Demos/AnimatedTextures/textures/map2.png AnimatedTextures/textures/
cp /Volumes/teaching/NGL9Demos/AnimatedTextures/textures/map3.png AnimatedTextures/textures/
```

- [ ] **Step 2: Write the shaders**

Create `AnimatedTextures/shaders/BillboardVert.glsl`:

```glsl
#version 330 core
layout(location = 0) in vec4 inVert;
layout(location = 1) in float inOffset;

flat out float whichTexture;
flat out float frameOffset;

void main()
{
    gl_Position = inVert;
    whichTexture = inVert.w;
    frameOffset = inOffset;
}
```

Create `AnimatedTextures/shaders/BillboardGeo.glsl`:

```glsl
#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float whichTexture[];
in float frameOffset[];

flat out float texID;
out vec2 texCoord;

uniform mat4 MVP;
uniform mat4 MV;
uniform vec3 cameraPos;
uniform float time;

const float bbWidth = 0.5;
const float bbHeight = 1.0;
const float spriteOffset = 0.1;

void main()
{
    vec3 pos = gl_in[0].gl_Position.xyz;
    vec3 toCamera = normalize(cameraPos - pos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(toCamera, up);

    float ctime = floor(time + frameOffset[0]);
    float u0 = ctime * spriteOffset;
    float u1 = (ctime + 1.0) * spriteOffset;

    texID = whichTexture[0];

    vec3 p0 = pos - right * bbWidth;
    gl_Position = MVP * vec4(p0, 1.0);
    texCoord = vec2(u0, 0.0);
    EmitVertex();

    vec3 p1 = pos - right * bbWidth + up * bbHeight;
    gl_Position = MVP * vec4(p1, 1.0);
    texCoord = vec2(u0, 1.0);
    EmitVertex();

    vec3 p2 = pos + right * bbWidth;
    gl_Position = MVP * vec4(p2, 1.0);
    texCoord = vec2(u1, 0.0);
    EmitVertex();

    vec3 p3 = pos + right * bbWidth + up * bbHeight;
    gl_Position = MVP * vec4(p3, 1.0);
    texCoord = vec2(u1, 1.0);
    EmitVertex();

    EndPrimitive();
}
```

Create `AnimatedTextures/shaders/BillboardFrag.glsl`:

```glsl
#version 330 core
flat in float texID;
in vec2 texCoord;
out vec4 fragColour;

uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;

void main()
{
    vec4 colour;
    if (texID < 0.5)
        colour = texture(tex1, texCoord);
    else if (texID < 1.5)
        colour = texture(tex2, texCoord);
    else
        colour = texture(tex3, texCoord);

    if (colour.rgb == vec3(0.0))
        discard;

    fragColour = colour;
}
```

Note: `MV` and `cameraPos` are both declared as uniforms in `BillboardGeo.glsl` above; the C++ used only camera-space billboarding via `cameraPos`. Keep `cameraPos` as the one actually used in `main()`, and drop the unused `MV` uniform declaration if `ShaderLib.set_uniform` warns about an unused/unfound location (harmless either way — a warning, not an error).

- [ ] **Step 3: Confirm the geometry-shader loading and multi-texture-unit API**

Run:
```bash
grep -n "def load_shader\b" -A 20 /Volumes/teaching/Code/PyNGL/src/ncca/ngl/shader_lib.py
grep -n "def set_multi_texture\|def load_image\|def set_texture_gl" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/texture.py
```
Expected: confirms `load_shader(name, vert, frag, geo=None)` accepts a geometry-shader path via `geo=`, and `Texture.set_multi_texture(unit: int)` binds the texture to `GL_TEXTUREunit`. Use these exact signatures in Step 4.

- [ ] **Step 4: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `AnimatedTextures/main.py`:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.view = look_at(Vec3(0, 4, 20), Vec3(0, 2, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Billboard",
            "shaders/BillboardVert.glsl",
            "shaders/BillboardFrag.glsl",
            geo="shaders/BillboardGeo.glsl",
        )

        self.textures = []
        for i, fname in enumerate(["map1.png", "map2.png", "map3.png"]):
            tex = Texture(f"textures/{fname}")
            tex.set_texture_gl()
            self.textures.append(tex)

        import math
        import random

        data = []
        for i in range(500):
            radius = random.uniform(8.0, 9.0)
            angle = math.radians(i)
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = random.uniform(0.0, 8.0)
            tex_index = float(random.randint(0, 2))
            offset = random.uniform(0.0, 10.0)
            data.extend([x, y, z, tex_index, offset])

        import numpy as np

        arr = np.array(data, dtype=np.float32)
        self.vao = VAOFactory.create_vao("simple", mode=gl.GL_POINTS)
        self.vao.bind()
        self.vao.set_data(arr)
        self.vao.set_vertex_attribute_pointer(0, 4, "float", 5 * 4, 0)
        self.vao.set_vertex_attribute_pointer(1, 1, "float", 5 * 4, 4 * 4)
        self.vao.set_num_indices(500)
        self.vao.unbind()

        self.time: float = 0.0
        self.animate: bool = True

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Billboard")
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("cameraPos", 0.0, 4.0, 20.0)
        ShaderLib.set_uniform("time", self.time)
        for i, tex in enumerate(self.textures):
            tex.set_multi_texture(i)
        ShaderLib.set_uniform("tex1", 0)
        ShaderLib.set_uniform("tex2", 1)
        ShaderLib.set_uniform("tex3", 2)

        self.vao.bind()
        self.vao.draw()
        self.vao.unbind()

        if self.animate:
            self.time += 0.1
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.1, 100.0)
```

Add `from ncca.ngl import ShaderLib, Texture, VAOFactory`, `self.setTitle("AnimatedTextures")`. Add `Qt.Key_Space` toggling `self.animate` in `keyPressEvent`.

- [ ] **Step 5: Make executable and smoke-test**

```bash
chmod +x AnimatedTextures/main.py
cd AnimatedTextures && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0. If `ShaderLib.load_shader` errors on the `geo=` kwarg or geometry-shader compilation fails, re-check Step 3's grep output for the correct kwarg name/order and adjust the call.

- [ ] **Step 6: Write README.md**

```markdown
# AnimatedTextures

500 GPU billboards (camera-facing quads built in a geometry shader from a
single point each) sample one of 3 fire sprite-sheet textures, scrolling
through 10 animation frames over time. Pure-black texels are discarded for
chroma-key transparency.

## Controls
`space` : toggle animation
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

(Note the `space` key is dual-purpose here — toggling animation takes priority over camera reset in `keyPressEvent`; document only the animation toggle since camera reset isn't essential for this demo. If both behaviours are wired to `space`, pick one in the implementation — prefer the animation toggle and drop the camera-reset binding for this demo only.)

- [ ] **Step 7: Commit**

```bash
git add AnimatedTextures/
git commit -m "feat: add AnimatedTextures demo"
```

---

## Task 9: Interpolation

**Files:**
- Create: `Interpolation/main.py`
- Create: `Interpolation/README.md`
- Create: `Interpolation/shaders/PhongVertex.glsl`, `Interpolation/shaders/PhongFragment.glsl`
- Create: `Interpolation/easing.py`

**Interfaces:**
- Produces: `trig_interp(a: Vec3, b: Vec3, t: float) -> Vec3`, `cubic_interp(a: Vec3, b: Vec3, t: float) -> Vec3` in `Interpolation/easing.py`.

- [ ] **Step 1: Write the easing functions**

Create `Interpolation/easing.py`:

```python
"""Trigonometric and cubic easing/interpolation, reimplemented since ncca.ngl
only provides linear `lerp`. Standard smoothstep-family formulas."""

import math

from ncca.ngl import Vec3


def trig_interp(a: Vec3, b: Vec3, t: float) -> Vec3:
    eased = 0.5 * (1.0 - math.cos(t * math.pi))
    return a + (b - a) * eased


def cubic_interp(a: Vec3, b: Vec3, t: float) -> Vec3:
    eased = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * eased
```

- [ ] **Step 2: Reuse the Phong shaders from KleinBottle, parameterizing the material via uniforms**

```bash
mkdir -p Interpolation/shaders
```

Create `Interpolation/shaders/PhongVertex.glsl` (identical to `KleinBottle/shaders/PhongVertex.glsl` — copy it):

```bash
cp KleinBottle/shaders/PhongVertex.glsl Interpolation/shaders/PhongVertex.glsl
```

Create `Interpolation/shaders/PhongFragment.glsl` (same lighting model as KleinBottle's but with the material passed in as uniforms instead of hard-coded, so 3 different materials can share one shader):

```glsl
#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
out vec4 fragColour;

uniform vec3 lightPos;
uniform vec3 viewerPos;
uniform vec3 ambient;
uniform vec3 diffuseColour;
uniform vec3 specularColour;
uniform float shininess;

void main()
{
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(lightPos - fragPos);
    vec3 v = normalize(viewerPos - fragPos);
    vec3 r = reflect(-l, n);

    vec3 diffuse = diffuseColour * max(dot(n, l), 0.0);
    vec3 specular = specularColour * pow(max(dot(r, v), 0.0), shininess);

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
```

- [ ] **Step 3: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `Interpolation/main.py`:

```python
    MATERIALS = {
        "gold": (
            Vec3(0.274725, 0.1995, 0.0745),
            Vec3(0.75164, 0.60648, 0.22648),
            Vec3(0.628281, 0.555802, 0.3666065),
            51.2,
        ),
        "brass": (
            Vec3(0.329412, 0.223529, 0.027451),
            Vec3(0.780392, 0.568627, 0.113725),
            Vec3(0.992157, 0.941176, 0.807843),
            27.8974,
        ),
        "pewter": (
            Vec3(0.10588, 0.058824, 0.113725),
            Vec3(0.427451, 0.470588, 0.541176),
            Vec3(0.3333, 0.3333, 0.521569),
            9.84615,
        ),
    }

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 25), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        )
        Primitives.load_default_primitives()

        self.start = Vec3(-8, -5, 0)
        self.end = Vec3(8, 5, 0)
        self.time: float = 0.0
        self.animate: bool = True

    def _set_material(self, name: str) -> None:
        ambient, diffuse, specular, shininess = self.MATERIALS[name]
        ShaderLib.set_uniform("ambient", ambient.x, ambient.y, ambient.z)
        ShaderLib.set_uniform("diffuseColour", diffuse.x, diffuse.y, diffuse.z)
        ShaderLib.set_uniform("specularColour", specular.x, specular.y, specular.z)
        ShaderLib.set_uniform("shininess", shininess)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Phong")
        ShaderLib.set_uniform("lightPos", 5.0, 5.0, 10.0)
        ShaderLib.set_uniform("viewerPos", 0.0, 0.0, 25.0)

        def draw_teapot(pos: Vec3, material: str) -> None:
            mv = self.view @ self.mouse_global_tx @ Mat4().translate(pos.x, pos.y, pos.z)
            mvp = self.project @ mv
            normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("MV", mv)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            self._set_material(material)
            Primitives.draw("teapot")

        linear_pos = self.start + (self.end - self.start) * self.time
        trig_pos = trig_interp(self.start, self.end, self.time) + Vec3(0, 2, 0)
        cubic_pos = cubic_interp(self.start, self.end, self.time) + Vec3(0, -2, 0)

        draw_teapot(linear_pos, "gold")
        draw_teapot(trig_pos, "brass")
        draw_teapot(cubic_pos, "pewter")

        if self.animate:
            self.time += 0.005
            if self.time >= 1.0:
                self.time = 0.0
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
```

Add `from ncca.ngl import Mat3, Primitives, ShaderLib`, `from easing import trig_interp, cubic_interp`, `self.setTitle("Interpolation")`. In `keyPressEvent` add `Qt.Key_Space` to toggle `self.animate`, and `Qt.Key_Left`/`Qt.Key_Right` to step `self.time` by ∓0.01 clamped to `[0, 1]`.

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x Interpolation/main.py
cd Interpolation && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 5: Write README.md**

```markdown
# Interpolation

Animates 3 teapots (gold/brass/pewter) from the same start to end point over
time, using linear, trigonometric-eased, and cubic-eased interpolation
respectively, to visually compare the resulting spacing/timing.

## Controls
`space` : toggle animation
`Left`/`Right` : step time manually
Left-drag : orbit, Right-drag : pan, Wheel : zoom
```

- [ ] **Step 6: Commit**

```bash
git add Interpolation/
git commit -m "feat: add Interpolation demo"
```

---

## Task 10: ImageHeightMap

**Files:**
- Create: `ImageHeightMap/main.py`
- Create: `ImageHeightMap/README.md`
- Create: `ImageHeightMap/shaders/ColourVert.glsl`, `ImageHeightMap/shaders/ColourFrag.glsl`
- Create: `ImageHeightMap/textures/FractalMap.bmp` (copied; the smaller 16KB map, not the 3.1MB `MountainBig.bmp`)

**Interfaces:**
- Produces: `build_heightmap_mesh(image_path: str, width: float = 40.0, depth: float = 40.0, max_height: float = 4.0, max_resolution: int = 200) -> tuple[np.ndarray, np.ndarray]` in `ImageHeightMap/heightmap.py`, returning `(vertex_data, indices)` where `vertex_data` is interleaved `x,y,z,r,g,b` float32 and `indices` is `uint32` triangle-list indices (one draw call, no primitive-restart dependency).

- [ ] **Step 1: Copy the smaller heightmap texture**

```bash
mkdir -p ImageHeightMap/shaders ImageHeightMap/textures
cp /Volumes/teaching/NGL9Demos/ImageHeightMap/textures/FractalMap.bmp ImageHeightMap/textures/
```

- [ ] **Step 2: Write the shaders**

Create `ImageHeightMap/shaders/ColourVert.glsl`:

```glsl
#version 330 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inColour;

out vec3 vertColour;

uniform mat4 MVP;

void main()
{
    vertColour = inColour;
    gl_Position = MVP * vec4(inVert, 1.0);
}
```

Create `ImageHeightMap/shaders/ColourFrag.glsl`:

```glsl
#version 330 core
in vec3 vertColour;
out vec4 fragColour;

void main()
{
    fragColour = vec4(vertColour, 1.0);
}
```

- [ ] **Step 3: Write the heightmap mesh builder**

Create `ImageHeightMap/heightmap.py`:

```python
"""Builds a coloured terrain mesh from an image, sampling the red channel for
height and using RGB directly as vertex colour. Ported from
NGL9Demos/ImageHeightMap, but uses a plain triangle-list index buffer (one
glDrawElements call) instead of GL_PRIMITIVE_RESTART, and downsamples large
source images to keep vertex counts interactive-friendly.

Reports back to the caller: 200x200 max grid resolution regardless of source
image size, since the C++ original samples 1 vertex per source pixel (up to
1M+ vertices for MountainBig.bmp) which isn't necessary for a demo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def build_heightmap_mesh(
    image_path: str,
    width: float = 40.0,
    depth: float = 40.0,
    max_height: float = 4.0,
    max_resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    if img.width > max_resolution or img.height > max_resolution:
        img = img.resize((max_resolution, max_resolution))
    pixels = np.asarray(img, dtype=np.float32) / 255.0  # (h, w, 3)

    h, w = pixels.shape[0], pixels.shape[1]
    verts: list[float] = []
    for z in range(h):
        z_pos = -depth / 2 + depth * (z / (h - 1))
        for x in range(w):
            x_pos = -width / 2 + width * (x / (w - 1))
            r, g, b = pixels[z, x]
            y_pos = r * max_height
            verts.extend([x_pos, y_pos, z_pos, r, g, b])

    indices: list[int] = []
    for z in range(h - 1):
        for x in range(w - 1):
            top_left = z * w + x
            top_right = z * w + x + 1
            bottom_left = (z + 1) * w + x
            bottom_right = (z + 1) * w + x + 1
            indices.extend([top_left, bottom_left, top_right])
            indices.extend([top_right, bottom_left, bottom_right])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)
```

- [ ] **Step 4: Confirm `PIL`/`Pillow` is available and check the indexed-VAO API**

Run:
```bash
uv run --script -c "from PIL import Image; print('ok')"
grep -n "def create_index_vao\|IndexVAO\|def set_indices\|class MultiBufferVAO\|def create_vao" /Volumes/teaching/Code/PyNGL/src/ncca/ngl/vao_factory.py /Volumes/teaching/Code/PyNGL/src/ncca/ngl/simple_index_vao.py
```
Expected: `PIL` import succeeds (add `Pillow` as a script dependency via `uv add --script ImageHeightMap/main.py pillow` if it's missing — check `pyproject.toml`/`uv.lock` first, since other demos already using `Image` (`ncca.ngl.image`) may already depend on it transitively); and the grep shows the exact indexed-VAO class/method names (likely `SimpleIndexVAO` or `VAOFactory.create_vao("simpleIndex", ...)` with a `set_indices`/`set_index_data` method) to use in Step 5.

- [ ] **Step 5: Write main.py**

Copy `VAOPrimitives/main.py` skeleton to `ImageHeightMap/main.py`:

```python
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 10, 54), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Colour", "shaders/ColourVert.glsl", "shaders/ColourFrag.glsl"
        )

        verts, indices = build_heightmap_mesh("textures/FractalMap.bmp")
        self.index_count = len(indices)
        self.vao = VAOFactory.create_vao("simpleIndex", mode=gl.GL_TRIANGLES)
        self.vao.bind()
        self.vao.set_data(verts, indices)
        self.vao.set_vertex_attribute_pointer(0, 3, "float", 6 * 4, 0)
        self.vao.set_vertex_attribute_pointer(1, 3, "float", 6 * 4, 3 * 4)
        self.vao.set_num_indices(self.index_count)
        self.vao.unbind()

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        ShaderLib.use("Colour")
        mvp = self.project @ self.view @ self.mouse_global_tx
        ShaderLib.set_uniform("MVP", mvp)

        self.vao.bind()
        self.vao.draw()
        self.vao.unbind()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.001, 150.0)
```

Add `from ncca.ngl import ShaderLib, VAOFactory`, `from heightmap import build_heightmap_mesh`, `self.setTitle("ImageHeightMap")`. Note: `VAOFactory.create_vao("simpleIndex", ...)` and `.set_data(verts, indices)` are placeholder names pending Step 4's grep confirmation — replace with the real indexed-VAO class name/constructor and however it distinguishes vertex data from index data (may be two separate calls, e.g. `set_data(verts)` then `set_indices(indices)`).

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x ImageHeightMap/main.py
cd ImageHeightMap && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0.

- [ ] **Step 7: Write README.md**

```markdown
# ImageHeightMap

Builds a terrain mesh from an image: each vertex's height comes from the
pixel's red channel, and the vertex is coloured by the pixel's RGB. Uses
`FractalMap.bmp`, downsampled to a max 200x200 grid for interactivity.

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 8: Commit**

```bash
git add ImageHeightMap/
git commit -m "feat: add ImageHeightMap demo"
```

---

## Final check

- [ ] **Run every Phase 1 demo's smoke test in sequence to confirm nothing regressed**

```bash
for d in Camera ColourObj CurveDemos QuatSlerp KleinBottle FrustumCull PointCloud AnimatedTextures Interpolation ImageHeightMap; do
  echo "=== $d ==="
  (cd "$d" && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest)
done
```
Expected: every demo prints `SMOKETEST OK` with no traceback.

- [ ] **Run repo linter/formatter if configured**

```bash
ruff check .
ruff format --check .
```
Expected: no errors (fix any that appear, following existing code style, then re-run and commit fixes).
