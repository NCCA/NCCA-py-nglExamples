# Core Demos Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 4 NGL9Demos (MatrixStack, LookAtDemos, ViewToWorldTransform, AffineTransforms) to PyNGLDemos, each with an OpenGL and a WebGPU entry point except where a technique has no WebGPU equivalent.

**Architecture:** Each demo is a self-contained top-level folder. OpenGL entry points are `QOpenGLWindow` subclasses using `PySideEventHandlingMixin` for the standard orbit/pan/zoom controls; WebGPU entry points subclass `ncca.ngl.webgpu.WebGPUWidget`. Where a demo's core logic is backend-independent pure maths (the matrix stack, the unprojection formula), it lives in its own module with no GL/Qt/wgpu imports so both entry points import the same code.

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, `uv run --script`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md`

## Global Constraints

- Work happens in branch `agent/core-demos-phase1`, worktree at `.worktrees/core-demos-phase1` (create with `git worktree add .worktrees/core-demos-phase1 -b agent/core-demos-phase1` before starting Task 1 — not yet created).
- No edits to `/Users/jmacey/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder.
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) which runs one paint pass via `QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))` then exits 0 — copy the pattern verbatim from `VAOPrimitives/main.py`'s `__main__` block (OpenGL) or `Blending/BlendingWebGPU.py`'s `main()` (WebGPU).
- OpenGL entry points: `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)`, calling `self.setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01, zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` in `__init__`. The mixin already provides Escape/W/S/Space key handling and full mouse orbit/pan/zoom — demo-specific `keyPressEvent` overrides only need to handle their own extra keys and end with `super().keyPressEvent(event)`. GL 4.1 core profile via the standard `QSurfaceFormat` block (copy from `VAOPrimitives/main.py`'s `__main__`).
- WebGPU entry points: `class WebGPUScene(WebGPUWidget)` importing `from ncca.ngl.webgpu import WebGPUWidget` directly — **do not** copy a `WebGPUWidget.py` file into the demo folder; that file no longer exists standalone anywhere in this repo (verified: only a stale `.pyc` remains under `FBODemos/WebGPURenderToTexture/__pycache__`), the class was promoted into the `ncca.ngl.webgpu` package itself. Set `self.msaa_sample_count = 4`, call `get_default_device()`, build pipelines/scene, then `self._create_render_buffer()`. Mouse/keyboard handlers are hand-copied (no mixin for `QWidget`) — copy the block from `Blending/BlendingWebGPU.py`.
- Maths convention: numpy/PyNGL row-vector convention — points transform as `row_vec @ M`, translation lives in row 3 (`mat[3, 0..2]`). Matrix composition order reads the same as the C++ source (`A @ B @ C` applies exactly like `A * B * C` did in NGL9Demos) — do not reorder when porting.
- `ruff check` and `ruff format --check` must pass.
- README.md per demo (description, controls table, teaching points, `![](<Demo>.png)` image reference); add a row to the root `README.md` under an appropriate section (create the section if none fits). Screenshots cannot be captured by an agent — list the missing `.png` in the final report for Jon.
- One commit per task: `git add <files> && git commit -m "feat(<demo>): <what>"`.
- Verify every entry script with `QT_QPA_PLATFORM=offscreen uv run --script <path> --smoketest` from the repo root; expect `SMOKETEST OK` and exit 0, no traceback.

---

## Task 1: MatrixStack (OpenGL)

**Files:**
- Create: `MatrixStack/matrix_stack.py`
- Create: `MatrixStack/main.py`
- Create: `MatrixStack/README.md`

**Interfaces:**
- Produces: `MatrixStack` class in `MatrixStack/matrix_stack.py` — `push_matrix()`, `pop_matrix()`, `identity()`, `top() -> Mat4`, `set_view(view: Mat4)`, `set_projection(projection: Mat4)`, `rotate_xyz(x: float, y: float, z: float)` (three separate-axis rolls composed `Rz @ Ry @ Rx`), `rotate_axis_angle(angle: float, x: float, y: float, z: float)` (single rotation about an arbitrary axis), `translate(x: float, y: float, z: float)`, `scale(x: float, y: float, z: float)`, `mvp() -> Mat4`, `mv() -> Mat4`. Also `MatrixStackError` exception (overflow/underflow). This module has no GL/Qt imports — Task 2 (WebGPU) imports it unchanged.

- [ ] **Step 1: Write the matrix stack module**

Create `MatrixStack/matrix_stack.py`:

```python
"""A push/pop OpenGL-style matrix stack, ported from NGL9Demos/MatrixStack.

Backend-independent: no GL/Qt/wgpu imports, so both the OpenGL and WebGPU
entry points of this demo share this exact module.
"""

from __future__ import annotations

from ncca.ngl import Mat4, Quaternion, Vec3

_STACK_SIZE = 40


class MatrixStackError(Exception):
    """Raised on stack overflow or underflow."""


class MatrixStack:
    """An OpenGL-style push/pop transform stack with a fixed view/projection."""

    def __init__(self) -> None:
        self._stack: list[Mat4] = [Mat4() for _ in range(_STACK_SIZE)]
        self._top: int = 0
        self._view: Mat4 = Mat4()
        self._projection: Mat4 = Mat4()

    def push_matrix(self) -> None:
        if self._top + 1 >= _STACK_SIZE:
            raise MatrixStackError("Matrix stack overflow")
        self._top += 1
        self._stack[self._top] = self._stack[self._top - 1]

    def pop_matrix(self) -> None:
        self._stack[self._top] = Mat4()
        if self._top <= 0:
            raise MatrixStackError("Matrix stack underflow")
        self._top -= 1

    def identity(self) -> None:
        self._stack[self._top] = Mat4()

    def top(self) -> Mat4:
        return self._stack[self._top]

    def set_view(self, view: Mat4) -> None:
        self._view = view

    def set_projection(self, projection: Mat4) -> None:
        self._projection = projection

    def rotate_xyz(self, x: float, y: float, z: float) -> None:
        final = Mat4().rotate_z(z) @ Mat4().rotate_y(y) @ Mat4().rotate_x(x)
        self._stack[self._top] = self._stack[self._top] @ final

    def rotate_axis_angle(self, angle: float, x: float, y: float, z: float) -> None:
        r = Quaternion.from_axis_angle(Vec3(x, y, z), angle).to_mat4()
        self._stack[self._top] = self._stack[self._top] @ r

    def translate(self, x: float, y: float, z: float) -> None:
        self._stack[self._top] = self._stack[self._top] @ Mat4().translate(x, y, z)

    def scale(self, x: float, y: float, z: float) -> None:
        self._stack[self._top] = self._stack[self._top] @ Mat4().scale(x, y, z)

    def mvp(self) -> Mat4:
        return self._projection @ self._view @ self._stack[self._top]

    def mv(self) -> Mat4:
        return self._view @ self._stack[self._top]
```

Note: the C++ source has two overloaded `rotate()` methods (three-axis roll vs single axis-angle) — Python doesn't overload by argument count, so they're split into `rotate_xyz` and `rotate_axis_angle` above. Every call site in the C++ `paintGL` uses the axis-angle form (`m_stack.rotate(angle, 1,0,0)` etc.), so Step 3 below uses `rotate_axis_angle` throughout; `rotate_xyz` exists for interface completeness (it mirrors the header) but isn't called by this demo.

- [ ] **Step 2: Verify `Quaternion.from_axis_angle` and `Mat4` operator behaviour**

Run: `grep -n "def from_axis_angle\|def to_mat4" /Users/jmacey/teaching/Code/PyNGL/src/ncca/ngl/quaternion.py`
Expected: `from_axis_angle(cls, axis: Vec3, angle: float) -> Quaternion` and `to_mat4(self) -> Mat4` (already confirmed present during planning — this step is a final sanity check before relying on it).

- [ ] **Step 3: Write main.py**

Create `MatrixStack/main.py` (copy the `PySideEventHandlingMixin` skeleton from `Blending/main.py`, replace scene logic):

```python
#!/usr/bin/env -S uv run --script
"""
MatrixStack: an OpenGL-style push/pop matrix stack (OpenGL).

Demonstrates building a scene graph by hand with push_matrix()/pop_matrix()
instead of a Transform tree: three trolls, a ring of orbiting spheres, and a
reference grid, each pushed/popped around a shared stack.

Controls:
    I/O  increase / decrease the sphere ring's vertical wave frequency
    W/S  wireframe / solid
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import math
import sys
import traceback

import OpenGL.GL as gl
from matrix_stack import MatrixStack
from ncca.ngl import Mat3, Prims, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, PySideEventHandlingMixin, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


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
        self.setTitle("MatrixStack (OpenGL)")
        self.stack = MatrixStack()
        self.rotation: float = 0.0
        self.freq: float = 1.0

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        self.stack.set_view(look_at(Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0)))

        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 10.0, 10.0, 100)
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)

        self.startTimer(10)

    def load_matrices_to_shader(self) -> None:
        normal_matrix = Mat3.from_mat4(self.stack.mv()).inverse().transposed()
        ShaderLib.set_uniform("MVP", self.stack.mvp())
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

        self.stack.push_matrix()
        self.stack.translate(0, 0, self.model_position.z)
        self.stack.translate(self.model_position.x, self.model_position.y, 0)
        self.stack.rotate_axis_angle(self.spin_x_face, 1, 0, 0)
        self.stack.rotate_axis_angle(self.spin_y_face, 0, 1, 0)

        self.stack.push_matrix()
        self.stack.translate(0, -0.65, 0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(-1.0, -1.85, -1.0)
        self.stack.rotate_axis_angle(45, 0, 1, 0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(1.0, -1.85, -1.0)
        self.load_matrices_to_shader()
        Primitives.draw("troll")
        self.stack.pop_matrix()

        i = 0.0
        while i < 2.0 * math.pi:
            self.stack.push_matrix()
            x = math.cos(i) * 2.0
            z = math.sin(i) * 2.0
            y = math.sin(i * self.freq) * 0.5
            ShaderLib.set_uniform("Colour", abs(x), abs(y), abs(z), 1.0)
            self.stack.rotate_axis_angle(self.rotation, 0, 1, 0)
            self.stack.translate(x, y, z)
            self.stack.push_matrix()
            self.stack.scale(0.04, 0.04, 0.04)
            self.stack.rotate_axis_angle(self.rotation * 2, 0, 1, 0)
            self.load_matrices_to_shader()
            Primitives.draw("sphere")
            self.stack.pop_matrix()
            self.stack.pop_matrix()
            i += 0.05

        self.stack.push_matrix()
        self.stack.translate(0, -1.2, 0)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        self.load_matrices_to_shader()
        Primitives.draw("grid")
        self.stack.pop_matrix()
        self.stack.pop_matrix()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.stack.set_projection(perspective(45.0, float(w) / h, 0.05, 350.0))

    def timerEvent(self, event) -> None:
        self.rotation += 1.0
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_I:
            self.freq += 1.0
        elif key == Qt.Key_O:
            self.freq -= 1.0
        self.update()
        super().keyPressEvent(event)


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
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
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

Note: the C++ `paintGL` colours each orbiting sphere with `ShaderLib::setUniform("Colour",x,y,z,1.0f)` using the *signed* sine/cosine values directly as colour components (which OpenGL clamps to `[0,1]`, so negative components just render black-ish) — the Python port above uses `abs(x/y/z)` so the intended colour cycling is actually visible rather than mostly clamped to zero; this is a deliberate visual improvement over a literal port, not a bug. Also note: the C++ `if(i>180)` face-normal-vs-y-axis branch in the ring loop is dead code (`i` is radians, max ~6.28, never `>180`), so only its `else` branch (`rotate_axis_angle` about Y) ever executed — the Python port above keeps just that behaviour, dropping the unreachable branch.

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x MatrixStack/main.py
cd MatrixStack && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 5: Write README.md**

Create `MatrixStack/README.md`:

```markdown
# MatrixStack

An OpenGL-style push/pop matrix stack (`matrix_stack.py`), built by hand
instead of using `ncca.ngl.Transform`. Three trolls sit on a stack of
pushed/popped transforms, a ring of small spheres orbits with a wave in Y
whose frequency you can change live, and a reference grid sits underneath —
every draw call is wrapped in its own `push_matrix()`/`pop_matrix()` pair so
you can see exactly which state each object inherits from its parent.

`matrix_stack.py` has no GL/Qt dependency; the WebGPU version of this demo
(`main_webgpu.py`) imports the identical module, since the stack is pure
CPU-side matrix bookkeeping regardless of the rendering backend.

## Controls
- `I` / `O` : increase / decrease the sphere ring's wave frequency
- `W` / `S` : wireframe / solid
- Left-drag : orbit, Right-drag : pan, Wheel : zoom, `Space` : reset, `Esc` : quit

![MatrixStack](MatrixStack.png)
```

- [ ] **Step 6: Commit**

```bash
git add MatrixStack/matrix_stack.py MatrixStack/main.py MatrixStack/README.md
git commit -m "feat(matrix-stack): add OpenGL push/pop matrix stack demo"
```

---

## Task 2: MatrixStack (WebGPU)

**Files:**
- Create: `MatrixStack/main_webgpu.py`
- Create: `MatrixStack/MatrixStackShader.wgsl`
- Modify: `MatrixStack/README.md` (add WebGPU note)

**Interfaces:**
- Consumes: `MatrixStack` from `matrix_stack.py` (Task 1) unchanged.
- Produces: nothing consumed by later tasks.

**Note on scope:** WebGPU has no runtime primitive generator, only the baked `Primitives.npz` mesh set (verified: `troll`, `teapot`, `cube`, `bunny`, `buddah`, `dragon`, `football`, `octahedron`, `dodecahedron`, `icosahedron`, `tetrahedron` — **not** `sphere` or `line_grid`, which are GL-only runtime tessellations). This version therefore substitutes the ring's small GL-generated spheres with the baked `octahedron` mesh, and the reference grid with a small flat numpy quad (same helper pattern as `Blending/BlendingWebGPU.py`'s `quad()`). The matrix-stack push/pop logic — the actual teaching point — is identical to Task 1.

- [ ] **Step 1: Write the WGSL shader**

Create `MatrixStack/MatrixStackShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
    light_pos: vec4<f32>,
    light_diffuse: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(u.light_pos.xyz);
    let diffuse = max(dot(n, l), 0.0);
    let colour = u.colour.rgb * u.light_diffuse.rgb * diffuse;
    return vec4<f32>(colour, u.colour.a);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `MatrixStack/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""
MatrixStack: an OpenGL-style push/pop matrix stack (WebGPU).

Same push/pop matrix-stack logic as the OpenGL version (main.py) — the stack
itself (matrix_stack.py) is pure CPU-side maths shared unchanged between
both entry points. The ring of small GL-generated spheres is replaced with
the baked octahedron mesh, and the reference grid with a flat quad, since
WebGPU has no equivalent runtime primitive generator.

Controls:
    I/O  increase / decrease the sphere ring's vertical wave frequency
    LMB rotate  RMB pan  wheel zoom  Space reset  Esc quit
"""

import argparse
import math
import sys
import traceback

import numpy as np
import wgpu
from matrix_stack import MatrixStack
from ncca.ngl import PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype(
    [
        ("mvp", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("colour", np.float32, 4),
        ("light_pos", np.float32, 4),
        ("light_diffuse", np.float32, 4),
    ]
)


def quad_floor(size: float) -> np.ndarray:
    """Interleaved x,y,z,nx,ny,nz,u,v flat quad facing +y, centred at the origin."""
    h = size * 0.5
    corners = [(-h, 0, h), (h, 0, h), (h, 0, -h), (-h, 0, -h)]
    n = (0.0, 1.0, 0.0)
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    order = (0, 1, 2, 0, 2, 3)
    verts = [(*corners[i], *n, *uvs[i]) for i in order]
    return np.array(verts, dtype=np.float32).reshape(-1)


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MatrixStack (WebGPU)")
        self.msaa_sample_count = 4

        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.stack = MatrixStack()
        self.rotation = 0.0
        self.freq = 1.0

        self.stack.set_view(look_at(Vec3(0, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0)))
        self.stack.set_projection(
            perspective(45.0, self.width() / self.height(), 0.05, 350.0, PerspMode.WebGPU)
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_geometry()
        self._create_render_buffer()

        timer = QTimer(self)
        timer.timeout.connect(self._advance)
        timer.start(10)

    def _advance(self) -> None:
        self.rotation += 1.0
        self.update()

    def _create_pipeline(self) -> None:
        from pathlib import Path

        shader_src = (Path(__file__).parent / "MatrixStackShader.wgsl").read_text()
        self.shader_module = self.device.create_shader_module(code=shader_src)
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
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.shader_module,
                "entry_point": "fragment_main",
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

    def _make_object(self, data: np.ndarray):
        vertex_buffer = self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        uniform_buffer = self.device.create_buffer(
            size=uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": uniform_buffer,
                        "offset": 0,
                        "size": uniform_buffer.size,
                    },
                }
            ],
        )
        return {
            "vertex_buffer": vertex_buffer,
            "count": data.size // 8,
            "uniforms": uniforms,
            "uniform_buffer": uniform_buffer,
            "bind_group": bind_group,
        }

    def _create_geometry(self) -> None:
        self.troll = self._make_object(PrimData.primitive(Prims.TROLL.value))
        self.octahedron = self._make_object(PrimData.primitive(Prims.OCTAHEDRON.value))
        self.floor = self._make_object(quad_floor(10.0))

    def _draw_current(self, render_pass, obj: dict, colour: tuple) -> None:
        """Draw obj using the stack's *current* top-of-stack transform.

        Reuses MatrixStack.mvp()/mv() directly -- the exact same methods the
        OpenGL version calls -- so the only thing that differs between the
        two backends is how the result reaches the GPU (a GL uniform vs a
        WebGPU uniform buffer), not how it's computed.
        """
        obj["uniforms"]["mvp"] = self.stack.mvp().to_numpy()
        obj["uniforms"]["normal_matrix"] = self.stack.mv().inverse().transposed().to_numpy()
        obj["uniforms"]["colour"] = colour
        obj["uniforms"]["light_pos"] = (1.0, 1.0, 1.0, 0.0)
        obj["uniforms"]["light_diffuse"] = (1.0, 1.0, 1.0, 1.0)
        self.device.queue.write_buffer(obj["uniform_buffer"], 0, obj["uniforms"].tobytes())
        render_pass.set_bind_group(0, obj["bind_group"], [], 0, 999999)
        render_pass.set_vertex_buffer(0, obj["vertex_buffer"])
        render_pass.draw(obj["count"])

    def paintWebGPU(self) -> None:
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.pipeline)

        # Identical push/pop structure to the OpenGL version's paintGL --
        # this is the actual point of the demo: the same MatrixStack calls
        # drive both backends.
        self.stack.push_matrix()
        self.stack.translate(0, 0, self.model_position.z)
        self.stack.translate(self.model_position.x, self.model_position.y, 0)
        self.stack.rotate_axis_angle(self.spin_x_face, 1, 0, 0)
        self.stack.rotate_axis_angle(self.spin_y_face, 0, 1, 0)

        self.stack.push_matrix()
        self.stack.translate(0, -0.65, 0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(-1.0, -1.85, -1.0)
        self.stack.rotate_axis_angle(45, 0, 1, 0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        self.stack.push_matrix()
        self.stack.scale(0.5, 0.5, 0.5)
        self.stack.translate(1.0, -1.85, -1.0)
        self._draw_current(render_pass, self.troll, (1, 1, 1, 1))
        self.stack.pop_matrix()

        i = 0.0
        while i < 2.0 * math.pi:
            self.stack.push_matrix()
            x = math.cos(i) * 2.0
            z = math.sin(i) * 2.0
            y = math.sin(i * self.freq) * 0.5
            self.stack.rotate_axis_angle(self.rotation, 0, 1, 0)
            self.stack.translate(x, y, z)
            self.stack.push_matrix()
            self.stack.scale(0.04, 0.04, 0.04)
            self.stack.rotate_axis_angle(self.rotation * 2, 0, 1, 0)
            self._draw_current(render_pass, self.octahedron, (abs(x), abs(y), abs(z), 1.0))
            self.stack.pop_matrix()
            self.stack.pop_matrix()
            i += 0.05

        self.stack.push_matrix()
        self.stack.translate(0, -1.2, 0)
        self._draw_current(render_pass, self.floor, (1, 1, 1, 1))
        self.stack.pop_matrix()
        self.stack.pop_matrix()

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.stack.set_projection(
            perspective(45.0, width / height, 0.05, 350.0, PerspMode.WebGPU)
        )
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_I:
            self.freq += 1.0
        elif key == Qt.Key_O:
            self.freq -= 1.0
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x MatrixStack/main_webgpu.py
cd MatrixStack && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `MatrixStack/README.md` (after the Controls section):

```markdown
## WebGPU version

`main_webgpu.py` shares `matrix_stack.py` unchanged with the OpenGL version
— the push/pop logic is pure CPU-side matrix maths, independent of the
rendering backend. WebGPU has no runtime primitive generator, so the ring's
small spheres are the baked octahedron mesh instead, and the grid is a flat
quad.
```

- [ ] **Step 5: Commit**

```bash
git add MatrixStack/main_webgpu.py MatrixStack/MatrixStackShader.wgsl MatrixStack/README.md
git commit -m "feat(matrix-stack): add WebGPU entry point"
```

---

## Task 3: LookAtDemos (OpenGL)

**Files:**
- Create: `LookAtDemos/main.py`
- Create: `LookAtDemos/README.md`

**Interfaces:**
- Produces: nothing consumed by later tasks (WebGPU version in Task 4 duplicates the camera-rig data, since it's a handful of `Vec3` literals, not worth extracting to a shared module).

**Design (combining the two NGL9Demos sub-demos into one, per the spec):** press `Tab` to switch between **Simple** mode (one perspective `look_at` camera, full mouse orbit/pan/zoom via the mixin, draws "troll") and **Multi** mode (a fixed 2x2 grid: top-left = top-down ortho, top-right = perspective view driven by the same mouse orbit, bottom-left = front ortho, bottom-right = side ortho — each shows "troll" + a reference grid). Only the perspective quadrant rotates with the mouse in Multi mode; the three ortho views are fixed reference angles, matching the original's role as "orthogonal reference views vs. an interactive perspective view." This drops the original C++'s per-quadrant independent mouse-routing/fullscreen-toggle state (`m_activeWindow`, `getActiveQuadrant()`) as unnecessary complexity for the teaching point (comparing `lookAt`+`perspective` vs `lookAt`+`ortho`).

- [ ] **Step 1: Write main.py**

Create `LookAtDemos/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""
LookAtDemos: ngl::lookAt and ngl::perspective/ortho (OpenGL).

Combines NGL9Demos' SimpleLookAt and MultipleViews demos. Tab switches
between a single interactive perspective camera and a 2x2 grid comparing
that same perspective view against three fixed orthographic reference views
(top, front, side) of the identical scene.

Controls:
    Tab  toggle simple / multi-view mode
    LMB rotate  RMB pan  wheel zoom (perspective view only)  Space reset  Esc quit
"""

import argparse
import sys
import traceback

import OpenGL.GL as gl
from ncca.ngl import Mat4, Prims, Vec3, logger, look_at, ortho, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, PySideEventHandlingMixin, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


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
        self.setTitle("LookAtDemos (OpenGL)")
        self.multi_view: bool = False
        self.mouse_global_tx: Mat4 = Mat4()

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.LINE_GRID, "grid", 4.0, 4.0, 40)

    def _load_matrices(self, view: Mat4, project: Mat4, model: Mat4) -> None:
        ShaderLib.set_uniform("MVP", project @ view @ model)
        mv = view @ model
        from ncca.ngl import Mat3

        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(mv).inverse().transposed())

    def _draw_scene(self, view: Mat4, project: Mat4, global_tx: Mat4) -> None:
        self._load_matrices(view, project, global_tx)
        Primitives.draw("troll")
        self._load_matrices(view, project, global_tx @ Mat4().translate(0, -1.0, 0))
        Primitives.draw("grid")

    def _viewport_rect(self, quadrant: str) -> tuple[int, int, int, int]:
        half_w = self.window_width // 2
        half_h = self.window_height // 2
        return {
            "top": (0, half_h, half_w, half_h),
            "persp": (half_w, half_h, half_w, half_h),
            "front": (0, 0, half_w, half_h),
            "side": (half_w, 0, half_w, half_h),
        }[quadrant]

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        if not self.multi_view:
            gl.glViewport(0, 0, self.window_width, self.window_height)
            view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
            aspect = self.window_width / self.window_height
            project = perspective(45.0, aspect, 0.05, 350.0)
            self._draw_scene(view, project, self.mouse_global_tx)
            return

        aspect = 1.0
        x, y, w, h = self._viewport_rect("top")
        gl.glViewport(x, y, w, h)
        view = look_at(Vec3(0, 2, 0), Vec3(0, 0, 0), Vec3(0, 0, -1))
        self._draw_scene(view, ortho(-1, 1, -1, 1, 0.1, 100), Mat4())

        x, y, w, h = self._viewport_rect("front")
        gl.glViewport(x, y, w, h)
        view = look_at(Vec3(0, 0, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self._draw_scene(view, ortho(-1, 1, -1, 1, 0.01, 200), Mat4())

        x, y, w, h = self._viewport_rect("side")
        gl.glViewport(x, y, w, h)
        view = look_at(Vec3(2, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self._draw_scene(view, ortho(-1, 1, -1, 1, 0.1, 100), Mat4())

        x, y, w, h = self._viewport_rect("persp")
        gl.glViewport(x, y, w, h)
        view = look_at(Vec3(0, 1, 1), Vec3(0, 0, 0), Vec3(0, 1, 0))
        project = perspective(45.0, w / h, 0.01, 100.0)
        self._draw_scene(view, project, self.mouse_global_tx)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Tab:
            self.multi_view = not self.multi_view
        self.update()
        super().keyPressEvent(event)


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

Note: `_load_matrices` imports `Mat3` locally to keep the top-level import block matching what's actually used at module scope elsewhere in this file — if `ruff` flags this as an avoidable local import in Step 3's lint pass, move `Mat3` into the top-level `from ncca.ngl import ...` line instead (either is fine; just be consistent).

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x LookAtDemos/main.py
cd LookAtDemos && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 3: Write README.md**

Create `LookAtDemos/README.md`:

```markdown
# LookAtDemos

Combines NGL9Demos' SimpleLookAt and MultipleViews demos. `Tab` switches
between a single interactive perspective camera (`ngl.look_at` +
`ngl.perspective`) and a 2x2 grid comparing that same perspective view
against three fixed orthographic reference views (top, front, side) of the
identical troll-and-grid scene, built with `ngl.ortho`.

## Controls
- `Tab` : toggle simple / multi-view mode
- Left-drag : orbit, Right-drag : pan, Wheel : zoom (perspective view only)
- `Space` : reset, `Esc` : quit

![LookAtDemos](LookAtDemos.png)
```

- [ ] **Step 4: Commit**

```bash
git add LookAtDemos/main.py LookAtDemos/README.md
git commit -m "feat(look-at-demos): add OpenGL lookAt/perspective/ortho comparison demo"
```

---

## Task 4: LookAtDemos (WebGPU)

**Files:**
- Create: `LookAtDemos/main_webgpu.py`
- Create: `LookAtDemos/LookAtShader.wgsl`
- Modify: `LookAtDemos/README.md` (add WebGPU note)

**Interfaces:** none consumed by later tasks.

**Note on scope:** multi-viewport rendering within a single WebGPU render pass follows the exact pattern already used in `BVHViewer/main_webgpu.py` (`render_pass.set_viewport(x, y, width, height, 0.0, 1.0)` + `render_pass.set_scissor_rect(x, y, width, height)` per pane, one `draw` call each, inside one `begin_render_pass`). The reference grid is dropped for this version (no baked `line_grid` WebGPU data) — each viewport shows the troll only; the camera comparison (the actual teaching point) is unaffected.

- [ ] **Step 1: Write the WGSL shader**

Create `LookAtDemos/LookAtShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(n, l), 0.0);
    return vec4<f32>(vec3<f32>(1.0, 1.0, 1.0) * diffuse, 1.0);
}
```

- [ ] **Step 2: Verify the wgpu-py viewport/scissor API**

Run: `grep -n "set_viewport\|set_scissor_rect" /Volumes/teaching/Code/PyNGLDemos/BVHViewer/main_webgpu.py`
Expected: `render_pass.set_viewport(x, y, width, height, 0.0, 1.0)` and `render_pass.set_scissor_rect(x, y, width, height)` (already confirmed present during planning — final sanity check).

- [ ] **Step 3: Write main_webgpu.py**

Create `LookAtDemos/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""
LookAtDemos: ngl::lookAt and ngl::perspective/ortho (WebGPU).

Same simple/multi-view comparison as the OpenGL version (main.py), using
render_pass.set_viewport()/set_scissor_rect() to draw all four quadrants in
one WebGPU render pass (same technique as BVHViewer's four-view mode). The
reference grid is dropped (no baked WebGPU line-grid data); each viewport
shows the troll only.

Controls:
    Tab  toggle simple / multi-view mode
    LMB rotate  RMB pan  wheel zoom (perspective view only)  Space reset  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, ortho, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype([("mvp", np.float32, (4, 4)), ("normal_matrix", np.float32, (4, 4))])


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LookAtDemos (WebGPU)")
        self.msaa_sample_count = 4
        self.multi_view = False

        self.mouse_global_tx = Mat4()
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.device = get_default_device()
        self._create_pipeline()

        troll_data = PrimData.primitive(Prims.TROLL.value)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=troll_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = troll_data.size // 8
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = self.device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
        )
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "LookAtShader.wgsl").read_text()
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
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
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

    def _draw_pane(self, render_pass, view: Mat4, project: Mat4, model: Mat4) -> None:
        mv = view @ model
        self.uniforms["mvp"] = (project @ mv).to_numpy()
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)

    def paintWebGPU(self) -> None:
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
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.pipeline)

        w, h = self.width(), self.height()
        if not self.multi_view:
            render_pass.set_viewport(0, 0, w, h, 0.0, 1.0)
            render_pass.set_scissor_rect(0, 0, w, h)
            view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
            project = perspective(45.0, w / h, 0.05, 350.0, PerspMode.WebGPU)
            self._draw_pane(render_pass, view, project, self.mouse_global_tx)
        else:
            half_w, half_h = w // 2, h // 2
            panes = [
                ((0, half_h, half_w, half_h), Vec3(0, 2, 0), Vec3(0, 0, -1), True),
                ((half_w, half_h, half_w, half_h), Vec3(0, 1, 1), Vec3(0, 1, 0), False),
                ((0, 0, half_w, half_h), Vec3(0, 0, 2), Vec3(0, 1, 0), True),
                ((half_w, 0, half_w, half_h), Vec3(2, 0, 0), Vec3(0, 1, 0), True),
            ]
            for (x, y, pw, ph), eye, up, is_ortho in panes:
                render_pass.set_viewport(x, y, pw, ph, 0.0, 1.0)
                render_pass.set_scissor_rect(x, y, pw, ph)
                view = look_at(eye, Vec3(0, 0, 0), up)
                if is_ortho:
                    project = ortho(-1, 1, -1, 1, 0.01, 200, PerspMode.WebGPU)
                    model = Mat4()
                else:
                    project = perspective(45.0, pw / max(ph, 1), 0.01, 100.0, PerspMode.WebGPU)
                    model = self.mouse_global_tx
                self._draw_pane(render_pass, view, project, model)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Tab:
            self.multi_view = not self.multi_view
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x LookAtDemos/main_webgpu.py
cd LookAtDemos && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. If `render_pass.set_viewport`/`set_scissor_rect` reject the integer types passed here (BVHViewer's call sites may pass floats for the viewport call specifically — check its exact call if this errors), cast `x, y, pw, ph` to `float(...)` for the `set_viewport` call only (`set_scissor_rect` takes ints).

- [ ] **Step 5: Add the WebGPU note to README.md**

Append to `LookAtDemos/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` draws all four quadrants in a single render pass using
`render_pass.set_viewport()` / `set_scissor_rect()` per pane (the same
technique `BVHViewer`'s four-view mode uses). It omits the reference grid,
since WebGPU has no baked line-grid primitive data — the camera comparison
itself is unaffected.
```

- [ ] **Step 6: Commit**

```bash
git add LookAtDemos/main_webgpu.py LookAtDemos/LookAtShader.wgsl LookAtDemos/README.md
git commit -m "feat(look-at-demos): add WebGPU entry point"
```

---

## Task 5: ViewToWorldTransform maths + OpenGL

**Files:**
- Create: `ViewToWorldTransform/view_to_world.py`
- Create: `ViewToWorldTransform/tests/test_view_to_world.py`
- Create: `ViewToWorldTransform/main.py`
- Create: `ViewToWorldTransform/README.md`

**Interfaces:**
- Produces: `unproject_point(x: float, y: float, width: int, height: int, view_projection: np.ndarray, ndc_z: float = 1.0) -> np.ndarray` in `ViewToWorldTransform/view_to_world.py`. Pure numpy, no GL/Qt imports — Task 6 (WebGPU) imports it unchanged. `view_projection` is `(projection @ view).to_numpy()` (world-space, no model term).

- [ ] **Step 1: Write the failing tests**

Create `ViewToWorldTransform/tests/test_view_to_world.py`:

```python
"""Headless tests for the screen-to-world unprojection maths."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from view_to_world import unproject_point  # noqa: E402


class TestUnprojectPoint:
    def test_identity_centre_of_screen_far_plane(self):
        """With an identity view-projection, the screen centre at the far
        plane (ndc_z=1, the default) maps to NDC (0,0,1)."""
        point = unproject_point(400, 300, 800, 600, np.eye(4))
        np.testing.assert_allclose(point, [0.0, 0.0, 1.0], atol=1e-6)

    def test_identity_top_left_corner(self):
        """Top-left pixel maps to NDC (-1, +1)."""
        point = unproject_point(0, 0, 800, 600, np.eye(4))
        np.testing.assert_allclose(point[:2], [-1.0, 1.0], atol=1e-6)

    def test_near_plane_selectable(self):
        """ndc_z=-1 selects the near plane instead of the far plane."""
        point = unproject_point(400, 300, 800, 600, np.eye(4), ndc_z=-1.0)
        np.testing.assert_allclose(point, [0.0, 0.0, -1.0], atol=1e-6)

    def test_translated_view_projection_offsets_result(self):
        """A translation in the view-projection matrix shows up in the
        unprojected point (row-vector convention: translation in row 3)."""
        vp = np.eye(4)
        vp[3, :3] = [5.0, -2.0, 1.0]
        point = unproject_point(400, 300, 800, 600, vp)
        np.testing.assert_allclose(point, [5.0, -2.0, 2.0], atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ViewToWorldTransform && uv run pytest tests/ -v; cd ..`
Expected: `ModuleNotFoundError: No module named 'view_to_world'` (the module doesn't exist yet).

- [ ] **Step 3: Write the maths module**

Create `ViewToWorldTransform/view_to_world.py`:

```python
"""Screen-to-world unprojection, ported from NGL9Demos/ViewToWorldTransform.

Pure numpy, no GL/Qt/wgpu imports, so this is unit-testable headless and
shared unchanged between the OpenGL and WebGPU entry points of this demo.
All matrices follow the PyNGL row-vector convention: points transform as
``row_vector @ matrix``.
"""

import numpy as np


def unproject_point(
    x: float,
    y: float,
    width: int,
    height: int,
    view_projection: np.ndarray,
    ndc_z: float = 1.0,
) -> np.ndarray:
    """Unproject a screen pixel at a fixed NDC depth into world space.

    x, y are pixel coordinates with Qt's top-left origin. view_projection is
    ``(projection @ view).to_numpy()`` (no model term, so the result is a
    world-space point directly). ndc_z selects the depth plane in OpenGL NDC
    (-1 near .. +1 far); the demo defaults to the far plane (ndc_z=1.0),
    matching NGL9Demos' ``ngl::unProject(Vec3(x, y, 1.0f), ...)`` call.
    """
    ndc_x = 2.0 * x / width - 1.0
    ndc_y = 1.0 - 2.0 * y / height
    inverse = np.linalg.inv(view_projection.astype(np.float64))
    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0]) @ inverse
    return (clip[:3] / clip[3]).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ViewToWorldTransform && uv run pytest tests/ -v; cd ..`
Expected: all 4 tests PASS.

- [ ] **Step 5: Write main.py**

Create `ViewToWorldTransform/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""
ViewToWorldTransform: click to place objects in world space (OpenGL).

Shift+left-click unprojects the mouse position at the camera's far plane
into a world-space point (view_to_world.unproject_point) and places a cube
there. The HUD shows the last click's screen and world coordinates.

Controls:
    Shift+LMB  place a cube at the unprojected point
    Space  clear placed cubes
    LMB rotate  RMB pan  wheel zoom  Esc quit
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
    Text,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from view_to_world import unproject_point


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
        self.setTitle("ViewToWorldTransform (OpenGL)")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.click_positions: list[Vec3] = []
        self.last_click_screen: tuple[int, int] | None = None

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        self.view = look_at(Vec3(0, 0, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        Primitives.load_default_primitives()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 16
        )

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ShaderLib.use(DefaultShader.DIFFUSE)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        tx = Transform()
        for position in self.click_positions:
            tx.set_position(position.x, position.y, position.z)
            mv = self.view @ global_tx @ tx.matrix()
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(mv).inverse().transposed())
            Primitives.draw("cube")

        if self.last_click_screen is not None and self.click_positions:
            last = self.click_positions[-1]
            sx, sy = self.last_click_screen
            text = f"Pos=({sx},{sy}) World=({last.x:.2f},{last.y:.2f},{last.z:.2f})"
            Text.render_text("Arial", 10, 10, text, Vec3(1.0, 1.0, 1.0))

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 50.0)
        Text.set_screen_size(w, h)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self.click_positions.clear()
            self.last_click_screen = None
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            position = event.position()
            x, y = int(position.x()), int(position.y())
            view_projection = (self.project @ self.view).to_numpy()
            world = unproject_point(x, y, self.window_width, self.window_height, view_projection)
            self.click_positions.append(Vec3(*world))
            self.last_click_screen = (x, y)
            self.update()
        else:
            super().mousePressEvent(event)


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

Note: `window_width`/`window_height` in `mousePressEvent` are set from `resizeGL` (which multiplies by `devicePixelRatio()`), but `event.position()` reports *logical* (non-DPI-scaled) pixels — on a HiDPI display the two won't quite agree, giving a slightly-off unprojected point. None of this repo's existing demos correct for this either (see `RayPickingSelection`, which has the same mismatch), so this is consistent with established practice, not a new bug — leave it as is.

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x ViewToWorldTransform/main.py
cd ViewToWorldTransform && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 7: Write README.md**

Create `ViewToWorldTransform/README.md`:

```markdown
# ViewToWorldTransform

Shift-click anywhere in the viewport to unproject the screen position into a
world-space point and drop a cube there. The unprojection
(`view_to_world.unproject_point`) is a pure-numpy module with pytest
coverage (`tests/test_view_to_world.py`), shared unchanged with the WebGPU
version of this demo.

## Controls
- `Shift`+LMB : place a cube at the unprojected point
- `Space` : clear placed cubes
- Left-drag : orbit, Right-drag : pan, Wheel : zoom, `Esc` : quit

![ViewToWorldTransform](ViewToWorldTransform.png)
```

- [ ] **Step 8: Commit**

```bash
git add ViewToWorldTransform/view_to_world.py ViewToWorldTransform/tests/ ViewToWorldTransform/main.py ViewToWorldTransform/README.md
git commit -m "feat(view-to-world-transform): add unprojection maths and OpenGL demo"
```

---

## Task 6: ViewToWorldTransform (WebGPU)

**Files:**
- Create: `ViewToWorldTransform/main_webgpu.py`
- Create: `ViewToWorldTransform/ViewToWorldShader.wgsl`
- Modify: `ViewToWorldTransform/README.md` (add WebGPU note)

**Interfaces:**
- Consumes: `unproject_point` from `view_to_world.py` (Task 5) unchanged.

- [ ] **Step 1: Write the WGSL shader**

Create `ViewToWorldTransform/ViewToWorldShader.wgsl`:

```wgsl
struct Uniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(n, l), 0.0);
    return vec4<f32>(vec3<f32>(1.0, 1.0, 1.0) * diffuse, 1.0);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `ViewToWorldTransform/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""
ViewToWorldTransform: click to place objects in world space (WebGPU).

Same shift-click-to-place behaviour as the OpenGL version (main.py), sharing
the identical view_to_world.unproject_point maths.

Controls:
    Shift+LMB  place a cube at the unprojected point
    Space  clear placed cubes
    LMB rotate  RMB pan  wheel zoom  Esc quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from view_to_world import unproject_point
from wgpu.utils import get_default_device

UNIFORM_DTYPE = np.dtype([("mvp", np.float32, (4, 4)), ("normal_matrix", np.float32, (4, 4))])


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ViewToWorldTransform (WebGPU)")
        self.msaa_sample_count = 4

        self.mouse_global_tx = Mat4()
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.click_positions: list[Vec3] = []
        self.last_click_screen: tuple[int, int] | None = None

        self.view = look_at(Vec3(0, 0, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, self.width() / self.height(), 0.5, 50.0, PerspMode.WebGPU)

        self.device = get_default_device()
        self._create_pipeline()

        cube_data = PrimData.primitive(Prims.CUBE.value)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=cube_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.vertex_count = cube_data.size // 8
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "ViewToWorldShader.wgsl").read_text()
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
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
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

    def paintWebGPU(self) -> None:
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
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.pipeline)

        uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        for position in self.click_positions:
            model = Mat4().translate(position.x, position.y, position.z)
            mv = self.view @ self.mouse_global_tx @ model
            uniforms["mvp"] = (self.project @ mv).to_numpy()
            uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
            uniform_buffer = self.device.create_buffer_with_data(
                data=uniforms.tobytes(), usage=wgpu.BufferUsage.UNIFORM
            )
            bind_group = self.device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": uniform_buffer,
                            "offset": 0,
                            "size": uniform_buffer.size,
                        },
                    }
                ],
            )
            render_pass.set_bind_group(0, bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.vertex_count)

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

        if self.last_click_screen is not None and self.click_positions:
            last = self.click_positions[-1]
            sx, sy = self.last_click_screen
            text = f"Pos=({sx},{sy}) World=({last.x:.2f},{last.y:.2f},{last.z:.2f})"
            self.render_text(10, 20, text, 14, "Arial", QColor(255, 255, 255))

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.5, 50.0, PerspMode.WebGPU)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.click_positions.clear()
            self.last_click_screen = None
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            x, y = int(position.x()), int(position.y())
            view_projection = (self.project @ self.view).to_numpy()
            world = unproject_point(x, y, self.width(), self.height(), view_projection)
            self.click_positions.append(Vec3(*world))
            self.last_click_screen = (x, y)
            self.update()
        elif event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x ViewToWorldTransform/main_webgpu.py
cd ViewToWorldTransform && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `ViewToWorldTransform/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` shares `view_to_world.py`'s `unproject_point` unchanged
with the OpenGL version — screen-to-world unprojection is pure numpy maths,
independent of the rendering backend.
```

- [ ] **Step 5: Commit**

```bash
git add ViewToWorldTransform/main_webgpu.py ViewToWorldTransform/ViewToWorldShader.wgsl ViewToWorldTransform/README.md
git commit -m "feat(view-to-world-transform): add WebGPU entry point"
```

---

## Task 7: AffineTransforms (OpenGL)

**Files:**
- Create: `AffineTransforms/axis.py`
- Create: `AffineTransforms/shaders/PBRVertex.glsl` (copy from `Camera/shaders/PBRVertex.glsl`)
- Create: `AffineTransforms/shaders/PBRFragment.glsl` (copy from `Camera/shaders/PBRFragment.glsl`)
- Create: `AffineTransforms/shaders/normalVertex.glsl`, `AffineTransforms/shaders/normalGeo.glsl`, `AffineTransforms/shaders/normalFragment.glsl`
- Create: `AffineTransforms/main.py`
- Create: `AffineTransforms/README.md`

**Interfaces:**
- Produces: `draw_axis(view: Mat4, project: Mat4, global_tx: Mat4, scale: float = 1.5) -> None` in `AffineTransforms/axis.py` (uses `DefaultShader.COLOUR`; caller must have `Primitives.load_default_primitives()` already called since it draws `"cylinder"`/`"cone"`).

**Design notes (scope decisions from the design spec, see spec doc for detail):**
- Matrix-order combo is simplified from the C++ original's 5 options to 3: **Rotate→Translate→Scale**, **Translate→Rotate→Scale**, **Translate→Axis-Angle→Scale** — dropping the hand-crafted `GIMBALLOCK` raw-matrix-element demo (a C++-memory-layout-specific hack, not portable) and the redundant `TEULERS` variant (same ingredients as axis-angle mode, different order, marginal teaching value on top of RTS/TRS already showing order matters).
- Reuses `Camera/shaders/PBRVertex.glsl` + `PBRFragment.glsl` (3-light array PBR, already proven working in this repo) rather than porting AffineTransforms' own single-light PBR shader variant from NGL9Demos — avoids introducing and debugging a second GLSL PBR permutation for no teaching benefit (the point of this demo is matrix order, not lighting).
- No `.ui` file: the control panel is built as plain PySide widgets in code (matches `MassSpring/main.py`'s established precedent in this repo, not `ShadingModels`' heavier `QUiLoader` pattern).
- Reuses `ncca.ngl.widgets.Vec3Widget` and `Mat4Widget` (already exist in the library — verified via `ncca/ngl/widgets/__init__.py`) instead of hand-building spinbox grids.

- [ ] **Step 1: Copy the PBR shaders**

```bash
mkdir -p AffineTransforms/shaders
cp Camera/shaders/PBRVertex.glsl AffineTransforms/shaders/PBRVertex.glsl
cp Camera/shaders/PBRFragment.glsl AffineTransforms/shaders/PBRFragment.glsl
```

- [ ] **Step 2: Write the normal-visualization shaders**

Create `AffineTransforms/shaders/normalVertex.glsl`:

```glsl
#version 330 core
layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
uniform mat4 MVP;

uniform float normalSize;
uniform vec4 vertNormalColour;
uniform vec4 faceNormalColour;

out vec4 normal;

uniform bool drawFaceNormals;
uniform bool drawVertexNormals;

void main(void)
{
  gl_Position = MVP*vec4(inVert,1);
  normal = MVP*vec4(inNormal,0);
}
```

Create `AffineTransforms/shaders/normalGeo.glsl`:

```glsl
#version 330 core
layout(triangles) in;
layout(line_strip, max_vertices = 8) out;

in vec4 normal[];

uniform float normalSize;
uniform vec4 vertNormalColour;
uniform vec4 faceNormalColour;
uniform bool drawFaceNormals;
uniform bool drawVertexNormals;
out vec4 perNormalColour;

void main()
{
    if (drawVertexNormals == true)
    {
        perNormalColour = vertNormalColour;
        for(int i = 0; i < gl_in.length(); ++i)
        {
            gl_Position = gl_in[i].gl_Position;
            EmitVertex();
            gl_Position = gl_in[i].gl_Position + (normal[i] * normalSize);
            EmitVertex();
            EndPrimitive();
        }
    }
    if (drawFaceNormals == true)
    {
        perNormalColour = faceNormalColour;
        vec4 cent = (gl_in[0].gl_Position + gl_in[1].gl_Position + gl_in[2].gl_Position) / 3.0;
        vec3 face_normal = normalize(cross(gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz,
                                            gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz));
        gl_Position = cent;
        EmitVertex();
        gl_Position = (cent + vec4(face_normal * normalSize, 0.0));
        EmitVertex();
        EndPrimitive();
    }
}
```

Create `AffineTransforms/shaders/normalFragment.glsl`:

```glsl
#version 330 core
layout (location=0) out vec4 fragColour;
in vec4 perNormalColour;

void main()
{
  fragColour = perNormalColour;
}
```

- [ ] **Step 3: Verify the PBR shader's uniform names**

Run: `grep -n "uniform" Camera/shaders/PBRFragment.glsl`
Expected: confirms the exact uniform names used by `Camera/main.py` (`lightPositions[0..2]`, `lightColours[0..2]`, `albedo`, `metallic`, `roughness`, `ao`, `camPos`) — use these exact names in Step 5 below, matching how `Camera/main.py` sets them.

- [ ] **Step 4: Write the axis helper**

Create `AffineTransforms/axis.py`:

```python
"""A simple RGB axis gizmo, ported from NGL9Demos/AffineTransforms/Axis."""

from __future__ import annotations

from ncca.ngl import Mat4, Transform
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib


def _load_matrices(view: Mat4, project: Mat4, global_tx: Mat4, model: Mat4) -> None:
    mv = view @ global_tx @ model
    ShaderLib.set_uniform("MVP", project @ mv)


def draw_axis(view: Mat4, project: Mat4, global_tx: Mat4, scale: float = 1.5) -> None:
    """Draw a red/green/blue X/Y/Z axis gizmo at the origin.

    Requires Primitives.load_default_primitives() to already have been
    called (draws "cylinder" and "cone").
    """
    ShaderLib.use(DefaultShader.COLOUR)
    tx = Transform()

    # X axis (red)
    ShaderLib.set_uniform("Colour", 1.0, 0.0, 0.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(scale, 0, 0)
    tx.set_rotation(0, 90, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(scale, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(-scale, 0, 0)
    tx.set_rotation(0, -90, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")

    # Y axis (green)
    ShaderLib.set_uniform("Colour", 0.0, 1.0, 0.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(0, -scale, 0)
    tx.set_rotation(90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(0, scale, 0)
    tx.set_rotation(-90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(0, -scale, 0)
    tx.set_rotation(90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")

    # Z axis (blue)
    ShaderLib.set_uniform("Colour", 0.0, 0.0, 1.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(0, 0, scale)
    tx.set_rotation(0, 0, -90)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(0, 0, scale)
    tx.set_rotation(0, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(0, 0, -scale)
    tx.set_rotation(180, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
```

- [ ] **Step 5: Write main.py**

Create `AffineTransforms/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""
AffineTransforms: interactively compose translate/rotate/scale matrices (OpenGL).

A PBR-shaded primitive sits at the origin next to an RGB axis gizmo. Sliders
set independent translate/rotate/scale values and a combo box picks the
*order* those are composed in — Rotate-Translate-Scale, Translate-Rotate-
Scale, or Translate-(axis-angle)-Scale — so you can see directly how order
changes the result. A read-only matrix grid shows the composed transform.

Controls: all on the panel; left-drag in the viewport orbits the camera.
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from axis import draw_axis
from ncca.ngl import Mat3, Mat4, Prims, Quaternion, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import Primitives, PySideEventHandlingMixin, ShaderLib
from ncca.ngl.widgets import Mat4Widget, Vec3Widget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_VBO_NAMES = [
    "sphere", "cylinder", "cone", "disk", "plane", "torus", "teapot",
    "octahedron", "dodecahedron", "icosahedron", "tetrahedron", "football",
    "cube", "troll", "buddah", "dragon", "bunny",
]


class Scene(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.window_width: int = 800
        self.window_height: int = 720
        self.setTitle("AffineTransforms (OpenGL)")

        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.draw_index: int = 6  # teapot
        self.wireframe: bool = False
        self.draw_normals: bool = False
        self.normal_size: float = 0.6
        self.colour: tuple[float, float, float] = (0.95, 0.71, 0.29)

        self.translate_v = Vec3(0, 0, 0)
        self.rotate_v = Vec3(0, 0, 0)
        self.scale_v = Vec3(1, 1, 1)
        self.axis_angle: float = 0.0
        self.axis_v = Vec3(1, 0, 0)
        self.order: str = "RTS"

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        from_pos = Vec3(0, 0, 8)
        self.view = look_at(from_pos, Vec3(0, 0, 0), Vec3(0, 1, 0))

        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 40)
        Primitives.create(Prims.CYLINDER, "cylinder", 0.5, 1.4, 40, 40)
        Primitives.create(Prims.CONE, "cone", 0.5, 1.4, 20, 20)
        Primitives.create(Prims.DISK, "disk", 0.5, 40)
        Primitives.create(Prims.TRIANGLE_PLANE, "plane", 1.0, 1.0, 10, 10, Vec3(0, 1, 0))
        Primitives.create(Prims.TORUS, "torus", 0.15, 0.4, 40, 40)

        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "PBR", str(shader_dir / "PBRVertex.glsl"), str(shader_dir / "PBRFragment.glsl")
        )
        ShaderLib.use("PBR")
        ShaderLib.set_uniform("camPos", from_pos)
        ShaderLib.set_uniform("lightPositions[0]", 0.0, 2.0, 2.0)
        ShaderLib.set_uniform("lightColours[0]", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("lightPositions[1]", -10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[1]", 0.0, 0.0, 0.0)
        ShaderLib.set_uniform("lightPositions[2]", 10.0, 4.0, -10.0)
        ShaderLib.set_uniform("lightColours[2]", 0.0, 0.0, 0.0)
        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)

        ShaderLib.load_shader(
            "NormalViz",
            str(shader_dir / "normalVertex.glsl"),
            str(shader_dir / "normalFragment.glsl"),
            str(shader_dir / "normalGeo.glsl"),
        )
        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("vertNormalColour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("faceNormalColour", 1.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("drawFaceNormals", True)
        ShaderLib.set_uniform("drawVertexNormals", True)

    def transform_matrix(self) -> Mat4:
        t = Mat4().translate(self.translate_v.x, self.translate_v.y, self.translate_v.z)
        s = Mat4().scale(self.scale_v.x, self.scale_v.y, self.scale_v.z)
        if self.order == "RTS":
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return r @ t @ s
        elif self.order == "TRS":
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return t @ r @ s
        else:  # "TAxisS": translate, axis-angle rotation, scale
            r = Quaternion.from_axis_angle(self.axis_v, self.axis_angle).to_mat4()
            return t @ r @ s

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        global_tx = rot_y @ rot_x
        global_tx[3, 0] = self.model_position.x
        global_tx[3, 1] = self.model_position.y
        global_tx[3, 2] = self.model_position.z

        model = self.transform_matrix()
        mv = self.view @ global_tx @ model

        ShaderLib.use("PBR")
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("M", global_tx @ model)
        ShaderLib.set_uniform("normal_matrix", Mat3.from_mat4(mv).inverse().transposed())
        ShaderLib.set_uniform("albedo", *self.colour)
        Primitives.draw(_VBO_NAMES[self.draw_index])

        if self.draw_normals:
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("MVP", self.project @ mv)
            ShaderLib.set_uniform("normalSize", self.normal_size / 10.0)
            Primitives.draw(_VBO_NAMES[self.draw_index])

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        draw_axis(self.view, self.project, global_tx)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.05, 450.0)


class MainWindow(QMainWindow):
    _ORDERS = [
        ("Rotate -> Translate -> Scale", "RTS"),
        ("Translate -> Rotate -> Scale", "TRS"),
        ("Translate -> Axis-Angle -> Scale", "TAxisS"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (OpenGL)")
        self.scene = Scene()
        gl_container = QWidget.createWindowContainer(self.scene, self)
        gl_container.setMinimumSize(600, 600)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addWidget(gl_container, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.resize(1200, 720)

    def _build_panel(self) -> QWidget:
        panel = QWidget(self)
        outer = QVBoxLayout(panel)

        self.primitive_combo = QComboBox()
        self.primitive_combo.addItems(_VBO_NAMES)
        self.primitive_combo.setCurrentIndex(self.scene.draw_index)
        self.primitive_combo.currentIndexChanged.connect(self._on_primitive_changed)
        outer.addWidget(QLabel("Primitive"))
        outer.addWidget(self.primitive_combo)

        self.order_combo = QComboBox()
        for label, _ in self._ORDERS:
            self.order_combo.addItem(label)
        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        outer.addWidget(QLabel("Matrix Order"))
        outer.addWidget(self.order_combo)

        self.translate_widget = Vec3Widget(panel, "Translate", Vec3(0, 0, 0))
        self.translate_widget.set_range(-20, 20)
        self.translate_widget.valueChanged.connect(self._on_translate_changed)
        outer.addWidget(self.translate_widget)

        self.rotate_widget = Vec3Widget(panel, "Rotate", Vec3(0, 0, 0))
        self.rotate_widget.set_range(-180, 180)
        self.rotate_widget.valueChanged.connect(self._on_rotate_changed)
        outer.addWidget(self.rotate_widget)

        self.scale_widget = Vec3Widget(panel, "Scale", Vec3(1, 1, 1))
        self.scale_widget.set_range(-20, 20)
        self.scale_widget.valueChanged.connect(self._on_scale_changed)
        outer.addWidget(self.scale_widget)

        axis_group = QGroupBox("Axis-Angle (used when order is Translate -> Axis-Angle -> Scale)")
        axis_layout = QVBoxLayout(axis_group)
        self.axis_widget = Vec3Widget(axis_group, "Axis", Vec3(1, 0, 0))
        self.axis_widget.valueChanged.connect(self._on_axis_changed)
        axis_layout.addWidget(self.axis_widget)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._on_angle_changed)
        axis_layout.addWidget(QLabel("Angle"))
        axis_layout.addWidget(self.angle_slider)
        outer.addWidget(axis_group)

        toggles = QHBoxLayout()
        self.wireframe_check = QCheckBox("Wireframe")
        self.wireframe_check.toggled.connect(self._on_wireframe_toggled)
        toggles.addWidget(self.wireframe_check)
        self.normals_check = QCheckBox("Normals")
        self.normals_check.toggled.connect(self._on_normals_toggled)
        toggles.addWidget(self.normals_check)
        outer.addLayout(toggles)

        self.normal_size_slider = QSlider(Qt.Horizontal)
        self.normal_size_slider.setRange(1, 20)
        self.normal_size_slider.setValue(6)
        self.normal_size_slider.valueChanged.connect(self._on_normal_size_changed)
        outer.addWidget(QLabel("Normal Size"))
        outer.addWidget(self.normal_size_slider)

        colour_button = QPushButton("Colour")
        colour_button.clicked.connect(self._on_colour_clicked)
        outer.addWidget(colour_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._on_reset_clicked)
        outer.addWidget(reset_button)

        self.matrix_widget = Mat4Widget(panel, "Transform Matrix", read_only=True)
        outer.addWidget(self.matrix_widget)

        outer.addStretch(1)
        return panel

    def _refresh_matrix_display(self) -> None:
        self.matrix_widget.set_value(self.scene.transform_matrix())
        self.scene.update()

    def _on_primitive_changed(self, index: int) -> None:
        self.scene.draw_index = index
        self.scene.update()

    def _on_order_changed(self, index: int) -> None:
        self.scene.order = self._ORDERS[index][1]
        self._refresh_matrix_display()

    def _on_translate_changed(self, value: Vec3) -> None:
        self.scene.translate_v = value
        self._refresh_matrix_display()

    def _on_rotate_changed(self, value: Vec3) -> None:
        self.scene.rotate_v = value
        self._refresh_matrix_display()

    def _on_scale_changed(self, value: Vec3) -> None:
        self.scene.scale_v = value
        self._refresh_matrix_display()

    def _on_axis_changed(self, value: Vec3) -> None:
        self.scene.axis_v = value
        self._refresh_matrix_display()

    def _on_angle_changed(self, value: int) -> None:
        self.scene.axis_angle = float(value)
        self._refresh_matrix_display()

    def _on_wireframe_toggled(self, checked: bool) -> None:
        self.scene.wireframe = checked
        self.scene.update()

    def _on_normals_toggled(self, checked: bool) -> None:
        self.scene.draw_normals = checked
        self.scene.update()

    def _on_normal_size_changed(self, value: int) -> None:
        self.scene.normal_size = float(value)
        self.scene.update()

    def _on_colour_clicked(self) -> None:
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.scene.colour = (colour.redF(), colour.greenF(), colour.blueF())
            self.scene.update()

    def _on_reset_clicked(self) -> None:
        self.translate_widget.set_value(Vec3(0, 0, 0))
        self.rotate_widget.set_value(Vec3(0, 0, 0))
        self.scale_widget.set_value(Vec3(1, 1, 1))
        self.angle_slider.setValue(0)
        self.wireframe_check.setChecked(False)
        self.normals_check.setChecked(False)
        self.scene.spin_x_face = 0
        self.scene.spin_y_face = 0
        self.scene.model_position.set(0, 0, 0)
        self._refresh_matrix_display()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()


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
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

Note on risk: this is the highest-risk task in this plan because it's the only one embedding a `QOpenGLWindow` inside a `QMainWindow` via `QWidget.createWindowContainer` — no existing OpenGL demo in this repo does this (`ShadingModels`/`MassSpring` embed a custom `QOpenGLWidget`-based scene class instead, not a windowed `QOpenGLWindow`). If `createWindowContainer` proves unreliable (rendering glitches, focus/input issues) when this is actually run, the fallback is to change `Scene`'s base class from `QOpenGLWindow` to `QOpenGLWidget` (drop the `PySideEventHandlingMixin`'s window-specific assumptions if any break — check by grepping `pyside_event_handling_mixin.py` for anything assuming `QOpenGLWindow` specifically, e.g. `self.makeCurrent()`/`self.update()` both exist on `QOpenGLWidget` too, so the mixin should work unchanged) and embed it directly as a normal child widget instead of through a window container.

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x AffineTransforms/main.py
cd AffineTransforms && QT_QPA_PLATFORM=offscreen uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. If `createWindowContainer` fails under the offscreen platform plugin specifically (rather than being a real bug), note that in the final report — check by also trying `QT_QPA_PLATFORM=minimal` as a second data point before concluding it's a real problem.

- [ ] **Step 7: Write README.md**

Create `AffineTransforms/README.md`:

```markdown
# AffineTransforms

Interactively compose a translate/rotate/scale transform and see how the
*order* of composition changes the result — Rotate-Translate-Scale,
Translate-Rotate-Scale, or Translate-(axis-angle rotation)-Scale — on a
PBR-shaded primitive next to an RGB axis gizmo. A read-only matrix grid
(`ncca.ngl.widgets.Mat4Widget`) shows the composed 4x4 matrix live.

Simplified from the NGL9Demos original: the hand-crafted "Gimbal Lock" raw
matrix-element demo and the redundant Translate-Euler-Scale variant are
dropped (see the design spec for why); axis-angle rotation replaces Euler
XYZ as the third mode, giving one built-in ("nested Euler rotations",
Rotate/Translate/Scale) and one hand-built ("does the axis stay put",
axis-angle) rotation to compare.

## Controls
- Primitive combo, Matrix Order combo, Translate/Rotate/Scale/Axis sliders, Angle slider
- Wireframe / Normals checkboxes, Normal Size slider (geometry-shader normal visualization)
- Colour button, Reset button
- Left-drag in the viewport orbits the camera; right-drag pans, wheel zooms

![AffineTransforms](AffineTransforms.png)
```

- [ ] **Step 8: Commit**

```bash
git add AffineTransforms/axis.py AffineTransforms/shaders/ AffineTransforms/main.py AffineTransforms/README.md
git commit -m "feat(affine-transforms): add OpenGL transform-order comparison demo"
```

---

## Task 8: AffineTransforms (WebGPU)

**Files:**
- Create: `AffineTransforms/AffineTransformsShader.wgsl`
- Create: `AffineTransforms/main_webgpu.py`
- Modify: `AffineTransforms/README.md` (add WebGPU note)

**Interfaces:** none consumed by later tasks.

**Scope cuts for this entry point (both documented in the design spec):**
- No axis gizmo (kept OpenGL-only — the WebGPU version would need its own small pipeline for 9 extra draw calls per frame for a secondary visual aid; the transform-order comparison itself doesn't need it).
- No geometry-shader normal visualization (WebGPU has no geometry-shader stage at all — this is the whole reason the spec scoped this feature as GL-only).
- Simpler diffuse shading in place of the OpenGL version's PBR material — porting a second full 3-light PBR WGSL pipeline is a large lift with no benefit to this demo's actual teaching point (transform order, not lighting); see `PBR/PBRTexture` for a full WebGPU PBR pipeline example elsewhere in this repo.
- Primitive selector is limited to the baked mesh set available via `PrimData.primitive` (`teapot`, `cube`, `troll`, `buddah`, `dragon`, `bunny`, `football`, `octahedron`, `dodecahedron`, `icosahedron`, `tetrahedron`) — no `sphere`/`cylinder`/`cone`/`disk`/`plane`/`torus`, which are GL-only runtime tessellations.

- [ ] **Step 1: Write the WGSL shader**

Create `AffineTransforms/AffineTransformsShader.wgsl`:

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
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = normalize((u.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(vec3<f32>(0.5, 1.0, 0.8));
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.15;
    return vec4<f32>(u.colour.rgb * (ambient + diffuse * 0.85), 1.0);
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `AffineTransforms/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""
AffineTransforms: interactively compose translate/rotate/scale matrices (WebGPU).

Same matrix-order comparison as the OpenGL version (main.py) — Rotate-
Translate-Scale, Translate-Rotate-Scale, or Translate-(axis-angle)-Scale —
using a simpler diffuse shader (no PBR, no geometry-shader normal
visualization: WebGPU has no geometry-shader stage) and a primitive
selector limited to the baked mesh set.

Controls: all on the panel; left-drag in the viewport orbits the camera.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Prims, Quaternion, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from ncca.ngl.widgets import Mat4Widget, Vec3Widget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from wgpu.utils import get_default_device

_MESH_NAMES = [
    "teapot", "cube", "troll", "buddah", "dragon", "bunny", "football",
    "octahedron", "dodecahedron", "icosahedron", "tetrahedron",
]

UNIFORM_DTYPE = np.dtype(
    [("mvp", np.float32, (4, 4)), ("normal_matrix", np.float32, (4, 4)), ("colour", np.float32, 4)]
)


class WebGPUScene(WebGPUWidget):
    _ORDERS = [
        ("Rotate -> Translate -> Scale", "RTS"),
        ("Translate -> Rotate -> Scale", "TRS"),
        ("Translate -> Axis-Angle -> Scale", "TAxisS"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (WebGPU)")
        self.msaa_sample_count = 4

        self.mouse_global_tx = Mat4()
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.mesh_index = 0
        self.order = "RTS"
        self.translate_v = Vec3(0, 0, 0)
        self.rotate_v = Vec3(0, 0, 0)
        self.scale_v = Vec3(1, 1, 1)
        self.axis_angle = 0.0
        self.axis_v = Vec3(1, 0, 0)
        self.colour = (0.95, 0.71, 0.29)

        self.view = look_at(Vec3(0, 0, 8), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, self.width() / self.height(), 0.05, 450.0, PerspMode.WebGPU)

        self.device = get_default_device()
        self._create_pipeline()
        self._load_meshes()
        self._create_render_buffer()

    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "AffineTransformsShader.wgsl").read_text()
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
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
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
        self.uniforms = np.zeros((), dtype=UNIFORM_DTYPE)
        self.uniform_buffer = self.device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
        )

    def _load_meshes(self) -> None:
        self.meshes = {}
        for name in _MESH_NAMES:
            data = PrimData.primitive(name)
            buf = self.device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.VERTEX)
            self.meshes[name] = (buf, data.size // 8)

    def transform_matrix(self) -> Mat4:
        t = Mat4().translate(self.translate_v.x, self.translate_v.y, self.translate_v.z)
        s = Mat4().scale(self.scale_v.x, self.scale_v.y, self.scale_v.z)
        if self.order in ("RTS", "TRS"):
            r = (
                Mat4().rotate_z(self.rotate_v.z)
                @ Mat4().rotate_y(self.rotate_v.y)
                @ Mat4().rotate_x(self.rotate_v.x)
            )
            return (r @ t @ s) if self.order == "RTS" else (t @ r @ s)
        r = Quaternion.from_axis_angle(self.axis_v, self.axis_angle).to_mat4()
        return t @ r @ s

    def paintWebGPU(self) -> None:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

        model = self.transform_matrix()
        mv = self.view @ self.mouse_global_tx @ model
        self.uniforms["mvp"] = (self.project @ mv).to_numpy()
        self.uniforms["normal_matrix"] = mv.inverse().transposed().to_numpy()
        self.uniforms["colour"] = (*self.colour, 1.0)
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        buf, count = self.meshes[_MESH_NAMES[self.mesh_index]]
        render_pass.set_vertex_buffer(0, buf)
        render_pass.draw(count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(45.0, width / height, 0.05, 450.0, PerspMode.WebGPU)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        self.update()
        super().keyPressEvent(event)

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AffineTransforms (WebGPU)")
        self.scene = WebGPUScene()

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addWidget(self.scene, 1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.resize(1200, 720)

    def _build_panel(self) -> QWidget:
        panel = QWidget(self)
        outer = QVBoxLayout(panel)

        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems(_MESH_NAMES)
        self.mesh_combo.currentIndexChanged.connect(self._on_mesh_changed)
        outer.addWidget(QLabel("Mesh"))
        outer.addWidget(self.mesh_combo)

        self.order_combo = QComboBox()
        for label, _ in WebGPUScene._ORDERS:
            self.order_combo.addItem(label)
        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        outer.addWidget(QLabel("Matrix Order"))
        outer.addWidget(self.order_combo)

        self.translate_widget = Vec3Widget(panel, "Translate", Vec3(0, 0, 0))
        self.translate_widget.set_range(-20, 20)
        self.translate_widget.valueChanged.connect(self._on_translate_changed)
        outer.addWidget(self.translate_widget)

        self.rotate_widget = Vec3Widget(panel, "Rotate", Vec3(0, 0, 0))
        self.rotate_widget.set_range(-180, 180)
        self.rotate_widget.valueChanged.connect(self._on_rotate_changed)
        outer.addWidget(self.rotate_widget)

        self.scale_widget = Vec3Widget(panel, "Scale", Vec3(1, 1, 1))
        self.scale_widget.set_range(-20, 20)
        self.scale_widget.valueChanged.connect(self._on_scale_changed)
        outer.addWidget(self.scale_widget)

        axis_group = QGroupBox("Axis-Angle (used when order is Translate -> Axis-Angle -> Scale)")
        axis_layout = QVBoxLayout(axis_group)
        self.axis_widget = Vec3Widget(axis_group, "Axis", Vec3(1, 0, 0))
        self.axis_widget.valueChanged.connect(self._on_axis_changed)
        axis_layout.addWidget(self.axis_widget)
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self._on_angle_changed)
        axis_layout.addWidget(QLabel("Angle"))
        axis_layout.addWidget(self.angle_slider)
        outer.addWidget(axis_group)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._on_reset_clicked)
        outer.addWidget(reset_button)

        self.matrix_widget = Mat4Widget(panel, "Transform Matrix", read_only=True)
        outer.addWidget(self.matrix_widget)

        outer.addStretch(1)
        return panel

    def _refresh_matrix_display(self) -> None:
        self.matrix_widget.set_value(self.scene.transform_matrix())
        self.scene.update()

    def _on_mesh_changed(self, index: int) -> None:
        self.scene.mesh_index = index
        self.scene.update()

    def _on_order_changed(self, index: int) -> None:
        self.scene.order = WebGPUScene._ORDERS[index][1]
        self._refresh_matrix_display()

    def _on_translate_changed(self, value: Vec3) -> None:
        self.scene.translate_v = value
        self._refresh_matrix_display()

    def _on_rotate_changed(self, value: Vec3) -> None:
        self.scene.rotate_v = value
        self._refresh_matrix_display()

    def _on_scale_changed(self, value: Vec3) -> None:
        self.scene.scale_v = value
        self._refresh_matrix_display()

    def _on_axis_changed(self, value: Vec3) -> None:
        self.scene.axis_v = value
        self._refresh_matrix_display()

    def _on_angle_changed(self, value: int) -> None:
        self.scene.axis_angle = float(value)
        self._refresh_matrix_display()

    def _on_reset_clicked(self) -> None:
        self.translate_widget.set_value(Vec3(0, 0, 0))
        self.rotate_widget.set_value(Vec3(0, 0, 0))
        self.scale_widget.set_value(Vec3(1, 1, 1))
        self.angle_slider.setValue(0)
        self.scene.spin_x_face = 0
        self.scene.spin_y_face = 0
        self.scene.model_position.set(0, 0, 0)
        self._refresh_matrix_display()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    format: QSurfaceFormat = QSurfaceFormat()
    QSurfaceFormat.setDefaultFormat(format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x AffineTransforms/main_webgpu.py
cd AffineTransforms && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `AffineTransforms/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` compares the same three matrix orders using a simpler
diffuse shader (no PBR) and a primitive selector limited to the baked mesh
set (`PrimData.primitive` has no sphere/cylinder/cone/disk/plane/torus data
— those are GL-only runtime tessellations). It omits the axis gizmo and the
geometry-shader normal visualization; WebGPU has no geometry-shader stage
at all, which is why that feature is GL-only in the first place.
```

- [ ] **Step 5: Commit**

```bash
git add AffineTransforms/AffineTransformsShader.wgsl AffineTransforms/main_webgpu.py AffineTransforms/README.md
git commit -m "feat(affine-transforms): add WebGPU entry point"
```

---

## Final steps (after all 8 tasks)

- [ ] **Add root README.md entries**

Add a row for each of the 4 demos to the root `README.md`, under whichever
existing section fits best (likely alongside `Camera`/`FrustumCull` in a
"transforms/camera" grouping) — follow the existing row format exactly
(name, link, thumbnail).

- [ ] **Run full verification**

```bash
uv run ruff check MatrixStack/ LookAtDemos/ ViewToWorldTransform/ AffineTransforms/
uv run ruff format --check MatrixStack/ LookAtDemos/ ViewToWorldTransform/ AffineTransforms/
uv run pytest ViewToWorldTransform/tests/ -v
```
Expected: ruff clean, all tests pass.

- [ ] **Report to Jon**

List the 4 `.png` screenshots that still need capturing (`MatrixStack.png`,
`LookAtDemos.png`, `ViewToWorldTransform.png`, `AffineTransforms.png`) and
flag the two known risk areas from Task 7/8 (whether `createWindowContainer`
behaves correctly, and the intentional GIMBALLOCK/TEULERS/axis-gizmo/PBR
scope cuts) for Jon to sanity-check by actually running each demo.
