# Core Demos Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 2 NGL9Demos (Spotlight, ShadedGrid) to PyNGLDemos. Spotlight ships with both an OpenGL and a WebGPU entry point; ShadedGrid is OpenGL-only because it depends on a geometry-shader stage, which WebGPU has no equivalent of.

**Architecture:** Each demo is a self-contained top-level folder. OpenGL entry points are `QOpenGLWindow` subclasses using `PySideEventHandlingMixin` for the standard orbit/pan/zoom controls; the Spotlight WebGPU entry point subclasses `ncca.ngl.webgpu.WebGPUWidget`. Both demos deliberately use **world-space lighting** (light positions/directions and the normal matrix derived from the model matrix alone, never model-view) — the NGL9Demos C++ originals for both compute lighting in a mix of world- and eye-space that is internally inconsistent (documented per-task below); this repo's own PBR family was already fixed to a consistent world-space convention in commit `8b77482`, and Phase 1 repeatedly ran into bugs from copying C++ space-inconsistencies verbatim, so this plan does not repeat that mistake.

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, `uv run --script`, numpy.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md`

## Global Constraints

- Work happens in branch `agent/core-demos-phase2`, worktree at `.worktrees/core-demos-phase2` (create with `git worktree add .worktrees/core-demos-phase2 -b agent/core-demos-phase2` before starting Task 1 — not yet created).
- No edits to `/Users/jmacey/teaching/Code/PyNGL` — every demo is self-contained in its own PyNGLDemos folder.
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) which runs one paint pass via `QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))` then exits 0 — copy the pattern verbatim from `VAOPrimitives/main.py`'s `__main__` block (OpenGL) or `Blending/BlendingWebGPU.py`'s `main()` (WebGPU).
- OpenGL entry points: `class MainWindow(PySideEventHandlingMixin, QOpenGLWindow)` (or `Scene`, matching the demo's own class-naming precedent — Phase 1 used both), calling `self.setup_event_handling(rotation_sensitivity=0.5, translation_sensitivity=0.01, zoom_sensitivity=0.1, initial_position=Vec3(0,0,0))` in `__init__`. The mixin already provides Escape/W/S/Space key handling and full mouse orbit/pan/zoom — demo-specific `keyPressEvent` overrides only need to handle their own extra keys and end with `super().keyPressEvent(event)`. **Do not re-implement W/S wireframe toggling** — the mixin already owns it (a Phase 1 review flagged a demo that redundantly reset polygon mode every frame, making its own W/S handling inert; don't repeat that). GL 4.1 core profile via the standard `QSurfaceFormat` block (copy from `VAOPrimitives/main.py`'s `__main__`).
- WebGPU entry points: `class WebGPUScene(WebGPUWidget)` importing `from ncca.ngl.webgpu import WebGPUWidget` directly — do not copy a `WebGPUWidget.py` file into the demo folder, that class lives in the `ncca.ngl.webgpu` package itself. Set `self.msaa_sample_count = 4`, call `get_default_device()` from `wgpu.utils`, build pipelines/scene, then `self._create_render_buffer()`. Mouse/keyboard handlers are hand-copied (no mixin for `QWidget`) — copy the block from `Blending/BlendingWebGPU.py`.
- **WebGPU multi-object rendering MUST use a per-draw uniform-buffer pool, never one shared buffer rewritten per draw.** Phase 1 shipped a Critical bug (`MatrixStack/main_webgpu.py`, fixed in commit `6bfa0ef`) where a single uniform buffer, rewritten before each of ~130 draws inside one render pass, meant every draw observed only the *last* write (WebGPU queue-timeline ordering) — objects silently rendered on top of each other. Any task in this plan that draws more than one object per frame in WebGPU must pre-allocate one uniform buffer + bind group per draw slot (sized to the max draws/frame) and index into the pool by a per-frame draw counter reset at the top of `paintWebGPU`. See `MatrixStack/main_webgpu.py`'s `_create_draw_buffer_pool`/`_draw_current` for the exact, already-working pattern to copy.
- Maths convention: numpy/PyNGL row-vector convention — points transform as `row_vec @ M`, translation lives in row 3 (`mat[3, 0..2]`). Matrix composition order reads the same as the C++ source (`A @ B @ C` applies exactly like `A * B * C` did in NGL9Demos) for any matrix chain that IS ported faithfully — but see the per-task notes below for where this plan deliberately does NOT port the source's lighting-space handling.
- `ruff check` and `ruff format --check` must pass.
- README.md per demo (description, controls table, teaching points, `![](<Demo>.png)` image reference); add a row to the root `README.md` under an appropriate existing section (`## Lighting & Shadows` for Spotlight, `## Geometry & Tessellation Shaders` for ShadedGrid — both sections already exist, confirmed present in the current README). Screenshots cannot be captured by an agent — list the missing `.png` in the final report for Jon.
- One commit per task: `git add <files> && git commit -m "feat(<demo>): <what>"`.
- **Smoketest verification differs by backend on this machine** (discovered during Phase 1): `QT_QPA_PLATFORM=offscreen` segfaults for every `QOpenGLWindow` demo here ("This plugin does not support createPlatformOpenGLContext!") — OpenGL entry points (`main.py`) must be smoke-tested WITHOUT that env var (`uv run --script main.py --smoketest`, real display, briefly flashes a window, auto-quits). WebGPU entry points (`main_webgpu.py`) work fine WITH `QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest`.

---

## Task 1: Spotlight (OpenGL)

**Files:**
- Create: `Spotlight/main.py`
- Create: `Spotlight/shaders/SpotlightVertex.glsl`
- Create: `Spotlight/shaders/SpotlightFragment.glsl`
- Create: `Spotlight/README.md`

**Interfaces:**
- Produces: nothing consumed by Task 3 (ShadedGrid is an independent demo with its own Phong shader). Task 2 (Spotlight WebGPU) is also independent — it does not import this task's Python code, only mirrors the same visual design and light-parameter values in WGSL, per the design note below.

**Design notes:**

The NGL9Demos C++ source (`Spotlight/src/NGLScene.cpp`) has two bugs this port deliberately does not reproduce:
1. **Space inconsistency**: light positions/directions are computed via `pos * iv` where `iv` is a never-initialized (default-identity) `Mat4` meant to be an inverse-view matrix — so despite the comment claiming eye-space, lights are actually left in world space, then compared against `vPosition` which IS in eye space (`MV * vertex`) in the fragment shader. This plan uses **world-space lighting throughout** instead: light positions/directions stay in world space, the fragment shader compares them against a world-space vertex position, and the normal matrix comes from the model matrix alone.
2. **Copy-paste uniform bug**: `loadSpotlightToShader` sets `light[i].diffuse` and `light[i].specular` both from `_l.ambient` (not `_l.diffuse`/`_l.specular`) — this plan sets each field from its own source value.

Scene: a `TRIANGLE_PLANE` ground plane plus a 5×5 grid of teapots (25 total, spaced 3 units apart, `x, z` in `range(-6, 7, 3)`), lit entirely by 4 animated spotlights. Each light orbits a fixed centre point above the grid at its own radius/phase (so the 4 cones sweep independently), always aiming straight down (`direction = (0, -1, 0)`) — this is a deliberate simplification of the C++'s more elaborate per-light "aim point" wobble, which added complexity without much extra teaching value (the point of this demo is cone attenuation, not light-path choreography). Each light has a distinct, fixed diffuse/specular colour (red, green, blue, white) so the 4 cones are visually distinguishable. Cone parameters (`spotCosCutoff`, `spotCosInnerCutoff`, `spotExponent`) and attenuation constants are set once at init and never change.

- [ ] **Step 1: Write the vertex shader**

Create `Spotlight/shaders/SpotlightVertex.glsl`:

```glsl
#version 410 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

out vec3 fragmentNormal;
out vec3 worldPosition;

uniform mat4 M;
uniform mat4 MVP;
uniform mat3 normal_matrix;

void main()
{
    fragmentNormal = normalize(normal_matrix * inNormal);
    worldPosition = vec3(M * vec4(inVert, 1.0));
    gl_Position = MVP * vec4(inVert, 1.0);
}
```

Note: the uniform is named `normal_matrix` (snake_case), matching this repo's established convention (confirmed in `Camera/shaders/PBRVertex.glsl` and `AffineTransforms/shaders/PBRVertex.glsl`), not the NGL9Demos C++ source's `normalMatrix` — the Python-side `ShaderLib.set_uniform("normal_matrix", ...)` call in Step 3 depends on this exact name matching.

- [ ] **Step 2: Write the fragment shader**

Create `Spotlight/shaders/SpotlightFragment.glsl`:

```glsl
#version 410 core
in vec3 fragmentNormal;
in vec3 worldPosition;
layout(location = 0) out vec4 fragColour;

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};

struct SpotLight
{
    vec3 position;
    vec3 direction;
    vec4 diffuse;
    vec4 specular;
    float spotCosCutoff;
    float spotCosInnerCutoff;
    float spotExponent;
    float constantAttenuation;
    float linearAttenuation;
    float quadraticAttenuation;
};

uniform Material material;
#define NUM_LIGHTS 4
uniform SpotLight light[NUM_LIGHTS];
uniform vec3 viewerPos;

vec4 spotLight(int i, vec3 n, vec3 eyeDir)
{
    vec3 toLight = light[i].position - worldPosition;
    float d = length(toLight);
    vec3 L = toLight / d;

    float attenuation = 1.0 / (light[i].constantAttenuation +
                                light[i].linearAttenuation * d +
                                light[i].quadraticAttenuation * d * d);

    float spotDot = dot(-L, normalize(light[i].direction));
    float spotAttenuation;
    if (spotDot < light[i].spotCosCutoff)
    {
        spotAttenuation = 0.0;
    }
    else
    {
        float spotValue = smoothstep(light[i].spotCosCutoff, light[i].spotCosInnerCutoff, spotDot);
        spotAttenuation = pow(spotValue, light[i].spotExponent);
    }
    attenuation *= spotAttenuation;

    vec3 reflection = normalize(reflect(-L, n));
    float nDotL = max(0.0, dot(n, L));
    float nDotR = max(0.0, dot(eyeDir, reflection));

    vec4 diffuse = material.diffuse * light[i].diffuse * nDotL * attenuation;
    vec4 specular = material.specular * light[i].specular * pow(nDotR, material.shininess) * attenuation;
    return diffuse + specular;
}

void main()
{
    vec3 n = normalize(fragmentNormal);
    vec3 eyeDir = normalize(viewerPos - worldPosition);

    fragColour = material.ambient * 0.2;
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        fragColour += spotLight(i, n, eyeDir);
    }
    fragColour.a = 1.0;
}
```

Note: `NUM_LIGHTS` is 4, not the C++'s 8 — half as many orbiting cones is plenty to teach cone attenuation and keeps the fragment-shader loop and the Python-side per-frame uniform updates lighter.

- [ ] **Step 3: Write main.py**

Create `Spotlight/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""Spotlight demo: 4 animated cone-attenuation spotlights over a grid of teapots."""

import argparse
import math
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import Primitives, PySideEventHandlingMixin, ShaderLib
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

_NUM_LIGHTS = 4
_LIGHT_COLOURS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.3, 0.4, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
]
_LIGHT_CENTRES = [(-4.0, -4.0), (4.0, -4.0), (-4.0, 4.0), (4.0, 4.0)]
_LIGHT_RADII = [2.5, 3.0, 2.0, 3.5]
_LIGHT_HEIGHT = 6.0


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
        self.setTitle("Spotlight")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.animate: bool = True
        self.time: float = 0.0
        self.grid_positions: list[tuple[float, float]] = [
            (float(x), float(z)) for x in range(-6, 7, 3) for z in range(-6, 7, 3)
        ]

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.15, 0.15, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 8, 16), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Spotlight",
            "shaders/SpotlightVertex.glsl",
            "shaders/SpotlightFragment.glsl",
        )
        ShaderLib.use("Spotlight")
        ShaderLib.set_uniform("material.ambient", 0.274725, 0.1995, 0.0745, 1.0)
        ShaderLib.set_uniform("material.diffuse", 0.75164, 0.60648, 0.22648, 1.0)
        ShaderLib.set_uniform("material.specular", 0.628281, 0.555802, 0.3666065, 1.0)
        ShaderLib.set_uniform("material.shininess", 51.2)

        for i in range(_NUM_LIGHTS):
            r, g, b, a = _LIGHT_COLOURS[i]
            ShaderLib.set_uniform(f"light[{i}].diffuse", r, g, b, a)
            ShaderLib.set_uniform(f"light[{i}].specular", r, g, b, a)
            ShaderLib.set_uniform(f"light[{i}].direction", 0.0, -1.0, 0.0)
            ShaderLib.set_uniform(
                f"light[{i}].spotCosCutoff", math.cos(math.radians(25.0))
            )
            ShaderLib.set_uniform(
                f"light[{i}].spotCosInnerCutoff", math.cos(math.radians(15.0))
            )
            ShaderLib.set_uniform(f"light[{i}].spotExponent", 8.0)
            ShaderLib.set_uniform(f"light[{i}].constantAttenuation", 1.0)
            ShaderLib.set_uniform(f"light[{i}].linearAttenuation", 0.02)
            ShaderLib.set_uniform(f"light[{i}].quadraticAttenuation", 0.005)

        Primitives.load_default_primitives()
        Primitives.create(Prims.TRIANGLE_PLANE, "ground", 30, 30, 20, 20, Vec3(0, 1, 0))

        self._update_lights()
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _update_lights(self) -> None:
        ShaderLib.use("Spotlight")
        for i in range(_NUM_LIGHTS):
            cx, cz = _LIGHT_CENTRES[i]
            radius = _LIGHT_RADII[i]
            phase = self.time + i * 1.7
            x = cx + math.cos(phase) * radius
            z = cz + math.sin(phase) * radius
            ShaderLib.set_uniform(f"light[{i}].position", x, _LIGHT_HEIGHT, z)

    def _on_tick(self) -> None:
        if self.animate:
            self.time += 0.03
            self._update_lights()
            self.update()

    def load_matrices(self, global_tx: Mat4, tx: Transform) -> None:
        m = global_tx @ tx.matrix()
        mvp = self.project @ self.view @ m
        normal_matrix = Mat3.from_mat4(m).inverse().transposed()
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("viewerPos", 0.0, 8.0, 16.0)

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

        ShaderLib.use("Spotlight")
        tx = Transform()
        self.load_matrices(global_tx, tx)
        Primitives.draw("ground")

        for x, z in self.grid_positions:
            tx.reset()
            tx.set_position(x, 0.5, z)
            self.load_matrices(global_tx, tx)
            Primitives.draw("teapot")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 150.0)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
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

Note: `Transform.matrix()`/`.set_position()`/`.reset()`, `Mat3.from_mat4(...).inverse().transposed()`, `Mat4().rotate_x/rotate_y`, and `ShaderLib.set_uniform` with struct-array-style string keys (`f"light[{i}].position"`) are all confirmed-working patterns already used by multiple merged Phase 1 demos (`Camera/main.py`, `AffineTransforms/main.py`) — no further API verification needed before using them here.

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x Spotlight/main.py
cd Spotlight && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. (Run WITHOUT `QT_QPA_PLATFORM=offscreen` — that segfaults for `QOpenGLWindow` demos on this machine, per Global Constraints.)

- [ ] **Step 5: Write README.md**

Create `Spotlight/README.md`:

```markdown
# Spotlight

Four animated spotlights sweep cones of light across a grid of teapots, each
light showing cone-attenuation falloff (a smoothstep ramp between an inner
and outer cutoff angle, raised to a spot exponent) alongside normal
distance attenuation. Lighting is computed in world space.

## Controls
- `a` : toggle light animation
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 6: Commit**

```bash
git add Spotlight/
git commit -m "feat(spotlight): add OpenGL cone-attenuation spotlight demo"
```

---

## Task 2: Spotlight (WebGPU)

**Files:**
- Create: `Spotlight/main_webgpu.py`
- Create: `Spotlight/SpotlightShader.wgsl`

**Interfaces:** none consumed by later tasks. Independent of Task 1 — does not import `main.py`, mirrors the same visual design (4 orbiting spotlights, same colours/centres/radii/cone angles, world-space lighting) and mirrors Task 1's README with an appended section rather than duplicating it.

**Design notes:**

WebGPU has no runtime primitive generator here — geometry comes from the baked `PrimData.primitive("teapot")` mesh, and the ground plane is a small hand-built numpy quad (same helper pattern as `Blending/BlendingWebGPU.py`'s `quad()` / `MatrixStack/main_webgpu.py`'s `quad_floor()`).

**This task draws 26 objects per frame (25 teapots + 1 ground plane) inside one render pass — it MUST use the per-draw uniform-buffer-pool pattern from `MatrixStack/main_webgpu.py`'s `_create_draw_buffer_pool`/`_draw_current`, not a single shared buffer.** Read that file directly before writing this one; the pool size here is `26` (vs `MatrixStack`'s `130`), and the per-draw uniform struct is different (this demo needs `M`, `MVP`, `normal_matrix`, `viewerPos` rather than `MatrixStack`'s `MVP`/`normal_matrix`), but the buffer-pool *mechanism* — pre-allocate N buffers + bind groups at init, index by a per-frame counter reset in `paintWebGPU`, write only the current draw's slot before each `draw()` call — is identical and must be copied, not reinvented.

- [ ] **Step 1: Write the WGSL shader**

Create `Spotlight/SpotlightShader.wgsl`:

```wgsl
struct Uniforms {
    m: mat4x4<f32>,
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    viewer_pos: vec4<f32>,
};

struct SpotLight {
    position: vec4<f32>,
    direction: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    params: vec4<f32>,      // spotCosCutoff, spotCosInnerCutoff, spotExponent, unused
    attenuation: vec4<f32>, // constant, linear, quadratic, unused
};

struct Lights {
    lights: array<SpotLight, 4>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> lights: Lights;

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

const material_ambient = vec4<f32>(0.274725, 0.1995, 0.0745, 1.0);
const material_diffuse = vec4<f32>(0.75164, 0.60648, 0.22648, 1.0);
const material_specular = vec4<f32>(0.628281, 0.555802, 0.3666065, 1.0);
const material_shininess = 51.2;

fn spot_light(i: u32, n: vec3<f32>, world_pos: vec3<f32>, eye_dir: vec3<f32>) -> vec4<f32> {
    let light = lights.lights[i];
    let to_light = light.position.xyz - world_pos;
    let d = length(to_light);
    let l = to_light / d;

    let attenuation = 1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d);

    let spot_dot = dot(-l, normalize(light.direction.xyz));
    var spot_attenuation = 0.0;
    if (spot_dot >= light.params.x) {
        let spot_value = smoothstep(light.params.x, light.params.y, spot_dot);
        spot_attenuation = pow(spot_value, light.params.z);
    }
    let total_attenuation = attenuation * spot_attenuation;

    let reflection = normalize(reflect(-l, n));
    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_r = max(0.0, dot(eye_dir, reflection));

    let diffuse = material_diffuse * light.diffuse * n_dot_l * total_attenuation;
    let specular = material_specular * light.specular * pow(n_dot_r, material_shininess) * total_attenuation;
    return diffuse + specular;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let eye_dir = normalize(u.viewer_pos.xyz - in.world_position);

    var colour = material_ambient * 0.2;
    for (var i = 0u; i < 4u; i = i + 1u) {
        colour = colour + spot_light(i, n, in.world_position, eye_dir);
    }
    colour.a = 1.0;
    return colour;
}
```

- [ ] **Step 2: Write main_webgpu.py**

Create `Spotlight/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""Spotlight demo: 4 animated cone-attenuation spotlights over a grid of teapots (WebGPU)."""

import argparse
import math
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
from wgpu.utils import get_default_device

_NUM_LIGHTS = 4
_LIGHT_COLOURS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.3, 0.4, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
]
_LIGHT_CENTRES = [(-4.0, -4.0), (4.0, -4.0), (-4.0, 4.0), (4.0, 4.0)]
_LIGHT_RADII = [2.5, 3.0, 2.0, 3.5]
_LIGHT_HEIGHT = 6.0
_DRAW_POOL_SIZE = 26  # 25 teapots + 1 ground plane


def quad_floor(size: float) -> np.ndarray:
    """A flat, upward-facing quad, interleaved x,y,z,nx,ny,nz,u,v float32."""
    h = size / 2.0
    verts = [
        (-h, 0.0, -h, 0.0, 1.0, 0.0, 0.0, 0.0),
        (h, 0.0, -h, 0.0, 1.0, 0.0, 1.0, 0.0),
        (h, 0.0, h, 0.0, 1.0, 0.0, 1.0, 1.0),
        (-h, 0.0, -h, 0.0, 1.0, 0.0, 0.0, 0.0),
        (h, 0.0, h, 0.0, 1.0, 0.0, 1.0, 1.0),
        (-h, 0.0, h, 0.0, 1.0, 0.0, 0.0, 1.0),
    ]
    return np.array(verts, dtype=np.float32).flatten()


class WebGPUScene(WebGPUWidget):
    def __init__(self) -> None:
        super().__init__()
        self.msaa_sample_count = 4
        self.setTitle("Spotlight (WebGPU)")
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.rotate = False
        self.translate = False
        self.original_x = 0
        self.original_y = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.model_position = Vec3(0, 0, 0)
        self.animate = True
        self.time = 0.0
        self.grid_positions = [
            (float(x), float(z)) for x in range(-6, 7, 3) for z in range(-6, 7, 3)
        ]
        self.view = look_at(Vec3(0, 8, 16), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, 1024.0 / 720.0, 0.5, 150.0)

    def initializeWebGPU(self) -> None:
        self.device = get_default_device()
        shader_src = (Path(__file__).parent / "SpotlightShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        self.bind_group_layout = bind_group_layout

        vertex_layout = {
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
            ],
        }
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_layout],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": self.render_texture_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_buffer = self.device.create_buffer_with_data(
            data=teapot.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.teapot_count = teapot.size // 8

        ground = quad_floor(30.0)
        self.ground_buffer = self.device.create_buffer_with_data(
            data=ground.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )
        self.ground_count = ground.size // 8

        # 208-byte Uniforms struct (3 mat4x4 @ 64 bytes each + 1 vec4 @ 16 bytes), std140-aligned.
        self._uniform_size = 4 * 4 * 4 * 3 + 16
        self.pane_uniform_buffers = []
        self.pane_bind_groups = []
        for _ in range(_DRAW_POOL_SIZE):
            buf = self.device.create_buffer(
                size=self._uniform_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self.pane_uniform_buffers.append(buf)

        light_dtype = np.dtype(
            [
                ("position", np.float32, 4),
                ("direction", np.float32, 4),
                ("diffuse", np.float32, 4),
                ("specular", np.float32, 4),
                ("params", np.float32, 4),
                ("attenuation", np.float32, 4),
            ]
        )
        self.light_data = np.zeros(_NUM_LIGHTS, dtype=light_dtype)
        for i in range(_NUM_LIGHTS):
            r, g, b, a = _LIGHT_COLOURS[i]
            self.light_data[i]["direction"] = (0.0, -1.0, 0.0, 0.0)
            self.light_data[i]["diffuse"] = (r, g, b, a)
            self.light_data[i]["specular"] = (r, g, b, a)
            self.light_data[i]["params"] = (
                math.cos(math.radians(25.0)),
                math.cos(math.radians(15.0)),
                8.0,
                0.0,
            )
            self.light_data[i]["attenuation"] = (1.0, 0.02, 0.005, 0.0)
        self.light_buffer = self.device.create_buffer(
            size=self.light_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        for buf in self.pane_uniform_buffers:
            self.pane_bind_groups.append(
                self.device.create_bind_group(
                    layout=self.bind_group_layout,
                    entries=[
                        {
                            "binding": 0,
                            "resource": {
                                "buffer": buf,
                                "offset": 0,
                                "size": self._uniform_size,
                            },
                        },
                        {
                            "binding": 1,
                            "resource": {
                                "buffer": self.light_buffer,
                                "offset": 0,
                                "size": self.light_data.nbytes,
                            },
                        },
                    ],
                )
            )

        self._update_lights()
        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _update_lights(self) -> None:
        for i in range(_NUM_LIGHTS):
            cx, cz = _LIGHT_CENTRES[i]
            radius = _LIGHT_RADII[i]
            phase = self.time + i * 1.7
            x = cx + math.cos(phase) * radius
            z = cz + math.sin(phase) * radius
            self.light_data[i]["position"] = (x, _LIGHT_HEIGHT, z, 1.0)
        self.device.queue.write_buffer(self.light_buffer, 0, self.light_data.tobytes())

    def _on_tick(self) -> None:
        if self.animate:
            self.time += 0.03
            self._update_lights()
            self.update()

    def _global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _draw_object(
        self, render_pass, draw_index: int, model: Mat4, vertex_buffer, count: int
    ) -> None:
        mvp = self.project @ self.view @ model
        normal_matrix = model.inverse().transposed()
        data = np.zeros(self._uniform_size // 4, dtype=np.float32)
        data[0:16] = model.to_numpy().flatten()
        data[16:32] = mvp.to_numpy().flatten()
        data[32:48] = normal_matrix.to_numpy().flatten()
        data[48:51] = (0.0, 8.0, 16.0)
        self.device.queue.write_buffer(
            self.pane_uniform_buffers[draw_index], 0, data.tobytes()
        )
        render_pass.set_bind_group(0, self.pane_bind_groups[draw_index], [], 0, 999999)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.draw(count)

    def paintWebGPU(self) -> None:
        if not hasattr(self, "device"):
            return
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
                "view": self.depth_buffer_texture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.pipeline)

        global_tx = self._global_tx()
        draw_index = 0
        self._draw_object(
            render_pass, draw_index, global_tx, self.ground_buffer, self.ground_count
        )
        draw_index += 1
        for x, z in self.grid_positions:
            model = global_tx @ Mat4().translate(x, 0.5, z)
            self._draw_object(
                render_pass, draw_index, model, self.teapot_buffer, self.teapot_count
            )
            draw_index += 1

        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

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
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position = Vec3(0, 0, 0)
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

Note: `PerspMode`/`QColor` imports are unused in this listing — confirm during implementation whether the `perspective()` call needs a `PerspMode.WebGPU` argument (it did in several Phase 1 WebGPU demos, e.g. `LookAtDemos/main_webgpu.py`); if so add it to the `perspective(...)` call and keep the `PerspMode` import, otherwise drop the unused import per `ruff`.

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x Spotlight/main_webgpu.py
cd Spotlight && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. Additionally verify (mirroring how Phase 1's `MatrixStack`/`LookAtDemos` WebGPU fixes were verified) that the 26 draws are NOT all landing on the same transform — read back the rendered frame buffer and confirm the teapots at different grid positions occupy visually distinct screen regions, not one stacked blob.

- [ ] **Step 4: Add the WebGPU note to README.md**

Append to `Spotlight/README.md`:

```markdown
## WebGPU version

`main_webgpu.py` renders the identical scene and lighting model (same 4
orbiting spotlights, same colours/cone angles) using a WGSL fragment
shader. Each of the 26 draws (25 teapots + the ground plane) gets its own
uniform buffer from a pre-allocated pool, avoiding the WebGPU
queue-timeline aliasing bug a shared buffer would cause.
```

- [ ] **Step 5: Commit**

```bash
git add Spotlight/main_webgpu.py Spotlight/SpotlightShader.wgsl Spotlight/README.md
git commit -m "feat(spotlight): add WebGPU entry point"
```

---

## Task 3: ShadedGrid (OpenGL)

**Files:**
- Create: `ShadedGrid/main.py`
- Create: `ShadedGrid/shaders/PhongVertex.glsl`
- Create: `ShadedGrid/shaders/PhongFragment.glsl`
- Create: `ShadedGrid/shaders/normalVertex.glsl` (copy)
- Create: `ShadedGrid/shaders/normalFragment.glsl` (copy)
- Create: `ShadedGrid/shaders/normalGeo.glsl` (copy)
- Create: `ShadedGrid/README.md`

**Interfaces:** none consumed by other tasks; independent demo. WebGPU has no geometry-shader stage at all, so there is no WebGPU entry point for this demo — this is the reason the design spec scoped it OpenGL-only.

**Design notes:**

This demo has two parts: an undulating wave-grid mesh, Phong-shaded with 3-point lighting, plus a geometry-shader pass drawn on top that visualizes each triangle's face normal and each vertex's normal as coloured line segments — the actual teaching point of the demo (watching normals rotate/stretch as the surface animates).

**Reuse, don't re-derive, the normal-visualization shaders.** `AffineTransforms/shaders/{normalVertex,normalFragment,normalGeo}.glsl` (already merged, working, reviewed) is a cleaned-up version of the exact same NGL9Demos `ShadedGrid` geometry-shader-normal-viz technique (verified: NGL9Demos' `ShadedGrid` and `AffineTransforms` both ship near-identical `normalGeo.glsl` — this is why the design spec's Phase-1 write-up called them the same underlying technique). Copy those 3 files byte-for-byte into `ShadedGrid/shaders/` rather than porting fresh from the NGL9Demos C++ source. Wire them up exactly the way `AffineTransforms/main.py` already does: `ShaderLib.load_shader("NormalViz", vert_path, frag_path, geo_path)` — `ShaderLib.load_shader`'s 4th positional argument (`geo`) IS supported and already proven working in this repo; there is no need for the manual `Shader`/`ShaderProgram`/low-level-class assembly the older `2026-07-04-ngl9-demo-port-design.md` spec's "Library boundary" section describes — that guidance predates `ShaderLib.load_shader` gaining a geometry-shader parameter and is now superseded by the actual working code in `AffineTransforms`.

**The wave-grid normal calculation is deliberately NOT a port of the C++'s per-vertex neighbour-cross-product method.** The C++ `ShadedGrid::createVAO()` computes each interior vertex's normal from 4 neighbouring cross products, but its edge-handling is incomplete (the bottom edge and both side edges are commented-out dead code — only the top edge gets a normal fixup, everything else keeps a zero-length placeholder normal, which would light as pure black/undefined at most of the grid's boundary). Since this is a regular height-field (`y = f(x, z)`), this plan uses the standard, fully-vectorizable heightfield-normal formula instead: for each grid vertex, take clamped left/right neighbours for the `x` central difference and clamped front/back neighbours for the `z` central difference, then `normal = normalize(cross(tangent_x, tangent_z))` — this produces correct, non-degenerate normals everywhere, including every edge and corner, with one numpy expression instead of a per-vertex Python loop.

- [ ] **Step 1: Copy the normal-visualization shaders**

```bash
mkdir -p ShadedGrid/shaders
cp AffineTransforms/shaders/normalVertex.glsl ShadedGrid/shaders/normalVertex.glsl
cp AffineTransforms/shaders/normalFragment.glsl ShadedGrid/shaders/normalFragment.glsl
cp AffineTransforms/shaders/normalGeo.glsl ShadedGrid/shaders/normalGeo.glsl
```

- [ ] **Step 2: Write the Phong shaders (world-space lighting)**

Create `ShadedGrid/shaders/PhongVertex.glsl`:

```glsl
#version 410 core
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

out vec3 fragmentNormal;
out vec3 worldPosition;

uniform mat4 M;
uniform mat4 MVP;
uniform mat3 normal_matrix;

void main()
{
    fragmentNormal = normalize(normal_matrix * inNormal);
    worldPosition = vec3(M * vec4(inVert, 1.0));
    gl_Position = MVP * vec4(inVert, 1.0);
}
```

Note: `normal_matrix` (snake_case) again, matching the same repo convention as Spotlight's shader above.

Create `ShadedGrid/shaders/PhongFragment.glsl`:

```glsl
#version 410 core
in vec3 fragmentNormal;
in vec3 worldPosition;
layout(location = 0) out vec4 fragColour;

struct Material
{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float shininess;
};

struct Light
{
    vec3 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform Material material;
uniform vec3 viewerPos;
#define NUM_LIGHTS 3
uniform Light light[NUM_LIGHTS];

vec4 pointLight(int i, vec3 n, vec3 eyeDir)
{
    vec3 L = normalize(light[i].position - worldPosition);
    float lambert = max(dot(n, L), 0.0);
    vec4 diffuse = material.diffuse * light[i].diffuse * lambert;
    vec4 ambient = material.ambient * light[i].ambient;
    vec4 specular = vec4(0.0);
    if (lambert > 0.0)
    {
        vec3 halfV = normalize(eyeDir + L);
        float nDotHV = max(dot(n, halfV), 0.0);
        specular = material.specular * light[i].specular * pow(nDotHV, material.shininess);
    }
    return ambient + diffuse + specular;
}

void main()
{
    vec3 n = normalize(fragmentNormal);
    vec3 eyeDir = normalize(viewerPos - worldPosition);
    vec4 colour = vec4(0.0);
    for (int i = 0; i < NUM_LIGHTS; ++i)
    {
        colour += pointLight(i, n, eyeDir);
    }
    colour.a = 1.0;
    fragColour = colour;
}
```

- [ ] **Step 3: Write the wave-grid generator and main.py**

Create `ShadedGrid/main.py`:

```python
#!/usr/bin/env -S uv run --script
"""ShadedGrid demo: an animated wave grid with a geometry-shader normal visualization."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import PySideEventHandlingMixin, ShaderLib, VAOFactory, VAOType
from ncca.ngl.opengl.abstract_vao import VertexData
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

_GRID_N = 40
_GRID_SIZE = 4.0


def _wave_heights(x: np.ndarray, z: np.ndarray, offset: float) -> np.ndarray:
    y = np.sin(x + offset) + np.cos(x - offset)
    y += np.sin(z + offset) + np.cos(z - offset)
    return y * 0.5


def build_wave_grid(n: int, size: float, offset: float) -> np.ndarray:
    """Non-indexed triangle-list grid, interleaved x,y,z,nx,ny,nz,u,v float32.

    Normals use the standard heightfield formula (central differences of
    height, clamped at the boundary), not the NGL9Demos source's incomplete
    per-vertex neighbour-cross-product method (see plan design notes).
    """
    coords = np.linspace(-size / 2.0, size / 2.0, n, dtype=np.float64)
    xs, zs = np.meshgrid(coords, coords, indexing="xy")
    ys = _wave_heights(xs, zs, offset)

    step = size / (n - 1)
    x_prev = _wave_heights(np.roll(xs, 1, axis=1), zs, offset)
    x_next = _wave_heights(np.roll(xs, -1, axis=1), zs, offset)
    x_prev[:, 0] = ys[:, 0]
    x_next[:, -1] = ys[:, -1]
    dy_dx = (x_next - x_prev) / (2.0 * step)
    dy_dx[:, 0] = (ys[:, 1] - ys[:, 0]) / step
    dy_dx[:, -1] = (ys[:, -1] - ys[:, -2]) / step

    z_prev = _wave_heights(xs, np.roll(zs, 1, axis=0), offset)
    z_next = _wave_heights(xs, np.roll(zs, -1, axis=0), offset)
    z_prev[0, :] = ys[0, :]
    z_next[-1, :] = ys[-1, :]
    dy_dz = (z_next - z_prev) / (2.0 * step)
    dy_dz[0, :] = (ys[1, :] - ys[0, :]) / step
    dy_dz[-1, :] = (ys[-1, :] - ys[-2, :]) / step

    normals = np.stack([-dy_dx, np.ones_like(ys), -dy_dz], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    u = (xs - xs.min()) / size
    v = (zs - zs.min()) / size

    def vertex(i: int, j: int) -> tuple[float, ...]:
        return (
            xs[j, i],
            ys[j, i],
            zs[j, i],
            normals[j, i, 0],
            normals[j, i, 1],
            normals[j, i, 2],
            u[j, i],
            v[j, i],
        )

    tris: list[float] = []
    for j in range(n - 1):
        for i in range(n - 1):
            tris.extend(vertex(i, j + 1))
            tris.extend(vertex(i + 1, j))
            tris.extend(vertex(i, j))
            tris.extend(vertex(i, j + 1))
            tris.extend(vertex(i + 1, j + 1))
            tris.extend(vertex(i + 1, j))
    return np.array(tris, dtype=np.float32)


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
        self.setTitle("ShadedGrid")
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.animate: bool = True
        self.offset: float = 0.0
        self.normal_size: float = 0.1
        self.draw_face_normals: bool = True
        self.draw_vertex_normals: bool = True
        self.vertex_count: int = 0

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 3, 6), Vec3(0, 0, 0), Vec3(0, 1, 0))

        ShaderLib.load_shader(
            "Phong", "shaders/PhongVertex.glsl", "shaders/PhongFragment.glsl"
        )
        ShaderLib.use("Phong")
        ShaderLib.set_uniform("material.ambient", 0.329412, 0.223529, 0.027451, 1.0)
        ShaderLib.set_uniform("material.diffuse", 0.780392, 0.568627, 0.113725, 1.0)
        ShaderLib.set_uniform("material.specular", 0.992157, 0.941176, 0.807843, 1.0)
        ShaderLib.set_uniform("material.shininess", 57.8974)
        ShaderLib.set_uniform("light[0].position", 3.0, 2.0, 2.0)
        ShaderLib.set_uniform("light[0].ambient", 0.1, 0.1, 0.1, 1.0)
        ShaderLib.set_uniform("light[0].diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light[0].specular", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light[1].position", -3.0, 1.5, 2.0)
        ShaderLib.set_uniform("light[1].ambient", 0.05, 0.05, 0.05, 1.0)
        ShaderLib.set_uniform("light[1].diffuse", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("light[1].specular", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("light[2].position", 0.0, 1.0, -3.0)
        ShaderLib.set_uniform("light[2].ambient", 0.05, 0.05, 0.05, 1.0)
        ShaderLib.set_uniform("light[2].diffuse", 0.4, 0.4, 0.4, 1.0)
        ShaderLib.set_uniform("light[2].specular", 0.4, 0.4, 0.4, 1.0)

        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "NormalViz",
            str(shader_dir / "normalVertex.glsl"),
            str(shader_dir / "normalFragment.glsl"),
            str(shader_dir / "normalGeo.glsl"),
        )
        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("vertNormalColour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("faceNormalColour", 1.0, 0.0, 0.0, 1.0)
        ShaderLib.set_uniform("drawFaceNormals", self.draw_face_normals)
        ShaderLib.set_uniform("drawVertexNormals", self.draw_vertex_normals)

        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        self._upload_grid()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(30)

    def _upload_grid(self) -> None:
        data = build_wave_grid(_GRID_N, _GRID_SIZE, self.offset)
        self.vertex_count = len(data) // 8
        self.vao.bind()
        self.vao.set_data(VertexData(data, self.vertex_count))
        stride = 8 * 4
        self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)
        self.vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, stride, 3 * 4)
        self.vao.set_vertex_attribute_pointer(2, 2, gl.GL_FLOAT, stride, 6 * 4)
        self.vao.set_num_indices(self.vertex_count)
        self.vao.unbind()

    def _on_tick(self) -> None:
        if self.animate:
            self.offset += 0.02
            self._upload_grid()
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

        mvp = self.project @ self.view @ global_tx
        normal_matrix = Mat3.from_mat4(global_tx).inverse().transposed()

        self.vao.bind()

        ShaderLib.use("Phong")
        ShaderLib.set_uniform("M", global_tx)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normal_matrix", normal_matrix)
        ShaderLib.set_uniform("viewerPos", 0.0, 3.0, 6.0)
        self.vao.draw()

        ShaderLib.use("NormalViz")
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalSize", self.normal_size)
        self.vao.draw()

        self.vao.unbind()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.5, 150.0)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.normal_size += 0.01
        elif key == Qt.Key_Minus:
            self.normal_size = max(0.0, self.normal_size - 0.01)
        elif key == Qt.Key_1:
            self.draw_face_normals = not self.draw_face_normals
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("drawFaceNormals", self.draw_face_normals)
        elif key == Qt.Key_2:
            self.draw_vertex_normals = not self.draw_vertex_normals
            ShaderLib.use("NormalViz")
            ShaderLib.set_uniform("drawVertexNormals", self.draw_vertex_normals)
        elif key == Qt.Key_U:
            self.animate = not self.animate
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

Note: `VAOType.SIMPLE` — confirm this exact enum member name and its import location (`from ncca.ngl.opengl import VAOType` or wherever `VAOFactory` itself is imported from) before relying on it; `KleinBottle/main.py` (already merged) uses `VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)` with `from ncca.ngl.opengl.abstract_vao import VertexData` — mirror that file's exact import lines if anything here doesn't match on the first run.

Note: re-uploading the full grid via `set_data()` every animated frame (rather than a `glBufferSubData` partial update) mirrors the C++ source's own approach (`createVAO()` is called fresh every `paintGL`) and is simple/correct; `_GRID_N = 40` (1521 vertices, 2 legs shy of the previous vertex, 3042 triangles) keeps this comfortably interactive in PyOpenGL — reduce it if the smoke test or interactive framerate is unexpectedly sluggish, but don't reduce it preemptively without evidence it's needed.

- [ ] **Step 4: Make executable and smoke-test**

```bash
chmod +x ShadedGrid/main.py
cd ShadedGrid && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. (Run WITHOUT `QT_QPA_PLATFORM=offscreen`.)

- [ ] **Step 5: Write README.md**

Create `ShadedGrid/README.md`:

```markdown
# ShadedGrid

An animated wave-height grid, Phong-shaded with 3-point lighting, with a
geometry-shader pass drawn on top that visualizes each triangle's face
normal (red) and each vertex's normal (yellow) as line segments — watch
them rotate and stretch as the surface undulates. Normals use the standard
heightfield central-difference formula, correct at every edge (the
NGL9Demos C++ original's per-vertex neighbour method left most of the grid
boundary with degenerate normals).

## Controls
- `1` : toggle face-normal lines
- `2` : toggle vertex-normal lines
- `+` / `-` : normal line length
- `u` : toggle wave animation
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset
```

- [ ] **Step 6: Commit**

```bash
git add ShadedGrid/
git commit -m "feat(shaded-grid): add geometry-shader normal visualization demo"
```

---

## Final steps (after all 3 tasks)

- [ ] **Add root README.md entries**

Add a row for `Spotlight` under `## Lighting & Shadows` and a row for `ShadedGrid` under `## Geometry & Tessellation Shaders` in the root `README.md` — both sections already exist, follow the existing row format exactly (name, link, thumbnail).

- [ ] **Run full verification**

```bash
uv run ruff check Spotlight/ ShadedGrid/
uv run ruff format --check Spotlight/ ShadedGrid/
```
Expected: ruff clean. Neither demo has a pure-maths module requiring pytest coverage (Spotlight's light math and ShadedGrid's wave-grid generator are visual/interactive, not the kind of standalone testable unit `ViewToWorldTransform`'s unprojection was — no test suite is required by the spec for this phase).

- [ ] **Report to Jon**

List the 2 `.png` screenshots that still need capturing (`Spotlight.png`, `ShadedGrid.png`).
