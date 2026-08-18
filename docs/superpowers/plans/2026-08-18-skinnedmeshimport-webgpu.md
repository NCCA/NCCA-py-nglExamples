# SkinnedMeshImport WebGPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WebGPU renderer for `SkinnedMeshImport`, alongside its existing OpenGL one, mirroring the split `BVHViewer` already has (same window, timeline and camera controls; different renderer).

**Architecture:** `main.py`'s `MainWindow` becomes backend-agnostic (an injectable `viewport`, matching `BVHViewer/main.py`). A new `SkinWebGPUViewport` (in `main_webgpu.py`) owns camera/four-view/input state, duplicated from `SkinViewport` rather than shared, because `QOpenGLWindow` and `WebGPUWidget` are unrelated base classes — the same choice `BVHViewer` already made. A new `SkinWebGPURenderer` (in `webgpu_renderer.py`) owns the wgpu pipeline, vertex/index/bone buffers, and one texture bind group per submesh, driven by `skin_webgpu.wgsl`.

**Tech Stack:** Python, PySide6/Qt, `wgpu-py`, `ncca.ngl` / `ncca.ngl.webgpu`, `impasse`, numpy, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-18-skinnedmeshimport-webgpu-design.md`

## Global Constraints

- No change to `mesh.py` / `SkinnedMesh`'s public behaviour or its existing tests (`uv run pytest SkinnedMeshImport/tests` must stay green throughout).
- Demo folder stays self-contained — no code shared with `BVHViewer` or extracted into a common module.
- Bone palette on the WebGPU side is a storage buffer sized to the mesh's actual bone count — no `MAX_BONES` cap (that ceiling is OpenGL-only, forced by GLSL's fixed-size uniform arrays).
- `mesh.py`'s existing V-flip (baked in for OpenGL) is left as-is; WebGPU un-flips it in `skin_webgpu.wgsl`, not in the shared loader.
- Every new/modified `.py` file must pass `uv run ruff check` and `uv run ruff format --check`.

---

## File Structure

- `SkinnedMeshImport/main.py` — **modify**: injectable `viewport` param on `MainWindow`; extract `_parse_args()`/`main()`. No other change.
- `SkinnedMeshImport/webgpu_renderer.py` — **create**: `SkinWebGPURenderer` — pipeline, vertex/index buffers, bone storage buffer, per-submesh textures, per-pane camera uniforms.
- `SkinnedMeshImport/skin_webgpu.wgsl` — **create**: vertex skin + world-space Blinn-Phong + texture sample.
- `SkinnedMeshImport/main_webgpu.py` — **create**: `SkinWebGPUViewport` (camera/four-view/input, duplicated from `SkinViewport`) + `MainWindow` subclass + entry point.
- `SkinnedMeshImport/README.md` — **modify**: WebGPU run instructions, file list, differences section.
- `README.md` (repo root) — **modify**: one line, mention both backends.

---

### Task 1: Make `main.py`'s `MainWindow` backend-agnostic

**Files:**
- Modify: `SkinnedMeshImport/main.py:657-666` (`MainWindow.__init__`)
- Modify: `SkinnedMeshImport/main.py:805-855` (`DebugApplication` tail / `__main__` block)

**Interfaces:**
- Produces: `MainWindow(model_path: Path = DEFAULT_MODEL, viewport: SkinViewport | QWidget | None = None)`, `_parse_args(argv: list[str] | None = None) -> argparse.Namespace`, `main(argv: list[str] | None = None) -> int`. `main_webgpu.py` (Task 3) imports all three from `main`, plus the already-existing `DEFAULT_MODEL`, `MESH_FILE_FILTER`, `DebugApplication`, `OrthoView`, `TOP_VIEW`/`PERSPECTIVE_VIEW`/`FRONT_VIEW`/`SIDE_VIEW`.

- [ ] **Step 1: Give `MainWindow` an injectable viewport**

Replace (`main.py:657-666`):

```python
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        super().__init__()
        self.setWindowTitle("SkinnedMeshImport")
        self.resize(1100, 780)
        self.setStyleSheet(_APP_STYLE)

        self.viewport = SkinViewport(model_path)
        viewport_widget = QWidget.createWindowContainer(self.viewport, self)
        viewport_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        viewport_widget.setFocus()
```

with:

```python
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        viewport: SkinViewport | QWidget | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SkinnedMeshImport")
        self.resize(1100, 780)
        self.setStyleSheet(_APP_STYLE)

        self.viewport = viewport if viewport is not None else SkinViewport(model_path)
        if isinstance(self.viewport, QWidget):
            viewport_widget = self.viewport
        else:
            viewport_widget = QWidget.createWindowContainer(self.viewport, self)
        viewport_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        viewport_widget.setFocus()
```

(This is the same shape as `BVHViewer/main.py`'s `MainWindow.__init__`.)

- [ ] **Step 2: Extract `_parse_args()` and `main()`**

Replace the tail of the file from the `if __name__ == "__main__":` block (`main.py:817-855`):

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL), help="mesh file to load"
    )
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)

    window = MainWindow(Path(args.model))
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

with:

```python
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL), help="mesh file to load"
    )
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    return parser.parse_args(argv)


def _configure_surface_format() -> None:
    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_surface_format()
    app_type = DebugApplication if args.debug else QApplication
    app = app_type(sys.argv if argv is None else [sys.argv[0], *argv])

    window = MainWindow(Path(args.model))
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify no regression**

Run:

```bash
uv run pytest SkinnedMeshImport/tests
uv run SkinnedMeshImport/main.py --smoketest
```

Expected: pytest passes; the second command prints `SMOKETEST OK` and exits 0.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check SkinnedMeshImport/main.py
uv run ruff format SkinnedMeshImport/main.py
git add SkinnedMeshImport/main.py
git commit -m "refactor(skinned-mesh-import): make MainWindow accept an injectable viewport"
```

---

### Task 2: `webgpu_renderer.py` and `skin_webgpu.wgsl`

**Files:**
- Create: `SkinnedMeshImport/skin_webgpu.wgsl`
- Create: `SkinnedMeshImport/webgpu_renderer.py`

**Interfaces:**
- Consumes: `mesh.SkinnedMesh` (attributes `positions`, `normals`, `texcoords`, `bone_ids`, `bone_weights`, `indices`, `bone_names`, `submeshes` — each `SubMesh` has `index_count`, `index_offset`, `texture_path`), `mesh.SkinnedMesh.bone_transforms(time_seconds: float) -> list[Mat4]`.
- Produces: `SkinWebGPURenderer(device: wgpu.GPUDevice)` with methods `set_mesh(mesh: SkinnedMesh) -> None`, `update_bones(transforms: list[Mat4]) -> None`, `update_camera(index: int, view_projection: Mat4, model: Mat4, eye_position: Vec3, light_position: Vec3) -> None`, `render(render_pass: wgpu.GPURenderPassEncoder, camera_index: int) -> None`. Used by `main_webgpu.py` (Task 3).

- [ ] **Step 1: Write the shader**

Create `SkinnedMeshImport/skin_webgpu.wgsl`:

```wgsl
struct Camera {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    eye_position: vec4<f32>,
    light_position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> bones: array<mat4x4<f32>>;

@group(2) @binding(0) var t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var s_diffuse: sampler;

const LIGHT_AMBIENT: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
const LIGHT_DIFFUSE: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const LIGHT_SPECULAR: vec3<f32> = vec3<f32>(0.8, 0.8, 0.8);
const MATERIAL_AMBIENT: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
const MATERIAL_SPECULAR: vec3<f32> = vec3<f32>(0.4, 0.4, 0.4);
const MATERIAL_SHININESS: f32 = 32.0;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bone_ids: vec4<f32>,
    @location(4) bone_weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var skin: mat4x4<f32> = bones[u32(input.bone_ids.x)] * input.bone_weights.x;
    skin += bones[u32(input.bone_ids.y)] * input.bone_weights.y;
    skin += bones[u32(input.bone_ids.z)] * input.bone_weights.z;
    skin += bones[u32(input.bone_ids.w)] * input.bone_weights.w;

    let skinned_position = skin * vec4<f32>(input.position, 1.0);
    let skinned_normal = skin * vec4<f32>(input.normal, 0.0);
    let world_position = camera.model * skinned_position;

    var output: VertexOutput;
    output.position = camera.view_projection * world_position;
    output.world_position = world_position.xyz;
    output.normal = normalize((camera.model * skinned_normal).xyz);
    output.uv = input.uv;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // mesh.py flips V once for OpenGL's bottom-left texture origin; WebGPU's
    // origin is top-left, so this flips it back rather than giving the
    // shared loader a second UV buffer.
    let flipped_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    let tex_colour = textureSample(t_diffuse, s_diffuse, flipped_uv);

    let normal = normalize(input.normal);
    let eye_direction = normalize(camera.eye_position.xyz - input.world_position);
    let light_direction = normalize(camera.light_position.xyz - input.world_position);
    let half_vector = normalize(eye_direction + light_direction);

    let lambert = max(dot(normal, light_direction), 0.0);
    var colour = MATERIAL_AMBIENT * LIGHT_AMBIENT;
    if lambert > 0.0 {
        colour += tex_colour.rgb * LIGHT_DIFFUSE * lambert;
        let specular_term = pow(max(dot(normal, half_vector), 0.0), MATERIAL_SHININESS);
        colour += MATERIAL_SPECULAR * LIGHT_SPECULAR * specular_term;
    }
    return vec4<f32>(colour, tex_colour.a);
}
```

This is the WGSL port of `shaders/SkinVertex.glsl` + `shaders/SkinFragment.glsl`: same four-bone linear blend skin, same Blinn-Phong terms and constants, computed in world space instead of eye space (see the design spec's "skin_webgpu.wgsl" section for why), with the light coincident with the viewer per pane (a "headlamp" — see Task 3, where `light_position` is always passed as the same value as `eye_position`), exactly matching the original GLSL's `light.position = view @ Vec4(eye, 1.0)` trick.

- [ ] **Step 2: Write the renderer**

Create `SkinnedMeshImport/webgpu_renderer.py`:

```python
"""The WebGPU pipeline used to draw a skinned mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu
from mesh import SkinnedMesh
from ncca.ngl import Image, Mat4, Vec3, logger

SHADER_PATH = Path(__file__).with_name("skin_webgpu.wgsl")

_CAMERA_DTYPE = np.dtype(
    [
        ("view_projection", np.float32, (4, 4)),
        ("model", np.float32, (4, 4)),
        ("eye_position", np.float32, (4,)),
        ("light_position", np.float32, (4,)),
    ]
)

_VERTEX_BUFFER_LAYOUTS = [
    {
        "array_stride": 12,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
    },
    {
        "array_stride": 12,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 1}],
    },
    {
        "array_stride": 8,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 2}],
    },
    {
        "array_stride": 16,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x4", "offset": 0, "shader_location": 3}],
    },
    {
        "array_stride": 16,
        "step_mode": "vertex",
        "attributes": [{"format": "float32x4", "offset": 0, "shader_location": 4}],
    },
]


class SkinWebGPURenderer:
    """Own the wgpu pipeline, buffers and textures used to draw one SkinnedMesh."""

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.index_count = 0
        self.submeshes: list = []
        self._bone_capacity = 1
        self._texture_cache: dict[str | None, wgpu.GPUBindGroup] = {}
        self._fallback_texture: wgpu.GPUBindGroup | None = None

        shader = device.create_shader_module(code=SHADER_PATH.read_text())
        self._camera_layout = self._create_camera_layout()
        self._bone_layout = self._create_bone_layout()
        self._texture_layout = self._create_texture_layout()
        self._camera_buffers, self._camera_bind_groups = self._create_camera_bindings()
        self._bone_buffer = self._create_bone_buffer(self._bone_capacity)
        self._bone_bind_group = self._create_bone_bind_group()
        self._sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear
        )
        self._pipeline = self._create_pipeline(shader)

    # ------------------------------------------------------------ layouts

    def _create_camera_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )

    def _create_bone_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                }
            ]
        )

    def _create_texture_layout(self) -> wgpu.GPUBindGroupLayout:
        return self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )

    def _create_camera_bindings(
        self,
    ) -> tuple[list[wgpu.GPUBuffer], list[wgpu.GPUBindGroup]]:
        buffers = []
        bind_groups = []
        for index in range(4):
            buffer = self.device.create_buffer(
                size=_CAMERA_DTYPE.itemsize,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                label=f"skin_camera_{index}",
            )
            buffers.append(buffer)
            bind_groups.append(
                self.device.create_bind_group(
                    layout=self._camera_layout,
                    entries=[{"binding": 0, "resource": {"buffer": buffer}}],
                )
            )
        return buffers, bind_groups

    def _create_bone_buffer(self, capacity: int) -> wgpu.GPUBuffer:
        return self.device.create_buffer(
            size=capacity * 64,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="skin_bones",
        )

    def _create_bone_bind_group(self) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(
            layout=self._bone_layout,
            entries=[{"binding": 0, "resource": {"buffer": self._bone_buffer}}],
        )

    def _create_pipeline(self, shader: wgpu.GPUShaderModule) -> wgpu.GPURenderPipeline:
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[
                self._camera_layout,
                self._bone_layout,
                self._texture_layout,
            ]
        )
        return self.device.create_render_pipeline(
            label="skin_pipeline",
            layout=layout,
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": _VERTEX_BUFFER_LAYOUTS,
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": 4},
        )

    # --------------------------------------------------------------- mesh

    def set_mesh(self, mesh: SkinnedMesh) -> None:
        """Upload a new mesh's geometry, grow the bone buffer if needed, load textures."""
        self.submeshes = mesh.submeshes
        self.index_count = len(mesh.indices)

        self._position_buffer = self._vertex_buffer(mesh.positions)
        self._normal_buffer = self._vertex_buffer(mesh.normals)
        self._uv_buffer = self._vertex_buffer(mesh.texcoords)
        self._bone_id_buffer = self._vertex_buffer(mesh.bone_ids)
        self._bone_weight_buffer = self._vertex_buffer(mesh.bone_weights)
        self._index_buffer = self.device.create_buffer_with_data(
            data=mesh.indices.astype(np.uint32, copy=False),
            usage=wgpu.BufferUsage.INDEX,
        )

        bone_count = max(len(mesh.bone_names), 1)
        if bone_count > self._bone_capacity:
            self._bone_buffer.destroy()
            self._bone_capacity = bone_count
            self._bone_buffer = self._create_bone_buffer(self._bone_capacity)
            self._bone_bind_group = self._create_bone_bind_group()

        self._texture_cache = {}
        for submesh in mesh.submeshes:
            self._texture_bind_group(submesh.texture_path)

    def _vertex_buffer(self, data: np.ndarray) -> wgpu.GPUBuffer:
        return self.device.create_buffer_with_data(
            data=data.astype(np.float32, copy=False),
            usage=wgpu.BufferUsage.VERTEX,
        )

    def update_bones(self, transforms: list[Mat4]) -> None:
        """Rewrite the bone storage buffer for the current animation frame."""
        if not transforms:
            return
        matrices = np.stack([t.to_numpy() for t in transforms], axis=0).astype(
            np.float32, copy=False
        )
        self.device.queue.write_buffer(self._bone_buffer, 0, matrices.tobytes())

    def update_camera(
        self,
        index: int,
        view_projection: Mat4,
        model: Mat4,
        eye_position: Vec3,
        light_position: Vec3,
    ) -> None:
        """Write one pane's (index 0-3) camera/model/light uniform."""
        data = np.zeros((), dtype=_CAMERA_DTYPE)
        data["view_projection"] = view_projection.to_numpy()
        data["model"] = model.to_numpy()
        data["eye_position"] = (eye_position.x, eye_position.y, eye_position.z, 1.0)
        data["light_position"] = (
            light_position.x,
            light_position.y,
            light_position.z,
            1.0,
        )
        self.device.queue.write_buffer(self._camera_buffers[index], 0, data.tobytes())

    # ----------------------------------------------------------- textures

    def _fallback_bind_group(self) -> wgpu.GPUBindGroup:
        if self._fallback_texture is None:
            self._fallback_texture = self._create_texture_bind_group(
                np.full((1, 1, 4), 255, dtype=np.uint8)
            )
        return self._fallback_texture

    def _texture_bind_group(self, texture_path: str | None) -> wgpu.GPUBindGroup:
        if texture_path in self._texture_cache:
            return self._texture_cache[texture_path]
        bind_group = self._load_texture_bind_group(texture_path)
        self._texture_cache[texture_path] = bind_group
        return bind_group

    def _load_texture_bind_group(self, texture_path: str | None) -> wgpu.GPUBindGroup:
        if texture_path is None:
            return self._fallback_bind_group()
        try:
            pixels = Image(texture_path).get_pixels()
            if pixels.shape[2] == 3:
                rgba = np.empty((*pixels.shape[:2], 4), dtype=np.uint8)
                rgba[:, :, :3] = pixels
                rgba[:, :, 3] = 255
            else:
                rgba = pixels
        except Exception as error:
            # impasse.errors.AssimpError and a missing/corrupt image file both
            # land here -- keep the mesh visible (flat white) rather than
            # losing it, same fallback the OpenGL path uses.
            logger.warning(
                f"Could not load texture {texture_path!r} ({error}); "
                "using a flat fallback"
            )
            return self._fallback_bind_group()
        return self._create_texture_bind_group(rgba)

    def _create_texture_bind_group(self, rgba: np.ndarray) -> wgpu.GPUBindGroup:
        height, width = rgba.shape[:2]
        texture = self.device.create_texture(
            size=(width, height, 1),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"bytes_per_row": width * 4, "rows_per_image": height},
            (width, height, 1),
        )
        return self.device.create_bind_group(
            layout=self._texture_layout,
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": self._sampler},
            ],
        )

    # -------------------------------------------------------------- draw

    def render(self, render_pass: wgpu.GPURenderPassEncoder, camera_index: int) -> None:
        if self.index_count == 0:
            return
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(
            0, self._camera_bind_groups[camera_index], [], 0, 999999
        )
        render_pass.set_bind_group(1, self._bone_bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self._position_buffer)
        render_pass.set_vertex_buffer(1, self._normal_buffer)
        render_pass.set_vertex_buffer(2, self._uv_buffer)
        render_pass.set_vertex_buffer(3, self._bone_id_buffer)
        render_pass.set_vertex_buffer(4, self._bone_weight_buffer)
        render_pass.set_index_buffer(self._index_buffer, wgpu.IndexFormat.uint32)
        for submesh in self.submeshes:
            bind_group = self._texture_bind_group(submesh.texture_path)
            render_pass.set_bind_group(2, bind_group, [], 0, 999999)
            render_pass.draw_indexed(submesh.index_count, 1, submesh.index_offset)
```

- [ ] **Step 3: Verify it constructs and uploads without error**

Run from the repo root:

```bash
uv run python -c "
import sys
from pathlib import Path

from ncca.ngl import Mat4, Vec3
from wgpu.utils import get_default_device

sys.path.insert(0, 'SkinnedMeshImport')
from mesh import SkinnedMesh
from webgpu_renderer import SkinWebGPURenderer

device = get_default_device()
renderer = SkinWebGPURenderer(device)
mesh = SkinnedMesh(str(Path('SkinnedMeshImport/models/guard/boblampclean.md5mesh')))
renderer.set_mesh(mesh)
renderer.update_bones(mesh.bone_transforms(0.0))
renderer.update_camera(0, Mat4(), Mat4(), Vec3(0, 0, 5), Vec3(0, 0, 5))
print('RENDERER SMOKE OK', renderer.index_count, len(renderer.submeshes))
"
```

Expected: prints `RENDERER SMOKE OK <index_count> <submesh_count>` with no traceback (`index_count` should be a positive number in the tens of thousands for the guard model; `submesh_count` should be a small positive integer). This exercises pipeline creation, all five vertex buffers, the index buffer, bone storage buffer, texture loading for every submesh (including the fallback path if any `.tga` fails to load) and one uniform write, without needing a Qt window.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check SkinnedMeshImport/webgpu_renderer.py
uv run ruff format SkinnedMeshImport/webgpu_renderer.py
git add SkinnedMeshImport/skin_webgpu.wgsl SkinnedMeshImport/webgpu_renderer.py
git commit -m "feat(skinned-mesh-import): add the WebGPU skinning pipeline"
```

---

### Task 3: `main_webgpu.py` — viewport, camera, four-view, entry point

**Files:**
- Create: `SkinnedMeshImport/main_webgpu.py`

**Interfaces:**
- Consumes: from `main.py` (Task 1) — `DEFAULT_MODEL`, `TOP_VIEW`, `PERSPECTIVE_VIEW`, `FRONT_VIEW`, `SIDE_VIEW`, `OrthoView`, `DebugApplication`, `_parse_args`, `MainWindow as ViewerMainWindow` (its inherited `_open_file_dialog` uses `MESH_FILE_FILTER` internally — `main_webgpu.py` doesn't need to import it itself). From `webgpu_renderer.py` (Task 2) — `SkinWebGPURenderer`. From `mesh.py` — `SkinnedMesh`.
- Produces: `main(argv: list[str] | None = None) -> int`, run via `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 1: Write the file**

Create `SkinnedMeshImport/main_webgpu.py`:

```python
#!/usr/bin/env -S uv run --script
"""A WebGPU version of the PyNGL skinned-mesh import demo.

See ``main.py`` for the OpenGL version this mirrors, and ``mesh.py`` for
the impasse-based loader and skinning maths shared by both.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import wgpu
from main import (
    DEFAULT_MODEL,
    FRONT_VIEW,
    PERSPECTIVE_VIEW,
    SIDE_VIEW,
    TOP_VIEW,
    DebugApplication,
    OrthoView,
    _parse_args,
)
from main import (
    MainWindow as ViewerMainWindow,
)
from mesh import SkinnedMesh
from ncca.ngl import FirstPersonCamera, Mat4, PerspMode, Vec3, look_at, ortho
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from webgpu_renderer import SkinWebGPURenderer
from wgpu.utils import get_default_device

_VIEW_LABELS = ("TOP", "PERSPECTIVE", "FRONT", "SIDE")


class WebGPUOrthoView(OrthoView):
    """An orthographic pane using WebGPU's zero-to-one depth range."""

    def matrices(self, pane_width: int, pane_height: int) -> tuple[Mat4, Mat4]:
        half_width = self.half_height * pane_width / max(pane_height, 1)
        project = ortho(
            -half_width,
            half_width,
            -self.half_height,
            self.half_height,
            0.05,
            5000.0,
            PerspMode.WebGPU,
        )
        view = look_at(self.eye + self.pan, self.target + self.pan, self.up)
        return view, project


class SkinWebGPUViewport(WebGPUWidget):
    """The WebGPU viewport: loads the mesh, skins it on the GPU, and draws it.

    Camera/navigation duplicates ``main.py``'s ``SkinViewport`` -- see the
    design spec for why this isn't shared: ``QOpenGLWindow`` and
    ``WebGPUWidget`` are unrelated Qt base classes, the same reason
    ``BVHViewer/main_webgpu.py`` duplicates ``BvhViewport`` rather than
    subclassing it.
    """

    _MOVE_KEYS = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D}
    _DIVIDER_WIDTH = 2

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("Skinned Mesh Import WebGPU")
        self.window_width, self.window_height = self.texture_size
        self.model_path = model_path
        self.mesh = SkinnedMesh(str(model_path))
        self.current_frame = 0

        self.four_view = False
        self._maximized_pane: int | None = None
        self._panning_pane: int | None = None
        self.keys_pressed: set[Qt.Key] = set()
        self._rotating_camera = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()
        self._last_frame_time = 0.0

        self._compute_view_setup()
        self._initialise_webgpu()

    # -------------------------------------------------------- camera setup

    def _compute_view_setup(self) -> None:
        """(Re)build the model-axis correction, camera and ortho panes for the current mesh."""
        z_up = self.model_path.suffix.lower() == ".md5mesh"
        self.model_matrix = Mat4().rotate_x(-90.0) if z_up else Mat4()

        bbox_min, bbox_max = self.mesh.bounding_box()

        def to_display(corner: list[float]) -> Vec3:
            if z_up:
                return Vec3(corner[0], corner[2], -corner[1])
            return Vec3(corner[0], corner[1], corner[2])

        a = to_display(bbox_min)
        b = to_display(bbox_max)
        lo = Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
        hi = Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))
        centre = Vec3((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5)
        height = float(max(hi.y - lo.y, 0.001))
        distance = height * 2.5

        eye = Vec3(centre.x, centre.y, hi.z + height * 1.5)
        self.camera = FirstPersonCamera(
            eye, centre, Vec3(0.0, 1.0, 0.0), 45.0, PerspMode.WebGPU
        )
        self.camera.speed = height * 0.4
        self._look_at(eye, centre)

        half_height = height * 0.75
        self.ortho_views: dict[int, WebGPUOrthoView] = {
            TOP_VIEW: WebGPUOrthoView(
                eye=Vec3(centre.x, hi.y + distance, centre.z),
                target=centre,
                up=Vec3(0.0, 0.0, -1.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            FRONT_VIEW: WebGPUOrthoView(
                eye=Vec3(centre.x, centre.y, hi.z + distance),
                target=centre,
                up=Vec3(0.0, 1.0, 0.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            SIDE_VIEW: WebGPUOrthoView(
                eye=Vec3(hi.x + distance, centre.y, centre.z),
                target=centre,
                up=Vec3(0.0, 1.0, 0.0),
                right=Vec3(0.0, 0.0, -1.0),
                half_height=half_height,
            ),
        }

    def _look_at(self, eye: Vec3, target: Vec3) -> None:
        """Point the FirstPersonCamera at ``target`` -- its constructor ignores ``look``."""
        direction = (target - eye).normalized()
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, direction.y))))
        yaw = math.degrees(math.atan2(direction.z, direction.x))
        self.camera.yaw = yaw
        self.camera.pitch = pitch
        self.camera._update_camera_vectors()

    # ------------------------------------------------------------- webgpu

    def _initialise_webgpu(self) -> None:
        self.device = get_default_device()
        self._create_render_buffer()
        self.renderer = SkinWebGPURenderer(self.device)
        self.renderer.set_mesh(self.mesh)
        self.start_update_timer(16)
        self.set_frame(0)
        self.update()

    def set_frame(self, frame: int) -> None:
        """Pose the mesh at the given timeline frame and request a repaint."""
        self.current_frame = frame
        time_seconds = frame / self.mesh.ticks_per_second()
        self.renderer.update_bones(self.mesh.bone_transforms(time_seconds))
        self.update()

    def load_model(self, model_path: Path) -> None:
        """Replace the current mesh with the one at ``model_path``.

        Raises whatever ``SkinnedMesh`` raises without touching the
        currently-loaded mesh -- same contract as ``SkinViewport.load_model``.
        """
        new_mesh = SkinnedMesh(str(model_path))
        self.model_path = model_path
        self.mesh = new_mesh
        self._compute_view_setup()
        self.renderer.set_mesh(self.mesh)
        self.set_frame(0)

    def paintWebGPU(self) -> None:
        frame_time = self._frame_timer.elapsed() * 0.001
        delta_time = min(max(frame_time - self._last_frame_time, 0.0), 0.05)
        self._last_frame_time = frame_time
        self._advance_camera(delta_time)

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.25, 0.25, 0.28, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        for index, rectangle, view, project, eye in self._pane_draws():
            x, y, width, height = rectangle
            render_pass.set_viewport(x, y, width, height, 0.0, 1.0)
            render_pass.set_scissor_rect(x, y, width, height)
            # The light sits at the viewer's own position (a "headlamp"),
            # the same trick main.py's _draw_mesh uses (light.position set
            # to the eye transformed by its own view matrix).
            self.renderer.update_camera(
                index, project @ view, self.model_matrix, eye, eye
            )
            self.renderer.render(render_pass, index)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        if self.four_view:
            self._draw_view_labels()
        self._update_colour_buffer()

        if self.keys_pressed & self._MOVE_KEYS:
            self.update()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.window_width = width
        self.window_height = height
        if self.four_view and self._maximized_pane is None:
            _, _, pane_width, pane_height = self._four_view_rectangles()[0]
            aspect = pane_width / max(pane_height, 1)
        else:
            aspect = width / max(height, 1)
        self._set_perspective_projection(aspect)
        self.update()

    def _set_perspective_projection(self, aspect: float) -> None:
        self.camera.aspect = aspect
        self.camera.near = 0.05
        self.camera.far = 5000.0
        self.camera._projection = self.camera.set_projection(
            self.camera.zoom,
            aspect,
            self.camera.near,
            self.camera.far,
            PerspMode.WebGPU,
        )

    # --------------------------------------------------------- four-view

    def set_four_view(self, enabled: bool) -> None:
        self.four_view = enabled
        self._maximized_pane = None
        self.resizeWebGPU(self.window_width, self.window_height)

    def toggle_maximized_pane(self, x: float, y: float) -> None:
        if not self.four_view:
            return
        if self._maximized_pane is not None:
            self._maximized_pane = None
        else:
            index = self._pane_index_at(x, y)
            if index is None:
                return
            self._maximized_pane = index
        self.resizeWebGPU(self.window_width, self.window_height)

    def _four_view_rectangles(self) -> list[tuple[int, int, int, int]]:
        divider = self._DIVIDER_WIDTH
        left_width = max(1, (self.window_width - divider) // 2)
        top_height = max(1, (self.window_height - divider) // 2)
        right_x = left_width + divider
        bottom_y = top_height + divider
        right_width = max(1, self.window_width - right_x)
        bottom_height = max(1, self.window_height - bottom_y)
        return [
            (0, 0, left_width, top_height),
            (right_x, 0, right_width, top_height),
            (0, bottom_y, left_width, bottom_height),
            (right_x, bottom_y, right_width, bottom_height),
        ]

    def _pane_rectangles(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        if self._maximized_pane is not None:
            return [
                (self._maximized_pane, (0, 0, self.window_width, self.window_height))
            ]
        if self.four_view:
            return list(enumerate(self._four_view_rectangles()))
        return [(PERSPECTIVE_VIEW, (0, 0, self.window_width, self.window_height))]

    def _pane_draws(
        self,
    ) -> list[tuple[int, tuple[int, int, int, int], Mat4, Mat4, Vec3]]:
        draws = []
        for index, rectangle in self._pane_rectangles():
            if index == PERSPECTIVE_VIEW:
                draws.append(
                    (
                        index,
                        rectangle,
                        self.camera.view,
                        self.camera.projection,
                        self.camera.eye,
                    )
                )
            else:
                _, _, pane_width, pane_height = rectangle
                ortho_view = self.ortho_views[index]
                view, project = ortho_view.matrices(pane_width, pane_height)
                draws.append(
                    (index, rectangle, view, project, ortho_view.eye + ortho_view.pan)
                )
        return draws

    def _pane_index_at(self, x: float, y: float) -> int | None:
        if not self.four_view:
            return None
        device_x = x * self.ratio
        device_y = y * self.ratio
        for index, (rx, ry, width, height) in self._pane_rectangles():
            if rx <= device_x < rx + width and ry <= device_y < ry + height:
                return index
        return None

    def _draw_view_labels(self) -> None:
        colour = QColor(209, 214, 219)
        for index, (x, y, _, _) in self._pane_rectangles():
            self.render_text(
                round(x / self.ratio) + 10,
                round(y / self.ratio) + 20,
                _VIEW_LABELS[index],
                12,
                colour=colour,
            )

    # ------------------------------------------------------------- input

    def _advance_camera(self, delta_time: float) -> None:
        forward = float(Qt.Key.Key_W in self.keys_pressed) - float(
            Qt.Key.Key_S in self.keys_pressed
        )
        strafe = float(Qt.Key.Key_D in self.keys_pressed) - float(
            Qt.Key.Key_A in self.keys_pressed
        )
        if forward != 0.0 or strafe != 0.0:
            self.camera.move(forward, strafe, delta_time)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.add(event.key())
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.discard(event.key())
            self.update()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self.four_view
                and self._pane_index_at(position.x(), position.y()) != PERSPECTIVE_VIEW
            ):
                return
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self._rotating_camera = True
            return
        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            index = self._pane_index_at(position.x(), position.y())
            if index is not None and index != PERSPECTIVE_VIEW:
                self._panning_pane = index
                self._last_mouse_x = position.x()
                self._last_mouse_y = position.y()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if self._rotating_camera and event.buttons() & Qt.MouseButton.LeftButton:
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self.camera.process_mouse_movement(diff_x, -diff_y)
            self.update()
            return
        if self._panning_pane is not None and event.buttons() & (
            Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton
        ):
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            _, _, _, pane_height = dict(self._pane_rectangles())[self._panning_pane]
            self.ortho_views[self._panning_pane].pan_by(
                diff_x, diff_y, pane_height / self.ratio
            )
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating_camera = False
            return
        if event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning_pane = None
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        position = event.position()
        index = self._pane_index_at(position.x(), position.y())
        if index is not None and index != PERSPECTIVE_VIEW:
            view = self.ortho_views[index]
            wheel_steps = event.angleDelta().y() / 120.0
            view.half_height = max(
                5.0, min(view.half_height * 0.9**wheel_steps, 5000.0)
            )
        else:
            self.camera.process_mouse_scroll(event.angleDelta().y() * 0.01)
        self.update()


class MainWindow(ViewerMainWindow):
    """The existing SkinnedMeshImport application shell with a WebGPU viewport."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        super().__init__(model_path, viewport=SkinWebGPUViewport(model_path))

    def _update_title(self) -> None:
        self.setWindowTitle(
            f"SkinnedMeshImport WebGPU — {self.viewport.model_path.name}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app_type = DebugApplication if args.debug else QApplication
    app = app_type(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("SkinnedMeshImport WebGPU")

    window = MainWindow(Path(args.model))
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoketest**

```bash
uv run SkinnedMeshImport/main_webgpu.py --smoketest
```

Expected: prints `SMOKETEST OK` and exits 0, no traceback.

- [ ] **Step 3: Manual verification**

Use the `run` skill (or run directly) to check, against `uv run SkinnedMeshImport/main.py` side by side for comparison:

```bash
uv run SkinnedMeshImport/main_webgpu.py
```

Confirm all of:
- The guard model appears lit and textured (not flat white, not upside-down — the face/body textures should read right-side-up).
- The animation plays from the timeline transport at the bottom; scrubbing the slider re-poses the mesh.
- `W`/`A`/`S`/`D` + left-mouse-drag fly and look around the perspective pane.
- Pressing `4` splits into the TOP/PERSPECTIVE/FRONT/SIDE four-view; wheel zooms an individual ortho pane; middle/right-drag pans one; clicking a pane maximizes it, clicking again restores the split.
- `Cmd`/`Ctrl`+`O` (File > Open) loads a different mesh from `SkinnedMeshImport/models/` (or another rigged file) without crashing.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check SkinnedMeshImport/main_webgpu.py
uv run ruff format SkinnedMeshImport/main_webgpu.py
git add SkinnedMeshImport/main_webgpu.py
git commit -m "feat(skinned-mesh-import): add the WebGPU viewport and entry point"
```

---

### Task 4: Documentation

**Files:**
- Modify: `SkinnedMeshImport/README.md`
- Modify: `README.md` (repo root)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the WebGPU run instructions**

In `SkinnedMeshImport/README.md`, immediately after the existing intro paragraph and its `uv run SkinnedMeshImport/main.py` code block, insert:

```markdown
There is also a WebGPU version. It has the same timeline, camera and
four-view controls, and File > Open, but skins and renders the mesh with
WebGPU instead:

```bash
uv run SkinnedMeshImport/main_webgpu.py
```
```

- [ ] **Step 2: Add the new files to the file list**

In the existing bullet list (the one starting `- `mesh.py` -- loads the scene...`), add three entries after the existing ones:

```markdown
- `main_webgpu.py` -- the WebGPU viewport and application entry point, mirroring `BVHViewer/main_webgpu.py`'s split from its OpenGL `main.py`
- `webgpu_renderer.py` -- the WebGPU pipeline: five vertex buffers, an index buffer, a bone storage buffer, and one texture bind group per submesh
- `skin_webgpu.wgsl` -- the same four-bones-per-vertex linear blend skin and Blinn-Phong lighting as `shaders/SkinVertex.glsl` / `SkinFragment.glsl`, ported to WGSL
```

- [ ] **Step 3: Add a differences section**

After the existing `## impasse has two struct bugs` section (before `## Tests`), insert:

```markdown
## Differences from the OpenGL version

**No bone-count ceiling.** The OpenGL shader's `gBones[128]` is a fixed-size
GLSL uniform array, so `main.py` warns and leaves extra bones unanimated
past `MAX_BONES = 128`. `skin_webgpu.wgsl` reads the bone palette from a
WGSL storage buffer instead (`var<storage, read> bones: array<mat4x4<f32>>`),
sized to the mesh's actual bone count -- there's no equivalent cap here.

**A second V-flip.** `mesh.py` flips every UV's V coordinate once
(`1.0 - v`), because `ncca.ngl.opengl.Texture` uploads pixels in PIL's
top-row-first order without flipping for OpenGL's bottom-left texture
origin. WebGPU's texture origin is top-left, so that first flip is wrong
here -- `skin_webgpu.wgsl`'s fragment shader flips V back a second time
(`vec2(uv.x, 1.0 - uv.y)`) rather than giving the shared, backend-agnostic
loader a second UV buffer.
```

- [ ] **Step 4: Update the root README**

In `README.md` (repo root), find the `SkinnedMeshImport` table row (currently ending `...skinning it on the GPU (OpenGL)`) and change `(OpenGL)` to `(OpenGL and WebGPU)`.

- [ ] **Step 5: Commit**

```bash
git add SkinnedMeshImport/README.md README.md
git commit -m "docs(skinned-mesh-import): document the WebGPU version"
```

---

### Task 5: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite and lint**

```bash
uv run pytest SkinnedMeshImport/tests
uv run pytest
uv run ruff check SkinnedMeshImport/
uv run ruff format --check SkinnedMeshImport/
```

Expected: everything passes. `uv run pytest` (no path) confirms nothing elsewhere in the repo broke.

- [ ] **Step 2: Both smoketests**

```bash
uv run SkinnedMeshImport/main.py --smoketest
uv run SkinnedMeshImport/main_webgpu.py --smoketest
```

Expected: both print `SMOKETEST OK`.

- [ ] **Step 3: Report to the user**

Summarize what was built, which manual checks were done from Task 3 Step 3, and hand off — this branch (`agent/skinnedmesh-webgpu`) is ready for the user to review and merge via the `finishing-a-development-branch` flow. Do not merge or push without being asked.
