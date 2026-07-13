#!/usr/bin/env -S uv run --script
"""
Uniform Buffer Objects and the std140 padding trap (OpenGL).

One ``SceneBlock`` UBO (VP matrix, light position, light colour) is bound
once, at binding point 0, and read by TWO completely different shader
programs -- a diffuse teapot and a flat-coloured checker grid -- via
``glUniformBlockBinding`` + ``glBindBufferBase``. One ``glBufferSubData``
call per frame updates both draws; nothing is re-uploaded per-shader.

A second UBO, ``MaterialBlock { vec3 albedo; vec3 specularColour;
float shininess; }``, exists to demonstrate the classic std140 trap:
because a vec3 has a 16-byte base alignment, ``specularColour`` is NOT
packed immediately after albedo's 3 floats at byte offset 12 -- the
compiler pushes it to offset 16 (and ``shininess``, a mere float, then
packs at 28, not 32). At startup the demo asks the driver for the linked
program's actual member offsets (``GL_UNIFORM_OFFSET``) and shows them on
the HUD as ground truth. Key ``X`` swaps the CPU-side numpy dtype used to
build the upload buffer between the correct std140 layout and a naive
tightly-packed one; the shader itself never changes, only the bytes fed to
it -- so toggling X visibly corrupts the teapot's specular highlight (the
tight warm highlight smears into a blue-white glare: shininess reads back
as 0 and the scrambled specularColour picks up the CPU-side shininess
value in its blue channel) without touching a single line of GLSL.

    X            toggle MaterialBlock CPU layout: std140-correct / naive
    LMB rotate   RMB pan   wheel zoom   Space reset camera   Esc quit

GL 4.1 core (the macOS ceiling) has no shader storage buffer objects
(SSBOs are GL 4.3+) -- see README.md for what that means and where the
WebGPU half of this demo (``StorageWebGPU.py``) picks up the story with a
runtime-sized ``var<storage, read>`` array of point lights.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from layouts import (
    MATERIAL_BLOCK_BINDING,
    MATERIAL_BLOCK_STD140_DTYPE,
    SCENE_BLOCK_BINDING,
    SCENE_BLOCK_DTYPE,
    naive_bytes_padded_to_std140,
)
from ncca.ngl import Mat3, Mat4, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

DIFFUSE_SHADER = "SceneDiffuse"
GRID_SHADER = "SceneGrid"

GRID_CELLS = 8
GRID_SIZE = 10.0


def build_checker_grid(cells: int, size: float) -> np.ndarray:
    """Interleaved x,y,z,r,g,b for a flat, checker-coloured cells x cells
    grid of quads (2 triangles each) centred on the origin at y = -1."""
    half = size * 0.5
    step = size / cells
    verts = []
    colour_a = (0.85, 0.35, 0.2)
    colour_b = (0.2, 0.35, 0.85)
    for i in range(cells):
        for j in range(cells):
            x0, z0 = -half + i * step, -half + j * step
            x1, z1 = x0 + step, z0 + step
            colour = colour_a if (i + j) % 2 == 0 else colour_b
            corners = [(x0, -1.0, z0), (x1, -1.0, z0), (x1, -1.0, z1), (x0, -1.0, z1)]
            for idx in (0, 1, 2, 0, 2, 3):
                verts.append((*corners[idx], *colour))
    return np.array(verts, dtype=np.float32).reshape(-1)


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """Shared-UBO teapot + checker-grid scene with a togglable padding bug."""

    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0, 0, 0),
        )
        self.view: Mat4 = Mat4()
        self.project: Mat4 = Mat4()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setTitle("UBO / std140 padding trap (OpenGL)")

        self.eye = Vec3(0.0, 3.0, 9.0)
        self.correct_material_layout: bool = True
        self.ubo_scene: int = 0
        self.ubo_material: int = 0
        self.grid_vao = None
        self.material_offsets: dict = {}

    # ------------------------------------------------------------------
    # GL setup
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.12, 0.12, 0.14, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(self.eye, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))

        shader_dir = Path(__file__).parent / "shaders"
        if not ShaderLib.load_shader(
            DIFFUSE_SHADER,
            str(shader_dir / "SceneDiffuseVertex.glsl"),
            str(shader_dir / "SceneDiffuseFragment.glsl"),
        ):
            print("error loading diffuse shader")
            self.close()
        if not ShaderLib.load_shader(
            GRID_SHADER,
            str(shader_dir / "SceneGridVertex.glsl"),
            str(shader_dir / "SceneGridFragment.glsl"),
        ):
            print("error loading grid shader")
            self.close()

        Primitives.load_default_primitives()

        self._create_ubos()
        self._bind_scene_block_to_both_programs()
        self.material_offsets = self._query_material_offsets()
        print(
            f"driver-reported MaterialBlock GL_UNIFORM_OFFSETs: {self.material_offsets}"
        )

        self.grid_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        grid_data = build_checker_grid(GRID_CELLS, GRID_SIZE)
        with self.grid_vao:
            self.grid_vao.set_data(VertexData(data=grid_data, size=grid_data.size // 6))
            stride = 6 * 4
            self.grid_vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, stride, 0)
            self.grid_vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, stride, 12)

        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 18
        )

    def _create_ubos(self) -> None:
        """Two raw GL_UNIFORM_BUFFERs, deliberately NOT going through
        ShaderLib.set_uniform_buffer -- that helper allocates one buffer
        PER SHADER PROGRAM, which would defeat the point of this demo (one
        buffer, one binding point, many programs)."""
        self.ubo_scene = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.ubo_scene)
        gl.glBufferData(
            gl.GL_UNIFORM_BUFFER, SCENE_BLOCK_DTYPE.itemsize, None, gl.GL_DYNAMIC_DRAW
        )
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, SCENE_BLOCK_BINDING, self.ubo_scene)

        self.ubo_material = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.ubo_material)
        gl.glBufferData(
            gl.GL_UNIFORM_BUFFER,
            MATERIAL_BLOCK_STD140_DTYPE.itemsize,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBufferBase(
            gl.GL_UNIFORM_BUFFER, MATERIAL_BLOCK_BINDING, self.ubo_material
        )
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)

    def _bind_scene_block_to_both_programs(self) -> None:
        """The core lesson: a uniform *location* (glGetUniformLocation) is
        per-program and per-variable. A uniform *block binding* is a level
        of indirection above that -- glUniformBlockBinding tells a program
        which numbered GL_UNIFORM_BUFFER slot to read its named block from,
        and glBindBufferBase fills that slot with an actual buffer. Two
        programs pointed at the same slot share the same buffer."""
        diffuse_id = ShaderLib.get_program_id(DIFFUSE_SHADER)
        grid_id = ShaderLib.get_program_id(GRID_SHADER)

        scene_idx = gl.glGetUniformBlockIndex(diffuse_id, "SceneBlock")
        gl.glUniformBlockBinding(diffuse_id, scene_idx, SCENE_BLOCK_BINDING)
        material_idx = gl.glGetUniformBlockIndex(diffuse_id, "MaterialBlock")
        gl.glUniformBlockBinding(diffuse_id, material_idx, MATERIAL_BLOCK_BINDING)

        grid_scene_idx = gl.glGetUniformBlockIndex(grid_id, "SceneBlock")
        gl.glUniformBlockBinding(grid_id, grid_scene_idx, SCENE_BLOCK_BINDING)

    def _query_material_offsets(self) -> dict:
        """Ask the driver where the LINKED program actually placed each
        MaterialBlock member (GL_UNIFORM_OFFSET). This is the ground truth
        the HUD shows next to the toggled CPU-side layouts -- the demo
        verifies its own std140 claims at runtime instead of asserting
        offsets in comments."""
        program = ShaderLib.get_program_id(DIFFUSE_SHADER)
        wanted = ("albedo", "specularColour", "shininess")
        found = {}
        count = gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS)
        for index in range(count):
            name, _size, _type = gl.glGetActiveUniform(program, index)
            name = name.decode()
            if name not in wanted:
                continue
            offset = np.zeros(1, dtype=np.int32)
            gl.glGetActiveUniformsiv(
                program,
                1,
                np.array([index], dtype=np.uint32),
                gl.GL_UNIFORM_OFFSET,
                offset,
            )
            found[name] = int(offset[0])
        return {name: found[name] for name in wanted}

    # ------------------------------------------------------------------
    # per-frame UBO updates
    # ------------------------------------------------------------------
    def _update_scene_block(self, vp: Mat4) -> None:
        data = np.zeros((), dtype=SCENE_BLOCK_DTYPE)
        data["VP"] = vp.to_numpy()
        data["lightPos"] = (4.0, 5.0, 4.0, 1.0)
        data["lightColour"] = (1.0, 0.95, 0.85, 1.0)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.ubo_scene)
        # pitfall: glBufferSubData needs raw bytes, not the structured array
        # itself -- .tobytes() is required.
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, data.nbytes, data.tobytes())

    def _update_material_block(self) -> None:
        albedo = (0.8, 0.65, 0.2)
        specular_colour = (1.0, 0.6, 0.3)
        shininess = 64.0
        if self.correct_material_layout:
            data = np.zeros((), dtype=MATERIAL_BLOCK_STD140_DTYPE)
            data["albedo"] = albedo
            data["specularColour"] = specular_colour
            data["shininess"] = shininess
            payload = data.tobytes()
        else:
            payload = naive_bytes_padded_to_std140(albedo, specular_colour, shininess)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.ubo_material)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, len(payload), payload)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        global_tx = self.scene_global_tx()
        vp = self.project @ self.view @ global_tx
        self._update_scene_block(vp)
        self._update_material_block()

        # ---- grid: unlit, only reads SceneBlock.VP ----
        ShaderLib.use(GRID_SHADER)
        ShaderLib.set_uniform("M", Mat4())
        with self.grid_vao:
            self.grid_vao.draw()

        # ---- teapot: reads SceneBlock (VP/light) + MaterialBlock ----
        ShaderLib.use(DIFFUSE_SHADER)
        model = Mat4()
        mv = self.view @ global_tx @ model
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("M", model)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("viewPos", self.eye)
        Primitives.draw("teapot")

        self._draw_hud()

    def _draw_hud(self) -> None:
        layout_name = (
            "CORRECT std140 (specularColour@16, shininess@28)"
            if self.correct_material_layout
            else "NAIVE packed (specularColour@12, shininess@24) -- WRONG, "
            "colour scrambled + shininess reads 0"
        )
        Text.render_text(
            "Arial",
            10,
            20,
            "SceneBlock (binding 0) shared by SceneDiffuse + SceneGrid programs",
            Vec3(1.0, 1.0, 1.0),
        )
        Text.render_text(
            "Arial",
            10,
            42,
            f"[X] MaterialBlock (binding 1) CPU layout: {layout_name}",
            Vec3(1.0, 1.0, 0.6)
            if self.correct_material_layout
            else Vec3(1.0, 0.4, 0.4),
        )
        driver = " ".join(f"{k}@{v}" for k, v in self.material_offsets.items())
        Text.render_text(
            "Arial",
            10,
            64,
            f"driver GL_UNIFORM_OFFSET ground truth: {driver}",
            Vec3(0.6, 1.0, 0.6),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        Text.set_screen_size(w, h)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_X:
            self.correct_material_layout = not self.correct_material_layout
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by the Qt event loop."""

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
