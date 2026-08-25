#!/usr/bin/env -S uv run --script
"""Import a rigged mesh with impasse and animate it with GPU linear blend skinning.

A PyNGL port of the NGL9Demos ``AssetImportDemos/SkeletalAnimation`` tutorial
(itself based on ogldev's assimp skinning tutorial), swapping NGL's C++
assimp bindings for the Python ``impasse`` package. See ``mesh.py`` for the
loader and skinning maths, and its module docstring for a real bug in
impasse's bundled ``aiBone`` struct definition that this demo works around.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import OpenGL.GL as gl
from impasse.errors import AssimpError
from mesh import MAX_BONES_PER_VERTEX, SkinnedMesh
from MultiBufferIndexVAO import MultiBufferIndexVAO
from ncca.ngl import FirstPersonCamera, Mat4, Vec3, Vec4, logger, look_at, ortho
from ncca.ngl.opengl import (
    ShaderLib,
    Text,
    Texture,
    VAOFactory,
    VertexData,
)
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QKeySequence,
    QMouseEvent,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from timeline import TimelineWidget

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = DEMO_DIR / "models" / "guard" / "boblampclean.md5mesh"

SKIN_SHADER = "Skin"
VAO_NAME = "SkinnedMeshVAO"
FONT_NAME = "SkinnedMeshImport"
# Matches shaders/SkinVertex.glsl's `const int MAX_BONES`. A mesh with more
# bones than this still loads and draws -- the extra bones are just left
# unanimated, since there's no gBones[] uniform slot to put them in.
MAX_BONES = 128
MESH_FILE_FILTER = (
    "Rigged meshes (*.md5mesh *.dae *.fbx *.gltf *.glb *.x *.bvh *.obj);;All files (*)"
)

TOP_VIEW = 0
PERSPECTIVE_VIEW = 1
FRONT_VIEW = 2
SIDE_VIEW = 3
_VIEW_LABELS = ("TOP", "PERSPECTIVE", "FRONT", "SIDE")

_APP_STYLE = """
QMainWindow, QWidget { background: #34373b; color: #dedede; }
#timelinePanel { background: #292c30; border-top: 1px solid #17191b; }
"""


@dataclass
class OrthoView:
    """Zoom and pan state for one orthographic pane, independent of the others."""

    eye: Vec3
    target: Vec3
    up: Vec3
    right: Vec3
    half_height: float = 40.0
    pan: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def matrices(self, pane_width: int, pane_height: int) -> tuple[Mat4, Mat4]:
        """Return this pane's current (view, projection) matrices."""
        half_width = self.half_height * pane_width / max(pane_height, 1)
        project = ortho(
            -half_width, half_width, -self.half_height, self.half_height, 0.05, 5000.0
        )
        view = look_at(self.eye + self.pan, self.target + self.pan, self.up)
        return view, project

    def pan_by(self, dx_pixels: float, dy_pixels: float, pane_height: float) -> None:
        """Shift this pane's pan offset by a screen-space drag delta."""
        units_per_pixel = (2.0 * self.half_height) / max(pane_height, 1.0)
        self.pan = (
            self.pan
            + self.right * (-dx_pixels * units_per_pixel)
            + self.up * (dy_pixels * units_per_pixel)
        )


class SkinViewport(QOpenGLWindow):
    """The OpenGL viewport: loads the mesh, skins it on the GPU, and draws it.

    Camera/navigation mirrors ``BVHViewer/main.py``'s ``BvhViewport``: a
    ``FirstPersonCamera`` (WASD + mouse-look) for the perspective view, plus
    an optional four-pane TOP/PERSPECTIVE/FRONT/SIDE split with independent
    pan/zoom orthographic panes, click-to-maximize.
    """

    _MOVE_KEYS = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D}
    _DIVIDER_WIDTH = 2

    def __init__(self, model_path: Path, parent: object = None) -> None:
        super().__init__()
        self.setTitle("Skinned Mesh Import")
        self.window_width = 1024
        self.window_height = 720
        self.model_path = model_path
        # Loading the scene (impasse) needs no GL context, so it happens
        # eagerly here rather than in initializeGL -- MainWindow needs
        # mesh.duration()/ticks_per_second() to set up the timeline right
        # away, before the window is even shown.
        self.mesh = SkinnedMesh(str(model_path))
        self.current_frame = 0
        self.bone_transforms: list[Mat4] = []
        self._texture_ids: dict[str, int] = {}
        self._fallback_texture_id: int | None = None
        self._text_ready = False

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

    # -------------------------------------------------------- camera setup

    def _compute_view_setup(self) -> None:
        """(Re)build the model-axis correction, camera and ortho panes for the current mesh.

        MD5 (idTech) is always Z-up; every other format this demo can load
        ends up Y-up in its post-import world space -- keyed off the
        ``.md5mesh`` extension specifically (a fixed, known property of that
        format) rather than guessed from the bounding box, since a
        character posed with a limb held out (like this demo's own guard,
        holding its lamp arm out) can be wider than it is tall.

        ``FirstPersonCamera``'s yaw/pitch parameterisation always treats Y
        as vertical, regardless of the ``up`` passed to its constructor,
        so a Z-up mesh is presented to the camera (and to lighting) already
        rotated into Y-up via ``self.model_matrix`` -- applied once here as
        a constant, uniformly, in ``_draw_mesh`` -- rather than teaching
        the camera two different "up" conventions.
        """
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
        # Vec3 components are numpy float32 scalars (its `_data` is a numpy
        # array internally); Vec3.__mul__ rejects anything but a native
        # Python int/float, so this needs an explicit cast at the source --
        # otherwise it silently poisons camera.speed and every value
        # derived from it below, only surfacing as a crash once WASD/mouse
        # movement actually multiplies a Vec3 by it.
        height = float(max(hi.y - lo.y, 0.001))
        distance = height * 2.5

        eye = Vec3(centre.x, centre.y, hi.z + height * 1.5)
        self.camera = FirstPersonCamera(eye, centre, Vec3(0.0, 1.0, 0.0), 45.0)
        self.camera.speed = height * 0.4
        self._look_at(eye, centre)

        half_height = height * 0.75
        self.ortho_views: dict[int, OrthoView] = {
            TOP_VIEW: OrthoView(
                eye=Vec3(centre.x, hi.y + distance, centre.z),
                target=centre,
                up=Vec3(0.0, 0.0, -1.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            FRONT_VIEW: OrthoView(
                eye=Vec3(centre.x, centre.y, hi.z + distance),
                target=centre,
                up=Vec3(0.0, 1.0, 0.0),
                right=Vec3(1.0, 0.0, 0.0),
                half_height=half_height,
            ),
            SIDE_VIEW: OrthoView(
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

    # ------------------------------------------------------------- OpenGL

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.25, 0.25, 0.28, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        VAOFactory.register_vao_creator(VAO_NAME, MultiBufferIndexVAO)

        if not ShaderLib.load_shader(
            SKIN_SHADER,
            vert="shaders/SkinVertex.glsl",
            frag="shaders/SkinFragment.glsl",
        ):
            raise RuntimeError("Failed to load skinning shader")

        Text.add_font(FONT_NAME, str(DEMO_DIR.parent / "font" / "Arial.ttf"), 18)
        Text.set_screen_size(self.window_width, self.window_height)
        self._text_ready = True

        self._build_vao()
        self._load_textures()
        self.set_frame(0)

    def _build_vao(self) -> None:
        mesh = self.mesh
        self.vao = VAOFactory.create_vao(VAO_NAME, gl.GL_TRIANGLES)
        with self.vao:
            self.vao.set_data(
                VertexData(data=mesh.positions.flatten(), size=len(mesh.positions))
            )
            self.vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 0, 0)

            self.vao.set_data(
                VertexData(data=mesh.texcoords.flatten(), size=len(mesh.texcoords))
            )
            self.vao.set_vertex_attribute_pointer(1, 2, gl.GL_FLOAT, 0, 0)

            self.vao.set_data(
                VertexData(data=mesh.normals.flatten(), size=len(mesh.normals))
            )
            self.vao.set_vertex_attribute_pointer(2, 3, gl.GL_FLOAT, 0, 0)

            self.vao.set_data(
                VertexData(data=mesh.bone_ids.flatten(), size=len(mesh.bone_ids))
            )
            self.vao.set_vertex_attribute_pointer(
                3, MAX_BONES_PER_VERTEX, gl.GL_FLOAT, 0, 0
            )

            self.vao.set_data(
                VertexData(
                    data=mesh.bone_weights.flatten(), size=len(mesh.bone_weights)
                )
            )
            self.vao.set_vertex_attribute_pointer(
                4, MAX_BONES_PER_VERTEX, gl.GL_FLOAT, 0, 0
            )

            self.vao.set_indices(mesh.indices.tolist(), gl.GL_UNSIGNED_INT)

    def _load_textures(self) -> None:
        for submesh in self.mesh.submeshes:
            path = submesh.texture_path
            if path is None or path in self._texture_ids:
                continue
            try:
                texture_id = Texture(path).set_texture_gl()
                if texture_id == 0:
                    raise RuntimeError("empty image")
            except Exception as error:
                # Test assets pulled in from elsewhere sometimes reference a
                # texture that was never shipped alongside them -- keep the
                # mesh visible (flat white) rather than losing the whole load.
                logger.warning(
                    f"Could not load texture {path!r} ({error}); using a flat fallback"
                )
                texture_id = self._fallback_texture()
            self._texture_ids[path] = texture_id

    def _fallback_texture(self) -> int:
        if self._fallback_texture_id is None:
            self._fallback_texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._fallback_texture_id)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                1,
                1,
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                bytes([255, 255, 255]),
            )
        return self._fallback_texture_id

    def load_model(self, model_path: Path) -> None:
        """Replace the current mesh with the one at ``model_path``.

        Raises whatever ``SkinnedMesh`` raises (bad file, no animation
        track, ...) without touching the currently-loaded mesh -- the
        caller is expected to catch that and report it.
        """
        new_mesh = SkinnedMesh(str(model_path))
        if len(new_mesh.bone_names) > MAX_BONES:
            logger.warning(
                f"{model_path.name} has {len(new_mesh.bone_names)} bones, "
                f"more than the shader's MAX_BONES ({MAX_BONES}) -- the "
                "extras will be unanimated"
            )

        self.makeCurrent()
        if hasattr(self, "vao"):
            self.vao.remove_vao()
        if self._texture_ids:
            gl.glDeleteTextures(list(set(self._texture_ids.values())))
        self._texture_ids = {}
        self._fallback_texture_id = None

        self.model_path = model_path
        self.mesh = new_mesh
        self._compute_view_setup()
        self._build_vao()
        self._load_textures()
        self.set_frame(0)

    def set_frame(self, frame: int) -> None:
        """Pose the mesh at the given timeline frame and request a repaint."""
        self.current_frame = frame
        time_seconds = frame / self.mesh.ticks_per_second()
        self.bone_transforms = self.mesh.bone_transforms(time_seconds)
        self.update()

    # ------------------------------------------------------------ drawing

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        if self._text_ready:
            Text.set_screen_size(self.window_width, self.window_height)
        if self.four_view:
            if self._maximized_pane is not None:
                pane_width, pane_height = self.window_width, self.window_height
            else:
                _, _, pane_width, pane_height = self._four_view_rectangles()[0]
            aspect = pane_width / max(pane_height, 1)
        else:
            aspect = self.window_width / max(self.window_height, 1)
        self._set_perspective_projection(aspect)

    def _set_perspective_projection(self, aspect: float) -> None:
        # FirstPersonCamera.set_projection() returns a Mat4 but never
        # stores it -- only the constructor assigns self._projection, so
        # this has to be reassigned by hand on every resize/zoom.
        self.camera.aspect = aspect
        self.camera.near = 0.05
        self.camera.far = 5000.0
        self.camera._projection = self.camera.set_projection(
            self.camera.zoom, self.camera.aspect, self.camera.near, self.camera.far
        )

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if not hasattr(self, "vao"):
            return  # initializeGL hasn't run yet

        frame_time = self._frame_timer.elapsed() * 0.001
        delta_time = min(max(frame_time - self._last_frame_time, 0.0), 0.05)
        self._last_frame_time = frame_time
        self._advance_camera(delta_time)

        if self.four_view:
            gl.glEnable(gl.GL_SCISSOR_TEST)
            for rectangle, view, project, eye in self._pane_draws():
                gl.glViewport(*rectangle)
                gl.glScissor(*rectangle)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                self._draw_mesh(view, project, eye)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            self._draw_view_labels()
        else:
            gl.glViewport(0, 0, self.window_width, self.window_height)
            self._draw_mesh(self.camera.view, self.camera.projection, self.camera.eye)

        if self.keys_pressed & self._MOVE_KEYS:
            self.update()

    def _draw_mesh(self, view: Mat4, project: Mat4, eye: Vec3) -> None:
        ShaderLib.use(SKIN_SHADER)

        m = self.model_matrix
        mv = view @ m
        mvp = project @ mv
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("viewerPos", eye)

        light_eye = view @ Vec4(eye.x, eye.y, eye.z, 1.0)
        ShaderLib.set_uniform(
            "light.position", light_eye.x, light_eye.y, light_eye.z, 1.0
        )
        ShaderLib.set_uniform("light.ambient", 0.2, 0.2, 0.2, 1.0)
        ShaderLib.set_uniform("light.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("light.specular", 0.8, 0.8, 0.8, 1.0)
        ShaderLib.set_uniform("material.ambient", 0.2, 0.2, 0.2, 1.0)
        ShaderLib.set_uniform("material.diffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("material.specular", 0.4, 0.4, 0.4, 1.0)
        ShaderLib.set_uniform("material.shininess", 32.0)

        for index, transform in enumerate(self.bone_transforms[:MAX_BONES]):
            ShaderLib.set_uniform(f"gBones[{index}]", transform)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        ShaderLib.set_uniform("diffuseTexture", 0)
        with self.vao:
            for submesh in self.mesh.submeshes:
                texture_id = self._texture_ids.get(submesh.texture_path, 0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
                self.vao.draw(submesh.index_offset, submesh.index_count)

    # --------------------------------------------------------- four-view

    def set_four_view(self, enabled: bool) -> None:
        self.four_view = enabled
        self._maximized_pane = None
        self.resizeGL(
            round(self.window_width / self.devicePixelRatio()),
            round(self.window_height / self.devicePixelRatio()),
        )
        self.update()

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
        self.resizeGL(
            round(self.window_width / self.devicePixelRatio()),
            round(self.window_height / self.devicePixelRatio()),
        )
        self.update()

    def _four_view_rectangles(self) -> list[tuple[int, int, int, int]]:
        divider = self._DIVIDER_WIDTH
        left_width = max(1, (self.window_width - divider) // 2)
        bottom_height = max(1, (self.window_height - divider) // 2)
        right_x = left_width + divider
        top_y = bottom_height + divider
        right_width = max(1, self.window_width - right_x)
        top_height = max(1, self.window_height - top_y)
        return [
            (0, top_y, left_width, top_height),
            (right_x, top_y, right_width, top_height),
            (0, 0, left_width, bottom_height),
            (right_x, 0, right_width, bottom_height),
        ]

    def _pane_rectangles(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Return (pane_index, rectangle) for every pane currently being drawn."""
        if self._maximized_pane is not None:
            return [
                (self._maximized_pane, (0, 0, self.window_width, self.window_height))
            ]
        return list(enumerate(self._four_view_rectangles()))

    def _pane_index_at(self, x: float, y: float) -> int | None:
        """Return the pane index under a window-local logical-pixel position."""
        if not self.four_view:
            return None
        ratio = self.devicePixelRatio()
        device_x = x * ratio
        device_y = y * ratio
        for index, (rx, ry, rw, rh) in self._pane_rectangles():
            top = self.window_height - (ry + rh)
            if rx <= device_x < rx + rw and top <= device_y < top + rh:
                return index
        return None

    def _pane_draws(self) -> list[tuple[tuple[int, int, int, int], Mat4, Mat4, Vec3]]:
        draws = []
        for index, rectangle in self._pane_rectangles():
            if index == PERSPECTIVE_VIEW:
                draws.append(
                    (
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
                    (rectangle, view, project, ortho_view.eye + ortho_view.pan)
                )
        return draws

    def _draw_view_labels(self) -> None:
        gl.glViewport(0, 0, self.window_width, self.window_height)
        for index, (x, y, _, height) in self._pane_rectangles():
            screen_y = self.window_height - (y + height)
            Text.render_text(
                FONT_NAME,
                x + 10,
                screen_y + 20,
                _VIEW_LABELS[index],
                Vec3(0.82, 0.84, 0.86),
            )

    def _is_perspective_position(self, x: float, y: float) -> bool:
        if not self.four_view:
            return True
        return self._pane_index_at(x, y) == PERSPECTIVE_VIEW

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

    def keyPressEvent(self, event) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.add(event.key())
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() in self._MOVE_KEYS:
            if not event.isAutoRepeat():
                self.keys_pressed.discard(event.key())
            self.update()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            if not self._is_perspective_position(position.x(), position.y()):
                self._rotating_camera = False
                return
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            self._rotating_camera = True
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            position = event.position()
            index = self._pane_index_at(position.x(), position.y())
            if index is not None and index != PERSPECTIVE_VIEW:
                self._panning_pane = index
                self._last_mouse_x = position.x()
                self._last_mouse_y = position.y()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._rotating_camera and event.buttons() & Qt.MouseButton.LeftButton:
            position = event.position()
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
            position = event.position()
            diff_x = position.x() - self._last_mouse_x
            diff_y = position.y() - self._last_mouse_y
            self._last_mouse_x = position.x()
            self._last_mouse_y = position.y()
            _, _, _, pane_height = dict(self._pane_rectangles())[self._panning_pane]
            pane_height_logical = pane_height / self.devicePixelRatio()
            self.ortho_views[self._panning_pane].pan_by(
                diff_x, diff_y, pane_height_logical
            )
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._rotating_camera = False
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
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


class MainWindow(QMainWindow):
    """The skinned-mesh viewport plus an animation transport underneath."""

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

        self.timeline = TimelineWidget(self)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(viewport_widget, 1)
        layout.addWidget(self.timeline)
        self.setCentralWidget(central)

        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._playing = True

        self.timeline.frame_requested.connect(self._seek_frame)
        self.timeline.playback_range_changed.connect(lambda *_: None)
        self.timeline.fps_changed.connect(self._set_playback_fps)
        self.timeline.first_requested.connect(self._go_to_first_frame)
        self.timeline.previous_requested.connect(self._step_backward)
        self.timeline.play_toggled.connect(self._toggle_playback)
        self.timeline.next_requested.connect(self._step_forward)
        self.timeline.last_requested.connect(self._go_to_last_frame)

        self._create_menu()
        self._sync_timeline_to_mesh()
        self._set_playing(True)
        self._update_title()

    def _create_menu(self) -> None:
        self.menuBar().setNativeMenuBar(False)
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open Mesh...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.four_view_action = QAction("Four Views", self)
        self.four_view_action.setCheckable(True)
        self.four_view_action.setShortcut(QKeySequence(Qt.Key.Key_4))
        self.four_view_action.toggled.connect(self.viewport.set_four_view)
        view_menu.addAction(self.four_view_action)

    def _open_file_dialog(self) -> None:
        start_dir = str(self.viewport.model_path.parent)
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Rigged Mesh", start_dir, MESH_FILE_FILTER
        )
        if not filename:
            return
        self._set_playing(False)
        try:
            self.viewport.load_model(Path(filename))
        # impasse.errors.AssimpError subclasses BaseException, not Exception,
        # so a plain `except Exception` misses the exact case this exists
        # for -- a file assimp can't import.
        except (Exception, AssimpError) as error:
            QMessageBox.critical(
                self,
                "Unable to load mesh",
                f"Could not load {Path(filename).name}.\n\n{error}",
            )
            return
        self._sync_timeline_to_mesh()
        self._set_playing(True)
        self._update_title()

    def _update_title(self) -> None:
        self.setWindowTitle(f"SkinnedMeshImport — {self.viewport.model_path.name}")

    def _sync_timeline_to_mesh(self) -> None:
        mesh = self.viewport.mesh
        frame_count = int(round(mesh.duration())) + 1
        frame_time = 1.0 / mesh.ticks_per_second()
        self.timeline.set_clip(frame_count, frame_time)
        self._set_playback_fps(self.timeline.playback_fps())

    def _set_playing(self, playing: bool) -> None:
        self._playing = playing
        if playing:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()
        self.timeline.set_playing(playing)

    def _toggle_playback(self) -> None:
        self._set_playing(not self._playing)

    def _set_playback_fps(self, fps: float) -> None:
        bounded_fps = max(1.0, min(float(fps), 240.0))
        self._playback_timer.setInterval(max(1, round(1000.0 / bounded_fps)))

    def _seek_frame(self, frame: int) -> None:
        self._set_playing(False)
        self.viewport.set_frame(frame)
        self.timeline.set_frame(frame)

    def _go_to_first_frame(self) -> None:
        start, _ = self.timeline.playback_range()
        self._seek_frame(start)

    def _go_to_last_frame(self) -> None:
        _, end = self.timeline.playback_range()
        self._seek_frame(end)

    def _step_forward(self) -> None:
        start, end = self.timeline.playback_range()
        target = min(self.viewport.current_frame + 1, end)
        if self.viewport.current_frame >= end:
            target = start
        self._seek_frame(target)

    def _step_backward(self) -> None:
        start, end = self.timeline.playback_range()
        target = max(self.viewport.current_frame - 1, start)
        if self.viewport.current_frame <= start:
            target = end
        self._seek_frame(target)

    def _advance_playback(self) -> None:
        start, end = self.timeline.playback_range()
        target = self.viewport.current_frame + 1
        if target > end:
            target = start
        self.viewport.set_frame(target)
        self.timeline.set_frame(target)

    def closeEvent(self, event) -> None:
        self._playback_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)

    def notify(self, receiver, event) -> bool:
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


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
