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
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from impasse.errors import AssimpError
from mesh import MAX_BONES_PER_VERTEX, SkinnedMesh
from MultiBufferIndexVAO import MultiBufferIndexVAO
from ncca.ngl import Mat4, Vec3, Vec4, logger, look_at, perspective
from ncca.ngl.opengl import (
    PySideEventHandlingMixin,
    ShaderLib,
    Texture,
    VAOFactory,
    VertexData,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QSurfaceFormat
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
# Matches shaders/SkinVertex.glsl's `const int MAX_BONES`. A mesh with more
# bones than this still loads and draws -- the extra bones are just left
# unanimated, since there's no gBones[] uniform slot to put them in.
MAX_BONES = 128
MESH_FILE_FILTER = (
    "Rigged meshes (*.md5mesh *.dae *.fbx *.gltf *.glb *.x *.bvh *.obj);;All files (*)"
)

_APP_STYLE = """
QMainWindow, QWidget { background: #34373b; color: #dedede; }
#timelinePanel { background: #292c30; border-top: 1px solid #17191b; }
"""


class SkinViewport(PySideEventHandlingMixin, QOpenGLWindow):
    """The OpenGL viewport: loads the mesh, skins it on the GPU, and draws it."""

    def __init__(self, model_path: Path, parent: object = None) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.5,
            zoom_sensitivity=0.5,
            initial_position=Vec3(0, 0, 0),
        )
        self.setTitle("Skinned Mesh Import")
        self.window_width = 1024
        self.window_height = 720
        self.model_path = model_path
        # Loading the scene (impasse) needs no GL context, so it happens
        # eagerly here rather than in initializeGL -- MainWindow needs
        # mesh.duration()/ticks_per_second() to set up the timeline right
        # away, before the window is even shown.
        self.mesh = SkinnedMesh(str(model_path))
        self.view = Mat4()
        self.project = Mat4()
        self.current_frame = 0
        self.bone_transforms: list[Mat4] = []
        self._texture_ids: dict[str, int] = {}
        self._fallback_texture_id: int | None = None

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

        self._build_vao()
        self._load_textures()
        self._frame_camera()
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
        self._build_vao()
        self._load_textures()
        self._frame_camera()
        self.set_frame(0)

    def _frame_camera(self) -> None:
        bbox_min, bbox_max = self.mesh.bounding_box()
        centre = Vec3(
            (bbox_min[0] + bbox_max[0]) * 0.5,
            (bbox_min[1] + bbox_max[1]) * 0.5,
            (bbox_min[2] + bbox_max[2]) * 0.5,
        )
        # MD5 (idTech) is always Z-up; every other format this demo can
        # load ends up Y-up in its post-import world space. There's no
        # reliable way to tell up from bounding-box shape alone -- a
        # character posed with an arm held out (like this demo's own
        # guard model) can be wider than it is tall -- so this keys off
        # the one format whose up axis is a fixed, known property rather
        # than guessing per-asset.
        if self.model_path.suffix.lower() == ".md5mesh":
            height = bbox_max[2] - bbox_min[2]
            eye = Vec3(centre.x, bbox_min[1] - height * 1.5, centre.z)
            up = Vec3(0.0, 0.0, 1.0)
        else:
            height = bbox_max[1] - bbox_min[1]
            eye = Vec3(centre.x, centre.y, bbox_max[2] + height * 1.5)
            up = Vec3(0.0, 1.0, 0.0)
        self.view = look_at(eye, centre, up)
        self.eye = eye

    def set_frame(self, frame: int) -> None:
        """Pose the mesh at the given timeline frame and request a repaint."""
        self.current_frame = frame
        time_seconds = frame / self.mesh.ticks_per_second()
        self.bone_transforms = self.mesh.bone_transforms(time_seconds)
        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / max(h, 1), 0.5, 500.0)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if not hasattr(self, "vao"):
            return  # initializeGL hasn't run yet
        ShaderLib.use(SKIN_SHADER)

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        mouse_global_tx = rot_y @ rot_x
        mouse_global_tx[3, 0] = self.model_position.x
        mouse_global_tx[3, 1] = self.model_position.y
        mouse_global_tx[3, 2] = self.model_position.z

        m = mouse_global_tx
        mv = self.view @ m
        mvp = self.project @ mv
        ShaderLib.set_uniform("M", m)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("viewerPos", self.eye)

        light_eye = self.view @ Vec4(self.eye.x, self.eye.y, self.eye.z, 1.0)
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


class MainWindow(QMainWindow):
    """The skinned-mesh viewport plus an animation transport underneath."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        super().__init__()
        self.setWindowTitle("SkinnedMeshImport")
        self.resize(1100, 780)
        self.setStyleSheet(_APP_STYLE)

        self.viewport = SkinViewport(model_path)
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
