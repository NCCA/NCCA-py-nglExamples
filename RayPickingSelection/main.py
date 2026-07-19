#!/usr/bin/env -S uv run --script
"""
Object selection and manipulation demo using CPU ray casting.

The same scene and gizmos as ``SelectionManipulator``, but with the
colour-ID picking replaced by two analytic techniques:

  objects -> a mouse ray is unprojected through inverse(MVP) and intersected
             with each object: a bounding-sphere broad phase, then an exact
             vectorised Moller-Trumbore triangle test. The nearest hit wins.
  gizmo   -> each handle's skeleton is projected to screen space and hit
             with simple 2D point/segment/polyline distance tests.

Compared with the colour-ID version there is no extra ID render pass, no
glReadPixels (and therefore no GPU pipeline stall on click), and picking
returns real hit distances so the nearest object is selected exactly.

Controls (matching Maya):
    Q            select mode (no gizmo)
    W            translate mode
    E            rotate mode
    R            scale mode
    Left click   select object (replaces selection)
    Ctrl+click   toggle object in/out of the selection (multi-select)
    Drag handle  transform every selected object
    Alt+LMB      tumble camera
    Alt+RMB      pan camera
    Wheel        dolly camera
    Space        reset camera
    Escape       quit
"""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import Mat4, Prims, Vec3, Vec4, logger, look_at, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib, Text
from picking_maths import ray_from_screen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from ScreenGizmo import CENTER, Axis, ManipMode, ScreenGizmo
from SelectionObject import (
    CubeObject,
    DodecahedronObject,
    SelectionObject,
    SphereObject,
    TeapotObject,
    TrollObject,
    load_pick_meshes,
)


class MainWindow(QOpenGLWindow):
    def __init__(self, parent: object = None) -> None:
        super().__init__()
        self.setTitle("Ray-Cast Selection and Manipulation")
        self.mouse_global_tx = Mat4()
        self.view = Mat4()
        self.project = Mat4()
        self.model_position = Vec3()
        self.window_width = 1024
        self.window_height = 720

        # camera control state (Alt + mouse, Maya style)
        self.rotate_camera = False
        self.translate_camera = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.mode = ManipMode.TRANSLATE
        self.manipulator = ScreenGizmo()
        self.dragging = False

        self.objects: list[SelectionObject] = [
            TeapotObject(Vec3(-3.0, 0.0, 0.0), Vec4(0.8, 0.2, 0.2, 1.0)),
            CubeObject(Vec3(0.0, 0.5, -3.0), Vec4(0.2, 0.6, 0.8, 1.0)),
            SphereObject(Vec3(3.0, 1.0, 0.0), Vec4(0.9, 0.7, 0.1, 1.0)),
            TrollObject(Vec3(0.0, 0.8, 3.0), Vec4(0.6, 0.3, 0.7, 1.0)),
            DodecahedronObject(Vec3(0.0, 0.6, 0.0), Vec4(0.2, 0.7, 0.4, 1.0)),
        ]

    # selection helpers
    def selected_objects(self) -> list[SelectionObject]:
        return [o for o in self.objects if o.selected]

    def update_pivot(self) -> None:
        """Place the gizmo at the centroid of the selection."""
        selected = self.selected_objects()
        if not selected:
            return
        centre = Vec3(0.0, 0.0, 0.0)
        for obj in selected:
            centre += obj.position
        self.manipulator.position = centre * (1.0 / len(selected))

    # GL setup / drawing
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(
            Vec3(0.0, 8.0, 14.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 0.0, 4.0, 4.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "selSphere", 1.0, 40)
        Primitives.create(Prims.LINE_GRID, "grid", 20.0, 20.0, 20)
        ScreenGizmo.create_geometry()
        # CPU triangle cache used by the ray caster (the GPU primitives
        # above are only ever drawn, never read back)
        load_pick_meshes()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 20
        )

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
        self.mouse_global_tx = self.scene_global_tx()

        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ self.mouse_global_tx)
        ShaderLib.set_uniform("Colour", 0.6, 0.6, 0.6, 1.0)
        Primitives.draw("grid")

        for obj in self.objects:
            obj.draw(self.mouse_global_tx, self.view, self.project)

        if self.selected_objects():
            self.manipulator.draw(
                self.mode, self.mouse_global_tx, self.view, self.project
            )

        num = len(self.selected_objects())
        Text.render_text(
            "Arial",
            10,
            20,
            f"Mode [{self.mode.value}]  selected {num}   "
            "Q select W translate E rotate R scale | click select, ctrl+click multi, alt+mouse camera",
            Vec3(1.0, 1.0, 1.0),
        )

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
        Text.set_screen_size(w, h)

    # picking
    def pick(self, x: float, y: float):
        """Return ('axis', Axis) or ('object', obj) or None.

        x, y are in device pixels with Qt's top-left origin. No rendering
        happens here: the gizmo is tested with 2D screen-space distances and
        the objects with a CPU ray cast.
        """
        global_tx = self.scene_global_tx()

        # gizmo handles win over objects so they stay grabbable in front of
        # geometry (same priority the colour-ID demo gives them)
        if self.selected_objects():
            axis = self.manipulator.pick_handle(
                x,
                y,
                self.mode,
                global_tx,
                self.view,
                self.project,
                self.window_width,
                self.window_height,
            )
            if axis is not None:
                return ("axis", axis)

        # folding the scene's global transform into the unproject matrix
        # gives a ray in scene space, so each object only needs its own
        # local transform to test against
        mvp = (self.project @ self.view @ global_tx).to_numpy()
        origin, direction = ray_from_screen(
            x, y, self.window_width, self.window_height, mvp
        )
        nearest = None
        nearest_t = float("inf")
        for obj in self.objects:
            t = obj.intersect(origin, direction)
            if t is not None and t < nearest_t:
                nearest = obj
                nearest_t = t
        if nearest is not None:
            return ("object", nearest)
        return None

    # events
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Q:
            self.mode = ManipMode.SELECT
        elif key == Qt.Key_W:
            self.mode = ManipMode.TRANSLATE
        elif key == Qt.Key_E:
            self.mode = ManipMode.ROTATE
        elif key == Qt.Key_R:
            self.mode = ManipMode.SCALE
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0.0, 0.0, 0.0)
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        modifiers = event.modifiers()

        # Alt + mouse drives the camera, Maya style
        if modifiers & Qt.AltModifier:
            if event.button() == Qt.LeftButton:
                self.original_x_rotation = position.x()
                self.original_y_rotation = position.y()
                self.rotate_camera = True
            elif event.button() == Qt.RightButton:
                self.original_x_pos = position.x()
                self.original_y_pos = position.y()
                self.translate_camera = True
            return

        if event.button() != Qt.LeftButton:
            return

        dpr = self.devicePixelRatio()
        hit = self.pick(position.x() * dpr, position.y() * dpr)

        if hit is not None and hit[0] == "axis" and self.mode != ManipMode.SELECT:
            self.manipulator.start_drag(
                hit[1],
                position.x() * dpr,
                position.y() * dpr,
                self.scene_global_tx(),
                self.view,
                self.project,
                self.window_width,
                self.window_height,
            )
            self.dragging = True
        elif hit is not None and hit[0] == "object":
            obj = hit[1]
            if modifiers & Qt.ControlModifier:
                obj.selected = not obj.selected
            else:
                for other in self.objects:
                    other.selected = False
                obj.selected = True
            self.update_pivot()
        elif not modifiers & Qt.ControlModifier:
            for obj in self.objects:
                obj.selected = False
        self.update()

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        if self.dragging:
            dpr = self.devicePixelRatio()
            x, y = position.x() * dpr, position.y() * dpr
            if self.mode == ManipMode.TRANSLATE:
                if self.manipulator.active_axis == CENTER:
                    delta = self.manipulator.drag_free_translate(x, y)
                else:
                    delta = self.manipulator.drag_translate(x, y)
                for obj in self.selected_objects():
                    obj.position += delta
                self.manipulator.position += delta
            elif self.mode == ManipMode.SCALE:
                if self.manipulator.active_axis == CENTER:
                    factors = self.manipulator.drag_uniform_scale(x, y)
                else:
                    factors = self.manipulator.drag_scale(x, y)
                for obj in self.selected_objects():
                    obj.scale = Vec3(
                        obj.scale.x * factors.x,
                        obj.scale.y * factors.y,
                        obj.scale.z * factors.z,
                    )
            elif self.mode == ManipMode.ROTATE:
                angle = self.manipulator.drag_rotate(x, y)
                axis = self.manipulator.active_axis
                if axis is not None:
                    for obj in self.selected_objects():
                        rot = obj.rotation
                        if axis == Axis.X:
                            rot.x += angle
                        elif axis == Axis.Y:
                            rot.y += angle
                        else:
                            rot.z += angle
            self.update()
        elif self.rotate_camera and event.buttons() == Qt.LeftButton:
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate_camera and event.buttons() == Qt.RightButton:
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self.dragging:
                self.dragging = False
                self.manipulator.end_drag()
            self.rotate_camera = False
        elif event.button() == Qt.RightButton:
            self.translate_camera = False
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions swallowed by Qt event handlers."""

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    format = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
