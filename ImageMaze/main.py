#!/usr/bin/env -S uv run --script
"""Draw an image as a cube maze and move a troll through its white pixels."""

import argparse
import sys
import traceback
from pathlib import Path

import OpenGL.GL as gl
from maze_scene import (
    ActorState,
    Direction,
    Maze,
    actor_forward,
    actor_world_position,
    move_actor,
    top_view,
)
from ncca.ngl import Mat4, Prims, Vec3, logger, look_at, perspective
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

MAPS_DIR = Path(__file__).parent / "maps"


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    def __init__(self, map_path: str | Path) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(0.0, 0.0, 0.0),
        )
        self.setTitle("Image Maze (OpenGL)")
        self.maze = Maze.from_file(map_path)
        self.walls = self.maze.wall_cells()
        self.actor = ActorState(2, 2)
        self.active_camera = 1
        self.wireframe = False
        self.window_width = 1024
        self.window_height = 720
        self.mouse_global_tx = Mat4()
        self.project = Mat4()
        self.top_view = Mat4()
        self.actor_view = Mat4()

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self.top_view = top_view()
        self.project = perspective(
            45.0, self.width() / max(self.height(), 1), 0.5, 50.0
        )
        self._update_actor_camera()

        ShaderLib.use(DefaultShader.COLOUR)
        Primitives.load_default_primitives()
        Primitives.create(
            Prims.TRIANGLE_PLANE,
            "image_maze_ground",
            40.0,
            40.0,
            10,
            10,
            Vec3(0.0, 1.0, 0.0),
        )

    def _update_global_transform(self) -> None:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

    def _update_actor_camera(self) -> None:
        x, y, z = actor_world_position(self.maze, self.actor)
        dx, dy, dz = actor_forward(self.actor)
        self.actor_view = look_at(
            Vec3(x, y, z),
            Vec3(x + dx, y + dy, z + dz),
            Vec3(0.0, 1.0, 0.0),
        )

    def _actor_model(self) -> Mat4:
        x, y, z = actor_world_position(self.maze, self.actor)
        model = Mat4().rotate_y(self.actor.rotation)
        model[3, 0] = x
        model[3, 1] = y
        model[3, 2] = z
        return model

    def _draw_cube(self, model: Mat4, colour) -> None:
        view = self.top_view if self.active_camera == 1 else self.actor_view
        ShaderLib.set_uniform("MVP", self.project @ view @ self.mouse_global_tx @ model)
        ShaderLib.set_uniform("Colour", *colour)
        Primitives.draw(Prims.CUBE)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.window_width, self.window_height)
        ShaderLib.use(DefaultShader.COLOUR)
        self._update_global_transform()

        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        for wall in self.walls:
            self._draw_cube(
                Mat4().translate(wall.x, 0.0, wall.z),
                wall.colour,
            )
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        view = self.top_view if self.active_camera == 1 else self.actor_view
        ShaderLib.set_uniform(
            "MVP",
            self.project @ view @ self.mouse_global_tx @ self._actor_model(),
        )
        ShaderLib.set_uniform("Colour", 1.0, 0.0, 0.0, 1.0)
        Primitives.draw(Prims.TROLL)

        ground = Mat4().translate(0.0, -0.55, 0.0)
        ShaderLib.set_uniform(
            "MVP", self.project @ view @ self.mouse_global_tx @ ground
        )
        ShaderLib.set_uniform("Colour", 0.3, 0.3, 0.3, 1.0)
        Primitives.draw("image_maze_ground")

    def resizeGL(self, width: int, height: int) -> None:
        self.window_width = int(width * self.devicePixelRatio())
        self.window_height = int(height * self.devicePixelRatio())
        self.project = perspective(45.0, width / max(height, 1), 0.5, 50.0)

    def keyPressEvent(self, event) -> None:
        directions = {
            Qt.Key_Up: Direction.NORTH,
            Qt.Key_Down: Direction.SOUTH,
            Qt.Key_Left: Direction.WEST,
            Qt.Key_Right: Direction.EAST,
        }
        key = event.key()
        if key in directions:
            self.actor = move_actor(self.maze, self.actor, directions[key])
            self._update_actor_camera()
        elif key == Qt.Key_1:
            self.active_camera = 1
        elif key == Qt.Key_2:
            self.active_camera = 2
        elif key == Qt.Key_W:
            self.wireframe = not self.wireframe
        elif key == Qt.Key_Space:
            self.reset_camera()
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
            return
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=MAPS_DIR / "small.png",
        help="maze image (default: maps/small.png)",
    )
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow(args.map)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
