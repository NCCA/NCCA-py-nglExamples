"""OpenGL instanced-particle version of the abstract octree demo."""

import argparse
import ctypes
import sys
import traceback
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, PrimData, Vec3, logger, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    PySideEventHandlingMixin,
    ShaderLib,
    Text,
    VAOFactory,
    VAOType,
    VertexData,
)
from octree import ParticleSystem, bounds_lines
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(PySideEventHandlingMixin, QOpenGLWindow):
    """CPU octree simulation rendered with one instanced sphere draw."""

    def __init__(self, grid_size: int = 10, seed: int = 7) -> None:
        super().__init__()
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.01,
            zoom_sensitivity=0.1,
            initial_position=Vec3(),
        )
        self.setTitle("OctreeAbstract (OpenGL)")
        self.window_width = 1024
        self.window_height = 720
        self.project = Mat4()
        self.view = look_at(Vec3(0, 0, 35), Vec3(), Vec3(0, 1, 0))
        self.grid_size = grid_size
        self.seed = seed
        self.system = ParticleSystem.grid(grid_size, seed)
        self.animate = True
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setInterval(50)
        self.simulation_timer.timeout.connect(self._tick)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        shader_dir = Path(__file__).parent / "shaders"
        ShaderLib.load_shader(
            "OctreeParticles",
            str(shader_dir / "ParticleVertex.glsl"),
            str(shader_dir / "ParticleFragment.glsl"),
        )
        self._create_particle_vao()
        self._create_bounds_vao()
        Text.add_font(
            "Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 18
        )
        self.simulation_timer.start()

    def _create_particle_vao(self) -> None:
        sphere = PrimData.sphere(1.0, 12)
        self.sphere_vertex_count = len(sphere)
        self.particle_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.particle_vao)

        self.sphere_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.sphere_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sphere.nbytes, sphere, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * 4, ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4)
        )

        self.instance_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        self._update_instance_buffer()
        for location, offset in ((2, 0), (3, 4 * 4)):
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(
                location,
                4,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                8 * 4,
                ctypes.c_void_p(offset),
            )
            gl.glVertexAttribDivisor(location, 1)
        gl.glBindVertexArray(0)

    def _instance_data(self) -> np.ndarray:
        data = np.ones((self.system.count, 8), dtype=np.float32)
        data[:, :3] = self.system.positions
        data[:, 3] = self.system.radii
        data[:, 4:7] = self.system.colours
        return data

    def _update_instance_buffer(self) -> None:
        self.instances = self._instance_data()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.instances.nbytes,
            self.instances,
            gl.GL_DYNAMIC_DRAW,
        )

    def _create_bounds_vao(self) -> None:
        lines = bounds_lines()
        self.bounds_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINES)
        with self.bounds_vao:
            self.bounds_vao.set_data(VertexData(lines, len(lines)))
            self.bounds_vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            self.bounds_vao.set_num_indices(len(lines))

    def _global_transform(self) -> Mat4:
        transform = Mat4().rotate_y(self.spin_y_face) @ Mat4().rotate_x(
            self.spin_x_face
        )
        transform[3, 0] = self.model_position.x
        transform[3, 1] = self.model_position.y
        transform[3, 2] = self.model_position.z
        return transform

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_transform = self._global_transform()
        model_view = self.view @ global_transform
        mvp = self.project @ model_view
        ShaderLib.use("OctreeParticles")
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform(
            "normalMatrix", Mat3.from_mat4(model_view).inverse().transposed()
        )
        gl.glBindVertexArray(self.particle_vao)
        gl.glDrawArraysInstanced(
            gl.GL_TRIANGLES, 0, self.sphere_vertex_count, self.system.count
        )
        gl.glBindVertexArray(0)

        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("Colour", 0.8, 0.8, 0.8, 1.0)
        with self.bounds_vao:
            self.bounds_vao.draw()
        Text.render_text(
            "Arial",
            10,
            25,
            f"particles {self.system.count}   Space reset   A pause",
            Vec3(1, 1, 1),
        )

    def resizeGL(self, width: int, height: int) -> None:
        self.window_width = int(width * self.devicePixelRatio())
        self.window_height = int(height * self.devicePixelRatio())
        self.project = perspective(45.0, width / max(height, 1), 0.05, 350.0)
        Text.set_screen_size(width, height)

    def _tick(self) -> None:
        if self.animate:
            self.system.step()
            self.makeCurrent()
            self._update_instance_buffer()
        self.update()

    def _reset_particles(self) -> None:
        self.system = ParticleSystem.grid(self.grid_size, self.seed)
        self.makeCurrent()
        self._update_instance_buffer()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self.animate = not self.animate
        elif event.key() == Qt.Key_Space:
            self._reset_particles()
        else:
            super().keyPressEvent(event)
        self.update()

    def closeEvent(self, event) -> None:
        self.simulation_timer.stop()
        super().closeEvent(event)


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
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoketest", nargs="?", const=300, default=None, type=int, metavar="MS"
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
    window = MainWindow(args.grid, args.seed)
    window.resize(1024, 720)
    window.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
