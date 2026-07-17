"""The OpenGL view of the mass spring chain.

Draws what the C++ NGLScene drew, generalised to N masses: a line through
the chain, a cube at each mass (red when pinned, green when free) and a
ghost sphere at each mass's start position so you can see how far the chain
has moved.
"""

import numpy as np
import OpenGL.GL as gl
from mass_spring import MassSpringChain
from ncca.ngl import Mat3, Mat4, Prims, Transform, Vec3, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
    VAOFactory,
    VAOType,
    VertexData,
)
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget

_MASS_SCALE = 0.1
# Ghosts are smaller than the masses so they read as reference markers rather
# than competing with the chain itself.
_GHOST_SCALE = 0.06
_FIXED_COLOUR = (1.0, 0.0, 0.0, 1.0)
_FREE_COLOUR = (0.0, 1.0, 0.0, 1.0)
_GHOST_COLOUR = (0.4, 0.4, 0.8, 1.0)
_LINE_COLOUR = (1.0, 1.0, 1.0, 1.0)


class MassSpringScene(PySideEventHandlingMixin, QOpenGLWidget):
    """Draws the chain and drives the sim timer."""

    def __init__(
        self, chain: MassSpringChain, timer_interval: int = 20, parent=None
    ) -> None:
        super().__init__(parent)
        self.chain = chain
        self.window_width = 1024
        self.window_height = 720
        self.transform = Transform()
        self._timer_interval = timer_interval
        self._timer_id = None
        self.setup_event_handling()
        self.start_sim_timer()

    # ------------------------------------------------------------ sim timer
    def start_sim_timer(self) -> None:
        if self._timer_id is None:
            self._timer_id = self.startTimer(self._timer_interval)

    def stop_sim_timer(self) -> None:
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

    @Slot(int)
    def set_timer_duration(self, interval: int) -> None:
        """Restart the timer at a new interval, if it is running."""
        self._timer_interval = interval
        if self._timer_id is not None:
            self.stop_sim_timer()
            self.start_sim_timer()

    @Slot(bool)
    def toggle_sim(self, running: bool) -> None:
        if running:
            self.start_sim_timer()
        else:
            self.stop_sim_timer()

    def timerEvent(self, event) -> None:
        self.chain.update()
        self.update()

    # ---------------------------------------------------------------- setup
    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / max(self.height(), 1), 0.5, 150.0
        )
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        # cube and teapot are mesh defaults -- Prims.CUBE is not a create()
        # type and passing it to create raises ValueError.
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 20)

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, w / max(h, 1), 0.5, 150.0)

    # -------------------------------------------------------------- drawing
    def _mouse_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        return rot_y @ rot_x

    def load_matrices_to_shader(self, global_tx: Mat4) -> None:
        """The ngl diffuse shader lights in VIEW space (it takes MV and a
        normalMatrix built from MV), so unlike the world-space PBR demos the
        normal matrix here correctly comes from MV."""
        ShaderLib.use(DefaultShader.DIFFUSE)
        M = global_tx @ self.transform.matrix()
        MV = self.view @ M
        MVP = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV).inverse().transposed()
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def _draw_chain_line(self, global_tx: Mat4) -> None:
        """One GL_LINE_STRIP through every mass, rebuilt each frame because the
        positions change every step."""
        points = self.chain.positions.astype(np.float32).reshape(-1)
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", *_LINE_COLOUR)
        ShaderLib.set_uniform("MVP", self.project @ self.view @ global_tx)
        vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINE_STRIP)
        with vao as v:
            v.set_data(VertexData(points, self.chain.num_masses))
            v.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            v.set_num_indices(self.chain.num_masses)
            v.draw()

    def _draw_mass(
        self,
        position: np.ndarray,
        colour,
        prim: str,
        global_tx: Mat4,
        scale: float = _MASS_SCALE,
    ) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", *colour)
        self.transform.reset()
        self.transform.set_scale(scale, scale, scale)
        self.transform.set_position(
            float(position[0]), float(position[1]), float(position[2])
        )
        self.load_matrices_to_shader(global_tx)
        Primitives.draw(prim)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_tx = self._mouse_global_tx()

        self._draw_chain_line(global_tx)
        for i in range(self.chain.num_masses):
            colour = _FIXED_COLOUR if self.chain.is_fixed(i) else _FREE_COLOUR
            self._draw_mass(self.chain.positions[i], colour, "cube", global_tx)
        # Ghosts of where each mass started, as the original drew its targets.
        # A pinned mass never leaves its start, so its ghost would sit exactly
        # on top of it and hide the red cube that says it is pinned -- skip it.
        for i in range(self.chain.num_masses):
            if self.chain.is_fixed(i):
                continue
            self._draw_mass(
                self.chain.initial_positions[i],
                _GHOST_COLOUR,
                "sphere",
                global_tx,
                scale=_GHOST_SCALE,
            )
