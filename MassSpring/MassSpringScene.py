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
from picking import (
    decode_id,
    encode_id,
    intersect_plane,
    ray_from_screen,
    transform_point,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget

_MASS_SCALE = 0.1
# Ghosts are smaller than the masses so they read as reference markers rather
# than competing with the chain itself.
_GHOST_SCALE = 0.06
_FIXED_COLOUR = (1.0, 0.0, 0.0, 1.0)
_FREE_COLOUR = (0.0, 1.0, 0.0, 1.0)
_GHOST_COLOUR = (0.4, 0.4, 0.8, 1.0)
_LINE_COLOUR = (1.0, 1.0, 1.0, 1.0)
_HELD_COLOUR = (1.0, 1.0, 0.0, 1.0)

# The masses are small on screen, so the pick reads a block around the cursor
# and falls back to it when the exact pixel is background. That makes grabbing
# one forgiving without having to click it dead centre.
_PICK_BLOCK = 9


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
        # the world-space plane a held mass is dragged in, set on press
        self._drag_plane_point = None
        # a single-sample framebuffer to render the ID pass into (see _pick)
        self._pick_fbo = None
        self._pick_size = None
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

    # -------------------------------------------------------------- picking
    def _world_mvp(self) -> Mat4:
        """Projection @ view, with no arcball -- the space the drag happens in."""
        return self.project @ self.view

    def _ensure_pick_fbo(self) -> None:
        """Make a single-sample framebuffer to render the ID pass into.

        A QOpenGLWidget draws into a multisampled FBO, and glReadPixels on a
        multisampled buffer is an invalid operation -- so the ID pass cannot
        just be drawn over the widget's own target the way it can in a
        QOpenGLWindow demo. Rendering IDs somewhere without antialiasing is
        what we want regardless: no pixel is ever a blend of two IDs.
        """
        size = (self.window_width, self.window_height)
        if self._pick_fbo is not None and self._pick_size == size:
            return
        if self._pick_fbo is not None:
            gl.glDeleteFramebuffers(1, [self._pick_fbo])
            gl.glDeleteRenderbuffers(1, [self._pick_colour])
            gl.glDeleteRenderbuffers(1, [self._pick_depth])

        width, height = size
        self._pick_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._pick_fbo)

        self._pick_colour = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._pick_colour)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, width, height)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER,
            self._pick_colour,
        )

        self._pick_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._pick_depth)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self._pick_depth,
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        self._pick_size = size

    def _pick(self, x: float, y: float) -> int | None:
        """Render every mass flat in its ID colour and read back which one is
        under the cursor. x, y are device pixels, Qt's top-left origin."""
        self.makeCurrent()
        self._ensure_pick_fbo()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._pick_fbo)
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # black means nothing
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        global_tx = self._mouse_global_tx()
        ShaderLib.use(DefaultShader.COLOUR)
        for i in range(self.chain.num_masses):
            r, g, b = encode_id(i)
            self.transform.reset()
            self.transform.set_scale(_MASS_SCALE, _MASS_SCALE, _MASS_SCALE)
            p = self.chain.positions[i]
            self.transform.set_position(float(p[0]), float(p[1]), float(p[2]))
            mvp = self.project @ self.view @ global_tx @ self.transform.matrix()
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("Colour", r / 255.0, g / 255.0, b / 255.0, 1.0)
            Primitives.draw("cube")
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)

        half = _PICK_BLOCK // 2
        read_x = max(0, int(x) - half)
        read_y = max(0, self.window_height - int(y) - half)
        data = gl.glReadPixels(
            read_x, read_y, _PICK_BLOCK, _PICK_BLOCK, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())

        # glReadPixels hands back raw bytes here, not an array
        buffer = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
        pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, 3)
        # what is directly under the cursor wins; only if that is background do
        # we accept a neighbour, so two masses close together stay separable
        ordered = [pixels[len(pixels) // 2], *pixels]
        for pixel in ordered:
            index = decode_id(pixel)
            if index is not None and index < self.chain.num_masses:
                return index
        return None

    def _drag_to(self, x: float, y: float) -> None:
        """Slide the held mass along its drag plane to follow the cursor."""
        if self.chain.dragged is None or self._drag_plane_point is None:
            return
        origin, direction = ray_from_screen(
            x, y, self.window_width, self.window_height, self._world_mvp().to_numpy()
        )
        # the camera is fixed looking down -z, so the screen-parallel plane
        # has this normal in world space whatever the arcball is doing
        hit = intersect_plane(
            origin, direction, self._drag_plane_point, np.array([0.0, 0.0, -1.0])
        )
        if hit is None:
            return
        # back out of the arcball into the chain's own space
        global_np = self._mouse_global_tx().to_numpy()
        self.chain.move_dragged(transform_point(hit, np.linalg.inv(global_np)))
        self.update()

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
            # a held mass is in the fixed set too (it is kinematic), so this
            # check has to come first or it would show red rather than held
            if i == self.chain.dragged:
                colour = _HELD_COLOUR
            elif self.chain.is_fixed(i):
                colour = _FIXED_COLOUR
            else:
                colour = _FREE_COLOUR
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

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, event) -> None:
        """Left-press grabs a mass if one is under the cursor, otherwise it
        falls through to the mixin and rotates the camera."""
        if event.button() == Qt.LeftButton:
            dpr = self.devicePixelRatio()
            position = event.position()
            index = self._pick(position.x() * dpr, position.y() * dpr)
            if index is not None:
                self.chain.set_dragged(index)
                # the drag plane passes through the mass, in world space
                self._drag_plane_point = transform_point(
                    self.chain.positions[index], self._mouse_global_tx().to_numpy()
                )
                self.update()
                return  # do NOT let the mixin start a camera rotate too
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.chain.dragged is not None:
            dpr = self.devicePixelRatio()
            position = event.position()
            self._drag_to(position.x() * dpr, position.y() * dpr)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.chain.dragged is not None:
            self.chain.set_dragged(None)
            self._drag_plane_point = None
            self.update()
            return
        super().mouseReleaseEvent(event)
