"""Drawing the MuJoCo world with NGL.

There is no physics in here. The scene asks the world for a list of bodies, each
of which is a shape name and a model matrix, and draws the matching primitive or
mesh. That split is what keeps the physics testable without a window, and it
means the scene neither knows nor cares which spawning strategy is running.

The camera is the usual demo arrangement: left mouse rotates, right mouse pans,
wheel zooms, all of it handled by `PySideEventHandlingMixin`.
"""

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Obj, Prims, Vec3, look_at, perspective
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    PySideEventHandlingMixin,
    ShaderLib,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

GROUND_COLOUR = (0.8, 0.8, 0.8, 1.0)


class MuJoCoScene(PySideEventHandlingMixin, QOpenGLWidget):
    """Draws whatever the physics world currently holds.

    Attributes
    ----------
        world : PhysicsWorld
            the world being drawn, swapped when the strategy changes
        wireframe : bool
            draw everything as lines, the W and S keys in the original
    """

    def __init__(self, world, model_dir: str = "models", parent=None):
        super().__init__(parent)
        self.world = world
        self._model_dir = model_dir
        self.wireframe = False
        self.window_width = self.width()
        self.window_height = self.height()
        self._meshes: dict[str, Obj] = {}
        self.setup_event_handling(
            rotation_sensitivity=0.5,
            translation_sensitivity=0.05,
            zoom_sensitivity=0.5,
            initial_position=Vec3(0.0, -3.0, -8.0),
            # Every key binding in this demo lives on the main window, so the
            # mixin is here for the mouse only. Left to itself it would take
            # Escape, W, S and Space for its own use, and since this widget has
            # the keyboard focus it would see them first.
            handle_key_shortcuts=False,
        )
        self.setFocusPolicy(self.focusPolicy().StrongFocus)

    def initializeGL(self) -> None:
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self.view = look_at(
            Vec3(0.0, 6.0, 18.0), Vec3(0.0, 2.0, 0.0), Vec3(0.0, 1.0, 0.0)
        )
        self.project = perspective(
            45.0, self.width() / max(self.height(), 1), 0.05, 500.0
        )

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        # The sizes here have to match the collision geoms in
        # collision_shapes.py or the drawing will not sit on the hull that is
        # actually colliding. The cube default primitive is already 1x1x1,
        # matching the box's 0.5 half-extents.
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 0.5, 40)
        Primitives.create(Prims.CAPSULE, "capsule", 0.5, 1.0, 20)
        Primitives.create(Prims.CYLINDER, "cylinder", 0.5, 2.0, 20, 20)
        Primitives.create(Prims.CONE, "cone", 0.5, 2.0, 32, 2)
        Primitives.create(Prims.LINE_GRID, "grid", 140.0, 140.0, 40)

        # High-res meshes for drawing; the low-res ones went to MuJoCo.
        for name in ("teapot", "apple"):
            self._meshes[name] = Obj.obj_with_vao(f"{self._model_dir}/{name}.obj")

    def resizeGL(self, w: int, h: int) -> None:
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, w / max(h, 1), 0.05, 500.0)

    def _mouse_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    def _load_matrices(self, model: Mat4, global_tx: Mat4) -> None:
        """The ngl diffuse shader lights in view space, so the normal matrix
        comes from MV rather than the model matrix."""
        mv = self.view @ global_tx @ model
        ShaderLib.set_uniform("MVP", self.project @ mv)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", Mat3.from_mat4(mv).inverse().transposed())

    def _draw_shape(self, name: str) -> None:
        mesh = self._meshes.get(name)
        if mesh is not None:
            mesh.draw()
        else:
            Primitives.draw(name)

    def paintGL(self) -> None:
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPolygonMode(
            gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL
        )

        ShaderLib.use(DefaultShader.DIFFUSE)
        global_tx = self._mouse_global_tx()

        for body in self.world.bodies():
            shape = self.world.catalogue.shapes[body.shape]
            self._load_matrices(body.transform, global_tx)
            ShaderLib.set_uniform("Colour", shape.colour)
            self._draw_shape(shape.drawn_as)

        # The ground plane is infinite in MuJoCo, so the grid is only there to
        # give the eye something to judge the fall against.
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self._load_matrices(Mat4(), global_tx)
        ShaderLib.set_uniform("Colour", *GROUND_COLOUR)
        Primitives.draw("grid")
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
