from typing import Optional

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Prims, Vec3
from ncca.ngl.opengl import Primitives
from PySide6.QtCore import QEvent, Qt, Signal, Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from ShaderLoader import ShaderLoader

from Camera import Camera


class PyNGLScene(QOpenGLWidget):
    """A QOpenGLWidget for rendering a 3D scene using PyNGL."""

    uniform_found = Signal(str, str, object, object)
    double_clicked = Signal()  # emitted when the scene is double-clicked

    def __init__(self, parent=None) -> None:
        """
        Initialise the NGL scene
        """
        super().__init__(parent)

        self._wireframe: bool = False
        self._model_name: str = "Teapot"
        self._model_transform: Mat4 = Mat4()
        self._model_rotation: Vec3 = Vec3(0.0, 0.0, 0.0)
        self.shader: Optional[ShaderLoader] = None
        self.camera: Optional[Camera] = None
        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height

    def mouseDoubleClickEvent(self, event: QEvent) -> None:
        """
        Handle mouse double-click events.

        Emits the `double_clicked` signal to allow the main window to toggle
        fullscreen for this widget.

        Args:
            event: The mouse event.
        """
        # Emit a signal so the main window can toggle fullscreen for this widget
        self.double_clicked.emit()
        event.accept()

    def update_perspective(self, fov: float, near: float, far: float) -> None:
        """
        Update the perspective projection matrix.

        Args:
            fov: The field of view in degrees.
            near: The near clipping plane distance.
            far: The far clipping plane distance.
        """
        if self.camera:
            self.camera.fov = fov
            self.camera.near = near
            self.camera.far = far
            self.camera.update_projection(self.window_width, self.window_height)
        self.update()

    @Slot(bool)
    def set_wireframe(self, value: bool) -> None:
        """
        Set the wireframe mode for the model.

        Args:
            value: True to enable wireframe, False for solid.
        """
        self._wireframe = value
        self.update()  # Tell the scene to repaint

    @Slot(float, float, float)
    def set_model_rotation(self, x: float, y: float, z: float) -> None:
        """
        Set the rotation of the model.

        Args:
            x: Rotation around the x-axis.
            y: Rotation around the y-axis.
            z: Rotation around the z-axis.
        """
        self._model_rotation = Vec3(x, y, z)
        self.update()  # Tell the scene to repaint

    @Slot(str)
    def set_model_name(self, name: str) -> None:
        """
        Set the name of the model to draw.

        Args:
            name: The name of the primitive to draw (e.g., "Teapot", "Sphere").
        """
        self._model_name = name
        self.update()  # Tell the scene to repaint

    @Slot(Mat4)
    def set_transform(self, transform: Mat4) -> None:
        """
        Set the position and scale of the model.

        Args:
            transform: The model's transformation matrix.
        """
        self._model_transform = transform
        self.update()  # Tell the scene to repaint

    @Slot(Mat4)
    def set_view_matrix(self, view: Mat4) -> None:
        """
        Set the camera's view matrix.

        Args:
            view: The new view matrix.
        """
        if self.camera:
            self.camera.view = view
            self.update()

    @Slot(str, object)
    def set_uniform_value(self, name: str, value: object) -> None:
        """
        Set the value of a uniform.

        Args:
            name: The name of the uniform to set.
            value: The new value for the uniform.
        """
        if self.shader is not None:
            self.shader.set_uniform_value(name, value)
        self.update()

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.

        This is the place to set up global OpenGL state, load shaders,
        and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self.camera = Camera(self.width(), self.height())

        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 32)
        Primitives.create(Prims.TRIANGLE_PLANE, "plane", 2, 2, 20, 20, Vec3(0, 1, 0))

        self.new_shader("shaders/Constant.json")

    def new_shader(self, path: str) -> None:
        """
        Load a new shader from a JSON file and emit signals for its uniforms.

        Args:
            path: The file path to the shader's JSON definition.
        """
        self.shader = ShaderLoader(path)
        for name, definition in self.shader.get_uniform_definitions().items():
            self.uniform_found.emit(
                name, definition["type"], definition["range"], definition["value"]
            )

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.

        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self.shader is None or self.camera is None:
            return

        model_matrix = self.camera.get_model_matrix()
        MV = self.camera.get_view_matrix() @ self._model_transform @ model_matrix
        MVP = self.camera.get_projection_matrix() @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix = normal_matrix.inverse().transposed()
        self.shader.apply_uniforms(MVP, MV, normal_matrix)

        if self._wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        match self._model_name:
            case "Teapot":
                Primitives.draw("teapot")
            case "Sphere":
                Primitives.draw("sphere")
            case "Cube":
                Primitives.draw("cube")
            case "Plane":
                Primitives.draw("plane")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        """
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        if self.camera:
            self.camera.update_projection(self.window_width, self.window_height)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_W:
            self._wireframe = not self._wireframe
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            if self.camera:
                self.camera.reset()
        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.camera:
            self.camera.mouse_move_event(event)
            self.update()

    def mousePressEvent(self, event) -> None:
        if self.camera:
            self.camera.mouse_press_event(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.camera:
            self.camera.mouse_release_event(event)

    def wheelEvent(self, event) -> None:
        if self.camera:
            self.camera.wheel_event(event)
            self.update()
