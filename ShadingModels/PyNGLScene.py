import OpenGL.GL as gl
from ncca.ngl import (
    Mat3,
    Mat4,
    Primitives,
    Prims,
    Vec3,
    Vec4,
    look_at,
    perspective,
)
from PySide6.QtCore import Signal, Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from ShaderLoader import ShaderLoader


class PyNGLScene(QOpenGLWidget):
    uniform_found = Signal(str, str, object)

    def __init__(self, parent=None):
        """
        Initialise the NGL scene
        """
        super().__init__(parent)
        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height

        # --- Mouse Control Attributes for Camera Manipulation ---
        self._wireframe: bool = False
        self._model_name = "Teapot"
        self._model_colour = Vec4(1.0, 1.0, 0.0, 1.0)
        self._model_transform = Mat4()
        self.fov = 45.0
        self.near = 0.1
        self.far = 100.0
        self.shader = None

    def update_perspective(self, fov, near, far) -> None:
        """Update the perspective projection matrix."""
        self.fov = fov
        self.near = near
        self.far = far
        print(f"Updated perspective: fov={fov}, near={near}, far={far}")
        self.project = perspective(
            self.fov, self.window_width / self.window_height, self.near, self.far
        )

        self.update()

    @Slot(Mat4)
    def set_camera(self, view):
        self.view = view
        self.update()

    @Slot(bool)
    def set_wireframe(self, value: bool) -> None:
        """
        Set the wireframe mode for the model
        """
        self._wireframe = value
        self.update()  # Tell the scene to repaint

    @Slot(float, float, float)
    def set_model_rotation(self, x: float, y: float, z: float) -> None:
        """
        Set the rotation of the model
        """
        self._model_rotation = Vec3(x, y, z)
        self.update()  # Tell the scene to repaint

    @Slot(str)
    def set_model_name(self, name: str) -> None:
        """
        Set the name of the model to draw
        """
        self._model_name = name
        self.update()  # Tell the scene to repaint

    @Slot(Mat4)
    def set_transform(self, transform: Mat4) -> None:
        """
        Set the position and scale of the model
        """
        self._model_transform = transform
        self.update()  # Tell the scene to repaint

    @Slot(str, object)
    def set_uniform_value(self, name: str, value) -> None:
        """
        Set the value of a uniform.
        """
        if self.shader is not None:
            for uniform in self.shader.uniforms:
                if uniform["Name"] == name:
                    if isinstance(value, (Vec3, Vec4)):
                        uniform["Value"] = list(value)
                    else:
                        uniform["Value"] = value
                    break
        self.update()

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        # Set the background color to a dark grey
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        # Enable depth testing, which ensures that objects closer to the camera obscure those further away
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Enable multisampling for anti-aliasing, which smooths jagged edges
        gl.glEnable(gl.GL_MULTISAMPLE)
        # Set up the camera's view matrix.
        # It looks from (0, 1, 4) towards (0, 0, 0) with the 'up' direction along the Y-axis.
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, self.width() / self.height(), 0.1, 100.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 32)

        self.new_shader("shaders/Constant.json")

    def new_shader(self, path):
        self.shader = ShaderLoader(path)
        for uniform in self.shader.uniforms:
            if uniform["Type"] in ["Vec3", "Vec4", "Colour3", "Colour4"]:
                if uniform["Type"] == "Vec3" or uniform["Type"] == "Colour3":
                    value = Vec3(
                        float(uniform["Value"][0]),
                        float(uniform["Value"][1]),
                        float(uniform["Value"][2]),
                    )
                elif uniform["Type"] == "Vec4" or uniform["Type"] == "Colour4":
                    value = Vec4(
                        float(uniform["Value"][0]),
                        float(uniform["Value"][1]),
                        float(uniform["Value"][2]),
                        float(uniform["Value"][3]),
                    )
                self.uniform_found.emit(uniform["Name"], uniform["Type"], value)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self.shader is None:
            return
        MV = self.view @ self._model_transform
        MVP = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()
        self.shader.set_uniforms(MVP, MV, normal_matrix)
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

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        # Update the projection matrix to match the new aspect ratio.
        # This creates a perspective projection with a 45-degree field of view.
        self.project = perspective(self.fov, float(w) / h, self.near, self.far)
