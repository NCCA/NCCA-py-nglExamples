import OpenGL.GL as gl
from ncca.ngl import Mat3, Prims, Transform, Vec3, Vec4, look_at, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib
from PySide6.QtCore import Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class PyNGLScene(QOpenGLWidget):
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
        self._model_name = "teapot"
        self._model_scale = Vec3(1.0, 1.0, 1.0)
        self._model_rotation = Vec3(0.0, 0.0, 0.0)
        self._model_position = Vec3(0.0, 0.0, 0.0)
        self._model_colour = Vec4(1.0, 1.0, 0.0, 1.0)

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

    @Slot(float, float, float)
    def set_model_position(self, x: float, y: float, z: float) -> None:
        """
        Set the position of the model
        """
        self._model_position = Vec3(x, y, z)
        self.update()  # Tell the scene to repaint

    @Slot(float, float, float)
    def set_model_scale(self, x: float, y: float, z: float) -> None:
        """
        Set the scale of the model
        """
        self._model_scale = Vec3(x, y, z)
        self.update()  # Tell the scene to repaint

    @Slot(str)
    def set_model_name(self, name: str) -> None:
        """
        Set the name of the model to draw
        """
        self._model_name = name
        self.update()  # Tell the scene to repaint

    @Slot(float, float, float)
    def set_colour(self, x: float, y: float, z: float) -> None:
        """
        Set the colour of the model
        """
        self._model_colour = Vec4(x, y, z, 1.0)
        self.update()  # Tell the scene to repaint

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
        # It looks from (0, 1, 4) towards (0, 0, 0) with the 'up' direction along the Y-axis.
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, self.width() / self.height(), 0.1, 100.0)
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        Primitives.load_default_primitives()
        Primitives.create(Prims.SPHERE, "sphere", 1.0, 32)

    def load_matrices_to_shader(self, transform: Transform) -> None:
        """
        Load transformation matrices to the shader uniforms.
        """
        ShaderLib.use(DefaultShader.DIFFUSE)
        MV = self.view @ transform.matrix()
        mvp = self.project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix = normal_matrix.inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Apply translation
        tx = Transform()
        tx.set_position(
            self._model_position.x, self._model_position.y, self._model_position.z
        )
        tx.set_scale(self._model_scale.x, self._model_scale.y, self._model_scale.z)
        tx.set_rotation(
            self._model_rotation.x, self._model_rotation.y, self._model_rotation.z
        )
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", self._model_colour)
        self.load_matrices_to_shader(tx)
        if self._wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        match self._model_name:
            case "teapot":
                Primitives.draw("teapot")
            case "Sphere":
                Primitives.draw("sphere")
            case "Cube":
                Primitives.draw("cube")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Parameters
        ----------
            w : int
                The new width of the window.
            h : int
                The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.project = perspective(45.0, float(w) / h, 0.01, 350.0)
