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
from PySide6.QtCore import Qt, Signal, Slot
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
        # --- Camera and Transformation Attributes ---
        self.mouse_global_tx: Mat4 = (
            Mat4()
        )  # Global transformation matrix controlled by the mouse
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)
        self.model_position: Vec3 = Vec3()  # Position of the model in world space

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width¦
        self.window_height: int = 720  # Window height

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.original_x_rotation: int = (
            0  # Initial X position of the mouse when a rotation starts
        )
        self.original_y_rotation: int = (
            0  # Initial Y position of the mouse when a rotation starts
        )
        self.original_x_pos: int = (
            0  # Initial X position of the mouse when a translation starts
        )
        self.original_y_pos: int = (
            0  # Initial Y position of the mouse when a translation starts
        )
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming

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
                        # keep shader_data format simple: use scalar or list as needed
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

    def new_shader(self, path):
        self.shader = ShaderLoader(path)
        for uniform in self.shader.uniforms:
            utype = uniform.get("Type", "")
            # vectors & colours
            if utype in ["Vec3", "Colour3"]:
                value = Vec3(
                    float(uniform["Value"][0]),
                    float(uniform["Value"][1]),
                    float(uniform["Value"][2]),
                )
                self.uniform_found.emit(uniform["Name"], utype, value)
            elif utype in ["Vec4", "Colour4"]:
                value = Vec4(
                    float(uniform["Value"][0]),
                    float(uniform["Value"][1]),
                    float(uniform["Value"][2]),
                    float(uniform["Value"][3]),
                )
                self.uniform_found.emit(uniform["Name"], utype, value)
            # float scalar support
            elif str(utype).lower() == "float":
                # Value may be [40.0] or 40.0, support both
                v = uniform["Value"]
                try:
                    if isinstance(v, list) and len(v) > 0:
                        fval = float(v[0])
                    else:
                        fval = float(v)
                except Exception:
                    fval = 0.0
                # emit the data_type as 'Float' so main.add_uniform_widget matches
                self.uniform_found.emit(uniform["Name"], "Float", fval)
            else:
                # other types not creating GUI controls for now
                continue

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

        # Apply rotation based on user input
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z

        MV = self.view @ self._model_transform @ self.mouse_global_tx
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

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement events for camera control.

        Args:
            event: The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        # Translate (pan) the scene if the right mouse button is pressed
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mousePressEvent(self, event) -> None:
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        position = event.position()
        # Left button initiates rotation
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        # Right button initiates translation
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handles mouse button release events to stop rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        # Stop rotating when the left button is released
        if event.button() == Qt.LeftButton:
            self.rotate = False
        # Stop translating when the right button is released
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming.

        Args:
            event: The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
        self.update()
