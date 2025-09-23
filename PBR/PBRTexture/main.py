#!/usr/bin/env -S uv run --script
"""
A demonstration of Physically Based Rendering (PBR) with multiple textures using py-ngl.

This script showcases how to load and apply different PBR texture sets (albedo, normal,
metallic, roughness, AO) to a grid of teapots. It also includes dynamic lights that
can be toggled on and off. The scene features a floor and uses a first-person camera
for navigation.
"""

import logging
import math
import random
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ngl import (
    DefaultShader,
    FirstPersonCamera,
    Mat2,
    Mat3,
    Mat4,
    Primitives,
    Random,
    ShaderLib,
    Transform,
    Vec3,
    Vec3Array,
    logger,
)
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from texture_pack import TexturePack

PBR_SHADER = "pbr"


class MainWindow(QOpenGLWindow):
    """
    The main window for the PBR OpenGL application.

    This class handles the setup of the OpenGL context, loading of shaders and textures,
    rendering the scene, and processing user input for camera control and interaction.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initializes the main window, sets up scene parameters, and configures the camera.
        """
        super().__init__()

        # --- Window and UI Attributes ---
        self.width: int = 1024
        self.height: int = 720
        self.setTitle("PBR Texture Demo")

        # --- Scene and Transformation Attributes ---
        self.transform = Transform()
        self.mouse_global_tx: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()
        self.seed: int = 12345
        self.light_on = [True, True, True, True]
        self.show_lights = True

        # --- Mouse Control for Camera ---
        self.rotate: bool = False
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0

        # --- Keyboard Control ---
        self.keys_pressed: set = set()

        # --- Frame Timing used to update the camera
        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 0.0
        self._setup_camera()

    def _setup_camera(self):
        eye = Vec3(0, 5, 20)
        look = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        self.camera = FirstPersonCamera(eye, look, up, 45.0)
        self.camera.set_projection(45.0, self.width / self.height, 0.05, 350.0)

    def initializeGL(self) -> None:
        """
        Called once to set up the OpenGL environment.

        This method configures global OpenGL states, loads shaders, sets up lights,
        and creates the geometry used in the scene.
        """
        self.makeCurrent()
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self._setup_pbr_shader()
        self._setup_lights()

        # Create and load geometry
        Primitives.create_sphere("sphere", 0.5, 40)
        Primitives.create_triangle_plane("floor", 30, 30, 10, 10, Vec3(0, 1, 0))
        TexturePack.load_json("textures/textures.json")
        Primitives.load_default_primitives()

    def _setup_pbr_shader(self) -> None:
        """Load and configure the PBR shader program."""
        if not ShaderLib.load_shader(
            PBR_SHADER,
            vert="shaders/PBRVertex.glsl",
            frag="shaders/PBRFragment.glsl",
        ):
            logging.error("Error loading PBR shaders")
            self.close()
        ShaderLib.use(PBR_SHADER)
        # Map texture units to shader samplers
        ShaderLib.set_uniform("albedoMap", 0)
        ShaderLib.set_uniform("normalMap", 1)
        ShaderLib.set_uniform("metallicMap", 2)
        ShaderLib.set_uniform("roughnessMap", 3)
        ShaderLib.set_uniform("aoMap", 4)
        ShaderLib.print_registered_uniforms()

    def _setup_lights(self) -> None:
        """Configure the light sources for the scene."""
        light_colors = Vec3Array(
            [
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
                Vec3(250.0, 250.0, 250.0),
            ]
        )
        self.light_positions = Vec3Array(
            [
                Vec3(-5.0, 4.0, -5.0),
                Vec3(5.0, 4.0, -5.0),
                Vec3(-5.0, 4.0, 5.0),
                Vec3(5.0, 4.0, 5.0),
            ]
        )
        ShaderLib.use(PBR_SHADER)
        for i in range(4):
            ShaderLib.set_uniform(f"lightPositions[{i}]", self.light_positions[i])
            ShaderLib.set_uniform(f"lightColors[{i}]", light_colors[i])

        # Setup a simple color for the light indicators
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 1.0, 1.0)

    def paintGL(self) -> None:
        """
        The main rendering loop. Called for every frame.
        """
        self.makeCurrent()
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self._update_camera_movement()
        self._render_lights()
        self._render_scene()

    def _update_camera_movement(self) -> None:
        """Calculates and applies camera movement based on currently pressed keys."""
        x_direction = 0.0
        y_direction = 0.0
        for key in self.keys_pressed:
            if key == Qt.Key_Left:
                y_direction = -1.0
            elif key == Qt.Key_Right:
                y_direction = 1.0
            elif key == Qt.Key_Up:
                x_direction = 1.0
            elif key == Qt.Key_Down:
                x_direction = -1.0

        if x_direction != 0.0 or y_direction != 0.0:
            current_frame = self.timer.elapsed() * 0.001
            delta_time = current_frame - self.last_frame
            self.last_frame = current_frame
            self.camera.move(x_direction, y_direction, delta_time)

    def _render_lights(self) -> None:
        """Renders the light sources as spheres in the scene."""
        if not self.show_lights:
            return
        ShaderLib.use(DefaultShader.COLOUR)
        for i in range(4):
            self.transform.reset()
            self.transform.set_position(
                self.light_positions[i][0],
                self.light_positions[i][1],
                self.light_positions[i][2],
            )
            self.load_matrices_to_colour_shader()
            Primitives.draw("sphere")

    def _render_scene(self) -> None:
        """Renders the main PBR scene, including the grid of teapots and the floor."""
        ShaderLib.use(PBR_SHADER)
        Random.set_seed_value(self.seed)
        textures = ["copper", "greasy", "panel", "rusty", "wood"]

        # Render a grid of teapots with random PBR materials and rotations
        for row in np.arange(-10, 10, 1.6):
            for col in np.arange(-10, 10, 1.6):
                TexturePack.activate_texture_pack(random.choice(textures))
                self.transform.set_position(col, 0.0, row)
                self.transform.set_rotation(
                    0.0, Random.random_positive_number() * 360.0, 0.0
                )
                self.load_matrices_to_shader()
                Primitives.draw("teapot")

        # Render the floor
        TexturePack.activate_texture_pack("greasy")
        self.transform.reset()
        self.transform.set_position(0.0, -0.5, 0.0)
        self.load_matrices_to_shader()
        Primitives.draw("floor")

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handles window resize events. Updates viewport and projection matrix.
        """
        self.width = int(w * self.devicePixelRatio())
        self.height = int(h * self.devicePixelRatio())
        self.camera.set_projection(45.0, w / h, 0.05, 350.0)

    def load_matrices_to_shader(self) -> None:
        """
        Calculates and sends the required matrices and uniforms to the PBR shader.
        """
        M = self.transform.get_matrix()
        MV = self.camera.view @ M
        MVP = self.camera.get_vp() @ M

        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()

        ShaderLib.use(PBR_SHADER)
        ShaderLib.set_uniform("MVP", MVP)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("M", M)

        # Apply a random texture rotation for variation
        texture_rotation = math.radians(Random.random_number(180.0))
        cos_theta = math.cos(texture_rotation)
        sin_theta = math.sin(texture_rotation)
        tex_rot = Mat2.from_list([cos_theta, sin_theta, -sin_theta, cos_theta])
        ShaderLib.set_uniform("textureRotation", tex_rot)
        ShaderLib.set_uniform("camPos", self.camera.eye)

    def load_matrices_to_colour_shader(self) -> None:
        """
        Calculates and sends the MVP matrix to the simple colour shader.
        """
        M = self.mouse_global_tx @ self.transform.get_matrix()
        MVP = self.camera.get_vp() @ M
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", MVP)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events for scene and camera control.
        """
        key = event.key()
        self.keys_pressed.add(key)

        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_R:
            self.seed = random.randint(0, 1000000)
            logger.info(f"New random seed set: {self.seed}")
        elif key == Qt.Key_Space:
            self._setup_camera()

            # A proper camera reset would be needed here if implemented in the camera class
        elif key == Qt.Key_1:
            self._toggle_light(0)
        elif key == Qt.Key_2:
            self._toggle_light(1)
        elif key == Qt.Key_3:
            self._toggle_light(2)
        elif key == Qt.Key_4:
            self._toggle_light(3)
        elif key == Qt.Key_L:
            self.show_lights ^= True

        self.update()
        super().keyPressEvent(event)

    def _toggle_light(self, light_index: int) -> None:
        """
        Toggles a light on or off and updates the corresponding shader uniform.

        Args:
            light_index: The index of the light to toggle (0-3).
        """
        self.light_on[light_index] ^= True
        ShaderLib.use(PBR_SHADER)
        if self.light_on[light_index]:
            colour = Vec3(250.0, 250.0, 250.0)
        else:
            colour = Vec3(0.0, 0.0, 0.0)
        ShaderLib.set_uniform(f"lightColors[{light_index}]", colour)

    def keyReleaseEvent(self, event) -> None:
        """
        Handles keyboard release events.
        """
        key = event.key()
        self.keys_pressed.discard(key)
        self.update()
        super().keyReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement for camera rotation.
        """
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.camera.process_mouse_movement(
                diff_x, -diff_y
            )  # Invert Y for intuitive rotation
            self.update()

    def mousePressEvent(self, event) -> None:
        """
        Handles mouse button presses to initiate camera rotation or translation.
        """
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handles mouse button releases to stop camera control actions.
        """
        if event.button() == Qt.LeftButton:
            self.rotate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming the camera.
        """
        num_pixels = event.angleDelta().y()  # Use y() for vertical scroll
        self.camera.process_mouse_scroll(num_pixels * 0.01)  # Adjust sensitivity
        self.update()


class DebugApplication(QApplication):
    """
    A custom QApplication subclass for improved debugging.

    This class overrides the `notify` method to catch and report exceptions that
    occur within event handlers, which Qt might otherwise suppress.
    """

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # --- Application Entry Point ---
    logging.basicConfig(level=logging.INFO)

    format = QSurfaceFormat()
    format.setSamples(4)
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    if "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1024, 720)
    window.show()
    sys.exit(app.exec())
