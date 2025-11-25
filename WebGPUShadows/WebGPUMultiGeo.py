#!/usr/bin/env -S uv run --script
import sys

import numpy as np
import wgpu
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Prims,
    Transform,
    Vec3,
    Vec4,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

from Pipeline import Pipeline
from WebGPUWidget import WebGPUWidget


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of NumpyBufferWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Multi Mesh")
        self.device = None
        self.mouse_global_tx: Mat4 = Mat4()
        self.model_position: Vec3 = Vec3()  # Position of the model in world space
        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = False  # Flag to check if the scene is being translated (panned)
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.original_x_rotation: int = 0  # Initial X position of the mouse when a rotation starts
        self.original_y_rotation: int = 0  # Initial Y position of the mouse when a rotation starts
        self.original_x_pos: int = 0  # Initial X position of the mouse when a translation starts
        self.original_y_pos: int = 0  # Initial Y position of the mouse when a translation starts
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming
        self.light_one_state = True
        self.light_rotation = 0.0

        self.pipeline = None
        self.rotation = 0.0
        self.eye = Vec3(0.0, 4.0, 8.0)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(45.0, self.width() / self.height(), 0.1, 100.0, PerspMode.WebGPU)
        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            # exit(1)
            raise e

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        self.pipeline = Pipeline(self.device, self.eye, self.view, self.project)

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.
        """
        self.update_transformations()
        self.pipeline.update_lights(self.light_one_state)

        # Animate the single light
        self.light_rotation += 0.5
        light_rot_mat = Mat4().rotate_y(self.light_rotation)

        light_pos_1 = Vec4(0.0, 2.0, 2.0, 1.0)
        rotated_light_1 = light_rot_mat @ light_pos_1

        self.pipeline.update_light_positions([rotated_light_1])

        # 1. Define the objects in our scene as a list of tuples:
        # (mesh_name, transform_matrix, colour)
        scene_objects = []

        tx = Transform()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-1.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("buddah", self.mouse_global_tx @ tx.get_matrix(), (1, 0, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-2.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("bunny", self.mouse_global_tx @ tx.get_matrix(), (1, 0, 1, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, 0.0)  # Move it to the right
        scene_objects.append(("teapot", self.mouse_global_tx @ tx.get_matrix(), (0, 1, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(1.5, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dragon", self.mouse_global_tx @ tx.get_matrix(), (1, 1, 0, 1)))

        tx.reset()
        tx.set_position(0.0, 0.1, 1.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("troll", self.mouse_global_tx @ tx.get_matrix(), (0, 0.2, 1, 1)))
        tx.reset()
        tx.set_position(-1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, 0, 0)
        scene_objects.append(("icosahedron", self.mouse_global_tx @ tx.get_matrix(), (0.2, 0.2, 0.8, 1)))
        tx.reset()
        tx.set_position(-2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dodecahedron", self.mouse_global_tx @ tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("football", self.mouse_global_tx @ tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("tetrahedron", self.mouse_global_tx @ tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(1.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("octahedron", self.mouse_global_tx @ tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("cube", self.mouse_global_tx @ tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0, -0.5, 0)
        scene_objects.append(("floor", self.mouse_global_tx @ tx.get_matrix(), (1, 1, 1, 1)))

        ## Render single light
        tx.reset()
        tx.set_position(rotated_light_1.x, rotated_light_1.y, rotated_light_1.z)
        scene_objects.append(("light1", self.mouse_global_tx @ tx.get_matrix(), (1, 1, 1, 1)))

        # 2. Pass the entire scene to the pipeline to be rendered
        self.pipeline.render(
            self.colour_buffer_texture_view,
            self.multisample_texture_view,
            self.depth_buffer_view,
            self.texture_size,
            scene_objects,
        )

        self._update_colour_buffer()

    def update_transformations(self) -> None:
        """
        Update the global transformation matrix from mouse input.
        """
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.
        """
        key = event.key()

        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        elif key == Qt.Key_1:
            self.light_one_state ^= True

        self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement events for camera control.
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
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
        self.update()

    def timerEvent(self, event):
        self.update()


def main():
    """
    Main function to run the application.
    """
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
