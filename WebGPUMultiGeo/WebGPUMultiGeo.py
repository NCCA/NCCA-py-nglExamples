#!/usr/bin/env -S uv run --script
import sys

import numpy as np
import wgpu
from ncca.ngl import (
    FirstPersonCamera,
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
from Pipeline import Pipeline
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


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
        self.light_one_state = True
        self.light_two_state = True
        self.light_three_state = True
        self.keys_pressed = set()
        # --- Frame Timing used to update the camera
        self.timer = QElapsedTimer()
        self.timer.start()
        self.last_frame = 1.0

        self.pipeline = None
        self.rotation = 0.0
        self.camera = FirstPersonCamera(
            Vec3(0, 2, 8), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
        )

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
        self.pipeline = Pipeline(self.device, self.camera)

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.
        """
        # Update frame timing here
        current_frame = self.timer.elapsed() * 0.001
        delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        self._update_camera_movement(delta_time)
        self.pipeline.update_lights(
            self.light_one_state, self.light_two_state, self.light_three_state
        )
        # 1. Define the objects in our scene as a list of tuples:
        # (mesh_name, transform_matrix, colour)
        scene_objects = []

        tx = Transform()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-1.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("buddah", tx.get_matrix(), (1, 0, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-2.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("bunny", tx.get_matrix(), (0.1, 0.2, 1, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, 0.0)  # Move it to the right
        scene_objects.append(("teapot", tx.get_matrix(), (0, 1, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(1.5, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dragon", tx.get_matrix(), (1, 1, 0, 1)))

        tx.reset()
        tx.set_position(0.0, 0.1, 1.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("troll", tx.get_matrix(), (0, 0.2, 1, 1)))
        tx.reset()
        tx.set_position(-1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, 0, 0)
        scene_objects.append(("icosahedron", tx.get_matrix(), (0.2, 0.2, 0.8, 1)))
        tx.reset()
        tx.set_position(-2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dodecahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("football", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("tetrahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(1.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("octahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("cube", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0, -0.5, 0)
        scene_objects.append(("floor", tx.get_matrix(), (1, 1, 1, 1)))

        ## Render Lights
        tx.reset()
        tx.set_position(0.0, 1.0, 1.0)
        scene_objects.append(("light1", tx.get_matrix(), (1, 1, 1, 1)))
        tx.reset()
        tx.set_position(-1.0, 1.0, -1.0)
        scene_objects.append(("light2", tx.get_matrix(), (1, 1, 1, 1)))
        # tx.reset()
        tx.set_position(1.0, 1.0, -1.0)
        scene_objects.append(("light3", tx.get_matrix(), (1, 1, 1, 1)))

        # 2. Pass the entire scene to the pipeline to be rendered
        self.pipeline.render(
            self.colour_buffer_texture_view,
            self.multisample_texture_view,
            self.depth_buffer_view,
            self.texture_size,
            scene_objects,
        )

        self._update_colour_buffer()

    def resizeWebGPU(self, width, height):
        ratio = self.devicePixelRatio()
        w = width * ratio
        h = height * ratio
        self.camera.set_projection(45.0, (w / h), 0.1, 300.0, PerspMode.WebGPU)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.
        """
        key = event.key()
        self.keys_pressed.add(key)

        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        elif key == Qt.Key_1:
            self.light_one_state ^= True
        elif key == Qt.Key_2:
            self.light_two_state ^= True
        elif key == Qt.Key_3:
            self.light_three_state ^= True
        self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """
        Handles keyboard release events.
        """
        key = event.key()
        self.keys_pressed.discard(key)
        self.update()
        super().keyReleaseEvent(event)

    def _update_camera_movement(self, dt) -> None:
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

        current_frame = self.timer.elapsed() * 0.001
        # Clamp delta_time to avoid jumps
        delta_time = min(dt, 0.05)  # max 50ms per frame
        self.last_frame = current_frame

        if x_direction != 0.0 or y_direction != 0.0:
            self.camera.move(x_direction, y_direction, delta_time)

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
