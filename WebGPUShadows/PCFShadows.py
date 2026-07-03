#!/usr/bin/env -S uv run --script
import sys
from typing import List, Set, Tuple

import wgpu
from ncca.ngl import (
    FirstPersonCamera,
    Mat4,
    PerspMode,
    Transform,
    Vec3,
    Vec4,
)
from Pipeline import Pipeline
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a multi-mesh scene.

    This class sets up a WebGPU scene with multiple mesh objects, lighting,
    and a first-person camera controlled by mouse and keyboard.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Multi Mesh")
        self.device: wgpu.GPUDevice = None
        self.rotate: bool = False
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0

        self.light_one_state = True
        self.light_rotation = 0.0
        self.keys_pressed: Set[Qt.Key] = set()
        self.timer: QElapsedTimer = QElapsedTimer()
        self.timer.start()
        self.last_frame: float = 1.0

        self.pipeline = None
        self.camera: FirstPersonCamera = FirstPersonCamera(
            Vec3(0.0, 2.0, 8.0), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
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
        current_frame = self.timer.elapsed() * 0.001
        delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        self._update_camera_movement(delta_time)

        self.pipeline.update_lights(self.light_one_state)

        # Animate the single light
        self.light_rotation += 0.5
        light_rot_mat = Mat4().rotate_y(self.light_rotation)

        light_pos_1 = Vec4(0.0, 4.0, 3.0, 1.0)
        rotated_light_1 = light_rot_mat @ light_pos_1

        self.pipeline.update_light_positions([rotated_light_1])

        # 1. Define the objects in our scene as a list of tuples:
        # (mesh_name, transform_matrix, colour)
        scene_objects: List[Tuple[str, Mat4, Tuple[float, float, float, float]]] = []

        tx = Transform()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-1.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("buddah", tx.matrix(), (1, 0, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(-2.0, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("bunny", tx.matrix(), (1, 0, 1, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, 0.0)  # Move it to the right
        scene_objects.append(("teapot", tx.matrix(), (0, 1, 0, 1)))

        tx.reset()
        tx.set_scale(0.1, 0.1, 0.1)
        tx.set_position(1.5, -0.5, 0.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dragon", tx.matrix(), (1, 1, 0, 1)))

        tx.reset()
        tx.set_position(0.0, 0.1, 1.0)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("troll", tx.matrix(), (0, 0.2, 1, 1)))
        tx.reset()
        tx.set_position(-1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, 0, 0)
        scene_objects.append(("icosahedron", tx.matrix(), (0.2, 0.2, 0.8, 1)))
        tx.reset()
        tx.set_position(-2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("dodecahedron", tx.matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("football", tx.matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("tetrahedron", tx.matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(1.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("octahedron", tx.matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        tx.set_rotation(0, -90, 0)
        scene_objects.append(("cube", tx.matrix(), (0.8, 0.2, 0.2, 1)))

        tx.reset()
        tx.set_position(0, -0.5, 0)
        scene_objects.append(("floor", tx.matrix(), (1, 1, 1, 1)))

        ## Render single light
        tx.reset()
        tx.set_position(rotated_light_1.x, rotated_light_1.y, rotated_light_1.z)
        scene_objects.append(("light1", tx.matrix(), (1, 1, 1, 1)))

        # 2. Pass the entire scene to the pipeline to be rendered
        self.pipeline.render(
            self.colour_buffer_texture_view,
            self.multisample_texture_view,
            self.depth_buffer_view,
            self.texture_size,
            scene_objects,
        )

        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Handles window resize events.

        Updates the camera's projection matrix to match the new aspect ratio.
        """
        ratio = self.devicePixelRatio()
        w = width * ratio
        h = height * ratio
        self.camera.set_projection(45.0, (w / h), 0.1, 300.0, PerspMode.WebGPU)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles keyboard press events for camera control and toggling lights.
        """
        key = event.key()
        self.keys_pressed.add(key)

        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_1:
            self.light_one_state ^= True
        self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """
        Handles keyboard release events.
        """
        key = event.key()
        self.keys_pressed.discard(key)
        self.update()
        super().keyReleaseEvent(event)

    def _update_camera_movement(self, dt: float) -> None:
        """
        (Internal) Calculates and applies camera movement based on currently pressed keys.
        """
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

        delta_time = min(dt, 0.05)  # Clamp to avoid jumps

        if x_direction != 0.0 or y_direction != 0.0:
            self.camera.move(x_direction, y_direction, delta_time)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse movement for camera rotation (orbiting).
        """
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.camera.process_mouse_movement(diff_x, -diff_y)
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse button presses to initiate camera rotation.
        """
        if event.button() == Qt.LeftButton:
            position = event.position()
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse button releases to stop camera rotation.
        """
        if event.button() == Qt.LeftButton:
            self.rotate = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events for zooming the camera.
        """
        num_pixels = event.angleDelta().y()
        self.camera.process_mouse_scroll(num_pixels * 0.01)
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
