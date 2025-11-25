#!/usr/bin/env -S uv run --script
import sys
from typing import List, Set, Tuple

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
from PySide6.QtCore import QElapsedTimer, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

from Pipeline import Pipeline
from WebGPUWidget import WebGPUWidget


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a multi-mesh scene.

    This class sets up a WebGPU scene with multiple mesh objects, lighting,
    and a first-person camera controlled by mouse and keyboard.
    """

    def __init__(self) -> None:
        """
        Initializes the WebGPU scene, camera, and mouse/keyboard controls.
        """
        super().__init__()
        self.setWindowTitle("WebGPU Multi Mesh")
        self.device: wgpu.GPUDevice = None
        self.rotate: bool = False
        self.translate: bool = False
        self.spin_x_face: int = 0
        self.spin_y_face: int = 0
        self.original_x_rotation: int = 0
        self.original_y_rotation: int = 0
        self.original_x_pos: int = 0
        self.original_y_pos: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.1
        self.light_one_state: bool = True
        self.light_two_state: bool = True
        self.light_three_state: bool = True
        self.keys_pressed: Set[Qt.Key] = set()
        self.timer: QElapsedTimer = QElapsedTimer()
        self.timer.start()
        self.last_frame: float = 1.0
        self.pipeline: Pipeline = None
        self.camera: FirstPersonCamera = FirstPersonCamera(
            Vec3(0, 2, 8), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, PerspMode.WebGPU
        )

        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        (Internal) Initializes the WebGPU device and creates the render pipeline.
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
        (Internal) Creates the render pipeline instance.
        """
        self.pipeline = Pipeline(self.device, self.camera)

    def paintWebGPU(self) -> None:
        """
        The main rendering method, called each frame.

        This method updates camera, lights, defines the scene objects,
        and calls the pipeline to render the scene.
        """
        current_frame = self.timer.elapsed() * 0.001
        delta_time = current_frame - self.last_frame
        self.last_frame = current_frame
        self._update_camera_movement(delta_time)
        self.pipeline.update_lights(self.light_one_state, self.light_two_state, self.light_three_state)
        scene_objects: List[Tuple[str, Mat4, Tuple[float, float, float, float]]] = []

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
        tx.set_position(0.0, 0.0, 0.0)
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
        scene_objects.append(("icosahedron", tx.get_matrix(), (0.2, 0.2, 0.8, 1)))
        tx.reset()
        tx.set_position(-2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        scene_objects.append(("dodecahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(2.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        scene_objects.append(("football", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(1.0, 0.0, 1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        scene_objects.append(("tetrahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(1.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        scene_objects.append(("octahedron", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(0.0, 0.0, -1.0)
        tx.set_scale(0.5, 0.5, 0.5)
        scene_objects.append(("cube", tx.get_matrix(), (0.8, 0.2, 0.2, 1)))
        tx.reset()
        tx.set_position(0, -0.5, 0)
        scene_objects.append(("floor", tx.get_matrix(), (1, 1, 1, 1)))

        tx.reset()
        tx.set_position(0.0, 1.0, 1.0)
        scene_objects.append(("light1", tx.get_matrix(), (1, 1, 1, 1)))
        tx.reset()
        tx.set_position(-1.0, 1.0, -1.0)
        scene_objects.append(("light2", tx.get_matrix(), (1, 1, 1, 1)))
        tx.set_position(1.0, 1.0, -1.0)
        scene_objects.append(("light3", tx.get_matrix(), (1, 1, 1, 1)))

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
        elif key == Qt.Key_2:
            self.light_two_state ^= True
        elif key == Qt.Key_3:
            self.light_three_state ^= True
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


def main() -> None:
    """
    Main function to create and run the QApplication.
    """
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
