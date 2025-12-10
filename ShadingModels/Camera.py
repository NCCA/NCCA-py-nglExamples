import math

from ncca.ngl import Mat4, Vec3, look_at, perspective
from PySide6.QtCore import Qt


class Camera:
    def __init__(
        self,
        width: int,
        height: int,
        fov: float = 45.0,
        near: float = 0.1,
        far: float = 100.0,
    ):
        self.width = width
        self.height = height
        self.pos = Vec3(0, 1, 4)
        self.target = Vec3(0, 0, 0)
        self.up = Vec3(0, 1, 0)

        self.fov = fov
        self.near = near
        self.far = far

        self.view = look_at(self.pos, self.target, self.up)
        self.projection = perspective(
            self.fov, self.width / self.height, self.near, self.far
        )

        # Mouse interaction state
        self.rotate = False
        self.translate = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.model_position = Vec3()
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

    def get_view_matrix(self) -> Mat4:
        return self.view

    def get_projection_matrix(self) -> Mat4:
        return self.projection

    def get_model_matrix(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        mouse_global_tx = rot_y @ rot_x
        mouse_global_tx.m_30 = self.model_position.x
        mouse_global_tx.m_31 = self.model_position.y
        mouse_global_tx.m_32 = self.model_position.z
        return mouse_global_tx

    def update_projection(self, w: int, h: int):
        self.width = w
        self.height = h
        if self.height == 0:
            self.height = 1
        self.projection = perspective(
            self.fov, self.width / self.height, self.near, self.far
        )

    def mouse_press_event(self, event):
        pos = event.position()
        if event.button() == Qt.LeftButton:
            self.rotate = True
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()
        elif event.button() == Qt.RightButton:
            self.translate = True
            self.last_mouse_x = pos.x()
            self.last_mouse_y = pos.y()

    def mouse_move_event(self, event):
        x = event.position().x()
        y = event.position().y()
        if self.rotate and event.buttons() == Qt.LeftButton:
            diff_x = x - self.last_mouse_x
            diff_y = y - self.last_mouse_y
            self.spin_x_face += 0.5 * diff_y
            self.spin_y_face += 0.5 * diff_x
        elif self.translate and event.buttons() == Qt.RightButton:
            diff_x = x - self.last_mouse_x
            diff_y = y - self.last_mouse_y
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y

        self.last_mouse_x = x
        self.last_mouse_y = y

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheel_event(self, event):
        num_pixels = event.angleDelta()
        if num_pixels.y() > 0:
            self.model_position.z += self.ZOOM
        else:
            self.model_position.z -= self.ZOOM

    def reset(self):
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.model_position.set(0, 0, 0)
