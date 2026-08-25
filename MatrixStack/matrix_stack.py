"""A push/pop OpenGL-style matrix stack, ported from NGL9Demos/MatrixStack.

Backend-independent: no GL/Qt/wgpu imports, so both the OpenGL and WebGPU
entry points of this demo share this exact module.
"""

from __future__ import annotations

from ncca.ngl import Mat4, Quaternion, Vec3

_STACK_SIZE = 40


class MatrixStackError(Exception):
    """Raised on stack overflow or underflow."""


class MatrixStack:
    """An OpenGL-style push/pop transform stack with a fixed view/projection."""

    def __init__(self) -> None:
        self._stack: list[Mat4] = [Mat4() for _ in range(_STACK_SIZE)]
        self._top: int = 0
        self._view: Mat4 = Mat4()
        self._projection: Mat4 = Mat4()

    def push_matrix(self) -> None:
        if self._top + 1 >= _STACK_SIZE:
            raise MatrixStackError("Matrix stack overflow")
        self._top += 1
        self._stack[self._top] = self._stack[self._top - 1]

    def pop_matrix(self) -> None:
        self._stack[self._top] = Mat4()
        if self._top <= 0:
            raise MatrixStackError("Matrix stack underflow")
        self._top -= 1

    def identity(self) -> None:
        self._stack[self._top] = Mat4()

    def top(self) -> Mat4:
        return self._stack[self._top]

    def set_view(self, view: Mat4) -> None:
        self._view = view

    def set_projection(self, projection: Mat4) -> None:
        self._projection = projection

    def rotate_xyz(self, x: float, y: float, z: float) -> None:
        final = Mat4().rotate_z(z) @ Mat4().rotate_y(y) @ Mat4().rotate_x(x)
        self._stack[self._top] = self._stack[self._top] @ final

    def rotate_axis_angle(self, angle: float, x: float, y: float, z: float) -> None:
        # Quaternion.from_axis_angle() does not normalize its axis, so a
        # non-unit axis would silently bake extra scale into what is meant
        # to be a pure rotation. A zero-length axis has no defined direction
        # -- fall back to identity rotation rather than letting
        # Vec3.normalized() raise (mirrors AffineTransforms/main.py's
        # equivalent axis-angle branch).
        try:
            axis = Vec3(x, y, z).normalized()
            r = Quaternion.from_axis_angle(axis, angle).to_mat4()
        except ZeroDivisionError:
            r = Mat4()
        self._stack[self._top] = self._stack[self._top] @ r

    def translate(self, x: float, y: float, z: float) -> None:
        self._stack[self._top] = self._stack[self._top] @ Mat4().translate(x, y, z)

    def scale(self, x: float, y: float, z: float) -> None:
        self._stack[self._top] = self._stack[self._top] @ Mat4().scale(x, y, z)

    def mvp(self) -> Mat4:
        return self._projection @ self._view @ self._stack[self._top]

    def mv(self) -> Mat4:
        return self._view @ self._stack[self._top]
