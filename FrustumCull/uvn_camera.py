"""A hand-rolled UVN camera, ported from NGL9Demos/Camera (not part of ncca.ngl)."""

from __future__ import annotations

import math

from ncca.ngl import Mat3, Mat4, Plane, Vec3, look_at, perspective


def _rotate_vec3(rotation: Mat4, v: Vec3) -> Vec3:
    """Rotate a Vec3 by the rotation part of a Mat4.

    Mat4 only multiplies with Vec4, so the rotation is extracted into a Mat3
    which does support Mat3 @ Vec3.
    """
    return Mat3.from_mat4(rotation) @ v


class UVNCamera:
    def __init__(
        self,
        eye: Vec3,
        look: Vec3,
        up: Vec3,
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ) -> None:
        self.eye = Vec3(eye.x, eye.y, eye.z)
        self.look = Vec3(look.x, look.y, look.z)
        self.up = Vec3(up.x, up.y, up.z)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self._update_view()
        self._update_projection()

    def _update_view(self) -> None:
        self.view = look_at(self.eye, self.look, self.up)
        n = (self.look - self.eye).normalized()
        u = n.cross(self.up).normalized()
        v = u.cross(n).normalized()
        self.n = n
        self.u = u
        self.v = v

    def _update_projection(self) -> None:
        self.project = perspective(self.fov, self.aspect, self.near, self.far)

    def set_shape(self, fov: float, aspect: float, near: float, far: float) -> None:
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self._update_projection()

    def move_eye(self, dx: float, dy: float, dz: float) -> None:
        self.eye = self.eye + self.u * dx + self.v * dy + self.n * dz
        self._update_view()

    def move_look(self, dx: float, dy: float, dz: float) -> None:
        self.look = self.look + self.u * dx + self.v * dy + self.n * dz
        self._update_view()

    def move_both(self, dx: float, dy: float, dz: float) -> None:
        offset = self.u * dx + self.v * dy + self.n * dz
        self.eye = self.eye + offset
        self.look = self.look + offset
        self._update_view()

    def slide(self, dx: float, dy: float, dz: float) -> None:
        self.move_both(dx, dy, dz)

    def roll(self, degrees: float) -> None:
        r = Mat4().rotate_z(degrees)
        self.up = _rotate_vec3(r, self.up).normalized()
        self._update_view()

    def pitch(self, degrees: float) -> None:
        r = Mat4().rotate_x(degrees)
        self.look = self.eye + _rotate_vec3(r, self.look - self.eye)
        self._update_view()

    def yaw(self, degrees: float) -> None:
        r = Mat4().rotate_y(degrees)
        self.look = self.eye + _rotate_vec3(r, self.look - self.eye)
        self._update_view()


class FrustumCamera(UVNCamera):
    """A UVNCamera that also extracts its 6 view-frustum planes for culling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.planes: dict[str, Plane] = {}
        self.corners: list[Vec3] = []
        self.calculate_frustum()

    def calculate_frustum(self) -> None:
        near_height = 2.0 * math.tan(math.radians(self.fov) / 2.0) * self.near
        near_width = near_height * self.aspect
        far_height = 2.0 * math.tan(math.radians(self.fov) / 2.0) * self.far
        far_width = far_height * self.aspect

        near_center = self.eye + self.n * self.near
        far_center = self.eye + self.n * self.far

        ntl = near_center + self.v * (near_height / 2) - self.u * (near_width / 2)
        ntr = near_center + self.v * (near_height / 2) + self.u * (near_width / 2)
        nbl = near_center - self.v * (near_height / 2) - self.u * (near_width / 2)
        nbr = near_center - self.v * (near_height / 2) + self.u * (near_width / 2)

        ftl = far_center + self.v * (far_height / 2) - self.u * (far_width / 2)
        ftr = far_center + self.v * (far_height / 2) + self.u * (far_width / 2)
        fbl = far_center - self.v * (far_height / 2) - self.u * (far_width / 2)
        fbr = far_center - self.v * (far_height / 2) + self.u * (far_width / 2)

        self.planes["NEAR"] = Plane(ntl, ntr, nbr)
        self.planes["FAR"] = Plane(ftr, ftl, fbl)
        self.planes["LEFT"] = Plane(ntl, nbl, fbl)
        self.planes["RIGHT"] = Plane(ntr, ftr, fbr)
        self.planes["TOP"] = Plane(ntl, ftl, ftr)
        self.planes["BOTTOM"] = Plane(nbl, nbr, fbr)
        self.corners = [ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr]

    def is_sphere_in_frustum(self, center: Vec3, radius: float) -> str:
        result = "INSIDE"
        for plane in self.planes.values():
            d = plane.distance(center)
            if d < -radius:
                return "OUTSIDE"
            if d < radius:
                result = "INTERSECT"
        return result

    def set_shape(self, fov: float, aspect: float, near: float, far: float) -> None:
        super().set_shape(fov, aspect, near, far)
        self.calculate_frustum()

    def move_eye(self, dx: float, dy: float, dz: float) -> None:
        super().move_eye(dx, dy, dz)
        self.calculate_frustum()
