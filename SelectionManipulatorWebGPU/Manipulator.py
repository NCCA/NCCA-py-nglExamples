"""
Manipulator: a Maya / Houdini style transform gizmo (WebGPU port).

Draws colour-coded X/Y/Z handles at the selection pivot:
  - Translate : arrows (cylinder shaft + cone head)
  - Rotate    : rings (torus per axis)
  - Scale     : shafts ending in boxes

Geometry and the screen-space drag maths are identical to the OpenGL version.
The difference is drawing: instead of issuing GL calls this class hands back a
list of *part draws* - ``(vertex_buffer, mvp, colour)`` tuples - which main.py
renders with the stock factory ``SINGLE_COLOUR_TRIANGLES`` pipeline, one small
pass per part, on top of the scene (the caller clears the depth buffer first so
the gizmo always sits over the geometry).

Handles are picked with the same colour-ID scheme as the objects: each axis
handle has a reserved ID colour. Dragging works in screen space - the axis is
projected to the screen and mouse motion along it is converted back to a
world-space translation, per-axis scale factor, or rotation angle.
"""

import math
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import wgpu
from ncca.ngl import Mat4, PrimData, Prims, Vec3


class ManipMode(Enum):
    SELECT = "Select"
    TRANSLATE = "Translate"
    ROTATE = "Rotate"
    SCALE = "Scale"


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


AXIS_COLOURS = {
    Axis.X: (0.9, 0.15, 0.15),
    Axis.Y: (0.15, 0.8, 0.15),
    Axis.Z: (0.2, 0.35, 0.95),
}
ACTIVE_COLOUR = (1.0, 1.0, 0.2)

# reserved picking ID colours, far away from the sequential object IDs
PICK_COLOURS = {
    Axis.X: (255, 200, 10),
    Axis.Y: (255, 200, 20),
    Axis.Z: (255, 200, 30),
}

AXIS_DIRECTIONS = {
    Axis.X: Vec3(1.0, 0.0, 0.0),
    Axis.Y: Vec3(0.0, 1.0, 0.0),
    Axis.Z: Vec3(0.0, 0.0, 1.0),
}


def _axis_rotation(axis: Axis) -> Mat4:
    """Rotation taking a +Y aligned handle onto the given axis."""
    if axis == Axis.X:
        return Mat4().rotate_z(-90.0)
    if axis == Axis.Z:
        return Mat4().rotate_x(90.0)
    return Mat4()


def _world_to_screen(point: Vec3, mvp: np.ndarray, width: int, height: int):
    """Project a world point to pixel coordinates (y down, matching Qt).

    The maths classes use a row-vector convention so points transform as
    row @ matrix. Returns None if the point is behind the camera.
    """
    clip = np.array([point.x, point.y, point.z, 1.0], dtype=np.float32) @ mvp
    if clip[3] <= 0.0:
        return None
    ndc = clip[:2] / clip[3]
    sx = (ndc[0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return np.array([sx, sy], dtype=np.float32)


def _prim_buffer(
    device: wgpu.GPUDevice, data, label: str
) -> Tuple[wgpu.GPUBuffer, int]:
    """Create a vertex buffer from flat interleaved (pos,normal,uv) prim data."""
    arr = np.asarray(data, dtype=np.float32).ravel()
    buf = device.create_buffer_with_data(
        data=arr.tobytes(), usage=wgpu.BufferUsage.VERTEX, label=label
    )
    return buf, arr.size // 8


class Manipulator:
    """Transform gizmo: geometry, part draws, picking and drag maths."""

    SCREEN_SCALE = 0.18  # gizmo size as a fraction of view-space depth
    SHAFT_LENGTH = 0.85  # arrow / scale shaft length in gizmo units

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device
        self.position = Vec3(0.0, 0.0, 0.0)
        self.active_axis: Optional[Axis] = None

        # gizmo part geometry (shared across axes)
        self._buffers = {
            "shaft": _prim_buffer(
                device, PrimData.cylinder(0.02, 1.0, 12, 1), "manipShaft"
            ),
            "cone": _prim_buffer(device, PrimData.cone(0.06, 0.25, 12, 2), "manipCone"),
            "ring": _prim_buffer(
                device, PrimData.torus(0.02, 1.0, 12, 48), "manipRing"
            ),
            "box": _prim_buffer(device, PrimData.primitive(Prims.CUBE), "manipBox"),
        }

        # drag state
        self._screen_origin = None
        self._screen_axis = None
        self._pixels_per_unit = 0.0
        self._last_mouse = None
        self._last_angle = 0.0
        self._rotation_sign = 1.0

    # ------------------------------------------------------------------
    # geometry / part matrices
    # ------------------------------------------------------------------
    def _gizmo_scale(self, global_tx: Mat4, view: Mat4) -> float:
        """Scale factor giving a roughly constant on-screen size."""
        mv = (view @ global_tx).to_numpy()
        p = np.array(
            [self.position.x, self.position.y, self.position.z, 1.0], np.float32
        )
        view_pos = p @ mv
        return max(0.1, -view_pos[2]) * self.SCREEN_SCALE

    def _part_matrices(self, mode: ManipMode, size: float):
        """Yield (buffer_key, axis, local_matrix) for every handle part."""
        s = size
        shaft = Mat4().translate(0.0, s * self.SHAFT_LENGTH * 0.5, 0.0) @ Mat4().scale(
            s, s * self.SHAFT_LENGTH, s
        )
        for axis in Axis:
            rot = _axis_rotation(axis)
            if mode == ManipMode.TRANSLATE:
                head = (
                    Mat4().translate(0.0, s * self.SHAFT_LENGTH, 0.0)
                    @ Mat4().rotate_x(-90.0)
                    @ Mat4().scale(s, s, s)
                )
                yield "shaft", axis, rot @ shaft
                yield "cone", axis, rot @ head
            elif mode == ManipMode.SCALE:
                box = Mat4().translate(0.0, s * self.SHAFT_LENGTH, 0.0) @ Mat4().scale(
                    s * 0.12, s * 0.12, s * 0.12
                )
                yield "shaft", axis, rot @ shaft
                yield "box", axis, rot @ box
            elif mode == ManipMode.ROTATE:
                yield "ring", axis, rot @ Mat4().scale(s, s, s)

    def _part_draws(self, mode, global_tx, view, project, colour_for_axis):
        """Build the list of (buffer, num_verts, mvp_np, colour) for the parts."""
        if mode == ManipMode.SELECT:
            return []
        size = self._gizmo_scale(global_tx, view)
        pivot = Mat4().translate(self.position.x, self.position.y, self.position.z)
        vp = project @ view @ global_tx
        draws = []
        for key, axis, local in self._part_matrices(mode, size):
            buf, count = self._buffers[key]
            mvp = (vp @ pivot @ local).to_numpy()
            draws.append((buf, count, mvp, colour_for_axis(axis)))
        return draws

    def part_draws(self, mode, global_tx, view, project):
        """Draws for on-screen rendering (axis colours, active axis highlighted)."""

        def colour(axis: Axis):
            return ACTIVE_COLOUR if axis == self.active_axis else AXIS_COLOURS[axis]

        return self._part_draws(mode, global_tx, view, project, colour)

    def pick_draws(self, mode, global_tx, view, project):
        """Draws for the picking pass (each handle in its reserved ID colour)."""

        def colour(axis: Axis):
            r, g, b = PICK_COLOURS[axis]
            return (r / 255.0, g / 255.0, b / 255.0)

        return self._part_draws(mode, global_tx, view, project, colour)

    @staticmethod
    def axis_for_pick_colour(pixel: tuple):
        for axis, colour in PICK_COLOURS.items():
            if tuple(pixel) == colour:
                return axis
        return None

    # ------------------------------------------------------------------
    # dragging (all incremental: deltas are relative to the last event)
    # ------------------------------------------------------------------
    def start_drag(
        self,
        axis: Axis,
        mouse_x: float,
        mouse_y: float,
        global_tx: Mat4,
        view: Mat4,
        project: Mat4,
        width: int,
        height: int,
    ) -> None:
        self.active_axis = axis
        mvp = (project @ view @ global_tx).to_numpy()
        direction = AXIS_DIRECTIONS[axis]
        origin = self.position
        tip = origin + direction

        self._screen_origin = _world_to_screen(origin, mvp, width, height)
        screen_tip = _world_to_screen(tip, mvp, width, height)
        self._screen_axis = None
        self._pixels_per_unit = 0.0
        if self._screen_origin is not None and screen_tip is not None:
            axis_px = screen_tip - self._screen_origin
            length = float(np.linalg.norm(axis_px))
            # an axis pointing straight at the camera has no usable screen
            # direction, so leave it disabled rather than divide by ~zero
            if length > 1e-3:
                self._screen_axis = axis_px / length
                self._pixels_per_unit = length

        self._last_mouse = np.array([mouse_x, mouse_y], dtype=np.float32)
        self._last_angle = self._mouse_angle(mouse_x, mouse_y)
        # right-hand rule: screen-CCW drag is a positive rotation when the
        # axis points towards the camera, negative when it points away
        axis_view = (
            np.array([direction.x, direction.y, direction.z, 0.0], np.float32)
            @ (view @ global_tx).to_numpy()
        )
        self._rotation_sign = 1.0 if axis_view[2] >= 0.0 else -1.0

    def end_drag(self) -> None:
        self.active_axis = None
        self._last_mouse = None

    def _mouse_angle(self, mouse_x: float, mouse_y: float) -> float:
        """Angle of the mouse around the gizmo centre, CCW positive, degrees."""
        if self._screen_origin is None:
            return 0.0
        dx = mouse_x - self._screen_origin[0]
        dy = -(mouse_y - self._screen_origin[1])  # flip to maths orientation
        return math.degrees(math.atan2(dy, dx))

    def _mouse_step_along_axis(self, mouse_x: float, mouse_y: float) -> float:
        """Pixels moved along the screen axis since the last event."""
        if self._screen_axis is None or self._last_mouse is None:
            return 0.0
        mouse = np.array([mouse_x, mouse_y], dtype=np.float32)
        step = float(np.dot(mouse - self._last_mouse, self._screen_axis))
        self._last_mouse = mouse
        return step

    def drag_translate(self, mouse_x: float, mouse_y: float) -> Vec3:
        """World-space translation delta for this mouse move."""
        step = self._mouse_step_along_axis(mouse_x, mouse_y)
        if self._pixels_per_unit < 1e-3 or self.active_axis is None:
            return Vec3(0.0, 0.0, 0.0)
        return AXIS_DIRECTIONS[self.active_axis] * (step / self._pixels_per_unit)

    def drag_scale(self, mouse_x: float, mouse_y: float) -> Vec3:
        """Per-axis scale factors (two components are 1) for this mouse move."""
        step = self._mouse_step_along_axis(mouse_x, mouse_y)
        factor = max(0.01, 1.0 + step * 0.01)
        factors = [1.0, 1.0, 1.0]
        if self.active_axis is not None:
            factors[self.active_axis.value] = factor
        return Vec3(*factors)

    def drag_rotate(self, mouse_x: float, mouse_y: float) -> float:
        """Rotation delta in degrees around the active axis for this move."""
        angle = self._mouse_angle(mouse_x, mouse_y)
        delta = angle - self._last_angle
        self._last_angle = angle
        # keep the incremental step in (-180, 180] so crossing the atan2
        # seam doesn't produce a full-turn jump
        delta = (delta + 180.0) % 360.0 - 180.0
        return self._rotation_sign * delta
