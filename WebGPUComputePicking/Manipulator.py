"""
Manipulator: a Maya / Houdini style transform gizmo (WebGPU, integer-ID picking).

Draws colour-coded X/Y/Z handles at the selection pivot:
  - Translate : arrows (cylinder shaft + cone head)
  - Rotate    : rings (torus per axis)
  - Scale     : shafts ending in boxes

Geometry and the screen-space drag maths are identical to the other
manipulator demos. Instead of issuing draw calls this class hands back a
list of *part draws* - ``(vertex_buffer, count, mvp, colour_or_id)`` tuples -
which main.py renders one small pass per part, on top of the scene (the
caller clears the depth buffer on the first part so the gizmo always sits
over the geometry).

Handles are picked with the same integer-ID scheme as the objects: each
handle has a reserved ID at the top of the 20-bit ID range (GIZMO_ID_BASE
up). The compute kernel treats those priority IDs as distance zero, so a
handle anywhere inside the pick block beats every object - the integer
version of the "gizmo colours win over object colours" rule in the
colour-ID demo.
"""

import math
from enum import Enum

import numpy as np
import wgpu
from ncca.ngl import Mat4, PrimData, Prims, Vec3

# reserved handle IDs at the top of the 20-bit ID range; must stay above
# every object ID and below (1 << ID_BITS), and must match PRIORITY_BASE
# in PickCompute.wgsl
GIZMO_ID_BASE = 0xFFF00


class ManipMode(Enum):
    SELECT = "Select"
    TRANSLATE = "Translate"
    ROTATE = "Rotate"
    SCALE = "Scale"


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


# the centre handle: a cube at the pivot. In translate mode it drags freely in
# the screen plane, in scale mode it scales all three axes uniformly. It is not
# an Axis (it has no direction) so it gets its own sentinel/colours.
CENTER = "center"

AXIS_COLOURS = {
    Axis.X: (0.9, 0.15, 0.15),
    Axis.Y: (0.15, 0.8, 0.15),
    Axis.Z: (0.2, 0.35, 0.95),
}
CENTER_COLOUR = (0.85, 0.85, 0.85)
ACTIVE_COLOUR = (1.0, 1.0, 0.2)

# reserved integer pick IDs - just GIZMO_ID_BASE plus an offset, no colour
# encoding needed
PICK_IDS = {
    Axis.X: GIZMO_ID_BASE + 1,
    Axis.Y: GIZMO_ID_BASE + 2,
    Axis.Z: GIZMO_ID_BASE + 3,
}
CENTER_PICK_ID = GIZMO_ID_BASE + 4

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
) -> tuple[wgpu.GPUBuffer, int]:
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
        self.active_axis: Axis | None = None

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
        # centre-handle state (free translate / uniform scale)
        self._center_right = np.array([1.0, 0.0, 0.0], np.float32)
        self._center_up = np.array([0.0, 1.0, 0.0], np.float32)
        self._center_pixels_per_unit = 1.0
        self._last_distance = 0.0

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
        # plain float: Vec3 * scalar rejects numpy scalar types
        return float(max(0.1, -view_pos[2])) * self.SCREEN_SCALE

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
        # centre cube: free translate / uniform scale handle
        if mode in (ManipMode.TRANSLATE, ManipMode.SCALE):
            centre = Mat4().scale(s * 0.16, s * 0.16, s * 0.16)
            yield "box", CENTER, centre

    def _part_draws(self, mode, global_tx, view, project, value_for_axis):
        """Build the list of (buffer, num_verts, mvp_np, value) for the parts."""
        if mode == ManipMode.SELECT:
            return []
        size = self._gizmo_scale(global_tx, view)
        pivot = Mat4().translate(self.position.x, self.position.y, self.position.z)
        vp = project @ view @ global_tx
        draws = []
        for key, axis, local in self._part_matrices(mode, size):
            buf, count = self._buffers[key]
            mvp = (vp @ pivot @ local).to_numpy()
            draws.append((buf, count, mvp, value_for_axis(axis)))
        return draws

    def part_draws(self, mode, global_tx, view, project):
        """Draws for on-screen rendering (axis colours, active axis highlighted)."""

        def colour(axis):
            if axis == self.active_axis:
                return ACTIVE_COLOUR
            return CENTER_COLOUR if axis == CENTER else AXIS_COLOURS[axis]

        return self._part_draws(mode, global_tx, view, project, colour)

    def id_draws(self, mode, global_tx, view, project):
        """Draws for the ID pass (each handle carrying its reserved pick ID)."""

        def pick_id(axis):
            return CENTER_PICK_ID if axis == CENTER else PICK_IDS[axis]

        return self._part_draws(mode, global_tx, view, project, pick_id)

    @staticmethod
    def axis_for_pick_id(pick_id: int):
        """Map a reserved handle ID back to its Axis / CENTER, or None."""
        if pick_id == CENTER_PICK_ID:
            return CENTER
        for axis, reserved in PICK_IDS.items():
            if pick_id == reserved:
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
        if axis == CENTER:
            self._start_center_drag(
                mvp, view, global_tx, mouse_x, mouse_y, width, height
            )
            return
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

    def _start_center_drag(
        self, mvp, view: Mat4, global_tx: Mat4, mouse_x, mouse_y, width, height
    ) -> None:
        """Set up screen-plane basis and scale reference for the centre handle."""
        # columns of the model-view 3x3 are the object-space directions that map
        # to view +X / +Y, i.e. screen right / up at the pivot
        mv3 = (view @ global_tx).to_numpy()[:3, :3]
        self._center_right = np.ascontiguousarray(mv3[:, 0], np.float32)
        self._center_up = np.ascontiguousarray(mv3[:, 1], np.float32)

        self._screen_origin = _world_to_screen(self.position, mvp, width, height)
        tip = self.position + Vec3(*self._center_right)
        screen_tip = _world_to_screen(tip, mvp, width, height)
        self._center_pixels_per_unit = 1.0
        if self._screen_origin is not None and screen_tip is not None:
            length = float(np.linalg.norm(screen_tip - self._screen_origin))
            self._center_pixels_per_unit = max(1e-3, length)

        self._last_mouse = np.array([mouse_x, mouse_y], np.float32)
        if self._screen_origin is not None:
            self._last_distance = float(
                np.linalg.norm(self._last_mouse - self._screen_origin)
            )
        else:
            self._last_distance = 0.0

    def drag_free_translate(self, mouse_x: float, mouse_y: float) -> Vec3:
        """World-space translation delta from screen-plane mouse motion."""
        if self._last_mouse is None:
            return Vec3(0.0, 0.0, 0.0)
        mouse = np.array([mouse_x, mouse_y], np.float32)
        dpx = mouse - self._last_mouse
        self._last_mouse = mouse
        wx = float(dpx[0]) / self._center_pixels_per_unit
        wy = -float(dpx[1]) / self._center_pixels_per_unit  # screen y is down
        delta = Vec3(*self._center_right) * wx + Vec3(*self._center_up) * wy
        return delta

    def drag_uniform_scale(self, mouse_x: float, mouse_y: float) -> Vec3:
        """Uniform scale factor (same on all axes) from distance to the pivot."""
        if self._screen_origin is None:
            return Vec3(1.0, 1.0, 1.0)
        mouse = np.array([mouse_x, mouse_y], np.float32)
        distance = float(np.linalg.norm(mouse - self._screen_origin))
        step = distance - self._last_distance
        self._last_distance = distance
        factor = max(0.01, 1.0 + step * 0.01)
        return Vec3(factor, factor, factor)

    def drag_rotate(self, mouse_x: float, mouse_y: float) -> float:
        """Rotation delta in degrees around the active axis for this move."""
        angle = self._mouse_angle(mouse_x, mouse_y)
        delta = angle - self._last_angle
        self._last_angle = angle
        # keep the incremental step in (-180, 180] so crossing the atan2
        # seam doesn't produce a full-turn jump
        delta = (delta + 180.0) % 360.0 - 180.0
        return self._rotation_sign * delta
