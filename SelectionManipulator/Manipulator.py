"""
Manipulator: a Maya / Houdini style transform gizmo.

Draws colour-coded X/Y/Z handles at the selection pivot:
  - Translate : arrows (cylinder shaft + cone head)
  - Rotate    : rings (torus per axis)
  - Scale     : shafts ending in boxes

The gizmo is drawn after clearing the depth buffer so it always sits on top
of the scene, and is scaled by its view-space depth so it keeps a roughly
constant size on screen.

Handles are picked with the same colour-ID scheme as the objects (each axis
handle has a reserved ID colour). Dragging works in screen space: the axis
is projected to the screen and mouse motion along it is converted back to a
world-space translation, per-axis scale factor, or rotation angle.
"""

import math
from enum import Enum

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat4, Prims, Vec3, Vec4
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib


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
    Axis.X: Vec4(0.9, 0.15, 0.15, 1.0),
    Axis.Y: Vec4(0.15, 0.8, 0.15, 1.0),
    Axis.Z: Vec4(0.2, 0.35, 0.95, 1.0),
}
CENTER_COLOUR = Vec4(0.85, 0.85, 0.85, 1.0)
ACTIVE_COLOUR = Vec4(1.0, 1.0, 0.2, 1.0)

# reserved picking ID colours, far away from the sequential object IDs
PICK_COLOURS = {
    Axis.X: (255, 200, 10),
    Axis.Y: (255, 200, 20),
    Axis.Z: (255, 200, 30),
}
CENTER_PICK_COLOUR = (255, 200, 40)

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


class Manipulator:
    """Transform gizmo: drawing, picking and screen-space drag maths."""

    SCREEN_SCALE = 0.18  # gizmo size as a fraction of view-space depth
    SHAFT_LENGTH = 0.85  # arrow / scale shaft length in gizmo units

    def __init__(self) -> None:
        self.position = Vec3(0.0, 0.0, 0.0)
        self.active_axis: Axis | None = None
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

    @staticmethod
    def create_geometry() -> None:
        """Create the shared gizmo primitives (call once in initializeGL)."""
        Primitives.create(Prims.CYLINDER, "manipShaft", 0.02, 1.0, 12, 1)
        Primitives.create(Prims.CONE, "manipCone", 0.06, 0.25, 12, 2)
        Primitives.create(Prims.TORUS, "manipRing", 0.02, 1.0, 12, 48)

    # ------------------------------------------------------------------
    # drawing
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
        """Yield (prim_name, axis, local_matrix) for every handle part."""
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
                yield "manipShaft", axis, rot @ shaft
                yield "manipCone", axis, rot @ head
            elif mode == ManipMode.SCALE:
                box = Mat4().translate(0.0, s * self.SHAFT_LENGTH, 0.0) @ Mat4().scale(
                    s * 0.12, s * 0.12, s * 0.12
                )
                yield "manipShaft", axis, rot @ shaft
                yield "cube", axis, rot @ box
            elif mode == ManipMode.ROTATE:
                yield "manipRing", axis, rot @ Mat4().scale(s, s, s)
        # centre cube: free translate / uniform scale handle
        if mode in (ManipMode.TRANSLATE, ManipMode.SCALE):
            centre = Mat4().scale(s * 0.16, s * 0.16, s * 0.16)
            yield "cube", CENTER, centre

    def _draw_parts(self, mode, global_tx, view, project, colour_for_axis) -> None:
        size = self._gizmo_scale(global_tx, view)
        pivot = Mat4().translate(self.position.x, self.position.y, self.position.z)
        vp = project @ view @ global_tx
        ShaderLib.use(DefaultShader.COLOUR)
        for prim, axis, local in self._part_matrices(mode, size):
            ShaderLib.set_uniform("MVP", vp @ pivot @ local)
            ShaderLib.set_uniform("Colour", colour_for_axis(axis))
            Primitives.draw(prim)

    def draw(self, mode: ManipMode, global_tx: Mat4, view: Mat4, project: Mat4) -> None:
        """Draw the gizmo on top of the scene (clears the depth buffer)."""
        if mode in (ManipMode.SELECT,):
            return
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        def colour(axis) -> Vec4:
            if axis == self.active_axis:
                return ACTIVE_COLOUR
            return CENTER_COLOUR if axis == CENTER else AXIS_COLOURS[axis]

        self._draw_parts(mode, global_tx, view, project, colour)

    def draw_pick(self, mode, global_tx: Mat4, view: Mat4, project: Mat4) -> None:
        """Draw the handles in their reserved ID colours for picking."""
        if mode in (ManipMode.SELECT,):
            return
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        def colour(axis) -> Vec4:
            r, g, b = CENTER_PICK_COLOUR if axis == CENTER else PICK_COLOURS[axis]
            return Vec4(r / 255.0, g / 255.0, b / 255.0, 1.0)

        self._draw_parts(mode, global_tx, view, project, colour)

    @staticmethod
    def axis_for_pick_colour(pixel: tuple):
        if tuple(pixel) == CENTER_PICK_COLOUR:
            return CENTER
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
