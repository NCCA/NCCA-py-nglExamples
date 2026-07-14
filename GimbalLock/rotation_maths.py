"""Pure-numpy rotation maths for the GimbalLock demo.

No GL/Qt imports here (bar `ncca.ngl.Mat4`/`Quaternion`, which are themselves
numpy-only) so this module -- and the gimbal-lock argument it encodes -- can
be pytest'd headless.

Convention
----------
Row-vector, matching the rest of `ncca.ngl`: a point transforms as
``v_row @ M``, so composing two rotations as ``A @ B`` (plain numpy matmul,
no tricks) applies ``A`` to the vector first, then ``B``. `rotate_x`,
`rotate_y`, `rotate_z` below are bit-identical to `Mat4.rotate_x/y/z`.

`euler_to_mat` composes explicitly as::

    m = rotate_x(rx) @ rotate_y(ry) @ rotate_z(rz)

i.e. roll (X) is applied first, then pitch (Y), then yaw (Z) -- the classic
aircraft order. This is the whole lesson: the GimbalLock GL rig nests its
rings the same way (Z ring outermost, carrying a Y ring, carrying an X ring,
carrying the aeroplane), so the picture and the maths agree about which
axis goes "inside" which. Get the two out of step and the rings would spin
in an order that doesn't match what the numbers say -- a very easy way to
teach the wrong lesson by accident.

(Note for readers of `ncca.ngl` call sites such as
``Mat4.rotate_z(rz) @ Mat4.rotate_y(ry) @ Mat4.rotate_x(rx)``: `Mat4.__matmul__`
deliberately reverses its operands, so that chain produces the *same*
X-then-Y-then-Z application order as the line above, just spelled
back-to-front. Plain numpy has no such trick, hence the explicit
`rotate_x @ rotate_y @ rotate_z` order here.)

Gimbal lock itself: at ry = +/-90 the X and Z rotations end up spinning
about the same world axis, so the triple (rx, ry, rz) stops being a unique
description of an orientation -- a whole line of (rx, rz) pairs (those
sharing rx - rz) map to the same matrix. See
`TestGimbalLockCollapse` in the tests for the demonstration.
"""

import numpy as np
from ncca.ngl import Mat4, Quaternion

_LOCK_EPS = 1e-3


def rotate_x(angle_deg: float) -> np.ndarray:
    """3x3 rotation about X by angle_deg degrees (row-vector convention)."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    m = np.eye(3)
    m[1, 1] = c
    m[1, 2] = s
    m[2, 1] = -s
    m[2, 2] = c
    return m


def rotate_y(angle_deg: float) -> np.ndarray:
    """3x3 rotation about Y by angle_deg degrees (row-vector convention)."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    m = np.eye(3)
    m[0, 0] = c
    m[0, 2] = -s
    m[2, 0] = s
    m[2, 2] = c
    return m


def rotate_z(angle_deg: float) -> np.ndarray:
    """3x3 rotation about Z by angle_deg degrees (row-vector convention)."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    m = np.eye(3)
    m[0, 0] = c
    m[0, 1] = s
    m[1, 0] = -s
    m[1, 1] = c
    return m


def euler_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return the 4x4 homogeneous rotation matrix for Euler angles in degrees.

    Composed as ``rotate_x(rx) @ rotate_y(ry) @ rotate_z(rz)`` -- X applied
    first. See the module docstring for why this order matters.
    """
    r3 = rotate_x(rx) @ rotate_y(ry) @ rotate_z(rz)
    m = np.eye(4)
    m[:3, :3] = r3
    return m


def is_gimbal_locked(ry: float, eps: float = _LOCK_EPS) -> bool:
    """True when the middle (pitch/Y) axis sits at +/-90 degrees.

    At that pose ``|cos(ry)| < eps``: the outer (Z) and inner (X) rings have
    rotated into the same plane, so they now spin about the same world
    axis -- the definition of gimbal lock for an XYZ Euler rig.
    """
    return abs(np.cos(np.radians(ry))) < eps


def lost_dof_axis(
    rx: float, ry: float, rz: float, eps: float = _LOCK_EPS
) -> str | None:
    """Return which degree of freedom has collapsed, or None if not locked.

    Ignores rx/rz -- the collapse is a function of ry alone -- but takes
    them so call sites can pass the full pose without unpacking it first.
    """
    del rx, rz  # collapse depends only on ry; kept for a uniform call site
    if is_gimbal_locked(ry, eps):
        return "x/z (roll and yaw now spin about the same world axis)"
    return None


def euler_to_quat(rx: float, ry: float, rz: float) -> Quaternion:
    """Build the Quaternion for the same Euler angles via `ncca.ngl.Mat4`.

    Uses the same explicit-compose idiom as `QuatSlerp` and the "avoid
    `Transform.set_rotation` with more than one axis" convention: build the
    matrix by hand, then let `Quaternion.from_mat4` read it off. `Mat4`'s
    `@` reverses its operands (see the module docstring), so writing
    ``rotate_z @ rotate_y @ rotate_x`` here applies X first -- the same
    order as `euler_to_mat` above.
    """
    m = Mat4().rotate_z(rz) @ Mat4().rotate_y(ry) @ Mat4().rotate_x(rx)
    return Quaternion.from_mat4(m)


def quat_to_mat(q: Quaternion) -> np.ndarray:
    """Return the 4x4 homogeneous rotation matrix a Quaternion represents."""
    return q.to_mat4().to_numpy()
