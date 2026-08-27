"""A simple RGB axis gizmo, ported from NGL9Demos/AffineTransforms/Axis."""

from __future__ import annotations

from ncca.ngl import Mat4, Prims, Transform
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib

# The gizmo builds its own unit shaft and head rather than borrowing whatever
# the host demo happens to have registered as "cylinder"/"cone" -- those are
# sized for the thing being drawn (radius 0.5 here), which made the axis
# thicker than it was long. A unit cylinder (y aligned, centred, height and
# diameter both 1) and a unit cone (z aligned, base at the origin, apex at
# z = 1) let the transform below set every dimension in units of the axis
# length, so the proportions hold at any scale.
_SHAFT = "_axis_shaft"
_HEAD = "_axis_head"
_created = False

# Proportions, as fractions of the axis half-length.
_SHAFT_RADIUS = 0.02
_HEAD_RADIUS = 0.07
_HEAD_LENGTH = 0.2


def _create_primitives() -> None:
    global _created
    if not _created:
        Primitives.create(Prims.CYLINDER, _SHAFT, 1.0, 1.0, 20, 1)
        Primitives.create(Prims.CONE, _HEAD, 1.0, 1.0, 20, 1)
        _created = True


def _load_matrices(view: Mat4, project: Mat4, global_tx: Mat4, model: Mat4) -> None:
    mv = view @ global_tx @ model
    ShaderLib.set_uniform("MVP", project @ mv)


def _draw(
    view: Mat4,
    project: Mat4,
    global_tx: Mat4,
    prim: str,
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float],
    position: tuple[float, float, float],
) -> None:
    tx = Transform()
    tx.set_scale(*scale)
    tx.set_rotation(*rotation)
    tx.set_position(*position)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw(prim)


def draw_axis(view: Mat4, project: Mat4, global_tx: Mat4, scale: float = 1.0) -> None:
    """
    Draw a red/green/blue X/Y/Z axis gizmo at the origin.

    Each axis is a thin shaft through the origin with an arrow head at both
    ends, so the negative half of each axis is as visible as the positive one.

    Parameters
    ----------
        view : Mat4
            the camera view matrix
        project : Mat4
            the camera projection matrix
        global_tx : Mat4
            the scene's mouse rotation/translation matrix
        scale : float
            half-length of each axis, i.e. the arrow tips sit at +/- scale
    """
    _create_primitives()
    ShaderLib.use(DefaultShader.COLOUR)

    r = _SHAFT_RADIUS * scale
    head_len = _HEAD_LENGTH * scale
    head_r = _HEAD_RADIUS * scale
    # The shaft stops where the heads start, so the tips land exactly on
    # +/- scale; the unit cylinder is centred, hence the full span here.
    shaft_len = 2.0 * (scale - head_len)
    tip = scale - head_len

    shaft = (r, shaft_len, r)
    head = (head_r, head_r, head_len)

    # Rotations take the primitive's own axis onto the drawn one: the shaft
    # points down +y and the head down +z, both in a right handed frame.
    axes = (
        # colour,         shaft rotation, +head rotation, -head rotation, direction
        ((1.0, 0.0, 0.0), (0, 0, -90), (0, 90, 0), (0, -90, 0), (1, 0, 0)),
        ((0.0, 1.0, 0.0), (0, 0, 0), (-90, 0, 0), (90, 0, 0), (0, 1, 0)),
        ((0.0, 0.0, 1.0), (90, 0, 0), (0, 0, 0), (0, 180, 0), (0, 0, 1)),
    )

    for colour, shaft_rot, pos_rot, neg_rot, direction in axes:
        ShaderLib.set_uniform("Colour", *colour, 1.0)
        _draw(view, project, global_tx, _SHAFT, shaft, shaft_rot, (0, 0, 0))
        offset = tuple(d * tip for d in direction)
        _draw(view, project, global_tx, _HEAD, head, pos_rot, offset)
        _draw(view, project, global_tx, _HEAD, head, neg_rot, tuple(-o for o in offset))
