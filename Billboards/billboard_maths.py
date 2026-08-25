"""Camera-facing basis maths for billboarded sprites.

This module is deliberately numpy-only (no GL, no Qt) so the basis
extraction and depth sorting can be unit tested headless, same pattern as
``RayPickingSelection/picking_maths.py`` and ``Blending/blend_scene.py``.

All matrices follow the repo's row-vector convention: a point transforms as
``row_vector @ matrix``. For a view matrix built the way ``ncca.ngl.look_at``
builds one, that convention puts the camera's world-space *right* axis in
column 0, its world-space *up* axis in column 1, and its world-space
*backward* axis (pointing from the look-at target back towards the eye) in
column 2 -- equivalently, rows 0..2 of ``view.T``. Billboarding is just
reading those columns back out and using them as the quad's edge vectors
instead of the model's own rotation, so every billboard turns to face the
same way the camera is looking, no matter how the scene itself is spun.
"""

import numpy as np

# world +y, used to lock the cylindrical mode's up vector
_WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
# used as a last-resort right axis when the camera looks straight down/up
# and cross(world_up, backward) collapses to zero -- see cylindrical_basis
_FALLBACK_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)

_ZERO_CROSS_EPSILON = 1e-8


def spherical_basis(view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right/up vectors that always face the camera dead-on.

    Straight lift of columns 0 and 1 of the view matrix (rows 0/1 of
    ``view.T``) back into world space: whatever rotation the view matrix
    applies to turn world space into camera space, this undoes it for just
    the billboard's two edge vectors. The implied quad normal (``right x
    up``) then points exactly at the camera from any angle -- this is the
    "always correct" mode, used for particles, impostors and HUD markers.
    """
    v = np.asarray(view, dtype=np.float64)
    right = v[:3, 0].copy()
    up = v[:3, 1].copy()
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    return right, up


def cylindrical_basis(view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right/up vectors that face the camera around the vertical axis only.

    ``up`` is locked to world +y (trees, lampposts, anything that should
    stay upright rather than tilt with the camera). ``right`` is rebuilt
    from scratch as ``cross(world_up, backward)``, where ``backward`` is
    the view matrix's column 2 (the world-space direction from the target
    back to the eye) -- this is the same "read the camera's own axes back
    out of the view matrix" trick as spherical_basis, just with one axis
    pinned instead of copied.

    Degenerate case: when the camera looks straight up or down, ``backward``
    is parallel to world +y, so the cross product is zero and there is no
    well-defined horizontal "sideways" for the billboard to face. Real
    engines hit the same wall (a tree billboard has no sensible answer to
    "which way does it face" viewed from directly above). Rather than
    return NaN, this falls back to a fixed right axis of world +x, so the
    quad stays a valid (if arbitrarily oriented) rectangle instead of
    collapsing to a degenerate one -- documented here and pinned by
    ``test_degenerate_straight_down_view_has_no_nan``.
    """
    v = np.asarray(view, dtype=np.float64)
    backward = v[:3, 2].copy()
    backward /= np.linalg.norm(backward)

    right = np.cross(_WORLD_UP, backward)
    norm = np.linalg.norm(right)
    if norm < _ZERO_CROSS_EPSILON:
        right = _FALLBACK_RIGHT.copy()
    else:
        right /= norm
    return right, _WORLD_UP.copy()


def billboard_depth(position: np.ndarray, view: np.ndarray) -> float:
    """View-space depth of a world-space billboard centre.

    Same idea as ``Blending/blend_scene.py``'s ``view_space_z``, just for a
    bare world-space point rather than a point transformed through a model
    matrix first (billboards have no rotation/scale of their own to fold
    in -- their position *is* their placement). The camera looks down -z in
    view space, so a more negative result is further away.
    """
    p = np.asarray(position, dtype=np.float64)
    homogeneous = np.array([p[0], p[1], p[2], 1.0])
    return float((homogeneous @ np.asarray(view, dtype=np.float64))[2])


def back_to_front(positions: np.ndarray, view: np.ndarray) -> list[int]:
    """Indices of the given billboard positions sorted furthest first.

    Required for correct alpha blending (the OVER operator is order
    dependent): draw the furthest sprite first so nearer, more-visible
    sprites composite on top of it, exactly the same reasoning as the
    Blending demo's ``back_to_front`` for its transparent panels.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    view = np.asarray(view, dtype=np.float64)
    return sorted(
        range(len(positions)), key=lambda i: billboard_depth(positions[i], view)
    )
