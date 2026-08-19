"""A simple RGB axis gizmo, ported from NGL9Demos/AffineTransforms/Axis."""

from __future__ import annotations

from ncca.ngl import Mat4, Transform
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib


def _load_matrices(view: Mat4, project: Mat4, global_tx: Mat4, model: Mat4) -> None:
    mv = view @ global_tx @ model
    ShaderLib.set_uniform("MVP", project @ mv)


def draw_axis(view: Mat4, project: Mat4, global_tx: Mat4, scale: float = 1.5) -> None:
    """Draw a red/green/blue X/Y/Z axis gizmo at the origin.

    Requires Primitives.load_default_primitives() to already have been
    called (draws "cylinder" and "cone").
    """
    ShaderLib.use(DefaultShader.COLOUR)
    tx = Transform()

    # X axis (red)
    ShaderLib.set_uniform("Colour", 1.0, 0.0, 0.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(scale, 0, 0)
    tx.set_rotation(0, 90, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(scale, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(-scale, 0, 0)
    tx.set_rotation(0, -90, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")

    # Y axis (green)
    ShaderLib.set_uniform("Colour", 0.0, 1.0, 0.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(0, -scale, 0)
    tx.set_rotation(90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(0, scale, 0)
    tx.set_rotation(-90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(0, -scale, 0)
    tx.set_rotation(90, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")

    # Z axis (blue)
    ShaderLib.set_uniform("Colour", 0.0, 0.0, 1.0, 1.0)
    tx.set_scale(scale, scale, scale * 2)
    tx.set_position(0, 0, scale)
    tx.set_rotation(0, 0, -90)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cylinder")
    tx.set_position(0, 0, scale)
    tx.set_rotation(0, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
    tx.set_position(0, 0, -scale)
    tx.set_rotation(180, 0, 0)
    _load_matrices(view, project, global_tx, tx.matrix())
    Primitives.draw("cone")
