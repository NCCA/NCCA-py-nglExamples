"""
SelectionObject: base class for pickable, transformable scene objects (WebGPU).

Mirrors the OpenGL demo's SelectionObject. Each object owns a
position / rotation / scale, a display colour and a unique colour ID used for
picking (the scene is re-rendered with each object flat-shaded in its ID colour
and the pixel under the mouse is read back to find what was clicked).

Selection is shown by the object pipeline as a diffuse fill plus a white
wireframe overdraw (a single-pass barycentric effect in ObjectShader.wgsl).
Instead of issuing GL draw calls, each object hands the pipeline the data for
one storage-buffer instance via :meth:`instance`.
"""

from abc import ABC

from ncca.ngl import Mat4, Transform, Vec3, Vec4


def _colour_id_generator():
    """Yield unique RGB tuples for picking, starting at 1 (0,0,0 is background)."""
    for i in range(1, 1 << 24):
        yield (i & 0xFF, (i >> 8) & 0xFF, (i >> 16) & 0xFF)


_colour_gen = _colour_id_generator()


class SelectionObject(ABC):
    """A pickable scene object; ``mesh`` names the geometry in the pipeline."""

    mesh: str = ""  # overridden by subclasses

    def __init__(self, position: Vec3, colour: Vec4) -> None:
        self.position = position
        self.rotation = Vec3(0.0, 0.0, 0.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.colour = colour
        self.selected = False
        self.colour_id = next(_colour_gen)

    def transform_matrix(self) -> Mat4:
        tx = Transform()
        tx.set_position(self.position.x, self.position.y, self.position.z)
        tx.set_rotation(self.rotation.x, self.rotation.y, self.rotation.z)
        tx.set_scale(self.scale.x, self.scale.y, self.scale.z)
        return tx.matrix()

    def matches_colour_id(self, pixel: tuple[int, int, int]) -> bool:
        return tuple(pixel) == self.colour_id

    def instance(self, global_tx: Mat4) -> dict:
        """Build the storage-buffer instance for the object pipeline."""
        model = global_tx @ self.transform_matrix()
        normal_matrix = model.copy().inverse().transposed()
        r, g, b = self.colour_id
        return {
            "mesh": self.mesh,
            "model": model.to_numpy(),
            "normal_matrix": normal_matrix.to_numpy(),
            "colour": self.colour.to_list(),
            "pick_colour": (r / 255.0, g / 255.0, b / 255.0, 1.0),
            "selected": self.selected,
        }


class TeapotObject(SelectionObject):
    mesh = "teapot"


class CubeObject(SelectionObject):
    mesh = "cube"


class SphereObject(SelectionObject):
    mesh = "selSphere"


class TrollObject(SelectionObject):
    mesh = "troll"


class DodecahedronObject(SelectionObject):
    mesh = "dodecahedron"
