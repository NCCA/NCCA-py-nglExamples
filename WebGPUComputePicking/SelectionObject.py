"""
SelectionObject: base class for pickable scene objects (WebGPU compute picking).

Mirrors the SelectionObject in ``SelectionManipulatorWebGPU`` but replaces
the colour ID with a plain integer ``pick_id``. There is no float->byte
encoding of IDs into colours and no 24-bit ceiling: the ID pass writes the
u32 straight into an ``r32uint`` attachment and the compute kernel returns
it as-is, so ``pick_id`` only needs a dictionary lookup to resolve the
clicked object.

Selection is shown by the object pipeline as a diffuse fill plus a white
wireframe overdraw (a single-pass barycentric effect in ObjectShader.wgsl).
Instead of issuing draw calls, each object hands the pipeline the data for
one storage-buffer instance via :meth:`instance`.
"""

import itertools
from abc import ABC

from ncca.ngl import Mat4, Transform, Vec3, Vec4

# IDs start at 1: the ID texture clears to 0 for "background"
_id_counter = itertools.count(1)


class SelectionObject(ABC):
    """A pickable scene object; ``mesh`` names the geometry in the pipeline."""

    mesh: str = ""  # overridden by subclasses

    def __init__(self, position: Vec3, colour: Vec4) -> None:
        self.position = position
        self.rotation = Vec3(0.0, 0.0, 0.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.colour = colour
        self.selected = False
        self.pick_id = next(_id_counter)

    def transform_matrix(self) -> Mat4:
        tx = Transform()
        tx.set_position(self.position.x, self.position.y, self.position.z)
        tx.set_rotation(self.rotation.x, self.rotation.y, self.rotation.z)
        tx.set_scale(self.scale.x, self.scale.y, self.scale.z)
        return tx.matrix()

    def instance(self, global_tx: Mat4) -> dict:
        """Build the storage-buffer instance for the object pipeline."""
        model = global_tx @ self.transform_matrix()
        normal_matrix = model.copy().inverse().transposed()
        return {
            "mesh": self.mesh,
            "model": model.to_numpy(),
            "normal_matrix": normal_matrix.to_numpy(),
            "colour": self.colour.to_list(),
            "pick_id": self.pick_id,
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
