"""
SelectionObject: base class for pickable, transformable scene objects.

Unlike the colour-ID version in ``SelectionManipulator``, picking here is
done entirely on the CPU by ray casting: each object keeps its triangle data
(the same interleaved arrays used to build the GPU primitives) and answers
``intersect(origin, direction)`` with the hit distance along the ray, or
None. A bounding-sphere broad phase rejects most objects before the exact
per-triangle Moller-Trumbore test runs.

There is no ID render pass and no glReadPixels: nothing extra is drawn to
pick, and the hit distance falls out of the maths for free.

When an object is selected it is drawn with solid diffuse shading plus a
wireframe overdraw (polygon offset pushes the lines towards the camera so
they sit cleanly on top of the filled triangles).
"""

from abc import ABC, abstractmethod

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, PrimData, Prims, Transform, Vec3, Vec4
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib
from picking_maths import (
    bounding_sphere,
    intersect_sphere,
    intersect_triangles,
    transform_ray,
)

# colour used for the wireframe overlay on selected objects
WIREFRAME_COLOUR = Vec4(1.0, 1.0, 1.0, 1.0)

# CPU-side triangle data shared by every instance of a mesh:
# name -> (triangles (N,3,3), sphere centre, sphere radius)
_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}


def _register_mesh(name: str, interleaved: np.ndarray) -> None:
    """Cache the local-space triangles + bounding sphere of an interleaved
    (pos, normal, uv) primitive array."""
    positions = np.asarray(interleaved, dtype=np.float32).reshape(-1, 8)[:, :3]
    triangles = positions.reshape(-1, 3, 3)
    centre, radius = bounding_sphere(positions)
    _MESH_CACHE[name] = (triangles, centre, radius)


def load_pick_meshes() -> None:
    """Build the CPU triangle cache for every mesh used in the scene.

    Call once at startup (the GPU primitives are created separately with
    Primitives.load_default_primitives / Primitives.create).
    """
    for name, prim in {
        "teapot": Prims.TEAPOT,
        "cube": Prims.CUBE,
        "troll": Prims.TROLL,
        "dodecahedron": Prims.DODECAHEDRON,
    }.items():
        _register_mesh(name, PrimData.primitive(prim))
    _register_mesh("selSphere", PrimData.sphere(1.0, 40))


class SelectionObject(ABC):
    """A pickable scene object; subclasses name the mesh they draw."""

    mesh: str = ""  # overridden by subclasses

    def __init__(self, position: Vec3, colour: Vec4) -> None:
        self.position = position
        self.rotation = Vec3(0.0, 0.0, 0.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.colour = colour
        self.selected = False

    @abstractmethod
    def draw_geometry(self) -> None:
        """Issue the draw call for this object's geometry."""

    def transform_matrix(self) -> Mat4:
        tx = Transform()
        tx.set_position(self.position.x, self.position.y, self.position.z)
        tx.set_rotation(self.rotation.x, self.rotation.y, self.rotation.z)
        tx.set_scale(self.scale.x, self.scale.y, self.scale.z)
        return tx.matrix()

    # ------------------------------------------------------------------
    # picking
    # ------------------------------------------------------------------
    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        """Distance along the (scene-space) ray to this object, or None.

        The ray is transformed into the object's local space (one 4x4
        inverse) and tested there, so the cached triangles never need
        re-transforming. Because the direction keeps its transformed length,
        the local-space t parameter is also the scene-space hit distance and
        can be compared directly across objects to find the nearest.
        """
        triangles, centre, radius = _MESH_CACHE[self.mesh]
        inverse = np.linalg.inv(self.transform_matrix().to_numpy().astype(np.float64))
        local_origin, local_direction = transform_ray(origin, direction, inverse)
        # broad phase: cheap sphere test rejects most objects outright
        if not intersect_sphere(local_origin, local_direction, centre, radius):
            return None
        return intersect_triangles(local_origin, local_direction, triangles)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def draw(self, global_tx: Mat4, view: Mat4, project: Mat4) -> None:
        """Draw with diffuse shading; add a wireframe overdraw when selected."""
        model = global_tx @ self.transform_matrix()
        mv = view @ model
        mvp = project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        ShaderLib.set_uniform("Colour", self.colour)
        self.draw_geometry()

        if self.selected:
            ShaderLib.use(DefaultShader.COLOUR)
            ShaderLib.set_uniform("MVP", mvp)
            ShaderLib.set_uniform("Colour", WIREFRAME_COLOUR)
            # pull the wireframe towards the camera so it isn't z-fighting
            # with the filled surface underneath it
            gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
            gl.glPolygonOffset(-1.0, -1.0)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            self.draw_geometry()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)


class TeapotObject(SelectionObject):
    mesh = "teapot"

    def draw_geometry(self) -> None:
        Primitives.draw("teapot")


class CubeObject(SelectionObject):
    mesh = "cube"

    def draw_geometry(self) -> None:
        Primitives.draw("cube")


class SphereObject(SelectionObject):
    """Uses the parametric sphere created in main.py as 'selSphere'."""

    mesh = "selSphere"

    def draw_geometry(self) -> None:
        Primitives.draw("selSphere")


class TrollObject(SelectionObject):
    mesh = "troll"

    def draw_geometry(self) -> None:
        Primitives.draw("troll")


class DodecahedronObject(SelectionObject):
    mesh = "dodecahedron"

    def draw_geometry(self) -> None:
        Primitives.draw("dodecahedron")
