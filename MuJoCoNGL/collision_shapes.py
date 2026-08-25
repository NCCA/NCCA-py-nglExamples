"""The catalogue of shapes the demo can drop into the world.

This is the MuJoCo equivalent of the C++ demo's `CollisionShape` singleton. In
Bullet you build a `btCollisionShape` once and hand the same pointer to every
rigid body that wants it; in MuJoCo the equivalent is a geom description that
gets written into an `MjSpec` and compiled. So rather than holding live objects,
this module holds the recipe for each shape and knows how to write it into a
spec on demand.

Everything is Y-up to match NGL. MuJoCo has no opinion about which way is up --
there is only `option.gravity` -- so pointing gravity down -Y and rotating the
ground plane is the whole of the conversion. The primitives that have an axis
(capsule, cylinder, cone) are Z-aligned in both MuJoCo and NGL, so they line up
without any help.
"""

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
from ncca.ngl import Obj, Vec4

# Where the drop-in shapes are spawned from and how hard they are thrown about.
SPAWN_HEIGHT = 10.0

# The C++ demo gave everything a mass of 2, which makes for a lively pile. Left
# to itself MuJoCo would derive mass from a default density of 1000 kg/m^3, so a
# 1m cube would come out at a tonne and the arrow-key impulses would do nothing.
DEFAULT_MASS = 2.0


@dataclass(frozen=True)
class Shape:
    """A shape the world can spawn and the scene can draw.

    Attributes
    ----------
        name : str
            key used by both the physics world and the scene
        colour : Vec4
            colour the scene draws it in, matching the C++ demo
        add_geom : Callable
            writes this shape's geom onto an MjSpec body
        mesh : str | None
            name of the MjSpec mesh this shape needs, if any
        obj_path : str | None
            low-res OBJ collided against, loaded through PyNGL's Obj
        draw_name : str | None
            NGL primitive or mesh to draw, when it is not the shape's own name
    """

    name: str
    colour: Vec4
    add_geom: Callable[["mujoco.MjsBody"], "mujoco.MjsGeom"]
    mesh: str | None = None
    obj_path: str | None = None
    draw_name: str | None = None

    @property
    def drawn_as(self) -> str:
        """What the scene should draw for this shape."""
        return self.draw_name or self.name


def _primitive(geom_type: mujoco.mjtGeom, size: list[float]) -> Callable:
    def add(body: "mujoco.MjsBody") -> "mujoco.MjsGeom":
        geom = body.add_geom()
        geom.type = geom_type
        geom.size = size
        geom.mass = DEFAULT_MASS
        return geom

    return add


def _mesh_geom(mesh_name: str) -> Callable:
    def add(body: "mujoco.MjsBody") -> "mujoco.MjsGeom":
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
        geom.mass = DEFAULT_MASS
        return geom

    return add


def cone_hull(radius: float, height: float, slices: int = 32) -> tuple:
    """Builds the vertices and faces of a cone for MuJoCo to collide against.

    MuJoCo has no cone geom -- the primitive list stops at plane, sphere,
    capsule, ellipsoid, cylinder, box and mesh -- so the cone has to go in as a
    mesh. A cone is convex, which is exactly what MuJoCo wants, and this matches
    NGL's `Prims.CONE`: Z-aligned, base ring at z=0, apex at z=height.

    Parameters
    ----------
        radius : float
            radius of the base ring
        height : float
            distance from base to apex
        slices : int
            number of segments around the base

    Returns
    -------
        tuple of (vertices, faces) as flat arrays ready for MjSpec
    """
    angles = np.linspace(0.0, 2.0 * np.pi, slices, endpoint=False)
    ring = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(slices)], axis=1
    )
    verts = np.vstack([ring, [0.0, 0.0, height], [0.0, 0.0, 0.0]])
    apex, centre = slices, slices + 1

    faces = []
    for i in range(slices):
        nxt = (i + 1) % slices
        faces.append([i, nxt, apex])  # side
        faces.append([nxt, i, centre])  # base cap, wound the other way
    return verts.flatten(), np.array(faces, dtype=np.int32).flatten()


def load_obj_hull(path: str) -> tuple:
    """Loads an OBJ with PyNGL and returns its vertices and faces for MuJoCo.

    This is the direct descendant of the C++ demo's loop that walked
    `ngl::Obj::getVertexList()` calling `btConvexHullShape::addPoint` on each
    vertex. Same idea here: PyNGL reads the file, MuJoCo takes the points and
    computes the convex hull itself at compile time.

    Parameters
    ----------
        path : str
            path to the OBJ file, normally a low-res collision mesh

    Returns
    -------
        tuple of (vertices, faces) as flat arrays ready for MjSpec
    """
    mesh = Obj.from_file(path)
    verts = np.array([[v.x, v.y, v.z] for v in mesh.vertex], dtype=np.float64)
    # Only triangles -- the collision meshes are triangulated and MuJoCo builds
    # a hull from the points anyway, so anything beyond the first three indices
    # of a face would not change the result.
    faces = np.array([f.vertex[:3] for f in mesh.faces], dtype=np.int32)
    return verts.flatten(), faces.flatten()


@dataclass
class ShapeCatalogue:
    """The shapes available to spawn, and the meshes they need declaring.

    Attributes
    ----------
        shapes : dict
            shape name to Shape, in the order the number keys 1-7 use them
    """

    shapes: dict[str, Shape] = field(default_factory=dict)

    @classmethod
    def default(cls, model_dir: str = "models") -> "ShapeCatalogue":
        """Builds the same seven shapes the C++ demo offered.

        Parameters
        ----------
            model_dir : str
                directory holding the teapot and apple OBJ files
        """
        shapes = {
            "box": Shape(
                "box",
                Vec4(1.0, 0.0, 0.0, 1.0),
                _primitive(mujoco.mjtGeom.mjGEOM_BOX, [0.5, 0.5, 0.5]),
                # NGL's stock unit cube is exactly the box's 0.5 half-extents.
                draw_name="cube",
            ),
            "sphere": Shape(
                "sphere",
                Vec4(0.0, 1.0, 0.0, 1.0),
                _primitive(mujoco.mjtGeom.mjGEOM_SPHERE, [0.5, 0.0, 0.0]),
            ),
            # MuJoCo sizes a capsule as (radius, half-length of the cylinder
            # section), so this is a 0.5 radius capsule with a 1.0 long middle.
            "capsule": Shape(
                "capsule",
                Vec4(0.0, 0.0, 1.0, 1.0),
                _primitive(mujoco.mjtGeom.mjGEOM_CAPSULE, [0.5, 0.5, 0.0]),
            ),
            # Cylinders are (radius, half-height), so 1.0 here is 2.0 tall.
            "cylinder": Shape(
                "cylinder",
                Vec4(1.0, 1.0, 0.0, 1.0),
                _primitive(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.5, 1.0, 0.0]),
            ),
            "cone": Shape(
                "cone",
                Vec4(0.0, 1.0, 1.0, 1.0),
                _mesh_geom("cone"),
                mesh="cone",
            ),
            "teapot": Shape(
                "teapot",
                Vec4(1.0, 1.0, 0.0, 1.0),
                _mesh_geom("teapot"),
                mesh="teapot",
                obj_path=f"{model_dir}/teapotCollisionMesh.obj",
            ),
            "apple": Shape(
                "apple",
                Vec4(0.0, 1.0, 0.0, 1.0),
                _mesh_geom("apple"),
                mesh="apple",
                obj_path=f"{model_dir}/appleCollisionMesh.obj",
            ),
        }
        return cls(shapes)

    @property
    def names(self) -> list[str]:
        return list(self.shapes)

    def declare_meshes(self, spec: "mujoco.MjSpec") -> None:
        """Adds every mesh the catalogue needs to a spec before compiling.

        Meshes have to exist on the spec before any geom refers to them by name,
        so this is called once when a world is built.
        """
        for shape in self.shapes.values():
            if shape.mesh is None:
                continue
            mesh = spec.add_mesh()
            mesh.name = shape.mesh
            if shape.obj_path is not None:
                verts, faces = load_obj_hull(shape.obj_path)
            else:
                verts, faces = cone_hull(0.5, 2.0)
            mesh.uservert = verts
            mesh.userface = faces
            # Cap the hull so the narrow-phase stays cheap; the teapot's low-res
            # mesh is still 2271 points before hulling.
            mesh.maxhullvert = 64
