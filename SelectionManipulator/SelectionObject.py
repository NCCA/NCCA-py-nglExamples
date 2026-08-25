"""
SelectionObject: base class for pickable, transformable scene objects.

Each object owns a position / rotation / scale, a display colour and a unique
colour ID used for picking (the scene is redrawn with each object in its ID
colour and the pixel under the mouse is read back to find what was clicked).

When an object is selected it is drawn with solid diffuse shading plus a
wireframe overdraw (polygon offset pushes the lines towards the camera so
they sit cleanly on top of the filled triangles).
"""

from abc import ABC, abstractmethod

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, Transform, Vec3, Vec4
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib

# colour used for the wireframe overlay on selected objects
WIREFRAME_COLOUR = Vec4(1.0, 1.0, 1.0, 1.0)


def _colour_id_generator():
    """Yield unique RGB tuples for picking, starting at 1 (0,0,0 is background)."""
    for i in range(1, 1 << 24):
        yield (i & 0xFF, (i >> 8) & 0xFF, (i >> 16) & 0xFF)


_colour_gen = _colour_id_generator()


class SelectionObject(ABC):
    """A pickable scene object; subclasses supply the geometry to draw."""

    def __init__(self, position: Vec3, colour: Vec4) -> None:
        self.position = position
        self.rotation = Vec3(0.0, 0.0, 0.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.colour = colour
        self.selected = False
        self.colour_id = next(_colour_gen)

    @abstractmethod
    def draw_geometry(self) -> None:
        """Issue the draw call for this object's geometry."""

    def transform_matrix(self) -> Mat4:
        tx = Transform()
        tx.set_position(self.position.x, self.position.y, self.position.z)
        tx.set_rotation(self.rotation.x, self.rotation.y, self.rotation.z)
        tx.set_scale(self.scale.x, self.scale.y, self.scale.z)
        return tx.matrix()

    def matches_colour_id(self, pixel: tuple) -> bool:
        return tuple(pixel) == self.colour_id

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

    def draw_pick(self, global_tx: Mat4, view: Mat4, project: Mat4) -> None:
        """Draw flat-shaded in this object's unique ID colour for picking."""
        model = global_tx @ self.transform_matrix()
        mvp = project @ view @ model
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform(
            "Colour",
            self.colour_id[0] / 255.0,
            self.colour_id[1] / 255.0,
            self.colour_id[2] / 255.0,
            1.0,
        )
        self.draw_geometry()


class TeapotObject(SelectionObject):
    def draw_geometry(self) -> None:
        Primitives.draw("teapot")


class CubeObject(SelectionObject):
    def draw_geometry(self) -> None:
        Primitives.draw("cube")


class SphereObject(SelectionObject):
    """Uses the parametric sphere created in main.py as 'selSphere'."""

    def draw_geometry(self) -> None:
        Primitives.draw("selSphere")


class TrollObject(SelectionObject):
    def draw_geometry(self) -> None:
        Primitives.draw("troll")


class DodecahedronObject(SelectionObject):
    def draw_geometry(self) -> None:
        Primitives.draw("dodecahedron")
