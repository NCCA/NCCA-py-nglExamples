"""Renderer for MeshLab OBJ files with baked vertex colours."""

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Obj
from ncca.ngl.opengl import AbstractVAO, VAOFactory, VAOType
from ncca.ngl.opengl.abstract_vao import VertexData


class ColourObj:
    """Draw a parser-only Obj using the ColourObj eleven-float layout."""

    def __init__(self, data: Obj) -> None:
        """Store the CPU-side OBJ data and defer GPU upload."""
        self.data = data
        self._vao: AbstractVAO | None = None

    @classmethod
    def from_file(cls, path: str) -> "ColourObj":
        """Load the OBJ data used by this renderer."""
        return cls(Obj.from_file(path))

    def create_colour_vao(self, reset_vao: bool = False) -> None:
        """Upload the custom position, normal, UV and colour vertex layout."""
        if self._vao is not None:
            if reset_vao:
                return
            self._vao.remove_vao()
        self.data.validate()
        if not self.data.is_triangular():
            raise RuntimeError("ColourObj requires triangular mesh data")
        rows: list[float] = []
        for face in self.data.faces:
            for corner, vertex_index in enumerate(face.vertex):
                vertex = self.data.vertex[vertex_index]
                normal = (
                    self.data.normals[face.normal[corner]]
                    if face.normal
                    else (0.0, 0.0, 0.0)
                )
                uv = self.data.uv[face.uv[corner]] if face.uv else (0.0, 0.0)
                colour = self.data.colour[vertex_index]
                rows.extend(
                    (
                        vertex.x,
                        vertex.y,
                        vertex.z,
                        normal.x if face.normal else normal[0],
                        normal.y if face.normal else normal[1],
                        normal.z if face.normal else normal[2],
                        uv.x if face.uv else uv[0],
                        1.0 - uv.y if face.uv else uv[1],
                        colour.x,
                        colour.y,
                        colour.z,
                    )
                )
        mesh_data = np.asarray(rows, dtype=np.float32)
        self._vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self._vao as vao:
            vertex_count = mesh_data.size // 11
            vao.set_data(VertexData(mesh_data, vertex_count))
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 44, 0)
            vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 44, 12)
            vao.set_vertex_attribute_pointer(2, 2, gl.GL_FLOAT, 44, 24)
            vao.set_vertex_attribute_pointer(3, 3, gl.GL_FLOAT, 44, 32)
            vao.set_num_indices(vertex_count)
        self.data.calc_dimensions()

    def draw(self) -> None:
        """Draw the uploaded colour mesh."""
        if self._vao is None:
            raise RuntimeError("ColourObj must be uploaded before drawing")
        with self._vao as vao:
            vao.draw()

    def cleanup(self) -> None:
        """Release the colour mesh VAO."""
        if self._vao is not None:
            self._vao.remove_vao()
            self._vao = None
