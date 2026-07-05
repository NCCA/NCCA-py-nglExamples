"""Loads the MeshLab-style 6-float-per-vertex OBJ variant used by NGL9Demos/ColourObj,
where each `v` line is `x y z r g b` (position + baked vertex colour).

Note: `ncca.ngl.Obj._parse_vertex` already recognises this 7-token `v` line
(`v` + 6 floats) and stores the trailing colour triplet in `self.colour`, so
no parsing override is needed here. `ColourObj` only adds a VAO builder that
interleaves position/normal/uv/colour into a single buffer for the
`ColourVertex.glsl`/`ColourFragment.glsl` shader pair (locations 0-3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import BBox, Obj
from ncca.ngl.opengl import VAOFactory, VAOType
from ncca.ngl.opengl.abstract_vao import VertexData


class ColourObj(Obj):
    @classmethod
    def from_file(cls, path: str) -> "ColourObj":
        obj = cls()
        obj.load(path)
        return obj

    def create_colour_vao(self, reset_vao: bool = False) -> None:
        if self._should_skip_vao_creation(reset_vao):
            return
        self._validate_triangular_mesh()

        colours = getattr(self, "colour", [])

        @dataclass
        class VertData:
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0
            nx: float = 0.0
            ny: float = 0.0
            nz: float = 0.0
            u: float = 0.0
            v: float = 0.0
            r: float = 1.0
            g: float = 1.0
            b: float = 1.0

            def as_array(self) -> np.ndarray:
                return np.array(
                    [
                        self.x,
                        self.y,
                        self.z,
                        self.nx,
                        self.ny,
                        self.nz,
                        self.u,
                        self.v,
                        self.r,
                        self.g,
                        self.b,
                    ],
                    dtype=np.float32,
                )

        vbo_mesh: list[VertData] = []
        for face in self.faces:
            for i in range(3):
                d = VertData()
                idx = face.vertex[i]
                d.x = self.vertex[idx].x
                d.y = self.vertex[idx].y
                d.z = self.vertex[idx].z
                if self.normals and face.normal:
                    n = self.normals[face.normal[i]]
                    d.nx, d.ny, d.nz = n.x, n.y, n.z
                if self.uv and face.uv:
                    uv = self.uv[face.uv[i]]
                    d.u = uv.x
                    d.v = 1 - uv.y
                if colours:
                    c = colours[idx]
                    d.r, d.g, d.b = c.x, c.y, c.z
                vbo_mesh.append(d)

        mesh_data = np.concatenate([v.as_array() for v in vbo_mesh]).astype(np.float32)
        self.vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_TRIANGLES)
        with self.vao as vao:
            mesh_size = len(mesh_data) // 11
            vao.set_data(VertexData(mesh_data, mesh_size))
            # vertex
            vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 11 * 4, 0)
            # normal
            vao.set_vertex_attribute_pointer(1, 3, gl.GL_FLOAT, 11 * 4, 3 * 4)
            # uv
            vao.set_vertex_attribute_pointer(2, 2, gl.GL_FLOAT, 11 * 4, 6 * 4)
            # colour
            vao.set_vertex_attribute_pointer(3, 3, gl.GL_FLOAT, 11 * 4, 8 * 4)
            vao.set_num_indices(mesh_size)
        self.calc_dimensions()
        self.bbox = BBox.from_extents(
            self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z
        )
