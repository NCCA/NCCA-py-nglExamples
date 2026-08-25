#!/usr/bin/env -S uv run --script
import sys
from dataclasses import dataclass

import numpy as np
from ncca.ngl import Obj


def create_np_array(obj) -> None:
    if not obj.is_triangular():
        print("can only process triangular meshes")
        return

    @dataclass
    class VertData:
        """
        Structure for a single vertex's data, including position, normal, and UV.
        """

        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        nx: float = 0.0
        ny: float = 0.0
        nz: float = 0.0
        u: float = 0.0
        v: float = 0.0

        def as_array(self) -> np.ndarray:
            return np.array(
                [self.x, self.y, self.z, self.nx, self.ny, self.nz, self.u, self.v],
                dtype=np.float32,
            )

    mesh: list[VertData] = []
    for face in obj.faces:
        for i in range(3):
            d = VertData()
            d.x = obj.vertex[face.vertex[i]].x
            d.y = obj.vertex[face.vertex[i]].y
            d.z = obj.vertex[face.vertex[i]].z
            if obj.normals and obj.uv:
                d.nx = obj.normals[face.normal[i]].x
                d.ny = obj.normals[face.normal[i]].y
                d.nz = obj.normals[face.normal[i]].z
                d.u = obj.uv[face.uv[i]].x
                d.v = 1 - obj.uv[face.uv[i]].y  # Flip V for OpenGL
            elif obj.normals and not obj.uv:
                d.nx = obj.normals[face.normal[i]].x
                d.ny = obj.normals[face.normal[i]].y
                d.nz = obj.normals[face.normal[i]].z
            elif not obj.normals and obj.uv:
                d.u = obj.uv[face.uv[i]].x
                d.v = 1 - obj.uv[face.uv[i]].y
            mesh.append(d)

    mesh_data = np.concatenate([v.as_array() for v in mesh]).astype(np.float32)
    return mesh_data


def dump_numpy(file_path):
    obj = Obj()
    try:
        if not obj.load(file_path):
            print(f"Error: Invalid OBJ file {file_path}")

    except Exception as e:
        print(f"Error: {file_path} {e}")
    mesh_data = create_np_array(obj)
    print(mesh_data.shape, mesh_data.dtype, mesh_data.nbytes)
    np.save(file_path[: len(file_path) - 4] + ".npy", mesh_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Obj2Numpy.py <input_file>.obj ")
        print("you can pass multiple files")
    for i in range(1, len(sys.argv)):
        print(f"Processing file {sys.argv[i]}")
        dump_numpy(sys.argv[i])
    sys.exit(1)
