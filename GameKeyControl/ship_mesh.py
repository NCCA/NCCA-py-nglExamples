"""Interleaved ship vertex data for the WebGPU backend.

`ncca.ngl.Obj.create_vao()` builds the same `[x, y, z, nx, ny, nz, u, v]`
interleave internally, but only exposes it via a GL-specific VAO -- there is
no numpy-only accessor. `load_ship_vertex_data()` replicates
`BaseMesh.create_vao()`'s triangulation/interleave loop by hand (same
V-flip-for-consistency the library applies) without touching any GL call, so
the WebGPU entry point can build a vertex buffer straight from a plain numpy
array.
"""

from pathlib import Path

import numpy as np
from ncca.ngl import Obj, Vec3


def load_ship_vertex_data(path: Path) -> tuple[np.ndarray, int]:
    """Load an OBJ mesh and return its interleaved vertex data.

    Args:
        path: Path to a triangulated `.obj` file.

    Returns:
        A tuple of `(interleaved_float32_array, vertex_count)`, where the
        array is laid out as `[x, y, z, nx, ny, nz, u, v]` per vertex (so
        its length is `vertex_count * 8`).

    Raises:
        RuntimeError: If the mesh is not triangulated.
    """
    obj = Obj()
    obj.load(str(path))
    if not obj.is_triangular():
        raise RuntimeError(f"{path} is not a triangulated mesh")

    rows = []
    for face in obj.faces:
        for i in range(3):
            v = obj.vertex[face.vertex[i]]
            if obj.normals:
                n = obj.normals[face.normal[i]]
            else:
                n = Vec3(0.0, 0.0, 0.0)
            if obj.uv:
                uv = obj.uv[face.uv[i]]
                u, vv = uv.x, 1.0 - uv.y
            else:
                u, vv = 0.0, 0.0
            rows.append([v.x, v.y, v.z, n.x, n.y, n.z, u, vv])

    data = np.array(rows, dtype=np.float32).flatten()
    return data, len(rows)
