#!/usr/bin/env -S uv run --script
import sys

import numpy as np
from ncca.ngl import Obj


def create_np_array(obj: Obj) -> np.ndarray:
    """Return the OpenGL-compatible packed triangle data for an OBJ mesh."""
    return obj.triangle_vertex_data(flip_v=True)


def dump_numpy(file_path):
    obj = Obj()
    try:
        if not obj.load(file_path):
            print(f"Error: Invalid OBJ file {file_path}")

    except Exception as e:  # noqa: BLE001 - OBJ backends do not share one error type.
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
