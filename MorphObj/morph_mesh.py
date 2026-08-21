"""OBJ pose packing and weight helpers shared by the morph demos."""

from pathlib import Path

import numpy as np
from ncca.ngl import Obj


def load_morph_mesh(
    base_path: str | Path,
    pose_one_path: str | Path,
    pose_two_path: str | Path,
) -> np.ndarray:
    """Loads three matching OBJ poses into the source demo's 18-float layout."""
    meshes = [
        Obj.from_file(str(path)) for path in (base_path, pose_one_path, pose_two_path)
    ]
    base_counts = (len(meshes[0].vertex), len(meshes[0].normals))
    if not meshes[0].faces or any(
        (len(mesh.vertex), len(mesh.normals)) != base_counts for mesh in meshes[1:]
    ):
        raise ValueError("morph poses must have matching vertex and normal counts")
    if not meshes[0].is_triangular():
        raise ValueError("base morph pose must have triangular topology")

    packed: list[list[float]] = []
    base, pose_one, pose_two = meshes
    for face in base.faces:
        for corner in range(3):
            vertex_index = face.vertex[corner]
            normal_index = face.normal[corner]
            base_position = base.vertex[vertex_index].to_numpy()
            base_normal = base.normals[normal_index].to_numpy()
            pose_one_position = pose_one.vertex[vertex_index].to_numpy()
            pose_one_normal = pose_one.normals[normal_index].to_numpy()
            pose_two_position = pose_two.vertex[vertex_index].to_numpy()
            pose_two_normal = pose_two.normals[normal_index].to_numpy()
            packed.append(
                np.concatenate(
                    (
                        base_position,
                        base_normal,
                        pose_one_position - base_position,
                        pose_one_normal - base_normal,
                        pose_two_position - base_position,
                        pose_two_normal - base_normal,
                    )
                ).tolist()
            )
    return np.asarray(packed, dtype=np.float32)


def adjust_weight(weight: float, delta: float) -> float:
    """Changes one morph weight and clamps it to zero to one."""
    return max(0.0, min(1.0, weight + delta))


def advance_punch(
    weight: float,
    direction: int,
    step: float = 0.2,
) -> tuple[float, int, bool]:
    """Advances one out-and-back punch animation."""
    weight += step * direction
    if weight >= 1.0:
        return 1.0, -1, True
    if weight <= 0.0:
        return 0.0, 1, False
    return weight, direction, True
