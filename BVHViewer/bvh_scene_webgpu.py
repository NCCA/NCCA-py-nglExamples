"""WebGPU draw data for the BVH skeleton, ground and motion traces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from bvh import Bvh, Joint, rotation_from_y
from bvh_scene import BvhScene
from ncca.ngl import Mat4, Vec3

INSTANCE_DTYPE = np.dtype(
    [
        ("model", np.float32, (4, 4)),
        ("normal_matrix", np.float32, (4, 4)),
        ("colour", np.float32, (4,)),
    ]
)

LINE_VERTEX_DTYPE = np.dtype(
    [
        ("position", np.float32, (3,)),
        ("colour", np.float32, (4,)),
    ]
)


def _bone_model(parent_world: Mat4, offset: Vec3) -> Mat4 | None:
    length = float(offset.length())
    if length < 1e-6:
        return None
    rotate = rotation_from_y(offset.normalized())
    midpoint = Mat4.translate(offset.x * 0.5, offset.y * 0.5, offset.z * 0.5)
    return parent_world @ midpoint @ rotate @ Mat4.scale(1.0, length, 1.0)


def build_skeleton_instances(
    characters: Sequence[Bvh],
) -> tuple[np.ndarray, int, int]:
    """Pack joint and bone transforms for two instanced mesh draws."""
    joint_models: list[Mat4] = []
    bone_models: list[Mat4] = []

    def visit(character: Bvh, joint: Joint, parent_world: Mat4) -> None:
        world = parent_world @ character.local_matrix(joint)
        joint_models.append(world)
        for child in joint.children:
            bone = _bone_model(world, child.offset)
            if bone is not None:
                bone_models.append(bone)
            visit(character, child, world)

    for character in characters:
        visit(character, character.root, Mat4())

    models = [*joint_models, *bone_models]
    instances = np.zeros(len(models), dtype=INSTANCE_DTYPE)
    for index, model in enumerate(models):
        instances[index]["model"] = model.to_numpy()
        instances[index]["normal_matrix"] = model.inverse().transposed().to_numpy()
        instances[index]["colour"] = (1.0, 0.82, 0.12, 1.0)
    return instances, len(joint_models), len(bone_models)


def build_ground_line_vertices(size: float = 200.0, divisions: int = 40) -> np.ndarray:
    """Build an XZ grid as a WebGPU line list."""
    coordinates = np.linspace(-size * 0.5, size * 0.5, divisions + 1, dtype=np.float32)
    vertices = np.zeros((divisions + 1) * 4, dtype=LINE_VERTEX_DTYPE)
    cursor = 0
    for coordinate in coordinates:
        vertices[cursor : cursor + 4]["position"] = (
            (coordinate, 0.0, -size * 0.5),
            (coordinate, 0.0, size * 0.5),
            (-size * 0.5, 0.0, coordinate),
            (size * 0.5, 0.0, coordinate),
        )
        cursor += 4
    vertices["colour"] = (0.6, 0.6, 0.6, 1.0)
    return vertices


def build_trace_line_vertices(
    positions: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    colours: np.ndarray,
) -> np.ndarray:
    """Expand trace strips into the independent segments WebGPU draws."""
    segment_count = sum(max(count - 1, 0) for _, count in ranges)
    vertices = np.zeros(segment_count * 2, dtype=LINE_VERTEX_DTYPE)
    cursor = 0
    for (first, count), colour in zip(ranges, colours, strict=True):
        for index in range(first, first + count - 1):
            vertices[cursor : cursor + 2]["position"] = positions[index : index + 2]
            vertices[cursor : cursor + 2]["colour"] = colour
            cursor += 2
    return vertices


class BvhWebGPUScene(BvhScene):
    """Keep BVH playback state and feed its draw data to a WebGPU renderer."""

    def __init__(self) -> None:
        super().__init__()
        self.renderer = None

    def initialise_webgpu(self, device, shader_path) -> None:
        from webgpu_renderer import BvhWebGPURenderer

        self.renderer = BvhWebGPURenderer(device, shader_path)
        self._trace_data_dirty = True

    def prepare_frame(self) -> None:
        if self.renderer is None:
            return
        instances, joint_count, bone_count = build_skeleton_instances(self.characters)
        self.renderer.update_instances(instances, joint_count, bone_count)
        if self._trace_data_dirty:
            ground = build_ground_line_vertices()
            traces = build_trace_line_vertices(
                self._trace_vertices,
                self._trace_ranges,
                self._trace_colours,
            )
            self.renderer.update_lines(ground, traces)
            self._trace_data_dirty = False

    def draw_webgpu(
        self,
        render_pass,
        view: Mat4,
        project: Mat4,
        model: Mat4,
        camera_index: int,
        trace: bool = False,
    ) -> None:
        if self.renderer is None:
            return
        self.renderer.update_camera(camera_index, project @ view @ model)
        self.renderer.render(render_pass, camera_index, trace)
