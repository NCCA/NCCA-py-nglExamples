"""Shared blade geometry for the ResetLine OpenGL and WebGPU demos."""

from dataclasses import dataclass

import numpy as np

BASE_COLOUR = np.array([0.1, 0.2, 0.1], dtype=np.float32)
TIP_COLOUR = np.array([0.0, 1.0, 0.0], dtype=np.float32)
RESTART_INDEX = np.uint32(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class BladeField:
    """Packed blade vertices, OpenGL indices and per-blade ranges."""

    vertices: np.ndarray
    indices: np.ndarray
    ranges: np.ndarray


def create_blade_field(
    rows: int = 120,
    cols: int = 120,
    row_size: float = 20.0,
    col_size: float = 20.0,
    seed: int | None = None,
) -> BladeField:
    """Creates the irregular line field used by both renderers."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    rng = np.random.default_rng(seed)
    x_positions = np.linspace(
        -row_size * 0.5, row_size * 0.5, rows, endpoint=False, dtype=np.float32
    )
    z_positions = np.linspace(
        -col_size * 0.5, col_size * 0.5, cols, endpoint=False, dtype=np.float32
    )
    vertices: list[list[float]] = []
    indices: list[int] = []
    ranges: list[tuple[int, int]] = []

    for z in z_positions:
        for x in x_positions:
            start = len(vertices)
            segments = int(rng.integers(2, 13))
            height = 1.0 + float(rng.random()) * 1.5
            step = height / segments
            position = np.array([x, 0.0, z], dtype=np.float32)

            for point in range(segments + 1):
                if point:
                    position[1] += step
                    position[0] += float(rng.uniform(-0.1, 0.1))
                    position[2] += float(rng.uniform(-0.1, 0.1))
                t = point / (segments + 1)
                colour = BASE_COLOUR + (TIP_COLOUR - BASE_COLOUR) * t
                vertices.append([*position.tolist(), *colour.tolist()])
                indices.append(len(vertices) - 1)

            count = len(vertices) - start
            ranges.append((start, count))
            indices.append(int(RESTART_INDEX))

    return BladeField(
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        ranges=np.asarray(ranges, dtype=np.int32),
    )


def expand_line_list(vertices: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Expands separate strips into line-list pairs for WebGPU."""
    line_indices: list[int] = []
    for start, count in ranges:
        for index in range(int(start), int(start + count - 1)):
            line_indices.extend((index, index + 1))
    return np.ascontiguousarray(vertices[np.asarray(line_indices, dtype=np.int32)])


def animate_blades(
    vertices: np.ndarray,
    ranges: np.ndarray,
    phase: float,
    amplitude: float = 0.001,
) -> np.ndarray:
    """Moves each blade stem whilst leaving its root fixed."""
    moved = np.array(vertices, dtype=np.float32, copy=True)
    stem_mask = np.ones(len(moved), dtype=bool)
    stem_mask[ranges[:, 0]] = False
    heights = moved[stem_mask, 1]
    moved[stem_mask, 0] += np.sin(phase * heights) * amplitude
    moved[stem_mask, 2] += np.cos(phase * heights) * amplitude
    return moved
