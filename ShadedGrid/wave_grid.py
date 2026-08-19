"""Animated wave-height grid maths, shared by the OpenGL and WebGPU entry points.

Pure numpy, no GL/Qt/wgpu imports, so this is unit-testable headless and
shared unchanged between ``main.py`` and ``main_webgpu.py``.
"""

import numpy as np


def _wave_heights(x: np.ndarray, z: np.ndarray, offset: float) -> np.ndarray:
    y = np.sin(x + offset) + np.cos(x - offset)
    y += np.sin(z + offset) + np.cos(z - offset)
    return y * 0.5


def build_wave_grid(n: int, size: float, offset: float) -> np.ndarray:
    """Non-indexed triangle-list grid, interleaved x,y,z,nx,ny,nz,u,v float32.

    Normals use the standard heightfield formula (central differences of
    height, clamped at the boundary), not the NGL9Demos source's incomplete
    per-vertex neighbour-cross-product method (see plan design notes).
    """
    coords = np.linspace(-size / 2.0, size / 2.0, n, dtype=np.float64)
    xs, zs = np.meshgrid(coords, coords, indexing="xy")
    ys = _wave_heights(xs, zs, offset)

    step = size / (n - 1)
    x_prev = _wave_heights(np.roll(xs, 1, axis=1), zs, offset)
    x_next = _wave_heights(np.roll(xs, -1, axis=1), zs, offset)
    x_prev[:, 0] = ys[:, 0]
    x_next[:, -1] = ys[:, -1]
    dy_dx = (x_next - x_prev) / (2.0 * step)
    dy_dx[:, 0] = (ys[:, 1] - ys[:, 0]) / step
    dy_dx[:, -1] = (ys[:, -1] - ys[:, -2]) / step

    z_prev = _wave_heights(xs, np.roll(zs, 1, axis=0), offset)
    z_next = _wave_heights(xs, np.roll(zs, -1, axis=0), offset)
    z_prev[0, :] = ys[0, :]
    z_next[-1, :] = ys[-1, :]
    dy_dz = (z_next - z_prev) / (2.0 * step)
    dy_dz[0, :] = (ys[1, :] - ys[0, :]) / step
    dy_dz[-1, :] = (ys[-1, :] - ys[-2, :]) / step

    normals = np.stack([-dy_dx, np.ones_like(ys), -dy_dz], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    u = (xs - xs.min()) / size
    v = (zs - zs.min()) / size

    def vertex(i: int, j: int) -> tuple[float, ...]:
        return (
            xs[j, i],
            ys[j, i],
            zs[j, i],
            normals[j, i, 0],
            normals[j, i, 1],
            normals[j, i, 2],
            u[j, i],
            v[j, i],
        )

    tris: list[float] = []
    for j in range(n - 1):
        for i in range(n - 1):
            tris.extend(vertex(i, j + 1))
            tris.extend(vertex(i + 1, j))
            tris.extend(vertex(i, j))
            tris.extend(vertex(i, j + 1))
            tris.extend(vertex(i + 1, j + 1))
            tris.extend(vertex(i + 1, j))
    return np.array(tris, dtype=np.float32)
