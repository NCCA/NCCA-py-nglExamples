"""CPU reference for the Reynolds flocking rules used by BoidsCompute.

`step()` below is the ground truth: `BoidsCompute.wgsl` is a deliberate,
line-by-line transcription of this function onto the GPU (one thread per
boid, same three rules, same clamp, same wall force) - see the comment
block at the top of the compute shader that points back here. If you
change a rule, change it in both places.

Everything is plain numpy so the tests run headless with no wgpu/Qt
dependency - see `tests/test_boid_maths.py`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BoidParams:
    """Tunable knobs for one step() call - mirrors the SimParams uniform
    struct in BoidsCompute.wgsl field for field."""

    rs: float = 0.6  # separation radius: push away from boids closer than this
    ra: float = 1.2  # alignment radius: match heading of boids within this
    rc: float = 1.5  # cohesion radius: steer towards the centre of boids within this
    w_sep: float = 1.5  # separation weight
    w_align: float = 1.0  # alignment weight
    w_coh: float = 1.0  # cohesion weight
    v_min: float = 0.5  # minimum boid speed after clamping
    v_max: float = 2.0  # maximum boid speed after clamping
    dt: float = 0.1  # integration time step
    bounds: tuple[float, float, float] = (5.0, 5.0, 5.0)  # half-extents of the box
    wall_margin: float = 1.0  # distance from a wall at which avoidance starts
    w_wall: float = 3.0  # wall avoidance weight


def _wall_force(positions: np.ndarray, params: BoidParams) -> np.ndarray:
    """Soft push back towards the centre of the box once a boid gets within
    `wall_margin` of any face, growing linearly to full strength at the
    wall itself. Same idea as the bounce-with-damping walls in
    WebGPUCompute/SpatialHash3D, but a steering force rather than a bounce."""
    bounds = np.asarray(params.bounds, dtype=np.float64)
    margin = max(params.wall_margin, 1e-6)
    force = np.zeros_like(positions)
    for axis in range(3):
        half = bounds[axis]
        depth_hi = positions[:, axis] - (
            half - margin
        )  # >0 once inside margin at +face
        depth_lo = (-half + margin) - positions[
            :, axis
        ]  # >0 once inside margin at -face
        force[:, axis] -= np.clip(depth_hi, 0.0, margin) / margin
        force[:, axis] += np.clip(depth_lo, 0.0, margin) / margin
    return force


def step(
    positions: np.ndarray, velocities: np.ndarray, params: BoidParams
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every boid by one time step and return (new_positions,
    new_velocities). O(N^2): every boid looks at every other boid, exactly
    like the naive compute pass in BoidsCompute.wgsl - see that file's
    docstring for the spatial-hash optimisation this trades away for
    teaching clarity (WebGPUCompute/SpatialHash3D shows the alternative)."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    # pairwise displacement diff[i, j] = pos_i - pos_j, self-distance forced
    # to infinity so a boid never treats itself as a neighbour.
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    # --- separation: push away from anyone closer than rs, weighted by
    # direction only (not falloff) and averaged over however many are close ---
    sep_mask = dist < params.rs
    safe_dist = np.where(dist > 1e-6, dist, 1e-6)
    sep_dir = diff / safe_dist[..., None]
    sep_counts = sep_mask.sum(axis=1)
    sep_sum = np.where(sep_mask[..., None], sep_dir, 0.0).sum(axis=1)
    separation = np.divide(
        sep_sum,
        sep_counts[:, None],
        out=np.zeros_like(sep_sum),
        where=sep_counts[:, None] > 0,
    )

    # --- alignment: steer towards the average heading of boids within ra ---
    align_mask = dist < params.ra
    align_counts = align_mask.sum(axis=1)
    vel_sum = np.where(align_mask[..., None], velocities[None, :, :], 0.0).sum(axis=1)
    avg_vel = np.divide(
        vel_sum,
        align_counts[:, None],
        out=np.zeros_like(vel_sum),
        where=align_counts[:, None] > 0,
    )
    alignment = np.where(align_counts[:, None] > 0, avg_vel - velocities, 0.0)

    # --- cohesion: steer towards the centre of mass of boids within rc ---
    coh_mask = dist < params.rc
    coh_counts = coh_mask.sum(axis=1)
    pos_sum = np.where(coh_mask[..., None], positions[None, :, :], 0.0).sum(axis=1)
    avg_pos = np.divide(
        pos_sum,
        coh_counts[:, None],
        out=np.zeros_like(pos_sum),
        where=coh_counts[:, None] > 0,
    )
    cohesion = np.where(coh_counts[:, None] > 0, avg_pos - positions, 0.0)

    wall = _wall_force(positions, params)

    accel = (
        params.w_sep * separation
        + params.w_align * alignment
        + params.w_coh * cohesion
        + params.w_wall * wall
    )

    new_velocities = velocities + accel * params.dt

    # clamp speed to [v_min, v_max], preserving heading
    speed = np.linalg.norm(new_velocities, axis=1, keepdims=True)
    safe_speed = np.where(speed > 1e-6, speed, 1e-6)
    clamped_speed = np.clip(speed, params.v_min, params.v_max)
    new_velocities = new_velocities / safe_speed * clamped_speed

    new_positions = positions + new_velocities * params.dt

    return new_positions, new_velocities
