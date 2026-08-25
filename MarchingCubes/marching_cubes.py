"""Metaballs -> scalar field -> triangle mesh, vectorised over numpy.

This module is deliberately numpy-only (no GL/Qt) so the algorithm can be
unit tested headless -- see ``tests/test_marching_cubes.py``. It has two
entry points:

``sample_metaballs``
    Evaluates a classic Blinn-style metaball field (each ball contributes
    ``radius**2 / distance**2``, so a lone ball's iso == 1.0 surface sits
    at exactly ``distance == radius``) on a regular grid.

``polygonise``
    Runs Marching Cubes over that grid and returns a triangle soup. The
    lookup tables (``mc_tables.py``) turn "which corners of this cell are
    above the isovalue" into "which of the cell's 12 edges the surface
    crosses, and how those crossings join into triangles" -- see that
    module's docstring for numbering and provenance.

The implementation classifies every cell in the grid at once (rather than
looping cell-by-cell in Python) because a 48^3 grid is ~108k cells and
this needs to run once per frame; see MarchingCubes/README.md for the
measured cost.
"""

import numpy as np
from mc_tables import CORNER_OFFSETS, EDGE_VERTEX_INDICES, TRI_TABLE

# distance-squared floor so a metaball centre doesn't divide by zero
_EPS = 1e-8

# TRI_TABLE as a numpy array for fancy indexing: TRI_TABLE_NP[cubeindex]
# gives the 16-entry (edge, edge, edge, -1-padded) row for that cell.
_TRI_TABLE_NP = np.asarray(TRI_TABLE, dtype=np.int32)


def sample_metaballs(
    centres: np.ndarray, radii: np.ndarray, grid_n: int, extent: float
) -> np.ndarray:
    """Sample a metaball field on a `grid_n`^3 grid spanning [-extent, extent]^3.

    Parameters
    ----------
        centres : np.ndarray
            (M, 3) array of metaball centre positions.
        radii : np.ndarray
            (M,) array of metaball radii.
        grid_n : int
            number of samples along each axis.
        extent : float
            half-width of the sampled cube -- the grid covers
            [-extent, extent] on x, y and z.

    Returns
    -------
        np.ndarray
            (grid_n, grid_n, grid_n) float32 field, indexed [x, y, z]. Each
            metaball contributes ``radius**2 / (distance**2 + eps)``, so with
            `radii` all equal a lone ball's isosurface at iso=1.0 is the sphere
            of that radius (matched by `polygonise`'s vertex-distance test).
    """
    centres = np.asarray(centres, dtype=np.float32).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)

    coords = np.linspace(-extent, extent, grid_n, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")

    field = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
    for centre, radius in zip(centres, radii):
        dist_sq = (gx - centre[0]) ** 2 + (gy - centre[1]) ** 2 + (gz - centre[2]) ** 2
        field += (radius * radius) / (dist_sq + _EPS)
    return field.astype(np.float32)


def polygonise(
    field: np.ndarray, iso: float, extent: float
) -> tuple[np.ndarray, np.ndarray]:
    """Polygonise a scalar field into a triangle soup via Marching Cubes.

    Parameters
    ----------
        field : np.ndarray
            (n, n, n) float32 scalar field, as produced by
            `sample_metaballs` (same [-extent, extent]^3 grid convention).
        iso : float
            isovalue -- the surface is where `field == iso`.
        extent : float
            half-width of the field's sample domain, matching the
            `extent` originally passed to `sample_metaballs`.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            (verts, normals): both (3*T, 3) float32 arrays for T triangles,
            vertices grouped in consecutive triples per triangle. Normals come
            from the field gradient (`np.gradient`), trilinearly interpolated
            at each vertex the same way the vertex position itself is -- this
            gives smooth (Phong-style) shading rather than flat per-face
            normals.
    """
    n = field.shape[0]
    coords = np.linspace(-extent, extent, n, dtype=np.float32)

    grad_x, grad_y, grad_z = np.gradient(field, coords, coords, coords)
    # outward normal points from high field (blob interior) to low field
    grad = -np.stack([grad_x, grad_y, grad_z], axis=-1)  # (n, n, n, 3)

    m = n - 1  # cells per axis
    if m <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    # --- per-corner values, positions and gradients for every cell at once ---
    corner_val = np.empty((8, m, m, m), dtype=np.float32)
    corner_pos = np.empty((8, m, m, m, 3), dtype=np.float32)
    corner_grad = np.empty((8, m, m, m, 3), dtype=np.float32)
    for c, (dx, dy, dz) in enumerate(CORNER_OFFSETS):
        sl = (slice(dx, dx + m), slice(dy, dy + m), slice(dz, dz + m))
        corner_val[c] = field[sl]
        corner_grad[c] = grad[sl]
        # world-space position of this corner for every cell
        px = coords[dx : dx + m][:, None, None]
        py = coords[dy : dy + m][None, :, None]
        pz = coords[dz : dz + m][None, None, :]
        corner_pos[c, ..., 0] = px
        corner_pos[c, ..., 1] = py
        corner_pos[c, ..., 2] = pz

    cubeindex = np.zeros((m, m, m), dtype=np.int32)
    for c in range(8):
        cubeindex |= (corner_val[c] > iso).astype(np.int32) << c

    # --- interpolate every one of the cell's 12 edges, unconditionally ---
    edge_pos = np.empty((12, m, m, m, 3), dtype=np.float32)
    edge_norm = np.empty((12, m, m, m, 3), dtype=np.float32)
    for e, (ca, cb) in enumerate(EDGE_VERTEX_INDICES):
        va, vb = corner_val[ca], corner_val[cb]
        denom = vb - va
        # cells where this edge isn't actually cut get a bogus t, but their
        # contribution is discarded below via the -1 sentinels in TRI_TABLE
        safe_denom = np.where(np.abs(denom) < _EPS, _EPS, denom)
        t = (iso - va) / safe_denom
        t = np.clip(t, 0.0, 1.0)[..., None]
        edge_pos[e] = corner_pos[ca] + t * (corner_pos[cb] - corner_pos[ca])
        edge_norm[e] = corner_grad[ca] + t * (corner_grad[cb] - corner_grad[ca])

    # --- gather triangles from the per-cell case index ---
    cubeindex_flat = cubeindex.reshape(-1)
    tri_rows = _TRI_TABLE_NP[cubeindex_flat]  # (M, 16)
    edge_pos_flat = edge_pos.reshape(12, -1, 3)  # (12, M, 3)
    edge_norm_flat = edge_norm.reshape(12, -1, 3)

    num_cells = cubeindex_flat.shape[0]
    cell_ids = np.arange(num_cells)

    tri_vert_chunks = []
    tri_norm_chunks = []
    for t in range(5):  # at most 5 triangles per cell
        col = tri_rows[:, t * 3 : t * 3 + 3]  # (M, 3) edge indices, or -1
        valid = col[:, 0] >= 0
        if not np.any(valid):
            continue
        idx = np.clip(col[valid], 0, 11)  # (V, 3)
        # TRI_TABLE's winding comes out backwards relative to our "high
        # field value = inside" / outward-gradient normal convention (the
        # table was authored for "value < isolevel" = inside); flipping the
        # last two corners of every triangle corrects it uniformly.
        idx = idx[:, [0, 2, 1]]
        cells = cell_ids[valid]
        # (V, 3 corners, 3 xyz), corner order already matches triangle winding
        verts_vc3 = edge_pos_flat[idx, cells[:, None]]
        norms_vc3 = edge_norm_flat[idx, cells[:, None]]
        tri_vert_chunks.append(verts_vc3.reshape(-1, 3))
        tri_norm_chunks.append(norms_vc3.reshape(-1, 3))

    if not tri_vert_chunks:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    verts = np.concatenate(tri_vert_chunks, axis=0)
    normals = np.concatenate(tri_norm_chunks, axis=0)

    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths = np.where(norm_lengths < _EPS, 1.0, norm_lengths)
    normals = normals / norm_lengths

    return verts, normals
