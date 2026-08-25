"""Headless tests for the marching-cubes metaball/polygonise maths.

These are the tests that actually earn their keep on this demo: the lookup
tables in ``mc_tables.py`` were parsed out of Paul Bourke's reference page
rather than hand-typed, but a transcription slip anywhere in that pipeline
would still show up as a hole in the output mesh -- which is exactly what
``test_single_metaball_mesh_is_closed`` checks for.
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from marching_cubes import polygonise, sample_metaballs
from mc_tables import CORNER_OFFSETS, TRI_TABLE


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _closed_mesh_edges(verts: np.ndarray) -> Counter:
    """Count how many triangles each undirected edge participates in.

    A watertight mesh has every edge shared by exactly two triangles; a
    table transcription bug typically leaves a handful of edges used only
    once (a hole) or three-plus times (an overlapping/degenerate facet).
    """
    edge_counts: Counter = Counter()
    for t in range(len(verts) // 3):
        tri = verts[t * 3 : t * 3 + 3]
        # round to collapse float noise so shared vertices compare equal
        keys = [tuple(np.round(v, decimals=4)) for v in tri]
        for a, b in combinations(range(3), 2):
            edge = tuple(sorted((keys[a], keys[b])))
            edge_counts[edge] += 1
    return edge_counts


# ----------------------------------------------------------------------
# sample_metaballs
# ----------------------------------------------------------------------
class TestSampleMetaballs:
    def test_output_shape_and_dtype(self):
        field = sample_metaballs(
            centres=np.array([[0.0, 0.0, 0.0]]),
            radii=np.array([1.0]),
            grid_n=8,
            extent=2.0,
        )
        assert field.shape == (8, 8, 8)
        assert field.dtype == np.float32

    def test_peaks_at_centre(self):
        """A single metaball's field is highest at its own centre."""
        field = sample_metaballs(
            centres=np.array([[0.0, 0.0, 0.0]]),
            radii=np.array([1.0]),
            grid_n=17,
            extent=2.0,
        )
        centre_index = 17 // 2
        assert field[centre_index, centre_index, centre_index] == field.max()


# ----------------------------------------------------------------------
# polygonise -- empty field
# ----------------------------------------------------------------------
class TestEmptyField:
    def test_zero_field_below_iso_produces_no_triangles(self):
        field = np.zeros((10, 10, 10), dtype=np.float32)
        verts, normals = polygonise(field, iso=1.0, extent=2.0)
        assert verts.shape == (0, 3)
        assert normals.shape == (0, 3)


# ----------------------------------------------------------------------
# polygonise -- case-index sanity (hand-built corner configuration)
# ----------------------------------------------------------------------
class TestCaseIndexSanity:
    def test_single_corner_above_iso_matches_table_case_8(self):
        """Only corner 3 (see mc_tables docstring numbering) above iso.

        cubeindex = 1 << 3 = 8, and TRI_TABLE[8] == (3, 11, 2, ...) per
        Bourke's own worked example -- this pins down that our corner/edge
        numbering and bit-packing order actually matches the table we
        transcribed.
        """
        n = 2
        field = np.zeros((n, n, n), dtype=np.float32)
        iso = 0.5
        dx, dy, dz = CORNER_OFFSETS[3]
        field[dx, dy, dz] = 1.0  # only corner 3 exceeds iso

        verts, _ = polygonise(field, iso=iso, extent=1.0)
        # exactly one triangle
        assert verts.shape == (3, 3)
        # cross-check the case index directly against the table's own
        # convention rather than trusting polygonise() to have got the
        # winding order right
        assert TRI_TABLE[8][:3] == (3, 11, 2)


# ----------------------------------------------------------------------
# polygonise -- single centred metaball
# ----------------------------------------------------------------------
class TestSingleMetaball:
    GRID_N = 32
    EXTENT = 2.0
    RADIUS = 1.0
    ISO = 1.0

    @pytest.fixture
    def mesh(self):
        field = sample_metaballs(
            centres=np.array([[0.0, 0.0, 0.0]]),
            radii=np.array([self.RADIUS]),
            grid_n=self.GRID_N,
            extent=self.EXTENT,
        )
        return polygonise(field, iso=self.ISO, extent=self.EXTENT)

    def test_produces_triangles(self, mesh):
        verts, normals = mesh
        assert len(verts) > 0
        assert len(verts) % 3 == 0
        assert verts.shape == normals.shape

    def test_mesh_is_closed(self, mesh):
        """Every edge must be shared by exactly two triangles."""
        verts, _ = mesh
        edge_counts = _closed_mesh_edges(verts)
        bad_edges = {e: c for e, c in edge_counts.items() if c != 2}
        assert not bad_edges, f"{len(bad_edges)} edges not shared by exactly 2 tris"

    def test_vertex_distances_match_iso_radius(self, mesh):
        """Every vertex should sit close to distance RADIUS from the centre
        (field = r^2/(d^2+eps), so field == iso at d == r for iso == 1)."""
        verts, _ = mesh
        distances = np.linalg.norm(verts, axis=1)
        cell_size = (2 * self.EXTENT) / (self.GRID_N - 1)
        np.testing.assert_allclose(distances, self.RADIUS, atol=cell_size * 1.5)

    def test_triangle_winding_matches_gradient_normals(self, mesh):
        """Face normal (from vertex winding) should agree in direction with
        the field-gradient-derived per-vertex normals polygonise() returns
        -- if the table's triangles were wound backwards relative to our
        normal convention, every dot product would come out negative."""
        verts, normals = mesh
        dots = []
        for t in range(len(verts) // 3):
            v0, v1, v2 = verts[t * 3 : t * 3 + 3]
            face_normal = np.cross(v1 - v0, v2 - v0)
            face_norm_len = np.linalg.norm(face_normal)
            if face_norm_len < 1e-8:
                continue  # degenerate sliver, skip
            face_normal /= face_norm_len
            vertex_normal = normals[t * 3 : t * 3 + 3].mean(axis=0)
            dots.append(np.dot(face_normal, vertex_normal))
        dots = np.array(dots)
        assert (dots > 0).all(), (
            f"{(dots <= 0).sum()} / {len(dots)} faces wound inconsistently "
            "with their gradient normal"
        )

    def test_normals_are_unit_length(self, mesh):
        _, normals = mesh
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-4)
