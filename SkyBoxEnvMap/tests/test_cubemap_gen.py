"""Headless tests for the procedural cubemap generator (numpy-only maths)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cubemap_gen import FACE_ORDER, generate_cubemap_faces


class TestFaceCountShapeDtype:
    def test_returns_six_faces(self):
        faces = generate_cubemap_faces(64)
        assert len(faces) == 6

    def test_face_order_is_gl_order(self):
        assert FACE_ORDER == ("+x", "-x", "+y", "-y", "+z", "-z")

    def test_face_shape(self):
        size = 64
        faces = generate_cubemap_faces(size)
        for face in faces:
            assert face.shape == (size, size, 4)

    def test_face_dtype_is_uint8(self):
        faces = generate_cubemap_faces(32)
        for face in faces:
            assert face.dtype == np.uint8

    def test_default_size_is_256(self):
        faces = generate_cubemap_faces()
        assert faces[0].shape == (256, 256, 4)

    def test_alpha_channel_is_opaque(self):
        faces = generate_cubemap_faces(16)
        for face in faces:
            assert np.all(face[..., 3] == 255)


class TestHorizonContinuity:
    """The classic cubemap bug is a face-order / edge-orientation mistake --
    catch it by checking that the shared cube edge between +x and +z (the
    line x=1, z=1) produces identical pixels from both faces."""

    def test_right_edge_of_plus_x_matches_left_edge_of_plus_z(self):
        faces = generate_cubemap_faces(64)
        plus_x = faces[FACE_ORDER.index("+x")]
        plus_z = faces[FACE_ORDER.index("+z")]
        # right edge (last column) of +x vs left edge (first column) of +z
        np.testing.assert_array_equal(plus_x[:, -1, :], plus_z[:, 0, :])

    def test_faces_are_not_trivially_identical(self):
        """Sanity check that the continuity test above isn't passing because
        every face is a flat constant colour."""
        faces = generate_cubemap_faces(64)
        plus_y = faces[FACE_ORDER.index("+y")]
        assert not np.all(plus_y == plus_y[0, 0])


class TestSunDisc:
    def test_sun_bright_spot_only_on_plus_z_face(self):
        """A bright sun disc is baked into the +z face; the equivalent
        region of -z (directly behind the camera) should stay dim sky/
        ground colour, not a bright disc."""
        faces = generate_cubemap_faces(128)
        plus_z = faces[FACE_ORDER.index("+z")]
        minus_z = faces[FACE_ORDER.index("-z")]
        # brightest pixel on +z should be much brighter than the brightest
        # pixel on -z (which has no sun baked in)
        assert plus_z[..., :3].max() > minus_z[..., :3].max()
