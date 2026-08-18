"""Headless tests for the screen-to-world unprojection maths."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from view_to_world import unproject_point  # noqa: E402


class TestUnprojectPoint:
    def test_identity_centre_of_screen_far_plane(self):
        """With an identity view-projection, the screen centre at the far
        plane (ndc_z=1, the default) maps to NDC (0,0,1)."""
        point = unproject_point(400, 300, 800, 600, np.eye(4))
        np.testing.assert_allclose(point, [0.0, 0.0, 1.0], atol=1e-6)

    def test_identity_top_left_corner(self):
        """Top-left pixel maps to NDC (-1, +1)."""
        point = unproject_point(0, 0, 800, 600, np.eye(4))
        np.testing.assert_allclose(point[:2], [-1.0, 1.0], atol=1e-6)

    def test_near_plane_selectable(self):
        """ndc_z=-1 selects the near plane instead of the far plane."""
        point = unproject_point(400, 300, 800, 600, np.eye(4), ndc_z=-1.0)
        np.testing.assert_allclose(point, [0.0, 0.0, -1.0], atol=1e-6)

    def test_translated_view_projection_offsets_result(self):
        """A translation in the view-projection matrix shows up in the
        unprojected point (row-vector convention: translation in row 3).

        vp maps world -> clip as clip_xyz = world_xyz + (5, -2, 1) (a pure
        translation, since the 3x3 block is identity). Unprojecting the NDC
        point (0, 0, 1) must therefore recover world = (0,0,1) - (5,-2,1) =
        (-5, 2, 0); verified independently by a forward round-trip below.
        """
        vp = np.eye(4)
        vp[3, :3] = [5.0, -2.0, 1.0]
        point = unproject_point(400, 300, 800, 600, vp)
        np.testing.assert_allclose(point, [-5.0, 2.0, 0.0], atol=1e-6)

        # Round-trip check: projecting the unprojected point forward through
        # vp (world_h @ vp, per the row-vector convention) must land back on
        # the same NDC coordinate we unprojected from.
        world_h = np.array([point[0], point[1], point[2], 1.0])
        clip = world_h @ vp
        np.testing.assert_allclose(clip[:3] / clip[3], [0.0, 0.0, 1.0], atol=1e-6)
