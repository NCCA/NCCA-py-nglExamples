"""Tests for the colour-picking and drag maths.

None of this touches Qt or OpenGL, so it can all be checked headlessly --
which matters, because getting an unproject subtly wrong produces a drag
that merely feels a bit off rather than something that crashes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Mat4, Vec3, look_at, perspective
from picking import (
    decode_id,
    encode_id,
    intersect_plane,
    ray_from_screen,
    transform_point,
)


def _camera() -> np.ndarray:
    view = look_at(Vec3(0, 0, 7), Vec3(0, 0, 0), Vec3(0, 1, 0))
    project = perspective(45.0, 1.0, 0.5, 150.0)
    return (project @ view).to_numpy()


def test_ids_round_trip_over_the_whole_range():
    for index in range(64):
        assert decode_id(encode_id(index)) == index


def test_black_is_the_background_not_an_index():
    assert decode_id((0, 0, 0)) is None
    # so no index may ever encode to black
    assert encode_id(0) != (0, 0, 0)


def test_ids_stay_inside_a_byte_per_channel():
    for index in range(64):
        assert all(0 <= c <= 255 for c in encode_id(index))


def test_a_ray_down_the_screen_centre_points_along_the_view():
    origin, direction = ray_from_screen(50.0, 50.0, 100, 100, _camera())
    # camera sits at +7z looking at the origin, so the centre ray runs -z
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0], atol=1e-5)
    assert origin[2] < 7.0


def test_the_centre_ray_hits_the_origin_plane_at_the_origin():
    origin, direction = ray_from_screen(50.0, 50.0, 100, 100, _camera())
    hit = intersect_plane(origin, direction, np.zeros(3), np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(hit, [0.0, 0.0, 0.0], atol=1e-4)


def test_a_ray_above_centre_hits_the_plane_higher_up():
    # y=25 is above the centre in Qt's top-left pixel origin
    origin, direction = ray_from_screen(50.0, 25.0, 100, 100, _camera())
    hit = intersect_plane(origin, direction, np.zeros(3), np.array([0.0, 0.0, -1.0]))
    assert hit[1] > 0.1
    assert abs(hit[0]) < 1e-4


def test_a_ray_parallel_to_the_plane_misses():
    hit = intersect_plane(
        np.array([0.0, 0.0, 5.0]),
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -1.0]),
    )
    assert hit is None


def test_transform_point_round_trips_through_a_matrix_and_its_inverse():
    m = Mat4().rotate_y(35.0) @ Mat4().rotate_x(20.0)
    mat = m.to_numpy()
    p = np.array([1.0, -2.0, 0.5])
    there = transform_point(p, mat)
    back = transform_point(there, np.linalg.inv(mat))
    np.testing.assert_allclose(back, p, atol=1e-6)
