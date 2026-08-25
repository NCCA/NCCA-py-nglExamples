"""Headless tests for the physics side of the MuJoCo demo.

There is no Qt and no OpenGL in `physics_world.py` or `collision_shapes.py`, so
all of this runs without a window. What is worth testing is the part that is
easy to get wrong and invisible when it is: that the world really is Y-up, that
recompiling does not disturb the bodies already in the pile, that the pool's
parked bodies stay parked, and that the mesh frame correction actually cancels
what MuJoCo did to the mesh.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from ncca.ngl import Vec3

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

from collision_shapes import ShapeCatalogue, cone_hull, load_obj_hull  # noqa: E402
from physics_world import PoolWorld, RecompileWorld, make_world  # noqa: E402

PRIMITIVES = ["box", "sphere", "capsule", "cylinder"]
ALL_SHAPES = PRIMITIVES + ["cone", "teapot", "apple"]


@pytest.fixture(scope="module")
def catalogue():
    return ShapeCatalogue.default(model_dir=str(DEMO_DIR / "models"))


@pytest.fixture(params=["recompile", "pool"])
def world(request, catalogue):
    return make_world(request.param, catalogue)


def settle(world, seconds=8.0):
    for _ in range(int(seconds * 60)):
        world.step(1.0 / 60.0, 4)


def position_of(state):
    return np.array([float(state.transform[3, i]) for i in range(3)])


class TestGravityAndGround:
    def test_body_falls_and_lands_on_the_ground_plane(self, world):
        world.add_body("sphere", Vec3(0.0, 10.0, 0.0))
        settle(world)
        (state,) = list(world.bodies())
        # A 0.5 radius sphere at rest sits with its centre 0.5 above the plane.
        assert position_of(state)[1] == pytest.approx(0.5, abs=0.02)

    @pytest.mark.parametrize("shape", PRIMITIVES)
    def test_primitives_come_to_rest_above_the_ground(self, world, shape):
        world.add_body(shape, Vec3(0.0, 10.0, 0.0))
        settle(world)
        (state,) = list(world.bodies())
        assert position_of(state)[1] > 0.0

    def test_gravity_pulls_along_negative_y_not_z(self, world):
        """MuJoCo is Z-up by default, so this is the check that the port took."""
        world.add_body("sphere", Vec3(0.0, 10.0, 0.0))
        for _ in range(30):
            world.step(1.0 / 60.0, 4)
        (state,) = list(world.bodies())
        moved = position_of(state) - np.array([0.0, 10.0, 0.0])
        assert moved[1] < -0.05
        assert abs(moved[2]) < 1e-3

    def test_gravity_can_be_turned_off(self, world):
        world.add_body("sphere", Vec3(0.0, 10.0, 0.0))
        world.gravity = Vec3(0.0, 0.0, 0.0)
        settle(world, seconds=1.0)
        (state,) = list(world.bodies())
        assert position_of(state)[1] == pytest.approx(10.0, abs=1e-3)


class TestSpawning:
    def test_every_catalogue_shape_can_be_spawned(self, world, catalogue):
        for i, shape in enumerate(catalogue.names):
            world.add_body(shape, Vec3(float(i) * 4.0, 10.0, 0.0))
        assert world.num_bodies == len(catalogue.names)
        assert {b.shape for b in world.bodies()} == set(catalogue.names)

    def test_reset_clears_the_bodies(self, world):
        for i in range(3):
            world.add_body("box", Vec3(0.0, 10.0 + i * 3.0, 0.0))
        world.reset()
        assert world.num_bodies == 0
        assert list(world.bodies()) == []

    def test_bodies_can_be_added_after_a_reset(self, world):
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        world.reset()
        world.add_body("sphere", Vec3(0.0, 10.0, 0.0))
        settle(world)
        (state,) = list(world.bodies())
        assert state.shape == "sphere"
        assert position_of(state)[1] == pytest.approx(0.5, abs=0.02)


class TestRecompilePreservesState:
    """The reason `RecompileWorld` is viable at all."""

    def test_existing_body_does_not_move_when_another_is_added(self, catalogue):
        world = RecompileWorld(catalogue)
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        settle(world)
        before = position_of(next(iter(world.bodies())))

        world.add_body("sphere", Vec3(20.0, 10.0, 0.0))
        after = position_of(next(b for b in world.bodies() if b.shape == "box"))
        assert after == pytest.approx(before, abs=1e-6)

    def test_body_count_grows_the_model(self, catalogue):
        world = RecompileWorld(catalogue)
        before = world.model.nbody
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        assert world.model.nbody == before + 1


class TestPoolParking:
    """Parked bodies must not fall, collide, or show up in `bodies()`."""

    def test_unused_pool_bodies_stay_parked(self, catalogue):
        world = PoolWorld(catalogue, pool_size=4)
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        settle(world)
        # Only the one that was spawned is reported, whatever the pool holds.
        assert world.num_bodies == 1
        parked = world.data.qpos.reshape(-1, 7)[1:, 1]
        assert np.all(parked > 900.0)

    def test_gravity_compensation_is_compiled_in(self, catalogue):
        """If `ngravcomp` is 0 MuJoCo skips gravcomp and parked bodies drift."""
        world = PoolWorld(catalogue, pool_size=2)
        assert world.model.ngravcomp > 0

    def test_parked_bodies_do_not_collide(self, catalogue):
        world = PoolWorld(catalogue, pool_size=4)
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        settle(world, seconds=2.0)
        (state,) = list(world.bodies())
        # Nothing invisible in the park zone should have knocked it sideways.
        assert abs(position_of(state)[0]) < 0.5


class TestImpulse:
    def test_impulse_changes_velocity(self, world):
        world.add_body("sphere", Vec3(0.0, 10.0, 0.0))
        world.step(1.0 / 60.0, 1)
        # The pool compiles every shape in, so the spawned body's degrees of
        # freedom are not necessarily the first ones in the model.
        (body_id, _) = next(iter(world._live_bodies()))
        dof = world.model.jnt_dofadr[world.model.body_jntadr[body_id]]
        before = world.data.qvel[dof]
        world.add_impulse(Vec3(10.0, 0.0, 0.0))
        assert world.data.qvel[dof] > before + 1.0

    def test_impulse_leaves_an_empty_world_alone(self, world):
        world.add_impulse(Vec3(10.0, 0.0, 0.0))
        assert world.num_bodies == 0


class TestStrategiesAgree:
    def test_same_shape_from_the_same_height_lands_the_same(self, catalogue):
        results = {}
        for name in ("recompile", "pool"):
            world = make_world(name, catalogue)
            world.add_body("box", Vec3(0.0, 6.0, 0.0))
            settle(world)
            results[name] = position_of(next(iter(world.bodies())))
        assert results["recompile"] == pytest.approx(results["pool"], abs=1e-4)


class TestMeshes:
    def test_cone_hull_matches_ngl_orientation(self):
        """NGL's cone is Z-aligned with the base at z=0 and apex at z=height."""
        verts, faces = cone_hull(0.5, 2.0, slices=16)
        verts = verts.reshape(-1, 3)
        assert verts[:, 2].min() == pytest.approx(0.0)
        assert verts[:, 2].max() == pytest.approx(2.0)
        ring = verts[:16]
        assert np.hypot(ring[:, 0], ring[:, 1]) == pytest.approx(0.5)
        assert faces.reshape(-1, 3).max() < len(verts)

    def test_collision_mesh_loads_through_pyngl(self):
        verts, faces = load_obj_hull(
            str(DEMO_DIR / "models" / "teapotCollisionMesh.obj")
        )
        assert len(verts) % 3 == 0
        assert len(verts) > 0
        assert faces.reshape(-1, 3).max() < len(verts) // 3

    def test_mesh_correction_undoes_mujocos_reframing(self, catalogue):
        """MuJoCo moves mesh vertices into the principal inertia frame.

        The drawn OBJ has to be pushed back the other way or the visible teapot
        sits at an angle to the hull that is colliding. This checks the
        correction reproduces the original vertices from the stored ones.
        """
        world = RecompileWorld(catalogue)
        model = world.model
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "teapot")
        rot, pos = world._mesh_correction["teapot"]

        stored = model.mesh_vert[
            model.mesh_vertadr[index] : model.mesh_vertadr[index]
            + model.mesh_vertnum[index]
        ].astype(np.float64)
        recovered = (rot @ stored.T).T + pos

        original = load_obj_hull(str(DEMO_DIR / "models" / "teapotCollisionMesh.obj"))[
            0
        ].reshape(-1, 3)
        # Every hull vertex should land back on a vertex of the original file.
        nearest = np.abs(recovered[:, None, :] - original[None, :, :]).sum(-1).min(1)
        assert nearest.max() < 1e-5


class TestTransforms:
    def test_transform_is_row_vector_with_translation_in_row_three(self, world):
        world.add_body("box", Vec3(3.0, 10.0, -2.0))
        (state,) = list(world.bodies())
        assert position_of(state) == pytest.approx([3.0, 10.0, -2.0], abs=1e-5)

    def test_rotation_block_stays_orthonormal(self, world):
        world.add_body("box", Vec3(0.0, 10.0, 0.0))
        settle(world, seconds=1.0)
        (state,) = list(world.bodies())
        rot = np.array(
            [[float(state.transform[r, c]) for c in range(3)] for r in range(3)]
        )
        assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-5)
