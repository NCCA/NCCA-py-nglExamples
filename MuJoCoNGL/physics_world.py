"""The physics side of the demo, wrapping MuJoCo behind the C++ demo's facade.

The C++ version of this demo has a `PhysicsWorld` class with `addBox`,
`addSphere`, `step`, `reset` and so on, and Bullet is happy to take a new
`btRigidBody` at any point in the simulation. MuJoCo is not built that way. You
describe a model, compile it into an `mjModel`, and from then on the model is
fixed -- there is no `addBody`. That single difference is the interesting part
of this port, and there are two reasonable ways round it, so both are here
behind the same interface:

`RecompileWorld` keeps a live `MjSpec`, appends a body to it and calls
`spec.recompile(model, data)`, which hands back a new model and data with the
existing bodies still where they were. This is the idiomatic MuJoCo 3.x answer
and it reads almost exactly like the Bullet original, at the cost of a compile
every time you press a key.

`PoolWorld` compiles every body it will ever need up front, parks them out of
the way and "spawns" one by teleporting it into place. Nothing is ever
recompiled, so spawning is free, but the number of bodies is capped and the
model is a good deal less obvious to read.

Neither class imports anything from OpenGL or Qt. `bodies()` hands out plain
PyNGL `Mat4` transforms, which is what lets the tests run headless.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco
import numpy as np
from collision_shapes import SPAWN_HEIGHT, ShapeCatalogue
from ncca.ngl import Mat4, Vec3

# MuJoCo planes have their normal along local +Z and NGL is Y-up, so the ground
# is rotated -90 degrees about X. Written out as a quaternion (w, x, y, z).
_GROUND_QUAT = [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]

# Where PoolWorld parks a body it is not currently using. Well above the ground
# rather than below it: a MuJoCo plane is an infinite half-space, so a body
# parked underneath one is deeply penetrating it and the solver will fire it
# back out the moment collisions are re-enabled.
_PARK_POSITION = [0.0, 1000.0, 0.0]

DEFAULT_GRAVITY = Vec3(0.0, -9.81, 0.0)
POOL_SIZE_PER_SHAPE = 32


@dataclass(frozen=True)
class BodyState:
    """One body's shape and where it is, ready to draw.

    Attributes
    ----------
        shape : str
            catalogue name, telling the scene which mesh to draw
        transform : Mat4
            full model matrix, mesh frame correction already applied
    """

    shape: str
    transform: Mat4


def _mesh_frame_correction(model: "mujoco.MjModel") -> dict:
    """Works out how to undo MuJoCo's repositioning of each mesh.

    MuJoCo does not keep a mesh where you gave it to it. On compile it moves the
    vertices into the mesh's principal inertia frame, so both `mesh_pos` and
    `mesh_quat` come back as a real translation and rotation. Drawing the
    original OBJ at the body transform would put the visible teapot at an angle
    to the hull that is actually colliding, which looks like a physics bug and
    is not one.

    A vertex of the original file relates to the one MuJoCo stored by
    `original = R @ stored + p`, so drawing the original needs the inverse of
    that folded in ahead of the body transform.

    Returns
    -------
        dict of mesh name to (3x3 rotation, translation) to correct by
    """
    correction = {}
    for i in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.mesh_quat[i])
        correction[name] = (rot.reshape(3, 3), model.mesh_pos[i].copy())
    return correction


class PhysicsWorld(ABC):
    """What both spawning strategies have in common.

    Attributes
    ----------
        catalogue : ShapeCatalogue
            the shapes this world can spawn
        last_spawn_ms : float
            how long the most recent spawn took, which is the whole point of
            having two strategies to compare
    """

    def __init__(self, catalogue: ShapeCatalogue, gravity: Vec3 = DEFAULT_GRAVITY):
        self.catalogue = catalogue
        self._gravity = gravity
        self.last_spawn_ms: float = 0.0
        self._spawned: list[tuple[str, Vec3]] = []
        self.model: "mujoco.MjModel" = None
        self.data: "mujoco.MjData" = None

    def _new_spec(self) -> "mujoco.MjSpec":
        """Builds a spec holding the ground plane and every mesh, but no bodies."""
        spec = mujoco.MjSpec()
        spec.option.gravity = [self._gravity.x, self._gravity.y, self._gravity.z]
        self.catalogue.declare_meshes(spec)
        ground = spec.worldbody.add_geom()
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground.size = [70.0, 70.0, 0.1]
        ground.quat = _GROUND_QUAT
        return spec

    @abstractmethod
    def add_body(self, shape: str, position: Vec3) -> None:
        """Drops a new body of the named shape into the world."""

    @abstractmethod
    def reset(self) -> None:
        """Removes everything except the ground plane."""

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """How many bodies are in play, not counting the ground."""

    @abstractmethod
    def bodies(self):
        """Yields a BodyState for each live body."""

    def step(self, dt: float, substeps: int = 1) -> None:
        self.model.opt.timestep = dt / substeps
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    @property
    def gravity(self) -> Vec3:
        return self._gravity

    @gravity.setter
    def gravity(self, value: Vec3) -> None:
        self._gravity = value
        self.model.opt.gravity[:] = [value.x, value.y, value.z]

    @property
    def integrator(self) -> int:
        return int(self.model.opt.integrator)

    @integrator.setter
    def integrator(self, value: int) -> None:
        self.model.opt.integrator = value

    @property
    def solver_iterations(self) -> int:
        return int(self.model.opt.iterations)

    @solver_iterations.setter
    def solver_iterations(self, value: int) -> None:
        self.model.opt.iterations = value

    def _body_transform(self, body_id: int, shape: str) -> Mat4:
        """Turns MuJoCo's body frame into a PyNGL model matrix.

        MuJoCo works in the column-vector convention, so `world = xmat @ local +
        xpos`, whilst PyNGL's `Mat4` is row-vector with the translation along
        row 3. That means the rotation goes in transposed. Mesh shapes also pick
        up the correction that undoes MuJoCo's move to the principal inertia
        frame.
        """
        rotation = self.data.xmat[body_id].reshape(3, 3)
        position = self.data.xpos[body_id]
        correction = self._mesh_correction.get(shape)
        if correction is not None:
            mesh_rot, mesh_pos = correction
            rotation = rotation @ mesh_rot.T
            position = position - rotation @ mesh_pos

        matrix = Mat4()
        matrix[0:3, 0:3] = rotation.T
        matrix[3, 0] = position[0]
        matrix[3, 1] = position[1]
        matrix[3, 2] = position[2]
        return matrix

    def add_impulse(self, impulse: Vec3) -> None:
        """Kicks every body, the way the C++ demo's arrow keys did.

        Bullet had `applyCentralImpulse`; the equivalent here is to add the
        change in velocity straight onto the free joint's velocities, since an
        impulse over a mass is a change in velocity.
        """
        for body_id, _ in self._live_bodies():
            joint = self.model.body_jntadr[body_id]
            if joint < 0:
                continue
            adr = self.model.jnt_dofadr[joint]
            mass = self.model.body_mass[body_id]
            if mass <= 0.0:
                continue
            self.data.qvel[adr : adr + 3] += (
                np.array([impulse.x, impulse.y, impulse.z]) / mass
            )

    @abstractmethod
    def _live_bodies(self):
        """Yields (body_id, shape name) for each body that is in play."""


class RecompileWorld(PhysicsWorld):
    """Adds bodies by editing the spec and recompiling it.

    `MjSpec.recompile` carries the state of the existing bodies across into the
    new model, so the pile on the floor does not so much as twitch when a new
    body is added above it. The cost is a full model compile per spawn, which is
    a few milliseconds and shows up in the panel's readout.
    """

    def __init__(self, catalogue: ShapeCatalogue, gravity: Vec3 = DEFAULT_GRAVITY):
        super().__init__(catalogue, gravity)
        self.reset()

    def reset(self) -> None:
        self._spec = self._new_spec()
        self._names: list[str] = []
        self._spawned = []
        self.model = self._spec.compile()
        self.data = mujoco.MjData(self.model)
        self._mesh_correction = _mesh_frame_correction(self.model)
        self.last_spawn_ms = 0.0

    def add_body(self, shape: str, position: Vec3) -> None:
        start = time.perf_counter()
        name = f"{shape}_{len(self._names)}"
        body = self._spec.worldbody.add_body()
        body.name = name
        body.pos = [position.x, position.y, position.z]
        body.add_freejoint()
        self.catalogue.shapes[shape].add_geom(body)
        # recompile takes the old model and data and returns new ones with the
        # existing state mapped across, which is the whole reason this approach
        # works at all.
        self.model, self.data = self._spec.recompile(self.model, self.data)
        self._mesh_correction = _mesh_frame_correction(self.model)
        # recompile leaves the derived quantities stale, so xpos and xmat are
        # all zero until the kinematics are run. Without this a body drawn
        # before the next step lands at the origin.
        mujoco.mj_forward(self.model, self.data)
        self._names.append(name)
        self._spawned.append((shape, position))
        self.last_spawn_ms = (time.perf_counter() - start) * 1000.0

    @property
    def num_bodies(self) -> int:
        return len(self._names)

    def _live_bodies(self):
        for index, name in enumerate(self._names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                yield body_id, self._spawned[index][0]

    def bodies(self):
        for body_id, shape in self._live_bodies():
            yield BodyState(shape, self._body_transform(body_id, shape))


class PoolWorld(PhysicsWorld):
    """Compiles a fixed pool of bodies up front and recycles them.

    Every body the world will ever have is in the model from the start, parked
    at `_PARK_POSITION` with collisions switched off and gravity compensated so
    it sits still. Spawning moves one into place and switches both back on, so
    it costs nothing beyond writing a few numbers into `qpos`.

    The gravity compensation is worth a note. `body_gravcomp` can be written at
    runtime, but MuJoCo counts the bodies using it at compile time and skips the
    whole calculation when that count is zero -- so if none of the bodies asked
    for it in the spec, writing to `body_gravcomp` later is quietly ignored and
    the parked bodies slide away. Setting it in the spec, which is where the
    pool wants it anyway, is what makes the runtime toggle work.

    Attributes
    ----------
        pool_size : int
            how many of each shape are compiled in
    """

    def __init__(
        self,
        catalogue: ShapeCatalogue,
        gravity: Vec3 = DEFAULT_GRAVITY,
        pool_size: int = POOL_SIZE_PER_SHAPE,
    ):
        self.pool_size = pool_size
        super().__init__(catalogue, gravity)
        self._build()

    def _build(self) -> None:
        spec = self._new_spec()
        self._slots: dict[str, list[str]] = {}
        for shape in self.catalogue.names:
            names = []
            for i in range(self.pool_size):
                name = f"{shape}_{i}"
                body = spec.worldbody.add_body()
                body.name = name
                body.pos = _PARK_POSITION
                # Asking for gravity compensation here is what makes it
                # available to switch on and off later, see the class docstring.
                body.gravcomp = 1.0
                body.add_freejoint()
                self.catalogue.shapes[shape].add_geom(body)
                names.append(name)
            self._slots[shape] = names

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._mesh_correction = _mesh_frame_correction(self.model)
        self._in_use: list[tuple[int, str]] = []
        self._next: dict[str, int] = {name: 0 for name in self.catalogue.names}
        for shape, names in self._slots.items():
            for name in names:
                self._park(self._body_id(name))

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _set_active(self, body_id: int, active: bool) -> None:
        """Switches a pooled body between in play and parked."""
        start = self.model.body_geomadr[body_id]
        for geom in range(start, start + self.model.body_geomnum[body_id]):
            self.model.geom_contype[geom] = 1 if active else 0
            self.model.geom_conaffinity[geom] = 1 if active else 0
        self.model.body_gravcomp[body_id] = 0.0 if active else 1.0

    def _park(self, body_id: int) -> None:
        self._set_active(body_id, False)
        self._write_pose(body_id, Vec3(*_PARK_POSITION))

    def _write_pose(self, body_id: int, position: Vec3) -> None:
        joint = self.model.body_jntadr[body_id]
        adr = self.model.jnt_qposadr[joint]
        dof = self.model.jnt_dofadr[joint]
        self.data.qpos[adr : adr + 3] = [position.x, position.y, position.z]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dof : dof + 6] = 0.0

    def reset(self) -> None:
        for body_id, _ in self._in_use:
            self._park(body_id)
        self._in_use = []
        self._spawned = []
        self._next = {name: 0 for name in self.catalogue.names}
        self.last_spawn_ms = 0.0
        mujoco.mj_forward(self.model, self.data)

    def add_body(self, shape: str, position: Vec3) -> None:
        start = time.perf_counter()
        slot = self._next[shape]
        if slot >= self.pool_size:
            # The pool is the price of never recompiling; when it runs out the
            # oldest body of that shape is recycled rather than growing it.
            slot = 0
            self._next[shape] = 0
        name = self._slots[shape][slot]
        body_id = self._body_id(name)
        self._write_pose(body_id, position)
        self._set_active(body_id, True)
        self._next[shape] = slot + 1
        entry = (body_id, shape)
        if entry not in self._in_use:
            self._in_use.append(entry)
        self._spawned.append((shape, position))
        mujoco.mj_forward(self.model, self.data)
        self.last_spawn_ms = (time.perf_counter() - start) * 1000.0

    @property
    def num_bodies(self) -> int:
        return len(self._in_use)

    def _live_bodies(self):
        yield from self._in_use

    def bodies(self):
        for body_id, shape in self._in_use:
            yield BodyState(shape, self._body_transform(body_id, shape))


STRATEGIES = {"recompile": RecompileWorld, "pool": PoolWorld}


def make_world(
    strategy: str,
    catalogue: ShapeCatalogue,
    gravity: Vec3 = DEFAULT_GRAVITY,
) -> PhysicsWorld:
    """Builds a world using the named spawning strategy.

    Parameters
    ----------
        strategy : str
            either "recompile" or "pool"
        catalogue : ShapeCatalogue
            shapes the world can spawn
        gravity : Vec3
            gravity vector, Y-up to match NGL
    """
    return STRATEGIES[strategy](catalogue, gravity)


__all__ = [
    "BodyState",
    "PhysicsWorld",
    "PoolWorld",
    "RecompileWorld",
    "SPAWN_HEIGHT",
    "make_world",
]
