"""Parsing and frame playback for .bvh motion-capture files.

A port of the C++ `Bvh::load()` in the original NGL BvhViewer, redone as a
recursive-descent parser over a flat token stream -- the hierarchy block is
literally an S-expression with `{`/`}` as the parens, which Python's call
stack handles far more directly than the hand-rolled `std::stack<Joint*>`
the C++ needed.

No Qt, no OpenGL: this only uses `ncca.ngl`'s pure-numpy maths, so it is
fully unit-testable without a GL context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ncca.ngl import Mat4, Quaternion, Transform, Vec3

# BVH files are conventionally authored in centimetres; this viewer's camera,
# ground grid and bone radii are all sized for the original C++ demo's
# world scale, so offsets and translations are scaled down to match it.
_UNIT_SCALE = 1.0 / 10.0


class BvhParseError(Exception):
    """Raised when a .bvh file's hierarchy or motion block is malformed."""


@dataclass
class Joint:
    """One joint in the skeleton hierarchy.

    Attributes
    ----------
        name : str
            the joint's name, or "End Site" for a leaf marker
        offset : Vec3
            rest-pose translation from the parent joint, already scaled
        channels : list[str]
            this joint's animation channels (e.g. "Xrotation"), in the
            order they are declared -- BVH files may declare rotation
            channels in any axis order, and that order is significant
        children : list[Joint]
            child joints, in declared order
        motion : np.ndarray | None
            per-frame channel values, shape (num_frames, len(channels));
            None for an End Site, which has no channels
    """

    name: str
    offset: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    channels: list[str] = field(default_factory=list)
    children: list["Joint"] = field(default_factory=list)
    motion: np.ndarray | None = None


class Bvh:
    """A parsed .bvh skeleton plus its current playback position.

    Attributes
    ----------
        root : Joint
            the root of the skeleton hierarchy
        num_frames : int
            how many frames of motion data the file has
        frame_time : float
            seconds between frames, as declared in the file
        current_frame : int
            the frame currently being displayed
    """

    def __init__(self, path: str | Path) -> None:
        """Load and parse a .bvh file.

        Args:
            path: path to the .bvh file.

        Raises:
            FileNotFoundError: if `path` does not exist.
            BvhParseError: if the file doesn't parse as a .bvh file.
        """
        self.path = Path(path)
        self._parse(self.path.read_text())

    @classmethod
    def from_text(cls, text: str) -> "Bvh":
        """Parse .bvh content directly from a string (mainly for tests)."""
        bvh = cls.__new__(cls)
        bvh.path = None
        bvh._parse(text)
        return bvh

    def _parse(self, text: str) -> None:
        try:
            hierarchy_text, motion_text = text.split("MOTION", 1)
        except ValueError as exc:
            raise BvhParseError("no MOTION section found") from exc

        tokens = hierarchy_text.split()
        if not tokens or tokens[0] != "HIERARCHY":
            raise BvhParseError("expected HIERARCHY at the start of the file")
        if len(tokens) < 2 or tokens[1] != "ROOT":
            raise BvhParseError("expected ROOT after HIERARCHY")

        self.root, _, joints_in_order = self._parse_joint(tokens, 3, tokens[2])
        self._load_motion(motion_text, joints_in_order)
        self.current_frame = 0
        self._root_offset = self._compute_root_offset()

    def _compute_root_offset(self) -> Vec3:
        """A constant world-space shift for the root, applied every frame.

        Most .bvh files use whatever coordinate system the capture rig was
        calibrated in -- the root's translation channels are rarely anywhere
        near the origin, and there's no guarantee the character's feet ever
        reach world Y=0. Left uncorrected, the character can end up floating
        well above the ground grid and wandering far from wherever the
        camera happens to be looking. This computes a fixed offset so frame 0
        starts centred over the origin in X/Z, and the lowest point of the
        *rest pose* (found by walking the offset chain with no rotation
        applied -- typically a foot or toe) sits on the ground at Y=0.

        This only fixes the starting point: X/Z motion relative to frame 0,
        and any vertical bob the animation itself has, are untouched.
        """
        x = self.root.offset.x
        y = self.root.offset.y
        z = self.root.offset.z
        if self.root.channels:
            frame0 = self.root.motion[0]
            for value, channel in zip(frame0, self.root.channels):
                if channel == "Xposition":
                    x += float(value) * _UNIT_SCALE
                elif channel == "Yposition":
                    y += float(value) * _UNIT_SCALE
                elif channel == "Zposition":
                    z += float(value) * _UNIT_SCALE

        lowest = 0.0

        def lowest_beneath(joint: Joint, cumulative_y: float) -> None:
            nonlocal lowest
            for child in joint.children:
                child_y = cumulative_y + child.offset.y
                lowest = min(lowest, child_y)
                lowest_beneath(child, child_y)

        lowest_beneath(self.root, 0.0)
        # -y cancels frame 0's own height, then -lowest drops the rest
        # pose's lowest point (still directly beneath the now-cancelled
        # root) the rest of the way to the floor.
        return Vec3(-x, -y - lowest, -z)

    def _parse_joint(
        self, tokens: list[str], pos: int, name: str, is_end_site: bool = False
    ) -> tuple[Joint, int, list[Joint]]:
        if pos >= len(tokens) or tokens[pos] != "{":
            raise BvhParseError(f"expected '{{' after joint name {name!r}")
        pos += 1

        joint = Joint(name=name)
        joints_in_order = [] if is_end_site else [joint]

        while True:
            if pos >= len(tokens):
                raise BvhParseError(f"unexpected end of file inside {name!r}")
            token = tokens[pos]
            if token == "}":
                return joint, pos + 1, joints_in_order
            elif token == "OFFSET":
                x, y, z = (float(v) for v in tokens[pos + 1 : pos + 4])
                joint.offset = Vec3(x, y, z) * _UNIT_SCALE
                pos += 4
            elif token == "CHANNELS":
                n = int(tokens[pos + 1])
                joint.channels = tokens[pos + 2 : pos + 2 + n]
                pos += 2 + n
            elif token == "JOINT":
                child, pos, child_joints = self._parse_joint(
                    tokens, pos + 2, tokens[pos + 1]
                )
                joint.children.append(child)
                joints_in_order.extend(child_joints)
            elif token == "End" and tokens[pos + 1] == "Site":
                child, pos, _ = self._parse_joint(
                    tokens, pos + 2, "End Site", is_end_site=True
                )
                joint.children.append(child)
            else:
                raise BvhParseError(f"unexpected token {token!r} in {name!r}")

    def _load_motion(self, motion_text: str, joints_in_order: list[Joint]) -> None:
        lines = [line for line in motion_text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            raise BvhParseError("MOTION section is missing Frames:/Frame Time:")

        frames_tokens = lines[0].split()
        if frames_tokens[0] != "Frames:":
            raise BvhParseError(f"expected 'Frames:', found {lines[0]!r}")
        self.num_frames = int(frames_tokens[1])

        time_tokens = lines[1].split()
        if time_tokens[:2] != ["Frame", "Time:"]:
            raise BvhParseError(f"expected 'Frame Time:', found {lines[1]!r}")
        self.frame_time = float(time_tokens[2])

        frame_lines = lines[2:]
        if len(frame_lines) < self.num_frames:
            raise BvhParseError(
                f"expected {self.num_frames} frames of motion data, "
                f"found {len(frame_lines)}"
            )

        data = np.array(
            [
                [float(v) for v in line.split()]
                for line in frame_lines[: self.num_frames]
            ],
            dtype=np.float32,
        )

        total_channels = sum(len(joint.channels) for joint in joints_in_order)
        if data.shape[1] != total_channels:
            raise BvhParseError(
                f"motion data has {data.shape[1]} columns, "
                f"but the hierarchy declares {total_channels} channels"
            )

        column = 0
        for joint in joints_in_order:
            n = len(joint.channels)
            joint.motion = data[:, column : column + n]
            column += n

    # ------------------------------------------------------------ playback
    def replay(self) -> None:
        """Jump back to the first frame."""
        self.current_frame = 0

    def seek(self, frame: int) -> None:
        """Move to a frame, clamping requests outside the clip range.

        Parameters
        ----------
            frame : int
                frame number requested by the timeline
        """
        self.current_frame = max(0, min(int(frame), self.num_frames - 1))

    def step_forward(self) -> None:
        """Move one frame forward, stopping at the last frame."""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1

    def step_backward(self) -> None:
        """Move one frame back, stopping at the first frame."""
        if self.current_frame > 0:
            self.current_frame -= 1

    def advance(self) -> None:
        """Move one frame forward, looping back to the start at the end.

        Called on every playback timer tick; `step_forward` is for manual
        single-stepping and deliberately does not loop.
        """
        if self.current_frame >= self.num_frames - 1:
            self.current_frame = 0
        else:
            self.current_frame += 1

    # ------------------------------------------------------------- pose
    def local_matrix(self, joint: Joint) -> Mat4:
        """The transform from `joint`'s parent to `joint`, at the current frame.

        Translates by the joint's rest-pose offset (plus this frame's
        translation channels, for the root) and rotates by this frame's
        rotation channels, composed in the order they were declared -- a
        joint may declare its rotation channels in any of the six axis
        orders, and that order changes the result. The root additionally
        gets `_root_offset` added, so it starts centred on the origin with
        its rest pose grounded at Y=0 (see `_compute_root_offset`).
        """
        position = Vec3(joint.offset.x, joint.offset.y, joint.offset.z)
        rotation = Vec3(0.0, 0.0, 0.0)
        rotation_axes: list[str] = []

        if joint.channels:
            frame = joint.motion[self.current_frame]
            for value, channel in zip(frame, joint.channels):
                axis = channel[0].lower()
                if channel.endswith("position"):
                    setattr(
                        position, axis, getattr(position, axis) + value * _UNIT_SCALE
                    )
                elif channel.endswith("rotation"):
                    setattr(rotation, axis, float(value))
                    rotation_axes.append(axis)

        if joint is self.root:
            position = position + self._root_offset

        # PyNGL's Transform.rot_order composes with the *last* axis applied
        # first (order "xyz" -> physically X, then Y, then Z last), whereas a
        # BVH file's declared channel order is the physical application
        # order (first declared applied first) -- so the axis letters have
        # to be reversed to get the order string Transform expects.
        order = "".join(reversed(rotation_axes))
        if order not in Transform.rot_order:
            order = "xyz"  # channels missing/incomplete: unused axes are 0 anyway

        transform = Transform()
        transform.set_position(position)
        transform.set_rotation(rotation)
        transform.set_order(order)
        return transform.matrix()


def rotation_from_y(direction: Vec3) -> Mat4:
    """A rotation matrix that takes the world +Y axis onto `direction`.

    PyNGL's stock cylinder primitive runs along +Y, so this is what orients
    a bone -- built by scaling that cylinder along Y -- to point along the
    offset vector between two joints.
    """
    d = direction.normalized()
    up = Vec3(0.0, 1.0, 0.0)
    axis = up.cross(d)
    axis_length = float(axis.length())
    cos_angle = max(-1.0, min(1.0, float(up.dot(d))))

    if axis_length < 1e-6:
        # `direction` is parallel or antiparallel to +Y: no rotation axis is
        # defined by the cross product, so pick one by hand.
        if cos_angle > 0.0:
            return Mat4()
        return Mat4.rotate_x(180.0)

    angle = math.degrees(math.acos(cos_angle))
    return Quaternion.from_axis_angle(axis / axis_length, angle).to_mat4()
