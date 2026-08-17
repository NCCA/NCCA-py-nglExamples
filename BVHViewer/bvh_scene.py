"""The OpenGL view of a BVH skeleton, or several of them.

A port of the drawing half of the C++ `Bvh`/`Scene` pair: `Bvh::drawBones`
walked the joint tree building bone transforms by hand from a fixed-Z-axis
cylinder; here it's the same recursive walk, but the maths is built for
PyNGL's cylinder, which runs along +Y (see `rotation_from_y` in `bvh.py`).

There is no `Wall` class here as there was in the C++ `Scene` -- the app
only ever draws one flat ground grid, so that generality wasn't earning its
keep and became a single hardcoded ground primitive instead.
"""

import colorsys

import numpy as np
import OpenGL.GL as gl
from bvh import Bvh, Joint, rotation_from_y
from ncca.ngl import Mat3, Mat4, Prims, Vec3
from ncca.ngl.opengl import (
    DefaultShader,
    Primitives,
    ShaderLib,
    VAOFactory,
    VAOType,
    VertexData,
)

_JOINT_RADIUS = 0.25
_BONE_RADIUS = 0.15
_GROUND_SIZE = 200.0


def joint_position_traces(character: Bvh) -> np.ndarray:
    """Build one world-space position line for each joint in a clip.

    Parameters
    ----------
        character : Bvh
            character whose animation frames are sampled

    Returns
    -------
        np.ndarray
            joint positions with shape ``(joints, frames, 3)``
    """
    joints: list[Joint] = []

    def collect_joints(joint: Joint) -> None:
        joints.append(joint)
        for child in joint.children:
            collect_joints(child)

    collect_joints(character.root)
    traces = np.empty((len(joints), character.num_frames, 3), dtype=np.float32)
    saved_frame = character.current_frame

    def store_positions(joint: Joint, parent_world: Mat4, joint_index: int) -> int:
        world = parent_world @ character.local_matrix(joint)
        traces[joint_index, frame] = [world[3, 0], world[3, 1], world[3, 2]]
        next_index = joint_index + 1
        for child in joint.children:
            next_index = store_positions(child, world, next_index)
        return next_index

    try:
        for frame in range(character.num_frames):
            character.seek(frame)
            store_positions(character.root, Mat4(), 0)
    finally:
        character.seek(saved_frame)
    return traces


def joint_trace_colours(count: int) -> np.ndarray:
    """Return a different RGBA colour for each joint trace."""
    colours = [
        (*colorsys.hsv_to_rgb(index / max(count, 1), 0.75, 1.0), 1.0)
        for index in range(count)
    ]
    return np.asarray(colours, dtype=np.float32).reshape(count, 4)


class BvhScene:
    """Owns the animated characters and the ground grid, and draws them.

    Attributes
    ----------
        characters : list[Bvh]
            the skeletons currently in the scene
        paused : bool
            when true, `advance()` does nothing
    """

    def __init__(self) -> None:
        """Initialize an empty scene."""
        self.characters: list[Bvh] = []
        self.paused = False
        self._view = Mat4()
        self._project = Mat4()
        self._mouse_global_tx = Mat4()
        self._trace_vertices = np.empty((0, 3), dtype=np.float32)
        self._trace_ranges: list[tuple[int, int]] = []
        self._trace_colours = np.empty((0, 4), dtype=np.float32)
        self._trace_vao = None
        self._trace_data_dirty = False

    # --------------------------------------------------------- characters
    def add_character(self, character: Bvh) -> None:
        """Add a character to the scene."""
        self.characters.append(character)
        self._rebuild_trace_data()

    def set_character(self, character: Bvh) -> None:
        """Replace the scene contents with one character."""
        self.characters = [character]
        self._rebuild_trace_data()

    def clear_characters(self) -> None:
        """Remove every character from the scene."""
        self.characters.clear()
        self._rebuild_trace_data()

    def _rebuild_trace_data(self) -> None:
        lines = [
            line
            for character in self.characters
            for line in joint_position_traces(character)
        ]
        self._trace_ranges = []
        first = 0
        for line in lines:
            self._trace_ranges.append((first, len(line)))
            first += len(line)
        self._trace_vertices = (
            np.concatenate(lines).astype(np.float32, copy=False)
            if lines
            else np.empty((0, 3), dtype=np.float32)
        )
        self._trace_colours = joint_trace_colours(len(lines))
        self._trace_data_dirty = True

    def replay(self) -> None:
        """Jump every character back to its first frame."""
        for character in self.characters:
            character.replay()

    def step_forward(self) -> None:
        """Step every character forward one frame, regardless of pause state."""
        for character in self.characters:
            character.step_forward()

    def step_backward(self) -> None:
        """Step every character back one frame, regardless of pause state."""
        for character in self.characters:
            character.step_backward()

    def seek(self, frame: int) -> None:
        """Move every character to a frame in its own clip range.

        Parameters
        ----------
            frame : int
                requested timeline frame
        """
        for character in self.characters:
            character.seek(frame)

    def toggle_pause(self) -> None:
        """Toggle whether `advance()` moves the animation on."""
        self.paused = not self.paused

    def advance(self) -> None:
        """Advance every character by one frame, unless paused."""
        if self.paused:
            return
        for character in self.characters:
            character.advance()

    def advance_in_range(self, range_start: int, range_end: int) -> None:
        """Advance each character within an inclusive playback range.

        Parameters
        ----------
            range_start : int
                first frame in the playback range
            range_end : int
                last frame in the playback range
        """
        if self.paused:
            return
        for character in self.characters:
            character.advance_in_range(range_start, range_end)

    def current_frame_number(self) -> int:
        """The first character's current frame, or 0 if the scene is empty."""
        if not self.characters:
            return 0
        return self.characters[0].current_frame

    def frame_count(self) -> int:
        """The first character's frame count, or zero for an empty scene."""
        if not self.characters:
            return 0
        return self.characters[0].num_frames

    # -------------------------------------------------------------- setup
    def initialize_gl(self) -> None:
        """Create the primitives this scene draws. Call once with a GL context current."""
        Primitives.create(Prims.SPHERE, "joint", _JOINT_RADIUS, 16)
        # radius baked in here; only ever scaled along Y (its length) at draw time
        Primitives.create(Prims.CYLINDER, "bone", _BONE_RADIUS, 1.0, 12, 1)
        Primitives.create(Prims.LINE_GRID, "ground", _GROUND_SIZE, _GROUND_SIZE, 40)
        self._trace_vao = VAOFactory.create_vao(VAOType.SIMPLE, gl.GL_LINE_STRIP)
        self._trace_data_dirty = True

    # -------------------------------------------------------------- drawing
    def draw(
        self,
        view: Mat4,
        project: Mat4,
        mouse_global_tx: Mat4,
        trace: bool = False,
    ) -> None:
        """Draw the scene or the joint position traces.

        Parameters
        ----------
            view : Mat4
                camera view matrix
            project : Mat4
                camera projection matrix
            mouse_global_tx : Mat4
                global scene transform
            trace : bool
                draw joint trajectories alongside the character when true
        """
        self._view = view
        self._project = project
        self._mouse_global_tx = mouse_global_tx

        self._draw_ground()
        if trace:
            self._draw_trace_lines()
        self._draw_characters()

    def _draw_characters(self) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        for character in self.characters:
            self._draw_joint(character, character.root, Mat4())

    def _draw_trace_lines(self) -> None:
        if self._trace_vao is None or not self._trace_ranges:
            return
        if self._trace_data_dirty:
            with self._trace_vao as vao:
                vao.set_data(
                    VertexData(self._trace_vertices, len(self._trace_vertices))
                )
                vao.set_vertex_attribute_pointer(0, 3, gl.GL_FLOAT, 3 * 4, 0)
            self._trace_data_dirty = False

        ShaderLib.use(DefaultShader.COLOUR)
        self._set_mvp_colour(Mat4())
        with self._trace_vao:
            for (first, count), colour in zip(
                self._trace_ranges, self._trace_colours, strict=True
            ):
                ShaderLib.set_uniform("Colour", *colour)
                gl.glDrawArrays(gl.GL_LINE_STRIP, first, count)

    def _draw_ground(self) -> None:
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform("Colour", 0.6, 0.6, 0.6, 1.0)
        self._set_mvp_colour(Mat4())
        Primitives.draw("ground")

    def _draw_joint(self, character: Bvh, joint: Joint, parent_world: Mat4) -> None:
        world = parent_world @ character.local_matrix(joint)
        self._set_mvp_diffuse(world)
        Primitives.draw("joint")
        for child in joint.children:
            self._draw_bone(world, child.offset)
            self._draw_joint(character, child, world)

    def _draw_bone(self, world: Mat4, offset: Vec3) -> None:
        length = float(offset.length())
        if length < 1e-6:
            return  # coincident joints: nothing meaningful to orient a bone along
        rotate = rotation_from_y(offset.normalized())
        midpoint = Mat4.translate(offset.x * 0.5, offset.y * 0.5, offset.z * 0.5)
        local = midpoint @ rotate @ Mat4.scale(1.0, length, 1.0)
        self._set_mvp_diffuse(world @ local)
        Primitives.draw("bone")

    def _set_mvp_diffuse(self, model: Mat4) -> None:
        ShaderLib.use(DefaultShader.DIFFUSE)
        mv = self._view @ self._mouse_global_tx @ model
        mvp = self._project @ mv
        normal_matrix = Mat3.from_mat4(mv).inverse().transposed()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", mv)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def _set_mvp_colour(self, model: Mat4) -> None:
        mvp = self._project @ self._view @ self._mouse_global_tx @ model
        ShaderLib.set_uniform("MVP", mvp)
