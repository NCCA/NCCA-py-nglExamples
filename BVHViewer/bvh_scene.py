"""The OpenGL view of a BVH skeleton, or several of them.

A port of the drawing half of the C++ `Bvh`/`Scene` pair: `Bvh::drawBones`
walked the joint tree building bone transforms by hand from a fixed-Z-axis
cylinder; here it's the same recursive walk, but the maths is built for
PyNGL's cylinder, which runs along +Y (see `rotation_from_y` in `bvh.py`).

There is no `Wall` class here as there was in the C++ `Scene` -- the app
only ever draws one flat ground grid, so that generality wasn't earning its
keep and became a single hardcoded ground primitive instead.
"""

from bvh import Bvh, Joint, rotation_from_y
from ncca.ngl import Mat3, Mat4, Prims, Vec3
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib

_JOINT_RADIUS = 0.25
_BONE_RADIUS = 0.15
_GROUND_SIZE = 200.0


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

    # --------------------------------------------------------- characters
    def add_character(self, character: Bvh) -> None:
        """Add a character to the scene."""
        self.characters.append(character)

    def set_character(self, character: Bvh) -> None:
        """Replace the scene contents with one character."""
        self.characters = [character]

    def clear_characters(self) -> None:
        """Remove every character from the scene."""
        self.characters.clear()

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

    # -------------------------------------------------------------- drawing
    def draw(self, view: Mat4, project: Mat4, mouse_global_tx: Mat4) -> None:
        """Draw the ground grid and every character.

        Args:
            view: the camera's view matrix.
            project: the camera's projection matrix.
            mouse_global_tx: the mouse-driven global rotate/pan transform.
        """
        self._view = view
        self._project = project
        self._mouse_global_tx = mouse_global_tx

        self._draw_ground()
        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)
        ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        for character in self.characters:
            self._draw_joint(character, character.root, Mat4())

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
