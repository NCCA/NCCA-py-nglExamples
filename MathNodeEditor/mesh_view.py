"""OpenGL rendering shared by the Mesh Viewer node's embedded and popup views.

Two separate GL surfaces can be open for one Mesh Viewer node at once (the
small embedded preview and the pop-out window), and they are not guaranteed
to share a GL context with each other. Each view therefore keeps its own
``Obj``/VAO, rebuilt from the same plain-data ``MeshViewerInputs`` whenever
:class:`MeshRenderState` reports a new version -- only the source data is
shared, never GL objects.
"""

from __future__ import annotations

import OpenGL.GL as gl
from math_graph import MeshViewerInputs, obj_from_arrays
from ncca.ngl import Mat3, Mat4, Obj, Vec3, look_at, perspective
from ncca.ngl.opengl import DefaultShader, PySideEventHandlingMixin, ShaderLib
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget

SHADING_SOLID = "Solid Colour"
SHADING_DIFFUSE = "Diffuse"
SHADING_MODES: tuple[str, ...] = (SHADING_SOLID, SHADING_DIFFUSE)

DEFAULT_COLOUR: tuple[float, float, float, float] = (0.6, 0.62, 0.66, 1.0)


class MeshRenderState:
    """Data and settings shared by every view of one Mesh Viewer node.

    Holds plain data only (no GL objects), so it is safe to read from
    multiple GL contexts.
    """

    def __init__(self) -> None:
        """Create an empty, unrendered mesh state."""
        self.mesh_inputs: MeshViewerInputs | None = None
        self.version = 0
        self.shading_mode: str = SHADING_SOLID
        self.wireframe: bool = False
        self.colour: tuple[float, float, float, float] = DEFAULT_COLOUR

    def set_mesh(self, mesh_inputs: MeshViewerInputs | None) -> None:
        """Replace the mesh data and mark every view's VAO as stale."""
        self.mesh_inputs = mesh_inputs
        if mesh_inputs is not None and mesh_inputs.colour is not None:
            self.colour = (
                mesh_inputs.colour.x,
                mesh_inputs.colour.y,
                mesh_inputs.colour.z,
                mesh_inputs.colour.w,
            )
        else:
            self.colour = DEFAULT_COLOUR
        self.version += 1

    def set_shading_mode(self, shading_mode: str) -> None:
        """Change the shading mode and mark every view's next paint dirty."""
        self.shading_mode = shading_mode
        self.version += 1

    def set_wireframe(self, wireframe: bool) -> None:
        """Toggle wireframe polygon mode."""
        self.wireframe = wireframe
        self.version += 1


class MeshRenderMixin:
    """Shared camera/paint body for the embedded widget and the popup window.

    A class using this mixin must also inherit ``PySideEventHandlingMixin``
    (before the Qt GL base class, per its own MRO requirement) and a GL
    surface (``QOpenGLWidget`` or ``QOpenGLWindow``).
    """

    def _init_mesh_render(self, state: MeshRenderState) -> None:
        """Set up the camera and per-view VAO cache."""
        self.state = state
        self.view: Mat4 = look_at(Vec3(0, 1, 6), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project: Mat4 = Mat4()
        self.window_width = 1
        self.window_height = 1
        self._built_version = -1
        self._obj: Obj | None = None

    def _gl_setup(self) -> None:
        """Set the GL state every Mesh Viewer view starts with."""
        gl.glClearColor(0.15, 0.16, 0.19, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

    def _rebuild_vao_if_needed(self) -> None:
        """Rebuild this view's own Obj/VAO if the shared mesh data changed."""
        if self._built_version == self.state.version:
            return
        self._built_version = self.state.version
        self._obj = None
        inputs = self.state.mesh_inputs
        if inputs is None or not inputs.vertices.values or not inputs.faces.triangles:
            return
        obj = obj_from_arrays(inputs.vertices, inputs.faces, inputs.uvs, inputs.normals)
        obj.create_vao()
        self._obj = obj

    def _draw_mesh(self) -> None:
        """Render the current mesh with the node's shading/wireframe settings."""
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self._rebuild_vao_if_needed()
        if self._obj is None:
            return

        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        mouse_global_tx = rot_y @ rot_x
        mouse_global_tx[3, 0] = self.model_position.x
        mouse_global_tx[3, 1] = self.model_position.y
        mouse_global_tx[3, 2] = self.model_position.z

        model_view = self.view @ mouse_global_tx
        mvp = self.project @ model_view

        use_diffuse = self.state.shading_mode == SHADING_DIFFUSE and self._obj.normals
        shader = DefaultShader.DIFFUSE if use_diffuse else DefaultShader.COLOUR
        ShaderLib.use(shader)
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("Colour", *self.state.colour)
        if use_diffuse:
            normal_matrix = Mat3.from_mat4(model_view).inverse().transposed()
            ShaderLib.set_uniform("MV", model_view)
            ShaderLib.set_uniform("normalMatrix", normal_matrix)
            ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)

        gl.glPolygonMode(
            gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.state.wireframe else gl.GL_FILL
        )
        self._obj.draw()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def _resize_mesh_view(self, width: int, height: int) -> None:
        """Rebuild the projection matrix for a new viewport size."""
        pixel_ratio = self.devicePixelRatio()
        self.window_width = max(1, int(width * pixel_ratio))
        self.window_height = max(1, int(height * pixel_ratio))
        aspect = self.window_width / self.window_height
        self.project = perspective(45.0, aspect, 0.05, 500.0)


class MeshPreviewWidget(PySideEventHandlingMixin, QOpenGLWidget, MeshRenderMixin):
    """Small live preview embedded directly in the Mesh Viewer node body."""

    def __init__(self, state: MeshRenderState, parent: object = None) -> None:
        """Create the embedded viewport and wire up mouse-driven orbiting."""
        super().__init__(parent)
        self.setup_event_handling(initial_position=Vec3(0, 0, -2))
        self._init_mesh_render(state)
        self.setMinimumSize(220, 160)

    def initializeGL(self) -> None:
        """Set up GL state once the embedded context is ready."""
        self._gl_setup()

    def paintGL(self) -> None:
        """Redraw the embedded preview."""
        self._draw_mesh()

    def resizeGL(self, width: int, height: int) -> None:
        """Rebuild the projection matrix when the embedded viewport resizes."""
        self._resize_mesh_view(width, height)


class MeshPopupWindow(PySideEventHandlingMixin, QOpenGLWindow, MeshRenderMixin):
    """Standalone window opened by a Mesh Viewer node's Pop Out button."""

    def __init__(self, state: MeshRenderState, title: str) -> None:
        """Create the popup window and wire up mouse-driven orbiting."""
        super().__init__()
        self.setup_event_handling(initial_position=Vec3(0, 0, -2))
        self._init_mesh_render(state)
        self.setTitle(title)
        self.resize(640, 480)

    def initializeGL(self) -> None:
        """Set up GL state once the popup's context is ready."""
        self.makeCurrent()
        self._gl_setup()

    def paintGL(self) -> None:
        """Redraw the popup window."""
        self.makeCurrent()
        self._draw_mesh()

    def resizeGL(self, width: int, height: int) -> None:
        """Rebuild the projection matrix when the popup window resizes."""
        self._resize_mesh_view(width, height)
