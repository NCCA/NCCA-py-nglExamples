"""WebGPU teapot scene driven by pre-composed matrices from ncca.ngl.qml models.

This is the WebGPU counterpart of QMLOverlayApp's OpenGL PyNGLScene. It exposes
the same slots (set_model_matrix / set_view_matrix / set_colour) so the shared
main.qml drives it unchanged, but renders offscreen into a numpy buffer via
ncca.ngl.webgpu.WebGPUWidget instead of a QOpenGLWidget. That is what lets the
floating-panel + 3D-viewport idea work under Qt 6's RHI/Metal scene graph, where
the original QMLFloatingWidgets' QQuickFramebufferObject could not obtain a GL
context.
"""

from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Slot
from TeapotPipeline import TeapotPipeline
from wgpu.utils import get_default_device


class PyNGLScene(WebGPUWidget):
    """Offscreen WebGPU teapot, updated on demand from the QML panels."""

    def __init__(self) -> None:
        super().__init__()
        self._model_matrix = Mat4()
        # A sensible default camera so the teapot is visible before the Look At
        # panel first emits; the panel overwrites this on Component.onCompleted.
        self._view_matrix = look_at(Vec3(0, 2, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self._colour = Vec3(1.0, 1.0, 0.0)
        self.light_pos = Vec3(1.0, 1.0, 1.0)
        # Aspect always comes from the surface's own size (resizeWebGPU), not
        # the PerspectiveWidget panel, so the view is never distorted by it.
        self._fov = 45.0
        self._near = 0.1
        self._far = 350.0

        self.project = perspective(
            self._fov,
            self.width() / max(self.height(), 1),
            self._near,
            self._far,
            PerspMode.WebGPU,
        )
        self.device = get_default_device()
        self._create_render_buffer()
        width = int(self.ratio * self.width())
        height = int(self.ratio * self.height())
        self.pipeline = TeapotPipeline(self.device, self.light_pos, width, height)
        self.update()

    @Slot(Mat4)
    def set_model_matrix(self, matrix: Mat4) -> None:
        self._model_matrix = matrix
        self.update()

    @Slot(Mat4)
    def set_view_matrix(self, matrix: Mat4) -> None:
        self._view_matrix = matrix
        self.update()

    @Slot(float, float, float)
    def set_colour(self, r: float, g: float, b: float) -> None:
        self._colour = Vec3(r, g, b)
        self.update()

    @Slot(float, float, float)
    def set_perspective_params(self, fov: float, near: float, far: float) -> None:
        self._fov = fov
        self._near = near
        self._far = far
        self.project = perspective(
            self._fov,
            self.width() / max(self.height(), 1),
            self._near,
            self._far,
            PerspMode.WebGPU,
        )
        self.update()

    def paintWebGPU(self) -> None:
        self.pipeline.update_uniform_buffers(
            self._model_matrix, self._view_matrix, self.project, self._colour
        )
        self.pipeline.paint(
            self.colour_buffer_texture_view,
            self.multisample_texture_view,
            self.depth_buffer_view,
        )
        # Copy the resolved colour target back into the numpy frame buffer the
        # widget blits with QPainter. Since ncca-ngl Version1.0 this is the
        # subclass's job, not paintEvent's - without it the render pass fills
        # its textures and nothing ever reaches the screen.
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            self._fov,
            width / max(height, 1),
            self._near,
            self._far,
            PerspMode.WebGPU,
        )
        # The base resizeEvent recreates the render buffers around this call;
        # keep the pipeline's viewport in step with the new size.
        if hasattr(self, "pipeline"):
            self.pipeline.width = width
            self.pipeline.height = height
        self.update()
