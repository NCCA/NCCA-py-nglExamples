"""QQuickFramebufferObject that renders a PyNGL teapot driven by ncca.ngl.qml models."""

from ncca.ngl import Mat3, Vec3, perspective
from ncca.ngl.opengl import DefaultShader, Primitives, ShaderLib
from PySide6.QtCore import Property, QObject, QSize
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickFramebufferObject

QML_IMPORT_NAME = "qmlfloatingwidgets"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class TeapotView(QQuickFramebufferObject):
    """Renders a teapot using matrices/colour pulled from ncca.ngl.qml models."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._transform_model: QObject | None = None
        self._look_at_model: QObject | None = None
        self._colour_model: QObject | None = None
        self._perspective_model: QObject | None = None

    def get_transform_model(self) -> QObject | None:
        return self._transform_model

    def set_transform_model(self, model: QObject) -> None:
        if self._transform_model is not None:
            self._transform_model.valueChanged.disconnect(self.update)
        self._transform_model = model
        model.valueChanged.connect(self.update)
        self.update()

    transformModel = Property(QObject, get_transform_model, set_transform_model)

    def get_look_at_model(self) -> QObject | None:
        return self._look_at_model

    def set_look_at_model(self, model: QObject) -> None:
        if self._look_at_model is not None:
            self._look_at_model.valueChanged.disconnect(self.update)
        self._look_at_model = model
        model.valueChanged.connect(self.update)
        self.update()

    lookAtModel = Property(QObject, get_look_at_model, set_look_at_model)

    def get_colour_model(self) -> QObject | None:
        return self._colour_model

    def set_colour_model(self, model: QObject) -> None:
        if self._colour_model is not None:
            self._colour_model.colourChanged.disconnect(self.update)
        self._colour_model = model
        model.colourChanged.connect(self.update)
        self.update()

    colourModel = Property(QObject, get_colour_model, set_colour_model)

    def get_perspective_model(self) -> QObject | None:
        return self._perspective_model

    def set_perspective_model(self, model: QObject) -> None:
        if self._perspective_model is not None:
            self._perspective_model.valueChanged.disconnect(self.update)
        self._perspective_model = model
        model.valueChanged.connect(self.update)
        self.update()

    perspectiveModel = Property(QObject, get_perspective_model, set_perspective_model)

    def createRenderer(self) -> "TeapotRenderer":
        return TeapotRenderer()


class TeapotRenderer(QQuickFramebufferObject.Renderer):
    """Owns the GL state and issues the teapot draw call each frame."""

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False
        self._mvp = None
        self._mv = None
        self._normal_matrix = None
        self._colour = Vec3(1.0, 1.0, 0.0)
        self._aspect = 1.0

    def createFramebufferObject(self, size: QSize) -> QOpenGLFramebufferObject:
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        fmt.setSamples(4)
        self._aspect = size.width() / max(size.height(), 1)
        return QOpenGLFramebufferObject(size, fmt)

    def synchronize(self, item: TeapotView) -> None:
        if (
            item.transformModel is None
            or item.lookAtModel is None
            or item.colourModel is None
            or item.perspectiveModel is None
        ):
            return
        model_matrix = item.transformModel.get_matrix()
        view_matrix = item.lookAtModel.get_matrix()
        # Aspect always comes from the framebuffer's own size (set in
        # createFramebufferObject), not the PerspectiveWidget panel, so the
        # view is never distorted by it.
        project = perspective(
            item.perspectiveModel.fov,
            self._aspect,
            item.perspectiveModel.near,
            item.perspectiveModel.far,
        )
        mv = view_matrix @ model_matrix
        self._mvp = project @ mv
        self._mv = mv
        normal_matrix = Mat3.from_mat4(mv)
        self._normal_matrix = normal_matrix.inverse().transposed()
        self._colour = item.colourModel.get_value()

    def render(self) -> None:
        import OpenGL.GL as gl

        if not self._initialized:
            ShaderLib.use(DefaultShader.DIFFUSE)
            ShaderLib.set_uniform("lightPos", 1.0, 1.0, 1.0)
            ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
            Primitives.load_default_primitives()
            self._initialized = True

        if self._mvp is None:
            return

        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        ShaderLib.use(DefaultShader.DIFFUSE)
        ShaderLib.set_uniform("MVP", self._mvp)
        ShaderLib.set_uniform("MV", self._mv)
        ShaderLib.set_uniform("normalMatrix", self._normal_matrix)
        ShaderLib.set_uniform(
            "Colour", self._colour.x, self._colour.y, self._colour.z, 1.0
        )
        Primitives.draw("teapot")
