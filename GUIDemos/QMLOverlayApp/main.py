#!/usr/bin/env -S uv run --script
# GUIDemos/QMLOverlayApp/main.py
import sys
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml widget types)
from panel_registry import PanelRegistry
from PyNGLScene import PyNGLScene
from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QMouseEvent, QSurfaceFormat
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QMainWindow


class OverlayQuickWidget(QQuickWidget):
    """Transparent QQuickWidget that forwards clicks outside any panel to the GL widget beneath."""

    def __init__(self, scene: PyNGLScene, registry: PanelRegistry, parent=None) -> None:
        super().__init__(parent)
        self._scene = scene
        self._registry = registry
        # ncca.ngl.qml's qmldir declares module "ncca.ngl.qml", so the import path
        # must be the directory that CONTAINS the ncca/ package root (four levels
        # up from ncca/ngl/qml/__init__.py), not the qml/ leaf directory itself.
        # Without this, QQuickWidget's QML engine cannot reliably resolve the
        # file-based components (TransformWidget.qml, RGBColourWidget.qml,
        # LookAtWidget.qml), causing intermittent "<Type> is not a type" errors.
        self.engine().addImportPath(
            str(Path(ncca.ngl.qml.__file__).parent.parent.parent.parent)
        )
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setClearColor(Qt.GlobalColor.transparent)
        self.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)

    def _forward_to_scene(self, event: QMouseEvent) -> None:
        forwarded = QMouseEvent(
            event.type(),
            self._scene.mapFromGlobal(event.globalPosition()),
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        QApplication.sendEvent(self._scene, forwarded)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mousePressEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mouseMoveEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._registry.hit_test(event.position()):
            super().mouseReleaseEvent(event)
        else:
            event.ignore()
            self._forward_to_scene(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self.scene = PyNGLScene()
        self.setCentralWidget(self.scene)

        self.registry = PanelRegistry()
        self.overlay = OverlayQuickWidget(self.scene, self.registry, self.scene)
        self.overlay.rootContext().setContextProperty("panelRegistry", self.registry)
        self.overlay.rootContext().setContextProperty("pyNGLScene", self.scene)
        self.overlay.setSource(
            QUrl.fromLocalFile(str(Path(__file__).parent / "main.qml"))
        )
        self.overlay.setGeometry(self.scene.rect())

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self.overlay.setGeometry(self.scene.rect())


def main() -> int:
    # The central widget is a QOpenGLWidget, which forces the top-level
    # window's native surface to composite via OpenGL. Qt Quick's default
    # RHI backend on macOS is Metal, which is incompatible with that
    # top-level surface: the overlay QQuickWidget fails to obtain a QRhi and
    # renders nothing (silently, no exception) unless the Qt Quick scene
    # graph is forced onto the OpenGL backend too, before QApplication is
    # constructed.
    QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)
    # The native macOS QtQuickControls2 style doesn't support Frame
    # background customization (DraggablePanel.qml's dark background/border)
    # and gives ncca.ngl.qml's widgets (spin boxes, combo boxes) a light,
    # native appearance that doesn't match. Fusion supports both consistently
    # — same fix already applied in the sibling QMLFloatingWidgets demo.
    QQuickStyle.setStyle("Fusion")
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setMajorVersion(4)
    fmt.setMinorVersion(1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
