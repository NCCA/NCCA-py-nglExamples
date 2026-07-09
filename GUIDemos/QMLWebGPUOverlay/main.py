#!/usr/bin/env -S uv run --script
# GUIDemos/QMLWebGPUOverlay/main.py
import sys
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml widget types)
from panel_registry import PanelRegistry
from PyNGLScene import PyNGLScene
from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import QMouseEvent
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QMainWindow


class OverlayQuickWidget(QQuickWidget):
    """Transparent QQuickWidget that forwards clicks outside any panel to the scene beneath."""

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
        # parents[3]: qml -> ngl -> ncca -> the dir that CONTAINS ncca/.
        ncca_root = Path(ncca.ngl.qml.__file__).parents[3]
        self.engine().addImportPath(str(ncca_root))
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
    # Unlike the OpenGL QMLOverlayApp, the central widget here is a plain
    # QWidget (ncca.ngl.webgpu.WebGPUWidget blits an offscreen-rendered numpy
    # buffer via QPainter), so the top-level surface is NOT forced onto OpenGL.
    # That means the overlay QQuickWidget can use Qt's default RHI/Metal scene
    # graph backend directly - no QQuickWindow.setGraphicsApi(OpenGL) and no
    # QSurfaceFormat are needed. This is the crux of why WebGPU sidesteps the
    # QQuickFramebufferObject/RHI limitation documented in QMLFloatingWidgets.
    #
    # The native macOS QtQuickControls2 style doesn't support Frame background
    # customization (DraggablePanel.qml's dark background/border) and gives
    # ncca.ngl.qml's widgets a light, native appearance; Fusion supports both
    # consistently.
    QQuickStyle.setStyle("Fusion")
    app = QApplication(sys.argv)
    # main.qml's Settings element (import QtCore) persists panel positions and
    # the selected theme via QSettings, which derives its storage path from the
    # organization/application name - set them before any Settings is created.
    app.setOrganizationName("NCCA")
    app.setApplicationName("QMLWebGPUOverlay")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
