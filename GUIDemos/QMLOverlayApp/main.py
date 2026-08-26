#!/usr/bin/env -S uv run --script
# GUIDemos/QMLOverlayApp/main.py
import argparse
import sys
import traceback
from pathlib import Path

import ncca.ngl.qml
from panel_registry import PanelRegistry
from PyNGLScene import PyNGLScene
from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import QKeyEvent, QMouseEvent, QSurfaceFormat
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QMainWindow


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


class OverlayQuickWidget(QQuickWidget):
    """Transparent QQuickWidget that forwards clicks outside any panel to the GL widget beneath."""

    def __init__(self, scene: PyNGLScene, registry: PanelRegistry, parent=None) -> None:
        super().__init__(parent)
        self._scene = scene
        self._registry = registry
        # Let the QML engine resolve `import ncca.ngl.qml 1.0` in main.qml.
        ncca.ngl.qml.add_import_path(self.engine())
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Neither the QML overlay nor the scene beneath it takes Escape, so it
        # arrives here by the usual widget parent chain and this is the only
        # place the demo can be quit from the keyboard.
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self.overlay.setGeometry(self.scene.rect())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

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
    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)
    # main.qml's Settings element (import QtCore) persists panel positions and
    # the selected theme via QSettings. QSettings derives its storage path from
    # the organization/application name, so set them before any Settings is
    # created or it falls back to an unnamed store and warns.
    app.setOrganizationName("NCCA")
    app.setApplicationName("QMLOverlayApp")
    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setMajorVersion(4)
    fmt.setMinorVersion(1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
