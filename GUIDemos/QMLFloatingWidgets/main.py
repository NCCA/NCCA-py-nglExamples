#!/usr/bin/env -S uv run --script
# GUIDemos/QMLFloatingWidgets/main.py
import sys
from pathlib import Path

import ncca.ngl.qml  # noqa: F401  (import registers ncca.ngl.qml widget types)
import teapot_view  # noqa: F401  (import registers TeapotView)
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtQuickControls2 import QQuickStyle


def main() -> int:
    # QQuickFramebufferObject (used by TeapotView) only works with Qt Quick's
    # OpenGL scene graph backend. Qt 6's default backend is RHI/Metal on
    # macOS, under which QQuickFramebufferObject.Renderer is never
    # instantiated at all (createRenderer()/synchronize()/render() are
    # silently never called) — force OpenGL explicitly, before the
    # application is constructed.
    QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)
    QQuickStyle.setStyle("Fusion")
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    # ncca.ngl.qml's qmldir declares module "ncca.ngl.qml", so the import path
    # must be the directory that CONTAINS the ncca/ package root (four levels
    # up from ncca/ngl/qml/__init__.py), not the qml/ leaf directory itself.
    engine.addImportPath(str(Path(ncca.ngl.qml.__file__).parent.parent.parent.parent))
    engine.load(QUrl.fromLocalFile(str(Path(__file__).parent / "main.qml")))
    if not engine.rootObjects():
        return -1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
