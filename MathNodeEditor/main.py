#!/usr/bin/env -S uv run --script
"""Run the PyNGL maths node editor."""

from __future__ import annotations

import sys

from node_editor import MathNodeWindow
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication


class MainWindow(MathNodeWindow):
    """Main application window for the maths node editor."""


def main() -> int:
    """Run the maths node editor application."""
    application = QApplication.instance()
    if application is None:
        surface_format = QSurfaceFormat()
        surface_format.setSamples(4)
        surface_format.setMajorVersion(4)
        surface_format.setMinorVersion(1)
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface_format.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(surface_format)
        # Embedded previews and pop-out windows need to share ShaderLib's
        # OpenGL program ids.
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        application = QApplication(sys.argv)
    application.setOrganizationName("NCCA")
    application.setApplicationName("MathNodeEditor")
    window = MainWindow(load_example=True)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
