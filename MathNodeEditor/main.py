#!/usr/bin/env -S uv run --script
"""Run the PyNGL maths node editor."""

from __future__ import annotations

import argparse
import sys
import traceback

from ncca.ngl import logger
from node_editor import MathNodeWindow
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication


class DebugApplication(QApplication):
    """A QApplication that reports exceptions raised inside Qt event handlers.

    Qt's event loop normally swallows exceptions raised in ``paintGL``,
    ``mouseMoveEvent`` and similar handlers, so the app just freezes or
    crashes with no traceback. Overriding ``notify`` lets us print one
    before re-raising.
    """

    def __init__(self, argv: list[str]) -> None:
        """Create the application and log that debug mode is active."""
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        """Forward to Qt's normal dispatch, printing a traceback on failure."""
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def _parse_args() -> argparse.Namespace:
    """Parse the --debug and --smoketest command-line flags."""
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
    return parser.parse_args()


def _configure_surface_format() -> None:
    """Set the default OpenGL surface format for every window this app creates.

    Must run before any QApplication or QOpenGLWindow is constructed.
    """
    surface_format = QSurfaceFormat()
    surface_format.setSamples(4)  # 4x multisampling for anti-aliasing
    # OpenGL 4.1 Core is the highest version macOS supports.
    surface_format.setMajorVersion(4)
    surface_format.setMinorVersion(1)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setDepthBufferSize(24)
    # Embedded previews and pop-out windows need to share ShaderLib's
    # OpenGL program ids.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    QSurfaceFormat.setDefaultFormat(surface_format)


def main() -> int:
    """Show the editor window and run the event loop; return the exit code.

    Reuses an existing QApplication if one was already created (by the
    ``__main__`` block below, or by a caller such as a test), rather than
    always constructing a plain one, so callers can still choose
    ``DebugApplication`` first.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("NCCA")
    app.setApplicationName("MathNodeEditor")
    window = MathNodeWindow(load_example=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    args = _parse_args()
    _configure_surface_format()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(main())
