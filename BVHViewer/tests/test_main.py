"""Checks for the files and application shell used by the BVH viewer."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402
from bvh_scene import BvhScene  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def application() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class WidgetViewport(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.scene = BvhScene()
        self.trace = False
        self.four_view = False

    def set_four_view(self, enabled: bool) -> None:
        self.four_view = enabled

    def toggle_maximized_pane(self, x: float, y: float) -> None:
        del x, y


def test_default_clip_is_shipped_with_the_viewer() -> None:
    assert bvh_viewer.DEFAULT_BVH.parent == Path(__file__).parent.parent / "bvh"
    assert bvh_viewer.DEFAULT_BVH.is_file()


def test_main_window_accepts_a_widget_viewport(application: QApplication) -> None:
    viewport = WidgetViewport()

    window = bvh_viewer.MainWindow(
        Path(__file__).parent.parent / "bvh" / "test.bvh",
        viewport=viewport,
    )

    assert window.viewport is viewport
    assert window.centralWidget().layout().itemAt(0).widget() is viewport
