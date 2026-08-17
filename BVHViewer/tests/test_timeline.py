"""Tests for the BVH viewer timeline controls."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import main as bvh_viewer  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QSignalSpy, QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDoubleSpinBox,
    QMainWindow,
    QSpinBox,
)

TEST_BVH = Path(__file__).parent.parent / "bvh" / "test.bvh"


@pytest.fixture(scope="module")
def application() -> QApplication:
    """Return the application required for constructing Qt widgets."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_timeline_uses_the_loaded_clip_range(application: QApplication) -> None:
    timeline = bvh_viewer.TimelineWidget()

    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)

    assert timeline.frame_range() == (0, 119)
    assert timeline.range_text() == "0 - 119"
    assert timeline.rate_text() == "30.00 fps"
    assert timeline.playback_range() == (0, 119)


def test_setting_frame_updates_the_readout_without_requesting_a_seek(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    spy = QSignalSpy(timeline.frame_requested)

    timeline.set_frame(42)

    assert timeline.current_frame() == 42
    assert spy.count() == 0


def test_scrubbing_requests_the_selected_frame(application: QApplication) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    spy = QSignalSpy(timeline.frame_requested)

    timeline.scrub_to(73)

    assert timeline.current_frame() == 73
    assert list(spy.at(0)) == [73]


def test_entering_a_frame_requests_the_selected_frame(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    frame_field = timeline.findChild(QSpinBox, "currentFrame")
    spy = QSignalSpy(timeline.frame_requested)

    frame_field.setValue(37)

    assert timeline.current_frame() == 37
    assert list(spy.at(0)) == [37]


def test_play_state_updates_the_transport_button(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()

    timeline.set_playing(True)
    assert timeline.is_playing()
    assert timeline.play_tooltip() == "Pause"

    timeline.set_playing(False)
    assert not timeline.is_playing()
    assert timeline.play_tooltip() == "Play"


def test_numeric_fields_change_the_playback_range(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    start_field = timeline.findChild(QSpinBox, "playbackStart")
    end_field = timeline.findChild(QSpinBox, "playbackEnd")
    spy = QSignalSpy(timeline.playback_range_changed)

    start_field.setValue(24)
    end_field.setValue(80)

    assert timeline.playback_range() == (24, 80)
    assert list(spy.at(spy.count() - 1)) == [24, 80]


def test_range_fields_sit_on_the_matching_sides_of_the_slider(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    timeline.resize(720, 120)
    timeline.show()
    application.processEvents()
    start_field = timeline.findChild(QSpinBox, "playbackStart")
    range_slider = timeline.findChild(
        bvh_viewer.FrameRangeSlider, "playbackRangeSlider"
    )
    end_field = timeline.findChild(QSpinBox, "playbackEnd")

    assert start_field.x() < range_slider.x() < end_field.x()


def test_dragging_a_range_handle_changes_the_playback_range(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=100, frame_time=1.0 / 24.0)
    range_slider = timeline.findChild(
        bvh_viewer.FrameRangeSlider, "playbackRangeSlider"
    )
    range_slider.resize(400, 24)
    range_slider.show()
    application.processEvents()
    spy = QSignalSpy(timeline.playback_range_changed)

    QTest.mousePress(
        range_slider,
        Qt.MouseButton.LeftButton,
        pos=QPoint(8, range_slider.height() // 2),
    )
    QTest.mouseMove(range_slider, QPoint(104, range_slider.height() // 2))
    QTest.mouseRelease(
        range_slider,
        Qt.MouseButton.LeftButton,
        pos=QPoint(104, range_slider.height() // 2),
    )

    start, end = timeline.playback_range()
    assert start == pytest.approx(25, abs=1)
    assert end == 99
    assert spy.count() >= 1


def test_editing_fps_emits_the_new_playback_rate(
    application: QApplication,
) -> None:
    timeline = bvh_viewer.TimelineWidget()
    timeline.set_clip(frame_count=120, frame_time=1.0 / 30.0)
    fps_field = timeline.findChild(QDoubleSpinBox, "playbackFps")
    spy = QSignalSpy(timeline.fps_changed)

    fps_field.setValue(48.0)

    assert timeline.playback_fps() == pytest.approx(48.0)
    assert timeline.rate_text() == "48.00 fps"
    assert float(spy.at(0)[0]) == pytest.approx(48.0)


def test_application_window_is_a_qt_main_window() -> None:
    assert issubclass(bvh_viewer.MainWindow, QMainWindow)


def test_application_window_loads_the_initial_clip(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)

    assert window.current_path() == TEST_BVH.resolve()
    assert window.viewport.scene.frame_count() == 135
    assert window.timeline.frame_range() == (0, 134)
    assert "test.bvh" in window.windowTitle()


def test_file_menu_provides_open_and_quit_actions(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)

    assert window.open_action.shortcut().matches(QKeySequence.StandardKey.Open)
    assert window.quit_action.shortcut().matches(QKeySequence.StandardKey.Quit)
    assert window.open_action in window.file_menu.actions()
    assert window.quit_action in window.file_menu.actions()


def test_transport_can_jump_to_the_end_of_the_loaded_clip(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)

    window.go_to_last_frame()

    assert window.viewport.scene.current_frame_number() == 134
    assert window.timeline.current_frame() == 134


def test_transport_uses_the_selected_playback_range(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)
    window.timeline.set_playback_range(10, 20)

    window.go_to_last_frame()
    assert window.viewport.scene.current_frame_number() == 20

    window.go_to_first_frame()
    assert window.viewport.scene.current_frame_number() == 10


def test_playback_loops_inside_the_selected_range(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)
    window.timeline.set_playback_range(10, 20)
    window.viewport.scene.seek(20)

    window.advance_playback()

    assert window.viewport.scene.current_frame_number() == 10


def test_editing_fps_changes_the_playback_timer_interval(
    application: QApplication,
) -> None:
    window = bvh_viewer.MainWindow(TEST_BVH)
    fps_field = window.timeline.findChild(QDoubleSpinBox, "playbackFps")

    fps_field.setValue(50.0)

    assert window.playback_fps() == pytest.approx(50.0)
    assert window.playback_interval_ms() == 20
