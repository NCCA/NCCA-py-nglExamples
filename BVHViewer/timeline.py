"""Timeline and transport controls for the BVH viewer."""

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class FrameRangeSlider(QWidget):
    """A two-handle slider used to choose inclusive playback frames.

    Signals
    -------
    range_changed : Signal(int, int)
        emitted whilst either handle is dragged
    """

    range_changed = Signal(int, int)

    _HANDLE_RADIUS = 7.0
    _TRACK_MARGIN = 9.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("playbackRangeSlider")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(24)
        self.setToolTip("Drag the handles to set the playback range")
        self._minimum = 0
        self._maximum = 0
        self._start = 0
        self._end = 0
        self._active_handle: str | None = None

    def sizeHint(self) -> QSize:
        return QSize(240, 24)

    def set_limits(self, minimum: int, maximum: int) -> None:
        """Set the complete frame limits shown by the control.

        Parameters
        ----------
            minimum : int
                first available frame
            maximum : int
                last available frame
        """
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self.set_values(self._minimum, self._maximum)

    def set_values(self, start: int, end: int) -> None:
        """Set the selected playback range without emitting a signal.

        Parameters
        ----------
            start : int
                first selected frame
            end : int
                last selected frame
        """
        bounded_start = max(self._minimum, min(int(start), self._maximum))
        bounded_end = max(bounded_start, min(int(end), self._maximum))
        self._start = bounded_start
        self._end = bounded_end
        self.update()

    def values(self) -> tuple[int, int]:
        """Return the inclusive selected range."""
        return self._start, self._end

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        centre_y = self.height() / 2.0
        left = self._track_left()
        right = self._track_right()
        start_x = self._position_for_value(self._start)
        end_x = self._position_for_value(self._end)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#17191b"))
        painter.drawRoundedRect(left, centre_y - 2.5, right - left, 5.0, 2.0, 2.0)
        painter.setBrush(QColor("#cf7b2a"))
        painter.drawRoundedRect(
            start_x,
            centre_y - 3.0,
            max(1.0, end_x - start_x),
            6.0,
            2.0,
            2.0,
        )

        painter.setBrush(QColor("#e6a14e"))
        painter.setPen(QColor("#111111"))
        for position in (start_x, end_x):
            painter.drawRoundedRect(
                position - self._HANDLE_RADIUS,
                centre_y - self._HANDLE_RADIUS,
                self._HANDLE_RADIUS * 2.0,
                self._HANDLE_RADIUS * 2.0,
                2.0,
                2.0,
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position().x()
        start_distance = abs(position - self._position_for_value(self._start))
        end_distance = abs(position - self._position_for_value(self._end))
        self._active_handle = "start" if start_distance <= end_distance else "end"
        self._move_active_handle(position)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_handle is not None:
            self._move_active_handle(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._move_active_handle(event.position().x())
            self._active_handle = None

    def _move_active_handle(self, position: float) -> None:
        value = self._value_for_position(position)
        previous = self.values()
        if self._active_handle == "start":
            self._start = min(value, self._end)
        elif self._active_handle == "end":
            self._end = max(value, self._start)
        if self.values() != previous:
            self.update()
            self.range_changed.emit(self._start, self._end)

    def _track_left(self) -> float:
        return self._TRACK_MARGIN

    def _track_right(self) -> float:
        return max(self._track_left(), self.width() - self._TRACK_MARGIN)

    def _position_for_value(self, value: int) -> float:
        if self._maximum == self._minimum:
            return self._track_left()
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return self._track_left() + fraction * (
            self._track_right() - self._track_left()
        )

    def _value_for_position(self, position: float) -> int:
        width = self._track_right() - self._track_left()
        if width <= 0.0:
            return self._minimum
        fraction = (position - self._track_left()) / width
        fraction = max(0.0, min(fraction, 1.0))
        return round(self._minimum + fraction * (self._maximum - self._minimum))


class TimelineWidget(QWidget):
    """A frame scrubber, playback range and animation transport.

    Signals
    -------
    frame_requested : Signal(int)
        emitted when the user changes the current frame
    playback_range_changed : Signal(int, int)
        emitted when the user changes the inclusive playback range
    fps_changed : Signal(float)
        emitted when the user changes the playback frame rate
    first_requested, previous_requested, play_toggled, next_requested,
    last_requested : Signal()
        emitted by the matching transport button
    """

    frame_requested = Signal(int)
    playback_range_changed = Signal(int, int)
    fps_changed = Signal(float)
    first_requested = Signal()
    previous_requested = Signal()
    play_toggled = Signal()
    next_requested = Signal()
    last_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelinePanel")
        self._has_clip = False

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setObjectName("timelineSlider")
        self._slider.setTracking(True)
        self._slider.setToolTip("Scrub the animation")

        self._frame_spin = self._frame_field("currentFrame", "Current frame")

        self._range_label = QLabel("0 - 0")
        self._range_label.setObjectName("frameRange")
        self._range_label.setMinimumWidth(80)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setObjectName("playbackFps")
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setDecimals(2)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._fps_spin.setFixedWidth(92)
        self._fps_spin.setToolTip("Playback frames per second")

        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(8, 4, 8, 0)
        timeline_row.setSpacing(8)
        timeline_row.addWidget(QLabel("Frame"))
        timeline_row.addWidget(self._slider, 1)
        timeline_row.addWidget(self._frame_spin)
        timeline_row.addWidget(self._range_label)
        timeline_row.addWidget(self._fps_spin)

        self._range_slider = FrameRangeSlider()
        self._range_start_spin = self._frame_field(
            "playbackStart", "Playback start frame"
        )
        self._range_end_spin = self._frame_field("playbackEnd", "Playback end frame")

        range_row = QHBoxLayout()
        range_row.setContentsMargins(8, 0, 8, 0)
        range_row.setSpacing(6)
        range_row.addWidget(QLabel("Playback"))
        range_row.addWidget(QLabel("Start"))
        range_row.addWidget(self._range_start_spin)
        range_row.addWidget(self._range_slider, 1)
        range_row.addWidget(QLabel("End"))
        range_row.addWidget(self._range_end_spin)

        self._first_button = self._transport_button(
            "firstFrame", QStyle.StandardPixmap.SP_MediaSkipBackward, "First frame"
        )
        self._previous_button = self._transport_button(
            "previousFrame",
            QStyle.StandardPixmap.SP_MediaSeekBackward,
            "Previous frame",
        )
        self._play_button = self._transport_button(
            "playPause", QStyle.StandardPixmap.SP_MediaPlay, "Play"
        )
        self._play_button.setCheckable(True)
        self._next_button = self._transport_button(
            "nextFrame", QStyle.StandardPixmap.SP_MediaSeekForward, "Next frame"
        )
        self._last_button = self._transport_button(
            "lastFrame", QStyle.StandardPixmap.SP_MediaSkipForward, "Last frame"
        )

        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(8, 0, 8, 6)
        transport_row.setSpacing(3)
        transport_row.addStretch(1)
        transport_row.addWidget(self._first_button)
        transport_row.addWidget(self._previous_button)
        transport_row.addWidget(self._play_button)
        transport_row.addWidget(self._next_button)
        transport_row.addWidget(self._last_button)
        transport_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(timeline_row)
        layout.addLayout(range_row)
        layout.addLayout(transport_row)

        self._slider.valueChanged.connect(self._slider_changed)
        self._frame_spin.valueChanged.connect(self._spin_changed)
        self._range_start_spin.valueChanged.connect(self._range_start_changed)
        self._range_end_spin.valueChanged.connect(self._range_end_changed)
        self._range_slider.range_changed.connect(self._dragged_range_changed)
        self._fps_spin.valueChanged.connect(self.fps_changed)
        self._first_button.clicked.connect(self.first_requested)
        self._previous_button.clicked.connect(self.previous_requested)
        self._play_button.clicked.connect(self.play_toggled)
        self._next_button.clicked.connect(self.next_requested)
        self._last_button.clicked.connect(self.last_requested)
        self.set_clip(0, 0.0)

    def _frame_field(self, name: str, tooltip: str) -> QSpinBox:
        field = QSpinBox()
        field.setObjectName(name)
        field.setAlignment(Qt.AlignmentFlag.AlignRight)
        field.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        field.setFixedWidth(66)
        field.setToolTip(tooltip)
        return field

    def _transport_button(
        self, name: str, icon: QStyle.StandardPixmap, tooltip: str
    ) -> QPushButton:
        button = QPushButton()
        button.setObjectName(name)
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setFixedSize(30, 26)
        return button

    def set_clip(self, frame_count: int, frame_time: float) -> None:
        """Set the available frames, playback range and source frame rate.

        Parameters
        ----------
            frame_count : int
                number of frames in the loaded clip
            frame_time : float
                duration of one source frame in seconds
        """
        last_frame = max(0, int(frame_count) - 1)
        self._has_clip = frame_count > 0
        self._slider.setRange(0, last_frame)
        self._frame_spin.setRange(0, last_frame)
        self._range_start_spin.setRange(0, last_frame)
        self._range_end_spin.setRange(0, last_frame)
        self._range_slider.set_limits(0, last_frame)
        self._range_label.setText(f"0 - {last_frame}")
        self.set_playback_range(0, last_frame)
        if frame_time > 0.0:
            self.set_playback_fps(1.0 / frame_time)
        enabled = frame_count > 0
        for control in (
            self._slider,
            self._frame_spin,
            self._range_slider,
            self._range_start_spin,
            self._range_end_spin,
            self._fps_spin,
            self._first_button,
            self._previous_button,
            self._play_button,
            self._next_button,
            self._last_button,
        ):
            control.setEnabled(enabled)
        self.set_frame(0)

    def set_frame(self, frame: int) -> None:
        """Update the displayed frame without emitting a seek request.

        Parameters
        ----------
            frame : int
                frame currently displayed by the viewport
        """
        bounded = max(self._slider.minimum(), min(int(frame), self._slider.maximum()))
        with QSignalBlocker(self._slider), QSignalBlocker(self._frame_spin):
            self._slider.setValue(bounded)
            self._frame_spin.setValue(bounded)

    def scrub_to(self, frame: int) -> None:
        """Move the scrubber as if the frame was selected by the user.

        Parameters
        ----------
            frame : int
                requested frame
        """
        self._slider.setValue(frame)

    def current_frame(self) -> int:
        """Return the frame shown in the timeline."""
        return self._slider.value()

    def frame_range(self) -> tuple[int, int]:
        """Return the inclusive clip limits shown by the timeline."""
        return self._slider.minimum(), self._slider.maximum()

    def range_text(self) -> str:
        """Return the visible clip-range text."""
        return self._range_label.text()

    def set_playback_range(self, start: int, end: int) -> None:
        """Set the inclusive playback range without emitting a signal.

        Parameters
        ----------
            start : int
                first playback frame
            end : int
                last playback frame
        """
        minimum, maximum = self.frame_range()
        bounded_start = max(minimum, min(int(start), maximum))
        bounded_end = max(bounded_start, min(int(end), maximum))
        self._apply_playback_range(bounded_start, bounded_end, emit=False)

    def playback_range(self) -> tuple[int, int]:
        """Return the inclusive playback range."""
        return self._range_slider.values()

    def set_playback_fps(self, fps: float) -> None:
        """Set playback speed without emitting a change signal.

        Parameters
        ----------
            fps : float
                playback frames per second
        """
        with QSignalBlocker(self._fps_spin):
            self._fps_spin.setValue(fps)

    def playback_fps(self) -> float:
        """Return the selected playback frame rate."""
        return self._fps_spin.value()

    def rate_text(self) -> str:
        """Return the visible playback-rate text."""
        if not self._has_clip:
            return "-- fps"
        return f"{self.playback_fps():.2f} fps"

    def set_playing(self, playing: bool) -> None:
        """Display either the playing or paused transport state.

        Parameters
        ----------
            playing : bool
                true when playback is running
        """
        with QSignalBlocker(self._play_button):
            self._play_button.setChecked(playing)
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self._play_button.setIcon(self.style().standardIcon(icon))
        self._play_button.setToolTip("Pause" if playing else "Play")

    def is_playing(self) -> bool:
        """Return whether the transport displays its playing state."""
        return self._play_button.isChecked()

    def play_tooltip(self) -> str:
        """Return the transport tooltip used for its current state."""
        return self._play_button.toolTip()

    def _slider_changed(self, frame: int) -> None:
        with QSignalBlocker(self._frame_spin):
            self._frame_spin.setValue(frame)
        self.frame_requested.emit(frame)

    def _spin_changed(self, frame: int) -> None:
        with QSignalBlocker(self._slider):
            self._slider.setValue(frame)
        self.frame_requested.emit(frame)

    def _range_start_changed(self, start: int) -> None:
        end = max(start, self._range_end_spin.value())
        self._apply_playback_range(start, end, emit=True)

    def _range_end_changed(self, end: int) -> None:
        start = min(self._range_start_spin.value(), end)
        self._apply_playback_range(start, end, emit=True)

    def _dragged_range_changed(self, start: int, end: int) -> None:
        self._apply_playback_range(start, end, emit=True)

    def _apply_playback_range(self, start: int, end: int, emit: bool) -> None:
        with (
            QSignalBlocker(self._range_start_spin),
            QSignalBlocker(self._range_end_spin),
        ):
            self._range_start_spin.setValue(start)
            self._range_end_spin.setValue(end)
        self._range_slider.set_values(start, end)
        if emit:
            self.playback_range_changed.emit(start, end)
