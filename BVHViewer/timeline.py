"""Timeline and transport controls for the BVH viewer."""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class TimelineWidget(QWidget):
    """A frame scrubber and the usual animation transport controls.

    Signals
    -------
    frame_requested : Signal(int)
        emitted when the user changes the current frame
    first_requested, previous_requested, play_toggled, next_requested,
    last_requested : Signal()
        emitted by the matching transport button
    """

    frame_requested = Signal(int)
    first_requested = Signal()
    previous_requested = Signal()
    play_toggled = Signal()
    next_requested = Signal()
    last_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelinePanel")
        self._frame_time = 0.0

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setObjectName("timelineSlider")
        self._slider.setTracking(True)
        self._slider.setToolTip("Scrub the animation")

        self._frame_spin = QSpinBox()
        self._frame_spin.setObjectName("currentFrame")
        self._frame_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._frame_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._frame_spin.setFixedWidth(72)
        self._frame_spin.setToolTip("Current frame")

        self._range_label = QLabel("0 - 0")
        self._range_label.setObjectName("frameRange")
        self._range_label.setMinimumWidth(80)

        self._rate_label = QLabel("-- fps")
        self._rate_label.setObjectName("frameRate")
        self._rate_label.setMinimumWidth(72)
        self._rate_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(8, 4, 8, 0)
        timeline_row.setSpacing(8)
        timeline_row.addWidget(QLabel("Frame"))
        timeline_row.addWidget(self._slider, 1)
        timeline_row.addWidget(self._frame_spin)
        timeline_row.addWidget(self._range_label)
        timeline_row.addWidget(self._rate_label)

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
        transport_row.setContentsMargins(8, 2, 8, 6)
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
        layout.addLayout(transport_row)

        self._slider.valueChanged.connect(self._slider_changed)
        self._frame_spin.valueChanged.connect(self._spin_changed)
        self._first_button.clicked.connect(self.first_requested)
        self._previous_button.clicked.connect(self.previous_requested)
        self._play_button.clicked.connect(self.play_toggled)
        self._next_button.clicked.connect(self.next_requested)
        self._last_button.clicked.connect(self.last_requested)
        self.set_clip(0, 0.0)

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
        """Set the available frame range and playback rate.

        Parameters
        ----------
            frame_count : int
                number of frames in the loaded clip
            frame_time : float
                duration of one frame in seconds
        """
        last_frame = max(0, int(frame_count) - 1)
        self._frame_time = max(0.0, float(frame_time))
        self._slider.setRange(0, last_frame)
        self._frame_spin.setRange(0, last_frame)
        self._range_label.setText(f"0 - {last_frame}")
        self._rate_label.setText(
            f"{1.0 / self._frame_time:.2f} fps" if self._frame_time else "-- fps"
        )
        enabled = frame_count > 0
        for control in (
            self._slider,
            self._frame_spin,
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
        """Return the inclusive range shown by the timeline."""
        return self._slider.minimum(), self._slider.maximum()

    def range_text(self) -> str:
        """Return the visible clip range text."""
        return self._range_label.text()

    def rate_text(self) -> str:
        """Return the visible playback-rate text."""
        return self._rate_label.text()

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
