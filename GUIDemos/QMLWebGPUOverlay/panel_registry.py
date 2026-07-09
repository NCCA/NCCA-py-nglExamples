"""QML-exposed registry of floating panel screen rects, used for click-through hit testing."""

from PySide6.QtCore import QObject, QPointF, QRectF, Slot
from PySide6.QtQml import QmlElement

QML_IMPORT_NAME = "qmlwebgpuoverlay"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class PanelRegistry(QObject):
    """Tracks each floating panel's current screen rect for hit testing."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rects: dict[str, QRectF] = {}
        # Count of currently-open popups (e.g. ComboBox dropdowns). A popup's
        # list is reparented into the window overlay and opens OUTSIDE any
        # panel rect, so a click on a popup item would otherwise fail hit_test
        # and be forwarded to the WebGPU scene instead of the QML overlay -
        # the popup would never see the click and so neither select nor close.
        # While any popup is open, every click must reach the QML overlay.
        self._open_popups = 0

    @Slot(str, float, float, float, float)
    def update_rect(
        self, panel_id: str, x: float, y: float, w: float, h: float
    ) -> None:
        """Record (or replace) the current screen rect of a panel.

        Args:
            panel_id: A stable identifier for the panel (e.g. its QML `objectName`).
            x: Left edge, in overlay-widget-local pixels.
            y: Top edge, in overlay-widget-local pixels.
            w: Width in pixels.
            h: Height in pixels.
        """
        self._rects[panel_id] = QRectF(x, y, w, h)

    @Slot(bool)
    def set_popup_open(self, is_open: bool) -> None:
        """Track that a panel's popup (ComboBox dropdown, menu...) opened/closed.

        Bind this to each popup's ``visible`` state from QML. While the count
        is non-zero, hit_test always returns True so clicks on popup contents
        (which live outside every panel rect) still reach the QML overlay.
        """
        self._open_popups = max(0, self._open_popups + (1 if is_open else -1))

    def hit_test(self, pos: QPointF) -> bool:
        """Return True if pos falls inside any currently-registered panel rect.

        Args:
            pos: A position in overlay-widget-local pixels.

        Returns:
            True if a popup is open, or any registered panel rect contains pos.
        """
        if self._open_popups > 0:
            return True
        return any(rect.contains(pos) for rect in self._rects.values())
