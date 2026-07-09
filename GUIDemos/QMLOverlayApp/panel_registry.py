"""QML-exposed registry of floating panel screen rects, used for click-through hit testing."""

from PySide6.QtCore import QObject, QPointF, QRectF, Slot
from PySide6.QtQml import QmlElement

QML_IMPORT_NAME = "qmloverlayapp"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class PanelRegistry(QObject):
    """Tracks each floating panel's current screen rect for hit testing."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rects: dict[str, QRectF] = {}

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

    def hit_test(self, pos: QPointF) -> bool:
        """Return True if pos falls inside any currently-registered panel rect.

        Args:
            pos: A position in overlay-widget-local pixels.

        Returns:
            True if any registered panel rect contains pos.
        """
        return any(rect.contains(pos) for rect in self._rects.values())
