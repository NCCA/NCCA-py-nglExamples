import sys
from pathlib import Path

from PySide6.QtCore import QPointF

sys.path.insert(0, str(Path(__file__).parent.parent))

from panel_registry import PanelRegistry


def test_hit_test_false_when_no_panels_registered():
    registry = PanelRegistry()
    assert registry.hit_test(QPointF(10, 10)) is False


def test_hit_test_true_inside_a_registered_panel():
    registry = PanelRegistry()
    registry.update_rect("transform", 100.0, 100.0, 200.0, 150.0)
    assert registry.hit_test(QPointF(150.0, 150.0)) is True


def test_hit_test_false_outside_all_registered_panels():
    registry = PanelRegistry()
    registry.update_rect("transform", 100.0, 100.0, 200.0, 150.0)
    assert registry.hit_test(QPointF(0.0, 0.0)) is False


def test_update_rect_replaces_previous_rect_for_same_id():
    registry = PanelRegistry()
    registry.update_rect("transform", 0.0, 0.0, 10.0, 10.0)
    registry.update_rect("transform", 500.0, 500.0, 10.0, 10.0)
    assert registry.hit_test(QPointF(5.0, 5.0)) is False
    assert registry.hit_test(QPointF(505.0, 505.0)) is True


def test_hit_test_true_anywhere_while_popup_open():
    # A ComboBox dropdown opens outside every panel rect; clicks on it must
    # still reach the QML overlay, so any position must hit-test True.
    registry = PanelRegistry()
    registry.set_popup_open(True)
    assert registry.hit_test(QPointF(0.0, 0.0)) is True


def test_hit_test_reverts_after_popup_closes():
    registry = PanelRegistry()
    registry.set_popup_open(True)
    registry.set_popup_open(False)
    assert registry.hit_test(QPointF(0.0, 0.0)) is False


def test_popup_count_never_goes_negative():
    # A spurious close (e.g. visibleChanged firing false before true) must not
    # drive the count negative and mask a later genuinely-open popup.
    registry = PanelRegistry()
    registry.set_popup_open(False)
    registry.set_popup_open(True)
    assert registry.hit_test(QPointF(0.0, 0.0)) is True
