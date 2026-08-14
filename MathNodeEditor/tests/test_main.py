"""Tests for the maths node editor application entry point."""

from importlib import import_module

from PySide6.QtWidgets import QMainWindow


def test_main_module_owns_the_application_window() -> None:
    main_module = import_module("MathNodeEditor.main")

    assert issubclass(main_module.MainWindow, QMainWindow)
    assert main_module.MainWindow.__module__ == main_module.__name__


def test_main_module_owns_the_application_entry_point() -> None:
    main_module = import_module("MathNodeEditor.main")

    assert main_module.main.__module__ == main_module.__name__
