"""Tests for the maths node editor application entry point."""

import ast
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
from PySide6.QtWidgets import QMainWindow

DEMO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DEMO_DIR))


def _main_module() -> ModuleType:
    """Load this demo's main.py without clashing with the other demos."""
    spec = spec_from_file_location("math_node_editor_main", DEMO_DIR / "main.py")
    if spec is None or spec.loader is None:
        pytest.fail("MathNodeEditor/main.py could not be loaded")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_module_owns_the_application_window() -> None:
    main_module = _main_module()

    assert issubclass(main_module.MainWindow, QMainWindow)
    assert main_module.MainWindow.__module__ == main_module.__name__


def test_main_module_owns_the_application_entry_point() -> None:
    main_module = _main_module()

    assert main_module.main.__module__ == main_module.__name__


def test_main_window_uses_the_demo_local_editor_module() -> None:
    main_module = _main_module()

    assert main_module.MainWindow.__base__.__module__ == "node_editor"


def test_demo_is_not_a_python_package() -> None:
    assert not (DEMO_DIR / "__init__.py").exists()


def test_demo_modules_do_not_use_package_relative_imports() -> None:
    relative_imports = [
        f"{path.name}:{node.lineno}"
        for path in DEMO_DIR.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]

    assert relative_imports == []
