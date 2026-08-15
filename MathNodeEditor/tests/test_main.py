"""Tests for the maths node editor application entry point."""

import ast
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

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


def test_main_creates_and_shows_the_demo_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = _main_module()
    calls: list[object] = []

    class FakeApplication:
        @staticmethod
        def instance() -> "FakeApplication":
            return FakeApplication()

        def setOrganizationName(self, name: str) -> None:
            calls.append(("organisation", name))

        def setApplicationName(self, name: str) -> None:
            calls.append(("application", name))

        def exec(self) -> int:
            calls.append("exec")
            return 17

    class FakeWindow:
        def __init__(self, load_example: bool) -> None:
            calls.append(("window", load_example))

        def show(self) -> None:
            calls.append("show")

    monkeypatch.setattr(main_module, "QApplication", FakeApplication)
    monkeypatch.setattr(main_module, "MathNodeWindow", FakeWindow)

    result = main_module.main()

    assert result == 17
    assert calls == [
        ("organisation", "NCCA"),
        ("application", "MathNodeEditor"),
        ("window", True),
        "show",
        "exec",
    ]


def test_main_module_owns_the_application_entry_point() -> None:
    main_module = _main_module()

    assert main_module.main.__module__ == main_module.__name__


def test_main_uses_the_demo_local_editor_module() -> None:
    main_module = _main_module()

    assert main_module.MathNodeWindow.__module__ == "node_editor"


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
