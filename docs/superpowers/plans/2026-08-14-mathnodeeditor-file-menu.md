# MathNodeEditor File Menu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MathNodeEditor's `Load Vec3 example`/`Clear graph`/`Save graph...`/`Load graph...` palette buttons with a `File` menu (`New`/`Open.../Save/Save As.../Quit`, standard shortcuts) that persists the last-used file and window geometry via `QSettings`, prompts before discarding unsaved changes, and falls back to a bundled JSON example instead of an in-code-built graph.

**Architecture:** `MathNodeScene` (`canvas.py`) gains a `modified` flag plus a `_loading` guard so every real edit — but not a file load — marks the graph dirty; `MathNodeWindow` (`node_editor.py`) gains a `QMenuBar` File menu whose actions wrap the existing save/load JSON logic (moved up from `palette.py`), track a `current_file`, and read/write `QSettings` for the last file and geometry. The hardcoded Vec3-multiply demo becomes `examples/vec3_multiply_demo.json`, loaded through the normal `load_from_file` path.

**Tech Stack:** PySide6 (`QAction`, `QKeySequence`, `QSettings`, `QMessageBox`), existing `MathGraph`/`MathNodeScene` JSON round-trip (`to_dict`/`from_dict`/`save_to_file`/`load_from_file`), pytest + `QT_QPA_PLATFORM=offscreen`.

## Global Constraints

- No new dependencies — `QSettings`/`QAction`/`QKeySequence`/`QMessageBox` are all already part of PySide6.
- `QSettings` organization/application name follow this repo's existing convention (`GUIDemos/QMLOverlayApp/main.py`): `app.setOrganizationName("NCCA")` / `app.setApplicationName("<FolderName>")`.
- Tests must never read or write the real user's `QSettings` store (no `~/Library/Preferences/...` on macOS, no real `.config`/registry entries) — every test either goes through the autouse fixture added in Task 3 or passes an explicit temp-file-backed `QSettings`.
- Keep changes scoped to `MathNodeEditor/` (`canvas.py`, `graphics_items.py`, `node_editor.py`, `palette.py`, `examples/`, `README.md`, `tests/test_node_editor.py`) — no changes to `RunDemos.py` or repo-wide config.
- `uv run ruff check MathNodeEditor/` and `uv run ruff format --check MathNodeEditor/` must stay clean; run `uv run pytest MathNodeEditor/tests` after every task.
- Follow the spec at `docs/superpowers/specs/2026-08-14-mathnodeeditor-file-menu-design.md` — this plan implements it task-by-task.

---

## Task 1: Bundle the Vec3-multiply demo as a JSON example file

**Files:**
- Create: `MathNodeEditor/examples/vec3_multiply_demo.json`
- Modify: `MathNodeEditor/canvas.py` (`_OBJ_PARSE_ERRORS` block around line 72, `load_example` around line 333)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Produces: `canvas.DEFAULT_EXAMPLE_PATH` (module-level `Path` constant), used by Task 3's startup-fallback logic.
- `MathNodeScene.load_example()` keeps its existing signature/behaviour (still produces the two-`Vec3`-into-`Multiply`-into-`Output` graph); every later task and every existing test that calls it or constructs `MathNodeWindow(load_example=True)` keeps working unchanged.

- [ ] **Step 1: Create the example JSON file**

This is the exact `to_dict()` output of the current hand-built `load_example()` graph (verified by running it directly against the pre-change code). Write it verbatim:

```json
{
  "nodes": [
    {
      "id": "node-1",
      "x": -520.0,
      "y": -130.0,
      "kind": "value",
      "math_type": "VEC3",
      "components": [
        1.0,
        2.0,
        3.0
      ]
    },
    {
      "id": "node-2",
      "x": -520.0,
      "y": 100.0,
      "kind": "value",
      "math_type": "VEC3",
      "components": [
        4.0,
        5.0,
        6.0
      ]
    },
    {
      "id": "node-3",
      "x": -140.0,
      "y": 0.0,
      "kind": "operation",
      "operation": "MULTIPLY"
    },
    {
      "id": "node-4",
      "x": 220.0,
      "y": 0.0,
      "kind": "output"
    }
  ],
  "connections": [
    {
      "source": "node-1",
      "target": "node-3",
      "input": 0
    },
    {
      "source": "node-2",
      "target": "node-3",
      "input": 1
    },
    {
      "source": "node-3",
      "target": "node-4",
      "input": 0
    }
  ]
}
```

- [ ] **Step 2: Write a test that the bundled file loads to the expected result**

Add to `tests/test_node_editor.py`, near `test_mesh_pipeline_example_loads_without_errors` (which uses the same `Path(__file__).parent.parent / "examples" / ...` pattern):

```python
def test_default_example_file_loads_the_vec3_multiply_result(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "vec3_multiply_demo.json"

    window.canvas.load_from_file(example_path)
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()
```

- [ ] **Step 3: Run the new test to confirm it already passes**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py::test_default_example_file_loads_the_vec3_multiply_result -v`
Expected: PASS — this only exercises the JSON file created in Step 1 through the existing `load_from_file`, so it doesn't require any `canvas.py` change yet.

- [ ] **Step 4: Point `load_example()` at the bundled file**

In `MathNodeEditor/canvas.py`, add the path constant right after the `_OBJ_PARSE_ERRORS` tuple (around line 72-78):

```python
_OBJ_PARSE_ERRORS = (
    ObjParseVertexError,
    ObjParseNormalError,
    ObjParseUVError,
    ObjParseFaceError,
)

DEFAULT_EXAMPLE_PATH = Path(__file__).resolve().parent / "examples" / "vec3_multiply_demo.json"


class MathNodeScene(QGraphicsScene):
```

Then replace the hand-built `load_example` method:

```python
    def load_example(self) -> None:
        """Build the requested component-wise Vec3 multiplication example."""
        self.clear_graph()
        left = self.add_value_node(
            MathType.VEC3, QPointF(-520.0, -130.0), (1.0, 2.0, 3.0)
        )
        right = self.add_value_node(
            MathType.VEC3, QPointF(-520.0, 100.0), (4.0, 5.0, 6.0)
        )
        multiply = self.add_operation_node(Operation.MULTIPLY, QPointF(-140.0, 0.0))
        output = self.add_output_node(QPointF(220.0, 0.0))
        assert left.output_port is not None
        assert right.output_port is not None
        assert multiply.output_port is not None
        self.connect_ports(left.output_port, multiply.input_ports[0])
        self.connect_ports(right.output_port, multiply.input_ports[1])
        self.connect_ports(multiply.output_port, output.input_ports[0])
```

with:

```python
    def load_example(self) -> None:
        """Load the bundled Vec3 component-multiply example graph."""
        self.load_from_file(DEFAULT_EXAMPLE_PATH)
```

- [ ] **Step 5: Run the full test file to confirm nothing regressed**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS, including `test_example_graph_displays_the_vec3_multiply_result` and `test_save_to_file_and_load_from_file_round_trip` (both call `MathNodeWindow(load_example=True)` / `canvas.load_example()` and still expect `Vec3(4, 10, 18)`).

- [ ] **Step 6: Commit**

```bash
git add MathNodeEditor/examples/vec3_multiply_demo.json MathNodeEditor/canvas.py MathNodeEditor/tests/test_node_editor.py
git commit -m "refactor: load the Vec3-multiply demo from examples/vec3_multiply_demo.json"
```

---

## Task 2: Track unsaved changes on `MathNodeScene`

**Files:**
- Modify: `MathNodeEditor/canvas.py` (`__init__` ~line 83, `clear_graph` ~line 323, `from_dict` ~line 405, `update_outputs` ~line 476)
- Modify: `MathNodeEditor/graphics_items.py` (`BaseNodeItem.itemChange` ~line 343)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Produces: `MathNodeScene.modified: bool`, `MathNodeScene.modifiedChanged: Signal(bool)` (emitted only on a `False`→`True` or `True`→`False` transition), `MathNodeScene.mark_modified() -> None`. Task 3's title bar and Task 4's confirm-discard prompt both read `canvas.modified` and connect to `modifiedChanged`.
- `_loading` is a private implementation detail — nothing outside `canvas.py` touches it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_node_editor.py`:

```python
def test_adding_a_node_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    window.canvas.add_value_node(node_editor.MathType.VEC3)

    assert window.canvas.modified is True
    window.close()


def test_editing_a_value_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.modified = False

    node.spin_boxes[0].setValue(9.0)

    assert window.canvas.modified is True
    window.close()


def test_dragging_a_node_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.modified = False

    node.setPos(QPointF(50.0, 50.0))

    assert window.canvas.modified is True
    window.close()


def test_loading_a_file_leaves_the_scene_unmodified(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)

    window.canvas.load_from_file(file_path)

    assert window.canvas.modified is False
    window.close()


def test_clear_graph_resets_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    assert window.canvas.modified is True

    window.canvas.clear_graph()

    assert window.canvas.modified is False
    window.close()


def test_modified_changed_signal_fires_once_on_transition(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    seen: list[bool] = []
    window.canvas.modifiedChanged.connect(seen.append)

    window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.add_value_node(node_editor.MathType.VEC3)

    assert seen == [True]
    window.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k "modified" -v`
Expected: FAIL with `AttributeError: 'MathNodeScene' object has no attribute 'modified'`.

- [ ] **Step 3: Add the `modified`/`modifiedChanged`/`_loading` state**

In `MathNodeEditor/canvas.py`, add `Signal` to the `PySide6.QtCore` import (currently `from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRectF, Qt`):

```python
from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRectF, Qt, Signal
```

Then in the `MathNodeScene` class, add the signal and extend `__init__`:

```python
class MathNodeScene(QGraphicsScene):
    """Graphics scene which keeps visible wires and the maths graph in sync."""

    modifiedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty node graph canvas."""
        super().__init__(parent)
        self.setSceneRect(-1800.0, -1200.0, 3600.0, 2400.0)
        self.graph = MathGraph()
        self.nodes: dict[str, BaseNodeItem] = {}
        self.connections: list[ConnectionItem] = []
        self._drag_source: PortItem | None = None
        self._preview_connection: QGraphicsPathItem | None = None
        self._insertion_index = 0
        self.modified = False
        self._loading = False

    def mark_modified(self) -> None:
        """Flag the graph as having unsaved changes, unless a file load is in progress."""
        if self._loading or self.modified:
            return
        self.modified = True
        self.modifiedChanged.emit(True)
```

Place `mark_modified` right after `__init__`, before `_next_position`.

- [ ] **Step 4: Hook `update_outputs`, `clear_graph`, `from_dict`, and node position changes**

In `canvas.py`, add a `mark_modified()` call as the first line of `update_outputs`:

```python
    def update_outputs(self) -> None:
        """Evaluate and refresh every output and mesh viewer node in the scene."""
        self.mark_modified()
        for node_id, node in self.nodes.items():
```

At the end of `clear_graph`, reset the flag:

```python
    def clear_graph(self) -> None:
        """Remove all visible items and start a fresh calculation graph."""
        self.clear()
        self.graph = MathGraph()
        self.nodes.clear()
        self.connections.clear()
        self._drag_source = None
        self._preview_connection = None
        self._insertion_index = 0
        if self.modified:
            self.modified = False
            self.modifiedChanged.emit(False)
```

Wrap `from_dict`'s body in the `_loading` guard so rebuilding a graph from a file never marks it dirty (the surrounding `clear_graph()` call and the trailing `update_outputs()` call are both covered by the same guard):

```python
    def from_dict(self, data: dict[str, object]) -> None:
        """Replace the current graph with one built from ``to_dict`` data."""
        self._loading = True
        try:
            self.clear_graph()
            id_map: dict[str, str] = {}
            for entry in data["nodes"]:
                position = QPointF(entry["x"], entry["y"])
                kind = entry["kind"]
                if kind == "value":
                    node = self.add_value_node(
                        MathType[entry["math_type"]],
                        position,
                        tuple(entry["components"]),
                    )
                    id_map[entry["id"]] = node.node_id
                elif kind == "operation":
                    node = self.add_operation_node(Operation[entry["operation"]], position)
                    id_map[entry["id"]] = node.node_id
                elif kind == "generator":
                    node = self.add_generator_node(
                        Operation[entry["operation"]],
                        position,
                        tuple(tuple(p) for p in entry["parameters"]),
                    )
                    id_map[entry["id"]] = node.node_id
                elif kind == "output":
                    node = self.add_output_node(position)
                    id_map[entry["id"]] = node.node_id
                elif kind == "obj_loader":
                    node = self.add_obj_loader_node(position)
                    node.set_status(entry.get("status", ""))
                    vertices = VertexArray(tuple(Vec3(*v) for v in entry["vertices"]))
                    faces = FaceArray(
                        tuple(
                            tuple((corner[0], corner[1], corner[2]) for corner in triangle)
                            for triangle in entry["faces"]
                        )
                    )
                    uvs = UVArray(tuple(Vec2(*uv) for uv in entry["uvs"]))
                    normals = NormalArray(tuple(Vec3(*n) for n in entry["normals"]))
                    self.graph.set_literal(node.array_node_ids[0], vertices)
                    self.graph.set_literal(node.array_node_ids[1], faces)
                    self.graph.set_literal(node.array_node_ids[2], uvs)
                    self.graph.set_literal(node.array_node_ids[3], normals)
                    for old_id, new_id in zip(entry["array_ids"], node.array_node_ids):
                        id_map[old_id] = new_id
                elif kind == "mesh_viewer":
                    node = self.add_mesh_viewer_node(
                        position,
                        shading_mode=entry.get("shading_mode", SHADING_SOLID),
                        wireframe=bool(entry.get("wireframe", False)),
                    )
                    id_map[entry["id"]] = node.node_id
                else:
                    raise GraphError(f"Unknown node kind {kind!r}")
            for connection in data["connections"]:
                source_port = self._output_port_for_node_id(id_map[connection["source"]])
                target_node = self.nodes[id_map[connection["target"]]]
                self.connect_ports(
                    source_port,
                    target_node.input_ports[connection["input"]],
                )
            self.update_outputs()
        finally:
            self._loading = False
```

(Only the first and last lines of the body actually change — `self._loading = True` / `try:` at the top and `finally: self._loading = False` at the bottom; the rest of the method is reproduced above unindented-then-reindented for a clean copy-paste.)

- [ ] **Step 5: Hook node dragging**

In `MathNodeEditor/graphics_items.py`, update `BaseNodeItem.itemChange`:

```python
    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        """Keep attached wires aligned when the node moves, and flag the scene dirty."""
        if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            ports = [*self.input_ports, *self.output_ports]
            for port in ports:
                for connection in port.connections:
                    connection.update_path()
            scene = self.scene()
            if scene is not None:
                scene.mark_modified()
        return super().itemChange(change, value)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k "modified" -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Run the full test file**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS — in particular `test_mesh_pipeline_example_loads_without_errors`,
`test_default_example_file_loads_the_vec3_multiply_result` and every other
`load_from_file`/`from_dict` based test must still pass with `canvas.modified`
ending up `False` (they don't assert on `modified` yet, but a broken `_loading`
guard would break `from_dict` itself, e.g. via a stuck `True` value leaking
into a later reused scene — there isn't one here since each test builds its
own window, but keep an eye on tracebacks from the `try/finally`).

- [ ] **Step 8: Commit**

```bash
git add MathNodeEditor/canvas.py MathNodeEditor/graphics_items.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: track unsaved changes on MathNodeScene"
```

---

## Task 3: File menu with New/Open/Save/Save As, `QSettings` persistence, and startup fallback

**Files:**
- Modify: `MathNodeEditor/node_editor.py` (whole file — imports, `MathNodeWindow`, `main`)
- Modify: `MathNodeEditor/palette.py` (imports, `NodePalette.__init__` button block, delete `_save_graph`/`_load_graph`)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Consumes: `MathNodeScene.modified`/`modifiedChanged` (Task 2), `canvas.DEFAULT_EXAMPLE_PATH`/`load_example()` (Task 1).
- Produces: `MathNodeWindow.action_new/action_open/action_save/action_save_as/action_quit` (`QAction`), `MathNodeWindow.current_file: Path | None`, `MathNodeWindow.settings: QSettings`, `MathNodeWindow._open_path(path) -> bool`, `MathNodeWindow._save_path(path) -> bool`. Task 4 adds a confirm-discard guard in front of `_new_graph`/`_open_graph`/`closeEvent`; it calls these exact names.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_node_editor.py`. These need two of the existing import
lines widened. Replace:

```python
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QFontMetrics, QWheelEvent
```

with:

```python
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QAction, QFont, QFontMetrics, QKeySequence, QWheelEvent
```

```python
def _isolated_settings(tmp_path: Path) -> QSettings:
    """Build a throwaway QSettings store so tests never touch real user prefs."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture(autouse=True)
def _redirect_default_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point every bare QSettings() at a throwaway file for this test only.

    ``MathNodeWindow(load_example=...)`` calls elsewhere in this file (there
    are dozens, from before this task) never pass ``settings=`` explicitly,
    so without this autouse fixture they'd fall through to
    ``node_editor._default_settings()`` -> a bare ``QSettings()`` with no
    organization/application name set (``main()`` is what sets those, and
    tests never call ``main()``), which warns on stderr and, on macOS,
    resolves to a native preferences store keyed by an empty bundle id
    instead of a clean temp file.
    """
    node_editor = _node_editor_module()
    ini_path = str(tmp_path / "default-settings.ini")
    monkeypatch.setattr(
        node_editor,
        "_default_settings",
        lambda: QSettings(ini_path, QSettings.Format.IniFormat),
    )


def test_file_menu_has_the_expected_actions_and_shortcuts(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )

    assert window.action_new.shortcut() == QKeySequence.StandardKey.New
    assert window.action_open.shortcut() == QKeySequence.StandardKey.Open
    assert window.action_save.shortcut() == QKeySequence.StandardKey.Save
    assert window.action_save_as.shortcut() == QKeySequence.StandardKey.SaveAs
    window.close()


def test_new_clears_the_graph_and_forgets_the_current_file(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    assert window.current_file is None
    window.close()


def test_save_as_writes_a_file_and_becomes_the_current_file(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_save_as.trigger()

    assert file_path.exists()
    assert window.current_file == file_path
    window.close()


def test_save_writes_to_the_current_file_without_prompting(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    window.current_file = tmp_path / "graph.json"

    def _fail_if_called(*_args: object, **_kwargs: object) -> tuple[str, str]:
        raise AssertionError("Save must not prompt when a current file is set")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fail_if_called)

    window.action_save.trigger()

    assert window.current_file.exists()
    window.close()


def test_save_without_a_current_file_behaves_like_save_as(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_save.trigger()

    assert window.current_file == file_path
    window.close()


def test_open_replaces_the_graph_and_updates_recent_file_setting(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)
    window.action_new.trigger()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_open.trigger()
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    assert settings.value("recentFile") == str(file_path)
    window.close()


def test_startup_reopens_the_recent_file_when_present(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    seed_window = node_editor.MathNodeWindow(load_example=False, settings=settings)
    seed_window.canvas.add_value_node(node_editor.MathType.VEC3)
    output = seed_window.canvas.add_output_node()
    value_node = next(iter(seed_window.canvas.nodes.values()))
    file_path = tmp_path / "graph.json"
    seed_window.canvas.save_to_file(file_path)
    seed_window.close()
    settings.setValue("recentFile", str(file_path))

    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    application.processEvents()

    assert window.current_file == file_path
    assert len(window.canvas.nodes) == 2
    window.close()


def test_startup_falls_back_to_the_bundled_demo_without_a_recent_file(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    application.processEvents()

    assert window.current_file is None
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_startup_falls_back_when_the_recent_file_is_missing(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    settings.setValue("recentFile", str(tmp_path / "does-not-exist.json"))

    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    application.processEvents()

    assert window.current_file is None
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_window_geometry_round_trips_through_settings(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    first = node_editor.MathNodeWindow(load_example=False, settings=settings)
    first.resize(999, 555)
    first.show()
    application.processEvents()
    first.close()

    second = node_editor.MathNodeWindow(load_example=False, settings=settings)

    assert second.size().width() == 999
    assert second.size().height() == 555
    second.close()


def test_title_shows_the_current_file_name_and_dirty_marker(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    assert window.windowTitle().endswith("Untitled")

    window.canvas.add_value_node(node_editor.MathType.VEC3)
    application.processEvents()

    assert window.windowTitle().endswith("Untitled*")
    window.close()
```

Also rewrite the three tests that currently look up palette buttons by text
for save/load (`test_save_graph_button_writes_a_json_file`,
`test_load_graph_button_replaces_the_current_graph`,
`test_load_graph_button_reports_a_malformed_file_instead_of_crashing`) to use
the new actions instead. Replace all three with:

```python
def test_save_action_writes_a_json_file(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )

    window.action_save.trigger()
    application.processEvents()

    assert file_path.exists()
    window.close()


def test_open_action_replaces_the_current_graph(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)
    window.canvas.clear_graph()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )

    window.action_open.trigger()
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_open_action_reports_a_malformed_file_instead_of_crashing(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "broken.json"
    file_path.write_text("not valid json")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )
    warnings: list[object] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    window.action_open.trigger()
    application.processEvents()

    assert len(warnings) == 1
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()
```

Remove the old three tests (`test_save_graph_button_writes_a_json_file`,
`test_load_graph_button_replaces_the_current_graph`,
`test_load_graph_button_reports_a_malformed_file_instead_of_crashing`)
entirely — they assert against `QPushButton`s this task deletes.

- [ ] **Step 2: Run the full file to verify it fails**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: FAIL across essentially every test in the file, not just the new
ones — the `_redirect_default_settings` autouse fixture calls
`monkeypatch.setattr(node_editor, "_default_settings", ...)`, and
`node_editor._default_settings` doesn't exist until Step 3, so fixture setup
itself raises `AttributeError` for every test. That's expected here; Step 3
adds the attribute and Step 5 re-runs the full file to confirm it's fixed.

- [ ] **Step 3: Rewrite `node_editor.py`**

Replace the whole file:

```python
"""PySide6 node editor for experimenting with PyNGL maths types."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from .canvas import MathNodeScene, MathNodeView
from .graphics_items import (
    GENERIC_PORT_COLOUR,
    NODE_HEADER_HEIGHT,
    PORT_RADIUS,
    TYPE_COLOURS,
    BaseNodeItem,
    ConnectionItem,
    GeneratorNodeItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    default_components,
    node_title_font,
)
from .math_graph import (
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    TYPE_SHAPES,
    GraphError,
    MathGraph,
    MathType,
    Operation,
    format_value,
)
from .palette import (
    MAT4_OPERATIONS,
    MATH_OPERATIONS,
    MESH_OPERATIONS,
    NODE_CATALOGUE,
    QUATERNION_OPERATIONS,
    CatalogueEntry,
    CatalogueSection,
    NodeCreationMenu,
    NodePalette,
)

__all__ = [
    "GENERIC_PORT_COLOUR",
    "MAT4_OPERATIONS",
    "MATH_OPERATIONS",
    "MESH_OPERATIONS",
    "NODE_CATALOGUE",
    "NODE_HEADER_HEIGHT",
    "OPERATION_ARITY",
    "OPERATION_INPUT_NAMES",
    "PORT_RADIUS",
    "QUATERNION_OPERATIONS",
    "TYPE_COLOURS",
    "TYPE_SHAPES",
    "BaseNodeItem",
    "CatalogueEntry",
    "CatalogueSection",
    "ConnectionItem",
    "GeneratorNodeItem",
    "GraphError",
    "MathGraph",
    "MathNodeScene",
    "MathNodeView",
    "MathNodeWindow",
    "MathType",
    "MeshViewerNodeItem",
    "NodeCreationMenu",
    "NodePalette",
    "ObjLoaderNodeItem",
    "Operation",
    "OperationNodeItem",
    "OutputNodeItem",
    "PortItem",
    "ValueNodeItem",
    "default_components",
    "format_value",
    "main",
    "node_title_font",
]

WINDOW_TITLE = "PyNGL Maths Node Editor"

_LOAD_ERRORS = (
    OSError,
    json.JSONDecodeError,
    GraphError,
    KeyError,
    TypeError,
    ValueError,
    IndexError,
)


def _default_settings() -> QSettings:
    """Return the QSettings store used when a window isn't given one explicitly."""
    return QSettings()


class MathNodeWindow(QMainWindow):
    """Main window containing the palette and node graph canvas."""

    def __init__(
        self,
        load_example: bool = True,
        settings: QSettings | None = None,
    ) -> None:
        """Create the editor, restore its settings, and load a starting graph."""
        super().__init__()
        self.settings = settings if settings is not None else _default_settings()
        self.current_file: Path | None = None
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1280, 760)
        self.canvas = MathNodeScene(self)
        self.view = MathNodeView(self.canvas, self)
        self.view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.palette = NodePalette(self.canvas, self.view, self)
        self.palette.setMinimumHeight(self.palette.sizeHint().height())
        self.palette_scroll = QScrollArea(self)
        self.palette_scroll.setWidget(self.palette)
        self.palette_scroll.setWidgetResizable(True)
        self.palette_scroll.setFixedWidth(240)
        self.palette_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.palette_scroll)
        central_layout.addWidget(self.view, 1)
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage(
            "Press Tab to add a node; edit values and wire nodes together"
        )
        self._build_file_menu()
        self.canvas.modifiedChanged.connect(lambda _modified: self._update_title())
        if load_example:
            self._load_startup_graph()
            self.view.centerOn(0.0, 0.0)
        self._update_title()

    def _build_file_menu(self) -> None:
        """Build the File menu's New/Open/Save/Save As/Quit actions."""
        file_menu = self.menuBar().addMenu("&File")

        self.action_new = QAction("&New", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.triggered.connect(self._new_graph)
        file_menu.addAction(self.action_new)

        self.action_open = QAction("&Open...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._open_graph)
        file_menu.addAction(self.action_open)

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_graph)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self._save_graph_as)
        file_menu.addAction(self.action_save_as)

        file_menu.addSeparator()

        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)

    def _update_title(self) -> None:
        """Show the current file name and an unsaved-changes marker in the title bar."""
        name = self.current_file.name if self.current_file else "Untitled"
        star = "*" if self.canvas.modified else ""
        self.setWindowTitle(f"{WINDOW_TITLE} — {name}{star}")

    def _load_startup_graph(self) -> None:
        """Reopen the last file used, falling back to the bundled example."""
        recent_file = self.settings.value("recentFile", "", type=str)
        if recent_file and Path(recent_file).is_file():
            if self._open_path(Path(recent_file)):
                return
        self.canvas.load_example()
        self.current_file = None

    def _new_graph(self) -> None:
        """Discard the current graph and start a blank one."""
        self.canvas.clear_graph()
        self.current_file = None
        self._update_title()

    def _open_graph(self) -> None:
        """Prompt for a path and replace the current graph with its contents."""
        path, _name_filter = QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._open_path(Path(path))

    def _open_path(self, path: Path) -> bool:
        """Load a graph file, reporting failure instead of raising. Return success."""
        try:
            self.canvas.load_from_file(path)
        except _LOAD_ERRORS as error:
            QMessageBox.warning(self, "Open Graph", f"Could not open graph: {error}")
            return False
        self.current_file = path
        self.settings.setValue("recentFile", str(path))
        self._update_title()
        return True

    def _save_graph(self) -> None:
        """Save to the current file, or prompt for one if there isn't one yet."""
        if self.current_file is None:
            self._save_graph_as()
            return
        self._save_path(self.current_file)

    def _save_graph_as(self) -> None:
        """Prompt for a path and save the current graph to it."""
        path, _name_filter = QFileDialog.getSaveFileName(
            self, "Save Graph As", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._save_path(Path(path))

    def _save_path(self, path: Path) -> bool:
        """Write the current graph to a path, reporting failure. Return success."""
        try:
            self.canvas.save_to_file(path)
        except OSError as error:
            QMessageBox.warning(self, "Save Graph", f"Could not save graph: {error}")
            return False
        self.current_file = path
        self.settings.setValue("recentFile", str(path))
        self.canvas.modified = False
        self._update_title()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist the window geometry before closing."""
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def main() -> int:
    """Run the maths node editor application."""
    application = QApplication.instance()
    if application is None:
        surface_format = QSurfaceFormat()
        surface_format.setSamples(4)
        surface_format.setMajorVersion(4)
        surface_format.setMinorVersion(1)
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface_format.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(surface_format)
        # The Mesh Viewer node's embedded preview and its pop-out window are
        # two separate top-level GL surfaces; without this they don't share
        # a context and ShaderLib's program ids from one are invalid in the
        # other.
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        application = QApplication(sys.argv)
    # QSettings derives its storage path from the organization/application
    # name, so this has to happen before the first QSettings() is created
    # (inside MathNodeWindow.__init__) or it falls back to an unnamed store.
    application.setOrganizationName("NCCA")
    application.setApplicationName("MathNodeEditor")
    window = MathNodeWindow(load_example=True)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Remove the four superseded palette buttons**

In `MathNodeEditor/palette.py`, drop the now-unused imports. Replace:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .math_graph import GENERATOR_OPERATIONS, GraphError, MathType, Operation
```

with:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .math_graph import GENERATOR_OPERATIONS, MathType, Operation
```

Then replace the button-creation block plus the two handler methods:

```python
        frame_all_button = QPushButton("Frame All")
        frame_all_button.clicked.connect(lambda _checked=False: self.view.frame_all())
        layout.addWidget(frame_all_button)
        example_button = QPushButton("Load Vec3 example")
        example_button.clicked.connect(
            lambda _checked=False: self.canvas.load_example()
        )
        layout.addWidget(example_button)
        clear_button = QPushButton("Clear graph")
        clear_button.clicked.connect(lambda _checked=False: self.canvas.clear_graph())
        layout.addWidget(clear_button)
        save_button = QPushButton("Save graph...")
        save_button.clicked.connect(lambda _checked=False: self._save_graph())
        layout.addWidget(save_button)
        load_button = QPushButton("Load graph...")
        load_button.clicked.connect(lambda _checked=False: self._load_graph())
        layout.addWidget(load_button)

    def _save_graph(self) -> None:
        """Prompt for a path and write the current graph to it."""
        path, _name_filter = QFileDialog.getSaveFileName(
            self, "Save Graph", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.canvas.save_to_file(path)
        except OSError as error:
            QMessageBox.warning(self, "Save Graph", f"Could not save graph: {error}")

    def _load_graph(self) -> None:
        """Prompt for a path and replace the current graph with its contents."""
        path, _name_filter = QFileDialog.getOpenFileName(
            self, "Load Graph", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.canvas.load_from_file(path)
        except (
            OSError,
            json.JSONDecodeError,
            GraphError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
        ) as error:
            QMessageBox.warning(self, "Load Graph", f"Could not load graph: {error}")
```

with just:

```python
        frame_all_button = QPushButton("Frame All")
        frame_all_button.clicked.connect(lambda _checked=False: self.view.frame_all())
        layout.addWidget(frame_all_button)
```

- [ ] **Step 5: Run the target tests to verify they pass**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS, all tests including the ones added/rewritten in Step 1.

- [ ] **Step 6: Commit**

```bash
git add MathNodeEditor/node_editor.py MathNodeEditor/palette.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: replace palette save/load/clear buttons with a File menu and QSettings persistence"
```

---

## Task 4: Prompt before discarding unsaved changes

**Files:**
- Modify: `MathNodeEditor/node_editor.py` (`_new_graph`, `_open_graph`, `closeEvent`)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Consumes: `MathNodeScene.modified` (Task 2), `MathNodeWindow._save_graph()`/`current_file` (Task 3).
- Produces: `MathNodeWindow._confirm_discard_changes() -> bool` — no other task depends on it, it's the terminal piece of the spec.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_node_editor.py`:

```python
def test_new_prompts_and_cancels_when_discarding_is_declined(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    window.action_new.trigger()

    assert len(window.canvas.nodes) == 1
    window.close()


def test_new_discards_without_saving_when_chosen(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    window.close()


def test_new_saves_first_when_save_is_chosen(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )

    window.action_new.trigger()

    assert file_path.exists()
    assert window.canvas.nodes == {}
    window.close()


def test_new_does_not_prompt_when_the_graph_is_unmodified(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )

    def _fail_if_called(*_args: object, **_kwargs: object) -> QMessageBox.StandardButton:
        raise AssertionError("Must not prompt on a clean graph")

    monkeypatch.setattr(QMessageBox, "question", _fail_if_called)

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    window.close()


def test_open_prompts_before_replacing_a_modified_graph(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    other_path = tmp_path / "other.json"
    seed_window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    seed_window.canvas.save_to_file(other_path)
    seed_window.close()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(other_path), "")
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    window.action_open.trigger()

    assert len(window.canvas.nodes) == 1
    window.close()
```

A widget's `isVisible()`/`isHidden()` under `QT_QPA_PLATFORM=offscreen`
doesn't reliably reflect whether a `close()` call was actually accepted or
ignored, so don't assert on those — call `closeEvent` directly against a
real event object and check `isAccepted()` instead:

```python
def test_close_event_ignored_when_discard_is_cancelled(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )
    close_event = QCloseEvent()

    window.closeEvent(close_event)

    assert close_event.isAccepted() is False
    window.canvas.clear_graph()
    window.close()
```

This needs `QCloseEvent`, which isn't imported yet. Task 3 already widened
the `PySide6.QtGui` import to
`from PySide6.QtGui import QAction, QFont, QFontMetrics, QKeySequence, QWheelEvent`
— add `QCloseEvent` to that same line, alphabetically after `QAction`:

```python
from PySide6.QtGui import QAction, QCloseEvent, QFont, QFontMetrics, QKeySequence, QWheelEvent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k "discard or close_event" -v`
Expected: FAIL — `New`/`Open`/`closeEvent` don't prompt yet, so `QMessageBox.question` is never called and the graph is always discarded/closed immediately.

- [ ] **Step 3: Add `_confirm_discard_changes` and wire it in**

In `MathNodeEditor/node_editor.py`, add the method (place it right after
`_update_title`):

```python
    def _confirm_discard_changes(self) -> bool:
        """Ask to save unsaved changes; return whether it's safe to proceed."""
        if not self.canvas.modified:
            return True
        response = QMessageBox.question(
            self,
            WINDOW_TITLE,
            "Save changes to the current graph before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Discard:
            return True
        self._save_graph()
        return not self.canvas.modified
```

Then gate `_new_graph` and `_open_graph`:

```python
    def _new_graph(self) -> None:
        """Discard the current graph and start a blank one."""
        if not self._confirm_discard_changes():
            return
        self.canvas.clear_graph()
        self.current_file = None
        self._update_title()

    def _open_graph(self) -> None:
        """Prompt for a path and replace the current graph with its contents."""
        if not self._confirm_discard_changes():
            return
        path, _name_filter = QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._open_path(Path(path))
```

And gate `closeEvent` — `Quit` (`self.action_quit.triggered.connect(self.close)`,
unchanged from Task 3) calls `self.close()`, which is what triggers
`closeEvent`, so this one check covers both the `Quit` menu action and the
window's own close button:

```python
    def closeEvent(self, event: QCloseEvent) -> None:
        """Confirm discarding unsaved changes, then persist the window geometry."""
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k "discard or close_event" -v`
Expected: PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS, all tests (every test that calls `window.close()` on an
unmodified or already-saved graph must keep working without a stray prompt —
this is exactly what `test_new_does_not_prompt_when_the_graph_is_unmodified`
and the "must not prompt on a clean graph" pattern guard against, but check
the full run too since dozens of existing tests call `window.close()` after
adding nodes without saving, which would now normally count as "modified"
and could hang on an unmocked `QMessageBox.question` — those calls go through
`close()`, not `closeEvent()`'s Python override being invoked by Qt directly
in the offscreen platform without a real close request, so verify this in
Step 5's run rather than assuming it).

- [ ] **Step 6: Commit**

```bash
git add MathNodeEditor/node_editor.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: prompt to save unsaved changes before New, Open, or closing the window"
```

---

## Task 5: Update the README and refresh the screenshot

**Files:**
- Modify: `MathNodeEditor/README.md`
- Modify: `MathNodeEditor/MathNodeEditor.png`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update the opening paragraph**

In `MathNodeEditor/README.md`, replace:

```
This is a small PySide6 node editor for experimenting with the PyNGL maths classes. It starts with two editable `Vec3` nodes wired through a component multiply node, so changing any value updates the output straight away.
```

with:

```
This is a small PySide6 node editor for experimenting with the PyNGL maths classes. It reopens whichever graph you had open last; the first time you run it (or if that file's gone missing) it opens `examples/vec3_multiply_demo.json` instead — two editable `Vec3` nodes wired through a component multiply node, so changing any value updates the output straight away.
```

- [ ] **Step 2: Replace the Save/Load paragraph**

Replace:

```
Select a node or wire and press `Delete`/`Backspace` to remove it, or right-click it for the same option. The `Save graph...`/`Load graph...` buttons write the current graph out as JSON and read it back in.
```

with:

```
Select a node or wire and press `Delete`/`Backspace` to remove it, or right-click it for the same option. The `File` menu's `New` (`Ctrl+N`), `Open...` (`Ctrl+O`), `Save` (`Ctrl+S`) and `Save As...` (`Ctrl+Shift+S`) read and write the graph as JSON — `Save` writes straight back to whichever file is open, prompting for one the first time. `New`, `Open...` and closing the window all ask to save first if the graph has unsaved changes. The window remembers its size and the last file you had open between runs.
```

- [ ] **Step 3: List the new example file**

Right before the `examples/mesh_pipeline_demo.json` paragraph, insert:

```
`examples/vec3_multiply_demo.json` is the bundled starting graph described above — two `Vec3` values multiplied component-wise into an `Output` node.
```

- [ ] **Step 4: Capture a fresh screenshot**

Run the app, open the File menu so it's visible, and capture the window:

```bash
uv run MathNodeEditor/main.py
```

On macOS, `Cmd+Shift+4` then `Space` and click the window captures just that
window; save over `MathNodeEditor/MathNodeEditor.png`. Confirm the README's
`![MathNodeEditor](MathNodeEditor.png)` still renders (no filename change
needed).

- [ ] **Step 5: Commit**

```bash
git add MathNodeEditor/README.md MathNodeEditor/MathNodeEditor.png
git commit -m "docs: describe the File menu and QSettings persistence in the MathNodeEditor README"
```

---

## Task 6: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest MathNodeEditor/tests -v`
Expected: PASS, no failures, no errors.

- [ ] **Step 2: Lint and format check**

Run: `uv run ruff check MathNodeEditor/`
Run: `uv run ruff format --check MathNodeEditor/`
Expected: both clean. If `ruff format --check` reports files needing
reformatting, run `uv run ruff format MathNodeEditor/` and re-run the test
suite before committing the formatting fix.

- [ ] **Step 3: Manual smoke test**

Run: `uv run MathNodeEditor/main.py`

Walk through:
- The window opens showing `examples/vec3_multiply_demo.json` (title bar
  reads `PyNGL Maths Node Editor — vec3_multiply_demo.json`) on a first run
  with no prior settings.
- Add a node, confirm the title gains a trailing `*`.
- `File > Save As...`, save to a temp path; confirm the `*` clears and the
  title shows the new filename.
- `File > New`; confirm it clears without prompting (graph was just saved).
- Add a node, `File > New` again; confirm the Save/Discard/Cancel prompt
  appears, and each of the three choices behaves as expected.
- `File > Open...`, pick the file saved above; confirm it loads.
- Resize the window, quit via `File > Quit`, then relaunch: confirm the
  window reopens at the same size and with the same file loaded.

- [ ] **Step 4: Confirm `RunDemos.py` still launches the demo**

Run: `uv run RunDemos.py`, locate MathNodeEditor in the list, launch it from
there, and confirm it starts without error, then close both windows.

No commit for this task — it's verification of the work already committed in
Tasks 1-5. If any step surfaces a problem, fix it in a follow-up commit
before moving on to `superpowers:finishing-a-development-branch`.
