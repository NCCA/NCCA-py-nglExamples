# MathNodeEditor: File menu, QSettings persistence, JSON default demo

## Goal

Replace the palette's `Load Vec3 example` / `Clear graph` / `Save graph...` /
`Load graph...` buttons with a proper `File` menu (`New`, `Open...`, `Save`,
`Save As...`, `Quit`, all with standard shortcuts). On launch, reopen
whichever file was last opened or saved instead of always rebuilding the
hardcoded Vec3-multiply example; if there is no last file (first run, or it's
gone missing), fall back to a bundled JSON scene file instead of the
in-code-built graph. Window geometry and the last-used file path persist via
`QSettings` between runs.

## Scope

**In:** File menu + shortcuts, QSettings-backed last-file/geometry
persistence, converting the hardcoded Vec3-multiply demo into
`examples/vec3_multiply_demo.json`, unsaved-changes prompts (per Jon: New /
Open / Quit ask to save first when the graph is dirty).

**Out:** recent-files submenu (just the single last file), autosave, undo
history, renaming/moving any other example file.

## File menu (`node_editor.py`)

`MathNodeWindow` gains a `QMenuBar` with one `File` menu, built in a new
`_build_file_menu()`:

| Action | Shortcut | Behaviour |
|---|---|---|
| `New` | `Ctrl+N` | confirm-discard, then `canvas.clear_graph()`, `current_file = None` |
| `Open...` | `Ctrl+O` | confirm-discard, then `QFileDialog.getOpenFileName`, `_open_path(path)` |
| `Save` | `Ctrl+S` | `_open_path`'s counterpart `_save_path(current_file)` if set, else same as Save As |
| `Save As...` | `Ctrl+Shift+S` | `QFileDialog.getSaveFileName`, always prompts, then `_save_path(path)` |
| `Quit` | `QKeySequence.StandardKey.Quit` | `self.close()` — `closeEvent` owns the confirm-discard check, so `Quit` doesn't duplicate it |

Each action is kept as an attribute (`self.action_new`, `self.action_open`,
`self.action_save`, `self.action_save_as`, `self.action_quit`) so tests can
trigger them directly instead of walking the menu bar.

`_open_path(path)` / `_save_path(path)` are the shared error-handling wrappers
the current palette `_load_graph`/`_save_graph` already have (same
`QMessageBox.warning` on `OSError`/`json.JSONDecodeError`/`GraphError`/etc.),
moved from `NodePalette` onto `MathNodeWindow`. On success both update
`self.current_file`, write `settings.setValue("recentFile", str(path))`, and
refresh the title bar (`_update_title()`, see below).

`_confirm_discard_changes() -> bool`: if `canvas.modified` is `False`,
returns `True` immediately. Otherwise shows a `QMessageBox.question` with
Save / Discard / Cancel. Save runs the same Save-or-Save-As logic and returns
its outcome (a cancelled Save-As counts as an overall Cancel); Discard
returns `True`; Cancel returns `False`. `New` and `Open...` call this
directly before doing anything destructive; `Quit` doesn't call it itself —
it just calls `self.close()`, and `closeEvent` is the single place that
calls `_confirm_discard_changes()` and `ignore()`s the event on `False`, so
closing via the window's own close button goes through the same check
without `Quit` prompting twice.

Title bar shows `PyNGL Maths Node Editor — <filename or "Untitled">` with a
trailing `*` while `canvas.modified` is `True`, refreshed from
`canvas.modifiedChanged` (new `Signal(bool)`, see below) plus after every
open/save/new.

## Palette (`palette.py`)

Remove the `Load Vec3 example`, `Clear graph`, `Save graph...` and
`Load graph...` buttons and their handlers (`_save_graph`/`_load_graph`) —
superseded by the File menu. `Frame All` stays; it's a view action, not a
file one.

## Startup + QSettings persistence (`node_editor.py`)

`MathNodeWindow.__init__(self, load_example: bool = True, settings: QSettings | None = None)`:

- `self.settings = settings if settings is not None else QSettings()`
  (relies on `QApplication.setOrganizationName("NCCA")` /
  `setApplicationName("MathNodeEditor")`, set in `main()` before the window
  is constructed — matches the convention already used by
  `GUIDemos/QMLOverlayApp/main.py`).
- Geometry: `if geometry := self.settings.value("geometry"): self.restoreGeometry(geometry)` else `self.resize(1280, 760)`.
- If `load_example` is `True` (the default; tests that don't want a starting
  graph still pass `load_example=False` exactly as today):
  - Read `settings.value("recentFile", "", type=str)`. If it names an
    existing file, try `_open_path(path)`; on any load failure fall through.
  - Otherwise (or on that failure) load the bundled default:
    `canvas.load_example()`, and leave `current_file = None` — opening the
    shipped example must never let a plain `Ctrl+S` overwrite it.
- `closeEvent` writes `settings.setValue("geometry", self.saveGeometry())`
  before accepting (after `_confirm_discard_changes()` allows the close).

`main()` sets `application.setOrganizationName("NCCA")` and
`application.setApplicationName("MathNodeEditor")` right after constructing
`QApplication`, alongside the existing `setApplicationName("PyNGL Maths Node
Editor")` call (that one stays — it's the display name already used in the
window title).

## Dirty tracking (`canvas.py`, `graphics_items.py`)

`MathNodeScene` gains `modified: bool`, `modifiedChanged = Signal(bool)`, and
a private `_loading: bool` guard:

- `mark_modified()`: no-op while `_loading`, otherwise sets `modified = True`
  and emits `modifiedChanged(True)` on a False→True transition.
- `update_outputs()` calls `mark_modified()` first thing. This alone covers
  every value/generator edit, connect/disconnect, node delete and Obj/mesh
  viewer change, since all of those already end by calling
  `update_outputs()` today.
- `BaseNodeItem.itemChange` (`graphics_items.py`) calls
  `self.scene().mark_modified()` on `ItemPositionHasChanged` — one hook
  covers both "node added" (every `add_*_node` ends with `setPos(...)`) and
  "node dragged", with no per-`add_*_node` edits needed.
- `clear_graph()` resets `modified = False` (and emits `modifiedChanged(False)`
  if it was `True`) unconditionally — correct both for `New` (a blank graph
  has nothing unsaved) and as the first step inside `from_dict`.
- `from_dict()` wraps its whole body in `self._loading = True` /
  `finally: self._loading = False`, so the node/connection rebuild it drives
  through `add_*_node`/`connect_ports`/`update_outputs` never marks the
  scene dirty. A freshly opened or reloaded file is clean.
- `load_example()` becomes `self.load_from_file(DEFAULT_EXAMPLE_PATH)` (see
  below), so it's covered by the same `_loading` guard.

No changes needed at `connect_ports`, `_remove_connection`, `_delete_node`,
`_disconnect` or `_load_obj_into_node` individually — they all already
funnel through `update_outputs()`.

## Default demo JSON (`canvas.py`, `examples/`)

- New `examples/vec3_multiply_demo.json`: today's hardcoded `load_example()`
  graph (two `Vec3(1,2,3)`/`Vec3(4,5,6)` nodes → `Multiply` → `Output`),
  generated once by running the current builder and calling `save_to_file`,
  matching the naming convention of `mvp_demo.json` /
  `mesh_pipeline_demo.json` / `mvp_mesh_demo.json`.
- `DEFAULT_EXAMPLE_PATH = Path(__file__).resolve().parent / "examples" / "vec3_multiply_demo.json"`
  module constant in `canvas.py`.
- `load_example()` shrinks to a single line:
  `self.load_from_file(DEFAULT_EXAMPLE_PATH)`. The hand-built version is
  deleted.

## Tests (`tests/test_node_editor.py`)

- Add a session-scoped fixture that redirects `QSettings`'s default storage
  to a temp directory (`QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory...))`)
  and sets `QCoreApplication.setOrganizationName("NCCA")` /
  `setApplicationName("MathNodeEditor")` once, so every bare
  `MathNodeWindow(load_example=...)` call already in the suite stays valid
  (no `recentFile` yet → falls back to the bundled demo, same
  `Vec3(4, 10, 18)` result) without touching the real user's settings file or
  needing a `settings=` kwarg threaded through ~40 existing call sites.
- `test_save_graph_button_writes_a_json_file`,
  `test_load_graph_button_replaces_the_current_graph`,
  `test_load_graph_button_reports_a_malformed_file_instead_of_crashing`
  rewritten against `window.action_save`/`window.action_open` (still
  monkeypatching `QFileDialog`) instead of hunting a `QPushButton` by text.
- New tests: File menu actions exist with the right shortcuts; `Save`
  writes to `current_file` without prompting once one is set, `Save As`
  always prompts; startup restores a `recentFile` from `QSettings` when
  present and falls back to the bundled demo when absent or missing on
  disk; `current_file`/`recentFile` survive a Save; geometry round-trips
  through `saveGeometry`/`restoreGeometry`; `_confirm_discard_changes`
  returns `True` untouched, and Save/Discard/Cancel each do the right thing,
  driven by monkeypatching `QMessageBox.question`; dragging a node,
  editing a value, and adding a palette node each flip `canvas.modified`
  True while loading a file leaves it `False`.

## Docs (`MathNodeEditor/README.md`)

- Replace the "starts with two editable Vec3 nodes..." opening line with a
  description of the new startup behaviour (reopens the last file used, or
  `examples/vec3_multiply_demo.json` the first time / if that file's gone).
- Replace the `Save graph.../Load graph...` paragraph with the File menu
  and its shortcuts.
- Mention `examples/vec3_multiply_demo.json` alongside the other three
  example files.
- Refresh the screenshot once the menu bar is in place.
