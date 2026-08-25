# NGLWidgetsOpenGL

![](PySideGUI.png)

This is the same Qt Designer / PySide6 GUI as [PySideGUIOpenGL](../PySideGUIOpenGL) but built from the custom widgets in `ncca.ngl.widgets` (RGBColourWidget, TransformWidget, LookAtWidget and PerspectiveWidget) rather than individual spin boxes and buttons.

The widgets are promoted in the .ui file, so a QUiLoader subclass creates each one by class name as the ui is loaded, then their signals are connected to the scene in the usual way.

```bash
uv run GUIDemos/NGLWidgetsOpenGL/main.py
```

## References

- [Qt for Python documentation](https://doc.qt.io/qtforpython-6/) — PySide6 reference.
- [Qt for Python — Signals and Slots](https://doc.qt.io/qtforpython-6/tutorials/basictutorial/signals_and_slots.html) — the `@Signal` / `@Slot` decorators used here.
- [Qt Designer Manual](https://doc.qt.io/qt-6/qtdesigner-manual.html) — building the `.ui` file, including promoting custom widgets.
- [QOpenGLWidget (Qt for Python)](https://doc.qt.io/qtforpython-6/PySide6/QtOpenGLWidgets/QOpenGLWidget.html) — embedding a GL viewport inside a widget layout.
