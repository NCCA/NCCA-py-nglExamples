# BlankPySide6NGL / using_mixin

![](BlankPySide6NGL.png)

These demos are a starting point for creating a PySide6 application with NGL support.

It has a basic full screen OpenGL window and simple mouse and keyboard controls.

The BlankPySide6NGL demo has all of the mouse / keyboard controls implemented in the file, whilst the using_mixin version imports the controls from ncca.ngl instead as it is a more modular approach and this code is repeated a lot in my demos.

## References

- [QOpenGLWindow (Qt for Python)](https://doc.qt.io/qtforpython-6/PySide6/QtOpenGL/QOpenGLWindow.html) — the window class providing `initializeGL` / `paintGL` / `resizeGL`.
- [PyNGL documentation](https://ncca.github.io/PyNGL/) — the `ncca.ngl` library these demos are built on.
- [LearnOpenGL — Hello Window](https://learnopengl.com/Getting-started/Hello-Window) — the same context/loop concepts in raw form.
