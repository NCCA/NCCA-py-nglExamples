# 2026-08-27 -- QMLWebGPUOverlay drew no teapot

## Goal

`GUIDemos/QMLWebGPUOverlay` opened, laid out its floating QML panels, and showed
a flat grey rectangle where the teapot should have been.

## What was wrong

The same fault as `WebGPUCompute/SpatialHash3D` earlier today (`0a64470`), in
the last demo still carrying it.

`PyNGLScene.paintWebGPU()` updated the uniforms and called
`TeapotPipeline.paint()`, then stopped. Nothing copied the resolved colour
target back into `self.frame_buffer`, which is the numpy array `QPainter`
actually blits. Since the PyNGL Version1.0 merge that copy is the subclass's
job rather than `WebGPUWidget.paintEvent`'s — the `paintWebGPU` docstring says
implementations should end by calling `_update_colour_buffer()` — and this one
never did.

No exception, no log line: the render pass ran perfectly well into textures
nobody read.

I checked the other 46 demos with a `paintWebGPU` and they all make the call.
This was the last one.

## Evidence

Rather than guess from the source, I ran the real `MainWindow` under a timer and
printed the frame buffer. Before:

```
frame_buffer: (1600, 2400, 4) uint8
  min/max/mean: 0 0 0.0
  unique colours: [[0 0 0 0]]
```

After:

```
  min/max/mean: 26 255 146.1
```

and dumping the array to a PNG shows the teapot.

## The fix

Five lines at the end of `paintWebGPU`, four of them comment:

```python
# Copy the resolved colour target back into the numpy frame buffer the
# widget blits with QPainter. Since ncca-ngl Version1.0 this is the
# subclass's job, not paintEvent's - without it the render pass fills
# its textures and nothing ever reaches the screen.
self._update_colour_buffer()
```

`pipelined_readback` is left off. This scene only repaints in response to a
panel edit or a mouse event, so presenting a frame late would mean a one-off
change rendering to the texture and not appearing until whatever unrelated
event happens to trigger the next paint.

## Files changed

- `GUIDemos/QMLWebGPUOverlay/PyNGLScene.py`
- `GUIDemos/QMLWebGPUOverlay/QMLWebGPUOverlay.png` — the old one was a
  screenshot of the empty viewport

## Commands run

```bash
uv run GUIDemos/QMLWebGPUOverlay/main.py --smoketest 1500
uv run ruff check GUIDemos/QMLWebGPUOverlay
uv run ruff format --check GUIDemos/QMLWebGPUOverlay
uv run pytest                   # 834 passed
```

The screenshot is composed in-process from `scene.frame_buffer` and
`overlay.grabFramebuffer()`, painted one over the other. I first tried
`screencapture -R` with the window's frame geometry, which grabs that region of
the desktop regardless of what is in front of it and caught a browser window
instead; that file was deleted immediately. Grabbing the app's own two surfaces
is both accurate and incapable of picking up anything else on screen.

## Not changed

Two things noticed and left alone, since neither is why the teapot was missing.

`OverlayQuickWidget` forwards any click that misses a panel down to the scene
with `QApplication.sendEvent`, but `PyNGLScene` implements no mouse handlers and
neither does `WebGPUWidget`, so those events arrive nowhere. The camera is
driven entirely by the Look At panel. Either the forwarding is vestigial and
should go, or the scene wants the usual arcball handlers — worth deciding, but
it is a feature question rather than a bug.

Shutting the demo down logs a stream of `TypeError: Cannot read property 'x' of
null` from `Vec3Widget.qml`, `LookAtWidget.qml`, `RGBColourWidget.qml` and
`PerspectiveWidget.qml`. It is teardown noise — the Python model objects are
collected while the QML bindings still reference them — and it comes from
`ncca.ngl.qml`, so if it is worth silencing it belongs in the PyNGL repo, not
here.
