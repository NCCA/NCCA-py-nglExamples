# Selection and Manipulation

![](SelectionManipulator.png)

A demo of DCC style object selection and manipulation, in the spirit of Maya
or Houdini. A small scene of objects can be picked with the mouse and
transformed with visual Translate / Rotate / Scale gizmos.

```bash
uv run main.py          # or ./main.py
uv run main.py --debug  # re-raise exceptions swallowed by Qt event handlers
```

## Controls (Maya-style)

| Input                | Action                                                     |
| -------------------- | ---------------------------------------------------------- |
| `Q`                  | Select mode (gizmo hidden)                                 |
| `W`                  | Translate mode (arrows)                                    |
| `E`                  | Rotate mode (rings)                                        |
| `R`                  | Scale mode (boxes)                                         |
| Left click           | Select the object under the cursor (replaces selection)    |
| `Ctrl` + click       | Toggle an object in / out of the selection (multi-select)  |
| Drag an axis handle  | Transform **all** selected objects along that axis         |
| Drag the centre cube | Free screen-plane move (translate) / uniform scale (scale) |
| `Alt` + LMB drag     | Tumble the camera                                          |
| `Alt` + RMB drag     | Pan the camera                                             |
| Mouse wheel          | Dolly in / out                                             |
| `Space`              | Reset the camera                                           |
| `Escape`             | Quit                                                       |

## How it works

### SelectionObject

Every pickable object inherits from the abstract `SelectionObject`
(`SelectionObject.py`) and only has to implement `draw_geometry()`. The base
class owns the transform (position / rotation / scale), the display colour,
and a unique **colour ID** handed out by a generator. Selected objects are
drawn twice: once with the solid diffuse shader, then again as a white
wireframe overdraw (`glPolygonMode(GL_LINE)` with a negative
`glPolygonOffset` so the lines sit in front of the filled surface without
z-fighting).

### Colour-ID picking

On click the scene is redrawn (but never presented) with every object flat
shaded in its ID colour on a black background, and the gizmo handles in
reserved ID colours drawn on top with the depth buffer cleared. A small
9x9 pixel block under the cursor is read back with `glReadPixels`; handle
colours are checked first (so handles stay grabbable in front of geometry),
then object IDs. The block gives a few pixels of slop, which makes the thin
gizmo handles much easier to grab.

### The manipulator

`Manipulator.py` draws the gizmo at the centroid of the selection. All
handles are built from stock primitives (cylinder shaft + cone head for
translate, torus rings for rotate, shaft + cube for scale), authored once
along +Y and instanced onto the three axes with a rotation matrix. The
gizmo is scaled by its view-space depth so it keeps a constant on-screen
size, and drawn after clearing the depth buffer so it always renders on top,
exactly like a DCC viewport.

Dragging works in _screen space_:

- **Translate** :- the gizmo axis is projected to the screen; mouse motion is
  dotted with the projected axis direction and divided by its pixels-per-
  world-unit length, giving a world-space delta along the axis.
- **Scale** :- the same projected motion drives a per-axis multiply factor.
- **Free translate** (center cube) :- mouse motion is converted to a world
  delta in the camera's screen plane, using the model-view basis vectors that
  map to screen right / up at the pivot, so the selection tracks the cursor.
- **Uniform scale** (center cube) :- the change in the mouse's distance from
  the projected pivot drives a single factor applied equally to all axes.
- **Rotate** :- the angle of the mouse around the projected gizmo centre is
  tracked (`atan2`), and its per-event change is applied around the handle
  axis, with the sign flipped when the axis points away from the camera so
  the rotation always follows the mouse.

All deltas are applied to _every_ selected object, so a multi-selection
moves, scales, and rotates together (rotation and scale are applied about
each object's own pivot).

## Possible extensions

- Rotating multiple objects about the shared pivot rather than their own.
- Marquee (rubber-band) selection.

## References

- [opengl-tutorial — Picking with an OpenGL hack](http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/picking-with-an-opengl-hack/) — the colour-ID picking pass used for objects and gizmo handles.
- [glReadPixels — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glReadPixels.xhtml) — the (stalling) readback; see [`RayPickingSelection`](../RayPickingSelection) for the analytic alternative.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) — a widely used open-source implementation of the same Maya-style translate/rotate/scale gizmos.
