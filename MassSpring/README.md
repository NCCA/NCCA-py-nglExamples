# Mass Spring Chain (RK4)

![](MassSpring.png)

A PyNGL / PySide6 port of the C++ [MassSpring](https://github.com/NCCA/NGL9Demos)
demo I use in the lectures, using RK4 integration as described on
[gafferongames](http://gafferongames.com/game-physics/). The spring force is
the usual `F = -k(|x|-d)(x/|x|) - bv`.

The one change from the C++ is that the single spring has become a chain. Leave
Masses at 2 and this is the original demo, with the original's values (a
vertical spring from (0,2,0) to (0,-2,0) pinned at the top, k=5, damping=2,
resting length 1, dt=0.01); wind the count up and you get a rope. Start and End
place the two ends and the masses in between are spaced evenly along them, so
the chain is a generalisation rather than a different demo. Gravity is the
other addition -- the C++ has none, and with it off you get exactly the old
behaviour.

To run it:

```bash
cd MassSpring
uv run main.py
```

Left mouse drags a mass if you grab one and rotates the camera if you miss.
Everything else is on the panel. Pin either end, turn on gravity and watch it
swing. The cubes are the masses -- red when pinned, yellow while you are
holding one, green otherwise -- and the small blue spheres are ghosts showing
where each mass started.

Picking is done by colour: the masses are rendered flat, each in a unique
colour keyed to its index, and the pixel under the cursor is read back and
decoded. The catch on the way there is worth knowing about. A `QOpenGLWidget`
draws into a *multisampled* framebuffer, and `glReadPixels` on a multisampled
buffer is an invalid operation, so you cannot simply draw the ID pass over the
widget's own target -- which is what you can get away with in a
`QOpenGLWindow` demo like `SelectionManipulator`. So the ID pass renders into
its own single-sample framebuffer instead, which is what you want regardless:
with no antialiasing no pixel is ever a blend of two IDs decoding to a third.

A mass you are holding is kinematic -- it goes into the same fixed set as a
pinned mass, so the integrator leaves it alone rather than fighting the mouse.
Let go and it drops from a standstill. Drag a pinned mass and you move where
it is pinned to.

Worth a play: with soft springs a long chain under full gravity stretches a
very long way -- which is correct, not a bug, and is easier to see here than to
argue about on a slide. Wind k up or gravity down to pull it back into frame.

## How it is put together

The C++ gives each spring its own two endpoints and integrates the
displacement between them. That works for one spring but not for a chain -- a
mass in the middle belongs to two springs and each would integrate its own copy
of it, and the two would fight. So here the state belongs to the masses and the
springs only produce forces, which every spring touching a mass adds into the
same acceleration slot.

- `rk4.py` -- the integrator, straight from the C++ `AbstractRK4Integrator`,
  over `(N,3)` arrays rather than a single `Vec3`. No Qt, no OpenGL.
- `mass_spring.py` -- the chain and the spring force law.
- `picking.py` -- the colour ID encoding, the unproject and the ray/plane
  intersection used to drag a mass. No Qt, no OpenGL.
- `MassSpringScene.py` -- the drawing and the ID pass.
- `main.py` -- the GUI, using `Vec3Widget` from `ncca.ngl.widgets` for the two
  ends.

Because the physics has no Qt or OpenGL in it, the tests are headless and check
the maths rather than the pixels -- the integrator goes against the analytic
simple harmonic oscillator, and the chain is checked for sagging, damping
bleeding energy away and pinned masses staying put:

```bash
uv run pytest MassSpring/tests/
```
