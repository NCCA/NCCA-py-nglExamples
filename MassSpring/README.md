# Mass Spring Chain (RK4)

![](MassSpring.png)

A PyNGL / PySide6 port of the C++ [MassSpring](https://github.com/NCCA/MassSpring)
demo I use in the lectures, using RK4 integration as described on
[gafferongames](https://gafferongames.com/categories/game-physics/). The spring
force is the usual `F = -k(|x|-d)(x/|x|) - bv`. There are links to all of the
background reading at the [bottom](#references).

The one change from the C++ is that the single spring has become a chain. Leave
Masses at 2 and this is the original demo, with the original's values (a
vertical spring from (0,2,0) to (0,-2,0) pinned at the top, k=5, damping=2,
resting length 1, dt=0.01); wind the count up and you get a rope. Start and End
place the two ends and the masses in between are spaced evenly along them, so
the chain is a generalisation gravity has also been added.

To run it:

```bash
cd MassSpring
uv run main.py
```

Left mouse drags a mass if you grab one and rotates the camera if you miss.
Everything else is on the panel. Pin either end, turn on gravity and watch it
swing. The cubes are the masses, red when pinned, yellow while you are
holding one, green otherwise. The small blue spheres are ghosts showing
where each mass started.

Picking is done by colour: the masses are rendered flat, each in a unique
colour keyed to its index, and the pixel under the cursor is read back and
decoded. The catch on the way there is worth knowing about. A `QOpenGLWidget`
draws into a _multisampled_ framebuffer, and `glReadPixels` on a multisampled
buffer is an invalid operation, so you cannot simply draw the ID pass over the
widget's own target -- which is what you can get away with in a
`QOpenGLWindow` demo like `SelectionManipulator`. So the ID pass renders into
its own single-sample framebuffer instead, which is what you want regardless:
with no antialiasing no pixel is ever a blend of two IDs decoding to a third.

A mass you are holding is kinematic, it goes into the same fixed set as a
pinned mass, so the integrator leaves it alone rather than fighting the mouse.
Let go and it drops from a standstill. Drag a pinned mass and you move the pinned
mass to the new location.

## How it works,

Each spring has its own two endpoints and integrates the displacement between them. That works for one spring but not for a chain,
where the mass in the middle belongs to two springs and each would integrate its own copy
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

Because the physics has no Qt or OpenGL in it, the tests are headless and check the maths rather than the pixelsm. The integrator goes against the analytic
simple harmonic oscillator, and the chain is checked for sagging, damping
bleeding energy away and pinned masses staying put.

```bash
uv run pytest MassSpring/tests/
```

## References

The original demo, which this is a port of:

- [NCCA/MassSpring](https://github.com/NCCA/MassSpring) -- the C++ NGL version.
  It is pulled in as a submodule of
  [NCCA/NGL9Demos](https://github.com/NCCA/NGL9Demos), which is where you will
  find the rest of the NGL9 demos.

RK4 and integration generally:

- [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
  -- the same page the C++ `AbstractRK4Integrator` cites, and the clearest
  statement of the four evaluations the code does.
- [Integration Basics](https://gafferongames.com/post/integration_basics/) --
  Glenn Fiedler on why you would bother with RK4 rather than Euler. This is the
  article the original demo is built from.
- [Differential Equation Basics](https://www.cs.cmu.edu/~baraff/sigcourse/notesb.pdf)
  -- Baraff and Witkin's SIGGRAPH course notes, if you want the ODE view of what
  the integrator is doing rather than the games one.
- [Verlet integration](https://en.wikipedia.org/wiki/Verlet_integration) -- the
  usual alternative, and what a lot of cloth and rope solvers actually use.

Mass spring systems:

- [Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law) -- where the `-kx`
  in the force comes from.
- [Spring Physics](https://gafferongames.com/post/spring_physics/) -- the
  damped spring the force law here is lifted from.
- [Particle Dynamics](https://www.cs.cmu.edu/~baraff/sigcourse/notesc.pdf) --
  Baraff and Witkin again, and the best explanation I know of why a chain wants
  its state on the particles with the springs as forces, which is exactly the
  change this port had to make.
- [Soft body dynamics](https://en.wikipedia.org/wiki/Soft-body_dynamics) -- for
  where you go next, once a chain of springs turns into cloth.
