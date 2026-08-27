# 2026-08-27 -- ImageMaze troll would not move

## Goal

`ImageMaze` was reported as not working: you cannot move the troll. Check it
against the C++ original in `/Users/jmacey/teaching/NGL9Demos/ImageMaze`.

## What was wrong

The walls and the actor were living in mirrored copies of the maze.

`Maze.wall_cells()` put image row `y` at world z of `y - height / 2`, whilst
`actor_world_position()` put actor grid z at `height / 2 - z`. Opposite signs.
`move_actor()` hid that by testing pixel row `height - next_z`, which does keep
the collision test agreeing with where the troll is *drawn* — but it means grid
(2, 2), the default start lifted straight from the C++, resolves to row 18 of
`small.png`:

```
row 18: #.#......###.......#
              ^ column 2
```

A wall. The troll spawned inside a cube, north and south were both blocked, and
only left and right did anything — and even those you could not see, because the
troll was buried in the wall it was standing in.

The C++ is not wrong, it just does not transcribe. `Actor::move` indexes
`getColour(m_posX, height - m_posZ)` and that works there because `ngl::Image`
flips the image on load, so index `height - posZ` reaches source row `posZ - 1`.
`Map::draw` reads through the same flip, so the two agree. PIL and numpy hand
you the rows top-first with no flip, so copying the index across breaks it.

Worth saying that camera 2 is fine — it looked broken whilst I was testing
because `grabFramebuffer()` kept handing back the previous frame. The grey
filling the bottom half of that view is (146, 146, 146), the wall colour from
`small.png`, not the (77, 77, 77) ground. It is a wall two cells in front of the
troll's face, seen from an eye height level with the top of the cubes.

## The fix

Two lines in `maze_scene.py`. Walls take the same mapping as the actor:

```python
z=half_height - float(image_y),
```

which makes the actor's grid position its pixel position, so the collision test
indexes the image directly and the `height -` fudge goes:

```python
if not maze.is_open(next_x, next_z):
```

Orientation still matches the C++ (the whole maze sits one unit off it, from the
`for z = -halfZ; z < halfZ` loop, which nobody will notice). Both backends share
`maze_scene.py`, so the OpenGL and WebGPU versions are fixed together.

## Files changed

- `ImageMaze/maze_scene.py`
- `ImageMaze/tests/test_maze_scene.py`
- `ImageMaze/ImageMaze.png` -- reshot, the old one had the troll in the wall

Three tests were added first and failed for the right reason: the start cell is
open, no wall shares the actor's world position, and all four directions move
from the default start. Two existing tests encoded the old mirrored mapping and
were corrected.

## Commands run

```bash
uv run pytest ImageMaze/tests -q        # 18 passed
uv run pytest -q                        # 837 passed
uv run ruff check ImageMaze/
uv run ruff format --check ImageMaze/
uv run ImageMaze/main.py --smoketest 800
uv run ImageMaze/main_webgpu.py --smoketest 800
```

Both windows were also driven with real `QTest.keyClick` arrow presses to check
the keys reach the window rather than only that the model is right:

```
start:            ActorState(x=2, z=2, rotation=0.0)
after Key_Up:     ActorState(x=2, z=1, rotation=0.0)
after Key_Up:     ActorState(x=2, z=1, rotation=0.0)   # row 0 is the border
after Key_Right:  ActorState(x=3, z=1, rotation=270.0)
after Key_Down:   ActorState(x=4, z=2, rotation=180.0)
```
