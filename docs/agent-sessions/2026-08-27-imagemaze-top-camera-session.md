# 2026-08-27 -- ImageMaze arrow keys ran backwards

## Goal

Follow-on from the actor mapping fix earlier today. With the troll finally
moving, the arrow keys turned out to drive it the wrong way: up went down,
left went right.

## What was wrong

One vector, transcribed as its opposite.

The C++ builds the overhead camera as

```cpp
m_cam.view = ngl::lookAt(ngl::Vec3(0, 30, 0), ngl::Vec3::zero(), ngl::Vec3::in());
```

and `ngl::Vec3::in()` is `(0, 0, 1)` -- `Vec3.h:327`. The Python had

```python
Vec3(0.0, 0.0, -1.0)
```

which is `ngl::Vec3::out()`. The names are easy to read the wrong way round;
`in()` is the one pointing along positive z.

Looking straight down, that up vector is the only thing fixing the roll, so
picking the opposite one spins the whole view through 180 degrees. Two things
follow, and only the second one gets reported:

- the maze renders upside down and back to front, which nobody notices because
  a maze looks like a maze either way up
- every arrow key moves the troll the opposite way to the one pressed

With `in()`, screen right is `-x` and screen up is `+z`. The actor sits at
`(width / 2 - x, height / 2 - z)`, so pressing Up (grid row `-1`) raises world
z and the troll goes up the screen. As a bonus the maze now renders in the same
orientation as the source PNG, which makes the demo far easier to follow
against the image it was built from.

## The fix

The camera moved into `maze_scene.py` as `top_view()`, because both demos had
their own copy of it and both were wrong the same way:

```python
def top_view() -> Mat4:
    return look_at(Vec3(0.0, 30.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0))
```

`main.py` and `main_webgpu.py` both call it now, so they cannot drift apart
again.

## Files changed

- `ImageMaze/maze_scene.py`
- `ImageMaze/main.py`
- `ImageMaze/main_webgpu.py`
- `ImageMaze/tests/test_maze_scene.py`
- `ImageMaze/ImageMaze.png` -- reshot, the view is the other way up now

## Commands run

```bash
uv run pytest -q                        # 838 passed
uv run ruff check ImageMaze/
uv run ruff format --check ImageMaze/
uv run ImageMaze/main.py --smoketest 800
uv run ImageMaze/main_webgpu.py --smoketest 800
```

The new test projects the actor's world position through `top_view()` and the
demo's own perspective, then checks the sign of the movement in NDC for each of
the four directions. It failed on the old up vector with `At index 1 diff:
-1.0 != 1.0` -- north going down the screen -- which is exactly the reported
symptom.

That is a model-level test, so it was backed up by measuring the red troll's
centroid in three real rendered frames:

```
start  (grid 10,10) screen x,y = 1025.2 719.5
NORTH  (grid 10, 9) screen x,y = 1025.2 660.4  -> dx  -0.1 dy -59.1
EAST   (grid 11,10) screen x,y = 1084.4 719.5  -> dx +59.2 dy  -0.0
```

North up, east right.
