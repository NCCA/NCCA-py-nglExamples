# 2026-08-27 -- ImageMaze troll never faced the way it walked

## Goal

Third and last of today's ImageMaze reports. The troll moves, and the keys now
agree with the screen, but the model never turns to face where it is going.

## What was wrong

The rotations in `Direction` assume the troll mesh looks down +z. It looks down
+x.

The demo turns the model with `Mat4().rotate_y(actor.rotation)` and nothing
else, so the rotation is only correct if you know which way the mesh points to
begin with. Nobody had checked. Rendering `Prims.TROLL` at rotation 0 from a
camera on +z gives a clean side profile with the face towards screen right, and
screen right from there is +x:

![](../../ImageMaze/ImageMaze.png)

Everything was therefore a quarter turn out, in the same direction for all four
compass points, which is why it reads as "never faces the right way" rather
than "one of the directions is wrong".

`actor_forward()` hid how fragile this was. It was a dict keyed on the rotation
value, so the model rotation and the troll camera were two separate hand-written
tables that had to be kept in step by hand -- change one and the camera silently
looks somewhere the model is not.

## The fix

The mesh's facing is now stated once, where it can be read:

```python
# The NGL troll mesh looks down +x in model space, not the +z you might assume,
# so every rotation below is its compass bearing turned a quarter turn.
TROLL_FACING = (1.0, 0.0, 0.0)
```

and every rotation drops by 90 degrees -- north 270, south 90, east 180, west 0.
`actor_forward()` now rotates `TROLL_FACING` by the actor's rotation instead of
looking the answer up, so the troll camera cannot end up pointing anywhere other
than where the model is looking.

`ActorState.rotation` defaulted to `0.0`, which used to mean north and now means
west, so it defaults to `Direction.NORTH.value[2]` and will stay pointing north
whatever the numbers become.

## Files changed

- `ImageMaze/maze_scene.py`
- `ImageMaze/tests/test_maze_scene.py`
- `ImageMaze/ImageMaze.png` -- reshot, the troll starts on a different bearing

## Commands run

```bash
uv run pytest -q                        # 839 passed
uv run ruff check ImageMaze/
uv run ruff format --check ImageMaze/
uv run ImageMaze/main.py --smoketest 800
uv run ImageMaze/main_webgpu.py --smoketest 800
```

The new test rotates `TROLL_FACING` through PyNGL's own `Mat4.rotate_y` and
checks it lands on the world step the actor actually took, so the test cannot
disagree with the demo about which way `rotate_y` turns -- my first attempt
hand-rolled the matrix and got the sign wrong, which is exactly the mistake the
demo made.

Two older tests had the previous rotations baked in and were corrected. Then the
running app was driven with real key presses, printing the world step against
the direction the model is looking:

```
key       actor                                travelled          model faces
Key_Down  ActorState(x=2, z=3, rotation=90.0)  (0.0, 0.0, -1.0)   (0.0, 0.0, -1.0)
Key_Right ActorState(x=3, z=3, rotation=180.0) (-1.0, 0.0, 0.0)   (-1.0, 0.0, 0.0)
Key_Right ActorState(x=3, z=3, rotation=180.0) (0.0, 0.0, 0.0)    (-1.0, 0.0, 0.0)
Key_Up    ActorState(x=3, z=2, rotation=270.0) (0.0, 0.0, 1.0)    (0.0, 0.0, 1.0)
Key_Left  ActorState(x=2, z=2, rotation=0.0)   (1.0, 0.0, 0.0)    (1.0, 0.0, 0.0)
```

The third row is a move into a wall: no travel, and the troll keeps the bearing
it already had.

## Note for the C++ demo

`NGL9Demos/ImageMaze` has the same quarter turn -- its `Actor::ROTATION` enum is
the same four numbers and it draws the same mesh. Not changed here, but it is
wrong there too.
