# GimbalLock

![GimbalLock](GimbalLock.png)

Euler angles vs quaternions, side by side, so you can watch gimbal lock
happen rather than just take my word for it. Companion to `QuatSlerp`.

The left half is the classic three-ring rig: an outer ring turning about
world Z (yaw), carrying a ring turning about its own Y (pitch), carrying a
ring turning about its own X (roll), carrying a little cube aeroplane. The
right half drives an identical aeroplane straight off a quaternion built
from the same three angles, with no rings at all.

## The maths

Nesting three single-axis rotations means the second one is measured
relative to the *result* of the first, and the third relative to the
result of the first two — that is the whole point of an Euler rig, and
also its one flaw. At pitch = ±90° the outer (yaw) ring and inner (roll)
ring end up spinning about the same world axis. One whole degree of
freedom vanishes: turning "yaw" from that pose does exactly what turning
"roll" does. That is gimbal lock, and it is a property of the
three-angle *representation*, not of the underlying rotation — which is
why the quaternion side never notices anything. It was never built by
decomposing the rotation into three sequential single-axis turns, so
there is no sequence for two of those turns to collapse into.

`rotation_maths.py` is the numpy-only module behind the demo:
`euler_to_mat` composes the three axis rotations explicitly (`rotate_x
@ rotate_y @ rotate_z`, X applied first — read the docstring, the exact
order is the lesson), `is_gimbal_locked`/`lost_dof_axis` flag the locked
pose, and `euler_to_quat`/`quat_to_mat` build the same orientation via
`ncca.ngl.Quaternion` for comparison. The tests pin down the classic
symptom directly: at pitch = 90°, two different (roll, yaw) pairs that
share the same `roll - yaw` produce the identical matrix — a whole line
of "different" angle triples describing one orientation.

## When Euler angles are still fine

Gimbal lock only bites when all three axes need independent, arbitrary
rotation and you interpolate or drive them near ±90° pitch. Single-axis
rig rotation (a hinge, a wheel), small perturbations away from a locked
pose, or values you only ever inspect/scrub in a UI rather than compose,
are all fine with Euler angles — they are readable in a way quaternions
never quite are, which is worth something. Reach for quaternions once you
need to interpolate between two arbitrary orientations (`QuatSlerp`) or
drive a rig through poses that pass near the pole.

## Controls

- `X` / `Y` / `Z` : nudge roll / pitch / yaw by +5° (hold Shift for -5°)
- `G` : scripted "watch the DOF vanish" — pitch ramps to 90°, then yaw
  wobbles while roll sits still; on the left the aeroplane barely reacts,
  on the right the quaternion just keeps working
- `Space` : reset all three angles to zero
- Left-drag : orbit, Right-drag : pan, Wheel : zoom (both halves share the
  same camera)
- `w`/`s` : wireframe / solid fill
- `Esc` : quit

## Running it

```bash
uv run GimbalLock/main.py
uv run pytest GimbalLock/tests
```

## References

- [Gimbal lock (Wikipedia)](https://en.wikipedia.org/wiki/Gimbal_lock)
- K. Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985 — [ACM](https://dl.acm.org/doi/10.1145/325165.325242)
- [Quaternion (songho.ca)](https://www.songho.ca/math/quaternion/quaternion.html)
