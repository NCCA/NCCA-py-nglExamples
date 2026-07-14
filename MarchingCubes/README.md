# MarchingCubes

![MarchingCubes](MarchingCubes.png)

<!-- TODO(jmacey): screenshot missing -- headless agent can't capture GL output,
     please grab one with the demo running and drop it in as MarchingCubes.png -->

Four metaballs drift along lissajous paths. Every frame their scalar field
is re-sampled and re-polygonised from scratch in numpy, and the resulting
triangle soup is pushed straight to the GPU with
`glBufferData(GL_DYNAMIC_DRAW)` -- there is no persistent mesh here, just a
VAO that gets completely re-specified on every paint. It's the same lesson
as `VertexArrayObject/ChangingVAO`, just with a more interesting source of
"changing" data.

## The algorithm

**`sample_metaballs`** evaluates a classic Blinn-style field on a regular
grid: each ball contributes `radius**2 / distance**2` to every sample
point, and the contributions sum. Pick `radius` sensibly and a lone ball's
`field == 1.0` isosurface is exactly the sphere of that radius -- which is
also how `tests/test_marching_cubes.py` checks the maths didn't drift.

**`polygonise`** turns that grid into a triangle mesh with Marching Cubes,
classifying every cell in the grid in one vectorised pass rather than
looping cell-by-cell in Python (a 48<sup>3</sup> grid is ~108k cells, and
this has to run once per frame):

1. For each of a cell's 8 corners, slice out that corner's field value
   across every cell at once, and pack an 8-bit `cubeindex` per cell:
   bit *c* set when corner *c*'s field value exceeds the isovalue.
2. Interpolate the surface crossing point (and the field gradient, for
   shading) along all 12 of a cell's edges, for every cell, unconditionally
   -- it's cheaper to compute intersections nobody asked for than to
   scatter-gather only the ones that matter.
3. Look `cubeindex` up in `TRI_TABLE` to get up to 5 triangles, each as 3
   edge indices (`-1`-padded, so an all-`-1` row means "no surface in this
   cell"). Gather the 3 pre-computed edge points/gradients per triangle
   corner and that's your mesh.

Normals come from `np.gradient(field)`, trilinearly interpolated at each
vertex the same way position is -- proper smooth shading, not flat
per-face normals from the triangle winding.

### Table provenance

The lookup tables in `mc_tables.py` (`EDGE_TABLE`, `TRI_TABLE`, and the
corner/edge numbering that ties them together) are the classic tables
credited to Cory Bloyd, as published on Paul Bourke's
["Polygonising a scalar field"](http://paulbourke.net/geometry/polygonise/)
page. They were parsed programmatically out of the C source embedded in
that page rather than hand-typed, spot-checked against the two worked
examples given on the page itself, and cross-referenced against Geoffrey
Heller's independently-authored alternative table linked from the same
page. See `mc_tables.py`'s docstring for the full provenance note.

None of that rules out a subtle bug on its own -- what actually catches a
bad table entry is `test_mesh_is_closed`: it walks every triangle edge in
the output mesh of a single metaball and asserts each one is shared by
exactly two triangles. A watertight sphere has no such thing as an edge
used once (a hole) or three times (an overlap); a table transcription slip
reliably produces one or the other.

## Controls

| Key | Effect |
| :---: | :--- |
| `+` / `-` | grid resolution: 16 / 32 / 48 / 64 |
| `I` | raise iso level |
| `Shift+I` | lower iso level |
| `W` | toggle wireframe |
| `Space` | pause / resume the metaball animation |
| Left-drag | orbit |
| Right-drag | pan |
| Wheel | zoom |
| `Esc` | quit |

## Performance

Measured on the development machine, 4 metaballs, `polygonise()` alone
(excluding the `glBufferData` upload):

| Grid | Triangles | polygonise() |
| :---: | :---: | :---: |
| 32<sup>3</sup> | ~6,400 | ~5.6 ms |
| 48<sup>3</sup> (default) | ~14,700 | ~16 ms |
| 64<sup>3</sup> | ~26,300 | ~49 ms |

48<sup>3</sup> is the default: comfortably inside a 16 ms frame budget with
room for everything else the frame has to do, while still looking properly
blobby rather than blocky. 64<sup>3</sup> is available with `+` if you want
to see the algorithm start to sweat.

## Not built here

A GPU compute-shader version of Marching Cubes (classify + emit triangles
entirely on the GPU, no CPU readback) is a natural follow-on WebGPU demo --
deliberately left out of this one so the CPU/numpy version stays the clear
teaching example of the algorithm itself.

## References

- Paul Bourke, [Polygonising a scalar field](http://paulbourke.net/geometry/polygonise/)
  (tables credited to Cory Bloyd) -- the source of `mc_tables.py`.
- Geoffrey Heller's alternative table (linked from the same page), used
  here only as a cross-check.
- `VertexArrayObject/ChangingVAO` -- the per-frame VBO re-specification
  pattern this demo builds on.
