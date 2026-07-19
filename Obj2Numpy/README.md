# Obj2Numpy

![](Obj2Numpy.png)

A command line tool (not a graphical demo) that converts triangulated Wavefront
OBJ meshes into NumPy `.npy` files for fast loading in other demos.

Each triangle vertex is expanded into an interleaved record of 8 `float32`
values — position, normal and UV:

```
x y z  nx ny nz  u v
```

The V coordinate is flipped (`1 - v`) to match OpenGL texture conventions.
Meshes without normals or UVs are padded with zeros. The output is written
next to the input file as `<name>.npy` and can be loaded with `np.load()` and
passed straight to a VAO or GPU buffer.

## Usage

```bash
uv run Obj2Numpy/Obj2Numpy.py mesh.obj [more.obj ...]
```

One `.npy` file is written per input OBJ; the mesh must be triangulated.
