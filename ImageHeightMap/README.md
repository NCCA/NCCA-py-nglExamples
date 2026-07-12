# ImageHeightMap

Builds a terrain mesh from an image: each vertex's height comes from the
pixel's red channel, and the vertex is coloured by the pixel's RGB. Uses
`FractalMap.bmp`, downsampled to a max 200x200 grid for interactivity.

## Controls
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset

## References

- [LearnOpenGL — Height map](https://learnopengl.com/Guest-Articles/2021/Tessellation/Height-map) — building a terrain mesh by displacing grid vertices by image intensity.
- [Heightmap — Wikipedia](https://en.wikipedia.org/wiki/Heightmap) — background on heightfield terrain representation.
