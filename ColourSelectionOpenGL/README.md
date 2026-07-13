## Colour Selection

![](ColourSelect.png)

Picking of objects using Colour values based on this [post](https://moddb.fandom.com/wiki/OpenGL_Selection_Using_Unique_Color_IDs).

### Implementation Details

When the mouse is clicked we use glReadPixels to read the color of the pixel under the mouse cursor. This color is then compared to the colors assigned to each object in the scene. If a match is found, the corresponding object is selected.

This is not the fastest method, but it is simple and easy to implement. All colors are generated using a single generator instance.

```
def color_id_generator():
    """
    A generator that yields unique, sequential RGB color IDs (0-255).
    """
    # The maximum number of unique colors is 256*256*256
    max_colors = 1 << 24

    for i in range(max_colors):
        # Use bitwise operators to derive R, G, B from a single integer.
        # This is cleaner and faster than manual nested loops.
        r = i & 0xFF
        g = (i >> 8) & 0xFF
        b = (i >> 16) & 0xFF
        yield (r, g, b)


# 1. Create a single, shared instance of the generator
color_gen = color_id_generator()
```

At present the background colour is not taken into account (128,128,128)

## References

- [opengl-tutorial — Picking with an OpenGL hack](http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/picking-with-an-opengl-hack/) — the classic colour-ID picking technique.
- [glReadPixels — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glReadPixels.xhtml) — the readback call used to fetch the pixel under the cursor (note it stalls the pipeline).
- See [`RayPickingSelection`](../RayPickingSelection) and [`WebGPUComputePicking`](../WebGPUComputePicking) in this repo for alternatives that avoid the readback stall.
