# Voxels

Renders a block of voxels (Minecraft-style terrain) on the GPU using OpenGL
texture buffer objects. Per-voxel position, active flag and texture index are
stored in texture buffers and instanced in the shader, so the whole terrain is
drawn in a single call. An offscreen FrameBufferObject is used to pick the
voxel under the mouse cursor (index and depth), allowing interactive editing
of the terrain.

## Files

- `main.py` - main application and picking logic
- `Terrain.py` - voxel grid generation and texture buffer management
- `FrameBufferObject.py` / `TextureTypes.py` - Python port of the NGL FBO class
- `shaders/` - GLSL voxel and picking shaders
- `textures/` - voxel textures

## Controls

- Left-drag : rotate camera, Right-drag : pan, Wheel : zoom
- Arrow keys : move towards / away from and around the picked point
- `s` : remove the voxel under the cursor
- `z` / `x` : change the texture of the voxel under the cursor
- `d` : toggle debug view of the picking FBO
- `Esc` : quit

## References

- [OpenGL Wiki — Buffer Texture](https://www.khronos.org/opengl/wiki/Buffer_Texture) — texture buffer objects, used to store per-voxel position/active/texture data.
- [LearnOpenGL — Instancing](https://learnopengl.com/Advanced-OpenGL/Instancing) — drawing the whole terrain in one instanced call.
- [0fps — Meshing in a Minecraft Game](https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/) — the broader design space of voxel terrain rendering.
- [LearnOpenGL — Framebuffers](https://learnopengl.com/Advanced-OpenGL/Framebuffers) — offscreen render targets, used here for the index/depth picking pass.
