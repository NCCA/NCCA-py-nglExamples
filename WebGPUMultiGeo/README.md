# WebGPU Multi Geo

This demo shows how to render multiple mesh objects within a single WebGPU draw call. It demonstrates an efficient way to render a scene by consolidating geometry and using storage buffers for per-instance data like transformations and colours.

Basic diffuse shading is used, with buffers setup to have Camera (view,project,eye) and light (position and colour) values passed per frame, and mesh data instanced per mesh (model transform, normal matrix and colour)

## How It Works

The program is structured into three main classes:

1.  **`WebGPUScene`**: This is the main application window that inherits from `WebGPUWidget`. It is responsible for:
    *   Setting up the WebGPU device and context.
    *   Handling user input (mouse for FirstPersonCamera control, keyboard for toggling lights).
    *   Defining the list of objects in the scene (`scene_objects`), specifying their transformations and colours.
    *   Calling the `Pipeline's` render method each frame.

2.  **`Pipeline`**: This class manages the rendering process. Its key responsibilities include :-
    *   Creating the WebGPU render pipeline, including loading and compiling the WGSL shader.
    *   Managing GPU buffers for global data like the camera's view/projection matrices and lighting information.
    *   Using the `MeshData` class to load all the geometry and instance information into the necessary GPU buffers.
    *   Executing the render pass, which involves binding the appropriate buffers and issuing a single draw call to render all objects.

3.  **`MeshData`**: This class is central to the efficient rendering approach. It is designed to handle shared geometries:
    *   **`add_geometry(mesh_name, prim_data)`**: This method is used to load the raw vertex data for a unique mesh shape (e.g., a sphere, a cube, or a loaded model) and associate it with a name. This data is only stored once, regardless of how many times it's used in the scene.
    *   **`add_mesh(name, mesh_name)`**: This method creates an *instance* of a previously loaded geometry. Each instance is given a unique name (e.g., "light1") and refers to a geometry name (e.g., "light_sphere").
    *   **`create_buffers()`**: This method combines the data. It creates a single large vertex buffer containing all unique geometries and a storage buffer to hold the per-instance data (model matrix, normal matrix, and colour) for every object in the scene.

### Rendering Process

1.  **Setup**: The `Pipeline` uses `MeshData` to load all required geometries and create instances for every object in the scene. `MeshData` then creates two main buffers: one for all the vertex data and another for all the instance data.
2.  **Update**: Each frame, `WebGPUScene` updates the camera and light information. It then loops through its `scene_objects` list and calls `pipeline.update_mesh_storage_buffer()` for each object. This updates the object's transformation and colour in the host-side (CPU) storage array within `MeshData`.
3.  **Render**: `pipeline.render()` is called. This method first uploads the entire updated storage array to the GPU storage buffer. It then begins a render pass, binds the consolidated vertex and storage buffers, and issues a single `draw()` command.
4.  **Shader Execution**: The vertex shader uses `instance_index` (a built-in variable) to look up the correct transformation and colour for each instance from the storage buffer, allowing a single set of geometry to be rendered in multiple positions with different appearances.

## References

- [WebGPU Fundamentals — Storage Buffers](https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html) — per-instance transforms/colours in a storage buffer indexed in the shader.
- [WebGPU Specification](https://www.w3.org/TR/webgpu/) — bind groups, buffer usage flags and draw calls.
- [LearnOpenGL — Instancing](https://learnopengl.com/Advanced-OpenGL/Instancing) — the same consolidate-the-draw-calls idea in its OpenGL form.
