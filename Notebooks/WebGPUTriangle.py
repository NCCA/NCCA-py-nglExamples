import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #Web GPU triangle

    In this notebook we will enable WebGPU and create a simple triangle which will be stored in a numpy buffer and displayed using matplot lib and imshow. 

    First we need to import the modules we need.
    """
    )


@app.cell
def _():
    import numpy as np
    import wgpu
    import wgpu.utils
    from wgpu.utils import get_default_device

    return get_default_device, np, wgpu


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vertex Buffers 

    All the data rendered to the screen is created in a Vertex Buffer. We use numpy to generate our buffer. In this case it is a flat array of floats for each of the vertices, combine in the format 

    ```
    x,y,z
    r,g,b 
    ```
    For each of the triangle vertices.  Note the winding of the triangle and how the y axis is negative in this case. We will see later how the different windings effect things and are different from OpenGL.
    """
    )


@app.cell
def _(np, wgpu):
    def create_vertex_buffer(device: wgpu.GPUDevice) -> wgpu.GPUBuffer:
        """
        Create a vertex buffer.

        Parameters
        ----------
            device : wgpu.GPUDevice
                The GPU device.

        Returns
        -------
            wgpu.GPUBuffer
                The created vertex buffer.
        """
        vertices = np.array(
            [
                -0.75,
                -0.75,
                0.0,
                1.0,
                0.0,
                0.0,  # Bottom-left vertex (red)
                0.0,
                0.75,
                0.0,
                0.0,
                1.0,
                0.0,  # Top vertex (green)
                0.75,
                -0.75,
                0.0,
                0.0,
                0.0,
                1.0,  # Bottom-right vertex (blue)
            ],
            dtype=np.float32,
        )
        return device.create_buffer_with_data(
            data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    return (create_vertex_buffer,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Render Pipeline

    The render pipeline describes how our data is formatted and what shaders to use. In this case I define each shader as a simple string and create two shader modules. In WebGPU it is also possible to create a single shader and use different names for the entry points.

    ### Vertex Shader

    This shader has two inputs, a position and colour of type vec3. These are combined in the VertexIn structure, which we will need to bind in our pipeline. 

    The output of the shader will be the position (this is a build in type) and the colour value passed onto the fragment shader. In this simplest of shaders, all we do is pass the data through.

    ```glsl
    struct VertexIn {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) fragColor: vec3<f32>,
        };

        @vertex
        fn main(input: VertexIn) -> VertexOut {
            var output: VertexOut;
            output.position = vec4<f32>(input.position, 1.0);
            output.fragColor = input.color;
            return output;
        }
    ```

    ## Fragment Shader 

    This shader just outputs the input colour to the output (which will be bound to a buffer in our pipline code).

    ```glsl
    @fragment
        fn main(@location(0) fragColor: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(fragColor, 1.0); // Simple color output
        }
    ```

    Finally, this function defines the pipeline layout (in a json like structure) and in particular the stride to get to the different input elements (position and colour).
    """
    )


@app.cell
def _(wgpu):
    def create_render_pipeline(device: wgpu.GPUDevice) -> wgpu.GPURenderPipeline:
        """
        Create a render pipeline.

        Parameters
        ----------
            device : wgpu.GPUDevice
                The GPU device.

        Returns
        -------
            wgpu.GPURenderPipeline
                The created render pipeline.
        """
        vertex_shader_code = """
        struct VertexIn {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) fragColor: vec3<f32>,
        };

        @vertex
        fn main(input: VertexIn) -> VertexOut {
            var output: VertexOut;
            output.position = vec4<f32>(input.position, 1.0);
            output.fragColor = input.color;
            return output;
        }
        """

        fragment_shader_code = """
        @fragment
        fn main(@location(0) fragColor: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(fragColor, 1.0); // Simple color output
        }
        """

        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])
        return device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": device.create_shader_module(code=vertex_shader_code),
                "entry_point": "main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": device.create_shader_module(code=fragment_shader_code),
                "entry_point": "main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    return (create_render_pipeline,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Render_Triangle

    This function will render the triangle to a buffer (texture) which we can use later. To do this we need to use the pipeline we created earlier (which describes the data and shaders to use). 

    We then create a command queue and issue commands to execute our pipeline and render the data.
    """
    )


@app.cell
def _(wgpu):
    def render_triangle(
        device: wgpu.GPUDevice,
        pipeline: wgpu.GPURenderPipeline,
        vertex_buffer: wgpu.GPUBuffer,
        width: int,
        height: int,
    ) -> wgpu.GPUTexture:
        """
        Render a triangle to a texture.

        Parameters
        ----------
            device : wgpu.GPUDevice
                The GPU device.
            pipeline : wgpu.GPURenderPipeline
                The render pipeline.
            vertex_buffer : wgpu.GPUBuffer
                The vertex buffer.
            width : int
                The width of the texture.
            height : int
                The height of the texture.

        Returns
        -------
            wgpu.GPUTexture
                The rendered texture.
        """
        texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        texture_view = texture.create_view()

        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.3, 0.3, 0.3, 1.0),
                }
            ]
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.draw(3)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

        return texture

    return (render_triangle,)


@app.cell
def _(mo):
    mo.md(
        r"""Once the render has been generated we have a buffer GPU side, we need to now copy this buffer back to the "client" side so we can use it. In this case we will use a numpy array and then display it (we could do this in this function directly). """
    )


@app.cell
def _(np, wgpu):
    def copy_texture_to_buffer(
        device: wgpu.GPUDevice, texture: wgpu.GPUTexture, width: int, height: int
    ) -> np.ndarray:
        """
        Copy the texture to a buffer and return it as a NumPy array.

        Parameters
        ----------
            device : wgpu.GPUDevice
                The GPU device.
            texture : wgpu.GPUTexture
                The texture to copy.
            width : int
                The width of the texture.
            height : int
                The height of the texture.

        Returns
        -------
            np.ndarray
                The texture data as a NumPy array.
        """
        buffer_size = width * height * 4
        readback_buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        command_encoder = device.create_command_encoder()
        command_encoder.copy_texture_to_buffer(
            {"texture": texture},
            {
                "buffer": readback_buffer,
                "bytes_per_row": width * 4,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        device.queue.submit([command_encoder.finish()])

        readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        raw_data = readback_buffer.read_mapped()
        buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
        readback_buffer.unmap()

        return buffer

    return (copy_texture_to_buffer,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Render a Triangle 

    Next we execute the commands in sequence to get our triangle data in a buffer. 

    1. create a device to use. 
    2. create the pipeline
    3. create the buffer of data to be rendered (our triangle)
    4. render the triangle (into a GPU buffer)
    5. copy back to the client side.
    """
    )


@app.cell
def _(
    copy_texture_to_buffer,
    create_render_pipeline,
    create_vertex_buffer,
    get_default_device,
    render_triangle,
):
    WIDTH = 1024
    HEIGHT = 1024

    device = get_default_device()
    pipeline = create_render_pipeline(device)
    vertex_buffer = create_vertex_buffer(device)
    texture = render_triangle(device, pipeline, vertex_buffer, WIDTH, HEIGHT)
    buffer = copy_texture_to_buffer(device, texture, WIDTH, HEIGHT)
    return (buffer,)


@app.cell
def _(mo):
    mo.md(r"""Finally we can draw our triangle using the buffer""")


@app.cell
def _(buffer):
    import matplotlib.pyplot as plt

    plt.imshow(buffer)
    plt.axis("off")  # hides both x and y axes
    plt.show()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
