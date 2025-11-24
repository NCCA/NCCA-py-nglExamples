

At present the buffer is setup but we need to add 1 buffer per transform

To do this I need to

1.

Update line 101 in pipeline.py to have something like

```
# allocate space for 3 meshes
        num_meshes = 3
        buffer_size = 256 * num_meshes
        vertex_uniform_buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="vertex_uniform_data",
        )
```

2. update_uniform_buffers with offset


Ideally need to identify a way to do this dynamically on start up
