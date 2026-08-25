import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    from ncca.ngl import Mat4, Vec3, look_at, perspective

    return Mat4, Vec3, look_at, np, perspective


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Uniform Buffers

    It is common to create buffers in shaders to pass in uniform data such as the MVP matrix and other information. To do this in WebGPU we can use numpy. The following example shows how we can create a simple buffer. 
    """
    )


@app.cell
def _(np):
    vertex_uniform_data = np.zeros(
        (),
        dtype=[
            ("MVP", "float32", (4, 4)),
            ("model_view", "float32", (4, 4)),
            ("normal_matrix", "float32", (4, 4)),  # need 4x4 for mat3
            ("colour", "float32", (4)),
            ("padding", "float32", (12)),  # to 256 bytes
        ],
    )
    print(vertex_uniform_data.shape)
    print(vertex_uniform_data.nbytes)
    return (vertex_uniform_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can set the data in this buffer using the following methods """)


@app.cell
def _(Mat4, Vec3, look_at, perspective, vertex_uniform_data):
    project = perspective(45.0, 1.0, 0.1, 100)
    view = look_at(Vec3(2, 2, 2), Vec3(0, 0, 0), Vec3(0, 1, 0))
    model_tx = Mat4()

    MV = view @ model_tx
    MVP = project @ MV
    normal_matrix = MV.copy()  # need to copy else we get the same matrix
    normal_matrix = normal_matrix.inverse().transposed()

    vertex_uniform_data["MVP"] = MVP.to_numpy()
    vertex_uniform_data["model_view"] = MV.to_numpy()
    vertex_uniform_data["normal_matrix"] = normal_matrix.to_numpy()
    vertex_uniform_data["colour"] = (1, 0, 0, 0)

    print(vertex_uniform_data)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
