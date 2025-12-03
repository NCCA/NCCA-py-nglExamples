from ncca.ngl import DefaultShader, Mat3, Mat4, Primitives, ShaderLib, Transform, Vec3


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


class SelectObject:
    def __init__(self, position):
        self.position = position
        self.selected = False
        try:
            self.colour_id = next(color_gen)
        except StopIteration:
            print("Error: Color ID generator is exhausted.")
            self.colour_id = (-1, -1, -1)

    def check_selection_colour(self, colour):
        if colour == self.colour_id:
            self.selected ^= True
            return True
        return False

    def load_matrices_to_shader(self, local_tx, global_tx, view, project):
        ShaderLib.use(DefaultShader.DIFFUSE)
        model_matrix = global_tx @ local_tx.get_matrix()
        MV = view @ model_matrix
        mvp = project @ MV
        normal_matrix = Mat3.from_mat4(MV)
        normal_matrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("MV", MV)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)

    def load_matrices_to_colour_shader(self, local_tx, global_tx, view, project):
        ShaderLib.use(DefaultShader.COLOUR)
        ShaderLib.set_uniform(
            "Colour",
            self.colour_id[0] / 255.0,
            self.colour_id[1] / 255.0,
            self.colour_id[2] / 255.0,
            1.0,
        )
        model_matrix = global_tx @ local_tx.get_matrix()
        MV = view @ model_matrix
        mvp = project @ MV
        ShaderLib.set_uniform("MVP", mvp)

    def draw(self, selection, global_tx, view, project):
        tx = Transform()
        tx.set_position(self.position.x, self.position.y, self.position.z)
        if selection:
            self.load_matrices_to_colour_shader(tx, global_tx, view, project)
        else:
            self.load_matrices_to_shader(tx, global_tx, view, project)
        if self.selected:
            Primitives.draw("teapot")
        else:
            Primitives.draw("cube")
