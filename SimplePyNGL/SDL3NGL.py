#!/usr/bin/env -S uv run --script
"""
A template for creating a PySDL3 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It mirrors the functionality of the BlankPySide6NGL example using the SDL3 library.
"""

import sys

import numpy as np
import OpenGL.GL as gl
import sdl3
from ngl import (
    DefaultShader,
    Mat3,
    Mat4,
    Primitives,
    ShaderLib,
    Vec3,
    look_at,
    perspective,
)

PBR_SHADER = "pbr"


class Scene:
    """
    Encapsulates the scene's state, rendering, and input handling logic.

    This class holds all the necessary attributes for camera control, transformations,
    and user input state, similar to the MainWindow class in the PySide6 example.
    """

    def __init__(self, width: int, height: int):
        """
        Initializes the scene and sets up default parameters.

        Args:
            width: The initial width of the window.
            height: The initial height of the window.
        """
        # --- Camera and Transformation Attributes ---
        self.mouse_global_tx: Mat4 = (
            Mat4()
        )  # Global transformation matrix controlled by the mouse
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)
        self.model_position: Vec3 = Vec3()  # Position of the model in world space

        # --- Window and UI Attributes ---
        self.window_width: int = width
        self.window_height: int = height

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming

    def initialize(self) -> None:
        """
        Set up global OpenGL state.
        """
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        # Set initial projection matrix
        self.resize(self.window_width, self.window_height)
        # Set up the camera's view matrix.
        # It looks from (0, 1, 4) towards (0, 0, 0) with the 'up' direction along the Y-axis.
        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))

        # Load the PBR shader
        ShaderLib.load_shader(
            PBR_SHADER, "shaders/PBRVertex.glsl", "shaders/PBRFragment.glsl"
        )
        ShaderLib.use(PBR_SHADER)
        # We now create our view matrix for a static camera
        eye = Vec3(0.0, 2.0, 4.0)
        to = Vec3(0.0, 0.0, 0.0)
        up = Vec3(0.0, 1.0, 0.0)
        # now load to our new camera
        self.view = look_at(eye, to, up)
        ShaderLib.set_uniform("camPos", eye)
        # now a light
        self.light_pos = Vec3(0.0, 2.0, 2.0)
        # Setup the default shader material and light properties
        # these are "uniform" so will retain their values
        ShaderLib.set_uniform("lightPosition", self.light_pos)
        ShaderLib.set_uniform("lightColor", 400.0, 400.0, 400.0)
        ShaderLib.set_uniform("exposure", 2.2)
        ShaderLib.set_uniform("albedo", 0.950, 0.71, 0.29)

        ShaderLib.set_uniform("metallic", 1.02)
        ShaderLib.set_uniform("roughness", 0.38)
        ShaderLib.set_uniform("ao", 0.2)
        Primitives.create_triangle_plane("floor", 20, 20, 1, 1, Vec3(0, 1, 0))
        ShaderLib.print_registered_uniforms(PBR_SHADER)
        ShaderLib.use(DefaultShader.CHECKER)
        ShaderLib.set_uniform("lightDiffuse", 1.0, 1.0, 1.0, 1.0)
        ShaderLib.set_uniform("checkOn", True)
        ShaderLib.set_uniform("lightPos", self.light_pos)
        ShaderLib.set_uniform("colour1", 0.9, 0.9, 0.9, 1.0)
        ShaderLib.set_uniform("colour2", 0.6, 0.6, 0.6, 1.0)
        ShaderLib.set_uniform("checkSize", 60.0)
        ShaderLib.print_registered_uniforms(DefaultShader.CHECKER)
        Primitives.load_default_primitives()
        Primitives.create_triangle_plane("floor", 20, 20, 1, 1, Vec3(0, 1, 0))

    def resize(self, w: int, h: int) -> None:
        """
        Called when the window is resized to update the viewport and projection matrix.
        """
        self.window_width = w
        self.window_height = h
        gl.glViewport(0, 0, self.window_width, self.window_height)
        self.project = perspective(
            45.0, self.window_width / self.window_height, 0.01, 350.0
        )

    def load_matrices_to_shader(self) -> None:
        ShaderLib.use(PBR_SHADER)
        # use numpy to create a buffer object for our UBO

        # Define the structured dtype for the transform struct
        transform_dtype = np.dtype(
            [
                ("MVP", np.float32, (16)),  # Model-View-Projection matrix
                (
                    "normal_matrix",
                    np.float32,
                    (16),
                ),  # Normal transformation matrix typically this is 3x3
                ("M", np.float32, (16)),  # Model matrix
            ]
        )

        # Create an actual structured array instance
        t = np.zeros(1, dtype=transform_dtype)

        M = self.view @ self.mouse_global_tx
        MVP = self.project @ M
        normal_matrix = self.view @ self.mouse_global_tx
        normal_matrix.inverse().transpose()

        t[0]["MVP"] = MVP.to_numpy().flatten()
        t[0]["normal_matrix"] = normal_matrix.to_numpy().flatten()
        t[0]["M"] = M.to_numpy().flatten()
        ShaderLib.set_uniform_buffer("TransformUBO", data=t.data, size=t.data.nbytes)

    def handle_event(self, event: sdl3.SDL_Event) -> bool:
        """
        Process a single SDL event and update the scene state.

        Args:
            event: The SDL_Event to process.

        Returns:
            False if the application should quit, True otherwise.
        """
        if event.type == sdl3.SDL_EVENT_QUIT:
            return False

        if event.type == sdl3.SDL_EVENT_KEY_DOWN:
            if event.key.key == sdl3.SDLK_ESCAPE:
                return False
            elif event.key.key == sdl3.SDLK_W:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            elif event.key.key == sdl3.SDLK_S:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            elif event.key.key == sdl3.SDLK_SPACE:
                self.spin_x_face = 0
                self.spin_y_face = 0
                self.model_position.set(0, 0, 0)

        elif event.type == sdl3.SDL_EVENT_WINDOW_RESIZED:
            self.resize(event.window.data1, event.window.data2)

        elif event.type == sdl3.SDL_EVENT_MOUSE_BUTTON_DOWN:
            if event.button.button == sdl3.SDL_BUTTON_LEFT:
                self.rotate = True
            elif event.button.button == sdl3.SDL_BUTTON_RIGHT:
                self.translate = True

        elif event.type == sdl3.SDL_EVENT_MOUSE_BUTTON_UP:
            if event.button.button == sdl3.SDL_BUTTON_LEFT:
                self.rotate = False
            elif event.button.button == sdl3.SDL_BUTTON_RIGHT:
                self.translate = False

        elif event.type == sdl3.SDL_EVENT_MOUSE_MOTION:
            if self.rotate:
                self.spin_x_face += event.motion.yrel * 0.5
                self.spin_y_face += event.motion.xrel * 0.5
            if self.translate:
                self.model_position.x += event.motion.xrel * self.INCREMENT
                self.model_position.y -= event.motion.yrel * self.INCREMENT

        elif event.type == sdl3.SDL_EVENT_MOUSE_WHEEL:
            self.model_position.z += event.wheel.y * self.ZOOM

        return True

    def update(self) -> None:
        """
        Update transformation matrices based on current state.
        This should be called once per frame before rendering.
        """
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z

    def render(self) -> None:
        """
        Render the scene.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the color and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.load_matrices_to_shader()
        # Apply rotation based on user input
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        self.load_matrices_to_shader()
        Primitives.draw("teapot")
        ShaderLib.use(DefaultShader.CHECKER)
        tx = Mat4().translate(0.0, -0.45, 0.0)
        mvp = self.project @ self.view @ self.mouse_global_tx @ tx
        normal_matrix = Mat3.from_mat4(mvp)
        normal_matrix.inverse().transpose()
        ShaderLib.set_uniform("MVP", mvp)
        ShaderLib.set_uniform("normalMatrix", normal_matrix)
        Primitives.draw("floor")


def main():
    """
    The main entry point for the application.
    """
    if sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO) < 0:
        sys.exit(f"Error: could not initialize SDL: {sdl3.SDL_GetError()}")

    # Request an OpenGL 4.1 Core context.
    sdl3.SDL_GL_SetAttribute(sdl3.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl3.SDL_GL_SetAttribute(sdl3.SDL_GL_CONTEXT_MINOR_VERSION, 1)
    sdl3.SDL_GL_SetAttribute(
        sdl3.SDL_GL_CONTEXT_PROFILE_MASK, sdl3.SDL_GL_CONTEXT_PROFILE_CORE
    )
    # Enable multisampling for anti-aliasing
    sdl3.SDL_GL_SetAttribute(sdl3.SDL_GL_MULTISAMPLEBUFFERS, 1)
    sdl3.SDL_GL_SetAttribute(sdl3.SDL_GL_MULTISAMPLESAMPLES, 4)
    # Set depth buffer size
    sdl3.SDL_GL_SetAttribute(sdl3.SDL_GL_DEPTH_SIZE, 24)

    width, height = 1024, 720
    window = sdl3.SDL_CreateWindow(
        b"PySDL3 NGL Demo",
        width,
        height,
        sdl3.SDL_WINDOW_OPENGL | sdl3.SDL_WINDOW_RESIZABLE,
    )
    if not window:
        sys.exit(f"Error: could not create window: {sdl3.SDL_GetError()}")

    context = sdl3.SDL_GL_CreateContext(window)
    if not context:
        sys.exit(f"Error: could not create context: {sdl3.SDL_GetError()}")

    scene = Scene(width, height)
    scene.initialize()

    running = True
    event = sdl3.SDL_Event()
    while running:
        # Process all pending events
        while sdl3.SDL_PollEvent(event):
            running = scene.handle_event(event)

        # Update scene logic
        scene.update()
        # Render the scene
        scene.render()
        # Swap the front and back buffers to display the rendered frame
        sdl3.SDL_GL_SwapWindow(window)

    # Cleanup
    sdl3.SDL_GL_DestroyContext(context)
    sdl3.SDL_DestroyWindow(window)
    sdl3.SDL_Quit()


if __name__ == "__main__":
    main()
