#!/usr/bin/env -S uv run --script
"""
A template for creating a PySDL3 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It mirrors the functionality of the BlankPySide6NGL example using the SDL3 library.
"""

import sys

import OpenGL.GL as gl
import sdl3
from ngl import Mat4, Vec3, look_at, perspective


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
        self.mouseGlobalTX: Mat4 = (
            Mat4()
        )  # Global transformation matrix controlled by the mouse
        self.view: Mat4 = Mat4()  # View matrix (camera's position and orientation)
        self.project: Mat4 = (
            Mat4()
        )  # Projection matrix (defines the camera's viewing frustum)
        self.modelPos: Vec3 = Vec3()  # Position of the model in world space

        # --- Window and UI Attributes ---
        self.window_width: int = width
        self.window_height: int = height

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spinXFace: int = 0  # Accumulated rotation around the X-axis
        self.spinYFace: int = 0  # Accumulated rotation around the Y-axis
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
            elif event.key.key == sdl3.SDLK_w:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            elif event.key.key == sdl3.SDLK_s:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            elif event.key.key == sdl3.SDLK_SPACE:
                self.spinXFace = 0
                self.spinYFace = 0
                self.modelPos.set(0, 0, 0)

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
                self.spinXFace += event.motion.yrel * 0.5
                self.spinYFace += event.motion.xrel * 0.5
            if self.translate:
                self.modelPos.x += event.motion.xrel * self.INCREMENT
                self.modelPos.y -= event.motion.yrel * self.INCREMENT

        elif event.type == sdl3.SDL_EVENT_MOUSE_WHEEL:
            self.modelPos.z += event.wheel.y * self.ZOOM

        return True

    def update(self) -> None:
        """
        Update transformation matrices based on current state.
        This should be called once per frame before rendering.
        """
        # Apply rotation based on user input
        rotX = Mat4().rotate_x(self.spinXFace)
        rotY = Mat4().rotate_y(self.spinYFace)
        self.mouseGlobalTX = rotY @ rotX
        # Update model position
        self.mouseGlobalTX[3][0] = self.modelPos.x
        self.mouseGlobalTX[3][1] = self.modelPos.y
        self.mouseGlobalTX[3][2] = self.modelPos.z

    def render(self) -> None:
        """
        Render the scene.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # --- All drawing code would go here, using the scene's matrices ---
        # For example:
        # mvp = self.project @ self.view @ self.mouseGlobalTX
        # ShaderLib.set_uniform("MVP", mvp)
        # vao.draw()


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
