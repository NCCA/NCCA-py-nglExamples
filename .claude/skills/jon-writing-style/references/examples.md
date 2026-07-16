# Real excerpts of Jon's writing

Everything below is quoted verbatim from Jon's repos (mainly [NCCA/PyNGL](https://github.com/NCCA/PyNGL) and [NCCA/PyNGLDemos](https://github.com/NCCA/PyNGLDemos)) or his blog. Read for rhythm, not rules: sentences are quick, first-person, sometimes carrying a typo or a slightly loose construction — that looseness is part of the voice and should not be "polished" away when writing new prose (though obviously don't introduce typos on purpose).

## Repo descriptions (one line is enough)

> Maya API demos used in Lectures

> This is all the code used in my design patterns lectures

## Project README, work in progress (PyNGLDemos/README.md, complete)

> # Py-NGL Demos
>
> This is work in progress examples for the new all python version of the NGL library.
>
> Whilst I will mainly be using WebGP for my teaching I thought It would be good to also have an OpenGL example for quick demos and easier coding in some cases.
>
> The Py-NGL library is being developed at the same time as this at present but will be on PyPi once everything is fully developed and I will document it all properly then.

Note what this does: says what it is, why it exists (his teaching), what state it's in, and what's coming — four sentences, no headings beyond the title, no feature list.

## Library README opening (PyNGL/README.md)

> # PyNGL
>
> This is the code for the full python version of [NGL](https://github.com/NCCA/NGL) the ncca graphics library.
>
> This project is available on PyPI and can be installed using uv.
>
> For the current build status see our CI logs here

(Badges follow because the project already has CI and Sonar — badges exist where they earn their place, they are not decoration.) Then straight into usage:

> ## Test
>
> use
>
> ```
> uv run pytest
> ```
>
> To run tests,
>
> ```
> uv run pytest --cov=src --cov-report=term-missing
> ```
>
> For coverage reports.

## Small demo READMEs

A two-sentence README is a finished README (PyNGLDemos/BlankWebGPU):

> ## Blank WebGPU
>
> A bare bones WebGPU demo used as a starting point for other demos. As WebGPU needs a pipeline to even clear the screen this demo draws the simplest triangle geometry so all the relevant elements of the code are in place to build more complex demos.

Another (PyNGLDemos/ObjViewer):

> # ObjViewer
>
> This demo shows how to use the PyNGL Obj class to load and display obj files.
>
> This demo loads the obj and an optional texture file which is then converted into a Vertex Array Object (VAO) for rendering.

## A longer demo README (PyNGLDemos/ScreenTri, complete)

Explains the idea, links out instead of re-explaining, then a numbered walkthrough in "we" voice and a bare key list:

> # ScreenTri
>
> This demo shows how we can generate a full screen rendering area by using a simple triangle.
>
> The triangle will be much larger than the viewport area but only a part of it will be visible as the rest will be clipped.  This is a very efficient way of rendering what used to be called a "Screen Quad"
>
> There is a good article [here](https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/) showing this and this demo is a good starting point for any project that requires rendering some form of buffer / texture to the screen.
>
> ## Process
>
> 1. We first load the shaders which will be called when drawing our triangle.
> 2. Next we need a VertexArrayObject to enable drawing. This will not contain any buffers or data but still needs to be bound.
> 3. call glDrawArrays(GL_TRIANGLES,0,3); This calls the vertex shader 3 times, where gl_VertexID is used to determine which of the triangle vertices to generate and also the texture co-ordinates
> 4. These are passed onto the fragment shader.
>
> In this demo we will fill a buffer with values then bind to a texture to render to the screen.
>
> ## Keys
>
> - Space : reset the canvas to white
> - a : toggle animation
> - p : draw points mode
> - l : draw lines mode

## Candour about trade-offs and loose ends (PyNGLDemos/ColourSelectionOpenGL)

He credits his sources, admits the approach is not optimal, and leaves an honest note about what isn't handled yet:

> Picking of objects using Colour values based on this [post](https://moddb.fandom.com/wiki/OpenGL_Selection_Using_Unique_Color_IDs).

> This is not the fastest method, but it is simple and easy to implement. All colors are generated using a single generator instance.

> At present the background colour is not taken into account (128,128,128)

## Usage documentation as short prose + real code (PyNGLDemos/FontRendering)

> We only need to load the fonts once, we typically do this in the ```initializeGL``` function as we need an OpenGL context to create the texture atlas.
>
> ```python
> Text.add_font("70s", "70SdiscopersonaluseBold-w14z2.otf", FONT_SIZE)
> Text.add_font("Painter", "Painter-LxXg.ttf", FONT_SIZE)
> ```
>
> To render text we can use the ```render_text``` function.
>
> ```python
> Text.render_text("Arial", 10, 440, "To Render we call")
> ```
>
> With and optional colour parameter (default is white)

## Python docstrings (PyNGL source)

Class docstring with an `Attributes` section (`src/ncca/ngl/vec3.py`):

```python
class Vec3(VectorBase["Vec3"]):
    """
    A simple 3D vector class for 3D graphics, using numpy for efficient operations.

    Attributes:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        z (float): The z-coordinate of the vector.
    """
```

Method docstring — one plain sentence of purpose, then parameters (`src/ncca/ngl/vec3.py`):

```python
def reflect(self, n: "Vec3") -> "Vec3":
    """
    Reflect a vector about a normal.

    Args:
        n (Vec3): The normal to reflect about.

    Returns:
        Vec3: A new vector that is the result of reflecting this vector about the normal.
    """
    d = self.dot(n)
    # I - 2.0 * dot(N, I) * N
    result = Vec3()
    result._data = self._data - 2.0 * d * n._data
    return result
```

The one inline comment there states the formula being implemented — the *why*, not a narration of the line. Trivial functions get a bare one-liner or nothing at all (`src/ncca/ngl/util.py`):

```python
def clamp(num, low, high):
    "clamp to range min and max will throw ValueError is low>=high"
    if low > high or low == high:
        raise ValueError
    return max(min(num, high), low)


def lerp(a, b, t):
    return a + (b - a) * t
```

A class-level docstring for infrastructure code stays to two flat sentences (`src/ncca/ngl/shader_lib.py`):

```python
class _ShaderLib:
    """
    Shader library for managing OpenGL shader programs and shaders.
    Provides methods to load, compile, link, and use shaders, as well as manage uniforms and uniform blocks.
    """
```

## Blog fragments

Short lines quoted from his blog posts, for the voice:

> To enable WSL2 we need to activate the WSL feature

> the advantages of getting AI to do work for you!

First person for decisions, "we" for walkthroughs, dry aside in parentheses or after a dash, honest about using AI where he did.
