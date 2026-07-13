# Restart Line

![](PrimRestart.png)

This project demonstrates how to use [glPrimitiveRestartIndex](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glPrimitiveRestartIndex.xhtml) to draw different lines of different lengths using the [SimpleIndexVAO](https://github.com/NCCA/PyNGL/blob/main/src/ncca/ngl/simple_index_vao.py) class

The [PrimRestartLine.py](PrimRestartLine.py) uses the build in PyNGL elements for ease but the update is slow.

The [FasterVersion.py](FasterVersion.py) uses buffer created using Numpy to make things quicker, it also uses raw OpenGL commands to generate an Element Buffer Object

## References

- [OpenGL Wiki — Vertex Rendering: Primitive Restart](https://www.khronos.org/opengl/wiki/Vertex_Rendering#Primitive_Restart) — how the restart index splits one indexed draw into many strips/lines.
- [glPrimitiveRestartIndex — OpenGL Reference](https://registry.khronos.org/OpenGL-Refpages/gl4/html/glPrimitiveRestartIndex.xhtml) — the API used here.
- [OpenGL Wiki — Buffer Object Streaming](https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming) — why the NumPy/raw-EBO `FasterVersion.py` beats per-frame rebuilds.
