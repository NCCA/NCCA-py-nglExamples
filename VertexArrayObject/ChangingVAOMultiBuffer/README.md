# Changing MultiBuffer VAO

![](ChangingVAO.png)

This demo uses the MultiBuffer VAO to demonstrate how to change the VAO dynamically. In this case the vertices change each frame, but the colour data is only created once.

## References

- [OpenGL Wiki — Buffer Object Streaming](https://www.khronos.org/opengl/wiki/Buffer_Object_Streaming) — updating only the dynamic (vertex) buffer while the static (colour) buffer stays put.
- [OpenGL Wiki — Vertex Specification Best Practices](https://www.khronos.org/opengl/wiki/Vertex_Specification_Best_Practices) — when separate buffers beat interleaving.
