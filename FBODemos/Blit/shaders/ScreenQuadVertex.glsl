#version 330 core

layout(location=0) in vec4 inVert;

 void main()
 {
    gl_Position = inVert;
 }
