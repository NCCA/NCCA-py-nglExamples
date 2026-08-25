#version 330 core

// Nothing to write: the FBO bound for this pass has no colour attachment
// (glDrawBuffer(GL_NONE) / glReadBuffer(GL_NONE) in main.py) and only the
// GL_DEPTH_COMPONENT24 texture is written by the fixed-function depth test.
void main()
{
}
