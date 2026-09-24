/*
* Copyright (c) 2026 Sultim Tsyrendashiev
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#version 460
#extension GL_EXT_ray_tracing : require

// The miss of the debug pair: it writes the fixed background colour of the debug view and never
// touches a descriptor. The pair's three files must keep RhiDebugPayload identical; the
// background value must stay the one RhiDebugTrace.rchit uses for a sky instance.
//
// Unlike RtMiss.rmiss this stage has no use for ShaderCommonGLSLFunc.h: it reads no framebuffer,
// no uniform and no packed index, so it declares nothing and includes nothing. That is also why
// the payload location is the literal 0 of PAYLOAD_INDEX_DEFAULT - the generated header that
// defines the name is not included, and the constant would be its only use.

struct RhiDebugPayload
{
    vec3 color;
};

layout(location = 0) rayPayloadInEXT RhiDebugPayload payload;

void main()
{
    // A near-black blue: the traced geometry is drawn unlit, so the background has to stay dark
    // for the hits to stand out against it.
    payload.color = vec3(0.01, 0.01, 0.02);
}
