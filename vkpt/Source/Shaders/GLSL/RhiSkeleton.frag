// Copyright (c) 2021 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#version 460

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 oColor;

void main()
{
    // A checkerboard with a horizontal gradient: it only has to make it obvious
    // that the image on screen is produced by the RHI pass of the current frame.
    const vec2 cell = floor(vUV * vec2(16.0, 9.0));
    const float checker = mod(cell.x + cell.y, 2.0);

    vec3 color = mix(vec3(0.04, 0.05, 0.08), vec3(0.20, 0.35, 0.60), checker);
    color += vec3(vUV.x * 0.35, vUV.y * 0.15, 0.0);

    oColor = vec4(color, 1.0);
}
