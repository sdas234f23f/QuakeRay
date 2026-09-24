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

// Set 0 is the pass's only set, and it holds all three resources at the bindings NVRHI assigns to
// slot 0: the constant buffer at 256, the ALBEDO image at 0 and its sampler at 128. The present
// reads the engine's ALBEDO render target - the image the raster passes of the frame write - and
// writes the display-ready value to the single colour attachment of the pass.
//
// Unlike RhiSkeleton.frag there is no bindless table and no set 1: the present samples exactly one
// image, so a table of its own would declare a descriptor the host would have to fill for nothing.
layout(std140, set = 0, binding = 256) uniform RhiPresentParams
{
    vec4 exposure; // x = exposure multiplier applied before the curve; y, z and w are unused
} params;

layout(set = 0, binding = 0) uniform texture2D albedoTexture;
layout(set = 0, binding = 128) uniform sampler albedoTexture_Sampler;

void main()
{
    // LOD 0 is explicit, as in RhiSkeleton.frag: the target can differ in size from the source,
    // and an implicit LOD would make the presented image depend on the screen's derivative.
    const vec3 albedo = textureLod(sampler2D(albedoTexture, albedoTexture_Sampler), vUV, 0.0).rgb;

    // The exposure scales the linear HDR value, then the monotone x / (1 + x) curve folds the
    // unbounded result into [0, 1) before it reaches the display attachment.
    const vec3 exposed = albedo * params.exposure.x;
    const vec3 display = exposed / (1.0 + exposed);

    oColor = vec4(display, 1.0);
}
