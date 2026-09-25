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

// The packed unfiltered-direct image is decoded with the shared helper, exactly as every other
// shader that reads a packed framebuffer does.
#include "Utils.h"

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
    vec4 exposure; // x = exposure multiplier applied before the curve; y = vertical mirror of the
                   // sample coordinate (0 or 1); z = non-zero enables the direct-lighting term of
                   // the traced chain; w unused
} params;

layout(set = 0, binding = 0) uniform texture2D albedoTexture;
layout(set = 0, binding = 128) uniform sampler albedoTexture_Sampler;
layout(set = 0, binding = 384, r32ui) uniform uimage2D directTexture;

void main()
{
    // The sample coordinate: params.exposure.y mirrors it vertically. The traced modes' ALBEDO is
    // written by the engine's ray-tracing passes, whose pixel-to-UV convention puts the view's
    // first row in the image's first row, while the raster passes of the frame feed this present in
    // NVRHI's raster convention, which puts the view's first row in the image's last row (the two
    // were measured in RhiDebugTrace.rgen.hlsl:107-114). The host sets the flag per frame mode, so
    // one present serves both.
    const vec2 uv = vec2(vUV.x, mix(vUV.y, 1.0 - vUV.y, params.exposure.y));

    // LOD 0 is explicit, as in RhiSkeleton.frag: the target can differ in size from the source,
    // and an implicit LOD would make the presented image depend on the screen's derivative.
    const vec3 albedo = textureLod(sampler2D(albedoTexture, albedoTexture_Sampler), uv, 0.0).rgb;

    // The direct-lighting term of the traced chain: the image is checkerboard-packed, so this
    // screen pixel's trace is addressed through the same mapping the raygens write with
    // (getCheckerboardPix, ShaderCommonGLSLFunc.h:332-341; the render width halves into the two
    // fields). params.exposure.z is set while the direct pass runs; the raster and debug modes leave
    // it zero and read nothing.
    vec3 direct = vec3(0.0);
    if (params.exposure.z != 0.0)
    {
        const ivec2 size = textureSize(sampler2D(albedoTexture, albedoTexture_Sampler), 0);

        const ivec2 p = clamp( ivec2( uv * vec2(size) ), ivec2(0), size - ivec2(1) );
        const int sep = size.x / 2;
        const int odd = ( p.x + p.y % 2 ) % 2;
        const ivec2 cb = ivec2( odd * sep + p.x / 2, p.y );

        direct = decodeE5B9G9R9( imageLoad(directTexture, cb).r );
    }

    // The exposure scales the linear HDR value, then the monotone x / (1 + x) curve folds the
    // unbounded result into [0, 1) before it reaches the display attachment. The direct term is
    // added to the albedo, the A4.2 diagnostic compose (sky pixels keep their color).
    const vec3 illuminated = albedo * ( 1.0 + direct );
    const vec3 exposed = illuminated * params.exposure.x;
    const vec3 display = exposed / (1.0 + exposed);

    oColor = vec4(display, 1.0);
}
