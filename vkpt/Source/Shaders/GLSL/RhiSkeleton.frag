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

// Set 0 holds all three resources at the bindings NVRHI assigns to slot 0: the constant buffer at
// 256, the texture at 0 and its sampler at 128. darkColor.rgb and highlightColor.rgb are the two
// A1 colors the host writes into the block, and the caller binds a 1x1 white texture, so the
// sampled texel leaves the drawn checkerboard unchanged.
//
// Set 1 is the pass's bindless table: the split shape the engine declares for its table
// (ShaderCommonGLSLFunc.h), an unbounded array of textures at binding 0 and an unbounded array of
// samplers at binding 1. The set index is NVRHI's, not the pair's: in the legacy mode a pipeline's
// layouts keep the order the caller adds them in, with nothing appended or reordered
// (vulkan-resource-bindings.cpp:1090-1099), and a bindless layout turns its register spaces into
// Vulkan bindings counted from 0, in the order they are listed (:119 and :145-161). The pass adds
// its regular layout first and the table second, so the table is set 1.
// main samples slot 0 of the table and blends it over the checkerboard in the top-right patch, so
// the table is visible in game and a wrong slot is immediately obvious.
layout(std140, set = 0, binding = 256) uniform RhiSkeletonParams
{
    vec4 darkColor;      // rgb = the dark cell color (0.04, 0.05, 0.08), a = unused
    vec4 highlightColor; // rgb = the bright cell color (0.20, 0.35, 0.60), a = unused
} params;

layout(set = 0, binding = 0) uniform texture2D whiteTexture;
layout(set = 0, binding = 128) uniform sampler whiteTexture_Sampler;

// Set 1, the bindless table of the pass: one binding per register space, the same split the
// engine's shaders use (ShaderCommonGLSLFunc.h). It is not a slot the pass fills.
layout(set = 1, binding = 0) uniform texture2D rhiTextures[];
layout(set = 1, binding = 1) uniform sampler rhiTextures_Sampler[];

void main()
{
    // A checkerboard with a horizontal gradient: it only has to make it obvious
    // that the image on screen is produced by the RHI pass of the current frame.
    const vec2 cell = floor(vUV * vec2(16.0, 9.0));
    const float checker = mod(cell.x + cell.y, 2.0);

    vec3 color = mix(params.darkColor.rgb, params.highlightColor.rgb, checker);
    color += vec3(vUV.x * 0.35, vUV.y * 0.15, 0.0);

    // The caller binds a 1x1 white texture, so this multiply leaves the A1 image unchanged.
    color *= textureLod(sampler2D(whiteTexture, whiteTexture_Sampler), vUV, 0.0).rgb;

    // The bindless table is sampled in the top-right patch so that the table is visible in game:
    // the texture replaces the checkerboard inside the patch, and a wrong slot or an unbound table
    // is immediately obvious.
    const float patchMask = smoothstep(0.75, 0.77, vUV.x) * smoothstep(0.75, 0.77, vUV.y);
    color = mix(color, textureLod(sampler2D(rhiTextures[0], rhiTextures_Sampler[0]), vUV, 0.0).rgb, patchMask);

    oColor = vec4(color, 1.0);
}
