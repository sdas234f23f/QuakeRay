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


// HLSL counterpart of CmBloomApply.comp.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID -> SV_DispatchThreadID, and the golden's local_size_x/y =
//     COMPUTE_BLOOM_APPLY_GROUP_SIZE_X/Y become [numthreads(...)] on the entry point, because in
//     HLSL the workgroup size is an attribute of the shader and not of the module
//   * layout(constant_id = 0) -> [[vk::constant_id(0)]]
//   * textureLod(sampler2D(framebufBloom_Result_Sampled, framebufBloom_Result_Sampler), uv, 0) ->
//     framebufBloom_Result_Sampled.SampleLevel(framebufBloom_Result_Sampler, uv, 0): the same
//     explicit-level fetch, through the sampled image and the sampler the host binds as two
//     descriptors at the two bindings of the GLSL pair
//   * effect_loadFromSource and effect_storeToTarget come from EfCommon.hlsli, the twin of
//     EfCommon.inl, and keep their names, their arguments and both branches on
//     EFFECT_SOURCE_IS_PING
//
// What did not change: the early out on the UV, the operand order of the bloom product, and the
// sum of the source color and the bloom.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint isSourcePing = 0;

#define EFFECT_SOURCE_IS_PING (isSourcePing != 0)
#include "EfCommon.hlsli"

[numthreads(COMPUTE_BLOOM_APPLY_GROUP_SIZE_X, COMPUTE_BLOOM_APPLY_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    const float2 uv = effect_getFramebufUV(pix);

    if (uv.x > 1.0 || uv.y > 1.0)
    {
        return;
    }

    const float3 bloom = globalUniform.bloomIntensity * framebufBloom_Result_Sampled.SampleLevel(framebufBloom_Result_Sampler, uv, 0).rgb;
    
    float3 c = effect_loadFromSource(pix) + bloom;
    effect_storeToTarget(c, pix);
}
