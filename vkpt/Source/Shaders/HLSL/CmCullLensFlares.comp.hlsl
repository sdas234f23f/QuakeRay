// Copyright (c) 2022 Sultim Tsyrendashiev
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


// HLSL counterpart of CmCullLensFlares.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = COMPUTE_INDIRECT_DRAW_FLARES_GROUP_SIZE_X, local_size_y = 1,
//     local_size_z = 1) -> [numthreads(COMPUTE_INDIRECT_DRAW_FLARES_GROUP_SIZE_X, 1, 1)]
//   * layout(constant_id = 0) const uint positionToCheckIsInScreenSpace -> [[vk::constant_id(0)]],
//     and the declaration moves above the entry point so that [numthreads] stays attached to it
//   * gl_GlobalInvocationID.x -> SV_DispatchThreadID.x
//   * the two matrix products keep the order of their operands, as ShaderCommonHLSL.hlsli
//     prescribes: globalUniform.view * vec4(position, 1.0) becomes
//     mul(globalUniform.view, float4(position, 1.0)) and globalUniform.projection * viewSpacePos
//     becomes mul(globalUniform.projection, viewSpacePos). The matrices themselves are declared
//     float4x4 in the generated header, which is the transposed declaration of the golden's mat4,
//     and they are only multiplied here, never indexed or built
//   * vec2(...) -> float2(...), vec3(...) -> float3(...), ivec2(...) -> int2(...): the same
//     component lists, and float2(globalUniform.renderWidth, globalUniform.renderHeight) converts
//     the two floats of the uniform exactly as the golden's vec2(...) does
//   * ivec2(round(...)) -> int2(round(...)): the same round-then-truncate. round itself keeps its
//     name; the two compilers lower it to different round-to-integer instructions that agree on
//     every input that is not exactly between two integers (the same caveat Q2Asvgf.hlsli records
//     for its own round)
//   * texelFetch(framebufDepthNdc_Sampled, pix, 0) -> framebufDepthNdc_Sampled.Load(int3(pix, 0))
//   * all(greaterThan(pix, ivec2(0))) -> all(pix > (int2)0) and all(lessThan(pix, ivec2(...))) ->
//     all(pix < int2(...)): HLSL compares vectors with the operators and hands back a bool vector
//     for all(), and (int2)0 is the broadcast that the golden gets from its one argument
//     constructor
//   * groupMemoryBarrier() -> AllMemoryBarrier(), and this one is a deliberate divergence from
//     the name: measured, GLSL groupMemoryBarrier() emits
//     OpMemoryBarrier Workgroup, semantics 0xD48 = AcquireRelease | UniformMemory |
//     WorkgroupMemory | ImageMemory | AtomicCounterMemory, i.e. it orders every memory class,
//     while dxc's GroupMemoryBarrier() emits Workgroup, 0x108 = AcquireRelease | WorkgroupMemory
//     only. What the golden orders here is the storage buffer that holds lensFlareDrawCmdsCount
//     (Uniform memory in SPIR-V terms), so the name-faithful GroupMemoryBarrier would drop the
//     very ordering the barrier exists for, and AllMemoryBarrier is the closest dxc has: Device,
//     0x948 = AcquireRelease | UniformMemory | WorkgroupMemory | ImageMemory, the same classes
//     minus the atomic counter (this pass has none), at the wider device scope instead of the
//     workgroup one, which is a strict superset of what the golden asks for. AllMemoryBarrier
//     carries no execution sync either, exactly like the golden
//   * atomicAdd(lensFlareDrawCmdsCount, 1) -> the three argument InterlockedAdd, which hands the
//     original value back through its out parameter:
//     InterlockedAdd(lensFlareDrawCmds[0].lensFlareDrawCmdsCount, 1, lensFlareDrawCmdIndex).
//     HLSL has no atomicAdd that returns the old value as an expression result. Both spellings
//     emit the same OpAtomicIAdd (measured: one each, same pointer, scope Device, semantics
//     Unordered on both halves)
//   * the draw command block holds an array and a counter, which no HLSL block can, so it is read
//     through one instance, as ShaderCommonHLSLFunc.hlsli declares it: the golden's
//     lensFlareDrawCmdsCount and lensFlareDrawCmds[i] become
//     lensFlareDrawCmds[0].lensFlareDrawCmdsCount and lensFlareDrawCmds[0].lensFlareDrawCmds[i].
//     lensFlareCullingInput is a plain readonly buffer and keeps its spelling
//
// What did not change: the two projections helpers with their names, parameter lists and out
// parameters, the clip space to NDC division, ndc.xy * 0.5 + 0.5, the round against the screen
// size, the early out on the culling input count, the zeroing of the counter by index 0 with the
// group barrier right after it, the depth test against occluderDepth + 0.00001 and its two
// bounds checks, and the store into the command list.


#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_LENS_FLARES 2
#include "ShaderCommonHLSLFunc.hlsli"

// A specialization constant, same as the GLSL `layout(constant_id = 0)`. It is declared before
// the entry point because [numthreads] has to be attached to the function it describes.
[[vk::constant_id(0)]] const uint positionToCheckIsInScreenSpace = 0;


void getPixAndDepthFromWorldPoint(const float3 position, out int2 pix, out float depthNdc)
{
    float4 viewSpacePos = mul(globalUniform.view, float4(position, 1.0));
    float4 clipSpacePos = mul(globalUniform.projection, viewSpacePos);

    float3 ndc = clipSpacePos.xyz / clipSpacePos.w;
    depthNdc = ndc.z;

    float2 screenSpace = ndc.xy * 0.5 + 0.5;
    float2 screenSize = float2(globalUniform.renderWidth, globalUniform.renderHeight);
    pix = int2(round(screenSpace * screenSize));
}


void getPixAndDepthFromScreenPoint(const float3 screenSpacePosition, out int2 pix, out float depthNdc)
{
    depthNdc = screenSpacePosition.z;

    // screenSpacePosition.xy is in [0..1]
    float2 screenSize = float2(globalUniform.renderWidth, globalUniform.renderHeight);
    pix = int2(round(screenSpacePosition.xy * screenSize));
}


[numthreads(COMPUTE_INDIRECT_DRAW_FLARES_GROUP_SIZE_X, 1, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    uint index = dispatchThreadID.x;

    if (index >= globalUniform.lensFlareCullingInputCount)
    {
        return;
    }


    if (index == 0)
    {
        lensFlareDrawCmds[0].lensFlareDrawCmdsCount = 0;
    }
    AllMemoryBarrier();


    const ShIndirectDrawCommand l = lensFlareCullingInput[index];
    
    int2 pix; 
    float depth;

    if (positionToCheckIsInScreenSpace != 0)
    {
        getPixAndDepthFromScreenPoint(float3(l.positionToCheck_X, l.positionToCheck_Y, l.positionToCheck_Z), pix, depth);
    }
    else
    {
        getPixAndDepthFromWorldPoint(float3(l.positionToCheck_X, l.positionToCheck_Y, l.positionToCheck_Z), pix, depth);
    }
    
    float occluderDepth = framebufDepthNdc_Sampled.Load(int3(pix, 0)).r;

    if (depth < occluderDepth + 0.00001 && 
        all(pix > (int2)0) &&
        all(pix < int2(globalUniform.renderWidth, globalUniform.renderHeight)))
    {
        uint lensFlareDrawCmdIndex;
        InterlockedAdd(lensFlareDrawCmds[0].lensFlareDrawCmdsCount, 1, lensFlareDrawCmdIndex);
        lensFlareDrawCmds[0].lensFlareDrawCmds[lensFlareDrawCmdIndex] = l;
    }
}
