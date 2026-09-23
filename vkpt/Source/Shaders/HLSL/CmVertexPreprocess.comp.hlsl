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


// HLSL counterpart of CmVertexPreprocess.comp.
//
// Spellings that had to change:
//   * gl_WorkGroupID.x -> SV_GroupID.x, and the golden's local_size_x =
//     COMPUTE_VERT_PREPROC_GROUP_SIZE_X with local_size_y/z = 1 becomes the
//     [numthreads(COMPUTE_VERT_PREPROC_GROUP_SIZE_X, 1, 1)] attribute of the entry point
//   * gl_LocalInvocationID is read by VertexPreprocessPartial.hlsli from the scope it is included
//     into; in HLSL that is this entry point's SV_GroupThreadID, so main passes groupThreadID
//   * layout(constant_id = 0) -> [[vk::constant_id(0)]]
//   * layout(push_constant) uniform Push_BT { ShVertPreprocessing push; } -> the block's single
//     member is declared directly as [[vk::push_constant]] ConstantBuffer<ShVertPreprocessing>
//     push, so the golden's `push.field` stays `push.field`, and the member layout is the
//     generated ShVertPreprocessing of the generated header on both sides
//   * the three inclusion sites of VertexPreprocessPartial.inl become the same three sites of
//     VertexPreprocessPartial.hlsli, behind the same three macros and with the middle one
//     commented out exactly as in the golden; that fragment is deliberately unguarded and
//     undefines its own macro at the end, so including it again under another macro is what it
//     was written for
//   * the `1 << tlasInstanceIndex` of the golden is written `1u << tlasInstanceIndex`: dxc rejects
//     the mixed literal/uint shift with "ambiguous type for bit shift" and asks for the unsigned
//     suffix, and the unsigned shift is what GLSL performs (a mixed shift converts the left
//     operand to the type of the right one, so the golden's 1 is a uint there as well)
//
// What did not change: the dynamic-instance bit test with its `/ 32` array index and its raw
// shift, the "always process dynamic" branch and the VERT_PREPROC_MODE_ALL branch, and the value
// of the preprocessMode specialization constant.

#define VERTEX_BUFFER_WRITEABLE
#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_VERTEX_DATA 1
#include "ShaderCommonHLSLFunc.hlsli"

[[vk::constant_id(0)]] const uint preprocessMode = VERT_PREPROC_MODE_ONLY_DYNAMIC;

[[vk::push_constant]] ConstantBuffer<ShVertPreprocessing> push;

[numthreads(COMPUTE_VERT_PREPROC_GROUP_SIZE_X, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{    
    uint tlasInstanceIndex = groupID.x;
    bool isDynamic = (push.tlasInstanceIsDynamicBits[tlasInstanceIndex / 32] & (1u << tlasInstanceIndex)) != 0;


    // always process dynamic
    if (isDynamic)
    {
        #define VERTEX_PREPROCESS_PARTIAL_DYNAMIC
        #include "VertexPreprocessPartial.hlsli"
    }
    /*else if (preprocessMode == VERT_PREPROC_MODE_DYNAMIC_AND_MOVABLE)
    {       
        #define VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE
        #include "VertexPreprocessPartial.hlsli"
    }*/
    else if (preprocessMode == VERT_PREPROC_MODE_ALL)
    {       
        #define VERTEX_PREPROCESS_PARTIAL_STATIC_ALL
        #include "VertexPreprocessPartial.hlsli"
    }
}
