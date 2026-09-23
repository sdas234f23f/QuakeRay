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


// HLSL counterpart of RsDepthCopying.frag.
//
// `gl_FragCoord` is the `SV_Position` input parameter of a fragment shader and `gl_FragDepth` is
// the `SV_Depth` output parameter. The push constant block keeps the golden's name and the golden's
// two defines, which have to precede the include because ShaderCommonHLSLFunc.hlsli guards the
// checkerboard helpers with them, exactly as the GLSL header does.

struct DepthCopyingFrag_BT
{
    uint renderWidth;
    uint renderHeight;
};

[[vk::push_constant]] ConstantBuffer<DepthCopyingFrag_BT> depthCopyingPush;

#define CHECKERBOARD_FULL_WIDTH depthCopyingPush.renderWidth
#define CHECKERBOARD_FULL_HEIGHT depthCopyingPush.renderHeight

#define DESC_SET_FRAMEBUFFERS 0
#include "ShaderCommonHLSLFunc.hlsli"

void main(
    float4 fragCoord : SV_Position,
    out float depth : SV_Depth)
{
    const int2 pix = int2(fragCoord.xy);

    depth = framebufDepthNdc_Sampled.Load(int3(pix, 0)).r;
}
