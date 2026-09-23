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


// from https://github.com/libretro/glsl-shaders



#include "EfSimple.hlsli"

// The argument list is the golden's transposed: HLSL fills rows where GLSL fills columns, and the
// two objects hold the same element at the same (row, column).
static const float3x3 yiq_mat = float3x3(
      0.2989, 0.5959, 0.2115,
      0.5870, -0.2744, -0.5229,
      0.1140, -0.3216, 0.3114
);

float3 rgb2yiq(float3 col)
{
   return mul(col, yiq_mat);
}

#define CHROMA_MOD_FREQ (4.0 * M_PI / 15.0)

#define SATURATION 1.0
#define BRIGHTNESS 1.0
#define ARTIFACTING 0.0
#define FRINGING 0.0

// Transposed argument list again, for the reason above.
static const float3x3 mix_mat = float3x3(
	BRIGHTNESS, ARTIFACTING, ARTIFACTING,
	FRINGING, 2.0 * SATURATION, 0.0,
	FRINGING, 0.0, 2.0 * SATURATION
);

// GLSL mod(x, y) is floor based; HLSL fmod truncates towards zero instead, so the golden's mod()
// is spelled out as in RaygenPrimary.hlsli.
float crtDemodulateMod(float x, float y)
{
    return x - y * floor(x / y);
}

float3 demodulateAndEncode(int2 pix)
{
	float3 yiq = rgb2yiq(effect_loadFromSource(pix));

    float chroma_phase = M_PI * (crtDemodulateMod(float(pix.y), 2.0) + globalUniform.frameId);

	float mod_phase = chroma_phase + float(pix.x) * CHROMA_MOD_FREQ;

	float i_mod = cos(mod_phase);
	float q_mod = sin(mod_phase);

	yiq.yz *= float2(i_mod, q_mod); // Modulate.
	yiq = mul(yiq, mix_mat); // Cross-talk.
	yiq.yz *= float2(i_mod, q_mod); // Demodulate.
    
    return yiq;
}

[numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    
    if (!effect_isPixValid(pix))
    {
        return;
    }

    float3 yiq = demodulateAndEncode(pix);
    effect_storeToTarget(encodeYiqForStorage(yiq), pix);
}
