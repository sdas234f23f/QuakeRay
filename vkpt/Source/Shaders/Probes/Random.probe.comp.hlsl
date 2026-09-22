// Random.h has no shader stage of its own, so this probe exists to check Random.hlsli against the
// GLSL half in GLSL/Random.probe.comp, which calls the same helpers. CheckShaderProperties.py
// compares what glslc and dxc derive from the two: the descriptor set and binding of the blue
// noise texture, its resource kind, and the properties of the two entry points. Probes are not
// part of the shader build, so no probe blob is ever shipped.

#define DESC_SET_RANDOM 5

// ShaderCommonHLSL.hlsli brings in the generated header and Utils.hlsli, so unlike the GLSL half,
// which has to name Utils.h itself, this one does not name Utils.hlsli on its own. Every guarded
// header of the base carries a content based `#ifndef`, so naming it twice would be harmless.
#include "ShaderCommonHLSL.hlsli"
// The probe reads columns of the basis below, so it needs getColumn. Random.hlsli does not use the
// helper itself, but the GLSL half reads the same columns, so the two halves must read them alike.
#include "ShaderCommonHLSLFunc.hlsli"
#include "Random.hlsli"

[[vk::binding(0, 8)]] RWStructuredBuffer<float> o;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    const uint seed = packRandomSeed(3, uint2(1, 2));

    uint texIndex;
    uint2 offset;
    unpackRandomSeed(seed, texIndex, offset);
    v += (float)texIndex + (float)offset.x + (float)offset.y;

    v += rndBlueNoise8(seed, 7).x;

    v += rnd16(seed, 1) + rnd16_2(seed, 2).y + rnd8_4(seed, 3).w;
    v += (float)wellonsLowBias32(seed);
    v += (float)murmurHash33(uint3(1, 2, 3)).z + (float)getRandomSeed(int2(1, 2), 4);

    v += sampleDisk(1.0, 0.25, 0.75).x;
    v += sampleTriangle(float3(0.0, 0.0, 0.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), 0.5, 0.25).y;

    float oneOverPdf = 0.0;
    v += sampleHemisphere(0.25, 0.75, oneOverPdf).z;
    v += oneOverPdf;
    v += sampleSphere(0.25, 0.75).x;

    const float3 n = normalize(float3(0.3, 0.4, 0.5));
    float3 b1;
    float3 b2;

    revisedONB(n, b1, b2);
    v += b1.x + b2.y;
    frisvadONB(n, b1, b2);
    v += b1.z + b2.z;

    const float3x3 basis = getONB(n);
    v += getColumn(basis, 0).y + getColumn(basis, 2).z;

    v += sampleOrientedHemisphere(n, 0.25, 0.75, oneOverPdf).y;
    v += oneOverPdf;

    o[0] = v;
}
