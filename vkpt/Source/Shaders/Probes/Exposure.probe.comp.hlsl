// Hand written counterpart of GLSL/Exposure.probe.comp.
//
// Exposure.h has no shader stage of its own, so this pair exists to check Exposure.hlsli against
// it. CmPrepareFinal.comp and RsWorld.frag are the consumers of Exposure.h, and they reach it
// after ShaderCommonGLSLFunc.h, so the probe pulls in the same layer of accessors: the histogram
// buffer that getAutoEV100 reads. That set is the only one the probe enables, and the resource is
// touched on both halves, because dxc drops a resource that nothing reads after the accessors have
// been inlined while glslc emits every declaration of the headers.
//
// Exposure.h declares no resource of its own, so what the pair pins is the layout of the histogram
// buffer that the header reads -- through its single member, avgLuminance, which the GLSL half
// spells tonemapping.avgLuminance and this one tonemapping[0].avgLuminance -- together with the
// properties of the entry point that both compilers derive from calling all five functions of the
// header. All five are instantiated on both halves, and this file has to be edited together with
// the GLSL one: what one side reaches and the other does not is reported as a mismatch, which is
// exactly the point.
//
// The header reads no member of the histogram block other than avgLuminance, and it writes
// nothing, so those are the whole of its resource usage.

#define DESC_SET_TONEMAPPING 2

#include "ShaderCommonHLSLFunc.hlsli"
#include "Exposure.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    // DESC_SET_TONEMAPPING, the only resource of the header
    v += tonemapping[0].avgLuminance;

    // All five functions of the header. getManualEV100 is fed the same values that the header
    // itself passes to it, the int 100 included, so that the implicit conversion of that argument
    // is exercised here as well and not only inside getCurrentEV100, where the false `manual` makes
    // the call unreachable.
    v += getManualEV100(1.0 / 16.0, 1.0 / 125.0, 100);
    v += getAutoEV100();
    v += getCurrentEV100();
    v += ev100ToLuminousExposure(5.0);
    v += ev100ToLuminance(5.0);

    probeOutput[0] = v;
}
