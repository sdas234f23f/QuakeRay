// Hand written counterpart of GLSL/Volumetric.probe.comp.
//
// Volumetric.h has no shader stage of its own, so this pair exists to check Volumetric.hlsli
// against it. CmVolumetricProcess.comp, CmPrepareFinal.comp and RsWorld.frag are the consumers of
// Volumetric.h, and they all reach it after ShaderCommonGLSLFunc.h, so the probe pulls in the same
// layer of accessors: the global uniform, from which the two matrix products and the reprojection
// read the four volume matrices and the two camera positions, and the volume that the three
// samplers read. Those two sets are the only ones the probe enables, and the resources of them that
// the header touches are touched on both halves, because dxc drops a resource that nothing reads
// after the accessors have been inlined while glslc emits every declaration of the headers.
//
// Volumetric.h declares no resource of its own, so what the pair pins is the layout of the global
// uniform members that the header reads (volumeViewProj, volumeViewProjInv, volumeViewProj_Prev,
// volumeViewProjInv_Prev, cameraPosition, cameraPositionPrev, volumeCameraNear, volumeCameraFar),
// the descriptors of the two sampled views and of their two samplers, and the properties of the
// entry point that both compilers derive from calling all eight functions of the header. All eight
// are instantiated on both halves: volume_getCenter_T and volume_toSamplePosition_T directly, so
// that both matrix products of the header are pinned, with the matrices of both frames, and the
// other six through them. The two products keep the order of the operands on both sides, so what
// this half multiplies is the transposed declaration of the generated header and not the transpose
// of the product. This file has to be edited together with the GLSL one: what one side reaches and
// the other does not is reported as a mismatch, which is exactly the point.
//
// The header writes nothing -- it reads the global uniform and samples the volumes -- so those are
// the whole of its resource usage, and the probe adds nothing to them. In particular the storage
// images of DESC_SET_VOLUMETRIC (g_volumetric, g_illuminationVolume) and the sampled view of the
// illumination volume are not touched here, because the header does not touch them either.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_VOLUMETRIC     4

#include "ShaderCommonHLSLFunc.hlsli"
#include "Volumetric.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    // ShGlobalUniform: the four matrices and the two camera positions that the two functions which
    // take their helpers as parameters are called with, and the near and far plane that
    // volume_getCenter_T and volume_toSamplePosition_T read
    const float4x4 viewproj = globalUniform.volumeViewProj;
    const float4x4 viewprojInv = globalUniform.volumeViewProjInv;
    const float3 camPos = globalUniform.cameraPosition.xyz;
    const float3 camPosPrev = globalUniform.cameraPositionPrev.xyz;
    v += globalUniform.volumeCameraNear + globalUniform.volumeCameraFar;

    const int3 cell = int3(1, 2, 3);
    const float3 world = float3(0.5, 0.5, 0.5);
    const float3 rnd01 = float3(0.25, 0.5, 0.75);

    // The two functions that take a matrix directly: the inverse projection of a cell centre, of
    // the current and of the previous frame, and the forward projection of a sample position. The
    // first and the last are the two matrix products of the header
    v += volume_getCenter_T(cell, viewprojInv, camPos).x;
    v += volume_getCenter_T(cell, globalUniform.volumeViewProjInv_Prev, camPosPrev).z;
    v += volume_toSamplePosition_T(world, viewproj, camPos).y;

    // The wrappers of the header: volume_getCenter and volume_getCenter_Prev read their matrix
    // member themselves, volume_sample and volume_sample_Prev reproject the world position through
    // volume_toSamplePosition_T
    v += volume_getCenter(cell).z;
    v += volume_getCenter_Prev(cell).x;
    v += (float)volume_toCellIndex(world).y;

    // The three samplers: the current volume, the previous one through the reprojection, and the
    // dithered one, which is the only place where SHIPPING_HACK matters
    v += volume_sample(world).x;
    v += volume_sample_Prev(cell).y;
    v += volume_sampleDithered(world, rnd01, 1.0).z;

    probeOutput[0] = v;
}
