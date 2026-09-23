// Hand written counterpart of GLSL/Q2Fog.probe.comp.
//
// Q2Fog.h has no shader stage of its own, so this pair exists to check Q2Fog.hlsli against it.
// CmQ2Fog.comp and RaygenPrimary.inl are the consumers of Q2Fog.h, and both reach it after
// ShaderCommonGLSLFunc.h, so the probe pulls in the same layer of accessors: the global uniform,
// which is the whole of what the header reads (fogIsActive, fogMins, fogMaxs, fogColor,
// fogDensity) plus the MAX_FOG_VOLUMES bound of its loop. The header declares no resource of its
// own and reads no texture, so that one set is the only one the probe enables, and all five fog
// arrays are read on both halves.
//
// All nine functions of the header are instantiated here on both halves, with the constants of
// their arguments so that everything that glslang and dxc are willing to fold is folded and the
// two disassemblies can be compared number by number:
//   q2FogVMin and q2FogVMax       -> on (float3)2.0 and on the reciprocal of float3(2,4,8)
//   q2PackHalf4x16                -> on float4(1,2,3,4), read back through q2UnpackHalf4x16
//   q2UnpackHalf4x16              -> on uint2(0x3C00, 0x4200)
//   q2FindFogVolumes              -> the uniform-driven path; it is the only caller of the three
//                                    that follow and it cannot fold, because everything it reads
//                                    comes from the global uniform
//   q2EvaluateFog                 -> on the volume the finder returned, with t1 = 1, t2 = 3
//   q2AlphaBlendPremultiplied     -> on that result and a constant
//   q2SegmentFog and q2ApplyFog   -> the two wrappers, on the two volumes the finder returned
//   Q2_FOG_BRIGHTNESS             -> the one macro of the header, on its own as well, so that its
//                                    value is pinned by a folded number and not only by its use
//
// The one substitution of this port that has no HLSL intrinsic is GLSL's packHalf2x16 /
// unpackHalf2x16: this half spells them f32tof16 / f16tof32, through the two helpers of the
// header. Both are worked into this file as a raw pair next to the functions of the header that use
// them, on the same constant, so that the two halves can be read against each other: the GLSL half
// hands the word 0xC0003C00 to the builtin, this half gets there as
// f32tof16(1.0) | (f32tof16(-2.0) << 16), and the -2.0 of the unpacked pair is read the same way.
// Neither front end folds the pack or the unpack itself -- both keep an OpExtInst PackHalf2x16 /
// UnpackHalf2x16 -- so those two sites are compared on the instruction and on the operands, and the
// folded numbers of this pair are taken from the sites that do fold, see the slot table.
// The single-scalar constructors of the golden are casts here ((float3)2.0 for vec3(2.0)), which
// the folded operands below cover as well: a cast that did not broadcast would give 0.0 in the
// other two lanes where the golden gives 2.0, and dxc's own form of the broadcast is visible as the
// repeated %float_2 of the two NMin.
//
// Q2Fog.h reads nothing but the global uniform and writes nothing, so there is no storage image
// and no sampler in this file. This file has to be edited together with the GLSL one: what one side
// reaches and the other does not is reported as a mismatch, which is exactly the point.

#define DESC_SET_GLOBAL_UNIFORM 0

#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Fog.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    // The two helpers that the slab test of q2FindFogVolumes is built from. q2FogVMin((float3)2.0)
    // is 2.0 only if the cast broadcasts its argument into all three lanes, and 0.0 if it filled
    // one lane and left the other two at zero; q2FogVMax((float3)1.0 / float3(2,4,8)) is the same
    // construction under the reciprocal division of the header
    v += q2FogVMin((float3)2.0);
    v += q2FogVMax((float3)1.0 / float3(2.0, 4.0, 8.0));

    // The half packing of the header, on the two shapes it is used with: the (t_in, t_out) pair in
    // .z, and the four color components of the .xy pair, both read back through q2UnpackHalf4x16
    v += (float)q2PackHalf4x16(float4(1.0, 2.0, 3.0, 4.0)).x;
    v += q2UnpackHalf4x16(uint2(0x3C00u, 0x4200u)).x;
    v += q2UnpackHalf4x16(q2PackHalf4x16(float4(1.0, 2.0, 3.0, 4.0))).y;

    // The density pair, which the header stores scaled by 65536.0 because the values are in the
    // denormal range of fp16, and which q2EvaluateFog divides back by the same constant
    v += q2UnpackHalf2x16(q2PackHalf2x16(float2(0.5, -0.25) * 65536.0)).x;

    // The raw half pack pair, spelled with f32tof16 / f16tof32 here and with the builtins on the
    // other half. 0xC0003C00 is (1.0, -2.0) packed and its high half is -2.0, but neither front end
    // folds the pack of a constant operand, so both halves keep the extended instruction here
    v += (float)q2PackHalf2x16(float2(1.0, -2.0));
    v += q2UnpackHalf2x16(0xC0003C00u).y;

    // Q2_FOG_BRIGHTNESS, the one macro of the header and the multiplier of the color it packs
    v += Q2_FOG_BRIGHTNESS * 100.0;

    // The two closest volumes of the ray, over the five members of the global uniform that only
    // this header reads. The finder cannot fold: every value it uses comes from the uniform
    const float3 origin = float3(1.0, 2.0, 3.0);
    const float3 dir = float3(0.0, 0.0, 1.0);
    uint4 fog1;
    uint4 fog2;
    q2FindFogVolumes(origin, dir, 0.0, 1000.0, fog1, fog2);
    v += (float)fog1.w + (float)fog2.w;

    // The analytic transmittance of the closer volume, the premultiplied blend that both wrappers
    // are built from, and the two wrappers over the furthest span of the ray
    const float4 fog_eval = q2EvaluateFog(fog1, 1.0, 3.0);
    v += fog_eval.x + fog_eval.w;
    v += q2AlphaBlendPremultiplied(fog_eval, float4(0.25, 0.5, 0.75, 1.0)).x;
    v += q2SegmentFog(fog1, fog2, 1000.0).y;
    v += q2ApplyFog(fog1, fog2, 1000.0, float3(0.25, 0.5, 0.75)).z;

    // Slot 0 is the accumulator: it is the one store the reach of the header is read from, because
    // every function above contributed to it. The slots after it hold one site of the header each,
    // with the constants of the site written out, so that the two disassemblies can be read against
    // each other site by site:
    //   [1] q2FogVMin((float3)2.0)               operand: %float_2 twice in %93 = NMin(2, %92),
    //                                            %92 = NMin(2, 2), so all three lanes carry the cast
    //   [2] q2FogVMax((float3)1.0 / float3(2,4,8))   operand: %float_0_5, %float_0_25, %float_0_125
    //                                            in %95 = NMax(0.5, %94), %94 = NMax(0.25, 0.125)
    //   [3] the low half of q2PackHalf4x16(float4(1,2,3,4))   operand: %75 = (1, 0) and
    //                                            %76 = (2, 0), the two-lane form f32tof16 is fed
    //   [4] q2UnpackHalf4x16(uint2(0x3C00, 0x4200)).x   operand %uint_15360 (0x3C00): the "and
    //                                            0xFFFF" of the helper folds away on a literal
    //   [5] the round trip of [3] through q2UnpackHalf4x16 .y   as [3], high half by
    //                                            %106 = %100 >> 16
    //   [6] the density pair at the scale of the header, 0.5 * 65536.0   operands %79 = (32768, 0)
    //                                            and %80 = (-16384, 0); here the "and 0xFFFF" stays,
    //                                            because the word is computed, not a literal
    //   [7] q2PackHalf2x16(float2(1.0, -2.0))     operand %81 = (-2, 0) and %75 = (1, 0), the word
    //                                            itself is the unfolded shift/or
    //   [8] q2UnpackHalf2x16(0xC0003C00u).y       FOLDED operand: the ">> 16" of the literal turns
    //                                            0xC0003C00 into %uint_49152 (0xC000) = -2.0
    //   [9] Q2_FOG_BRIGHTNESS                     FOLDED: %float_0_00999999978 = 0.00999999978
    //  [10] Q2_FOG_BRIGHTNESS * 100.0             FOLDED: %float_1 = 1
    //  [11] to [15]: the five sites that read the global uniform, in the order the header reaches
    //   them -- the two volume indices, the transmittance pair, the blend and the two wrappers.
    //   They cannot fold on either half, because every value they use comes from the uniform, and
    //   they are in this file for reach only. [11] and [12] still show the arithmetic as an OpFAdd
    //   of two loads, so the two read paths are visible even there.
    //
    // [1] to [8] do NOT fold, on either front end: min/max stay OpExtInst NMin / NMax here and
    // FMin / FMax on the glslc half, and the half pack stays OpExtInst PackHalf2x16 / UnpackHalf2x16
    // on both. What folds on both halves is only what is listed in the "operand" column above --
    // the same numbers on both halves, in dxc's zero-second-lane form -- and the two macro sites [9]
    // and [10], which are the same literal on both sides, and the ">> 16" of [8] as a folded
    // operand. Measured with spirv-dis on Build/Q2Fog.probe.comp.{glsl,hlsl}.spv.
    probeOutput[0] = v;
    probeOutput[1] = q2FogVMin((float3)2.0);
    probeOutput[2] = q2FogVMax((float3)1.0 / float3(2.0, 4.0, 8.0));
    probeOutput[3] = (float)q2PackHalf4x16(float4(1.0, 2.0, 3.0, 4.0)).x;
    probeOutput[4] = q2UnpackHalf4x16(uint2(0x3C00u, 0x4200u)).x;
    probeOutput[5] = q2UnpackHalf4x16(q2PackHalf4x16(float4(1.0, 2.0, 3.0, 4.0))).y;
    probeOutput[6] = q2UnpackHalf2x16(q2PackHalf2x16(float2(0.5, -0.25) * 65536.0)).x;
    probeOutput[7] = (float)q2PackHalf2x16(float2(1.0, -2.0));
    probeOutput[8] = q2UnpackHalf2x16(0xC0003C00u).y;
    probeOutput[9] = Q2_FOG_BRIGHTNESS;
    probeOutput[10] = Q2_FOG_BRIGHTNESS * 100.0;
    probeOutput[11] = (float)fog1.w + (float)fog2.w;
    probeOutput[12] = fog_eval.x + fog_eval.w;
    probeOutput[13] = q2AlphaBlendPremultiplied(fog_eval, float4(0.25, 0.5, 0.75, 1.0)).x;
    probeOutput[14] = q2SegmentFog(fog1, fog2, 1000.0).y;
    probeOutput[15] = q2ApplyFog(fog1, fog2, 1000.0, float3(0.25, 0.5, 0.75)).z;
}
