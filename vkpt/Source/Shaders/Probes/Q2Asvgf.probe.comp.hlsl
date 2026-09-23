// Hand written counterpart of GLSL/Q2Asvgf.probe.comp.
//
// Q2Asvgf.hlsli has no shader stage of its own, so this pair exists to check Q2Asvgf.hlsli against
// its golden. The ten consumers of the header reach it after ShaderCommonHLSLFunc.hlsli, and the
// header declares no resource of its own: everything it touches goes through the two access macros,
// which take their images as parameters. So the probe enables DESC_SET_FRAMEBUFFERS and instantiates
// both macros on the LF history pair that CmQ2Temporal.comp:346 stores to, framebufQ2HistColorLF_SH
// (binding 91, rgba16f) and framebufQ2HistColorLF_COCG (binding 93, rg16f). Those two images and the
// one incidental read that gives the two constant arrays a non-constant index are the whole of the
// resource usage, and this file has to be edited together with the GLSL one: what one side reaches
// and the other does not is reported as a mismatch, which is exactly the point. Q2_LOAD_SH appears
// nowhere else in the golden tree -- the consumers that read an LF pair define a local helper for it
// -- so this probe is its only instantiation on either half. Probes are not part of the shader
// build, so no probe blob is ever shipped.
//
// The header is mostly constants and packing helpers, and the pair pins all of it on both halves:
// the ten functions (q2InitSH, q2AccumulateSH, q2MixSH, q2IrradianceToSH, q2SHToIrradiance,
// q2PackRGBE, q2UnpackRGBE, q2GetGradient, q2CheckerToFlat, q2FlatToChecker), the struct Q2SH, the
// thirty-three #define lines of which thirty-one carry a value, and the three file-scope constants.
// probeOutput[0] accumulates every function call, and the slots after it carry the parts that are
// expected to fold, so that the number of each one can be read out of the two disassemblies side by
// side:
//
//   probeOutput[1]..[9]    the thirty-one value macros in the nine groups the header declares them in
//   probeOutput[10]        the three constant arrays, indexed with constants
//   probeOutput[11]        the values of the same arrays under an index taken from the LF chroma image,
//                          so that they are still reachable as a dynamic index would reach them; this
//                          one is a gate, not a fold, because its index is data dependent. It reads
//                          copies, because a dynamic index into the header arrays themselves stops dxc
//                          from folding the constant reads of probeOutput[10] (measured, see the note
//                          at probeOutput[10])
//   probeOutput[12]        the splat casts of the header, isolated on constants
//   probeOutput[13]..[15]  the three sites whose fold the two front ends do not agree on (the
//                          (uint3)511 clamp of q2PackRGBE, the abs of q2GetGradient, the unpack of
//                          q2UnpackRGBE)
//   probeOutput[16]        the expansion of Q2_STRATUM_OFFSET_MASK
//
// The re-spellings this file has to prove are the casts that replace the golden's single-scalar
// constructors: float4(0.0) -> (float4)0.0 in q2InitSH, float4(s)/float2(s) -> (float4)s/(float2)s
// in q2MixSH, float3(0.0) -> (float3)0.0 in q2PackRGBE and q2SHToIrradiance, uint3(511) ->
// (uint3)511 in q2PackRGBE, and uint3(round(va * scale)) -> (uint3)round(va * scale). probeOutput[12]
// isolates the spelling on constants: (float4)0.5, (float2)0.0625, (float3)0.25 and
// min((uint3)511, uint3(514u, 6u, 519u)), whose .y and .z lanes are 6.0 and 511.0 only if the cast
// filled all three lanes of the min bound. The float2(Co, Cg), float3(R, G, B) and float4((sh).CoCg,
// 0.0, 0.0) constructors of the header collect distinct components and are not casts on either half;
// they are here as themselves. The one place where the two halves cannot be spelled the same way is
// the Q2SH(...) of the golden's own Q2_LOAD_SH: HLSL has no constructor for a struct, so the HLSL
// twin collects the two loads in q2MakeSH(float4, float2), the single addition of Q2Asvgf.hlsli. The
// GLSL half has to write its one LF pair read out in full instead: texelFetch is not overloaded for
// an image2D, so the golden's Q2_LOAD_SH does not compile (measured: 'texelFetch : no matching
// overloaded function found'), while RWTexture2D.Load is exactly the mip-0 fetch it meant. Both
// halves read the same two texels and reach the same two descriptors.
//
// dxc reports one warning per RWTexture2D.Load on this header: 'implicit truncation of vector type'
// on the int3(p, 0) argument. It is measured noise -- 3 of them here, two for the Q2_LOAD_SH pair and
// one for the chroma read of the array index, and t13 measured the same warning for int3(p.x, p.y, 0)
// and for a float2 target, and none for the Texture2D form -- and the port's contract is a zero exit
// code, not a warning-free build.

#define DESC_SET_FRAMEBUFFERS 1

#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Asvgf.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    // q2InitSH, and the four lanes of the two members it zeroes
    Q2SH sh = q2InitSH();
    v += sh.shY.x + sh.shY.w + sh.CoCg.x + sh.CoCg.y;

    // q2AccumulateSH, on a second SH collected through the q2MakeSH helper with the same operands the
    // GLSL half gives the golden's struct constructor, so that the two operands are distinct:
    // shY = (1,2,3,4) * 0.5
    Q2SH b = q2MakeSH(float4(1.0, 2.0, 3.0, 4.0), float2(0.5, -0.5));
    q2AccumulateSH(sh, b, 0.5);
    v += sh.shY.w + sh.CoCg.x;

    // q2MixSH, and the (float4)s / (float2)s splats of its two lerps
    Q2SH m = q2MixSH(b, sh, 0.25);
    v += m.shY.x + m.shY.w + m.CoCg.y;

    // q2IrradianceToSH and q2SHToIrradiance: the L11/L1-1/L10/L00 swap, the chroma of the DC terms,
    // the two constants of the irradiance evaluation and the YCoCg round trip back to RGB
    const Q2SH irr = q2IrradianceToSH(float4(0.25, 0.5, 0.75, 1.0), float4(1.0, 0.5, 0.25, 0.75),
                                      float4(0.125, 0.25, 0.5, 0.5));
    v += irr.shY.w + irr.CoCg.x + irr.CoCg.y;
    v += dot(q2SHToIrradiance(irr, float3(0.0, 0.0, 1.0)), float3(1.0, 2.0, 4.0));
    v += q2SHToIrradiance(irr, float3(0.0, 1.0, 0.0)).z;

    // q2PackRGBE: a positive triple, a triple with a negative component (the max against (float3)0.0),
    // the early out of the all-zero triple, and the clamp of the max channel against (uint3)511
    v += (float)q2PackRGBE(float3(1.0, 2.0, 3.0));
    v += (float)q2PackRGBE(float3(1.0, -2.0, 3.0));
    v += (float)q2PackRGBE(float3(0.0, 0.0, 0.0));
    v += (float)q2PackRGBE(float3(1.9999999, 0.0, 0.0));

    // q2UnpackRGBE, on a raw packed word whose three mantissas differ from each other, and on a word
    // that q2PackRGBE produced, which is the round trip the LF/HF/SPEC channels rely on
    v += dot(q2UnpackRGBE(0x9FC80201u), float3(1.0, 2.0, 4.0));
    v += dot(q2UnpackRGBE(q2PackRGBE(float3(1.0, 2.0, 3.0))), float3(1.0, 2.0, 4.0));

    // q2GetGradient, with the square of the relative difference and with the early out
    v += q2GetGradient(1.0, 4.0);
    v += q2GetGradient(0.0, 0.0);

    // q2CheckerToFlat and q2FlatToChecker, both branches of each ternary and the int divisions
    v += (float)(q2CheckerToFlat(int2(3, 2), 8).x + q2CheckerToFlat(int2(6, 3), 8).x);
    v += (float)(q2FlatToChecker(int2(5, 3), 8).x + q2FlatToChecker(int2(6, 3), 8).x);

    // The LF history pair, through the two access macros. Q2_LOAD_SH is only instantiated here; see
    // the header comment for why the GLSL half writes the same two reads out in full instead
    const Q2SH ld = Q2_LOAD_SH(framebufQ2HistColorLF_SH, framebufQ2HistColorLF_COCG, int2(3, 5));
    v += ld.shY.x + ld.CoCg.y;

    Q2SH st = q2MixSH(ld, b, 0.5);
    Q2_STORE_SH(framebufQ2HistColorLF_SH, framebufQ2HistColorLF_COCG, int2(3, 5), st);

    probeOutput[0] = v;

    // The thirty-one value macros, in nine groups. Every group is a constant expression, so each one
    // is a single folded number in both disassemblies
    probeOutput[1] = Q2_GRAD_DWN + Q2_STRATUM_OFFSET_SHIFT + Q2_STRATUM_OFFSET_MASK;              // 3 + 3 + 7
    probeOutput[2] = Q2_STORAGE_SCALE_LF + Q2_STORAGE_SCALE_HF + Q2_STORAGE_SCALE_SPEC + Q2_STORAGE_SCALE_HDR;
    probeOutput[3] = Q2_FLT_ANTILAG_HF + Q2_FLT_ANTILAG_LF + Q2_FLT_ANTILAG_SPEC + Q2_FLT_ANTILAG_SPEC_MOTION
                   + Q2_FLT_GRAD_WEAPON;
    probeOutput[4] = Q2_FLT_MIN_ALPHA_COLOR_HF + Q2_FLT_MIN_ALPHA_COLOR_LF + Q2_FLT_MIN_ALPHA_COLOR_SPEC
                   + Q2_FLT_MIN_ALPHA_MOMENTS_HF;
    probeOutput[5] = Q2_FLT_TEMPORAL_HF + Q2_FLT_TEMPORAL_LF + Q2_FLT_TEMPORAL_SPEC;
    probeOutput[6] = Q2_ATROUS_ITERATIONS_LF + Q2_ATROUS_ITERATIONS_HF + Q2_ATROUS_ITERATIONS_SPEC;
    probeOutput[7] = Q2_FLT_ATROUS_DEPTH + Q2_FLT_ATROUS_NORMAL_LF + Q2_FLT_ATROUS_NORMAL_HF
                   + Q2_FLT_ATROUS_NORMAL_SPEC + Q2_FLT_ATROUS_LUM_HF;
    probeOutput[8] = Q2_FLT_ATROUS_DEFLICKER_LF + Q2_FLT_DEFLICKER_DEPTH + Q2_FLT_DEFLICKER_NORMAL;
    probeOutput[9] = Q2_DEPTH_GRAD_MIN_STEP;

    // The three constant arrays with constant indices. dxc folds these reads only while no dynamic
    // index into the same array exists anywhere in the module: measured on a pair of scratch files,
    // the same sum is the constant 3.3125 when nothing indexes the arrays dynamically and becomes a
    // chain of loads out of a Function-storage array as soon as one does, glslc folding the constant
    // reads in both cases. The gate below therefore reads copies of the values.
    probeOutput[10] = Q2_GAUSSIAN_KERNEL[0][0] + Q2_GAUSSIAN_KERNEL[0][1] + Q2_GAUSSIAN_KERNEL[1][0]
                    + Q2_GAUSSIAN_KERNEL[1][1] + Q2_WAVELET_FACTOR
                    + Q2_WAVELET_KERNEL[0][0] + Q2_WAVELET_KERNEL[0][1] + Q2_WAVELET_KERNEL[1][0]
                    + Q2_WAVELET_KERNEL[1][1];

    // The same values under an index that is not a constant: the LF chroma image decides which of the
    // two rows and columns is taken. The image load is spelled .Load here and imageLoad on the GLSL
    // half, and this is the only read of that image in the file that does not go through Q2_LOAD_SH.
    // The copies are brace initializers here and array constructors on the GLSL half
    const float gaussianCopy[2][2] = { { Q2_GAUSSIAN_KERNEL[0][0], Q2_GAUSSIAN_KERNEL[0][1] },
                                       { Q2_GAUSSIAN_KERNEL[1][0], Q2_GAUSSIAN_KERNEL[1][1] } };
    const float waveletCopy[2][2] = { { Q2_WAVELET_KERNEL[0][0], Q2_WAVELET_KERNEL[0][1] },
                                      { Q2_WAVELET_KERNEL[1][0], Q2_WAVELET_KERNEL[1][1] } };
    const int k = (int)framebufQ2HistColorLF_COCG.Load(int3(0, 0, 0)).x & 1;
    probeOutput[11] = gaussianCopy[k][k] + waveletCopy[k][k] + Q2_WAVELET_FACTOR;

    // The splat casts of the header, isolated on constants. The two lanes of the min bound are the
    // proof that the cast filled all three lanes: had (uint3)511 filled only .x, .y would clamp to
    // 0.0 and .z would clamp to 0.0 for a bound of 0 in that lane
    const float4 splat4 = (float4)0.5;
    const float2 splat2 = (float2)0.0625;
    const float3 splat3 = (float3)0.25;
    const uint3 clamp3 = min((uint3)511, uint3(514u, 6u, 519u));
    probeOutput[12] = splat4.x + splat4.w + splat2.y + splat3.z + (float)clamp3.y + (float)clamp3.z;

    // The three sites the front ends fold differently, all three of them calls, so that neither
    // front end folds them here: glslc runs without any optimization (the checker's command line has
    // no -O) and does not inline them, and dxc inlines q2UnpackRGBE and folds that one to its
    // constant while the RoundEven and FAbs of the other two survive as extended instructions.
    // `spirv-opt -O --target-env=vulkan1.2` on top of both modules narrows this to two: glslc then
    // folds [15] to that same 3.89648438, and [13]/[14] stay unfolded on both sides -- dxc keeps
    // NMin/NMax/FAbs/RoundEven as extended instructions in its SPIR-V, so [14]'s early-out is still
    // a Phi where glslc has resolved the branch and left a straight line.
    probeOutput[13] = (float)q2PackRGBE(float3(1.9999999, 0.0, 0.0));
    probeOutput[14] = q2GetGradient(1.0, 4.0);
    probeOutput[15] = dot(q2UnpackRGBE(0x9FC80201u), float3(1.0, 2.0, 4.0));

    // The expansion of Q2_STRATUM_OFFSET_MASK, which the a-trous reads of the pass write out next to
    // the shift; it has to fold to the same 7 as the mask itself
    probeOutput[16] = (float)((1 << Q2_STRATUM_OFFSET_SHIFT) - 1);
}
