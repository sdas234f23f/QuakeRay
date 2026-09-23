// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf.glsl from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
// which is distributed under the terms of the GNU General Public License
// version 2.  It has been adapted to the renderer interface of this project.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// Ported from Q2RTX (GPL v2, Copyright (C) 2018 Christoph Schied,
// Copyright (C) 2019, NVIDIA CORPORATION) - shader/asvgf.glsl, utils.glsl.
//
// HLSL counterpart of Q2Asvgf.h. Like the GLSL one it includes nothing itself and declares no
// resource of its own: everything it needs arrives through the two macros. The ten consumers of
// the header (CmQ2Adapter.comp, CmQ2Atrous.comp, CmQ2AtrousLF.comp, CmQ2GradientAtrous.comp,
// CmQ2GradientImg.comp, CmQ2GradientReproject.comp, CmQ2Temporal.comp, RaygenPrimary.inl,
// RtQ2Indirect.rgen, RtRaygenDirect.rgen) include ShaderCommonGLSLFunc.h before it, so the two
// macros are the only place in the whole pass that touches a descriptor through this header, and
// the probe pair GLSL/Q2Asvgf.probe.comp <-> Probes/Q2Asvgf.probe.comp.hlsl is what pins them: it
// instantiates the two macros on the LF history pair (framebufQ2HistColorLF_SH, binding 91,
// rgba16f, and framebufQ2HistColorLF_COCG, binding 93, rg16f, the pair CmQ2Temporal.comp:346
// stores to) and every one of the thirty-three #define lines, the three constants and the ten
// functions of the file on both halves. Note that this header is the only file that mentions
// Q2_LOAD_SH at all: see the note on it below.
//
// Spellings that had to change:
//   * vec2/vec3/vec4 -> float2/float3/float4, uvec3 -> uint3, ivec2 -> int2; uint, int, bool and
//     float keep their names. The file has no matrix, no texture, no sampler and no image
//     declaration of its own, so nothing has to be transposed and no descriptor has to be split
//   * the single-scalar constructors of the golden become casts, which HLSL does not need as
//     constructors because a scalar broadcasts: vec4(0.0) -> (float4)0.0 and vec2(0.0) ->
//     (float2)0.0 (q2InitSH), vec4(s) -> (float4)s and vec2(s) -> (float2)s (q2MixSH, the two
//     float splats of the file), vec3(0.0) -> (float3)0.0 (q2PackRGBE and q2SHToIrradiance, four
//     sites together), uvec3(511) -> (uint3)511 (q2PackRGBE). The broadcast is not an assumption:
//     the probe isolates the casts on constants and both halves fold them to the same 518.3125
//     (0.5 + 0.5 + 0.0625 + 0.25 + 6 + 511). The two 0.5 are the .x and .w of (float4)0.5, so all
//     four lanes carry the scalar, the 0.0625 and 0.25 are the .y and .z of (float2)0.0625 and
//     (float3)0.25, and the 6 and 511 are the .y and .z of min((uint3)511, uint3(514u, 6u, 519u)):
//     a cast that had filled only .x would clamp both of those lanes to 0 and the sum would be
//     0.8125 instead. The
//     constructors that collect several distinct components keep their meaning and only change
//     spelling: vec2(Co, Cg) -> float2(Co, Cg), vec3(R, G, B) -> float3(R, G, B),
//     vec4((sh).CoCg, 0.0, 0.0) -> the same with float4, and the four-component shY constructor
//     of q2IrradianceToSH is copied untouched
//   * uvec3(round(va * scale)) -> (uint3)round(va * scale): the same round-then-truncate, with the
//     conversion written as a cast (the GLSL uvec3(...) of a float vector is that same conversion).
//     round itself keeps its name, but the two compilers do not agree on what it lowers to: dxc
//     emits RoundEven, glslang emits Round. The two agree on every input that is not exactly
//     between two integers, and the values that reach the conversion are masked to 9 bits
//     afterwards, so the difference can only show up for a tie, as Q2RTX's own shader already had
//     to live with (GLSL leaves the tie direction to the implementation)
//   * Q2SH(texelFetch(img_shY, p, 0), texelFetch(img_CoCg, p, 0).xy) cannot be spelled in HLSL:
//     dxc rejects a constructor call on a struct ("constructors only defined for numeric base
//     types") and it rejects the brace initializer in expression position as well, so Q2_LOAD_SH
//     expands to the new helper q2MakeSH, which is the only name this file adds and which performs
//     those same two loads, of shY and of CoCg, in that order. The helper and the struct are
//     declared before the two macros so that the macro bodies refer to names that are already
//     known at the point of their definition; a macro in HLSL is expanded where it is used, so
//     this reordering of the golden is a readability choice and not a necessity. Note that the
//     golden's own reason for using a macro -- "GLSL requires image formats to match, so
//     different-format images cannot be passed to the same function parameter" -- does not survive
//     the port: dxc accepts the rgba16f and the rg16f storage image as two parameters of one
//     function and emits both formats (measured, exit 0). The macro is kept because it is the
//     golden's spelling of the access and because the probe then pins the identical expression on
//     both halves
//   * texelFetch(img, p, 0) -> img.Load(int3(p, 0)) and imageStore(img, p, v) -> img[p] = v, the
//     same substitutions as in every other port of the pass. Q2_STORE_SH is still a braced macro
//     that performs both stores, in the same order, shY first; its six call sites in the golden
//     (CmQ2AtrousLF.comp:263/267/271/275, CmQ2Temporal.comp:346/422) all invoke it as a statement,
//     so the braced form keeps that meaning. Q2_LOAD_SH has no call site anywhere in the golden
//     tree -- the consumers that read an LF pair define their own local q2LoadSH(texture2D,
//     texture2D, ivec2) instead (CmQ2Atrous.comp:44, CmQ2AtrousLF.comp:43, CmQ2Temporal.comp:55),
//     which works for them because both of their images are sampled textures of the same type --
//     so the probe half is the only instantiation of that macro on either side. It is kept for
//     parity with the golden and with Q2RTX, and its body is ported, not removed
//   * mix(a, b, s) -> lerp(a, b, s), the same definition under the HLSL name, as in BRDF.hlsli
//   * the file-scope constant arrays become static const: const float Q2_GAUSSIAN_KERNEL[2][2]
//     and the two of Q2_WAVELET_* -> static const float ..., because a namespace-scope const
//     without static is a uniform whose contents are only known at runtime in HLSL, while it is a
//     compile-time constant in GLSL. Indexing them with a constant folds on both halves (verified:
//     the probe's o[10] sums all eight kernel entries plus Q2_WAVELET_FACTOR and folds to 3.3125 on
//     both halves, and the dynamic indexing of the probe survives on both halves as well)
//   * log2, exp, pow, floor, round, abs, min, max, clamp and dot keep their names and their
//     argument order. log2 is spelled log2 in HLSL too; the substitution that the pass makes for
//     the vector form, log2(vec3) -> log(vec3) / log(2.0), does not apply here because every log2
//     of this file is scalar
//   * the #ifndef Q2_ASVGF_H_ / #define Q2_ASVGF_H_ guard -> #ifndef Q2_ASVGF_HLSLI_ /
//     #define Q2_ASVGF_HLSLI_, as in Q2LightLists.hlsli. Q2_STRATUM_OFFSET_MASK keeps its
//     ((1 << Q2_STRATUM_OFFSET_SHIFT) - 1) spelling: the shift of an int literal is an int in both
//     languages, so the macro needs no unsigned suffix, and the probe shows it folding to 7 on
//     both halves
//
// What did not change: the thirteen declarations of the golden with their names, their parameter
// lists and their return types -- the ten functions q2InitSH, q2AccumulateSH, q2MixSH,
// q2IrradianceToSH, q2SHToIrradiance, q2PackRGBE, q2UnpackRGBE, q2GetGradient, q2CheckerToFlat and
// q2FlatToChecker, the two access macros and the struct Q2SH -- plus the struct with its two
// members and the order shY, CoCg, the thirty-three #define lines of the file, of which
// thirty-one carry a value (Q2_GRAD_DWN, Q2_STRATUM_OFFSET_SHIFT, Q2_STRATUM_OFFSET_MASK, the four
// Q2_STORAGE_SCALE_*, the twelve Q2_FLT_* of the antilag, weapon, min-alpha and temporal group,
// Q2_ATROUS_ITERATIONS_LF/HF/SPEC, Q2_FLT_ATROUS_DEPTH, the three ATROUS_NORMAL_*,
// Q2_FLT_ATROUS_LUM_HF, Q2_FLT_ATROUS_DEFLICKER_LF, Q2_FLT_DEFLICKER_DEPTH,
// Q2_FLT_DEFLICKER_NORMAL and Q2_DEPTH_GRAD_MIN_STEP) and two are the access macros, the three
// constants with their values and their initializers, the
// SH storage layout, the coefficient order of q2IrradianceToSH (the L11/L1-1/L10/L00 swap that its
// comment describes), the YCoCg round trip of q2SHToIrradiance with its T/G/B/R order, the chroma
// clamp to [0, 8], the 1e-6 of the DC guard, the two constants of the irradiance evaluation
// (1.023326, 0.886226) and its factor 2.0, the RGBE layout of q2PackRGBE and q2UnpackRGBE (exponent
// at bit 27 with its -20 bias, mantissas at 0/9/18 with their 0x1ffu masks, the 256.0 and the
// *256.0/x256.0 pair of the scale), the early outs of q2PackRGBE and q2GetGradient, the
// square of q2GetGradient and its comment, and the arithmetic of the two checkerboard conversions,
// including the (pos.y & 1) ^ 1 of q2CheckerToFlat, the comparisons of both is_even_checkerboard
// booleans and the truncating int divisions of half_width and of q2FlatToChecker
//

#ifndef Q2_ASVGF_HLSLI_
#define Q2_ASVGF_HLSLI_

// The ASVGF works on three lighting channels:
//   - LF:  indirect diffuse, stored as luma spherical harmonics (4 coeffs) in
//          YCoCg color space. 1/3 resolution.
//   - HF:  direct diffuse irradiance, packed RGBE.
//   - SPEC: demodulated specular irradiance, packed RGBE.

#define Q2_GRAD_DWN 3
#define Q2_STRATUM_OFFSET_SHIFT 3
#define Q2_STRATUM_OFFSET_MASK ((1 << Q2_STRATUM_OFFSET_SHIFT) - 1)

// Q2RTX storage scales: colors are multiplied by these when stored to keep
// precision in 16-bit / 9-bit formats, and divided back during compositing.
#define Q2_STORAGE_SCALE_LF 1024.0
#define Q2_STORAGE_SCALE_HF 32.0
#define Q2_STORAGE_SCALE_SPEC 32.0
#define Q2_STORAGE_SCALE_HDR 128.0

#define Q2_FLT_ANTILAG_HF 1.0
#define Q2_FLT_ANTILAG_LF 0.2
#define Q2_FLT_ANTILAG_SPEC 2.0
#define Q2_FLT_ANTILAG_SPEC_MOTION 0.004
#define Q2_FLT_GRAD_WEAPON 0.25
#define Q2_FLT_MIN_ALPHA_COLOR_HF 0.02
#define Q2_FLT_MIN_ALPHA_COLOR_LF 0.01
#define Q2_FLT_MIN_ALPHA_COLOR_SPEC 0.01
#define Q2_FLT_MIN_ALPHA_MOMENTS_HF 0.01
#define Q2_FLT_TEMPORAL_HF 1.0
#define Q2_FLT_TEMPORAL_LF 1.0
#define Q2_FLT_TEMPORAL_SPEC 1.0

// a-trous filter iterations for LF/HF/SPEC
#define Q2_ATROUS_ITERATIONS_LF 4
#define Q2_ATROUS_ITERATIONS_HF 4
#define Q2_ATROUS_ITERATIONS_SPEC 4
#define Q2_FLT_ATROUS_DEPTH 0.5
#define Q2_FLT_ATROUS_NORMAL_LF 8.0
#define Q2_FLT_ATROUS_NORMAL_HF 16.0
#define Q2_FLT_ATROUS_NORMAL_SPEC 1.0
#define Q2_FLT_ATROUS_LUM_HF 16.0
// Ratio against the mean of the valid neighbours used by the LF deflicker pass
// (CmQ2AtrousLF.comp, iteration 0). It has to stay >= 1 so that flat regions are left
// untouched; Q2RTX uses flt_atrous_deflicker_lf = 2.0, and 0.75 darkened flat regions.
#define Q2_FLT_ATROUS_DEFLICKER_LF 2.0

// Neighbour validation of the LF deflicker pass: a tap only counts as a lower bound when
// it lies on the same surface as the center, so that a dark silhouette in front of a lit
// wall (weapon, screen frame) cannot drag the limit down. Same criteria as the history
// validation of CmQ2Temporal.comp: relative depth difference and geometric normal.
#define Q2_FLT_DEFLICKER_DEPTH 0.1
#define Q2_FLT_DEFLICKER_NORMAL 0.5

// Lower clamp for fwidth_depth: the reciprocal of the per-pixel depth change of
// the pixel footprint (Q2RTX path_tracer_rgen.h). Same value as upstream, in
// world units (1 Quake unit ~= 2.4 cm).
#define Q2_DEPTH_GRAD_MIN_STEP 0.1

static const float Q2_GAUSSIAN_KERNEL[2][2] = {
    { 1.0 / 4.0, 1.0 / 8.0  },
    { 1.0 / 8.0, 1.0 / 16.0 }
};

static const float Q2_WAVELET_FACTOR = 0.5;
static const float Q2_WAVELET_KERNEL[2][2] = {
    { 1.0, Q2_WAVELET_FACTOR  },
    { Q2_WAVELET_FACTOR, Q2_WAVELET_FACTOR * Q2_WAVELET_FACTOR }
};

// Q2RTX-style SH: luma in 4 SH coefficients + chroma (YCoCg).
struct Q2SH
{
    float4 shY;
    float2 CoCg;
};

// The Q2SH(load, load) of the golden: HLSL has no constructor for a struct, so the two loads that
// Q2_LOAD_SH makes are collected here. This declaration is the only addition of the file.
Q2SH q2MakeSH(float4 shY, float2 CoCg)
{
    Q2SH r;
    r.shY = shY;
    r.CoCg = CoCg;
    return r;
}

// Use macros for storage image access: GLSL requires image formats to match,
// so different-format images cannot be passed to the same function parameter.
// (Same workaround as Q2RTX's STORE_SH.)
#define Q2_LOAD_SH(img_shY, img_CoCg, p) (q2MakeSH(img_shY.Load(int3(p, 0)), img_CoCg.Load(int3(p, 0)).xy))
#define Q2_STORE_SH(img_shY, img_CoCg, p, sh) { img_shY[p] = (sh).shY; img_CoCg[p] = float4((sh).CoCg, 0.0, 0.0); }

Q2SH q2InitSH()
{
    Q2SH r;
    r.shY = (float4)0.0;
    r.CoCg = (float2)0.0;
    return r;
}

void q2AccumulateSH(inout Q2SH accum, Q2SH b, float scale)
{
    accum.shY += b.shY * scale;
    accum.CoCg += b.CoCg * scale;
}

Q2SH q2MixSH(Q2SH a, Q2SH b, float s)
{
    Q2SH r;
    r.shY = lerp(a.shY, b.shY, (float4)s);
    r.CoCg = lerp(a.CoCg, b.CoCg, (float2)s);
    return r;
}

// Convert an RGB irradiance SH (per-channel L00,L1-1,L10,L11) into the
// Q2RTX YCoCg luma-SH + chroma representation used by the LF channel.
// shY = luma coefficients, CoCg = chroma from the DC (L00) terms.
// x,y,z,w per color channel, while the Q2RTX luma-SH is stored as
// vec4(L11, L1-1, L10, L00). The order must be swapped here.
Q2SH q2IrradianceToSH(const float4 shR, const float4 shG, const float4 shB)
{
    Q2SH r;

    // Y = 0.25*R + 0.5*G + 0.25*B  (from the YCoCg transform, applied per coefficient)
    r.shY = float4(
        0.25 * shR.w + 0.5 * shG.w + 0.25 * shB.w,  // L11
        0.25 * shR.y + 0.5 * shG.y + 0.25 * shB.y,  // L1-1
        0.25 * shR.z + 0.5 * shG.z + 0.25 * shB.z,  // L10
        0.25 * shR.x + 0.5 * shG.x + 0.25 * shB.x); // L00

    // chroma from the DC terms
    const float Co = shR.x - shB.x;
    const float Cg = shG.x - 0.5 * (shR.x + shB.x);
    r.CoCg = float2(Co, Cg);

    return r;
}

// Evaluate the Q2RTX LF SH at a given normal, back to RGB irradiance.
float3 q2SHToIrradiance(const Q2SH sh, const float3 N)
{
    const float d = dot(sh.shY.xyz, N);
    float Y = 2.0 * (1.023326 * d + 0.886226 * sh.shY.w);
    Y = max(Y, 0.0);

    // Guard against chroma explosion when the DC (L00) term is near zero
    const float chromaScale = clamp(Y * 0.282095 / (sh.shY.w + 1e-6), 0.0, 8.0);
    const float2 CoCg = sh.CoCg * chromaScale;

    const float T = Y - CoCg.y * 0.5;
    const float G = CoCg.y + T;
    const float B = T - CoCg.x * 0.5;
    const float R = B + CoCg.x;

    return max(float3(R, G, B), (float3)0.0);
}

// Q2RTX packed RGBE (9 bit mantissa, 5 bit exponent)
uint q2PackRGBE(float3 v)
{
    float3 va = max((float3)0.0, v);
    float max_abs = max(va.r, max(va.g, va.b));
    if (max_abs == 0.0)
    {
        return 0u;
    }

    float exponent = floor(log2(max_abs));

    uint result = uint(clamp(exponent + 20.0, 0.0, 31.0)) << 27;

    float scale = pow(2.0, -exponent) * 256.0;
    uint3 vu = min((uint3)511, (uint3)round(va * scale));
    result |= vu.r;
    result |= vu.g << 9;
    result |= vu.b << 18;

    return result;
}

float3 q2UnpackRGBE(uint x)
{
    int exponent = int(x >> 27) - 20;
    float scale = pow(2.0, float(exponent)) / 256.0;

    float3 v;
    v.r = float(x & 0x1ffu) * scale;
    v.g = float((x >> 9) & 0x1ffu) * scale;
    v.b = float((x >> 18) & 0x1ffu) * scale;

    return v;
}

// Relative lighting difference between current and previous frame (Q2RTX)
float q2GetGradient(float l_curr, float l_prev)
{
    float l_max = max(l_curr, l_prev);

    if (l_max == 0.0)
    {
        return 0.0;
    }

    float ret = abs(l_curr - l_prev) / l_max;
    ret *= ret; // make small changes less significant

    return ret;
}

// Convert a checkerboarded pixel position (left/right fields) to flat-screen position
int2 q2CheckerToFlat(int2 pos, int width)
{
    const int half_width = width / 2;
    const bool is_even_checkerboard = pos.x < half_width;

    return int2(
        is_even_checkerboard
            ? (pos.x * 2) + (pos.y & 1)
            : ((pos.x - half_width) * 2) + ((pos.y & 1) ^ 1),
        pos.y);
}

// Convert a flat-screen (regular) pixel position to checkerboarded (left/right fields)
int2 q2FlatToChecker(int2 pos, int width)
{
    const int half_width = width / 2;
    const bool is_even_checkerboard = (pos.x & 1) == (pos.y & 1);

    return int2(
        (pos.x / 2) + (is_even_checkerboard ? 0 : half_width),
        pos.y);
}

#endif // Q2_ASVGF_HLSLI_
