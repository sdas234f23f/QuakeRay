// Copyright (C) 2021, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of fog.h from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2): path_tracer_rgen.h::find_fog_volumes,
// path_tracer_transparency.glsl::evaluate_fog / alpha_blend_premultiplied.
// Adapted to the vkpt global uniform (flat fog arrays).
//
// HLSL counterpart of Q2Fog.h. Like the GLSL header it declares no resource of its own: it reads
// globalUniform (the four fog arrays and fogIsActive) and MAX_FOG_VOLUMES, both of which come from
// ShaderCommonHLSLFunc.hlsli -- through ShaderCommonHLSL.hlsli, where globalUniform is declared
// under #ifdef DESC_SET_GLOBAL_UNIFORM. CmQ2Fog.comp and RaygenPrimary.inl, the two consumers of
// the GLSL one, likewise include ShaderCommonGLSLFunc.h before Q2Fog.h. The probe pair
// (GLSL/Q2Fog.probe.comp and Probes/Q2Fog.probe.comp.hlsl) instantiates all nine functions of the
// golden -- q2FogVMin, q2FogVMax, q2PackHalf4x16, q2UnpackHalf4x16, q2FindFogVolumes,
// q2EvaluateFog, q2AlphaBlendPremultiplied, q2SegmentFog, q2ApplyFog -- plus the two half-pack
// helpers this file adds, and its one value macro, on both halves.
//
// Spellings that had to change:
//   * vec2/vec3/vec4 -> float2/float3/float4, uvec2/uvec4 -> uint2/uint4. This file has no ivecN,
//     no matrix, no texture, no sampler and no image, so nothing else of the type mapping applies
//     to it and nothing has to be transposed or split into a view and a sampler
//   * the splat constructors of the golden become casts, because a scalar does not broadcast into
//     a constructor in HLSL: vec3(1.0) -> (float3)1.0 (the reciprocal direction of
//     q2FindFogVolumes), vec4(0) -> (float4)0 (q2EvaluateFog) and uvec4(0) -> (uint4)0
//     (q2FindFogVolumes, q2SegmentFog's accumulator is a (float4)0). The pure re-typing
//     constructors keep their meaning and only change shape: uvec2(a, b) -> uint2(a, b) and so on
//   * packHalf2x16 and unpackHalf2x16 have no HLSL intrinsic. Both are re-spelled as the two
//     helpers q2PackHalf2x16 / q2UnpackHalf2x16 below, which are the only new names this file
//     introduces, and every call site of the golden goes through them:
//       packHalf2x16(v2)   -> f32tof16(v2.x) | (f32tof16(v2.y) << 16)
//       unpackHalf2x16(x)  -> float2(f16tof32(x & 0xFFFFu), f16tof32(x >> 16))
//     On the SPIR-V target dxc lowers f32tof16 to OpExtInst PackHalf2x16 on a two-lane vector
//     whose second lane is zero, and f16tof32 to OpExtInst UnpackHalf2x16 plus a lane extract,
//     which is exactly the instruction glslang emits for the two builtins: the emulation is the
//     same GLSL.std.450 instruction per half, only twice as many of them. The costs of the
//     substitution are the extra shift and or of the pack and one redundant "and 0xFFFF" in front
//     of each low-half unpack. No folded number can be read out for the conversion itself: neither
//     front end folds PackHalf2x16 or UnpackHalf2x16, not even on constant operands (measured on
//     the probe -- glslc keeps OpExtInst PackHalf2x16 on the constant composite (1, -2), dxc keeps
//     it on (-2, 0), the zero second lane being how dxc feeds one scalar to the two-lane
//     instruction). What does fold on both halves are the operands and the arithmetic around them:
//     0.5 * 65536.0 and -0.25 * 65536.0 fold to (32768, -16384) on the glslc half and to
//     (32768, 0) / (-16384, 0) on the dxc half, and the ">> 16" of the high half turns the word
//     0xC0003C00 into the literal 49152 on the dxc half, where the glslc half hands the whole word
//     to the instruction. The name "q2PackHalf2x16" does not exist anywhere else yet; when
//     CmQ2Temporal.comp is ported its four sites need the same pair of helpers
//   * min(a, b) and max(a, b) keep the golden's spelling, but the two front ends do not lower them
//     to the same GLSL.std.450 instruction: glslang emits FMin/FMax, which return the non-NaN
//     operand when one operand is a NaN, dxc emits NMin/NMax, which propagate the NaN. The
//     compiled golden therefore differs from this twin on NaN inputs, and the difference is
//     reachable in q2FindFogVolumes: for an axis-aligned ray whose origin lies exactly on a fog
//     volume's slab plane, (mins - origin) * inv_dir is 0 * inf = NaN in that one lane of t1 and
//     t2 on both halves. glslang's q2FogVMax/q2FogVMin then drop that lane at the inner
//     FMax(v.y, v.z) and hand back a finite t_in/t_out, so the volume is entered; dxc's NMax/NMin
//     keep the NaN through both reductions and through the tMin/tMax clamps, so t_out > t_in is
//     false and the twin skips the volume. There is no HLSL spelling of FMin, so this is reported
//     and not repaired: the golden's behaviour is the reference, and a repair would have to be a
//     decision of the port as a whole, not of one file
//   * 0u and the "!= 0u" comparisons of the golden are legal HLSL and stay as they are; the golden
//     contains no shift on an unsigned literal, so no suffix had to be added
//   * the #ifndef Q2_FOG_H_ / #define Q2_FOG_H_ guard -> #ifndef Q2_FOG_HLSLI_ /
//     #define Q2_FOG_HLSLI_, as in Volumetric.hlsli and Q2LightLists.hlsli. The DESC_SET_
//     GLOBAL_UNIFORM check that the two GLSL consumers satisfy by defining the macro is kept as
//     the same #error, as in Media.hlsli and TurbWarp.hlsli
//
// What did not change: the nine functions of the golden with their names, parameter lists and
// return types, the
// call graph of the golden (q2FindFogVolumes through q2FogVMax/q2FogVMin/q2PackHalf4x16 and the
// two half packs, q2SegmentFog and q2ApplyFog through q2AlphaBlendPremultiplied and q2EvaluateFog,
// which itself goes through q2UnpackHalf4x16), the fields of globalUniform that q2FindFogVolumes
// reads -- fogIsActive[i], fogMins[i].xyz, fogMaxs[i].xyz, fogColor[i].xyz, fogDensity[i] -- and
// the loop bound MAX_FOG_VOLUMES, the sign and order of every arithmetic operation (the slab test
// (mins - origin) * inv_dir, the swap through min/max -- whose two calls keep this spelling but not
// this lowering, see the NaN bullet above -- the two clamps, the density linearization
// dot(density.xyz, dir) * 0.5 and dot(density.xyz, origin) + density.w with its 65536.0 scale, the
// transmittance 1.0 - exp((t1*t1 - t2*t2)*x + (t1 - t2)*y), the premultiplied blend
// src.rgb + dst.rgb*(1.0 - src.a) / src.a + dst.a*(1.0 - src.a) and q2ApplyFog's
// acc.rgb + color*(1.0 - acc.a)), the "out" parameters of q2FindFogVolumes, which HLSL passes the
// same way, the early returns and the two-write cascade of the packed uints (fog1 -> fog2), and
// Q2_FOG_BRIGHTNESS as a macro with the same value. The struct-free packing layout -- .xy the
// color, .z the (t_in, t_out) pair, .w the (variable, constant) density pair -- is copied
// literally, including the swizzled writes packed.xy/.z/.w, which HLSL allows on a local uint4.
//

#ifndef Q2_FOG_HLSLI_
#define Q2_FOG_HLSLI_

#ifndef DESC_SET_GLOBAL_UNIFORM
    #error DESC_SET_GLOBAL_UNIFORM must be defined
#endif

// Global multiplier for the color of fog volumes (Q2RTX "pt_fog_brightness").
#define Q2_FOG_BRIGHTNESS 0.01

float q2FogVMin(float3 v) { return min(v.x, min(v.y, v.z)); }
float q2FogVMax(float3 v) { return max(v.x, max(v.y, v.z)); }

// packHalf2x16 / unpackHalf2x16 re-spelled through f32tof16 / f16tof32 (no HLSL intrinsic), see the
// comment on the spellings above.
uint q2PackHalf2x16(float2 v) { return f32tof16(v.x) | (f32tof16(v.y) << 16); }
float2 q2UnpackHalf2x16(uint v) { return float2(f16tof32(v & 0xFFFFu), f16tof32(v >> 16)); }

uint2 q2PackHalf4x16(float4 v) { return uint2(q2PackHalf2x16(v.xy), q2PackHalf2x16(v.zw)); }
float4 q2UnpackHalf4x16(uint2 v) { return float4(q2UnpackHalf2x16(v.x), q2UnpackHalf2x16(v.y)); }

// Loops over the fog volumes and finds the two closest ones along the ray.
// They are stored in the order of min distance in fog1 (closer) and fog2 (further away).
void q2FindFogVolumes(float3 origin, float3 dir, float tMin, float tMax,
                      out uint4 fog1, out uint4 fog2)
{
    fog1 = (uint4)0;
    fog2 = (uint4)0;

    const float3 inv_dir = (float3)1.0 / dir;
    for (int i = 0; i < MAX_FOG_VOLUMES; i++)
    {
        if (globalUniform.fogIsActive[i] == 0u)
        {
            continue;
        }

        const float3 mins = globalUniform.fogMins[i].xyz;
        const float3 maxs = globalUniform.fogMaxs[i].xyz;
        const float3 color = globalUniform.fogColor[i].xyz;
        const float4 density = globalUniform.fogDensity[i];

        const float3 t1 = (mins - origin) * inv_dir;
        const float3 t2 = (maxs - origin) * inv_dir;
        float t_in = q2FogVMax(min(t1, t2));
        float t_out = q2FogVMin(max(t1, t2));
        t_in = max(t_in, tMin);
        t_out = min(t_out, tMax);

        if (t_out > t_in)
        {
            const float2 first_t_min_max = q2UnpackHalf2x16(fog1.w);
            const float2 second_t_min_max = q2UnpackHalf2x16(fog2.w);

            const bool replaces_first = t_in < first_t_min_max.x || first_t_min_max.y == 0.0;
            const bool replaces_second = t_in < second_t_min_max.x || second_t_min_max.y == 0.0;

            if (replaces_first || replaces_second)
            {
                uint4 packed;
                packed.xy = q2PackHalf4x16(float4(color * Q2_FOG_BRIGHTNESS, 0.0));
                packed.z = q2PackHalf2x16(float2(t_in, t_out));

                // Convert the volumetric density function into a 1D function along the ray
                const float density_variable = dot(density.xyz, dir) * 0.5;
                const float density_constant = dot(density.xyz, origin) + density.w;
                // Scale the density stored here because typical values are very small
                // (in fp16 denormal range)
                packed.w = q2PackHalf2x16(float2(density_variable, density_constant) * 65536.0);

                if (replaces_first)
                {
                    // Push fog1 to fog2, replace fog1 with the new volume
                    fog2 = fog1;
                    fog1 = packed;
                }
                else // if (replaces_second) -- must be true
                {
                    // Replace fog2 with the new volume
                    fog2 = packed;
                }
            }
        }
    }
}

// Analytic transmittance of a linear density fog volume between t1 and t2.
float4 q2EvaluateFog(uint4 fog, float t1, float t2)
{
    const float2 fog_bounds = q2UnpackHalf2x16(fog.z);
    const float2 fog_density = q2UnpackHalf2x16(fog.w) / 65536.0;
    const float3 fog_color = q2UnpackHalf4x16(fog.xy).rgb;

    t1 = max(t1, fog_bounds.x);
    t2 = min(t2, fog_bounds.y);

    if (t1 >= t2)
    {
        return (float4)0;
    }

    // Solution to the differential equation: dL = -(a*t + b) dt,
    // where L is luminance and alpha is (1 - L_out / L_in),
    // a and b are the density function parameters.
    const float alpha = 1.0 - exp((t1 * t1 - t2 * t2) * fog_density.x + (t1 - t2) * fog_density.y);

    return float4(fog_color * alpha, alpha);
}

float4 q2AlphaBlendPremultiplied(float4 src, float4 dst)
{
    return float4(src.rgb + dst.rgb * (1.0 - src.a), src.a + dst.a * (1.0 - src.a));
}

// Accumulated premultiplied fog over [0, tMax] for the two closest volumes on
// the ray (fog1 is closer, fog2 further). Used per ray segment; multiple
// segments are combined with q2AlphaBlendPremultiplied (nearest in front).
float4 q2SegmentFog(uint4 fog1, uint4 fog2, float tMax)
{
    float4 acc = (float4)0;
    if (fog2.w != 0u)
    {
        acc = q2AlphaBlendPremultiplied(q2EvaluateFog(fog2, 0.0, tMax), acc);
    }
    if (fog1.w != 0u)
    {
        acc = q2AlphaBlendPremultiplied(q2EvaluateFog(fog1, 0.0, tMax), acc);
    }
    return acc;
}

// Blends the two closest fog volumes along the ray over [0, tMax] into the surface color.
float3 q2ApplyFog(uint4 fog1, uint4 fog2, float tMax, float3 color)
{
    // Blend the further volume first
    float4 acc = (float4)0;
    if (fog2.w != 0u)
    {
        acc = q2AlphaBlendPremultiplied(q2EvaluateFog(fog2, 0.0, tMax), acc);
    }

    // ...then the closer volume
    if (fog1.w != 0u)
    {
        acc = q2AlphaBlendPremultiplied(q2EvaluateFog(fog1, 0.0, tMax), acc);
    }

    return acc.rgb + color * (1.0 - acc.a);
}

#endif // Q2_FOG_HLSLI_
