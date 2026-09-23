// Copyright (C) 2018 Tobias Zirr
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/light_lists.h from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/light_lists.h, adapted to the vkpt
// framework: the per-BSP-cluster light lists (q2LightListOffsets /
// q2LightListLights, built on the CPU from the PVS and uploaded each frame)
// are used verbatim like Q2RTX, and the Q2RTX polygon light sampling is
// replaced by vkpt's ShLightEncoded (sphere/spot/triangle).
//

// HLSL counterpart of Q2LightLists.h. The golden includes no resource layer of its own, because
// both of its consumers -- RtQ2Indirect.rgen and RtRaygenDirect.rgen -- reach it after
// ShaderCommonGLSLFunc.h and Light.h: this header pulls the two counterparts in itself, the
// accessors first so that the light source buffers, globalUniform and the sampled framebuffers
// exist when Light.hlsli is parsed.
//
// Spellings that had to change:
//   * ivec2 -> int2, vec3 -> float3, mat3 -> float3x3. The header has exactly two matrices and
//     both are float3x3: q2SphericalTriArea takes one as a parameter and q2LightSelectionMass
//     builds one, both declared transposed, as the port rules in ShaderCommonHLSL.hlsli require.
//     Their indices live only inside q2SphericalTriArea, see below.
//   * q2GetIsGradient: texelFetch(framebufQ2GradSmplPos_Sampled, pix / Q2_GRAD_DWN, 0) ->
//     framebufQ2GradSmplPos_Sampled.Load(int3(pix / Q2_GRAD_DWN, 0)), the spelling the accessor
//     layer of the ported base uses for the same resource, the samplerless utexture2D of
//     ShaderCommonGLSL.h binding 239 at set DESC_SET_FRAMEBUFFERS (Texture2D<uint4> in
//     ShaderCommonHLSL.hlsli:834). The .r stays .r, the rgba swizzle set of HLSL is the same one.
//   * q2GetIsGradient: all(equal(gradStrataPos, pix % Q2_GRAD_DWN)) -> all(gradStrataPos == pix % Q2_GRAD_DWN).
//     dxc has no equal() intrinsic for integer vectors ("use of undeclared identifier 'equal'"),
//     because the HLSL comparison operators are already componentwise and yield the same bool2 that
//     glslc lowers both spellings to (OpIEqual + OpAll). EfCommon.hlsli:53 is the same spelling on
//     this side of the port. Q2_GRAD_DWN, Q2_STRATUM_OFFSET_SHIFT and Q2_STRATUM_OFFSET_MASK are not
//     defined by the golden either: they arrive from Q2Asvgf.h, which both of its consumers include
//     immediately before this header (RtQ2Indirect.rgen:43, RtRaygenDirect.rgen:36), so the
//     counterpart of that header stays responsible for them here.
//   * q2AccumulateLightStats: atomicAdd(q2LightStats[addr], 1u) -> InterlockedAdd(q2LightStats[addr], 1u).
//     HLSL has no atomicAdd (dxc stops at "use of undeclared identifier"); the two-argument form of
//     InterlockedAdd is the same atomic add that throws its return value away, which is how the
//     golden's statement uses it, and on this side it lowers to the same OpAtomicIAdd with the
//     same scope and semantics that glslc gives the golden's atomicAdd (measured, see the probe).
//   * q2SphericalTriArea: the golden shifts the three points in place with
//     `positions[i] = positions[i] - p`, a write through a single index, which writes a ROW in
//     HLSL and has no counterpart there at all. The three shifted points are therefore named
//     positions0..2, and the matrix is rebuilt from them as rows and transposed in one statement:
//     transpose(float3x3(positions0, positions1, positions2)) has exactly the three shifted points
//     as its columns, so the one later read of the matrix -- the product of the golden's line that
//     normalizes the centroid -- sees the same values. Rebuilding the matrix also keeps the write,
//     and the read of the same variable that follows it, as the golden has them. Every other read
//     of the golden's `positions[i]` -- the cross product, the three dots of the two early outs,
//     and the three normalized points -- is one of those three locals, which is the value the read
//     had in the golden.
//   * q2SphericalTriArea: `normalize(positions * vec3(1.0 / 3.0))` -> `normalize(mul(positions, (float3)(1.0 / 3.0)))`.
//     The operand order is kept, as the product rule requires, and the scalar is broadcast with a
//     C style cast because dxc refuses vec3(x) with a single scalar. The product was NOT rewritten
//     to (c0 + c1 + c2) / 3, which is the same sum mathematically but not bit-exactly: measured on
//     the probe's constants, the golden's order folds to a different float than the sum of the
//     three columns divided by three. The product is fed by the matrix that was rebuilt from
//     positions0..2 above.
//   * q2LightSelectionMass: the golden declares `mat3 positions` and fills it column by column from
//     l.position[0..2]. A column by column fill has no assignment target in HLSL either, so the
//     matrix is built in one expression and transposed: transpose(float3x3(l.position[0],
//     l.position[1], l.position[2])). l.position is the vec3 position[3] array of TriangleLight, so
//     nothing else about it changes.
//   * mix -> lerp, atan(y, x) -> atan2(y, x). Both are the same operations: HLSL's atan takes one
//     argument and its atan atan2 argument order matches GLSL's atan(y, x).
//   * the #ifndef Q2_LIGHT_LISTS_H_ guard -> #ifndef Q2_LIGHT_LISTS_HLSLI_, the mechanism of the
//     golden with the extension of this base, as the include guard rule requires.
//
// What did not change: every constant and its value, the member set of ShLightEncoded that the
// light selection reads, the uint(...) / int(...) / float(...) constructors (HLSL accepts them for
// the scalar types and they truncate and round the same way), the truncating int(...) casts of
// q2SampleClusterLights' partitions, both of its loops and their break conditions, the early outs of
// q2SphericalTriArea and of q2LightSelectionMass, the `max(area - 1e-5, 0.0) * brdf` tail, the
// 0.1 floor of the statistics read, and the pdf of the last accepted mass over totalMassScaled.
// q2Phong's pow(max(dot(reflect(-L, n), V), 0.0), phongExp) is copied as it is, pow(0.0, x)
// included: the golden does not guard it either.
//
// The header declares no descriptor of its own -- it reads the light lists, the light statistics,
// the cluster sky visibility and the sampled gradient positions of the accessor layer -- so the
// probe pair GLSL/Q2LightLists.probe.comp <-> Probes/Q2LightLists.probe.comp.hlsl is what pins it.

#ifndef Q2_LIGHT_LISTS_HLSLI_
#define Q2_LIGHT_LISTS_HLSLI_
#include "ShaderCommonHLSLFunc.hlsli"
#include "Light.hlsli"

#define Q2_MAX_BRUTEFORCE_SAMPLING 16

// Light statistics modes, mirroring the rt_q2_lightstats cvar: they only affect how
// often the statistics are accumulated and whether they are applied, so the modes can
// be compared in-game to measure what the accumulation and the read actually cost.
#define Q2_LIGHT_STATS_DISABLED     0u
#define Q2_LIGHT_STATS_DEFAULT      1u
#define Q2_LIGHT_STATS_FIRST_SAMPLE 2u
#define Q2_LIGHT_STATS_NO_READ      3u
#define Q2_LIGHT_STATS_NON_ATOMIC   4u

uint q2GetClusterLightCount(const uint cluster)
{
    return q2LightListOffsets[cluster + 1] - q2LightListOffsets[cluster];
}

uint q2GetClusterLight(const uint cluster, const uint slot)
{
    return q2LightListLights[q2LightListOffsets[cluster] + slot];
}

/* Per-cluster sky visibility of the host (Q2RTX's sky_visibility): bit c of the table is
   1 when a sun ray from cluster c can still reach the sky, which is the union of the PVS
   of every cluster that holds a sky surface. A cluster the host never classified - the
   solid cluster 0, or one past the table - keeps its sun ray, as Q2RTX keeps it for an
   invalid cluster. */
bool q2ClusterSeesSky(const uint cluster)
{
    if (cluster >= uint(Q2_MAX_CLUSTERS))
    {
        return true;
    }

    return (q2ClusterSkyVis[cluster >> 5] & (1u << (cluster & 31u))) != 0u;
}

bool q2GetIsGradient(const int2 pix)
{
    // The gradient sample positions are only produced by the denoiser; without
    // it no pixel may be treated as a gradient sample (Q2RTX get_is_gradient).
    if (globalUniform.fltEnable.x < 0.5)
    {
        return false;
    }

    const uint u = framebufQ2GradSmplPos_Sampled.Load(int3(pix / Q2_GRAD_DWN, 0)).r;
    if (u == 0u)
    {
        return false;
    }

    const int2 gradStrataPos = int2(
        (u >> (Q2_STRATUM_OFFSET_SHIFT * 0)) & Q2_STRATUM_OFFSET_MASK,
        (u >> (Q2_STRATUM_OFFSET_SHIFT * 1)) & Q2_STRATUM_OFFSET_MASK);

    return all(gradStrataPos == pix % Q2_GRAD_DWN);
}

float q2RoughnessSquareToSpecPower(const float alpha)
{
    return max(0.01, 2.0 / (alpha * alpha + 1e-4) - 2.0);
}

uint q2GetPrimaryDirectionSide(const float3 n)
{
    const float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) return n.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.z)               return n.y >= 0.0 ? 2u : 3u;
    return                            n.z >= 0.0 ? 4u : 5u;
}

uint q2GetLightStatsAddr(const uint cluster, const uint slot, const uint side)
{
    uint addr = cluster;
    addr = addr * uint(Q2_LIGHT_LIST_MAX_PER_CELL) + slot;
    addr = addr * uint(Q2_LIGHT_LIST_STATS_SIDES) + side;
    addr = addr * 2u;
    return addr;
}

// sampleIdx is the index of the NEE light sample within the pixel. The statistics are
// per-(cell, slot, side) counters averaged over the whole frame, so a single
// accumulation per pixel is enough for them to converge; the remaining samples of the
// same pixel only reduce the sample count of the frame's statistics.
void q2AccumulateLightStats(const uint cluster, const uint slot, const float3 n, const float vis,
                            const uint sampleIdx)
{
    const uint mode = globalUniform.q2LightStatsMode;
    if (mode == Q2_LIGHT_STATS_DISABLED ||
        (mode == Q2_LIGHT_STATS_FIRST_SAMPLE && sampleIdx != 0u))
    {
        return;
    }

    const uint frameSlot = globalUniform.frameId % uint(Q2_LIGHT_LIST_STATS_BUFFERS);
    const uint statsFrameBase = frameSlot
        * uint(Q2_MAX_CLUSTERS) * uint(Q2_LIGHT_LIST_MAX_PER_CELL)
        * uint(Q2_LIGHT_LIST_STATS_SIDES) * 2u;
    const uint side = q2GetPrimaryDirectionSide(n);
    const uint addr = statsFrameBase + q2GetLightStatsAddr(cluster, slot, side)
        + (vis > 0.5 ? 0u : 1u);

    if (mode == Q2_LIGHT_STATS_NON_ATOMIC)
    {
        q2LightStats[addr] += 1u;
    }
    else
    {
        InterlockedAdd(q2LightStats[addr], 1u);
    }
}

float q2Phong(float3 n, float3 L, float3 V, float phongExp)
{
    return pow(max(dot(reflect(-L, n), V), 0.0), phongExp);
}

float q2SphericalTriArea(float3x3 positions, float3 p, float3 n, float3 V, float phongExp, float phongScale, float phongWeight)
{
    const float3 positions0 = getColumn(positions, 0) - p;
    const float3 positions1 = getColumn(positions, 1) - p;
    const float3 positions2 = getColumn(positions, 2) - p;
    positions = transpose(float3x3(positions0, positions1, positions2));

    float3 g = cross(positions1 - positions0, positions2 - positions0);
    if (dot(n, positions0) <= 0 && dot(n, positions1) <= 0 && dot(n, positions2) <= 0)
        return 0;
    if (dot(g, positions0) >= 0 && dot(g, positions1) >= 0 && dot(g, positions2) >= 0)
        return 0;

    float3 L = normalize(mul(positions, (float3)(1.0 / 3.0)));
    float specular = q2Phong(n, L, V, phongExp) * phongScale;
    float brdf = lerp(1.0, specular, phongWeight);

    float3 A = normalize(positions0);
    float3 B = normalize(positions1);
    float3 C = normalize(positions2);

    float area = 2 * atan2(abs(dot(A, cross(B, C))), 1 + dot(A, B) + dot(B, C) + dot(A, C));
    return max(area - 1e-5, 0.0) * brdf;
}

float q2LightSelectionMass(const ShLightEncoded encoded, const float3 p, const float3 n, const float3 V,
                           const float phongExp, const float phongScale, const float phongWeight)
{
    if (encoded.lightType == LIGHT_TYPE_TRIANGLE)
    {
        const TriangleLight l = decodeAsTriangleLight(encoded);
        const float3x3 positions = transpose(float3x3(l.position[0], l.position[1], l.position[2]));
        return q2SphericalTriArea(positions, p, n, V, phongExp, phongScale, phongWeight);
    }
    else if (encoded.lightType == LIGHT_TYPE_TEXTURED_AREA)
    {
        const TexturedAreaLight l = decodeAsTexturedAreaLight(encoded);
        const float3 center = getTexturedAreaLightCenter(l);
        const DirectionAndLength centerToSurf = calcDirectionAndLength(center, p);
        return safeSolidAngle(l.area * getGeometryFactorClamped(l.normal, centerToSurf.dir, centerToSurf.len));
    }
    else
    {
        const float dist = max(length(p - encoded.data_0.xyz), encoded.data_0.w);
        float sa = calcSolidAngleForSphere(encoded.data_0.w, dist);

        if (encoded.lightType == LIGHT_TYPE_SPOT)
        {
            const SpotLight l = decodeAsSpotLight(encoded);
            const float3 toLight = normalize(encoded.data_0.xyz - p);
            sa *= getSpotFactor(max(dot(l.direction, toLight), 0.0), l.cosAngleInner, l.cosAngleOuter);
        }

        return sa;
    }
}

void q2SampleClusterLights(
    const uint cluster, const float3 p, const float3 n, const float3 V,
    const float phongExp, const float phongScale, const float phongWeight,
    const bool isGradient, const float3 rng,
    out uint outLightIndex, out uint outSlot, out float outPdf)
{
    outLightIndex = LIGHT_INDEX_NONE;
    outSlot = 0u;
    outPdf = 0.0;

    // The host folds the map into the table these lists are built on, so a cluster past its end
    // means a leaf index reached this far. Reading it would leave the offsets and the lists
    // behind, and the statistics this cell keeps are written to as well.
    if (cluster >= uint(Q2_MAX_CLUSTERS))
    {
        return;
    }

    const int lightCount = int(q2GetClusterLightCount(cluster));
    if (lightCount <= 0)
    {
        return;
    }

    const int listBase = int(q2LightListOffsets[cluster]);

    const float partitions = ceil(float(lightCount) / float(Q2_MAX_BRUTEFORCE_SAMPLING));
    float r0 = rng.x * partitions;
    const int fpart = int(min(floor(r0), partitions - 1.0));
    r0 -= float(fpart);
    const int stride = int(partitions);
    const int listStart = listBase + fpart;

    const uint frameSlot = globalUniform.frameId % uint(Q2_LIGHT_LIST_STATS_BUFFERS);
    const uint statsSlot = isGradient
        ? (frameSlot + uint(Q2_LIGHT_LIST_STATS_BUFFERS) - 2u) % uint(Q2_LIGHT_LIST_STATS_BUFFERS)
        : (frameSlot + uint(Q2_LIGHT_LIST_STATS_BUFFERS) - 1u) % uint(Q2_LIGHT_LIST_STATS_BUFFERS);
    const uint statsFrameBase = statsSlot
        * uint(Q2_MAX_CLUSTERS) * uint(Q2_LIGHT_LIST_MAX_PER_CELL)
        * uint(Q2_LIGHT_LIST_STATS_SIDES) * 2u;
    const uint side = q2GetPrimaryDirectionSide(n);
    const uint statsMode = globalUniform.q2LightStatsMode;

    float masses[Q2_MAX_BRUTEFORCE_SAMPLING];
    float massSum = 0.0;

    for (int i = 0; i < Q2_MAX_BRUTEFORCE_SAMPLING; i++)
    {
        const int nIdx = listStart + i * stride;
        if (nIdx >= listBase + lightCount)
        {
            break;
        }

        const int slot = nIdx - listBase;
        const uint li = q2GetClusterLight(cluster, uint(slot));

        if (li < uint(LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET) ||
            li >= uint(LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET) + globalUniform.lightCount)
        {
            masses[i] = 0.0;
            continue;
        }

        const ShLightEncoded l = lightSources[li];

        float m = q2LightSelectionMass(l, p, n, V, phongExp, phongScale, phongWeight);
        m *= abs(getLuminance(l.color));

        if (m > 0.0 && statsMode != Q2_LIGHT_STATS_DISABLED && statsMode != Q2_LIGHT_STATS_NO_READ)
        {
            const uint statsAddr = statsFrameBase + q2GetLightStatsAddr(cluster, uint(slot), side);
            const uint numHits = q2LightStats[statsAddr];
            const uint numMisses = q2LightStats[statsAddr + 1];
            const uint numTotal = numHits + numMisses;

            if (numTotal > 0)
            {
                m *= max(float(numHits) / float(numTotal), 0.1);
            }
        }

        massSum += m;
        masses[i] = m;
    }

    if (massSum <= 0.0)
    {
        return;
    }

    float r = r0 * massSum;
    const float totalMassScaled = massSum * partitions;
    float pdf = 0.0;
    int selectedSlot = -1;

    for (int i = 0; i < Q2_MAX_BRUTEFORCE_SAMPLING; i++)
    {
        const int nIdx = listStart + i * stride;
        if (nIdx >= listBase + lightCount)
        {
            break;
        }

        pdf = masses[i];
        r -= pdf;
        if (r <= 0.0)
        {
            selectedSlot = nIdx - listBase;
            break;
        }
    }

    if (selectedSlot < 0)
    {
        return;
    }

    outLightIndex = q2GetClusterLight(cluster, uint(selectedSlot));
    outSlot = uint(selectedSlot);
    outPdf = pdf / totalMassScaled;
}

#endif
