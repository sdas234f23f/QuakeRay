// Copyright (c) 2026 QuakeRay contributors
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
// How much of the sun the cloud layer takes away on its way down to a spot of the
// world, read from the volume CmCloudShadow.comp fills: the layer is a slab of
// cloud over the ground, and what the sun crosses to reach a point of it is the
// column of cloud standing between the two -- a march of that column is what the
// eye cannot pay for per point, so it is walked once per spot of the world, for
// every height a point of the layer can stand at, and read back as a lookup.
//
// The volume is the world's horizontal plane along the sun (x and y), on which the
// layer is projected, and the height over the base of the layer (z): a texel of a
// slice holds the tau of the cloud column that stands above that height, which is
// the whole column at the base of the layer and nothing at its top. A point on the
// ground reads the base slice, and a point inside the layer reads the slice it
// stands at -- what the sun still crosses to reach it is the cloud above it.
//
// The placement of the volume is a parameter of every function here: the passes
// that light the world take it from the global uniform the host leaves it in -- the
// tail of the sky transform, which nothing but the cubemap sky has any other use
// for (see RenderCubemap::UpdateCloudShadow) -- and the sky pass, which has no
// global uniform, from its own parameters (CmSkyClouds.comp).
//
// The volume only covers the part of the world the eye is near, and where it does
// not reach the clouds are simply not covering the sun for what it is asked about,
// so a point outside it keeps the full sun rather than losing it to a guess. That
// is what the blend factor is for, and why the volume fades out instead of ending
// on a line the eye could find.
//

// How far inside the mapped area a point has to be for the volume to be trusted
// for it, in fractions of the volume.
const float CLOUD_SHADOW_FADE = 0.04;

// Where a point of the world stands in the volume, for a sun in the given direction
// (pointing towards it). False means "the cloud layer covers no part of the sun
// here": either the clouds are not casting a shadow at all, or the point is
// beyond what the volume was laid out over. `mapPlacement`: x = 1 while the clouds
// cast a shadow, yz = the world-space corner of the volume (the world being the
// Quake one, up is z, so the corner's x rides in y and its y in z), w = its extent
// in metres.
bool cloudShadowUV(vec3 worldPos, vec3 sunDir, vec4 mapPlacement, out vec2 uv, out float blend)
{
    uv = vec2(0.0);
    blend = 0.0;

    if (mapPlacement.x <= 0.5 || sunDir.z <= 1.0e-3)
    {
        return false;
    }

    // The volume is the layer projected onto the world's horizontal plane along
    // the sun, so a point stands in it where the sun ray through it meets that
    // plane: everything along one such ray shares a texel, the cloud above it being
    // the same cloud.
    vec2 flatPos = worldPos.xy - sunDir.xy * (worldPos.z / sunDir.z);
    vec2 mapped = (flatPos - mapPlacement.yz) / max(mapPlacement.w, 1.0);

    vec2 fade = smoothstep(vec2(0.0), vec2(CLOUD_SHADOW_FADE), mapped) *
                (vec2(1.0) - smoothstep(vec2(1.0 - CLOUD_SHADOW_FADE), vec2(1.0), mapped));
    blend = fade.x * fade.y;
    if (blend <= 0.0)
    {
        return false;
    }

    uv = clamp(mapped, vec2(0.0), vec2(1.0));
    return true;
}

// What the volume holds for a height of the layer: the tau of the cloud column
// that stands above it along the sun (see the top of this header). `uv` is where
// the point stands in the volume (cloudShadowUV) and `height` is its height in
// fractions of the thickness of the layer, 0 at its base.
//
// The slices of the volume stand at even heights over the layer, the first at its
// base and the last at its top, so a height is read with the centres of the slices
// in mind: each holds the value at one of them, and everything between is what the
// sampler interpolates.
float cloudShadowTauFromUV(sampler3D shadowVolume, vec2 uv, float height)
{
    float slices = float(textureSize(shadowVolume, 0).z);
    float z = clamp(height, 0.0, 1.0) * (slices - 1.0) / slices + 0.5 / slices;

    return texture(shadowVolume, vec3(uv, z)).r;
}

// The tau of the cloud between a point of the world and the sun, read from the
// volume of the layer's shadow: the light that reaches the point is exp(-tau) of
// what the sun sends (Beer-Lambert).
//
// Negative when the volume does not reach the point -- it is beyond what it was
// laid out over, or the clouds cast no shadow at all -- and then the caller
// answers for the point with a march towards the sun of its own.
//
// `height` is the fraction of the layer the point stands at: 0 at its base (a
// point on the ground), 1 at its top. `sunDir` points towards the sun and
// `mapPlacement` is where the volume stands (see cloudShadowUV).
float cloudShadowTau(sampler3D shadowVolume, vec3 worldPos, vec3 sunDir, vec4 mapPlacement, float height)
{
    vec2 uv;
    float blend;

    if (!cloudShadowUV(worldPos, sunDir, mapPlacement, uv, blend) || blend < 0.999)
    {
        return -1.0;
    }

    return cloudShadowTauFromUV(shadowVolume, uv, height);
}

// How much of the sun gets past the clouds on the way down to a point of the world:
// 1 with no cloud in the way, 0 with the sun fully behind them. Where the volume
// does not reach the point the clouds are simply not covering the sun for it, so it
// keeps the full sun rather than losing it to a guess: the tau is what fades out
// towards the edge of the volume, and a point beyond it is not shaded at all.
//
// `height` is the fraction of the layer the point stands at, as in cloudShadowTau.
float cloudShadowTransmittance(sampler3D shadowVolume, vec3 worldPos, vec3 sunDir, vec4 mapPlacement, float height)
{
    vec2 uv;
    float blend;

    if (!cloudShadowUV(worldPos, sunDir, mapPlacement, uv, blend))
    {
        return 1.0;
    }

    return mix(1.0, exp(-cloudShadowTauFromUV(shadowVolume, uv, height)), blend);
}
