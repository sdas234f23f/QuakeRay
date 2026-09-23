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
// How much of the sun the cloud layer lets past on its way down to a spot of the
// world, read from the map CmCloudShadow.comp fills: the layer projected onto the
// world's horizontal plane along the sun, so a point only has to be looked up, not
// marched through.
//
// Needs the global uniform (ShaderCommonGLSLFunc.h), which the host puts the map's
// layout in -- the tail of the sky transform, which nothing but the cubemap sky
// has any other use for. See RenderCubemap::UpdateCloudShadow.
//
// The map only covers the part of the world the eye is near, and where it does not
// reach the clouds are simply not covering the sun for what it is asked about, so
// a point outside it keeps the full sun rather than losing it to a guess. That is
// what the blend factor is for, and why the map fades out instead of ending on a
// line the eye could find.
//

// How far inside the mapped area a point has to be for the map to be trusted for
// it, in fractions of the map.
const float CLOUD_SHADOW_FADE = 0.04;

// Where a point of the world stands in the map, for a sun in the given direction
// (pointing towards it). False means "the cloud layer covers no part of the sun
// here": either the clouds are not casting a shadow at all, or the point is
// beyond what the map was laid out over.
bool cloudShadowUV(vec3 worldPos, vec3 sunDir, out vec2 uv, out float blend)
{
    uv = vec2(0.0);
    blend = 0.0;

    // x = 1 while the clouds cast a shadow, yz = the world-space corner of the map
    // (the world being the Quake one, up is z, so the corner's x rides in y and
    // its y in z), w = its extent in metres. The host leaves all of it at zero when
    // the sky is anything but the procedural one, as the cubemap sky puts its own
    // meaning into that transform.
    vec4 map = globalUniform.skyCubemapRotationTransform[3];
    if (map.x <= 0.5 || sunDir.z <= 1.0e-3)
    {
        return false;
    }

    // The map is the layer projected onto the world's horizontal plane along the
    // sun, so a point stands in it where the sun ray through it meets that plane:
    // everything along one such ray shares a texel, the cloud above it being the
    // same cloud.
    vec2 flatPos = worldPos.xy - sunDir.xy * (worldPos.z / sunDir.z);
    vec2 mapped = (flatPos - map.yz) / max(map.w, 1.0);

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

// How much of the sun gets past the clouds on the way down to a point of the
// world: 1 with no cloud in the way, 0 with the sun fully behind them. `sunDir`
// points towards the sun and `shadowMap` holds the transmittance.
float cloudShadowAt(sampler2D shadowMap, vec3 worldPos, vec3 sunDir)
{
    vec2 uv;
    float blend;

    if (!cloudShadowUV(worldPos, sunDir, uv, blend))
    {
        return 1.0;
    }

    return mix(1.0, texture(shadowMap, uv).r, blend);
}
