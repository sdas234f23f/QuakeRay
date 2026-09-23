// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/physical_sky.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Procedural atmospheric sky: Rayleigh and Mie single scattering.
//
// Procedural atmospheric sky.
// Single-scattering atmosphere (Rayleigh + Mie) evaluated analytically, no LUTs.
// The per-face camera bases are passed from the CPU side and match the convention
// of the rasterized sky cubemap path (Matrix::GetCubemapViewProjMat).
//
// HLSL counterpart of CmProceduralSky.comp. The golden includes nothing, declares its own
// resources and is the only user of them, so this file does the same and the two halves of the
// pair carry the same set 0: binding 0 and 2 are the two rgba16f cube storage images, binding 1
// is the std140 block of parameters.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in -> the
//     [numthreads(16, 16, 1)] attribute of the entry point, and gl_GlobalInvocationID ->
//     SV_DispatchThreadID, which the golden already converts with ivec3(...) / ivec2(...), so the
//     casts stay and become int3(dispatchThreadID) / int2(dispatchThreadID.xy)
//   * layout(std140, set = 0, binding = 1) uniform Params { ... } params -> the struct Params_BT
//     with the same seven members in the same order and [[vk::binding(1, 0)]]
//     ConstantBuffer<Params_BT> params. The golden's vec4 faceBasis[18] stays a float4[18]; a
//     ConstantBuffer uses the same std140 rules as the GLSL block, so every Offset and the
//     ArrayStride stay where the golden put them
//   * layout(rgba16f, set = 0, binding = 0/2) uniform imageCube -> RWTexture2DArray<float4> with
//     [[vk::image_format("rgba16f")]], because HLSL has no writable cube texture type: dxc
//     rejects RWTextureCube as an unknown template and lands RWTexture2DArray on
//     OpTypeImage 2D with Arrayed 1. The pair checker therefore reports exactly one difference,
//     on both descriptors: GLSL image:Cube:0:0:0:2:Rgba16f against HLSL image:2D:0:1:0:2:Rgba16f.
//     The host binds VK_IMAGE_VIEW_TYPE_CUBE views to these two storage images today
//     (RenderCubemap.cpp CreateAttch), so the two descriptors need a 2D array view of the same
//     six layers for this half -- which is also the only view type D3D12 allows for a cube UAV
//     TODO(refactor): the cube view plus the 2D array view of one image is a port shim, not a
//     design. It goes away with the cubemap-write path in the NVRHI rewrite (A2/A5), which should
//     pick one view model for both backends instead of the two the port needs today.
//   * imageStore(img, pix, v) -> img[pix] = v, imageSize(img).xy -> GetDimensions, and the ivec3
//     store position is the HLSL (x, y, layer) coordinate, so the face ipos.z addresses the same
//     layer as in the golden
//   * vecN(s) of one scalar -> (floatN)s, vec3(a, b, c) -> float3(a, b, c), xyz swizzles and the
//     logical operators keep their spelling
//   * mix -> lerp, fract -> frac (the x - floor(x) intrinsic, same as in Utils.hlsli)
//   * the file scope constants go through static const, as in Q2Asvgf.hlsli: a namespace scope
//     const array in HLSL is a uniform whose contents are only known at runtime, while these are
//     compile time constants that the loop bounds and the folded values below rely on
//   * the up front declarations of the golden (vec2 raySphere, float phaseRayleigh, ...) keep
//     their names, their parameter lists and their return types; every operand order, every
//     clamp, max, min, pow, exp, dot and length is copied as it stands, and no arithmetic
//     expression was re-spelled

struct Params_BT
{
    float4 faceBasis[18]; // 6 faces * (right, up, forward)
    float4 sunDirection;  // xyz = normalized direction TOWARD the sun, w = how much sun the sky shows (0 = no sun)
    float4 skyTint;       // xyz = sky color (rt_sky_color) that tints the atmosphere, w = physical sun angular radius (radians)
    float4 skyParams;     // x = sky color multiplier, y = tint strength, z = sun disc intensity, w = sun disc display radius (radians)
    float4 cloudColor;    // xyz = cloud color, w = cloud time (seconds)
    float4 cloudParams;   // x = cloud coverage, y = cloud density, z = drift speed, w = clouds enabled
    float4 sunDiscColor;  // xyz = color of the sun disc (rt_sun_color), w unused
};

[[vk::binding(1, 0)]] ConstantBuffer<Params_BT> params;

[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2DArray<float4> cubemapOut;
[[vk::binding(2, 0), vk::image_format("rgba16f")]] RWTexture2DArray<float4> envCubemapOut;

static const float PI = 3.14159265358979323846;

static const float PLANET_RADIUS         = 6360.0;   // km
static const float ATMOSPHERE_RADIUS     = 6420.0;   // km
static const float SCALE_HEIGHT_RAYLEIGH = 8.0;      // km
static const float SCALE_HEIGHT_MIE      = 1.2;      // km

// scattering coefficients at sea level (per km), same values as Q2RTX Params_Earth
static const float3  BETA_RAYLEIGH = float3(0.005802, 0.013558, 0.033100);
static const float BETA_MIE      = 0.0014985;
static const float MIE_G         = 0.8;

static const float3  SOLAR_IRRADIANCE = float3(1.474, 1.8504, 1.91198);
static const float RADIANCE_SCALE   = 30.0; // empirical scale to land in a good 0..1 range

float2 raySphere(float3 ro, float3 rd, float radius)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0)
        return (float2)-1.0;
    float s = sqrt(h);
    return float2(-b - s, -b + s);
}

float phaseRayleigh(float mu)
{
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

float phaseMie(float mu)
{
    float g = MIE_G;
    float g2 = g * g;
    return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) /
           ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-5), 1.5));
}

// --- procedural clouds (textureless value noise fBm) ---
float hash13(float3 p)
{
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float valueNoise3(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash13(i);
    float n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0));
    float n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1));
    float n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1));
    float n111 = hash13(i + float3(1, 1, 1));

    return lerp(
        lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
        lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y),
        f.z);
}

float fbm3(float3 p)
{
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++)
    {
        v += amp * valueNoise3(p);
        p = p * 2.03;
        amp *= 0.5;
    }
    return v;
}

float cloudMask(float3 dir, float time, float speed)
{
    // sample the noise on the sky dome; drift slowly with time
    float3 p = dir * 3.0 + float3(time * speed, time * speed * 0.4, 0.0);
    float n = fbm3(p);
    return n;
}

float3 evaluateSky(float3 rd, float3 sunDir, float3 skyTint, float tintStrength, float sunAmount)
{
    float3 camera = float3(0.0, 0.0, PLANET_RADIUS);
    // exit distance of the atmosphere along the view ray
    float2 at = raySphere(camera, rd, ATMOSPHERE_RADIUS);

    if (at.y < 0.0)
        return (float3)0.0;

    const int VIEW_SAMPLES = 16;
    const int SUN_SAMPLES  = 6;

    float t0 = 0.0;
    float t1 = at.y;
    float ds = (t1 - t0) / (float)VIEW_SAMPLES;

    float3 inScatter     = (float3)0.0;
    float3 transmittance = (float3)1.0;
    float3 p             = camera + rd * t0;

    for (int i = 0; i < VIEW_SAMPLES; i++)
    {
        float h = max(length(p) - PLANET_RADIUS, 0.0);
        float densR = exp(-h / SCALE_HEIGHT_RAYLEIGH);
        float densM = exp(-h / SCALE_HEIGHT_MIE);

        // optical depth toward the sun
        float odSunR = 0.0;
        float odSunM = 0.0;
        {
            float3 sp = p;
            float sds = ds;
            for (int j = 0; j < SUN_SAMPLES; j++)
            {
                sp += sunDir * sds;
                float sh = max(length(sp) - PLANET_RADIUS, 0.0);
                odSunR += exp(-sh / SCALE_HEIGHT_RAYLEIGH) * sds;
                odSunM += exp(-sh / SCALE_HEIGHT_MIE) * sds;
            }
        }
        float3 sunTrans = exp(-(BETA_RAYLEIGH * odSunR + BETA_MIE * odSunM));

        float mu = dot(rd, sunDir);
        // mie is the forward scattering that makes the sun's halo, so it follows
        // the amount of sun the sky has; rayleigh (the atmosphere itself) stays.
        float3 scatter = sunTrans * (BETA_RAYLEIGH * phaseRayleigh(mu) * densR +
                                     BETA_MIE      * phaseMie(mu)      * densM * sunAmount) *
                       SOLAR_IRRADIANCE;

        inScatter += scatter * transmittance * ds;
        transmittance *= exp(-(BETA_RAYLEIGH * densR + BETA_MIE * densM) * ds);
        p += rd * ds;
    }

    inScatter *= RADIANCE_SCALE;

    // tint the sky by the sky color (rt_sky_color): a neutral/white tint
    // keeps the physical blue atmosphere, while colored presets strongly re-tint
    // the sky (the physical blue would otherwise dominate and hide the preset).
    // tintStrength (rt_sky_tint) scales how strongly the tint is applied.
    // How dark the tint is counts as well: chroma alone made every achromatic
    // colour a no-op, so a black sky colour (no chroma at all) kept the bright
    // blue atmosphere instead of turning the sky black.
    float tintLum = dot(skyTint, float3(0.2126, 0.7152, 0.0722));
    float tintSat = length(skyTint - (float3)tintLum);
    float tintChroma = tintSat / (tintLum + 1e-5);
    float tintDarkness = 1.0 - clamp(tintLum, 0.0, 1.0);
    float tintAmount = clamp(tintStrength * max(tintChroma, tintDarkness), 0.0, 1.0);
    float skyLum = dot(inScatter, float3(0.2126, 0.7152, 0.0722));
    float3 col = lerp(inScatter, (float3)skyLum * skyTint, tintAmount);

    // procedural clouds (color is set separately from the sky, e.g. black clouds on a colored sky)
    if (params.cloudParams.w > 0.5)
    {
        float n = cloudMask(rd, params.cloudColor.w, params.cloudParams.z);
        float mask = smoothstep(params.cloudParams.x, params.cloudParams.x + 0.35, n);
        float opacity = mask * params.cloudParams.y;
        col = lerp(col, params.cloudColor.xyz, clamp(opacity, 0.0, 1.0));
    }

    return col;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int3 ipos = int3(dispatchThreadID);

    uint sizeX, sizeY, sizeZ;
    cubemapOut.GetDimensions(sizeX, sizeY, sizeZ);
    int2 size = int2(sizeX, sizeY);

    if (ipos.x >= size.x || ipos.y >= size.y || ipos.z >= 6)
        return;

    int face = ipos.z;
    float3 right   = params.faceBasis[face * 3 + 0].xyz;
    float3 up      = params.faceBasis[face * 3 + 1].xyz;
    float3 forward = params.faceBasis[face * 3 + 2].xyz;

    float2 ndc = (float2(ipos.xy) + (float2)0.5) / (float2)size * 2.0 - 1.0;
    float3 dir = normalize(ndc.x * right + (-ndc.y) * up + forward);

    float3 sunDir       = normalize(params.sunDirection.xyz);
    float3 skyTint      = max(params.skyTint.xyz, (float3)0.0);
    float sunAmount    = params.sunDirection.w;      // 0 when the host has no sun (rt_sun 0)
    float sunAngRad    = max(params.skyParams.w, 0.0); // display sun disc radius
    float multiplier   = params.skyParams.x;
    float tintStrength = params.skyParams.y;
    float sunIntensity = params.skyParams.z;

    float3 base         = evaluateSky(dir, sunDir, skyTint, tintStrength, sunAmount);

    envCubemapOut[ipos] = float4(base * multiplier, 1.0);

    float cosAng = cos(sunAngRad);
    // smoothstep() is evaluated for the whole cubemap before the amount scales it
    float disc = smoothstep(cosAng, 1.0, dot(dir, sunDir)) * sunAmount;
    float3 col = (base + params.sunDiscColor.xyz * sunIntensity * disc) * multiplier;

    cubemapOut[ipos] = float4(col, 1.0);
}
