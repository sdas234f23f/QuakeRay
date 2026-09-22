// Copyright (c) 2021 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// HLSL counterpart of BRDF.h. Like the GLSL one it includes nothing but Random.hlsli and is
// compiled from a translation unit that has already pulled in the layer above it: square, M_PI,
// safePositiveRcp and getLuminance come from Utils.hlsli through ShaderCommonHLSLFunc.hlsli, and
// sampleOrientedHemisphere and getONB from Random.hlsli.
//
// Spellings that had to change:
//   * mix() -> lerp() and inversesqrt() -> rsqrt()
//   * vec3(x) -> (float3)x, and vec3(0.0) -> (float3)0.0: the four places that build a vector out
//     of one scalar, which dxc refuses to widen (see Utils.hlsli)
//   * the two products of sampleSmithGGX keep the order of their operands, as every product of the
//     port does: `transpose(basis) * v` is written `mul(transpose(basis), v)` and `basis * l` is
//     written `mul(basis, l)`. The mirrored spellings `mul(v, transpose(basis))` and `mul(l, basis)`
//     multiply by the transpose instead, which on a square matrix is invisible in the image and is
//     how this file was ported at first. getONB hands back the same matrix as the GLSL one
//     element-wise, so its columns are the three basis vectors as the GLSL has them: the GLSL
//     `basis[0]`, `basis[1]` and `basis[2]` are `getColumn(basis, 0)`, `getColumn(basis, 1)` and
//     `getColumn(basis, 2)`, i.e. b1, b2 and n. That is what keeps `ve` in the normal's space and
//     the returned direction back in world space
//
// Like the GLSL one this file contains no descriptor of its own, so nothing pins it: it is
// compiled from the translation units that use it, and CheckShaderProperties.py covers it through
// the probe pair that pins ShaderCommonGLSLFunc.h / ShaderCommonHLSLFunc.hlsli.
//
// What did not change: every constant, the visible-normal sampling of Heitz line for line
// including its 0.98 and 0.02 guards and the pdf that it accumulates into the caller's
// oneOverPdf, the overload of getFresnelSchlick, and the int literals that are mixed into float
// expressions (`max(nl, 0)`, `2 - alpha`), which HLSL promotes the way GLSL does.

#ifndef BRDF_HLSLI_
#define BRDF_HLSLI_
#include "Random.hlsli"

float roughnessSquaredToSpecPower(in float alpha) {
    return max(0.01, 2.0f / (square(alpha) + 1e-4) - 2.0f);
}


// subsurfaceAlbedo -- 0 if all light is absorbed,
//                     1 if no light is absorbed
float evalBRDFLambertian(float subsurfaceAlbedo)
{
    return subsurfaceAlbedo / M_PI;
}

// u1, u2   -- uniform random numbers
float3 sampleLambertian(const float3 n, float u1, float u2, out float oneOverPdf)
{
    return sampleOrientedHemisphere(n, u1, u2, oneOverPdf);
}



#define BRDF_MIN_SPECULAR_COLOR 0.04

float3 getSpecularColor(const float3 albedo, float metallic)
{
    float3 minSpec = (float3)BRDF_MIN_SPECULAR_COLOR;
    return lerp(minSpec, albedo, metallic);
}

#define AO_ALBEDO_THRESHOLD 0.02

float getMaterialAmbient(const float3 albedo)
{
    float l = getLuminance(albedo);

    return l > AO_ALBEDO_THRESHOLD ? 
        1.0 :
        1.0 - square((l - AO_ALBEDO_THRESHOLD) / AO_ALBEDO_THRESHOLD);
}

float3 demodulateSpecular(const float3 contrib, const float3 surfSpecularColor)
{
    return contrib / max((float3)0.01, surfSpecularColor);
}

// nl -- cos between surface normal and light direction
// specularColor -- reflectance color at zero angle
float3 getFresnelSchlick(float nl, const float3 specularColor)
{
    return specularColor + ((float3)1.0 - specularColor) * pow(1 - max(nl, 0), 5);
}

float getFresnelSchlick(float n1, float n2, const float3 V, const float3 N)
{
    float R0 = (n1 - n2) / (n1 + n2);
    R0 *= R0;

    return lerp(R0, 1.0, pow(1.0 - abs(dot(N, V)), 5.0));
}

// GGX distribution
float D_GGX( float nm, float alpha )
{
#if SHIPPING_HACK
#ifdef FORCE_EVALBRDF_GGX_LOOSE
    // shipping hack: make ggx factor be non-zero, so we can reuse for reprojection,
    // let there be some lag, but at least less noise
    alpha = max( 0.2, alpha );
#endif
#endif

    const float alphaSq = square( alpha );

    nm = max( 0.0, nm );
    return alphaSq / M_PI / square( nm * nm * ( alphaSq - 1 ) + 1 );
}

// Smith G1 for GGX, Karis' approximation ("Real Shading in Unreal Engine 4")
// ns = dot( macrosurface normal, s ), where s is either v or l
float G1_GGX( float ns, float alpha )
{
    return 2 * ns * safePositiveRcp( ns * ( 2 - alpha ) + alpha );
}

#define MIN_GGX_ROUGHNESS 0.005

// n -- macrosurface normal
// v -- direction to viewer
// l -- direction to light
// alpha -- roughness
float3 evalBRDFSmithGGX(const float3 n, const float3 v, const float3 l, float alpha, const float3 specularColor)
{
    alpha = max(alpha, MIN_GGX_ROUGHNESS);

    const float nl = max(dot(n, l), 0);

    if (nl <= 0)
    {
        return (float3)0.0;
    }

    // here, microfacet normal is a half-vector,
    // since we know in which direction l should be reflected
    const float3  h = normalize( v + l );
    const float3 F = getFresnelSchlick(
        clamp(dot(v, h), 0.0, 1.0),
        specularColor
        );
    const float D = D_GGX( dot( n, h ), alpha );

    float G2Modif;
    {
        const float nv = max(dot(n, v), 0);

        // approximation for SmithGGX, Hammon ("PBR Diffuse Lighting for GGX+Smith Microsurfaces")
        // inlcudes 1 / (4 * nl * nv)
        G2Modif = 0.5 / lerp(2 * nl * nv, nl + nv, alpha);
    }

    return F * G2Modif * D;
}



// "Sampling the GGX Distribution of Visible Normals", Heitz
// v        -- direction to viewer, normal's direction is (0,0,1)
// alpha    -- roughness
// u1, u2   -- uniform random numbers
// output   -- normal sampled with PDF D_v(Ne) = G1(v) * max(0, dot(v, Ne)) * D(Ne) / v.z
float3 sampleGGXVNDF(const float3 v, float alpha, float u1, float u2, out float oneOverPdf)
{
    alpha = max( alpha, MIN_GGX_ROUGHNESS );

    // fix: avoid grazing angles
    u1 *= 0.98;
    u2 *= 0.98;

    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));
    
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    const float3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(u1);    
    float phi = 2.0 * M_PI * u2;    
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    const float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    const float3 Ne = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.02, Nh.z)));

    {
        // here, macro normal is (0,0,1), so nm=m.z
        const float nm = Ne.z;
        const float D = D_GGX( nm, alpha );

        // here, macro normal is (0,0,1), so nv=v.z
        const float nv = v.z;
        const float G1 = G1_GGX( nv, alpha );

        oneOverPdf = v.z * safePositiveRcp(G1 * max(0, dot(v, Ne)) * D);
    }

    return Ne;
}

// Sample microfacet normal
// n        -- macrosurface normal, world space
// v        -- direction to viewer, world space
// alpha    -- roughness
// u1, u2   -- uniform random numbers
// Check Heitz's paper for the special representation of rendering equation term 
float3 sampleSmithGGX(const float3 n, const float3 v, float alpha, float u1, float u2, out float oneOverPdf)
{
    const float3x3 basis = getONB(n);

    // get v in normal's space, basis is orthogonal
    const float3 ve = mul(transpose(basis), v);

    // microfacet normal
    const float3 m = sampleGGXVNDF(ve, alpha, u1, u2, oneOverPdf);

    // reflect viewer dir by a microfacet
    const float3 l = reflect( -ve, m );
    // reflection jacobian
    oneOverPdf *= 4 * dot( ve, m );

    // back to world space
    return mul(basis, l);
}

float evalSpecularBouncePdf(const float3 n, const float3 v, float alpha, const float3 l)
{
    const float nv = dot(n, v);
    const float nl = dot(n, l);

    if (nv <= 0.0 || nl <= 0.0)
    {
        return 0.0;
    }

    alpha = max(alpha, MIN_GGX_ROUGHNESS);

    const float3 m = normalize(v + l);
    const float nm = dot(n, m);

    if (nm <= 0.0)
    {
        return 0.0;
    }

    const float D = D_GGX(nm, alpha);
    const float G1 = G1_GGX(nv, alpha);

    return G1 * D / (4.0 * nv);
}

#endif // BRDF_HLSLI_
