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

#pragma once

// HLSL counterpart of Structs.h.


struct ShTriangle
{
    float3x3 positions;
    float3x3 prevPositions;
    float3x3 normals;
    // GLSL mat3x2: three columns of two floats, so it is declared with two rows of three
    float2x3 layerTexCoord[3];
    float4   materialColors[3];
    uint3    materials[3];
    uint     vertexColors[3];
    uint     geometryInstanceFlags;
    float4   tangent;
    float     geomRoughness;
    float     geomEmission;
    float     geomMetallicity;
    uint     portalIndex;
    uint     cluster;
    uint     lightStyleIndices;
};

struct ShPayload
{
    float2 baryCoords;
    uint   instIdAndIndex;
    uint   geomAndPrimIndex;
};

struct ShPayloadShadow
{
    uint isShadowed;
};

struct ShHitInfo
{
    float3 albedo;
    float  metallic;
    float3 normal;
    float  roughness;
    float3 normalGeom;
    float  emission;
    float3 hitPosition;
    uint   instCustomIndex;
    uint   geometryInstanceFlags;
    uint   portalIndex;
    uint   cluster;
};
