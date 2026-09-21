// Copyright (c) 2022 Sultim Tsyrendashiev
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


// HLSL counterpart of the blocks that GenerateShaderCommon.py emits into
// Generated/ShaderCommonGLSL.h. It is written by hand for now, but it has to stay byte
// compatible with the GLSL layouts, so any change here must pass CheckShaderProperties.py.
//
// Rules to keep the port honest, apply them everywhere in the HLSL shader base:
//   * matrices: GLSL `m * v` becomes HLSL `mul(v, m)`, and GLSL `m[i][j]` becomes
//     HLSL `m[j][i]`. For the very same bytes dxc declares matrices as RowMajor where
//     glslang declares them as ColMajor, i.e. the logical matrix is transposed, and the
//     two rules above compensate exactly for that.
//   * struct members: dxc packs nested struct members on 4-byte boundaries, GLSL std430
//     uses the alignment of the member type, so e.g. a float2 after a float lands on
//     offset 4 instead of 8. Where the checker reports such a difference, pin the member
//     with [[vk::offset(N)]].


// Mirrors BINDING_GLOBAL_UNIFORM of the generated GLSL header.
#define BINDING_GLOBAL_UNIFORM (0)

// Mirrors COMPUTE_EFFECT_GROUP_SIZE_X / _Y of the generated GLSL header. HLSL declares the
// workgroup size per entry point, so a compute shader carries
// [numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)].
#define COMPUTE_EFFECT_GROUP_SIZE_X (16)
#define COMPUTE_EFFECT_GROUP_SIZE_Y (16)


#ifdef DESC_SET_GLOBAL_UNIFORM

struct ShGlobalUniform
{
    float4x4 view;
    float4x4 invView;
    float4x4 viewPrev;
    float4x4 projection;
    float4x4 invProjection;
    float4x4 projectionPrev;
    float4x4 volumeViewProj;
    float4x4 volumeViewProjInv;
    float4x4 volumeViewProj_Prev;
    float4x4 volumeViewProjInv_Prev;
    float cellWorldSize;
    float renderWidth;
    uint __pad0;
    uint __pad1;
    float renderHeight;
    uint frameId;
    float timeDelta;
    float minLogLuminance;
    float maxLogLuminance;
    float luminanceWhitePoint;
    uint stopEyeAdaptation;
    uint directionalLightExists;
    float polyLightSpotlightFactor;
    uint skyType;
    float skyColorMultiplier;
    uint skyCubemapIndex;
    float4 skyColorDefault;
    float4 cameraPosition;
    float4 cameraPositionPrev;
    uint debugShowFlags;
    uint indirSecondBounce;
    uint lightCount;
    uint lightCountPrev;
    float emissionMapBoost;
    float emissionMaxScreenColor;
    float normalMapStrength;
    float skyColorSaturation;
    float emissionSharpMask;
    float talSelfLitOffset;
    uint emissionBlendMode;
    float emissionBlendStrength;
    float skyAmbientLod;
    float rayLength;
    uint rayCullBackFaces;
    uint rayCullMaskWorld;
    float bloomIntensity;
    float bloomThreshold;
    float bloomEmissionMultiplier;
    uint reflectRefractMaxDepth;
    uint cameraMediaType;
    float indexOfRefractionWater;
    float indexOfRefractionGlass;
    float waterTextureDerivativesMultiplier;
    uint volumeEnableType;
    float volumeScattering;
    uint forceNoWaterRefraction;
    uint waterNormalTextureIndex;
    uint noBackfaceReflForNoMediaChange;
    float time;
    float waterWaveSpeed;
    float waterWaveStrength;
    float4 waterColorAndDensity;
    float4 acidColorAndDensity;
    float cameraRayConeSpreadAngle;
    float waterTextureAreaScale;
    uint squareInputRoughness;
    float upscaledRenderWidth;
    float4 worldUpVector;
    float upscaledRenderHeight;
    float jitterX;
    float jitterY;
    float primaryRayMinDist;
    uint rayCullMaskWorld_Shadow;
    uint lensFlareCullingInputCount;
    uint applyViewProjToLensFlares;
    uint twirlPortalNormal;
    uint lightIndexIgnoreFPVShadows;
    float gradientMultDiffuse;
    float gradientMultIndirect;
    float gradientMultSpecular;
    float minRoughness;
    float volumeCameraNear;
    float volumeCameraFar;
    uint antiFireflyEnabled;
    float4 volumeAmbient;
    float4 volumeSourceColor;
    float4 volumeDirToSource;
    float volumeSourceAsymmetry;
    uint coreQ2RTX;
    uint q2DepthGradMode;
    float skyNee;
    uint q2LightStatsMode;
    uint reflRefrEarlyOut;
    uint neeLightSamples;
    float turbWarpStrength;
    int4 instanceGeomInfoOffset[12];
    int4 instanceGeomInfoOffsetPrev[12];
    int4 instanceGeomCount[12];
    float4x4 viewProjCubemap[6];
    float4x4 skyCubemapRotationTransform;
    float4 fogMins[8];
    uint fogIsActive[8];
    float4 fogMaxs[8];
    float4 fogColor[8];
    float4 fogDensity[8];
    float4 lightStyleScales[16];
    float4 giBounceRays;
    float4 fltEnable;
    float4 fixedAlbedo;
    float4 sunBounce;
    float4 levelFogColorDensity;
    float4 levelFogSkyBlend;
};

[[vk::binding(BINDING_GLOBAL_UNIFORM, DESC_SET_GLOBAL_UNIFORM)]] ConstantBuffer<ShGlobalUniform> globalUniform;

#endif // DESC_SET_GLOBAL_UNIFORM



#ifdef DESC_SET_FRAMEBUFFERS

// Bindings of the generated GLSL header. The format has to be spelled out, otherwise dxc
// declares the image as Unknown and the descriptor does not match the one the host creates.
// dxc only accepts a subset of the SPIR-V names, and for this one that spelling is
// "r11g11b10f" (which is emitted as R11fG11fB10f).
[[vk::binding(29, DESC_SET_FRAMEBUFFERS), vk::image_format("r11g11b10f")]] RWTexture2D<float4> framebufUpscaledPing;
[[vk::binding(30, DESC_SET_FRAMEBUFFERS), vk::image_format("r11g11b10f")]] RWTexture2D<float4> framebufUpscaledPong;

#endif // DESC_SET_FRAMEBUFFERS
