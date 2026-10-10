// Copyright (c) 2026 f1ames0ff <f1am3sdev.github@protonmail.com>
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

#ifndef qray_H_
#define qray_H_

#include <stdint.h>

#if !defined(QR_STATIC)
    #ifdef QR_LIBRARY_EXPORTS
        #define QRAPI __declspec(dllexport)
    #else
        #define QRAPI __declspec(dllimport)
    #endif
    #define QRCONV __cdecl
#else
    #define QRAPI
    #define QRCONV
#endif

#define QR_API_VERSION "1.03.0000"

#ifdef QR_USE_SURFACE_WIN32
    #include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(QR_DEFINE_NON_DISPATCHABLE_HANDLE)
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define QR_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T *object;
    #else
        #define QR_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
    #endif
#endif

typedef uint32_t QrBool32;
QR_DEFINE_NON_DISPATCHABLE_HANDLE(QrInstance)
typedef uint32_t QrMaterial;
typedef uint32_t QrCubemap;
typedef uint32_t QrFlags;

#define QR_NULL_HANDLE      0
#define QR_NO_MATERIAL      0
#define QR_EMPTY_CUBEMAP    0
#define QR_FALSE            0
#define QR_TRUE             1

typedef enum QrResult
{
    QR_SUCCESS,
    QR_GRAPHICS_API_ERROR,
    QR_CANT_FIND_PHYSICAL_DEVICE,
    QR_WRONG_ARGUMENT,
    QR_TOO_MANY_INSTANCES,
    QR_WRONG_INSTANCE,
    QR_FRAME_WASNT_STARTED,
    QR_FRAME_WASNT_ENDED,
    QR_CANT_UPDATE_TRANSFORM,
    QR_CANT_UPDATE_TEXCOORDS,
    QR_CANT_UPDATE_MATERIAL,
    QR_CANT_UPDATE_ANIMATED_MATERIAL,
    QR_CANT_UPLOAD_RASTERIZED_GEOMETRY,
    QR_WRONG_MATERIAL_PARAMETER,
    QR_WRONG_FUNCTION_CALL,
    QR_ERROR_CANT_FIND_BLUE_NOISE,
    QR_ERROR_CANT_FIND_WATER_TEXTURES,
} QrResult;

typedef void (*PFN_qrPrint)(const char *pMessage, void *pUserData);
typedef void (*PFN_qrOpenFile)(const char *pFilePath, void *pUserData, const void **ppOutData, uint32_t *pOutDataSize, void **ppOutFileUserHandle);
typedef void (*PFN_qrCloseFile)(void *pFileUserHandle, void *pUserData);

typedef struct QrWin32SurfaceCreateInfo QrWin32SurfaceCreateInfo;

#ifdef QR_USE_SURFACE_WIN32
typedef struct QrWin32SurfaceCreateInfo
{
    HINSTANCE           hinstance;
    HWND                hwnd;
} QrWin32SurfaceCreateInfo;
#endif

typedef enum QrTextureSwizzling
{
    QR_TEXTURE_SWIZZLING_ROUGHNESS_METALLIC_EMISSIVE,
    QR_TEXTURE_SWIZZLING_ROUGHNESS_METALLIC,
    QR_TEXTURE_SWIZZLING_METALLIC_ROUGHNESS_EMISSIVE,
    QR_TEXTURE_SWIZZLING_METALLIC_ROUGHNESS,
    QR_TEXTURE_SWIZZLING_NULL_ROUGHNESS_METALLIC,
} QrTextureSwizzling;

typedef struct QrInstanceCreateInfo
{
    const char                  *pAppName;

    const char                  *pAppGUID;

    QrWin32SurfaceCreateInfo    *pWin32SurfaceInfo;

    const char                  *pConfigPath;

    PFN_qrPrint                 pfnPrint;

    void                        *pUserPrintData;

    const char                  *pShaderFolderPath;

    const char                  *pBlueNoiseFilePath;

    PFN_qrOpenFile              pfnOpenFile;
    PFN_qrCloseFile             pfnCloseFile;

    void                        *pUserLoadFileData;

    uint32_t                    primaryRaysMaxAlbedoLayers;
    uint32_t                    indirectIlluminationMaxAlbedoLayers;

    QrBool32                    rayCullBackFacingTriangles;

    QrBool32                    allowGeometryWithSkyFlag;

    uint32_t                    rasterizedMaxVertexCount;
    uint32_t                    rasterizedMaxIndexCount;

    QrBool32                    rasterizedVertexColorGamma;

    uint32_t                    rasterizedSkyCubemapSize;

    uint32_t                    maxTextureCount;

    QrBool32                    textureSamplerForceMinificationFilterLinear;
    QrBool32                    textureSamplerForceNormalMapFilterLinear;

    const char                  *pOverridenTexturesFolderPath;

    const char                  *pOverridenTexturesFolderPathDeveloper;

    const char                  *pOverridenAlbedoAlphaTexturePostfix;

    const char                  *pOverridenRoughnessMetallicEmissionTexturePostfix;

    const char                  *pOverridenNormalTexturePostfix;

    QrBool32                    originalAlbedoAlphaTextureIsSRGB;
    QrBool32                    originalRoughnessMetallicEmissionTextureIsSRGB;
    QrBool32                    originalNormalTextureIsSRGB;

    QrBool32                    overridenAlbedoAlphaTextureIsSRGB;
    QrBool32                    overridenRoughnessMetallicEmissionTextureIsSRGB;
    QrBool32                    overridenNormalTextureIsSRGB;

    const char                  *pWaterNormalTexturePath;

    QrTextureSwizzling          pbrTextureSwizzling;

    QrBool32                    effectWipeIsUsed;

    uint32_t                    godRaysQuality;
} QrInstanceCreateInfo;

QRAPI QrResult QRCONV qrCreateInstance(
    const QrInstanceCreateInfo          *pInfo,
    QrInstance                          *pResult);

QRAPI QrResult QRCONV qrDestroyInstance(
    QrInstance                          qrInstance);

typedef struct QrLayeredMaterial
{
    QrMaterial  layerMaterials[3];
} QrLayeredMaterial;

typedef enum QrGeometryType
{
    QR_GEOMETRY_TYPE_STATIC,
    QR_GEOMETRY_TYPE_STATIC_MOVABLE,
    QR_GEOMETRY_TYPE_DYNAMIC
} QrGeometryType;

typedef enum QrGeometryPassThroughType
{
    QR_GEOMETRY_PASS_THROUGH_TYPE_OPAQUE,
    QR_GEOMETRY_PASS_THROUGH_TYPE_ALPHA_TESTED,
    QR_GEOMETRY_PASS_THROUGH_TYPE_MIRROR,
    QR_GEOMETRY_PASS_THROUGH_TYPE_PORTAL,
    QR_GEOMETRY_PASS_THROUGH_TYPE_WATER_ONLY_REFLECT,
    QR_GEOMETRY_PASS_THROUGH_TYPE_WATER_REFLECT_REFRACT,
    QR_GEOMETRY_PASS_THROUGH_TYPE_GLASS_REFLECT_REFRACT,
    QR_GEOMETRY_PASS_THROUGH_TYPE_ACID_REFLECT_REFRACT,
} QrGeometryPassThroughType;

typedef enum QrGeometryPrimaryVisibilityType
{
    QR_GEOMETRY_VISIBILITY_TYPE_WORLD_0,
    QR_GEOMETRY_VISIBILITY_TYPE_WORLD_1,
    QR_GEOMETRY_VISIBILITY_TYPE_WORLD_2,
    QR_GEOMETRY_VISIBILITY_TYPE_FIRST_PERSON,
    QR_GEOMETRY_VISIBILITY_TYPE_FIRST_PERSON_VIEWER,

    QR_GEOMETRY_VISIBILITY_TYPE_SKY,
} QrGeometryPrimaryVisibilityType;

typedef enum QrGeometryMaterialBlendType
{
    QR_GEOMETRY_MATERIAL_BLEND_TYPE_OPAQUE,
    QR_GEOMETRY_MATERIAL_BLEND_TYPE_ALPHA,
    QR_GEOMETRY_MATERIAL_BLEND_TYPE_ADD,
    QR_GEOMETRY_MATERIAL_BLEND_TYPE_SHADE
} QrGeometryMaterialBlendType;

typedef struct QrTransform
{
    float       matrix[3][4];
} QrTransform;

typedef struct QrMatrix3D
{
    float       matrix[3][3];
} QrMatrix3D;

typedef struct QrFloat2D
{
    float       data[2];
} QrFloat2D;

typedef struct QrFloat3D
{
    float       data[3];
} QrFloat3D;

typedef struct QrFloat4D
{
    float       data[4];
} QrFloat4D;

typedef struct QrVertex
{
    float       position[3];        uint32_t _padding0;
    float       normal[3];          uint32_t _padding1;
    float       texCoord[2];
    float       texCoordLayer1[2];
    float       texCoordLayer2[2];

    uint32_t    packedColor;
    uint32_t    cluster;
    uint32_t    lightStyles;
    uint32_t    _padding2[3];
} QrVertex;

typedef enum QrGeometryUploadFlagBits
{
    QR_GEOMETRY_UPLOAD_GENERATE_NORMALS_BIT = 1,
    QR_GEOMETRY_UPLOAD_EXACT_NORMALS_BIT = 2,
    QR_GEOMETRY_UPLOAD_GENERATE_INVERTED_NORMALS_BIT = 4,

    QR_GEOMETRY_UPLOAD_NO_MEDIA_CHANGE_ON_REFRACT_BIT = 8,

    QR_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_MULTIPLY_BIT = 16,
    QR_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_ADD_BIT = 32,

    QR_GEOMETRY_UPLOAD_IGNORE_REFRACT_AFTER_REFRACT_BIT = 64,

    QR_GEOMETRY_UPLOAD_TURB_WARP_BIT = 128,
    QR_GEOMETRY_UPLOAD_ALPHA_TRANSMISSION_BIT = 256,

    /* The alpha is a cutout even where the surface traces as glass: the glass
       any-hit runs the alpha test for it (a lattice window keeps its holes). */
    QR_GEOMETRY_UPLOAD_GLASS_CUTOUT_BIT = 512,
} QrGeometryUploadFlagBits;
typedef QrFlags QrGeometryUploadFlags;

typedef struct QrGeometryUploadInfo
{
    uint64_t                        uniqueID;
    QrGeometryUploadFlags           flags;

    QrGeometryType                  geomType;
    QrGeometryPassThroughType       passThroughType;
    QrGeometryPrimaryVisibilityType visibilityType;

    uint32_t                        vertexCount;
    const QrVertex                  *pVertices;

    uint32_t                        indexCount;
    const uint32_t                  *pIndices;

    uint8_t                         *pPortalIndex;

    QrFloat4D                       layerColors[3];
    QrGeometryMaterialBlendType     layerBlendingTypes[3];

    float                           defaultRoughness;
    float                           defaultMetallicity;

    float                           defaultEmission;

    QrLayeredMaterial               geomMaterial;
    QrTransform                     transform;
} QrGeometryUploadInfo;

typedef struct QrUpdateTransformInfo
{
    uint64_t        movableStaticUniqueID;
    QrTransform     transform;
} QrUpdateTransformInfo;

typedef struct QrUpdateTexCoordsInfo
{
    uint64_t        staticUniqueID;
    uint32_t        vertexOffset;
    uint32_t        vertexCount;

    const void      *pTexCoordLayerData[3];
} QrUpdateTexCoordsInfo;

QRAPI QrResult QRCONV qrUploadGeometry(
    QrInstance                              qrInstance,
    const QrGeometryUploadInfo              *pUploadInfo);

QRAPI QrResult QRCONV qrUpdateGeometryTransform(
    QrInstance                              qrInstance,
    const QrUpdateTransformInfo             *pUpdateInfo);

QRAPI QrResult QRCONV qrUpdateGeometryTexCoords(
    QrInstance                              qrInstance,
    const QrUpdateTexCoordsInfo             *pUpdateInfo);

QRAPI QrResult QRCONV qrBeginStaticGeometries(
    QrInstance                          qrInstance);

QRAPI QrResult QRCONV qrSubmitStaticGeometries(
    QrInstance                          qrInstance);

typedef enum QrBlendFactor
{
    QR_BLEND_FACTOR_ONE,
    QR_BLEND_FACTOR_ZERO,
    QR_BLEND_FACTOR_SRC_COLOR,
    QR_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    QR_BLEND_FACTOR_DST_COLOR,
    QR_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    QR_BLEND_FACTOR_SRC_ALPHA,
    QR_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
} QrBlendFactor;

typedef enum QrRasterizedGeometryRenderType
{
    QR_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
    QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN,
    QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY
} QrRasterizedGeometryRenderType;

// Rectangle in pixels. (x, y) defines the top-left corner.
typedef struct QrRect2D
{
    int32_t     x;
    int32_t     y;
    uint32_t    width;
    uint32_t    height;
} QrRect2D;

typedef enum QrRasterizedGeometryStateFlagBits
{
    QR_RASTERIZED_GEOMETRY_STATE_ALPHA_TEST         = 1,
    QR_RASTERIZED_GEOMETRY_STATE_BLEND_ENABLE       = 2,
    QR_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST         = 4,
    QR_RASTERIZED_GEOMETRY_STATE_DEPTH_WRITE        = 8,
    QR_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST    = 16,
    QR_RASTERIZED_GEOMETRY_STATE_SMOKE              = 32,
    QR_RASTERIZED_GEOMETRY_STATE_PARTICLE           = 64,
    // Marks a particle sprite regardless of what shades it: `QR_RASTERIZED_GEOMETRY_STATE_PARTICLE`
    // is the lit pipeline selector and only set while `r_particle_lighting` is on, so the traced
    // stand-ins of the classic sprites have to key on a flag the particle uploads always carry.
    QR_RASTERIZED_GEOMETRY_STATE_PARTICLE_SPRITE    = 128,
} QrRasterizedGeometryStateFlagBits;
typedef uint32_t QrRasterizedGeometryStateFlags;

typedef struct QrRasterizedGeometryUploadInfo
{
    QrRasterizedGeometryRenderType          renderType;

    uint32_t                                vertexCount;
    const QrVertex                          *pVertices;

    uint32_t                                indexCount;
    const void                              *pIndices;

    QrTransform                             transform;

    QrFloat4D                               color;

    QrMaterial                              material;
    QrRasterizedGeometryStateFlags          pipelineState;
    QrBlendFactor                           blendFuncSrc;
    QrBlendFactor                           blendFuncDst;

    QrFloat4D                               smokeNoise;
    QrFloat4D                               smokeLook;

    // Scissor rectangle in pixels, top-left origin. width == 0 means no
    // scissor (the whole viewport is used). For QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY
    // it must be zero as well.
    QrRect2D                                scissor;
} QrRasterizedGeometryUploadInfo;

typedef struct QrExtent2D
{
    uint32_t    width;
    uint32_t    height;
} QrExtent2D;

typedef struct QrExtent3D
{
    uint32_t    width;
    uint32_t    height;
    uint32_t    depth;
} QrExtent3D;

typedef struct QrViewport
{
    float       x;
    float       y;
    float       width;
    float       height;
    float       minDepth;
    float       maxDepth;
} QrViewport;

QRAPI QrResult QRCONV qrUploadRasterizedGeometry(
    QrInstance                              qrInstance,
    const QrRasterizedGeometryUploadInfo    *pUploadInfo,
    const float                             *pViewProjection,
    const QrViewport                        *pViewport);

typedef struct QrParticlePoint
{
    float       position[3];
    uint32_t    packedColor;
    float       size;
    uint32_t    cluster;
} QrParticlePoint;

typedef struct QrParticleUploadInfo
{
    const QrParticlePoint   *pPoints;
    uint32_t                count;

    QrMaterial              material;
    uint32_t                pipelineState;
    QrFloat4D               smokeLook;

    QrFloat3D               viewRight;
    QrFloat3D               viewUp;
} QrParticleUploadInfo;

QRAPI QrResult QRCONV qrUploadParticles(
    QrInstance                              qrInstance,
    const QrParticleUploadInfo              *pUploadInfo);

typedef struct QrDecalUploadInfo
{
    QrTransform     transform;
    QrMaterial      material;
} QrDecalUploadInfo;

QRAPI QrResult QRCONV qrUploadDecal(
    QrInstance                              qrInstance,
    const QrDecalUploadInfo                 *pUploadInfo);

typedef struct QrPortalUploadInfo
{
    uint8_t         portalIndex;
    QrFloat3D       inPosition;
    QrFloat3D       outPosition;
    QrFloat3D       outDirection;
    QrFloat3D       outUp;
} QrPortalUploadInfo;

QRAPI QrResult QRCONV qrUploadPortal(
    QrInstance                              qrInstance,
    const QrPortalUploadInfo                *pUploadInfo);

typedef struct QrDirectionalLightUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       direction;
    float           angularDiameterDegrees;
} QrDirectionalLightUploadInfo;

typedef struct QrSphericalLightUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       position;

    float           radius;
    QrFloat3D       normal;
} QrSphericalLightUploadInfo;

typedef struct QrPolygonalLightUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       positions[3];
} QrPolygonalLightUploadInfo;

#define MAX_TEXTURED_AREA_LIGHT_VERTS 8
typedef struct QrTexturedAreaLightUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       A;
    QrFloat3D       B;
    QrFloat3D       C;
    QrFloat3D       normal;
    float           area;
    int             numVerts;
    QrFloat2D       uvVerts[MAX_TEXTURED_AREA_LIGHT_VERTS];

    QrMaterial      material;
    float           meanEmiss;
    float           angleInner;
    float           angleOuter;
    float           projector;

    int             fit;
    int             isStatic;
} QrTexturedAreaLightUploadInfo;

/* Spot lights are regular lights in the light array: any number of them can be uploaded. */
typedef struct QrSpotLightUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       position;
    QrFloat3D       direction;
    // Light source disk radius.
    float           radius;

    // Outer cone half-angle. In radians.
    float           angleOuter;

    // Inner cone half-angle. In radians; the intensity is full below it.
    float           angleInner;
} QrSpotLightUploadInfo;

QRAPI QrResult QRCONV qrUploadDirectionalLight(
    QrInstance                          qrInstance,
    const QrDirectionalLightUploadInfo  *pUploadInfo);

QRAPI QrResult QRCONV qrUploadSphericalLight(
    QrInstance                          qrInstance,
    const QrSphericalLightUploadInfo    *pUploadInfo);

QRAPI QrResult QRCONV qrUploadSpotLight(
    QrInstance                          qrInstance,
    const QrSpotLightUploadInfo         *pUploadInfo);

QRAPI QrResult QRCONV qrBeginDeferredLightUploads(
    QrInstance                          qrInstance,
    uint32_t                            slot);

QRAPI QrResult QRCONV qrEndDeferredLightUploads(
    QrInstance                          qrInstance);

QRAPI QrResult QRCONV qrFlushDeferredLightUploads(
    QrInstance                          qrInstance);

QRAPI QrResult QRCONV qrSetParticleProxyGate(
    QrInstance                          qrInstance,
    uint32_t                            gate,
    uint32_t                            glassParticles);

QRAPI QrResult QRCONV qrUploadPolygonalLight(
    QrInstance                          qrInstance,
    const QrPolygonalLightUploadInfo    *pUploadInfo);

QRAPI QrResult QRCONV qrUploadTexturedAreaLight(
    QrInstance                          qrInstance,
    const QrTexturedAreaLightUploadInfo *pUploadInfo);

QRAPI QrResult QRCONV qrUploadTexturedAreaLights(
    QrInstance                          qrInstance,
    const QrTexturedAreaLightUploadInfo *pUploadInfos,
    uint32_t                            count);

#define QR_DTAL_MAX_UPLOAD_MEMBERS 131072

typedef struct QrDtalMemberUpload
{
    QrFloat3D       A;
    float           area;

    QrFloat3D       B;
    float           numVerts;

    QrFloat3D       C;
    float           prob;

    QrFloat3D       normal;
    float           marginalProb;

    QrFloat2D       uv[MAX_TEXTURED_AREA_LIGHT_VERTS];

    uint32_t        aliasIndex;
    uint32_t        reserved[3];
} QrDtalMemberUpload;

typedef struct QrDtalGroupUploadInfo
{
    uint64_t        uniqueID;
    QrFloat3D       color;
    QrFloat3D       center;
    QrFloat3D       normal;
    QrFloat3D       boundsMin;
    QrFloat3D       boundsMax;

    QrMaterial      material;

    float           area;
    float           meanEmiss;
    float           angleInner;
    float           angleOuter;
    float           projector;
    float           reach;
    float           estimatedPower;
    float           boundsRadius;

    uint32_t        memberBase;
    uint32_t        memberCount;
} QrDtalGroupUploadInfo;

typedef struct QrDtalGroupUploadBatch
{
    uint32_t                     groupCount;
    const QrDtalGroupUploadInfo *pGroups;
    uint32_t                     memberCount;
    const QrDtalMemberUpload    *pMembers;
} QrDtalGroupUploadBatch;

QRAPI QrResult QRCONV qrUploadDtalGroups(
    QrInstance                          qrInstance,
    const QrDtalGroupUploadBatch        *pUploadInfo);

#define QR_CLUSTER_LIGHT_NO_CLUSTER    (~0u)

#define QR_CLUSTER_MAX_REGISTERED_LIGHTS 4095

#define QR_CLUSTER_LIGHT_MAX_SOURCES_CLUSTERS 16

typedef struct QrClusterLightSource
{
    uint64_t  uniqueID;
    QrFloat3D origin;

    uint32_t  cluster;

    float     reach;

    float           radius;
    uint32_t        clusterCount;
    const uint32_t *pClusters;

    float           power;
} QrClusterLightSource;

typedef struct QrClusterLightSourcesUploadInfo
{
    uint32_t                    numLights;
    const QrClusterLightSource *pLights;

    float                       topUpReach;

    int32_t                     allowIncremental;

    int32_t                     allowOverflow;

    int32_t                     validate;
} QrClusterLightSourcesUploadInfo;

QRAPI QrResult QRCONV qrUploadClusterLightSources(
    QrInstance                              qrInstance,
    const QrClusterLightSourcesUploadInfo   *pUploadInfo);

QRAPI QrResult QRCONV qrBeginClusterLightSources(
    QrInstance                              qrInstance,
    const QrClusterLightSourcesUploadInfo   *pUploadInfo,
    uint32_t                                 sliceCount);

QRAPI QrResult QRCONV qrRunClusterLightSourceSlice(
    QrInstance                              qrInstance,
    uint32_t                                 slice,
    uint32_t                                 sliceCount);

QRAPI QrResult QRCONV qrFinishClusterLightSources(
    QrInstance                              qrInstance);

QRAPI QrResult QRCONV qrRunClusterListPublishSlice(
    QrInstance                              qrInstance,
    uint32_t                                 slice,
    uint32_t                                 sliceCount);

QRAPI QrResult QRCONV qrCommitClusterListPublication(
    QrInstance                              qrInstance);

#define QR_CLUSTER_PUB_STALE_VALID       (1u << 0)
#define QR_CLUSTER_PUB_STALE_GENERATION  (1u << 1)
#define QR_CLUSTER_PUB_STALE_CLUSTERS    (1u << 2)
#define QR_CLUSTER_PUB_STALE_WORDS       (1u << 3)
#define QR_CLUSTER_PUB_STALE_TAILS       (1u << 4)
#define QR_CLUSTER_PUB_STALE_PLACES      (1u << 5)
#define QR_CLUSTER_PUB_STALE_ORDER       (1u << 6)
#define QR_CLUSTER_PUB_SKIP_DEVICE       (1u << 7)
#define QR_CLUSTER_PUB_SKIP_SLOT         (1u << 8)
#define QR_CLUSTER_PUB_COPY              (1u << 9)

typedef struct QrClusterLightStats
{
    uint32_t clusters;
    uint32_t sources;

    uint32_t unresolved;

    uint32_t listEntries;
    uint32_t grants;
    uint32_t denied;

    uint32_t topUpGrants;

    uint32_t reachGated;

    uint32_t walkedSources;
    uint32_t cachedSources;

    uint32_t addedSources;
    uint32_t removedSources;
    uint32_t movedSources;

    uint32_t composedFrames;
    uint32_t reusedFrames;

    uint32_t fullClusters;

    uint32_t incrementalDirty;
    uint32_t moveFootprint;

    uint32_t tailEntries;
    uint32_t clustersWithTail;
    uint32_t tailBudgetExceeded;
    uint32_t candidateMax;
    uint32_t candidateMedian;
    uint32_t candidateP95;

    float    visMs;
    float    topUpMs;
    float    markMs;
    float    gridMs;
    float    fillMs;
    float    tailMs;
    float    publishMs;
    float    totalMs;

    uint32_t publicationMask;
} QrClusterLightStats;

QRAPI QrResult QRCONV qrGetClusterLightStats(
    QrInstance              qrInstance,
    QrClusterLightStats    *pStats);

QRAPI QrResult QRCONV qrGetClusterLightGrants(
    QrInstance  qrInstance,
    uint32_t   *pGranted,
    uint32_t   *pDenied,
    uint32_t    maxCount,
    uint32_t   *pCount);

QRAPI QrResult QRCONV qrGetClusterLightList(
    QrInstance  qrInstance,
    uint32_t    cluster,
    uint64_t   *pLightUniqueIds,
    uint32_t    maxCount,
    uint32_t   *pCount);

QRAPI QrResult QRCONV qrGetClusterLightTail(
    QrInstance  qrInstance,
    uint32_t    cluster,
    uint64_t   *pLightUniqueIds,
    float      *pProb,
    float      *pMarginal,
    uint32_t   *pAlias,
    float      *pBeta,
    uint32_t    maxCount,
    uint32_t   *pCount);

typedef enum QrWorldLightFaceFlags
{
    QR_WORLD_LIGHT_FACE_MASKED_BIT       = 1 << 0,

    QR_WORLD_LIGHT_FACE_INLINE_MODEL_BIT = 1 << 1,
} QrWorldLightFaceFlags;

typedef struct QrWorldLightFace
{
    uint64_t  uniqueID;
    uint32_t  firstVertex;
    uint32_t  numVertices;

    uint32_t  cluster;

    uint32_t  flags;

    QrMaterial material;
    QrFloat3D  color;
    float      meanEmiss;

    uint32_t   isStatic;
} QrWorldLightFace;

typedef enum QrWorldLightsUploadFlags
{
    QR_WORLD_LIGHTS_UPLOAD_PRINT_STATS_BIT = 1 << 0,
} QrWorldLightsUploadFlags;

typedef enum QrWorldClusterFlags
{
    QR_WORLD_CLUSTER_SOLID_BIT = 1 << 0,
} QrWorldClusterFlags;

typedef struct QrWorldLightsUploadInfo
{
    uint32_t flags;

    uint32_t         numClusters;
    const QrFloat3D *pClusterMins;
    const QrFloat3D *pClusterMaxs;

    const uint8_t   *pClusterFlags;

    uint32_t                numFaces;
    const QrWorldLightFace *pFaces;
    uint32_t                numFaceVertices;
    const QrVertex         *pFaceVertices;

    const uint8_t *pVisData;
    uint32_t       visDataSize;
    uint32_t       pvsRowBytes;
    const int32_t *pVisOffsets;

    const uint8_t *pClusterSkyVisibility;
} QrWorldLightsUploadInfo;

QRAPI QrResult QRCONV qrUploadWorldLights(
    QrInstance                    qrInstance,
    const QrWorldLightsUploadInfo *pUploadInfo);

typedef enum QrSamplerAddressMode
{
    QR_SAMPLER_ADDRESS_MODE_REPEAT,
    QR_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    QR_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    QR_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    QR_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
} QrSamplerAddressMode;

typedef enum QrSamplerFilter
{
    QR_SAMPLER_FILTER_LINEAR,
    QR_SAMPLER_FILTER_NEAREST,
} QrSamplerFilter;

typedef struct QrTextureSet
{
    const void *pDataAlbedoAlpha;
    const void *pDataRoughnessMetallicEmission;
    const void *pDataNormal;
} QrTextureSet;

typedef enum QrMaterialCreateFlagBits
{
    QR_MATERIAL_CREATE_DONT_GENERATE_MIPMAPS_BIT = 1,

    QR_MATERIAL_CREATE_FORCE_LOWEST_MIP_BIT = 2,

    QR_MATERIAL_CREATE_DYNAMIC_SAMPLER_FILTER_BIT = 4,

    QR_MATERIAL_CREATE_UPDATEABLE_BIT = 8,
} QrMaterialCreateFlagBits;
typedef QrFlags QrMaterialCreateFlags;

typedef struct QrMaterialCreateInfo
{
    QrMaterialCreateFlags   flags;

    QrExtent2D              size;

    QrTextureSet            textures;

    const char              *pRelativePath;
    QrSamplerFilter         filter;
    QrSamplerAddressMode    addressModeU;
    QrSamplerAddressMode    addressModeV;
} QrMaterialCreateInfo;

typedef struct QrMaterialUpdateInfo
{
    QrMaterial              target;
    QrTextureSet            textures;
} QrMaterialUpdateInfo;

typedef struct QrAnimatedMaterialCreateInfo
{
    uint32_t                            frameCount;
    QrMaterialCreateInfo                *pFrames;
} QrAnimatedMaterialCreateInfo;

QRAPI QrResult QRCONV qrCreateMaterial(
    QrInstance                          qrInstance,
    const QrMaterialCreateInfo          *pCreateInfo,
    QrMaterial                          *pResult);

QRAPI QrResult QRCONV qrCreateAnimatedMaterial(
    QrInstance                          qrInstance,
    const QrAnimatedMaterialCreateInfo  *pCreateInfo,
    QrMaterial                          *pResult);

QRAPI QrResult QRCONV qrChangeAnimatedMaterialFrame(
    QrInstance                          qrInstance,
    QrMaterial                          animatedMaterial,
    uint32_t                            frameIndex);

QRAPI QrResult QRCONV qrUpdateMaterialContents(
    QrInstance                          qrInstance,
    const QrMaterialUpdateInfo          *pUpdateInfo);

QRAPI QrResult QRCONV qrCanUpdateMaterialContents(
    QrInstance                          qrInstance,
    QrMaterial                          material,
    QrExtent2D                          size);

QRAPI QrResult QRCONV qrDestroyMaterial(
    QrInstance                          qrInstance,
    QrMaterial                          material);

typedef struct QrCubemapFaceData
{
    const void *pPositiveX;
    const void *pNegativeX;
    const void *pPositiveY;
    const void *pNegativeY;
    const void *pPositiveZ;
    const void *pNegativeZ;
} QrCubemapFaceData;

typedef struct QrCubemapFacePaths
{
    const char *pPositiveX;
    const char *pNegativeX;
    const char *pPositiveY;
    const char *pNegativeY;
    const char *pPositiveZ;
    const char *pNegativeZ;
} QrCubemapFacePaths;

typedef struct QrCubemapCreateInfo
{
    union
    {
        const void          *pData[6];
        QrCubemapFaceData   dataFaces;
    };

    union
    {
        const char          *pRelativePaths[6];
        QrCubemapFacePaths  relativePathFaces;
    };

    uint32_t                sideSize;
    QrBool32                useMipmaps;
    QrSamplerFilter         filter;
} QrCubemapCreateInfo;

QRAPI QrResult QRCONV qrCreateCubemap(
    QrInstance                          qrInstance,
    const QrCubemapCreateInfo           *pCreateInfo,
    QrCubemap                           *pResult);

QRAPI QrResult QRCONV qrDestroyCubemap(
    QrInstance                          qrInstance,
    QrCubemap                           cubemap);

typedef enum QrPresentMode
{
    QR_PRESENT_MODE_MAILBOX = 0,
    QR_PRESENT_MODE_VSYNC,
    QR_PRESENT_MODE_ADAPTIVE,
} QrPresentMode;

typedef struct QrStartFrameInfo
{
    QrPresentMode   presentMode;
    uint32_t        maxFrameLatency;
    QrBool32        requestShaderReload;
} QrStartFrameInfo;

QRAPI QrResult QRCONV qrStartFrame(
    QrInstance                          qrInstance,
    const QrStartFrameInfo              *pStartInfo);

typedef enum QrSkyType
{
    QR_SKY_TYPE_COLOR,
    QR_SKY_TYPE_CUBEMAP,
    QR_SKY_TYPE_RASTERIZED_GEOMETRY,

    QR_SKY_TYPE_PROCEDURAL
} QrSkyType;

typedef struct QrDrawFrameTonemappingParams
{
    float       minLogLuminance;
    float       maxLogLuminance;
    float       luminanceWhitePoint;

    float       exposureBias;

    float       tonemapPower;
    float       exposureSpeedUp;
    float       exposureSpeedDown;
    float       exposureLowPercentile;
    float       exposureHighPercentile;
    float       minAdaptedLuminance;
    float       maxAdaptedLuminance;

    uint32_t    tonemapType;
} QrDrawFrameTonemappingParams;

#define QR_SKY_CLOUDS_MAX_QUALITY 3

typedef struct QrDrawFrameSkyParams
{
    QrSkyType   skyType;

    QrFloat3D   skyColorDefault;

    QrFloat3D   sunDiscColor;

    float       skyColorMultiplier;

    float       skyColorSaturation;
    float       skyAmbientLod;
    float       skyLightMultiplier;
    QrBool32    skyNee;

    QrFloat3D   skyViewerPosition;

    QrCubemap   skyCubemap;

    QrMatrix3D  skyCubemapRotationTransform;

    QrBool32    godRaysEnabled;

    float       godRaysIntensity;

    QrBool32    godRaysFromSkyTexture;
    QrFloat3D   godRaysSkyDirection;
    QrFloat3D   godRaysSkyColor;

    uint32_t    skyCloudsQuality;
    uint32_t    godRaysQuality;
    float       sunDiscSize;
} QrDrawFrameSkyParams;

#define QR_LIGHT_STYLE_COUNT 64

typedef struct QrDrawFrameTexturesParams
{
    QrSamplerFilter dynamicSamplerFilter;
    float           normalMapStrength;

    float           emissionMapBoost;

    float           emissionMaxScreenColor;
    float           emissionSharpMask;
    float           talSelfLitOffset;

    QrBool32        squareInputRoughness;

    float           minRoughness;

    uint32_t        emissionBlendMode;

    float           emissionBlendStrength;
    float           lightStyleScales[QR_LIGHT_STYLE_COUNT];
} QrDrawFrameTexturesParams;

typedef enum QrDebugDrawFlagBits
{
    QR_DEBUG_DRAW_ONLY_DIFFUSE_DIRECT_BIT = 1,
    QR_DEBUG_DRAW_ONLY_DIFFUSE_INDIRECT_BIT = 2,
    QR_DEBUG_DRAW_ONLY_SPECULAR_BIT = 4,
    QR_DEBUG_DRAW_UNFILTERED_DIFFUSE_DIRECT_BIT = 8,
    QR_DEBUG_DRAW_UNFILTERED_DIFFUSE_INDIRECT_BIT = 16,
    QR_DEBUG_DRAW_UNFILTERED_SPECULAR_BIT = 32,
    QR_DEBUG_DRAW_ALBEDO_WHITE_BIT = 64,
    QR_DEBUG_DRAW_MOTION_VECTORS_BIT = 128,
    QR_DEBUG_DRAW_GRADIENTS_BIT = 256,

    QR_DEBUG_DRAW_Q2RTX_CORE_BIT = 1024,

    QR_DEBUG_DRAW_GOD_RAYS_BIT = 2048,
    QR_DEBUG_DRAW_STATS_BIT = 4096,
    QR_DEBUG_DRAW_LUMA_BIT = 8192,

    QR_DEBUG_DRAW_PASS_STATS_BIT = 16384,
} QrDebugDrawFlagBits;
typedef QrFlags QrDebugDrawFlags;

typedef struct QrDrawFrameDebugParams
{
    QrDebugDrawFlags drawFlags;
} QrDrawFrameDebugParams;

typedef struct QrDrawFrameIlluminationParams
{
    uint32_t    maxBounceShadows;

    QrBool32    enableSecondBounceForIndirect;

    float       cellWorldSize;

    float       directDiffuseSensitivityToChange;

    float       indirectDiffuseSensitivityToChange;

    float       specularSensitivityToChange;

    float       polygonalLightSpotlightFactor;

    uint32_t    q2DepthGradMode;

    uint32_t    q2LightStatsMode;

    uint32_t    reflRefrEarlyOut;

    uint32_t    neeLightSamples;

    // 1: the direct pass samples the global light array with RIS instead of the
    // per-cluster light lists (host cvar rt_restir; 0 keeps the cluster path).
    // Default: 0
    uint32_t    restirEnabled;
    // Candidates drawn per NEE light sample in the global light RIS (host cvar
    // rt_restir_candidates), clamped to 1..64.
    // Default: 8
    uint32_t    restirCandidates;

    float       giBounceRays;

    QrBool32    denoiserEnabled;

    float       fixedAlbedo;

    float       sunBounceRange;

    float       sunBounceScale;

    uint64_t    *lightUniqueIdIgnoreFirstPersonViewerShadows;
} QrDrawFrameIlluminationParams;

typedef struct QrDrawFrameVolumetricParams
{
    QrBool32    enable;

    QrBool32    useSimpleDepthBased;

    float       volumetricFar;
    QrFloat3D   ambientColor;

    float       scaterring;

    QrFloat3D   sourceColor;
    QrFloat3D   sourceDirection;

    float       sourceAssymetry;
} QrDrawFrameVolumetricParams;

typedef struct QrDrawFrameLevelFogParams
{
    QrFloat3D   color;

    float       density;

    float       skyBlend;
} QrDrawFrameLevelFogParams;

#define QR_MAX_FOG_VOLUMES 8

typedef struct QrFogVolume
{
    QrFloat3D   pointA;
    QrFloat3D   pointB;

    QrFloat3D   color;

    float       halfExtinctionDistance;

    uint32_t    softface;
} QrFogVolume;

typedef struct QrDrawFrameBloomParams
{
    float       bloomIntensity;
    float       inputThreshold;
    float       bloomEmissionMultiplier;
} QrDrawFrameBloomParams;

typedef struct QrPostEffectWipe
{
    float       stripWidth;
    QrBool32    beginNow;
    float       duration;
} QrPostEffectWipe;

typedef struct QrPostEffectRadialBlur
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} QrPostEffectRadialBlur;

typedef struct QrPostEffectChromaticAberration
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
} QrPostEffectChromaticAberration;

typedef struct QrPostEffectInverseBlackAndWhite
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} QrPostEffectInverseBlackAndWhite;

typedef struct QrPostEffectHueShift
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} QrPostEffectHueShift;

typedef struct QrPostEffectDistortedSides
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} QrPostEffectDistortedSides;

typedef struct QrPostEffectWaves
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       amplitude;
    float       speed;
    float       xMultiplier;
} QrPostEffectWaves;

typedef struct QrPostEffectColorTint
{
    QrBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
    QrFloat3D   color;
} QrPostEffectColorTint;

typedef struct QrPostEffectCRT
{
    QrBool32    isActive;
} QrPostEffectCRT;

typedef struct QrPostEffectsBloomParams
{
    QrBool32    isActive;
    float       intensity;
    float       threshold;
    float       knee;
    float       scatter;
    float       radius;
    uint32_t    quality;
} QrPostEffectsBloomParams;

typedef struct QrPostEffectsNearDofParams
{
    float       strength;
    float       focusDistance;
    float       maxRadius;
} QrPostEffectsNearDofParams;

typedef struct QrPostEffectsSharpenParams
{
    QrBool32    isActive;
    float       strength;
} QrPostEffectsSharpenParams;

typedef struct QrPostEffectsGameplayFeedback
{
    float       damage;
    float       liquid;
    float       pickup;
    float       pickupHeight;
    float       aberration;
    QrFloat3D   pickupColor;
    float       suit;
} QrPostEffectsGameplayFeedback;

typedef struct QrPostEffectsVignetteParams
{
    float       intensity;
    float       start;
    float       end;
    float       roundness;
} QrPostEffectsVignetteParams;

typedef struct QrPostEffectsFilmGrainParams
{
    float       intensity;
    float       size;
} QrPostEffectsFilmGrainParams;

typedef struct QrDrawFramePostEffectsParams
{
    const QrPostEffectWipe                  *pWipe;
    const QrPostEffectRadialBlur            *pRadialBlur;
    const QrPostEffectChromaticAberration   *pChromaticAberration;
    const QrPostEffectInverseBlackAndWhite  *pInverseBlackAndWhite;
    const QrPostEffectHueShift              *pHueShift;
    const QrPostEffectDistortedSides        *pDistortedSides;
    const QrPostEffectWaves                 *pWaves;
    const QrPostEffectColorTint             *pColorTint;
    const QrPostEffectCRT                   *pCRT;
    const QrPostEffectsBloomParams          *pBloom;
    const QrPostEffectsNearDofParams        *pNearDof;
    const QrPostEffectsSharpenParams        *pSharpen;
    const QrPostEffectsGameplayFeedback     *pGameplayFeedback;
    const QrPostEffectsVignetteParams        *pVignette;
    const QrPostEffectsFilmGrainParams       *pFilmGrain;
    float                                  localExposure;
} QrDrawFramePostEffectsParams;

typedef enum QrMediaType
{
    QR_MEDIA_TYPE_VACUUM,
    QR_MEDIA_TYPE_WATER,
    QR_MEDIA_TYPE_GLASS,
    QR_MEDIA_TYPE_ACID,
} QrMediaType;

typedef struct QrDrawFrameReflectRefractParams
{
    uint32_t    maxReflectRefractDepth;

    QrMediaType typeOfMediaAroundCamera;

    float       indexOfRefractionGlass;

    float       indexOfRefractionWater;
    QrBool32    forceNoWaterRefraction;
    float       waterWaveSpeed;
    float       waterWaveNormalStrength;

    float       turbWarpStrength;

    QrFloat3D   waterColor;

    QrFloat3D   acidColor;
    float       acidDensity;

    float       waterWaveTextureDerivativesMultiplier;

    float       waterTextureAreaScale;

    QrBool32    disableBackfaceReflectionsForNoMediaChange;

    QrBool32    portalNormalTwirl;

    QrBool32    glassShadows;
    QrBool32    glassDenoise;
    QrBool32    glassParticles;
} QrDrawFrameReflectRefractParams;

typedef enum QrRenderUpscaleTechnique
{
    QR_RENDER_UPSCALE_TECHNIQUE_LINEAR,
    QR_RENDER_UPSCALE_TECHNIQUE_NEAREST,
    QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3,
    QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS,
} QrRenderUpscaleTechnique;

typedef enum QrRenderSharpenTechnique
{
    QR_RENDER_SHARPEN_TECHNIQUE_NONE,
    QR_RENDER_SHARPEN_TECHNIQUE_NAIVE,
    QR_RENDER_SHARPEN_TECHNIQUE_AMD_CAS,
} QrRenderSharpenTechnique;

typedef enum QrRenderResolutionMode
{
    QR_RENDER_RESOLUTION_MODE_CUSTOM,
    QR_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE,
    QR_RENDER_RESOLUTION_MODE_PERFORMANCE,
    QR_RENDER_RESOLUTION_MODE_BALANCED,
    QR_RENDER_RESOLUTION_MODE_QUALITY,
    QR_RENDER_RESOLUTION_MODE_ULTRA_QUALITY,
    QR_RENDER_RESOLUTION_MODE_NATIVE_AA,
} QrRenderResolutionMode;

typedef struct QrDrawFrameRenderResolutionParams
{
    QrRenderUpscaleTechnique    upscaleTechnique;
    QrRenderSharpenTechnique    sharpenTechnique;
    QrRenderResolutionMode      resolutionMode;

    QrExtent2D                  customRenderSize;

    const QrExtent2D            *pPixelizedRenderSize;
} QrDrawFrameRenderResolutionParams;

typedef enum QrDrawFrameRayCullFlagBits
{
    QR_DRAW_FRAME_RAY_CULL_WORLD_0_BIT  = 1,
    QR_DRAW_FRAME_RAY_CULL_WORLD_1_BIT  = 2,
    QR_DRAW_FRAME_RAY_CULL_WORLD_2_BIT  = 4,
    QR_DRAW_FRAME_RAY_CULL_SKY_BIT      = 8,
} QrDrawFrameRayCullFlagBits;
typedef QrFlags QrDrawFrameRayCullFlags;

typedef struct QrDrawFrameInfo
{
    float                   view[16];

    QrFloat3D               worldUpVector;

    float                   fovYRadians;

    float                   cameraNear;
    float                   cameraFar;

    float                   rayLength;

    QrDrawFrameRayCullFlags rayCullMaskWorld;

    QrBool32                disableRayTracedGeometry;
    QrBool32                disableRasterization;

    double                  currentTime;
    QrBool32                disableEyeAdaptation;
    QrBool32                forceAntiFirefly;

    const QrDrawFrameRenderResolutionParams     *pRenderResolutionParams;
    const QrDrawFrameIlluminationParams         *pIlluminationParams;
    const QrDrawFrameVolumetricParams           *pVolumetricParams;
    const QrDrawFrameTonemappingParams          *pTonemappingParams;
    const QrDrawFrameBloomParams                *pBloomParams;
    const QrDrawFrameReflectRefractParams       *pReflectRefractParams;
    const QrDrawFrameSkyParams                  *pSkyParams;
    const QrDrawFrameTexturesParams             *pTexturesParams;
    const QrDrawFrameLevelFogParams             *pLevelFogParams;
    const QrDrawFrameDebugParams                *pDebugParams;
    QrDrawFramePostEffectsParams                postEffectParams;
    QrBool32                                   renderUiOnly;
    QrBool32                                   enableCpuProfiling;
} QrDrawFrameInfo;

QRAPI QrResult QRCONV qrDrawFrame(
    QrInstance                          qrInstance,
    const QrDrawFrameInfo               *pDrawInfo);

QRAPI QrResult QRCONV qrSetFogVolumes(
    QrInstance                          qrInstance,
    uint32_t                            count,
    const QrFogVolume                   *pVolumes);

QRAPI QrBool32 QRCONV qrIsRenderUpscaleTechniqueAvailable(
    QrInstance                          qrInstance,
    QrRenderUpscaleTechnique            technique);

QRAPI QrBool32 QRCONV qrIsSuspended(
    QrInstance                          qrInstance);

#define QR_GPU_PASS_COUNT 18

#define QR_RAY_STATS_CATEGORY_COUNT 5

typedef enum QrCpuPassIndex
{
    QR_CPU_PASS_PREPARE = 0,
    QR_CPU_PASS_HOT_RELOAD,
    QR_CPU_PASS_DESCRIPTORS,
    QR_CPU_PASS_STAGING,
    QR_CPU_PASS_LEGACY_AS,
    QR_CPU_PASS_SLOT_WAIT,
    QR_CPU_PASS_SLOT_GC,
    QR_CPU_PASS_GPU_TIMINGS,
    QR_CPU_PASS_RHI_SETUP,
    QR_CPU_PASS_SCENE,
    QR_CPU_PASS_COMPOSE,
    QR_CPU_PASS_UPSCALE,
    QR_CPU_PASS_POST,
    QR_CPU_PASS_UI,
    QR_CPU_PASS_POSTUI,
    QR_CPU_PASS_PRESENT_RECORD,
    QR_CPU_PASS_RHI_SUBMIT,
    QR_CPU_PASS_LEGACY_SUBMIT,
    QR_CPU_PASS_PRESENT,
    QR_CPU_PASS_COUNT,
} QrCpuPassIndex;

typedef struct QrFrameStats
{
    uint32_t    raysTotal;
    uint32_t    raysPerCategory[QR_RAY_STATS_CATEGORY_COUNT];
    uint32_t    fpsX10;
    QrBool32    gpuTimingValid;
    float       gpuFrameMs;
    float       gpuPassMs[QR_GPU_PASS_COUNT];

    uint32_t    apiCalls;
    uint32_t    apiCallsGeometry;
    uint32_t    apiCallsRasterized;
    uint32_t    apiCallsLights;

    QrBool32    cpuTimingValid;
    QrBool32    renderedUiOnly;
    float       cpuPassMs[QR_CPU_PASS_COUNT];

    uint32_t    raysParticle;
    uint64_t    rasterUploadBytes;
    uint32_t    rasterUploadDroppedBatches;
} QrFrameStats;

QRAPI QrResult QRCONV qrGetFrameStatsEx(
    QrInstance                          qrInstance,
    QrFrameStats                       *pStats);

QRAPI const char *QRCONV qrGetCpuPassName(uint32_t passIndex);

typedef struct QrAdapterInfo
{
    char        name[256];
    char        driverName[256];
    char        driverInfo[256];
    uint32_t    vendorId;
    uint32_t    deviceId;
    uint32_t    driverVersion;
    uint32_t    apiVersion;
} QrAdapterInfo;

QRAPI QrResult QRCONV qrGetAdapterInfo(
    QrInstance                          qrInstance,
    QrAdapterInfo                      *pInfo);

QRAPI QrResult QRCONV qrRequestScreenshot(
    QrInstance                          qrInstance,
    const char                         *pFilePath);

QRAPI QrResult QRCONV qrGetFrameStats(
    QrInstance                          qrInstance,
    uint32_t                           *pRays,
    uint32_t                           *pFpsX10);

QRAPI const char* QRCONV qrGetGpuPassName(
    uint32_t                            passIndex);

QRAPI const char* QRCONV qrGetResultDescription(QrResult result);

#ifdef __cplusplus
}
#endif

#endif
