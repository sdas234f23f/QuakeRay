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

#include "RasterizedDataCollector.h"

#include <algorithm>
#include <cmath>

#include "Utils.h"
#include "QrException.h"
#include "Generated/ShaderCommonC.h"

using namespace qray;

namespace
{
    struct VertexAttribute
    {
        uint32_t location;
        VkFormat format;
        size_t   offset;
    };

    void FillVertexAttributes(
        const VertexAttribute *attrs, size_t attrCount,
        VkVertexInputAttributeDescription *outAttrs, uint32_t *outAttrsCount)
    {
        for (size_t i = 0; i < attrCount; i++)
        {
            outAttrs[i].binding = 0;
            outAttrs[i].location = attrs[i].location;
            outAttrs[i].format = attrs[i].format;
            outAttrs[i].offset = (uint32_t)attrs[i].offset;
        }

        *outAttrsCount = (uint32_t)attrCount;
    }

    bool IsWorld(QrRasterizedGeometryRenderType type)
    {
        return type == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT;
    }

    bool IsSwapchain(QrRasterizedGeometryRenderType type)
    {
        return type == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN;
    }

    bool IsSky(QrRasterizedGeometryRenderType type)
    {
        return type == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY;
    }

    VkViewport ToVkViewport(const QrViewport &v)
    {
        return VkViewport{
            .x        = v.x,
            .y        = v.y,
            .width    = v.width,
            .height   = v.height,
            .minDepth = v.minDepth,
            .maxDepth = v.maxDepth,
        };
    }

    VkRect2D ToVkRect2D(const QrRect2D &r)
    {
        return VkRect2D{
            .offset = { r.x, r.y },
            .extent = { r.width, r.height },
        };
    }

    uint32_t ResolveTextureIndex_AlbedoAlpha(
        const qray::TextureManager &manager, QrMaterial material)
    {
        if (material == QR_NO_MATERIAL)
        {
            return EMPTY_TEXTURE_INDEX;
        }

        return manager.GetMaterialTextures(material).indices[MATERIAL_ALBEDO_ALPHA_INDEX];
    }

    uint32_t ResolveTextureIndex_AlbedoAlpha(
        const qray::TextureManager &manager, const QrRasterizedGeometryUploadInfo &info)
    {
        return ResolveTextureIndex_AlbedoAlpha(manager, info.material);
    }

    uint32_t ResolveTextureIndex_RME(
        const qray::TextureManager &manager, const QrRasterizedGeometryUploadInfo &info)
    {
        if (info.material == QR_NO_MATERIAL)
        {
            return EMPTY_TEXTURE_INDEX;
        }

        if (!IsWorld(info.renderType))
        {
            return EMPTY_TEXTURE_INDEX;
        }

        return manager.GetMaterialTextures(info.material).indices[MATERIAL_ROUGHNESS_METALLIC_EMISSION_INDEX];
    }

    constexpr float PARTICLE_POINT_LEG_SCALE = 1.5f;
}

void RasterizedDataCollector::GetVertexLayout(
    VkVertexInputAttributeDescription *outAttrs, uint32_t *outAttrsCount)
{
    const VertexAttribute attrs[] =
    {
        { 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(QrVertex, position)    },
        { 1, VK_FORMAT_R8G8B8A8_UNORM,   offsetof(QrVertex, packedColor) },
        { 2, VK_FORMAT_R32G32_SFLOAT,    offsetof(QrVertex, texCoord)    },
    };

    FillVertexAttributes(attrs, std::size(attrs), outAttrs, outAttrsCount);
}

uint32_t RasterizedDataCollector::GetVertexStride()
{
    return static_cast<uint32_t>(sizeof(QrVertex));
}

void RasterizedDataCollector::GetSmokeVertexLayout(
    VkVertexInputAttributeDescription *outAttrs, uint32_t *outAttrsCount)
{
    const VertexAttribute attrs[] =
    {
        { 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(QrVertex, position)     },
        { 1, VK_FORMAT_R8G8B8A8_UNORM,   offsetof(QrVertex, packedColor)  },
        { 2, VK_FORMAT_R32G32_SFLOAT,    offsetof(QrVertex, texCoord)     },
        { 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(QrVertex, normal)       },
        { 4, VK_FORMAT_R32G32_SFLOAT,    offsetof(QrVertex, texCoordLayer1) },
        { 5, VK_FORMAT_R32_UINT,         offsetof(QrVertex, cluster)      },
    };

    FillVertexAttributes(attrs, std::size(attrs), outAttrs, outAttrsCount);
}

void RasterizedDataCollector::GetParticleVertexLayout(
    VkVertexInputAttributeDescription *outAttrs, uint32_t *outAttrsCount)
{
    const VertexAttribute attrs[] =
    {
        { 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(QrVertex, position)    },
        { 1, VK_FORMAT_R8G8B8A8_UNORM,   offsetof(QrVertex, packedColor) },
        { 2, VK_FORMAT_R32G32_SFLOAT,    offsetof(QrVertex, texCoord)    },
        { 3, VK_FORMAT_R32_UINT,         offsetof(QrVertex, cluster)     },
    };

    FillVertexAttributes(attrs, std::size(attrs), outAttrs, outAttrsCount);
}

RasterizedDataCollector::RasterizedDataCollector( VkDevice                            _device,
                                                  std::shared_ptr< MemoryAllocator >& _allocator,
                                                  std::shared_ptr< TextureManager >   _textureMgr,
                                                  uint32_t _maxVertexCount,
                                                  uint32_t _maxIndexCount )
    : device( _device )
    , textureMgr( std::move( _textureMgr ) )
    , curVertexCount( 0 )
    , curIndexCount( 0 )
    , curParticlePointCount( 0 )
{
    vertexBuffer = std::make_shared<AutoBuffer>(_device, _allocator);
    indexBuffer = std::make_shared<AutoBuffer>(_device, _allocator);
    particlePointBuffer = std::make_shared<AutoBuffer>(_device, _allocator);

    _maxVertexCount = std::max(_maxVertexCount, 64u);
    _maxIndexCount = std::max(_maxIndexCount, 64u);

    vertexBuffer->Create(_maxVertexCount * sizeof(QrVertex),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         "Rasterizer vertex buffer");
    indexBuffer->Create(_maxIndexCount * sizeof(uint32_t),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        "Rasterizer index buffer");
    particlePointBuffer->Create(_maxVertexCount * sizeof(QrParticlePoint),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                "Rasterizer particle point buffer");
}

RasterizedDataCollector::~RasterizedDataCollector()
{
}

bool RasterizedDataCollector::AddGeometry(uint32_t frameIndex,
                                          const QrRasterizedGeometryUploadInfo &info,
                                          const float *pViewProjection, const QrViewport *pViewport)
{
    assert(info.vertexCount > 0);
    assert(info.pVertices != nullptr);

    if (IsSwapchain(info.renderType))
    {
        if (info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST)
        {
            assert(0);
            return false;
        }

        if (info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_DEPTH_WRITE)
        {
            assert(0);
            return false;
        }
    }

    if (IsSky(info.renderType))
    {
        if (pViewProjection != nullptr || pViewport != nullptr)
        {
            throw QrException(QR_CANT_UPLOAD_RASTERIZED_GEOMETRY, "pViewProjection and pViewport must be null if renderType is QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY");
        }
    }

    if (curVertexCount + info.vertexCount >= vertexBuffer->GetSize() / sizeof(QrVertex))
    {
        droppedUploadBatches++;
        return false;
    }

    if (curIndexCount + info.indexCount >= indexBuffer->GetSize() / sizeof(uint32_t))
    {
        droppedUploadBatches++;
        return false;
    }

    DrawInfo &drawInfo = PushInfo(info.renderType);

    ShVertex* const vertsBase   = static_cast< ShVertex* >( vertexBuffer->GetMapped( frameIndex ) );
    uint32_t* const indicesBase = static_cast< uint32_t* >( indexBuffer->GetMapped( frameIndex ) );

    drawInfo = {
        .transform            = info.transform,
        .viewProj             = IfNotNull( pViewProjection, Float16D( pViewProjection ) ),
        .viewport             = IfNotNull( pViewport, ToVkViewport( *pViewport ) ),
        .scissor              = info.scissor.width > 0 ? std::optional< VkRect2D >( ToVkRect2D( info.scissor ) )
                                                      : std::nullopt,
        .color                = Float4D( info.color.data ),
        .textureIndex         = ResolveTextureIndex_AlbedoAlpha( *textureMgr, info ),
        .emissionTextureIndex = ResolveTextureIndex_RME( *textureMgr, info ),
        .pipelineState        = info.pipelineState,
        .blendFuncSrc         = info.blendFuncSrc,
        .blendFuncDst         = info.blendFuncDst,
        .smokeNoise           = Float4D( info.smokeNoise.data ),
        .smokeLook            = Float4D( info.smokeLook.data ),
    };

    CopyFromArrayOfStructs( info, &vertsBase[ curVertexCount ] );

    uploadedBytes += static_cast< uint64_t >( info.vertexCount ) * sizeof( QrVertex );

    drawInfo.vertexCount = info.vertexCount;
    drawInfo.firstVertex  = static_cast< uint32_t >( curVertexCount );
    curVertexCount += info.vertexCount;

    if( info.indexCount != 0 && info.pIndices != nullptr )
    {
        if( curIndexCount + info.indexCount >= indexBuffer->GetSize() / sizeof( uint32_t ) )
        {
            droppedUploadBatches++;
            return false;
        }

        memcpy( &indicesBase[ curIndexCount ], info.pIndices, info.indexCount * sizeof( uint32_t ) );

        uploadedBytes += static_cast< uint64_t >( info.indexCount ) * sizeof( uint32_t );

        drawInfo.indexCount = info.indexCount;
        drawInfo.firstIndex = static_cast< uint32_t >( curIndexCount );

        curIndexCount += info.indexCount;
    }

    drawInfo.particleProxy = CaptureParticleProxies(info);

    return true;
}

bool RasterizedDataCollector::AddParticles(uint32_t frameIndex, const QrParticleUploadInfo &info)
{
    assert(info.count > 0);
    assert(info.pPoints != nullptr);

    if (curParticlePointCount + info.count > particlePointBuffer->GetSize() / sizeof(QrParticlePoint))
    {
        droppedUploadBatches++;
        return false;
    }

    ParticlePointDrawInfo &drawInfo = particlePointDrawInfos.emplace_back();

    drawInfo = {
        .firstPoint    = static_cast< uint32_t >( curParticlePointCount ),
        .count         = info.count,
        .textureIndex  = ResolveTextureIndex_AlbedoAlpha( *textureMgr, info.material ),
        .pipelineState = info.pipelineState,
        .smokeLook     = Float4D( info.smokeLook.data ),
    };

    QrParticlePoint *const pointsBase =
        static_cast< QrParticlePoint* >( particlePointBuffer->GetMapped( frameIndex ) );

    memcpy( &pointsBase[ curParticlePointCount ], info.pPoints, info.count * sizeof( QrParticlePoint ) );

    uploadedBytes += static_cast< uint64_t >( info.count ) * sizeof( QrParticlePoint );

    curParticlePointCount += info.count;

    drawInfo.particleProxy = CaptureParticlePointProxies(info) ? 1u : 0u;

    return true;
}

RasterizedDataCollector::DrawInfo& RasterizedDataCollector::PushInfo(
    QrRasterizedGeometryRenderType renderType )
{
    if( renderType == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT )
    {
        return rasterDrawInfos.emplace_back();
    }

    if( renderType == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN )
    {
        return swapchainDrawInfos.emplace_back();
    }

    if( renderType == QR_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY )
    {
        return skyDrawInfos.emplace_back();
    }

    throw QrException( QR_GRAPHICS_API_ERROR, "RasterizedDataCollector::PushInfo error" );
}

void RasterizedDataCollector::CopyFromArrayOfStructs(
    const QrRasterizedGeometryUploadInfo &info, ShVertex *dstVerts)
{
    assert(info.pVertices != nullptr);

    static_assert(std::is_same_v<decltype(info.pVertices), const QrVertex * >);
    static_assert(sizeof(ShVertex)                      == sizeof(QrVertex));
    static_assert(offsetof(ShVertex, position)          == offsetof(QrVertex, position));
    static_assert(offsetof(ShVertex, normal)            == offsetof(QrVertex, normal));
    static_assert(offsetof(ShVertex, texCoord)          == offsetof(QrVertex, texCoord));
    static_assert(offsetof(ShVertex, texCoordLayer1)    == offsetof(QrVertex, texCoordLayer1));
    static_assert(offsetof(ShVertex, texCoordLayer2)    == offsetof(QrVertex, texCoordLayer2));
    static_assert(offsetof(ShVertex, packedColor)       == offsetof(QrVertex, packedColor));

    memcpy(dstVerts, info.pVertices, sizeof(QrVertex) * info.vertexCount);
}

bool RasterizedDataCollector::CaptureParticleProxies(const QrRasterizedGeometryUploadInfo &info)
{
    if (!particleProxyCaptureEnabled)
    {
        return false;
    }

    // The sprite marker, not the lit-pipeline selector: the traced stand-ins have to exist for the
    // unlit sprites too (the raster copy of those is drawn by the world pipeline).
    if ((info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_PARTICLE_SPRITE) == 0)
    {
        return false;
    }

    particleCaptureStats.candidateDraws++;

    const bool lit = (info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_PARTICLE) != 0;

    // The blend the raster copy uses, folded into the composite's premultiplied "over". The modes
    // that scale the background per channel (SRC_COLOR/ONE_MINUS_SRC_COLOR and the two relatives)
    // have no fold here and stay raster-only, without a proxy and so without the discard.
    uint32_t blendOp = ~0u;
    if (info.blendFuncSrc == QR_BLEND_FACTOR_SRC_ALPHA && info.blendFuncDst == QR_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
    {
        blendOp = PARTICLE_BLEND_ALPHA_OVER;
    }
    else if (info.blendFuncSrc == QR_BLEND_FACTOR_ONE && info.blendFuncDst == QR_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
    {
        blendOp = PARTICLE_BLEND_PREMUL;
    }
    else if (info.blendFuncSrc == QR_BLEND_FACTOR_SRC_ALPHA && info.blendFuncDst == QR_BLEND_FACTOR_ONE)
    {
        blendOp = PARTICLE_BLEND_ADD_ALPHA;
    }
    else if (info.blendFuncSrc == QR_BLEND_FACTOR_SRC_COLOR && info.blendFuncDst == QR_BLEND_FACTOR_ONE)
    {
        blendOp = PARTICLE_BLEND_ADD_COLOR;
    }
    else if (info.blendFuncSrc == QR_BLEND_FACTOR_ZERO && info.blendFuncDst == QR_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
    {
        blendOp = PARTICLE_BLEND_MUL_INV_ALPHA;
    }

    if (blendOp == ~0u)
    {
        particleCaptureStats.rejectedBlend++;
        return false;
    }

    const uint32_t textureIndex = ResolveTextureIndex_AlbedoAlpha(*textureMgr, info);
    const size_t firstProxy = particleProxies.size();

    if (info.indexCount == 0)
    {
        // The classic sprite: the non-indexed three-vertex camera-facing triangle r_part.c emits
        // (Quake/r_part.c), which only ever blends straight alpha.
        if (blendOp != PARTICLE_BLEND_ALPHA_OVER || info.vertexCount < 3 || (info.vertexCount % 3) != 0)
        {
            return false;
        }

        const uint32_t particleCount = info.vertexCount / 3;
        if (uint64_t(particleProxies.size()) + particleCount > MAX_PARTICLE_PROXY_COUNT)
        {
            particleProxyOverflow = true;
            particleCaptureStats.rejectedCap++;
        return false;
        }

        static constexpr float cornerU[3] = { -1.0f / 3.0f, 2.0f / 3.0f, -1.0f / 3.0f };
        static constexpr float cornerV[3] = { -1.0f / 3.0f, -1.0f / 3.0f, 2.0f / 3.0f };

        for (uint32_t i = 0; i < particleCount; i++)
        {
            const QrVertex &v0 = info.pVertices[3 * i];
            const QrVertex &v1 = info.pVertices[3 * i + 1];
            const QrVertex &v2 = info.pVertices[3 * i + 2];

            float leg[2][3];
            for (int c = 0; c < 3; c++)
            {
                leg[0][c] = v1.position[c] - v0.position[c];
                leg[1][c] = v2.position[c] - v0.position[c];
            }

            const float legRight = std::sqrt(leg[0][0] * leg[0][0] + leg[0][1] * leg[0][1] +
                                             leg[0][2] * leg[0][2]);
            const float legUp = std::sqrt(leg[1][0] * leg[1][0] + leg[1][1] * leg[1][1] +
                                          leg[1][2] * leg[1][2]);
            if (!(legRight > 1e-4f) || !(legUp > 1e-4f))
            {
                continue;
            }

            ParticleProxy &proxy = particleProxies.emplace_back();
            proxy = ParticleProxy{};
            proxy.kind = PARTICLE_PROXY_KIND_BILLBOARD;
            proxy.blendOp = blendOp;

            for (int c = 0; c < 3; c++)
            {
                proxy.center[c] = v0.position[c] + (leg[0][c] + leg[1][c]) / 3.0f;
            }

            float radius = 0.0f;
            for (int corner = 0; corner < 3; corner++)
            {
                float offset[3];
                for (int c = 0; c < 3; c++)
                {
                    offset[c] = cornerU[corner] * leg[0][c] + cornerV[corner] * leg[1][c];
                }

                radius = std::max(radius, std::sqrt(offset[0] * offset[0] + offset[1] * offset[1] +
                                                    offset[2] * offset[2]));
            }

            proxy.radius       = radius;
            proxy.legRight     = legRight;
            proxy.legUp        = legUp;
            proxy.packedColor  = v0.packedColor;
            proxy.textureIndex = textureIndex;
            proxy.cluster      = v0.cluster;
            proxy.direct       = lit ? info.smokeLook.data[1] : 0.0f;
            proxy.gain         = lit ? info.smokeLook.data[2] : -1.0f;
            proxy.lightFloor   = lit ? info.smokeLook.data[3] : 0.0f;
        }

        particleCaptureStats.capturedSprites += static_cast<uint32_t>(particleProxies.size() - firstProxy);
        return particleProxies.size() > firstProxy;
    }

    // An FTE effect: indexed world-space triangles (Quake/r_part_fte.c). They are exact geometry,
    // so the record carries the triangle itself and the traced copy shows the same shape whichever
    // way the pane bends the view at it. The three vertex colours ride the record's spare words.
    // The line primitives (BEF_LINES) are not triangles and cannot be captured as such: their
    // raster copy stays, the same documented exception as the per-channel blend modes.
    if (info.pIndices == nullptr || info.indexCount < 3 ||
        (info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST) != 0)
    {
        particleCaptureStats.rejectedLines++;
        return false;
    }

    const uint32_t triangleCount = info.indexCount / 3;
    if (triangleCount > MAX_FTE_TRIANGLES_PER_DRAW ||
        uint64_t(fteTriangleCount) + triangleCount > MAX_FTE_TRIANGLE_COUNT ||
        uint64_t(particleProxies.size()) + triangleCount > MAX_PARTICLE_PROXY_COUNT)
    {
        particleProxyOverflow = true;
        particleCaptureStats.rejectedCap++;
        return false;
    }

    const uint32_t *const indices = static_cast<const uint32_t *>(info.pIndices);
    uint32_t captured = 0;

    for (uint32_t t = 0; t < triangleCount; t++)
    {
        const uint32_t i0 = indices[3 * t + 0];
        const uint32_t i1 = indices[3 * t + 1];
        const uint32_t i2 = indices[3 * t + 2];
        if (i0 >= info.vertexCount || i1 >= info.vertexCount || i2 >= info.vertexCount)
        {
            continue;
        }

        const QrVertex &a = info.pVertices[i0];
        const QrVertex &b = info.pVertices[i1];
        const QrVertex &c = info.pVertices[i2];

        ParticleProxy &proxy = particleProxies.emplace_back();
        proxy = ParticleProxy{};
        proxy.kind         = PARTICLE_PROXY_KIND_TRIANGLE;
        proxy.blendOp      = blendOp;
        proxy.textureIndex = textureIndex;
        proxy.cluster      = a.cluster;
        proxy.packedColor  = a.packedColor;
        proxy.unused2      = b.packedColor;
        proxy.unused3      = c.packedColor;
        proxy.direct       = lit ? info.smokeLook.data[1] : 0.0f;
        proxy.gain         = lit ? info.smokeLook.data[2] : -1.0f;
        proxy.lightFloor   = lit ? info.smokeLook.data[3] : 0.0f;

        float minimum[3];
        float maximum[3];
        float halfExtent[3];
        for (int k = 0; k < 3; k++)
        {
            proxy.v0[k]     = a.position[k];
            proxy.edge1[k]  = b.position[k] - a.position[k];
            proxy.edge2[k]  = c.position[k] - a.position[k];

            minimum[k] = std::min(a.position[k], std::min(b.position[k], c.position[k]));
            maximum[k] = std::max(a.position[k], std::max(b.position[k], c.position[k]));
            proxy.center[k] = 0.5f * (minimum[k] + maximum[k]);
            halfExtent[k]   = 0.5f * (maximum[k] - minimum[k]);
        }
        proxy.radius = std::sqrt(halfExtent[0] * halfExtent[0] + halfExtent[1] * halfExtent[1] +
                                 halfExtent[2] * halfExtent[2]) + 1e-3f;

        proxy.uv0[0] = a.texCoord[0];
        proxy.uv0[1] = a.texCoord[1];
        proxy.uv1[0] = b.texCoord[0];
        proxy.uv1[1] = b.texCoord[1];
        proxy.uv2[0] = c.texCoord[0];
        proxy.uv2[1] = c.texCoord[1];

        captured++;
    }

    fteTriangleCount += captured;
    particleCaptureStats.capturedTriangles += captured;
    return particleProxies.size() > firstProxy;
}

bool RasterizedDataCollector::CaptureParticlePointProxies(const QrParticleUploadInfo &info)
{
    if (!particleProxyCaptureEnabled)
    {
        return false;
    }

    if ((info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_PARTICLE_SPRITE) == 0)
    {
        return false;
    }

    particleCaptureStats.candidateDraws++;

    const bool lit = (info.pipelineState & QR_RASTERIZED_GEOMETRY_STATE_PARTICLE) != 0;

    if (uint64_t(particleProxies.size()) + info.count > MAX_PARTICLE_PROXY_COUNT)
    {
        particleProxyOverflow = true;
        particleCaptureStats.rejectedCap++;
        return false;
    }

    const uint32_t textureIndex = ResolveTextureIndex_AlbedoAlpha(*textureMgr, info.material);
    const size_t firstProxy = particleProxies.size();

    for (uint32_t i = 0; i < info.count; i++)
    {
        const QrParticlePoint &point = info.pPoints[i];
        const float leg = PARTICLE_POINT_LEG_SCALE * point.size;
        if (!(leg > 1e-4f))
        {
            continue;
        }

        ParticleProxy &proxy = particleProxies.emplace_back();
        proxy = ParticleProxy{};
        proxy.kind = PARTICLE_PROXY_KIND_BILLBOARD;
        proxy.blendOp = PARTICLE_BLEND_ALPHA_OVER;

        for (int c = 0; c < 3; c++)
        {
            proxy.center[c] = point.position[c] +
                              (info.viewRight.data[c] + info.viewUp.data[c]) * (leg / 3.0f);
        }

        proxy.radius       = std::sqrt(5.0f) / 3.0f * leg;
        proxy.legRight     = leg;
        proxy.legUp        = leg;
        proxy.packedColor  = point.packedColor;
        proxy.textureIndex = textureIndex;
        proxy.cluster      = point.cluster;
        proxy.direct       = lit ? info.smokeLook.data[1] : 0.0f;
        proxy.gain         = lit ? info.smokeLook.data[2] : -1.0f;
        proxy.lightFloor   = lit ? info.smokeLook.data[3] : 0.0f;
    }

    particleCaptureStats.capturedSprites += static_cast<uint32_t>(particleProxies.size() - firstProxy);
    return particleProxies.size() > firstProxy;
}

void RasterizedDataCollector::Clear(uint32_t frameIndex)
{
    rasterDrawInfos.clear();
    swapchainDrawInfos.clear();
    skyDrawInfos.clear();
    particlePointDrawInfos.clear();
    particleProxies.clear();
    fteTriangleCount = 0;
    particleProxyOverflow = false;
    particleCaptureStats = ParticleCaptureStats{};

    curVertexCount = 0;
    curIndexCount = 0;
    curParticlePointCount = 0;

    uploadedBytes = 0;
    droppedUploadBatches = 0;
}

void RasterizedDataCollector::CopyFromStaging(VkCommandBuffer cmd, uint32_t frameIndex)
{
    vertexBuffer->CopyFromStaging(cmd, frameIndex, sizeof(QrVertex) * curVertexCount);
    indexBuffer->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * curIndexCount);
}

VkBuffer RasterizedDataCollector::GetVertexBuffer() const
{
    return vertexBuffer->GetDeviceLocal();
}

VkBuffer RasterizedDataCollector::GetIndexBuffer() const
{
    return indexBuffer->GetDeviceLocal();
}

VkBuffer RasterizedDataCollector::GetParticlePointBuffer() const
{
    return particlePointBuffer->GetDeviceLocal();
}

VkBuffer RasterizedDataCollector::GetVertexStagingBuffer(uint32_t frameIndex)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);
    return vertexBuffer->GetStaging(frameIndex);
}

VkBuffer RasterizedDataCollector::GetIndexStagingBuffer(uint32_t frameIndex)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);
    return indexBuffer->GetStaging(frameIndex);
}

VkBuffer RasterizedDataCollector::GetParticlePointStagingBuffer(uint32_t frameIndex)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);
    return particlePointBuffer->GetStaging(frameIndex);
}

VkDeviceSize RasterizedDataCollector::GetVertexBufferSize() const
{
    return vertexBuffer->GetSize();
}

VkDeviceSize RasterizedDataCollector::GetIndexBufferSize() const
{
    return indexBuffer->GetSize();
}

VkDeviceSize RasterizedDataCollector::GetParticlePointBufferSize() const
{
    return particlePointBuffer->GetSize();
}

const std::vector< RasterizedDataCollector::DrawInfo >& RasterizedDataCollector::
    GetRasterDrawInfos() const
{
    return rasterDrawInfos;
}

const std::vector< ParticleProxy >& RasterizedDataCollector::GetParticleProxies() const
{
    return particleProxies;
}

bool RasterizedDataCollector::HasParticleProxyOverflow() const
{
    return particleProxyOverflow;
}

const RasterizedDataCollector::ParticleCaptureStats& RasterizedDataCollector::GetParticleCaptureStats() const
{
    return particleCaptureStats;
}

const std::vector< RasterizedDataCollector::DrawInfo >& RasterizedDataCollector::
    GetSwapchainDrawInfos() const
{
    return swapchainDrawInfos;
}

const std::vector< RasterizedDataCollector::DrawInfo >& RasterizedDataCollector::
    GetSkyDrawInfos() const
{
    return skyDrawInfos;
}

const std::vector< RasterizedDataCollector::ParticlePointDrawInfo >& RasterizedDataCollector::
    GetParticlePointDrawInfos() const
{
    return particlePointDrawInfos;
}

uint64_t RasterizedDataCollector::GetUploadedBytes() const
{
    return uploadedBytes;
}

uint32_t RasterizedDataCollector::GetDroppedUploadBatches() const
{
    return droppedUploadBatches;
}
