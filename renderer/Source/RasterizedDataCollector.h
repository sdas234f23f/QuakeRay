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

#pragma once

#include <vector>

#include <qray/qray.h>
#include "AutoBuffer.h"
#include "Common.h"
#include "ParticleProxies.h"
#include "TextureManager.h"
#include "Utils.h"

namespace qray
{
    struct ShVertex;

    class RasterizedDataCollector final
    {
    public:
        struct DrawInfo
        {
            QrTransform                 transform = {};
            std::optional< Float16D >   viewProj  = std::nullopt;
            std::optional< VkViewport > viewport  = std::nullopt;
            std::optional< VkRect2D >   scissor   = std::nullopt;

            uint32_t vertexCount = 0;
            uint32_t firstVertex = 0;
            uint32_t indexCount  = 0;
            uint32_t firstIndex  = 0;

            Float4D  color                = Float4D( NullifyToken );
            uint32_t textureIndex         = 0;
            uint32_t emissionTextureIndex = 0;

            QrRasterizedGeometryStateFlags pipelineState = 0;
            QrBlendFactor                  blendFuncSrc  = QR_BLEND_FACTOR_ONE;
            QrBlendFactor                  blendFuncDst  = QR_BLEND_FACTOR_ONE;

            Float4D  smokeNoise = Float4D( NullifyToken );
            Float4D  smokeLook  = Float4D( NullifyToken );

            // Whether every sprite of this draw has a traced stand-in in the frame's particle
            // proxy list (CaptureParticleProxies). The raster particle fragment discards the
            // sprites that lie behind a glass pane - the traced copy is what the pane shows -
            // and the flag keeps draws the proxy list does not describe rasterized.
            bool particleProxy = false;
        };

        struct ParticlePointDrawInfo
        {
            uint32_t firstPoint = 0;
            uint32_t count = 0;
            uint32_t textureIndex = 0;
            uint32_t pipelineState = 0;
            Float4D  smokeLook = Float4D( NullifyToken );
            uint32_t particleProxy = 0;
        };

    public:
        explicit RasterizedDataCollector( VkDevice                            device,
                                          std::shared_ptr< MemoryAllocator >& allocator,
                                          std::shared_ptr< TextureManager >   textureMgr,
                                          uint32_t                            maxVertexCount,
                                          uint32_t                            maxIndexCount );
        ~RasterizedDataCollector();

        RasterizedDataCollector( const RasterizedDataCollector& other )     = delete;
        RasterizedDataCollector( RasterizedDataCollector&& other ) noexcept = delete;
        RasterizedDataCollector& operator=( const RasterizedDataCollector& other ) = delete;

        RasterizedDataCollector& operator=( RasterizedDataCollector&& other ) noexcept = delete;
        bool                     AddGeometry( uint32_t                              frameIndex,
                                              const QrRasterizedGeometryUploadInfo& info,
                                              const float*                          viewProjection,
                                              const QrViewport*                     viewport );

        bool                     AddParticles( uint32_t                   frameIndex,
                                               const QrParticleUploadInfo& info );

        void Clear( uint32_t frameIndex );

        void CopyFromStaging( VkCommandBuffer cmd, uint32_t frameIndex );

        VkBuffer GetVertexBuffer() const;
        VkBuffer GetIndexBuffer() const;
        VkBuffer GetParticlePointBuffer() const;

        VkBuffer GetVertexStagingBuffer(uint32_t frameIndex);
        VkBuffer GetIndexStagingBuffer(uint32_t frameIndex);
        VkBuffer GetParticlePointStagingBuffer(uint32_t frameIndex);

        VkDeviceSize GetVertexBufferSize() const;
        VkDeviceSize GetIndexBufferSize() const;
        VkDeviceSize GetParticlePointBufferSize() const;

        static uint32_t GetVertexStride();
        static void     GetVertexLayout( VkVertexInputAttributeDescription* outAttrs,
                                         uint32_t*                          outAttrsCount );
        static void     GetSmokeVertexLayout( VkVertexInputAttributeDescription* outAttrs,
                                              uint32_t*                          outAttrsCount );

        static void     GetParticleVertexLayout( VkVertexInputAttributeDescription* outAttrs,
                                                 uint32_t*                          outAttrsCount );

        const std::vector< DrawInfo >& GetRasterDrawInfos() const;
        const std::vector< DrawInfo >& GetSwapchainDrawInfos() const;
        const std::vector< DrawInfo >& GetSkyDrawInfos() const;

        const std::vector< ParticlePointDrawInfo >& GetParticlePointDrawInfos() const;

        uint64_t GetUploadedBytes() const;
        uint32_t GetDroppedUploadBatches() const;

        // The frame's lit particle sprites as traced stand-ins (ParticleProxies.h), collected from
        // the same uploads the raster list sees. 'HasParticleProxyOverflow' reports a frame whose
        // sprites did not all fit, which disables the traced particle path for that frame - a
        // partial list would leave a raster hole wherever a proxy is missing.
        const std::vector< ParticleProxy >& GetParticleProxies() const;
        bool                                HasParticleProxyOverflow() const;

        // Diagnostics of the traced stand-in capture: what arrived as a particle draw and what the
        // capture did with it.
        struct ParticleCaptureStats
        {
            uint32_t candidateDraws   = 0;
            uint32_t capturedSprites  = 0;
            uint32_t capturedTriangles = 0;
            uint32_t rejectedBlend    = 0;
            uint32_t rejectedCap      = 0;
            uint32_t rejectedLines    = 0;
        };

        const ParticleCaptureStats& GetParticleCaptureStats() const;

    protected:
        DrawInfo& PushInfo( QrRasterizedGeometryRenderType renderType );

    private:
        static void CopyFromArrayOfStructs( const QrRasterizedGeometryUploadInfo& info,
                                            ShVertex*                             dstVerts );

        bool CaptureParticleProxies( const QrRasterizedGeometryUploadInfo& info );
        bool CaptureParticlePointProxies( const QrParticleUploadInfo& info );

    public:
        void SetParticleProxyCaptureEnabled( bool enabled ) { particleProxyCaptureEnabled = enabled; }

    private:
        bool particleProxyCaptureEnabled = true;

        VkDevice                          device;
        std::shared_ptr< TextureManager > textureMgr;

        std::shared_ptr< AutoBuffer > vertexBuffer;
        std::shared_ptr< AutoBuffer > indexBuffer;
        std::shared_ptr< AutoBuffer > particlePointBuffer;

        uint64_t curVertexCount;
        uint64_t curIndexCount;
        uint64_t curParticlePointCount;

        uint64_t uploadedBytes = 0;
        uint32_t droppedUploadBatches = 0;

        std::vector< DrawInfo > rasterDrawInfos;
        std::vector< DrawInfo > swapchainDrawInfos;
        std::vector< DrawInfo > skyDrawInfos;

        std::vector< ParticlePointDrawInfo > particlePointDrawInfos;

        std::vector< ParticleProxy > particleProxies;
        uint32_t                     fteTriangleCount = 0;
        bool                         particleProxyOverflow = false;
        ParticleCaptureStats         particleCaptureStats;
    };

}
