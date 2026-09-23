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

#include "Common.h"
#include "Buffer.h"
#include "GlobalUniform.h"
#include "MemoryAllocator.h"
#include "RasterizedDataCollector.h"
#include "RasterizerPipelines.h"
#include "TextureManager.h"

namespace vkpt
{

class RenderCubemap : public IShaderDependency
{
public:
    // params for the procedural sky compute pass (std140)
    struct ProceduralSkyParams
    {
        float faceBasis[18][4]; // 6 faces * (right, up, forward)
        float sunDirection[4];  // xyz = direction toward the sun, w = how much sun the sky shows (0 = no sun)
        float skyColor[4];      // xyz = the colour of the sky itself (rt_sky_color); w = 1 while the layer's shadow map is there to be read (see DrawProcedural), the sky itself not using it
        float skyParams[4];     // x = multiplier, y = cloud opacity (rt_sky_clouds_alpha), z = sun disc intensity, w = sun disc radius
        float cloudColor[4];    // xyz = cloud color, w = cloud time (s)
        float cloudParams[4];   // x = coverage, y = density (rt_sky_clouds_density), z = drift speed, w = enabled
        // Appended after everything else so that a stale compiled shader (which
        // does not know the field) still reads every field it does know at the
        // same offset.
        float sunDiscColor[4];  // xyz = colour of the sun disc (rt_sky_sun_color); w = the extent of the layer's shadow map in metres, the disc not using it
        float cloudLayer[4];    // x = altitude of the layer's bottom over the eye, y = thickness, z = sunlight strength, w = sky light strength
        float cloudMarch[4];    // x = view march steps, y = sun march steps, z = detail erosion strength, w = forward scattering
        float cloudAnchor[4];   // xy = the eye's place in the world's horizontal plane, z = its height in the world, w = which quarter of the layer's map this frame marches (CLOUD_UPDATE_FRAMES meaning all of it)
    };

public:
    RenderCubemap(VkDevice device,
                  const std::shared_ptr<MemoryAllocator> &allocator,
                  const std::shared_ptr<ShaderManager> &shaderManager,
                  const std::shared_ptr<TextureManager> &textureManager,
                  const std::shared_ptr<GlobalUniform> &uniform,
                  const std::shared_ptr<SamplerManager> &samplerManager,
                  const std::shared_ptr<CommandBufferManager> &cmdManager,
                  const RgInstanceCreateInfo &instanceInfo);
    ~RenderCubemap() override;

    RenderCubemap(const RenderCubemap &other) = delete;
    RenderCubemap(RenderCubemap &&other) noexcept = delete;
    RenderCubemap &operator=(const RenderCubemap &other) = delete;
    RenderCubemap &operator=(RenderCubemap &&other) noexcept = delete;

    // Draw to a cubemap
    void Draw(VkCommandBuffer cmd, uint32_t frameIndex,
               const std::shared_ptr< RasterizedDataCollector >& skyDataCollector,
              const std::shared_ptr<TextureManager> &textureManager,
              const std::shared_ptr<GlobalUniform> &uniform);

    // Fill the cubemap with a procedural atmospheric sky (compute)
    void DrawProcedural(VkCommandBuffer cmd, const ProceduralSkyParams &params);

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet() const;

    // Fills the volume of the shadow the cloud layer puts on the world. It is
    // laid out around `cameraPos` and it is redrawn only when the clouds, the sun
    // or that placement have moved enough to matter, so the volume the lighting
    // passes read always matches what GetCloudShadowPlacement() reports.
    void UpdateCloudShadow(VkCommandBuffer cmd, const ProceduralSkyParams &params,
                           const float cameraPos[3]);

    // Where the standing volume lies in the world: [0] is 1 while it holds
    // anything, [1..2] its world-space origin, [3] its extent in metres. The host
    // puts this in the global uniform for every pass that lights the world, which
    // is where CloudShadowMap.h reads it back from.
    void GetCloudShadowPlacement(float placement[4]) const;

    // Drops the standing volume: the clouds it was filled from are no longer drawn,
    // and until it is filled again nothing may be shadowed by it.
    void InvalidateCloudShadow();

    // The quality levels of rt_sky_clouds_quality and rt_sky_godrays_quality: low,
    // medium, high, ultra, extreme. Every level doubles the resolution the clouds
    // are drawn at and the volume of their shadow is laid over the ground with, and
    // leaves that volume standing for fewer frames (except the two finest levels,
    // which share the finest size of it).
    static constexpr uint32_t QUALITY_LOW     = 0;
    static constexpr uint32_t QUALITY_HIGH    = 2;
    static constexpr uint32_t QUALITY_EXTREME = 4;
    static constexpr uint32_t QUALITY_LEVELS  = QUALITY_EXTREME + 1;

    static uint32_t ClampQuality(uint32_t quality);

    // Turns the cloud layer and its shadow into the maps the level asks for: a
    // finer cubemap for the layer, a finer volume of its shadow read slice by
    // slice, and that volume is left standing for fewer frames. Called from the
    // sky pass once a frame with the command buffer the frame draws with; it does
    // nothing while the level is the one already in use. The layer is drawn again
    // from scratch, so the call is only worth making where the sky is about to be
    // drawn anyway.
    void SetQuality(VkCommandBuffer cmd, uint32_t quality);

    void OnShaderReload(const ShaderManager *shaderManager) override;
    

private:
    struct Attachment
    {
        VkImage image;
        VkImageView view;
        VkDeviceMemory memory;
    };

private:
    void CreatePipelineLayout(VkDescriptorSetLayout texturesSetLayout, VkDescriptorSetLayout uniformSetLayout);
    void CreateRenderPass();
    void InitPipelines(const std::shared_ptr<ShaderManager> &shaderManager, uint32_t sideSize, bool applyVertexColorGamma);
    void CreateAttch(const std::shared_ptr<MemoryAllocator> &allocator, VkCommandBuffer cmd, uint32_t sideSize, uint32_t mipLevels,
                     const char *debugName, Attachment &result, bool isDepth, bool allowFailure = false);
    void CreateFramebuffer(uint32_t sideSize);
    void CreateDescriptors(const std::shared_ptr<SamplerManager> &samplerManager);

    void BindPipelineIfNew(VkCommandBuffer cmd, const RasterizedDataCollector::DrawInfo &info, VkPipeline &curPipeline);

    void GenerateMipmaps(VkCommandBuffer cmd, VkImage image, VkImageLayout mip0Layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // procedural sky (compute)
    void CreateProceduralSkyPipelineLayout();
    void CreateProceduralSkyDescriptors(const std::shared_ptr<SamplerManager> &samplerManager);
    void CreateProceduralSkyParamsBuffer();
    void CreateProceduralSkyPipeline(const ShaderManager *shaderManager);
    void DestroyProceduralSkyPipelines();

    // cloud layer (compute, writes into the cubemap the sky composites)
    void CreateCloudsPipeline(const ShaderManager *shaderManager);
    void DestroyCloudsPipeline();
    void DispatchClouds(VkCommandBuffer cmd, const ProceduralSkyParams &params);

    // cloud shadow volume (compute, read back by every pass that lights the world)
    void CreateCloudShadowImage(const std::shared_ptr<MemoryAllocator> &allocator, VkCommandBuffer cmd,
                                uint32_t size, Attachment &image, bool allowFailure = false);
    void CreateCloudShadowDescriptors();
    void CreateCloudShadowParamsBuffer();
    void CreateCloudShadowPipelineLayout();
    void CreateCloudShadowPipeline(const ShaderManager *shaderManager);
    void DestroyCloudShadowPipeline();
    void DispatchCloudShadow(VkCommandBuffer cmd);

    // Both are recreated by SetQuality, so the descriptor sets that name them are
    // written again: nothing else about them changes, so no set, pool or layout is
    // recreated.
    void UpdateQualityDescriptors();

    uint32_t CloudShadowRefreshFrames() const;

private:
    // The cloud layer's shadow on the world: how much of the sun gets past the
    // clouds on the way down to a spot of it. That only depends on the column of
    // cloud a sun ray crosses before it reaches the spot, which is the same column
    // wherever along the ray the spot is, so the layer can be projected onto the
    // ground along the sun once and read back as a lookup (CloudShadowMap.h) --
    // walked up the whole column in one go, so that a point standing inside the
    // layer reads what is left above it rather than the whole of it.
    struct CloudShadowParams
    {
        float sunDirection[4];  // xyz = unit direction towards the sun, w = height of the layer's bottom over the eye
        float cloudLayer[4];    // x = thickness, y = coverage, z = density, w = detail erosion strength
        float cloudMarch[4];    // x = cloud time (s), y = drift speed, z = march steps per column, w = height of the layer over the plane the volume is keyed on
        float mapProjection[4]; // xy = world-space corner of the volume, z = its extent (m), w = its extent in texels a side
    };

    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;
    // SetQuality recreates the maps the frames in flight are still reading, so it
    // has to be able to wait for them.
    std::shared_ptr<CommandBufferManager> cmdManager;

    VkPipelineLayout pipelineLayout;
    std::shared_ptr<RasterizerPipelines> pipelines;

    VkRenderPass multiviewRenderPass;

    Attachment cubemap;
    Attachment envCubemap;
    Attachment cubemapDepth;

    VkFramebuffer cubemapFramebuffer;

    uint32_t cubemapSize;
    uint32_t cubemapMipLevels;

    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;

    // procedural sky (compute)
    Buffer procSkyParamsBuffer;
    void *mappedProcSkyParams = nullptr;

    VkDescriptorSetLayout procSkyDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      procSkyDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       procSkyDescSet       = VK_NULL_HANDLE;

    VkPipelineLayout procSkyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       procSkyPipeline       = VK_NULL_HANDLE;

    // The cloud layer that the procedural sky composites in front of everything it
    // draws: a small cubemap (rgb = light scattered in the cloud, a = how much of
    // the sky behind it gets through) sampled by direction.
    Attachment clouds;
    uint32_t   cloudsSize = 0;
    VkPipeline cloudsPipeline = VK_NULL_HANDLE;
    VkSampler  cloudsSampler  = VK_NULL_HANDLE; // names the layer in the sky's descriptor set

    // The layer is marched in a quarter of its map a frame (see DispatchClouds):
    // which quarter this frame is for, and whether the map is new and all of it has
    // to be filled at once rather than a quarter of it.
    uint32_t cloudsCycle = 0;
    bool     cloudsFullUpdate = true;

    // The quality level both the cloud cubemap and the volume of its shadow are
    // sized and refreshed at (see SetQuality).
    uint32_t   quality = QUALITY_HIGH;

    // A level whose maps could not be made (out of memory) is not asked for again
    // on every frame -- making and tearing down a map of that size is not free even
    // when the memory for it is missing -- but once in a while, so that memory freed
    // by another setting brings the level up on its own. QUALITY_LEVELS in
    // failedQuality means that no level is waiting to be taken.
    static constexpr uint32_t QUALITY_RETRY_FRAMES = 600;
    uint32_t   failedQuality = QUALITY_LEVELS;
    uint32_t   qualityRetryAge = 0;

    // cloud shadow volume (compute, read back by every pass that lights the world)
    Attachment cloudShadow;
    uint32_t   cloudShadowSize = 0;
    CloudShadowParams cloudShadowParams = {};   // the state the standing volume was filled with
    bool     cloudShadowValid = false;          // false until it is dispatched once
    uint32_t cloudShadowAge = 0;                // frames the standing map has been left alone
    float    cloudShadowPlacement[4] = {};      // where it stands: on, origin.x, origin.z, extent

    VkDescriptorSetLayout cloudShadowDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      cloudShadowDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       cloudShadowDescSet       = VK_NULL_HANDLE;

    VkPipelineLayout cloudShadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       cloudShadowPipeline       = VK_NULL_HANDLE;
    VkSampler        cloudShadowSampler        = VK_NULL_HANDLE; // names the volume in the cubemap descriptor set

    Buffer cloudShadowParamsBuffer;
    void *mappedCloudShadowParams = nullptr;
};

}
