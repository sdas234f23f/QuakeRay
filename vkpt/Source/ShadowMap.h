// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shadow_map.c from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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

#pragma once

#include "Common.h"
#include "CommandBufferManager.h"
#include "IShaderDependency.h"
#include "MemoryAllocator.h"
#include "ShaderManager.h"
#include "VertexCollector.h"

namespace vkpt
{

// Renders the world geometry into a depth-only shadow map from the sun's
// point of view. Used by god rays for volumetric sunlight.
// Ported from Quake 2 RTX (shadow_map.c / shadow_map.vert), GPL v2.
class ShadowMap : public IShaderDependency
{
public:
    ShadowMap(VkDevice device, std::shared_ptr<MemoryAllocator> &allocator,
              const std::shared_ptr<const ShaderManager> &shaderManager,
              const std::shared_ptr<CommandBufferManager> &cmdManager,
              uint32_t quality);
    ~ShadowMap();

    ShadowMap(const ShadowMap &other) = delete;
    ShadowMap(ShadowMap &&other) noexcept = delete;
    ShadowMap &operator=(const ShadowMap &other) = delete;
    ShadowMap &operator=(ShadowMap &&other) noexcept = delete;

    // Renders static+dynamic world geometry into the shadow map.
    // outViewProjection (column-major, 16 floats) and outDepthScale are used
    // by the god rays shader. Returns false if there is nothing to draw.
    bool Render(VkCommandBuffer cmd,
                const float sunDirection[3],
                const float aabbMin[3], const float aabbMax[3],
                const VertexCollector *staticCollector,
                const VertexCollector *dynamicCollector,
                float outViewProjection[16], float *outDepthScale);

    // The quality levels of rt_sky_godrays_quality, the ladder the renderer's quality
    // settings are all on: low, medium, high, ultra, extreme. The level is the
    // resolution the map is drawn at, and that is what decides how crisp the edges
    // of the shafts traced through it are -- and what they cost to draw.
    static constexpr uint32_t QUALITY_LOW     = 0;
    static constexpr uint32_t QUALITY_HIGH    = 2;
    static constexpr uint32_t QUALITY_EXTREME = 4;
    static constexpr uint32_t QUALITY_LEVELS  = QUALITY_EXTREME + 1;

    static uint32_t ClampQuality(uint32_t quality);
    static uint32_t SizeForQuality(uint32_t quality);

    // Redraws the map at the resolution the level asks for. The map is named by the
    // descriptor set the shaft pass binds, so the frames still in flight may be
    // reading it and this waits for the device; the viewport of the pipeline that
    // fills it is dynamic, so nothing else has to be built again.
    void SetQuality(uint32_t quality);

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet() const;

    void OnShaderReload(const ShaderManager *shaderManager);

private:
    void CreateImage();
    // Fills the handles with an image of the given size, its view and the sampler
    // that names it. False (and no handles) when the map could not be made, which
    // leaves the map in hand where it is and the level to be asked for again. A map
    // that may be given up on says so through allowFailure; every other one is
    // checked as a bug.
    bool CreateImageObjects(uint32_t size, VkImage &image, VkDeviceMemory &memory,
                            VkImageView &view, VkSampler &sampler, bool allowFailure = false);
    void CreateRenderPass();
    void CreateFramebuffer();
    void CreatePipelineLayout();
    void CreatePipelines(const ShaderManager *shaderManager);
    void CreateDescriptors();
    void WriteDescriptors();
    void DestroyAll();

    void DrawGeometry(VkCommandBuffer cmd, const VertexCollector::GeometryDrawInfo &draw);

private:
    // Q2RTX uses SHADOWMAP_SIZE 4096 (shader/constants.h). We used 2048, which
    // quantized the god rays shaft edges to twice the texel size and made the
    // shafts look like coarse "individual rays" instead of smooth beams. That is
    // the resolution of the high quality level, and the one an instance is made
    // with when the host asks for nothing in particular.
    static constexpr uint32_t SHADOW_MAP_SIZE_DEFAULT = 4096;

    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;
    // SetQuality recreates the map the frames in flight are still reading, so it
    // has to be able to wait for them.
    std::shared_ptr<CommandBufferManager> cmdManager;

    uint32_t shadowMapSize = SHADOW_MAP_SIZE_DEFAULT;

    // A level whose map could not be made (out of memory) is not asked for again on
    // every frame -- making and tearing down a map of that size is not free even
    // when the memory for it is missing -- but once in a while, so that memory freed
    // by another setting brings the level up on its own. QUALITY_LEVELS in
    // failedQuality means that no level is waiting to be taken.
    static constexpr uint32_t QUALITY_RETRY_FRAMES = 600;
    uint32_t failedQuality = QUALITY_LEVELS;
    uint32_t qualityRetryAge = 0;

    VkImage       depthImage    = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory  = VK_NULL_HANDLE;
    VkImageView   depthImageView = VK_NULL_HANDLE;
    VkSampler     shadowSampler = VK_NULL_HANDLE;

    VkRenderPass   renderPass   = VK_NULL_HANDLE;
    VkFramebuffer  framebuffer  = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       pipeline       = VK_NULL_HANDLE;

    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool      = VK_NULL_HANDLE;
    VkDescriptorSet       descSet       = VK_NULL_HANDLE;
};

}
