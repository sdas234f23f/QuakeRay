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
              const std::shared_ptr<const ShaderManager> &shaderManager);
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

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet() const;

    void OnShaderReload(const ShaderManager *shaderManager);

private:
    void CreateImage();
    void CreateRenderPass();
    void CreatePipelineLayout();
    void CreatePipelines(const ShaderManager *shaderManager);
    void CreateDescriptors();
    void DestroyAll();

    void DrawGeometry(VkCommandBuffer cmd, const VertexCollector::GeometryDrawInfo &draw);

private:
    // Q2RTX uses SHADOWMAP_SIZE 4096 (shader/constants.h). We used 2048, which
    // quantized the god rays shaft edges to twice the texel size and made the
    // shafts look like coarse "individual rays" instead of smooth beams.
    static constexpr uint32_t SHADOW_MAP_SIZE = 4096;

    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;

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
