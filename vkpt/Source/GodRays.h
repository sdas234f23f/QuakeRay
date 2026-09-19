// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of god_rays.c from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
#include "Buffer.h"
#include "IShaderDependency.h"
#include "MemoryAllocator.h"
#include "ShaderManager.h"

namespace vkpt
{

class Framebuffers;
class GlobalUniform;
class BlueNoise;
class ShadowMap;

// Volumetric sunlight ("god rays") — ray marches through the shadow map.
// Ported from Quake 2 RTX (god_rays.c / god_rays.comp), GPL v2.
class GodRays : public IShaderDependency
{
public:
    struct Params
    {
        float sunDirection[4];
        float sunColor[4];
        float worldCenter[4];
        float worldHalfSizeInv[4];
        float shadowMapVP[16];
        float shadowMapDepthScale;
        float godRaysIntensity;
        float godRaysEccentricity;
        uint32_t godRaysEnabled;
        float _padding;
    };

public:
    GodRays(VkDevice device, std::shared_ptr<MemoryAllocator> &allocator,
            const std::shared_ptr<Framebuffers> &framebuffers,
            const std::shared_ptr<const ShaderManager> &shaderManager,
            const std::shared_ptr<const GlobalUniform> &uniform,
            const std::shared_ptr<const BlueNoise> &blueNoise,
            const std::shared_ptr<const ShadowMap> &shadowMap);
    ~GodRays();

    GodRays(const GodRays &other) = delete;
    GodRays(GodRays &&other) noexcept = delete;
    GodRays &operator=(const GodRays &other) = delete;
    GodRays &operator=(GodRays &&other) noexcept = delete;

    // passIndex: 0 = primary rays, 1 = reflection/refraction rays (the
    // reflected segment god rays are accumulated on top of the primary result).
    void Trace(VkCommandBuffer cmd, uint32_t frameIndex,
               const Params &params, uint32_t passIndex = 0);

    // Bilateral upscale of the half-resolution result to full resolution
    // (Q2RTX god_rays_filter.comp). Call after all Trace passes.
    void Filter(VkCommandBuffer cmd, uint32_t frameIndex);

    void OnShaderReload(const ShaderManager *shaderManager);

private:
    void CreateParamsBuffer();
    void CreateDescriptors();
    void CreatePipelineLayout();
    void CreatePipelines(const ShaderManager *shaderManager);
    void DestroyPipelines();

private:
    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;
    std::shared_ptr<Framebuffers> framebuffers;
    std::shared_ptr<const GlobalUniform> uniform;
    std::shared_ptr<const BlueNoise> blueNoise;
    std::shared_ptr<const ShadowMap> shadowMap;

    // One params buffer per frame in flight: the host writes the params for
    // frame N while the GPU may still be reading frame N-1's copy.
    Buffer paramsBuffer[MAX_FRAMES_IN_FLIGHT];
    void *mappedParams[MAX_FRAMES_IN_FLIGHT] = {};

    VkDescriptorSetLayout paramsDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      paramsDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       paramsDescSet[MAX_FRAMES_IN_FLIGHT] = {};

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       tracePipeline  = VK_NULL_HANDLE;
    VkPipeline       filterPipeline = VK_NULL_HANDLE;
};

}
