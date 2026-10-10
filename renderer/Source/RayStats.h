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

#include <array>

#include "Common.h"
#include "Buffer.h"
#include "MemoryAllocator.h"

// The generated shader common header defines the same name as a macro. Drop it here so that the
// declaration below is never rewritten when that header is included first.
#ifdef RAY_STATS_CATEGORY_COUNT
#undef RAY_STATS_CATEGORY_COUNT
#endif

namespace qray
{

constexpr uint32_t RAY_STATS_CATEGORY_COUNT = 6;

constexpr uint32_t RAY_STATS_CATEGORY_REFLECTION_REFRACTION = 1;

constexpr uint32_t RAY_STATS_CATEGORY_PARTICLE = 5;

class RayStats
{
public:
    RayStats(VkDevice device, std::shared_ptr<MemoryAllocator> &allocator);
    ~RayStats();

    RayStats(const RayStats &other) = delete;
    RayStats(RayStats &&other) noexcept = delete;
    RayStats &operator=(const RayStats &other) = delete;
    RayStats &operator=(RayStats &&other) noexcept = delete;

    void Reset(uint32_t frameIndex);
    uint32_t GetRays(uint32_t frameIndex) const;
    void GetRaysPerCategory(uint32_t frameIndex, uint32_t *pOutCounts) const;

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet(uint32_t frameIndex) const;

    // The slot's raw buffer. Additive accessor for consumers that bind the buffer through their own
    // API instead of this class's descriptor set: the RHI layer wraps each slot's buffer as the
    // ray-stats structured UAV of the RT passes, so the counters the raygens accumulate are the
    // very ones GetRays/GetRaysPerCategory read back and Reset clears.
    VkBuffer GetBuffer(uint32_t frameIndex) const;

private:
    void CreateBuffers(std::shared_ptr<MemoryAllocator> &allocator);
    void CreateDescSetLayout();
    void CreateDescSet();

private:
    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;

    std::array<Buffer, MAX_FRAMES_IN_FLIGHT> buffers;
    std::array<void *, MAX_FRAMES_IN_FLIGHT> mapped;

    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSets[MAX_FRAMES_IN_FLIGHT] = {};
};

}
