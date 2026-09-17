// Copyright (c) 2026 QuakeRay contributors
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

namespace vkpt
{

constexpr uint32_t RAY_STATS_CATEGORY_COUNT = 4;

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
