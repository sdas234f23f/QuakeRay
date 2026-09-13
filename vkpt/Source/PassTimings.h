// Copyright (c) 2024
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

#include <array>

#include "Common.h"

namespace vkpt
{

enum GpuPassIndex : uint32_t
{
    GPU_PASS_SETUP = 0,
    GPU_PASS_LIGHTS,
    GPU_PASS_PRIMARY_DECALS,
    GPU_PASS_SHADOW_GODRAYS,
    GPU_PASS_REFL_REFR,
    GPU_PASS_REFL_GODRAYS,
    GPU_PASS_GRADIENT,
    GPU_PASS_DIRECT,
    GPU_PASS_INDIRECT,
    GPU_PASS_DENOISE,
    GPU_PASS_EXPOSURE,
    GPU_PASS_COMPOSITE_RASTER,
    GPU_PASS_UPSCALE,
    GPU_PASS_POST,
    GPU_PASS_PRESENT,

    GPU_PASS_COUNT
};

constexpr uint32_t GPU_MARK_COUNT = GPU_PASS_COUNT + 1;

class PassTimings
{
public:
    PassTimings(VkDevice device, VkPhysicalDevice physDevice, uint32_t queueFamilyIndex);
    ~PassTimings();

    PassTimings(const PassTimings &other) = delete;
    PassTimings(PassTimings &&other) noexcept = delete;
    PassTimings &operator=(const PassTimings &other) = delete;
    PassTimings &operator=(PassTimings &&other) noexcept = delete;

    bool IsSupported() const;

    void BeginFrame(VkCommandBuffer cmd, uint32_t frameIndex, bool enabled);
    void Mark(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t markIndex);

    float GetPassMs(uint32_t passIndex) const;
    float GetTotalMs() const;

private:
    void Fetch(uint32_t frameIndex);

private:
    VkDevice device = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;

    bool supported = false;
    bool recording[MAX_FRAMES_IN_FLIGHT] = {};
    uint32_t marksWritten[MAX_FRAMES_IN_FLIGHT] = {};

    float timestampPeriodNs = 0.0f;
    uint64_t timestampMask = ~uint64_t(0);

    std::array<float, GPU_PASS_COUNT> passMs = {};
    float totalMs = 0.0f;
};

const char *GetGpuPassName(uint32_t passIndex);

}
