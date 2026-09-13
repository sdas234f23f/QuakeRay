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

#include "PassTimings.h"

#include <vector>

#include "Utils.h"

namespace vkpt
{

namespace
{
    const char *const passNames[GPU_PASS_COUNT] =
    {
        "setup",
        "lights",
        "primary",
        "shadow",
        "reflrefr",
        "reflgodr",
        "gradient",
        "direct",
        "indirect",
        "denoise",
        "exposure",
        "composite",
        "upscale",
        "post",
        "present",
    };

    constexpr float SMOOTH_FACTOR = 0.9f;
}

PassTimings::PassTimings(VkDevice _device, VkPhysicalDevice physDevice, uint32_t queueFamilyIndex)
    : device(_device)
{
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physDevice, &props);
    timestampPeriodNs = props.limits.timestampPeriod;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &familyCount, nullptr);

    std::vector<VkQueueFamilyProperties> families(familyCount);
    if (familyCount > 0)
    {
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &familyCount, families.data());
    }

    uint32_t validBits = 0;
    if (queueFamilyIndex < familyCount)
    {
        validBits = families[queueFamilyIndex].timestampValidBits;
    }

    if (validBits == 0 || timestampPeriodNs <= 0.0f)
    {
        return;
    }

    timestampMask = validBits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << validBits) - 1);

    VkQueryPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = MAX_FRAMES_IN_FLIGHT * GPU_MARK_COUNT;

    const VkResult r = vkCreateQueryPool(device, &poolInfo, nullptr, &queryPool);
    if (r != VK_SUCCESS)
    {
        return;
    }

    SET_DEBUG_NAME(device, queryPool, VK_OBJECT_TYPE_QUERY_POOL, "GPU pass timings");
    supported = true;
}

PassTimings::~PassTimings()
{
    if (queryPool != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device, queryPool, nullptr);
    }
}

bool PassTimings::IsSupported() const
{
    return supported;
}

void PassTimings::BeginFrame(VkCommandBuffer cmd, uint32_t frameIndex, bool enabled)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);

    if (recording[frameIndex])
    {
        Fetch(frameIndex);
    }

    recording[frameIndex] = enabled && supported;
    marksWritten[frameIndex] = 0;

    if (!recording[frameIndex])
    {
        passMs.fill(0.0f);
        totalMs = 0.0f;
        return;
    }

    vkCmdResetQueryPool(cmd, queryPool, frameIndex * GPU_MARK_COUNT, GPU_MARK_COUNT);
}

void PassTimings::Mark(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t markIndex)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);
    assert(markIndex < GPU_MARK_COUNT);

    if (!recording[frameIndex])
    {
        return;
    }

    marksWritten[frameIndex]++;

    vkCmdWriteTimestamp(cmd,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        queryPool,
                        frameIndex * GPU_MARK_COUNT + markIndex);
}

void PassTimings::Fetch(uint32_t frameIndex)
{
    if (marksWritten[frameIndex] != GPU_MARK_COUNT)
    {
        return;
    }

    uint64_t timestamps[GPU_MARK_COUNT] = {};

    const VkResult r = vkGetQueryPoolResults(device,
                                             queryPool,
                                             frameIndex * GPU_MARK_COUNT,
                                             GPU_MARK_COUNT,
                                             sizeof(timestamps),
                                             timestamps,
                                             sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS)
    {
        return;
    }

    float total = 0.0f;

    for (uint32_t i = 0; i < GPU_PASS_COUNT; i++)
    {
        const uint64_t begin = timestamps[i] & timestampMask;
        const uint64_t end = timestamps[i + 1] & timestampMask;
        const uint64_t ticks = (end - begin) & timestampMask;

        const float ms = static_cast<float>(static_cast<double>(ticks) * static_cast<double>(timestampPeriodNs) * 1e-6);

        passMs[i] = passMs[i] <= 0.0f ? ms : passMs[i] * SMOOTH_FACTOR + ms * (1.0f - SMOOTH_FACTOR);
        total += ms;
    }

    totalMs = totalMs <= 0.0f ? total : totalMs * SMOOTH_FACTOR + total * (1.0f - SMOOTH_FACTOR);
}

float PassTimings::GetPassMs(uint32_t passIndex) const
{
    assert(passIndex < GPU_PASS_COUNT);
    return passMs[passIndex];
}

float PassTimings::GetTotalMs() const
{
    return totalMs;
}

const char *GetGpuPassName(uint32_t passIndex)
{
    if (passIndex >= GPU_PASS_COUNT)
    {
        return "";
    }
    return passNames[passIndex];
}

}
