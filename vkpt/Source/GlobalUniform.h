// Copyright (c) 2020-2021 Sultim Tsyrendashiev
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

#include "AutoBuffer.h"

namespace vkpt
{

struct ShGlobalUniform;

class GlobalUniform
{
public:
    explicit GlobalUniform(VkDevice device, std::shared_ptr<MemoryAllocator> &allocator);
    ~GlobalUniform();

    GlobalUniform(const GlobalUniform &other) = delete;
    GlobalUniform(GlobalUniform &&other) noexcept = delete;
    GlobalUniform &operator=(const GlobalUniform &other) = delete;
    GlobalUniform &operator=(GlobalUniform &&other) noexcept = delete;

    // Send current data
    void Upload(VkCommandBuffer cmd, uint32_t frameIndex);

    // Getters for modifying uniform buffer data that will be uploaded
    ShGlobalUniform *GetData();
    const ShGlobalUniform *GetData() const;

    VkDescriptorSet GetDescSet(uint32_t frameIndex) const;
    VkDescriptorSetLayout GetDescSetLayout() const;

private:
    void CreateDescriptors();
    void SetData(uint32_t frameIndex, const void *data, VkDeviceSize dataSize);

private:
    VkDevice device;

    std::shared_ptr<ShGlobalUniform> uniformData;

    // One buffer and one descriptor set per frame in flight: the frames behind this
    // one may still be reading the camera, the sun and the place the cloud layer's
    // shadow volume stands in theirs while this one is written, and a single copy
    // showed as a frame drawn with another frame's values.
    std::shared_ptr<AutoBuffer> uniformBuffer[MAX_FRAMES_IN_FLIGHT];

    VkDescriptorPool        descPool;
    VkDescriptorSetLayout   descSetLayout;
    VkDescriptorSet         descSet[MAX_FRAMES_IN_FLIGHT];
};

}