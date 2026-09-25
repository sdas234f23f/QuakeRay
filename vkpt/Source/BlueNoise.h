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

#include "Common.h"
#include "CommandBufferManager.h"
#include "MemoryAllocator.h"
#include "UserFunction.h"

namespace vkpt
{

class BlueNoise
{
public:
    explicit BlueNoise(
        VkDevice device, 
        const char *blueNoiseFilePath,
        std::shared_ptr<MemoryAllocator> allocator,
        const std::shared_ptr<CommandBufferManager> &cmdManager,
        std::shared_ptr<UserFileLoad> userFileLoad);
    ~BlueNoise();

    BlueNoise(const BlueNoise &other) = delete;
    BlueNoise(BlueNoise &&other) noexcept = delete;
    BlueNoise &operator=(const BlueNoise &other) = delete;
    BlueNoise &operator=(BlueNoise &&other) noexcept = delete;

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet() const;

    // The engine handles of the layered blue-noise image the shaders fetch from. The image is
    // created once in the constructor from the KTX2 file, is never recreated, updated or resized,
    // and is left in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (BlueNoise.cpp:157-161), so the
    // handles are valid for the object's lifetime and need no separate release path.
    VkImage GetImage() const;
    VkImageView GetImageView() const;

    // The shape of that image, for a caller that has to describe it without knowing the constants:
    // R8G8B8A8_UNORM, BLUE_NOISE_TEXTURE_SIZE x BLUE_NOISE_TEXTURE_SIZE, one mip and
    // BLUE_NOISE_TEXTURE_COUNT array layers (BlueNoise.cpp:104-119). These are what the RHI's
    // array-texture wrap (rhi::wrapEngineTextureArray) needs besides the two handles.
    VkFormat GetFormat() const;
    VkExtent2D GetExtent() const;
    uint32_t GetLayerCount() const;

private:
    void CreateDescriptors();

private:
    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;

    VkImage blueNoiseImages;
    VkImageView blueNoiseImagesView;

    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
};

}