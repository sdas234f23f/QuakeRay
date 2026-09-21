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

#include "TextureDescriptors.h"
#include "Const.h"

using namespace vkpt;

TextureDescriptors::TextureDescriptors(VkDevice _device, std::shared_ptr<SamplerManager> _samplerManager, uint32_t _maxTextureCount, uint32_t _bindingIndex, uint32_t _samplerBindingIndex) :
    device(_device),
    samplerManager(std::move(_samplerManager)),
    bindingIndex(_bindingIndex),
    samplerBindingIndex(_samplerBindingIndex),
    descPool(VK_NULL_HANDLE),
    descLayout(VK_NULL_HANDLE),
    descSets{},
    emptyTextureImageView(VK_NULL_HANDLE),
    emptyTextureImageLayout(VK_IMAGE_LAYOUT_UNDEFINED),
    currentWriteCount(0)
{
    writeImageInfos.resize(_maxTextureCount);
    writeSamplerInfos.resize(_maxTextureCount);
    writeInfos.resize(_maxTextureCount * 2);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        writeCache[i].resize(_maxTextureCount);
    }

    CreateDescriptors(_maxTextureCount);
}

TextureDescriptors::~TextureDescriptors()
{
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
}

VkDescriptorSet TextureDescriptors::GetDescSet(uint32_t frameIndex) const
{
    return descSets[frameIndex];
}

VkDescriptorSetLayout TextureDescriptors::GetDescSetLayout() const
{
    return descLayout;
}

void TextureDescriptors::SetEmptyTextureInfo(VkImageView view)
{
    emptyTextureImageView = view;
    emptyTextureImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void TextureDescriptors::CreateDescriptors(uint32_t maxTextureCount)
{
    VkDescriptorSetLayoutBinding bindings[2] = {};

    bindings[0].binding = bindingIndex;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = maxTextureCount;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = samplerBindingIndex;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = maxTextureCount;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    VkResult r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Textures Desc set layout");

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[0].descriptorCount = maxTextureCount * MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[1].descriptorCount = maxTextureCount * MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Textures Desc pool");

    VkDescriptorSetAllocateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = descPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &descLayout;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        r = vkAllocateDescriptorSets(device, &setInfo, &descSets[i]);
        VK_CHECKERROR(r);

        SET_DEBUG_NAME(device, descSets[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, "Textures desc set");
    }
}

bool TextureDescriptors::IsCached(uint32_t frameIndex, uint32_t textureIndex, VkImageView view, SamplerManager::Handle samplerHandle)
{
    return writeCache[frameIndex][textureIndex].view == view
        && writeCache[frameIndex][textureIndex].samplerHandle == samplerHandle;
}

void TextureDescriptors::AddToCache(uint32_t frameIndex, uint32_t textureIndex, VkImageView view, SamplerManager::Handle samplerHandle)
{
    writeCache[frameIndex][textureIndex].view = view;
    writeCache[frameIndex][textureIndex].samplerHandle = samplerHandle;
}

void TextureDescriptors::ResetCache(uint32_t frameIndex, uint32_t textureIndex)
{
    writeCache[frameIndex][textureIndex] = { VK_NULL_HANDLE, SamplerManager::Handle() };
}

void vkpt::TextureDescriptors::ResetAllCache(uint32_t frameIndex)
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        for (auto &f : writeCache[i])
        {
            f.view = VK_NULL_HANDLE;
            f.samplerHandle = SamplerManager::Handle();
        }
    }
}

void TextureDescriptors::UpdateTextureDesc(uint32_t frameIndex, uint32_t textureIndex, VkImageView view, SamplerManager::Handle samplerHandle)
{
    assert(view != VK_NULL_HANDLE);

    // don't update if already is set to given parameters
    if (IsCached(frameIndex, textureIndex, view, samplerHandle))
    {
        return;
    }

    if  (currentWriteCount >= writeImageInfos.size())
    {
        // the batch is full: flush it and start a new one. The desc set receives exactly the same
        // writes, they are just split between several vkUpdateDescriptorSets calls.
        FlushDescWrites();
    }

    VkDescriptorImageInfo &imageInfo = writeImageInfos[currentWriteCount];
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo &samplerInfo = writeSamplerInfos[currentWriteCount];
    samplerInfo.sampler = samplerManager->GetSampler(samplerHandle);
    samplerInfo.imageView = VK_NULL_HANDLE;
    samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkWriteDescriptorSet &viewWrite = writeInfos[currentWriteCount * 2 + 0];
    viewWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    viewWrite.dstSet = descSets[frameIndex];
    viewWrite.dstBinding = bindingIndex;
    viewWrite.dstArrayElement = textureIndex;
    viewWrite.descriptorCount = 1;
    viewWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    viewWrite.pImageInfo = &imageInfo;

    VkWriteDescriptorSet &samplerWrite = writeInfos[currentWriteCount * 2 + 1];
    samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    samplerWrite.dstSet = descSets[frameIndex];
    samplerWrite.dstBinding = samplerBindingIndex;
    samplerWrite.dstArrayElement = textureIndex;
    samplerWrite.descriptorCount = 1;
    samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    samplerWrite.pImageInfo = &samplerInfo;

    currentWriteCount++;

    AddToCache(frameIndex, textureIndex, view, samplerHandle);
}

void TextureDescriptors::ResetTextureDesc(uint32_t frameIndex, uint32_t textureIndex)
{
    assert(emptyTextureImageView != VK_NULL_HANDLE &&
           emptyTextureImageLayout != VK_IMAGE_LAYOUT_UNDEFINED);

    // try to update with empty data
    UpdateTextureDesc(frameIndex, textureIndex, 
                      emptyTextureImageView, SamplerManager::Handle(RG_SAMPLER_FILTER_NEAREST, RG_SAMPLER_ADDRESS_MODE_REPEAT, RG_SAMPLER_ADDRESS_MODE_REPEAT, 0));
}

void TextureDescriptors::FlushDescWrites()
{
    vkUpdateDescriptorSets(device, currentWriteCount * 2, writeInfos.data(), 0, nullptr);
    currentWriteCount = 0;
}
