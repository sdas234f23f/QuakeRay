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

#include "GlobalUniform.h"

#include "Generated/ShaderCommonC.h"
#include "CmdLabel.h"
#include <cstring>

using namespace vkpt;

GlobalUniform::GlobalUniform(VkDevice _device, std::shared_ptr<MemoryAllocator> &_allocator)
:
    device(_device),
    descPool(VK_NULL_HANDLE),
    descSetLayout(VK_NULL_HANDLE),
    descSet(VK_NULL_HANDLE)
{
    uniformData = std::make_shared<ShGlobalUniform>();

    uniformBuffer = std::make_shared<AutoBuffer>(_device, _allocator);
    // The RHI layer wraps this buffer through a native handle, and NVRHI queries the buffer device
    // address on every wrap - vulkan-buffer.cpp:215-220 - so the usage bit is mandatory there. The
    // memory is address-capable regardless: Buffer::Init always allocates through
    // AllocType::WITH_ADDRESS_QUERY (Buffer.cpp:67, MemoryAllocator.cpp:269-277). Same reason the
    // collector's geometry buffers carry it (RasterizedDataCollector.cpp:72-81).
    uniformBuffer->Create(sizeof(ShGlobalUniform),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          "Uniform buffer");

    CreateDescriptors();
}

void GlobalUniform::CreateDescriptors()
{
    VkResult r;

    VkDescriptorSetLayoutBinding uniformBinding = {};
    uniformBinding.binding = BINDING_GLOBAL_UNIFORM;
    uniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBinding.descriptorCount = 1;
    uniformBinding.stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uniformBinding;

    r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Uniform Desc set layout");

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Uniform Desc pool");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descSetLayout;

    r = vkAllocateDescriptorSets(device, &allocInfo, &descSet);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, "Uniform Desc set");

    // bind buffers to sets once
    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = uniformBuffer->GetDeviceLocal();
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet wrt = {};
    wrt.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrt.dstSet = descSet;
    wrt.dstBinding = BINDING_GLOBAL_UNIFORM;
    wrt.dstArrayElement = 0;
    wrt.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wrt.descriptorCount = 1;
    wrt.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(device, 1, &wrt, 0, nullptr);
}

GlobalUniform::~GlobalUniform()
{
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
}

void GlobalUniform::Upload(VkCommandBuffer cmd, uint32_t frameIndex)
{
    CmdLabel label(cmd, "Copying uniform");

    SetData(frameIndex, uniformData.get(), sizeof(ShGlobalUniform));
    uniformBuffer->CopyFromStaging(cmd, frameIndex, sizeof(ShGlobalUniform));
}

ShGlobalUniform *GlobalUniform::GetData()
{
    return uniformData.get();
}

const ShGlobalUniform *GlobalUniform::GetData() const
{
    return uniformData.get();
}

VkBuffer GlobalUniform::GetBuffer() const
{
    return uniformBuffer->GetDeviceLocal();
}

VkDescriptorSet GlobalUniform::GetDescSet(uint32_t frameIndex) const
{
    return descSet;
}

VkDescriptorSetLayout GlobalUniform::GetDescSetLayout() const
{
    return descSetLayout;
}

void GlobalUniform::SetData(uint32_t frameIndex, const void *data, VkDeviceSize dataSize)
{
    assert(frameIndex >= 0 && frameIndex < MAX_FRAMES_IN_FLIGHT);
    assert(uniformBuffer->GetSize() <= dataSize);

    void *mapped = uniformBuffer->GetMapped(frameIndex);
    memcpy(mapped, data, dataSize);
}

