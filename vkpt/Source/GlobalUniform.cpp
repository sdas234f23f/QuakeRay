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
    descSetLayout(VK_NULL_HANDLE)
{
    uniformData = std::make_shared<ShGlobalUniform>();

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++)
    {
        uniformBuffer[frame] = std::make_shared<AutoBuffer>(_device, _allocator);
        uniformBuffer[frame]->Create(sizeof(ShGlobalUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "Uniform buffer");
    }

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
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    VkDescriptorSetLayout setLayouts[MAX_FRAMES_IN_FLIGHT] = {};
    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++)
    {
        setLayouts[frame] = descSetLayout;
    }
    allocInfo.pSetLayouts = setLayouts;

    r = vkAllocateDescriptorSets(device, &allocInfo, descSet);
    VK_CHECKERROR(r);

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++)
    {
        char name[64];
        snprintf(name, sizeof(name), "Uniform Desc set %u", frame);
        SET_DEBUG_NAME(device, descSet[frame], VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
    }

    // bind buffers to sets once, each frame's set naming its own buffer
    VkDescriptorBufferInfo bufInfo[MAX_FRAMES_IN_FLIGHT] = {};
    VkWriteDescriptorSet wrt[MAX_FRAMES_IN_FLIGHT] = {};

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++)
    {
        bufInfo[frame].buffer = uniformBuffer[frame]->GetDeviceLocal();
        bufInfo[frame].offset = 0;
        bufInfo[frame].range = VK_WHOLE_SIZE;

        wrt[frame].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt[frame].dstSet = descSet[frame];
        wrt[frame].dstBinding = BINDING_GLOBAL_UNIFORM;
        wrt[frame].dstArrayElement = 0;
        wrt[frame].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wrt[frame].descriptorCount = 1;
        wrt[frame].pBufferInfo = &bufInfo[frame];
    }

    vkUpdateDescriptorSets(device, MAX_FRAMES_IN_FLIGHT, wrt, 0, nullptr);
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
    uniformBuffer[frameIndex]->CopyFromStaging(cmd, frameIndex, sizeof(ShGlobalUniform));
}

ShGlobalUniform *GlobalUniform::GetData()
{
    return uniformData.get();
}

const ShGlobalUniform *GlobalUniform::GetData() const
{
    return uniformData.get();
}

VkDescriptorSet GlobalUniform::GetDescSet(uint32_t frameIndex) const
{
    return descSet[frameIndex];
}

VkDescriptorSetLayout GlobalUniform::GetDescSetLayout() const
{
    return descSetLayout;
}

void GlobalUniform::SetData(uint32_t frameIndex, const void *data, VkDeviceSize dataSize)
{
    assert(frameIndex >= 0 && frameIndex < MAX_FRAMES_IN_FLIGHT);
    assert(uniformBuffer[frameIndex]->GetSize() <= dataSize);

    void *mapped = uniformBuffer[frameIndex]->GetMapped(frameIndex);
    memcpy(mapped, data, dataSize);
}

