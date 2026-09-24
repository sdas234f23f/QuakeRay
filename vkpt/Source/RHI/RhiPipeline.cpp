#include "RhiPipeline.h"

#include <fstream>
#include <vector>

namespace vkpt::rhi
{

nvrhi::ShaderHandle createShaderFromBlob(nvrhi::IDevice *device,
                                         const void *data,
                                         size_t size,
                                         nvrhi::ShaderType type,
                                         std::string_view debugName)
{
    if (device == nullptr || data == nullptr || size == 0)
    {
        return nullptr;
    }

    nvrhi::ShaderDesc desc;
    desc.shaderType = type;
    desc.debugName = std::string(debugName);
    // 'entryName' keeps its NVRHI default, "main" (nvrhi.h:920): the shader build compiles a
    // single "main" entry point, whether the blob came from dxc or from glslc.

    return device->createShader(desc, data, size);
}

nvrhi::ShaderHandle loadShader(nvrhi::IDevice *device,
                               const std::string &blobPath,
                               nvrhi::ShaderType type,
                               std::string_view debugName)
{
    // Binary and positioned at the end, so that tellg gives the size: the same read as
    // NvrhiFrameSkeleton::LoadShader.
    std::ifstream file(blobPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        return nullptr;
    }

    const std::streampos fileSize = file.tellg();
    if (fileSize <= std::streampos(0))
    {
        return nullptr;
    }

    file.seekg(0, std::ios::beg);

    std::vector<char> binary(static_cast<size_t>(fileSize));
    if (!file.read(binary.data(), static_cast<std::streamsize>(fileSize)))
    {
        return nullptr;
    }

    return createShaderFromBlob(device, binary.data(), binary.size(), type, debugName);
}

nvrhi::BindingLayoutHandle createBindingLayout(nvrhi::IDevice *device,
                                               std::span<const nvrhi::BindingLayoutItem> items,
                                               std::string_view debugName,
                                               const nvrhi::VulkanBindingOffsets *pBindingOffsets)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    // BindingLayoutDesc has no debugName in the pinned NVRHI (nvrhi.h:2074-2107); the argument
    // documents the caller's intent until the pin is moved.
    (void)debugName;

    nvrhi::BindingLayoutDesc desc;
    // A pass is one set for now: registerSpace 0 and registerSpaceIsDescriptorSet false put the
    // first regular layout of a pipeline into descriptor set 0 (see the header).
    desc.registerSpace = 0;
    // ShaderType::None would leave the Vulkan bindings with empty stageFlags; the header explains
    // why every item is visible to every stage.
    desc.visibility = nvrhi::ShaderType::All;

    // A null pointer keeps the NVRHI defaults (nvrhi.h:2063-2066); a ported engine pass passes the
    // per-set offsets that reproduce its shader's raw binding numbers, and then uses
    // slot = binding - offset (the header documents the arithmetic).
    if (pBindingOffsets != nullptr)
    {
        desc.setBindingOffsets(*pBindingOffsets);
    }

    for (const nvrhi::BindingLayoutItem &item : items)
    {
        desc.addItem(item);
    }

    return device->createBindingLayout(desc);
}

nvrhi::GraphicsPipelineHandle createGraphicsPipeline(nvrhi::IDevice *device,
                                                     const nvrhi::GraphicsPipelineDesc &desc,
                                                     const nvrhi::FramebufferInfo &framebufferInfo,
                                                     std::string_view debugName)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    // GraphicsPipelineDesc has no debugName in the pinned NVRHI (nvrhi.h:2630-2661), so there is
    // nothing to set and nothing to forward; the desc is used as the caller assembled it.
    (void)debugName;

    return device->createGraphicsPipeline(desc, framebufferInfo);
}

nvrhi::ComputePipelineHandle createComputePipeline(nvrhi::IDevice *device,
                                                   const nvrhi::ComputePipelineDesc &desc,
                                                   std::string_view debugName)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    // ComputePipelineDesc has no debugName either (nvrhi.h:2672-2680); the desc is used as is.
    (void)debugName;

    return device->createComputePipeline(desc);
}

}
