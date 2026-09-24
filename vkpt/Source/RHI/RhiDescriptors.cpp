#include "RhiDescriptors.h"

namespace vkpt::rhi
{

nvrhi::BindingLayoutHandle createTextureTableLayout(nvrhi::IDevice *device,
                                                    uint32_t capacity,
                                                    std::string_view debugName)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    // The validation layer treats maxCapacity 0 as an error (validation-device.cpp:1460-1462);
    // without it the Vulkan backend would create a descriptor array of size 0, which is invalid.
    // Refuse it here so that the caller gets a null layout, like the factories of RhiResources.cpp.
    if (capacity == 0)
    {
        return nullptr;
    }

    // BindlessLayoutDesc has no debugName in the pinned NVRHI (nvrhi.h:2115-2154); the argument
    // documents the caller's intent until the pin is moved, as in RhiPipeline::createBindingLayout.
    (void)debugName;

    nvrhi::BindlessLayoutDesc desc;
    // ShaderType::None would leave the Vulkan bindings with empty stageFlags
    // (vulkan-constants.cpp:166-190); the header explains why every stage sees the table.
    desc.visibility = nvrhi::ShaderType::All;
    desc.maxCapacity = capacity;
    // Immutable: each register space carries one fixed descriptor type (nvrhi.h:2129), which is
    // exactly what the two arrays are. The mutable layout types are for ResourceDescriptorHeap /
    // SamplerDescriptorHeap and need VK_EXT_mutable_descriptor_type
    // (vulkan-resource-bindings.cpp:122-141).
    desc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::Immutable;
    // The order of the spaces defines the Vulkan bindings of the set: textures 0, samplers 1
    // (vulkan-resource-bindings.cpp:145-160). Only the type of each item is used by that backend;
    // the slots are the conventional 0.
    desc.addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0));
    desc.addRegisterSpace(nvrhi::BindingLayoutItem::Sampler(0));
    // 'firstSlot' keeps its default 0: only the D3D12 backend reads it
    // (d3d12-resource-bindings.cpp:649), and this renderer is Vulkan-only.

    // The validation layer also requires register spaces for the Immutable type and rejects them
    // for the mutable ones (validation-device.cpp:1448-1458); the two spaces above satisfy that.
    return device->createBindlessLayout(desc);
}

nvrhi::DescriptorTableHandle createTextureTable(nvrhi::IDevice *device,
                                                nvrhi::IBindingLayout *layout,
                                                std::string_view debugName)
{
    if (device == nullptr || layout == nullptr)
    {
        return nullptr;
    }

    // A table over a regular layout would silently use its first binding as the array
    // (vulkan-resource-bindings.cpp:672-678); the validation layer refuses it
    // (validation-device.cpp:1901-1907), and so does this helper.
    if (layout->getBindlessDesc() == nullptr)
    {
        return nullptr;
    }

    // createDescriptorTable takes only the layout (nvrhi.h:3866) and IDescriptorTable has no
    // debugName (nvrhi.h:2526-2531), so the name cannot be attached; kept for symmetry.
    (void)debugName;

    return device->createDescriptorTable(layout);
}

bool setTexture(nvrhi::IDevice *device,
                nvrhi::IDescriptorTable *table,
                uint32_t slot,
                nvrhi::ITexture *texture)
{
    if (device == nullptr || table == nullptr || texture == nullptr)
    {
        return false;
    }

    // The backend writes binding.slot as the array element of the Vulkan binding whose type
    // matches the item (vulkan-resource-bindings.cpp:922-933 for the type match, :771 for the
    // array element), so the Texture_SRV item below lands in the texture array of the table, not
    // in the sampler one. The default dimension and format mean "the texture's own"
    // (vulkan-texture.cpp:294-298), which is what a table of arbitrary 2D textures needs.
    return device->writeDescriptorTable(table, nvrhi::BindingSetItem::Texture_SRV(slot, texture));
}

bool setSampler(nvrhi::IDevice *device,
                nvrhi::IDescriptorTable *table,
                uint32_t slot,
                nvrhi::ISampler *sampler)
{
    if (device == nullptr || table == nullptr || sampler == nullptr)
    {
        return false;
    }

    // Same path as setTexture: the Sampler item is matched against the sampler register space
    // (vulkan-resource-bindings.cpp:887-899, 922-933).
    return device->writeDescriptorTable(table, nvrhi::BindingSetItem::Sampler(slot, sampler));
}

}
