#pragma once

#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <string_view>

namespace vkpt::rhi
{

// The bindless texture table: an unbounded array of textures plus an unbounded array of samplers -
// the NVRHI shape of the engine's split table (TextureDescriptors). One layout and one table hold
// both arrays, so a shader can pick a texture and its sampler by index out of the same set.
//
// Two questions a caller has to answer before using the table, both measured against the pinned
// NVRHI (third_party/nvrhi, c8d34b4f):
//
// 1) Which set and which bindings the table ends up at. For a pipeline whose regular layout is
//    added first and this layout after it (NvrhiFrameSkeleton::CreatePipeline is that shape today),
//    the table is descriptor set 1, textures are binding 0 and samplers are binding 1.
//    - The Vulkan backend in the legacy mode (registerSpaceIsDescriptorSet == false, the mode
//      RhiPipeline::createBindingLayout uses) pushes the layouts of a pipeline into the descriptor
//      set list in the order the pipeline added them (vulkan-resource-bindings.cpp:1090-1099), so
//      the regular layout takes set 0, and the bindless layout that follows takes set 1.
//    - Inside the set, one Vulkan binding is made per register space, numbered by the position of
//      the space in the desc, starting at 0 (vulkan-resource-bindings.cpp:119, 145-160), so the
//      first space - textures - is binding 0 and the second - samplers - is binding 1. The
//      BindingLayoutItem slot inside a register space is not read by this backend at all: only the
//      item's type and the desc's maxCapacity are (vulkan-resource-bindings.cpp:120, 147). Slot 0
//      below is therefore a conventional value, not one the backend acts on.
//    - Volatile constant buffers cannot be put into a bindless layout
//      (vulkan-resource-bindings.cpp:149-150), so this table carries nothing but the two arrays.
//
// 2) How the table is bound in a command list. There is no setDescriptorTable on
//    nvrhi::ICommandList at the pinned revision; a descriptor table is an nvrhi::IBindingSet
//    (nvrhi.h:2525-2526), so it is appended to the binding array of the state - after the regular
//    binding set - through GraphicsState::addBindingSet (nvrhi.h:2795) or
//    ComputeState::addBindingSet (nvrhi.h:2854), and the state is submitted as usual
//    (ICommandList::setGraphicsState, nvrhi.h:3450; setComputeState, nvrhi.h:3503). The array of
//    binding sets is bound to descriptor sets in order (vulkan-resource-bindings.cpp:940-1019),
//    which is what puts the table at set 1 when the pipeline declared the layouts in the same
//    order.
//
// The layout is created with ShaderType::All visibility: ShaderType::None, the NVRHI default,
// would leave the Vulkan bindings with empty stageFlags (vulkan-constants.cpp:166-190), which is
// invalid. RhiPipeline::createBindingLayout follows the same rule.

// Creates the bindless layout of the table: two register spaces - textures and samplers, each an
// unbounded array of 'capacity' descriptors - visible to every stage.
//
// 'capacity' is the number of slots the table will hold and has to be at least 1: the validation
// layer rejects maxCapacity 0 (validation-device.cpp:1460-1462). The layout is null when 'device'
// is null or 'capacity' is 0.
//
// 'debugName' cannot be attached: BindlessLayoutDesc has no debugName field at the pinned revision
// (nvrhi.h:2115-2154). The argument stays in the signature for a future NVRHI bump, as in
// RhiPipeline::createBindingLayout.
//
// The return type is nvrhi::BindingLayoutHandle: the pinned revision has no BindlessLayoutHandle
// alias, and createBindlessLayout returns the same handle as createBindingLayout (nvrhi.h:3863,
// 2163).
nvrhi::BindingLayoutHandle createTextureTableLayout(nvrhi::IDevice *device,
                                                    uint32_t capacity,
                                                    std::string_view debugName);

// Creates the table for a layout from createTextureTableLayout. Returns null when 'device' or
// 'layout' is null, and when 'layout' is not a bindless layout - the validation layer rejects that
// case as well (validation-device.cpp:1901-1907).
//
// 'debugName' cannot be attached: createDescriptorTable takes only the layout (nvrhi.h:3866) and
// IDescriptorTable has no debugName (nvrhi.h:2526-2531). The argument stays for symmetry.
nvrhi::DescriptorTableHandle createTextureTable(nvrhi::IDevice *device,
                                                nvrhi::IBindingLayout *layout,
                                                std::string_view debugName);

// Writes one texture (setTexture) or one sampler (setSampler) into the given slot of the table;
// 'slot' is the array element that the shader indexes. The item's type selects the array: the
// backend matches it against the register spaces of the layout (vulkan-resource-bindings.cpp:
// 922-933) and writes slot as the array element of that Vulkan binding (:771).
//
// The pinned revision has no writer on nvrhi::IDescriptorTable (nvrhi.h:2526-2531) and no way to
// reach a device from a resource (common/resource.h:105-125), so the write goes through
// IDevice::writeDescriptorTable (nvrhi.h:3869) and the caller has to pass the device that created
// the table.
//
// Returns what IDevice::writeDescriptorTable returns: false when 'slot' is not below the table's
// capacity (vulkan-resource-bindings.cpp:745-746) or when the validation layer rejects the item. A
// null argument also returns false, instead of a null write that the backend would follow into a
// dereference (vulkan-resource-bindings.cpp:784-806).
bool setTexture(nvrhi::IDevice *device,
                nvrhi::IDescriptorTable *table,
                uint32_t slot,
                nvrhi::ITexture *texture);
bool setSampler(nvrhi::IDevice *device,
                nvrhi::IDescriptorTable *table,
                uint32_t slot,
                nvrhi::ISampler *sampler);

}
