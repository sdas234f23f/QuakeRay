#pragma once

#include <nvrhi/nvrhi.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace vkpt::rhi
{

// Shader, binding layout and pipeline creation for the RHI layer.
//
// These helpers keep the conventions that every pass of the NVRHI renderer has to follow in one
// place: how a compiled blob becomes an nvrhi::IShader, how a pass declares its bindings and how
// a graphics or compute pipeline is created. NvrhiFrameSkeleton is their first user; the A2
// passes replace the legacy ShaderManager and the hand-written Vulkan pipelines with them.

// Loads one of the compiled `<name>.<stage>.spv` blobs that the shader build produces - the same
// blobs the legacy ShaderManager reads. 'blobPath' is the full path, folder included (the
// NvrhiFrameSkeleton keeps the folder in its own member and only the caller knows it).
//
// Returns a null handle if the file cannot be opened, is empty, cannot be read, or if NVRHI
// rejects the blob, and does nothing else about it: the caller reports the failure and marks the
// pass unavailable, exactly as NvrhiFrameSkeleton::LoadShader does after warning through its
// print callback. This helper has no callback of its own, so it stays silent.
nvrhi::ShaderHandle loadShader(nvrhi::IDevice *device,
                               const std::string &blobPath,
                               nvrhi::ShaderType type,
                               std::string_view debugName);

// Same as loadShader for a blob that is already in memory. Returns a null handle if 'data' is
// null or 'size' is zero, so that an empty read cannot reach vkCreateShaderModule.
nvrhi::ShaderHandle createShaderFromBlob(nvrhi::IDevice *device,
                                         const void *data,
                                         size_t size,
                                         nvrhi::ShaderType type,
                                         std::string_view debugName);

// Creates the set-0 binding layout of a pass from the items it declares.
//
// Callers pass NVRHI slots, not SPIR-V bindings. On the Vulkan backend with the default
// nvrhi::VulkanBindingOffsets (nvrhi.h:2063-2066: shaderResource 0, sampler 128, constantBuffer
// 256, unorderedAccess 384) an item's binding number is `binding = registerOffset + slot`
// (vulkan-resource-bindings.cpp:48-78 and 98-100):
//
//     shader resource (Texture/TypedBuffer/StructuredBuffer/RawBuffer SRV,
//                      RayTracingAccelStruct)         slot N -> binding N
//     sampler                                        slot N -> binding N + 128
//     constant buffer (ConstantBuffer, VolatileConstantBuffer)  slot N -> binding N + 256
//     unordered access (Texture/TypedBuffer/StructuredBuffer/RawBuffer UAV)  slot N -> binding N + 384
//
// An HLSL shader must spell the resulting binding with [[vk::binding(...)]], e.g. a constant
// buffer at slot 2 is [[vk::binding(258)]]. Push constants do not become descriptors at all
// (vulkan-resource-bindings.cpp:90-94).
//
// 'pBindingOffsets' replaces the defaults for this one layout. The engine's own shaders do not
// follow the DXC numbering above - the global uniform sits at raw binding 0, the framebuffer
// samplers at 248..371 and the framebuffer storage images at 0..123 - so a ported engine pass
// chooses offsets per set. The offsets belong to the layout (vulkan-resource-bindings.cpp:98-100
// builds the VkDescriptorSetLayout from this desc's own offsets, and :394 uses the same layout's
// offsets again when it turns a BindingSetItem into a descriptor write), and the rule above
// becomes, with the supplied offsets:
//
//     binding = offset + slot    <=>    slot = binding - offset, offset from 'pBindingOffsets'
//
// i.e. the caller picks the offset for each kind so that `offset + slot` equals the raw number the
// shader spells with [[vk::binding(...)]], and passes that slot - the shader's binding minus the
// offset - both in the layout items given here and in the BindingSetItem writes of the matching
// set. Example: a framebuffers layout with shaderResource 124, unorderedAccess 0 and sampler 248
// turns shader binding 25 (a UAV) into Texture_UAV(slot 25) and shader binding 371 (a sampler)
// into Sampler(slot 123). A PushConstants item carries no descriptor, so no offset applies to it.
//
// 'pBindingOffsets' is optional and may be null: the desc then keeps the nvrhi::VulkanBindingOffsets
// defaults (nvrhi.h:2063-2066) exactly as if the parameter did not exist, so a caller that follows
// the DXC numbering passes nothing.
//
// The layout is created with ShaderType::All visibility: this helper has no visibility argument,
// and the NVRHI default, ShaderType::None, would make every Vulkan binding carry empty stageFlags
// (vulkan-constants.cpp:166-190), which is invalid.
//
// The layout is bound as descriptor set 0: 'registerSpace' stays 0 and
// 'registerSpaceIsDescriptorSet' stays false, so the Vulkan backend places the regular layouts of
// a pipeline into descriptor sets in the order they are listed (vulkan-resource-bindings.cpp:
// 1090-1099).
//
// 'debugName' cannot be attached: the pinned NVRHI has no debugName in BindingLayoutDesc
// (nvrhi.h:2074-2107). The argument stays in the signature for a future NVRHI bump.
nvrhi::BindingLayoutHandle createBindingLayout(nvrhi::IDevice *device,
                                               std::span<const nvrhi::BindingLayoutItem> items,
                                               std::string_view debugName,
                                               const nvrhi::VulkanBindingOffsets *pBindingOffsets = nullptr);

// Creates a graphics pipeline for the framebuffer described by 'framebufferInfo' (the
// FramebufferInfo overload of createGraphicsPipeline, nvrhi.h:3848).
//
// 'debugName' cannot be attached: GraphicsPipelineDesc has no debugName (nvrhi.h:2630-2661) and
// the pinned Vulkan backend does not name pipeline objects (only shaders, buffers, textures,
// heaps and acceleration structures are named). The assembly of the desc - shaders, render state,
// input layout, binding layouts - is entirely the caller's, as in
// NvrhiFrameSkeleton::CreatePipeline.
nvrhi::GraphicsPipelineHandle createGraphicsPipeline(nvrhi::IDevice *device,
                                                     const nvrhi::GraphicsPipelineDesc &desc,
                                                     const nvrhi::FramebufferInfo &framebufferInfo,
                                                     std::string_view debugName);

// Creates a compute pipeline (nvrhi.h:3853). 'debugName' cannot be attached, for the same reason
// as for the graphics pipeline: ComputePipelineDesc has no debugName (nvrhi.h:2672-2680).
nvrhi::ComputePipelineHandle createComputePipeline(nvrhi::IDevice *device,
                                                   const nvrhi::ComputePipelineDesc &desc,
                                                   std::string_view debugName);

}
