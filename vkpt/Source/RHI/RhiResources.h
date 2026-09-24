#pragma once

#include <nvrhi/nvrhi.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkpt::rhi
{

// Thin helpers that add the vkpt conventions (debug names, the upload path, the states) on top
// of nvrhi::IDevice, so that later stages create resources in one place only.
//
// The factories copy the desc, set the debug name and forward to the device. A failed creation
// gives a null handle; the caller decides what that means for its stage.

nvrhi::BufferHandle createBuffer(nvrhi::IDevice *device, const nvrhi::BufferDesc &desc, std::string_view debugName);
nvrhi::TextureHandle createTexture(nvrhi::IDevice *device, const nvrhi::TextureDesc &desc, std::string_view debugName);

// nvrhi::SamplerDesc has no debugName field at the pinned revision, so the argument cannot be
// applied; it is kept for symmetry with the other two factories.
nvrhi::SamplerHandle createSampler(nvrhi::IDevice *device, const nvrhi::SamplerDesc &desc, std::string_view debugName);

// Uploads `size` bytes at `offset` into `buffer` through the command list. This is what the
// legacy AutoBuffer did by hand: staging buffer, vkCmdCopyBuffer, barrier. NVRHI owns that path
// inside the command list (on Vulkan: vkCmdUpdateBuffer for writes of 64 KiB or less with a
// 4-byte-aligned offset, an upload-manager staging copy plus vkCmdCopyBuffer otherwise) and
// places the barriers around CopyDest itself.
//
// With automatic barriers on, the state tracker has to know the buffer's prior state: create
// buffers that go through this helper with keepInitialState = true.
void writeBuffer(nvrhi::ICommandList *commandList, nvrhi::IBuffer *buffer, const void *data, size_t size, size_t offset = 0);

// A 1x1 opaque white texture, filled through the given command list; the skeleton samples it so
// that the texture+sampler path is exercised while the image stays what it was.
nvrhi::TextureHandle createWhiteTexture(nvrhi::IDevice *device, nvrhi::ICommandList *commandList, std::string_view debugName);

}
