#include "RhiResources.h"

#include <string>

namespace vkpt::rhi
{

nvrhi::BufferHandle createBuffer(nvrhi::IDevice *device, const nvrhi::BufferDesc &desc, std::string_view debugName)
{
    nvrhi::BufferDesc namedDesc = desc;
    namedDesc.debugName = std::string(debugName);

    return device->createBuffer(namedDesc);
}

nvrhi::TextureHandle createTexture(nvrhi::IDevice *device, const nvrhi::TextureDesc &desc, std::string_view debugName)
{
    nvrhi::TextureDesc namedDesc = desc;
    namedDesc.debugName = std::string(debugName);

    return device->createTexture(namedDesc);
}

nvrhi::SamplerHandle createSampler(nvrhi::IDevice *device, const nvrhi::SamplerDesc &desc, std::string_view debugName)
{
    // nvrhi::SamplerDesc has no debugName at the pinned revision, so the name is not applied.
    (void)debugName;

    return device->createSampler(desc);
}

void writeBuffer(nvrhi::ICommandList *commandList, nvrhi::IBuffer *buffer, const void *data, size_t size, size_t offset)
{
    commandList->writeBuffer(buffer, data, size, uint64_t(offset));
}

nvrhi::TextureHandle createWhiteTexture(nvrhi::IDevice *device, nvrhi::ICommandList *commandList, std::string_view debugName)
{
    nvrhi::TextureDesc desc;
    desc.width = 1;
    desc.height = 1;
    desc.format = nvrhi::Format::RGBA8_UNORM;
    desc.isShaderResource = true;

    // The texture rests in ShaderResource, and that is what initialState has to say. It is not
    // CopyDest: NVRHI's writeTexture moves the texture to CopyDest by itself (it places the
    // undefined -> TransferDst barrier as the first use), while initialState is the state that
    // keepInitialState restores at the end of every command list. With CopyDest there, the
    // setTextureState(ShaderResource) below would be undone on close and every sampling pass
    // would have to transition the texture again. With ShaderResource there, the tracker starts
    // every later command list already knowing the texture is samplable, so no pass needs a
    // beginTrackingTextureState call and no state change is emitted per frame.
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;

    nvrhi::TextureHandle texture = createTexture(device, desc, debugName);
    if (texture == nullptr)
    {
        return nullptr;
    }

    const uint32_t white = uint32_t(0xFFFFFFFF);
    commandList->writeTexture(texture, 0, 0, &white, sizeof(white));

    // writeTexture only guarantees CopyDest; this turns the texture into a shader resource now
    // (a CopyDest -> ShaderResource barrier goes into this command list) and, because the
    // resting state above equals it, the close of the command list leaves it there.
    commandList->setTextureState(texture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);

    return texture;
}

}
