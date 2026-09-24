#include "RhiTextureSource.h"

#include <nvrhi/vulkan.h>

#include <string>

namespace vkpt::rhi
{

namespace
{

// The pinned NVRHI converts formats in one direction only: nvrhi::vulkan::convertFormat maps
// nvrhi::Format -> VkFormat (declared in vulkan.h:90, implemented over the c_FormatMap table in
// vulkan-constants.cpp:39-119) and the pinned tree has no reverse helper - the only other
// convertFormat overloads are the D3D11/D3D12 ones (d3d11.h:49, d3d12.h:146). So the bridge
// inverts that exact table by search, which also means the result cannot drift from the mapping
// the Vulkan backend itself uses to make image views (vulkan-texture.cpp:317-332).
//
// Consequences of the inverse search, both harmless for sampled engine textures:
//  - VK_FORMAT_UNDEFINED matches Format::UNKNOWN at index 0 (vulkan-constants.cpp:40), so an
//    undefined engine format is reported as "cannot wrap".
//  - Formats that NVRHI maps only for depth/stencil share one VkFormat with a sibling
//    (D24S8/X24G8_UINT and D32S8/X32G8_UINT, vulkan-constants.cpp:91-95); the first entry wins.
nvrhi::Format convertVkFormat(VkFormat format)
{
    for (uint32_t i = 0; i < uint32_t(nvrhi::Format::COUNT); i++)
    {
        if (nvrhi::vulkan::convertFormat(nvrhi::Format(i)) == format)
        {
            return nvrhi::Format(i);
        }
    }

    return nvrhi::Format::UNKNOWN;
}

}

nvrhi::TextureHandle wrapEngineTexture(nvrhi::IDevice *device,
                                       uint64_t vkImage,
                                       uint32_t vkFormat,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t mipLevels,
                                       std::string_view debugName)
{
    if (device == nullptr || vkImage == 0 || width == 0 || height == 0)
    {
        return nullptr;
    }

    const nvrhi::Format format = convertVkFormat(VkFormat(vkFormat));
    if (format == nvrhi::Format::UNKNOWN)
    {
        return nullptr;
    }

    nvrhi::TextureDesc desc;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.format = format;
    desc.width = width;
    desc.height = height;
    // The caller passes the image's level count; arraySize still exposes a single layer only (the
    // header documents both).
    desc.mipLevels = mipLevels;
    desc.arraySize = 1;
    desc.sampleCount = 1;

    desc.isShaderResource = true;
    desc.isRenderTarget = false;
    desc.isUAV = false;

    // The engine leaves the image in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (TextureUploader.cpp:
    // 375-380 and :364-369, expected by the update path at :275-277); that is ShaderResource in
    // NVRHI terms (vulkan-constants.cpp:234-241). keepInitialState makes every command list start
    // the image in that state and return it there on close (state-tracking.cpp:350-361), so a pass
    // that binds the handle does not transition the image and cannot disturb the legacy renderer.
    // The same pair is what rhi::createWhiteTexture uses for its resting texture
    // (RhiResources.cpp:53-54); TextureDesc::enableAutomaticStateTracking(ShaderResource)
    // (nvrhi.h:553-559) is the equivalent spelling.
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;

    desc.debugName = std::string(debugName);

    // The native image travels as an nvrhi::Object, exactly as in
    // NvrhiFrameSkeleton::CreateSwapchainResources (NvrhiFrameSkeleton.cpp:398-401).
    const nvrhi::Object nativeImage(vkImage);

    // The backend rejects a null Object and any object type other than VK_Image
    // (vulkan-texture.cpp:805-811); the desc above is stored as-is on the wrapper.
    return device->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, nativeImage, desc);
}

nvrhi::TextureHandle wrapEngineRenderTarget(nvrhi::IDevice *device,
                                           uint64_t image,
                                           uint64_t imageView,
                                           VkFormat vkFormat,
                                           uint32_t width,
                                           uint32_t height,
                                           std::string_view debugName)
{
    if (device == nullptr || image == 0 || width == 0 || height == 0)
    {
        return nullptr;
    }

    const nvrhi::Format format = convertVkFormat(vkFormat);
    if (format == nvrhi::Format::UNKNOWN)
    {
        return nullptr;
    }

    // The backend wraps the VkImage and builds its own views through Texture::getSubresourceView
    // (vulkan-texture.cpp:287-349); no NVRHI entry point adopts a native view - the only entry
    // point, createHandleForNativeTexture, takes the image alone (vulkan-texture.cpp:805-822) - so
    // the engine's view has nowhere to go and the parameter stays unused.
    (void)imageView;

    nvrhi::TextureDesc desc;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.format = format;
    desc.width = width;
    desc.height = height;
    // The engine's offscreen images are single-mip, single-layer, single-sample 2D images
    // (Framebuffers.cpp:699-706), so the wrap exposes exactly that shape; the caller's width and
    // height must be the image's real extent or NVRHI's framebuffer compatibility checks reject it
    // (vulkan-graphics.cpp:67-68, :93-94).
    desc.mipLevels = 1;
    desc.arraySize = 1;
    desc.sampleCount = 1;

    // The pass renders into the image and the present step samples the result, so NVRHI must be
    // able to move the texture between RenderTarget and ShaderResource within one frame.
    desc.isRenderTarget = true;
    desc.isShaderResource = true;
    desc.isUAV = false;

    // keepInitialState is false: the engine's own writes are invisible to NVRHI and, while rhiframe
    // is on, the RHI is the only user, so the tracker must follow the RHI's own states and leave the
    // image wherever the last use put it. keepInitialState = true would instead seed every command
    // list from initialState and force the image back there on close (state-tracking.cpp:350-361,
    // :422-425); with the flag false no state survives a command list, so the caller repeats the
    // beginTrackingTextureState announcement on each list (RhiTextureSource.h documents the rule).
    desc.keepInitialState = false;

    // initialState is deliberately left at its default (ResourceStates::Unknown, nvrhi.h:525)
    // instead of RenderTarget: with keepInitialState = false the Vulkan backend never reads the
    // field. The reads are all guarded - getTextureSubresourceState returns Unknown outright
    // (state-tracking.cpp:140-142), getTextureStateTracking seeds the tracked state from it only
    // under the flag (:422-425), keepTextureInitialStates skips the texture (:350-361) - and
    // createHandleForNativeTexture itself only stores the desc (vulkan-texture.cpp:805-822); the
    // validation wrapper rejects the pair only when the flag is set (validation-device.cpp:254-257).
    // D3D12 consumes it only when creating a texture of its own (d3d12-texture.cpp:344-345), not in
    // its native wrap (:515-537), and this bridge targets the Vulkan backend.

    desc.debugName = std::string(debugName);

    const nvrhi::Object nativeImage(image);

    // Same tail as wrapEngineTexture: the backend rejects a null Object and any object type other
    // than VK_Image (vulkan-texture.cpp:805-811) and marks the wrapper managed = false (:819), so
    // neither the handle nor NVRHI owns the image.
    return device->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, nativeImage, desc);
}

nvrhi::TextureHandle wrapEngineStorageImage(nvrhi::IDevice *device,
                                            uint64_t image,
                                            uint64_t imageView,
                                            VkFormat vkFormat,
                                            uint32_t width,
                                            uint32_t height,
                                            std::string_view debugName)
{
    if (device == nullptr || image == 0 || width == 0 || height == 0)
    {
        return nullptr;
    }

    const nvrhi::Format format = convertVkFormat(vkFormat);
    if (format == nvrhi::Format::UNKNOWN)
    {
        return nullptr;
    }

    // As in wrapEngineRenderTarget: no NVRHI entry point adopts a native view - the only entry
    // point, createHandleForNativeTexture, takes the image alone (vulkan-texture.cpp:805-822) and
    // the backend rebuilds every view from the desc (:287-349) - so the engine's view stays with
    // the caller and the parameter is unused.
    (void)imageView;

    nvrhi::TextureDesc desc;
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.format = format;
    desc.width = width;
    desc.height = height;
    // Single-mip, single-layer, single-sample 2D, exactly the shape the engine creates its
    // framebuffer images with (Framebuffers.cpp:699-706).
    desc.mipLevels = 1;
    desc.arraySize = 1;
    desc.sampleCount = 1;

    // The flags describe the engine image's usage bits, not the binding this wrap is made for.
    // PRIMARY_TO_REFL_REFR has flags 0 (ShaderCommonCFramebuf.cpp:162), so its VkImage has no
    // COLOR_ATTACHMENT usage (the bit is added only for FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_IS_ATTACHMENT,
    // Framebuffers.cpp:714-717) while STORAGE and SAMPLED are part of every framebuffer image's
    // usage set (:708-711). isUAV = true is the field the render-target wrap never sets: the
    // validation device refuses a Texture_UAV binding on a texture without it
    // (validation-device.cpp:1631-1634). The Vulkan backend itself derives the view from the
    // texture's desc format in the Texture_UAV path and never reads the flags
    // (vulkan-resource-bindings.cpp:439-456; pickImageUsage is only used when NVRHI creates an
    // image of its own, vulkan-texture.cpp:102-129, never in a native wrap :805-822).
    desc.isRenderTarget = false;
    desc.isShaderResource = true;
    desc.isUAV = true;

    // keepInitialState is false for the same reason as in wrapEngineRenderTarget: the engine's own
    // writes are invisible to NVRHI and, while rhiframe is on, the RHI is the only user, so the
    // tracker must follow the RHI's own states and no state survives a command list. The image's
    // real resting state is the engine's VK_IMAGE_LAYOUT_GENERAL (Framebuffers.cpp:754-758), which
    // NVRHI names UnorderedAccess (vulkan-constants.cpp:242-245) - the same state a UAV binding
    // requires - so the caller's per-list announcement names UnorderedAccess and the first use
    // issues no layout transition (RhiTextureSource.h documents the rule).
    desc.keepInitialState = false;

    // initialState is deliberately left at its default (ResourceStates::Unknown, nvrhi.h:525) for
    // the reason spelled out in wrapEngineRenderTarget: with keepInitialState = false the backend
    // never reads it and createHandleForNativeTexture only stores the desc.
    desc.debugName = std::string(debugName);

    const nvrhi::Object nativeImage(image);

    return device->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, nativeImage, desc);
}

nvrhi::SamplerHandle createEngineTextureSampler(nvrhi::IDevice *device, std::string_view debugName)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    // nvrhi::SamplerDesc has no debugName member at the pinned revision (nvrhi.h:1327-1353), so the
    // argument cannot be applied; it stays for symmetry with wrapEngineTexture.
    (void)debugName;

    nvrhi::SamplerDesc desc;

    // true selects eLinear on the Vulkan side for each of the three modes
    // (vulkan-texture.cpp:859-861). The engine's linear filter (SamplerManager.cpp:34) and linear
    // mip mode (SamplerManager.cpp:89) are what a sampled 2D texture needs.
    desc.minFilter = true;
    desc.magFilter = true;
    desc.mipFilter = true;

    // ClampToEdge is NVRHI's name for the engine's VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
    // (SamplerManager.cpp:46, nvrhi.h:1302-1317). One sampler serves every engine texture on the
    // RHI side, and clamp is the safe choice for arbitrary UVs.
    desc.addressU = nvrhi::SamplerAddressMode::ClampToEdge;
    desc.addressV = nvrhi::SamplerAddressMode::ClampToEdge;
    desc.addressW = nvrhi::SamplerAddressMode::ClampToEdge;

    // Anisotropy stays off: the engine enables it only from its config (SamplerManager.cpp:91-92)
    // and the NVRHI default maxAnisotropy = 1 turns it off in the backend (vulkan-texture.cpp:855,
    // :866-867).

    return device->createSampler(desc);
}

}
