#pragma once

#include <nvrhi/vulkan.h>

#include <cstdint>
#include <string_view>

namespace vkpt::rhi
{

// Bridges the legacy texture system to the RHI layer: an engine VkImage that TextureManager
// uploaded through TextureUploader becomes an nvrhi::ITexture that an NVRHI pass can sample,
// without re-uploading the image or touching the legacy renderer.
//
// Lifetime and state:
//  - The handle references the image, it does not own it: it must not outlive vkImage, and the
//    engine may destroy the image only after the RHI side has stopped using the handle.
//  - The engine leaves every image it uploads in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: the last
//    barrier of the upload goes to that layout for the single-mip case (TextureUploader.cpp:375-380)
//    and for the mipmapped one (:364-369), and the update path even expects to find the image there
//    before it re-uses it (TextureUploader.cpp:275-277). That is the layout NVRHI calls
//    ResourceStates::ShaderResource (vulkan-constants.cpp:234-241 maps PixelShaderResource and
//    NonPixelShaderResource to eShaderReadOnlyOptimal with eShaderRead access), so the desc built
//    by wrapEngineTexture declares exactly that as initialState with keepInitialState = true.
//  - The desc takes its mip count from the caller, which passes the image's real level count (the
//    legacy engine's TextureUploader::GetMipmapCount rule mirrors it), while arraySize = 1 still
//    stands: NVRHI builds its views over a single 2D layer (vulkan-texture.cpp:287-349). The engine
//    can create cubemaps (TextureUploader.cpp:226-227), but the RHI side does not see their layers.
//
// First use of the returned handle (must be respected by the caller):
//  - On the very first command list that samples the handle, NVRHI's tracker starts from
//    ResourceStates::Common instead of initialState - state-tracking.cpp:422-425 reads
//    'stateInitialized', which only becomes true when a command list that used the texture closes
//    (state-tracking.cpp:395-399) - and a Common -> ShaderResource transition lowers to Undefined ->
//    ShaderReadOnlyOptimal (vulkan-constants.cpp:214-217 and :234-241), which lets Vulkan discard
//    the existing contents (vulkan-state-tracking.cpp:242-252 emits exactly that barrier). On the
//    first command list the caller must therefore announce the true state before the first draw
//    that binds the handle:
//        commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources,
//                                              nvrhi::ResourceStates::ShaderResource);
//    (the same call the skeleton makes for the swapchain images, NvrhiFrameSkeleton.cpp:207). After
//    that command list closes, the tracker starts every later list from initialState
//    (state-tracking.cpp:350-361) and no barrier is emitted for the image again, as long as the
//    engine keeps it in the read-only layout.
//
// Returns a null handle if 'device' or the image is null, or if 'width'/'height' is zero, or if
// 'vkFormat' is one the pinned NVRHI cannot map (including VK_FORMAT_UNDEFINED).
nvrhi::TextureHandle wrapEngineTexture(nvrhi::IDevice *device,
                                       uint64_t vkImage,
                                       uint32_t vkFormat,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t mipLevels = 1,
                                       std::string_view debugName = {});

// Wraps an engine image that is a texture array, i.e. one whose layers a shader addresses as
// Texture2DArray. The blue-noise image is the first instance: the indirect raygen's set 5 binding 0
// declares `Texture2DArray<float4> blueNoiseTextures` (Random.hlsli:249-250), the image is
// R8G8B8A8_UNORM, 128 x 128, with 128 array layers and one mip (BlueNoise.cpp:104-119), and it rests
// in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (BlueNoise.cpp:157-161).
//
// wrapEngineTexture cannot wrap it: that helper hard-codes dimension = Texture2D and arraySize = 1,
// while NVRHI builds every view from the desc shape (vulkan-texture.cpp:287-349), so the view would
// carry a single layer, the shader's layer index would be out of range, and the validation device
// would reject a Texture2DArray binding whose dimension does not match the texture's
// (textureDimensionsCompatible, validation-device.cpp:1512-1524, used at :1638-1646). This helper
// takes the array size from the caller instead: dimension = Texture2DArray and arraySize = arraySize
// make textureDimensionToImageViewType emit an e2DArray view (vulkan-texture.cpp:72-74) spanning
// every layer the binding's subresources resolve to (vulkan-resource-bindings.cpp:402-405).
//
// The state contract is wrapEngineTexture's, unchanged:
//  - initialState = ShaderResource with keepInitialState = true, because
//    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL is the layout the engine leaves the image in and
//    declares on its descriptor (BlueNoise.cpp:157-161, :249), and ShaderResource maps to exactly
//    that layout with shader read access (vulkan-constants.cpp:234-241).
//  - On the very first command list that samples the handle, NVRHI's tracker still starts from
//    ResourceStates::Common instead of initialState - stateInitialized only becomes true when a
//    command list that used the texture closes (state-tracking.cpp:395-399), and until then the
//    seeded state is Common (:422-425) - while Common lowers to Undefined (vulkan-constants.cpp:
//    214-217). The first use would therefore emit an Undefined -> ShaderReadOnlyOptimal barrier that
//    Vulkan is allowed to satisfy by discarding the contents (vulkan-state-tracking.cpp:242-252).
//    The caller must announce the true state before the first binding on that list:
//        commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources,
//                                              nvrhi::ResourceStates::ShaderResource);
//    (the same call RhiTextureTable::TrackPendingTextures makes for the engine textures it wraps,
//    RhiTextureTable.cpp:301-319). After that command list closes, every later list starts from
//    initialState (state-tracking.cpp:350-361) and no barrier is emitted for the image again, as
//    long as the engine keeps it in the read-only layout.
//
// The image is static - BlueNoise creates it once and never updates it - so the handle never has to
// be re-wrapped or released.
//
// Returns a null handle if 'device' or the image is null, if 'width', 'height' or 'arraySize' is
// zero, or if 'vkFormat' is one the pinned NVRHI cannot map (including VK_FORMAT_UNDEFINED);
// VK_FORMAT_R8G8B8A8_UNORM maps to Format::RGBA8_UNORM (vulkan-constants.cpp:59). 'imageView' is
// not consumed: NVRHI wraps the VkImage and builds its own views (vulkan-texture.cpp:805-822), so
// the engine's view stays with the caller and the parameter is kept for call-site symmetry with
// wrapEngineRenderTarget and wrapEngineStorageImage.
nvrhi::TextureHandle wrapEngineTextureArray(nvrhi::IDevice *device,
                                            uint64_t vkImage,
                                            uint64_t imageView,
                                            uint32_t vkFormat,
                                            uint32_t width,
                                            uint32_t height,
                                            uint32_t arraySize,
                                            uint32_t mipLevels = 1,
                                            std::string_view debugName = {});

// Wraps an engine image that the pass renders into. The engine's offscreen images live in
// VK_IMAGE_LAYOUT_GENERAL and are recreated on resize, so the caller re-wraps on size change. The
// state rules of wrapEngineTexture apply unchanged: NVRHI cannot see the engine's own layout, so the
// first command list that touches the image declares its first use
// (beginTrackingTextureState(image, AllSubresources, ResourceStates::RenderTarget)).
//
// Because keepInitialState is false for this wrap, that declaration cannot be a one-off: the
// tracker keeps no texture state between command lists - commandListSubmitted clears the per-list
// map (state-tracking.cpp:395-402, called from vulkan-commandlist.cpp:134) and a new list starts the
// image at Unknown (state-tracking.cpp:417-425; getTextureSubresourceState answers Unknown outright,
// :140-142). Every command list must therefore announce the state the image is really in before its
// first use - RenderTarget before the pass that draws into it, ShaderResource before a list that
// only samples the present result (nvrhi.h:3711-3716 states the rule). Without the announcement
// NVRHI reports an unknown prior state (:175-182) and emits an Undefined-sourced transition that
// discards the contents.
//
// desc.initialState stays at its default (Unknown, nvrhi.h:525): with keepInitialState = false the
// Vulkan backend never reads it, so setting it would be dead weight. The field is consulted only
// under the keepInitialState guard (state-tracking.cpp:140-142, :422-425), the validation wrapper
// only rejects it in that combination (validation-device.cpp:254-257), and
// createHandleForNativeTexture stores the desc as-is without touching it (vulkan-texture.cpp:
// 805-822). D3D12 reads it only when it creates a texture of its own (d3d12-texture.cpp:344-345),
// never in its native wrap (:515-537), and this bridge is Vulkan-only regardless.
//
// Layout note for the caller: NVRHI maps RenderTarget to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
// (vulkan-constants.cpp:246-249) while the engine leaves the image in VK_IMAGE_LAYOUT_GENERAL
// (Framebuffers.cpp:754-758, declared on its descriptors at :786, :794), which NVRHI would call
// UnorderedAccess (vulkan-constants.cpp:242-245). A beginTrackingTextureState declaration suppresses
// the transition for the state it announces, so the caller must either announce the state that
// matches the image's real layout or have moved the image to the layout NVRHI assumes; otherwise
// the later RenderTarget -> ShaderResource transition names an oldLayout the image is not in.
//
// Returns a null handle if 'device' or the image is null, or if 'width'/'height' is zero, or if
// 'vkFormat' is one the pinned NVRHI cannot map (including VK_FORMAT_UNDEFINED). 'imageView' is not
// consumed: NVRHI wraps the VkImage and builds its own views (vulkan-texture.cpp:287-349), so the
// engine's view stays with the caller and the parameter is kept for call-site symmetry.
nvrhi::TextureHandle wrapEngineRenderTarget(nvrhi::IDevice *device,
                                           uint64_t image,
                                           uint64_t imageView,
                                           VkFormat vkFormat,
                                           uint32_t width,
                                           uint32_t height,
                                           std::string_view debugName = {});

// Wraps an engine image that a pass uses as a storage image (a UAV) instead of rendering into it.
// The world pass's set-4 binding 25 is that use: FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR
// (ShaderCommonCFramebuf.h:39) is a VK_FORMAT_R32G32B32A32_UINT image (ShaderCommonCFramebuf.cpp:34)
// that the fragment shader reads and writes as Rgba32ui (Generated/ShaderCommonHLSL.hlsli:598). The
// image is recreated on resize exactly like the render target above, so the caller re-wraps on size
// change and the owning pass's release contract applies.
//
// desc.isUAV = true is the point of this wrap: the validation device refuses a Texture_UAV binding
// on a texture whose desc lacks the flag (validation-device.cpp:1631-1634). The Vulkan backend does
// not read the flag to make the view - the Texture_UAV path builds it from the binding's format, or
// from the texture's desc format when the binding leaves it UNKNOWN, with
// vk::ImageUsageFlagBits::eStorage and VK_IMAGE_LAYOUT_GENERAL (vulkan-resource-bindings.cpp:
// 439-456), and getSubresourceView reads only the desc's format, dimension and shape
// (vulkan-texture.cpp:287-349). The flag triple feeds pickImageUsage (:102-129), which runs only
// when NVRHI creates an image of its own, never for a native wrap (:805-822), so on this backend
// the flags describe the engine image rather than drive it.
//
// That image is not an attachment and has no COLOR_ATTACHMENT usage: PRIMARY_TO_REFL_REFR carries
// flags 0 (ShaderCommonCFramebuf.cpp:162) and the usage bit is added only for
// FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_IS_ATTACHMENT (Framebuffers.cpp:714-717), the common set being
// transfer-source | storage | sampled (:708-711). desc.isRenderTarget therefore stays false while
// desc.isShaderResource stays true, matching the two real usages; the wrap exists for the UAV view.
//
// State contract: the rules of wrapEngineRenderTarget apply unchanged - keepInitialState is false,
// no tracked state survives a command list (state-tracking.cpp:395-402), so every list announces
// the image's true state before its first use (state-tracking.cpp:417-425, nvrhi.h:3711-3716). The
// state to announce is UnorderedAccess, not RenderTarget: the engine leaves every framebuffer image
// in VK_IMAGE_LAYOUT_GENERAL (Framebuffers.cpp:754-758) and declares that layout on its own storage
// descriptors (:786), and NVRHI maps UnorderedAccess to exactly that layout with shader read and
// write access (vulkan-constants.cpp:242-245). UnorderedAccess is also the state a Texture_UAV
// binding requires (vulkan-resource-bindings.cpp:439-463), so the announcement matches and the
// first use issues no transition; the image stays in the layout the engine knows:
//     commandList->beginTrackingTextureState(image, nvrhi::AllSubresources,
//                                            nvrhi::ResourceStates::UnorderedAccess);
//
// Returns a null handle if 'device' or the image is null, or if 'width'/'height' is zero, or if
// 'vkFormat' is one the pinned NVRHI cannot map (including VK_FORMAT_UNDEFINED).
// VK_FORMAT_R32G32B32A32_UINT is mapped: the translation unit's inverse search finds
// Format::RGBA32_UINT at vulkan-constants.cpp:87. 'imageView' is not consumed, as in
// wrapEngineRenderTarget.
nvrhi::TextureHandle wrapEngineStorageImage(nvrhi::IDevice *device,
                                            uint64_t image,
                                            uint64_t imageView,
                                            VkFormat vkFormat,
                                            uint32_t width,
                                            uint32_t height,
                                            std::string_view debugName = {});

// The sampler the RHI side uses for engine textures. The pinned NVRHI cannot wrap the engine's
// VkSampler objects: the device has native entry points for textures and buffers only
// (createHandleForNativeTexture, nvrhi.h:3800; createHandleForNativeBuffer, nvrhi.h:3818) and none
// for samplers, so the bridge creates an RHI sampler of its own instead.
//
// It is linear/linear/clamp: linear minification and magnification mirror the engine's
// RG_SAMPLER_FILTER_LINEAR -> VK_FILTER_LINEAR (SamplerManager.cpp:34), the linear mip mode mirrors
// the engine's VkSamplerCreateInfo (SamplerManager.cpp:89), and clamp on all axes mirrors
// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE (SamplerManager.cpp:46). Anisotropy stays off, which the
// engine only enables from its config (SamplerManager.cpp:91-92) and the NVRHI backend disables at
// its default maxAnisotropy = 1 (vulkan-texture.cpp:855).
//
// nvrhi::SamplerDesc has no debugName at the pinned revision (nvrhi.h:1327-1353), so 'debugName'
// cannot be attached; it stays in the signature for symmetry with wrapEngineTexture.
nvrhi::SamplerHandle createEngineTextureSampler(nvrhi::IDevice *device, std::string_view debugName);

}
