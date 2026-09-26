#include "RhiRtGodRaysPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../Generated/ShaderCommonCFramebuf.h"
#include "../Utils.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string>

using namespace vkpt;

namespace
{

// The two engine blobs, by the file names the shader build writes into the folder the host passes in
// (ShaderManager.cpp:41-49) - the same names the legacy `GodRays` pipelines load. The trace runs at
// half resolution, the filter at full resolution.
const char *const TRACE_SHADER_FILE_NAME = "CmGodRays.comp.spv";
const char *const FILTER_SHADER_FILE_NAME = "CmGodRaysFilter.comp.spv";

// The engine's framebuffer binding block: the storage images sit at raw binding = image index, the
// sampled views at raw binding = 124 + image index (Generated/ShaderCommonCFramebuf.cpp's
// ShFramebuffers_Bindings / ShFramebuffers_Sampled_Bindings, ShaderCommonC.h:31-32-style offsets).
// An item's NVRHI slot is `raw - offset`.
constexpr uint32_t FRAMEBUFFER_UAV_OFFSET = 0;
constexpr uint32_t FRAMEBUFFER_SRV_OFFSET = 124;

// The engine images of the module, in the module's own slot order (RhiRtGodRaysPass::ImageSlot).
// 63/64 are the god-rays pair, 9/19/23/26/81 are the checkerboard-stored surface data the trace
// reads, and 89 is the primary's `storeQ2GBuffer` output (`RaygenPrimary.hlsli:207`; the reflection
// pass's segment length is its .w).
constexpr uint32_t GOD_RAYS_IMAGE_COUNT = RhiRtGodRaysPass::IMAGE_COUNT;
constexpr FramebufferImageIndex GOD_RAYS_IMAGES[GOD_RAYS_IMAGE_COUNT] =
{
    FB_IMAGE_INDEX_GOD_RAYS,                    // 63
    FB_IMAGE_INDEX_GOD_RAYS_FILTERED,           // 64
    FB_IMAGE_INDEX_DEPTH_WORLD,                 // 9
    FB_IMAGE_INDEX_SURFACE_POSITION,            // 19
    FB_IMAGE_INDEX_VIEW_DIRECTION,              // 23
    FB_IMAGE_INDEX_THROUGHPUT,                  // 26
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH,               // 81
    FB_IMAGE_INDEX_Q2_GOD_RAYS_THROUGHPUT_DIST, // 89
};
static_assert(GOD_RAYS_IMAGE_COUNT == 8, "RhiRtGodRaysPass::Target is sized by IMAGE_COUNT");

// One item of a framebuffer set: the image slot of the module's table and the binding kind. The
// tables below are the exact measured footprints of the two blobs; the raw bindings they produce
// are stated per entry.
struct GodRaysFramebufferBinding
{
    uint32_t imageSlot;
    bool isUAV;
};

// CmGodRays set 2 (measured: `framebufGodRays` UAV at raw 63, then the sampled views at raw 133,
// 143, 147, 150, 205 and 213), which is seven items: the UAV of 63 and the SRVs of 9/19/23/26/81/89.
constexpr uint32_t TRACE_FRAMEBUFFER_BINDING_COUNT = 7;
constexpr GodRaysFramebufferBinding TRACE_FRAMEBUFFER_BINDINGS[TRACE_FRAMEBUFFER_BINDING_COUNT] =
{
    { RhiRtGodRaysPass::IMAGE_GOD_RAYS,                    true  }, // raw 63  framebufGodRays (UAV)
    { RhiRtGodRaysPass::IMAGE_DEPTH_WORLD,                 false }, // raw 133 framebufDepthWorld_Sampled
    { RhiRtGodRaysPass::IMAGE_SURFACE_POSITION,            false }, // raw 143 framebufSurfacePosition_Sampled
    { RhiRtGodRaysPass::IMAGE_VIEW_DIRECTION,              false }, // raw 147 framebufViewDirection_Sampled
    { RhiRtGodRaysPass::IMAGE_THROUGHPUT,                  false }, // raw 150 framebufThroughput_Sampled
    { RhiRtGodRaysPass::IMAGE_Q2_VIEW_DEPTH,               false }, // raw 205 framebufQ2ViewDepth_Sampled
    { RhiRtGodRaysPass::IMAGE_Q2_GODRAYS_THROUGHPUT_DIST,  false }, // raw 213 framebufQ2GodRaysThroughputDist_Sampled
};

// CmGodRaysFilter set 2 (measured: `framebufGodRaysFiltered` UAV at raw 64 and the sampled views at
// raw 187 (= 124 + 63) and 205 (= 124 + 81)).
constexpr uint32_t FILTER_FRAMEBUFFER_BINDING_COUNT = 3;
constexpr GodRaysFramebufferBinding FILTER_FRAMEBUFFER_BINDINGS[FILTER_FRAMEBUFFER_BINDING_COUNT] =
{
    { RhiRtGodRaysPass::IMAGE_GOD_RAYS_FILTERED, true  }, // raw 64  framebufGodRaysFiltered (UAV)
    { RhiRtGodRaysPass::IMAGE_GOD_RAYS,          false }, // raw 187 framebufGodRays_Sampled
    { RhiRtGodRaysPass::IMAGE_Q2_VIEW_DEPTH,     false }, // raw 205 framebufQ2ViewDepth_Sampled
};

// The images whose last use on a frame list is a sampled read: the trace reads 9/19/23/26/81/89 and
// the filter reads 63 (image 81 again), so all of them are moved back to UnorderedAccess after the
// filter dispatch. 64's last use is the filter's write, so it ends in UnorderedAccess already.
constexpr uint32_t GOD_RAYS_RESTORE_IMAGE_COUNT = 7;
constexpr uint32_t GOD_RAYS_RESTORE_IMAGES[GOD_RAYS_RESTORE_IMAGE_COUNT] =
{
    RhiRtGodRaysPass::IMAGE_GOD_RAYS,
    RhiRtGodRaysPass::IMAGE_DEPTH_WORLD,
    RhiRtGodRaysPass::IMAGE_SURFACE_POSITION,
    RhiRtGodRaysPass::IMAGE_VIEW_DIRECTION,
    RhiRtGodRaysPass::IMAGE_THROUGHPUT,
    RhiRtGodRaysPass::IMAGE_Q2_VIEW_DEPTH,
    RhiRtGodRaysPass::IMAGE_Q2_GODRAYS_THROUGHPUT_DIST,
};

// The shape the shader's `blueNoiseTextures.Load(int4(x, y, layer, 0))` needs (Random.hlsli:257-266):
// 128x128 with at least 128 layers, exactly the indirect pass's set-5 contract.
constexpr uint32_t BLUE_NOISE_EXTENT = BLUE_NOISE_TEXTURE_SIZE;
constexpr uint32_t BLUE_NOISE_LAYER_COUNT = BLUE_NOISE_TEXTURE_COUNT;

uint32_t GetFramebufferUavSlot(FramebufferImageIndex image)
{
    return ShFramebuffers_Bindings[image] - FRAMEBUFFER_UAV_OFFSET;
}

uint32_t GetFramebufferSrvSlot(FramebufferImageIndex image)
{
    return ShFramebuffers_Sampled_Bindings[image] - FRAMEBUFFER_SRV_OFFSET;
}

// The extent the engine gives one module image at the passed resolution
// (`Framebuffers::GetFramebufSize`): 63 is FORCE_SIZE_1_2, everything else is render-sized.
VkExtent2D GetGodRaysImageExtent(uint32_t imageSlot, uint32_t width, uint32_t height)
{
    if (imageSlot == RhiRtGodRaysPass::IMAGE_GOD_RAYS)
    {
        return { std::max(1u, (width + 1) / 2), std::max(1u, (height + 1) / 2) };
    }

    return { width, height };
}

// The exact set-2 layout of one pass: Compute visibility (the blobs are compute shaders), one item
// per declared raw binding, the framebuffer block's offsets. The engine's own set-2 layout is not
// reused because it exists only in the engine's Vulkan descriptor world and would carry the wrong
// stage flags for a compute pipeline.
nvrhi::BindingLayoutHandle CreateFramebufferLayout(nvrhi::IDevice *device,
                                                   const GodRaysFramebufferBinding *pBindings,
                                                   uint32_t count)
{
    if (device == nullptr)
    {
        return nullptr;
    }

    nvrhi::BindingLayoutDesc desc;
    desc.visibility = nvrhi::ShaderType::Compute;
    desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                               .setShaderResourceOffset(FRAMEBUFFER_SRV_OFFSET)
                               .setUnorderedAccessViewOffset(FRAMEBUFFER_UAV_OFFSET));

    for (uint32_t i = 0; i < count; i++)
    {
        const FramebufferImageIndex image = GOD_RAYS_IMAGES[pBindings[i].imageSlot];

        desc.addItem(pBindings[i].isUAV
                         ? nvrhi::BindingLayoutItem::Texture_UAV(GetFramebufferUavSlot(image))
                         : nvrhi::BindingLayoutItem::Texture_SRV(GetFramebufferSrvSlot(image)));
    }

    return device->createBindingLayout(desc);
}

// The set over one exact layout: one item per table entry, filled with the slot's wrap of that
// image. Returns null when an entry cannot be filled, which the Render-time wrap success and the
// exact tables make unreachable for a prepared target.
nvrhi::BindingSetHandle CreateFramebufferSet(nvrhi::IDevice *device,
                                            const nvrhi::TextureHandle *pTextures,
                                            const GodRaysFramebufferBinding *pBindings,
                                            uint32_t count,
                                            nvrhi::IBindingLayout *pLayout)
{
    if (device == nullptr || pLayout == nullptr)
    {
        return nullptr;
    }

    nvrhi::BindingSetDesc setDesc;

    for (uint32_t i = 0; i < count; i++)
    {
        nvrhi::ITexture *texture = pTextures[pBindings[i].imageSlot].Get();
        if (texture == nullptr)
        {
            return nullptr;
        }

        const FramebufferImageIndex image = GOD_RAYS_IMAGES[pBindings[i].imageSlot];

        setDesc.addItem(pBindings[i].isUAV
                            ? nvrhi::BindingSetItem::Texture_UAV(GetFramebufferUavSlot(image), texture)
                            : nvrhi::BindingSetItem::Texture_SRV(GetFramebufferSrvSlot(image), texture));
    }

    return device->createBindingSet(setDesc, pLayout);
}

void LogMessage(const RhiRtGodRaysPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtGodRaysPass::RhiRtGodRaysPass() = default;

RhiRtGodRaysPass::~RhiRtGodRaysPass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images and the sets reference the device's resources; the host
        // destroys the pass while it can still idle the device (VulkanDevice does that before the
        // skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    for (Target &target : targets)
    {
        target.traceFramebufferSet = nullptr;
        target.filterFramebufferSet = nullptr;

        for (nvrhi::TextureHandle &texture : target.engineTextures)
        {
            texture = nullptr;
        }

        target.uniformSet = nullptr;
        target.uniformBuffer = nullptr;
        std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
        target.width = 0;
        target.height = 0;
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        paramsSets[i] = nullptr;
        paramsBuffers[i] = nullptr;
    }

    emptySet = nullptr;
    shadowSet = nullptr;
    shadowMapSampler = nullptr;
    shadowMapTexture = nullptr;
    blueNoiseSet = nullptr;
    blueNoiseTexture = nullptr;
    filterPipeline = nullptr;
    tracePipeline = nullptr;
    emptyLayout = nullptr;
    pushConstantLayout = nullptr;
    blueNoiseLayout = nullptr;
    uniformLayout = nullptr;
    filterFramebufferLayout = nullptr;
    traceFramebufferLayout = nullptr;
    paramsLayout = nullptr;
    shadowLayout = nullptr;
    filterShader = nullptr;
    traceShader = nullptr;
}

bool RhiRtGodRaysPass::Create(nvrhi::IDevice *pDevice,
                              rhi::RhiFrameContext *pFrameContext,
                              const char *pShaderFolderPath,
                              PrintFunction pfnPrint)
{
    if (created)
    {
        return true;
    }

    device = pDevice;
    print = std::move(pfnPrint);
    shaderFolderPath = pShaderFolderPath != nullptr ? pShaderFolderPath : "";
    frameContext = pFrameContext;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the god-rays pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the god-rays pass needs the frame context of the RHI layer");
        return false;
    }

    if (!LoadShader(TRACE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, traceShader) ||
        !LoadShader(FILTER_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, filterShader))
    {
        return false;
    }

    // Set 0: the shadow map's `Texture2D<float4>` at raw binding 0 and its sampler at raw binding 1.
    // A sampler offset of 0 keeps an item's slot equal to the raw binding.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setShaderResourceOffset(0)
                                   .setSamplerOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(0));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(1));

        shadowLayout = device->createBindingLayout(desc);
    }

    // Set 1: the `StructuredBuffer<GodRaysParams_BT>` at raw binding 0.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0));

        paramsLayout = device->createBindingLayout(desc);
    }

    // The two set-2 layouts, one per blob (7 and 3 items, the tables above).
    traceFramebufferLayout = CreateFramebufferLayout(
        device, TRACE_FRAMEBUFFER_BINDINGS, TRACE_FRAMEBUFFER_BINDING_COUNT);
    filterFramebufferLayout = CreateFramebufferLayout(
        device, FILTER_FRAMEBUFFER_BINDINGS, FILTER_FRAMEBUFFER_BINDING_COUNT);

    // Set 3: the engine's global uniform at raw binding 0
    // (ShaderCommonHLSL.hlsli:22 spells it with BINDING_GLOBAL_UNIFORM and DESC_SET_GLOBAL_UNIFORM).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM));

        uniformLayout = device->createBindingLayout(desc);
    }

    // Set 4: the blue-noise `Texture2DArray<float4>` at raw binding 0, no sampler.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_BLUE_NOISE));

        blueNoiseLayout = device->createBindingLayout(desc);
    }

    // The trace's push constant. The pinned backend skips a PushConstants item when it builds the
    // Vulkan descriptor set layout and takes the pipeline's single VkPushConstantRange from it
    // (vulkan-resource-bindings.cpp:90-94, :1110-1140), so the layout creates an empty descriptor
    // set layout and no set is bound for it - the same shape RhiRtComposePass uses for its
    // push-constant iterations.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, GOD_RAYS_PUSH_SIZE));

        pushConstantLayout = device->createBindingLayout(desc);
    }

    // The filter's blob declares its framebuffer and uniform bindings in engine sets 2 and 3, so
    // its pipeline layout needs two leading positions. The legacy bound all five engine sets; this
    // module fills the two holes with the module's one empty layout, the same object twice.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;

        emptyLayout = device->createBindingLayout(desc);
    }

    if (shadowLayout == nullptr || paramsLayout == nullptr || traceFramebufferLayout == nullptr ||
        filterFramebufferLayout == nullptr || uniformLayout == nullptr || blueNoiseLayout == nullptr ||
        pushConstantLayout == nullptr || emptyLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a god-rays pass binding layout");
        return false;
    }

    // The trace pipeline, in the engine's set order: 0 shadow, 1 params, 2 framebuffers, 3 uniform,
    // 4 blue noise, 5 the descriptor-less push-constant layout. The blob declares exactly sets 0-4,
    // so the list order is also the shader's set numbering (the backend places the regular layouts
    // into descriptor sets in list order, vulkan-resource-bindings.cpp:1090-1099).
    {
        nvrhi::ComputePipelineDesc desc;
        desc.setComputeShader(traceShader);
        desc.addBindingLayout(shadowLayout);
        desc.addBindingLayout(paramsLayout);
        desc.addBindingLayout(traceFramebufferLayout);
        desc.addBindingLayout(uniformLayout);
        desc.addBindingLayout(blueNoiseLayout);
        desc.addBindingLayout(pushConstantLayout);

        tracePipeline = rhi::createComputePipeline(device, desc, "RhiRtGodRaysPass trace");
    }

    // The filter pipeline: the two placeholder positions (engine sets 0/1, which the blob does not
    // use) and then the framebuffer set at 2 and the uniform at 3. No push constant.
    {
        nvrhi::ComputePipelineDesc desc;
        desc.setComputeShader(filterShader);
        desc.addBindingLayout(emptyLayout);
        desc.addBindingLayout(emptyLayout);
        desc.addBindingLayout(filterFramebufferLayout);
        desc.addBindingLayout(uniformLayout);

        filterPipeline = rhi::createComputePipeline(device, desc, "RhiRtGodRaysPass filter");
    }

    if (tracePipeline == nullptr || filterPipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a god-rays pass compute pipeline");
        return false;
    }

    // The one empty set the filter binds at both placeholder positions. One set bound at two
    // positions is legal Vulkan; the layouts are the same object (RhiRtComposePass uses the same
    // mechanism for its descriptor-set holes).
    emptySet = device->createBindingSet(nvrhi::BindingSetDesc(), emptyLayout);
    if (emptySet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the god-rays pass empty binding set");
        return false;
    }

    // The per-slot params buffers and the sets over them. The buffer is created with the shader's
    // measured element stride 160, not sizeof(Params) = 148, and with the write helper's state
    // contract: keepInitialState = true and initialState = CopyDest, which `writeBuffer` requires
    // and the SRV binding then transitions out of before the dispatch (RhiResources.h:25-33).
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = GOD_RAYS_PARAMS_STRIDE;
        desc.structStride = GOD_RAYS_PARAMS_STRIDE;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.keepInitialState = true;

        paramsBuffers[i] = rhi::createBuffer(device, desc, "RhiRtGodRaysPass params " + std::to_string(i));
        if (paramsBuffers[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create a god-rays pass params buffer");
            return false;
        }

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, paramsBuffers[i]));

        paramsSets[i] = device->createBindingSet(setDesc, paramsLayout);
        if (paramsSets[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create a god-rays pass params binding set");
            return false;
        }
    }

    created = true;
    return true;
}

void RhiRtGodRaysPass::SetShadowMap(nvrhi::ITexture *pShadowMap, nvrhi::ISampler *pShadowMapSampler)
{
    if (pShadowMap == nullptr || pShadowMapSampler == nullptr)
    {
        shadowMapTexture = nullptr;
        shadowMapSampler = nullptr;
        return;
    }

    const nvrhi::TextureDesc &desc = pShadowMap->getDesc();

    // The shape the shader's `texShadowMap.SampleLevel(...)` reads and the legacy image has: a
    // single-mip 2D D32 depth map with one layer (ShadowMap.cpp:172, :195-205). A desc outside it
    // is a host-side mistake, not a per-frame condition: it is refused with a one-shot warning and
    // the pass keeps skipping until a valid texture arrives.
    const bool valid = desc.dimension == nvrhi::TextureDimension::Texture2D &&
                       desc.format == nvrhi::Format::D32 &&
                       desc.mipLevels == 1 &&
                       desc.arraySize == 1 &&
                       desc.width > 0 && desc.height > 0;

    if (!valid)
    {
        if (!warnedBadShadowMap)
        {
            warnedBadShadowMap = true;
            LogMessage(print, "Warning: RHI: the god-rays shadow map has to be a Texture2D of format "
                              "D32 (D32_SFLOAT) with one mip and one array layer");
        }
        shadowMapTexture = nullptr;
        shadowMapSampler = nullptr;
        return;
    }

    // The legacy sampler is a linear MIN-reduction clamp-to-border sampler with a white border
    // (ShadowMap.cpp:210-223). A different one changes the shafts silently, so a deviation is
    // reported once - the module keeps using what it was given.
    const nvrhi::SamplerDesc &samplerDesc = pShadowMapSampler->getDesc();

    if (samplerDesc.reductionType != nvrhi::SamplerReductionType::Minimum ||
        !samplerDesc.minFilter || !samplerDesc.magFilter ||
        samplerDesc.addressU != nvrhi::SamplerAddressMode::Border ||
        samplerDesc.addressV != nvrhi::SamplerAddressMode::Border ||
        samplerDesc.addressW != nvrhi::SamplerAddressMode::Border ||
        samplerDesc.borderColor != nvrhi::Color(1.f))
    {
        if (!warnedShadowSampler)
        {
            warnedShadowSampler = true;
            LogMessage(print, "Warning: RHI: the god-rays shadow map sampler is not the legacy "
                              "linear/min-reduction/clamp-to-border/white sampler; the shafts may "
                              "leak or shift");
        }
    }

    shadowMapTexture = pShadowMap;
    shadowMapSampler = pShadowMapSampler;
}

void RhiRtGodRaysPass::SetBlueNoiseTexture(nvrhi::ITexture *pBlueNoise)
{
    if (pBlueNoise == nullptr)
    {
        blueNoiseTexture = nullptr;
        return;
    }

    const nvrhi::TextureDesc &desc = pBlueNoise->getDesc();

    // The same shape contract as the indirect pass's set 5 (Random.hlsli:249-266): the shader reads
    // `blueNoiseTextures.Load(int4(x, y, layer, 0))` with a wrapping layer index and a 128x128 texel
    // footprint, so a smaller array or extent would read outside the descriptor.
    const bool valid = desc.dimension == nvrhi::TextureDimension::Texture2DArray &&
                       desc.format == nvrhi::Format::RGBA8_UNORM &&
                       desc.width >= BLUE_NOISE_EXTENT &&
                       desc.height >= BLUE_NOISE_EXTENT &&
                       desc.arraySize >= BLUE_NOISE_LAYER_COUNT;

    if (!valid)
    {
        if (!warnedBadBlueNoise)
        {
            warnedBadBlueNoise = true;
            LogMessage(print, "Warning: RHI: the god-rays blue-noise texture (set 4) has to be a "
                              "Texture2DArray of at least 128x128 RGBA8_UNORM with at least 128 layers");
        }
        blueNoiseTexture = nullptr;
        return;
    }

    blueNoiseTexture = pBlueNoise;
}

bool RhiRtGodRaysPass::PrepareShadowSet()
{
    if (shadowMapTexture == nullptr || shadowMapSampler == nullptr)
    {
        if (!warnedMissingShadowMap)
        {
            warnedMissingShadowMap = true;
            LogMessage(print, "Warning: RHI: the god-rays pass got no shadow map (set 0), the pass is skipped");
        }
        return false;
    }

    // The texture and sampler are static, so the two-item set is built once per pair; only a
    // replaced handle makes the pointers differ. The replaced set goes through the retire queue.
    if (shadowSet == nullptr || shadowMapSetTexture != shadowMapTexture.Get() ||
        shadowMapSetSampler != shadowMapSampler.Get())
    {
        if (shadowSet != nullptr)
        {
            frameContext->Retire(shadowSet);
        }
        shadowSet = nullptr;
        shadowMapSetTexture = nullptr;
        shadowMapSetSampler = nullptr;

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, shadowMapTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(1, shadowMapSampler));

        shadowSet = device->createBindingSet(setDesc, shadowLayout);

        if (shadowSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the god-rays pass shadow binding set");
            return false;
        }

        shadowMapSetTexture = shadowMapTexture.Get();
        shadowMapSetSampler = shadowMapSampler.Get();
    }

    // No state announcement here, deliberately: the module binds the very NVRHI texture handle
    // RhiShadowMapPass owns, so the list's tracker already knows the state the shadow pass left -
    // `ShaderResource | DepthRead` after a render, or the image's `DepthWrite` resting state when
    // the clear path runs without a render. Announcing a state would overwrite that truth and could
    // leave the wrong layout in the close-time restore the image's keepInitialState triggers. The
    // SRV item's own requirement (NonPixelShaderResource for a compute-visibility layout) is what
    // transitions the image, and it is a no-op after a render because both states lower to
    // SHADER_READ_ONLY_OPTIMAL.
    return true;
}

bool RhiRtGodRaysPass::PrepareBlueNoiseSet(nvrhi::ICommandList *pCommandList)
{
    if (blueNoiseTexture == nullptr)
    {
        if (!warnedMissingBlueNoise)
        {
            warnedMissingBlueNoise = true;
            LogMessage(print, "Warning: RHI: the god-rays pass got no blue-noise texture (set 4), the pass is skipped");
        }
        return false;
    }

    if (blueNoiseSet == nullptr || blueNoiseSetTexture != blueNoiseTexture.Get())
    {
        if (blueNoiseSet != nullptr)
        {
            frameContext->Retire(blueNoiseSet);
        }
        blueNoiseSet = nullptr;
        blueNoiseSetTexture = nullptr;

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_BLUE_NOISE, blueNoiseTexture));

        blueNoiseSet = device->createBindingSet(setDesc, blueNoiseLayout);

        if (blueNoiseSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the god-rays pass blue-noise binding set");
            return false;
        }

        blueNoiseSetTexture = blueNoiseTexture.Get();
    }

    // The same announcement the indirect pass makes for the same engine image: the state its SRV
    // item requires, which the engine's resting SHADER_READ_ONLY_OPTIMAL layout matches.
    pCommandList->beginTrackingTextureState(blueNoiseTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::NonPixelShaderResource);

    return true;
}

RhiRtGodRaysPass::Target *RhiRtGodRaysPass::PrepareFrame(uint32_t frameIndex,
                                                         const Framebuffers *pFramebuffers,
                                                         uint32_t width,
                                                         uint32_t height,
                                                         nvrhi::IBuffer *pUniformBuffer)
{
    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the god-rays pass got no engine framebuffers, the pass is skipped");
        }
        return nullptr;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the god-rays pass got no global uniform, the pass is skipped");
        }
        return nullptr;
    }

    Target &target = targets[frameIndex];

    // The eight images of this slot. `GetImageHandles` resolves the engine's per-slot swap; the
    // 4-tuple overload supplies each image's extent, which has to match the module's expectation
    // (63 half-res, everything else render-sized) - a different extent is a wrongly sized wrap and
    // the frame is skipped instead of reading outside the image.
    const ResolutionState resolutionState = { width, height, 0, 0 };

    uint64_t imageHandles[GOD_RAYS_IMAGE_COUNT] = {};
    uint64_t imageViews[GOD_RAYS_IMAGE_COUNT] = {};
    VkFormat imageFormats[GOD_RAYS_IMAGE_COUNT] = {};
    VkExtent2D imageExtents[GOD_RAYS_IMAGE_COUNT] = {};

    for (uint32_t i = 0; i < GOD_RAYS_IMAGE_COUNT; i++)
    {
        const auto [image, view, format, extent] =
            pFramebuffers->GetImageHandles(GOD_RAYS_IMAGES[i], frameIndex, resolutionState);

        imageHandles[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));
        imageViews[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        imageFormats[i] = format;
        imageExtents[i] = extent;

        if (imageHandles[i] == 0)
        {
            if (!warnedMissingFramebuffers)
            {
                warnedMissingFramebuffers = true;
                LogMessage(print, "Warning: RHI: the god-rays pass got no framebuffer image, the pass is skipped");
            }
            return nullptr;
        }

        const VkExtent2D expectedExtent = GetGodRaysImageExtent(i, width, height);

        if (imageExtents[i].width != expectedExtent.width || imageExtents[i].height != expectedExtent.height)
        {
            if (!warnedUnexpectedSize)
            {
                warnedUnexpectedSize = true;
                LogMessage(print, std::string("Warning: RHI: the god-rays pass got \"") +
                                      ShFramebuffers_DebugNames[GOD_RAYS_IMAGES[i]] +
                                      "\", which is not sized as the god-rays table expects; the pass is skipped");
            }
            return nullptr;
        }
    }

    bool framebuffersChanged = target.width != width || target.height != height;

    for (uint32_t i = 0; i < GOD_RAYS_IMAGE_COUNT; i++)
    {
        framebuffersChanged = framebuffersChanged || target.imageHandles[i] != imageHandles[i];
    }

    if (framebuffersChanged)
    {
        // The engine re-created its framebuffers (a resize or an image-format change) or the
        // resolution changed: the wraps and the sets over them are retired and rebuilt. Retiring
        // first is safe even if the rebuild below fails - the next frame retries and the pass is
        // skipped until it succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < GOD_RAYS_IMAGE_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtGodRaysPass ") +
                                          ShFramebuffers_DebugNames[GOD_RAYS_IMAGES[i]] +
                                          " frame " + std::to_string(frameIndex);

            target.engineTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i],
                imageExtents[i].width, imageExtents[i].height, debugName);

            if (target.engineTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap a god-rays pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return nullptr;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
    }

    if (!PrepareFramebufferSets(target) || !PrepareUniformSet(target, pUniformBuffer))
    {
        return nullptr;
    }

    return &target;
}

void RhiRtGodRaysPass::AnnounceFrameImages(nvrhi::ICommandList *pCommandList, const Target &target) const
{
    // The image state contract of the class: the engine leaves every framebuffer image in
    // VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap keeps no state
    // between command lists (RhiTextureSource.h), so every list announces that state for all eight
    // images before the first use. The primary pass wrote some of them on this very list through
    // its own wraps; the physical layout it left them in is exactly this state, so the
    // announcement is the truth and the bindings' automatic transitions start from it.
    for (uint32_t i = 0; i < GOD_RAYS_IMAGE_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.engineTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }
}

void RhiRtGodRaysPass::RequireImageUnorderedAccess(nvrhi::ICommandList *pCommandList,
                                                   const Target &target,
                                                   uint32_t imageSlot) const
{
    pCommandList->setTextureState(target.engineTextures[imageSlot], nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
}

bool RhiRtGodRaysPass::PrepareFramebufferSets(Target &target)
{
    if (target.traceFramebufferSet == nullptr)
    {
        target.traceFramebufferSet = CreateFramebufferSet(device, target.engineTextures,
                                                          TRACE_FRAMEBUFFER_BINDINGS,
                                                          TRACE_FRAMEBUFFER_BINDING_COUNT,
                                                          traceFramebufferLayout);

        if (target.traceFramebufferSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the god-rays pass trace binding set");
            }
            return false;
        }
    }

    if (target.filterFramebufferSet == nullptr)
    {
        target.filterFramebufferSet = CreateFramebufferSet(device, target.engineTextures,
                                                           FILTER_FRAMEBUFFER_BINDINGS,
                                                           FILTER_FRAMEBUFFER_BINDING_COUNT,
                                                           filterFramebufferLayout);

        if (target.filterFramebufferSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the god-rays pass filter binding set");
            }
            return false;
        }
    }

    return true;
}

bool RhiRtGodRaysPass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformSet != nullptr && target.uniformBuffer == pUniformBuffer)
    {
        return true;
    }

    // The shader declares set 3 as ConstantBuffer<ShGlobalUniform>; the validation device refuses
    // such a binding on a buffer whose desc lacks isConstantBuffer and on a volatile one
    // (validation-device.cpp:1717-1730), so a wrong wrap is reported once and the frame is skipped.
    if (!pUniformBuffer->getDesc().isConstantBuffer || pUniformBuffer->getDesc().isVolatile)
    {
        if (!warnedBadUniform)
        {
            warnedBadUniform = true;
            LogMessage(print, "Warning: RHI: the god-rays pass needs the global uniform as a static constant-buffer wrap");
        }
        return false;
    }

    if (target.uniformSet != nullptr)
    {
        frameContext->Retire(target.uniformSet);
    }
    target.uniformSet = nullptr;

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM, pUniformBuffer));

    target.uniformSet = device->createBindingSet(setDesc, uniformLayout);

    if (target.uniformSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the god-rays pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

void RhiRtGodRaysPass::RecordDispatch(nvrhi::ICommandList *pCommandList,
                                      nvrhi::IComputePipeline *pPipeline,
                                      std::initializer_list<nvrhi::IBindingSet *> sets,
                                      uint32_t groupsX,
                                      uint32_t groupsY,
                                      uint32_t groupsZ,
                                      const uint32_t *pPassIndex)
{
    nvrhi::ComputeState state;
    state.setPipeline(pPipeline);

    for (nvrhi::IBindingSet *pSet : sets)
    {
        state.addBindingSet(pSet);
    }

    pCommandList->setComputeState(state);

    // The passIndex of the two trace dispatches. The write needs the pipeline's layout, which
    // `setComputeState` just installed; the state cached by the backend keeps it until the next
    // state set (vulkan-compute.cpp:142-145), so the dispatch below sees the same layout. The
    // filter declares no push constant, so it passes null.
    if (pPassIndex != nullptr)
    {
        pCommandList->setPushConstants(pPassIndex, sizeof(uint32_t));
    }

    pCommandList->dispatch(groupsX, groupsY, groupsZ);
}

void RhiRtGodRaysPass::Render(nvrhi::ICommandList *pCommandList,
                              uint32_t frameIndex,
                              const Framebuffers *pFramebuffers,
                              uint32_t width,
                              uint32_t height,
                              nvrhi::IBuffer *pUniformBuffer,
                              const Params &params,
                              bool traceReflections)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    if (width == 0 || height == 0)
    {
        return;
    }

    // The two static sets first: a missing shadow map or blue-noise texture skips the frame before
    // any wrap work, and the blue-noise texture's first-use announcement happens inside, on this
    // list. The shadow map is needed even on the clear path: every item of the trace's set 2 layout
    // has to be filled, although the shader never samples it when `godRaysEnabled == 0`.
    if (!PrepareShadowSet() || !PrepareBlueNoiseSet(pCommandList))
    {
        return;
    }

    Target *pTarget = PrepareFrame(frameIndex, pFramebuffers, width, height, pUniformBuffer);
    if (pTarget == nullptr)
    {
        return;
    }

    Target &target = *pTarget;

    // The frame's params. The buffer is 160 bytes (the shader's element stride); the CPU mirror is
    // 148, so the stride padding is zeroed here - the shader never reads it (it only reads element
    // [0] and no member at 144+), but a fully defined buffer keeps the view range deterministic.
    uint8_t paramsBytes[GOD_RAYS_PARAMS_STRIDE] = {};
    std::memcpy(paramsBytes, &params, sizeof(params));
    rhi::writeBuffer(pCommandList, paramsBuffers[frameIndex], paramsBytes, sizeof(paramsBytes));

    AnnounceFrameImages(pCommandList, target);

    // The legacy dispatch arithmetic (GodRays.cpp:268-275 and :305-306): the trace covers the
    // shader's own half-resolution bound `((width + 1) / 2, (height + 1) / 2)`, the filter the full
    // render size; `Utils::GetWorkGroupCount` is `1 + ceil(size / 16)`, one workgroup beyond the
    // ceiling whose invocations the shaders' bounds checks discard.
    const uint32_t halfWidth = (width + 1) / 2;
    const uint32_t halfHeight = (height + 1) / 2;
    const uint32_t traceGroupsX = Utils::GetWorkGroupCount(halfWidth, GOD_RAYS_GROUP_SIZE);
    const uint32_t traceGroupsY = Utils::GetWorkGroupCount(halfHeight, GOD_RAYS_GROUP_SIZE);
    const uint32_t filterGroupsX = Utils::GetWorkGroupCount(width, GOD_RAYS_GROUP_SIZE);
    const uint32_t filterGroupsY = Utils::GetWorkGroupCount(height, GOD_RAYS_GROUP_SIZE);

    // The primary march (passIndex 0). On the clear path the shader writes zeros to every
    // half-resolution pixel and the filter then zeroes image 64 (the legacy shape,
    // VulkanDevice.cpp:964-969).
    const uint32_t passIndexPrimary = 0;
    RecordDispatch(pCommandList, tracePipeline,
                   { shadowSet, paramsSets[frameIndex], target.traceFramebufferSet,
                     target.uniformSet, blueNoiseSet },
                   traceGroupsX, traceGroupsY, 1, &passIndexPrimary);

    // The reflected-segment accumulation (passIndex 1), only when the host says the refl/refr
    // raygen ran and the shader is not on the clear path. It reads the result of pass 0 from the
    // same UAV; the explicit same-state requirement documents the write -> read ordering (the
    // backend also re-requires the UAV-binding set on every state set,
    // vulkan-state-tracking.cpp:111).
    if (params.godRaysEnabled != 0 && traceReflections)
    {
        RequireImageUnorderedAccess(pCommandList, target, IMAGE_GOD_RAYS);

        const uint32_t passIndexReflections = 1;
        RecordDispatch(pCommandList, tracePipeline,
                       { shadowSet, paramsSets[frameIndex], target.traceFramebufferSet,
                         target.uniformSet, blueNoiseSet },
                       traceGroupsX, traceGroupsY, 1, &passIndexReflections);
    }

    // The full-resolution bilateral upscale. The blob's framebuffer and uniform bindings live in
    // engine sets 2 and 3, so the two leading placeholder positions are filled with the empty set.
    RecordDispatch(pCommandList, filterPipeline,
                   { emptySet, emptySet, target.filterFramebufferSet, target.uniformSet },
                   filterGroupsX, filterGroupsY, 1, nullptr);

    // The sampled reads rest in a read-only layout now; move them back to UnorderedAccess, the
    // engine's GENERAL, so the frame ends with 63/64 in the state the engine and `CmPrepareFinal`
    // expect (64 is left there by its own write).
    for (uint32_t i = 0; i < GOD_RAYS_RESTORE_IMAGE_COUNT; i++)
    {
        RequireImageUnorderedAccess(pCommandList, target, GOD_RAYS_RESTORE_IMAGES[i]);
    }
}

void RhiRtGodRaysPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtGodRaysPass::ReleaseFramebufferSets(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the sets reference the wraps of engine images the GPU may still be reading. The queue
    // takes its reference now, so the handles below can be cleared immediately.
    if (frameContext != nullptr)
    {
        if (target.traceFramebufferSet != nullptr)
        {
            frameContext->Retire(target.traceFramebufferSet);
        }
        if (target.filterFramebufferSet != nullptr)
        {
            frameContext->Retire(target.filterFramebufferSet);
        }
    }

    target.traceFramebufferSet = nullptr;
    target.filterFramebufferSet = nullptr;
}

void RhiRtGodRaysPass::ReleaseFramebufferTarget(Target &target)
{
    ReleaseFramebufferSets(target);

    if (frameContext != nullptr)
    {
        for (nvrhi::TextureHandle &texture : target.engineTextures)
        {
            if (texture != nullptr)
            {
                frameContext->Retire(texture);
            }
        }
    }

    for (nvrhi::TextureHandle &texture : target.engineTextures)
    {
        texture = nullptr;
    }

    std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
    target.width = 0;
    target.height = 0;
}

void RhiRtGodRaysPass::ReleaseTarget(Target &target)
{
    ReleaseFramebufferTarget(target);

    if (frameContext != nullptr && target.uniformSet != nullptr)
    {
        frameContext->Retire(target.uniformSet);
    }

    target.uniformSet = nullptr;
    target.uniformBuffer = nullptr;
}

bool RhiRtGodRaysPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the god-rays pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
