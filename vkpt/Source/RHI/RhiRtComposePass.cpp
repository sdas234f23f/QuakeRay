#include "RhiRtComposePass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiTextureSource.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../Utils.h"

#include <cstring>
#include <string>
#include <tuple>

using namespace vkpt;

namespace
{

// The three engine blobs, by the file names the shader build writes into the folder the host passes
// in (ShaderManager.cpp:41-49) - the same `Cm*` names the legacy Q2Denoiser and ImageComposition
// pipelines load, and the blobs the reconnaissance measured. The adapter and the interleave are
// the `flt_enable = 0` half of the denoiser (Q2Denoiser.cpp:412-425, :600-621), the checkerboard is
// the resolve ImageComposition::ProcessCheckerboard records (ImageComposition.cpp:163-202).
const char *const ADAPTER_SHADER_FILE_NAME = "CmQ2Adapter.comp.spv";
const char *const INTERLEAVE_SHADER_FILE_NAME = "CmQ2Interleave.comp.spv";
const char *const CHECKERBOARD_SHADER_FILE_NAME = "CmCheckerboard.comp.spv";

// The workgroup of all three shaders (measured with `spirv-dis` over the three blobs:
// `OpExecutionMode %main LocalSize 16 16 1`; the sources are CmQ2Adapter.comp.hlsl:181,
// CmQ2Interleave.comp.hlsl:55 and CmCheckerboard.comp.hlsl:71). The dispatch itself is the
// legacy `Utils::GetWorkGroupCount(size, 16)` (Q2Denoiser.cpp:400-401, :606-607,
// ImageComposition.cpp:198-199).
constexpr uint32_t COMPOSE_GROUP_SIZE = 16;

// The sets of the three pipelines were measured 2026-09-25 with `spirv-dis` over the shipped blobs;
// the shader sources are the same three files. The engine's raw bindings are the identity for the
// storage images (`ShFramebuffers_Bindings[index] == index`) and `124 + index` for the sampled
// views (`ShFramebuffers_Sampled_Bindings`), so the layout offsets below turn a raw binding into
// its NVRHI slot: raw = offset + slot (RhiPipeline.h). The numbers are still derived from the
// generated arrays at run time (GetComposeSlot), so a generator change cannot drift from them.
constexpr uint32_t FRAMEBUFFER_UAV_OFFSET = 0;
constexpr uint32_t FRAMEBUFFER_SRV_OFFSET = 124;

// The union of the engine images the three passes touch, deduplicated: 23 images, each wrapped once
// per slot and shared by the three sets, so NVRHI's own UAV -> SRV -> UAV transitions on one wrap
// order the pass-to-pass hand-offs (Q2_COLOR 117 is written by the adapter and read by the
// interleave; PRE_FINAL 27 is written by the interleave and read by the checkerboard). The three
// UNFILTERED_INDIRECT_S_H images (16-18) are written by the A4.3 indirect pass earlier on the same
// command list (RhiRtIndirectPass), and the adapter samples them through the 140/141/142 bindings
// below like the rest of the union.
constexpr uint32_t COMPOSE_IMAGE_COUNT = 23;
constexpr FramebufferImageIndex COMPOSE_IMAGES[COMPOSE_IMAGE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,                //   0  framebufAlbedo           (adapter SRV)
    FB_IMAGE_INDEX_IS_SKY,                //   2  framebufIsSky            (adapter SRV)
    FB_IMAGE_INDEX_NORMAL,                //   3  framebufNormal           (adapter SRV)
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,    //   7  framebufMetallicRoughness (adapter SRV)
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,     //  14  framebufUnfilteredDirect  (adapter SRV)
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,   //  15  framebufUnfilteredSpecular (adapter SRV)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, //  16  framebufUnfilteredIndirectSH_R (adapter SRV; written by the indirect pass)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, //  17  framebufUnfilteredIndirectSH_G (adapter SRV; written by the indirect pass)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, //  18  framebufUnfilteredIndirectSH_B (adapter SRV; written by the indirect pass)
    FB_IMAGE_INDEX_THROUGHPUT,            //  26  framebufThroughput        (adapter/interleave/checkerboard SRV)
    FB_IMAGE_INDEX_PRE_FINAL,             //  27  framebufPreFinal          (interleave UAV, checkerboard SRV)
    FB_IMAGE_INDEX_FINAL,                 //  28  framebufFinal             (checkerboard UAV)
    FB_IMAGE_INDEX_ACID_FOG_R_T,          //  59  framebufAcidFogRT         (checkerboard SRV)
    FB_IMAGE_INDEX_ACID_FOG,              //  60  framebufAcidFog           (checkerboard UAV)
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,       //  61  framebufScreenEmisRT      (checkerboard SRV)
    FB_IMAGE_INDEX_SCREEN_EMISSION,       //  62  framebufScreenEmission    (checkerboard UAV)
    FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,      //  73  framebufQ2ColorLF_SH      (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G,  //  75  framebufQ2ColorLF_COCG    (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_H_F,          //  77  framebufQ2ColorHF         (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_SPEC,         //  79  framebufQ2ColorSpec       (adapter UAV)
    FB_IMAGE_INDEX_Q2_TRANSPARENT,        //  88  framebufQ2Transparent     (adapter SRV)
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,          //  90  framebufQ2FogAccum        (adapter SRV)
    FB_IMAGE_INDEX_Q2_COLOR,              // 117  framebufQ2Color           (adapter UAV, interleave SRV)
};

static_assert(COMPOSE_IMAGE_COUNT == 23,
              "the per-slot image arrays of RhiRtComposePass.h are sized 23 (COMPOSE_IMAGES)");

// One item of a pass's set 0: the engine image and the binding kind. 'isUAV' selects both the
// layout item and the set item type; 'image' also carries the raw binding, through the generated
// array of the matching kind.
struct ComposeBinding
{
    FramebufferImageIndex image;
    bool isUAV;
};

// CmQ2Adapter: 5 storage images and 12 sampled images (measured: exactly these 17 bindings, all in
// set 0; the module's own uniform layout carries set 1). The order is the reconnaissance's table
// order - the UAVs first, then the SRVs. 16-18 are sampled like the rest; the A4.3 indirect pass
// writes them earlier on the same command list.
constexpr uint32_t ADAPTER_BINDING_COUNT = 17;
constexpr ComposeBinding ADAPTER_BINDINGS[ADAPTER_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,     true  }, //  73  framebufQ2ColorLF_SH
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G, true  }, //  75  framebufQ2ColorLF_COCG
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,         true  }, //  77  framebufQ2ColorHF
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,        true  }, //  79  framebufQ2ColorSpec
    { FB_IMAGE_INDEX_Q2_COLOR,             true  }, // 117  framebufQ2Color
    { FB_IMAGE_INDEX_ALBEDO,               false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,               false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,               false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,   false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,    false }, // 138  framebufUnfilteredDirect_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,  false }, // 139  framebufUnfilteredSpecular_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, false }, // 140  framebufUnfilteredIndirectSH_R_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, false }, // 141  framebufUnfilteredIndirectSH_G_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, false }, // 142  framebufUnfilteredIndirectSH_B_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_TRANSPARENT,       false }, // 212  framebufQ2Transparent_Sampled
    { FB_IMAGE_INDEX_Q2_FOG_ACCUM,         false }, // 214  framebufQ2FogAccum_Sampled
};

// CmQ2Interleave: 1 storage image and 2 sampled images.
constexpr uint32_t INTERLEAVE_BINDING_COUNT = 3;
constexpr ComposeBinding INTERLEAVE_BINDINGS[INTERLEAVE_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_PRE_FINAL, true  }, //  27  framebufPreFinal
    { FB_IMAGE_INDEX_THROUGHPUT, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR,  false }, // 241  framebufQ2Color_Sampled
};

// CmCheckerboard: 3 storage images and 4 sampled images.
constexpr uint32_t CHECKERBOARD_BINDING_COUNT = 7;
constexpr ComposeBinding CHECKERBOARD_BINDINGS[CHECKERBOARD_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_FINAL,            true  }, //  28  framebufFinal
    { FB_IMAGE_INDEX_ACID_FOG,         true  }, //  60  framebufAcidFog
    { FB_IMAGE_INDEX_SCREEN_EMISSION,  true  }, //  62  framebufScreenEmission
    { FB_IMAGE_INDEX_THROUGHPUT,       false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_PRE_FINAL,        false }, // 151  framebufPreFinal_Sampled
    { FB_IMAGE_INDEX_ACID_FOG_R_T,     false }, // 183  framebufAcidFogRT_Sampled
    { FB_IMAGE_INDEX_SCREEN_EMIS_R_T,  false }, // 185  framebufScreenEmisRT_Sampled
};

// The engine raw binding of one entry: the storage-image array for the UAVs, the sampled array for
// the SRVs - the numbers the blobs' OpDecorate bindings carry.
uint32_t GetComposeRawBinding(const ComposeBinding &binding)
{
    return binding.isUAV ? ShFramebuffers_Bindings[binding.image]
                         : ShFramebuffers_Sampled_Bindings[binding.image];
}

// The NVRHI slot of one entry under the module's set-0 offsets.
uint32_t GetComposeSlot(const ComposeBinding &binding)
{
    return binding.isUAV ? GetComposeRawBinding(binding) - FRAMEBUFFER_UAV_OFFSET
                         : GetComposeRawBinding(binding) - FRAMEBUFFER_SRV_OFFSET;
}

constexpr uint32_t COMPOSE_IMAGE_NONE = COMPOSE_IMAGE_COUNT;

// The union-table position of an engine image, or COMPOSE_IMAGE_NONE when the image is not wrapped
// by the module (which no table entry may name).
constexpr uint32_t FindComposeImage(FramebufferImageIndex image)
{
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        if (COMPOSE_IMAGES[i] == image)
        {
            return i;
        }
    }

    return COMPOSE_IMAGE_NONE;
}

// The three pass tables have to agree with the union table item by item: every entry has to
// resolve to an image the module wraps. The static assertion below runs at compile time, so a
// hand-edited table cannot produce a half-filled binding set at run time.
constexpr bool ComposeTablesAreConsistent()
{
    const ComposeBinding *const tables[] =
    {
        ADAPTER_BINDINGS,
        INTERLEAVE_BINDINGS,
        CHECKERBOARD_BINDINGS,
    };
    const uint32_t counts[] =
    {
        ADAPTER_BINDING_COUNT,
        INTERLEAVE_BINDING_COUNT,
        CHECKERBOARD_BINDING_COUNT,
    };

    for (uint32_t table = 0; table < 3; table++)
    {
        for (uint32_t i = 0; i < counts[table]; i++)
        {
            if (FindComposeImage(tables[table][i].image) == COMPOSE_IMAGE_NONE)
            {
                return false;
            }
        }
    }

    return true;
}

constexpr bool AreComposeImagesDistinct()
{
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        for (uint32_t j = i + 1; j < COMPOSE_IMAGE_COUNT; j++)
        {
            if (COMPOSE_IMAGES[i] == COMPOSE_IMAGES[j])
            {
                return false;
            }
        }
    }

    return true;
}

// The images that end the chain in a sampled read. NVRHI moves a sampled image to
// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (vulkan-resource-bindings.cpp:398-435), while the engine
// leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL (= UnorderedAccess) and the next
// frame's primary/direct/indirect passes write most of these through their own wraps, so the
// compose pass has to move each one back after the checkerboard. The restore of 16-18 is what lets
// the next frame's indirect pass write them: its own UnorderedAccess announcement finds the
// read-only state and emits no transition. The 7 images left as the last pass's UAV outputs
// (73, 75, 77, 79, 28, 60, 62) already rest in GENERAL.
constexpr uint32_t COMPOSE_RESTORE_COUNT = 16;
constexpr FramebufferImageIndex COMPOSE_RESTORE_IMAGES[COMPOSE_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,
    FB_IMAGE_INDEX_IS_SKY,
    FB_IMAGE_INDEX_NORMAL,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R,
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G,
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B,
    FB_IMAGE_INDEX_THROUGHPUT,
    FB_IMAGE_INDEX_ACID_FOG_R_T,
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,
    FB_IMAGE_INDEX_Q2_TRANSPARENT,
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,
    FB_IMAGE_INDEX_PRE_FINAL,
    FB_IMAGE_INDEX_Q2_COLOR,
};

constexpr bool AreRestoreImagesWrapped()
{
    for (const FramebufferImageIndex image : COMPOSE_RESTORE_IMAGES)
    {
        if (FindComposeImage(image) == COMPOSE_IMAGE_NONE)
        {
            return false;
        }
    }

    return true;
}

static_assert(ComposeTablesAreConsistent(),
              "every compose binding has to resolve to an image in the compose image union");
static_assert(AreComposeImagesDistinct(), "the compose image union must not repeat an image");
static_assert(AreRestoreImagesWrapped(), "every restored image has to be in the compose image union");

// The union position of FINAL, the image the present samples (GetFinalTexture).
constexpr uint32_t FINAL_IMAGE_SLOT = FindComposeImage(FB_IMAGE_INDEX_FINAL);
static_assert(FINAL_IMAGE_SLOT != COMPOSE_IMAGE_NONE, "FINAL has to be part of the compose image union");

// The exact set-0 layout of one pass: Compute visibility (the RT passes' AllRayTracing layouts have
// no Compute bit and the pinned backend would skip them for a compute pipeline,
// validation-device.cpp:1002-1003), one item per declared raw binding, the block's offsets.
nvrhi::BindingLayoutHandle CreateFramebufferLayout(nvrhi::IDevice *device,
                                                   const ComposeBinding *pBindings,
                                                   uint32_t count)
{
    nvrhi::BindingLayoutDesc desc;
    desc.visibility = nvrhi::ShaderType::Compute;
    desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                               .setShaderResourceOffset(FRAMEBUFFER_SRV_OFFSET)
                               .setUnorderedAccessViewOffset(FRAMEBUFFER_UAV_OFFSET));

    for (uint32_t i = 0; i < count; i++)
    {
        desc.addItem(pBindings[i].isUAV
                         ? nvrhi::BindingLayoutItem::Texture_UAV(GetComposeSlot(pBindings[i]))
                         : nvrhi::BindingLayoutItem::Texture_SRV(GetComposeSlot(pBindings[i])));
    }

    return device->createBindingLayout(desc);
}

// One compute pipeline over the pass's set-0 layout and the module's shared uniform layout. The
// list order is the descriptor-set number the backend binds positionally
// (vulkan-resource-bindings.cpp:940-1019), so set 0 is the pass's framebuffers and set 1 the
// uniform.
nvrhi::ComputePipelineHandle CreateComposePipeline(nvrhi::IDevice *device,
                                                   nvrhi::IShader *pShader,
                                                   nvrhi::IBindingLayout *pFramebufferLayout,
                                                   nvrhi::IBindingLayout *pUniformLayout)
{
    if (device == nullptr || pShader == nullptr || pFramebufferLayout == nullptr ||
        pUniformLayout == nullptr)
    {
        return nullptr;
    }

    nvrhi::ComputePipelineDesc desc;
    desc.setComputeShader(pShader);
    desc.addBindingLayout(pFramebufferLayout);
    desc.addBindingLayout(pUniformLayout);

    return rhi::createComputePipeline(device, desc, "RhiRtComposePass");
}

// The set over one exact layout. Every entry takes the slot's wrap of that engine image. Returns
// null when an entry cannot be filled, which the module's static assertions make unreachable for
// the shipped tables.
nvrhi::BindingSetHandle CreateFramebufferSet(nvrhi::IDevice *device,
                                             const ComposeBinding *pBindings,
                                             uint32_t count,
                                             const nvrhi::TextureHandle *pEngineTextures,
                                             nvrhi::IBindingLayout *pLayout)
{
    if (device == nullptr || pLayout == nullptr)
    {
        return nullptr;
    }

    nvrhi::BindingSetDesc setDesc;

    for (uint32_t i = 0; i < count; i++)
    {
        const uint32_t imageSlot = FindComposeImage(pBindings[i].image);
        nvrhi::ITexture *texture =
            imageSlot != COMPOSE_IMAGE_NONE ? pEngineTextures[imageSlot].Get() : nullptr;

        if (texture == nullptr)
        {
            return nullptr;
        }

        setDesc.addItem(pBindings[i].isUAV
                            ? nvrhi::BindingSetItem::Texture_UAV(GetComposeSlot(pBindings[i]), texture)
                            : nvrhi::BindingSetItem::Texture_SRV(GetComposeSlot(pBindings[i]), texture));
    }

    return device->createBindingSet(setDesc, pLayout);
}

void LogMessage(const RhiRtComposePass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtComposePass::RhiRtComposePass() = default;

RhiRtComposePass::~RhiRtComposePass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images; the host destroys the pass while it can still idle
        // the device (VulkanDevice does that before the skeleton as well), so nothing has to go
        // through a retire queue here.
        device->waitForIdle();
    }

    for (Target &target : targets)
    {
        target.adapterSet = nullptr;
        target.interleaveSet = nullptr;
        target.checkerboardSet = nullptr;

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

    checkerboardPipeline = nullptr;
    interleavePipeline = nullptr;
    adapterPipeline = nullptr;
    uniformLayout = nullptr;
    checkerboardFramebufferLayout = nullptr;
    interleaveFramebufferLayout = nullptr;
    adapterFramebufferLayout = nullptr;
    checkerboardShader = nullptr;
    interleaveShader = nullptr;
    adapterShader = nullptr;
}

bool RhiRtComposePass::Create(nvrhi::IDevice *pDevice,
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
        LogMessage(print, "Warning: RHI: the compose pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the compose pass needs the frame context of the RHI layer");
        return false;
    }

    // Each pipeline declares two binding layouts (its set 0 and the shared uniform), so the pinned
    // NVRHI's binding-layout cap does not matter here and the module has no
    // `c_MaxBindingLayouts` guard: the RT passes' twelve-layout pipelines are the only users of
    // the raised cap (third_party/nvrhi-max-binding-layouts.patch).

    if (!LoadShader(ADAPTER_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, adapterShader) ||
        !LoadShader(INTERLEAVE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, interleaveShader) ||
        !LoadShader(CHECKERBOARD_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, checkerboardShader))
    {
        return false;
    }

    // The three exact set-0 layouts: 17, 3 and 7 items, Compute visibility, SRV offset 124 and UAV
    // offset 0. A union layout or a shared set would fail the exact-set rule
    // (validation-device.cpp:1855-1871), and the primary pass's AllRayTracing uniform layout would
    // leave the compute stage undeclared, so each pass owns its own layout handle here.
    adapterFramebufferLayout =
        CreateFramebufferLayout(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT);
    interleaveFramebufferLayout =
        CreateFramebufferLayout(device, INTERLEAVE_BINDINGS, INTERLEAVE_BINDING_COUNT);
    checkerboardFramebufferLayout =
        CreateFramebufferLayout(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT);

    // Set 1 for all three pipelines: the engine's global uniform at raw binding 0
    // (ShaderCommonHLSL.hlsli:104 spells it with BINDING_GLOBAL_UNIFORM and DESC_SET_GLOBAL_UNIFORM).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM));

        uniformLayout = device->createBindingLayout(desc);
    }

    if (adapterFramebufferLayout == nullptr || interleaveFramebufferLayout == nullptr ||
        checkerboardFramebufferLayout == nullptr || uniformLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass binding layout");
        return false;
    }

    adapterPipeline = CreateComposePipeline(device, adapterShader, adapterFramebufferLayout, uniformLayout);
    interleavePipeline =
        CreateComposePipeline(device, interleaveShader, interleaveFramebufferLayout, uniformLayout);
    checkerboardPipeline =
        CreateComposePipeline(device, checkerboardShader, checkerboardFramebufferLayout, uniformLayout);

    if (adapterPipeline == nullptr || interleavePipeline == nullptr || checkerboardPipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass compute pipeline");
        return false;
    }

    created = true;
    return true;
}

void RhiRtComposePass::Render(nvrhi::ICommandList *pCommandList,
                              uint32_t frameIndex,
                              const Framebuffers *pFramebuffers,
                              uint32_t width,
                              uint32_t height,
                              nvrhi::IBuffer *pUniformBuffer)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    if (width == 0 || height == 0)
    {
        return;
    }

    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the compose pass got no engine framebuffers, the compose is skipped");
        }
        return;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the compose pass got no global uniform, the compose is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // The 23 images of this slot. `GetImageHandles` resolves the engine's ping-pong swap
    // (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's current
    // image, which is the one the engine's own per-slot descriptor set binds to the variable's
    // fixed raw binding - the shader never sees the swap. The 4-tuple overload supplies each
    // image's extent, which every image here has to answer with the render size: all 23 are
    // render-sized by their generated flags (none carries a FORCE_SIZE/UPSCALED/SINGLE_PIXEL flag),
    // and the shaders access them at `globalUniform.renderWidth/renderHeight`, so a different
    // extent is a wrongly sized wrap and the compose is skipped instead of sampling outside the
    // image.
    const ResolutionState resolutionState = { width, height, width, height };

    uint64_t imageHandles[COMPOSE_IMAGE_COUNT] = {};
    uint64_t imageViews[COMPOSE_IMAGE_COUNT] = {};
    VkFormat imageFormats[COMPOSE_IMAGE_COUNT] = {};
    VkExtent2D imageExtents[COMPOSE_IMAGE_COUNT] = {};

    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        const auto [image, view, format, extent] =
            pFramebuffers->GetImageHandles(COMPOSE_IMAGES[i], frameIndex, resolutionState);

        imageHandles[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));
        imageViews[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        imageFormats[i] = format;
        imageExtents[i] = extent;

        if (imageHandles[i] == 0)
        {
            if (!warnedMissingFramebuffers)
            {
                warnedMissingFramebuffers = true;
                LogMessage(print, "Warning: RHI: the compose pass got no framebuffer image, the compose is skipped");
            }
            return;
        }

        if (imageExtents[i].width != width || imageExtents[i].height != height)
        {
            if (!warnedUnexpectedSize)
            {
                warnedUnexpectedSize = true;
                LogMessage(print, std::string("Warning: RHI: the compose pass got \"") +
                                      ShFramebuffers_DebugNames[COMPOSE_IMAGES[i]] +
                                      "\", which is not render-sized; the compose is skipped");
            }
            return;
        }
    }

    bool framebuffersChanged = target.width != width || target.height != height;
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        framebuffersChanged = framebuffersChanged || target.imageHandles[i] != imageHandles[i];
    }

    if (framebuffersChanged)
    {
        // The engine re-created its framebuffers (a resize or an image-format change): the wraps
        // and the sets over them are retired and rebuilt. Retiring first is safe even if the
        // rebuild below fails - the next frame retries, and the compose is skipped until it
        // succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtComposePass ") +
                                          ShFramebuffers_DebugNames[COMPOSE_IMAGES[i]] +
                                          " frame " + std::to_string(frameIndex);

            target.engineTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i], width, height, debugName);

            if (target.engineTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap a compose pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
    }

    if (!PrepareFramebufferSets(target) || !PrepareUniformSet(target, pUniformBuffer))
    {
        return;
    }

    // The image state contract, spelled out on the class: the engine leaves every framebuffer image
    // in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap keeps no state
    // between command lists (RhiTextureSource.h), so every list announces that state for all 23
    // images before the first use. The primary, direct and indirect passes wrote some of them on
    // this very list through their own wraps; the physical layout they left them in is exactly this
    // GENERAL state, so the announcement is the truth and the SRV bindings' automatic transitions
    // start from it.
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.engineTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    // The legacy dispatch arithmetic, all three passes over the render resolution
    // (Q2Denoiser.cpp:400-401, :606-607, ImageComposition.cpp:198-199). `Utils::GetWorkGroupCount`
    // is `1 + ceil(size / 16)`, one workgroup beyond the ceiling; the shaders' bounds check on
    // `globalUniform.renderWidth/renderHeight` discards those invocations.
    const uint32_t groupsX = Utils::GetWorkGroupCount(width, COMPOSE_GROUP_SIZE);
    const uint32_t groupsY = Utils::GetWorkGroupCount(height, COMPOSE_GROUP_SIZE);

    // The legacy order: the adapter composites into Q2_COLOR, the interleave resolves it into
    // PRE_FINAL, the checkerboard writes FINAL and its two companions (Q2Denoiser.cpp:412-436 then
    // :600-621, ImageComposition.cpp:93-98). The shared wraps make the two hand-offs ordinary
    // automatic-barrier transitions.
    RecordDispatch(pCommandList, adapterPipeline, target.adapterSet, target.uniformSet, groupsX, groupsY);
    RecordDispatch(pCommandList, interleavePipeline, target.interleaveSet, target.uniformSet, groupsX, groupsY);
    RecordDispatch(pCommandList, checkerboardPipeline, target.checkerboardSet, target.uniformSet, groupsX, groupsY);

    // The 16 images whose last use was a sampled read rest in the read-only layout now; move them
    // back to UnorderedAccess, the engine's GENERAL, so the next frame's primary/direct/indirect
    // UAV writes start from the state their own announcements claim. The restore includes 16-18:
    // without it the next frame's indirect UAV write would run against a read-only image. The last
    // pass's three UAV images (28, 60, 62) and the adapter's four channel UAVs already end in
    // GENERAL.
    for (uint32_t i = 0; i < COMPOSE_RESTORE_COUNT; i++)
    {
        const uint32_t imageSlot = FindComposeImage(COMPOSE_RESTORE_IMAGES[i]);

        if (imageSlot != COMPOSE_IMAGE_NONE)
        {
            pCommandList->setTextureState(target.engineTextures[imageSlot], nvrhi::AllSubresources,
                                          nvrhi::ResourceStates::UnorderedAccess);
        }
    }
}

nvrhi::ITexture *RhiRtComposePass::GetFinalTexture(uint32_t frameIndex) const
{
    if (!created || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    return targets[frameIndex].engineTextures[FINAL_IMAGE_SLOT].Get();
}

void RhiRtComposePass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtComposePass::ReleaseFramebufferTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the wraps reference engine images the GPU may still be reading, and the sets reference
    // those wraps. The queue takes its reference now, so the handles below can be cleared
    // immediately.
    if (frameContext != nullptr)
    {
        if (target.adapterSet != nullptr)
        {
            frameContext->Retire(target.adapterSet);
        }
        if (target.interleaveSet != nullptr)
        {
            frameContext->Retire(target.interleaveSet);
        }
        if (target.checkerboardSet != nullptr)
        {
            frameContext->Retire(target.checkerboardSet);
        }

        for (nvrhi::TextureHandle &texture : target.engineTextures)
        {
            if (texture != nullptr)
            {
                frameContext->Retire(texture);
            }
        }
    }

    target.adapterSet = nullptr;
    target.interleaveSet = nullptr;
    target.checkerboardSet = nullptr;

    for (nvrhi::TextureHandle &texture : target.engineTextures)
    {
        texture = nullptr;
    }

    std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
    target.width = 0;
    target.height = 0;
}

void RhiRtComposePass::ReleaseTarget(Target &target)
{
    ReleaseFramebufferTarget(target);

    if (frameContext != nullptr && target.uniformSet != nullptr)
    {
        frameContext->Retire(target.uniformSet);
    }

    target.uniformSet = nullptr;
    target.uniformBuffer = nullptr;
}

bool RhiRtComposePass::PrepareFramebufferSets(Target &target)
{
    if (target.adapterSet == nullptr)
    {
        target.adapterSet = CreateFramebufferSet(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT,
                                                 target.engineTextures,
                                                 adapterFramebufferLayout);

        if (target.adapterSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass adapter binding set");
            }
            return false;
        }
    }

    if (target.interleaveSet == nullptr)
    {
        target.interleaveSet = CreateFramebufferSet(device, INTERLEAVE_BINDINGS, INTERLEAVE_BINDING_COUNT,
                                                    target.engineTextures,
                                                    interleaveFramebufferLayout);

        if (target.interleaveSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass interleave binding set");
            }
            return false;
        }
    }

    if (target.checkerboardSet == nullptr)
    {
        target.checkerboardSet = CreateFramebufferSet(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT,
                                                      target.engineTextures,
                                                      checkerboardFramebufferLayout);

        if (target.checkerboardSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass checkerboard binding set");
            }
            return false;
        }
    }

    return true;
}

bool RhiRtComposePass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformSet != nullptr && target.uniformBuffer == pUniformBuffer)
    {
        return true;
    }

    // The shader declares set 1 as ConstantBuffer<ShGlobalUniform>; the validation device refuses
    // such a binding on a buffer whose desc lacks isConstantBuffer (validation-device.cpp:1717-1723)
    // and on a volatile one (:1725-1730, which would also become a dynamic-offset binding the
    // static layout item cannot take).
    if (!pUniformBuffer->getDesc().isConstantBuffer || pUniformBuffer->getDesc().isVolatile)
    {
        if (!warnedBadUniform)
        {
            warnedBadUniform = true;
            LogMessage(print, "Warning: RHI: the compose pass needs the global uniform as a static constant-buffer wrap");
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
        LogMessage(print, "Warning: RHI: failed to create the compose pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

void RhiRtComposePass::RecordDispatch(nvrhi::ICommandList *pCommandList,
                                      nvrhi::IComputePipeline *pPipeline,
                                      nvrhi::IBindingSet *pFramebufferSet,
                                      nvrhi::IBindingSet *pUniformSet,
                                      uint32_t groupsX,
                                      uint32_t groupsY)
{
    nvrhi::ComputeState state;
    state.setPipeline(pPipeline);
    state.addBindingSet(pFramebufferSet); // set 0: the pass's framebuffers
    state.addBindingSet(pUniformSet);     // set 1: the engine's global uniform

    pCommandList->setComputeState(state);
    pCommandList->dispatch(groupsX, groupsY, 1);
}

bool RhiRtComposePass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the compose pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
