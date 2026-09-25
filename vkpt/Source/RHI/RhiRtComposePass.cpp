#include "RhiRtComposePass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../Tonemapping.h"
#include "../Utils.h"

#include <cstring>
#include <initializer_list>
#include <string>
#include <tuple>

using namespace vkpt;

namespace
{

// The six engine blobs, by the file names the shader build writes into the folder the host passes
// in (ShaderManager.cpp:41-49) - the same `Cm*` names the legacy Q2Denoiser, Tonemapping and
// ImageComposition pipelines load, and the blobs the reconnaissance measured. The adapter and the
// interleave are the `flt_enable = 0` half of the denoiser (Q2Denoiser.cpp:412-425, :600-621), the
// histogram and the average are the engine's exposure chain (Tonemapping.cpp:171-208), the
// checkerboard is the resolve ImageComposition::ProcessCheckerboard records
// (ImageComposition.cpp:163-202), and the final composition is
// ImageComposition::ApplyTonemapping (ImageComposition.cpp:113-161, called from
// ImageComposition::Finalize).
const char *const ADAPTER_SHADER_FILE_NAME = "CmQ2Adapter.comp.spv";
const char *const INTERLEAVE_SHADER_FILE_NAME = "CmQ2Interleave.comp.spv";
const char *const HISTOGRAM_SHADER_FILE_NAME = "CmLuminanceHistogram.comp.spv";
const char *const AVERAGE_SHADER_FILE_NAME = "CmLuminanceAvg.comp.spv";
const char *const CHECKERBOARD_SHADER_FILE_NAME = "CmCheckerboard.comp.spv";
const char *const PREPARE_FINAL_SHADER_FILE_NAME = "CmPrepareFinal.comp.spv";

// The workgroup of five of the six shaders (measured with `spirv-dis` over the blobs:
// `OpExecutionMode %main LocalSize 16 16 1`; the sources are CmQ2Adapter.comp.hlsl:181,
// CmQ2Interleave.comp.hlsl:55, CmLuminanceHistogram.comp.hlsl, CmCheckerboard.comp.hlsl:71 and
// CmPrepareFinal.comp.hlsl:420). The dispatch is the legacy `Utils::GetWorkGroupCount(size, 16)`
// (Q2Denoiser.cpp:400-401, :606-607, ImageComposition.cpp:198-199, Tonemapping.cpp:176-179).
// CmLuminanceAvg is the exception: `LocalSize 128 1 1` and a single workgroup
// (Tonemapping.cpp:205-208), dispatched with 1 x 1 x 1 below.
constexpr uint32_t COMPOSE_GROUP_SIZE = 16;

// The sets of the six pipelines were measured 2026-09-25 with `spirv-dis` over the shipped blobs;
// the shader sources are the same files. The engine's raw bindings are the identity for the
// storage images (`ShFramebuffers_Bindings[index] == index`) and `124 + index` for the sampled
// views (`ShFramebuffers_Sampled_Bindings`), so the layout offsets below turn a raw binding into
// its NVRHI slot: raw = offset + slot (RhiPipeline.h). The numbers are still derived from the
// generated arrays at run time (GetComposeSlot), so a generator change cannot drift from them.
constexpr uint32_t FRAMEBUFFER_UAV_OFFSET = 0;
constexpr uint32_t FRAMEBUFFER_SRV_OFFSET = 124;

// The union of the engine images the six passes touch, deduplicated: 27 images, each wrapped once
// per slot and shared by the five framebuffer sets, so NVRHI's own UAV -> SRV -> UAV transitions
// on one wrap order the pass-to-pass hand-offs (Q2_COLOR 117 is written by the adapter and read by
// the interleave; PRE_FINAL 27 is written by the interleave and read by the exposure histogram and
// the checkerboard; ACID_FOG 60 and SCREEN_EMISSION 62 are written by the checkerboard and read by
// the final composition). The three UNFILTERED_INDIRECT_S_H images (16-18) are written by the A4.3
// indirect pass earlier on the same command list (RhiRtIndirectPass), and the adapter samples them
// through the 140/141/142 bindings below like the rest of the union. Image 64 GOD_RAYS_FILTERED is
// deliberately *not* here: its only read binds the module's zero stand-in instead (see the
// class comment), because nothing writes the engine image under `rhiframe`.
constexpr uint32_t COMPOSE_IMAGE_COUNT = 27;
constexpr FramebufferImageIndex COMPOSE_IMAGES[COMPOSE_IMAGE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,                //   0  framebufAlbedo           (adapter/final SRV)
    FB_IMAGE_INDEX_IS_SKY,                //   2  framebufIsSky            (adapter SRV)
    FB_IMAGE_INDEX_NORMAL,                //   3  framebufNormal           (adapter/final SRV)
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,    //   7  framebufMetallicRoughness (adapter SRV)
    FB_IMAGE_INDEX_DEPTH_WORLD,           //   9  framebufDepthWorld       (final SRV)
    FB_IMAGE_INDEX_MOTION,                //  13  framebufMotion           (final SRV)
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,     //  14  framebufUnfilteredDirect  (adapter/final SRV)
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,   //  15  framebufUnfilteredSpecular (adapter/final SRV)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, //  16  framebufUnfilteredIndirectSH_R (adapter SRV; final UAV; written by the indirect pass)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, //  17  framebufUnfilteredIndirectSH_G (adapter SRV; final UAV; written by the indirect pass)
    FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, //  18  framebufUnfilteredIndirectSH_B (adapter SRV; final UAV; written by the indirect pass)
    FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,  //  25  framebufPrimaryToReflRefr (final SRV)
    FB_IMAGE_INDEX_THROUGHPUT,            //  26  framebufThroughput        (adapter/interleave/checkerboard/final SRV)
    FB_IMAGE_INDEX_PRE_FINAL,             //  27  framebufPreFinal          (interleave UAV, histogram/checkerboard SRV)
    FB_IMAGE_INDEX_FINAL,                 //  28  framebufFinal             (checkerboard/final UAV)
    FB_IMAGE_INDEX_ACID_FOG_R_T,          //  59  framebufAcidFogRT         (checkerboard SRV)
    FB_IMAGE_INDEX_ACID_FOG,              //  60  framebufAcidFog           (checkerboard UAV, final SRV)
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,       //  61  framebufScreenEmisRT      (checkerboard SRV)
    FB_IMAGE_INDEX_SCREEN_EMISSION,       //  62  framebufScreenEmission    (checkerboard UAV, final SRV)
    FB_IMAGE_INDEX_BLOOM_INPUT,           //  65  framebufBloomInput        (final UAV; bloom is off)
    FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,      //  73  framebufQ2ColorLF_SH      (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G,  //  75  framebufQ2ColorLF_COCG    (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_H_F,          //  77  framebufQ2ColorHF         (adapter UAV)
    FB_IMAGE_INDEX_Q2_COLOR_SPEC,         //  79  framebufQ2ColorSpec       (adapter UAV)
    FB_IMAGE_INDEX_Q2_TRANSPARENT,        //  88  framebufQ2Transparent     (adapter SRV)
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,          //  90  framebufQ2FogAccum        (adapter SRV)
    FB_IMAGE_INDEX_Q2_COLOR,              // 117  framebufQ2Color           (adapter UAV, interleave SRV)
};

static_assert(COMPOSE_IMAGE_COUNT == 27,
              "the per-slot image arrays of RhiRtComposePass.h are sized 27 (COMPOSE_IMAGES)");

// One item of a pass's set 0: the engine image, the binding kind, and whether the module
// substitutes its god-rays zero stand-in for that binding. 'isUAV' selects both the layout item and
// the set item type; 'image' also carries the raw binding, through the generated array of the
// matching kind (for the stand-in entry the raw binding is still the engine image's, 188).
struct ComposeBinding
{
    FramebufferImageIndex image;
    bool isUAV;
    bool isGodRaysStandIn;
};

// CmQ2Adapter: 5 storage images and 12 sampled images (measured: exactly these 17 bindings, all in
// set 0; the module's own uniform layout carries set 1). The order is the reconnaissance's table
// order - the UAVs first, then the SRVs. 16-18 are sampled like the rest; the A4.3 indirect pass
// writes them earlier on the same command list.
constexpr uint32_t ADAPTER_BINDING_COUNT = 17;
constexpr ComposeBinding ADAPTER_BINDINGS[ADAPTER_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,     true,  false }, //  73  framebufQ2ColorLF_SH
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G, true,  false }, //  75  framebufQ2ColorLF_COCG
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,         true,  false }, //  77  framebufQ2ColorHF
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,        true,  false }, //  79  framebufQ2ColorSpec
    { FB_IMAGE_INDEX_Q2_COLOR,             true,  false }, // 117  framebufQ2Color
    { FB_IMAGE_INDEX_ALBEDO,               false, false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,               false, false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,               false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,   false, false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,    false, false }, // 138  framebufUnfilteredDirect_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,  false, false }, // 139  framebufUnfilteredSpecular_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, false, false }, // 140  framebufUnfilteredIndirectSH_R_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, false, false }, // 141  framebufUnfilteredIndirectSH_G_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, false, false }, // 142  framebufUnfilteredIndirectSH_B_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_TRANSPARENT,       false, false }, // 212  framebufQ2Transparent_Sampled
    { FB_IMAGE_INDEX_Q2_FOG_ACCUM,         false, false }, // 214  framebufQ2FogAccum_Sampled
};

// CmQ2Interleave: 1 storage image and 2 sampled images.
constexpr uint32_t INTERLEAVE_BINDING_COUNT = 3;
constexpr ComposeBinding INTERLEAVE_BINDINGS[INTERLEAVE_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_PRE_FINAL, true,  false }, //  27  framebufPreFinal
    { FB_IMAGE_INDEX_THROUGHPUT, false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR,  false, false }, // 241  framebufQ2Color_Sampled
};

// CmLuminanceHistogram: 1 sampled image and no storage image - the exposure histogram only reads
// the interleave's PRE_FINAL (the 128-bin `groupshared` histogram is private to the workgroup).
constexpr uint32_t HISTOGRAM_BINDING_COUNT = 1;
constexpr ComposeBinding HISTOGRAM_BINDINGS[HISTOGRAM_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_PRE_FINAL, false, false }, // 151  framebufPreFinal_Sampled
};

// CmCheckerboard: 3 storage images and 4 sampled images.
constexpr uint32_t CHECKERBOARD_BINDING_COUNT = 7;
constexpr ComposeBinding CHECKERBOARD_BINDINGS[CHECKERBOARD_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_FINAL,            true,  false }, //  28  framebufFinal
    { FB_IMAGE_INDEX_ACID_FOG,         true,  false }, //  60  framebufAcidFog
    { FB_IMAGE_INDEX_SCREEN_EMISSION,  true,  false }, //  62  framebufScreenEmission
    { FB_IMAGE_INDEX_THROUGHPUT,       false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_PRE_FINAL,        false, false }, // 151  framebufPreFinal_Sampled
    { FB_IMAGE_INDEX_ACID_FOG_R_T,     false, false }, // 183  framebufAcidFogRT_Sampled
    { FB_IMAGE_INDEX_SCREEN_EMIS_R_T,  false, false }, // 185  framebufScreenEmisRT_Sampled
};

// CmPrepareFinal: 5 storage images and 11 sampled images (measured: 17 set-0 items in the shipped
// blob, of which the FINAL sampled view at raw 152 is the pair the A4.4 S1 shader fix removes; see
// the class comment). Unlike the other four tables the UAVs come last: if a future edit ever binds
// one image both ways in this set, the last requirement applied would be the UAV's and the image
// would end the chain in GENERAL, which the engine's convention and the next frame's writes need;
// today no image is bound both ways here. The god-rays entry binds the module's zero stand-in
// instead of the engine image 64 (the A4.2b zero-texture mechanism): the image has no writer under
// `rhiframe` and the engine leaves the framebuffer images undefined, so binding it would add stale
// light to every frame.
constexpr uint32_t PREPARE_FINAL_BINDING_COUNT = 16;
constexpr ComposeBinding PREPARE_FINAL_BINDINGS[PREPARE_FINAL_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_ALBEDO,               false, false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_NORMAL,               false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_DEPTH_WORLD,          false, false }, // 133  framebufDepthWorld_Sampled
    { FB_IMAGE_INDEX_MOTION,               false, false }, // 137  framebufMotion_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,    false, false }, // 138  framebufUnfilteredDirect_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,  false, false }, // 139  framebufUnfilteredSpecular_Sampled
    { FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR, false, false }, // 149  framebufPrimaryToReflRefr_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_ACID_FOG,             false, false }, // 184  framebufAcidFog_Sampled
    { FB_IMAGE_INDEX_SCREEN_EMISSION,      false, false }, // 186  framebufScreenEmission_Sampled
    { FB_IMAGE_INDEX_GOD_RAYS_FILTERED,    false, true  }, // 188  framebufGodRaysFiltered_Sampled -> zero stand-in
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, true, false }, //  16  framebufUnfilteredIndirectSH_R
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, true, false }, //  17  framebufUnfilteredIndirectSH_G
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, true, false }, //  18  framebufUnfilteredIndirectSH_B
    { FB_IMAGE_INDEX_FINAL,                true,  false }, //  28  framebufFinal
    { FB_IMAGE_INDEX_BLOOM_INPUT,          true,  false }, //  65  framebufBloomInput
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
// by the module (which is legal only for the god-rays stand-in entry).
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

// The five pass tables have to agree with the union table item by item: every engine-bound entry
// has to resolve to an image the module wraps, and the stand-in entry has to be the one binding
// over the one image that is deliberately not wrapped. The static assertion below runs at compile
// time, so a hand-edited table cannot produce a half-filled binding set at run time.
constexpr bool ComposeTablesAreConsistent()
{
    const ComposeBinding *const tables[] =
    {
        ADAPTER_BINDINGS,
        INTERLEAVE_BINDINGS,
        HISTOGRAM_BINDINGS,
        CHECKERBOARD_BINDINGS,
        PREPARE_FINAL_BINDINGS,
    };
    const uint32_t counts[] =
    {
        ADAPTER_BINDING_COUNT,
        INTERLEAVE_BINDING_COUNT,
        HISTOGRAM_BINDING_COUNT,
        CHECKERBOARD_BINDING_COUNT,
        PREPARE_FINAL_BINDING_COUNT,
    };

    for (uint32_t table = 0; table < 5; table++)
    {
        for (uint32_t i = 0; i < counts[table]; i++)
        {
            const bool wrapped = FindComposeImage(tables[table][i].image) != COMPOSE_IMAGE_NONE;

            if (tables[table][i].isGodRaysStandIn)
            {
                // The substitute replaces exactly one sampled read of image 64, the god-rays
                // filtered image the module does not wrap. Any other stand-in entry is a bug.
                if (wrapped || tables[table][i].isUAV ||
                    tables[table][i].image != FB_IMAGE_INDEX_GOD_RAYS_FILTERED)
                {
                    return false;
                }
            }
            else if (!wrapped)
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

// The images that end the chain in a sampled read. The last pass that touches an image decides its
// state: NVRHI moves every sampled image of a bound set to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
// (vulkan-resource-bindings.cpp:398-435) and applies the requirements of every binding item, used
// by the shader or not (vulkan-state-tracking.cpp:29-63) - so the final composition's ten engine
// SRVs (the eleventh is the module's stand-in) and the histogram's PRE_FINAL read decide, not the
// shader's runtime branches. The engine leaves
// every framebuffer image in VK_IMAGE_LAYOUT_GENERAL (= UnorderedAccess) and the next frame's
// primary/direct/indirect passes write most of these through their own wraps, so the compose pass
// has to move each one back after the final composition. The images that end the chain as the last
// pass's UAV outputs (16-18, 28, 65) or as the adapter's UAVs (73, 75, 77, 79) already rest in
// GENERAL; the restore of 16-18 is not needed any more because the final composition's UAV items
// are their last use. The restore of PRE_FINAL (27) and Q2_COLOR (117) is what lets the next
// frame's interleave and adapter writes start from GENERAL.
constexpr uint32_t COMPOSE_RESTORE_COUNT = 18;
constexpr FramebufferImageIndex COMPOSE_RESTORE_IMAGES[COMPOSE_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,
    FB_IMAGE_INDEX_IS_SKY,
    FB_IMAGE_INDEX_NORMAL,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,
    FB_IMAGE_INDEX_DEPTH_WORLD,
    FB_IMAGE_INDEX_MOTION,
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,
    FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,
    FB_IMAGE_INDEX_THROUGHPUT,
    FB_IMAGE_INDEX_PRE_FINAL,
    FB_IMAGE_INDEX_ACID_FOG_R_T,
    FB_IMAGE_INDEX_ACID_FOG,
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,
    FB_IMAGE_INDEX_SCREEN_EMISSION,
    FB_IMAGE_INDEX_Q2_TRANSPARENT,
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,
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
              "every compose binding has to resolve to a union image or be the god-rays stand-in");
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

// One compute pipeline over the pass's layouts, in descriptor-set order: the list order is the
// descriptor-set number the backend binds positionally (vulkan-resource-bindings.cpp:940-1019), so
// a pass's framebuffer set is first, the shared uniform set second, and the tonemapping, empty and
// volumetric sets follow where the pass declares them.
nvrhi::ComputePipelineHandle CreateComposePipeline(nvrhi::IDevice *device,
                                                   nvrhi::IShader *pShader,
                                                   std::initializer_list<nvrhi::IBindingLayout *> layouts)
{
    if (device == nullptr || pShader == nullptr || layouts.size() == 0)
    {
        return nullptr;
    }

    nvrhi::ComputePipelineDesc desc;
    desc.setComputeShader(pShader);

    for (nvrhi::IBindingLayout *pLayout : layouts)
    {
        if (pLayout == nullptr)
        {
            return nullptr;
        }

        desc.addBindingLayout(pLayout);
    }

    return rhi::createComputePipeline(device, desc, "RhiRtComposePass");
}

// The set over one exact layout. Every engine-bound entry takes the slot's wrap of that image; the
// stand-in entry takes the module's god-rays zero texture. Returns null when an entry cannot be
// filled, which the module's static assertions and the Render-time stand-in check make unreachable
// for the shipped tables.
nvrhi::BindingSetHandle CreateFramebufferSet(nvrhi::IDevice *device,
                                             const ComposeBinding *pBindings,
                                             uint32_t count,
                                             const nvrhi::TextureHandle *pEngineTextures,
                                             nvrhi::ITexture *pGodRaysStandIn,
                                             nvrhi::IBindingLayout *pLayout)
{
    if (device == nullptr || pLayout == nullptr)
    {
        return nullptr;
    }

    nvrhi::BindingSetDesc setDesc;

    for (uint32_t i = 0; i < count; i++)
    {
        nvrhi::ITexture *texture = nullptr;

        if (pBindings[i].isGodRaysStandIn)
        {
            texture = pGodRaysStandIn;
        }
        else
        {
            const uint32_t imageSlot = FindComposeImage(pBindings[i].image);
            texture = imageSlot != COMPOSE_IMAGE_NONE ? pEngineTextures[imageSlot].Get() : nullptr;
        }

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
        // The wraps reference engine images and buffers and the stand-ins are NVRHI resources; the
        // host destroys the pass while it can still idle the device (VulkanDevice does that before
        // the skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    for (Target &target : targets)
    {
        target.adapterSet = nullptr;
        target.interleaveSet = nullptr;
        target.histogramSet = nullptr;
        target.checkerboardSet = nullptr;
        target.prepareFinalSet = nullptr;

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

    // The tonemapping wraps and their sets reference an engine buffer that outlives the pass; the
    // handles themselves are the module's.
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        tonemappingUavSets[i] = nullptr;
        tonemappingSrvSets[i] = nullptr;
        tonemappingBuffers[i] = nullptr;
    }

    godRaysZeroTexture = nullptr;
    godRaysZeroWidth = 0;
    godRaysZeroHeight = 0;

    volumetricSet = nullptr;
    emptySet = nullptr;
    volumetricDummySampler = nullptr;
    volumetricDummyTexture = nullptr;

    prepareFinalPipeline = nullptr;
    checkerboardPipeline = nullptr;
    averagePipeline = nullptr;
    histogramPipeline = nullptr;
    interleavePipeline = nullptr;
    adapterPipeline = nullptr;

    volumetricLayout = nullptr;
    emptyLayout = nullptr;
    tonemappingSrvLayout = nullptr;
    tonemappingUavLayout = nullptr;
    uniformLayout = nullptr;
    prepareFinalFramebufferLayout = nullptr;
    checkerboardFramebufferLayout = nullptr;
    histogramFramebufferLayout = nullptr;
    interleaveFramebufferLayout = nullptr;
    adapterFramebufferLayout = nullptr;

    prepareFinalShader = nullptr;
    checkerboardShader = nullptr;
    averageShader = nullptr;
    histogramShader = nullptr;
    interleaveShader = nullptr;
    adapterShader = nullptr;
}

bool RhiRtComposePass::Create(nvrhi::IDevice *pDevice,
                              rhi::RhiFrameContext *pFrameContext,
                              const Tonemapping *pTonemapping,
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

    // The engine tonemapping object is what the exposure chain writes and the final composition
    // reads; without it the module has no ShTonemapping buffer to bind and the chain would produce
    // the NaN the class comment warns about, so a missing object makes the pass unusable.
    if (pTonemapping == nullptr)
    {
        LogMessage(print, "Warning: RHI: the compose pass needs the engine tonemapping object");
        return false;
    }

    // The widest pipeline declares five binding layouts (the final composition), so the pinned
    // NVRHI's binding-layout cap does not matter here and the module has no
    // `c_MaxBindingLayouts` guard: the RT passes' twelve-layout pipelines are the only users of
    // the raised cap (third_party/nvrhi-max-binding-layouts.patch).

    if (!LoadShader(ADAPTER_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, adapterShader) ||
        !LoadShader(INTERLEAVE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, interleaveShader) ||
        !LoadShader(HISTOGRAM_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, histogramShader) ||
        !LoadShader(AVERAGE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, averageShader) ||
        !LoadShader(CHECKERBOARD_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, checkerboardShader) ||
        !LoadShader(PREPARE_FINAL_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, prepareFinalShader))
    {
        return false;
    }

    // The five exact set-0 layouts: 17, 3, 1, 7 and 16 items, Compute visibility, SRV offset 124
    // and UAV offset 0. A union layout or a shared set would fail the exact-set rule
    // (validation-device.cpp:1855-1871), and the primary pass's AllRayTracing uniform layout would
    // leave the compute stage undeclared, so each pass owns its own layout handle here.
    adapterFramebufferLayout =
        CreateFramebufferLayout(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT);
    interleaveFramebufferLayout =
        CreateFramebufferLayout(device, INTERLEAVE_BINDINGS, INTERLEAVE_BINDING_COUNT);
    histogramFramebufferLayout =
        CreateFramebufferLayout(device, HISTOGRAM_BINDINGS, HISTOGRAM_BINDING_COUNT);
    checkerboardFramebufferLayout =
        CreateFramebufferLayout(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT);
    prepareFinalFramebufferLayout =
        CreateFramebufferLayout(device, PREPARE_FINAL_BINDINGS, PREPARE_FINAL_BINDING_COUNT);

    // Set 1 for all six pipelines: the engine's global uniform at raw binding 0
    // (ShaderCommonHLSL.hlsli:104 spells it with BINDING_GLOBAL_UNIFORM and DESC_SET_GLOBAL_UNIFORM).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM));

        uniformLayout = device->createBindingLayout(desc);
    }

    // Set 2 of the histogram and the average: the engine's tonemapping buffer as
    // `RWStructuredBuffer<ShTonemapping>` at raw binding 0 (measured: `OpDecorate %tonemapping
    // DescriptorSet 2 / Binding 0`, `type_RWStructuredBuffer_ShTonemapping`, no NonWritable).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setUnorderedAccessViewOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(BINDING_LUM_HISTOGRAM));

        tonemappingUavLayout = device->createBindingLayout(desc);
    }

    // Set 2 of the final composition: the same buffer as a read-only
    // `StructuredBuffer<ShTonemapping>` (measured: `OpDecorate %tonemapping DescriptorSet 2 /
    // Binding 0`, `type_StructuredBuffer_ShTonemapping`, NonWritable on the member).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(BINDING_LUM_HISTOGRAM));

        tonemappingSrvLayout = device->createBindingLayout(desc);
    }

    // The holes: the average declares no set 0 at all and the final composition's set 3 is the
    // LPM block the shipped blob does not reference (measured: no `DescriptorSet 3` decoration).
    // A real, zero-item layout keeps the positions of the uniform and tonemapping sets, and the
    // empty set below fills each position because the automatic-barrier pass dereferences every
    // entry (vulkan-state-tracking.cpp:105). Visibility still has to be set: the validation device
    // rejects a layout with visibility = None (validation-device.cpp:1340-1344).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;

        emptyLayout = device->createBindingLayout(desc);
    }

    // Set 4: the sampled volumetric volume at raw binding 1 and its sampler at raw binding 2
    // (measured: `OpDecorate %g_volumetric_Sampled DescriptorSet 4 / Binding 1`, `Texture3D<float4>`
    // rgba16f, and `%g_volumetric_Sampler` at raw 2). The sampler offset of 0 keeps the slot as the
    // raw binding. The read is dead at runtime (see the zero stand-in below), so the module binds a
    // 1x1x1 dummy instead of the engine's 160x88x64 image.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setShaderResourceOffset(0)
                                   .setSamplerOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_VOLUMETRIC_SAMPLED));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(BINDING_VOLUMETRIC_SAMPLER));

        volumetricLayout = device->createBindingLayout(desc);
    }

    if (adapterFramebufferLayout == nullptr || interleaveFramebufferLayout == nullptr ||
        histogramFramebufferLayout == nullptr || checkerboardFramebufferLayout == nullptr ||
        prepareFinalFramebufferLayout == nullptr || uniformLayout == nullptr ||
        tonemappingUavLayout == nullptr || tonemappingSrvLayout == nullptr ||
        emptyLayout == nullptr || volumetricLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass binding layout");
        return false;
    }

    adapterPipeline = CreateComposePipeline(device, adapterShader,
                                            { adapterFramebufferLayout, uniformLayout });
    interleavePipeline = CreateComposePipeline(device, interleaveShader,
                                               { interleaveFramebufferLayout, uniformLayout });
    histogramPipeline = CreateComposePipeline(device, histogramShader,
                                              { histogramFramebufferLayout, uniformLayout,
                                                tonemappingUavLayout });
    averagePipeline = CreateComposePipeline(device, averageShader,
                                            { emptyLayout, uniformLayout, tonemappingUavLayout });
    checkerboardPipeline = CreateComposePipeline(device, checkerboardShader,
                                                 { checkerboardFramebufferLayout, uniformLayout });
    prepareFinalPipeline = CreateComposePipeline(device, prepareFinalShader,
                                                 { prepareFinalFramebufferLayout, uniformLayout,
                                                   tonemappingSrvLayout, emptyLayout,
                                                   volumetricLayout });

    if (adapterPipeline == nullptr || interleavePipeline == nullptr || histogramPipeline == nullptr ||
        averagePipeline == nullptr || checkerboardPipeline == nullptr || prepareFinalPipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass compute pipeline");
        return false;
    }

    // The engine's per-slot `ShTonemapping` buffers as module-owned native wraps, and the two sets
    // over each. What the wrap has to declare, measured in the pinned NVRHI:
    //  - `structStride != 0` for either structured kind (validation-device.cpp:1693-1699), which the
    //    backend also asserts when the set is created (vulkan-resource-bindings.cpp:535-536);
    //  - `canHaveUAVs` for the StructuredBuffer_UAV of the exposure pair
    //    (validation-device.cpp:1709-1713);
    //  - `initialState = UnorderedAccess` with `keepInitialState = true`: the exposure pair writes
    //    the buffer on every list and the final composition reads it later in the same list, so the
    //    slot starts in the state the first UAV binding requires. A buffer has no image layout, so
    //    the choice only decides the direction of the memory-dependency barriers; the raster mode's
    //    separate SRV wrap (NvrhiFrameSkeleton::PrepareWorld) is never alive at the same time.
    // The engine buffer is created once in the Tonemapping constructor and `OnShaderReload` only
    // rebuilds its pipelines, so the wraps and the sets never have to follow a handle change.
    {
        const uint32_t elementSize = pTonemapping->GetElementSize();

        if (elementSize == 0)
        {
            LogMessage(print, "Warning: RHI: the engine tonemapping buffer has no element size");
            return false;
        }

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            const uint64_t bufferHandle =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pTonemapping->GetBuffer(i)));

            if (bufferHandle == 0)
            {
                LogMessage(print, "Warning: RHI: the engine tonemapping buffer is null");
                return false;
            }

            nvrhi::BufferDesc desc;
            desc.byteSize = elementSize;
            desc.structStride = elementSize;
            desc.canHaveUAVs = true;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            desc.keepInitialState = true;
            desc.debugName = "RhiRtComposePass tonemapping wrap " + std::to_string(i);

            tonemappingBuffers[i] = device->createHandleForNativeBuffer(
                nvrhi::ObjectTypes::VK_Buffer, nvrhi::Object(bufferHandle), desc);

            if (tonemappingBuffers[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap the engine tonemapping buffer");
                return false;
            }

            nvrhi::BindingSetDesc uavSetDesc;
            uavSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(
                BINDING_LUM_HISTOGRAM, tonemappingBuffers[i]));
            tonemappingUavSets[i] = device->createBindingSet(uavSetDesc, tonemappingUavLayout);

            nvrhi::BindingSetDesc srvSetDesc;
            srvSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(
                BINDING_LUM_HISTOGRAM, tonemappingBuffers[i]));
            tonemappingSrvSets[i] = device->createBindingSet(srvSetDesc, tonemappingSrvLayout);

            if (tonemappingUavSets[i] == nullptr || tonemappingSrvSets[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to create a compose pass tonemapping binding set");
                return false;
            }
        }
    }

    // The one real empty set the average's position 0 and the final composition's position 3 bind.
    // One set bound at two positions is legal Vulkan; the layouts are the same object.
    emptySet = device->createBindingSet(nvrhi::BindingSetDesc(), emptyLayout);
    if (emptySet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the compose pass empty binding set");
        return false;
    }

    // The 1x1x1 volumetric dummy of set 4. `applyVolumetrics` returns before touching the volume
    // when `coreQ2RTX != 0` (CmPrepareFinal.comp.hlsl:269-280) and FillUniform forces that value
    // (VulkanDevice.cpp:330), so a one-texel RGBA16F volume is behaviourally exact; a real wrap of
    // the engine's 160x88x64 GENERAL-layout image would need a Volumetric accessor and a state
    // announcement for zero pixels. The dummy is cleared to zero once, on the first list that binds
    // it (Render): isUAV is what `clearTextureFloat`'s validation requires and the state below is
    // the compute-visibility SRV resting state, which keeps the cleared contents across lists.
    {
        nvrhi::TextureDesc desc;
        desc.dimension = nvrhi::TextureDimension::Texture3D;
        desc.width = 1;
        desc.height = 1;
        desc.depth = 1;
        desc.mipLevels = 1;
        desc.arraySize = 1;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.isShaderResource = true;
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
        desc.keepInitialState = true;
        desc.debugName = "RhiRtComposePass volumetric dummy (1x1x1)";

        volumetricDummyTexture = rhi::createTexture(device, desc, desc.debugName);
        volumetricDummySampler =
            rhi::createEngineTextureSampler(device, "RhiRtComposePass volumetric dummy sampler");

        if (volumetricDummyTexture == nullptr || volumetricDummySampler == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the compose pass volumetric dummy");
            return false;
        }

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_VOLUMETRIC_SAMPLED,
                                                           volumetricDummyTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_VOLUMETRIC_SAMPLER,
                                                       volumetricDummySampler));

        volumetricSet = device->createBindingSet(setDesc, volumetricLayout);
        if (volumetricSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the compose pass volumetric binding set");
            return false;
        }
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

    // The 27 images of this slot. `GetImageHandles` resolves the engine's ping-pong swap
    // (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's current
    // image, which is the one the engine's own per-slot descriptor set binds to the variable's
    // fixed raw binding - the shader never sees the swap. The 4-tuple overload supplies each
    // image's extent, which every image here has to answer with the render size: all 27 are
    // render-sized by their generated flags (none carries a FORCE_SIZE/UPSCALED/SINGLE_PIXEL flag),
    // and the shaders access them at
    // `globalUniform.renderWidth/renderHeight`, so a different extent is a wrongly sized wrap and
    // the compose is skipped instead of sampling outside the image.
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

    // The stand-in for image 64, before the sets that bind it. It is created and cleared on this
    // list when it is new or the resolution changed; a replacement retires both slots' framebuffer
    // sets, because the prepare-final set references the old texture.
    if (!PrepareGodRaysStandIn(pCommandList, width, height))
    {
        return;
    }

    if (!PrepareFramebufferSets(target) || !PrepareUniformSet(target, pUniformBuffer))
    {
        return;
    }

    // The image state contract, spelled out on the class: the engine leaves every framebuffer image
    // in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap keeps no state
    // between command lists (RhiTextureSource.h), so every list announces that state for all 27
    // images before the first use. The primary, direct and indirect passes wrote some of them on
    // this very list through their own wraps; the physical layout they left them in is exactly this
    // GENERAL state, so the announcement is the truth and the SRV bindings' automatic transitions
    // start from it.
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.engineTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    // The volumetric dummy, cleared to zero once per pass lifetime on the first list that binds it
    // (`coreQ2RTX = 1` makes the shader's read unreachable, but the descriptor has to be valid and
    // a defined texel costs one 1x1 clear). clearTextureFloat moves it to CopyDest and the explicit
    // state call puts it back to the resting state of the compute-visibility SRV binding.
    if (!volumetricDummyCleared)
    {
        pCommandList->clearTextureFloat(volumetricDummyTexture, nvrhi::AllSubresources,
                                        nvrhi::Color(0.f, 0.f, 0.f, 0.f));
        pCommandList->setTextureState(volumetricDummyTexture, nvrhi::AllSubresources,
                                      nvrhi::ResourceStates::NonPixelShaderResource);
        volumetricDummyCleared = true;
    }

    // The legacy dispatch arithmetic, all passes over the render resolution
    // (Q2Denoiser.cpp:400-401, :606-607, ImageComposition.cpp:198-199, Tonemapping.cpp:176-179).
    // `Utils::GetWorkGroupCount` is `1 + ceil(size / 16)`, one workgroup beyond the ceiling; the
    // shaders' bounds check on `globalUniform.renderWidth/renderHeight` discards those invocations.
    const uint32_t groupsX = Utils::GetWorkGroupCount(width, COMPOSE_GROUP_SIZE);
    const uint32_t groupsY = Utils::GetWorkGroupCount(height, COMPOSE_GROUP_SIZE);

    // The legacy order: the adapter composites into Q2_COLOR, the interleave resolves it into
    // PRE_FINAL, the exposure pair histograms and averages that PRE_FINAL into the slot's
    // tonemapping buffer, the checkerboard writes FINAL and its two companions, and the final
    // composition tone-maps them into FINAL and BLOOM_INPUT (Q2Denoiser.cpp:412-436 then :600-621,
    // Tonemapping.cpp:153-208, ImageComposition.cpp:93-98, VulkanDevice.cpp:1061-1085). The shared
    // wraps make every hand-off an ordinary automatic-barrier transition.
    RecordDispatch(pCommandList, adapterPipeline, { target.adapterSet, target.uniformSet },
                   groupsX, groupsY, 1);
    RecordDispatch(pCommandList, interleavePipeline, { target.interleaveSet, target.uniformSet },
                   groupsX, groupsY, 1);

    // The exposure pair on the fresh PRE_FINAL and the slot's tonemapping buffer: the histogram
    // over the render resolution, then the average over exactly the 128 bins in one workgroup
    // (Tonemapping.cpp:171-208).
    RecordDispatch(pCommandList, histogramPipeline,
                   { target.histogramSet, target.uniformSet, tonemappingUavSets[frameIndex] },
                   groupsX, groupsY, 1);
    RecordDispatch(pCommandList, averagePipeline,
                   { emptySet, target.uniformSet, tonemappingUavSets[frameIndex] },
                   1, 1, 1);

    RecordDispatch(pCommandList, checkerboardPipeline, { target.checkerboardSet, target.uniformSet },
                   groupsX, groupsY, 1);

    // The final composition: set 3 is the dead LPM hole, set 4 the volumetric dummy, and the
    // tonemapping buffer is bound read-only, after the exposure pair wrote it in this list.
    RecordDispatch(pCommandList, prepareFinalPipeline,
                   { target.prepareFinalSet, target.uniformSet, tonemappingSrvSets[frameIndex],
                     emptySet, volumetricSet },
                   groupsX, groupsY, 1);

    // The 18 images whose last use was a sampled read rest in the read-only layout now; move them
    // back to UnorderedAccess, the engine's GENERAL, so the next frame's primary/direct/indirect
    // UAV writes start from the state their own announcements claim. The restore includes 27 and
    // 117: without it the next frame's interleave and adapter UAV writes would run against a
    // read-only image. The last pass's five UAV images (16-18, 28, 65) and the adapter's four
    // channel UAVs already end in GENERAL.
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

void RhiRtComposePass::ReleaseFramebufferSets(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the sets reference the wraps of engine images (and the god-rays stand-in) the GPU may
    // still be reading. The queue takes its reference now, so the handles below can be cleared
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
        if (target.histogramSet != nullptr)
        {
            frameContext->Retire(target.histogramSet);
        }
        if (target.checkerboardSet != nullptr)
        {
            frameContext->Retire(target.checkerboardSet);
        }
        if (target.prepareFinalSet != nullptr)
        {
            frameContext->Retire(target.prepareFinalSet);
        }
    }

    target.adapterSet = nullptr;
    target.interleaveSet = nullptr;
    target.histogramSet = nullptr;
    target.checkerboardSet = nullptr;
    target.prepareFinalSet = nullptr;
}

void RhiRtComposePass::ReleaseFramebufferTarget(Target &target)
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
    nvrhi::ITexture *const godRaysStandIn = godRaysZeroTexture.Get();

    if (target.adapterSet == nullptr)
    {
        target.adapterSet = CreateFramebufferSet(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT,
                                                 target.engineTextures, nullptr,
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
                                                    target.engineTextures, nullptr,
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

    if (target.histogramSet == nullptr)
    {
        target.histogramSet = CreateFramebufferSet(device, HISTOGRAM_BINDINGS, HISTOGRAM_BINDING_COUNT,
                                                   target.engineTextures, nullptr,
                                                   histogramFramebufferLayout);

        if (target.histogramSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass histogram binding set");
            }
            return false;
        }
    }

    if (target.checkerboardSet == nullptr)
    {
        target.checkerboardSet = CreateFramebufferSet(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT,
                                                      target.engineTextures, nullptr,
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

    if (target.prepareFinalSet == nullptr)
    {
        target.prepareFinalSet = CreateFramebufferSet(device, PREPARE_FINAL_BINDINGS, PREPARE_FINAL_BINDING_COUNT,
                                                      target.engineTextures,
                                                      godRaysStandIn,
                                                      prepareFinalFramebufferLayout);

        if (target.prepareFinalSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass prepare-final binding set");
            }
            return false;
        }
    }

    return true;
}

bool RhiRtComposePass::PrepareGodRaysStandIn(nvrhi::ICommandList *pCommandList, uint32_t width, uint32_t height)
{
    if (godRaysZeroTexture != nullptr && godRaysZeroWidth == width && godRaysZeroHeight == height)
    {
        return true;
    }

    if (godRaysZeroTexture != nullptr)
    {
        // A new resolution: the replaced texture goes through the retire queue like every other
        // resource a recorded list may still sample, and the sets that reference it are retired
        // too (the prepare-final set is the only one; the release is shared so the invariant stays
        // in one place). The next PrepareFramebufferSets rebuilds them over the new texture.
        for (Target &target : targets)
        {
            ReleaseFramebufferSets(target);
        }

        if (frameContext != nullptr)
        {
            frameContext->Retire(godRaysZeroTexture);
        }

        godRaysZeroTexture = nullptr;
        godRaysZeroWidth = 0;
        godRaysZeroHeight = 0;
    }

    // The stand-in has to match the read it replaces: `.Load(int3(pix, 0))` at render coordinates
    // (CmPrepareFinal.comp.hlsl:446-449), so it is render-sized, and its B10G11R11 format is the
    // engine image's `ShFramebuffers_Formats[FB_IMAGE_INDEX_GOD_RAYS_FILTERED]` so the view the
    // shader sees matches the one the engine's own set would have bound.
    nvrhi::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = nvrhi::Format::R11G11B10_FLOAT;
    // isShaderResource is the eSampled usage the Texture_SRV binding needs. isUAV is what
    // `clearTextureFloat`'s validation requires: it refuses a texture with both isRenderTarget and
    // isUAV false (validation-commandlist.cpp:234-241). An NVRHI-created texture always carries
    // TRANSFER_SRC | TRANSFER_DST (vulkan-texture.cpp:102-107), so the clear is legal here - unlike
    // on the engine's image 64, whose usage set has no TRANSFER_DST and where `clearTextureFloat`
    // would trip a new VUID.
    desc.isShaderResource = true;
    desc.isUAV = true;
    // The resting state of a compute-visibility SRV binding (state-tracking.cpp:455-464, mapping in
    // vulkan-constants.cpp:238-241). keepInitialState makes every later list start from it instead
    // of Unknown, so the once-cleared zero contents are not discarded by an undefined-sourced
    // transition on the second frame.
    desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
    desc.keepInitialState = true;
    desc.debugName = "RhiRtComposePass god-rays zero stand-in (render-sized, temporary until A5)";

    godRaysZeroTexture = rhi::createTexture(device, desc, desc.debugName);

    if (godRaysZeroTexture == nullptr)
    {
        if (!warnedZeroTexture)
        {
            warnedZeroTexture = true;
            LogMessage(print, "Warning: RHI: the compose pass cannot create its god-rays zero stand-in, the compose is skipped");
        }
        return false;
    }

    // One clear per wrap lifetime is enough: nothing ever writes the texture. clearTextureFloat
    // moves it to CopyDest and clears it; the setTextureState below is the CopyDest ->
    // NonPixelShaderResource transition the first SRV use would emit anyway, made explicit so the
    // desc's resting state is the truth at the list's close.
    pCommandList->clearTextureFloat(godRaysZeroTexture, nvrhi::AllSubresources,
                                    nvrhi::Color(0.f, 0.f, 0.f, 0.f));
    pCommandList->setTextureState(godRaysZeroTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::NonPixelShaderResource);

    godRaysZeroWidth = width;
    godRaysZeroHeight = height;
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
                                      std::initializer_list<nvrhi::IBindingSet *> sets,
                                      uint32_t groupsX,
                                      uint32_t groupsY,
                                      uint32_t groupsZ)
{
    nvrhi::ComputeState state;
    state.setPipeline(pPipeline);

    for (nvrhi::IBindingSet *pSet : sets)
    {
        state.addBindingSet(pSet);
    }

    pCommandList->setComputeState(state);
    pCommandList->dispatch(groupsX, groupsY, groupsZ);
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
