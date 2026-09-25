#include "RhiRtDirectPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiTextureSource.h"
#include "RhiTextureTable.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../LightManager.h"

#include <cstring>
#include <string>
#include <tuple>

using namespace vkpt;

namespace
{

// The five engine blobs, by the file names the shader build writes into the folder the host passes
// in (ShaderManager.cpp:41-49). The direct raygen is `RtRaygenDirect.rgen.spv`; the other four are
// the same stage set the primary pass loads. `RtMiss` is never entered by this raygen - its rays
// are shadow rays with `RAY_FLAG_SKIP_CLOSEST_HIT_SHADER`, and the misses they can reach are
// SBT_INDEX_MISS_SHADOW = 1 (RaygenCommon.hlsli:465-473) - but it is loaded and tabled anyway: the
// engine's SBT indices are fixed, so `RtMiss` has to fill index 0 for the shadow miss to stay at
// index 1.
const char *const RAYGEN_SHADER_FILE_NAME = "RtRaygenDirect.rgen.spv";
const char *const MISS_SHADER_FILE_NAME = "RtMiss.rmiss.spv";
const char *const SHADOW_MISS_SHADER_FILE_NAME = "RtMissShadowCheck.rmiss.spv";
const char *const CLOSEST_HIT_SHADER_FILE_NAME = "RtClsOpaque.rchit.spv";
const char *const ANY_HIT_SHADER_FILE_NAME = "RtAlphaTest.rahit.spv";

// Distinct export names are mandatory on Vulkan at this pin: the backend keys the pipeline's
// shader groups by PipelineShaderDesc::exportName, falling back to the shader entry name
// (vulkan-raytracing.cpp:1593-1603), and every engine blob's entry point is "main"
// (RhiPipeline.cpp:23-24). The hit group's export name is asserted non-empty (:1641), and the two
// hit groups need distinct names although they share the closest-hit shader - the order of the
// addHitGroup calls is what makes offset 0 the fully-opaque group and offset 1 the alpha-tested
// one, which is the order the engine's instance records address
// (instanceShaderBindingTableRecordOffset 0/1, ASManager.cpp:976-991). The names carry this
// module's prefix so a message or a capture can tell the two pipelines' groups apart.
const char *const RAYGEN_EXPORT_NAME = "RhiRtDirectRGen";
const char *const MISS_EXPORT_NAME = "RhiRtDirectRMiss";
const char *const SHADOW_MISS_EXPORT_NAME = "RhiRtDirectRMissShadow";
const char *const HIT_GROUP_OPAQUE_EXPORT_NAME = "RhiRtDirectRClsOpaque";
const char *const HIT_GROUP_ALPHA_TESTED_EXPORT_NAME = "RhiRtDirectRClsOpaqueAlphaTested";

// The engine's RT set order (RayTracingPipeline.cpp:62-86), the same numbers every RT shader
// transcribes as DESC_SET_*. The order is NOT the raster passes' order and is what the pipeline's
// addBindingLayout calls have to follow, because the pinned backend's legacy binding mode makes the
// list position the descriptor set number (vulkan-resource-bindings.cpp:1090-1099).
constexpr uint32_t SET_TLAS = 0;
constexpr uint32_t SET_FRAMEBUFFERS = 1;
constexpr uint32_t SET_GLOBAL_UNIFORM = 2;
constexpr uint32_t SET_VERTEX_DATA = 3;
constexpr uint32_t SET_TEXTURES = 4;
constexpr uint32_t SET_RANDOM = 5;
constexpr uint32_t SET_LIGHT_SOURCES = 6;
constexpr uint32_t SET_CUBEMAPS = 7;
constexpr uint32_t SET_RENDER_CUBEMAP = 8;
constexpr uint32_t SET_PORTALS = 9;
constexpr uint32_t SET_VOLUMETRIC = 10;
constexpr uint32_t SET_RAY_STATS = 11;
constexpr uint32_t PIPELINE_SET_COUNT = 12;

static_assert(SET_TLAS == 0 && SET_FRAMEBUFFERS == 1 && SET_GLOBAL_UNIFORM == 2 &&
              SET_VERTEX_DATA == 3 && SET_TEXTURES == 4 && SET_RANDOM == 5 &&
              SET_LIGHT_SOURCES == 6 && SET_CUBEMAPS == 7 && SET_RENDER_CUBEMAP == 8 &&
              SET_PORTALS == 9 && SET_VOLUMETRIC == 10 && SET_RAY_STATS == 11 &&
              PIPELINE_SET_COUNT == 12,
              "the RT set order has to stay RayTracingPipeline.cpp:62-86");

// The 12 images `RtRaygenDirect.rgen` references, measured 2026-09-25 with `spirv-dis` over the
// blob the current sources build to (the command of GenerateShaders.py:157-164, dxc from Vulkan
// SDK 1.4.321.1). The three UAVs first (the blob's set-1 storage images), then the nine SRVs (its
// sampled images); the entry-point interface and the descriptor decorations name exactly these and
// no other set-1 binding. The array order is not a requirement of NVRHI - a binding set's items are
// keyed by slot - it only groups the writes and the reads the way the raygen uses them.
//
// The engine binding numbers are the identity for the UAVs (`ShFramebuffers_Bindings[index]`) and
// `124 + index` for the sampled views (`ShFramebuffers_Sampled_Bindings[index]`), so the table
// carries the image index and the kind and the module derives both the layout item and the set
// item from the generated arrays; a hand-written binding number could drift from the shader build,
// the generated ones cannot.
//
// The A4.2a shader-side fix is what makes this list 12 items and not 13: the sampled view of the
// view direction (raw binding 147, the same image as the UAV at 23) is gone from the blob, so no
// image is bound as both an SRV and a UAV and the set has no layout conflict (a42_recon.md §0.1.1,
// §7.1). The sampled screen emission (185) is not on the list either - DXC dead-strips that load,
// because the raygen never reads `surf.emission` (a42_recon.md §1).
struct FramebufferBinding
{
    FramebufferImageIndex image;
    bool isUAV;
};

constexpr uint32_t FRAMEBUFFER_BINDING_COUNT = 12;
constexpr FramebufferBinding FRAMEBUFFER_BINDINGS[FRAMEBUFFER_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,      true  }, //  14  framebufUnfilteredDirect
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,    true  }, //  15  framebufUnfilteredSpecular
    { FB_IMAGE_INDEX_VIEW_DIRECTION,         true  }, //  23  framebufViewDirection
    { FB_IMAGE_INDEX_ALBEDO,                 false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,                 false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,                 false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,        false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,     false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_SURFACE_POSITION,       false }, // 143  framebufSurfacePosition_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,       false }, // 239  framebufQ2GradSmplPos_Sampled
    { FB_IMAGE_INDEX_Q2_RNG_SEED,            false }, // 245  framebufQ2RngSeed_Sampled
    { FB_IMAGE_INDEX_Q2_CLUSTER,             false }, // 247  framebufQ2Cluster_Sampled
};

// The engine raw binding of one entry: the UAV array for the storage images, the sampled array for
// the SRVs. Both are the numbers the blob's OpDecorate bindings carry.
uint32_t GetFramebufferRawBinding(const FramebufferBinding &binding)
{
    return binding.isUAV ? ShFramebuffers_Bindings[binding.image]
                         : ShFramebuffers_Sampled_Bindings[binding.image];
}

// Set 3's bindings: vertex data, indices and the geometry-instance buffer the alpha-tested any-hit
// (`RtAlphaTest.rahit`) fetches triangles through when a shadow ray enters its group. Binding 5
// (`geomIndexPrevToCur`) is left out because the any-hit's module does not reference it (measured:
// the blob declares set 3 bindings 0-4); the rest are the declarations of VertexData.hlsli. The
// list is the primary pass's, because the layout handle and the buffers are the primary's.
constexpr uint32_t VERTEX_DATA_BUFFER_COUNT = 7;
constexpr uint32_t VERTEX_DATA_BINDINGS[VERTEX_DATA_BUFFER_COUNT] =
{
    BINDING_VERTEX_BUFFER_STATIC,
    BINDING_VERTEX_BUFFER_DYNAMIC,
    BINDING_INDEX_BUFFER_STATIC,
    BINDING_INDEX_BUFFER_DYNAMIC,
    BINDING_GEOMETRY_INSTANCES,
    BINDING_PREV_POSITIONS_BUFFER_DYNAMIC,
    BINDING_PREV_INDEX_BUFFER_DYNAMIC,
};

// The engine's pipeline-wide RT limits (RayTracingPipeline.cpp:203 and the measured blob structs):
// recursion depth 2; the payload is ShPayload = {float2 baryCoords; uint instIdAndIndex; uint
// geomAndPrimIndex} = 16 B and the hit attribute HitAttributes = {float2 inBaryCoords} = 8 B
// (Structs.hlsli:47-57). The direct raygen and the shadow miss statically carry only
// ShPayloadShadow = {uint isShadowed} = 4 B (measured: their entry-point interfaces name
// `g_payloadShadow` alone), but the pipeline also carries `RtMiss` and `RtClsOpaque`, which use the
// 16 B `g_payload`, so the union stays 16 B. NVRHI's two size fields are D3D12-only at this pin, so
// on Vulkan they are documentation; they are kept exact for a future backend switch.
constexpr uint32_t MAX_RECURSION_DEPTH = 2;
constexpr uint32_t MAX_PAYLOAD_SIZE = 2 * sizeof(float) + 2 * sizeof(uint32_t);
constexpr uint32_t MAX_ATTRIBUTE_SIZE = 2 * sizeof(float);

// The five device-local engine buffers of set 6, in the order of `LightManager::Buffers` and of the
// engine's own descriptor writes (LightManager.cpp:942-983): the light array, the list offsets, the
// list words, the light statistics and the cluster sky visibility. The raw bindings are the
// generated BINDING_LIGHT_SOURCES* numbers (0, 4, 5, 6, 8); the sizes are the engine's own creation
// sizes (LightManager.cpp:71-96) and are what a native wrap has to carry, because
// `LightManager::Buffers` hands out the VkBuffers without their sizes. The strides are the
// shader's element strides: ShLightEncoded is 144 B and every other buffer is a uint array.
//
// The statistics buffer is the only UAV item (the raygen's `q2AccumulateLightStats` writes it) and
// the only one the RHI never copies: while `globalUniform.q2LightStatsMode` is
// Q2_LIGHT_STATS_DISABLED the shader neither reads nor writes it (Q2LightLists.hlsli:192-197,
// 348), which is the host contract this pass documents, so its contents do not matter and
// `ResetLightStats`'s fill (LightManager.cpp:765-793, legacy path only) has no RHI counterpart yet.
// A later cut that enables the statistics has to bring the fill and its barrier (a42_recon.md
// §2.5, §6.2.2b).
struct LightBufferBinding
{
    uint32_t binding;
    bool isUAV;
    uint64_t byteSize;
    uint32_t structStride;
};

constexpr uint32_t LIGHT_BUFFER_COUNT = 5;
constexpr LightBufferBinding LIGHT_BUFFER_BINDINGS[LIGHT_BUFFER_COUNT] =
{
    {
        BINDING_LIGHT_SOURCES,
        false,
        uint64_t(LightManager::LIGHT_ARRAY_ENTRY_COUNT) * sizeof(ShLightEncoded),
        static_cast<uint32_t>(sizeof(ShLightEncoded)),
    },
    {
        BINDING_LIGHT_SOURCES_Q2_LIGHT_LIST_OFFSETS,
        false,
        uint64_t(sizeof(uint32_t)) * (Q2_MAX_CLUSTERS + 1),
        static_cast<uint32_t>(sizeof(uint32_t)),
    },
    {
        BINDING_LIGHT_SOURCES_Q2_LIGHT_LIST_LIGHTS,
        false,
        uint64_t(sizeof(uint32_t)) * Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL,
        static_cast<uint32_t>(sizeof(uint32_t)),
    },
    {
        BINDING_LIGHT_SOURCES_Q2_LIGHT_STATS,
        true,
        uint64_t(sizeof(uint32_t)) * Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL *
            Q2_LIGHT_LIST_STATS_SIDES * 2 * Q2_LIGHT_LIST_STATS_BUFFERS,
        static_cast<uint32_t>(sizeof(uint32_t)),
    },
    {
        BINDING_LIGHT_SOURCES_Q2_CLUSTER_SKY_VIS,
        false,
        uint64_t(sizeof(uint32_t)) * LightManager::CLUSTER_SKY_VIS_WORD_COUNT,
        static_cast<uint32_t>(sizeof(uint32_t)),
    },
};

static_assert(sizeof(ShLightEncoded) == 144,
              "the shader's StructuredBuffer<ShLightEncoded> strides by 144 B (Generated/ShaderCommonC.h:389-401)");

// The four items `LightManager::GetFrameCopies` reports, mapped to the set-6 buffer each one copies
// into: the light-array prefix (binding 0), the list offsets (4), the list words (5) and the sky
// visibility (8). The statistics buffer (6) has no copy. The order is `FrameCopies`'s.
constexpr uint32_t LIGHT_COPY_COUNT = 4;
constexpr uint32_t LIGHT_COPY_BUFFER_INDICES[LIGHT_COPY_COUNT] =
{
    0, // lights
    1, // listOffsets
    2, // listLights
    4, // clusterSkyVis
};

const char *const LIGHT_BUFFER_DEBUG_NAMES[LIGHT_BUFFER_COUNT] =
{
    "light sources",
    "q2 light list offsets",
    "q2 light list lights",
    "q2 light stats",
    "q2 cluster sky visibility",
};

// The descriptor of a native wrap of an engine device-local buffer that a shader reads as a
// structured buffer, or writes as one for the statistics.
//
//  - structStride is what a StructuredBuffer binding requires: the validation device rejects a
//    buffer whose desc has none (validation-device.cpp:1693-1699) and the binding-set creation
//    asserts the same (vulkan-resource-bindings.cpp:535-536). The value is the shader's element
//    stride, not the wrapped buffer's size, and on a native wrap it is bookkeeping only.
//  - canHaveUAVs is what a StructuredBuffer_UAV binding requires (the validation device refuses a
//    UAV binding on a buffer whose desc lacks it, as it does for textures at :1631-1634).
//  - initialState claims the state the buffer is in at the start of every command list, because
//    the engine's writes are invisible to NVRHI: the four SRV buffers are written by the module's
//    own copyBuffer and claim CopyDest (the same choice RhiAccelStructs::MakeCopyDest describes),
//    the statistics buffer claims UnorderedAccess and is never touched at all. keepInitialState
//    makes the claim stick for every later list instead of reporting an unknown prior state
//    (state-tracking.cpp:290-297); the automatic barrier pass then turns the copy or the shader
//    use into a transition out of that claim.
nvrhi::BufferDesc MakeLightBufferDesc(const LightBufferBinding &binding, std::string debugName)
{
    nvrhi::BufferDesc desc;
    desc.byteSize = binding.byteSize;
    desc.structStride = binding.structStride;
    desc.canHaveUAVs = binding.isUAV;
    desc.initialState = binding.isUAV ? nvrhi::ResourceStates::UnorderedAccess
                                      : nvrhi::ResourceStates::CopyDest;
    desc.keepInitialState = true;
    desc.debugName = std::move(debugName);
    return desc;
}

// The descriptor of a native wrap of an engine *staging* buffer that the RHI list only copies from.
// The engine fills these buffers with plain host stores (persistently mapped, host coherent,
// AutoBuffer.cpp:52-63), so NVRHI has never seen a write and, without a claim, would report an
// unknown prior state on the first use (state-tracking.cpp:290-297). CopySource is the state
// copyBuffer requires of the source; claiming it with keepInitialState keeps the copy
// transition-free on every list, which is correct because the engine never writes the buffer from
// the GPU. This is RhiAccelStructs::MakeCopySourceBufferDesc's shape (RhiAccelStructs.cpp:
// 100-116), repeated here because that helper is file-local to the other module.
nvrhi::BufferDesc MakeCopySourceBufferDesc(uint64_t byteSize, std::string debugName)
{
    nvrhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.initialState = nvrhi::ResourceStates::CopySource;
    desc.keepInitialState = true;
    desc.debugName = std::move(debugName);
    return desc;
}

void LogMessage(const RhiRtDirectPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtDirectPass::RhiRtDirectPass() = default;

RhiRtDirectPass::~RhiRtDirectPass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images and buffers and the shader table references the
        // pipeline; the host destroys the pass while it can still idle the device (VulkanDevice
        // does that before the skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    // The shader table holds a reference to its pipeline (vulkan-backend.h:959), so it goes first;
    // both are reference-counted and the order below only documents the dependency.
    shaderTable = nullptr;
    pipeline = nullptr;

    for (Target &target : targets)
    {
        target.framebufferSet = nullptr;
        for (nvrhi::TextureHandle &texture : target.framebufferTextures)
        {
            texture = nullptr;
        }
        target.tlasSet = nullptr;
        target.topLevel = nullptr;
        target.uniformSet = nullptr;
        target.uniformBuffer = nullptr;
        target.vertexDataSet = nullptr;
        for (nvrhi::IBuffer *&buffer : target.vertexBuffers)
        {
            buffer = nullptr;
        }
        target.lightSet = nullptr;
        for (nvrhi::BufferHandle &wrap : target.lightWraps)
        {
            wrap = nullptr;
        }
        for (nvrhi::BufferHandle &wrap : target.lightStagingWraps)
        {
            wrap = nullptr;
        }
        std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
        std::memset(target.lightHandles, 0, sizeof(target.lightHandles));
        std::memset(target.lightStagingHandles, 0, sizeof(target.lightStagingHandles));
        target.width = 0;
        target.height = 0;
    }

    lightLayout = nullptr;
    framebufferLayout = nullptr;
    anyHitShader = nullptr;
    closestHitShader = nullptr;
    shadowMissShader = nullptr;
    missShader = nullptr;
    raygenShader = nullptr;
}

bool RhiRtDirectPass::Create(nvrhi::IDevice *pDevice,
                             rhi::RhiFrameContext *pFrameContext,
                             rhi::RhiTextureTable *pTextureTable,
                             const LightManager *pLightManager,
                             const RhiRtPrimaryPass *pPrimaryPass,
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
    textureTable = pTextureTable;
    lightManager = pLightManager;
    primaryPass = pPrimaryPass;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs the frame context of the RHI layer");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated() ||
        textureTable->GetLayout() == nullptr || textureTable->GetTable() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs the shared RHI texture table (set 4)");
        return false;
    }

    if (lightManager == nullptr)
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs the engine's light manager (set 6)");
        return false;
    }

    // The shared layouts, the empty set of 5/7/8/9/10 and the ray-stats set of 11 come from the
    // primary pass: NVRHI accepts a state's set only when it was created over the very layout
    // handle the pipeline declared (validation-commandlist.cpp:509-520), so this module can share
    // those positions but not re-create them. The primary pass therefore has to be created first
    // and has to outlive this object.
    if (primaryPass == nullptr || !primaryPass->IsCreated() ||
        primaryPass->GetTlasLayout() == nullptr || primaryPass->GetUniformLayout() == nullptr ||
        primaryPass->GetVertexDataLayout() == nullptr || primaryPass->GetHoleLayout() == nullptr ||
        primaryPass->GetHoleSet() == nullptr || primaryPass->GetRayStatsLayout() == nullptr ||
        primaryPass->GetRayStatsSet() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs the created primary RT pass for its shared set layouts");
        return false;
    }

    // The pipeline declares twelve binding layouts, so the pinned NVRHI's cap has to be the raised
    // one: the module is written against `c_MaxBindingLayouts == 16` (the patch build_win.ps1
    // applies for the duration of a build, third_party/nvrhi-max-binding-layouts.patch), exactly as
    // RhiRtPrimaryPass is. 'if constexpr' keeps the check out of the compiled path of a patched
    // build.
    if constexpr (nvrhi::c_MaxBindingLayouts < PIPELINE_SET_COUNT)
    {
        LogMessage(print, "Warning: RHI: the direct RT pass needs c_MaxBindingLayouts >= " +
                          std::to_string(PIPELINE_SET_COUNT) +
                          " (apply third_party/nvrhi-max-binding-layouts.patch)");
        return false;
    }

    if (!LoadShader(RAYGEN_SHADER_FILE_NAME, nvrhi::ShaderType::RayGeneration, raygenShader) ||
        !LoadShader(MISS_SHADER_FILE_NAME, nvrhi::ShaderType::Miss, missShader) ||
        !LoadShader(SHADOW_MISS_SHADER_FILE_NAME, nvrhi::ShaderType::Miss, shadowMissShader) ||
        !LoadShader(CLOSEST_HIT_SHADER_FILE_NAME, nvrhi::ShaderType::ClosestHit, closestHitShader) ||
        !LoadShader(ANY_HIT_SHADER_FILE_NAME, nvrhi::ShaderType::AnyHit, anyHitShader))
    {
        return false;
    }

    // The two layouts this module owns. Every one is AllRayTracing visibility, for the reason
    // RhiDebugTracePass documents: the acceleration-structure-read barrier names the compute stage
    // too (vulkan-constants.cpp:282-285) and this device has no rayQuery feature to make that
    // legal (the A3.1 fix).
    {
        // Set 1: one item per image of `RtgDirect`, at the engine's raw binding of that image. A
        // partial layout in the same sense the primary's set 1 is - only the 12 bindings the
        // raygen declares have an item. Both kinds keep NVRHI's default shader-resource offset (0)
        // and the UAV offset is set to 0 explicitly (its default is 384, nvrhi.h:2063-2066), so an
        // item's slot is exactly the raw binding the generated arrays carry.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setShaderResourceOffset(0)
                                   .setUnorderedAccessViewOffset(0));

        for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
        {
            const uint32_t rawBinding = GetFramebufferRawBinding(FRAMEBUFFER_BINDINGS[i]);

            desc.addItem(FRAMEBUFFER_BINDINGS[i].isUAV
                             ? nvrhi::BindingLayoutItem::Texture_UAV(rawBinding)
                             : nvrhi::BindingLayoutItem::Texture_SRV(rawBinding));
        }

        framebufferLayout = device->createBindingLayout(desc);
    }
    {
        // Set 6: the five light buffers at their raw bindings, the same offsets as set 1. The
        // statistics buffer is the only UAV item; the other four are structured SRVs. The partial
        // shape is right here too: the engine's layout has nine items, the raygen declares five.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setShaderResourceOffset(0)
                                   .setUnorderedAccessViewOffset(0));

        for (uint32_t i = 0; i < LIGHT_BUFFER_COUNT; i++)
        {
            desc.addItem(LIGHT_BUFFER_BINDINGS[i].isUAV
                             ? nvrhi::BindingLayoutItem::StructuredBuffer_UAV(LIGHT_BUFFER_BINDINGS[i].binding)
                             : nvrhi::BindingLayoutItem::StructuredBuffer_SRV(LIGHT_BUFFER_BINDINGS[i].binding));
        }

        lightLayout = device->createBindingLayout(desc);
    }

    if (framebufferLayout == nullptr || lightLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a direct RT pass binding layout");
        return false;
    }

    // The pipeline: one raygen (RGenDirect, no specialization), the engine's two miss shaders as
    // two GENERAL groups, and the engine's two hit groups in the order the instance records address
    // (fully opaque first, alpha tested second). The twelve layouts are added in set order; the
    // shared positions (0, 2, 3, 5, 7, 8, 9, 10, 11) are the primary pass's own handles, and set 4
    // is the table's layout, so a set of the host's or the primary's makes it through
    // validation-commandlist.cpp:509-520. Per-shader and per-hit-group binding layouts stay null -
    // the Vulkan backend rejects them with NotSupported (vulkan-raytracing.cpp:1531-1535,
    // :1543-1547).
    {
        nvrhi::rt::PipelineDesc desc;

        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(RAYGEN_EXPORT_NAME)
                           .setShader(raygenShader));
        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(MISS_EXPORT_NAME)
                           .setShader(missShader));
        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(SHADOW_MISS_EXPORT_NAME)
                           .setShader(shadowMissShader));
        desc.addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
                             .setExportName(HIT_GROUP_OPAQUE_EXPORT_NAME)
                             .setClosestHitShader(closestHitShader));
        desc.addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
                             .setExportName(HIT_GROUP_ALPHA_TESTED_EXPORT_NAME)
                             .setClosestHitShader(closestHitShader)
                             .setAnyHitShader(anyHitShader));

        desc.addBindingLayout(primaryPass->GetTlasLayout());           // 0
        desc.addBindingLayout(framebufferLayout);                      // 1
        desc.addBindingLayout(primaryPass->GetUniformLayout());        // 2
        desc.addBindingLayout(primaryPass->GetVertexDataLayout());     // 3
        desc.addBindingLayout(textureTable->GetLayout());              // 4
        desc.addBindingLayout(primaryPass->GetHoleLayout());           // 5
        desc.addBindingLayout(lightLayout);                            // 6
        desc.addBindingLayout(primaryPass->GetHoleLayout());           // 7
        desc.addBindingLayout(primaryPass->GetHoleLayout());           // 8
        desc.addBindingLayout(primaryPass->GetHoleLayout());           // 9
        desc.addBindingLayout(primaryPass->GetHoleLayout());           // 10
        desc.addBindingLayout(primaryPass->GetRayStatsLayout());       // 11

        desc.setMaxPayloadSize(MAX_PAYLOAD_SIZE);
        desc.setMaxAttributeSize(MAX_ATTRIBUTE_SIZE);
        desc.setMaxRecursionDepth(MAX_RECURSION_DEPTH);

        pipeline = device->createRayTracingPipeline(desc);
    }

    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the direct RT pass ray-tracing pipeline");
        return false;
    }

    // The table: one raygen, two misses and two hit groups, at the engine's own SBT indices. The
    // returned indices are verified against the SBT_INDEX_* contract, so a shader-table shape
    // regression cannot pass silently - the shadow rays address miss 1, so the order is load
    // bearing. Uncached (the ShaderTableDesc default): with five entries the per-list bake of the
    // uncached path is cheaper than a cached buffer (vulkan-raytracing.cpp:1710-1736). The
    // per-entry binding-set argument must stay null on Vulkan (:1771-1836).
    shaderTable = pipeline->createShaderTable(nvrhi::rt::ShaderTableDesc());
    if (shaderTable == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the direct RT pass shader table");
        return false;
    }

    shaderTable->setRayGenerationShader(RAYGEN_EXPORT_NAME);

    if (shaderTable->addMissShader(MISS_EXPORT_NAME) != SBT_INDEX_MISS_DEFAULT ||
        shaderTable->addMissShader(SHADOW_MISS_EXPORT_NAME) != SBT_INDEX_MISS_SHADOW ||
        shaderTable->addHitGroup(HIT_GROUP_OPAQUE_EXPORT_NAME) != SBT_INDEX_HITGROUP_FULLY_OPAQUE ||
        shaderTable->addHitGroup(HIT_GROUP_ALPHA_TESTED_EXPORT_NAME) != SBT_INDEX_HITGROUP_ALPHA_TESTED)
    {
        // findShaderGroup already logged the missing export name through the device's message
        // callback; the pass only records that its table is unusable.
        LogMessage(print, "Warning: RHI: the direct RT pass shader table does not match the engine's SBT indices");
        shaderTable = nullptr;
        return false;
    }

    created = true;
    return true;
}

void RhiRtDirectPass::Render(nvrhi::ICommandList *pCommandList,
                             uint32_t frameIndex,
                             nvrhi::rt::IAccelStruct *pTopLevel,
                             nvrhi::IBuffer *pUniformBuffer,
                             const RhiRtPrimaryPass::VertexData &vertexData,
                             const Framebuffers *pFramebuffers,
                             uint32_t width,
                             uint32_t height)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    if (width == 0 || height == 0)
    {
        return;
    }

    if (pTopLevel == nullptr)
    {
        if (!warnedMissingTopLevel)
        {
            warnedMissingTopLevel = true;
            LogMessage(print, "Warning: RHI: the direct RT pass got no top-level acceleration structure, the trace is skipped");
        }
        return;
    }

    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the direct RT pass got no engine framebuffers, the trace is skipped");
        }
        return;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the direct RT pass got no global uniform, the trace is skipped");
        }
        return;
    }

    if (!vertexData.IsComplete())
    {
        if (!warnedMissingVertexData)
        {
            warnedMissingVertexData = true;
            LogMessage(print, "Warning: RHI: the direct RT pass got no vertex-data buffers, the trace is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // The 12 images of this slot. `GetImageHandles` resolves the engine's ping-pong swap
    // (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's current
    // image, which is the one the engine's own per-slot descriptor set binds to the variable's
    // fixed raw binding - the shader never sees the swap. The 4-tuple overload is what supplies the
    // one image that is not render-sized: Q2_GRAD_SMPL_POS carries FORCE_SIZE_1_3
    // (ShaderCommonCFramebuf.cpp:252) and its extent is ((w+1)/3, (h+1)/3) (Framebuffers.cpp:
    // 654-691); the render-sized images answer with exactly the width/height passed in. None of
    // these images is upscaled-sized, so the render size stands in for the upscaled fields of the
    // resolution state - a future set-1 image with that flag would be a wrong wrap and be caught by
    // the null-wrap check below.
    const ResolutionState resolutionState = { width, height, width, height };

    uint64_t imageHandles[FRAMEBUFFER_BINDING_COUNT] = {};
    uint64_t imageViews[FRAMEBUFFER_BINDING_COUNT] = {};
    VkFormat imageFormats[FRAMEBUFFER_BINDING_COUNT] = {};
    VkExtent2D imageExtents[FRAMEBUFFER_BINDING_COUNT] = {};

    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        const auto [image, view, format, extent] =
            pFramebuffers->GetImageHandles(FRAMEBUFFER_BINDINGS[i].image, frameIndex, resolutionState);

        imageHandles[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));
        imageViews[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        imageFormats[i] = format;
        imageExtents[i] = extent;

        if (imageHandles[i] == 0)
        {
            if (!warnedMissingFramebuffers)
            {
                warnedMissingFramebuffers = true;
                LogMessage(print, "Warning: RHI: the direct RT pass got no framebuffer image, the trace is skipped");
            }
            return;
        }
    }

    bool framebuffersChanged = target.width != width || target.height != height;
    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        framebuffersChanged = framebuffersChanged || target.imageHandles[i] != imageHandles[i];
    }

    if (framebuffersChanged)
    {
        // The engine re-created its framebuffers (a resize or an image-format change): the wraps
        // and the set over them are retired and rebuilt. Retiring first is safe even if the rebuild
        // below fails - the next frame retries, and the trace is skipped until it succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtDirectPass ") +
                                          ShFramebuffers_DebugNames[FRAMEBUFFER_BINDINGS[i].image] +
                                          " frame " + std::to_string(frameIndex);

            target.framebufferTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i],
                imageExtents[i].width, imageExtents[i].height, debugName);

            if (target.framebufferTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap a direct RT pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
    }

    // Set 1 over the wraps; rebuilt exactly when ReleaseFramebufferTarget nulled it. The item
    // slots are the raw bindings, because the layout's offsets are both 0.
    if (target.framebufferSet == nullptr)
    {
        nvrhi::BindingSetDesc setDesc;

        for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
        {
            const uint32_t rawBinding = GetFramebufferRawBinding(FRAMEBUFFER_BINDINGS[i]);

            setDesc.addItem(FRAMEBUFFER_BINDINGS[i].isUAV
                                ? nvrhi::BindingSetItem::Texture_UAV(rawBinding, target.framebufferTextures[i])
                                : nvrhi::BindingSetItem::Texture_SRV(rawBinding, target.framebufferTextures[i]));
        }

        target.framebufferSet = device->createBindingSet(setDesc, framebufferLayout);

        if (target.framebufferSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the direct RT pass framebuffer binding set");
            return;
        }
    }

    // The per-slot sets 0, 2 and 3, over the primary's layout handles. Their keys are the pointers
    // the host passes, so a re-created AS or a re-created vertex-data buffer is picked up without a
    // second Create.
    if (!PrepareTlasSet(target, pTopLevel) ||
        !PrepareUniformSet(target, pUniformBuffer) ||
        !PrepareVertexDataSet(target, vertexData))
    {
        return;
    }

    // Set 6 and the four light copies. The copies go before the state below, so the automatic
    // barriers order each copy before the dispatch that reads the copied buffer.
    if (!PrepareLightSet(pCommandList, target, frameIndex))
    {
        return;
    }

    // The image state contract, spelled out on Render in the header: the engine leaves every
    // framebuffer image in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native
    // wrap keeps no state between command lists (RhiTextureSource.h), so every list announces that
    // state for all 12 images before the first use. The primary pass wrote the images this pass
    // samples, but it wrote them through its own wraps; the physical layout it left them in is
    // exactly this GENERAL state, so the announcement is the truth and the SRV bindings' automatic
    // transitions start from it.
    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.framebufferTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    // The shared table's slots wrapped since the last frame need their first-use state declared in
    // the list that samples them (RhiTextureTable.h); the primary pass does this too, and a
    // repeated call is a no-op.
    textureTable->TrackPendingTextures(pCommandList);

    // The twelve sets, in the layout order the pipeline was built with; the pinned backend's legacy
    // binding mode binds the list positionally (vulkan-resource-bindings.cpp:940-958). Sets 5, 7, 8,
    // 9 and 10 are the primary's empty set: neither the raygen nor any hit/miss stage of this
    // pipeline declares those sets, but the automatic-barrier pass dereferences every entry, so a
    // real set has to fill each position.
    nvrhi::rt::State state;
    state.setShaderTable(shaderTable);
    state.addBindingSet(target.tlasSet);                  // 0
    state.addBindingSet(target.framebufferSet);           // 1
    state.addBindingSet(target.uniformSet);               // 2
    state.addBindingSet(target.vertexDataSet);            // 3
    state.addBindingSet(textureTable->GetTable());        // 4
    state.addBindingSet(primaryPass->GetHoleSet());       // 5
    state.addBindingSet(target.lightSet);                 // 6
    state.addBindingSet(primaryPass->GetHoleSet());       // 7
    state.addBindingSet(primaryPass->GetHoleSet());       // 8
    state.addBindingSet(primaryPass->GetHoleSet());       // 9
    state.addBindingSet(primaryPass->GetHoleSet());       // 10
    state.addBindingSet(primaryPass->GetRayStatsSet());   // 11

    pCommandList->setRayTracingState(state);

    // One ray per pixel at the full render resolution. The raygen's DispatchRaysIndex is the
    // checkerboard-packed texel itself: the shader reads the G-buffer at that index
    // (`fetchGbufferSurface`, Surface.hlsli:77-108, which maps it to the regular pixel only for
    // ALBEDO) and stores into the two direct images through it (`imageStoreUnfilteredDirect(pix,
    // ...)`, RtRaygenDirect.rgen.hlsl:186-187). The packed images are created at the render size
    // (none of the set-1 flags forces a smaller extent for them), so the w x h dispatch covers
    // every packed texel once and the dispatch is never halved.
    nvrhi::rt::DispatchRaysArguments args;
    args.setDimensions(width, height, 1);
    pCommandList->dispatchRays(args);

    // The sampled images end the list in the read-only layout (vulkan-resource-bindings.cpp:
    // 398-435), but the engine's framebuffer images rest in GENERAL and the next frame's primary
    // pass writes 8 of these 9 as storage images. Move every image that is bound as an SRV and by
    // no UAV back to UnorderedAccess; the three UAV images are already there.
    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        if (!FRAMEBUFFER_BINDINGS[i].isUAV)
        {
            pCommandList->setTextureState(
                target.framebufferTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }
    }
}

void RhiRtDirectPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtDirectPass::ReleaseFramebufferTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the wraps reference engine images the GPU may still be writing through the UAVs, and
    // the set references those wraps. The queue takes its reference now, so the handles below can
    // be cleared immediately.
    if (frameContext != nullptr)
    {
        if (target.framebufferSet != nullptr)
        {
            frameContext->Retire(target.framebufferSet);
        }

        for (nvrhi::TextureHandle &texture : target.framebufferTextures)
        {
            if (texture != nullptr)
            {
                frameContext->Retire(texture);
            }
        }
    }

    target.framebufferSet = nullptr;

    for (nvrhi::TextureHandle &texture : target.framebufferTextures)
    {
        texture = nullptr;
    }

    std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
    target.width = 0;
    target.height = 0;
}

void RhiRtDirectPass::ReleaseTarget(Target &target)
{
    ReleaseFramebufferTarget(target);

    if (frameContext != nullptr)
    {
        if (target.tlasSet != nullptr)
        {
            frameContext->Retire(target.tlasSet);
        }
        if (target.uniformSet != nullptr)
        {
            frameContext->Retire(target.uniformSet);
        }
        if (target.vertexDataSet != nullptr)
        {
            frameContext->Retire(target.vertexDataSet);
        }
    }

    target.tlasSet = nullptr;
    target.topLevel = nullptr;
    target.uniformSet = nullptr;
    target.uniformBuffer = nullptr;
    target.vertexDataSet = nullptr;

    for (nvrhi::IBuffer *&buffer : target.vertexBuffers)
    {
        buffer = nullptr;
    }

    // The light wraps too, so a resize leaves no per-slot state behind; the engine's buffers
    // survive it, and the next Render re-wraps from LightManager::GetBuffers().
    ReleaseLightTarget(target);
}

void RhiRtDirectPass::ReleaseLightTarget(Target &target)
{
    if (frameContext != nullptr)
    {
        if (target.lightSet != nullptr)
        {
            frameContext->Retire(target.lightSet);
        }

        for (nvrhi::BufferHandle &wrap : target.lightWraps)
        {
            if (wrap != nullptr)
            {
                frameContext->Retire(wrap);
            }
        }

        for (nvrhi::BufferHandle &wrap : target.lightStagingWraps)
        {
            if (wrap != nullptr)
            {
                frameContext->Retire(wrap);
            }
        }
    }

    target.lightSet = nullptr;

    for (nvrhi::BufferHandle &wrap : target.lightWraps)
    {
        wrap = nullptr;
    }

    for (nvrhi::BufferHandle &wrap : target.lightStagingWraps)
    {
        wrap = nullptr;
    }

    std::memset(target.lightHandles, 0, sizeof(target.lightHandles));
    std::memset(target.lightStagingHandles, 0, sizeof(target.lightStagingHandles));
}

bool RhiRtDirectPass::PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel)
{
    if (target.tlasSet != nullptr && target.topLevel == pTopLevel)
    {
        return true;
    }

    // The AS object is per frame slot and can be replaced when the scene changes; the set holds a
    // reference that keeps it alive, so the replaced one goes through the retire queue.
    if (target.tlasSet != nullptr)
    {
        frameContext->Retire(target.tlasSet);
    }
    target.tlasSet = nullptr;

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(BINDING_ACCELERATION_STRUCTURE_MAIN, pTopLevel));

    target.tlasSet = device->createBindingSet(setDesc, primaryPass->GetTlasLayout());

    if (target.tlasSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the direct RT pass TLAS binding set");
        return false;
    }

    target.topLevel = pTopLevel;
    return true;
}

bool RhiRtDirectPass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformSet != nullptr && target.uniformBuffer == pUniformBuffer)
    {
        return true;
    }

    // The shader declares set 2 as ConstantBuffer<ShGlobalUniform>; the validation device refuses
    // such a binding on a buffer whose desc lacks isConstantBuffer (validation-device.cpp:1717-1723)
    // and on a volatile one (:1725-1730, which would also become a dynamic-offset binding the
    // static layout item cannot take).
    if (!pUniformBuffer->getDesc().isConstantBuffer || pUniformBuffer->getDesc().isVolatile)
    {
        if (!warnedBadUniform)
        {
            warnedBadUniform = true;
            LogMessage(print, "Warning: RHI: the direct RT pass needs the global uniform as a static constant-buffer wrap");
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

    target.uniformSet = device->createBindingSet(setDesc, primaryPass->GetUniformLayout());

    if (target.uniformSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the direct RT pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

bool RhiRtDirectPass::PrepareVertexDataSet(Target &target, const RhiRtPrimaryPass::VertexData &vertexData)
{
    nvrhi::IBuffer *const buffers[VERTEX_DATA_BUFFER_COUNT] =
    {
        vertexData.staticVertices,
        vertexData.dynamicVertices,
        vertexData.staticIndices,
        vertexData.dynamicIndices,
        vertexData.geometryInstances,
        vertexData.dynamicVerticesPrev,
        vertexData.prevDynamicIndices,
    };

    bool sameBuffers = target.vertexDataSet != nullptr;
    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        sameBuffers = sameBuffers && target.vertexBuffers[i] == buffers[i];
    }

    if (sameBuffers)
    {
        return true;
    }

    // The backend asserts a non-zero structStride when a structured-buffer binding is written
    // (vulkan-resource-bindings.cpp:535-536) and the validation device refuses it too
    // (validation-device.cpp:1693-1699), so a wrap that lost its stride is a host-side setup error,
    // not a per-frame condition.
    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        if (buffers[i]->getDesc().structStride == 0)
        {
            if (!warnedBadVertexData)
            {
                warnedBadVertexData = true;
                LogMessage(print, "Warning: RHI: the direct RT pass vertex-data buffers need a non-zero structStride");
            }
            return false;
        }
    }

    if (target.vertexDataSet != nullptr)
    {
        frameContext->Retire(target.vertexDataSet);
    }
    target.vertexDataSet = nullptr;

    nvrhi::BindingSetDesc setDesc;
    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(VERTEX_DATA_BINDINGS[i], buffers[i]));
    }

    target.vertexDataSet = device->createBindingSet(setDesc, primaryPass->GetVertexDataLayout());

    if (target.vertexDataSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the direct RT pass vertex-data binding set");
        return false;
    }

    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        target.vertexBuffers[i] = buffers[i];
    }

    return true;
}

bool RhiRtDirectPass::PrepareLightSet(nvrhi::ICommandList *pCommandList, Target &target, uint32_t frameIndex)
{
    const LightManager::Buffers buffers = lightManager->GetBuffers();
    const VkBuffer rawBuffers[LIGHT_BUFFER_COUNT] =
    {
        buffers.lights,
        buffers.listOffsets,
        buffers.listLights,
        buffers.lightStats,
        buffers.clusterSkyVis,
    };

    for (uint32_t i = 0; i < LIGHT_BUFFER_COUNT; i++)
    {
        if (rawBuffers[i] == VK_NULL_HANDLE)
        {
            if (!warnedMissingLights)
            {
                warnedMissingLights = true;
                LogMessage(print, "Warning: RHI: the direct RT pass got no engine light buffer, the trace is skipped");
            }
            return false;
        }
    }

    bool buffersChanged = false;
    for (uint32_t i = 0; i < LIGHT_BUFFER_COUNT; i++)
    {
        buffersChanged = buffersChanged || target.lightHandles[i] != rawBuffers[i];
    }

    if (buffersChanged)
    {
        // A light buffer handle changed (the engine re-created its buffers): the wraps, the set
        // over them and this slot's copy sources are retired and rebuilt from the new handles.
        ReleaseLightTarget(target);

        for (uint32_t i = 0; i < LIGHT_BUFFER_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtDirectPass ") +
                                          LIGHT_BUFFER_DEBUG_NAMES[i] +
                                          " frame " + std::to_string(frameIndex);

            target.lightWraps[i] = device->createHandleForNativeBuffer(
                nvrhi::ObjectTypes::VK_Buffer,
                nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rawBuffers[i]))),
                MakeLightBufferDesc(LIGHT_BUFFER_BINDINGS[i], debugName));

            if (target.lightWraps[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap an engine light buffer for the direct RT pass");
                ReleaseLightTarget(target);
                return false;
            }
        }

        std::memcpy(target.lightHandles, rawBuffers, sizeof(target.lightHandles));
    }

    if (target.lightSet == nullptr)
    {
        nvrhi::BindingSetDesc setDesc;

        for (uint32_t i = 0; i < LIGHT_BUFFER_COUNT; i++)
        {
            setDesc.addItem(LIGHT_BUFFER_BINDINGS[i].isUAV
                                ? nvrhi::BindingSetItem::StructuredBuffer_UAV(LIGHT_BUFFER_BINDINGS[i].binding,
                                                                               target.lightWraps[i])
                                : nvrhi::BindingSetItem::StructuredBuffer_SRV(LIGHT_BUFFER_BINDINGS[i].binding,
                                                                               target.lightWraps[i]));
        }

        target.lightSet = device->createBindingSet(setDesc, lightLayout);

        if (target.lightSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the direct RT pass light-source binding set");
            return false;
        }
    }

    // The frame's light copies, the RHI replacement for the `LightManager::CopyFromStaging` calls
    // the `rhiframe` path bypasses (a42_recon.md §2.2, §5). `Copy::size == 0` means nothing is
    // pending for that item. The prefix copy of the light array runs every frame and includes the
    // sun slot at index 0; the two list buffers and the sky-visibility table run while their
    // publication/update is pending. Note that this accessor reports, it does not consume: the
    // pending flags live in the engine and the legacy path that clears them does not run under
    // `rhiframe`, so an unchanged list buffer is copied again on every frame that binds this set.
    // That is redundant traffic, not a correctness issue - the per-slot staging holds exactly the
    // words `GetFrameCopies` measures.
    const LightManager::FrameCopies copies = lightManager->GetFrameCopies(frameIndex);
    const LightManager::Copy copyItems[LIGHT_COPY_COUNT] =
    {
        copies.lights,
        copies.listOffsets,
        copies.listLights,
        copies.clusterSkyVis,
    };

    for (uint32_t i = 0; i < LIGHT_COPY_COUNT; i++)
    {
        const LightManager::Copy &copy = copyItems[i];

        if (copy.size == 0)
        {
            continue;
        }

        if (copy.staging == VK_NULL_HANDLE)
        {
            if (!warnedMissingLightStaging)
            {
                warnedMissingLightStaging = true;
                LogMessage(print, "Warning: RHI: the direct RT pass got no light-staging buffer, the trace is skipped");
            }
            return false;
        }

        const uint32_t bufferIndex = LIGHT_COPY_BUFFER_INDICES[i];

        if (target.lightStagingHandles[i] != copy.staging)
        {
            if (target.lightStagingWraps[i] != nullptr)
            {
                frameContext->Retire(target.lightStagingWraps[i]);
            }

            // The key is cleared before the wrap that replaces it is made: a failed creation must
            // not leave the old handle as the key of a null wrap, or the next frame that brings the
            // same staging back would copy from a null source.
            target.lightStagingHandles[i] = VK_NULL_HANDLE;
            target.lightStagingWraps[i] = nullptr;

            const std::string debugName = std::string("RhiRtDirectPass ") +
                                          LIGHT_BUFFER_DEBUG_NAMES[bufferIndex] +
                                          " staging frame " + std::to_string(frameIndex);

            target.lightStagingWraps[i] = device->createHandleForNativeBuffer(
                nvrhi::ObjectTypes::VK_Buffer,
                nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(copy.staging))),
                MakeCopySourceBufferDesc(LIGHT_BUFFER_BINDINGS[bufferIndex].byteSize, debugName));

            if (target.lightStagingWraps[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap an engine light-staging buffer for the direct RT pass");
                return false;
            }

            target.lightStagingHandles[i] = copy.staging;
        }

        // The destination buffer is in this list's CopyDest claim, the source in its CopySource
        // claim, so the automatic barrier pass needs no transition for either side; the shader use
        // later in the list is what moves the destination to its shader state.
        pCommandList->copyBuffer(target.lightWraps[bufferIndex], 0, target.lightStagingWraps[i], 0, copy.size);
    }

    return true;
}

bool RhiRtDirectPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the direct RT pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
