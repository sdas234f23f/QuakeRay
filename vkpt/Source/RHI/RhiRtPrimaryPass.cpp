#include "RhiRtPrimaryPass.h"

#include "RhiDescriptors.h"
#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"
#include "RhiTextureTable.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"

#include <cstring>
#include <string>
#include <tuple>

using namespace vkpt;

namespace
{

// The five engine blobs, by the file names the shader build writes into the folder the host passes
// in. The names are ShaderManager's (`ShaderManager.cpp:41-49`), so the RHI and the legacy renderer
// run off the same blobs. The shadow miss is not entered by this raygen - `RtRaygenPrimary.rgen`
// traces one ray with miss index SBT_INDEX_MISS_DEFAULT - but it is loaded and tabled anyway: the
// engine's SBT indices are fixed (SBT_INDEX_MISS_SHADOW is 1), and the table the engine's other
// raygens will need has the same shape.
const char *const RAYGEN_SHADER_FILE_NAME = "RtRaygenPrimary.rgen.spv";
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
// (instanceShaderBindingTableRecordOffset 0/1, ASManager.cpp:976-991).
const char *const RAYGEN_EXPORT_NAME = "RhiRtPrimaryRGen";
const char *const MISS_EXPORT_NAME = "RhiRtPrimaryRMiss";
const char *const SHADOW_MISS_EXPORT_NAME = "RhiRtPrimaryRMissShadow";
const char *const HIT_GROUP_OPAQUE_EXPORT_NAME = "RhiRtPrimaryRClsOpaque";
const char *const HIT_GROUP_ALPHA_TESTED_EXPORT_NAME = "RhiRtPrimaryRClsOpaqueAlphaTested";

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

// The 26 storage images `RtRaygenPrimary.rgen` writes, by the engine's image indices (measured with
// spirv-dis over the shipped blob: the 26 UAVs are at these raw bindings, and the engine's
// image-to-binding table is the identity at these indices, so the slot below is the index). The
// other 98 framebuffer images are neither referenced by the raygen's interface nor declared by its
// module, so set 1 stays a partial layout exactly like the world pass's framebuffer layout - an
// unwritten binding needs no descriptor (a4_recon §2.3).
constexpr uint32_t FRAMEBUFFER_UAV_COUNT = 26;
constexpr FramebufferImageIndex FRAMEBUFFER_UAV_IMAGES[FRAMEBUFFER_UAV_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,                       //  0  framebufAlbedo
    FB_IMAGE_INDEX_IS_SKY,                       //  2  framebufIsSky
    FB_IMAGE_INDEX_NORMAL,                       //  3  framebufNormal
    FB_IMAGE_INDEX_NORMAL_GEOMETRY,              //  5  framebufNormalGeometry
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,           //  7  framebufMetallicRoughness
    FB_IMAGE_INDEX_DEPTH_WORLD,                  //  9  framebufDepthWorld
    FB_IMAGE_INDEX_DEPTH_GRAD,                   // 11  framebufDepthGrad
    FB_IMAGE_INDEX_DEPTH_NDC,                    // 12  framebufDepthNdc
    FB_IMAGE_INDEX_MOTION,                       // 13  framebufMotion
    FB_IMAGE_INDEX_SURFACE_POSITION,             // 19  framebufSurfacePosition
    FB_IMAGE_INDEX_VISIBILITY_BUFFER,            // 21  framebufVisibilityBuffer
    FB_IMAGE_INDEX_VIEW_DIRECTION,               // 23  framebufViewDirection
    FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,         // 25  framebufPrimaryToReflRefr
    FB_IMAGE_INDEX_THROUGHPUT,                   // 26  framebufThroughput
    FB_IMAGE_INDEX_MOTION_DLSS,                  // 31  framebufMotionDlss
    FB_IMAGE_INDEX_ACID_FOG_R_T,                 // 59  framebufAcidFogRT
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,              // 61  framebufScreenEmisRT
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH,                // 81  framebufQ2ViewDepth
    FB_IMAGE_INDEX_Q2_BASE_COLOR,                // 83  framebufQ2BaseColor
    FB_IMAGE_INDEX_Q2_METALLIC,                  // 85  framebufQ2Metallic
    FB_IMAGE_INDEX_Q2_BOUNCE_THROUGHPUT,         // 87  framebufQ2BounceThroughput
    FB_IMAGE_INDEX_Q2_TRANSPARENT,               // 88  framebufQ2Transparent
    FB_IMAGE_INDEX_Q2_GOD_RAYS_THROUGHPUT_DIST,  // 89  framebufQ2GodRaysThroughputDist
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,                 // 90  framebufQ2FogAccum
    FB_IMAGE_INDEX_Q2_RNG_SEED,                  // 121 framebufQ2RngSeed
    FB_IMAGE_INDEX_Q2_CLUSTER,                   // 123 framebufQ2Cluster
};

// Set 3's bindings: vertex data, indices and the geometry-instance buffer `getTriangle` reads.
// Binding 5 (`geomIndexPrevToCur`) is left out because the raygen's module does not reference it;
// the rest are the eight declarations of VertexData.hlsli, seven of which are statically used.
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

// The specialization constant of `RtRaygenPrimary.rgen`: [[vk::constant_id(0)]] const uint
// maxAlbedoLayerCount = 0 (SpecId 0, uint; the blob declares exactly that). The engine passes
// `primaryRaysMaxAlbedoLayers` (RayTracingPipeline.cpp:96), which the game sets to 2
// (Quake/gl_vidsdl.c:1458) and which has to reach the shader through createShaderSpecialization,
// because SpecId 0's SPIR-V default is 0 (a materially different path).
constexpr uint32_t MAX_ALBEDO_LAYERS_SPEC_ID = 0;
constexpr uint32_t PRIMARY_RAYS_MAX_ALBEDO_LAYERS = 2;

// The engine's pipeline-wide RT limits (RayTracingPipeline.cpp:203 and the measured blob structs):
// recursion depth 2; the payload is ShPayload = {float2 baryCoords; uint instIdAndIndex; uint
// geomAndPrimIndex} = 16 B and the hit attribute HitAttributes = {float2 inBaryCoords} = 8 B
// (Structs.hlsli:47-57). NVRHI's two size fields are D3D12-only at this pin, so on Vulkan they are
// documentation; they are kept exact for a future backend switch, as in RhiDebugTracePass.
constexpr uint32_t MAX_RECURSION_DEPTH = 2;
constexpr uint32_t MAX_PAYLOAD_SIZE = 2 * sizeof(float) + 2 * sizeof(uint32_t);
constexpr uint32_t MAX_ATTRIBUTE_SIZE = 2 * sizeof(float);

// The engine's cubemap capacity: CubemapManager.cpp:30 MAX_CUBEMAP_COUNT. The cubemap table of set
// 7 is filled with the placeholder cube up to this capacity, so any `skyCubemapIndex` the uniform
// can carry stays inside the descriptor array.
constexpr uint32_t CUBEMAP_TABLE_CAPACITY = 32;

// RtRayStats = uint counts[RAY_STATS_CATEGORY_COUNT] (RaygenCommon.hlsli:602-605), 5 uints, the
// size of each engine RayStats buffer (RayStats.cpp:59).
constexpr uint32_t RAY_STATS_BUFFER_SIZE = RAY_STATS_CATEGORY_COUNT * sizeof(uint32_t);

// The flags that make a framebuffer image a size other than the render size. None of the 26 images
// above carries one today (measured against the generated tables), so the module may wrap them at
// the render resolution; a future shader change that adds one is caught by Render's check instead
// of producing a wrongly sized native view.
constexpr uint32_t NOT_RENDER_SIZED_FLAGS =
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_2 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_3 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_4 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_8 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_16 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_32 |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_UPSCALED_SIZE |
    FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE;

void LogMessage(const RhiRtPrimaryPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtPrimaryPass::RhiRtPrimaryPass() = default;

RhiRtPrimaryPass::~RhiRtPrimaryPass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images and the shader table references the pipeline; the host
        // destroys the pass while it can still idle the device (VulkanDevice does that before the
        // skeleton as well), so nothing has to go through a retire queue here.
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
        std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
        target.width = 0;
        target.height = 0;
    }

    rayStatsSet = nullptr;
    rayStatsBuffer = nullptr;
    renderCubemapSet = nullptr;
    dummyCubemapSampler = nullptr;
    dummyCubemapTexture = nullptr;
    cubemapTable = nullptr;
    holeSet = nullptr;
    rayStatsLayout = nullptr;
    renderCubemapLayout = nullptr;
    cubemapLayout = nullptr;
    holeLayout = nullptr;
    vertexDataLayout = nullptr;
    uniformLayout = nullptr;
    framebufferLayout = nullptr;
    tlasLayout = nullptr;
    anyHitShader = nullptr;
    closestHitShader = nullptr;
    shadowMissShader = nullptr;
    missShader = nullptr;
    specializedRaygenShader = nullptr;
    raygenShader = nullptr;
}

bool RhiRtPrimaryPass::Create(nvrhi::IDevice *pDevice,
                             rhi::RhiFrameContext *pFrameContext,
                             rhi::RhiTextureTable *pTextureTable,
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

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the primary RT pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the primary RT pass needs the frame context of the RHI layer");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated() ||
        textureTable->GetLayout() == nullptr || textureTable->GetTable() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the primary RT pass needs the shared RHI texture table (set 4)");
        return false;
    }

    // Set 4 has to be the table's own layout and set 7 is another bindless layout, so the pipeline
    // declares twelve binding layouts. The pinned NVRHI's cap is 8 and the build raises it to 16
    // through third_party/nvrhi-max-binding-layouts.patch; without that patch the layout vector
    // would overflow, so the pass refuses to create instead. 'if constexpr' keeps the check out of
    // the compiled path of a patched build.
    if constexpr (nvrhi::c_MaxBindingLayouts < PIPELINE_SET_COUNT)
    {
        LogMessage(print, "Warning: RHI: the primary RT pass needs c_MaxBindingLayouts >= " +
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

    // The raygen is the one stage of this pipeline with a specialization constant. The backend's
    // createShaderSpecialization returns a derived shader that carries the constant into the
    // pipeline's VkSpecializationInfo (vulkan-raytracing.cpp:1591, vulkan-graphics.cpp:167-214) and
    // keeps the base module alive through its reference.
    {
        const nvrhi::ShaderSpecialization specialization =
            nvrhi::ShaderSpecialization::UInt32(MAX_ALBEDO_LAYERS_SPEC_ID, PRIMARY_RAYS_MAX_ALBEDO_LAYERS);

        specializedRaygenShader = device->createShaderSpecialization(raygenShader.Get(), &specialization, 1);

        if (specializedRaygenShader == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to specialize the primary RT pass raygen (maxAlbedoLayerCount)");
            return false;
        }
    }

    // The twelve layouts, in the engine's RT set order. Every one is AllRayTracing visibility, for
    // the reason RhiDebugTracePass documents: the acceleration-structure-read barrier names the
    // compute stage too (vulkan-constants.cpp:282-285) and this device has no rayQuery feature to
    // make that legal (the A3.1 fix). The offsets below turn each shader's raw binding numbers into
    // NVRHI slots: raw binding = offset + slot (RhiPipeline.h).
    {
        // Set 0: the top-level AS at raw binding 0.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(BINDING_ACCELERATION_STRUCTURE_MAIN));

        tlasLayout = device->createBindingLayout(desc);
    }
    {
        // Set 1: one UAV item per storage image, at the engine's raw binding of that image
        // (ShFramebuffers_Bindings, the identity for these 26 indices). A partial layout: a binding
        // the module does not declare needs no descriptor.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setUnorderedAccessViewOffset(0));

        for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
        {
            desc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(ShFramebuffers_Bindings[FRAMEBUFFER_UAV_IMAGES[i]]));
        }

        framebufferLayout = device->createBindingLayout(desc);
    }
    {
        // Set 2: the global uniform at raw binding 0 (DXC's default would put it at 256).
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM));

        uniformLayout = device->createBindingLayout(desc);
    }
    {
        // Set 3: the vertex-data storage buffers at their raw bindings (shaderResource offset 0 is
        // NVRHI's default; it is set explicitly to document that the slot is the raw binding).
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));

        for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
        {
            desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(VERTEX_DATA_BINDINGS[i]));
        }

        vertexDataLayout = device->createBindingLayout(desc);
    }
    {
        // The holes at 5, 6, 9 and 10: a real, zero-item layout. NVRHI accepts it (the validation
        // device only rejects None items, zero sizes and duplicate bindings) and the backend creates
        // a VkDescriptorSetLayout with no bindings for it; the state still needs a real empty set
        // per position, because the automatic-barrier pass dereferences every entry
        // (vulkan-state-tracking.cpp:105) - the same shape the world pass's set-3 hole uses.
        // Visibility still has to be set: the validation device rejects a layout with
        // visibility = None (validation-device.cpp:1340-1344).
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;

        holeLayout = device->createBindingLayout(desc);
        if (holeLayout == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass empty layout");
            return false;
        }
    }
    {
        // Set 7: `globalCubemaps[]` is a runtime descriptor array of cubes (measured in the blob),
        // so the set has to be a bindless table - a regular single-item layout cannot express the
        // engine's array. The register-space order makes the textures binding 0 and the samplers
        // binding 1, the numbers the shader declares.
        nvrhi::BindlessLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.maxCapacity = CUBEMAP_TABLE_CAPACITY;
        desc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::Immutable;
        desc.addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_CUBEMAPS));
        desc.addRegisterSpace(nvrhi::BindingLayoutItem::Sampler(BINDING_CUBEMAPS_SAMPLER));

        cubemapLayout = device->createBindlessLayout(desc);
    }
    {
        // Set 8: `renderCubemap` is a single cube at raw binding 0 and its sampler at raw binding 2
        // (BINDING_RENDER_CUBEMAP / BINDING_RENDER_CUBEMAP_SAMPLER; the env cube at 1/3 is not in
        // the raygen's interface, so the layout stays partial). A sampler offset of 0 keeps the
        // slot as the raw binding.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setSamplerOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_RENDER_CUBEMAP));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(BINDING_RENDER_CUBEMAP_SAMPLER));

        renderCubemapLayout = device->createBindingLayout(desc);
    }
    {
        // Set 11: rtStats at raw binding 0 as an RWStructuredBuffer.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setUnorderedAccessViewOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));

        rayStatsLayout = device->createBindingLayout(desc);
    }

    if (tlasLayout == nullptr || framebufferLayout == nullptr || uniformLayout == nullptr ||
        vertexDataLayout == nullptr || cubemapLayout == nullptr || renderCubemapLayout == nullptr ||
        rayStatsLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a primary RT pass binding layout");
        return false;
    }

    // The real empty set that fills sets 5, 6, 9 and 10. One set bound at four positions is legal
    // Vulkan; the empty layouts are identical, and the backend binds the same VkDescriptorSet to
    // each index of the state's array (vulkan-resource-bindings.cpp:940-1019).
    holeSet = device->createBindingSet(nvrhi::BindingSetDesc(), holeLayout);
    if (holeSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass empty binding set");
        return false;
    }

    // The 1x1 placeholder cube of sets 7 and 8. The engine's cubemaps have no RHI accessor yet
    // (CubemapManager.h:61-62, RenderCubemap.h:77-78) and their content is produced by the legacy
    // sky path, so this pass binds a dummy: the sky modes this pass supports (COLOR / the raster
    // sky) never sample it, and the first list that binds it transitions it out of Common, so even
    // a wrong sky mode reads defined memory instead of undefined layouts. Its contents are
    // undefined; replacing it with the engine's real cubes is the A5 follow-up.
    {
        nvrhi::TextureDesc desc;
        desc.dimension = nvrhi::TextureDimension::TextureCube;
        desc.width = 1;
        desc.height = 1;
        desc.arraySize = 6;
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.isShaderResource = true;
        // The state a Texture_SRV binding requires is derived from the *layout's* visibility:
        // without the pixel stage NVRHI demands NonPixelShaderResource (which maps to the same
        // eShaderReadOnlyOptimal layout as ShaderResource, vulkan-constants.cpp:234-241), so the
        // dummy declares that as its initial state - no transition is needed after the first list.
        desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
        desc.keepInitialState = true;
        desc.debugName = "RhiRtPrimaryPass dummy cubemap";

        dummyCubemapTexture = rhi::createTexture(device, desc, desc.debugName);
        dummyCubemapSampler = rhi::createEngineTextureSampler(device, "RhiRtPrimaryPass dummy cubemap sampler");

        if (dummyCubemapTexture == nullptr || dummyCubemapSampler == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass placeholder cubemap");
            return false;
        }

        cubemapTable = rhi::createTextureTable(device, cubemapLayout, "RhiRtPrimaryPass cubemaps (placeholder)");
        if (cubemapTable == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass cubemap table");
            return false;
        }

        for (uint32_t slot = 0; slot < CUBEMAP_TABLE_CAPACITY; slot++)
        {
            if (!rhi::setTexture(device, cubemapTable, slot, dummyCubemapTexture) ||
                !rhi::setSampler(device, cubemapTable, slot, dummyCubemapSampler))
            {
                LogMessage(print, "Warning: RHI: failed to fill the primary RT pass cubemap table");
                return false;
            }
        }

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_RENDER_CUBEMAP, dummyCubemapTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_RENDER_CUBEMAP_SAMPLER, dummyCubemapSampler));

        renderCubemapSet = device->createBindingSet(setDesc, renderCubemapLayout);
        if (renderCubemapSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass render-cubemap set");
            return false;
        }
    }

    // The ray-stats stand-in. The engine's RayStats has no RHI accessor and its buffer is
    // host-visible and reset by a CPU memset the RHI path never runs (RayStats.cpp:142-149), so
    // the pass owns a small device-local RWStructuredBuffer; the raygen's `rayStatsAdd` only
    // increments it and nothing reads it back, so the contents do not matter. The desc carries the
    // stride and the UAV flag the structured UAV binding requires, and the initial state keeps the
    // automatic-barrier pass from reporting an unknown prior state.
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = RAY_STATS_BUFFER_SIZE;
        desc.structStride = RAY_STATS_BUFFER_SIZE;
        desc.canHaveUAVs = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.keepInitialState = true;
        desc.debugName = "RhiRtPrimaryPass ray stats";

        rayStatsBuffer = rhi::createBuffer(device, desc, desc.debugName);

        if (rayStatsBuffer != nullptr)
        {
            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, rayStatsBuffer));
            rayStatsSet = device->createBindingSet(setDesc, rayStatsLayout);
        }

        if (rayStatsBuffer == nullptr || rayStatsSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass ray-stats buffer");
            return false;
        }
    }

    // The pipeline: one raygen (the specialized RGenPrimary), the engine's two miss shaders as two
    // GENERAL groups, and the engine's two hit groups in the order the instance records address
    // (fully opaque first, alpha tested second). The twelve layouts are added in set order;
    // per-shader and per-hit-group binding layouts stay null - the Vulkan backend rejects them with
    // NotSupported (vulkan-raytracing.cpp:1531-1535, :1543-1547).
    {
        nvrhi::rt::PipelineDesc desc;

        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(RAYGEN_EXPORT_NAME)
                           .setShader(specializedRaygenShader));
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

        desc.addBindingLayout(tlasLayout);                                            // 0
        desc.addBindingLayout(framebufferLayout);                                     // 1
        desc.addBindingLayout(uniformLayout);                                         // 2
        desc.addBindingLayout(vertexDataLayout);                                      // 3
        desc.addBindingLayout(textureTable->GetLayout());                             // 4
        desc.addBindingLayout(holeLayout);                                            // 5
        desc.addBindingLayout(holeLayout);                                            // 6
        desc.addBindingLayout(cubemapLayout);                                         // 7
        desc.addBindingLayout(renderCubemapLayout);                                   // 8
        desc.addBindingLayout(holeLayout);                                            // 9
        desc.addBindingLayout(holeLayout);                                            // 10
        desc.addBindingLayout(rayStatsLayout);                                        // 11

        desc.setMaxPayloadSize(MAX_PAYLOAD_SIZE);
        desc.setMaxAttributeSize(MAX_ATTRIBUTE_SIZE);
        desc.setMaxRecursionDepth(MAX_RECURSION_DEPTH);

        pipeline = device->createRayTracingPipeline(desc);
    }

    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass ray-tracing pipeline");
        return false;
    }

    // The table: one raygen, two misses and two hit groups, at the engine's own SBT indices. The
    // returned indices are verified against the SBT_INDEX_* contract, so a shader-table shape
    // regression cannot pass silently. Uncached (the ShaderTableDesc default): with five entries
    // the per-list bake of the uncached path is cheaper than a cached buffer
    // (vulkan-raytracing.cpp:1710-1736). The per-entry binding-set argument must stay null on
    // Vulkan (:1771-1836).
    shaderTable = pipeline->createShaderTable(nvrhi::rt::ShaderTableDesc());
    if (shaderTable == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass shader table");
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
        LogMessage(print, "Warning: RHI: the primary RT pass shader table does not match the engine's SBT indices");
        shaderTable = nullptr;
        return false;
    }

    created = true;
    return true;
}

void RhiRtPrimaryPass::Render(nvrhi::ICommandList *pCommandList,
                             uint32_t frameIndex,
                             nvrhi::rt::IAccelStruct *pTopLevel,
                             nvrhi::IBuffer *pUniformBuffer,
                             const VertexData &vertexData,
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
            LogMessage(print, "Warning: RHI: the primary RT pass got no top-level acceleration structure, the trace is skipped");
        }
        return;
    }

    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the primary RT pass got no engine framebuffers, the trace is skipped");
        }
        return;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the primary RT pass got no global uniform, the trace is skipped");
        }
        return;
    }

    if (!vertexData.IsComplete())
    {
        if (!warnedMissingVertexData)
        {
            warnedMissingVertexData = true;
            LogMessage(print, "Warning: RHI: the primary RT pass got no vertex-data buffers, the trace is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // The 26 storage images of this slot. `GetImageHandles` resolves the engine's ping-pong swap
    // (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's current
    // image, which is the one the engine's own per-slot descriptor set binds to the variable's
    // fixed raw binding - the shader never sees the swap.
    uint64_t imageHandles[FRAMEBUFFER_UAV_COUNT] = {};
    uint64_t imageViews[FRAMEBUFFER_UAV_COUNT] = {};
    VkFormat imageFormats[FRAMEBUFFER_UAV_COUNT] = {};

    for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
    {
        const FramebufferImageIndex index = FRAMEBUFFER_UAV_IMAGES[i];

        if ((ShFramebuffers_Flags[index] & NOT_RENDER_SIZED_FLAGS) != 0)
        {
            if (!warnedUnexpectedSize)
            {
                warnedUnexpectedSize = true;
                LogMessage(print, std::string("Warning: RHI: the primary RT pass writes \"") +
                                      ShFramebuffers_DebugNames[index] +
                                      "\", which is not render-sized; the trace is skipped");
            }
            return;
        }

        const auto [image, view, format] = pFramebuffers->GetImageHandles(index, frameIndex);

        imageHandles[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));
        imageViews[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        imageFormats[i] = format;

        if (imageHandles[i] == 0)
        {
            if (!warnedMissingFramebuffers)
            {
                warnedMissingFramebuffers = true;
                LogMessage(print, "Warning: RHI: the primary RT pass got no framebuffer image, the trace is skipped");
            }
            return;
        }
    }

    bool framebuffersChanged = target.width != width || target.height != height;
    for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
    {
        framebuffersChanged = framebuffersChanged || target.imageHandles[i] != imageHandles[i];
    }

    if (framebuffersChanged)
    {
        // The engine re-created its framebuffers (a resize or an image-format change): the wraps
        // and the set over them are retired and rebuilt. Retiring first is safe even if the rebuild
        // below fails - the next frame retries, and the trace is skipped until it succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtPrimaryPass ") +
                                          ShFramebuffers_DebugNames[FRAMEBUFFER_UAV_IMAGES[i]] +
                                          " frame " + std::to_string(frameIndex);

            target.framebufferTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i], width, height, debugName);

            if (target.framebufferTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap a primary RT pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
    }

    // Set 1 over the wraps; rebuilt exactly when ReleaseFramebufferTarget nulled it.
    if (target.framebufferSet == nullptr)
    {
        nvrhi::BindingSetDesc setDesc;

        for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
        {
            setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(
                ShFramebuffers_Bindings[FRAMEBUFFER_UAV_IMAGES[i]], target.framebufferTextures[i]));
        }

        target.framebufferSet = device->createBindingSet(setDesc, framebufferLayout);

        if (target.framebufferSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the primary RT pass framebuffer binding set");
            return;
        }
    }

    // The per-slot sets 0, 2 and 3. Their keys are the pointers the host passes, so a re-created AS
    // or a re-created vertex-data buffer is picked up without a second Create.
    if (!PrepareTlasSet(target, pTopLevel) ||
        !PrepareUniformSet(target, pUniformBuffer) ||
        !PrepareVertexDataSet(target, vertexData))
    {
        return;
    }

    // The image state contract, spelled out on Render in the header: the engine leaves every
    // framebuffer image in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native
    // wrap keeps no state between command lists (RhiTextureSource.h), so every list announces that
    // state for all 26 images before the first use. The Texture_UAV bindings require the same
    // state, so the announcement turns the automatic barrier into a same-layout UAV barrier and the
    // images stay in the layout the engine knows.
    for (uint32_t i = 0; i < FRAMEBUFFER_UAV_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.framebufferTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    // The shared table's slots wrapped since the last frame need their first-use state declared in
    // the list that samples them (RhiTextureTable.h); the host's frame skeleton does this too, and
    // a repeated call is a no-op.
    textureTable->TrackPendingTextures(pCommandList);

    // The twelve sets, in the layout order the pipeline was built with; the pinned backend's legacy
    // binding mode binds the list positionally (vulkan-resource-bindings.cpp:940-958). Sets 5, 6, 9
    // and 10 are the same empty set: the shaders declare nothing there, but the automatic-barrier
    // pass dereferences every entry, so a real empty set has to fill each position.
    nvrhi::rt::State state;
    state.setShaderTable(shaderTable);
    state.addBindingSet(target.tlasSet);          // 0
    state.addBindingSet(target.framebufferSet);   // 1
    state.addBindingSet(target.uniformSet);       // 2
    state.addBindingSet(target.vertexDataSet);    // 3
    state.addBindingSet(textureTable->GetTable()); // 4
    state.addBindingSet(holeSet);                 // 5
    state.addBindingSet(holeSet);                 // 6
    state.addBindingSet(cubemapTable);            // 7
    state.addBindingSet(renderCubemapSet);        // 8
    state.addBindingSet(holeSet);                 // 9
    state.addBindingSet(holeSet);                 // 10
    state.addBindingSet(rayStatsSet);             // 11

    pCommandList->setRayTracingState(state);

    // One ray per pixel at the full render resolution. The raygen's DispatchRaysIndex is the
    // regular (non-checkerboard) pixel; the shader maps it to the checkerboard slot itself
    // (`getCheckerboardPix`, RaygenPrimary.hlsli:414-415), so the dispatch is never halved.
    nvrhi::rt::DispatchRaysArguments args;
    args.setDimensions(width, height, 1);
    pCommandList->dispatchRays(args);
}

void RhiRtPrimaryPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtPrimaryPass::ReleaseFramebufferTarget(Target &target)
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

void RhiRtPrimaryPass::ReleaseTarget(Target &target)
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
}

bool RhiRtPrimaryPass::PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel)
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

    target.tlasSet = device->createBindingSet(setDesc, tlasLayout);

    if (target.tlasSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass TLAS binding set");
        return false;
    }

    target.topLevel = pTopLevel;
    return true;
}

bool RhiRtPrimaryPass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
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
            LogMessage(print, "Warning: RHI: the primary RT pass needs the global uniform as a static constant-buffer wrap");
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
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

bool RhiRtPrimaryPass::PrepareVertexDataSet(Target &target, const VertexData &vertexData)
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
                LogMessage(print, "Warning: RHI: the primary RT pass vertex-data buffers need a non-zero structStride");
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

    target.vertexDataSet = device->createBindingSet(setDesc, vertexDataLayout);

    if (target.vertexDataSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the primary RT pass vertex-data binding set");
        return false;
    }

    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        target.vertexBuffers[i] = buffers[i];
    }

    return true;
}

bool RhiRtPrimaryPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the primary RT pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
