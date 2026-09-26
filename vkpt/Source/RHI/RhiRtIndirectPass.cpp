#include "RhiRtIndirectPass.h"

#include "RhiDescriptors.h"
#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiRtDirectPass.h"
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
// in (ShaderManager.cpp:41-49). The indirect raygen is `RtQ2Indirect.rgen.spv`; the other four are
// the same stage set the primary and direct passes load. Both misses are entered by this raygen -
// the bounce rays use `RtMiss` at index 0, the three shadow sites `RtMissShadowCheck` at index 1
// (measured over the blob: six OpTraceRayKHR, two with miss index 0 and four with miss index 1) - so
// the table's miss order is load bearing in both directions.
const char *const RAYGEN_SHADER_FILE_NAME = "RtQ2Indirect.rgen.spv";
const char *const MISS_SHADER_FILE_NAME = "RtMiss.rmiss.spv";
const char *const SHADOW_MISS_SHADER_FILE_NAME = "RtMissShadowCheck.rmiss.spv";
const char *const CLOSEST_HIT_SHADER_FILE_NAME = "RtClsOpaque.rchit.spv";
const char *const ANY_HIT_SHADER_FILE_NAME = "RtAlphaTest.rahit.spv";

// Distinct export names are mandatory on Vulkan at this pin: the backend keys the pipeline's shader
// groups by PipelineShaderDesc::exportName, falling back to the shader entry name
// (vulkan-raytracing.cpp:1593-1603), and every engine blob's entry point is "main"
// (RhiPipeline.cpp:23-24). The hit group's export name is asserted non-empty (:1641), and the two
// hit groups need distinct names although they share the closest-hit shader - the order of the
// addHitGroup calls is what makes offset 0 the fully-opaque group and offset 1 the alpha-tested one,
// which is the order the engine's instance records address
// (instanceShaderBindingTableRecordOffset 0/1, ASManager.cpp:976-991). The names carry this module's
// prefix so a message or a capture can tell the three pipelines' groups apart.
const char *const RAYGEN_EXPORT_NAME = "RhiRtIndirectRGen";
const char *const MISS_EXPORT_NAME = "RhiRtIndirectRMiss";
const char *const SHADOW_MISS_EXPORT_NAME = "RhiRtIndirectRMissShadow";
const char *const HIT_GROUP_OPAQUE_EXPORT_NAME = "RhiRtIndirectRClsOpaque";
const char *const HIT_GROUP_ALPHA_TESTED_EXPORT_NAME = "RhiRtIndirectRClsOpaqueAlphaTested";

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

// The 14 images `RtQ2Indirect.rgen` references after stream 1's fix: the five UAVs (its storage
// images) first, then the nine SRVs (its sampled images). The list is the direct pass's 12-item list
// plus the three indirect-SH writes and minus nothing; the entry-point interface and the descriptor
// decorations of the pre-fix blob named exactly these and the extra sampled specular view 139
// (a43_recon.md §1.1, measured with `spirv-dis`), which stream 1's storage-image read of image 15
// drops (a43_recon.md §4.2). DXC dead-strips the unused sampled resource, and the rebuilt blob in
// `vkpt/Build` was re-measured: 36 descriptor pairs, set 1 without 139, and an independent dxc
// compile of the fixed source is byte-identical to it. The order is not a requirement of NVRHI - a
// binding set's items are keyed by slot - it only groups the writes and the reads the way the raygen
// uses them.
//
// The engine binding numbers are the identity for the UAVs (`ShFramebuffers_Bindings[index]`) and
// `124 + index` for the sampled views (`ShFramebuffers_Sampled_Bindings[index]`), so the table
// carries the image index and the kind and the module derives both the layout item and the set item
// from the generated arrays; a hand-written binding number could drift from the shader build, the
// generated ones cannot.
struct FramebufferBinding
{
    FramebufferImageIndex image;
    bool isUAV;
};

constexpr uint32_t FRAMEBUFFER_BINDING_COUNT = 14;
constexpr FramebufferBinding FRAMEBUFFER_BINDINGS[FRAMEBUFFER_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,        true  }, //  15  framebufUnfilteredSpecular (r32ui)
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R,  true  }, //  16  framebufUnfilteredIndirectSH_R
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G,  true  }, //  17  framebufUnfilteredIndirectSH_G
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B,  true  }, //  18  framebufUnfilteredIndirectSH_B
    { FB_IMAGE_INDEX_VIEW_DIRECTION,             true  }, //  23  framebufViewDirection
    { FB_IMAGE_INDEX_ALBEDO,                     false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,                     false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,                     false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,            false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,         false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_SURFACE_POSITION,           false }, // 143  framebufSurfacePosition_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,           false }, // 239  framebufQ2GradSmplPos_Sampled
    { FB_IMAGE_INDEX_Q2_RNG_SEED,                false }, // 245  framebufQ2RngSeed_Sampled
    { FB_IMAGE_INDEX_Q2_CLUSTER,                 false }, // 247  framebufQ2Cluster_Sampled
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
// list is the primary pass's, because the layout handle is the primary's and the module only builds
// its own set over it.
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

// The specialization constant of `RtQ2Indirect.rgen`: [[vk::constant_id(0)]] const uint
// maxAlbedoLayerCount = 0 (SpecId 0, uint; the blob declares exactly that). The engine passes
// `indirectIlluminationMaxAlbedoLayers` (RayTracingPipeline.cpp:100), which the game sets to 1
// (Quake/gl_vidsdl.c:1459) and which has to reach the shader through createShaderSpecialization,
// because SpecId 0's SPIR-V default is 0 (a materially different path).
constexpr uint32_t MAX_ALBEDO_LAYERS_SPEC_ID = 0;
constexpr uint32_t INDIRECT_RAYS_MAX_ALBEDO_LAYERS = 1;

// The engine's pipeline-wide RT limits (RayTracingPipeline.cpp:203 and the measured blob structs):
// recursion depth 2; the payload is ShPayload = {float2 baryCoords; uint instIdAndIndex; uint
// geomAndPrimIndex} = 16 B and the hit attribute HitAttributes = {float2 inBaryCoords} = 8 B
// (Structs.hlsli:47-57). The indirect raygen's own interface names both payloads (measured: unlike
// the direct raygen, which names only the shadow payload), but the pipeline's union stays the same
// because it also carries the two misses and the two hit groups. NVRHI's two size fields are
// D3D12-only at this pin, so on Vulkan they are documentation; they are kept exact for a future
// backend switch.
constexpr uint32_t MAX_RECURSION_DEPTH = 2;
constexpr uint32_t MAX_PAYLOAD_SIZE = 2 * sizeof(float) + 2 * sizeof(uint32_t);
constexpr uint32_t MAX_ATTRIBUTE_SIZE = 2 * sizeof(float);

// The engine's cubemap capacity: CubemapManager.cpp:30 MAX_CUBEMAP_COUNT. The placeholder table of
// set 7 is filled up to this capacity, so any `skyCubemapIndex` the uniform can carry stays inside
// the descriptor array.
constexpr uint32_t CUBEMAP_TABLE_CAPACITY = 32;

// The engine render cubemap's level count and the shader's SKY_MIP_COUNT (RenderCubemap.cpp:34-37
// creates both cubes with cubemapMipLevels = 11; RaygenCommon.hlsli:390 defines SKY_MIP_COUNT
// 11.0): `getSkyFiltered` scales its lod by the mip count, so a shorter chain would clamp the
// ambient/roughness lookups to its last mip. Set 8's real cubes have to carry the full chain.
constexpr uint32_t RENDER_CUBEMAP_MIP_COUNT = 11;

// The shape the shader's `renderCubemap`/`renderCubemapEnv` `SampleLevel(..., lod)` reads and the
// engine render cubemap has (RenderCubemap.cpp:34-37, :489-503): a six-layer R16G16B16A16_SFLOAT
// cube of square faces with the eleven-mip chain above. A desc outside it is a host-side mistake,
// not a per-frame condition: SetRenderCubemaps refuses it with a one-shot warning and the
// placeholder pair stays.
bool IsRenderCubemapDesc(const nvrhi::TextureDesc &desc)
{
    return desc.dimension == nvrhi::TextureDimension::TextureCube &&
           desc.format == nvrhi::Format::RGBA16_FLOAT &&
           desc.arraySize == 6 &&
           desc.width > 0 && desc.width == desc.height &&
           desc.mipLevels >= RENDER_CUBEMAP_MIP_COUNT;
}

// The engine's set-8 sampler: RG_SAMPLER_FILTER_LINEAR with REPEAT on both axes
// (RenderCubemap.cpp:696-697; SamplerManager.cpp:142 fills W with REPEAT too). A different sampler
// changes the sky look silently, so SetRenderCubemaps reports a deviation once and still uses what
// it got.
bool IsRenderCubemapSampler(const nvrhi::SamplerDesc &desc)
{
    return desc.minFilter && desc.magFilter &&
           desc.addressU == nvrhi::SamplerAddressMode::Repeat &&
           desc.addressV == nvrhi::SamplerAddressMode::Repeat &&
           desc.addressW == nvrhi::SamplerAddressMode::Repeat;
}

// The blue-noise asset's shape (ShaderCommonC.h:120-122 and the class comment of the header): the
// shader indexes 128x128 texels across 128 layers, `layer = (texIndex + salt) %
// BLUE_NOISE_TEXTURE_COUNT` (Random.hlsli:257-266), so the wrap SetBlueNoiseTexture receives has to
// cover at least this array and extent.
constexpr uint32_t BLUE_NOISE_LAYER_COUNT = BLUE_NOISE_TEXTURE_COUNT;
constexpr uint32_t BLUE_NOISE_EXTENT = BLUE_NOISE_TEXTURE_SIZE;

// The half-res decision of PathTracer.cpp:203-210: the dispatch height is halved exactly when
// `giBounceRays[0]` is in this open interval. The raygen reads the same uniform member for its own
// row mapping and for the GI-off zero store, so the host and the shader cannot disagree.
constexpr float HALF_RES_MIN_BOUNCE_RAYS = 0.25f;
constexpr float HALF_RES_MAX_BOUNCE_RAYS = 0.75f;

void LogMessage(const RhiRtIndirectPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtIndirectPass::RhiRtIndirectPass() = default;

RhiRtIndirectPass::~RhiRtIndirectPass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images, the sets reference the primary's and the direct's
        // handles and the shader table references the pipeline; the host destroys the pass while it
        // can still idle the device (VulkanDevice does that before the skeleton as well), so nothing
        // has to go through a retire queue here.
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

    renderCubemapSet = nullptr;
    renderCubemapSetSampler = nullptr;
    renderCubemapSampler = nullptr;
    renderCubemapEnvSetTexture = nullptr;
    renderCubemapEnvTexture = nullptr;
    renderCubemapSetTexture = nullptr;
    renderCubemapTexture = nullptr;
    cubemapTable = nullptr;
    dummyCubemapSampler = nullptr;
    dummyCubemapTexture = nullptr;
    blueNoiseSet = nullptr;
    blueNoiseTexture = nullptr;
    blueNoiseSetTexture = nullptr;
    renderCubemapLayout = nullptr;
    cubemapLayout = nullptr;
    blueNoiseLayout = nullptr;
    framebufferLayout = nullptr;
    anyHitShader = nullptr;
    closestHitShader = nullptr;
    shadowMissShader = nullptr;
    missShader = nullptr;
    specializedRaygenShader = nullptr;
    raygenShader = nullptr;
}

bool RhiRtIndirectPass::Create(nvrhi::IDevice *pDevice,
                               rhi::RhiFrameContext *pFrameContext,
                               rhi::RhiTextureTable *pTextureTable,
                               const RhiRtPrimaryPass *pPrimaryPass,
                               const RhiRtDirectPass *pDirectPass,
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
    primaryPass = pPrimaryPass;
    directPass = pDirectPass;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs the frame context of the RHI layer");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated() ||
        textureTable->GetLayout() == nullptr || textureTable->GetTable() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs the shared RHI texture table (set 4)");
        return false;
    }

    // The shared layouts, the empty set of 9/10 and the ray-stats set of 11 come from the primary
    // pass: NVRHI accepts a state's set only when it was created over the very layout handle the
    // pipeline declared (validation-commandlist.cpp:509-520), so this module can share those
    // positions but not re-create them. The primary pass therefore has to be created first and has
    // to outlive this object.
    if (primaryPass == nullptr || !primaryPass->IsCreated() ||
        primaryPass->GetTlasLayout() == nullptr || primaryPass->GetUniformLayout() == nullptr ||
        primaryPass->GetVertexDataLayout() == nullptr || primaryPass->GetHoleLayout() == nullptr ||
        primaryPass->GetHoleSet() == nullptr || primaryPass->GetRayStatsLayout() == nullptr ||
        primaryPass->GetRayStatsSet() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs the created primary RT pass for its shared set layouts");
        return false;
    }

    // Set 6 is the direct pass's: the wraps, the four light copies and the statistics fill are that
    // module's, single-consumer work (a43_recon.md §6.2), and this pass only reads through the same
    // NVRHI objects. It needs that pass's layout to declare the position and its per-slot set at
    // Render time; a set is not required here, because the direct pass may not have rendered a slot
    // yet.
    if (directPass == nullptr || !directPass->IsCreated() || directPass->GetLightLayout() == nullptr)
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs the created direct RT pass for its shared light layout (set 6)");
        return false;
    }

    // The pipeline declares twelve binding layouts, so the pinned NVRHI's cap has to be the raised
    // one: the module is written against `c_MaxBindingLayouts == 16` (the patch build_win.ps1
    // applies for the duration of a build, third_party/nvrhi-max-binding-layouts.patch), exactly as
    // RhiRtPrimaryPass and RhiRtDirectPass are. 'if constexpr' keeps the check out of the compiled
    // path of a patched build.
    if constexpr (nvrhi::c_MaxBindingLayouts < PIPELINE_SET_COUNT)
    {
        LogMessage(print, "Warning: RHI: the indirect RT pass needs c_MaxBindingLayouts >= " +
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

    // The raygen is the one stage with a specialization constant (SpecId 0 = 1, see above). The
    // backend's createShaderSpecialization returns a derived shader that carries the constant into
    // the pipeline's VkSpecializationInfo (vulkan-raytracing.cpp:1591, vulkan-graphics.cpp:167-214)
    // and keeps the base module alive through its reference.
    {
        const nvrhi::ShaderSpecialization specialization =
            nvrhi::ShaderSpecialization::UInt32(MAX_ALBEDO_LAYERS_SPEC_ID, INDIRECT_RAYS_MAX_ALBEDO_LAYERS);

        specializedRaygenShader = device->createShaderSpecialization(raygenShader.Get(), &specialization, 1);

        if (specializedRaygenShader == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to specialize the indirect RT pass raygen (maxAlbedoLayerCount)");
            return false;
        }
    }

    // The four layouts this module owns. Every one is AllRayTracing visibility, for the reason
    // RhiDebugTracePass documents: the acceleration-structure-read barrier names the compute stage
    // too (vulkan-constants.cpp:282-285) and this device has no rayQuery feature to make that legal
    // (the A3.1 fix).
    {
        // Set 1: one item per image of the raygen, at the engine's raw binding of that image. A
        // partial layout in the same sense the direct pass's set 1 is - only the 14 bindings the
        // post-fix raygen declares have an item. Both kinds keep an offset of 0, so an item's slot
        // is exactly the raw binding the generated arrays carry (the UAV offset's default is 384,
        // nvrhi.h:2063-2066).
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
        // Set 5: the single blue-noise array SRV at raw binding 0, no sampler. The shader declares
        // `Texture2DArray<float4> blueNoiseTextures` and reads it with a plain `Load`, so the only
        // descriptor is the image; the wrap SetBlueNoiseTexture takes is a Texture2DArray of 128
        // layers.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_BLUE_NOISE));

        blueNoiseLayout = device->createBindingLayout(desc);
    }
    {
        // Set 7: `globalCubemaps[]` is a runtime descriptor array of cubes (measured in the blob),
        // so the set has to be a bindless table - a regular single-item layout cannot express the
        // engine's array. The primary's cubemap layout and table are private and have no accessors,
        // so this module owns a second bindless layout, filled with the same 1x1 placeholder cube
        // (a43_recon.md §8.9). The register-space order makes the textures binding 0 and the
        // samplers binding 1, the numbers the shader declares.
        nvrhi::BindlessLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.maxCapacity = CUBEMAP_TABLE_CAPACITY;
        desc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::Immutable;
        desc.addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_CUBEMAPS));
        desc.addRegisterSpace(nvrhi::BindingLayoutItem::Sampler(BINDING_CUBEMAPS_SAMPLER));

        cubemapLayout = device->createBindlessLayout(desc);
    }
    {
        // Set 8: the four items the blob declares - `renderCubemap` at raw 0, `renderCubemapEnv` at
        // raw 1 and their samplers at raw 2/3. The primary's raygen declares only 0/2, which is why
        // this module owns its own layout instead of reusing the primary's. A sampler offset of 0
        // keeps the slot as the raw binding.
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setSamplerOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_RENDER_CUBEMAP));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_RENDER_CUBEMAP_ENV));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(BINDING_RENDER_CUBEMAP_SAMPLER));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(BINDING_RENDER_CUBEMAP_ENV_SAMPLER));

        renderCubemapLayout = device->createBindingLayout(desc);
    }

    if (framebufferLayout == nullptr || blueNoiseLayout == nullptr || cubemapLayout == nullptr ||
        renderCubemapLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create an indirect RT pass binding layout");
        return false;
    }

    // The 1x1 placeholder cube of sets 7 and 8. The engine's cubemaps have no RHI accessor yet
    // (CubemapManager.h, RenderCubemap.h) and their content is produced by the legacy sky path, so
    // this pass binds a dummy like the primary's: the sky modes A4.3 supports (COLOR / the raster
    // sky) never sample it. Unlike the primary's placeholder this one is cleared to black once, on
    // the first list that binds it, so the PROCEDURAL path the default `rt_physical_sky 1` selects
    // reads defined black radiance instead of undefined contents (a43_recon.md §8.3). Set 8 gets
    // the procedural-sky module's real cubes through SetRenderCubemaps, and the dummy stays the
    // fallback and the set-7 slot (A5.4).
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
        desc.debugName = "RhiRtIndirectPass dummy cubemap";

        dummyCubemapTexture = rhi::createTexture(device, desc, desc.debugName);
        dummyCubemapSampler = rhi::createEngineTextureSampler(device, "RhiRtIndirectPass dummy cubemap sampler");

        if (dummyCubemapTexture == nullptr || dummyCubemapSampler == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the indirect RT pass placeholder cubemap");
            return false;
        }

        cubemapTable = rhi::createTextureTable(device, cubemapLayout, "RhiRtIndirectPass cubemaps (placeholder)");
        if (cubemapTable == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the indirect RT pass cubemap table");
            return false;
        }

        for (uint32_t slot = 0; slot < CUBEMAP_TABLE_CAPACITY; slot++)
        {
            if (!rhi::setTexture(device, cubemapTable, slot, dummyCubemapTexture) ||
                !rhi::setSampler(device, cubemapTable, slot, dummyCubemapSampler))
            {
                LogMessage(print, "Warning: RHI: failed to fill the indirect RT pass cubemap table");
                return false;
            }
        }

        // One texture backs both cube bindings and one sampler both sampler bindings: the blob
        // samples `renderCubemap` on the RASTERIZED path and `renderCubemapEnv` on the PROCEDURAL
        // one (RaygenCommon.hlsli:360-427) and never both in one dispatch.
        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_RENDER_CUBEMAP, dummyCubemapTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_RENDER_CUBEMAP_ENV, dummyCubemapTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_RENDER_CUBEMAP_SAMPLER, dummyCubemapSampler));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_RENDER_CUBEMAP_ENV_SAMPLER, dummyCubemapSampler));

        renderCubemapSet = device->createBindingSet(setDesc, renderCubemapLayout);
        if (renderCubemapSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the indirect RT pass render-cubemap set");
            return false;
        }
    }

    // The pipeline: one raygen (the specialized RGenQ2Indirect), the engine's two miss shaders as
    // two GENERAL groups, and the engine's two hit groups in the order the instance records address
    // (fully opaque first, alpha tested second). The twelve layouts are added in set order; the
    // shared positions are the primary's and the direct's own handles, so a set of theirs makes it
    // through validation-commandlist.cpp:509-520. Per-shader and per-hit-group binding layouts stay
    // null - the Vulkan backend rejects them with NotSupported (vulkan-raytracing.cpp:1531-1535,
    // :1543-1547).
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

        desc.addBindingLayout(primaryPass->GetTlasLayout());           // 0
        desc.addBindingLayout(framebufferLayout);                      // 1
        desc.addBindingLayout(primaryPass->GetUniformLayout());        // 2
        desc.addBindingLayout(primaryPass->GetVertexDataLayout());     // 3
        desc.addBindingLayout(textureTable->GetLayout());              // 4
        desc.addBindingLayout(blueNoiseLayout);                        // 5
        desc.addBindingLayout(directPass->GetLightLayout());           // 6
        desc.addBindingLayout(cubemapLayout);                          // 7
        desc.addBindingLayout(renderCubemapLayout);                    // 8
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
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass ray-tracing pipeline");
        return false;
    }

    // The table: one raygen, two misses and two hit groups, at the engine's own SBT indices. The
    // returned indices are verified against the SBT_INDEX_* contract, so a shader-table shape
    // regression cannot pass silently - the bounce rays address miss 0 and the shadow rays miss 1,
    // so both miss positions are load bearing. Uncached (the ShaderTableDesc default): with five
    // entries the per-list bake of the uncached path is cheaper than a cached buffer
    // (vulkan-raytracing.cpp:1710-1736). The per-entry binding-set argument must stay null on
    // Vulkan (:1771-1836). NVRHI has one raygen per table, so this module cannot share the direct
    // pass's table; the engine's SBT_INDEX_RAYGEN_Q2_INDIRECT = 4 has no counterpart here.
    shaderTable = pipeline->createShaderTable(nvrhi::rt::ShaderTableDesc());
    if (shaderTable == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass shader table");
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
        LogMessage(print, "Warning: RHI: the indirect RT pass shader table does not match the engine's SBT indices");
        shaderTable = nullptr;
        return false;
    }

    created = true;
    return true;
}

void RhiRtIndirectPass::SetBlueNoiseTexture(nvrhi::ITexture *pTexture)
{
    if (pTexture == nullptr)
    {
        blueNoiseTexture = nullptr;
        return;
    }

    const nvrhi::TextureDesc &desc = pTexture->getDesc();

    // The shape the shader's `blueNoiseTextures.Load(int4(x, y, layer, 0))` needs (Random.hlsli:
    // 257-266): the layer index wraps modulo BLUE_NOISE_TEXTURE_COUNT and the texel offsets are
    // below 128, so a smaller array or extent would read outside the descriptor or the mip. A desc
    // outside the shape is a host-side mistake, not a per-frame condition: it is refused with a
    // one-shot warning and the pass keeps skipping until a valid wrap arrives.
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
            LogMessage(print, "Warning: RHI: the indirect RT pass blue-noise texture (set 5) has to be a "
                              "Texture2DArray of at least 128x128 RGBA8_UNORM with at least 128 layers");
        }
        blueNoiseTexture = nullptr;
        return;
    }

    blueNoiseTexture = pTexture;
}

void RhiRtIndirectPass::SetRenderCubemaps(nvrhi::ITexture *pCubemap, nvrhi::ITexture *pEnvCubemap,
                                          nvrhi::ISampler *pSampler)
{
    if (pCubemap == nullptr || pEnvCubemap == nullptr || pSampler == nullptr)
    {
        // The placeholder fallback: the module keeps its black-cleared 1x1 dummy pair in set 8
        // until a valid trio arrives (or again after a null call).
        renderCubemapTexture = nullptr;
        renderCubemapEnvTexture = nullptr;
        renderCubemapSampler = nullptr;
        return;
    }

    // The shape of the procedural-sky module's cubes, which is the engine render cubemap's: see
    // IsRenderCubemapDesc. A desc outside it - or a partial pair, which the four-item set cannot
    // express - is a host-side mistake, not a per-frame condition: it is refused with a one-shot
    // warning and the placeholder pair stays in place.
    if (!IsRenderCubemapDesc(pCubemap->getDesc()) || !IsRenderCubemapDesc(pEnvCubemap->getDesc()))
    {
        if (!warnedBadRenderCubemap)
        {
            warnedBadRenderCubemap = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass render cubemaps (set 8) have to be "
                              "six-layer RGBA16F TextureCubes with square faces and at least eleven mips");
        }
        renderCubemapTexture = nullptr;
        renderCubemapEnvTexture = nullptr;
        renderCubemapSampler = nullptr;
        return;
    }

    if (!IsRenderCubemapSampler(pSampler->getDesc()) && !warnedRenderCubemapSampler)
    {
        warnedRenderCubemapSampler = true;
        LogMessage(print, "Warning: RHI: the indirect RT pass render-cubemap sampler is not the engine's "
                          "linear/repeat one; the sky lookups may shift");
    }

    renderCubemapTexture = pCubemap;
    renderCubemapEnvTexture = pEnvCubemap;
    renderCubemapSampler = pSampler;
}

void RhiRtIndirectPass::Render(nvrhi::ICommandList *pCommandList,
                               uint32_t frameIndex,
                               float numBounceRays,
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

    // Set 5 first: a missing blue-noise texture or a failed set creation skips the frame before any
    // wrap work. The announcement of the texture's first-use state happens inside, on this list.
    if (!PrepareBlueNoiseSet(pCommandList))
    {
        return;
    }

    if (pTopLevel == nullptr)
    {
        if (!warnedMissingTopLevel)
        {
            warnedMissingTopLevel = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no top-level acceleration structure, the trace is skipped");
        }
        return;
    }

    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no engine framebuffers, the trace is skipped");
        }
        return;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no global uniform, the trace is skipped");
        }
        return;
    }

    if (!vertexData.IsComplete())
    {
        if (!warnedMissingVertexData)
        {
            warnedMissingVertexData = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no vertex-data buffers, the trace is skipped");
        }
        return;
    }

    // Set 6: the direct pass's per-slot set, fetched each frame instead of cached. That module
    // wraps the five device-local light buffers, records this frame's copies and consumes the copy
    // flags before its own dispatch; its set is the only consumer of those copies, so this pass
    // binds it as it is (a43_recon.md §6.2). A null set means the direct pass did not render this
    // slot (it was not created, failed, or its light buffers are missing), and then this pass has no
    // light data either. The handle keeps the set alive for the duration of the recording.
    const nvrhi::BindingSetHandle lightSet = directPass->GetLightSet(frameIndex);

    if (lightSet == nullptr)
    {
        if (!warnedMissingLightSet)
        {
            warnedMissingLightSet = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no direct-pass light set (set 6), the trace is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // The 14 images of this slot. `GetImageHandles` resolves the engine's ping-pong swap
    // (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's current
    // image, which is the one the engine's own per-slot descriptor set binds to the variable's fixed
    // raw binding - the shader never sees the swap. The 4-tuple overload is what supplies the one
    // image that is not render-sized: Q2_GRAD_SMPL_POS carries FORCE_SIZE_1_3
    // (ShaderCommonCFramebuf.cpp) and its extent is ((w+1)/3, (h+1)/3) (Framebuffers.cpp:654-691);
    // the render-sized images answer with exactly the width/height passed in. None of these images
    // is upscaled-sized, so the render size stands in for the upscaled fields of the resolution
    // state - a future set-1 image with that flag would be a wrong wrap.
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
                LogMessage(print, "Warning: RHI: the indirect RT pass got no framebuffer image, the trace is skipped");
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
        // The engine re-created its framebuffers (a resize or an image-format change): the wraps and
        // the set over them are retired and rebuilt. Retiring first is safe even if the rebuild
        // below fails - the next frame retries, and the trace is skipped until it succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtIndirectPass ") +
                                          ShFramebuffers_DebugNames[FRAMEBUFFER_BINDINGS[i].image] +
                                          " frame " + std::to_string(frameIndex);

            target.framebufferTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i],
                imageExtents[i].width, imageExtents[i].height, debugName);

            if (target.framebufferTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap an indirect RT pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
    }

    // Set 1 over the wraps; rebuilt exactly when ReleaseFramebufferTarget nulled it. The item slots
    // are the raw bindings, because the layout's offsets are both 0. Image 15 is bound once, as its
    // storage view - the pre-fix blob's second descriptor for the same image (the sampled 139) is
    // gone after stream 1's fix, which is what lets the layout hold one item per image.
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
            LogMessage(print, "Warning: RHI: failed to create the indirect RT pass framebuffer binding set");
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

    // The image state contract, spelled out on Render in the header: the engine leaves every
    // framebuffer image in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap
    // keeps no state between command lists (RhiTextureSource.h), so every list announces that state
    // for all 14 images before the first use. The direct pass wrote 15 and 23 through its own wraps
    // on this list; the physical layout it left them in is exactly this GENERAL state, so the
    // announcement is the truth and the automatic transitions of this pass's own bindings (the
    // 9 SRV-only images move to the read-only layout) start from it.
    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.framebufferTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }

    // The shared table's slots wrapped since the last frame need their first-use state declared in
    // the list that samples them (RhiTextureTable.h); the primary and direct passes do this too, and
    // a repeated call is a no-op.
    textureTable->TrackPendingTextures(pCommandList);

    // Set 8 over the coordinator's render-cubemap pair, or over the placeholder pair until
    // SetRenderCubemaps delivered the real cubes; a changed key rebuilds the set through the retire
    // queue. It runs before the clear below, which reads the current set's keys.
    if (!PrepareRenderCubemapSet())
    {
        return;
    }

    // The set-8 placeholder is cleared to black once, on the first list that binds the placeholder:
    // the PROCEDURAL sky path - which `rt_physical_sky 1`, the default, selects - samples
    // `renderCubemapEnv` (RaygenCommon.hlsli:397-407) and would otherwise read the undefined
    // contents a never-cleared dummy has (a43_recon.md §8.3). The real cubes need no clear, so the
    // recorded clear follows the placeholder set: while the placeholder is current and still
    // uncleared, this list clears it. NVRHI's clear is a vkCmdClearColorImage behind the automatic
    // barrier pair (CopyDest in, the SRV requirement of the binding out), and the texture is
    // NVRHI-created, so it carries TRANSFER_DST; this is one 1x1x6 write on one list of the pass's
    // lifetime.
    if (!dummyCubemapCleared && renderCubemapSetTexture == nullptr)
    {
        pCommandList->clearTextureFloat(dummyCubemapTexture, nvrhi::AllSubresources,
                                        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
        dummyCubemapCleared = true;
    }

    // The twelve sets, in the layout order the pipeline was built with; the pinned backend's legacy
    // binding mode binds the list positionally (vulkan-resource-bindings.cpp:940-958). Sets 9 and 10
    // are the primary's empty set: the blob declares nothing there, but the automatic-barrier pass
    // dereferences every entry, so a real empty set has to fill each position. Set 6 is the direct
    // pass's per-slot set, the same object its own dispatch used.
    nvrhi::rt::State state;
    state.setShaderTable(shaderTable);
    state.addBindingSet(target.tlasSet);                  // 0
    state.addBindingSet(target.framebufferSet);           // 1
    state.addBindingSet(target.uniformSet);               // 2
    state.addBindingSet(target.vertexDataSet);            // 3
    state.addBindingSet(textureTable->GetTable());        // 4
    state.addBindingSet(blueNoiseSet);                    // 5
    state.addBindingSet(lightSet);                        // 6
    state.addBindingSet(cubemapTable);                    // 7
    state.addBindingSet(renderCubemapSet);                // 8
    state.addBindingSet(primaryPass->GetHoleSet());       // 9
    state.addBindingSet(primaryPass->GetHoleSet());       // 10
    state.addBindingSet(primaryPass->GetRayStatsSet());   // 11

    pCommandList->setRayTracingState(state);

    // The dispatch: the full render resolution normally, the half-height one when the uniform's
    // `giBounceRays[0]` selects the half mode - the exact condition of PathTracer.cpp:203-210, and
    // the same uniform member the raygen reads for its row mapping. The raygen's DispatchRaysIndex
    // is the checkerboard-packed texel itself, so the images stay at the render size and no dispatch
    // index is unused. The width is never halved.
    const bool halfRes = numBounceRays > HALF_RES_MIN_BOUNCE_RAYS &&
                         numBounceRays < HALF_RES_MAX_BOUNCE_RAYS;

    nvrhi::rt::DispatchRaysArguments args;
    args.setDimensions(width, halfRes ? (height + 1) / 2 : height, 1);
    pCommandList->dispatchRays(args);

    // The sampled images end the list in the read-only layout (vulkan-resource-bindings.cpp:
    // 398-435), but the engine's framebuffer images rest in GENERAL, and the next frame's primary,
    // direct and indirect passes write 8 of these 9 as storage images. Move every image that is
    // bound as an SRV and by no UAV back to UnorderedAccess; the five UAV images (15-18, 23) are
    // already there. The compose chain announces and restores 16-18 on its own list.
    for (uint32_t i = 0; i < FRAMEBUFFER_BINDING_COUNT; i++)
    {
        if (!FRAMEBUFFER_BINDINGS[i].isUAV)
        {
            pCommandList->setTextureState(
                target.framebufferTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
        }
    }
}

void RhiRtIndirectPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtIndirectPass::ReleaseFramebufferTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the wraps reference engine images the GPU may still be reading or writing through the
    // bindings, and the set references those wraps. The queue takes its reference now, so the
    // handles below can be cleared immediately.
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

void RhiRtIndirectPass::ReleaseTarget(Target &target)
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

bool RhiRtIndirectPass::PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel)
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
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass TLAS binding set");
        return false;
    }

    target.topLevel = pTopLevel;
    return true;
}

bool RhiRtIndirectPass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformSet != nullptr && target.uniformBuffer == pUniformBuffer)
    {
        return true;
    }

    // The shader declares set 2 as ConstantBuffer<ShGlobalUniform>; the validation device refuses
    // such a binding on a buffer whose desc lacks isConstantBuffer (validation-device.cpp:1717-1723)
    // and on a volatile one (:1725-1730, which would also become a dynamic-offset binding the static
    // layout item cannot take).
    if (!pUniformBuffer->getDesc().isConstantBuffer || pUniformBuffer->getDesc().isVolatile)
    {
        if (!warnedBadUniform)
        {
            warnedBadUniform = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass needs the global uniform as a static constant-buffer wrap");
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
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

bool RhiRtIndirectPass::PrepareVertexDataSet(Target &target, const RhiRtPrimaryPass::VertexData &vertexData)
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
                LogMessage(print, "Warning: RHI: the indirect RT pass vertex-data buffers need a non-zero structStride");
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
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass vertex-data binding set");
        return false;
    }

    for (uint32_t i = 0; i < VERTEX_DATA_BUFFER_COUNT; i++)
    {
        target.vertexBuffers[i] = buffers[i];
    }

    return true;
}

bool RhiRtIndirectPass::PrepareBlueNoiseSet(nvrhi::ICommandList *pCommandList)
{
    if (blueNoiseTexture == nullptr)
    {
        if (!warnedMissingBlueNoise)
        {
            warnedMissingBlueNoise = true;
            LogMessage(print, "Warning: RHI: the indirect RT pass got no blue-noise texture (set 5), the trace is skipped");
        }
        return false;
    }

    // The texture is static, so the one-item set is built once per wrap; only a coordinator that
    // replaced the wrap makes the pointer differ. The replaced set goes through the retire queue, so
    // a list that is still executing keeps its descriptors alive.
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
            LogMessage(print, "Warning: RHI: failed to create the indirect RT pass blue-noise binding set");
            return false;
        }

        blueNoiseSetTexture = blueNoiseTexture.Get();
    }

    // The first-use state contract of a foreign wrap (RhiTextureSource.h): the tracker starts the
    // texture at Common on the first list that touches it, and a Common -> NonPixelShaderResource
    // transition lowers to Undefined -> ShaderReadOnlyOptimal, which may discard the image. The
    // announcement names the true state, which is also what the Texture_SRV item of this
    // AllRayTracing-visibility layout requires (state-tracking.cpp:455-464 maps it to
    // NonPixelShaderResource because no pixel stage is visible): the binding therefore issues no
    // transition at all, on this or any later list, and the image stays in the layout the engine's
    // upload left it in (SHADER_READ_ONLY_OPTIMAL, BlueNoise.cpp:124-165).
    pCommandList->beginTrackingTextureState(blueNoiseTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::NonPixelShaderResource);

    return true;
}

bool RhiRtIndirectPass::PrepareRenderCubemapSet()
{
    // The set follows the coordinator's keys: the two real cubes with their sampler, or the
    // module's placeholder pair while the handles are null. A pointer change (SetRenderCubemaps, or
    // a null call restoring the placeholder) rebuilds it; the module starts with the null keys,
    // which is exactly the placeholder state of the set Create built, so the first Render is a
    // no-op until the setter changes something. The replaced set goes through the retire queue, so
    // a list that is still executing keeps its descriptors alive.
    const bool usePlaceholder = renderCubemapTexture == nullptr;
    nvrhi::ITexture *const texture =
        usePlaceholder ? dummyCubemapTexture.Get() : renderCubemapTexture.Get();
    nvrhi::ITexture *const envTexture =
        usePlaceholder ? dummyCubemapTexture.Get() : renderCubemapEnvTexture.Get();
    nvrhi::ISampler *const sampler =
        usePlaceholder ? dummyCubemapSampler.Get() : renderCubemapSampler.Get();

    if (renderCubemapSet != nullptr &&
        renderCubemapSetTexture == renderCubemapTexture.Get() &&
        renderCubemapEnvSetTexture == renderCubemapEnvTexture.Get() &&
        renderCubemapSetSampler == renderCubemapSampler.Get())
    {
        return true;
    }

    if (renderCubemapSet != nullptr)
    {
        frameContext->Retire(renderCubemapSet);
    }
    renderCubemapSet = nullptr;

    // One sampler fills both sampler items, exactly as the placeholder and the engine's set do
    // (both engine SAMPLER bindings name the same LINEAR/REPEAT sampler, RenderCubemap.cpp:696-697).
    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_RENDER_CUBEMAP, texture));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_RENDER_CUBEMAP_ENV, envTexture));
    setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_RENDER_CUBEMAP_SAMPLER, sampler));
    setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_RENDER_CUBEMAP_ENV_SAMPLER, sampler));

    renderCubemapSet = device->createBindingSet(setDesc, renderCubemapLayout);

    if (renderCubemapSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the indirect RT pass render-cubemap binding set");
        return false;
    }

    renderCubemapSetTexture = renderCubemapTexture.Get();
    renderCubemapEnvSetTexture = renderCubemapEnvTexture.Get();
    renderCubemapSetSampler = renderCubemapSampler.Get();
    return true;
}

bool RhiRtIndirectPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the indirect RT pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
