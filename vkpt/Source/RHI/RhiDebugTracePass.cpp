#include "RhiDebugTracePass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiTextureSource.h"

#include <string>

using namespace vkpt;

namespace
{

// The debug trio's blobs, by the file names the shader build writes into the engine's Build folder
// (the folder the host passes in as the shader folder path). The legacy ShaderManager has no
// entries for them: the trio is the RHI path's own proof pass, not a port of a legacy one.
const char *const RAYGEN_SHADER_FILE_NAME = "RhiDebugTrace.rgen.spv";
const char *const MISS_SHADER_FILE_NAME = "RhiDebugTrace.rmiss.spv";
const char *const CLOSEST_HIT_SHADER_FILE_NAME = "RhiDebugTrace.rchit.spv";

// Distinct export names are mandatory on Vulkan at this pin. The backend stores the pipeline's
// shader groups in a map keyed by PipelineShaderDesc::exportName, falling back to the shader's
// entry name when the export name is empty (vulkan-raytracing.cpp:1593-1603), and every engine
// blob's entry point is "main" (RhiPipeline.cpp:23-24) - so an empty export name would make the
// raygen and the miss share one map entry and the shader table would address the wrong group. The
// hit group's export name is asserted non-empty (:1641).
const char *const RAYGEN_EXPORT_NAME = "RhiDebugTraceRGen";
const char *const MISS_EXPORT_NAME = "RhiDebugTraceRMiss";
const char *const CLOSEST_HIT_EXPORT_NAME = "RhiDebugTraceRClosestHit";
// The second hit group exists because the engine's instance records address hit group 1 for
// alpha-tested geometry (SBT_INDEX_HITGROUP_ALPHA_TESTED, a31 reconnaissance §5). The table entry order
// is what makes offset 0 the fully-opaque group and offset 1 this one; both run the same closest-hit
// shader, so the debug appearance does not depend on the offset.
const char *const CLOSEST_HIT_ALPHA_TESTED_EXPORT_NAME = "RhiDebugTraceRClosestHitAlphaTested";

// The slot of each set's only item. For the TLAS the default shaderResource offset (0,
// nvrhi.h:2063) already puts the item at the raw binding 0 the shaders declare; for the uniform and
// the UAV the offsets below bring their raw bindings down to 0, so offset + slot equals the frozen
// binding 0 of every set (the arithmetic is documented in RhiPipeline.h).
constexpr uint32_t TLAS_SLOT = 0;
constexpr uint32_t UNIFORM_SLOT = 0;
constexpr uint32_t ALBEDO_SLOT = 0;

// Set 1's uniform sits at raw binding 0 in the engine's shaders, not at the DXC default 256.
constexpr uint32_t UNIFORM_BINDING_OFFSET = 0;
// Set 2's storage image sits at raw binding 0, not at the DXC default 384.
constexpr uint32_t ALBEDO_BINDING_OFFSET = 0;

// The debug trio's recursion depth: the raygen traces and the hit shader reports a hit; nothing
// traces inside a hit or miss shader, so 1 is the minimum Vulkan accepts. It is the only field of
// these three the Vulkan backend consumes - it passes it to setMaxPipelineRayRecursionDepth
// (vulkan-raytracing.cpp:1671).
constexpr uint32_t MAX_RECURSION_DEPTH = 1;

// The trio's payload and hit attribute, measured in the built blobs (spirv-dis, 2026-09-24):
// RhiDebugTrace.rgen.spv and .rmiss.spv declare %RhiDebugPayload = OpTypeStruct of one v3float
// (12 bytes), RhiDebugTrace.rchit.spv declares the same payload plus %RhiDebugHitAttributes =
// OpTypeStruct of one v2float (8 bytes) - exactly the structs the three sources repeat.
//
// The Vulkan backend never reads these two fields: grep over src/vulkan finds maxRecursionDepth as
// the only field of the trio it consumes, and the engine builds Vulkan only (NVRHI_WITH_DX12=OFF in
// the build cache). The D3D12 backend reads them into the DXIL subobject
// (d3d12-raytracing.cpp:1355-1356), so they are kept exact for a future backend switch. 8 bytes is
// also NVRHI's own default for maxAttributeSize (nvrhi.h:2934).
constexpr uint32_t MAX_PAYLOAD_SIZE = static_cast<uint32_t>(3 * sizeof(float));
constexpr uint32_t MAX_ATTRIBUTE_SIZE = static_cast<uint32_t>(2 * sizeof(float));

void LogMessage(const RhiDebugTracePass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiDebugTracePass::RhiDebugTracePass() = default;

RhiDebugTracePass::~RhiDebugTracePass()
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
        target.albedoSet = nullptr;
        target.tlasSet = nullptr;
        target.albedoTexture = nullptr;
        target.topLevel = nullptr;
        target.albedoImage = 0;
        target.width = 0;
        target.height = 0;
    }

    uniformSet = nullptr;
    albedoLayout = nullptr;
    uniformLayout = nullptr;
    tlasLayout = nullptr;
    closestHitShader = nullptr;
    missShader = nullptr;
    raygenShader = nullptr;
}

bool RhiDebugTracePass::Create(nvrhi::IDevice *pDevice,
                               rhi::RhiFrameContext *pFrameContext,
                               nvrhi::IBuffer *pUniformBuffer,
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
        LogMessage(print, "Warning: RHI: the debug trace pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the debug trace pass needs the frame context of the RHI layer");
        return false;
    }

    if (pUniformBuffer == nullptr)
    {
        LogMessage(print, "Warning: RHI: the debug trace pass needs the wrapped global uniform buffer (set 1)");
        return false;
    }

    if (pUniformBuffer->getDesc().isVolatile)
    {
        // BindingSetItem::ConstantBuffer turns a volatile buffer into a dynamic-offset binding
        // (nvrhi.h:2311-2318), which the static ConstantBuffer layout item would reject.
        LogMessage(print, "Warning: RHI: the debug trace pass uniform buffer must be a static wrap, not a volatile buffer");
        return false;
    }

    if (!LoadShader(RAYGEN_SHADER_FILE_NAME, nvrhi::ShaderType::RayGeneration, raygenShader) ||
        !LoadShader(MISS_SHADER_FILE_NAME, nvrhi::ShaderType::Miss, missShader) ||
        !LoadShader(CLOSEST_HIT_SHADER_FILE_NAME, nvrhi::ShaderType::ClosestHit, closestHitShader))
    {
        return false;
    }

    // The three sets of the frozen mapping, in order. The shader resource kind (the TLAS) keeps the
    // default offset 0; the other two replace their defaults with 0 as explained above.
    //
    // Visibility is AllRayTracing, not the layout helper's ShaderType::All: this device runs without
    // the rayQuery feature (the capability log says "ray query no"), and
    // VUID-VkBufferMemoryBarrier2-srcAccessMask-06256 then forbids the acceleration-structure-read
    // access at any shader stage except the ray-tracing ones. NVRHI derives the barrier's stage mask
    // from the layout's visibility, so an All layout produced that VUID on the first traced frame.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(TLAS_SLOT));

        tlasLayout = device->createBindingLayout(desc);
    }
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(UNIFORM_BINDING_OFFSET));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(UNIFORM_SLOT));

        uniformLayout = device->createBindingLayout(desc);
    }
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::AllRayTracing;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setUnorderedAccessViewOffset(ALBEDO_BINDING_OFFSET));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(ALBEDO_SLOT));

        albedoLayout = device->createBindingLayout(desc);
    }

    if (tlasLayout == nullptr || uniformLayout == nullptr || albedoLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a debug trace pass binding layout");
        return false;
    }

    // Set 1 is one static set for the whole run: the engine creates its global uniform once and
    // only its contents change, so the set never has to be rebuilt and no Retire contract applies
    // to it (unlike the per-frame wraps below).
    {
        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(UNIFORM_SLOT, pUniformBuffer));

        uniformSet = device->createBindingSet(setDesc, uniformLayout);
    }

    if (uniformSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the debug trace pass uniform binding set");
        return false;
    }

    // The pipeline: the raygen and the miss are two GENERAL groups (the backend makes one eGeneral
    // group per PipelineShaderDesc, vulkan-raytracing.cpp:1577-1604), the closest-hit shader is one
    // TRIANGLES hit group (:1610-1645). The three layouts are added in set order; in the pinned
    // backend's legacy binding mode that order is the descriptor set number
    // (vulkan-resource-bindings.cpp:1090-1099). Per-shader and per-hit-group binding layouts stay
    // null - the Vulkan backend rejects them with NotSupported (:1531-1535, :1543-1547).
    {
        nvrhi::rt::PipelineDesc desc;

        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(RAYGEN_EXPORT_NAME)
                           .setShader(raygenShader));
        desc.addShader(nvrhi::rt::PipelineShaderDesc()
                           .setExportName(MISS_EXPORT_NAME)
                           .setShader(missShader));
        desc.addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
                             .setExportName(CLOSEST_HIT_EXPORT_NAME)
                             .setClosestHitShader(closestHitShader));
        desc.addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
                             .setExportName(CLOSEST_HIT_ALPHA_TESTED_EXPORT_NAME)
                             .setClosestHitShader(closestHitShader));

        desc.addBindingLayout(tlasLayout);
        desc.addBindingLayout(uniformLayout);
        desc.addBindingLayout(albedoLayout);

        desc.setMaxPayloadSize(MAX_PAYLOAD_SIZE);
        desc.setMaxAttributeSize(MAX_ATTRIBUTE_SIZE);
        desc.setMaxRecursionDepth(MAX_RECURSION_DEPTH);

        pipeline = device->createRayTracingPipeline(desc);
    }

    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the debug trace pass ray-tracing pipeline");
        return false;
    }

    // The table: one raygen, one miss and two hit groups (offset 0 fully opaque, offset 1 alpha
    // tested - the order of the addHitGroup calls is the SBT index), uncached (the ShaderTableDesc
    // default) - with four entries the per-list bake of the uncached path is cheaper than a cached
    // buffer (vulkan-raytracing.cpp:1710-1736; the bake writes entry i at i * shaderGroupBaseAlignment,
    // :1247-1326). The per-entry binding-set argument must stay null on Vulkan (:1771-1836).
    shaderTable = pipeline->createShaderTable(nvrhi::rt::ShaderTableDesc());
    if (shaderTable == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the debug trace pass shader table");
        return false;
    }

    shaderTable->setRayGenerationShader(RAYGEN_EXPORT_NAME);

    if (shaderTable->addMissShader(MISS_EXPORT_NAME) < 0 ||
        shaderTable->addHitGroup(CLOSEST_HIT_EXPORT_NAME) < 0 ||
        shaderTable->addHitGroup(CLOSEST_HIT_ALPHA_TESTED_EXPORT_NAME) < 0)
    {
        // findShaderGroup already logged the missing export name through the device's message
        // callback; the pass only records that its table is unusable.
        LogMessage(print, "Warning: RHI: the debug trace pass shader table has no group for one of its export names");
        shaderTable = nullptr;
        return false;
    }

    created = true;
    return true;
}

void RhiDebugTracePass::Render(nvrhi::ICommandList *pCommandList,
                               uint32_t frameIndex,
                               nvrhi::rt::IAccelStruct *pTopLevel,
                               uint32_t width,
                               uint32_t height,
                               uint64_t albedoImage,
                               uint64_t albedoView,
                               VkFormat albedoFormat)
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
            LogMessage(print, "Warning: RHI: the debug trace pass got no top-level acceleration structure, the trace is skipped");
        }
        return;
    }

    if (albedoImage == 0)
    {
        if (!warnedMissingAlbedo)
        {
            warnedMissingAlbedo = true;
            LogMessage(print, "Warning: RHI: the debug trace pass got no ALBEDO image, the trace is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // Re-wrap when the slot's image or size changed: the engine re-created its framebuffers, the
    // swap permutation flipped, or this is the slot's first frame. The replaced wrap and the set
    // over it go through the retire queue, because a recorded list may still reference them.
    if (target.albedoTexture == nullptr || target.albedoImage != albedoImage ||
        target.width != width || target.height != height)
    {
        ReleaseTarget(target);

        target.albedoTexture = rhi::wrapEngineStorageImage(
            device, albedoImage, albedoView, albedoFormat, width, height,
            "RhiDebugTrace ALBEDO frame " + std::to_string(frameIndex));

        if (target.albedoTexture == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to wrap the ALBEDO image for the debug trace pass");
            return;
        }

        target.albedoImage = albedoImage;
        target.width = width;
        target.height = height;
    }

    // Set 2 follows the wrap; ReleaseTarget above nulls it, so this rebuilds it exactly when the
    // wrap changed (or when an earlier createBindingSet failed and is being retried).
    if (target.albedoSet == nullptr)
    {
        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(ALBEDO_SLOT, target.albedoTexture));

        target.albedoSet = device->createBindingSet(setDesc, albedoLayout);

        if (target.albedoSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the debug trace pass ALBEDO binding set");
            return;
        }
    }

    // Set 0 follows the TLAS stream 1 hands over. The AS object is per frame slot and can be
    // replaced when the scene changes, so the set is rebuilt when the pointer changes; the set
    // holds a reference that keeps the AS alive, hence the replaced one goes through the retire
    // queue, never a plain drop.
    if (target.tlasSet == nullptr || target.topLevel != pTopLevel)
    {
        if (target.tlasSet != nullptr)
        {
            frameContext->Retire(target.tlasSet);
        }
        target.tlasSet = nullptr;

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(TLAS_SLOT, pTopLevel));

        target.tlasSet = device->createBindingSet(setDesc, tlasLayout);

        if (target.tlasSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the debug trace pass TLAS binding set");
            return;
        }

        target.topLevel = pTopLevel;
    }

    // The ALBEDO state contract, spelled out on Render in the header: an engine framebuffer image
    // rests in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap keeps no
    // state between command lists (keepInitialState is false, RhiTextureSource.h), so every list
    // announces that state before the first use. The Texture_UAV binding requires exactly the same
    // state (vulkan-state-tracking.cpp:61-63), so the announcement turns the automatic barrier into
    // a same-layout UAV barrier and the image stays in the engine's GENERAL layout.
    pCommandList->beginTrackingTextureState(target.albedoTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);

    // Sets 0, 1 and 2, in the layout order the pipeline was built with; the pinned backend's legacy
    // binding mode binds the list positionally (vulkan-resource-bindings.cpp:940-958). The
    // automatic-barrier pass of setRayTracingState derives the state of each binding from its kind:
    // the TLAS's buffer must be AccelStructRead (its create-time resting state), the UAV must be
    // UnorderedAccess.
    nvrhi::rt::State state;
    state.setShaderTable(shaderTable);
    state.addBindingSet(target.tlasSet);
    state.addBindingSet(uniformSet);
    state.addBindingSet(target.albedoSet);

    pCommandList->setRayTracingState(state);

    // dispatchRays is traceRaysKHR over the four regions of the baked table and these dimensions
    // (vulkan-raytracing.cpp:1459-1473): one ray per pixel of the render resolution.
    nvrhi::rt::DispatchRaysArguments args;
    args.setDimensions(width, height, 1);
    pCommandList->dispatchRays(args);
}

void RhiDebugTracePass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiDebugTracePass::ReleaseTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the wrap references an engine image the GPU may still be writing through the UAV, and
    // the sets reference that wrap and stream 1's AS (RhiFrameContext.h). The queue takes its
    // reference now, so the handles below can be cleared immediately.
    if (frameContext != nullptr)
    {
        if (target.albedoSet != nullptr)
        {
            frameContext->Retire(target.albedoSet);
        }
        if (target.tlasSet != nullptr)
        {
            frameContext->Retire(target.tlasSet);
        }
        if (target.albedoTexture != nullptr)
        {
            frameContext->Retire(target.albedoTexture);
        }
    }

    target.albedoSet = nullptr;
    target.tlasSet = nullptr;
    target.albedoTexture = nullptr;
    target.topLevel = nullptr;
    target.albedoImage = 0;
    target.width = 0;
    target.height = 0;
}

bool RhiDebugTracePass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the debug trace pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
