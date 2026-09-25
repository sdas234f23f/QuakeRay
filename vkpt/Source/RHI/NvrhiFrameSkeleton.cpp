/*
* Copyright (c) 2024 Sultim Tsyrendashiev
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "NvrhiFrameSkeleton.h"

#include "RhiAccelStructs.h"
#include "RhiDebugTracePass.h"
#include "RhiRtComposePass.h"
#include "RhiRtDirectPass.h"
#include "RhiRtIndirectPass.h"
#include "RhiRtPrimaryPass.h"
#include "RhiTextureSource.h"
#include "RhiDescriptors.h"
#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiSkyPass.h"
#include "RhiTextureTable.h"

#include <fstream>

#include "../Const.h"
#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../GlobalUniform.h"
#include "../Swapchain.h"
#include "../Tonemapping.h"

using namespace vkpt;

namespace
{

// The fullscreen triangle keeps its blob name and its source; only the fragment half changes from
// the A1 checkerboard (RhiSkeleton.frag) to the present of the rasterized sky (RhiPresent.frag).
const char *const VERTEX_SHADER_FILE_NAME = "RhiSkeleton.vert.spv";
const char *const PIXEL_SHADER_FILE_NAME = "RhiPresent.frag.spv";

// The present's constant buffer: exposure.x scales the linear ALBEDO sample before the shader's
// x / (1 + x) curve, and exposure.y selects the vertical mirror of the sample coordinate - the
// traced modes' ALBEDO follows the engine's ray-tracing convention and the raster modes follow
// NVRHI's, and one present serves both (RhiPresent.frag.hlsl:50-80). The engine's own exposure
// control (Tonemapping) is not on the RHI path yet, so this first cut writes a fixed 1.0; the
// members stay a float4 to keep the shader's block shape.
struct RhiPresentParams
{
    float exposure[4];
};

const RhiPresentParams PRESENT_PARAMS =
{
    { 1.0f, 0.0f, 0.0f, 0.0f },
};

// The constant buffer is written every frame, and up to two frames (the engine's pacing) plus the
// swapchain's images can be in flight, so NVRHI keeps this many versions of it.
const uint32_t PRESENT_PARAMS_VERSIONS = 4;

}


NvrhiFrameSkeleton::NvrhiFrameSkeleton(nvrhi::IDevice *pDevice,
                                       const Swapchain *pSwapchain,
                                       const char *pShaderFolderPath,
                                       rhi::RhiTextureTable *pTextureTable,
                                       rhi::RhiFrameContext *pFrameContext,
                                       rhi::RhiAccelStructs *pAccelStructs,
                                       RhiDebugTracePass *pDebugTracePass,
                                       RhiRtPrimaryPass *pRtPrimaryPass,
                                       RhiRtDirectPass *pRtDirectPass,
                                       RhiRtIndirectPass *pRtIndirectPass,
                                       RhiRtComposePass *pRtComposePass,
                                       FrameMode mode,
                                       PrintFunction pfnPrint)
    : device(dynamic_cast<nvrhi::vulkan::IDevice *>(pDevice))
    , print(std::move(pfnPrint))
    , shaderFolderPath(pShaderFolderPath != nullptr ? pShaderFolderPath : "")
    , accelStructs(pAccelStructs)
    , debugTracePass(pDebugTracePass)
    , rtPrimaryPass(pRtPrimaryPass)
    , rtDirectPass(pRtDirectPass)
    , rtIndirectPass(pRtIndirectPass)
    , rtComposePass(pRtComposePass)
    , frameMode(mode)
    , textureTable(pTextureTable)
    , frameContext(pFrameContext)
    , unavailable(false)
{
    assert(device != nullptr);
    assert(pSwapchain != nullptr);

    if (device == nullptr)
    {
        print("Warning: RHI: the frame skeleton needs the Vulkan backend of the RHI layer");
        unavailable = true;
        return;
    }

    if (textureTable == nullptr)
    {
        print("Warning: RHI: the frame skeleton needs the shared texture table of the RHI layer");
        unavailable = true;
        return;
    }

    if (frameContext == nullptr)
    {
        print("Warning: RHI: the frame skeleton needs the frame context of the RHI layer");
        unavailable = true;
        return;
    }

    // The acceleration-structure stream is recorded every frame, in every mode, so it is a hard
    // dependency of the skeleton: a frame that cannot build its structures falls back to the legacy
    // renderer instead of recording a half-wired one.
    if (accelStructs == nullptr || !accelStructs->IsCreated())
    {
        print("Warning: RHI: the frame skeleton needs the acceleration structures of the RHI layer");
        unavailable = true;
        return;
    }

    // The ray-tracing passes are hard dependencies only of their own modes; a pass whose flag is
    // off is not created by the host at all. The traced chain needs both of its passes: they share
    // the pipeline layouts and the per-slot sets, and one without the other is a half-wired frame.
    if (frameMode == FrameMode::DebugTrace && (debugTracePass == nullptr || !debugTracePass->IsCreated()))
    {
        print("Warning: RHI: the debug-traced frame needs the debug ray-tracing pass of the RHI layer");
        unavailable = true;
        return;
    }

    if (frameMode == FrameMode::Traced &&
        (rtPrimaryPass == nullptr || !rtPrimaryPass->IsCreated() ||
         rtDirectPass == nullptr || !rtDirectPass->IsCreated() ||
         rtIndirectPass == nullptr || !rtIndirectPass->IsCreated()))
    {
        print("Warning: RHI: the traced frame needs the primary, direct and indirect ray-tracing passes of the RHI layer");
        unavailable = true;
        return;
    }

    if (!LoadShader(VERTEX_SHADER_FILE_NAME, nvrhi::ShaderType::Vertex, vertexShader) ||
        !LoadShader(PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, pixelShader))
    {
        unavailable = true;
        return;
    }

    // The rasterized sky pass: the engine's first real pass on the RHI path. It is created here
    // because this class already owns every input its Create needs (the device, the table, the
    // frame context and the shader folder). A failure is not fatal for the engine: it makes this
    // skeleton unavailable, RenderThroughRhi returns false and the legacy renderer keeps the frame.
    skyPass = std::make_unique<RhiSkyPass>();
    if (!skyPass->Create(device, textureTable, frameContext, shaderFolderPath.c_str(), print))
    {
        skyPass.reset();
        print("Warning: RHI: the rasterized sky pass is unavailable, the legacy renderer is kept");
        unavailable = true;
        return;
    }

    if (!CreatePassResources())
    {
        unavailable = true;
        return;
    }

    OnSwapchainCreate(pSwapchain);
}

NvrhiFrameSkeleton::~NvrhiFrameSkeleton()
{
    if (device != nullptr)
    {
        // The textures wrap images that the swapchain owns.
        device->waitForIdle();
    }

    DestroySwapchainResources();

    // The pass wraps engine images and owns the pipelines and the target of the sky; it goes after
    // the swapchain resources that dropped its presented wraps above.
    skyPass.reset();

    pipeline = nullptr;
    vertexShader = nullptr;
    pixelShader = nullptr;

    bindingLayout = nullptr;
    presentParamsBuffer = nullptr;
    presentSampler = nullptr;
}

void NvrhiFrameSkeleton::OnSwapchainCreate(const Swapchain *pSwapchain)
{
    if (unavailable)
    {
        return;
    }

    DestroySwapchainResources();

    const nvrhi::Format colorFormat = ConvertSurfaceFormat(pSwapchain->GetSurfaceFormat());
    if (colorFormat == nvrhi::Format::UNKNOWN)
    {
        print("Warning: RHI: the frame skeleton does not support the swapchain format");
        unavailable = true;
        return;
    }

    if (pipeline == nullptr || pipelineColorFormat != colorFormat)
    {
        if (!CreatePipeline(colorFormat))
        {
            unavailable = true;
            return;
        }
    }

    if (!CreateSwapchainResources(pSwapchain))
    {
        unavailable = true;
        return;
    }
}

void NvrhiFrameSkeleton::OnSwapchainDestroy()
{
    if (device != nullptr)
    {
        device->waitForIdle();
    }

    DestroySwapchainResources();
}

bool NvrhiFrameSkeleton::IsUnavailable() const
{
    return unavailable;
}

bool NvrhiFrameSkeleton::Render(const Swapchain *pSwapchain, uint32_t frameIndex, const SkyFrameInputs &sky,
                                VkSemaphore semaphoreToWait, VkSemaphore semaphoreToSignal)
{
    if (unavailable)
    {
        return false;
    }

    // A minimized window gives the swapchain a zero extent, and the renderer
    // skips such frames too instead of creating an empty target.
    if (pSwapchain->GetWidth() == 0 || pSwapchain->GetHeight() == 0)
    {
        return false;
    }

    const uint32_t imageIndex = pSwapchain->GetCurrentImageIndex();
    if (imageIndex >= swapchainFramebuffers.size())
    {
        return false;
    }

    nvrhi::ITexture *backBuffer = swapchainTextures[imageIndex];
    nvrhi::IFramebuffer *framebuffer = swapchainFramebuffers[imageIndex];

    // The frame context owns the frame model: BeginSlot waits for the slot's previous submission,
    // drains the slot's retire queue and opens its list, so no long-lived list is opened here.
    frameContext->BeginSlot(frameIndex);

    nvrhi::ICommandList *commandList = frameContext->GetCommandList(frameIndex);
    assert(commandList != nullptr);

    // Newly wrapped engine textures are foreign to NVRHI and need their first-use state declared in
    // the first command list that samples them (RhiTextureSource.h); the shared table hands over
    // what was wrapped since the last frame.
    textureTable->TrackPendingTextures(commandList);

    // The sky pass's Prepare runs in both modes: it selects the slot's ALBEDO target, wraps it and
    // announces the state the engine leaves it in (UnorderedAccess, i.e. GENERAL) - the wrap the
    // present samples below and, in the traced mode, the announcement the debug pass's UAV write
    // relies on (the trace is then the image's first use of the list, so no transition precedes it).
    if (skyPass != nullptr && sky.framebuffers != nullptr)
    {
        skyPass->Prepare(commandList, frameIndex, *sky.framebuffers, sky.width, sky.height);
    }

    // The engine's GlobalUniform::Upload runs only from Scene::SubmitForFrame (Scene.cpp:112), which
    // the RHI path does not call, so the device-local uniform would hold stale or zero data. Every
    // mode reads it - the world shader takes renderWidth from it (member 11, byte 644) for its
    // checkerboard remap, and the traced raygens take the camera, the jitter, the ray limits and the
    // cull masks - so this write is a common step of the frame. It is a static wrap of the engine's
    // buffer, and under `rhiframe` nothing else touches that buffer.
    // The world's wrap of the engine uniform is what the write below targets, and the traced mode
    // never runs PrepareWorld, so the wrap is created here, lazily, for both modes. The wrap's desc
    // mirrors the one PrepareWorld uses: isConstantBuffer (the validation device refuses a
    // ConstantBuffer binding on a desc without it, validation-device.cpp:1717-1723), not volatile,
    // resting in the state the engine buffer really is in.
    if (worldUniformBuffer == nullptr && sky.uniform != nullptr)
    {
        nvrhi::BufferDesc uniformDesc;
        uniformDesc.byteSize = sizeof(ShGlobalUniform);
        uniformDesc.isConstantBuffer = true;
        uniformDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        uniformDesc.keepInitialState = true;
        uniformDesc.debugName = "RHI world uniform (GlobalUniform wrap)";

        worldUniformBuffer = device->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sky.uniform->GetBuffer()))),
            uniformDesc);
    }

    const bool tracedFrame = frameMode != FrameMode::Rasterized;

    // The acceleration-structure stream of the frame, recorded in every mode and before anything
    // reads it: BuildStatic keeps the static BLAS in sync with the engine's components (it rebuilds
    // them when the engine's static generation changes) and BuildTopLevel records the slot's TLAS -
    // the module synthesises the instance list itself, from the engine's components and filters with
    // our own BLAS handles, so nothing here depends on the engine's instance buffer any more. Only
    // the traced modes need the TLAS, so the rasterized frame records no top-level build.
    if (accelStructs != nullptr)
    {
        accelStructs->BuildStatic(commandList);

        if (tracedFrame)
        {
            accelStructs->BuildTopLevel(commandList, frameIndex, sky.rayCullMaskWorld,
                                        sky.allowGeometryWithSkyFlag, sky.disableRayTracedGeometry);
        }
    }

    // The engine's device-local uniform (see the wrap above). The write follows the
    // acceleration-structure stream because the traced modes build their instance list in
    // RhiAccelStructs, and the per-instance geometry offsets the shaders read
    // (globalUniform.instanceGeomInfoOffset/Count) must describe that list: the module fills them
    // in its own TLAS order right after the build, and they are patched into the CPU copy here,
    // before it is uploaded - the host's PrepareForBuildingTLAS fills only the static prefix of the
    // engine's own list. A frame whose build produced no instance list leaves the CPU copy alone.
    if (sky.uniform != nullptr && worldUniformBuffer != nullptr)
    {
        if (tracedFrame && accelStructs != nullptr)
        {
            int32_t instanceGeomInfoOffset[MAX_TOP_LEVEL_INSTANCE_COUNT] = {};
            int32_t instanceGeomInfoCount[MAX_TOP_LEVEL_INSTANCE_COUNT] = {};
            const uint32_t instanceCount = accelStructs->GetInstanceGeometryInfo(
                instanceGeomInfoOffset, instanceGeomInfoCount);

            if (instanceCount > 0)
            {
                memcpy(sky.uniform->GetData()->instanceGeomInfoOffset, instanceGeomInfoOffset,
                       instanceCount * sizeof(int32_t));
                memcpy(sky.uniform->GetData()->instanceGeomCount, instanceGeomInfoCount,
                       instanceCount * sizeof(int32_t));
            }
        }

        // The A4.2a switch, forced for the traced chain until A4.5 lifts it: fltEnable[0] = 0 makes
        // q2GetIsGradient return before reading the gradient-sample-position image, which the
        // traced chain never writes under `rhiframe` (the ASVGF chain is A4.5). q2LightStatsMode is
        // no longer forced: the direct pass fills the statistics slots itself from the engine's
        // cvar-driven value (RhiRtDirectPass.h documents the host contract).
        if (tracedFrame)
        {
            sky.uniform->GetData()->fltEnable[0] = 0.0f;
        }

        rhi::writeBuffer(commandList, worldUniformBuffer, sky.uniform->GetData(), sizeof(ShGlobalUniform));
    }

    if (!tracedFrame)
    {
        // The engine's rasterized sky, exactly the calls the pass's contract requires, on the one
        // open list: Prepare above selected the slot's ALBEDO target, SetSkyCamera carries the same
        // values the legacy DrawSkyToAlbedo call receives (VulkanDevice.cpp:748-756), and Render
        // records the frame's sky list.
        if (skyPass != nullptr && sky.framebuffers != nullptr)
        {
            skyPass->SetSkyCamera(sky.view, sky.projection, sky.jitter, sky.skyViewerPos);
            skyPass->Render(commandList, sky.draws, sky.drawCount, sky.applyVertexColorGamma);
        }

        // The rasterized world sub-pass on top of the sky, into the same ALBEDO and depth. The world
        // is created here, on the first frame that can have it, and not in the constructor: its
        // engine image handles come from the Framebuffers accessors, and those answer from the image
        // vector only Framebuffers::PrepareForSize fills - which the host calls per frame, before this
        // Render, so the sky's Prepare above is the earliest point at which the handles are valid.
        // When the world was not created (the frame carried no engine inputs yet, or PrepareWorld hit
        // a permanent failure and logged it), the world's per-frame steps below - the exposure
        // stand-in, the uniform write and RenderWorld - are all skipped, and the frame is exactly what
        // the sky above and the present below made of it.
        if (skyPass != nullptr && !skyPass->IsWorldCreated() && !worldCreationFailed)
        {
            PrepareWorld(frameIndex, sky);
        }

        if (skyPass != nullptr && skyPass->IsWorldCreated() &&
            sky.uniform != nullptr && sky.tonemapping != nullptr)
        {
            // The exposure chain (Tonemapping::CalculateExposure) runs only from the legacy Render
            // (VulkanDevice.cpp:1061), which the RHI path takes over, so nothing computes the engine
            // buffer's avgLuminance under `rhiframe`. The world shader multiplies every output by a
            // factor derived from it (Exposure.h:33-51):
            //   factor = 1 / (1.2 * exp2(log2(avg * 100 / 12.5))) = 1 / (9.6 * avg).
            // The buffer is constructed zeroed, which would make the factor zero and the whole rasterized
            // world black; the neutral stand-in is the value that gives factor 1,
            // 1 / 9.6 = 0.10417, which is also close to a dim map's real average. DEV STAND-IN until the
            // exposure chain is ported.
            sky.tonemapping->SetAvgLuminance(frameIndex, 1.0f / 9.6f);

            // The world draws, exactly the sequence the sky got above: the pass owns the target, the
            // binding sets, the state changes and the depth - it does not clear the depth a second
            // time, so the world depth-tests against the sky's.
            skyPass->RenderWorld(commandList, sky.worldDraws, sky.worldDrawCount, sky.applyVertexColorGamma);
        }
    }
    else if (frameMode == FrameMode::Traced && rtPrimaryPass != nullptr && sky.framebuffers != nullptr)
    {
        // The real ray-tracing first cut: the engine's primary-visibility raygen fills the slot's
        // checkerboard G-buffer images (ALBEDO included) over the TLAS the stream above built. The
        // set 3 vertex-data buffers are the module's per-frame answer (RhiAccelStructs: the static
        // collector's buffers wrapped, the dynamic data as the per-slot copies the BLAS builds from
        // - the same bytes both consumers describe), in the engine's raw binding order. The pass
        // resolves the 26 framebuffer image handles and wraps them itself, so only the Framebuffers
        // object is passed on. The image is expected to look flat and unlit: the light-source set
        // stays empty until A4.2. The present shows the raygen's ALBEDO, which - unlike the rest of
        // the checkerboard G-buffer - the raygen writes in regular space (it maps the packed pixel
        // back with getRegularPixFromCheckerboardPix, RaygenPrimary.hlsli:469), so the flat albedo
        // image is artifact-free while the other G-buffer images stay packed for A4.2-A4.4.
        const rhi::RhiAccelStructs::VertexDataBuffers vertexData = accelStructs->GetVertexDataBuffers(frameIndex);

        RhiRtPrimaryPass::VertexData passVertexData;
        passVertexData.staticVertices = vertexData.staticVertices;
        passVertexData.dynamicVertices = vertexData.dynamicVertices;
        passVertexData.staticIndices = vertexData.staticIndices;
        passVertexData.dynamicIndices = vertexData.dynamicIndices;
        passVertexData.geometryInstances = vertexData.geometryInstances;
        passVertexData.dynamicVerticesPrev = vertexData.previousDynamicVertices;
        passVertexData.prevDynamicIndices = vertexData.previousDynamicIndices;

        rtPrimaryPass->Render(
            commandList, frameIndex,
            accelStructs != nullptr ? accelStructs->GetTopLevel(frameIndex) : nullptr,
            worldUniformBuffer.Get(),
            passVertexData,
            sky.framebuffers,
            sky.width, sky.height);

        // The uniform bytes the traced raygens read; the passes below take their per-frame inputs
        // from the same CPU copy the write above uploaded.
        const ShGlobalUniform *uniform = sky.uniform != nullptr ? sky.uniform->GetData() : nullptr;

        // The direct-lighting pass reads what the primary just wrote (the G-buffer, the Q2 cluster
        // and the seed) and the frame's light buffers; it records on the same list right after the
        // primary, the order the legacy frame uses (VulkanDevice.cpp:901 then :1039) and the order
        // its state announcements and the present's direct read assume. The light-statistics
        // bookkeeping follows the uniform bytes the raygen reads, not the frame slot: the statistics
        // buffer has three rotating slots and the raygen addresses frameId % 3 (RhiRtDirectPass.h
        // documents the contract).
        if (rtDirectPass != nullptr && uniform != nullptr)
        {
            rtDirectPass->Render(
                commandList, frameIndex,
                uniform->frameId, uniform->q2LightStatsMode,
                accelStructs != nullptr ? accelStructs->GetTopLevel(frameIndex) : nullptr,
                worldUniformBuffer.Get(),
                passVertexData,
                sky.framebuffers,
                sky.width, sky.height);
        }

        // The indirect / GI pass reads the G-buffer and the direct outputs and writes the unfiltered
        // indirect images the compose pass then filters; it records right after the direct pass. The
        // half-resolution switch is the engine's own giBounceRays[0], applied inside the pass
        // (RhiRtIndirectPass.h documents the sizing).
        if (rtIndirectPass != nullptr && uniform != nullptr)
        {
            rtIndirectPass->Render(
                commandList, frameIndex,
                uniform->giBounceRays[0],
                accelStructs != nullptr ? accelStructs->GetTopLevel(frameIndex) : nullptr,
                worldUniformBuffer.Get(),
                passVertexData,
                sky.framebuffers,
                sky.width, sky.height);
        }

        // The compose preview: the real adapter -> interleave -> checkerboard chain over the
        // G-buffer and the direct and indirect buffers, writing FINAL, which the present samples
        // when the pass exists. It records after the indirect pass for the same reason it does.
        if (rtComposePass != nullptr)
        {
            rtComposePass->Render(commandList, frameIndex, sky.framebuffers, sky.width, sky.height,
                                  worldUniformBuffer.Get());
        }
    }
    else if (debugTracePass != nullptr && sky.framebuffers != nullptr)
    {
        // The debug trace frame: one primary ray per pixel over the TLAS the stream above just
        // built, into the slot's ALBEDO image. The handles are the same ones the sky pass resolves
        // for its own target (Framebuffers resolves the slot's swap permutation inside), and the
        // debug pass wraps them itself; the raster sky/world sub-passes are deliberately not
        // recorded in this mode, so the trace is ALBEDO's first user of the list. A null TLAS (the
        // stream has not built one yet) leaves the image untouched and the present shows the
        // previous frame's content; the pass warns once about it.
        const auto [albedoImage, albedoView, albedoFormat] =
            sky.framebuffers->GetImageHandles(FB_IMAGE_INDEX_ALBEDO, frameIndex);

        debugTracePass->Render(
            commandList, frameIndex,
            accelStructs != nullptr ? accelStructs->GetTopLevel(frameIndex) : nullptr,
            sky.width, sky.height,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(albedoImage)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(albedoView)),
            albedoFormat);
    }

    // The present samples the source of this slot: the ALBEDO wrap of the raster and diagnostic
    // modes (the sky pass is the only owner of that wrap), or the compose preview's FINAL image
    // when the preview ran - both are borrowed and only valid until their pass re-wraps the slot.
    // The direct term's image is the skeleton's own wrap (ResolvePresentDirectTexture): every mode
    // resolves it, because the layout's unordered-access item is always filled, but the shader
    // reads it only in the diagnostic mode (the params flag below).
    nvrhi::ITexture *albedo = nullptr;
    if (rtComposePass != nullptr)
    {
        albedo = rtComposePass->GetFinalTexture(frameIndex);
    }
    else if (skyPass != nullptr)
    {
        albedo = skyPass->GetAlbedoTexture(frameIndex);
    }

    nvrhi::ITexture *directTexture = albedo != nullptr ? ResolvePresentDirectTexture(frameIndex, sky) : nullptr;

    if (albedo == nullptr)
    {
        // No target means Prepare did not succeed for this slot (no framebuffers yet, or a failed
        // wrap). The frame is still submitted and presented; the swapchain image keeps its previous
        // content.
        if (!warnedMissingAlbedo)
        {
            warnedMissingAlbedo = true;
            print("Warning: RHI: the present has no ALBEDO target, the swapchain image is left untouched");
        }
    }
    else if (directTexture != nullptr && PreparePresentBindingSet(frameIndex, albedo, directTexture))
    {
        // The exposure of the present: the write takes the next version of the volatile buffer and,
        // as every volatile-buffer write, has to follow the list's open(), which BeginSlot did.
        // exposure.y is the mirror flag of the frame's mode: the engine-convention traced frames
        // need the present to flip the sample coordinate (RhiPresent.frag.hlsl:64-76), the raster
        // and debug-trace frames do not (the debug raygen flips its own store,
        // RhiDebugTrace.rgen.hlsl:107-114). exposure.z enables the direct-lighting term of the
        // diagnostic present - the compose preview feeds FINAL, which already carries the light, so
        // it keeps the term off.
        const bool traced = frameMode == FrameMode::Traced;
        const bool compose = rtComposePass != nullptr;
        RhiPresentParams presentParams = PRESENT_PARAMS;
        presentParams.exposure[1] = traced ? 1.0f : 0.0f;
        presentParams.exposure[2] = (traced && !compose) ? 1.0f : 0.0f;
        rhi::writeBuffer(commandList, presentParamsBuffer, &presentParams, sizeof(presentParams));

        // The swapchain hands the image over in the present layout and expects it
        // back in the same layout, so the pass is wrapped into two transitions.
        // The direct term is read through its storage image; a Texture_UAV binding requires the
        // image's true state - UnorderedAccess, i.e. the GENERAL layout the engine leaves its
        // framebuffer images in and the direct pass writes them in (RhiTextureSource.h:123-134). The
        // announcement makes the first use of every list transition-free; nothing restores the
        // state afterwards, because UnorderedAccess is exactly what the engine and the next frame's
        // trace expect.
        commandList->beginTrackingTextureState(directTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        commandList->beginTrackingTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::Present);
        commandList->setTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::RenderTarget);

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewportAndScissorRect(nvrhi::Viewport(
            float(backBuffer->getDesc().width), float(backBuffer->getDesc().height)));
        state.addBindingSet(presentBindingSets[frameIndex]);

        commandList->setGraphicsState(state);

        nvrhi::DrawArguments args;
        args.vertexCount = 3;
        commandList->draw(args);

        // Sampling moved the ALBEDO wrap to the shader-resource state, and the engine's framebuffer
        // images rest in GENERAL - the layout its own framebuffer descriptors declare
        // (Framebuffers.cpp:786, :794) and the state the sky pass announces at the start of every
        // list. The barrier is committed when the list closes (vulkan-commandlist.cpp:74-80).
        commandList->setTextureState(albedo, nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);

        commandList->setTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::Present);
    }

    // The context closes the slot's list and submits it: the image is not available until the
    // acquire semaphore is signalled, and the presentation engine cannot start before the pass is
    // done, so EndSlot waits on 'semaphoreToWait' and signals 'semaphoreToSignal' where the manual
    // queue state and the execute used to be.
    frameContext->EndSlot(frameIndex, semaphoreToWait, semaphoreToSignal);

    // The fidelity gap (unwrappable formats or swizzles fall back to the white texture) is reported
    // once, after the engine has had a few frames to fill the table; the counter then stops.
    if (framesUntilFallbackLog > 0 && --framesUntilFallbackLog == 0)
    {
        print(("RHI: the texture table holds " + std::to_string(textureTable->GetFallbackSlotCount()) +
               " fallback slot(s) (unwrapable format/swizzle)").c_str());
    }

    return true;
}

bool NvrhiFrameSkeleton::PrepareWorld(uint32_t frameIndex, const SkyFrameInputs &sky)
{
    // A frame without the engine objects, or a frame whose framebuffers are not there yet, is not
    // a failure: the world simply waits for a later frame to bring them.
    if (skyPass == nullptr || sky.framebuffers == nullptr ||
        sky.uniform == nullptr || sky.tonemapping == nullptr)
    {
        return false;
    }

    if (!worldBuffersWrapped)
    {
        // What the two wraps have to declare, measured in the pinned NVRHI:
        //  - the world shader's set 1 is a BindingLayoutItem::ConstantBuffer, and the validation
        //    device refuses such a binding on a buffer whose desc lacks isConstantBuffer
        //    (validation-device.cpp:1717-1723) and on a volatile one (:1725-1730); a volatile wrap
        //    would also become a dynamic-offset binding (nvrhi.h:2311-2318), which the static
        //    layout item cannot take;
        //  - set 2 is a StructuredBuffer_SRV, and the same validation refuses a wrap with
        //    structStride == 0 (:1693-1699); the backend asserts the same condition when the
        //    binding set is created (vulkan-resource-bindings.cpp:535-536).
        // Both descs declare the state the engine buffer really rests in, so the automatic-barrier
        // pass neither reports an unknown prior state on the first use (state-tracking.cpp:290-297)
        // nor emits a transition on a plain bind: the uniform is read as a uniform buffer, the
        // tonemapping buffer as a storage buffer. keepInitialState makes the claim stand for every
        // command list.
        nvrhi::BufferDesc uniformDesc;
        uniformDesc.byteSize = sizeof(ShGlobalUniform);
        uniformDesc.isConstantBuffer = true;
        uniformDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        uniformDesc.keepInitialState = true;
        uniformDesc.debugName = "RHI world uniform (GlobalUniform wrap)";

        worldUniformBuffer = device->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sky.uniform->GetBuffer()))),
            uniformDesc);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            nvrhi::BufferDesc tonemappingDesc;
            tonemappingDesc.byteSize = sky.tonemapping->GetElementSize();
            tonemappingDesc.structStride = sky.tonemapping->GetElementSize();
            tonemappingDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            tonemappingDesc.keepInitialState = true;
            tonemappingDesc.debugName = "RHI world tonemapping wrap " + std::to_string(i);

            worldTonemappingBuffers[i] = device->createHandleForNativeBuffer(
                nvrhi::ObjectTypes::VK_Buffer,
                nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sky.tonemapping->GetBuffer(i)))),
                tonemappingDesc);
        }

        if (worldUniformBuffer == nullptr ||
            worldTonemappingBuffers[0] == nullptr || worldTonemappingBuffers[1] == nullptr)
        {
            print("Warning: RHI: failed to wrap the world uniform/tonemapping buffers, the rasterized world is skipped");
            worldCreationFailed = true;
            return false;
        }

        worldBuffersWrapped = true;
    }

    // The engine handles of this frame are only the seeds: the pass re-reads the same accessors in
    // every Prepare, so a later engine framebuffer re-create is picked up without a second
    // CreateWorld. The raw pointers the pass's array parameter needs are built from the handles
    // above, which stay the owners.
    nvrhi::IBuffer *tonemappingBuffers[MAX_FRAMES_IN_FLIGHT];
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        tonemappingBuffers[i] = worldTonemappingBuffers[i].Get();
    }

    if (!skyPass->CreateWorld(worldUniformBuffer.Get(), tonemappingBuffers,
                              sky.framebuffers->GetScreenEmissionHandles(frameIndex),
                              sky.framebuffers->GetPrimaryToReflRefrHandles(frameIndex)))
    {
        // CreateWorld logged the reason; every failure it reports is permanent.
        worldCreationFailed = true;
        return false;
    }

    return true;
}

nvrhi::Format NvrhiFrameSkeleton::ConvertSurfaceFormat(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8A8_SRGB:
            return nvrhi::Format::SRGBA8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return nvrhi::Format::SBGRA8_UNORM;
        default:
            return nvrhi::Format::UNKNOWN;
    }
}

bool NvrhiFrameSkeleton::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'unavailable' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        print(("Warning: RHI: cannot load the frame skeleton shader \"" + path + "\"").c_str());
        return false;
    }

    return true;
}

bool NvrhiFrameSkeleton::CreatePassResources()
{
    // The four bindings of the present, at the NVRHI slots whose Vulkan binding numbers the shader
    // spells [[vk::binding(...)]] for: the constant buffer at 256, the ALBEDO texture at 0 and its
    // sampler at 128, and the storage image of the direct term at 384 - the default offset of an
    // unordered-access view of slot 0 (nvrhi.h:2066-2072; the slot-to-binding rule is documented in
    // RhiPipeline.h). The layout becomes descriptor set 0, the only set RhiPresent.frag declares.
    const nvrhi::BindingLayoutItem layoutItems[] =
    {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
    };

    bindingLayout = rhi::createBindingLayout(device, layoutItems, "RhiPresent bindings");
    if (bindingLayout == nullptr)
    {
        print("Warning: RHI: failed to create the present binding layout");
        return false;
    }

    // Per-frame contents, so the buffer is volatile: NVRHI allocates PRESENT_PARAMS_VERSIONS copies
    // of it and its state tracker skips volatile buffers entirely, which is the canonical shape of
    // nvrhi::CreateVolatileConstantBufferDesc (third_party/nvrhi/src/common/utils.cpp). A write is
    // only allowed after the command list has been opened, and every presented frame writes one
    // version.
    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = sizeof(RhiPresentParams);
    bufferDesc.isConstantBuffer = true;
    bufferDesc.isVolatile = true;
    bufferDesc.maxVersions = PRESENT_PARAMS_VERSIONS;

    presentParamsBuffer = rhi::createBuffer(device, bufferDesc, "RhiPresent params");
    if (presentParamsBuffer == nullptr)
    {
        print("Warning: RHI: failed to create the present constant buffer");
        return false;
    }

    // The sampler of the present: the ALBEDO render image can be smaller than the swapchain (the
    // engine's resolution modes render below the window size by default), so the present magnifies
    // it. Linear minification/magnification and clamp on all axes are nvrhi::SamplerDesc's defaults
    // (nvrhi.h:1333-1338) and the shape of the engine's own bilinear sampler
    // (SamplerManager.cpp:89-92). The shader's sample is an explicit LOD 0, so the mip mode only
    // stays consistent with the engine's linear mip mode.
    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setAllFilters(true);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Clamp);

    presentSampler = rhi::createSampler(device, samplerDesc, "RhiPresent sampler");
    if (presentSampler == nullptr)
    {
        print("Warning: RHI: failed to create the present sampler");
        return false;
    }

    return true;
}

nvrhi::ITexture *NvrhiFrameSkeleton::ResolvePresentDirectTexture(uint32_t frameIndex, const SkyFrameInputs &sky)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);

    if (sky.framebuffers == nullptr || sky.width == 0 || sky.height == 0)
    {
        return nullptr;
    }

    const auto [image, view, format] = sky.framebuffers->GetImageHandles(FB_IMAGE_INDEX_UNFILTERED_DIRECT, frameIndex);
    const uint64_t handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));

    if (handle == 0)
    {
        if (!warnedMissingDirect)
        {
            warnedMissingDirect = true;
            print("Warning: RHI: the present has no direct-lighting image, the frame is left untouched");
        }
        return nullptr;
    }

    if (presentDirectTextures[frameIndex] == nullptr || presentDirectImageHandles[frameIndex] != handle)
    {
        // The image is new or the engine re-created its framebuffers (the same point at which the
        // sky pass re-wraps ALBEDO): the previous wrap goes through the retire queue, and the new
        // one is announced to the present's list by Render.
        if (presentDirectTextures[frameIndex] != nullptr && frameContext != nullptr)
        {
            frameContext->Retire(presentDirectTextures[frameIndex]);
        }

        presentDirectTextures[frameIndex] = rhi::wrapEngineStorageImage(
            device, handle,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view)), format,
            sky.width, sky.height,
            "RhiPresent direct image frame " + std::to_string(frameIndex));
        presentDirectImageHandles[frameIndex] = handle;

        if (presentDirectTextures[frameIndex] == nullptr)
        {
            if (!warnedMissingDirect)
            {
                warnedMissingDirect = true;
                print("Warning: RHI: failed to wrap the direct-lighting image, the frame is left untouched");
            }
            return nullptr;
        }
    }

    return presentDirectTextures[frameIndex].Get();
}

bool NvrhiFrameSkeleton::PreparePresentBindingSet(uint32_t frameIndex, nvrhi::ITexture *albedo,
                                                  nvrhi::ITexture *directTexture)
{
    assert(frameIndex < MAX_FRAMES_IN_FLIGHT);

    // The set references the slot's ALBEDO wrap and its direct-term image, so it follows both: the
    // sky pass replaces the ALBEDO wrap when the engine re-creates its framebuffers, and
    // ResolvePresentDirectTexture replaces the direct wrap at the same point. A set built for a
    // replaced wrap would sample an image that is on its way out.
    if (presentBindingSets[frameIndex] != nullptr &&
        presentAlbedoTextures[frameIndex].Get() == albedo &&
        presentDirectSetTextures[frameIndex].Get() == directTexture)
    {
        return true;
    }

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, presentParamsBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, albedo));
    setDesc.addItem(nvrhi::BindingSetItem::Sampler(0, presentSampler));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(0, directTexture));

    nvrhi::BindingSetHandle bindingSet = device->createBindingSet(setDesc, bindingLayout);
    if (bindingSet == nullptr)
    {
        print("Warning: RHI: failed to create the present binding set");
        return false;
    }

    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue (RhiFrameContext.h): the old set and the texture handles it held are released only after
    // the queue finished the submission that could still use them.
    if (frameContext != nullptr)
    {
        if (presentBindingSets[frameIndex] != nullptr)
        {
            frameContext->Retire(presentBindingSets[frameIndex]);
        }

        if (presentAlbedoTextures[frameIndex] != nullptr)
        {
            frameContext->Retire(presentAlbedoTextures[frameIndex]);
        }

        if (presentDirectSetTextures[frameIndex] != nullptr)
        {
            frameContext->Retire(presentDirectSetTextures[frameIndex]);
        }
    }

    presentBindingSets[frameIndex] = std::move(bindingSet);
    presentAlbedoTextures[frameIndex] = albedo;
    presentDirectSetTextures[frameIndex] = directTexture;
    return true;
}

bool NvrhiFrameSkeleton::CreatePipeline(nvrhi::Format colorFormat)
{
    nvrhi::GraphicsPipelineDesc desc;
    desc.setVertexShader(vertexShader);
    desc.setPixelShader(pixelShader);
    desc.primType = nvrhi::PrimitiveType::TriangleList;
    // One layout, the present's own set 0. RhiPresent.frag declares no bindless table, so the
    // pipeline of the checkerboard (which carried the shared table as its second set) is gone.
    desc.addBindingLayout(bindingLayout);

    // The fullscreen triangle is drawn in clip space, without any vertex buffer.
    desc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
    desc.renderState.rasterState.depthClipEnable = true;
    desc.renderState.depthStencilState.disableDepthTest();
    desc.renderState.depthStencilState.disableDepthWrite();

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.addColorFormat(colorFormat);

    pipeline = rhi::createGraphicsPipeline(device, desc, framebufferInfo, "RhiPresent pipeline");
    if (pipeline == nullptr)
    {
        print("Warning: RHI: failed to create the frame skeleton pipeline");
        return false;
    }

    pipelineColorFormat = colorFormat;
    return true;
}

bool NvrhiFrameSkeleton::CreateSwapchainResources(const Swapchain *pSwapchain)
{
    nvrhi::TextureDesc textureDesc;
    textureDesc.width = pSwapchain->GetWidth();
    textureDesc.height = pSwapchain->GetHeight();
    textureDesc.dimension = nvrhi::TextureDimension::Texture2D;
    textureDesc.format = pipelineColorFormat;
    textureDesc.mipLevels = 1;
    textureDesc.arraySize = 1;
    textureDesc.sampleCount = 1;
    textureDesc.isRenderTarget = true;
    textureDesc.debugName = "Swapchain image (RHI frame skeleton)";

    for (uint32_t i = 0; i < pSwapchain->GetImageCount(); i++)
    {
        const VkImage image = pSwapchain->GetImage(i);

        const nvrhi::Object nativeImage(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image)));

        nvrhi::TextureHandle texture =
            device->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, nativeImage, textureDesc);

        if (texture == nullptr)
        {
            print("Warning: RHI: failed to wrap a swapchain image");
            return false;
        }

        nvrhi::FramebufferDesc framebufferDesc;

        framebufferDesc.addColorAttachment(texture);

        nvrhi::FramebufferHandle framebuffer = device->createFramebuffer(framebufferDesc);
        if (framebuffer == nullptr)
        {
            print("Warning: RHI: failed to create a framebuffer for a swapchain image");
            return false;
        }

        swapchainTextures.push_back(std::move(texture));
        swapchainFramebuffers.push_back(std::move(framebuffer));
    }

    return true;
}

void NvrhiFrameSkeleton::DestroySwapchainResources()
{
    // The sky pass wraps the engine's ALBEDO images and keeps one wrapped target per frame slot.
    // Dropping them here is what lets the engine destroy and re-create its framebuffers: the
    // re-create path of the swapchain waits for the device before it notifies
    // (Swapchain.cpp:323-326), and the next Prepare re-wraps the new images. The releases go
    // through the frame context's retire queue.
    if (skyPass != nullptr)
    {
        skyPass->ReleaseTargets();
    }

    // The debug trace pass wraps the same engine ALBEDO images and keeps its own per-slot wrap, UAV
    // set and TLAS set over them, so it has to drop them at the same point and for the same reason
    // as the sky pass above.
    if (debugTracePass != nullptr)
    {
        debugTracePass->ReleaseTargets();
    }

    // The primary ray-tracing pass wraps the engine's 26 checkerboard G-buffer images and keeps its
    // own per-slot wraps and sets over them, so it drops them here for the same reason as the two
    // passes above.
    if (rtPrimaryPass != nullptr)
    {
        rtPrimaryPass->ReleaseTargets();
    }

    // The direct-lighting pass wraps its twelve set-1 images kept alongside their light-buffer
    // wraps, so it drops them here for the same reason. The light buffers themselves outlive a
    // resize; the next Render re-wraps them.
    if (rtDirectPass != nullptr)
    {
        rtDirectPass->ReleaseTargets();
    }

    // The compose preview wraps the framebuffer images it reads and writes and keeps the FINAL wrap
    // the present borrows when it ran, so it drops them here as well.
    if (rtComposePass != nullptr)
    {
        rtComposePass->ReleaseTargets();
    }

    // The indirect pass wraps its fourteen set-1 images - the indirect SH images among them - so it
    // drops them here as well; the blue-noise wrap is the host's and outlives the pass.
    if (rtIndirectPass != nullptr)
    {
        rtIndirectPass->ReleaseTargets();
    }

    // A present binding set references the ALBEDO wrap and the direct-term image of one slot, so it
    // goes with them; the next frame builds a set for the new wraps. Every caller runs after a
    // device wait - the destructor and OnSwapchainDestroy wait themselves, and the swapchain's
    // re-create waits before it notifies - so the handles can be dropped directly instead of
    // through the retire queue.
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        presentBindingSets[i] = nullptr;
        presentAlbedoTextures[i] = nullptr;
        presentDirectSetTextures[i] = nullptr;
        presentDirectTextures[i] = nullptr;
        presentDirectImageHandles[i] = 0;
    }

    swapchainFramebuffers.clear();
    swapchainTextures.clear();
}
