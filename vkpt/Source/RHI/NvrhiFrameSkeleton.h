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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nvrhi/vulkan.h>

#include "../Common.h"
#include "../ISwapchainDependency.h"
#include "../RasterizedDataCollector.h"

namespace vkpt
{

class Framebuffers;
class GlobalUniform;
class RhiDebugTracePass;
class RhiRtPrimaryPass;
class RhiSkyPass;
class Swapchain;
class Tonemapping;

namespace rhi
{
class RhiAccelStructs;
class RhiFrameContext;
class RhiTextureTable;
}

// The RHI frame skeleton: the frame is recorded and submitted through the RHI
// layer instead of the Vulkan command buffers of the renderer. It drives the
// rasterized sky - the first real engine pass on the RHI path (RHI/RhiSkyPass.h)
// - and presents the result: the sky draws into the engine's ALBEDO image and a
// fullscreen triangle samples that ALBEDO into the swapchain image that was
// acquired for this frame.
// The pass uses the RHI resource, binding and pipeline helpers
// (RhiResources.h, RhiPipeline.h) - a volatile constant buffer that carries the
// present's exposure and one sampler - so the plumbing of stage A2 is exercised
// while the drawn image is the engine's own sky.
// The sky pass itself is owned here, because every input its Create() needs is
// one this class already receives; the host hands over the collector's geometry
// buffers once through GetSkyPass().
// On top of the sky, the same pass also draws the rasterized world into the same
// ALBEDO and depth: once the engine's framebuffers exist, this class wraps the
// engine's global uniform and tonemapping buffers, calls RhiSkyPass::CreateWorld
// with the engine's image handles and, every frame, records the world draw list
// with a stand-in avgLuminance and a fresh copy of the uniform (see Render and
// PrepareWorld).
class NvrhiFrameSkeleton final : public ISwapchainDependency
{
public:
    using PrintFunction = std::function<void(const char *)>;

    // The per-frame inputs of the rasterized sky, filled by the host from the same values the
    // legacy Rasterizer::DrawSkyToAlbedo call receives (VulkanDevice.cpp:748-756): 'framebuffers'
    // is the engine framebuffer set ALBEDO lives in, 'width'/'height' are the render resolution
    // ALBEDO is sized to, and 'view', 'projection', 'jitter' and 'skyViewerPos' are passed through
    // to RhiSkyPass::SetSkyCamera unchanged. 'draws'/'drawCount' are the frame's
    // RasterizedDataCollector::GetSkyDrawInfos() (Rasterizer::GetDataCollector), and
    // 'applyVertexColorGamma' is the instance-wide boolean of the same name (RasterPass.cpp:67).
    //
    // The rasterized world sub-pass on top of the sky reads three more frame inputs:
    //  - 'worldDraws'/'worldDrawCount', the frame's RasterizedDataCollector::GetRasterDrawInfos()
    //    (Rasterizer::GetDataCollector) - the list the legacy world draw consumes
    //    (VulkanDevice.cpp:1071-1082);
    //  - 'uniform', the engine's global uniform: the world shader's set 1, and the source of the
    //    bytes the skeleton writes into the uniform wrap every frame (the engine's own
    //    GlobalUniform::Upload never runs under `rhiframe`);
    //  - 'tonemapping', the engine's tonemapping object: the source of the per-slot set 2 buffers
    //    and the carrier of the avgLuminance stand-in the exposure chain would normally fill.
    // All three are optional: when one is null, or the world could not be created, the skeleton
    // records the sky and the present exactly as it did before the world existed.
    struct SkyFrameInputs
    {
        const Framebuffers *framebuffers = nullptr;
        const RasterizedDataCollector::DrawInfo *draws = nullptr;
        uint32_t drawCount = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        float view[16] = {};
        float projection[16] = {};
        float jitter[2] = {};
        float skyViewerPos[3] = {};
        bool applyVertexColorGamma = false;

        // -- the rasterized world sub-pass --
        const RasterizedDataCollector::DrawInfo *worldDraws = nullptr;
        uint32_t worldDrawCount = 0;
        GlobalUniform *uniform = nullptr;
        Tonemapping *tonemapping = nullptr;

        // -- the acceleration-structure stream --

        // -- the acceleration-structure stream --

        // The frame inputs RhiAccelStructs::BuildTopLevel needs to synthesise the instance list and
        // build the slot's TLAS from the module's own BLAS: the uniform's world-ray cull mask, the
        // instance-wide sky flag and the draw info's "disable ray-traced geometry" flag - the same
        // three values the host passes to the engine's own TLAS preparation (VulkanDevice.cpp:1285).
        uint32_t rayCullMaskWorld = 0;
        bool allowGeometryWithSkyFlag = false;
        bool disableRayTracedGeometry = false;
    };

    // The frame mode of the whole run, chosen by the host from the library config flags:
    //  - Rasterized:   'rhirt' and 'rhitrace' off - the raster sky and world sub-passes draw into
    //                  ALBEDO and the present samples it;
    //  - DebugTrace:   'rhitrace' - the A3.1 debug trace of the acceleration structures (the
    //                  id-coloured image) replaces the raster sub-passes;
    //  - PrimaryTrace: 'rhirt' - the real primary-visibility ray-tracing pass of A4.1 fills the
    //                  engine's checkerboard G-buffer (ALBEDO included) and the present samples it.
    // The host resolves 'rhirt' over 'rhitrace' when both are set.
    enum class FrameMode
    {
        Rasterized,
        DebugTrace,
        PrimaryTrace,
    };

    // 'pTextureTable' is the shared RHI texture table of the host (RHI/RhiTextureTable.h), bound by
    // the sky pass as descriptor set 0; the table owns every wrapped engine texture and its
    // samplers. Not owned, it has to outlive this object, and a null pointer makes the pass
    // unavailable.
    // 'pFrameContext' is the host's RHI frame context (RHI/RhiFrameContext.h): it owns the per-slot
    // command lists and the retire queues, and Render records through it. Not owned, it has to
    // outlive this object, and a null pointer makes the pass unavailable.
    // 'pAccelStructs' is the host's acceleration-structure stream (rhi::RhiAccelStructs,
    // RHI/RhiAccelStructs.h): Render records its static and per-frame builds on the same open list,
    // in every mode, before the sky/trace and the present. Not owned; a null or not-created one
    // makes the skeleton unavailable.
    // 'pDebugTracePass' is the host's debug ray-tracing pass (RhiDebugTracePass,
    // RHI/RhiDebugTracePass.h): when 'mode' is DebugTrace, Render drives it - into the same ALBEDO
    // the raster chain would draw into - instead of the raster sky/world sub-passes. Not owned; a
    // null or not-created one with that mode makes the skeleton unavailable.
    // 'pRtPrimaryPass' is the host's primary-visibility ray-tracing pass (RhiRtPrimaryPass,
    // RHI/RhiRtPrimaryPass.h): when 'mode' is PrimaryTrace, Render drives it - into the engine's
    // checkerboard G-buffer images, ALBEDO included. Not owned; a null or not-created one with
    // that mode makes the skeleton unavailable.
    explicit NvrhiFrameSkeleton(nvrhi::IDevice *pDevice,
                                const Swapchain *pSwapchain,
                                const char *pShaderFolderPath,
                                rhi::RhiTextureTable *pTextureTable,
                                rhi::RhiFrameContext *pFrameContext,
                                rhi::RhiAccelStructs *pAccelStructs,
                                RhiDebugTracePass *pDebugTracePass,
                                RhiRtPrimaryPass *pRtPrimaryPass,
                                FrameMode mode,
                                PrintFunction pfnPrint);
    ~NvrhiFrameSkeleton() override;

    NvrhiFrameSkeleton(const NvrhiFrameSkeleton &other) = delete;
    NvrhiFrameSkeleton(NvrhiFrameSkeleton &&other) noexcept = delete;
    NvrhiFrameSkeleton &operator=(const NvrhiFrameSkeleton &other) = delete;
    NvrhiFrameSkeleton &operator=(NvrhiFrameSkeleton &&other) noexcept = delete;

    // Called by the swapchain when its images are (re)created or destroyed.
    void OnSwapchainCreate(const Swapchain *pSwapchain) override;
    void OnSwapchainDestroy() override;

    // Records the frame for the image the swapchain acquired into the frame context's slot
    // 'frameIndex' and submits it on the RHI graphics queue; the context owns that slot's command
    // list and its submission. The acceleration-structure stream (RhiAccelStructs) is recorded
    // first - its builds, and the patch of the uniform's per-instance geometry offsets the traced
    // modes need - then the sky pass's Prepare (which prepares the ALBEDO wrap the present samples
    // and, in the traced modes, the announcement the trace's own ALBEDO wrap relies on) and then
    // the frame splits by frameMode:
    //  - Rasterized: the sky pass's SetSkyCamera/Render run, the rasterized world sub-pass (once it
    //    could be created) draws into the same target with the engine's uniform and the
    //    avgLuminance stand-in, and the present samples the ALBEDO wrap of the same slot into the
    //    swapchain image of the acquired index;
    //  - DebugTrace: the debug pass traces one primary ray per pixel over the acceleration
    //    structures into the slot's ALBEDO and the same present follows;
    //  - PrimaryTrace: the engine's primary-visibility raygen writes the slot's checkerboard
    //    G-buffer (ALBEDO included) over the same acceleration structures and the same present
    //    follows.
    // The raster sky/world sub-passes are not recorded in the two traced modes.
    // 'semaphoreToWait' is the semaphore the swapchain signals on acquire, 'semaphoreToSignal' is
    // the one the presentation engine waits on. Returns false if the pass is unavailable.
    bool Render(const Swapchain *pSwapchain, uint32_t frameIndex, const SkyFrameInputs &sky,
                VkSemaphore semaphoreToWait, VkSemaphore semaphoreToSignal);

    // True if the pass cannot be used at all, so that the caller can avoid
    // taking the frame apart on every frame.
    bool IsUnavailable() const;

    // The rasterized sky pass this skeleton drives: the host creates the geometry buffers over the
    // collector's VkBuffers once and hands them over with SetGeometryBuffers(). Null when the pass
    // could not be created (the skeleton is unavailable then).
    RhiSkyPass *GetSkyPass() const { return skyPass.get(); }

private:
    static nvrhi::Format ConvertSurfaceFormat(VkFormat format);

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Creates the present's binding layout, constant buffer and sampler. The binding set is per
    // frame slot and is built by PreparePresentBindingSet once the slot's ALBEDO wrap is known.
    bool CreatePassResources();

    // Creates or reuses the present binding set of 'frameIndex' for the slot's ALBEDO wrap: a
    // regular NVRHI binding set holds one texture, and the pass replaces that wrap when the engine
    // re-creates its framebuffers, so the set follows it.
    bool PreparePresentBindingSet(uint32_t frameIndex, nvrhi::ITexture *albedo);

    // The one-time world setup, called by Render on the first frame that got past the sky's
    // Prepare: wraps the engine's uniform and tonemapping buffers (once - the engine never
    // re-creates either) and hands them, together with this frame's engine image handles, to
    // RhiSkyPass::CreateWorld. It has to run this late because the Framebuffers accessors read the
    // image vector only Framebuffers::PrepareForSize fills, which the host calls per frame right
    // before Render - calling CreateWorld from the constructor would resolve empty handles.
    // Returns true once the world is created; a frame that simply has no world inputs yet is not a
    // failure and leaves the world to be attempted again, while a wrap or CreateWorld failure sets
    // worldCreationFailed (every failure mode there is permanent).
    bool PrepareWorld(uint32_t frameIndex, const SkyFrameInputs &sky);

    bool CreatePipeline(nvrhi::Format colorFormat);
    bool CreateSwapchainResources(const Swapchain *pSwapchain);
    void DestroySwapchainResources();

    nvrhi::vulkan::IDevice *device;
    PrintFunction print;
    std::string shaderFolderPath;

    nvrhi::ShaderHandle vertexShader;
    nvrhi::ShaderHandle pixelShader;

    // The rasterized sky pass of the RHI path; created in the constructor from the same device,
    // table, frame context and shader folder, and driven by Render below.
    std::unique_ptr<RhiSkyPass> skyPass;

    // The host's acceleration-structure stream (rhi::RhiAccelStructs, RHI/RhiAccelStructs.h):
    // Render records its static and per-frame builds on the slot's list, in both modes, before the
    // sky/trace and the present. Not owned; the host creates it next to this skeleton and keeps it
    // alive until after the skeleton (VulkanDevice_Init), so it outlives every Render.
    rhi::RhiAccelStructs *accelStructs = nullptr;

    // The host's debug ray-tracing pass (RhiDebugTracePass, RHI/RhiDebugTracePass.h), driven
    // instead of the raster sky/world chain when frameMode is DebugTrace. Not owned; null when the
    // host's 'rhitrace' flag is off.
    RhiDebugTracePass *debugTracePass = nullptr;

    // The host's primary-visibility ray-tracing pass (RhiRtPrimaryPass, RHI/RhiRtPrimaryPass.h),
    // driven instead of the raster sky/world chain when frameMode is PrimaryTrace. Not owned; null
    // when the host's 'rhirt' flag is off.
    RhiRtPrimaryPass *rtPrimaryPass = nullptr;

    // The frame mode of the whole run: which chain Render records into ALBEDO. The host picks it
    // once from 'rhirt'/'rhitrace' (VulkanDevice_Init.cpp) and it does not change while the
    // skeleton lives.
    FrameMode frameMode = FrameMode::Rasterized;

    // Descriptor set 0 of the present: the constant buffer (binding 256), the ALBEDO texture
    // (binding 0) and its sampler (binding 128), the numbers the shader declares with
    // [[vk::binding(...)]]; RhiPresent.frag declares no other set. The texture moves into one of the
    // per-slot binding sets below, so only the layout, the buffer and the sampler live here.
    nvrhi::BindingLayoutHandle bindingLayout;
    nvrhi::BufferHandle presentParamsBuffer;
    nvrhi::SamplerHandle presentSampler;

    // One present binding set per frame slot: it references that slot's ALBEDO wrap (ALBEDO is a
    // ping-pong engine image), so it cannot be shared across slots, and it is rebuilt when the pass
    // re-wraps the slot (a framebuffer re-create). presentAlbedoTextures is what the rebuild
    // compares against, and it keeps the wrap the set references alive.
    nvrhi::BindingSetHandle presentBindingSets[MAX_FRAMES_IN_FLIGHT] = {};
    nvrhi::TextureHandle presentAlbedoTextures[MAX_FRAMES_IN_FLIGHT];

    // The shared RHI texture table (RhiTextureTable.h, owned by the host): the sky pass binds it as
    // descriptor set 0 and its slot 0 holds the engine's empty texture. Render also asks it to
    // declare the first-use state of the engine textures wrapped since the last frame. Not owned.
    rhi::RhiTextureTable *textureTable = nullptr;

    // The host's RHI frame context (RhiFrameContext.h, owned by the host): the per-slot command
    // lists and retire queues of the RHI frame model. Render records through it, so it owns the
    // only lists this pass uses and the pass itself has no long-lived one. Not owned.
    rhi::RhiFrameContext *frameContext = nullptr;

    // -- the rasterized world sub-pass --

    // The world sub-pass's buffer wraps, created once by PrepareWorld on the first frame whose
    // inputs allow it. The engine's global uniform and tonemapping buffers are created once and
    // never re-created - only their contents change - so one wrap per buffer serves the whole run
    // and the RhiSkyPass::ReleaseTargets re-wrap contract does not apply to them. RhiSkyPass
    // borrows them through CreateWorld; the handles stay here, where they are released with the
    // skeleton, before the engine destroys the buffers.
    nvrhi::BufferHandle worldUniformBuffer;
    nvrhi::BufferHandle worldTonemappingBuffers[MAX_FRAMES_IN_FLIGHT];

    // True once the wraps above exist, whether or not CreateWorld accepted them.
    bool worldBuffersWrapped = false;

    // Set after a failed world creation. Every failure CreateWorld and the wraps can report is
    // permanent (a missing shader, a failed layout or binding set, a buffer in the wrong shape),
    // so Render does not retry it - and the pass's per-call warnings stay one-shot.
    bool worldCreationFailed = false;

    // Frames left until the one-time fallback-slot log; 0 after it has been printed.
    uint32_t framesUntilFallbackLog = 300;

    // Set after the one-time warning that there is no ALBEDO wrap to present.
    bool warnedMissingAlbedo = false;

    // Valid only for the format the pipeline was created with: the swapchain can
    // switch between its formats when it is recreated.
    nvrhi::GraphicsPipelineHandle pipeline;
    nvrhi::Format pipelineColorFormat = nvrhi::Format::UNKNOWN;

    // One entry per swapchain image, indexed by the acquired image index.
    std::vector<nvrhi::TextureHandle> swapchainTextures;
    std::vector<nvrhi::FramebufferHandle> swapchainFramebuffers;

    // Set when the pass cannot be used at all: the caller then has to keep the
    // old renderer instead of presenting an empty image.
    bool unavailable;
};

}
