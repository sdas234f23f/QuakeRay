/*
* Copyright (c) 2026 Sultim Tsyrendashiev
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
#include <string>
#include <unordered_map>

#include <nvrhi/vulkan.h>

#include "../Common.h"
#include "../RasterizedDataCollector.h"

namespace vkpt
{

namespace rhi
{
class RhiFrameContext;
class RhiTextureTable;
}

// The A5.1 first cut of the 2D UI on the RHI path: the game's `SWAPCHAIN` overlay list - the HUD,
// the console, the menus, the text, the damage tint and the screen-space effects - drawn into the
// compose's upscaled image 29 (UPSCALED_PING, or its PONG partner 30 on the frame slot the engine's
// swap permutation assigns it; `GetUpscaledTexture` resolves that) after the TAAU wrote it and
// before the present samples it. It is the RHI counterpart of `SwapchainPass` (SwapchainPass.cpp:
// 34-45, :135-183) and of the legacy `Rasterizer::DrawToSwapchain` call (Rasterizer.cpp:299-335;
// VulkanDevice.cpp:1198-1209).
//
// What it reproduces, exactly as the legacy pass has it:
//  - Shaders: `VertDefault` = RsRasterizer.vert.spv and `FragSwapchain` = RsSwapchain.frag.spv, the
//    pair `SwapchainPass` hands to its RasterizerPipelines (SwapchainPass.cpp:37-45;
//    ShaderManager.cpp:61-62). The fragment half is code-identical to RsSky.frag, so the pipeline
//    layout is the one RhiSkyPass already proves: the bindless texture table (descriptor set 0) plus
//    one push-constant layout, no other descriptors. The vertex half is the shared rasterizer
//    vertex shader, so its input layout is the RgVertex layout RhiSkyPass builds.
//  - Target: one colour attachment in the upscaled B10G11R11 format, no depth, loadOp LOAD (NVRHI's
//    attachment loadOp is always LOAD, vulkan-graphics.cpp:80), exactly the single-attachment
//    legacy swapchain render pass (SwapchainPass.cpp:135-183). The target arrives as an NVRHI
//    render-target wrap; the caller passes the compose's `GetUpscaledTexture(frameIndex)`.
//  - Per-draw state: the alpha-test / blend / line-list key of RasterizerPipelines::
//    ConvertToStateFlags and one pipeline per key, created lazily and cached, because NVRHI does
//    not deduplicate pipelines.
//  - Push constant: one 88-byte block, the legacy RasterizedPushConst (Rasterizer.cpp:33-65),
//    carrying the per-draw model-view-projection (the draw's own viewProj when it has one, the
//    frame's plain view x projection otherwise, Rasterizer.cpp:299-312) plus the colour and the two
//    texture indices.
//  - Per draw: the viewport of the DrawInfo (or the full target), the scissor kept at the full
//    render area, one pipeline switch if the key changed, the push block and the indexed or
//    non-indexed draw - the exact loop of Rasterizer::Draw (Rasterizer.cpp:356-420).
//
// What it needs from the host:
//  - the shared RHI texture table and the RHI frame context (the same objects the frame skeleton
//    uses), and the engine shader folder path, in Create();
//  - the NVRHI buffers that hold the frame's RgVertex/index data in the collector's formats, once,
//    through SetGeometryBuffers() - the same wraps the sky pass receives (VulkanDevice_Init.cpp:
//    562-607). The collector's ranges (DrawInfo::firstVertex/firstIndex) are relative to these
//    buffers;
//  - one Render() call per frame, on the frame context's open list of `frameIndex`, after
//    `RhiRtComposePass::RenderTaaU` and before the present, with the compose's upscaled wrap, the
//    upscaled extent, the frame's `RasterizedDataCollector::GetSwapchainDrawInfos()` array, the
//    frame's view/projection (the uniform bytes the legacy call receives, VulkanDevice.cpp:
//    1205-1206) and the instance-wide `rasterizedVertexColorGamma` flag.
//
// The Y convention of the target, the one thing this port had to decide (a5c_overlay_recon.md 6.2):
//  - The legacy `DrawToSwapchain` feeds the game's RgViewport straight to `vkCmdSetViewport`
//    (Rasterizer.cpp:436-446): a Vulkan viewport with a positive height, whose `y` is the rect's
//    top edge, and the game's ortho maps the top of its canvas to NDC y = -1 (gl_draw.c:1015-1040,
//    :1047-1058). The UI therefore lands with the target's row `y` at the canvas top, exactly like
//    the engine's other passes put the view's first row in the image's first row.
//  - NVRHI takes viewports in the D3D convention and the Vulkan backend converts them with
//    `VKViewportWithDXCoords` (vulkan-graphics.cpp:528-531): the emitted VkViewport has a negative
//    height, so NDC y = -1 maps to the rectangle's *bottom*. Reinterpreting the legacy viewport
//    the way RhiSkyPass does (`minY = v.y`, `maxY = v.y + v.height`) would draw the UI vertically
//    mirrored relative to the legacy.
//  - The pass's target is the traced image family: the present mirrors the sample coordinate for
//    traced frames (exposure.y, NvrhiFrameSkeleton.cpp:605-613, RhiPresent.frag.hlsl:82-88), so
//    image row 0 is the screen's top row, exactly as in the legacy blit. To reproduce the legacy
//    rasterization the pass passes the *inverted* NVRHI rectangle (`minY = v.y + v.height`,
//    `maxY = v.y`), which `VKViewportWithDXCoords` turns back into the legacy's own positive-height
//    VkViewport and therefore into the same pixels as the legacy path. This is deliberate and
//    Vulkan-backend-specific; the first A5.1 gate has to confirm the HUD/console orientation, and
//    the alternative (flipping the pushed matrix and keeping a D3D-convention viewport) is noted
//    for A5.0.
//
// The target-state contract with the present: the compose's wrap keeps no state between command
// lists (RhiTextureSource.h), and the engine's framebuffer images rest in VK_IMAGE_LAYOUT_GENERAL,
// NVRHI's UnorderedAccess. Render announces that state before the framebuffer use, the use moves
// the image to COLOR_ATTACHMENT_OPTIMAL, and Render restores UnorderedAccess at the end - the
// pattern RhiSkyPass::RenderWorld applies to SCREEN_EMISSION. The present's
// `beginTrackingTextureState(upscaled, UnorderedAccess)` (NvrhiFrameSkeleton.cpp:624-646 announces
// ALBEDO the same way) is then truthful and the next frame's TAAU UAV write finds GENERAL, as it
// expects.
//
// What the host has to do around it:
//  - record the call only while the frame wants rasterization (`!drawInfo.disableRasterization`,
//    VulkanDevice.cpp:1199; false in every shipped configuration, gl_vidsdl.c:2132-2133);
//  - call ReleaseTargets() before Framebuffers::PrepareForSize destroys the engine framebuffer
//    images, at the same point RhiRtComposePass::ReleaseTargets() runs (the skeleton's
//    OnSwapchainDestroy, NvrhiFrameSkeleton.cpp:1028-1079) - the per-slot framebuffers reference
//    the compose's wraps of image 29 and must not outlive them.
//
// The pass is a no-op until Create succeeded and while an input is missing (no target, no geometry
// buffers, no camera, an empty draw list); every early return is quiet after the first warning. It
// is not thread-safe: the framebuffer Render uses is the one the same call selected for
// `frameIndex`, which is the engine's single-threaded per-slot frame model (RhiFrameContext).
//
// A5.0 note: the push-constant struct, the state-key mirror and the lazy pipeline cache below are a
// deliberate second copy of RhiSkyPass's raster-draw machinery - this first cut stays self-contained
// instead of refactoring the sky pass. A5.0's shared raster-draw helper is expected to fold the two
// copies together.
class RhiUiPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiUiPass();
    ~RhiUiPass();

    RhiUiPass(const RhiUiPass &other) = delete;
    RhiUiPass(RhiUiPass &&other) noexcept = delete;
    RhiUiPass &operator=(const RhiUiPass &other) = delete;
    RhiUiPass &operator=(RhiUiPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pTextureTable' is the host's shared RHI texture table
    // (RHI/RhiTextureTable.h), bound as descriptor set 0; 'pFrameContext' is the host's RHI frame
    // context (RHI/RhiFrameContext.h), which owns the retire queues the framebuffers of replaced
    // targets go through. None of the three is owned, they have to outlive this object, and a null
    // one makes Create fail. 'pShaderFolderPath' is the folder ShaderManager loads the engine blobs
    // from, with the trailing separator; RsRasterizer.vert.spv and RsSwapchain.frag.spv are loaded
    // from it. Returns false and leaves the pass unusable if a shader, the input layout or the
    // push-constant layout cannot be created; the host logs that through 'pfnPrint'.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiTextureTable *pTextureTable,
                rhi::RhiFrameContext *pFrameContext,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // The buffers every draw is bound to, in the collector's own formats: RgVertex records of
    // RasterizedDataCollector::GetVertexStride() bytes and R32_UINT indices (Rasterizer.cpp:
    // 385-386). The pass neither owns nor fills them - the caller has to make sure that the
    // frame's geometry is in them when Render records and that they stay valid until the
    // submission finishes. Render is a no-op until both are set.
    //
    // Wiring hazard the host has to settle: the legacy `CopyFromStaging` that fills these buffers
    // is recorded on the legacy command buffer (VulkanDevice.cpp:1471) which the engine submits
    // *after* the RHI list (VulkanDevice.cpp:1346-1353), so an RHI list of frame N reads the copy
    // of frame N-1. The sky tolerates that (static level geometry, VulkanDevice_Init.cpp:552-561);
    // the UI does not, because the HUD is re-uploaded every frame and the ranges move with it. The
    // host has to record the frame's own copy on the RHI list before Render (or bind buffers it
    // copied there), otherwise the overlay shows the previous frame's - possibly mismatched -
    // vertex data.
    void SetGeometryBuffers(nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer);

    // One call per frame, on the frame context's open command list of 'frameIndex', after
    // `RhiRtComposePass::RenderTaaU` of the same slot and before the present. It (re)builds the
    // slot's one-attachment framebuffer when 'pTarget' or its extent changed, announces the
    // target's real state, records the draws and restores the target to UnorderedAccess.
    //
    // 'pTarget' is the engine image the UI is drawn into: the compose's wrap of the upscaled image
    // (`RhiRtComposePass::GetUpscaledTexture(frameIndex)`, UPSCALED_PING 29 or the PONG partner the
    // slot's swap permutation resolves to). The pass borrows it - it never releases or re-wraps it,
    // and the caller must keep the handle valid until the submission finishes. The wrap must be an
    // engine-convention image (row 0 = the first row of the view/display; see the class comment on
    // the Y convention) and its extent must equal 'targetWidth' x 'targetHeight' - the upscaled size
    // `RenderTaaU` dispatched over. A null target, a zero extent or a mismatch with the wrap's own
    // desc skips the call.
    //
    // 'pDraws'/'drawCount' are the frame's `RasterizedDataCollector::GetSwapchainDrawInfos()`
    // (Rasterizer::GetDataCollector). An entry is skipped exactly when the legacy never draws it:
    // a `DEPTH_TEST`/`DEPTH_WRITE` entry, which the collector rejects at upload for this stream
    // (RasterizedDataCollector.cpp:153-167) and which could not be drawn by the legacy swapchain
    // pass (its render pass has no depth attachment), and an entry with neither vertices nor
    // indices, which the legacy loop turns into a zero-vertex draw that rasterizes nothing.
    //
    // 'pView'/'pProj' are the same two column-major matrices the legacy call receives
    // (VulkanDevice.cpp:1205-1206): the frame's view and projection. They build the fallback
    // view-projection `view x proj` for the draws that carry no viewProj of their own, exactly as
    // DrawToSwapchain builds it (Rasterizer.cpp:311-312). Today every SWAPCHAIN upload passes its
    // own ortho (gl_draw.c:556-1006, pr_ext.c:4378-4618), so the fallback is the parity value, not
    // the common path. A null matrix skips the call.
    //
    // 'applyVertexColorGamma' is the per-frame value of the instance's rasterizedVertexColorGamma
    // (RasterPass.cpp:67); it is baked into the pipeline's vertex spec constant, so it joins the
    // pipeline key.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                nvrhi::ITexture *pTarget,
                uint32_t targetWidth,
                uint32_t targetHeight,
                const RasterizedDataCollector::DrawInfo *pDraws,
                uint32_t drawCount,
                const float *pView,
                const float *pProj,
                bool applyVertexColorGamma);

    // Drops the per-slot framebuffers. The caller has to call it before the engine destroys the
    // images they reference (the Framebuffers::PrepareForSize path), next to the compose pass's
    // ReleaseTargets(). The next Render rebuilds them from the wraps it is given, so the pass
    // survives a resize without re-creation. Releases go through the frame context's retire queue
    // when one exists; the destructor drops directly, after a device idle.
    void ReleaseTargets();

private:
    struct Target
    {
        // The wrap Render was last given for this slot; a changed pointer means the compose pass
        // re-wrapped the image (a framebuffer re-create, a size change or the ping-pong flip), so
        // the framebuffer over it has to follow. The pass does not own the wrap.
        nvrhi::ITexture *target = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        nvrhi::FramebufferHandle framebuffer;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    void ReleaseTarget(Target &target);
    void ReleasePipelineCache();

    nvrhi::IGraphicsPipeline *GetPipeline(uint32_t stateFlags, bool applyVertexColorGamma);
    nvrhi::GraphicsPipelineHandle CreatePipeline(uint32_t stateFlags, bool applyVertexColorGamma);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    nvrhi::ShaderHandle vertexShader;
    nvrhi::ShaderHandle pixelShader;
    nvrhi::InputLayoutHandle inputLayout;

    // The pipeline's second layout: one 88-byte push-constant item, no descriptors. The shader's
    // set 0 is the texture table.
    nvrhi::BindingLayoutHandle pushConstantLayout;

    // The host's table and frame model; not owned, both outlive this object. The table provides the
    // bindless set and the first-use tracking of the engine textures it wrapped; the frame context
    // owns the retire queues.
    rhi::RhiTextureTable *textureTable = nullptr;
    rhi::RhiFrameContext *frameContext = nullptr;

    // The geometry the draws bind; not owned, the caller keeps them alive.
    nvrhi::BufferHandle vertexBuffer;
    nvrhi::BufferHandle indexBuffer;

    // One entry per engine frame slot: the compose's upscaled image is a swapped engine image, so
    // the slot's wrap is not shared with the other slot's and the framebuffer over it follows.
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // Valid only for the format the pipelines were created with; the upscaled images are
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32 (ShaderCommonCFramebuf.cpp), which the wrapped texture's
    // own format reports as nvrhi::Format::R11G11B10_FLOAT.
    nvrhi::Format pipelineColorFormat = nvrhi::Format::UNKNOWN;
    std::unordered_map<uint32_t, nvrhi::GraphicsPipelineHandle> pipelines;

    bool warnedMissingTarget = false;
    bool warnedMissingGeometry = false;
    bool warnedMissingCamera = false;
    bool warnedDepthState = false;
    bool warnedFailedPipeline = false;

    bool created = false;
};

}
