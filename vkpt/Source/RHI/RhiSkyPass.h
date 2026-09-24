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
#include <tuple>
#include <unordered_map>

#include <nvrhi/vulkan.h>

#include "../Common.h"
#include "../RasterizedDataCollector.h"

namespace vkpt
{

class Framebuffers;

namespace rhi
{
class RhiFrameContext;
class RhiTextureTable;
}

// The first real engine pass on the RHI path: the rasterized sky, i.e. the `RsRasterizer.vert` +
// `RsSky.frag` pair of RasterPass's sky sub-pass (RasterPass.cpp:56-77), recorded through the RHI
// layer into the engine's ALBEDO image instead of into the legacy sky render pass.
//
// What it reproduces, exactly as the legacy pass has it:
//  - Pipeline layout: the shared bindless texture table as the first layout, so it lands at
//    descriptor set 0 (the shader's DESC_SET_TEXTURES) in NVRHI's legacy binding mode, plus one
//    push-constant layout. No other set is declared by RsSky.frag.
//  - Push constants: one 88-byte block, the legacy range (Rasterizer.cpp:485-489); the shader block
//    declares 92 bytes - the last member is declared but never read, and the legacy host never
//    writes it (recon 2.3), so mirroring the legacy value keeps the two renderers byte-identical.
//  - Spec constants: the vertex stage's applyVertexColorGamma and the fragment stage's alphaTest,
//    both SpecId 0 and 4 bytes (RasterizerPipelines.cpp:315-343).
//  - Vertex input: binding 0, the collector's stride, locations in the declaration order that
//    RasterizedDataCollector::GetVertexLayout produces.
//  - Per-draw state: the alpha-test / blend / depth / line-list key of
//    RasterizerPipelines::ConvertToStateFlags, and one pipeline per key created lazily and cached,
//    because NVRHI does not deduplicate pipelines (recon 4.7).
//  - The target: FB_IMAGE_INDEX_ALBEDO plus a depth image this pass owns, with the depth cleared
//    before the first draw - NVRHI's loadOp is always LOAD (vulkan-graphics.cpp:80), so the clear
//    the legacy sky render pass does (RasterPass.cpp:355) is an explicit command here.
//
// What it needs from the host:
//  - the shared RHI texture table and the RHI frame context (the same objects the frame skeleton
//    uses), and the engine shader folder path, in Create();
//  - the NVRHI buffers that hold the frame's RgVertex/index data in the collector's formats, once,
//    through SetGeometryBuffers();
//  - one Prepare() per frame, before Render(), with the frame's index, the engine framebuffers and
//    the render resolution - it wraps that frame's ALBEDO image and (re)creates the depth image and
//    the framebuffer over both when the image or the size changed;
//  - SetSkyCamera() once per frame with the same camera arguments Rasterizer::DrawSkyToAlbedo
//    receives, and the frame's draw list through Render().
//
// The world sub-pass (CreateWorld + RenderWorld) draws into the same target and depth after the
// sky, so the present below keeps sampling what it already samples. It reuses the same state
// encoder, push-constant block, input layout and geometry buffers and adds what only the world
// shader declares: the engine's global uniform (set 1), the tonemapping buffer (set 2), a set-3
// hole, the partial framebuffers layout with the binding-25 storage image (set 4) and a second
// colour attachment (FB_IMAGE_INDEX_SCREEN_EMISSION) - and it depth-tests against the sky's own
// depth (clearing it only when the sky had no draws and left the clear undone). The legacy world
// pass loads the traced depth instead (RasterPass.cpp:99, :267); there is no traced depth under
// `rhiframe`, which is this first cut's recorded deviation.
//
// The pass is a no-op until all of that has happened: it never creates a target on its own, and
// every method that could record something checks its inputs first. It is not thread-safe: the
// target that Render uses is the one the last Prepare selected, which is the engine's
// single-threaded per-slot frame model (RhiFrameContext).
class RhiSkyPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiSkyPass();
    ~RhiSkyPass();

    RhiSkyPass(const RhiSkyPass &other) = delete;
    RhiSkyPass(RhiSkyPass &&other) noexcept = delete;
    RhiSkyPass &operator=(const RhiSkyPass &other) = delete;
    RhiSkyPass &operator=(RhiSkyPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pTextureTable' is the host's shared RHI texture table
    // (RHI/RhiTextureTable.h), bound as descriptor set 0; 'pFrameContext' is the host's RHI frame
    // context (RHI/RhiFrameContext.h), which owns the retire queues the wrapped ALBEDO image, the
    // depth image and the pipelines of replaced targets go through. None of the three is owned,
    // they have to outlive this object, and a null one makes Create fail. 'pShaderFolderPath' is
    // the folder ShaderManager loads the engine blobs from, with the trailing separator;
    // RsRasterizer.vert.spv and RsSky.frag.spv are loaded from it, and RsWorld.frag.spv when
    // CreateWorld runs. Returns false and leaves the
    // pass unusable if a shader, the input layout or the push-constant layout cannot be created;
    // the host logs that through 'pfnPrint'.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiTextureTable *pTextureTable,
                rhi::RhiFrameContext *pFrameContext,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    // Enables the rasterized world sub-pass: the `RsRasterizer.vert` + `RsWorld.frag` pair of
    // RasterPass's world sub-pass (RasterPass.cpp:52-77), drawn into the sky's ALBEDO and depth plus
    // a second colour attachment and a storage image, after Create() succeeded.
    //
    // 'pUniformBuffer' is the world shader's set 1 binding 0, the engine's ShGlobalUniform
    // (RsWorld.frag reads member 11, renderWidth, at byte 644). The host wraps
    // GlobalUniform::GetBuffer() through createHandleForNativeBuffer; the wrap must not be volatile,
    // because the pass binds a static ConstantBuffer item.
    //
    // 'pTonemappingBuffers' are the world shader's set 2 binding 0, one per frame slot, the engine's
    // ShTonemapping buffers wrapped over Tonemapping::GetBuffer(frameIndex) with
    // structStride = Tonemapping::GetElementSize(): the shader declares the block as
    // StructuredBuffer<ShTonemapping> and the backend asserts a non-zero stride. The exposure chain
    // does not run under `rhiframe`, so nothing fills avgLuminance on the engine side - the host has
    // to supply a stand-in through Tonemapping::SetAvgLuminance (a value of 1 gives the world colour
    // the constant factor 1/9.6, a non-positive one outputs black).
    //
    // 'screenEmission' and 'storageImage' are the engine handles of Framebuffers' colour 1 (index
    // 62) and of the binding-25 image (index 25), exactly what
    // Framebuffers::GetScreenEmissionHandles / GetPrimaryToReflRefrHandles return. Their images are
    // not swapped (Bindings[62] == BindingsSwapped[62], same for 25), so one pair of handles serves
    // every frame slot. Either may carry a null image: the engine has not created its framebuffers
    // yet, or the host chose a first cut without them - the pass then falls back to what a later
    // Prepare reads from the same accessors, and a null storage image becomes the 1x1 RGBA32_UINT
    // dummy described in the .cpp.
    //
    // Returns false and leaves the world disabled if the pass was not created, a buffer is null or
    // not in the shape the bindings need, or a shader or layout cannot be created; the host logs
    // that through the print callback. Calling it again re-reads every argument and rebuilds only
    // what actually changed, so the host may call it once, once per frame or once per engine
    // framebuffer generation.
    bool CreateWorld(nvrhi::IBuffer *pUniformBuffer,
                     nvrhi::IBuffer *const pTonemappingBuffers[MAX_FRAMES_IN_FLIGHT],
                     const std::tuple<VkImage, VkImageView, VkFormat> &screenEmission,
                     const std::tuple<VkImage, VkImageView, VkFormat> &storageImage);

    bool IsCreated() const { return created; }

    // True once CreateWorld succeeded; RenderWorld draws when the frame's target and inputs are
    // there as well.
    bool IsWorldCreated() const { return worldCreated; }

    // The buffers every draw is bound to, in the collector's own formats: RgVertex records of
    // RasterizedDataCollector::GetVertexStride() bytes and R32_UINT indices
    // (Rasterizer.cpp:385-386). The pass neither owns nor fills them - the caller has to make sure
    // that the frame's geometry is in them when Render records and that they stay valid until the
    // submission finishes. The collector's device-local buffers can be wrapped with
    // IDevice::createHandleForNativeBuffer, but they only hold the frame's data after its
    // CopyFromStaging ran, which the legacy renderer submits after the RHI list; a pass-owned copy
    // into an RHI buffer through writeBuffer is the alternative that does not depend on that order.
    // Render is a no-op until both are set.
    void SetGeometryBuffers(nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer);

    // The frame's camera, with exactly the arguments Rasterizer::DrawSkyToAlbedo takes
    // (Rasterizer.cpp:187-194): a column-major view matrix, a column-major projection matrix, the
    // frame's jitter in pixels and the sky viewer position. The pass builds the default sky
    // view-projection the legacy pass builds from them - the sky view with that viewer position,
    // multiplied by the jittered projection (Rasterizer.cpp:176-185, :203-210) - and uses it for
    // every draw whose DrawInfo carries no viewProj of its own. Every sky upload today passes a
    // null viewProjection (Quake/gl_sky.c:1085), so that default is the only transform there is.
    // Call it once per frame before Render; Render warns once and does nothing until it is set.
    // RenderWorld reuses the stored view and projection (it applies the jitter itself and does not
    // move the viewer to the sky position), so the same call carries the camera of both sub-passes.
    void SetSkyCamera(const float *pView, const float *pProj, const float jitter[2],
                      const float skyViewerPos[3]);

    // One call per frame, after the frame context's BeginSlot has opened the slot's command list
    // and before Render. 'frameIndex' is the engine frame slot; it selects the ALBEDO image of that
    // slot (Framebuffers::GetImageHandles resolves the swap permutation, Framebuffers.cpp:33-53)
    // and the target entry that Render will use. 'width'/'height' are the render resolution the
    // sky is drawn at, i.e. the size of the ALBEDO image. Wraps the image on first use or when the
    // slot's image or size changed, announces the state the image is really in to 'pCommandList' -
    // a render-target wrap keeps no state between lists, so the announcement is per list
    // (RhiTextureSource.h:56-70) - and (re)creates the depth image and the framebuffer over the
    // pair. Returns false when the framebuffers have no ALBEDO image yet; Render then does nothing
    // for that frame. Must be followed by SetSkyCamera and Render for the same frame index and on
    // the same open command list.
    //
    // When the world sub-pass is enabled, the same call also resolves the slot's SCREEN_EMISSION and
    // binding-25 images (through the Framebuffers accessors, with the handles CreateWorld received
    // as the fallback), (re)wraps them when the engine recreated an image or the size changed,
    // announces their state and builds the world framebuffer over the slot's ALBEDO, the
    // SCREEN_EMISSION wrap and the slot's depth - so RenderWorld can draw right after Render.
    //
    // Consequence of the announcement: after a Render the wrapped image rests in the
    // render-target layout, not in the engine's VK_IMAGE_LAYOUT_GENERAL. That is invisible to the
    // engine while rhiframe is on - nothing else reads ALBEDO in that path - but anything that
    // touches the image outside this pass has to go through the RHI layer (which announces
    // RenderTarget or ShaderResource) or move the image back itself.
    bool Prepare(nvrhi::ICommandList *pCommandList,
                 uint32_t frameIndex,
                 const Framebuffers &framebuffers,
                 uint32_t width,
                 uint32_t height);

    // Records the sky draws: binds the table as set 0 and the geometry buffers, clears the depth
    // once, and for every DrawInfo sets the viewport and the scissor, switches to the pipeline of
    // its state key, pushes the block of Rasterizer's RasterizedPushConst and draws indexed or
    // non-indexed as the legacy loop does (Rasterizer.cpp:394-420). 'applyVertexColorGamma' is the
    // per-frame value of the instance's rasterizedVertexColorGamma (RasterPass.cpp:67, :76); it is
    // baked into the pipeline's vertex spec constant, so it joins the pipeline key. Does nothing if
    // the pass was not created, has no draws, has no prepared target for the last Prepare's frame,
    // has no geometry buffers, or has no camera. When the world sub-pass is enabled, the caller
    // records RenderWorld with the frame's raster draw list on the same command list right after
    // this call, before the present.
    void Render(nvrhi::ICommandList *pCommandList,
                const RasterizedDataCollector::DrawInfo *pDraws,
                uint32_t drawCount,
                bool applyVertexColorGamma);

    // Records the world draws into the same target and depth Render just used: binds the texture
    // table as set 0, the uniform as set 1, the slot's tonemapping buffer as set 2, the real empty
    // set that fills the set-3 hole, the partial framebuffers set (binding 25) as set 4 and the
    // geometry buffers, and then, per DrawInfo, the same viewport/scissor/pipeline/push-constant/
    // draw sequence Render records (Rasterizer.cpp:394-420). The depth is not cleared again - the
    // sky's Render cleared it and the world has to depth-test against it, which is why the caller
    // records Render first and this call second in the same command list.
    //
    // The default view-projection is the frame's plain view times the jittered projection
    // (Rasterizer.cpp:266-271): unlike the sky, the world does not move the viewer to the sky
    // position, so Matrix::SetNewViewerPosition is not applied. The camera comes from the last
    // SetSkyCamera - the legacy frame passes the same view/projection/jitter to both sub-passes.
    //
    // What the world writes into ALBEDO is what the present samples, so the ALBEDO state contract
    // is unchanged: Prepare announced UnorderedAccess, the sky's framebuffer use moved the image to
    // the render-target layout and the world's use finds it there, so no extra announcement or
    // transition belongs to this call. The second colour target (SCREEN_EMISSION) does get one last
    // transition: the world's framebuffer leaves it in the render-target layout and this call moves
    // it back to UnorderedAccess (= the engine's GENERAL), which is the state the next Prepare
    // announces. Does nothing if the world was not enabled, the frame has no target, no world
    // framebuffer or no geometry/camera.
    void RenderWorld(nvrhi::ICommandList *pCommandList,
                     const RasterizedDataCollector::DrawInfo *pDraws,
                     uint32_t drawCount,
                     bool applyVertexColorGamma);

    // The ALBEDO image of 'frameIndex' as the wrapped RHI texture, or null while that slot has no
    // prepared target (the engine has not created its framebuffers yet, or the last Prepare failed).
    // The present of the host samples this handle: ALBEDO is a ping-pong engine image, so the slot's
    // wrap is not shared with the other slot's, and a re-created engine framebuffer replaces it - the
    // caller must rebuild whatever binding set references it when the returned pointer changes. The
    // handle is owned by this pass; the caller must not release it, and it has to call
    // Prepare() (and Render()) for 'frameIndex' before it samples what the handle points to.
    nvrhi::ITexture *GetAlbedoTexture(uint32_t frameIndex) const;

    // Drops the wrapped ALBEDO image, the depth image and the framebuffer of every frame slot, and
    // - when the world sub-pass is enabled - the shared SCREEN_EMISSION and binding-25 wraps and the
    // world framebuffers built over them. The caller has to call it before the engine destroys its
    // framebuffer images (the Framebuffers::PrepareForSize path, which waits for the device first,
    // Framebuffers.cpp:227-241) - otherwise the wraps reference destroyed VkImages. The next
    // Prepare re-reads the engine's accessors and re-wraps, so the world survives a resize without
    // a second CreateWorld, but the host may call CreateWorld again instead. Releases go through the
    // frame context's retire queue when one exists; the destructor drops directly, after a device
    // idle.
    void ReleaseTargets();

private:
    struct Target
    {
        // The engine image the slot currently wraps; a change means the engine recreated it (or the
        // swap permutation flipped), so the wrap and the framebuffer over it have to follow.
        VkImage albedoImage = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;

        nvrhi::TextureHandle albedoTexture;
        nvrhi::TextureHandle depthTexture;
        nvrhi::FramebufferHandle framebuffer;

        // The world sub-pass's framebuffer of this slot: the slot's ALBEDO and depth plus the
        // shared SCREEN_EMISSION wrap, in the attachment order of the legacy world framebuffer
        // (RasterPass.cpp:115-119). Null until the world is enabled and its wraps exist.
        nvrhi::FramebufferHandle worldFramebuffer;

        // The SCREEN_EMISSION wrap that framebuffer references; a changed pointer means the engine
        // re-created the image (or the size changed) and the framebuffer has to be rebuilt.
        nvrhi::ITexture *worldScreenEmissionTexture = nullptr;

        // False until the world framebuffer exists and the frame's world binding sets are ready.
        bool worldValid = false;

        // True once the sky's Render cleared the depth for this frame; RenderWorld clears it itself
        // when the sky had no draws and left the clear undone.
        bool depthCleared = false;

        // False until Prepare succeeded for this slot in the current configuration.
        bool valid = false;

        // The wrapped ALBEDO image rests in the engine's VK_IMAGE_LAYOUT_GENERAL (NVRHI's
        // UnorderedAccess) between frames - the layout the engine's own framebuffer descriptors
        // declare for it. The wrap keeps no state between command lists (RhiTextureSource.h:56-70),
        // so every Prepare announces that state and the first framebuffer use issues the real
        // transition to COLOR_ATTACHMENT_OPTIMAL.
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    void ReleaseTarget(Target &target);
    void ReleasePipelineCache();

    // The world sub-pass's per-frame work: resolve the engine images (the Framebuffers accessors
    // first, the handles of CreateWorld as the fallback), (re)wrap them when an image or the size
    // changed and rebuild the binding-25 set; then UpdateWorldFramebuffer builds the slot's world
    // framebuffer over the current SCREEN_EMISSION wrap. PrepareWorldTarget is the one call Prepare
    // makes and it also announces the two wraps' states on the list.
    void PrepareWorldTarget(nvrhi::ICommandList *pCommandList, Target &target,
                            const Framebuffers &framebuffers, uint32_t frameIndex,
                            uint32_t width, uint32_t height);
    void UpdateWorldWraps(const Framebuffers &framebuffers, uint32_t frameIndex,
                          uint32_t width, uint32_t height);
    void UpdateWorldFramebuffer(Target &target);

    void ReleaseWorldWraps();
    void ReleaseWorldTexture(nvrhi::TextureHandle &texture);
    void ReleaseWorldFramebuffersSet();
    void ReleaseWorldBufferSets();

    nvrhi::IGraphicsPipeline *GetPipeline(uint32_t stateFlags, bool applyVertexColorGamma, bool world);
    nvrhi::GraphicsPipelineHandle CreatePipeline(uint32_t stateFlags, bool applyVertexColorGamma, bool world);

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

    // The camera of SetSkyCamera, kept raw until Render knows the target size the jitter is
    // normalized by.
    float view[16] = {};
    float proj[16] = {};
    float jitter[2] = {};
    float skyViewerPos[3] = {};
    bool hasSkyCamera = false;

    // Built by Render from the camera above, in the target's size.
    float defaultViewProj[16] = {};

    // One entry per engine frame slot: ALBEDO is a swapped engine image, so the slot's image is not
    // shared with the other slot's (Framebuffers_BindingsSwapped, Framebuffers.cpp:40-50), and the
    // depth image and the framebuffer over the pair follow it, exactly like RasterPass's per-slot
    // depth and sky framebuffer arrays (RasterPass.h:91-98).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // Set by Prepare, read by Render: the slot whose target the next Render records into.
    uint32_t activeTargetIndex = 0;

    // Valid only for the format the pipelines were created with; ALBEDO is
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32 (ShaderCommonCFramebuf.cpp:9), but the wrapped texture's
    // own format is what the framebuffer and the pipeline are built from.
    nvrhi::Format pipelineColorFormat = nvrhi::Format::UNKNOWN;
    std::unordered_map<uint32_t, nvrhi::GraphicsPipelineHandle> pipelines;

    // -- the world sub-pass --

    bool worldCreated = false;

    // "FragWorld" is RsWorld.frag.spv; the vertex half is the sky's RsRasterizer.vert.spv, so
    // vertexShader is shared.
    nvrhi::ShaderHandle worldPixelShader;

    // The world pipeline's layouts 1..4, in the shader's set order: the engine's global uniform,
    // the tonemapping buffer, the set-3 hole (a zero-item layout that carries the push constants)
    // and the partial framebuffers layout (one Texture_UAV for the shader's binding 25).
    nvrhi::BindingLayoutHandle worldUniformLayout;
    nvrhi::BindingLayoutHandle worldTonemappingLayout;
    nvrhi::BindingLayoutHandle worldPushConstantLayout;
    nvrhi::BindingLayoutHandle worldFramebuffersLayout;

    // The world's buffer sets, rebuilt by CreateWorld when a buffer changes. The buffers are
    // borrowed from the host and outlive the pass.
    nvrhi::IBuffer *worldUniformBuffer = nullptr;
    nvrhi::IBuffer *worldTonemappingBuffers[MAX_FRAMES_IN_FLIGHT] = {};
    nvrhi::BindingSetHandle worldUniformSet;
    nvrhi::BindingSetHandle worldTonemappingSets[MAX_FRAMES_IN_FLIGHT];

    // The real, empty set that fills the set-3 hole. A null entry in GraphicsState::bindings would
    // be dereferenced by the backend's automatic-barrier pass (vulkan-state-tracking.cpp:105)
    // before the bind pass could treat it as a hole.
    nvrhi::BindingSetHandle worldHoleSet;

    // The engine handles the host passed to CreateWorld: the fallback for the frames the engine has
    // no framebuffers yet. Both images are not swapped, so one pair of handles serves every slot
    // until the engine re-creates them (measured: Bindings[62] == BindingsSwapped[62] == 62,
    // Bindings[25] == BindingsSwapped[25] == 25 in Generated/ShaderCommonCFramebuf.cpp).
    std::tuple<VkImage, VkImageView, VkFormat> worldScreenEmissionHandles;
    std::tuple<VkImage, VkImageView, VkFormat> worldStorageHandles;

    // The shared wraps of the two images and the images/size they were built from. The wraps are
    // not per slot: neither image is swapped, while ALBEDO and the depth are.
    nvrhi::TextureHandle worldScreenEmissionTexture;
    nvrhi::TextureHandle worldStorageTexture;
    VkImage worldScreenEmissionImage = VK_NULL_HANDLE;
    VkImage worldStorageImage = VK_NULL_HANDLE;
    uint32_t worldWrapWidth = 0;
    uint32_t worldWrapHeight = 0;

    // Set 4, rebuilt whenever the storage wrap above changes. Null while there is nothing to bind.
    nvrhi::BindingSetHandle worldFramebuffersSet;

    // True while worldStorageTexture is the 1x1 dummy the .cpp creates when the engine has no
    // binding-25 image: an owned texture, so it needs no per-list state announcement.
    bool worldStorageIsDummy = false;

    // The format of the world's colour 1 the cached world pipelines were built with; a change
    // means every cached pipeline belongs to the wrong framebuffer info.
    nvrhi::Format worldColorFormat = nvrhi::Format::UNKNOWN;
    std::unordered_map<uint32_t, nvrhi::GraphicsPipelineHandle> worldPipelines;

    bool warnedMissingCamera = false;
    bool warnedMissingGeometry = false;
    bool warnedMissingWorldTarget = false;

    bool created = false;
};

}
