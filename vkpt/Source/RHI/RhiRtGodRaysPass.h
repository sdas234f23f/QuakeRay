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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include <nvrhi/vulkan.h>

#include "../Common.h"

namespace vkpt
{

class Framebuffers;

namespace rhi
{
class RhiFrameContext;
}

// The RHI module of the legacy god rays: the half-resolution volumetric sunlight march into image 63
// (GOD_RAYS) and the full-resolution bilateral upscale into image 64 (GOD_RAYS_FILTERED), which
// `CmPrepareFinal` consumes. The legacy reference is the `GodRays` class (GodRays.{h,cpp}) and the
// host block that drives it (VulkanDevice.cpp:906-1028); the RHI host mirrors the same order:
//
//   primary (the G-buffer images the trace reads) -> shadow map -> THIS -> [refl/refr] -> the
//   compose chain's gradient reproject -> direct -> indirect -> compose -> present
//
// The module records the frame's compute dispatches on the caller's open list:
//
//  1. `CmGodRays.comp` (`CmGodRays.comp.spv`), `[numthreads(16, 16, 1)]` (measured 2026-09-26 with
//     `spirv-dis` over the runtime blob). One invocation per HALF-resolution pixel; the shader's own
//     `screenSize` bound is `((renderWidth + 1) / 2, (renderHeight + 1) / 2)`. `passIndex = 0` is
//     the primary march (there the shader also clears image 63 when `godRaysEnabled == 0` or the
//     sun colour is zero); `passIndex = 1` is the reflected-segment accumulation, which early-outs
//     on `Q2ViewDepth >= 0` and adds its inscatter on top of the primary result. The passIndex is a
//     4-byte push constant written between `setComputeState` and `dispatch` (the A4.5 pattern,
//     RhiRtComposePass.cpp:2375-2381). The blob's sets, exactly:
//       set 0: `texShadowMap` (Texture2D<float4>, raw 0) + `texShadowMap_Sampler` (raw 1);
//       set 1: `godRaysParams` (StructuredBuffer<GodRaysParams_BT>, raw 0);
//       set 2: `framebufGodRays` (raw 63, UAV) + `framebufDepthWorld_Sampled` (raw 133),
//              `framebufSurfacePosition_Sampled` (raw 143), `framebufViewDirection_Sampled`
//              (raw 147), `framebufThroughput_Sampled` (raw 150), `framebufQ2ViewDepth_Sampled`
//              (raw 205), `framebufQ2GodRaysThroughputDist_Sampled` (raw 213) - i.e. engine images
//              9, 19, 23, 26, 81 and 89 as `GetImageHandles` resolves them;
//       set 3: `globalUniform` (raw 0);
//       set 4: `blueNoiseTextures` (Texture2DArray<float4>, raw 0, no sampler).
//  2. `CmGodRaysFilter.comp` (`CmGodRaysFilter.comp.spv`), `[numthreads(16, 16, 1)]`, over the full
//     render size. The blob declares `framebufGodRaysFiltered` (raw 64, UAV),
//     `framebufGodRays_Sampled` (raw 187 = 124 + 63) and `framebufQ2ViewDepth_Sampled` (raw 205),
//     set 3 `globalUniform` - and NO set 0, set 1, set 4 or push constant. The legacy host still
//     bound all five sets; this module builds a four-layout pipeline (two placeholder positions
//     plus the effective framebuffer and uniform ones) so the blob's engine set numbers 2 and 3
//     line up, and contains no set 0/1/4 descriptors.
//
// The dispatch sizes are the legacy `Utils::GetWorkGroupCount(size, 16)` = `1 + ceil(size / 16)`
// (Utils.cpp:319-328), i.e. one workgroup beyond the ceiling, whose invocations the shader's own
// bounds check discards: trace over `(width + 1) / 2` and filter over `width`/`height`
// (GodRays.cpp:268-275 and :305-306).
//
// The engine images this module needs, resolved through
// `Framebuffers::GetImageHandles(image, frameIndex, ResolutionState)` like every other RHI module,
// wrapped once per slot with `rhi::wrapEngineStorageImage` (VK_IMAGE_LAYOUT_GENERAL = NVRHI's
// UnorderedAccess, `keepInitialState = false`, so every list announces the state):
//   63  GOD_RAYS                       B10G11R11_UFLOAT_PACK32  `(w + 1) / 2 x (h + 1) / 2`  UAV + SRV (filter)
//   64  GOD_RAYS_FILTERED              B10G11R11_UFLOAT_PACK32  `w x h`                      UAV
//   9   DEPTH_WORLD                   R16_SFLOAT               `w x h`                      SRV
//   19  SURFACE_POSITION              R32G32B32A32_SFLOAT      `w x h`                      SRV
//   23  VIEW_DIRECTION                R16G16B16A16_SFLOAT      `w x h`                      SRV
//   26  THROUGHPUT                    R16G16B16A16_SFLOAT      `w x h`                      SRV
//   81  Q2_VIEW_DEPTH                 R32_SFLOAT               `w x h`                      SRV
//   89  Q2_GOD_RAYS_THROUGHPUT_DIST   R16G16B16A16_SFLOAT      `w x h`                      SRV
// (formats from `ShFramebuffers_Formats`, sizes from `Framebuffers::GetFramebufSize`,
// Generated/ShaderCommonCFramebuf.cpp; image 89 is NOT part of the compose union, it is resolved
// like any other engine framebuffer image.) Images 9/19/23/26/81/89 are written by
// `RhiRtPrimaryPass::Render`, which has to run before this module on the same list; image 89 is the
// primary's `storeQ2GBuffer` output the reflection pass keys on.
//
// Image-state contract, per Render call: the module announces UnorderedAccess for all eight wraps
// (the engine's real GENERAL layout) before its first binding - the automatic barrier pass then
// transitions the SRV reads out of it - and after the dispatches moves every image whose last use
// was a sampled read (63, 9, 19, 23, 26, 81, 89) back to UnorderedAccess, so the frame ends with
// 63/64 in UnorderedAccess (the engine's convention and what `CmPrepareFinal` and the next frame
// expect) and no image left in a read-only layout. Between `passIndex` 0 and 1 the module requires
// image 63 in UnorderedAccess explicitly: the backend re-requires a binding set that has any UAV
// item on every state set (vulkan-state-tracking.cpp:111, :40-63), which already places the
// same-state UAV barrier for pass 1, and the explicit call documents and guarantees the write ->
// read ordering between the two dispatches.
//
// The clear path keeps the legacy shape (VulkanDevice.cpp:964-969): when the host passes
// `params.godRaysEnabled == 0`, Render still records trace(0) + filter with those params, and the
// shader itself writes zeros to the half-resolution image which the filter then upscales into 64.
// That is one code path, needs no clear API or CopyDest transitions, and matches the reference
// exactly; a frame with god rays enabled but no shadow map render (no AABB, no geometry) is not
// dispatched at all and 64 keeps its previous frame, also exactly like the legacy
// (VulkanDevice.cpp:970-1006). The module does require a shadow map texture and sampler for ANY
// dispatch, because every item of the trace's layout has to be filled; the host sets them once
// (SetShadowMap), and the shader never samples them on the clear path.
//
// Host inputs of `Render`, and where the legacy takes them from
// (VulkanDevice.cpp:909-1028):
//  - `pFramebuffers`/`width`/`height`: the engine framebuffer registry and the render resolution
//    (`globalUniform.renderWidth/renderHeight`; the shader bounds and the dispatch arithmetic use
//    the same numbers).
//  - `pUniformBuffer`: the engine global uniform as a static constant-buffer wrap, the same handle
//    the primary/compose passes take. The shader reads `cameraPosition`, `cameraMediaType`,
//    `frameId`, `renderWidth/renderHeight`, `debugShowFlags` (the `DEBUG_SHOW_FLAG_GOD_RAYS`
//    diagnostic branch, CmGodRays.comp.hlsl:169-186), `waterColorAndDensity` and
//    `acidColorAndDensity` from it - no separate arguments.
//  - `params`: the `GodRaysParams_BT` fields, all host-computed:
//      * sunDirection  - toward the sun: the sun path stores `-sunDir` of
//                        `LightManager::GetLastDirectionalLight` (:985-987), the sky-brightest path
//                        the game's `godRaysSkyDirection` unchanged (:976-978);
//      * sunColor      - the light-fixed-up colour (:988-990) or `godRaysSkyColor` (:979-981);
//      * worldCenter/worldHalfSizeInv - `scene->GetAABB`: centre and `1 / max(halfSize, 1)` per
//                        axis (:953-998);
//      * shadowMapVP/shadowMapDepthScale - the view-projection the shadow map was rendered with and
//                        its depth scale, returned by `ShadowMap::Render` (:1000-1001); the shipped
//                        blob carries `shadowMapDepthScale` but never reads it (measured: no
//                        access to member 5 in the SPIR-V);
//      * godRaysIntensity - `8.0f * rt_godrays_intensity` (clamped at 0, :918-931);
//      * godRaysEccentricity - `0.75f` (:932);
//      * godRaysEnabled - the FINAL switch `godRaysEnabled && (sunExists || useSkyBrightest)`
//                        (:920-943), NOT the raw cvar: when it is 0 the shader clears image 63.
//    `sunAngularRadius` is NOT a param: the legacy fetches it from
//    `LightManager::GetLastDirectionalLight` (:934-935) but neither `GodRays::Params` nor the
//    shader's `GodRaysParams_BT` carries it, so the coordinator does not pass it.
//    The `debugShowFlags` god-rays bit needs no argument either: the shader reads it from the
//    uniform bound at set 3 (the `DEBUG_SHOW_FLAG_GOD_RAYS` branch above), so the only host
//    requirement is that the frame's uniform bytes were written before Render, as the skeleton
//    already does.
//    The sky-brightest substitution (`RASTERIZED_GEOMETRY` + `godRaysFromSkyTexture`) also feeds
//    the shadow map's light direction (`-godRaysSkyDirection`, :945-951) - that is the shadow
//    pass's input, not this one's.
//  - `traceReflections`: the legacy `reflectRefractMaxDepth > 0` gate (:1011-1014). When true and
//    `godRaysEnabled != 0`, Render records trace(1) between trace(0) and the filter. The
//    reflected-segment god rays key on the negative `Q2_VIEW_DEPTH` the refl/refr raygen writes;
//    until that raygen is ported every pixel early-outs in the shader, so the host passes false (or
//    the uniform's reflect-refract depth is 0) and the dispatch is then not recorded at all.
//
// Shadow map contract (`SetShadowMap`): the module binds the exact texture and sampler the host
// passes, and never owns either. Expected objects:
//  - a `nvrhi::ITexture` with dimension Texture2D, format `nvrhi::Format::D32`
//    (`VK_FORMAT_D32_SFLOAT`), one mip, one array layer and a non-zero square extent (the legacy
//    image is 4096x4096, ShadowMap.cpp:172), created as a shader resource; the module does not
//    create it (RhiShadowMapPass does);
//  - a `nvrhi::ISampler` with min/mag linear, mip nearest, `SamplerReductionType::Minimum`,
//    clamp-to-border on all axes and a white border colour - exactly the legacy sampler
//    (ShadowMap.cpp:210-223); a sampler that deviates silently changes the shafts, so the module
//    logs a one-shot warning (it keeps using it).
//  The module deliberately does not announce a state for the shadow map: it binds the very NVRHI
//  texture handle `RhiShadowMapPass::GetTexture()` returns, so the list's tracker already carries
//  the state the shadow pass left (`ShaderResource | DepthRead` after a render; the image's
//  `DepthWrite` resting state when no render happened), and the SRV item's own requirement is what
//  transitions the image when needed. Announcing NonPixelShaderResource instead would overwrite
//  that truth and could name the wrong old layout in the close-time restore of the image's
//  `keepInitialState`; a host that passes a different wrap of the same VkImage would make the
//  state unknown, which is why the exact handle matters.
//
// Blue-noise contract (`SetBlueNoiseTexture`): the coordinator hands over the indirect pass's wrap
// of the engine BlueNoise image (the same wrap RhiRtIndirectPass's set 5 binds), a `Texture2DArray`
// read as `Texture2DArray<float4>` with no sampler. SetBlueNoiseTexture enforces the shape of the
// indirect pass's set 5 - dimension Texture2DArray, format RGBA8_UNORM, at least 128x128 with at
// least 128 layers (Random.hlsli's wrapping layer index) - and the module announces
// NonPixelShaderResource for it, the same contract RhiRtIndirectPass::SetBlueNoiseTexture documents.
//
// Resize and lifetime: the eight engine images are watched per slot (handles + size); a change
// retires the slot's wraps and sets through `RhiFrameContext::Retire` and rebuilds them, so the
// module survives `Framebuffers::PrepareForSize` if the host calls `ReleaseTargets()` before it,
// like every other module. The params buffers, the shadow/blue-noise sets and the pipelines are not
// framebuffer-dependent and survive a resize.
//
// The pass is a no-op until Create succeeded and while an input is missing (no shadow map, no
// blue-noise texture, no framebuffers, no uniform, an image handle or extent the RHI cannot wrap);
// every early return is quiet after the first warning. It is not thread-safe: Render uses the
// per-slot target of the frameIndex it is given, which is the engine's single-threaded per-slot
// frame model (RhiFrameContext).
class RhiRtGodRaysPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    // The CPU mirror of the shader's `GodRaysParams_BT` (CmGodRays.comp.hlsl:87-99). The offsets
    // are the blob's own, measured 2026-09-26 with `spirv-dis` over `vkpt\Build\CmGodRays.comp.spv`:
    // OpMemberDecorate offsets 0, 16, 32, 48, 64 (MatrixStride 16, RowMajor), 128, 132, 136, 140
    // and 144. The struct is 148 bytes while the shader's `ArrayStride` is 160 (the runtime
    // `%_runtimearr_GodRaysParams_BT` decoration) - the trailing 12 bytes are the HLSL structured
    // buffer's stride padding, and the buffer is created with `structStride = GOD_RAYS_PARAMS_STRIDE`
    // (160), NOT `sizeof(Params)`. This is deliberately not the legacy `GodRays::Params` layout
    // (GodRays.h:44-56), which is the same field sequence but was bound as a plain storage buffer
    // of its own size; the members and their meaning are identical.
    struct Params
    {
        // Offset 0. The direction TOWARD the sun (the shader's `dot(direction, sunDirection)`).
        float sunDirection[4];
        // Offset 16. The sun colour (xyz; w is not read).
        float sunColor[4];
        // Offset 32. The scene AABB centre (xyz).
        float worldCenter[4];
        // Offset 48. 1 / max(halfSize, 1) per axis (xyz).
        float worldHalfSizeInv[4];
        // Offset 64. The sun view-projection the shadow map was rendered with, 16 floats copied
        // verbatim from `RhiShadowMapPass::GetViewProjection()` (the legacy `ShadowMap::Render`
        // out-parameter, VulkanDevice.cpp:1000). The blob decorates the member RowMajor with
        // MatrixStride 16 and projects with `mul(godRaysParams[0].shadowMapVP, float4(worldPos, 1))`,
        // so the bytes are the ones the shadow map itself used.
        float shadowMapVP[16];
        // Offset 128. The shadow map depth scale the legacy carries; the shipped blob never reads it.
        float shadowMapDepthScale;
        // Offset 132. 8.0f * rt_godrays_intensity.
        float godRaysIntensity;
        // Offset 136. 0.75f.
        float godRaysEccentricity;
        // Offset 140. The final switch; 0 makes the shader clear image 63 on passIndex 0.
        uint32_t godRaysEnabled;
        // Offset 144. The HLSL struct's `_padding`; never read.
        float padding;
    };

    // The shader's structured-buffer element stride (measured ArrayStride 160), and the size of the
    // one-element buffer that carries Params.
    static constexpr uint32_t GOD_RAYS_PARAMS_STRIDE = 160;

    // The `GodRaysPush_BT` push constant: one uint passIndex at offset 0 (measured; GodRays.cpp:170-174).
    static constexpr uint32_t GOD_RAYS_PUSH_SIZE = 4;

    // The workgroup size of both blobs (measured: [numthreads(16, 16, 1)]).
    static constexpr uint32_t GOD_RAYS_GROUP_SIZE = 16;

    // The module's own slot order for the eight engine images (the .cpp's image table). Public so
    // the file-local binding tables can name the slots; the numeric order is not an engine index.
    enum ImageSlot : uint32_t
    {
        IMAGE_GOD_RAYS = 0,                        // 63
        IMAGE_GOD_RAYS_FILTERED,                   // 64
        IMAGE_DEPTH_WORLD,                         // 9
        IMAGE_SURFACE_POSITION,                    // 19
        IMAGE_VIEW_DIRECTION,                      // 23
        IMAGE_THROUGHPUT,                          // 26
        IMAGE_Q2_VIEW_DEPTH,                       // 81
        IMAGE_Q2_GODRAYS_THROUGHPUT_DIST,          // 89
        IMAGE_COUNT,
    };

    RhiRtGodRaysPass();
    ~RhiRtGodRaysPass();

    RhiRtGodRaysPass(const RhiRtGodRaysPass &other) = delete;
    RhiRtGodRaysPass(RhiRtGodRaysPass &&other) noexcept = delete;
    RhiRtGodRaysPass &operator=(const RhiRtGodRaysPass &other) = delete;
    RhiRtGodRaysPass &operator=(RhiRtGodRaysPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queue every replaced wrap and set goes
    // through; 'pShaderFolderPath' is the folder the engine blobs load from, with the trailing
    // separator ('CmGodRays.comp.spv' and 'CmGodRaysFilter.comp.spv' are read from it). None is
    // owned; all have to outlive this object, and a null or unusable one makes Create fail. The
    // pass logs through 'pfnPrint'. Returns false and leaves the pass unusable if a shader, layout,
    // pipeline, buffer or set cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // Set 0's texture and sampler, owned by RhiShadowMapPass. The coordinator passes
    // `RhiShadowMapPass::GetTexture()`/`GetSampler()` - the exact NVRHI objects, so the shared
    // tracker state drives the image's transition (see the class comment for the expected objects).
    // The setters may be called any time; a replaced texture rebuilds the set (the old one goes
    // through the retire queue). A null texture or sampler clears both, and Render then warns once
    // and skips.
    void SetShadowMap(nvrhi::ITexture *pShadowMap, nvrhi::ISampler *pShadowMapSampler);

    // Set 4's texture: the indirect pass's blue-noise wrap (see the class comment). A null clears
    // it, a replaced one rebuilds the set; Render skips until a valid texture is set.
    void SetBlueNoiseTexture(nvrhi::ITexture *pBlueNoise);

    // One call per traced frame, on the frame context's open command list of 'frameIndex', after
    // RhiRtPrimaryPass::Render (the G-buffer images it reads) and before the compose chain.
    // 'pFramebuffers'/'width'/'height' are the engine framebuffer registry and the render
    // resolution; 'pUniformBuffer' is the engine global uniform as a static constant-buffer wrap;
    // 'params' are the GodRaysParams_BT fields, all host-computed (see the class comment for each
    // field's legacy source); 'traceReflections' is the legacy `reflectRefractMaxDepth > 0` gate and
    // records the passIndex 1 dispatch between trace(0) and the filter when it is true and
    // `params.godRaysEnabled != 0`.
    //
    // What is recorded: the wraps of the eight engine images (created on first use, re-created when
    // the engine re-created an image or the size changed; replaced wraps and sets go through the
    // frame context's retire queue), the per-slot framebuffer and uniform sets, the write of the
    // slot's params buffer, then trace(0) [+ trace(1)] + filter. The images end in UnorderedAccess
    // (see the class comment). The call is a no-op when the pass is not created, the frame index is
    // out of range, the size is zero, the shadow map or blue-noise texture is not set, or the
    // framebuffers/uniform are missing or in a shape NVRHI refuses.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height,
                nvrhi::IBuffer *pUniformBuffer,
                const Params &params,
                bool traceReflections);

    // Drops every slot's image wraps and the sets over them and the per-slot uniform sets, and
    // retires them through the frame context's queue. The caller has to call it before the engine
    // destroys its framebuffer images (the Framebuffers::PrepareForSize path) - otherwise the wraps
    // reference destroyed VkImages. The next Render re-reads the handles and re-wraps, so the pass
    // survives a resize without a second Create. The params buffers, the shadow map and blue-noise
    // sets and the pipelines do not reference framebuffer images and stay.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: the engine images are per-slot (or shared) framebuffer
    // images and the sets reference their wraps, so neither the wraps nor the sets can be shared
    // across slots. The uniform set follows the buffer pointer the way the other passes' uniform
    // sets do.
    struct Target
    {
        uint64_t imageHandles[IMAGE_COUNT] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle engineTextures[IMAGE_COUNT];
        nvrhi::BindingSetHandle traceFramebufferSet;
        nvrhi::BindingSetHandle filterFramebufferSet;
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // The shared preamble of Render: validates the arguments, resolves the eight image handles and
    // extents of 'frameIndex', retires and re-wraps everything when the engine re-created an image
    // or the size changed, prepares the slot's sets and the uniform set. Returns the slot's target,
    // or null when the call cannot be recorded (the caller treats it as a silent skip).
    Target *PrepareFrame(uint32_t frameIndex,
                         const Framebuffers *pFramebuffers,
                         uint32_t width,
                         uint32_t height,
                         nvrhi::IBuffer *pUniformBuffer);

    // Declares the engine's resting state (UnorderedAccess, i.e. GENERAL) for all eight wraps on
    // the list, before the entry point's first binding touches one. A repeated call is a no-op.
    void AnnounceFrameImages(nvrhi::ICommandList *pCommandList, const Target &target) const;

    // Queues a same-state UnorderedAccess requirement for one union image - the pass-to-pass and
    // end-of-call discipline of the class comment: NVRHI emits the same-state UAV barrier (or the
    // read-only -> GENERAL transition) and commits it before the next state set or at the list's
    // close.
    void RequireImageUnorderedAccess(nvrhi::ICommandList *pCommandList,
                                     const Target &target,
                                     uint32_t imageSlot) const;

    // Builds the two sets over the module's exact layouts when they are missing. Returns false when
    // a binding cannot be filled - a table/image mismatch, which the module's tables make
    // unreachable.
    bool PrepareFramebufferSets(Target &target);

    // Set 3 over the module's constant-buffer layout, rebuilt when the pointer changed. Returns
    // false when the buffer is not the static constant-buffer wrap the shader's
    // `ConstantBuffer<ShGlobalUniform>` requires.
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);

    // Set 0: builds (or rebuilds, when either pointer changed) the shadow map set. Returns false
    // when no valid shadow map was set.
    bool PrepareShadowSet();

    // Set 4: builds (or rebuilds, when the texture changed) the blue-noise set and announces the
    // state its SRV requires. Returns false when no valid texture was set.
    bool PrepareBlueNoiseSet(nvrhi::ICommandList *pCommandList);

    // One `setComputeState` + optional `setPushConstants` + `dispatch` for one pass. The sets are
    // added in the pipeline's layout order (the backend binds the list positionally,
    // vulkan-resource-bindings.cpp:940-1019). 'pPassIndex' is the 4-byte GodRaysPush_BT written
    // between the state and the dispatch, the order `setPushConstants` requires
    // (nvrhi.h:3430-3440, RhiRtComposePass.cpp:2375-2381).
    void RecordDispatch(nvrhi::ICommandList *pCommandList,
                        nvrhi::IComputePipeline *pPipeline,
                        std::initializer_list<nvrhi::IBindingSet *> sets,
                        uint32_t groupsX,
                        uint32_t groupsY,
                        uint32_t groupsZ,
                        const uint32_t *pPassIndex);

    void ReleaseFramebufferSets(Target &target);
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, which outlives this object.
    rhi::RhiFrameContext *frameContext = nullptr;

    // The two engine blobs.
    nvrhi::ShaderHandle traceShader;
    nvrhi::ShaderHandle filterShader;

    // The eight layouts this module owns. The trace's mirror the engine's descriptor set numbers
    // (0 shadow, 1 params, 2 framebuffers, 3 uniform, 4 blue noise, 5 the descriptor-less
    // push-constant layout); the filter's lead with two placeholder positions because its blob
    // declares its framebuffer and uniform bindings in engine sets 2 and 3.
    nvrhi::BindingLayoutHandle shadowLayout;
    nvrhi::BindingLayoutHandle paramsLayout;
    nvrhi::BindingLayoutHandle traceFramebufferLayout;
    nvrhi::BindingLayoutHandle filterFramebufferLayout;
    nvrhi::BindingLayoutHandle uniformLayout;
    nvrhi::BindingLayoutHandle blueNoiseLayout;
    nvrhi::BindingLayoutHandle pushConstantLayout;
    nvrhi::BindingLayoutHandle emptyLayout;

    // The two compute pipelines. The filter carries no push-constant layout: its blob declares no
    // push constant (measured).
    nvrhi::ComputePipelineHandle tracePipeline;
    nvrhi::ComputePipelineHandle filterPipeline;

    // The empty set of the filter's two leading placeholder positions. One set bound at two
    // positions is legal Vulkan; the layouts are the same object (the compose module does the same
    // for its descriptor-set holes).
    nvrhi::BindingSetHandle emptySet;

    // The per-slot `GodRaysParams_BT` buffers (one element each, stride 160) and the one-item sets
    // over them. They are module-owned, written per frame with writeBuffer and never
    // framebuffer-dependent, so they survive a resize.
    nvrhi::BufferHandle paramsBuffers[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle paramsSets[MAX_FRAMES_IN_FLIGHT];

    // Set 0's objects: the raw handles keep the texture and sampler alive, the raw pointers are the
    // set's cache key.
    nvrhi::TextureHandle shadowMapTexture;
    nvrhi::ITexture *shadowMapSetTexture = nullptr;
    nvrhi::SamplerHandle shadowMapSampler;
    nvrhi::ISampler *shadowMapSetSampler = nullptr;
    nvrhi::BindingSetHandle shadowSet;

    // Set 4's texture and set, with the same replace rule.
    nvrhi::TextureHandle blueNoiseTexture;
    nvrhi::ITexture *blueNoiseSetTexture = nullptr;
    nvrhi::BindingSetHandle blueNoiseSet;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // One-shot warnings for the inputs that can legitimately be missing for a few frames or are a
    // permanent host-side mistake.
    bool warnedMissingShadowMap = false;
    bool warnedBadShadowMap = false;
    bool warnedShadowSampler = false;
    bool warnedMissingBlueNoise = false;
    bool warnedBadBlueNoise = false;
    bool warnedMissingFramebuffers = false;
    bool warnedUnexpectedSize = false;
    bool warnedMissingUniform = false;
    bool warnedBadUniform = false;
    bool warnedBadTable = false;

    bool created = false;
};

// The CPU mirror has to match the measured member layout member by member; the shader's element
// stride is GOD_RAYS_PARAMS_STRIDE (160), not this size, which is why the buffer is created with
// the stride, not with sizeof().
static_assert(sizeof(RhiRtGodRaysPass::Params) == 148,
              "GodRaysParams_BT is 148 bytes; the structured buffer uses the 160-byte stride");
static_assert(offsetof(RhiRtGodRaysPass::Params, sunDirection) == 0, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, sunColor) == 16, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, worldCenter) == 32, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, worldHalfSizeInv) == 48, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, shadowMapVP) == 64, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, shadowMapDepthScale) == 128, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, godRaysIntensity) == 132, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, godRaysEccentricity) == 136, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, godRaysEnabled) == 140, "measured member offset");
static_assert(offsetof(RhiRtGodRaysPass::Params, padding) == 144, "measured member offset");
static_assert(RhiRtGodRaysPass::GOD_RAYS_PARAMS_STRIDE > sizeof(RhiRtGodRaysPass::Params),
              "the shader's stride carries the HLSL struct's padding");

}
