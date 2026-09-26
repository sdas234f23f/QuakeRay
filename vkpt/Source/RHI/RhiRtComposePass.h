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

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>

#include <nvrhi/vulkan.h>

#include "../Common.h"
#include "../Generated/ShaderCommonCFramebuf.h"

namespace vkpt
{

class Framebuffers;
class Tonemapping;

namespace rhi
{
class RhiFrameContext;
}

// The RHI module of the composed traced frame: the engine's ASVGF denoiser chain and the minimal
// TAAU upscaler around the A4.2b/A4.4 compose preview. The legacy reference is `Q2Denoiser`'s
// three entry points, and the RHI host has to mirror the order the legacy frame records them in
// (VulkanDevice.cpp:1032-1136):
//
//  1. RenderGradientReproject - between `RhiRtPrimaryPass::Render` and `RhiRtDirectPass::Render`,
//     only while the denoiser switch is on. It reprojects the previous frame's G-buffer onto the
//     current one and fills the checkerboard half the primary did not trace, and it writes the
//     1/3-res `Q2GradSmplPos` image (115) the direct and indirect raygens classify gradient
//     pixels with (`q2GetIsGradient`, Q2LightLists.hlsli:141-161; Q2Denoiser.cpp:327-380;
//     VulkanDevice.cpp:1032-1039);
//  2. Render - after `RhiRtIndirectPass::Render`. It always runs the adapter; between the adapter
//     and the interleave it either records the full ASVGF chain - `CmQ2GradientImg`,
//     seven `CmQ2GradientAtrous` iterations, `CmQ2Temporal`, four `CmQ2AtrousLF` iterations, four
//     `CmQ2Atrous` iterations - when the host reports the switch on, or skips straight to the
//     interleave when it is off (the legacy `Denoise` with its `fltEnable[0] < 0.5` short-circuit,
//     Q2Denoiser.cpp:410-436 and :438-597). After the interleave the A4.4 tail follows unchanged:
//     the exposure pair (`CmLuminanceHistogram` + `CmLuminanceAvg`), the checkerboard resolve and
//     the final composition (`CmPrepareFinal`), which end in the display-referred FINAL image (28)
//     the present samples (`Tonemapping.cpp:153-208`, `ImageComposition.cpp:93-98, :113-202`,
//     VulkanDevice.cpp:1061-1085);
//  3. RenderTaaU - after the compose, the minimal temporal-AA upscaler of
//     `Q2Denoiser::ApplyTAAU` (Q2Denoiser.cpp:625-657): it reads the display-referred FINAL (28)
//     and MOTION_DLSS (31) at the render size and the previous upscaled history, and writes
//     UPSCALED_PING (29) plus the next history at the upscaled size. The present samples
//     UPSCALED_PING (`GetUpscaledTexture`) once this call is made; FINAL stays available through
//     `GetFinalTexture` for callers that skip the TAAU.
//
// The denoiser switch is one value, shared by all three entry points and the raygens:
// `globalUniform.fltEnable[0] >= 0.5`. This module never reads the engine's uniform bytes - the
// host passes `filterEnabled = uniform->GetData()->fltEnable[0] >= 0.5f` to `Render` and gates the
// `RenderGradientReproject` call on the same predicate (the legacy order at VulkanDevice.cpp:1032
// and the adapter's own `fltEnable.x < 0.5` branch inside `CmQ2Adapter.comp.hlsl:193-213`). The
// switch has to be the value the raygens were dispatched with, or the gradient classification and
// this chain disagree about the frame.
//
// The passes and their exact set-0 items (measured 2026-09-26 with `spirv-dis` over the engine
// blobs in vkpt\Build; the four pair-fixed blobs - gradientImg, gradientAtrous, atrousLF, atrous -
// carry the A4.5 stream-1 lists, where the sampled views of an image the pass also writes are gone
// and the reads go through the storage image):
//   CmQ2GradientReproject - 10 storage images (0 ALBEDO, 3 NORMAL, 7 METALLIC_ROUGHNESS,
//                           19 SURFACE_POSITION, 23 VIEW_DIRECTION, 83 Q2_BASE_COLOR,
//                           85 Q2_METALLIC, 113 Q2_GRAD_HF_SPEC_PING, 115 Q2_GRAD_SMPL_POS,
//                           121 Q2_RNG_SEED) and 17 sampled images (1, 4, 5, 6, 8, 13, 20, 24, 26,
//                           78, 80, 81, 82, 84, 86, 116, 122); no constants. Dispatch over
//                           `GWC(renderWidth / 3, 16) x GWC(renderHeight / 3, 16)`;
//   CmQ2Adapter           - 5 storage images (73, 75, 77, 79, 117) and 12 sampled images (0, 2, 3,
//                           7, 14, 15, 16-18, 26, 88, 90); the `fltEnable` branch inside the shader
//                           selects the compositing path;
//   CmQ2GradientImg       - 2 storage images (111, 113) and 6 sampled images (13, 73, 77, 79, 92,
//                           115); dispatch over the 1/3 size;
//   CmQ2GradientAtrous    - 4 storage images (111-114) and no sampled image; a 4-byte
//                           `IterationInfo_BT` push constant carries the iteration; dispatch over
//                           the 1/3 size; byte-identical for all seven iterations, so one pipeline
//                           serves them all (the legacy creates seven, Q2Denoiser.cpp:141-163);
//   CmQ2Temporal          - 9 storage images (91, 93, 97, 99, 101, 103, 105, 107, 109) and 21
//                           sampled images (3-8, 11, 13, 26, 73, 75, 77, 79, 81, 82, 92, 94, 96,
//                           98, 100, 112, 114); `[numthreads(15, 15, 1)]` (the one pass whose
//                           workgroup is not 16 x 16);
//   CmQ2AtrousLF          - 4 storage images (101-104) and 3 sampled images (5, 11, 81); the same
//                           4-byte push constant; dispatch over the 1/3 size;
//   CmQ2Atrous            - 8 storage images (95, 105-110, 117) and 15 sampled images (0, 2, 3, 5,
//                           7, 11, 26, 81, 88, 90, 91, 93, 97, 101, 103); the iteration is a
//                           specialization constant (`SpecId 0`), so the module builds four
//                           pipelines from four `createShaderSpecialization` handles;
//   CmQ2Interleave        - 1 storage image (27 PRE_FINAL) and 2 sampled images (26, 117);
//   CmLuminanceHistogram  - 1 sampled image (27 PRE_FINAL at raw 151) and no storage image;
//   CmLuminanceAvg        - no set 0; one 128x1x1 workgroup;
//   CmCheckerboard        - 3 storage images (28 FINAL, 60 ACID_FOG, 62 SCREEN_EMISSION) and 4
//                           sampled images (26, 27, 59, 61);
//   CmPrepareFinal        - 5 storage images (16-18, 28, 65) and 11 sampled images (0, 3, 9, 13,
//                           14, 15, 25, 26, 60, 62, 64) - 16 items after the A4.4 shader-side fix
//                           of the 28/152 pair;
//   CmQ2TAAU              - 2 storage images (29 UPSCALED_PING, 119 Q2_TAA_HISTORY), 3 sampled
//                           images (28 FINAL, 31 MOTION_DLSS, 120 Q2_TAA_HISTORY_PREV) and the
//                           game's sampler for 120 (raw 368); dispatch over the upscaled size.
// Every pass declares set 1 = the engine's global uniform at raw binding 0. The module's set-0
// offsets turn the engine's raw bindings into NVRHI slots (RhiPipeline.h): the storage images sit
// at raw binding = image index (`ShFramebuffers_Bindings`) with offset 0, the sampled views at raw
// binding = 124 + image index (`ShFramebuffers_Sampled_Bindings`) with offset 124, and the TAAU
// sampler at raw binding = 248 + 120 (`ShFramebuffers_Sampler_Bindings`) with offset 248.
//
// The iterative passes read one field per iteration and write the opposite one
// (CmQ2GradientAtrous.comp.hlsl:119-162, CmQ2AtrousLF.comp.hlsl:295-313,
// CmQ2Atrous.comp.hlsl:305-330); after the A4.5 shader fix both roles go through the storage image,
// so one set per pass carries the whole loop. Two dispatches that bind the same set object do not
// make NVRHI re-apply the set's state requirements (vulkan-state-tracking.cpp:91-115), which would
// leave each read unbarriered against the previous dispatch's write; the module therefore requires
// UnorderedAccess explicitly on the images each iteration just wrote, which NVRHI turns into a
// same-state UAV barrier (state-tracking.cpp:188-207) committed by the next `setComputeState`
// (vulkan-compute.cpp:145). The push-constant iterations receive their index with
// `setPushConstants` between `setComputeState` and `dispatch` (nvrhi.h:3430-3440).
//
// 80 engine images are wrapped per slot, one wrap per union entry (the .cpp's COMPOSE_IMAGES
// table), and the passes share the wraps, so the pass-to-pass hand-offs are NVRHI's own
// UAV -> SRV -> UAV transitions on one wrap: 117 written by the adapter and read by the interleave
// or the atrous composite; 111-114 written by the gradient chain and read by the temporal pass;
// 91/93/97/99/101/103/105/107/109 written by the temporal pass and read (or written further) by
// the a-trous chain; 95/105-110 ping-ponged by the HF a-trous iterations; 27 written by the
// interleave and read by the exposure histogram and the checkerboard. The engine's per-slot
// history scheme makes every `_PREV` pair a role swap, not a copy (Framebuffers.cpp:33-53,
// ShFramebuffers_BindingsSwapped): the module adds the pairs' previous-role members (1, 4, 6, 8, 20,
// 24, 78, 80, 82, 84, 86, 92, 94, 96, 98, 100, 116, 120, 122) beside their current-role
// counterparts, each resolved through `Framebuffers::GetImageHandles(image, frameIndex)` so that
// every slot has a wrap for the current and the previous role. Three of the 80 are the god-rays
// hand-off images (63 GOD_RAYS, 64 GOD_RAYS_FILTERED, 89 Q2_GOD_RAYS_THROUGHPUT_DIST): they are in
// the union for the announcement and the restore discipline below, but only 64 is bound by a set of
// this module (the final composition's SRV at raw 188); the god-rays module is their writer and
// has to leave them in GENERAL. On the first two frames of a run the `_PREV` images are undefined
// (the engine clears nothing, Framebuffers.cpp:694-768); the shaders' bounds checks bound the
// effect exactly as in the legacy path.
//
// The image-state contract, per entry point: the engine leaves every framebuffer image in
// VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and a native wrap keeps no state between
// command lists (RhiTextureSource.h), so every call announces that state for all 80 wraps before
// its first binding and moves every image whose last use on the list was a sampled read back to
// UnorderedAccess (RenderGradientReproject restores its 17 sampled reads, the filtered `Render`
// path its 54-image CHAIN_RESTORE_IMAGES list, `RenderTaaU` its 3, and the unfiltered path its
// 21-image list). The announcement is true for images the earlier RHI passes (primary, direct,
// indirect, god rays) wrote or read on the same list, because each of those passes restores its
// own sampled reads to GENERAL as well. The two module-owned stand-ins are the 1x1x1 volumetric
// dummy and the nearest-filter sampler the TAAU history is sampled with (the engine's own choice
// for image 120, whose flags lack BILINEAR_SAMPLER; Framebuffers.cpp:813-815).
//
// What the legacy chain has around this sequence and the module deliberately does not record:
//  - `CmQ2Fog` (`ApplyFog`): there is no caller anywhere in the tree (Q2Denoiser.cpp:659-690 is
//    dead API; the only live fog is the per-segment `Q2_FOG_ACCUM` blend inside `CmQ2Atrous`);
//  - FSR2/FSR3/DLSS and the `BlitForEffects` tail: the TAAU writes UPSCALED_PING directly and the
//    present samples it, without the effect/sharpen chain (`rt_bloom` 0, `rt_sharpen` 0 and the
//    `rt_upscale_*` 0 default configuration make that equivalent, VulkanDevice.cpp:1100-1136);
//  - the raster world/emissive overlay (`Rasterizer::DrawToFinalImage`, VulkanDevice.cpp:1071):
//    the module records no overlay of its own, but `Render` opens the pass's exact legacy window
//    for the host through its trailing `pfnRasterOverlay` callback, between the checkerboard and
//    the prepare-final dispatch; with an empty callback the chain is what it was in A4.5 (the
//    `Render` contract below);
//  - god rays: the shadow map, the two `GodRays` dispatches and their host block are
//    `RhiShadowMapPass`'s and `RhiRtGodRaysPass`'s; the host records them before this module on the
//    same list. This module consumes only their output image 64 as a plain union SRV in the final
//    composition and restores it to UnorderedAccess, and it announces 63/64/89 with the rest of the
//    union: the god-rays module must leave all of them in GENERAL (its own restore discipline) or
//    this module's announcement names a state the images are not in.
//
// Host contract (the skeleton wires the passes; the module only records):
//  - Create takes the engine `Tonemapping` object (the exposure pair's per-slot buffers) and the
//    folder the engine blobs load from; the thirteen blobs above are read from it;
//  - the host order per traced frame, all on the frame context's open list of the same slot and
//    between the ray-tracing passes and the present:
//      shadow map -> primary -> god rays -> (filterEnabled: RenderGradientReproject) -> direct ->
//      indirect -> Render (the raster-overlay callback, if any, runs inside it between the
//      checkerboard and the prepare-final) -> RenderTaaU -> present;
//  - `filterEnabled` is `uniform->GetData()->fltEnable[0] >= 0.5f`, computed after the host wrote
//    the frame's uniform bytes: the adapter and the gradient chain it selects must match the
//    value the primary/direct/indirect raygens were dispatched with (`q2GetIsGradient` reads image
//    115 only when it is raised). `RenderGradientReproject` is called only while it is raised;
//  - 'width'/'height' are the render resolution and 'upscaledWidth'/'upscaledHeight' the engine's
//    upscaled size (`ResolutionState`): the half image (63), the 1/3 images (101-104, 111-116), the
//    upscaled images (29, 119, 120) and every render-sized image are wrapped to those extents, and
//    the same values have to be passed to all three calls of a frame. The TAAU dispatch runs over
//    the upscaled size; the shaders read `globalUniform.renderWidth/renderHeight/upscaledRenderWidth/
//    upscaledRenderHeight`, which the host's uniform bytes have to carry;
//  - the exposure params (the legacy `Tonemapping::CalculateExposure` host block) have to be
//    written before `Render`, because the histogram and the average consume them in the same list;
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images;
//    the next call re-reads the handles and re-wraps, so the pass survives a resize. The
//    tonemapping wraps and the module's stand-ins are not framebuffer-dependent and stay.
//
// The pass is a no-op until Create succeeded and while an input is missing (no framebuffers, no
// uniform, an image handle or extent the RHI cannot wrap, a zero upscaled size); every early
// return is quiet after the first warning. It is not thread-safe: each call uses the per-slot
// target of the frameIndex it is given, which is the engine's single-threaded per-slot frame model
// (RhiFrameContext).
class RhiRtComposePass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiRtComposePass();
    ~RhiRtComposePass();

    RhiRtComposePass(const RhiRtComposePass &other) = delete;
    RhiRtComposePass(RhiRtComposePass &&other) noexcept = delete;
    RhiRtComposePass &operator=(const RhiRtComposePass &other) = delete;
    RhiRtComposePass &operator=(RhiRtComposePass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queue every replaced wrap and set goes
    // through; 'pTonemapping' is the engine's tonemapping object (Tonemapping.h) whose per-slot
    // buffers this module wraps. None is owned; all have to outlive this object, and a null or
    // unusable one makes Create fail. 'pShaderFolderPath' is the folder ShaderManager loads the
    // engine blobs from, with the trailing separator; the thirteen blobs above are read from it
    // (the seven chain blobs are a hard requirement of A4.5). The pass logs through 'pfnPrint'.
    // Returns false and leaves the pass unusable if a shader, a layout, a pipeline, a
    // specialization, the TAAU sampler or the tonemapping wrap cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                const Tonemapping *pTonemapping,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // The pre-direct step of a denoised frame: the host calls it between
    // RhiRtPrimaryPass::Render and RhiRtDirectPass::Render, only when
    // `uniform->GetData()->fltEnable[0] >= 0.5f` (the legacy `Q2Denoiser::GradientReproject` gate,
    // VulkanDevice.cpp:1032). It records `CmQ2GradientReproject` over its 27-item set, dispatched
    // over `GWC(renderWidth / 3, 16) x GWC(renderHeight / 3, 16)`, and restores the 17 images it
    // sampled to UnorderedAccess. 'pFramebuffers'/'width'/'height'/'upscaledWidth'/
    // 'upscaledHeight'/'pUniformBuffer' are `Render`'s and have to be the same values; the call
    // may be the first one of the frame, so it resolves the wraps and sets on its own.
    void RenderGradientReproject(nvrhi::ICommandList *pCommandList,
                                 uint32_t frameIndex,
                                 const Framebuffers *pFramebuffers,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t upscaledWidth,
                                 uint32_t upscaledHeight,
                                 nvrhi::IBuffer *pUniformBuffer);

    // One call per frame, on the frame context's open command list of 'frameIndex', after the
    // frame's ray-tracing passes (RhiRtDirectPass::Render and RhiRtIndirectPass::Render) of the
    // same slot. 'pFramebuffers' is the engine's framebuffer registry, 'width'/'height' the render
    // resolution, 'upscaledWidth'/'upscaledHeight' the engine's upscaled size, and
    // 'pUniformBuffer' the engine's global uniform as a static constant-buffer wrap (the same wrap
    // the primary and direct passes take). 'filterEnabled' is the denoiser switch: true selects
    // the adapter -> ASVGF chain -> interleave shape, false the adapter -> interleave shape
    // (the legacy `fltEnable[0] < 0.5` short-circuit). The exposure params of the two luminance
    // dispatches are the host's: it has to write them into the engine tonemapping object before
    // this call (see the class comment); RenderTaaU follows after it.
    //
    // 'pfnRasterOverlay' is the host's raster-overlay window: when it is non-empty it is invoked
    // exactly once per recorded call, on this same open list, between the checkerboard dispatch
    // that writes FINAL (28) and SCREEN_EMISSION (62) and the prepare-final dispatch that reads
    // them - the legacy `Rasterizer::DrawToFinalImage` position (VulkanDevice.cpp:1071, between
    // ProcessCheckerboard at :1066 and Finalize at :1085). The callback records the raster overlay
    // into FINAL + SCREEN_EMISSION and, for the authored emissive blends, into
    // PRIMARY_TO_REFL_REFR (25), which the final composition then reads in checkerboard space
    // (the `RsWorld.frag.hlsl:149-157` write). The compose guarantees for the callback: the
    // checkerboard has just resolved the HDR colour into FINAL, the emission into SCREEN_EMISSION
    // and the fog into ACID_FOG (the pre-tonemap values the prepare-final consumes,
    // CmCheckerboard.comp.hlsl:86-98), the exposure pair has already written the slot's
    // tonemapping buffer, and everything the callback records is ordered on this list before the
    // prepare-final dispatches. The callback has to carry the state discipline of the images it
    // touches itself: it must leave FINAL, SCREEN_EMISSION and - when it writes it -
    // PRIMARY_TO_REFL_REFR in the engine's GENERAL state by announcing UnorderedAccess before it
    // binds them and setting them back to UnorderedAccess after the overlay's last use of each,
    // exactly the present's pattern (NvrhiFrameSkeleton.cpp:626-648) - no matter whether the
    // callback carries the transitions on its own wraps of 28/62/25 or on the compose's
    // `GetFinalTexture` wrap for 28. The callback's render-target -> UnorderedAccess transition is
    // the barrier that makes its attachment writes visible to the compute reads; the compose
    // guarantees that barrier is committed on this list before the prepare-final dispatch,
    // because that dispatch's state set commits the queued barriers (vulkan-compute.cpp:145). The
    // compose's own announcement (the top of every entry point) and its restore lists keep naming
    // UnorderedAccess for its wraps; the callback's restore is what keeps that claim true
    // physically, and the prepare-final then reads FINAL through its storage image and samples
    // SCREEN_EMISSION as before. A callback that ends with the images still bound as render
    // targets leaves their physical layouts outside the announced General state: through the
    // callback's own wraps the compose cannot see the change (its wrap tracked UnorderedAccess all
    // along), and the prepare-final read is undefined. An empty callback records nothing and
    // changes nothing about the chain.
    // A callback and not two public entry points, because the overlay's only contract with this
    // module is that one window: splitting the pre-TAAU chain in two would hand the host the
    // module's intermediate wrap and set state to sequence around, while one `Render` per frame
    // stays atomic (a5c_overlay_recon.md §2.3). Like the chain, the call is skipped when an input
    // is missing and the callback is not invoked.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height,
                uint32_t upscaledWidth,
                uint32_t upscaledHeight,
                bool filterEnabled,
                nvrhi::IBuffer *pUniformBuffer,
                const std::function<void(nvrhi::ICommandList *)> &pfnRasterOverlay = {});

    // The upscaler step: the host calls it after `Render` of the same frame slot and before the
    // present. It records `CmQ2TAAU` over its 6-item set - 2 storage images (29 UPSCALED_PING,
    // 119 Q2_TAA_HISTORY), 3 sampled images (28 FINAL, 31 MOTION_DLSS, 120
    // Q2_TAA_HISTORY_PREV) and the module's nearest sampler at raw 368 - dispatched over
    // `GWC(upscaledWidth, 16) x GWC(upscaledHeight, 16)`, and restores the three sampled images to
    // UnorderedAccess so the next frame's passes find the engine's GENERAL layout. The arguments
    // have to be `Render`'s; 'upscaledWidth'/'upscaledHeight' are the dispatch size.
    void RenderTaaU(nvrhi::ICommandList *pCommandList,
                    uint32_t frameIndex,
                    const Framebuffers *pFramebuffers,
                    uint32_t width,
                    uint32_t height,
                    uint32_t upscaledWidth,
                    uint32_t upscaledHeight,
                    nvrhi::IBuffer *pUniformBuffer);

    // The FINAL image of 'frameIndex' as the wrapped RHI texture, or null while that slot has no
    // prepared target (the engine has not created its framebuffers yet, or the last call failed).
    // The present samples this handle when the TAAU is not recorded; after a `RenderTaaU` it
    // samples `GetUpscaledTexture` instead. The caller has to rebuild whatever set references the
    // returned pointer when it changes, because a re-created engine framebuffer or a size change
    // replaces the wrap. The handle is owned by this pass; the caller must not release it, and it
    // has to call Render() for 'frameIndex' before it samples what the handle points to.
    //
    // State discipline for the caller (a42b_recon.md §3.3): FINAL rests in UnorderedAccess after
    // the final composition and Render restores it there, so the present has to
    // beginTrackingTextureState(final, AllSubresources, UnorderedAccess) before its draw and
    // setTextureState(final, AllSubresources, UnorderedAccess) after it - the same pair the
    // skeleton performs for ALBEDO (NvrhiFrameSkeleton.cpp:560-564) - or the next frame's
    // checkerboard UAV write runs against a read-only layout.
    nvrhi::ITexture *GetFinalTexture(uint32_t frameIndex) const;

    // The TAAU output (29 UPSCALED_PING) of 'frameIndex' as the wrapped RHI texture, or null while
    // that slot has no prepared target. The present samples this handle after `RenderTaaU`; the
    // same ownership, rebuild-on-change and state rules as `GetFinalTexture` apply (the TAAU
    // leaves 29 in UnorderedAccess, and its own sampled reads are restored before the call
    // returns, so the present's beginTrackingTextureState/ setTextureState pair is all the state
    // churn the image needs).
    nvrhi::ITexture *GetUpscaledTexture(uint32_t frameIndex) const;

    // Drops every slot's image wraps and the twelve sets over them and the per-slot uniform sets on
    // the shared uniform layout, and retires them through the frame context's queue. The caller has
    // to call it before the engine destroys its framebuffer images (the Framebuffers::PrepareForSize
    // path) - otherwise the wraps reference destroyed VkImages. The next entry point re-reads the
    // handles and re-wraps, so the pass survives a resize without a second Create. The
    // tonemapping wraps, the TAAU sampler and the module-owned stand-ins are not
    // framebuffer-dependent and stay.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: the images of the chain are engine framebuffer images
    // (ping-ponged ones and the `_PREV` role pairs included), so neither the wraps over them nor
    // the sets can be shared across slots. The uniform set is kept per slot with the rest, under
    // the same pointer-change rule the other passes use.
    struct Target
    {
        // The 80 engine images (the .cpp's COMPOSE_IMAGES table) the slot currently wraps and the
        // twelve sets over them. The handles are kept in the form the entry point received them,
        // not as VkImages, because they are what the change detection compares; a change in any of
        // them or in any of the three sizes means the engine re-created the framebuffers (or the
        // resolution changed) and the wraps and the sets have to follow.
        uint64_t imageHandles[80] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t upscaledWidth = 0;
        uint32_t upscaledHeight = 0;
        nvrhi::TextureHandle engineTextures[80];
        nvrhi::BindingSetHandle gradientReprojectSet;
        nvrhi::BindingSetHandle adapterSet;
        nvrhi::BindingSetHandle gradientImgSet;
        nvrhi::BindingSetHandle gradientAtrousSet;
        nvrhi::BindingSetHandle temporalSet;
        nvrhi::BindingSetHandle atrousLfSet;
        nvrhi::BindingSetHandle atrousSet;
        nvrhi::BindingSetHandle interleaveSet;
        nvrhi::BindingSetHandle histogramSet;
        nvrhi::BindingSetHandle checkerboardSet;
        nvrhi::BindingSetHandle prepareFinalSet;
        nvrhi::BindingSetHandle taauSet;

        // Set 1: the engine's global uniform never changes, but it is an argument here, so the
        // set follows the pointer the way the other passes' uniform sets do.
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // The shared preamble of the three entry points: validates the arguments (including the command
    // list, whose null the entry points would otherwise dereference in AnnounceFrameImages),
    // resolves the 80 image handles and extents of 'frameIndex', retires and re-wraps everything
    // when the engine re-created an image or any of the three sizes changed, prepares the twelve
    // sets and the uniform set. Returns the slot's target, or null when the call cannot be
    // recorded - which the caller treats as a silent skip.
    Target *PrepareFrame(nvrhi::ICommandList *pCommandList,
                         uint32_t frameIndex,
                         const Framebuffers *pFramebuffers,
                         uint32_t width,
                         uint32_t height,
                         uint32_t upscaledWidth,
                         uint32_t upscaledHeight,
                         nvrhi::IBuffer *pUniformBuffer);

    // Declares the engine's resting state (UnorderedAccess, i.e. GENERAL) for all 80 wraps on the
    // list, before the entry point's first binding touches one. A repeated call is a no-op: the
    // declaration writes the tracked state, it emits no barrier. The declaration is a claim about
    // the physical layout each earlier pass left behind; the god-rays images (63/64/89) rest on the
    // god-rays module's own restore discipline (the class comment).
    void AnnounceFrameImages(nvrhi::ICommandList *pCommandList, const Target &target) const;

    // Queues a same-state UnorderedAccess requirement for one union image - the per-iteration and
    // restore discipline of the class comment: NVRHI emits the same-state UAV barrier (or the
    // read-only -> GENERAL transition) and commits it before the next state set or at the list's
    // close. A no-op when the image is not part of the union (which the tables' static assertions
    // make unreachable).
    void RequireImageUnorderedAccess(nvrhi::ICommandList *pCommandList,
                                     const Target &target,
                                     FramebufferImageIndex image) const;

    // Builds the twelve set-0 sets over the module's exact layouts when they are missing (the
    // first frame of a slot or after a framebuffer re-create). Returns false when a binding cannot
    // be filled - a table/image mismatch, which the module's static assertions make unreachable.
    bool PrepareFramebufferSets(Target &target);

    // Set 1 over the module's compute uniform layout, rebuilt when the pointer changed. Returns
    // false when the buffer is not the static constant-buffer wrap the shader's
    // `ConstantBuffer<ShGlobalUniform>` requires.
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);

    // One `setComputeState` + `dispatch` for one pass: the sets are added in the pipeline's layout
    // order (the backend binds the list positionally, vulkan-resource-bindings.cpp:940-958), which
    // is the pass's framebuffer set at 0, the shared uniform set at 1, and the tonemapping/empty/
    // volumetric sets where the pass declares them. 'pPushConstant' is the iterative passes' 4-byte
    // iteration index: when it is non-null it is written between the state and the dispatch, the
    // order `setPushConstants` requires (nvrhi.h:3430-3440).
    void RecordDispatch(nvrhi::ICommandList *pCommandList,
                        nvrhi::IComputePipeline *pPipeline,
                        std::initializer_list<nvrhi::IBindingSet *> sets,
                        uint32_t groupsX,
                        uint32_t groupsY,
                        uint32_t groupsZ,
                        const uint32_t *pPushConstant = nullptr);

    // Retires the slot's twelve framebuffer sets. Used by the framebuffer re-create path and by
    // ReleaseTargets.
    void ReleaseFramebufferSets(Target &target);
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, which outlives this object.
    rhi::RhiFrameContext *frameContext = nullptr;

    // The thirteen engine blobs. Only `CmQ2Atrous` declares a specialization constant
    // (`SpecId 0`); its four iterations are the derived `atrousIterationShaders` below.
    nvrhi::ShaderHandle gradientReprojectShader;
    nvrhi::ShaderHandle adapterShader;
    nvrhi::ShaderHandle gradientImgShader;
    nvrhi::ShaderHandle gradientAtrousShader;
    nvrhi::ShaderHandle temporalShader;
    nvrhi::ShaderHandle atrousLfShader;
    nvrhi::ShaderHandle atrousShader;
    nvrhi::ShaderHandle atrousIterationShaders[4];
    nvrhi::ShaderHandle interleaveShader;
    nvrhi::ShaderHandle histogramShader;
    nvrhi::ShaderHandle averageShader;
    nvrhi::ShaderHandle checkerboardShader;
    nvrhi::ShaderHandle prepareFinalShader;
    nvrhi::ShaderHandle taauShader;

    // The layouts this module owns: the twelve exact set-0 layouts, the shared compute uniform
    // layout, the push-constant layout of the two push-constant iterations, the two tonemapping
    // layouts (UAV for the exposure pair, SRV for the final composition), the empty layout that
    // fills the average's set 0 and the prepare-final's set 3, and the volumetric layout. Every RT
    // pass's layouts are AllRayTracing visibility and cannot be reused.
    nvrhi::BindingLayoutHandle gradientReprojectFramebufferLayout;
    nvrhi::BindingLayoutHandle adapterFramebufferLayout;
    nvrhi::BindingLayoutHandle gradientImgFramebufferLayout;
    nvrhi::BindingLayoutHandle gradientAtrousFramebufferLayout;
    nvrhi::BindingLayoutHandle temporalFramebufferLayout;
    nvrhi::BindingLayoutHandle atrousLfFramebufferLayout;
    nvrhi::BindingLayoutHandle atrousFramebufferLayout;
    nvrhi::BindingLayoutHandle interleaveFramebufferLayout;
    nvrhi::BindingLayoutHandle histogramFramebufferLayout;
    nvrhi::BindingLayoutHandle checkerboardFramebufferLayout;
    nvrhi::BindingLayoutHandle prepareFinalFramebufferLayout;
    nvrhi::BindingLayoutHandle taauFramebufferLayout;
    nvrhi::BindingLayoutHandle uniformLayout;
    nvrhi::BindingLayoutHandle pushConstantLayout;
    nvrhi::BindingLayoutHandle tonemappingUavLayout;
    nvrhi::BindingLayoutHandle tonemappingSrvLayout;
    nvrhi::BindingLayoutHandle emptyLayout;
    nvrhi::BindingLayoutHandle volumetricLayout;

    // The sixteen compute pipelines: one per chain pass, four specializations for the HF atrous,
    // and A4.4's five. The two push-constant pipelines carry the module's pushConstantLayout as
    // their last layout; no set is created or bound for it, because it has no descriptors
    // (RhiSkyPass uses the same shape).
    nvrhi::ComputePipelineHandle gradientReprojectPipeline;
    nvrhi::ComputePipelineHandle adapterPipeline;
    nvrhi::ComputePipelineHandle gradientImgPipeline;
    nvrhi::ComputePipelineHandle gradientAtrousPipeline;
    nvrhi::ComputePipelineHandle temporalPipeline;
    nvrhi::ComputePipelineHandle atrousLfPipeline;
    nvrhi::ComputePipelineHandle atrousPipelines[4];
    nvrhi::ComputePipelineHandle interleavePipeline;
    nvrhi::ComputePipelineHandle histogramPipeline;
    nvrhi::ComputePipelineHandle averagePipeline;
    nvrhi::ComputePipelineHandle checkerboardPipeline;
    nvrhi::ComputePipelineHandle prepareFinalPipeline;
    nvrhi::ComputePipelineHandle taauPipeline;

    // The module's per-slot wrap of the engine's tonemapping buffer and the two sets over it: the
    // UAV set the histogram and the average bind, and the SRV set the final composition binds. The
    // engine buffer is created once per slot and never re-created, so all three follow the pass's
    // lifetime, not the framebuffers'.
    nvrhi::BufferHandle tonemappingBuffers[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle tonemappingUavSets[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle tonemappingSrvSets[MAX_FRAMES_IN_FLIGHT];

    // The module-owned stand-ins: the real empty set, the 1x1x1 RGBA16F volumetric dummy with its
    // sampler and set, and the nearest-filter sampler the TAAU binds for the history (the engine
    // binds its nearest sampler for image 120, Framebuffers.cpp:813-815). The dummy is cleared to
    // zero once, on the first list that binds it.
    nvrhi::TextureHandle volumetricDummyTexture;
    nvrhi::SamplerHandle volumetricDummySampler;
    nvrhi::BindingSetHandle volumetricSet;
    nvrhi::BindingSetHandle emptySet;
    bool volumetricDummyCleared = false;
    nvrhi::SamplerHandle taauHistorySampler;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // One-shot warnings for the inputs that can legitimately be missing for a few frames or are a
    // permanent host-side mistake.
    bool warnedMissingFramebuffers = false;
    bool warnedUnexpectedSize = false;
    bool warnedMissingUniform = false;
    bool warnedBadUniform = false;
    bool warnedBadTable = false;

    bool created = false;
};

}
