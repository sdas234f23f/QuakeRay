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

namespace vkpt
{

class Framebuffers;
class Tonemapping;

namespace rhi
{
class RhiFrameContext;
}

// The RHI module of the composed traced frame: the engine's `flt_enable = 0` compositing chain
// `CmQ2Adapter` -> `CmQ2Interleave` -> `CmLuminanceHistogram` -> `CmLuminanceAvg` ->
// `CmCheckerboard` -> `CmPrepareFinal` as six compute pipelines on the frame's command list,
// ending in the engine's display-referred FINAL image (28) the present shows. The three leading
// passes and the checkerboard are the A4.2b "route-(b) compose preview" (a42b_recon.md §3); the
// exposure pair and the final composition close the chain in A4.4.
//
// The chain is what the legacy frame runs with the denoiser disabled: `Q2Denoiser::Denoise`
// records the adapter and, because `fltEnable[0] < 0.5`, immediately calls
// `InterleaveCheckerboard` and returns (Q2Denoiser.cpp:412-436, :600-623); the interleave writes
// PRE_FINAL (27); `Tonemapping::CalculateExposure` (Tonemapping.cpp:86-233) histograms that
// PRE_FINAL and writes the frame's curve and adapted luminance into the slot's `ShTonemapping`
// buffer; `ImageComposition::PrepareForRaster` resolves the checkerboard into FINAL (28) with its
// SCREEN_EMISSION (62) and ACID_FOG (60) companions (ImageComposition.cpp:93-98, :163-202, called
// at VulkanDevice.cpp:1061, :1066); and `ImageComposition::Finalize` ->
// `CmPrepareFinal` (VulkanDevice.cpp:1085) applies the tone curve, the level fog, the debug view,
// the dither and the output clamp, and writes FINAL and BLOOM_INPUT (65). The adapter composites
// the raw ReSTIR signal into Q2_COLOR (117) in its `fltEnable[0] < 0.5` branch, and the
// checkerboard pass writes FINAL with the two companions.
//
// What the legacy chain has around that sequence and this module deliberately does not:
//  - the **raster world/emissive overlay** (`Rasterizer::DrawToFinalImage`, VulkanDevice.cpp:1071,
//    inside `if (!drawInfo.disableRasterization)`) is drawn into FINAL between the checkerboard
//    and the final composition; reproducing it means a second raster module with
//    `GetRasterPipelines`/`GetWorldRenderPass`, the `PrepareForFinal` depth copy, the
//    tonemapping/volumetric/texture-table sets and the lens-flare cull. The A4.4 skip is
//    deliberate: FINAL means the traced G-buffer (primary + direct + indirect + the
//    adapter/interleave/checkerboard resolve), without the viewmodel/sprites/particles/lens
//    flares;
//  - the light statistics stayed with the direct pass (RhiRtDirectPass); the ASVGF chain (A4.5),
//    god rays, bloom, sharpening/post effects, FSR/DLSS/TAAU and the HUD tail
//    (VulkanDevice.cpp:1088-1229) are later increments.
//
// The six pipelines and their exact sets (measured 2026-09-25 with `spirv-dis` over the engine
// blobs in vkpt\Build - the blobs the engine's build produces and the legacy ShaderManager loads,
// and the blobs this module loads too):
//   CmQ2Adapter    - set 0: 5 storage images (73 Q2ColorLF_SH, 75 Q2ColorLF_COCG, 77 Q2ColorHF,
//                    79 Q2ColorSpec, 117 Q2Color) and 12 sampled images (0 ALBEDO, 2 IS_SKY,
//                    3 NORMAL, 7 METALLIC_ROUGHNESS, 14 UNFILTERED_DIRECT, 15 UNFILTERED_SPECULAR,
//                    16-18 UNFILTERED_INDIRECT_S_H_R/G/B, 26 THROUGHPUT, 88 Q2_TRANSPARENT,
//                    90 Q2_FOG_ACCUM); set 1: the engine's global uniform at raw binding 0.
//   CmQ2Interleave - set 0: 1 storage image (27 PRE_FINAL) and 2 sampled images (26 THROUGHPUT,
//                    117 Q2Color); set 1: the global uniform.
//   CmLuminanceHistogram - set 0: 1 sampled image (27 PRE_FINAL at raw 151) and no storage image;
//                    set 1: the global uniform; set 2: `RWStructuredBuffer<ShTonemapping>` at raw
//                    binding 0.
//   CmLuminanceAvg - **no set 0**; set 1: the global uniform; set 2: the same read-write
//                    tonemapping buffer. The pipeline still declares a set 0, an empty layout that
//                    an empty set fills, so the uniform and tonemapping sets keep positions 1/2.
//   CmCheckerboard - set 0: 3 storage images (28 FINAL, 60 ACID_FOG, 62 SCREEN_EMISSION) and 4
//                    sampled images (26 THROUGHPUT, 27 PRE_FINAL, 59 ACID_FOG_RT, 61
//                    SCREEN_EMIS_RT); set 1: the global uniform.
//   CmPrepareFinal - set 0: 5 storage images (16-18 UNFILTERED_INDIRECT_S_H_R/G/B, 28 FINAL, 65
//                    BLOOM_INPUT) and 11 sampled images (0 ALBEDO, 3 NORMAL, 9 DEPTH_WORLD, 13
//                    MOTION, 14 UNFILTERED_DIRECT, 15 UNFILTERED_SPECULAR, 25 PRIMARY_TO_REFL_REFR,
//                    26 THROUGHPUT, 60 ACID_FOG, 62 SCREEN_EMISSION, 64 GOD_RAYS_FILTERED); set 1:
//                    the global uniform; set 2: the tonemapping buffer as a read-only
//                    `StructuredBuffer<ShTonemapping>`; set 3: the LPM block, which the shipped
//                    blob does not reference at all and which the module occupies with an empty
//                    layout and an empty set; set 4: the sampled volumetric volume at raw 1 and
//                    its sampler at raw 2.
// All are `[numthreads(16, 16, 1)]` except `CmLuminanceAvg`, which is `[numthreads(128, 1, 1)]`
// and is dispatched once; the rest use the legacy `Utils::GetWorkGroupCount(w, 16) x ...(h, 16)`
// (Q2Denoiser.cpp:400-401, :606-607, ImageComposition.cpp:198-199; the helper is `1 + ceil(size /
// group)`, Utils.cpp:319-328; Tonemapping.cpp:176-179) and their own
// `renderWidth`/`renderHeight` bounds check discards the extra workgroup.
//
// Two deviations from the shipped `CmPrepareFinal` blob's set-0 declaration:
//  - the blob declares 17 set-0 items while the table fed to the layout has 16: the 17th is the
//    FINAL sampled view (raw 152) over the same image (28) the storage-image item writes. NVRHI
//    fixes an SRV descriptor to eShaderReadOnlyOptimal and a UAV descriptor to eGeneral
//    (vulkan-resource-bindings.cpp:404-421, :439-451) and applies a set's state requirements in
//    item order (vulkan-state-tracking.cpp:29-63), so one of the two live descriptors would always
//    name a layout the image is not in. The A4.4 S1 shader fix reads FINAL through its storage
//    image in both twins, which dead-strips 152; this module's prepare-final set has to be wired
//    only after that fix lands, and it declares the expected 16 items;
//  - the LPM set (set 3) and the volumetric storage/sampled-prev/illumination bindings 0/3-7 are
//    dead in the shipped blob, so the module declares only what the blob references.
//
// The sets cannot reuse the RT passes' layouts: every RT set is `AllRayTracing` visibility
// (nvrhi.h:883), which has no Compute bit, and the pinned NVRHI skips a binding layout whose
// visibility lacks the shader's stage when it validates a compute pipeline
// (validation-device.cpp:1002-1003). The module therefore creates its own layouts with
// `ShaderType::Compute` visibility and its own compute uniform layout at raw binding 0. The
// set-0 offsets turn the engine's raw bindings into NVRHI slots (RhiPipeline.h): the storage
// images sit at raw binding = image index (`ShFramebuffers_Bindings`, the identity) with offset 0,
// the sampled views at raw binding = 124 + image index (`ShFramebuffers_Sampled_Bindings`) with
// offset 124. The set-2 tonemapping items sit at raw binding 0
// (`BINDING_LUM_HISTOGRAM`) with a matching offset; set 4's sampled volume is at raw
// `BINDING_VOLUMETRIC_SAMPLED` and its sampler at raw `BINDING_VOLUMETRIC_SAMPLER`, both with
// offset 0. The framebuffer layouts are partial in the same way the RT passes' are: one item per
// binding the blob declares, no union, because NVRHI refuses a set that leaves a layout item
// unfilled (validation-device.cpp:1855-1871), so adapter, interleave, histogram, checkerboard and
// prepare-final own a 17-, a 3-, a 1-, a 7- and a 16-item layout, and each pipeline declares its
// own set 0 beside the shared set 1 and the tonemapping/empty/volumetric sets.
//
// 27 engine images are wrapped per slot (the .cpp's COMPOSE_IMAGES table), each once, and the five
// framebuffer sets share the wraps, so the pass-to-pass hand-offs (117 written by the adapter and
// read by the interleave; 27 written by the interleave and read by the histogram and the
// checkerboard; 60/62 written by the checkerboard and read by the final composition) are covered
// by NVRHI's own UAV -> SRV -> UAV transitions on one wrap. The three UNFILTERED_INDIRECT_S_H
// images (16-18) are written by the A4.3 indirect pass (RhiRtIndirectPass), which runs earlier on
// the same command list, so the adapter's 140/141/142 sampled bindings and the prepare-final's
// storage bindings read and write real data. The invariant this module relies on is the host
// order primary -> direct -> indirect -> compose: without the indirect pass the adapter would
// sample the images' engine-initialized contents.
//
// The module-owned stand-ins (the two inputs no RHI pass can produce yet):
//  - image 64 GOD_RAYS_FILTERED has no writer under `rhiframe` (the god-rays pass runs only from
//    the legacy Render) and the engine creates every framebuffer image with undefined contents
//    (Framebuffers.cpp:709-768), so the final composition's raw-188 read binds a module-owned
//    render-sized zero substitute of the image's B10G11R11 format instead of the engine image (the
//    A4.2b zero-texture mechanism). The read is a `.Load(int3(pix, 0))` at render coordinates
//    (CmPrepareFinal.comp.hlsl:446-449), so the substitute has to be render-sized; it is cleared
//    to zero once per wrap lifetime and re-created on a size change. The gate has to force
//    `rt_godrays 0` as well, or the legacy reference would carry shafts this module does not
//    reproduce;
//  - the set-4 volumetric volume and sampler: `applyVolumetrics` returns before touching them when
//    `volumeEnableType == VOLUME_ENABLE_NONE` or `coreQ2RTX != 0`
//    (CmPrepareFinal.comp.hlsl:269-280), and `FillUniform` forces `coreQ2RTX = 1`
//    (VulkanDevice.cpp:330), so a module-owned 1x1x1 RGBA16F 3D texture and the RHI engine sampler
//    are behaviourally exact while that holds.
//
// The image-state contract: the engine leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL
// (= NVRHI's UnorderedAccess) and a native wrap keeps no state between command lists
// (RhiTextureSource.h), so every list announces UnorderedAccess for all 27 wraps before the first
// pass and moves every image whose last use is a sampled read back to UnorderedAccess after the
// final composition (the A4.2a discipline: an unannounced first SRV use would discard the
// contents, and a finish in SHADER_READ_ONLY_OPTIMAL would mismatch the next frame's UAV
// announcements). Since the final composition reads ALBEDO, NORMAL, DEPTH_WORLD, MOTION, the two
// direct images, PRIMARY_TO_REFL_REFR, THROUGHPUT, ACID_FOG, SCREEN_EMISSION and the god-rays
// substitute, and reads nothing after it, its set decides the reads that have to be restored:
// 18 images (the .cpp's COMPOSE_RESTORE_IMAGES). The images that end the chain as the last pass's
// UAV outputs (16-18, 28, 65) or as an earlier UAV whose reader comes before the final
// composition (73, 75, 77, 79) already rest in GENERAL; the next frame's indirect pass (16-18)
// and checkerboard (28) find them there through their own UnorderedAccess announcements, which is
// why those images are not in the restore list any more.
//
// Host contract (the skeleton wires the pass, the pass only records):
//  - Create takes the engine `Tonemapping` object: the module wraps its per-slot
//    `ShTonemapping` buffers itself, because the histogram and the average bind them as
//    `RWStructuredBuffer<ShTonemapping>` through an NVRHI UAV-capable native wrap, and the final
//    composition binds the same wrap as a read-only `StructuredBuffer`. The raster mode's separate
//    SRV wrap (NvrhiFrameSkeleton::PrepareWorld) is never alive at the same time, and a native
//    buffer wrap carries no image layout, so the two can coexist for the traced mode. The engine
//    buffer is created once per slot and never re-created, so the wraps and the sets over them are
//    created in Create and released only with the pass;
//  - the exposure params (the legacy `Tonemapping::CalculateExposure` host block,
//    Tonemapping.cpp:95-128) have to be written by the host before this pass records the two
//    luminance dispatches: the module has no access to the engine's `GlobalUniform`, which the
//    params block reads for `frameTime`, so this pass deliberately takes no exposure-bias or
//    contrast argument and the host calls the engine object's host-only params entry point
//    instead. The histogram and the average fill the same slot's curve, adapted luminance and
//    average luminance on the GPU, and the final composition reads them later in the same list,
//    so a host stand-in for `avgLuminance` must not be written in the traced mode;
//  - Record after the frame's ray-tracing passes (RhiRtDirectPass::Render and
//    RhiRtIndirectPass::Render) on the same list and before the present: the adapter samples the
//    G-buffer, the direct images, the indirect SH the indirect pass wrote and the throughput the
//    primary/direct chain wrote; the exposure pair reads the PRE_FINAL the interleave just wrote;
//    the final composition reads the FINAL and the companions the checkerboard just wrote;
//  - 'width'/'height' are the render resolution the shaders read as `globalUniform.renderWidth/
//    renderHeight` and the images are sized to; the dispatch has to run over the same numbers, or
//    pixels beyond the smaller one keep their previous content;
//  - `globalUniform.fltEnable[0]` has to stay below 0.5 (the adapter's unfiltered branch); with
//    the filtered branch selected the adapter writes the ASVGF channel images instead of Q2_COLOR
//    and the image would show stale color. This is the skeleton's forced switch, shared with
//    RhiRtDirectPass;
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images;
//    the next Render re-reads the handles and re-wraps, so the pass survives a resize. The
//    module-owned stand-ins follow the pass's own lifetime; the god-rays substitute is re-sized by
//    the next Render.
//
// The pass is a no-op until Create succeeded and while an input is missing (no framebuffers, no
// uniform, an image handle or extent the RHI cannot wrap); every early return is quiet after the
// first warning. It is not thread-safe: Render uses the per-slot target of the frameIndex it is
// given, which is the engine's single-threaded per-slot frame model (RhiFrameContext).
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
    // engine blobs from, with the trailing separator; the six compose blobs above are read from
    // it. The pass logs through 'pfnPrint'. Returns false and leaves the pass unusable if a shader,
    // a layout, a pipeline, the tonemapping wrap or a stand-in cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                const Tonemapping *pTonemapping,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // One call per frame, on the frame context's open command list of 'frameIndex', after the
    // frame's ray-tracing passes (RhiRtDirectPass::Render and RhiRtIndirectPass::Render) of the
    // same slot. 'pFramebuffers' is the engine's framebuffer registry, 'width'/'height' the render
    // resolution, and 'pUniformBuffer' the engine's global uniform as a static constant-buffer wrap
    // (the same wrap the primary and direct passes take). The exposure params of the two luminance
    // dispatches are the host's: it has to write them into the engine tonemapping object before
    // this call (see the class comment).
    //
    // What is recorded, in the legacy order: the wraps of the 27 engine images (created on first
    // use, re-created when the engine re-created an image or the size changed; the replaced wraps
    // and the sets over them go through the frame context's retire queue), the render-sized
    // god-rays zero substitute, the per-slot sets, the UnorderedAccess announcement of the 27
    // images, then six `setComputeState` + `dispatch` pairs - adapter, interleave, histogram,
    // average (1 x 1 x 1), checkerboard, prepare-final - and the UnorderedAccess restore of the 18
    // images that end in the read-only state.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height,
                nvrhi::IBuffer *pUniformBuffer);

    // The FINAL image of 'frameIndex' as the wrapped RHI texture, or null while that slot has no
    // prepared target (the engine has not created its framebuffers yet, or the last Render failed).
    // The present (stream S4) samples this handle instead of the sky pass's ALBEDO: the caller has
    // to rebuild whatever set references it when the returned pointer changes, because a re-created
    // engine framebuffer or a size change replaces the wrap. The handle is owned by this pass; the
    // caller must not release it, and it has to call Render() for 'frameIndex' before it samples
    // what the handle points to.
    //
    // State discipline for the caller (a42b_recon.md §3.3): FINAL rests in UnorderedAccess after
    // the final composition and Render restores it there, so the present has to
    // beginTrackingTextureState(final, AllSubresources, UnorderedAccess) before its draw and
    // setTextureState(final, AllSubresources, UnorderedAccess) after it - the same pair the
    // skeleton performs for ALBEDO (NvrhiFrameSkeleton.cpp:560-564) - or the next frame's
    // checkerboard UAV write runs against a read-only layout.
    nvrhi::ITexture *GetFinalTexture(uint32_t frameIndex) const;

    // Drops every slot's image wraps and the five sets over them and the per-slot uniform sets on
    // the shared uniform layout, and retires them through the frame context's queue. The caller has
    // to call it before the engine destroys its framebuffer images (the Framebuffers::PrepareForSize
    // path) - otherwise the wraps reference destroyed VkImages. The next Render re-reads the
    // handles and re-wraps, so the pass survives a resize without a second Create. The
    // tonemapping wraps and the module-owned stand-ins are not framebuffer-dependent and stay.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: the images of the compose chain are engine framebuffer
    // images (ping-ponged ones like ALBEDO and Q2_COLOR included), so neither the wraps over them
    // nor the sets can be shared across slots. The uniform set is kept per slot with the rest,
    // under the same pointer-change rule the other passes use.
    struct Target
    {
        // The 27 engine images (the .cpp's COMPOSE_IMAGES table) the slot currently wraps and the
        // five sets over them. The handles are kept in the form Render received them, not as
        // VkImages, because they are what the change detection compares; a change in any of them
        // or in the size means the engine re-created the framebuffers and the wraps and the sets
        // have to follow.
        uint64_t imageHandles[27] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle engineTextures[27];
        nvrhi::BindingSetHandle adapterSet;
        nvrhi::BindingSetHandle interleaveSet;
        nvrhi::BindingSetHandle histogramSet;
        nvrhi::BindingSetHandle checkerboardSet;
        nvrhi::BindingSetHandle prepareFinalSet;

        // Set 1: the engine's global uniform never changes, but it is a Render argument here, so
        // the set follows the pointer the way the other passes' uniform sets do.
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Builds the five set-0 sets over the module's exact layouts when they are missing (the first
    // frame of a slot, after a framebuffer re-create and after a god-rays substitute replacement).
    // Returns false when a binding cannot be filled - a table/image mismatch, which the module's
    // static assertions make unreachable.
    bool PrepareFramebufferSets(Target &target);

    // The render-sized zero stand-in for the god-rays image 64, which no RHI pass writes. It is
    // size-keyed, cleared to zero once per wrap lifetime, and its replacement retires every slot's
    // framebuffer sets (the prepare-final set references it). Returns false when it cannot be
    // created, which skips the frame.
    bool PrepareGodRaysStandIn(nvrhi::ICommandList *pCommandList, uint32_t width, uint32_t height);

    // Set 1 over the module's compute uniform layout, rebuilt when the pointer changed. Returns
    // false when the buffer is not the static constant-buffer wrap the shader's
    // `ConstantBuffer<ShGlobalUniform>` requires.
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);

    // One `setComputeState` + `dispatch` for one of the six passes: the sets are added in the
    // pipeline's layout order (the backend binds the list positionally,
    // vulkan-resource-bindings.cpp:940-958), which is the pass's framebuffer set at 0, the shared
    // uniform set at 1, and the tonemapping/empty/volumetric sets where the pass declares them.
    void RecordDispatch(nvrhi::ICommandList *pCommandList,
                        nvrhi::IComputePipeline *pPipeline,
                        std::initializer_list<nvrhi::IBindingSet *> sets,
                        uint32_t groupsX,
                        uint32_t groupsY,
                        uint32_t groupsZ);

    // Retires the slot's five framebuffer sets. Used by the framebuffer re-create path and by the
    // god-rays substitute replacement, whose new texture the prepare-final set has to reference.
    void ReleaseFramebufferSets(Target &target);
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, which outlives this object.
    rhi::RhiFrameContext *frameContext = nullptr;

    // The six engine blobs. None of them declares a specialization constant (measured: no
    // `OpSpecConstant` in the blobs), so the base modules are what the pipelines use.
    nvrhi::ShaderHandle adapterShader;
    nvrhi::ShaderHandle interleaveShader;
    nvrhi::ShaderHandle histogramShader;
    nvrhi::ShaderHandle averageShader;
    nvrhi::ShaderHandle checkerboardShader;
    nvrhi::ShaderHandle prepareFinalShader;

    // The layouts this module owns: the five exact set-0 layouts, the shared compute uniform
    // layout, the two tonemapping layouts (UAV for the exposure pair, SRV for the final
    // composition), the empty layout that fills the average's set 0 and the prepare-final's set 3,
    // and the volumetric layout. Every RT pass's layouts are AllRayTracing visibility and cannot
    // be reused.
    nvrhi::BindingLayoutHandle adapterFramebufferLayout;
    nvrhi::BindingLayoutHandle interleaveFramebufferLayout;
    nvrhi::BindingLayoutHandle histogramFramebufferLayout;
    nvrhi::BindingLayoutHandle checkerboardFramebufferLayout;
    nvrhi::BindingLayoutHandle prepareFinalFramebufferLayout;
    nvrhi::BindingLayoutHandle uniformLayout;
    nvrhi::BindingLayoutHandle tonemappingUavLayout;
    nvrhi::BindingLayoutHandle tonemappingSrvLayout;
    nvrhi::BindingLayoutHandle emptyLayout;
    nvrhi::BindingLayoutHandle volumetricLayout;

    // The six compute pipelines, each over its own set-0 layout and the shared sets.
    nvrhi::ComputePipelineHandle adapterPipeline;
    nvrhi::ComputePipelineHandle interleavePipeline;
    nvrhi::ComputePipelineHandle histogramPipeline;
    nvrhi::ComputePipelineHandle averagePipeline;
    nvrhi::ComputePipelineHandle checkerboardPipeline;
    nvrhi::ComputePipelineHandle prepareFinalPipeline;

    // The module's per-slot wrap of the engine's tonemapping buffer and the two sets over it: the
    // UAV set the histogram and the average bind, and the SRV set the final composition binds. The
    // engine buffer is created once per slot and never re-created, so all three follow the pass's
    // lifetime, not the framebuffers'.
    nvrhi::BufferHandle tonemappingBuffers[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle tonemappingUavSets[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle tonemappingSrvSets[MAX_FRAMES_IN_FLIGHT];

    // The module-owned stand-ins: the render-sized god-rays zero substitute (re-created on a size
    // change), the real empty set, and the 1x1x1 RGBA16F volumetric dummy with its sampler and
    // set. The dummy is cleared to zero once, on the first list that binds it.
    nvrhi::TextureHandle godRaysZeroTexture;
    uint32_t godRaysZeroWidth = 0;
    uint32_t godRaysZeroHeight = 0;
    nvrhi::TextureHandle volumetricDummyTexture;
    nvrhi::SamplerHandle volumetricDummySampler;
    nvrhi::BindingSetHandle volumetricSet;
    nvrhi::BindingSetHandle emptySet;
    bool volumetricDummyCleared = false;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // One-shot warnings for the inputs that can legitimately be missing for a few frames or are a
    // permanent host-side mistake.
    bool warnedMissingFramebuffers = false;
    bool warnedUnexpectedSize = false;
    bool warnedMissingUniform = false;
    bool warnedBadUniform = false;
    bool warnedBadTable = false;
    bool warnedZeroTexture = false;

    bool created = false;
};

}
