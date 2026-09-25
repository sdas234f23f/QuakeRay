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

// The RHI module of the compose preview: the engine's `flt_enable = 0` compositing chain
// `CmQ2Adapter` -> `CmQ2Interleave` -> `CmCheckerboard` as three compute pipelines on the frame's
// command list, ending in the engine's FINAL image (28) the present shows. It is stream 2 of the
// A4.2b cut of the renderer refactor, the "route-(b) compose preview" of the A4.2b reconnaissance
// (a42b_recon.md §3); the present switch that samples FINAL instead of ALBEDO is the sibling
// stream S4 and is not part of this module - the module exposes the FINAL wrap through
// GetFinalTexture for it.
//
// The chain is what the legacy frame runs with the denoiser disabled: `Q2Denoiser::Denoise`
// records the adapter and, because `fltEnable[0] < 0.5`, immediately calls
// `InterleaveCheckerboard` and returns (Q2Denoiser.cpp:412-436, :600-623); the checkerboard
// resolve follows from `ImageComposition::PrepareForRaster` (ImageComposition.cpp:93-98,
// :163-202, called at VulkanDevice.cpp:1066). The adapter composites the raw ReSTIR signal into
// Q2_COLOR (117) in its `fltEnable[0] < 0.5` branch, the interleave resolves the checkerboard of
// Q2_COLOR into PRE_FINAL (27), and the checkerboard pass writes FINAL (28) with the
// SCREEN_EMISSION (62) and ACID_FOG (60) companions. The tonemapping/raster/bloom/FSR tail behind
// the legacy checkerboard call is deliberately not part of the preview (a42b_recon.md §3.4: no
// CmPrepareFinal, LPM, tonemapping, bloom, FSR/DLSS/TAAU, ASVGF chain or rasterizer overlay).
//
// The three pipelines and their exact sets (measured 2026-09-25 with `spirv-dis` over
// `vkpt\Build\CmQ2Adapter.comp.spv`, `CmQ2Interleave.comp.spv` and `CmCheckerboard.comp.spv` - the
// blobs the engine's build produces and the legacy ShaderManager loads, and the blobs this module
// loads too):
//   CmQ2Adapter    - set 0: 5 storage images (73 Q2ColorLF_SH, 75 Q2ColorLF_COCG, 77 Q2ColorHF,
//                    79 Q2ColorSpec, 117 Q2Color) and 12 sampled images (0 ALBEDO, 2 IS_SKY,
//                    3 NORMAL, 7 METALLIC_ROUGHNESS, 14 UNFILTERED_DIRECT, 15 UNFILTERED_SPECULAR,
//                    16-18 UNFILTERED_INDIRECT_S_H_R/G/B, 26 THROUGHPUT, 88 Q2_TRANSPARENT,
//                    90 Q2_FOG_ACCUM); set 1: the engine's global uniform at raw binding 0.
//   CmQ2Interleave - set 0: 1 storage image (27 PRE_FINAL) and 2 sampled images (26 THROUGHPUT,
//                    117 Q2Color); set 1: the global uniform.
//   CmCheckerboard - set 0: 3 storage images (28 FINAL, 60 ACID_FOG, 62 SCREEN_EMISSION) and 4
//                    sampled images (26 THROUGHPUT, 27 PRE_FINAL, 59 ACID_FOG_RT, 61
//                    SCREEN_EMIS_RT); set 1: the global uniform.
// All three are `[numthreads(16, 16, 1)]`, so the dispatch is the legacy
// `Utils::GetWorkGroupCount(w, 16) x ...(h, 16)` (Q2Denoiser.cpp:400-401, :606-607,
// ImageComposition.cpp:198-199). The helper is 1 + ceil(size / group); the shaders' own
// `renderWidth`/`renderHeight` bounds check discards the extra workgroup.
//
// The sets cannot reuse the RT passes' layouts: every RT set is `AllRayTracing` visibility
// (nvrhi.h:883), which has no Compute bit, and the pinned NVRHI skips a binding layout whose
// visibility lacks the shader's stage when it validates a compute pipeline
// (validation-device.cpp:1002-1003). The module therefore creates its own set-0 layouts with
// `ShaderType::Compute` visibility and its own compute uniform layout at raw binding 0. The
// set-0 offsets turn the engine's raw bindings into NVRHI slots (RhiPipeline.h): the storage
// images sit at raw binding = image index (`ShFramebuffers_Bindings`, the identity) with offset 0,
// the sampled views at raw binding = 124 + image index (`ShFramebuffers_Sampled_Bindings`) with
// offset 124. The three set-0 layouts are partial in the same way the RT passes' are: one item per
// binding the blob declares, no union, because NVRHI refuses a set that leaves a layout item
// unfilled (validation-device.cpp:1855-1871), so adapter, interleave and checkerboard own a 17-,
// a 3- and a 7-item layout, and each pipeline declares its own set 0 beside the shared set 1.
//
// 23 engine images are wrapped per slot (the .cpp's COMPOSE_IMAGES table), each once, and the
// three sets share the wraps, so the pass-to-pass hand-offs (117 written by the adapter and read
// by the interleave, 27 written by the interleave and read by the checkerboard) are covered by
// NVRHI's own UAV -> SRV -> UAV transitions on one wrap. The three UNFILTERED_INDIRECT_S_H images
// (16-18) are written by the A4.3 indirect pass (RhiRtIndirectPass), which runs earlier on the
// same command list, so the adapter's 140/141/142 sampled bindings read real data. The invariant
// this module relies on is the host order primary -> direct -> indirect -> compose: without the
// indirect pass the adapter would sample the images' engine-initialized contents.
//
// The image-state contract: the engine leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL
// (= NVRHI's UnorderedAccess) and a native wrap keeps no state between command lists
// (RhiTextureSource.h), so every list announces UnorderedAccess for all 23 wraps before the first
// pass and moves the 16 images whose last use is a sampled read back to UnorderedAccess after the
// checkerboard (the A4.2a discipline: an unannounced first SRV use would discard the contents, and
// a finish in SHADER_READ_ONLY_OPTIMAL would mismatch the next frame's UAV announcements). The
// restore is also what lets the next frame's indirect pass write 16-18: its own UnorderedAccess
// announcement finds the read-only state and emits no transition, so an unrestored image would be
// written as read-only.
//
// Host contract (the skeleton wires the pass, the pass only records):
//  - Record after the frame's ray-tracing passes (RhiRtDirectPass::Render and
//    RhiRtIndirectPass::Render) on the same list: the adapter samples the G-buffer, the direct
//    images, the indirect SH the indirect pass wrote and the throughput the primary/direct chain
//    wrote.
//  - 'width'/'height' are the render resolution the shaders read as `globalUniform.renderWidth/
//    renderHeight` and the images are sized to; the dispatch has to run over the same numbers, or
//    pixels beyond the smaller one keep their previous content.
//  - `globalUniform.fltEnable[0]` has to stay below 0.5 (the adapter's unfiltered branch); with
//    the filtered branch selected the adapter writes the ASVGF channel images instead of Q2_COLOR
//    and the preview would show stale color. This is the skeleton's forced switch, shared with
//    RhiRtDirectPass.
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images;
//    the next Render re-reads the handles and re-wraps, so the pass survives a resize.
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
    // through. Neither is owned; both have to outlive this object, and a null or
    // unusable one makes Create fail. 'pShaderFolderPath' is the folder ShaderManager loads the
    // engine blobs from, with the trailing separator; the three compose blobs above are read from
    // it. The pass logs through 'pfnPrint'. Returns false and leaves the pass unusable if a shader,
    // a layout or a pipeline cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // One call per frame, on the frame context's open command list of 'frameIndex', after the
    // frame's ray-tracing passes (RhiRtDirectPass::Render and RhiRtIndirectPass::Render) of the
    // same slot. 'pFramebuffers' is the engine's framebuffer registry, 'width'/'height' the render
    // resolution, and 'pUniformBuffer' the engine's global uniform as a static constant-buffer wrap
    // (the same wrap the primary and direct passes take).
    //
    // What is recorded: the wraps of the 23 engine images (created on first use, re-created when
    // the engine re-created an image or the size changed; the replaced wraps and the sets over them
    // go through the frame context's retire queue), the per-slot sets, the UnorderedAccess
    // announcement of the 23 images, then three `setComputeState` + `dispatch` pairs - adapter,
    // interleave, checkerboard - and the UnorderedAccess restore of the 16 images that end in the
    // read-only state.
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
    // the checkerboard and Render restores it there, so the present has to
    // beginTrackingTextureState(final, AllSubresources, UnorderedAccess) before its draw and
    // setTextureState(final, AllSubresources, UnorderedAccess) after it - the same pair the
    // skeleton performs for ALBEDO (NvrhiFrameSkeleton.cpp:560-564) - or the next frame's
    // checkerboard UAV write runs against a read-only layout.
    nvrhi::ITexture *GetFinalTexture(uint32_t frameIndex) const;

    // Drops every slot's image wraps and the three sets over them and the per-slot uniform sets on
    // the shared uniform layout, and retires them through the frame context's queue. The caller has
    // to call it before the engine destroys its framebuffer images (the Framebuffers::PrepareForSize
    // path) - otherwise the wraps reference destroyed VkImages. The next Render re-reads the
    // handles and re-wraps, so the pass survives a resize without a second Create.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: the images of the compose chain are engine framebuffer
    // images (ping-ponged ones like ALBEDO and Q2_COLOR included), so neither the wraps over them
    // nor the sets can be shared across slots. The uniform set is kept per slot with the rest,
    // under the same pointer-change rule the other passes use.
    struct Target
    {
        // The 23 engine images (the .cpp's COMPOSE_IMAGES table) the slot currently wraps and the
        // three sets over them. The handles are kept in the form Render received them, not as
        // VkImages, because they are what the change detection compares; a change in any of them
        // or in the size means the engine re-created the framebuffers and the wraps and the sets
        // have to follow.
        uint64_t imageHandles[23] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle engineTextures[23];
        nvrhi::BindingSetHandle adapterSet;
        nvrhi::BindingSetHandle interleaveSet;
        nvrhi::BindingSetHandle checkerboardSet;

        // Set 1: the engine's global uniform never changes, but it is a Render argument here, so
        // the set follows the pointer the way the other passes' uniform sets do.
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Builds the three set-0 sets over the module's exact layouts when they are missing (the first
    // frame of a slot and after a framebuffer re-create). Returns false when a binding cannot be
    // filled - a table/image mismatch, which the module's static assertions make unreachable.
    bool PrepareFramebufferSets(Target &target);

    // Set 1 over the module's compute uniform layout, rebuilt when the pointer changed. Returns
    // false when the buffer is not the static constant-buffer wrap the shader's
    // `ConstantBuffer<ShGlobalUniform>` requires.
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);

    // One `setComputeState` + `dispatch` for one of the three passes: set 0 is the pass's
    // framebuffer set, set 1 the shared uniform set (the pipeline's layout order).
    void RecordDispatch(nvrhi::ICommandList *pCommandList,
                        nvrhi::IComputePipeline *pPipeline,
                        nvrhi::IBindingSet *pFramebufferSet,
                        nvrhi::IBindingSet *pUniformSet,
                        uint32_t groupsX,
                        uint32_t groupsY);

    // Retires the slot's image wraps and the three sets over them (the framebuffer re-create path),
    // and the slot's uniform set (ReleaseTargets).
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, which outlives this object.
    rhi::RhiFrameContext *frameContext = nullptr;

    // The three engine blobs. None of the three declares a specialization constant (measured: no
    // `OpSpecConstant` in the three blobs), so the base modules are what the pipelines use.
    nvrhi::ShaderHandle adapterShader;
    nvrhi::ShaderHandle interleaveShader;
    nvrhi::ShaderHandle checkerboardShader;

    // The four layouts this module owns: the three exact set-0 layouts and the shared compute
    // uniform layout. Every RT pass's layouts are AllRayTracing visibility and cannot be reused.
    nvrhi::BindingLayoutHandle adapterFramebufferLayout;
    nvrhi::BindingLayoutHandle interleaveFramebufferLayout;
    nvrhi::BindingLayoutHandle checkerboardFramebufferLayout;
    nvrhi::BindingLayoutHandle uniformLayout;

    // The three compute pipelines, each over its own set-0 layout and the shared uniform layout.
    nvrhi::ComputePipelineHandle adapterPipeline;
    nvrhi::ComputePipelineHandle interleavePipeline;
    nvrhi::ComputePipelineHandle checkerboardPipeline;

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
