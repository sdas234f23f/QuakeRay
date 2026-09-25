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

#include "RhiRtPrimaryPass.h"

namespace vkpt
{

class Framebuffers;
class LightManager;

namespace rhi
{
class RhiFrameContext;
class RhiTextureTable;
}

// The RHI module of the renderer's second ray-tracing pass: the engine's direct-lighting raygen
// `RGenDirect` (`RtRaygenDirect.rgen.spv`) with the same two misses and two hit groups the primary
// pass carries, dispatched over the RHI acceleration structures into the engine's
// UNFILTERED_DIRECT / UNFILTERED_SPECULAR storage images and read back through the G-buffer the
// primary pass wrote. It is stream 3 of the A4.2a cut of the refactor plan, the A4.2 reconnaissance
// (a42_recon.md) §4/§6.2; the sibling of RhiRtPrimaryPass, dispatched on the same command list
// directly after it (the legacy order is primary `VulkanDevice.cpp:901` -> direct `:1039`).
//
// The raygen's interface (measured 2026-09-25 with `spirv-dis` over the blob built from the current
// sources by the very command `GenerateShaders.py:157-164` produces, dxc from Vulkan SDK 1.4.321.1;
// the A4.2a shader-side fix that drops the sampled view-direction binding 147 is in the source
// state and in that blob):
//   set 0  TLAS                - RhiAccelStructs::GetTopLevel, raw binding 0;
//   set 1  framebuffers        - 12 items, exactly: the UAVs at raw bindings 14
//                                (framebufUnfilteredDirect), 15 (framebufUnfilteredSpecular) and
//                                23 (framebufViewDirection), and the SRVs at raw bindings 124
//                                (framebufAlbedo_Sampled), 126 (IsSky), 127 (Normal), 129
//                                (NormalGeometry), 131 (MetallicRoughness), 143
//                                (SurfacePosition), 239 (Q2GradSmplPos), 245 (Q2RngSeed) and 247
//                                (Q2Cluster). The blob declares no other set-1 binding: the
//                                sampled view direction (147) is gone (S1's fix, a42_recon.md §0.1.1
//                                and §6.2.1) and the sampled screen emission (185) is dead-stripped
//                                because `RGenDirect` never reads `surf.emission`;
//   set 2  global uniform      - the engine's wrapped ShGlobalUniform, raw binding 0;
//   set 4  textures            - the shared bindless table (RhiTextureTable), its own two arrays;
//   set 6  light sources       - the 5 engine buffers at raw bindings 0 (lightSources,
//                                StructuredBuffer<ShLightEncoded>, stride 144), 4
//                                (q2LightListOffsets), 5 (q2LightListLights), 6 (q2LightStats, the
//                                only UAV) and 8 (q2ClusterSkyVis);
//   set 11 ray stats           - an RHI-owned RWStructuredBuffer<RtRayStats> stand-in, raw
//                                binding 0.
// The raygen has no specialization constants (measured: no `OpSpecConstant` in the blob), so no
// createShaderSpecialization is needed; it declares only `ShPayloadShadow` = 4 B in its own
// interface, but the pipeline also carries the two misses and the two hit groups, whose payload is
// `ShPayload` = 16 B, so the pipeline keeps the engine's per-pipeline 16 B / 8 B limits (see the
// .cpp). The shadow rays address miss index SBT_INDEX_MISS_SHADOW = 1
// (`RaygenCommon.hlsli:465-473`), which fixes the shader table's miss order: `RtMiss` has to fill
// index 0 although the direct raygen never enters it.
//
// Why this pass needs its own pipeline and shader table (a42_recon.md §4): the pinned NVRHI refuses
// a binding set that leaves any item of its layout unfilled (validation-device.cpp:1855-1871), so
// one pipeline cannot serve the primary's 26-UAV set 1 and this raygen's 3-UAV/9-SRV set over a
// single layout, and set 6 is empty for the primary but real here. The layouts the two pipelines
// *can* share have to be the same layout handles, not equal copies: the backend compares
// `set->getLayout() != expectedLayout` when it applies a state (validation-commandlist.cpp:
// 509-520). This module therefore builds its own sets 0/2/3 over `primaryPass`'s layout handles,
// uses the primary's empty layout at 5/7/8/9/10 and the primary's ray-stats layout and set at 11,
// and owns only two layouts: the exact set-1 layout above and the exact set-6 layout below. The
// direct raygen declares neither set 7 nor set 8, so no cubemap placeholders are needed - the
// positions are the same empty hole the primary binds at 5/6/9/10 (a42_recon.md §4).
//
// The twelve sets, in the engine's RT set order (RayTracingPipeline.cpp:62-86):
//   set 0  TLAS                - own per-slot set over the primary's tlasLayout;
//   set 1  framebuffers        - own per-slot set over the module's exact 12-item layout;
//   set 2  global uniform      - own per-slot set over the primary's uniformLayout;
//   set 3  vertex data         - own per-slot set over the primary's 7-item vertexDataLayout (the
//                                direct raygen itself fetches no geometry, but `RtAlphaTest.rahit`,
//                                the any-hit of the alpha-tested hit group the shadow rays enter,
//                                reads the static/dynamic vertices, indices and the geometry
//                                records - measured: the any-hit blob declares set 3 bindings 0-4);
//   set 4  textures            - the shared texture table itself;
//   set 5  (random)            - the primary's empty set;
//   set 6  light sources       - own per-slot set over the module's exact 5-item layout;
//   set 7  cubemaps            - the primary's empty set (`RGenDirect` declares no set 7);
//   set 8  render cubemap      - the primary's empty set (`RGenDirect` declares no set 8);
//   set 9  (portals)           - the primary's empty set;
//   set 10 (volumetric)        - the primary's empty set;
//   set 11 ray stats           - the primary's ray-stats set.
//
// Light data (a42_recon.md §2, §5): the engine runs its host-side light composition every frame but
// under `rhiframe` skips the device copies that used to publish it (`LightManager::CopyFromStaging`
// is only called from `Scene::SubmitForFrame`, which `RenderThroughRhi` bypasses,
// VulkanDevice.cpp:1234-1347), so this pass records them itself, on the same list and before its
// dispatch: the light-array prefix (always, including the sun slot at index 0), the list offsets
// and words while their publication is pending, and the cluster sky-visibility table while its
// update is pending. `LightManager::GetBuffers()` hands over the five device-local engine buffers,
// `LightManager::GetFrameCopies(frameIndex)` the frame's staging handles and byte counts
// (`Copy::size == 0` means nothing to copy). Once the copies are recorded the pass consumes the
// pending flags through `LightManager::ConsumeFrameCopies(frameIndex)`, so a list buffer or the
// sky-visibility table is copied again only when a new publication raises its flag; the light-array
// prefix has no flag and keeps its every-frame copy. The wraps and the copies follow the module's
// usual change detection and Retire contract.
//
// The light statistics (a42_recon.md §1): while `globalUniform.q2LightStatsMode` is not
// Q2_LIGHT_STATS_DISABLED the direct raygen accumulates into and reads the per-cluster counters of
// `LightManager`'s statistics buffer (Q2LightLists.hlsli:189-215, 312-358). The RHI list replaces
// the legacy `LightManager::ResetLightStats` (`VulkanDevice.cpp:873-876`): the pass zeroes the
// statistics slots the frame needs with the exact range that function fills
// (`GetLightStatsClusterTarget` / `GetLightStatsClusterSize`, a prefix of 6,144 B per cluster), the
// slot `frameId % LIGHT_STATS_SLOT_COUNT` always and the slots the lists have grown past what they
// were last zeroed to. The fill is recorded on the set-6 statistics wrap with the automatic-barrier
// pass (CopyDest before it, UnorderedAccess after it), and the pass mirrors the engine's cleared
// bookkeeping itself, so the legacy path's state is untouched. `fltEnable[0] = 0` remains forced:
// image 115 is still not written.
//
// Host contract (the skeleton wires the pass, the pass only records):
//  - Record after the primary pass on the same list: the raygen reads the G-buffer the primary
//    wrote into the same slot's images, and the NVRHI barriers between the two dispatches are what
//    orders the writes before the reads.
//  - Pass the frame's `frameId` and `lightStatsMode`, exactly the `ShGlobalUniform::frameId` and
//    `ShGlobalUniform::q2LightStatsMode` the host wrote into the uniform this frame's raygen reads
//    (`sky.uniform->GetData()->frameId` and `sky.uniform->GetData()->q2LightStatsMode`), so that
//    the fill slot and the shader's slot agree: `frameId` is the engine counter, NOT `frameIndex`
//    (the statistics rotate over LIGHT_STATS_SLOT_COUNT = 3 slots while MAX_FRAMES_IN_FLIGHT = 2).
//    A mode of Q2_LIGHT_STATS_DISABLED (0) records no fill and leaves the statistics buffer to the
//    same untouched path A4.2a had; any other mode makes the pass fill the slots the frame needs
//    before the dispatch that writes them.
//  - Force `globalUniform.fltEnable[0] = 0` in the uniform patch for the traced mode: the direct
//    raygen reads the gradient-sample image 115 only when `fltEnable[0] >= 0.5`
//    (`Q2LightLists.hlsli:141-154`) and the RHI path never writes image 115 (only the denoiser
//    produces it). Binding 239 still has to be a valid descriptor, which it is; without the switch
//    the shader reads undefined memory. Temporary until A4.5 (a42_recon.md §2.7).
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images.
//
// The pass is a no-op until Create succeeded and while an input is missing (no TLAS, no
// framebuffers, no uniform, no vertex data, no light buffers); every early return is quiet after
// the first warning. It is not thread-safe: Render uses the per-slot target of the frameIndex it is
// given, which is the engine's single-threaded per-slot frame model (RhiFrameContext).
class RhiRtDirectPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiRtDirectPass();
    ~RhiRtDirectPass();

    RhiRtDirectPass(const RhiRtDirectPass &other) = delete;
    RhiRtDirectPass(RhiRtDirectPass &&other) noexcept = delete;
    RhiRtDirectPass &operator=(const RhiRtDirectPass &other) = delete;
    RhiRtDirectPass &operator=(RhiRtDirectPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queue every replaced wrap and set goes
    // through; 'pTextureTable' is the host's shared bindless table (RHI/RhiTextureTable.h), whose
    // layout becomes set 4 and whose table is bound with it; 'pLightManager' is the engine's light
    // registry (LightManager.h), the source of the set-6 buffers and of the frame's light copies -
    // the pass consumes the pending copies it records, so it is taken non-const (the call site does
    // not change: Scene::GetLightManager().get() is already non-const); 'pPrimaryPass' is the
    // created primary-visibility pass (RHI/RhiRtPrimaryPass.h), which owns the shared set layouts,
    // the empty set and the ray-stats set this pipeline and its states use.
    // None of them is owned; all have to outlive this object, and a null or unusable one makes
    // Create fail. 'pShaderFolderPath' is the folder ShaderManager loads the engine blobs from,
    // with the trailing separator; the five RT blobs above are read from it. The pass logs through
    // 'pfnPrint'. Returns false and leaves the pass unusable if a shader, a layout, a resource, the
    // pipeline or the shader table cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                rhi::RhiTextureTable *pTextureTable,
                LightManager *pLightManager,
                const RhiRtPrimaryPass *pPrimaryPass,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // One call per frame, on the frame context's open command list of 'frameIndex', after
    // RhiRtPrimaryPass::Render of the same slot. 'frameId' is the engine counter
    // (`ShGlobalUniform::frameId`, i.e. sky.uniform->GetData()->frameId) and 'lightStatsMode' the
    // ShGlobalUniform::q2LightStatsMode of the same uniform, both exactly as the host wrote them
    // into this frame's uniform: the mode gates the statistics fill and the frame id picks the
    // statistics slot (frameId % LIGHT_STATS_SLOT_COUNT, not frameIndex). 'pTopLevel' is the slot's
    // top-level structure, 'pUniformBuffer' the engine's global uniform as a static constant-buffer
    // wrap (the same wrap the primary takes), 'vertexData' the seven set 3 buffers
    // ('RhiRtPrimaryPass::VertexData', the type is reused because the list is the same one the
    // primary binds; this pass builds its own set over the primary's layout handle), and
    // 'pFramebuffers' the engine's framebuffer registry.
    // 'width'/'height' are the render resolution the images are sized to; the pass resolves every
    // image extent through Framebuffers::GetImageHandles' 4-tuple form, so the one set-1 image that
    // is not render-sized (Q2_GRAD_SMPL_POS, 1/3) is wrapped at its own extent.
    //
    // What is recorded: the wraps of the 12 set-1 images (created on first use, re-created when the
    // engine re-created an image or the size changed; the replaced wraps and the sets over them go
    // through the frame context's retire queue), the per-slot sets 0-3 and 6, the four light copies
    // and the light-statistics fill of the slots the frame needs while the mode is not disabled
    // (both before the state, so the automatic barriers order them before the dispatch), then one
    // `dispatchRays(width, height, 1)`. The full-size dispatch is what the raygen wants: its
    // `DispatchRaysIndex` is the checkerboard-packed texel itself (the shader hands it straight to
    // `fetchGbufferSurface` and to the two image stores, RtRaygenDirect.rgen.hlsl:82,186-188), and
    // the packed images are created at the render size, so one invocation covers one packed texel
    // - nothing is halved and no dispatch index is unused.
    //
    // The image-state contract: the engine leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL
    // (= NVRHI's UnorderedAccess), and a native wrap keeps no state between command lists, so the
    // pass announces that state for all 12 images before its first use on every list
    // (beginTrackingTextureState, exactly RhiRtPrimaryPass's contract). After the dispatch the 9
    // images that are bound as SRVs and by no UAV are moved back to UnorderedAccess: their
    // Texture_SRV requirement left them in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    // (vulkan-resource-bindings.cpp:398-435), and the next frame's primary pass writes them as
    // storage images (a42_recon.md §0.1.3). The 3 UAV images end the list in GENERAL already.
    //
    // No-op when the pass is not created, the frame index is out of range, the size is zero, the
    // TLAS or the framebuffers are missing, the uniform/vertex-data inputs are missing or in a
    // shape NVRHI's validation refuses, or a light buffer (or a staging buffer of a pending copy)
    // is missing.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                uint32_t frameId,
                uint32_t lightStatsMode,
                nvrhi::rt::IAccelStruct *pTopLevel,
                nvrhi::IBuffer *pUniformBuffer,
                const RhiRtPrimaryPass::VertexData &vertexData,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height);

    // Drops every slot's image wraps and the sets over them, the per-slot TLAS/uniform/vertex sets
    // and the set-6 wraps and set, and retires them through the frame context's queue. The caller
    // has to call it before the engine destroys its framebuffer images (the
    // Framebuffers::PrepareForSize path) - otherwise the wraps reference destroyed VkImages. The
    // next Render re-reads the handles and re-wraps both the images and the engine's light buffers,
    // so the pass survives a resize without a second Create.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: every framebuffer image is a per-slot (swapped) image and the
    // vertex-data buffers are the slot's RHI copies, so neither the sets over them nor the light
    // staging wraps can be shared across slots. The five device-local light buffers are shared by
    // the engine, but their wraps and the set over them are kept per slot with the rest, under the
    // same handle-change rule.
    struct Target
    {
        // Set 1: the 12 engine images (the .cpp's FRAMEBUFFER_BINDINGS table) the slot currently
        // wraps and the set over them. The handles are kept in the form Render received them, not
        // as VkImages, because they are what the change detection compares; a change in any of them
        // or in the size means the engine re-created the framebuffers and the wraps and the set
        // have to follow.
        uint64_t imageHandles[12] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle framebufferTextures[12];
        nvrhi::BindingSetHandle framebufferSet;

        // Set 0: the pointer is only the cache key that tells whether the set still addresses the
        // current AS; the set itself holds the reference that keeps the AS alive.
        nvrhi::rt::IAccelStruct *topLevel = nullptr;
        nvrhi::BindingSetHandle tlasSet;

        // Set 2: the engine's global uniform never changes, but it is a Render argument here, so
        // the set follows the pointer the same way.
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;

        // Set 3: the seven buffers of this slot (the host's RHI copies can be re-created when a
        // later frame needs more room, so the pointers are the key).
        nvrhi::IBuffer *vertexBuffers[7] = {};
        nvrhi::BindingSetHandle vertexDataSet;

        // Set 6: the five device-local engine buffers as the slot wraps them (the raw handles are
        // the key) and the set over the wraps, plus the four copy-source wraps of this slot's
        // pending light copies, keyed by their staging handles.
        VkBuffer lightHandles[5] = {};
        nvrhi::BufferHandle lightWraps[5];
        nvrhi::BindingSetHandle lightSet;
        VkBuffer lightStagingHandles[4] = {};
        nvrhi::BufferHandle lightStagingWraps[4];
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Retires the slot's image wraps and the framebuffer set over them (the framebuffer re-create
    // path), the slot's TLAS/uniform/vertex-data sets (ReleaseTargets), and the slot's set-6 wraps,
    // set and staging wraps (a light-buffer handle change).
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);
    void ReleaseLightTarget(Target &target);

    // The per-slot sets 0-3, rebuilt when their key changed. Return false when the input is
    // unusable for this frame (the caller skips the dispatch).
    bool PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel);
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);
    bool PrepareVertexDataSet(Target &target, const RhiRtPrimaryPass::VertexData &vertexData);

    // Set 6 and this frame's light copies: wraps the five device-local engine buffers (re-wrapping
    // when a handle changed), builds the set over the module's light layout, records one
    // copyBuffer per pending item of LightManager::GetFrameCopies(frameIndex) and consumes the
    // flags it recorded, and records the light-statistics fill while 'lightStatsMode' is not 0.
    // Returns false when an engine buffer or a pending copy source is missing.
    bool PrepareLightSet(nvrhi::ICommandList *pCommandList, Target &target, uint32_t frameIndex,
                         uint32_t frameId, uint32_t lightStatsMode);

    // The ranged statistics fill of the frame: it is recorded on the set-6 wrap of the engine's
    // statistics buffer inside 'target' (the same wrap the binding set binds, which is what makes
    // the automatic barriers cover the native fill) and on that buffer's raw handle 'statsBuffer'.
    // Zeroes exactly the slots LightManager::ResetLightStats would zero for 'frameId', on the range
    // the lists the device holds can name, and mirrors that function's statsClusterCleared in the
    // pass. Does nothing while the mode is Q2_LIGHT_STATS_DISABLED.
    void RecordLightStatsFill(nvrhi::ICommandList *pCommandList, const Target &target,
                              VkBuffer statsBuffer, uint32_t frameId, uint32_t lightStatsMode);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model and shared texture table, the engine's light registry and
    // the primary-visibility pass, all outlive this object. The light registry is non-const because
    // the pass consumes the pending copies it records (LightManager::ConsumeFrameCopies).
    rhi::RhiFrameContext *frameContext = nullptr;
    rhi::RhiTextureTable *textureTable = nullptr;
    LightManager *lightManager = nullptr;
    const RhiRtPrimaryPass *primaryPass = nullptr;

    // The five engine blobs. The raygen needs no specialization (measured: the module declares no
    // SpecId), so the base module is what the pipeline uses.
    nvrhi::ShaderHandle raygenShader;
    nvrhi::ShaderHandle missShader;
    nvrhi::ShaderHandle shadowMissShader;
    nvrhi::ShaderHandle closestHitShader;
    nvrhi::ShaderHandle anyHitShader;

    // The two layouts this module owns: set 1 (the exact 12 items above) and set 6 (the exact 5
    // items). Every other layout is the primary's.
    nvrhi::BindingLayoutHandle framebufferLayout;
    nvrhi::BindingLayoutHandle lightLayout;

    // The pipeline and its table: one raygen (RGenDirect), the engine's two misses, the two engine
    // hit groups. Both are created once; the table is uncached, so the backend bakes it per list.
    nvrhi::rt::PipelineHandle pipeline;
    nvrhi::rt::ShaderTableHandle shaderTable;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    /* The rotating slots of the engine's light-statistics buffer and this pass's mirror of what
       LightManager::statsClusterCleared holds: the RHI fills the slots without telling the engine,
       so the mirror has to live here, one entry per statistics slot and independent of the
       two-entry frame-slot Target array (three statistics slots against MAX_FRAMES_IN_FLIGHT = 2).
       The constant is LightManager::LIGHT_STATS_SLOT_COUNT; this header keeps the engine headers
       out, so it stands here and the .cpp asserts that the two agree. The entries are zero - every
       slot is filled - until the pass fills one, and they are dropped when the engine's statistics
       buffer handle changes, because a re-created buffer holds no counters this pass zeroed. */
    static constexpr uint32_t LIGHT_STATS_SLOT_COUNT = 3;
    uint32_t statsClusterCleared[LIGHT_STATS_SLOT_COUNT] = {};

    // One-shot warnings for the inputs that can legitimately be missing for a few frames or are a
    // permanent host-side mistake.
    bool warnedMissingTopLevel = false;
    bool warnedMissingFramebuffers = false;
    bool warnedMissingUniform = false;
    bool warnedBadUniform = false;
    bool warnedMissingVertexData = false;
    bool warnedBadVertexData = false;
    bool warnedMissingLights = false;
    bool warnedMissingLightStaging = false;

    bool created = false;
};

}
