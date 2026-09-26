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

namespace rhi
{
class RhiFrameContext;
class RhiTextureTable;
}

// The RHI module of the renderer's reflect/refract ray-tracing pass: the engine's Q2
// reflection/refraction raygen `RGenQ2ReflRefr` (`RtQ2ReflectRefract.rgen.spv`) with the engine's
// default miss and the two engine hit groups, dispatched over the RHI acceleration structures into
// the G-buffer images the raygen overwrites for the surfaces that reflect or refract. It is stream
// 2 of the A5.3 cut of the refactor plan, the A5b reconnaissance (a5b_volumetric_recon.md) §3. The
// legacy twin of the family, `RtRaygenReflRefr.rgen.spv` (the engine's SBT index 1), is NOT
// ported: it has no dispatch site anywhere in the tree (a3/a4 reconnaissance), its set 1 is a
// different 36-item shape and its set 8 a different two-item one, so it could not share this
// pipeline anyway.
//
// The legacy call site this module replaces is
// `PathTracer::TraceQ2ReflectionRefractionRays` (PathTracer.cpp:131-155): the engine barriers the
// twelve images the raygen touches (its `BarrierMultiple` list includes ALBEDO, NORMAL,
// NORMAL_GEOMETRY, METALLIC_ROUGHNESS, DEPTH_WORLD, DEPTH_NDC, MOTION, SURFACE_POSITION,
// VISIBILITY_BUFFER, VIEW_DIRECTION, THROUGHPUT and PRIMARY_TO_REFL_REFR) and then
// `TraceRays(cmd, SBT_INDEX_RAYGEN_Q2_REFL_REFR, width, height)`, one ray per pixel at the full
// render resolution. The host gate is `VulkanDevice.cpp:1011-1014`: the dispatch happens exactly
// when `globalUniform.reflectRefractMaxDepth > 0` (the game's `rt_reflrefr_depth`, default 2,
// clamped to 8, VulkanDevice.cpp:437). In the legacy frame the call sits between the god-rays
// input trace and the reflected-segment god-rays trace (VulkanDevice.cpp:901 -> :908-1028 -> :1011
// -> :1018-1028), i.e. after the primary and the shadow-map/god-rays pair; the RHI frame's matching
// slot is after `RhiRtPrimaryPass::Render` and the input side of the god rays, before their
// reflected-segment dispatch (the current A5.2 module bundles trace(0) [+ trace(1)] + filter into
// one Render, so the coordinator has to order or split that call around this pass: the reflected
// god rays consume the Q2GodRaysThroughputDist/negative-depth marks this raygen writes) and before
// the compose chain's gradient reproject (NvrhiFrameSkeleton.cpp's traced branch), because the
// reproject also reads the images this pass overwrites for reflection pixels. The module records
// only the dispatch; the ordering and the gate belong to the coordinator.
//
// The raygen's interface (measured with `spirv-dis` over `vkpt/Build/RtQ2ReflectRefract.rgen.spv`;
// the set 1 numbers below are the POST-FIX shape stream S1's storage-image reads leave - see the
// interface assumption section):
//   set 0  TLAS                - the slot's rt::IAccelStruct from RhiAccelStructs::GetTopLevel,
//                                raw binding 0;
//   set 1  framebuffers        - **24 items after S1's fix**: the 23 storage images the raygen
//                                writes and reads and the one sampled image it only reads. The
//                                UAVs are at raw bindings 0 (framebufAlbedo), 2 (IsSky), 3 (Normal),
//                                5 (NormalGeometry), 7 (MetallicRoughness), 9 (DepthWorld), 12
//                                (DepthNdc), 13 (Motion), 19 (SurfacePosition), 21
//                                (VisibilityBuffer), 23 (ViewDirection), 26 (Throughput), 59
//                                (AcidFogRT), 61 (ScreenEmisRT), 81 (Q2ViewDepth), 83
//                                (Q2BaseColor), 85 (Q2Metallic), 87 (Q2BounceThroughput), 88
//                                (Q2Transparent), 89 (Q2GodRaysThroughputDist), 90 (Q2FogAccum),
//                                121 (Q2RngSeed) and 123 (Q2Cluster); the only SRV is raw 149
//                                (framebufPrimaryToReflRefr_Sampled, image 25). The pre-fix blob had
//                                39 items: the fifteen same-image SRV/UAV pairs (images 0, 2, 3, 5,
//                                7, 9, 13, 19, 26, 59, 61, 83, 87, 88, 90) were removed by S1's fix,
//                                which reads those images through their storage views
//                                (`framebufX.Load`), and DXC dead-strips the now-unused sampled
//                                resources. Image 25 has no UAV in the blob (SRV-only, safe);
//   set 2  global uniform      - the engine's wrapped ShGlobalUniform, raw binding 0;
//   set 3  vertex data         - bindings 0-4 only in the blob (the alpha-tested any-hit's
//                                triangle fetches: static/dynamic vertices, indices and the
//                                geometry records). The pipeline still declares the primary's
//                                seven-item layout and this module builds its own seven-item set
//                                over that layout handle: NVRHI refuses a set that leaves any item
//                                of its layout unfilled (validation-device.cpp:1855-1871), so a
//                                five-item set-3 layout of its own would be the only alternative -
//                                the shared layout is the cheaper, established shape;
//   set 4  textures            - the shared bindless table (RhiTextureTable), its own two arrays;
//   set 5  (random)            - empty: the blob declares no set 5 (unlike the indirect raygen);
//   set 6  (light sources)     - empty: the blob declares no set 6 (the reflect/refract segments
//                                do not trace the ambient/light loop, they recurse through
//                                `traceReflectionRefractionRay` only);
//   set 7  cubemaps            - the `globalCubemaps[]` bindless array and its sampler, raw
//                                bindings 0/1. The primary's and the indirect's cubemap layouts are
//                                private and have no accessors, so this module owns a third bindless
//                                layout, filled with a 1x1 dummy cube in every slot exactly like the
//                                siblings' placeholders (RhiRtPrimaryPass.cpp:435-493,
//                                RhiRtIndirectPass.cpp:431-495);
//   set 8  render cubemap      - FOUR items, unlike the primary's two: `renderCubemap` at raw 0,
//                                `renderCubemapEnv` at raw 1 and their samplers at raw 2/3
//                                (BINDING_RENDER_CUBEMAP*), the same shape the indirect module owns;
//   set 9  portal instances    - `ConstantBuffer<PortalInstances_BT> portalInstances` at raw
//                                binding 0 (RaygenCommon.hlsli:149-160): one module-owned one-item
//                                layout and a per-slot set over it. The buffer is the engine's
//                                PortalList device-local array (63 x 64 B `ShPortalInstance`,
//                                ShaderCommonC.h:435-441; PortalList.cpp:40) that the coordinator
//                                wraps and fills per frame (the host contract below);
//   set 10 (volumetric)        - empty: no RT shader declares set 10 (a5b §1.3);
//   set 11 ray stats           - the primary's RHI-owned RWStructuredBuffer<RtRayStats> stand-in,
//                                raw binding 0.
//
// The raygen has exactly one specialization constant, `maxAlbedoLayerCount` (SpecId 0, uint, the
// SPIR-V default 0; RaygenPrimary.hlsli's reflrefr entry points), which the engine passes as
// `primaryRaysMaxAlbedoLayers` (RayTracingPipeline.cpp:96-98, i.e. the primary family's value) and
// the game sets to 2 (gl_vidsdl.c:1458). The pipeline therefore uses an
// `createShaderSpecialization` with UInt32(0, 2) - leaving the blob's default 0 compiles quietly
// but changes the multi-layer material evaluation, so the value must be set explicitly. The blob's
// own interface names `g_payload` (ShPayload = 16 B) and `HitAttributes` = 8 B; the pipeline keeps
// the engine's per-pipeline 16 B / 8 B / recursion-depth-2 limits (RayTracingPipeline.cpp:203).
//
// Why this pass needs its own pipeline and shader table: the pinned NVRHI refuses a binding set
// that leaves any item of its layout unfilled (validation-device.cpp:1855-1871) and the backend
// accepts a state's set only when it was created over the very layout handle the pipeline declared
// (validation-commandlist.cpp:509-520), so the primary's 26-UAV set 1, the direct pass's 12-item
// one and the indirect pass's 14-item one cannot express this raygen's 23+1 items, and this
// raygen's four-item set 8 / declared set 9 differ from all three. The layouts the pipelines *can*
// share have to be the same layout handles, not equal copies; this module therefore builds its own
// sets 0/2/3/11 over the primary pass's TLAS, uniform, vertex-data and ray-stats layouts, uses the
// primary's empty layout and empty set at the 5/6/10 holes, the shared texture table at 4, and
// owns only four layouts: the exact 24-item set 1, the one-item set 9, the set-7 bindless
// placeholder and the four-item set 8.
//
// The shader-binding table: one raygen, `RtMiss` at miss index 0, the fully-opaque hit group at
// offset 0 and the alpha-tested one at offset 1 - the engine's SBT indices for the two hit groups.
// There is NO shadow miss: the raygen has no shadow rays (measured: one `OpTraceRayKHR`, miss
// index 0, the reflective/refractive ray of `traceReflectionRefractionRay`,
// RaygenCommon.hlsli:306-329), so `RtMissShadowCheck` is neither loaded nor tabled and miss index 1
// is never addressed. The alpha-tested any-hit does run for reflective hits (`RtAlphaTest.rahit`
// reads the warped UV and sets 2/3/4). NVRHI has one raygen per table, so this module owns its own
// uncached table; the engine's raygen index for this shader (SBT_INDEX_RAYGEN_Q2_REFL_REFR = 3) has
// no NVRHI counterpart.
//
// The dispatch (a5b §3.3): `dispatchRays(width, height, 1)`, one ray per pixel over the full render
// resolution; the raygen maps `DispatchRaysIndex` to the checkerboard itself and early-outs on
// `isSkyPix`, `reflectRefractMaxDepth == 0` and `reflRefrEarlyOut`, so no host-side gating exists
// inside this module - the coordinator calls Render only while `reflectRefractMaxDepth > 0`
// (VulkanDevice.cpp:1011-1014).
//
// The image-state contract (a5b §6 risks 4/6): the engine leaves every framebuffer image in
// VK_IMAGE_LAYOUT_GENERAL (= NVRHI's UnorderedAccess) and a native wrap keeps no state between
// command lists (RhiTextureSource.h), so this pass announces that state for all 24 set-1 images
// before its first binding, which turns every automatic barrier into a same-layout UAV barrier or
// an SRV transition that starts from the truth (the primary's contract). After the dispatch the 23
// UAV images end the list in GENERAL already; image 25 was bound as an SRV only, so its
// Texture_SRV requirement left it in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
// (vulkan-resource-bindings.cpp:398-435) and this module moves exactly that one image back to
// UnorderedAccess - the next frame's primary pass writes it through its own wrap and the engine's
// present expects GENERAL. The restore list is exactly {25}.
//
// The interface assumption this module is written against: stream S1's fix in `RaygenPrimary.hlsli`
// - the fifteen sampled reads of the conflicting images converted to storage-image reads, the
// legacy twin optionally fixed for source hygiene only. That fix removes the `_Sampled`
// declarations of images 0, 2, 3, 5, 7, 9, 13, 19, 26, 59, 61, 83, 87, 88 and 90 from the Q2
// blob and leaves set 1 at exactly the 24 items the .cpp's table lists; the module's layout
// declares exactly those and nothing else. If the blob in `vkpt/Build` is still the pre-fix 39-item
// one, the reflrefr pipeline's set 1 would be refused at dispatch time by the validation device
// (the unfilled pre-fix items) - rebuild the shader before running the pass. The other
// measured facts (SpecId 0, miss 0, four set-8 items, the nine declared sets) do not depend on the
// fix.
//
// Host contract (the skeleton wires the pass, the pass only records):
//  - Create after RhiRtPrimaryPass was created: the pass borrows that module's TLAS, uniform,
//    vertex-data and ray-stats layout handles, its empty layout and empty set, and the shared
//    texture table; all of them have to outlive this object.
//  - SetPortalBuffer once after Create, when the coordinator has wrapped the engine's
//    `PortalList` device-local buffer (4032 B) as a static constant-buffer wrap. `PortalList` owns
//    the buffer; the stream S3 accessors hand the coordinator its staging slot, its device-local
//    buffer and the byte size. The coordinator wraps the device-local buffer with a BufferDesc
//    of at least `PORTAL_MAX_COUNT * sizeof(ShPortalInstance)` bytes, `isConstantBuffer = true`,
//    not volatile, resting in `ResourceStates::ConstantBuffer` with `keepInitialState = true`
//    (the world-uniform wrap's shape, NvrhiFrameSkeleton.cpp:329-342), records the per-frame
//    staging -> device-local copy on the RHI command list before `Render`, and only then calls
//    this setter. The pass takes the raw pointer as its cache key and builds one set per frame
//    slot over the module's set-9 layout; a changed pointer rebuilds the sets through the retire
//    queue, and a null pointer (or a buffer outside the shape above) is refused with a one-shot
//    warning and makes Render skip until a valid buffer arrives. The module does not record the
//    copy itself.
//  - Render on the slot's open command list (RhiFrameContext::BeginSlot first), after
//    `RhiRtPrimaryPass::Render` and the input side of the god rays, before the reflected-segment
//    god-rays dispatch (see the ordering paragraph above) and the compose chain's gradient
//    reproject, with the slot's top-level AS, the wrapped global uniform, the
//    same seven vertex-data buffers the primary got (this pass builds its own set over the
//    primary's layout handle), the engine's Framebuffers object and the render resolution. The
//    gate `globalUniform.reflectRefractMaxDepth > 0` is the caller's; this module never looks at
//    the uniform's contents.
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images.
//
// Must not pull in: the god rays and their shadow map (A5.2), the real cubemap content (A5a), the
// volumetric path (dead in this tree), the legacy `RtRaygenReflRefr`, the raster overlay, the
// upscalers, and the portals' *content* - only the set-9 binding of the PortalList buffer is this
// module's, the upload and the copy stay the coordinator's.
//
// The pass is a no-op until Create succeeded and while an input is missing (no TLAS, no
// framebuffers, no uniform, no vertex data, no portal buffer, an image handle the RHI cannot
// wrap); every early return is quiet after the first warning. It is not thread-safe: Render uses
// the per-slot target of the frameIndex it is given, which is the engine's single-threaded per-slot
// frame model (RhiFrameContext).
class RhiRtReflRefrPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiRtReflRefrPass();
    ~RhiRtReflRefrPass();

    RhiRtReflRefrPass(const RhiRtReflRefrPass &other) = delete;
    RhiRtReflRefrPass(RhiRtReflRefrPass &&other) noexcept = delete;
    RhiRtReflRefrPass &operator=(const RhiRtReflRefrPass &other) = delete;
    RhiRtReflRefrPass &operator=(RhiRtReflRefrPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queue every replaced wrap and set goes
    // through; 'pTextureTable' is the host's shared bindless table (RHI/RhiTextureTable.h), whose
    // layout becomes set 4 and whose table is bound with it; 'pPrimaryPass' is the created
    // primary-visibility pass (RHI/RhiRtPrimaryPass.h), which owns the shared set layouts and sets
    // this pipeline and its states use. None of the arguments is owned; all have to outlive this
    // object, and a null or unusable one makes Create fail. 'pShaderFolderPath' is the folder
    // ShaderManager loads the engine blobs from, with the trailing separator; the four RT blobs
    // above are read from it. The pass logs through 'pfnPrint'. Returns false and leaves the pass
    // unusable if a shader, a layout, a resource, the pipeline or the shader table cannot be
    // created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                rhi::RhiTextureTable *pTextureTable,
                const RhiRtPrimaryPass *pPrimaryPass,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // Set 9's one buffer: the coordinator's static constant-buffer wrap of the engine's PortalList
    // device-local array (63 x 64 B); see the class comment's host contract for how and when the
    // wrap and the per-frame copy are made. The handle has to satisfy the shader's contract
    // (`ConstantBuffer<PortalInstances_BT>`, 63 `ShPortalInstance` records), so the setter enforces
    //   desc.isConstantBuffer, !desc.isVolatile, desc.byteSize >= 63 * 64,
    // and a desc outside it is refused with a one-shot warning - the pass then keeps skipping until
    // a valid buffer arrives. The pass holds no reference to the buffer (the per-slot sets do); it
    // keeps the raw pointer only as the sets' cache key, exactly as the primary pass keeps its
    // uniform pointer. Calling the method again with a different buffer rebuilds the per-slot sets
    // through the retire queue, and passing null clears the key (Render then warns once and skips).
    void SetPortalBuffer(nvrhi::IBuffer *pPortalBuffer);

    // One call per frame, on the frame context's open command list of 'frameIndex', after
    // RhiRtPrimaryPass::Render and the input side of the god rays, before the reflected-segment
    // god-rays dispatch and the compose chain's gradient reproject. The caller applies the legacy
    // gate itself: `globalUniform.reflectRefractMaxDepth > 0` (VulkanDevice.cpp:1011-1014).
    // 'pTopLevel' is the slot's top-level structure, 'pUniformBuffer' the engine's global uniform
    // as a static constant-buffer wrap (the same wrap the primary takes), 'vertexData' the seven
    // set 3 buffers ('RhiRtPrimaryPass::VertexData'; this pass builds its own set over the
    // primary's layout handle), and 'pFramebuffers' the engine's framebuffer registry.
    // 'width'/'height' are the render resolution the images are sized to and the dispatch extents
    // (one ray per pixel, never halved).
    //
    // What is recorded: the wraps of the 24 set-1 images (created on first use, re-created when the
    // engine re-created an image or the size changed; the replaced wraps and the sets over them go
    // through the frame context's retire queue), the per-slot sets 0-3 and 9, the clear of the
    // render-cubemap placeholder on the first list that binds it, then one
    // `dispatchRays(width, height, 1)`.
    //
    // The image state contract, spelled out on the class: UnorderedAccess is announced for all 24
    // images before the first binding, and after the dispatch exactly image 25 - the only SRV-only
    // image - is moved back to UnorderedAccess; the 23 UAV images end the list in GENERAL.
    //
    // No-op when the pass is not created, the frame index is out of range, the size is zero, the
    // portal buffer was not set, the TLAS or the framebuffers are missing, or the uniform or
    // vertex-data inputs are missing or in a shape NVRHI's validation refuses.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                nvrhi::rt::IAccelStruct *pTopLevel,
                nvrhi::IBuffer *pUniformBuffer,
                const RhiRtPrimaryPass::VertexData &vertexData,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height);

    // Drops every slot's image wraps and the sets over them, the per-slot TLAS/uniform/vertex-data
    // sets and the per-slot set-9 sets, and retires them through the frame context's queue. The
    // caller has to call it before the engine destroys its framebuffer images (the
    // Framebuffers::PrepareForSize path) - otherwise the wraps reference destroyed VkImages. The
    // next Render re-reads the handles and re-wraps, so the pass survives a resize without a second
    // Create; the set-9 sets are rebuilt over the same buffer pointer. The module-owned placeholder
    // cubes and the render-cubemap set do not reference framebuffer images and survive.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: every framebuffer image is a per-slot (swapped) image and
    // the vertex-data buffers are the slot's RHI copies, so neither the sets over them can be
    // shared across slots. Set 9's set follows the portal buffer pointer the way the primary's
    // uniform set follows the uniform pointer.
    struct Target
    {
        // Set 1: the 24 engine images (the .cpp's FRAMEBUFFER_BINDINGS table) the slot currently
        // wraps and the set over them. The handles are kept in the form Render received them, not
        // as VkImages, because they are what the change detection compares; a change in any of them
        // or in the size means the engine re-created the framebuffers and the wraps and the set
        // have to follow.
        uint64_t imageHandles[24] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle framebufferTextures[24];
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

        // Set 9: the portal buffer the set was built over (the module's SetPortalBuffer pointer is
        // the key) and the set itself.
        nvrhi::IBuffer *portalBuffer = nullptr;
        nvrhi::BindingSetHandle portalSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Retires the slot's image wraps and the framebuffer set over them (the framebuffer re-create
    // path), and the slot's TLAS/uniform/vertex-data/portal sets (ReleaseTargets).
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    // The per-slot sets 0-3, rebuilt when their key changed. Return false when the input is
    // unusable for this frame (the caller skips the dispatch).
    bool PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel);
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);
    bool PrepareVertexDataSet(Target &target, const RhiRtPrimaryPass::VertexData &vertexData);

    // Set 9: builds (or rebuilds, when the pointer changed) the one-item set over the module's
    // portal layout. Returns false when no valid buffer was set (a one-shot warning).
    bool PreparePortalSet(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, the shared texture table and the primary pass, all
    // outlive this object.
    rhi::RhiFrameContext *frameContext = nullptr;
    rhi::RhiTextureTable *textureTable = nullptr;
    const RhiRtPrimaryPass *primaryPass = nullptr;

    // The four engine blobs. 'specializedRaygenShader' is the RGenQ2ReflRefr module with its
    // SpecId 0 set to the game's primaryRaysMaxAlbedoLayers (2); it is what the pipeline uses, and
    // it keeps the base module alive through its own reference. There is no shadow-miss shader: the
    // raygen traces no shadow rays.
    nvrhi::ShaderHandle raygenShader;
    nvrhi::ShaderHandle specializedRaygenShader;
    nvrhi::ShaderHandle missShader;
    nvrhi::ShaderHandle closestHitShader;
    nvrhi::ShaderHandle anyHitShader;

    // The four layouts this module owns: set 1 (the exact 24 items above), set 9 (the one portal
    // constant buffer), set 7 (a bindless cube table like the siblings' placeholder) and set 8
    // (the four render-cubemap items, unlike the primary's two). Every other layout is the
    // primary's own handle.
    nvrhi::BindingLayoutHandle framebufferLayout;
    nvrhi::BindingLayoutHandle portalLayout;
    nvrhi::BindingLayoutHandle cubemapLayout;
    nvrhi::BindingLayoutHandle renderCubemapLayout;

    // Set 9's key (the coordinator's wrap) and the one-shot warnings of its shape.
    nvrhi::IBuffer *portalBuffer = nullptr;
    bool warnedBadPortalBuffer = false;
    bool warnedMissingPortalBuffer = false;

    // The placeholder cube of sets 7 and 8: one 1x1 RGBA8_UNORM TextureCube, one engine sampler,
    // the 32-slot bindless table and the four-item render-cubemap set, all module-owned (until the
    // real cubemaps get RHI accessors, A5a). 'dummyCubemapCleared' makes the black clear of the
    // set-8 dummy a one-off; the default `rt_physical_sky 1` path samples `renderCubemapEnv` and
    // must not read undefined contents.
    nvrhi::TextureHandle dummyCubemapTexture;
    nvrhi::SamplerHandle dummyCubemapSampler;
    nvrhi::DescriptorTableHandle cubemapTable;
    nvrhi::BindingSetHandle renderCubemapSet;
    bool dummyCubemapCleared = false;

    // The pipeline and its table: one raygen (RGenQ2ReflRefr), the engine's default miss and the
    // two engine hit groups. Both are created once; the table is uncached, so the backend bakes it
    // per list.
    nvrhi::rt::PipelineHandle pipeline;
    nvrhi::rt::ShaderTableHandle shaderTable;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // One-shot warnings for the inputs that can legitimately be missing for a few frames or are a
    // permanent host-side mistake.
    bool warnedMissingTopLevel = false;
    bool warnedMissingFramebuffers = false;
    bool warnedMissingUniform = false;
    bool warnedBadUniform = false;
    bool warnedMissingVertexData = false;
    bool warnedBadVertexData = false;
    bool warnedUnexpectedSize = false;

    bool created = false;
};

}
