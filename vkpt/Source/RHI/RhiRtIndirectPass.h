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

class RhiRtDirectPass;

// The RHI module of the renderer's third ray-tracing pass: the engine's indirect / GI raygen
// `RGenQ2Indirect` (`RtQ2Indirect.rgen.spv`) with the same two misses and two hit groups the primary
// and direct passes carry, dispatched over the RHI acceleration structures after the direct pass. It
// reads the G-buffer the primary pass wrote and the direct pass's specular and view-direction
// storage images, and writes the three spherical-harmonics images 16-18 (plus its share of 15 and
// 23) that the compose chain's adapter consumes. It is stream 3 of the A4.3 cut of the refactor
// plan, the A4.3 reconnaissance (a43_recon.md) §1-§6.2; the sibling of RhiRtPrimaryPass and
// RhiRtDirectPass, dispatched on the same command list directly after the direct pass (the legacy
// order is direct `VulkanDevice.cpp:1039` -> indirect `:1043`, both before the denoise and compose
// chain).
//
// Why this pass needs its own pipeline and shader table (a43_recon.md §6.2): the indirect raygen
// declares set 5 (the blue-noise array, which neither of the other two blobs declares), a set-1
// footprint that differs from both (the direct pass's 12 items, the primary's 26 UAVs), and a
// four-item set 8 (the primary's render-cubemap layout has two items). The pinned NVRHI refuses a
// binding set that leaves any item of its layout unfilled (validation-device.cpp:1855-1871) and the
// backend accepts a state's set only when it was created over the very layout handle the pipeline
// declared (validation-commandlist.cpp:509-520), so one pipeline cannot serve all three raygens.
// Like the two siblings this module therefore declares twelve layout positions and shares the
// handles it can: the primary's TLAS/uniform/vertex-data layouts (0/2/3) and its empty and
// ray-stats layouts and sets (9/10/11), the shared texture table's layout and table (4), and the
// direct pass's light layout and per-slot set (6, new read-only accessors on that module). It owns
// three layouts: the exact 14-item set 1 below, the one-item set 5, and the four-item set 8; set 7
// needs a bindless layout and is owned as a placeholder too (see below).
//
// The raygen's interface (measured 2026-09-25 with `spirv-dis` over `vkpt/Build/RtQ2Indirect.rgen.spv`,
// both the pre-fix blob of 101,596 B and the post-fix one S1's rebuild left in the tree; the command
// is the one GenerateShaders.py:157-164 produces, dxc from Vulkan SDK 1.4.321.1, and an independent
// compile of the fixed source reproduces the tree's blob byte for byte):
//   set 0  TLAS                - RhiAccelStructs::GetTopLevel, raw binding 0;
//   set 1  framebuffers        - **14 items after S1's fix** (the pre-fix blob had 15): the UAVs at
//                                raw bindings 15 (framebufUnfilteredSpecular, r32ui), 16/17/18
//                                (framebufUnfilteredIndirectSH_R/G/B, rgba16f), 23
//                                (framebufViewDirection, rgba16f), and the SRVs at raw bindings 124
//                                (framebufAlbedo_Sampled), 126 (IsSky), 127 (Normal), 129
//                                (NormalGeometry), 131 (MetallicRoughness), 143 (SurfacePosition),
//                                239 (Q2GradSmplPos), 245 (Q2RngSeed) and 247 (Q2Cluster). The pre-fix
//                                blob's binding 139 (framebufUnfilteredSpecular_Sampled) was the
//                                sampled view of the same image 15 the raygen writes; S1's storage
//                                read replaced `texelFetchUnfilteredSpecular(pix)`
//                                (RtQ2Indirect.rgen.hlsl:523 -> GLSL/RtQ2Indirect.rgen:470;
//                                a43_recon.md §4.2) and DXC dead-stripped the sampled resource, so the
//                                descriptor list has 36 pairs instead of 37 and no image is bound as
//                                both an SRV and a UAV. The nine remaining SRVs map to images
//                                0/2/3/5/7/19/115/121/123, none of which is a UAV of this blob;
//   set 2  global uniform      - the engine's wrapped ShGlobalUniform, raw binding 0;
//   set 3  vertex data         - bindings 0-4 only (the alpha-tested any-hit's triangle fetches); the
//                                primary's 7-item layout and this module's own 7-item set over it;
//   set 4  textures            - the shared bindless table (RhiTextureTable), its own two arrays;
//   set 5  random              - one `Texture2DArray<float4>` SRV at raw binding 0 (BINDING_BLUE_NOISE,
//                                `blueNoiseTextures`), no sampler. The engine's BlueNoise asset is
//                                128x128 RGBA8_UNORM with 128 array layers (BlueNoise.cpp:51-55,
//                                66-74; ShaderCommonC.h:120-122) and rests read-only;
//   set 6  light sources       - the 5 engine buffers at raw bindings 0 (lightSources,
//                                StructuredBuffer<ShLightEncoded>, 144 B stride), 4
//                                (q2LightListOffsets), 5 (q2LightListLights), 6 (q2LightStats, the
//                                only UAV) and 8 (q2ClusterSkyVis) - exactly the direct pass's five
//                                (a43_recon.md §1.1). This module binds the direct pass's own layout
//                                handle and per-slot set: that pass already wraps the buffers and
//                                records the four light copies, and the copies must have one consumer
//                                (a43_recon.md §6.2). If the direct pass did not render the slot this
//                                frame there is no set and this pass skips;
//   set 7  cubemaps            - the `globalCubemaps[]` bindless array and its sampler, raw bindings
//                                0/1. The primary's cubemap layout/table are private and have no
//                                accessors, so this module owns a second bindless layout and fills it
//                                with a 1x1 dummy cube in every slot, exactly like the primary's
//                                placeholder (RhiRtPrimaryPass.cpp:435-493; a43_recon.md §8.9);
//   set 8  render cubemap      - FOUR items, unlike the primary's two: `renderCubemap` at raw 0,
//                                `renderCubemapEnv` at raw 1 and their samplers at raw 2/3
//                                (BINDING_RENDER_CUBEMAP*). The module owns a 4-item layout and one
//                                dummy cube set; the dummy cube is cleared to black once, so the
//                                PROCEDURAL sky path the default `rt_physical_sky 1` selects reads a
//                                defined value instead of the primary placeholder's undefined
//                                contents (a43_recon.md §8.3). SetRenderCubemaps replaces the
//                                placeholder pair with the procedural-sky module's real cubes;
//   set 9/10 (portals/volumetric) - the primary's empty set; neither set is declared by the blob;
//   set 11 ray stats           - the primary's RHI-owned RWStructuredBuffer<RtRayStats> stand-in,
//                                raw binding 0.
//
// The raygen has exactly one specialization constant, `maxAlbedoLayerCount` (SpecId 0, uint, the
// SPIR-V default 0; RtQ2Indirect.rgen.hlsl:73-74), which the engine passes as
// `indirectIlluminationMaxAlbedoLayers` and the game sets to 1 (RayTracingPipeline.cpp:100,
// Quake/gl_vidsdl.c:1459). The pipeline therefore uses an `createShaderSpecialization` with
// UInt32(0, 1) - the direct sibling needs none, the primary uses 2. The payloads are ShPayload = 16 B
// (used by the bounce rays) and ShPayloadShadow = 4 B; the measured entry-point interface names both,
// and the pipeline also carries the two misses and the two hit groups, so the pipeline keeps the
// engine's per-pipeline 16 B / 8 B / recursion-depth-2 limits (a43_recon.md §1.3).
//
// The shader-binding table: one raygen, `RtMiss` at miss index 0 and `RtMissShadowCheck` at index 1,
// the fully-opaque hit group at offset 0 and the alpha-tested one at offset 1 - the same shape and
// the same engine SBT indices the primary and direct passes build. The bounce rays enter miss 0, the
// three shadow sites enter miss 1 (a43_recon.md §1.4); `RtMiss` has to fill index 0 although the
// raygen's bounce path is the only user of it. NVRHI has one raygen per table, so this module owns
// its own uncached table; the engine's raygen index for this shader (SBT_INDEX_RAYGEN_Q2_INDIRECT = 4)
// has no NVRHI counterpart.
//
// The dispatch (a43_recon.md §5): the normal mode is `width x height`; the half mode is
// `width x (height + 1) / 2` - the height is halved, the width stays full. The host chooses the mode
// exactly as PathTracer.cpp:203-210 does, from `globalUniform.giBounceRays[0]`: the value in
// (0.25, 0.75) selects the half dispatch. The raygen itself doubles the row back and picks the
// checkerboard parity from `globalUniform.frameId & 1`, and stores zeros when `giBounceRays[0] <
// 0.25`, so no host-side skip exists for `rt_gi_level 0`. The dispatch index is the packed texel, so
// the images keep the render extents and `width`/`height` are the full render resolution.
//
// The image-state contract (a43_recon.md §3.2): the engine leaves every framebuffer image in
// VK_IMAGE_LAYOUT_GENERAL (= NVRHI's UnorderedAccess) and a native wrap keeps no state between
// command lists, so this pass announces that state for all 14 set-1 images before its dispatch and
// moves the 9 images it binds as SRVs only back to UnorderedAccess after it. The five UAV images
// (15-18, 23) end the list in GENERAL already; the direct pass wrote 15 and 23 before this pass on
// the same list and the automatic barriers between the two dispatches order those writes before
// these reads through the shared images (each pass has its own wrap objects, but the states are the
// engine's real layouts). The next frame's compose chain re-announces UnorderedAccess for its own
// wraps of 16-18, and the end-of-chain restore (stream 4) is what keeps the following frame's UAV
// writes legal. The blue-noise texture is engine-owned and never changes state: the module announces
// NonPixelShaderResource - the state its AllRayTracing-visibility Texture_SRV binding requires,
// which maps to the same SHADER_READ_ONLY_OPTIMAL layout the engine leaves it in - on every list
// before binding set 5. The two placeholder cubes are module-owned; the clear of the render-cubemap
// dummy is recorded once on the first list that binds it.
//
// Host contract (the skeleton wires the pass, the pass only records):
//  - Create after RhiRtPrimaryPass and RhiRtDirectPass were created: the pass shares the primary's
//    layouts/sets and the direct pass's light layout/set, both of which have to outlive it.
//  - SetBlueNoiseTexture once after Create (and only again if the wrap is replaced): the texture is
//    the coordinator's wrap of the engine's BlueNoise image (stream 2's array-texture helper). The
//    pass takes a reference but does not own the image. A null call clears it; Render then warns
//    once and skips.
//  - SetRenderCubemaps once after the procedural-sky module (stream S1, RhiProceduralSkyPass) was
//    created and before the first Render (and only again if the cubes are replaced): the real
//    `renderCubemap`/`renderCubemapEnv` pair and the module's LINEAR/REPEAT sampler. Set 8 keeps
//    the black-cleared placeholder pair until then, a null call restores it, and the module's
//    textures have to outlive this pass's use.
//  - Render after RhiRtDirectPass::Render on the same list and before the compose chain: the raygen
//    reads the direct pass's 15/23 writes and the primary's G-buffer, and the compose chain's
//    adapter reads the 16-18 this pass writes. Pass the same `frameIndex`, the slot's TLAS, the
//    uniform wrap, the vertex-data buffers and the framebuffers the direct pass got, the render
//    resolution, and the `giBounceRays[0]` of the uniform this frame's raygen reads.
//  - The gradient-sample image 115 follows the engine's switch (A4.5): when
//    `globalUniform.fltEnable[0] >= 0.5` the raygen's gradient path (`q2GetIsGradient`,
//    Q2LightLists.hlsli:141-154) reads image 115, which the host's ordering has the compose module's
//    reproject write before the direct pass; with the switch below 0.5 no reproject runs and the
//    raygen keeps to the non-gradient path.
//  - Call ReleaseTargets() before Framebuffers::PrepareForSize destroys the framebuffer images.
//
// The pass is a no-op until Create succeeded and while an input is missing (no blue-noise texture,
// no direct-pass light set, no TLAS, no framebuffers, no uniform, no vertex data, an image handle
// the RHI cannot wrap); every early return is quiet after the first warning. It is not thread-safe:
// Render uses the per-slot target of the frameIndex it is given, which is the engine's
// single-threaded per-slot frame model (RhiFrameContext).
class RhiRtIndirectPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiRtIndirectPass();
    ~RhiRtIndirectPass();

    RhiRtIndirectPass(const RhiRtIndirectPass &other) = delete;
    RhiRtIndirectPass(RhiRtIndirectPass &&other) noexcept = delete;
    RhiRtIndirectPass &operator=(const RhiRtIndirectPass &other) = delete;
    RhiRtIndirectPass &operator=(RhiRtIndirectPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queue every replaced wrap and set goes
    // through; 'pTextureTable' is the host's shared bindless table (RHI/RhiTextureTable.h), whose
    // layout becomes set 4 and whose table is bound with it; 'pPrimaryPass' is the created
    // primary-visibility pass (RHI/RhiRtPrimaryPass.h), which owns the shared set layouts, the empty
    // set and the ray-stats set this pipeline and its states use; 'pDirectPass' is the created
    // direct-lighting pass (RHI/RhiRtDirectPass.h), whose set-6 light layout and per-slot light sets
    // this pass binds instead of wrapping the engine's light buffers a second time.
    // There is no LightManager argument: the light data path stays single-consumer - the direct pass
    // wraps the buffers, records the four copies and consumes the copy flags, and this module only
    // reads through the same NVRHI objects (a43_recon.md §6.2).
    // None of the arguments is owned; all have to outlive this object, and a null or unusable one
    // makes Create fail. 'pShaderFolderPath' is the folder ShaderManager loads the engine blobs
    // from, with the trailing separator; the five RT blobs above are read from it. The pass logs
    // through 'pfnPrint'. Returns false and leaves the pass unusable if a shader, a layout, a
    // resource, the pipeline or the shader table cannot be created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                rhi::RhiTextureTable *pTextureTable,
                const RhiRtPrimaryPass *pPrimaryPass,
                const RhiRtDirectPass *pDirectPass,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // Set 5's one texture: the coordinator's wrap of the engine's BlueNoise image, mirroring
    // RhiSkyPass::SetGeometryBuffers. The handle has to satisfy the shader's contract
    // (`Texture2DArray<float4>`, `blueNoiseTextures.Load(int4(x, y, layer, 0))` with
    // `layer = (texIndex + salt) % BLUE_NOISE_TEXTURE_COUNT`, Random.hlsli:257-266); the shape the
    // setter enforces is
    //   dimension Texture2DArray, format RGBA8_UNORM, width/height >= BLUE_NOISE_TEXTURE_SIZE (128),
    //   arraySize >= BLUE_NOISE_TEXTURE_COUNT (128),
    // and a desc outside it is refused with a one-shot warning - the pass then keeps skipping until a
    // valid texture arrives. `isShaderResource` (which the RHI's array-texture helper sets) is not
    // checked: on a native wrap the flag is descriptive and the validation device does not consult it
    // for an SRV binding.
    // The wrap must be in (or claim) the engine's resting
    // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL and must not be used by another pass with a different
    // state: this module announces NonPixelShaderResource for it before binding set 5 on every list,
    // which is the state an AllRayTracing visibility Texture_SRV binding requires and the NVRHI name
    // of that layout. The pass holds a reference to the handle but does not own the image; calling
    // the method again with a different texture replaces the set-5 set (the replaced one goes through
    // the retire queue), and passing null clears it.
    void SetBlueNoiseTexture(nvrhi::ITexture *pTexture);

    // Set 8's real content: the coordinator's `renderCubemap`/`renderCubemapEnv` pair and their
    // sampler - stream S1's RhiProceduralSkyPass objects, handed over as bare pointers (its
    // `GetCubemapTexture()`, `GetEnvironmentTexture()` and `GetCubemapSampler()`). The blob
    // declares all four items (measured over RtQ2Indirect.rgen), so both cubes are required;
    // one sampler fills both sampler positions, exactly as the placeholder pair does and as the
    // engine's set does (both engine SAMPLER bindings name the same LINEAR/REPEAT sampler,
    // RenderCubemap.cpp:696-697).
    //
    // The coordinator calls it once after RhiProceduralSkyPass was created and after this pass's
    // Create, before the first Render; the module owns the objects and they have to outlive this
    // pass's use (the pass keeps references but does not own the images), and the module's cubes
    // keep NonPixelShaderResource as their initial state (RhiProceduralSkyPass), which is the
    // state this set's Texture_SRV bindings require, so they issue no texture transition. Both
    // cubes have to satisfy the shader's `renderCubemap`/`renderCubemapEnv` `SampleLevel(..., lod)`
    // contracts and the engine render cubemap's shape (RenderCubemap.cpp:34-37, :489-503), so the
    // setter enforces, for each,
    //   dimension TextureCube, format RGBA16_FLOAT (R16G16B16A16_SFLOAT), arraySize 6,
    //   width == height > 0,
    //   mipLevels >= 11 (the SKY_MIP_COUNT chain `getSkyFiltered` scales its lod against;
    //   RaygenCommon.hlsli:390-417)
    // and refuses a cube outside it with a one-shot warning - the black-cleared 1x1 placeholder
    // pair then stays in place. The sampler is expected to be the module's LINEAR/REPEAT one; a
    // deviation is reported once and the sampler is used as given. A null cube (either one) or a
    // null sampler restores the placeholder pair.
    // The set-8 set over the pair is rebuilt on the next Render and the replaced set goes through
    // the frame context's retire queue; the real cubes are not framebuffer images, so they and the
    // set survive ReleaseTargets exactly like the placeholder's.
    void SetRenderCubemaps(nvrhi::ITexture *pCubemap, nvrhi::ITexture *pEnvCubemap, nvrhi::ISampler *pSampler);

    // One call per frame, on the frame context's open command list of 'frameIndex', after
    // RhiRtDirectPass::Render and before the compose chain of the same slot. 'numBounceRays' is
    // `ShGlobalUniform::giBounceRays[0]` exactly as the host wrote it into this frame's uniform
    // (sky.uniform->GetData()->giBounceRays[0]) - the pass halves the dispatch height exactly when
    // it is in (0.25, 0.75), the condition of PathTracer.cpp:203-210, and the shader's own
    // half-res/zero behavior keys off the same uniform member. 'pTopLevel' is the slot's top-level
    // structure, 'pUniformBuffer' the engine's global uniform as a static constant-buffer wrap (the
    // same wrap the primary and direct passes take), 'vertexData' the seven set 3 buffers
    // ('RhiRtPrimaryPass::VertexData'; this pass builds its own set over the primary's layout
    // handle), and 'pFramebuffers' the engine's framebuffer registry. 'width'/'height' are the
    // render resolution the images are sized to; the pass resolves every image extent through
    // Framebuffers::GetImageHandles' 4-tuple form, so the one set-1 image that is not render-sized
    // (Q2_GRAD_SMPL_POS, 1/3) is wrapped at its own extent.
    //
    // What is recorded: the wraps of the 14 set-1 images (created on first use, re-created when the
    // engine re-created an image or the size changed; the replaced wraps and the sets over them go
    // through the frame context's retire queue), the per-slot sets 0-3, the set-5 set if the
    // blue-noise texture changed, the set-8 set if the render-cubemap pair changed, the clear of the
    // placeholder cube on the first list that binds the placeholder (the real cubes skip it), then
    // one `dispatchRays(width, halfRes ? (height + 1) / 2 : height, 1)` over the direct pass's set-6
    // set and the primary's sets 9/10/11. The five UAV images end the list in
    // UnorderedAccess; the 9 SRV-only images are moved back there after the dispatch, so the next
    // frame's writers (and this pass's own UAVs) find the layout the engine knows.
    //
    // No-op when the pass is not created, the frame index is out of range, the size is zero, the
    // blue-noise texture is not set, the direct pass has no light set for the slot, the TLAS or the
    // framebuffers are missing, or the uniform/vertex-data inputs are missing or in a shape NVRHI's
    // validation refuses.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                float numBounceRays,
                nvrhi::rt::IAccelStruct *pTopLevel,
                nvrhi::IBuffer *pUniformBuffer,
                const RhiRtPrimaryPass::VertexData &vertexData,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height);

    // Drops every slot's image wraps and the set over them and the per-slot TLAS/uniform/vertex
    // sets, and retires them through the frame context's queue. The caller has to call it before the
    // engine destroys its framebuffer images (the Framebuffers::PrepareForSize path) - otherwise the
    // wraps reference destroyed VkImages. The module-owned placeholders (the blue-noise set, the two
    // cubemap dummies and their sets) and the set-8 set over the real cubes after SetRenderCubemaps
    // do not reference framebuffer images and survive: the blue-noise asset, the dummies and the
    // procedural-sky cubes live across a resize. The next Render re-reads the handles and re-wraps,
    // so the pass survives a resize without a second Create.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: every framebuffer image is a per-slot (swapped) image and the
    // vertex-data buffers are the slot's RHI copies, so neither the sets over them can be shared
    // across slots. Set 6 is not stored: its set belongs to the direct pass and is fetched per
    // frame, because that pass may replace it (a light-buffer handle change) and the RHI list has to
    // bind whatever set the direct pass's own dispatch used.
    struct Target
    {
        // Set 1: the 14 engine images (the .cpp's FRAMEBUFFER_BINDINGS table) the slot currently
        // wraps and the set over them. The handles are kept in the form Render received them, not as
        // VkImages, because they are what the change detection compares; a change in any of them or
        // in the size means the engine re-created the framebuffers and the wraps and the set have to
        // follow.
        uint64_t imageHandles[14] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle framebufferTextures[14];
        nvrhi::BindingSetHandle framebufferSet;

        // Set 0: the pointer is only the cache key that tells whether the set still addresses the
        // current AS; the set itself holds the reference that keeps the AS alive.
        nvrhi::rt::IAccelStruct *topLevel = nullptr;
        nvrhi::BindingSetHandle tlasSet;

        // Set 2: the engine's global uniform never changes, but it is a Render argument here, so the
        // set follows the pointer the same way.
        nvrhi::IBuffer *uniformBuffer = nullptr;
        nvrhi::BindingSetHandle uniformSet;

        // Set 3: the seven buffers of this slot (the host's RHI copies can be re-created when a
        // later frame needs more room, so the pointers are the key).
        nvrhi::IBuffer *vertexBuffers[7] = {};
        nvrhi::BindingSetHandle vertexDataSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Retires the slot's image wraps and the framebuffer set over them (the framebuffer re-create
    // path), and the slot's TLAS/uniform/vertex-data sets (ReleaseTargets).
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    // The per-slot sets 0-3, rebuilt when their key changed. Return false when the input is unusable
    // for this frame (the caller skips the dispatch).
    bool PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel);
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);
    bool PrepareVertexDataSet(Target &target, const RhiRtPrimaryPass::VertexData &vertexData);

    // Set 5: builds (or rebuilds, when the texture pointer changed) the one-item set over the
    // blue-noise wrap and announces the state the binding requires on 'pCommandList'. Returns false
    // when no valid texture was set.
    bool PrepareBlueNoiseSet(nvrhi::ICommandList *pCommandList);

    // Set 8: builds (or rebuilds, when a key changed) the four-item set over the coordinator's
    // render-cubemap pair, or over the placeholder pair when none was set. Returns false when the
    // set cannot be created.
    bool PrepareRenderCubemapSet();

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model and shared texture table, the primary and direct passes, all
    // outlive this object.
    rhi::RhiFrameContext *frameContext = nullptr;
    rhi::RhiTextureTable *textureTable = nullptr;
    const RhiRtPrimaryPass *primaryPass = nullptr;
    const RhiRtDirectPass *directPass = nullptr;

    // The five engine blobs. 'specializedRaygenShader' is the RGenQ2Indirect module with its SpecId 0
    // set to the game's indirectIlluminationMaxAlbedoLayers (1); it is what the pipeline uses, and it
    // keeps the base module alive through its own reference.
    nvrhi::ShaderHandle raygenShader;
    nvrhi::ShaderHandle specializedRaygenShader;
    nvrhi::ShaderHandle missShader;
    nvrhi::ShaderHandle shadowMissShader;
    nvrhi::ShaderHandle closestHitShader;
    nvrhi::ShaderHandle anyHitShader;

    // The three layouts this module owns: set 1 (the exact 14 items above), set 5 (the blue-noise
    // array) and set 8 (the four render-cubemap items). Set 7 also needs its own bindless layout,
    // because the primary's is private; it is a placeholder like the primary's.
    nvrhi::BindingLayoutHandle framebufferLayout;
    nvrhi::BindingLayoutHandle blueNoiseLayout;
    nvrhi::BindingLayoutHandle cubemapLayout;
    nvrhi::BindingLayoutHandle renderCubemapLayout;

    // Set 5's texture and set. 'blueNoiseSetTexture' is the raw pointer the set was built over, so a
    // replaced wrap re-creates the set; the handle keeps the texture alive and the replaced set goes
    // through the retire queue.
    nvrhi::TextureHandle blueNoiseTexture;
    nvrhi::ITexture *blueNoiseSetTexture = nullptr;
    nvrhi::BindingSetHandle blueNoiseSet;

    // The two placeholder cube sets: the 32-slot bindless table for set 7 and the two-cube set for
    // set 8, all backed by the same 1x1 dummy texture (the primary pass's placeholder, until the
    // real cubemaps get RHI accessors - a43_recon.md §8.9). 'dummyCubemapCleared' makes the black
    // clear of the set-8 dummy a one-off; the clear is recorded only while the placeholder pair is
    // the current set (the real cubes need no clear).
    nvrhi::TextureHandle dummyCubemapTexture;
    nvrhi::SamplerHandle dummyCubemapSampler;
    nvrhi::DescriptorTableHandle cubemapTable;

    // Set 8's coordinator keys: the raw handles keep the real cubes and their sampler alive (the
    // coordinator's SetRenderCubemaps stores them), the raw pointers are the coordinator keys the
    // current set was built over, so a replaced key rebuilds it in PrepareRenderCubemapSet. Both
    // texture pointers null mean the placeholder pair is the current set (what the module starts
    // with).
    nvrhi::TextureHandle renderCubemapTexture;
    nvrhi::ITexture *renderCubemapSetTexture = nullptr;
    nvrhi::TextureHandle renderCubemapEnvTexture;
    nvrhi::ITexture *renderCubemapEnvSetTexture = nullptr;
    nvrhi::SamplerHandle renderCubemapSampler;
    nvrhi::ISampler *renderCubemapSetSampler = nullptr;
    nvrhi::BindingSetHandle renderCubemapSet;
    bool dummyCubemapCleared = false;

    // The pipeline and its table: one raygen (RGenQ2Indirect), the engine's two misses, the two
    // engine hit groups. Both are created once; the table is uncached, so the backend bakes it per
    // list.
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
    bool warnedMissingBlueNoise = false;
    bool warnedBadBlueNoise = false;
    bool warnedBadRenderCubemap = false;
    bool warnedRenderCubemapSampler = false;
    bool warnedMissingLightSet = false;

    bool created = false;
};

}
