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
class RhiTextureTable;
}

// The RHI module of the renderer's first real ray-tracing pass: the engine's primary-visibility
// raygen `RGenPrimary` (`RtRaygenPrimary.rgen.spv`) with the engine's two miss shaders and two hit
// groups, dispatched over the RHI acceleration structures into the engine's ALBEDO framebuffer
// storage image and the rest of the G-buffer the raygen writes (26 storage images in total). It is
// the pass the A4.1 cut of the refactor plan describes ("the primary visibility pass into ALBEDO",
// a4_recon §6), the successor of the A3.1 debug trace: the same plumbing - an `rt::IPipeline`, an
// uncached `rt::IShaderTable`, the per-list engine-image state announcements, the `Retire`
// contract - with the engine's real shaders, binding sets and shader-binding-table layout behind
// it.
//
// The module was created under the temporary "direct" name and renamed to "primary" once its
// raygen was confirmed against the shipped blobs: `RGenPrimary` is the visibility raygen that
// writes the G-buffer and ALBEDO, and the engine's `RGenDirect` cannot run until that G-buffer
// exists (a4_recon §6), so the primary pass is the first cut. The direct-lighting chain of A4.2
// becomes a sibling module; the layouts, the SBT shape and the per-slot machinery here are what it
// will reuse. The five shaders are the union of the A4.1 stage set and the shadow miss, so the
// shader table carries the engine's miss and hit-group indices unchanged.
//
// The pipeline, in the engine's RT order (RayTracingPipeline.cpp:62-86; the order differs from the
// raster passes' and must not be re-derived from them):
//   set 0  TLAS                - the slot's rt::IAccelStruct from RhiAccelStructs::GetTopLevel;
//   set 1  framebuffers        - the 26 storage images the raygen writes, at their engine binding
//                                numbers (= their image indices);
//   set 2  global uniform      - the engine's wrapped ShGlobalUniform, raw binding 0;
//   set 3  vertex data         - the engine's vertex/index/instance storage buffers;
//   set 4  textures            - the shared bindless table (RhiTextureTable), its own two arrays;
//   set 5  (random)            - empty: `RGenPrimary` declares no set 5;
//   set 6  (light sources)     - empty: no light data yet (see Render);
//   set 7  cubemaps            - a bindless table with a 1x1 dummy cube in every slot (placeholder
//                                until the real cubemaps have RHI accessors, a4_recon §5);
//   set 8  render cubemap      - `renderCubemap` and its sampler: the procedural sky's real cube
//                                once the host handed it over (SetRenderCubemap), a 1x1 dummy
//                                cube until then;
//   set 9  (portals)           - empty: `RGenPrimary` declares no set 9;
//   set 10 (volumetric)        - empty: no RT shader declares set 10;
//   set 11 ray stats           - an RHI-owned RWStructuredBuffer<RtRayStats> stand-in.
// Twelve layouts, so the pinned NVRHI's binding-layout cap has to be the raised one: the module
// is written against `c_MaxBindingLayouts == 16` (the patch build_win.ps1 applies for the duration
// of a build, third_party/nvrhi-max-binding-layouts.patch). Every set is laid out with
// `ShaderType::AllRayTracing` visibility, the same choice RhiDebugTracePass documents and the A3.1
// gate proved validation-clean: the acceleration-structure-read state names the ray-tracing and
// the compute stages (vulkan-constants.cpp:282-285), so the descriptor bindings stay in the stages
// these shaders actually run in.
//
// What the pass needs from the host:
//  - the RHI device, the frame context, the shared texture table and the shader folder once, in
//    Create() - the pass loads the five engine blobs itself the way RhiDebugTracePass does. The
//    texture table is an argument Create() adds to the A4 sketch's shape because set 4 has to be
//    the table's own layout in the pipeline; the table is host-owned and outlives the pass, as it
//    does for RhiSkyPass;
//  - one SetRenderCubemap() after the procedural-sky module (stream S1, RhiProceduralSkyPass) was
//    created and before the first Render, with that module's real cube and LINEAR/REPEAT sampler;
//    set 8 keeps the placeholder until then, a null call restores it, and the module's textures
//    have to outlive this pass's use;
//  - one Render() per frame on the slot's open command list (RhiFrameContext::BeginSlot first),
//    with the slot's top-level AS, the wrapped global uniform and the slot's vertex-data buffers
//    (the host owns their sources), and the engine's Framebuffers object (the pass resolves the 26
//    image handles itself, exactly the accessor RhiSkyPass uses).
//
// Everything the engine re-creates on resize (the 26 wrapped images) follows the ReleaseTargets()
// contract: the caller has to call it before Framebuffers::PrepareForSize destroys the images.
//
// The pass is a no-op until Create succeeded and while an input is missing (no TLAS, no
// framebuffers, no vertex data); every early return is quiet after the first warning. It is not
// thread-safe: Render uses the per-slot target of the frameIndex it is given, which is the
// engine's single-threaded per-slot frame model (RhiFrameContext).
class RhiRtPrimaryPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    // The engine's vertex-data storage buffers of set 3, in the binding order of
    // VertexData.hlsli / ShaderCommonC.h:19-26. The host owns the RHI buffers; the pass only binds
    // them. Binding 5 (`geomIndexPrevToCur`) is not in the list: `RGenPrimary` does not
    // reference it (spirv-dis of the shipped blob names bindings 0,1,2,3,4,6,7 only), so the
    // layout stays partial like the world pass's framebuffer layout.
    struct VertexData
    {
        nvrhi::IBuffer *staticVertices = nullptr;      // set 3 binding 0, StructuredBuffer<ShVertex>
        nvrhi::IBuffer *dynamicVertices = nullptr;     // set 3 binding 1, StructuredBuffer<ShVertex>
        nvrhi::IBuffer *staticIndices = nullptr;       // set 3 binding 2, StructuredBuffer<uint>
        nvrhi::IBuffer *dynamicIndices = nullptr;      // set 3 binding 3, StructuredBuffer<uint>
        nvrhi::IBuffer *geometryInstances = nullptr;   // set 3 binding 4, StructuredBuffer<ShGeometryInstance>
        nvrhi::IBuffer *dynamicVerticesPrev = nullptr; // set 3 binding 6, StructuredBuffer<ShVertex>
        nvrhi::IBuffer *prevDynamicIndices = nullptr;  // set 3 binding 7, StructuredBuffer<uint>

        bool IsComplete() const
        {
            return staticVertices != nullptr && dynamicVertices != nullptr &&
                   staticIndices != nullptr && dynamicIndices != nullptr &&
                   geometryInstances != nullptr && dynamicVerticesPrev != nullptr &&
                   prevDynamicIndices != nullptr;
        }
    };

    RhiRtPrimaryPass();
    ~RhiRtPrimaryPass();

    RhiRtPrimaryPass(const RhiRtPrimaryPass &other) = delete;
    RhiRtPrimaryPass(RhiRtPrimaryPass &&other) noexcept = delete;
    RhiRtPrimaryPass &operator=(const RhiRtPrimaryPass &other) = delete;
    RhiRtPrimaryPass &operator=(RhiRtPrimaryPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h)
    // that owns the per-slot command lists and the retire queues every replaced wrap and set goes
    // through; 'pTextureTable' is the host's shared bindless table (RHI/RhiTextureTable.h), whose
    // layout becomes set 4 and whose table is bound with it - neither is owned, both have to
    // outlive this object, and a null or unusable one makes Create fail. 'pShaderFolderPath' is the
    // folder ShaderManager loads the engine blobs from, with the trailing separator; the five RT
    // blobs above are read from it. The pass logs through 'pfnPrint'. Returns false and leaves the
    // pass unusable if a shader, a layout, a resource, the pipeline or the shader table cannot be
    // created.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                rhi::RhiTextureTable *pTextureTable,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // The handles the direct-lighting sibling (RhiRtDirectPass, A4.2a) shares instead of building
    // its own: the two pipelines cannot own equal-but-distinct layouts for the positions they share
    // - the backend accepts a state's binding set only when it was created over the very layout
    // handle the pipeline declared (validation-commandlist.cpp:509-520) - and the empty layout
    // (bound at 5, 7, 8, 9 and 10 by the sibling) and the ray-stats layout/set (11) are cheap to
    // hand over as they are. Read-only: the handles are owned here, stay valid while this object is
    // created, and the sibling must not outlive it.
    nvrhi::BindingLayoutHandle GetTlasLayout() const { return tlasLayout; }
    nvrhi::BindingLayoutHandle GetUniformLayout() const { return uniformLayout; }
    nvrhi::BindingLayoutHandle GetVertexDataLayout() const { return vertexDataLayout; }
    nvrhi::BindingLayoutHandle GetHoleLayout() const { return holeLayout; }
    nvrhi::BindingLayoutHandle GetRayStatsLayout() const { return rayStatsLayout; }
    nvrhi::BindingSetHandle GetHoleSet() const { return holeSet; }
    nvrhi::BindingSetHandle GetRayStatsSet() const { return rayStatsSet; }

    // Set 8's real content: the coordinator's render cubemap and its sampler - stream S1's
    // RhiProceduralSkyPass objects, handed over as bare pointers (its `GetCubemapTexture()` and
    // `GetCubemapSampler()`). The primary raygen's set 8 declares only `renderCubemap` and its
    // sampler (measured over RtRaygenPrimary.rgen; there is no env position), which is why this
    // setter takes one cube where the indirect and reflect/refract siblings' four-item sets take
    // two.
    //
    // The coordinator calls it once after RhiProceduralSkyPass was created and after this pass's
    // Create, before the first Render; the module owns the objects and they have to outlive this
    // pass's use (the pass keeps a reference but does not own the image), and the module's cube
    // keeps NonPixelShaderResource as its initial state (RhiProceduralSkyPass), which is the state
    // this set's Texture_SRV binding requires, so the binding issues no texture transition. The
    // cube has to satisfy the shader's `renderCubemap.SampleLevel(renderCubemap_Sampler, direction,
    // lod)` contract and the engine render cubemap's shape (RenderCubemap.cpp:34-37, :489-503), so
    // the setter enforces
    //   dimension TextureCube, format RGBA16_FLOAT (R16G16B16A16_SFLOAT), arraySize 6,
    //   width == height > 0,
    //   mipLevels >= 11 (the SKY_MIP_COUNT chain `getSkyFiltered` scales its lod against;
    //   RaygenCommon.hlsli:390-417)
    // and refuses a desc outside it with a one-shot warning - the 1x1 placeholder then stays in
    // place. The sampler is expected to be the module's LINEAR/REPEAT one; a deviation is reported
    // once and the sampler is used as given. A null cube or sampler restores the placeholder.
    // The set-8 set over the pair is rebuilt on the next Render and the replaced set goes through
    // the frame context's retire queue; the real cube is not a framebuffer image, so it and the
    // set survive ReleaseTargets exactly like the placeholder's.
    void SetRenderCubemap(nvrhi::ITexture *pCubemap, nvrhi::ISampler *pSampler);

    // One call per frame, on the frame context's open command list of 'frameIndex'. 'pTopLevel' is
    // the RHI acceleration-structure stream's top-level structure of the slot
    // (RhiAccelStructs::GetTopLevel); 'pUniformBuffer' is the engine's global uniform as a static
    // constant-buffer wrap (the same wrap NvrhiFrameSkeleton writes every frame); 'vertexData' are
    // the seven set 3 buffers; 'pFramebuffers' is the engine's framebuffer registry, and
    // 'width'/'height' are the render resolution the images are sized to.
    //
    // What is recorded: the wraps of the 26 storage images the raygen writes (created on first use,
    // re-created when the engine re-created an image or the size changed; the replaced wraps and
    // the sets over them go through the frame context's retire queue), the per-slot sets 0-3, the
    // set-8 set over the coordinator's render cubemap (rebuilt when its key changed), then one
    // `traceRays` at the full render resolution (one ray per pixel; the raygen maps the pixel
    // to the checkerboard itself through `getCheckerboardPix`, so no halving happens here).
    //
    // The image state contract: the engine leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL
    // (= NVRHI's UnorderedAccess), and a native wrap keeps no state between command lists, so the
    // pass announces that state for all 26 images before its first use on every list
    // (beginTrackingTextureState, exactly RhiDebugTracePass's contract). The Texture_UAV bindings
    // require the same state, so the announcement turns the automatic barrier into a same-layout
    // UAV barrier and the images stay in GENERAL - the layout the engine and the present expect. An
    // engine image therefore has to be in GENERAL when the list starts, which the traced frame
    // guarantees: the raster sky/world sub-passes are not recorded in that mode.
    //
    // Light: this pass traces visibility only - no light buffer is bound (set 6 stays empty), so
    // ALBEDO and the rest of the G-buffer come out without direct or indirect illumination. That is
    // the expected A4.1 image: the direct-lighting raygen (A4.2) is what turns the unfiltered
    // buffers into lit shading, the indirect raygen (A4.3) what adds GI, and the compose chain
    // (A4.4) what produces the final image. Until then the traced picture is expected to look flat
    // and unlit (a material/albedo pass), not wrong.
    //
    // No-op when the pass is not created, the frame index is out of range, the size is zero, the
    // TLAS or the framebuffers are missing, or the uniform/vertex-data inputs are missing or in a
    // shape NVRHI's validation refuses.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                nvrhi::rt::IAccelStruct *pTopLevel,
                nvrhi::IBuffer *pUniformBuffer,
                const VertexData &vertexData,
                const Framebuffers *pFramebuffers,
                uint32_t width,
                uint32_t height);

    // Drops every slot's image wraps and the sets over them, plus the per-slot TLAS/uniform/vertex
    // sets, and retires them through the frame context's queue. The caller has to call it before
    // the engine destroys its framebuffer images (the Framebuffers::PrepareForSize path) -
    // otherwise the wraps reference destroyed VkImages. The next Render re-reads the handles and
    // re-wraps, so the pass survives a resize without a second Create. The module-owned
    // placeholder cubes and the set-8 set (over the real render cubemap once SetRenderCubemap
    // delivered it) do not reference framebuffer images and survive. The destructor drops
    // directly, after a device idle.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: every image the engine owns is a per-slot (swapped) image
    // and the vertex-data buffers are the slot's RHI copies, so neither set can be shared across
    // slots.
    struct Target
    {
        // Set 1: the 26 engine images (the .cpp's FRAMEBUFFER_UAV_IMAGES table) the slot currently
        // wraps and the set over them. The handles are kept in the form Render received them, not
        // as VkImages, because they are what the change detection compares; a change in any of them
        // or in the size means the engine re-created the framebuffers and the wraps and the set
        // have to follow.
        uint64_t imageHandles[26] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        nvrhi::TextureHandle framebufferTextures[26];
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
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    // Retires the slot's image wraps and the framebuffer set over them (the framebuffer re-create
    // path), and the slot's TLAS/uniform/vertex-data sets (ReleaseTargets).
    void ReleaseFramebufferTarget(Target &target);
    void ReleaseTarget(Target &target);

    // The per-slot sets 0-3, rebuilt when their key changed. Return false when the input is
    // unusable for this frame (the caller skips the dispatch).
    bool PrepareTlasSet(Target &target, nvrhi::rt::IAccelStruct *pTopLevel);
    bool PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer);
    bool PrepareVertexDataSet(Target &target, const VertexData &vertexData);

    // Set 8: builds (or rebuilds, when a key changed) the one-cube set over the coordinator's
    // render cubemap, or over the placeholder pair when none was set. Returns false when the set
    // cannot be created; the caller skips the dispatch.
    bool PrepareRenderCubemapSet();

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model and shared texture table, both outlive this object.
    rhi::RhiFrameContext *frameContext = nullptr;
    rhi::RhiTextureTable *textureTable = nullptr;

    // The five engine blobs. 'specializedRaygenShader' is the RGenPrimary module with its SpecId 0
    // set to the game's primaryRaysMaxAlbedoLayers; it is what the pipeline uses, and it keeps the
    // base module alive through its own reference.
    nvrhi::ShaderHandle raygenShader;
    nvrhi::ShaderHandle specializedRaygenShader;
    nvrhi::ShaderHandle missShader;
    nvrhi::ShaderHandle shadowMissShader;
    nvrhi::ShaderHandle closestHitShader;
    nvrhi::ShaderHandle anyHitShader;

    // The twelve set layouts, in the engine's RT order (the empty ones are one shared zero-item
    // layout bound at 5, 6, 9 and 10).
    nvrhi::BindingLayoutHandle tlasLayout;
    nvrhi::BindingLayoutHandle framebufferLayout;
    nvrhi::BindingLayoutHandle uniformLayout;
    nvrhi::BindingLayoutHandle vertexDataLayout;
    nvrhi::BindingLayoutHandle holeLayout;
    nvrhi::BindingLayoutHandle cubemapLayout;
    nvrhi::BindingLayoutHandle renderCubemapLayout;
    nvrhi::BindingLayoutHandle rayStatsLayout;

    // The sets that do not depend on the frame: the four holes (one real empty set bound four
    // times), the cubemap placeholder table, the set-8 set (the render-cubemap placeholder or the
    // real cube once SetRenderCubemap delivered it), and the ray-stats stand-in the raygen's
    // `rayStatsAdd` writes.
    nvrhi::BindingSetHandle holeSet;
    nvrhi::DescriptorTableHandle cubemapTable;
    nvrhi::TextureHandle dummyCubemapTexture;
    nvrhi::SamplerHandle dummyCubemapSampler;

    // Set 8's coordinator keys: the raw handles keep the real cube and its sampler alive (the
    // coordinator's SetRenderCubemap stores them), the raw pointers are the coordinator keys the
    // current set was built over, so a replaced key rebuilds it in PrepareRenderCubemapSet. A null
    // pointer means the placeholder pair is the current set (what the module starts with).
    nvrhi::TextureHandle renderCubemapTexture;
    nvrhi::ITexture *renderCubemapSetTexture = nullptr;
    nvrhi::SamplerHandle renderCubemapSampler;
    nvrhi::ISampler *renderCubemapSetSampler = nullptr;
    nvrhi::BindingSetHandle renderCubemapSet;

    nvrhi::BufferHandle rayStatsBuffer;
    nvrhi::BindingSetHandle rayStatsSet;

    // The pipeline and its table: one raygen (RGenPrimary), the two engine misses, the two engine
    // hit groups. Both are created once; the table is uncached, so the backend bakes it per list.
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
    bool warnedBadRenderCubemap = false;
    bool warnedRenderCubemapSampler = false;

    bool created = false;
};

}
