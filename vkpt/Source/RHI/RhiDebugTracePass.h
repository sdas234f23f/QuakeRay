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

namespace rhi
{
class RhiFrameContext;
}

// The debug ray-tracing pass: the `RhiDebugTrace.rgen` / `.rmiss` / `.rchit` trio, dispatched
// through the RHI layer over the top-level acceleration structure stream 1 built, with the traced
// result written into the engine's ALBEDO storage image. It is the first pass on the RHI path that
// owns an `rt::IPipeline` and an `rt::IShaderTable` and the first caller of the pinned NVRHI's
// ray-tracing dispatch API (the .cpp documents the measured contracts of that API).
//
// The shaders and their bindings are the frozen debug interface, the mapping every blob of the trio
// declares:
//   set 0 binding 0 - the top-level acceleration structure (`RayTracingAccelStruct`),
//   set 1 binding 0 - the engine's global uniform (`ConstantBuffer`, raw binding 0),
//   set 2 binding 0 - the ALBEDO storage image (`Texture_UAV`, raw binding 0).
// One NVRHI layout per set, added to the pipeline in that order, so the pinned backend's legacy
// binding mode places them at descriptor sets 0, 1 and 2 (vulkan-resource-bindings.cpp:1090-1099).
// The three sets are dense, so the layout vector stays well under NVRHI's eight-set cap
// (c_MaxBindingLayouts, nvrhi.h:77).
//
// What the pass needs from the host:
//  - the RHI device, the frame context and the wrapped global uniform once, in Create(); the pass
//    loads the three blobs itself from the shader folder, the way RhiSkyPass does, and assigns
//    them the distinct export names the Vulkan backend's group map requires;
//  - one Render() per frame on the slot's open command list, with stream 1's top-level AS of that
//    frame slot and the slot's ALBEDO handles, exactly what
//    Framebuffers::GetImageHandles(FB_IMAGE_INDEX_ALBEDO, frameIndex) returns.
//
// What Render records, in order (the same shape RhiSkyPass::Prepare uses for a wrapped image):
//  1. the ALBEDO wrap is created on first use and re-created when the engine re-created the image
//     or the size changed; the replaced wrap and the binding set over it are retired through the
//     frame context's queue;
//  2. the ALBEDO image is announced as ResourceStates::UnorderedAccess on the current list before
//     its first use - the module's state contract, spelled out on Render below;
//  3. the TLAS, the uniform and the ALBEDO UAV are bound as sets 0, 1 and 2 and `dispatchRays` runs
//     over the whole render resolution (one ray per pixel, depth 1).
//
// Integration rule the caller has to keep: the announcement in step 2 assumes the image really is
// in VK_IMAGE_LAYOUT_GENERAL when the list starts. The pass must therefore be the first ALBEDO user
// of the list, or the caller must have left the image in GENERAL. If RhiSkyPass's framebuffer use
// already moved ALBEDO to COLOR_ATTACHMENT_OPTIMAL in the same list, the announcement would claim
// a layout the image is not in and the UAV write would be invalid - the other wrap of the same
// VkImage keeps its own tracker and cannot be seen from here.
//
// The pass is a no-op until Create succeeded and until Render receives a non-null TLAS and ALBEDO
// handles; it never creates a target on its own, and every early return is quiet after the first
// warning. It is not thread-safe: Render uses the per-slot target of the frameIndex it is given,
// which is the engine's single-threaded per-slot frame model (RhiFrameContext).
class RhiDebugTracePass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiDebugTracePass();
    ~RhiDebugTracePass();

    RhiDebugTracePass(const RhiDebugTracePass &other) = delete;
    RhiDebugTracePass(RhiDebugTracePass &&other) noexcept = delete;
    RhiDebugTracePass &operator=(const RhiDebugTracePass &other) = delete;
    RhiDebugTracePass &operator=(RhiDebugTracePass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's RHI frame context
    // (RHI/RhiFrameContext.h), which owns the retire queues every replaced wrap and binding set
    // goes through. Neither is owned, both have to outlive this object, and a null one makes Create
    // fail. 'pUniformBuffer' is the debug shaders' set 1 binding 0: the engine's global uniform
    // (ShGlobalUniform), wrapped by the host with IDevice::createHandleForNativeBuffer as a static
    // constant buffer - the same wrap NvrhiFrameSkeleton::PrepareWorld builds for the rasterized
    // world. A volatile buffer is rejected: BindingSetItem::ConstantBuffer turns it into a
    // dynamic-offset binding the static ConstantBuffer layout item cannot take. 'pShaderFolderPath'
    // is the folder ShaderManager loads the engine blobs from, with the trailing separator;
    // RhiDebugTrace.rgen.spv, .rmiss.spv and .rchit.spv are loaded from it. Returns false and
    // leaves the pass unusable if a shader, a layout, the pipeline or the shader table cannot be
    // created; the host logs that through 'pfnPrint' (the pass logs it itself).
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                nvrhi::IBuffer *pUniformBuffer,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // One call per frame, on the frame context's open command list for 'frameIndex' (the same list
    // the sky/world/present of that slot uses). 'pTopLevel' is stream 1's top-level acceleration
    // structure of that frame slot (RhiAccelStructs::GetTopLevel) - the set 0 binding is rebuilt
    // when the pointer changes, because the AS object can be replaced by a scene rebuild. 'width'
    // and 'height' are the render resolution ALBEDO is sized to. 'albedoImage', 'albedoView' and
    // 'albedoFormat' are the engine's ALBEDO handles as Framebuffers::GetImageHandles returns them
    // ('albedoView' is not consumed by the wrap - the pinned backend builds its own views - but the
    // frozen call shape keeps the parameter).
    //
    // The ALBEDO state contract, in the module's words: an engine framebuffer image rests in
    // VK_IMAGE_LAYOUT_GENERAL (= NVRHI's ResourceStates::UnorderedAccess), and a native wrap keeps
    // no state between command lists (keepInitialState is false, RhiTextureSource.h), so every list
    // that writes the image announces that state before its first use:
    //     commandList->beginTrackingTextureState(albedo, nvrhi::AllSubresources,
    //                                            nvrhi::ResourceStates::UnorderedAccess);
    // The Texture_UAV binding requires exactly that state (vulkan-state-tracking.cpp:61-63), so the
    // announcement turns the automatic barrier into a same-layout UAV barrier and the image stays
    // in GENERAL - no transition and nothing to move back after the dispatch.
    //
    // No-op when the pass is not created, the frame index is out of range, the size is zero, the
    // TLAS is null (the scene has no acceleration structures yet), or the ALBEDO handles are null.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                nvrhi::rt::IAccelStruct *pTopLevel,
                uint32_t width,
                uint32_t height,
                uint64_t albedoImage,
                uint64_t albedoView,
                VkFormat albedoFormat);

    // Drops every slot's ALBEDO wrap, the UAV binding set over it and the per-slot TLAS binding
    // set, and retires them through the frame context's queue. The caller has to call it before the
    // engine destroys its framebuffer images (the Framebuffers::PrepareForSize path, which waits
    // for the device first, Framebuffers.cpp:227-241) - otherwise the wraps reference destroyed
    // VkImages. The next Render re-reads the handles and re-wraps, so the pass survives a resize
    // without a second Create. The destructor drops directly, after a device idle.
    void ReleaseTargets();

private:
    // One entry per engine frame slot: ALBEDO is a swapped engine image, so the slot's image is not
    // shared with the other slot's (Framebuffers_BindingsSwapped), and the TLAS stream 1 hands over
    // is a per-slot object too.
    struct Target
    {
        // The engine image the slot currently wraps; a change (or a size change) means the engine
        // re-created it or the swap permutation flipped, so the wrap and the set over it have to
        // follow. The image is kept as the raw handle Render received, not as a VkImage, because
        // that is the form the caller passes.
        uint64_t albedoImage = 0;
        uint32_t width = 0;
        uint32_t height = 0;

        nvrhi::TextureHandle albedoTexture;
        nvrhi::BindingSetHandle albedoSet;

        // Set 0 of this slot. The pointer is only the cache key that tells whether the set still
        // addresses the current AS; the set itself holds the reference that keeps the AS alive.
        nvrhi::rt::IAccelStruct *topLevel = nullptr;
        nvrhi::BindingSetHandle tlasSet;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);

    void ReleaseTarget(Target &target);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // The host's frame model; not owned, it outlives this object and owns the retire queues.
    rhi::RhiFrameContext *frameContext = nullptr;

    nvrhi::ShaderHandle raygenShader;
    nvrhi::ShaderHandle missShader;
    nvrhi::ShaderHandle closestHitShader;

    // The three sets of the frozen mapping, in the pipeline's layout order: the TLAS (0), the
    // global uniform (1) and the ALBEDO storage image (2).
    nvrhi::BindingLayoutHandle tlasLayout;
    nvrhi::BindingLayoutHandle uniformLayout;
    nvrhi::BindingLayoutHandle albedoLayout;

    // Set 1, built once: the engine creates its global uniform once and only its contents change,
    // so this set never has to be rebuilt and no Retire contract applies to it.
    nvrhi::BindingSetHandle uniformSet;

    // The debug trio's pipeline and its table: one raygen, one miss and one hit group. Both are
    // created once; the table is baked per command list by the backend on the uncached path.
    nvrhi::rt::PipelineHandle pipeline;
    nvrhi::rt::ShaderTableHandle shaderTable;

    // One entry per engine frame slot (MAX_FRAMES_IN_FLIGHT, Common.h:31).
    Target targets[MAX_FRAMES_IN_FLIGHT];

    // One-shot warnings for the two inputs that can legitimately be missing for a few frames (the
    // AS before stream 1 built it, the ALBEDO handles before the engine framebuffers exist).
    bool warnedMissingTopLevel = false;
    bool warnedMissingAlbedo = false;

    bool created = false;
};

}
