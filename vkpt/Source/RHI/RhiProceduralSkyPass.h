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

#include <nvrhi/nvrhi.h>

#include "../Common.h"

namespace vkpt
{

namespace rhi
{
class RhiFrameContext;
}

// The RHI module of the default sky's cube content: the legacy `RenderCubemap`'s procedural half -
// `CmProceduralSky.comp` dispatched over two cube images plus their mip chains - exposing the four
// set-8 items the traced sky samples. The legacy reference is `RenderCubemap::DrawProcedural`
// (RenderCubemap.cpp:890-950), the images it writes (RenderCubemap.cpp:34-37, :482-591) and the host
// block that fills its params (VulkanDevice.cpp:756-862).
//
// Why it exists: the default traced frame is `r_fastsky 0` + `rt_physical_sky 1`, i.e.
// SKY_TYPE_PROCEDURAL (gl_vidsdl.c:1908-1910), whose cube content the legacy produces only inside
// the legacy `Render` that `rhiframe` skips (a5a_sky_recon.md §0.2). Until this module the three RT
// passes that declare set 8 bind 1x1 placeholders there (RhiRtPrimaryPass.cpp:435-457,
// RhiRtIndirectPass.h:115-121). The coordinator records one Render per traced frame, before the
// primary pass; the three consumers take `GetCubemapTexture()` / `GetEnvironmentTexture()` /
// `GetCubemapSampler()` as their set 8 (stream S2).
//
// The images (the engine's `cubemap` and `envCubemap`, RenderCubemap.cpp:34-37, :489-503):
//  - RGBA16F (`VK_FORMAT_R16G16B16A16_SFLOAT`), 1024x1024 (`CUBEMAP_SIDE_SIZE`), 6 array layers,
//    11 mips (`1 + log2(1024)`), cube-compatible. `envCubemap` is the same compute result without
//    the sun disc: the shader writes the disc-less `evaluateSky` result there and the disc-carrying
//    one into `cubemap` (CmProceduralSky.comp.hlsl:285-294), and only the visible cube is sampled
//    at lod 0 (`getSkyPrimary`) while the env cube is the PROCEDURAL target of `getSkyFiltered`
//    (RaygenCommon.hlsli:365-367, :403-406; SKY_MIP_COUNT 11.0 at :390).
//  - The engine keeps two VkImageViews over each image: a cube view for sampling and a 2D-array
//    view for the storage-image writes, because HLSL has no writable cube texture and D3D12 has no
//    cube UAV (`TODO(refactor)` shim, RenderCubemap.cpp:545-557, CmProceduralSky.comp.hlsl:44-54).
//    NVRHI needs no second texture and no native wrap for that: this module creates ONE
//    `TextureDimension::TextureCube` texture per image (the desc the sampling consumers see) and
//    lets the backend build the write view from the binding - each `Texture_UAV` item names
//    `TextureDimension::Texture2DArray` and the mip it writes, so NVRHI creates an e2DArray
//    `VkImageView` over the same `VkImage` exactly where the engine's `viewArray` sits. The
//    validation device explicitly allows the cube -> 2D-array view (textureDimensionsCompatible,
//    validation-device.cpp:1512-1524), and the pinned NVRHI is enough: nothing has to be added to
//    RhiTextureSource for this module.
//  - No depth image. The engine's `D16_UNORM` depth serves the raster multiview `Draw` only
//    (RenderCubemap.cpp:499-502, :602-606); `DrawProcedural` never binds it, and this module's
//    compute path does not either. A future raster `DrawSkyToCubemap` port is the caller that would
//    need one (see the seam note below), and it can add its own.
//
// The compute blob, exactly (measured 2026-09-26 with spirv-dis over
// vkpt\Build\CmProceduralSky.comp.spv; source CmProceduralSky.comp.hlsl:69-83):
//   [numthreads(16, 16, 1)] (OpExecutionMode LocalSize 16 16 1);
//   set 0, binding 0: `cubemapOut`      RWTexture2DArray<float4>, image format rgba16f;
//   set 0, binding 1: `params`          ConstantBuffer<Params_BT>, Block, members at
//                                       0/288/304/320/336/352/368 (the CPU mirror below);
//   set 0, binding 2: `envCubemapOut`   RWTexture2DArray<float4>, image format rgba16f;
//   no push constants, no other set, no sampler. The module's one binding layout lays that set out
//   with the UAV and constant-buffer register offsets at 0, so an item's slot equals its raw
//   binding (`Texture_UAV(0)`, `ConstantBuffer(1)`, `Texture_UAV(2)`).
// The shader derives its own resolution from `GetDimensions` of the bound storage image
// (CmProceduralSky.comp.hlsl:262-267), bounds-checks the invocation, builds the per-texel direction
// from the params' per-face bases and writes both images at that invocation's (x, y, face).
//
// The dispatch is the legacy's (`RenderCubemap.cpp:944-946`): the z extent is the face count 6 and
// the x/y extents are `Utils::GetWorkGroupCount(1024, 16)` = `1 + ceil(1024 / 16)` = 65
// (Utils.cpp:319-328) - one workgroup beyond the ceiling, whose invocations the shader's own bounds
// check discards, exactly as in the legacy.
//
// The mip chain: the legacy generates mips 1..10 with `vkCmdBlitImage` (VK_FILTER_LINEAR) from the
// previous level and barriers the whole image back to SHADER_READ_ONLY_OPTIMAL
// (RenderCubemap.cpp:233-330). The pinned NVRHI has no blit - `copyTexture` is a 1:1 copy and
// refuses to resize (nvrhi.h:3343-3349) - and this stage may not add a shader, so the module
// re-runs the same compute once per mip level, binding each image's UAV at `baseMipLevel = m`
// with one level for that dispatch. Mip 0 is byte-for-byte the legacy result; mip m is the analytic
// sky at mip m's texel centres rather than the legacy's linear box average of level m-1. That is a
// resampling difference confined to the mips (the visible cube is sampled at lod 0 under
// PROCEDURAL), not a structural one, and it costs about a third of the mip-0 dispatch on the frames
// that dispatch at all. What it does cost is state-tracker entries: each per-mip binding requires
// the six layers of its mip and the end-of-Render requirement covers all 66 subresources, but NVRHI
// batches every barrier into one `vkCmdPipelineBarrier2` per commit
// (vulkan-state-tracking.cpp:214-261), so the overhead is entries inside one call, not calls. A
// later increment that wants exact parity can add a mip-downsample blob or a native blit helper;
// the images do not change shape for it.
//
// The CPU early-out is the legacy's (`RenderCubemap.cpp:890-913`): Render works on a copy of the
// params, zeroes `cloudColor[3]` when `cloudParams[3] <= 0.5` (clouds off -> the animation time is
// frozen, so an unchanged sky keeps its cached cubemap), memcmps that copy against the params of
// the last recorded dispatch and returns without recording anything when they are equal. `clouds
// on` keeps the raw time, so the sky re-renders every frame, exactly like the legacy. `lastParams`
// starts zeroed, mirroring the legacy's zero-initialized mapped buffer (RenderCubemap.cpp:746-750):
// an all-zero param set on the very first call is treated as "already rendered", like the legacy.
//
// State discipline:
//  - Both textures are created with `initialState = NonPixelShaderResource` and
//    `keepInitialState = true`. NonPixelShaderResource is the state every consumer's set-8
//    `Texture_SRV` requires (the layouts' visibility is the RT one, not the pixel one - the same
//    choice the set-8 dummies document, RhiRtIndirectPass.cpp:446-451), and it lowers to
//    `SHADER_READ_ONLY_OPTIMAL`, which is exactly the layout the legacy leaves the images in between
//    frames. Every command list therefore starts both textures in that state and the close-time
//    `keepInitialState` requirement returns them there (state-tracking.cpp:350-361).
//  - Render announces nothing: the textures are module-owned, so NVRHI's own tracker is the truth.
//    The per-mip `Texture_UAV` bindings require UnorderedAccess for the six layers of the mip they
//    name, which the automatic barrier pass emits before each dispatch (and with the first-use
//    Common state on the very first list, an Undefined-sourced transition - harmless, because the
//    same Render writes every mip).
//  - Render ends by requiring NonPixelShaderResource for both whole textures. That mirrors the
//    legacy's final `GenerateMipmaps` barrier (RenderCubemap.cpp:324-329) and makes every later
//    SRV binding on the same list a no-op: the consumers are expected to bind these very handles
//    (there is no second wrap of the image), so the tracker sees one texture and no state can
//    conflict. Between submissions both images rest read-only.
//  - A consumer that samples the images on a list where this module did not run (Render skipped by
//    the early-out, or a colour sky) reads the previous content; on the very first list of the
//    module's life the read-only state is still uninitialized there and the transition discards the
//    never-written contents. The coordinator's contract is therefore: record one Render before the
//    consumers whenever the uniform's `skyType` is SKY_TYPE_PROCEDURAL - the legacy order
//    (VulkanDevice.cpp:756-762) and the first frame included.
//
// Exposure and lifetime: `GetCubemapTexture()` is set-8 binding 0 (`renderCubemap`),
// `GetEnvironmentTexture()` binding 1 (`renderCubemapEnv`) and `GetCubemapSampler()` both sampler
// bindings 2 and 3 - the legacy writes one sampler into both (RenderCubemap.cpp:696-733,
// Generated/ShaderCommonC.h:33-36). All three are module-owned and valid from a successful Create
// until the pass is destroyed; the pass never re-creates them (the cube size is the legacy's fixed
// 1024, not render-resolution-dependent), so a consumer may hold the pointers for the run and does
// not need a release hook. `GetCubemapSampler()` is a fresh RHI sampler with the legacy's
// LINEAR/REPEAT shape (mip linear, repeat on all axes, SamplerManager.cpp:34-46, :89,
// RenderCubemap.cpp:697): the engine's VkSampler objects cannot be wrapped by the pinned NVRHI
// (RhiTextureSource.h:201-215).
//
// Seam for the future raster sky (`SKY_TYPE_RASTERIZED_GEOMETRY`, a5a_sky_recon.md §5):
// `GetCubemapTexture()` is the image a later `DrawSkyToCubemap` port writes into - the desc already
// declares `isRenderTarget` (the engine's COLOR_ATTACHMENT usage and the cube-compatible flag, so
// the image would not have to be re-created or re-wrapped for a raster port, and the pass's accessor
// is the one such a port or the RT consumers would use). What is missing is the depth attachment
// and the six-face raster pipeline (NVRHI has no multiview at this pin, so six draws with per-face
// view-projections); neither is part of this module.
//
// Not owned and not needed: no engine image, no framebuffer registry, no uniform buffer. The shader
// reads only the params and writes the two images, so the module has nothing to `ReleaseTargets()`
// on a resize and owns everything for the whole run.
//
// The pass is a no-op until Create succeeded (Render checks `created` and the frame index only); it
// is not thread-safe, using the per-slot params buffer of the frameIndex it is given, the engine's
// single-threaded per-slot frame model (RhiFrameContext).
class RhiProceduralSkyPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    // The legacy image shape (RenderCubemap.cpp:34-37, :80): 1024x1024, 6 faces, 11 mips.
    static constexpr uint32_t CUBEMAP_SIZE = 1024;
    static constexpr uint32_t CUBEMAP_FACE_COUNT = 6;
    static constexpr uint32_t CUBEMAP_MIP_LEVELS = 11;

    // The CPU mirror of the shader's `Params_BT` (CmProceduralSky.comp.hlsl:69-78), which is also
    // the legacy `RenderCubemap::ProceduralSkyParams` layout (RenderCubemap.h:38-50): a std140
    // block of 24 float4. The coordinator fills it exactly as VulkanDevice.cpp:756-862 does (each
    // field's legacy source is on the member) and must zero-initialize it first, like the legacy's
    // `ProceduralSkyParams p = {}` - the early-out memcmps the whole struct, so an uninitialized
    // padding member would only cost a redundant dispatch, but the legacy never leaves one.
    struct Params
    {
        // Offset 0. The 6 faces * (right, up, forward) basis every invocation builds its ray from
        // (CmProceduralSky.comp.hlsl:270-275). Legacy: the per-face loop at VulkanDevice.cpp:839-860
        // (`Matrix::GetViewMatrix` with the faceAngles table; its column-major columns are the three
        // basis vectors, the 4th component unused).
        float faceBasis[18][4];

        // Offset 288. xyz = the unit direction TOWARD the sun; w = the amount of sun the sky shows
        // (the shader's `sunAmount`, CmProceduralSky.comp.hlsl:279), which drops the mie halo and
        // the sun's disc when 0. Legacy: VulkanDevice.cpp:787-806 - `-sunDir` of
        // `LightManager::GetLastDirectionalLight` (the light points from the sun, the sky wants the
        // direction to it), w = 1 when the light exists; without one, the normalized
        // (0.3, 0.5, 0.8) fallback and w = 0.
        float sunDirection[4];

        // Offset 304. xyz = the atmosphere tint; w = the physical sun angular radius (radians).
        // Legacy: xyz is the uniform's `skyColorDefault` (rt_sky_color, VulkanDevice.cpp:763-765,
        // :807) and w the light's angular radius (the 0.0047 default when there is no light). The
        // shipped blob reads only .xyz (the display disc uses `skyParams.w`), so w is carried for
        // the legacy layout.
        float skyTint[4];

        // Offset 320. x = the sky colour multiplier (the uniform's `skyColorMultiplier`);
        // y = the tint strength (the uniform's `skyColorSaturation`); z = the sun disc intensity
        // (6.0f); w = the display sun disc angular radius in radians (0.025f). Legacy:
        // VulkanDevice.cpp:808-821, including the comment on the disc intensity.
        float skyParams[4];

        // Offset 336. xyz = the cloud colour; w = the cloud animation time in seconds, which the
        // early-out freezes to 0 when clouds are off. Legacy: xyz from
        // `drawInfo.pSkyParams->skyCubemapRotationTransform.matrix[0][0..2]`, w from the uniform's
        // `time` (VulkanDevice.cpp:826-836; the cloud fields are packed into the otherwise-unused
        // rotation-transform field, gl_vidsdl.c:1925-1933).
        float cloudColor[4];

        // Offset 352. x = coverage, y = density, z = drift speed, w = enabled (0/1). Legacy: the
        // packed fields c[3..6] of `drawInfo.pSkyParams->skyCubemapRotationTransform.matrix`
        // (VulkanDevice.cpp:827-837); a value outside (0, 1) is how the host turns clouds off.
        float cloudParams[4];

        // Offset 368. xyz = the sun disc colour (rt_sun_color); w unused. Legacy:
        // `drawInfo.pSkyParams->sunDiscColor`, defaulting to white when the host sends no sky params
        // (VulkanDevice.cpp:772-778). Appended after every other field, like the legacy struct
        // (RenderCubemap.h:46-49), so a stale compiled shader still reads all older fields at the
        // same offsets.
        float sunDiscColor[4];
    };

    RhiProceduralSkyPass();
    ~RhiProceduralSkyPass();

    RhiProceduralSkyPass(const RhiProceduralSkyPass &other) = delete;
    RhiProceduralSkyPass(RhiProceduralSkyPass &&other) noexcept = delete;
    RhiProceduralSkyPass &operator=(const RhiProceduralSkyPass &other) = delete;
    RhiProceduralSkyPass &operator=(RhiProceduralSkyPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pFrameContext' is the host's frame model
    // (RHI/RhiFrameContext.h), which every sibling pass is created with and whose open list Render
    // records on; 'pShaderFolderPath' is the folder the engine blobs load from, with the trailing
    // separator ('CmProceduralSky.comp.spv' - the file ShaderManager.cpp:56 maps "CProceduralSky"
    // to - is read from it). None is owned; all have to outlive this object, and a null or unusable
    // one makes Create fail. The pass logs through 'pfnPrint'. Creates the two cube images, the
    // LINEAR/REPEAT sampler, the binding layout, the compute pipeline and the per-slot params
    // buffers with their per-mip binding sets. Returns false and leaves the pass unusable if a
    // resource cannot be created; the host logs that through 'pfnPrint'.
    bool Create(nvrhi::IDevice *pDevice,
                rhi::RhiFrameContext *pFrameContext,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // The three set-8 items the traced sky binds (Generated/ShaderCommonC.h:33-36):
    //  - GetCubemapTexture()     raw binding 0, `renderCubemap` in the shaders (the disc-carrying
    //                            visible cube, sampled at lod 0 by `getSkyPrimary`);
    //  - GetEnvironmentTexture() raw binding 1, `renderCubemapEnv` (the disc-less env cube,
    //                            sampled at the roughness lod by `getSkyFiltered` under
    //                            SKY_TYPE_PROCEDURAL);
    //  - GetCubemapSampler()     raw bindings 2 and 3 (the legacy writes the same LINEAR/REPEAT
    //                            sampler into both).
    // Module-owned and valid from a successful Create until destruction (see the class comment);
    // null before Create. The consumers bind these exact handles - there is no second wrap of the
    // images.
    nvrhi::ITexture *GetCubemapTexture() const { return cubemapTexture.Get(); }
    nvrhi::ITexture *GetEnvironmentTexture() const { return environmentTexture.Get(); }
    nvrhi::ISampler *GetCubemapSampler() const { return skySampler.Get(); }

    // One call per traced frame, on the frame context's open command list of 'frameIndex'
    // (RhiFrameContext::GetCommandList), when the uniform's `skyType` is SKY_TYPE_PROCEDURAL and
    // before the passes that sample the cubes (the skeleton's primary/indirect/reflrefr order).
    // Works on a copy of 'params', applies the legacy cloud freeze, compares it with the params of
    // the last recorded dispatch and returns without recording anything when they are equal
    // (RenderCubemap.cpp:890-913). Otherwise writes the slot's params buffer and records one
    // `CmProceduralSky` dispatch per mip level - the legacy dispatch at mip 0 (`(65, 65, 6)`) and
    // the module's per-mip equivalent above for mips 1..10 - then requires NonPixelShaderResource
    // for both textures (see the state discipline in the class comment). A no-op when the pass is
    // not created or the frame index is out of range.
    void Render(nvrhi::ICommandList *pCommandList,
                uint32_t frameIndex,
                const Params &params);

private:
    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    // Not owned: the host's frame model, which outlives this object.
    rhi::RhiFrameContext *frameContext = nullptr;

    // The one engine blob.
    nvrhi::ShaderHandle skyShader;

    // Set 0 of the blob: `Texture_UAV(0)`, `ConstantBuffer(1)`, `Texture_UAV(2)` with the UAV and
    // constant-buffer register offsets at 0.
    nvrhi::BindingLayoutHandle skyLayout;
    nvrhi::ComputePipelineHandle skyPipeline;

    // The two cube images (see the class comment). Owned for the whole run; never wrapped, never
    // re-created.
    nvrhi::TextureHandle cubemapTexture;
    nvrhi::TextureHandle environmentTexture;

    // The LINEAR/REPEAT sampler of set-8 bindings 2/3. Owned for the whole run.
    nvrhi::SamplerHandle skySampler;

    // The per-slot `Params_BT` constant buffer (one per engine frame slot, so a slot's write cannot
    // race the submission still reading it) and the binding sets over it. One set per (slot, mip):
    // each set carries the two UAV items at that mip's single-level subresource range and the slot's
    // params buffer, because the blob declares all three items in one descriptor set.
    nvrhi::BufferHandle paramsBuffers[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BindingSetHandle skySets[MAX_FRAMES_IN_FLIGHT][CUBEMAP_MIP_LEVELS];

    // The params of the last recorded dispatch, compared with memcmp exactly like the legacy's
    // mapped buffer (RenderCubemap.cpp:899-913). Zero-initialized, like the legacy buffer.
    Params lastParams = {};

    bool created = false;
};

// The CPU mirror has to match the measured member layout of the blob's `Params_BT` member by member
// (offsets from the 2026-09-26 spirv-dis of vkpt\Build\CmProceduralSky.comp.spv): member 0 at 0
// with ArrayStride 16, then 288, 304, 320, 336, 352, 368 - and the block is 24 float4, 384 bytes.
static_assert(sizeof(RhiProceduralSkyPass::Params) == 384,
              "Params_BT is 24 float4; the constant buffer is created with sizeof(Params)");
static_assert(offsetof(RhiProceduralSkyPass::Params, faceBasis) == 0, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, sunDirection) == 288, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, skyTint) == 304, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, skyParams) == 320, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, cloudColor) == 336, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, cloudParams) == 352, "measured member offset");
static_assert(offsetof(RhiProceduralSkyPass::Params, sunDiscColor) == 368, "measured member offset");

}
