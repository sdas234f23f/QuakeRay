// Copyright (c) 2020-2021 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <vkpt/vkpt.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Common.h"

#include "CommandBufferManager.h"
#include "PhysicalDevice.h"
#include "Scene.h"
#include "Swapchain.h"
#include "Queues.h"
#include "GlobalUniform.h"
#include "PathTracer.h"
#include "Rasterizer.h"
#include "Framebuffers.h"
#include "MemoryAllocator.h"
#include "TextureManager.h"
#include "BlueNoise.h"
#include "ImageComposition.h"
#include "Tonemapping.h"
#include "CubemapManager.h"
#include "Q2Denoiser.h"
#include "UserFunction.h"
#include "Bloom.h"
#include "Sharpening.h"
#include "DLSS.h"
#include "RenderResolutionHelper.h"
#include "DecalManager.h"
#include "EffectWipe.h"
#include "EffectSimple_Instances.h"
#include "FSR.h"
#include "FrameState.h"
#include "LibraryConfig.h"
#include "PortalList.h"
#include "Volumetric.h"
#include "ShadowMap.h"
#include "WorldLights.h"
#include "ClusterLightLists.h"
#include "GodRays.h"
#include "RayStats.h"
#include "PassTimings.h"

namespace vkpt
{

class NvrhiContext;
class NvrhiFrameSkeleton;
class RhiDebugTracePass;
class RhiRtComposePass;
class RhiRtDirectPass;
class RhiRtIndirectPass;
class RhiRtPrimaryPass;
struct NvrhiRequirements;

namespace rhi
{
class RhiAccelStructs;
class RhiFrameContext;
class RhiTextureTable;
}

class VulkanDevice
{
public:
    explicit VulkanDevice(const RgInstanceCreateInfo *pInfo);
    ~VulkanDevice();


    VulkanDevice(const VulkanDevice& other) = delete;
    VulkanDevice(VulkanDevice&& other) noexcept = delete;
    VulkanDevice& operator=(const VulkanDevice& other) = delete;
    VulkanDevice& operator=(VulkanDevice&& other) noexcept = delete;


    void UploadGeometry(const RgGeometryUploadInfo *pUploadInfo);
    void UpdateGeometryTransform(const RgUpdateTransformInfo *pUpdateInfo);
    void UpdateGeometryTexCoords(const RgUpdateTexCoordsInfo *pUpdateInfo);

    void UploadRasterizedGeometry(const RgRasterizedGeometryUploadInfo *pUploadInfo,
                                  const float *pViewProjection, const RgViewport *pViewport);
    void UploadLensFlare(const RgLensFlareUploadInfo *pUploadInfo);
    void UploadDecal(const RgDecalUploadInfo *pUploadInfo);
    void UploadPortal(const RgPortalUploadInfo *pUploadInfo);

    void SubmitStaticGeometries();
    void StartNewStaticScene();

    void UploadDirectionalLight(const RgDirectionalLightUploadInfo *pLightInfo);
    void UploadSphericalLight(const RgSphericalLightUploadInfo *pLightInfo);
    void UploadSpotlight(const RgSpotLightUploadInfo *pLightInfo);
    void UploadPolygonalLight(const RgPolygonalLightUploadInfo *pLightInfo);
    void UploadTexturedAreaLight(const RgTexturedAreaLightUploadInfo *pLightInfo);

    void UploadTexturedAreaLights(const RgTexturedAreaLightUploadInfo *pLightInfos, uint32_t count);

    void UploadClusterLightSources(const RgClusterLightSourcesUploadInfo *pInfo);

    void GetClusterLightStats(RgClusterLightStats *pStats);
    void GetClusterLightGrants(uint32_t *pGranted, uint32_t *pDenied, uint32_t maxCount, uint32_t *pCount);
    void GetClusterLightList(uint32_t cluster, uint64_t *pLightUniqueIds, uint32_t maxCount, uint32_t *pCount);

    void UploadWorldLights(const RgWorldLightsUploadInfo *pInfo);

    // Q2RTX-style fog volumes (used by the new Q2RTX core path)
    void SetFogVolumes(uint32_t count, const RgFogVolume *pVolumes);

    void CreateMaterial(const RgMaterialCreateInfo *pCreateInfo, RgMaterial *pResult);
    void CreateAnimatedMaterial(const RgAnimatedMaterialCreateInfo *pCreateInfo, RgMaterial *pResult);
    void ChangeAnimatedMaterialFrame(RgMaterial animatedMaterial, uint32_t frameIndex);
    void UpdateMaterial(const RgMaterialUpdateInfo *pUpdateInfo);
    void DestroyMaterial(RgMaterial material);

    void CreateSkyboxCubemap(const RgCubemapCreateInfo *pCreateInfo, RgCubemap *pResult);
    void DestroyCubemap(RgCubemap cubemap);


    void StartFrame(const RgStartFrameInfo *pStartInfo);
    void DrawFrame(const RgDrawFrameInfo *pFrameInfo);


    bool IsSuspended() const;
    bool IsRenderUpscaleTechniqueAvailable(RgRenderUpscaleTechnique technique) const;

    void GetFrameStats(uint32_t *pRays, uint32_t *pFpsX10) const;

    void GetFrameStatsEx(RgFrameStats *pStats) const;


    void Print(const char *pMessage) const;

private:
    void CreateInstance(const RgInstanceCreateInfo &info);
    void CreateDevice();
    void CreateNvrhiDevice();
    void CreateSyncPrimitives();
    static VkSurfaceKHR GetSurfaceFromUser(VkInstance instance, const RgInstanceCreateInfo &info);
    void ValidateCreateInfo(const RgInstanceCreateInfo *pInfo);

    void DestroyInstance();
    void DestroyDevice();
    void DestroySyncPrimitives();

    void FillUniform(ShGlobalUniform *gu, const RgDrawFrameInfo &drawInfo) const;

    VkCommandBuffer BeginFrame(const RgStartFrameInfo &startInfo);
    void Render(VkCommandBuffer cmd, const RgDrawFrameInfo &drawInfo);
    // Draws the current frame through the RHI layer and submits it, together with
    // the command buffer of the frame. Returns false if the frame has to be drawn
    // by the renderer instead. 'drawInfo' is the same struct Render receives; only
    // the sky params of it are read (the sky viewer position).
    bool RenderThroughRhi(const RgDrawFrameInfo &drawInfo);
    void EndFrame(VkCommandBuffer cmd);

private:
    VkInstance          instance;
    VkDevice            device;
    VkSurfaceKHR        surface;

    FrameState          currentFrameState;

    // incremented every frame
    uint32_t            frameId;

    VkFence             frameFences[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore         imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore         renderFinishedSemaphores[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore         inFrameSemaphores[MAX_FRAMES_IN_FLIGHT] = {};

    bool                waitForOutOfFrameFence;
    VkFence             outOfFrameFences[MAX_FRAMES_IN_FLIGHT] = {};

    std::shared_ptr<PhysicalDevice>         physDevice;
    std::shared_ptr<Queues>                 queues;
    std::shared_ptr<Swapchain>              swapchain;

    std::shared_ptr<MemoryAllocator>        memAllocator;

    std::shared_ptr<CommandBufferManager>   cmdManager;

    std::shared_ptr<Framebuffers>           framebuffers;
    std::shared_ptr<Volumetric>             volumetric;

    std::shared_ptr<GlobalUniform>          uniform;
    std::shared_ptr<Scene>                  scene;

    std::shared_ptr<ShaderManager>          shaderManager;
    std::shared_ptr<RayTracingPipeline>     rtPipeline;
    std::shared_ptr<PathTracer>             pathTracer;
    std::shared_ptr<Rasterizer>             rasterizer;
    std::shared_ptr<DecalManager>           decalManager;
    std::shared_ptr<PortalList>             portalList;
    std::shared_ptr<Q2Denoiser>             q2Denoiser;
    std::shared_ptr<Tonemapping>            tonemapping;
    std::shared_ptr<ImageComposition>       imageComposition;
    std::shared_ptr<Bloom>                  bloom;
    std::shared_ptr<ShadowMap>              shadowMap;
    std::shared_ptr<GodRays>                godRays;
    std::shared_ptr<RayStats>               rayStats;
    std::shared_ptr<PassTimings>            passTimings;
    std::shared_ptr<FidelityFX::FSR>        amdFsr;
    std::shared_ptr<DLSS>                   nvDlss;
    std::shared_ptr<Sharpening>             sharpening;
    std::shared_ptr<EffectWipe>                 effectWipe;
    std::shared_ptr<EffectRadialBlur>           effectRadialBlur;
    std::shared_ptr<EffectChromaticAberration>  effectChromaticAberration;
    std::shared_ptr<EffectInverseBW>            effectInverseBW;
    std::shared_ptr<EffectHueShift>             effectHueShift;
    std::shared_ptr<EffectDistortedSides>       effectDistortedSides;
    std::shared_ptr<EffectWaves>                effectWaves;
    std::shared_ptr<EffectColorTint>            effectColorTint;
    std::shared_ptr<EffectCrtDemodulateEncode>  effectCrtDemodulateEncode;
    std::shared_ptr<EffectCrtDecode>            effectCrtDecode;

    std::shared_ptr<SamplerManager>         worldSamplerManager;
    std::shared_ptr<SamplerManager>         genericSamplerManager;
    std::shared_ptr<BlueNoise>              blueNoise;
    std::shared_ptr<TextureManager>         textureManager;
    std::shared_ptr<CubemapManager>         cubemapManager;

    // The RHI frame model (RHI/RhiFrameContext.h): one command list per engine frame slot plus the
    // retire queue that keeps a resource alive until the queue finished the submission that used it.
    // Created next to the NVRHI device, before the texture table (which retires through it) and the
    // frame skeleton; null if it could not be created.
    std::shared_ptr<rhi::RhiFrameContext>   rhiFrameContext;

    // The RHI copy of the engine's texture table (RHI/RhiTextureTable.h): created next to the NVRHI
    // device, shared with the sampler managers and the frame skeleton. Null if it could not be
    // created; the legacy renderer stays functional then.
    std::shared_ptr<rhi::RhiTextureTable>   rhiTextureTable;

    // World tables of the current map (clusters, PVS, emissive faces), built by the host.
    std::shared_ptr<WorldLights>            worldLights;
    // Per-cluster light lists, composed out of the registered sources and the tables above.
    std::shared_ptr<ClusterLightLists>      clusterLightLists;

    LibraryConfig::Config                   libconfig;
    VkDebugUtilsMessengerEXT                debugMessenger;
    std::unique_ptr<UserPrint>              userPrint;
    std::shared_ptr<UserFileLoad>           userFileLoad;

    // Names of the extensions the instance and the device were created with.
    // The RHI layer reads them to know which Vulkan features are available.
    std::vector<std::string>                enabledInstanceExtensions;
    std::vector<std::string>                enabledDeviceExtensions;

    // RHI device (NVIDIA NVRHI) created over the Vulkan device above.
    std::unique_ptr<NvrhiContext>           nvrhi;
    // The RHI acceleration structures (RHI/RhiAccelStructs.h): the NVRHI copy of the engine's
    // static BLAS plus one TLAS per frame slot, built from the engine's ASManager. Created next to
    // the skeleton and referenced by it; a null one makes the skeleton unavailable. Null when
    // 'rhiframe' is off.
    std::shared_ptr<rhi::RhiAccelStructs>   rhiAccelStructs;
    // The RHI debug ray-tracing pass (RHI/RhiDebugTracePass.h): the first traced image of the RHI
    // path, created next to the skeleton only when 'rhitrace' is on and referenced by it. Null when
    // the flag is off or the creation failed; with the flag on and no pass the skeleton stays
    // unavailable and the legacy renderer is kept.
    std::shared_ptr<RhiDebugTracePass>      rhiDebugTracePass;
    // The RHI primary-visibility ray-tracing pass (RHI/RhiRtPrimaryPass.h): the real traced G-buffer
    // of A4.1, created next to the skeleton only when 'rhirt' is on and referenced by it. Null when
    // the flag is off or the creation failed; with the flag on and no pass the skeleton stays
    // unavailable and the legacy renderer is kept.
    std::shared_ptr<RhiRtPrimaryPass>       rhiRtPrimaryPass;
    // The RHI direct-lighting ray-tracing pass (RHI/RhiRtDirectPass.h): the light term of the traced
    // chain (A4.2), created next to the primary pass - it borrows the primary's shared layout
    // handles, so it has to be destroyed before it - and referenced by the skeleton. Null when the
    // flag is off or the creation failed; with the flag on and no pair the skeleton stays
    // unavailable and the legacy renderer is kept.
    std::shared_ptr<RhiRtDirectPass>        rhiRtDirectPass;
    // The RHI indirect / GI pass (RHI/RhiRtIndirectPass.h): the bounce-light term of the traced
    // chain (A4.3), created next to the direct pass - it borrows the primary's layout handles and
    // the direct pass's light set, so both have to outlive it and be destroyed after it - and
    // referenced by the skeleton. Null when the flag is off or the creation failed; with the flag
    // on and no pass the skeleton stays unavailable and the legacy renderer is kept.
    std::shared_ptr<RhiRtIndirectPass>      rhiRtIndirectPass;
    // The RHI compose pass (RHI/RhiRtComposePass.h): the real adapter -> interleave -> exposure
    // histogram/average -> checkerboard -> prepare-final chain writing the display-referred FINAL
    // for the traced frame, created only when 'rhicompose' is on and referenced by the skeleton,
    // which then presents its FINAL image. Null when the flag is off or the creation failed; the
    // traced chain then keeps the A4.2a diagnostic present.
    std::shared_ptr<RhiRtComposePass>       rhiRtComposePass;
    // The RHI frame skeleton: the first frame pass that is recorded through the
    // RHI layer. Null unless 'rhiframe' is set in vkpt.txt.
    std::shared_ptr<NvrhiFrameSkeleton>     nvrhiFrameSkeleton;

    // Q2RTX-style fog volumes (host data, uploaded into the uniform each frame)
    std::array<RgFogVolume, RG_MAX_FOG_VOLUMES> fogVolumes{};
    uint32_t                                    fogVolumeCount = 0;

    bool                                    rayCullBackFacingTriangles;
    bool                                    allowGeometryWithSkyFlag;
    bool                                    lensFlareVerticesInScreenSpace;

    // The instance-wide applyVertexColorGamma of the rasterized geometry (RgInstanceCreateInfo), the
    // vertex spec constant the RHI sky pipelines bake into their key (RhiSkyPass::Render).
    bool                                    rasterizedVertexColorGamma;

    RenderResolutionHelper                  renderResolution;

    // Last upscale technique requested by the application; used to detect
    // actual FSR version switches (FSR 2 <-> FSR 3.1) without polling each frame.
    std::optional<RgRenderUpscaleTechnique> lastUpscaleTechnique;

    double                                  previousFrameTime;
    double                                  currentFrameTime;

    uint32_t                                statsRays = 0;
    uint32_t                                statsRaysPerCategory[RAY_STATS_CATEGORY_COUNT] = {};
    uint32_t                                statsFpsX10 = 0;
    float                                   statsSmoothedFps = 0.0f;
};

}
