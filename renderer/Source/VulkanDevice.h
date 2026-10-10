// Copyright (c) 2026 f1ames0ff <f1am3sdev.github@protonmail.com>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#pragma once

#include <qray/qray.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
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
#include "RasterizedDataCollector.h"
#include "Framebuffers.h"
#include "MemoryAllocator.h"
#include "TextureManager.h"
#include "BlueNoise.h"
#include "Tonemapping.h"
#include "CubemapManager.h"
#include "UserFunction.h"
#include "DLSS.h"
#include "RenderResolutionHelper.h"
#include "DecalManager.h"
#include "FSR.h"
#include "FrameState.h"
#include "LibraryConfig.h"
#include "PortalList.h"
#include "WorldLights.h"
#include "ClusterLightLists.h"
#include "RayStats.h"
#include "CpuFrameProfiler.h"

namespace qray
{
class NvrhiContext;
class NvrhiFrameSkeleton;
class RhiBloomPass;
class RhiDecalPass;
class RhiFsrPass;
class RhiPostEffectPass;
class RhiProceduralSkyPass;
class RhiCloudsPass;
class RhiRasterOverlayPass;
class RhiRasterSkyPass;
class RhiRtComposePass;
class RhiRtDirectPass;
class RhiRtGodRaysPass;
class RhiRtIndirectPass;
class RhiRtPrimaryPass;
class RhiRtReflRefrPass;
class RhiShadowMapPass;
class RhiUiPass;
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
    explicit VulkanDevice(const QrInstanceCreateInfo *pInfo);
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice& other) = delete;
    VulkanDevice(VulkanDevice&& other) noexcept = delete;
    VulkanDevice& operator=(const VulkanDevice& other) = delete;
    VulkanDevice& operator=(VulkanDevice&& other) noexcept = delete;

    void UploadGeometry(const QrGeometryUploadInfo *pUploadInfo);
    void UpdateGeometryTransform(const QrUpdateTransformInfo *pUpdateInfo);
    void UpdateGeometryTexCoords(const QrUpdateTexCoordsInfo *pUpdateInfo);

    void UploadRasterizedGeometry(const QrRasterizedGeometryUploadInfo *pUploadInfo,
                                  const float *pViewProjection, const QrViewport *pViewport);
    QrResult UploadParticles(const QrParticleUploadInfo *pUploadInfo);
    void UploadDecal(const QrDecalUploadInfo *pUploadInfo);
    void UploadPortal(const QrPortalUploadInfo *pUploadInfo);

    void SubmitStaticGeometries();
    void StartNewStaticScene();

    void UploadDirectionalLight(const QrDirectionalLightUploadInfo *pLightInfo);
    void UploadSphericalLight(const QrSphericalLightUploadInfo *pLightInfo);
    void BeginDeferredLightUploads(uint32_t slot);
    void EndDeferredLightUploads();
    void FlushDeferredLightUploads();
    void SetParticleProxyGate(uint32_t gate, uint32_t glassParticles);
    void UploadSpotlight(const QrSpotLightUploadInfo *pLightInfo);
    void UploadPolygonalLight(const QrPolygonalLightUploadInfo *pLightInfo);
    void UploadTexturedAreaLight(const QrTexturedAreaLightUploadInfo *pLightInfo);

    void UploadTexturedAreaLights(const QrTexturedAreaLightUploadInfo *pLightInfos, uint32_t count);

    void UploadDtalGroups(const QrDtalGroupUploadBatch *pUploadInfo);

    void UploadClusterLightSources(const QrClusterLightSourcesUploadInfo *pInfo);
    void BeginClusterLightSources(const QrClusterLightSourcesUploadInfo *pInfo, uint32_t sliceCount);
    void RunClusterLightSourceSlice(uint32_t slice, uint32_t sliceCount);
    void FinishClusterLightSources();
    void RunClusterListPublishSlice(uint32_t slice, uint32_t sliceCount);
    void CommitClusterListPublication();

    void GetClusterLightStats(QrClusterLightStats *pStats);
    void GetClusterLightGrants(uint32_t *pGranted, uint32_t *pDenied, uint32_t maxCount, uint32_t *pCount);
    void GetClusterLightList(uint32_t cluster, uint64_t *pLightUniqueIds, uint32_t maxCount, uint32_t *pCount);
    void GetClusterLightTail(uint32_t cluster, uint64_t *pLightUniqueIds, float *pProb, float *pMarginal,
                             uint32_t *pAlias, float *pBeta, uint32_t maxCount, uint32_t *pCount);

    void UploadWorldLights(const QrWorldLightsUploadInfo *pInfo);

    void SetFogVolumes(uint32_t count, const QrFogVolume *pVolumes);

    void CreateMaterial(const QrMaterialCreateInfo *pCreateInfo, QrMaterial *pResult);
    void CreateAnimatedMaterial(const QrAnimatedMaterialCreateInfo *pCreateInfo, QrMaterial *pResult);
    void ChangeAnimatedMaterialFrame(QrMaterial animatedMaterial, uint32_t frameIndex);
    void UpdateMaterial(const QrMaterialUpdateInfo *pUpdateInfo);
    bool CanUpdateMaterialContents(QrMaterial material, QrExtent2D size) const;
    void DestroyMaterial(QrMaterial material);

    void CreateSkyboxCubemap(const QrCubemapCreateInfo *pCreateInfo, QrCubemap *pResult);
    void DestroyCubemap(QrCubemap cubemap);

    void StartFrame(const QrStartFrameInfo *pStartInfo);
    void DrawFrame(const QrDrawFrameInfo *pFrameInfo);

    bool IsSuspended() const;
    bool IsSurfaceUnavailable() const;
    bool IsRenderUpscaleTechniqueAvailable(QrRenderUpscaleTechnique technique) const;

    void GetFrameStats(uint32_t *pRays, uint32_t *pFpsX10) const;

    void GetFrameStatsEx(QrFrameStats *pStats) const;

    void GetAdapterInfo(QrAdapterInfo *pInfo) const;

    void RequestScreenshot(const char *pFilePath);

    void Print(const char *pMessage) const;

private:
    void CreateInstance(const QrInstanceCreateInfo &info);
    void CreateDevice();
    void CreateNvrhiDevice();
    void CreateSyncPrimitives();
    static VkSurfaceKHR GetSurfaceFromUser(VkInstance instance, const QrInstanceCreateInfo &info);
    void ValidateCreateInfo(const QrInstanceCreateInfo *pInfo);

    void DestroyInstance();
    void DestroyDevice();
    void DestroySyncPrimitives();

    void FillUniform(ShGlobalUniform *gu, const QrDrawFrameInfo &drawInfo) const;

    VkCommandBuffer BeginFrame(const QrStartFrameInfo &startInfo);
    bool RenderThroughRhi(const QrDrawFrameInfo &drawInfo);
    void EndFrame(VkCommandBuffer cmd);

private:
    VkInstance          instance;
    VkDevice            device;
    VkSurfaceKHR        surface;

    FrameState          currentFrameState;

    uint32_t            frameId;

    VkFence             frameFences[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore         imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT] = {};
    VkSemaphore         inFrameSemaphores[MAX_FRAMES_IN_FLIGHT] = {};

    bool                waitForOutOfFrameFence;
    VkFence             outOfFrameFences[MAX_FRAMES_IN_FLIGHT] = {};

    std::shared_ptr<PhysicalDevice>         physDevice;
    std::shared_ptr<Queues>                 queues;
    std::shared_ptr<Swapchain>              swapchain;
    bool                                    presentWait2Enabled = false;
    bool                                    swapchainMaintenance1Enabled = false;
    std::string                             printedPresentModeName;
    bool                                    printedPresentWaitActive = false;
    bool                                    printedRasterUploadWithoutFrame = false;
    bool                                    printedRasterOverflow = false;
    bool                                    printedParticlePointOverflow = false;
    uint32_t                                lastParticlePointOverflowWarnFrameId = 0;
    std::string                             pendingScreenshotPath;

    std::shared_ptr<MemoryAllocator>        memAllocator;

    std::shared_ptr<CommandBufferManager>   cmdManager;

    std::shared_ptr<Framebuffers>           framebuffers;

    std::shared_ptr<GlobalUniform>          uniform;
    std::shared_ptr<Scene>                  scene;

    std::shared_ptr<ShaderManager>          shaderManager;
    std::shared_ptr<RasterizedDataCollector> rasterizedDataCollector;
    bool particleProxyGateOpen = true;
    std::shared_ptr<DecalManager>           decalManager;
    std::shared_ptr<PortalList>             portalList;
    std::shared_ptr<Tonemapping>            tonemapping;
    std::shared_ptr<RayStats>               rayStats;
    std::shared_ptr<FidelityFX::FSR>        amdFsr;
    std::shared_ptr<DLSS>                   nvDlss;

    std::shared_ptr<SamplerManager>         worldSamplerManager;
    std::shared_ptr<SamplerManager>         genericSamplerManager;
    std::shared_ptr<BlueNoise>              blueNoise;
    std::shared_ptr<TextureManager>         textureManager;
    std::shared_ptr<CubemapManager>         cubemapManager;

    std::shared_ptr<rhi::RhiFrameContext>   rhiFrameContext;

    std::shared_ptr<rhi::RhiTextureTable>   rhiTextureTable;

    std::shared_ptr<WorldLights>            worldLights;
    std::shared_ptr<ClusterLightLists>      clusterLightLists;

    LibraryConfig::Config                   libconfig;
    VkDebugUtilsMessengerEXT                debugMessenger;
    std::unique_ptr<UserPrint>              userPrint;
    std::shared_ptr<UserFileLoad>           userFileLoad;

    std::vector<std::string>                enabledInstanceExtensions;
    std::vector<std::string>                enabledDeviceExtensions;

    std::unique_ptr<NvrhiContext>           nvrhi;
    std::shared_ptr<rhi::RhiAccelStructs>   rhiAccelStructs;
    std::shared_ptr<RhiRtPrimaryPass>       rhiRtPrimaryPass;
    std::shared_ptr<RhiRtDirectPass>        rhiRtDirectPass;
    std::shared_ptr<RhiRtIndirectPass>      rhiRtIndirectPass;
    std::shared_ptr<RhiRtComposePass>       rhiRtComposePass;
    std::shared_ptr<RhiBloomPass>           rhiBloomPass;
    std::shared_ptr<RhiShadowMapPass>       rhiShadowMapPass;
    std::shared_ptr<RhiRtGodRaysPass>       rhiRtGodRaysPass;

    std::shared_ptr<RhiRtReflRefrPass>      rhiRtReflRefrPass;

    std::shared_ptr<RhiProceduralSkyPass>   rhiProceduralSkyPass;
    std::shared_ptr<RhiCloudsPass>          rhiCloudsPass;

    std::shared_ptr<RhiRasterSkyPass>       rhiRasterSkyPass;

    std::shared_ptr<RhiRasterOverlayPass>   rhiRasterOverlayPass;

    std::vector<RasterizedDataCollector::DrawInfo> smokeDraws;
    std::vector<RasterizedDataCollector::DrawInfo> particleDraws;

    std::shared_ptr<RhiDecalPass>           rhiDecalPass;

    std::shared_ptr<RhiFsrPass>             rhiFsrPass;

    std::shared_ptr<RhiPostEffectPass>      rhiPostEffectPass;

    std::shared_ptr<RhiUiPass>              rhiUiPass;
    std::shared_ptr<NvrhiFrameSkeleton>     nvrhiFrameSkeleton;

    bool                                    warnedSkeletonRefusedFrame = false;

    std::array<QrFogVolume, QR_MAX_FOG_VOLUMES> fogVolumes{};
    uint32_t                                    fogVolumeCount = 0;

    bool                                    rayCullBackFacingTriangles;
    bool                                    allowGeometryWithSkyFlag;

    bool                                    rasterizedVertexColorGamma;

    RenderResolutionHelper                  renderResolution;

    std::optional<QrRenderUpscaleTechnique> lastUpscaleTechnique;

    double                                  previousFrameTime;
    double                                  currentFrameTime;

    uint32_t                                statsRays = 0;
    uint32_t                                statsRaysPerCategory[RAY_STATS_CATEGORY_COUNT] = {};
    uint32_t                                statsFpsX10 = 0;
    float                                   statsSmoothedFps = 0.0f;
    bool                                    statsGpuTimingValid = false;
    float                                   statsGpuFrameMs = 0.0f;
    float                                   statsGpuPassMs[QR_GPU_PASS_COUNT] = {};
    std::mutex                              geometryUploadMutex;
    CpuFrameProfiler                        cpuFrameProfiler;
    bool                                    statsCpuTimingValid = false;
    bool                                    statsRenderedUiOnly = false;
    std::atomic<uint32_t>                   statsApiCallsGeometry{0};
    std::atomic<uint32_t>                   statsApiCallsRasterized{0};
    std::atomic<uint32_t>                   statsApiCallsLights{0};
    uint64_t                                statsRasterUploadBytes = 0;
    uint32_t                                statsRasterUploadDroppedBatches = 0;
};
}
