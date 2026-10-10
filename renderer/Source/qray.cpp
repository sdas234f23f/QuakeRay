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

#include "VulkanDevice.h"
#include "QrException.h"

#include <atomic>

using namespace qray;

static_assert(sizeof(QrParticlePoint) == 24, "QrParticlePoint must stay 24 bytes");

constexpr uint32_t MAX_DEVICE_COUNT = 8;
static rgl::unordered_map<QrInstance, std::unique_ptr<VulkanDevice>> G_DEVICES;

static std::atomic<uint32_t> G_EntryPointCalls{ 0 };

static QrInstance GetNextID()
{
    return reinterpret_cast<QrInstance>(G_DEVICES.size() + 1024);
}

static VulkanDevice &GetDevice(QrInstance qrInstance)
{
    auto it = G_DEVICES.find(qrInstance);

    if (it == G_DEVICES.end())
    {
        throw QrException(QR_WRONG_INSTANCE);
    }

    return *(it->second);
}

static void TryPrintError(QrInstance qrInstance, const char *pMessage)
{
    auto it = G_DEVICES.find(qrInstance);

    if (it != G_DEVICES.end())
    {
        it->second->Print(pMessage);
    }
}

QrResult qrCreateInstance(const QrInstanceCreateInfo *pInfo, QrInstance *pResult)
{
    *pResult = nullptr;

    if (G_DEVICES.size() >= MAX_DEVICE_COUNT)
    {
        return QR_TOO_MANY_INSTANCES;
    }

    const QrInstance qrInstance = GetNextID();
    assert(G_DEVICES.find(qrInstance) == G_DEVICES.end());

    try
    {
        G_DEVICES[qrInstance] = std::make_unique<VulkanDevice>(pInfo);
        *pResult = qrInstance;
    }
    catch (QrException &e)
    {
        if (pInfo->pfnPrint != nullptr)
        {
            pInfo->pfnPrint(e.what(), pInfo->pUserPrintData);
        }

        return e.GetErrorCode();
    }
    return QR_SUCCESS;
}

QrResult qrDestroyInstance(QrInstance qrInstance)
{
    if (G_DEVICES.find(qrInstance) == G_DEVICES.end())
    {
        return QR_WRONG_INSTANCE;
    }

    try
    {
        G_DEVICES.erase(qrInstance);
    }
    catch (QrException &e)
    {
        TryPrintError(qrInstance, e.what());
        return e.GetErrorCode();
    }
    return QR_SUCCESS;
}

template<typename Func, typename... Args> requires (
    std::is_same_v< std::invoke_result_t<Func, VulkanDevice, Args...>, void>
    )
static auto Call(QrInstance qrInstance, Func f, Args&&... args)
{
    try
    {
        VulkanDevice &dev = GetDevice(qrInstance);

        G_EntryPointCalls.fetch_add(1, std::memory_order_relaxed);

        if (dev.IsSuspended())
        {
            return QR_SUCCESS;
        }

        (dev.*f)(std::forward<Args>(args)...);
    }
    catch (QrException &e)
    {
        TryPrintError(qrInstance, e.what());
        return e.GetErrorCode();
    }
    return QR_SUCCESS;
}

template<typename Func, typename... Args> requires (
    !std::is_same_v< std::invoke_result_t<Func, VulkanDevice, Args...>, void> &&
    std::is_default_constructible_v< std::invoke_result_t<Func, VulkanDevice, Args...> >
    )
static auto Call(QrInstance qrInstance, Func f, Args&&... args)
{
    using ReturnType = std::invoke_result_t<Func, VulkanDevice, Args...>;

    try
    {
        VulkanDevice &dev = GetDevice(qrInstance);

        G_EntryPointCalls.fetch_add(1, std::memory_order_relaxed);

        if (!dev.IsSuspended())
        {
            return (dev.*f)(std::forward<Args>(args)...);
        }
    }
    catch (QrException &e)
    {
        TryPrintError(qrInstance, e.what());
    }
    return ReturnType{};
}

QrResult qrUploadGeometry(QrInstance qrInstance, const QrGeometryUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadGeometry, pUploadInfo );
}

QrResult qrUpdateGeometryTransform(QrInstance qrInstance, const QrUpdateTransformInfo* pUpdateInfo)
{
    return Call(qrInstance, &VulkanDevice::UpdateGeometryTransform, pUpdateInfo);
}

QrResult qrUpdateGeometryTexCoords(QrInstance qrInstance, const QrUpdateTexCoordsInfo *pUpdateInfo)
{
    return Call(qrInstance, &VulkanDevice::UpdateGeometryTexCoords, pUpdateInfo);
}

QrResult qrUploadRasterizedGeometry(QrInstance qrInstance, const QrRasterizedGeometryUploadInfo *pUploadInfo,
                                    const float *pViewProjection, const QrViewport *pViewport)
{
    return Call(qrInstance, &VulkanDevice::UploadRasterizedGeometry, pUploadInfo, pViewProjection, pViewport);
}

QrResult qrUploadParticles(QrInstance qrInstance, const QrParticleUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadParticles, pUploadInfo);
}

QrResult qrUploadDecal(QrInstance qrInstance, const QrDecalUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadDecal, pUploadInfo);
}

QrResult qrUploadPortal(QrInstance qrInstance, const QrPortalUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadPortal, pUploadInfo);
}

QrResult qrBeginStaticGeometries(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::StartNewStaticScene);
}

QrResult qrSubmitStaticGeometries(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::SubmitStaticGeometries);
}

QrResult qrUploadDirectionalLight(QrInstance qrInstance, const QrDirectionalLightUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadDirectionalLight, pUploadInfo);
}

QrResult qrUploadSphericalLight(QrInstance qrInstance, const QrSphericalLightUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadSphericalLight, pUploadInfo);
}

QrResult qrUploadSpotLight(QrInstance qrInstance, const QrSpotLightUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadSpotlight, pUploadInfo);
}

QrResult qrBeginDeferredLightUploads(QrInstance qrInstance, uint32_t slot)
{
    return Call(qrInstance, &VulkanDevice::BeginDeferredLightUploads, slot);
}

QrResult qrEndDeferredLightUploads(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::EndDeferredLightUploads);
}

QrResult qrFlushDeferredLightUploads(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::FlushDeferredLightUploads);
}

QrResult qrSetParticleProxyGate(QrInstance qrInstance, uint32_t gate, uint32_t glassParticles)
{
    return Call(qrInstance, &VulkanDevice::SetParticleProxyGate, gate, glassParticles);
}

QrResult qrUploadPolygonalLight(QrInstance qrInstance, const QrPolygonalLightUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadPolygonalLight, pUploadInfo);
}

QrResult qrUploadTexturedAreaLight(QrInstance qrInstance, const QrTexturedAreaLightUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadTexturedAreaLight, pUploadInfo);
}

QrResult qrUploadTexturedAreaLights(QrInstance qrInstance, const QrTexturedAreaLightUploadInfo *pUploadInfos,
                                    uint32_t count)
{
    return Call(qrInstance, &VulkanDevice::UploadTexturedAreaLights, pUploadInfos, count);
}

QrResult qrUploadDtalGroups(QrInstance qrInstance, const QrDtalGroupUploadBatch *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadDtalGroups, pUploadInfo);
}

QrResult qrUploadClusterLightSources(QrInstance qrInstance, const QrClusterLightSourcesUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadClusterLightSources, pUploadInfo);
}

QrResult qrBeginClusterLightSources(QrInstance qrInstance, const QrClusterLightSourcesUploadInfo *pUploadInfo,
                                    uint32_t sliceCount)
{
    return Call(qrInstance, &VulkanDevice::BeginClusterLightSources, pUploadInfo, sliceCount);
}

QrResult qrRunClusterLightSourceSlice(QrInstance qrInstance, uint32_t slice, uint32_t sliceCount)
{
    return Call(qrInstance, &VulkanDevice::RunClusterLightSourceSlice, slice, sliceCount);
}

QrResult qrFinishClusterLightSources(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::FinishClusterLightSources);
}

QrResult qrRunClusterListPublishSlice(QrInstance qrInstance, uint32_t slice, uint32_t sliceCount)
{
    return Call(qrInstance, &VulkanDevice::RunClusterListPublishSlice, slice, sliceCount);
}

QrResult qrCommitClusterListPublication(QrInstance qrInstance)
{
    return Call(qrInstance, &VulkanDevice::CommitClusterListPublication);
}

QrResult qrGetClusterLightStats(QrInstance qrInstance, QrClusterLightStats *pStats)
{
    return Call(qrInstance, &VulkanDevice::GetClusterLightStats, pStats);
}

QrResult qrGetClusterLightGrants(QrInstance qrInstance, uint32_t *pGranted, uint32_t *pDenied,
                                 uint32_t maxCount, uint32_t *pCount)
{
    return Call(qrInstance, &VulkanDevice::GetClusterLightGrants, pGranted, pDenied, maxCount, pCount);
}

QrResult qrGetClusterLightList(QrInstance qrInstance, uint32_t cluster, uint64_t *pLightUniqueIds,
                               uint32_t maxCount, uint32_t *pCount)
{
    return Call(qrInstance, &VulkanDevice::GetClusterLightList, cluster, pLightUniqueIds, maxCount, pCount);
}

QrResult qrGetClusterLightTail(QrInstance qrInstance, uint32_t cluster, uint64_t *pLightUniqueIds,
                               float *pProb, float *pMarginal, uint32_t *pAlias, float *pBeta,
                               uint32_t maxCount, uint32_t *pCount)
{
    return Call(qrInstance, &VulkanDevice::GetClusterLightTail, cluster, pLightUniqueIds, pProb, pMarginal,
                pAlias, pBeta, maxCount, pCount);
}

QrResult qrUploadWorldLights(QrInstance qrInstance, const QrWorldLightsUploadInfo *pUploadInfo)
{
    return Call(qrInstance, &VulkanDevice::UploadWorldLights, pUploadInfo);
}

QrResult qrCreateMaterial(QrInstance qrInstance, const QrMaterialCreateInfo *pCreateInfo, QrMaterial *pResult)
{
    *pResult = QR_NO_MATERIAL;
    return Call(qrInstance, &VulkanDevice::CreateMaterial, pCreateInfo, pResult);
}

QrResult qrCreateAnimatedMaterial(QrInstance qrInstance, const QrAnimatedMaterialCreateInfo *pCreateInfo, QrMaterial *pResult)
{
    *pResult = QR_NO_MATERIAL;
    return Call(qrInstance, &VulkanDevice::CreateAnimatedMaterial, pCreateInfo, pResult);
}

QrResult qrChangeAnimatedMaterialFrame(QrInstance qrInstance, QrMaterial animatedMaterial, uint32_t frameIndex)
{
    return Call(qrInstance, &VulkanDevice::ChangeAnimatedMaterialFrame, animatedMaterial, frameIndex);
}

QrResult qrUpdateMaterialContents(QrInstance qrInstance, const QrMaterialUpdateInfo *pUpdateInfo)
{
    return Call(qrInstance, &VulkanDevice::UpdateMaterial, pUpdateInfo);
}

QrResult qrCanUpdateMaterialContents(QrInstance qrInstance, QrMaterial material, QrExtent2D size)
{
    if (material == QR_NO_MATERIAL)
    {
        return QR_WRONG_ARGUMENT;
    }

    return Call(qrInstance, &VulkanDevice::CanUpdateMaterialContents, material, size) ? QR_SUCCESS
                                                                                     : QR_CANT_UPDATE_MATERIAL;
}

QrResult qrDestroyMaterial(QrInstance qrInstance, QrMaterial material)
{
    return Call(qrInstance, &VulkanDevice::DestroyMaterial, material);
}

QrResult qrCreateCubemap(QrInstance qrInstance, const QrCubemapCreateInfo *pCreateInfo, QrCubemap *pResult)
{
    *pResult = QR_EMPTY_CUBEMAP;
    return Call(qrInstance, &VulkanDevice::CreateSkyboxCubemap, pCreateInfo, pResult);
}

QrResult qrDestroyCubemap(QrInstance qrInstance, QrCubemap cubemap)
{
    return Call(qrInstance, &VulkanDevice::DestroyCubemap, cubemap);
}

QrResult qrStartFrame(QrInstance qrInstance, const QrStartFrameInfo *pStartInfo)
{
    G_EntryPointCalls.store(0, std::memory_order_relaxed);

    return Call(qrInstance, &VulkanDevice::StartFrame, pStartInfo);
}

QrResult qrDrawFrame(QrInstance qrInstance, const QrDrawFrameInfo *pDrawInfo)
{
    return Call(qrInstance, &VulkanDevice::DrawFrame, pDrawInfo);
}

QrResult qrSetFogVolumes(QrInstance qrInstance, uint32_t count, const QrFogVolume *pVolumes)
{
    return Call(qrInstance, &VulkanDevice::SetFogVolumes, count, pVolumes);
}

QrBool32 qrIsRenderUpscaleTechniqueAvailable(QrInstance qrInstance, QrRenderUpscaleTechnique technique)
{
    return Call(qrInstance, &VulkanDevice::IsRenderUpscaleTechniqueAvailable, technique);
}

QrBool32 qrIsSuspended(QrInstance qrInstance)
{
    try
    {
        return GetDevice(qrInstance).IsSurfaceUnavailable() ? QR_TRUE : QR_FALSE;
    }
    catch (QrException &e)
    {
        TryPrintError(qrInstance, e.what());
    }

    return QR_TRUE;
}

QrResult qrGetFrameStats(QrInstance qrInstance, uint32_t *pRays, uint32_t *pFpsX10)
{
    return Call(qrInstance, &VulkanDevice::GetFrameStats, pRays, pFpsX10);
}

QrResult qrGetFrameStatsEx(QrInstance qrInstance, QrFrameStats *pStats)
{
    QrResult r = Call(qrInstance, &VulkanDevice::GetFrameStatsEx, pStats);

    if (r == QR_SUCCESS && pStats != nullptr)
    {
        pStats->apiCalls = G_EntryPointCalls.load(std::memory_order_relaxed);
    }

    return r;
}

QrResult qrGetAdapterInfo(QrInstance qrInstance, QrAdapterInfo *pInfo)
{
    return Call(qrInstance, &VulkanDevice::GetAdapterInfo, pInfo);
}

QrResult qrRequestScreenshot(QrInstance qrInstance, const char *pFilePath)
{
    return Call(qrInstance, &VulkanDevice::RequestScreenshot, pFilePath);
}

const char *qrGetCpuPassName(uint32_t passIndex)
{
    static const char *const names[QR_CPU_PASS_COUNT] =
    {
        "prepare",
        "hot reload",
        "descriptors",
        "staging",
        "legacy AS",
        "slot wait",
        "slot GC",
        "GPU queries",
        "RHI setup",
        "scene record",
        "compose",
        "upscale",
        "post",
        "UI record",
        "postui",
        "present record",
        "RHI submit",
        "legacy submit",
        "present",
    };
    return passIndex < QR_CPU_PASS_COUNT ? names[passIndex] : "";
}

const char *qrGetGpuPassName(uint32_t passIndex)
{
    static const char *const passNames[QR_GPU_PASS_COUNT] =
    {
        "setup",
        "clouds",
        "sky",
        "primary",
        "decals",
        "godrays",
        "reflrefr",
        "reflgodr",
        "gradient",
        "direct",
        "indirect",
        "compose",
        "upscale",
        "post",
        "ui",
        "postui",
        "present",
        "particles",
    };

    if (passIndex >= QR_GPU_PASS_COUNT)
    {
        return "";
    }

    return passNames[passIndex];
}

const char *qrGetResultDescription(QrResult result)
{
    return QrException::GetQrResultName(result);
}
