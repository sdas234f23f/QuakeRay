// Copyright (c) 2021 Sultim Tsyrendashiev
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

#include "LightManager.h"

#include <cmath>
#include <cstring>
#include <array>
#include <cstdio>

#include "Generated/ShaderCommonC.h"
#include "CmdLabel.h"
#include "RgException.h"
#include "Utils.h"

namespace vkpt
{
constexpr double RG_PI = 3.1415926535897932384626433;

constexpr float MIN_COLOR_SUM = 0.0001f;
constexpr float MIN_SPHERE_RADIUS = 0.005f;

constexpr uint32_t LIGHT_ARRAY_MAX_SIZE = vkpt::LightManager::LIGHT_ARRAY_ENTRY_COUNT;

// The header mirrors these to stay free of the generated macros, so they have to agree.
static_assert(vkpt::LightManager::LIGHT_STATS_CLUSTER_COUNT == Q2_MAX_CLUSTERS, "cluster count of the light statistics buffer");
static_assert(vkpt::LightManager::LIGHT_STATS_SLOT_COUNT == Q2_LIGHT_LIST_STATS_BUFFERS, "slots of the light statistics buffer");

}

vkpt::LightManager::LightManager(
    VkDevice _device,
    std::shared_ptr<MemoryAllocator> &_allocator,
    VkBuffer _talCdfBuffer)
:
    device(_device),
    talCdf(_talCdfBuffer),
    regLightCount(0),
    regLightCount_Prev(0),
    dirLightCount(0),
    dirLightCount_Prev(0),
    descSetLayout(VK_NULL_HANDLE),
    descPool(VK_NULL_HANDLE),
    descSets{},
    needDescSetUpdate{}
{
    // No frame is ever on generation zero, which is what an entry of a slot no frame has
    // written holds, so no light is registered before the frame that registers it ran.
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        registryGeneration[i] = 1;
    }

    lightsBuffer    = std::make_shared<AutoBuffer>(device, _allocator);
    lightsBuffer->Create(sizeof(ShLightEncoded) * LIGHT_ARRAY_MAX_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "Lights buffer");

    lightsBuffer_Prev.Init(_allocator, sizeof(ShLightEncoded) * LIGHT_ARRAY_MAX_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Lights buffer - prev");

    lightListOffsets = std::make_shared<AutoBuffer>(device, _allocator);
    lightListOffsets->Create(sizeof(uint32_t) * (Q2_MAX_CLUSTERS + 1),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        "Q2 light list offsets");

    lightListLights = std::make_shared<AutoBuffer>(device, _allocator);
    lightListLights->Create(sizeof(uint32_t) * Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        "Q2 light list lights");

    lightStats.Init(_allocator,
        sizeof(uint32_t) * Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL * Q2_LIGHT_LIST_STATS_SIDES * 2 * Q2_LIGHT_LIST_STATS_BUFFERS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Q2 light stats");

    /* One bit per cluster: whether a sun ray from it can still reach the sky. The staging
       starts all-visible, so a frame before the first map upload traces as it always has. */
    clusterSkyVis = std::make_shared<AutoBuffer>(device, _allocator);
    clusterSkyVis->Create(sizeof(uint32_t) * CLUSTER_SKY_VIS_WORD_COUNT,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        "Q2 cluster sky visibility");

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        memset(clusterSkyVis->GetMapped(i), 0xFF, sizeof(uint32_t) * CLUSTER_SKY_VIS_WORD_COUNT);
        clusterSkyVisCopyPending[i] = true;
    }

    prevToCurIndex = std::make_shared<AutoBuffer>(device, _allocator);
    prevToCurIndex->Create(sizeof(uint32_t) * LIGHT_ARRAY_MAX_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "Lights buffer - prev to cur");

    curToPrevIndex = std::make_shared<AutoBuffer>(device, _allocator);
    curToPrevIndex->Create(sizeof(uint32_t) * LIGHT_ARRAY_MAX_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "Lights buffer - cur to prev");

    CreateDescriptors();
}

vkpt::LightManager::~LightManager()
{
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
}

static vkpt::ShLightEncoded EncodeAsDirectionalLight(const RgDirectionalLightUploadInfo &info)
{
    RgFloat3D direction = info.direction;
    vkpt::Utils::Normalize(direction.data);

    float angularRadius = static_cast<float>(0.5 * static_cast<double>(info.angularDiameterDegrees) * vkpt::RG_PI / 180.0);


    vkpt::ShLightEncoded lt = {};
    lt.lightType = LIGHT_TYPE_DIRECTIONAL;

    lt.color[0] = info.color.data[0];
    lt.color[1] = info.color.data[1];
    lt.color[2] = info.color.data[2];

    lt.data_0[0] = direction.data[0];
    lt.data_0[1] = direction.data[1];
    lt.data_0[2] = direction.data[2];

    lt.data_0[3] = angularRadius;

    return lt;
}

static vkpt::ShLightEncoded EncodeAsSphereLight(const RgSphericalLightUploadInfo &info)
{
    float radius = std::max(vkpt::MIN_SPHERE_RADIUS, info.radius);
    // disk is visible from the point
    float area = static_cast<float>(vkpt::RG_PI) * radius * radius;


    vkpt::ShLightEncoded lt = {};
    lt.lightType = LIGHT_TYPE_SPHERE;

    lt.color[0] = info.color.data[0] / area;
    lt.color[1] = info.color.data[1] / area;
    lt.color[2] = info.color.data[2] / area;

    lt.data_0[0] = info.position.data[0];
    lt.data_0[1] = info.position.data[1];
    lt.data_0[2] = info.position.data[2];

    lt.data_0[3] = radius;

    lt.data_1[0] = info.normal.data[0];
    lt.data_1[1] = info.normal.data[1];
    lt.data_1[2] = info.normal.data[2];

    return lt;
}

static vkpt::ShLightEncoded EncodeAsTriangleLight(const RgPolygonalLightUploadInfo &info, const RgFloat3D &unnormalizedNormal)
{
    RgFloat3D n = unnormalizedNormal;
    float len = vkpt::Utils::Length(n.data);
    n.data[0] /= len;
    n.data[1] /= len;
    n.data[2] /= len;

    float area = len * 0.5f;
    assert(area > 0.0f);


    vkpt::ShLightEncoded lt = {};
    lt.lightType = LIGHT_TYPE_TRIANGLE;



    lt.color[0] = info.color.data[0] / area;
    lt.color[1] = info.color.data[1] / area;
    lt.color[2] = info.color.data[2] / area;

    lt.data_0[0] = info.positions[0].data[0];
    lt.data_0[1] = info.positions[0].data[1];
    lt.data_0[2] = info.positions[0].data[2];

    lt.data_1[0] = info.positions[1].data[0];
    lt.data_1[1] = info.positions[1].data[1];
    lt.data_1[2] = info.positions[1].data[2];

    lt.data_2[0] = info.positions[2].data[0];
    lt.data_2[1] = info.positions[2].data[1];
    lt.data_2[2] = info.positions[2].data[2];

    lt.data_0[3] = unnormalizedNormal.data[0];
    lt.data_1[3] = unnormalizedNormal.data[1];
    lt.data_2[3] = unnormalizedNormal.data[2];

    return lt;
}

static vkpt::ShLightEncoded EncodeAsTexturedAreaLight(const RgTexturedAreaLightUploadInfo &info, uint32_t textureIndex)
{
    vkpt::ShLightEncoded lt = {};
    lt.lightType = LIGHT_TYPE_TEXTURED_AREA;

    lt.color[0] = info.color.data[0];
    lt.color[1] = info.color.data[1];
    lt.color[2] = info.color.data[2];

    lt.data_0[0] = info.A.data[0];
    lt.data_0[1] = info.A.data[1];
    lt.data_0[2] = info.A.data[2];
    memcpy(&lt.data_0[3], &textureIndex, sizeof(uint32_t));

    lt.data_1[0] = info.B.data[0];
    lt.data_1[1] = info.B.data[1];
    lt.data_1[2] = info.B.data[2];
    lt.data_1[3] = info.meanEmiss;

    lt.data_2[0] = info.C.data[0];
    lt.data_2[1] = info.C.data[1];
    lt.data_2[2] = info.C.data[2];
    lt.data_2[3] = static_cast<float>(info.numVerts);

    const int n = std::max(0, std::min(static_cast<int>(info.numVerts), static_cast<int>(MAX_TEXTURED_AREA_LIGHT_VERTS)));
    float *slots[4] = { lt.data_3, lt.data_4, lt.data_5, lt.data_6 };
    for (int i = 0; i < n; i++)
    {
        float *slot = slots[i >> 1];
        const int k = (i & 1) * 2;
        slot[k]     = info.uvVerts[i].data[0];
        slot[k + 1] = info.uvVerts[i].data[1];
    }

    lt.data_7[0] = info.normal.data[0];
    lt.data_7[1] = info.normal.data[1];
    lt.data_7[2] = info.normal.data[2];
    lt.data_7[3] = info.area;

    return lt;
}

static vkpt::ShLightEncoded EncodeAsSpotLight(const RgSpotLightUploadInfo &info)
{
    RgFloat3D direction = info.direction;
    vkpt::Utils::Normalize(direction.data);

    float radius = std::max(vkpt::MIN_SPHERE_RADIUS, info.radius);
    float area = static_cast<float>(vkpt::RG_PI) * radius * radius;

    float cosAngleInner = std::cos(std::min(info.angleInner, info.angleOuter));
    float cosAngleOuter = std::cos(info.angleOuter);


    vkpt::ShLightEncoded lt = {};
    lt.lightType = LIGHT_TYPE_SPOT;

    lt.color[0] = info.color.data[0] / area;
    lt.color[1] = info.color.data[1] / area;
    lt.color[2] = info.color.data[2] / area;

    lt.data_0[0] = info.position.data[0];
    lt.data_0[1] = info.position.data[1];
    lt.data_0[2] = info.position.data[2];

    lt.data_0[3] = radius;

    lt.data_1[0] = direction.data[0];
    lt.data_1[1] = direction.data[1];
    lt.data_1[2] = direction.data[2];

    lt.data_2[0] = cosAngleInner;
    lt.data_2[1] = cosAngleOuter;

    return lt;
}

static uint32_t GetLightArrayEnd(uint32_t regCount, uint32_t dirCount)
{
    // assuming that reg lights are always after directional ones
    return LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + regCount;
}

void vkpt::LightManager::PrepareForFrame(VkCommandBuffer cmd, uint32_t frameIndex)
{
    regLightCount_Prev = regLightCount;
    dirLightCount_Prev = dirLightCount;

    regLightCount = 0;
    dirLightCount = 0;

    // TODO: similar system to just swap desc sets, instead of actual copying
    if (GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev) > 0)
    {
        VkBufferCopy info = {};
        info.srcOffset = 0;
        info.dstOffset = 0;
        info.size = GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev) * sizeof(ShLightEncoded);

        vkCmdCopyBuffer(
            cmd,
            lightsBuffer->GetDeviceLocal(), lightsBuffer_Prev.GetBuffer(),
            1, &info);
    }

    memset(prevToCurIndex->GetMapped(frameIndex), 0xFF, sizeof(uint32_t) * GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev));
    // no need to clear curToPrevIndex, as it'll be filled in the cur frame

    /* Emptying the registry of the frame costs a generation of it, as the entries the slot holds
       are then of a frame the lookups of this one reject on their first read. */
    if (++registryGeneration[frameIndex] == 0)
    {
        memset(registry[frameIndex], 0, sizeof(registry[frameIndex]));
        registryGeneration[frameIndex] = 1;
    }
    registeredLightOrder[frameIndex].clear();
    registeredLightIndex[frameIndex].clear();
}

void vkpt::LightManager::Reset()
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        memset(prevToCurIndex->GetMapped(i), 0xFF, sizeof(uint32_t) * std::max(GetLightArrayEnd(regLightCount, dirLightCount), GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev)));
        memset(curToPrevIndex->GetMapped(i), 0xFF, sizeof(uint32_t) * std::max(GetLightArrayEnd(regLightCount, dirLightCount), GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev)));

        registeredLightOrder[i].clear();
        registeredLightIndex[i].clear();

        memset(registry[i], 0, sizeof(registry[i]));
        registryGeneration[i] = 1;

        publishedListValid[i] = false;
        publishedLightOrder[i].clear();
        publishedLightIndex[i].clear();
        lightListCopyPending[i] = false;

        memset(clusterSkyVis->GetMapped(i), 0xFF, sizeof(uint32_t) * CLUSTER_SKY_VIS_WORD_COUNT);
        clusterSkyVisCopyPending[i] = true;
    }

    /* No list of the scene to come is known yet, so the statistics fill has to keep covering the
       whole range until one is published. */
    statsClusterTarget = Q2_MAX_CLUSTERS;
    memset(statsClusterCleared, 0, sizeof(statsClusterCleared));

    regLightCount_Prev = regLightCount = 0;
    dirLightCount_Prev = dirLightCount = 0;
}

static bool IsColorTooDim(const float c[3])
{
    return
        std::max(c[0], 0.0f) +
        std::max(c[1], 0.0f) + 
        std::max(c[2], 0.0f) < vkpt::MIN_COLOR_SUM;
}

vkpt::LightArrayIndex vkpt::LightManager::GetIndex(const vkpt::ShLightEncoded &encodedLight) const
{
    switch (encodedLight.lightType)
    {
    case LIGHT_TYPE_DIRECTIONAL:
        return LightArrayIndex{ LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET + dirLightCount };
    case LIGHT_TYPE_SPHERE:
    case LIGHT_TYPE_TRIANGLE:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_TEXTURED_AREA:
        return LightArrayIndex{ LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + regLightCount };
    default:
        assert(0);
        return LightArrayIndex{ 0 };
    }
}

void vkpt::LightManager::IncrementCount(const ShLightEncoded& encodedLight)
{
    switch (encodedLight.lightType)
    {
        case LIGHT_TYPE_DIRECTIONAL:
            dirLightCount++;
            break;
        case LIGHT_TYPE_SPHERE:
        case LIGHT_TYPE_TRIANGLE:
        case LIGHT_TYPE_SPOT:
        case LIGHT_TYPE_TEXTURED_AREA:
            regLightCount++;
            break;
        default:
            assert(0);
    }
}

uint32_t vkpt::LightManager::GetRegistrySlot(const RegistryEntry *entries, uint32_t generation, uint64_t uniqueID, bool &found)
{
    const uint64_t hash = uniqueID * 0x9E3779B97F4A7C15ull;
    uint32_t       slot = static_cast<uint32_t>(hash >> 32) & LIGHT_REGISTRY_MASK;

    found = false;

    for (;;)
    {
        const RegistryEntry &entry = entries[slot];
        if (entry.generation != generation)
        {
            // Either the slot was never written or it holds a light of a frame before this one,
            // and then it holds a light this frame neither registered nor can match a lookup to.
            return slot;
        }
        if (entry.uniqueID == uniqueID)
        {
            found = true;
            return slot;
        }
        slot = (slot + 1) & LIGHT_REGISTRY_MASK;
    }
}

bool vkpt::LightManager::FindRegisteredLight(uint32_t frameIndex, uint64_t uniqueID, uint32_t &outArrayIndex) const
{
    bool found = false;
    const uint32_t slot = GetRegistrySlot(registry[frameIndex], registryGeneration[frameIndex], uniqueID, found);

    if (found)
    {
        outArrayIndex = registry[frameIndex][slot].arrayIndex;
    }
    return found;
}

void vkpt::LightManager::AddLight(uint32_t frameIndex, uint64_t uniqueId, const vkpt::ShLightEncoded &encodedLight)
{
    bool found = false;
    const uint32_t registrySlot = GetRegistrySlot(registry[frameIndex], registryGeneration[frameIndex], uniqueId, found);

    if (found)
    {
        // The frame already registered this light, and has to keep naming the light it gave it
        // then, or the cluster lists resolved so far would name a place that is not the light.
        return;
    }

    if (GetLightArrayEnd(regLightCount, dirLightCount) >= LIGHT_ARRAY_MAX_SIZE)
    {
        fprintf(stderr, "vkpt: light array overflow (regLightCount=%u dirLightCount=%u LIGHT_ARRAY_MAX_SIZE=%u) - dropping light(s)\n",
                regLightCount, dirLightCount, LIGHT_ARRAY_MAX_SIZE);
        assert(0);
        return;
    }

    const LightArrayIndex index = GetIndex(encodedLight);
    IncrementCount(encodedLight);

    auto *dst = (ShLightEncoded *)lightsBuffer->GetMapped(frameIndex);
    memcpy(&dst[index.GetArrayIndex()], &encodedLight, sizeof(vkpt::ShLightEncoded));

    const uint32_t ordinal = uint32_t(registeredLightOrder[frameIndex].size());

    FillMatchPrev(frameIndex, index, uniqueId, ordinal);

    /* The generation goes in first, so that a lookup of the next frame, which is on another
       generation, rejects this entry without reading the rest of it. */
    RegistryEntry &entry = registry[frameIndex][registrySlot];
    entry.generation = registryGeneration[frameIndex];
    entry.arrayIndex = index.GetArrayIndex();
    entry.uniqueID = uniqueId;

    /* The array index just given to the light is its place in this sequence, so the sequence is
       what says whether the words a previous frame published still name the same lights. */
    registeredLightOrder[frameIndex].push_back(uniqueId);
    registeredLightIndex[frameIndex].push_back(index.GetArrayIndex());
}

void vkpt::LightManager::AddSphericalLight(uint32_t frameIndex, const RgSphericalLightUploadInfo &info)
{
    if (IsColorTooDim(info.color.data))
    {
        return;
    }

    AddLight(frameIndex, info.uniqueID, EncodeAsSphereLight(info));
}

void vkpt::LightManager::AddPolygonalLight(uint32_t frameIndex, const RgPolygonalLightUploadInfo &info)
{
    if (IsColorTooDim(info.color.data))
    {
        return;
    }

    RgFloat3D unnormalizedNormal = Utils::GetUnnormalizedNormal(info.positions);
    if (Utils::Dot(unnormalizedNormal.data, unnormalizedNormal.data) <= 0.0f)
    {
        return;
    }

    AddLight(frameIndex, info.uniqueID, EncodeAsTriangleLight(info, unnormalizedNormal));
}

void vkpt::LightManager::AddTexturedAreaLight(uint32_t frameIndex, const RgTexturedAreaLightUploadInfo &info, uint32_t textureIndex)
{
    if (IsColorTooDim(info.color.data))
    {
        return;
    }

    AddLight(frameIndex, info.uniqueID, EncodeAsTexturedAreaLight(info, textureIndex));
}

void vkpt::LightManager::AddSpotlight(uint32_t frameIndex, const RgSpotLightUploadInfo &info)
{
    if (IsColorTooDim(info.color.data) || info.radius < 0.0f || info.angleOuter <= 0.0f)
    {
        return;
    }

    AddLight(frameIndex, info.uniqueID, EncodeAsSpotLight(info));
}

void vkpt::LightManager::AddDirectionalLight(uint32_t frameIndex, const RgDirectionalLightUploadInfo &info)
{
    if (dirLightCount > 0)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Only one directional light is allowed");
    }

    if (IsColorTooDim(info.color.data) || info.angularDiameterDegrees < 0.0f)
    {
        return;
    }

    // keep a host-side copy for god rays / shadow map
    lastDirLightColor[0] = info.color.data[0];
    lastDirLightColor[1] = info.color.data[1];
    lastDirLightColor[2] = info.color.data[2];

    float dir[3] = { info.direction.data[0], info.direction.data[1], info.direction.data[2] };
    vkpt::Utils::Normalize(dir);
    lastDirLightDirection[0] = dir[0];
    lastDirLightDirection[1] = dir[1];
    lastDirLightDirection[2] = dir[2];

    lastDirLightAngularRadius = static_cast<float>(0.5 * static_cast<double>(info.angularDiameterDegrees) * vkpt::RG_PI / 180.0);

    AddLight(frameIndex, info.uniqueID, EncodeAsDirectionalLight(info));
}

bool vkpt::LightManager::GetLastDirectionalLight(float outColor[3], float outDirection[3], float *outAngularRadius) const
{
    if (dirLightCount == 0)
    {
        return false;
    }

    outColor[0] = lastDirLightColor[0];
    outColor[1] = lastDirLightColor[1];
    outColor[2] = lastDirLightColor[2];

    outDirection[0] = lastDirLightDirection[0];
    outDirection[1] = lastDirLightDirection[1];
    outDirection[2] = lastDirLightDirection[2];

    *outAngularRadius = lastDirLightAngularRadius;

    return true;
}

void vkpt::LightManager::CopyFromStaging(VkCommandBuffer cmd, uint32_t frameIndex)
{
    CmdLabel label(cmd, "Copying lights");

    lightsBuffer->CopyFromStaging(cmd, frameIndex, sizeof(ShLightEncoded) * GetLightArrayEnd(regLightCount, dirLightCount));

    /* The list buffers of the device are written once per change of the lists, not once per
       frame: a frame whose lists are the ones the frame before it published leaves them as they
       are, and with them the staging words that frame would have copied keep their age. */
    if (lightListCopyPending[frameIndex])
    {
        lightListOffsets->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * (Q2_MAX_CLUSTERS + 1));
        /* The words of the publication this slot staged, which the offsets of that same staging
           buffer delimit. It is the count that publication recorded and not whichever frame
           published last: a publication is copied two frames after it was made, when the other
           slot has published a count of its own. */
        lightListLights->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * publishedListWords[frameIndex]);

        lightListCopyPending[frameIndex] = false;
    }

    if (clusterSkyVisCopyPending[frameIndex])
    {
        clusterSkyVis->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * CLUSTER_SKY_VIS_WORD_COUNT);
        clusterSkyVisCopyPending[frameIndex] = false;
    }

    prevToCurIndex->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * GetLightArrayEnd(regLightCount_Prev, dirLightCount_Prev));
    curToPrevIndex->CopyFromStaging(cmd, frameIndex, sizeof(uint32_t) * GetLightArrayEnd(regLightCount, dirLightCount));

    // should be used when buffers changed
    if (needDescSetUpdate[frameIndex])
    {
        UpdateDescriptors(frameIndex);
        needDescSetUpdate[frameIndex] = false;
    }
}

void vkpt::LightManager::SetClusterLightLists(uint32_t frameIndex, uint32_t numClusters,
                                              const uint32_t *pOffsets, const uint64_t *pLightUniqueIds,
                                              uint32_t totalLightCount, uint64_t listGeneration)
{
    if (numClusters > Q2_MAX_CLUSTERS)
    {
        numClusters = Q2_MAX_CLUSTERS;
    }

    /* This is the cluster count the lists the device holds from now on can name, which is how far
       the light statistics of a cluster band can reach. */
    statsClusterTarget = numClusters;

    const uint32_t lightsWords = Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL;
    const uint32_t count = std::min(totalLightCount, lightsWords);

    /* A word of a publication is the place an id resolves to, so this slot holds these words if
       the composition says so and every id resolves the way it did then, which the places the
       frame gave its lights in registration order say. Thousands of words are compared here
       against the hundreds of thousands the loop below resolves one by one. */
    const uint32_t registeredCount = uint32_t(registeredLightOrder[frameIndex].size());
    const bool     samePlaces =
        publishedLightIndex[frameIndex].size() == registeredCount &&
        (registeredCount == 0 ||
         memcmp(publishedLightIndex[frameIndex].data(), registeredLightIndex[frameIndex].data(), sizeof(uint32_t) * registeredCount) == 0);

    const bool sameAsPublished =
        publishedListValid[frameIndex] && publishedListGeneration[frameIndex] == listGeneration &&
        publishedListClusters[frameIndex] == numClusters && publishedListWords[frameIndex] == count &&
        samePlaces && publishedLightOrder[frameIndex] == registeredLightOrder[frameIndex];

    if (sameAsPublished)
    {
        return;
    }

    uint32_t *dstOffsets = static_cast<uint32_t *>(lightListOffsets->GetMapped(frameIndex));
    for (uint32_t i = 0; i <= Q2_MAX_CLUSTERS; i++)
    {
        dstOffsets[i] = 0;
    }
    for (uint32_t i = 0; i <= numClusters && i <= Q2_MAX_CLUSTERS; i++)
    {
        dstOffsets[i] = pOffsets[i];
    }

    uint32_t *dstLights = static_cast<uint32_t *>(lightListLights->GetMapped(frameIndex));

    /* One light is listed in every cluster its PVS covers, so a registry of at most a thousand
       lights still produces hundreds of thousands of entries per frame. Resolving each unique id
       through the light map once and remembering the answer keeps the loop off the hash map for
       all but the first entry of a light. */
    struct CachedIndex
    {
        uint64_t uid;
        uint32_t index; // kNotCached marks an empty slot
    };

    constexpr uint32_t kCacheSize = 2048; // larger than the light registry, so the table stays sparse
    constexpr uint32_t kNotCached = ~0u;

    CachedIndex cache[kCacheSize];
    memset(cache, 0xFF, sizeof(cache));

    for (uint32_t i = 0; i < count; i++)
    {
        const uint64_t uid = pLightUniqueIds[i];
        const uint64_t hash = uid * 0x9E3779B97F4A7C15ull;
        uint32_t       slot = static_cast<uint32_t>(hash >> 32) & (kCacheSize - 1);

        uint32_t index = 0;
        bool     found = false;
        for (uint32_t probe = 0; probe < kCacheSize; probe++)
        {
            const CachedIndex &entry = cache[slot];
            if (entry.index == kNotCached)
            {
                break; // this slot is where the uid would be inserted
            }
            if (entry.uid == uid)
            {
                index = entry.index;
                found = true;
                break;
            }
            slot = (slot + 1) & (kCacheSize - 1);
        }

        if (!found)
        {
            uint32_t resolved = 0;
            /* An id the frame does not know about resolves to no light at all. Zero is the
               directional light slot, so publishing it would name the sun instead, and the
               shader has to be able to tell the two apart to skip the entry. */
            index = FindRegisteredLight(frameIndex, uid, resolved) ? resolved : uint32_t(LIGHT_INDEX_NONE);

            if (cache[slot].index == kNotCached) // the probe above can leave slot occupied when full
            {
                cache[slot].uid = uid;
                cache[slot].index = index;
            }
        }

        dstLights[i] = index;
    }

    /* The words of this publication, for this slot to recognize as its own the next time it is
       the one being recorded. */
    publishedListValid[frameIndex] = true;
    publishedListGeneration[frameIndex] = listGeneration;
    publishedListClusters[frameIndex] = numClusters;
    publishedListWords[frameIndex] = count;
    publishedLightOrder[frameIndex] = registeredLightOrder[frameIndex];
    publishedLightIndex[frameIndex].assign(registeredLightIndex[frameIndex].begin(),
                                           registeredLightIndex[frameIndex].end());

    lightListCopyPending[frameIndex] = true;
}

void vkpt::LightManager::SetClusterSkyVisibility(const uint8_t *pBits, uint32_t numClusters)
{
    /* The table is packed into words the way the shader unpacks them. A cluster the host
       never sent a bit for - one past numClusters, or every cluster when the map sent no
       table at all - keeps the 1 it started with and keeps tracing its sun ray. */
    uint32_t words[CLUSTER_SKY_VIS_WORD_COUNT];
    memset(words, 0xFF, sizeof(words));

    if (pBits != nullptr)
    {
        const uint32_t count = std::min(numClusters, LIGHT_STATS_CLUSTER_COUNT);

        for (uint32_t c = 0; c < count; c++)
        {
            if ((pBits[c >> 3] & (1u << (c & 7u))) == 0)
            {
                words[c >> 5] &= ~(1u << (c & 31u));
            }
        }
    }

    /* A map load publishes to every slot at once, not to the one being recorded the way a
       light list publication does: the upload arrives between frames and the copy of
       whichever slot records next has to carry it. */
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        memcpy(clusterSkyVis->GetMapped(i), words, sizeof(words));
        clusterSkyVisCopyPending[i] = true;
    }
}

void vkpt::LightManager::ResetLightStats(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t frameId)
{
    const uint32_t slot = frameId % Q2_LIGHT_LIST_STATS_BUFFERS;

    const VkDeviceSize statsSlotSize =
        sizeof(uint32_t) * Q2_MAX_CLUSTERS * Q2_LIGHT_LIST_MAX_PER_CELL *
        Q2_LIGHT_LIST_STATS_SIDES * 2;

    /* A cluster no list names has no counter any shader can read, so the clusters of a slot have to
       be zero only as far as the lists the device holds can reach: on the maps whose lists stay
       inside a fraction of the cluster grid this fills a fraction of the slot, and a fill is
       proportional to its size. The slot of this frame is the one being handed over to the frames
       after it and is filled again at every rotation; the others are filled only when the lists
       grew past what a slot was last filled up to. */
    const uint32_t clusterCount = std::min(statsClusterTarget, static_cast<uint32_t>(Q2_MAX_CLUSTERS));
    const VkDeviceSize clusterSize = statsSlotSize / Q2_MAX_CLUSTERS;
    const VkDeviceSize fillSize = clusterSize * clusterCount;

    for (uint32_t i = 0; i < Q2_LIGHT_LIST_STATS_BUFFERS; i++)
    {
        if (i != slot && statsClusterCleared[i] >= clusterCount)
        {
            continue;
        }

        vkCmdFillBuffer(cmd, lightStats.GetBuffer(), statsSlotSize * i, fillSize, 0);
        statsClusterCleared[i] = clusterCount;
    }
}

void vkpt::LightManager::BarrierQ2ClusterLists(VkCommandBuffer cmd, uint32_t frameIndex)
{
    VkBuffer buffers[] =
    {
        lightListOffsets->GetDeviceLocal(),
        lightListLights->GetDeviceLocal(),
        lightStats.GetBuffer(),
    };

    VkBufferMemoryBarrier2 barriers[std::size(buffers)] = {};
    for (uint32_t i = 0; i < std::size(buffers); i++)
    {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[i].buffer = buffers[i];
        barriers[i].offset = 0;
        barriers[i].size = VK_WHOLE_SIZE;
    }

    VkDependencyInfo dependency =
    {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = std::size(barriers),
        .pBufferMemoryBarriers = barriers
    };

    svkCmdPipelineBarrier2KHR(cmd, &dependency);
}

VkDescriptorSetLayout vkpt::LightManager::GetDescSetLayout()
{
    return descSetLayout;
}

VkDescriptorSet vkpt::LightManager::GetDescSet(uint32_t frameIndex)
{
    return descSets[frameIndex];
}

void vkpt::LightManager::FillMatchPrev(uint32_t curFrameIndex, LightArrayIndex lightIndexInCurFrame, UniqueLightID uniqueID, uint32_t ordinal)
{
    uint32_t prevFrame = (curFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;

    /* The previous frame registered the same ids in the same order when the light set of the scene
       did not change, which is what happens in almost every frame: the light that is registered in
       this place then is the light that had this place, and the place it had is the one the
       previous frame recorded for it. */
    uint32_t lightIndexInPrevFrame = 0;
    if (ordinal < registeredLightOrder[prevFrame].size() && registeredLightOrder[prevFrame][ordinal] == uniqueID)
    {
        lightIndexInPrevFrame = registeredLightIndex[prevFrame][ordinal];
    }
    else if (!FindRegisteredLight(prevFrame, uniqueID, lightIndexInPrevFrame))
    {
        // The light is new to the scene, so it has no place in the frame before this one.
        return;
    }

    uint32_t *prev2cur = static_cast<uint32_t *>(prevToCurIndex->GetMapped(curFrameIndex));
    prev2cur[lightIndexInPrevFrame] = lightIndexInCurFrame.GetArrayIndex();

    uint32_t *cur2prev = static_cast<uint32_t *>(curToPrevIndex->GetMapped(curFrameIndex));
    cur2prev[lightIndexInCurFrame.GetArrayIndex()] = lightIndexInPrevFrame;
}

constexpr uint32_t BINDINGS[] =
{
    BINDING_LIGHT_SOURCES,
    BINDING_LIGHT_SOURCES_PREV,
    BINDING_LIGHT_SOURCES_INDEX_PREV_TO_CUR,
    BINDING_LIGHT_SOURCES_INDEX_CUR_TO_PREV,
    BINDING_LIGHT_SOURCES_Q2_LIGHT_LIST_OFFSETS,
    BINDING_LIGHT_SOURCES_Q2_LIGHT_LIST_LIGHTS,
    BINDING_LIGHT_SOURCES_Q2_LIGHT_STATS,
    BINDING_LIGHT_SOURCES_TAL_CDF,
    BINDING_LIGHT_SOURCES_Q2_CLUSTER_SKY_VIS,
};

void vkpt::LightManager::CreateDescriptors()
{
    VkResult r;
    
    std::array<VkDescriptorSetLayoutBinding, std::size(BINDINGS)> bindings = {};

    for (uint32_t i = 0; i < std::size(BINDINGS); i++)
    {
        uint32_t bnd = BINDINGS[i];
        assert(i == bnd);

        VkDescriptorSetLayoutBinding &b = bindings[bnd];
        b.binding = bnd;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Light buffers Desc set layout");

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindings.size() * MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Light buffers Desc set pool");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descSetLayout;
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        r = vkAllocateDescriptorSets(device, &allocInfo, &descSets[i]);
        VK_CHECKERROR(r);

        SET_DEBUG_NAME(device, descSets[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, "Light buffers Desc set");
    }
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        UpdateDescriptors(i);
    }
}

void vkpt::LightManager::UpdateDescriptors(uint32_t frameIndex)
{
    const VkBuffer buffers[] =
    {
        lightsBuffer->GetDeviceLocal(),
        lightsBuffer_Prev.GetBuffer(),
        prevToCurIndex->GetDeviceLocal(),
        curToPrevIndex->GetDeviceLocal(),
        lightListOffsets->GetDeviceLocal(),
        lightListLights->GetDeviceLocal(),
        lightStats.GetBuffer(),
        talCdf,
        clusterSkyVis->GetDeviceLocal(),
    };
    static_assert(std::size(BINDINGS) == std::size(buffers));

    std::array<VkDescriptorBufferInfo, std::size(BINDINGS)> bufs = {};
    std::array<VkWriteDescriptorSet, std::size(BINDINGS)> wrts = {};

    for (uint32_t i = 0; i < std::size(BINDINGS); i++)
    {
        uint32_t bnd = BINDINGS[i];
        // 'buffers' should be actually a map (binding->buffer), but a plain array works too, if this is true
        assert(i == bnd);

        VkDescriptorBufferInfo &b = bufs[bnd];
        b.buffer = buffers[bnd];
        b.offset = 0;
        b.range = VK_WHOLE_SIZE;
        
        VkWriteDescriptorSet &w = wrts[bnd];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descSets[frameIndex];
        w.dstBinding = bnd;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &b;
    }

    vkUpdateDescriptorSets(device, wrts.size(), wrts.data(), 0, nullptr);
}

uint32_t vkpt::LightManager::GetLightCount() const
{
    return regLightCount;
}

uint32_t vkpt::LightManager::GetLightCountPrev() const
{
    return regLightCount_Prev;
}


uint32_t vkpt::LightManager::DoesDirectionalLightExist() const
{
    return dirLightCount > 0 ? 1 : 0;
}

uint32_t vkpt::LightManager::GetLightIndexIgnoreFPVShadows(uint32_t frameIndex, uint64_t *pLightUniqueId) const
{
    if (pLightUniqueId == nullptr)
    {
        return LIGHT_INDEX_NONE;
    }

    uint32_t index = 0;
    if (!FindRegisteredLight(frameIndex, *pLightUniqueId, index))
    {
        return LIGHT_INDEX_NONE;
    }

    return index;
}

static_assert(vkpt::MAX_FRAMES_IN_FLIGHT == 2);
