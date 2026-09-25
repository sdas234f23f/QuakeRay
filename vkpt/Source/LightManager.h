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

#pragma once

#include <cstdint>
#include <vector>

#include "vkpt/vkpt.h"
#include "Common.h"
#include "Containers.h"
#include "AutoBuffer.h"
#include "LightDefs.h"

namespace vkpt
{

struct ShLightEncoded;

class LightManager
{
public:
    // The light array the shader indexes with the light count a frame publishes,
    // which is also LIGHT_ARRAY_MAX_SIZE of the upload paths.
    static constexpr uint32_t LIGHT_ARRAY_ENTRY_COUNT = 4096;

    /* The cluster count and the number of rotating slots the light statistics buffer is cut into,
       which the shaders derive from the constants the generator shares with the host. They stand
       here as their own values, because including the generated header would hand its macros to
       every translation unit that includes this one; the upload path checks that the two agree. */
    static constexpr uint32_t LIGHT_STATS_CLUSTER_COUNT = 8192;
    static constexpr uint32_t LIGHT_STATS_SLOT_COUNT = 3;

    // One bit per cluster of the sky visibility table, kept in uint32 words like the shader
    // reads them; the same table the upload path checks Q2_MAX_CLUSTERS against.
    static constexpr uint32_t CLUSTER_SKY_VIS_WORD_COUNT = LIGHT_STATS_CLUSTER_COUNT / 32;

    LightManager(VkDevice device, std::shared_ptr<MemoryAllocator> &allocator, VkBuffer talCdfBuffer);
    ~LightManager();

    LightManager(const LightManager &other) = delete;
    LightManager(LightManager &&other) noexcept = delete;
    LightManager &operator=(const LightManager &other) = delete;
    LightManager &operator=(LightManager &&other) noexcept = delete;

    void PrepareForFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void Reset();

    uint32_t GetLightCount() const;
    uint32_t GetLightCountPrev() const;
    uint32_t DoesDirectionalLightExist() const;

    // Host-side copy of the last uploaded directional light (for god rays /
    // shadow map). Returns false if no directional light was ever uploaded.
    bool GetLastDirectionalLight(float outColor[3], float outDirection[3], float *outAngularRadius) const;

    uint32_t GetLightIndexIgnoreFPVShadows(uint32_t frameIndex, uint64_t *pLightUniqueId) const;

    void AddSphericalLight(uint32_t frameIndex, const RgSphericalLightUploadInfo &info);
    void AddPolygonalLight(uint32_t frameIndex, const RgPolygonalLightUploadInfo &info);
    void AddTexturedAreaLight(uint32_t frameIndex, const RgTexturedAreaLightUploadInfo &info, uint32_t textureIndex);
    void AddDirectionalLight(uint32_t frameIndex, const RgDirectionalLightUploadInfo &info);
    void AddSpotlight(uint32_t frameIndex, const RgSpotLightUploadInfo &info);

    void CopyFromStaging(VkCommandBuffer cmd, uint32_t frameIndex);

    void SetClusterLightLists(uint32_t frameIndex, uint32_t numClusters,
                              const uint32_t *pOffsets, const uint64_t *pLightUniqueIds,
                              uint32_t totalLightCount, uint64_t listGeneration);

    /* Replaces the per-cluster sky visibility the shaders gate the sun shadow ray with.
       pBits is the host table, bit c of byte c/8 naming cluster c; nullptr (a map that
       sent none) leaves every cluster tracing. */
    void SetClusterSkyVisibility(const uint8_t *pBits, uint32_t numClusters);

    void ResetLightStats(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t frameId);
    void BarrierQ2ClusterLists(VkCommandBuffer cmd, uint32_t frameIndex);

    /* The read-only fill plan the RHI direct pass mirrors from ResetLightStats: the count of
       clusters the lists the device holds can name, clamped to LIGHT_STATS_CLUSTER_COUNT, and the
       bytes one cluster's counter pair spans. One rotating slot spans
       GetLightStatsClusterSize() * LIGHT_STATS_CLUSTER_COUNT bytes, and there are
       LIGHT_STATS_SLOT_COUNT of them; both values are what ResetLightStats fills with
       (LightManager.cpp:858-886). The cleared state itself stays private: the RHI pass keeps its
       own mirror of statsClusterCleared and reports the fill it actually records, so the legacy
       path's bookkeeping is untouched. */
    uint32_t GetLightStatsClusterTarget() const;
    VkDeviceSize GetLightStatsClusterSize() const;

    VkDescriptorSetLayout GetDescSetLayout();
    VkDescriptorSet GetDescSet(uint32_t frameIndex);

    /* The engine buffers the light-source descriptor set binds, as the RHI pass wraps them: the
       device-local buffer of every item it declares. The statistics buffer is cut into rotating
       slots, but it is one buffer and is bound whole, as the engine set binds it. */
    struct Buffers
    {
        VkBuffer lights;        // device-local, 4096 * sizeof(ShLightEncoded)
        VkBuffer listOffsets;
        VkBuffer listLights;
        VkBuffer lightStats;
        VkBuffer clusterSkyVis;
    };

    // One copy the RHI has to record itself: the frame's staging buffer and the byte count to
    // copy from it.
    struct Copy
    {
        VkBuffer staging;       // AutoBuffer::GetStaging(frame)
        VkDeviceSize size;      // 0 == nothing pending this frame
    };

    struct FrameCopies
    {
        Copy lights;
        Copy listOffsets;
        Copy listLights;
        Copy clusterSkyVis;
    };

    Buffers GetBuffers() const;

    /* Mirrors what CopyFromStaging uploads for the frame: the light-array prefix is copied every
       frame and always holds the sun slot, while each list buffer is copied only while its
       publication or sky visibility update is pending. The prefix size is
       (LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + GetLightCount()) * sizeof(ShLightEncoded), the sun
       prefix being what GetLightCount() alone would drop, and the list-word count is
       publishedListWords[frame], the word count the publication of this slot recorded, not the
       count of the frame being recorded. */
    FrameCopies GetFrameCopies(uint32_t frame) const;

    /* Clears the pending flags GetFrameCopies reports for the frame, exactly the way
       CopyFromStaging clears them once it has copied the staging buffers. It has to be called on
       the path that recorded the copies and only there: a frame that leaves the flags set is a
       frame a later pass (or the legacy path) still copies, while a frame that clears them
       without copying leaves the device buffers stale. The light-array prefix has no flag - it is
       copied every frame by design - and is not consumed here. */
    void ConsumeFrameCopies(uint32_t frame);

private:
    /* What one frame has registered so far. A light is looked up once per frame it survives in and
       inserted once, and a hash map pays a node indirection for each of those, so the table is
       open addressed and flat. An entry holds the generation of the frame that wrote it, which is
       what makes the entries of the frames before it free without the table being cleared: taking
       a new generation is all that emptying a frame's registry costs. Sixteen bytes an entry, of
       which a lookup reads the generation first and then, on a hit, the two words that follow. */
    struct RegistryEntry
    {
        uint32_t generation; // 0 while no frame has written the slot
        uint32_t arrayIndex;
        uint64_t uniqueID;
    };

    // Twice the light array, so a full frame still leaves the table half empty and the
    // probe sequences short.
    static constexpr uint32_t LIGHT_REGISTRY_SIZE = 2 * LIGHT_ARRAY_ENTRY_COUNT;
    static constexpr uint32_t LIGHT_REGISTRY_MASK = LIGHT_REGISTRY_SIZE - 1;

    // The slot the light of this id occupies in the registry of `generation`, or the slot it has
    // to be written to when `found` is false.
    static uint32_t GetRegistrySlot(const RegistryEntry *entries, uint32_t generation, uint64_t uniqueID, bool &found);

    // Where the frame put the light of this id in its light array, if that frame registered it.
    bool FindRegisteredLight(uint32_t frameIndex, uint64_t uniqueID, uint32_t &outArrayIndex) const;

    LightArrayIndex GetIndex(const ShLightEncoded &encodedLight) const;
    void IncrementCount(const ShLightEncoded &encodedLight);
    void AddLight(uint32_t frameIndex, uint64_t uniqueId, const ShLightEncoded &encodedLight);

    void FillMatchPrev(uint32_t curFrameIndex, LightArrayIndex lightIndexInCurFrame, UniqueLightID uniqueID, uint32_t ordinal);

    void CreateDescriptors();
    void UpdateDescriptors(uint32_t frameIndex);

private:
    VkDevice device;
    VkBuffer talCdf;

    std::shared_ptr<AutoBuffer> lightsBuffer;
    Buffer lightsBuffer_Prev;

    std::shared_ptr<AutoBuffer> lightListOffsets;
    std::shared_ptr<AutoBuffer> lightListLights;

    /* One bit per cluster: whether a sun ray from it can still reach the sky. Written once
       per map load rather than per frame, but staged and copied through the same per-slot
       pending flags as the light lists. */
    std::shared_ptr<AutoBuffer> clusterSkyVis;
    bool                        clusterSkyVisCopyPending[MAX_FRAMES_IN_FLIGHT] = {};

    /* What the words of the list buffers are. A published word is the place of a light in the light
       array, and a light is named by its id, so the words a frame would publish are the words
       already published exactly when the composition and the place of every id the frame registered
       are both the same; then there is neither a list to resolve nor a publication to make. Both
       the registration of the frame being recorded and the record of what the slot was last given
       are kept per frame slot, as the registry the places belong to is: an AutoBuffer keeps one
       staging buffer per frame in flight and a single device buffer, so the words a publication is
       written into belong to the slot that made it, and the copy that follows it carries the words
       that slot published (`publishedListWords`) two frames later. */
    bool     lightListCopyPending[MAX_FRAMES_IN_FLIGHT] = {};
    bool     publishedListValid[MAX_FRAMES_IN_FLIGHT] = {};
    uint64_t publishedListGeneration[MAX_FRAMES_IN_FLIGHT] = {};
    uint32_t publishedListClusters[MAX_FRAMES_IN_FLIGHT] = {};
    uint32_t publishedListWords[MAX_FRAMES_IN_FLIGHT] = {};
    std::vector<uint64_t> publishedLightOrder[MAX_FRAMES_IN_FLIGHT];
    // The place in the light array each of those ids was given, which together with the order is
    // what the ids of a publication resolve to.
    std::vector<uint32_t> publishedLightIndex[MAX_FRAMES_IN_FLIGHT];

    Buffer lightStats;

    /* Highest index of the light statistics buffer that the lists the device holds can name, which
       is the cluster count of the last publication, and how many clusters from the start of each
       statistics slot are known to hold no counter. A cluster that no list names is never read, so
       clearing the clusters the lists can name is what the fill has to cover, and the 48 MiB a
       slot spans are only touched while the lists of a scene still reach that far. */
    uint32_t statsClusterTarget = LIGHT_STATS_CLUSTER_COUNT;
    uint32_t statsClusterCleared[LIGHT_STATS_SLOT_COUNT] = {};

    // Match light indices between current and previous frames
    std::shared_ptr<AutoBuffer> prevToCurIndex;
    std::shared_ptr<AutoBuffer> curToPrevIndex;

    RegistryEntry registry[MAX_FRAMES_IN_FLIGHT][LIGHT_REGISTRY_SIZE] = {};
    // Never zero: zero is the generation of a slot no frame has written, so a frame before its
    // first PrepareForFrame still sees no light as registered instead of seeing all of them.
    uint32_t registryGeneration[MAX_FRAMES_IN_FLIGHT] = {};

    /* The place each light of the frame had in the light array, in the order the frame registered
       them in, which is the order registeredLightOrder holds the ids in: the two together say what
       a place in the light array meant to the frame before, so a frame that registers the same ids
       in the same order as the frame before it needs no lookup to match one of them. Both are as
       long as the frame registered lights, which the light array bounds. */
    std::vector<uint64_t> registeredLightOrder[MAX_FRAMES_IN_FLIGHT];
    std::vector<uint32_t> registeredLightIndex[MAX_FRAMES_IN_FLIGHT];

    uint32_t regLightCount;
    uint32_t regLightCount_Prev;
    uint32_t dirLightCount;
    uint32_t dirLightCount_Prev;

    // host-side copy of the last directional light
    float lastDirLightColor[3];
    float lastDirLightDirection[3];
    float lastDirLightAngularRadius;

    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool descPool;
    VkDescriptorSet descSets[MAX_FRAMES_IN_FLIGHT];

    bool needDescSetUpdate[MAX_FRAMES_IN_FLIGHT];
};

}