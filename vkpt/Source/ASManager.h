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

#include "ASBuilder.h"
#include "CommandBufferManager.h"
#include "GlobalUniform.h"
#include "ScratchBuffer.h"
#include "TextureManager.h"
#include "VertexBufferProperties.h"
#include "VertexCollector.h"
#include "ASComponent.h"

namespace vkpt
{

struct ShVertPreprocessing;

class ASManager
{
public:
    struct TLASPrepareResult
    {
        VkAccelerationStructureInstanceKHR instances[45];
        uint32_t instanceCount;
    };

public:
    ASManager(VkDevice device, 
              std::shared_ptr<PhysicalDevice> physDevice,
              std::shared_ptr<MemoryAllocator> allocator,
              std::shared_ptr<CommandBufferManager> cmdManager,
              std::shared_ptr<TextureManager> textureManager,
              std::shared_ptr<GeomInfoManager> geomInfoManager);
    ~ASManager();

    ASManager(const ASManager& other) = delete;
    ASManager(ASManager&& other) noexcept = delete;
    ASManager& operator=(const ASManager& other) = delete;
    ASManager& operator=(ASManager&& other) noexcept = delete;


    void BeginStaticGeometry();
    uint32_t AddStaticGeometry(uint32_t frameIndex, const RgGeometryUploadInfo &info);
    // Submitting static geometry to the building is a heavy operation
    // with waiting for it to complete.
    void SubmitStaticGeometry();
    // If all the added geometries must be removed, call this function before submitting
    void ResetStaticGeometry();

    void BeginDynamicGeometry(VkCommandBuffer cmd, uint32_t frameIndex);
    uint32_t AddDynamicGeometry(uint32_t frameIndex, const RgGeometryUploadInfo &info);
    void SubmitDynamicGeometry(VkCommandBuffer cmd, uint32_t frameIndex);


    // Update transform for static movable geometry
    void UpdateStaticMovableTransform(uint32_t simpleIndex, const RgUpdateTransformInfo &updateInfo);
    // After updating transforms, acceleration structures should be rebuilt
    void ResubmitStaticMovable(VkCommandBuffer cmd);

    // Update texture coordinates for static geometry, it 
    // doesn't require AS rebuilding, but only copying from staging to device-local 
    void UpdateStaticTexCoords(uint32_t simpleIndex, const RgUpdateTexCoordsInfo &texCoordsInfo);
    void ResubmitStaticTexCoords(VkCommandBuffer cmd);


    // Prepare data for building TLAS.
    // Also fill uniform with current state.
    std::pair<TLASPrepareResult, ShVertPreprocessing> PrepareForBuildingTLAS(
        uint32_t frameIndex,
        ShGlobalUniform &uniformData,
        uint32_t uniformData_rayCullMaskWorld,
        bool allowGeometryWithSkyFlag,
        bool disableRTGeometry) const;
    void BuildTLAS(
        VkCommandBuffer cmd, uint32_t frameIndex, 
        const TLASPrepareResult &info);


    // Copy current dynamic vertex and index data to
    // special buffers for using current frame's data in the next frame.
    void CopyDynamicDataToPrevBuffers(VkCommandBuffer cmd, uint32_t frameIndex);


    void OnVertexPreprocessingBegin(VkCommandBuffer cmd, uint32_t frameIndex, bool onlyDynamic);
    void OnVertexPreprocessingFinish(VkCommandBuffer cmd, uint32_t frameIndex, bool onlyDynamic);


    VkDescriptorSet GetBuffersDescSet(uint32_t frameIndex) const;
    VkDescriptorSet GetTLASDescSet(uint32_t frameIndex) const;

    VkDescriptorSetLayout GetBuffersDescSetLayout() const;
    VkDescriptorSetLayout GetTLASDescSetLayout() const;

    // World geometry collectors (for shadow map rendering).
    const std::shared_ptr<VertexCollector> &GetStaticCollector() const;
    const std::shared_ptr<VertexCollector> &GetDynamicCollector(uint32_t frameIndex) const;

    // The engine's geometry-instance manager: the owner of the geometryInstances and
    // geomIndexPrevToCur buffers of the RT vertex-data set (set 3). The RHI layer reads it through
    // this getter to keep its own per-slot copies of both, because the engine's own
    // GeomInfoManager::CopyFromStaging runs only from the legacy frame (Scene::SubmitForFrame) and
    // from the level-load submission, neither of which is part of the `rhiframe` frame
    // (RHI/RhiAccelStructs.h).
    const std::shared_ptr<GeomInfoManager> &GetGeomInfoManager() const;

    // Read-only views for the RHI layer's acceleration structures (RHI/RhiAccelStructs.cpp); no
    // behaviour change, the manager keeps owning everything below.
    // The static BLAS components, one per static filter: the RHI layer mirrors this set, creating
    // one rt::IAccelStruct per non-empty component.
    const std::vector<std::unique_ptr<BLASComponent>> &GetStaticBlasComponents() const;
    // The engine's TLAS instance buffer (device-local, MAX_TOP_LEVEL_INSTANCE_COUNT records) and
    // its byte size: the RHI layer wraps the buffer and builds its own per-slot TLAS from it.
    VkBuffer GetInstanceBuffer() const;
    VkDeviceSize GetInstanceBufferSize() const;

    // Fills every attribute of the TLAS instance that 'filter' contributes, with exactly the rules
    // the engine's own TLAS build applies: the mask against 'rayCullMaskWorld' (a missing world bit
    // drops the instance and zeroes it), the first-person/viewer/sky custom-index bits, the refract
    // mask rewrite, the alpha-tested SBT offset and the instance flags.
    // The acceleration structure reference is deliberately not touched: SetupTLASInstanceFromBLAS
    // stamps the engine's BLAS address there and the RHI layer overwrites it with its own. A bare
    // filter carries no BLAS, so the "no AS / empty component" check stays with the caller as well.
    static bool GetTLASInstanceForFilter(VertexCollectorFilterTypeFlags filter,
                                         uint32_t rayCullMaskWorld,
                                         bool allowGeometryWithSkyFlag,
                                         VkAccelerationStructureInstanceKHR &instance);

    // Incremented by SubmitStaticGeometry at every (re)submission, including the submission of an
    // empty static set. The RHI layer samples it to detect a level change, as the static BLAS
    // handles and addresses are recreated on every submission.
    uint32_t GetStaticGeneration() const;

private:
    // amount of possible VertexCollectorFilterTypeFlags_GetID values
    static constexpr uint32_t MAX_FILTER_TYPE_COUNT =
        (sizeof(VertexCollectorFilterGroup_ChangeFrequency) / sizeof(VertexCollectorFilterGroup_ChangeFrequency[0])) *
        (sizeof(VertexCollectorFilterGroup_PassThrough) / sizeof(VertexCollectorFilterGroup_PassThrough[0])) *
        (sizeof(VertexCollectorFilterGroup_PrimaryVisibility) / sizeof(VertexCollectorFilterGroup_PrimaryVisibility[0]));

    void CreateDescriptors();
    void UpdateBufferDescriptors(uint32_t frameIndex);
    void UpdateASDescriptors(uint32_t frameIndex);

    bool SetupBLAS(
        BLASComponent &as,
        const std::shared_ptr<VertexCollector> &vertCollector);

    void UpdateBLAS(
        BLASComponent &as,
        const std::shared_ptr<VertexCollector> &vertCollector);

    static bool SetupTLASInstanceFromBLAS(
        const BLASComponent &as,
        uint32_t rayCullMaskWorld, 
        bool allowGeometryWithSkyFlag,
        VkAccelerationStructureInstanceKHR &instance);

    static bool IsFastBuild(VertexCollectorFilterTypeFlags filter);

private:
    VkDevice device;
    std::shared_ptr<MemoryAllocator> allocator;

    VkFence staticCopyFence;

    // for filling buffers
    std::shared_ptr<VertexCollector> collectorStatic;
    std::shared_ptr<VertexCollector> collectorDynamic[MAX_FRAMES_IN_FLIGHT];
    // device-local buffer for storing previous info
    Buffer previousDynamicPositions;
    Buffer previousDynamicIndices;

    // building
    std::shared_ptr<ScratchBuffer> scratchBuffer;
    std::shared_ptr<ASBuilder> asBuilder;

    std::shared_ptr<CommandBufferManager> cmdManager;
    std::shared_ptr<TextureManager> textureMgr;
    std::shared_ptr<GeomInfoManager> geomInfoMgr;

    std::vector<std::unique_ptr<BLASComponent>> allStaticBlas;
    std::vector<std::unique_ptr<BLASComponent>> allDynamicBlas[MAX_FRAMES_IN_FLIGHT];

    // A new value means a new static set, even if that set is empty (see GetStaticGeneration).
    uint32_t staticGeneration = 0;

    // A dynamic BLAS is rebuilt only if the input that it was built from changed. The
    // state is per frame slot: the AS that a slot reuses is the one that was built for
    // the same slot MAX_FRAMES_IN_FLIGHT frames ago, and each slot owns its own
    // dynamic BLAS objects. dynBuildHash is the hash of the current frame's geometry,
    // dynBlasKey is the hash that the slot's BLAS was built from.
    uint64_t dynBuildHash[MAX_FRAMES_IN_FLIGHT][MAX_FILTER_TYPE_COUNT] = {};
    uint64_t dynBlasKey[MAX_FRAMES_IN_FLIGHT][MAX_FILTER_TYPE_COUNT] = {};
    bool dynBlasKeyValid[MAX_FRAMES_IN_FLIGHT][MAX_FILTER_TYPE_COUNT] = {};

    // TLAS build sizes depend only on the instance count
    VkAccelerationStructureBuildSizesInfoKHR tlasBuildSizes[MAX_FRAMES_IN_FLIGHT] = {};
    uint32_t tlasBuildSizesInstanceCount[MAX_FRAMES_IN_FLIGHT] = {};
    bool tlasBuildSizesValid[MAX_FRAMES_IN_FLIGHT] = {};

    // top level AS
    std::unique_ptr<AutoBuffer> instanceBuffer;
    std::unique_ptr<TLASComponent> tlas[MAX_FRAMES_IN_FLIGHT];

    // TLAS and buffer descriptors
    VkDescriptorPool descPool;

    VkDescriptorSetLayout buffersDescSetLayout;
    VkDescriptorSet buffersDescSets[MAX_FRAMES_IN_FLIGHT];

    VkDescriptorSetLayout asDescSetLayout;
    VkDescriptorSet asDescSets[MAX_FRAMES_IN_FLIGHT];
    // AS handle that asDescSets[i] was written with
    VkAccelerationStructureKHR asDescHandles[MAX_FRAMES_IN_FLIGHT] = {};
};

}