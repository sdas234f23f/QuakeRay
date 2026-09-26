#include "RhiAccelStructs.h"

#include "RhiFrameContext.h"

#include "../ASComponent.h"
#include "../ASManager.h"
#include "../Generated/ShaderCommonC.h"
#include "../GeomInfoManager.h"
#include "../VertexCollector.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vkpt;

// The two structures the instance synthesis crosses. The engine's VkAccelerationStructureInstanceKHR
// has a 48-byte row-major transform followed by the 24-bit custom index, the 8-bit mask, the 24-bit
// SBT offset, the 8-bit flags and the 64-bit BLAS reference; rt::InstanceDesc mirrors that field for
// field and flag bit for flag (nvrhi.h:1752-1783). The conversion below is field by field, so that
// the reference can become this module's BLAS handle, but the sizes and the flag bits still have to
// agree for the two records to describe the same instance.
static_assert(sizeof(VkAccelerationStructureInstanceKHR) == 64, "The engine's instance records must stay 64 bytes");
static_assert(sizeof(nvrhi::rt::InstanceDesc) == 64, "rt::InstanceDesc is the 64-byte Vulkan instance");
static_assert(sizeof(VkTransformMatrixKHR) == sizeof(nvrhi::rt::AffineTransform),
              "The engine's VkTransformMatrixKHR and rt::AffineTransform are both 12 row-major floats");
static_assert(static_cast<uint32_t>(nvrhi::rt::InstanceFlags::TriangleCullDisable) ==
                  static_cast<uint32_t>(VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR),
              "InstanceFlags and VkGeometryInstanceFlagsKHR must agree");
static_assert(static_cast<uint32_t>(nvrhi::rt::InstanceFlags::TriangleFrontCounterclockwise) ==
                  static_cast<uint32_t>(VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR),
              "InstanceFlags and VkGeometryInstanceFlagsKHR must agree");
static_assert(static_cast<uint32_t>(nvrhi::rt::InstanceFlags::ForceOpaque) ==
                  static_cast<uint32_t>(VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR),
              "InstanceFlags and VkGeometryInstanceFlagsKHR must agree");
static_assert(static_cast<uint32_t>(nvrhi::rt::InstanceFlags::ForceNonOpaque) ==
                  static_cast<uint32_t>(VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR),
              "InstanceFlags and VkGeometryInstanceFlagsKHR must agree");

// The element strides the vertex-data set binds with (its StructuredBuffer_SRV items): the sizes the
// RT shaders' std430 types have, so a change to either struct has to change the bindings as well.
static_assert(sizeof(ShVertex) == 80, "ShVertex must stay the 80-byte vertex the vertex-data set binds");
static_assert(sizeof(ShGeometryInstance) == 256,
              "ShGeometryInstance must stay the 256-byte geometry record the vertex-data set binds");

namespace
{

// The engine builds every static component with PREFER_FAST_TRACE and every dynamic one with
// PREFER_FAST_BUILD: SetupBLAS/UpdateBLAS pass fastTrace = !IsFastBuild(filter) to ASBuilder, and
// IsFastBuild is true for CF_DYNAMIC only (ASManager.cpp:460, :495, :1197-1205). NVRHI re-queries the
// build sizes at build time with these same flags and requires the create-time allocation to cover
// them (vulkan-raytracing.cpp:803-815), so the flag has to be identical here and in createAccelStruct.
constexpr nvrhi::rt::AccelStructBuildFlags STATIC_BLAS_BUILD_FLAGS =
    nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
constexpr nvrhi::rt::AccelStructBuildFlags DYNAMIC_BLAS_BUILD_FLAGS =
    nvrhi::rt::AccelStructBuildFlags::PreferFastBuild;
constexpr nvrhi::rt::AccelStructBuildFlags TOP_LEVEL_BUILD_FLAGS =
    nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;

// The module's TLAS bound: the engine's filter grid holds one static component and one dynamic
// filter per (change frequency, pass through, primary visibility) combination, at most one instance
// each (ASManager.cpp:55-68, VertexCollectorFilterType.cpp:29-33 keeps the grid at
// MAX_TOP_LEVEL_INSTANCE_COUNT). The engine's per-instance uniform arrays index by the same order, so
// never exceeding this keeps the two indexings aligned.
constexpr uint32_t MAX_TLAS_INSTANCES = MAX_TOP_LEVEL_INSTANCE_COUNT;

// The descriptor of a native wrap of an engine buffer that an acceleration-structure build reads.
//
//  - isAccelStructBuildInput is what the validation device checks on every vertex, index or AABB
//    buffer of a BLAS build (validation-commandlist.cpp:1342, :1451, :1487); without it the NVRHI
//    validation wrapper would reject the build. The Vulkan backend turns the flag into
//    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR only for buffers it creates
//    itself (vulkan-buffer.cpp:72-73); on a native wrap it is bookkeeping, and the engine's own
//    buffers already carry the usage bit (VertexCollector.cpp:66-84).
//  - The engine writes these buffers with its own Vulkan commands, so NVRHI cannot have seen the
//    writes and would report an unknown prior state on the first use without a claim
//    (state-tracking.cpp:290-297). Claiming AccelStructBuildInput - the state every build requires
//    (vulkan-raytracing.cpp:709-711, :1054) - with keepInitialState makes the first use
//    transition-free and keeps the claim for every later list.
//  - A buffer starts from initialState even on the very first command list, because
//    getBufferStateTracking reads it unconditionally under keepInitialState
//    (state-tracking.cpp:447-450), unlike a texture, which starts from Common and needs a
//    beginTrackingTextureState call (RhiTextureSource.h:30-44). No such call is needed here.
nvrhi::BufferDesc MakeASInputBufferDesc(uint64_t byteSize, std::string debugName)
{
    nvrhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.isAccelStructBuildInput = true;
    desc.initialState = nvrhi::ResourceStates::AccelStructBuildInput;
    desc.keepInitialState = true;
    desc.debugName = std::move(debugName);
    return desc;
}

// The descriptor of a native wrap of an engine *staging* buffer that the RHI list only copies from.
//
// The engine fills these buffers with plain host stores (persistently mapped, host coherent,
// VertexCollector.cpp:112-150), so NVRHI has never seen a write and, without a claim, would report an
// unknown prior state on the first use (state-tracking.cpp:290-297). CopySource is the state
// copyBuffer requires of the source (vulkan-buffer.cpp:249); claiming it with keepInitialState keeps
// the copy transition-free on every list, which is correct because the engine never writes the
// buffer from the GPU.
nvrhi::BufferDesc MakeCopySourceBufferDesc(uint64_t byteSize, std::string debugName)
{
    nvrhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.initialState = nvrhi::ResourceStates::CopySource;
    desc.keepInitialState = true;
    desc.debugName = std::move(debugName);
    return desc;
}

// The descriptor of a native wrap of an engine device-local buffer that an RT shader reads as a
// structured buffer (the static vertex and index buffers of the vertex-data set, set 3).
//
//  - structStride is what a StructuredBuffer binding requires: the validation device rejects a
//    buffer whose desc has none (validation-device.cpp:1693-1699) and the binding-set creation
//    asserts the same (vulkan-resource-bindings.cpp:535-536). The value is the shader's element
//    stride, not the wrapped buffer's size, and on a native wrap it is bookkeeping only.
//  - NonPixelShaderResource is the state the engine's own barriers leave these buffers in after the
//    level load (CopyFromStaging's barrier declares SHADER_READ | SHADER_WRITE for the vertex data,
//    VertexCollector.cpp:674-675) and the state a ray-tracing-visible binding layout requires of a
//    shader-resource item (state-tracking.cpp:455-464 returns NonPixelShaderResource when the layout
//    has no pixel visibility, which the RT sets use). Claiming it with keepInitialState makes the
//    first bind transition-free and keeps the claim for every later list. The engine's writes are
//    invisible to NVRHI, so without a claim the first use would report an unknown prior state
//    (state-tracking.cpp:290-297).
nvrhi::BufferDesc MakeVertexDataBufferDesc(uint64_t byteSize, uint32_t structStride,
                                           bool isVertexBuffer, bool isIndexBuffer,
                                           std::string debugName)
{
    nvrhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.structStride = structStride;

    // The flags serve the raster consumer of the same static buffers: the RHI shadow-map pass binds
    // them as vertex/index buffers and NVRHI's binding checks read the flags (the Vulkan usage bits
    // come from the engine's own creation, VertexCollector.cpp - a native wrap cannot add bits).
    desc.isVertexBuffer = isVertexBuffer;
    desc.isIndexBuffer = isIndexBuffer;

    desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
    desc.keepInitialState = true;
    desc.debugName = std::move(debugName);
    return desc;
}

// The debug name of one of this module's BLAS: the engine's name for the change-frequency /
// pass-through pair ("BLAS static opaque", VertexCollectorFilterType.cpp:177-205) plus the filter
// grid index, which disambiguates the five primary-visibility variants that share it (GetID is the
// 0..44 grid index the engine keys its dynamic build hashes by, ASManager.cpp:637).
std::string MakeBlasDebugName(VertexCollectorFilterTypeFlags filter)
{
    const char *filterName = VertexCollectorFilterTypeFlags_GetNameForBLAS(filter);

    return std::string("RHI ") + (filterName != nullptr ? filterName : "BLAS") +
           " (filter " + std::to_string(VertexCollectorFilterTypeFlags_GetID(filter)) + ")";
}

// One entry of the engine's per-surface geometry list as an NVRHI geometry descriptor.
//
// The engine's descriptors store absolute device addresses into the collector's device-local
// buffers; NVRHI takes a handle plus an offset (getBufferAddress adds the offset to the wrapped
// buffer's own device address, vulkan-raytracing.cpp:40-47), so the offsets below are the same
// subtractions GetGeometryDrawInfos performs (VertexCollector.cpp:832-835, :861-866). For a dynamic
// build the handle is this module's copy buffer instead of the engine's device-local buffer, but the
// offsets stay the same: the copy starts at 0 with the same layout, and the addresses used for the
// subtraction are the engine's device-local ones.
//
// 'transforms' is the collector's CPU-side staging array of one row-major 3x4 matrix per geometry;
// the engine's transformData address selects the matrix, exactly as GetGeometryDrawInfos does
// (VertexCollector.cpp:873-885). NVRHI copies the 12 floats into its own upload buffer at build time
// and gives the build that device address (vulkan-raytracing.cpp:120-146), so the engine's device
// transforms buffer is never referenced by NVRHI.
nvrhi::rt::GeometryDesc MakeGeometryDesc(const VkAccelerationStructureGeometryKHR &src,
                                         const VkAccelerationStructureBuildRangeInfoKHR &range,
                                         VkDeviceAddress vertexBufferAddress,
                                         VkDeviceAddress indexBufferAddress,
                                         VkDeviceAddress transformsBufferAddress,
                                         const VkTransformMatrixKHR *transforms,
                                         nvrhi::IBuffer *vertexBuffer,
                                         nvrhi::IBuffer *indexBuffer)
{
    const VkAccelerationStructureGeometryTrianglesDataKHR &srcTriangles = src.geometry.triangles;

    nvrhi::rt::GeometryTriangles triangles;
    triangles.vertexBuffer = vertexBuffer;
    triangles.vertexOffset = srcTriangles.vertexData.deviceAddress - vertexBufferAddress;
    // The engine's vertex format is R32G32B32_SFLOAT with the position as the first member of the
    // 80-byte ShVertex (VertexCollector.cpp:121-123, :285-288).
    triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
    triangles.vertexStride = srcTriangles.vertexStride;
    // The engine stores the vertex count in maxVertex (VertexCollector.cpp:286); NVRHI turns its
    // vertexCount into maxVertex = count - 1 (vulkan-raytracing.cpp:118), the highest index the
    // data can reference.
    triangles.vertexCount = srcTriangles.maxVertex;

    if (srcTriangles.indexType == VK_INDEX_TYPE_UINT32)
    {
        triangles.indexBuffer = indexBuffer;
        triangles.indexOffset = srcTriangles.indexData.deviceAddress - indexBufferAddress;
        triangles.indexFormat = nvrhi::Format::R32_UINT;
        // NVRHI derives its primitive count as indexCount / 3 (vulkan-raytracing.cpp:177-179) and
        // the engine's range holds its own count (VertexCollector.cpp:226, :311), so this
        // reproduces it exactly.
        triangles.indexCount = range.primitiveCount * 3;
    }
    else
    {
        // Non-indexed geometry: both sides take vertexCount / 3 primitives.
        triangles.indexFormat = nvrhi::Format::UNKNOWN;
    }

    nvrhi::rt::GeometryDesc result;
    result.setTriangles(triangles);

    if (transforms != nullptr && transformsBufferAddress != 0 &&
        srcTriangles.transformData.deviceAddress >= transformsBufferAddress)
    {
        const uint32_t transformIndex =
            uint32_t((srcTriangles.transformData.deviceAddress - transformsBufferAddress) /
                     sizeof(VkTransformMatrixKHR));

        nvrhi::rt::AffineTransform transform;
        memcpy(transform, &transforms[transformIndex], sizeof(transform));
        result.setTransform(transform);
    }

    // The engine marks fully opaque geometry OPAQUE and everything else NO_DUPLICATE_ANY_HIT
    // (VertexCollector.cpp:280-281).
    result.setFlags((src.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0
                        ? nvrhi::rt::GeometryFlags::Opaque
                        : nvrhi::rt::GeometryFlags::NoDuplicateAnyHitInvocation);

    return result;
}

// One record filled by ASManager::GetTLASInstanceForFilter as this module's instance: every field is
// translated, and the reference - which the engine's helper deliberately leaves alone, because a
// bare filter carries no BLAS - becomes this module's handle (the one thing the engine cannot
// supply). Field-by-field, not memcpy, so the reference is setBLAS and the flag bits are the
// explicitly asserted ones.
nvrhi::rt::InstanceDesc MakeInstanceDesc(const VkAccelerationStructureInstanceKHR &src,
                                         nvrhi::rt::IAccelStruct *blas)
{
    nvrhi::rt::InstanceDesc result;

    nvrhi::rt::AffineTransform transform;
    memcpy(transform, &src.transform, sizeof(transform));
    result.setTransform(transform);

    result.setInstanceID(src.instanceCustomIndex);
    result.setInstanceMask(src.mask);
    result.setInstanceContributionToHitGroupIndex(src.instanceShaderBindingTableRecordOffset);
    result.setFlags(static_cast<nvrhi::rt::InstanceFlags>(src.flags));
    result.setBLAS(blas);

    return result;
}

}

namespace vkpt::rhi
{

RhiAccelStructs::RhiAccelStructs() = default;

RhiAccelStructs::~RhiAccelStructs()
{
    if (device != nullptr)
    {
        // The structures and the wraps may still be referenced by a recorded, not yet finished
        // submission, and the engine's own fences prove nothing about the NVRHI queue
        // (RhiFrameContext.h). Teardown therefore waits for the device first - exactly as the frame
        // skeleton's destructor does (NvrhiFrameSkeleton.cpp:138-144) - and then drops the handles
        // directly, because the retire queue may be gone by now.
        device->waitForIdle();
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        topLevel[i] = nullptr;

        for (DynamicBlas &blas : dynamicBlas[i])
        {
            blas.handle = nullptr;
        }
        dynamicBlas[i].clear();

        dynamicCopies[i].vertex = nullptr;
        dynamicCopies[i].index = nullptr;
        dynamicCopies[i].vertexCapacity = 0;
        dynamicCopies[i].indexCapacity = 0;
        dynamicCopies[i].vertexBytes = 0;
        dynamicCopies[i].indexBytes = 0;

        stagingVertexBuffer[i] = nullptr;
        stagingIndexBuffer[i] = nullptr;

        geometryStagingBuffer[i] = nullptr;
        vertexDataCopies[i].geometryInstances = nullptr;
        vertexDataCopies[i].geometryInstancesCapacity = 0;
        vertexDataCopies[i].matchPrev = nullptr;
        vertexDataCopies[i].matchPrevCapacity = 0;
    }

    staticBlas.clear();

    vertexBuffer = nullptr;
    indexBuffer = nullptr;
    staticVertexDataBuffer = nullptr;
    staticIndexDataBuffer = nullptr;
    previousVertexBuffer = nullptr;
    previousVertexCapacity = 0;
    previousIndexBuffer = nullptr;
    previousIndexCapacity = 0;

    device = nullptr;
    frameContext = nullptr;
    asManager = nullptr;
}

bool RhiAccelStructs::Create(nvrhi::IDevice *pDevice,
                             RhiFrameContext *pFrameContext,
                             ASManager *pAsManager,
                             PrintFunction pfnPrint)
{
    if (device != nullptr)
    {
        return true;
    }

    print = std::move(pfnPrint);

    if (pDevice == nullptr || pFrameContext == nullptr || pAsManager == nullptr)
    {
        print("Warning: RHI: the acceleration structures need the RHI device, the frame context and "
              "the engine's AS manager");
        return false;
    }

    const std::shared_ptr<VertexCollector> &staticCollector = pAsManager->GetStaticCollector();
    if (staticCollector == nullptr)
    {
        print("Warning: RHI: the engine's static vertex collector is unavailable, no acceleration "
              "structures are created");
        return false;
    }

    // The wraps reference engine buffers that belong to the collectors and stay alive for the run
    // (the collectors never re-create their buffers; Reset only clears the collected geometry), so
    // one wrap per buffer is enough.
    nvrhi::BufferHandle vertex = pDevice->createHandleForNativeBuffer(
        nvrhi::ObjectTypes::VK_Buffer,
        nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staticCollector->GetVertexBuffer()))),
        MakeASInputBufferDesc(staticCollector->GetVertexBufferSize(), "RHI static vertices (AS input)"));

    nvrhi::BufferHandle index = pDevice->createHandleForNativeBuffer(
        nvrhi::ObjectTypes::VK_Buffer,
        nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staticCollector->GetIndexBuffer()))),
        MakeASInputBufferDesc(staticCollector->GetIndexBufferSize(), "RHI static indices (AS input)"));

    if (vertex == nullptr || index == nullptr)
    {
        print("Warning: RHI: failed to wrap the engine's static geometry buffers, no acceleration "
              "structures are created");
        return false;
    }

    // The per-slot dynamic staging buffers are the copy sources of the per-frame transfer
    // (MakeCopySourceBufferDesc above holds the state rationale); their sizes equal the device-local
    // sizes (VertexCollector.cpp:121-136), which the wrap needs for its bookkeeping desc. Slot 0
    // owns the device-local buffers, the other slots have their own staging (VertexCollector.cpp:
    // 91-110).
    nvrhi::BufferHandle stagingVertex[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BufferHandle stagingIndex[MAX_FRAMES_IN_FLIGHT];

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        const std::shared_ptr<VertexCollector> &dynamicCollector = pAsManager->GetDynamicCollector(i);
        if (dynamicCollector == nullptr)
        {
            print("Warning: RHI: the engine's dynamic vertex collector is unavailable, no "
                  "acceleration structures are created");
            return false;
        }

        const VkBuffer engineStagingVertex = dynamicCollector->GetStagingVertexBuffer();
        const VkBuffer engineStagingIndex = dynamicCollector->GetStagingIndexBuffer();
        if (engineStagingVertex == VK_NULL_HANDLE || engineStagingIndex == VK_NULL_HANDLE)
        {
            print("Warning: RHI: the engine's dynamic staging buffers are unavailable, no "
                  "acceleration structures are created");
            return false;
        }

        stagingVertex[i] = pDevice->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(engineStagingVertex))),
            MakeCopySourceBufferDesc(dynamicCollector->GetVertexBufferSize(),
                                     "RHI dynamic vertices slot " + std::to_string(i) + " (copy source)"));

        stagingIndex[i] = pDevice->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(engineStagingIndex))),
            MakeCopySourceBufferDesc(dynamicCollector->GetIndexBufferSize(),
                                     "RHI dynamic indices slot " + std::to_string(i) + " (copy source)"));

        if (stagingVertex[i] == nullptr || stagingIndex[i] == nullptr)
        {
            print("Warning: RHI: failed to wrap the engine's dynamic staging buffers, no "
                  "acceleration structures are created");
            return false;
        }
    }

    device = pDevice;
    frameContext = pFrameContext;
    asManager = pAsManager;

    // The vertex-data set (RT set 3) the traced passes read. The static buffers are wrapped a second
    // time, with the stride the StructuredBuffer bindings need and the state the engine's own level
    // load leaves them in (MakeVertexDataBufferDesc); the AS-input wraps above are deliberately left
    // without a stride, so the two uses never share a NVRHI tracking entry. The geometry manager's
    // per-slot staging buffers become the copy sources of the per-frame geometry-record copies
    // (MakeCopySourceBufferDesc holds the state rationale: the engine writes them with host stores).
    // A missing engine object or a failed wrap does not disable the AS stream - it only makes the
    // vertex-data getters answer with nulls, and the caller then keeps the set unbound.
    nvrhi::BufferHandle staticVertexData;
    nvrhi::BufferHandle staticIndexData;
    nvrhi::BufferHandle geometryStaging[MAX_FRAMES_IN_FLIGHT];

    const std::shared_ptr<GeomInfoManager> &geomInfoMgr = pAsManager->GetGeomInfoManager();
    if (geomInfoMgr == nullptr)
    {
        print("Warning: RHI: the engine's geometry-instance manager is unavailable, the RT "
              "vertex-data set is skipped");
        vertexDataCreationFailed = true;
    }

    if (!vertexDataCreationFailed)
    {
        staticVertexData = pDevice->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staticCollector->GetVertexBuffer()))),
            MakeVertexDataBufferDesc(staticCollector->GetVertexBufferSize(), sizeof(ShVertex),
                                     true, false,
                                     "RHI static vertices (vertex data)"));

        staticIndexData = pDevice->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staticCollector->GetIndexBuffer()))),
            MakeVertexDataBufferDesc(staticCollector->GetIndexBufferSize(), sizeof(uint32_t),
                                     false, true,
                                     "RHI static indices (vertex data)"));

        if (staticVertexData == nullptr || staticIndexData == nullptr)
        {
            print("Warning: RHI: failed to wrap the engine's static geometry buffers for the "
                  "vertex-data set");
            vertexDataCreationFailed = true;
        }
    }

    // Slot 0 and the other slots share one geometry buffer, but each slot has its own staging buffer
    // (AutoBuffer.cpp:44-55 creates one per frame), and the RHI copies read the slot's own (dynamic
    // records are written to the current slot's staging only, WriteGeomInfo's frameBegin/frameEnd).
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT && !vertexDataCreationFailed; i++)
    {
        const VkBuffer engineStaging = geomInfoMgr->GetStagingBuffer(i);
        if (engineStaging == VK_NULL_HANDLE)
        {
            print("Warning: RHI: the engine's geometry staging buffer is unavailable, the RT "
                  "vertex-data set is skipped");
            vertexDataCreationFailed = true;
            break;
        }

        geometryStaging[i] = pDevice->createHandleForNativeBuffer(
            nvrhi::ObjectTypes::VK_Buffer,
            nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(engineStaging))),
            MakeCopySourceBufferDesc(geomInfoMgr->GetBufferSize(),
                                     "RHI geometry records staging slot " + std::to_string(i) + " (copy source)"));

        if (geometryStaging[i] == nullptr)
        {
            print("Warning: RHI: failed to wrap the engine's geometry staging buffer, the RT "
                  "vertex-data set is skipped");
            vertexDataCreationFailed = true;
        }
    }

    // Every RHI-side vertex-data copy is created here, with a minimal size (the match table with its
    // full one), so that GetVertexDataBuffers answers with eight non-null handles from the first
    // frame on and the caller never has to track which of them has been needed before; the first
    // frame that brings more grows them under the doubling policy above. The contents are undefined
    // until the frame that copies or writes them, and no instance references an undefined range.
    if (!vertexDataCreationFailed)
    {
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT && !vertexDataCreationFailed; i++)
        {
            const std::shared_ptr<VertexCollector> &dynamicCollector = pAsManager->GetDynamicCollector(i);

            vertexDataCreationFailed =
                !EnsureDynamicCopyBuffer(dynamicCopies[i].vertex, dynamicCopies[i].vertexCapacity,
                                         sizeof(ShVertex), uint64_t(dynamicCollector->GetVertexBufferSize()),
                                         sizeof(ShVertex), true, false, i, "vertex") ||
                !EnsureDynamicCopyBuffer(dynamicCopies[i].index, dynamicCopies[i].indexCapacity,
                                         sizeof(uint32_t), uint64_t(dynamicCollector->GetIndexBufferSize()),
                                         sizeof(uint32_t), false, true, i, "index") ||
                !EnsureVertexDataCopyBuffer(vertexDataCopies[i].geometryInstances,
                                            vertexDataCopies[i].geometryInstancesCapacity,
                                            sizeof(ShGeometryInstance), uint64_t(geomInfoMgr->GetBufferSize()),
                                            sizeof(ShGeometryInstance), i, "geometry records") ||
                !EnsureVertexDataCopyBuffer(vertexDataCopies[i].matchPrev, vertexDataCopies[i].matchPrevCapacity,
                                            uint64_t(geomInfoMgr->GetMatchPrevSize()),
                                            uint64_t(geomInfoMgr->GetMatchPrevSize()), sizeof(int32_t), i,
                                            "geometry match");
        }

        if (!vertexDataCreationFailed)
        {
            const std::shared_ptr<VertexCollector> &dynamicCollector = pAsManager->GetDynamicCollector(0);

            vertexDataCreationFailed =
                !EnsureVertexDataCopyBuffer(previousVertexBuffer, previousVertexCapacity, sizeof(ShVertex),
                                            uint64_t(dynamicCollector->GetVertexBufferSize()), sizeof(ShVertex), 0,
                                            "previous dynamic vertices") ||
                !EnsureVertexDataCopyBuffer(previousIndexBuffer, previousIndexCapacity, sizeof(uint32_t),
                                            uint64_t(dynamicCollector->GetIndexBufferSize()), sizeof(uint32_t), 0,
                                            "previous dynamic indices");
        }
    }

    vertexBuffer = std::move(vertex);
    indexBuffer = std::move(index);
    staticVertexDataBuffer = std::move(staticVertexData);
    staticIndexDataBuffer = std::move(staticIndexData);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        stagingVertexBuffer[i] = std::move(stagingVertex[i]);
        stagingIndexBuffer[i] = std::move(stagingIndex[i]);
        geometryStagingBuffer[i] = std::move(geometryStaging[i]);
    }

    // The generation the "no static set built yet" state belongs to. A later value means the engine
    // submitted a static set (see BuildStatic); aligning the counter here keeps that decision
    // correct when the first level was submitted before this object existed.
    staticGeneration = pAsManager->GetStaticGeneration();

    return true;
}

void RhiAccelStructs::RetireStaticSet()
{
    for (StaticBlas &blas : staticBlas)
    {
        if (blas.handle != nullptr)
        {
            frameContext->Retire(std::move(blas.handle));
        }
    }

    staticBlas.clear();
}

void RhiAccelStructs::BuildStatic(nvrhi::ICommandList *pCommandList)
{
    if (device == nullptr || pCommandList == nullptr || asManager == nullptr || staticCreationFailed)
    {
        return;
    }

    // ASManager::SubmitStaticGeometry destroys every static component and rebuilds the non-empty
    // ones (and it runs for an empty new level too), so every handle and address this module
    // mirrored belongs to the previous level once the counter moved. Retire the whole set and fall
    // through to the rebuild; the vkDeviceWaitIdle inside SubmitStaticGeometry already drained the
    // queue, and Retire keeps the ordering contract with the frame context anyway.
    const uint32_t generation = asManager->GetStaticGeneration();
    if (generation != staticGeneration)
    {
        staticGeneration = generation;
        RetireStaticSet();
        staticBuilt = false;
        staticGeometryCount = 0;
        staticVertexCount = 0;
        staticPrimitiveCount = 0;
    }

    if (staticBuilt)
    {
        return;
    }

    const std::shared_ptr<VertexCollector> &collector = asManager->GetStaticCollector();
    if (collector == nullptr)
    {
        return;
    }

    const VkDeviceAddress vertexBufferAddress = collector->GetVertexBufferAddress();
    const VkDeviceAddress indexBufferAddress = collector->GetIndexBufferAddress();
    const VkDeviceAddress transformsBufferAddress = collector->GetTransformsBufferAddress();
    const VkTransformMatrixKHR *transforms = collector->GetTransformsStaging();

    if (vertexBufferAddress == 0 || indexBufferAddress == 0 || transformsBufferAddress == 0 ||
        transforms == nullptr)
    {
        // The engine's buffers carry SHADER_DEVICE_ADDRESS by construction (VertexCollector.cpp:
        // 66-84), so this is a configuration error, not a state that fixes itself.
        print("Warning: RHI: the engine's static geometry buffers have no device address, no static "
              "BLAS is built");
        staticCreationFailed = true;
        return;
    }

    // Gather first, create second: the create-time descriptors of an NVRHI acceleration structure
    // decide the size of its allocation (vulkan-raytracing.cpp:376-386) and the build re-queries the
    // size from the descriptors it is given (:803-815), so both calls have to describe the same
    // geometry. Collecting a frame's geometry once and using that one list for both is what makes
    // the sizes match by construction.
    std::vector<StaticBlas> candidates;
    uint32_t geometryCount = 0;
    uint64_t vertexCount = 0;
    uint64_t primitiveCount = 0;

    for (const std::unique_ptr<BLASComponent> &component : asManager->GetStaticBlasComponents())
    {
        if (component == nullptr)
        {
            continue;
        }

        const VertexCollectorFilterTypeFlags filter = component->GetFilter();

        // The manager's static array holds static filters only (ASManager.cpp:55-68), but the check
        // keeps the intent explicit should the set ever be shared with the dynamic one.
        if (filter & VertexCollectorFilterTypeFlagBits::CF_DYNAMIC)
        {
            continue;
        }

        const std::vector<VkAccelerationStructureGeometryKHR> &geoms = collector->GetASGeometries(filter);
        const std::vector<VkAccelerationStructureBuildRangeInfoKHR> &ranges =
            collector->GetASBuildRangeInfos(filter);

        // An empty component is not built by the engine either (SetupBLAS returns early,
        // ASManager.cpp:450-455), and a structure without geometries cannot be created. The two
        // engine arrays are written together per surface (VertexCollector.cpp:307-315), so a
        // mismatch would be an engine bug, not a state to skip silently.
        assert(geoms.size() == ranges.size());
        if (geoms.empty() || geoms.size() != ranges.size())
        {
            continue;
        }

        StaticBlas blas;
        blas.filter = filter;
        blas.debugName = MakeBlasDebugName(filter);
        blas.geometries.reserve(geoms.size());

        for (size_t i = 0; i < geoms.size(); i++)
        {
            blas.geometries.push_back(MakeGeometryDesc(geoms[i], ranges[i], vertexBufferAddress,
                                                       indexBufferAddress, transformsBufferAddress,
                                                       transforms, vertexBuffer, indexBuffer));
            geometryCount++;
            vertexCount += geoms[i].geometry.triangles.maxVertex;
            primitiveCount += ranges[i].primitiveCount;
        }

        candidates.push_back(std::move(blas));
    }

    if (candidates.empty())
    {
        // No decision: static geometry arrives with a level load, so a later frame may still bring
        // it and the build stays pending.
        return;
    }

    // One structure per non-empty static component, created and recorded in the same call, so that
    // the first frame that follows a level load is also the frame that builds.
    for (StaticBlas &candidate : candidates)
    {
        nvrhi::rt::AccelStructDesc desc;
        desc.bottomLevelGeometries = candidate.geometries;
        desc.setBuildFlags(STATIC_BLAS_BUILD_FLAGS);
        desc.setDebugName(candidate.debugName);

        nvrhi::rt::AccelStructHandle handle = device->createAccelStruct(desc);
        if (handle == nullptr)
        {
            print(("Warning: RHI: failed to create the BLAS \"" + candidate.debugName +
                   "\", the static acceleration structures are skipped").c_str());
            staticCreationFailed = true;
            return;
        }

        pCommandList->buildBottomLevelAccelStruct(handle.Get(), candidate.geometries.data(),
                                                  candidate.geometries.size(), STATIC_BLAS_BUILD_FLAGS);

        candidate.handle = std::move(handle);
        staticBlas.push_back(std::move(candidate));
    }

    staticGeometryCount = geometryCount;
    staticVertexCount = vertexCount;
    staticPrimitiveCount = primitiveCount;
    staticBuilt = true;
}

const RhiAccelStructs::StaticBlas *RhiAccelStructs::FindStaticBlas(uint32_t filter) const
{
    for (const StaticBlas &blas : staticBlas)
    {
        if (blas.filter == filter)
        {
            return &blas;
        }
    }

    return nullptr;
}

void RhiAccelStructs::AppendStaticInstances(uint32_t rayCullMaskWorld,
                                            bool allowGeometryWithSkyFlag,
                                            std::vector<nvrhi::rt::InstanceDesc> &instances)
{
    const std::shared_ptr<VertexCollector> &collector = asManager->GetStaticCollector();
    if (collector == nullptr)
    {
        return;
    }

    // GetStaticBlasComponents() enumerates the static components in construction order, which is the
    // filter grid order (ASManager.cpp:55-68 iterates VertexCollectorFilterTypeFlags_IterateOverFlags)
    // - the same order the engine's own prepared instance list has, and the order this module built
    // staticBlas in. Iterating the components (rather than staticBlas) keeps that order authoritative
    // and makes a non-empty component without a handle visible.
    for (const std::unique_ptr<BLASComponent> &component : asManager->GetStaticBlasComponents())
    {
        if (component == nullptr)
        {
            continue;
        }

        const VertexCollectorFilterTypeFlags filter = component->GetFilter();
        if (filter & VertexCollectorFilterTypeFlagBits::CF_DYNAMIC)
        {
            continue;
        }

        const StaticBlas *blas = FindStaticBlas(filter);
        if (blas == nullptr || blas->handle == nullptr)
        {
            // Only a create failure (already reported) or a frame that skipped BuildStatic can get
            // here; an empty component is not an error and is simply not an instance.
            if (!collector->GetASGeometries(filter).empty())
            {
                WarnUnresolvedFilter(filter);
            }
            continue;
        }

        // The engine's own attribute rules - the mask against the cull mask, the first-person /
        // viewer / sky custom-index bits, the refract rewrite, the alpha-tested SBT offset and the
        // instance flags (ASManager.cpp:877-995). The reference is this module's BLAS.
        VkAccelerationStructureInstanceKHR record = {};
        if (!ASManager::GetTLASInstanceForFilter(filter, rayCullMaskWorld, allowGeometryWithSkyFlag, record))
        {
            continue;
        }

        // The uniform's per-instance geometry information for this instance (see
        // GetInstanceGeometryInfo): the filter's global-array offset and its geometry count - the
        // values the engine's WriteInstanceGeomInfo writes for the same filter
        // (ASManager.cpp:1016-1028). One instance per static filter, so the count is the number of
        // geometries this module's BLAS was built from, which equals the component's geometry count
        // (BLASComponent::GetGeomCount, ASManager.cpp:450).
        const uint32_t instanceIndex = static_cast<uint32_t>(instances.size());
        if (instanceIndex < MAX_TLAS_INSTANCES)
        {
            instanceGeomInfo[instanceIndex].offset =
                static_cast<int32_t>(VertexCollectorFilterTypeFlags_GetOffsetInGlobalArray(filter));
            instanceGeomInfo[instanceIndex].count = static_cast<int32_t>(blas->geometries.size());
        }

        instances.push_back(MakeInstanceDesc(record, blas->handle.Get()));
    }
}

bool RhiAccelStructs::EnsureDynamicCopyBuffer(nvrhi::BufferHandle &buffer,
                                              uint64_t &capacity,
                                              uint64_t needed,
                                              uint64_t maxCapacity,
                                              uint32_t structStride,
                                              bool isVertexBuffer,
                                              bool isIndexBuffer,
                                              uint32_t frameIndex,
                                              const char *kind)
{
    if (needed == 0 || (buffer != nullptr && capacity >= needed))
    {
        return true;
    }

    // Doubling bounds the number of replacements over a run; the cap is the collector's staging
    // size, which is also the largest prefix a frame can have (the collector asserts its counters
    // against the same buffer size, VertexCollector.cpp:236-246). The replaced buffer goes through
    // the retire queue: a submission of this slot may still be reading it.
    uint64_t newCapacity = std::max(needed, capacity * 2);
    newCapacity = std::min(newCapacity, maxCapacity);

    // The copy destination: an RHI-created buffer (so it carries TRANSFER_SRC | TRANSFER_DST,
    // vulkan-buffer.cpp:48-49, and the build-input usage the flag adds, :72-73) that copyBuffer
    // writes, the BLAS build reads and the RT shaders read as a structured buffer in the same list.
    // structStride makes the same handle bindable as a StructuredBuffer_SRV of the vertex-data set
    // (validation-device.cpp:1693-1699 and vulkan-resource-bindings.cpp:535-536 both require it);
    // adding it does not change the build, which addresses the buffer by device address. CopyDest is
    // claimed as the initial state, so the copy itself is transition-free and NVRHI emits exactly the
    // barriers the build (CopyDest -> AccelStructBuildInput, vulkan-raytracing.cpp:708-711) and the
    // shader read (-> the list's shader-resource state) need. The next list starts the buffer at
    // CopyDest again; the shared queue, the per-slot buffers and the frame context's wait for the
    // slot's previous submission (RhiFrameContext.h) cover the write-after-read against that
    // submission.
    nvrhi::BufferDesc desc;
    desc.byteSize = newCapacity;
    desc.structStride = structStride;
    desc.isAccelStructBuildInput = true;

    // The raster consumer of the same copies: the RHI shadow-map pass draws the dynamic geometry
    // from them as vertex/index buffers, so the flags (and the usage bits they add on an
    // RHI-created buffer, vulkan-buffer.cpp:51-55) follow the kind of data the call site passes.
    desc.isVertexBuffer = isVertexBuffer;
    desc.isIndexBuffer = isIndexBuffer;

    desc.initialState = nvrhi::ResourceStates::CopyDest;
    desc.keepInitialState = true;
    desc.debugName = std::string("RHI dynamic ") + kind + " slot " + std::to_string(frameIndex) + " (AS input)";

    nvrhi::BufferHandle created = device->createBuffer(desc);
    if (created == nullptr)
    {
        print(("Warning: RHI: failed to create the slot " + std::to_string(frameIndex) + " dynamic " + kind +
               " copy buffer, the dynamic acceleration structures are skipped").c_str());
        return false;
    }

    if (buffer != nullptr)
    {
        frameContext->Retire(std::move(buffer));
    }

    buffer = std::move(created);
    capacity = newCapacity;
    return true;
}

bool RhiAccelStructs::EnsureVertexDataCopyBuffer(nvrhi::BufferHandle &buffer,
                                                 uint64_t &capacity,
                                                 uint64_t needed,
                                                 uint64_t maxCapacity,
                                                 uint32_t structStride,
                                                 uint32_t frameIndex,
                                                 const char *kind)
{
    if (needed == 0 || (buffer != nullptr && capacity >= needed))
    {
        return true;
    }

    if (needed > maxCapacity)
    {
        // The engine's own bound is the maximum a frame can produce; a larger request means the
        // engine's invariant broke, and a clamp here would silently drop records.
        print(("Warning: RHI: the " + std::string(kind) + " copy needs " + std::to_string(needed) +
               " bytes, more than the engine's " + std::to_string(maxCapacity) +
               " byte buffer; the RT vertex-data set is skipped").c_str());
        return false;
    }

    uint64_t newCapacity = std::max(needed, capacity * 2);
    newCapacity = std::min(newCapacity, maxCapacity);

    // A buffer only the RT shaders read: CopyDest claimed as the initial state so the frame's own
    // copy (and the writeBuffer of the match table) is transition-free, and structStride for the
    // StructuredBuffer binding. The replaced buffer is retired - a submission of this slot may still
    // be reading it - and the slot's wait in BeginSlot covers the write-after-read.
    nvrhi::BufferDesc desc;
    desc.byteSize = newCapacity;
    desc.structStride = structStride;
    desc.initialState = nvrhi::ResourceStates::CopyDest;
    desc.keepInitialState = true;
    desc.debugName = std::string("RHI ") + kind + " slot " + std::to_string(frameIndex) + " (vertex data)";

    nvrhi::BufferHandle created = device->createBuffer(desc);
    if (created == nullptr)
    {
        print(("Warning: RHI: failed to create the slot " + std::to_string(frameIndex) + " " + kind +
               " copy buffer, the RT vertex-data set is skipped").c_str());
        return false;
    }

    if (buffer != nullptr)
    {
        frameContext->Retire(std::move(buffer));
    }

    buffer = std::move(created);
    capacity = newCapacity;
    return true;
}

void RhiAccelStructs::AppendDynamicSlot(nvrhi::ICommandList *pCommandList,
                                        uint32_t frameIndex,
                                        uint32_t rayCullMaskWorld,
                                        bool allowGeometryWithSkyFlag,
                                        std::vector<nvrhi::rt::InstanceDesc> &instances)
{
    // Only a filter built in this frame is active: a skipped or failed dynamic frame must produce no
    // instances, while the handles of the earlier frames stay alive for TLAS builds still in flight.
    for (DynamicBlas &blas : dynamicBlas[frameIndex])
    {
        blas.active = false;
    }

    if (dynamicCreationFailed)
    {
        return;
    }

    const std::shared_ptr<VertexCollector> &collector = asManager->GetDynamicCollector(frameIndex);
    if (collector == nullptr)
    {
        return;
    }

    DynamicCopies &copies = dynamicCopies[frameIndex];

    // The used prefixes of this frame, and what the next frame's previous-frame copies must read
    // from this slot. Reset before every early return, so a frame that fails or brings nothing does
    // not leave the previous frame's lengths behind.
    copies.vertexBytes = 0;
    copies.indexBytes = 0;

    const VkDeviceAddress vertexBufferAddress = collector->GetVertexBufferAddress();
    const VkDeviceAddress indexBufferAddress = collector->GetIndexBufferAddress();
    const VkDeviceAddress transformsBufferAddress = collector->GetTransformsBufferAddress();
    const VkTransformMatrixKHR *transforms = collector->GetTransformsStaging();

    if (vertexBufferAddress == 0 || indexBufferAddress == 0 || transformsBufferAddress == 0 ||
        transforms == nullptr)
    {
        // The engine's buffers carry SHADER_DEVICE_ADDRESS by construction, so this is a
        // configuration error, not a state that fixes itself.
        dynamicCreationFailed = true;
        print("Warning: RHI: the engine's dynamic geometry buffers have no device address, no "
              "dynamic BLAS is built");
        return;
    }

    const uint64_t vertexBytes = uint64_t(collector->GetCurrentVertexCount()) * sizeof(ShVertex);
    const uint64_t indexBytes = uint64_t(collector->GetCurrentIndexCount()) * sizeof(uint32_t);

    copies.vertexBytes = vertexBytes;
    copies.indexBytes = indexBytes;

    // The used prefix starts at 0 and is contiguous: AddGeometry aligns each geometry up from the
    // running counters starting at 0 (VertexCollector.cpp:221-232) and BeginDynamicGeometry resets
    // the slot's collector before the game fills it (ASManager.cpp:736-737), so one copy per buffer
    // describes the frame exactly.
    if (vertexBytes > 0)
    {
        if (!EnsureDynamicCopyBuffer(copies.vertex, copies.vertexCapacity, vertexBytes,
                                     uint64_t(collector->GetVertexBufferSize()), sizeof(ShVertex),
                                     true, false, frameIndex,
                                     "vertex"))
        {
            dynamicCreationFailed = true;
            return;
        }

        pCommandList->copyBuffer(copies.vertex.Get(), 0, stagingVertexBuffer[frameIndex].Get(), 0, vertexBytes);
        dynamicCopyBytes += vertexBytes;
    }

    if (indexBytes > 0)
    {
        if (!EnsureDynamicCopyBuffer(copies.index, copies.indexCapacity, indexBytes,
                                     uint64_t(collector->GetIndexBufferSize()), sizeof(uint32_t),
                                     false, true, frameIndex,
                                     "index"))
        {
            dynamicCreationFailed = true;
            return;
        }

        pCommandList->copyBuffer(copies.index.Get(), 0, stagingIndexBuffer[frameIndex].Get(), 0, indexBytes);
        dynamicCopyBytes += indexBytes;
    }

    // The public filter iteration is the authoritative grid order (VertexCollectorFilterType.cpp:
    // 45-57), which the instance ordering and the engine's per-instance uniform arrays rely on.
    VertexCollectorFilterTypeFlags_IterateOverFlags(
        [&](VertexCollectorFilterTypeFlags filter)
        {
            if (!(filter & VertexCollectorFilterTypeFlagBits::CF_DYNAMIC))
            {
                return;
            }

            const std::vector<VkAccelerationStructureGeometryKHR> &geoms = collector->GetASGeometries(filter);
            const std::vector<VkAccelerationStructureBuildRangeInfoKHR> &ranges =
                collector->GetASBuildRangeInfos(filter);

            assert(geoms.size() == ranges.size());
            if (geoms.empty() || geoms.size() != ranges.size())
            {
                return;
            }

            dynamicActiveFilterCount++;

            // The descriptors of this frame's geometry, built from this module's copy buffers. The
            // engine's addresses only serve the offset arithmetic (copy starts at 0, same layout),
            // and the transforms come from the collector's CPU staging array, which NVRHI copies into
            // its own upload buffer at build time.
            std::vector<nvrhi::rt::GeometryDesc> geometries;
            geometries.reserve(geoms.size());

            // The shape the handle has to cover. A size query is unreachable from here (no native
            // VkDevice in the pinned NVRHI, no ASManager wrapper), so the module recreates the
            // structure exactly when this changes, which keeps the create-time allocation valid for
            // every build of the handle by construction (see RhiAccelStructs.h).
            std::vector<uint32_t> shape;
            shape.reserve(geoms.size() * 5);

            for (size_t i = 0; i < geoms.size(); i++)
            {
                geometries.push_back(MakeGeometryDesc(geoms[i], ranges[i], vertexBufferAddress,
                                                      indexBufferAddress, transformsBufferAddress,
                                                      transforms, copies.vertex.Get(), copies.index.Get()));

                const VkAccelerationStructureGeometryTrianglesDataKHR &triangles = geoms[i].geometry.triangles;
                shape.push_back(ranges[i].primitiveCount);
                shape.push_back(triangles.maxVertex);
                shape.push_back(triangles.vertexStride);
                shape.push_back(triangles.indexType == VK_INDEX_TYPE_UINT32 ? 1u : 0u);
                shape.push_back(triangles.transformData.deviceAddress != 0 ? 1u : 0u);
            }

            DynamicBlas *blas = nullptr;
            for (DynamicBlas &candidate : dynamicBlas[frameIndex])
            {
                if (candidate.filter == filter)
                {
                    blas = &candidate;
                    break;
                }
            }

            if (blas == nullptr)
            {
                dynamicBlas[frameIndex].push_back(DynamicBlas{});
                blas = &dynamicBlas[frameIndex].back();
                blas->filter = filter;
                blas->debugName = MakeBlasDebugName(filter);
            }

            if (blas->handle == nullptr || blas->shape != shape)
            {
                // Create from the very list this build uses, so the allocation covers the build
                // (vulkan-raytracing.cpp:376-386 against :803-815). The replaced handle is not
                // dropped while the queue may still read it - it goes through the frame context's
                // retire queue (rt::AccelStructHandle is a ref-counted NVRHI handle, nvrhi.h:1835).
                nvrhi::rt::AccelStructDesc desc;
                desc.bottomLevelGeometries = geometries;
                desc.setBuildFlags(DYNAMIC_BLAS_BUILD_FLAGS);
                desc.setDebugName(blas->debugName);

                nvrhi::rt::AccelStructHandle handle = device->createAccelStruct(desc);
                if (handle == nullptr)
                {
                    dynamicCreationFailed = true;
                    print(("Warning: RHI: failed to create the BLAS \"" + blas->debugName +
                           "\", the dynamic acceleration structures are skipped").c_str());
                    return;
                }

                if (blas->handle != nullptr)
                {
                    frameContext->Retire(std::move(blas->handle));
                }

                blas->handle = std::move(handle);
                blas->shape = std::move(shape);
            }

            blas->geometries = std::move(geometries);
            blas->active = true;

            pCommandList->buildBottomLevelAccelStruct(blas->handle.Get(), blas->geometries.data(),
                                                      blas->geometries.size(), DYNAMIC_BLAS_BUILD_FLAGS);

            dynamicBlasCount++;
            dynamicGeometryCount += uint32_t(blas->geometries.size());
            for (size_t i = 0; i < geoms.size(); i++)
            {
                dynamicVertexCount += geoms[i].geometry.triangles.maxVertex;
                dynamicPrimitiveCount += ranges[i].primitiveCount;
            }

            // The engine's attribute rules with this module's handle, exactly as for the static set.
            VkAccelerationStructureInstanceKHR record = {};
            if (!ASManager::GetTLASInstanceForFilter(filter, rayCullMaskWorld, allowGeometryWithSkyFlag, record))
            {
                return;
            }

            // The uniform's per-instance geometry information for this instance (see
            // GetInstanceGeometryInfo): the filter's global-array offset and its geometry count -
            // the values the engine's WriteInstanceGeomInfo writes for the same filter
            // (ASManager.cpp:1016-1028).
            const uint32_t instanceIndex = static_cast<uint32_t>(instances.size());
            if (instanceIndex < MAX_TLAS_INSTANCES)
            {
                instanceGeomInfo[instanceIndex].offset =
                    static_cast<int32_t>(VertexCollectorFilterTypeFlags_GetOffsetInGlobalArray(filter));
                instanceGeomInfo[instanceIndex].count = static_cast<int32_t>(blas->geometries.size());
            }

            instances.push_back(MakeInstanceDesc(record, blas->handle.Get()));
        });
}

void RhiAccelStructs::RecordVertexDataCopies(nvrhi::ICommandList *pCommandList, uint32_t frameIndex)
{
    if (vertexDataCreationFailed || tlasInstanceCount == 0)
    {
        return;
    }

    const std::shared_ptr<GeomInfoManager> &geomInfoMgr = asManager->GetGeomInfoManager();
    if (geomInfoMgr == nullptr || geometryStagingBuffer[frameIndex] == nullptr)
    {
        return;
    }

    VertexDataCopies &copies = vertexDataCopies[frameIndex];

    // The geometry records: one copy per instance's geometry range, in the global geometry index
    // space the shaders address (instanceGeomInfoOffset[i] .. + instanceGeomInfoCount[i], times the
    // record size). The destination is sized to the highest range end, so the offsets stay absolute;
    // a range the frame does not use is not copied, which also lets a frame without an instance list
    // keep the records of the TLAS it still binds.
    uint64_t geometryInstancesNeeded = 0;
    for (uint32_t i = 0; i < tlasInstanceCount; i++)
    {
        geometryInstancesNeeded =
            std::max(geometryInstancesNeeded,
                     (uint64_t(instanceGeomInfo[i].offset) + uint64_t(instanceGeomInfo[i].count)) *
                         sizeof(ShGeometryInstance));
    }

    if (geometryInstancesNeeded > 0)
    {
        if (!EnsureVertexDataCopyBuffer(copies.geometryInstances, copies.geometryInstancesCapacity,
                                        geometryInstancesNeeded, uint64_t(geomInfoMgr->GetBufferSize()),
                                        sizeof(ShGeometryInstance), frameIndex, "geometry records"))
        {
            vertexDataCreationFailed = true;
        }
        else
        {
            for (uint32_t i = 0; i < tlasInstanceCount; i++)
            {
                const uint64_t offset = uint64_t(instanceGeomInfo[i].offset) * sizeof(ShGeometryInstance);
                const uint64_t size = uint64_t(instanceGeomInfo[i].count) * sizeof(ShGeometryInstance);

                if (size == 0)
                {
                    continue;
                }

                pCommandList->copyBuffer(copies.geometryInstances.Get(), offset,
                                         geometryStagingBuffer[frameIndex].Get(), offset, size);
                vertexDataCopyBytes += size;
            }
        }
    }

    // The previous-frame match table: its write into the staging buffer happens inside the bypassed
    // CopyFromStaging (GeomInfoManager.cpp:89-95), so the CPU shadow - the array the engine would
    // memcpy into staging - is the source. The whole table is written, as the engine copies whole
    // group ranges (GeomInfoManager.cpp:63-141).
    const int32_t *matchPrev = geomInfoMgr->GetMatchPrevData();
    const VkDeviceSize matchPrevSize = geomInfoMgr->GetMatchPrevSize();
    if (matchPrev != nullptr && matchPrevSize > 0)
    {
        if (!EnsureVertexDataCopyBuffer(copies.matchPrev, copies.matchPrevCapacity, uint64_t(matchPrevSize),
                                        uint64_t(matchPrevSize), sizeof(int32_t), frameIndex, "geometry match"))
        {
            vertexDataCreationFailed = true;
        }
        else
        {
            pCommandList->writeBuffer(copies.matchPrev.Get(), matchPrev, size_t(matchPrevSize), 0);
            vertexDataCopyBytes += matchPrevSize;
        }
    }

    // The previous-frame dynamic vertices and indices: the engine's previousDynamicPositions/Indices
    // are filled from its own device-local dynamic buffers (ASManager.cpp:1158-1190), which the
    // `rhiframe` path never writes, so the previous frame's bytes come from the other slot's dynamic
    // copy - the only source that still holds them. Needed only when the frame has a dynamic
    // instance; both are shared between the slots, like the engine's single pair.
    if (dynamicActiveFilterCount == 0)
    {
        return;
    }

    const std::shared_ptr<VertexCollector> &collector = asManager->GetDynamicCollector(frameIndex);
    if (collector == nullptr)
    {
        return;
    }

    const DynamicCopies &other = dynamicCopies[(frameIndex + 1) % MAX_FRAMES_IN_FLIGHT];

    if (other.vertex != nullptr && other.vertexBytes > 0)
    {
        if (!EnsureVertexDataCopyBuffer(previousVertexBuffer, previousVertexCapacity, other.vertexBytes,
                                        uint64_t(collector->GetVertexBufferSize()), sizeof(ShVertex), frameIndex,
                                        "previous dynamic vertices"))
        {
            vertexDataCreationFailed = true;
        }
        else
        {
            pCommandList->copyBuffer(previousVertexBuffer.Get(), 0, other.vertex.Get(), 0, other.vertexBytes);
            vertexDataCopyBytes += other.vertexBytes;
        }
    }

    if (other.index != nullptr && other.indexBytes > 0)
    {
        if (!EnsureVertexDataCopyBuffer(previousIndexBuffer, previousIndexCapacity, other.indexBytes,
                                        uint64_t(collector->GetIndexBufferSize()), sizeof(uint32_t), frameIndex,
                                        "previous dynamic indices"))
        {
            vertexDataCreationFailed = true;
        }
        else
        {
            pCommandList->copyBuffer(previousIndexBuffer.Get(), 0, other.index.Get(), 0, other.indexBytes);
            vertexDataCopyBytes += other.indexBytes;
        }
    }
}

void RhiAccelStructs::WarnUnresolvedFilter(uint32_t filter)
{
    if (warnedUnresolvedInstance || print == nullptr)
    {
        return;
    }

    warnedUnresolvedInstance = true;

    print(("Warning: RHI: " + MakeBlasDebugName(filter) +
           " has geometry but no RHI BLAS; its instances are missing from the top-level structure").c_str());
}

void RhiAccelStructs::BuildTopLevel(nvrhi::ICommandList *pCommandList,
                                    uint32_t frameIndex,
                                    uint32_t rayCullMaskWorld,
                                    bool allowGeometryWithSkyFlag,
                                    bool disableRayTracedGeometry)
{
    if (device == nullptr || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    // The summary is due on this frame or not; decided before the work so that its grace period
    // ticks on every frame the TLAS build is asked for, and printed after the work so that its
    // figures are this frame's.
    const bool summaryDue = IsSummaryDue();

    // This frame's figures, for the summary.
    dynamicBlasCount = 0;
    dynamicActiveFilterCount = 0;
    dynamicGeometryCount = 0;
    dynamicVertexCount = 0;
    dynamicPrimitiveCount = 0;
    dynamicCopyBytes = 0;
    vertexDataCopyBytes = 0;
    tlasInstanceCount = 0;

    std::vector<nvrhi::rt::InstanceDesc> instances;
    instances.reserve(MAX_TLAS_INSTANCES);

    if (!disableRayTracedGeometry)
    {
        // The engine's order: the static components first, then the slot's dynamic filters, each in
        // filter-grid order (ASManager.cpp:1045-1049 walks allStaticBlas then allDynamicBlas[slot]).
        AppendStaticInstances(rayCullMaskWorld, allowGeometryWithSkyFlag, instances);
        AppendDynamicSlot(pCommandList, frameIndex, rayCullMaskWorld, allowGeometryWithSkyFlag, instances);
    }

    if (instances.size() > MAX_TLAS_INSTANCES)
    {
        if (!warnedInstanceOverflow)
        {
            warnedInstanceOverflow = true;
            print(("Warning: RHI: the instance synthesis produced " + std::to_string(instances.size()) +
                   " instances, more than the " + std::to_string(MAX_TLAS_INSTANCES) +
                   " the engine's filter grid can contribute; the build is clamped").c_str());
        }

        instances.resize(MAX_TLAS_INSTANCES);
    }

    tlasInstanceCount = static_cast<uint32_t>(instances.size());

    // The vertex-data set's per-frame copies (set 3): the geometry records of the instances above
    // and the previous-frame dynamic data. Recorded before the TLAS build so the pass that follows
    // in the same list finds them current; a frame without an instance list copies nothing and keeps
    // the previous frame's content, which is exactly what the TLAS it still binds describes.
    RecordVertexDataCopies(pCommandList, frameIndex);

    if (tlasInstanceCount > 0)
    {
        if (topLevel[frameIndex] == nullptr || topLevelCapacity[frameIndex] < tlasInstanceCount)
        {
            // The first build creates the structure for the instance count it is actually used with;
            // a later frame that needs more grows the slot once, straight to the module's maximum.
            // Either way a replaced structure is not dropped while the queue may still read it - it
            // goes through the frame context's retire queue.
            const uint32_t capacity =
                topLevel[frameIndex] == nullptr ? tlasInstanceCount : MAX_TLAS_INSTANCES;

            nvrhi::rt::AccelStructDesc desc;
            desc.setTopLevelMaxInstances(capacity);
            desc.setBuildFlags(TOP_LEVEL_BUILD_FLAGS);
            desc.setDebugName("RHI TLAS slot " + std::to_string(frameIndex));

            nvrhi::rt::AccelStructHandle handle = device->createAccelStruct(desc);
            if (handle == nullptr)
            {
                // Keep the previous structure (if any) and skip the build; the frame still reaches
                // the summary below.
                print("Warning: RHI: failed to create the top-level acceleration structure");
            }
            else
            {
                if (topLevel[frameIndex] != nullptr)
                {
                    frameContext->Retire(std::move(topLevel[frameIndex]));
                }

                topLevel[frameIndex] = std::move(handle);
                topLevelCapacity[frameIndex] = capacity;
            }
        }

        if (topLevel[frameIndex] != nullptr)
        {
            // The instances were synthesized on the CPU this frame (their references are this
            // module's BLAS handles), so NVRHI copies the array into its own upload buffer and the
            // TLAS of frame N describes frame N - no engine instance buffer, no one-frame lag, no
            // warm-up.
            pCommandList->buildTopLevelAccelStruct(topLevel[frameIndex].Get(), instances.data(),
                                                   instances.size(), TOP_LEVEL_BUILD_FLAGS);
        }
    }

    if (summaryDue)
    {
        PrintSummary();
    }
}

nvrhi::rt::IAccelStruct *RhiAccelStructs::GetTopLevel(uint32_t frameIndex) const
{
    if (frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    return topLevel[frameIndex].Get();
}

RhiAccelStructs::VertexDataBuffers RhiAccelStructs::GetVertexDataBuffers(uint32_t frameIndex) const
{
    VertexDataBuffers result;

    if (frameIndex >= MAX_FRAMES_IN_FLIGHT || vertexDataCreationFailed)
    {
        return result;
    }

    const DynamicCopies &dynamic = dynamicCopies[frameIndex];
    const VertexDataCopies &copies = vertexDataCopies[frameIndex];

    result.staticVertices = staticVertexDataBuffer.Get();
    result.dynamicVertices = dynamic.vertex.Get();
    result.staticIndices = staticIndexDataBuffer.Get();
    result.dynamicIndices = dynamic.index.Get();
    result.geometryInstances = copies.geometryInstances.Get();
    result.geometryInstancesMatchPrev = copies.matchPrev.Get();
    result.previousDynamicVertices = previousVertexBuffer.Get();
    result.previousDynamicIndices = previousIndexBuffer.Get();

    return result;
}

uint32_t RhiAccelStructs::GetInstanceGeometryInfo(int32_t *pInstanceGeomInfoOffset,
                                                  int32_t *pInstanceGeomInfoCount) const
{
    if (pInstanceGeomInfoOffset == nullptr || pInstanceGeomInfoCount == nullptr)
    {
        return 0;
    }

    for (uint32_t i = 0; i < tlasInstanceCount; i++)
    {
        pInstanceGeomInfoOffset[i] = instanceGeomInfo[i].offset;
        pInstanceGeomInfoCount[i] = instanceGeomInfo[i].count;
    }

    return tlasInstanceCount;
}

bool RhiAccelStructs::IsSummaryDue()
{
    if (summaryPrinted || print == nullptr)
    {
        return false;
    }

    // The line is the gate's evidence, so it must not claim a "0 static BLAS" state that a level
    // load is about to change: it waits for the static decision, with a grace period that covers a
    // scene that never brings static geometry.
    if (!staticBuilt && !staticCreationFailed)
    {
        if (summaryFrameGrace > 0)
        {
            summaryFrameGrace--;
            return false;
        }
    }

    summaryPrinted = true;
    return true;
}

void RhiAccelStructs::PrintSummary()
{
    if (print == nullptr)
    {
        return;
    }

    const uint32_t componentCount = asManager != nullptr
                                        ? static_cast<uint32_t>(asManager->GetStaticBlasComponents().size())
                                        : 0;

    char buffer[512];
    snprintf(buffer, sizeof(buffer) / sizeof(buffer[0]),
             "RHI: acceleration structures: %u/%u static BLAS, %u geometries, %llu triangles, "
             "%llu vertices; %u/%u dynamic BLAS, %u geometries, %llu triangles, %llu vertices; "
             "%llu KiB dynamic copies and %llu KiB vertex-data copies this frame; TLAS instances %u\n",
             static_cast<uint32_t>(staticBlas.size()), componentCount, staticGeometryCount,
             static_cast<unsigned long long>(staticPrimitiveCount),
             static_cast<unsigned long long>(staticVertexCount),
             dynamicBlasCount, dynamicActiveFilterCount, dynamicGeometryCount,
             static_cast<unsigned long long>(dynamicPrimitiveCount),
             static_cast<unsigned long long>(dynamicVertexCount),
             static_cast<unsigned long long>(dynamicCopyBytes / 1024),
             static_cast<unsigned long long>(vertexDataCopyBytes / 1024),
             tlasInstanceCount);
    print(buffer);
}

}
