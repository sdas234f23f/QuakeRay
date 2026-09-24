#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nvrhi/nvrhi.h>

#include "../Common.h"

namespace vkpt
{
class ASManager;
}

namespace vkpt::rhi
{

class RhiFrameContext;

// The RHI copy of the engine's acceleration structures, on the RHI side of the `rhiframe` split: one
// NVRHI bottom-level structure (BLAS) per non-empty geometry component of the engine's ASManager and
// one top-level structure (TLAS) per frame slot that references **these** structures.
//
// Coexistence with the legacy AS path - this module removes or changes nothing:
//  - Under `rhiframe` the game still calls its level-load entry points, so ASManager builds its own
//    static BLAS synchronously (ASManager::SubmitStaticGeometry, which waits for its fence) and its
//    own TLAS preparation/build keep running on the legacy command buffer. This module adds a second,
//    parallel set of NVRHI-owned structures over the same engine geometry; GetTopLevel() is what the
//    debug-trace pass binds.
//  - The engine's dynamic components stay stale under `rhiframe` (their geomCount/AS are only
//    written by the legacy SubmitDynamicGeometry, which the RHI path bypasses), which is exactly why
//    the instance records are synthesized here from the collectors and the engine's public attribute
//    rules instead of being translated from the engine's prepared list.
//
// Where the data comes from:
//  - Static BLAS: the geometry descriptors are translated from the static collector's own
//    VkAccelerationStructureGeometryKHR entries (VertexCollector::GetASGeometries): triangles,
//    position at offset 0 with an 80-byte stride (ShVertex), R32_UINT indices, one NVRHI geometry
//    entry per surface, and the surface's row-major 12-float transform (GeometryDesc::setTransform)
//    read from the collector's staging transform array - the same source GetGeometryDrawInfos uses.
//    NVRHI copies that transform into its own upload buffer at build time, so the engine's
//    transforms buffer is never referenced by NVRHI. The addresses in those engine descriptors are
//    absolute device addresses; NVRHI wants offsets into a buffer it has a handle for, which is what
//    the wrap of the collector's vertex and index buffers and the subtraction in the .cpp are for.
//  - Static level changes: ASManager::SubmitStaticGeometry destroys every static component and
//    rebuilds the non-empty ones (and also runs when the new level is empty), so every handle and
//    address this module mirrored is from the previous level. The module samples
//    ASManager::GetStaticGeneration() on every BuildStatic and, when it changed, retires the whole
//    static set and rebuilds from the new components in the same call.
//  - Dynamic BLAS: the dynamic collector's *staging* buffers are what the game fills every frame
//    (VertexCollector::AddGeometry), but they carry VK_BUFFER_USAGE_TRANSFER_SRC_BIT only, so they
//    can never be an acceleration-structure build input themselves. Each slot therefore owns two RHI
//    copy buffers (vertex and index) and the RHI list copies the used prefix of the frame into them
//    before the builds; the engine's shared device-local dynamic buffers are never written by this
//    module. The geometry descriptors are then the same translation as the static ones, with the RHI
//    copy buffers as the NVRHI buffers and the engine's own addresses only used for the offset
//    arithmetic (the copy starts at offset 0 with the same layout).
//  - Instance records: built on the CPU, per instance from ASManager::GetTLASInstanceForFilter - the
//    engine's own mask / custom-index / SBT-offset / flag rules (ASManager.cpp) - plus this module's
//    BLAS handle in the reference field, which that function deliberately leaves alone. A
//    VkAccelerationStructureInstanceKHR and an rt::InstanceDesc are layout-identical and their flag
//    bits agree, but the translation is explicit so the reference becomes setBLAS(handle).
//  - Order: the engine's own traversal. Static components come first, in the order
//    GetStaticBlasComponents() enumerates them (= the filter grid order of
//    VertexCollectorFilterTypeFlags_IterateOverFlags), then the slot's dynamic filters in the same
//    grid order. Every instance the module adds is one the engine would add if its own dynamic
//    components were live: the same filters, the same cull decisions, the same order. That order is
//    also what the engine's per-instance uniform arrays index by (WriteInstanceGeomInfo), so
//    instanceGeomInfoOffset[i] describes instance i of the TLAS this module builds.
//  - Two-frame warm-up: none. The records are host-side and NVRHI copies them into its own upload
//    buffer inside buildTopLevelAccelStruct, so the slot's TLAS of frame N describes frame N. The
//    one-frame lag the A3.0 instance-buffer path documented came from reading the engine's
//    device-local instance buffer, which is no longer used.
//
// Dynamic BLAS sizing and replacement:
//  - NVRHI sizes each rt::IAccelStruct from the descriptors given to createAccelStruct and errors at
//    build time if the allocation is smaller than the build requires, and neither rt::IAccelStruct
//    nor the pinned NVRHI exposes the data buffer, its size, or the native VkDevice needed to query
//    the requirement (there is no ASManager size wrapper either). The growth policy is therefore
//    "recreate exactly when the geometry shape changes": the handle is created from the same
//    descriptor list it is built with, and while a later frame's shape (per-geometry primitive
//    counts, vertex counts, stride, indexedness and transform presence) matches the create-time one,
//    the handle is reused and only rebuilt. A changed shape retires the old handle through
//    RhiFrameContext::Retire and creates a new one for the new shape. This is the conservative
//    fallback of the two policies the reconnaissance listed; it never under-allocates because the
//    create-time and build-time descriptors are identical by construction.
//  - A dynamic filter that is empty in a frame records nothing and produces no instance, but its
//    handle stays alive: a TLAS of an earlier frame may still reference it while the queue has not
//    finished. The next frame that brings geometry either rebuilds it (same shape) or replaces it.
//  - NVRHI never holds a reference to the BLAS an instance points at (it only transitions its state
//    when the TLAS is recorded), so a replaced BLAS is protected from release solely by this module's
//    retire call - every replaced handle goes through RhiFrameContext::Retire.
//
// Call order per frame, on the slot's open command list (RhiFrameContext::BeginSlot first):
//   1. BuildStatic(commandList)                          - every frame; rebuilds after a level change,
//                                                          a no-op while the set is current;
//   2. BuildTopLevel(commandList, frameIndex, rayCullMaskWorld, allowGeometryWithSkyFlag,
//      disableRayTracedGeometry)                         - every frame the TLAS is wanted.
// The one-time summary line (static figures as in A3.0, plus the dynamic BLAS/geometry/triangle
// figures, the frame's dynamic copy volume and the total TLAS instance count) is printed on the
// first BuildTopLevel after the static decision, with a grace period of 300 frames for a scene that
// never brings static geometry. A filter that has geometry but no RHI BLAS (a failed create) is
// reported once as well.
class RhiAccelStructs final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    RhiAccelStructs();
    ~RhiAccelStructs();

    RhiAccelStructs(const RhiAccelStructs &other) = delete;
    RhiAccelStructs(RhiAccelStructs &&other) noexcept = delete;
    RhiAccelStructs &operator=(const RhiAccelStructs &other) = delete;
    RhiAccelStructs &operator=(RhiAccelStructs &&other) noexcept = delete;

    // 'pDevice' is the RHI device (only the Vulkan backend is used, but nothing here is backend
    // specific). 'pFrameContext' is the host's frame model (RHI/RhiFrameContext.h): it owns the
    // retire queues every replaced structure and copy buffer goes through, and it is not owned.
    // 'pAsManager' is the engine's AS registry (ASManager.h), reachable from the host through
    // Scene::GetASManager(): it provides the collectors, the static geometry component set, the
    // TLAS instance attribute rules and the static generation counter; it is not owned either and
    // has to outlive this object (the host's device does). 'pfnPrint' is the host's log callback
    // (NvrhiFrameSkeleton::PrintFunction), used once for the summary and for warnings.
    //
    // Wraps the static collector's device-local vertex and index buffers and the per-slot dynamic
    // collectors' staging vertex and index buffers, so that the engine's VkBuffers can be referenced
    // by the NVRHI builds and copies; the static BLAS themselves are created on the first
    // BuildStatic that sees static geometry (at this point the geometry that the create-time
    // descriptors are sized from must be the geometry that is built, and at host-creation time the
    // static collector is normally still empty).
    //
    // Returns false and leaves the object empty if the device, the frame context or the AS manager
    // is null, or if one of the wraps fails; the host then keeps the legacy path only. Calling it
    // again on a created object is a no-op that returns true.
    bool Create(nvrhi::IDevice *pDevice,
                RhiFrameContext *pFrameContext,
                ASManager *pAsManager,
                PrintFunction pfnPrint);

    bool IsCreated() const { return device != nullptr; }

    // Records the static BLAS of the current static geometry: one rt::IAccelStruct per non-empty
    // static component, one NVRHI geometry per surface. Retires and recreates the whole set when
    // ASManager::GetStaticGeneration changes (a level load, including one that leaves the static
    // geometry empty); otherwise does nothing after the first successful recording, and does nothing
    // (but stays ready) on a frame whose static collector has no geometry yet, so the build happens
    // on the first frame that follows a level load. Must be called on the slot's open command list,
    // before BuildTopLevel.
    void BuildStatic(nvrhi::ICommandList *pCommandList);

    // Records the slot's dynamic geometry and TLAS build for this frame:
    //  - copies the used prefix of the slot's dynamic vertex and index staging into the slot's RHI
    //    copy buffers (growing them on demand through Retire) and records one PreferFastBuild BLAS
    //    per non-empty dynamic filter from them;
    //  - synthesizes the instance array: the static components first, then the dynamic filters, each
    //    from ASManager::GetTLASInstanceForFilter with this module's BLAS handle, in the engine's
    //    filter-grid order (see the class comment);
    //  - creates (and, when a later frame needs more, grows once) the slot's TLAS and records the
    //    build with ICommandList::buildTopLevelAccelStruct from that host array.
    // 'rayCullMaskWorld', 'allowGeometryWithSkyFlag' and 'disableRayTracedGeometry' are the frame
    // inputs the host passes to ASManager::PrepareForBuildingTLAS, so the culling decisions match
    // the legacy path; 'disableRayTracedGeometry' skips all instance work and records nothing,
    // leaving the slot's previous TLAS in place (as a zero instance count always has). A frame whose
    // instances were all culled does the same. Must be called on the same open command list as
    // BuildStatic.
    void BuildTopLevel(nvrhi::ICommandList *pCommandList,
                       uint32_t frameIndex,
                       uint32_t rayCullMaskWorld,
                       bool allowGeometryWithSkyFlag,
                       bool disableRayTracedGeometry);

    // The slot's TLAS, or null while the slot has never been built. The caller does not own the
    // handle; a re-created TLAS replaces it, and the debug-trace pass has to re-read it.
    nvrhi::rt::IAccelStruct *GetTopLevel(uint32_t frameIndex) const;

private:
    struct StaticBlas
    {
        // The engine filter this structure mirrors (VertexCollectorFilterTypeFlags): the mapping
        // between the engine's component grid and this module's structures, which the debug name
        // also carries as its index.
        uint32_t filter = 0;
        std::string debugName;
        nvrhi::rt::AccelStructHandle handle;

        // The descriptors the structure was created and is built with. createAccelStruct copies
        // them but nulls the buffer pointers in its own copy (vulkan-raytracing.cpp:406-427), so
        // the build call needs this original vector, which therefore has to stay alive.
        std::vector<nvrhi::rt::GeometryDesc> geometries;
    };

    struct DynamicBlas
    {
        uint32_t filter = 0;
        std::string debugName;
        nvrhi::rt::AccelStructHandle handle;

        // The descriptors of the last recorded build (the build call needs the buffer pointers,
        // which the copy inside createAccelStruct nulls).
        std::vector<nvrhi::rt::GeometryDesc> geometries;

        // The geometry shape the current handle was created for (see the class comment): the
        // per-geometry primitive count, vertex count, stride, indexedness and transform presence.
        // Recreated when this changes.
        std::vector<uint32_t> shape;

        // Whether this filter's BLAS was built in the slot's current frame; only an active entry is
        // added to the TLAS. Empty filters keep the handle but stay inactive.
        bool active = false;
    };

    // The slot's copies of the used dynamic vertex/index prefix: the NVRHI buffers an NVRHI build
    // may read (isAccelStructBuildInput) and their byte capacities.
    struct DynamicCopies
    {
        nvrhi::BufferHandle vertex;
        nvrhi::BufferHandle index;
        uint64_t vertexCapacity = 0;
        uint64_t indexCapacity = 0;
    };

    // Retires and clears the whole static set (a level change, or teardown before the handles are
    // dropped directly).
    void RetireStaticSet();

    // The static structure that mirrors 'filter', or null.
    const StaticBlas *FindStaticBlas(uint32_t filter) const;

    // Appends the static instances, in GetStaticBlasComponents() order.
    void AppendStaticInstances(uint32_t rayCullMaskWorld,
                               bool allowGeometryWithSkyFlag,
                               std::vector<nvrhi::rt::InstanceDesc> &instances);

    // Does this slot's dynamic work: marks the slot's entries inactive, copies the used
    // vertex/index prefixes, records one BLAS build per non-empty dynamic filter (creating or
    // replacing the handle under the shape policy) and appends the dynamic instances in filter-grid
    // order. Fills this frame's dynamic figures for the summary.
    void AppendDynamicSlot(nvrhi::ICommandList *pCommandList,
                           uint32_t frameIndex,
                           uint32_t rayCullMaskWorld,
                           bool allowGeometryWithSkyFlag,
                           std::vector<nvrhi::rt::InstanceDesc> &instances);

    // Makes 'buffer' at least 'needed' bytes large: creates a new buffer sized to the doubled
    // capacity (capped at 'maxCapacity', the collector's staging size) and retires the replaced one.
    // Returns false if the new buffer could not be created.
    bool EnsureDynamicCopyBuffer(nvrhi::BufferHandle &buffer,
                                 uint64_t &capacity,
                                 uint64_t needed,
                                 uint64_t maxCapacity,
                                 uint32_t frameIndex,
                                 const char *kind);

    // The one-shot warning for a non-empty filter that has no RHI BLAS ('filter' is its grid filter).
    void WarnUnresolvedFilter(uint32_t filter);

    // Decides whether the one-time summary prints on this frame (see the class comment). Sets the
    // 'summaryPrinted' flag when it does.
    bool IsSummaryDue();

    // Prints the summary from the current static/dynamic figures and the frame's TLAS instance count.
    void PrintSummary();

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;

    // Not owned: the host's frame model and the engine's AS registry, both outlive this object.
    RhiFrameContext *frameContext = nullptr;
    ASManager *asManager = nullptr;

    // The wraps of the engine buffers the builds read: the static collector's device-local vertex
    // and index buffers and the per-slot dynamic collectors' staging vertex and index buffers.
    // Created once, alive for the run (the collectors never re-create their buffers; Reset only
    // clears the collected geometry).
    nvrhi::BufferHandle vertexBuffer;
    nvrhi::BufferHandle indexBuffer;
    nvrhi::BufferHandle stagingVertexBuffer[MAX_FRAMES_IN_FLIGHT];
    nvrhi::BufferHandle stagingIndexBuffer[MAX_FRAMES_IN_FLIGHT];

    // The per-slot dynamic copy buffers (the dynamic BLAS build inputs).
    DynamicCopies dynamicCopies[MAX_FRAMES_IN_FLIGHT];

    // The static structures, one per non-empty static component of the current generation.
    std::vector<StaticBlas> staticBlas;
    bool staticBuilt = false;
    // Set after a permanent failure inside the static path (a removed device, mostly): the module
    // then stops trying instead of logging the same failure every frame.
    bool staticCreationFailed = false;
    // The ASManager::GetStaticGeneration() value the current static set (or its absence) belongs to.
    uint32_t staticGeneration = 0;

    // The dynamic structures, up to one per dynamic filter per slot (a filter that is empty in a
    // frame keeps its handle but is not active).
    std::vector<DynamicBlas> dynamicBlas[MAX_FRAMES_IN_FLIGHT];
    // Set after a permanent failure inside the dynamic path (a buffer or BLAS that could not be
    // created): the module stops the per-frame work and logs nothing further.
    bool dynamicCreationFailed = false;

    // The per-slot top-level structures and the instance counts they were created for. A TLAS is
    // created with exactly the count of its first build and grown to the maximum when a later frame
    // needs more, so it is sized to what it is used for instead of to the engine cap.
    nvrhi::rt::AccelStructHandle topLevel[MAX_FRAMES_IN_FLIGHT];
    uint32_t topLevelCapacity[MAX_FRAMES_IN_FLIGHT] = {};

    // The one-time evidence line: printed after the static decision so that its figures are final,
    // with a grace period for a scene that never brings static geometry.
    bool summaryPrinted = false;
    uint32_t summaryFrameGrace = 300;

    // The static BLAS figures of the summary: what the built structures cover.
    uint32_t staticGeometryCount = 0;
    uint64_t staticVertexCount = 0;
    uint64_t staticPrimitiveCount = 0;

    // The current frame's dynamic figures, for the summary: the BLAS built and the non-empty
    // dynamic filters / geometries / triangles / vertices they cover, and the copied bytes.
    uint32_t dynamicBlasCount = 0;
    uint32_t dynamicActiveFilterCount = 0;
    uint32_t dynamicGeometryCount = 0;
    uint64_t dynamicVertexCount = 0;
    uint64_t dynamicPrimitiveCount = 0;
    uint64_t dynamicCopyBytes = 0;

    // The instance count of the last BuildTopLevel, for the summary.
    uint32_t tlasInstanceCount = 0;

    // The one-time warning for a non-empty filter without an RHI BLAS.
    bool warnedUnresolvedInstance = false;
    // The one-time warning for a TLAS build that would need more instances than the module's bound.
    bool warnedInstanceOverflow = false;
};

}
