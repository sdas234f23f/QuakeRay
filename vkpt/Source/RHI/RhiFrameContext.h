#pragma once

#include <nvrhi/nvrhi.h>
#include <nvrhi/vulkan.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace vkpt::rhi
{

// The RHI frame model: one command list per engine frame slot plus a retire queue per slot, so that the
// RHI records the way the engine records and resources outlive exactly the frames the engine's own
// deferred destroy queues do.
//
// The engine's model is the reference: MAX_FRAMES_IN_FLIGHT (Common.h:31) slots, and anything retired in
// frame N is destroyed at the start of frame N+2, right after that slot's fence wait
// (TextureManager::texturesToDestroy, SamplerManager::samplersToDelete,
// CubemapManager::cubemapsToDestroy). BeginSlot mirrors that moment: it waits for the slot's previous
// submission and only then releases the slot's retire queue.
//
// Synchronization: the engine's frame fence cannot be used for RHI work - it is signalled by the legacy
// submit, which happens after NvrhiFrameSkeleton::Render already submitted to the NVRHI queue - so the
// only honest sync point is the command queue's completed instance, which executeCommandList returns and
// queueGetCompletedInstance reports. NVRHI has no blocking per-queue wait, so BeginSlot polls that
// counter; in practice the slot's submission is two frames old and the check succeeds immediately.
//
// Semaphore ownership stays with the engine: the caller passes the acquire and the render-finished
// semaphore per slot, and EndSlot sets them as transient queue state right before the submit, exactly as
// NvrhiFrameSkeleton does today (NVRHI clears that state on the next submission, vulkan-queue.cpp:203-206).
//
// NVRHI allows a list to be re-opened while its previous submission is still executing: open() takes a
// fresh VkCommandBuffer and pool from the queue's shared pool, and the old buffer returns only at
// runGarbageCollection (vulkan-queue.cpp:339-384), so one list per slot is what lets the CPU run ahead.
//
// Retiring: anything a recorded list may still reference has to pass through Retire instead of being
// dropped. The backend does not register descriptor tables in referencedResources
// (vulkan-resource-bindings.cpp:1003-1009), so an immediate release can destroy a VkSampler the GPU is
// still reading, and the engine's fence proves nothing about the NVRHI queue.
class RhiFrameContext
{
public:
    RhiFrameContext();
    ~RhiFrameContext();

    RhiFrameContext(const RhiFrameContext &other) = delete;
    RhiFrameContext(RhiFrameContext &&other) noexcept = delete;
    RhiFrameContext &operator=(const RhiFrameContext &other) = delete;
    RhiFrameContext &operator=(RhiFrameContext &&other) noexcept = delete;

    // Creates one command list per slot. 'frameCount' is the engine's MAX_FRAMES_IN_FLIGHT. Returns false
    // and leaves the object empty if the device has no Vulkan interface or a list cannot be created.
    bool Create(nvrhi::IDevice *device, uint32_t frameCount);

    bool IsCreated() const { return !slots.empty(); }

    // Waits until the queue finished the slot's previous submission, releases everything retired in that
    // slot, runs NVRHI's garbage collection and opens the slot's command list. Call it where the engine
    // waits its frame fence - inside the frame's recording, before the first RHI command.
    void BeginSlot(uint32_t slot);

    // The slot's open command list; null unless the slot is between BeginSlot and EndSlot.
    nvrhi::ICommandList *GetCommandList(uint32_t slot) const;

    // Closes the slot's list and submits it, waiting on 'semaphoreToWait' and signalling
    // 'semaphoreToSignal' (either may be VK_NULL_HANDLE). The submission's instance becomes the slot's
    // sync point for the next BeginSlot.
    void EndSlot(uint32_t slot,
                 VkSemaphore semaphoreToWait = VK_NULL_HANDLE,
                 VkSemaphore semaphoreToSignal = VK_NULL_HANDLE,
                 nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics);

    // Defers the release of a resource until the queue has finished the submission it was used in. The
    // queue takes a reference immediately, so the caller may drop its own handle right after. Before the
    // first BeginSlot everything lands in slot 0, which is safe because nothing is in flight yet.
    template <typename T>
    void Retire(nvrhi::RefCountPtr<T> handle)
    {
        RetireRelease([handle = std::move(handle)]() mutable { handle = nullptr; });
    }

    // Waits until every submission finished and releases all retire queues. For teardown, not for the
    // frame loop.
    void WaitForIdle();

private:
    void RetireRelease(std::function<void()> release);

    struct Slot
    {
        nvrhi::CommandListHandle commandList;
        uint64_t lastSubmissionInstance = 0;
        bool hasSubmission = false;
        std::vector<std::function<void()>> retireQueue;
    };

    nvrhi::vulkan::IDevice *device = nullptr;
    std::vector<Slot> slots;
    uint32_t currentSlot = 0;
};

}
