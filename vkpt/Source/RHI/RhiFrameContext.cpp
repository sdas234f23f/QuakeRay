#include "RhiFrameContext.h"

#include <thread>
#include <utility>

namespace vkpt::rhi
{

RhiFrameContext::RhiFrameContext() = default;

RhiFrameContext::~RhiFrameContext()
{
    // The header leaves the drain to the teardown caller (WaitForIdle), so the destructor only drops
    // what the context owns and does not wait itself.
    for (Slot &s : slots)
    {
        s.retireQueue.clear();
        s.commandList = nullptr;
    }

    slots.clear();
    device = nullptr;
}

bool RhiFrameContext::Create(nvrhi::IDevice *pDevice, uint32_t frameCount)
{
    if (pDevice == nullptr || frameCount == 0)
    {
        return false;
    }

    // Every failure below funnels through this, so the object is empty whatever failed: IsCreated()
    // stays false, the destructor has nothing to release, and a retry starts from scratch instead of
    // mixing the lists of the old attempt with the new ones (RhiTextureTable.cpp:63).
    auto empty = [this]()
    {
        slots.clear();

        device = nullptr;
        currentSlot = 0;
    };

    empty();

    // The retire queue is Vulkan-only work and the semaphore calls come from nvrhi/vulkan.h, so the
    // context needs the Vulkan backend; the frame skeleton makes the same cast
    // (NvrhiFrameSkeleton.cpp:69).
    device = dynamic_cast<nvrhi::vulkan::IDevice *>(pDevice);
    if (device == nullptr)
    {
        empty();
        return false;
    }

    // One list per engine frame slot: a list may be re-opened while its previous submission is still
    // executing (the header), so one list per slot is what lets the CPU run ahead by the engine's
    // MAX_FRAMES_IN_FLIGHT.
    slots.resize(frameCount);

    for (uint32_t i = 0; i < frameCount; i++)
    {
        slots[i].commandList = device->createCommandList();
        if (slots[i].commandList == nullptr)
        {
            empty();
            return false;
        }
    }

    return true;
}

void RhiFrameContext::BeginSlot(uint32_t slot)
{
    if (!IsCreated() || slot >= slots.size())
    {
        return;
    }

    Slot &s = slots[slot];

    // WaitForIdle releases the lists while the slots stay in place; a BeginSlot that arrives after it
    // is out of the frame loop already, so it must not open a list that is not there.
    if (s.commandList == nullptr)
    {
        return;
    }

    // The engine's frame fence cannot order RHI work (the header), and NVRHI has no blocking per-queue
    // wait, so poll the Graphics queue's completed instance. The slot's submission is two frames old,
    // so the first read normally satisfies the condition and the loop only spins when the GPU is
    // behind the engine's pacing.
    if (s.hasSubmission)
    {
        while (device->queueGetCompletedInstance(nvrhi::CommandQueue::Graphics) < s.lastSubmissionInstance)
        {
            std::this_thread::yield();
        }
    }

    // The engine's drain moment: retired in frame N, released at the start of frame N+2, right after
    // the fence wait (TextureManager::texturesToDestroy and the two sibling queues). Releasing here,
    // after the wait above, means the queue has finished the submission that the resources could
    // still be referenced by.
    s.retireQueue.clear();

    // The command buffers of the closed lists (this slot's and every other slot's) return to the
    // queue's shared pool only here (the header); without it the pool grows by one buffer per frame.
    device->runGarbageCollection();

    currentSlot = slot;
    s.commandList->open();
}

nvrhi::ICommandList *RhiFrameContext::GetCommandList(uint32_t slot) const
{
    if (!IsCreated() || slot >= slots.size())
    {
        return nullptr;
    }

    return slots[slot].commandList.Get();
}

void RhiFrameContext::EndSlot(uint32_t slot,
                              VkSemaphore semaphoreToWait,
                              VkSemaphore semaphoreToSignal,
                              nvrhi::CommandQueue queue)
{
    if (!IsCreated() || slot >= slots.size())
    {
        return;
    }

    Slot &s = slots[slot];

    // Nothing was opened for this slot, so there is nothing to close and submit.
    if (s.commandList == nullptr)
    {
        return;
    }

    // Transient queue state, exactly as the frame skeleton sets it (NvrhiFrameSkeleton.cpp:247-248):
    // the submit below consumes it and a later submission clears whatever remains
    // (vulkan-queue.cpp:203-206). NVRHI skips a null handle by itself (vulkan-queue.cpp:104,113);
    // the checks keep the semaphore ownership explicit - they belong to the caller.
    if (semaphoreToWait != VK_NULL_HANDLE)
    {
        device->queueWaitForSemaphore(queue, semaphoreToWait, 0);
    }

    if (semaphoreToSignal != VK_NULL_HANDLE)
    {
        device->queueSignalSemaphore(queue, semaphoreToSignal, 0);
    }

    s.commandList->close();

    // The returned instance is the slot's new sync point: the next BeginSlot of this slot waits for it
    // before it releases the slot's retire queue.
    s.lastSubmissionInstance = device->executeCommandList(s.commandList, queue);
    s.hasSubmission = true;
}

void RhiFrameContext::RetireRelease(std::function<void()> release)
{
    // Without slots there is no submission of this context that could still reference the resource,
    // so nothing can be deferred behind: release at once instead of writing out of bounds. Before the
    // first BeginSlot everything lands in slot 0, which the header calls safe - nothing is in flight.
    if (slots.empty())
    {
        release();
        return;
    }

    slots[currentSlot].retireQueue.push_back(std::move(release));
}

void RhiFrameContext::WaitForIdle()
{
    if (device != nullptr)
    {
        device->waitForIdle();
    }

    // Every submission has finished by now, so the retire queues can run and the lists can go. The
    // slots themselves stay; IsCreated() turns false only at the destructor.
    for (Slot &s : slots)
    {
        s.retireQueue.clear();
        s.hasSubmission = false;
        s.commandList = nullptr;
    }
}

}
