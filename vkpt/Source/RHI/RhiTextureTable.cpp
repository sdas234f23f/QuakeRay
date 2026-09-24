#include "RhiTextureTable.h"

#include "RhiDescriptors.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"

#include <algorithm>
#include <string>

namespace vkpt::rhi
{

namespace
{

// The sampler handle of an engine sampler index, or 'fallback' when that index has no handle: a slot
// can arrive before its sampler was reported through SetSamplerDesc, and the engine's bit-packed
// indices leave gaps that never get a desc at all. The table has no PARTIALLY_BOUND
// (RhiDescriptors.cpp), so a slot's sampler element must hold a valid sampler even then, and the
// fallback is what goes in.
nvrhi::ISampler *resolveSampler(const std::vector<nvrhi::SamplerHandle> &samplers,
                                uint32_t samplerIndex,
                                nvrhi::ISampler *fallback)
{
    if (samplerIndex >= samplers.size() || samplers[samplerIndex] == nullptr)
    {
        return fallback;
    }

    return samplers[samplerIndex];
}

}

RhiTextureTable::RhiTextureTable() = default;

RhiTextureTable::~RhiTextureTable()
{
    // Teardown does not go through the frame tracker: the host guarantees the device is idle before
    // the table goes away, so everything below can be dropped directly.
    bindingLayout = nullptr;
    table = nullptr;
    fallbackTexture = nullptr;
    fallbackSampler = nullptr;

    // The wrapped engine textures reference their images, they do not own them (RhiTextureSource.h):
    // clearing the slots drops those references and destroys no VkImage. The engine has to have
    // overwritten the slot of every image it destroys (ResetSlot) and to have stopped sampling the
    // table before it goes away.
    slots.clear();
    samplers.clear();
    samplerDescs.clear();
}

bool RhiTextureTable::Create(nvrhi::IDevice *pDevice, uint32_t capacity, rhi::RhiFrameContext *pFrameContext)
{
    if (pDevice == nullptr || capacity == 0)
    {
        return false;
    }

    // Every failure below funnels through this, so the object is empty whatever failed: IsCreated()
    // is false, the destructor has nothing to release, and a retry starts from scratch instead of
    // mixing records of the old table with the new one.
    auto empty = [this]()
    {
        bindingLayout = nullptr;
        table = nullptr;
        fallbackTexture = nullptr;
        fallbackSampler = nullptr;

        slots.clear();
        samplers.clear();
        samplerDescs.clear();

        device = nullptr;
        frameContext = nullptr;
        fallbackSlotCount = 0;
    };

    empty();
    device = pDevice;
    frameContext = pFrameContext;

    // The layout fixes the table's capacity - a bindless layout's maxCapacity cannot change later -
    // so 'capacity' has to be the clamped value the legacy table was built with, not the constant
    // (RhiDescriptors.h).
    bindingLayout = rhi::createTextureTableLayout(device, capacity, "RhiTextureTable layout");
    if (bindingLayout == nullptr)
    {
        empty();
        return false;
    }

    table = rhi::createTextureTable(device, bindingLayout, "RhiTextureTable");
    if (table == nullptr)
    {
        empty();
        return false;
    }

    // The 1x1 white fallback is filled through a command list of its own, executed and waited for
    // before Create() returns, exactly as the frame skeleton fills its white texture
    // (NvrhiFrameSkeleton.cpp:321-338). Nothing samples it before that.
    nvrhi::CommandListHandle commandList = device->createCommandList();
    if (commandList == nullptr)
    {
        empty();
        return false;
    }

    commandList->open();
    fallbackTexture = rhi::createWhiteTexture(device, commandList, "RhiTextureTable white fallback");
    commandList->close();

    if (fallbackTexture == nullptr)
    {
        empty();
        return false;
    }

    device->executeCommandList(commandList, nvrhi::CommandQueue::Graphics);
    device->waitForIdle();

    // The list was needed for this one upload; the frame skeleton keeps a private one for its
    // frames. After waitForIdle() nothing is in flight that the release could disturb.
    commandList = nullptr;

    nvrhi::SamplerDesc fallbackDesc;
    // The texture is 1x1 white, so the filter mode cannot show through; nearest is the least
    // surprising of the two, and the frame skeleton's white sampler makes the same choice
    // (NvrhiFrameSkeleton.cpp:340-345).
    fallbackDesc.minFilter = false;
    fallbackDesc.magFilter = false;
    fallbackDesc.mipFilter = false;

    fallbackSampler = rhi::createSampler(device, fallbackDesc, "RhiTextureTable fallback sampler");
    if (fallbackSampler == nullptr)
    {
        empty();
        return false;
    }

    // The engine's sampler indices are bit-packed (SamplerManager.cpp:252-297): 2 filters x 5 x 5
    // addresses plus one force-lowest-mip sampler are ever created (SamplerManager.cpp:86-153), so
    // at most 51 entries are filled, but the largest index *value* is 256 (the force-lowest-mip
    // flag), not 51. The engine clamps its texture count to [TEXTURE_COUNT_MIN, TEXTURE_COUNT_MAX]
    // = 1024..4096 (Const.h:33-34), so 'capacity' always covers every sampler index too, and the
    // registry lookup stays a plain array element. The never-filled entries cost a few hundred
    // kilobytes at capacity 4096, accepted for that.
    slots.resize(capacity);
    samplers.resize(capacity);
    samplerDescs.resize(capacity);

    return true;
}

void RhiTextureTable::SetSlot(uint32_t slot, VkImage image, VkFormat format, uint32_t width, uint32_t height,
                              uint32_t mipLevels, uint32_t samplerIndex)
{
    // A slot past the capacity cannot be written anyway (writeDescriptorTable returns false, and a
    // bindless table has no PARTIALLY_BOUND), and a null image is the engine's "no texture" state,
    // which ResetSlot has already described.
    if (!IsCreated() || slot >= slots.size() || image == VK_NULL_HANDLE)
    {
        return;
    }

    Slot &s = slots[slot];

    // The engine's updateable textures re-copy into the same VkImage, so the wrapped texture of a
    // repeated call is still valid and only the descriptor is written again below. What can change
    // across repeated calls is the sampler index (the dynamic filter), which is why the sampler
    // element is rewritten even when the texture is not.
    if (s.image != image)
    {
        // The wrap the slot holds now is about to be replaced. A recorded list can still sample it,
        // so it stays referenced by this local until the new texture is in place and then goes to
        // the tracker; the table's own fallback is never retired (class comment).
        nvrhi::TextureHandle previousTexture = s.texture;

        // The bridge keeps the image's level count (RhiTextureSource.h) but still exposes a single
        // 2D layer; NVRHI cannot see the engine's cubemap layers.
        nvrhi::TextureHandle texture = rhi::wrapEngineTexture(
            device,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image)),
            uint32_t(format),
            width,
            height,
            mipLevels,
            "Engine texture slot " + std::to_string(slot) + " (RhiTextureTable)");

        if (texture == nullptr)
        {
            // A format missing from NVRHI's map or an engine PBR view swizzle cannot be wrapped
            // (RhiTextureSource.h): the slot holds the white fallback instead, and the count reports
            // the fidelity gap to the caller.
            texture = fallbackTexture;

            if (!s.fallback)
            {
                s.fallback = true;
                fallbackSlotCount++;
            }
        }
        else
        {
            if (s.fallback)
            {
                // The slot no longer holds the fallback, so it gives its entry back: the count describes
                // the slots that hold the fallback now, not the wraps that ever failed.
                s.fallback = false;
                fallbackSlotCount--;
            }

            // A newly wrapped engine image is foreign to NVRHI: the first command list that samples
            // it has to declare its state (RhiTextureSource.h), so it is left for
            // TrackPendingTextures. The fallback path needs no tracking - the table created that
            // texture itself.
            pendingTracking.push_back(texture);
        }

        s.image = image;
        s.format = format;
        s.width = width;
        s.height = height;
        s.mipLevels = mipLevels;
        s.texture = texture;

        // The slot no longer references the previous wrap; hand it to the tracker so the queue, not
        // this assignment, drops the last reference.
        if (frameContext != nullptr && previousTexture != nullptr &&
            previousTexture.Get() != fallbackTexture.Get())
        {
            frameContext->Retire(std::move(previousTexture));
        }
    }

    s.samplerIndex = samplerIndex;

    // Both elements of the slot, one vkUpdateDescriptorSets call each (the backend does not batch,
    // RhiDescriptors.h). The writes can only fail on a null argument or a slot past the capacity,
    // and neither can reach this point, so their results need no handling.
    rhi::setTexture(device, table, slot, s.texture);
    rhi::setSampler(device, table, slot, resolveSampler(samplers, samplerIndex, fallbackSampler));
}

void RhiTextureTable::ResetSlot(uint32_t slot)
{
    if (!IsCreated() || slot >= slots.size())
    {
        return;
    }

    Slot &s = slots[slot];

    // The caller destroys the image right after this call, so the slot must stop referencing the
    // wrapped texture: the backend cannot clear a descriptor (RhiDescriptors.h) and must not be
    // left pointing at an image that is about to die. The white texture and the fallback sampler
    // take the pair's place, the same values an unwrappable slot starts with.
    rhi::setTexture(device, table, slot, fallbackTexture);

    // A reset slot holds the fallback because it has no image any more, not because a wrap failed:
    // GetFallbackSlotCount() tracks only the fidelity gaps, so a counted slot gives its entry back.
    if (s.fallback)
    {
        s.fallback = false;
        fallbackSlotCount--;
    }

    s.image = VK_NULL_HANDLE;
    s.format = VK_FORMAT_UNDEFINED;
    s.width = 0;
    s.height = 0;
    s.mipLevels = 1;
    s.samplerIndex = 0;

    // The image is going away: a wrap that has not reached a command list yet must not be tracked
    // later.
    const auto pending = std::find(pendingTracking.begin(), pendingTracking.end(), s.texture);
    if (pending != pendingTracking.end())
    {
        pendingTracking.erase(pending);
    }

    // The caller destroys the image right after this call, and a recorded list can still sample the
    // wrap the slot drops, so it goes to the tracker instead of dying with the assignment below. A
    // slot holding the table's own fallback has nothing to release.
    if (frameContext != nullptr && s.texture != nullptr && s.texture.Get() != fallbackTexture.Get())
    {
        frameContext->Retire(std::move(s.texture));
    }

    s.texture = nullptr;

    // The sampler element follows the texture into the reset state, and RebuildSamplers skips the
    // slot from now on because it carries no texture.
    rhi::setSampler(device, table, slot, fallbackSampler);
}

void RhiTextureTable::TrackPendingTextures(nvrhi::ICommandList *commandList)
{
    if (commandList == nullptr || pendingTracking.empty())
    {
        return;
    }

    // The wrapped images are foreign to NVRHI (RhiTextureSource.h): without this the first command
    // list that samples one would transition it out of an unknown layout, and Vulkan would be free
    // to discard its contents. After this list closes, the tracker starts from the wrapper's
    // initialState (ShaderResource) on its own.
    for (const nvrhi::TextureHandle &texture : pendingTracking)
    {
        commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources,
                                               nvrhi::ResourceStates::ShaderResource);
    }

    pendingTracking.clear();
}

void RhiTextureTable::SetSamplerDesc(uint32_t samplerIndex, const nvrhi::SamplerDesc &desc)
{
    if (!IsCreated() || samplerIndex >= samplers.size())
    {
        return;
    }

    samplerDescs[samplerIndex] = desc;

    // The handle the index held can still be sampled by a recorded list: it is kept in this local
    // until the new one is in place, then handed to the tracker.
    nvrhi::SamplerHandle previous = samplers[samplerIndex];

    // The handle is created right away instead of being left to RebuildSamplers for two reasons:
    // a non-null entry is the record of which indices the engine has reported - RebuildSamplers
    // recreates exactly those and must not turn the default descs of the unreported ones into 4096
    // samplers - and a slot set before the next wholesale rebuild then gets a real sampler instead
    // of the fallback. The price is that a wholesale rebuild creates each reported sampler twice,
    // once here and once in RebuildSamplers; that is the engine's documented sequence
    // (RhiTextureTable.h), the count is 51 at most, and the rebuild happens only on the rare
    // mip-lod-bias change.
    samplers[samplerIndex] =
        rhi::createSampler(device, desc, "Engine sampler " + std::to_string(samplerIndex) + " (RhiTextureTable)");

    if (frameContext != nullptr && previous != nullptr)
    {
        frameContext->Retire(std::move(previous));
    }
}

void RhiTextureTable::RebuildSamplers()
{
    if (!IsCreated())
    {
        return;
    }

    // The handles the current descriptors reference are kept until every slot has been rewritten:
    // an nvrhi::Sampler owns its VkSampler and destroys it when the last reference goes - the
    // descriptor table holds none, its writes copy the VkSampler into the descriptor set
    // (vulkan-resource-bindings.cpp:887-898) - and the engine can have frames in flight that still
    // sample them. Keeping the old vector alive holds all of them for the whole rebuild.
    std::vector<nvrhi::SamplerHandle> previousSamplers = samplers;

    for (size_t i = 0; i < samplers.size(); i++)
    {
        // A null entry is an index the engine has not reported: there is no desc to recreate it
        // from, and the default descs would be materialized as samplers otherwise.
        if (samplers[i] == nullptr)
        {
            continue;
        }

        samplers[i] = rhi::createSampler(device, samplerDescs[i],
                                         "Engine sampler " + std::to_string(i) + " (RhiTextureTable)");
    }

    // Recreating the handles changes what the slot elements must point to, and tracking which slot
    // referenced which sampler would be a second registry. The engine's own mip-lod-bias path marks
    // every slot dirty and rewrites its whole table for the same reason (TextureManager.cpp:410-416),
    // so rewriting every slot that carries a texture here is accepted; slots without one (never
    // set, or reset) are left to their fallback sampler.
    for (size_t slot = 0; slot < slots.size(); slot++)
    {
        const Slot &s = slots[slot];

        if (s.texture == nullptr)
        {
            continue;
        }

        rhi::setSampler(device, table, uint32_t(slot), resolveSampler(samplers, s.samplerIndex, fallbackSampler));
    }

    // The rebuild is done writing; the handles it replaced are referenced only by that vector now,
    // and recorded lists can still sample them. Each goes to the tracker instead of being destroyed
    // when the vector dies; with no tracker attached the vector releases them here, exactly as
    // before.
    if (frameContext != nullptr)
    {
        for (nvrhi::SamplerHandle &previous : previousSamplers)
        {
            if (previous != nullptr)
            {
                frameContext->Retire(std::move(previous));
            }
        }
    }
}

}
