#pragma once

#include "RhiFrameContext.h"

#include <nvrhi/vulkan.h>

#include <cstdint>
#include <vector>

namespace vkpt::rhi
{

// The RHI copy of the engine's texture table (`vkpt/Source/TextureDescriptors.{h,cpp}`): one bindless
// descriptor set that holds every texture slot and its sampler, filled by the same code paths that
// fill the legacy table. The legacy table keeps working untouched - both renderers stay runnable,
// because the `rhiframe` switch is a runtime switch - so the engine writes to both, guarded by
// whether this object exists.
//
// Set numbers: NVRHI's legacy binding mode - the default, and the mode RhiPipeline selects - places a
// pipeline's layouts into descriptor sets 0..n-1 in the order they were added, bindless or not, with
// nothing appended or reordered (vulkan-resource-bindings.cpp:1090-1099; only the registerSpace mode
// moves bindless layouts to the end, and there a bindless layout could never be set 0). The set index of
// this table is therefore decided by where the caller adds GetLayout(): set 1 for the frame skeleton
// (whose regular layout is added first), set 0 for an engine pass whose shader spells
// DESC_SET_TEXTURES 0 - and the engine's later sets keep their numbers as long as the caller adds the
// layouts in that same shader order. Inside the table's own set the textures sit at binding 0 and the
// samplers at binding 1 (RhiDescriptors.h).
//
// Capacity: the engine clamps its texture count to [TEXTURE_COUNT_MIN, TEXTURE_COUNT_MAX] and a
// bindless table's capacity is fixed at layout creation, so Create() must be given the same clamped
// value the legacy table is built with, not the constant.
//
// Clearing: the backend cannot clear a descriptor (a `ResourceType::None` item is a silent no-op that
// leaves the old one), so a slot whose texture is about to be destroyed has to be overwritten first -
// that is what ResetSlot() is for, and the engine must call it before vkDestroyImage.
//
// Lifetime: every wrapped texture and every engine sampler this table drops goes through the frame
// context's retire queue (RhiFrameContext::Retire) instead of being destroyed on the spot. NVRHI's
// backend does not register descriptor tables in referencedResources
// (vulkan-resource-bindings.cpp:1003-1009), so a recorded list can still sample a VkSampler or a
// view after the last CPU reference is gone - and the engine's own fences say nothing about NVRHI's
// queue. Teardown is the one exception: the host guarantees the device is idle before the table is
// destroyed, so the destructor drops the fallback pair and the slots directly.
//
// Cost: every SetSlot/ResetSlot ends in one vkUpdateDescriptorSets call (the backend does not batch),
// and the engine's initial all-dirty fill is up to TEXTURE_COUNT_MAX of them - accepted as a one-time
// start-up cost.
//
// Wrap fidelity (TODO(refactor), not solved here): a slot is filled through
// RhiTextureSource::wrapEngineTexture, which cannot reproduce everything the legacy table can -
// formats missing from NVRHI's map and the engine's PBR view swizzles have no counterpart. Such slots
// fall back to the 1x1 white texture and are counted by GetFallbackSlotCount(); the caller is meant to
// log that count once, so a fidelity gap cannot pass silently.
class RhiTextureTable
{
public:
    RhiTextureTable();
    ~RhiTextureTable();

    RhiTextureTable(const RhiTextureTable &other) = delete;
    RhiTextureTable(RhiTextureTable &&other) noexcept = delete;
    RhiTextureTable &operator=(const RhiTextureTable &other) = delete;
    RhiTextureTable &operator=(RhiTextureTable &&other) noexcept = delete;

    // Builds the layout, the table and the 1x1 white fallback. 'capacity' is the engine's clamped
    // texture count. 'frameContext' is the host's frame model (RhiFrameContext.h): it owns the retire
    // queues every release goes through, and the table does not own it. Returns false and leaves the
    // object empty if anything fails.
    bool Create(nvrhi::IDevice *device, uint32_t capacity, rhi::RhiFrameContext *frameContext);

    bool IsCreated() const { return table != nullptr; }

    // The layout for the pipeline assembly and the table for GraphicsState::addBindingSet. Both are
    // null before Create() returns true.
    nvrhi::IBindingLayout *GetLayout() const { return bindingLayout; }
    nvrhi::IDescriptorTable *GetTable() const { return table; }

    // Mirrors one legacy table write: element 'slot' of the texture array and of the sampler array.
    // 'samplerIndex' is the engine sampler index of the slot (SamplerManager), which is what makes the
    // sampler side follow dynamic filter changes. A repeated call with the same 'image' reuses the
    // wrapped texture instead of creating another view of it.
    void SetSlot(uint32_t slot, VkImage image, VkFormat format, uint32_t width, uint32_t height,
                 uint32_t mipLevels, uint32_t samplerIndex);

    // Overwrites the slot with the fallback texture and its sampler. Must precede vkDestroyImage for
    // the slot's image.
    void ResetSlot(uint32_t slot);

    // A wrapped engine image is foreign to NVRHI, so the first command list that will sample it has
    // to declare its state (RhiTextureSource.h documents why). The passes call this once per frame,
    // before they bind the table; it is a no-op when nothing is pending.
    void TrackPendingTextures(nvrhi::ICommandList *commandList);

    // The engine calls SetSamplerDesc while it creates its VkSamplers (index -> filter/address/
    // anisotropy/mip bias) and RebuildSamplers after it recreates them wholesale (the mip-lod-bias
    // change). RebuildSamplers recreates the NVRHI samplers from the stored descs and rewrites the
    // sampler side of every slot that is currently set.
    void SetSamplerDesc(uint32_t samplerIndex, const nvrhi::SamplerDesc &desc);
    void RebuildSamplers();

    // Slots that could not be wrapped and hold the fallback texture instead (see the class comment).
    uint32_t GetFallbackSlotCount() const { return fallbackSlotCount; }

private:
    nvrhi::IDevice *device = nullptr;

    // The host's frame model (RhiFrameContext.h), which owns the retire queues; not owned here. Null
    // only if the host passed none, in which case the release paths drop handles directly.
    rhi::RhiFrameContext *frameContext = nullptr;

    nvrhi::BindingLayoutHandle bindingLayout;
    nvrhi::DescriptorTableHandle table;

    nvrhi::TextureHandle fallbackTexture;
    nvrhi::SamplerHandle fallbackSampler;

    struct Slot
    {
        VkImage image = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mipLevels = 1;
        uint32_t samplerIndex = 0;
        nvrhi::TextureHandle texture;
        bool fallback = false;
    };

    std::vector<Slot> slots;
    std::vector<nvrhi::SamplerHandle> samplers;
    std::vector<nvrhi::SamplerDesc> samplerDescs;

    // Wrapped engine textures whose first-use state has not been declared in a command list yet;
    // drained by TrackPendingTextures.
    std::vector<nvrhi::TextureHandle> pendingTracking;
    uint32_t fallbackSlotCount = 0;
};

}
