// Copyright (c) 2022 Sultim Tsyrendashiev
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

#include <bitset>

#include "vkpt/vkpt.h"

#include "AutoBuffer.h"

namespace vkpt
{
    namespace detail
    {
        constexpr size_t PORTAL_LIST_BITCOUNT = 63;
    }

    class PortalList
    {
    public:
        PortalList(VkDevice device, std::shared_ptr<MemoryAllocator> allocator);
        ~PortalList();

        PortalList(const PortalList &other) = delete;
        PortalList(PortalList &&other) noexcept = delete;
        PortalList &operator=(const PortalList &other) = delete;
        PortalList &operator=(PortalList &&other) noexcept = delete;

        void Upload(uint32_t frameIndex, const RgPortalUploadInfo &info);
        void SubmitForFrame(VkCommandBuffer cmd, uint32_t frameIndex);

        // Read-only views for the RHI layer's portal-buffer copy: the engine's own
        // PortalList::SubmitForFrame runs only from the legacy VulkanDevice::Render
        // (VulkanDevice.cpp:882) and is bypassed under `rhiframe`, so the RHI pass that dispatches
        // the reflrefr raygen - the only reader of the portal buffer (set 9,
        // BINDING_PORTAL_INSTANCES) - records the copy itself: all GetBufferSize() bytes
        // (= PORTAL_MAX_COUNT * sizeof(ShPortalInstance), 4032 B) from the frame slot's
        // GetStagingBuffer(frameIndex) into GetDeviceLocalBuffer(), before its dispatch. The game
        // stages portals through VulkanDevice::UploadPortal -> PortalList::Upload every frame while
        // `rt_teleport_portals` is 1 (r_world.c:3319-3354), independently of the reflrefr gate.
        // Both buffers stay valid while this object lives.
        // (Non-const for the staging buffer only: AutoBuffer's accessor is not const.)
        VkBuffer GetStagingBuffer(uint32_t frameIndex);
        VkBuffer GetDeviceLocalBuffer() const;
        // Byte size of the two buffers above; the size the RHI copy must use.
        VkDeviceSize GetBufferSize() const;

        // Clears the per-frame "already uploaded" bookkeeping Upload throws on (PortalList.cpp:58-61),
        // exactly the way SubmitForFrame clears it after recording its copy. It must be called once
        // per rendered frame on the path that stages the uploads - under `rhiframe` that is the RHI
        // frame - after the frame's Upload calls and not between them; the legacy path keeps its own
        // call inside SubmitForFrame, and the two paths are per-frame exclusive:
        //  - a frame that leaves the bits set makes the next frame's Upload of the same portal index
        //    throw, and with `rt_teleport_portals 1` the game re-uploads every teleport every frame
        //    (r_world.c:3319-3354);
        //  - a frame whose reflrefr pass was gated off and recorded no copy must still clear them:
        //    its uploads are dropped, which is safe because nothing read the device buffer that
        //    frame, but the bits must not survive into the next frame.
        // Clearing before the recorded copy is submitted is safe: it touches no buffer memory, only
        // the bits that detect a second upload in the same frame; the copy reads the staging bytes
        // later, and the next Upload that could overwrite them for this slot runs after the slot's
        // submission has completed (the property AutoBuffer's per-slot staging exists for).
        void ResetUploads();

        VkDescriptorSet GetDescSet(uint32_t frameIndex) const;
        VkDescriptorSetLayout GetDescSetLayout() const;

    private:
        void CreateDescriptors();

    private:
        VkDevice device;
        std::shared_ptr<AutoBuffer> buffer;

        VkDescriptorPool        descPool;
        VkDescriptorSetLayout   descSetLayout;
        VkDescriptorSet         descSet;

        std::bitset<detail::PORTAL_LIST_BITCOUNT> uploadedIndices;
    };
}
