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

#include <vector>

#include "Common.h"
#include "Containers.h"
#include "vkpt/vkpt.h"

// The RHI layer's own sampler table (RHI/RhiTextureTable.h). Forward declared on purpose: this
// header must stay free of the NVRHI include.
namespace vkpt::rhi
{
    class RhiTextureTable;
}

namespace vkpt
{

// TODO: separate classes for fixed / updateable(with SamplerHandle) samplers
class SamplerManager
{
public:
    class Handle
    {
        friend class SamplerManager;

    public:
        explicit Handle();
        explicit Handle(RgSamplerFilter filter, RgSamplerAddressMode addressModeU, RgSamplerAddressMode addressModeV, RgMaterialCreateFlags flags);

        bool operator==(const Handle &other) const
        {
            return other.internalIndex == internalIndex;
        }

        // The RHI texture table needs the raw index to point a slot at the mirrored nvrhi sampler.
        uint32_t GetIndex() const { return internalIndex; }

        bool SetIfHasDynamicSamplerFilter(RgSamplerFilter newDynamicSamplerFilter);

    private:
        uint32_t internalIndex;
        bool hasDynamicSamplerFilter;
    };

public:
    // 'pRhiTextureTable' may be null (legacy-only): the manager then mirrors nothing. The world
    // manager gets the table so that it receives the sampler descs while it creates them; the
    // cubemap manager stays null because it would overwrite the same indices.
    SamplerManager(VkDevice device, uint32_t anisotropy, bool forceMinificationFilterLinear,
                   rhi::RhiTextureTable *pRhiTextureTable = nullptr);
    ~SamplerManager();

    SamplerManager(const SamplerManager &other) = delete;
    SamplerManager(SamplerManager &&other) noexcept = delete;
    SamplerManager &operator=(const SamplerManager &other) = delete;
    SamplerManager &operator=(SamplerManager &&other) noexcept = delete;

    void PrepareForFrame(uint32_t frameIndex);

    VkSampler GetSampler(
        RgSamplerFilter filter, 
        RgSamplerAddressMode addressModeU, 
        RgSamplerAddressMode addressModeV,
        bool forceLowestMip = false) const;

    // In case, if mip load bias was updated and a fresh sampler is required
    VkSampler GetSampler(const Handle &handle) const;

    // Wait idle and recreate all the samplers with new lod bias
    bool TryChangeMipLodBias(uint32_t frameIndex, float newMipLodBias);

    // The host hands over the RHI copy of the sampler table once both objects exist. Not owned;
    // null means legacy-only, and every use of the pointer is guarded.
    void SetRhiTextureTable(rhi::RhiTextureTable *pTable);

private:
    void CreateAllSamplers(uint32_t anisotropy, float mipLodBias);
    void AddAllSamplersToDestroy(uint32_t frameIndex);

    static uint32_t ToIndex(
        RgSamplerFilter filter,
        RgSamplerAddressMode addressModeU,
        RgSamplerAddressMode addressModeV,
        bool forceLowestMip);

    static uint32_t ToIndex(
        VkFilter filter,
        VkSamplerAddressMode addressModeU,
        VkSamplerAddressMode addressModeV,
        bool forceLowestMip);

private:
    VkDevice device;

    rgl::unordered_map<uint32_t, VkSampler> samplers;
    std::vector<VkSampler> samplersToDelete[MAX_FRAMES_IN_FLIGHT];
    float mipLodBias;
    uint32_t anisotropy;
    bool forceMinificationFilterLinear;

    // The RHI copy of the sampler table (RHI/RhiTextureTable.h); null until the host wires it up.
    rhi::RhiTextureTable *rhiTextureTable = nullptr;
};

}