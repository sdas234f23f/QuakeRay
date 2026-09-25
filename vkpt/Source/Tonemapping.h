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

#include "Common.h"
#include "ShaderManager.h"
#include "Framebuffers.h"
#include "GlobalUniform.h"
#include "MemoryAllocator.h"
#include "Buffer.h"

namespace vkpt
{

class Tonemapping : public IShaderDependency
{
public:
    Tonemapping(
        VkDevice device,
        std::shared_ptr<Framebuffers> framebuffers,
        const std::shared_ptr<const ShaderManager> &shaderManager,
        const std::shared_ptr<const GlobalUniform> &uniform,
        const std::shared_ptr<MemoryAllocator> &allocator);
    ~Tonemapping() override;

    Tonemapping(const Tonemapping &other) = delete;
    Tonemapping(Tonemapping &&other) noexcept = delete;
    Tonemapping &operator=(const Tonemapping &other) = delete;
    Tonemapping &operator=(Tonemapping &&other) noexcept = delete;

    // The host-only half of the exposure chain: writes the params prefix of the 'frameIndex'
    // tonemapping buffer (tmExposureBias, the fixed tone-curve parameters, frameTime from
    // 'uniform', resetCurve from the slot's pending-reset flag, and the derived knee) through
    // the persistent mapping, and clears that flag. This is exactly the block CalculateExposure
    // wrote inline before it was split out, so the legacy bytes stay identical. No command
    // buffer, no barrier and no dispatch: the histogram and average are CalculateExposure's.
    // The traced RHI path calls it on the slot's own buffer after the slot fence is waited
    // (rhi::RhiFrameContext::BeginSlot) and before it records those two dispatches.
    // 'exposureBias' and 'contrast' come from the draw info's tone-mapping params with the same
    // sources and fallbacks the legacy Render uses (VulkanDevice.cpp:1051-1059).
    void PrepareExposureParams(
        uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform,
        float exposureBias, float contrast);

    // The full legacy exposure chain: the host write above, then CmLuminanceHistogram and
    // CmLuminanceAvg over the same slot's buffer on 'cmd'. Not used by the RHI path.
    void CalculateExposure(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform,
        float exposureBias, float contrast);

    VkDescriptorSetLayout GetDescSetLayout() const;
    VkDescriptorSet GetDescSet(uint32_t frameIndex) const;

    // The host-visible VkBuffer of 'frameIndex' (one buffer per frame in flight, see
    // CreateTonemappingBuffer). The RHI layer wraps it as the world shader's set 2 binding 0,
    // which the shader declares as StructuredBuffer<ShTonemapping>; pass GetElementSize() as the
    // wrap's structStride.
    VkBuffer GetBuffer(uint32_t frameIndex) const;

    // sizeof(ShTonemapping): the distance between frame slots and the structStride the shader's
    // StructuredBuffer declares (SPIR-V ArrayStride 1620 in RsWorld.frag.spv).
    uint32_t GetElementSize() const;

    // Writes the value the world shader reads as avgLuminance (member 23 of ShTonemapping,
    // Offset 1616 in RsWorld.frag.spv) for 'frameIndex'. On the legacy path and in the traced
    // RHI path the GPU owns that field: CmLuminanceAvg stores adapted_luminance into it, and
    // in the traced path the exposure chain runs earlier in the same frame's list, before
    // CmPrepareFinal reads it. This setter is then the raster-mode dev fallback only:
    // RhiSkyPass::RenderWorld draws before any chain could run and keeps its 1/9.6 stand-in,
    // because the RHI raster mode has no PRE_FINAL to histogram - decoding the value through
    // Exposure.h:33-51, 1 gives the world colour the constant factor 1/9.6, and a non-positive
    // value outputs black. Do not call it in the traced mode: the average pass overwrites the
    // field in the same frame.
    void SetAvgLuminance(uint32_t frameIndex, float avgLuminance);
    
    void OnShaderReload(const ShaderManager *shaderManager) override;

private:
    void CreateTonemappingBuffer(const std::shared_ptr<MemoryAllocator> &allocator);
    void CreateTonemappingDescriptors();

    void CreatePipelineLayout(VkDescriptorSetLayout *pSetLayouts, uint32_t setLayoutCount);
    void CreatePipelines(const ShaderManager *shaderManager);
    void DestroyPipelines();

private:
    VkDevice device;

    std::shared_ptr<Framebuffers> framebuffers;

    // One buffer per frame in flight: the host rewrites the params prefix each
    // frame, while the GPU owns the histogram/curve/adaptedLuminance state that
    // lives past it. The temporal state is naturally double-buffered as well,
    // which is harmless for eye adaptation (slow convergence).
    Buffer tmBuffer[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorSetLayout tmDescSetLayout;
    VkDescriptorPool tmDescPool;
    VkDescriptorSet tmDescSet[MAX_FRAMES_IN_FLIGHT] = {};

    // host-mapped views of tmBuffer (HOST_VISIBLE) for writing tone mapper params
    void *mappedTmBuffer[MAX_FRAMES_IN_FLIGHT] = {};
    bool  resetRequired[MAX_FRAMES_IN_FLIGHT] = {};

    VkPipelineLayout pipelineLayout;

    VkPipeline histogramPipeline;
    VkPipeline avgLuminancePipeline;
};

}