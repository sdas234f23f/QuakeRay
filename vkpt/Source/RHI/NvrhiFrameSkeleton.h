/*
* Copyright (c) 2024 Sultim Tsyrendashiev
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nvrhi/vulkan.h>

#include "../Common.h"
#include "../ISwapchainDependency.h"

namespace vkpt
{

class Swapchain;

// The RHI frame skeleton: the frame is recorded and submitted through the RHI
// layer instead of the Vulkan command buffers of the renderer. It draws a
// fullscreen pattern into the swapchain image that was acquired for this frame,
// which is enough to prove that the frame model and the acquire/present
// semaphores are wired up correctly.
// Everything else (scene, path tracing, effects) is not touched yet.
class NvrhiFrameSkeleton final : public ISwapchainDependency
{
public:
    using PrintFunction = std::function<void(const char *)>;

    explicit NvrhiFrameSkeleton(nvrhi::IDevice *pDevice,
                                const Swapchain *pSwapchain,
                                const char *pShaderFolderPath,
                                PrintFunction pfnPrint);
    ~NvrhiFrameSkeleton() override;

    NvrhiFrameSkeleton(const NvrhiFrameSkeleton &other) = delete;
    NvrhiFrameSkeleton(NvrhiFrameSkeleton &&other) noexcept = delete;
    NvrhiFrameSkeleton &operator=(const NvrhiFrameSkeleton &other) = delete;
    NvrhiFrameSkeleton &operator=(NvrhiFrameSkeleton &&other) noexcept = delete;

    // Called by the swapchain when its images are (re)created or destroyed.
    void OnSwapchainCreate(const Swapchain *pSwapchain) override;
    void OnSwapchainDestroy() override;

    // Records the pass for the image that the swapchain acquired for the current
    // frame and submits it on the RHI graphics queue. 'semaphoreToWait' is the
    // semaphore the swapchain signals on acquire, 'semaphoreToSignal' is the one
    // the presentation engine waits on. Returns false if the pass is unavailable.
    bool Render(const Swapchain *pSwapchain, VkSemaphore semaphoreToWait, VkSemaphore semaphoreToSignal);

    // True if the pass cannot be used at all, so that the caller can avoid
    // taking the frame apart on every frame.
    bool IsUnavailable() const;

private:
    static nvrhi::Format ConvertSurfaceFormat(VkFormat format);

    bool LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result);
    bool CreatePipeline(nvrhi::Format colorFormat);
    bool CreateSwapchainResources(const Swapchain *pSwapchain);
    void DestroySwapchainResources();

    nvrhi::vulkan::IDevice *device;
    PrintFunction print;
    std::string shaderFolderPath;

    nvrhi::ShaderHandle vertexShader;
    nvrhi::ShaderHandle pixelShader;

    // Valid only for the format the pipeline was created with: the swapchain can
    // switch between its formats when it is recreated.
    nvrhi::GraphicsPipelineHandle pipeline;
    nvrhi::Format pipelineColorFormat = nvrhi::Format::UNKNOWN;

    // One entry per swapchain image, indexed by the acquired image index.
    std::vector<nvrhi::TextureHandle> swapchainTextures;
    std::vector<nvrhi::FramebufferHandle> swapchainFramebuffers;

    nvrhi::CommandListHandle commandList;

    // Set when the pass cannot be used at all: the caller then has to keep the
    // old renderer instead of presenting an empty image.
    bool unavailable;
};

}
