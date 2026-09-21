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

#include "NvrhiFrameSkeleton.h"

#include <fstream>

#include "../Swapchain.h"

using namespace vkpt;

namespace
{

const char *const VERTEX_SHADER_FILE_NAME = "RhiSkeleton.vert.spv";
const char *const PIXEL_SHADER_FILE_NAME = "RhiSkeleton.frag.spv";

}


NvrhiFrameSkeleton::NvrhiFrameSkeleton(nvrhi::IDevice *pDevice,
                                       const Swapchain *pSwapchain,
                                       const char *pShaderFolderPath,
                                       PrintFunction pfnPrint)
    : device(dynamic_cast<nvrhi::vulkan::IDevice *>(pDevice))
    , print(std::move(pfnPrint))
    , shaderFolderPath(pShaderFolderPath != nullptr ? pShaderFolderPath : "")
    , unavailable(false)
{
    assert(device != nullptr);
    assert(pSwapchain != nullptr);

    if (device == nullptr)
    {
        print("Warning: RHI: the frame skeleton needs the Vulkan backend of the RHI layer");
        unavailable = true;
        return;
    }

    commandList = device->createCommandList();
    if (commandList == nullptr)
    {
        print("Warning: RHI: failed to create the command list of the frame skeleton");
        unavailable = true;
        return;
    }

    if (!LoadShader(VERTEX_SHADER_FILE_NAME, nvrhi::ShaderType::Vertex, vertexShader) ||
        !LoadShader(PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, pixelShader))
    {
        unavailable = true;
        return;
    }

    OnSwapchainCreate(pSwapchain);
}

NvrhiFrameSkeleton::~NvrhiFrameSkeleton()
{
    if (device != nullptr)
    {
        // The textures wrap images that the swapchain owns.
        device->waitForIdle();
    }

    DestroySwapchainResources();

    pipeline = nullptr;
    vertexShader = nullptr;
    pixelShader = nullptr;
    commandList = nullptr;
}

void NvrhiFrameSkeleton::OnSwapchainCreate(const Swapchain *pSwapchain)
{
    if (unavailable)
    {
        return;
    }

    DestroySwapchainResources();

    const nvrhi::Format colorFormat = ConvertSurfaceFormat(pSwapchain->GetSurfaceFormat());
    if (colorFormat == nvrhi::Format::UNKNOWN)
    {
        print("Warning: RHI: the frame skeleton does not support the swapchain format");
        unavailable = true;
        return;
    }

    if (pipeline == nullptr || pipelineColorFormat != colorFormat)
    {
        if (!CreatePipeline(colorFormat))
        {
            unavailable = true;
            return;
        }
    }

    if (!CreateSwapchainResources(pSwapchain))
    {
        unavailable = true;
        return;
    }
}

void NvrhiFrameSkeleton::OnSwapchainDestroy()
{
    if (device != nullptr)
    {
        device->waitForIdle();
    }

    DestroySwapchainResources();
}

bool NvrhiFrameSkeleton::IsUnavailable() const
{
    return unavailable;
}

bool NvrhiFrameSkeleton::Render(const Swapchain *pSwapchain, VkSemaphore semaphoreToWait, VkSemaphore semaphoreToSignal)
{
    if (unavailable)
    {
        return false;
    }

    // A minimized window gives the swapchain a zero extent, and the renderer
    // skips such frames too instead of creating an empty target.
    if (pSwapchain->GetWidth() == 0 || pSwapchain->GetHeight() == 0)
    {
        return false;
    }

    const uint32_t imageIndex = pSwapchain->GetCurrentImageIndex();
    if (imageIndex >= swapchainFramebuffers.size())
    {
        return false;
    }

    nvrhi::ITexture *backBuffer = swapchainTextures[imageIndex];
    nvrhi::IFramebuffer *framebuffer = swapchainFramebuffers[imageIndex];

    commandList->open();

    // The swapchain hands the image over in the present layout and expects it
    // back in the same layout, so the pass is wrapped into two transitions.
    commandList->beginTrackingTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::Present);
    commandList->setTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::RenderTarget);

    nvrhi::GraphicsState state;
    state.pipeline = pipeline;
    state.framebuffer = framebuffer;
    state.viewport.addViewportAndScissorRect(nvrhi::Viewport(
        float(backBuffer->getDesc().width), float(backBuffer->getDesc().height)));

    commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.vertexCount = 3;
    commandList->draw(args);

    commandList->setTextureState(backBuffer, nvrhi::AllSubresources, nvrhi::ResourceStates::Present);
    commandList->close();

    // The image is not available until the acquire semaphore is signalled, and
    // the presentation engine cannot start before the pass is done.
    device->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, semaphoreToWait, 0);
    device->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, semaphoreToSignal, 0);

    device->executeCommandList(commandList, nvrhi::CommandQueue::Graphics);
    device->runGarbageCollection();

    return true;
}

nvrhi::Format NvrhiFrameSkeleton::ConvertSurfaceFormat(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8A8_SRGB:
            return nvrhi::Format::SRGBA8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return nvrhi::Format::SBGRA8_UNORM;
        default:
            return nvrhi::Format::UNKNOWN;
    }
}

bool NvrhiFrameSkeleton::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        print(("Warning: RHI: cannot open the frame skeleton shader \"" + path + "\"").c_str());
        return false;
    }

    const std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> binary(static_cast<size_t>(fileSize));
    if (!file.read(binary.data(), static_cast<std::streamsize>(fileSize)))
    {
        print(("Warning: RHI: cannot read the frame skeleton shader \"" + path + "\"").c_str());
        return false;
    }

    nvrhi::ShaderDesc desc;
    desc.shaderType = type;
    desc.debugName = pFileName;

    result = device->createShader(desc, binary.data(), binary.size());
    if (result == nullptr)
    {
        print(("Warning: RHI: failed to create the frame skeleton shader \"" + path + "\"").c_str());
        return false;
    }

    return true;
}

bool NvrhiFrameSkeleton::CreatePipeline(nvrhi::Format colorFormat)
{
    nvrhi::GraphicsPipelineDesc desc;
    desc.setVertexShader(vertexShader);
    desc.setPixelShader(pixelShader);
    desc.primType = nvrhi::PrimitiveType::TriangleList;

    // The fullscreen triangle is drawn in clip space, without any vertex buffer.
    desc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
    desc.renderState.rasterState.depthClipEnable = true;
    desc.renderState.depthStencilState.disableDepthTest();
    desc.renderState.depthStencilState.disableDepthWrite();

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.addColorFormat(colorFormat);

    pipeline = device->createGraphicsPipeline(desc, framebufferInfo);
    if (pipeline == nullptr)
    {
        print("Warning: RHI: failed to create the frame skeleton pipeline");
        return false;
    }

    pipelineColorFormat = colorFormat;
    return true;
}

bool NvrhiFrameSkeleton::CreateSwapchainResources(const Swapchain *pSwapchain)
{
    nvrhi::TextureDesc textureDesc;
    textureDesc.width = pSwapchain->GetWidth();
    textureDesc.height = pSwapchain->GetHeight();
    textureDesc.dimension = nvrhi::TextureDimension::Texture2D;
    textureDesc.format = pipelineColorFormat;
    textureDesc.mipLevels = 1;
    textureDesc.arraySize = 1;
    textureDesc.sampleCount = 1;
    textureDesc.isRenderTarget = true;
    textureDesc.debugName = "Swapchain image (RHI frame skeleton)";

    for (uint32_t i = 0; i < pSwapchain->GetImageCount(); i++)
    {
        const VkImage image = pSwapchain->GetImage(i);

        const nvrhi::Object nativeImage(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image)));

        nvrhi::TextureHandle texture =
            device->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image, nativeImage, textureDesc);

        if (texture == nullptr)
        {
            print("Warning: RHI: failed to wrap a swapchain image");
            return false;
        }

        nvrhi::FramebufferDesc framebufferDesc;

        framebufferDesc.addColorAttachment(texture);

        nvrhi::FramebufferHandle framebuffer = device->createFramebuffer(framebufferDesc);
        if (framebuffer == nullptr)
        {
            print("Warning: RHI: failed to create a framebuffer for a swapchain image");
            return false;
        }

        swapchainTextures.push_back(std::move(texture));
        swapchainFramebuffers.push_back(std::move(framebuffer));
    }

    return true;
}

void NvrhiFrameSkeleton::DestroySwapchainResources()
{
    swapchainFramebuffers.clear();
    swapchainTextures.clear();
}
