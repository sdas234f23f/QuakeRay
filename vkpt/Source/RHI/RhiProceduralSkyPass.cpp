/*
* Copyright (c) 2026 Sultim Tsyrendashiev
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

#include "RhiProceduralSkyPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"

#include "../Utils.h"

#include <algorithm>
#include <cstring>
#include <string>

using namespace vkpt;

namespace
{

// The engine blob, by the file name the shader build writes and ShaderManager.cpp:56 maps
// "CProceduralSky" to. The legacy loads exactly this blob (RenderCubemap.cpp:873).
const char *const SKY_SHADER_FILE_NAME = "CmProceduralSky.comp.spv";

// The workgroup size of the blob's [numthreads(16, 16, 1)] (measured: OpExecutionMode LocalSize
// 16 16 1). The legacy's dispatch divides by it through Utils::GetWorkGroupCount.
constexpr uint32_t SKY_GROUP_SIZE = 16;

// The raw bindings of the blob's only set (CmProceduralSky.comp.hlsl:80-83, measured with
// spirv-dis): the visible cube's UAV at 0, the params constant buffer at 1, the env cube's UAV
// at 2. The layout below sets both the unordered-access and constant-buffer register offsets to 0,
// so an item's slot equals its raw binding number.
constexpr uint32_t SKY_CUBEMAP_UAV_SLOT = 0;
constexpr uint32_t SKY_PARAMS_CB_SLOT = 1;
constexpr uint32_t SKY_ENV_CUBEMAP_UAV_SLOT = 2;

// The extent of one mip of a CUBEMAP_SIZE x CUBEMAP_SIZE face: 1024 >> m for m = 0..10, floored at
// one texel for the last level (the same halving the legacy's blit chain walks,
// RenderCubemap.cpp:259-299).
uint32_t GetMipExtent(uint32_t mipLevel)
{
    return std::max(1u, RhiProceduralSkyPass::CUBEMAP_SIZE >> mipLevel);
}

// One of the two cube images, created once for the whole run (the legacy creates both in its
// constructor and never re-creates them, RenderCubemap.cpp:90-91). The desc is the sampling shape
// the set-8 consumers need (`TextureCube`, 6 layers, 11 mips, RGBA16F) plus the usage bits of the
// engine image (SAMPLED | COLOR_ATTACHMENT | STORAGE | TRANSFER_SRC/DST, RenderCubemap.cpp:501-502)
// and the 2D-array write path's storage usage. isRenderTarget is kept for the future raster sky:
// the engine's image carries COLOR_ATTACHMENT and a later `DrawSkyToCubemap` port writes through
// this same texture instead of re-creating one (see the class comment's seam note).
//
// NonPixelShaderResource is the state every consumer's set-8 Texture_SRV binding requires (the
// layouts' visibility is the RT one, not the pixel one: RhiRtIndirectPass.cpp:446-451 documents the
// same choice for its dummies) and it lowers to SHADER_READ_ONLY_OPTIMAL - exactly the layout the
// legacy leaves the images in between frames. keepInitialState makes every command list start and
// end there (state-tracking.cpp:350-361), so the UAV writes in between are the only transitions the
// module causes.
nvrhi::TextureHandle CreateCubemapTexture(nvrhi::IDevice *device, const char *pDebugName)
{
    nvrhi::TextureDesc desc;
    desc.dimension = nvrhi::TextureDimension::TextureCube;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.width = RhiProceduralSkyPass::CUBEMAP_SIZE;
    desc.height = RhiProceduralSkyPass::CUBEMAP_SIZE;
    desc.arraySize = RhiProceduralSkyPass::CUBEMAP_FACE_COUNT;
    desc.mipLevels = RhiProceduralSkyPass::CUBEMAP_MIP_LEVELS;
    desc.sampleCount = 1;

    desc.isShaderResource = true;
    desc.isRenderTarget = true;
    desc.isUAV = true;

    desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
    desc.keepInitialState = true;

    return rhi::createTexture(device, desc, pDebugName);
}

// One binding set of the blob's set 0, bound for exactly one dispatch: the two images' UAVs at
// mipLevel and the params constant buffer. The UAV items carry the mip as a single-level
// subresource range and an explicit Texture2DArray dimension: that is how the engine's
// `viewArray` shim is expressed in NVRHI - the backend builds an e2DArray VkImageView over the
// same VkImage for this binding, beside the eCube view the sampling consumers bind from the
// texture's own desc (vulkan-texture.cpp:287-349). The explicit dimension is legal for a cube
// texture (validation-device.cpp:1520-1521) and required for the write view: a cube view as a
// storage image would violate the Vulkan view-type rule the engine's TODO(refactor) comment names
// (RenderCubemap.cpp:548-552). The image format is named on the items so the view's format matches
// the shader's `Rgba16f` decorations (measured); leaving it UNKNOWN would use the desc's format,
// which is the same value.
nvrhi::BindingSetHandle CreateSkySet(nvrhi::IDevice *device,
                                    nvrhi::IBindingLayout *pLayout,
                                    nvrhi::ITexture *pCubemap,
                                    nvrhi::ITexture *pEnvironment,
                                    nvrhi::IBuffer *pParamsBuffer,
                                    uint32_t mipLevel)
{
    const nvrhi::TextureSubresourceSet subresources(
        nvrhi::MipLevel(mipLevel), nvrhi::MipLevel(1),
        nvrhi::ArraySlice(0), nvrhi::TextureSubresourceSet::AllArraySlices);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(
        SKY_CUBEMAP_UAV_SLOT, pCubemap, nvrhi::Format::RGBA16_FLOAT, subresources,
        nvrhi::TextureDimension::Texture2DArray));
    setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(SKY_PARAMS_CB_SLOT, pParamsBuffer));
    setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(
        SKY_ENV_CUBEMAP_UAV_SLOT, pEnvironment, nvrhi::Format::RGBA16_FLOAT, subresources,
        nvrhi::TextureDimension::Texture2DArray));

    return device->createBindingSet(setDesc, pLayout);
}

void LogMessage(const RhiProceduralSkyPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiProceduralSkyPass::RhiProceduralSkyPass() = default;

RhiProceduralSkyPass::~RhiProceduralSkyPass()
{
    if (device != nullptr)
    {
        // Everything the module owns was created by NVRHI for the module alone: no engine image is
        // wrapped, no set ever has to be retired, so a device idle is the whole teardown contract
        // (the host destroys the pass while it can still idle the device, like the siblings).
        device->waitForIdle();
    }

    for (uint32_t frameIndex = 0; frameIndex < MAX_FRAMES_IN_FLIGHT; frameIndex++)
    {
        for (uint32_t mipLevel = 0; mipLevel < CUBEMAP_MIP_LEVELS; mipLevel++)
        {
            skySets[frameIndex][mipLevel] = nullptr;
        }

        paramsBuffers[frameIndex] = nullptr;
    }

    skySampler = nullptr;
    environmentTexture = nullptr;
    cubemapTexture = nullptr;
    skyPipeline = nullptr;
    skyLayout = nullptr;
    skyShader = nullptr;
}

bool RhiProceduralSkyPass::Create(nvrhi::IDevice *pDevice,
                                  rhi::RhiFrameContext *pFrameContext,
                                  const char *pShaderFolderPath,
                                  PrintFunction pfnPrint)
{
    if (created)
    {
        return true;
    }

    device = pDevice;
    print = std::move(pfnPrint);
    shaderFolderPath = pShaderFolderPath != nullptr ? pShaderFolderPath : "";
    frameContext = pFrameContext;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the procedural-sky pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the procedural-sky pass needs the frame context of the RHI layer");
        return false;
    }

    skyShader = rhi::loadShader(device, shaderFolderPath + SKY_SHADER_FILE_NAME,
                                nvrhi::ShaderType::Compute, SKY_SHADER_FILE_NAME);
    if (skyShader == nullptr)
    {
        LogMessage(print, std::string("Warning: RHI: cannot load the procedural-sky pass shader \"") +
                              shaderFolderPath + SKY_SHADER_FILE_NAME + "\"");
        return false;
    }

    // Set 0 of the blob, in its raw binding order: the visible cube's UAV at 0, the params constant
    // buffer at 1, the env cube's UAV at 2. Both offsets are 0 so an item's slot is its raw binding
    // (RhiPipeline.h documents the offset rule). The visibility is Compute, the stage the blob runs
    // in; a sampler offset is irrelevant because the set declares no sampler.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setUnorderedAccessViewOffset(0)
                                   .setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(SKY_CUBEMAP_UAV_SLOT));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(SKY_PARAMS_CB_SLOT));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(SKY_ENV_CUBEMAP_UAV_SLOT));

        skyLayout = device->createBindingLayout(desc);
    }

    if (skyLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the procedural-sky pass binding layout");
        return false;
    }

    cubemapTexture = CreateCubemapTexture(device, "RhiProceduralSkyPass cubemap (RGBA16F 1024, 6x11)");
    environmentTexture = CreateCubemapTexture(device, "RhiProceduralSkyPass env cubemap (RGBA16F 1024, 6x11)");

    if (cubemapTexture == nullptr || environmentTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the procedural-sky pass cube images");
        return false;
    }

    // Set 8's samplers (bindings 2 and 3), one object for both, with the legacy's shape
    // (RenderCubemap.cpp:696-697): LINEAR minification, magnification and mip mode
    // (SamplerManager.cpp:34, :89) and REPEAT on all three axes (SamplerManager.cpp:46). The
    // engine's own sampler cannot be wrapped (RhiTextureSource.h:201-215), so the module creates
    // its own; the same shape as `createEngineTextureSampler` except for the non-clamp addressing.
    {
        const nvrhi::SamplerDesc samplerDesc = nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(true)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Repeat);

        skySampler = rhi::createSampler(device, samplerDesc, "RhiProceduralSkyPass sampler (linear, repeat)");
    }

    if (skySampler == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the procedural-sky pass sampler");
        return false;
    }

    // The compute pipeline: the blob declares exactly one descriptor set, so the layout list is one
    // entry and the backend places it at set 0 (vulkan-resource-bindings.cpp:1090-1099). No push
    // constants.
    {
        nvrhi::ComputePipelineDesc desc;
        desc.setComputeShader(skyShader);
        desc.addBindingLayout(skyLayout);

        skyPipeline = rhi::createComputePipeline(device, desc, "RhiProceduralSkyPass compute");
    }

    if (skyPipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the procedural-sky pass compute pipeline");
        return false;
    }

    // The per-slot params buffers and the per-(slot, mip) sets over them. The buffer is the
    // shader's `ConstantBuffer<Params_BT>`: `isConstantBuffer` is what the validation device
    // requires of a ConstantBuffer binding, and the write helper's state contract is
    // keepInitialState = true with initialState = CopyDest (RhiResources.h:31-33), the same pair
    // the god-rays params buffers use. One buffer per engine frame slot, because a shared one could
    // be overwritten while the slot's previous submission still reads it.
    for (uint32_t frameIndex = 0; frameIndex < MAX_FRAMES_IN_FLIGHT; frameIndex++)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = sizeof(Params);
        desc.isConstantBuffer = true;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.keepInitialState = true;

        paramsBuffers[frameIndex] = rhi::createBuffer(
            device, desc, "RhiProceduralSkyPass params " + std::to_string(frameIndex));

        if (paramsBuffers[frameIndex] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create a procedural-sky pass params buffer");
            return false;
        }

        for (uint32_t mipLevel = 0; mipLevel < CUBEMAP_MIP_LEVELS; mipLevel++)
        {
            skySets[frameIndex][mipLevel] = CreateSkySet(device, skyLayout, cubemapTexture,
                                                         environmentTexture, paramsBuffers[frameIndex],
                                                         mipLevel);

            if (skySets[frameIndex][mipLevel] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to create a procedural-sky pass binding set");
                return false;
            }
        }
    }

    created = true;
    return true;
}

void RhiProceduralSkyPass::Render(nvrhi::ICommandList *pCommandList,
                                  uint32_t frameIndex,
                                  const Params &inParams)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return;
    }

    Params params = inParams;

    // Clouds off: freeze the animation time so the cached sky is not re-rendered every frame (only
    // when the sun/sky params change). Clouds on: keep the raw time, so the sky re-renders every
    // frame (RenderCubemap.cpp:896-902).
    if (params.cloudParams[3] <= 0.5f)
    {
        params.cloudColor[3] = 0.0f;
    }

    // No changes since the last dispatch: keep the cached cubemap, exactly like the legacy's memcmp
    // against the mapped params buffer (RenderCubemap.cpp:904-913). lastParams starts zeroed, the
    // state the legacy's buffer is initialized in (RenderCubemap.cpp:746-750).
    if (std::memcmp(&lastParams, &params, sizeof(Params)) == 0)
    {
        return;
    }

    lastParams = params;

    rhi::writeBuffer(pCommandList, paramsBuffers[frameIndex], &params, sizeof(params));

    // One dispatch per mip. Each set names its own single-level storage views, so the shader's
    // GetDimensions returns that mip's extent and its bounds check, direction math and stores all
    // work at that resolution; the z extent is the legacy's face count 6
    // (RenderCubemap.cpp:946). The per-mip UAV bindings also make the automatic barrier pass move
    // exactly the six layers of the mip being written out of the resting read-only layout before
    // its dispatch - the mip-chain equivalent of the legacy's blit barriers (RenderCubemap.cpp:
    // 240-313). The dispatches write disjoint subresources, so no barrier orders one level against
    // another and none has to: each level is produced from the params alone, not from level m-1.
    for (uint32_t mipLevel = 0; mipLevel < CUBEMAP_MIP_LEVELS; mipLevel++)
    {
        nvrhi::ComputeState state;
        state.setPipeline(skyPipeline);
        state.addBindingSet(skySets[frameIndex][mipLevel]);

        pCommandList->setComputeState(state);

        // Utils::GetWorkGroupCount is the legacy's `1 + ceil(size / 16)` (Utils.cpp:319-328), so mip
        // 0 dispatches (65, 65, 6) like the legacy and the shader's bounds check discards the extra
        // workgroup's invocations.
        const uint32_t groups = Utils::GetWorkGroupCount(GetMipExtent(mipLevel), SKY_GROUP_SIZE);

        pCommandList->dispatch(groups, groups, CUBEMAP_FACE_COUNT);
    }

    // The images are written now; put them back into the state the consumers' set-8 SRVs require and
    // the next list starts from (see the state discipline in the class comment). This mirrors the
    // legacy's final SHADER_READ_ONLY_OPTIMAL barrier after the mip chain (RenderCubemap.cpp:
    // 324-329) and makes every later SRV binding on this list a no-op - the consumers bind these
    // very handles, so the tracker sees one texture and no state can conflict.
    pCommandList->setTextureState(cubemapTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::NonPixelShaderResource);
    pCommandList->setTextureState(environmentTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::NonPixelShaderResource);
}
