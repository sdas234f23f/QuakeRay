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

#include "RenderCubemap.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "CmdLabel.h"
#include "Matrix.h"
#include "RasterizedDataCollector.h"
#include "Utils.h"
#include "Generated/ShaderCommonC.h"


// high-res HDR cubemap, matching Q2RTX physical sky quality (1024^2, R16G16B16A16_SFLOAT)
constexpr VkFormat CUBEMAP_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT; 
constexpr VkFormat CUBEMAP_DEPTH_FORMAT = VK_FORMAT_D16_UNORM; 
constexpr uint32_t CUBEMAP_SIDE_SIZE = 1024;

// The cloud layer is a volume: what the sky shader samples of it is a
// low-frequency signal (the light that scattered in it, and how much of the sky
// it hides), and it is marched at a step size that already blurs away anything
// finer than this. So its cubemap is a small one, without mips -- and how small it
// is is what the quality level picks.
constexpr uint32_t CLOUDS_SIDE_SIZES[vkpt::RenderCubemap::QUALITY_LEVELS] = { 256, 512, 1024, 2048, 4096 };

// The layer's shadow on the world is a volume of cloud over the ground, read with
// whatever resolution the eye needs from it: a texel of it holds the tau of the
// column of cloud the sun crosses over a spot of the ground, and of the parts of
// that column standing above each height of the layer, so that the sky can ask what
// the sun still crosses to reach a point inside it (CloudShadowMap.h). It is what
// the volumetric sun shafts are gated through as well, so its texels are what the
// edge of a cloud's shadow is drawn with. A level doubles the texels a side over
// the same CLOUD_SHADOW_EXTENT metres of ground -- four times the volume, and four
// times the march filling it -- from a texel every four metres at the bottom of the
// ladder to one every metre at the top. The finest texels stop paying for
// themselves where they are finer than the cone a cloud's light was walked through
// (CLOUD_LIGHT_CONE), which is why the middle levels share a size and so do the two
// above them: what a finer texel would buy there is drawn soft anyway, and every
// texel costs four slices of two bytes.
constexpr uint32_t CLOUD_SHADOW_SIZES[vkpt::RenderCubemap::QUALITY_LEVELS] = { 1024, 2048, 2048, 4096, 4096 };

// The slices the volume holds over the height of the layer, the base of the layer
// in the first and the sky above it in the last (CmCloudShadow.comp fills as many,
// CloudShadowMap.h reads them by the height of a point). They are the only reason
// the shadow is a volume rather than a map, and a few of them are enough: what is
// read between two of them is the light of a cloud, which is a gradient rather than
// a step.
constexpr uint32_t CLOUD_SHADOW_SLICES = 4;
constexpr float    CLOUD_SHADOW_EXTENT = 4000.0f;

// How many steps of the layer's thickness a column is marched in. The march
// finds how much cloud the sun crosses, not where it is, so it neither needs
// the detail erosion nor many steps.
constexpr uint32_t CLOUD_SHADOW_MARCH_STEPS = 16;

// The map is redrawn when the eye has moved a texel over it, and at least this
// often whatever the eye does: the clouds drift on their own, and a stale map
// would hold the shadow still under them. The finer the map, the sooner it is
// filled again -- the two lowest levels redraw it every 4 frames, and above them
// every 2, and every frame from ultra on -- so a level buys the sharpness of the
// edge and the freshness of the shadow together. What the freshness is seen in
// is the shafts and the sunlight under a cloud, which are gated through this map.
constexpr uint32_t CLOUD_SHADOW_REFRESH_FRAMES[vkpt::RenderCubemap::QUALITY_LEVELS] = { 4, 4, 2, 1, 1 };

// Steps the sky pass marches a ray through the layer in, one entry per quality
// level: per view ray, and per sunlight sample taken inside the layer. The sun is
// the dearer of the two -- every step of it is a sample of the shape -- and the
// noisier: what a cloud's light is estimated from is a handful of taps through the
// cone, so even the bottom of the ladder takes six of them. What noise is left is
// finer than white noise would leave, the taps of both marches being spread by
// low-discrepancy sequences (CloudLayer.h), so a step here buys smoothness as much
// as it buys detail.
constexpr uint32_t CLOUDS_VIEW_STEPS[vkpt::RenderCubemap::QUALITY_LEVELS] = { 40, 48, 56, 64, 72 };
constexpr uint32_t CLOUDS_SUN_STEPS[vkpt::RenderCubemap::QUALITY_LEVELS]  = { 6, 6, 6, 8, 8 };

// Below this the sun is under the layer rather than over it, and the layer
// shades nothing that can be
// seen.
constexpr float CLOUD_SHADOW_MIN_SUN_HEIGHT = 0.05f;

// The layer of a frame is marched in a quarter of its map: a dispatch writes one
// quarter of the texels and four frames fill the map. The clouds drift slowly
// enough that a texel of the other three quarters, left from the frames before, is
// the same cloud to any eye that can see it, and the pass then costs a quarter of
// what the whole map would. CmSkyClouds.comp shifts its texels by the same number.
constexpr uint32_t CLOUD_UPDATE_QUARTERS = 2;
constexpr uint32_t CLOUD_UPDATE_FRAMES = CLOUD_UPDATE_QUARTERS * CLOUD_UPDATE_QUARTERS;

// What the volume holds is the tau of a column of cloud, so a single channel is
// all it needs -- and a tau rather than a transmittance because the slices of the
// volume are read interpolated, which only adds up if what stands between two of
// them is linear in the cloud.
constexpr VkFormat CLOUD_SHADOW_FORMAT = VK_FORMAT_R16_SFLOAT;

namespace
{

// Whether two sets of the sky's parameters describe the same look of the cloud
// layer: the sun, the sky that lights the layer, the layer's own colour and body,
// and the march through it. The rest of ProceduralSkyParams is where the frame is
// rather than what is drawn with it -- the time the layer drifts by and the phase of
// the taps taken from it, the eye's own place, the quarter of the map the frame
// marches, and the two fields the map of the layer's shadow rides in -- and a frame
// that differs in those alone is a frame the history of the layer's cubemap holds
// (RenderCubemap::DrawProcedural).
bool SameCloudLook(const vkpt::RenderCubemap::ProceduralSkyParams &a,
                   const vkpt::RenderCubemap::ProceduralSkyParams &b)
{
    const auto same = [](const float *x, const float *y, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (fabsf(x[i] - y[i]) > 1.0e-4f)
            {
                return false;
            }
        }

        return true;
    };

    return same(a.sunDirection, b.sunDirection, 4) &&   // xyz and how much sun the sky shows
           same(a.skyColor, b.skyColor, 3) &&           // w is the shadow map's "there is one" flag
           same(a.skyParams, b.skyParams, 4) &&
           same(a.sunDiscColor, b.sunDiscColor, 3) &&   // w is the extent of that map
           same(a.cloudColor, b.cloudColor, 3) &&       // w is the time the layer drifts by
           same(a.cloudParams, b.cloudParams, 4) &&
           same(a.cloudLayer, b.cloudLayer, 4) &&
           same(a.cloudMarch, b.cloudMarch, 4);
}

}


namespace vkpt
{

struct RasterizedMultiviewPushConst
{
    float model[16];
    float color[4];
    uint32_t textureIndex;

    explicit RasterizedMultiviewPushConst(const RasterizedDataCollector::DrawInfo &info)
    {
        Matrix::ToMat4Transposed(model, info.transform);
        memcpy(color, info.color.Get(), 4 * sizeof(float));
        textureIndex = info.textureIndex;
    }
};

}



vkpt::RenderCubemap::RenderCubemap(
    VkDevice _device, 
    const std::shared_ptr<MemoryAllocator> &_allocator,
    const std::shared_ptr<ShaderManager> &_shaderManager,
    const std::shared_ptr<TextureManager> &_textureManager,
    const std::shared_ptr<GlobalUniform> &_uniform,
    const std::shared_ptr<SamplerManager> &_samplerManager,
    const std::shared_ptr<CommandBufferManager> &_cmdManager,
    const RgInstanceCreateInfo &_instanceInfo)
:
    device(_device),
    allocator(_allocator),
    cmdManager(_cmdManager),
    pipelineLayout(VK_NULL_HANDLE),
    multiviewRenderPass(VK_NULL_HANDLE),
    cubemap{},
    envCubemap{},
    cubemapDepth{},
    cubemapFramebuffer(VK_NULL_HANDLE),
    cubemapSize(CUBEMAP_SIDE_SIZE),
    cubemapMipLevels(static_cast<uint32_t>(std::log2(CUBEMAP_SIDE_SIZE)) + 1),
    descSetLayout(VK_NULL_HANDLE),
    descPool(VK_NULL_HANDLE),
    descSet(VK_NULL_HANDLE),
    clouds{},
    cloudsSize(CLOUDS_SIDE_SIZES[QUALITY_HIGH]),
    cloudShadow{},
    cloudShadowSize(CLOUD_SHADOW_SIZES[QUALITY_HIGH])
{
    CreatePipelineLayout(_textureManager->GetDescSetLayout(), _uniform->GetDescSetLayout());
    CreateRenderPass();
    InitPipelines(_shaderManager, cubemapSize, _instanceInfo.rasterizedVertexColorGamma);

    VkCommandBuffer cmd = _cmdManager->StartGraphicsCmd();
    CreateAttch(_allocator, cmd, cubemapSize, cubemapMipLevels, "Render cubemap", cubemap, false);
    CreateAttch(_allocator, cmd, cubemapSize, cubemapMipLevels, "Render cubemap env", envCubemap, false);
    CreateAttch(_allocator, cmd, cubemapSize, 1, "Render cubemap depth", cubemapDepth, true);
    CreateAttch(_allocator, cmd, cloudsSize, 1, "Cloud cubemap", clouds, false);
    CreateCloudShadowImage(_allocator, cmd, cloudShadowSize, cloudShadow);
    _cmdManager->Submit(cmd);
    _cmdManager->WaitGraphicsIdle();

    CreateFramebuffer(cubemapSize);
    CreateDescriptors(_samplerManager);

    CreateProceduralSkyParamsBuffer();
    CreateProceduralSkyDescriptors(_samplerManager);
    CreateProceduralSkyPipelineLayout();
    CreateProceduralSkyPipeline(_shaderManager.get());
    CreateCloudsPipeline(_shaderManager.get());

    CreateCloudShadowParamsBuffer();
    CreateCloudShadowDescriptors();
    CreateCloudShadowPipelineLayout();
    CreateCloudShadowPipeline(_shaderManager.get());
}

vkpt::RenderCubemap::~RenderCubemap()
{
    if (mappedProcSkyParams)
    {
        procSkyParamsBuffer.TryUnmap();
    }
    procSkyParamsBuffer.Destroy();

    if (mappedCloudShadowParams)
    {
        cloudShadowParamsBuffer.TryUnmap();
    }
    cloudShadowParamsBuffer.Destroy();

    vkDestroyDescriptorPool(device, cloudShadowDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, cloudShadowDescSetLayout, nullptr);
    vkDestroyPipelineLayout(device, cloudShadowPipelineLayout, nullptr);
    DestroyCloudShadowPipeline();

    vkDestroyDescriptorPool(device, procSkyDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, procSkyDescSetLayout, nullptr);
    vkDestroyPipelineLayout(device, procSkyPipelineLayout, nullptr);
    DestroyProceduralSkyPipelines();
    DestroyCloudsPipeline();

    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, multiviewRenderPass, nullptr);

    vkDestroyImage(device, clouds.image, nullptr);
    vkDestroyImageView(device, clouds.view, nullptr);
    vkFreeMemory(device, clouds.memory, nullptr);

    vkDestroyImage(device, cloudShadow.image, nullptr);
    vkDestroyImageView(device, cloudShadow.view, nullptr);
    vkFreeMemory(device, cloudShadow.memory, nullptr);

    vkDestroyImage(device, cubemap.image, nullptr);
    vkDestroyImageView(device, cubemap.view, nullptr);
    vkFreeMemory(device, cubemap.memory, nullptr);

    vkDestroyImage(device, envCubemap.image, nullptr);
    vkDestroyImageView(device, envCubemap.view, nullptr);
    vkFreeMemory(device, envCubemap.memory, nullptr);

    vkDestroyImage(device, cubemapDepth.image, nullptr);
    vkDestroyImageView(device, cubemapDepth.view, nullptr);
    vkFreeMemory(device, cubemapDepth.memory, nullptr);

    vkDestroyFramebuffer(device, cubemapFramebuffer, nullptr);
}

void vkpt::RenderCubemap::OnShaderReload(const ShaderManager *shaderManager)
{
    pipelines->OnShaderReload( shaderManager );
    DestroyProceduralSkyPipelines();
    CreateProceduralSkyPipeline(shaderManager);
    DestroyCloudsPipeline();
    CreateCloudsPipeline(shaderManager);
    DestroyCloudShadowPipeline();
    CreateCloudShadowPipeline(shaderManager);
}

void vkpt::RenderCubemap::Draw(VkCommandBuffer cmd, uint32_t frameIndex,
                                const std::shared_ptr<RasterizedDataCollector> &skyDataCollector,
                                const std::shared_ptr<TextureManager> &textureManager,
                                const std::shared_ptr<GlobalUniform> &uniform)
{
    const auto &drawInfos = skyDataCollector->GetSkyDrawInfos();

    if (drawInfos.empty())
    {
        return;
    }

    VkBuffer vertexBuffer = skyDataCollector->GetVertexBuffer();
    VkBuffer indexBuffer = skyDataCollector->GetIndexBuffer();

    VkDescriptorSet descSets[] =
    {
        textureManager->GetDescSet(frameIndex),
        uniform->GetDescSet(frameIndex),
    };
    const uint32_t descSetCount = sizeof(descSets) / sizeof(descSets[0]);


    VkClearValue clearValues[2] = {};
    clearValues[0].color = {};
    clearValues[1].depthStencil.depth = 1.0f;


    VkRenderPassBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    beginInfo.renderPass = multiviewRenderPass;
    beginInfo.framebuffer = cubemapFramebuffer;
    beginInfo.renderArea.offset = { 0, 0 };
    beginInfo.renderArea.extent = { cubemapSize, cubemapSize };
    beginInfo.clearValueCount = 2;
    beginInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);


    VkPipeline curPipeline = VK_NULL_HANDLE;
    BindPipelineIfNew(cmd, drawInfos[0], curPipeline);


    VkDeviceSize offset = 0;

    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines->GetPipelineLayout(), 0,
        descSetCount, descSets,
        0, nullptr);
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer, offset, VK_INDEX_TYPE_UINT32);


    for (const auto &info : drawInfos)
    {
        BindPipelineIfNew(cmd, info, curPipeline);

        // push const
        {
            RasterizedMultiviewPushConst push(info);

            vkCmdPushConstants(
                cmd, pipelines->GetPipelineLayout(),
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(push),
                &push);
        }

        // draw
        if (info.indexCount > 0)
        {
            vkCmdDrawIndexed(cmd, info.indexCount, 1, info.firstIndex, info.firstVertex, 0);
        }
        else
        {
            vkCmdDraw(cmd, info.vertexCount, 1, info.firstVertex, 0);
        }
    }

    vkCmdEndRenderPass(cmd);

    GenerateMipmaps(cmd, cubemap.image);
}

void vkpt::RenderCubemap::GenerateMipmaps(VkCommandBuffer cmd, VkImage image, VkImageLayout mip0Layout)
{
    if (cubemapMipLevels <= 1)
    {
        return;
    }

    const VkImageSubresourceRange mip0Range =
    {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 6,
    };

    const VkAccessFlags mip0SrcAccess =
        mip0Layout == VK_IMAGE_LAYOUT_GENERAL ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;

    Utils::BarrierImage(
        cmd, image,
        mip0SrcAccess, VK_ACCESS_TRANSFER_READ_BIT,
        mip0Layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        mip0Range);

    uint32_t mipWidth = cubemapSize;
    uint32_t mipHeight = cubemapSize;

    for (uint32_t mipLevel = 1; mipLevel < cubemapMipLevels; mipLevel++)
    {
        const uint32_t prevMipWidth = mipWidth;
        const uint32_t prevMipHeight = mipHeight;

        mipWidth >>= 1;
        mipHeight >>= 1;

        const VkImageSubresourceRange curMipmap =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = mipLevel,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 6,
        };

        Utils::BarrierImage(
            cmd, image,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            curMipmap);

        VkImageBlit curBlit = {};
        curBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        curBlit.srcSubresource.mipLevel = mipLevel - 1;
        curBlit.srcSubresource.baseArrayLayer = 0;
        curBlit.srcSubresource.layerCount = 6;
        curBlit.srcOffsets[0] = { 0, 0, 0 };
        curBlit.srcOffsets[1] = { (int32_t)prevMipWidth, (int32_t)prevMipHeight, 1 };

        curBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        curBlit.dstSubresource.mipLevel = mipLevel;
        curBlit.dstSubresource.baseArrayLayer = 0;
        curBlit.dstSubresource.layerCount = 6;
        curBlit.dstOffsets[0] = { 0, 0, 0 };
        curBlit.dstOffsets[1] = { (int32_t)mipWidth, (int32_t)mipHeight, 1 };

        vkCmdBlitImage(
            cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &curBlit, VK_FILTER_LINEAR);

        Utils::BarrierImage(
            cmd, image,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            curMipmap);
    }

    const VkImageSubresourceRange allMips =
    {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = cubemapMipLevels,
        .baseArrayLayer = 0,
        .layerCount = 6,
    };

    Utils::BarrierImage(
        cmd, image,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        allMips);
}

VkDescriptorSetLayout vkpt::RenderCubemap::GetDescSetLayout() const
{
    return descSetLayout;
}

VkDescriptorSet vkpt::RenderCubemap::GetDescSet() const
{
    return descSet;
}

void vkpt::RenderCubemap::BindPipelineIfNew(VkCommandBuffer cmd, const RasterizedDataCollector::DrawInfo &info, VkPipeline &curPipeline)
{
    pipelines->BindPipelineIfNew(cmd, curPipeline, info.pipelineState, info.blendFuncSrc, info.blendFuncDst);
}


void vkpt::RenderCubemap::CreatePipelineLayout(VkDescriptorSetLayout texturesSetLayout, VkDescriptorSetLayout uniformSetLayout)
{
    VkDescriptorSetLayout setLayouts[] =
    {
        texturesSetLayout,
        uniformSetLayout
    };
    const uint32_t setLayoutCount = sizeof(setLayouts) / sizeof(setLayouts[0]);

    static_assert(sizeof(RasterizedMultiviewPushConst) == 16 * sizeof(float) + 4 * sizeof(float) + sizeof(uint32_t), "");

    VkPushConstantRange pushConst = {};
    pushConst.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConst.offset = 0;
    pushConst.size = sizeof(RasterizedMultiviewPushConst);

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConst;
    layoutInfo.setLayoutCount = setLayoutCount;
    layoutInfo.pSetLayouts = setLayouts;

    VkResult r = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, pipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Render cubemap pipeline layout");
}

void vkpt::RenderCubemap::CreateRenderPass()
{
    const int attchCount = 2;
    VkAttachmentDescription attchs[attchCount] = {};

    auto &colorAttch = attchs[0];
    colorAttch.format = CUBEMAP_FORMAT;
    colorAttch.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttch.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttch.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttch.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttch.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttch.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAttch.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    auto &depthAttch = attchs[1];
    depthAttch.format = CUBEMAP_DEPTH_FORMAT; 
    depthAttch.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttch.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttch.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttch.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttch.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttch.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttch.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


    VkAttachmentReference colorRef = {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef = {};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;


    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;


    // cubemap, 6 faces
    uint32_t viewMask = 0b00111111;
    int32_t viewOffset = 0;

    VkRenderPassMultiviewCreateInfo multiview = {};
    multiview.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
    multiview.subpassCount = 1;
    multiview.pViewMasks = &viewMask;
    multiview.dependencyCount = 1;
    multiview.pViewOffsets = &viewOffset;
    // no correlation between cubemap faces
    multiview.correlationMaskCount = 0;
    multiview.pCorrelationMasks = nullptr;


    VkRenderPassCreateInfo passInfo = {};
    passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    passInfo.pNext = &multiview;
    passInfo.attachmentCount = attchCount;
    passInfo.pAttachments = attchs;
    passInfo.subpassCount = 1;
    passInfo.pSubpasses = &subpass;
    passInfo.dependencyCount = 1;
    passInfo.pDependencies = &dependency;

    VkResult r = vkCreateRenderPass(device, &passInfo, nullptr, &multiviewRenderPass);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, multiviewRenderPass, VK_OBJECT_TYPE_RENDER_PASS, "Render cubemap multiview render pass");
}

void vkpt::RenderCubemap::InitPipelines(const std::shared_ptr<ShaderManager> &shaderManager, uint32_t sideSize, bool applyVertexColorGamma)
{
    VkViewport viewport = {};
    viewport.x = viewport.y = 0;
    viewport.width = viewport.height = (float)sideSize;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissors = {};
    scissors.offset = { 0, 0 };
    scissors.extent = { sideSize, sideSize };


    pipelines = std::make_shared< RasterizerPipelines >( device,
                                                         pipelineLayout,
                                                         multiviewRenderPass,
                                                         shaderManager.get(),
                                                         "VertDefaultMultiview",
                                                         "FragSky",
                                                         0,
                                                         applyVertexColorGamma );
    pipelines->DisableDynamicState(viewport, scissors);
}

void vkpt::RenderCubemap::CreateAttch(
    const std::shared_ptr<MemoryAllocator> &allocator,
    VkCommandBuffer cmd,
    uint32_t sideSize, uint32_t mipLevels, const char *debugName,
    Attachment &result, bool isDepth, bool allowFailure)
{
    char nameBuf[128];
    // One name at a time: the buffer is reused, so a name is only valid until the
    // next call (which is all a debug name needs to be).
    const auto name = [debugName, &nameBuf](const char *suffix) -> const char *
    {
        snprintf(nameBuf, sizeof(nameBuf), "%s %s", debugName, suffix);
        return nameBuf;
    };

    // A map that could not be made leaves nothing behind, so the caller that may go
    // on without it (a quality level that was asked for) keeps the map it had.
    const auto giveUp = [this, &result]()
    {
        if (result.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, result.view, nullptr);
        }
        if (result.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device, result.image, nullptr);
        }
        if (result.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, result.memory, nullptr);
        }

        result = {};
    };

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.format = isDepth ? CUBEMAP_DEPTH_FORMAT : CUBEMAP_FORMAT;
    imageInfo.extent = { sideSize, sideSize, 1 };
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 6;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = isDepth ?
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult r = vkCreateImage(device, &imageInfo, nullptr, &result.image);
    if (r != VK_SUCCESS)
    {
        // Whether the map may be given up on is the caller's to say.
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }
    SET_DEBUG_NAME(device, result.image, VK_OBJECT_TYPE_IMAGE, name(isDepth ? "depth image" : "image"));


    // allocate dedicated memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, result.image, &memReqs);

    result.memory = allowFailure ?
        allocator->TryAllocDedicated(memReqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MemoryAllocator::AllocType::DEFAULT, name(isDepth ? "depth image memory" : "image memory")) :
        allocator->AllocDedicated(memReqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MemoryAllocator::AllocType::DEFAULT, name(isDepth ? "depth image memory" : "image memory"));

    if (result.memory == VK_NULL_HANDLE)
    {
        giveUp();

        return;
    }

    r = vkBindImageMemory(device, result.image, result.memory, 0);
    if (r != VK_SUCCESS)
    {
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }


    // create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = isDepth ? CUBEMAP_DEPTH_FORMAT : CUBEMAP_FORMAT;
    viewInfo.subresourceRange = {};
    viewInfo.subresourceRange.aspectMask = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;
    viewInfo.image = result.image;

    r = vkCreateImageView(device, &viewInfo, nullptr, &result.view);
    if (r != VK_SUCCESS)
    {
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }
    SET_DEBUG_NAME(device, result.view, VK_OBJECT_TYPE_IMAGE_VIEW, name(isDepth ? "depth image view" : "image view"));


    // make transition from undefined manually, so initialLayout can be specified
    VkImageMemoryBarrier imageBarrier = {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.image = result.image;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.srcAccessMask = 0;
    if (isDepth)
    {
        imageBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    else
    {
        imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = mipLevels;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 6;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &imageBarrier);
}

void vkpt::RenderCubemap::CreateFramebuffer(uint32_t sideSize)
{
    if (cubemap.image == VK_NULL_HANDLE || cubemap.view == VK_NULL_HANDLE ||
        cubemapDepth.image == VK_NULL_HANDLE || cubemapDepth.view == VK_NULL_HANDLE)
    {
        return;
    }

    VkImageView attchs[] =
    {
        cubemap.view,
        cubemapDepth.view,
    };

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = multiviewRenderPass;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = attchs;
    fbInfo.width = sideSize;
    fbInfo.height = sideSize;
    fbInfo.layers = 1;

    VkResult r = vkCreateFramebuffer(device, &fbInfo, nullptr, &cubemapFramebuffer);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cubemapFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER, "Render cubemap framebuffer");
}

void vkpt::RenderCubemap::CreateDescriptors(const std::shared_ptr<SamplerManager> &samplerManager)
{
    VkDescriptorSetLayoutBinding bindings[3] = {};

    bindings[0].binding = BINDING_RENDER_CUBEMAP;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = BINDING_RENDER_CUBEMAP_ENV;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    // The cloud layer's shadow on the world, read by the passes that light it
    // (CloudShadowMap.h).
    bindings[2].binding = BINDING_RENDER_CUBEMAP_CLOUD_SHADOW;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkResult r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Render cubemap Desc set layout");


    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Render cubemap Desc pool");


    VkDescriptorSetAllocateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = descPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &descSetLayout;

    r = vkAllocateDescriptorSets(device, &setInfo, &descSet);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, descSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, "Render cubemap desc set");


    VkDescriptorImageInfo img = {};
    img.sampler = samplerManager->GetSampler(RG_SAMPLER_FILTER_LINEAR, RG_SAMPLER_ADDRESS_MODE_REPEAT, RG_SAMPLER_ADDRESS_MODE_REPEAT);
    img.imageView = cubemap.view;
    img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo envImg = img;
    envImg.imageView = envCubemap.view;

    VkDescriptorImageInfo cloudShadowImg = {};
    cloudShadowSampler = samplerManager->GetSampler(RG_SAMPLER_FILTER_LINEAR, RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    cloudShadowImg.sampler = cloudShadowSampler;
    cloudShadowImg.imageView = cloudShadow.view;
    cloudShadowImg.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wrt[3] = {};
    wrt[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrt[0].dstSet = descSet;
    wrt[0].dstBinding = BINDING_RENDER_CUBEMAP;
    wrt[0].dstArrayElement = 0;
    wrt[0].descriptorCount = 1;
    wrt[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wrt[0].pImageInfo = &img;

    wrt[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrt[1].dstSet = descSet;
    wrt[1].dstBinding = BINDING_RENDER_CUBEMAP_ENV;
    wrt[1].dstArrayElement = 0;
    wrt[1].descriptorCount = 1;
    wrt[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wrt[1].pImageInfo = &envImg;

    wrt[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrt[2].dstSet = descSet;
    wrt[2].dstBinding = BINDING_RENDER_CUBEMAP_CLOUD_SHADOW;
    wrt[2].dstArrayElement = 0;
    wrt[2].descriptorCount = 1;
    wrt[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wrt[2].pImageInfo = &cloudShadowImg;

    vkUpdateDescriptorSets(device, 3, wrt, 0, nullptr);
}

void vkpt::RenderCubemap::CreateProceduralSkyParamsBuffer()
{
    procSkyParamsBuffer.Init(
        allocator,
        sizeof(ProceduralSkyParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        "Procedural sky params buffer");

    mappedProcSkyParams = procSkyParamsBuffer.Map();
    if (mappedProcSkyParams)
    {
        memset(mappedProcSkyParams, 0, sizeof(ProceduralSkyParams));
    }
}

void vkpt::RenderCubemap::CreateProceduralSkyDescriptors(const std::shared_ptr<SamplerManager> &samplerManager)
{
    VkResult r;

    VkDescriptorSetLayoutBinding bindings[7] = {};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 1: params buffer
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 3: the cloud layer, written by CSkyClouds and read by CProceduralSky
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 4: the same cloud layer, sampled by direction instead of written
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 5: the map of the layer's shadow on the world, read by the cloud pass itself
    // so that the light of a cloud is a lookup and not a march to the sun
    // (CloudShadowMap.h)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // 6: that same cubemap of the layer again, in the layout the pass that fills it
    // writes it in, read back as its own history (CmSkyClouds.comp)
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &procSkyDescSetLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, procSkyDescSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Procedural sky desc set layout");

    VkDescriptorPoolSize poolSizes[3] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 3;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &procSkyDescPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, procSkyDescPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Procedural sky desc pool");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = procSkyDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &procSkyDescSetLayout;

    r = vkAllocateDescriptorSets(device, &allocInfo, &procSkyDescSet);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, procSkyDescSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, "Procedural sky desc set");

    VkDescriptorImageInfo imgInfo = {};
    imgInfo.imageView = cubemap.view;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo envImgInfo = {};
    envImgInfo.imageView = envCubemap.view;
    envImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo cloudsImgInfo = {};
    cloudsImgInfo.imageView = clouds.view;
    cloudsImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo cloudsSampledInfo = {};
    cloudsSampler = samplerManager->GetSampler(RG_SAMPLER_FILTER_LINEAR, RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    cloudsSampledInfo.sampler = cloudsSampler;
    cloudsSampledInfo.imageView = clouds.view;
    cloudsSampledInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // The same cubemap again, in the layout the cloud pass writes it in and reads its
    // own history in (binding 6), where binding 4 is it in the layout the sky reads.
    VkDescriptorImageInfo cloudsHistoryInfo = {};
    cloudsHistoryInfo.sampler = cloudsSampler;
    cloudsHistoryInfo.imageView = clouds.view;
    cloudsHistoryInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // The map of the layer's shadow, which is made before this set is (see the
    // constructor) and is read by the cloud pass through it.
    VkDescriptorImageInfo cloudShadowImgInfo = {};
    cloudShadowImgInfo.sampler = cloudShadowSampler;
    cloudShadowImgInfo.imageView = cloudShadow.view;
    cloudShadowImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = procSkyParamsBuffer.GetBuffer();
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[7] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = procSkyDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &imgInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = procSkyDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &bufInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = procSkyDescSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &envImgInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = procSkyDescSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo = &cloudsImgInfo;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = procSkyDescSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &cloudsSampledInfo;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = procSkyDescSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &cloudShadowImgInfo;

    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = procSkyDescSet;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &cloudsHistoryInfo;

    vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);
}

void vkpt::RenderCubemap::CreateProceduralSkyPipelineLayout()
{
    VkResult r;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &procSkyDescSetLayout;

    r = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &procSkyPipelineLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, procSkyPipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Procedural sky pipeline layout");
}

void vkpt::RenderCubemap::CreateProceduralSkyPipeline(const ShaderManager *shaderManager)
{
    VkResult r;

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = procSkyPipelineLayout;
    pipelineInfo.stage = shaderManager->GetStageInfo("CProceduralSky");

    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &procSkyPipeline);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, procSkyPipeline, VK_OBJECT_TYPE_PIPELINE, "Procedural sky pipeline");
}

void vkpt::RenderCubemap::DestroyProceduralSkyPipelines()
{
    if (procSkyPipeline)
    {
        vkDestroyPipeline(device, procSkyPipeline, nullptr);
        procSkyPipeline = VK_NULL_HANDLE;
    }
}

void vkpt::RenderCubemap::CreateCloudsPipeline(const ShaderManager *shaderManager)
{
    VkResult r;

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = procSkyPipelineLayout;
    pipelineInfo.stage = shaderManager->GetStageInfo("CSkyClouds");

    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cloudsPipeline);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudsPipeline, VK_OBJECT_TYPE_PIPELINE, "Clouds pipeline");
}

void vkpt::RenderCubemap::DestroyCloudsPipeline()
{
    if (cloudsPipeline)
    {
        vkDestroyPipeline(device, cloudsPipeline, nullptr);
        cloudsPipeline = VK_NULL_HANDLE;
    }
}

void vkpt::RenderCubemap::DispatchClouds(VkCommandBuffer cmd, const ProceduralSkyParams &params)
{
    // Nothing to march when the host turned the clouds off or made them fully
    // transparent. The cubemap keeps whatever it holds: no shader reads it then --
    // and the next time the layer is drawn, all of it is filled at once.
    if (params.cloudParams[3] <= 0.5f || params.skyParams[1] <= 0.0f)
    {
        cloudsFullUpdate = true;

        return;
    }

    CmdLabel label(cmd, "Cloud layer");

    // The cloud layer and the sky it is composited into are dispatched back to
    // back in the same layout, so a write of one is the read of the other.
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = clouds.image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 6;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, procSkyPipelineLayout,
                            0, 1, &procSkyDescSet, 0, nullptr);

    // A quarter of the map this frame, the next quarter the next: four frames fill
    // it, and the frame a map was made in fills it whole, so a new map never shows
    // the texels of the one it replaced.
    const uint32_t quarters = cloudsFullUpdate ? 1 : CLOUD_UPDATE_QUARTERS;
    const uint32_t wg = Utils::GetWorkGroupCount(cloudsSize / quarters, 16);
    vkCmdDispatch(cmd, wg, wg, 6);

    cloudsCycle = cloudsFullUpdate ? 0 : (cloudsCycle + 1) % CLOUD_UPDATE_FRAMES;
    cloudsFullUpdate = false;

    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

void vkpt::RenderCubemap::CreateCloudShadowImage(const std::shared_ptr<MemoryAllocator> &allocator, VkCommandBuffer cmd,
                                                 uint32_t size, Attachment &image, bool allowFailure)
{
    // A map that could not be made leaves nothing behind, so the caller that may go
    // on without it (a quality level that was asked for) keeps the map it had.
    const auto giveUp = [this, &image]()
    {
        if (image.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, image.view, nullptr);
        }
        if (image.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device, image.image, nullptr);
        }
        if (image.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, image.memory, nullptr);
        }

        image = {};
    };

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = CLOUD_SHADOW_FORMAT;
    imageInfo.extent = { size, size, CLOUD_SHADOW_SLICES };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult r = vkCreateImage(device, &imageInfo, nullptr, &image.image);
    if (r != VK_SUCCESS)
    {
        // Whether the map may be given up on is the caller's to say.
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }
    SET_DEBUG_NAME(device, image.image, VK_OBJECT_TYPE_IMAGE, "Cloud shadow image");

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, image.image, &memReqs);

    image.memory = allowFailure ?
        allocator->TryAllocDedicated(memReqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     MemoryAllocator::AllocType::DEFAULT, "Cloud shadow image memory") :
        allocator->AllocDedicated(memReqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                  MemoryAllocator::AllocType::DEFAULT, "Cloud shadow image memory");
    if (image.memory == VK_NULL_HANDLE)
    {
        giveUp();

        return;
    }

    r = vkBindImageMemory(device, image.image, image.memory, 0);
    if (r != VK_SUCCESS)
    {
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = CLOUD_SHADOW_FORMAT;
    viewInfo.subresourceRange = {};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.image = image.image;

    r = vkCreateImageView(device, &viewInfo, nullptr, &image.view);
    if (r != VK_SUCCESS)
    {
        if (!allowFailure)
        {
            VK_CHECKERROR(r);
        }

        giveUp();

        return;
    }
    SET_DEBUG_NAME(device, image.view, VK_OBJECT_TYPE_IMAGE_VIEW, "Cloud shadow image view");

    // Written as a storage image and read as a sampled one, so it settles in
    // shader-read and the dispatch that fills it moves it to GENERAL and back.
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image.image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

void vkpt::RenderCubemap::CreateCloudShadowParamsBuffer()
{
    cloudShadowParamsBuffer.Init(
        allocator,
        sizeof(CloudShadowParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        "Cloud shadow params buffer");

    mappedCloudShadowParams = cloudShadowParamsBuffer.Map();
    if (mappedCloudShadowParams)
    {
        memset(mappedCloudShadowParams, 0, sizeof(CloudShadowParams));
    }
}

void vkpt::RenderCubemap::CreateCloudShadowDescriptors()
{
    VkResult r;

    VkDescriptorSetLayoutBinding bindings[2] = {};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    r = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cloudShadowDescSetLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudShadowDescSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "Cloud shadow desc set layout");

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    r = vkCreateDescriptorPool(device, &poolInfo, nullptr, &cloudShadowDescPool);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudShadowDescPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Cloud shadow desc pool");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = cloudShadowDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &cloudShadowDescSetLayout;

    r = vkAllocateDescriptorSets(device, &allocInfo, &cloudShadowDescSet);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudShadowDescSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, "Cloud shadow desc set");

    VkDescriptorImageInfo imgInfo = {};
    imgInfo.imageView = cloudShadow.view;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = cloudShadowParamsBuffer.GetBuffer();
    bufInfo.offset = 0;
    bufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = cloudShadowDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &imgInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = cloudShadowDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void vkpt::RenderCubemap::CreateCloudShadowPipelineLayout()
{
    VkResult r;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &cloudShadowDescSetLayout;

    r = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &cloudShadowPipelineLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudShadowPipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Cloud shadow pipeline layout");
}

void vkpt::RenderCubemap::CreateCloudShadowPipeline(const ShaderManager *shaderManager)
{
    VkResult r;

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = cloudShadowPipelineLayout;
    pipelineInfo.stage = shaderManager->GetStageInfo("CCloudShadow");

    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cloudShadowPipeline);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, cloudShadowPipeline, VK_OBJECT_TYPE_PIPELINE, "Cloud shadow pipeline");
}

void vkpt::RenderCubemap::DestroyCloudShadowPipeline()
{
    if (cloudShadowPipeline)
    {
        vkDestroyPipeline(device, cloudShadowPipeline, nullptr);
        cloudShadowPipeline = VK_NULL_HANDLE;
    }
}

void vkpt::RenderCubemap::DispatchCloudShadow(VkCommandBuffer cmd)
{
    CmdLabel label(cmd, "Cloud shadow");

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = cloudShadow.image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudShadowPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudShadowPipelineLayout,
                            0, 1, &cloudShadowDescSet, 0, nullptr);

    const uint32_t wg = Utils::GetWorkGroupCount(cloudShadowSize, 16);
    vkCmdDispatch(cmd, wg, wg, 1);

    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

void vkpt::RenderCubemap::GetCloudShadowPlacement(float placement[4]) const
{
    for (uint32_t i = 0; i < 4; i++)
    {
        placement[i] = cloudShadowValid ? cloudShadowPlacement[i] : 0.0f;
    }
}

void vkpt::RenderCubemap::InvalidateCloudShadow()
{
    cloudShadowValid = false;
    cloudShadowAge = 0;
}

void vkpt::RenderCubemap::UpdateCloudShadow(VkCommandBuffer cmd, const ProceduralSkyParams &params,
                                            const float cameraPos[3])
{
    // The layer casts nothing when the clouds are off, when the sky does not draw
    // them, when the host keeps no sun for them to hide, or when the sun is so low
    // that the layer is behind the horizon.
    if (params.cloudParams[3] <= 0.5f || params.skyParams[1] <= 0.0f ||
        params.sunDirection[3] <= 0.5f ||
        params.sunDirection[2] <= CLOUD_SHADOW_MIN_SUN_HEIGHT ||
        !mappedCloudShadowParams || cloudShadow.view == VK_NULL_HANDLE ||
        cloudShadowSize == 0 || cloudShadowPipeline == VK_NULL_HANDLE)
    {
        InvalidateCloudShadow();

        return;
    }

    const float extent = CLOUD_SHADOW_EXTENT;
    const float texelSize = extent / float(cloudShadowSize);

    // The map is laid out over the world's horizontal plane around the eye,
    // snapped to whole texels: an eye that has not moved a texel over the map keeps
    // the map it has, and when it has, the redrawn map still puts a texel where it
    // was.
    const float anchor[2] =
    {
        floorf((cameraPos[0] - extent * 0.5f) / texelSize) * texelSize,
        floorf((cameraPos[1] - extent * 0.5f) / texelSize) * texelSize,
    };

    const auto differs = [](float a, float b) { return fabsf(a - b) > 1.0e-4f; };

    // A map older than a few frames holds the drifting clouds still, and one the
    // eye has walked off is looking at the wrong ground, so both are redrawn. A
    // layer the host has just changed, or a sun that has moved, has to show at
    // once rather than on the next tick of that clock.
    const CloudShadowParams &standing = cloudShadowParams;
    const bool stale =
        !cloudShadowValid ||
        cloudShadowAge + 1 >= CloudShadowRefreshFrames() ||
        differs(standing.sunDirection[0], params.sunDirection[0]) ||
        differs(standing.sunDirection[1], params.sunDirection[1]) ||
        differs(standing.sunDirection[2], params.sunDirection[2]) ||
        differs(standing.sunDirection[3], params.cloudLayer[0]) ||
        differs(standing.cloudLayer[0], params.cloudLayer[1]) ||
        differs(standing.cloudLayer[1], params.cloudParams[0]) ||
        differs(standing.cloudLayer[2], params.cloudParams[1]) ||
        differs(standing.cloudLayer[3], params.cloudMarch[2]) ||
        differs(standing.cloudMarch[1], params.cloudParams[2]) ||
        differs(standing.cloudMarch[3], cameraPos[2] + params.cloudLayer[0]) ||
        differs(standing.mapProjection[0], anchor[0]) ||
        differs(standing.mapProjection[1], anchor[1]);

    if (!stale)
    {
        cloudShadowAge++;

        return;
    }

    cloudShadowParams.sunDirection[0] = params.sunDirection[0];
    cloudShadowParams.sunDirection[1] = params.sunDirection[1];
    cloudShadowParams.sunDirection[2] = params.sunDirection[2];
    // The layer is held over the eye, which is how the sky marches it (its bottom
    // is that high above the eye), while the map itself is keyed on the world: the
    // two meet in cloudMarch[3] below.
    cloudShadowParams.sunDirection[3] = params.cloudLayer[0];
    cloudShadowParams.cloudLayer[0] = params.cloudLayer[1];
    cloudShadowParams.cloudLayer[1] = params.cloudParams[0];
    cloudShadowParams.cloudLayer[2] = params.cloudParams[1];
    cloudShadowParams.cloudLayer[3] = params.cloudMarch[2];
    cloudShadowParams.cloudMarch[0] = params.cloudColor[3];
    cloudShadowParams.cloudMarch[1] = params.cloudParams[2];
    cloudShadowParams.cloudMarch[2] = float(CLOUD_SHADOW_MARCH_STEPS);
    // Where the base of the layer stands over the plane the map is keyed on, which
    // is what a spot of that plane has to be moved back along the sun by to reach
    // the cloud that shades it.
    cloudShadowParams.cloudMarch[3] = cameraPos[2] + params.cloudLayer[0];
    cloudShadowParams.mapProjection[0] = anchor[0];
    cloudShadowParams.mapProjection[1] = anchor[1];
    cloudShadowParams.mapProjection[2] = extent;
    cloudShadowParams.mapProjection[3] = float(cloudShadowSize);

    memcpy(mappedCloudShadowParams, &cloudShadowParams, sizeof(CloudShadowParams));

    DispatchCloudShadow(cmd);

    cloudShadowValid = true;
    cloudShadowAge = 0;
    cloudShadowPlacement[0] = 1.0f;
    cloudShadowPlacement[1] = anchor[0];
    cloudShadowPlacement[2] = anchor[1];
    cloudShadowPlacement[3] = extent;
}

uint32_t vkpt::RenderCubemap::ClampQuality(uint32_t quality)
{
    return quality > QUALITY_EXTREME ? QUALITY_EXTREME : quality;
}

uint32_t vkpt::RenderCubemap::CloudShadowRefreshFrames() const
{
    return CLOUD_SHADOW_REFRESH_FRAMES[quality];
}

void vkpt::RenderCubemap::SetQuality(VkCommandBuffer cmd, uint32_t newQuality)
{
    newQuality = ClampQuality(newQuality);

    if (newQuality == quality)
    {
        // The level in use is the level asked for, so nothing is waiting to be
        // taken.
        failedQuality = QUALITY_LEVELS;
        qualityRetryAge = 0;

        return;
    }

    // A level whose maps could not be made is not asked for again on every frame
    // (see failedQuality): once in a while is enough for memory freed elsewhere to
    // bring it up, and until then the attempt would only cost.
    if (newQuality == failedQuality)
    {
        qualityRetryAge++;

        if (qualityRetryAge < QUALITY_RETRY_FRAMES)
        {
            return;
        }
    }
    else
    {
        qualityRetryAge = 0;
    }

    // The maps are named by descriptor sets and read by the passes of the frame
    // being recorded and of the frames still in flight behind it, so the ones
    // being replaced may only go once the queue has run dry. What replaces them
    // is written into the command buffer of the frame in hand, which is still
    // being recorded and orders itself after those frames by itself.
    cmdManager->WaitGraphicsIdle();

    if (CLOUDS_SIDE_SIZES[newQuality] != cloudsSize)
    {
        Attachment newClouds = {};
        CreateAttch(allocator, cmd, CLOUDS_SIDE_SIZES[newQuality], 1, "Cloud cubemap", newClouds, false, true);

        // A map that could not be made (out of memory) leaves the one in hand as
        // the one the layer is drawn into, and the level is asked for again later
        // (see failedQuality).
        if (newClouds.image == VK_NULL_HANDLE)
        {
            failedQuality = newQuality;
            qualityRetryAge = 0;

            return;
        }

        vkDestroyImage(device, clouds.image, nullptr);
        vkDestroyImageView(device, clouds.view, nullptr);
        vkFreeMemory(device, clouds.memory, nullptr);

        clouds = newClouds;
        cloudsSize = CLOUDS_SIDE_SIZES[newQuality];

        // The map is a new one: all of it is filled at once rather than a quarter,
        // and what the old map held is not in it.
        cloudsFullUpdate = true;

        // The layer that was drawn into the old map went with it, so the sky pass
        // draws it into the new one this frame, whatever the host has asked of it
        // since -- even nothing at all.
        if (mappedProcSkyParams)
        {
            memset(mappedProcSkyParams, 0, sizeof(ProceduralSkyParams));
        }
    }

    if (CLOUD_SHADOW_SIZES[newQuality] != cloudShadowSize)
    {
        Attachment newShadow = {};
        CreateCloudShadowImage(allocator, cmd, CLOUD_SHADOW_SIZES[newQuality], newShadow, true);

        // A map that could not be made (out of memory) leaves the one in hand as
        // the one the shadow is drawn into, and the level is asked for again later
        // (see failedQuality). The cubemap above may already be the new one, and
        // the descriptors have to name what is here rather than what was asked for.
        if (newShadow.image == VK_NULL_HANDLE)
        {
            UpdateQualityDescriptors();

            failedQuality = newQuality;
            qualityRetryAge = 0;

            return;
        }

        vkDestroyImage(device, cloudShadow.image, nullptr);
        vkDestroyImageView(device, cloudShadow.view, nullptr);
        vkFreeMemory(device, cloudShadow.memory, nullptr);

        cloudShadow = newShadow;
        cloudShadowSize = CLOUD_SHADOW_SIZES[newQuality];
    }

    // Nothing of what either map held is in the new ones.
    cloudShadowValid = false;
    cloudShadowAge = 0;

    UpdateQualityDescriptors();

    quality = newQuality;
    failedQuality = QUALITY_LEVELS;
    qualityRetryAge = 0;
}

void vkpt::RenderCubemap::UpdateQualityDescriptors()
{
    VkDescriptorImageInfo cloudsWritten = {};
    cloudsWritten.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudsWritten.imageView = clouds.view;

    VkDescriptorImageInfo cloudsSampled = {};
    cloudsSampled.sampler = cloudsSampler;
    cloudsSampled.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    cloudsSampled.imageView = clouds.view;

    // The same layer as the pass that fills it reads its own history: the layout it
    // writes in, which is the layout it reads in (binding 6).
    VkDescriptorImageInfo cloudsHistory = {};
    cloudsHistory.sampler = cloudsSampler;
    cloudsHistory.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    cloudsHistory.imageView = clouds.view;

    VkDescriptorImageInfo shadowSampled = {};
    shadowSampled.sampler = cloudShadowSampler;
    shadowSampled.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadowSampled.imageView = cloudShadow.view;

    // The map as the compute pass that fills it names it. Without this the dispatch
    // would keep writing into the map the level replaced, which is gone.
    VkDescriptorImageInfo shadowWritten = {};
    shadowWritten.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    shadowWritten.imageView = cloudShadow.view;

    // The two bindings of the sky pass that name the layer (the one it is drawn
    // into and the one it is sampled through), the binding of the cubemap set that
    // names the map of its shadow, the storage image of the pass that fills it, the
    // binding the cloud pass itself reads the map through, and the binding it reads
    // its own history through.
    VkWriteDescriptorSet writes[6] = {};

    for (VkWriteDescriptorSet &write : writes)
    {
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }

    writes[0].dstSet = procSkyDescSet;
    writes[0].dstBinding = 3;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &cloudsWritten;

    writes[1].dstSet = procSkyDescSet;
    writes[1].dstBinding = 4;
    writes[1].pImageInfo = &cloudsSampled;

    writes[2].dstSet = descSet;
    writes[2].dstBinding = BINDING_RENDER_CUBEMAP_CLOUD_SHADOW;
    writes[2].pImageInfo = &shadowSampled;

    writes[3].dstSet = cloudShadowDescSet;
    writes[3].dstBinding = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo = &shadowWritten;

    writes[4].dstSet = procSkyDescSet;
    writes[4].dstBinding = 5;
    writes[4].pImageInfo = &shadowSampled;

    writes[5].dstSet = procSkyDescSet;
    writes[5].dstBinding = 6;
    writes[5].pImageInfo = &cloudsHistory;

    vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
}

void vkpt::RenderCubemap::DrawProcedural(VkCommandBuffer cmd, const ProceduralSkyParams &inParams)
{
    CmdLabel label(cmd, "Procedural sky");

    ProceduralSkyParams params = inParams;

    // The layer is what the host asks for; how finely it is drawn is what the
    // quality level decides, and the march is part of that: the map doubles with
    // the level, so the march has to resolve what the finer map can hold.
    params.cloudMarch[0] = float(CLOUDS_VIEW_STEPS[quality]);
    params.cloudMarch[1] = float(CLOUDS_SUN_STEPS[quality]);

    // A frame the look of the layer changed in is not a frame the history of its
    // cubemap can be averaged into: the layer, the sky that lights it and the sun are
    // what is being drawn, and nothing of the frames before them is that. Such a
    // frame fills the whole map rather than a quarter of it, and the pass reads no
    // history in it. The sun editor moves the sun every frame it is on, and a sun a
    // tenth of a second late is a look of its own (SameCloudLook above says what the
    // look is).
    if (!SameCloudLook(procSkyLook, params))
    {
        cloudsFullUpdate = true;
    }

    procSkyLook = params;

    // What the pass reads the history of the layer through: the frames before this
    // one drew the same texels of the cubemap, and the eye moving over the world is
    // what makes a texel stand for another column of cloud than it did -- the layer
    // is anchored in the world's plane, so the point of it a direction names moves
    // with where the eye stands. Filled before the anchor is frozen below, so that a
    // frame with the clouds off still carries the eye's real movement.
    const float anchor[2] = { params.cloudAnchor[0], params.cloudAnchor[1] };
    params.cloudAnchorDelta[0] = anchor[0] - cloudAnchorPrev[0];
    params.cloudAnchorDelta[1] = anchor[1] - cloudAnchorPrev[1];
    params.cloudAnchorDelta[2] = 0.0f;
    params.cloudAnchorDelta[3] = 0.0f;
    cloudAnchorPrev[0] = anchor[0];
    cloudAnchorPrev[1] = anchor[1];

    // Clouds off: freeze the animation time so the cached sky isn't re-rendered
    // every frame (only when sun/sky params change).
    // Clouds on: keep the raw time -> the sky re-renders every frame (smooth per-frame drift).
    // Nothing can be seen drifting when the clouds are disabled or their opacity
    // is 0 (rt_sky_clouds_alpha), so the time is frozen in either case. The anchor
    // goes with it: with no clouds drawn, the eye moving over the world is no
    // reason to redraw.
    if (params.cloudParams[3] <= 0.5f || params.skyParams[1] <= 0.0f)
    {
        params.cloudColor[3] = 0.0f;
        params.cloudAnchor[0] = 0.0f;
        params.cloudAnchor[1] = 0.0f;
    }
    else
    {
        // Which quarter of the layer's map this frame marches, CLOUD_UPDATE_FRAMES
        // meaning the map is new and has to be filled whole (DispatchClouds); the
        // eye's own height in the world stays where the host put it, in
        // cloudAnchor.z, because the map of the layer's shadow is read with it. What
        // that map stands on rides in the two fields the sky itself does not read:
        // skyColor.w is 1 while there is a map to read, sunDiscColor.w its extent in
        // metres. All of them are part of the params the cache below compares, so the
        // layer is never remembered away while it is being drawn.
        params.cloudAnchor[3] = cloudsFullUpdate ? float(CLOUD_UPDATE_FRAMES) : float(cloudsCycle);
        params.skyColor[3] = cloudShadowValid ? 1.0f : 0.0f;
        params.sunDiscColor[3] = CLOUD_SHADOW_EXTENT;
    }

    if (mappedProcSkyParams)
    {
        // no changes since the last render - keep the cached cubemap
        if (memcmp(mappedProcSkyParams, &params, sizeof(ProceduralSkyParams)) == 0)
        {
            return;
        }

        memcpy(mappedProcSkyParams, &params, sizeof(ProceduralSkyParams));
    }

    DispatchClouds(cmd, params);

    for (VkImage image : { cubemap.image, envCubemap.image })
    {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 6;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, procSkyPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, procSkyPipelineLayout,
                            0, 1, &procSkyDescSet, 0, nullptr);

    const uint32_t wgX = Utils::GetWorkGroupCount(cubemapSize, 16);
    const uint32_t wgY = Utils::GetWorkGroupCount(cubemapSize, 16);
    vkCmdDispatch(cmd, wgX, wgY, 6);

    GenerateMipmaps(cmd, cubemap.image, VK_IMAGE_LAYOUT_GENERAL);
    GenerateMipmaps(cmd, envCubemap.image, VK_IMAGE_LAYOUT_GENERAL);
}

