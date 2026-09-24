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

#include "RhiSkyPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"
#include "RhiTextureTable.h"

#include <tuple>
#include <utility>

#include "../Framebuffers.h"
#include "../Matrix.h"
#include "../Utils.h"

using namespace vkpt;

namespace
{

// The engine's blobs by the names ShaderManager knows them: "VertDefault" is RsRasterizer.vert.spv
// (ShaderManager.cpp:62), "FragSky" is RsSky.frag.spv (ShaderManager.cpp:60). RasterPass passes
// exactly that pair, plus "FragWorld", to its RasterizerPipelines objects (RasterPass.cpp:64-65,
// :74-75).
const char *const VERTEX_SHADER_FILE_NAME = "RsRasterizer.vert.spv";
const char *const PIXEL_SHADER_FILE_NAME  = "RsSky.frag.spv";

// The world sub-pass's fragment half: "FragWorld" is RsWorld.frag.spv (ShaderManager.cpp:59), the
// pair RasterPass hands to its world RasterizerPipelines (RasterPass.cpp:74-75). The vertex half is
// the same RsRasterizer.vert.spv the sky uses, so the shared vertexShader is bound to both.
const char *const WORLD_PIXEL_SHADER_FILE_NAME = "RsWorld.frag.spv";

// The engine images of the world sub-pass, by the indices RasterPass's world pass uses: colour 1 is
// FB_IMAGE_INDEX_SCREEN_EMISSION (62), the shader's binding-25 image is
// FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR (25) - an Rgba32ui storage image
// (RasterPass.cpp:53, :118). Prepare reads them through Framebuffers::GetScreenEmissionHandles /
// GetPrimaryToReflRefrHandles.
//
// The binding-25 image as an NVRHI slot: with the layout's unorderedAccess offset 0 the Vulkan
// binding is 0 + slot, so the slot is the shader's raw binding 25 (which for the engine's storage
// images equals the framebuffer image index).
constexpr uint32_t WORLD_STORAGE_SLOT = 25;

// The world shader's set 1 binding 0 is the engine's global uniform block at raw binding 0
// (BINDING_GLOBAL_UNIFORM, ShaderCommonC.h), so the layout's constant-buffer offset must be 0
// instead of the NVRHI default 256 (recon 4.2).
constexpr uint32_t WORLD_UNIFORM_OFFSET = 0;

// The world shader's set 4 binding 25. The legacy layout declares the whole engine triple -
// shaderResource 124, unorderedAccess 0, sampler 248 - and set 4's storage images sit at
// unorderedAccess + index; a one-item layout only consults the UAV offset, the other two are
// repeated to document which set this is (recon 5).
constexpr uint32_t WORLD_FRAMEBUFFERS_SRV_OFFSET = 124;
constexpr uint32_t WORLD_FRAMEBUFFERS_UAV_OFFSET = 0;
constexpr uint32_t WORLD_FRAMEBUFFERS_SAMPLER_OFFSET = 248;

// FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR is VK_FORMAT_R32G32B32A32_UINT (ShaderCommonCFramebuf.cpp:34),
// which the pinned NVRHI maps to this format (vulkan-constants.cpp:87). Both the real wrap and the
// dummy use it.
constexpr nvrhi::Format WORLD_STORAGE_FORMAT = nvrhi::Format::RGBA32_UINT;

// The legacy push-constant range is 88 bytes (Rasterizer.cpp:66, :485-489) while RsSky.frag and
// RsSwapchain.frag declare a 92-byte block: their last member, emissionMultiplier at offset 88, is
// declared but never read, and the legacy host never writes those four bytes (HLSL/RsSky.frag.hlsl:35,
// recon 2.3). The pass mirrors the legacy value, so both renderers push the same bytes and the
// shader-visible prefix stays what the legacy path produces.
constexpr uint32_t RASTERIZED_PUSH_CONSTANT_SIZE = 88;

// Both stages use SpecId 0 for their single constant: RsRasterizer.vert declares
// applyVertexColorGamma, RsSky.frag declares alphaTest, each 4 bytes and written as a uint32
// (RasterizerPipelines.cpp:315-343).
constexpr uint32_t SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA = 0;
constexpr uint32_t SPEC_CONSTANT_ALPHA_TEST = 0;

// The legacy sky depth buffer's format (RasterPass.cpp:29-30) and the NVRHI name of the same
// VkFormat.
constexpr nvrhi::Format DEPTH_FORMAT = nvrhi::Format::D32;

// The state key of RasterizerPipelines::ConvertToStateFlags, bit for bit (RasterizerPipelines.cpp:
// 29-55). The numbers are repeated instead of shared because that function is private to the legacy
// translation unit; keeping them identical is what makes a DrawInfo land in the same state in both
// renderers.
constexpr uint32_t PIPELINE_STATE_MASK_IS_ALPHA_TEST          = 1 << 0;
constexpr uint32_t PIPELINE_STATE_MASK_BLEND_ENABLE           = 1 << 1;
constexpr uint32_t PIPELINE_STATE_MASK_DEPTH_TEST_ENABLE      = 1 << 2;
constexpr uint32_t PIPELINE_STATE_MASK_DEPTH_WRITE_ENABLE     = 1 << 3;
constexpr uint32_t PIPELINE_STATE_MASK_IS_LINES               = 1 << 4;
constexpr uint32_t PS_SRC_OFFSET                              = 5;

constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_ONE                 = 1 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_ZERO                = 2 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_SRC_COLOR           = 3 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_SRC_COLOR = 4 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_DST_COLOR           = 5 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_DST_COLOR = 6 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_SRC_ALPHA           = 7 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_SRC_ALPHA = 8 << PS_SRC_OFFSET;
constexpr uint32_t PIPELINE_STATE_MASK_BLEND_SRC                      = 15 << PS_SRC_OFFSET;
constexpr uint32_t PS_DST_OFFSET                                      = 4 + PS_SRC_OFFSET;

constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_ONE                 = 1 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_ZERO                = 2 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_SRC_COLOR           = 3 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_SRC_COLOR = 4 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_DST_COLOR           = 5 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_DST_COLOR = 6 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_SRC_ALPHA           = 7 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_SRC_ALPHA = 8 << PS_DST_OFFSET;
constexpr uint32_t PIPELINE_STATE_MASK_BLEND_DST                      = 15 << PS_DST_OFFSET;

// The vertex spec constant carries no state of a DrawInfo: RasterizerPipelines gets one value per
// object and bakes it into every pipeline it creates (RasterizerPipelines.cpp:96, :230, :323),
// while this pass is told the value per Render call. The legacy key uses bits 0..12, so bit 13 is
// free and the flag joins the key instead of forcing a cache flush when it changes.
constexpr uint32_t PIPELINE_STATE_VALUE_VERTEX_COLOR_GAMMA = 1 << 13;

// The mirror of RasterizerPipelines::ConvertToStateFlags (RasterizerPipelines.cpp:57-113), byte for
// byte, including the quirk that an unknown blend factor zeroes the whole key.
uint32_t ConvertToStateFlags(RgRasterizedGeometryStateFlags pipelineState, RgBlendFactor blendFuncSrc, RgBlendFactor blendFuncDst)
{
    uint32_t r = 0;

    if (pipelineState & RG_RASTERIZED_GEOMETRY_STATE_BLEND_ENABLE)
    {
        r |= PIPELINE_STATE_MASK_BLEND_ENABLE;

        switch (blendFuncSrc)
        {
            case RG_BLEND_FACTOR_ONE:                   r |= PIPELINE_STATE_VALUE_BLEND_SRC_ONE;            break;
            case RG_BLEND_FACTOR_ZERO:                  r |= PIPELINE_STATE_VALUE_BLEND_SRC_ZERO;           break;
            case RG_BLEND_FACTOR_SRC_COLOR:             r |= PIPELINE_STATE_VALUE_BLEND_SRC_SRC_COLOR;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:   r |= PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_SRC_COLOR;  break;
            case RG_BLEND_FACTOR_DST_COLOR:             r |= PIPELINE_STATE_VALUE_BLEND_SRC_DST_COLOR;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_DST_COLOR:   r |= PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_DST_COLOR;  break;
            case RG_BLEND_FACTOR_SRC_ALPHA:             r |= PIPELINE_STATE_VALUE_BLEND_SRC_SRC_ALPHA;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:   r |= PIPELINE_STATE_VALUE_BLEND_SRC_ONE_MINUS_SRC_ALPHA;  break;
            default: assert(0); r = 0;
        }

        switch (blendFuncDst)
        {
            case RG_BLEND_FACTOR_ONE:                   r |= PIPELINE_STATE_VALUE_BLEND_DST_ONE;            break;
            case RG_BLEND_FACTOR_ZERO:                  r |= PIPELINE_STATE_VALUE_BLEND_DST_ZERO;           break;
            case RG_BLEND_FACTOR_SRC_COLOR:             r |= PIPELINE_STATE_VALUE_BLEND_DST_SRC_COLOR;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:   r |= PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_SRC_COLOR;  break;
            case RG_BLEND_FACTOR_DST_COLOR:             r |= PIPELINE_STATE_VALUE_BLEND_DST_DST_COLOR;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_DST_COLOR:   r |= PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_DST_COLOR;  break;
            case RG_BLEND_FACTOR_SRC_ALPHA:             r |= PIPELINE_STATE_VALUE_BLEND_DST_SRC_ALPHA;      break;
            case RG_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:   r |= PIPELINE_STATE_VALUE_BLEND_DST_ONE_MINUS_SRC_ALPHA;  break;
            default: assert(0); r = 0;
        }
    }

    if (pipelineState & RG_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST)
    {
        r |= PIPELINE_STATE_MASK_DEPTH_TEST_ENABLE;
    }

    if (pipelineState & RG_RASTERIZED_GEOMETRY_STATE_DEPTH_WRITE)
    {
        r |= PIPELINE_STATE_MASK_DEPTH_WRITE_ENABLE;
    }

    if (pipelineState & RG_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST)
    {
        r |= PIPELINE_STATE_MASK_IS_LINES;
    }

    if (pipelineState & RG_RASTERIZED_GEOMETRY_STATE_ALPHA_TEST)
    {
        r |= PIPELINE_STATE_MASK_IS_ALPHA_TEST;
    }

    return r;
}

// The blend factor the state key encodes in its four-bit field, as the NVRHI name of the same
// factor (RasterizerPipelines::ConvertBlendFactorToVk, RasterizerPipelines.cpp:198-212).
nvrhi::BlendFactor DecodeBlendFactor(uint32_t stateFlags, uint32_t offset)
{
    switch ((stateFlags >> offset) & 15)
    {
        case 1:  return nvrhi::BlendFactor::One;
        case 2:  return nvrhi::BlendFactor::Zero;
        case 3:  return nvrhi::BlendFactor::SrcColor;
        case 4:  return nvrhi::BlendFactor::InvSrcColor; // OneMinusSrcColor
        case 5:  return nvrhi::BlendFactor::DstColor;
        case 6:  return nvrhi::BlendFactor::InvDstColor; // OneMinusDstColor
        case 7:  return nvrhi::BlendFactor::SrcAlpha;
        case 8:  return nvrhi::BlendFactor::InvSrcAlpha; // OneMinusSrcAlpha
        // Code 0 is what the legacy key holds when blending is off: ConvertToStateFlags only fills
        // these two fields inside its RG_RASTERIZED_GEOMETRY_STATE_BLEND_ENABLE branch
        // (RasterizerPipelines.cpp:57-113), so an opaque draw - the sky is one - reaches this decoder
        // with zeros. The factor is ignored by setBlendEnable(false) either way.
        case 0:  return nvrhi::BlendFactor::One;
        default: assert(0); return nvrhi::BlendFactor::One;
    }
}

// The per-draw block, byte for byte the legacy RasterizedPushConst (Rasterizer.cpp:33-65): the
// model-view-projection, the color, and the two texture indices. The type is repeated because
// RasterizedPushConst is private to the Rasterizer translation unit; the offsets below are the ones
// Rasterizer.cpp asserts, and the fragment half matches the members the sky shader reads at 64/80/84.
struct SkyPushConstants
{
    float    vp[16];
    float    c[4];
    uint32_t t;
    uint32_t e;

    explicit SkyPushConstants(const RasterizedDataCollector::DrawInfo &info, const float *defaultViewProj)
    {
        float model[16];
        Matrix::ToMat4Transposed(model, info.transform);

        if (info.viewProj)
        {
            Matrix::Multiply(vp, model, info.viewProj->Get());
        }
        else
        {
            Matrix::Multiply(vp, model, defaultViewProj);
        }

        memcpy(c, info.color.Get(), 4 * sizeof(float));
        t = info.textureIndex;
        e = info.emissionTextureIndex;
    }
};

static_assert(offsetof(SkyPushConstants, vp) == 0);
static_assert(offsetof(SkyPushConstants, c) == 64);
static_assert(offsetof(SkyPushConstants, t) == 80);
static_assert(offsetof(SkyPushConstants, e) == 84);
static_assert(sizeof(SkyPushConstants) == 88);

void LogMessage(const RhiSkyPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

// The numbers the input layout of Create is built from, taken from the collector's own struct and
// asserted so that a change of RgVertex cannot silently break the RHI pipeline. They are the
// offsets the legacy VkVertexInputAttributeDescriptions use (RasterizedDataCollector.cpp:31-54);
// recon 3.3 lists different ones (60 and 40), and the collector is the authority.
static_assert(offsetof(RgVertex, position) == 0);
static_assert(offsetof(RgVertex, texCoord) == 32);
static_assert(offsetof(RgVertex, packedColor) == 56);
static_assert(sizeof(RgVertex) == 80);

}

RhiSkyPass::RhiSkyPass() = default;

RhiSkyPass::~RhiSkyPass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images and the depth images and framebuffers reference the
        // device; the host destroys the pass while it can still idle the device (VulkanDevice does
        // that before the skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    // Pipelines reference their (specialized) shaders, so they go first.
    pipelines.clear();
    worldPipelines.clear();

    for (Target &target : targets)
    {
        target.worldFramebuffer = nullptr;
        target.worldScreenEmissionTexture = nullptr;
        target.worldValid = false;
        target.framebuffer = nullptr;
        target.depthTexture = nullptr;
        target.albedoTexture = nullptr;
        target.albedoImage = VK_NULL_HANDLE;
        target.valid = false;
    }

    worldFramebuffersSet = nullptr;
    worldHoleSet = nullptr;
    worldUniformSet = nullptr;
    for (nvrhi::BindingSetHandle &set : worldTonemappingSets)
    {
        set = nullptr;
    }
    worldStorageTexture = nullptr;
    worldScreenEmissionTexture = nullptr;

    worldFramebuffersLayout = nullptr;
    worldPushConstantLayout = nullptr;
    worldTonemappingLayout = nullptr;
    worldUniformLayout = nullptr;
    worldPixelShader = nullptr;

    pushConstantLayout = nullptr;
    inputLayout = nullptr;
    vertexShader = nullptr;
    pixelShader = nullptr;
}

bool RhiSkyPass::Create(nvrhi::IDevice *pDevice,
                        rhi::RhiTextureTable *pTextureTable,
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
    textureTable = pTextureTable;
    frameContext = pFrameContext;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the sky pass needs an RHI device");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the sky pass needs the shared texture table of the RHI layer");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the sky pass needs the frame context of the RHI layer");
        return false;
    }

    if (!LoadShader(VERTEX_SHADER_FILE_NAME, nvrhi::ShaderType::Vertex, vertexShader) ||
        !LoadShader(PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, pixelShader))
    {
        return false;
    }

    // The RgVertex input layout the collector feeds: one binding at slot 0 with the collector's
    // stride and three attributes, whose order is their location (the Vulkan backend numbers the
    // attributes by their position in the array, vulkan-shader.cpp:176-197). The offsets are taken
    // from the same struct RasterizedDataCollector::GetVertexLayout uses, so the RHI pipeline and
    // the legacy one cannot drift: position at 0, the packed color at 56 and the first texture
    // coordinate at 32 (RgVertex, vkpt.h:349-361).
    const uint32_t vertexStride = RasterizedDataCollector::GetVertexStride();

    const nvrhi::VertexAttributeDesc vertexAttributes[] =
    {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setBufferIndex(0)
            .setOffset(offsetof(RgVertex, position))
            .setElementStride(vertexStride),
        nvrhi::VertexAttributeDesc()
            .setName("COLOR")
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setBufferIndex(0)
            .setOffset(offsetof(RgVertex, packedColor))
            .setElementStride(vertexStride),
        nvrhi::VertexAttributeDesc()
            .setName("TEXCOORD")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setBufferIndex(0)
            .setOffset(offsetof(RgVertex, texCoord))
            .setElementStride(vertexStride),
    };

    // The vertex shader argument is ignored by the Vulkan backend (vulkan-shader.cpp:136-138) and
    // passed for the D3D backends NVRHI supports.
    inputLayout = device->createInputLayout(vertexAttributes, uint32_t(std::size(vertexAttributes)), vertexShader);
    if (inputLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the sky pass input layout");
        return false;
    }

    // The pipeline's second layout, and its only job is the push-constant block: no descriptors, so
    // no binding set has to be created or bound for it. The texture table stays the first layout and
    // therefore descriptor set 0, which is the shader's DESC_SET_TEXTURES (recon 5).
    const nvrhi::BindingLayoutItem layoutItems[] =
    {
        nvrhi::BindingLayoutItem::PushConstants(0, RASTERIZED_PUSH_CONSTANT_SIZE),
    };

    pushConstantLayout = rhi::createBindingLayout(device, layoutItems, "RhiSky push constants");
    if (pushConstantLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the sky pass push-constant layout");
        return false;
    }

    created = true;
    return true;
}

bool RhiSkyPass::CreateWorld(nvrhi::IBuffer *pUniformBuffer,
                             nvrhi::IBuffer *const pTonemappingBuffers[MAX_FRAMES_IN_FLIGHT],
                             const std::tuple<VkImage, VkImageView, VkFormat> &screenEmission,
                             const std::tuple<VkImage, VkImageView, VkFormat> &storageImage)
{
    if (!created)
    {
        LogMessage(print, "Warning: RHI: the world sub-pass needs the raster pass created first");
        return false;
    }

    if (pUniformBuffer == nullptr)
    {
        LogMessage(print, "Warning: RHI: the world sub-pass needs the wrapped global uniform buffer (set 1)");
        return false;
    }

    if (pUniformBuffer->getDesc().isVolatile)
    {
        // BindingSetItem::ConstantBuffer turns a volatile buffer into a dynamic-offset binding
        // (nvrhi.h:2311-2318), which the static ConstantBuffer layout item would reject.
        LogMessage(print, "Warning: RHI: the world uniform buffer must be a static wrap, not a volatile buffer");
        return false;
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (pTonemappingBuffers == nullptr || pTonemappingBuffers[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: the world sub-pass needs the wrapped tonemapping buffer of every frame slot (set 2)");
            return false;
        }

        // The shader declares set 2 as StructuredBuffer<ShTonemapping>; the backend asserts a
        // non-zero structStride when the binding set is created
        // (vulkan-resource-bindings.cpp:535-536), so a wrap without
        // Tonemapping::GetElementSize() would trip a debug assert inside NVRHI.
        if (pTonemappingBuffers[i]->getDesc().structStride == 0)
        {
            LogMessage(print, "Warning: RHI: the world tonemapping buffer needs Tonemapping::GetElementSize() as structStride");
            return false;
        }
    }

    // The engine handles are kept as the fallback of the per-frame resolution UpdateWorldWraps
    // does: the Framebuffers accessors answer with the current images, and these handles are what
    // this pass wraps while the engine has none (or after ReleaseTargets cleared the wraps).
    worldScreenEmissionHandles = screenEmission;
    worldStorageHandles = storageImage;

    if (!worldCreated)
    {
        if (!LoadShader(WORLD_PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, worldPixelShader))
        {
            return false;
        }

        // The world pipeline layout in the shader's own set order: the shared texture table (0),
        // the engine's global uniform (1), the tonemapping buffer (2), the set-3 hole (3) and the
        // partial framebuffers layout (4). NVRHI's legacy binding mode keeps the order the layouts
        // are added in as the descriptor set numbers, so adding them in this order reproduces the
        // numbers RsWorld.frag spells (recon 4).
        {
            const nvrhi::BindingLayoutItem layoutItems[] =
            {
                nvrhi::BindingLayoutItem::ConstantBuffer(0),
            };
            const nvrhi::VulkanBindingOffsets offsets =
                nvrhi::VulkanBindingOffsets().setConstantBufferOffset(WORLD_UNIFORM_OFFSET);

            worldUniformLayout = rhi::createBindingLayout(device, layoutItems, "RhiSky world uniform", &offsets);
        }
        {
            // The engine's tonemapping set is a raw binding 0 storage buffer; the default NVRHI
            // offsets already put a StructuredBuffer_SRV there (shaderResource 0).
            const nvrhi::BindingLayoutItem layoutItems[] =
            {
                nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
            };

            worldTonemappingLayout = rhi::createBindingLayout(device, layoutItems, "RhiSky world tonemapping");
        }

        // Set 3 is the hole between the tonemapping set and the framebuffers set: RsWorld.frag
        // declares no set 3, but the framebuffers set has to land at 4, so a layout must occupy the
        // position. It is a real, zero-item layout that carries the pipeline's only push-constant
        // item: the backend skips that item when it builds the Vulkan bindings
        // (vulkan-resource-bindings.cpp:90-94) and takes the push range from it
        // (vulkan-resource-bindings.cpp:1110-1121), so the layout still creates an empty descriptor
        // set layout - the shape the sky's pushConstantLayout already has.
        {
            const nvrhi::BindingLayoutItem layoutItems[] =
            {
                nvrhi::BindingLayoutItem::PushConstants(0, RASTERIZED_PUSH_CONSTANT_SIZE),
            };

            worldPushConstantLayout = rhi::createBindingLayout(device, layoutItems, "RhiSky world set 3 (push constants)");
        }

        // The partial framebuffers layout: the engine's set declares 124 UAVs, 124 SRVs and 124
        // samplers, but RsWorld.frag statically uses exactly one binding - 25, the Rgba32ui storage
        // image it reads and writes. A one-item layout produces the Vulkan binding 0 + 25, and the
        // unwritten descriptors of the other engine bindings are irrelevant (recon 5).
        {
            const nvrhi::BindingLayoutItem layoutItems[] =
            {
                nvrhi::BindingLayoutItem::Texture_UAV(WORLD_STORAGE_SLOT),
            };
            const nvrhi::VulkanBindingOffsets offsets = nvrhi::VulkanBindingOffsets()
                .setShaderResourceOffset(WORLD_FRAMEBUFFERS_SRV_OFFSET)
                .setUnorderedAccessViewOffset(WORLD_FRAMEBUFFERS_UAV_OFFSET)
                .setSamplerOffset(WORLD_FRAMEBUFFERS_SAMPLER_OFFSET);

            worldFramebuffersLayout = rhi::createBindingLayout(device, layoutItems, "RhiSky world framebuffers (binding 25)", &offsets);
        }

        if (worldUniformLayout == nullptr || worldTonemappingLayout == nullptr ||
            worldPushConstantLayout == nullptr || worldFramebuffersLayout == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create a world pipeline layout");
            return false;
        }

        // The empty set that fills the set-3 hole. The recon's other route - a null entry in
        // GraphicsState::bindings - is unsafe in the pinned NVRHI: the automatic-barrier pass
        // dereferences every entry before the bind pass gets to treat a null as a hole
        // (vulkan-state-tracking.cpp:105), while an empty BindingSet is a normal set that happens
        // to bind no descriptors. Allocating one from a zero-item layout is legal Vulkan (a
        // descriptor pool with no pool sizes serves set layouts with no bindings).
        worldHoleSet = device->createBindingSet(nvrhi::BindingSetDesc(), worldPushConstantLayout);
        if (worldHoleSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the empty set of the world set-3 hole");
            return false;
        }
    }

    bool buffersChanged = worldUniformBuffer != pUniformBuffer;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        buffersChanged = buffersChanged || worldTonemappingBuffers[i] != pTonemappingBuffers[i];
    }

    if (buffersChanged)
    {
        ReleaseWorldBufferSets();

        worldUniformBuffer = pUniformBuffer;
        memcpy(worldTonemappingBuffers, pTonemappingBuffers, sizeof(worldTonemappingBuffers));

        nvrhi::BindingSetDesc uniformDesc;
        uniformDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, worldUniformBuffer));
        worldUniformSet = device->createBindingSet(uniformDesc, worldUniformLayout);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            nvrhi::BindingSetDesc tonemappingDesc;
            tonemappingDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, worldTonemappingBuffers[i]));
            worldTonemappingSets[i] = device->createBindingSet(tonemappingDesc, worldTonemappingLayout);
        }

        if (worldUniformSet == nullptr || worldTonemappingSets[0] == nullptr || worldTonemappingSets[1] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the world buffer binding sets");
            return false;
        }
    }

    // The binding-25 set needs the storage wrap, which Prepare (re)creates when it knows the size
    // and the current engine image; until then the world target stays invalid and RenderWorld
    // skips the frame.
    worldCreated = true;
    return true;
}

void RhiSkyPass::SetGeometryBuffers(nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer)
{
    vertexBuffer = pVertexBuffer;
    indexBuffer = pIndexBuffer;
}

void RhiSkyPass::SetSkyCamera(const float *pView, const float *pProj, const float jitter[2],
                              const float skyViewerPos[3])
{
    if (pView == nullptr || pProj == nullptr || jitter == nullptr || skyViewerPos == nullptr)
    {
        assert(0);
        return;
    }

    memcpy(view, pView, sizeof(view));
    memcpy(proj, pProj, sizeof(proj));
    memcpy(this->jitter, jitter, sizeof(this->jitter));
    memcpy(this->skyViewerPos, skyViewerPos, sizeof(this->skyViewerPos));

    hasSkyCamera = true;
}

bool RhiSkyPass::Prepare(nvrhi::ICommandList *pCommandList,
                         uint32_t frameIndex,
                         const Framebuffers &framebuffers,
                         uint32_t width,
                         uint32_t height)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return false;
    }

    // Render reads the target of the last Prepare, so the index is recorded before any early return.
    activeTargetIndex = frameIndex;
    Target &target = targets[frameIndex];

    // The frame's depth clear happens in the sky's Render; this flag lets RenderWorld tell whether
    // that happened when the sky had no draws.
    target.depthCleared = false;

    if (width == 0 || height == 0)
    {
        target.valid = false;
        return false;
    }

    // The engine's ALBEDO image of that frame slot: Framebuffers resolves the swap permutation that
    // makes the slot's image the one the legacy passes would render into (Framebuffers.cpp:33-53).
    const auto [albedoImage, albedoView, albedoFormat] =
        framebuffers.GetImageHandles(FB_IMAGE_INDEX_ALBEDO, frameIndex);

    if (albedoImage == VK_NULL_HANDLE || albedoView == VK_NULL_HANDLE || albedoFormat == VK_FORMAT_UNDEFINED)
    {
        LogMessage(print, "Warning: RHI: the sky pass got no ALBEDO image for frame " + std::to_string(frameIndex));
        target.valid = false;
        return false;
    }

    // Nothing to do while the slot still wraps the same image at the same size: replacing the wrap
    // every frame would create a texture and a view per frame for nothing. The state announcement
    // still has to happen, because a render-target wrap keeps no state between command lists
    // (RhiTextureSource.h:56-70).
    if (target.valid && target.albedoImage == albedoImage && target.width == width && target.height == height)
    {
        // The image rests in GENERAL - NVRHI's UnorderedAccess - between frames: that is the layout
        // the engine's own framebuffer descriptors declare for it (Framebuffers.cpp:786, :794), and
        // the frame skeleton's last state of the previous list puts it back there. Announcing
        // anything else would leave the image in a layout the engine's descriptors do not expect
        // (VUID-vkCmdDraw-None-09600).
        pCommandList->beginTrackingTextureState(target.albedoTexture, nvrhi::AllSubresources,
                                                nvrhi::ResourceStates::UnorderedAccess);

        PrepareWorldTarget(pCommandList, target, framebuffers, frameIndex, width, height);
        return true;
    }

    ReleaseTarget(target);

    const uint64_t albedoImageHandle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(albedoImage));
    const uint64_t albedoViewHandle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(albedoView));

    target.albedoTexture = rhi::wrapEngineRenderTarget(
        device, albedoImageHandle, albedoViewHandle, albedoFormat, width, height,
        "RhiSky ALBEDO frame " + std::to_string(frameIndex));

    if (target.albedoTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to wrap the ALBEDO image for the sky pass");
        return false;
    }

    // The announcement of the newly wrapped image, in the first list that uses it. An engine
    // framebuffer image rests in VK_IMAGE_LAYOUT_GENERAL - Framebuffers creates every image and
    // immediately barriers it there (Framebuffers.cpp:754-758) and never leaves that layout - which
    // NVRHI names ResourceStates::UnorderedAccess (vulkan-constants.cpp:242-245). Announcing that
    // real state lets the first framebuffer use emit the GENERAL -> COLOR_ATTACHMENT_OPTIMAL
    // transition; announcing RenderTarget here would claim a layout the image is not in and the
    // transition would be missing, which is the trap RhiTextureSource.h:80-86 warns about. Every list
    // announces the same GENERAL state, because the frame skeleton's last state of a list puts the
    // image back there after the present sampled it.
    pCommandList->beginTrackingTextureState(target.albedoTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);

    // The pass's own depth image, per frame slot like RasterPass's depthImages (RasterPass.h:96-98)
    // and in the legacy format (RasterPass.cpp:29). initialState/keepInitialState make NVRHI return
    // it to DepthWrite when a list closes; the first use is the explicit clear, which the backend
    // does through CopyDest.
    nvrhi::TextureDesc depthDesc;
    depthDesc.width = width;
    depthDesc.height = height;
    depthDesc.format = DEPTH_FORMAT;
    depthDesc.dimension = nvrhi::TextureDimension::Texture2D;
    depthDesc.mipLevels = 1;
    depthDesc.arraySize = 1;
    depthDesc.sampleCount = 1;
    depthDesc.isShaderResource = false;
    depthDesc.isRenderTarget = true;
    depthDesc.initialState = nvrhi::ResourceStates::DepthWrite;
    depthDesc.keepInitialState = true;

    target.depthTexture = rhi::createTexture(device, depthDesc, "RhiSky depth frame " + std::to_string(frameIndex));
    if (target.depthTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the sky depth image");
        return false;
    }

    // The framebuffer over the pair, exactly the attachments of the legacy sky framebuffer
    // (RasterPass.cpp:142-145): ALBEDO as the only color attachment, the depth image as the depth
    // attachment.
    nvrhi::FramebufferDesc framebufferDesc;
    framebufferDesc.addColorAttachment(target.albedoTexture);
    framebufferDesc.setDepthAttachment(target.depthTexture);

    target.framebuffer = device->createFramebuffer(framebufferDesc);
    if (target.framebuffer == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the sky framebuffer");
        return false;
    }

    // The pipeline is built against the framebuffer's color format, so a change means every cached
    // pipeline belongs to the wrong framebuffer and has to be replaced. ALBEDO is
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32 (ShaderCommonCFramebuf.cpp:9), so this only happens if the
    // engine ever changes that; the wrapped texture's own format is authoritative because it is the
    // one NVRHI mapped the engine format to.
    const nvrhi::Format colorFormat = target.albedoTexture->getDesc().format;
    if (pipelineColorFormat != colorFormat)
    {
        ReleasePipelineCache();
        pipelineColorFormat = colorFormat;
    }

    target.albedoImage = albedoImage;
    target.width = width;
    target.height = height;
    target.valid = true;

    PrepareWorldTarget(pCommandList, target, framebuffers, frameIndex, width, height);
    return true;
}

void RhiSkyPass::Render(nvrhi::ICommandList *pCommandList,
                        const RasterizedDataCollector::DrawInfo *pDraws,
                        uint32_t drawCount,
                        bool applyVertexColorGamma)
{
    if (!created || pCommandList == nullptr || pDraws == nullptr || drawCount == 0)
    {
        return;
    }

    Target &target = targets[activeTargetIndex];
    if (!target.valid)
    {
        return;
    }

    if (vertexBuffer == nullptr || indexBuffer == nullptr)
    {
        if (!warnedMissingGeometry)
        {
            warnedMissingGeometry = true;
            LogMessage(print, "Warning: RHI: the sky pass has no geometry buffers, the sky is skipped");
        }
        return;
    }

    if (!hasSkyCamera)
    {
        if (!warnedMissingCamera)
        {
            warnedMissingCamera = true;
            LogMessage(print, "Warning: RHI: the sky pass got no camera, the sky is skipped");
        }
        return;
    }

    // The default sky view-projection, built the way Rasterizer::DrawSkyToAlbedo builds it
    // (Rasterizer.cpp:176-210): the jitter is added to the third row of the projection in pixels of
    // the render resolution, then the sky view (the frame's view with the viewer moved to the sky
    // viewer position) multiplies it. Every sky upload passes a null viewProjection today
    // (Quake/gl_sky.c:1085), so this default is the transform of every draw.
    float jitteredProj[16];
    memcpy(jitteredProj, proj, sizeof(jitteredProj));
    jitteredProj[2 * 4 + 0] += jitter[0] / float(target.width);
    jitteredProj[2 * 4 + 1] += jitter[1] / float(target.height);

    float skyView[16];
    Matrix::SetNewViewerPosition(skyView, view, skyViewerPos);
    Matrix::Multiply(defaultViewProj, skyView, jitteredProj);

    // The engine textures the table wrapped since the last frame need their first-use state
    // declared in the first list that binds the table (RhiTextureSource.h); this is that list.
    textureTable->TrackPendingTextures(pCommandList);

    // NVRHI's attachment loadOp is always LOAD (vulkan-graphics.cpp:80, :111), so the clear the
    // legacy sky render pass does on its depth attachment (RasterPass.cpp:355) is an explicit
    // command. It runs once, before the first draw, and outside any render pass.
    pCommandList->clearDepthStencilTexture(target.depthTexture, nvrhi::AllSubresources, true, 1.0f, false, 0);
    target.depthCleared = true;

    const float targetWidth = float(target.width);
    const float targetHeight = float(target.height);

    for (uint32_t i = 0; i < drawCount; i++)
    {
        const RasterizedDataCollector::DrawInfo &info = pDraws[i];

        const uint32_t stateFlags =
            ConvertToStateFlags(info.pipelineState, info.blendFuncSrc, info.blendFuncDst);

        nvrhi::IGraphicsPipeline *pipeline = GetPipeline(stateFlags, applyVertexColorGamma, false);
        if (pipeline == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the sky pipeline, the sky is incomplete");
            return;
        }

        // The viewport of the draw, or the whole target: the legacy loop keeps the scissor at the
        // whole render area and switches only the viewport (Rasterizer.cpp:356-357, :389-396).
        nvrhi::Viewport viewport(targetWidth, targetHeight);
        if (info.viewport)
        {
            const VkViewport &v = *info.viewport;
            viewport = nvrhi::Viewport(v.x, v.x + v.width, v.y, v.y + v.height, v.minDepth, v.maxDepth);
        }

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = target.framebuffer;
        state.viewport.addViewport(viewport);
        state.viewport.addScissorRect(nvrhi::Rect(0, int(target.width), 0, int(target.height)));
        // Set 0 is the bindless texture table, the pipeline's first layout; the second layout only
        // carries the push constants and has no descriptors to bind.
        state.addBindingSet(textureTable->GetTable());
        state.addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vertexBuffer).setSlot(0).setOffset(0));
        state.setIndexBuffer(nvrhi::IndexBufferBinding()
                                 .setBuffer(indexBuffer)
                                 .setFormat(nvrhi::Format::R32_UINT)
                                 .setOffset(0));

        pCommandList->setGraphicsState(state);

        // The framebuffer use above emits the GENERAL -> COLOR_ATTACHMENT_OPTIMAL barrier. The image
        // does not rest there: the present samples it, and the frame skeleton's last state of the
        // list moves it back to GENERAL, which is what the engine's framebuffer descriptors declare.

        // After the state: changing the state invalidates push constants (nvrhi.h:3430-3432), and
        // the block is rebuilt per draw exactly as Rasterizer::Draw does (Rasterizer.cpp:399-409).
        const SkyPushConstants push(info, defaultViewProj);
        pCommandList->setPushConstants(&push, sizeof(push));

        nvrhi::DrawArguments args;
        if (info.indexCount > 0)
        {
            // NVRHI carries the index count in 'vertexCount' for an indexed draw: drawIndexed maps
            // it to vkCmdDrawIndexed's indexCount (vulkan-graphics.cpp:689-700).
            args.vertexCount = info.indexCount;
            args.startIndexLocation = info.firstIndex;
            args.startVertexLocation = info.firstVertex;
            pCommandList->drawIndexed(args);
        }
        else
        {
            args.vertexCount = info.vertexCount;
            args.startVertexLocation = info.firstVertex;
            pCommandList->draw(args);
        }
    }
}

void RhiSkyPass::RenderWorld(nvrhi::ICommandList *pCommandList,
                             const RasterizedDataCollector::DrawInfo *pDraws,
                             uint32_t drawCount,
                             bool applyVertexColorGamma)
{
    if (!created || !worldCreated || pCommandList == nullptr || pDraws == nullptr || drawCount == 0)
    {
        return;
    }

    Target &target = targets[activeTargetIndex];
    if (!target.valid || !target.worldValid ||
        worldUniformSet == nullptr || worldTonemappingSets[activeTargetIndex] == nullptr ||
        worldFramebuffersSet == nullptr)
    {
        if (!warnedMissingWorldTarget)
        {
            warnedMissingWorldTarget = true;
            LogMessage(print, "Warning: RHI: the world sub-pass has no target (no CreateWorld handles or no engine framebuffers), the world is skipped");
        }
        return;
    }

    if (vertexBuffer == nullptr || indexBuffer == nullptr)
    {
        if (!warnedMissingGeometry)
        {
            warnedMissingGeometry = true;
            LogMessage(print, "Warning: RHI: the raster pass has no geometry buffers, the sky and the world are skipped");
        }
        return;
    }

    if (!hasSkyCamera)
    {
        if (!warnedMissingCamera)
        {
            warnedMissingCamera = true;
            LogMessage(print, "Warning: RHI: the raster pass got no camera, the sky and the world are skipped");
        }
        return;
    }

    // The world's default view-projection, built the way Rasterizer::DrawToFinalImage builds it
    // (Rasterizer.cpp:266-271): the jitter is added to the third row of the projection in pixels of
    // the render resolution and the frame's plain view multiplies it. The sky moves the viewer to
    // the sky position first (SetNewViewerPosition, Rasterizer.cpp:203-210); the world deliberately
    // does not, so the shared camera stored by SetSkyCamera is used as it is.
    float jitteredProj[16];
    memcpy(jitteredProj, proj, sizeof(jitteredProj));
    jitteredProj[2 * 4 + 0] += jitter[0] / float(target.width);
    jitteredProj[2 * 4 + 1] += jitter[1] / float(target.height);

    float defaultViewProj[16];
    Matrix::Multiply(defaultViewProj, view, jitteredProj);

    // A no-op when the sky's Render already drained the table's pending wraps in this list; kept
    // so that a list that records the world without the sky still declares them.
    textureTable->TrackPendingTextures(pCommandList);

    // The sky's Render clears the depth before its draws; when the sky had no draws at all (or the
    // caller records the world alone), the clear is still owed to the world, so it happens here.
    if (!target.depthCleared)
    {
        pCommandList->clearDepthStencilTexture(target.depthTexture, nvrhi::AllSubresources, true, 1.0f, false, 0);
        target.depthCleared = true;
    }

    const float targetWidth = float(target.width);
    const float targetHeight = float(target.height);

    for (uint32_t i = 0; i < drawCount; i++)
    {
        const RasterizedDataCollector::DrawInfo &info = pDraws[i];

        const uint32_t stateFlags =
            ConvertToStateFlags(info.pipelineState, info.blendFuncSrc, info.blendFuncDst);

        nvrhi::IGraphicsPipeline *pipeline = GetPipeline(stateFlags, applyVertexColorGamma, true);
        if (pipeline == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the world pipeline, the world is incomplete");
            return;
        }

        // The same viewport/scissor rule as the sky's Render (Rasterizer.cpp:356-357, :389-396).
        nvrhi::Viewport viewport(targetWidth, targetHeight);
        if (info.viewport)
        {
            const VkViewport &v = *info.viewport;
            viewport = nvrhi::Viewport(v.x, v.x + v.width, v.y, v.y + v.height, v.minDepth, v.maxDepth);
        }

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = target.worldFramebuffer;
        state.viewport.addViewport(viewport);
        state.viewport.addScissorRect(nvrhi::Rect(0, int(target.width), 0, int(target.height)));
        // Sets 0..4 in the layout order of CreateWorld: the table, the uniform, the slot's
        // tonemapping buffer, the real empty set of the set-3 hole and the partial framebuffers
        // set with the binding-25 storage image.
        state.addBindingSet(textureTable->GetTable());
        state.addBindingSet(worldUniformSet);
        state.addBindingSet(worldTonemappingSets[activeTargetIndex]);
        state.addBindingSet(worldHoleSet);
        state.addBindingSet(worldFramebuffersSet);
        state.addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vertexBuffer).setSlot(0).setOffset(0));
        state.setIndexBuffer(nvrhi::IndexBufferBinding()
                                 .setBuffer(indexBuffer)
                                 .setFormat(nvrhi::Format::R32_UINT)
                                 .setOffset(0));

        pCommandList->setGraphicsState(state);

        const SkyPushConstants push(info, defaultViewProj);
        pCommandList->setPushConstants(&push, sizeof(push));

        nvrhi::DrawArguments args;
        if (info.indexCount > 0)
        {
            args.vertexCount = info.indexCount;
            args.startIndexLocation = info.firstIndex;
            args.startVertexLocation = info.firstVertex;
            pCommandList->drawIndexed(args);
        }
        else
        {
            args.vertexCount = info.vertexCount;
            args.startVertexLocation = info.firstVertex;
            pCommandList->draw(args);
        }
    }

    // The world's framebuffer leaves SCREEN_EMISSION in the render-target layout, while the engine's
    // own descriptors declare VK_IMAGE_LAYOUT_GENERAL for it (Framebuffers.cpp:786) - NVRHI's
    // UnorderedAccess - and the next Prepare announces exactly that. Moving the image back here, at
    // the end of the list that used it, is the same pattern the frame skeleton applies to ALBEDO
    // after the present sampled it; the barrier is placed now and committed when the list closes.
    pCommandList->setTextureState(worldScreenEmissionTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
}

void RhiSkyPass::PrepareWorldTarget(nvrhi::ICommandList *pCommandList, Target &target,
                                    const Framebuffers &framebuffers, uint32_t frameIndex,
                                    uint32_t width, uint32_t height)
{
    if (!worldCreated)
    {
        return;
    }

    UpdateWorldWraps(framebuffers, frameIndex, width, height);

    // The two engine images rest in GENERAL (Framebuffers.cpp:754-758), which NVRHI names
    // UnorderedAccess; announcing that real state every list is the rule of the two wraps
    // (RhiTextureSource.h). The dummy of a missing binding-25 image is an NVRHI-owned texture with
    // keepInitialState, so it needs no announcement.
    if (worldScreenEmissionTexture != nullptr)
    {
        pCommandList->beginTrackingTextureState(worldScreenEmissionTexture, nvrhi::AllSubresources,
                                                nvrhi::ResourceStates::UnorderedAccess);
    }
    if (worldStorageTexture != nullptr && !worldStorageIsDummy)
    {
        pCommandList->beginTrackingTextureState(worldStorageTexture, nvrhi::AllSubresources,
                                                nvrhi::ResourceStates::UnorderedAccess);
    }

    UpdateWorldFramebuffer(target);
}

nvrhi::ITexture *RhiSkyPass::GetAlbedoTexture(uint32_t frameIndex) const
{
    if (!created || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    // Only a target that Prepare finished is presentable: a slot that failed halfway has a wrap
    // without a framebuffer (or without a valid size), and Render does not draw into it either.
    const Target &target = targets[frameIndex];
    return target.valid ? target.albedoTexture.Get() : nullptr;
}

void RhiSkyPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }

    // The world's shared wraps and the set over them reference the same engine images, so they go
    // with the per-slot targets. The next Prepare re-reads the engine's accessors and re-wraps.
    ReleaseWorldWraps();
}

void RhiSkyPass::ReleaseTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the wrap would otherwise destroy a view of an engine image the GPU may still be
    // reading, and the RHI-owned resources have the same problem (RhiFrameContext.h). The queue
    // takes its reference now, so the handles below can be cleared immediately.
    if (frameContext != nullptr)
    {
        if (target.worldFramebuffer != nullptr)
        {
            frameContext->Retire(target.worldFramebuffer);
        }
        if (target.framebuffer != nullptr)
        {
            frameContext->Retire(target.framebuffer);
        }
        if (target.depthTexture != nullptr)
        {
            frameContext->Retire(target.depthTexture);
        }
        if (target.albedoTexture != nullptr)
        {
            frameContext->Retire(target.albedoTexture);
        }
    }

    target.worldFramebuffer = nullptr;
    target.worldScreenEmissionTexture = nullptr;
    target.worldValid = false;
    target.framebuffer = nullptr;
    target.depthTexture = nullptr;
    target.albedoTexture = nullptr;
    target.albedoImage = VK_NULL_HANDLE;
    target.width = 0;
    target.height = 0;
    target.valid = false;
}

void RhiSkyPass::ReleaseWorldWraps()
{
    for (Target &target : targets)
    {
        if (frameContext != nullptr && target.worldFramebuffer != nullptr)
        {
            frameContext->Retire(target.worldFramebuffer);
        }
        target.worldFramebuffer = nullptr;
        target.worldScreenEmissionTexture = nullptr;
        target.worldValid = false;
    }

    ReleaseWorldFramebuffersSet();
    ReleaseWorldTexture(worldScreenEmissionTexture);
    ReleaseWorldTexture(worldStorageTexture);

    worldScreenEmissionImage = VK_NULL_HANDLE;
    worldStorageImage = VK_NULL_HANDLE;
    worldStorageIsDummy = false;
    worldWrapWidth = 0;
    worldWrapHeight = 0;
}

void RhiSkyPass::ReleaseWorldTexture(nvrhi::TextureHandle &texture)
{
    if (texture != nullptr && frameContext != nullptr)
    {
        frameContext->Retire(texture);
    }

    texture = nullptr;
}

void RhiSkyPass::ReleaseWorldFramebuffersSet()
{
    if (worldFramebuffersSet != nullptr && frameContext != nullptr)
    {
        frameContext->Retire(worldFramebuffersSet);
    }

    worldFramebuffersSet = nullptr;
}

void RhiSkyPass::ReleaseWorldBufferSets()
{
    if (frameContext != nullptr)
    {
        if (worldUniformSet != nullptr)
        {
            frameContext->Retire(worldUniformSet);
        }

        for (nvrhi::BindingSetHandle &set : worldTonemappingSets)
        {
            if (set != nullptr)
            {
                frameContext->Retire(set);
            }
        }
    }

    worldUniformSet = nullptr;
    for (nvrhi::BindingSetHandle &set : worldTonemappingSets)
    {
        set = nullptr;
    }
}

void RhiSkyPass::UpdateWorldWraps(const Framebuffers &framebuffers, uint32_t frameIndex,
                                  uint32_t width, uint32_t height)
{
    // The engine's accessors are the record of the images the world has to use now: the engine
    // re-creates its framebuffer images on a resize (Framebuffers::PrepareForSize, whose waits the
    // ReleaseTargets contract covers) and this pass cannot see that, because NVRHI does not own
    // them. The handles the host passed to CreateWorld are the fallback for the frames the engine
    // has none yet - and then a null storage image becomes the dummy below.
    std::tuple<VkImage, VkImageView, VkFormat> screenEmission =
        framebuffers.GetScreenEmissionHandles(frameIndex);
    std::tuple<VkImage, VkImageView, VkFormat> storageImage =
        framebuffers.GetPrimaryToReflRefrHandles(frameIndex);

    if (std::get<0>(screenEmission) == VK_NULL_HANDLE)
    {
        screenEmission = worldScreenEmissionHandles;
    }
    if (std::get<0>(storageImage) == VK_NULL_HANDLE)
    {
        storageImage = worldStorageHandles;
    }

    // SCREEN_EMISSION (62) and PRIMARY_TO_REFL_REFR (25) are not swapped images (measured:
    // Bindings[62] == BindingsSwapped[62] == 62, Bindings[25] == BindingsSwapped[25] == 25), so one
    // wrap of each serves both slots and follows image or size changes.
    const bool sizeChanged = width != worldWrapWidth || height != worldWrapHeight;
    bool storageWrapChanged = false;

    if (std::get<0>(screenEmission) != worldScreenEmissionImage || sizeChanged)
    {
        ReleaseWorldTexture(worldScreenEmissionTexture);
        worldScreenEmissionImage = std::get<0>(screenEmission);
    }

    if (std::get<0>(storageImage) != worldStorageImage || sizeChanged)
    {
        ReleaseWorldTexture(worldStorageTexture);
        worldStorageIsDummy = false;
        worldStorageImage = std::get<0>(storageImage);
        storageWrapChanged = true;
    }

    worldWrapWidth = width;
    worldWrapHeight = height;

    if (worldScreenEmissionImage != VK_NULL_HANDLE && worldScreenEmissionTexture == nullptr)
    {
        worldScreenEmissionTexture = rhi::wrapEngineRenderTarget(
            device,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(worldScreenEmissionImage)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(std::get<1>(screenEmission))),
            std::get<2>(screenEmission),
            width, height,
            "RhiSky world SCREEN_EMISSION frame " + std::to_string(frameIndex));

        if (worldScreenEmissionTexture == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to wrap the world's SCREEN_EMISSION image");
        }
    }

    if (worldStorageImage != VK_NULL_HANDLE && worldStorageTexture == nullptr)
    {
        worldStorageTexture = rhi::wrapEngineStorageImage(
            device,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(worldStorageImage)),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(std::get<1>(storageImage))),
            std::get<2>(storageImage),
            width, height,
            "RhiSky world PRIMARY_TO_REFL_REFR frame " + std::to_string(frameIndex));

        if (worldStorageTexture == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to wrap the world's binding-25 storage image");
        }
    }

    // The first cut without the engine image: a 1x1 RGBA32_UINT dummy so that the shader's
    // descriptor exists. RsWorld.frag reads and writes the image (its checkerboard remap), and
    // nothing under `rhiframe` consumes the result; the dummy keeps the pass legal instead of
    // binding an unwritten or dangling descriptor.
    if (worldStorageImage == VK_NULL_HANDLE && worldStorageTexture == nullptr)
    {
        nvrhi::TextureDesc dummyDesc;
        dummyDesc.width = 1;
        dummyDesc.height = 1;
        dummyDesc.format = WORLD_STORAGE_FORMAT;
        dummyDesc.dimension = nvrhi::TextureDimension::Texture2D;
        dummyDesc.mipLevels = 1;
        dummyDesc.arraySize = 1;
        dummyDesc.sampleCount = 1;
        dummyDesc.isUAV = true;
        dummyDesc.isShaderResource = false;
        dummyDesc.isRenderTarget = false;
        // An owned texture with a known resting state: the UAV binding requires UnorderedAccess and
        // the tracker starts every list in that state, so no announcement or transition is needed.
        dummyDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        dummyDesc.keepInitialState = true;

        worldStorageTexture = rhi::createTexture(device, dummyDesc, "RhiSky world binding-25 dummy (1x1)");
        worldStorageIsDummy = worldStorageTexture != nullptr;
        storageWrapChanged = worldStorageIsDummy;

        if (!worldStorageIsDummy)
        {
            LogMessage(print, "Warning: RHI: failed to create the world's binding-25 dummy texture");
        }
    }

    // The world pipelines are built against the SCREEN_EMISSION format; a change means every
    // cached world pipeline belongs to the wrong framebuffer info.
    if (worldScreenEmissionTexture != nullptr)
    {
        const nvrhi::Format colorFormat = worldScreenEmissionTexture->getDesc().format;
        if (worldColorFormat != colorFormat)
        {
            ReleasePipelineCache();
            worldColorFormat = colorFormat;
        }
    }

    // The set 4 binding references the current storage wrap, so it follows every replacement (and
    // is retried while a replacement failed).
    if (storageWrapChanged || worldFramebuffersSet == nullptr)
    {
        ReleaseWorldFramebuffersSet();
        if (worldStorageTexture != nullptr)
        {
            nvrhi::BindingSetDesc desc;
            desc.addItem(nvrhi::BindingSetItem::Texture_UAV(WORLD_STORAGE_SLOT, worldStorageTexture));
            worldFramebuffersSet = device->createBindingSet(desc, worldFramebuffersLayout);

            if (worldFramebuffersSet == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to create the world framebuffers binding set");
            }
        }
    }
}

void RhiSkyPass::UpdateWorldFramebuffer(Target &target)
{
    target.worldValid = false;

    if (worldScreenEmissionTexture == nullptr || worldFramebuffersSet == nullptr || !target.valid)
    {
        return;
    }

    // Rebuild only when a texture the framebuffer references changed: the ALBEDO wrap and the depth
    // belong to the target and are re-created with it (ReleaseTarget drops the world framebuffer
    // then), and this check covers the shared SCREEN_EMISSION wrap.
    if (target.worldFramebuffer != nullptr &&
        target.worldScreenEmissionTexture == worldScreenEmissionTexture.Get())
    {
        target.worldValid = true;
        return;
    }

    if (target.worldFramebuffer != nullptr && frameContext != nullptr)
    {
        frameContext->Retire(target.worldFramebuffer);
    }
    target.worldFramebuffer = nullptr;

    // The attachments of the legacy world framebuffer in its own order (RasterPass.cpp:115-119):
    // colour 0 corresponds to the legacy FINAL image, which this first cut replaces with the shared
    // ALBEDO the sky draws into, colour 1 is SCREEN_EMISSION and the depth is this pass's own
    // image. The legacy world loads the traced depth instead of the sky's (RasterPass.cpp:99,
    // :267); there is no traced depth under `rhiframe`, so the world depth-tests against the sky's
    // cleared values - the recorded deviation of this first cut.
    nvrhi::FramebufferDesc framebufferDesc;
    framebufferDesc.addColorAttachment(target.albedoTexture);
    framebufferDesc.addColorAttachment(worldScreenEmissionTexture);
    framebufferDesc.setDepthAttachment(target.depthTexture);

    target.worldFramebuffer = device->createFramebuffer(framebufferDesc);
    if (target.worldFramebuffer == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the world framebuffer");
        return;
    }

    target.worldScreenEmissionTexture = worldScreenEmissionTexture.Get();
    target.worldValid = true;
}

void RhiSkyPass::ReleasePipelineCache()
{
    if (frameContext != nullptr)
    {
        for (auto &entry : pipelines)
        {
            frameContext->Retire(entry.second);
        }

        for (auto &entry : worldPipelines)
        {
            frameContext->Retire(entry.second);
        }
    }

    pipelines.clear();
    worldPipelines.clear();
}

nvrhi::IGraphicsPipeline *RhiSkyPass::GetPipeline(uint32_t stateFlags, bool applyVertexColorGamma, bool world)
{
    assert(pipelineColorFormat != nvrhi::Format::UNKNOWN);

    const uint32_t key = stateFlags | (applyVertexColorGamma ? PIPELINE_STATE_VALUE_VERTEX_COLOR_GAMMA : 0);

    // The sky and the world are two different pipelines built from the same key, so they have one
    // cache each; the key itself stays the one RasterizerPipelines uses.
    auto &cache = world ? worldPipelines : pipelines;

    const auto found = cache.find(key);
    if (found != cache.end())
    {
        return found->second;
    }

    // NVRHI creates a new pipeline object on every call and deduplicates nothing, so this map is the
    // cache (recon 4.7) and it is keyed the way RasterizerPipelines::pipelines is
    // (RasterizerPipelines.cpp:278-296).
    nvrhi::GraphicsPipelineHandle pipeline = CreatePipeline(stateFlags, applyVertexColorGamma, world);
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    const auto inserted = cache.emplace(key, std::move(pipeline));
    return inserted.first->second;
}

nvrhi::GraphicsPipelineHandle RhiSkyPass::CreatePipeline(uint32_t stateFlags, bool applyVertexColorGamma, bool world)
{
    const bool alphaTest   = (stateFlags & PIPELINE_STATE_MASK_IS_ALPHA_TEST) != 0;
    const bool blendEnable = (stateFlags & PIPELINE_STATE_MASK_BLEND_ENABLE) != 0;
    const bool depthTest   = (stateFlags & PIPELINE_STATE_MASK_DEPTH_TEST_ENABLE) != 0;
    const bool depthWrite  = (stateFlags & PIPELINE_STATE_MASK_DEPTH_WRITE_ENABLE) != 0;
    const bool isLines     = (stateFlags & PIPELINE_STATE_MASK_IS_LINES) != 0;

    // One spec constant per stage, both SpecId 0 and 4 bytes (RasterizerPipelines.cpp:315-343):
    // the vertex stage's applyVertexColorGamma and the fragment stage's alphaTest, from the state
    // key. This is what makes the per-state pipelines differ in the state the shader sees, not only
    // in the fixed-function state. RsWorld.frag declares alphaTest the same way RsSky.frag does
    // (measured in vkpt/Build/RsWorld.frag.spv), so the same specialization applies to both.
    const nvrhi::ShaderSpecialization vertexSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA, applyVertexColorGamma ? 1u : 0u);
    const nvrhi::ShaderSpecialization pixelSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_ALPHA_TEST, alphaTest ? 1u : 0u);

    nvrhi::ShaderHandle specializedVertexShader =
        device->createShaderSpecialization(vertexShader, &vertexSpecialization, 1);
    nvrhi::ShaderHandle specializedPixelShader =
        device->createShaderSpecialization(world ? worldPixelShader : pixelShader, &pixelSpecialization, 1);

    if (specializedVertexShader == nullptr || specializedPixelShader == nullptr)
    {
        LogMessage(print, world ? "Warning: RHI: failed to specialize the world shaders"
                                : "Warning: RHI: failed to specialize the sky shaders");
        return nullptr;
    }

    // The same blend attachment for every color target, with the alpha factors mirroring the color
    // ones and an add op, as the legacy pipeline state describes it (RasterizerPipelines.cpp:406-420);
    // the legacy world pass uses the same attachment state for both of its targets
    // (:427-429), so the world pipeline declares the second one explicitly.
    nvrhi::BlendState::RenderTarget blendTarget;
    blendTarget.setBlendEnable(blendEnable)
               .setSrcBlend(DecodeBlendFactor(stateFlags, PS_SRC_OFFSET))
               .setDestBlend(DecodeBlendFactor(stateFlags, PS_DST_OFFSET))
               .setBlendOp(nvrhi::BlendOp::Add)
               .setSrcBlendAlpha(DecodeBlendFactor(stateFlags, PS_SRC_OFFSET))
               .setDestBlendAlpha(DecodeBlendFactor(stateFlags, PS_DST_OFFSET))
               .setBlendOpAlpha(nvrhi::BlendOp::Add)
               .setColorWriteMask(nvrhi::ColorMask::All);

    nvrhi::GraphicsPipelineDesc desc;
    desc.setVertexShader(specializedVertexShader);
    desc.setPixelShader(specializedPixelShader);
    desc.inputLayout = inputLayout;
    desc.primType = isLines ? nvrhi::PrimitiveType::LineList : nvrhi::PrimitiveType::TriangleList;
    // Front face counter-clockwise, fill, no culling and depth clipping on, which is the legacy
    // rasterization state (RasterizerPipelines.cpp:382-390; depthClampEnable = FALSE there means
    // clipping stays enabled).
    desc.renderState.rasterState.setFillSolid();
    desc.renderState.rasterState.setCullMode(nvrhi::RasterCullMode::None);
    desc.renderState.rasterState.setFrontCounterClockwise(true);
    desc.renderState.rasterState.setDepthClipEnable(true);
    // LESS_OR_EQUAL with the test forced on whenever the depth is written, stencil off
    // (RasterizerPipelines.cpp:397-404).
    desc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
    desc.renderState.depthStencilState.setDepthTestEnable(depthTest || depthWrite);
    desc.renderState.depthStencilState.setDepthWriteEnable(depthWrite);
    desc.renderState.depthStencilState.setStencilEnable(false);
    desc.renderState.blendState.setRenderTarget(0, blendTarget);
    if (world)
    {
        desc.renderState.blendState.setRenderTarget(1, blendTarget);
    }

    // The table first, so it lands at descriptor set 0, then the rest of the sets in shader order;
    // NVRHI's legacy binding mode keeps the order the pass adds the layouts in (recon 4). The sky
    // only declares the table and the push-constant layout; the world adds the uniform (1), the
    // tonemapping (2), the set-3 hole with the push constants (3) and the framebuffers (4).
    desc.addBindingLayout(textureTable->GetLayout());
    if (world)
    {
        desc.addBindingLayout(worldUniformLayout);
        desc.addBindingLayout(worldTonemappingLayout);
        desc.addBindingLayout(worldPushConstantLayout);
        desc.addBindingLayout(worldFramebuffersLayout);
    }
    else
    {
        desc.addBindingLayout(pushConstantLayout);
    }

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.addColorFormat(pipelineColorFormat);
    if (world)
    {
        assert(worldColorFormat != nvrhi::Format::UNKNOWN);
        framebufferInfo.addColorFormat(worldColorFormat);
    }
    framebufferInfo.setDepthFormat(DEPTH_FORMAT);
    framebufferInfo.setSampleCount(1);

    nvrhi::GraphicsPipelineHandle pipeline =
        rhi::createGraphicsPipeline(device, desc, framebufferInfo,
                                    world ? "RhiSky world pipeline" : "RhiSky pipeline");

    if (pipeline == nullptr)
    {
        LogMessage(print, world ? "Warning: RHI: failed to create a world pipeline"
                                : "Warning: RHI: failed to create a sky pipeline");
    }

    return pipeline;
}

bool RhiSkyPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the raster pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
