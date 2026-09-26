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

#include "RhiRasterOverlayPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"
#include "RhiTextureTable.h"

#include <utility>

#include "../Framebuffers.h"
#include "../Matrix.h"

using namespace vkpt;

namespace
{

// The engine's blobs by the names ShaderManager knows them: "VertDefault" is RsRasterizer.vert.spv
// and "FragWorld" is RsWorld.frag.spv, exactly the pair RasterPass hands to its world
// RasterizerPipelines (RasterPass.cpp:74-75; ShaderManager.cpp:59, :62). "VertFullscreenQuad" is
// RsFullscreenQuad.vert.spv and "FragDepthCopying" is RsDepthCopying.frag.spv, the pair
// DepthCopying loads (DepthCopying.cpp:24-25; ShaderManager.cpp:64-65).
const char *const VERTEX_SHADER_FILE_NAME           = "RsRasterizer.vert.spv";
const char *const WORLD_PIXEL_SHADER_FILE_NAME      = "RsWorld.frag.spv";
const char *const DEPTH_COPY_VERTEX_SHADER_FILE_NAME = "RsFullscreenQuad.vert.spv";
const char *const DEPTH_COPY_PIXEL_SHADER_FILE_NAME  = "RsDepthCopying.frag.spv";

// The legacy push-constant range is 88 bytes (Rasterizer.cpp:66, :485-489) while RsWorld.frag
// declares a 92-byte block: its last member, emissionMultiplier at offset 88, is declared but never
// read, and the legacy host never writes those four bytes (HLSL/RsWorld.frag.hlsl:35). The pass
// mirrors the legacy value, so both renderers push the same bytes and the shader-visible prefix
// stays what the legacy path produces.
constexpr uint32_t RASTERIZED_PUSH_CONSTANT_SIZE = 88;

// The legacy DepthCopying pipeline layout declares exactly the two uints of DepthCopyingFrag_BT
// (DepthCopying.cpp:205-208) and the process pushes { width, height } to the fragment stage
// (DepthCopying.cpp:92-93). The compiled blob reads no push constants - its only descriptor is the
// DEPTH_NDC sampled view - but the NVRHI validation device requires a pipeline whose layout
// declares push constants to have them set before the draw, so the pass keeps the legacy's pair and
// pushes it, which is also the byte-for-byte faithful thing to do.
constexpr uint32_t DEPTH_COPY_PUSH_CONSTANT_SIZE = 2 * sizeof(uint32_t);

// Both stages use SpecId 0 for their single constant: RsRasterizer.vert declares
// applyVertexColorGamma, RsWorld.frag declares alphaTest, each 4 bytes and written as a uint32
// (RasterizerPipelines.cpp:315-343).
constexpr uint32_t SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA = 0;
constexpr uint32_t SPEC_CONSTANT_ALPHA_TEST = 0;

// The legacy depth buffer's format (RasterPass.cpp:29-30) and the NVRHI name of the same VkFormat.
constexpr nvrhi::Format DEPTH_FORMAT = nvrhi::Format::D32;

// The depth copy's set 0: the shader's `framebufDepthNdc_Sampled` (Texture2D<float4>, raw binding
// 136) as an NVRHI Texture_SRV. The engine's sampled-view binding is 124 + image index
// (ShFramebuffers_Sampled_Bindings), so the slot is the image index and the layout carries the
// shader-resource offset 124 - the arithmetic RhiSkyPass's world framebuffers layout documents.
constexpr uint32_t DEPTH_NDC_SRV_OFFSET = 124;
constexpr uint32_t DEPTH_NDC_SRV_SLOT = static_cast<uint32_t>(FB_IMAGE_INDEX_DEPTH_NDC);

// The world shader's set 4 binding 25: FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR, the Rgba32ui storage
// image the shader reads and writes for its emissive blend mode. The engine's storage-image
// binding is the image index, so the slot is 25; the three offsets repeat the engine layout's
// triple exactly as RhiSkyPass's world framebuffers layout does (recon 5).
constexpr uint32_t WORLD_STORAGE_SLOT = static_cast<uint32_t>(FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR);
constexpr uint32_t WORLD_FRAMEBUFFERS_SRV_OFFSET = 124;
constexpr uint32_t WORLD_FRAMEBUFFERS_UAV_OFFSET = 0;
constexpr uint32_t WORLD_FRAMEBUFFERS_SAMPLER_OFFSET = 248;

// The world shader's set 1 binding 0 is the engine's global uniform block at raw binding 0
// (BINDING_GLOBAL_UNIFORM, ShaderCommonC.h), so the layout's constant-buffer offset must be 0
// instead of the NVRHI default 256 (recon 4.2).
constexpr uint32_t WORLD_UNIFORM_OFFSET = 0;

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
        // (RasterizerPipelines.cpp:57-113), so an opaque draw reaches this decoder with zeros. The
        // factor is ignored by setBlendEnable(false) either way.
        case 0:  return nvrhi::BlendFactor::One;
        default: assert(0); return nvrhi::BlendFactor::One;
    }
}

// The per-draw block, byte for byte the legacy RasterizedPushConst (Rasterizer.cpp:33-65): the
// model-view-projection, the color, and the two texture indices. The type is repeated because
// RasterizedPushConst is private to the Rasterizer translation unit; the offsets below are the ones
// Rasterizer.cpp asserts, and the fragment half matches the members RsWorld.frag reads at 64/80/84
// (RsWorld.frag.hlsl:79-84).
struct OverlayPushConstants
{
    float    vp[16];
    float    c[4];
    uint32_t t;
    uint32_t e;

    explicit OverlayPushConstants(const RasterizedDataCollector::DrawInfo &info, const float *defaultViewProj)
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

static_assert(offsetof(OverlayPushConstants, vp) == 0);
static_assert(offsetof(OverlayPushConstants, c) == 64);
static_assert(offsetof(OverlayPushConstants, t) == 80);
static_assert(offsetof(OverlayPushConstants, e) == 84);
static_assert(sizeof(OverlayPushConstants) == 88);

// The legacy viewport of a DrawInfo, as an NVRHI viewport that makes the Vulkan backend emit the
// legacy's own VkViewport. The legacy `vkCmdSetViewport` takes (x, y, w, +h) (Rasterizer.cpp:
// 436-446), while `VKViewportWithDXCoords` (vulkan-graphics.cpp:528-531) computes
// `(minX, maxY, maxX - minX, -(maxY - minY))`: with minY = y + h and maxY = y the emitted viewport
// is (x, y, w, +h) again. The inverted rectangle is the point of the helper, and the class comment
// of the header explains why this pass - unlike RhiSkyPass's world sub-pass, which draws into the
// raster mode's NVRHI-convention ALBEDO - needs the legacy convention: FINAL is an
// engine-convention traced image.
nvrhi::Viewport ToLegacyViewport(const VkViewport &v)
{
    return nvrhi::Viewport(v.x, v.x + v.width, v.y + v.height, v.y, v.minDepth, v.maxDepth);
}

void LogMessage(const RhiRasterOverlayPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

// The numbers the input layout of Create is built from, taken from the collector's own struct and
// asserted so that a change of RgVertex cannot silently break the RHI pipeline. They are the
// offsets the legacy VkVertexInputAttributeDescriptions use (RasterizedDataCollector.cpp:31-54).
static_assert(offsetof(RgVertex, position) == 0);
static_assert(offsetof(RgVertex, texCoord) == 32);
static_assert(offsetof(RgVertex, packedColor) == 56);
static_assert(sizeof(RgVertex) == 80);

}

RhiRasterOverlayPass::RhiRasterOverlayPass() = default;

RhiRasterOverlayPass::~RhiRasterOverlayPass()
{
    if (device != nullptr)
    {
        // The framebuffers reference the device, and the borrowed texture wraps and the pipeline
        // objects reference the device; the host destroys the pass while it can still idle the
        // device (VulkanDevice does that before the skeleton as well), so nothing has to go through
        // a retire queue here.
        device->waitForIdle();
    }

    // Pipelines reference their (specialized) shaders, so they go first.
    worldPipelines.clear();
    depthCopyPipeline = nullptr;

    for (Target &target : targets)
    {
        target.framebuffer = nullptr;
        target.depthCopyFramebuffer = nullptr;
        target.depthTexture = nullptr;
        target.finalTexture = nullptr;
        target.screenEmissionTexture = nullptr;
        target.depthNdcTexture = nullptr;
        target.storageTexture = nullptr;
        target.uniformSet = nullptr;
        target.tonemappingSet = nullptr;
        target.depthCopySet = nullptr;
        target.framebuffersSet = nullptr;
        target.valid = false;
    }

    worldFramebuffersLayout = nullptr;
    worldPushConstantLayout = nullptr;
    worldTonemappingLayout = nullptr;
    worldUniformLayout = nullptr;
    depthCopyLayout = nullptr;
    worldHoleSet = nullptr;

    depthCopyPixelShader = nullptr;
    depthCopyVertexShader = nullptr;
    worldPixelShader = nullptr;
    inputLayout = nullptr;
    vertexShader = nullptr;
}

bool RhiRasterOverlayPass::Create(nvrhi::IDevice *pDevice,
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
        LogMessage(print, "Warning: RHI: the raster overlay pass needs an RHI device");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the raster overlay pass needs the shared texture table of the RHI layer");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the raster overlay pass needs the frame context of the RHI layer");
        return false;
    }

    if (!LoadShader(VERTEX_SHADER_FILE_NAME, nvrhi::ShaderType::Vertex, vertexShader) ||
        !LoadShader(WORLD_PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, worldPixelShader) ||
        !LoadShader(DEPTH_COPY_VERTEX_SHADER_FILE_NAME, nvrhi::ShaderType::Vertex, depthCopyVertexShader) ||
        !LoadShader(DEPTH_COPY_PIXEL_SHADER_FILE_NAME, nvrhi::ShaderType::Pixel, depthCopyPixelShader))
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
        LogMessage(print, "Warning: RHI: failed to create the raster overlay pass input layout");
        return false;
    }

    // The world pipeline layout in the shader's own set order: the shared texture table (0), the
    // engine's global uniform (1), the tonemapping block (2), the set-3 hole (3) and the partial
    // framebuffers layout (4). NVRHI's legacy binding mode keeps the order the layouts are added in
    // as the descriptor set numbers, so adding them in this order reproduces the numbers
    // RsWorld.frag spells (recon 4). The depth copy has one set: its DEPTH_NDC sampled view.
    {
        const nvrhi::BindingLayoutItem layoutItems[] =
        {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
        };
        const nvrhi::VulkanBindingOffsets offsets =
            nvrhi::VulkanBindingOffsets().setConstantBufferOffset(WORLD_UNIFORM_OFFSET);

        worldUniformLayout = rhi::createBindingLayout(device, layoutItems, "RhiRasterOverlay world uniform", &offsets);
    }
    {
        // The engine's tonemapping set is a raw binding 0 storage buffer; the default NVRHI offsets
        // already put a StructuredBuffer_SRV there (shaderResource 0).
        const nvrhi::BindingLayoutItem layoutItems[] =
        {
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
        };

        worldTonemappingLayout = rhi::createBindingLayout(device, layoutItems, "RhiRasterOverlay world tonemapping");
    }

    // Set 3 is the hole between the tonemapping set and the framebuffers set: RsWorld.frag declares
    // no set 3 (the engine's volumetric set is dead while volumeEnableType == 0), but the
    // framebuffers set has to land at 4, so a layout must occupy the position. It is a real,
    // zero-item layout that carries the pipeline's only push-constant item: the backend skips that
    // item when it builds the Vulkan bindings (vulkan-resource-bindings.cpp:90-94) and takes the
    // push range from it (vulkan-resource-bindings.cpp:1110-1121), so the layout still creates an
    // empty descriptor set layout.
    {
        const nvrhi::BindingLayoutItem layoutItems[] =
        {
            nvrhi::BindingLayoutItem::PushConstants(0, RASTERIZED_PUSH_CONSTANT_SIZE),
        };

        worldPushConstantLayout = rhi::createBindingLayout(device, layoutItems, "RhiRasterOverlay world set 3 (push constants)");
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

        worldFramebuffersLayout = rhi::createBindingLayout(device, layoutItems, "RhiRasterOverlay world framebuffers (binding 25)", &offsets);
    }

    // The depth copy's one layout: the shader's sampled DEPTH_NDC view at raw 136 and the legacy's
    // 8-byte fragment push block in the same set. The push item must live in exactly one layout of
    // a pipeline - the backend assumes a single push block across all layouts
    // (vulkan-resource-bindings.cpp:1118) - and this pipeline has no other set.
    {
        const nvrhi::BindingLayoutItem layoutItems[] =
        {
            nvrhi::BindingLayoutItem::Texture_SRV(DEPTH_NDC_SRV_SLOT),
            nvrhi::BindingLayoutItem::PushConstants(0, DEPTH_COPY_PUSH_CONSTANT_SIZE),
        };
        const nvrhi::VulkanBindingOffsets offsets =
            nvrhi::VulkanBindingOffsets().setShaderResourceOffset(DEPTH_NDC_SRV_OFFSET);

        depthCopyLayout = rhi::createBindingLayout(device, layoutItems, "RhiRasterOverlay depth copy (DEPTH_NDC SRV)", &offsets);
    }

    if (worldUniformLayout == nullptr || worldTonemappingLayout == nullptr ||
        worldPushConstantLayout == nullptr || worldFramebuffersLayout == nullptr ||
        depthCopyLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a raster overlay pipeline layout");
        return false;
    }

    // The empty set that fills the set-3 hole. A null entry in GraphicsState::bindings is unsafe in
    // the pinned NVRHI: the automatic-barrier pass dereferences every entry before the bind pass
    // gets to treat a null as a hole (vulkan-state-tracking.cpp:105), while an empty BindingSet is a
    // normal set that happens to bind no descriptors. Allocating one from a zero-item layout is
    // legal Vulkan (a descriptor pool with no pool sizes serves set layouts with no bindings).
    worldHoleSet = device->createBindingSet(nvrhi::BindingSetDesc(), worldPushConstantLayout);
    if (worldHoleSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the empty set of the raster overlay set-3 hole");
        return false;
    }

    // The depth copy's pipeline, the legacy's one static pipeline (DepthCopying.cpp:223-329): no
    // vertex input (the fullscreen quad is driven by SV_VertexID, DepthCopying.cpp:240-249), a
    // depth-only target, `comparison ALWAYS` with the write on so `SV_Depth` replaces the value
    // (DepthCopying.cpp:278-286), and the legacy's rasterization state (:263-271).
    {
        nvrhi::GraphicsPipelineDesc desc;
        desc.setVertexShader(depthCopyVertexShader);
        desc.setPixelShader(depthCopyPixelShader);
        desc.inputLayout = nullptr;
        desc.primType = nvrhi::PrimitiveType::TriangleList;
        desc.renderState.rasterState.setFillSolid();
        desc.renderState.rasterState.setCullMode(nvrhi::RasterCullMode::None);
        desc.renderState.rasterState.setFrontCounterClockwise(true);
        desc.renderState.rasterState.setDepthClipEnable(true);
        desc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Always);
        desc.renderState.depthStencilState.setDepthTestEnable(true);
        desc.renderState.depthStencilState.setDepthWriteEnable(true);
        desc.renderState.depthStencilState.setStencilEnable(false);
        // No blend state: the depth copy has no colour attachment (the legacy's render pass declares
        // one, DepthCopying.cpp:297-301, but with attachmentCount 0 the backend ignores it).
        desc.addBindingLayout(depthCopyLayout);

        nvrhi::FramebufferInfo framebufferInfo;
        framebufferInfo.setDepthFormat(DEPTH_FORMAT);
        framebufferInfo.setSampleCount(1);

        depthCopyPipeline = rhi::createGraphicsPipeline(device, desc, framebufferInfo, "RhiRasterOverlay depth copy pipeline");
        if (depthCopyPipeline == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay depth copy pipeline");
            return false;
        }
    }

    created = true;
    return true;
}

bool RhiRasterOverlayPass::SetTonemappingBuffers(nvrhi::IBuffer *const pTonemappingBuffers[MAX_FRAMES_IN_FLIGHT])
{
    if (!created)
    {
        LogMessage(print, "Warning: RHI: the raster overlay pass needs the pass created first");
        return false;
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (pTonemappingBuffers == nullptr || pTonemappingBuffers[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: the raster overlay pass needs the wrapped tonemapping buffer of every frame slot (set 2)");
            return false;
        }

        // The shader declares set 2 as StructuredBuffer<ShTonemapping>; the backend asserts a
        // non-zero structStride when the binding set is created
        // (vulkan-resource-bindings.cpp:535-536), so a wrap without
        // Tonemapping::GetElementSize() would trip a debug assert inside NVRHI.
        if (pTonemappingBuffers[i]->getDesc().structStride == 0)
        {
            LogMessage(print, "Warning: RHI: the raster overlay tonemapping buffer needs Tonemapping::GetElementSize() as structStride");
            return false;
        }
    }

    memcpy(tonemappingBuffers, pTonemappingBuffers, sizeof(tonemappingBuffers));
    return true;
}

void RhiRasterOverlayPass::SetGeometryBuffers(nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer)
{
    vertexBuffer = pVertexBuffer;
    indexBuffer = pIndexBuffer;
}

void RhiRasterOverlayPass::Render(nvrhi::ICommandList *pCommandList,
                                  uint32_t frameIndex,
                                  const Framebuffers *pFramebuffers,
                                  uint32_t width,
                                  uint32_t height,
                                  const float jitter[2],
                                  nvrhi::IBuffer *pUniformBuffer,
                                  const RasterizedDataCollector::DrawInfo *pDraws,
                                  uint32_t drawCount,
                                  const float *pView,
                                  const float *pProj,
                                  bool applyVertexColorGamma)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT ||
        pFramebuffers == nullptr || width == 0 || height == 0)
    {
        return;
    }

    if (vertexBuffer == nullptr || indexBuffer == nullptr)
    {
        if (!warnedMissingGeometry)
        {
            warnedMissingGeometry = true;
            LogMessage(print, "Warning: RHI: the raster overlay pass has no geometry buffers, the overlay is skipped");
        }
        return;
    }

    if (pView == nullptr || pProj == nullptr || jitter == nullptr)
    {
        if (!warnedMissingCamera)
        {
            warnedMissingCamera = true;
            LogMessage(print, "Warning: RHI: the raster overlay pass got no view/projection/jitter, the overlay is skipped");
        }
        return;
    }

    if (pUniformBuffer == nullptr || pUniformBuffer->getDesc().isVolatile ||
        !pUniformBuffer->getDesc().isConstantBuffer)
    {
        // A volatile buffer would become a dynamic-offset binding (nvrhi.h:2311-2318), which the
        // static ConstantBuffer layout item rejects, and the validation device refuses a
        // ConstantBuffer binding on a desc without isConstantBuffer (validation-device.cpp:
        // 1717-1730). The skeleton's wrap of the engine uniform is neither.
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the raster overlay pass needs the static wrap of the global uniform (set 1)");
        }
        return;
    }

    if (tonemappingBuffers[frameIndex] == nullptr)
    {
        if (!warnedMissingTonemapping)
        {
            warnedMissingTonemapping = true;
            LogMessage(print, "Warning: RHI: the raster overlay pass has no tonemapping buffer, the overlay is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    if (!PrepareTarget(pCommandList, frameIndex, target, *pFramebuffers, width, height, pUniformBuffer))
    {
        return;
    }

    // The pass-owned depth: the legacy copies DEPTH_NDC into it right before the draws
    // (RasterPass.cpp:96-99) and the world pass then loads it (RasterPass.cpp:263-275). Runs for an
    // empty draw list too, which is what the legacy does - PrepareForFinal precedes the empty-list
    // early-out of Rasterizer::Draw (Rasterizer.cpp:262-263, :341-347).
    RecordDepthCopy(pCommandList, target, width, height);

    // The legacy has no draws for an empty list; with one it builds the default view-projection and
    // runs the state/push/draw loop of Rasterizer::Draw (:266-271, :394-420).
    if (pDraws != nullptr && drawCount > 0)
    {
        // The engine textures the table wrapped since the last frame need their first-use state
        // declared in the first list that binds the table (RhiTextureSource.h); this may be that
        // list.
        textureTable->TrackPendingTextures(pCommandList);

        // The legacy's ApplyJitter and default view-projection (Rasterizer.cpp:176-184, :266-271):
        // the jitter is added to the third row of the projection in pixels of the render resolution,
        // then the frame's plain view multiplies it. The world pass deliberately does not move the
        // viewer position (the sky's SetNewViewerPosition stays out of this path).
        float jitteredProj[16];
        memcpy(jitteredProj, pProj, sizeof(jitteredProj));
        jitteredProj[2 * 4 + 0] += jitter[0] / float(width);
        jitteredProj[2 * 4 + 1] += jitter[1] / float(height);

        float defaultViewProj[16];
        Matrix::Multiply(defaultViewProj, pView, jitteredProj);

        RecordWorldDraws(pCommandList, target, width, height, defaultViewProj,
                         pDraws, drawCount, applyVertexColorGamma);
    }

    // The framebuffer use left FINAL and SCREEN_EMISSION in the render-target layout, and the
    // binding-25 storage image is a shader-write state; the compose's continuation reads all of
    // them through its own wraps, whose tracked state is UnorderedAccess from the compose's own
    // announcement - naming GENERAL as the old layout is only correct because the physical layouts
    // are moved back here, at the end of the list that used them. The engine's own descriptors
    // declare VK_IMAGE_LAYOUT_GENERAL for every framebuffer image (Framebuffers.cpp:754-758, :786).
    pCommandList->setTextureState(target.finalTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->setTextureState(target.screenEmissionTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->setTextureState(target.depthNdcTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->setTextureState(target.storageTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
}

bool RhiRasterOverlayPass::PrepareTarget(nvrhi::ICommandList *pCommandList, uint32_t frameIndex,
                                         Target &target, const Framebuffers &framebuffers,
                                         uint32_t width, uint32_t height, nvrhi::IBuffer *pUniformBuffer)
{
    // The engine images of this slot, exactly the ones the legacy pass touches: FINAL and
    // SCREEN_EMISSION as the two colour attachments, DEPTH_NDC as the depth copy's source and
    // PRIMARY_TO_REFL_REFR as the shader's binding 25. Framebuffers resolves the slot's own image
    // inside (Framebuffers.cpp:33-53, the `_Prev` role permutation; none of the four is one of the
    // swapped history images), so the pass keeps every wrap per slot and follows whatever the
    // accessors answer - an engine framebuffer re-create is picked up without a second Create.
    const std::tuple<VkImage, VkImageView, VkFormat> finalImage =
        framebuffers.GetImageHandles(FB_IMAGE_INDEX_FINAL, frameIndex);
    const std::tuple<VkImage, VkImageView, VkFormat> screenEmission =
        framebuffers.GetScreenEmissionHandles(frameIndex);
    const std::tuple<VkImage, VkImageView, VkFormat> depthNdc =
        framebuffers.GetImageHandles(FB_IMAGE_INDEX_DEPTH_NDC, frameIndex);
    const std::tuple<VkImage, VkImageView, VkFormat> storageImage =
        framebuffers.GetPrimaryToReflRefrHandles(frameIndex);

    if (std::get<0>(finalImage) == VK_NULL_HANDLE || std::get<1>(finalImage) == VK_NULL_HANDLE ||
        std::get<2>(finalImage) == VK_FORMAT_UNDEFINED ||
        std::get<0>(screenEmission) == VK_NULL_HANDLE || std::get<1>(screenEmission) == VK_NULL_HANDLE ||
        std::get<2>(screenEmission) == VK_FORMAT_UNDEFINED ||
        std::get<0>(depthNdc) == VK_NULL_HANDLE || std::get<1>(depthNdc) == VK_NULL_HANDLE ||
        std::get<2>(depthNdc) == VK_FORMAT_UNDEFINED ||
        std::get<0>(storageImage) == VK_NULL_HANDLE || std::get<1>(storageImage) == VK_NULL_HANDLE ||
        std::get<2>(storageImage) == VK_FORMAT_UNDEFINED)
    {
        if (!warnedMissingTargets)
        {
            warnedMissingTargets = true;
            LogMessage(print, "Warning: RHI: the raster overlay pass got no engine framebuffer image for frame " + std::to_string(frameIndex));
        }
        target.valid = false;
        return false;
    }

    // Nothing to do while the slot still wraps the same images at the same size: replacing the
    // wraps and the framebuffers every frame would create four textures, one depth and two
    // framebuffers per frame for nothing. The state announcement below still has to happen, because
    // a render-target wrap keeps no state between command lists (RhiTextureSource.h:56-70).
    const bool targetChanged =
        !target.valid ||
        target.finalImage != std::get<0>(finalImage) ||
        target.screenEmissionImage != std::get<0>(screenEmission) ||
        target.depthNdcImage != std::get<0>(depthNdc) ||
        target.storageImage != std::get<0>(storageImage) ||
        target.width != width || target.height != height;

    if (targetChanged)
    {
        ReleaseTarget(target);

        if (!CreateTargetObjects(target, finalImage, screenEmission, depthNdc, storageImage,
                                 frameIndex, width, height))
        {
            ReleaseTarget(target);
            return false;
        }
    }

    // The engine leaves every framebuffer image in VK_IMAGE_LAYOUT_GENERAL - Framebuffers creates
    // each image and immediately barriers it there (Framebuffers.cpp:754-758) and never leaves that
    // layout - which NVRHI names UnorderedAccess. Announcing that real state lets the first
    // framebuffer use and the first SRV binding emit the correct transitions; the compose's
    // checkerboard left FINAL and SCREEN_EMISSION in exactly that state, and the raygens left
    // DEPTH_NDC and the binding-25 image there.
    pCommandList->beginTrackingTextureState(target.finalTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->beginTrackingTextureState(target.screenEmissionTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->beginTrackingTextureState(target.depthNdcTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);
    pCommandList->beginTrackingTextureState(target.storageTexture, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);

    return UpdateBufferSets(target, frameIndex, pUniformBuffer);
}

bool RhiRasterOverlayPass::CreateTargetObjects(
    Target &target,
    const std::tuple<VkImage, VkImageView, VkFormat> &finalImage,
    const std::tuple<VkImage, VkImageView, VkFormat> &screenEmission,
    const std::tuple<VkImage, VkImageView, VkFormat> &depthNdc,
    const std::tuple<VkImage, VkImageView, VkFormat> &storageImage,
    uint32_t frameIndex, uint32_t width, uint32_t height)
{
    const std::string frameTag = std::to_string(frameIndex);

    const auto imageHandle = [](const std::tuple<VkImage, VkImageView, VkFormat> &handles)
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(std::get<0>(handles)));
    };
    const auto viewHandle = [](const std::tuple<VkImage, VkImageView, VkFormat> &handles)
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(std::get<1>(handles)));
    };

    // The three render-target wraps and the storage wrap. The storage image is bound as a UAV by
    // the world shader (set 4 binding 25) and rests in GENERAL, so it takes
    // wrapEngineStorageImage; the other three take wrapEngineRenderTarget (both helpers document
    // their state contracts in RhiTextureSource.h).
    target.finalTexture = rhi::wrapEngineRenderTarget(
        device, imageHandle(finalImage), viewHandle(finalImage), std::get<2>(finalImage),
        width, height, "RhiRasterOverlay FINAL frame " + frameTag);
    target.screenEmissionTexture = rhi::wrapEngineRenderTarget(
        device, imageHandle(screenEmission), viewHandle(screenEmission), std::get<2>(screenEmission),
        width, height, "RhiRasterOverlay SCREEN_EMISSION frame " + frameTag);
    target.depthNdcTexture = rhi::wrapEngineStorageImage(
        device, imageHandle(depthNdc), viewHandle(depthNdc), std::get<2>(depthNdc),
        width, height, "RhiRasterOverlay DEPTH_NDC frame " + frameTag);
    target.storageTexture = rhi::wrapEngineStorageImage(
        device, imageHandle(storageImage), viewHandle(storageImage), std::get<2>(storageImage),
        width, height, "RhiRasterOverlay PRIMARY_TO_REFL_REFR frame " + frameTag);

    if (target.finalTexture == nullptr || target.screenEmissionTexture == nullptr ||
        target.depthNdcTexture == nullptr || target.storageTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to wrap an engine image of the raster overlay pass");
        return false;
    }

    // The pass-owned depth, per frame slot like RasterPass's depthImages (RasterPass.h:96-98) and in
    // the legacy format (RasterPass.cpp:29). initialState/keepInitialState make NVRHI return it to
    // DepthWrite when a list closes; the depth copy's full-screen write defines every value, so
    // unlike the legacy path there is no clear.
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

    target.depthTexture = rhi::createTexture(device, depthDesc, "RhiRasterOverlay depth frame " + frameTag);
    if (target.depthTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the raster overlay depth image");
        return false;
    }

    // The depth copy's framebuffer: the pass-owned depth alone. NVRHI's loadOp is always LOAD
    // (vulkan-graphics.cpp:80), which the full-screen write makes equivalent to the legacy's
    // DONT_CARE (DepthCopying.cpp:128).
    {
        nvrhi::FramebufferDesc framebufferDesc;
        framebufferDesc.setDepthAttachment(target.depthTexture);

        target.depthCopyFramebuffer = device->createFramebuffer(framebufferDesc);
        if (target.depthCopyFramebuffer == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay depth copy framebuffer");
            return false;
        }
    }

    // The world framebuffer over the attachments of the legacy world framebuffer in its own order
    // (RasterPass.cpp:115-119): colour 0 is FINAL, colour 1 is SCREEN_EMISSION and the depth is the
    // copied one. The world pass loads the depth the copy just wrote (RasterPass.cpp:263-275).
    {
        nvrhi::FramebufferDesc framebufferDesc;
        framebufferDesc.addColorAttachment(target.finalTexture);
        framebufferDesc.addColorAttachment(target.screenEmissionTexture);
        framebufferDesc.setDepthAttachment(target.depthTexture);

        target.framebuffer = device->createFramebuffer(framebufferDesc);
        if (target.framebuffer == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay framebuffer");
            return false;
        }
    }

    // The two sets over the per-slot engine wraps.
    {
        nvrhi::BindingSetDesc desc;
        desc.addItem(nvrhi::BindingSetItem::Texture_SRV(DEPTH_NDC_SRV_SLOT, target.depthNdcTexture));
        target.depthCopySet = device->createBindingSet(desc, depthCopyLayout);

        if (target.depthCopySet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay depth copy binding set");
            return false;
        }
    }
    {
        nvrhi::BindingSetDesc desc;
        desc.addItem(nvrhi::BindingSetItem::Texture_UAV(WORLD_STORAGE_SLOT, target.storageTexture));
        target.framebuffersSet = device->createBindingSet(desc, worldFramebuffersLayout);

        if (target.framebuffersSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay framebuffers binding set (binding 25)");
            return false;
        }
    }

    // The world pipelines are built against the two colour formats; a change means every cached
    // pipeline belongs to the wrong framebuffer info. FINAL and SCREEN_EMISSION are both
    // VK_FORMAT_B10G11R11_UFLOAT_PACK32 (ShaderCommonCFramebuf.cpp:37, :71), but the wrap's own
    // format is authoritative because it is the one NVRHI mapped the engine format to.
    const nvrhi::Format color0Format = target.finalTexture->getDesc().format;
    const nvrhi::Format color1Format = target.screenEmissionTexture->getDesc().format;
    if (pipelineColor0Format != color0Format || pipelineColor1Format != color1Format)
    {
        ReleasePipelineCache();
        pipelineColor0Format = color0Format;
        pipelineColor1Format = color1Format;
    }

    target.finalImage = std::get<0>(finalImage);
    target.screenEmissionImage = std::get<0>(screenEmission);
    target.depthNdcImage = std::get<0>(depthNdc);
    target.storageImage = std::get<0>(storageImage);
    target.width = width;
    target.height = height;
    target.valid = true;

    return true;
}

bool RhiRasterOverlayPass::UpdateBufferSets(Target &target, uint32_t frameIndex, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformBuffer != pUniformBuffer || target.uniformSet == nullptr)
    {
        if (target.uniformSet != nullptr && frameContext != nullptr)
        {
            frameContext->Retire(target.uniformSet);
        }

        target.uniformSet = nullptr;

        nvrhi::BindingSetDesc desc;
        desc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, pUniformBuffer));
        target.uniformSet = device->createBindingSet(desc, worldUniformLayout);

        if (target.uniformSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay uniform set");
            target.uniformBuffer = nullptr;
            return false;
        }

        target.uniformBuffer = pUniformBuffer;
    }

    nvrhi::IBuffer *tonemappingBuffer = tonemappingBuffers[frameIndex];
    if (target.tonemappingBuffer != tonemappingBuffer || target.tonemappingSet == nullptr)
    {
        if (target.tonemappingSet != nullptr && frameContext != nullptr)
        {
            frameContext->Retire(target.tonemappingSet);
        }

        target.tonemappingSet = nullptr;

        nvrhi::BindingSetDesc desc;
        desc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, tonemappingBuffer));
        target.tonemappingSet = device->createBindingSet(desc, worldTonemappingLayout);

        if (target.tonemappingSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the raster overlay tonemapping set");
            target.tonemappingBuffer = nullptr;
            return false;
        }

        target.tonemappingBuffer = tonemappingBuffer;
    }

    return true;
}

void RhiRasterOverlayPass::RecordDepthCopy(nvrhi::ICommandList *pCommandList, const Target &target,
                                           uint32_t width, uint32_t height)
{
    // The legacy Process (DepthCopying.cpp:54-113): the fullscreen quad, the fragment push block
    // with the render size, and one draw of six vertices. The quad's fragment shader writes
    // `framebufDepthNdc_Sampled.Load(int3(fragCoord.xy, 0)).r` into the depth attachment
    // (RsDepthCopying.frag.hlsl:43-50), which covers every pixel of the D32.
    const uint32_t push[2] = { width, height };

    const VkViewport legacyViewport = { 0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f };

    nvrhi::GraphicsState state;
    state.pipeline = depthCopyPipeline;
    state.framebuffer = target.depthCopyFramebuffer;
    state.viewport.addViewport(ToLegacyViewport(legacyViewport));
    state.viewport.addScissorRect(nvrhi::Rect(0, static_cast<int>(width), 0, static_cast<int>(height)));
    // Set 0 is the only layout of this pipeline; it carries the DEPTH_NDC sampled view and the push
    // block, and it has no other descriptors to bind.
    state.addBindingSet(target.depthCopySet);

    pCommandList->setGraphicsState(state);

    // After the state: changing the state invalidates push constants (nvrhi.h:3430-3432), and the
    // block is the legacy's { width, height } pair (DepthCopying.cpp:92-93).
    pCommandList->setPushConstants(push, sizeof(push));

    nvrhi::DrawArguments args;
    args.vertexCount = 6;
    pCommandList->draw(args);
}

void RhiRasterOverlayPass::RecordWorldDraws(nvrhi::ICommandList *pCommandList, const Target &target,
                                            uint32_t width, uint32_t height, const float *defaultViewProj,
                                            const RasterizedDataCollector::DrawInfo *pDraws,
                                            uint32_t drawCount, bool applyVertexColorGamma)
{
    // The legacy loop keeps the scissor at the whole render area and switches only the viewport
    // (Rasterizer.cpp:356-357, :389-396).
    const nvrhi::Rect fullTarget = nvrhi::Rect(0, static_cast<int>(width), 0, static_cast<int>(height));

    // The legacy default viewport: {0, 0, width, height, 0, 1}, through the same conversion as the
    // per-draw viewports.
    const VkViewport legacyDefaultViewport = { 0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f };
    const nvrhi::Viewport defaultViewport = ToLegacyViewport(legacyDefaultViewport);

    for (uint32_t i = 0; i < drawCount; i++)
    {
        const RasterizedDataCollector::DrawInfo &info = pDraws[i];

        // A draw with no vertices and no indices records nothing: the legacy loop would issue a
        // zero-vertex draw (Rasterizer.cpp:412-419), which rasterizes no primitive. The collector
        // itself rejects nothing for this stream (only SWAPCHAIN loses its depth state,
        // RasterizedDataCollector.cpp:153-167), so this is the only skip.
        if (info.vertexCount == 0 && info.indexCount == 0)
        {
            continue;
        }

        const uint32_t stateFlags =
            ConvertToStateFlags(info.pipelineState, info.blendFuncSrc, info.blendFuncDst);

        nvrhi::IGraphicsPipeline *pipeline = GetWorldPipeline(stateFlags, applyVertexColorGamma);
        if (pipeline == nullptr)
        {
            // A permanent failure (a shader specialization or the pipeline itself); drawing the
            // rest of the overlay with the wrong state would be worse than stopping.
            if (!warnedFailedPipeline)
            {
                warnedFailedPipeline = true;
                LogMessage(print, "Warning: RHI: failed to create a raster overlay pipeline, the overlay is incomplete");
            }
            return;
        }

        const nvrhi::Viewport viewport =
            info.viewport ? ToLegacyViewport(*info.viewport) : defaultViewport;

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = target.framebuffer;
        state.viewport.addViewport(viewport);
        state.viewport.addScissorRect(fullTarget);
        // Sets 0..4 in the layout order of Create: the table, the uniform, the slot's tonemapping
        // buffer, the real empty set of the set-3 hole and the partial framebuffers set with the
        // binding-25 storage image.
        state.addBindingSet(textureTable->GetTable());
        state.addBindingSet(target.uniformSet);
        state.addBindingSet(target.tonemappingSet);
        state.addBindingSet(worldHoleSet);
        state.addBindingSet(target.framebuffersSet);
        state.addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vertexBuffer).setSlot(0).setOffset(0));
        state.setIndexBuffer(nvrhi::IndexBufferBinding()
                                 .setBuffer(indexBuffer)
                                 .setFormat(nvrhi::Format::R32_UINT)
                                 .setOffset(0));

        pCommandList->setGraphicsState(state);

        // After the state: changing the state invalidates push constants (nvrhi.h:3430-3432), and
        // the block is rebuilt per draw exactly as Rasterizer::Draw does (Rasterizer.cpp:399-409).
        const OverlayPushConstants push(info, defaultViewProj);
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

void RhiRasterOverlayPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRasterOverlayPass::ReleaseTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the sets and the framebuffers reference wraps of engine images the GPU may still be
    // reading, and the depth texture is an RHI-owned image with the same problem (RhiFrameContext.h
    // documents the queue). The queue takes its reference now, so the handles below can be cleared
    // immediately.
    if (frameContext != nullptr)
    {
        if (target.uniformSet != nullptr)
        {
            frameContext->Retire(target.uniformSet);
        }
        if (target.tonemappingSet != nullptr)
        {
            frameContext->Retire(target.tonemappingSet);
        }
        if (target.depthCopySet != nullptr)
        {
            frameContext->Retire(target.depthCopySet);
        }
        if (target.framebuffersSet != nullptr)
        {
            frameContext->Retire(target.framebuffersSet);
        }
        if (target.framebuffer != nullptr)
        {
            frameContext->Retire(target.framebuffer);
        }
        if (target.depthCopyFramebuffer != nullptr)
        {
            frameContext->Retire(target.depthCopyFramebuffer);
        }
        if (target.depthTexture != nullptr)
        {
            frameContext->Retire(target.depthTexture);
        }
        if (target.finalTexture != nullptr)
        {
            frameContext->Retire(target.finalTexture);
        }
        if (target.screenEmissionTexture != nullptr)
        {
            frameContext->Retire(target.screenEmissionTexture);
        }
        if (target.depthNdcTexture != nullptr)
        {
            frameContext->Retire(target.depthNdcTexture);
        }
        if (target.storageTexture != nullptr)
        {
            frameContext->Retire(target.storageTexture);
        }
    }

    target.framebuffer = nullptr;
    target.depthCopyFramebuffer = nullptr;
    target.depthTexture = nullptr;
    target.finalTexture = nullptr;
    target.screenEmissionTexture = nullptr;
    target.depthNdcTexture = nullptr;
    target.storageTexture = nullptr;
    target.uniformSet = nullptr;
    target.tonemappingSet = nullptr;
    target.depthCopySet = nullptr;
    target.framebuffersSet = nullptr;
    target.uniformBuffer = nullptr;
    target.tonemappingBuffer = nullptr;
    target.finalImage = VK_NULL_HANDLE;
    target.screenEmissionImage = VK_NULL_HANDLE;
    target.depthNdcImage = VK_NULL_HANDLE;
    target.storageImage = VK_NULL_HANDLE;
    target.width = 0;
    target.height = 0;
    target.valid = false;
}

void RhiRasterOverlayPass::ReleasePipelineCache()
{
    if (frameContext != nullptr)
    {
        for (auto &entry : worldPipelines)
        {
            frameContext->Retire(entry.second);
        }
    }

    worldPipelines.clear();
}

nvrhi::IGraphicsPipeline *RhiRasterOverlayPass::GetWorldPipeline(uint32_t stateFlags, bool applyVertexColorGamma)
{
    assert(pipelineColor0Format != nvrhi::Format::UNKNOWN);
    assert(pipelineColor1Format != nvrhi::Format::UNKNOWN);

    const uint32_t key = stateFlags | (applyVertexColorGamma ? PIPELINE_STATE_VALUE_VERTEX_COLOR_GAMMA : 0);

    const auto found = worldPipelines.find(key);
    if (found != worldPipelines.end())
    {
        return found->second;
    }

    // NVRHI creates a new pipeline object on every call and deduplicates nothing, so this map is the
    // cache and it is keyed the way RasterizerPipelines::pipelines is (RasterizerPipelines.cpp:
    // 278-296).
    nvrhi::GraphicsPipelineHandle pipeline = CreateWorldPipeline(stateFlags, applyVertexColorGamma);
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    const auto inserted = worldPipelines.emplace(key, std::move(pipeline));
    return inserted.first->second;
}

nvrhi::GraphicsPipelineHandle RhiRasterOverlayPass::CreateWorldPipeline(uint32_t stateFlags, bool applyVertexColorGamma)
{
    const bool alphaTest   = (stateFlags & PIPELINE_STATE_MASK_IS_ALPHA_TEST) != 0;
    const bool blendEnable = (stateFlags & PIPELINE_STATE_MASK_BLEND_ENABLE) != 0;
    const bool depthTest   = (stateFlags & PIPELINE_STATE_MASK_DEPTH_TEST_ENABLE) != 0;
    const bool depthWrite  = (stateFlags & PIPELINE_STATE_MASK_DEPTH_WRITE_ENABLE) != 0;
    const bool isLines     = (stateFlags & PIPELINE_STATE_MASK_IS_LINES) != 0;

    // One spec constant per stage, both SpecId 0 and 4 bytes (RasterizerPipelines.cpp:315-343):
    // the vertex stage's applyVertexColorGamma and the fragment stage's alphaTest, from the state
    // key. RsWorld.frag declares alphaTest the same way RsSky.frag does, so the same specialization
    // applies.
    const nvrhi::ShaderSpecialization vertexSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA, applyVertexColorGamma ? 1u : 0u);
    const nvrhi::ShaderSpecialization pixelSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_ALPHA_TEST, alphaTest ? 1u : 0u);

    nvrhi::ShaderHandle specializedVertexShader =
        device->createShaderSpecialization(vertexShader, &vertexSpecialization, 1);
    nvrhi::ShaderHandle specializedPixelShader =
        device->createShaderSpecialization(worldPixelShader, &pixelSpecialization, 1);

    if (specializedVertexShader == nullptr || specializedPixelShader == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to specialize the raster overlay shaders");
        return nullptr;
    }

    // The legacy world pass uses the same blend attachment state for both of its colour targets
    // (RasterizerPipelines.cpp:406-429), with the alpha factors mirroring the colour ones and an add
    // op, so the pipeline declares the second target explicitly.
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
    // (RasterizerPipelines.cpp:397-404). The depth is the copied DEPTH_NDC, so this is the legacy's
    // depth test against the traced scene.
    desc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
    desc.renderState.depthStencilState.setDepthTestEnable(depthTest || depthWrite);
    desc.renderState.depthStencilState.setDepthWriteEnable(depthWrite);
    desc.renderState.depthStencilState.setStencilEnable(false);
    desc.renderState.blendState.setRenderTarget(0, blendTarget);
    desc.renderState.blendState.setRenderTarget(1, blendTarget);

    // The table first, so it lands at descriptor set 0, then the rest of the sets in shader order;
    // NVRHI's legacy binding mode keeps the order the pass adds the layouts in. The world shader
    // declares the table (0), the uniform (1), the tonemapping (2), the dead volumetric set (3, the
    // hole that carries the push constants) and the framebuffers set (4).
    desc.addBindingLayout(textureTable->GetLayout());
    desc.addBindingLayout(worldUniformLayout);
    desc.addBindingLayout(worldTonemappingLayout);
    desc.addBindingLayout(worldPushConstantLayout);
    desc.addBindingLayout(worldFramebuffersLayout);

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.addColorFormat(pipelineColor0Format);
    framebufferInfo.addColorFormat(pipelineColor1Format);
    framebufferInfo.setDepthFormat(DEPTH_FORMAT);
    framebufferInfo.setSampleCount(1);

    nvrhi::GraphicsPipelineHandle pipeline =
        rhi::createGraphicsPipeline(device, desc, framebufferInfo, "RhiRasterOverlay world pipeline");

    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a raster overlay world pipeline");
    }

    return pipeline;
}

bool RhiRasterOverlayPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the raster overlay pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
