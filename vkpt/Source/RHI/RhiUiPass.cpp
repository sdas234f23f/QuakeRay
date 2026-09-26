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

#include "RhiUiPass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiTextureTable.h"

#include <utility>

#include "../Matrix.h"

using namespace vkpt;

namespace
{

// The engine's blobs by the names ShaderManager knows them: "VertDefault" is RsRasterizer.vert.spv
// and "FragSwapchain" is RsSwapchain.frag.spv (ShaderManager.cpp:61-62), exactly the pair
// SwapchainPass hands to its RasterizerPipelines (SwapchainPass.cpp:37-45). The fragment blob is
// code-identical to RsSky.frag.spv, the blob RhiSkyPass loads.
const char *const VERTEX_SHADER_FILE_NAME = "RsRasterizer.vert.spv";
const char *const PIXEL_SHADER_FILE_NAME  = "RsSwapchain.frag.spv";

// The legacy push-constant range is 88 bytes (Rasterizer.cpp:66, :485-489) while RsSwapchain.frag
// declares a 92-byte block: its last member, emissionMultiplier at offset 88, is declared but never
// read, and the legacy host never writes those four bytes (HLSL/RsSwapchain.frag.hlsl:33-39). The
// pass mirrors the legacy value, so both renderers push the same bytes and the shader-visible
// prefix stays what the legacy path produces.
constexpr uint32_t RASTERIZED_PUSH_CONSTANT_SIZE = 88;

// Both stages use SpecId 0 for their single constant: RsRasterizer.vert declares
// applyVertexColorGamma, RsSwapchain.frag declares alphaTest, each 4 bytes and written as a uint32
// (RasterizerPipelines.cpp:315-343).
constexpr uint32_t SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA = 0;
constexpr uint32_t SPEC_CONSTANT_ALPHA_TEST = 0;

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
        // (RasterizerPipelines.cpp:57-113), so an opaque or alpha-tested UI draw reaches this
        // decoder with zeros. The factor is ignored by setBlendEnable(false) either way.
        case 0:  return nvrhi::BlendFactor::One;
        default: assert(0); return nvrhi::BlendFactor::One;
    }
}

// The per-draw block, byte for byte the legacy RasterizedPushConst (Rasterizer.cpp:33-65): the
// model-view-projection, the color, and the two texture indices. The type is repeated because
// RasterizedPushConst is private to the Rasterizer translation unit; the offsets below are the ones
// Rasterizer.cpp asserts, and the fragment half matches the members RsSwapchain.frag reads at
// 64/80/84.
struct UiPushConstants
{
    float    vp[16];
    float    c[4];
    uint32_t t;
    uint32_t e;

    explicit UiPushConstants(const RasterizedDataCollector::DrawInfo &info, const float *defaultViewProj)
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

static_assert(offsetof(UiPushConstants, vp) == 0);
static_assert(offsetof(UiPushConstants, c) == 64);
static_assert(offsetof(UiPushConstants, t) == 80);
static_assert(offsetof(UiPushConstants, e) == 84);
static_assert(sizeof(UiPushConstants) == 88);

// The legacy viewport of a DrawInfo, as an NVRHI viewport that makes the Vulkan backend emit the
// legacy's own VkViewport. The legacy `vkCmdSetViewport` takes (x, y, w, +h) (Rasterizer.cpp:
// 436-446), while `VKViewportWithDXCoords` (vulkan-graphics.cpp:528-531) computes
// `(minX, maxY, maxX - minX, -(maxY - minY))`: with minY = y + h and maxY = y the emitted viewport
// is (x, y, w, +h) again. The inverted rectangle is the point of the helper; the class comment of
// the header explains why the UI needs the legacy convention while the raster passes use NVRHI's.
nvrhi::Viewport ToLegacyViewport(const VkViewport &v)
{
    return nvrhi::Viewport(v.x, v.x + v.width, v.y + v.height, v.y, v.minDepth, v.maxDepth);
}

void LogMessage(const RhiUiPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

// The numbers the input layout of Create is built from, taken from the collector's own struct and
// asserted so that a change of RgVertex cannot silently break the RHI pipeline. They are the
// offsets the legacy VkVertexInputAttributeDescriptions use (RasterizedDataCollector.cpp:31-54);
// the collector is the authority.
static_assert(offsetof(RgVertex, position) == 0);
static_assert(offsetof(RgVertex, texCoord) == 32);
static_assert(offsetof(RgVertex, packedColor) == 56);
static_assert(sizeof(RgVertex) == 80);

}

RhiUiPass::RhiUiPass() = default;

RhiUiPass::~RhiUiPass()
{
    if (device != nullptr)
    {
        // The framebuffers reference the device, and the borrowed texture wraps and the pipeline
        // objects reference the device; the host destroys the pass while it can still idle the
        // device (VulkanDevice does that before the skeleton as well), so nothing has to go
        // through a retire queue here.
        device->waitForIdle();
    }

    // Pipelines reference their (specialized) shaders, so they go first.
    pipelines.clear();

    for (Target &target : targets)
    {
        target.framebuffer = nullptr;
        target.target = nullptr;
        target.width = 0;
        target.height = 0;
    }

    pushConstantLayout = nullptr;
    inputLayout = nullptr;
    vertexShader = nullptr;
    pixelShader = nullptr;
}

bool RhiUiPass::Create(nvrhi::IDevice *pDevice,
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
        LogMessage(print, "Warning: RHI: the UI pass needs an RHI device");
        return false;
    }

    if (textureTable == nullptr || !textureTable->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the UI pass needs the shared texture table of the RHI layer");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the UI pass needs the frame context of the RHI layer");
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
        LogMessage(print, "Warning: RHI: failed to create the UI pass input layout");
        return false;
    }

    // The pipeline's second layout, and its only job is the push-constant block: no descriptors, so
    // no binding set has to be created or bound for it. The texture table stays the first layout and
    // therefore descriptor set 0, which is the shader's DESC_SET_TEXTURES.
    const nvrhi::BindingLayoutItem layoutItems[] =
    {
        nvrhi::BindingLayoutItem::PushConstants(0, RASTERIZED_PUSH_CONSTANT_SIZE),
    };

    pushConstantLayout = rhi::createBindingLayout(device, layoutItems, "RhiUi push constants");
    if (pushConstantLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the UI pass push-constant layout");
        return false;
    }

    created = true;
    return true;
}

void RhiUiPass::SetGeometryBuffers(nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer)
{
    vertexBuffer = pVertexBuffer;
    indexBuffer = pIndexBuffer;
}

void RhiUiPass::Render(nvrhi::ICommandList *pCommandList,
                       uint32_t frameIndex,
                       nvrhi::ITexture *pTarget,
                       uint32_t targetWidth,
                       uint32_t targetHeight,
                       const RasterizedDataCollector::DrawInfo *pDraws,
                       uint32_t drawCount,
                       const float *pView,
                       const float *pProj,
                       bool applyVertexColorGamma)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT ||
        pTarget == nullptr || targetWidth == 0 || targetHeight == 0 ||
        pDraws == nullptr || drawCount == 0)
    {
        return;
    }

    if (vertexBuffer == nullptr || indexBuffer == nullptr)
    {
        if (!warnedMissingGeometry)
        {
            warnedMissingGeometry = true;
            LogMessage(print, "Warning: RHI: the UI pass has no geometry buffers, the overlay is skipped");
        }
        return;
    }

    if (pView == nullptr || pProj == nullptr)
    {
        if (!warnedMissingCamera)
        {
            warnedMissingCamera = true;
            LogMessage(print, "Warning: RHI: the UI pass got no view/projection, the overlay is skipped");
        }
        return;
    }

    // The wrap's own extent is what the backend asserts the framebuffer against
    // (vulkan-graphics.cpp:67-68), so a mismatch is a host error and the frame is skipped instead
    // of building an invalid framebuffer.
    const nvrhi::TextureDesc &targetDesc = pTarget->getDesc();
    if (targetDesc.width != targetWidth || targetDesc.height != targetHeight)
    {
        if (!warnedMissingTarget)
        {
            warnedMissingTarget = true;
            LogMessage(print, "Warning: RHI: the UI pass got a target of a different extent, the overlay is skipped");
        }
        return;
    }

    Target &target = targets[frameIndex];

    // Nothing to do while the slot still wraps the same target at the same size: replacing the
    // framebuffer every frame would create one per frame for nothing.
    if (target.framebuffer == nullptr || target.target != pTarget ||
        target.width != targetWidth || target.height != targetHeight)
    {
        ReleaseTarget(target);

        target.target = pTarget;
        target.width = targetWidth;
        target.height = targetHeight;

        // The one-attachment framebuffer of the legacy swapchain render pass (SwapchainPass.cpp:
        // 135-183): the colour target only, no depth. NVRHI's loadOp is always LOAD
        // (vulkan-graphics.cpp:80), which is the legacy loadOp too.
        nvrhi::FramebufferDesc framebufferDesc;
        framebufferDesc.addColorAttachment(pTarget);

        target.framebuffer = device->createFramebuffer(framebufferDesc);
        if (target.framebuffer == nullptr)
        {
            if (!warnedMissingTarget)
            {
                warnedMissingTarget = true;
                LogMessage(print, "Warning: RHI: failed to create the UI framebuffer");
            }
            return;
        }

        // The pipeline is built against the framebuffer's colour format, so a change means every
        // cached pipeline belongs to the wrong framebuffer and has to be replaced. The upscaled
        // image is VK_FORMAT_B10G11R11_UFLOAT_PACK32 (ShaderCommonCFramebuf.cpp:9-10), which the
        // wrapped texture reports as nvrhi::Format::R11G11B10_FLOAT; the wrap's own format is
        // authoritative because it is the one NVRHI mapped the engine format to.
        const nvrhi::Format colorFormat = targetDesc.format;
        if (pipelineColorFormat != colorFormat)
        {
            ReleasePipelineCache();
            pipelineColorFormat = colorFormat;
        }
    }

    // The fallback view-projection, built the way Rasterizer::DrawToSwapchain builds it
    // (Rasterizer.cpp:311-312): the frame's plain view times the projection, no jitter. Today every
    // SWAPCHAIN upload passes its own ortho, so this value serves only a draw that carries none.
    float defaultViewProj[16];
    Matrix::Multiply(defaultViewProj, pView, pProj);

    // The engine textures the table wrapped since the last frame need their first-use state
    // declared in the first list that binds the table (RhiTextureSource.h); this may be that list.
    textureTable->TrackPendingTextures(pCommandList);

    // The target rests in the engine's VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - after
    // the TAAU wrote it through its storage image, and the compose's render-target wrap keeps no
    // state between command lists (RhiTextureSource.h), so the real state is announced every list.
    // The framebuffer use below moves the image to COLOR_ATTACHMENT_OPTIMAL.
    pCommandList->beginTrackingTextureState(pTarget, nvrhi::AllSubresources,
                                            nvrhi::ResourceStates::UnorderedAccess);

    // The full render area the legacy loop keeps its scissor at (Rasterizer.cpp:356-357): it never
    // switches the scissor per draw.
    const nvrhi::Rect fullTarget = nvrhi::Rect(0, static_cast<int>(targetWidth), 0, static_cast<int>(targetHeight));

    // The legacy viewport default: {0, 0, width, height, 0, 1}, through the same conversion as the
    // per-draw viewports, so the backend emits the same positive-height VkViewport.
    const VkViewport legacyDefaultViewport = { 0.0f, 0.0f, float(targetWidth), float(targetHeight), 0.0f, 1.0f };
    const nvrhi::Viewport defaultViewport = ToLegacyViewport(legacyDefaultViewport);

    for (uint32_t i = 0; i < drawCount; i++)
    {
        const RasterizedDataCollector::DrawInfo &info = pDraws[i];

        // The entries the legacy never draws for this stream: the collector rejects a SWAPCHAIN
        // upload that asks for the depth test or the depth write (RasterizedDataCollector.cpp:
        // 153-167), so such an entry cannot come from the game - and the legacy swapchain render
        // pass has no depth attachment, so drawing it would be invalid. The pass skips it instead
        // of building a depth-tested pipeline.
        if (info.pipelineState & (RG_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST |
                                  RG_RASTERIZED_GEOMETRY_STATE_DEPTH_WRITE))
        {
            if (!warnedDepthState)
            {
                warnedDepthState = true;
                LogMessage(print, "Warning: RHI: the UI pass skipped a draw with depth state, the SWAPCHAIN stream has no depth");
            }
            continue;
        }

        // A draw with no vertices and no indices records nothing: the legacy loop would issue a
        // zero-vertex draw (Rasterizer.cpp:412-419), which rasterizes no primitive.
        if (info.vertexCount == 0 && info.indexCount == 0)
        {
            continue;
        }

        const uint32_t stateFlags =
            ConvertToStateFlags(info.pipelineState, info.blendFuncSrc, info.blendFuncDst);

        nvrhi::IGraphicsPipeline *pipeline = GetPipeline(stateFlags, applyVertexColorGamma);
        if (pipeline == nullptr)
        {
            // A permanent failure (a shader specialization or the pipeline itself); drawing the
            // rest of the overlay with the wrong state would be worse than stopping.
            if (!warnedFailedPipeline)
            {
                warnedFailedPipeline = true;
                LogMessage(print, "Warning: RHI: failed to create a UI pipeline, the overlay is incomplete");
            }
            return;
        }

        // The viewport of the draw, or the whole target: the legacy loop switches only the
        // viewport and keeps the scissor at the full render area (Rasterizer.cpp:389-396).
        const nvrhi::Viewport viewport =
            info.viewport ? ToLegacyViewport(*info.viewport) : defaultViewport;

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = target.framebuffer;
        state.viewport.addViewport(viewport);
        state.viewport.addScissorRect(fullTarget);
        // Set 0 is the bindless texture table, the pipeline's first layout; the second layout only
        // carries the push constants and has no descriptors to bind.
        state.addBindingSet(textureTable->GetTable());
        state.addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vertexBuffer).setSlot(0).setOffset(0));
        state.setIndexBuffer(nvrhi::IndexBufferBinding()
                                 .setBuffer(indexBuffer)
                                 .setFormat(nvrhi::Format::R32_UINT)
                                 .setOffset(0));

        pCommandList->setGraphicsState(state);

        // After the state: changing the state invalidates push constants (nvrhi.h:3430-3432), and
        // the block is rebuilt per draw exactly as Rasterizer::Draw does (Rasterizer.cpp:399-409).
        const UiPushConstants push(info, defaultViewProj);
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

    // The framebuffer use left the target in the render-target layout, while the engine's own
    // descriptors declare VK_IMAGE_LAYOUT_GENERAL for it - NVRHI's UnorderedAccess - and the
    // present announces exactly that before it samples the image (NvrhiFrameSkeleton.cpp:573-580,
    // :624-646). Moving the image back here, at the end of the list that used it, is the same
    // pattern RhiSkyPass::RenderWorld applies to SCREEN_EMISSION.
    pCommandList->setTextureState(pTarget, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::UnorderedAccess);
}

void RhiUiPass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiUiPass::ReleaseTarget(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the framebuffer references a wrap of an engine image the GPU may still be reading.
    // The queue takes its reference now, so the handle below can be cleared immediately.
    if (frameContext != nullptr && target.framebuffer != nullptr)
    {
        frameContext->Retire(target.framebuffer);
    }

    target.framebuffer = nullptr;
    target.target = nullptr;
    target.width = 0;
    target.height = 0;
}

void RhiUiPass::ReleasePipelineCache()
{
    if (frameContext != nullptr)
    {
        for (auto &entry : pipelines)
        {
            frameContext->Retire(entry.second);
        }
    }

    pipelines.clear();
}

nvrhi::IGraphicsPipeline *RhiUiPass::GetPipeline(uint32_t stateFlags, bool applyVertexColorGamma)
{
    assert(pipelineColorFormat != nvrhi::Format::UNKNOWN);

    const uint32_t key = stateFlags | (applyVertexColorGamma ? PIPELINE_STATE_VALUE_VERTEX_COLOR_GAMMA : 0);

    const auto found = pipelines.find(key);
    if (found != pipelines.end())
    {
        return found->second;
    }

    // NVRHI creates a new pipeline object on every call and deduplicates nothing, so this map is
    // the cache and it is keyed the way RasterizerPipelines::pipelines is (RasterizerPipelines.cpp:
    // 278-296).
    nvrhi::GraphicsPipelineHandle pipeline = CreatePipeline(stateFlags, applyVertexColorGamma);
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    const auto inserted = pipelines.emplace(key, std::move(pipeline));
    return inserted.first->second;
}

nvrhi::GraphicsPipelineHandle RhiUiPass::CreatePipeline(uint32_t stateFlags, bool applyVertexColorGamma)
{
    const bool alphaTest   = (stateFlags & PIPELINE_STATE_MASK_IS_ALPHA_TEST) != 0;
    const bool blendEnable = (stateFlags & PIPELINE_STATE_MASK_BLEND_ENABLE) != 0;
    const bool isLines     = (stateFlags & PIPELINE_STATE_MASK_IS_LINES) != 0;

    // One spec constant per stage, both SpecId 0 and 4 bytes (RasterizerPipelines.cpp:315-343):
    // the vertex stage's applyVertexColorGamma and the fragment stage's alphaTest, from the state
    // key. This is what makes the per-state pipelines differ in the state the shader sees, not only
    // in the fixed-function state.
    const nvrhi::ShaderSpecialization vertexSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_APPLY_VERTEX_COLOR_GAMMA, applyVertexColorGamma ? 1u : 0u);
    const nvrhi::ShaderSpecialization pixelSpecialization =
        nvrhi::ShaderSpecialization::UInt32(SPEC_CONSTANT_ALPHA_TEST, alphaTest ? 1u : 0u);

    nvrhi::ShaderHandle specializedVertexShader =
        device->createShaderSpecialization(vertexShader, &vertexSpecialization, 1);
    nvrhi::ShaderHandle specializedPixelShader =
        device->createShaderSpecialization(pixelShader, &pixelSpecialization, 1);

    if (specializedVertexShader == nullptr || specializedPixelShader == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to specialize the UI shaders");
        return nullptr;
    }

    // The blend attachment of the legacy pipeline state, with the alpha factors mirroring the
    // colour ones and an add op (RasterizerPipelines.cpp:406-420); the UI has one colour target,
    // so there is exactly one attachment (additionalAttachmentsCount = 0 in SwapchainPass.cpp:44).
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
    // The swapchain pass has one colour attachment and no depth attachment (SwapchainPass.cpp:
    // 135-183), and the only reachable SWAPCHAIN state flags exclude the depth bits
    // (RasterizedDataCollector.cpp:153-167), so the depth test and write stay off for every
    // pipeline of the key space - the legacy creation with depthTestEnable = depthTest || depthWrite
    // (RasterizerPipelines.cpp:397-404) can never raise them here. LESS_OR_EQUAL and stencil-off
    // mirror the same block.
    desc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
    desc.renderState.depthStencilState.setDepthTestEnable(false);
    desc.renderState.depthStencilState.setDepthWriteEnable(false);
    desc.renderState.depthStencilState.setStencilEnable(false);
    desc.renderState.blendState.setRenderTarget(0, blendTarget);

    // The table first, so it lands at descriptor set 0, then the push-constant layout; NVRHI's
    // legacy binding mode keeps the order the pass adds the layouts in. The swapchain shaders
    // declare no other set.
    desc.addBindingLayout(textureTable->GetLayout());
    desc.addBindingLayout(pushConstantLayout);

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.addColorFormat(pipelineColorFormat);
    framebufferInfo.setSampleCount(1);

    nvrhi::GraphicsPipelineHandle pipeline =
        rhi::createGraphicsPipeline(device, desc, framebufferInfo, "RhiUi pipeline");

    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a UI pipeline");
    }

    return pipeline;
}

bool RhiUiPass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the UI pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
