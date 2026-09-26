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

#include "RhiShadowMapPass.h"

#include "RhiPipeline.h"
#include "RhiResources.h"

#include "../Generated/ShaderCommonC.h"
#include "../VertexCollector.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>

using namespace vkpt;

namespace
{

// The engine blob, by the name ShaderManager knows it: "ShadowMap" is ShadowMap.vert.spv
// (ShaderManager.cpp:57) - the only stage the legacy shadow map pipeline has (ShadowMap.cpp:313-409).
const char *const VERTEX_SHADER_FILE_NAME = "ShadowMap.vert.spv";

// The legacy push-constant range is 32 floats (ShadowMap.cpp:296-305): the view-projection at offset
// 0, written once per Render, and the model matrix at offset 64, written per draw. The HLSL push
// block declares the same two float4x4 members at the same offsets (measured: ShadowMap.vert.hlsl
// members 0/1 at offsets 0/64 with MatrixStride 16, RowMajor), so the bytes below are the block the
// shader reads.
struct ShadowMapPushConstants
{
    float viewProjection[16];
    float model[16];
};

static_assert(offsetof(ShadowMapPushConstants, viewProjection) == 0);
static_assert(offsetof(ShadowMapPushConstants, model) == 64);
static_assert(sizeof(ShadowMapPushConstants) == 128);

// The numbers the input layout is built from, taken from the two structs the geometry actually uses:
// the legacy pipeline binds RgVertex (ShadowMap.cpp:325, :332) while the collector stores the
// byte-identical ShVertex (VertexCollector.cpp:296 sets vertexStride = sizeof(ShVertex), and
// ShaderCommonC.h:203-216 declares it). Asserting both shapes keeps the RHI pipeline and the data
// from drifting: position first, 80-byte stride.
static_assert(offsetof(RgVertex, position) == 0);
static_assert(sizeof(RgVertex) == 80);
static_assert(offsetof(ShVertex, position) == 0);
static_assert(sizeof(ShVertex) == sizeof(RgVertex));

// The legacy depth image: VK_FORMAT_D32_SFLOAT, 4096x4096, one mip, one layer, one sample
// (ShadowMap.cpp:161-203). NVRHI calls the format D32 (vulkan-constants.cpp:93) and the legacy
// clear value is 1.0 (ShadowMap.cpp:590-591).
constexpr nvrhi::Format DEPTH_FORMAT = nvrhi::Format::D32;
constexpr float DEPTH_CLEAR_VALUE = 1.0f;

constexpr uint32_t PUSH_CONSTANT_SIZE = uint32_t(sizeof(ShadowMapPushConstants));

void LogMessage(const RhiShadowMapPass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

// The vector helpers of the legacy helper (vkpt::Utils::Normalize, Utils.cpp:216-231, and
// vkpt::Utils::Cross, Utils.cpp:248-253), reimplemented here so that this translation unit does not
// pull in Utils.h and its Vulkan declarations. The arithmetic is the same, including the 0.01 length
// threshold and the "a degenerate direction becomes zero" fallback of the legacy Normalize.
float Dot3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float Length3(const float v[3])
{
    return std::sqrt(Dot3(v, v));
}

void Normalize3(float inout[3])
{
    const float len = Length3(inout);

    if (len > 0.01f)
    {
        inout[0] /= len;
        inout[1] /= len;
        inout[2] /= len;
    }
    else
    {
        assert(0);
        inout[0] = inout[1] = inout[2] = 0.0f;
    }
}

void Cross3(const float a[3], const float b[3], float r[3])
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}

// The 4x4 product of the engine's Matrix::Multiply (Matrix.cpp:210-223), reimplemented with the same
// index arithmetic and the same summation order. For the column-major storage both matrices use,
// `result[i * 4 + j] = sum_k a[i * 4 + k] * b[k * 4 + j]` computes result = b * a, which is why the
// legacy calls Multiply(out, viewMatrix, projectionMatrix) to get Proj * View (ShadowMap.cpp:
// 131-136).
void Multiply4x4(float *result, const float *a, const float *b)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            result[i * 4 + j] =
                a[i * 4 + 0] * b[0 * 4 + j] +
                a[i * 4 + 1] * b[1 * 4 + j] +
                a[i * 4 + 2] * b[2 * 4 + j] +
                a[i * 4 + 3] * b[3 * 4 + j];
        }
    }
}

// The faithful reimplementation of the legacy file-local ComputeViewProjection (ShadowMap.cpp:
// 36-139). Every step is the same: the world-up basis with the z-axis special case, the
// from-sun look direction used as-is (the legacy comment at ShadowMap.cpp:48-54 explains why the
// direction must NOT be negated - the camera looks FROM the sun), the view matrix whose rows are
// left/up/look, the world AABB fitted into view space, the squared XY footprint, the [0,1]
// orthographic projection, the `Proj * View` product and the returned depth extent.
//
// 'sunDirection' is the from-sun direction the host's shadowLightDir carries (VulkanDevice.cpp:
// 945-951). 'aabbMin'/'aabbMax' are the world bounds (VulkanDevice.cpp:953-956). Both outputs are
// always written, exactly like the legacy helper, which Render calls before it checks whether there
// is anything to draw.
void ComputeViewProjection(const float sunDirection[3],
                           const float aabbMin[3], const float aabbMax[3],
                           float outViewProjection[16], float *outDepthScale)
{
    float upDir[3] = {0, 0, 1};
    // handle both sun straight up (z=+1) and straight down / zenith (z=-1);
    // in both cases lookDir is parallel to world up and the cross product would be zero
    if (std::fabs(sunDirection[2]) >= 0.99f)
    {
        upDir[0] = 1.0f; upDir[1] = 0.0f; upDir[2] = 0.0f;
    }

    // The caller passes the FROM-sun light direction, so the shadow camera looks from the sun down
    // at the scene; see the reference above.
    float lookDir[3] = {sunDirection[0], sunDirection[1], sunDirection[2]};
    Normalize3(lookDir);

    float leftDir[3];
    Cross3(upDir, lookDir, leftDir);
    Normalize3(leftDir);

    Cross3(lookDir, leftDir, upDir);
    Normalize3(upDir);

    // view matrix (column-major); rows are left/up/look
    float viewMatrix[16] = {
        leftDir[0], upDir[0], lookDir[0], 0.0f,
        leftDir[1], upDir[1], lookDir[1], 0.0f,
        leftDir[2], upDir[2], lookDir[2], 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    // fit world AABB into view space
    float viewMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float viewMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (int i = 0; i < 8; i++)
    {
        const float corner[4] = {
            (i & 1) ? aabbMax[0] : aabbMin[0],
            (i & 2) ? aabbMax[1] : aabbMin[1],
            (i & 4) ? aabbMax[2] : aabbMin[2],
            1.0f,
        };

        for (int k = 0; k < 4; k++)
        {
            const float v =
                viewMatrix[0 * 4 + k] * corner[0] +
                viewMatrix[1 * 4 + k] * corner[1] +
                viewMatrix[2 * 4 + k] * corner[2] +
                viewMatrix[3 * 4 + k] * corner[3];

            if (k < 3)
            {
                viewMin[k] = std::min(viewMin[k], v);
                viewMax[k] = std::max(viewMax[k], v);
            }
        }
    }

    float diagonal[3];
    for (int k = 0; k < 3; k++)
    {
        diagonal[k] = viewMax[k] - viewMin[k];
    }

    const float maxXY = std::max(diagonal[0], diagonal[1]);

    // make the XY footprint square
    viewMin[0] -= (maxXY - diagonal[0]) * 0.5f;
    viewMin[1] -= (maxXY - diagonal[1]) * 0.5f;
    viewMax[0] += (maxXY - diagonal[0]) * 0.5f;
    viewMax[1] += (maxXY - diagonal[1]) * 0.5f;

    // orthographic projection (column-major)
    const float width  = viewMax[0] - viewMin[0];
    const float height = viewMax[1] - viewMin[1];
    const float depth  = viewMax[2] - viewMin[2];

    float projectionMatrix[16] = {
        2.0f / width, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / height, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f / depth, 0.0f,
        -(viewMax[0] + viewMin[0]) / width,
        -(viewMax[1] + viewMin[1]) / height,
        -viewMin[2] / depth,
        1.0f,
    };

    // Proj * View = world -> view -> clip, the convention every other engine matrix uses; see the
    // Multiply4x4 comment for the operand order.
    Multiply4x4(outViewProjection, viewMatrix, projectionMatrix);

    *outDepthScale = depth;
}

}

// One collector's GeometryDrawInfos in the shape the shadow map consumes. The collector builds them
// from its CPU-side transforms and its live geometry descriptors (VertexCollector.cpp:871-940), so
// they are valid under `rhiframe`; the two VkBuffer fields of every entry are dropped because the
// NVRHI handles of RhiShadowMapPass::GeometryBuffers are what the RHI pass binds instead.
//
// The definition sits outside the anonymous namespace: a member function has to be defined in a
// namespace that encloses its class, and the helpers above deliberately keep file-local linkage.
void RhiShadowMapPass::MakeDrawItems(const VertexCollector *pCollector, std::vector<DrawItem> &outItems)
{
    outItems.clear();

    if (pCollector == nullptr)
    {
        return;
    }

    const std::vector<VertexCollector::GeometryDrawInfo> infos = pCollector->GetGeometryDrawInfos();
    outItems.reserve(infos.size());

    for (const VertexCollector::GeometryDrawInfo &info : infos)
    {
        DrawItem item = {};
        memcpy(item.model, info.model, sizeof(item.model));
        item.baseVertex = info.baseVertex;
        item.firstIndex = info.firstIndex;
        item.indexCount = info.indexCount;
        outItems.push_back(item);
    }
}

RhiShadowMapPass::RhiShadowMapPass() = default;

RhiShadowMapPass::~RhiShadowMapPass()
{
    if (device != nullptr)
    {
        // The pipeline, the framebuffer and the two textures reference the device and the shader
        // module; the host destroys the pass while it can still idle the device (VulkanDevice does
        // that before the skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    // The pipeline references its shader and layout, so it goes first.
    pipeline = nullptr;
    framebuffer = nullptr;
    shadowSampler = nullptr;
    depthTexture = nullptr;
    pushConstantLayout = nullptr;
    inputLayout = nullptr;
    vertexShader = nullptr;
}

bool RhiShadowMapPass::Create(nvrhi::IDevice *pDevice,
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

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the shadow map pass needs an RHI device");
        return false;
    }

    if (!LoadShader(VERTEX_SHADER_FILE_NAME, vertexShader))
    {
        return false;
    }

    // The vertex input of the legacy pipeline (ShadowMap.cpp:323-339): one binding at slot 0 with the
    // RgVertex stride and one attribute, the position at offset 0 as RGB32_FLOAT (the shader's
    // location 0 reads .xyz of the vec4 input, the same as the legacy). The stride assertion above
    // pins the value to the collector's ShVertex.
    const uint32_t vertexStride = uint32_t(sizeof(RgVertex));

    const nvrhi::VertexAttributeDesc vertexAttributes[] =
    {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setBufferIndex(0)
            .setOffset(offsetof(RgVertex, position))
            .setElementStride(vertexStride),
    };

    // The vertex shader argument is ignored by the Vulkan backend (vulkan-shader.cpp:136-138) and
    // passed for the D3D backends NVRHI supports.
    inputLayout = device->createInputLayout(vertexAttributes, uint32_t(std::size(vertexAttributes)), vertexShader);
    if (inputLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map input layout");
        return false;
    }

    // The pipeline's only layout: the 128-byte push-constant block of the legacy pipeline
    // (ShadowMap.cpp:296-311). It declares no descriptors, so the pass creates and binds no binding
    // set. The helper gives the layout ShaderType::All visibility, which is a superset of the
    // legacy's VK_SHADER_STAGE_VERTEX_BIT and is what every RHI pass in this layer uses
    // (RhiPipeline.h documents the helper).
    const nvrhi::BindingLayoutItem layoutItems[] =
    {
        nvrhi::BindingLayoutItem::PushConstants(0, PUSH_CONSTANT_SIZE),
    };

    pushConstantLayout = rhi::createBindingLayout(device, layoutItems, "RhiShadowMap push constants");
    if (pushConstantLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map push-constant layout");
        return false;
    }

    // The depth image (ShadowMap.cpp:161-203): 4096x4096, D32, one mip, one layer, one sample, both
    // sampled and an attachment. keepInitialState = DepthWrite makes the attachment state the state
    // every command list starts from and returns to (the class comment has the contract); the clear
    // of each Render makes the contents of the previous frame irrelevant, exactly like the legacy
    // render pass's VK_ATTACHMENT_LOAD_OP_CLEAR over an UNDEFINED-sourced transition.
    nvrhi::TextureDesc depthDesc;
    depthDesc.width = SHADOW_MAP_SIZE;
    depthDesc.height = SHADOW_MAP_SIZE;
    depthDesc.format = DEPTH_FORMAT;
    depthDesc.dimension = nvrhi::TextureDimension::Texture2D;
    depthDesc.mipLevels = 1;
    depthDesc.arraySize = 1;
    depthDesc.sampleCount = 1;
    depthDesc.isShaderResource = true;
    depthDesc.isRenderTarget = true;
    depthDesc.isUAV = false;
    depthDesc.initialState = nvrhi::ResourceStates::DepthWrite;
    depthDesc.keepInitialState = true;

    depthTexture = rhi::createTexture(device, depthDesc, "RhiShadowMap 4096x4096 (D32)");
    if (depthTexture == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map depth image");
        return false;
    }

    // The sampler of ShadowMap.cpp:210-228 as an nvrhi::SamplerDesc: linear minification and
    // magnification (minFilter/magFilter = true mean vk::Filter::eLinear), nearest mip mode
    // (mipFilter = false), clamp-to-border on all three axes, the opaque-white border
    // (VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; NVRHI's pickSamplerBorderColor maps exactly the three
    // special colors - transparent black, opaque black and opaque white, vulkan-texture.cpp:824-849
    // - and opaque white is the legacy's) and reduction MIN (VK_SAMPLER_REDUCTION_MODE_MIN; NVRHI
    // attaches the VkSamplerReductionModeCreateInfoEXT itself, vulkan-texture.cpp:874-882). The
    // god-rays shader samples with SampleLevel(..., 0), so the mip mode is a shape detail, but it is
    // the legacy's and is kept.
    const nvrhi::SamplerDesc samplerDesc = nvrhi::SamplerDesc()
        .setMinFilter(true)
        .setMagFilter(true)
        .setMipFilter(false)
        .setAllAddressModes(nvrhi::SamplerAddressMode::ClampToBorder)
        .setBorderColor(nvrhi::Color(1.0f, 1.0f, 1.0f, 1.0f))
        .setReductionType(nvrhi::SamplerReductionType::Minimum);

    shadowSampler = rhi::createSampler(device, samplerDesc, "RhiShadowMap sampler (MIN, clamp-to-border)");
    if (shadowSampler == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map sampler");
        return false;
    }

    // The framebuffer of the legacy pass (ShadowMap.cpp:277-289): the depth image as the only
    // attachment, no colour attachments.
    nvrhi::FramebufferDesc framebufferDesc;
    framebufferDesc.setDepthAttachment(depthTexture);

    framebuffer = device->createFramebuffer(framebufferDesc);
    if (framebuffer == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map framebuffer");
        return false;
    }

    // The depth-only pipeline: the vertex shader alone (no fragment stage, like the legacy pipeline
    // that sets no VkPipelineShaderStageCreateInfo for one), triangle list, front-face culling with
    // the clockwise front face, fill solid with depth clipping on, LESS with depth test and write
    // on, stencil off, no blend targets (ShadowMap.cpp:341-400). The raster/viewport/scissor states
    // are dynamic (the backend always makes viewport and scissor dynamic,
    // vulkan-graphics.cpp:390-393), so only the viewport and scissor are set per Render.
    nvrhi::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.setVertexShader(vertexShader);
    pipelineDesc.inputLayout = inputLayout;
    pipelineDesc.primType = nvrhi::PrimitiveType::TriangleList;
    pipelineDesc.renderState.rasterState.setFillSolid();
    pipelineDesc.renderState.rasterState.setCullMode(nvrhi::RasterCullMode::Front);
    pipelineDesc.renderState.rasterState.setFrontCounterClockwise(false);
    pipelineDesc.renderState.rasterState.setDepthClipEnable(true);
    pipelineDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    pipelineDesc.renderState.depthStencilState.setDepthTestEnable(true);
    pipelineDesc.renderState.depthStencilState.setDepthWriteEnable(true);
    pipelineDesc.renderState.depthStencilState.setStencilEnable(false);
    pipelineDesc.addBindingLayout(pushConstantLayout);

    nvrhi::FramebufferInfo framebufferInfo;
    framebufferInfo.setDepthFormat(DEPTH_FORMAT);
    framebufferInfo.setSampleCount(1);

    pipeline = rhi::createGraphicsPipeline(device, pipelineDesc, framebufferInfo, "RhiShadowMap pipeline");
    if (pipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the shadow map pipeline");
        return false;
    }

    created = true;
    return true;
}

bool RhiShadowMapPass::Render(nvrhi::ICommandList *pCommandList,
                              const float sunDirection[3],
                              const float aabbMin[3], const float aabbMax[3],
                              const VertexCollector *pStaticCollector,
                              const VertexCollector *pDynamicCollector,
                              const GeometryBuffers &geometryBuffers,
                              float outViewProjection[16], float *outDepthScale)
{
    if (sunDirection == nullptr || aabbMin == nullptr || aabbMax == nullptr ||
        outViewProjection == nullptr || outDepthScale == nullptr)
    {
        assert(0);
        return false;
    }

    // The legacy computes the view-projection before it looks at the geometry and writes both
    // outputs even when it draws nothing (ShadowMap.cpp:546-554); the god-rays params then carry
    // them unchanged. The single image keeps a single pair of these values coherent.
    ComputeViewProjection(sunDirection, aabbMin, aabbMax, outViewProjection, outDepthScale);
    memcpy(lastViewProjection, outViewProjection, sizeof(lastViewProjection));
    lastDepthScale = *outDepthScale;

    // The two lists of the legacy loops (ShadowMap.cpp:548-549). Both are host-side data - the
    // collector builds them from its CPU-side transforms - so they are valid under `rhiframe`.
    std::vector<DrawItem> staticDraws;
    std::vector<DrawItem> dynamicDraws;
    MakeDrawItems(pStaticCollector, staticDraws);
    MakeDrawItems(pDynamicCollector, dynamicDraws);

    if (staticDraws.empty() && dynamicDraws.empty())
    {
        // Nothing to draw: the image keeps the previous frame's contents, exactly like the legacy,
        // which returns before its barrier and render pass (ShadowMap.cpp:551-554).
        return false;
    }

    if (!created || pCommandList == nullptr)
    {
        LogMessage(print, "Warning: RHI: the shadow map pass is not created, the shadow map is skipped");
        return false;
    }

    // A draw list can only be recorded against the buffers that hold its frame data. The legacy
    // binds the collector's own buffers unconditionally; the RHI pass needs the host's handles, so a
    // missing pair is a wiring error and the frame is skipped rather than drawn with wrong data.
    const bool staticReady = staticDraws.empty() ||
        (geometryBuffers.staticVertices != nullptr && geometryBuffers.staticIndices != nullptr);
    const bool dynamicReady = dynamicDraws.empty() ||
        (geometryBuffers.dynamicVertices != nullptr && geometryBuffers.dynamicIndices != nullptr);

    if (!staticReady || !dynamicReady)
    {
        if (!warnedMissingGeometryBuffers)
        {
            warnedMissingGeometryBuffers = true;
            LogMessage(print, std::string("Warning: RHI: the shadow map got no geometry buffers for ") +
                               (!staticReady && !dynamicReady ? "the static and the dynamic" :
                                !staticReady ? "the static" : "the dynamic") +
                               " draw list, the shadow map is skipped");
        }
        return false;
    }

    // The flags the NVRHI validation device checks on every vertex/index binding
    // (validation-commandlist.cpp:595-614) and that the Vulkan usage VUIDs of the buffers themselves
    // back (VUID-vkCmdBindVertexBuffers-pBuffers-00627 and VUID-vkCmdBindIndexBuffer-buffer-08784,
    // both checked by the layer this project runs); a handle without them is a wiring error the
    // module reports once (see the header).
    if (!warnedBufferFlags)
    {
        const auto lacksFlag = [](nvrhi::IBuffer *pBuffer, bool isVertex)
        {
            if (pBuffer == nullptr)
            {
                return false;
            }
            const nvrhi::BufferDesc &desc = pBuffer->getDesc();
            return isVertex ? !desc.isVertexBuffer : !desc.isIndexBuffer;
        };

        if (lacksFlag(geometryBuffers.staticVertices, true) ||
            lacksFlag(geometryBuffers.staticIndices, false) ||
            lacksFlag(geometryBuffers.dynamicVertices, true) ||
            lacksFlag(geometryBuffers.dynamicIndices, false))
        {
            warnedBufferFlags = true;
            LogMessage(print, "Warning: RHI: a shadow map geometry buffer lacks the "
                              "isVertexBuffer/isIndexBuffer flag (and the VkBuffer behind it the "
                              "vertex/index usage bits), so the bind breaks the usage VUIDs");
        }
    }

    // NVRHI's attachment loadOp is always LOAD (vulkan-graphics.cpp:80, :111), so the clear the
    // legacy render pass does on its depth attachment (ShadowMap.cpp:590-591) is an explicit
    // command. It runs once, before the first draw, and the backend places it through CopyDest
    // (vulkan-texture.cpp:647-688), which is also the transition into the attachment use.
    pCommandList->clearDepthStencilTexture(depthTexture, nvrhi::AllSubresources,
                                           true, DEPTH_CLEAR_VALUE, false, 0);

    // The legacy draw order: the static collector's list, then the dynamic collector's
    // (ShadowMap.cpp:622-629).
    RecordDrawList(pCommandList, staticDraws, geometryBuffers.staticVertices, geometryBuffers.staticIndices);
    RecordDrawList(pCommandList, dynamicDraws, geometryBuffers.dynamicVertices, geometryBuffers.dynamicIndices);

    // The legacy ends the pass with a barrier that leaves the image readable by the god-rays compute
    // (DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, ShadowMap.cpp:633-661). The
    // equivalent here is to require the sampled state on the image, which the god-rays binding set
    // then finds already satisfied; because the image has keepInitialState, the list close returns it
    // to DepthWrite for the next frame's attachment use. The mask is the exact state that pass's
    // compute-visibility SRV binding requires: its layout has ShaderType::Compute visibility
    // (RhiRtGodRaysPass.cpp:294-303), so the tracker asks for NonPixelShaderResource
    // (state-tracking.cpp:455-464) and adds DepthRead for a depth format
    // (vulkan-state-tracking.cpp:46-58).
    pCommandList->setTextureState(depthTexture, nvrhi::AllSubresources,
                                  nvrhi::ResourceStates::NonPixelShaderResource | nvrhi::ResourceStates::DepthRead);

    return true;
}

nvrhi::ITexture *RhiShadowMapPass::GetTexture() const
{
    return depthTexture.Get();
}

nvrhi::ISampler *RhiShadowMapPass::GetSampler() const
{
    return shadowSampler.Get();
}

void RhiShadowMapPass::RecordDrawList(nvrhi::ICommandList *pCommandList, const std::vector<DrawItem> &draws,
                                      nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer)
{
    if (draws.empty())
    {
        return;
    }

    // The legacy viewport and scissor are one constant rect for every draw: (0, 0, 4096, 4096, 0, 1)
    // (ShadowMap.cpp:604-618). NVRHI takes viewports in the D3D convention and the Vulkan backend
    // converts them with VKViewportWithDXCoords, which negates the height
    // (vulkan-graphics.cpp:528-531); passing the rectangle with minY = height and maxY = 0 therefore
    // emits the legacy's own positive-height VkViewport and not a vertically mirrored one. The
    // shadow map's Y orientation is part of its contract with the god-rays shader, whose GetShadow
    // flips the y coordinate itself (CmGodRays.comp.hlsl:106-110), so the legacy viewport is what
    // this pass must reproduce; RhiUiPass.cpp:217-226 uses the same inversion for the same reason.
    const float mapSize = float(SHADOW_MAP_SIZE);
    const nvrhi::Viewport viewport(0.0f, mapSize, mapSize, 0.0f, 0.0f, 1.0f);
    const nvrhi::Rect scissor(0, int(SHADOW_MAP_SIZE), 0, int(SHADOW_MAP_SIZE));

    for (const DrawItem &draw : draws)
    {
        // The push constant block of ShadowMap.cpp:620 and :669: the view-projection is constant for
        // the pass and the model matrix is the draw's own. The model matrix is the collector's
        // column-major 4x4 (Matrix::ToMat4Transposed of the CPU-side transform,
        // VertexCollector.cpp:931-933), so it is copied as-is.
        ShadowMapPushConstants push = {};
        memcpy(push.viewProjection, lastViewProjection, sizeof(push.viewProjection));
        memcpy(push.model, draw.model, sizeof(push.model));

        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewport(viewport);
        state.viewport.addScissorRect(scissor);
        // The draws bind the list's buffers at offset 0, the legacy's binding offsets
        // (ShadowMap.cpp:671-676): all of a list's GeometryDrawInfos address that list's own buffers,
        // and their baseVertex/firstIndex are absolute into them.
        state.addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(pVertexBuffer).setSlot(0).setOffset(0));
        state.setIndexBuffer(nvrhi::IndexBufferBinding()
                                 .setBuffer(pIndexBuffer)
                                 .setFormat(nvrhi::Format::R32_UINT)
                                 .setOffset(0));

        pCommandList->setGraphicsState(state);

        // After the state: changing the state invalidates push constants (nvrhi.h:3430-3432), and
        // the block is rebuilt per draw exactly as the legacy pushes it (ShadowMap.cpp:667-669).
        pCommandList->setPushConstants(&push, sizeof(push));

        nvrhi::DrawArguments args;
        if (draw.indexCount > 0)
        {
            // NVRHI carries the index count in 'vertexCount' for an indexed draw and maps the count,
            // the first index and the base vertex onto vkCmdDrawIndexed's indexCount, firstIndex and
            // vertexOffset (vulkan-graphics.cpp:689-700), which is the legacy call
            // (ShadowMap.cpp:677).
            args.vertexCount = draw.indexCount;
            args.startIndexLocation = draw.firstIndex;
            args.startVertexLocation = draw.baseVertex;
            pCommandList->drawIndexed(args);
        }
        else
        {
            // The legacy's non-indexed branch (ShadowMap.cpp:681), which draws the geometry's vertex
            // count (indexCount is zero there) starting at its base vertex.
            args.vertexCount = draw.indexCount;
            args.startVertexLocation = draw.baseVertex;
            pCommandList->draw(args);
        }
    }
}

bool RhiShadowMapPass::LoadShader(const char *pFileName, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, nvrhi::ShaderType::Vertex, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the shadow map shader \"" + path + "\"");
        return false;
    }

    return true;
}
