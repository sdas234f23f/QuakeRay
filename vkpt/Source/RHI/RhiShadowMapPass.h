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

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nvrhi/nvrhi.h>

namespace vkpt
{

class VertexCollector;

// The RHI port of the engine's ShadowMap (ShadowMap.h/.cpp, itself a port of Q2RTX's shadow_map.c):
// a depth-only raster pass that draws the frame's static and dynamic world geometry from the sun's
// point of view into a 4096x4096 VK_FORMAT_D32_SFLOAT image, which the god-rays compute pass samples
// to decide whether a march sample is lit. This module is the writer; the god-rays pass
// (RhiRtGodRaysPass, the sibling A5.2 stream) is the reader.
//
// What it reproduces, exactly as the legacy pass has it:
//  - The target: ONE 4096x4096 D32_SFLOAT image for the whole run, like the legacy's single
//    depthImage (ShadowMap.cpp:161-229) - not one per frame slot, and not one per swap permutation.
//    The image is written and read inside one frame's command list (the god-rays dispatch follows
//    the shadow map on the same list) and NVRHI re-requires every keepInitialState texture's
//    initial state at the close of each list (state-tracking.cpp:336-361), so the image's resting
//    state between submissions is the attachment state the next list starts from; see the state
//    contract below. A per-slot image would only add 64 MiB per slot and deviate from the legacy
//    without changing what the god rays see.
//  - The sampler: linear minification and magnification, nearest mip mode, clamp-to-border on all
//    three axes with the opaque-white border and VK_SAMPLER_REDUCTION_MODE_MIN (ShadowMap.cpp:
//    210-228). The god-rays shader's SampleLevel therefore reads a 2x2 MIN footprint and 1.0
//    outside the map - the "no occluder" value the march compares against.
//  - The pipeline: the `ShadowMap.vert.spv` blob as the only stage (the shader has no fragment
//    half, ShadowMap.cpp:313-409), one vertex binding at slot 0 with the RgVertex stride and the
//    position at offset 0, triangle list, front-face culling with VK_FRONT_FACE_CLOCKWISE
//    (ShadowMap.cpp:376-377), depth test LESS with depth writes on, depth clipping on, no colour
//    attachments.
//  - The 128-byte push constant: `shadowMapVP` at offset 0, written once per Render, and the
//    per-draw `model` matrix at offset 64 (ShadowMap.cpp:296-305, :620, :669).
//  - The draws: VertexCollector::GetGeometryDrawInfos() of the static collector first, then of the
//    dynamic collector (ShadowMap.cpp:548-549, :622-629), with the draw's model matrix, baseVertex,
//    firstIndex and indexCount and the legacy's vertex/index binding offsets and draw split
//    (ShadowMap.cpp:667-683).
//  - ComputeViewProjection (ShadowMap.cpp:36-139): the sun-direction view basis, the fitted world
//    AABB, the squared XY footprint, the [0,1] orthographic depth and the returned depth scale.
//    Reimplemented in the .cpp with the reference lines on the implementation.
//
// What it needs from the host:
//  - the RHI device and the engine shader folder path in Create(): the vertex blob is loaded from
//    `<pShaderFolderPath>ShadowMap.vert.spv`, the file ShaderManager maps "ShadowMap" to
//    (ShaderManager.cpp:57), the same blob the legacy pass loads;
//  - one Render() call per frame, on the frame context's open command list and before the god-rays
//    pass, with the same host inputs the legacy call receives (VulkanDevice.cpp:935-972): the
//    from-sun light direction, the world AABB, the two collectors - plus the four NVRHI geometry
//    buffers the draws bind (see GeometryBuffers).
//
// Geometry buffers - the one place where the RHI path has to differ from the legacy:
//  - The legacy binds the collector's own device-local VkBuffers, which every GeometryDrawInfo
//    carries (VertexCollector.cpp:907-908), and it never needs another source: both lists live in
//    the engine's buffers. The RHI module instead binds NVRHI handles the host passes in and ignores
//    the two VkBuffer fields of every draw info: all static draws bind
//    `staticVertices`/`staticIndices`, all dynamic draws bind `dynamicVertices`/`dynamicIndices`.
//    That is equivalent because GetGeometryDrawInfos fills every entry of one collector with that
//    collector's own two buffers, and it is necessary because the dynamic list is the one input that
//    must come from the RHI side: under `rhiframe` the engine's dynamic device-local buffers are
//    never written (only ASManager::SubmitDynamicGeometry - which the RHI path bypasses - fills
//    them), while RhiAccelStructs copies the same used prefix into its per-slot buffers every frame.
//  - The indices and offsets stay valid unchanged: `baseVertex` and `firstIndex` are computed
//    against the collector's buffers (VertexCollector.cpp:909-919) and the legacy's vertex and
//    index binding offsets are both 0, so a buffer that starts at offset 0 with the collector's
//    layout - which is exactly what the RHI layer's per-slot dynamic copies are, the used prefix
//    copied from offset 0 before the BLAS builds (RhiAccelStructs.cpp:955-985) - accepts them
//    as-is. The host is expected to pass
//    RhiAccelStructs::GetVertexDataBuffers(frameIndex).dynamicVertices/dynamicIndices as the two
//    dynamic handles.
//  - The handles have to be bindable as vertex/index buffers: BufferDesc::isVertexBuffer /
//    isIndexBuffer == true and, on the VkBuffer itself, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT /
//    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, or the engine's validation layer reports the usage VUID and
//    NVRHI's own validation device rejects the bind (validation-commandlist.cpp:595-614). Neither
//    the engine's collectors (VertexCollector.cpp:66-77) nor the RHI copies
//    (RhiAccelStructs.cpp:820-826, whose NVRHI desc never sets those flags either,
//    vulkan-buffer.cpp:48-55) are created with those usage bits, so the handles have to be produced
//    by the wiring the coordinator owns; the module warns once when a bound handle lacks the
//    NVRHI-side flag.
//  - The buffers' states are the NVRHI tracker's: the first graphics state of a list requires
//    VertexBuffer/IndexBuffer on the bound handles (vulkan-state-tracking.cpp:117-132) and the close
//    of the list re-requires each handle's own initialState (state-tracking.cpp:336-348). The host
//    should give the handles it passes here the same initialState as the other wraps of the same
//    VkBuffer (RhiAccelStructs' CopyDest for the dynamic pair), so that every wrap of one buffer
//    agrees about the state it rests in between submissions. The module requires nothing from the
//    states itself and restores nothing; it binds and draws.
//
// State contract:
//  - The depth image is created with initialState = DepthWrite and keepInitialState = true. The
//    framebuffer attachment use requires exactly that state and NVRHI re-requires it when the list
//    closes, so every list starts from the attachment state and needs no barrier for it. The
//    frame's clear is an explicit command (NVRHI's attachment loadOp is always LOAD,
//    vulkan-graphics.cpp:80, :111), which also makes the previous frame's contents irrelevant.
//  - A successful Render ends by requiring the sampled state on the image
//    (NonPixelShaderResource | DepthRead, the exact state the god-rays pass's compute-visibility
//    depth-SRV binding requires, RhiRtGodRaysPass.cpp:294-303), the analogue of the legacy's
//    end-of-pass barrier to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (ShadowMap.cpp:633-661). A
//    consumer binding the image later on the same list finds the state already correct, and the list
//    close returns the image to DepthWrite for the next frame.
//  - The module owns the image, the sampler, the pipeline layout, the pipeline and the framebuffer
//    for the whole run: nothing is re-created per frame and nothing has to be retired, so the pass
//    needs no frame context. The destructor waits for the device to idle and then drops them.
//
// Consumption by the god rays (the shader is Shadow common's `CmGodRays.comp` twin):
//  - GetTexture()/GetSampler() are the two items the god-rays shadow set needs: the shader declares
//    `texShadowMap` at binding 0 and `texShadowMap_Sampler` at binding 1 of
//    DESC_SET_GOD_RAYS_SHADOW (CmGodRays.comp.hlsl:84-85), so the set's layout is
//    `Texture_SRV(0)` + `Sampler(1)` and both handles go in as a Texture_SRV/Sampler pair.
//  - Render's out-parameters - also kept in GetViewProjection()/GetDepthScale() - are the
//    `shadowMapVP` and `shadowMapDepthScale` fields of the god-rays params struct. Their exact
//    conventions are documented on Render.
class RhiShadowMapPass final
{
public:
    using PrintFunction = std::function<void(const char *)>;

    // The legacy SHADOW_MAP_SIZE (ShadowMap.h:78; Q2RTX's SHADOWMAP_SIZE). Public because the
    // god-rays side may want the texel size for diagnostics.
    static constexpr uint32_t SHADOW_MAP_SIZE = 4096;

    RhiShadowMapPass();
    ~RhiShadowMapPass();

    RhiShadowMapPass(const RhiShadowMapPass &other) = delete;
    RhiShadowMapPass(RhiShadowMapPass &&other) noexcept = delete;
    RhiShadowMapPass &operator=(const RhiShadowMapPass &other) = delete;
    RhiShadowMapPass &operator=(RhiShadowMapPass &&other) noexcept = delete;

    // 'pDevice' is the RHI device; 'pShaderFolderPath' is the folder ShaderManager loads the engine
    // blobs from, with the trailing separator (the host passes `info->pShaderFolderPath`); 'pfnPrint'
    // is the host's log callback. Creates the image, the sampler, the framebuffer, the input layout,
    // the push-constant layout and the depth-only pipeline. Returns false and leaves the pass
    // unusable if the device is null, a shader cannot be loaded or a resource cannot be created; the
    // host logs that through 'pfnPrint' (Render then does nothing, and the god rays must be skipped).
    bool Create(nvrhi::IDevice *pDevice,
                const char *pShaderFolderPath,
                PrintFunction pfnPrint);

    bool IsCreated() const { return created; }

    // The NVRHI buffers both draw lists bind, in the collector's own layout: RgVertex records of 80
    // bytes with the position at offset 0 and R32_UINT indices, both starting at offset 0. Not owned:
    // the host keeps them alive for the frame (and, for the static pair, for the run).
    struct GeometryBuffers
    {
        // The static collector's vertex and index data: the geometry the engine's level load put into
        // the collector's device-local buffers, which are filled under `rhiframe` as well. The host
        // provides handles over those buffers; the class comment above has the usage-bit caveat.
        nvrhi::IBuffer *staticVertices = nullptr;
        nvrhi::IBuffer *staticIndices = nullptr;

        // This frame slot's dynamic vertex and index data: the RHI layer's per-slot copies, i.e.
        // RhiAccelStructs::GetVertexDataBuffers(frameIndex).dynamicVertices/dynamicIndices, which
        // that module fills from the collector's staging buffers in BuildTopLevel
        // (RhiAccelStructs.cpp:959-985). Render must be recorded after that copy and before the next
        // frame overwrites it; the RHI list's own ordering gives both.
        nvrhi::IBuffer *dynamicVertices = nullptr;
        nvrhi::IBuffer *dynamicIndices = nullptr;
    };

    // Records the frame's shadow map: computes the sun view-projection, gathers both collectors'
    // GeometryDrawInfos, clears the depth image to 1.0 and draws the static list and then the
    // dynamic list. Returns false - and leaves the image untouched, exactly like the legacy
    // (ShadowMap.cpp:546-554) - when there is nothing to draw or when the buffers of a non-empty
    // list are missing; the caller then skips the god-rays dispatch, as the legacy host does.
    //
    // Every argument's host-side source (the legacy call is VulkanDevice.cpp:970-972):
    //  - 'pCommandList': the slot's open list (RhiFrameContext::GetCommandList(frameIndex)).
    //  - 'sunDirection': the from-sun light direction, the `shadowLightDir` array
    //    (VulkanDevice.cpp:945-951): `sunDir` from
    //    LightManager::GetLastDirectionalLight (VulkanDevice.cpp:935, LightManager.cpp:562-580),
    //    replaced by the negated `godRaysSkyDirection` when the god rays take their sun from the sky
    //    texture. The math uses it as-is (no negation), so the shadow camera looks FROM the sun
    //    down at the scene - see ComputeViewProjection in the .cpp.
    //  - 'aabbMin'/'aabbMax': the world bounds, `scene->GetAABB(aabbMin, aabbMax)`
    //    (VulkanDevice.cpp:953-956); the legacy only calls Render when `scene->HasAABB()`.
    //  - 'pStaticCollector'/'pDynamicCollector': `scene->GetASManager()->GetStaticCollector()` and
    //    `GetDynamicCollector(frameIndex)` (VulkanDevice.cpp:961-962). Null is accepted and means an
    //    empty list (the legacy guards the same way).
    //  - 'geometryBuffers': see above.
    //  - 'outViewProjection': 16 floats, column-major, the product projection * view of the fitted
    //    shadow camera - the same bytes the legacy computes with
    //    `Matrix::Multiply(out, viewMatrix, projectionMatrix)` (ShadowMap.cpp:136), which is
    //    `Proj * View` in column-major storage. It is the `shadowMapVP` the god-rays params struct
    //    takes (VulkanDevice.cpp:1000): the shader projects with
    //    `mul(shadowMapVP, float4(worldPos, 1))` (CmGodRays.comp.hlsl:106), so the 16 floats are
    //    copied in as-is. Always written by Render, even when it returns false - the legacy computes
    //    them before it checks the geometry.
    //  - 'outDepthScale': the view-space Z extent of the fitted AABB (viewMax[2] - viewMin[2] along
    //    the sun direction); the legacy's `shadowMapDepthScale` (ShadowMap.cpp:138). The god-rays
    //    shader declares the field but does not read it, so it is informational.
    bool Render(nvrhi::ICommandList *pCommandList,
                const float sunDirection[3],
                const float aabbMin[3], const float aabbMax[3],
                const VertexCollector *pStaticCollector,
                const VertexCollector *pDynamicCollector,
                const GeometryBuffers &geometryBuffers,
                float outViewProjection[16], float *outDepthScale);

    // The shadow map image as the wrapped RHI texture, or null before Create. Borrowed: the pass
    // owns it and it lives until the pass is destroyed. The image is shared by every frame slot, so a
    // consumer that binds it must re-read this pointer only if the pass was re-created. Born with
    // isShaderResource and isRenderTarget, format D32 (VK_FORMAT_D32_SFLOAT).
    nvrhi::ITexture *GetTexture() const;

    // The MIN-reduction clamp-to-border sampler the image must be sampled with, or null before
    // Create. Borrowed. Its shape is the legacy's (ShadowMap.cpp:210-223): linear/linear/nearest,
    // clamp-to-border, opaque-white border, reduction MIN.
    nvrhi::ISampler *GetSampler() const;

    // The values the last Render wrote: the column-major `shadowMapVP` and the depth scale. Valid
    // from the first Render on (a Render that draws nothing still writes them, like the legacy), and
    // they describe the image that is currently in the shadow map.
    const float *GetViewProjection() const { return lastViewProjection; }
    float GetDepthScale() const { return lastDepthScale; }

private:
    // One draw of one list, with only the fields the shadow map consumes: the model matrix the
    // collector built from its CPU-side transform array (VertexCollector.cpp:925-933) and the three
    // index values. The collector's VkBuffer fields are deliberately not copied - the NVRHI handles
    // of GeometryBuffers replace them (see the class comment).
    struct DrawItem
    {
        float model[16];
        uint32_t baseVertex;
        uint32_t firstIndex;
        uint32_t indexCount;
    };

    bool LoadShader(const char *pFileName, nvrhi::ShaderHandle &result);

    // Converts one collector's GeometryDrawInfos into the module's DrawItems (the VkBuffer fields are
    // deliberately dropped, see the class comment). A null collector gives an empty list.
    static void MakeDrawItems(const VertexCollector *pCollector, std::vector<DrawItem> &outItems);

    // Records one draw list against one buffer pair: the per-draw push constant with the model
    // matrix, the graphics state and the indexed/non-indexed draw, in the legacy's order. An empty
    // list records nothing.
    void RecordDrawList(nvrhi::ICommandList *pCommandList, const std::vector<DrawItem> &draws,
                        nvrhi::IBuffer *pVertexBuffer, nvrhi::IBuffer *pIndexBuffer);

    nvrhi::IDevice *device = nullptr;
    PrintFunction print;
    std::string shaderFolderPath;

    nvrhi::ShaderHandle vertexShader;
    nvrhi::InputLayoutHandle inputLayout;

    // The pipeline's only layout: one 128-byte push-constant item and no descriptors, so no binding
    // set has to be created or bound for the pass.
    nvrhi::BindingLayoutHandle pushConstantLayout;

    nvrhi::TextureHandle depthTexture;
    nvrhi::SamplerHandle shadowSampler;
    nvrhi::FramebufferHandle framebuffer;
    nvrhi::GraphicsPipelineHandle pipeline;

    // The view-projection and depth scale of the last Render, exposed through GetViewProjection() /
    // GetDepthScale(). The single image means a single slot of these values is coherent.
    float lastViewProjection[16] = {};
    float lastDepthScale = 0.0f;

    bool warnedMissingGeometryBuffers = false;
    bool warnedBufferFlags = false;

    bool created = false;
};

}
