// Copyright (c) 2020-2021 Sultim Tsyrendashiev
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

#include "VulkanDevice.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "HaltonSequence.h"
#include "RenderResolutionHelper.h"
#include "RgException.h"
#include "Const.h"
#include "Generated/ShaderCommonC.h"
#include "LibraryConfig.h"
#include "RHI/NvrhiContext.h"
#include "RHI/NvrhiFrameSkeleton.h"
#include "RHI/NvrhiRequirements.h"
#include "RHI/RhiAccelStructs.h"
#include "RHI/RhiDebugTracePass.h"
#include "RHI/RhiRtComposePass.h"
#include "RHI/RhiRtDirectPass.h"
#include "RHI/RhiRtIndirectPass.h"
#include "RHI/RhiRtPrimaryPass.h"
#include "RHI/RhiFrameContext.h"
#include "RHI/RhiSkyPass.h"
#include "RHI/RhiTextureSource.h"
#include "RHI/RhiTextureTable.h"
#include "RHI/RhiUiPass.h"

using namespace vkpt;

VulkanDevice::VulkanDevice( const RgInstanceCreateInfo* info )
    : instance( VK_NULL_HANDLE )
    , device( VK_NULL_HANDLE )
    , surface( VK_NULL_HANDLE )
    , currentFrameState()
    , frameId( 1 )
    , waitForOutOfFrameFence( false )
    , libconfig( LibraryConfig::Read( info->pConfigPath ) )
    , debugMessenger( VK_NULL_HANDLE )
    , userPrint{ std::make_unique< UserPrint >( info->pfnPrint, info->pUserPrintData ) }
    , userFileLoad{ std::make_shared< UserFileLoad >(
          info->pfnOpenFile, info->pfnCloseFile, info->pUserLoadFileData ) }
    , rayCullBackFacingTriangles( info->rayCullBackFacingTriangles )
    , allowGeometryWithSkyFlag( info->allowGeometryWithSkyFlag )
    , lensFlareVerticesInScreenSpace( info->lensFlareVerticesInScreenSpace )
    , rasterizedVertexColorGamma( info->rasterizedVertexColorGamma != 0 )
    , previousFrameTime( -1.0 / 60.0 )
    , currentFrameTime( 0 )
{
    ValidateCreateInfo( info );



    // init vulkan instance
    CreateInstance( *info );


    // create VkSurfaceKHR using user's function
    surface = GetSurfaceFromUser( instance, *info );


    // create selected physical device
    physDevice = std::make_shared< PhysicalDevice >( instance );
    queues     = std::make_shared< Queues >( physDevice->Get(), surface );

    // create vulkan device and set extension function pointers
    CreateDevice();

    CreateSyncPrimitives();

    // set device
    queues->SetDevice( device );

    // the RHI device wraps the device and the queues above, so it is created
    // once both exist
    CreateNvrhiDevice();

    // The RHI texture table (RHI/RhiTextureTable.h) has to exist before the sampler managers: the
    // world manager mirrors its sampler descs into it while it creates the samplers. The capacity is
    // the same clamped value TextureManager builds its own table with - a bindless table's capacity
    // is fixed at layout creation, and the two tables address the same slots. On failure the pointer
    // stays null and the game continues without the RHI textures. The table is built only when the
    // RHI frame skeleton is on: with `rhiframe` off the engine writes to the legacy table alone, so
    // the legacy path pays neither the wrapping of every texture nor the descriptor writes. The world
    // stage of A2 revisits this when the RHI renderer owns the frame.
    if (libconfig.rhiFrameSkeleton)
    {
        const uint32_t maxTextureCount =
            std::clamp(info->maxTextureCount, TEXTURE_COUNT_MIN, TEXTURE_COUNT_MAX);

        // The frame context comes first: the table retires its dropped samplers and wrapped textures
        // through it, and the skeleton records through it, one command list per engine frame slot.
        rhiFrameContext = std::make_shared<rhi::RhiFrameContext>();

        if (!rhiFrameContext->Create(nvrhi->GetDevice(), MAX_FRAMES_IN_FLIGHT))
        {
            rhiFrameContext.reset();
            Print("Warning: RHI: failed to create the frame context, the RHI renderer is unavailable");
        }
        else
        {
            rhiTextureTable = std::make_shared<rhi::RhiTextureTable>();

            if (!rhiTextureTable->Create(nvrhi->GetDevice(), maxTextureCount, rhiFrameContext.get()))
            {
                rhiTextureTable.reset();
                Print("Warning: RHI: failed to create the texture table, the RHI renderer will not see engine textures");
            }
        }
    }


    memAllocator        = std::make_shared<MemoryAllocator>(instance, device, physDevice);

    cmdManager          = std::make_shared<CommandBufferManager>(device, queues);

    uniform             = std::make_shared<GlobalUniform>(device, memAllocator);

    swapchain           = std::make_shared<Swapchain>(device, surface, physDevice->Get(), cmdManager);

    // for world samplers with modifyable lod biad
    worldSamplerManager     = std::make_shared<SamplerManager>(device, 8, info->textureSamplerForceMinificationFilterLinear,
                                                               rhiTextureTable.get());
    genericSamplerManager   = std::make_shared<SamplerManager>(device, 0, info->textureSamplerForceMinificationFilterLinear);

    framebuffers        = std::make_shared<Framebuffers>(
        device,
        memAllocator, 
        cmdManager, 
        *info );

    blueNoise           = std::make_shared<BlueNoise>(
        device,
        info->pBlueNoiseFilePath,
        memAllocator,
        cmdManager, 
        userFileLoad);

    textureManager      = std::make_shared<TextureManager>(
        device, 
        memAllocator,
        worldSamplerManager,
        cmdManager,
        userFileLoad,
        *info,
        libconfig);

    // The table was created before the texture manager; attaching it afterwards fills every live
    // slot of the new table through the manager's all-dirty path (TextureManager::SetRhiTextureTable).
    textureManager->SetRhiTextureTable(rhiTextureTable.get());

    cubemapManager      = std::make_shared<CubemapManager>(
        device,
        memAllocator,
        genericSamplerManager,
        cmdManager,
        userFileLoad,
        *info,
        libconfig);

    worldLights         = std::make_shared<WorldLights>();
    clusterLightLists   = std::make_shared<ClusterLightLists>();

    shaderManager       = std::make_shared<ShaderManager>(
        device,
        info->pShaderFolderPath,
        userFileLoad);

    scene               = std::make_shared<Scene>(
        device,
        physDevice,
        memAllocator,
        cmdManager,
        textureManager,
        uniform,
        shaderManager);
   
    tonemapping         = std::make_shared<Tonemapping>(
        device,
        framebuffers,
        shaderManager,
        uniform,
        memAllocator);

    volumetric          = std::make_shared< Volumetric >( 
        device,
        cmdManager.get(),
        memAllocator.get(),
        shaderManager.get(),
        uniform.get(),
        blueNoise.get() );

    rasterizer          = std::make_shared<Rasterizer>(
        device,
        physDevice->Get(),
        shaderManager,
        textureManager,
        uniform,
        genericSamplerManager,
        tonemapping,
        volumetric,
        memAllocator,
        framebuffers,
        cmdManager,
        *info);

    decalManager        = std::make_shared<DecalManager>(
        device,
        memAllocator,
        shaderManager,
        uniform,
        framebuffers,
        textureManager);

    portalList          = std::make_shared<PortalList>(
        device,
        memAllocator);

    shadowMap           = std::make_shared<ShadowMap>(
        device,
        memAllocator,
        shaderManager);

    godRays             = std::make_shared<GodRays>(
        device,
        memAllocator,
        framebuffers,
        shaderManager,
        uniform,
        blueNoise,
        shadowMap);

    rayStats            = std::make_shared<RayStats>(device, memAllocator);
    passTimings         = std::make_shared<PassTimings>(device, physDevice->Get(), queues->GetIndexGraphics());

    rtPipeline = std::make_shared< RayTracingPipeline >( 
        device,
        physDevice,
        memAllocator,
        shaderManager.get(),
        scene.get(),
        uniform.get(),
        textureManager.get(),
        framebuffers.get(),
        blueNoise.get(),
        cubemapManager.get(),
        rasterizer->GetRenderCubemap().get(),
        portalList.get(),
        volumetric.get(),
        rayStats.get(),
        *info );

    pathTracer          = std::make_shared<PathTracer>(device, rtPipeline);

    imageComposition    = std::make_shared<ImageComposition>(
        device,
        memAllocator,
        framebuffers, 
        shaderManager, 
        uniform, 
        tonemapping, 
        volumetric.get() );

    bloom               = std::make_shared<Bloom>(
        device,
        framebuffers,
        shaderManager,
        uniform,
        tonemapping);

    amdFsr              = std::make_shared<FidelityFX::FSR>(
        device,
        physDevice->Get(),
        userPrint.get());

    nvDlss              = std::make_shared<DLSS>(
        instance, 
        device, 
        physDevice->Get(),
        info->pAppGUID,
        libconfig.dlssValidation);

    sharpening          = std::make_shared<Sharpening>(
        device,
        framebuffers,
        shaderManager);

    q2Denoiser          = std::make_shared<Q2Denoiser>(
        device,
        framebuffers,
        shaderManager,
        uniform,
        scene->GetASManager());

    effectWipe          = std::make_shared<EffectWipe>(
        device,
        framebuffers,
        uniform,
        blueNoise,
        shaderManager, 
        info->effectWipeIsUsed );


#define CONSTRUCT_SIMPLE_EFFECT(T) std::make_shared<T>(device, framebuffers, uniform, shaderManager)
    effectRadialBlur            = CONSTRUCT_SIMPLE_EFFECT(EffectRadialBlur);
    effectChromaticAberration   = CONSTRUCT_SIMPLE_EFFECT(EffectChromaticAberration);
    effectInverseBW             = CONSTRUCT_SIMPLE_EFFECT(EffectInverseBW);
    effectHueShift              = CONSTRUCT_SIMPLE_EFFECT(EffectHueShift);
    effectDistortedSides        = CONSTRUCT_SIMPLE_EFFECT(EffectDistortedSides);
    effectWaves                 = CONSTRUCT_SIMPLE_EFFECT(EffectWaves);
    effectColorTint             = CONSTRUCT_SIMPLE_EFFECT(EffectColorTint);
    effectCrtDemodulateEncode   = CONSTRUCT_SIMPLE_EFFECT(EffectCrtDemodulateEncode);
    effectCrtDecode             = CONSTRUCT_SIMPLE_EFFECT(EffectCrtDecode);
#undef SIMPLE_EFFECT_CONSTRUCTOR_PARAMS


    shaderManager->Subscribe(q2Denoiser);
    shaderManager->Subscribe(imageComposition);
    shaderManager->Subscribe(rasterizer);
    shaderManager->Subscribe(volumetric);
    shaderManager->Subscribe(decalManager);
    shaderManager->Subscribe(rtPipeline);
    shaderManager->Subscribe(tonemapping);
    shaderManager->Subscribe(scene->GetVertexPreprocessing());
    shaderManager->Subscribe(bloom);
    shaderManager->Subscribe(sharpening);
    shaderManager->Subscribe(shadowMap);
    shaderManager->Subscribe(godRays);
    shaderManager->Subscribe(effectWipe);
    shaderManager->Subscribe(effectRadialBlur);
    shaderManager->Subscribe(effectChromaticAberration);
    shaderManager->Subscribe(effectInverseBW);
    shaderManager->Subscribe(effectHueShift);
    shaderManager->Subscribe(effectDistortedSides);
    shaderManager->Subscribe(effectWaves);
    shaderManager->Subscribe(effectColorTint);
    shaderManager->Subscribe(effectCrtDemodulateEncode);
    shaderManager->Subscribe(effectCrtDecode);

    framebuffers->Subscribe(rasterizer);
    framebuffers->Subscribe(decalManager);
    framebuffers->Subscribe(amdFsr);

    // The RHI frame skeleton draws the frame through the RHI layer instead of
    // the renderer above. Off by default: enabled by 'rhiframe' in vkpt.txt.
    if (libconfig.rhiFrameSkeleton)
    {
        if (nvrhi != nullptr)
        {
            // The acceleration structures of the RHI path (RHI/RhiAccelStructs.h): the NVRHI copy of
            // the engine's static BLAS plus one TLAS per frame slot, built from the engine's
            // ASManager. Created after the scene exists (the manager is their geometry registry) and
            // next to the skeleton, which records their builds and reads the TLAS. A failure leaves
            // the pointer null; the skeleton refuses to be available without it, so the legacy
            // renderer keeps the frame.
            rhiAccelStructs = std::make_shared<rhi::RhiAccelStructs>();
            if (!rhiAccelStructs->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                         scene->GetASManager().get(),
                                         [this](const char *pMessage) { Print(pMessage); }))
            {
                rhiAccelStructs.reset();
                Print("Warning: RHI: the acceleration structures are unavailable, the legacy renderer is kept");
            }

            // The debug ray-tracing pass (RHI/RhiDebugTracePass.h), created only when 'rhitrace' is
            // on. Its set 1 is the engine's global uniform, wrapped here as a static constant buffer
            // - the same shape the skeleton's world pass wraps it with, because a volatile wrap
            // would become a dynamic-offset binding the pass's static layout item cannot take. The
            // binding set the pass builds over the wrap keeps it alive; the pass retires its per-slot
            // ALBEDO wraps through the frame context. A failure leaves the pointer null: with the
            // flag on the skeleton then refuses to be available and the legacy renderer keeps the
            // frame.
            if (libconfig.rhiDebugTrace)
            {
                nvrhi::BufferDesc debugUniformDesc;
                debugUniformDesc.byteSize = sizeof(ShGlobalUniform);
                debugUniformDesc.isConstantBuffer = true;
                debugUniformDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
                debugUniformDesc.keepInitialState = true;
                debugUniformDesc.debugName = "RHI debug trace uniform (GlobalUniform wrap)";

                nvrhi::BufferHandle debugTraceUniformBuffer = nvrhi->GetDevice()->createHandleForNativeBuffer(
                    nvrhi::ObjectTypes::VK_Buffer,
                    nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(uniform->GetBuffer()))),
                    debugUniformDesc);

                rhiDebugTracePass = std::make_shared<RhiDebugTracePass>();
                if (!rhiDebugTracePass->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                               debugTraceUniformBuffer, info->pShaderFolderPath,
                                               [this](const char *pMessage) { Print(pMessage); }))
                {
                    rhiDebugTracePass.reset();
                    Print("Warning: RHI: the debug trace pass is unavailable, the legacy renderer is kept");
                }
            }

            // The real ray-tracing pass of A4.1 (RHI/RhiRtPrimaryPass.h), created only when 'rhirt'
            // is on: the engine's primary-visibility raygen with the engine's two misses and its two
            // hit groups, dispatched over the same RHI acceleration structures. Its set 4 is the
            // shared texture table, so the host hands the table over; the engine's framebuffer
            // images are resolved per frame by the pass itself, so no wrap is created here. A
            // failure leaves the pointer null: with the flag on the skeleton then refuses to be
            // available and the legacy renderer keeps the frame.
            if (libconfig.rhiRayTracing)
            {
                rhiRtPrimaryPass = std::make_shared<RhiRtPrimaryPass>();
                if (!rhiRtPrimaryPass->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                              rhiTextureTable.get(), info->pShaderFolderPath,
                                              [this](const char *pMessage) { Print(pMessage); }))
                {
                    rhiRtPrimaryPass.reset();
                    Print("Warning: RHI: the primary ray-tracing pass is unavailable, the legacy renderer is kept");
                }

                // The direct-lighting pass of A4.2 (RHI/RhiRtDirectPass.h), created only with the
                // primary: it borrows the primary's shared layout handles, so it must be destroyed
                // before it, and the light-source buffers it wraps come from the scene's light
                // manager. A failure leaves the pointer null: with the flag on the skeleton then
                // refuses to be available and the legacy renderer keeps the frame.
                if (rhiRtPrimaryPass != nullptr)
                {
                    rhiRtDirectPass = std::make_shared<RhiRtDirectPass>();
                    if (!rhiRtDirectPass->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                                 rhiTextureTable.get(), scene->GetLightManager().get(),
                                                 rhiRtPrimaryPass.get(), info->pShaderFolderPath,
                                                 [this](const char *pMessage) { Print(pMessage); }))
                    {
                        rhiRtDirectPass.reset();
                        Print("Warning: RHI: the direct ray-tracing pass is unavailable, the legacy renderer is kept");
                    }
                }

                // The indirect / GI pass of A4.3 (RHI/RhiRtIndirectPass.h): the bounce-light term of
                // the traced chain, created only with the primary and the direct pass - it borrows
                // their layout handles and the light set, so it is destroyed before both. Its set 5
                // is the engine's blue-noise array, wrapped once here (the image is static) and
                // announced by the pass on the list that first binds it. A failure leaves the
                // pointer null: with the flag on the skeleton then refuses to be available and the
                // legacy renderer keeps the frame.
                if (rhiRtDirectPass != nullptr)
                {
                    const nvrhi::TextureHandle blueNoiseTexture = rhi::wrapEngineTextureArray(
                        nvrhi->GetDevice(),
                        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blueNoise->GetImage())),
                        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blueNoise->GetImageView())),
                        static_cast<uint32_t>(blueNoise->GetFormat()),
                        blueNoise->GetExtent().width, blueNoise->GetExtent().height,
                        blueNoise->GetLayerCount(), 1,
                        "RHI blue noise (indirect set 5)");

                    if (blueNoiseTexture == nullptr)
                    {
                        Print("Warning: RHI: failed to wrap the blue-noise texture, the indirect ray-tracing pass is unavailable");
                    }
                    else
                    {
                        rhiRtIndirectPass = std::make_shared<RhiRtIndirectPass>();
                        if (!rhiRtIndirectPass->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                                       rhiTextureTable.get(), rhiRtPrimaryPass.get(),
                                                       rhiRtDirectPass.get(), info->pShaderFolderPath,
                                                       [this](const char *pMessage) { Print(pMessage); }))
                        {
                            rhiRtIndirectPass.reset();
                            Print("Warning: RHI: the indirect ray-tracing pass is unavailable, the legacy renderer is kept");
                        }
                        else
                        {
                            rhiRtIndirectPass->SetBlueNoiseTexture(blueNoiseTexture);
                        }
                    }
                }

                // The compose pass of A4.4 (RHI/RhiRtComposePass.h), created only under
                // 'rhicompose': the real adapter -> interleave -> exposure histogram/average ->
                // checkerboard -> prepare-final chain ending in the display-referred FINAL, which
                // the skeleton then presents raw. It wraps the per-slot tonemapping buffers, so the
                // engine object has to outlive it. A failure leaves the pointer null: the traced
                // chain keeps the A4.2a diagnostic present instead of failing the frame.
                if (libconfig.rhiCompose)
                {
                    rhiRtComposePass = std::make_shared<RhiRtComposePass>();
                    if (!rhiRtComposePass->Create(nvrhi->GetDevice(), rhiFrameContext.get(),
                                                  tonemapping.get(),
                                                  info->pShaderFolderPath,
                                                  [this](const char *pMessage) { Print(pMessage); }))
                    {
                        rhiRtComposePass.reset();
                        Print("Warning: RHI: the compose pass is unavailable, the diagnostic present is kept");
                    }
                }

                // The 2D UI pass of A5.1 (RHI/RhiUiPass.h): the game's SWAPCHAIN overlay drawn into
                // the compose's upscaled image after the TAAU. It needs the shared texture table and
                // the frame context only; the per-slot staging geometry wraps are the skeleton's (it
                // is handed the collector's handles every frame). A failure leaves the pointer null:
                // the frame is still drawn, just without the UI.
                rhiUiPass = std::make_shared<RhiUiPass>();
                if (!rhiUiPass->Create(nvrhi->GetDevice(), rhiTextureTable.get(),
                                       rhiFrameContext.get(), info->pShaderFolderPath,
                                       [this](const char *pMessage) { Print(pMessage); }))
                {
                    rhiUiPass.reset();
                    Print("Warning: RHI: the 2D UI pass is unavailable, the frame is drawn without the UI");
                }
            }

            // The pass binds the shared RHI texture table (its slot 0 holds the engine's empty
            // texture: 1x1, VK_FORMAT_R8G8B8A8_UNORM, created and left in the read-only layout by
            // TextureManager) and records through the shared frame context, so the host hands both
            // over here.
            //
            // The mode the skeleton records for the whole run: 'rhirt' selects the real
            // ray-tracing pass of A4.1, 'rhitrace' the A3.1 debug trace, and neither the rasterized
            // chain. 'rhirt' wins when both are set (LibraryConfig.h documents it), and the
            // skeleton itself refuses to be available when its mode's pass failed to be created.
            if (libconfig.rhiRayTracing && libconfig.rhiDebugTrace)
            {
                Print("Warning: RHI: both 'rhirt' and 'rhitrace' are set, the primary ray-tracing pass is used");
            }

            NvrhiFrameSkeleton::FrameMode frameMode = NvrhiFrameSkeleton::FrameMode::Rasterized;
            if (libconfig.rhiRayTracing)
            {
                frameMode = NvrhiFrameSkeleton::FrameMode::Traced;
            }
            else if (libconfig.rhiDebugTrace)
            {
                frameMode = NvrhiFrameSkeleton::FrameMode::DebugTrace;
            }

            nvrhiFrameSkeleton = std::make_shared<NvrhiFrameSkeleton>(
                nvrhi->GetDevice(),
                swapchain.get(),
                info->pShaderFolderPath,
                rhiTextureTable.get(),
                rhiFrameContext.get(),
                rhiAccelStructs.get(),
                rhiDebugTracePass.get(),
                rhiRtPrimaryPass.get(),
                rhiRtDirectPass.get(),
                rhiRtIndirectPass.get(),
                rhiRtComposePass.get(),
                rhiUiPass.get(),
                frameMode,
                [this](const char *pMessage) { Print(pMessage); });

            swapchain->Subscribe(nvrhiFrameSkeleton);

            // The sky pass binds the geometry the legacy collector recorded, so its device-local
            // VkBuffers are wrapped once, here, with NVRHI handles. The legacy command buffer that
            // fills them (Rasterizer::SubmitForFrame -> CopyFromStaging, VulkanDevice.cpp:1261) is
            // submitted after the RHI list of the same frame, so the sky of a frame reads the copy
            // of the previous frame. The geometry is static per level - the collector's copy only
            // moves what the level uploaded (RasterizedDataCollector::CopyFromStaging) - so only
            // the very first frame after a level load reads the zeroed buffer; from the second
            // frame on the sky draws the same data the legacy renderer would draw. The wraps are
            // deliberately long-lived: the engine's AutoBuffer keeps the same VkBuffer for the
            // whole run.
            if (RhiSkyPass *skyPass = nvrhiFrameSkeleton->GetSkyPass())
            {
                const RasterizedDataCollector &collector = rasterizer->GetDataCollector();

                // The byte counts are the sizes the collector's AutoBuffers are created with
                // (RasterizedDataCollector.cpp:69-73); on a native wrap NVRHI only keeps the desc
                // for bookkeeping (vulkan-buffer.cpp:202-213).
                nvrhi::BufferDesc vertexBufferDesc;
                vertexBufferDesc.byteSize =
                    static_cast<uint64_t>(std::max(info->rasterizedMaxVertexCount, 64u)) * sizeof(RgVertex);
                vertexBufferDesc.isVertexBuffer = true;
                // The engine's buffer is written by its own command buffer, so NVRHI cannot have seen
                // it: the first list that binds it would report an unknown prior state
                // (state-tracking.cpp:290-297). Declaring the state the engine's AutoBuffer leaves
                // the data in - readable for vertex fetch (AutoBuffer.cpp:115-125) - makes the first
                // use transition-free and keeps the claim for every later list.
                vertexBufferDesc.initialState = nvrhi::ResourceStates::VertexBuffer;
                vertexBufferDesc.keepInitialState = true;
                vertexBufferDesc.debugName = "Rasterizer vertex buffer (RHI)";

                nvrhi::BufferDesc indexBufferDesc;
                indexBufferDesc.byteSize =
                    static_cast<uint64_t>(std::max(info->rasterizedMaxIndexCount, 64u)) * sizeof(uint32_t);
                indexBufferDesc.isIndexBuffer = true;
                indexBufferDesc.initialState = nvrhi::ResourceStates::IndexBuffer;
                indexBufferDesc.keepInitialState = true;
                indexBufferDesc.debugName = "Rasterizer index buffer (RHI)";

                nvrhi::BufferHandle vertexBuffer = nvrhi->GetDevice()->createHandleForNativeBuffer(
                    nvrhi::ObjectTypes::VK_Buffer,
                    nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(collector.GetVertexBuffer()))),
                    vertexBufferDesc);
                nvrhi::BufferHandle indexBuffer = nvrhi->GetDevice()->createHandleForNativeBuffer(
                    nvrhi::ObjectTypes::VK_Buffer,
                    nvrhi::Object(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(collector.GetIndexBuffer()))),
                    indexBufferDesc);

                if (vertexBuffer == nullptr || indexBuffer == nullptr)
                {
                    Print("Warning: RHI: failed to wrap the rasterized geometry buffers, the sky is skipped");
                }
                else
                {
                    skyPass->SetGeometryBuffers(vertexBuffer, indexBuffer);
                }
            }
        }
        else
        {
            Print("Warning: RHI: 'rhiframe' is ignored, there is no RHI device");
        }
    }
}

VulkanDevice::~VulkanDevice()
{
    vkDeviceWaitIdle(device);

    // the skeleton wraps the swapchain images with the RHI device, so it has to
    // be released before both of them
    nvrhiFrameSkeleton.reset();

    // The skeleton references all of them, so they follow it immediately; all seven wrap engine
    // buffers/images and quote the RHI device, so they precede the table/context and the device
    // below. The direct pass borrows the primary's layout handles and the indirect pass borrows
    // both, so the indirect goes before the direct, and the direct before the primary; the UI pass
    // borrows the table and the frame context only.
    rhiDebugTracePass.reset();
    rhiRtComposePass.reset();
    rhiUiPass.reset();
    rhiRtIndirectPass.reset();
    rhiRtDirectPass.reset();
    rhiRtPrimaryPass.reset();
    rhiAccelStructs.reset();

    // The table's wrapped textures reference engine images and its samplers belong to the NVRHI
    // device, so it goes before the texture/sampler managers and the RHI device. The frame context
    // goes after both of its users (the skeleton and the table) and before the RHI device.
    rhiTextureTable.reset();

    rhiFrameContext.reset();

    // the RHI device holds Vulkan objects created from this device,
    // so it has to be released before them
    nvrhi.reset();

    physDevice.reset();
    queues.reset();
    swapchain.reset();
    cmdManager.reset();
    framebuffers.reset();
    volumetric.reset();
    tonemapping.reset();
    imageComposition.reset();
    bloom.reset();
    amdFsr.reset();
    nvDlss.reset();
    sharpening.reset();
    effectWipe.reset();
    effectRadialBlur.reset();
    effectChromaticAberration.reset();
    effectInverseBW.reset();
    effectHueShift.reset();
    effectDistortedSides.reset();
    effectWaves.reset();
    effectColorTint.reset();
    effectCrtDemodulateEncode.reset();
    effectCrtDecode.reset();
    uniform.reset();
    scene.reset();
    shaderManager.reset();
    rtPipeline.reset();
    pathTracer.reset();
    rasterizer.reset();
    decalManager.reset();
    portalList.reset();
    shadowMap.reset();
    godRays.reset();
    worldSamplerManager.reset();
    genericSamplerManager.reset();
    blueNoise.reset();
    textureManager.reset();
    cubemapManager.reset();

    // not covered by the list above: these own device resources too, and as
    // members they would otherwise be destroyed after DestroyDevice()
    passTimings.reset();
    q2Denoiser.reset();
    rayStats.reset();

    memAllocator.reset();

    vkDestroySurfaceKHR(instance, surface, nullptr);
    DestroySyncPrimitives();

    DestroyDevice();
    DestroyInstance();
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    if (pUserData == nullptr)
    {
        return VK_FALSE;
    }


    // DLSS: ignore error 'VUID-VkCuLaunchInfoNVX-paramCount-arraylength' - 'paramCount must be greater than 0'
    if (pCallbackData->messageIdNumber == 2044605652)
    {
        return VK_FALSE;
    }


    const char *msg;

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
    {
        msg = "Vulkan::VERBOSE::[%d][%s]\n%s\n\n";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        msg = "Vulkan::INFO::[%d][%s]\n%s\n\n";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        msg = "Vulkan::WARNING::[%d][%s]\n%s\n\n";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        msg = "Vulkan::ERROR::[%d][%s]\n%s\n\n";
    }
    else
    {
        msg = "Vulkan::[%d][%s]\n%s\n\n";
    }

    char buf[1024];
    snprintf(buf, sizeof(buf) / sizeof(buf[0]), msg, pCallbackData->messageIdNumber, pCallbackData->pMessageIdName, pCallbackData->pMessage);

    auto *userPrint = static_cast<UserPrint*>(pUserData);
    userPrint->Print(buf);

    return VK_FALSE;
}

void VulkanDevice::CreateInstance(const RgInstanceCreateInfo &info)
{
    std::vector<const char *> layerNames;

    if (libconfig.vulkanValidation)
    {
        layerNames.push_back("VK_LAYER_KHRONOS_validation");
    }

    if (libconfig.fpsMonitor)
    {
        layerNames.push_back("VK_LAYER_LUNARG_monitor");
    }

    std::vector<VkExtensionProperties> supportedInstanceExtensions;
    uint32_t supportedExtensionsCount;

    if (vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionsCount, nullptr) == VK_SUCCESS)
    {
        supportedInstanceExtensions.resize(supportedExtensionsCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionsCount, supportedInstanceExtensions.data());
    }

    std::vector<const char *> extensions =
    {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,

    #ifdef RG_USE_SURFACE_WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    #endif // RG_USE_SURFACE_WIN32

    #ifdef RG_USE_SURFACE_METAL
        VK_EXT_METAL_SURFACE_EXTENSION_NAME,
    #endif // RG_USE_SURFACE_METAL

    #ifdef RG_USE_SURFACE_WAYLAND
        VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    #endif // RG_USE_SURFACE_WAYLAND

    #ifdef RG_USE_SURFACE_XCB
        VK_KHR_XCB_SURFACE_EXTENSION_NAME,
    #endif // RG_USE_SURFACE_XCB

    #ifdef RG_USE_SURFACE_XLIB
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
    #endif // RG_USE_SURFACE_XLIB
    };

    if (libconfig.vulkanValidation)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    for (const char *n : DLSS::GetDlssVulkanInstanceExtensions())
    {
        const bool isSupported = std::any_of(supportedInstanceExtensions.cbegin(), supportedInstanceExtensions.cend(),
            [&](const VkExtensionProperties& ext)
            {
                return !std::strcmp(ext.extensionName, n);
            }
        );

        if (!isSupported)
        {
            continue;
        }

        extensions.push_back(n);
    }

    enabledInstanceExtensions.clear();
    for (const char *n : extensions)
    {
        enabledInstanceExtensions.push_back(n);
    }

    VkApplicationInfo appInfo = {};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    appInfo.pApplicationName = info.pAppName;

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.ppEnabledLayerNames = layerNames.data();
    instanceInfo.enabledLayerCount = layerNames.size();
    instanceInfo.ppEnabledExtensionNames = extensions.data();
    instanceInfo.enabledExtensionCount = extensions.size();

    VkResult r = vkCreateInstance(&instanceInfo, nullptr, &instance);
    VK_CHECKERROR(r);


    if (libconfig.vulkanValidation)
    {
        InitInstanceExtensionFunctions_DebugUtils(instance);

        if (userPrint)
        {
            // init debug utilsdebugMessenger
            VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo = {};
            debugMessengerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debugMessengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debugMessengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
            debugMessengerInfo.pfnUserCallback = DebugMessengerCallback;
            debugMessengerInfo.pUserData = static_cast<void *>(userPrint.get());

            r = svkCreateDebugUtilsMessengerEXT(instance, &debugMessengerInfo, nullptr, &debugMessenger);
            VK_CHECKERROR(r);
        }
    }
}

void VulkanDevice::CreateDevice()
{
    VkPhysicalDeviceFeatures features = {};
    features.robustBufferAccess = 1;
    features.fullDrawIndexUint32 = 1;
    features.imageCubeArray = 1;
    features.independentBlend = 1;
    features.geometryShader = 0;
    features.tessellationShader = 0;
    features.sampleRateShading = 0;
    features.dualSrcBlend = 0;
    features.logicOp = 1;
    features.multiDrawIndirect = 1;
    features.drawIndirectFirstInstance = 1;
    features.depthClamp = 1;
    features.depthBiasClamp = 1;
    features.fillModeNonSolid = 0;
    features.depthBounds = 1;
    features.wideLines = 0;
    features.largePoints = 0;
    features.alphaToOne = 0;
    features.multiViewport = 1;
    features.samplerAnisotropy = 1;
    features.textureCompressionETC2 = 0;
    features.textureCompressionASTC_LDR = 0;
    features.textureCompressionBC = 0;
    features.occlusionQueryPrecise = 0;
    features.pipelineStatisticsQuery = 1;
    features.vertexPipelineStoresAndAtomics = 1;
    features.fragmentStoresAndAtomics = 1;
    features.shaderTessellationAndGeometryPointSize = 1;
    features.shaderImageGatherExtended = 1;
    features.shaderStorageImageExtendedFormats = 1;
    features.shaderStorageImageMultisample = 1;
    features.shaderStorageImageReadWithoutFormat = 1;
    features.shaderStorageImageWriteWithoutFormat = 1;
    features.shaderUniformBufferArrayDynamicIndexing = 1;
    features.shaderSampledImageArrayDynamicIndexing = 1;
    features.shaderStorageBufferArrayDynamicIndexing = 1;
    features.shaderStorageImageArrayDynamicIndexing = 1;
    features.shaderClipDistance = 1;
    features.shaderCullDistance = 1;
    features.shaderFloat64 = 1;
    features.shaderInt64 = 1;
    features.shaderInt16 = 1;
    features.shaderResourceResidency = 1;
    features.shaderResourceMinLod = 1;
    features.sparseBinding = 0;
    features.sparseResidencyBuffer = 0;
    features.sparseResidencyImage2D = 0;
    features.sparseResidencyImage3D = 0;
    features.sparseResidency2Samples = 0;
    features.sparseResidency4Samples = 0;
    features.sparseResidency8Samples = 0;
    features.sparseResidency16Samples = 0;
    features.sparseResidencyAliased = 0;
    features.variableMultisampleRate = 0;
    features.inheritedQueries = 1;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.samplerMirrorClampToEdge = 1;
    vulkan12Features.runtimeDescriptorArray = 1;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = 1;
    vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = 1;
    vulkan12Features.bufferDeviceAddress = 1;
    vulkan12Features.shaderFloat16 = 1;
    vulkan12Features.drawIndirectCount = 1;

    // Features the RHI layer depends on, enabled only if the driver supports them.
    const NvrhiRequirements nvrhiRequirements = QueryNvrhiRequirements(physDevice->Get());
    const std::vector<std::string> unsupportedNvrhiFeatures = nvrhiRequirements.GetUnsupported();

    if (!nvrhiRequirements.IsCriticalSupported())
    {
        std::string message = "RHI: the device does not support the required features:";
        for (const std::string &name : unsupportedNvrhiFeatures)
        {
            message += " ";
            message += name;
        }

        throw RgException(RG_GRAPHICS_API_ERROR, message);
    }

    ApplyNvrhiRequirements(nvrhiRequirements, vulkan12Features);

    if (!unsupportedNvrhiFeatures.empty())
    {
        std::string message = "RHI: features not supported by the device:";
        for (const std::string &name : unsupportedNvrhiFeatures)
        {
            message += " ";
            message += name;
        }
        message += "\n";

        Print(message.c_str());
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features = {};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = nullptr; // end of chain
    vulkan13Features.computeFullSubgroups = 1;
    vulkan13Features.subgroupSizeControl = 1;
    // The RHI layer records every pass with vkCmdBeginRendering. Enabling it
    // unconditionally is safe here: IsCriticalSupported() above refuses to
    // continue on a device without dynamic rendering, so this line is only
    // reached when the feature exists. ApplyNvrhiRequirements cannot set it,
    // because this structure does not exist yet when it runs.
    vulkan13Features.dynamicRendering = 1;

    vulkan12Features.pNext = &vulkan13Features;  // chain: vk12 → vk13

    VkPhysicalDeviceMultiviewFeatures multiviewFeatures = {};
    multiviewFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
    multiviewFeatures.pNext = &vulkan12Features;
    multiviewFeatures.multiview = 1;

    VkPhysicalDevice16BitStorageFeatures storage16 = {};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16.pNext = &multiviewFeatures;
    storage16.storageBuffer16BitAccess = 1;

    VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features = {};
    sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    sync2Features.pNext = &storage16;
    sync2Features.synchronization2 = 1;

    std::vector<VkExtensionProperties> supportedDeviceExtensions;
    uint32_t supportedExtensionsCount;

    if (vkEnumerateDeviceExtensionProperties(physDevice->Get(), nullptr, &supportedExtensionsCount, nullptr) == VK_SUCCESS)
    {
        supportedDeviceExtensions.resize(supportedExtensionsCount);
        vkEnumerateDeviceExtensionProperties(physDevice->Get(), nullptr, &supportedExtensionsCount, supportedDeviceExtensions.data());
    }

    // Optional: the ray-query feature. NVRHI's state tracking maps the acceleration-structure-read
    // state to the compute stage as well as the ray-tracing one (vulkan-constants.cpp:282-285), and
    // the spec forbids the acceleration-structure-read access at any shader stage except the
    // ray-tracing ones while rayQuery is disabled (VUID-VkBufferMemoryBarrier2-srcAccessMask-06256).
    // No shader here uses inline ray tracing, but enabling the feature is what keeps a traced frame
    // validation-clean, so it is requested whenever the physical device offers the extension.
    const bool rayQuerySupported = std::any_of(supportedDeviceExtensions.cbegin(), supportedDeviceExtensions.cend(),
        [](const VkExtensionProperties& ext)
        {
            return !std::strcmp(ext.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME);
        });

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.pNext = &sync2Features;
    rayQueryFeatures.rayQuery = rayQuerySupported ? 1 : 0;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures = {};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.pNext = rayQuerySupported ? static_cast<void *>(&rayQueryFeatures) : static_cast<void *>(&sync2Features);
    rtPipelineFeatures.rayTracingPipeline = 1;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtPipelineFeatures;
    asFeatures.accelerationStructure = 1;

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
    physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    physicalDeviceFeatures2.pNext = &asFeatures;
    physicalDeviceFeatures2.features = features;

    std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,         // FSR 3.1 needs vkGetBufferMemoryRequirements2KHR
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,             // FSR 3.1 shader subgroup size
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,             // FSR 3.1 shader FP ops
    };

    for (const char *n : DLSS::GetDlssVulkanDeviceExtensions())
    {
        const bool isSupported = std::any_of(supportedDeviceExtensions.cbegin(), supportedDeviceExtensions.cend(),
            [&](const VkExtensionProperties& ext)
            {
                return !std::strcmp(ext.extensionName, n);
            }
        );

        if (!isSupported)
        {
            continue;
        }

        deviceExtensions.push_back(n);
    }

    if (rayQuerySupported)
    {
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }

    enabledDeviceExtensions.clear();
    for (const char *n : deviceExtensions)
    {
        enabledDeviceExtensions.push_back(n);
    }


    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queues->GetDeviceQueueCreateInfos(queueCreateInfos);

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;
    deviceCreateInfo.pNext = &physicalDeviceFeatures2;
    deviceCreateInfo.enabledExtensionCount = (uint32_t) deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkResult r = vkCreateDevice(physDevice->Get(), &deviceCreateInfo, nullptr, &device);
    VK_CHECKERROR(r);

    InitDeviceExtensionFunctions(device);

    if (libconfig.vulkanValidation)
    {
        InitDeviceExtensionFunctions_DebugUtils(device);
    }
}

void VulkanDevice::CreateNvrhiDevice()
{
    // The features were already enabled on the device in CreateDevice; they are
    // queried again here to know which of them the driver actually has.
    const NvrhiRequirements requirements = QueryNvrhiRequirements(physDevice->Get());

    std::vector<const char *> instanceExtensions;
    instanceExtensions.reserve(enabledInstanceExtensions.size());
    for (const std::string &name : enabledInstanceExtensions)
    {
        instanceExtensions.push_back(name.c_str());
    }

    std::vector<const char *> deviceExtensions;
    deviceExtensions.reserve(enabledDeviceExtensions.size());
    for (const std::string &name : enabledDeviceExtensions)
    {
        deviceExtensions.push_back(name.c_str());
    }

    NvrhiDeviceInfo info = {};
    info.instance = instance;
    info.physicalDevice = physDevice->Get();
    info.device = device;

    info.graphicsQueue = queues->GetGraphics();
    info.graphicsQueueIndex = queues->GetIndexGraphics();
    info.computeQueue = queues->GetCompute();
    info.computeQueueIndex = queues->GetIndexCompute();
    info.transferQueue = queues->GetTransfer();
    info.transferQueueIndex = queues->GetIndexTransfer();

    info.instanceExtensions = instanceExtensions.data();
    info.instanceExtensionCount = instanceExtensions.size();
    info.deviceExtensions = deviceExtensions.data();
    info.deviceExtensionCount = deviceExtensions.size();

    info.bufferDeviceAddressSupported = true;
    info.uniformBufferUpdateAfterBindSupported = requirements.descriptorBindingUniformBufferUpdateAfterBind;

    nvrhi = std::make_unique<NvrhiContext>();

    std::string errorMessage;
    if (!nvrhi->Init(info, [this](const char *pMessage) { Print(pMessage); }, errorMessage))
    {
        nvrhi.reset();
        throw RgException(RG_GRAPHICS_API_ERROR, errorMessage);
    }

    nvrhi->LogCapabilities();
}

void VulkanDevice::CreateSyncPrimitives()
{
    VkResult r;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFenceCreateInfo nonSignaledFenceInfo = {};
    nonSignaledFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        r = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
        VK_CHECKERROR(r);
        r = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
        VK_CHECKERROR(r);
        r = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &inFrameSemaphores[i]);
        VK_CHECKERROR(r);

        r = vkCreateFence(device, &fenceInfo, nullptr, &frameFences[i]);
        VK_CHECKERROR(r);
        r = vkCreateFence(device, &nonSignaledFenceInfo, nullptr, &outOfFrameFences[i]);
        VK_CHECKERROR(r);

        SET_DEBUG_NAME(device, imageAvailableSemaphores[i], VK_OBJECT_TYPE_SEMAPHORE, "Image available semaphore");
        SET_DEBUG_NAME(device, renderFinishedSemaphores[i], VK_OBJECT_TYPE_SEMAPHORE, "Render finished semaphore");
        SET_DEBUG_NAME(device, inFrameSemaphores[i], VK_OBJECT_TYPE_SEMAPHORE, "In-frame semaphore");
        SET_DEBUG_NAME(device, frameFences[i], VK_OBJECT_TYPE_FENCE, "Frame fence");
        SET_DEBUG_NAME(device, outOfFrameFences[i], VK_OBJECT_TYPE_FENCE, "Out of frame fence");
    }
}

VkSurfaceKHR VulkanDevice::GetSurfaceFromUser(VkInstance instance, const RgInstanceCreateInfo &info)
{
    VkSurfaceKHR surface;
    VkResult r;


#ifdef RG_USE_SURFACE_WIN32
    if (info.pWin32SurfaceInfo != nullptr)
    {
        VkWin32SurfaceCreateInfoKHR win32Info = {};
        win32Info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        win32Info.hinstance = info.pWin32SurfaceInfo->hinstance;
        win32Info.hwnd = info.pWin32SurfaceInfo->hwnd;

        r = vkCreateWin32SurfaceKHR(instance, &win32Info, nullptr, &surface);
        VK_CHECKERROR(r);

        return surface;
    }
#else
    if (info.pWin32SurfaceInfo != nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "pWin32SurfaceInfo is specified, but the library wasn't built with RG_USE_SURFACE_WIN32 option");
    }
#endif // RG_USE_SURFACE_WIN32


#ifdef RG_USE_SURFACE_METAL
    if (info.pMetalSurfaceCreateInfo != nullptr)
    {
        VkMetalSurfaceCreateInfoEXT metalInfo = {};
        metalInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        metalInfo.pLayer = info.pMetalSurfaceCreateInfo->pLayer;

        r = vkCreateMetalSurfaceEXT(instance, &metalInfo, nullptr, &surface);
        VK_CHECKERROR(r);

        return surface;
    }
#else
    if (info.pMetalSurfaceCreateInfo != nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "pMetalSurfaceCreateInfo is specified, but the library wasn't built with RG_USE_SURFACE_METAL option");
    }
#endif // RG_USE_SURFACE_METAL


#ifdef RG_USE_SURFACE_WAYLAND
    if (info.pWaylandSurfaceCreateInfo != nullptr)
    {
        VkWaylandSurfaceCreateInfoKHR wlInfo = {};
        wlInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        wlInfo.display = info.pWaylandSurfaceCreateInfo->display;
        wlInfo.surface = info.pWaylandSurfaceCreateInfo->surface;

        r = (instance, &wlInfo, nullptr, &surface);
        VK_CHECKERROR(r);

        return surface;
    }
#else
    if (info.pWaylandSurfaceCreateInfo != nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "pWaylandSurfaceCreateInfo is specified, but the library wasn't built with RG_USE_SURFACE_WAYLAND option");
    }
#endif // RG_USE_SURFACE_WAYLAND


#ifdef RG_USE_SURFACE_XCB
    if (info.pXcbSurfaceCreateInfo != nullptr)
    {
        VkXcbSurfaceCreateInfoKHR xcbInfo = {};
        xcbInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        xcbInfo.connection = info.pXcbSurfaceCreateInfo->connection;
        xcbInfo.window = info.pXcbSurfaceCreateInfo->window;

        r = vkCreateXcbSurfaceKHR(instance, &xcbInfo, nullptr, &surface);
        VK_CHECKERROR(r);

        return surface;
    }
#else
    if (info.pXcbSurfaceCreateInfo != nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "pXcbSurfaceCreateInfo is specified, but the library wasn't built with RG_USE_SURFACE_XCB option");
    }
#endif // RG_USE_SURFACE_XCB


#ifdef RG_USE_SURFACE_XLIB
    if (info.pXlibSurfaceCreateInfo != nullptr)
    {
        VkXlibSurfaceCreateInfoKHR xlibInfo = {};
        xlibInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        xlibInfo.dpy = info.pXlibSurfaceCreateInfo->dpy;
        xlibInfo.window = info.pXlibSurfaceCreateInfo->window;

        r = vkCreateXlibSurfaceKHR(instance, &xlibInfo, nullptr, &surface);
        VK_CHECKERROR(r);

        return surface;
    }
#else
    if (info.pXlibSurfaceCreateInfo != nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "pXlibSurfaceCreateInfo is specified, but the library wasn't built with RG_USE_SURFACE_XLIB option");
    }
#endif // RG_USE_SURFACE_XLIB


    throw RgException(RG_WRONG_ARGUMENT, "Surface info wasn't specified");
}

void VulkanDevice::DestroyInstance()
{
    if (debugMessenger != VK_NULL_HANDLE)
    {
        svkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroyInstance(instance, nullptr);
}

void VulkanDevice::DestroyDevice()
{
    vkDestroyDevice(device, nullptr);
}

void VulkanDevice::DestroySyncPrimitives()
{
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, inFrameSemaphores[i], nullptr);

        vkDestroyFence(device, frameFences[i], nullptr);
        vkDestroyFence(device, outOfFrameFences[i], nullptr);
    }
}

void vkpt::VulkanDevice::ValidateCreateInfo(const RgInstanceCreateInfo *pInfo)
{
    using namespace std::string_literals;

    if (pInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    {
        int count =
            !!pInfo->pWin32SurfaceInfo +
            !!pInfo->pMetalSurfaceCreateInfo +
            !!pInfo->pWaylandSurfaceCreateInfo +
            !!pInfo->pXcbSurfaceCreateInfo +
            !!pInfo->pXlibSurfaceCreateInfo;

        if (count != 1)
        {
            throw RgException(RG_WRONG_ARGUMENT, "Exactly one of the surface infos must be not null");
        }
    }

    if (pInfo->rasterizedSkyCubemapSize == 0)
    {
        throw RgException(RG_WRONG_ARGUMENT, "rasterizedSkyCubemapSize must be non-zero");
    }

    if (pInfo->primaryRaysMaxAlbedoLayers > MATERIALS_MAX_LAYER_COUNT)
    {
        throw RgException(RG_WRONG_ARGUMENT, "primaryRaysMaxAlbedoLayers must be <="s + std::to_string(MATERIALS_MAX_LAYER_COUNT));
    }

    if (pInfo->indirectIlluminationMaxAlbedoLayers > MATERIALS_MAX_LAYER_COUNT)
    {
        throw RgException(RG_WRONG_ARGUMENT, "indirectIlluminationMaxAlbedoLayers must be <="s + std::to_string(MATERIALS_MAX_LAYER_COUNT));
    }
}
