// FidelityFX Vulkan backend: stub for the frame interpolation swap chain.
//
// `ffxGetInterfaceVK` (sdk/src/backends/vk/ffx_vk.cpp) unconditionally assigns
// `fpSwapChainConfigureFrameGeneration = ffxSetFrameGenerationConfigToSwapchainVK`, and that
// symbol lives in `sdk/src/backends/vk/FrameInterpolationSwapchain/FrameInterpolationSwapchainVK.cpp`.
// That directory is not vendored here: it drags in the whole FSR3 host side
// (`FidelityFX/host/ffx_fsr3.h`) plus the prebuilt FSR3/DX12 signed DLLs, none of which the
// shadow denoiser needs.
//
// The denoiser only validates fpGetSDKVersion, fpGetDeviceCapabilities, fpCreateBackendContext and
// fpDestroyBackendContext when the context is created, so this callback is never invoked by the
// denoiser and a stub keeps the vendored VK backend linkable. If frame interpolation is ever
// integrated, vendor the real swap chain and delete this file.

#include <FidelityFX/host/ffx_error.h>
#include <FidelityFX/host/ffx_interface.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>

FFX_API FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const* /*config*/)
{
    return FFX_ERROR_INVALID_ARGUMENT;
}
