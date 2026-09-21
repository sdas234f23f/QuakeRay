#pragma once

#include <nvrhi/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vkpt
{

// Handles of the Vulkan objects the renderer has already created.
//
// NVRHI does not create the instance, the physical device, the device or the
// queues on Vulkan: it wraps the handles it is given.
struct NvrhiDeviceInfo
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueIndex = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueIndex = 0;
    VkQueue transferQueue = VK_NULL_HANDLE;
    uint32_t transferQueueIndex = 0;

    const char **instanceExtensions = nullptr;
    size_t instanceExtensionCount = 0;
    const char **deviceExtensions = nullptr;
    size_t deviceExtensionCount = 0;

    bool bufferDeviceAddressSupported = false;
    bool uniformBufferUpdateAfterBindSupported = false;
};

// Owns the NVRHI device that is created over the Vulkan device of the caller.
class NvrhiContext
{
public:
    using PrintFunction = std::function<void(const char *)>;

    NvrhiContext();
    ~NvrhiContext();

    NvrhiContext(const NvrhiContext &other) = delete;
    NvrhiContext(NvrhiContext &&other) noexcept = delete;
    NvrhiContext &operator=(const NvrhiContext &other) = delete;
    NvrhiContext &operator=(NvrhiContext &&other) noexcept = delete;

    // Creates the NVRHI device over the already created Vulkan device.
    // Returns false and fills errorMessage on failure.
    bool Init(const NvrhiDeviceInfo &info, PrintFunction pfnPrint, std::string &errorMessage);

    // Releases the NVRHI device. Has to be called before vkDestroyDevice.
    void Shutdown();

    nvrhi::IDevice *GetDevice() const;

    // Logs the capabilities that the RHI layer exposes.
    void LogCapabilities() const;

private:
    void Print(const char *pMessage) const;

    class MessageCallback;

    PrintFunction print;
    std::unique_ptr<MessageCallback> messageCallback;
    nvrhi::DeviceHandle device;
};

}
