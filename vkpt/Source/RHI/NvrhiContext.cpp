#include "NvrhiContext.h"

// NVRHI creates its Vulkan objects through the default dispatcher of
// Vulkan-Hpp. When NVRHI is built as a shared library it defines the storage of
// that dispatcher itself; a static build (the one the renderer uses) leaves it
// to the application, and it has to be defined in exactly one translation unit
// of the program - this one.
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <cstdio>
#include <exception>
#include <utility>

namespace vkpt
{

namespace
{
    const char *GetGraphicsAPIName(nvrhi::GraphicsAPI api)
    {
        switch (api)
        {
        case nvrhi::GraphicsAPI::VULKAN: return "Vulkan";
        case nvrhi::GraphicsAPI::D3D12:  return "D3D12";
        case nvrhi::GraphicsAPI::D3D11:  return "D3D11";
        }

        return "unknown";
    }

    const char *GetSeverityName(nvrhi::MessageSeverity severity)
    {
        switch (severity)
        {
        case nvrhi::MessageSeverity::Info:    return "INFO";
        case nvrhi::MessageSeverity::Warning: return "WARNING";
        case nvrhi::MessageSeverity::Error:   return "ERROR";
        case nvrhi::MessageSeverity::Fatal:   return "FATAL";
        }

        return "UNKNOWN";
    }
}

class NvrhiContext::MessageCallback : public nvrhi::IMessageCallback
{
public:
    explicit MessageCallback(NvrhiContext *pOwner) : owner(pOwner) {}

    void message(nvrhi::MessageSeverity severity, const char *messageText) override
    {
        std::string text = "NVRHI::";
        text += GetSeverityName(severity);
        text += ":: ";
        text += messageText != nullptr ? messageText : "";
        text += "\n";

        owner->Print(text.c_str());
    }

private:
    NvrhiContext *owner;
};

NvrhiContext::NvrhiContext()
{
}

NvrhiContext::~NvrhiContext()
{
    Shutdown();
}

bool NvrhiContext::Init(const NvrhiDeviceInfo &info, PrintFunction pfnPrint, std::string &errorMessage)
{
    print = std::move(pfnPrint);
    messageCallback = std::make_unique<MessageCallback>(this);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(info.instance, vkGetInstanceProcAddr, info.device);

    // NVRHI never creates the instance, the device or the queues itself: it is
    // given the handles of the ones the renderer created.
    nvrhi::vulkan::DeviceDesc desc;
    desc.errorCB = messageCallback.get();
    desc.instance = info.instance;
    desc.physicalDevice = info.physicalDevice;
    desc.device = info.device;

    desc.graphicsQueue = info.graphicsQueue;
    desc.graphicsQueueIndex = info.graphicsQueue != VK_NULL_HANDLE ? static_cast<int>(info.graphicsQueueIndex) : -1;
    desc.computeQueue = info.computeQueue;
    desc.computeQueueIndex = info.computeQueue != VK_NULL_HANDLE ? static_cast<int>(info.computeQueueIndex) : -1;
    desc.transferQueue = info.transferQueue;
    desc.transferQueueIndex = info.transferQueue != VK_NULL_HANDLE ? static_cast<int>(info.transferQueueIndex) : -1;

    desc.instanceExtensions = info.instanceExtensions;
    desc.numInstanceExtensions = info.instanceExtensionCount;
    desc.deviceExtensions = info.deviceExtensions;
    desc.numDeviceExtensions = info.deviceExtensionCount;

    desc.bufferDeviceAddressSupported = info.bufferDeviceAddressSupported;
    desc.descriptorBindingUniformBufferUpdateAfterBind = info.uniformBufferUpdateAfterBindSupported;

    try
    {
        device = nvrhi::vulkan::createDevice(desc);
    }
    catch (const std::exception &e)
    {
        errorMessage = "NVRHI device creation failed: ";
        errorMessage += e.what();
        messageCallback.reset();
        return false;
    }

    if (!device)
    {
        errorMessage = "NVRHI device creation failed";
        messageCallback.reset();
        return false;
    }

    return true;
}

void NvrhiContext::Shutdown()
{
    device = nullptr;
    messageCallback.reset();
}

nvrhi::IDevice *NvrhiContext::GetDevice() const
{
    return device.Get();
}

void NvrhiContext::LogCapabilities() const
{
    if (!device)
    {
        return;
    }

    char buf[512];
    snprintf(buf, sizeof(buf) / sizeof(buf[0]),
        "NVRHI: API %s; queues: compute %s, copy %s; ray tracing: pipeline %s, acceleration structures %s, ray query %s\n",
        GetGraphicsAPIName(device->getGraphicsAPI()),
        device->queryFeatureSupport(nvrhi::Feature::ComputeQueue) ? "yes" : "no",
        device->queryFeatureSupport(nvrhi::Feature::CopyQueue) ? "yes" : "no",
        device->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline) ? "yes" : "no",
        device->queryFeatureSupport(nvrhi::Feature::RayTracingAccelStruct) ? "yes" : "no",
        device->queryFeatureSupport(nvrhi::Feature::RayQuery) ? "yes" : "no");

    Print(buf);

    snprintf(buf, sizeof(buf) / sizeof(buf[0]),
        "NVRHI: shader specializations %s\n",
        device->queryFeatureSupport(nvrhi::Feature::ShaderSpecializations) ? "yes" : "no");

    Print(buf);
}

void NvrhiContext::Print(const char *pMessage) const
{
    if (print)
    {
        print(pMessage);
    }
}

}
