#include "NvrhiRequirements.h"

#include <cstddef>

namespace vkpt
{

std::vector<std::string> NvrhiRequirements::GetUnsupported() const
{
    std::vector<std::string> unsupported;

    if (!timelineSemaphore)
    {
        unsupported.push_back("timelineSemaphore");
    }
    if (!descriptorBindingSampledImageUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingSampledImageUpdateAfterBind");
    }
    if (!descriptorBindingStorageImageUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingStorageImageUpdateAfterBind");
    }
    if (!descriptorBindingStorageBufferUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingStorageBufferUpdateAfterBind");
    }
    if (!descriptorBindingUniformTexelBufferUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingUniformTexelBufferUpdateAfterBind");
    }
    if (!descriptorBindingStorageTexelBufferUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingStorageTexelBufferUpdateAfterBind");
    }
    if (!descriptorBindingUniformBufferUpdateAfterBind)
    {
        unsupported.push_back("descriptorBindingUniformBufferUpdateAfterBind");
    }
    if (!descriptorBindingPartiallyBound)
    {
        unsupported.push_back("descriptorBindingPartiallyBound");
    }
    if (!descriptorBindingUpdateUnusedWhilePending)
    {
        unsupported.push_back("descriptorBindingUpdateUnusedWhilePending");
    }
    if (!shaderStorageImageArrayNonUniformIndexing)
    {
        unsupported.push_back("shaderStorageImageArrayNonUniformIndexing");
    }
    if (!dynamicRendering)
    {
        unsupported.push_back("dynamicRendering");
    }

    return unsupported;
}

bool NvrhiRequirements::IsCriticalSupported() const
{
    return timelineSemaphore
        && descriptorBindingPartiallyBound
        && descriptorBindingUpdateUnusedWhilePending
        && descriptorBindingSampledImageUpdateAfterBind
        && descriptorBindingStorageImageUpdateAfterBind
        && descriptorBindingStorageBufferUpdateAfterBind
        && dynamicRendering;
}

NvrhiRequirements QueryNvrhiRequirements(VkPhysicalDevice physDevice)
{
    VkPhysicalDeviceVulkan13Features supported13 = {};
    supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features supported12 = {};
    supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supported12.pNext = &supported13;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &supported12;

    vkGetPhysicalDeviceFeatures2(physDevice, &features2);

    NvrhiRequirements requirements;
    requirements.timelineSemaphore = supported12.timelineSemaphore == VK_TRUE;
    requirements.dynamicRendering = supported13.dynamicRendering == VK_TRUE;

    requirements.descriptorBindingSampledImageUpdateAfterBind  = supported12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingStorageImageUpdateAfterBind  = supported12.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingStorageBufferUpdateAfterBind = supported12.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingUniformTexelBufferUpdateAfterBind = supported12.descriptorBindingUniformTexelBufferUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingStorageTexelBufferUpdateAfterBind = supported12.descriptorBindingStorageTexelBufferUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingUniformBufferUpdateAfterBind = supported12.descriptorBindingUniformBufferUpdateAfterBind == VK_TRUE;
    requirements.descriptorBindingPartiallyBound               = supported12.descriptorBindingPartiallyBound == VK_TRUE;
    requirements.descriptorBindingUpdateUnusedWhilePending     = supported12.descriptorBindingUpdateUnusedWhilePending == VK_TRUE;
    requirements.shaderStorageImageArrayNonUniformIndexing     = supported12.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;

    return requirements;
}

void ApplyNvrhiRequirements(const NvrhiRequirements &requirements,
                            VkPhysicalDeviceVulkan12Features &features12)
{
    if (requirements.timelineSemaphore)
    {
        features12.timelineSemaphore = VK_TRUE;
    }
    if (requirements.descriptorBindingSampledImageUpdateAfterBind)
    {
        features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingStorageImageUpdateAfterBind)
    {
        features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingStorageBufferUpdateAfterBind)
    {
        features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingUniformTexelBufferUpdateAfterBind)
    {
        features12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingStorageTexelBufferUpdateAfterBind)
    {
        features12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingUniformBufferUpdateAfterBind)
    {
        features12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    }
    if (requirements.descriptorBindingPartiallyBound)
    {
        features12.descriptorBindingPartiallyBound = VK_TRUE;
    }
    if (requirements.descriptorBindingUpdateUnusedWhilePending)
    {
        features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    }
    if (requirements.shaderStorageImageArrayNonUniformIndexing)
    {
        features12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    }
}

}
