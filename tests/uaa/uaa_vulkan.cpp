/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN

#define VK_NO_PROTOTYPES
#include <nvrhi/vulkan.h>

#if !defined(UAA_NVRHI_SHARED) || !UAA_NVRHI_SHARED
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

#include "uaa_backend.h"
#include "uaa_fixture.h"
#include "uaa_runner.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nvrhi::uaa
{
    namespace
    {
        template <typename T>
        nvrhi::Object vkHandleToObject(T handle)
        {
            if constexpr (std::is_pointer_v<T>)
            {
                return nvrhi::Object(static_cast<void*>(handle));
            }
            else
            {
                return nvrhi::Object(static_cast<uint64_t>(handle));
            }
        }

        std::string toLowerAscii(std::string value)
        {
            for (char& ch : value)
            {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return value;
        }

        bool containsCaseInsensitive(const std::string& haystack, const std::string& needle)
        {
            if (needle.empty())
            {
                return true;
            }
            return toLowerAscii(haystack).find(toLowerAscii(needle)) != std::string::npos;
        }

        bool tryParseIndex(const std::string& str, size_t& outIndex)
        {
            if (str.empty())
            {
                return false;
            }
            for (char ch : str)
            {
                if (!std::isdigit(static_cast<unsigned char>(ch)))
                {
                    return false;
                }
            }
            char* endPtr = nullptr;
            unsigned long val = std::strtoul(str.c_str(), &endPtr, 10);
            if (endPtr != str.c_str() + str.size())
            {
                return false;
            }
            outIndex = static_cast<size_t>(val);
            return true;
        }

        const char* getOsName()
        {
#ifdef _WIN32
            return "Windows";
#elif defined(__APPLE__)
            return "macOS";
#elif defined(__linux__)
            return "Linux";
#else
            return "Unknown";
#endif
        }

        struct DynamicLibrary
        {
#ifdef _WIN32
            HMODULE handle = nullptr;
#else
            void* handle = nullptr;
#endif
            std::string path;

            ~DynamicLibrary()
            {
                close();
            }

            bool open(const char* libPath)
            {
                close();
#ifdef _WIN32
                handle = LoadLibraryA(libPath);
#else
                handle = dlopen(libPath, RTLD_NOW | RTLD_LOCAL);
#endif
                if (handle)
                {
                    path = libPath ? libPath : "";
                    return true;
                }
                return false;
            }

            void close()
            {
                if (handle)
                {
#ifdef _WIN32
                    FreeLibrary(handle);
#else
                    dlclose(handle);
#endif
                    handle = nullptr;
                }
                path.clear();
            }

            template <typename Fn>
            Fn loadSymbol(const char* name) const
            {
                if (!handle)
                {
                    return nullptr;
                }
#ifdef _WIN32
                return reinterpret_cast<Fn>(GetProcAddress(handle, name));
#else
                return reinterpret_cast<Fn>(dlsym(handle, name));
#endif
            }
        };

        struct VulkanFunctions
        {
            PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
            PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
            PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = nullptr;
            PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = nullptr;
            PFN_vkCreateInstance vkCreateInstance = nullptr;
            PFN_vkDestroyInstance vkDestroyInstance = nullptr;
            PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
            PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
            PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
            PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
            PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;
            PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
            PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
            PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = nullptr;
            PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
            PFN_vkCreateDevice vkCreateDevice = nullptr;
            PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
            PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = nullptr;
            PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;

            PFN_vkDestroyDevice vkDestroyDevice = nullptr;
            PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
            PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
            PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
            PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
            PFN_vkSignalSemaphore vkSignalSemaphore = nullptr;
            PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue = nullptr;
            PFN_vkCreateBuffer vkCreateBuffer = nullptr;
            PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
            PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
            PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
            PFN_vkCreateImage vkCreateImage = nullptr;
            PFN_vkDestroyImage vkDestroyImage = nullptr;
            PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = nullptr;
            PFN_vkBindImageMemory vkBindImageMemory = nullptr;
            PFN_vkAllocateMemory vkAllocateMemory = nullptr;
            PFN_vkFreeMemory vkFreeMemory = nullptr;
            PFN_vkMapMemory vkMapMemory = nullptr;
            PFN_vkUnmapMemory vkUnmapMemory = nullptr;
            PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
            PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
            PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
            PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
            PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
            PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
            PFN_vkQueueSubmit vkQueueSubmit = nullptr;
            PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
        };

        bool hasExtension(const std::vector<VkExtensionProperties>& available, const char* name)
        {
            for (const auto& ext : available)
            {
                if (std::strcmp(ext.extensionName, name) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        bool hasExtensionName(const std::vector<const char*>& list, const char* name)
        {
            for (const char* item : list)
            {
                if (std::strcmp(item, name) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        bool hasLayer(const std::vector<VkLayerProperties>& available, const char* name)
        {
            for (const auto& layer : available)
            {
                if (std::strcmp(layer.layerName, name) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        template <typename StructT>
        void chainFeatureStruct(void*& head, StructT& s)
        {
            s.pNext = head;
            head = &s;
        }

        struct VulkanValidationState
        {
            std::mutex mutex;
            std::vector<std::string> errors;
        };

        VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugUtilsCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData)
        {
            auto* state = static_cast<VulkanValidationState*>(pUserData);
            if (!state || !pCallbackData || !pCallbackData->pMessage)
            {
                return VK_FALSE;
            }

            const bool isError = (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
            const bool isValidationWarning =
                ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) &&
                ((messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0);

            if (isError || isValidationWarning)
            {
                std::ostringstream oss;
                oss << (isError ? "[VulkanError]" : "[VulkanWarning]");
                if (pCallbackData->pMessageIdName)
                {
                    oss << " [" << pCallbackData->pMessageIdName << "]";
                }
                oss << " " << pCallbackData->pMessage;
                std::lock_guard<std::mutex> lock(state->mutex);
                state->errors.push_back(oss.str());
            }

            return VK_FALSE;
        }

        struct SelectedQueues
        {
            int graphicsFamily = -1;
            int graphicsIndex = -1;
            int computeFamily = -1;
            int computeIndex = -1;
            int transferFamily = -1;
            int transferIndex = -1;
            std::map<uint32_t, uint32_t> familyQueueCounts;
        };

        SelectedQueues selectVulkanQueues(const std::vector<VkQueueFamilyProperties>& families)
        {
            SelectedQueues q;
            for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i)
            {
                if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && families[i].queueCount > 0)
                {
                    q.graphicsFamily = static_cast<int>(i);
                    q.graphicsIndex = 0;
                    q.familyQueueCounts[i] = 1;
                    break;
                }
            }

            if (q.graphicsFamily < 0)
            {
                return q;
            }

            // Prefer a dedicated compute family without graphics.
            for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i)
            {
                if (families[i].queueCount > 0 &&
                    (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                {
                    uint32_t used = q.familyQueueCounts[i];
                    if (used < families[i].queueCount)
                    {
                        q.computeFamily = static_cast<int>(i);
                        q.computeIndex = static_cast<int>(used);
                        q.familyQueueCounts[i] = used + 1;
                        break;
                    }
                }
            }

            // Otherwise try a distinct compute queue in another family or second queue in graphics family.
            if (q.computeFamily < 0)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i)
                {
                    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                    {
                        uint32_t used = q.familyQueueCounts[i];
                        if (used < families[i].queueCount)
                        {
                            q.computeFamily = static_cast<int>(i);
                            q.computeIndex = static_cast<int>(used);
                            q.familyQueueCounts[i] = used + 1;
                            break;
                        }
                    }
                }
            }

            // Prefer a dedicated transfer-only family (no graphics, no compute).
            for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i)
            {
                if (families[i].queueCount > 0 &&
                    (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    !(families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
                {
                    uint32_t used = q.familyQueueCounts[i];
                    if (used < families[i].queueCount)
                    {
                        q.transferFamily = static_cast<int>(i);
                        q.transferIndex = static_cast<int>(used);
                        q.familyQueueCounts[i] = used + 1;
                        break;
                    }
                }
            }

            // Otherwise pick another distinct queue that supports transfer.
            if (q.transferFamily < 0)
            {
                for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i)
                {
                    if (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
                    {
                        uint32_t used = q.familyQueueCounts[i];
                        if (used < families[i].queueCount)
                        {
                            q.transferFamily = static_cast<int>(i);
                            q.transferIndex = static_cast<int>(used);
                            q.familyQueueCounts[i] = used + 1;
                            break;
                        }
                    }
                }
            }

            return q;
        }

        struct VulkanLogicalDeviceSetup
        {
            VkDevice device = VK_NULL_HANDLE;
            SelectedQueues queues;
            std::vector<const char*> enabledExtensions;
            bool timelineSemaphoreSupported = false;
            bool bufferDeviceAddressSupported = false;
            bool drawIndirectCount = false;
            bool samplerFilterMinmax = false;
            bool descriptorBindingUniformBufferUpdateAfterBind = false;
        };

        bool createLogicalVulkanDevice(
            const VulkanFunctions& fn,
            VkPhysicalDevice physDevice,
            const VkPhysicalDeviceProperties& physProps,
            uint32_t instanceApiVersion,
            VulkanLogicalDeviceSetup& outSetup,
            std::string& outError)
        {
            uint32_t qfCount = 0;
            fn.vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(qfCount);
            if (qfCount > 0)
            {
                fn.vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, families.data());
            }

            outSetup.queues = selectVulkanQueues(families);
            if (outSetup.queues.graphicsFamily < 0)
            {
                outError = "Selected Vulkan physical device has no graphics queue family";
                return false;
            }

            uint32_t extCount = 0;
            if (fn.vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extCount, nullptr) != VK_SUCCESS)
            {
                outError = "Failed to enumerate Vulkan device extension count";
                return false;
            }
            std::vector<VkExtensionProperties> availableExts(extCount);
            if (extCount > 0 &&
                fn.vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extCount, availableExts.data()) != VK_SUCCESS)
            {
                outError = "Failed to enumerate Vulkan device extensions";
                return false;
            }

            const bool hasPortabilitySubset = hasExtension(availableExts, "VK_KHR_portability_subset");
            const uint32_t effectiveApiVersion = std::min(instanceApiVersion, physProps.apiVersion);
            const bool supportsVulkan12 = (effectiveApiVersion >= VK_API_VERSION_1_2);
            const bool supportsVulkan13 = (effectiveApiVersion >= VK_API_VERSION_1_3);
            const bool hasFeatures2 = (fn.vkGetPhysicalDeviceFeatures2 != nullptr) && (effectiveApiVersion >= VK_API_VERSION_1_1);

            VkPhysicalDeviceFeatures2 queriedFeatures2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            VkPhysicalDeviceVulkan11Features queriedV11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
            VkPhysicalDeviceVulkan12Features queriedV12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
            VkPhysicalDeviceVulkan13Features queriedV13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
            VkPhysicalDeviceSynchronization2FeaturesKHR queriedSync2Khr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR };
            VkPhysicalDeviceDynamicRenderingFeaturesKHR queriedDynRenderKhr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
            VkPhysicalDeviceTimelineSemaphoreFeaturesKHR queriedTimelineKhr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR queriedAccelStruct{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
            VkPhysicalDeviceRayQueryFeaturesKHR queriedRayQuery{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR queriedRtPipeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
            VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR queriedRtPosFetch{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR };
            VkPhysicalDeviceFragmentShadingRateFeaturesKHR queriedVrs{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR };
            VkPhysicalDeviceMeshShaderFeaturesEXT queriedMeshShader{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
            VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT queriedMutableDesc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
            VkPhysicalDeviceDepthClipEnableFeaturesEXT queriedDepthClip{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT };
            VkPhysicalDeviceOpacityMicromapFeaturesEXT queriedOmm{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT };
#ifdef VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME
            VkPhysicalDeviceCooperativeVectorFeaturesNV queriedCoopVec{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV };
#endif
#ifdef VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME
            VkPhysicalDeviceClusterAccelerationStructureFeaturesNV queriedClusterAs{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV };
#endif
#ifdef VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME
            VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV queriedLss{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV };
#endif
#ifdef VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME
            VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV queriedSer{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV };
#endif

            const bool availDeferredHostOps = hasExtension(availableExts, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            const bool availAccelStruct = hasExtension(availableExts, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            const bool availRayQuery = hasExtension(availableExts, VK_KHR_RAY_QUERY_EXTENSION_NAME);
            const bool availRtPipeline = hasExtension(availableExts, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            const bool availRtPosFetch = hasExtension(availableExts, VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME);
            const bool availVrs = hasExtension(availableExts, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
            const bool availMeshShader = hasExtension(availableExts, VK_EXT_MESH_SHADER_EXTENSION_NAME);
            const bool availConservativeRast = hasExtension(availableExts, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            const bool availMutableDesc = hasExtension(availableExts, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
            const bool availDepthClip = hasExtension(availableExts, VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
            const bool availOmm = hasExtension(availableExts, VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
            const bool availTimelineKhr = !supportsVulkan12 && hasExtension(availableExts, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            const bool availSync2Khr = !supportsVulkan13 && hasExtension(availableExts, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
            const bool availDynRenderKhr = !supportsVulkan13 && supportsVulkan12 &&
                hasExtension(availableExts, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
#ifdef VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME
            const bool availCoopVec = hasExtension(availableExts, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
#endif
#ifdef VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME
            const bool availClusterAs = hasExtension(availableExts, VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
#endif
#ifdef VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME
            const bool availLss = hasExtension(availableExts, VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME);
#endif
#ifdef VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME
            const bool availSer = hasExtension(availableExts, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
#endif

            if (hasFeatures2)
            {
                void* qNext = nullptr;
                if (supportsVulkan12)
                {
                    chainFeatureStruct(qNext, queriedV11);
                    chainFeatureStruct(qNext, queriedV12);
                }
                else if (availTimelineKhr)
                {
                    chainFeatureStruct(qNext, queriedTimelineKhr);
                }

                if (supportsVulkan13)
                {
                    chainFeatureStruct(qNext, queriedV13);
                }
                else
                {
                    if (availSync2Khr) chainFeatureStruct(qNext, queriedSync2Khr);
                    if (availDynRenderKhr) chainFeatureStruct(qNext, queriedDynRenderKhr);
                }

                if (availAccelStruct) chainFeatureStruct(qNext, queriedAccelStruct);
                if (availRayQuery) chainFeatureStruct(qNext, queriedRayQuery);
                if (availRtPipeline) chainFeatureStruct(qNext, queriedRtPipeline);
                if (availRtPosFetch) chainFeatureStruct(qNext, queriedRtPosFetch);
                if (availVrs) chainFeatureStruct(qNext, queriedVrs);
                if (availMeshShader) chainFeatureStruct(qNext, queriedMeshShader);
                if (availMutableDesc) chainFeatureStruct(qNext, queriedMutableDesc);
                if (availDepthClip) chainFeatureStruct(qNext, queriedDepthClip);
                if (availOmm) chainFeatureStruct(qNext, queriedOmm);
#ifdef VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME
                if (availCoopVec) chainFeatureStruct(qNext, queriedCoopVec);
#endif
#ifdef VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME
                if (availClusterAs) chainFeatureStruct(qNext, queriedClusterAs);
#endif
#ifdef VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME
                if (availLss) chainFeatureStruct(qNext, queriedLss);
#endif
#ifdef VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME
                if (availSer) chainFeatureStruct(qNext, queriedSer);
#endif

                queriedFeatures2.pNext = qNext;
                fn.vkGetPhysicalDeviceFeatures2(physDevice, &queriedFeatures2);
            }
            else
            {
                fn.vkGetPhysicalDeviceFeatures(physDevice, &queriedFeatures2.features);
            }

            VkPhysicalDeviceFeatures enabledCore{};
            const VkPhysicalDeviceFeatures& supportedCore = queriedFeatures2.features;
            enabledCore.samplerAnisotropy = supportedCore.samplerAnisotropy;
            enabledCore.fillModeNonSolid = supportedCore.fillModeNonSolid;
            enabledCore.depthClamp = supportedCore.depthClamp;
            enabledCore.depthBiasClamp = supportedCore.depthBiasClamp;
            enabledCore.independentBlend = supportedCore.independentBlend;
            enabledCore.fragmentStoresAndAtomics = supportedCore.fragmentStoresAndAtomics;
            enabledCore.vertexPipelineStoresAndAtomics = supportedCore.vertexPipelineStoresAndAtomics;
            enabledCore.multiDrawIndirect = supportedCore.multiDrawIndirect;
            enabledCore.geometryShader = supportedCore.geometryShader;
            enabledCore.tessellationShader = supportedCore.tessellationShader;
            enabledCore.shaderInt64 = supportedCore.shaderInt64;
            enabledCore.shaderFloat64 = supportedCore.shaderFloat64;
            enabledCore.imageCubeArray = supportedCore.imageCubeArray;
            enabledCore.textureCompressionBC = supportedCore.textureCompressionBC;

            VkPhysicalDeviceVulkan11Features enabledV11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
            VkPhysicalDeviceVulkan12Features enabledV12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
            VkPhysicalDeviceVulkan13Features enabledV13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
            VkPhysicalDeviceSynchronization2FeaturesKHR enabledSync2Khr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR };
            VkPhysicalDeviceDynamicRenderingFeaturesKHR enabledDynRenderKhr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
            VkPhysicalDeviceTimelineSemaphoreFeaturesKHR enabledTimelineKhr{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR };

            if (supportsVulkan12)
            {
                enabledV11.multiview = queriedV11.multiview;
                enabledV11.shaderDrawParameters = queriedV11.shaderDrawParameters;
                enabledV11.samplerYcbcrConversion = queriedV11.samplerYcbcrConversion;

                enabledV12.timelineSemaphore = queriedV12.timelineSemaphore;
                enabledV12.bufferDeviceAddress = queriedV12.bufferDeviceAddress;
                enabledV12.descriptorIndexing = queriedV12.descriptorIndexing;
                enabledV12.descriptorBindingUniformBufferUpdateAfterBind = queriedV12.descriptorBindingUniformBufferUpdateAfterBind;
                enabledV12.descriptorBindingSampledImageUpdateAfterBind = queriedV12.descriptorBindingSampledImageUpdateAfterBind;
                enabledV12.descriptorBindingStorageImageUpdateAfterBind = queriedV12.descriptorBindingStorageImageUpdateAfterBind;
                enabledV12.descriptorBindingStorageBufferUpdateAfterBind = queriedV12.descriptorBindingStorageBufferUpdateAfterBind;
                enabledV12.descriptorBindingUniformTexelBufferUpdateAfterBind = queriedV12.descriptorBindingUniformTexelBufferUpdateAfterBind;
                enabledV12.descriptorBindingStorageTexelBufferUpdateAfterBind = queriedV12.descriptorBindingStorageTexelBufferUpdateAfterBind;
                enabledV12.descriptorBindingPartiallyBound = queriedV12.descriptorBindingPartiallyBound;
                enabledV12.descriptorBindingUpdateUnusedWhilePending = queriedV12.descriptorBindingUpdateUnusedWhilePending;
                enabledV12.runtimeDescriptorArray = queriedV12.runtimeDescriptorArray;
                enabledV12.shaderSampledImageArrayNonUniformIndexing = queriedV12.shaderSampledImageArrayNonUniformIndexing;
                enabledV12.shaderStorageBufferArrayNonUniformIndexing = queriedV12.shaderStorageBufferArrayNonUniformIndexing;
                enabledV12.drawIndirectCount = queriedV12.drawIndirectCount;
                enabledV12.samplerFilterMinmax = queriedV12.samplerFilterMinmax;
                enabledV12.scalarBlockLayout = queriedV12.scalarBlockLayout;
                enabledV12.hostQueryReset = queriedV12.hostQueryReset;
                enabledV12.imagelessFramebuffer = queriedV12.imagelessFramebuffer;
            }
            else if (availTimelineKhr && queriedTimelineKhr.timelineSemaphore == VK_TRUE)
            {
                enabledTimelineKhr.timelineSemaphore = VK_TRUE;
            }

            if (supportsVulkan13)
            {
                enabledV13.synchronization2 = queriedV13.synchronization2;
                enabledV13.dynamicRendering = queriedV13.dynamicRendering;
                enabledV13.maintenance4 = queriedV13.maintenance4;
            }
            else
            {
                if (availSync2Khr && queriedSync2Khr.synchronization2 == VK_TRUE)
                {
                    enabledSync2Khr.synchronization2 = VK_TRUE;
                }
                if (availDynRenderKhr && queriedDynRenderKhr.dynamicRendering == VK_TRUE)
                {
                    enabledDynRenderKhr.dynamicRendering = VK_TRUE;
                }
            }

            outSetup.enabledExtensions.clear();
            if (hasPortabilitySubset)
            {
                outSetup.enabledExtensions.push_back("VK_KHR_portability_subset");
            }
            if (availTimelineKhr && enabledTimelineKhr.timelineSemaphore == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            }
            if (availSync2Khr && enabledSync2Khr.synchronization2 == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
            }
            if (availDynRenderKhr && enabledDynRenderKhr.dynamicRendering == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            }
            if (availConservativeRast)
            {
                outSetup.enabledExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            }

            void* enabledNext = nullptr;
            if (supportsVulkan12)
            {
                chainFeatureStruct(enabledNext, enabledV11);
                chainFeatureStruct(enabledNext, enabledV12);
            }
            else if (enabledTimelineKhr.timelineSemaphore == VK_TRUE)
            {
                chainFeatureStruct(enabledNext, enabledTimelineKhr);
            }

            if (supportsVulkan13)
            {
                chainFeatureStruct(enabledNext, enabledV13);
            }
            else
            {
                if (enabledSync2Khr.synchronization2 == VK_TRUE)
                {
                    chainFeatureStruct(enabledNext, enabledSync2Khr);
                }
                if (enabledDynRenderKhr.dynamicRendering == VK_TRUE)
                {
                    chainFeatureStruct(enabledNext, enabledDynRenderKhr);
                }
            }

            const bool canUseRayTracingBase =
                availDeferredHostOps && availAccelStruct &&
                queriedAccelStruct.accelerationStructure == VK_TRUE &&
                supportsVulkan12 && enabledV12.bufferDeviceAddress == VK_TRUE;

            VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelStruct{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
            if (canUseRayTracingBase)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
                outSetup.enabledExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
                enabledAccelStruct.accelerationStructure = VK_TRUE;
                enabledAccelStruct.descriptorBindingAccelerationStructureUpdateAfterBind =
                    queriedAccelStruct.descriptorBindingAccelerationStructureUpdateAfterBind;
                chainFeatureStruct(enabledNext, enabledAccelStruct);
            }

            VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQuery{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
            if (canUseRayTracingBase && availRayQuery && queriedRayQuery.rayQuery == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
                enabledRayQuery.rayQuery = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledRayQuery);
            }

            VkPhysicalDeviceRayTracingPipelineFeaturesKHR enabledRtPipeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
            if (canUseRayTracingBase && availRtPipeline && queriedRtPipeline.rayTracingPipeline == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
                enabledRtPipeline.rayTracingPipeline = VK_TRUE;
                enabledRtPipeline.rayTraversalPrimitiveCulling = queriedRtPipeline.rayTraversalPrimitiveCulling;
                chainFeatureStruct(enabledNext, enabledRtPipeline);
            }

            VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR enabledRtPosFetch{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR };
            if (canUseRayTracingBase && availRtPosFetch && queriedRtPosFetch.rayTracingPositionFetch == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME);
                enabledRtPosFetch.rayTracingPositionFetch = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledRtPosFetch);
            }

            VkPhysicalDeviceFragmentShadingRateFeaturesKHR enabledVrs{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR };
            if (availVrs && hasFeatures2 &&
                (queriedVrs.pipelineFragmentShadingRate || queriedVrs.primitiveFragmentShadingRate || queriedVrs.attachmentFragmentShadingRate))
            {
                outSetup.enabledExtensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
                enabledVrs.pipelineFragmentShadingRate = queriedVrs.pipelineFragmentShadingRate;
                enabledVrs.primitiveFragmentShadingRate = queriedVrs.primitiveFragmentShadingRate;
                enabledVrs.attachmentFragmentShadingRate = queriedVrs.attachmentFragmentShadingRate;
                chainFeatureStruct(enabledNext, enabledVrs);
            }

            VkPhysicalDeviceMeshShaderFeaturesEXT enabledMeshShader{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
            if (availMeshShader && supportsVulkan12 &&
                (queriedMeshShader.taskShader == VK_TRUE || queriedMeshShader.meshShader == VK_TRUE))
            {
                outSetup.enabledExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
                enabledMeshShader.taskShader = queriedMeshShader.taskShader;
                enabledMeshShader.meshShader = queriedMeshShader.meshShader;
                enabledMeshShader.multiviewMeshShader = queriedMeshShader.multiviewMeshShader;
                enabledMeshShader.primitiveFragmentShadingRateMeshShader = queriedMeshShader.primitiveFragmentShadingRateMeshShader;
                enabledMeshShader.meshShaderQueries = queriedMeshShader.meshShaderQueries;
                chainFeatureStruct(enabledNext, enabledMeshShader);
            }

            VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT enabledMutableDesc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
            if (availMutableDesc && hasFeatures2 && queriedMutableDesc.mutableDescriptorType == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
                enabledMutableDesc.mutableDescriptorType = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledMutableDesc);
            }

            VkPhysicalDeviceDepthClipEnableFeaturesEXT enabledDepthClip{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT };
            if (availDepthClip && hasFeatures2 && queriedDepthClip.depthClipEnable == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
                enabledDepthClip.depthClipEnable = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledDepthClip);
            }

            VkPhysicalDeviceOpacityMicromapFeaturesEXT enabledOmm{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT };
            if (canUseRayTracingBase && availOmm && queriedOmm.micromap == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
                enabledOmm.micromap = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledOmm);
            }

#ifdef VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME
            VkPhysicalDeviceCooperativeVectorFeaturesNV enabledCoopVec{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV };
            if (availCoopVec && hasFeatures2 && queriedCoopVec.cooperativeVector == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
                enabledCoopVec.cooperativeVector = VK_TRUE;
                enabledCoopVec.cooperativeVectorTraining = queriedCoopVec.cooperativeVectorTraining;
                chainFeatureStruct(enabledNext, enabledCoopVec);
            }
#endif

#ifdef VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME
            VkPhysicalDeviceClusterAccelerationStructureFeaturesNV enabledClusterAs{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV };
            if (canUseRayTracingBase && availClusterAs && queriedClusterAs.clusterAccelerationStructure == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
                enabledClusterAs.clusterAccelerationStructure = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledClusterAs);
            }
#endif

#ifdef VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME
            VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV enabledLss{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV };
            if (canUseRayTracingBase && availLss && (queriedLss.spheres == VK_TRUE || queriedLss.linearSweptSpheres == VK_TRUE))
            {
                outSetup.enabledExtensions.push_back(VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME);
                enabledLss.spheres = queriedLss.spheres;
                enabledLss.linearSweptSpheres = queriedLss.linearSweptSpheres;
                chainFeatureStruct(enabledNext, enabledLss);
            }
#endif

#ifdef VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME
            VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV enabledSer{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV };
            if (canUseRayTracingBase && availSer && queriedSer.rayTracingInvocationReorder == VK_TRUE)
            {
                outSetup.enabledExtensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
                enabledSer.rayTracingInvocationReorder = VK_TRUE;
                chainFeatureStruct(enabledNext, enabledSer);
            }
#endif

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::vector<std::vector<float>> queuePriorities;
            for (const auto& entry : outSetup.queues.familyQueueCounts)
            {
                if (entry.second == 0)
                {
                    continue;
                }
                queuePriorities.emplace_back(entry.second, 1.0f);
                VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
                qci.queueFamilyIndex = entry.first;
                qci.queueCount = entry.second;
                qci.pQueuePriorities = queuePriorities.back().data();
                queueCreateInfos.push_back(qci);
            }

            VkDeviceCreateInfo devInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
            devInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
            devInfo.pQueueCreateInfos = queueCreateInfos.data();
            devInfo.enabledExtensionCount = static_cast<uint32_t>(outSetup.enabledExtensions.size());
            devInfo.ppEnabledExtensionNames = outSetup.enabledExtensions.empty() ? nullptr : outSetup.enabledExtensions.data();

            VkPhysicalDeviceFeatures2 enabledFeatures2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            if (hasFeatures2 && enabledNext != nullptr)
            {
                enabledFeatures2.features = enabledCore;
                enabledFeatures2.pNext = enabledNext;
                devInfo.pNext = &enabledFeatures2;
            }
            else
            {
                devInfo.pEnabledFeatures = &enabledCore;
            }

            VkResult res = fn.vkCreateDevice(physDevice, &devInfo, nullptr, &outSetup.device);
            if (res != VK_SUCCESS || outSetup.device == VK_NULL_HANDLE)
            {
                std::ostringstream oss;
                oss << "vkCreateDevice failed with VkResult " << static_cast<int>(res);
                outError = oss.str();
                return false;
            }

            outSetup.timelineSemaphoreSupported =
                (enabledV12.timelineSemaphore == VK_TRUE) || (enabledTimelineKhr.timelineSemaphore == VK_TRUE);
            outSetup.bufferDeviceAddressSupported = (enabledV12.bufferDeviceAddress == VK_TRUE);
            outSetup.drawIndirectCount = (enabledV12.drawIndirectCount == VK_TRUE);
            outSetup.samplerFilterMinmax = (enabledV12.samplerFilterMinmax == VK_TRUE);
            outSetup.descriptorBindingUniformBufferUpdateAfterBind =
                (enabledV12.descriptorBindingUniformBufferUpdateAfterBind == VK_TRUE);
            return true;
        }

        void loadDeviceFunctions(VulkanFunctions& fn, VkDevice device)
        {
#define LOAD_DEV_FN(name) fn.name = reinterpret_cast<PFN_##name>(fn.vkGetDeviceProcAddr(device, #name))
            LOAD_DEV_FN(vkDestroyDevice);
            LOAD_DEV_FN(vkGetDeviceQueue);
            LOAD_DEV_FN(vkDeviceWaitIdle);
            LOAD_DEV_FN(vkCreateSemaphore);
            LOAD_DEV_FN(vkDestroySemaphore);
            LOAD_DEV_FN(vkSignalSemaphore);
            if (!fn.vkSignalSemaphore)
            {
                fn.vkSignalSemaphore = reinterpret_cast<PFN_vkSignalSemaphore>(
                    fn.vkGetDeviceProcAddr(device, "vkSignalSemaphoreKHR"));
            }
            LOAD_DEV_FN(vkGetSemaphoreCounterValue);
            if (!fn.vkGetSemaphoreCounterValue)
            {
                fn.vkGetSemaphoreCounterValue = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
                    fn.vkGetDeviceProcAddr(device, "vkGetSemaphoreCounterValueKHR"));
            }
            LOAD_DEV_FN(vkCreateBuffer);
            LOAD_DEV_FN(vkDestroyBuffer);
            LOAD_DEV_FN(vkGetBufferMemoryRequirements);
            LOAD_DEV_FN(vkBindBufferMemory);
            LOAD_DEV_FN(vkCreateImage);
            LOAD_DEV_FN(vkDestroyImage);
            LOAD_DEV_FN(vkGetImageMemoryRequirements);
            LOAD_DEV_FN(vkBindImageMemory);
            LOAD_DEV_FN(vkAllocateMemory);
            LOAD_DEV_FN(vkFreeMemory);
            LOAD_DEV_FN(vkMapMemory);
            LOAD_DEV_FN(vkUnmapMemory);
            LOAD_DEV_FN(vkCreateCommandPool);
            LOAD_DEV_FN(vkDestroyCommandPool);
            LOAD_DEV_FN(vkAllocateCommandBuffers);
            LOAD_DEV_FN(vkBeginCommandBuffer);
            LOAD_DEV_FN(vkEndCommandBuffer);
            LOAD_DEV_FN(vkCmdPipelineBarrier);
            LOAD_DEV_FN(vkQueueSubmit);
            LOAD_DEV_FN(vkQueueWaitIdle);
#undef LOAD_DEV_FN
        }

        uint32_t findMemoryType(
            const VkPhysicalDeviceMemoryProperties& memProps,
            uint32_t typeBits,
            VkMemoryPropertyFlags requiredFlags)
        {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            {
                if ((typeBits & (1u << i)) != 0 &&
                    (memProps.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags)
                {
                    return i;
                }
            }
            return UINT32_MAX;
        }

        class VulkanGpuGate final : public IGpuGate
        {
        public:
            VulkanGpuGate(
                nvrhi::IDevice* rawDevice,
                VkDevice vkDevice,
                const VulkanFunctions& fn,
                bool hasComputeQueue,
                bool hasCopyQueue)
                : m_RawDevice(rawDevice)
                , m_VkDevice(vkDevice)
                , m_Fn(fn)
                , m_HasComputeQueue(hasComputeQueue)
                , m_HasCopyQueue(hasCopyQueue)
            {
                createSemaphore();
            }

            ~VulkanGpuGate() override
            {
                destroySemaphore();
            }

            bool isValid() const
            {
                return m_Semaphore != VK_NULL_HANDLE;
            }

            bool enqueueWait(nvrhi::CommandQueue queue) override
            {
                if (!m_RawDevice || m_Semaphore == VK_NULL_HANDLE)
                {
                    return false;
                }
                if (queue == nvrhi::CommandQueue::Compute && !m_HasComputeQueue)
                {
                    return false;
                }
                if (queue == nvrhi::CommandQueue::Copy && !m_HasCopyQueue)
                {
                    return false;
                }
                auto* vkDev = static_cast<nvrhi::vulkan::IDevice*>(
                    m_RawDevice->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
                if (!vkDev)
                {
                    return false;
                }
                vkDev->queueWaitForSemaphore(queue, m_Semaphore, 1);
                return true;
            }

            void signal() override
            {
                if (m_Semaphore == VK_NULL_HANDLE || !m_Fn.vkSignalSemaphore)
                {
                    return;
                }
                VkSemaphoreSignalInfo signalInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO };
                signalInfo.semaphore = m_Semaphore;
                signalInfo.value = 1;
                m_Fn.vkSignalSemaphore(m_VkDevice, &signalInfo);
            }

            void reset() override
            {
                if (m_RawDevice)
                {
                    m_RawDevice->waitForIdle();
                }
                destroySemaphore();
                createSemaphore();
            }

        private:
            void createSemaphore()
            {
                if (!m_Fn.vkCreateSemaphore)
                {
                    return;
                }
                VkSemaphoreTypeCreateInfo typeInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
                typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                typeInfo.initialValue = 0;

                VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
                semInfo.pNext = &typeInfo;
                if (m_Fn.vkCreateSemaphore(m_VkDevice, &semInfo, nullptr, &m_Semaphore) != VK_SUCCESS)
                {
                    m_Semaphore = VK_NULL_HANDLE;
                }
            }

            void destroySemaphore()
            {
                if (m_Semaphore != VK_NULL_HANDLE && m_Fn.vkDestroySemaphore)
                {
                    m_Fn.vkDestroySemaphore(m_VkDevice, m_Semaphore, nullptr);
                    m_Semaphore = VK_NULL_HANDLE;
                }
            }

            nvrhi::IDevice* m_RawDevice = nullptr;
            VkDevice m_VkDevice = VK_NULL_HANDLE;
            VulkanFunctions m_Fn{};
            bool m_HasComputeQueue = false;
            bool m_HasCopyQueue = false;
            VkSemaphore m_Semaphore = VK_NULL_HANDLE;
        };

        class VulkanBackendContext final : public IBackendContext
        {
        public:
            VulkanBackendContext() = default;

            ~VulkanBackendContext() override
            {
                if (m_NvrhiDevice)
                {
                    m_NvrhiDevice->waitForIdle();
                    m_NvrhiDevice->runGarbageCollection();
                    m_NvrhiDevice = nullptr;
                }
                if (m_VkDevice != VK_NULL_HANDLE && m_Fn.vkDestroyDevice)
                {
                    m_Fn.vkDeviceWaitIdle(m_VkDevice);
                    m_Fn.vkDestroyDevice(m_VkDevice, nullptr);
                    m_VkDevice = VK_NULL_HANDLE;
                }
                if (m_OwnsInstance)
                {
                    if (m_DebugMessenger != VK_NULL_HANDLE && m_Fn.vkDestroyDebugUtilsMessengerEXT)
                    {
                        m_Fn.vkDestroyDebugUtilsMessengerEXT(m_Instance, m_DebugMessenger, nullptr);
                        m_DebugMessenger = VK_NULL_HANDLE;
                    }
                    if (m_Instance != VK_NULL_HANDLE && m_Fn.vkDestroyInstance)
                    {
                        m_Fn.vkDestroyInstance(m_Instance, nullptr);
                        m_Instance = VK_NULL_HANDLE;
                    }
                    m_Loader.close();
                }
                else
                {
#if !defined(UAA_NVRHI_SHARED) || !UAA_NVRHI_SHARED
                    if (m_PrimaryVkDevice != VK_NULL_HANDLE && m_Instance != VK_NULL_HANDLE && m_Fn.vkGetInstanceProcAddr)
                    {
                        VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Instance, m_Fn.vkGetInstanceProcAddr, m_PrimaryVkDevice);
                    }
#endif
                }
            }

            nvrhi::IDevice* getRawDevice() override
            {
                return m_NvrhiDevice.Get();
            }

            const DeviceInfo& getDeviceInfo() const override
            {
                return m_DeviceInfo;
            }

            std::unique_ptr<IGpuGate> createGpuGate() override
            {
                if (!m_TimelineSemaphoreSupported || !m_Fn.vkCreateSemaphore || !m_Fn.vkSignalSemaphore)
                {
                    return nullptr;
                }
                auto gate = std::make_unique<VulkanGpuGate>(
                    m_NvrhiDevice.Get(),
                    m_VkDevice,
                    m_Fn,
                    m_HasComputeQueue,
                    m_HasCopyQueue);
                if (!gate->isValid())
                {
                    return nullptr;
                }
                return gate;
            }

            bool createNativeImportBuffer(uint64_t byteSize, const void* initialData, NativeBufferImport& out) override
            {
                if (!m_Fn.vkCreateBuffer || !m_Fn.vkAllocateMemory || !m_Fn.vkBindBufferMemory)
                {
                    return false;
                }

                VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bufInfo.size = byteSize > 0 ? byteSize : 4;
                bufInfo.usage =
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
                if (m_BufferDeviceAddressSupported)
                {
                    bufInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                }
                bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkBuffer buffer = VK_NULL_HANDLE;
                if (m_Fn.vkCreateBuffer(m_VkDevice, &bufInfo, nullptr, &buffer) != VK_SUCCESS)
                {
                    return false;
                }

                VkMemoryRequirements memReq{};
                m_Fn.vkGetBufferMemoryRequirements(m_VkDevice, buffer, &memReq);

                VkPhysicalDeviceMemoryProperties memProps{};
                m_Fn.vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevice, &memProps);
                const uint32_t memType = findMemoryType(
                    memProps,
                    memReq.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (memType == UINT32_MAX)
                {
                    m_Fn.vkDestroyBuffer(m_VkDevice, buffer, nullptr);
                    return false;
                }

                VkMemoryAllocateFlagsInfo allocFlagsInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
                allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

                VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = memType;
                if (m_BufferDeviceAddressSupported)
                {
                    allocInfo.pNext = &allocFlagsInfo;
                }

                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (m_Fn.vkAllocateMemory(m_VkDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyBuffer(m_VkDevice, buffer, nullptr);
                    return false;
                }

                if (m_Fn.vkBindBufferMemory(m_VkDevice, buffer, memory, 0) != VK_SUCCESS)
                {
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyBuffer(m_VkDevice, buffer, nullptr);
                    return false;
                }

                if (initialData && byteSize > 0)
                {
                    void* mapped = nullptr;
                    if (m_Fn.vkMapMemory(m_VkDevice, memory, 0, byteSize, 0, &mapped) != VK_SUCCESS || !mapped)
                    {
                        m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                        m_Fn.vkDestroyBuffer(m_VkDevice, buffer, nullptr);
                        return false;
                    }
                    std::memcpy(mapped, initialData, static_cast<size_t>(byteSize));
                    m_Fn.vkUnmapMemory(m_VkDevice, memory);
                }

                VkDevice vkDev = m_VkDevice;
                PFN_vkDestroyBuffer destroyBufFn = m_Fn.vkDestroyBuffer;
                PFN_vkFreeMemory freeMemFn = m_Fn.vkFreeMemory;

                out.objectType = nvrhi::ObjectTypes::VK_Buffer;
                out.handle = vkHandleToObject(buffer);
                out.byteSize = byteSize;
                out.release = [vkDev, buffer, memory, destroyBufFn, freeMemFn]()
                {
                    if (buffer != VK_NULL_HANDLE && destroyBufFn)
                    {
                        destroyBufFn(vkDev, buffer, nullptr);
                    }
                    if (memory != VK_NULL_HANDLE && freeMemFn)
                    {
                        freeMemFn(vkDev, memory, nullptr);
                    }
                };
                return true;
            }

            bool createNativeImportTexture(uint32_t width, uint32_t height, nvrhi::Format format, NativeTextureImport& out) override
            {
                if (!m_Fn.vkCreateImage || !m_Fn.vkAllocateMemory || !m_Fn.vkBindImageMemory ||
                    !m_Fn.vkCreateCommandPool || !m_Fn.vkDestroyCommandPool ||
                    !m_Fn.vkAllocateCommandBuffers || !m_Fn.vkBeginCommandBuffer ||
                    !m_Fn.vkEndCommandBuffer || !m_Fn.vkCmdPipelineBarrier ||
                    !m_Fn.vkQueueSubmit || !m_Fn.vkQueueWaitIdle ||
                    m_GraphicsQueue == VK_NULL_HANDLE)
                {
                    return false;
                }

                const VkFormat vkFormat = nvrhi::vulkan::convertFormat(format);
                if (vkFormat == VK_FORMAT_UNDEFINED)
                {
                    return false;
                }

                VkFormatProperties fmtProps{};
                m_Fn.vkGetPhysicalDeviceFormatProperties(m_PhysicalDevice, vkFormat, &fmtProps);

                VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                if (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                {
                    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                }
                if (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                {
                    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                }
                if (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
                {
                    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                }
                else if (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                {
                    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                }

                VkImageCreateInfo imgInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
                imgInfo.imageType = VK_IMAGE_TYPE_2D;
                imgInfo.format = vkFormat;
                imgInfo.extent = VkExtent3D{ width, height, 1 };
                imgInfo.mipLevels = 1;
                imgInfo.arrayLayers = 1;
                imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                imgInfo.usage = usage;
                imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                VkImage image = VK_NULL_HANDLE;
                if (m_Fn.vkCreateImage(m_VkDevice, &imgInfo, nullptr, &image) != VK_SUCCESS)
                {
                    return false;
                }

                VkMemoryRequirements memReq{};
                m_Fn.vkGetImageMemoryRequirements(m_VkDevice, image, &memReq);

                VkPhysicalDeviceMemoryProperties memProps{};
                m_Fn.vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevice, &memProps);
                uint32_t memType = findMemoryType(memProps, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (memType == UINT32_MAX)
                {
                    memType = findMemoryType(memProps, memReq.memoryTypeBits, 0);
                }
                if (memType == UINT32_MAX)
                {
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                allocInfo.allocationSize = memReq.size;
                allocInfo.memoryTypeIndex = memType;

                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (m_Fn.vkAllocateMemory(m_VkDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                if (m_Fn.vkBindImageMemory(m_VkDevice, image, memory, 0) != VK_SUCCESS)
                {
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                // Transition out of VK_IMAGE_LAYOUT_UNDEFINED so NVRHI does not attempt a keepInitialState barrier back to eUndefined.
                VkImageLayout targetLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkAccessFlags dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                nvrhi::ResourceStates initialState = nvrhi::ResourceStates::ShaderResource;

                if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0)
                {
                    if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
                    {
                        targetLayout = VK_IMAGE_LAYOUT_GENERAL;
                        dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                        initialState = nvrhi::ResourceStates::UnorderedAccess;
                    }
                    else
                    {
                        targetLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
                        initialState = nvrhi::ResourceStates::CopyDest;
                    }
                }

                const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(format);
                VkImageAspectFlags aspectMask = 0;
                if (fmtInfo.hasDepth)
                {
                    aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
                }
                if (fmtInfo.hasStencil)
                {
                    aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
                if (aspectMask == 0)
                {
                    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                }

                VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
                poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                poolInfo.queueFamilyIndex = m_GraphicsQueueFamilyIndex;

                VkCommandPool cmdPool = VK_NULL_HANDLE;
                if (m_Fn.vkCreateCommandPool(m_VkDevice, &poolInfo, nullptr, &cmdPool) != VK_SUCCESS)
                {
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                cmdAllocInfo.commandPool = cmdPool;
                cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cmdAllocInfo.commandBufferCount = 1;

                VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
                if (m_Fn.vkAllocateCommandBuffers(m_VkDevice, &cmdAllocInfo, &cmdBuf) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyCommandPool(m_VkDevice, cmdPool, nullptr);
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (m_Fn.vkBeginCommandBuffer(cmdBuf, &beginInfo) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyCommandPool(m_VkDevice, cmdPool, nullptr);
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = dstAccessMask;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = targetLayout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange.aspectMask = aspectMask;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;

                m_Fn.vkCmdPipelineBarrier(
                    cmdBuf,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    dstStageMask,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier);

                if (m_Fn.vkEndCommandBuffer(cmdBuf) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyCommandPool(m_VkDevice, cmdPool, nullptr);
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &cmdBuf;
                if (m_Fn.vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
                    m_Fn.vkQueueWaitIdle(m_GraphicsQueue) != VK_SUCCESS)
                {
                    m_Fn.vkDestroyCommandPool(m_VkDevice, cmdPool, nullptr);
                    m_Fn.vkFreeMemory(m_VkDevice, memory, nullptr);
                    m_Fn.vkDestroyImage(m_VkDevice, image, nullptr);
                    return false;
                }

                m_Fn.vkDestroyCommandPool(m_VkDevice, cmdPool, nullptr);

                nvrhi::TextureDesc desc;
                desc.width = width;
                desc.height = height;
                desc.depth = 1;
                desc.arraySize = 1;
                desc.mipLevels = 1;
                desc.sampleCount = 1;
                desc.format = format;
                desc.dimension = nvrhi::TextureDimension::Texture2D;
                desc.isShaderResource = (usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
                desc.isRenderTarget = (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0;
                desc.isUAV = (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
                desc.initialState = initialState;
                desc.keepInitialState = true;
                desc.debugName = "uaa_native_import_vk_tex";

                VkDevice vkDev = m_VkDevice;
                PFN_vkDestroyImage destroyImgFn = m_Fn.vkDestroyImage;
                PFN_vkFreeMemory freeMemFn = m_Fn.vkFreeMemory;

                out.objectType = nvrhi::ObjectTypes::VK_Image;
                out.handle = vkHandleToObject(image);
                out.desc = desc;
                out.release = [vkDev, image, memory, destroyImgFn, freeMemFn]()
                {
                    if (image != VK_NULL_HANDLE && destroyImgFn)
                    {
                        destroyImgFn(vkDev, image, nullptr);
                    }
                    if (memory != VK_NULL_HANDLE && freeMemFn)
                    {
                        freeMemFn(vkDev, memory, nullptr);
                    }
                };
                return true;
            }

            std::unique_ptr<IBackendContext> createSecondaryDevice(nvrhi::IMessageCallback* cb) override
            {
                if (m_Instance == VK_NULL_HANDLE || m_PhysicalDevice == VK_NULL_HANDLE)
                {
                    return nullptr;
                }

                VulkanLogicalDeviceSetup setup;
                std::string err;
                if (!createLogicalVulkanDevice(
                        m_Fn,
                        m_PhysicalDevice,
                        m_PhysicalProps,
                        m_InstanceApiVersion,
                        setup,
                        err))
                {
                    if (cb)
                    {
                        cb->message(nvrhi::MessageSeverity::Error, err.c_str());
                    }
                    return nullptr;
                }

                auto sec = std::make_unique<VulkanBackendContext>();
                sec->m_OwnsInstance = false;
                sec->m_Instance = m_Instance;
                sec->m_PhysicalDevice = m_PhysicalDevice;
                sec->m_PhysicalProps = m_PhysicalProps;
                sec->m_InstanceApiVersion = m_InstanceApiVersion;
                sec->m_VkDevice = setup.device;
                sec->m_PrimaryVkDevice = m_VkDevice;
                sec->m_Fn = m_Fn;
                sec->m_LoaderPath = m_LoaderPath;
                sec->m_TimelineSemaphoreSupported = setup.timelineSemaphoreSupported;
                sec->m_BufferDeviceAddressSupported = setup.bufferDeviceAddressSupported;
                loadDeviceFunctions(sec->m_Fn, setup.device);

                nvrhi::vulkan::DeviceDesc desc{};
                desc.errorCB = cb;
                desc.instance = m_Instance;
                desc.physicalDevice = m_PhysicalDevice;
                desc.device = setup.device;
                desc.vulkanLibraryName = m_LoaderPath;
                sec->m_GraphicsQueueFamilyIndex = static_cast<uint32_t>(setup.queues.graphicsFamily);
                desc.graphicsQueueIndex = setup.queues.graphicsFamily;
                sec->m_Fn.vkGetDeviceQueue(
                    setup.device,
                    static_cast<uint32_t>(setup.queues.graphicsFamily),
                    static_cast<uint32_t>(setup.queues.graphicsIndex),
                    &desc.graphicsQueue);
                sec->m_GraphicsQueue = desc.graphicsQueue;

                if (setup.queues.computeFamily >= 0)
                {
                    desc.computeQueueIndex = setup.queues.computeFamily;
                    sec->m_Fn.vkGetDeviceQueue(
                        setup.device,
                        static_cast<uint32_t>(setup.queues.computeFamily),
                        static_cast<uint32_t>(setup.queues.computeIndex),
                        &desc.computeQueue);
                    sec->m_HasComputeQueue = (desc.computeQueue != VK_NULL_HANDLE);
                }

                if (setup.queues.transferFamily >= 0)
                {
                    desc.transferQueueIndex = setup.queues.transferFamily;
                    sec->m_Fn.vkGetDeviceQueue(
                        setup.device,
                        static_cast<uint32_t>(setup.queues.transferFamily),
                        static_cast<uint32_t>(setup.queues.transferIndex),
                        &desc.transferQueue);
                    sec->m_HasCopyQueue = (desc.transferQueue != VK_NULL_HANDLE);
                }

                std::vector<const char*> instExtPtrs;
                instExtPtrs.reserve(m_InstanceExtensionNames.size());
                for (const auto& s : m_InstanceExtensionNames)
                {
                    instExtPtrs.push_back(s.c_str());
                }

                desc.instanceExtensions = instExtPtrs.empty() ? nullptr : instExtPtrs.data();
                desc.numInstanceExtensions = instExtPtrs.size();
                desc.deviceExtensions = setup.enabledExtensions.empty() ? nullptr : setup.enabledExtensions.data();
                desc.numDeviceExtensions = setup.enabledExtensions.size();
                desc.bufferDeviceAddressSupported = setup.bufferDeviceAddressSupported;
                desc.descriptorBindingUniformBufferUpdateAfterBind = setup.descriptorBindingUniformBufferUpdateAfterBind;

#if !defined(UAA_NVRHI_SHARED) || !UAA_NVRHI_SHARED
                VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Instance, m_Fn.vkGetInstanceProcAddr, setup.device);
#endif

                sec->m_NvrhiDevice = nvrhi::vulkan::createDevice(desc);
                if (!sec->m_NvrhiDevice)
                {
                    return nullptr;
                }

                sec->m_DeviceInfo = m_DeviceInfo;
                sec->m_DeviceInfo.nativeValidation = false;
                sec->m_InstanceExtensionNames = m_InstanceExtensionNames;
                return sec;
            }

            bool hasNativeValidationErrors(std::vector<std::string>& outErrors) override
            {
                if (!m_ValidationState)
                {
                    return false;
                }
                std::lock_guard<std::mutex> lock(m_ValidationState->mutex);
                outErrors = m_ValidationState->errors;
                return !outErrors.empty();
            }

            void clearNativeValidationMessages() override
            {
                if (!m_ValidationState)
                {
                    return;
                }
                std::lock_guard<std::mutex> lock(m_ValidationState->mutex);
                m_ValidationState->errors.clear();
            }

            uint64_t getMappedBufferFlushAlignment() override
            {
                if (!m_NvrhiDevice)
                {
                    return 0;
                }
                auto* vkDev = static_cast<nvrhi::vulkan::IDevice*>(
                    m_NvrhiDevice->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
                return vkDev ? vkDev->getMappedBufferFlushAlignment() : 0;
            }

            bool flushMappedBuffer(nvrhi::IDevice* activeDevice, nvrhi::IBuffer* buffer, uint64_t offset, uint64_t size) override
            {
                nvrhi::IDevice* queryDev = activeDevice ? activeDevice : m_NvrhiDevice.Get();
                if (!queryDev)
                {
                    return false;
                }
                auto* vkDev = static_cast<nvrhi::vulkan::IDevice*>(
                    queryDev->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
                return vkDev ? vkDev->flushMappedBuffer(buffer, offset, size) : false;
            }

            bool init(
                const BackendConfig& config,
                nvrhi::IMessageCallback* messageCallback,
                std::string& outSkipOrError,
                bool& outIsUnavailable)
            {
                outSkipOrError.clear();
                outIsUnavailable = false;

                std::vector<std::string> candidateLibs;
#ifdef _WIN32
                candidateLibs.emplace_back("vulkan-1.dll");
#elif defined(__APPLE__)
                candidateLibs.emplace_back("libvulkan.1.dylib");
                candidateLibs.emplace_back("libvulkan.dylib");
                if (const char* sdkEnv = std::getenv("VULKAN_SDK"))
                {
                    if (sdkEnv[0] != '\0')
                    {
                        candidateLibs.emplace_back(std::string(sdkEnv) + "/lib/libvulkan.1.dylib");
                    }
                }
                candidateLibs.emplace_back("/Users/hypernewbie/VulkanSDK/macOS/lib/libvulkan.1.dylib");
                candidateLibs.emplace_back("/usr/local/lib/libvulkan.1.dylib");
                candidateLibs.emplace_back("/opt/homebrew/lib/libvulkan.1.dylib");
#else
                candidateLibs.emplace_back("libvulkan.so.1");
                candidateLibs.emplace_back("libvulkan.so");
#endif

                bool loaded = false;
                for (const std::string& candidate : candidateLibs)
                {
                    if (m_Loader.open(candidate.c_str()))
                    {
                        m_Fn.vkGetInstanceProcAddr = m_Loader.loadSymbol<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
                        if (m_Fn.vkGetInstanceProcAddr)
                        {
                            m_LoaderPath = candidate;
                            loaded = true;
                            break;
                        }
                        m_Loader.close();
                    }
                }

                if (!loaded)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "Vulkan loader library not found";
                    return false;
                }

                m_Fn.vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                    m_Fn.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
                m_Fn.vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                    m_Fn.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties"));
                m_Fn.vkEnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
                    m_Fn.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties"));
                m_Fn.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
                    m_Fn.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));

                if (!m_Fn.vkEnumerateInstanceExtensionProperties || !m_Fn.vkCreateInstance)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "Vulkan loader missing required global entry points";
                    return false;
                }

                uint32_t loaderVersion = VK_API_VERSION_1_0;
                if (m_Fn.vkEnumerateInstanceVersion)
                {
                    m_Fn.vkEnumerateInstanceVersion(&loaderVersion);
                }
                m_InstanceApiVersion = std::min(loaderVersion, VK_API_VERSION_1_3);

                uint32_t instExtCount = 0;
                m_Fn.vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, nullptr);
                std::vector<VkExtensionProperties> availInstExts(instExtCount);
                if (instExtCount > 0)
                {
                    m_Fn.vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, availInstExts.data());
                }

                uint32_t instLayerCount = 0;
                if (m_Fn.vkEnumerateInstanceLayerProperties)
                {
                    m_Fn.vkEnumerateInstanceLayerProperties(&instLayerCount, nullptr);
                }
                std::vector<VkLayerProperties> availLayers(instLayerCount);
                if (instLayerCount > 0 && m_Fn.vkEnumerateInstanceLayerProperties)
                {
                    m_Fn.vkEnumerateInstanceLayerProperties(&instLayerCount, availLayers.data());
                }

                std::vector<const char*> enabledInstExts;
                std::vector<const char*> enabledLayers;
                VkInstanceCreateFlags instFlags = 0;

                if (hasExtension(availInstExts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
                {
                    enabledInstExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
                    instFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
                }
                if (hasExtension(availInstExts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
                {
                    enabledInstExts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
                }
                const bool hasDebugUtils = hasExtension(availInstExts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                if (hasDebugUtils)
                {
                    enabledInstExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                }

                bool validationLayerEnabled = false;
                if (config.nativeValidation)
                {
                    if (hasLayer(availLayers, "VK_LAYER_KHRONOS_validation"))
                    {
                        enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
                        validationLayerEnabled = true;
                    }
                }

                VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
                appInfo.pApplicationName = "nvrhi_uaa";
                appInfo.applicationVersion = 1;
                appInfo.pEngineName = "nvrhi";
                appInfo.engineVersion = 1;
                appInfo.apiVersion = m_InstanceApiVersion;

                VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
                instInfo.flags = instFlags;
                instInfo.pApplicationInfo = &appInfo;
                instInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
                instInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();
                instInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstExts.size());
                instInfo.ppEnabledExtensionNames = enabledInstExts.empty() ? nullptr : enabledInstExts.data();

                VkDebugUtilsMessengerCreateInfoEXT debugInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
                if (config.nativeValidation && hasDebugUtils)
                {
                    m_ValidationState = std::make_unique<VulkanValidationState>();
                    debugInfo.messageSeverity =
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
                    debugInfo.messageType =
                        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                    debugInfo.pfnUserCallback = vulkanDebugUtilsCallback;
                    debugInfo.pUserData = m_ValidationState.get();
                    instInfo.pNext = &debugInfo;
                }

                VkResult instRes = m_Fn.vkCreateInstance(&instInfo, nullptr, &m_Instance);
                if (instRes != VK_SUCCESS || m_Instance == VK_NULL_HANDLE)
                {
                    outIsUnavailable = true;
                    std::ostringstream oss;
                    oss << "vkCreateInstance failed with VkResult " << static_cast<int>(instRes);
                    outSkipOrError = oss.str();
                    return false;
                }

#define LOAD_INST_FN(name) m_Fn.name = reinterpret_cast<PFN_##name>(m_Fn.vkGetInstanceProcAddr(m_Instance, #name))
                LOAD_INST_FN(vkDestroyInstance);
                LOAD_INST_FN(vkEnumeratePhysicalDevices);
                LOAD_INST_FN(vkGetPhysicalDeviceProperties);
                LOAD_INST_FN(vkGetPhysicalDeviceProperties2);
                if (!m_Fn.vkGetPhysicalDeviceProperties2)
                {
                    m_Fn.vkGetPhysicalDeviceProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                        m_Fn.vkGetInstanceProcAddr(m_Instance, "vkGetPhysicalDeviceProperties2KHR"));
                }
                LOAD_INST_FN(vkGetPhysicalDeviceFeatures);
                LOAD_INST_FN(vkGetPhysicalDeviceFeatures2);
                if (!m_Fn.vkGetPhysicalDeviceFeatures2)
                {
                    m_Fn.vkGetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                        m_Fn.vkGetInstanceProcAddr(m_Instance, "vkGetPhysicalDeviceFeatures2KHR"));
                }
                LOAD_INST_FN(vkGetPhysicalDeviceQueueFamilyProperties);
                LOAD_INST_FN(vkGetPhysicalDeviceMemoryProperties);
                LOAD_INST_FN(vkGetPhysicalDeviceFormatProperties);
                LOAD_INST_FN(vkEnumerateDeviceExtensionProperties);
                LOAD_INST_FN(vkCreateDevice);
                LOAD_INST_FN(vkGetDeviceProcAddr);
                LOAD_INST_FN(vkCreateDebugUtilsMessengerEXT);
                LOAD_INST_FN(vkDestroyDebugUtilsMessengerEXT);
#undef LOAD_INST_FN

                if (config.nativeValidation && hasDebugUtils && m_Fn.vkCreateDebugUtilsMessengerEXT)
                {
                    if (m_Fn.vkCreateDebugUtilsMessengerEXT(m_Instance, &debugInfo, nullptr, &m_DebugMessenger) != VK_SUCCESS)
                    {
                        m_DebugMessenger = VK_NULL_HANDLE;
                    }
                }

                uint32_t physCount = 0;
                if (m_Fn.vkEnumeratePhysicalDevices(m_Instance, &physCount, nullptr) != VK_SUCCESS || physCount == 0)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "No Vulkan physical devices available";
                    return false;
                }

                std::vector<VkPhysicalDevice> physDevices(physCount);
                if (m_Fn.vkEnumeratePhysicalDevices(m_Instance, &physCount, physDevices.data()) != VK_SUCCESS)
                {
                    outIsUnavailable = false;
                    outSkipOrError = "Failed to enumerate Vulkan physical devices";
                    return false;
                }

                struct CandidateDevice
                {
                    VkPhysicalDevice handle = VK_NULL_HANDLE;
                    VkPhysicalDeviceProperties props{};
                    std::string driverName;
                    std::string driverInfo;
                };

                std::vector<CandidateDevice> candidates(physCount);
                for (uint32_t i = 0; i < physCount; ++i)
                {
                    candidates[i].handle = physDevices[i];
                    m_Fn.vkGetPhysicalDeviceProperties(physDevices[i], &candidates[i].props);

                    uint32_t devExtCount = 0;
                    m_Fn.vkEnumerateDeviceExtensionProperties(physDevices[i], nullptr, &devExtCount, nullptr);
                    std::vector<VkExtensionProperties> devExts(devExtCount);
                    if (devExtCount > 0)
                    {
                        m_Fn.vkEnumerateDeviceExtensionProperties(physDevices[i], nullptr, &devExtCount, devExts.data());
                    }

                    const bool hasDriverProps =
                        (candidates[i].props.apiVersion >= VK_API_VERSION_1_2) ||
                        hasExtension(devExts, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);

                    if (hasDriverProps && m_Fn.vkGetPhysicalDeviceProperties2 && m_InstanceApiVersion >= VK_API_VERSION_1_1)
                    {
                        VkPhysicalDeviceDriverProperties driverProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
                        VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
                        props2.pNext = &driverProps;
                        m_Fn.vkGetPhysicalDeviceProperties2(physDevices[i], &props2);
                        candidates[i].driverName = driverProps.driverName;
                        candidates[i].driverInfo = driverProps.driverInfo;
                    }
                }

                size_t selectedIndex = 0;
                bool matched = false;
                if (!config.deviceSelector.empty())
                {
                    size_t parsedIdx = 0;
                    if (tryParseIndex(config.deviceSelector, parsedIdx))
                    {
                        if (parsedIdx < candidates.size())
                        {
                            selectedIndex = parsedIdx;
                            matched = true;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < candidates.size(); ++i)
                        {
                            if (containsCaseInsensitive(candidates[i].props.deviceName, config.deviceSelector) ||
                                containsCaseInsensitive(candidates[i].driverName, config.deviceSelector))
                            {
                                selectedIndex = i;
                                matched = true;
                                break;
                            }
                        }
                    }

                    if (!matched)
                    {
                        outIsUnavailable = true;
                        outSkipOrError = "No Vulkan physical device matching selector '" + config.deviceSelector + "'";
                        return false;
                    }
                }

                m_PhysicalDevice = candidates[selectedIndex].handle;
                m_PhysicalProps = candidates[selectedIndex].props;

                VulkanLogicalDeviceSetup setup;
                std::string devErr;
                if (!createLogicalVulkanDevice(
                        m_Fn,
                        m_PhysicalDevice,
                        m_PhysicalProps,
                        m_InstanceApiVersion,
                        setup,
                        devErr))
                {
                    outIsUnavailable = false;
                    outSkipOrError = devErr;
                    return false;
                }

                m_VkDevice = setup.device;
                m_TimelineSemaphoreSupported = setup.timelineSemaphoreSupported;
                m_BufferDeviceAddressSupported = setup.bufferDeviceAddressSupported;
                loadDeviceFunctions(m_Fn, m_VkDevice);

                nvrhi::vulkan::DeviceDesc desc{};
                desc.errorCB = messageCallback;
                desc.instance = m_Instance;
                desc.physicalDevice = m_PhysicalDevice;
                desc.device = m_VkDevice;
                desc.vulkanLibraryName = m_LoaderPath;
                desc.pipelineCacheInitialData = config.vulkanPipelineCacheData;
                desc.pipelineCacheInitialDataSize = config.vulkanPipelineCacheSize;

                m_GraphicsQueueFamilyIndex = static_cast<uint32_t>(setup.queues.graphicsFamily);
                desc.graphicsQueueIndex = setup.queues.graphicsFamily;
                m_Fn.vkGetDeviceQueue(
                    m_VkDevice,
                    static_cast<uint32_t>(setup.queues.graphicsFamily),
                    static_cast<uint32_t>(setup.queues.graphicsIndex),
                    &desc.graphicsQueue);
                m_GraphicsQueue = desc.graphicsQueue;

                if (setup.queues.computeFamily >= 0)
                {
                    desc.computeQueueIndex = setup.queues.computeFamily;
                    m_Fn.vkGetDeviceQueue(
                        m_VkDevice,
                        static_cast<uint32_t>(setup.queues.computeFamily),
                        static_cast<uint32_t>(setup.queues.computeIndex),
                        &desc.computeQueue);
                    m_HasComputeQueue = (desc.computeQueue != VK_NULL_HANDLE);
                }

                if (setup.queues.transferFamily >= 0)
                {
                    desc.transferQueueIndex = setup.queues.transferFamily;
                    m_Fn.vkGetDeviceQueue(
                        m_VkDevice,
                        static_cast<uint32_t>(setup.queues.transferFamily),
                        static_cast<uint32_t>(setup.queues.transferIndex),
                        &desc.transferQueue);
                    m_HasCopyQueue = (desc.transferQueue != VK_NULL_HANDLE);
                }

                desc.instanceExtensions = enabledInstExts.empty() ? nullptr : enabledInstExts.data();
                desc.numInstanceExtensions = enabledInstExts.size();
                desc.deviceExtensions = setup.enabledExtensions.empty() ? nullptr : setup.enabledExtensions.data();
                desc.numDeviceExtensions = setup.enabledExtensions.size();
                desc.bufferDeviceAddressSupported = setup.bufferDeviceAddressSupported;
                desc.descriptorBindingUniformBufferUpdateAfterBind = setup.descriptorBindingUniformBufferUpdateAfterBind;

#if !defined(UAA_NVRHI_SHARED) || !UAA_NVRHI_SHARED
                VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Instance, m_Fn.vkGetInstanceProcAddr, m_VkDevice);
#endif

                m_NvrhiDevice = nvrhi::vulkan::createDevice(desc);
                if (!m_NvrhiDevice)
                {
                    outIsUnavailable = false;
                    outSkipOrError = "nvrhi::vulkan::createDevice returned null";
                    return false;
                }

                m_InstanceExtensionNames.clear();
                for (const char* ext : enabledInstExts)
                {
                    m_InstanceExtensionNames.emplace_back(ext);
                }

                m_DeviceInfo.backendName = "vulkan";
                m_DeviceInfo.deviceName = candidates[selectedIndex].props.deviceName;
                m_DeviceInfo.driverName = candidates[selectedIndex].driverName.empty()
                    ? "Vulkan"
                    : candidates[selectedIndex].driverName;
                if (!candidates[selectedIndex].driverInfo.empty())
                {
                    m_DeviceInfo.driverVersion = candidates[selectedIndex].driverInfo;
                }
                else
                {
                    std::ostringstream verOss;
                    const uint32_t dv = candidates[selectedIndex].props.driverVersion;
                    verOss << VK_VERSION_MAJOR(dv) << "." << VK_VERSION_MINOR(dv) << "." << VK_VERSION_PATCH(dv);
                    m_DeviceInfo.driverVersion = verOss.str();
                }
                m_DeviceInfo.osName = getOsName();
#ifdef UAA_BUILD_CONFIG
                m_DeviceInfo.buildConfig = UAA_BUILD_CONFIG;
#endif
                m_DeviceInfo.vulkanDrawIndirectCount = setup.drawIndirectCount;
                m_DeviceInfo.vulkanSamplerFilterMinmax = setup.samplerFilterMinmax;
                m_DeviceInfo.nativeValidation = validationLayerEnabled && (m_DebugMessenger != VK_NULL_HANDLE);
                m_DeviceInfo.enabledExtensions = m_InstanceExtensionNames;
                for (const char* ext : setup.enabledExtensions)
                {
                    m_DeviceInfo.enabledExtensions.emplace_back(ext);
                }

                return true;
            }

        private:
            bool m_OwnsInstance = true;
            DynamicLibrary m_Loader;
            std::string m_LoaderPath;
            VulkanFunctions m_Fn{};
            VkInstance m_Instance = VK_NULL_HANDLE;
            VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;
            std::unique_ptr<VulkanValidationState> m_ValidationState;
            VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
            VkPhysicalDeviceProperties m_PhysicalProps{};
            uint32_t m_InstanceApiVersion = VK_API_VERSION_1_0;
            VkDevice m_VkDevice = VK_NULL_HANDLE;
            VkDevice m_PrimaryVkDevice = VK_NULL_HANDLE;
            uint32_t m_GraphicsQueueFamilyIndex = 0;
            VkQueue m_GraphicsQueue = VK_NULL_HANDLE;
            bool m_TimelineSemaphoreSupported = false;
            bool m_BufferDeviceAddressSupported = false;
            bool m_HasComputeQueue = false;
            bool m_HasCopyQueue = false;
            std::vector<std::string> m_InstanceExtensionNames;
            nvrhi::vulkan::DeviceHandle m_NvrhiDevice;
            DeviceInfo m_DeviceInfo;
        };

        void runSimpleComputeAdd(
            TestContext& ctx,
            nvrhi::IDevice* device,
            const nvrhi::BindingLayoutDesc& layoutDesc,
            nvrhi::IBuffer* inputBuf,
            nvrhi::IBuffer* outputBuf,
            uint32_t count,
            uint32_t addVal,
            uint32_t xorVal)
        {
            const std::vector<uint8_t> spirv = ctx.fixture->loadShaderBinary("cs_buffer_copy_add");
            nvrhi::ShaderDesc shaderDesc;
            shaderDesc.setShaderType(nvrhi::ShaderType::Compute)
                .setEntryName("main")
                .setDebugName("cs_buffer_copy_add");
            nvrhi::ShaderHandle cs = device->createShader(shaderDesc, spirv.data(), spirv.size());
            UAA_CHECK(cs != nullptr, "createShader(cs_buffer_copy_add) returned null");

            struct Params
            {
                uint32_t count;
                uint32_t addVal;
                uint32_t xorVal;
                uint32_t pad;
            };
            Params params{ count, addVal, xorVal, 0u };

            nvrhi::BufferDesc cbDesc;
            cbDesc.byteSize = sizeof(Params);
            cbDesc.isConstantBuffer = true;
            cbDesc.isVolatile = true;
            cbDesc.maxVersions = 4;
            cbDesc.debugName = "uaa_vk_sync_params";
            nvrhi::BufferHandle cb = device->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "createBuffer(cb) returned null");

            nvrhi::BindingLayoutDesc bld = layoutDesc;
            bld.addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
               .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
               .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layout = device->createBindingLayout(bld);
            UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

            nvrhi::ComputePipelineDesc pipeDesc;
            pipeDesc.CS = cs;
            pipeDesc.addBindingLayout(layout);
            nvrhi::ComputePipelineHandle pso = device->createComputePipeline(pipeDesc);
            UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                   .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inputBuf))
                   .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outputBuf));
            nvrhi::BindingSetHandle bset = device->createBindingSet(setDesc, layout);
            UAA_CHECK(bset != nullptr, "createBindingSet returned null");

            nvrhi::CommandListHandle cmd = device->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(cb, &params, sizeof(params));
            nvrhi::ComputeState state;
            state.pipeline = pso;
            state.addBindingSet(bset);
            cmd->setComputeState(state);
            cmd->dispatch((count + 63u) / 64u, 1u, 1u);
            cmd->close();

            device->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(device->waitForIdle(), "waitForIdle failed");
            device->runGarbageCollection();
        }
    } // namespace

    std::unique_ptr<IBackendContext> createVulkanBackend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        auto ctx = std::make_unique<VulkanBackendContext>();
        if (!ctx->init(config, messageCallback, outSkipOrError, outIsUnavailable))
        {
            return nullptr;
        }
        return ctx;
    }

    static void testVulkanSync(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::VULKAN)
        {
            UAA_SKIP("NAT.VulkanSync requires the Vulkan backend");
        }

        nvrhi::IDevice* device = ctx.fixture->device();
        auto* vkDev = static_cast<nvrhi::vulkan::IDevice*>(
            device->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
        UAA_CHECK(vkDev != nullptr, "Nvrhi_VK_Device native object returned null");

        VkDevice nativeVkDevice = static_cast<VkDevice>(
            device->getNativeObject(nvrhi::ObjectTypes::VK_Device).pointer);
        UAA_CHECK(nativeVkDevice != VK_NULL_HANDLE, "VK_Device native object returned null");

        VkSemaphore gfxQueueSem = vkDev->getQueueSemaphore(nvrhi::CommandQueue::Graphics);
        UAA_CHECK(gfxQueueSem != VK_NULL_HANDLE, "getQueueSemaphore(Graphics) returned null");

        const uint64_t initialCompleted = vkDev->queueGetCompletedInstance(nvrhi::CommandQueue::Graphics);

        // Load device-level timeline semaphore functions via dynamic loader matching this Vulkan instance/device.
        DynamicLibrary loader;
        std::vector<std::string> candidateLibs;
#ifdef _WIN32
        candidateLibs.emplace_back("vulkan-1.dll");
#elif defined(__APPLE__)
        candidateLibs.emplace_back("libvulkan.1.dylib");
        candidateLibs.emplace_back("libvulkan.dylib");
        if (const char* sdkEnv = std::getenv("VULKAN_SDK"))
        {
            if (sdkEnv[0] != '\0')
            {
                candidateLibs.emplace_back(std::string(sdkEnv) + "/lib/libvulkan.1.dylib");
            }
        }
        candidateLibs.emplace_back("/Users/hypernewbie/VulkanSDK/macOS/lib/libvulkan.1.dylib");
        candidateLibs.emplace_back("/usr/local/lib/libvulkan.1.dylib");
        candidateLibs.emplace_back("/opt/homebrew/lib/libvulkan.1.dylib");
#else
        candidateLibs.emplace_back("libvulkan.so.1");
        candidateLibs.emplace_back("libvulkan.so");
#endif
        PFN_vkGetInstanceProcAddr getInstProc = nullptr;
        for (const std::string& lib : candidateLibs)
        {
            if (loader.open(lib.c_str()))
            {
                getInstProc = loader.loadSymbol<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
                if (getInstProc)
                {
                    break;
                }
                loader.close();
            }
        }
        UAA_CHECK(getInstProc != nullptr, "Failed to load vkGetInstanceProcAddr for VulkanSync");

        VkInstance nativeVkInstance = static_cast<VkInstance>(
            device->getNativeObject(nvrhi::ObjectTypes::VK_Instance).pointer);
        UAA_CHECK(nativeVkInstance != VK_NULL_HANDLE, "VK_Instance native object returned null");

        auto getDevProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            getInstProc(nativeVkInstance, "vkGetDeviceProcAddr"));
        UAA_CHECK(getDevProc != nullptr, "Failed to resolve vkGetDeviceProcAddr");

        auto createSemFn = reinterpret_cast<PFN_vkCreateSemaphore>(getDevProc(nativeVkDevice, "vkCreateSemaphore"));
        auto destroySemFn = reinterpret_cast<PFN_vkDestroySemaphore>(getDevProc(nativeVkDevice, "vkDestroySemaphore"));
        auto signalSemFn = reinterpret_cast<PFN_vkSignalSemaphore>(getDevProc(nativeVkDevice, "vkSignalSemaphore"));
        if (!signalSemFn)
        {
            signalSemFn = reinterpret_cast<PFN_vkSignalSemaphore>(getDevProc(nativeVkDevice, "vkSignalSemaphoreKHR"));
        }
        auto getCounterFn = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            getDevProc(nativeVkDevice, "vkGetSemaphoreCounterValue"));
        if (!getCounterFn)
        {
            getCounterFn = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
                getDevProc(nativeVkDevice, "vkGetSemaphoreCounterValueKHR"));
        }
        UAA_CHECK(createSemFn && destroySemFn && signalSemFn && getCounterFn,
            "Timeline semaphore device entry points unavailable");

        auto createTimelineSem = [&](uint64_t initVal) -> VkSemaphore
        {
            VkSemaphoreTypeCreateInfo typeInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
            typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            typeInfo.initialValue = initVal;

            VkSemaphoreCreateInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            info.pNext = &typeInfo;
            VkSemaphore sem = VK_NULL_HANDLE;
            if (createSemFn(nativeVkDevice, &info, nullptr, &sem) != VK_SUCCESS)
            {
                return VK_NULL_HANDLE;
            }
            return sem;
        };

        VkSemaphore extWaitSem = createTimelineSem(0);
        VkSemaphore extSignalSem = createTimelineSem(0);
        UAA_CHECK(extWaitSem != VK_NULL_HANDLE && extSignalSem != VK_NULL_HANDLE,
            "Failed to create timeline VkSemaphores");

        constexpr uint32_t kCount = 256;
        constexpr uint32_t kAddVal = 17;
        constexpr uint32_t kXorVal = 0x5A5A1234u;
        const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed());

        nvrhi::BufferDesc inDesc;
        inDesc.byteSize = kCount * sizeof(uint32_t);
        inDesc.structStride = sizeof(uint32_t);
        inDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        inDesc.keepInitialState = true;
        inDesc.debugName = "uaa_vk_sync_in";
        nvrhi::BufferHandle inBuf = device->createBuffer(inDesc);
        UAA_CHECK(inBuf != nullptr, "Failed to create input buffer");
        ctx.fixture->uploadBuffer(inBuf, inputWords.data(), inDesc.byteSize);

        nvrhi::BufferDesc outDesc;
        outDesc.byteSize = kCount * sizeof(uint32_t);
        outDesc.structStride = sizeof(uint32_t);
        outDesc.canHaveUAVs = true;
        outDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        outDesc.keepInitialState = true;
        outDesc.debugName = "uaa_vk_sync_out";
        nvrhi::BufferHandle outBuf = device->createBuffer(outDesc);
        UAA_CHECK(outBuf != nullptr, "Failed to create output buffer");

        nvrhi::ShaderHandle cs = ctx.fixture->createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
        struct Params
        {
            uint32_t count;
            uint32_t addVal;
            uint32_t xorVal;
            uint32_t pad;
        };
        Params params{ kCount, kAddVal, kXorVal, 0u };

        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = sizeof(Params);
        cbDesc.isConstantBuffer = true;
        cbDesc.isVolatile = true;
        cbDesc.maxVersions = 4;
        cbDesc.debugName = "uaa_vk_sync_cb";
        nvrhi::BufferHandle cb = device->createBuffer(cbDesc);
        UAA_CHECK(cb != nullptr, "Failed to create constant buffer");

        nvrhi::BindingLayoutDesc bld = ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
        bld.addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
           .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
           .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
        nvrhi::BindingLayoutHandle layout = device->createBindingLayout(bld);
        UAA_CHECK(layout != nullptr, "Failed to create binding layout");

        nvrhi::ComputePipelineDesc psoDesc;
        psoDesc.CS = cs;
        psoDesc.addBindingLayout(layout);
        nvrhi::ComputePipelineHandle pso = device->createComputePipeline(psoDesc);
        UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
               .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
               .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf));
        nvrhi::BindingSetHandle bset = device->createBindingSet(setDesc, layout);
        UAA_CHECK(bset != nullptr, "Failed to create binding set");

        nvrhi::CommandListHandle cmd = device->createCommandList();
        UAA_CHECK(cmd != nullptr, "Failed to create command list");
        cmd->open();
        cmd->writeBuffer(cb, &params, sizeof(params));
        nvrhi::ComputeState csState;
        csState.pipeline = pso;
        csState.addBindingSet(bset);
        cmd->setComputeState(csState);
        cmd->dispatch((kCount + 63u) / 64u, 1u, 1u);
        cmd->close();

        vkDev->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, extWaitSem, 1);
        vkDev->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, extSignalSem, 5);
        const uint64_t submittedInstance = device->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
        UAA_CHECK(submittedInstance > initialCompleted, "Submitted instance must advance queue instance counter");

        uint64_t preSignalVal = 999;
        UAA_CHECK(getCounterFn(nativeVkDevice, extSignalSem, &preSignalVal) == VK_SUCCESS,
            "vkGetSemaphoreCounterValue failed");
        UAA_CHECK_EQ(preSignalVal, 0ull, "extSignalSem should remain 0 while GPU waits on extWaitSem");

        VkSemaphoreSignalInfo sigInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO };
        sigInfo.semaphore = extWaitSem;
        sigInfo.value = 1;
        UAA_CHECK(signalSemFn(nativeVkDevice, &sigInfo) == VK_SUCCESS, "vkSignalSemaphore failed");

        UAA_CHECK(device->waitForIdle(), "waitForIdle failed after signaling gate semaphore");
        const uint64_t postCompleted = vkDev->queueGetCompletedInstance(nvrhi::CommandQueue::Graphics);
        UAA_CHECK(postCompleted >= submittedInstance, "queueGetCompletedInstance must reach submitted instance");

        uint64_t postSignalVal = 0;
        UAA_CHECK(getCounterFn(nativeVkDevice, extSignalSem, &postSignalVal) == VK_SUCCESS,
            "vkGetSemaphoreCounterValue failed after completion");
        UAA_CHECK_EQ(postSignalVal, 5ull, "queueSignalSemaphore did not signal expected timeline value 5");

        std::vector<uint32_t> actualWords(kCount, 0u);
        ctx.fixture->readbackBuffer(outBuf, actualWords.data(), outDesc.byteSize);
        for (uint32_t i = 0; i < kCount; ++i)
        {
            const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
            UAA_CHECK_EQ(actualWords[i], expected, "Compute output mismatch after semaphore sync");
        }

        if (device->queryFeatureSupport(nvrhi::Feature::ComputeQueue))
        {
            VkSemaphore computeQueueSem = vkDev->getQueueSemaphore(nvrhi::CommandQueue::Compute);
            UAA_CHECK(computeQueueSem != VK_NULL_HANDLE, "getQueueSemaphore(Compute) returned null");

            nvrhi::BufferDesc crossOutDesc = outDesc;
            crossOutDesc.debugName = "uaa_vk_sync_cross_out";
            nvrhi::BufferHandle crossOutBuf = device->createBuffer(crossOutDesc);
            UAA_CHECK(crossOutBuf != nullptr, "Failed to create cross-queue output buffer");

            nvrhi::BindingSetDesc crossSetDesc;
            crossSetDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, outBuf))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, crossOutBuf));
            nvrhi::BindingSetHandle crossSet = device->createBindingSet(crossSetDesc, layout);
            UAA_CHECK(crossSet != nullptr, "Failed to create cross-queue binding set");

            nvrhi::CommandListParameters compParams;
            compParams.queueType = nvrhi::CommandQueue::Compute;
            nvrhi::CommandListHandle compCmd = device->createCommandList(compParams);
            UAA_CHECK(compCmd != nullptr, "Failed to create compute queue command list");
            compCmd->open();
            compCmd->writeBuffer(cb, &params, sizeof(params));
            nvrhi::ComputeState compState;
            compState.pipeline = pso;
            compState.addBindingSet(crossSet);
            compCmd->setComputeState(compState);
            compCmd->dispatch((kCount + 63u) / 64u, 1u, 1u);
            compCmd->close();

            const uint64_t compInstance = device->executeCommandList(compCmd, nvrhi::CommandQueue::Compute);
            vkDev->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics, computeQueueSem, compInstance);

            nvrhi::BufferDesc rbDesc;
            rbDesc.byteSize = crossOutDesc.byteSize;
            rbDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            rbDesc.debugName = "uaa_vk_sync_cross_rb";
            nvrhi::BufferHandle rbBuf = device->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "Failed to create cross-queue readback buffer");

            nvrhi::CommandListHandle copyCmd = device->createCommandList();
            UAA_CHECK(copyCmd != nullptr, "Failed to create graphics copy command list");
            copyCmd->open();
            copyCmd->copyBuffer(rbBuf, 0, crossOutBuf, 0, crossOutDesc.byteSize);
            copyCmd->close();
            device->executeCommandList(copyCmd, nvrhi::CommandQueue::Graphics);

            UAA_CHECK(device->waitForIdle(), "waitForIdle failed after cross-queue sync");
            UAA_CHECK(vkDev->queueGetCompletedInstance(nvrhi::CommandQueue::Compute) >= compInstance,
                "Compute queue completed instance did not advance");

            void* mapped = device->mapBuffer(rbBuf, nvrhi::CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer failed on cross-queue readback");
            std::vector<uint32_t> crossWords(kCount, 0u);
            std::memcpy(crossWords.data(), mapped, crossOutDesc.byteSize);
            device->unmapBuffer(rbBuf);

            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = (actualWords[i] + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(crossWords[i], expected, "Cross-queue compute->graphics output mismatch");
            }
        }

        device->waitForIdle();
        device->runGarbageCollection();
        destroySemFn(nativeVkDevice, extWaitSem, nullptr);
        destroySemFn(nativeVkDevice, extSignalSem, nullptr);
    }

    static void testVulkanPipelineCache(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::VULKAN)
        {
            UAA_SKIP("NAT.VulkanPipelineCache requires the Vulkan backend");
        }

        // First create a fresh empty-cache Vulkan backend so we can observe initial cache size vs post-PSO size.
        MessageCapture localCapture;
        BackendConfig freshCfg;
        freshCfg.nativeValidation = ctx.options.nativeValidation;
        freshCfg.deviceSelector = ctx.options.deviceSelector;
        std::string msg;
        bool unavailable = false;
        std::unique_ptr<IBackendContext> freshBackend = createVulkanBackend(freshCfg, &localCapture, msg, unavailable);
        UAA_CHECK(freshBackend != nullptr, "Failed to create fresh Vulkan backend for pipeline cache test: " + msg);
        UAA_CHECK_EQ(localCapture.errorCount(), size_t(0), "Fresh Vulkan backend reported errors during init");

        nvrhi::IDevice* freshRaw = freshBackend->getRawDevice();
        auto* freshVk = static_cast<nvrhi::vulkan::IDevice*>(
            freshRaw->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
        UAA_CHECK(freshVk != nullptr, "Fresh Nvrhi_VK_Device returned null");

        const size_t initialCacheSize = freshVk->getPipelineCacheDataSize();
        std::vector<uint8_t> initialBlob;
        const bool initialGot = freshVk->getPipelineCacheData(initialBlob);
        if (initialCacheSize == 0)
        {
            UAA_CHECK(!initialGot && initialBlob.empty(), "Empty cache must return false and empty vector");
        }
        else
        {
            UAA_CHECK(initialGot && initialBlob.size() >= initialCacheSize,
                "Non-empty initial cache query must populate blob");
        }

        constexpr uint32_t kCount = 64;
        const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xCAFEu);
        nvrhi::BufferDesc inDesc;
        inDesc.byteSize = kCount * sizeof(uint32_t);
        inDesc.structStride = sizeof(uint32_t);
        inDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        inDesc.keepInitialState = true;
        nvrhi::BufferHandle inBuf = freshRaw->createBuffer(inDesc);
        nvrhi::BufferDesc outDesc = inDesc;
        outDesc.canHaveUAVs = true;
        outDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        nvrhi::BufferHandle outBuf = freshRaw->createBuffer(outDesc);
        UAA_CHECK(inBuf != nullptr && outBuf != nullptr, "Failed to create fresh backend buffers");

        nvrhi::CommandListHandle upCmd = freshRaw->createCommandList();
        upCmd->open();
        upCmd->writeBuffer(inBuf, inputWords.data(), inDesc.byteSize);
        upCmd->close();
        freshRaw->executeCommandList(upCmd);
        freshRaw->waitForIdle();

        runSimpleComputeAdd(
            ctx,
            freshRaw,
            ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute),
            inBuf,
            outBuf,
            kCount,
            3u,
            0x1111u);

        const size_t populatedSize = freshVk->getPipelineCacheDataSize();
        std::vector<uint8_t> exportedBlob;
        const bool exportedOk = freshVk->getPipelineCacheData(exportedBlob);
        UAA_CHECK(populatedSize > 0, "Pipeline cache size should be > 0 after creating compute pipeline");
        UAA_CHECK(exportedOk, "getPipelineCacheData() failed after creating compute pipeline");
        UAA_CHECK(!exportedBlob.empty() && exportedBlob.size() >= 16,
            "Exported Vulkan pipeline cache blob is unexpectedly small");

        // Verify standard Vulkan pipeline cache header (uint32 headerSize >= 16, uint32 version == 1).
        uint32_t headerSize = 0;
        uint32_t headerVersion = 0;
        std::memcpy(&headerSize, exportedBlob.data(), sizeof(uint32_t));
        std::memcpy(&headerVersion, exportedBlob.data() + sizeof(uint32_t), sizeof(uint32_t));
        UAA_CHECK(headerSize >= 16 && headerSize <= exportedBlob.size(),
            "Exported pipeline cache has invalid header length");
        UAA_CHECK_EQ(headerVersion, 1u, "Exported pipeline cache header version should be 1");

        std::ostringstream noteOss;
        noteOss << "initialSize=" << initialCacheSize
                << " populatedSize=" << populatedSize
                << " exportedBytes=" << exportedBlob.size();
        ctx.note(noteOss.str());

        inBuf = nullptr;
        outBuf = nullptr;
        upCmd = nullptr;
        freshBackend.reset();

        // Round-trip: seed a second independent device from the exported blob.
        MessageCapture seededCapture;
        BackendConfig seededCfg = freshCfg;
        seededCfg.vulkanPipelineCacheData = exportedBlob.data();
        seededCfg.vulkanPipelineCacheSize = exportedBlob.size();
        std::unique_ptr<IBackendContext> seededBackend = createVulkanBackend(seededCfg, &seededCapture, msg, unavailable);
        UAA_CHECK(seededBackend != nullptr, "Failed to create seeded Vulkan backend: " + msg);
        UAA_CHECK_EQ(seededCapture.errorCount(), size_t(0),
            "Seeded Vulkan backend reported errors: " + seededCapture.formatAll());

        nvrhi::IDevice* seededRaw = seededBackend->getRawDevice();
        auto* seededVk = static_cast<nvrhi::vulkan::IDevice*>(
            seededRaw->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device).pointer);
        UAA_CHECK(seededVk != nullptr, "Seeded Nvrhi_VK_Device returned null");

        nvrhi::BufferHandle sIn = seededRaw->createBuffer(inDesc);
        nvrhi::BufferHandle sOut = seededRaw->createBuffer(outDesc);
        nvrhi::BufferDesc rbDesc;
        rbDesc.byteSize = outDesc.byteSize;
        rbDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        nvrhi::BufferHandle sRb = seededRaw->createBuffer(rbDesc);
        UAA_CHECK(sIn != nullptr && sOut != nullptr && sRb != nullptr, "Failed to create seeded backend buffers");

        nvrhi::CommandListHandle sUp = seededRaw->createCommandList();
        sUp->open();
        sUp->writeBuffer(sIn, inputWords.data(), inDesc.byteSize);
        sUp->close();
        seededRaw->executeCommandList(sUp);
        seededRaw->waitForIdle();

        runSimpleComputeAdd(
            ctx,
            seededRaw,
            ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute),
            sIn,
            sOut,
            kCount,
            5u,
            0x2222u);

        nvrhi::CommandListHandle sReadCmd = seededRaw->createCommandList();
        sReadCmd->open();
        sReadCmd->copyBuffer(sRb, 0, sOut, 0, outDesc.byteSize);
        sReadCmd->close();
        seededRaw->executeCommandList(sReadCmd);
        seededRaw->waitForIdle();

        void* sMapped = seededRaw->mapBuffer(sRb, nvrhi::CpuAccessMode::Read);
        UAA_CHECK(sMapped != nullptr, "mapBuffer failed on seeded backend");
        std::vector<uint32_t> sActual(kCount, 0u);
        std::memcpy(sActual.data(), sMapped, outDesc.byteSize);
        seededRaw->unmapBuffer(sRb);
        for (uint32_t i = 0; i < kCount; ++i)
        {
            UAA_CHECK_EQ(sActual[i], (inputWords[i] + 5u) ^ 0x2222u, "Seeded pipeline output mismatch");
        }

        sIn = nullptr;
        sOut = nullptr;
        sRb = nullptr;
        sUp = nullptr;
        sReadCmd = nullptr;
        seededBackend.reset();

        // Corrupt / stale blob fallback: must not crash and must still produce a working device.
        std::vector<uint8_t> corruptBlob(64, 0xAAu);
        MessageCapture corruptCapture;
        BackendConfig corruptCfg = freshCfg;
        corruptCfg.vulkanPipelineCacheData = corruptBlob.data();
        corruptCfg.vulkanPipelineCacheSize = corruptBlob.size();
        std::unique_ptr<IBackendContext> corruptBackend = createVulkanBackend(corruptCfg, &corruptCapture, msg, unavailable);
        UAA_CHECK(corruptBackend != nullptr, "Corrupt pipeline cache blob caused Vulkan backend init to fail: " + msg);
        UAA_CHECK_EQ(corruptCapture.errorCount(), size_t(0),
            "Corrupt pipeline cache fallback should warn rather than error: " + corruptCapture.formatAll());

        nvrhi::IDevice* corruptRaw = corruptBackend->getRawDevice();
        nvrhi::BufferHandle cIn = corruptRaw->createBuffer(inDesc);
        nvrhi::BufferHandle cOut = corruptRaw->createBuffer(outDesc);
        nvrhi::BufferHandle cRb = corruptRaw->createBuffer(rbDesc);
        UAA_CHECK(cIn != nullptr && cOut != nullptr && cRb != nullptr, "Failed to create buffers on corrupt-fallback device");

        nvrhi::CommandListHandle cUp = corruptRaw->createCommandList();
        cUp->open();
        cUp->writeBuffer(cIn, inputWords.data(), inDesc.byteSize);
        cUp->close();
        corruptRaw->executeCommandList(cUp);
        corruptRaw->waitForIdle();

        runSimpleComputeAdd(
            ctx,
            corruptRaw,
            ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute),
            cIn,
            cOut,
            kCount,
            7u,
            0x3333u);

        nvrhi::CommandListHandle cReadCmd = corruptRaw->createCommandList();
        cReadCmd->open();
        cReadCmd->copyBuffer(cRb, 0, cOut, 0, outDesc.byteSize);
        cReadCmd->close();
        corruptRaw->executeCommandList(cReadCmd);
        corruptRaw->waitForIdle();

        void* cMapped = corruptRaw->mapBuffer(cRb, nvrhi::CpuAccessMode::Read);
        UAA_CHECK(cMapped != nullptr, "mapBuffer failed on corrupt-fallback backend");
        std::vector<uint32_t> cActual(kCount, 0u);
        std::memcpy(cActual.data(), cMapped, outDesc.byteSize);
        corruptRaw->unmapBuffer(cRb);
        for (uint32_t i = 0; i < kCount; ++i)
        {
            UAA_CHECK_EQ(cActual[i], (inputWords[i] + 7u) ^ 0x3333u, "Corrupt-fallback pipeline output mismatch");
        }

        cIn = nullptr;
        cOut = nullptr;
        cRb = nullptr;
        cUp = nullptr;
        cReadCmd = nullptr;
        corruptBackend.reset();
    }

    UAA_REGISTER_TEST("NAT.VulkanSync", "native,vulkan", testVulkanSync);
    UAA_REGISTER_TEST("NAT.VulkanPipelineCache", "native,vulkan", testVulkanPipelineCache);
} // namespace nvrhi::uaa

#endif // defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
