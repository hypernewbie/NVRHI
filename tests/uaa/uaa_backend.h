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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <nvrhi/nvrhi.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    struct BackendConfig
    {
        bool nativeValidation = false;
        std::string deviceSelector;
        const void* vulkanPipelineCacheData = nullptr;
        size_t vulkanPipelineCacheSize = 0;
    };

    struct DeviceInfo
    {
        std::string backendName;
        std::string deviceName;
        std::string driverName;
        std::string driverVersion;
        std::string osName;
        std::string buildConfig;
        bool nativeValidation = false;
        bool nvrhiValidation = false;
        bool vulkanDrawIndirectCount = false;
        bool vulkanSamplerFilterMinmax = false;
        bool vulkanShader16Bit = false;
        bool metalArgumentBuffersTier2 = false;
        std::vector<std::string> enabledExtensions;
    };

    class IGpuGate
    {
    public:
        virtual ~IGpuGate() = default;
        virtual bool enqueueWait(nvrhi::CommandQueue queue) = 0;
        virtual void signal() = 0;
        virtual void reset() = 0;
    };

    struct NativeBufferImport
    {
        nvrhi::ObjectType objectType = 0;
        nvrhi::Object handle = nullptr;
        uint64_t byteSize = 0;
        std::function<void()> release;
    };

    struct NativeTextureImport
    {
        nvrhi::ObjectType objectType = 0;
        nvrhi::Object handle = nullptr;
        nvrhi::TextureDesc desc;
        std::function<void()> release;
    };

    class IBackendContext
    {
    public:
        virtual ~IBackendContext() = default;
        virtual nvrhi::IDevice* getRawDevice() = 0;
        virtual const DeviceInfo& getDeviceInfo() const = 0;

        virtual std::unique_ptr<IGpuGate> createGpuGate()
        {
            return nullptr;
        }

        virtual bool createNativeImportBuffer(uint64_t byteSize, const void* initialData, NativeBufferImport& out)
        {
            (void)byteSize;
            (void)initialData;
            (void)out;
            return false;
        }

        virtual bool createNativeImportTexture(uint32_t width, uint32_t height, nvrhi::Format format, NativeTextureImport& out)
        {
            (void)width;
            (void)height;
            (void)format;
            (void)out;
            return false;
        }

        virtual std::unique_ptr<IBackendContext> createSecondaryDevice(nvrhi::IMessageCallback* cb)
        {
            (void)cb;
            return nullptr;
        }

        virtual bool hasNativeValidationErrors(std::vector<std::string>& outErrors)
        {
            (void)outErrors;
            return false;
        }

        virtual void clearNativeValidationMessages()
        {
        }

        virtual uint64_t getMappedBufferFlushAlignment()
        {
            return 0;
        }

        virtual bool flushMappedBuffer(nvrhi::IDevice* activeDevice, nvrhi::IBuffer* buffer, uint64_t offset, uint64_t size)
        {
            (void)activeDevice;
            (void)buffer;
            (void)offset;
            (void)size;
            return false;
        }
    };

    std::unique_ptr<IBackendContext> createBackendContext(
        const std::string& backendName,
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable);

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
    std::unique_ptr<IBackendContext> createVulkanBackend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable);
#endif

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
    std::unique_ptr<IBackendContext> createMetalBackend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable);
#endif

#if (defined(UAA_WITH_D3D11) && UAA_WITH_D3D11) || (defined(UAA_WITH_DX11) && UAA_WITH_DX11)
    std::unique_ptr<IBackendContext> createD3D11Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable);
#endif

#if (defined(UAA_WITH_D3D12) && UAA_WITH_D3D12) || (defined(UAA_WITH_DX12) && UAA_WITH_DX12)
    std::unique_ptr<IBackendContext> createD3D12Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable);
#endif
} // namespace nvrhi::uaa
