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

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL

#include "uaa_backend.h"
#include "uaa_fixture.h"
#include "uaa_runner.h"

#include <nvrhi/metal.h>
#if UAA_WITH_VALIDATION
#include <nvrhi/validation.h>
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
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

        MTLPixelFormat convertFormatForMetalImport(nvrhi::Format format, bool& outIsDepth)
        {
            outIsDepth = false;
            switch (format)
            {
            case nvrhi::Format::R8_UNORM:
                return MTLPixelFormatR8Unorm;
            case nvrhi::Format::RG8_UNORM:
                return MTLPixelFormatRG8Unorm;
            case nvrhi::Format::RGBA8_UNORM:
                return MTLPixelFormatRGBA8Unorm;
            case nvrhi::Format::SRGBA8_UNORM:
                return MTLPixelFormatRGBA8Unorm_sRGB;
            case nvrhi::Format::BGRA8_UNORM:
                return MTLPixelFormatBGRA8Unorm;
            case nvrhi::Format::SBGRA8_UNORM:
                return MTLPixelFormatBGRA8Unorm_sRGB;
            case nvrhi::Format::R16_FLOAT:
                return MTLPixelFormatR16Float;
            case nvrhi::Format::RGBA16_FLOAT:
                return MTLPixelFormatRGBA16Float;
            case nvrhi::Format::R32_UINT:
                return MTLPixelFormatR32Uint;
            case nvrhi::Format::R32_FLOAT:
                return MTLPixelFormatR32Float;
            case nvrhi::Format::RGBA32_FLOAT:
                return MTLPixelFormatRGBA32Float;
            case nvrhi::Format::D32:
                outIsDepth = true;
                return MTLPixelFormatDepth32Float;
            default:
                return MTLPixelFormatInvalid;
            }
        }

        class MetalGpuGate final : public IGpuGate
        {
        public:
            MetalGpuGate(id<MTLDevice> mtlDevice, nvrhi::metal::IDevice* nvrhiMetalDevice)
                : m_MtlDevice(mtlDevice)
                , m_NvrhiMetalDevice(nvrhiMetalDevice)
            {
                @autoreleasepool
                {
                    if (m_MtlDevice)
                    {
                        m_SharedEvent = [m_MtlDevice newSharedEvent];
                        if (m_SharedEvent)
                        {
                            m_SharedEvent.signaledValue = 0;
                        }
                    }
                }
            }

            bool isValid() const
            {
                return m_MtlDevice != nil && m_NvrhiMetalDevice != nullptr && m_SharedEvent != nil;
            }

            bool enqueueWait(nvrhi::CommandQueue queue) override
            {
                (void)queue;
                if (!isValid())
                {
                    return false;
                }
                @autoreleasepool
                {
                    id<MTLCommandQueue> mtlQueue =
                        (__bridge id<MTLCommandQueue>)m_NvrhiMetalDevice->getNativeCommandQueue().pointer;
                    if (!mtlQueue)
                    {
                        return false;
                    }
                    id<MTLCommandBuffer> cb = [mtlQueue commandBuffer];
                    if (!cb)
                    {
                        return false;
                    }
                    [cb encodeWaitForEvent:m_SharedEvent value:1];
                    [cb commit];
                    return true;
                }
            }

            void signal() override
            {
                @autoreleasepool
                {
                    if (m_SharedEvent)
                    {
                        m_SharedEvent.signaledValue = 1;
                    }
                }
            }

            void reset() override
            {
                @autoreleasepool
                {
                    if (m_MtlDevice)
                    {
                        m_SharedEvent = [m_MtlDevice newSharedEvent];
                        if (m_SharedEvent)
                        {
                            m_SharedEvent.signaledValue = 0;
                        }
                    }
                }
            }

        private:
            id<MTLDevice> m_MtlDevice = nil;
            nvrhi::metal::IDevice* m_NvrhiMetalDevice = nullptr;
            id<MTLSharedEvent> m_SharedEvent = nil;
        };

        class MetalBackendContext final : public IBackendContext
        {
        public:
            MetalBackendContext() = default;

            ~MetalBackendContext() override
            {
                if (m_NvrhiDevice)
                {
                    m_NvrhiDevice->waitForIdle();
                    m_NvrhiDevice->runGarbageCollection();
                    m_NvrhiDevice = nullptr;
                }
                m_MtlDevice = nil;
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
                if (!m_MtlDevice || !m_NvrhiDevice)
                {
                    return nullptr;
                }
                auto gate = std::make_unique<MetalGpuGate>(m_MtlDevice, m_NvrhiDevice.Get());
                return gate->isValid() ? std::move(gate) : nullptr;
            }

            bool createNativeImportBuffer(uint64_t byteSize, const void* initialData, NativeBufferImport& out) override
            {
                if (!m_MtlDevice)
                {
                    return false;
                }

                @autoreleasepool
                {
                    const NSUInteger allocLen = static_cast<NSUInteger>(byteSize > 0 ? byteSize : 4);
                    id<MTLBuffer> buf = [m_MtlDevice newBufferWithLength:allocLen options:MTLResourceStorageModeShared];
                    if (!buf)
                    {
                        return false;
                    }

                    if (initialData && byteSize > 0 && buf.contents)
                    {
                        std::memcpy(buf.contents, initialData, static_cast<size_t>(byteSize));
                    }

                    void* retained = (__bridge_retained void*)buf;
                    auto holder = std::shared_ptr<void>(retained, [](void* ptr)
                    {
                        if (ptr)
                        {
                            @autoreleasepool
                            {
                                id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)ptr;
                                (void)released;
                            }
                        }
                    });

                    out.objectType = nvrhi::ObjectTypes::Metal_Buffer;
                    out.handle = nvrhi::Object(retained);
                    out.byteSize = byteSize;
                    out.release = [holder]() mutable
                    {
                        holder.reset();
                    };
                    return true;
                }
            }

            bool createNativeImportTexture(uint32_t width, uint32_t height, nvrhi::Format format, NativeTextureImport& out) override
            {
                if (!m_MtlDevice)
                {
                    return false;
                }

                bool isDepth = false;
                const MTLPixelFormat pixelFormat = convertFormatForMetalImport(format, isDepth);
                if (pixelFormat == MTLPixelFormatInvalid)
                {
                    return false;
                }

                @autoreleasepool
                {
                    MTLTextureDescriptor* td = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:pixelFormat
                        width:width
                        height:height
                        mipmapped:NO];
                    td.storageMode = MTLStorageModePrivate;
                    td.usage = isDepth
                        ? (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget)
                        : (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget);

                    id<MTLTexture> tex = [m_MtlDevice newTextureWithDescriptor:td];
                    if (!tex)
                    {
                        return false;
                    }

                    nvrhi::TextureDesc desc;
                    desc.width = width;
                    desc.height = height;
                    desc.depth = 1;
                    desc.arraySize = 1;
                    desc.mipLevels = 1;
                    desc.sampleCount = 1;
                    desc.format = format;
                    desc.dimension = nvrhi::TextureDimension::Texture2D;
                    desc.isRenderTarget = true;
                    desc.isUAV = !isDepth;
                    desc.initialState = nvrhi::ResourceStates::Common;
                    desc.keepInitialState = true;
                    desc.debugName = "uaa_native_import_mtl_tex";

                    void* retained = (__bridge_retained void*)tex;
                    auto holder = std::shared_ptr<void>(retained, [](void* ptr)
                    {
                        if (ptr)
                        {
                            @autoreleasepool
                            {
                                id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)ptr;
                                (void)released;
                            }
                        }
                    });

                    out.objectType = nvrhi::ObjectTypes::Metal_Texture;
                    out.handle = nvrhi::Object(retained);
                    out.desc = desc;
                    out.release = [holder]() mutable
                    {
                        holder.reset();
                    };
                    return true;
                }
            }

            std::unique_ptr<IBackendContext> createSecondaryDevice(nvrhi::IMessageCallback* cb) override
            {
                if (!m_MtlDevice)
                {
                    return nullptr;
                }

                @autoreleasepool
                {
                    nvrhi::metal::DeviceDesc desc;
                    desc.errorCB = cb;
                    desc.device = nvrhi::Object((__bridge void*)m_MtlDevice);

                    nvrhi::metal::DeviceHandle secDev = nvrhi::metal::createDevice(desc);
                    if (!secDev)
                    {
                        return nullptr;
                    }

                    auto sec = std::make_unique<MetalBackendContext>();
                    sec->m_MtlDevice = m_MtlDevice;
                    sec->m_NvrhiDevice = secDev;
                    sec->m_DeviceInfo = m_DeviceInfo;
                    return sec;
                }
            }

            uint64_t getMappedBufferFlushAlignment() override
            {
                return m_NvrhiDevice ? m_NvrhiDevice->getMappedBufferFlushAlignment() : 0;
            }

            bool flushMappedBuffer(nvrhi::IDevice* activeDevice, nvrhi::IBuffer* buffer, uint64_t offset, uint64_t size) override
            {
                (void)activeDevice;
                return m_NvrhiDevice ? m_NvrhiDevice->flushMappedBuffer(buffer, offset, size) : false;
            }

            bool init(
                const BackendConfig& config,
                nvrhi::IMessageCallback* messageCallback,
                std::string& outSkipOrError,
                bool& outIsUnavailable)
            {
                outSkipOrError.clear();
                outIsUnavailable = false;

                const char* debugLayer = std::getenv("MTL_DEBUG_LAYER");
                const bool nativeValidation = debugLayer && std::strcmp(debugLayer, "1") == 0;
                if (config.nativeValidation && !nativeValidation)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "Metal API validation requires launching with MTL_DEBUG_LAYER=1";
                    return false;
                }

                @autoreleasepool
                {
                    NSMutableArray<id<MTLDevice>>* allDevices = [NSMutableArray array];
#if TARGET_OS_OSX
                    NSArray<id<MTLDevice>>* copied = MTLCopyAllDevices();
                    if (copied)
                    {
                        [allDevices addObjectsFromArray:copied];
                    }
#endif
                    if (allDevices.count == 0)
                    {
                        id<MTLDevice> defaultDev = MTLCreateSystemDefaultDevice();
                        if (defaultDev)
                        {
                            [allDevices addObject:defaultDev];
                        }
                    }

                    if (allDevices.count == 0)
                    {
                        outIsUnavailable = true;
                        outSkipOrError = "No Metal device available";
                        return false;
                    }

                    id<MTLDevice> selected = nil;
                    if (!config.deviceSelector.empty())
                    {
                        size_t parsedIndex = 0;
                        if (tryParseIndex(config.deviceSelector, parsedIndex))
                        {
                            if (parsedIndex < allDevices.count)
                            {
                                selected = allDevices[parsedIndex];
                            }
                        }
                        else
                        {
                            for (id<MTLDevice> dev in allDevices)
                            {
                                const std::string name = dev.name ? dev.name.UTF8String : "";
                                if (containsCaseInsensitive(name, config.deviceSelector))
                                {
                                    selected = dev;
                                    break;
                                }
                            }
                        }

                        if (!selected)
                        {
                            outIsUnavailable = true;
                            outSkipOrError = "No Metal device matching selector '" + config.deviceSelector + "'";
                            return false;
                        }
                    }
                    else
                    {
                        selected = MTLCreateSystemDefaultDevice();
                        if (!selected)
                        {
                            selected = allDevices[0];
                        }
                    }

                    m_MtlDevice = selected;

                    nvrhi::metal::DeviceDesc desc;
                    desc.errorCB = messageCallback;
                    desc.device = nvrhi::Object((__bridge void*)m_MtlDevice);

                    m_NvrhiDevice = nvrhi::metal::createDevice(desc);
                    if (!m_NvrhiDevice)
                    {
                        outIsUnavailable = false;
                        outSkipOrError = "nvrhi::metal::createDevice returned null";
                        return false;
                    }

                    m_DeviceInfo.backendName = "metal";
                    m_DeviceInfo.deviceName = m_MtlDevice.name ? m_MtlDevice.name.UTF8String : "MetalDevice";
                    m_DeviceInfo.driverName = "Apple Metal";
                    NSString* osVer = [[NSProcessInfo processInfo] operatingSystemVersionString];
                    m_DeviceInfo.driverVersion = osVer ? osVer.UTF8String : "Metal";
                    m_DeviceInfo.osName = "macOS";
#ifdef UAA_BUILD_CONFIG
                    m_DeviceInfo.buildConfig = UAA_BUILD_CONFIG;
#endif
                    m_DeviceInfo.nativeValidation = nativeValidation;
                    return true;
                }
            }

        private:
            id<MTLDevice> m_MtlDevice = nil;
            nvrhi::metal::DeviceHandle m_NvrhiDevice;
            DeviceInfo m_DeviceInfo;
        };
    } // namespace

    std::unique_ptr<IBackendContext> createMetalBackend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        auto ctx = std::make_unique<MetalBackendContext>();
        if (!ctx->init(config, messageCallback, outSkipOrError, outIsUnavailable))
        {
            return nullptr;
        }
        return ctx;
    }

    static void testMetalNative(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::METAL)
        {
            UAA_SKIP("NAT.Metal requires the Metal backend");
        }

        nvrhi::IDevice* device = ctx.fixture->device();
        auto* metalDev = dynamic_cast<nvrhi::metal::IDevice*>(ctx.fixture->rawDevice());
        UAA_CHECK(metalDev != nullptr, "Raw device does not implement nvrhi::metal::IDevice");

        @autoreleasepool
        {
            id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)metalDev->getNativeDevice().pointer;
            id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)metalDev->getNativeCommandQueue().pointer;
            UAA_CHECK(mtlDevice != nil, "getNativeDevice() returned nil");
            UAA_CHECK(mtlQueue != nil, "getNativeCommandQueue() returned nil");
            UAA_CHECK(
                device->getNativeObject(nvrhi::ObjectTypes::Metal_Device).pointer == (__bridge void*)mtlDevice,
                "getNativeObject(Metal_Device) must match getNativeDevice()");
            UAA_CHECK(
                device->getNativeObject(nvrhi::ObjectTypes::Metal_CommandQueue).pointer == (__bridge void*)mtlQueue,
                "getNativeObject(Metal_CommandQueue) must match getNativeCommandQueue()");

            const uint64_t memSize = metalDev->getNativeDeviceMemorySize();
            UAA_CHECK(memSize > 0, "getNativeDeviceMemorySize() must be > 0");

            const uint64_t allocBefore = metalDev->getNativeAllocatedSize();
            {
                nvrhi::BufferDesc largeDesc;
                largeDesc.byteSize = 4ull * 1024ull * 1024ull;
                largeDesc.cpuAccess = nvrhi::CpuAccessMode::None;
                largeDesc.debugName = "uaa_metal_alloc_probe";
                nvrhi::BufferHandle largeBuf = device->createBuffer(largeDesc);
                UAA_CHECK(largeBuf != nullptr, "Failed to create 4 MiB Metal probe buffer");

                const uint64_t allocDuring = metalDev->getNativeAllocatedSize();
                UAA_CHECK(allocDuring > 0, "getNativeAllocatedSize() must be > 0 after buffer allocation");
                UAA_CHECK(allocDuring >= allocBefore,
                    "getNativeAllocatedSize() decreased unexpectedly after buffer allocation");

                std::ostringstream noteOss;
                noteOss << "recommendedMaxWorkingSet=" << memSize
                        << " allocatedBefore=" << allocBefore
                        << " allocatedDuring=" << allocDuring;
                ctx.note(noteOss.str());
            }

            // Test imported MTLBuffer.
            {
                constexpr uint64_t kBufSize = 256;
                const std::vector<uint8_t> initBytes = TestFixture::makeDeterministicBytes(kBufSize, ctx.seed() ^ 0x4D544Cull);
                NativeBufferImport bufImport;
                UAA_CHECK(
                    ctx.fixture->backend()->createNativeImportBuffer(kBufSize, initBytes.data(), bufImport),
                    "createNativeImportBuffer failed on Metal");
                UAA_CHECK_EQ(bufImport.objectType, nvrhi::ObjectTypes::Metal_Buffer,
                    "Metal import buffer type mismatch");

                nvrhi::BufferDesc importDesc;
                importDesc.byteSize = kBufSize;
                importDesc.initialState = nvrhi::ResourceStates::CopySource;
                importDesc.keepInitialState = true;
                importDesc.debugName = "uaa_metal_imported_buf";
                nvrhi::BufferHandle imported = device->createHandleForNativeBuffer(
                    bufImport.objectType, bufImport.handle, importDesc);
                UAA_CHECK(imported != nullptr, "createHandleForNativeBuffer failed on Metal");
                UAA_CHECK(
                    imported->getNativeObject(nvrhi::ObjectTypes::Metal_Buffer).pointer == bufImport.handle.pointer,
                    "Imported buffer native object pointer mismatch");

                std::vector<uint8_t> readbackBytes(kBufSize, 0);
                ctx.fixture->readbackBuffer(imported, readbackBytes.data(), kBufSize);
                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(readbackBytes.data(), initBytes.data(), kBufSize, diff), diff);

                imported = nullptr;
                device->runGarbageCollection();
                if (bufImport.release)
                {
                    bufImport.release();
                }
            }

            // Test imported MTLTexture.
            {
                constexpr uint32_t kW = 32;
                constexpr uint32_t kH = 32;
                NativeTextureImport texImport;
                UAA_CHECK(
                    ctx.fixture->backend()->createNativeImportTexture(kW, kH, nvrhi::Format::RGBA8_UNORM, texImport),
                    "createNativeImportTexture failed on Metal");
                UAA_CHECK_EQ(texImport.objectType, nvrhi::ObjectTypes::Metal_Texture,
                    "Metal import texture type mismatch");

                nvrhi::TextureHandle importedTex = device->createHandleForNativeTexture(
                    texImport.objectType, texImport.handle, texImport.desc);
                UAA_CHECK(importedTex != nullptr, "createHandleForNativeTexture failed on Metal");
                UAA_CHECK(
                    importedTex->getNativeObject(nvrhi::ObjectTypes::Metal_Texture).pointer == texImport.handle.pointer,
                    "Imported texture native object pointer mismatch");

                const std::vector<uint8_t> pixels = TestFixture::makeDeterministicBytes(size_t(kW) * kH * 4u, ctx.seed() ^ 0x77u);
                ctx.fixture->uploadTexture2D(importedTex, 0, 0, pixels.data());
                std::vector<uint8_t> rbPixels;
                ctx.fixture->readbackTexture2D(importedTex, 0, 0, rbPixels);
                UAA_CHECK_EQ(rbPixels.size(), pixels.size(), "Readback texture byte size mismatch");
                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(rbPixels.data(), pixels.data(), pixels.size(), diff), diff);

                importedTex = nullptr;
                device->runGarbageCollection();
                if (texImport.release)
                {
                    texImport.release();
                }
            }

            // Test ObjectTypes::Metal_Heap.
            {
                nvrhi::HeapDesc heapDesc;
                heapDesc.type = nvrhi::HeapType::DeviceLocal;
                heapDesc.capacity = 2ull * 1024ull * 1024ull;
                heapDesc.debugName = "uaa_metal_heap";
                nvrhi::HeapHandle heap = device->createHeap(heapDesc);
                UAA_CHECK(heap != nullptr, "createHeap failed on Metal");

                id<MTLHeap> mtlHeap = (__bridge id<MTLHeap>)heap->getNativeObject(nvrhi::ObjectTypes::Metal_Heap).pointer;
                UAA_CHECK(mtlHeap != nil, "getNativeObject(Metal_Heap) returned nil");
                UAA_CHECK(mtlHeap.size >= heapDesc.capacity, "Native MTLHeap size smaller than requested capacity");
            }

            // Test ObjectTypes::Metal_ShaderLibrary.
            {
                nvrhi::ShaderHandle shader = ctx.fixture->createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
                UAA_CHECK(shader != nullptr, "createShader failed on Metal");
                id<MTLLibrary> mtlLib =
                    (__bridge id<MTLLibrary>)shader->getNativeObject(nvrhi::ObjectTypes::Metal_ShaderLibrary).pointer;
                UAA_CHECK(mtlLib != nil, "getNativeObject(Metal_ShaderLibrary) returned nil");
            }

            // Test ObjectTypes::Metal_CommandBuffer (verify no double submission).
            {
                nvrhi::BufferDesc bDesc;
                bDesc.byteSize = 64;
                bDesc.initialState = nvrhi::ResourceStates::CopyDest;
                bDesc.keepInitialState = true;
                bDesc.debugName = "uaa_metal_cmdbuf_target";
                nvrhi::BufferHandle buf = device->createBuffer(bDesc);
                UAA_CHECK(buf != nullptr, "createBuffer failed for Metal_CommandBuffer test");

                const std::vector<uint8_t> payload = TestFixture::makeDeterministicBytes(64, ctx.seed() ^ 0xCB01u);

                nvrhi::CommandListHandle cmd = device->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed");
                cmd->open();
                id<MTLCommandBuffer> mtlCmdBuf =
                    (__bridge id<MTLCommandBuffer>)cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer;
                UAA_CHECK(mtlCmdBuf != nil, "getNativeObject(Metal_CommandBuffer) returned nil while open");
                UAA_CHECK_EQ(
                    static_cast<uint64_t>(mtlCmdBuf.status),
                    static_cast<uint64_t>(MTLCommandBufferStatusNotEnqueued),
                    "MTLCommandBuffer should not be committed before executeCommandList");

                cmd->writeBuffer(buf, payload.data(), payload.size());
                cmd->close();
                UAA_CHECK(
                    cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer == (__bridge void*)mtlCmdBuf,
                    "MTLCommandBuffer changed unexpectedly after close()");

                device->executeCommandList(cmd);
                UAA_CHECK(device->waitForIdle(), "waitForIdle failed after executing Metal command buffer");
                UAA_CHECK(
                    cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer == nullptr,
                    "CommandList should release its native MTLCommandBuffer upon submission");
                UAA_CHECK_EQ(
                    static_cast<uint64_t>(mtlCmdBuf.status),
                    static_cast<uint64_t>(MTLCommandBufferStatusCompleted),
                    "Submitted MTLCommandBuffer did not reach Completed status");
                UAA_CHECK(mtlCmdBuf.error == nil, "Submitted MTLCommandBuffer reported an error");

                std::vector<uint8_t> actual(64, 0);
                ctx.fixture->readbackBuffer(buf, actual.data(), actual.size());
                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), payload.data(), payload.size(), diff), diff);
            }

            // Test MetalGpuGate wait/signal/reset.
            {
                std::unique_ptr<IGpuGate> gate = ctx.fixture->backend()->createGpuGate();
                UAA_CHECK(gate != nullptr, "createGpuGate returned null on Metal");

                nvrhi::BufferDesc dstDesc;
                dstDesc.byteSize = 128;
                dstDesc.initialState = nvrhi::ResourceStates::CopyDest;
                dstDesc.keepInitialState = true;
                dstDesc.debugName = "uaa_metal_gate_dst";
                nvrhi::BufferHandle dstBuf = device->createBuffer(dstDesc);
                UAA_CHECK(dstBuf != nullptr, "Failed to create MetalGpuGate destination buffer");

                const std::vector<uint8_t> pattern1 = TestFixture::makeDeterministicBytes(128, ctx.seed() ^ 0x1111u);
                nvrhi::CommandListHandle cmd = device->createCommandList();
                cmd->open();
                cmd->writeBuffer(dstBuf, pattern1.data(), pattern1.size());
                cmd->close();

                UAA_CHECK(gate->enqueueWait(nvrhi::CommandQueue::Graphics), "MetalGpuGate::enqueueWait failed");
                device->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
                gate->signal();
                UAA_CHECK(device->waitForIdle(), "waitForIdle failed after MetalGpuGate::signal()");

                std::vector<uint8_t> read1(128, 0);
                ctx.fixture->readbackBuffer(dstBuf, read1.data(), read1.size());
                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(read1.data(), pattern1.data(), pattern1.size(), diff), diff);

                // Reset and verify a second wait/signal cycle.
                gate->reset();
                const std::vector<uint8_t> pattern2 = TestFixture::makeDeterministicBytes(128, ctx.seed() ^ 0x2222u);
                cmd->open();
                cmd->writeBuffer(dstBuf, pattern2.data(), pattern2.size());
                cmd->close();

                UAA_CHECK(gate->enqueueWait(nvrhi::CommandQueue::Graphics), "MetalGpuGate::enqueueWait after reset failed");
                device->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
                gate->signal();
                UAA_CHECK(device->waitForIdle(), "waitForIdle failed after second MetalGpuGate::signal()");

                std::vector<uint8_t> read2(128, 0);
                ctx.fixture->readbackBuffer(dstBuf, read2.data(), read2.size());
                UAA_CHECK(TestFixture::checkBytesEqual(read2.data(), pattern2.data(), pattern2.size(), diff), diff);
            }
        }
    }

    // A dedicated device is released without the fixture's waitForIdle call.
    static void testMetalDeviceTeardown(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::METAL)
            UAA_SKIP("SUB.MetalTeardown requires the Metal backend");

        for (uint32_t cycle = 0; cycle < 6; ++cycle)
        {
            @autoreleasepool
            {
                nvrhi::metal::DeviceDesc desc;
                desc.errorCB = &ctx.fixture->messages();
                nvrhi::metal::DeviceHandle raw = nvrhi::metal::createDevice(desc);
                UAA_CHECK(raw != nullptr, "metal::createDevice returned null");
                nvrhi::DeviceHandle device = raw;
#if UAA_WITH_VALIDATION
                if (ctx.fixture->isValidationActive())
                    device = nvrhi::validation::createValidationLayer(raw);
#endif
                UAA_CHECK(device != nullptr, "failed to create active device");
                if (cycle == 0)
                {
                    device = nullptr;
                    raw = nullptr;
                    continue;
                }

                nvrhi::BufferDesc bufferDesc;
                bufferDesc.setByteSize(16)
                    .setInitialState(nvrhi::ResourceStates::CopyDest)
                    .setKeepInitialState(true)
                    .setDebugName("teardown_output");
                nvrhi::BufferHandle buffer = device->createBuffer(bufferDesc);
                nvrhi::CommandListHandle cmd = device->createCommandList();
                UAA_CHECK(buffer != nullptr && cmd != nullptr, "failed to create teardown resources");
                const uint32_t marker = 0xA11CEu + cycle;
                cmd->open();
                cmd->writeBuffer(buffer, &marker, sizeof(marker));
                cmd->close();

                id<MTLCommandBuffer> nativeCommand =
                    (__bridge id<MTLCommandBuffer>)cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer;
                id<MTLDevice> nativeDevice = (__bridge id<MTLDevice>)raw->getNativeDevice().pointer;
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)raw->getNativeCommandQueue().pointer;
                id<MTLSharedEvent> gate = [nativeDevice newSharedEvent];
                id<MTLSharedEvent> completed = [nativeDevice newSharedEvent];
                UAA_CHECK(nativeCommand && queue && gate && completed, "failed to create native teardown objects");
                [nativeCommand encodeSignalEvent:completed value:1];
                auto callbackDone = std::make_shared<std::atomic<bool>>(false);
                [nativeCommand addCompletedHandler:^(id<MTLCommandBuffer>) { callbackDone->store(true); }];

                if (cycle > 1)
                {
                    id<MTLCommandBuffer> wait = [queue commandBuffer];
                    UAA_CHECK(wait != nil, "failed to create the GPU gate command buffer");
                    [wait encodeWaitForEvent:gate value:1];
                    [wait commit];
                }

                // Signal independently of the destructor. The test runner watchdog
                // bounds the case if destruction or completion fails to return.
                std::thread signaler([gate] {
                    @autoreleasepool
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(300));
                        gate.signaledValue = 1;
                    }
                });
                const uint64_t serial = device->executeCommandList(cmd);
                const bool wasPending = cycle == 1 || completed.signaledValue == 0;
                if (cycle == 1)
                    device->waitForIdle();
                cmd = nullptr;
                buffer = nullptr;
                device = nullptr;
                raw = nullptr;
                signaler.join();

                UAA_CHECK(serial != 0, "executeCommandList returned 0");
                UAA_CHECK(wasPending, "the submission completed before the GPU gate was released");
                [nativeCommand waitUntilCompleted];
                UAA_CHECK(nativeCommand.status == MTLCommandBufferStatusCompleted, "native command buffer failed");
                UAA_CHECK_EQ(completed.signaledValue, uint64_t(1), "GPU completion marker missing");
                UAA_CHECK(ctx.fixture->waitWithWatchdog([&] { return callbackDone->load(); }, 3000),
                    "native completion callback did not run");
            }
        }
    }

    UAA_REGISTER_TEST("NAT.Metal", "native,metal", testMetalNative);
    static void testMetalMissingCommandBuffer(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::METAL)
            UAA_SKIP("SUB.MetalMissingCommandBuffer requires the Metal backend");

        ExpectErrorsGuard guard(*ctx.fixture);
        nvrhi::metal::DeviceDesc desc;
        desc.errorCB = &ctx.fixture->messages();
        nvrhi::metal::DeviceHandle device = nvrhi::metal::createDevice(desc);
        UAA_CHECK(device != nullptr, "metal::createDevice returned null");
        nvrhi::CommandListHandle first = device->createCommandList();
        nvrhi::CommandListHandle missing = device->createCommandList();
        nvrhi::CommandListHandle last = device->createCommandList();
        UAA_CHECK(first && missing && last, "command list creation failed");
        first->open();
        first->close();
        last->open();
        last->close();
        id<MTLCommandBuffer> firstNative =
            (__bridge id<MTLCommandBuffer>)first->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer;
        id<MTLCommandBuffer> lastNative =
            (__bridge id<MTLCommandBuffer>)last->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer).pointer;
        UAA_CHECK(firstNative && lastNative, "recorded command buffers are missing");

        nvrhi::ICommandList* batch[] = { first, missing, last };
        const uint64_t serial = device->executeCommandLists(batch, 3);
        const bool idle = device->waitForIdle();
        UAA_CHECK_EQ(serial, uint64_t(0), "a batch with a missing command buffer must fail");
        UAA_CHECK(idle, "the earlier submission did not complete");
        UAA_CHECK_EQ(firstNative.status, MTLCommandBufferStatusCompleted, "the first command buffer did not complete");
        UAA_CHECK_EQ(lastNative.status, MTLCommandBufferStatusNotEnqueued, "the batch continued after the missing command buffer");
        UAA_CHECK_EQ(guard.errorCount(), size_t(1), "the missing command buffer was not reported once");
        UAA_CHECK(guard.hasMessageContaining("command list has no Metal command buffer"), "the missing buffer diagnostic was not reported");
    }

    UAA_REGISTER_TEST("SUB.MetalTeardown", "core,metal,isolated", testMetalDeviceTeardown);
    UAA_REGISTER_TEST("SUB.MetalMissingCommandBuffer", "core,metal", testMetalMissingCommandBuffer);
} // namespace nvrhi::uaa

#endif // defined(UAA_WITH_METAL) && UAA_WITH_METAL
