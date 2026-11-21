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

#include "uaa_fixture.h"
#include "uaa_runner.h"

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
#include <nvrhi/metal.h>
#endif

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct TexFillParams
        {
            float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mode = 0;
            uint32_t pad = 0;
        };
        static_assert(sizeof(TexFillParams) == 32, "TexFillParams size mismatch");

        struct SamplerTestParams
        {
            float uvOrigin[2] = {0.0f, 0.0f};
            float uvStep[2] = {0.0f, 0.0f};
            float lod = 0.0f;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mode = 0;
        };
        static_assert(sizeof(SamplerTestParams) == 32, "SamplerTestParams size mismatch");

        struct TexFillBundle
        {
            ShaderHandle shader;
            BindingLayoutHandle layout;
            ComputePipelineHandle pipeline;
        };

        struct SamplerBundle
        {
            ShaderHandle shader;
            BindingLayoutHandle layout;
            ComputePipelineHandle pipeline;
            SamplerHandle sampler;
        };

        uint64_t alignUp(uint64_t value, uint64_t alignment)
        {
            if (alignment <= 1u)
            {
                return value;
            }
            const uint64_t rem = value % alignment;
            return rem == 0u ? value : (value + (alignment - rem));
        }

        bool isPowerOfTwo(uint64_t v)
        {
            return v != 0u && (v & (v - 1u)) == 0u;
        }

        TexFillBundle createTexFillBundle(TestContext& ctx, TestFixture& fix)
        {
            TexFillBundle bundle;
            bundle.shader = fix.createShader("cs_texture_uav_fill", ShaderType::Compute);
            UAA_CHECK(bundle.shader != nullptr, "cs_texture_uav_fill shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::Texture_UAV(0))
                .addItem(BindingLayoutItem::ConstantBuffer(1));
            bundle.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "createBindingLayout for cs_texture_uav_fill returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(bundle.shader)
                .addBindingLayout(bundle.layout);
            bundle.pipeline = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(bundle.pipeline != nullptr, "createComputePipeline for cs_texture_uav_fill returned null");
            return bundle;
        }

        SamplerBundle createSamplerBundle(TestContext& ctx, TestFixture& fix)
        {
            SamplerBundle bundle;
            bundle.shader = fix.createShader("cs_sampler_test", ShaderType::Compute);
            UAA_CHECK(bundle.shader != nullptr, "cs_sampler_test shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::Texture_SRV(0))
                .addItem(BindingLayoutItem::Sampler(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                .addItem(BindingLayoutItem::ConstantBuffer(3));
            bundle.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "createBindingLayout for cs_sampler_test returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(bundle.shader)
                .addBindingLayout(bundle.layout);
            bundle.pipeline = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(bundle.pipeline != nullptr, "createComputePipeline for cs_sampler_test returned null");

            SamplerDesc sampDesc;
            sampDesc.setAllFilters(false)
                .setAllAddressModes(SamplerAddressMode::ClampToEdge);
            bundle.sampler = fix.device()->createSampler(sampDesc);
            UAA_CHECK(bundle.sampler != nullptr, "createSampler returned null");
            return bundle;
        }

        std::vector<uint8_t> makeSolidRgba8(uint32_t width, uint32_t height, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            std::vector<uint8_t> pixels(size_t(width) * height * 4u);
            for (size_t i = 0; i < size_t(width) * height; ++i)
            {
                pixels[i * 4u + 0u] = r;
                pixels[i * 4u + 1u] = g;
                pixels[i * 4u + 2u] = b;
                pixels[i * 4u + 3u] = a;
            }
            return pixels;
        }

        void verifySampledFloat4Buffer(
            TestContext& ctx,
            TestFixture& fix,
            IBuffer* outBuf,
            uint32_t width,
            uint32_t height,
            float r,
            float g,
            float b,
            float a,
            const char* label)
        {
            const size_t floatCount = size_t(width) * height * 4u;
            std::vector<float> actual(floatCount, 0.0f);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(float));

            constexpr float kEps = 1.0f / 255.0f + 1e-4f;
            for (size_t i = 0; i < size_t(width) * height; ++i)
            {
                UAA_CHECK_NEAR(actual[i * 4u + 0u], r, kEps, std::string(label) + " R mismatch");
                UAA_CHECK_NEAR(actual[i * 4u + 1u], g, kEps, std::string(label) + " G mismatch");
                UAA_CHECK_NEAR(actual[i * 4u + 2u], b, kEps, std::string(label) + " B mismatch");
                UAA_CHECK_NEAR(actual[i * 4u + 3u], a, kEps, std::string(label) + " A mismatch");
            }
        }

        void readbackSubresourceTracked(
            TestContext& ctx,
            TestFixture& fix,
            ITexture* src,
            uint32_t slice,
            uint32_t mip,
            ResourceStates priorState,
            std::vector<float>& outTightPixels)
        {
            IDevice* dev = fix.device();
            const TextureDesc& srcDesc = src->getDesc();
            const uint32_t mipW = std::max(1u, srcDesc.width >> mip);
            const uint32_t mipH = std::max(1u, srcDesc.height >> mip);
            const size_t tightRow = size_t(mipW) * 4u * sizeof(float);

            TextureDesc stgDesc;
            stgDesc.setWidth(srcDesc.width)
                .setHeight(srcDesc.height)
                .setMipLevels(srcDesc.mipLevels)
                .setFormat(srcDesc.format)
                .setDebugName("heap_subres_stg");
            StagingTextureHandle stg = dev->createStagingTexture(stgDesc, CpuAccessMode::Read);
            UAA_CHECK(stg != nullptr, "createStagingTexture returned null");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->beginTrackingTextureState(src, TextureSubresourceSet(mip, 1, slice, 1), priorState);
            cmd->copyTexture(
                stg,
                TextureSlice().setMipLevel(mip).setArraySlice(0),
                src,
                TextureSlice().setMipLevel(mip).setArraySlice(slice));
            cmd->close();
            fix.submitAndWait(cmd);

            size_t rowPitch = 0;
            const void* mapped = dev->mapStagingTexture(
                stg,
                TextureSlice().setMipLevel(mip).setArraySlice(0),
                CpuAccessMode::Read,
                &rowPitch);
            UAA_CHECK(mapped != nullptr, "mapStagingTexture returned null");
            UAA_CHECK(rowPitch >= tightRow, "staging rowPitch smaller than tight row");

            outTightPixels.resize(size_t(mipW) * mipH * 4u);
            const auto* srcBytes = static_cast<const uint8_t*>(mapped);
            for (uint32_t r = 0; r < mipH; ++r)
            {
                std::memcpy(reinterpret_cast<uint8_t*>(outTightPixels.data()) + size_t(r) * tightRow,
                    srcBytes + size_t(r) * rowPitch, tightRow);
            }
            dev->unmapStagingTexture(stg);
        }

        // Tests creation of DeviceLocal, Upload, and Readback heaps and checks their descriptors and native objects.
        void testHeapCreate(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint64_t kCapacity = 1024ull * 1024ull;

            const std::array<HeapType, 3> types = {
                HeapType::DeviceLocal,
                HeapType::Upload,
                HeapType::Readback
            };

            for (HeapType type : types)
            {
                HeapDesc desc;
                desc.setCapacity(kCapacity)
                    .setType(type)
                    .setDebugName("uaa_heap_create");

                HeapHandle heap = dev->createHeap(desc);
                UAA_CHECK(heap != nullptr, "createHeap returned null for HeapType " + std::to_string(static_cast<int>(type)));
                UAA_CHECK_EQ(heap->getDesc().capacity, kCapacity, "heap capacity mismatch");
                UAA_CHECK_EQ(heap->getDesc().type, type, "heap type mismatch");

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
                if (fix.api() == GraphicsAPI::METAL)
                {
                    UAA_CHECK(heap->getNativeObject(ObjectTypes::Metal_Heap).pointer != nullptr,
                        "Metal heap must expose ObjectTypes::Metal_Heap");
                }
#endif
            }
        }

        // Tests memory requirement queries on unbound virtual buffers, textures, and acceleration structures.
        void testHeapRequirements(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();

            BufferDesc bufDesc;
            bufDesc.setByteSize(65536u)
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setIsVirtual(true)
                .setDebugName("heap_req_vbuf");
            BufferHandle vbuf = dev->createBuffer(bufDesc);
            UAA_CHECK(vbuf != nullptr, "createBuffer virtual returned null");

            const MemoryRequirements bufReq = dev->getBufferMemoryRequirements(vbuf);
            UAA_CHECK(bufReq.size >= bufDesc.byteSize, "virtual buffer size requirement must be >= byteSize");
            UAA_CHECK(bufReq.alignment > 0u, "virtual buffer alignment must be > 0");
            UAA_CHECK(isPowerOfTwo(bufReq.alignment), "virtual buffer alignment must be a power of two");

            MemoryRequirements bufResReq{};
            if (vbuf->queryMemoryRequirements(bufResReq))
            {
                UAA_CHECK_EQ(bufResReq.size, bufReq.size, "IResource::queryMemoryRequirements buffer size mismatch");
                UAA_CHECK_EQ(bufResReq.alignment, bufReq.alignment, "IResource::queryMemoryRequirements buffer alignment mismatch");
            }
            else if (fix.api() == GraphicsAPI::VULKAN || fix.api() == GraphicsAPI::D3D12)
            {
                UAA_CHECK(false, "queryMemoryRequirements must succeed for virtual buffers on Vulkan/D3D12");
            }

            TextureDesc texDesc;
            texDesc.setWidth(64)
                .setHeight(64)
                .setMipLevels(3)
                .setFormat(Format::RGBA8_UNORM)
                .setIsUAV(true)
                .setIsVirtual(true)
                .setDebugName("heap_req_vtex");
            TextureHandle vtex = dev->createTexture(texDesc);
            UAA_CHECK(vtex != nullptr, "createTexture virtual returned null");

            const MemoryRequirements texReq = dev->getTextureMemoryRequirements(vtex);
            UAA_CHECK(texReq.size > 0u, "virtual texture size requirement must be > 0");
            UAA_CHECK(texReq.alignment > 0u, "virtual texture alignment must be > 0");
            UAA_CHECK(isPowerOfTwo(texReq.alignment), "virtual texture alignment must be a power of two");

            if (fix.features().supports(Feature::RayTracingAccelStruct))
            {
                rt::AccelStructDesc asDesc;
                asDesc.setTopLevelMaxInstances(8)
                    .setIsVirtual(true)
                    .setDebugName("heap_req_vas");
                rt::AccelStructHandle vas = dev->createAccelStruct(asDesc);
                UAA_CHECK(vas != nullptr, "createAccelStruct virtual returned null");

                const MemoryRequirements asReq = dev->getAccelStructMemoryRequirements(vas);
                UAA_CHECK(asReq.size > 0u, "virtual accel struct size requirement must be > 0");
                UAA_CHECK(asReq.alignment > 0u, "virtual accel struct alignment must be > 0");
                UAA_CHECK(isPowerOfTwo(asReq.alignment), "virtual accel struct alignment must be a power of two");
            }
        }

        // Tests placing virtual textures and buffers at offset 0, an aligned offset, and the exact heap end.
        void testHeapPlacement(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint32_t kTexSize = 16;

            // Place three virtual textures at offset 0, aligned middle offset, and exact heap end.
            std::array<TextureHandle, 3> textures;
            std::array<MemoryRequirements, 3> texReqs;
            for (size_t i = 0; i < textures.size(); ++i)
            {
                TextureDesc desc;
                desc.setWidth(kTexSize)
                    .setHeight(kTexSize)
                    .setFormat(Format::RGBA8_UNORM)
                    .setIsUAV(true)
                    .setIsVirtual(true)
                    .setInitialState(ResourceStates::CopyDest)
                    .setKeepInitialState(true)
                    .setDebugName("heap_place_tex_" + std::to_string(i));
                textures[i] = dev->createTexture(desc);
                UAA_CHECK(textures[i] != nullptr, "createTexture virtual returned null");
                texReqs[i] = dev->getTextureMemoryRequirements(textures[i]);
                UAA_CHECK(texReqs[i].size > 0u && texReqs[i].alignment > 0u, "invalid virtual texture requirements");
            }

            const uint64_t texOff0 = 0u;
            const uint64_t texOff1 = std::max<uint64_t>(texReqs[1].alignment, alignUp(texReqs[0].size, texReqs[1].alignment));
            const uint64_t texOff2 = alignUp(texOff1 + texReqs[1].size, texReqs[2].alignment);
            const uint64_t texHeapCap = texOff2 + texReqs[2].size;

            HeapHandle texHeap = dev->createHeap(HeapDesc()
                .setCapacity(texHeapCap)
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_place_tex_heap"));
            UAA_CHECK(texHeap != nullptr, "createHeap for textures returned null");

            const std::array<uint64_t, 3> texOffsets = {texOff0, texOff1, texOff2};
            for (size_t i = 0; i < textures.size(); ++i)
            {
                const bool bound = dev->bindTextureMemory(textures[i], texHeap, texOffsets[i]);
                UAA_CHECK(bound, "bindTextureMemory failed at offset " + std::to_string(texOffsets[i]));

                const uint8_t r = static_cast<uint8_t>(40u + i * 70u);
                const uint8_t g = static_cast<uint8_t>(80u + i * 50u);
                const uint8_t b = static_cast<uint8_t>(120u + i * 40u);
                const std::vector<uint8_t> pixels = makeSolidRgba8(kTexSize, kTexSize, r, g, b, 255u);
                fix.uploadTexture2D(textures[i], 0, 0, pixels.data(), size_t(kTexSize) * 4u);

                std::vector<uint8_t> readPixels;
                fix.readbackTexture2D(textures[i], 0, 0, readPixels);
                UAA_CHECK_EQ(readPixels.size(), pixels.size(), "placed texture readback size mismatch");
                std::string diff;
                const bool eq = TestFixture::checkBytesEqual(readPixels.data(), pixels.data(), pixels.size(), diff);
                UAA_CHECK(eq, "placed texture readback mismatch: " + diff);
            }

            // Place three virtual buffers at offset 0, aligned middle offset, and exact heap end.
            constexpr uint32_t kWords = 64;
            std::array<BufferHandle, 3> buffers;
            std::array<MemoryRequirements, 3> bufReqs;
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                BufferDesc desc;
                desc.setByteSize(kWords * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setIsVirtual(true)
                    .setInitialState(ResourceStates::CopyDest)
                    .setKeepInitialState(true)
                    .setDebugName("heap_place_buf_" + std::to_string(i));
                buffers[i] = dev->createBuffer(desc);
                UAA_CHECK(buffers[i] != nullptr, "createBuffer virtual returned null");
                bufReqs[i] = dev->getBufferMemoryRequirements(buffers[i]);
                UAA_CHECK(bufReqs[i].size >= desc.byteSize && bufReqs[i].alignment > 0u,
                    "invalid virtual buffer requirements");
            }

            const uint64_t bufOff0 = 0u;
            const uint64_t bufOff1 = std::max<uint64_t>(bufReqs[1].alignment, alignUp(bufReqs[0].size, bufReqs[1].alignment));
            const uint64_t bufOff2 = alignUp(bufOff1 + bufReqs[1].size, bufReqs[2].alignment);
            const uint64_t bufHeapCap = bufOff2 + bufReqs[2].size;

            HeapHandle bufHeap = dev->createHeap(HeapDesc()
                .setCapacity(bufHeapCap)
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_place_buf_heap"));
            UAA_CHECK(bufHeap != nullptr, "createHeap for buffers returned null");

            const std::array<uint64_t, 3> bufOffsets = {bufOff0, bufOff1, bufOff2};
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                const bool bound = dev->bindBufferMemory(buffers[i], bufHeap, bufOffsets[i]);
                UAA_CHECK(bound, "bindBufferMemory failed at offset " + std::to_string(bufOffsets[i]));

                const std::vector<uint32_t> words = TestFixture::makeDeterministicWords(kWords, ctx.seed() ^ (0x900u + i));
                fix.uploadBuffer(buffers[i], words.data(), words.size() * sizeof(uint32_t));

                std::vector<uint32_t> readWords(kWords, 0u);
                fix.readbackBuffer(buffers[i], readWords.data(), readWords.size() * sizeof(uint32_t));
                std::string diff;
                const bool eq = TestFixture::checkBytesEqual(
                    readWords.data(), words.data(), words.size() * sizeof(uint32_t), diff);
                UAA_CHECK(eq, "placed buffer readback mismatch: " + diff);
            }
        }

        // Tests rejection of invalid bind*Memory calls and verifies the unbound resource stays usable.
        void testHeapReject(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint32_t kTexSize = 32;

            TextureDesc vtexDesc;
            vtexDesc.setWidth(kTexSize)
                .setHeight(kTexSize)
                .setFormat(Format::RGBA8_UNORM)
                .setIsUAV(true)
                .setIsVirtual(true)
                .setInitialState(ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("heap_reject_vtex");
            TextureHandle vtex = dev->createTexture(vtexDesc);
            UAA_CHECK(vtex != nullptr, "createTexture virtual returned null");

            const MemoryRequirements req = dev->getTextureMemoryRequirements(vtex);
            UAA_CHECK(req.size > 0u && req.alignment > 0u, "invalid virtual texture requirements");

            const uint64_t heapCap = alignUp(req.size, req.alignment);
            HeapHandle heap = dev->createHeap(HeapDesc()
                .setCapacity(heapCap)
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_reject_heap"));
            UAA_CHECK(heap != nullptr, "createHeap returned null");

            TextureDesc nonVirtualTexDesc = vtexDesc;
            nonVirtualTexDesc.setIsVirtual(false)
                .setDebugName("heap_reject_nonvirtual_tex");
            TextureHandle nonVirtualTex = dev->createTexture(nonVirtualTexDesc);
            UAA_CHECK(nonVirtualTex != nullptr, "createTexture non-virtual returned null");

            BufferDesc nonVirtualBufDesc;
            nonVirtualBufDesc.setByteSize(1024u)
                .setIsVirtual(false)
                .setDebugName("heap_reject_nonvirtual_buf");
            BufferHandle nonVirtualBuf = dev->createBuffer(nonVirtualBufDesc);
            UAA_CHECK(nonVirtualBuf != nullptr, "createBuffer non-virtual returned null");

            {
                ExpectErrorsGuard guard(fix);

                // Non-virtual resources must be rejected by bind*Memory on every backend.
                UAA_CHECK(!dev->bindTextureMemory(nonVirtualTex, heap, 0u),
                    "bindTextureMemory must reject non-virtual texture");
                UAA_CHECK(!dev->bindBufferMemory(nonVirtualBuf, heap, 0u),
                    "bindBufferMemory must reject non-virtual buffer");

                // Misaligned offset and range overflow are checked by NVRHI validation and Metal heaps;
                // raw Vulkan/D3D12 do not bounds-check vkBindImageMemory without validation.
                if (fix.isValidationActive() || fix.api() == GraphicsAPI::METAL)
                {
                    if (req.alignment > 1u)
                    {
                        const uint64_t misaligned = req.alignment / 2u;
                        UAA_CHECK(!dev->bindTextureMemory(vtex, heap, misaligned),
                            "bindTextureMemory must reject misaligned offset");
                    }

                    const uint64_t overflowOffset = alignUp(heapCap - req.size / 2u, req.alignment);
                    const uint64_t safeOverflowOffset = (overflowOffset + req.size > heapCap) ? overflowOffset : heapCap;
                    UAA_CHECK(!dev->bindTextureMemory(vtex, heap, safeOverflowOffset),
                        "bindTextureMemory must reject overflowing offset");

                    // Exercise arithmetic wraparound directly on Metal, without relying on validation.
                    if (fix.api() == GraphicsAPI::METAL && !fix.isValidationActive())
                    {
                        const uint64_t nearMaxTex = UINT64_MAX - UINT64_MAX % req.alignment;
                        UAA_CHECK(!dev->bindTextureMemory(vtex, heap, nearMaxTex),
                            "bindTextureMemory must reject an offset near UINT64_MAX");

                        BufferHandle vbuf = dev->createBuffer(BufferDesc()
                            .setByteSize(65536).setIsVirtual(true).setDebugName("heap_reject_wrap_buf"));
                        UAA_CHECK(vbuf != nullptr, "createBuffer virtual returned null");
                        const MemoryRequirements bufReq = dev->getBufferMemoryRequirements(vbuf);
                        UAA_CHECK(bufReq.alignment > 0, "invalid virtual buffer alignment");
                        const uint64_t nearMaxBuf = UINT64_MAX - UINT64_MAX % bufReq.alignment;
                        UAA_CHECK(!dev->bindBufferMemory(vbuf, heap, nearMaxBuf),
                            "bindBufferMemory must reject an offset near UINT64_MAX");
                    }

                    UAA_CHECK(guard.errorCount() > 0u,
                        "Expected validation/backend errors on rejected bind calls");
                }
            }

            // Verify vtex remains unbound and usable after the rejected binds.
            UAA_CHECK(dev->bindTextureMemory(vtex, heap, 0u),
                "valid bindTextureMemory after rejected attempts must succeed");

            // Double-binding an already-bound virtual texture must fail cleanly.
            {
                ExpectErrorsGuard guard(fix);
                UAA_CHECK(!dev->bindTextureMemory(vtex, heap, 0u),
                    "second bindTextureMemory on already-bound texture must return false");
            }

            const std::vector<uint8_t> pixels = makeSolidRgba8(kTexSize, kTexSize, 17, 143, 211, 255);
            fix.uploadTexture2D(vtex, 0, 0, pixels.data(), size_t(kTexSize) * 4u);

            std::vector<uint8_t> actual;
            fix.readbackTexture2D(vtex, 0, 0, actual);
            std::string diff;
            const bool eq = TestFixture::checkBytesEqual(actual.data(), pixels.data(), pixels.size(), diff);
            UAA_CHECK(eq, "HEAP.Reject placed texture readback mismatch: " + diff);
        }

        // Tests overlapping virtual textures at offset 0 across repeated alias handoffs.
        void testHeapAliasHandoff(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint32_t kSize = 16;

            const TexFillBundle fillBundle = createTexFillBundle(ctx, fix);
            const SamplerBundle sampBundle = createSamplerBundle(ctx, fix);

            TextureDesc descA;
            descA.setWidth(kSize)
                .setHeight(kSize)
                .setFormat(Format::RGBA32_FLOAT)
                .setIsUAV(true)
                .setIsVirtual(true)
                .setKeepInitialState(false)
                .setDebugName("heap_alias_texA");
            TextureHandle texA = dev->createTexture(descA);
            UAA_CHECK(texA != nullptr, "createTexture texA returned null");

            TextureDesc descB = descA;
            descB.setDebugName("heap_alias_texB");
            TextureHandle texB = dev->createTexture(descB);
            UAA_CHECK(texB != nullptr, "createTexture texB returned null");

            const MemoryRequirements reqA = dev->getTextureMemoryRequirements(texA);
            const MemoryRequirements reqB = dev->getTextureMemoryRequirements(texB);
            const uint64_t heapCap = std::max(reqA.size, reqB.size);

            HeapHandle heap = dev->createHeap(HeapDesc()
                .setCapacity(heapCap)
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_alias_heap"));
            UAA_CHECK(heap != nullptr, "createHeap returned null");

            UAA_CHECK(dev->bindTextureMemory(texA, heap, 0u), "bindTextureMemory texA failed");
            UAA_CHECK(dev->bindTextureMemory(texB, heap, 0u), "bindTextureMemory texB failed");

            auto makeOutBuf = [&](const char* name) -> BufferHandle
            {
                BufferDesc d;
                d.setByteSize(size_t(kSize) * kSize * 16u)
                    .setStructStride(16u)
                    .setCanHaveUAVs(true)
                    .setInitialState(ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true)
                    .setDebugName(name);
                BufferHandle b = dev->createBuffer(d);
                UAA_CHECK(b != nullptr, std::string("createBuffer ") + name + " returned null");
                return b;
            };

            BufferHandle outBuf1 = makeOutBuf("heap_alias_out1");
            BufferHandle outBuf2 = makeOutBuf("heap_alias_out2");
            BufferHandle outBuf3 = makeOutBuf("heap_alias_out3");

            BufferDesc fillCbDesc;
            fillCbDesc.setByteSize(sizeof(TexFillParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("heap_alias_fill_cb");
            BufferHandle fillCb = dev->createBuffer(fillCbDesc);
            UAA_CHECK(fillCb != nullptr, "createBuffer fillCb returned null");

            BufferDesc sampCbDesc;
            sampCbDesc.setByteSize(sizeof(SamplerTestParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("heap_alias_samp_cb");
            BufferHandle sampCb = dev->createBuffer(sampCbDesc);
            UAA_CHECK(sampCb != nullptr, "createBuffer sampCb returned null");

            SamplerTestParams sampParams{};
            sampParams.uvOrigin[0] = 0.5f / float(kSize);
            sampParams.uvOrigin[1] = 0.5f / float(kSize);
            sampParams.uvStep[0] = 1.0f / float(kSize);
            sampParams.uvStep[1] = 1.0f / float(kSize);
            sampParams.width = kSize;
            sampParams.height = kSize;
            fix.uploadBuffer(sampCb, &sampParams, sizeof(sampParams));

            auto runAliasPhase = [&](ITexture* tex, IBuffer* outBuf, float r, float g, float b, float a)
            {
                TexFillParams fillParams{};
                fillParams.color[0] = r;
                fillParams.color[1] = g;
                fillParams.color[2] = b;
                fillParams.color[3] = a;
                fillParams.width = kSize;
                fillParams.height = kSize;

                BindingSetHandle fillSet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::Texture_UAV(0, tex))
                        .addItem(BindingSetItem::ConstantBuffer(1, fillCb)),
                    fillBundle.layout);
                UAA_CHECK(fillSet != nullptr, "createBindingSet fillSet returned null");

                BindingSetHandle sampSet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::Texture_SRV(0, tex))
                        .addItem(BindingSetItem::Sampler(1, sampBundle.sampler))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(BindingSetItem::ConstantBuffer(3, sampCb)),
                    sampBundle.layout);
                UAA_CHECK(sampSet != nullptr, "createBindingSet sampSet returned null");

                CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->beginTrackingTextureState(tex, AllSubresources, ResourceStates::Common);
                cmd->writeBuffer(fillCb, &fillParams, sizeof(fillParams));
                cmd->setComputeState(ComputeState().setPipeline(fillBundle.pipeline).addBindingSet(fillSet));
                cmd->dispatch((kSize + 7u) / 8u, (kSize + 7u) / 8u, 1u);
                cmd->setComputeState(ComputeState().setPipeline(sampBundle.pipeline).addBindingSet(sampSet));
                cmd->dispatch((kSize + 7u) / 8u, (kSize + 7u) / 8u, 1u);
                cmd->close();
                fix.submitAndWait(cmd);
            };

            runAliasPhase(texA, outBuf1, 0.25f, 0.50f, 0.75f, 1.00f);
            runAliasPhase(texB, outBuf2, 0.90f, 0.20f, 0.10f, 1.00f);
            runAliasPhase(texA, outBuf3, 0.10f, 0.80f, 0.40f, 1.00f);

            verifySampledFloat4Buffer(ctx, fix, outBuf1, kSize, kSize, 0.25f, 0.50f, 0.75f, 1.00f, "HEAP.AliasHandoff A1");
            verifySampledFloat4Buffer(ctx, fix, outBuf2, kSize, kSize, 0.90f, 0.20f, 0.10f, 1.00f, "HEAP.AliasHandoff B");
            verifySampledFloat4Buffer(ctx, fix, outBuf3, kSize, kSize, 0.10f, 0.80f, 0.40f, 1.00f, "HEAP.AliasHandoff A2");
        }

        // Tests alias handoffs across multi-mip, multi-slice virtual 2D array textures.
        void testHeapSubresources(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint32_t kWidth = 16;
            constexpr uint32_t kHeight = 16;
            constexpr uint32_t kSlices = 2;
            constexpr uint32_t kMips = 3;

            TextureDesc descA;
            descA.setDimension(TextureDimension::Texture2DArray)
                .setWidth(kWidth)
                .setHeight(kHeight)
                .setArraySize(kSlices)
                .setMipLevels(kMips)
                .setFormat(Format::RGBA32_FLOAT)
                .setIsUAV(true)
                .setIsVirtual(true)
                .setKeepInitialState(false)
                .setDebugName("heap_subres_texA");
            TextureHandle texA = dev->createTexture(descA);
            UAA_CHECK(texA != nullptr, "createTexture texA returned null");

            TextureDesc descB = descA;
            descB.setDebugName("heap_subres_texB");
            TextureHandle texB = dev->createTexture(descB);
            UAA_CHECK(texB != nullptr, "createTexture texB returned null");

            const MemoryRequirements reqA = dev->getTextureMemoryRequirements(texA);
            const MemoryRequirements reqB = dev->getTextureMemoryRequirements(texB);
            HeapHandle heap = dev->createHeap(HeapDesc()
                .setCapacity(std::max(reqA.size, reqB.size))
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_subres_heap"));
            UAA_CHECK(heap != nullptr, "createHeap returned null");

            UAA_CHECK(dev->bindTextureMemory(texA, heap, 0u), "bindTextureMemory texA failed");
            UAA_CHECK(dev->bindTextureMemory(texB, heap, 0u), "bindTextureMemory texB failed");

            struct SubresSpec
            {
                uint32_t slice;
                uint32_t mip;
                uint8_t r;
                uint8_t g;
                uint8_t b;
                uint8_t a;
            };

            auto runSubresources = [&](ITexture* tex, const std::vector<SubresSpec>& specs, const char* label)
            {
                for (const SubresSpec& s : specs)
                {
                    const uint32_t mipW = std::max(1u, kWidth >> s.mip);
                    const uint32_t mipH = std::max(1u, kHeight >> s.mip);

                    const float expected[] = {
                        float(s.r) / 255.0f, float(s.g) / 255.0f, float(s.b) / 255.0f, float(s.a) / 255.0f
                    };

                    CommandListHandle cmd = dev->createCommandList();
                    UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                    cmd->open();
                    cmd->beginTrackingTextureState(tex, AllSubresources, ResourceStates::Common);
                    cmd->clearTextureFloat(tex, TextureSubresourceSet(s.mip, 1, s.slice, 1),
                        Color(expected[0], expected[1], expected[2], expected[3]));
                    cmd->close();
                    fix.submitAndWait(cmd);

                    std::vector<float> pixels;
                    readbackSubresourceTracked(ctx, fix, tex, s.slice, s.mip, ResourceStates::CopyDest, pixels);
                    const size_t pixelCount = size_t(mipW) * mipH;
                    UAA_CHECK_EQ(pixels.size(), pixelCount * 4u, "subresource float readback size mismatch");
                    for (size_t i = 0; i < pixelCount; ++i)
                    {
                        for (size_t c = 0; c < 4u; ++c)
                        {
                            UAA_CHECK_NEAR(pixels[i * 4u + c], expected[c], 1e-6f,
                                std::string(label) + " subresource pixel " + std::to_string(i)
                                    + " channel " + std::to_string(c) + " mismatch");
                        }
                    }
                }
            };

            runSubresources(texA, {
                {0u, 0u, 40u, 120u, 200u, 255u},
                {0u, 1u, 90u, 180u, 60u, 255u},
                {1u, 2u, 220u, 70u, 140u, 255u},
            }, "HEAP.Subresources A1");

            runSubresources(texB, {
                {0u, 1u, 15u, 230u, 180u, 255u},
                {1u, 0u, 175u, 45u, 210u, 255u},
                {1u, 1u, 60u, 95u, 240u, 255u},
            }, "HEAP.Subresources B");

            runSubresources(texA, {
                {0u, 2u, 245u, 160u, 30u, 255u},
                {1u, 0u, 35u, 210u, 115u, 255u},
            }, "HEAP.Subresources A2");
        }

        // Tests that placed resources and their backing heap stay alive until pending GPU work finishes.
        void testHeapPendingLifetime(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr uint32_t kSize = 16;

            BufferDesc outDesc;
            outDesc.setByteSize(size_t(kSize) * kSize * 16u)
                .setStructStride(16u)
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("heap_pending_out");
            BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "createBuffer outBuf returned null");

            std::unique_ptr<IGpuGate> gate = fix.backend()->createGpuGate();
            if (gate)
            {
                if (!gate->enqueueWait(CommandQueue::Graphics))
                {
                    gate.reset();
                }
            }

            {
                TexFillBundle fillBundle = createTexFillBundle(ctx, fix);
                SamplerBundle sampBundle = createSamplerBundle(ctx, fix);

                TextureDesc vtexDesc;
                vtexDesc.setWidth(kSize)
                    .setHeight(kSize)
                    .setFormat(Format::RGBA32_FLOAT)
                    .setIsUAV(true)
                    .setIsVirtual(true)
                    .setInitialState(ResourceStates::CopyDest)
                    .setKeepInitialState(true)
                    .setDebugName("heap_pending_vtex");
                TextureHandle vtex = dev->createTexture(vtexDesc);
                UAA_CHECK(vtex != nullptr, "createTexture vtex returned null");

                const MemoryRequirements req = dev->getTextureMemoryRequirements(vtex);
                HeapHandle heap = dev->createHeap(HeapDesc()
                    .setCapacity(req.size)
                    .setType(HeapType::DeviceLocal)
                    .setDebugName("heap_pending_heap"));
                UAA_CHECK(heap != nullptr, "createHeap returned null");
                UAA_CHECK(dev->bindTextureMemory(vtex, heap, 0u), "bindTextureMemory failed");

                BufferDesc fillCbDesc;
                fillCbDesc.setByteSize(sizeof(TexFillParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true)
                    .setDebugName("heap_pending_fill_cb");
                BufferHandle fillCb = dev->createBuffer(fillCbDesc);
                UAA_CHECK(fillCb != nullptr, "createBuffer fillCb returned null");

                BufferDesc sampCbDesc;
                sampCbDesc.setByteSize(sizeof(SamplerTestParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true)
                    .setDebugName("heap_pending_samp_cb");
                BufferHandle sampCb = dev->createBuffer(sampCbDesc);
                UAA_CHECK(sampCb != nullptr, "createBuffer sampCb returned null");

                TexFillParams fillParams{};
                fillParams.color[0] = 0.20f;
                fillParams.color[1] = 0.60f;
                fillParams.color[2] = 0.85f;
                fillParams.color[3] = 1.00f;
                fillParams.width = kSize;
                fillParams.height = kSize;

                SamplerTestParams sampParams{};
                sampParams.uvOrigin[0] = 0.5f / float(kSize);
                sampParams.uvOrigin[1] = 0.5f / float(kSize);
                sampParams.uvStep[0] = 1.0f / float(kSize);
                sampParams.uvStep[1] = 1.0f / float(kSize);
                sampParams.width = kSize;
                sampParams.height = kSize;

                BindingSetHandle fillSet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::Texture_UAV(0, vtex))
                        .addItem(BindingSetItem::ConstantBuffer(1, fillCb)),
                    fillBundle.layout);
                UAA_CHECK(fillSet != nullptr, "createBindingSet fillSet returned null");

                BindingSetHandle sampSet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::Texture_SRV(0, vtex))
                        .addItem(BindingSetItem::Sampler(1, sampBundle.sampler))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(BindingSetItem::ConstantBuffer(3, sampCb)),
                    sampBundle.layout);
                UAA_CHECK(sampSet != nullptr, "createBindingSet sampSet returned null");

                CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->beginTrackingTextureState(vtex, AllSubresources, ResourceStates::Common);
                cmd->writeBuffer(fillCb, &fillParams, sizeof(fillParams));
                cmd->writeBuffer(sampCb, &sampParams, sizeof(sampParams));
                cmd->setComputeState(ComputeState().setPipeline(fillBundle.pipeline).addBindingSet(fillSet));
                cmd->dispatch((kSize + 7u) / 8u, (kSize + 7u) / 8u, 1u);
                cmd->setComputeState(ComputeState().setPipeline(sampBundle.pipeline).addBindingSet(sampSet));
                cmd->dispatch((kSize + 7u) / 8u, (kSize + 7u) / 8u, 1u);
                cmd->close();

                dev->executeCommandList(cmd);
            }
            // All caller handles to vtex, heap, pipelines, shaders, binding sets, and cmd are released here.

            if (gate)
            {
                gate->signal();
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in HEAP.PendingLifetime");
            dev->runGarbageCollection();

            verifySampledFloat4Buffer(ctx, fix, outBuf, kSize, kSize, 0.20f, 0.60f, 0.85f, 1.00f, "HEAP.PendingLifetime");
        }

        // Compares NVRHI virtual resource memory requirements against native API queries where exposed.
        void testHeapNative(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::VirtualResources))
            {
                ctx.skip("Feature::VirtualResources is not supported by this backend");
            }

            IDevice* dev = fix.device();

            TextureDesc vtexDesc;
            vtexDesc.setWidth(64)
                .setHeight(64)
                .setMipLevels(4)
                .setFormat(Format::RGBA8_UNORM)
                .setIsUAV(true)
                .setIsVirtual(true)
                .setDebugName("heap_native_vtex");
            TextureHandle vtex = dev->createTexture(vtexDesc);
            UAA_CHECK(vtex != nullptr, "createTexture virtual returned null");

            const MemoryRequirements unboundTexReq = dev->getTextureMemoryRequirements(vtex);
            UAA_CHECK(unboundTexReq.size > 0u && unboundTexReq.alignment > 0u,
                "virtual texture unbound requirements must be positive");
            UAA_CHECK(isPowerOfTwo(unboundTexReq.alignment),
                "virtual texture alignment must be power of two");

            BufferDesc vbufDesc;
            vbufDesc.setByteSize(131072u)
                .setCanHaveUAVs(true)
                .setIsVirtual(true)
                .setDebugName("heap_native_vbuf");
            BufferHandle vbuf = dev->createBuffer(vbufDesc);
            UAA_CHECK(vbuf != nullptr, "createBuffer virtual returned null");

            const MemoryRequirements unboundBufReq = dev->getBufferMemoryRequirements(vbuf);
            UAA_CHECK(unboundBufReq.size >= vbufDesc.byteSize && unboundBufReq.alignment > 0u,
                "virtual buffer unbound requirements must be valid");
            UAA_CHECK(isPowerOfTwo(unboundBufReq.alignment),
                "virtual buffer alignment must be power of two");

            HeapHandle heap = dev->createHeap(HeapDesc()
                .setCapacity(unboundTexReq.size)
                .setType(HeapType::DeviceLocal)
                .setDebugName("heap_native_heap"));
            UAA_CHECK(heap != nullptr, "createHeap returned null");
            UAA_CHECK(dev->bindTextureMemory(vtex, heap, 0u), "bindTextureMemory failed");

            const MemoryRequirements boundTexReq = dev->getTextureMemoryRequirements(vtex);
            UAA_CHECK(boundTexReq.size > 0u, "bound texture requirement size must be positive");

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
            if (fix.api() == GraphicsAPI::VULKAN)
            {
                auto vkDev = static_cast<VkDevice>(dev->getNativeObject(ObjectTypes::VK_Device).pointer);
                auto vkImg = static_cast<VkImage>(vtex->getNativeObject(ObjectTypes::VK_Image).pointer);
                auto vkBuf = static_cast<VkBuffer>(vbuf->getNativeObject(ObjectTypes::VK_Buffer).pointer);
                UAA_CHECK(vkDev != nullptr && vkImg != VK_NULL_HANDLE && vkBuf != VK_NULL_HANDLE,
                    "Vulkan native handles must be non-null");

                PFN_vkGetDeviceProcAddr getDevProc = nullptr;
#ifdef _WIN32
                if (HMODULE mod = GetModuleHandleA("vulkan-1.dll"))
                {
                    getDevProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(mod, "vkGetDeviceProcAddr"));
                }
#else
                getDevProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(RTLD_DEFAULT, "vkGetDeviceProcAddr"));
                if (!getDevProc)
                {
#ifdef __APPLE__
                    void* mod = dlopen("libvulkan.1.dylib", RTLD_NOLOAD | RTLD_LOCAL);
                    if (!mod) mod = dlopen("libMoltenVK.dylib", RTLD_NOLOAD | RTLD_LOCAL);
#else
                    void* mod = dlopen("libvulkan.so.1", RTLD_NOLOAD | RTLD_LOCAL);
#endif
                    if (mod)
                    {
                        getDevProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(mod, "vkGetDeviceProcAddr"));
                    }
                }
#endif
                if (getDevProc)
                {
                    auto fnGetImgReq = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(
                        getDevProc(vkDev, "vkGetImageMemoryRequirements"));
                    auto fnGetBufReq = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(
                        getDevProc(vkDev, "vkGetBufferMemoryRequirements"));
                    if (fnGetImgReq)
                    {
                        VkMemoryRequirements vkImgReq{};
                        fnGetImgReq(vkDev, vkImg, &vkImgReq);
                        UAA_CHECK_EQ(unboundTexReq.size, static_cast<uint64_t>(vkImgReq.size),
                            "Vulkan texture memory requirement size mismatch vs vkGetImageMemoryRequirements");
                        UAA_CHECK_EQ(unboundTexReq.alignment, static_cast<uint64_t>(vkImgReq.alignment),
                            "Vulkan texture memory requirement alignment mismatch vs vkGetImageMemoryRequirements");
                    }
                    if (fnGetBufReq)
                    {
                        VkMemoryRequirements vkBufReq{};
                        fnGetBufReq(vkDev, vkBuf, &vkBufReq);
                        UAA_CHECK_EQ(unboundBufReq.size, static_cast<uint64_t>(vkBufReq.size),
                            "Vulkan buffer memory requirement size mismatch vs vkGetBufferMemoryRequirements");
                        UAA_CHECK_EQ(unboundBufReq.alignment, static_cast<uint64_t>(vkBufReq.alignment),
                            "Vulkan buffer memory requirement alignment mismatch vs vkGetBufferMemoryRequirements");
                    }
                }

                MemoryRequirements resBufReq{};
                UAA_CHECK(vbuf->queryMemoryRequirements(resBufReq), "queryMemoryRequirements must succeed on Vulkan buffer");
                UAA_CHECK_EQ(resBufReq.size, unboundBufReq.size, "Vulkan queryMemoryRequirements size mismatch");
                UAA_CHECK_EQ(resBufReq.alignment, unboundBufReq.alignment, "Vulkan queryMemoryRequirements alignment mismatch");
            }
#endif

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            if (fix.api() == GraphicsAPI::METAL)
            {
                UAA_CHECK(heap->getNativeObject(ObjectTypes::Metal_Heap).pointer != nullptr,
                    "Metal heap must expose ObjectTypes::Metal_Heap");
                UAA_CHECK(vtex->getNativeObject(ObjectTypes::Metal_Texture).pointer != nullptr,
                    "Placed Metal virtual texture must expose ObjectTypes::Metal_Texture");
            }
#endif
        }
    } // namespace

    UAA_REGISTER_TEST("HEAP.Create", "core", testHeapCreate);
    UAA_REGISTER_TEST("HEAP.Requirements", "core", testHeapRequirements);
    UAA_REGISTER_TEST("HEAP.Placement", "core", testHeapPlacement);
    UAA_REGISTER_TEST("HEAP.Reject", "core", testHeapReject);
    UAA_REGISTER_TEST("HEAP.AliasHandoff", "core", testHeapAliasHandoff);
    UAA_REGISTER_TEST("HEAP.Subresources", "core", testHeapSubresources);
    UAA_REGISTER_TEST("HEAP.PendingLifetime", "core", testHeapPendingLifetime);
    UAA_REGISTER_TEST("HEAP.Native", "core", testHeapNative);
} // namespace nvrhi::uaa
