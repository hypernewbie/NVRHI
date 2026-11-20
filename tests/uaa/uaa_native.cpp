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

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
#define VK_NO_PROTOTYPES
#include <nvrhi/vulkan.h>
#endif

#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
#include <nvrhi/metal.h>
#endif

#if (defined(UAA_WITH_D3D11) && UAA_WITH_D3D11) || (defined(UAA_WITH_DX11) && UAA_WITH_DX11)
#include <nvrhi/d3d11.h>
#endif

#if (defined(UAA_WITH_D3D12) && UAA_WITH_D3D12) || (defined(UAA_WITH_DX12) && UAA_WITH_DX12)
#include <nvrhi/d3d12.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct CopyAddParams
        {
            uint32_t count = 0;
            uint32_t addVal = 0;
            uint32_t xorVal = 0;
            uint32_t pad = 0;
        };

        struct SamplerTestParams
        {
            float uvOrigin[2] = {0.f, 0.f};
            float uvStep[2] = {0.f, 0.f};
            float lod = 0.f;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mode = 0;
        };

        bool isNonNullObject(const nvrhi::Object& obj)
        {
            return obj.pointer != nullptr || obj.integer != 0;
        }

        nvrhi::ObjectType primaryTextureObjectType(nvrhi::GraphicsAPI api)
        {
            switch (api)
            {
            case nvrhi::GraphicsAPI::VULKAN:
                return nvrhi::ObjectTypes::VK_Image;
            case nvrhi::GraphicsAPI::D3D12:
                return nvrhi::ObjectTypes::D3D12_Resource;
            case nvrhi::GraphicsAPI::D3D11:
                return nvrhi::ObjectTypes::D3D11_Resource;
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            case nvrhi::GraphicsAPI::METAL:
                return nvrhi::ObjectTypes::Metal_Texture;
#endif
            default:
                return 0;
            }
        }

        nvrhi::ObjectType primarySrvViewObjectType(nvrhi::GraphicsAPI api)
        {
            switch (api)
            {
            case nvrhi::GraphicsAPI::VULKAN:
                return nvrhi::ObjectTypes::VK_ImageView;
            case nvrhi::GraphicsAPI::D3D12:
                return nvrhi::ObjectTypes::D3D12_ShaderResourceViewGpuDescriptor;
            case nvrhi::GraphicsAPI::D3D11:
                return nvrhi::ObjectTypes::D3D11_ShaderResourceView;
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            case nvrhi::GraphicsAPI::METAL:
                return nvrhi::ObjectTypes::Metal_Texture;
#endif
            default:
                return 0;
            }
        }

        nvrhi::ObjectType primaryDsvViewObjectType(nvrhi::GraphicsAPI api)
        {
            switch (api)
            {
            case nvrhi::GraphicsAPI::VULKAN:
                return nvrhi::ObjectTypes::VK_ImageView;
            case nvrhi::GraphicsAPI::D3D12:
                return nvrhi::ObjectTypes::D3D12_DepthStencilViewDescriptor;
            case nvrhi::GraphicsAPI::D3D11:
                return nvrhi::ObjectTypes::D3D11_DepthStencilView;
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            case nvrhi::GraphicsAPI::METAL:
                return nvrhi::ObjectTypes::Metal_Texture;
#endif
            default:
                return 0;
            }
        }

        void testNativeObjects(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            const nvrhi::GraphicsAPI api = fix.api();
            constexpr nvrhi::ObjectType kBogusType = 0xdeadbeefu;

            nvrhi::BufferHandle buf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setDebugName("nat_buf"));
            nvrhi::TextureHandle tex = dev->createTexture(
                nvrhi::TextureDesc().setWidth(16).setHeight(16).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("nat_tex"));
            nvrhi::SamplerHandle sampler = dev->createSampler(nvrhi::SamplerDesc());
            nvrhi::ShaderHandle shader = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            nvrhi::HeapHandle heap = dev->createHeap(
                nvrhi::HeapDesc().setCapacity(65536).setType(nvrhi::HeapType::DeviceLocal).setDebugName("nat_heap"));
            nvrhi::CommandListHandle cmd = dev->createCommandList();

            UAA_CHECK(buf != nullptr && tex != nullptr && sampler != nullptr && shader != nullptr && cmd != nullptr,
                "Core native test resources must not be null");

            cmd->open();

            switch (api)
            {
            case nvrhi::GraphicsAPI::VULKAN:
#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::VK_Device)), "VK_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice)), "VK_PhysicalDevice must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::VK_Instance)), "VK_Instance must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::Nvrhi_VK_Device)), "Nvrhi_VK_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Graphics)), "VK_Queue(Graphics) must be non-null");
                if (fix.features().supports(nvrhi::Feature::ComputeQueue))
                {
                    UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Compute)), "VK_Queue(Compute) must be non-null");
                }
                if (fix.features().supports(nvrhi::Feature::CopyQueue))
                {
                    UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Copy)), "VK_Queue(Copy) must be non-null");
                }
                UAA_CHECK(isNonNullObject(cmd->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer)), "VK_CommandBuffer must be non-null");
                UAA_CHECK(isNonNullObject(buf->getNativeObject(nvrhi::ObjectTypes::VK_Buffer)), "VK_Buffer must be non-null");
                UAA_CHECK(isNonNullObject(buf->getNativeObject(nvrhi::ObjectTypes::VK_DeviceMemory)), "Buffer VK_DeviceMemory must be non-null");
                UAA_CHECK(isNonNullObject(tex->getNativeObject(nvrhi::ObjectTypes::VK_Image)), "VK_Image must be non-null");
                UAA_CHECK(isNonNullObject(tex->getNativeObject(nvrhi::ObjectTypes::VK_DeviceMemory)), "Texture VK_DeviceMemory must be non-null");
                UAA_CHECK(isNonNullObject(sampler->getNativeObject(nvrhi::ObjectTypes::VK_Sampler)), "VK_Sampler must be non-null");
                UAA_CHECK(isNonNullObject(shader->getNativeObject(nvrhi::ObjectTypes::VK_ShaderModule)), "VK_ShaderModule must be non-null");
#else
                UAA_CHECK(false, "Vulkan backend active without UAA_WITH_VULKAN");
#endif
                break;

            case nvrhi::GraphicsAPI::METAL:
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::Metal_Device)), "Metal_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::Metal_CommandQueue)), "Metal_CommandQueue must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::Metal_CommandQueue, nvrhi::CommandQueue::Graphics)), "Metal_CommandQueue(Graphics) must be non-null");
                UAA_CHECK(isNonNullObject(cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer)), "Metal_CommandBuffer must be non-null");
                UAA_CHECK(isNonNullObject(buf->getNativeObject(nvrhi::ObjectTypes::Metal_Buffer)), "Metal_Buffer must be non-null");
                UAA_CHECK(isNonNullObject(tex->getNativeObject(nvrhi::ObjectTypes::Metal_Texture)), "Metal_Texture must be non-null");
                UAA_CHECK(isNonNullObject(sampler->getNativeObject(nvrhi::ObjectTypes::Metal_Sampler)), "Metal_Sampler must be non-null");
                UAA_CHECK(isNonNullObject(shader->getNativeObject(nvrhi::ObjectTypes::Metal_ShaderLibrary)), "Metal_ShaderLibrary must be non-null");
                if (heap)
                {
                    UAA_CHECK(isNonNullObject(heap->getNativeObject(nvrhi::ObjectTypes::Metal_Heap)), "Metal_Heap must be non-null");
                }
#else
                UAA_CHECK(false, "Metal backend active without UAA_WITH_METAL");
#endif
                break;

            case nvrhi::GraphicsAPI::D3D12:
#if (defined(UAA_WITH_D3D12) && UAA_WITH_D3D12) || (defined(UAA_WITH_DX12) && UAA_WITH_DX12)
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::D3D12_Device)), "D3D12_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::Nvrhi_D3D12_Device)), "Nvrhi_D3D12_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics)), "D3D12_CommandQueue(Graphics) must be non-null");
                if (fix.features().supports(nvrhi::Feature::ComputeQueue))
                {
                    UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Compute)), "D3D12_CommandQueue(Compute) must be non-null");
                }
                if (fix.features().supports(nvrhi::Feature::CopyQueue))
                {
                    UAA_CHECK(isNonNullObject(dev->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Copy)), "D3D12_CommandQueue(Copy) must be non-null");
                }
                UAA_CHECK(isNonNullObject(cmd->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList)), "D3D12_GraphicsCommandList must be non-null");
                UAA_CHECK(isNonNullObject(buf->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource)), "Buffer D3D12_Resource must be non-null");
                UAA_CHECK(isNonNullObject(tex->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource)), "Texture D3D12_Resource must be non-null");
#else
                UAA_CHECK(false, "D3D12 backend active without UAA_WITH_D3D12");
#endif
                break;

            case nvrhi::GraphicsAPI::D3D11:
#if (defined(UAA_WITH_D3D11) && UAA_WITH_D3D11) || (defined(UAA_WITH_DX11) && UAA_WITH_DX11)
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::D3D11_Device)), "D3D11_Device must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::D3D11_DeviceContext)), "D3D11_DeviceContext must be non-null");
                UAA_CHECK(isNonNullObject(dev->getNativeObject(nvrhi::ObjectTypes::Nvrhi_D3D11_Device)), "Nvrhi_D3D11_Device must be non-null");
                UAA_CHECK(isNonNullObject(cmd->getNativeObject(nvrhi::ObjectTypes::D3D11_DeviceContext)), "CommandList D3D11_DeviceContext must be non-null");
                UAA_CHECK(isNonNullObject(buf->getNativeObject(nvrhi::ObjectTypes::D3D11_Buffer)), "D3D11_Buffer must be non-null");
                UAA_CHECK(isNonNullObject(tex->getNativeObject(nvrhi::ObjectTypes::D3D11_Resource)), "Texture D3D11_Resource must be non-null");
#else
                UAA_CHECK(false, "D3D11 backend active without UAA_WITH_D3D11");
#endif
                break;

            default:
                ctx.skip("Unsupported GraphicsAPI for NAT.Objects");
            }

            // Bogus ObjectType must safely return null across all resource types.
            UAA_CHECK(!isNonNullObject(dev->getNativeObject(kBogusType)), "Device getNativeObject(bogus) must return null");
            UAA_CHECK(!isNonNullObject(dev->getNativeQueue(kBogusType, nvrhi::CommandQueue::Graphics)), "getNativeQueue(bogus) must return null");
            UAA_CHECK(!isNonNullObject(cmd->getNativeObject(kBogusType)), "CommandList getNativeObject(bogus) must return null");
            UAA_CHECK(!isNonNullObject(buf->getNativeObject(kBogusType)), "Buffer getNativeObject(bogus) must return null");
            UAA_CHECK(!isNonNullObject(tex->getNativeObject(kBogusType)), "Texture getNativeObject(bogus) must return null");
            UAA_CHECK(!isNonNullObject(sampler->getNativeObject(kBogusType)), "Sampler getNativeObject(bogus) must return null");
            UAA_CHECK(!isNonNullObject(shader->getNativeObject(kBogusType)), "Shader getNativeObject(bogus) must return null");
            if (heap)
            {
                UAA_CHECK(!isNonNullObject(heap->getNativeObject(kBogusType)), "Heap getNativeObject(bogus) must return null");
            }

            cmd->close();
            cmd = nullptr;
        }

        void testBorrowedOwnership(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            nvrhi::BufferHandle buf = dev->createBuffer(nvrhi::BufferDesc().setByteSize(256).setDebugName("nat_borrow_buf"));
            nvrhi::TextureHandle tex = dev->createTexture(
                nvrhi::TextureDesc().setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("nat_borrow_tex"));
            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(buf != nullptr && tex != nullptr && cmd != nullptr, "Resources must not be null");

            cmd->open();

            const unsigned long devRefBefore = dev->GetRefCount();
            const unsigned long bufRefBefore = buf->GetRefCount();
            const unsigned long texRefBefore = tex->GetRefCount();
            const unsigned long cmdRefBefore = cmd->GetRefCount();

            nvrhi::Object nativeDev = nullptr;
            nvrhi::Object nativeBuf = nullptr;
            nvrhi::Object nativeTex = nullptr;
            nvrhi::Object nativeCmd = nullptr;

            switch (fix.api())
            {
            case nvrhi::GraphicsAPI::VULKAN:
                nativeDev = dev->getNativeObject(nvrhi::ObjectTypes::VK_Device);
                nativeBuf = buf->getNativeObject(nvrhi::ObjectTypes::VK_Buffer);
                nativeTex = tex->getNativeObject(nvrhi::ObjectTypes::VK_Image);
                nativeCmd = cmd->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
                (void)dev->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Graphics);
                break;
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            case nvrhi::GraphicsAPI::METAL:
                nativeDev = dev->getNativeObject(nvrhi::ObjectTypes::Metal_Device);
                nativeBuf = buf->getNativeObject(nvrhi::ObjectTypes::Metal_Buffer);
                nativeTex = tex->getNativeObject(nvrhi::ObjectTypes::Metal_Texture);
                nativeCmd = cmd->getNativeObject(nvrhi::ObjectTypes::Metal_CommandBuffer);
                (void)dev->getNativeQueue(nvrhi::ObjectTypes::Metal_CommandQueue, nvrhi::CommandQueue::Graphics);
                break;
#endif
            case nvrhi::GraphicsAPI::D3D12:
                nativeDev = dev->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
                nativeBuf = buf->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
                nativeTex = tex->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
                nativeCmd = cmd->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
                (void)dev->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);
                break;
            case nvrhi::GraphicsAPI::D3D11:
                nativeDev = dev->getNativeObject(nvrhi::ObjectTypes::D3D11_Device);
                nativeBuf = buf->getNativeObject(nvrhi::ObjectTypes::D3D11_Buffer);
                nativeTex = tex->getNativeObject(nvrhi::ObjectTypes::D3D11_Resource);
                nativeCmd = cmd->getNativeObject(nvrhi::ObjectTypes::D3D11_DeviceContext);
                break;
            default:
                cmd->close();
                ctx.skip("Unsupported GraphicsAPI for NAT.BorrowedOwnership");
            }

            UAA_CHECK(isNonNullObject(nativeDev), "Borrowed native device must be non-null");
            UAA_CHECK(isNonNullObject(nativeBuf), "Borrowed native buffer must be non-null");
            UAA_CHECK(isNonNullObject(nativeTex), "Borrowed native texture must be non-null");
            UAA_CHECK(isNonNullObject(nativeCmd), "Borrowed native command list must be non-null");

            // Querying native objects must not alter NVRHI reference counts.
            UAA_CHECK_EQ(dev->GetRefCount(), devRefBefore, "Device GetRefCount changed after getNativeObject");
            UAA_CHECK_EQ(buf->GetRefCount(), bufRefBefore, "Buffer GetRefCount changed after getNativeObject");
            UAA_CHECK_EQ(tex->GetRefCount(), texRefBefore, "Texture GetRefCount changed after getNativeObject");
            UAA_CHECK_EQ(cmd->GetRefCount(), cmdRefBefore, "CommandList GetRefCount changed after getNativeObject");

            // Dropping borrowed handles and NVRHI wrappers must leave the device healthy.
            nativeDev = nullptr;
            nativeBuf = nullptr;
            nativeTex = nullptr;
            nativeCmd = nullptr;

            cmd->close();
            cmd = nullptr;
            buf = nullptr;
            tex = nullptr;

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after dropping borrowed handles");
            dev->runGarbageCollection();
        }

        void runBufferComputeAndVerify(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::IBuffer* inputBuf,
            const std::vector<uint32_t>& inputWords)
        {
            nvrhi::IDevice* dev = fix.device();
            const uint32_t count = static_cast<uint32_t>(inputWords.size());
            const uint32_t addVal = 41u;
            const uint32_t xorVal = 0x5A5Au;

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(layoutDesc);
            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));

            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(CopyAddParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true));
            nvrhi::BufferHandle dst = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(uint64_t(count) * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true));
            UAA_CHECK(cs != nullptr && layout != nullptr && pso != nullptr && cb != nullptr && dst != nullptr,
                "Failed to create compute verification resources");

            const CopyAddParams params{count, addVal, xorVal, 0u};
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::BindingSetHandle set = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inputBuf))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, dst)),
                layout);
            UAA_CHECK(set != nullptr, "Failed to create binding set for imported buffer");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
            cmd->dispatch((count + 63u) / 64u, 1u, 1u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(count, 0u);
            fix.readbackBuffer(dst, actual.data(), size_t(count) * sizeof(uint32_t), 0);
            std::vector<uint32_t> expected(count, 0u);
            for (uint32_t i = 0; i < count; ++i)
            {
                expected[i] = (inputWords[i] + addVal) ^ xorVal;
            }
            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(actual.data(), expected.data(), size_t(count) * sizeof(uint32_t), diff),
                "Imported buffer compute output mismatch: " + diff);
        }

        void testImportBufferAndTexture(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");
            if (!fix.backend())
            {
                ctx.skip("Backend context unavailable");
            }

            constexpr uint32_t kWordCount = 512u;
            const size_t kByteSize = size_t(kWordCount) * sizeof(uint32_t);
            const std::vector<uint32_t> wordsA = TestFixture::makeDeterministicWords(kWordCount, ctx.seed() ^ 0xA01ull);

            // Buffer import order 1: release NVRHI wrapper BEFORE native resource.
            NativeBufferImport nativeBuf1;
            if (!fix.backend()->createNativeImportBuffer(kByteSize, wordsA.data(), nativeBuf1))
            {
                ctx.skip("Backend does not support native buffer/texture import");
            }
            UAA_CHECK(isNonNullObject(nativeBuf1.handle), "Native import buffer handle must be non-null");

            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = nativeBuf1.byteSize;
            bufDesc.structStride = sizeof(uint32_t);
            bufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            bufDesc.debugName = "nat_imported_buf1";

            nvrhi::BufferHandle importedBuf1 = dev->createHandleForNativeBuffer(
                nativeBuf1.objectType, nativeBuf1.handle, bufDesc);
            UAA_CHECK(importedBuf1 != nullptr, "createHandleForNativeBuffer returned null");

            std::vector<uint32_t> rbWords(kWordCount, 0u);
            fix.readbackBuffer(importedBuf1, rbWords.data(), kByteSize, 0);
            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(rbWords.data(), wordsA.data(), kByteSize, diff),
                "Imported buffer initial data readback mismatch: " + diff);

            runBufferComputeAndVerify(fix, ctx, importedBuf1, wordsA);

            importedBuf1 = nullptr;
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed before releasing native buffer 1");
            dev->runGarbageCollection();
            if (nativeBuf1.release)
            {
                nativeBuf1.release();
            }

            // Buffer import order 2: wait+idle, release native backing, then drop NVRHI wrapper.
            const std::vector<uint32_t> wordsB = TestFixture::makeDeterministicWords(kWordCount, ctx.seed() ^ 0xB02ull);
            NativeBufferImport nativeBuf2;
            UAA_CHECK(
                fix.backend()->createNativeImportBuffer(kByteSize, wordsB.data(), nativeBuf2),
                "Second createNativeImportBuffer failed");
            nvrhi::BufferHandle importedBuf2 = dev->createHandleForNativeBuffer(
                nativeBuf2.objectType, nativeBuf2.handle, bufDesc);
            UAA_CHECK(importedBuf2 != nullptr, "Second createHandleForNativeBuffer returned null");

            runBufferComputeAndVerify(fix, ctx, importedBuf2, wordsB);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed before releasing native buffer 2");
            dev->runGarbageCollection();
            if (nativeBuf2.release)
            {
                nativeBuf2.release();
            }
            importedBuf2 = nullptr;
            dev->runGarbageCollection();

            // Texture import in both legal release orders.
            constexpr uint32_t kTexW = 32;
            constexpr uint32_t kTexH = 32;
            const std::vector<uint8_t> texPixels = TestFixture::makeDeterministicBytes(size_t(kTexW) * kTexH * 4u, ctx.seed() ^ 0xC03ull);

            NativeTextureImport nativeTex1;
            if (!fix.backend()->createNativeImportTexture(kTexW, kTexH, nvrhi::Format::RGBA8_UNORM, nativeTex1))
            {
                ctx.skip("Backend does not support native texture import");
            }
            UAA_CHECK(isNonNullObject(nativeTex1.handle), "Native import texture handle must be non-null");

            nvrhi::TextureHandle importedTex1 = dev->createHandleForNativeTexture(
                nativeTex1.objectType, nativeTex1.handle, nativeTex1.desc);
            UAA_CHECK(importedTex1 != nullptr, "createHandleForNativeTexture returned null");

            fix.uploadTexture2D(importedTex1, 0, 0, texPixels.data());
            std::vector<uint8_t> rbPixels;
            fix.readbackTexture2D(importedTex1, 0, 0, rbPixels);
            UAA_CHECK(
                TestFixture::checkBytesEqual(rbPixels.data(), texPixels.data(), texPixels.size(), diff),
                "Imported texture pixel readback mismatch: " + diff);

            if (nativeTex1.desc.isShaderResource)
            {
                // Sample the imported texture through a compute shader and verify texel(0,0).
                nvrhi::ShaderHandle sampCs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute, "main");
                nvrhi::BindingLayoutHandle sampLayout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                        .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
                nvrhi::ComputePipelineHandle sampPso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(sampCs).addBindingLayout(sampLayout));
                nvrhi::SamplerHandle sampler = dev->createSampler(
                    nvrhi::SamplerDesc().setAllFilters(false));
                nvrhi::BufferHandle outBuf = dev->createBuffer(
                    nvrhi::BufferDesc()
                        .setByteSize(sizeof(float) * 4)
                        .setStructStride(sizeof(float) * 4)
                        .setCanHaveUAVs(true)
                        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                        .setKeepInitialState(true));
                nvrhi::BufferHandle cb = dev->createBuffer(
                    nvrhi::BufferDesc()
                        .setByteSize(sizeof(SamplerTestParams))
                        .setIsConstantBuffer(true)
                        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                        .setKeepInitialState(true));
                UAA_CHECK(sampCs != nullptr && sampLayout != nullptr && sampPso != nullptr && sampler != nullptr && outBuf != nullptr && cb != nullptr,
                    "Sampler verification resources must not be null");

                const SamplerTestParams sp{{0.5f / float(kTexW), 0.5f / float(kTexH)}, {0.f, 0.f}, 0.f, 1u, 1u, 0u};
                fix.uploadBuffer(cb, &sp, sizeof(sp), 0);

                nvrhi::BindingSetHandle sampSet = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, importedTex1))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, sampler))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    sampLayout);
                UAA_CHECK(sampSet != nullptr, "Failed to create sampler binding set for imported texture");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(sampPso).addBindingSet(sampSet));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float sampled[4] = {};
                fix.readbackBuffer(outBuf, sampled, sizeof(sampled), 0);
                for (int c = 0; c < 4; ++c)
                {
                    const float expectedF = float(texPixels[c]) / 255.0f;
                    UAA_CHECK(
                        TestFixture::checkFloatNear(sampled[c], expectedF, 1.0f / 255.0f + 0.01f, diff),
                        "Sampled imported texture channel mismatch: " + diff);
                }
            }

            importedTex1 = nullptr;
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed before releasing native texture 1");
            dev->runGarbageCollection();
            if (nativeTex1.release)
            {
                nativeTex1.release();
            }

            NativeTextureImport nativeTex2;
            UAA_CHECK(
                fix.backend()->createNativeImportTexture(kTexW, kTexH, nvrhi::Format::RGBA8_UNORM, nativeTex2),
                "Second createNativeImportTexture failed");
            nvrhi::TextureHandle importedTex2 = dev->createHandleForNativeTexture(
                nativeTex2.objectType, nativeTex2.handle, nativeTex2.desc);
            UAA_CHECK(importedTex2 != nullptr, "Second createHandleForNativeTexture returned null");

            fix.uploadTexture2D(importedTex2, 0, 0, texPixels.data());
            fix.readbackTexture2D(importedTex2, 0, 0, rbPixels);
            UAA_CHECK(
                TestFixture::checkBytesEqual(rbPixels.data(), texPixels.data(), texPixels.size(), diff),
                "Second imported texture readback mismatch: " + diff);

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed before releasing native texture 2");
            dev->runGarbageCollection();
            if (nativeTex2.release)
            {
                nativeTex2.release();
            }
            importedTex2 = nullptr;
            dev->runGarbageCollection();
        }

        void testNativeViews(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            const nvrhi::GraphicsAPI api = fix.api();
            const nvrhi::ObjectType srvViewType = primarySrvViewObjectType(api);
            const nvrhi::ObjectType dsvViewType = primaryDsvViewObjectType(api);
            if (srvViewType == 0)
            {
                ctx.skip("Unsupported GraphicsAPI for NAT.NativeViews");
            }

            nvrhi::TextureDesc arrDesc;
            arrDesc.width = 64;
            arrDesc.height = 64;
            arrDesc.mipLevels = 4;
            arrDesc.arraySize = 3;
            arrDesc.dimension = nvrhi::TextureDimension::Texture2DArray;
            arrDesc.format = nvrhi::Format::RGBA8_UNORM;
            arrDesc.isShaderResource = true;
            arrDesc.isRenderTarget = true;
            arrDesc.debugName = "nat_views_arr";
            nvrhi::TextureHandle tex = dev->createTexture(arrDesc);
            UAA_CHECK(tex != nullptr, "Failed to create multi-mip/multi-slice texture");

            const nvrhi::Object fullView1 = tex->getNativeView(srvViewType, nvrhi::Format::UNKNOWN, nvrhi::AllSubresources);
            const nvrhi::Object fullView2 = tex->getNativeView(srvViewType, nvrhi::Format::UNKNOWN, nvrhi::AllSubresources);
            UAA_CHECK(isNonNullObject(fullView1), "Full texture getNativeView must be non-null");
            UAA_CHECK_EQ(fullView1.integer, fullView2.integer, "Repeated getNativeView with identical params must return cached handle");

            const nvrhi::TextureSubresourceSet subMip1(1u, 1u, 0u, 1u);
            const nvrhi::Object mip1View1 = tex->getNativeView(srvViewType, nvrhi::Format::RGBA8_UNORM, subMip1, nvrhi::TextureDimension::Texture2D);
            const nvrhi::Object mip1View2 = tex->getNativeView(srvViewType, nvrhi::Format::RGBA8_UNORM, subMip1, nvrhi::TextureDimension::Texture2D);
            UAA_CHECK(isNonNullObject(mip1View1), "Subresource mip 1 getNativeView must be non-null");
            UAA_CHECK_EQ(mip1View1.integer, mip1View2.integer, "Repeated subresource getNativeView must return cached handle");

            const nvrhi::TextureSubresourceSet subSlice2(0u, 1u, 2u, 1u);
            const nvrhi::Object slice2View = tex->getNativeView(srvViewType, nvrhi::Format::RGBA8_UNORM, subSlice2, nvrhi::TextureDimension::Texture2D);
            UAA_CHECK(isNonNullObject(slice2View), "Subresource slice 2 getNativeView must be non-null");

            const auto& extensions = fix.deviceInfo().enabledExtensions;
            // Vulkan fixture setup leaves imageViewFormatSwizzle disabled on portability devices.
            const bool hasPortabilitySubset = std::find(extensions.begin(), extensions.end(),
                "VK_KHR_portability_subset") != extensions.end();
            if (hasPortabilitySubset)
            {
                ctx.note("Vulkan portability device did not enable imageViewFormatSwizzle");
            }
            else
            {
                const nvrhi::ComponentMapping swizzle{
                    nvrhi::ComponentSwizzle::B,
                    nvrhi::ComponentSwizzle::G,
                    nvrhi::ComponentSwizzle::R,
                    nvrhi::ComponentSwizzle::A};
                const nvrhi::Object swizzleView1 = tex->getNativeView(
                    srvViewType,
                    nvrhi::Format::RGBA8_UNORM,
                    nvrhi::AllSubresources,
                    nvrhi::TextureDimension::Unknown,
                    false,
                    swizzle);
                const nvrhi::Object swizzleView2 = tex->getNativeView(
                    srvViewType,
                    nvrhi::Format::RGBA8_UNORM,
                    nvrhi::AllSubresources,
                    nvrhi::TextureDimension::Unknown,
                    false,
                    swizzle);
                UAA_CHECK(isNonNullObject(swizzleView1), "Swizzled getNativeView must be non-null");
                UAA_CHECK_EQ(swizzleView1.integer, swizzleView2.integer, "Repeated swizzled getNativeView must return cached handle");
            }

            // Verify bogus ObjectType returns null safely.
            UAA_CHECK(!isNonNullObject(tex->getNativeView(0xdeadbeefu)), "getNativeView(bogus) must return null");

            // Depth texture for isReadOnlyDSV query where supported.
            if (dsvViewType != 0)
            {
                nvrhi::Format depthFmt = nvrhi::Format::D32;
                if ((dev->queryFormatSupport(depthFmt) & nvrhi::FormatSupport::DepthStencil) == nvrhi::FormatSupport::None)
                {
                    depthFmt = nvrhi::Format::D24S8;
                }
                if ((dev->queryFormatSupport(depthFmt) & nvrhi::FormatSupport::DepthStencil) != nvrhi::FormatSupport::None)
                {
                    nvrhi::TextureHandle depthTex = dev->createTexture(
                        nvrhi::TextureDesc()
                            .setWidth(32)
                            .setHeight(32)
                            .setFormat(depthFmt)
                            .setIsRenderTarget(true)
                            .setDebugName("nat_views_depth"));
                    UAA_CHECK(depthTex != nullptr, "Failed to create depth texture");

                    const nvrhi::Object rwDsv = depthTex->getNativeView(dsvViewType, depthFmt, nvrhi::AllSubresources, nvrhi::TextureDimension::Unknown, false);
                    const nvrhi::Object roDsv = depthTex->getNativeView(dsvViewType, depthFmt, nvrhi::AllSubresources, nvrhi::TextureDimension::Unknown, true);
                    UAA_CHECK(isNonNullObject(rwDsv), "Read-write DSV native view must be non-null");
                    UAA_CHECK(isNonNullObject(roDsv), "Read-only DSV native view must be non-null");
                }
            }
        }

        void testTexNative(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");
            if (!fix.backend())
            {
                ctx.skip("Backend context unavailable");
            }

            constexpr uint32_t kWidth = 32;
            constexpr uint32_t kHeight = 32;
            NativeTextureImport nativeTex;
            if (!fix.backend()->createNativeImportTexture(kWidth, kHeight, nvrhi::Format::RGBA8_UNORM, nativeTex))
            {
                ctx.skip("Backend does not support native texture import");
            }

            nvrhi::TextureHandle tex = dev->createHandleForNativeTexture(
                nativeTex.objectType, nativeTex.handle, nativeTex.desc);
            UAA_CHECK(tex != nullptr, "createHandleForNativeTexture returned null");

            const nvrhi::TextureDesc& desc = tex->getDesc();
            UAA_CHECK_EQ(desc.width, nativeTex.desc.width, "Imported texture width mismatch");
            UAA_CHECK_EQ(desc.height, nativeTex.desc.height, "Imported texture height mismatch");
            UAA_CHECK_EQ(desc.depth, nativeTex.desc.depth, "Imported texture depth mismatch");
            UAA_CHECK_EQ(desc.arraySize, nativeTex.desc.arraySize, "Imported texture arraySize mismatch");
            UAA_CHECK_EQ(desc.mipLevels, nativeTex.desc.mipLevels, "Imported texture mipLevels mismatch");
            UAA_CHECK_EQ(desc.format, nativeTex.desc.format, "Imported texture format mismatch");
            UAA_CHECK_EQ(desc.dimension, nativeTex.desc.dimension, "Imported texture dimension mismatch");

            const nvrhi::ObjectType texObjType = primaryTextureObjectType(fix.api());
            if (texObjType != 0)
            {
                const nvrhi::Object queried = tex->getNativeObject(texObjType);
                UAA_CHECK(isNonNullObject(queried), "Imported texture getNativeObject must be non-null");
                UAA_CHECK_EQ(queried.integer, nativeTex.handle.integer, "Imported texture getNativeObject handle mismatch");
            }

            const nvrhi::ObjectType viewObjType = primarySrvViewObjectType(fix.api());
            if (viewObjType != 0)
            {
                const nvrhi::Object viewObj = tex->getNativeView(viewObjType);
                UAA_CHECK(isNonNullObject(viewObj), "Imported texture getNativeView must be non-null");
            }

            const std::vector<uint8_t> expectedPixels = TestFixture::makeDeterministicBytes(size_t(kWidth) * kHeight * 4u, ctx.seed() ^ 0xD04ull);
            fix.uploadTexture2D(tex, 0, 0, expectedPixels.data());

            std::vector<uint8_t> actualPixels;
            fix.readbackTexture2D(tex, 0, 0, actualPixels);
            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(actualPixels.data(), expectedPixels.data(), expectedPixels.size(), diff),
                "Imported texture fill/readback mismatch: " + diff);

            UAA_CHECK_EQ(tex->GetRefCount(), 1ul, "Imported texture wrapper should have refcount 1 before release");
            tex = nullptr;
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed before releasing native texture");
            dev->runGarbageCollection();
            if (nativeTex.release)
            {
                nativeTex.release();
            }
        }

        void testMappedBufferFlush(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");
            if (!fix.backend())
            {
                ctx.skip("Backend context unavailable");
            }

            const uint64_t alignment = fix.backend()->getMappedBufferFlushAlignment();
            if (alignment == 0)
            {
                ctx.skip("Backend does not expose mapped buffer flush extension");
            }

            const uint64_t halfSize = alignment * 2u;
            const uint64_t totalSize = alignment * 4u;

            nvrhi::BufferDesc writeDesc;
            writeDesc.byteSize = totalSize;
            writeDesc.cpuAccess = nvrhi::CpuAccessMode::Write;
            writeDesc.debugName = "nat_mapped_flush_src";
            nvrhi::BufferHandle writeBuf = dev->createBuffer(writeDesc);
            UAA_CHECK(writeBuf != nullptr, "Failed to create CpuAccessMode::Write buffer");

            nvrhi::BufferDesc gpuDstDesc;
            gpuDstDesc.byteSize = totalSize;
            gpuDstDesc.initialState = nvrhi::ResourceStates::CopyDest;
            gpuDstDesc.keepInitialState = true;
            gpuDstDesc.debugName = "nat_mapped_flush_dst";
            nvrhi::BufferHandle gpuDst = dev->createBuffer(gpuDstDesc);
            UAA_CHECK(gpuDst != nullptr, "Failed to create GPU destination buffer");

            const std::vector<uint8_t> patternA = TestFixture::makeDeterministicBytes(size_t(halfSize), ctx.seed() ^ 0xE05ull);
            const std::vector<uint8_t> patternB = TestFixture::makeDeterministicBytes(size_t(halfSize), ctx.seed() ^ 0xF06ull);

            void* mapped = dev->mapBuffer(writeBuf, nvrhi::CpuAccessMode::Write);
            UAA_CHECK(mapped != nullptr, "mapBuffer(Write) returned null");

            auto* mappedBytes = static_cast<uint8_t*>(mapped);
            std::memcpy(mappedBytes, patternA.data(), size_t(halfSize));
            UAA_CHECK(
                fix.backend()->flushMappedBuffer(dev, writeBuf, 0u, halfSize),
                "flushMappedBuffer failed for first range");

            std::memcpy(mappedBytes + halfSize, patternB.data(), size_t(halfSize));
            UAA_CHECK(
                fix.backend()->flushMappedBuffer(dev, writeBuf, halfSize, halfSize),
                "flushMappedBuffer failed for second range");

            dev->unmapBuffer(writeBuf);

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->copyBuffer(gpuDst, 0u, writeBuf, 0u, totalSize);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> actual(size_t(totalSize), 0u);
            fix.readbackBuffer(gpuDst, actual.data(), size_t(totalSize), 0u);

            std::vector<uint8_t> expected;
            expected.reserve(size_t(totalSize));
            expected.insert(expected.end(), patternA.begin(), patternA.end());
            expected.insert(expected.end(), patternB.begin(), patternB.end());

            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(actual.data(), expected.data(), size_t(totalSize), diff),
                "Mapped buffer flush readback mismatch: " + diff);

            // Validate error reporting on invalid flush ranges.
            ExpectErrorsGuard guard(fix);
            UAA_CHECK(!fix.backend()->flushMappedBuffer(dev, writeBuf, 0u, 0u), "flushMappedBuffer(size=0) must return false");
            UAA_CHECK(!fix.backend()->flushMappedBuffer(dev, writeBuf, halfSize, totalSize), "flushMappedBuffer(out-of-bounds) must return false");
            if (fix.api() == nvrhi::GraphicsAPI::VULKAN && alignment > 1u)
            {
                UAA_CHECK(!fix.backend()->flushMappedBuffer(dev, writeBuf, 1u, alignment), "flushMappedBuffer(unaligned) on Vulkan must return false");
            }
            UAA_CHECK(guard.errorCount() >= 2, "Expected error messages for invalid flushMappedBuffer calls");
        }

        void testAftermathQueries(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            UAA_CHECK(!dev->isAftermathEnabled(), "isAftermathEnabled() must return false on standard test configurations");
            nvrhi::AftermathCrashDumpHelper& helper = dev->getAftermathCrashDumpHelper();
            const uintptr_t helperAddr = reinterpret_cast<uintptr_t>(&helper);
            UAA_CHECK(helperAddr != 0u, "getAftermathCrashDumpHelper() must return a valid reference");

            if (fix.rawDevice() && fix.rawDevice() != dev)
            {
                UAA_CHECK(!fix.rawDevice()->isAftermathEnabled(), "Raw device isAftermathEnabled() must return false");
                nvrhi::AftermathCrashDumpHelper& rawHelper = fix.rawDevice()->getAftermathCrashDumpHelper();
                UAA_CHECK(&rawHelper == &helper, "Validation wrapper must forward getAftermathCrashDumpHelper() to raw device");
            }
        }
    } // namespace

    UAA_REGISTER_TEST("NAT.Objects", "core,native", testNativeObjects);
    UAA_REGISTER_TEST("NAT.BorrowedOwnership", "core,native", testBorrowedOwnership);
    UAA_REGISTER_TEST("NAT.ImportBufferTexture", "core,native", testImportBufferAndTexture);
    UAA_REGISTER_TEST("NAT.NativeViews", "core,native", testNativeViews);
    UAA_REGISTER_TEST("TEX.Native", "core,native", testTexNative);
    UAA_REGISTER_TEST("NAT.MappedFlush", "core,native", testMappedBufferFlush);
    UAA_REGISTER_TEST("NAT.Aftermath", "core,native", testAftermathQueries);
} // namespace nvrhi::uaa
