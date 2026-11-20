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

#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
#include <nvrhi/validation.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

        struct SolidColorPC
        {
            float color[4] = {0.f, 0.f, 0.f, 1.f};
            float depth = 0.f;
            float scale = 1.f;
            float offset[2] = {0.f, 0.f};
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

        struct WorkloadResults
        {
            std::vector<uint32_t> computeWords;
            std::vector<uint8_t> copyTexPixels;
            std::vector<uint8_t> gfxTexPixels;
        };

        nvrhi::IDevice* obtainValidationDevice(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::DeviceHandle& outOwnedWrapper)
        {
            if (fix.validationDevice())
            {
                return fix.validationDevice();
            }
#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
            outOwnedWrapper = nvrhi::validation::createValidationLayer(fix.rawDevice());
            UAA_CHECK(outOwnedWrapper != nullptr, "Failed to create validation layer wrapper");
            return outOwnedWrapper.Get();
#else
            (void)outOwnedWrapper;
            ctx.skip("NVRHI validation layer is not compiled in this build");
#endif
        }

        void uploadBufferOnDevice(
            nvrhi::IDevice* dev,
            nvrhi::IBuffer* dst,
            const void* data,
            size_t size)
        {
            nvrhi::CommandListHandle cmd = dev->createCommandList();
            if (!cmd)
            {
                throw TestFailureException("uploadBufferOnDevice: createCommandList returned null");
            }
            cmd->open();
            cmd->writeBuffer(dst, data, size, 0);
            cmd->close();
            dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            if (!dev->waitForIdle())
            {
                throw TestFailureException("uploadBufferOnDevice: waitForIdle returned false");
            }
            dev->runGarbageCollection();
        }

        void readbackBufferOnDevice(
            nvrhi::IDevice* dev,
            nvrhi::IBuffer* src,
            void* outData,
            size_t size)
        {
            nvrhi::BufferDesc rbDesc;
            rbDesc.byteSize = size;
            rbDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            rbDesc.debugName = "val_readback_buf";
            nvrhi::BufferHandle rb = dev->createBuffer(rbDesc);
            if (!rb)
            {
                throw TestFailureException("readbackBufferOnDevice: failed to create readback buffer");
            }

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            if (!cmd)
            {
                throw TestFailureException("readbackBufferOnDevice: createCommandList returned null");
            }
            cmd->open();
            cmd->copyBuffer(rb, 0, src, 0, size);
            cmd->close();
            dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            if (!dev->waitForIdle())
            {
                throw TestFailureException("readbackBufferOnDevice: waitForIdle returned false");
            }

            void* mapped = dev->mapBuffer(rb, nvrhi::CpuAccessMode::Read);
            if (!mapped)
            {
                throw TestFailureException("readbackBufferOnDevice: mapBuffer returned null");
            }
            std::memcpy(outData, mapped, size);
            dev->unmapBuffer(rb);
            dev->runGarbageCollection();
        }

        void readbackTexture2DOnDevice(
            nvrhi::IDevice* dev,
            nvrhi::ITexture* src,
            std::vector<uint8_t>& outPixels)
        {
            const nvrhi::TextureDesc& desc = src->getDesc();
            nvrhi::TextureDesc stgDesc;
            stgDesc.width = desc.width;
            stgDesc.height = desc.height;
            stgDesc.depth = 1;
            stgDesc.arraySize = 1;
            stgDesc.mipLevels = desc.mipLevels;
            stgDesc.format = desc.format;
            stgDesc.dimension = nvrhi::TextureDimension::Texture2D;
            stgDesc.debugName = "val_staging_tex";

            nvrhi::StagingTextureHandle stg = dev->createStagingTexture(stgDesc, nvrhi::CpuAccessMode::Read);
            if (!stg)
            {
                throw TestFailureException("readbackTexture2DOnDevice: createStagingTexture returned null");
            }

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            if (!cmd)
            {
                throw TestFailureException("readbackTexture2DOnDevice: createCommandList returned null");
            }
            cmd->open();
            cmd->copyTexture(stg, nvrhi::TextureSlice(), src, nvrhi::TextureSlice());
            cmd->close();
            dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            if (!dev->waitForIdle())
            {
                throw TestFailureException("readbackTexture2DOnDevice: waitForIdle returned false");
            }

            size_t rowPitch = 0;
            void* mapped = dev->mapStagingTexture(stg, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);
            if (!mapped)
            {
                throw TestFailureException("readbackTexture2DOnDevice: mapStagingTexture returned null");
            }

            const nvrhi::FormatInfo& fmt = nvrhi::getFormatInfo(desc.format);
            const size_t tightRow = size_t(desc.width) * fmt.bytesPerBlock;
            outPixels.resize(tightRow * desc.height);
            const auto* srcBytes = static_cast<const uint8_t*>(mapped);
            for (uint32_t y = 0; y < desc.height; ++y)
            {
                std::memcpy(outPixels.data() + size_t(y) * tightRow, srcBytes + size_t(y) * rowPitch, tightRow);
            }
            dev->unmapStagingTexture(stg);
            dev->runGarbageCollection();
        }

        WorkloadResults runRepresentativeWorkload(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::IDevice* dev)
        {
            UAA_CHECK(dev != nullptr, "Target device must not be null");
            WorkloadResults out;

            const std::vector<uint8_t> csBytes = fix.loadShaderBinary("cs_buffer_copy_add");
            const std::vector<uint8_t> vsBytes = fix.loadShaderBinary("vs_fullscreen_tri");
            const std::vector<uint8_t> psBytes = fix.loadShaderBinary("ps_solid_color");

            nvrhi::ShaderHandle cs = dev->createShader(
                nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute).setEntryName("main").setDebugName("cs_buffer_copy_add"),
                csBytes.data(), csBytes.size());
            nvrhi::ShaderHandle vs = dev->createShader(
                nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Vertex).setEntryName("main").setDebugName("vs_fullscreen_tri"),
                vsBytes.data(), vsBytes.size());
            nvrhi::ShaderHandle ps = dev->createShader(
                nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Pixel).setEntryName("main").setDebugName("ps_solid_color"),
                psBytes.data(), psBytes.size());
            UAA_CHECK(cs != nullptr && vs != nullptr && ps != nullptr, "Failed to create shaders for workload");

            // Compute copy + add
            constexpr uint32_t kCount = 256u;
            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x1111ull);

            nvrhi::BindingLayoutDesc csLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            csLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            csLayoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1));
            csLayoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle csLayout = dev->createBindingLayout(csLayoutDesc);
            UAA_CHECK(csLayout != nullptr, "Failed to create compute layout");

            nvrhi::ComputePipelineHandle csPso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(csLayout));
            UAA_CHECK(csPso != nullptr, "Failed to create compute pipeline");

            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(CopyAddParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true));
            nvrhi::BufferHandle src = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setInitialState(nvrhi::ResourceStates::ShaderResource)
                    .setKeepInitialState(true));
            nvrhi::BufferHandle dst = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true));
            UAA_CHECK(cb != nullptr && src != nullptr && dst != nullptr, "Failed to create compute buffers");

            const CopyAddParams params{kCount, 29u, 0xCAFEu, 0u};
            uploadBufferOnDevice(dev, cb, &params, sizeof(params));
            uploadBufferOnDevice(dev, src, inputWords.data(), kCount * sizeof(uint32_t));

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb));
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, src));
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, dst));
            nvrhi::BindingSetHandle set = dev->createBindingSet(setDesc, csLayout);
            UAA_CHECK(set != nullptr, "Failed to create compute binding set");

            // Texture clear + copy
            nvrhi::TextureDesc texDesc;
            texDesc.width = 16;
            texDesc.height = 16;
            texDesc.format = nvrhi::Format::RGBA8_UNORM;
            texDesc.isRenderTarget = true;
            texDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            texDesc.keepInitialState = true;
            nvrhi::TextureHandle texSrc = dev->createTexture(texDesc);
            nvrhi::TextureHandle texCopy = dev->createTexture(
                nvrhi::TextureDesc(texDesc).setInitialState(nvrhi::ResourceStates::CopyDest));
            nvrhi::TextureHandle texRt = dev->createTexture(texDesc);
            UAA_CHECK(texSrc != nullptr && texCopy != nullptr && texRt != nullptr, "Failed to create workload textures");

            nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(texRt));
            UAA_CHECK(fb != nullptr, "Failed to create workload framebuffer");

            nvrhi::BindingLayoutDesc gfxLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics);
            gfxLayoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SolidColorPC)));
            nvrhi::BindingLayoutHandle gfxLayout = dev->createBindingLayout(gfxLayoutDesc);
            UAA_CHECK(gfxLayout != nullptr, "Failed to create graphics binding layout");
            nvrhi::BindingSetHandle gfxSet = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(SolidColorPC))),
                gfxLayout);
            UAA_CHECK(gfxSet != nullptr, "Failed to create graphics binding set");

            nvrhi::GraphicsPipelineDesc gfxPsoDesc;
            gfxPsoDesc.setVertexShader(vs).setPixelShader(ps).addBindingLayout(gfxLayout);
            gfxPsoDesc.renderState.depthStencilState.depthTestEnable = false;
            gfxPsoDesc.renderState.depthStencilState.depthWriteEnable = false;
            gfxPsoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
            nvrhi::GraphicsPipelineHandle gfxPso = dev->createGraphicsPipeline(gfxPsoDesc, fb->getFramebufferInfo());
            UAA_CHECK(gfxPso != nullptr, "Failed to create graphics pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(csPso).addBindingSet(set));
            cmd->dispatch((kCount + 63u) / 64u, 1u, 1u);

            cmd->clearTextureFloat(texSrc, nvrhi::AllSubresources, nvrhi::Color(0.25f, 0.75f, 0.5f, 1.0f));
            cmd->copyTexture(texCopy, nvrhi::TextureSlice(), texSrc, nvrhi::TextureSlice());

            cmd->clearTextureFloat(texRt, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(gfxPso)
                    .setFramebuffer(fb)
                    .addBindingSet(gfxSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(16.f, 16.f))));
            SolidColorPC pc{{0.75f, 0.25f, 0.5f, 1.f}, 0.f, 1.f, {0.f, 0.f}};
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();

            dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(dev->waitForIdle(), "Workload waitForIdle failed");
            dev->runGarbageCollection();

            out.computeWords.resize(kCount, 0u);
            readbackBufferOnDevice(dev, dst, out.computeWords.data(), kCount * sizeof(uint32_t));
            readbackTexture2DOnDevice(dev, texCopy, out.copyTexPixels);
            readbackTexture2DOnDevice(dev, texRt, out.gfxTexPixels);
            return out;
        }

        class ThreadSafeMessageCallback final : public nvrhi::IMessageCallback
        {
        public:
            void message(nvrhi::MessageSeverity severity, const char* messageText) override
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                m_Messages.push_back(CapturedMessage{severity, messageText ? messageText : ""});
            }

            std::vector<CapturedMessage> snapshot() const
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                return m_Messages;
            }

            size_t errorCount() const
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                size_t count = 0;
                for (const auto& m : m_Messages)
                {
                    if (m.severity == nvrhi::MessageSeverity::Error || m.severity == nvrhi::MessageSeverity::Fatal)
                    {
                        ++count;
                    }
                }
                return count;
            }

        private:
            mutable std::mutex m_Mutex;
            std::vector<CapturedMessage> m_Messages;
        };

        void testValidationPassThrough(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* rawDev = fix.rawDevice();
            UAA_CHECK(rawDev != nullptr, "Raw device must not be null");

            const size_t baseErrors = fix.messages().errorCount();
            const WorkloadResults rawResults = runRepresentativeWorkload(fix, ctx, rawDev);
            UAA_CHECK_EQ(fix.messages().errorCount(), baseErrors, "Raw workload produced unexpected errors");

            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            const WorkloadResults valResults = runRepresentativeWorkload(fix, ctx, valDev);
            UAA_CHECK_EQ(fix.messages().errorCount(), baseErrors, "Validation pass-through produced unexpected errors");

            std::string diff;
            UAA_CHECK_EQ(valResults.computeWords.size(), rawResults.computeWords.size(), "Compute output size mismatch");
            UAA_CHECK(
                TestFixture::checkBytesEqual(
                    valResults.computeWords.data(),
                    rawResults.computeWords.data(),
                    rawResults.computeWords.size() * sizeof(uint32_t),
                    diff),
                "Compute output differed between raw and validation devices: " + diff);

            UAA_CHECK_EQ(valResults.copyTexPixels.size(), rawResults.copyTexPixels.size(), "Copy texture size mismatch");
            UAA_CHECK(
                TestFixture::checkBytesEqual(
                    valResults.copyTexPixels.data(),
                    rawResults.copyTexPixels.data(),
                    rawResults.copyTexPixels.size(),
                    diff),
                "Cleared/copied texture differed between raw and validation devices: " + diff);

            UAA_CHECK_EQ(valResults.gfxTexPixels.size(), rawResults.gfxTexPixels.size(), "Graphics RT size mismatch");
            UAA_CHECK(
                TestFixture::checkBytesEqual(
                    valResults.gfxTexPixels.data(),
                    rawResults.gfxTexPixels.data(),
                    rawResults.gfxTexPixels.size(),
                    diff),
                "Graphics draw output differed between raw and validation devices: " + diff);

            ownedValidation = nullptr;
        }

        void testInvalidDescriptors(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            UAA_CHECK(cs != nullptr, "Failed to create compute shader");

            ExpectErrorsGuard guard(fix);

            // Zero-capacity heap
            UAA_CHECK(
                valDev->createHeap(nvrhi::HeapDesc().setCapacity(0)) == nullptr,
                "createHeap(capacity=0) must return null");

            // Invalid buffers
            UAA_CHECK(
                valDev->createBuffer(nvrhi::BufferDesc().setByteSize(0)) == nullptr,
                "createBuffer(byteSize=0) must return null");
            UAA_CHECK(
                valDev->createBuffer(nvrhi::BufferDesc().setByteSize(256).setIsVolatile(true).setIsConstantBuffer(false)) == nullptr,
                "createBuffer(volatile non-CB) must return null");
            UAA_CHECK(
                valDev->createBuffer(nvrhi::BufferDesc().setByteSize(256).setIsVolatile(true).setIsConstantBuffer(true).setMaxVersions(0)) == nullptr,
                "createBuffer(volatile maxVersions=0) must return null");

            // Invalid textures
            UAA_CHECK(
                valDev->createTexture(nvrhi::TextureDesc().setWidth(0).setHeight(16).setFormat(nvrhi::Format::RGBA8_UNORM)) == nullptr,
                "createTexture(width=0) must return null");
            UAA_CHECK(
                valDev->createTexture(nvrhi::TextureDesc().setWidth(16).setHeight(16).setFormat(nvrhi::Format::RGBA8_UNORM).setDimension(nvrhi::TextureDimension::Unknown)) == nullptr,
                "createTexture(dimension=Unknown) must return null");
            UAA_CHECK(
                valDev->createTexture(nvrhi::TextureDesc().setWidth(16).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM).setDimension(nvrhi::TextureDimension::Texture1D)) == nullptr,
                "createTexture(1D height!=1) must return null");
            UAA_CHECK(
                valDev->createTexture(nvrhi::TextureDesc().setWidth(16).setHeight(16).setArraySize(1).setFormat(nvrhi::Format::RGBA8_UNORM).setDimension(nvrhi::TextureDimension::TextureCube)) == nullptr,
                "createTexture(Cube arraySize!=6) must return null");

            // Invalid binding layouts
            nvrhi::BindingLayoutDesc dupLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            dupLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            dupLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            UAA_CHECK(
                valDev->createBindingLayout(dupLayoutDesc) == nullptr,
                "createBindingLayout(duplicate slot) must return null");

            if (fix.api() == nvrhi::GraphicsAPI::VULKAN)
            {
                nvrhi::BindingLayoutDesc spaceDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute, 1u, false);
                spaceDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
                UAA_CHECK(
                    valDev->createBindingLayout(spaceDesc) == nullptr,
                    "createBindingLayout(Vulkan registerSpace!=0 without flag) must return null");
            }

            // Mismatched shader stage in graphics pipeline
            nvrhi::FramebufferInfo fbInfo;
            fbInfo.addColorFormat(nvrhi::Format::RGBA8_UNORM);
            nvrhi::GraphicsPipelineDesc badPsoDesc;
            badPsoDesc.setVertexShader(cs);
            UAA_CHECK(
                valDev->createGraphicsPipeline(badPsoDesc, fbInfo) == nullptr,
                "createGraphicsPipeline(compute shader as VS) must return null");

            UAA_CHECK(guard.errorCount() >= 8, "Expected validation errors for invalid descriptor creation calls");
            ownedValidation = nullptr;
        }

        void testCommandOrderValidation(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            nvrhi::BufferHandle bufA = valDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setInitialState(nvrhi::ResourceStates::CopyDest).setKeepInitialState(true));
            nvrhi::BufferHandle bufB = valDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setInitialState(nvrhi::ResourceStates::CopySource).setKeepInitialState(true));
            nvrhi::TimerQueryHandle tq = valDev->createTimerQuery();
            nvrhi::CommandListHandle cmd = valDev->createCommandList();
            UAA_CHECK(bufA != nullptr && bufB != nullptr && tq != nullptr && cmd != nullptr, "Setup handles must not be null");

            const uint32_t dummyWord = 0x12345678u;

            ExpectErrorsGuard guard(fix);

            // Operations on an unopened command list.
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->dispatch(1, 1, 1);
            cmd->writeBuffer(bufA, &dummyWord, sizeof(dummyWord), 0);
            cmd->copyBuffer(bufA, 0, bufB, 0, 16);
            cmd->beginTimerQuery(tq);
            cmd->endTimerQuery(tq);
            cmd->close();

            // Double open and missing pipeline state errors.
            cmd->open();
            cmd->open();
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->dispatch(1, 1, 1);
            cmd->close();

            // Recording and closing after close.
            cmd->writeBuffer(bufA, &dummyWord, sizeof(dummyWord), 0);
            cmd->copyBuffer(bufA, 0, bufB, 0, 16);
            cmd->beginTimerQuery(tq);
            cmd->endTimerQuery(tq);
            cmd->close();

            UAA_CHECK(guard.errorCount() >= 10, "Expected validation errors for command ordering violations");

            // Drop without submitting invalid command list.
            cmd = nullptr;
            ownedValidation = nullptr;
        }

        void testBindingMismatchValidation(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            UAA_CHECK(cs != nullptr, "Failed to create compute shader");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layoutA = valDev->createBindingLayout(layoutDesc);
            nvrhi::BindingLayoutHandle layoutOther = valDev->createBindingLayout(layoutDesc);
            UAA_CHECK(layoutA != nullptr && layoutOther != nullptr, "Failed to create compute layouts");

            nvrhi::ComputePipelineHandle psoA = valDev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layoutA));
            UAA_CHECK(psoA != nullptr, "Failed to create compute pipeline");

            nvrhi::BufferHandle cb = valDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setIsConstantBuffer(true));
            nvrhi::BufferHandle srvBuf = valDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setStructStride(4));
            nvrhi::BufferHandle uavBuf = valDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setStructStride(4).setCanHaveUAVs(true));
            nvrhi::TextureHandle nonUavTex = valDev->createTexture(
                nvrhi::TextureDesc().setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM));
            UAA_CHECK(cb != nullptr && srvBuf != nullptr && uavBuf != nullptr && nonUavTex != nullptr, "Setup resources must not be null");

            nvrhi::BindingSetHandle validSetOther = valDev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                layoutOther);
            UAA_CHECK(validSetOther != nullptr, "Failed to create binding set for layoutOther");

            nvrhi::BindingLayoutHandle layoutVertexOnly = valDev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Vertex)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layoutVertexOnly != nullptr, "Failed to create vertex-only layout");

            nvrhi::BindingSetHandle setVertexOnly = valDev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                layoutVertexOnly);
            UAA_CHECK(setVertexOnly != nullptr, "Failed to create binding set for vertex-only layout");

            ExpectErrorsGuard guard(fix);

            // Missing binding in set
            UAA_CHECK(
                valDev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf)),
                    layoutA) == nullptr,
                "BindingSet missing UAV binding must return null");

            // Wrong ResourceType (Texture_SRV in UAV slot, plus non-UAV buffer in UAV slot)
            UAA_CHECK(
                valDev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(2, nonUavTex)),
                    layoutA) == nullptr,
                "BindingSet with Texture_SRV in UAV slot must return null");

            UAA_CHECK(
                valDev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, srvBuf)),
                    layoutA) == nullptr,
                "BindingSet with non-UAV buffer as StructuredBuffer_UAV must return null");

            // Out-of-bounds ConstantBuffer range
            UAA_CHECK(
                valDev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb, nvrhi::BufferRange(512, 256)))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                    layoutA) == nullptr,
                "BindingSet with out-of-bounds ConstantBuffer range must return null");

            // Wrong shader visibility (ShaderType::None)
            UAA_CHECK(
                valDev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::None)
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))) == nullptr,
                "BindingLayout with ShaderType::None visibility must return null");

            // Mismatched BindingLayout and wrong-visibility layout passed to setComputeState
            nvrhi::CommandListHandle cmd = valDev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoA).addBindingSet(validSetOther));
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoA).addBindingSet(setVertexOnly));
            cmd->close();
            cmd = nullptr;

            UAA_CHECK(guard.errorCount() >= 6, "Expected validation errors for binding mismatches");
            ownedValidation = nullptr;
        }

        void testQueueMismatchValidation(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            nvrhi::CommandListHandle gfxCmd = valDev->createCommandList(
                nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
            UAA_CHECK(gfxCmd != nullptr, "Graphics command list must not be null");
            gfxCmd->open();
            gfxCmd->close();

            nvrhi::CommandListHandle uninitCmd = valDev->createCommandList();
            UAA_CHECK(uninitCmd != nullptr, "Unopened command list must not be null");

            ExpectErrorsGuard guard(fix);

            if (!fix.features().supports(nvrhi::Feature::ComputeQueue))
            {
                UAA_CHECK(
                    valDev->createCommandList(nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute)) == nullptr,
                    "Unsupported Compute queue command list must return null");
            }
            if (!fix.features().supports(nvrhi::Feature::CopyQueue))
            {
                UAA_CHECK(
                    valDev->createCommandList(nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy)) == nullptr,
                    "Unsupported Copy queue command list must return null");
            }

            // Executing Graphics command list on Compute queue triggers queue mismatch validation.
            UAA_CHECK_EQ(
                valDev->executeCommandList(gfxCmd, nvrhi::CommandQueue::Compute),
                0ull,
                "Executing Graphics command list on Compute queue must return 0");

            // Executing an unopened command list and null command list array must also fail validation.
            UAA_CHECK_EQ(
                valDev->executeCommandList(uninitCmd, nvrhi::CommandQueue::Graphics),
                0ull,
                "Executing unopened command list must return 0");
            UAA_CHECK_EQ(
                valDev->executeCommandLists(nullptr, 1, nvrhi::CommandQueue::Graphics),
                0ull,
                "executeCommandLists(nullptr) must return 0");

            UAA_CHECK(guard.errorCount() >= 3, "Expected validation errors for queue/execution mismatches");

            gfxCmd = nullptr;
            uninitCmd = nullptr;
            ownedValidation = nullptr;
        }

        void testErrorCaptureAndRecovery(TestContext& ctx)
        {
#if !defined(UAA_WITH_VALIDATION) || !UAA_WITH_VALIDATION
            ctx.skip("NVRHI validation layer is not compiled in this build");
#else
            TestFixture& fix = *ctx.fixture;
            ThreadSafeMessageCallback customCb;
            std::unique_ptr<IBackendContext> secondary =
                fix.backend() ? fix.backend()->createSecondaryDevice(&customCb) : nullptr;

            nvrhi::DeviceHandle ownedVal;
            nvrhi::IDevice* targetValDev = nullptr;
            bool usingCustomCb = false;

            if (secondary && secondary->getRawDevice())
            {
                ownedVal = nvrhi::validation::createValidationLayer(secondary->getRawDevice());
                UAA_CHECK(ownedVal != nullptr, "Failed to wrap secondary device in validation layer");
                targetValDev = ownedVal.Get();
                UAA_CHECK(
                    targetValDev->getMessageCallback() == &customCb,
                    "Validation wrapper getMessageCallback() must return custom callback");
                usingCustomCb = true;
            }
            else
            {
                targetValDev = obtainValidationDevice(fix, ctx, ownedVal);
                UAA_CHECK(
                    targetValDev->getMessageCallback() == fix.rawDevice()->getMessageCallback(),
                    "Validation wrapper getMessageCallback() must match raw device callback");
            }

            if (usingCustomCb)
            {
                auto worker = [targetValDev]() {
                    for (int i = 0; i < 4; ++i)
                    {
                        targetValDev->createHeap(nvrhi::HeapDesc().setCapacity(0));
                        targetValDev->createBuffer(
                            nvrhi::BufferDesc().setByteSize(256).setIsVolatile(true).setIsConstantBuffer(false));
                    }
                };

                std::thread t1(worker);
                std::thread t2(worker);
                t1.join();
                t2.join();

                const std::vector<CapturedMessage> msgs = customCb.snapshot();
                UAA_CHECK(customCb.errorCount() >= 16, "Custom callback did not capture all multi-threaded errors");
                for (const auto& entry : msgs)
                {
                    if (entry.severity == nvrhi::MessageSeverity::Error)
                    {
                        UAA_CHECK(!entry.text.empty(), "Captured error message must be non-empty");
                    }
                }
            }
            else
            {
                ExpectErrorsGuard guard(fix);
                auto worker = [targetValDev]() {
                    for (int i = 0; i < 4; ++i)
                    {
                        targetValDev->createHeap(nvrhi::HeapDesc().setCapacity(0));
                        targetValDev->createBuffer(
                            nvrhi::BufferDesc().setByteSize(256).setIsVolatile(true).setIsConstantBuffer(false));
                    }
                };

                std::thread t1(worker);
                std::thread t2(worker);
                t1.join();
                t2.join();

                const std::vector<CapturedMessage> msgs = guard.captured();
                UAA_CHECK(guard.errorCount() >= 16, "Message capture did not capture all multi-threaded errors");
                for (const auto& entry : msgs)
                {
                    if (entry.severity == nvrhi::MessageSeverity::Error)
                    {
                        UAA_CHECK(!entry.text.empty(), "Captured error message must be non-empty");
                    }
                }
            }

            // Verify recovery by running valid GPU copy work after validation errors.
            const size_t errorsBeforeRecovery = usingCustomCb ? customCb.errorCount() : fix.messages().errorCount();
            constexpr size_t kBytes = 1024;
            const std::vector<uint8_t> expected = TestFixture::makeDeterministicBytes(kBytes, ctx.seed() ^ 0x808ull);

            nvrhi::BufferHandle src = targetValDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kBytes).setInitialState(nvrhi::ResourceStates::CopySource).setKeepInitialState(true));
            nvrhi::BufferHandle dst = targetValDev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kBytes).setInitialState(nvrhi::ResourceStates::CopyDest).setKeepInitialState(true));
            UAA_CHECK(src != nullptr && dst != nullptr, "Recovery buffers must not be null");

            uploadBufferOnDevice(targetValDev, src, expected.data(), kBytes);

            nvrhi::CommandListHandle cmd = targetValDev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Recovery command list must not be null");
            cmd->open();
            cmd->copyBuffer(dst, 0, src, 0, kBytes);
            cmd->close();
            targetValDev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(targetValDev->waitForIdle(), "Recovery waitForIdle failed");

            std::vector<uint8_t> actual(kBytes, 0);
            readbackBufferOnDevice(targetValDev, dst, actual.data(), kBytes);
            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(actual.data(), expected.data(), kBytes, diff),
                "Recovery copy verification failed: " + diff);

            const size_t errorsAfterRecovery = usingCustomCb ? customCb.errorCount() : fix.messages().errorCount();
            UAA_CHECK_EQ(errorsAfterRecovery, errorsBeforeRecovery, "Valid recovery work emitted unexpected errors");

            cmd = nullptr;
            src = nullptr;
            dst = nullptr;
            ownedVal = nullptr;
            secondary.reset();
#endif
        }

        void testOutputPreservation(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            constexpr uint64_t kSentinelSize = 0x11223344ull;
            constexpr uint64_t kSentinelAlign = 0x55667788ull;
            nvrhi::MemoryRequirements memReq{kSentinelSize, kSentinelAlign};

            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0)));
            nvrhi::SamplerHandle sampler = dev->createSampler(nvrhi::SamplerDesc());
            UAA_CHECK(layout != nullptr && sampler != nullptr, "Setup objects must not be null");

            UAA_CHECK(!dev->queryMemoryRequirements(memReq), "IDevice::queryMemoryRequirements must return false");
            UAA_CHECK_EQ(memReq.size, kSentinelSize, "IDevice::queryMemoryRequirements must preserve size sentinel");
            UAA_CHECK_EQ(memReq.alignment, kSentinelAlign, "IDevice::queryMemoryRequirements must preserve alignment sentinel");

            UAA_CHECK(!layout->queryMemoryRequirements(memReq), "BindingLayout::queryMemoryRequirements must return false");
            UAA_CHECK_EQ(memReq.size, kSentinelSize, "BindingLayout::queryMemoryRequirements must preserve size sentinel");
            UAA_CHECK_EQ(memReq.alignment, kSentinelAlign, "BindingLayout::queryMemoryRequirements must preserve alignment sentinel");

            UAA_CHECK(!sampler->queryMemoryRequirements(memReq), "Sampler::queryMemoryRequirements must return false");
            UAA_CHECK_EQ(memReq.size, kSentinelSize, "Sampler::queryMemoryRequirements must preserve size sentinel");
            UAA_CHECK_EQ(memReq.alignment, kSentinelAlign, "Sampler::queryMemoryRequirements must preserve alignment sentinel");

            nvrhi::rt::AccelStructPrebuildInfo prebuild{111ull, 222ull, 333ull};
            if (fix.features().supports(nvrhi::Feature::RayTracingAccelStruct) || fix.api() == nvrhi::GraphicsAPI::METAL)
            {
                nvrhi::rt::AccelStructDesc blasDesc;
                blasDesc.isTopLevel = false;
                UAA_CHECK(
                    !dev->queryTopLevelAccelStructPrebuildInfo(blasDesc, 1u, prebuild),
                    "queryTopLevelAccelStructPrebuildInfo with BLAS desc must return false");
                UAA_CHECK_EQ(prebuild.resultBytes, 111ull, "Prebuild resultBytes sentinel corrupted on BLAS query");
                UAA_CHECK_EQ(prebuild.scratchBytes, 222ull, "Prebuild scratchBytes sentinel corrupted on BLAS query");
                UAA_CHECK_EQ(prebuild.updateScratchBytes, 333ull, "Prebuild updateScratchBytes sentinel corrupted on BLAS query");

                nvrhi::rt::AccelStructDesc tlasDesc;
                tlasDesc.isTopLevel = true;
                tlasDesc.topLevelMaxInstances = 4u;
                UAA_CHECK(
                    !dev->queryTopLevelAccelStructPrebuildInfo(tlasDesc, 5u, prebuild),
                    "queryTopLevelAccelStructPrebuildInfo over-capacity must return false");
                UAA_CHECK_EQ(prebuild.resultBytes, 111ull, "Prebuild resultBytes sentinel corrupted on over-capacity query");
                UAA_CHECK_EQ(prebuild.scratchBytes, 222ull, "Prebuild scratchBytes sentinel corrupted on over-capacity query");
                UAA_CHECK_EQ(prebuild.updateScratchBytes, 333ull, "Prebuild updateScratchBytes sentinel corrupted on over-capacity query");
            }
            else
            {
                ctx.note("RayTracingAccelStruct unsupported; skipped queryTopLevelAccelStructPrebuildInfo");
            }
        }

        void testNoFalseErrors(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::DeviceHandle ownedValidation;
            nvrhi::IDevice* valDev = obtainValidationDevice(fix, ctx, ownedValidation);

            const size_t baseErrors = fix.messages().errorCount();
            const size_t baseWarnings = fix.messages().warningCount();

            // Run representative valid buffer, texture, and graphics operations.
            const WorkloadResults results = runRepresentativeWorkload(fix, ctx, valDev);
            UAA_CHECK(!results.computeWords.empty(), "Compute workload output empty");

            // Also test valid sampler + texture sampling compute dispatch.
            const std::vector<uint8_t> sampShaderBytes = fix.loadShaderBinary("cs_sampler_test");
            nvrhi::ShaderHandle sampCs = valDev->createShader(
                nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute).setEntryName("main").setDebugName("cs_sampler_test"),
                sampShaderBytes.data(), sampShaderBytes.size());
            UAA_CHECK(sampCs != nullptr, "Failed to create cs_sampler_test shader");

            nvrhi::TextureHandle sampleTex = valDev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(4)
                    .setHeight(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsUAV(true)
                    .setInitialState(nvrhi::ResourceStates::ShaderResource)
                    .setKeepInitialState(true));
            nvrhi::SamplerHandle sampler = valDev->createSampler(nvrhi::SamplerDesc());
            nvrhi::BufferHandle sampleOut = valDev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(16 * sizeof(float) * 4)
                    .setStructStride(sizeof(float) * 4)
                    .setCanHaveUAVs(true)
                    .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true));
            nvrhi::BufferHandle sampleCb = valDev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(SamplerTestParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true));
            UAA_CHECK(sampleTex != nullptr && sampler != nullptr && sampleOut != nullptr && sampleCb != nullptr,
                "Sampler test resources must not be null");

            const SamplerTestParams sampParams{{0.125f, 0.125f}, {0.25f, 0.25f}, 0.f, 4u, 4u, 0u};
            uploadBufferOnDevice(valDev, sampleCb, &sampParams, sizeof(sampParams));

            nvrhi::BindingLayoutDesc sampLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            sampLayoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(0));
            sampLayoutDesc.addItem(nvrhi::BindingLayoutItem::Sampler(1));
            sampLayoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            sampLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3));
            nvrhi::BindingLayoutHandle sampLayout = valDev->createBindingLayout(sampLayoutDesc);
            nvrhi::ComputePipelineHandle sampPso = valDev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(sampCs).addBindingLayout(sampLayout));
            nvrhi::BindingSetHandle sampSet = valDev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::Texture_SRV(0, sampleTex))
                    .addItem(nvrhi::BindingSetItem::Sampler(1, sampler))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, sampleOut))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, sampleCb)),
                sampLayout);
            UAA_CHECK(sampLayout != nullptr && sampPso != nullptr && sampSet != nullptr, "Sampler pipeline objects must not be null");

            nvrhi::CommandListHandle cmd = valDev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Command list must not be null");
            cmd->open();
            cmd->clearTextureFloat(sampleTex, nvrhi::AllSubresources, nvrhi::Color(0.2f, 0.4f, 0.6f, 1.0f));
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(sampPso).addBindingSet(sampSet));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            valDev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(valDev->waitForIdle(), "waitForIdle failed");
            valDev->runGarbageCollection();

            const size_t newErrors = fix.messages().errorCount() - baseErrors;
            const size_t newWarnings = fix.messages().warningCount() - baseWarnings;
            ctx.note("Validation warning count during valid operations: " + std::to_string(newWarnings));
            UAA_CHECK_EQ(newErrors, size_t(0), "Valid operations under validation layer produced false errors: " + fix.messages().formatAll());

            cmd = nullptr;
            ownedValidation = nullptr;
        }
    } // namespace

    UAA_REGISTER_TEST("VAL.PassThrough", "core", testValidationPassThrough);
    UAA_REGISTER_TEST("VAL.InvalidDescriptors", "core", testInvalidDescriptors);
    UAA_REGISTER_TEST("VAL.CommandOrder", "core", testCommandOrderValidation);
    UAA_REGISTER_TEST("VAL.BindingMismatch", "core", testBindingMismatchValidation);
    UAA_REGISTER_TEST("VAL.QueueMismatch", "core", testQueueMismatchValidation);
    UAA_REGISTER_TEST("VAL.ErrorCapture", "core", testErrorCaptureAndRecovery);
    UAA_REGISTER_TEST("VAL.OutputPreservation", "core", testOutputPreservation);
    UAA_REGISTER_TEST("VAL.NoFalseErrors", "core", testNoFalseErrors);
} // namespace nvrhi::uaa
