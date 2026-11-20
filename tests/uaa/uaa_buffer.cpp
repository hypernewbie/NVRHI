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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
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

        struct RawTypedParams
        {
            uint32_t count = 0;
            uint32_t mode = 0;
            uint32_t addVal = 0;
            uint32_t pad = 0;
        };

        struct VertexPosColor
        {
            float px;
            float py;
            float pz;
            float r;
            float g;
            float b;
            float a;
        };

        struct InstanceData
        {
            float ox;
            float oy;
            float tr;
            float tg;
            float tb;
            float ta;
        };

        struct IndirectWorkPC
        {
            uint32_t baseOffset = 0;
            uint32_t marker = 0;
            uint32_t strideY = 0;
            uint32_t strideZ = 0;
        };

        bool isPowerOfTwo(uint64_t value)
        {
            return value != 0 && (value & (value - 1)) == 0;
        }

        TestFixture& requireFixture(TestContext& ctx)
        {
            if (!ctx.fixture || !ctx.fixture->device())
            {
                ctx.skip("GPU fixture is not initialised");
            }
            return *ctx.fixture;
        }

        // Creates a device buffer initialised with guard words or deterministic words.
        nvrhi::BufferHandle createWordBuffer(
            const TestFixture& fix,
            const nvrhi::BufferDesc& desc,
            const std::vector<uint32_t>& words,
            TestContext& ctx)
        {
            nvrhi::BufferHandle buf = fix.device()->createBuffer(desc);
            UAA_CHECK(buf != nullptr, "Failed to create buffer '" + desc.debugName + "'");
            if (!words.empty())
            {
                fix.uploadBuffer(buf, words.data(), words.size() * sizeof(uint32_t), 0);
            }
            return buf;
        }

        // Tests buffer creation and getDesc() round-trip invariants.
        void testBufCreate(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::BufferDesc descs[] = {
                nvrhi::BufferDesc().setByteSize(1024).setCpuAccess(nvrhi::CpuAccessMode::None).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("buf_device_local"),
                nvrhi::BufferDesc().setByteSize(1024).setCpuAccess(nvrhi::CpuAccessMode::Write).enableAutomaticStateTracking(nvrhi::ResourceStates::CopySource).setDebugName("buf_upload"),
                nvrhi::BufferDesc().setByteSize(1024).setCpuAccess(nvrhi::CpuAccessMode::Read).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("buf_readback"),
                nvrhi::BufferDesc().setByteSize(256).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("buf_constant"),
                nvrhi::BufferDesc().setByteSize(1024).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("buf_vertex"),
                nvrhi::BufferDesc().setByteSize(1024).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("buf_index"),
                nvrhi::BufferDesc().setByteSize(256).setIsDrawIndirectArgs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndirectArgument).setDebugName("buf_indirect"),
                nvrhi::BufferDesc().setByteSize(1024).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("buf_structured"),
                nvrhi::BufferDesc().setByteSize(1024).setCanHaveRawViews(true).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("buf_raw"),
                nvrhi::BufferDesc().setByteSize(1024).setCanHaveTypedViews(true).setCanHaveUAVs(true).setFormat(nvrhi::Format::R32_UINT).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("buf_typed")
            };

            for (const nvrhi::BufferDesc& expected : descs)
            {
                nvrhi::BufferHandle buf = dev->createBuffer(expected);
                UAA_CHECK(buf != nullptr, "createBuffer returned null for " + expected.debugName);

                const nvrhi::BufferDesc& actual = buf->getDesc();
                UAA_CHECK_EQ(actual.byteSize, expected.byteSize, "byteSize mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.structStride, expected.structStride, "structStride mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.cpuAccess, expected.cpuAccess, "cpuAccess mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.isConstantBuffer, expected.isConstantBuffer, "isConstantBuffer mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.isVertexBuffer, expected.isVertexBuffer, "isVertexBuffer mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.isIndexBuffer, expected.isIndexBuffer, "isIndexBuffer mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.isDrawIndirectArgs, expected.isDrawIndirectArgs, "isDrawIndirectArgs mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.canHaveUAVs, expected.canHaveUAVs, "canHaveUAVs mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.canHaveRawViews, expected.canHaveRawViews, "canHaveRawViews mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.canHaveTypedViews, expected.canHaveTypedViews, "canHaveTypedViews mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.format, expected.format, "format mismatch for " + expected.debugName);
            }
        }

        // Tests writeBuffer, copyBuffer, and readback across 1-byte, odd-size, nonzero-offset, and exact-end ranges.
        void testBufWriteCopyRead(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            // Verify 1-byte and 13-byte buffers; keep 4-byte padded host staging so inline Vulkan updates stay in bounds.
            for (const size_t smallSize : { size_t(1), size_t(13) })
            {
                const std::vector<uint8_t> expected = TestFixture::makeDeterministicBytes(smallSize, ctx.seed() + smallSize);
                std::vector<uint8_t> paddedUpload(16, 0u);
                std::memcpy(paddedUpload.data(), expected.data(), smallSize);

                nvrhi::BufferHandle src = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(smallSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopySource).setDebugName("wcr_small_src"));
                nvrhi::BufferHandle dst = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(smallSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("wcr_small_dst"));
                nvrhi::BufferHandle rb = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(smallSize).setCpuAccess(nvrhi::CpuAccessMode::Read).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("wcr_small_rb"));
                UAA_CHECK(src != nullptr && dst != nullptr && rb != nullptr, "Failed to allocate small write/copy buffers");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->writeBuffer(src, paddedUpload.data(), smallSize, 0);
                cmd->copyBuffer(dst, 0, src, 0, smallSize);
                cmd->copyBuffer(rb, 0, dst, 0, smallSize);
                cmd->close();
                fix.submitAndWait(cmd);

                void* mapped = dev->mapBuffer(rb, nvrhi::CpuAccessMode::Read);
                UAA_CHECK(mapped != nullptr, "mapBuffer returned null for small readback buffer");
                std::string diff;
                const bool equal = TestFixture::checkBytesEqual(mapped, expected.data(), smallSize, diff);
                dev->unmapBuffer(rb);
                UAA_CHECK(equal, "Small buffer (" + std::to_string(smallSize) + "B) mismatch: " + diff);
            }

            // Verify nonzero offsets and exact-end copies on a guard-filled buffer.
            constexpr uint64_t kBufSize = 64;
            constexpr uint64_t kSrcOffset = 13;
            constexpr uint64_t kDstOffset = 19;
            constexpr uint64_t kCopySize = kBufSize - kDstOffset; // 45 bytes reaching exact end of dst

            const std::vector<uint8_t> guardFill(kBufSize, TestFixture::kGuardByte);
            const std::vector<uint8_t> payload = TestFixture::makeDeterministicBytes(kCopySize, ctx.seed() ^ 0x1010ull);

            nvrhi::BufferHandle srcBuf = dev->createBuffer(nvrhi::BufferDesc().setByteSize(kBufSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopySource).setDebugName("wcr_off_src"));
            nvrhi::BufferHandle dstBuf = dev->createBuffer(nvrhi::BufferDesc().setByteSize(kBufSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("wcr_off_dst"));
            UAA_CHECK(srcBuf != nullptr && dstBuf != nullptr, "Failed to create offset copy buffers");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(srcBuf, guardFill.data(), kBufSize, 0);
            cmd->writeBuffer(dstBuf, guardFill.data(), kBufSize, 0);
            cmd->writeBuffer(srcBuf, payload.data(), kCopySize, kSrcOffset);
            cmd->copyBuffer(dstBuf, kDstOffset, srcBuf, kSrcOffset, kCopySize);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> actual(kBufSize, 0u);
            fix.readbackBuffer(dstBuf, actual.data(), kBufSize, 0);

            std::string diff;
            UAA_CHECK(TestFixture::checkGuardBytes(actual.data(), kDstOffset, TestFixture::kGuardByte, diff),
                "Leading guard bytes corrupted in dstBuf: " + diff);
            UAA_CHECK(TestFixture::checkBytesEqual(actual.data() + kDstOffset, payload.data(), kCopySize, diff),
                "Exact-end copied range mismatch: " + diff);
        }

        // Tests disjoint and overlapping writes preserving guard bytes and final write ordering.
        void testBufOverwrite(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint64_t kBufSize = 256;
            const std::vector<uint8_t> initBytes(kBufSize, TestFixture::kGuardByte);
            const std::vector<uint8_t> rangeA = TestFixture::makeDeterministicBytes(32, ctx.seed() + 1);  // [16, 48)
            const std::vector<uint8_t> rangeB = TestFixture::makeDeterministicBytes(48, ctx.seed() + 2);  // [80, 128)
            const std::vector<uint8_t> rangeC = TestFixture::makeDeterministicBytes(64, ctx.seed() + 3);  // [32, 96) overlaps A & B

            std::vector<uint8_t> expected = initBytes;
            std::memcpy(expected.data() + 16, rangeA.data(), rangeA.size());
            std::memcpy(expected.data() + 80, rangeB.data(), rangeB.size());
            std::memcpy(expected.data() + 32, rangeC.data(), rangeC.size());

            nvrhi::BufferHandle buf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kBufSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("overwrite_buf"));
            UAA_CHECK(buf != nullptr, "Failed to create overwrite buffer");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(buf, initBytes.data(), initBytes.size(), 0);
            cmd->writeBuffer(buf, rangeA.data(), rangeA.size(), 16);
            cmd->writeBuffer(buf, rangeB.data(), rangeB.size(), 80);
            cmd->writeBuffer(buf, rangeC.data(), rangeC.size(), 32);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> actual(kBufSize, 0u);
            fix.readbackBuffer(buf, actual.data(), kBufSize, 0);

            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), expected.data(), kBufSize, diff),
                "Overwrite byte order / guard verification failed: " + diff);
        }

        // Tests clearBufferUInt with zero and non-zero patterns, including subsequent shader reads.
        void testBufClear(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint64_t kByteSize = kCount * sizeof(uint32_t);
            constexpr uint32_t kPattern = 0xA55A1234u;
            constexpr uint32_t kAddVal = 0x00001001u;
            constexpr uint32_t kXorVal = 0x00FF00FFu;

            nvrhi::BufferHandle target = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("clear_target"));
            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("clear_shader_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(CopyAddParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("clear_cb"));
            UAA_CHECK(target != nullptr && outBuf != nullptr && cb != nullptr, "Failed to create clear test buffers");

            const CopyAddParams params{ kCount, kAddVal, kXorVal, 0u };
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layout != nullptr, "Failed to create binding layout");

            nvrhi::BindingSetHandle set = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, target))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                layout);
            UAA_CHECK(set != nullptr, "Failed to create binding set");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

            // Fill with non-zero first, then clear to 0 and verify full zero readback.
            {
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearBufferUInt(target, 0xDEADBEEFu);
                cmd->clearBufferUInt(target, 0u);
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> words(kCount, 1u);
                fix.readbackBuffer(target, words.data(), kByteSize, 0);
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    UAA_CHECK_EQ(words[i], 0u, "clearBufferUInt(0) mismatch at index " + std::to_string(i));
                }
            }

            // Clear to kPattern and immediately read from compute shader in the same command list.
            {
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearBufferUInt(target, kPattern);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> clearedWords(kCount, 0u);
                std::vector<uint32_t> shaderWords(kCount, 0u);
                fix.readbackBuffer(target, clearedWords.data(), kByteSize, 0);
                fix.readbackBuffer(outBuf, shaderWords.data(), kByteSize, 0);

                const uint32_t expectedOut = (kPattern + kAddVal) ^ kXorVal;
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    UAA_CHECK_EQ(clearedWords[i], kPattern, "clearBufferUInt(kPattern) mismatch at index " + std::to_string(i));
                    UAA_CHECK_EQ(shaderWords[i], expectedOut, "Shader did not observe cleared buffer value at index " + std::to_string(i));
                }
            }
        }

        // Tests CPU map/unmap upload and readback round-trip including remapping the same buffer.
        void testBufMap(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint64_t kByteSize = kCount * sizeof(uint32_t);
            constexpr uint32_t kAddVal = 17u;
            constexpr uint32_t kXorVal = 0x55AA33CCu;

            nvrhi::BufferHandle uploadBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setCpuAccess(nvrhi::CpuAccessMode::Write).enableAutomaticStateTracking(nvrhi::ResourceStates::CopySource).setDebugName("map_upload"));
            nvrhi::BufferHandle gpuIn = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("map_gpu_in"));
            nvrhi::BufferHandle gpuOut = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("map_gpu_out"));
            nvrhi::BufferHandle readbackBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setCpuAccess(nvrhi::CpuAccessMode::Read).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("map_readback"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(CopyAddParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("map_cb"));
            UAA_CHECK(uploadBuf != nullptr && gpuIn != nullptr && gpuOut != nullptr && readbackBuf != nullptr && cb != nullptr,
                "Failed to allocate buffers for BUF.Map");

            const CopyAddParams params{ kCount, kAddVal, kXorVal, 0u };
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layout != nullptr, "Failed to create binding layout");

            nvrhi::BindingSetHandle set = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, gpuIn))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, gpuOut)),
                layout);
            UAA_CHECK(set != nullptr, "Failed to create binding set");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

            for (uint32_t pass = 0; pass < 2; ++pass)
            {
                const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() + pass * 777ull);

                void* writePtr = dev->mapBuffer(uploadBuf, nvrhi::CpuAccessMode::Write);
                UAA_CHECK(writePtr != nullptr, "mapBuffer(Write) returned null on pass " + std::to_string(pass));
                std::memcpy(writePtr, inputWords.data(), kByteSize);
                if (fix.backend() && fix.backend()->getMappedBufferFlushAlignment() > 0)
                {
                    const uint64_t align = fix.backend()->getMappedBufferFlushAlignment();
                    if ((kByteSize % align) == 0)
                    {
                        UAA_CHECK(fix.backend()->flushMappedBuffer(dev, uploadBuf, 0, kByteSize),
                            "flushMappedBuffer failed on uploadBuf");
                    }
                }
                dev->unmapBuffer(uploadBuf);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->copyBuffer(gpuIn, 0, uploadBuf, 0, kByteSize);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->copyBuffer(readbackBuf, 0, gpuOut, 0, kByteSize);
                cmd->close();
                fix.submitAndWait(cmd);

                void* readPtr = dev->mapBuffer(readbackBuf, nvrhi::CpuAccessMode::Read);
                UAA_CHECK(readPtr != nullptr, "mapBuffer(Read) returned null on pass " + std::to_string(pass));

                std::vector<uint32_t> actual(kCount, 0u);
                std::memcpy(actual.data(), readPtr, kByteSize);
                dev->unmapBuffer(readbackBuf);

                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                    UAA_CHECK_EQ(actual[i], expected,
                        "Mapped round-trip mismatch at pass " + std::to_string(pass) + " index " + std::to_string(i));
                }
            }
        }

        // Tests invalid buffer mapping patterns under the NVRHI validation layer.
        void testBufMapInvalid(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (!fix.isValidationActive())
            {
                ctx.skip("Requires NVRHI validation layer");
            }

            nvrhi::IDevice* dev = fix.device();
            nvrhi::BufferHandle noneBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCpuAccess(nvrhi::CpuAccessMode::None).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("invalid_map_none"));
            nvrhi::BufferHandle writeBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCpuAccess(nvrhi::CpuAccessMode::Write).enableAutomaticStateTracking(nvrhi::ResourceStates::CopySource).setDebugName("invalid_map_write"));
            nvrhi::BufferHandle readBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCpuAccess(nvrhi::CpuAccessMode::Read).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("invalid_map_read"));
            UAA_CHECK(noneBuf != nullptr && writeBuf != nullptr && readBuf != nullptr, "Failed to create buffers for BUF.MapInvalid");

            {
                ExpectErrorsGuard guard(fix);

                // The validation layer rejects the call before it reaches the driver,
                // so the buffer is never mapped.
                void* pNone = dev->mapBuffer(noneBuf, nvrhi::CpuAccessMode::Write);
                (void)pNone;
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error when mapping CpuAccessMode::None buffer");
                UAA_CHECK(pNone == nullptr, "Mapping a buffer without CPU access must not return a pointer");
            }

            {
                ExpectErrorsGuard guard(fix);
                void* pReadAsWrite = dev->mapBuffer(readBuf, nvrhi::CpuAccessMode::Write);
                (void)pReadAsWrite;
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error when mapping Read buffer with Write");
            }

            {
                ExpectErrorsGuard guard(fix);
                void* pWriteAsRead = dev->mapBuffer(writeBuf, nvrhi::CpuAccessMode::Read);
                (void)pWriteAsRead;
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error when mapping Write buffer with Read");
            }

            {
                ExpectErrorsGuard guard(fix);
                void* first = dev->mapBuffer(writeBuf, nvrhi::CpuAccessMode::Write);
                void* second = dev->mapBuffer(writeBuf, nvrhi::CpuAccessMode::Write);
                (void)first;
                (void)second;
                if (first != nullptr)
                {
                    dev->unmapBuffer(writeBuf);
                }
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error on double-map");
            }

            {
                ExpectErrorsGuard guard(fix);
                dev->unmapBuffer(readBuf);
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error when unmapping an unmapped buffer");
            }
        }

        // Tests volatile constant buffer versioning across multiple dispatches and blocked queue submissions.
        void testBufVolatileConstants(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kElemCount = 16;
            constexpr uint32_t kStrideBytes = 256;
            constexpr uint32_t kDispatchesCmd1 = 4;
            constexpr uint32_t kDispatchesCmd2 = 3;

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kElemCount, ctx.seed() ^ 0x4444ull);
            nvrhi::BufferHandle inBuf = createWordBuffer(
                fix,
                nvrhi::BufferDesc().setByteSize(kElemCount * sizeof(uint32_t)).setStructStride(sizeof(uint32_t)).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("vol_in"),
                inputWords,
                ctx);

            nvrhi::BufferHandle outBuf1 = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(uint64_t(kDispatchesCmd1) * kStrideBytes).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("vol_out1"));
            nvrhi::BufferHandle outBuf2 = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(uint64_t(kDispatchesCmd2) * kStrideBytes).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("vol_out2"));
            nvrhi::BufferHandle volatileCB = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(CopyAddParams))
                    .setIsConstantBuffer(true)
                    .setIsVolatile(true)
                    .setMaxVersions(8)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
                    .setDebugName("vol_cb"));
            UAA_CHECK(outBuf1 != nullptr && outBuf2 != nullptr && volatileCB != nullptr, "Failed to create volatile test buffers");

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layout != nullptr, "Failed to create volatile binding layout");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

            auto makeSliceSet = [&](nvrhi::IBuffer* targetBuf, uint32_t sliceIndex) -> nvrhi::BindingSetHandle
            {
                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, volatileCB))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, targetBuf, nvrhi::Format::UNKNOWN,
                            nvrhi::BufferRange(uint64_t(sliceIndex) * kStrideBytes, kElemCount * sizeof(uint32_t)))),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create volatile slice binding set");
                return set;
            };

            std::vector<nvrhi::BindingSetHandle> sets1;
            std::vector<CopyAddParams> params1;
            for (uint32_t i = 0; i < kDispatchesCmd1; ++i)
            {
                sets1.push_back(makeSliceSet(outBuf1, i));
                params1.push_back(CopyAddParams{ kElemCount, 100u + i * 17u, 0x11110000u + i * 0x101u, 0u });
            }

            std::vector<nvrhi::BindingSetHandle> sets2;
            std::vector<CopyAddParams> params2;
            for (uint32_t i = 0; i < kDispatchesCmd2; ++i)
            {
                sets2.push_back(makeSliceSet(outBuf2, i));
                params2.push_back(CopyAddParams{ kElemCount, 500u + i * 29u, 0x22220000u + i * 0x203u, 0u });
            }

            nvrhi::CommandListHandle cmd1 = dev->createCommandList();
            UAA_CHECK(cmd1 != nullptr, "Failed to create cmd1");
            cmd1->open();
            for (uint32_t i = 0; i < kDispatchesCmd1; ++i)
            {
                cmd1->writeBuffer(volatileCB, &params1[i], sizeof(CopyAddParams));
                cmd1->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(sets1[i]));
                cmd1->dispatch(1, 1, 1);
            }
            cmd1->close();

            std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
            bool gateArmed = false;
            if (gate && gate->enqueueWait(nvrhi::CommandQueue::Graphics))
            {
                gateArmed = true;
            }
            else if (gate)
            {
                ctx.note("IGpuGate::enqueueWait returned false; running sequential volatile submissions");
            }
            else
            {
                ctx.note("IGpuGate unavailable on this backend; running sequential volatile submissions");
            }

            dev->executeCommandList(cmd1, nvrhi::CommandQueue::Graphics);

            // Submit cmd2 reusing the same volatile buffer while cmd1 is blocked in the GPU queue.
            nvrhi::CommandListHandle cmd2 = dev->createCommandList();
            UAA_CHECK(cmd2 != nullptr, "Failed to create cmd2");
            cmd2->open();
            for (uint32_t i = 0; i < kDispatchesCmd2; ++i)
            {
                cmd2->writeBuffer(volatileCB, &params2[i], sizeof(CopyAddParams));
                cmd2->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(sets2[i]));
                cmd2->dispatch(1, 1, 1);
            }
            cmd2->close();
            dev->executeCommandList(cmd2, nvrhi::CommandQueue::Graphics);

            if (gateArmed)
            {
                gate->signal();
            }
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after volatile constant dispatches");
            dev->runGarbageCollection();

            std::vector<uint32_t> actual1(uint64_t(kDispatchesCmd1) * (kStrideBytes / sizeof(uint32_t)), 0u);
            std::vector<uint32_t> actual2(uint64_t(kDispatchesCmd2) * (kStrideBytes / sizeof(uint32_t)), 0u);
            fix.readbackBuffer(outBuf1, actual1.data(), actual1.size() * sizeof(uint32_t), 0);
            fix.readbackBuffer(outBuf2, actual2.data(), actual2.size() * sizeof(uint32_t), 0);

            for (uint32_t d = 0; d < kDispatchesCmd1; ++d)
            {
                const uint32_t base = d * (kStrideBytes / sizeof(uint32_t));
                for (uint32_t i = 0; i < kElemCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + params1[d].addVal) ^ params1[d].xorVal;
                    UAA_CHECK_EQ(actual1[base + i], expected,
                        "cmd1 volatile CB mismatch at dispatch " + std::to_string(d) + " index " + std::to_string(i));
                }
            }

            for (uint32_t d = 0; d < kDispatchesCmd2; ++d)
            {
                const uint32_t base = d * (kStrideBytes / sizeof(uint32_t));
                for (uint32_t i = 0; i < kElemCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + params2[d].addVal) ^ params2[d].xorVal;
                    UAA_CHECK_EQ(actual2[base + i], expected,
                        "cmd2 volatile CB mismatch at dispatch " + std::to_string(d) + " index " + std::to_string(i));
                }
            }
        }

        // Tests structured and raw SRV/UAV views with nonzero aligned BufferRange offsets and producer-consumer reversal.
        void testBufStructuredRaw(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kElemCount = 32;
            constexpr uint64_t kRangeOffset = 256;
            constexpr uint64_t kRangeBytes = kElemCount * sizeof(uint32_t);
            constexpr uint64_t kTotalBytes = kRangeOffset + kRangeBytes + 256;
            constexpr uint64_t kTotalWords = kTotalBytes / sizeof(uint32_t);
            constexpr uint64_t kStartWord = kRangeOffset / sizeof(uint32_t);

            const std::vector<uint32_t> seedWords = TestFixture::makeDeterministicWords(kElemCount, ctx.seed() ^ 0x5555ull);
            std::vector<uint32_t> initA(kTotalWords, TestFixture::kGuardWord);
            std::copy(seedWords.begin(), seedWords.end(), initA.begin() + kStartWord);
            std::vector<uint32_t> initB(kTotalWords, TestFixture::kGuardWord);

            // Structured SRV -> UAV, then reverse producer/consumer in the same command list.
            {
                nvrhi::BufferDesc sDesc = nvrhi::BufferDesc()
                    .setByteSize(kTotalBytes)
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                    .setDebugName("struct_buf");
                nvrhi::BufferHandle bufA = createWordBuffer(fix, sDesc, initA, ctx);
                nvrhi::BufferHandle bufB = createWordBuffer(fix, sDesc, initB, ctx);
                nvrhi::BufferHandle cb1 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(CopyAddParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("struct_cb1"));
                nvrhi::BufferHandle cb2 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(CopyAddParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("struct_cb2"));
                UAA_CHECK(cb1 != nullptr && cb2 != nullptr, "Failed to create structured CBs");

                const CopyAddParams p1{ kElemCount, 11u, 0x0F0F0F0Fu, 0u };
                const CopyAddParams p2{ kElemCount, 23u, 0x00FF00FFu, 0u };
                fix.uploadBuffer(cb1, &p1, sizeof(p1), 0);
                fix.uploadBuffer(cb2, &p2, sizeof(p2), 0);

                nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
                UAA_CHECK(layout != nullptr, "Failed to create structured binding layout");

                const nvrhi::BufferRange range(kRangeOffset, kRangeBytes);
                nvrhi::BindingSetHandle setForward = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb1))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, bufA, nvrhi::Format::UNKNOWN, range))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, bufB, nvrhi::Format::UNKNOWN, range)),
                    layout);
                nvrhi::BindingSetHandle setReverse = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb2))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, bufB, nvrhi::Format::UNKNOWN, range))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, bufA, nvrhi::Format::UNKNOWN, range)),
                    layout);
                UAA_CHECK(setForward != nullptr && setReverse != nullptr, "Failed to create structured binding sets");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create structured pipeline");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(setForward));
                cmd->dispatch(1, 1, 1);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(setReverse));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> outA(kTotalWords, 0u);
                std::vector<uint32_t> outB(kTotalWords, 0u);
                fix.readbackBuffer(bufA, outA.data(), kTotalBytes, 0);
                fix.readbackBuffer(bufB, outB.data(), kTotalBytes, 0);

                for (uint64_t i = 0; i < kTotalWords; ++i)
                {
                    if (i < kStartWord || i >= kStartWord + kElemCount)
                    {
                        UAA_CHECK_EQ(outA[i], TestFixture::kGuardWord, "Structured bufA guard corrupted at word " + std::to_string(i));
                        UAA_CHECK_EQ(outB[i], TestFixture::kGuardWord, "Structured bufB guard corrupted at word " + std::to_string(i));
                    }
                    else
                    {
                        const uint32_t idx = static_cast<uint32_t>(i - kStartWord);
                        const uint32_t expB = (seedWords[idx] + p1.addVal) ^ p1.xorVal;
                        const uint32_t expA = (expB + p2.addVal) ^ p2.xorVal;
                        UAA_CHECK_EQ(outB[i], expB, "Structured intermediate mismatch at index " + std::to_string(idx));
                        UAA_CHECK_EQ(outA[i], expA, "Structured reversed output mismatch at index " + std::to_string(idx));
                    }
                }
            }

            // Raw SRV -> UAV (mode 0 of cs_raw_typed_rw), then reverse producer/consumer.
            {
                nvrhi::BufferDesc rDesc = nvrhi::BufferDesc()
                    .setByteSize(kTotalBytes)
                    .setCanHaveRawViews(true)
                    .setCanHaveUAVs(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                    .setDebugName("raw_buf");
                nvrhi::BufferHandle bufA = createWordBuffer(fix, rDesc, initA, ctx);
                nvrhi::BufferHandle bufB = createWordBuffer(fix, rDesc, initB, ctx);
                nvrhi::BufferHandle dummyTypedIn = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(256).setCanHaveTypedViews(true).setFormat(nvrhi::Format::R32_UINT).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("raw_dummy_typed_in"));
                nvrhi::BufferHandle dummyTypedOut = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(256).setCanHaveTypedViews(true).setCanHaveUAVs(true).setFormat(nvrhi::Format::R32_UINT).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("raw_dummy_typed_out"));
                nvrhi::BufferHandle cb1 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(RawTypedParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("raw_cb1"));
                nvrhi::BufferHandle cb2 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(RawTypedParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("raw_cb2"));
                UAA_CHECK(dummyTypedIn != nullptr && dummyTypedOut != nullptr && cb1 != nullptr && cb2 != nullptr,
                    "Failed to create raw auxiliary buffers");

                const RawTypedParams p1{ kElemCount, 0u, 37u, 0u };
                const RawTypedParams p2{ kElemCount, 0u, 19u, 0u };
                fix.uploadBuffer(cb1, &p1, sizeof(p1), 0);
                fix.uploadBuffer(cb2, &p2, sizeof(p2), 0);

                nvrhi::ShaderHandle cs = fix.createShader("cs_raw_typed_rw", nvrhi::ShaderType::Compute);
                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(0))
                        .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(1))
                        .addItem(nvrhi::BindingLayoutItem::TypedBuffer_SRV(2))
                        .addItem(nvrhi::BindingLayoutItem::TypedBuffer_UAV(3))
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(4)));
                UAA_CHECK(layout != nullptr, "Failed to create raw/typed binding layout");

                const nvrhi::BufferRange range(kRangeOffset, kRangeBytes);
                nvrhi::BindingSetHandle setForward = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, bufA, range))
                        .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, bufB, range))
                        .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, dummyTypedIn, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, dummyTypedOut, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb1)),
                    layout);
                nvrhi::BindingSetHandle setReverse = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, bufB, range))
                        .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, bufA, range))
                        .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, dummyTypedIn, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, dummyTypedOut, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb2)),
                    layout);
                UAA_CHECK(setForward != nullptr && setReverse != nullptr, "Failed to create raw binding sets");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create raw compute pipeline");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(setForward));
                cmd->dispatch(1, 1, 1);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(setReverse));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> outA(kTotalWords, 0u);
                std::vector<uint32_t> outB(kTotalWords, 0u);
                fix.readbackBuffer(bufA, outA.data(), kTotalBytes, 0);
                fix.readbackBuffer(bufB, outB.data(), kTotalBytes, 0);

                for (uint64_t i = 0; i < kTotalWords; ++i)
                {
                    if (i < kStartWord || i >= kStartWord + kElemCount)
                    {
                        UAA_CHECK_EQ(outA[i], TestFixture::kGuardWord, "Raw bufA guard corrupted at word " + std::to_string(i));
                        UAA_CHECK_EQ(outB[i], TestFixture::kGuardWord, "Raw bufB guard corrupted at word " + std::to_string(i));
                    }
                    else
                    {
                        const uint32_t idx = static_cast<uint32_t>(i - kStartWord);
                        const uint32_t expB = seedWords[idx] + p1.addVal;
                        const uint32_t expA = expB + p2.addVal;
                        UAA_CHECK_EQ(outB[i], expB, "Raw intermediate mismatch at index " + std::to_string(idx));
                        UAA_CHECK_EQ(outA[i], expA, "Raw reversed output mismatch at index " + std::to_string(idx));
                    }
                }
            }
        }

        // Tests raw/typed SRV/UAV combinations on one multi-use backing buffer and typed format view creation.
        void testBufTypedViews(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kElemCount = 32;
            constexpr uint64_t kRangeBytes = kElemCount * sizeof(uint32_t);
            constexpr uint64_t kOff0 = 256;
            constexpr uint64_t kOff1 = 512;
            constexpr uint64_t kTotalBytes = 1024;
            constexpr uint64_t kTotalWords = kTotalBytes / sizeof(uint32_t);

            const std::vector<uint32_t> seedWords = TestFixture::makeDeterministicWords(kElemCount, ctx.seed() ^ 0x7777ull);
            std::vector<uint32_t> initMulti(kTotalWords, TestFixture::kGuardWord);
            std::copy(seedWords.begin(), seedWords.end(), initMulti.begin() + (kOff0 / sizeof(uint32_t)));
            std::vector<uint32_t> initOther(kTotalWords, TestFixture::kGuardWord);

            // Single backing buffer supporting raw views, typed views, and UAVs at nonzero aligned offsets.
            nvrhi::BufferDesc multiDesc = nvrhi::BufferDesc()
                .setByteSize(kTotalBytes)
                .setCanHaveRawViews(true)
                .setCanHaveTypedViews(true)
                .setCanHaveUAVs(true)
                .setFormat(nvrhi::Format::R32_UINT)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                .setDebugName("typed_multi_buf");
            nvrhi::BufferHandle multiBuf = createWordBuffer(fix, multiDesc, initMulti, ctx);
            nvrhi::BufferHandle otherBuf = createWordBuffer(fix, multiDesc, initOther, ctx);

            nvrhi::BufferHandle dummyRawIn = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCanHaveRawViews(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("typed_dummy_raw_in"));
            nvrhi::BufferHandle dummyRawOut = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCanHaveRawViews(true).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("typed_dummy_raw_out"));
            nvrhi::BufferHandle dummyTypedIn = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCanHaveTypedViews(true).setFormat(nvrhi::Format::R32_UINT).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("typed_dummy_typed_in"));
            nvrhi::BufferHandle dummyTypedOut = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(256).setCanHaveTypedViews(true).setCanHaveUAVs(true).setFormat(nvrhi::Format::R32_UINT).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("typed_dummy_typed_out"));
            UAA_CHECK(dummyRawIn != nullptr && dummyRawOut != nullptr && dummyTypedIn != nullptr && dummyTypedOut != nullptr,
                "Failed to create dummy typed/raw buffers");

            nvrhi::BufferHandle cb1 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(RawTypedParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("typed_cb1"));
            nvrhi::BufferHandle cb2 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(RawTypedParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("typed_cb2"));
            nvrhi::BufferHandle cb3 = dev->createBuffer(nvrhi::BufferDesc().setByteSize(sizeof(RawTypedParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("typed_cb3"));
            UAA_CHECK(cb1 != nullptr && cb2 != nullptr && cb3 != nullptr, "Failed to create typed mode CBs");

            const RawTypedParams pMode1{ kElemCount, 1u, 10u, 0u }; // RawIn -> TypedOut
            const RawTypedParams pMode2{ kElemCount, 2u, 20u, 0u }; // TypedIn -> RawOut
            const RawTypedParams pMode3{ kElemCount, 3u, 30u, 0u }; // TypedIn -> TypedOut
            fix.uploadBuffer(cb1, &pMode1, sizeof(pMode1), 0);
            fix.uploadBuffer(cb2, &pMode2, sizeof(pMode2), 0);
            fix.uploadBuffer(cb3, &pMode3, sizeof(pMode3), 0);

            nvrhi::ShaderHandle cs = fix.createShader("cs_raw_typed_rw", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(1))
                    .addItem(nvrhi::BindingLayoutItem::TypedBuffer_SRV(2))
                    .addItem(nvrhi::BindingLayoutItem::TypedBuffer_UAV(3))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(4)));
            UAA_CHECK(layout != nullptr, "Failed to create typed binding layout");

            const nvrhi::BufferRange r0(kOff0, kRangeBytes);
            const nvrhi::BufferRange r1(kOff1, kRangeBytes);

            // Mode 1: multiBuf[kOff0] (Raw SRV) -> otherBuf[kOff0] (Typed UAV)
            nvrhi::BindingSetHandle set1 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, multiBuf, r0))
                    .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, dummyRawOut, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, dummyTypedIn, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, otherBuf, nvrhi::Format::R32_UINT, r0))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb1)),
                layout);

            // Mode 2: otherBuf[kOff0] (Typed SRV) -> multiBuf[kOff1] (Raw UAV)
            nvrhi::BindingSetHandle set2 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, dummyRawIn, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, multiBuf, r1))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, otherBuf, nvrhi::Format::R32_UINT, r0))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, dummyTypedOut, nvrhi::Format::R32_UINT, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb2)),
                layout);

            // Mode 3: multiBuf[kOff1] (Typed SRV) -> otherBuf[kOff1] (Typed UAV)
            nvrhi::BindingSetHandle set3 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, dummyRawIn, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, dummyRawOut, nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, multiBuf, nvrhi::Format::R32_UINT, r1))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3, otherBuf, nvrhi::Format::R32_UINT, r1))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb3)),
                layout);
            UAA_CHECK(set1 != nullptr && set2 != nullptr && set3 != nullptr, "Failed to create typed mode binding sets");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create typed compute pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set1));
            cmd->dispatch(1, 1, 1);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set2));
            cmd->dispatch(1, 1, 1);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set3));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actualMulti(kTotalWords, 0u);
            std::vector<uint32_t> actualOther(kTotalWords, 0u);
            fix.readbackBuffer(multiBuf, actualMulti.data(), kTotalBytes, 0);
            fix.readbackBuffer(otherBuf, actualOther.data(), kTotalBytes, 0);

            const uint64_t w0 = kOff0 / sizeof(uint32_t);
            const uint64_t w1 = kOff1 / sizeof(uint32_t);
            for (uint32_t i = 0; i < kElemCount; ++i)
            {
                const uint32_t exp1 = seedWords[i] + pMode1.addVal;
                const uint32_t exp2 = exp1 + pMode2.addVal;
                const uint32_t exp3 = exp2 + pMode3.addVal;
                UAA_CHECK_EQ(actualOther[w0 + i], exp1, "Mode 1 (raw->typed) mismatch at index " + std::to_string(i));
                UAA_CHECK_EQ(actualMulti[w1 + i], exp2, "Mode 2 (typed->raw) mismatch at index " + std::to_string(i));
                UAA_CHECK_EQ(actualOther[w1 + i], exp3, "Mode 3 (typed->typed) mismatch at index " + std::to_string(i));
            }
            UAA_CHECK_EQ(actualMulti[0], TestFixture::kGuardWord, "Guard word before offset 256 corrupted");
            UAA_CHECK_EQ(actualOther[w1 + kElemCount], TestFixture::kGuardWord, "Guard word after range corrupted");

            // Verify additional typed buffer view formats (RGBA8_UINT, R16_UINT) gated by queryFormatSupport.
            for (const nvrhi::Format fmt : { nvrhi::Format::RGBA8_UINT, nvrhi::Format::R16_UINT })
            {
                const nvrhi::FormatSupport sup = dev->queryFormatSupport(fmt);
                if ((sup & nvrhi::FormatSupport::Buffer) == 0)
                {
                    ctx.note("Typed buffer format unsupported by queryFormatSupport: " + std::to_string(static_cast<int>(fmt)));
                    continue;
                }

                nvrhi::BufferHandle fmtBuf = dev->createBuffer(
                    nvrhi::BufferDesc()
                        .setByteSize(1024)
                        .setCanHaveTypedViews(true)
                        .setCanHaveUAVs(true)
                        .setFormat(fmt)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
                        .setDebugName("typed_alt_fmt"));
                UAA_CHECK(fmtBuf != nullptr, "Failed to create typed buffer for supported format");

                nvrhi::BindingSetDesc bsDesc;
                bsDesc.addItem(nvrhi::BindingSetItem::RawBuffer_SRV(0, dummyRawIn))
                    .addItem(nvrhi::BindingSetItem::RawBuffer_UAV(1, dummyRawOut))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(2, fmtBuf, fmt, nvrhi::BufferRange(256, 256)))
                    .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(3,
                        (sup & nvrhi::FormatSupport::ShaderUavStore) != 0 ? fmtBuf.Get() : dummyTypedOut.Get(),
                        (sup & nvrhi::FormatSupport::ShaderUavStore) != 0 ? fmt : nvrhi::Format::R32_UINT,
                        (sup & nvrhi::FormatSupport::ShaderUavStore) != 0 ? nvrhi::BufferRange(512, 256) : nvrhi::BufferRange(0, 256)))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(4, cb1));
                nvrhi::BindingSetHandle fmtSet = dev->createBindingSet(bsDesc, layout);
                UAA_CHECK(fmtSet != nullptr, "Failed to create typed view binding set for format " + std::to_string(static_cast<int>(fmt)));
            }
        }

        // Tests vertex, index, and indirect argument buffers with nonzero 256-byte aligned offsets.
        void testBufVertexIndexIndirect(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            // Part 1: Vertex + index buffers with offset = 256 in drawIndexed.
            {
                constexpr uint64_t kOffset = 256;
                const std::array<VertexPosColor, 4> verts = {{
                    { -1.0f, -1.0f, 0.5f,  0.2f, 0.4f, 0.6f, 1.0f },
                    {  1.0f, -1.0f, 0.5f,  0.2f, 0.4f, 0.6f, 1.0f },
                    {  1.0f,  1.0f, 0.5f,  0.2f, 0.4f, 0.6f, 1.0f },
                    { -1.0f,  1.0f, 0.5f,  0.2f, 0.4f, 0.6f, 1.0f }
                }};
                const InstanceData inst{ 0.0f, 0.0f, 0.5f, 0.5f, 1.0f, 1.0f };
                const std::array<uint32_t, 6> indices = {{ 0u, 1u, 2u, 0u, 2u, 3u }};

                nvrhi::BufferHandle vb0 = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(kOffset + sizeof(verts)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("vii_vb0"));
                nvrhi::BufferHandle vb1 = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(kOffset + sizeof(inst)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("vii_vb1"));
                nvrhi::BufferHandle ib = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(kOffset + sizeof(indices)).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("vii_ib"));
                UAA_CHECK(vb0 != nullptr && vb1 != nullptr && ib != nullptr, "Failed to create vertex/index buffers");

                fix.uploadBuffer(vb0, verts.data(), sizeof(verts), kOffset);
                fix.uploadBuffer(vb1, &inst, sizeof(inst), kOffset);
                fix.uploadBuffer(ib, indices.data(), sizeof(indices), kOffset);

                const float vsParams[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
                const float psParams[4] = { 0.1f, 0.2f, 0.1f, 0.0f };
                nvrhi::BufferHandle vsCB = dev->createBuffer(nvrhi::BufferDesc().setByteSize(16).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("vii_vscb"));
                nvrhi::BufferHandle psCB = dev->createBuffer(nvrhi::BufferDesc().setByteSize(16).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("vii_pscb"));
                UAA_CHECK(vsCB != nullptr && psCB != nullptr, "Failed to create graphics CBs");
                fix.uploadBuffer(vsCB, vsParams, sizeof(vsParams), 0);
                fix.uploadBuffer(psCB, psParams, sizeof(psParams), 0);

                nvrhi::ShaderHandle vs = fix.createShader("vs_vertex_input", nvrhi::ShaderType::Vertex);
                nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

                const nvrhi::VertexAttributeDesc attrs[] = {
                    nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setBufferIndex(0).setOffset(0).setElementStride(sizeof(VertexPosColor)),
                    nvrhi::VertexAttributeDesc().setName("COLOR").setFormat(nvrhi::Format::RGBA32_FLOAT).setBufferIndex(0).setOffset(12).setElementStride(sizeof(VertexPosColor)),
                    nvrhi::VertexAttributeDesc().setName("TEXCOORD").setFormat(nvrhi::Format::RG32_FLOAT).setBufferIndex(1).setOffset(0).setElementStride(sizeof(InstanceData)).setIsInstanced(true),
                    nvrhi::VertexAttributeDesc().setName("COLOR").setFormat(nvrhi::Format::RGBA32_FLOAT).setBufferIndex(1).setOffset(8).setElementStride(sizeof(InstanceData)).setIsInstanced(true)
                };
                // Use exact semantic names matching HLSL declaration (COLOR0/COLOR1 -> COLOR with index or exact name).
                nvrhi::VertexAttributeDesc exactAttrs[4] = { attrs[0], attrs[1], attrs[2], attrs[3] };
                exactAttrs[1].setName("COLOR0");
                exactAttrs[3].setName("COLOR1");

                nvrhi::InputLayoutHandle inputLayout = dev->createInputLayout(exactAttrs, 4, vs);
                UAA_CHECK(inputLayout != nullptr, "Failed to create input layout");

                nvrhi::BindingLayoutHandle layout0 = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Vertex, 0, true)
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0)));
                nvrhi::BindingLayoutHandle layout1 = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Pixel, 1, true)
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0)));
                UAA_CHECK(layout0 != nullptr && layout1 != nullptr, "Failed to create graphics binding layouts");

                nvrhi::BindingSetHandle set0 = dev->createBindingSet(
                    nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::ConstantBuffer(0, vsCB)), layout0);
                nvrhi::BindingSetHandle set1 = dev->createBindingSet(
                    nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::ConstantBuffer(0, psCB)), layout1);
                UAA_CHECK(set0 != nullptr && set1 != nullptr, "Failed to create graphics binding sets");

                nvrhi::TextureHandle rt = dev->createTexture(
                    nvrhi::TextureDesc().setWidth(4).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget).setDebugName("vii_rt"));
                UAA_CHECK(rt != nullptr, "Failed to create render target");

                nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
                UAA_CHECK(fb != nullptr, "Failed to create framebuffer");

                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.setInputLayout(inputLayout)
                    .setVertexShader(vs)
                    .setPixelShader(ps)
                    .addBindingLayout(layout0)
                    .addBindingLayout(layout1);
                psoDesc.renderState.rasterState.setCullNone();
                psoDesc.renderState.depthStencilState.disableDepthTest().disableDepthWrite();

                nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
                UAA_CHECK(pso != nullptr, "Failed to create graphics pipeline");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                nvrhi::GraphicsState gs;
                gs.setPipeline(pso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .addBindingSet(set0)
                    .addBindingSet(set1)
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb0).setSlot(0).setOffset(kOffset))
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb1).setSlot(1).setOffset(kOffset))
                    .setIndexBuffer(nvrhi::IndexBufferBinding().setBuffer(ib).setFormat(nvrhi::Format::R32_UINT).setOffset(static_cast<uint32_t>(kOffset)));
                cmd->setGraphicsState(gs);
                cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(6).setInstanceCount(1));
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(rt, 0, 0, pixels);
                UAA_CHECK_EQ(pixels.size(), size_t(4 * 4 * 4), "Unexpected render target pixel byte count");

                // Expected colour: (0.2*0.5 + 0.1, 0.4*0.5 + 0.2, 0.6*1.0 + 0.1, 1.0) = (0.2, 0.4, 0.7, 1.0) -> (51, 102, 179, 255).
                std::string diff;
                for (size_t p = 0; p < 16; ++p)
                {
                    UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[p * 4], 51, 102, 179, 255, 2, diff),
                        "drawIndexed pixel mismatch at pixel " + std::to_string(p) + ": " + diff);
                }
            }

            // Part 2: isDrawIndirectArgs buffer in dispatchIndirect at offset = 256.
            {
                constexpr uint32_t kOffset = 256;
                constexpr uint32_t kTotalOut = 16;
                const nvrhi::DispatchIndirectArguments args = nvrhi::DispatchIndirectArguments().setGroups3D(2, 1, 1); // 2 * 4 = 8 threads

                nvrhi::BufferHandle indirectBuf = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(kOffset + sizeof(args)).setIsDrawIndirectArgs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndirectArgument).setDebugName("vii_indirect"));
                nvrhi::BufferHandle outBuf = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(kTotalOut * sizeof(uint32_t)).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("vii_indirect_out"));
                UAA_CHECK(indirectBuf != nullptr && outBuf != nullptr, "Failed to create indirect buffers");

                fix.uploadBuffer(indirectBuf, &args, sizeof(args), kOffset);

                nvrhi::ShaderHandle cs = fix.createShader("cs_indirect_work", nvrhi::ShaderType::Compute);
                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(IndirectWorkPC)))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)));
                UAA_CHECK(layout != nullptr, "Failed to create indirect binding layout");

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(IndirectWorkPC)))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create indirect binding set");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create indirect compute pipeline");

                const IndirectWorkPC pc{ 4u, 0xCAFE0000u, 8u, 16u };
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearBufferUInt(outBuf, 0u);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set).setIndirectParams(indirectBuf));
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->dispatchIndirect(kOffset);
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> outWords(kTotalOut, 0u);
                fix.readbackBuffer(outBuf, outWords.data(), outWords.size() * sizeof(uint32_t), 0);
                for (uint32_t i = 0; i < kTotalOut; ++i)
                {
                    const uint32_t expected = (i >= 4u && i < 12u) ? (pc.marker + (i - 4u)) : 0u;
                    UAA_CHECK_EQ(outWords[i], expected, "dispatchIndirect output mismatch at index " + std::to_string(i));
                }
            }
        }

        // Tests dropping caller handles for inputs, CB, and binding set before executeCommandList.
        void testBufLifetime(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kCount = 32;
            constexpr uint64_t kByteSize = kCount * sizeof(uint32_t);
            constexpr uint32_t kAddVal = 42u;
            constexpr uint32_t kXorVal = 0x13572468u;
            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x9999ull);

            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("life_out"));
            UAA_CHECK(outBuf != nullptr, "Failed to create lifetime output buffer");

            nvrhi::CommandListHandle cmd;

            {
                nvrhi::BufferHandle inBuf = createWordBuffer(
                    fix,
                    nvrhi::BufferDesc().setByteSize(kByteSize).setStructStride(sizeof(uint32_t)).enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource).setDebugName("life_in"),
                    inputWords,
                    ctx);
                nvrhi::BufferHandle cb = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(sizeof(CopyAddParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("life_cb"));
                UAA_CHECK(cb != nullptr, "Failed to create lifetime CB");

                const CopyAddParams params{ kCount, kAddVal, kXorVal, 0u };
                fix.uploadBuffer(cb, &params, sizeof(params), 0);

                // Upload helpers use an immediate command list, so finish all uploads
                // before opening the compute list on this device.
                cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();

                nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
                UAA_CHECK(layout != nullptr, "Failed to create lifetime binding layout");

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create lifetime binding set");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create lifetime pipeline");

                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
            }

            cmd->close();
            // All caller references to inBuf, cb, set, layout, pso, cs are released before submit.
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), kByteSize, 0);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(actual[i], expected, "Lifetime output mismatch at index " + std::to_string(i));
            }
        }

        // Tests queryMemoryRequirements and getBufferMemoryRequirements across standard, volatile, and virtual buffers.
        void testBufMemoryRequirements(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            // Non-memory resource (IDevice) must return false and preserve caller output.
            nvrhi::MemoryRequirements preserved{ 123u, 456u };
            UAA_CHECK(!dev->queryMemoryRequirements(preserved), "IDevice::queryMemoryRequirements should return false");
            UAA_CHECK_EQ(preserved.size, 123ull, "IDevice::queryMemoryRequirements must preserve size");
            UAA_CHECK_EQ(preserved.alignment, 456ull, "IDevice::queryMemoryRequirements must preserve alignment");

            const bool isD3D11 = (fix.api() == nvrhi::GraphicsAPI::D3D11);
            const bool isMetal = (fix.api() == nvrhi::GraphicsAPI::METAL);

            for (const uint64_t byteSize : { 1024ull, 131072ull })
            {
                nvrhi::BufferHandle buf = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(byteSize).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("memreq_std"));
                UAA_CHECK(buf != nullptr, "Failed to create standard buffer");

                const nvrhi::MemoryRequirements legacy = dev->getBufferMemoryRequirements(buf);
                nvrhi::MemoryRequirements queried{ 111u, 222u };
                const bool ok = static_cast<nvrhi::IResource*>(buf.Get())->queryMemoryRequirements(queried);

                if (isD3D11)
                {
                    UAA_CHECK(!ok, "D3D11 queryMemoryRequirements should return false");
                    UAA_CHECK_EQ(queried.size, 111ull, "D3D11 queryMemoryRequirements must preserve size");
                    UAA_CHECK_EQ(queried.alignment, 222ull, "D3D11 queryMemoryRequirements must preserve alignment");
                }
                else
                {
                    UAA_CHECK(legacy.size >= byteSize, "getBufferMemoryRequirements size smaller than byteSize");
                    UAA_CHECK(isPowerOfTwo(legacy.alignment), "getBufferMemoryRequirements alignment must be power of two");
                    if (ok)
                    {
                        UAA_CHECK_EQ(queried.size, legacy.size, "queryMemoryRequirements size disagrees with legacy API");
                        UAA_CHECK_EQ(queried.alignment, legacy.alignment, "queryMemoryRequirements alignment disagrees with legacy API");
                    }
                    else
                    {
                        UAA_CHECK(isMetal, "Only Metal is expected to omit IBuffer::queryMemoryRequirements");
                        UAA_CHECK_EQ(queried.size, 111ull, "Preserved output size mismatch when queryMemoryRequirements returns false");
                        UAA_CHECK_EQ(queried.alignment, 222ull, "Preserved output alignment mismatch when queryMemoryRequirements returns false");
                    }
                }
            }

            // Volatile constant buffer.
            {
                nvrhi::BufferHandle volBuf = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(256).setIsConstantBuffer(true).setIsVolatile(true).setMaxVersions(4).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName("memreq_vol"));
                UAA_CHECK(volBuf != nullptr, "Failed to create volatile buffer");

                nvrhi::MemoryRequirements req{ 333u, 444u };
                const bool ok = volBuf->queryMemoryRequirements(req);
                if (fix.api() == nvrhi::GraphicsAPI::VULKAN)
                {
                    UAA_CHECK(ok, "Vulkan volatile buffer queryMemoryRequirements should return true");
                    UAA_CHECK(req.size >= 256ull * 4ull, "Vulkan volatile buffer should cover maxVersions backing storage");
                    UAA_CHECK(isPowerOfTwo(req.alignment), "Vulkan volatile alignment must be power of two");
                    const nvrhi::MemoryRequirements legacy = dev->getBufferMemoryRequirements(volBuf);
                    UAA_CHECK_EQ(req.size, legacy.size, "Volatile query size disagrees with legacy API");
                    UAA_CHECK_EQ(req.alignment, legacy.alignment, "Volatile query alignment disagrees with legacy API");
                }
                else if (!ok)
                {
                    UAA_CHECK_EQ(req.size, 333ull, "Volatile query preserved size mismatch");
                    UAA_CHECK_EQ(req.alignment, 444ull, "Volatile query preserved alignment mismatch");
                }
            }

            // Virtual buffer when supported.
            if (fix.features().supports(nvrhi::Feature::VirtualResources))
            {
                nvrhi::BufferHandle virtBuf = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(131072).setIsVirtual(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest).setDebugName("memreq_virt"));
                UAA_CHECK(virtBuf != nullptr, "Failed to create virtual buffer");

                nvrhi::MemoryRequirements virtReq{ 555u, 666u };
                const bool ok = virtBuf->queryMemoryRequirements(virtReq);
                const nvrhi::MemoryRequirements legacy = dev->getBufferMemoryRequirements(virtBuf);
                UAA_CHECK(legacy.size >= 131072ull, "Virtual buffer legacy size too small");
                UAA_CHECK(isPowerOfTwo(legacy.alignment), "Virtual buffer alignment must be power of two");
                if (ok)
                {
                    UAA_CHECK_EQ(virtReq.size, legacy.size, "Virtual buffer query size mismatch");
                    UAA_CHECK_EQ(virtReq.alignment, legacy.alignment, "Virtual buffer query alignment mismatch");
                }
            }

            // Invalid null buffer check under validation.
            if (fix.isValidationActive())
            {
                ExpectErrorsGuard guard(fix);
                const nvrhi::MemoryRequirements nullReq = dev->getBufferMemoryRequirements(nullptr);
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error for getBufferMemoryRequirements(nullptr)");
                UAA_CHECK_EQ(nullReq.size, 0ull, "Expected zero size for null buffer memory requirements");
            }
        }
    } // namespace

    UAA_REGISTER_TEST("BUF.Create", "core", testBufCreate);
    UAA_REGISTER_TEST("BUF.WriteCopyRead", "core", testBufWriteCopyRead);
    UAA_REGISTER_TEST("BUF.Overwrite", "core", testBufOverwrite);
    UAA_REGISTER_TEST("BUF.Clear", "core", testBufClear);
    UAA_REGISTER_TEST("BUF.Map", "core", testBufMap);
    UAA_REGISTER_TEST("BUF.MapInvalid", "core,validation", testBufMapInvalid);
    UAA_REGISTER_TEST("BUF.VolatileConstants", "core", testBufVolatileConstants);
    UAA_REGISTER_TEST("BUF.StructuredRaw", "core", testBufStructuredRaw);
    UAA_REGISTER_TEST("BUF.TypedViews", "core", testBufTypedViews);
    UAA_REGISTER_TEST("BUF.VertexIndexIndirect", "core", testBufVertexIndexIndirect);
    UAA_REGISTER_TEST("BUF.Lifetime", "core", testBufLifetime);
    UAA_REGISTER_TEST("BUF.MemoryRequirements", "core", testBufMemoryRequirements);
} // namespace nvrhi::uaa
