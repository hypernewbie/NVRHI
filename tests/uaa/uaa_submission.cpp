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

#include <nvrhi/utils.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
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
        static_assert(sizeof(CopyAddParams) == 16, "CopyAddParams size mismatch");

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

        struct FullscreenPC
        {
            float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            float depth = 0.5f;
            float scale = 1.0f;
            float offset[2] = {0.0f, 0.0f};
        };
        static_assert(sizeof(FullscreenPC) == 32, "FullscreenPC size mismatch");

        struct CopyAddBundle
        {
            ShaderHandle shader;
            BindingLayoutHandle layout;
            ComputePipelineHandle pipeline;
        };

        CopyAddBundle createCopyAddBundle(TestContext& ctx, TestFixture& fix)
        {
            CopyAddBundle bundle;
            bundle.shader = fix.createShader("cs_buffer_copy_add", ShaderType::Compute);
            UAA_CHECK(bundle.shader != nullptr, "cs_buffer_copy_add shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2));
            bundle.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "createBindingLayout for cs_buffer_copy_add returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(bundle.shader)
                .addBindingLayout(bundle.layout);
            bundle.pipeline = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(bundle.pipeline != nullptr, "createComputePipeline for cs_buffer_copy_add returned null");
            return bundle;
        }

        BufferHandle makeGpuBuffer(TestContext& ctx, IDevice* dev, uint64_t bytes, uint32_t stride, bool uav, const char* name)
        {
            BufferDesc desc;
            desc.setByteSize(bytes)
                .setStructStride(stride)
                .setCanHaveUAVs(uav)
                .setInitialState(ResourceStates::Common)
                .setKeepInitialState(true)
                .setDebugName(name);
            BufferHandle buf = dev->createBuffer(desc);
            UAA_CHECK(buf != nullptr, std::string("createBuffer ") + name + " returned null");
            return buf;
        }

        BufferHandle makeConstantBuffer(TestContext& ctx, IDevice* dev, uint64_t bytes, const char* name)
        {
            BufferDesc desc;
            desc.setByteSize(bytes)
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName(name);
            BufferHandle buf = dev->createBuffer(desc);
            UAA_CHECK(buf != nullptr, std::string("createBuffer ") + name + " returned null");
            return buf;
        }

        void verifyCopyAddWords(
            TestContext& ctx,
            const std::vector<uint32_t>& actual,
            const std::vector<uint32_t>& input,
            uint32_t addVal,
            uint32_t xorVal,
            const char* label)
        {
            UAA_CHECK_EQ(actual.size(), input.size(), label);
            for (size_t i = 0; i < input.size(); ++i)
            {
                const uint32_t expected = (input[i] + addVal) ^ xorVal;
                UAA_CHECK_EQ(actual[i], expected, std::string(label) + " word mismatch at " + std::to_string(i));
            }
        }

        // Tests command list open/close/submit/reopen lifecycle, clearState, getDevice, getDesc, and memory usage.
        void testSubListLifecycle(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 5;
            constexpr uint32_t kXorVal = 0x11223344u;

            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);
            BufferHandle inBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_life_in");
            BufferHandle outBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_life_out");
            BufferHandle cb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_life_cb");

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                bundle.layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");

            CommandListParameters params;
            params.setQueueType(CommandQueue::Graphics)
                .setEnableImmediateExecution(true);
            CommandListHandle cmd = dev->createCommandList(params);
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            UAA_CHECK_EQ(cmd->getDevice(), dev, "getDevice mismatch");
            UAA_CHECK_EQ(cmd->getDesc().queueType, CommandQueue::Graphics, "getDesc().queueType mismatch");
            UAA_CHECK_EQ(cmd->getDesc().enableImmediateExecution, true, "getDesc().enableImmediateExecution mismatch");

            // Empty open -> close -> submit must complete safely without hanging.
            cmd->open();
            cmd->close();
            dev->executeCommandList(cmd);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after empty command list");

            const CommandListMemoryUsage usageBefore = cmd->getInternalMemoryUsage();

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xA01u);
            const CopyAddParams cbParams{kCount, kAddVal, kXorVal, 0};

            // Reopen, record compute work, clearState, close, submit, and verify.
            cmd->open();
            cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            cmd->writeBuffer(cb, &cbParams, sizeof(cbParams));
            cmd->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(set));
            cmd->dispatch(1, 1, 1);
            cmd->clearState();
            cmd->close();

            fix.submitAndWait(cmd);

            const CommandListMemoryUsage usageAfter = cmd->getInternalMemoryUsage();
            UAA_CHECK(usageAfter.uploadBytes >= usageBefore.uploadBytes,
                "getInternalMemoryUsage uploadBytes must not decrease");
            UAA_CHECK(usageAfter.scratchBytes >= usageBefore.scratchBytes,
                "getInternalMemoryUsage scratchBytes must not decrease");

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            verifyCopyAddWords(ctx, actual, inputWords, kAddVal, kXorVal, "SUB.ListLifecycle");
        }

        // Tests executeCommandLists with sequential chained lists, empty lists, and zero-count calls.
        void testSubListArrays(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            // Zero-length array submission must return safely without hanging or erroring.
            const uint64_t zeroSerial = dev->executeCommandLists(nullptr, 0u);
            (void)zeroSerial;
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after executeCommandLists(nullptr, 0)");

            constexpr uint32_t kCount = 64;
            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);

            BufferHandle buf0 = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_arr_buf0");
            BufferHandle buf1 = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_arr_buf1");
            BufferHandle buf2 = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_arr_buf2");
            BufferHandle buf3 = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_arr_buf3");

            BufferHandle cbA = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_arr_cbA");
            BufferHandle cbB = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_arr_cbB");
            BufferHandle cbC = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_arr_cbC");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xA02u);
            const CopyAddParams paramsA{kCount, 3u, 0x0F0F0F0Fu, 0u};
            const CopyAddParams paramsB{kCount, 5u, 0x33333333u, 0u};
            const CopyAddParams paramsC{kCount, 7u, 0x55555555u, 0u};

            fix.uploadBuffer(buf0, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            fix.uploadBuffer(cbA, &paramsA, sizeof(paramsA));
            fix.uploadBuffer(cbB, &paramsB, sizeof(paramsB));
            fix.uploadBuffer(cbC, &paramsC, sizeof(paramsC));

            auto makeSet = [&](IBuffer* cb, IBuffer* src, IBuffer* dst) -> BindingSetHandle
            {
                BindingSetHandle s = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(BindingSetItem::StructuredBuffer_SRV(1, src))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, dst)),
                    bundle.layout);
                UAA_CHECK(s != nullptr, "createBindingSet returned null");
                return s;
            };

            BindingSetHandle setA = makeSet(cbA, buf0, buf1);
            BindingSetHandle setB = makeSet(cbB, buf1, buf2);
            BindingSetHandle setC = makeSet(cbC, buf2, buf3);

            CommandListHandle listA = dev->createCommandList();
            CommandListHandle listB = dev->createCommandList();
            CommandListHandle listC = dev->createCommandList();
            CommandListHandle emptyList = dev->createCommandList();
            UAA_CHECK(listA && listB && listC && emptyList, "createCommandList returned null");

            listA->open();
            listA->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(setA));
            listA->dispatch(1, 1, 1);
            listA->close();

            listB->open();
            listB->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(setB));
            listB->dispatch(1, 1, 1);
            listB->close();

            listC->open();
            listC->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(setC));
            listC->dispatch(1, 1, 1);
            listC->close();

            emptyList->open();
            emptyList->close();

            std::array<ICommandList*, 3> chainLists = {listA.Get(), listB.Get(), listC.Get()};
            dev->executeCommandLists(chainLists.data(), chainLists.size(), CommandQueue::Graphics);

            std::array<ICommandList*, 1> emptyOnly = {emptyList.Get()};
            dev->executeCommandLists(emptyOnly.data(), emptyOnly.size(), CommandQueue::Graphics);

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.ListArrays");
            dev->runGarbageCollection();

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(buf3, actual.data(), actual.size() * sizeof(uint32_t));
            for (uint32_t i = 0; i < kCount; ++i)
            {
                uint32_t expected = (inputWords[i] + paramsA.addVal) ^ paramsA.xorVal;
                expected = (expected + paramsB.addVal) ^ paramsB.xorVal;
                expected = (expected + paramsC.addVal) ^ paramsC.xorVal;
                UAA_CHECK_EQ(actual[i], expected, "SUB.ListArrays chained mismatch at " + std::to_string(i));
            }
        }

        // Tests monotonic submission serials returned by executeCommandList across three submissions.
        void testSubSerials(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kWords = 16;
            BufferHandle buf = makeGpuBuffer(ctx, dev, kWords * sizeof(uint32_t), sizeof(uint32_t), false, "sub_serial_buf");

            std::array<uint64_t, 3> serials = {0u, 0u, 0u};
            std::vector<uint32_t> finalWords;

            for (size_t i = 0; i < 3; ++i)
            {
                finalWords = TestFixture::makeDeterministicWords(kWords, ctx.seed() ^ (0xB00u + i));
                CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->writeBuffer(buf, finalWords.data(), finalWords.size() * sizeof(uint32_t));
                cmd->close();
                serials[i] = dev->executeCommandList(cmd);
            }

            if (fix.api() != GraphicsAPI::D3D11)
            {
                UAA_CHECK(serials[0] > 0u, "first submission serial must be > 0");
                UAA_CHECK(serials[1] > serials[0], "second submission serial must be strictly greater than first");
                UAA_CHECK(serials[2] > serials[1], "third submission serial must be strictly greater than second");
            }
            else
            {
                UAA_CHECK(serials[1] >= serials[0] && serials[2] >= serials[1],
                    "submission serials must be monotonic non-decreasing");
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.Serials");

            std::vector<uint32_t> readWords(kWords, 0u);
            fix.readbackBuffer(buf, readWords.data(), readWords.size() * sizeof(uint32_t));
            std::string diff;
            const bool eq = TestFixture::checkBytesEqual(
                readWords.data(), finalWords.data(), finalWords.size() * sizeof(uint32_t), diff);
            UAA_CHECK(eq, "SUB.Serials readback mismatch: " + diff);
        }

        // Runs a single producer-to-consumer cross-queue dependency test with optional GPU gate verification.
        void verifyCrossQueueDependency(
            TestContext& ctx,
            TestFixture& fix,
            const CopyAddBundle& bundle,
            CommandQueue producerQueue,
            CommandQueue consumerQueue,
            const char* label)
        {
            IDevice* dev = fix.device();
            constexpr uint32_t kCount = 64;
            constexpr uint32_t kProducerWord = 0x12345678u;
            const bool copyOnlyConsumer = (consumerQueue == CommandQueue::Copy);

            BufferHandle sharedBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_qdep_shared");
            BufferHandle outBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), !copyOnlyConsumer, "sub_qdep_out");
            BufferHandle cb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_qdep_cb");

            const CopyAddParams params{kCount, 1u, 0u, 0u};
            fix.uploadBuffer(cb, &params, sizeof(params));

            BindingSetHandle set;
            if (!copyOnlyConsumer)
            {
                set = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(BindingSetItem::StructuredBuffer_SRV(1, sharedBuf))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                    bundle.layout);
                UAA_CHECK(set != nullptr, "createBindingSet returned null");
            }

            const std::vector<uint32_t> producerWords(kCount, kProducerWord);

            CommandListHandle prodCmd = dev->createCommandList(CommandListParameters().setQueueType(producerQueue));
            UAA_CHECK(prodCmd != nullptr, "createCommandList producer returned null");
            prodCmd->open();
            prodCmd->writeBuffer(sharedBuf, producerWords.data(), producerWords.size() * sizeof(uint32_t));
            prodCmd->close();

            CommandListHandle consCmd = dev->createCommandList(CommandListParameters().setQueueType(consumerQueue));
            UAA_CHECK(consCmd != nullptr, "createCommandList consumer returned null");
            consCmd->open();
            if (copyOnlyConsumer)
            {
                consCmd->copyBuffer(outBuf, 0u, sharedBuf, 0u, kCount * sizeof(uint32_t));
            }
            else
            {
                consCmd->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(set));
                consCmd->dispatch(1, 1, 1);
            }
            consCmd->close();

            std::unique_ptr<IGpuGate> gate = fix.backend()->createGpuGate();
            bool gateArmed = false;
            if (gate)
            {
                gateArmed = gate->enqueueWait(producerQueue);
            }

            const uint64_t prodSerial = dev->executeCommandList(prodCmd, producerQueue);

            // Run queueWaitForCommandList on a worker thread with a bounded timeout so a CPU-blocking backend cannot deadlock.
            std::mutex mtx;
            std::condition_variable cv;
            bool waitReturned = false;
            std::thread waitThread([&]()
            {
                dev->queueWaitForCommandList(consumerQueue, producerQueue, prodSerial);
                std::lock_guard<std::mutex> lock(mtx);
                waitReturned = true;
                cv.notify_all();
            });

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait_for(lock, std::chrono::milliseconds(200), [&]() { return waitReturned; });
            }

            if (!waitReturned)
            {
                if (gateArmed)
                {
                    gate->signal();
                }
                waitThread.join();
                dev->waitForIdle();
                UAA_CHECK(false, std::string(label) + ": queueWaitForCommandList blocked CPU instead of waiting on GPU");
            }
            waitThread.join();

            EventQueryHandle consumerDone = dev->createEventQuery();
            UAA_CHECK(consumerDone != nullptr, "createEventQuery returned null");
            dev->executeCommandList(consCmd, consumerQueue);
            dev->setEventQuery(consumerDone, consumerQueue);

            if (gateArmed)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                const bool completedEarly = dev->pollEventQuery(consumerDone);
                gate->signal();
                dev->waitEventQuery(consumerDone);
                UAA_CHECK(!completedEarly,
                    std::string(label) + ": consumer queue completed before gated producer queue was signalled");
            }
            else
            {
                dev->waitEventQuery(consumerDone);
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after cross-queue work");

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            const uint32_t expectedWord = copyOnlyConsumer ? kProducerWord : (kProducerWord + 1u);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                UAA_CHECK_EQ(actual[i], expectedWord,
                    std::string(label) + " output mismatch at index " + std::to_string(i));
            }
        }

        // Tests cross-queue dependencies across Graphics, Compute, and Copy queues.
        void testSubQueueDependencies(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            const bool hasCompute = fix.features().supports(Feature::ComputeQueue);
            const bool hasCopy = fix.features().supports(Feature::CopyQueue);
            if (!hasCompute && !hasCopy)
            {
                ctx.skip("Neither Feature::ComputeQueue nor Feature::CopyQueue is supported by this backend");
            }

            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);

            if (hasCompute)
            {
                verifyCrossQueueDependency(ctx, fix, bundle, CommandQueue::Graphics, CommandQueue::Compute,
                    "SUB.QueueDependencies Graphics->Compute");
                verifyCrossQueueDependency(ctx, fix, bundle, CommandQueue::Compute, CommandQueue::Graphics,
                    "SUB.QueueDependencies Compute->Graphics");
            }

            if (hasCopy)
            {
                verifyCrossQueueDependency(ctx, fix, bundle, CommandQueue::Graphics, CommandQueue::Copy,
                    "SUB.QueueDependencies Graphics->Copy");
            }
        }

        // Tests that queueWaitForCommandList does not block the calling CPU thread while a GPU queue is gated.
        void testSubWaitSemantics(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            std::unique_ptr<IGpuGate> gate = fix.backend()->createGpuGate();
            if (!gate)
            {
                ctx.skip("IGpuGate is not available on this backend");
            }

            const CommandQueue producerQueue = CommandQueue::Graphics;
            const CommandQueue consumerQueue = fix.features().supports(Feature::ComputeQueue)
                ? CommandQueue::Compute
                : (fix.features().supports(Feature::CopyQueue) ? CommandQueue::Copy : CommandQueue::Graphics);

            BufferHandle buf = makeGpuBuffer(ctx, dev, 64u, sizeof(uint32_t), false, "sub_wait_sem_buf");
            const std::array<uint32_t, 4> payload = {0xCAFEBABEu, 1u, 2u, 3u};

            CommandListHandle cmd = dev->createCommandList(CommandListParameters().setQueueType(producerQueue));
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(buf, payload.data(), sizeof(payload));
            cmd->close();

            if (!gate->enqueueWait(producerQueue))
            {
                ctx.skip("IGpuGate::enqueueWait returned false for Graphics queue");
            }

            const uint64_t serial = dev->executeCommandList(cmd, producerQueue);

            std::mutex mtx;
            std::condition_variable cv;
            bool returned = false;

            std::thread worker([&]()
            {
                dev->queueWaitForCommandList(consumerQueue, producerQueue, serial);
                std::lock_guard<std::mutex> lock(mtx);
                returned = true;
                cv.notify_all();
            });

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait_for(lock, std::chrono::milliseconds(200), [&]() { return returned; });
            }

            const bool blockedCpu = !returned;
            gate->signal();
            worker.join();
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.WaitSemantics");

            UAA_CHECK(!blockedCpu,
                "queueWaitForCommandList blocked the CPU thread while producer queue was gated");
        }

        // Tests waitForIdle before submission, after an empty submission, and after real GPU work.
        void testSubIdle(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed on already-idle device");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->close();
            dev->executeCommandList(cmd);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after empty command list");

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 9;
            constexpr uint32_t kXorVal = 0x76543210u;
            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);

            BufferHandle inBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_idle_in");
            BufferHandle outBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_idle_out");
            BufferHandle cb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_idle_cb");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xC01u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0u};
            fix.uploadBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            fix.uploadBuffer(cb, &params, sizeof(params));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                bundle.layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");

            cmd->open();
            cmd->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(set));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            dev->executeCommandList(cmd);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after compute submission");

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            verifyCopyAddWords(ctx, actual, inputWords, kAddVal, kXorVal, "SUB.Idle");
        }

        // Tests external CommandListLifetimeTracker creation and garbage collection after completion.
        void testSubLifetimeTracker(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            CommandListLifetimeTrackerHandle tracker = dev->createCommandListLifetimeTracker(CommandQueue::Graphics);
            if (!tracker)
            {
                ctx.skip("createCommandListLifetimeTracker is not supported by this backend");
            }

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 17;
            constexpr uint32_t kXorVal = 0x89ABCDEFu;
            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xD01u);
            BufferHandle outBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_lt_out");

            {
                const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);
                BufferHandle inBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_lt_in");
                BufferHandle cb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_lt_cb");
                const CopyAddParams params{kCount, kAddVal, kXorVal, 0u};

                BindingSetHandle set = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                    bundle.layout);
                UAA_CHECK(set != nullptr, "createBindingSet returned null");

                CommandListHandle cmd = dev->createCommandList(
                    CommandListParameters()
                        .setQueueType(CommandQueue::Graphics)
                        .setLifetimeTracker(tracker.Get()));
                UAA_CHECK(cmd != nullptr, "createCommandList with lifetimeTracker returned null");

                cmd->open();
                cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
                cmd->writeBuffer(cb, &params, sizeof(params));
                cmd->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();

                dev->executeCommandList(cmd);
                tracker->runGarbageCollection();
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.LifetimeTracker");
            tracker->runGarbageCollection();

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            verifyCopyAddWords(ctx, actual, inputWords, kAddVal, kXorVal, "SUB.LifetimeTracker");
        }

        // Tests that command lists retain all referenced graphics and compute resources after caller handles drop.
        void testSubResourceRetention(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kRtSize = 8;
            constexpr uint32_t kAddVal = 21;
            constexpr uint32_t kXorVal = 0xA5A55A5Au;

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xE01u);

            const size_t computeBytes = size_t(kCount) * sizeof(uint32_t);
            const size_t sampleBytes = size_t(kRtSize) * kRtSize * 4u * sizeof(float);

            BufferDesc rbDesc;
            rbDesc.setByteSize(computeBytes + sampleBytes)
                .setCpuAccess(CpuAccessMode::Read)
                .setDebugName("sub_retain_rb");
            BufferHandle rbBuf = dev->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "createBuffer rbBuf returned null");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            {
                const CopyAddBundle copyBundle = createCopyAddBundle(ctx, fix);

                ShaderHandle vs = fix.createShader("vs_fullscreen_tri", ShaderType::Vertex);
                ShaderHandle ps = fix.createShader("ps_solid_color", ShaderType::Pixel);
                ShaderHandle csSamp = fix.createShader("cs_sampler_test", ShaderType::Compute);
                UAA_CHECK(vs && ps && csSamp, "shaders must not be null");

                BindingLayoutHandle gfxLayout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(ShaderType::AllGraphics)
                        .addItem(BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC))));
                UAA_CHECK(gfxLayout != nullptr, "createBindingLayout gfxLayout returned null");

                BindingSetHandle gfxSet = dev->createBindingSet(
                    BindingSetDesc().addItem(BindingSetItem::PushConstants(0, sizeof(FullscreenPC))),
                    gfxLayout);
                UAA_CHECK(gfxSet != nullptr, "createBindingSet gfxSet returned null");

                BindingLayoutHandle sampLayout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(ShaderType::Compute)
                        .addItem(BindingLayoutItem::Texture_SRV(0))
                        .addItem(BindingLayoutItem::Sampler(1))
                        .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                        .addItem(BindingLayoutItem::ConstantBuffer(3)));
                UAA_CHECK(sampLayout != nullptr, "createBindingLayout sampLayout returned null");

                SamplerHandle sampler = dev->createSampler(
                    SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::ClampToEdge));
                UAA_CHECK(sampler != nullptr, "createSampler returned null");

                ComputePipelineHandle sampPso = dev->createComputePipeline(
                    ComputePipelineDesc().setComputeShader(csSamp).addBindingLayout(sampLayout));
                UAA_CHECK(sampPso != nullptr, "createComputePipeline sampPso returned null");

                TextureDesc rtDesc;
                rtDesc.setWidth(kRtSize)
                    .setHeight(kRtSize)
                    .setFormat(Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .setInitialState(ResourceStates::RenderTarget)
                    .setKeepInitialState(true)
                    .setDebugName("sub_retain_rt");
                TextureHandle rtTex = dev->createTexture(rtDesc);
                UAA_CHECK(rtTex != nullptr, "createTexture rtTex returned null");

                FramebufferHandle fb = dev->createFramebuffer(FramebufferDesc().addColorAttachment(rtTex));
                UAA_CHECK(fb != nullptr, "createFramebuffer returned null");

                GraphicsPipelineDesc gfxPsoDesc;
                gfxPsoDesc.setVertexShader(vs)
                    .setPixelShader(ps)
                    .addBindingLayout(gfxLayout);
                gfxPsoDesc.renderState.rasterState.setCullMode(RasterCullMode::None);
                gfxPsoDesc.renderState.depthStencilState.setDepthTestEnable(false).setDepthWriteEnable(false);
                GraphicsPipelineHandle gfxPso = dev->createGraphicsPipeline(gfxPsoDesc, fb->getFramebufferInfo());
                UAA_CHECK(gfxPso != nullptr, "createGraphicsPipeline returned null");

                BufferHandle inBuf = makeGpuBuffer(ctx, dev, computeBytes, sizeof(uint32_t), false, "sub_retain_in");
                BufferHandle outComputeBuf = makeGpuBuffer(ctx, dev, computeBytes, sizeof(uint32_t), true, "sub_retain_out_c");
                BufferHandle outSampleBuf = makeGpuBuffer(ctx, dev, sampleBytes, 16u, true, "sub_retain_out_s");
                BufferHandle copyCb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_retain_copy_cb");
                BufferHandle sampCb = makeConstantBuffer(ctx, dev, sizeof(SamplerTestParams), "sub_retain_samp_cb");

                BindingSetHandle copySet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::ConstantBuffer(0, copyCb))
                        .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outComputeBuf)),
                    copyBundle.layout);
                UAA_CHECK(copySet != nullptr, "createBindingSet copySet returned null");

                BindingSetHandle sampSet = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::Texture_SRV(0, rtTex))
                        .addItem(BindingSetItem::Sampler(1, sampler))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outSampleBuf))
                        .addItem(BindingSetItem::ConstantBuffer(3, sampCb)),
                    sampLayout);
                UAA_CHECK(sampSet != nullptr, "createBindingSet sampSet returned null");

                const CopyAddParams copyParams{kCount, kAddVal, kXorVal, 0u};
                SamplerTestParams sampParams{};
                sampParams.uvOrigin[0] = 0.5f / float(kRtSize);
                sampParams.uvOrigin[1] = 0.5f / float(kRtSize);
                sampParams.uvStep[0] = 1.0f / float(kRtSize);
                sampParams.uvStep[1] = 1.0f / float(kRtSize);
                sampParams.width = kRtSize;
                sampParams.height = kRtSize;

                FullscreenPC pc{};
                pc.color[0] = 0.30f;
                pc.color[1] = 0.65f;
                pc.color[2] = 0.90f;
                pc.color[3] = 1.00f;
                pc.depth = 0.5f;
                pc.scale = 1.0f;

                cmd->open();
                cmd->writeBuffer(inBuf, inputWords.data(), computeBytes);
                cmd->writeBuffer(copyCb, &copyParams, sizeof(copyParams));
                cmd->writeBuffer(sampCb, &sampParams, sizeof(sampParams));

                utils::ClearColorAttachment(cmd, fb, 0, Color(0.0f, 0.0f, 0.0f, 1.0f));
                GraphicsState gfxState;
                gfxState.setPipeline(gfxPso)
                    .setFramebuffer(fb)
                    .setViewport(ViewportState().addViewportAndScissorRect(Viewport(float(kRtSize), float(kRtSize))))
                    .addBindingSet(gfxSet);
                cmd->setGraphicsState(gfxState);
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(DrawArguments().setVertexCount(3));

                cmd->setComputeState(ComputeState().setPipeline(sampPso).addBindingSet(sampSet));
                cmd->dispatch(1, 1, 1);

                cmd->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(copySet));
                cmd->dispatch(1, 1, 1);

                cmd->copyBuffer(rbBuf, 0u, outComputeBuf, 0u, computeBytes);
                cmd->copyBuffer(rbBuf, computeBytes, outSampleBuf, 0u, sampleBytes);
                cmd->close();
            }
            // All buffers, textures, views, shaders, pipelines, layouts, and binding sets are released before submission.

            dev->executeCommandList(cmd);
            cmd = nullptr;

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.ResourceRetention");
            dev->runGarbageCollection();

            const void* mapped = dev->mapBuffer(rbBuf, CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer rbBuf returned null");

            const auto* computePtr = static_cast<const uint32_t*>(mapped);
            std::vector<uint32_t> actualCompute(computePtr, computePtr + kCount);
            const auto* samplePtr = reinterpret_cast<const float*>(static_cast<const uint8_t*>(mapped) + computeBytes);
            const float sampleR = samplePtr[0];
            const float sampleG = samplePtr[1];
            const float sampleB = samplePtr[2];
            const float sampleA = samplePtr[3];
            dev->unmapBuffer(rbBuf);

            verifyCopyAddWords(ctx, actualCompute, inputWords, kAddVal, kXorVal, "SUB.ResourceRetention compute");
            constexpr float kEps = 1.0f / 255.0f + 1e-3f;
            UAA_CHECK_NEAR(sampleR, 0.30f, kEps, "SUB.ResourceRetention render target R");
            UAA_CHECK_NEAR(sampleG, 0.65f, kEps, "SUB.ResourceRetention render target G");
            UAA_CHECK_NEAR(sampleB, 0.90f, kEps, "SUB.ResourceRetention render target B");
            UAA_CHECK_NEAR(sampleA, 1.00f, kEps, "SUB.ResourceRetention render target A");
        }

        // Tests concurrent command-list recording across four worker threads followed by main-thread submission.
        void testSubThreading(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::DeferredCommandLists))
            {
                ctx.skip("Feature::DeferredCommandLists is not supported by this backend");
            }

            IDevice* dev = fix.device();
            constexpr size_t kThreads = 4;
            constexpr uint32_t kCount = 64;

            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);

            std::array<CommandListHandle, kThreads> cmds;
            std::array<BufferHandle, kThreads> inBufs;
            std::array<BufferHandle, kThreads> outBufs;
            std::array<BufferHandle, kThreads> cbs;
            std::array<BindingSetHandle, kThreads> sets;
            std::array<std::vector<uint32_t>, kThreads> inputWords;
            std::array<CopyAddParams, kThreads> params;

            for (size_t t = 0; t < kThreads; ++t)
            {
                cmds[t] = dev->createCommandList(CommandListParameters()
                    .setEnableImmediateExecution(false)
                    .setQueueType(CommandQueue::Graphics));
                UAA_CHECK(cmds[t] != nullptr, "createCommandList deferred returned null");

                inBufs[t] = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_thr_in");
                outBufs[t] = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_thr_out");
                cbs[t] = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_thr_cb");
                sets[t] = dev->createBindingSet(
                    BindingSetDesc()
                        .addItem(BindingSetItem::ConstantBuffer(0, cbs[t]))
                        .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBufs[t]))
                        .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBufs[t])),
                    bundle.layout);
                UAA_CHECK(sets[t] != nullptr, "createBindingSet returned null");

                inputWords[t] = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ (0xF00u + t));
                params[t] = CopyAddParams{kCount, static_cast<uint32_t>(10u + t * 7u), static_cast<uint32_t>(0x11111111u * (t + 1u)), 0u};
            }

            std::array<std::thread, kThreads> workers;
            for (size_t t = 0; t < kThreads; ++t)
            {
                workers[t] = std::thread([&, t]()
                {
                    cmds[t]->open();
                    cmds[t]->writeBuffer(inBufs[t], inputWords[t].data(), inputWords[t].size() * sizeof(uint32_t));
                    cmds[t]->writeBuffer(cbs[t], &params[t], sizeof(params[t]));
                    cmds[t]->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(sets[t]));
                    cmds[t]->dispatch(1, 1, 1);
                    cmds[t]->close();
                });
            }

            for (size_t t = 0; t < kThreads; ++t)
            {
                workers[t].join();
            }

            std::array<ICommandList*, kThreads> rawCmds = {
                cmds[0].Get(), cmds[1].Get(), cmds[2].Get(), cmds[3].Get()
            };
            dev->executeCommandLists(rawCmds.data(), rawCmds.size(), CommandQueue::Graphics);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in SUB.Threading");

            for (size_t t = 0; t < kThreads; ++t)
            {
                std::vector<uint32_t> actual(kCount, 0u);
                fix.readbackBuffer(outBufs[t], actual.data(), actual.size() * sizeof(uint32_t));
                verifyCopyAddWords(ctx, actual, inputWords[t], params[t].addVal, params[t].xorVal, "SUB.Threading");
            }
        }

        // Tests that an isolated API/validation error does not poison subsequent valid command list submissions.
        void testSubFailureRecovery(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            // Trigger an isolated NVRHI-level failure without emitting native API validation errors.
            {
                ExpectErrorsGuard guard(fix);
                if (fix.isValidationActive())
                {
                    TextureDesc badTexDesc;
                    badTexDesc.setWidth(0)
                        .setHeight(0)
                        .setFormat(Format::RGBA8_UNORM)
                        .setDebugName("sub_fail_bad_tex");
                    TextureHandle badTex = dev->createTexture(badTexDesc);
                    UAA_CHECK(badTex == nullptr, "invalid 0x0 texture must be rejected by validation");
                    UAA_CHECK(guard.errorCount() > 0u, "expected validation error for 0x0 texture");
                }
                else
                {
                    TextureDesc nonVirtualDesc;
                    nonVirtualDesc.setWidth(16)
                        .setHeight(16)
                        .setFormat(Format::RGBA8_UNORM)
                        .setIsVirtual(false)
                        .setDebugName("sub_fail_nonvirtual");
                    TextureHandle tex = dev->createTexture(nonVirtualDesc);
                    UAA_CHECK(tex != nullptr, "createTexture non-virtual returned null");
                    const bool bound = dev->bindTextureMemory(tex, nullptr, 0u);
                    UAA_CHECK(!bound, "bindTextureMemory with null heap and non-virtual texture must fail");
                }
            }

            // Now submit a fresh valid compute command list and verify exact output.
            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 29;
            constexpr uint32_t kXorVal = 0xCAFE1234u;
            const CopyAddBundle bundle = createCopyAddBundle(ctx, fix);

            BufferHandle inBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), false, "sub_rec_in");
            BufferHandle outBuf = makeGpuBuffer(ctx, dev, kCount * sizeof(uint32_t), sizeof(uint32_t), true, "sub_rec_out");
            BufferHandle cb = makeConstantBuffer(ctx, dev, sizeof(CopyAddParams), "sub_rec_cb");

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
                bundle.layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xFA1u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0u};

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            cmd->writeBuffer(cb, &params, sizeof(params));
            cmd->setComputeState(ComputeState().setPipeline(bundle.pipeline).addBindingSet(set));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            verifyCopyAddWords(ctx, actual, inputWords, kAddVal, kXorVal, "SUB.FailureRecovery");
        }
    } // namespace

    UAA_REGISTER_TEST("SUB.ListLifecycle", "core", testSubListLifecycle);
    UAA_REGISTER_TEST("SUB.ListArrays", "core", testSubListArrays);
    UAA_REGISTER_TEST("SUB.Serials", "core", testSubSerials);
    UAA_REGISTER_TEST("SUB.QueueDependencies", "core", testSubQueueDependencies);
    UAA_REGISTER_TEST("SUB.WaitSemantics", "core", testSubWaitSemantics);
    UAA_REGISTER_TEST("SUB.Idle", "core", testSubIdle);
    UAA_REGISTER_TEST("SUB.LifetimeTracker", "core", testSubLifetimeTracker);
    UAA_REGISTER_TEST("SUB.ResourceRetention", "core", testSubResourceRetention);
    UAA_REGISTER_TEST("SUB.Threading", "core", testSubThreading);
    UAA_REGISTER_TEST("SUB.FailureRecovery", "core", testSubFailureRecovery);
} // namespace nvrhi::uaa
