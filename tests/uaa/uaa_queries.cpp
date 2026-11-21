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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
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

        struct SolidColorPC
        {
            float color[4] = {0.f, 0.f, 0.f, 1.f};
            float depth = 0.f;
            float scale = 1.f;
            float offset[2] = {0.f, 0.f};
        };

        struct ComputeCopyAddJob
        {
            nvrhi::ShaderHandle shader;
            nvrhi::BindingLayoutHandle layout;
            nvrhi::ComputePipelineHandle pipeline;
            nvrhi::BufferHandle cb;
            nvrhi::BufferHandle src;
            nvrhi::BufferHandle dst;
            nvrhi::BindingSetHandle bindingSet;
            uint32_t count = 0;
            uint32_t addVal = 0;
            uint32_t xorVal = 0;
            std::vector<uint32_t> inputWords;
        };

        ComputeCopyAddJob buildComputeCopyAddJob(
            TestFixture& fix,
            TestContext& ctx,
            uint32_t count,
            uint32_t addVal,
            uint32_t xorVal,
            uint64_t seed)
        {
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            ComputeCopyAddJob job;
            job.count = count;
            job.addVal = addVal;
            job.xorVal = xorVal;
            job.inputWords = TestFixture::makeDeterministicWords(count, seed);

            job.shader = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            UAA_CHECK(job.shader != nullptr, "Failed to create cs_buffer_copy_add shader");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1));
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));

            job.layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(job.layout != nullptr, "Failed to create compute binding layout");

            nvrhi::ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(job.shader).addBindingLayout(job.layout);
            job.pipeline = dev->createComputePipeline(psoDesc);
            UAA_CHECK(job.pipeline != nullptr, "Failed to create compute pipeline");

            nvrhi::BufferDesc cbDesc;
            cbDesc.byteSize = sizeof(CopyAddParams);
            cbDesc.isConstantBuffer = true;
            cbDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
            cbDesc.keepInitialState = true;
            cbDesc.debugName = "qry_copy_add_cb";
            job.cb = dev->createBuffer(cbDesc);
            UAA_CHECK(job.cb != nullptr, "Failed to create constant buffer");

            CopyAddParams params{count, addVal, xorVal, 0u};
            fix.uploadBuffer(job.cb, &params, sizeof(params), 0);

            const uint64_t dataBytes = uint64_t(count) * sizeof(uint32_t);

            nvrhi::BufferDesc srcDesc;
            srcDesc.byteSize = dataBytes;
            srcDesc.structStride = sizeof(uint32_t);
            srcDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            srcDesc.keepInitialState = true;
            srcDesc.debugName = "qry_copy_add_src";
            job.src = dev->createBuffer(srcDesc);
            UAA_CHECK(job.src != nullptr, "Failed to create source buffer");
            fix.uploadBuffer(job.src, job.inputWords.data(), dataBytes, 0);

            nvrhi::BufferDesc dstDesc;
            dstDesc.byteSize = dataBytes;
            dstDesc.structStride = sizeof(uint32_t);
            dstDesc.canHaveUAVs = true;
            dstDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            dstDesc.keepInitialState = true;
            dstDesc.debugName = "qry_copy_add_dst";
            job.dst = dev->createBuffer(dstDesc);
            UAA_CHECK(job.dst != nullptr, "Failed to create destination buffer");

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, job.cb));
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, job.src));
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, job.dst));
            job.bindingSet = dev->createBindingSet(setDesc, job.layout);
            UAA_CHECK(job.bindingSet != nullptr, "Failed to create compute binding set");

            return job;
        }

        void recordComputeCopyAdd(nvrhi::ICommandList* cmd, const ComputeCopyAddJob& job)
        {
            nvrhi::ComputeState state;
            state.setPipeline(job.pipeline).addBindingSet(job.bindingSet);
            cmd->setComputeState(state);
            const uint32_t groupsX = (job.count + 63u) / 64u;
            cmd->dispatch(groupsX, 1u, 1u);
        }

        void verifyComputeCopyAddOutput(TestFixture& fix, TestContext& ctx, const ComputeCopyAddJob& job)
        {
            std::vector<uint32_t> actual(job.count, 0u);
            fix.readbackBuffer(job.dst, actual.data(), size_t(job.count) * sizeof(uint32_t), 0);

            std::vector<uint32_t> expected(job.count, 0u);
            for (uint32_t i = 0; i < job.count; ++i)
            {
                expected[i] = (job.inputWords[i] + job.addVal) ^ job.xorVal;
            }

            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(
                    actual.data(),
                    expected.data(),
                    size_t(job.count) * sizeof(uint32_t),
                    diff),
                "Compute output mismatch: " + diff);
        }

        void testEventQueryLifecycle(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            nvrhi::EventQueryHandle query = dev->createEventQuery();
            UAA_CHECK(query != nullptr, "createEventQuery returned null");

            // Check reset and poll state before any work is attached.
            dev->resetEventQuery(query);
            const bool initialPoll = dev->pollEventQuery(query);
            ctx.note(std::string("Initial pollEventQuery after reset returned ") + (initialPoll ? "true" : "false"));

            ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, 4096u, 17u, 0xA55Au, ctx.seed() ^ 0x101ull);

            std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
            bool usedGate = false;
            if (gate && gate->enqueueWait(nvrhi::CommandQueue::Graphics))
            {
                usedGate = true;
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");

                cmd->open();
                recordComputeCopyAdd(cmd, job);
                cmd->close();

                dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
                dev->setEventQuery(query, nvrhi::CommandQueue::Graphics);

                UAA_CHECK(!dev->pollEventQuery(query), "Event query must remain unready while GPU gate is blocked");

                gate->signal();
                dev->waitEventQuery(query);
                UAA_CHECK(dev->pollEventQuery(query), "Event query must be ready after gate release and waitEventQuery");
            }
            else
            {
                ctx.note("IGpuGate unavailable on Graphics queue; verifying event query against real GPU compute work");
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");

                cmd->open();
                recordComputeCopyAdd(cmd, job);
                cmd->close();

                dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
                dev->setEventQuery(query, nvrhi::CommandQueue::Graphics);
                dev->waitEventQuery(query);
                UAA_CHECK(dev->pollEventQuery(query), "Event query must be ready after waitEventQuery");
            }

            verifyComputeCopyAddOutput(fix, ctx, job);

            // Reset and reuse the same event query for a second pass.
            dev->resetEventQuery(query);

            ComputeCopyAddJob job2 = buildComputeCopyAddJob(fix, ctx, 2048u, 33u, 0x1357u, ctx.seed() ^ 0x202ull);
            if (usedGate)
            {
                gate->reset();
                if (gate->enqueueWait(nvrhi::CommandQueue::Graphics))
                {
                    nvrhi::CommandListHandle cmd2 = dev->createCommandList();
                    UAA_CHECK(cmd2 != nullptr, "createCommandList returned null on reuse");
                    cmd2->open();
                    recordComputeCopyAdd(cmd2, job2);
                    cmd2->close();

                    dev->executeCommandList(cmd2, nvrhi::CommandQueue::Graphics);
                    dev->setEventQuery(query, nvrhi::CommandQueue::Graphics);
                    UAA_CHECK(!dev->pollEventQuery(query), "Reused event query must remain unready while GPU gate is blocked");

                    gate->signal();
                    dev->waitEventQuery(query);
                    UAA_CHECK(dev->pollEventQuery(query), "Reused event query must be ready after waitEventQuery");
                }
                else
                {
                    usedGate = false;
                }
            }

            if (!usedGate)
            {
                nvrhi::CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "createCommandList returned null on reuse");
                cmd2->open();
                recordComputeCopyAdd(cmd2, job2);
                cmd2->close();

                dev->executeCommandList(cmd2, nvrhi::CommandQueue::Graphics);
                dev->setEventQuery(query, nvrhi::CommandQueue::Graphics);
                dev->waitEventQuery(query);
                UAA_CHECK(dev->pollEventQuery(query), "Reused event query must be ready after waitEventQuery");
            }

            verifyComputeCopyAddOutput(fix, ctx, job2);
            dev->resetEventQuery(query);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
            dev->runGarbageCollection();
        }

        void testEventQueryAcrossQueues(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            std::vector<nvrhi::CommandQueue> queues = {nvrhi::CommandQueue::Graphics};
            if (fix.features().supports(nvrhi::Feature::ComputeQueue))
            {
                queues.push_back(nvrhi::CommandQueue::Compute);
            }
            if (fix.features().supports(nvrhi::Feature::CopyQueue))
            {
                queues.push_back(nvrhi::CommandQueue::Copy);
            }

            nvrhi::EventQueryHandle query = dev->createEventQuery();
            UAA_CHECK(query != nullptr, "createEventQuery returned null");

            for (nvrhi::CommandQueue queue : queues)
            {
                // Submit valid copy work on the target queue and track completion via the event query.
                constexpr size_t kCopyBytes = 4096;
                const std::vector<uint8_t> payload = TestFixture::makeDeterministicBytes(kCopyBytes, ctx.seed() ^ uint64_t(queue));

                nvrhi::BufferDesc srcDesc;
                srcDesc.byteSize = kCopyBytes;
                srcDesc.initialState = nvrhi::ResourceStates::CopySource;
                srcDesc.keepInitialState = true;
                srcDesc.debugName = "qry_queue_src";
                nvrhi::BufferHandle src = dev->createBuffer(srcDesc);
                UAA_CHECK(src != nullptr, "Failed to create queue test source buffer");
                fix.uploadBuffer(src, payload.data(), kCopyBytes, 0);

                nvrhi::BufferDesc dstDesc;
                dstDesc.byteSize = kCopyBytes;
                dstDesc.initialState = nvrhi::ResourceStates::CopyDest;
                dstDesc.keepInitialState = true;
                dstDesc.debugName = "qry_queue_dst";
                nvrhi::BufferHandle dst = dev->createBuffer(dstDesc);
                UAA_CHECK(dst != nullptr, "Failed to create queue test destination buffer");

                nvrhi::CommandListParameters params;
                params.setQueueType(queue);
                nvrhi::CommandListHandle cmd = dev->createCommandList(params);
                UAA_CHECK(cmd != nullptr, "createCommandList failed for supported queue");

                cmd->open();
                cmd->copyBuffer(dst, 0, src, 0, kCopyBytes);
                cmd->close();

                dev->executeCommandList(cmd, queue);
                dev->setEventQuery(query, queue);
                dev->waitEventQuery(query);
                UAA_CHECK(dev->pollEventQuery(query), "Event query must be signalled after queue copy completes");

                std::vector<uint8_t> actual(kCopyBytes, 0);
                fix.readbackBuffer(dst, actual.data(), kCopyBytes, 0);
                std::string diff;
                UAA_CHECK(
                    TestFixture::checkBytesEqual(actual.data(), payload.data(), kCopyBytes, diff),
                    "Queue copy verification failed: " + diff);

                dev->resetEventQuery(query);
            }

            // Drop event query handle after work completes and ensure clean GC.
            query = nullptr;
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
            dev->runGarbageCollection();
        }

        void testTimerQuery(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            nvrhi::TimerQueryHandle tq = dev->createTimerQuery();
            UAA_CHECK(tq != nullptr, "createTimerQuery returned null");

            constexpr uint32_t kElementCount = 65536u;
            ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, kElementCount, 7u, 0x55AAu, ctx.seed() ^ 0x303ull);

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            cmd->open();
            cmd->beginTimerQuery(tq);
            for (uint32_t iter = 0; iter < 4u; ++iter)
            {
                recordComputeCopyAdd(cmd, job);
            }
            cmd->endTimerQuery(tq);
            cmd->close();

            fix.submitAndWait(cmd, nvrhi::CommandQueue::Graphics);

            const bool resolved = fix.waitWithWatchdog([&]() {
                return dev->pollTimerQuery(tq);
            }, 2000u);

            if (!resolved)
            {
                UAA_CHECK(false, "Timer query did not resolve within watchdog timeout");
            }

            const float elapsed1 = dev->getTimerQueryTime(tq);
            UAA_CHECK(
                std::isfinite(elapsed1) && elapsed1 >= 0.0f,
                "First timer query elapsed time must be finite and non-negative");
            verifyComputeCopyAddOutput(fix, ctx, job);

            // Reset and reuse the timer query.
            dev->resetTimerQuery(tq);
            UAA_CHECK(!dev->pollTimerQuery(tq), "Reset timer query must not report ready before reuse");

            ComputeCopyAddJob job2 = buildComputeCopyAddJob(fix, ctx, kElementCount, 19u, 0xF00Du, ctx.seed() ^ 0x404ull);
            cmd->open();
            cmd->beginTimerQuery(tq);
            recordComputeCopyAdd(cmd, job2);
            cmd->endTimerQuery(tq);
            cmd->close();

            fix.submitAndWait(cmd, nvrhi::CommandQueue::Graphics);

            const bool resolved2 = fix.waitWithWatchdog([&]() {
                return dev->pollTimerQuery(tq);
            }, 2000u);
            UAA_CHECK(resolved2, "Reused timer query must resolve within watchdog timeout");

            const float elapsed2 = dev->getTimerQueryTime(tq);
            UAA_CHECK(
                std::isfinite(elapsed2) && elapsed2 >= 0.0f,
                "Reused timer query elapsed time must be finite and non-negative");
            verifyComputeCopyAddOutput(fix, ctx, job2);
        }

        class ScopedGateSignal
        {
        public:
            explicit ScopedGateSignal(IGpuGate* gate) : m_Gate(gate) { }
            ~ScopedGateSignal() { if (m_Gate) m_Gate->signal(); }

        private:
            IGpuGate* m_Gate;
        };

        void testMetalTimerSubmission(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (fix.api() != nvrhi::GraphicsAPI::METAL)
                ctx.skip("Requires the Metal backend");

            nvrhi::IDevice* dev = fix.device();
            ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, 65536u, 7u, 0x55AAu, ctx.seed() ^ 0x308ull);
            nvrhi::TimerQueryHandle query = dev->createTimerQuery();
            UAA_CHECK(query != nullptr, "createTimerQuery returned null");
            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            std::unique_ptr<IGpuGate> gate = fix.backend()->createGpuGate();
            UAA_CHECK(gate != nullptr, "Metal GPU gate is unavailable");

            for (uint32_t iteration = 0; iteration < 2; ++iteration)
            {
                if (iteration > 0)
                    gate->reset();
                cmd->open();
                cmd->beginTimerQuery(query);
                for (uint32_t i = 0; i < 4; ++i)
                    recordComputeCopyAdd(cmd, job);
                cmd->endTimerQuery(query);
                cmd->close();

                UAA_CHECK(!dev->pollTimerQuery(query), "Timer query became ready before submission");
                UAA_CHECK_EQ(dev->getTimerQueryTime(query), 0.0f, "Unsubmitted timer query must return zero");

                UAA_CHECK(gate->enqueueWait(nvrhi::CommandQueue::Graphics), "Failed to gate the GPU queue");
                ScopedGateSignal releaseGate(gate.get());
                UAA_CHECK(dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics) != 0, "Submission failed");
                UAA_CHECK(!dev->pollTimerQuery(query), "Timer query became ready while the GPU was gated");
                gate->signal();

                UAA_CHECK(fix.waitWithWatchdog([&]() { return dev->pollTimerQuery(query); }, 2000u),
                    "Submitted timer query did not resolve");
                const float elapsed = dev->getTimerQueryTime(query);
                UAA_CHECK(std::isfinite(elapsed) && elapsed > 0.0f, "Timer query must return a positive GPU interval");
                dev->resetTimerQuery(query);
            }
            verifyComputeCopyAddOutput(fix, ctx, job);
        }

        void testMetalTimerNested(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (fix.api() != nvrhi::GraphicsAPI::METAL)
                ctx.skip("Requires the Metal backend");

            nvrhi::IDevice* dev = fix.device();
            ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, 65536u, 7u, 0x55AAu, ctx.seed() ^ 0x309ull);
            nvrhi::BufferDesc copyDesc;
            copyDesc.byteSize = uint64_t(job.count) * sizeof(uint32_t);
            copyDesc.initialState = nvrhi::ResourceStates::CopyDest;
            copyDesc.keepInitialState = true;
            nvrhi::BufferHandle copy = dev->createBuffer(copyDesc);
            nvrhi::TimerQueryHandle outer = dev->createTimerQuery();
            nvrhi::TimerQueryHandle inner = dev->createTimerQuery();
            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(copy && outer && inner && cmd, "Failed to create timer test resources");

            cmd->open();
            cmd->beginTimerQuery(outer);
            cmd->copyBuffer(copy, 0, job.src, 0, copyDesc.byteSize);
            cmd->beginTimerQuery(inner);
            for (uint32_t i = 0; i < 4; ++i)
                recordComputeCopyAdd(cmd, job);
            cmd->endTimerQuery(inner);
            cmd->copyBuffer(copy, 0, job.src, 0, copyDesc.byteSize);
            cmd->endTimerQuery(outer);
            cmd->close();
            fix.submitAndWait(cmd);

            UAA_CHECK(dev->pollTimerQuery(outer) && dev->pollTimerQuery(inner), "Nested timer queries did not resolve");
            const float outerTime = dev->getTimerQueryTime(outer);
            const float innerTime = dev->getTimerQueryTime(inner);
            UAA_CHECK(std::isfinite(innerTime) && innerTime > 0.0f, "Inner timer did not measure a GPU interval");
            UAA_CHECK(std::isfinite(outerTime) && outerTime >= innerTime, "Outer timer did not include the inner interval");
            verifyComputeCopyAddOutput(fix, ctx, job);
            std::vector<uint32_t> copied(job.count);
            fix.readbackBuffer(copy, copied.data(), copyDesc.byteSize);
            UAA_CHECK(copied == job.inputWords, "Timed buffer copy produced incorrect data");
        }

        void testMetalTimerLifetime(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (fix.api() != nvrhi::GraphicsAPI::METAL)
                ctx.skip("Requires the Metal backend");

            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
            ExpectErrorsGuard errors(fix);

            std::vector<nvrhi::TimerQueryHandle> queries;
            for (uint32_t i = 0; i < 4096; ++i)
            {
                nvrhi::TimerQueryHandle query = dev->createTimerQuery();
                if (!query)
                    break;
                queries.push_back(query);
            }
            const size_t capacity = queries.size();
            UAA_CHECK(capacity > 1 && capacity < 4096, "Could not determine timer query pool capacity");
            queries.clear();

            nvrhi::TimerQueryHandle query = dev->createTimerQuery();
            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(query != nullptr && cmd != nullptr, "Failed to create timer query or command list");
            cmd->open();
            cmd->beginTimerQuery(query);
            cmd->endTimerQuery(query);
            cmd->close();

            std::unique_ptr<IGpuGate> gate = fix.backend()->createGpuGate();
            UAA_CHECK(gate != nullptr && gate->enqueueWait(nvrhi::CommandQueue::Graphics), "Failed to gate the GPU queue");
            ScopedGateSignal releaseGate(gate.get());
            UAA_CHECK(dev->executeCommandList(cmd, nvrhi::CommandQueue::Graphics) != 0, "Submission failed");
            query = nullptr;
            cmd = nullptr;

            for (size_t i = 0; i < capacity - 1; ++i)
            {
                nvrhi::TimerQueryHandle available = dev->createTimerQuery();
                UAA_CHECK(available != nullptr, "Timer query pool exhausted too early");
                queries.push_back(available);
            }
            const bool heldUntilCompletion = dev->createTimerQuery() == nullptr;
            gate->signal();
            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
            UAA_CHECK(heldUntilCompletion, "An in-flight timer query slot was recycled");
            UAA_CHECK(dev->createTimerQuery() != nullptr, "Completed timer query slot was not recycled");
        }

        void testUnsupportedOrUnresolvedTimerQuery(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            nvrhi::TimerQueryHandle tq = dev->createTimerQuery();
            UAA_CHECK(tq != nullptr, "createTimerQuery returned null");

            UAA_CHECK(!dev->pollTimerQuery(tq), "Newly created timer query must not report resolved");
            UAA_CHECK_EQ(dev->getTimerQueryTime(tq), 0.0f, "Unstarted timer query must return 0.0f");

            {
                // Run a valid timed pass, resolve, then reset and confirm unresolved state returns cleanly.
                ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, 1024u, 5u, 0x11u, ctx.seed() ^ 0x505ull);
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");

                cmd->open();
                cmd->beginTimerQuery(tq);
                recordComputeCopyAdd(cmd, job);
                cmd->endTimerQuery(tq);
                cmd->close();
                fix.submitAndWait(cmd, nvrhi::CommandQueue::Graphics);

                UAA_CHECK(
                    fix.waitWithWatchdog([&]() { return dev->pollTimerQuery(tq); }, 2000u),
                    "Timer query should resolve after GPU work");
                const float resolvedTime = dev->getTimerQueryTime(tq);
                UAA_CHECK(std::isfinite(resolvedTime) && resolvedTime >= 0.0f, "Resolved timer time must be valid");

                dev->resetTimerQuery(tq);
                UAA_CHECK(!dev->pollTimerQuery(tq), "Reset timer query must return false when polled");
                UAA_CHECK_EQ(dev->getTimerQueryTime(tq), 0.0f, "Reset timer query must return 0.0f without hanging");
            }
        }

        void testNestedMarkers(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();
            UAA_CHECK(dev != nullptr, "Active device must not be null");

            const size_t baseErrors = fix.messages().errorCount();

            // Copy resources
            constexpr size_t kCopyBytes = 512;
            const std::vector<uint8_t> copyInput = TestFixture::makeDeterministicBytes(kCopyBytes, ctx.seed() ^ 0x606ull);
            nvrhi::BufferDesc srcDesc;
            srcDesc.byteSize = kCopyBytes;
            srcDesc.initialState = nvrhi::ResourceStates::CopySource;
            srcDesc.keepInitialState = true;
            nvrhi::BufferHandle copySrc = dev->createBuffer(srcDesc);
            UAA_CHECK(copySrc != nullptr, "Failed to create copy source buffer");
            fix.uploadBuffer(copySrc, copyInput.data(), kCopyBytes, 0);

            nvrhi::BufferDesc dstDesc;
            dstDesc.byteSize = kCopyBytes;
            dstDesc.initialState = nvrhi::ResourceStates::CopyDest;
            dstDesc.keepInitialState = true;
            nvrhi::BufferHandle copyDst = dev->createBuffer(dstDesc);
            UAA_CHECK(copyDst != nullptr, "Failed to create copy destination buffer");

            // Compute job
            ComputeCopyAddJob job = buildComputeCopyAddJob(fix, ctx, 512u, 11u, 0x3C3Cu, ctx.seed() ^ 0x707ull);

            // Graphics resources
            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex, "main");
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel, "main");
            UAA_CHECK(vs != nullptr && ps != nullptr, "Failed to create graphics shaders");

            nvrhi::BindingLayoutDesc gfxLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics);
            gfxLayoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SolidColorPC)));
            nvrhi::BindingLayoutHandle gfxLayout = dev->createBindingLayout(gfxLayoutDesc);
            UAA_CHECK(gfxLayout != nullptr, "Failed to create graphics binding layout");
            nvrhi::BindingSetHandle gfxSet = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(SolidColorPC))),
                gfxLayout);
            UAA_CHECK(gfxSet != nullptr, "Failed to create graphics binding set");

            nvrhi::TextureDesc rtDesc;
            rtDesc.width = 32;
            rtDesc.height = 32;
            rtDesc.format = nvrhi::Format::RGBA8_UNORM;
            rtDesc.isRenderTarget = true;
            rtDesc.initialState = nvrhi::ResourceStates::RenderTarget;
            rtDesc.keepInitialState = true;
            rtDesc.debugName = "qry_marker_rt";
            nvrhi::TextureHandle rt = dev->createTexture(rtDesc);
            UAA_CHECK(rt != nullptr, "Failed to create render target texture");

            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(rt);
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(fbDesc);
            UAA_CHECK(fb != nullptr, "Failed to create framebuffer");

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.setVertexShader(vs)
                .setPixelShader(ps)
                .addBindingLayout(gfxLayout);
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.renderState.depthStencilState.depthWriteEnable = false;
            psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;

            nvrhi::GraphicsPipelineHandle gfxPso = dev->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
            UAA_CHECK(gfxPso != nullptr, "Failed to create graphics pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            cmd->open();
            cmd->beginMarker("outer");

            cmd->beginMarker("inner_copy");
            cmd->copyBuffer(copyDst, 0, copySrc, 0, kCopyBytes);
            cmd->endMarker();

            cmd->beginMarker("inner_compute");
            recordComputeCopyAdd(cmd, job);
            cmd->endMarker();

            cmd->beginMarker("inner_graphics");
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            nvrhi::GraphicsState gfxState;
            gfxState.setPipeline(gfxPso)
                .setFramebuffer(fb)
                .addBindingSet(gfxSet)
                .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(32.f, 32.f)));
            cmd->setGraphicsState(gfxState);
            SolidColorPC pc{{0.25f, 0.5f, 0.75f, 1.f}, 0.f, 1.f, {0.f, 0.f}};
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->endMarker();

            cmd->endMarker();
            cmd->close();

            fix.submitAndWait(cmd, nvrhi::CommandQueue::Graphics);

            std::vector<uint8_t> copyActual(kCopyBytes, 0);
            fix.readbackBuffer(copyDst, copyActual.data(), kCopyBytes, 0);
            std::string diff;
            UAA_CHECK(
                TestFixture::checkBytesEqual(copyActual.data(), copyInput.data(), kCopyBytes, diff),
                "Marked copy output mismatch: " + diff);

            verifyComputeCopyAddOutput(fix, ctx, job);

            std::vector<uint8_t> rtPixels;
            fix.readbackTexture2D(rt, 0, 0, rtPixels);
            UAA_CHECK(rtPixels.size() >= 4, "Readback pixels unexpectedly empty");
            UAA_CHECK(
                TestFixture::checkPixelRgba8(rtPixels.data(), 64, 128, 191, 255, 2, diff),
                "Marked graphics pixel mismatch: " + diff);

            UAA_CHECK_EQ(fix.messages().errorCount(), baseErrors, "Nested markers produced unexpected device errors");
        }

        void testInvalidQueryUsage(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* valDev = fix.validationDevice();
            nvrhi::DeviceHandle localValidation;
            if (!valDev)
            {
#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
                localValidation = nvrhi::validation::createValidationLayer(fix.rawDevice());
                UAA_CHECK(localValidation != nullptr, "Failed to create local validation layer");
                valDev = localValidation.Get();
#else
                ctx.skip("NVRHI validation layer is not compiled in this build");
#endif
            }

            nvrhi::TimerQueryHandle tq = valDev->createTimerQuery();
            UAA_CHECK(tq != nullptr, "createTimerQuery returned null");

            nvrhi::CommandListHandle cmd = valDev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            ExpectErrorsGuard guard(fix);

            // Recording query/marker commands on an unopened command list must trigger validation errors.
            cmd->beginTimerQuery(tq);
            cmd->endTimerQuery(tq);
            cmd->beginMarker("invalid_unopened");
            cmd->endMarker();

            // Open and close cleanly, then record after close without executing.
            cmd->open();
            cmd->close();
            cmd->beginTimerQuery(tq);
            cmd->endTimerQuery(tq);
            cmd->endMarker();

            // Executing a Graphics command list on a Compute queue must be rejected by validation.
            valDev->executeCommandList(cmd, nvrhi::CommandQueue::Compute);

            // Creating a command list for an unsupported queue must trigger a validation error.
            if (!fix.features().supports(nvrhi::Feature::ComputeQueue))
            {
                nvrhi::CommandListHandle badComputeCmd = valDev->createCommandList(
                    nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute));
                UAA_CHECK(badComputeCmd == nullptr, "Unsupported Compute queue should return null command list");
            }
            else if (!fix.features().supports(nvrhi::Feature::CopyQueue))
            {
                nvrhi::CommandListHandle badCopyCmd = valDev->createCommandList(
                    nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy));
                UAA_CHECK(badCopyCmd == nullptr, "Unsupported Copy queue should return null command list");
            }

            UAA_CHECK(guard.errorCount() > 0, "Expected validation errors for invalid query/command-list usage");

            cmd = nullptr;
            tq = nullptr;
            localValidation = nullptr;
        }
    } // namespace

    UAA_REGISTER_TEST("QRY.Event", "core", testEventQueryLifecycle);
    UAA_REGISTER_TEST("QRY.EventQueues", "core", testEventQueryAcrossQueues);
    UAA_REGISTER_TEST("QRY.Timer", "core", testTimerQuery);
    UAA_REGISTER_TEST("QRY.MetalTimerSubmission", "core", testMetalTimerSubmission);
    UAA_REGISTER_TEST("QRY.MetalTimerNested", "core", testMetalTimerNested);
    UAA_REGISTER_TEST("QRY.MetalTimerLifetime", "core", testMetalTimerLifetime);
    UAA_REGISTER_TEST("QRY.Unsupported", "core", testUnsupportedOrUnresolvedTimerQuery);
    UAA_REGISTER_TEST("QRY.Markers", "core", testNestedMarkers);
    UAA_REGISTER_TEST("QRY.Invalid", "core", testInvalidQueryUsage);
} // namespace nvrhi::uaa
