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

#include "uaa_fixture.h"
#include "uaa_runner.h"
#include "../../src/metal/metal-test-hooks.h"
#if UAA_WITH_VALIDATION
#include <nvrhi/validation.h>
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace nvrhi::uaa
{
    namespace
    {
        struct StandaloneDevice
        {
            metal::DeviceHandle raw;
            DeviceHandle active;

            explicit StandaloneDevice(TestFixture& fix)
            {
                metal::DeviceDesc desc;
                desc.errorCB = &fix.messages();
                raw = metal::createDevice(desc);
                active = raw;
#if UAA_WITH_VALIDATION
                if (raw && fix.isValidationActive())
                    active = validation::createValidationLayer(raw);
#endif
            }
        };

        class NativeGate
        {
        public:
            explicit NativeGate(metal::IDevice* device)
            {
                id<MTLDevice> native = (__bridge id<MTLDevice>)device->getNativeDevice().pointer;
                id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)device->getNativeCommandQueue().pointer;
                m_Event = [native newSharedEvent];
                id<MTLCommandBuffer> wait = [queue commandBuffer];
                if (!m_Event || !wait)
                    return;
                [wait encodeWaitForEvent:m_Event value:1];
                [wait commit];
                m_Valid = true;
                m_Watchdog = std::thread([this] {
                    std::unique_lock<std::mutex> lock(m_Mutex);
                    if (!m_Condition.wait_for(lock, std::chrono::seconds(5), [this] { return m_Released; }))
                    {
                        m_TimedOut = true;
                        m_Released = true;
                        m_Event.signaledValue = 1;
                    }
                });
            }

            ~NativeGate()
            {
                release();
                if (m_Watchdog.joinable())
                    m_Watchdog.join();
            }

            bool valid() const { return m_Valid; }

            void release()
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                if (!m_Released)
                {
                    m_Released = true;
                    if (m_Event)
                        m_Event.signaledValue = 1;
                    m_Condition.notify_all();
                }
            }

            bool timedOut()
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                return m_TimedOut;
            }

        private:
            id<MTLSharedEvent> m_Event = nil;
            bool m_Valid = false;
            bool m_Released = false;
            bool m_TimedOut = false;
            std::mutex m_Mutex;
            std::condition_variable m_Condition;
            std::thread m_Watchdog;
        };

        BufferHandle makeOutput(IDevice* device)
        {
            return device->createBuffer(BufferDesc().setByteSize(16).setCpuAccess(CpuAccessMode::Read)
                .setInitialState(ResourceStates::CopyDest).setKeepInitialState(true));
        }

        CommandListHandle makeWrite(IDevice* device, IBuffer* buffer, uint32_t value)
        {
            CommandListHandle command = device->createCommandList();
            if (command)
            {
                command->open();
                command->writeBuffer(buffer, &value, sizeof(value));
                command->close();
            }
            return command;
        }

        struct CommitBarrier
        {
            std::mutex mutex;
            std::condition_variable condition;
            bool entered = false;
            bool release = false;
            bool timedOut = false;
            uint64_t serial = 0;

            static void pause(void* context, uint64_t serial)
            {
                CommitBarrier& barrier = *static_cast<CommitBarrier*>(context);
                std::unique_lock<std::mutex> lock(barrier.mutex);
                barrier.serial = serial;
                barrier.entered = true;
                barrier.condition.notify_all();
                if (!barrier.condition.wait_for(lock, std::chrono::seconds(3), [&] { return barrier.release; }))
                    barrier.timedOut = true;
            }
        };

        static void testMetalHookSerialOrder(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            BufferHandle output = makeOutput(device.active);
            CommandListHandle first = makeWrite(device.active, output, 0x11111111u);
            CommandListHandle second = makeWrite(device.active, output, 0x22222222u);
            UAA_CHECK(output && first && second, "submission resources are missing");

            CommitBarrier barrier;
            metal::test::SubmissionHooks hooks;
            hooks.beforeCommit = &CommitBarrier::pause;
            hooks.context = &barrier;
            metal::test::setSubmissionHooks(device.raw, hooks);

            uint64_t firstSerial = 0;
            uint64_t secondSerial = 0;
            std::atomic<bool> attempted{false};
            std::atomic<bool> returned{false};
            std::thread firstWorker([&] { firstSerial = device.active->executeCommandList(first); });
            bool entered = false;
            {
                std::unique_lock<std::mutex> lock(barrier.mutex);
                entered = barrier.condition.wait_for(lock, std::chrono::seconds(2), [&] { return barrier.entered; });
            }
            std::thread secondWorker([&] {
                attempted.store(true);
                secondSerial = device.active->executeCommandList(second);
                returned.store(true);
            });
            while (!attempted.load())
                std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const bool returnedWhilePaused = returned.load();
            {
                std::lock_guard<std::mutex> lock(barrier.mutex);
                barrier.release = true;
                barrier.condition.notify_all();
            }
            firstWorker.join();
            secondWorker.join();
            UAA_CHECK(entered && !barrier.timedOut, "the first submission did not reach the commit barrier");
            UAA_CHECK(!returnedWhilePaused, "the second submission passed the first before commit");
            UAA_CHECK_EQ(firstSerial, barrier.serial, "the first returned serial changed");
            UAA_CHECK(firstSerial != 0 && secondSerial > firstSerial, "submission serials are out of order");
            UAA_CHECK(device.active->waitForIdle(), "waitForIdle failed");
            const uint32_t* mapped = static_cast<const uint32_t*>(device.active->mapBuffer(output, CpuAccessMode::Read));
            UAA_CHECK(mapped != nullptr, "output map failed");
            const uint32_t actual = *mapped;
            device.active->unmapBuffer(output);
            UAA_CHECK_EQ(actual, 0x22222222u, "GPU execution order did not match the submission serials");
        }

        static void testMetalHookWaitAllocation(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
            ExpectErrorsGuard guard(*ctx.fixture);
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            BufferHandle output = makeOutput(device.active);
            CommandListHandle producer = makeWrite(device.active, output, 0x11111111u);
            CommandListHandle failedConsumer = makeWrite(device.active, output, 0x22222222u);
            CommandListHandle retry = makeWrite(device.active, output, 0x33333333u);
            UAA_CHECK(output && producer && failedConsumer && retry, "submission resources are missing");
            id<MTLCommandBuffer> failedNative = (__bridge id<MTLCommandBuffer>)
                failedConsumer->getNativeObject(ObjectTypes::Metal_CommandBuffer).pointer;
            UAA_CHECK(failedNative != nil, "consumer command buffer is missing");

            NativeGate gate(device.raw);
            UAA_CHECK(gate.valid(), "GPU gate creation failed");
            const uint64_t producerSerial = device.active->executeCommandList(producer);
            UAA_CHECK(producerSerial != 0, "producer submission failed");
            device.active->queueWaitForCommandList(CommandQueue::Graphics, CommandQueue::Graphics, producerSerial);
            std::atomic<uint64_t> committedWait{0};
            metal::test::SubmissionHooks hooks;
            hooks.failNextQueueWaitAllocation = true;
            hooks.context = &committedWait;
            hooks.queueWaitCommitted = [](void* context, uint64_t serial) {
                static_cast<std::atomic<uint64_t>*>(context)->store(serial);
            };
            metal::test::setSubmissionHooks(device.raw, hooks);
            const uint64_t failedSerial = device.active->executeCommandList(failedConsumer);
            UAA_CHECK_EQ(failedSerial, uint64_t(0), "the consumer ran after wait allocation failed");
            UAA_CHECK_EQ(failedNative.status, MTLCommandBufferStatusNotEnqueued, "the failed consumer was committed");
            UAA_CHECK_EQ(committedWait.load(), uint64_t(0), "a queue wait was committed despite allocation failure");
            UAA_CHECK_EQ(guard.errorCount(), size_t(1), "the wait allocation failure was not reported once");
            UAA_CHECK(guard.hasMessageContaining("no command buffer is available for the pending queue wait"),
                "the wait allocation diagnostic was missing");
            const uint64_t retrySerial = device.active->executeCommandList(retry);
            UAA_CHECK(retrySerial > producerSerial, "the retry did not submit");
            UAA_CHECK_EQ(committedWait.load(), producerSerial, "the pending queue wait was not restored");
            EventQueryHandle query = device.active->createEventQuery();
            device.active->setEventQuery(query, CommandQueue::Graphics);
            UAA_CHECK(!device.active->pollEventQuery(query), "the retry completed before the GPU gate was released");
            gate.release();
            device.active->waitEventQuery(query);
            UAA_CHECK(!gate.timedOut(), "the GPU gate timed out");
            UAA_CHECK(device.active->waitForIdle(), "waitForIdle failed after the retry");
            const uint32_t* mapped = static_cast<const uint32_t*>(device.active->mapBuffer(output, CpuAccessMode::Read));
            UAA_CHECK(mapped != nullptr, "output map failed");
            const uint32_t actual = *mapped;
            device.active->unmapBuffer(output);
            UAA_CHECK_EQ(actual, 0x33333333u, "the retried consumer did not execute");
        }

        static void testMetalHookTimerLifetime(TestContext& ctx)
        {
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            TimerQueryHandle query = device.active->createTimerQuery();
            CommandListHandle command = device.active->createCommandList();
            UAA_CHECK(query && command, "timer resources are missing");
            command->open();
            command->beginTimerQuery(query);
            command->endTimerQuery(query);
            command->close();
            query = nullptr;

            NativeGate gate(device.raw);
            UAA_CHECK(gate.valid(), "GPU gate creation failed");
            UAA_CHECK(device.active->executeCommandList(command) != 0, "submission failed");
            command = nullptr;
            gate.release();
            UAA_CHECK(device.active->waitForIdle(), "waitForIdle failed");
            UAA_CHECK(!gate.timedOut(), "the GPU gate timed out");

            query = device.active->createTimerQuery();
            UAA_CHECK(query != nullptr, "timer query creation after completion failed");
            device.active = nullptr;
            device.raw = nullptr;
            query = nullptr;
        }

        static void testMetalHookTimerConcurrent(TestContext& ctx)
        {
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            constexpr uint32_t threadCount = 8;
            constexpr uint32_t queriesPerThread = 4;
            std::vector<TimerQueryHandle> queries(threadCount * queriesPerThread);
            std::vector<std::thread> threads;
            std::atomic<uint32_t> ready{0};
            std::atomic<bool> start{false};
            for (uint32_t thread = 0; thread < threadCount; ++thread)
            {
                threads.emplace_back([&, thread] {
                    @autoreleasepool
                    {
                        ++ready;
                        while (!start.load())
                            std::this_thread::yield();
                        for (uint32_t i = 0; i < queriesPerThread; ++i)
                            queries[thread * queriesPerThread + i] = device.active->createTimerQuery();
                    }
                });
            }
            while (ready.load() != threadCount)
                std::this_thread::yield();
            start = true;
            for (auto& thread : threads)
                thread.join();
            for (const auto& query : queries)
                UAA_CHECK(query != nullptr, "concurrent timer query creation failed");

            CommandListHandle command = device.active->createCommandList();
            BufferHandle buffer = device.active->createBuffer(BufferDesc().setByteSize(1024 * 1024)
                .setInitialState(ResourceStates::CopyDest).setKeepInitialState(true));
            UAA_CHECK(command && buffer, "timer test resources are missing");
            command->open();
            for (const auto& query : queries)
                command->beginTimerQuery(query);
            command->clearBufferUInt(buffer, 0x12345678u);
            for (auto it = queries.rbegin(); it != queries.rend(); ++it)
                command->endTimerQuery(*it);
            command->close();
            UAA_CHECK(device.active->executeCommandList(command) != 0, "submission failed");
            UAA_CHECK(device.active->waitForIdle(), "waitForIdle failed");
            for (const auto& query : queries)
                UAA_CHECK(device.active->getTimerQueryTime(query) > 0.f, "concurrently allocated timer did not resolve");
        }

        static void testMetalHookTimerFailure(TestContext& ctx)
        {
            ExpectErrorsGuard guard(*ctx.fixture);
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            TimerQueryHandle query = device.active->createTimerQuery();
            CommandListHandle command = device.active->createCommandList();
            BufferHandle buffer = device.active->createBuffer(BufferDesc().setByteSize(1024 * 1024)
                .setInitialState(ResourceStates::CopyDest).setKeepInitialState(true));
            UAA_CHECK(query && command && buffer, "timer resources are missing");

            command->open();
            command->beginTimerQuery(query);
            command->clearBufferUInt(buffer, 0x12345678u);
            command->endTimerQuery(query);
            command->close();
            UAA_CHECK(device.active->executeCommandList(command) != 0, "initial submission failed");
            UAA_CHECK(device.active->getTimerQueryTime(query) > 0.f, "initial timer interval was not measured");
            device.active->resetTimerQuery(query);

            command->open();
            command->beginTimerQuery(query);
            command->endTimerQuery(query);
            command->close();
            metal::test::SubmissionHooks hooks;
            hooks.failNextCompletion = true;
            metal::test::setSubmissionHooks(device.raw, hooks);
            UAA_CHECK(device.active->executeCommandList(command) != 0, "failed submission was not committed");
            UAA_CHECK_EQ(device.active->getTimerQueryTime(query), 0.f, "failed submission returned a timer value");
            UAA_CHECK(!device.active->waitForIdle(), "waitForIdle reported success after a completion error");
            command = nullptr;
            query = nullptr;
            device.active = nullptr;
            device.raw = nullptr;
            UAA_CHECK_EQ(guard.errorCount(), size_t(1), "the completion error was not reported once");
        }

        static void testMetalHookCompletionError(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
            ExpectErrorsGuard guard(*ctx.fixture);
            StandaloneDevice device(*ctx.fixture);
            UAA_CHECK(device.raw && device.active, "device creation failed");
            BufferHandle output = makeOutput(device.active);
            CommandListHandle command = makeWrite(device.active, output, 0x12345678u);
            CommandListHandle consumer = makeWrite(device.active, output, 0x87654321u);
            EventQueryHandle query = device.active->createEventQuery();
            UAA_CHECK(output && command && consumer && query, "submission resources are missing");
            id<MTLSharedEvent> submissionEvent = (__bridge id<MTLSharedEvent>)
                metal::test::getSubmissionEvent(device.raw).pointer;
            UAA_CHECK(submissionEvent != nil, "submission event is missing");
            metal::test::SubmissionHooks hooks;
            hooks.failNextCompletion = true;
            metal::test::setSubmissionHooks(device.raw, hooks);
            NativeGate gate(device.raw);
            UAA_CHECK(gate.valid(), "GPU gate creation failed");
            const uint64_t serial = device.active->executeCommandList(command);
            UAA_CHECK(serial != 0, "submission failed");
            device.active->queueWaitForCommandList(CommandQueue::Graphics, CommandQueue::Graphics, serial);
            const uint64_t consumerSerial = device.active->executeCommandList(consumer);
            UAA_CHECK(consumerSerial > serial, "consumer submission failed");
            device.active->setEventQuery(query, CommandQueue::Graphics);
            UAA_CHECK(!device.active->pollEventQuery(query), "the event completed before the GPU gate was released");
            std::atomic<bool> waited{false};
            std::thread waiter([&] {
                device.active->waitEventQuery(query);
                waited.store(true);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const bool returnedWhileBlocked = waited.load();
            gate.release();
            for (uint32_t i = 0; i < 200 && !waited.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const bool completedWithoutRescue = waited.load();
            if (!completedWithoutRescue)
                submissionEvent.signaledValue = serial;
            waiter.join();
            UAA_CHECK(!returnedWhileBlocked && !gate.timedOut(), "the event waiter did not follow GPU completion");
            UAA_CHECK(completedWithoutRescue, "the failed submission stranded a dependent GPU queue wait");
            UAA_CHECK(device.active->pollEventQuery(query), "the failed submission did not wake event waiters");
            UAA_CHECK(!device.active->waitForIdle(), "waitForIdle reported success after a completion error");
            command = nullptr;
            consumer = nullptr;
            output = nullptr;
            query = nullptr;
            device.active = nullptr;
            device.raw = nullptr;
            UAA_CHECK_EQ(guard.errorCount(), size_t(1), "the completion error was not reported once");
            UAA_CHECK(guard.hasMessageContaining("injected completion failure"), "the completion diagnostic was missing");
        }
    }

    UAA_REGISTER_TEST("SUB.MetalHookSerialOrder", "core,metal,isolated", testMetalHookSerialOrder);
    UAA_REGISTER_TEST("SUB.MetalHookWaitAllocation", "core,metal,isolated", testMetalHookWaitAllocation);
    UAA_REGISTER_TEST("SUB.MetalHookCompletionError", "core,metal,isolated", testMetalHookCompletionError);
    UAA_REGISTER_TEST("SUB.MetalHookTimerLifetime", "core,metal,isolated", testMetalHookTimerLifetime);
    UAA_REGISTER_TEST("SUB.MetalHookTimerConcurrent", "core,metal,isolated", testMetalHookTimerConcurrent);
    UAA_REGISTER_TEST("SUB.MetalHookTimerFailure", "core,metal,isolated", testMetalHookTimerFailure);
}
