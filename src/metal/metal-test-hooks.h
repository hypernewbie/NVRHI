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

#pragma once

#include <nvrhi/metal.h>

namespace nvrhi::metal::test
{
    // Only available when NVRHI_METAL_TEST_HOOKS is enabled for a test build.
    struct SubmissionHooks
    {
        bool failNextQueueWaitAllocation = false;
        bool failNextCompletion = false;
        void (*beforeCommit)(void* context, uint64_t serial) = nullptr;
        void (*queueWaitCommitted)(void* context, uint64_t serial) = nullptr;
        void* context = nullptr;
    };

    struct ArgumentBufferStats
    {
        uint64_t encodersCreated = 0;
        uint64_t buffersEncoded = 0;
        uint64_t buffersReused = 0;
        uint64_t bufferLength = 0;
        uint64_t bufferDeclarations = 0;
    };

    NVRHI_API void setSubmissionHooks(IDevice* device, const SubmissionHooks& hooks);
    NVRHI_API Object getSubmissionEvent(IDevice* device);
    NVRHI_API ArgumentBufferStats getArgumentBufferStats(IDevice* device);
    NVRHI_API void resetArgumentBufferStats(IDevice* device);
}
