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

#include "uaa_backend.h"
#include "uaa_runner.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    struct CapturedMessage
    {
        nvrhi::MessageSeverity severity = nvrhi::MessageSeverity::Info;
        std::string text;
    };

    class MessageCapture : public nvrhi::IMessageCallback
    {
    public:
        void message(nvrhi::MessageSeverity severity, const char* messageText) override;

        void clear();
        size_t errorCount() const;
        size_t warningCount() const;
        std::vector<CapturedMessage> snapshot() const;
        bool hasMessageContaining(const std::string& substr) const;
        std::string formatAll() const;

        void setSuppressErrors(bool suppress);
        void pushSuppressErrors();
        void popSuppressErrors(size_t trimFromIndex);

        size_t rawMessageCount() const;
        std::vector<CapturedMessage> sliceFrom(size_t startIndex) const;

    private:
        mutable std::mutex m_Mutex;
        std::vector<CapturedMessage> m_Messages;
        size_t m_ErrorCount = 0;
        size_t m_WarningCount = 0;
        size_t m_SuppressDepth = 0;
        bool m_DirectSuppress = false;
    };

    class ExpectErrorsGuard
    {
    public:
        explicit ExpectErrorsGuard(MessageCapture& capture);
        explicit ExpectErrorsGuard(TestFixture& fixture);
        ~ExpectErrorsGuard();

        ExpectErrorsGuard(const ExpectErrorsGuard&) = delete;
        ExpectErrorsGuard& operator=(const ExpectErrorsGuard&) = delete;

        size_t errorCount() const;
        bool hasMessageContaining(const std::string& substr) const;
        std::vector<CapturedMessage> captured() const;

    private:
        MessageCapture& m_Capture;
        size_t m_StartIndex = 0;
    };

    struct FeatureInfoEntry
    {
        nvrhi::Feature feature = nvrhi::Feature::ComputeQueue;
        const char* name = "";
        bool supported = false;
    };

    struct FeatureInventory
    {
        std::vector<FeatureInfoEntry> entries;
        bool vrsInfoQueried = false;
        nvrhi::VariableRateShadingFeatureInfo vrsInfo{};
        bool waveInfoQueried = false;
        nvrhi::WaveLaneCountMinMaxFeatureInfo waveInfo{};
        bool clusterInfoQueried = false;
        nvrhi::RayTracingClustersFeatureInfo clusterInfo{};

        void populate(nvrhi::IDevice* device);
        bool supports(nvrhi::Feature f) const;
        static const char* featureName(nvrhi::Feature f);
        static std::vector<nvrhi::Feature> allFeatures();
        std::string formatSummary() const;
    };

    class TestFixture
    {
    public:
        static constexpr uint8_t kGuardByte = 0xCD;
        static constexpr uint32_t kGuardWord = 0xDEADBEEFu;

        TestFixture() = default;
        ~TestFixture();

        TestFixture(const TestFixture&) = delete;
        TestFixture& operator=(const TestFixture&) = delete;

        bool init(const RunnerOptions& opts, bool enableNvrhiValidation, std::string& outMsg, bool& outUnavailable);
        void shutdown();

        nvrhi::IDevice* device() const { return m_ActiveDevice; }
        nvrhi::IDevice* rawDevice() const { return m_Backend ? m_Backend->getRawDevice() : nullptr; }
        nvrhi::IDevice* validationDevice() const { return m_ValidationDevice.Get(); }
        bool isValidationActive() const { return m_ValidationDevice != nullptr; }
        IBackendContext* backend() const { return m_Backend.get(); }
        const DeviceInfo& deviceInfo() const { return m_DeviceInfo; }
        MessageCapture& messages() { return m_Messages; }
        const MessageCapture& messages() const { return m_Messages; }
        const FeatureInventory& features() const { return m_Features; }
        nvrhi::GraphicsAPI api() const;
        const std::string& shaderDir() const { return m_ShaderDir; }
        const RunnerOptions& options() const { return m_Options; }

        nvrhi::BindingLayoutDesc makeBindingLayoutDesc(
            nvrhi::ShaderType visibility,
            uint32_t registerSpace = 0,
            bool registerSpaceIsDescriptorSet = false) const;

        std::vector<uint8_t> loadShaderBinary(const std::string& baseName, bool frozenMetal = false) const;
        nvrhi::ShaderHandle createShader(
            const std::string& baseName,
            nvrhi::ShaderType type,
            const std::string& entryName = "main",
            bool frozenMetal = false) const;
        nvrhi::ShaderHandle tryCreateShader(
            const std::string& baseName,
            nvrhi::ShaderType type,
            const std::string& entryName = "main",
            bool frozenMetal = false) const;

        void uploadBuffer(nvrhi::IBuffer* dst, const void* data, size_t size, uint64_t offset = 0) const;
        void readbackBuffer(nvrhi::IBuffer* src, void* outData, size_t size, uint64_t offset = 0) const;
        void uploadTexture2D(nvrhi::ITexture* dst, uint32_t slice, uint32_t mip, const void* data, size_t rowPitch = 0) const;
        void readbackTexture2D(nvrhi::ITexture* src, uint32_t slice, uint32_t mip, std::vector<uint8_t>& outTightPixels) const;

        uint64_t submitAndWait(nvrhi::ICommandList* cmd, nvrhi::CommandQueue queue = nvrhi::CommandQueue::Graphics) const;
        bool waitWithWatchdog(std::function<bool()> pollFn, uint32_t timeoutMs) const;

        static std::vector<uint8_t> makeDeterministicBytes(size_t size, uint64_t seed);
        static std::vector<uint32_t> makeDeterministicWords(size_t count, uint64_t seed);
        static bool checkBytesEqual(const void* actual, const void* expected, size_t size, std::string& outDiff);
        static bool checkGuardBytes(const void* data, size_t size, uint8_t guardByte, std::string& outDiff);
        static bool checkPixelRgba8(
            const uint8_t* actualRgba,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            uint8_t tolerance,
            std::string& outDiff);
        static bool checkFloatNear(float actual, float expected, float eps, std::string& outDiff);

    private:
        RunnerOptions m_Options;
        std::string m_ShaderDir;
        MessageCapture m_Messages;
        std::unique_ptr<IBackendContext> m_Backend;
        nvrhi::DeviceHandle m_ValidationDevice;
        nvrhi::IDevice* m_ActiveDevice = nullptr;
        DeviceInfo m_DeviceInfo;
        FeatureInventory m_Features;
    };
} // namespace nvrhi::uaa
