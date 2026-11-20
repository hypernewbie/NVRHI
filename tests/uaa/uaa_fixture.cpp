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

#include <nvrhi/validation.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace nvrhi::uaa
{
    namespace
    {
        const char* severityLabel(nvrhi::MessageSeverity severity)
        {
            switch (severity)
            {
            case nvrhi::MessageSeverity::Info:
                return "Info";
            case nvrhi::MessageSeverity::Warning:
                return "Warning";
            case nvrhi::MessageSeverity::Error:
                return "Error";
            case nvrhi::MessageSeverity::Fatal:
                return "Fatal";
            default:
                return "Unknown";
            }
        }

        std::string toLowerAscii(std::string value)
        {
            for (char& ch : value)
            {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return value;
        }

        uint64_t splitMix64(uint64_t& state)
        {
            uint64_t z = (state += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }
    } // namespace

    void MessageCapture::message(nvrhi::MessageSeverity severity, const char* messageText)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Messages.push_back(CapturedMessage{severity, messageText ? messageText : ""});

        const bool suppressed = m_DirectSuppress || (m_SuppressDepth > 0);
        if (!suppressed)
        {
            if (severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal)
            {
                ++m_ErrorCount;
            }
            else if (severity == nvrhi::MessageSeverity::Warning)
            {
                ++m_WarningCount;
            }
        }
    }

    void MessageCapture::clear()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Messages.clear();
        m_ErrorCount = 0;
        m_WarningCount = 0;
    }

    size_t MessageCapture::errorCount() const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_ErrorCount;
    }

    size_t MessageCapture::warningCount() const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_WarningCount;
    }

    std::vector<CapturedMessage> MessageCapture::snapshot() const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_Messages;
    }

    bool MessageCapture::hasMessageContaining(const std::string& substr) const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        for (const auto& entry : m_Messages)
        {
            if (entry.text.find(substr) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    std::string MessageCapture::formatAll() const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        std::ostringstream oss;
        for (size_t i = 0; i < m_Messages.size(); ++i)
        {
            if (i > 0)
            {
                oss << "\n";
            }
            oss << "[" << severityLabel(m_Messages[i].severity) << "] " << m_Messages[i].text;
        }
        return oss.str();
    }

    void MessageCapture::setSuppressErrors(bool suppress)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_DirectSuppress = suppress;
    }

    void MessageCapture::pushSuppressErrors()
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        ++m_SuppressDepth;
    }

    void MessageCapture::popSuppressErrors(size_t trimFromIndex)
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (m_SuppressDepth > 0)
        {
            --m_SuppressDepth;
        }
        if (trimFromIndex <= m_Messages.size())
        {
            m_Messages.erase(m_Messages.begin() + static_cast<ptrdiff_t>(trimFromIndex), m_Messages.end());
        }
    }

    size_t MessageCapture::rawMessageCount() const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        return m_Messages.size();
    }

    std::vector<CapturedMessage> MessageCapture::sliceFrom(size_t startIndex) const
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        if (startIndex >= m_Messages.size())
        {
            return {};
        }
        return std::vector<CapturedMessage>(
            m_Messages.begin() + static_cast<ptrdiff_t>(startIndex),
            m_Messages.end());
    }

    ExpectErrorsGuard::ExpectErrorsGuard(MessageCapture& capture)
        : m_Capture(capture)
        , m_StartIndex(capture.rawMessageCount())
    {
        m_Capture.pushSuppressErrors();
    }

    ExpectErrorsGuard::ExpectErrorsGuard(TestFixture& fixture)
        : ExpectErrorsGuard(fixture.messages())
    {
    }

    ExpectErrorsGuard::~ExpectErrorsGuard()
    {
        m_Capture.popSuppressErrors(m_StartIndex);
    }

    size_t ExpectErrorsGuard::errorCount() const
    {
        size_t count = 0;
        for (const auto& entry : m_Capture.sliceFrom(m_StartIndex))
        {
            if (entry.severity == nvrhi::MessageSeverity::Error || entry.severity == nvrhi::MessageSeverity::Fatal)
            {
                ++count;
            }
        }
        return count;
    }

    bool ExpectErrorsGuard::hasMessageContaining(const std::string& substr) const
    {
        for (const auto& entry : m_Capture.sliceFrom(m_StartIndex))
        {
            if (entry.text.find(substr) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<CapturedMessage> ExpectErrorsGuard::captured() const
    {
        return m_Capture.sliceFrom(m_StartIndex);
    }

    std::vector<nvrhi::Feature> FeatureInventory::allFeatures()
    {
        return {
            nvrhi::Feature::ComputeQueue,
            nvrhi::Feature::ConservativeRasterization,
            nvrhi::Feature::ConstantBufferRanges,
            nvrhi::Feature::CopyQueue,
            nvrhi::Feature::DeferredCommandLists,
            nvrhi::Feature::FastGeometryShader,
            nvrhi::Feature::HeapDirectlyIndexed,
            nvrhi::Feature::HlslExtensionUAV,
            nvrhi::Feature::LinearSweptSpheres,
            nvrhi::Feature::Meshlets,
            nvrhi::Feature::RayQuery,
            nvrhi::Feature::RayTracingAccelStruct,
            nvrhi::Feature::RayTracingClusters,
            nvrhi::Feature::RayTracingOpacityMicromap,
            nvrhi::Feature::RayTracingPipeline,
            nvrhi::Feature::RayTracingPositionFetch,
            nvrhi::Feature::SamplerFeedback,
            nvrhi::Feature::ShaderExecutionReordering,
            nvrhi::Feature::ShaderSpecializations,
            nvrhi::Feature::SinglePassStereo,
            nvrhi::Feature::Spheres,
            nvrhi::Feature::VariableRateShading,
            nvrhi::Feature::VirtualResources,
            nvrhi::Feature::WaveLaneCountMinMax,
            nvrhi::Feature::CooperativeVectorInferencing,
            nvrhi::Feature::CooperativeVectorTraining,
            nvrhi::Feature::EnhancedBarriers,
        };
    }

    const char* FeatureInventory::featureName(nvrhi::Feature f)
    {
        switch (f)
        {
        case nvrhi::Feature::ComputeQueue: return "ComputeQueue";
        case nvrhi::Feature::ConservativeRasterization: return "ConservativeRasterization";
        case nvrhi::Feature::ConstantBufferRanges: return "ConstantBufferRanges";
        case nvrhi::Feature::CopyQueue: return "CopyQueue";
        case nvrhi::Feature::DeferredCommandLists: return "DeferredCommandLists";
        case nvrhi::Feature::FastGeometryShader: return "FastGeometryShader";
        case nvrhi::Feature::HeapDirectlyIndexed: return "HeapDirectlyIndexed";
        case nvrhi::Feature::HlslExtensionUAV: return "HlslExtensionUAV";
        case nvrhi::Feature::LinearSweptSpheres: return "LinearSweptSpheres";
        case nvrhi::Feature::Meshlets: return "Meshlets";
        case nvrhi::Feature::RayQuery: return "RayQuery";
        case nvrhi::Feature::RayTracingAccelStruct: return "RayTracingAccelStruct";
        case nvrhi::Feature::RayTracingClusters: return "RayTracingClusters";
        case nvrhi::Feature::RayTracingOpacityMicromap: return "RayTracingOpacityMicromap";
        case nvrhi::Feature::RayTracingPipeline: return "RayTracingPipeline";
        case nvrhi::Feature::RayTracingPositionFetch: return "RayTracingPositionFetch";
        case nvrhi::Feature::SamplerFeedback: return "SamplerFeedback";
        case nvrhi::Feature::ShaderExecutionReordering: return "ShaderExecutionReordering";
        case nvrhi::Feature::ShaderSpecializations: return "ShaderSpecializations";
        case nvrhi::Feature::SinglePassStereo: return "SinglePassStereo";
        case nvrhi::Feature::Spheres: return "Spheres";
        case nvrhi::Feature::VariableRateShading: return "VariableRateShading";
        case nvrhi::Feature::VirtualResources: return "VirtualResources";
        case nvrhi::Feature::WaveLaneCountMinMax: return "WaveLaneCountMinMax";
        case nvrhi::Feature::CooperativeVectorInferencing: return "CooperativeVectorInferencing";
        case nvrhi::Feature::CooperativeVectorTraining: return "CooperativeVectorTraining";
        case nvrhi::Feature::EnhancedBarriers: return "EnhancedBarriers";
        default: return "UnknownFeature";
        }
    }

    void FeatureInventory::populate(nvrhi::IDevice* device)
    {
        entries.clear();
        vrsInfoQueried = false;
        vrsInfo = nvrhi::VariableRateShadingFeatureInfo{};
        waveInfoQueried = false;
        waveInfo = nvrhi::WaveLaneCountMinMaxFeatureInfo{};
        clusterInfoQueried = false;
        clusterInfo = nvrhi::RayTracingClustersFeatureInfo{};

        if (!device)
        {
            return;
        }

        for (nvrhi::Feature f : allFeatures())
        {
            entries.push_back(FeatureInfoEntry{
                f,
                featureName(f),
                device->queryFeatureSupport(f)
            });
        }

        vrsInfo.shadingRateImageTileSize = 0xDEADBEEFu;
        vrsInfoQueried = device->queryFeatureSupport(
            nvrhi::Feature::VariableRateShading, &vrsInfo, sizeof(vrsInfo));

        waveInfo.minWaveLaneCount = 0xDEADBEEFu;
        waveInfo.maxWaveLaneCount = 0xDEADBEEFu;
        waveInfoQueried = device->queryFeatureSupport(
            nvrhi::Feature::WaveLaneCountMinMax, &waveInfo, sizeof(waveInfo));

        clusterInfo.api = nvrhi::ClusterApi::None;
        clusterInfoQueried = device->queryFeatureSupport(
            nvrhi::Feature::RayTracingClusters, &clusterInfo, sizeof(clusterInfo));
    }

    bool FeatureInventory::supports(nvrhi::Feature f) const
    {
        for (const auto& entry : entries)
        {
            if (entry.feature == f)
            {
                return entry.supported;
            }
        }
        return false;
    }

    std::string FeatureInventory::formatSummary() const
    {
        std::ostringstream oss;
        bool first = true;
        for (const auto& entry : entries)
        {
            if (entry.supported)
            {
                if (!first)
                {
                    oss << ", ";
                }
                oss << entry.name;
                first = false;
            }
        }
        if (first)
        {
            oss << "(none)";
        }
        return oss.str();
    }

    std::unique_ptr<IBackendContext> createBackendContext(
        const std::string& backendName,
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        const std::string name = toLowerAscii(backendName);
        outSkipOrError.clear();
        outIsUnavailable = false;

        if (name == "vulkan" || name == "vk")
        {
#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
            return createVulkanBackend(config, messageCallback, outSkipOrError, outIsUnavailable);
#else
            (void)config;
            (void)messageCallback;
            outIsUnavailable = true;
            outSkipOrError = "Vulkan backend is not compiled in this build";
            return nullptr;
#endif
        }

        if (name == "metal" || name == "mtl")
        {
#if defined(UAA_WITH_METAL) && UAA_WITH_METAL
            return createMetalBackend(config, messageCallback, outSkipOrError, outIsUnavailable);
#else
            (void)config;
            (void)messageCallback;
            outIsUnavailable = true;
            outSkipOrError = "Metal backend is not compiled in this build";
            return nullptr;
#endif
        }

        if (name == "d3d11" || name == "dx11")
        {
#if (defined(UAA_WITH_D3D11) && UAA_WITH_D3D11) || (defined(UAA_WITH_DX11) && UAA_WITH_DX11)
            return createD3D11Backend(config, messageCallback, outSkipOrError, outIsUnavailable);
#else
            (void)config;
            (void)messageCallback;
            outIsUnavailable = true;
            outSkipOrError = "D3D11 backend is not compiled in this build";
            return nullptr;
#endif
        }

        if (name == "d3d12" || name == "dx12")
        {
#if (defined(UAA_WITH_D3D12) && UAA_WITH_D3D12) || (defined(UAA_WITH_DX12) && UAA_WITH_DX12)
            return createD3D12Backend(config, messageCallback, outSkipOrError, outIsUnavailable);
#else
            (void)config;
            (void)messageCallback;
            outIsUnavailable = true;
            outSkipOrError = "D3D12 backend is not compiled in this build";
            return nullptr;
#endif
        }

        outIsUnavailable = false;
        outSkipOrError = "Unknown GPU backend: " + backendName;
        return nullptr;
    }

    TestFixture::~TestFixture()
    {
        shutdown();
    }

    bool TestFixture::init(const RunnerOptions& opts, bool enableNvrhiValidation, std::string& outMsg, bool& outUnavailable)
    {
        shutdown();

        m_Options = opts;
#ifdef UAA_SHADER_DIR
        m_ShaderDir = opts.shaderDir.empty() ? std::string(UAA_SHADER_DIR) : opts.shaderDir;
#else
        m_ShaderDir = opts.shaderDir.empty() ? std::string("shaders") : opts.shaderDir;
#endif
        m_Messages.clear();

        BackendConfig cfg;
        cfg.nativeValidation = opts.nativeValidation;
        cfg.deviceSelector = opts.deviceSelector;

        m_Backend = createBackendContext(opts.backend, cfg, &m_Messages, outMsg, outUnavailable);
        if (!m_Backend)
        {
            return false;
        }

        nvrhi::IDevice* raw = m_Backend->getRawDevice();
        if (!raw)
        {
            outMsg = "Backend created context with null raw device";
            outUnavailable = false;
            m_Backend.reset();
            return false;
        }

        m_DeviceInfo = m_Backend->getDeviceInfo();
        if (opts.nativeValidation && !m_DeviceInfo.nativeValidation)
        {
            outMsg = "Requested native validation is unavailable on " + opts.backend;
            outUnavailable = true;
            m_Backend.reset();
            return false;
        }

        if (enableNvrhiValidation)
        {
#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
            m_ValidationDevice = nvrhi::validation::createValidationLayer(raw);
            if (!m_ValidationDevice)
            {
                outMsg = "Failed to create NVRHI validation layer";
                outUnavailable = false;
                m_Backend.reset();
                return false;
            }
            m_ActiveDevice = m_ValidationDevice.Get();
#else
            outMsg = "NVRHI validation layer is not enabled in this build";
            outUnavailable = true;
            m_Backend.reset();
            return false;
#endif
        }
        else
        {
            m_ActiveDevice = raw;
        }

        m_DeviceInfo.nvrhiValidation = (m_ValidationDevice != nullptr);
        m_Features.populate(m_ActiveDevice);
        return true;
    }

    void TestFixture::shutdown()
    {
        if (m_ActiveDevice)
        {
            m_ActiveDevice->waitForIdle();
            m_ActiveDevice->runGarbageCollection();
        }
        m_ActiveDevice = nullptr;
        m_ValidationDevice = nullptr;
        m_Backend.reset();
        m_Features = FeatureInventory{};
        m_DeviceInfo = DeviceInfo{};
    }

    nvrhi::GraphicsAPI TestFixture::api() const
    {
        if (!m_ActiveDevice)
        {
            throw TestFailureException("TestFixture::api() called before initialisation");
        }
        return m_ActiveDevice->getGraphicsAPI();
    }

    nvrhi::BindingLayoutDesc TestFixture::makeBindingLayoutDesc(
        nvrhi::ShaderType visibility,
        uint32_t registerSpace,
        bool registerSpaceIsDescriptorSet) const
    {
        nvrhi::BindingLayoutDesc desc;
        desc.setVisibility(visibility)
            .setRegisterSpace(registerSpace)
            .setRegisterSpaceIsDescriptorSet(registerSpaceIsDescriptorSet)
            .setBindingOffsets(nvrhi::VulkanBindingOffsets{0, 0, 0, 0});
        return desc;
    }

    std::vector<uint8_t> TestFixture::loadShaderBinary(const std::string& baseName, bool frozenMetal) const
    {
        std::string path = m_ShaderDir;
        if (!path.empty() && path.back() != '/' && path.back() != '\\')
        {
            path.push_back('/');
        }

        if (frozenMetal)
        {
            path += "frozen_metal/" + baseName + ".spv";
        }
        else
        {
            switch (api())
            {
            case nvrhi::GraphicsAPI::VULKAN:
            case nvrhi::GraphicsAPI::METAL:
                path += "bin/spirv/" + baseName + ".spv";
                break;
            case nvrhi::GraphicsAPI::D3D12:
                path += "bin/dxil/" + baseName + ".dxil";
                break;
            case nvrhi::GraphicsAPI::D3D11:
                path += "bin/dxbc/" + baseName + ".dxbc";
                break;
            default:
                throw TestFailureException("Unsupported GraphicsAPI in loadShaderBinary");
            }
        }

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            throw TestFailureException("Failed to open shader binary: " + path);
        }

        const std::streamoff size = file.tellg();
        if (size <= 0)
        {
            throw TestFailureException("Shader binary is empty: " + path);
        }

        std::vector<uint8_t> data(static_cast<size_t>(size));
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(data.data()), size))
        {
            throw TestFailureException("Failed to read shader binary: " + path);
        }
        return data;
    }

    nvrhi::ShaderHandle TestFixture::tryCreateShader(
        const std::string& baseName,
        nvrhi::ShaderType type,
        const std::string& entryName,
        bool frozenMetal) const
    {
        if (!m_ActiveDevice)
        {
            throw TestFailureException("TestFixture::tryCreateShader() called with null device");
        }

        const std::vector<uint8_t> bytes = loadShaderBinary(baseName, frozenMetal);
        nvrhi::ShaderDesc desc;
        desc.setShaderType(type)
            .setEntryName(entryName)
            .setDebugName(baseName);
        return m_ActiveDevice->createShader(desc, bytes.data(), bytes.size());
    }

    nvrhi::ShaderHandle TestFixture::createShader(
        const std::string& baseName,
        nvrhi::ShaderType type,
        const std::string& entryName,
        bool frozenMetal) const
    {
        nvrhi::ShaderHandle shader = tryCreateShader(baseName, type, entryName, frozenMetal);
        if (!shader)
        {
            throw TestFailureException("createShader returned null for '" + baseName + "'");
        }
        return shader;
    }

    void TestFixture::uploadBuffer(nvrhi::IBuffer* dst, const void* data, size_t size, uint64_t offset) const
    {
        if (!dst)
        {
            throw TestFailureException("uploadBuffer called with null buffer");
        }
        if (size == 0)
        {
            return;
        }
        if (!data)
        {
            throw TestFailureException("uploadBuffer called with null source pointer");
        }

        const nvrhi::BufferDesc& desc = dst->getDesc();
        if (desc.cpuAccess == nvrhi::CpuAccessMode::Write && !desc.isVolatile)
        {
            void* mapped = m_ActiveDevice->mapBuffer(dst, nvrhi::CpuAccessMode::Write);
            if (!mapped)
            {
                throw TestFailureException("uploadBuffer: mapBuffer returned null");
            }
            std::memcpy(static_cast<uint8_t*>(mapped) + offset, data, size);
            m_ActiveDevice->unmapBuffer(dst);
            return;
        }

        nvrhi::CommandListHandle cmd = m_ActiveDevice->createCommandList();
        if (!cmd)
        {
            throw TestFailureException("uploadBuffer: createCommandList returned null");
        }
        cmd->open();
        cmd->writeBuffer(dst, data, size, offset);
        cmd->close();
        submitAndWait(cmd);
    }

    void TestFixture::readbackBuffer(nvrhi::IBuffer* src, void* outData, size_t size, uint64_t offset) const
    {
        if (!src)
        {
            throw TestFailureException("readbackBuffer called with null buffer");
        }
        if (size == 0)
        {
            return;
        }
        if (!outData)
        {
            throw TestFailureException("readbackBuffer called with null output pointer");
        }

        nvrhi::BufferDesc rbDesc;
        rbDesc.byteSize = size;
        rbDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        rbDesc.debugName = "uaa_readback_buffer";
        nvrhi::BufferHandle rb = m_ActiveDevice->createBuffer(rbDesc);
        if (!rb)
        {
            throw TestFailureException("readbackBuffer: failed to create readback buffer");
        }

        nvrhi::CommandListHandle cmd = m_ActiveDevice->createCommandList();
        if (!cmd)
        {
            throw TestFailureException("readbackBuffer: createCommandList returned null");
        }
        cmd->open();
        cmd->copyBuffer(rb, 0, src, offset, size);
        cmd->close();
        submitAndWait(cmd);

        void* mapped = m_ActiveDevice->mapBuffer(rb, nvrhi::CpuAccessMode::Read);
        if (!mapped)
        {
            throw TestFailureException("readbackBuffer: mapBuffer returned null");
        }
        std::memcpy(outData, mapped, size);
        m_ActiveDevice->unmapBuffer(rb);
    }

    void TestFixture::uploadTexture2D(
        nvrhi::ITexture* dst,
        uint32_t slice,
        uint32_t mip,
        const void* data,
        size_t rowPitch) const
    {
        if (!dst)
        {
            throw TestFailureException("uploadTexture2D called with null texture");
        }
        if (!data)
        {
            throw TestFailureException("uploadTexture2D called with null source data");
        }

        const nvrhi::TextureDesc& dstDesc = dst->getDesc();
        const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(dstDesc.format);
        if (rowPitch == 0)
        {
            const uint32_t mipW = std::max(1u, dstDesc.width >> mip);
            const uint32_t blockSize = std::max<uint32_t>(1u, fmtInfo.blockSize);
            const uint32_t blocksW = (mipW + blockSize - 1) / blockSize;
            rowPitch = size_t(blocksW) * fmtInfo.bytesPerBlock;
        }

        nvrhi::CommandListHandle cmd = m_ActiveDevice->createCommandList();
        if (!cmd)
        {
            throw TestFailureException("uploadTexture2D: createCommandList returned null");
        }
        cmd->open();
        cmd->writeTexture(dst, slice, mip, data, rowPitch);
        cmd->close();
        submitAndWait(cmd);
    }

    void TestFixture::readbackTexture2D(
        nvrhi::ITexture* src,
        uint32_t slice,
        uint32_t mip,
        std::vector<uint8_t>& outTightPixels) const
    {
        if (!src)
        {
            throw TestFailureException("readbackTexture2D called with null texture");
        }

        const nvrhi::TextureDesc& srcDesc = src->getDesc();
        if (mip >= srcDesc.mipLevels)
        {
            throw TestFailureException("readbackTexture2D: mip level out of range");
        }

        const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(srcDesc.format);
        if (fmtInfo.bytesPerBlock == 0)
        {
            throw TestFailureException("readbackTexture2D: unsupported format");
        }

        const uint32_t mipW = std::max(1u, srcDesc.width >> mip);
        const uint32_t mipH = std::max(1u, srcDesc.height >> mip);
        const uint32_t blockSize = std::max<uint32_t>(1u, fmtInfo.blockSize);
        const uint32_t blocksW = (mipW + blockSize - 1) / blockSize;
        const uint32_t blocksH = (mipH + blockSize - 1) / blockSize;
        const size_t tightRowBytes = size_t(blocksW) * fmtInfo.bytesPerBlock;

        // Match mipLevels with the source texture so backends that index mip offsets by mipLevel work reliably.
        nvrhi::TextureDesc stagingDesc;
        stagingDesc.width = srcDesc.width;
        stagingDesc.height = srcDesc.height;
        stagingDesc.depth = 1;
        stagingDesc.arraySize = 1;
        stagingDesc.mipLevels = srcDesc.mipLevels;
        stagingDesc.format = srcDesc.format;
        stagingDesc.dimension = nvrhi::TextureDimension::Texture2D;
        stagingDesc.debugName = "uaa_staging_readback";

        nvrhi::StagingTextureHandle staging = m_ActiveDevice->createStagingTexture(stagingDesc, nvrhi::CpuAccessMode::Read);
        if (!staging)
        {
            throw TestFailureException("readbackTexture2D: createStagingTexture returned null");
        }

        nvrhi::TextureSlice srcSlice;
        srcSlice.setMipLevel(mip).setArraySlice(slice);

        nvrhi::TextureSlice dstSlice;
        dstSlice.setMipLevel(mip).setArraySlice(0);

        nvrhi::CommandListHandle cmd = m_ActiveDevice->createCommandList();
        if (!cmd)
        {
            throw TestFailureException("readbackTexture2D: createCommandList returned null");
        }
        cmd->open();
        cmd->copyTexture(staging, dstSlice, src, srcSlice);
        cmd->close();
        submitAndWait(cmd);

        size_t rowPitch = 0;
        void* mapped = m_ActiveDevice->mapStagingTexture(staging, dstSlice, nvrhi::CpuAccessMode::Read, &rowPitch);
        if (!mapped)
        {
            throw TestFailureException("readbackTexture2D: mapStagingTexture returned null");
        }
        if (rowPitch < tightRowBytes)
        {
            m_ActiveDevice->unmapStagingTexture(staging);
            throw TestFailureException("readbackTexture2D: staging rowPitch smaller than tight row bytes");
        }

        outTightPixels.resize(tightRowBytes * blocksH);
        const uint8_t* srcBytes = static_cast<const uint8_t*>(mapped);
        for (uint32_t r = 0; r < blocksH; ++r)
        {
            std::memcpy(outTightPixels.data() + size_t(r) * tightRowBytes, srcBytes + size_t(r) * rowPitch, tightRowBytes);
        }

        m_ActiveDevice->unmapStagingTexture(staging);
    }

    uint64_t TestFixture::submitAndWait(nvrhi::ICommandList* cmd, nvrhi::CommandQueue queue) const
    {
        if (!cmd)
        {
            throw TestFailureException("submitAndWait called with null command list");
        }
        const uint64_t instance = m_ActiveDevice->executeCommandList(cmd, queue);
        if (!m_ActiveDevice->waitForIdle())
        {
            throw TestFailureException("Device waitForIdle() returned false");
        }
        m_ActiveDevice->runGarbageCollection();
        return instance;
    }

    bool TestFixture::waitWithWatchdog(std::function<bool()> pollFn, uint32_t timeoutMs) const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (!pollFn())
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    std::vector<uint8_t> TestFixture::makeDeterministicBytes(size_t size, uint64_t seed)
    {
        std::vector<uint8_t> bytes(size);
        uint64_t state = seed ^ 0xA5A5A5A55A5A5A5Aull;
        for (size_t i = 0; i < size; ++i)
        {
            if ((i & 7u) == 0u)
            {
                splitMix64(state);
            }
            bytes[i] = static_cast<uint8_t>((state >> ((i & 7u) * 8u)) & 0xFFu);
        }
        return bytes;
    }

    std::vector<uint32_t> TestFixture::makeDeterministicWords(size_t count, uint64_t seed)
    {
        std::vector<uint32_t> words(count);
        uint64_t state = seed ^ 0x123456789ABCDEF0ull;
        for (size_t i = 0; i < count; ++i)
        {
            words[i] = static_cast<uint32_t>(splitMix64(state) & 0xFFFFFFFFu);
        }
        return words;
    }

    bool TestFixture::checkBytesEqual(const void* actual, const void* expected, size_t size, std::string& outDiff)
    {
        const auto* a = static_cast<const uint8_t*>(actual);
        const auto* e = static_cast<const uint8_t*>(expected);
        for (size_t i = 0; i < size; ++i)
        {
            if (a[i] != e[i])
            {
                std::ostringstream oss;
                oss << "Byte mismatch at offset " << i << ": actual=0x" << std::hex << static_cast<unsigned>(a[i])
                    << " expected=0x" << static_cast<unsigned>(e[i]);
                outDiff = oss.str();
                return false;
            }
        }
        return true;
    }

    bool TestFixture::checkGuardBytes(const void* data, size_t size, uint8_t guardByte, std::string& outDiff)
    {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            if (p[i] != guardByte)
            {
                std::ostringstream oss;
                oss << "Guard byte corruption at offset " << i << ": actual=0x" << std::hex << static_cast<unsigned>(p[i])
                    << " expected=0x" << static_cast<unsigned>(guardByte);
                outDiff = oss.str();
                return false;
            }
        }
        return true;
    }

    bool TestFixture::checkPixelRgba8(
        const uint8_t* actualRgba,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        uint8_t a,
        uint8_t tolerance,
        std::string& outDiff)
    {
        const uint8_t expected[4] = {r, g, b, a};
        for (int c = 0; c < 4; ++c)
        {
            const int diff = std::abs(static_cast<int>(actualRgba[c]) - static_cast<int>(expected[c]));
            if (diff > static_cast<int>(tolerance))
            {
                std::ostringstream oss;
                oss << "Pixel mismatch (tolerance " << static_cast<unsigned>(tolerance) << "): actual=("
                    << static_cast<unsigned>(actualRgba[0]) << ","
                    << static_cast<unsigned>(actualRgba[1]) << ","
                    << static_cast<unsigned>(actualRgba[2]) << ","
                    << static_cast<unsigned>(actualRgba[3]) << ") expected=("
                    << static_cast<unsigned>(r) << ","
                    << static_cast<unsigned>(g) << ","
                    << static_cast<unsigned>(b) << ","
                    << static_cast<unsigned>(a) << ")";
                outDiff = oss.str();
                return false;
            }
        }
        return true;
    }

    bool TestFixture::checkFloatNear(float actual, float expected, float eps, std::string& outDiff)
    {
        const float diff = std::fabs(actual - expected);
        if (!(diff <= eps) || std::isnan(actual) || std::isnan(expected))
        {
            std::ostringstream oss;
            oss << "Float mismatch: actual=" << actual << " expected=" << expected
                << " diff=" << diff << " eps=" << eps;
            outDiff = oss.str();
            return false;
        }
        return true;
    }
} // namespace nvrhi::uaa
