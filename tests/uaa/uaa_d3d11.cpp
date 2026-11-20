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

#if (defined(UAA_WITH_D3D11) && UAA_WITH_D3D11) || (defined(UAA_WITH_DX11) && UAA_WITH_DX11)

#include "uaa_backend.h"
#include "uaa_fixture.h"
#include "uaa_runner.h"

#ifdef _WIN32

#include <nvrhi/d3d11.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef GetMessage
#undef GetMessage
#endif
#include <dxgi1_4.h>
#include <d3d11sdklayers.h>
#include <wrl/client.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
        using PFN_D3D11CreateDevice = HRESULT(WINAPI*)(
            IDXGIAdapter*,
            D3D_DRIVER_TYPE,
            HMODULE,
            UINT,
            const D3D_FEATURE_LEVEL*,
            UINT,
            UINT,
            ID3D11Device**,
            D3D_FEATURE_LEVEL*,
            ID3D11DeviceContext**);

        std::string toLowerAscii(std::string value)
        {
            for (char& ch : value)
            {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return value;
        }

        bool containsCaseInsensitive(const std::string& haystack, const std::string& needle)
        {
            if (needle.empty())
            {
                return true;
            }
            return toLowerAscii(haystack).find(toLowerAscii(needle)) != std::string::npos;
        }

        bool tryParseIndex(const std::string& str, size_t& outIndex)
        {
            if (str.empty())
            {
                return false;
            }
            for (char ch : str)
            {
                if (!std::isdigit(static_cast<unsigned char>(ch)))
                {
                    return false;
                }
            }
            char* endPtr = nullptr;
            unsigned long val = std::strtoul(str.c_str(), &endPtr, 10);
            if (endPtr != str.c_str() + str.size())
            {
                return false;
            }
            outIndex = static_cast<size_t>(val);
            return true;
        }

        std::string wideToUtf8(const wchar_t* wstr)
        {
            if (!wstr || !wstr[0])
            {
                return {};
            }
            const int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
            if (len <= 1)
            {
                return {};
            }
            std::string out(static_cast<size_t>(len - 1), '\0');
            WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
            return out;
        }

        class D3D11BackendContext final : public IBackendContext
        {
        public:
            D3D11BackendContext() = default;

            ~D3D11BackendContext() override
            {
                if (m_NvrhiDevice)
                {
                    m_NvrhiDevice->waitForIdle();
                    m_NvrhiDevice->runGarbageCollection();
                    m_NvrhiDevice = nullptr;
                }
                m_InfoQueue.Reset();
                if (m_D3D11Context)
                {
                    m_D3D11Context->ClearState();
                    m_D3D11Context->Flush();
                    m_D3D11Context.Reset();
                }
                m_D3D11Device.Reset();
                m_Adapter.Reset();
                if (m_D3D11Dll)
                {
                    FreeLibrary(m_D3D11Dll);
                    m_D3D11Dll = nullptr;
                }
                if (m_DxgiDll)
                {
                    FreeLibrary(m_DxgiDll);
                    m_DxgiDll = nullptr;
                }
            }

            nvrhi::IDevice* getRawDevice() override
            {
                return m_NvrhiDevice.Get();
            }

            const DeviceInfo& getDeviceInfo() const override
            {
                return m_DeviceInfo;
            }

            bool createNativeImportBuffer(uint64_t byteSize, const void* initialData, NativeBufferImport& out) override
            {
                if (!m_D3D11Device)
                {
                    return false;
                }

                D3D11_BUFFER_DESC desc{};
                desc.ByteWidth = static_cast<UINT>(byteSize > 0 ? ((byteSize + 3ull) & ~3ull) : 4ull);
                desc.Usage = D3D11_USAGE_DEFAULT;
                desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
                desc.CPUAccessFlags = 0;
                desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

                std::vector<uint8_t> padded;
                D3D11_SUBRESOURCE_DATA initData{};
                const D3D11_SUBRESOURCE_DATA* pInit = nullptr;
                if (initialData && byteSize > 0)
                {
                    padded.assign(desc.ByteWidth, 0);
                    std::memcpy(padded.data(), initialData, static_cast<size_t>(byteSize));
                    initData.pSysMem = padded.data();
                    pInit = &initData;
                }

                Microsoft::WRL::ComPtr<ID3D11Buffer> buf;
                if (FAILED(m_D3D11Device->CreateBuffer(&desc, pInit, &buf)) || !buf)
                {
                    return false;
                }

                auto holder = std::make_shared<Microsoft::WRL::ComPtr<ID3D11Buffer>>(buf);
                out.objectType = nvrhi::ObjectTypes::D3D11_Buffer;
                out.handle = nvrhi::Object(static_cast<void*>(buf.Get()));
                out.byteSize = byteSize;
                out.release = [holder]() mutable
                {
                    holder.reset();
                };
                return true;
            }

            bool createNativeImportTexture(uint32_t width, uint32_t height, nvrhi::Format format, NativeTextureImport& out) override
            {
                if (!m_D3D11Device)
                {
                    return false;
                }

                const DXGI_FORMAT dxgiFormat = nvrhi::d3d11::convertFormat(format);
                if (dxgiFormat == DXGI_FORMAT_UNKNOWN)
                {
                    return false;
                }

                const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(format);
                const bool isDepth = fmtInfo.hasDepth || fmtInfo.hasStencil;

                D3D11_TEXTURE2D_DESC td{};
                td.Width = width;
                td.Height = height;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = dxgiFormat;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = isDepth
                    ? (D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE)
                    : (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS);

                Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
                if (FAILED(m_D3D11Device->CreateTexture2D(&td, nullptr, &tex)) || !tex)
                {
                    return false;
                }

                nvrhi::TextureDesc desc;
                desc.width = width;
                desc.height = height;
                desc.depth = 1;
                desc.arraySize = 1;
                desc.mipLevels = 1;
                desc.sampleCount = 1;
                desc.format = format;
                desc.dimension = nvrhi::TextureDimension::Texture2D;
                desc.isRenderTarget = true;
                desc.isUAV = !isDepth;
                desc.initialState = nvrhi::ResourceStates::Common;
                desc.keepInitialState = true;
                desc.debugName = "uaa_native_import_d3d11_tex";

                auto holder = std::make_shared<Microsoft::WRL::ComPtr<ID3D11Texture2D>>(tex);
                out.objectType = nvrhi::ObjectTypes::D3D11_Resource;
                out.handle = nvrhi::Object(static_cast<void*>(static_cast<ID3D11Resource*>(tex.Get())));
                out.desc = desc;
                out.release = [holder]() mutable
                {
                    holder.reset();
                };
                return true;
            }

            std::unique_ptr<IBackendContext> createSecondaryDevice(nvrhi::IMessageCallback* cb) override
            {
                BackendConfig secCfg;
                secCfg.nativeValidation = false;
                secCfg.deviceSelector = m_SelectorUsed;
                std::string msg;
                bool unavail = false;
                auto sec = std::make_unique<D3D11BackendContext>();
                if (!sec->init(secCfg, cb, msg, unavail))
                {
                    return nullptr;
                }
                return sec;
            }

            bool hasNativeValidationErrors(std::vector<std::string>& outErrors) override
            {
                outErrors.clear();
                if (!m_InfoQueue)
                {
                    return false;
                }

                const UINT64 count = m_InfoQueue->GetNumStoredMessages();
                for (UINT64 i = 0; i < count; ++i)
                {
                    SIZE_T msgLen = 0;
                    if (FAILED(m_InfoQueue->GetMessage(i, nullptr, &msgLen)) || msgLen == 0)
                    {
                        continue;
                    }
                    std::vector<uint8_t> storage(msgLen);
                    auto* message = reinterpret_cast<D3D11_MESSAGE*>(storage.data());
                    if (FAILED(m_InfoQueue->GetMessage(i, message, &msgLen)))
                    {
                        continue;
                    }
                    if (message->Severity == D3D11_MESSAGE_SEVERITY_CORRUPTION ||
                        message->Severity == D3D11_MESSAGE_SEVERITY_ERROR)
                    {
                        outErrors.emplace_back(message->pDescription ? message->pDescription : "D3D11 validation error");
                    }
                }
                return !outErrors.empty();
            }

            void clearNativeValidationMessages() override
            {
                if (m_InfoQueue)
                {
                    m_InfoQueue->ClearStoredMessages();
                }
            }

            bool init(
                const BackendConfig& config,
                nvrhi::IMessageCallback* messageCallback,
                std::string& outSkipOrError,
                bool& outIsUnavailable)
            {
                outSkipOrError.clear();
                outIsUnavailable = false;
                m_SelectorUsed = config.deviceSelector;

                m_DxgiDll = LoadLibraryA("dxgi.dll");
                m_D3D11Dll = LoadLibraryA("d3d11.dll");
                if (!m_DxgiDll || !m_D3D11Dll)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "dxgi.dll or d3d11.dll not available";
                    return false;
                }

                auto createDxgiFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(
                    GetProcAddress(m_DxgiDll, "CreateDXGIFactory1"));
                auto d3d11CreateDevice = reinterpret_cast<PFN_D3D11CreateDevice>(
                    GetProcAddress(m_D3D11Dll, "D3D11CreateDevice"));
                if (!createDxgiFactory1 || !d3d11CreateDevice)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "Required D3D11/DXGI entry points unavailable";
                    return false;
                }

                Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
                if (FAILED(createDxgiFactory1(IID_PPV_ARGS(&factory))) || !factory)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "CreateDXGIFactory1 failed";
                    return false;
                }

                const bool forceWarp = containsCaseInsensitive(config.deviceSelector, "warp");
                bool useWarp = forceWarp;

                if (!forceWarp)
                {
                    struct AdapterCandidate
                    {
                        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
                        DXGI_ADAPTER_DESC1 desc{};
                        std::string name;
                    };
                    std::vector<AdapterCandidate> adapters;
                    for (UINT idx = 0;; ++idx)
                    {
                        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
                        if (factory->EnumAdapters1(idx, &adapter) == DXGI_ERROR_NOT_FOUND)
                        {
                            break;
                        }
                        DXGI_ADAPTER_DESC1 ad{};
                        adapter->GetDesc1(&ad);
                        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                        {
                            continue;
                        }
                        adapters.push_back(AdapterCandidate{ adapter, ad, wideToUtf8(ad.Description) });
                    }

                    if (!config.deviceSelector.empty())
                    {
                        size_t parsedIdx = 0;
                        bool matched = false;
                        if (tryParseIndex(config.deviceSelector, parsedIdx))
                        {
                            if (parsedIdx < adapters.size())
                            {
                                m_Adapter = adapters[parsedIdx].adapter;
                                matched = true;
                            }
                        }
                        else
                        {
                            for (const auto& cand : adapters)
                            {
                                if (containsCaseInsensitive(cand.name, config.deviceSelector))
                                {
                                    m_Adapter = cand.adapter;
                                    matched = true;
                                    break;
                                }
                            }
                        }
                        if (!matched)
                        {
                            outIsUnavailable = true;
                            outSkipOrError = "No D3D11 adapter matching selector '" + config.deviceSelector + "'";
                            return false;
                        }
                    }
                    else if (!adapters.empty())
                    {
                        m_Adapter = adapters[0].adapter;
                    }
                    else
                    {
                        useWarp = true;
                    }
                }

                const D3D_FEATURE_LEVEL featureLevels[] = {
                    D3D_FEATURE_LEVEL_11_1,
                    D3D_FEATURE_LEVEL_11_0
                };

                auto tryCreate = [&](IDXGIAdapter* adapter, D3D_DRIVER_TYPE drvType, UINT flags) -> HRESULT
                {
                    D3D_FEATURE_LEVEL achieved{};
                    return d3d11CreateDevice(
                        adapter,
                        drvType,
                        nullptr,
                        flags,
                        featureLevels,
                        static_cast<UINT>(sizeof(featureLevels) / sizeof(featureLevels[0])),
                        D3D11_SDK_VERSION,
                        &m_D3D11Device,
                        &achieved,
                        &m_D3D11Context);
                };

                UINT flags = config.nativeValidation ? D3D11_CREATE_DEVICE_DEBUG : 0u;
                HRESULT hr = tryCreate(
                    useWarp ? nullptr : m_Adapter.Get(),
                    useWarp ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_UNKNOWN,
                    flags);

                if (FAILED(hr) && config.nativeValidation)
                {
                    flags = 0;
                    hr = tryCreate(
                        useWarp ? nullptr : m_Adapter.Get(),
                        useWarp ? D3D_DRIVER_TYPE_WARP : D3D_DRIVER_TYPE_UNKNOWN,
                        flags);
                }

                if (FAILED(hr) && !useWarp && config.deviceSelector.empty())
                {
                    useWarp = true;
                    flags = config.nativeValidation ? D3D11_CREATE_DEVICE_DEBUG : 0u;
                    hr = tryCreate(nullptr, D3D_DRIVER_TYPE_WARP, flags);
                    if (FAILED(hr) && config.nativeValidation)
                    {
                        flags = 0;
                        hr = tryCreate(nullptr, D3D_DRIVER_TYPE_WARP, flags);
                    }
                }

                if (FAILED(hr) || !m_D3D11Device || !m_D3D11Context)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "D3D11CreateDevice failed";
                    return false;
                }

                const bool debugLayerEnabled = (flags & D3D11_CREATE_DEVICE_DEBUG) != 0;
                if (debugLayerEnabled)
                {
                    m_D3D11Device.As(&m_InfoQueue);
                }

                nvrhi::d3d11::DeviceDesc desc;
                desc.context = m_D3D11Context.Get();
                desc.messageCallback = messageCallback;
                m_NvrhiDevice = nvrhi::d3d11::createDevice(desc);
                if (!m_NvrhiDevice)
                {
                    outIsUnavailable = false;
                    outSkipOrError = "nvrhi::d3d11::createDevice returned null";
                    return false;
                }

                std::string adapterName = useWarp ? "Microsoft Basic Render Driver (WARP)" : "D3D11 Adapter";
                if (!useWarp && m_Adapter)
                {
                    DXGI_ADAPTER_DESC1 ad{};
                    if (SUCCEEDED(m_Adapter->GetDesc1(&ad)))
                    {
                        adapterName = wideToUtf8(ad.Description);
                    }
                }

                m_DeviceInfo.backendName = "d3d11";
                m_DeviceInfo.deviceName = adapterName;
                m_DeviceInfo.driverName = useWarp ? "WARP" : "D3D11";
                m_DeviceInfo.driverVersion = "11.1";
                m_DeviceInfo.osName = "Windows";
#ifdef UAA_BUILD_CONFIG
                m_DeviceInfo.buildConfig = UAA_BUILD_CONFIG;
#endif
                m_DeviceInfo.nativeValidation = debugLayerEnabled && (m_InfoQueue != nullptr);
                return true;
            }

        private:
            HMODULE m_DxgiDll = nullptr;
            HMODULE m_D3D11Dll = nullptr;
            std::string m_SelectorUsed;
            Microsoft::WRL::ComPtr<IDXGIAdapter1> m_Adapter;
            Microsoft::WRL::ComPtr<ID3D11Device> m_D3D11Device;
            Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_D3D11Context;
            Microsoft::WRL::ComPtr<ID3D11InfoQueue> m_InfoQueue;
            nvrhi::DeviceHandle m_NvrhiDevice;
            DeviceInfo m_DeviceInfo;
        };
    } // namespace

    std::unique_ptr<IBackendContext> createD3D11Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        auto ctx = std::make_unique<D3D11BackendContext>();
        if (!ctx->init(config, messageCallback, outSkipOrError, outIsUnavailable))
        {
            return nullptr;
        }
        return ctx;
    }

    static void testD3D11Native(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::D3D11)
        {
            UAA_SKIP("NAT.D3D11 requires the D3D11 backend");
        }

        nvrhi::IDevice* device = ctx.fixture->device();
        auto* nativeDev = static_cast<ID3D11Device*>(
            device->getNativeObject(nvrhi::ObjectTypes::D3D11_Device).pointer);
        auto* nativeCtx = static_cast<ID3D11DeviceContext*>(
            device->getNativeObject(nvrhi::ObjectTypes::D3D11_DeviceContext).pointer);
        auto* nvrhiD3D11 = static_cast<nvrhi::IDevice*>(
            device->getNativeObject(nvrhi::ObjectTypes::Nvrhi_D3D11_Device).pointer);

        UAA_CHECK(nativeDev != nullptr, "D3D11_Device native object returned null");
        UAA_CHECK(nativeCtx != nullptr, "D3D11_DeviceContext native object returned null");
        UAA_CHECK(nvrhiD3D11 != nullptr, "Nvrhi_D3D11_Device native object returned null");

        // Test imported ID3D11Buffer.
        {
            constexpr uint64_t kBufSize = 256;
            const std::vector<uint8_t> initBytes = TestFixture::makeDeterministicBytes(kBufSize, ctx.seed() ^ 0xD311u);
            NativeBufferImport bufImport;
            UAA_CHECK(
                ctx.fixture->backend()->createNativeImportBuffer(kBufSize, initBytes.data(), bufImport),
                "createNativeImportBuffer failed on D3D11");
            UAA_CHECK_EQ(bufImport.objectType, nvrhi::ObjectTypes::D3D11_Buffer, "D3D11 buffer import type mismatch");

            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = kBufSize;
            bufDesc.initialState = nvrhi::ResourceStates::CopySource;
            bufDesc.keepInitialState = true;
            bufDesc.debugName = "uaa_d3d11_imported_buf";
            nvrhi::BufferHandle imported = device->createHandleForNativeBuffer(
                bufImport.objectType, bufImport.handle, bufDesc);
            UAA_CHECK(imported != nullptr, "createHandleForNativeBuffer failed on D3D11");
            UAA_CHECK(
                imported->getNativeObject(nvrhi::ObjectTypes::D3D11_Buffer).pointer == bufImport.handle.pointer,
                "Imported D3D11 buffer native pointer mismatch");

            std::vector<uint8_t> actual(kBufSize, 0);
            ctx.fixture->readbackBuffer(imported, actual.data(), kBufSize);
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), initBytes.data(), kBufSize, diff), diff);

            imported = nullptr;
            device->runGarbageCollection();
            if (bufImport.release)
            {
                bufImport.release();
            }
        }

        // Test imported ID3D11Texture2D.
        {
            constexpr uint32_t kW = 16;
            constexpr uint32_t kH = 16;
            NativeTextureImport texImport;
            UAA_CHECK(
                ctx.fixture->backend()->createNativeImportTexture(kW, kH, nvrhi::Format::RGBA8_UNORM, texImport),
                "createNativeImportTexture failed on D3D11");
            UAA_CHECK_EQ(texImport.objectType, nvrhi::ObjectTypes::D3D11_Resource, "D3D11 texture import type mismatch");

            nvrhi::TextureHandle importedTex = device->createHandleForNativeTexture(
                texImport.objectType, texImport.handle, texImport.desc);
            UAA_CHECK(importedTex != nullptr, "createHandleForNativeTexture failed on D3D11");
            UAA_CHECK(
                importedTex->getNativeObject(nvrhi::ObjectTypes::D3D11_Resource).pointer == texImport.handle.pointer,
                "Imported D3D11 texture native pointer mismatch");

            const std::vector<uint8_t> pixels = TestFixture::makeDeterministicBytes(size_t(kW) * kH * 4u, ctx.seed() ^ 0x1102u);
            ctx.fixture->uploadTexture2D(importedTex, 0, 0, pixels.data());
            std::vector<uint8_t> readPixels;
            ctx.fixture->readbackTexture2D(importedTex, 0, 0, readPixels);
            UAA_CHECK_EQ(readPixels.size(), pixels.size(), "D3D11 imported texture readback size mismatch");
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(readPixels.data(), pixels.data(), pixels.size(), diff), diff);

            importedTex = nullptr;
            device->runGarbageCollection();
            if (texImport.release)
            {
                texImport.release();
            }
        }

        // Test secondary D3D11 device creation.
        {
            MessageCapture secMsgs;
            std::unique_ptr<IBackendContext> sec = ctx.fixture->backend()->createSecondaryDevice(&secMsgs);
            UAA_CHECK(sec != nullptr && sec->getRawDevice() != nullptr, "createSecondaryDevice failed on D3D11");
            UAA_CHECK_EQ(secMsgs.errorCount(), size_t(0), "Secondary D3D11 device reported errors");
        }
    }

    UAA_REGISTER_TEST("NAT.D3D11", "native,d3d11", testD3D11Native);
} // namespace nvrhi::uaa

#else // !_WIN32

namespace nvrhi::uaa
{
    std::unique_ptr<IBackendContext> createD3D11Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        (void)config;
        (void)messageCallback;
        outIsUnavailable = true;
        outSkipOrError = "D3D11 backend requires Windows";
        return nullptr;
    }
} // namespace nvrhi::uaa

#endif // _WIN32

#endif // UAA_WITH_D3D11 || UAA_WITH_DX11
