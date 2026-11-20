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

#if (defined(UAA_WITH_D3D12) && UAA_WITH_D3D12) || (defined(UAA_WITH_DX12) && UAA_WITH_DX12)

#include "uaa_backend.h"
#include "uaa_fixture.h"
#include "uaa_runner.h"

#ifdef _WIN32

#include <nvrhi/d3d12.h>

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
#include <d3d12sdklayers.h>
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
        using PFN_D3D12GetDebugInterface = HRESULT(WINAPI*)(REFIID, void**);
        using PFN_D3D12CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

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

        class D3D12GpuGate final : public IGpuGate
        {
        public:
            D3D12GpuGate(ID3D12Device* d3dDevice, nvrhi::IDevice* nvrhiDevice)
                : m_D3DDevice(d3dDevice)
                , m_NvrhiDevice(nvrhiDevice)
            {
                createFence();
            }

            bool isValid() const
            {
                return m_Fence != nullptr;
            }

            bool enqueueWait(nvrhi::CommandQueue queue) override
            {
                if (!m_NvrhiDevice || !m_Fence)
                {
                    return false;
                }
                auto* nativeQueue = static_cast<ID3D12CommandQueue*>(
                    m_NvrhiDevice->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, queue).pointer);
                if (!nativeQueue)
                {
                    return false;
                }
                return SUCCEEDED(nativeQueue->Wait(m_Fence.Get(), 1));
            }

            void signal() override
            {
                if (m_Fence)
                {
                    m_Fence->Signal(1);
                }
            }

            void reset() override
            {
                if (m_NvrhiDevice)
                {
                    m_NvrhiDevice->waitForIdle();
                }
                m_Fence.Reset();
                createFence();
            }

        private:
            void createFence()
            {
                if (m_D3DDevice)
                {
                    if (FAILED(m_D3DDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence))))
                    {
                        m_Fence.Reset();
                    }
                }
            }

            ID3D12Device* m_D3DDevice = nullptr;
            nvrhi::IDevice* m_NvrhiDevice = nullptr;
            Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;
        };

        class D3D12BackendContext final : public IBackendContext
        {
        public:
            D3D12BackendContext() = default;

            ~D3D12BackendContext() override
            {
                if (m_NvrhiDevice)
                {
                    m_NvrhiDevice->waitForIdle();
                    m_NvrhiDevice->runGarbageCollection();
                    m_NvrhiDevice = nullptr;
                }
                m_InfoQueue.Reset();
                m_CopyQueue.Reset();
                m_ComputeQueue.Reset();
                m_GraphicsQueue.Reset();
                m_D3D12Device.Reset();
                m_Adapter.Reset();
                if (m_D3D12Dll)
                {
                    FreeLibrary(m_D3D12Dll);
                    m_D3D12Dll = nullptr;
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

            std::unique_ptr<IGpuGate> createGpuGate() override
            {
                if (!m_D3D12Device || !m_NvrhiDevice)
                {
                    return nullptr;
                }
                auto gate = std::make_unique<D3D12GpuGate>(m_D3D12Device.Get(), m_NvrhiDevice.Get());
                return gate->isValid() ? std::move(gate) : nullptr;
            }

            bool createNativeImportBuffer(uint64_t byteSize, const void* initialData, NativeBufferImport& out) override
            {
                if (!m_D3D12Device || !m_GraphicsQueue)
                {
                    return false;
                }

                const uint64_t allocSize = byteSize > 0 ? byteSize : 4ull;

                D3D12_HEAP_PROPERTIES defaultHeap{};
                defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

                D3D12_RESOURCE_DESC resDesc{};
                resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                resDesc.Width = allocSize;
                resDesc.Height = 1;
                resDesc.DepthOrArraySize = 1;
                resDesc.MipLevels = 1;
                resDesc.Format = DXGI_FORMAT_UNKNOWN;
                resDesc.SampleDesc.Count = 1;
                resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

                Microsoft::WRL::ComPtr<ID3D12Resource> buf;
                if (FAILED(m_D3D12Device->CreateCommittedResource(
                        &defaultHeap,
                        D3D12_HEAP_FLAG_NONE,
                        &resDesc,
                        D3D12_RESOURCE_STATE_COMMON,
                        nullptr,
                        IID_PPV_ARGS(&buf))) || !buf)
                {
                    return false;
                }

                if (initialData && byteSize > 0)
                {
                    D3D12_HEAP_PROPERTIES uploadHeap{};
                    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
                    D3D12_RESOURCE_DESC upDesc = resDesc;
                    upDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

                    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuf;
                    if (FAILED(m_D3D12Device->CreateCommittedResource(
                            &uploadHeap,
                            D3D12_HEAP_FLAG_NONE,
                            &upDesc,
                            D3D12_RESOURCE_STATE_GENERIC_READ,
                            nullptr,
                            IID_PPV_ARGS(&uploadBuf))) || !uploadBuf)
                    {
                        return false;
                    }

                    void* mapped = nullptr;
                    D3D12_RANGE emptyRange{ 0, 0 };
                    if (FAILED(uploadBuf->Map(0, &emptyRange, &mapped)) || !mapped)
                    {
                        return false;
                    }
                    std::memcpy(mapped, initialData, static_cast<size_t>(byteSize));
                    uploadBuf->Unmap(0, nullptr);

                    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdAlloc;
                    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmdList;
                    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
                    if (FAILED(m_D3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc))) ||
                        FAILED(m_D3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&cmdList))) ||
                        FAILED(m_D3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
                    {
                        return false;
                    }

                    D3D12_RESOURCE_BARRIER toCopyDest{};
                    toCopyDest.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    toCopyDest.Transition.pResource = buf.Get();
                    toCopyDest.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                    toCopyDest.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                    toCopyDest.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cmdList->ResourceBarrier(1, &toCopyDest);
                    cmdList->CopyBufferRegion(buf.Get(), 0, uploadBuf.Get(), 0, byteSize);
                    D3D12_RESOURCE_BARRIER toCommon = toCopyDest;
                    toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                    toCommon.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                    cmdList->ResourceBarrier(1, &toCommon);
                    cmdList->Close();

                    ID3D12CommandList* lists[] = { cmdList.Get() };
                    m_GraphicsQueue->ExecuteCommandLists(1, lists);
                    m_GraphicsQueue->Signal(fence.Get(), 1);
                    if (fence->GetCompletedValue() < 1)
                    {
                        HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                        if (ev)
                        {
                            fence->SetEventOnCompletion(1, ev);
                            WaitForSingleObject(ev, INFINITE);
                            CloseHandle(ev);
                        }
                    }
                }

                auto holder = std::make_shared<Microsoft::WRL::ComPtr<ID3D12Resource>>(buf);
                out.objectType = nvrhi::ObjectTypes::D3D12_Resource;
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
                if (!m_D3D12Device)
                {
                    return false;
                }

                const DXGI_FORMAT dxgiFormat = nvrhi::d3d12::convertFormat(format);
                if (dxgiFormat == DXGI_FORMAT_UNKNOWN)
                {
                    return false;
                }

                const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(format);
                const bool isDepth = fmtInfo.hasDepth || fmtInfo.hasStencil;

                D3D12_HEAP_PROPERTIES defaultHeap{};
                defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

                D3D12_RESOURCE_DESC resDesc{};
                resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                resDesc.Width = width;
                resDesc.Height = height;
                resDesc.DepthOrArraySize = 1;
                resDesc.MipLevels = 1;
                resDesc.Format = dxgiFormat;
                resDesc.SampleDesc.Count = 1;
                resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                resDesc.Flags = isDepth
                    ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                    : (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

                Microsoft::WRL::ComPtr<ID3D12Resource> tex;
                if (FAILED(m_D3D12Device->CreateCommittedResource(
                        &defaultHeap,
                        D3D12_HEAP_FLAG_NONE,
                        &resDesc,
                        D3D12_RESOURCE_STATE_COMMON,
                        nullptr,
                        IID_PPV_ARGS(&tex))) || !tex)
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
                desc.debugName = "uaa_native_import_d3d12_tex";

                auto holder = std::make_shared<Microsoft::WRL::ComPtr<ID3D12Resource>>(tex);
                out.objectType = nvrhi::ObjectTypes::D3D12_Resource;
                out.handle = nvrhi::Object(static_cast<void*>(tex.Get()));
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
                auto sec = std::make_unique<D3D12BackendContext>();
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
                    auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                    if (FAILED(m_InfoQueue->GetMessage(i, message, &msgLen)))
                    {
                        continue;
                    }
                    if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
                        message->Severity == D3D12_MESSAGE_SEVERITY_ERROR)
                    {
                        outErrors.emplace_back(message->pDescription ? message->pDescription : "D3D12 validation error");
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
                m_D3D12Dll = LoadLibraryA("d3d12.dll");
                if (!m_DxgiDll || !m_D3D12Dll)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "dxgi.dll or d3d12.dll not available";
                    return false;
                }

                auto createDxgiFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(
                    GetProcAddress(m_DxgiDll, "CreateDXGIFactory1"));
                auto d3d12GetDebugInterface = reinterpret_cast<PFN_D3D12GetDebugInterface>(
                    GetProcAddress(m_D3D12Dll, "D3D12GetDebugInterface"));
                auto d3d12CreateDevice = reinterpret_cast<PFN_D3D12CreateDevice>(
                    GetProcAddress(m_D3D12Dll, "D3D12CreateDevice"));

                if (!createDxgiFactory1 || !d3d12CreateDevice)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "Required D3D12/DXGI entry points unavailable";
                    return false;
                }

                bool debugEnabled = false;
                if (config.nativeValidation && d3d12GetDebugInterface)
                {
                    Microsoft::WRL::ComPtr<ID3D12Debug> debug;
                    if (SUCCEEDED(d3d12GetDebugInterface(IID_PPV_ARGS(&debug))) && debug)
                    {
                        debug->EnableDebugLayer();
                        debugEnabled = true;
                    }
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
                            outSkipOrError = "No D3D12 adapter matching selector '" + config.deviceSelector + "'";
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

                HRESULT hr = E_FAIL;
                if (!useWarp && m_Adapter)
                {
                    hr = d3d12CreateDevice(m_Adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_D3D12Device));
                    if (FAILED(hr) && config.deviceSelector.empty())
                    {
                        useWarp = true;
                    }
                }

                if (useWarp)
                {
                    Microsoft::WRL::ComPtr<IDXGIFactory4> factory4;
                    if (SUCCEEDED(factory.As(&factory4)) && factory4)
                    {
                        Microsoft::WRL::ComPtr<IDXGIAdapter1> warpAdapter;
                        if (SUCCEEDED(factory4->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter))) && warpAdapter)
                        {
                            m_Adapter = warpAdapter;
                            hr = d3d12CreateDevice(m_Adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_D3D12Device));
                        }
                    }
                }

                if (FAILED(hr) || !m_D3D12Device)
                {
                    outIsUnavailable = true;
                    outSkipOrError = "D3D12CreateDevice failed";
                    return false;
                }

                if (debugEnabled)
                {
                    m_D3D12Device.As(&m_InfoQueue);
                }

                D3D12_COMMAND_QUEUE_DESC gfxQDesc{};
                gfxQDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
                if (FAILED(m_D3D12Device->CreateCommandQueue(&gfxQDesc, IID_PPV_ARGS(&m_GraphicsQueue))) || !m_GraphicsQueue)
                {
                    outIsUnavailable = false;
                    outSkipOrError = "Failed to create D3D12 graphics command queue";
                    return false;
                }

                D3D12_COMMAND_QUEUE_DESC compQDesc{};
                compQDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
                m_D3D12Device->CreateCommandQueue(&compQDesc, IID_PPV_ARGS(&m_ComputeQueue));

                D3D12_COMMAND_QUEUE_DESC copyQDesc{};
                copyQDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
                m_D3D12Device->CreateCommandQueue(&copyQDesc, IID_PPV_ARGS(&m_CopyQueue));

                nvrhi::d3d12::DeviceDesc desc;
                desc.errorCB = messageCallback;
                desc.pDevice = m_D3D12Device.Get();
                desc.pGraphicsCommandQueue = m_GraphicsQueue.Get();
                desc.pComputeCommandQueue = m_ComputeQueue.Get();
                desc.pCopyCommandQueue = m_CopyQueue.Get();

                m_NvrhiDevice = nvrhi::d3d12::createDevice(desc);
                if (!m_NvrhiDevice)
                {
                    outIsUnavailable = false;
                    outSkipOrError = "nvrhi::d3d12::createDevice returned null";
                    return false;
                }

                std::string adapterName = useWarp ? "Microsoft Basic Render Driver (WARP)" : "D3D12 Adapter";
                if (m_Adapter)
                {
                    DXGI_ADAPTER_DESC1 ad{};
                    if (SUCCEEDED(m_Adapter->GetDesc1(&ad)))
                    {
                        adapterName = wideToUtf8(ad.Description);
                    }
                }

                m_DeviceInfo.backendName = "d3d12";
                m_DeviceInfo.deviceName = adapterName;
                m_DeviceInfo.driverName = useWarp ? "WARP" : "D3D12";
                m_DeviceInfo.driverVersion = "12.0";
                m_DeviceInfo.osName = "Windows";
#ifdef UAA_BUILD_CONFIG
                m_DeviceInfo.buildConfig = UAA_BUILD_CONFIG;
#endif
                m_DeviceInfo.nativeValidation = debugEnabled && (m_InfoQueue != nullptr);
                return true;
            }

        private:
            HMODULE m_DxgiDll = nullptr;
            HMODULE m_D3D12Dll = nullptr;
            std::string m_SelectorUsed;
            Microsoft::WRL::ComPtr<IDXGIAdapter1> m_Adapter;
            Microsoft::WRL::ComPtr<ID3D12Device> m_D3D12Device;
            Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_GraphicsQueue;
            Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_ComputeQueue;
            Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CopyQueue;
            Microsoft::WRL::ComPtr<ID3D12InfoQueue> m_InfoQueue;
            nvrhi::d3d12::DeviceHandle m_NvrhiDevice;
            DeviceInfo m_DeviceInfo;
        };
    } // namespace

    std::unique_ptr<IBackendContext> createD3D12Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        auto ctx = std::make_unique<D3D12BackendContext>();
        if (!ctx->init(config, messageCallback, outSkipOrError, outIsUnavailable))
        {
            return nullptr;
        }
        return ctx;
    }

    static void testD3D12Native(TestContext& ctx)
    {
        UAA_CHECK(ctx.fixture != nullptr, "Fixture must be initialised");
        if (ctx.fixture->api() != nvrhi::GraphicsAPI::D3D12)
        {
            UAA_SKIP("NAT.D3D12 requires the D3D12 backend");
        }

        nvrhi::IDevice* device = ctx.fixture->device();
        auto* nativeDev = static_cast<ID3D12Device*>(
            device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device).pointer);
        auto* nativeGfxQueue = static_cast<ID3D12CommandQueue*>(
            device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics).pointer);
        auto* d3d12Dev = static_cast<nvrhi::d3d12::IDevice*>(
            device->getNativeObject(nvrhi::ObjectTypes::Nvrhi_D3D12_Device).pointer);

        UAA_CHECK(nativeDev != nullptr, "D3D12_Device native object returned null");
        UAA_CHECK(nativeGfxQueue != nullptr, "D3D12_CommandQueue native object returned null");
        UAA_CHECK(d3d12Dev != nullptr, "Nvrhi_D3D12_Device native object returned null");

        UAA_CHECK(
            d3d12Dev->getDescriptorHeap(nvrhi::d3d12::DescriptorHeapType::ShaderResourceView) != nullptr,
            "D3D12 SRV descriptor heap returned null");

        // Test imported ID3D12Resource buffer & GPU virtual address query.
        {
            constexpr uint64_t kBufSize = 256;
            const std::vector<uint8_t> initBytes = TestFixture::makeDeterministicBytes(kBufSize, ctx.seed() ^ 0xD312u);
            NativeBufferImport bufImport;
            UAA_CHECK(
                ctx.fixture->backend()->createNativeImportBuffer(kBufSize, initBytes.data(), bufImport),
                "createNativeImportBuffer failed on D3D12");
            UAA_CHECK_EQ(bufImport.objectType, nvrhi::ObjectTypes::D3D12_Resource, "D3D12 buffer import type mismatch");

            nvrhi::BufferDesc bufDesc;
            bufDesc.byteSize = kBufSize;
            bufDesc.initialState = nvrhi::ResourceStates::Common;
            bufDesc.keepInitialState = true;
            bufDesc.debugName = "uaa_d3d12_imported_buf";
            nvrhi::BufferHandle imported = device->createHandleForNativeBuffer(
                bufImport.objectType, bufImport.handle, bufDesc);
            UAA_CHECK(imported != nullptr, "createHandleForNativeBuffer failed on D3D12");
            UAA_CHECK(
                imported->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer == bufImport.handle.pointer,
                "Imported D3D12 buffer native pointer mismatch");

            nvrhi::CommandListHandle cmd = device->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList failed on D3D12");
            auto* d3d12Cmd = static_cast<nvrhi::d3d12::ICommandList*>(
                cmd->getNativeObject(nvrhi::ObjectTypes::Nvrhi_D3D12_CommandList).pointer);
            UAA_CHECK(d3d12Cmd != nullptr, "Nvrhi_D3D12_CommandList returned null");
            UAA_CHECK(d3d12Cmd->getBufferGpuVA(imported) != 0, "getBufferGpuVA returned 0 for imported buffer");

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

        // Test imported ID3D12Resource texture.
        {
            constexpr uint32_t kW = 16;
            constexpr uint32_t kH = 16;
            NativeTextureImport texImport;
            UAA_CHECK(
                ctx.fixture->backend()->createNativeImportTexture(kW, kH, nvrhi::Format::RGBA8_UNORM, texImport),
                "createNativeImportTexture failed on D3D12");
            UAA_CHECK_EQ(texImport.objectType, nvrhi::ObjectTypes::D3D12_Resource, "D3D12 texture import type mismatch");

            nvrhi::TextureHandle importedTex = device->createHandleForNativeTexture(
                texImport.objectType, texImport.handle, texImport.desc);
            UAA_CHECK(importedTex != nullptr, "createHandleForNativeTexture failed on D3D12");
            UAA_CHECK(
                importedTex->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer == texImport.handle.pointer,
                "Imported D3D12 texture native pointer mismatch");

            const std::vector<uint8_t> pixels = TestFixture::makeDeterministicBytes(size_t(kW) * kH * 4u, ctx.seed() ^ 0x1202u);
            ctx.fixture->uploadTexture2D(importedTex, 0, 0, pixels.data());
            std::vector<uint8_t> readPixels;
            ctx.fixture->readbackTexture2D(importedTex, 0, 0, readPixels);
            UAA_CHECK_EQ(readPixels.size(), pixels.size(), "D3D12 imported texture readback size mismatch");
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(readPixels.data(), pixels.data(), pixels.size(), diff), diff);

            importedTex = nullptr;
            device->runGarbageCollection();
            if (texImport.release)
            {
                texImport.release();
            }
        }

        // Test D3D12GpuGate wait/signal/reset.
        {
            std::unique_ptr<IGpuGate> gate = ctx.fixture->backend()->createGpuGate();
            UAA_CHECK(gate != nullptr, "createGpuGate returned null on D3D12");

            nvrhi::BufferDesc dstDesc;
            dstDesc.byteSize = 64;
            dstDesc.initialState = nvrhi::ResourceStates::CopyDest;
            dstDesc.keepInitialState = true;
            dstDesc.debugName = "uaa_d3d12_gate_dst";
            nvrhi::BufferHandle dstBuf = device->createBuffer(dstDesc);
            UAA_CHECK(dstBuf != nullptr, "Failed to create D3D12 gate buffer");

            const std::vector<uint8_t> payload = TestFixture::makeDeterministicBytes(64, ctx.seed() ^ 0xF00Du);
            nvrhi::CommandListHandle cmd = device->createCommandList();
            cmd->open();
            cmd->writeBuffer(dstBuf, payload.data(), payload.size());
            cmd->close();

            UAA_CHECK(gate->enqueueWait(nvrhi::CommandQueue::Graphics), "D3D12GpuGate::enqueueWait failed");
            device->executeCommandList(cmd, nvrhi::CommandQueue::Graphics);
            gate->signal();
            UAA_CHECK(device->waitForIdle(), "waitForIdle failed after D3D12GpuGate::signal()");

            std::vector<uint8_t> actual(64, 0);
            ctx.fixture->readbackBuffer(dstBuf, actual.data(), actual.size());
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), payload.data(), payload.size(), diff), diff);

            gate->reset();
            UAA_CHECK(gate->enqueueWait(nvrhi::CommandQueue::Graphics), "D3D12GpuGate::enqueueWait after reset failed");
            gate->signal();
        }

        // Test secondary D3D12 device creation.
        {
            MessageCapture secMsgs;
            std::unique_ptr<IBackendContext> sec = ctx.fixture->backend()->createSecondaryDevice(&secMsgs);
            UAA_CHECK(sec != nullptr && sec->getRawDevice() != nullptr, "createSecondaryDevice failed on D3D12");
            UAA_CHECK_EQ(secMsgs.errorCount(), size_t(0), "Secondary D3D12 device reported errors");
        }
    }

    UAA_REGISTER_TEST("NAT.D3D12", "native,d3d12", testD3D12Native);
} // namespace nvrhi::uaa

#else // !_WIN32

namespace nvrhi::uaa
{
    std::unique_ptr<IBackendContext> createD3D12Backend(
        const BackendConfig& config,
        nvrhi::IMessageCallback* messageCallback,
        std::string& outSkipOrError,
        bool& outIsUnavailable)
    {
        (void)config;
        (void)messageCallback;
        outIsUnavailable = true;
        outSkipOrError = "D3D12 backend requires Windows";
        return nullptr;
    }
} // namespace nvrhi::uaa

#endif // _WIN32

#endif // UAA_WITH_D3D12 || UAA_WITH_DX12
