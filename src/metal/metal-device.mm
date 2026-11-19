/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "metal-backend.h"

#include <nvrhi/common/aftermath.h>
#include <nvrhi/common/misc.h>

namespace nvrhi::metal
{
    void Context::error(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Error, message.c_str());
    }

    void Context::warning(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Warning, message.c_str());
    }

    void Context::info(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Info, message.c_str());
    }

    // A typed buffer is a texture in the shader but a buffer in NVRHI, so the binding needs
    // a texture-buffer view of the range. The views are cached on the buffer because the
    // command list binds them at draw time.
    id<MTLTexture> getTypedBufferView(Context& context, Buffer* buffer, Format format, uint64_t byteOffset, uint64_t byteSize)
    {
        const Format viewFormat = format == Format::UNKNOWN ? Format::R32_UINT : format;
        const MTLPixelFormat pixelFormat = convertFormat(viewFormat);
        const FormatInfo& formatInfo = getFormatInfo(viewFormat);
        if (pixelFormat == MTLPixelFormatInvalid || formatInfo.bytesPerBlock == 0)
        {
            context.error("getTypedBufferView: the typed buffer format is not supported");
            return nil;
        }

        if (byteSize == 0)
            byteSize = buffer->desc.byteSize - byteOffset;

        const auto key = std::make_tuple(viewFormat, byteOffset, byteSize);
        const auto cached = buffer->typedViews.find(key);
        if (cached != buffer->typedViews.end())
            return cached->second;

        const uint64_t alignment = [context.device minimumTextureBufferAlignmentForPixelFormat:pixelFormat];
        const uint64_t width = byteSize / formatInfo.bytesPerBlock;
        if ((byteOffset % alignment) != 0 || (byteSize % formatInfo.bytesPerBlock) != 0
            || byteOffset + byteSize > buffer->desc.byteSize)
        {
            context.error("getTypedBufferView: the range cannot be viewed as a typed buffer");
            return nil;
        }

        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pixelFormat
            width:width resourceOptions:buffer->buffer.resourceOptions
            usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
        id<MTLTexture> view = [buffer->buffer newTextureWithDescriptor:descriptor offset:byteOffset bytesPerRow:width * formatInfo.bytesPerBlock];

        buffer->typedViews.emplace(key, view);
        return view;
    }

    DeviceHandle createDevice(const DeviceDesc& desc)
    {
        Device* device = new Device(desc);
        return DeviceHandle::Create(device);
    }

    Device::Device(const DeviceDesc& desc)
    {
        m_Context.messageCallback = desc.errorCB;

        if (desc.device.pointer)
        {
            m_Context.device = (__bridge id<MTLDevice>)desc.device.pointer;
        }
        else
        {
            m_Context.device = MTLCreateSystemDefaultDevice();
        }

        if (!m_Context.device)
        {
            m_Context.error("createDevice: no Metal device is available");
            return;
        }

        m_Context.queue = [m_Context.device newCommandQueue];
        if (!m_Context.queue)
        {
            m_Context.error("createDevice: no Metal command queue is available");
            return;
        }
    }

    Device::~Device()
    {
    }

    Object Device::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::Metal_Device: return Object((__bridge void*)m_Context.device);
        case ObjectTypes::Metal_CommandQueue: return Object((__bridge void*)m_Context.queue);
        default: return nullptr;
        }
    }

    Object Device::getNativeDevice()
    {
        return Object((__bridge void*)m_Context.device);
    }

    Object Device::getNativeCommandQueue()
    {
        return Object((__bridge void*)m_Context.queue);
    }

    // [UAA] - START - adapter memory size
    uint64_t Device::getNativeDeviceMemorySize()
    {
        return m_Context.device.recommendedMaxWorkingSetSize;
    }

    uint64_t Device::getNativeAllocatedSize()
    {
        return m_Context.device.currentAllocatedSize;
    }
    // [UAA] - END

    // [UAA] - START - mapped buffer write flush
    uint64_t Device::getMappedBufferFlushAlignment()
    {
        return 256;
    }

    bool Device::flushMappedBuffer(IBuffer* buffer, uint64_t offset, uint64_t size)
    {
        Buffer* buf = checked_cast<Buffer*>(buffer);
        const bool supported = buf && buf->buffer && buf->desc.cpuAccess == CpuAccessMode::Write;

        if (!supported)
        {
            m_Context.error("flushMappedBuffer: the buffer must be an NVRHI-owned write buffer");
            return false;
        }

        const bool inBounds = offset <= buf->desc.byteSize && size <= buf->desc.byteSize - offset;
        if (size == 0 || !inBounds)
        {
            m_Context.error("flushMappedBuffer: the range is invalid");
            return false;
        }

        // Shared storage is coherent on Apple silicon, so only managed storage needs a publish.
        if (buf->buffer.storageMode == MTLStorageModeManaged)
            [buf->buffer didModifyRange:NSMakeRange(offset, size)];

        return true;
    }
    // [UAA] - END

    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        if (objectType == ObjectTypes::Metal_CommandQueue)
            return Object((__bridge void*)m_Context.queue);

        return nullptr;
    }

    IMessageCallback* Device::getMessageCallback()
    {
        return m_Context.messageCallback;
    }

    bool Device::isAftermathEnabled()
    {
        return false;
    }

    AftermathCrashDumpHelper& Device::getAftermathCrashDumpHelper()
    {
        static AftermathCrashDumpHelper helper;
        return helper;
    }

    GraphicsAPI Device::getGraphicsAPI()
    {
        return GraphicsAPI::METAL;
    }

    // ------------------------------------------------------------------ Resources

    HeapHandle Device::createHeap(const HeapDesc& d)
    {
        MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
        descriptor.size = d.capacity;
        descriptor.type = MTLHeapTypePlacement;

        switch (d.type)
        {
            case HeapType::DeviceLocal:
                descriptor.storageMode = MTLStorageModePrivate;
                break;
            case HeapType::Upload:
                descriptor.storageMode = MTLStorageModeShared;
                descriptor.cpuCacheMode = MTLCPUCacheModeWriteCombined;
                break;
            case HeapType::Readback:
                descriptor.storageMode = MTLStorageModeShared;
                descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
                break;
            default:
                utils::InvalidEnum();
                return nullptr;
        }

        id<MTLHeap> heap = [m_Context.device newHeapWithDescriptor:descriptor];
        if (!heap)
        {
            m_Context.error("createHeap: failed to create the Metal heap");
            return nullptr;
        }

        if (!d.debugName.empty())
            heap.label = [NSString stringWithUTF8String:d.debugName.c_str()];

        Heap* result = new Heap();
        result->desc = d;
        result->heap = heap;
        return HeapHandle::Create(result);
    }

    TextureHandle Device::createTexture(const TextureDesc& d)
    {
        const MTLPixelFormat pixelFormat = convertFormat(d.format);
        if (pixelFormat == MTLPixelFormatInvalid)
        {
            m_Context.error("createTexture: unsupported texture format");
            return nullptr;
        }

        MTLTextureType textureType;
        switch (d.dimension)
        {
        case TextureDimension::Texture1D:      textureType = MTLTextureType1D; break;
        case TextureDimension::Texture1DArray: textureType = MTLTextureType1DArray; break;
        case TextureDimension::Texture2D:      textureType = MTLTextureType2D; break;
        case TextureDimension::Texture2DArray: textureType = MTLTextureType2DArray; break;
        case TextureDimension::Texture3D:      textureType = MTLTextureType3D; break;
        case TextureDimension::TextureCube:    textureType = MTLTextureTypeCube; break;
        case TextureDimension::TextureCubeArray: textureType = MTLTextureTypeCubeArray; break;
        default:
            m_Context.error("createTexture: unsupported texture dimension");
            return nullptr;
        }

        MTLTextureUsage usage = 0;
        if (d.isShaderResource) usage |= MTLTextureUsageShaderRead;
        if (d.isUAV) usage |= MTLTextureUsageShaderWrite;
        if (d.isRenderTarget) usage |= MTLTextureUsageRenderTarget;
        if (usage == 0)
        {
            m_Context.error("createTexture: the texture has no usage flags set");
            return nullptr;
        }

        MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
        descriptor.textureType = textureType;
        descriptor.pixelFormat = pixelFormat;
        descriptor.width = d.width;
        descriptor.height = d.height;
        descriptor.depth = d.depth;
        descriptor.mipmapLevelCount = d.mipLevels;
        descriptor.arrayLength = d.arraySize;
        descriptor.sampleCount = d.sampleCount;
        descriptor.usage = usage;
        descriptor.storageMode = MTLStorageModePrivate;

        // Metal counts the cube faces as part of the slice, not as array slices.
        if (textureType == MTLTextureTypeCube)
        {
            descriptor.arrayLength = 1;
        }
        else if (textureType == MTLTextureTypeCubeArray)
        {
            descriptor.arrayLength = d.arraySize / 6;
        }

        // A virtual texture defers its Metal object to bindTextureMemory, which places it
        // on a heap at the offset the caller chose from the requirements below.
        if (d.isVirtual)
        {
            Texture* result = new Texture();
            result->desc = d;
            result->descriptor = descriptor;
            return TextureHandle::Create(result);
        }

        id<MTLTexture> texture = [m_Context.device newTextureWithDescriptor:descriptor];

        if (!texture)
        {
            m_Context.error("createTexture: failed to create the Metal texture");
            return nullptr;
        }

        Texture* result = new Texture();
        result->desc = d;
        result->texture = texture;
        return TextureHandle::Create(result);
    }

    MemoryRequirements Device::getTextureMemoryRequirements(ITexture* texture)
    {
        MemoryRequirements requirements;
        Texture* tex = checked_cast<Texture*>(texture);
        if (tex && tex->texture)
        {
            requirements.size = tex->texture.allocatedSize;
        }
        else if (tex && tex->descriptor)
        {
            const MTLSizeAndAlign sizeAndAlign = [m_Context.device heapTextureSizeAndAlignWithDescriptor:tex->descriptor];
            requirements.size = sizeAndAlign.size;
            requirements.alignment = sizeAndAlign.align;
        }

        return requirements;
    }

    bool Device::bindTextureMemory(ITexture* _texture, IHeap* _heap, uint64_t offset)
    {
        Texture* texture = checked_cast<Texture*>(_texture);
        Heap* heap = checked_cast<Heap*>(_heap);
        if (!texture || !texture->descriptor || !heap || !heap->heap)
        {
            m_Context.error("bindTextureMemory: the texture is not virtual or the heap is invalid");
            return false;
        }

        // The storage modes have to match between the heap and the placed texture.
        if (texture->descriptor.storageMode != heap->heap.storageMode)
        {
            m_Context.error("bindTextureMemory: the heap storage mode does not match the texture");
            return false;
        }

        id<MTLTexture> placed = [heap->heap newTextureWithDescriptor:texture->descriptor offset:offset];
        if (!placed)
        {
            m_Context.error("bindTextureMemory: failed to place the texture on the heap");
            return false;
        }

        texture->texture = placed;
        texture->descriptor = nil;
        return true;
    }

    TextureHandle Device::createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc)
    {
        if (objectType != ObjectTypes::Metal_Texture)
            return nullptr;

        Texture* result = new Texture();
        result->desc = desc;
        result->texture = (__bridge id<MTLTexture>)texture.pointer;
        return TextureHandle::Create(result);
    }

    StagingTextureHandle Device::createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess)
    {
        StagingTexture* staging = new StagingTexture();
        staging->desc = d;

        const uint32_t bytesPerPixel = getFormatInfo(d.format).bytesPerBlock;
        if (bytesPerPixel == 0)
        {
            m_Context.error("createStagingTexture: unsupported texture format");
            return nullptr;
        }

        uint64_t offset = 0;
        for (uint32_t mip = 0; mip < d.mipLevels; ++mip)
        {
            const uint32_t width = std::max(1u, d.width >> mip);
            const uint32_t height = std::max(1u, d.height >> mip);

            staging->mipOffsets.push_back(offset);
            staging->mipRowPitches.push_back(alignSize(uint64_t(width) * bytesPerPixel, 256));
            offset += uint64_t(staging->mipRowPitches.back()) * height;
        }

        staging->bytesPerRow = staging->mipRowPitches.empty() ? 0 : staging->mipRowPitches.front();
        staging->buffer = [m_Context.device newBufferWithLength:offset options:MTLResourceStorageModeShared];

        if (!staging->buffer)
        {
            m_Context.error("createStagingTexture: failed to create the staging buffer");
            return nullptr;
        }

        return StagingTextureHandle::Create(staging);
    }

    void* Device::mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t* outRowPitch)
    {
        StagingTexture* staging = checked_cast<StagingTexture*>(tex);
        if (!staging || !staging->buffer)
            return nullptr;

        const uint32_t mip = slice.mipLevel;
        if (mip >= staging->mipOffsets.size())
            return nullptr;

        if (outRowPitch)
            *outRowPitch = staging->mipRowPitches[mip];

        staging->mappedMemory = staging->buffer.contents;
        return static_cast<uint8_t*>(staging->mappedMemory) + staging->mipOffsets[mip];
    }

    void Device::unmapStagingTexture(IStagingTexture* tex)
    {
        StagingTexture* staging = checked_cast<StagingTexture*>(tex);
        if (staging)
            staging->mappedMemory = nullptr;
    }

    void Device::getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings)
    {
        (void)texture;
        (void)desc;
        (void)tileShape;
        (void)subresourceTilings;

        if (numTiles)
            *numTiles = 0;
        if (subresourceTilingsNum)
            *subresourceTilingsNum = 0;
    }

    void Device::updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue)
    {
        (void)texture;
        (void)tileMappings;
        (void)numTileMappings;
        (void)executionQueue;
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc)
    {
        (void)pairedTexture;
        (void)desc;
        m_Context.error("createSamplerFeedbackTexture: sampler feedback is not supported by the Metal backend");
        return nullptr;
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture)
    {
        (void)objectType;
        (void)texture;
        (void)pairedTexture;
        return nullptr;
    }

    BufferHandle Device::createBuffer(const BufferDesc& d)
    {
        if (d.byteSize == 0)
        {
            m_Context.error("createBuffer: the buffer size is zero");
            return nullptr;
        }

        // Apple silicon has unified memory, so shared storage keeps the buffer host visible
        // and coherent for both the CPU and the GPU.
        id<MTLBuffer> buffer = [m_Context.device newBufferWithLength:d.byteSize options:MTLResourceStorageModeShared];
        if (!buffer)
        {
            m_Context.error("createBuffer: failed to create the Metal buffer");
            return nullptr;
        }

        Buffer* result = new Buffer();
        result->desc = d;
        result->buffer = buffer;
        result->isCPUVisible = true;
        return BufferHandle::Create(result);
    }

    void* Device::mapBuffer(IBuffer* buffer, CpuAccessMode cpuAccess)
    {
        Buffer* buf = checked_cast<Buffer*>(buffer);
        if (!buf || !buf->isCPUVisible)
            return nullptr;

        buf->mappedMemory = buf->buffer.contents;
        return buf->mappedMemory;
    }

    void Device::unmapBuffer(IBuffer* buffer)
    {
        Buffer* buf = checked_cast<Buffer*>(buffer);
        if (buf)
            buf->mappedMemory = nullptr;
    }

    MemoryRequirements Device::getBufferMemoryRequirements(IBuffer* buffer)
    {
        MemoryRequirements requirements;
        Buffer* buf = checked_cast<Buffer*>(buffer);
        if (buf)
        {
            requirements.size = buf->desc.byteSize;
            requirements.alignment = 256;
        }

        return requirements;
    }

    bool Device::bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset)
    {
        (void)buffer;
        (void)heap;
        (void)offset;
        m_Context.error("bindBufferMemory: heaps are not supported by the Metal backend");
        return false;
    }

    BufferHandle Device::createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc)
    {
        if (objectType != ObjectTypes::Metal_Buffer)
            return nullptr;

        Buffer* result = new Buffer();
        result->desc = desc;
        result->buffer = (__bridge id<MTLBuffer>)buffer.pointer;
        result->isCPUVisible = true;
        return BufferHandle::Create(result);
    }

    SamplerHandle Device::createSampler(const SamplerDesc& d)
    {
        MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
        descriptor.minFilter = convertSamplerFilter(d.minFilter);
        descriptor.magFilter = convertSamplerFilter(d.magFilter);
        descriptor.mipFilter = convertSamplerMipFilter(d.mipFilter);
        descriptor.sAddressMode = convertSamplerAddressMode(d.addressU);
        descriptor.tAddressMode = convertSamplerAddressMode(d.addressV);
        descriptor.rAddressMode = convertSamplerAddressMode(d.addressW);
        descriptor.borderColor = convertSamplerBorderColor(d.borderColor);
        descriptor.maxAnisotropy = d.maxAnisotropy;
        descriptor.lodMinClamp = 0.f;
        descriptor.lodMaxClamp = FLT_MAX;
        descriptor.compareFunction = convertCompareFunc(d.compareFunc);

        id<MTLSamplerState> sampler = [m_Context.device newSamplerStateWithDescriptor:descriptor];
        if (!sampler)
        {
            m_Context.error("createSampler: failed to create the Metal sampler");
            return nullptr;
        }

        Sampler* result = new Sampler();
        result->desc = d;
        result->sampler = sampler;
        return SamplerHandle::Create(result);
    }

    InputLayoutHandle Device::createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader)
    {
        (void)vertexShader;

        InputLayout* result = new InputLayout();
        result->attributes.assign(d, d + attributeCount);
        return InputLayoutHandle::Create(result);
    }

    // ------------------------------------------------------------------ Queries

    EventQueryHandle Device::createEventQuery()
    {
        return EventQueryHandle::Create(new EventQuery());
    }

    void Device::setEventQuery(IEventQuery* query, CommandQueue queue)
    {
        (void)queue;

        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (!eventQuery)
            return;

        std::lock_guard< std::mutex > lock(m_Mutex);
        eventQuery->serial = m_LastSubmittedSerial;
    }

    bool Device::pollEventQuery(IEventQuery* query)
    {
        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (!eventQuery)
            return true;

        std::lock_guard< std::mutex > lock(m_CompletionMutex);
        return m_LastCompletedSerial >= eventQuery->serial;
    }

    void Device::waitEventQuery(IEventQuery* query)
    {
        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (!eventQuery)
            return;

        std::unique_lock< std::mutex > lock(m_CompletionMutex);
        m_CompletionCondition.wait(lock, [this, eventQuery] { return m_LastCompletedSerial >= eventQuery->serial; });
    }

    void Device::resetEventQuery(IEventQuery* query)
    {
        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (eventQuery)
            eventQuery->serial = 0;
    }

    TimerQueryHandle Device::createTimerQuery()
    {
        return TimerQueryHandle::Create(new TimerQuery());
    }

    bool Device::pollTimerQuery(ITimerQuery* query)
    {
        (void)query;
        return false;
    }

    float Device::getTimerQueryTime(ITimerQuery* query)
    {
        TimerQuery* timerQuery = checked_cast<TimerQuery*>(query);
        return timerQuery ? timerQuery->time : 0.f;
    }

    void Device::resetTimerQuery(ITimerQuery* query)
    {
        TimerQuery* timerQuery = checked_cast<TimerQuery*>(query);
        if (timerQuery)
            timerQuery->resolved = false;
    }

    // ------------------------------------------------------------------ Pipelines

    FramebufferHandle Device::createFramebuffer(const FramebufferDesc& desc)
    {
        Framebuffer* result = new Framebuffer();
        result->desc = desc;
        result->framebufferInfo = FramebufferInfoEx(desc);

        // [UAA] - START - retain attachments as in the Vulkan framebuffer
        for (const auto& attachment : desc.colorAttachments)
            result->resources.push_back(attachment.texture);
        if (desc.depthAttachment.valid())
            result->resources.push_back(desc.depthAttachment.texture);
        if (desc.shadingRateAttachment.valid())
            result->resources.push_back(desc.shadingRateAttachment.texture);
        // [UAA] - END

        return FramebufferHandle::Create(result);
    }

    BindingLayoutHandle Device::createBindingLayout(const BindingLayoutDesc& desc)
    {
        BindingLayout* result = new BindingLayout();
        result->desc = desc;
        return BindingLayoutHandle::Create(result);
    }

    BindingLayoutHandle Device::createBindlessLayout(const BindlessLayoutDesc& desc)
    {
        BindingLayout* result = new BindingLayout();
        result->bindlessDesc = desc;
        result->isBindless = true;
        return BindingLayoutHandle::Create(result);
    }

    BindingSetHandle Device::createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout)
    {
        BindingSet* result = new BindingSet();
        result->desc = desc;
        result->layout = BindingLayoutHandle(layout);

        for (const auto& item : desc.bindings)
        {
            if (item.resourceHandle)
                result->resources.push_back(RefCountPtr<IResource>(item.resourceHandle));
        }

        return BindingSetHandle::Create(result);
    }

    DescriptorTableHandle Device::createDescriptorTable(IBindingLayout* _layout)
    {
        const BindingLayout* layout = checked_cast<BindingLayout*>(_layout);
        if (!layout || !layout->isBindless)
        {
            m_Context.error("createDescriptorTable: the layout is not a bindless layout");
            return nullptr;
        }

        DescriptorTable* result = new DescriptorTable();
        result->layout = _layout;
        result->capacity = layout->bindlessDesc.maxCapacity;
        result->slots.assign(layout->bindlessDesc.registerSpaces.size(),
            std::vector<BindingSetItem>(result->capacity, BindingSetItem::None(0)));
        return DescriptorTableHandle::Create(result);
    }

    void Device::resizeDescriptorTable(IDescriptorTable* _descriptorTable, uint32_t newSize, bool keepContents)
    {
        DescriptorTable* descriptorTable = checked_cast<DescriptorTable*>(_descriptorTable);
        if (!descriptorTable || newSize > descriptorTable->capacity)
            return;

        for (auto& slots : descriptorTable->slots)
        {
            std::vector<BindingSetItem> resized(newSize, BindingSetItem::None(0));
            if (keepContents)
                std::copy(slots.begin(), slots.begin() + std::min(newSize, uint32_t(slots.size())), resized.begin());
            slots = std::move(resized);
        }
    }

    // Verifies that a permanently pinned resource accepts a descriptor table write of
    // the given item's use, like the Vulkan backend does inside writeDescriptorTable.
    static void verifyPermanentBindingState(const BindingSetItem& item, Context& context)
    {
        switch (item.type)
        {
            case ResourceType::Texture_SRV:
            {
                const auto texture = checked_cast<Texture*>(item.resourceHandle);
                if (texture && texture->permanentState != ResourceStates::Unknown)
                    verifyPermanentResourceState(texture->permanentState, ResourceStates::ShaderResource, true, texture->desc.debugName, context.messageCallback);
            }
            break;

            case ResourceType::Texture_UAV:
            {
                const auto texture = checked_cast<Texture*>(item.resourceHandle);
                if (texture && texture->permanentState != ResourceStates::Unknown)
                    verifyPermanentResourceState(texture->permanentState, ResourceStates::UnorderedAccess, true, texture->desc.debugName, context.messageCallback);
            }
            break;

            case ResourceType::StructuredBuffer_SRV:
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_SRV:
            case ResourceType::RawBuffer_UAV:
            case ResourceType::TypedBuffer_SRV:
            case ResourceType::TypedBuffer_UAV:
            case ResourceType::ConstantBuffer:
            case ResourceType::VolatileConstantBuffer:
            {
                const auto buffer = checked_cast<Buffer*>(item.resourceHandle);
                if (buffer && buffer->permanentState != ResourceStates::Unknown)
                {
                    ResourceStates required = ResourceStates::ShaderResource;
                    if (item.type == ResourceType::StructuredBuffer_UAV || item.type == ResourceType::RawBuffer_UAV || item.type == ResourceType::TypedBuffer_UAV)
                        required = ResourceStates::UnorderedAccess;
                    else if (item.type == ResourceType::ConstantBuffer || item.type == ResourceType::VolatileConstantBuffer)
                        required = ResourceStates::ConstantBuffer;

                    verifyPermanentResourceState(buffer->permanentState, required, false, buffer->desc.debugName, context.messageCallback);
                }
            }
            break;

            default:
                break;
        }
    }

    bool Device::writeDescriptorTable(IDescriptorTable* _descriptorTable, const BindingSetItem& item)
    {
        DescriptorTable* descriptorTable = checked_cast<DescriptorTable*>(_descriptorTable);
        if (!descriptorTable || item.slot >= descriptorTable->capacity)
            return false;

        // The written type picks its register space; a table declares one array per space.
        const BindingLayout* layout = checked_cast<BindingLayout*>(descriptorTable->layout.Get());
        for (size_t space = 0; space < layout->bindlessDesc.registerSpaces.size(); ++space)
        {
            if (layout->bindlessDesc.registerSpaces[space].type != item.type)
                continue;

            if (item.type != ResourceType::None)
                descriptorTable->slots[space][item.slot] = item;

            // A slot written into a permanently pinned resource verifies the pin covers the
            // written use, the check the Vulkan backend makes at the same place.
            if (item.resourceHandle)
                verifyPermanentBindingState(item, getContext());

            return true;
        }

        return false;
    }

    // ------------------------------------------------------------------ Submission

    CommandListHandle Device::createCommandList(const CommandListParameters& params)
    {
        return CommandListHandle::Create(new CommandList(this, params));
    }

    uint64_t Device::submitCommandList(ICommandList* commandList)
    {
        CommandList* list = checked_cast<CommandList*>(commandList);
        if (!list)
            return 0;

        id<MTLCommandBuffer> commandBuffer = list->getCommandBuffer();
        if (!commandBuffer)
            return 0;

        // Permanent states pinned by descriptor table writes apply at submission.
        list->applyPermanentStates();

        uint64_t serial = 0;
        {
            std::lock_guard< std::mutex > lock(m_Mutex);
            serial = ++m_LastSubmittedSerial;
        }

        __block uint64_t completedSerial = serial;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            std::lock_guard< std::mutex > lock(m_CompletionMutex);
            if (completedSerial > m_LastCompletedSerial)
                m_LastCompletedSerial = completedSerial;
            m_CompletionCondition.notify_all();
        }];

        [commandBuffer commit];
        list->releaseCommandBuffer();

        return serial;
    }

    uint64_t Device::executeCommandLists(ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue)
    {
        (void)executionQueue;

        uint64_t serial = 0;
        for (size_t i = 0; i < numCommandLists; ++i)
            serial = submitCommandList(pCommandLists[i]);

        return serial;
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance)
    {
        (void)waitQueue;
        (void)executionQueue;

        if (instance == 0)
            return;

        std::unique_lock<std::mutex> lock(m_CompletionMutex);
        m_CompletionCondition.wait(lock, [this, instance] {
            std::lock_guard<std::mutex> submitLock(m_Mutex);
            return m_LastCompletedSerial >= instance;
        });
    }

    bool Device::waitForIdle()
    {
        std::unique_lock< std::mutex > lock(m_CompletionMutex);
        m_CompletionCondition.wait(lock, [this] {
            std::lock_guard< std::mutex > submitLock(m_Mutex);
            return m_LastCompletedSerial >= m_LastSubmittedSerial;
        });
        return true;
    }

    CommandListLifetimeTrackerHandle Device::createCommandListLifetimeTracker(CommandQueue executionQueue)
    {
        (void)executionQueue;
        return nullptr;
    }

    void Device::runGarbageCollection()
    {
    }

    // ------------------------------------------------------------------ Capabilities

    bool Device::queryFeatureSupport(Feature feature, void* pInfo, size_t infoSize)
    {
        switch (feature)
        {
            case Feature::ConstantBufferRanges:
                // A constant buffer range binds as the buffer with an offset, which Metal
                // supports everywhere.
                return true;

            case Feature::ComputeQueue:
            case Feature::CopyQueue:
                // The Metal command queue executes graphics, compute and copy work alike.
                return true;

            case Feature::VirtualResources:
                // A virtual texture defers its Metal object to bindTextureMemory on a heap.
                return true;

            default:
                return false;
        }
    }

    FormatSupport Device::queryFormatSupport(Format format)
    {
        const MTLPixelFormat pixelFormat = convertFormat(format);
        if (pixelFormat == MTLPixelFormatInvalid)
            return FormatSupport::None;

        const FormatInfo& info = getFormatInfo(format);
        if (info.kind == FormatKind::DepthStencil)
            return FormatSupport::Texture | FormatSupport::DepthStencil | FormatSupport::ShaderLoad | FormatSupport::Buffer;

        FormatSupport support = FormatSupport::Texture | FormatSupport::ShaderLoad | FormatSupport::Buffer;
        support = support | FormatSupport::ShaderUavLoad | FormatSupport::ShaderUavStore;

        // Integer formats cannot sample through a filtering sampler; the rest can.
        if (info.kind != FormatKind::Integer)
            support = support | FormatSupport::ShaderSample;

        support = support | FormatSupport::RenderTarget | FormatSupport::Blendable;

        return support;
    }

    coopvec::DeviceFeatures Device::queryCoopVecFeatures()
    {
        return {};
    }

    coopvec::MatMulFormatSupport Device::queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination)
    {
        (void)combination;
        return {};
    }

    coopvec::TrainingFormatSupport Device::queryCoopVecTrainingFormatSupport(coopvec::DataType componentType)
    {
        (void)componentType;
        return {};
    }

    size_t Device::getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns)
    {
        (void)type;
        (void)layout;
        (void)rows;
        (void)columns;
        return 0;
    }

    // ------------------------------------------------------------------ Ray tracing and meshlets

    bool Device::queryTopLevelAccelStructPrebuildInfo(const rt::AccelStructDesc& desc, uint32_t instanceCount, rt::AccelStructPrebuildInfo& outInfo)
    {
        (void)desc;
        (void)instanceCount;
        (void)outInfo;
        return false;
    }

    rt::PipelineHandle Device::createRayTracingPipeline(const rt::PipelineDesc& desc)
    {
        (void)desc;
        m_Context.error("createRayTracingPipeline: ray tracing is not supported by the Metal backend");
        return nullptr;
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        (void)desc;
        (void)fbinfo;
        m_Context.error("createMeshletPipeline: mesh shaders are not supported by the Metal backend");
        return nullptr;
    }

    MeshletPipelineHandle Device::createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb)
    {
        (void)desc;
        (void)fb;
        return nullptr;
    }

    rt::OpacityMicromapHandle Device::createOpacityMicromap(const rt::OpacityMicromapDesc& desc)
    {
        (void)desc;
        return nullptr;
    }

    rt::AccelStructHandle Device::createAccelStruct(const rt::AccelStructDesc& desc)
    {
        (void)desc;
        return nullptr;
    }

    MemoryRequirements Device::getAccelStructMemoryRequirements(rt::IAccelStruct* as)
    {
        (void)as;
        return {};
    }

    rt::cluster::OperationSizeInfo Device::getClusterOperationSizeInfo(const rt::cluster::OperationParams& params)
    {
        (void)params;
        return {};
    }

    bool Device::bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset)
    {
        (void)as;
        (void)heap;
        (void)offset;
        return false;
    }
}
