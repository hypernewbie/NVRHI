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

        // EntireBuffer is either a zero size or the full-range sentinel.
        if (byteSize == 0 || byteSize == ~0ull)
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

        m_SubmissionEvent = [m_Context.device newSharedEvent];
        m_SubmissionEvent.signaledValue = 0;
    }

    // A callback can release the final device reference while releasing resources.
    // Its completion state then outlives the device, without waiting on itself.
    static thread_local SubmissionCompletion* g_ActiveCompletion = nullptr;

    Device::~Device()
    {
        const auto completion = m_Completion;
        if (g_ActiveCompletion == completion.get())
            return;

        std::unique_lock< std::mutex > lock(completion->mutex);
        completion->condition.wait(lock, [&] { return completion->pendingCallbacks == 0; });
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

    uint64_t Device::getNativeDeviceMemorySize()
    {
        return m_Context.device.recommendedMaxWorkingSetSize;
    }

    uint64_t Device::getNativeAllocatedSize()
    {
        return m_Context.device.currentAllocatedSize;
    }

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
        // Heap hazards default to untracked. Track the whole heap so accesses to
        // overlapping placed resources are ordered across encoders and submissions.
        descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;

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
        case TextureDimension::Texture2DMS:    textureType = MTLTextureType2DMultisample; break;
        case TextureDimension::Texture2DMSArray: textureType = MTLTextureType2DMultisampleArray; break;
        case TextureDimension::Texture2DArray: textureType = MTLTextureType2DArray; break;
        case TextureDimension::Texture3D:      textureType = MTLTextureType3D; break;
        case TextureDimension::TextureCube:    textureType = MTLTextureTypeCube; break;
        case TextureDimension::TextureCubeArray: textureType = MTLTextureTypeCubeArray; break;
        default:
            m_Context.error("createTexture: unsupported texture dimension");
            return nullptr;
        }

        // A sample count above one needs one of the multisample texture types,
        // and the multisample dimensions are those types on their own.
        if (d.sampleCount > 1)
        {
            if (d.dimension == TextureDimension::Texture2D || d.dimension == TextureDimension::Texture2DMS)
                textureType = MTLTextureType2DMultisample;
            else if (d.dimension == TextureDimension::Texture2DArray || d.dimension == TextureDimension::Texture2DMSArray)
                textureType = MTLTextureType2DMultisampleArray;
            else
            {
                m_Context.error("createTexture: a multisample texture must be 2D");
                return nullptr;
            }
        }

        if (textureType == MTLTextureType2DMultisample || textureType == MTLTextureType2DMultisampleArray)
        {
            // Metal allows no mipmaps on a multisample texture, and the sample count
            // has to be one the device supports.
            if (d.sampleCount <= 1)
            {
                m_Context.error("createTexture: a multisample texture needs a sample count above one");
                return nullptr;
            }

            if (d.mipLevels > 1)
            {
                m_Context.error("createTexture: a multisample texture can have only one mip level");
                return nullptr;
            }

            if (textureType == MTLTextureType2DMultisample && d.arraySize > 1)
            {
                m_Context.error("createTexture: a 2D multisample texture can have only one array slice");
                return nullptr;
            }

            if (![m_Context.device supportsTextureSampleCount:d.sampleCount])
            {
                m_Context.error("createTexture: the sample count is not supported by the device");
                return nullptr;
            }
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

        // Create the Metal texture when bindTextureMemory supplies a heap and offset.
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
        result->usage = usage;
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

        // Metal only validates the placement with the debug layer, so the offset and
        // the end of the resource are checked against the heap explicitly. The
        // requested capacity is used, because Metal rounds the heap size up.
        const MTLSizeAndAlign sizeAndAlign = [m_Context.device heapTextureSizeAndAlignWithDescriptor:texture->descriptor];
        if (!sizeAndAlign.align || (offset % sizeAndAlign.align) != 0 || offset > heap->desc.capacity ||
            sizeAndAlign.size > heap->desc.capacity - offset)
        {
            m_Context.error("bindTextureMemory: the offset does not place the texture inside the heap");
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

        // Create the Metal buffer when bindBufferMemory supplies a heap and offset.
        if (d.isVirtual)
        {
            Buffer* result = new Buffer();
            result->desc = d;
            return BufferHandle::Create(result);
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
            if (buf->desc.isVirtual && !buf->buffer)
            {
                const MTLSizeAndAlign sizeAndAlign = [m_Context.device heapBufferSizeAndAlignWithLength:buf->desc.byteSize
                    options:MTLResourceStorageModePrivate];
                requirements.size = sizeAndAlign.size;
                requirements.alignment = sizeAndAlign.align;
            }
            else
            {
                requirements.size = buf->desc.byteSize;
                requirements.alignment = 256;
            }
        }

        return requirements;
    }

    bool Device::bindBufferMemory(IBuffer* _buffer, IHeap* _heap, uint64_t offset)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);
        Heap* heap = checked_cast<Heap*>(_heap);
        if (!buffer || !buffer->desc.isVirtual || buffer->buffer || !heap || !heap->heap)
        {
            m_Context.error("bindBufferMemory: the buffer is not virtual or the heap is invalid");
            return false;
        }

        // The storage mode has to match between the heap and the placed buffer.
        MTLResourceOptions options = MTLResourceStorageModePrivate;
        if (heap->heap.storageMode == MTLStorageModeShared)
            options = MTLResourceStorageModeShared;

        // Metal only validates the placement with the debug layer, so the offset and
        // the end of the resource are checked against the heap explicitly. The
        // requested capacity is used, because Metal rounds the heap size up.
        const MTLSizeAndAlign sizeAndAlign = [m_Context.device heapBufferSizeAndAlignWithLength:buffer->desc.byteSize options:options];
        if (!sizeAndAlign.align || (offset % sizeAndAlign.align) != 0 || offset > heap->desc.capacity ||
            sizeAndAlign.size > heap->desc.capacity - offset)
        {
            m_Context.error("bindBufferMemory: the offset does not place the buffer inside the heap");
            return false;
        }

        id<MTLBuffer> placed = [heap->heap newBufferWithLength:buffer->desc.byteSize options:options offset:offset];
        if (!placed)
        {
            m_Context.error("bindBufferMemory: failed to place the buffer on the heap");
            return false;
        }

        buffer->buffer = placed;
        buffer->isCPUVisible = options == MTLResourceStorageModeShared;
        return true;
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
        descriptor.compareFunction = convertSamplerCompareFunc(d.compareFunc);

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

        std::lock_guard< std::mutex > lock(m_Completion->mutex);
        return m_Completion->lastCompletedSerial >= eventQuery->serial;
    }

    void Device::waitEventQuery(IEventQuery* query)
    {
        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (!eventQuery)
            return;

        std::unique_lock< std::mutex > lock(m_Completion->mutex);
        m_Completion->condition.wait(lock, [this, eventQuery] { return m_Completion->lastCompletedSerial >= eventQuery->serial; });
    }

    void Device::resetEventQuery(IEventQuery* query)
    {
        EventQuery* eventQuery = checked_cast<EventQuery*>(query);
        if (eventQuery)
            eventQuery->serial = 0;
    }

    TimerQueryHandle Device::createTimerQuery()
    {
        std::lock_guard< std::mutex > lock(m_Mutex);

        // Create the counter resources on first use, under the device lock.
        if (!m_TimerQueryPoolInitAttempted)
            initTimerQueryPool();
        if (!m_CounterSampleBuffer || !m_TimerQuerySampleTarget)
        {
            m_Context.error("createTimerQuery: timestamp counter resources are unavailable");
            return nullptr;
        }

        std::lock_guard< std::mutex > allocatorLock(m_TimerQueryAllocator->mutex);
        if (m_TimerQueryAllocator->freeIndices.empty())
        {
            m_Context.error("createTimerQuery: no free timer query slot is left, increase the pool size");
            return nullptr;
        }

        TimerQuery* query = new TimerQuery();
        query->allocator = m_TimerQueryAllocator;
        query->poolIndex = m_TimerQueryAllocator->freeIndices.back();
        m_TimerQueryAllocator->freeIndices.pop_back();
        query->beginSampleIndex = query->poolIndex * 2;
        query->endSampleIndex = query->poolIndex * 2 + 1;
        return TimerQueryHandle::Create(query);
    }

    bool Device::initTimerQueryPool()
    {
        m_TimerQueryPoolInitAttempted = true;

        // The samples go at the boundaries of an empty pass, so the device has to
        // support sampling at a stage boundary.
        if (![m_Context.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            return false;

        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> counterSet in m_Context.device.counterSets)
        {
            if ([counterSet.name isEqualToString:MTLCommonCounterSetTimestamp])
            {
                timestampSet = counterSet;
                break;
            }
        }
        if (!timestampSet)
            return false;

        MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.counterSet = timestampSet;
        // Resolve the shared samples on the CPU after the submission completes.
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = c_TimerQueryPoolSize * 2;

        NSError* error = nil;
        m_CounterSampleBuffer = [m_Context.device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if (!m_CounterSampleBuffer)
            return false;

        MTLTextureDescriptor* targetDescriptor = [[MTLTextureDescriptor alloc] init];
        targetDescriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
        targetDescriptor.width = 1;
        targetDescriptor.height = 1;
        targetDescriptor.usage = MTLTextureUsageRenderTarget;
        targetDescriptor.storageMode = MTLStorageModePrivate;
        m_TimerQuerySampleTarget = [m_Context.device newTextureWithDescriptor:targetDescriptor];
        if (!m_TimerQuerySampleTarget)
            return false;

        m_TimerQueryAllocator->freeIndices.resize(c_TimerQueryPoolSize);
        for (uint32_t i = 0; i < c_TimerQueryPoolSize; ++i)
            m_TimerQueryAllocator->freeIndices[i] = c_TimerQueryPoolSize - 1 - i;

        return true;
    }

    TimerQuery::~TimerQuery()
    {
        if (allocator)
        {
            std::lock_guard< std::mutex > lock(allocator->mutex);
            allocator->freeIndices.push_back(poolIndex);
        }
    }

    bool Device::pollTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);
        if (!query)
            return false;

        std::lock_guard< std::mutex > queryLock(query->mutex);
        if (!query->started || query->serial == 0)
            return false;

        if (query->resolved)
            return true;

        // Resolve only after the submission that sampled the pair completed.
        {
            std::lock_guard< std::mutex > lock(m_Completion->mutex);
            if (m_Completion->lastCompletedSerial < query->serial)
                return false;
        }

        if (query->failed)
        {
            query->time = 0.f;
            query->resolved = true;
            return true;
        }

        NSData* data = [m_CounterSampleBuffer resolveCounterRange:NSMakeRange(query->beginSampleIndex, 2)];
        if (!data || data.length < 2 * sizeof(MTLCounterResultTimestamp))
        {
            query->time = 0.f;
            query->resolved = true;
            return true;
        }
        const MTLCounterResultTimestamp* results = (const MTLCounterResultTimestamp*)data.bytes;

        // Samples that fail to resolve report no time.
        if (results[0].timestamp == MTLCounterErrorValue || results[1].timestamp == MTLCounterErrorValue
            || results[0].timestamp == 0 || results[1].timestamp == 0)
        {
            query->time = 0.f;
            query->resolved = true;
            return true;
        }

        // The timestamps count nanoseconds. The stage boundary samples are not fully
        // isolated from other encoders, so a negative interval reads as zero.
        const uint64_t interval = results[1].timestamp > results[0].timestamp
            ? results[1].timestamp - results[0].timestamp
            : 0;
        query->time = float(double(interval) * 1e-9);
        query->resolved = true;
        return true;
    }

    float Device::getTimerQueryTime(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);
        if (!query)
            return 0.f;

        uint64_t serial;
        {
            std::lock_guard< std::mutex > lock(query->mutex);
            if (!query->started || query->serial == 0)
                return 0.f;
            serial = query->serial;
        }

        {
            std::unique_lock< std::mutex > lock(m_Completion->mutex);
            m_Completion->condition.wait(lock, [this, serial] { return m_Completion->lastCompletedSerial >= serial; });
        }
        if (!pollTimerQuery(query))
            return 0.f;

        std::lock_guard< std::mutex > lock(query->mutex);
        query->started = false;
        return query->time;
    }

    void Device::resetTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);
        if (query)
        {
            std::lock_guard< std::mutex > lock(query->mutex);
            query->serial = 0;
            query->started = false;
            query->resolved = false;
            query->failed = false;
            query->time = 0.f;
        }
    }

    // ------------------------------------------------------------------ Pipelines

    FramebufferHandle Device::createFramebuffer(const FramebufferDesc& desc)
    {
        Framebuffer* result = new Framebuffer();
        result->desc = desc;
        result->framebufferInfo = FramebufferInfoEx(desc);

        for (const auto& attachment : desc.colorAttachments)
            result->resources.push_back(attachment.texture);
        if (desc.depthAttachment.valid())
            result->resources.push_back(desc.depthAttachment.texture);
        if (desc.shadingRateAttachment.valid())
            result->resources.push_back(desc.shadingRateAttachment.texture);

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
        if (desc.updateAfterBind)
        {
            m_Context.error("createBindlessLayout: update-after-bind is not supported by the Metal backend");
            return nullptr;
        }
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
        if (!descriptorTable)
            return;

        const BindingLayout* layout = checked_cast<BindingLayout*>(descriptorTable->layout.Get());
        if (!layout || newSize > layout->bindlessDesc.maxCapacity)
            return;

        for (auto& slots : descriptorTable->slots)
        {
            std::vector<BindingSetItem> resized(newSize, BindingSetItem::None(0));
            if (keepContents)
                std::copy(slots.begin(), slots.begin() + std::min(newSize, uint32_t(slots.size())), resized.begin());
            slots = std::move(resized);
        }

        descriptorTable->capacity = newSize;
    }

    // Check the descriptor's required state against the resource's permanent state.
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

        // A null write leaves the slot as it is, like the Vulkan backend does.
        if (item.type == ResourceType::None)
            return true;

        // The written type picks its register spaces; a table declares one array per
        // space. Every space of the type is written, like the Vulkan backend does,
        // because two spaces of the shader can share one table.
        const BindingLayout* layout = checked_cast<BindingLayout*>(descriptorTable->layout.Get());
        bool matched = false;
        for (size_t space = 0; space < layout->bindlessDesc.registerSpaces.size(); ++space)
        {
            if (layout->bindlessDesc.registerSpaces[space].type != item.type)
                continue;

            descriptorTable->slots[space][item.slot] = item;

            // Check the descriptor type against the resource's permanent state.
            if (item.resourceHandle)
                verifyPermanentBindingState(item, getContext());

            matched = true;
        }

        return matched;
    }

    // ------------------------------------------------------------------ Submission

    CommandListHandle Device::createCommandList(const CommandListParameters& params)
    {
        return CommandListHandle::Create(new CommandList(this, params));
    }

    id<MTLBuffer> Device::getNullBuffer()
    {
        if (!m_NullBuffer)
            m_NullBuffer = [m_Context.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        return m_NullBuffer;
    }

    id<MTLTexture> Device::getNullTexture(TextureDimension dimension, Format format)
    {
        const Format viewFormat = format == Format::UNKNOWN ? Format::RGBA8_UNORM : format;
        const auto key = std::make_pair(dimension, viewFormat);
        const auto cached = m_NullTextures.find(key);
        if (cached != m_NullTextures.end())
            return cached->second;

        // The null texture has to match the texture type the shader declares, so
        // one null texture exists per dimension an array argument can take.
        MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
        switch (dimension)
        {
        case TextureDimension::Texture1D:
            descriptor.textureType = MTLTextureType1D;
            descriptor.width = 1;
            break;
        case TextureDimension::Texture1DArray:
            descriptor.textureType = MTLTextureType1DArray;
            descriptor.width = 1;
            descriptor.arrayLength = 1;
            break;
        case TextureDimension::Texture3D:
            descriptor.textureType = MTLTextureType3D;
            descriptor.width = 1;
            descriptor.height = 1;
            descriptor.depth = 1;
            break;
        case TextureDimension::TextureCube:
            descriptor.textureType = MTLTextureTypeCube;
            descriptor.width = 1;
            descriptor.height = 1;
            break;
        case TextureDimension::TextureCubeArray:
            descriptor.textureType = MTLTextureTypeCubeArray;
            descriptor.width = 1;
            descriptor.height = 1;
            descriptor.arrayLength = 1;
            break;
        case TextureDimension::Texture2DArray:
            descriptor.textureType = MTLTextureType2DArray;
            descriptor.width = 1;
            descriptor.height = 1;
            descriptor.arrayLength = 1;
            break;
        case TextureDimension::Texture2D:
        case TextureDimension::Unknown:
        default:
            descriptor.textureType = MTLTextureType2D;
            descriptor.width = 1;
            descriptor.height = 1;
            break;
        }

        descriptor.pixelFormat = convertFormat(viewFormat);
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        descriptor.storageMode = MTLStorageModeShared;
        id<MTLTexture> texture = [m_Context.device newTextureWithDescriptor:descriptor];

        m_NullTextures.emplace(key, texture);
        return texture;
    }

    id<MTLTexture> Device::getNullTextureBuffer(Format format)
    {
        const Format viewFormat = format == Format::UNKNOWN ? Format::R32_UINT : format;
        const auto cached = m_NullTextureBuffers.find(viewFormat);
        if (cached != m_NullTextureBuffers.end())
            return cached->second;

        const MTLPixelFormat pixelFormat = convertFormat(viewFormat);
        if (pixelFormat == MTLPixelFormatInvalid)
            return nil;

        // A texture buffer holds the typed buffer views of NVRHI, so the null
        // resource of a typed buffer array is a texture buffer as well.
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pixelFormat
            width:1 resourceOptions:MTLResourceStorageModeShared usage:MTLTextureUsageShaderRead];
        id<MTLTexture> texture = [m_Context.device newTextureWithDescriptor:descriptor];

        m_NullTextureBuffers.emplace(viewFormat, texture);
        return texture;
    }

    id<MTLSamplerState> Device::getNullSampler()
    {
        if (!m_NullSampler)
        {
            MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
            descriptor.minFilter = MTLSamplerMinMagFilterNearest;
            descriptor.magFilter = MTLSamplerMinMagFilterNearest;
            descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
            descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            m_NullSampler = [m_Context.device newSamplerStateWithDescriptor:descriptor];
        }
        return m_NullSampler;
    }

    id<MTLComputePipelineState> Device::getIndirectCountPipeline()
    {
        if (m_IndirectCountPipeline)
            return m_IndirectCountPipeline;

        // Metal has no counted indirect draw, so the arguments of the draws that the
        // count allows are copied into a scratch buffer. The commands that the count
        // excludes are zeroed, which makes them a no-op.
        static const char* source =
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void expand_draw_args(\n"
            "    device const uint* src_args [[buffer(0)]],\n"
            "    device const uint* count_args [[buffer(1)]],\n"
            "    device uint* dst_args [[buffer(2)]],\n"
            "    device const uint* params [[buffer(3)]],\n"
            "    uint gid [[thread_position_in_grid]])\n"
            "{\n"
            "    if (gid >= params[2])\n"
            "        return;\n"
            "    const uint count = min(count_args[params[1] >> 2], params[2]);\n"
            "    const uint base = (params[0] >> 2) + gid * 5;\n"
            "    for (uint i = 0; i < 5; ++i)\n"
            "        dst_args[gid * 5 + i] = (gid < count) ? src_args[base + i] : 0;\n"
            "}\n";

        NSError* nsError = nil;
        id<MTLLibrary> library = [m_Context.device newLibraryWithSource:@(source) options:nil error:&nsError];
        if (!library)
        {
            m_Context.error("getIndirectCountPipeline: failed to compile the helper kernel: "
                + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nil;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"expand_draw_args"];
        if (!function)
        {
            m_Context.error("getIndirectCountPipeline: the helper kernel is missing");
            return nil;
        }

        m_IndirectCountPipeline = [m_Context.device newComputePipelineStateWithFunction:function error:&nsError];
        if (!m_IndirectCountPipeline)
        {
            m_Context.error("getIndirectCountPipeline: failed to create the helper pipeline: "
                + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
        }

        return m_IndirectCountPipeline;
    }

#ifdef NVRHI_METAL_TEST_HOOKS
    void Device::setSubmissionHooks(const test::SubmissionHooks& hooks)
    {
        std::lock_guard< std::mutex > lock(m_Mutex);
        m_TestHooks = hooks;
    }

    void test::setSubmissionHooks(IDevice* device, const SubmissionHooks& hooks)
    {
        static_cast<Device*>(device)->setSubmissionHooks(hooks);
    }

    Object test::getSubmissionEvent(IDevice* device)
    {
        return static_cast<Device*>(device)->getSubmissionEvent();
    }
#endif

    uint64_t Device::submitCommandListLocked(ICommandList* commandList)
    {
        CommandList* list = checked_cast<CommandList*>(commandList);
        if (!list)
        {
            m_Context.error("submitCommandList: the command list is null");
            return 0;
        }

        id<MTLCommandBuffer> commandBuffer = list->getCommandBuffer();
        if (!commandBuffer)
        {
            m_Context.error("submitCommandList: the command list has no Metal command buffer");
            return 0;
        }
        if (!m_SubmissionEvent)
        {
            m_Context.error("submitCommandList: no Metal submission event is available");
            return 0;
        }

        // Permanent states pinned by descriptor table writes apply at submission.
        list->applyPermanentStates();

        const uint64_t serial = ++m_LastSubmittedSerial;

        // Hold each query until the GPU has finished using its sample indices.
        __block std::vector<RefCountPtr<TimerQuery>> sampledTimerQueries = list->takeSampledTimerQueries();
        for (const auto& query : sampledTimerQueries)
        {
            std::lock_guard< std::mutex > lock(query->mutex);
            query->serial = serial;
        }
        const auto completion = m_Completion;
        id<MTLSharedEvent> submissionEvent = m_SubmissionEvent;
        IMessageCallback* callback = m_Context.messageCallback;
#ifdef NVRHI_METAL_TEST_HOOKS
        const bool injectedFailure = m_TestHooks.failNextCompletion;
        m_TestHooks.failNextCompletion = false;
#else
        const bool injectedFailure = false;
#endif

        // The block owns the resources and completion state independently of Device.
        __block std::vector<RefCountPtr<IResource>> referencedResources = list->takeReferencedResources();
        {
            std::lock_guard< std::mutex > lock(completion->mutex);
            ++completion->pendingCallbacks;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            SubmissionCompletion* previous = g_ActiveCompletion;
            g_ActiveCompletion = completion.get();
            const bool failed = injectedFailure || buffer.status != MTLCommandBufferStatusCompleted;
            NSString* text = injectedFailure ? @"injected completion failure" : (failed ? buffer.error.localizedDescription : nil);
            std::string error = failed ? std::string(text ? text.UTF8String : "unknown error") : std::string();

            // A failed command buffer may not reach its encoded event signal.
            // Release dependent GPU waits and report the failure to host callers.
            if (failed && submissionEvent.signaledValue < serial)
                submissionEvent.signaledValue = serial;
            if (failed)
            {
                for (const auto& query : sampledTimerQueries)
                {
                    std::lock_guard< std::mutex > lock(query->mutex);
                    query->failed = true;
                }
            }
            sampledTimerQueries.clear();
            {
                std::lock_guard< std::mutex > lock(completion->mutex);
                if (failed && (completion->firstFailedSerial == 0 || serial < completion->firstFailedSerial))
                    completion->firstFailedSerial = serial;
                completion->finishedOutOfOrder.insert(serial);
                while (completion->finishedOutOfOrder.erase(completion->lastCompletedSerial + 1))
                    ++completion->lastCompletedSerial;
                completion->condition.notify_all();
            }

            if (failed && callback)
            {
                const std::string message = "the command buffer completed with an error: " + error;
                callback->message(MessageSeverity::Error, message.c_str());
            }
            referencedResources.clear();
            {
                std::lock_guard< std::mutex > lock(completion->mutex);
                --completion->pendingCallbacks;
                completion->condition.notify_all();
            }
            g_ActiveCompletion = previous;
        }];

        // The submission signals the event when its work is done, which is what
        // the queue waits of queueWaitForCommandList wait for.
        if (!injectedFailure)
            [commandBuffer encodeSignalEvent:m_SubmissionEvent value:serial];

#ifdef NVRHI_METAL_TEST_HOOKS
        if (m_TestHooks.beforeCommit)
        {
            auto hook = m_TestHooks.beforeCommit;
            m_TestHooks.beforeCommit = nullptr;
            hook(m_TestHooks.context, serial);
        }
#endif
        [commandBuffer commit];
        list->releaseCommandBuffer();

        return serial;
    }

    uint64_t Device::executeCommandLists(ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue)
    {
        // The lock orders serial assignment, queue waits and command-buffer commits.
        // The completion callbacks use a separate lock and never acquire this one.
        std::lock_guard< std::mutex > lock(m_Mutex);

        // The queue performs its pending waits on the GPU before this submission.
        if (!commitQueueWaits(executionQueue))
            return 0;

        uint64_t serial = 0;
        for (size_t i = 0; i < numCommandLists; ++i)
        {
            serial = submitCommandListLocked(pCommandLists[i]);
            if (serial == 0)
                return 0;
        }

        return serial;
    }

    bool Device::commitQueueWaits(CommandQueue executionQueue)
    {
        uint64_t instance = 0;
        {
            std::lock_guard< std::mutex > lock(m_QueueWaitMutex);
            instance = m_PendingQueueWaits[uint32_t(executionQueue)];
            m_PendingQueueWaits[uint32_t(executionQueue)] = 0;
        }

        if (instance == 0)
            return true;

#ifdef NVRHI_METAL_TEST_HOOKS
        const bool injectedFailure = m_TestHooks.failNextQueueWaitAllocation;
        m_TestHooks.failNextQueueWaitAllocation = false;
#else
        const bool injectedFailure = false;
#endif
        id<MTLCommandBuffer> commandBuffer = m_SubmissionEvent && !injectedFailure ? [m_Context.queue commandBuffer] : nil;
        if (!commandBuffer)
        {
            // The consumer would run without its dependency, so the wait stays
            // pending and the caller hears about the failure.
            m_Context.error("commitQueueWaits: no command buffer is available for the pending queue wait");
            std::lock_guard< std::mutex > lock(m_QueueWaitMutex);
            m_PendingQueueWaits[uint32_t(executionQueue)] = std::max(m_PendingQueueWaits[uint32_t(executionQueue)], instance);
            return false;
        }

        [commandBuffer encodeWaitForEvent:m_SubmissionEvent value:instance];
#ifdef NVRHI_METAL_TEST_HOOKS
        if (m_TestHooks.queueWaitCommitted)
            m_TestHooks.queueWaitCommitted(m_TestHooks.context, instance);
#endif
        [commandBuffer commit];
        return true;
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance)
    {
        (void)executionQueue;

        if (instance == 0)
            return;

        // The wait queue performs the wait on the GPU before its next submission,
        // so this call never blocks the calling thread.
        std::lock_guard< std::mutex > lock(m_QueueWaitMutex);
        m_PendingQueueWaits[uint32_t(waitQueue)] = std::max(m_PendingQueueWaits[uint32_t(waitQueue)], instance);
    }

    bool Device::waitForIdle()
    {
        // The target is taken under the submission lock alone, because the completion
        // handler needs the completion lock and must not wait for a submitter.
        uint64_t target = 0;
        {
            std::lock_guard< std::mutex > lock(m_Mutex);
            target = m_LastSubmittedSerial;
        }

        std::unique_lock< std::mutex > lock(m_Completion->mutex);
        m_Completion->condition.wait(lock, [this, target] {
            return m_Completion->lastCompletedSerial >= target;
        });
        return m_Completion->firstFailedSerial == 0 || m_Completion->firstFailedSerial > target;
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

            case Feature::ShaderSpecializations:
                // SPIRV-Cross turns the SPIR-V specialisation constants into Metal
                // function constants, which the entry function picks up.
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
        if (info.blockSize > 1)
        {
            // BC textures are sampled, but Metal does not allow them as render
            // targets, shader-write textures, or texture buffers.
            if (@available(macOS 11.0, *))
            {
                if (!m_Context.device.supportsBCTextureCompression)
                    return FormatSupport::None;
            }
            return FormatSupport::Texture | FormatSupport::ShaderLoad | FormatSupport::ShaderSample;
        }

        if (info.kind == FormatKind::DepthStencil)
            return FormatSupport::Texture | FormatSupport::DepthStencil | FormatSupport::ShaderLoad;

        FormatSupport support = FormatSupport::Texture | FormatSupport::ShaderLoad | FormatSupport::Buffer
            | FormatSupport::ShaderUavLoad | FormatSupport::ShaderUavStore | FormatSupport::RenderTarget;

        // Integer attachments support rendering, but not filtering or blending.
        if (info.kind != FormatKind::Integer)
        {
            support = support | FormatSupport::Blendable;
            bool filterable = true;
            if (format == Format::R32_FLOAT || format == Format::RG32_FLOAT || format == Format::RGBA32_FLOAT)
            {
                if (@available(macOS 11.0, *))
                    filterable = m_Context.device.supports32BitFloatFiltering;
            }
            if (filterable)
                support = support | FormatSupport::ShaderSample;
        }

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
