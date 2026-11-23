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

#pragma once

#include <Metal/Metal.h>

#include <nvrhi/metal.h>
#include <nvrhi/utils.h>

#include "../common/state-tracking.h"
#include "metal-spirv.h"
#ifdef NVRHI_METAL_TEST_HOOKS
#include "metal-test-hooks.h"
#endif

#include <atomic>
#include <array>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace nvrhi::metal
{
    class Device;
    class CommandList;
    class Buffer;
    class Texture;
    class Shader;
    class Sampler;
    class Framebuffer;
    class ComputePipeline;
    class GraphicsPipeline;
    class BindingLayout;
    class BindingSet;
    class InputLayout;

    // Vertex attributes and shader buffers share Metal's buffer index namespace.
    // Reserve the upper indices for vertex buffers.
    constexpr uint32_t c_VertexBufferBase = 16;

    // The number of timer queries the counter sample buffer holds sample pairs for.
    constexpr uint32_t c_TimerQueryPoolSize = 256;

    struct Context
    {
        id<MTLDevice> device = nil;
        id<MTLCommandQueue> queue = nil;
        IMessageCallback* messageCallback = nullptr;

        // The Metal language version the shaders are compiled against.
        uint32_t mslVersionMajor = 3;
        uint32_t mslVersionMinor = 0;

        void error(const std::string& message) const;
        void warning(const std::string& message) const;
        void info(const std::string& message) const;
    };

    // One resource as the Metal pipeline sees it: the SPIR-V descriptor set and binding
    // on the NVRHI side, the buffer/texture/sampler index on the Metal side.
    struct BoundResource
    {
        ResourceKind kind = ResourceKind::Buffer;
        uint32_t index = 0;
    };

    class Buffer : public RefCounter<IBuffer>, public BufferStateExtension
    {
    public:
        Buffer()
            : BufferStateExtension(desc)
        { }

        BufferDesc desc;
        id<MTLBuffer> buffer = nil;
        void* mappedMemory = nullptr;
        bool isCPUVisible = false;

        // Typed buffer views, keyed by format, byte offset and size.
        std::map<std::tuple<Format, uint64_t, uint64_t>, id<MTLTexture>> typedViews;

        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

    // Returns a texture-buffer view of the buffer range for a typed buffer binding,
    // creating and caching it on the buffer on first use.
    id<MTLTexture> getTypedBufferView(Context& context, Buffer* buffer, Format format, uint64_t byteOffset, uint64_t byteSize);

    class StagingTexture : public RefCounter<IStagingTexture>
    {
    public:
        TextureDesc desc;
        id<MTLBuffer> buffer = nil;
        void* mappedMemory = nullptr;
        uint64_t bytesPerRow = 0;
        std::vector<uint64_t> mipOffsets;
        std::vector<uint64_t> mipRowPitches;

        const TextureDesc& getDesc() const override { return desc; }
    };

    class Texture : public RefCounter<ITexture>, public TextureStateExtension
    {
    public:
        Texture()
            : TextureStateExtension(desc)
        { }

        TextureDesc desc;
        id<MTLTexture> texture = nil;
        bool isCPUVisible = false;

        // The usage the texture was created with. A colour clear needs the render target usage.
        MTLTextureUsage usage = 0;

        // A virtual texture keeps its descriptor until bindTextureMemory places it on a heap.
        MTLTextureDescriptor* descriptor = nil;

        // Subresource views keyed by range, dimension, format and channel mapping.
        struct ViewKey
        {
            uint32_t baseMipLevel = 0;
            uint32_t numMipLevels = 1;
            uint32_t baseArraySlice = 0;
            uint32_t numArraySlices = 1;
            TextureDimension dimension = TextureDimension::Unknown;
            Format format = Format::UNKNOWN;
            uint16_t componentMapping = 0;

            bool operator<(const ViewKey& other) const
            {
                return std::tie(baseMipLevel, numMipLevels, baseArraySlice, numArraySlices, dimension, format, componentMapping)
                    < std::tie(other.baseMipLevel, other.numMipLevels, other.baseArraySlice, other.numArraySlices, other.dimension, other.format, other.componentMapping);
            }
        };
        std::map<ViewKey, id<MTLTexture>> views;

        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool isReadOnlyDSV = false,
            std::optional<ComponentMapping> overrideComponentMapping = std::nullopt) override;

        // Returns a view of the subresources as the given dimension, format and channel
        // mapping, creating and caching it on first use. The entire texture with its own format
        // and an identity mapping returns the base texture.
        id<MTLTexture> getSubresourceView(const TextureSubresourceSet& subresources, TextureDimension dimension, Format format, ComponentMapping componentMapping);
    };

    class Sampler : public RefCounter<ISampler>
    {
    public:
        SamplerDesc desc;
        id<MTLSamplerState> sampler = nil;
        MTLSamplerDescriptor* descriptor = nil;
        id<MTLSamplerState> argumentSampler = nil;
        std::mutex mutex;

        const SamplerDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        id<MTLSamplerState> getArgumentSampler(Context& context);
    };

    class Heap : public RefCounter<IHeap>
    {
    public:
        HeapDesc desc;
        id<MTLHeap> heap = nil;

        const HeapDesc& getDesc() override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class Shader : public RefCounter<IShader>
    {
    public:
        ShaderDesc desc;
        id<MTLLibrary> library = nil;
        id<MTLFunction> function = nil;
        std::string source;
        TranslatedShader translated;
        // A specialised shader holds the base shader, which owns the library.
        ShaderHandle baseShader;

        // Shader variants use the bindless layout capacities of their pipeline.
        struct BindlessTranslation
        {
            TranslatedShader translated;
            id<MTLFunction> function = nil;
        };
        std::vector<uint32_t> spirv;
        mutable std::mutex bindlessTranslationsMutex;
        mutable std::map<std::vector<uint32_t>, BindlessTranslation> bindlessTranslations;

        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
        Object getNativeObject(ObjectType objectType) override;

        // Compile and cache shaders by bindless descriptor set capacity.
        const BindlessTranslation* getTranslationForCapacities(Context& context, const std::vector<uint32_t>& capacities) const;
    };

    class InputLayout : public RefCounter<IInputLayout>
    {
    public:
        std::vector<VertexAttributeDesc> attributes;

        uint32_t getNumAttributes() const override { return uint32_t(attributes.size()); }
        const VertexAttributeDesc* getAttributeDesc(uint32_t index) const override;
    };

    class Framebuffer : public RefCounter<IFramebuffer>
    {
    public:
        FramebufferDesc desc;
        FramebufferInfoEx framebufferInfo;
        std::vector<ResourceHandle> resources; // Own the attachments referenced by desc.

        const FramebufferDesc& getDesc() const override { return desc; }
        const FramebufferInfoEx& getFramebufferInfo() const override { return framebufferInfo; }
    };

    class BindingLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindingLayoutDesc desc;
        BindlessLayoutDesc bindlessDesc;
        bool isBindless = false;

        const BindingLayoutDesc* getDesc() const override { return isBindless ? nullptr : &desc; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return isBindless ? &bindlessDesc : nullptr; }
    };

    class BindingSet : public RefCounter<IBindingSet>
    {
    public:
        BindingSetDesc desc;
        BindingLayoutHandle layout;

        std::vector<RefCountPtr<IResource>> resources;

        const BindingSetDesc* getDesc() const override { return &desc; }
        IBindingLayout* getLayout() const override { return layout; }
    };

    // A bindless descriptor table keeps the written items until they are bound.
    class DescriptorTable : public RefCounter<IDescriptorTable>
    {
    public:
        struct ArgumentBuffer
        {
            struct Function
            {
                __weak id<MTLFunction> function = nil;
            };
            std::map<std::pair<uintptr_t, uint32_t>, Function> functions;
            id<MTLArgumentEncoder> encoder = nil;
            id<MTLBuffer> buffer = nil;
            std::array<uint64_t, 18> textureLayout = {};
            bool hasTextureLayout = false;
            std::vector<IResource*> referenced;
            std::map<void*, MTLResourceUsage> resources;
        };

        BindingLayoutHandle layout;
        uint32_t capacity = 0;
        // Each register space has its own array of slots.
        std::vector<std::vector<BindingSetItem>> slots;
        std::vector<ArgumentBuffer> argumentBuffers;
        mutable std::mutex mutex;

        void invalidateArgumentBuffers()
        {
            for (auto& argumentBuffer : argumentBuffers)
            {
                argumentBuffer.buffer = nil;
                argumentBuffer.referenced.clear();
                argumentBuffer.resources.clear();
            }
        }

        const BindingSetDesc* getDesc() const override { return nullptr; }
        IBindingLayout* getLayout() const override { return layout; }
        uint32_t getCapacity() const override
        {
            std::lock_guard< std::mutex > lock(mutex);
            return capacity;
        }
        uint32_t getFirstDescriptorIndexInHeap() const override { return 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

    // The resources a pipeline reads, resolved from the shaders' translated binding plans.
    class PipelineBindingPlan
    {
    public:
        struct Entry
        {
            ShaderType stage = ShaderType::None;
            uint32_t bindingIndex = 0;
            uint32_t slot = 0;
            BoundResource bound;
        };

        std::vector<Entry> entries;
        std::vector<uint32_t> descriptorSets;

        struct ArgumentBuffer
        {
            ShaderType stage = ShaderType::None;
            uint32_t descriptorSet = 0;
            uint32_t index = 0;
            id<MTLFunction> function = nil;
        };
        std::vector<ArgumentBuffer> argumentBuffers;

        // The shader resources backed by bindless descriptor tables.
        std::vector<ResourceBinding> bindlessArrays;

    };

    class ComputePipeline : public RefCounter<IComputePipeline>
    {
    public:
        ComputePipelineDesc desc;
        id<MTLComputePipelineState> pipeline = nil;
        PipelineBindingPlan plan;
        uint32_t threadGroupSize[3] = { 1, 1, 1 };
        uint32_t pushConstantBufferIndex = UINT32_MAX;
        uint32_t pushConstantSize = 0;

        const ComputePipelineDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class GraphicsPipeline : public RefCounter<IGraphicsPipeline>
    {
    public:
        GraphicsPipelineDesc desc;
        FramebufferInfo framebufferInfo;
        id<MTLRenderPipelineState> pipeline = nil;
        id<MTLDepthStencilState> depthStencilState = nil;
        MTLPrimitiveType primitiveType = MTLPrimitiveTypeTriangle;
        PipelineBindingPlan plan;
        uint32_t pushConstantBufferIndex = UINT32_MAX;
        uint32_t pushConstantSize = 0;
        bool vertexPushConstants = false;
        bool pixelPushConstants = false;

        const GraphicsPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class EventQuery : public RefCounter<IEventQuery>
    {
    public:
        uint64_t serial = 0;
        bool signalled = false;
    };

    struct TimerQueryAllocator
    {
        std::mutex mutex;
        std::vector<uint32_t> freeIndices;
    };

    // A timer query owns one pair of samples in the device's counter sample buffer.
    class TimerQuery : public RefCounter<ITimerQuery>
    {
    public:
        std::shared_ptr<TimerQueryAllocator> allocator;
        uint32_t poolIndex = 0;
        uint32_t beginSampleIndex = 0;
        uint32_t endSampleIndex = 0;
        std::mutex mutex;
        // Zero means the query has not been submitted.
        uint64_t serial = 0;
        float time = 0.f;
        bool started = false;
        bool resolved = false;
        bool failed = false;

        ~TimerQuery() override;
    };

    class CommandList : public RefCounter<ICommandList>
    {
    public:
        CommandList(Device* device, const CommandListParameters& params);
        ~CommandList() override;

        void open() override;
        void close() override;
        void clearState() override;
        void clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor) override;
        void clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor) override;
        void clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil) override;
        void clearBufferUInt(IBuffer* b, uint32_t clearValue) override;
        void copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice) override;
        void writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch = 0) override;
        void resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources) override;
        void writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes = 0) override;
        void copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes) override;
        void clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture) override;
        void decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, nvrhi::Format format) override;
        void setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits) override;
        void setPushConstants(const void* data, size_t byteSize) override;
        void setGraphicsState(const GraphicsState& state) override;
        void draw(const DrawArguments& args) override;
        void drawIndexed(const DrawArguments& args) override;
        void drawIndirect(uint32_t offsetBytes, uint32_t drawCount = 1) override;
        void drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount = 1) override;
        void drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;
        void setComputeState(const ComputeState& state) override;
        void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchIndirect(uint32_t offsetBytes) override;
        void setMeshletState(const MeshletState& state) override;
        void dispatchMesh(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchMeshIndirect(uint32_t offsetBytes, uint32_t maxDrawCount) override;
        void dispatchMeshIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount) override;
        void setRayTracingState(const rt::State& state) override;
        void dispatchRays(const rt::DispatchRaysArguments& args) override;
        void buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc) override;
        void buildBottomLevelAccelStruct(rt::IAccelStruct* as, const rt::GeometryDesc* pGeometries, size_t numGeometries, rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void compactBottomLevelAccelStructs() override;
        void copyRaytracingAccelerationStructure(rt::IAccelStruct* destination, rt::IAccelStruct* source) override;
        void buildTopLevelAccelStruct(rt::IAccelStruct* as, const rt::InstanceDesc* pInstances, size_t numInstances, rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) override;
        void buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as, nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset, size_t numInstances, rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs) override;
        void beginTimerQuery(ITimerQuery* query) override;
        void endTimerQuery(ITimerQuery* query) override;
        void beginMarker(const char* name) override;
        void endMarker() override;
        void setEnableAutomaticBarriers(bool enable) override;
        void setResourceStatesForBindingSet(IBindingSet* bindingSet) override;
        void setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers) override;
        void setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers) override;
        void beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void setBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits) override;
        void setPermanentTextureState(ITexture* texture, ResourceStates stateBits) override;
        void setPermanentBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void applyPermanentStates();
        void verifyPermanentState(TextureStateExtension* texture, ResourceStates state);
        void verifyPermanentState(BufferStateExtension* buffer, ResourceStates state);
        void commitBarriers() override;
        ResourceStates getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel) override;
        ResourceStates getBufferState(IBuffer* buffer) override;
        IDevice* getDevice() override;
        const CommandListParameters& getDesc() override;
        CommandListMemoryUsage getInternalMemoryUsage() const override;
        Object getNativeObject(ObjectType objectType) override;

        id<MTLCommandBuffer> getCommandBuffer() const { return m_CommandBuffer; }
        void releaseCommandBuffer() { m_CommandBuffer = nil; }

        // Hands the referenced resources over to the caller, which keeps them alive
        // until the GPU is done with the command buffer.
        std::vector<RefCountPtr<IResource>> takeReferencedResources()
        {
            return std::move( m_ReferencedResources );
        }

        // Hands the timer queries this command list sampled over to the caller,
        // which stamps its submission serial on them.
        std::vector<RefCountPtr<TimerQuery>> takeSampledTimerQueries()
        {
            return std::move( m_SampledTimerQueries );
        }

    private:
        void endEncoder();
        id<MTLComputeCommandEncoder> getComputeEncoder();
        id<MTLBlitCommandEncoder> getBlitEncoder();
        id<MTLRenderCommandEncoder> getRenderEncoder(Framebuffer* framebuffer);
        bool reopenRenderEncoder();
        void sampleTimestamp(TimerQuery* query, bool atPassEnd);

        void bindComputeResources(const ComputeState& state);
        void bindGraphicsResources(const GraphicsState& state);
        void bindResource(const BoundResource& bound, const nvrhi::BindingSetItem& item);
        void bindGraphicsResource(const BoundResource& bound, const nvrhi::BindingSetItem& item, bool vertexStage);
        void bindDescriptorTable(DescriptorTable* table, const PipelineBindingPlan& plan, uint32_t bindingIndex);
        void bindTableArgumentBuffer(DescriptorTable* table, const PipelineBindingPlan& plan, const PipelineBindingPlan::ArgumentBuffer& argumentBuffer, Context& context);
        bool encodeTableArgumentBuffer(DescriptorTable* table, const PipelineBindingPlan& plan, const PipelineBindingPlan::ArgumentBuffer& argumentBuffer, id<MTLArgumentEncoder> encoder, std::vector<IResource*>& referenced, std::map<void*, MTLResourceUsage>& resources, Context& context);
        void bindTableArray(const std::vector<nvrhi::BindingSetItem>& slots, const ResourceBinding& array);
        void bindNullResource(ResourceKind kind, uint32_t index, bool bufferArray, Format format, TextureDimension dimension,
            ShaderType stage = ShaderType::None);
        void bindPushConstants(uint32_t bufferIndex, uint32_t size, ShaderType stage);
        id<MTLBuffer> expandIndirectDrawArgs(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount);
        void clearTexture(Texture* texture, const TextureSubresourceSet& subresources, const Color& color, uint32_t intValue, bool integer);

        Device* m_Device;
        CommandListParameters m_Params;
        id<MTLCommandBuffer> m_CommandBuffer = nil;
        id<MTLComputeCommandEncoder> m_ComputeEncoder = nil;
        id<MTLBlitCommandEncoder> m_BlitEncoder = nil;
        id<MTLRenderCommandEncoder> m_RenderEncoder = nil;
        Framebuffer* m_RenderEncoderFramebuffer = nullptr;

        ComputePipeline* m_CurrentComputePipeline = nullptr;
        GraphicsPipeline* m_CurrentGraphicsPipeline = nullptr;
        nvrhi::IndexBufferBinding m_IndexBuffer;
        Buffer* m_IndirectBuffer = nullptr;

        // The current states are kept so a transfer or clear that ends an encoder can
        // re-apply the state when the next dispatch or draw opens a new one.
        ComputeState m_CurrentComputeState;
        GraphicsState m_CurrentGraphicsState;

        // A new encoder starts without any state, so the whole graphics state is
        // applied again. The flag is cleared when the encoder ends.
        bool m_RenderEncoderStateValid = false;

        // The graphics state that is currently set on the render encoder, so an
        // unchanged part is not set again.
        uint32_t m_AppliedStencilRefValue = 0;
        nvrhi::Color m_AppliedBlendConstantColor{};
        nvrhi::Viewport m_AppliedViewport{};
        size_t m_AppliedViewportCount = 0;
        nvrhi::Rect m_AppliedScissor{};
        size_t m_AppliedScissorCount = 0;
        static_vector<VertexBufferBinding, c_MaxVertexAttributes> m_AppliedVertexBuffers;
        BindingSetVector m_AppliedBindings;

        // The compute state that is currently set on the compute encoder.
        bool m_ComputeEncoderStateValid = false;
        ComputePipeline* m_AppliedComputePipeline = nullptr;
        BindingSetVector m_AppliedComputeBindings;

        // Apply permanent resource states when the command list is submitted.
        std::vector<std::pair<TextureStateExtension*, ResourceStates>> m_PermanentTextureStates;
        std::vector<std::pair<BufferStateExtension*, ResourceStates>> m_PermanentBufferStates;

        // The resources the command buffer uses, so they stay alive until it is
        // committed. Metal keeps the Metal objects alive on its own, but the NVRHI
        // wrappers are what the command list still points at.
        std::vector<RefCountPtr<IResource>> m_ReferencedResources;
        void* m_DeclaredArgumentBuffer = nullptr;
        ShaderType m_DeclaredArgumentBufferStage = ShaderType::None;

        // The timer queries the command buffer sampled, so the submission can stamp
        // its serial on them.
        std::vector<RefCountPtr<TimerQuery>> m_SampledTimerQueries;

        std::vector<uint8_t> m_PushConstants;

        bool m_IsOpen = false;
    };

    struct SubmissionCompletion
    {
        std::mutex mutex;
        std::condition_variable condition;
        std::set<uint64_t> finishedOutOfOrder;
        uint64_t lastCompletedSerial = 0;
        uint64_t firstFailedSerial = 0;
        uint32_t pendingCallbacks = 0;
    };

    class Device : public RefCounter<nvrhi::metal::IDevice>
    {
    public:
        Device(const DeviceDesc& desc);
        ~Device() override;

        // Internal backend methods
        Context& getContext() { return m_Context; }
        id<MTLCommandQueue> getQueue() const { return m_Context.queue; }
        // The caller holds m_Mutex.
        uint64_t submitCommandListLocked(ICommandList* commandList);
#ifdef NVRHI_METAL_TEST_HOOKS
        void setSubmissionHooks(const test::SubmissionHooks& hooks);
        Object getSubmissionEvent() const { return Object((__bridge void*)m_SubmissionEvent); }
        test::ArgumentBufferStats getArgumentBufferStats() const;
        void resetArgumentBufferStats();
        std::atomic<uint64_t> argumentEncodersCreated{0};
        std::atomic<uint64_t> argumentBuffersEncoded{0};
        std::atomic<uint64_t> argumentBuffersReused{0};
        std::atomic<uint64_t> argumentBufferLength{0};
        std::atomic<uint64_t> argumentBufferDeclarations{0};
#endif

        // Null resources bound to the unwritten slots of a shader array argument.
        id<MTLBuffer> getNullBuffer();
        // Null textures must match the shader's dimension and scalar type.
        id<MTLTexture> getNullTexture(TextureDimension dimension, Format format);
        id<MTLTexture> getNullTextureBuffer(Format format);
        id<MTLSamplerState> getNullSampler();

        // A queue wait the GPU performs before the queue's next submission.
        bool commitQueueWaits(CommandQueue executionQueue);

        // The shared counter sample buffer the timer queries sample into.
        id<MTLCounterSampleBuffer> getCounterSampleBuffer() const { return m_CounterSampleBuffer; }
        // The empty render pass that samples a timestamp needs a colour attachment.
        id<MTLTexture> getTimerQuerySampleTarget() const { return m_TimerQuerySampleTarget; }
        bool initTimerQueryPool();

        // The pipeline that expands the indirect draw arguments for a counted draw.
        id<MTLComputePipelineState> getIndirectCountPipeline();

        // IResource implementation
        Object getNativeObject(ObjectType objectType) override;

        // IDevice implementation
        HeapHandle createHeap(const HeapDesc& d) override;
        bool queryTopLevelAccelStructPrebuildInfo(const rt::AccelStructDesc& desc, uint32_t instanceCount, rt::AccelStructPrebuildInfo& outInfo) override;
        TextureHandle createTexture(const TextureDesc& d) override;
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
        bool bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset) override;
        TextureHandle createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc) override;
        StagingTextureHandle createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess) override;
        void* mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t* outRowPitch) override;
        void unmapStagingTexture(IStagingTexture* tex) override;
        void getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings) override;
        void updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue = CommandQueue::Graphics) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture) override;
        BufferHandle createBuffer(const BufferDesc& d) override;
        void* mapBuffer(IBuffer* buffer, CpuAccessMode cpuAccess) override;
        void unmapBuffer(IBuffer* buffer) override;
        MemoryRequirements getBufferMemoryRequirements(IBuffer* buffer) override;
        bool bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset) override;
        BufferHandle createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc) override;
        ShaderHandle createShader(const ShaderDesc& d, const void* binary, size_t binarySize) override;
        ShaderHandle createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants) override;
        ShaderLibraryHandle createShaderLibrary(const void* binary, size_t binarySize) override;
        SamplerHandle createSampler(const SamplerDesc& d) override;
        InputLayoutHandle createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader) override;
        EventQueryHandle createEventQuery() override;
        void setEventQuery(IEventQuery* query, CommandQueue queue) override;
        bool pollEventQuery(IEventQuery* query) override;
        void waitEventQuery(IEventQuery* query) override;
        void resetEventQuery(IEventQuery* query) override;
        TimerQueryHandle createTimerQuery() override;
        bool pollTimerQuery(ITimerQuery* query) override;
        float getTimerQueryTime(ITimerQuery* query) override;
        void resetTimerQuery(ITimerQuery* query) override;
        GraphicsAPI getGraphicsAPI() override;
        FramebufferHandle createFramebuffer(const FramebufferDesc& desc) override;
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo) override;
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb) override;
        ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) override;
        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo) override;
        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb) override;
        rt::PipelineHandle createRayTracingPipeline(const rt::PipelineDesc& desc) override;
        BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) override;
        BindingLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) override;
        BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) override;
        DescriptorTableHandle createDescriptorTable(IBindingLayout* layout) override;
        void resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents = true) override;
        bool writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item) override;
        rt::OpacityMicromapHandle createOpacityMicromap(const rt::OpacityMicromapDesc& desc) override;
        rt::AccelStructHandle createAccelStruct(const rt::AccelStructDesc& desc) override;
        MemoryRequirements getAccelStructMemoryRequirements(rt::IAccelStruct* as) override;
        rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) override;
        bool bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset) override;
        CommandListHandle createCommandList(const CommandListParameters& params = CommandListParameters()) override;
        uint64_t executeCommandLists(ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue = CommandQueue::Graphics) override;
        void queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance) override;
        bool waitForIdle() override;
        CommandListLifetimeTrackerHandle createCommandListLifetimeTracker(CommandQueue executionQueue) override;
        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
        coopvec::MatMulFormatSupport queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination) override;
        coopvec::TrainingFormatSupport queryCoopVecTrainingFormatSupport(coopvec::DataType componentType) override;
        size_t getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns) override;
        Object getNativeQueue(ObjectType objectType, CommandQueue queue) override;
        IMessageCallback* getMessageCallback() override;
        bool isAftermathEnabled() override;
        AftermathCrashDumpHelper& getAftermathCrashDumpHelper() override;

        // metal::IDevice implementation
        Object getNativeDevice() override;
        Object getNativeCommandQueue() override;

        uint64_t getNativeDeviceMemorySize() override;
        uint64_t getNativeAllocatedSize() override;

        uint64_t getMappedBufferFlushAlignment() override;
        bool flushMappedBuffer(IBuffer* buffer, uint64_t offset, uint64_t size) override;

    private:
        Context m_Context;
        std::mutex m_NullResourcesMutex;
        id<MTLBuffer> m_NullBuffer = nil;
        std::map<std::pair<TextureDimension, Format>, id<MTLTexture>> m_NullTextures;
        // The null texture buffers, one per element format.
        std::map<Format, id<MTLTexture>> m_NullTextureBuffers;
        id<MTLSamplerState> m_NullSampler = nil;
        // The shared timestamp sample buffer and attachment for the sampling passes.
        id<MTLCounterSampleBuffer> m_CounterSampleBuffer = nil;
        id<MTLTexture> m_TimerQuerySampleTarget = nil;
        // Queries can outlive the device, so their allocator has independent lifetime.
        std::shared_ptr<TimerQueryAllocator> m_TimerQueryAllocator = std::make_shared<TimerQueryAllocator>();
        bool m_TimerQueryPoolInitAttempted = false;
        // The submissions signal this event with their serial, and the queue waits of
        // queueWaitForCommandList wait on it before the next submission of the queue.
        id<MTLSharedEvent> m_SubmissionEvent = nil;
        id<MTLComputePipelineState> m_IndirectCountPipeline = nil;
        uint64_t m_PendingQueueWaits[3] = { 0, 0, 0 };
        std::mutex m_QueueWaitMutex;
        std::mutex m_Mutex;
        uint64_t m_LastSubmittedSerial = 0;
        std::shared_ptr<SubmissionCompletion> m_Completion = std::make_shared<SubmissionCompletion>();
#ifdef NVRHI_METAL_TEST_HOOKS
        test::SubmissionHooks m_TestHooks;
#endif
    };

    MTLPixelFormat convertFormat(Format format);
    bool isPacked16BitColorFormat(Format format);
    MTLSamplerMinMagFilter convertSamplerFilter(bool linear);
    MTLSamplerMipFilter convertSamplerMipFilter(bool linear);
    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode, bool zeroBorder);
    MTLSamplerBorderColor convertSamplerBorderColor(const Color& color);
    MTLCompareFunction convertCompareFunc(ComparisonFunc op);
    MTLCompareFunction convertSamplerCompareFunc(ComparisonFunc op);

    MTLVertexFormat convertVertexFormat(Format format);
    MTLPrimitiveType convertPrimitiveType(PrimitiveType type);
    MTLCullMode convertCullMode(RasterCullMode mode);
    MTLBlendFactor convertBlendFactor(BlendFactor factor);
    MTLBlendOperation convertBlendOp(BlendOp op);
    MTLStencilOperation convertStencilOp(StencilOp op);

    inline uint64_t alignSize(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }
}
