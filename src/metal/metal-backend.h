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

#include "metal-spirv.h"

#include <mutex>
#include <string>
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

    struct Context
    {
        id<MTLDevice> device = nil;
        id<MTLCommandQueue> queue = nil;
        IMessageCallback* messageCallback = nullptr;

        // The Metal language version the shaders are compiled against.
        uint32_t mslVersionMajor = 2;
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

    class Buffer : public RefCounter<IBuffer>
    {
    public:
        BufferDesc desc;
        id<MTLBuffer> buffer = nil;
        void* mappedMemory = nullptr;
        bool isCPUVisible = false;

        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

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

    class Texture : public RefCounter<ITexture>
    {
    public:
        TextureDesc desc;
        id<MTLTexture> texture = nil;
        bool isCPUVisible = false;

        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool isReadOnlyDSV = false,
            std::optional<ComponentMapping> overrideComponentMapping = std::nullopt) override;
    };

    class Sampler : public RefCounter<ISampler>
    {
    public:
        SamplerDesc desc;
        id<MTLSamplerState> sampler = nil;

        const SamplerDesc& getDesc() const override { return desc; }
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

        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
        Object getNativeObject(ObjectType objectType) override;
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
        std::vector<ResourceHandle> resources; // [UAA] Own the attachments referenced by desc.

        const FramebufferDesc& getDesc() const override { return desc; }
        const FramebufferInfoEx& getFramebufferInfo() const override { return framebufferInfo; }
    };

    class BindingLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindingLayoutDesc desc;

        const BindingLayoutDesc* getDesc() const override { return &desc; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return nullptr; }
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

    // The resources a pipeline reads, resolved from the shaders' translated binding plans.
    class PipelineBindingPlan
    {
    public:
        struct Entry
        {
            ShaderType stage = ShaderType::None;
            uint32_t descriptorSet = 0;
            uint32_t slot = 0;
            BoundResource bound;
        };

        std::vector<Entry> entries;

        const Entry* find(ShaderType stage, uint32_t descriptorSet, uint32_t slot) const
        {
            for (const auto& entry : entries)
            {
                if (entry.stage == stage && entry.descriptorSet == descriptorSet && entry.slot == slot)
                    return &entry;
            }

            return nullptr;
        }
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

    class TimerQuery : public RefCounter<ITimerQuery>
    {
    public:
        float time = 0.f;
        bool resolved = false;
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
        void commitBarriers() override;
        ResourceStates getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel) override;
        ResourceStates getBufferState(IBuffer* buffer) override;
        IDevice* getDevice() override;
        const CommandListParameters& getDesc() override;
        CommandListMemoryUsage getInternalMemoryUsage() const override;
        Object getNativeObject(ObjectType objectType) override;

        id<MTLCommandBuffer> getCommandBuffer() const { return m_CommandBuffer; }
        void releaseCommandBuffer() { m_CommandBuffer = nil; }

    private:
        void endEncoder();
        id<MTLComputeCommandEncoder> getComputeEncoder();
        id<MTLBlitCommandEncoder> getBlitEncoder();
        id<MTLRenderCommandEncoder> getRenderEncoder(Framebuffer* framebuffer);

        void bindComputeResources(const ComputeState& state);
        void bindGraphicsResources(const GraphicsState& state);
        void bindResource(const BoundResource& bound, const nvrhi::BindingSetItem& item);
        void bindGraphicsResource(const BoundResource& bound, const nvrhi::BindingSetItem& item, bool vertexStage);
        void bindPushConstants(uint32_t bufferIndex, uint32_t size, ShaderType stage);
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

        std::vector<uint8_t> m_PushConstants;
        id<MTLBuffer> m_PushConstantBuffer = nil;

        bool m_IsOpen = false;
    };

    class Device : public RefCounter<nvrhi::metal::IDevice>
    {
    public:
        Device(const DeviceDesc& desc);
        ~Device() override;

        // Internal backend methods
        Context& getContext() { return m_Context; }
        id<MTLCommandQueue> getQueue() const { return m_Context.queue; }
        uint64_t submitCommandList(ICommandList* commandList);

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

        // [UAA] - START - mapped buffer write flush
        uint64_t getMappedBufferFlushAlignment() override;
        bool flushMappedBuffer(IBuffer* buffer, uint64_t offset, uint64_t size) override;
        // [UAA] - END

    private:
        Context m_Context;
        std::mutex m_Mutex;
        uint64_t m_LastSubmittedSerial = 0;
        uint64_t m_LastCompletedSerial = 0;
        std::mutex m_CompletionMutex;
        std::condition_variable m_CompletionCondition;
    };

    MTLPixelFormat convertFormat(Format format);
    MTLSamplerMinMagFilter convertSamplerFilter(bool linear);
    MTLSamplerMipFilter convertSamplerMipFilter(bool linear);
    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode);
    MTLSamplerBorderColor convertSamplerBorderColor(const Color& color);
    MTLCompareFunction convertCompareFunc(ComparisonFunc op);

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
