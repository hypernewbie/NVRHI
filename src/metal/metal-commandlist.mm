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

#include <nvrhi/common/misc.h>

#include <algorithm>
#include <cfloat>
#include <vector>

namespace nvrhi::metal
{
    namespace
    {
        // Metal copies between a buffer and a texture require 256 byte aligned rows.
        constexpr uint64_t kBlitRowAlignment = 256;

        uint32_t getMipSize(uint32_t size, uint32_t mipLevel)
        {
            return std::max(1u, size >> mipLevel);
        }

        // Writes the clear value into a buffer of the given format.
        void fillClearData(std::vector<uint8_t>& data, uint64_t rowPitch, uint32_t pixelsPerRow, uint32_t rows, Format format, const Color& color, uint32_t intValue, bool integer)
        {
            const FormatInfo& info = getFormatInfo(format);
            const uint32_t components = uint32_t( info.hasRed ) + uint32_t( info.hasGreen ) + uint32_t( info.hasBlue ) + uint32_t( info.hasAlpha );
            if (components == 0)
                return;

            const uint32_t bytesPerComponent = info.kind == FormatKind::Float ? 4 : ( info.kind == FormatKind::Normalized ? 1 : info.bytesPerBlock / components );

            float values[4] = {};
            uint32_t valueIndex = 0;
            if ( info.hasRed ) values[valueIndex++] = color.r;
            if ( info.hasGreen ) values[valueIndex++] = color.g;
            if ( info.hasBlue ) values[valueIndex++] = color.b;
            if ( info.hasAlpha ) values[valueIndex++] = color.a;

            // BGRA texels store blue before red; the clear colour is RGBA.
            if (format == Format::BGRA8_UNORM || format == Format::SBGRA8_UNORM || format == Format::BGRA4_UNORM)
                std::swap(values[0], values[2]);

            for ( uint32_t row = 0; row < rows; ++row )
            for ( uint32_t pixel = 0; pixel < pixelsPerRow; ++pixel )
            {
                uint8_t* target = data.data() + row * rowPitch + pixel * components * bytesPerComponent;

                for ( uint32_t component = 0; component < components; ++component )
                {
                    uint8_t* componentData = target + component * bytesPerComponent;

                    if ( integer )
                    {
                        for ( uint32_t byte = 0; byte < bytesPerComponent; ++byte )
                            componentData[byte] = uint8_t(( intValue >> (byte * 8)) & 0xFF);
                        continue;
                    }

                    if ( info.kind == FormatKind::Float )
                    {
                        const float value = values[component];
                        memcpy(componentData, &value, 4);
                    }
                    else if ( info.kind == FormatKind::Normalized )
                    {
                        const float value = values[component];
                        componentData[0] = uint8_t(info.isSigned
                            ? std::max(-127.f, std::min(127.f, value * 127.f))
                            : std::max(0.f, std::min(1.f, value)) * 255.f);
                    }
                    else
                    {
                        const uint32_t value = uint32_t(values[component]);
                        for ( uint32_t byte = 0; byte < bytesPerComponent; ++byte )
                            componentData[byte] = uint8_t((value >> (byte * 8)) & 0xFF);
                    }
                }
            }
        }
    }

    CommandList::CommandList(Device* device, const CommandListParameters& params)
        : m_Device(device)
        , m_Params(params)
    {
    }

    CommandList::~CommandList()
    {
        endEncoder();
    }

    void CommandList::open()
    {
        endEncoder();
        m_CurrentComputePipeline = nullptr;
        m_CurrentGraphicsPipeline = nullptr;
        m_ReferencedResources.clear();
        m_SampledTimerQueries.clear();
        m_CommandBuffer = [m_Device->getQueue() commandBuffer];
        m_IsOpen = true;
    }

    void CommandList::close()
    {
        endEncoder();
        m_IsOpen = false;
    }

    void CommandList::clearState()
    {
        m_CurrentComputePipeline = nullptr;
        m_CurrentGraphicsPipeline = nullptr;
    }

    void CommandList::endEncoder()
    {
        if (m_ComputeEncoder)
        {
            [m_ComputeEncoder endEncoding];
            m_ComputeEncoder = nil;
            m_ComputeEncoderStateValid = false;
        }

        if (m_RenderEncoder)
        {
            [m_RenderEncoder endEncoding];
            m_RenderEncoder = nil;
            m_RenderEncoderStateValid = false;
            // The framebuffer stays, because the pass is reopened with it when the
            // next draw needs a render encoder.
        }

        if (m_BlitEncoder)
        {
            [m_BlitEncoder endEncoding];
            m_BlitEncoder = nil;
        }
    }

    id<MTLComputeCommandEncoder> CommandList::getComputeEncoder()
    {
        if (!m_ComputeEncoder)
        {
            endEncoder();
            m_ComputeEncoder = [m_CommandBuffer computeCommandEncoder];
            m_ComputeEncoderStateValid = false;

            // A new encoder starts with no state, so the current compute state is re-applied.
            if (m_ComputeEncoder && m_CurrentComputePipeline)
            {
                [m_ComputeEncoder setComputePipelineState:m_CurrentComputePipeline->pipeline];
                m_AppliedComputePipeline = m_CurrentComputePipeline;
                m_AppliedComputeBindings = m_CurrentComputeState.bindings;
                bindComputeResources(m_CurrentComputeState);
                m_ComputeEncoderStateValid = true;
            }
        }

        return m_ComputeEncoder;
    }

    id<MTLBlitCommandEncoder> CommandList::getBlitEncoder()
    {
        if (!m_BlitEncoder)
        {
            endEncoder();
            m_BlitEncoder = [m_CommandBuffer blitCommandEncoder];
        }

        return m_BlitEncoder;
    }

    // ------------------------------------------------------------------ Compute

    void CommandList::setComputeState(const ComputeState& state)
    {
        ComputePipeline* pso = checked_cast<ComputePipeline*>(state.pipeline);
        if (!pso)
            return;

        m_CurrentComputeState = state;
        m_CurrentComputePipeline = pso;
        m_IndirectBuffer = checked_cast<Buffer*>(state.indirectParams);

        id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
        if (!encoder)
            return;

        // A new encoder has no state, so everything is applied again. Otherwise only
        // the parts that changed are applied, like the Vulkan backend does.
        const bool fullUpdate = !m_ComputeEncoderStateValid;
        const bool pipelineChanged = fullUpdate || m_AppliedComputePipeline != pso;

        if (pipelineChanged)
        {
            [encoder setComputePipelineState:pso->pipeline];
            m_AppliedComputePipeline = pso;
        }

        // A new pipeline can map the same binding set to other argument slots.
        if (fullUpdate || pipelineChanged || arraysAreDifferent(m_AppliedComputeBindings, state.bindings))
        {
            m_AppliedComputeBindings = state.bindings;
            bindComputeResources(state);
        }

        m_ComputeEncoderStateValid = true;
    }

    namespace
    {
        // Map descriptor types to Metal argument namespaces.
        bool getBindlessArrayShape(ResourceType type, ResourceKind& kind, bool& writeable)
        {
            switch (type)
            {
            case ResourceType::Texture_SRV:
            case ResourceType::TypedBuffer_SRV:
                kind = ResourceKind::Texture; writeable = false; return true;
            case ResourceType::Texture_UAV:
            case ResourceType::TypedBuffer_UAV:
                kind = ResourceKind::Texture; writeable = true; return true;
            case ResourceType::ConstantBuffer:
            case ResourceType::StructuredBuffer_SRV:
            case ResourceType::RawBuffer_SRV:
                kind = ResourceKind::Buffer; writeable = false; return true;
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_UAV:
                kind = ResourceKind::Buffer; writeable = true; return true;
            case ResourceType::Sampler:
                kind = ResourceKind::Sampler; writeable = false; return true;
            default:
                return false;
            }
        }
    }

    void CommandList::bindComputeResources(const ComputeState& state)
    {
        id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
        if (!encoder)
            return;

        for (uint32_t i = 0; i < state.bindings.size(); ++i)
        {
            nvrhi::IBindingSet* binding = state.bindings[i];
            if (!binding || !binding->getLayout())
                continue;

            const BindingLayout* layout = checked_cast<BindingLayout*>(binding->getLayout());

            // A descriptor table binds its slots as array elements of the bindless arrays.
            if (layout->isBindless)
            {
                bindDescriptorTable(checked_cast<DescriptorTable*>(binding), m_CurrentComputePipeline->plan, i);
                continue;
            }

            const BindingSet* bindingSet = checked_cast<BindingSet*>(binding);
            const ShaderType stageMask = layout->desc.visibility;

            for (const auto& item : bindingSet->desc.bindings)
            {
                for (const auto& entry : m_CurrentComputePipeline->plan.entries)
                {
                    if ((entry.stage & stageMask) == ShaderType::None)
                        continue;
                    if (entry.bindingIndex != i || entry.slot != item.slot)
                        continue;

                    BoundResource bound = entry.bound;
                    bound.index += item.arrayElement;
                    bindResource(bound, item);
                }
            }
        }
    }

    void CommandList::bindResource(const BoundResource& bound, const nvrhi::BindingSetItem& item)
    {
        switch (bound.kind)
        {
        case ResourceKind::Buffer:
        {
            // The resource has to match the kind the shader expects.
            const bool isBuffer = item.type == ResourceType::ConstantBuffer || item.type == ResourceType::VolatileConstantBuffer
                || item.type == ResourceType::StructuredBuffer_SRV || item.type == ResourceType::StructuredBuffer_UAV
                || item.type == ResourceType::RawBuffer_SRV || item.type == ResourceType::RawBuffer_UAV;
            if (!isBuffer)
                return;

            Buffer* buffer = checked_cast<Buffer*>(item.resourceHandle);
            if (!buffer || !buffer->buffer)
                return;

            if (m_ComputeEncoder)
                [m_ComputeEncoder setBuffer:buffer->buffer offset:item.range.byteOffset atIndex:bound.index];
            break;
        }
        case ResourceKind::Texture:
        {
            // A typed buffer is a texture in the shader but a buffer in NVRHI.
            if (item.type == ResourceType::TypedBuffer_SRV || item.type == ResourceType::TypedBuffer_UAV)
            {
                Buffer* buffer = checked_cast<Buffer*>(item.resourceHandle);
                if (!buffer || !buffer->buffer)
                    return;

                id<MTLTexture> view = getTypedBufferView(m_Device->getContext(), buffer, item.format, item.range.byteOffset, item.range.byteSize);
                if (m_ComputeEncoder && view)
                    [m_ComputeEncoder setTexture:view atIndex:bound.index];
                break;
            }

            const bool isTexture = item.type == ResourceType::Texture_SRV || item.type == ResourceType::Texture_UAV;
            if (!isTexture)
                return;

            Texture* texture = checked_cast<Texture*>(item.resourceHandle);
            if (!texture || !texture->texture)
                return;

            const ComponentMapping componentMapping = resolveComponentMapping(item.overrideComponentMapping, texture->desc.defaultComponentMapping);
            id<MTLTexture> view = texture->getSubresourceView(item.subresources, item.dimension, item.format, componentMapping);
            if (m_ComputeEncoder && view)
                [m_ComputeEncoder setTexture:view atIndex:bound.index];
            break;
        }
        case ResourceKind::Sampler:
        {
            Sampler* sampler = checked_cast<Sampler*>(item.resourceHandle);
            if (!sampler || !sampler->sampler)
                return;

            if (m_ComputeEncoder)
                [m_ComputeEncoder setSamplerState:sampler->sampler atIndex:bound.index];
            break;
        }
        }
    }

    void CommandList::dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        if (!m_CurrentComputePipeline)
            return;

        id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
        if (!encoder)
            return;

        bindPushConstants(m_CurrentComputePipeline->pushConstantBufferIndex, m_CurrentComputePipeline->pushConstantSize, ShaderType::Compute);

        const uint32_t* threadGroupSize = m_CurrentComputePipeline->threadGroupSize;
        [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, groupsZ)
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2])];
    }

    void CommandList::dispatchIndirect(uint32_t offsetBytes)
    {
        if (!m_CurrentComputePipeline || !m_IndirectBuffer)
            return;

        id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
        if (!encoder)
            return;

        bindPushConstants(m_CurrentComputePipeline->pushConstantBufferIndex, m_CurrentComputePipeline->pushConstantSize, ShaderType::Compute);

        // The indirect buffer holds the same three group counts for Metal as for the other
        // backends; the thread group size comes from the pipeline like threadsPerThreadgroup.
        const uint32_t* threadGroupSize = m_CurrentComputePipeline->threadGroupSize;
        [encoder dispatchThreadgroupsWithIndirectBuffer:m_IndirectBuffer->buffer
            indirectBufferOffset:offsetBytes
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2])];
    }

    // ------------------------------------------------------------------ Transfers

    // Check the required state against the resource's permanent state.
    void CommandList::verifyPermanentState(TextureStateExtension* texture, ResourceStates state)
    {
        if (texture->permanentState != ResourceStates::Unknown)
            verifyPermanentResourceState(texture->permanentState, state, true, texture->descRef.debugName, m_Device->getMessageCallback());
    }

    void CommandList::verifyPermanentState(BufferStateExtension* buffer, ResourceStates state)
    {
        if (buffer->permanentState != ResourceStates::Unknown)
            verifyPermanentResourceState(buffer->permanentState, state, false, buffer->descRef.debugName, m_Device->getMessageCallback());
    }

    void CommandList::writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes)
    {
        Buffer* buffer = checked_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer || dataSize == 0)
            return;

        verifyPermanentState(buffer, ResourceStates::CopyDest);

        id<MTLBuffer> source = [m_Device->getContext().device newBufferWithBytes:data length:dataSize options:MTLResourceStorageModeShared];
        if (!source)
            return;

        [getBlitEncoder() copyFromBuffer:source sourceOffset:0 toBuffer:buffer->buffer destinationOffset:destOffsetBytes size:dataSize];
    }

    void CommandList::copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes)
    {
        Buffer* destBuffer = checked_cast<Buffer*>(dest);
        Buffer* srcBuffer = checked_cast<Buffer*>(src);
        if (!destBuffer || !srcBuffer || dataSizeBytes == 0)
            return;

        verifyPermanentState(destBuffer, ResourceStates::CopyDest);
        verifyPermanentState(srcBuffer, ResourceStates::CopySource);

        [getBlitEncoder() copyFromBuffer:srcBuffer->buffer sourceOffset:srcOffsetBytes toBuffer:destBuffer->buffer destinationOffset:destOffsetBytes size:dataSizeBytes];
    }

    void CommandList::clearBufferUInt(IBuffer* b, uint32_t clearValue)
    {
        Buffer* buffer = checked_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        verifyPermanentState(buffer, ResourceStates::CopyDest);

        const uint64_t size = buffer->desc.byteSize;
        std::vector<uint32_t> data(size / sizeof(uint32_t), clearValue);

        id<MTLBuffer> source = [m_Device->getContext().device newBufferWithBytes:data.data() length:data.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!source)
            return;

        [getBlitEncoder() copyFromBuffer:source sourceOffset:0 toBuffer:buffer->buffer destinationOffset:0 size:data.size() * sizeof(uint32_t)];
    }

    void CommandList::writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch)
    {
        Texture* texture = checked_cast<Texture*>(dest);
        if (!texture || !texture->texture || !data)
            return;

        verifyPermanentState(texture, ResourceStates::CopyDest);

        const TextureDesc& desc = texture->desc;
        const uint32_t width = getMipSize(desc.width, mipLevel);
        const uint32_t height = getMipSize(desc.height, mipLevel);
        const uint32_t depth = desc.dimension == TextureDimension::Texture3D ? getMipSize(desc.depth, mipLevel) : 1;
        const uint32_t bytesPerPixel = getFormatInfo(desc.format).bytesPerBlock;
        if (bytesPerPixel == 0)
            return;

        const uint64_t sourceRowPitch = rowPitch ? rowPitch : uint64_t(width) * bytesPerPixel;
        const uint64_t sourceDepthPitch = depthPitch ? depthPitch : sourceRowPitch * height;
        const uint64_t alignedRowPitch = alignSize(sourceRowPitch, kBlitRowAlignment);
        const uint64_t alignedDepthPitch = alignedRowPitch * height;

        id<MTLBuffer> source = [m_Device->getContext().device newBufferWithLength:alignedDepthPitch * depth options:MTLResourceStorageModeShared];
        if (!source)
            return;

        uint8_t* destination = (uint8_t*)source.contents;
        const uint8_t* sourceData = (const uint8_t*)data;
        for (uint32_t z = 0; z < depth; ++z)
            for (uint32_t y = 0; y < height; ++y)
                memcpy(destination + z * alignedDepthPitch + y * alignedRowPitch, sourceData + z * sourceDepthPitch + y * sourceRowPitch, sourceRowPitch);

        [getBlitEncoder() copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:alignedRowPitch sourceBytesPerImage:alignedDepthPitch
            sourceSize:MTLSizeMake(width, height, depth)
            toTexture:texture->texture destinationSlice:arraySlice destinationLevel:mipLevel destinationOrigin:MTLOriginMake(0, 0, 0)];
    }

    void CommandList::copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice)
    {
        Texture* destTexture = checked_cast<Texture*>(dest);
        Texture* srcTexture = checked_cast<Texture*>(src);
        if (!destTexture || !srcTexture)
            return;

        verifyPermanentState(destTexture, ResourceStates::CopyDest);
        verifyPermanentState(srcTexture, ResourceStates::CopySource);

        const uint32_t width = srcSlice.width == uint32_t(-1) ? getMipSize(srcTexture->desc.width, srcSlice.mipLevel) : srcSlice.width;
        const uint32_t height = srcSlice.height == uint32_t(-1) ? getMipSize(srcTexture->desc.height, srcSlice.mipLevel) : srcSlice.height;
        const uint32_t depth = srcSlice.depth == uint32_t(-1) ? 1 : srcSlice.depth;

        [getBlitEncoder() copyFromTexture:srcTexture->texture sourceSlice:srcSlice.arraySlice sourceLevel:srcSlice.mipLevel
            sourceOrigin:MTLOriginMake(srcSlice.x, srcSlice.y, srcSlice.z) sourceSize:MTLSizeMake(width, height, depth)
            toTexture:destTexture->texture destinationSlice:destSlice.arraySlice destinationLevel:destSlice.mipLevel
            destinationOrigin:MTLOriginMake(destSlice.x, destSlice.y, destSlice.z)];
    }

    void CommandList::copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice)
    {
        StagingTexture* staging = checked_cast<StagingTexture*>(dest);
        Texture* srcTexture = checked_cast<Texture*>(src);
        if (!staging || !srcTexture || !staging->buffer)
            return;

        verifyPermanentState(srcTexture, ResourceStates::CopySource);

        const uint32_t mip = srcSlice.mipLevel;
        if (mip >= staging->mipOffsets.size())
            return;

        const uint32_t width = getMipSize(srcTexture->desc.width, mip);
        const uint32_t height = getMipSize(srcTexture->desc.height, mip);
        const uint64_t rowPitch = staging->mipRowPitches[mip];

        [getBlitEncoder() copyFromTexture:srcTexture->texture sourceSlice:srcSlice.arraySlice sourceLevel:mip
            sourceOrigin:MTLOriginMake(srcSlice.x, srcSlice.y, srcSlice.z) sourceSize:MTLSizeMake(width, height, 1)
            toBuffer:staging->buffer destinationOffset:staging->mipOffsets[mip] destinationBytesPerRow:rowPitch destinationBytesPerImage:rowPitch * height];
    }

    void CommandList::copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice)
    {
        Texture* destTexture = checked_cast<Texture*>(dest);
        StagingTexture* staging = checked_cast<StagingTexture*>(src);
        if (!destTexture || !staging || !staging->buffer)
            return;

        verifyPermanentState(destTexture, ResourceStates::CopyDest);

        const uint32_t mip = srcSlice.mipLevel;
        if (mip >= staging->mipOffsets.size())
            return;

        const uint32_t width = getMipSize(destTexture->desc.width, mip);
        const uint32_t height = getMipSize(destTexture->desc.height, mip);
        const uint64_t rowPitch = staging->mipRowPitches[mip];

        [getBlitEncoder() copyFromBuffer:staging->buffer sourceOffset:staging->mipOffsets[mip] sourceBytesPerRow:rowPitch sourceBytesPerImage:rowPitch * height
            sourceSize:MTLSizeMake(width, height, 1)
            toTexture:destTexture->texture destinationSlice:destSlice.arraySlice destinationLevel:mip destinationOrigin:MTLOriginMake(0, 0, 0)];
    }

    void CommandList::resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources)
    {
        Texture* destTexture = checked_cast<Texture*>(dest);
        Texture* srcTexture = checked_cast<Texture*>(src);
        if (!destTexture || !srcTexture)
            return;

        verifyPermanentState(destTexture, ResourceStates::ResolveDest);
        verifyPermanentState(srcTexture, ResourceStates::ResolveSource);

        // Metal resolves through the render pass, one pass per subresource.
        const TextureSubresourceSet resolved = srcSubresources.resolve(srcTexture->desc, true);
        for (uint32_t mip = 0; mip < resolved.numMipLevels; ++mip)
        {
            for (uint32_t slice = 0; slice < resolved.numArraySlices; ++slice)
            {
                endEncoder();

                MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];

                const bool isDepth = getFormatInfo(srcTexture->desc.format).kind == FormatKind::DepthStencil;
                MTLRenderPassAttachmentDescriptor* attachment = isDepth ? (MTLRenderPassAttachmentDescriptor*)passDescriptor.depthAttachment
                    : (MTLRenderPassAttachmentDescriptor*)passDescriptor.colorAttachments[0];
                attachment.texture = srcTexture->texture;
                attachment.level = resolved.baseMipLevel + mip;
                attachment.slice = resolved.baseArraySlice + slice;
                attachment.loadAction = MTLLoadActionLoad;
                // Resolve must not discard the multisampled source.
                attachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
                attachment.resolveTexture = destTexture->texture;
                attachment.resolveLevel = dstSubresources.baseMipLevel + mip;
                attachment.resolveSlice = dstSubresources.baseArraySlice + slice;

                if (isDepth)
                {
                    passDescriptor.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;

                    if (getFormatInfo(srcTexture->desc.format).hasStencil)
                    {
                        passDescriptor.stencilAttachment.texture = srcTexture->texture;
                        passDescriptor.stencilAttachment.level = resolved.baseMipLevel + mip;
                        passDescriptor.stencilAttachment.slice = resolved.baseArraySlice + slice;
                        passDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
                        passDescriptor.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
                        passDescriptor.stencilAttachment.resolveTexture = destTexture->texture;
                        passDescriptor.stencilAttachment.resolveLevel = dstSubresources.baseMipLevel + mip;
                        passDescriptor.stencilAttachment.resolveSlice = dstSubresources.baseArraySlice + slice;
                        passDescriptor.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
                    }
                }

                id<MTLRenderCommandEncoder> encoder = [m_CommandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
                [encoder endEncoding];
            }
        }
    }

    void CommandList::clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor)
    {
        Texture* texture = checked_cast<Texture*>(t);
        if (!texture)
            return;

        verifyPermanentState(texture, ResourceStates::CopyDest);

        clearTexture(texture, subresources, clearColor, 0, false);
    }

    void CommandList::clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor)
    {
        Texture* texture = checked_cast<Texture*>(t);
        if (!texture)
            return;

        verifyPermanentState(texture, ResourceStates::CopyDest);

        clearTexture(texture, subresources, Color(), clearColor, true);
    }

    void CommandList::clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil)
    {
        (void)subresources;

        Texture* texture = checked_cast<Texture*>(t);
        if (!texture)
            return;

        verifyPermanentState(texture, ResourceStates::DepthWrite);

        // Metal clears a depth attachment through the render pass.
        endEncoder();

        MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        passDescriptor.depthAttachment.texture = texture->texture;
        passDescriptor.depthAttachment.loadAction = clearDepth ? MTLLoadActionClear : MTLLoadActionDontCare;
        passDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
        passDescriptor.depthAttachment.clearDepth = depth;

        if (getFormatInfo(texture->desc.format).hasStencil)
        {
            passDescriptor.stencilAttachment.texture = texture->texture;
            passDescriptor.stencilAttachment.loadAction = clearStencil ? MTLLoadActionClear : MTLLoadActionDontCare;
            passDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
            passDescriptor.stencilAttachment.clearStencil = stencil;
        }

        id<MTLRenderCommandEncoder> encoder = [m_CommandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
        [encoder endEncoding];
    }

    void CommandList::clearTexture(Texture* texture, const TextureSubresourceSet& subresources, const Color& color, uint32_t intValue, bool integer)
    {
        const TextureDesc& desc = texture->desc;
        const uint32_t bytesPerPixel = getFormatInfo(desc.format).bytesPerBlock;
        if (bytesPerPixel == 0)
            return;

        const uint32_t baseMip = subresources.baseMipLevel;
        const uint32_t numMips = subresources.numMipLevels == uint32_t(-1) ? desc.mipLevels - baseMip : subresources.numMipLevels;
        const uint32_t baseSlice = subresources.baseArraySlice;
        const uint32_t numSlices = subresources.numArraySlices == uint32_t(-1) ? desc.arraySize - baseSlice : subresources.numArraySlices;

        // Metal clears a colour attachment through the render pass. That keeps the
        // pattern off the CPU, so it is much cheaper than copying the pattern in.
        // The path needs a render target usage and a colour format Metal can clear.
        const FormatInfo& info = getFormatInfo(desc.format);
        const bool clearWithRenderPass = !integer
            && ( info.kind == FormatKind::Float || info.kind == FormatKind::Normalized )
            && ( texture->usage & MTLTextureUsageRenderTarget )
            && convertFormat( desc.format ) != MTLPixelFormatInvalid;

        if ( clearWithRenderPass )
        {
            endEncoder();

            const MTLClearColor clearColor = MTLClearColorMake( color.r, color.g, color.b, color.a );

            for ( uint32_t mip = 0; mip < numMips; ++mip )
            {
                for ( uint32_t slice = 0; slice < numSlices; ++slice )
                {
                    MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
                    MTLRenderPassColorAttachmentDescriptor* attachment = passDescriptor.colorAttachments[0];
                    attachment.texture = texture->texture;
                    attachment.loadAction = MTLLoadActionClear;
                    attachment.storeAction = MTLStoreActionStore;
                    attachment.clearColor = clearColor;
                    attachment.level = baseMip + mip;
                    if ( desc.dimension == TextureDimension::Texture3D )
                        attachment.depthPlane = baseSlice + slice;
                    else
                        attachment.slice = baseSlice + slice;

                    id<MTLRenderCommandEncoder> encoder = [m_CommandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
                    [encoder endEncoding];
                }
            }

            return;
        }

        for (uint32_t mip = 0; mip < numMips; ++mip)
        {
            const uint32_t width = getMipSize(desc.width, baseMip + mip);
            const uint32_t height = getMipSize(desc.height, baseMip + mip);
            const uint32_t depth = desc.dimension == TextureDimension::Texture3D ? getMipSize(desc.depth, baseMip + mip) : 1;
            const uint64_t rowPitch = alignSize(uint64_t(width) * bytesPerPixel, kBlitRowAlignment);

            // The clear pattern is identical for every depth plane and every array slice.
            std::vector<uint8_t> data(rowPitch * height);
            fillClearData(data, rowPitch, width, height, desc.format, color, intValue, integer);
            data.resize(rowPitch * height * depth, 0);
            for (uint32_t z = 1; z < depth; ++z)
                memcpy(data.data() + z * rowPitch * height, data.data(), rowPitch * height);

            id<MTLBuffer> source = [m_Device->getContext().device newBufferWithBytes:data.data() length:data.size() options:MTLResourceStorageModeShared];
            if (!source)
                return;

            for (uint32_t slice = 0; slice < numSlices; ++slice)
            {
                [getBlitEncoder() copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:rowPitch sourceBytesPerImage:rowPitch * height
                    sourceSize:MTLSizeMake(width, height, depth)
                    toTexture:texture->texture destinationSlice:baseSlice + slice destinationLevel:baseMip + mip destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
        }
    }

    // ------------------------------------------------------------------ Graphics

    id<MTLRenderCommandEncoder> CommandList::getRenderEncoder(Framebuffer* framebuffer)
    {
        // The render pass holds the attachments, so a new framebuffer needs a new pass.
        if (m_RenderEncoder)
        {
            if (m_RenderEncoderFramebuffer == framebuffer)
                return m_RenderEncoder;

            endEncoder();
        }

        endEncoder();

        MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];

        for (uint32_t i = 0; i < framebuffer->desc.colorAttachments.size(); ++i)
        {
            const FramebufferAttachment& attachment = framebuffer->desc.colorAttachments[i];
            const Texture* texture = checked_cast<Texture*>(attachment.texture);
            if (!texture || !texture->texture)
                continue;

            MTLRenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor.colorAttachments[i];
            colorAttachment.texture = texture->texture;
            colorAttachment.loadAction = MTLLoadActionLoad;
            colorAttachment.storeAction = MTLStoreActionStore;

            // The attachment selects its mip, array slice or depth plane from the framebuffer.
            colorAttachment.level = attachment.subresources.baseMipLevel;
            if (texture->desc.dimension == TextureDimension::Texture3D)
                colorAttachment.depthPlane = attachment.subresources.baseArraySlice;
            else
                colorAttachment.slice = attachment.subresources.baseArraySlice;
        }

        if (framebuffer->desc.depthAttachment.texture)
        {
            const FramebufferAttachment& attachment = framebuffer->desc.depthAttachment;
            const Texture* texture = checked_cast<Texture*>(attachment.texture);
            if (texture && texture->texture)
            {
                passDescriptor.depthAttachment.texture = texture->texture;
                passDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
                passDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
                passDescriptor.depthAttachment.level = attachment.subresources.baseMipLevel;
                passDescriptor.depthAttachment.slice = attachment.subresources.baseArraySlice;

                // A combined depth and stencil format needs the stencil attachment as well.
                if (getFormatInfo(texture->desc.format).hasStencil)
                {
                    passDescriptor.stencilAttachment.texture = texture->texture;
                    passDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
                    passDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
                    passDescriptor.stencilAttachment.level = attachment.subresources.baseMipLevel;
                    passDescriptor.stencilAttachment.slice = attachment.subresources.baseArraySlice;
                }
            }
        }

        m_RenderEncoder = [m_CommandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
        m_RenderEncoderFramebuffer = framebuffer;
        m_RenderEncoderStateValid = false;
        return m_RenderEncoder;
    }

    void CommandList::setGraphicsState(const GraphicsState& state)
    {
        GraphicsPipeline* pso = checked_cast<GraphicsPipeline*>(state.pipeline);
        Framebuffer* framebuffer = checked_cast<Framebuffer*>(state.framebuffer);
        if (!pso || !framebuffer)
            return;

        m_CurrentGraphicsState = state;

        id<MTLRenderCommandEncoder> encoder = getRenderEncoder(framebuffer);
        if (!encoder)
            return;

        // A new encoder has no state, so everything is applied again. Otherwise only
        // the parts that changed are applied, like the Vulkan backend does.
        const bool fullUpdate = !m_RenderEncoderStateValid;
        const bool pipelineChanged = fullUpdate || m_CurrentGraphicsPipeline != pso;

        if (pipelineChanged)
        {
            [encoder setRenderPipelineState:pso->pipeline];

            const RasterState& rasterState = pso->desc.renderState.rasterState;
            [encoder setCullMode:convertCullMode(rasterState.cullMode)];
            [encoder setTriangleFillMode:rasterState.fillMode == RasterFillMode::Wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
            [encoder setFrontFacingWinding:rasterState.frontCounterClockwise ? MTLWindingCounterClockwise : MTLWindingClockwise];
            [encoder setDepthClipMode:rasterState.depthClipEnable ? MTLDepthClipModeClip : MTLDepthClipModeClamp];

            // Metal rejects a depth state when the pass has no depth attachment.
            if (framebuffer->desc.depthAttachment.texture)
                [encoder setDepthStencilState:pso->depthStencilState];

            if (rasterState.depthBias != 0 || rasterState.slopeScaledDepthBias != 0.f || rasterState.depthBiasClamp != 0.f)
            {
                [encoder setDepthBias:float(rasterState.depthBias) slopeScale:rasterState.slopeScaledDepthBias clamp:rasterState.depthBiasClamp];
            }
        }

        // Metal sets the stencil reference value on the encoder. The pipeline holds the
        // static value, and the state overrides it when the pipeline uses a dynamic one.
        const DepthStencilState& depthStencilState = pso->desc.renderState.depthStencilState;
        const uint32_t stencilRefValue = depthStencilState.dynamicStencilRef
            ? state.dynamicStencilRefValue
            : depthStencilState.stencilRefValue;
        if (fullUpdate || m_AppliedStencilRefValue != stencilRefValue)
        {
            [encoder setStencilReferenceValue:stencilRefValue];
            m_AppliedStencilRefValue = stencilRefValue;
        }

        if (fullUpdate || m_AppliedBlendConstantColor != state.blendConstantColor)
        {
            [encoder setBlendColorRed:state.blendConstantColor.r green:state.blendConstantColor.g blue:state.blendConstantColor.b alpha:state.blendConstantColor.a];
            m_AppliedBlendConstantColor = state.blendConstantColor;
        }

        m_IndexBuffer = state.indexBuffer;
        m_IndirectBuffer = checked_cast<Buffer*>(state.indirectParams);

        if (!state.viewport.viewports.empty())
        {
            const Viewport& viewport = state.viewport.viewports[0];
            const bool viewportChanged = fullUpdate
                || state.viewport.viewports.size() != m_AppliedViewportCount
                || m_AppliedViewport != viewport;

            if (viewportChanged)
            {
                MTLViewport metalViewport;
                metalViewport.originX = viewport.minX;
                metalViewport.originY = viewport.minY;
                metalViewport.width = viewport.maxX - viewport.minX;
                metalViewport.height = viewport.maxY - viewport.minY;
                metalViewport.znear = viewport.minZ;
                metalViewport.zfar = viewport.maxZ;
                [encoder setViewport:metalViewport];
                m_AppliedViewport = viewport;
                m_AppliedViewportCount = state.viewport.viewports.size();
            }
        }

        if (!state.viewport.scissorRects.empty())
        {
            const Rect& rect = state.viewport.scissorRects[0];
            const bool scissorChanged = fullUpdate
                || state.viewport.scissorRects.size() != m_AppliedScissorCount
                || m_AppliedScissor != rect;

            if (scissorChanged)
            {
                MTLScissorRect scissorRect;
                scissorRect.x = rect.minX;
                scissorRect.y = rect.minY;
                scissorRect.width = rect.maxX - rect.minX;
                scissorRect.height = rect.maxY - rect.minY;
                [encoder setScissorRect:scissorRect];
                m_AppliedScissor = rect;
                m_AppliedScissorCount = state.viewport.scissorRects.size();
            }
        }

        if (fullUpdate || arraysAreDifferent(m_AppliedVertexBuffers, state.vertexBuffers))
        {
            for (const auto& vertexBuffer : state.vertexBuffers)
            {
                const Buffer* buffer = checked_cast<Buffer*>(vertexBuffer.buffer);
                if (!buffer || !buffer->buffer)
                    continue;

                [encoder setVertexBuffer:buffer->buffer offset:vertexBuffer.offset atIndex:c_VertexBufferBase + vertexBuffer.slot];
            }
            m_AppliedVertexBuffers = state.vertexBuffers;
        }

        m_CurrentGraphicsPipeline = pso;

        // A new pipeline can map the same binding set to other argument slots.
        if (fullUpdate || pipelineChanged || arraysAreDifferent(m_AppliedBindings, state.bindings))
        {
            m_AppliedBindings = state.bindings;
            bindGraphicsResources(state);
        }

        m_RenderEncoderStateValid = true;
    }

    void CommandList::bindGraphicsResources(const GraphicsState& state)
    {
        for (uint32_t i = 0; i < state.bindings.size(); ++i)
        {
            nvrhi::IBindingSet* binding = state.bindings[i];
            if (!binding || !binding->getLayout())
                continue;

            const BindingLayout* layout = checked_cast<BindingLayout*>(binding->getLayout());

            // A descriptor table binds its slots as array elements of the bindless arrays.
            if (layout->isBindless)
            {
                bindDescriptorTable(checked_cast<DescriptorTable*>(binding), m_CurrentGraphicsPipeline->plan, i);
                continue;
            }

            const BindingSet* bindingSet = checked_cast<BindingSet*>(binding);
            const ShaderType stageMask = layout->desc.visibility;

            for (const auto& item : bindingSet->desc.bindings)
            {
                // One item can serve several stages, and each stage has its own
                // translated index, so every matching entry is bound.
                for (const auto& entry : m_CurrentGraphicsPipeline->plan.entries)
                {
                    if ((entry.stage & stageMask) == ShaderType::None)
                        continue;
                    if (entry.bindingIndex != i || entry.slot != item.slot)
                        continue;

                    BoundResource bound = entry.bound;
                    bound.index += item.arrayElement;
                    bindGraphicsResource(bound, item, entry.stage == ShaderType::Vertex);
                }
            }
        }
    }

    void CommandList::bindDescriptorTable(DescriptorTable* table, const PipelineBindingPlan& plan, uint32_t bindingIndex)
    {
        const BindingLayout* layout = checked_cast<BindingLayout*>(table->layout.Get());
        if (bindingIndex >= plan.descriptorSets.size())
            return;
        const uint32_t descriptorSet = plan.descriptorSets[bindingIndex];

        for (const auto& argumentBuffer : plan.argumentBuffers)
        {
            if (argumentBuffer.descriptorSet == descriptorSet)
                bindTableArgumentBuffer(table, plan, argumentBuffer);
        }

        for (size_t space = 0; space < table->slots.size(); ++space)
        {
            ResourceKind kind = ResourceKind::Buffer;
            bool writeable = false;
            if (!getBindlessArrayShape(layout->bindlessDesc.registerSpaces[space].type, kind, writeable))
                continue;

            // The register spaces of the layout take the bindings of the bindless
            // set in their order. The same kind and access can appear in several
            // stages, and each stage has its own translated index.
            const uint32_t binding = uint32_t(space);

            for (const ResourceBinding& array : plan.bindlessArrays)
            {
                if (array.descriptorSet != descriptorSet || array.kind != kind || array.writeable != writeable || array.binding != binding)
                    continue;

                const auto argumentBuffer = std::find_if(plan.argumentBuffers.begin(), plan.argumentBuffers.end(),
                    [&array](const PipelineBindingPlan::ArgumentBuffer& entry)
                    { return entry.stage == array.stage && entry.descriptorSet == array.descriptorSet; });
                if (argumentBuffer == plan.argumentBuffers.end())
                    bindTableArray(table->slots[space], array);
            }
        }
    }

    void CommandList::bindTableArgumentBuffer(DescriptorTable* table, const PipelineBindingPlan& plan, const PipelineBindingPlan::ArgumentBuffer& argumentBuffer)
    {
        id<MTLArgumentEncoder> encoder = [argumentBuffer.function newArgumentEncoderWithBufferIndex:argumentBuffer.index];
        if (!encoder)
        {
            m_Device->getContext().error("bindDescriptorTable: failed to create an argument encoder");
            return;
        }

        id<MTLBuffer> buffer = [m_Device->getContext().device newBufferWithLength:encoder.encodedLength options:MTLResourceStorageModeShared];
        if (!buffer)
        {
            m_Device->getContext().error("bindDescriptorTable: failed to allocate an argument buffer");
            return;
        }
        [encoder setArgumentBuffer:buffer offset:0];

        // Argument buffers do not retain their resources. Keep each resource alive
        // through submission and declare its usage to Metal before the dispatch or draw.
        std::set<IResource*> referenced;
        std::map<void*, MTLResourceUsage> resources;
        const BindingLayout* layout = checked_cast<BindingLayout*>(table->layout.Get());
        for (const ResourceBinding& array : plan.bindlessArrays)
        {
            if (array.stage != argumentBuffer.stage || array.descriptorSet != argumentBuffer.descriptorSet)
                continue;

            const std::vector<BindingSetItem>* slots = nullptr;
            if (array.binding < table->slots.size())
            {
                ResourceKind kind = ResourceKind::Buffer;
                bool writeable = false;
                const ResourceType type = layout->bindlessDesc.registerSpaces[array.binding].type;
                const bool typedBuffer = type == ResourceType::TypedBuffer_SRV || type == ResourceType::TypedBuffer_UAV;
                if (getBindlessArrayShape(type, kind, writeable) && array.kind == kind && array.writeable == writeable
                    && (kind != ResourceKind::Texture || array.bufferArray == typedBuffer))
                    slots = &table->slots[array.binding];
            }

            id<MTLBuffer> nullBuffer = nil;
            id<MTLTexture> nullTexture = nil;
            id<MTLSamplerState> nullSampler = nil;
            for (uint32_t slot = 0; slot < array.count; ++slot)
            {
                const BindingSetItem* item = slots && slot < slots->size() ? &(*slots)[slot] : nullptr;
                if (item && (item->type == ResourceType::None || !item->resourceHandle))
                    item = nullptr;

                id<MTLResource> resource = nil;
                switch (array.kind)
                {
                case ResourceKind::Buffer:
                {
                    Buffer* boundBuffer = item ? checked_cast<Buffer*>(item->resourceHandle) : nullptr;
                    id<MTLBuffer> value = boundBuffer ? boundBuffer->buffer : nil;
                    if (!value)
                    {
                        if (!nullBuffer)
                            nullBuffer = m_Device->getNullBuffer();
                        value = nullBuffer;
                    }
                    if (!value)
                    {
                        m_Device->getContext().error("bindDescriptorTable: failed to create a null buffer");
                        return;
                    }
                    [encoder setBuffer:value offset:boundBuffer && boundBuffer->buffer ? item->range.byteOffset : 0 atIndex:array.index + slot];
                    resource = value;
                    break;
                }
                case ResourceKind::Texture:
                {
                    id<MTLTexture> value = nil;
                    if (item && array.bufferArray)
                    {
                        Buffer* boundBuffer = checked_cast<Buffer*>(item->resourceHandle);
                        if (boundBuffer && boundBuffer->buffer)
                            value = getTypedBufferView(m_Device->getContext(), boundBuffer, item->format, item->range.byteOffset, item->range.byteSize);
                    }
                    else if (item)
                    {
                        Texture* texture = checked_cast<Texture*>(item->resourceHandle);
                        if (texture && texture->texture)
                        {
                            const ComponentMapping mapping = resolveComponentMapping(item->overrideComponentMapping, texture->desc.defaultComponentMapping);
                            value = texture->getSubresourceView(item->subresources, item->dimension, item->format, mapping);
                        }
                    }
                    if (!value)
                    {
                        if (!nullTexture)
                            nullTexture = array.bufferArray ? m_Device->getNullTextureBuffer(array.format) : m_Device->getNullTexture(array.dimension, array.format);
                        value = nullTexture;
                    }
                    if (!value)
                    {
                        m_Device->getContext().error("bindDescriptorTable: failed to create a null texture");
                        return;
                    }
                    [encoder setTexture:value atIndex:array.index + slot];
                    resource = value;
                    break;
                }
                case ResourceKind::Sampler:
                {
                    Sampler* sampler = item ? checked_cast<Sampler*>(item->resourceHandle) : nullptr;
                    id<MTLSamplerState> value = sampler ? sampler->getArgumentSampler(m_Device->getContext()) : nil;
                    if (!sampler)
                    {
                        if (!nullSampler)
                            nullSampler = m_Device->getNullSampler();
                        value = nullSampler;
                    }
                    if (!value)
                    {
                        m_Device->getContext().error("bindDescriptorTable: failed to create a sampler");
                        return;
                    }
                    [encoder setSamplerState:value atIndex:array.index + slot];
                    break;
                }
                }

                if (item && referenced.insert(item->resourceHandle).second)
                    m_ReferencedResources.push_back(RefCountPtr<IResource>(item->resourceHandle));
                if (resource)
                {
                    const MTLResourceUsage usage = array.writeable ? MTLResourceUsage(MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead;
                    auto& currentUsage = resources[(__bridge void*)resource];
                    currentUsage = MTLResourceUsage(currentUsage | usage);
                }
            }
        }

        for (const auto& entry : resources)
        {
            id<MTLResource> resource = (__bridge id<MTLResource>)entry.first;
            if (argumentBuffer.stage == ShaderType::Compute && m_ComputeEncoder)
                [m_ComputeEncoder useResource:resource usage:entry.second];
            else if (m_RenderEncoder)
            {
                const MTLRenderStages stage = argumentBuffer.stage == ShaderType::Vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
                [m_RenderEncoder useResource:resource usage:entry.second stages:stage];
            }
        }

        if (argumentBuffer.stage == ShaderType::Compute && m_ComputeEncoder)
            [m_ComputeEncoder setBuffer:buffer offset:0 atIndex:argumentBuffer.index];
        else if (m_RenderEncoder && argumentBuffer.stage == ShaderType::Vertex)
            [m_RenderEncoder setVertexBuffer:buffer offset:0 atIndex:argumentBuffer.index];
        else if (m_RenderEncoder && argumentBuffer.stage == ShaderType::Pixel)
            [m_RenderEncoder setFragmentBuffer:buffer offset:0 atIndex:argumentBuffer.index];
    }

    void CommandList::bindTableArray(const std::vector<nvrhi::BindingSetItem>& slots, const ResourceBinding& array)
    {
        // A shader array argument must be bound completely, so the unwritten slots
        // of the array are tracked and filled with null resources afterwards.
        std::vector<bool> written(array.count, false);

        for (const auto& item : slots)
        {
            if (item.type == ResourceType::None || !item.resourceHandle)
                continue;

            if (item.slot >= array.count)
                continue;

            BoundResource bound;
            bound.kind = array.kind;
            bound.index = array.index + item.slot;
            written[item.slot] = true;

            switch (array.stage)
            {
            case ShaderType::Compute:
                bindResource(bound, item);
                break;
            case ShaderType::Vertex:
                bindGraphicsResource(bound, item, true);
                break;
            case ShaderType::Pixel:
                bindGraphicsResource(bound, item, false);
                break;
            default:
                bindResource(bound, item);
                bindGraphicsResource(bound, item, true);
                bindGraphicsResource(bound, item, false);
                break;
            }
        }

        for (uint32_t slot = 0; slot < array.count; ++slot)
        {
            if (!written[slot])
                bindNullResource(array.kind, array.index + slot, array.bufferArray, array.format, array.dimension, array.stage);
        }
    }

    void CommandList::bindNullResource(ResourceKind kind, uint32_t index, bool bufferArray, Format format, TextureDimension dimension, ShaderType stage)
    {
        const bool bindCompute = stage == ShaderType::Compute || stage == ShaderType::None;
        const bool bindVertex = stage == ShaderType::Vertex || stage == ShaderType::None;
        const bool bindPixel = stage == ShaderType::Pixel || stage == ShaderType::None;

        switch (kind)
        {
        case ResourceKind::Buffer:
            if (bindCompute && m_ComputeEncoder)
                [m_ComputeEncoder setBuffer:m_Device->getNullBuffer() offset:0 atIndex:index];
            if (m_RenderEncoder)
            {
                if (bindVertex)
                    [m_RenderEncoder setVertexBuffer:m_Device->getNullBuffer() offset:0 atIndex:index];
                if (bindPixel)
                    [m_RenderEncoder setFragmentBuffer:m_Device->getNullBuffer() offset:0 atIndex:index];
            }
            break;
        case ResourceKind::Texture:
        {
            // A typed buffer array holds texture buffers, so its null resource is one.
            id<MTLTexture> nullTexture = bufferArray
                ? m_Device->getNullTextureBuffer(format)
                : m_Device->getNullTexture(dimension, format);
            if (!nullTexture)
                return;

            if (bindCompute && m_ComputeEncoder)
                [m_ComputeEncoder setTexture:nullTexture atIndex:index];
            if (m_RenderEncoder)
            {
                if (bindVertex)
                    [m_RenderEncoder setVertexTexture:nullTexture atIndex:index];
                if (bindPixel)
                    [m_RenderEncoder setFragmentTexture:nullTexture atIndex:index];
            }
            break;
        }
        case ResourceKind::Sampler:
            if (bindCompute && m_ComputeEncoder)
                [m_ComputeEncoder setSamplerState:m_Device->getNullSampler() atIndex:index];
            if (m_RenderEncoder)
            {
                if (bindVertex)
                    [m_RenderEncoder setVertexSamplerState:m_Device->getNullSampler() atIndex:index];
                if (bindPixel)
                    [m_RenderEncoder setFragmentSamplerState:m_Device->getNullSampler() atIndex:index];
            }
            break;
        }
    }

    void CommandList::bindGraphicsResource(const BoundResource& bound, const nvrhi::BindingSetItem& item, bool vertexStage)
    {
        if (!m_RenderEncoder)
            return;

        switch (bound.kind)
        {
        case ResourceKind::Buffer:
        {
            const bool isBuffer = item.type == ResourceType::ConstantBuffer || item.type == ResourceType::VolatileConstantBuffer
                || item.type == ResourceType::StructuredBuffer_SRV || item.type == ResourceType::StructuredBuffer_UAV
                || item.type == ResourceType::RawBuffer_SRV || item.type == ResourceType::RawBuffer_UAV;
            if (!isBuffer)
                return;

            const Buffer* buffer = checked_cast<Buffer*>(item.resourceHandle);
            if (!buffer || !buffer->buffer)
                return;

            if (vertexStage)
                [m_RenderEncoder setVertexBuffer:buffer->buffer offset:item.range.byteOffset atIndex:bound.index];
            else
                [m_RenderEncoder setFragmentBuffer:buffer->buffer offset:item.range.byteOffset atIndex:bound.index];
            break;
        }
        case ResourceKind::Texture:
        {
            // A typed buffer is a texture in the shader but a buffer in NVRHI.
            if (item.type == ResourceType::TypedBuffer_SRV || item.type == ResourceType::TypedBuffer_UAV)
            {
                Buffer* buffer = checked_cast<Buffer*>(item.resourceHandle);
                if (!buffer || !buffer->buffer)
                    return;

                id<MTLTexture> view = getTypedBufferView(m_Device->getContext(), buffer, item.format, item.range.byteOffset, item.range.byteSize);
                if (view)
                {
                    if (vertexStage)
                        [m_RenderEncoder setVertexTexture:view atIndex:bound.index];
                    else
                        [m_RenderEncoder setFragmentTexture:view atIndex:bound.index];
                }
                break;
            }

            const bool isTexture = item.type == ResourceType::Texture_SRV || item.type == ResourceType::Texture_UAV;
            if (!isTexture)
                return;

            Texture* texture = checked_cast<Texture*>(item.resourceHandle);
            if (!texture || !texture->texture)
                return;

            const ComponentMapping componentMapping = resolveComponentMapping(item.overrideComponentMapping, texture->desc.defaultComponentMapping);
            id<MTLTexture> view = texture->getSubresourceView(item.subresources, item.dimension, item.format, componentMapping);
            if (view)
            {
                if (vertexStage)
                    [m_RenderEncoder setVertexTexture:view atIndex:bound.index];
                else
                    [m_RenderEncoder setFragmentTexture:view atIndex:bound.index];
            }
            break;
        }
        case ResourceKind::Sampler:
        {
            const Sampler* sampler = checked_cast<Sampler*>(item.resourceHandle);
            if (!sampler || !sampler->sampler)
                return;

            if (vertexStage)
                [m_RenderEncoder setVertexSamplerState:sampler->sampler atIndex:bound.index];
            else
                [m_RenderEncoder setFragmentSamplerState:sampler->sampler atIndex:bound.index];
            break;
        }
        }
    }

    bool CommandList::reopenRenderEncoder()
    {
        if (!m_RenderEncoderFramebuffer)
            return false;

        setGraphicsState(m_CurrentGraphicsState);
        return m_RenderEncoder != nil;
    }

    void CommandList::draw(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsPipeline)
            return;

        if (!m_RenderEncoder && !reopenRenderEncoder())
            return;

        if (m_CurrentGraphicsPipeline->vertexPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Vertex);
        if (m_CurrentGraphicsPipeline->pixelPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Pixel);

        [m_RenderEncoder drawPrimitives:m_CurrentGraphicsPipeline->primitiveType
            vertexStart:args.startVertexLocation vertexCount:args.vertexCount instanceCount:args.instanceCount];
    }

    void CommandList::drawIndexed(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsPipeline)
            return;

        if (!m_RenderEncoder && !reopenRenderEncoder())
            return;

        if (m_CurrentGraphicsPipeline->vertexPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Vertex);
        if (m_CurrentGraphicsPipeline->pixelPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Pixel);

        const Buffer* indexBuffer = checked_cast<Buffer*>(m_IndexBuffer.buffer);
        if (!indexBuffer || !indexBuffer->buffer)
            return;

        const MTLIndexType indexType = m_IndexBuffer.format == Format::R32_UINT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;

        // The start index location counts the indices, so it is added to the buffer offset.
        const uint64_t indexSize = indexType == MTLIndexTypeUInt32 ? 4 : 2;

        [m_RenderEncoder drawIndexedPrimitives:m_CurrentGraphicsPipeline->primitiveType
            indexCount:args.vertexCount indexType:indexType indexBuffer:indexBuffer->buffer
            indexBufferOffset:m_IndexBuffer.offset + args.startIndexLocation * indexSize
            instanceCount:args.instanceCount baseVertex:args.startVertexLocation baseInstance:args.startInstanceLocation];
    }

    void CommandList::drawIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsPipeline || !m_IndirectBuffer)
            return;

        if (!m_RenderEncoder && !reopenRenderEncoder())
            return;

        if (m_CurrentGraphicsPipeline->vertexPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Vertex);
        if (m_CurrentGraphicsPipeline->pixelPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Pixel);

        for (uint32_t draw = 0; draw < drawCount; ++draw)
        {
            [m_RenderEncoder drawPrimitives:m_CurrentGraphicsPipeline->primitiveType
                indirectBuffer:m_IndirectBuffer->buffer indirectBufferOffset:offsetBytes + draw * 16];
        }
    }

    void CommandList::drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsPipeline || !m_IndexBuffer.buffer || !m_IndirectBuffer)
            return;

        if (!m_RenderEncoder && !reopenRenderEncoder())
            return;

        if (m_CurrentGraphicsPipeline->vertexPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Vertex);
        if (m_CurrentGraphicsPipeline->pixelPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Pixel);

        const Buffer* indexBuffer = checked_cast<Buffer*>(m_IndexBuffer.buffer);
        const MTLIndexType indexType = m_IndexBuffer.format == Format::R32_UINT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;

        for (uint32_t draw = 0; draw < drawCount; ++draw)
        {
            [m_RenderEncoder drawIndexedPrimitives:m_CurrentGraphicsPipeline->primitiveType
                indexType:indexType indexBuffer:indexBuffer->buffer indexBufferOffset:m_IndexBuffer.offset
                indirectBuffer:m_IndirectBuffer->buffer indirectBufferOffset:offsetBytes + draw * 20];
        }
    }

    id<MTLBuffer> CommandList::expandIndirectDrawArgs(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        const Buffer* countBuffer = checked_cast<Buffer*>(m_CurrentGraphicsState.indirectCountBuffer);
        if (!countBuffer || !countBuffer->buffer || !m_IndirectBuffer || !m_IndirectBuffer->buffer)
            return nil;

        const uint64_t argsSize = uint64_t(maxDrawCount) * 20;
        if ((paramOffsetBytes & 3) || (countOffsetBytes & 3) || argsSize > UINT32_MAX ||
            paramOffsetBytes > m_IndirectBuffer->desc.byteSize ||
            argsSize > m_IndirectBuffer->desc.byteSize - paramOffsetBytes ||
            countOffsetBytes > countBuffer->desc.byteSize ||
            sizeof(uint32_t) > countBuffer->desc.byteSize - countOffsetBytes)
        {
            m_Device->getContext().error("drawIndexedIndirectCount: invalid argument or count buffer range");
            return nil;
        }

        id<MTLComputePipelineState> pipeline = m_Device->getIndirectCountPipeline();
        if (!pipeline)
            return nil;

        // The arguments of the counted draws are expanded into a scratch buffer, and
        // the commands that the count excludes are zeroed. Every counted draw gets its
        // own buffers, so a draw that is still running is not overwritten.
        id<MTLBuffer> argsBuffer = [m_Device->getContext().device newBufferWithLength:argsSize
            options:MTLResourceStorageModePrivate];
        id<MTLBuffer> paramsBuffer = [m_Device->getContext().device newBufferWithLength:16
            options:MTLResourceStorageModeShared];
        if (!argsBuffer || !paramsBuffer)
            return nil;

        const uint32_t offsets[3] = { paramOffsetBytes, countOffsetBytes, maxDrawCount };
        std::memcpy(paramsBuffer.contents, offsets, sizeof(offsets));

        id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
        if (!encoder)
            return nil;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:m_IndirectBuffer->buffer offset:0 atIndex:0];
        [encoder setBuffer:countBuffer->buffer offset:0 atIndex:1];
        [encoder setBuffer:argsBuffer offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(maxDrawCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        m_ComputeEncoder = nil;

        return argsBuffer;
    }

    void CommandList::drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        if (!m_CurrentGraphicsPipeline || !m_IndexBuffer.buffer || !m_IndirectBuffer || maxDrawCount == 0)
            return;

        // Metal has no counted indirect draw, so the arguments are expanded on the GPU.
        // The helper kernel needs its own encoder, so it runs before the render encoder
        // is opened and the state is applied.
        id<MTLBuffer> argsBuffer = expandIndirectDrawArgs(paramOffsetBytes, countOffsetBytes, maxDrawCount);
        if (!argsBuffer)
            return;

        if (!m_RenderEncoder && !reopenRenderEncoder())
            return;

        if (m_CurrentGraphicsPipeline->vertexPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Vertex);
        if (m_CurrentGraphicsPipeline->pixelPushConstants)
            bindPushConstants(m_CurrentGraphicsPipeline->pushConstantBufferIndex, m_CurrentGraphicsPipeline->pushConstantSize, ShaderType::Pixel);

        const Buffer* indexBuffer = checked_cast<Buffer*>(m_IndexBuffer.buffer);
        const MTLIndexType indexType = m_IndexBuffer.format == Format::R32_UINT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;

        for (uint32_t draw = 0; draw < maxDrawCount; ++draw)
        {
            [m_RenderEncoder drawIndexedPrimitives:m_CurrentGraphicsPipeline->primitiveType
                indexType:indexType indexBuffer:indexBuffer->buffer indexBufferOffset:m_IndexBuffer.offset
                indirectBuffer:argsBuffer indirectBufferOffset:draw * 20];
        }
    }

    void CommandList::setPushConstants(const void* data, size_t byteSize)
    {
        const uint8_t* bytes = (const uint8_t*)data;
        m_PushConstants.assign(bytes, bytes + byteSize);
    }

    void CommandList::bindPushConstants(uint32_t bufferIndex, uint32_t size, ShaderType stage)
    {
        if (bufferIndex == UINT32_MAX || size == 0 || m_PushConstants.empty())
            return;

        // The bytes are set on the encoder instead of a buffer, so a later setPushConstants
        // cannot change what an earlier encoded dispatch or draw has already received.
        const void* data = m_PushConstants.data();
        const NSUInteger length = std::min(size, uint32_t(m_PushConstants.size()));

        if (stage == ShaderType::Compute)
        {
            if (m_ComputeEncoder)
                [m_ComputeEncoder setBytes:data length:length atIndex:bufferIndex];
        }
        else if (stage == ShaderType::Vertex)
        {
            if (m_RenderEncoder)
                [m_RenderEncoder setVertexBytes:data length:length atIndex:bufferIndex];
        }
        else
        {
            if (m_RenderEncoder)
                [m_RenderEncoder setFragmentBytes:data length:length atIndex:bufferIndex];
        }
    }

    // ------------------------------------------------------------------ Unimplemented

    void CommandList::clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture)
    {
        (void)texture;
    }

    void CommandList::decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, nvrhi::Format format)
    {
        (void)buffer;
        (void)texture;
        (void)format;
    }

    void CommandList::setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits)
    {
        (void)texture;
        (void)stateBits;
    }

    void CommandList::setMeshletState(const MeshletState& state)
    {
        (void)state;
    }

    void CommandList::dispatchMesh(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        (void)groupsX;
        (void)groupsY;
        (void)groupsZ;
    }

    void CommandList::dispatchMeshIndirect(uint32_t offsetBytes, uint32_t maxDrawCount)
    {
        (void)offsetBytes;
        (void)maxDrawCount;
    }

    void CommandList::dispatchMeshIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        (void)paramOffsetBytes;
        (void)countOffsetBytes;
        (void)maxDrawCount;
    }

    void CommandList::setRayTracingState(const rt::State& state)
    {
        (void)state;
    }

    void CommandList::dispatchRays(const rt::DispatchRaysArguments& args)
    {
        (void)args;
    }

    void CommandList::buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc)
    {
        (void)omm;
        (void)desc;
    }

    void CommandList::buildBottomLevelAccelStruct(rt::IAccelStruct* as, const rt::GeometryDesc* pGeometries, size_t numGeometries, rt::AccelStructBuildFlags buildFlags)
    {
        (void)as;
        (void)pGeometries;
        (void)numGeometries;
        (void)buildFlags;
    }

    void CommandList::compactBottomLevelAccelStructs()
    {
    }

    void CommandList::copyRaytracingAccelerationStructure(rt::IAccelStruct* destination, rt::IAccelStruct* source)
    {
        (void)destination;
        (void)source;
    }

    void CommandList::buildTopLevelAccelStruct(rt::IAccelStruct* as, const rt::InstanceDesc* pInstances, size_t numInstances, rt::AccelStructBuildFlags buildFlags)
    {
        (void)as;
        (void)pInstances;
        (void)numInstances;
        (void)buildFlags;
    }

    void CommandList::executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc)
    {
        (void)desc;
    }

    void CommandList::buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as, nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset, size_t numInstances, rt::AccelStructBuildFlags buildFlags)
    {
        (void)as;
        (void)instanceBuffer;
        (void)instanceBufferOffset;
        (void)numInstances;
        (void)buildFlags;
    }

    void CommandList::convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs)
    {
        (void)convertDescs;
        (void)numDescs;
    }

    // Opens and closes an empty render pass that samples a timestamp at one of its
    // stage boundaries. A render pass attachment is the only sampling point this
    // driver generation honours; a compute pass attachment resolves to zeros.
    void CommandList::sampleTimestamp(TimerQuery* query, bool atPassEnd)
    {
        MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        passDescriptor.colorAttachments[0].texture = m_Device->getTimerQuerySampleTarget();
        passDescriptor.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        passDescriptor.colorAttachments[0].storeAction = MTLStoreActionDontCare;

        MTLRenderPassSampleBufferAttachmentDescriptor* attachment = passDescriptor.sampleBufferAttachments[0];
        attachment.sampleBuffer = m_Device->getCounterSampleBuffer();
        attachment.startOfVertexSampleIndex = atPassEnd ? MTLCounterDontSample : query->beginSampleIndex;
        attachment.endOfFragmentSampleIndex = atPassEnd ? query->endSampleIndex : MTLCounterDontSample;

        id<MTLRenderCommandEncoder> encoder = [m_CommandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
        [encoder endEncoding];
    }

    void CommandList::beginTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);
        if (!query || !m_CommandBuffer || !m_Device->getCounterSampleBuffer())
            return;

        // The sample goes at the start of an empty pass, so it measures everything
        // the command buffer runs between begin and end.
        endEncoder();

        sampleTimestamp(query, false);

        {
            std::lock_guard< std::mutex > lock(query->mutex);
            query->serial = 0;
            query->started = false;
            query->resolved = false;
            query->failed = false;
        }
        m_SampledTimerQueries.push_back(RefCountPtr<TimerQuery>(query));
    }

    void CommandList::endTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);
        if (!query || !m_CommandBuffer || !m_Device->getCounterSampleBuffer())
            return;

        endEncoder();

        sampleTimestamp(query, true);

        std::lock_guard< std::mutex > lock(query->mutex);
        query->started = true;
    }

    void CommandList::beginMarker(const char* name)
    {
        (void)name;
    }

    void CommandList::endMarker()
    {
    }

    void CommandList::setEnableAutomaticBarriers(bool enable)
    {
        (void)enable;
    }

    void CommandList::setResourceStatesForBindingSet(IBindingSet* bindingSet)
    {
        (void)bindingSet;
    }

    void CommandList::setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers)
    {
        (void)texture;
        (void)enableBarriers;
    }

    void CommandList::setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers)
    {
        (void)buffer;
        (void)enableBarriers;
    }

    void CommandList::beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        (void)texture;
        (void)subresources;
        (void)stateBits;
    }

    void CommandList::beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits)
    {
        (void)buffer;
        (void)stateBits;
    }

    void CommandList::setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        (void)subresources;
        (void)stateBits;

        Texture* tex = checked_cast<Texture*>(texture);
        if (tex)
            m_ReferencedResources.push_back(tex);
    }

    void CommandList::setBufferState(IBuffer* buffer, ResourceStates stateBits)
    {
        (void)stateBits;

        Buffer* buf = checked_cast<Buffer*>(buffer);
        if (buf)
            m_ReferencedResources.push_back(buf);
    }

    void CommandList::setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits)
    {
        (void)as;
        (void)stateBits;
    }

    void CommandList::setPermanentTextureState(ITexture* _texture, ResourceStates stateBits)
    {
        Texture* texture = checked_cast<Texture*>(_texture);
        if (texture)
            m_PermanentTextureStates.push_back({ texture, stateBits });
    }

    void CommandList::setPermanentBufferState(IBuffer* _buffer, ResourceStates stateBits)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);
        if (buffer)
            m_PermanentBufferStates.push_back({ buffer, stateBits });
    }

    void CommandList::applyPermanentStates()
    {
        for (auto& [texture, state] : m_PermanentTextureStates)
        {
            if (texture->permanentState != ResourceStates::Unknown && texture->permanentState != state)
            {
                m_Device->getContext().error("Attempted to switch permanent state of texture " + texture->descRef.debugName);
                continue;
            }

            texture->permanentState = state;
        }
        m_PermanentTextureStates.clear();

        for (auto& [buffer, state] : m_PermanentBufferStates)
        {
            if (buffer->permanentState != ResourceStates::Unknown && buffer->permanentState != state)
            {
                m_Device->getContext().error("Attempted to switch permanent state of buffer " + buffer->descRef.debugName);
                continue;
            }

            buffer->permanentState = state;
        }
        m_PermanentBufferStates.clear();
    }

    void CommandList::commitBarriers()
    {
    }

    ResourceStates CommandList::getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel)
    {
        (void)texture;
        (void)arraySlice;
        (void)mipLevel;
        return ResourceStates::Unknown;
    }

    ResourceStates CommandList::getBufferState(IBuffer* buffer)
    {
        (void)buffer;
        return ResourceStates::Unknown;
    }

    IDevice* CommandList::getDevice()
    {
        return m_Device;
    }

    const CommandListParameters& CommandList::getDesc()
    {
        return m_Params;
    }

    CommandListMemoryUsage CommandList::getInternalMemoryUsage() const
    {
        return {};
    }

    Object CommandList::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_CommandBuffer)
            return Object((__bridge void*)m_CommandBuffer);

        return nullptr;
    }
}
