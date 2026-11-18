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

namespace nvrhi::metal
{
    void Shader::getBytecode(const void** ppBytecode, size_t* pSize) const
    {
        if (ppBytecode)
            *ppBytecode = source.c_str();
        if (pSize)
            *pSize = source.size();
    }

    Object Shader::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_ShaderLibrary)
            return Object((__bridge void*)library);

        return nullptr;
    }

    Object ComputePipeline::getNativeObject(ObjectType objectType)
    {
        (void)objectType;
        return nullptr;
    }

    Object GraphicsPipeline::getNativeObject(ObjectType objectType)
    {
        (void)objectType;
        return nullptr;
    }

    const VertexAttributeDesc* InputLayout::getAttributeDesc(uint32_t index) const
    {
        if (index >= attributes.size())
            return nullptr;

        return &attributes[index];
    }

    Object Buffer::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_Buffer)
            return Object((__bridge void*)buffer);

        return nullptr;
    }

    Object Texture::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_Texture)
            return Object((__bridge void*)texture);

        return nullptr;
    }

    namespace
    {
        MTLTextureType convertViewTextureType(TextureDimension dimension)
        {
            switch (dimension)
            {
            case TextureDimension::Texture1D:        return MTLTextureType1D;
            case TextureDimension::Texture1DArray:   return MTLTextureType1DArray;
            case TextureDimension::Texture2DArray:   return MTLTextureType2DArray;
            case TextureDimension::Texture2DMS:      return MTLTextureType2DMultisample;
            case TextureDimension::Texture2DMSArray: return MTLTextureType2DMultisampleArray;
            case TextureDimension::Texture3D:        return MTLTextureType3D;
            case TextureDimension::TextureCube:      return MTLTextureTypeCube;
            case TextureDimension::TextureCubeArray: return MTLTextureTypeCubeArray;
            default:                                 return MTLTextureType2D;
            }
        }

        MTLTextureSwizzle convertComponentSwizzle(ComponentSwizzle swizzle)
        {
            switch (swizzle)
            {
            case ComponentSwizzle::Zero: return MTLTextureSwizzleZero;
            case ComponentSwizzle::One:  return MTLTextureSwizzleOne;
            case ComponentSwizzle::R:    return MTLTextureSwizzleRed;
            case ComponentSwizzle::G:    return MTLTextureSwizzleGreen;
            case ComponentSwizzle::B:    return MTLTextureSwizzleBlue;
            default:                     return MTLTextureSwizzleAlpha;
            }
        }
    }

    id<MTLTexture> Texture::getSubresourceView(const TextureSubresourceSet& subresources, TextureDimension dimension, Format format, ComponentMapping componentMapping)
    {
        const TextureSubresourceSet resolved = subresources.resolve(desc, true);
        const Format viewFormat = format == Format::UNKNOWN ? desc.format : format;
        const ComponentMapping viewMapping = componentMapping.isIdentity() ? desc.defaultComponentMapping : componentMapping;

        if (subresources.isEntireTexture(desc) && viewFormat == desc.format && viewMapping.isIdentity())
            return texture;

        const uint16_t packedMapping = uint16_t(viewMapping.r) | (uint16_t(viewMapping.g) << 4) | (uint16_t(viewMapping.b) << 8) | (uint16_t(viewMapping.a) << 12);

        ViewKey key;
        key.baseMipLevel = resolved.baseMipLevel;
        key.numMipLevels = resolved.numMipLevels;
        key.baseArraySlice = resolved.baseArraySlice;
        key.numArraySlices = resolved.numArraySlices;
        key.dimension = dimension;
        key.format = viewFormat;
        key.componentMapping = packedMapping;

        const auto cached = views.find(key);
        if (cached != views.end())
            return cached->second;

        MTLTextureViewDescriptor* descriptor = [[MTLTextureViewDescriptor alloc] init];
        descriptor.pixelFormat = convertFormat(viewFormat);
        descriptor.textureType = dimension == TextureDimension::Unknown ? texture.textureType : convertViewTextureType(dimension);
        descriptor.levelRange = NSMakeRange(resolved.baseMipLevel, resolved.numMipLevels);
        descriptor.sliceRange = NSMakeRange(resolved.baseArraySlice, resolved.numArraySlices);
        descriptor.swizzle = MTLTextureSwizzleChannelsMake(
            convertComponentSwizzle(viewMapping.r), convertComponentSwizzle(viewMapping.g),
            convertComponentSwizzle(viewMapping.b), convertComponentSwizzle(viewMapping.a));

        id<MTLTexture> view = [texture newTextureViewWithDescriptor:descriptor];
        views.emplace(key, view);
        return view;
    }

    Object Texture::getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool isReadOnlyDSV,
        std::optional<ComponentMapping> overrideComponentMapping)
    {
        (void)isReadOnlyDSV;

        if (objectType == ObjectTypes::Metal_Texture)
        {
            const ComponentMapping mapping = overrideComponentMapping ? *overrideComponentMapping : desc.defaultComponentMapping;
            return Object((__bridge void*)getSubresourceView(subresources, dimension, format, mapping));
        }

        return nullptr;
    }

    Object Sampler::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_Sampler)
            return Object((__bridge void*)sampler);

        return nullptr;
    }

    namespace
    {
        // The push constants of a pipeline live in their own buffer.
        void getPushConstantInfo(uint32_t& bufferIndex, uint32_t& size, const BindingLayoutVector& layouts, bool usesPushConstants)
        {
            if (!usesPushConstants)
                return;

            bufferIndex = 0;

            for (const auto& layoutHandle : layouts)
            {
                const BindingLayout* layout = checked_cast<BindingLayout*>(layoutHandle.Get());
                if (!layout)
                    continue;

                for (const auto& item : layout->desc.bindings)
                {
                    if (item.type == ResourceType::PushConstants)
                        size = uint32_t(item.size);
                }
            }
        }

        // The pipeline binds the resources of its shaders; the descriptor set of a resource
        // is the position of its binding layout in the pipeline.
        void buildBindingPlan(PipelineBindingPlan& plan, const BindingLayoutVector& layouts, const Shader* shader)
        {
            for (uint32_t descriptorSet = 0; descriptorSet < uint32_t(layouts.size()); ++descriptorSet)
            {
                const BindingLayout* layout = checked_cast<BindingLayout*>(layouts[descriptorSet].Get());
                if (!layout)
                    continue;

                for (const auto& item : layout->desc.bindings)
                {
                    for (const auto& resource : shader->translated.resources)
                    {
                        if (resource.descriptorSet != descriptorSet || resource.binding != item.slot)
                            continue;

                        PipelineBindingPlan::Entry entry;
                        entry.stage = shader->desc.shaderType;
                        entry.descriptorSet = descriptorSet;
                        entry.slot = item.slot;
                        entry.bound.kind = resource.kind;
                        entry.bound.index = resource.index;
                        plan.entries.push_back(entry);
                    }
                }
            }
        }
    }

    ShaderHandle Device::createShader(const ShaderDesc& d, const void* binary, size_t binarySize)
    {
        TranslatedShader translated;
        std::string error;
        if (!translateSpirvToMSL(binary, binarySize, d.shaderType, d.entryName, m_Context.mslVersionMajor, m_Context.mslVersionMinor, translated, error))
        {
            m_Context.error("createShader: failed to translate '" + d.debugName + "': " + error);
            return nullptr;
        }

        NSError* nsError = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [m_Context.device newLibraryWithSource:@(translated.source.c_str()) options:options error:&nsError];
        if (!library)
        {
            m_Context.error("createShader: failed to compile '" + d.debugName + "': " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nullptr;
        }

        // SPIRV-Cross renames the entry point when it collides with a reserved name.
        id<MTLFunction> function = [library newFunctionWithName:@(d.entryName.c_str())];
        if (!function)
        {
            const std::string renamedEntry = d.entryName + "0";
            function = [library newFunctionWithName:@(renamedEntry.c_str())];
        }

        if (!function)
        {
            m_Context.error("createShader: the entry point '" + d.entryName + "' is missing in '" + d.debugName + "'");
            return nullptr;
        }

        Shader* shader = new Shader();
        shader->desc = d;
        shader->library = library;
        shader->function = function;
        shader->source = translated.source;
        shader->translated = translated;

        return ShaderHandle::Create(shader);
    }

    ShaderHandle Device::createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants)
    {
        (void)baseShader;
        (void)constants;
        (void)numConstants;
        m_Context.error("createShaderSpecialization: specialization constants are not supported by the Metal backend");
        return nullptr;
    }

    ShaderLibraryHandle Device::createShaderLibrary(const void* binary, size_t binarySize)
    {
        (void)binary;
        (void)binarySize;
        m_Context.error("createShaderLibrary: shader libraries are not supported by the Metal backend");
        return nullptr;
    }

    ComputePipelineHandle Device::createComputePipeline(const ComputePipelineDesc& desc)
    {
        if (!desc.CS)
        {
            m_Context.error("createComputePipeline: the compute shader is missing");
            return nullptr;
        }

        const Shader* shader = checked_cast<Shader*>(desc.CS.Get());
        if (!shader || !shader->function)
        {
            m_Context.error("createComputePipeline: the compute shader is not a Metal shader");
            return nullptr;
        }

        NSError* nsError = nil;
        id<MTLComputePipelineState> pipeline = [m_Context.device newComputePipelineStateWithFunction:shader->function error:&nsError];
        if (!pipeline)
        {
            m_Context.error("createComputePipeline: failed to create the pipeline: " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nullptr;
        }

        ComputePipeline* result = new ComputePipeline();
        result->desc = desc;
        result->pipeline = pipeline;
        result->threadGroupSize[0] = shader->translated.threadGroupSize[0];
        result->threadGroupSize[1] = shader->translated.threadGroupSize[1];
        result->threadGroupSize[2] = shader->translated.threadGroupSize[2];

        buildBindingPlan(result->plan, desc.bindingLayouts, shader);
        getPushConstantInfo(result->pushConstantBufferIndex, result->pushConstantSize, desc.bindingLayouts, shader->translated.usesPushConstants);

        return ComputePipelineHandle::Create(result);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        const Shader* vertexShader = checked_cast<Shader*>(desc.VS.Get());
        const Shader* pixelShader = checked_cast<Shader*>(desc.PS.Get());
        if (!vertexShader || !pixelShader || !vertexShader->function || !pixelShader->function)
        {
            m_Context.error("createGraphicsPipeline: the vertex and pixel shaders are required");
            return nullptr;
        }

        MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.vertexFunction = vertexShader->function;
        pipelineDescriptor.fragmentFunction = pixelShader->function;

        const InputLayout* inputLayout = checked_cast<InputLayout*>(desc.inputLayout.Get());
        if (inputLayout && inputLayout->getNumAttributes() > 0)
        {
            const uint32_t numAttributes = inputLayout->getNumAttributes();
            uint32_t numBufferSlots = 0;
            for (uint32_t i = 0; i < numAttributes; ++i)
                numBufferSlots = std::max(numBufferSlots, inputLayout->getAttributeDesc(i)->bufferIndex + 1);

            std::vector<MTLVertexBufferLayoutDescriptor*> bufferLayouts(numBufferSlots, nil);
            NSMutableArray<MTLVertexAttributeDescriptor*>* attributeDescriptors = [NSMutableArray arrayWithCapacity:numAttributes];

            for (uint32_t i = 0; i < numAttributes; ++i)
            {
                const VertexAttributeDesc& attribute = *inputLayout->getAttributeDesc(i);

                MTLVertexBufferLayoutDescriptor* bufferLayout = bufferLayouts[attribute.bufferIndex];
                if (!bufferLayout)
                {
                    bufferLayout = [[MTLVertexBufferLayoutDescriptor alloc] init];
                    bufferLayout.stride = attribute.elementStride;
                    bufferLayout.stepFunction = attribute.isInstanced ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                    bufferLayouts[attribute.bufferIndex] = bufferLayout;
                }

                MTLVertexAttributeDescriptor* attributeDescriptor = [[MTLVertexAttributeDescriptor alloc] init];
                attributeDescriptor.format = convertVertexFormat(attribute.format);
                attributeDescriptor.offset = attribute.offset;
                attributeDescriptor.bufferIndex = c_VertexBufferBase + attribute.bufferIndex;
                [attributeDescriptors addObject:attributeDescriptor];
            }

            MTLVertexDescriptor* vertexDescriptor = [[MTLVertexDescriptor alloc] init];
            for (uint32_t i = 0; i < numAttributes; ++i)
                [vertexDescriptor.attributes setObject:attributeDescriptors[i] atIndexedSubscript:i];
            for (uint32_t i = 0; i < numBufferSlots; ++i)
            {
                if (bufferLayouts[i])
                    [vertexDescriptor.layouts setObject:bufferLayouts[i] atIndexedSubscript:c_VertexBufferBase + i];
            }
            pipelineDescriptor.vertexDescriptor = vertexDescriptor;
        }

        for (uint32_t i = 0; i < fbinfo.colorFormats.size(); ++i)
            pipelineDescriptor.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);

        if (fbinfo.depthFormat != Format::UNKNOWN)
        {
            pipelineDescriptor.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);

            if (getFormatInfo(fbinfo.depthFormat).hasStencil)
                pipelineDescriptor.stencilAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        }

        const RasterState& rasterState = desc.renderState.rasterState;

        const DepthStencilState& depthStencilState = desc.renderState.depthStencilState;

        MTLDepthStencilDescriptor* depthStencilDescriptor = [[MTLDepthStencilDescriptor alloc] init];
        depthStencilDescriptor.depthCompareFunction = depthStencilState.depthTestEnable ? convertCompareFunc(depthStencilState.depthFunc) : MTLCompareFunctionAlways;
        depthStencilDescriptor.depthWriteEnabled = depthStencilState.depthWriteEnable;

        if (depthStencilState.stencilEnable)
        {
            MTLStencilDescriptor* frontStencil = [[MTLStencilDescriptor alloc] init];
            frontStencil.stencilCompareFunction = convertCompareFunc(depthStencilState.frontFaceStencil.stencilFunc);
            frontStencil.stencilFailureOperation = convertStencilOp(depthStencilState.frontFaceStencil.failOp);
            frontStencil.depthFailureOperation = convertStencilOp(depthStencilState.frontFaceStencil.depthFailOp);
            frontStencil.depthStencilPassOperation = convertStencilOp(depthStencilState.frontFaceStencil.passOp);
            frontStencil.readMask = depthStencilState.stencilReadMask;
            frontStencil.writeMask = depthStencilState.stencilWriteMask;

            MTLStencilDescriptor* backStencil = [[MTLStencilDescriptor alloc] init];
            backStencil.stencilCompareFunction = convertCompareFunc(depthStencilState.backFaceStencil.stencilFunc);
            backStencil.stencilFailureOperation = convertStencilOp(depthStencilState.backFaceStencil.failOp);
            backStencil.depthFailureOperation = convertStencilOp(depthStencilState.backFaceStencil.depthFailOp);
            backStencil.depthStencilPassOperation = convertStencilOp(depthStencilState.backFaceStencil.passOp);
            backStencil.readMask = depthStencilState.stencilReadMask;
            backStencil.writeMask = depthStencilState.stencilWriteMask;

            depthStencilDescriptor.frontFaceStencil = frontStencil;
            depthStencilDescriptor.backFaceStencil = backStencil;
        }

        for (uint32_t i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            const BlendState::RenderTarget& target = desc.renderState.blendState.targets[i];
            pipelineDescriptor.colorAttachments[i].blendingEnabled = target.blendEnable;
            pipelineDescriptor.colorAttachments[i].sourceRGBBlendFactor = convertBlendFactor(target.srcBlend);
            pipelineDescriptor.colorAttachments[i].destinationRGBBlendFactor = convertBlendFactor(target.destBlend);
            pipelineDescriptor.colorAttachments[i].rgbBlendOperation = convertBlendOp(target.blendOp);
            pipelineDescriptor.colorAttachments[i].sourceAlphaBlendFactor = convertBlendFactor(target.srcBlendAlpha);
            pipelineDescriptor.colorAttachments[i].destinationAlphaBlendFactor = convertBlendFactor(target.destBlendAlpha);
            pipelineDescriptor.colorAttachments[i].alphaBlendOperation = convertBlendOp(target.blendOpAlpha);

            MTLColorWriteMask writeMask = MTLColorWriteMaskNone;
            if (uint32_t(target.colorWriteMask) & uint32_t(ColorMask::Red)) writeMask |= MTLColorWriteMaskRed;
            if (uint32_t(target.colorWriteMask) & uint32_t(ColorMask::Green)) writeMask |= MTLColorWriteMaskGreen;
            if (uint32_t(target.colorWriteMask) & uint32_t(ColorMask::Blue)) writeMask |= MTLColorWriteMaskBlue;
            if (uint32_t(target.colorWriteMask) & uint32_t(ColorMask::Alpha)) writeMask |= MTLColorWriteMaskAlpha;
            pipelineDescriptor.colorAttachments[i].writeMask = writeMask;
        }

        NSError* nsError = nil;
        id<MTLRenderPipelineState> pipeline = [m_Context.device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&nsError];
        if (!pipeline)
        {
            m_Context.error("createGraphicsPipeline: failed to create the pipeline: " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nullptr;
        }

        GraphicsPipeline* result = new GraphicsPipeline();
        result->desc = desc;
        result->framebufferInfo = fbinfo;
        result->pipeline = pipeline;
        result->primitiveType = convertPrimitiveType(desc.primType);
        result->depthStencilState = [m_Context.device newDepthStencilStateWithDescriptor:depthStencilDescriptor];

        buildBindingPlan(result->plan, desc.bindingLayouts, vertexShader);
        buildBindingPlan(result->plan, desc.bindingLayouts, pixelShader);
        getPushConstantInfo(result->pushConstantBufferIndex, result->pushConstantSize, desc.bindingLayouts,
            vertexShader->translated.usesPushConstants || pixelShader->translated.usesPushConstants);
        result->vertexPushConstants = vertexShader->translated.usesPushConstants;
        result->pixelPushConstants = pixelShader->translated.usesPushConstants;

        return GraphicsPipelineHandle::Create(result);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb)
    {
        (void)desc;
        (void)fb;
        return nullptr;
    }
}
