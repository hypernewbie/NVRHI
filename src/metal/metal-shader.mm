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

    // Fetches the entry function of a library. The function constants are resolved
    // with an empty list, so every specialisation constant keeps its default. Metal
    // rejects a pipeline that a function with unresolved constants builds.
    static id<MTLFunction> resolveEntryFunction(Context& context, id<MTLLibrary> library, const ShaderDesc& d)
    {
        MTLFunctionConstantValues* constantValues = [[MTLFunctionConstantValues alloc] init];
        NSError* nsError = nil;

        id<MTLFunction> function = [library newFunctionWithName:@(d.entryName.c_str()) constantValues:constantValues error:&nsError];
        if (!function)
        {
            // SPIRV-Cross renames the entry point when it collides with a reserved name.
            const std::string renamedEntry = d.entryName + "0";
            function = [library newFunctionWithName:@(renamedEntry.c_str()) constantValues:constantValues error:&nsError];
        }

        if (!function)
        {
            context.error("resolveEntryFunction: the entry point '" + d.entryName + "' is missing in '" + d.debugName + "': "
                + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nil;
        }

        return function;
    }

    // Compiles the entry point of a translation, including the reserved entry rename.
    static id<MTLFunction> compileEntryFunction(Context& context, const TranslatedShader& translated, const ShaderDesc& d)
    {
        NSError* nsError = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [context.device newLibraryWithSource:@(translated.source.c_str()) options:options error:&nsError];
        if (!library)
        {
            context.error("compileEntryFunction: failed to compile '" + d.debugName + "': " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nil;
        }

        return resolveEntryFunction(context, library, d);
    }

    const Shader::BindlessTranslation* Shader::getTranslationForCapacities(Context& context, const std::vector<uint32_t>& capacities) const
    {
        if (!translated.usesBindless)
            return nullptr;

        {
            std::lock_guard< std::mutex > lock(bindlessTranslationsMutex);
            const auto cached = bindlessTranslations.find(capacities);
            if (cached != bindlessTranslations.end())
                return &cached->second;
        }

        BindlessTranslation translation;
        std::string error;
        if (!translateSpirvToMSL(spirv.data(), spirv.size() * sizeof(uint32_t), desc.shaderType, desc.entryName,
            context.mslVersionMajor, context.mslVersionMinor, capacities,
            context.device.argumentBuffersSupport == MTLArgumentBuffersTier2, translation.translated, error))
        {
            context.error("getTranslationForCapacities: failed to translate '" + desc.debugName + "': " + error);
            return nullptr;
        }

        translation.function = compileEntryFunction(context, translation.translated, desc);
        if (!translation.function)
            return nullptr;

        std::lock_guard< std::mutex > lock(bindlessTranslationsMutex);
        return &bindlessTranslations.emplace(capacities, std::move(translation)).first->second;
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

        // The mapping is already resolved, so an identity override keeps the identity
        // instead of falling back to the default mapping of the texture.
        const ComponentMapping viewMapping = componentMapping;

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

    Object DescriptorTable::getNativeObject(ObjectType objectType)
    {
        (void)objectType;
        return nullptr;
    }

    Object Sampler::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_Sampler)
            return Object((__bridge void*)sampler);

        return nullptr;
    }

    Object Heap::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::Metal_Heap)
            return Object((__bridge void*)heap);

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

        // Map translated descriptor sets back to their binding positions in the pipeline.
        void buildBindingPlan(PipelineBindingPlan& plan, const BindingLayoutVector& layouts, const std::vector<uint32_t>& descriptorSets,
            const TranslatedShader& translated, id<MTLFunction> function)
        {
            plan.descriptorSets = descriptorSets;
            for (const auto& argumentBuffer : translated.argumentBuffers)
                plan.argumentBuffers.push_back({ translated.stage, argumentBuffer.descriptorSet, argumentBuffer.index, function });

            for (const auto& resource : translated.resources)
            {
                if (resource.runtimeArray)
                {
                    plan.bindlessArrays.push_back(resource);
                    continue;
                }

                const auto set = std::find(descriptorSets.begin(), descriptorSets.end(), resource.descriptorSet);
                if (set == descriptorSets.end())
                    continue;

                const uint32_t bindingIndex = uint32_t(set - descriptorSets.begin());
                const BindingLayout* layout = checked_cast<BindingLayout*>(layouts[bindingIndex].Get());
                if (!layout || layout->isBindless)
                    continue;

                for (const auto& item : layout->desc.bindings)
                {
                    if (item.slot != resource.binding)
                        continue;

                    PipelineBindingPlan::Entry entry;
                    entry.stage = translated.stage;
                    entry.bindingIndex = bindingIndex;
                    entry.slot = item.slot;
                    entry.bound.kind = resource.kind;
                    entry.bound.index = resource.index;
                    plan.entries.push_back(entry);
                }
            }
        }
    }

    ShaderHandle Device::createShader(const ShaderDesc& d, const void* binary, size_t binarySize)
    {
        Shader* shader = new Shader();
        shader->desc = d;
        shader->spirv.assign(reinterpret_cast<const uint32_t*>(binary), reinterpret_cast<const uint32_t*>(binary) + binarySize / sizeof(uint32_t));

        TranslatedShader& translated = shader->translated;
        std::string error;
        if (!translateSpirvToMSL(binary, binarySize, d.shaderType, d.entryName, m_Context.mslVersionMajor, m_Context.mslVersionMinor,
            {}, false, translated, error))
        {
            m_Context.error("createShader: failed to translate '" + d.debugName + "': " + error);
            delete shader;
            return nullptr;
        }

        // A runtime sized array has no function until the descriptor table capacity
        // is known; the pipeline compiles the translation for its capacity.
        if (translated.usesBindless)
            return ShaderHandle::Create(shader);

        NSError* nsError = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [m_Context.device newLibraryWithSource:@(translated.source.c_str()) options:options error:&nsError];
        if (!library)
        {
            m_Context.error("createShader: failed to compile '" + d.debugName + "': " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            delete shader;
            return nullptr;
        }

        // SPIRV-Cross renames the entry point when it collides with a reserved name.
        id<MTLFunction> function = resolveEntryFunction(m_Context, library, d);
        if (!function)
        {
            delete shader;
            return nullptr;
        }

        shader->library = library;
        shader->function = function;
        shader->source = translated.source;

        return ShaderHandle::Create(shader);
    }

    ShaderHandle Device::createShaderSpecialization(IShader* _baseShader, const ShaderSpecialization* constants, uint32_t numConstants)
    {
        Shader* baseShader = checked_cast<Shader*>(_baseShader);
        if (!baseShader || !constants || numConstants == 0)
        {
            m_Context.error("createShaderSpecialization: the base shader and the constants are required");
            return nullptr;
        }

        // A bindless shader gets its function per table capacity, so its constants
        // cannot fold into one function.
        if (baseShader->translated.usesBindless)
        {
            m_Context.error("createShaderSpecialization: a shader with a runtime sized array cannot be specialised");
            return nullptr;
        }

        // The constants of a specialisable function are read from the plain entry
        // function, because a function that resolved its constants reports none.
        id<MTLFunction> declarationFunction = [baseShader->library newFunctionWithName:@(baseShader->desc.entryName.c_str())];
        if (!declarationFunction)
        {
            const std::string renamedEntry = baseShader->desc.entryName + "0";
            declarationFunction = [baseShader->library newFunctionWithName:@(renamedEntry.c_str())];
        }
        if (!declarationFunction)
        {
            m_Context.error("createShaderSpecialization: the entry point is missing in '" + baseShader->desc.debugName + "'");
            return nullptr;
        }

        // The translated shader declares every specialisation constant as a Metal
        // function constant, and the entry function picks the values from this list.
        MTLFunctionConstantValues* constantValues = [[MTLFunctionConstantValues alloc] init];
        for (uint32_t i = 0; i < numConstants; ++i)
        {
            const ShaderSpecialization& constant = constants[i];

            MTLFunctionConstant* declaration = nil;
            for (MTLFunctionConstant* candidate in declarationFunction.functionConstantsDictionary.allValues)
            {
                if (candidate.index == constant.constantID)
                {
                    declaration = candidate;
                    break;
                }
            }

            // Metal can omit unused constants, including a workgroup size that the
            // shader does not read. Vulkan ignores IDs absent from the shader too.
            if (!declaration)
                continue;

            switch (declaration.type)
            {
            case MTLDataTypeUInt:
                [constantValues setConstantValue:&constant.value.u type:MTLDataTypeUInt atIndex:constant.constantID];
                break;
            case MTLDataTypeInt:
                [constantValues setConstantValue:&constant.value.i type:MTLDataTypeInt atIndex:constant.constantID];
                break;
            case MTLDataTypeFloat:
                [constantValues setConstantValue:&constant.value.f type:MTLDataTypeFloat atIndex:constant.constantID];
                break;
            case MTLDataTypeBool:
            {
                const BOOL value = constant.value.u != 0;
                [constantValues setConstantValue:&value type:MTLDataTypeBool atIndex:constant.constantID];
                break;
            }
            default:
                // Half constants have no place in the four byte value union, like
                // on the other backends.
                m_Context.error("createShaderSpecialization: the specialisation constant " + std::to_string(constant.constantID)
                    + " has an unsupported type");
                return nullptr;
            }
        }

        NSError* nsError = nil;
        id<MTLFunction> function = [baseShader->library newFunctionWithName:@(baseShader->desc.entryName.c_str())
            constantValues:constantValues
            error:&nsError];

        // SPIRV-Cross renames the entry point when it collides with a reserved name.
        if (!function)
        {
            const std::string renamedEntry = baseShader->desc.entryName + "0";
            function = [baseShader->library newFunctionWithName:@(renamedEntry.c_str())
                constantValues:constantValues
                error:&nsError];
        }

        if (!function)
        {
            m_Context.error("createShaderSpecialization: failed to specialise '" + baseShader->desc.debugName + "': "
                + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nullptr;
        }

        Shader* shader = new Shader();
        // The specialised shader holds the base shader, which owns the library.
        shader->baseShader = baseShader->baseShader ? baseShader->baseShader : ShaderHandle(_baseShader);
        shader->desc = baseShader->desc;
        shader->spirv = baseShader->spirv;
        const Shader* originalShader = baseShader->baseShader ? checked_cast<Shader*>(baseShader->baseShader.Get()) : baseShader;
        shader->translated = originalShader->translated;
        for (uint32_t i = 0; i < numConstants; ++i)
        {
            for (uint32_t dimension = 0; dimension < 3; ++dimension)
            {
                if (shader->translated.threadGroupSizeSpecialized[dimension]
                    && shader->translated.threadGroupSizeConstantId[dimension] == constants[i].constantID)
                {
                    if (constants[i].value.u == 0)
                    {
                        m_Context.error("createShaderSpecialization: the workgroup size must be nonzero");
                        delete shader;
                        return nullptr;
                    }
                    shader->translated.threadGroupSize[dimension] = constants[i].value.u;
                }
            }
        }
        shader->library = baseShader->library;
        shader->function = function;
        shader->source = baseShader->source;
        return ShaderHandle::Create(shader);
    }

    ShaderLibraryHandle Device::createShaderLibrary(const void* binary, size_t binarySize)
    {
        (void)binary;
        (void)binarySize;
        m_Context.error("createShaderLibrary: shader libraries are not supported by the Metal backend");
        return nullptr;
    }

    // Match Vulkan's descriptor set assignment: layouts use their order unless
    // regular layouts specify set indices, in which case bindless layouts follow them.
    static bool getDescriptorSets(Context& context, const BindingLayoutVector& layouts, std::vector<uint32_t>& sets)
    {
        bool explicitSets = false;
        bool foundRegular = false;
        uint32_t nextBindlessSet = 0;
        for (const auto& handle : layouts)
        {
            const BindingLayout* layout = checked_cast<BindingLayout*>(handle.Get());
            if (!layout || layout->isBindless)
                continue;
            if (!foundRegular)
                explicitSets = layout->desc.registerSpaceIsDescriptorSet;
            else if (explicitSets != layout->desc.registerSpaceIsDescriptorSet)
            {
                context.error("binding layouts disagree on descriptor set mapping");
                return false;
            }
            foundRegular = true;
            if (explicitSets)
                nextBindlessSet = std::max(nextBindlessSet, layout->desc.registerSpace + 1);
        }

        std::set<uint32_t> used;
        for (uint32_t index = 0; index < layouts.size(); ++index)
        {
            const BindingLayout* layout = checked_cast<BindingLayout*>(layouts[index].Get());
            uint32_t set = index;
            if (explicitSets && layout)
                set = layout->isBindless ? nextBindlessSet++ : layout->desc.registerSpace;
            if (set >= c_MaxBindingLayouts || !used.insert(set).second)
            {
                context.error("binding layouts exceed or overlap the Metal descriptor set limit");
                return false;
            }
            sets.push_back(set);
        }
        return true;
    }

    static std::vector<uint32_t> getBindlessCapacities(const BindingLayoutVector& layouts, const std::vector<uint32_t>& sets)
    {
        // An unbound table uses capacity one; the shader must not access it.
        std::vector<uint32_t> capacities(c_MaxBindingLayouts, 1);
        for (uint32_t index = 0; index < layouts.size(); ++index)
        {
            const BindingLayout* layout = checked_cast<BindingLayout*>(layouts[index].Get());
            if (layout && layout->isBindless)
                capacities[sets[index]] = layout->bindlessDesc.maxCapacity;
        }
        return capacities;
    }

    static const TranslatedShader* resolveTranslation(Context& context, const Shader* shader, const std::vector<uint32_t>& capacities)
    {
        if (!shader->translated.usesBindless)
            return &shader->translated;

        const Shader::BindlessTranslation* translation = shader->getTranslationForCapacities(context, capacities);
        return translation ? &translation->translated : nullptr;
    }

    static id<MTLFunction> resolveFunction(Context& context, const Shader* shader, const std::vector<uint32_t>& capacities)
    {
        if (!shader->translated.usesBindless)
            return shader->function;

        const Shader::BindlessTranslation* translation = shader->getTranslationForCapacities(context, capacities);
        return translation ? translation->function : nil;
    }

    ComputePipelineHandle Device::createComputePipeline(const ComputePipelineDesc& desc)
    {
        if (!desc.CS)
        {
            m_Context.error("createComputePipeline: the compute shader is missing");
            return nullptr;
        }

        std::vector<uint32_t> descriptorSets;
        if (!getDescriptorSets(m_Context, desc.bindingLayouts, descriptorSets))
            return nullptr;
        const std::vector<uint32_t> capacities = getBindlessCapacities(desc.bindingLayouts, descriptorSets);

        const Shader* shader = checked_cast<Shader*>(desc.CS.Get());
        if (!shader)
        {
            m_Context.error("createComputePipeline: the compute shader is not a Metal shader");
            return nullptr;
        }

        const TranslatedShader* translation = resolveTranslation(m_Context, shader, capacities);
        id<MTLFunction> function = resolveFunction(m_Context, shader, capacities);
        if (!translation || !function)
            return nullptr;

        NSError* nsError = nil;
        id<MTLComputePipelineState> pipeline = [m_Context.device newComputePipelineStateWithFunction:function error:&nsError];
        if (!pipeline)
        {
            m_Context.error("createComputePipeline: failed to create the pipeline: " + std::string(nsError ? nsError.localizedDescription.UTF8String : "unknown error"));
            return nullptr;
        }

        ComputePipeline* result = new ComputePipeline();
        result->desc = desc;
        result->pipeline = pipeline;
        result->threadGroupSize[0] = translation->threadGroupSize[0];
        result->threadGroupSize[1] = translation->threadGroupSize[1];
        result->threadGroupSize[2] = translation->threadGroupSize[2];

        buildBindingPlan(result->plan, desc.bindingLayouts, descriptorSets, *translation, function);
        getPushConstantInfo(result->pushConstantBufferIndex, result->pushConstantSize, desc.bindingLayouts, translation->usesPushConstants);

        return ComputePipelineHandle::Create(result);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        Shader* vertexShader = checked_cast<Shader*>(desc.VS.Get());
        Shader* pixelShader = checked_cast<Shader*>(desc.PS.Get());
        if (!vertexShader || !pixelShader)
        {
            m_Context.error("createGraphicsPipeline: the vertex and pixel shaders are required");
            return nullptr;
        }

        std::vector<uint32_t> descriptorSets;
        if (!getDescriptorSets(m_Context, desc.bindingLayouts, descriptorSets))
            return nullptr;
        const std::vector<uint32_t> capacities = getBindlessCapacities(desc.bindingLayouts, descriptorSets);

        const TranslatedShader* vertexTranslation = resolveTranslation(m_Context, vertexShader, capacities);
        const TranslatedShader* pixelTranslation = resolveTranslation(m_Context, pixelShader, capacities);
        id<MTLFunction> vertexFunction = resolveFunction(m_Context, vertexShader, capacities);
        id<MTLFunction> pixelFunction = resolveFunction(m_Context, pixelShader, capacities);
        if (!vertexTranslation || !pixelTranslation || !vertexFunction || !pixelFunction)
            return nullptr;

        MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = pixelFunction;
        pipelineDescriptor.rasterSampleCount = std::max(1u, fbinfo.sampleCount);

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

        buildBindingPlan(result->plan, desc.bindingLayouts, descriptorSets, *vertexTranslation, vertexFunction);
        buildBindingPlan(result->plan, desc.bindingLayouts, descriptorSets, *pixelTranslation, pixelFunction);
        getPushConstantInfo(result->pushConstantBufferIndex, result->pushConstantSize, desc.bindingLayouts,
            vertexTranslation->usesPushConstants || pixelTranslation->usesPushConstants);
        result->vertexPushConstants = vertexTranslation->usesPushConstants;
        result->pixelPushConstants = pixelTranslation->usesPushConstants;

        return GraphicsPipelineHandle::Create(result);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb)
    {
        const Framebuffer* framebuffer = checked_cast<Framebuffer*>(fb);
        if (!framebuffer)
        {
            m_Context.error("createGraphicsPipeline: the framebuffer is missing");
            return nullptr;
        }

        return createGraphicsPipeline(desc, framebuffer->getFramebufferInfo());
    }
}
