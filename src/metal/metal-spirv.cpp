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

#include "metal-spirv.h"

#include <spirv_msl.hpp>

#include <algorithm>

namespace nvrhi::metal
{
    namespace
    {
        struct ResourceEntry
        {
            uint32_t id = 0;
            uint32_t descriptorSet = 0;
            uint32_t binding = 0;
            ResourceKind kind = ResourceKind::Buffer;
            uint32_t index = 0;
            uint32_t count = 1;
        };

        bool toExecutionModel(ShaderType stage, spv::ExecutionModel& out)
        {
            switch (stage)
            {
            case ShaderType::Compute:  out = spv::ExecutionModelGLCompute; return true;
            case ShaderType::Vertex:   out = spv::ExecutionModelVertex; return true;
            case ShaderType::Pixel:    out = spv::ExecutionModelFragment; return true;
            case ShaderType::Hull:     out = spv::ExecutionModelTessellationControl; return true;
            case ShaderType::Domain:   out = spv::ExecutionModelTessellationEvaluation; return true;
            case ShaderType::Geometry: out = spv::ExecutionModelGeometry; return true;
            default: return false;
            }
        }

        uint32_t getResourceCount(spirv_cross::CompilerMSL& compiler, uint32_t id)
        {
            const spirv_cross::SPIRType& type = compiler.get_type_from_variable(id);
            if (type.array.empty())
                return 1;

            uint32_t count = 1;
            for (const auto dimension : type.array)
                count *= dimension;

            return count;
        }
    }

    bool translateSpirvToMSL(
        const void* spirv,
        size_t size,
        ShaderType stage,
        const std::string& entryName,
        uint32_t mslVersionMajor,
        uint32_t mslVersionMinor,
        TranslatedShader& out,
        std::string& error)
    {
        spv::ExecutionModel model;
        if (!toExecutionModel(stage, model))
        {
            error = "unsupported shader stage";
            return false;
        }

        if (size < 4 || (size % 4) != 0)
        {
            error = "SPIR-V size is not a multiple of four";
            return false;
        }

        try
        {
            spirv_cross::CompilerMSL compiler(reinterpret_cast<const uint32_t*>(spirv), size / 4);

            compiler.set_entry_point(entryName, model);

            spirv_cross::CompilerMSL::Options options;
            options.set_msl_version(mslVersionMajor, mslVersionMinor);
            options.platform = spirv_cross::CompilerMSL::Options::macOS;

            compiler.set_msl_options(options);

            const spirv_cross::ShaderResources resources = compiler.get_shader_resources();

            std::vector<ResourceEntry> entries;

            auto collect = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& list, ResourceKind kind)
            {
                for (const auto& resource : list)
                {
                    ResourceEntry entry;
                    entry.id = resource.id;
                    entry.descriptorSet = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
                    entry.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
                    entry.kind = kind;
                    entry.count = getResourceCount(compiler, resource.id);
                    entries.push_back(entry);
                }
            };

            collect(resources.uniform_buffers, ResourceKind::Buffer);
            collect(resources.storage_buffers, ResourceKind::Buffer);
            collect(resources.separate_images, ResourceKind::Texture);
            collect(resources.storage_images, ResourceKind::Texture);
            collect(resources.separate_samplers, ResourceKind::Sampler);

            // Texel buffers are images in SPIR-V, but they are buffers in Metal.
            for (const auto& resource : resources.separate_images)
            {
                const spirv_cross::SPIRType& type = compiler.get_type_from_variable(resource.id);
                if (type.image.dim == spv::DimBuffer)
                {
                    error = "typed buffers are not supported";
                    return false;
                }
            }

            if (!resources.sampled_images.empty())
            {
                error = "combined image samplers are not supported";
                return false;
            }

            if (!resources.subpass_inputs.empty() || !resources.acceleration_structures.empty())
            {
                error = "input attachments and acceleration structures are not supported";
                return false;
            }

            // Push constants live in their own buffer, so they take the first buffer slot
            // and the regular buffers start after them.
            const bool usesPushConstants = !resources.push_constant_buffers.empty();

            std::sort(entries.begin(), entries.end(), [](const ResourceEntry& a, const ResourceEntry& b)
            {
                if (a.descriptorSet != b.descriptorSet)
                    return a.descriptorSet < b.descriptorSet;
                return a.binding < b.binding;
            });

            uint32_t nextBuffer = usesPushConstants ? 1 : 0;
            uint32_t nextTexture = 0;
            uint32_t nextSampler = 0;

            for (auto& entry : entries)
            {
                switch (entry.kind)
                {
                case ResourceKind::Buffer:
                    entry.index = nextBuffer;
                    nextBuffer += entry.count;
                    break;
                case ResourceKind::Texture:
                    entry.index = nextTexture;
                    nextTexture += entry.count;
                    break;
                case ResourceKind::Sampler:
                    entry.index = nextSampler;
                    nextSampler += entry.count;
                    break;
                }
            }

            for (const auto& entry : entries)
            {
                spirv_cross::MSLResourceBinding binding;
                binding.stage = model;
                binding.desc_set = entry.descriptorSet;
                binding.binding = entry.binding;
                binding.count = entry.count;
                binding.msl_buffer = entry.kind == ResourceKind::Buffer ? entry.index : 0;
                binding.msl_texture = entry.kind == ResourceKind::Texture ? entry.index : 0;
                binding.msl_sampler = entry.kind == ResourceKind::Sampler ? entry.index : 0;
                compiler.add_msl_resource_binding(binding);
            }

            if (usesPushConstants)
            {
                spirv_cross::MSLResourceBinding binding;
                binding.stage = model;
                binding.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
                binding.binding = spirv_cross::ResourceBindingPushConstantBinding;
                binding.msl_buffer = 0;
                compiler.add_msl_resource_binding(binding);
            }

            out.source = compiler.compile();

            out.resources.clear();
            for (const auto& entry : entries)
            {
                ResourceBinding resource;
                resource.descriptorSet = entry.descriptorSet;
                resource.binding = entry.binding;
                resource.kind = entry.kind;
                resource.index = entry.index;
                resource.count = entry.count;
                out.resources.push_back(resource);
            }

            out.usesPushConstants = usesPushConstants;
            out.pushConstantBufferIndex = 0;

            if (model == spv::ExecutionModelGLCompute)
            {
                for (uint32_t i = 0; i < 3; ++i)
                    out.threadGroupSize[i] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, i);
            }

            return true;
        }
        catch (const spirv_cross::CompilerError& e)
        {
            error = e.what();
            return false;
        }
    }
}
