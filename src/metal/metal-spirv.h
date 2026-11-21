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

#include <nvrhi/nvrhi.h>

#include <string>
#include <vector>

namespace nvrhi::metal
{
    enum class ResourceKind : uint8_t
    {
        Buffer,
        Texture,
        Sampler
    };

    // A shader-visible resource: the SPIR-V descriptor set and binding identify it,
    // the index is the slot it takes in the Metal buffer, texture or sampler namespace.
    // A writeable resource is a storage buffer or storage image.
    struct ResourceBinding
    {
        uint32_t descriptorSet = 0;
        uint32_t binding = 0;
        ResourceKind kind = ResourceKind::Buffer;
        bool writeable = false;
        bool runtimeArray = false;
        uint32_t index = 0;
        uint32_t count = 1;
        // A texture array of texture buffers holds the typed buffer views of NVRHI,
        // so it needs a different null resource than an array of sampled textures.
        bool bufferArray = false;
        // The texture type an array argument declares, which the null resource of
        // an unwritten slot has to match.
        TextureDimension dimension = TextureDimension::Unknown;
        // The format of the texture elements, which is the format of a null texture buffer.
        Format format = Format::UNKNOWN;
        // The stage the resource belongs to. A graphics pipeline translates its
        // stages separately, so one binding can take different indices per stage.
        ShaderType stage = ShaderType::None;
    };

    struct TranslatedShader
    {
        std::string source;
        std::vector<ResourceBinding> resources;
        ShaderType stage = ShaderType::None;
        uint32_t threadGroupSize[3] = { 1, 1, 1 };

        bool usesPushConstants = false;
        uint32_t pushConstantBufferIndex = 0;

        // A runtime sized descriptor array cannot be sized until the descriptor table
        // capacity is known, so such a shader keeps its function deferred to the pipeline.
        bool usesBindless = false;
    };

    // Translates a SPIR-V module to MSL. The resources are given explicit Metal indices
    // so that the command list can bind them without parsing the generated source.
    // The runtime sized arrays of the bindless descriptor set are sized to the table
    // capacity; a capacity of zero only reports whether the module uses them.
    bool translateSpirvToMSL(
        const void* spirv,
        size_t size,
        ShaderType stage,
        const std::string& entryName,
        uint32_t mslVersionMajor,
        uint32_t mslVersionMinor,
        uint32_t bindlessSet,
        uint32_t bindlessCapacity,
        TranslatedShader& out,
        std::string& error);
}
