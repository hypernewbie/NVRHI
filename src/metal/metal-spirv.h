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
    struct ResourceBinding
    {
        uint32_t descriptorSet = 0;
        uint32_t binding = 0;
        ResourceKind kind = ResourceKind::Buffer;
        uint32_t index = 0;
        uint32_t count = 1;
    };

    struct TranslatedShader
    {
        std::string source;
        std::vector<ResourceBinding> resources;

        uint32_t threadGroupSize[3] = { 1, 1, 1 };

        bool usesPushConstants = false;
        uint32_t pushConstantBufferIndex = 0;
    };

    // Translates a SPIR-V module to MSL. The resources are given explicit Metal indices
    // so that the command list can bind them without parsing the generated source.
    bool translateSpirvToMSL(
        const void* spirv,
        size_t size,
        ShaderType stage,
        const std::string& entryName,
        uint32_t mslVersionMajor,
        uint32_t mslVersionMinor,
        TranslatedShader& out,
        std::string& error);
}
