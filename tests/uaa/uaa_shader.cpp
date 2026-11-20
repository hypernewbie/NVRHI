/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "uaa_fixture.h"
#include "uaa_runner.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        TestFixture& requireFixture(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr && ctx.fixture->device() != nullptr,
                "GPU test requires an initialised TestFixture");
            return *ctx.fixture;
        }

        uint32_t floatToBits(float v)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            return bits;
        }

        uint32_t packRgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            return static_cast<uint32_t>(r)
                | (static_cast<uint32_t>(g) << 8u)
                | (static_cast<uint32_t>(b) << 16u)
                | (static_cast<uint32_t>(a) << 24u);
        }

        BufferHandle createStorageBuffer(
            const TestFixture& fix,
            uint64_t byteSize,
            const char* debugName,
            uint32_t structStride = 4,
            bool uav = false,
            bool typed = false,
            bool raw = false,
            Format typedFormat = Format::UNKNOWN)
        {
            BufferDesc desc;
            desc.byteSize = byteSize;
            desc.structStride = structStride;
            desc.canHaveUAVs = uav;
            desc.canHaveTypedViews = typed;
            desc.canHaveRawViews = raw;
            desc.format = typedFormat;
            desc.initialState = uav ? ResourceStates::UnorderedAccess : ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            desc.debugName = debugName;
            return fix.device()->createBuffer(desc);
        }

        BufferHandle createConstantBuffer(const TestFixture& fix, uint64_t byteSize, const char* debugName)
        {
            BufferDesc desc;
            desc.byteSize = byteSize;
            desc.isConstantBuffer = true;
            desc.initialState = ResourceStates::ConstantBuffer;
            desc.keepInitialState = true;
            desc.debugName = debugName;
            return fix.device()->createBuffer(desc);
        }

        TextureHandle createRgba8Texture2D(
            const TestFixture& fix,
            uint32_t width,
            uint32_t height,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            const char* debugName)
        {
            TextureDesc desc;
            desc.width = width;
            desc.height = height;
            desc.format = Format::RGBA8_UNORM;
            desc.isShaderResource = true;
            desc.initialState = ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            desc.debugName = debugName;

            TextureHandle tex = fix.device()->createTexture(desc);
            if (!tex)
            {
                return nullptr;
            }

            std::vector<uint8_t> pixels(size_t(width) * size_t(height) * 4u);
            for (size_t i = 0; i < size_t(width) * size_t(height); ++i)
            {
                pixels[i * 4u + 0u] = r;
                pixels[i * 4u + 1u] = g;
                pixels[i * 4u + 2u] = b;
                pixels[i * 4u + 3u] = a;
            }
            fix.uploadTexture2D(tex, 0, 0, pixels.data(), size_t(width) * 4u);
            return tex;
        }

        void dispatchAndWait(
            const TestFixture& fix,
            IComputePipeline* pso,
            const std::vector<IBindingSet*>& sets,
            const void* pushData = nullptr,
            size_t pushSize = 0,
            uint32_t groupsX = 1)
        {
            CommandListHandle cmd = fix.device()->createCommandList();
            if (!cmd)
            {
                throw TestFailureException("dispatchAndWait: createCommandList returned null");
            }

            ComputeState state;
            state.pipeline = pso;
            for (IBindingSet* s : sets)
            {
                state.addBindingSet(s);
            }

            cmd->open();
            cmd->setComputeState(state);
            if (pushData && pushSize > 0)
            {
                cmd->setPushConstants(pushData, pushSize);
            }
            cmd->dispatch(groupsX, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);
        }

        void verifyShaderMetadata(
            TestContext& ctx,
            const TestFixture& fix,
            IShader* shader,
            ShaderType expectedType,
            const std::string& expectedEntry,
            const char* label)
        {
            UAA_CHECK(shader != nullptr, std::string(label) + ": createShader returned null");
            const ShaderDesc& desc = shader->getDesc();
            UAA_CHECK_EQ(desc.shaderType, expectedType, std::string(label) + ": shaderType mismatch");
            UAA_CHECK_EQ(desc.entryName, expectedEntry, std::string(label) + ": entryName mismatch");

            const void* bytecode = nullptr;
            size_t bytecodeSize = 0;
            shader->getBytecode(&bytecode, &bytecodeSize);

            if (fix.api() == GraphicsAPI::VULKAN)
            {
                UAA_CHECK(bytecode == nullptr && bytecodeSize == 0,
                    std::string(label) + ": Vulkan backend does not retain SPIR-V bytecode");
            }
            else
            {
                UAA_CHECK(bytecode != nullptr && bytecodeSize > 0,
                    std::string(label) + ": getBytecode must return non-null bytecode and nonzero size");
            }
        }

        void runSpecialisationTest(
            TestContext& ctx,
            const TestFixture& fix,
            IShader* shader,
            IBindingLayout* layout,
            IBindingSet* bindingSet,
            IBuffer* outBuf,
            uint32_t expected0,
            int32_t expected1,
            float expected2,
            const char* label)
        {
            ComputePipelineDesc psoDesc;
            psoDesc.CS = shader;
            psoDesc.addBindingLayout(layout);
            ComputePipelineHandle pso = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, std::string(label) + ": createComputePipeline returned null");

            const std::array<uint32_t, 4> zeros = { 0u, 0u, 0u, 0u };
            fix.uploadBuffer(outBuf, zeros.data(), sizeof(zeros));
            dispatchAndWait(fix, pso, { bindingSet });

            std::array<uint32_t, 4> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], expected0, std::string(label) + ": K0 mismatch");
            UAA_CHECK_EQ(actual[1], static_cast<uint32_t>(expected1), std::string(label) + ": K1 mismatch");
            UAA_CHECK_EQ(actual[2], floatToBits(expected2), std::string(label) + ": K2 mismatch");
        }
    } // namespace

    static void testShaderStages(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        ShaderHandle cs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");
        verifyShaderMetadata(ctx, fix, cs, ShaderType::Compute, "main", "cs_buffer_copy_add");

        ShaderHandle vs = fix.createShader("vs_fullscreen_tri", ShaderType::Vertex, "main");
        verifyShaderMetadata(ctx, fix, vs, ShaderType::Vertex, "main", "vs_fullscreen_tri");

        ShaderHandle ps = fix.createShader("ps_solid_color", ShaderType::Pixel, "main");
        verifyShaderMetadata(ctx, fix, ps, ShaderType::Pixel, "main", "ps_solid_color");

        if (fix.api() == GraphicsAPI::VULKAN)
        {
            ctx.note("Vulkan backend intentionally does not retain bytecode after VkShaderModule creation");
        }

        if (fix.features().supports(Feature::Meshlets))
        {
            ShaderHandle ms = fix.createShader("ms_triangle", ShaderType::Mesh, "main");
            verifyShaderMetadata(ctx, fix, ms, ShaderType::Mesh, "main", "ms_triangle");

            ShaderHandle as = fix.createShader("as_task", ShaderType::Amplification, "main");
            verifyShaderMetadata(ctx, fix, as, ShaderType::Amplification, "main", "as_task");
            ctx.note("Tested optional Mesh and Amplification shader stages");
        }

        if (fix.features().supports(Feature::RayQuery))
        {
            ShaderHandle rq = fix.createShader("cs_ray_query", ShaderType::Compute, "main");
            verifyShaderMetadata(ctx, fix, rq, ShaderType::Compute, "main", "cs_ray_query");
            ctx.note("Tested optional RayQuery compute shader");
        }

        if (fix.features().supports(Feature::RayTracingPipeline))
        {
            if (fix.api() == GraphicsAPI::D3D12)
            {
                std::vector<uint8_t> libBytes = fix.loadShaderBinary("rt_lib");
                ShaderLibraryHandle rtLib = fix.device()->createShaderLibrary(libBytes.data(), libBytes.size());
                UAA_CHECK(rtLib != nullptr, "rt_lib: createShaderLibrary returned null");
                ShaderHandle rayGen = rtLib->getShader("RayGen", ShaderType::RayGeneration);
                verifyShaderMetadata(ctx, fix, rayGen, ShaderType::RayGeneration, "RayGen", "rt_lib::RayGen");
            }
            else
            {
                ShaderHandle rayGen = fix.createShader("rt_lib", ShaderType::RayGeneration, "RayGen");
                verifyShaderMetadata(ctx, fix, rayGen, ShaderType::RayGeneration, "RayGen", "rt_lib::RayGen");
            }
            ctx.note("Tested optional RayGeneration shader stage");
        }
    }

    static void testShaderEntryPoints(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (fix.api() != GraphicsAPI::VULKAN && fix.api() != GraphicsAPI::METAL)
        {
            ctx.skip("Single-module multi-entry SPIR-V fixture is only built for Vulkan and Metal");
        }

        ShaderHandle csAdd = fix.createShader("cs_multi_entry", ShaderType::Compute, "entry_add");
        ShaderHandle csMul = fix.createShader("cs_multi_entry", ShaderType::Compute, "entry_mul");
        UAA_CHECK(csAdd != nullptr, "entry_add shader must not be null");
        UAA_CHECK(csMul != nullptr, "entry_mul shader must not be null");
        UAA_CHECK_EQ(csAdd->getDesc().entryName, std::string("entry_add"), "entry_add name mismatch");
        UAA_CHECK_EQ(csMul->getDesc().entryName, std::string("entry_mul"), "entry_mul name mismatch");

        BindingLayoutHandle layout = fix.device()->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
        UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

        BufferHandle outBuf = createStorageBuffer(fix, 16, "shd_multi_entry_out", 4, true);
        UAA_CHECK(outBuf != nullptr, "createBuffer returned null");

        BindingSetHandle set = fix.device()->createBindingSet(
            BindingSetDesc().addItem(BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
            layout);
        UAA_CHECK(set != nullptr, "createBindingSet returned null");

        ComputePipelineHandle psoAdd = fix.device()->createComputePipeline(
            ComputePipelineDesc().setComputeShader(csAdd).addBindingLayout(layout));
        ComputePipelineHandle psoMul = fix.device()->createComputePipeline(
            ComputePipelineDesc().setComputeShader(csMul).addBindingLayout(layout));
        UAA_CHECK(psoAdd != nullptr, "createComputePipeline for entry_add returned null");
        UAA_CHECK(psoMul != nullptr, "createComputePipeline for entry_mul returned null");

        const std::array<uint32_t, 4> initData = { 10u, 0u, 0u, 0u };
        std::array<uint32_t, 4> result = {};

        fix.uploadBuffer(outBuf, initData.data(), sizeof(initData));
        dispatchAndWait(fix, psoAdd, { set });
        fix.readbackBuffer(outBuf, result.data(), sizeof(result));
        UAA_CHECK_EQ(result[1], 27u, "entry_add output mismatch");

        fix.uploadBuffer(outBuf, initData.data(), sizeof(initData));
        dispatchAndWait(fix, psoMul, { set });
        fix.readbackBuffer(outBuf, result.data(), sizeof(result));
        UAA_CHECK_EQ(result[1], 30u, "entry_mul output mismatch");
    }

    static void testShaderInvalidBinary(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        const std::vector<uint8_t> validBinary = fix.loadShaderBinary("cs_buffer_copy_add");
        UAA_CHECK(!validBinary.empty(), "Valid shader binary must not be empty");

        {
            ExpectErrorsGuard guard(fix);

            const bool vkNativeValidation = (fix.api() == GraphicsAPI::VULKAN)
                && (fix.deviceInfo().nativeValidation || fix.options().nativeValidation);

            if (vkNativeValidation)
            {
                ctx.note("Skipped null, zero-size, and truncated SPIR-V createShader calls on Vulkan with native validation active because Khronos validation crashes on invalid SPIR-V headers");
            }
            else
            {
                auto checkInvalidShaderBinary = [&](const char* debugName, const void* data, size_t size, const char* label)
                {
                    ShaderDesc desc;
                    desc.setShaderType(ShaderType::Compute).setEntryName("main").setDebugName(debugName);
                    ShaderHandle shader = dev->createShader(desc, data, size);
                    if (fix.api() == GraphicsAPI::VULKAN && shader != nullptr)
                    {
                        // Some Vulkan drivers defer SPIR-V validation until pipeline creation.
                        ComputePipelineHandle badPso = dev->createComputePipeline(
                            ComputePipelineDesc().setComputeShader(shader));
                        UAA_CHECK(badPso == nullptr, std::string(label) + " must fail either shader or pipeline creation");
                    }
                    else
                    {
                        UAA_CHECK(shader == nullptr, std::string(label) + " must return nullptr");
                    }
                };

                checkInvalidShaderBinary("invalid_null", nullptr, 0, "createShader(nullptr, 0)");

                const uint8_t dummy[16] = {};
                checkInvalidShaderBinary("invalid_zero", dummy, 0, "createShader(validPtr, 0)");

                const uint8_t truncated[8] = { 0x03u, 0x02u, 0x23u, 0x07u, 0x00u, 0x01u, 0x00u, 0x00u };
                checkInvalidShaderBinary("invalid_truncated", truncated, sizeof(truncated), "Truncated 8-byte SPIR-V");
            }

            ShaderDesc badEntryDesc;
            badEntryDesc.setShaderType(ShaderType::Compute)
                .setEntryName("does_not_exist")
                .setDebugName("invalid_entry");
            ShaderHandle badEntry = dev->createShader(badEntryDesc, validBinary.data(), validBinary.size());
            if (fix.api() == GraphicsAPI::VULKAN && badEntry != nullptr)
            {
                BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(ShaderType::Compute)
                        .addItem(BindingLayoutItem::ConstantBuffer(0))
                        .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                        .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
                UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

                ComputePipelineHandle badPso = dev->createComputePipeline(
                    ComputePipelineDesc().setComputeShader(badEntry).addBindingLayout(layout));
                UAA_CHECK(badPso == nullptr, "Non-existent entry point must fail pipeline creation on Vulkan");
            }
            else
            {
                UAA_CHECK(badEntry == nullptr, "createShader with non-existent entry point must return nullptr");
            }
        }

        // Clear native validation messages produced deliberately by feeding invalid SPIR-V/entry points.
        if (fix.backend())
        {
            fix.backend()->clearNativeValidationMessages();
        }
    }

    static void testShaderSpecialisation(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (!fix.features().supports(Feature::ShaderSpecializations))
        {
            ctx.skip("ShaderSpecializations feature is not supported by this backend");
        }

        IDevice* dev = fix.device();
        ShaderHandle baseShader = fix.createShader("cs_spec_const", ShaderType::Compute, "main");
        UAA_CHECK(baseShader != nullptr, "createShader(cs_spec_const) returned null");

        BindingLayoutHandle layout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
        UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

        BufferHandle outBuf = createStorageBuffer(fix, 16, "shd_spec_out", 4, true);
        UAA_CHECK(outBuf != nullptr, "createBuffer returned null");

        BindingSetHandle bindingSet = dev->createBindingSet(
            BindingSetDesc().addItem(BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
            layout);
        UAA_CHECK(bindingSet != nullptr, "createBindingSet returned null");

        runSpecialisationTest(ctx, fix, baseShader, layout, bindingSet, outBuf, 7u, -3, 2.5f, "default");

        const ShaderSpecialization specsA[] = {
            ShaderSpecialization::UInt32(0, 42u),
            ShaderSpecialization::Int32(1, -99),
            ShaderSpecialization::Float(2, 12.25f)
        };
        const ShaderSpecialization specsB[] = {
            ShaderSpecialization::UInt32(0, 1337u),
            ShaderSpecialization::Int32(1, 555),
            ShaderSpecialization::Float(2, -0.5f)
        };

        ShaderHandle specShaderA = dev->createShaderSpecialization(baseShader, specsA, 3);
        ShaderHandle specShaderB = dev->createShaderSpecialization(baseShader, specsB, 3);
        UAA_CHECK(specShaderA != nullptr, "createShaderSpecialization(A) returned null");
        UAA_CHECK(specShaderB != nullptr, "createShaderSpecialization(B) returned null");

        runSpecialisationTest(ctx, fix, specShaderA, layout, bindingSet, outBuf, 42u, -99, 12.25f, "specA");
        runSpecialisationTest(ctx, fix, specShaderB, layout, bindingSet, outBuf, 1337u, 555, -0.5f, "specB");
        runSpecialisationTest(ctx, fix, baseShader, layout, bindingSet, outBuf, 7u, -3, 2.5f, "default_isolation");
    }

    static void testShaderLibrary(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        if (fix.api() == GraphicsAPI::METAL || fix.api() == GraphicsAPI::D3D11)
        {
            const std::vector<uint8_t> bytes = fix.loadShaderBinary("cs_buffer_copy_add");
            {
                ExpectErrorsGuard guard(fix);
                ShaderLibraryHandle lib = dev->createShaderLibrary(bytes.data(), bytes.size());
                UAA_CHECK(lib == nullptr, "Unsupported createShaderLibrary must return nullptr cleanly");
            }
            ctx.skip("createShaderLibrary is not supported on Metal or D3D11");
        }

        if (fix.api() == GraphicsAPI::VULKAN)
        {
            ShaderLibraryHandle lib;
            {
                std::vector<uint8_t> bytes = fix.loadShaderBinary("cs_multi_entry");
                lib = dev->createShaderLibrary(bytes.data(), bytes.size());
                std::fill(bytes.begin(), bytes.end(), uint8_t(0));
            }
            UAA_CHECK(lib != nullptr, "createShaderLibrary returned null on Vulkan");

            ShaderHandle shAdd = lib->getShader("entry_add", ShaderType::Compute);
            ShaderHandle shMul = lib->getShader("entry_mul", ShaderType::Compute);
            UAA_CHECK(shAdd != nullptr, "ShaderLibrary::getShader(entry_add) returned null");
            UAA_CHECK(shMul != nullptr, "ShaderLibrary::getShader(entry_mul) returned null");

            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
            UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

            BufferHandle outBuf = createStorageBuffer(fix, 16, "shd_lib_out", 4, true);
            UAA_CHECK(outBuf != nullptr, "createBuffer returned null");

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc().addItem(BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
                layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");

            ComputePipelineHandle psoAdd = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(shAdd).addBindingLayout(layout));
            ComputePipelineHandle psoMul = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(shMul).addBindingLayout(layout));
            UAA_CHECK(psoAdd != nullptr, "createComputePipeline(entry_add) returned null");
            UAA_CHECK(psoMul != nullptr, "createComputePipeline(entry_mul) returned null");

            const std::array<uint32_t, 4> initData = { 10u, 0u, 0u, 0u };
            std::array<uint32_t, 4> result = {};

            fix.uploadBuffer(outBuf, initData.data(), sizeof(initData));
            dispatchAndWait(fix, psoAdd, { set });
            fix.readbackBuffer(outBuf, result.data(), sizeof(result));
            UAA_CHECK_EQ(result[1], 27u, "ShaderLibrary entry_add output mismatch");

            fix.uploadBuffer(outBuf, initData.data(), sizeof(initData));
            dispatchAndWait(fix, psoMul, { set });
            fix.readbackBuffer(outBuf, result.data(), sizeof(result));
            UAA_CHECK_EQ(result[1], 30u, "ShaderLibrary entry_mul output mismatch");

            return;
        }

        // D3D12 shader library path (rt_lib DXIL library).
        ShaderLibraryHandle lib;
        {
            std::vector<uint8_t> bytes = fix.loadShaderBinary("rt_lib");
            lib = dev->createShaderLibrary(bytes.data(), bytes.size());
            std::fill(bytes.begin(), bytes.end(), uint8_t(0));
        }
        UAA_CHECK(lib != nullptr, "createShaderLibrary(rt_lib) returned null on D3D12");

        const void* bc = nullptr;
        size_t bcSize = 0;
        lib->getBytecode(&bc, &bcSize);
        UAA_CHECK(bc != nullptr && bcSize > 0, "ShaderLibrary::getBytecode must remain valid after source vector freed");

        ShaderHandle rg = lib->getShader("RayGen", ShaderType::RayGeneration);
        ShaderHandle miss = lib->getShader("Miss", ShaderType::Miss);
        UAA_CHECK(rg != nullptr, "ShaderLibrary::getShader(RayGen) returned null");
        UAA_CHECK(miss != nullptr, "ShaderLibrary::getShader(Miss) returned null");

        ShaderHandle missing = lib->getShader("does_not_exist", ShaderType::Compute);
        if (missing != nullptr)
        {
            ctx.note("D3D12 ShaderLibraryEntry defers entry-point lookup until RTPSO creation");
        }
    }

    static void testSpirvCompatibility(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (fix.api() != GraphicsAPI::METAL && fix.api() != GraphicsAPI::VULKAN)
        {
            ctx.skip("Frozen Metal SPIR-V compatibility test applies only to Vulkan and Metal backends");
        }

        IDevice* dev = fix.device();

        SamplerHandle pointClampSampler = dev->createSampler(
            SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
        UAA_CHECK(pointClampSampler != nullptr, "createSampler returned null");

        // Verify cs_metal_custom_entry with custom entry point, push constants, typed SRV, texture, and sampler.
        {
            ShaderHandle cs = fix.createShader("cs_metal_custom_entry", ShaderType::Compute, "uaa_custom_entry", true);
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::PushConstants(0, 16))
                    .addItem(BindingLayoutItem::TypedBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(1))
                    .addItem(BindingLayoutItem::Texture_SRV(2))
                    .addItem(BindingLayoutItem::Sampler(3)));
            UAA_CHECK(layout != nullptr, "cs_metal_custom_entry layout null");

            BufferHandle typedIn = createStorageBuffer(fix, 16, "spv_custom_typed", 4, false, true, false, Format::R32_UINT);
            BufferHandle outBuf = createStorageBuffer(fix, 16, "spv_custom_out", 4, true);
            TextureHandle tex = createRgba8Texture2D(fix, 2, 2, 20, 40, 60, 255, "spv_custom_tex");
            UAA_CHECK(typedIn != nullptr && outBuf != nullptr && tex != nullptr, "cs_metal_custom_entry resources null");

            const std::array<uint32_t, 4> inWords = { 23u, 0u, 0u, 0u };
            fix.uploadBuffer(typedIn, inWords.data(), sizeof(inWords));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::PushConstants(0, 16))
                    .addItem(BindingSetItem::TypedBuffer_SRV(0, typedIn, Format::R32_UINT, BufferRange(0, typedIn->getDesc().byteSize)))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(1, outBuf))
                    .addItem(BindingSetItem::Texture_SRV(2, tex))
                    .addItem(BindingSetItem::Sampler(3, pointClampSampler)),
                layout);
            UAA_CHECK(set != nullptr, "cs_metal_custom_entry binding set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "cs_metal_custom_entry pipeline null");

            const std::array<uint32_t, 4> pc = { 100u, 0u, 0u, 0u };
            dispatchAndWait(fix, pso, { set }, pc.data(), sizeof(pc));

            std::array<uint32_t, 4> outWords = {};
            fix.readbackBuffer(outBuf, outWords.data(), sizeof(outWords));
            UAA_CHECK_EQ(outWords[0], 123u, "cs_metal_custom_entry Out[0] mismatch");
            UAA_CHECK_EQ(outWords[1], packRgba8(20, 40, 60, 255), "cs_metal_custom_entry Out[1] texture sample mismatch");
        }

        // Verify frozen cs_push_constants.
        {
            ShaderHandle cs = fix.createShader("cs_push_constants", ShaderType::Compute, "main", true);
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::PushConstants(0, 16))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
            UAA_CHECK(layout != nullptr, "cs_push_constants layout null");

            BufferHandle outBuf = createStorageBuffer(fix, 16, "spv_pc_out", 4, true);
            UAA_CHECK(outBuf != nullptr, "cs_push_constants outBuf null");

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::PushConstants(0, 16))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
                layout);
            UAA_CHECK(set != nullptr, "cs_push_constants set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "cs_push_constants pipeline null");

            const std::array<uint32_t, 4> pc = { 2u, 0x1234u, 0x00FFu, 10u };
            dispatchAndWait(fix, pso, { set }, pc.data(), sizeof(pc));

            std::array<uint32_t, 4> outWords = {};
            fix.readbackBuffer(outBuf, outWords.data(), sizeof(outWords));
            UAA_CHECK_EQ(outWords[2], (0x1234u ^ 0x00FFu) + 10u, "cs_push_constants Out[2] mismatch");
        }

        // Verify frozen cs_sampler_test.
        {
            ShaderHandle cs = fix.createShader("cs_sampler_test", ShaderType::Compute, "main", true);
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::Texture_SRV(0))
                    .addItem(BindingLayoutItem::Sampler(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(layout != nullptr, "cs_sampler_test layout null");

            TextureHandle tex = createRgba8Texture2D(fix, 2, 2, 51, 102, 153, 255, "spv_sampler_tex");
            BufferHandle outBuf = createStorageBuffer(fix, 16, "spv_sampler_out", 16, true);
            BufferHandle cbBuf = createConstantBuffer(fix, 256, "spv_sampler_cb");
            UAA_CHECK(tex != nullptr && outBuf != nullptr && cbBuf != nullptr, "cs_sampler_test resources null");

            struct SamplerParams
            {
                float uvOrigin[2];
                float uvStep[2];
                float lod;
                uint32_t width;
                uint32_t height;
                uint32_t mode;
            };
            const SamplerParams params = { { 0.5f, 0.5f }, { 0.0f, 0.0f }, 0.0f, 1u, 1u, 0u };
            fix.uploadBuffer(cbBuf, &params, sizeof(params));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::Texture_SRV(0, tex))
                    .addItem(BindingSetItem::Sampler(1, pointClampSampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                    .addItem(BindingSetItem::ConstantBuffer(3, cbBuf)),
                layout);
            UAA_CHECK(set != nullptr, "cs_sampler_test set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "cs_sampler_test pipeline null");

            dispatchAndWait(fix, pso, { set });

            std::array<float, 4> rgba = {};
            fix.readbackBuffer(outBuf, rgba.data(), sizeof(rgba));
            UAA_CHECK_NEAR(rgba[0], 51.0f / 255.0f, 0.01f, "cs_sampler_test R mismatch");
            UAA_CHECK_NEAR(rgba[1], 102.0f / 255.0f, 0.01f, "cs_sampler_test G mismatch");
            UAA_CHECK_NEAR(rgba[2], 153.0f / 255.0f, 0.01f, "cs_sampler_test B mismatch");
            UAA_CHECK_NEAR(rgba[3], 1.0f, 0.01f, "cs_sampler_test A mismatch");
        }

        // Verify frozen cs_raw_typed_rw.
        {
            ShaderHandle cs = fix.createShader("cs_raw_typed_rw", ShaderType::Compute, "main", true);
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::RawBuffer_SRV(0))
                    .addItem(BindingLayoutItem::RawBuffer_UAV(1))
                    .addItem(BindingLayoutItem::TypedBuffer_SRV(2))
                    .addItem(BindingLayoutItem::TypedBuffer_UAV(3))
                    .addItem(BindingLayoutItem::ConstantBuffer(4)));
            UAA_CHECK(layout != nullptr, "cs_raw_typed_rw layout null");

            BufferHandle rawIn = createStorageBuffer(fix, 16, "spv_rt_raw_in", 0, false, false, true);
            BufferHandle rawOut = createStorageBuffer(fix, 16, "spv_rt_raw_out", 0, true, false, true);
            BufferHandle typedIn = createStorageBuffer(fix, 16, "spv_rt_typed_in", 4, false, true, false, Format::R32_UINT);
            BufferHandle typedOut = createStorageBuffer(fix, 16, "spv_rt_typed_out", 4, true, true, false, Format::R32_UINT);
            BufferHandle cbBuf = createConstantBuffer(fix, 256, "spv_rt_cb");
            UAA_CHECK(rawIn && rawOut && typedIn && typedOut && cbBuf, "cs_raw_typed_rw buffers null");

            const std::array<uint32_t, 4> inWords = { 10u, 20u, 30u, 40u };
            fix.uploadBuffer(rawIn, inWords.data(), sizeof(inWords));
            fix.uploadBuffer(typedIn, inWords.data(), sizeof(inWords));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::RawBuffer_SRV(0, rawIn))
                    .addItem(BindingSetItem::RawBuffer_UAV(1, rawOut))
                    .addItem(BindingSetItem::TypedBuffer_SRV(2, typedIn, Format::R32_UINT, BufferRange(0, typedIn->getDesc().byteSize)))
                    .addItem(BindingSetItem::TypedBuffer_UAV(3, typedOut, Format::R32_UINT, BufferRange(0, typedOut->getDesc().byteSize)))
                    .addItem(BindingSetItem::ConstantBuffer(4, cbBuf)),
                layout);
            UAA_CHECK(set != nullptr, "cs_raw_typed_rw set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "cs_raw_typed_rw pipeline null");

            const std::array<uint32_t, 4> paramsRaw = { 4u, 0u, 5u, 0u };
            fix.uploadBuffer(cbBuf, paramsRaw.data(), sizeof(paramsRaw));
            dispatchAndWait(fix, pso, { set });

            std::array<uint32_t, 4> actual = {};
            fix.readbackBuffer(rawOut, actual.data(), sizeof(actual));
            for (uint32_t i = 0; i < 4u; ++i)
            {
                UAA_CHECK_EQ(actual[i], inWords[i] + 5u, "cs_raw_typed_rw raw output mismatch");
            }

            const std::array<uint32_t, 4> paramsTyped = { 4u, 3u, 7u, 0u };
            fix.uploadBuffer(cbBuf, paramsTyped.data(), sizeof(paramsTyped));
            dispatchAndWait(fix, pso, { set });

            fix.readbackBuffer(typedOut, actual.data(), sizeof(actual));
            for (uint32_t i = 0; i < 4u; ++i)
            {
                UAA_CHECK_EQ(actual[i], inWords[i] + 7u, "cs_raw_typed_rw typed output mismatch");
            }
        }

        // Verify frozen cs_bindings_all.
        {
            ShaderHandle cs = fix.createShader("cs_bindings_all", ShaderType::Compute, "main", true);
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::RawBuffer_SRV(2))
                    .addItem(BindingLayoutItem::TypedBuffer_SRV(3))
                    .addItem(BindingLayoutItem::Texture_SRV(4))
                    .addItem(BindingLayoutItem::Sampler(5))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(6)));
            UAA_CHECK(layout != nullptr, "cs_bindings_all layout null");

            BufferHandle cbBuf = createConstantBuffer(fix, 256, "spv_all_cb");
            BufferHandle structIn = createStorageBuffer(fix, 16, "spv_all_struct", 4);
            BufferHandle rawIn = createStorageBuffer(fix, 16, "spv_all_raw", 0, false, false, true);
            BufferHandle typedIn = createStorageBuffer(fix, 16, "spv_all_typed", 4, false, true, false, Format::R32_UINT);
            TextureHandle tex = createRgba8Texture2D(fix, 2, 2, 11, 22, 33, 255, "spv_all_tex");
            BufferHandle outBuf = createStorageBuffer(fix, 32, "spv_all_out", 4, true);
            UAA_CHECK(cbBuf && structIn && rawIn && typedIn && tex && outBuf, "cs_bindings_all resources null");

            const std::array<uint32_t, 4> cbWords = { 101u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> sWords = { 202u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> rWords = { 303u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> tWords = { 404u, 0u, 0u, 0u };
            fix.uploadBuffer(cbBuf, cbWords.data(), sizeof(cbWords));
            fix.uploadBuffer(structIn, sWords.data(), sizeof(sWords));
            fix.uploadBuffer(rawIn, rWords.data(), sizeof(rWords));
            fix.uploadBuffer(typedIn, tWords.data(), sizeof(tWords));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, structIn))
                    .addItem(BindingSetItem::RawBuffer_SRV(2, rawIn))
                    .addItem(BindingSetItem::TypedBuffer_SRV(3, typedIn, Format::R32_UINT, BufferRange(0, typedIn->getDesc().byteSize)))
                    .addItem(BindingSetItem::Texture_SRV(4, tex))
                    .addItem(BindingSetItem::Sampler(5, pointClampSampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(6, outBuf)),
                layout);
            UAA_CHECK(set != nullptr, "cs_bindings_all set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "cs_bindings_all pipeline null");

            dispatchAndWait(fix, pso, { set });

            std::array<uint32_t, 8> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], 101u, "cs_bindings_all Out[0] mismatch");
            UAA_CHECK_EQ(actual[1], 202u, "cs_bindings_all Out[1] mismatch");
            UAA_CHECK_EQ(actual[2], 303u, "cs_bindings_all Out[2] mismatch");
            UAA_CHECK_EQ(actual[3], 404u, "cs_bindings_all Out[3] mismatch");
            UAA_CHECK_EQ(actual[4], packRgba8(11, 22, 33, 255), "cs_bindings_all Out[4] mismatch");
        }

        // Verify frozen cs_multi_set.
        {
            ShaderHandle cs = fix.createShader("cs_multi_set", ShaderType::Compute, "main", true);
            BindingLayoutHandle layout0 = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0, true)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0)));
            BindingLayoutHandle layout1 = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 1, true)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(1)));
            UAA_CHECK(layout0 != nullptr && layout1 != nullptr, "cs_multi_set layouts null");

            BufferHandle inA = createStorageBuffer(fix, 16, "spv_ms_inA", 4);
            BufferHandle inB = createStorageBuffer(fix, 16, "spv_ms_inB", 4);
            BufferHandle outBuf = createStorageBuffer(fix, 16, "spv_ms_out", 4, true);
            UAA_CHECK(inA && inB && outBuf, "cs_multi_set buffers null");

            const std::array<uint32_t, 4> aWords = { 7u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> bWords = { 4u, 0u, 0u, 0u };
            fix.uploadBuffer(inA, aWords.data(), sizeof(aWords));
            fix.uploadBuffer(inB, bWords.data(), sizeof(bWords));

            BindingSetHandle set0 = dev->createBindingSet(
                BindingSetDesc().addItem(BindingSetItem::StructuredBuffer_SRV(0, inA)),
                layout0);
            BindingSetHandle set1 = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, inB))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(1, outBuf)),
                layout1);
            UAA_CHECK(set0 != nullptr && set1 != nullptr, "cs_multi_set sets null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(cs)
                    .addBindingLayout(layout0)
                    .addBindingLayout(layout1));
            UAA_CHECK(pso != nullptr, "cs_multi_set pipeline null");

            dispatchAndWait(fix, pso, { set0, set1 });

            std::array<uint32_t, 4> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], 4007u, "cs_multi_set Out[0] mismatch");
        }

        // Verify frozen cs_bindless_tex_buf with bindless set 0 and regular set 1.
        {
            ShaderHandle cs = fix.createShader("cs_bindless_tex_buf", ShaderType::Compute, "main", true);

            BindlessLayoutDesc bindlessDesc;
            bindlessDesc.setVisibility(ShaderType::Compute)
                .setMaxCapacity(4)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0))
                .addRegisterSpace(BindingLayoutItem::StructuredBuffer_SRV(1));
            BindingLayoutHandle bindlessLayout = dev->createBindlessLayout(bindlessDesc);
            UAA_CHECK(bindlessLayout != nullptr, "cs_bindless_tex_buf bindlessLayout null");

            BindingLayoutHandle regularLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0, false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(BindingLayoutItem::Sampler(3))
                    .addItem(BindingLayoutItem::ConstantBuffer(4)));
            UAA_CHECK(regularLayout != nullptr, "cs_bindless_tex_buf regularLayout null");

            DescriptorTableHandle table = dev->createDescriptorTable(bindlessLayout);
            UAA_CHECK(table != nullptr, "cs_bindless_tex_buf createDescriptorTable null");

            TextureHandle tex0 = createRgba8Texture2D(fix, 2, 2, 10, 20, 30, 255, "spv_bl_tex0");
            TextureHandle tex1 = createRgba8Texture2D(fix, 2, 2, 40, 50, 60, 255, "spv_bl_tex1");
            BufferHandle buf0 = createStorageBuffer(fix, 16, "spv_bl_buf0", 4);
            BufferHandle buf1 = createStorageBuffer(fix, 16, "spv_bl_buf1", 4);
            UAA_CHECK(tex0 && tex1 && buf0 && buf1, "cs_bindless_tex_buf table resources null");

            const std::array<uint32_t, 4> b0Words = { 1111u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> b1Words = { 2222u, 0u, 0u, 0u };
            fix.uploadBuffer(buf0, b0Words.data(), sizeof(b0Words));
            fix.uploadBuffer(buf1, b1Words.data(), sizeof(b1Words));

            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(0, tex0)), "writeDescriptorTable tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(1, tex1)), "writeDescriptorTable tex1 failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "writeDescriptorTable buf0 failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::StructuredBuffer_SRV(1, buf1)), "writeDescriptorTable buf1 failed");

            BufferHandle texIndices = createStorageBuffer(fix, 16, "spv_bl_tex_idx", 4);
            BufferHandle bufIndices = createStorageBuffer(fix, 16, "spv_bl_buf_idx", 4);
            BufferHandle outBuf = createStorageBuffer(fix, 32, "spv_bl_out", 4, true);
            BufferHandle cbBuf = createConstantBuffer(fix, 256, "spv_bl_cb");
            UAA_CHECK(texIndices && bufIndices && outBuf && cbBuf, "cs_bindless_tex_buf regular buffers null");

            const std::array<uint32_t, 4> tIdx = { 1u, 0u, 0u, 0u };
            const std::array<uint32_t, 4> bIdx = { 0u, 1u, 0u, 0u };
            const std::array<uint32_t, 4> params = { 2u, 0u, 0u, 0u };
            fix.uploadBuffer(texIndices, tIdx.data(), sizeof(tIdx));
            fix.uploadBuffer(bufIndices, bIdx.data(), sizeof(bIdx));
            fix.uploadBuffer(cbBuf, params.data(), sizeof(params));

            BindingSetHandle regularSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, texIndices))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, bufIndices))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                    .addItem(BindingSetItem::Sampler(3, pointClampSampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, cbBuf)),
                regularLayout);
            UAA_CHECK(regularSet != nullptr, "cs_bindless_tex_buf regularSet null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(cs)
                    .addBindingLayout(bindlessLayout)
                    .addBindingLayout(regularLayout));
            UAA_CHECK(pso != nullptr, "cs_bindless_tex_buf pipeline null");

            dispatchAndWait(fix, pso, { table, regularSet });

            std::array<uint32_t, 8> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], packRgba8(40, 50, 60, 255), "cs_bindless_tex_buf lane 0 tex mismatch");
            UAA_CHECK_EQ(actual[1], 1111u, "cs_bindless_tex_buf lane 0 buf mismatch");
            UAA_CHECK_EQ(actual[2], packRgba8(10, 20, 30, 255), "cs_bindless_tex_buf lane 1 tex mismatch");
            UAA_CHECK_EQ(actual[3], 2222u, "cs_bindless_tex_buf lane 1 buf mismatch");
        }
    }

    static void testPipelineFailure(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (!fix.isValidationActive())
        {
            ctx.skip("Pipeline invalid-argument failure checks require NVRHI validation layer active");
        }

        IDevice* dev = fix.device();
        ShaderHandle cs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");
        ShaderHandle vs = fix.createShader("vs_fullscreen_tri", ShaderType::Vertex, "main");
        ShaderHandle ps = fix.createShader("ps_solid_color", ShaderType::Pixel, "main");

        {
            ExpectErrorsGuard guard(fix);

            // Missing compute shader.
            ComputePipelineHandle nullCsPso = dev->createComputePipeline(ComputePipelineDesc{});
            UAA_CHECK(nullCsPso == nullptr, "ComputePipelineDesc with null CS must return nullptr");

            // Wrong shader stage passed as CS.
            ComputePipelineHandle wrongStagePso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(vs));
            UAA_CHECK(wrongStagePso == nullptr, "ComputePipelineDesc with Vertex shader must return nullptr");

            // Mixed registerSpaceIsDescriptorSet across binding layouts.
            BindingLayoutHandle layoutTrue = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0, true)
                    .addItem(BindingLayoutItem::ConstantBuffer(0)));
            BindingLayoutHandle layoutFalse = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0, false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1)));
            UAA_CHECK(layoutTrue != nullptr && layoutFalse != nullptr, "createBindingLayout failed");

            ComputePipelineHandle mixedLayoutPso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(cs)
                    .addBindingLayout(layoutTrue)
                    .addBindingLayout(layoutFalse));
            UAA_CHECK(mixedLayoutPso == nullptr, "ComputePipelineDesc with mixed registerSpaceIsDescriptorSet must return nullptr");

            // Mismatched FramebufferInfo: depth test enabled without depth format.
            GraphicsPipelineDesc badDepthGfxDesc;
            badDepthGfxDesc.setVertexShader(vs).setPixelShader(ps);
            badDepthGfxDesc.renderState.depthStencilState.depthTestEnable = true;

            FramebufferInfo noDepthFbInfo;
            noDepthFbInfo.addColorFormat(Format::RGBA8_UNORM);
            GraphicsPipelineHandle badDepthPso = dev->createGraphicsPipeline(badDepthGfxDesc, noDepthFbInfo);
            UAA_CHECK(badDepthPso == nullptr, "GraphicsPipeline with depthTestEnable and no depth format must return nullptr");

            // Wrong shader stage passed as VS in graphics pipeline.
            GraphicsPipelineDesc wrongVsDesc;
            wrongVsDesc.setVertexShader(cs).setPixelShader(ps);
            GraphicsPipelineHandle wrongVsPso = dev->createGraphicsPipeline(wrongVsDesc, noDepthFbInfo);
            UAA_CHECK(wrongVsPso == nullptr, "GraphicsPipeline with Compute shader as VS must return nullptr");

            // Missing vertex shader (validated natively on Metal; on Vulkan clear expected native VUID).
            GraphicsPipelineDesc missingVsDesc;
            missingVsDesc.setPixelShader(ps);
            GraphicsPipelineHandle missingVsPso = dev->createGraphicsPipeline(missingVsDesc, noDepthFbInfo);
            UAA_CHECK(missingVsPso == nullptr, "GraphicsPipeline with missing VS must return nullptr");

            UAA_CHECK(guard.errorCount() > 0, "Expected validation errors during SHD.PipelineFailure");
        }

        if (fix.backend())
        {
            fix.backend()->clearNativeValidationMessages();
        }

        // Recover on the same device by creating and executing a valid compute pipeline.
        BindingLayoutHandle validLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
        UAA_CHECK(validLayout != nullptr, "Recovery createBindingLayout returned null");

        BufferHandle cbBuf = createConstantBuffer(fix, 256, "shd_recover_cb");
        BufferHandle inBuf = createStorageBuffer(fix, 16, "shd_recover_in", 4);
        BufferHandle outBuf = createStorageBuffer(fix, 16, "shd_recover_out", 4, true);
        UAA_CHECK(cbBuf && inBuf && outBuf, "Recovery buffers returned null");

        const std::array<uint32_t, 4> params = { 4u, 10u, 3u, 0u };
        const std::array<uint32_t, 4> inputs = { 1u, 2u, 3u, 4u };
        fix.uploadBuffer(cbBuf, params.data(), sizeof(params));
        fix.uploadBuffer(inBuf, inputs.data(), sizeof(inputs));

        BindingSetHandle validSet = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::ConstantBuffer(0, cbBuf))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf)),
            validLayout);
        UAA_CHECK(validSet != nullptr, "Recovery createBindingSet returned null");

        ComputePipelineHandle validPso = dev->createComputePipeline(
            ComputePipelineDesc().setComputeShader(cs).addBindingLayout(validLayout));
        UAA_CHECK(validPso != nullptr, "Recovery createComputePipeline returned null");

        dispatchAndWait(fix, validPso, { validSet });

        std::array<uint32_t, 4> actual = {};
        fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
        for (uint32_t i = 0; i < 4u; ++i)
        {
            UAA_CHECK_EQ(actual[i], (inputs[i] + 10u) ^ 3u, "Recovery dispatch output mismatch");
        }
    }

    UAA_REGISTER_TEST("SHD.Stages", "core", testShaderStages);
    UAA_REGISTER_TEST("SHD.EntryPoints", "core", testShaderEntryPoints);
    UAA_REGISTER_TEST("SHD.InvalidBinary", "validation,isolated", testShaderInvalidBinary);
    UAA_REGISTER_TEST("SHD.Specialisation", "core", testShaderSpecialisation);
    UAA_REGISTER_TEST("SHD.Library", "core", testShaderLibrary);
    UAA_REGISTER_TEST("SHD.SpirvCompatibility", "core", testSpirvCompatibility);
    UAA_REGISTER_TEST("SHD.PipelineFailure", "core", testPipelineFailure);
} // namespace nvrhi::uaa
