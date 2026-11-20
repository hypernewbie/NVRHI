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

        void fillTextureSubresourceRgba8(
            const TestFixture& fix,
            ITexture* tex,
            uint32_t slice,
            uint32_t mip,
            uint32_t width,
            uint32_t height,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a)
        {
            std::vector<uint8_t> pixels(size_t(width) * size_t(height) * 4u);
            for (size_t i = 0; i < size_t(width) * size_t(height); ++i)
            {
                pixels[i * 4u + 0u] = r;
                pixels[i * 4u + 1u] = g;
                pixels[i * 4u + 2u] = b;
                pixels[i * 4u + 3u] = a;
            }
            fix.uploadTexture2D(tex, slice, mip, pixels.data(), size_t(width) * 4u);
        }

        void dispatchAndWait(
            const TestFixture& fix,
            IComputePipeline* pso,
            const std::vector<IBindingSet*>& sets,
            const void* pushData = nullptr,
            size_t pushSize = 0)
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
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);
        }
    } // namespace

    static void testBindingLayout(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        // Empty layout.
        {
            BindingLayoutHandle emptyLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute));
            UAA_CHECK(emptyLayout != nullptr, "Empty BindingLayout must not be null");
            const BindingLayoutDesc* desc = emptyLayout->getDesc();
            UAA_CHECK(desc != nullptr, "Regular BindingLayout getDesc() must not be null");
            UAA_CHECK(emptyLayout->getBindlessDesc() == nullptr, "Regular BindingLayout getBindlessDesc() must be null");
            UAA_CHECK_EQ(desc->visibility, ShaderType::Compute, "Empty layout visibility mismatch");
            UAA_CHECK(desc->bindings.empty(), "Empty layout bindings must be empty");
        }

        // Single-stage layout.
        {
            BindingLayoutHandle csLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(csLayout != nullptr, "Compute BindingLayout must not be null");
            const BindingLayoutDesc* desc = csLayout->getDesc();
            UAA_CHECK(desc != nullptr, "Compute layout getDesc() null");
            UAA_CHECK_EQ(desc->visibility, ShaderType::Compute, "Compute layout visibility mismatch");
            UAA_CHECK_EQ(desc->bindings.size(), size_t(3), "Compute layout item count mismatch");
            UAA_CHECK_EQ(desc->bindings[0].type, ResourceType::ConstantBuffer, "Slot 0 type mismatch");
            UAA_CHECK_EQ(desc->bindings[1].slot, 1u, "Slot 1 index mismatch");
            UAA_CHECK_EQ(desc->bindings[2].type, ResourceType::StructuredBuffer_UAV, "Slot 2 type mismatch");
        }

        // Multi-stage layouts.
        {
            BindingLayoutHandle vpLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Vertex | ShaderType::Pixel)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::Texture_SRV(1))
                    .addItem(BindingLayoutItem::Sampler(2)));
            UAA_CHECK(vpLayout != nullptr, "Vertex|Pixel BindingLayout must not be null");
            UAA_CHECK_EQ(vpLayout->getDesc()->visibility, ShaderType::Vertex | ShaderType::Pixel,
                "Vertex|Pixel visibility mismatch");

            BindingLayoutHandle allLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::All)
                    .addItem(BindingLayoutItem::ConstantBuffer(0)));
            UAA_CHECK(allLayout != nullptr, "ShaderType::All BindingLayout must not be null");
            UAA_CHECK_EQ(allLayout->getDesc()->visibility, ShaderType::All, "All visibility mismatch");
        }

        // Nonzero registerSpace mapped as descriptor set.
        {
            BindingLayoutHandle dsLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Pixel, 2u, true)
                    .addItem(BindingLayoutItem::ConstantBuffer(0)));
            UAA_CHECK(dsLayout != nullptr, "registerSpaceIsDescriptorSet=true layout must not be null");
            const BindingLayoutDesc* desc = dsLayout->getDesc();
            UAA_CHECK(desc != nullptr, "Descriptor-set layout getDesc() null");
            UAA_CHECK_EQ(desc->registerSpace, 2u, "registerSpace mismatch");
            UAA_CHECK(desc->registerSpaceIsDescriptorSet, "registerSpaceIsDescriptorSet flag mismatch");
        }
    }

    static void testBindingSet(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        ShaderHandle cs = fix.createShader("cs_bindings_all", ShaderType::Compute, "main");
        BindingLayoutHandle layout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::RawBuffer_SRV(2))
                .addItem(BindingLayoutItem::TypedBuffer_SRV(3))
                .addItem(BindingLayoutItem::Texture_SRV(4))
                .addItem(BindingLayoutItem::Sampler(5))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(6)));
        UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

        BufferHandle cbBuf = createConstantBuffer(fix, 256, "bnd_all_cb");
        BufferHandle structIn = createStorageBuffer(fix, 16, "bnd_all_struct", 4);
        BufferHandle rawIn = createStorageBuffer(fix, 16, "bnd_all_raw", 0, false, false, true);
        BufferHandle typedIn = createStorageBuffer(fix, 16, "bnd_all_typed", 4, false, true, false, Format::R32_UINT);
        BufferHandle outBuf = createStorageBuffer(fix, 32, "bnd_all_out", 4, true);
        UAA_CHECK(cbBuf && structIn && rawIn && typedIn && outBuf, "Buffer creation failed");

        TextureDesc texDesc;
        texDesc.width = 2;
        texDesc.height = 2;
        texDesc.format = Format::RGBA8_UNORM;
        texDesc.isShaderResource = true;
        texDesc.initialState = ResourceStates::ShaderResource;
        texDesc.keepInitialState = true;
        texDesc.debugName = "bnd_all_tex";
        TextureHandle tex = dev->createTexture(texDesc);
        UAA_CHECK(tex != nullptr, "createTexture returned null");
        fillTextureSubresourceRgba8(fix, tex, 0, 0, 2, 2, 32, 96, 160, 224);

        SamplerHandle sampler = dev->createSampler(
            SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
        UAA_CHECK(sampler != nullptr, "createSampler returned null");

        const std::array<uint32_t, 4> cbWords = { 0x11223344u, 0u, 0u, 0u };
        const std::array<uint32_t, 4> structWords = { 0xAABBCCDDu, 0u, 0u, 0u };
        const std::array<uint32_t, 4> rawWords = { 0x55667788u, 0u, 0u, 0u };
        const std::array<uint32_t, 4> typedWords = { 0x9900AABBu, 0u, 0u, 0u };
        fix.uploadBuffer(cbBuf, cbWords.data(), sizeof(cbWords));
        fix.uploadBuffer(structIn, structWords.data(), sizeof(structWords));
        fix.uploadBuffer(rawIn, rawWords.data(), sizeof(rawWords));
        fix.uploadBuffer(typedIn, typedWords.data(), sizeof(typedWords));

        BindingSetHandle set = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::ConstantBuffer(0, cbBuf))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, structIn))
                .addItem(BindingSetItem::RawBuffer_SRV(2, rawIn))
                .addItem(BindingSetItem::TypedBuffer_SRV(3, typedIn, Format::R32_UINT))
                .addItem(BindingSetItem::Texture_SRV(4, tex))
                .addItem(BindingSetItem::Sampler(5, sampler))
                .addItem(BindingSetItem::StructuredBuffer_UAV(6, outBuf)),
            layout);
        UAA_CHECK(set != nullptr, "createBindingSet returned null");
        UAA_CHECK(set->getDesc() != nullptr, "BindingSet getDesc() must not be null");
        UAA_CHECK_EQ(set->getLayout(), layout.Get(), "BindingSet getLayout() mismatch");

        ComputePipelineHandle pso = dev->createComputePipeline(
            ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
        UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

        dispatchAndWait(fix, pso, { set });

        std::array<uint32_t, 8> actual = {};
        fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
        UAA_CHECK_EQ(actual[0], 0x11223344u, "ConstantBuffer binding mismatch");
        UAA_CHECK_EQ(actual[1], 0xAABBCCDDu, "StructuredBuffer_SRV binding mismatch");
        UAA_CHECK_EQ(actual[2], 0x55667788u, "RawBuffer_SRV binding mismatch");
        UAA_CHECK_EQ(actual[3], 0x9900AABBu, "TypedBuffer_SRV binding mismatch");
        UAA_CHECK_EQ(actual[4], packRgba8(32, 96, 160, 224), "Texture_SRV + Sampler binding mismatch");

        if (fix.features().supports(Feature::RayTracingAccelStruct) && fix.features().supports(Feature::RayQuery))
        {
            ShaderHandle rqCs = fix.createShader("cs_ray_query", ShaderType::Compute, "main");
            BindingLayoutHandle rtLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::RayTracingAccelStruct(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(1))
                    .addItem(BindingLayoutItem::ConstantBuffer(2)));
            UAA_CHECK(rtLayout != nullptr, "RT BindingLayout returned null");

            rt::AccelStructHandle tlas = dev->createAccelStruct(
                rt::AccelStructDesc().setTopLevelMaxInstances(1).setDebugName("bnd_empty_tlas"));
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned null");

            CommandListHandle buildCmd = dev->createCommandList();
            UAA_CHECK(buildCmd != nullptr, "createCommandList returned null");
            buildCmd->open();
            rt::InstanceDesc dummyInstance{};
            buildCmd->buildTopLevelAccelStruct(tlas, &dummyInstance, 0);
            buildCmd->close();
            fix.submitAndWait(buildCmd);

            BufferHandle rtOut = createStorageBuffer(fix, 32, "bnd_rt_out", 4, true);
            BufferHandle rtCb = createConstantBuffer(fix, 512, "bnd_rt_cb");
            UAA_CHECK(rtOut != nullptr && rtCb != nullptr, "RT buffers returned null");

            struct RayParams
            {
                float origins[8][4];
                float dirs[8][4];
                uint32_t rayMask;
                uint32_t rayCount;
                uint32_t p0;
                uint32_t p1;
            };
            RayParams rp{};
            rp.origins[0][2] = -2.0f;
            rp.dirs[0][2] = 1.0f;
            rp.rayMask = 0xFFu;
            rp.rayCount = 1u;
            fix.uploadBuffer(rtCb, &rp, sizeof(rp));

            const std::array<uint32_t, 4> initOut = { 0xFFFFFFFFu, 0u, 0u, 0u };
            fix.uploadBuffer(rtOut, initOut.data(), sizeof(initOut));

            BindingSetHandle rtSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::RayTracingAccelStruct(0, tlas))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(1, rtOut))
                    .addItem(BindingSetItem::ConstantBuffer(2, rtCb)),
                rtLayout);
            UAA_CHECK(rtSet != nullptr, "RT BindingSet returned null");

            ComputePipelineHandle rtPso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(rqCs).addBindingLayout(rtLayout));
            UAA_CHECK(rtPso != nullptr, "RT ComputePipeline returned null");

            dispatchAndWait(fix, rtPso, { rtSet });

            std::array<uint32_t, 4> rtActual = {};
            fix.readbackBuffer(rtOut, rtActual.data(), sizeof(rtActual));
            UAA_CHECK_EQ(rtActual[0], 0u, "Empty TLAS ray query expected miss (0)");
            ctx.note("Verified optional RayTracingAccelStruct binding with cs_ray_query");
        }
    }

    static void testBindingRanges(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (!fix.features().supports(Feature::ConstantBufferRanges))
        {
            ctx.skip("ConstantBufferRanges feature is not supported by this backend");
        }

        IDevice* dev = fix.device();
        ShaderHandle cs = fix.createShader("cs_bindings_all", ShaderType::Compute, "main");
        BindingLayoutHandle layout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::RawBuffer_SRV(2))
                .addItem(BindingLayoutItem::TypedBuffer_SRV(3))
                .addItem(BindingLayoutItem::Texture_SRV(4))
                .addItem(BindingLayoutItem::Sampler(5))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(6)));
        UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

        // Constant buffer with guard regions outside [256, 512).
        BufferHandle cbBuf = createConstantBuffer(fix, 768, "bnd_range_cb");
        UAA_CHECK(cbBuf != nullptr, "createBuffer(cb) returned null");
        std::vector<uint32_t> cbInit(768u / 4u, 0xDEAD0001u);
        cbInit[256u / 4u] = 0x13572468u;
        fix.uploadBuffer(cbBuf, cbInit.data(), cbInit.size() * sizeof(uint32_t));

        // Structured buffer with guard regions outside [256, 320).
        BufferHandle structBuf = createStorageBuffer(fix, 512, "bnd_range_struct", 4);
        UAA_CHECK(structBuf != nullptr, "createBuffer(struct) returned null");
        std::vector<uint32_t> structInit(512u / 4u, 0xBAD00002u);
        structInit[256u / 4u] = 0x2468ACE0u;
        fix.uploadBuffer(structBuf, structInit.data(), structInit.size() * sizeof(uint32_t));

        BufferHandle rawBuf = createStorageBuffer(fix, 16, "bnd_range_raw", 0, false, false, true);
        BufferHandle typedBuf = createStorageBuffer(fix, 16, "bnd_range_typed", 4, false, true, false, Format::R32_UINT);
        BufferHandle outBuf = createStorageBuffer(fix, 32, "bnd_range_out", 4, true);
        UAA_CHECK(rawBuf && typedBuf && outBuf, "Helper buffers returned null");

        const std::array<uint32_t, 4> dummyWords = { 77u, 0u, 0u, 0u };
        fix.uploadBuffer(rawBuf, dummyWords.data(), sizeof(dummyWords));
        fix.uploadBuffer(typedBuf, dummyWords.data(), sizeof(dummyWords));

        SamplerHandle sampler = dev->createSampler(
            SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
        UAA_CHECK(sampler != nullptr, "createSampler returned null");

        ComputePipelineHandle pso = dev->createComputePipeline(
            ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
        UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

        // Verify mip 1 of a mipmapped 2D texture in both raw and validation passes.
        {
            TextureDesc tex2DDesc;
            tex2DDesc.width = 8;
            tex2DDesc.height = 8;
            tex2DDesc.mipLevels = 3;
            tex2DDesc.format = Format::RGBA8_UNORM;
            tex2DDesc.isShaderResource = true;
            tex2DDesc.initialState = ResourceStates::ShaderResource;
            tex2DDesc.keepInitialState = true;
            tex2DDesc.debugName = "bnd_range_tex2d";
            TextureHandle tex2D = dev->createTexture(tex2DDesc);
            UAA_CHECK(tex2D != nullptr, "createTexture(tex2D) returned null");

            fillTextureSubresourceRgba8(fix, tex2D, 0, 0, 8, 8, 250, 10, 10, 255);
            fillTextureSubresourceRgba8(fix, tex2D, 0, 1, 4, 4, 40, 120, 200, 255);
            fillTextureSubresourceRgba8(fix, tex2D, 0, 2, 2, 2, 10, 250, 10, 255);

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf, BufferRange(256, 256)))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, structBuf, Format::UNKNOWN, BufferRange(256, 64)))
                    .addItem(BindingSetItem::RawBuffer_SRV(2, rawBuf))
                    .addItem(BindingSetItem::TypedBuffer_SRV(3, typedBuf, Format::R32_UINT))
                    .addItem(BindingSetItem::Texture_SRV(4, tex2D, Format::RGBA8_UNORM, TextureSubresourceSet(1, 1, 0, 1)))
                    .addItem(BindingSetItem::Sampler(5, sampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(6, outBuf)),
                layout);
            UAA_CHECK(set != nullptr, "createBindingSet(tex2D mip 1) returned null");

            dispatchAndWait(fix, pso, { set });

            std::array<uint32_t, 8> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], 0x13572468u, "CB range offset 256 mismatch");
            UAA_CHECK_EQ(actual[1], 0x2468ACE0u, "StructuredBuffer range offset 256 mismatch");
            UAA_CHECK_EQ(actual[4], packRgba8(40, 120, 200, 255), "Texture mip 1 subresource mismatch");
        }

        // Verify mip 1, slice 2 of a mipmapped Texture2DArray bound as a Texture2D view.
        // NVRHI's validation layer rejects Texture2DArray -> Texture2D in textureDimensionsCompatible().
        if (!fix.isValidationActive())
        {
            TextureDesc arrDesc;
            arrDesc.dimension = TextureDimension::Texture2DArray;
            arrDesc.width = 8;
            arrDesc.height = 8;
            arrDesc.mipLevels = 3;
            arrDesc.arraySize = 4;
            arrDesc.format = Format::RGBA8_UNORM;
            arrDesc.isShaderResource = true;
            arrDesc.initialState = ResourceStates::ShaderResource;
            arrDesc.keepInitialState = true;
            arrDesc.debugName = "bnd_range_arr";
            TextureHandle texArr = dev->createTexture(arrDesc);
            UAA_CHECK(texArr != nullptr, "createTexture(Texture2DArray) returned null");

            for (uint32_t slice = 0; slice < 4u; ++slice)
            {
                for (uint32_t mip = 0; mip < 3u; ++mip)
                {
                    const uint32_t w = 8u >> mip;
                    const uint32_t h = 8u >> mip;
                    if (slice == 2u && mip == 1u)
                    {
                        fillTextureSubresourceRgba8(fix, texArr, slice, mip, w, h, 44, 132, 220, 255);
                    }
                    else
                    {
                        fillTextureSubresourceRgba8(fix, texArr, slice, mip, w, h,
                            static_cast<uint8_t>(200u + slice * 10u),
                            static_cast<uint8_t>(20u + mip * 30u),
                            50u,
                            255u);
                    }
                }
            }

            BindingSetHandle arrSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf, BufferRange(256, 256)))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, structBuf, Format::UNKNOWN, BufferRange(256, 64)))
                    .addItem(BindingSetItem::RawBuffer_SRV(2, rawBuf))
                    .addItem(BindingSetItem::TypedBuffer_SRV(3, typedBuf, Format::R32_UINT))
                    .addItem(BindingSetItem::Texture_SRV(
                        4, texArr, Format::RGBA8_UNORM, TextureSubresourceSet(1, 1, 2, 1), TextureDimension::Texture2D))
                    .addItem(BindingSetItem::Sampler(5, sampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(6, outBuf)),
                layout);
            UAA_CHECK(arrSet != nullptr, "createBindingSet(Texture2DArray slice 2 mip 1) returned null");

            dispatchAndWait(fix, pso, { arrSet });

            std::array<uint32_t, 8> actual = {};
            fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
            UAA_CHECK_EQ(actual[0], 0x13572468u, "Array pass CB range offset 256 mismatch");
            UAA_CHECK_EQ(actual[1], 0x2468ACE0u, "Array pass StructuredBuffer range offset 256 mismatch");
            UAA_CHECK_EQ(actual[4], packRgba8(44, 132, 220, 255), "Texture2DArray mip 1 slice 2 mismatch");
        }
        else
        {
            ctx.note("NVRHI validation textureDimensionsCompatible() forbids Texture2D view of Texture2DArray; array slice test executed in raw pass");
        }
    }

    static void testStageIsolation(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        ShaderHandle vs = fix.createShader("vs_stage_iso", ShaderType::Vertex, "main");
        ShaderHandle ps = fix.createShader("ps_stage_iso", ShaderType::Pixel, "main");

        BindingLayoutHandle vsLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Vertex, 0u, true)
                .addItem(BindingLayoutItem::ConstantBuffer(0)));
        BindingLayoutHandle psLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Pixel, 1u, true)
                .addItem(BindingLayoutItem::ConstantBuffer(0)));
        UAA_CHECK(vsLayout != nullptr && psLayout != nullptr, "Stage-isolated BindingLayouts returned null");

        BufferHandle vsCb = createConstantBuffer(fix, 256, "bnd_iso_vs_cb");
        BufferHandle psCb = createConstantBuffer(fix, 256, "bnd_iso_ps_cb");
        UAA_CHECK(vsCb != nullptr && psCb != nullptr, "Stage-isolated CBs returned null");

        const std::array<float, 4> vsTint = { 0.2f, 0.0f, 0.0f, 0.5f };
        const std::array<float, 4> psTint = { 0.0f, 0.6f, 0.4f, 0.5f };
        fix.uploadBuffer(vsCb, vsTint.data(), sizeof(vsTint));
        fix.uploadBuffer(psCb, psTint.data(), sizeof(psTint));

        BindingSetHandle vsSet = dev->createBindingSet(
            BindingSetDesc().addItem(BindingSetItem::ConstantBuffer(0, vsCb)),
            vsLayout);
        BindingSetHandle psSet = dev->createBindingSet(
            BindingSetDesc().addItem(BindingSetItem::ConstantBuffer(0, psCb)),
            psLayout);
        UAA_CHECK(vsSet != nullptr && psSet != nullptr, "Stage-isolated BindingSets returned null");

        TextureDesc rtDesc;
        rtDesc.width = 8;
        rtDesc.height = 8;
        rtDesc.format = Format::RGBA8_UNORM;
        rtDesc.isRenderTarget = true;
        rtDesc.initialState = ResourceStates::RenderTarget;
        rtDesc.keepInitialState = true;
        rtDesc.debugName = "bnd_iso_rt";
        TextureHandle rt = dev->createTexture(rtDesc);
        UAA_CHECK(rt != nullptr, "Render target texture creation failed");

        FramebufferHandle fb = dev->createFramebuffer(
            FramebufferDesc().addColorAttachment(rt));
        UAA_CHECK(fb != nullptr, "createFramebuffer returned null");

        GraphicsPipelineDesc gpsoDesc;
        gpsoDesc.setVertexShader(vs)
            .setPixelShader(ps)
            .addBindingLayout(vsLayout)
            .addBindingLayout(psLayout);
        gpsoDesc.renderState.rasterState.cullMode = RasterCullMode::None;
        gpsoDesc.renderState.depthStencilState.depthTestEnable = false;
        gpsoDesc.renderState.depthStencilState.depthWriteEnable = false;

        GraphicsPipelineHandle pso = dev->createGraphicsPipeline(gpsoDesc, fb->getFramebufferInfo());
        UAA_CHECK(pso != nullptr, "createGraphicsPipeline returned null");

        CommandListHandle cmd = dev->createCommandList();
        UAA_CHECK(cmd != nullptr, "createCommandList returned null");

        GraphicsState state;
        state.pipeline = pso;
        state.framebuffer = fb;
        state.addBindingSet(vsSet);
        state.addBindingSet(psSet);
        state.viewport.addViewportAndScissorRect(Viewport(8.0f, 8.0f));

        cmd->open();
        cmd->clearTextureFloat(rt, AllSubresources, Color(0.0f, 0.0f, 0.0f, 0.0f));
        cmd->setGraphicsState(state);
        cmd->draw(DrawArguments().setVertexCount(3));
        cmd->close();
        fix.submitAndWait(cmd);

        std::vector<uint8_t> pixels;
        fix.readbackTexture2D(rt, 0, 0, pixels);
        UAA_CHECK_EQ(pixels.size(), size_t(8u * 8u * 4u), "Readback pixel buffer size mismatch");

        std::string diff;
        const uint8_t* center = pixels.data() + (4u * 8u + 4u) * 4u;
        UAA_CHECK(TestFixture::checkPixelRgba8(center, 51, 153, 102, 255, 2, diff),
            "Stage isolation pixel colour mismatch: " + diff);
    }

    static void testPushConstants(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        ShaderHandle cs = fix.createShader("cs_push_constants", ShaderType::Compute, "main");
        BindingLayoutHandle layout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::PushConstants(0, 16))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
        UAA_CHECK(layout != nullptr, "createBindingLayout returned null");

        BufferHandle outBuf = createStorageBuffer(fix, 32, "bnd_pc_out", 4, true);
        BufferHandle scratchBuf = createStorageBuffer(fix, 32, "bnd_pc_scratch", 4, true);
        UAA_CHECK(outBuf != nullptr && scratchBuf != nullptr, "Buffers returned null");

        BindingSetHandle set = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::PushConstants(0, 16))
                .addItem(BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
            layout);
        UAA_CHECK(set != nullptr, "createBindingSet returned null");

        ComputePipelineHandle pso = dev->createComputePipeline(
            ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
        UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

        struct PC
        {
            uint32_t index;
            uint32_t value;
            uint32_t xorMask;
            uint32_t addend;
        };

        std::array<PC, 8> pcs{};
        for (uint32_t i = 0; i < 8u; ++i)
        {
            pcs[i] = PC{ i, 0x1000u + i * 0x111u, 0x00FFu ^ (i * 0x11u), 10u + i * 7u };
        }

        ComputeState state;
        state.pipeline = pso;
        state.addBindingSet(set);

        CommandListHandle cmd1 = dev->createCommandList();
        CommandListHandle cmd2 = dev->createCommandList();
        UAA_CHECK(cmd1 != nullptr && cmd2 != nullptr, "createCommandList returned null");

        // Command list 1 dispatches indices 0..3 with encoder splits via copyBuffer and clearBufferUInt.
        cmd1->open();
        cmd1->clearBufferUInt(outBuf, 0u);
        cmd1->setComputeState(state);
        cmd1->setPushConstants(&pcs[0], sizeof(PC));
        cmd1->dispatch(1, 1, 1);

        cmd1->setPushConstants(&pcs[1], sizeof(PC));
        cmd1->dispatch(1, 1, 1);

        cmd1->copyBuffer(scratchBuf, 0, outBuf, 0, 16);
        cmd1->setComputeState(state);
        cmd1->setPushConstants(&pcs[2], sizeof(PC));
        cmd1->dispatch(1, 1, 1);

        cmd1->clearBufferUInt(scratchBuf, 0xCAFEBABEu);
        cmd1->setComputeState(state);
        cmd1->setPushConstants(&pcs[3], sizeof(PC));
        cmd1->dispatch(1, 1, 1);
        cmd1->close();

        // Command list 2 dispatches indices 4..7 across a second submission.
        cmd2->open();
        for (uint32_t i = 4; i < 8u; ++i)
        {
            cmd2->setComputeState(state);
            cmd2->setPushConstants(&pcs[i], sizeof(PC));
            cmd2->dispatch(1, 1, 1);
            if ((i & 1u) == 0u)
            {
                cmd2->copyBuffer(scratchBuf, 0, outBuf, 0, 16);
            }
            else
            {
                cmd2->clearBufferUInt(scratchBuf, i);
            }
        }
        cmd2->close();

        fix.submitAndWait(cmd1);
        fix.submitAndWait(cmd2);

        std::array<uint32_t, 8> actual = {};
        fix.readbackBuffer(outBuf, actual.data(), sizeof(actual));
        for (uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t expected = (pcs[i].value ^ pcs[i].xorMask) + pcs[i].addend;
            UAA_CHECK_EQ(actual[i], expected, "Push constants mismatch at index " + std::to_string(i));
        }
    }

    static void testBindingLifetime(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        IDevice* dev = fix.device();

        BufferHandle outBufA = createStorageBuffer(fix, 64, "bnd_life_outA", 4, true);
        BufferHandle outBufB = createStorageBuffer(fix, 64, "bnd_life_outB", 4, true);
        UAA_CHECK(outBufA != nullptr && outBufB != nullptr, "Output buffers returned null");

        const std::array<uint32_t, 4> inWordsA = { 10u, 20u, 30u, 40u };
        const std::array<uint32_t, 4> inWordsB = { 100u, 200u, 300u, 400u };

        CommandListHandle cmdA = dev->createCommandList();
        CommandListHandle cmdB = dev->createCommandList();
        UAA_CHECK(cmdA != nullptr && cmdB != nullptr, "createCommandList returned null");

        // Order 1: release buffers, binding set, layout, pipeline, shader before executeCommandList.
        {
            ShaderHandle cs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layout != nullptr, "Order 1 layout null");

            BufferHandle cb = createConstantBuffer(fix, 256, "bnd_life_cbA");
            BufferHandle inBuf = createStorageBuffer(fix, 16, "bnd_life_inA", 4);
            UAA_CHECK(cb != nullptr && inBuf != nullptr, "Order 1 buffers null");

            const std::array<uint32_t, 4> params = { 4u, 5u, 0u, 0u };
            fix.uploadBuffer(cb, params.data(), sizeof(params));
            fix.uploadBuffer(inBuf, inWordsA.data(), sizeof(inWordsA));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .setTrackLiveness(true)
                    .addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBufA)),
                layout);
            UAA_CHECK(set != nullptr, "Order 1 binding set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Order 1 pipeline null");

            ComputeState state;
            state.pipeline = pso;
            state.addBindingSet(set);

            cmdA->open();
            cmdA->setComputeState(state);
            cmdA->dispatch(1, 1, 1);
            cmdA->close();

            cb = nullptr;
            inBuf = nullptr;
            set = nullptr;
            layout = nullptr;
            pso = nullptr;
            cs = nullptr;
        }

        // Order 2: release pipeline and shader first, then layout, binding set, and input buffers.
        {
            ShaderHandle cs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");
            BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(layout != nullptr, "Order 2 layout null");

            BufferHandle cb = createConstantBuffer(fix, 256, "bnd_life_cbB");
            BufferHandle inBuf = createStorageBuffer(fix, 16, "bnd_life_inB", 4);
            UAA_CHECK(cb != nullptr && inBuf != nullptr, "Order 2 buffers null");

            const std::array<uint32_t, 4> params = { 4u, 11u, 1u, 0u };
            fix.uploadBuffer(cb, params.data(), sizeof(params));
            fix.uploadBuffer(inBuf, inWordsB.data(), sizeof(inWordsB));

            BindingSetHandle set = dev->createBindingSet(
                BindingSetDesc()
                    .setTrackLiveness(true)
                    .addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBufB)),
                layout);
            UAA_CHECK(set != nullptr, "Order 2 binding set null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Order 2 pipeline null");

            ComputeState state;
            state.pipeline = pso;
            state.addBindingSet(set);

            cmdB->open();
            cmdB->setComputeState(state);
            cmdB->dispatch(1, 1, 1);
            cmdB->close();

            pso = nullptr;
            cs = nullptr;
            layout = nullptr;
            set = nullptr;
            inBuf = nullptr;
            cb = nullptr;
        }

        fix.submitAndWait(cmdA);
        fix.submitAndWait(cmdB);

        std::array<uint32_t, 4> actualA = {};
        std::array<uint32_t, 4> actualB = {};
        fix.readbackBuffer(outBufA, actualA.data(), sizeof(actualA));
        fix.readbackBuffer(outBufB, actualB.data(), sizeof(actualB));

        for (uint32_t i = 0; i < 4u; ++i)
        {
            UAA_CHECK_EQ(actualA[i], inWordsA[i] + 5u, "Order 1 lifetime output mismatch");
            UAA_CHECK_EQ(actualB[i], (inWordsB[i] + 11u) ^ 1u, "Order 2 lifetime output mismatch");
        }
    }

    static void testInvalidBindings(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);

        if (!fix.isValidationActive())
        {
            ctx.skip("Invalid binding tests require NVRHI validation layer active");
        }

        IDevice* dev = fix.device();
        ShaderHandle cs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");

        BindingLayoutHandle validLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
        UAA_CHECK(validLayout != nullptr, "Valid layout returned null");

        BufferHandle cbBuf = createConstantBuffer(fix, 512, "bnd_inv_cb");
        BufferHandle srvBuf = createStorageBuffer(fix, 64, "bnd_inv_srv", 4);
        BufferHandle uavBuf = createStorageBuffer(fix, 64, "bnd_inv_uav", 4, true);
        BufferHandle nonUavBuf = createStorageBuffer(fix, 64, "bnd_inv_non_uav", 4, false);
        UAA_CHECK(cbBuf && srvBuf && uavBuf && nonUavBuf, "Buffers returned null");

        TextureDesc texDesc;
        texDesc.width = 4;
        texDesc.height = 4;
        texDesc.format = Format::RGBA8_UNORM;
        texDesc.isShaderResource = true;
        texDesc.initialState = ResourceStates::ShaderResource;
        texDesc.keepInitialState = true;
        texDesc.debugName = "bnd_inv_tex";
        TextureHandle tex = dev->createTexture(texDesc);
        UAA_CHECK(tex != nullptr, "Texture returned null");

        {
            ExpectErrorsGuard guard(fix);

            // Missing bindings in binding set.
            BindingSetHandle missingSet = dev->createBindingSet(
                BindingSetDesc().addItem(BindingSetItem::ConstantBuffer(0, cbBuf)),
                validLayout);
            UAA_CHECK(missingSet == nullptr, "BindingSet with missing slots must return nullptr");

            // Wrong resource flags / non-UAV buffer bound as UAV.
            BindingSetHandle wrongTypeSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, nonUavBuf))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, nonUavBuf)),
                validLayout);
            UAA_CHECK(wrongTypeSet == nullptr, "BindingSet with invalid resource flags must return nullptr");

            // Duplicate slots in layout and in binding set.
            BindingLayoutHandle dupLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::ConstantBuffer(0))
                    .addItem(BindingLayoutItem::ConstantBuffer(0)));
            UAA_CHECK(dupLayout == nullptr, "BindingLayout with duplicate slots must return nullptr");

            BindingSetHandle dupSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf))
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                validLayout);
            UAA_CHECK(dupSet == nullptr, "BindingSet with duplicate slots must return nullptr");

            // Incompatible pipeline layouts (mixed registerSpaceIsDescriptorSet).
            BindingLayoutHandle layoutTrue = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0u, true)
                    .addItem(BindingLayoutItem::ConstantBuffer(0)));
            BindingLayoutHandle layoutFalse = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, 0u, false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1)));
            UAA_CHECK(layoutTrue != nullptr && layoutFalse != nullptr, "Split layouts returned null");

            ComputePipelineHandle badPso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(cs)
                    .addBindingLayout(layoutTrue)
                    .addBindingLayout(layoutFalse));
            UAA_CHECK(badPso == nullptr, "Pipeline with incompatible registerSpaceIsDescriptorSet layouts must return nullptr");

            // Unaligned CB byteOffset.
            BindingSetHandle unalignedOffsetSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf, BufferRange(16, 256)))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                validLayout);
            UAA_CHECK(unalignedOffsetSet == nullptr, "BindingSet with unaligned CB offset must return nullptr");

            // Unaligned CB byteSize.
            BindingSetHandle unalignedSizeSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf, BufferRange(0, 128)))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                validLayout);
            UAA_CHECK(unalignedSizeSet == nullptr, "BindingSet with unaligned CB size must return nullptr");

            // Zero effective CB byteSize (offset == buffer byteSize with byteSize == 0).
            BindingSetHandle zeroCbRangeSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::ConstantBuffer(0, cbBuf, BufferRange(512, 0)))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, srvBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, uavBuf)),
                validLayout);
            UAA_CHECK(zeroCbRangeSet == nullptr, "BindingSet with zero effective CB size must return nullptr");

            // Texture dimension mismatch in binding set.
            BindingLayoutHandle texLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute)
                    .addItem(BindingLayoutItem::Texture_SRV(0)));
            UAA_CHECK(texLayout != nullptr, "Texture layout returned null");

            BindingSetHandle badDimSet = dev->createBindingSet(
                BindingSetDesc().addItem(
                    BindingSetItem::Texture_SRV(0, tex, Format::RGBA8_UNORM, AllSubresources, TextureDimension::Texture3D)),
                texLayout);
            UAA_CHECK(badDimSet == nullptr, "BindingSet with incompatible texture dimension must return nullptr");

            UAA_CHECK(guard.errorCount() > 0, "Expected validation errors during BND.Invalid");
        }
    }

    UAA_REGISTER_TEST("BND.Layout", "core", testBindingLayout);
    UAA_REGISTER_TEST("BND.BindingSet", "core", testBindingSet);
    UAA_REGISTER_TEST("BND.Ranges", "core", testBindingRanges);
    UAA_REGISTER_TEST("BND.StageIsolation", "core", testStageIsolation);
    UAA_REGISTER_TEST("BND.PushConstants", "core", testPushConstants);
    UAA_REGISTER_TEST("BND.BindingLifetime", "core", testBindingLifetime);
    UAA_REGISTER_TEST("BND.Invalid", "core", testInvalidBindings);
} // namespace nvrhi::uaa
