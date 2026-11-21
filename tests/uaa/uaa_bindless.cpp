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
#include <memory>
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

        void requireBindlessSupport(TestContext& ctx, const TestFixture& fix)
        {
            if (fix.api() == GraphicsAPI::D3D11)
            {
                ctx.skip("Descriptor tables and bindless layouts are not supported on D3D11");
            }
        }

        uint32_t packRgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            return static_cast<uint32_t>(r)
                | (static_cast<uint32_t>(g) << 8u)
                | (static_cast<uint32_t>(b) << 16u)
                | (static_cast<uint32_t>(a) << 24u);
        }

        uint32_t regularSpaceForBindless(const TestFixture& fix)
        {
            return (fix.api() == GraphicsAPI::D3D12) ? 2u : 0u;
        }

        BufferHandle createStructuredBuffer(
            const TestFixture& fix,
            uint64_t byteSize,
            const char* debugName,
            bool uav = false)
        {
            BufferDesc desc;
            desc.byteSize = byteSize;
            desc.structStride = 4;
            desc.canHaveUAVs = uav;
            desc.initialState = uav ? ResourceStates::UnorderedAccess : ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            desc.debugName = debugName;
            return fix.device()->createBuffer(desc);
        }

        BufferHandle createStructuredWord(const TestFixture& fix, uint32_t word, const char* debugName)
        {
            BufferHandle buf = createStructuredBuffer(fix, 16, debugName, false);
            if (!buf)
            {
                return nullptr;
            }
            const std::array<uint32_t, 4> words = { word, 0u, 0u, 0u };
            fix.uploadBuffer(buf, words.data(), sizeof(words));
            return buf;
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

        TextureHandle createSolidRgba8Texture(
            const TestFixture& fix,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            const char* debugName)
        {
            TextureDesc desc;
            desc.width = 2;
            desc.height = 2;
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

            const std::array<uint8_t, 16> pixels = {
                r, g, b, a,
                r, g, b, a,
                r, g, b, a,
                r, g, b, a
            };
            fix.uploadTexture2D(tex, 0, 0, pixels.data(), 8u);
            return tex;
        }

        BindingLayoutHandle makeTexBufBindlessLayout(
            const TestFixture& fix,
            uint32_t maxCapacity,
            bool updateAfterBind)
        {
            BindlessLayoutDesc desc;
            desc.setVisibility(ShaderType::Compute)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setMaxCapacity(maxCapacity)
                .setUpdateAfterBind(updateAfterBind)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0))
                .addRegisterSpace(BindingLayoutItem::StructuredBuffer_SRV(1));
            return fix.device()->createBindlessLayout(desc);
        }

        BindingLayoutHandle makeTexBufRegularLayout(const TestFixture& fix)
        {
            return fix.device()->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, regularSpaceForBindless(fix), false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(BindingLayoutItem::Sampler(3))
                    .addItem(BindingLayoutItem::ConstantBuffer(4)));
        }

        struct BindlessHarness
        {
            ShaderHandle cs;
            BindingLayoutHandle bindlessLayout;
            BindingLayoutHandle regularLayout;
            DescriptorTableHandle table;
            ComputePipelineHandle pso;
            SamplerHandle sampler;
            BufferHandle texIndices;
            BufferHandle bufIndices;
            BufferHandle outBuf;
            BufferHandle cbBuf;
            BindingSetHandle regularSet;
        };

        BindlessHarness createBindlessHarness(
            TestContext& ctx,
            const TestFixture& fix,
            uint32_t maxCapacity,
            uint32_t maxElements,
            bool updateAfterBind)
        {
            IDevice* dev = fix.device();
            BindlessHarness h;
            h.cs = fix.createShader("cs_bindless_tex_buf", ShaderType::Compute, "main");
            h.bindlessLayout = makeTexBufBindlessLayout(fix, maxCapacity, updateAfterBind);
            h.regularLayout = makeTexBufRegularLayout(fix);
            UAA_CHECK(h.bindlessLayout != nullptr, "createBindlessLayout returned null");
            UAA_CHECK(h.regularLayout != nullptr, "createBindingLayout returned null");

            h.table = dev->createDescriptorTable(h.bindlessLayout);
            UAA_CHECK(h.table != nullptr, "createDescriptorTable returned null");

            h.pso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(h.cs)
                    .addBindingLayout(h.bindlessLayout)
                    .addBindingLayout(h.regularLayout));
            UAA_CHECK(h.pso != nullptr, "createComputePipeline returned null");

            h.sampler = dev->createSampler(
                SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
            UAA_CHECK(h.sampler != nullptr, "createSampler returned null");

            const uint64_t idxBytes = std::max<uint64_t>(16u, uint64_t(maxElements) * sizeof(uint32_t));
            const uint64_t outBytes = std::max<uint64_t>(16u, uint64_t(maxElements) * 2u * sizeof(uint32_t));
            h.texIndices = createStructuredBuffer(fix, idxBytes, "tbl_tex_indices", false);
            h.bufIndices = createStructuredBuffer(fix, idxBytes, "tbl_buf_indices", false);
            h.outBuf = createStructuredBuffer(fix, outBytes, "tbl_out", true);
            h.cbBuf = createConstantBuffer(fix, 256, "tbl_params");
            UAA_CHECK(h.texIndices && h.bufIndices && h.outBuf && h.cbBuf, "Harness buffers returned null");

            h.regularSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, h.texIndices))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, h.bufIndices))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, h.outBuf))
                    .addItem(BindingSetItem::Sampler(3, h.sampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, h.cbBuf)),
                h.regularLayout);
            UAA_CHECK(h.regularSet != nullptr, "createBindingSet(regular) returned null");

            return h;
        }

        BindlessHarness createBindlessHarness(
            TestContext& ctx,
            const TestFixture& fix,
            uint32_t maxCapacity,
            uint32_t maxElements)
        {
            return createBindlessHarness(ctx, fix, maxCapacity, maxElements, fix.api() != GraphicsAPI::METAL);
        }

        void runBindlessDispatch(
            TestContext& ctx,
            const TestFixture& fix,
            const BindlessHarness& h,
            const std::vector<uint32_t>& texIdx,
            const std::vector<uint32_t>& bufIdx)
        {
            UAA_CHECK_EQ(texIdx.size(), bufIdx.size(), "Index array length mismatch");
            const uint32_t count = static_cast<uint32_t>(texIdx.size());

            fix.uploadBuffer(h.texIndices, texIdx.data(), texIdx.size() * sizeof(uint32_t));
            fix.uploadBuffer(h.bufIndices, bufIdx.data(), bufIdx.size() * sizeof(uint32_t));

            const std::array<uint32_t, 4> params = { count, 0u, 0u, 0u };
            fix.uploadBuffer(h.cbBuf, params.data(), sizeof(params));

            CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            ComputeState state;
            state.pipeline = h.pso;
            state.addBindingSet(h.table);
            state.addBindingSet(h.regularSet);

            cmd->open();
            cmd->setComputeState(state);
            cmd->dispatch((count + 7u) / 8u, 1u, 1u);
            cmd->close();
            fix.submitAndWait(cmd);
        }
    } // namespace

    static void testBindlessCreateWrite(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        // Tier 1 uses classic bindings, which limit this shader to 27 table buffers.
        const bool metalTier1 = fix.api() == GraphicsAPI::METAL && !fix.deviceInfo().metalArgumentBuffersTier2;
        const uint32_t capacity = metalTier1 ? 24u : 32u;
        const uint32_t midSlot = metalTier1 ? 11u : 15u;
        const uint32_t lastSlot = metalTier1 ? 23u : 31u;

        BindlessHarness h = createBindlessHarness(ctx, fix, capacity, 4u);
        UAA_CHECK_EQ(h.table->getCapacity(), capacity, "DescriptorTable getCapacity() mismatch");
        UAA_CHECK(h.table->getDesc() == nullptr, "DescriptorTable getDesc() must return nullptr");
        UAA_CHECK_EQ(h.table->getLayout(), h.bindlessLayout.Get(), "DescriptorTable getLayout() mismatch");

        TextureHandle texA = createSolidRgba8Texture(fix, 10, 20, 30, 255, "tbl_cw_texA");
        TextureHandle texB = createSolidRgba8Texture(fix, 40, 80, 120, 255, "tbl_cw_texB");
        TextureHandle texC = createSolidRgba8Texture(fix, 200, 150, 50, 255, "tbl_cw_texC");
        BufferHandle bufA = createStructuredWord(fix, 0x10101010u, "tbl_cw_bufA");
        BufferHandle bufB = createStructuredWord(fix, 0x20202020u, "tbl_cw_bufB");
        BufferHandle bufC = createStructuredWord(fix, 0x30303030u, "tbl_cw_bufC");
        UAA_CHECK(texA && texB && texC && bufA && bufB && bufC, "Resource creation failed");

        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, texA)), "Write tex slot 0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(midSlot, texB)), "Write tex midSlot failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(lastSlot, texC)), "Write tex lastSlot failed");

        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, bufA)), "Write buf slot 0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(midSlot, bufB)), "Write buf midSlot failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(lastSlot, bufC)), "Write buf lastSlot failed");

        runBindlessDispatch(ctx, fix, h, { 0u, midSlot, lastSlot }, { 0u, midSlot, lastSlot });

        std::array<uint32_t, 6> out = {};
        fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(10, 20, 30, 255), "Slot 0 texture mismatch");
        UAA_CHECK_EQ(out[1], 0x10101010u, "Slot 0 buffer mismatch");
        UAA_CHECK_EQ(out[2], packRgba8(40, 80, 120, 255), "Mid slot texture mismatch");
        UAA_CHECK_EQ(out[3], 0x20202020u, "Mid slot buffer mismatch");
        UAA_CHECK_EQ(out[4], packRgba8(200, 150, 50, 255), "Last slot texture mismatch");
        UAA_CHECK_EQ(out[5], 0x30303030u, "Last slot buffer mismatch");
    }

    static void testBindlessMixedSpaces(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        BindlessHarness h = createBindlessHarness(ctx, fix, 16u, 4u);

        TextureHandle tex5 = createSolidRgba8Texture(fix, 77, 88, 99, 255, "tbl_mix_tex5");
        TextureHandle tex2 = createSolidRgba8Texture(fix, 11, 22, 33, 255, "tbl_mix_tex2");
        BufferHandle buf5a = createStructuredWord(fix, 0xAAAA0001u, "tbl_mix_buf5a");
        BufferHandle buf5b = createStructuredWord(fix, 0xBBBB0005u, "tbl_mix_buf5b");
        BufferHandle buf2 = createStructuredWord(fix, 0xCCCC0002u, "tbl_mix_buf2");
        UAA_CHECK(tex5 && tex2 && buf5a && buf5b && buf2, "Resource creation failed");

        // Write both register spaces at identical slot 5, then overwrite only buffer space at slot 5.
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(5, tex5)), "Write tex slot 5 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(5, buf5a)), "Write buf5a slot 5 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(2, tex2)), "Write tex slot 2 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(2, buf2)), "Write buf2 slot 2 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(5, buf5b)), "Overwrite buf5b slot 5 failed");

        runBindlessDispatch(ctx, fix, h, { 5u, 2u }, { 5u, 2u });

        std::array<uint32_t, 4> out = {};
        fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(77, 88, 99, 255), "Slot 5 texture was overwritten by buffer write");
        UAA_CHECK_EQ(out[1], 0xBBBB0005u, "Slot 5 buffer mismatch");
        UAA_CHECK_EQ(out[2], packRgba8(11, 22, 33, 255), "Slot 2 texture mismatch");
        UAA_CHECK_EQ(out[3], 0xCCCC0002u, "Slot 2 buffer mismatch");
    }

    static void testBindlessNonUniform(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        BindlessHarness h = createBindlessHarness(ctx, fix, 16u, 8u);

        const std::array<uint32_t, 4> colors[4] = {
            { 10u, 20u, 30u, 255u },
            { 40u, 50u, 60u, 255u },
            { 70u, 80u, 90u, 255u },
            { 100u, 110u, 120u, 255u }
        };
        const std::array<uint32_t, 4> words = { 0x1111u, 0x2222u, 0x3333u, 0x4444u };

        std::array<TextureHandle, 4> textures;
        std::array<BufferHandle, 4> buffers;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            textures[i] = createSolidRgba8Texture(
                fix,
                static_cast<uint8_t>(colors[i][0]),
                static_cast<uint8_t>(colors[i][1]),
                static_cast<uint8_t>(colors[i][2]),
                static_cast<uint8_t>(colors[i][3]),
                "tbl_nu_tex");
            buffers[i] = createStructuredWord(fix, words[i], "tbl_nu_buf");
            UAA_CHECK(textures[i] != nullptr && buffers[i] != nullptr, "NonUniform resource creation failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(i, textures[i])), "Write tex failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(i, buffers[i])), "Write buf failed");
        }

        // Produce the non-uniform index arrays via GPU compute before feeding them to cs_bindless_tex_buf.
        ShaderHandle copyCs = fix.createShader("cs_buffer_copy_add", ShaderType::Compute, "main");
        BindingLayoutHandle copyLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2)));
        UAA_CHECK(copyLayout != nullptr, "Copy layout null");

        const std::array<uint32_t, 8> seedTexIdx = { 3u, 0u, 3u, 2u, 1u, 0u, 2u, 3u };
        const std::array<uint32_t, 8> seedBufIdx = { 2u, 1u, 0u, 3u, 3u, 2u, 1u, 0u };

        BufferHandle srcTexIdx = createStructuredBuffer(fix, 32, "tbl_nu_src_t", false);
        BufferHandle srcBufIdx = createStructuredBuffer(fix, 32, "tbl_nu_src_b", false);
        BufferHandle gpuTexIdx = createStructuredBuffer(fix, 32, "tbl_nu_gpu_t", true);
        BufferHandle gpuBufIdx = createStructuredBuffer(fix, 32, "tbl_nu_gpu_b", true);
        BufferHandle copyCb = createConstantBuffer(fix, 256, "tbl_nu_copy_cb");
        UAA_CHECK(srcTexIdx && srcBufIdx && gpuTexIdx && gpuBufIdx && copyCb, "Index generation buffers null");

        fix.uploadBuffer(srcTexIdx, seedTexIdx.data(), sizeof(seedTexIdx));
        fix.uploadBuffer(srcBufIdx, seedBufIdx.data(), sizeof(seedBufIdx));
        const std::array<uint32_t, 4> copyParams = { 8u, 0u, 0u, 0u };
        fix.uploadBuffer(copyCb, copyParams.data(), sizeof(copyParams));

        ComputePipelineHandle copyPso = dev->createComputePipeline(
            ComputePipelineDesc().setComputeShader(copyCs).addBindingLayout(copyLayout));
        UAA_CHECK(copyPso != nullptr, "Copy pipeline null");

        BindingSetHandle copySetT = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::ConstantBuffer(0, copyCb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, srcTexIdx))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, gpuTexIdx)),
            copyLayout);
        BindingSetHandle copySetB = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::ConstantBuffer(0, copyCb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, srcBufIdx))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, gpuBufIdx)),
            copyLayout);
        UAA_CHECK(copySetT != nullptr && copySetB != nullptr, "Copy binding sets null");

        BindingSetHandle nuRegularSet = dev->createBindingSet(
            BindingSetDesc()
                .addItem(BindingSetItem::StructuredBuffer_SRV(0, gpuTexIdx))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, gpuBufIdx))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, h.outBuf))
                .addItem(BindingSetItem::Sampler(3, h.sampler))
                .addItem(BindingSetItem::ConstantBuffer(4, h.cbBuf)),
            h.regularLayout);
        UAA_CHECK(nuRegularSet != nullptr, "NonUniform regular binding set null");

        const std::array<uint32_t, 4> blParams = { 8u, 0u, 0u, 0u };
        fix.uploadBuffer(h.cbBuf, blParams.data(), sizeof(blParams));

        CommandListHandle cmd = dev->createCommandList();
        UAA_CHECK(cmd != nullptr, "createCommandList returned null");
        cmd->open();
        {
            ComputeState st;
            st.pipeline = copyPso;
            st.addBindingSet(copySetT);
            cmd->setComputeState(st);
            cmd->dispatch(1, 1, 1);
        }
        {
            ComputeState st;
            st.pipeline = copyPso;
            st.addBindingSet(copySetB);
            cmd->setComputeState(st);
            cmd->dispatch(1, 1, 1);
        }
        {
            ComputeState st;
            st.pipeline = h.pso;
            st.addBindingSet(h.table);
            st.addBindingSet(nuRegularSet);
            cmd->setComputeState(st);
            cmd->dispatch(1, 1, 1);
        }
        cmd->close();
        fix.submitAndWait(cmd);

        std::array<uint32_t, 16> actual = {};
        fix.readbackBuffer(h.outBuf, actual.data(), sizeof(actual));
        for (uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t ti = seedTexIdx[i];
            const uint32_t bi = seedBufIdx[i];
            const uint32_t expectedTex = packRgba8(
                static_cast<uint8_t>(colors[ti][0]),
                static_cast<uint8_t>(colors[ti][1]),
                static_cast<uint8_t>(colors[ti][2]),
                static_cast<uint8_t>(colors[ti][3]));
            UAA_CHECK_EQ(actual[i * 2u], expectedTex, "NonUniform texture mismatch at lane " + std::to_string(i));
            UAA_CHECK_EQ(actual[i * 2u + 1u], words[bi], "NonUniform buffer mismatch at lane " + std::to_string(i));
        }
    }

    static void testBindlessResize(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        const uint32_t initialCapacity = (fix.api() == GraphicsAPI::METAL) ? 16u : 32u;
        BindlessHarness h = createBindlessHarness(ctx, fix, initialCapacity, 4u);
        UAA_CHECK_EQ(h.table->getCapacity(), initialCapacity, "Initial capacity mismatch");

        TextureHandle tex0 = createSolidRgba8Texture(fix, 15, 25, 35, 255, "tbl_rsz_tex0");
        TextureHandle tex3 = createSolidRgba8Texture(fix, 45, 55, 65, 255, "tbl_rsz_tex3");
        TextureHandle texNew = createSolidRgba8Texture(fix, 75, 85, 95, 255, "tbl_rsz_texNew");
        BufferHandle buf0 = createStructuredWord(fix, 0xA000u, "tbl_rsz_buf0");
        BufferHandle buf3 = createStructuredWord(fix, 0xA003u, "tbl_rsz_buf3");
        BufferHandle bufNew = createStructuredWord(fix, 0xA999u, "tbl_rsz_bufNew");
        UAA_CHECK(tex0 && tex3 && texNew && buf0 && buf3 && bufNew, "Resize resources null");

        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(3, tex3)), "Write tex3 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(3, buf3)), "Write buf3 failed");

        if (fix.api() == GraphicsAPI::D3D12)
        {
            // Grow past initial capacity with keepContents = true.
            dev->resizeDescriptorTable(h.table, 64u, true);
            UAA_CHECK_EQ(h.table->getCapacity(), 64u, "D3D12 grow capacity mismatch");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(48, texNew)), "Write slot 48 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(48, bufNew)), "Write buf 48 failed");

            runBindlessDispatch(ctx, fix, h, { 0u, 3u, 48u }, { 0u, 3u, 48u });
            std::array<uint32_t, 6> grownOut = {};
            fix.readbackBuffer(h.outBuf, grownOut.data(), sizeof(grownOut));
            UAA_CHECK_EQ(grownOut[0], packRgba8(15, 25, 35, 255), "Preserved slot 0 tex mismatch after grow");
            UAA_CHECK_EQ(grownOut[1], 0xA000u, "Preserved slot 0 buf mismatch after grow");
            UAA_CHECK_EQ(grownOut[4], packRgba8(75, 85, 95, 255), "Grown slot 48 tex mismatch");
            UAA_CHECK_EQ(grownOut[5], 0xA999u, "Grown slot 48 buf mismatch");

            // Shrink to 8 with keepContents = true and verify slot 12 is rejected.
            dev->resizeDescriptorTable(h.table, 8u, true);
            UAA_CHECK_EQ(h.table->getCapacity(), 8u, "D3D12 shrink capacity mismatch");
            UAA_CHECK(!dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(12, texNew)),
                "Write at slot 12 after shrink to 8 must fail");

            runBindlessDispatch(ctx, fix, h, { 0u, 3u }, { 0u, 3u });
            std::array<uint32_t, 4> shrunkOut = {};
            fix.readbackBuffer(h.outBuf, shrunkOut.data(), sizeof(shrunkOut));
            UAA_CHECK_EQ(shrunkOut[0], packRgba8(15, 25, 35, 255), "Preserved slot 0 tex mismatch after shrink");
            UAA_CHECK_EQ(shrunkOut[2], packRgba8(45, 55, 65, 255), "Preserved slot 3 tex mismatch after shrink");

            // Resize with keepContents = false and repopulate slot 0.
            dev->resizeDescriptorTable(h.table, 8u, false);
            UAA_CHECK_EQ(h.table->getCapacity(), 8u, "D3D12 resize(8, false) capacity mismatch");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, texNew)), "Repopulate tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, bufNew)), "Repopulate buf0 failed");

            runBindlessDispatch(ctx, fix, h, { 0u }, { 0u });
            std::array<uint32_t, 2> resetOut = {};
            fix.readbackBuffer(h.outBuf, resetOut.data(), sizeof(resetOut));
            UAA_CHECK_EQ(resetOut[0], packRgba8(75, 85, 95, 255), "Repopulated slot 0 tex mismatch");
            UAA_CHECK_EQ(resetOut[1], 0xA999u, "Repopulated slot 0 buf mismatch");
            return;
        }

        // On Metal, growing past capacity is ignored; on Vulkan, growing past maxCapacity hits an assert.
        if (fix.api() == GraphicsAPI::METAL)
        {
            dev->resizeDescriptorTable(h.table, 64u, true);
            UAA_CHECK_EQ(h.table->getCapacity(), initialCapacity, "Metal grow past maxCapacity must be ignored");
        }

        // Exercise shrink(8, true) and shrink(8, false) safely without exceeding slots vector bounds.
        dev->resizeDescriptorTable(h.table, 8u, true);
        runBindlessDispatch(ctx, fix, h, { 0u, 3u }, { 0u, 3u });
        std::array<uint32_t, 4> preserved = {};
        fix.readbackBuffer(h.outBuf, preserved.data(), sizeof(preserved));
        UAA_CHECK_EQ(preserved[0], packRgba8(15, 25, 35, 255), "Preserved slot 0 tex mismatch after resize(8, true)");
        UAA_CHECK_EQ(preserved[1], 0xA000u, "Preserved slot 0 buf mismatch after resize(8, true)");
        UAA_CHECK_EQ(preserved[2], packRgba8(45, 55, 65, 255), "Preserved slot 3 tex mismatch after resize(8, true)");
        UAA_CHECK_EQ(preserved[3], 0xA003u, "Preserved slot 3 buf mismatch after resize(8, true)");

        dev->resizeDescriptorTable(h.table, 8u, false);
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, texNew)), "Repopulate tex0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, bufNew)), "Repopulate buf0 failed");
        runBindlessDispatch(ctx, fix, h, { 0u }, { 0u });
        std::array<uint32_t, 2> repopulated = {};
        fix.readbackBuffer(h.outBuf, repopulated.data(), sizeof(repopulated));
        UAA_CHECK_EQ(repopulated[0], packRgba8(75, 85, 95, 255), "Repopulated slot 0 tex mismatch");
        UAA_CHECK_EQ(repopulated[1], 0xA999u, "Repopulated slot 0 buf mismatch");

        if (h.table->getCapacity() != 8u)
        {
            ctx.skip("Backend does not support dynamic descriptor table capacity resizing (getCapacity() remained "
                + std::to_string(h.table->getCapacity()) + " after resizeDescriptorTable(8))");
        }
    }

    static void testBindlessClear(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        BindlessHarness h = createBindlessHarness(ctx, fix, 16u, 4u);

        std::array<TextureHandle, 4> texs;
        std::array<BufferHandle, 4> bufs;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            texs[i] = createSolidRgba8Texture(fix,
                static_cast<uint8_t>(20u + i * 30u),
                static_cast<uint8_t>(30u + i * 20u),
                static_cast<uint8_t>(40u + i * 10u),
                255u,
                "tbl_clr_tex");
            bufs[i] = createStructuredWord(fix, 0x5000u + i, "tbl_clr_buf");
            UAA_CHECK(texs[i] != nullptr && bufs[i] != nullptr, "Clear test resource creation failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(i, texs[i])), "Write tex failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(i, bufs[i])), "Write buf failed");
        }

        const bool cleared = dev->writeDescriptorTable(h.table, BindingSetItem::None(1));
        if (fix.api() == GraphicsAPI::VULKAN)
        {
            UAA_CHECK(cleared, "Vulkan writeDescriptorTable(None) must return true");
            ctx.note("Vulkan writeDescriptorTable(None) returns true without writing a null descriptor, relying on ePartiallyBound");
        }
        else if (fix.api() == GraphicsAPI::METAL)
        {
            UAA_CHECK(cleared, "Metal writeDescriptorTable(None) must return true");
            ctx.note("Metal writeDescriptorTable(None) returns true without writing a null descriptor, so the slot keeps its descriptor");
        }
        else
        {
            UAA_CHECK(cleared, "D3D12 writeDescriptorTable(None) must return true");
            ctx.note("D3D12 writeDescriptorTable(None) writes a null SRV descriptor");
        }

        // Dispatch using only populated slots 0, 2, 3 without dereferencing cleared slot 1.
        runBindlessDispatch(ctx, fix, h, { 0u, 2u, 3u }, { 0u, 2u, 3u });

        std::array<uint32_t, 6> out = {};
        fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(20, 30, 40, 255), "Slot 0 tex mismatch after clearing slot 1");
        UAA_CHECK_EQ(out[1], 0x5000u, "Slot 0 buf mismatch after clearing slot 1");
        UAA_CHECK_EQ(out[2], packRgba8(80, 70, 60, 255), "Slot 2 tex mismatch after clearing slot 1");
        UAA_CHECK_EQ(out[3], 0x5002u, "Slot 2 buf mismatch after clearing slot 1");
        UAA_CHECK_EQ(out[4], packRgba8(110, 90, 70, 255), "Slot 3 tex mismatch after clearing slot 1");
        UAA_CHECK_EQ(out[5], 0x5003u, "Slot 3 buf mismatch after clearing slot 1");

        // Repopulate slot 1 and verify it becomes usable again.
        TextureHandle tex1New = createSolidRgba8Texture(fix, 210, 220, 230, 255, "tbl_clr_tex1_new");
        BufferHandle buf1New = createStructuredWord(fix, 0x9991u, "tbl_clr_buf1_new");
        UAA_CHECK(tex1New != nullptr && buf1New != nullptr, "Repopulation resources null");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(1, tex1New)), "Repopulate tex1 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(1, buf1New)), "Repopulate buf1 failed");

        runBindlessDispatch(ctx, fix, h, { 0u, 1u, 2u, 3u }, { 0u, 1u, 2u, 3u });
        std::array<uint32_t, 8> outAfter = {};
        fix.readbackBuffer(h.outBuf, outAfter.data(), sizeof(outAfter));
        UAA_CHECK_EQ(outAfter[2], packRgba8(210, 220, 230, 255), "Repopulated slot 1 tex mismatch");
        UAA_CHECK_EQ(outAfter[3], 0x9991u, "Repopulated slot 1 buf mismatch");
    }

    static void testBindlessUpdateAfterBind(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        // Non-updateAfterBind path: populate before submission and verify execution.
        {
            BindlessHarness hNoUab = createBindlessHarness(ctx, fix, 16u, 2u, false);
            UAA_CHECK(!hNoUab.bindlessLayout->getBindlessDesc()->updateAfterBind,
                "BindlessLayoutDesc::updateAfterBind=false must be preserved");

            TextureHandle tex0 = createSolidRgba8Texture(fix, 12, 34, 56, 255, "tbl_nouab_tex0");
            BufferHandle buf0 = createStructuredWord(fix, 0x7001u, "tbl_nouab_buf0");
            UAA_CHECK(tex0 && buf0, "Non-UAB resources null");
            UAA_CHECK(dev->writeDescriptorTable(hNoUab.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(hNoUab.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");

            runBindlessDispatch(ctx, fix, hNoUab, { 0u }, { 0u });
            std::array<uint32_t, 2> out = {};
            fix.readbackBuffer(hNoUab.outBuf, out.data(), sizeof(out));
            UAA_CHECK_EQ(out[0], packRgba8(12, 34, 56, 255), "Non-UAB tex0 mismatch");
            UAA_CHECK_EQ(out[1], 0x7001u, "Non-UAB buf0 mismatch");
        }

        if (fix.api() == GraphicsAPI::METAL)
            UAA_SKIP("Metal does not support update-after-bind; the ordinary table case passed");

        // updateAfterBind=true path with in-flight unused-slot update behind IGpuGate.
        {
            BindlessHarness hUab = createBindlessHarness(ctx, fix, 16u, 2u, true);
            UAA_CHECK(hUab.bindlessLayout->getBindlessDesc()->updateAfterBind,
                "BindlessLayoutDesc::updateAfterBind=true must be preserved");

            TextureHandle tex0 = createSolidRgba8Texture(fix, 90, 100, 110, 255, "tbl_uab_tex0");
            BufferHandle buf0 = createStructuredWord(fix, 0x8000u, "tbl_uab_buf0");
            TextureHandle tex4 = createSolidRgba8Texture(fix, 150, 160, 170, 255, "tbl_uab_tex4");
            BufferHandle buf4 = createStructuredWord(fix, 0x8004u, "tbl_uab_buf4");
            UAA_CHECK(tex0 && buf0 && tex4 && buf4, "UAB resources null");

            UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");

            // Prepare distinct regular binding sets for cmd1 (slot 0) and cmd2 (slot 4).
            BufferHandle texIdx1 = createStructuredWord(fix, 0u, "tbl_uab_ti1");
            BufferHandle bufIdx1 = createStructuredWord(fix, 0u, "tbl_uab_bi1");
            BufferHandle out1 = createStructuredBuffer(fix, 16u, "tbl_uab_out1", true);
            BufferHandle cb1 = createConstantBuffer(fix, 256u, "tbl_uab_cb1");

            BufferHandle texIdx2 = createStructuredWord(fix, 4u, "tbl_uab_ti2");
            BufferHandle bufIdx2 = createStructuredWord(fix, 4u, "tbl_uab_bi2");
            BufferHandle out2 = createStructuredBuffer(fix, 16u, "tbl_uab_out2", true);
            BufferHandle cb2 = createConstantBuffer(fix, 256u, "tbl_uab_cb2");
            UAA_CHECK(texIdx1 && bufIdx1 && out1 && cb1 && texIdx2 && bufIdx2 && out2 && cb2, "UAB IO buffers null");

            const std::array<uint32_t, 4> params = { 1u, 0u, 0u, 0u };
            fix.uploadBuffer(cb1, params.data(), sizeof(params));
            fix.uploadBuffer(cb2, params.data(), sizeof(params));

            BindingSetHandle set1 = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, texIdx1))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, bufIdx1))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, out1))
                    .addItem(BindingSetItem::Sampler(3, hUab.sampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, cb1)),
                hUab.regularLayout);
            BindingSetHandle set2 = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, texIdx2))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, bufIdx2))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, out2))
                    .addItem(BindingSetItem::Sampler(3, hUab.sampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, cb2)),
                hUab.regularLayout);
            UAA_CHECK(set1 != nullptr && set2 != nullptr, "UAB regular sets null");

            CommandListHandle cmd1 = dev->createCommandList();
            UAA_CHECK(cmd1 != nullptr, "cmd1 null");
            cmd1->open();
            {
                ComputeState st;
                st.pipeline = hUab.pso;
                st.addBindingSet(hUab.table);
                st.addBindingSet(set1);
                cmd1->setComputeState(st);
                cmd1->dispatch(1, 1, 1);
            }
            cmd1->close();

            std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
            if (gate)
            {
                UAA_CHECK(gate->enqueueWait(CommandQueue::Graphics), "IGpuGate::enqueueWait failed");
                dev->executeCommandList(cmd1);

                // Update unused slot 4 while cmd1 is pending behind the gate.
                UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::Texture_SRV(4, tex4)), "Pending write tex4 failed");
                UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::StructuredBuffer_SRV(4, buf4)), "Pending write buf4 failed");

                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "cmd2 null");
                cmd2->open();
                {
                    ComputeState st;
                    st.pipeline = hUab.pso;
                    st.addBindingSet(hUab.table);
                    st.addBindingSet(set2);
                    cmd2->setComputeState(st);
                    cmd2->dispatch(1, 1, 1);
                }
                cmd2->close();
                dev->executeCommandList(cmd2);

                gate->signal();
                UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
            }
            else
            {
                ctx.note("IGpuGate unavailable; testing sequential updateAfterBind=true slot updates");
                fix.submitAndWait(cmd1);

                UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::Texture_SRV(4, tex4)), "Write tex4 failed");
                UAA_CHECK(dev->writeDescriptorTable(hUab.table, BindingSetItem::StructuredBuffer_SRV(4, buf4)), "Write buf4 failed");

                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "cmd2 null");
                cmd2->open();
                {
                    ComputeState st;
                    st.pipeline = hUab.pso;
                    st.addBindingSet(hUab.table);
                    st.addBindingSet(set2);
                    cmd2->setComputeState(st);
                    cmd2->dispatch(1, 1, 1);
                }
                cmd2->close();
                fix.submitAndWait(cmd2);
            }

            std::array<uint32_t, 2> res1 = {};
            std::array<uint32_t, 2> res2 = {};
            fix.readbackBuffer(out1, res1.data(), sizeof(res1));
            fix.readbackBuffer(out2, res2.data(), sizeof(res2));
            UAA_CHECK_EQ(res1[0], packRgba8(90, 100, 110, 255), "cmd1 slot 0 tex mismatch");
            UAA_CHECK_EQ(res1[1], 0x8000u, "cmd1 slot 0 buf mismatch");
            UAA_CHECK_EQ(res2[0], packRgba8(150, 160, 170, 255), "cmd2 slot 4 tex mismatch");
            UAA_CHECK_EQ(res2[1], 0x8004u, "cmd2 slot 4 buf mismatch");
        }
    }

    static void testBindlessPendingLifetime(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
        if (!gate)
        {
            ctx.skip("IGpuGate is not supported by this backend");
        }

        BufferHandle outBuf = createStructuredBuffer(fix, 16u, "tbl_pl_out", true);
        UAA_CHECK(outBuf != nullptr, "outBuf null");

        DescriptorTableHandle table;
        TextureHandle tex0;
        BufferHandle buf0;

        {
            ShaderHandle cs = fix.createShader("cs_bindless_tex_buf", ShaderType::Compute, "main");
            BindingLayoutHandle bindlessLayout = makeTexBufBindlessLayout(fix, 16u, fix.api() != GraphicsAPI::METAL);
            BindingLayoutHandle regularLayout = makeTexBufRegularLayout(fix);
            UAA_CHECK(bindlessLayout && regularLayout, "Layouts null");

            table = dev->createDescriptorTable(bindlessLayout);
            UAA_CHECK(table != nullptr, "DescriptorTable null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc()
                    .setComputeShader(cs)
                    .addBindingLayout(bindlessLayout)
                    .addBindingLayout(regularLayout));
            UAA_CHECK(pso != nullptr, "Pipeline null");

            tex0 = createSolidRgba8Texture(fix, 25, 75, 125, 255, "tbl_pl_tex0");
            buf0 = createStructuredWord(fix, 0xCAFE0001u, "tbl_pl_buf0");
            TextureHandle tex4 = createSolidRgba8Texture(fix, 200, 210, 220, 255, "tbl_pl_tex4");
            BufferHandle buf4 = createStructuredWord(fix, 0xCAFE0004u, "tbl_pl_buf4");
            UAA_CHECK(tex0 && buf0 && tex4 && buf4, "Slot resources null");

            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");

            SamplerHandle sampler = dev->createSampler(
                SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
            BufferHandle texIndices = createStructuredWord(fix, 0u, "tbl_pl_ti");
            BufferHandle bufIndices = createStructuredWord(fix, 0u, "tbl_pl_bi");
            BufferHandle cbBuf = createConstantBuffer(fix, 256u, "tbl_pl_cb");
            UAA_CHECK(sampler && texIndices && bufIndices && cbBuf, "Regular resources null");

            const std::array<uint32_t, 4> params = { 1u, 0u, 0u, 0u };
            fix.uploadBuffer(cbBuf, params.data(), sizeof(params));

            BindingSetHandle regularSet = dev->createBindingSet(
                BindingSetDesc()
                    .setTrackLiveness(true)
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, texIndices))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, bufIndices))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                    .addItem(BindingSetItem::Sampler(3, sampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, cbBuf)),
                regularLayout);
            UAA_CHECK(regularSet != nullptr, "regularSet null");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "cmd null");
            cmd->open();
            {
                ComputeState st;
                st.pipeline = pso;
                st.addBindingSet(table);
                st.addBindingSet(regularSet);
                cmd->setComputeState(st);
                cmd->dispatch(1, 1, 1);
            }
            cmd->close();

            UAA_CHECK(gate->enqueueWait(CommandQueue::Graphics), "IGpuGate::enqueueWait failed");
            dev->executeCommandList(cmd);

            // A pending update requires update-after-bind support.
            if (fix.api() != GraphicsAPI::METAL)
            {
                UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(4, tex4)), "Update unused tex4 failed");
                UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::StructuredBuffer_SRV(4, buf4)), "Update unused buf4 failed");
            }

            // Drop tracked caller handles (and unused slot 4 handles) before releasing the gate.
            // Metal keeps the referenced resources alive through completion. Vulkan/D3D12
            // descriptor tables have no liveness tracking.
            if (fix.api() == GraphicsAPI::METAL)
            {
                table = nullptr;
                tex0 = nullptr;
                buf0 = nullptr;
            }
            else
            {
                ctx.note("IDescriptorTable disables liveness tracking; keeping table and slot 0 resources alive until waitForIdle while dropping tracked pipeline/binding set handles");
            }
        }

        gate->signal();
        UAA_CHECK(dev->waitForIdle(), "waitForIdle failed");
        dev->runGarbageCollection();

        table = nullptr;
        tex0 = nullptr;
        buf0 = nullptr;

        std::array<uint32_t, 2> out = {};
        fix.readbackBuffer(outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(25, 75, 125, 255), "Pending dispatch slot 0 texture mismatch");
        UAA_CHECK_EQ(out[1], 0xCAFE0001u, "Pending dispatch slot 0 buffer mismatch");
    }

    static void runBindlessMultipleLayouts(TestContext& ctx, bool explicitSets)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        if (fix.api() == GraphicsAPI::D3D12)
            ctx.skip("This shader uses Vulkan descriptor set indices");
        IDevice* dev = fix.device();

        ShaderHandle cs = fix.createShader(explicitSets ? "cs_bindless_two_sets_explicit" : "cs_bindless_two_sets", ShaderType::Compute);
        BindlessLayoutDesc firstDesc;
        firstDesc.setVisibility(ShaderType::Compute).setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
            .setMaxCapacity(4).setUpdateAfterBind(false).addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
        BindlessLayoutDesc secondDesc = firstDesc;
        secondDesc.setMaxCapacity(8);
        BindingLayoutHandle firstLayout = dev->createBindlessLayout(firstDesc);
        BindingLayoutHandle secondLayout = dev->createBindlessLayout(secondDesc);
        firstDesc.setMaxCapacity(8);
        secondDesc.setMaxCapacity(16);
        BindingLayoutHandle firstLargeLayout = dev->createBindlessLayout(firstDesc);
        BindingLayoutHandle secondLargeLayout = dev->createBindlessLayout(secondDesc);
        BindingLayoutHandle regularLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute, 0, explicitSets)
                .addItem(BindingLayoutItem::Sampler(0))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(1)));
        UAA_CHECK(cs && firstLayout && secondLayout && firstLargeLayout && secondLargeLayout && regularLayout, "Shaders or layouts are null");

        DescriptorTableHandle first = dev->createDescriptorTable(firstLayout);
        DescriptorTableHandle second = dev->createDescriptorTable(secondLayout);
        DescriptorTableHandle replacement = dev->createDescriptorTable(secondLayout);
        DescriptorTableHandle firstLarge = dev->createDescriptorTable(firstLargeLayout);
        DescriptorTableHandle secondLarge = dev->createDescriptorTable(secondLargeLayout);
        ComputePipelineHandle pso = dev->createComputePipeline(ComputePipelineDesc().setComputeShader(cs)
            .addBindingLayout(firstLayout).addBindingLayout(regularLayout).addBindingLayout(secondLayout));
        ComputePipelineHandle largePso = dev->createComputePipeline(ComputePipelineDesc().setComputeShader(cs)
            .addBindingLayout(firstLargeLayout).addBindingLayout(regularLayout).addBindingLayout(secondLargeLayout));
        UAA_CHECK(first && second && replacement && firstLarge && secondLarge && pso && largePso, "Tables or pipelines are null");

        TextureHandle red = createSolidRgba8Texture(fix, 255, 0, 0, 255, "tbl_two_first");
        TextureHandle green = createSolidRgba8Texture(fix, 0, 255, 0, 255, "tbl_two_second");
        TextureHandle blue = createSolidRgba8Texture(fix, 0, 0, 255, 255, "tbl_two_replacement");
        SamplerHandle sampler = dev->createSampler(SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
        BufferHandle outFirst = createStructuredBuffer(fix, 16, "tbl_two_out_first", true);
        BufferHandle outSecond = createStructuredBuffer(fix, 16, "tbl_two_out_second", true);
        BufferHandle outLarge = createStructuredBuffer(fix, 16, "tbl_two_out_large", true);
        UAA_CHECK(red && green && blue && sampler && outFirst && outSecond && outLarge, "Two-table resources are null");
        UAA_CHECK(dev->writeDescriptorTable(first, BindingSetItem::Texture_SRV(1, red)), "First table write failed");
        UAA_CHECK(dev->writeDescriptorTable(second, BindingSetItem::Texture_SRV(6, green)), "Second table write failed");
        UAA_CHECK(dev->writeDescriptorTable(replacement, BindingSetItem::Texture_SRV(6, blue)), "Replacement table write failed");
        UAA_CHECK(dev->writeDescriptorTable(firstLarge, BindingSetItem::Texture_SRV(1, green)), "Large first table write failed");
        UAA_CHECK(dev->writeDescriptorTable(secondLarge, BindingSetItem::Texture_SRV(6, red)), "Large second table write failed");

        BindingSetHandle regularFirst = dev->createBindingSet(BindingSetDesc()
            .addItem(BindingSetItem::Sampler(0, sampler)).addItem(BindingSetItem::StructuredBuffer_UAV(1, outFirst)), regularLayout);
        BindingSetHandle regularSecond = dev->createBindingSet(BindingSetDesc()
            .addItem(BindingSetItem::Sampler(0, sampler)).addItem(BindingSetItem::StructuredBuffer_UAV(1, outSecond)), regularLayout);
        BindingSetHandle regularLarge = dev->createBindingSet(BindingSetDesc()
            .addItem(BindingSetItem::Sampler(0, sampler)).addItem(BindingSetItem::StructuredBuffer_UAV(1, outLarge)), regularLayout);
        UAA_CHECK(regularFirst && regularSecond && regularLarge, "Regular binding sets are null");

        CommandListHandle cmd = dev->createCommandList();
        cmd->open();
        cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(first).addBindingSet(regularFirst).addBindingSet(second));
        cmd->dispatch(1);
        cmd->setComputeState(ComputeState().setPipeline(largePso).addBindingSet(firstLarge).addBindingSet(regularLarge).addBindingSet(secondLarge));
        cmd->dispatch(1);
        cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(first).addBindingSet(regularSecond).addBindingSet(replacement));
        cmd->dispatch(1);
        cmd->close();
        fix.submitAndWait(cmd);

        std::array<uint32_t, 2> out = {};
        fix.readbackBuffer(outFirst, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(255, 0, 0, 255), "First table result mismatch");
        UAA_CHECK_EQ(out[1], packRgba8(0, 255, 0, 255), "Second table result mismatch");
        fix.readbackBuffer(outSecond, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(255, 0, 0, 255), "First table changed after rebind");
        UAA_CHECK_EQ(out[1], packRgba8(0, 0, 255, 255), "Replacement table result mismatch");
        fix.readbackBuffer(outLarge, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(0, 255, 0, 255), "Large first table result mismatch");
        UAA_CHECK_EQ(out[1], packRgba8(255, 0, 0, 255), "Large second table result mismatch");
    }

    static void testBindlessMultipleLayouts(TestContext& ctx)
    {
        runBindlessMultipleLayouts(ctx, false);
    }

    static void testBindlessMultipleLayoutsExplicit(TestContext& ctx)
    {
        runBindlessMultipleLayouts(ctx, true);
    }

    static void testBindlessTypedUav(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        if (fix.api() == GraphicsAPI::D3D12)
            ctx.skip("D3D12 descriptor tables require explicit allocation");
        IDevice* dev = fix.device();

        ShaderHandle cs = fix.createShader("cs_bindless_typed_uav", ShaderType::Compute);
        BindlessLayoutDesc desc;
        desc.setVisibility(ShaderType::Compute).setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
            .setMaxCapacity(4).setUpdateAfterBind(false).addRegisterSpace(BindingLayoutItem::TypedBuffer_UAV(0));
        BindingLayoutHandle layout = dev->createBindlessLayout(desc);
        UAA_CHECK(cs && layout, "Typed buffer shader or layout is null");
        DescriptorTableHandle table = dev->createDescriptorTable(layout);
        ComputePipelineHandle pso = dev->createComputePipeline(ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));

        BufferDesc bufferDesc;
        bufferDesc.setByteSize(16).setCanHaveTypedViews(true).setCanHaveUAVs(true).setFormat(Format::R32_UINT)
            .setInitialState(ResourceStates::UnorderedAccess).setKeepInitialState(true);
        BufferHandle buffer = dev->createBuffer(bufferDesc);
        UAA_CHECK(table && pso && buffer, "Typed buffer resources are null");
        UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::TypedBuffer_UAV(1, buffer, Format::R32_UINT)), "Typed buffer write failed");

        CommandListHandle cmd = dev->createCommandList();
        cmd->open();
        cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(table));
        cmd->dispatch(1);
        cmd->close();
        fix.submitAndWait(cmd);
        uint32_t out = 0;
        fix.readbackBuffer(buffer, &out, sizeof(out));
        UAA_CHECK_EQ(out, 0x12345678u, "Typed buffer UAV result mismatch");
    }

    static void testBindlessSamplers(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        if (fix.api() == GraphicsAPI::D3D12)
            ctx.skip("D3D12 immutable sampler tables use a different descriptor heap");
        if (fix.isValidationActive())
            ctx.skip("NVRHI validation does not support bindless samplers");
        IDevice* dev = fix.device();

        ShaderHandle cs = fix.createShader("cs_bindless_sampler", ShaderType::Compute);
        BindlessLayoutDesc tableDesc;
        tableDesc.setVisibility(ShaderType::Compute)
            .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
            .setMaxCapacity(4)
            .setUpdateAfterBind(false)
            .addRegisterSpace(BindingLayoutItem::Sampler(0));
        BindingLayoutHandle tableLayout = dev->createBindlessLayout(tableDesc);
        BindingLayoutHandle regularLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute, regularSpaceForBindless(fix), false)
                .addItem(BindingLayoutItem::Texture_SRV(0))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(1)));
        UAA_CHECK(cs && tableLayout && regularLayout, "Sampler table layouts or shader are null");

        DescriptorTableHandle table = dev->createDescriptorTable(tableLayout);
        ComputePipelineHandle pso = dev->createComputePipeline(ComputePipelineDesc()
            .setComputeShader(cs).addBindingLayout(tableLayout).addBindingLayout(regularLayout));
        SamplerHandle clamp = dev->createSampler(SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
        SamplerHandle wrap = dev->createSampler(SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Wrap));
        UAA_CHECK(table && pso && clamp && wrap, "Sampler table resources are null");
        UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Sampler(0, clamp)), "Write clamp sampler failed");
        UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Sampler(1, wrap)), "Write wrap sampler failed");

        TextureDesc textureDesc;
        textureDesc.setWidth(2).setHeight(1).setFormat(Format::RGBA8_UNORM)
            .setInitialState(ResourceStates::ShaderResource).setKeepInitialState(true);
        textureDesc.isShaderResource = true;
        TextureHandle texture = dev->createTexture(textureDesc);
        BufferHandle output = createStructuredBuffer(fix, 32, "tbl_samplers_out", true);
        UAA_CHECK(texture && output, "Sampler test texture or output is null");
        const std::array<uint8_t, 8> pixels = { 255, 0, 0, 255, 0, 255, 0, 255 };
        fix.uploadTexture2D(texture, 0, 0, pixels.data(), 8u);
        BindingSetHandle regularSet = dev->createBindingSet(BindingSetDesc()
            .addItem(BindingSetItem::Texture_SRV(0, texture))
            .addItem(BindingSetItem::StructuredBuffer_UAV(1, output)), regularLayout);
        UAA_CHECK(regularSet != nullptr, "Sampler test binding set is null");

        CommandListHandle cmd = dev->createCommandList();
        cmd->open();
        cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(table).addBindingSet(regularSet));
        cmd->dispatch(1);
        cmd->close();
        fix.submitAndWait(cmd);
        std::array<uint32_t, 8> out = {};
        fix.readbackBuffer(output, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], 0u, "Clamp sampler red mismatch");
        UAA_CHECK_EQ(out[1], 255u, "Clamp sampler green mismatch");
        UAA_CHECK_EQ(out[4], 255u, "Wrap sampler red mismatch");
        UAA_CHECK_EQ(out[5], 0u, "Wrap sampler green mismatch");
    }

    static void testBindlessLargeOrdinary(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        if (fix.api() == GraphicsAPI::D3D12)
            ctx.skip("D3D12 descriptor tables require explicit allocation");
        if (fix.api() == GraphicsAPI::METAL && !fix.deviceInfo().metalArgumentBuffersTier2)
            ctx.skip("Metal Tier 2 argument buffers are required for this capacity");
        IDevice* dev = fix.device();

        BindlessHarness h = createBindlessHarness(ctx, fix, 256u, 3u, false);
        TextureHandle tex0 = createSolidRgba8Texture(fix, 12, 24, 36, 255, "tbl_large_tex0");
        TextureHandle texMid = createSolidRgba8Texture(fix, 48, 60, 72, 255, "tbl_large_tex127");
        TextureHandle texLast = createSolidRgba8Texture(fix, 84, 96, 108, 255, "tbl_large_tex255");
        BufferHandle buf0 = createStructuredWord(fix, 0xE000u, "tbl_large_buf0");
        BufferHandle bufMid = createStructuredBuffer(fix, 512, "tbl_large_buf127");
        BufferHandle bufLast = createStructuredWord(fix, 0xE0FFu, "tbl_large_buf255");
        UAA_CHECK(tex0 && texMid && texLast && buf0 && bufMid && bufLast, "Large table resources null");
        const uint32_t midWord = 0xE07Fu;
        fix.uploadBuffer(bufMid, &midWord, sizeof(midWord), 256);

        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(127, texMid)), "Write tex127 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(255, texLast)), "Write tex255 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(127, bufMid, Format::UNKNOWN, BufferRange(256, 16))), "Write buf127 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(255, bufLast)), "Write buf255 failed");

        runBindlessDispatch(ctx, fix, h, { 0u, 127u, 255u }, { 0u, 127u, 255u });

        std::array<uint32_t, 6> out = {};
        fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(12, 24, 36, 255), "Slot 0 tex mismatch");
        UAA_CHECK_EQ(out[1], 0xE000u, "Slot 0 buf mismatch");
        UAA_CHECK_EQ(out[2], packRgba8(48, 60, 72, 255), "Slot 127 tex mismatch");
        UAA_CHECK_EQ(out[3], 0xE07Fu, "Slot 127 buf mismatch");
        UAA_CHECK_EQ(out[4], packRgba8(84, 96, 108, 255), "Slot 255 tex mismatch");
        UAA_CHECK_EQ(out[5], 0xE0FFu, "Slot 255 buf mismatch");
    }

    static void testBindlessLimits(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        if (fix.api() == GraphicsAPI::METAL)
        {
            // Exercise the boundary of the classic binding path, which is also valid on Tier 2.
            const uint32_t metalLimit = 27u;
            BindlessHarness h = createBindlessHarness(ctx, fix, metalLimit, 4u);
            UAA_CHECK_EQ(h.table->getCapacity(), metalLimit, "Metal boundary capacity 27 mismatch");

            TextureHandle tex0 = createSolidRgba8Texture(fix, 5, 15, 25, 255, "tbl_lim_tex0");
            TextureHandle texMid = createSolidRgba8Texture(fix, 35, 45, 55, 255, "tbl_lim_tex13");
            TextureHandle texLast = createSolidRgba8Texture(fix, 65, 75, 85, 255, "tbl_lim_tex26");
            BufferHandle buf0 = createStructuredWord(fix, 0xD000u, "tbl_lim_buf0");
            BufferHandle bufMid = createStructuredWord(fix, 0xD013u, "tbl_lim_buf13");
            BufferHandle bufLast = createStructuredWord(fix, 0xD026u, "tbl_lim_buf26");
            UAA_CHECK(tex0 && texMid && texLast && buf0 && bufMid && bufLast, "Metal limit resources null");

            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(13, texMid)), "Write tex13 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(26, texLast)), "Write tex26 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(13, bufMid)), "Write buf13 failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(26, bufLast)), "Write buf26 failed");

            runBindlessDispatch(ctx, fix, h, { 0u, 13u, 26u }, { 0u, 13u, 26u });

            std::array<uint32_t, 6> out = {};
            fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
            UAA_CHECK_EQ(out[0], packRgba8(5, 15, 25, 255), "Metal limit slot 0 tex mismatch");
            UAA_CHECK_EQ(out[1], 0xD000u, "Metal limit slot 0 buf mismatch");
            UAA_CHECK_EQ(out[2], packRgba8(35, 45, 55, 255), "Metal limit slot 13 tex mismatch");
            UAA_CHECK_EQ(out[3], 0xD013u, "Metal limit slot 13 buf mismatch");
            UAA_CHECK_EQ(out[4], packRgba8(65, 75, 85, 255), "Metal limit slot 26 tex mismatch");
            UAA_CHECK_EQ(out[5], 0xD026u, "Metal limit slot 26 buf mismatch");

            if (!fix.deviceInfo().metalArgumentBuffersTier2)
            {
                ExpectErrorsGuard guard(fix);
                BindingLayoutHandle largeLayout = makeTexBufBindlessLayout(fix, 256u, false);
                UAA_CHECK(largeLayout != nullptr, "createBindlessLayout(256) returned null");
                ComputePipelineHandle oversizedPso = dev->createComputePipeline(
                    ComputePipelineDesc()
                        .setComputeShader(h.cs)
                        .addBindingLayout(largeLayout)
                        .addBindingLayout(h.regularLayout));
                UAA_CHECK(oversizedPso == nullptr, "Oversized capacity 256 must fail on the Metal classic binding path");
            }

            ctx.skip("Metal does not support update-after-bind");
        }

        const uint32_t largeCapacity = 256u;
        BindlessHarness h = createBindlessHarness(ctx, fix, largeCapacity, 4u);
        UAA_CHECK_EQ(h.table->getCapacity(), largeCapacity, "Large table capacity 256 mismatch");

        TextureHandle tex0 = createSolidRgba8Texture(fix, 12, 24, 36, 255, "tbl_lim_tex0");
        TextureHandle texMid = createSolidRgba8Texture(fix, 48, 60, 72, 255, "tbl_lim_tex127");
        TextureHandle texLast = createSolidRgba8Texture(fix, 84, 96, 108, 255, "tbl_lim_tex255");
        BufferHandle buf0 = createStructuredWord(fix, 0xE000u, "tbl_lim_buf0");
        BufferHandle bufMid = createStructuredWord(fix, 0xE07Fu, "tbl_lim_buf127");
        BufferHandle bufLast = createStructuredWord(fix, 0xE0FFu, "tbl_lim_buf255");
        UAA_CHECK(tex0 && texMid && texLast && buf0 && bufMid && bufLast, "Large limit resources null");

        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, tex0)), "Write tex0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(127, texMid)), "Write tex127 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(255, texLast)), "Write tex255 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, buf0)), "Write buf0 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(127, bufMid)), "Write buf127 failed");
        UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(255, bufLast)), "Write buf255 failed");

        runBindlessDispatch(ctx, fix, h, { 0u, 127u, 255u }, { 0u, 127u, 255u });

        std::array<uint32_t, 6> out = {};
        fix.readbackBuffer(h.outBuf, out.data(), sizeof(out));
        UAA_CHECK_EQ(out[0], packRgba8(12, 24, 36, 255), "Slot 0 tex mismatch");
        UAA_CHECK_EQ(out[1], 0xE000u, "Slot 0 buf mismatch");
        UAA_CHECK_EQ(out[2], packRgba8(48, 60, 72, 255), "Slot 127 tex mismatch");
        UAA_CHECK_EQ(out[3], 0xE07Fu, "Slot 127 buf mismatch");
        UAA_CHECK_EQ(out[4], packRgba8(84, 96, 108, 255), "Slot 255 tex mismatch");
        UAA_CHECK_EQ(out[5], 0xE0FFu, "Slot 255 buf mismatch");

        // Also check 1024-capacity allocation and query.
        BindingLayoutHandle layout1024 = makeTexBufBindlessLayout(fix, 1024u, true);
        UAA_CHECK(layout1024 != nullptr, "1024-capacity BindlessLayout returned null");
        DescriptorTableHandle table1024 = dev->createDescriptorTable(layout1024);
        UAA_CHECK(table1024 != nullptr, "1024-capacity DescriptorTable returned null");
        UAA_CHECK_EQ(table1024->getCapacity(), 1024u, "1024-capacity getCapacity() mismatch");
    }

    static void testBindlessUnsupported(TestContext& ctx)
    {
        TestFixture& fix = requireFixture(ctx);
        requireBindlessSupport(ctx, fix);
        IDevice* dev = fix.device();

        BindingLayoutHandle validBindless = makeTexBufBindlessLayout(fix, 8u, fix.api() != GraphicsAPI::METAL);
        BindingLayoutHandle regularLayout = dev->createBindingLayout(
            fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0)));
        UAA_CHECK(validBindless != nullptr && regularLayout != nullptr, "Base layouts returned null");

        DescriptorTableHandle table = dev->createDescriptorTable(validBindless);
        UAA_CHECK(table != nullptr, "createDescriptorTable returned null");

        TextureHandle tex = createSolidRgba8Texture(fix, 1, 2, 3, 255, "tbl_unsup_tex");
        UAA_CHECK(tex != nullptr, "Texture creation failed");

        {
            ExpectErrorsGuard guard(fix);

            // Out-of-range descriptor table writes must return false.
            UAA_CHECK(!dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(8, tex)),
                "writeDescriptorTable at slot == capacity must return false");
            UAA_CHECK(!dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(18, tex)),
                "writeDescriptorTable at slot > capacity must return false");

            // Non-bindless layout passed to createDescriptorTable (Vulkan raw dereferences getBindlessDesc() without checking).
            if (fix.isValidationActive() || fix.api() == GraphicsAPI::METAL)
            {
                DescriptorTableHandle badTable = dev->createDescriptorTable(regularLayout);
                UAA_CHECK(badTable == nullptr, "createDescriptorTable(regularLayout) must return nullptr");
            }

            if (fix.isValidationActive())
            {
                BindlessLayoutDesc noVis;
                noVis.setVisibility(ShaderType::None).setMaxCapacity(8).addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
                UAA_CHECK(dev->createBindlessLayout(noVis) == nullptr, "visibility=None bindless layout must return nullptr");

                BindlessLayoutDesc zeroCap;
                zeroCap.setVisibility(ShaderType::Compute).setMaxCapacity(0).addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
                UAA_CHECK(dev->createBindlessLayout(zeroCap) == nullptr, "maxCapacity=0 bindless layout must return nullptr");

                BindlessLayoutDesc emptySpaces;
                emptySpaces.setVisibility(ShaderType::Compute).setMaxCapacity(8);
                UAA_CHECK(dev->createBindlessLayout(emptySpaces) == nullptr, "Immutable layout without registerSpaces must return nullptr");

                BindlessLayoutDesc mutableWithSpaces;
                mutableWithSpaces.setVisibility(ShaderType::Compute)
                    .setMaxCapacity(8)
                    .setLayoutType(BindlessLayoutDesc::LayoutType::MutableSrvUavCbv)
                    .addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
                UAA_CHECK(dev->createBindlessLayout(mutableWithSpaces) == nullptr,
                    "Mutable layout with registerSpaces must return nullptr");

                BindlessLayoutDesc pushConstSpace;
                pushConstSpace.setVisibility(ShaderType::Compute)
                    .setMaxCapacity(8)
                    .addRegisterSpace(BindingLayoutItem::PushConstants(0, 16));
                UAA_CHECK(dev->createBindlessLayout(pushConstSpace) == nullptr,
                    "Bindless layout with PushConstants must return nullptr");
            }

            if (!fix.features().supports(Feature::HeapDirectlyIndexed))
            {
                if (fix.api() == GraphicsAPI::VULKAN)
                {
                    ExpectErrorsGuard mutGuard(fix);
                    BindlessLayoutDesc mutDesc;
                    mutDesc.setVisibility(ShaderType::Compute)
                        .setMaxCapacity(8)
                        .setLayoutType(BindlessLayoutDesc::LayoutType::MutableSrvUavCbv);
                    BindingLayoutHandle mutLayout = dev->createBindlessLayout(mutDesc);
                    // Drivers such as MoltenVK log an error via m_Context.error while still returning a layout handle.
                    const bool rejected = mutGuard.errorCount() > 0
                        || mutLayout == nullptr
                        || mutLayout->getNativeObject(ObjectTypes::VK_DescriptorSetLayout).pointer == nullptr;
                    UAA_CHECK(rejected,
                        "Vulkan mutable bindless layout must be rejected when HeapDirectlyIndexed is false");
                }
                else
                {
                    ctx.note("Metal does not enforce HeapDirectlyIndexed in createBindlessLayout metadata; skipping mutable descriptor set rejection check");
                }
            }
        }

        if (fix.backend())
        {
            fix.backend()->clearNativeValidationMessages();
        }
    }

        // Two read-only texture arrays in the bindless set: a sampled texture array
        // and a typed buffer array. Both take texture arguments on Metal, so a
        // backend that picks the array by resource kind alone confuses the two.
        static void testBindlessTwoTextureKinds(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            IDevice* dev = fix.device();

            const bool isMetal = (fix.api() == GraphicsAPI::METAL);
            const uint32_t capacity = isMetal ? 12u : 16u;

            ShaderHandle cs = fix.createShader("cs_bindless_two_tex_kinds", ShaderType::Compute);
            UAA_CHECK(cs != nullptr, "shader creation failed");

            BindlessLayoutDesc bindlessDesc;
            bindlessDesc.setVisibility(ShaderType::Compute)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setMaxCapacity(capacity)
                .setUpdateAfterBind(false)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0))
                .addRegisterSpace(BindingLayoutItem::TypedBuffer_SRV(1));
            BindingLayoutHandle bindlessLayout = dev->createBindlessLayout(bindlessDesc);
            UAA_CHECK(bindlessLayout != nullptr, "createBindlessLayout returned null");

            BindingLayoutHandle regularLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, regularSpaceForBindless(fix), false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(BindingLayoutItem::Sampler(3))
                    .addItem(BindingLayoutItem::ConstantBuffer(4)));
            UAA_CHECK(regularLayout != nullptr, "regular layout creation failed");

            DescriptorTableHandle table = dev->createDescriptorTable(bindlessLayout);
            UAA_CHECK(table != nullptr, "createDescriptorTable returned null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs)
                    .addBindingLayout(bindlessLayout)
                    .addBindingLayout(regularLayout));
            UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

            TextureHandle texRed = createSolidRgba8Texture(fix, 255, 0, 0, 255, "tbl_2k_texRed");
            TextureHandle texBlue = createSolidRgba8Texture(fix, 0, 0, 255, 255, "tbl_2k_texBlue");
            UAA_CHECK(texRed && texBlue, "texture creation failed");

            auto makeTyped = [&fix](uint32_t word, const char* name)
            {
                BufferDesc desc;
                desc.setByteSize(64)
                    .setCanHaveTypedViews(true)
                    .setFormat(Format::RGBA32_UINT)
                    .setInitialState(ResourceStates::ShaderResource)
                    .setKeepInitialState(true)
                    .setDebugName(name);
                BufferHandle buf = fix.device()->createBuffer(desc);
                if (buf)
                {
                    const std::array<uint32_t, 4> words = { word, 0u, 0u, 0u };
                    fix.uploadBuffer(buf, words.data(), sizeof(words));
                }
                return buf;
            };
            BufferHandle typedA = makeTyped(0x1000u, "tbl_2k_typedA");
            BufferHandle typedB = makeTyped(0x2000u, "tbl_2k_typedB");
            UAA_CHECK(typedA && typedB, "typed buffer creation failed");

            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(0, texRed)), "tex slot 0 write failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(1, texBlue)), "tex slot 1 write failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::TypedBuffer_SRV(0, typedA, Format::RGBA32_UINT)),
                "typed slot 0 write failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::TypedBuffer_SRV(1, typedB, Format::RGBA32_UINT)),
                "typed slot 1 write failed");

            BufferHandle texIndices = createStructuredBuffer(fix, 16, "tbl_2k_texIdx");
            BufferHandle typedIndices = createStructuredBuffer(fix, 16, "tbl_2k_typedIdx");
            BufferDesc outDesc;
            outDesc.setByteSize(128)
                .setStructStride(16)
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("tbl_2k_out");
            BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "out buffer creation failed");
            BufferHandle cbBuf = createConstantBuffer(fix, 256, "tbl_2k_cb");
            UAA_CHECK(texIndices && typedIndices && outBuf && cbBuf, "harness buffer creation failed");

            SamplerHandle sampler = dev->createSampler(
                SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
            UAA_CHECK(sampler != nullptr, "sampler creation failed");

            BindingSetHandle regularSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, texIndices))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, typedIndices))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                    .addItem(BindingSetItem::Sampler(3, sampler))
                    .addItem(BindingSetItem::ConstantBuffer(4, cbBuf)),
                regularLayout);
            UAA_CHECK(regularSet != nullptr, "regular binding set creation failed");

            const std::array<uint32_t, 4> texIdx = { 0u, 1u, 0u, 1u };
            const std::array<uint32_t, 4> typedIdx = { 0u, 0u, 1u, 1u };
            fix.uploadBuffer(texIndices, texIdx.data(), sizeof(texIdx));
            fix.uploadBuffer(typedIndices, typedIdx.data(), sizeof(typedIdx));
            const std::array<uint32_t, 4> params = { 4u, 0u, 0u, 0u };
            fix.uploadBuffer(cbBuf, params.data(), sizeof(params));

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(table).addBindingSet(regularSet));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            std::array<uint32_t, 16> out = {};
            fix.readbackBuffer(outBuf, out.data(), sizeof(out));

            // The shader writes one uint4 per element: the sampled colour plus the
            // typed words, so both arrays contribute to the result.
            const uint32_t expected[4][4] = {
                { 255u + 0x1000u, 0u, 0u, 255u },
                { 0u + 0x1000u, 0u, 255u, 255u },
                { 255u + 0x2000u, 0u, 0u, 255u },
                { 0u + 0x2000u, 0u, 255u, 255u }
            };
            for (uint32_t e = 0; e < 4u; ++e)
            {
                for (uint32_t w = 0; w < 4u; ++w)
                {
                    UAA_CHECK_EQ(out[e * 4u + w], expected[e][w],
                        "combined texture+typed mismatch at element " + std::to_string(e) + " word " + std::to_string(w));
                }
            }
        }

        // A cube array of the bindless set with holes between the written slots. The
        // null resources of the holes must match the cube texture type of the shader.
        static void testBindlessCubeSparse(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            IDevice* dev = fix.device();

            const uint32_t capacity = 8u;
            const uint32_t slotA = 0u;
            const uint32_t slotB = 2u;
            const uint32_t slotC = 5u;

            ShaderHandle cs = fix.createShader("cs_bindless_cube_sparse", ShaderType::Compute);
            UAA_CHECK(cs != nullptr, "shader creation failed");

            BindlessLayoutDesc bindlessDesc;
            bindlessDesc.setVisibility(ShaderType::Compute)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setMaxCapacity(capacity)
                .setUpdateAfterBind(false)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
            BindingLayoutHandle bindlessLayout = dev->createBindlessLayout(bindlessDesc);
            UAA_CHECK(bindlessLayout != nullptr, "createBindlessLayout returned null");

            BindingLayoutHandle regularLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, regularSpaceForBindless(fix), false)
                    .addItem(BindingLayoutItem::StructuredBuffer_SRV(0))
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(1))
                    .addItem(BindingLayoutItem::Sampler(2))
                    .addItem(BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(regularLayout != nullptr, "regular layout creation failed");

            DescriptorTableHandle table = dev->createDescriptorTable(bindlessLayout);
            UAA_CHECK(table != nullptr, "createDescriptorTable returned null");

            ComputePipelineHandle pso = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs)
                    .addBindingLayout(bindlessLayout)
                    .addBindingLayout(regularLayout));
            UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

            auto makeCube = [&fix](uint8_t r, uint8_t g, uint8_t b, const char* name)
            {
                TextureDesc desc;
                desc.setWidth(2)
                    .setHeight(2)
                    .setArraySize(6)
                    .setDimension(TextureDimension::TextureCube)
                    .setFormat(Format::RGBA8_UNORM)
                    .setInitialState(ResourceStates::ShaderResource)
                    .setKeepInitialState(true)
                    .setDebugName(name);
                TextureHandle tex = fix.device()->createTexture(desc);
                if (tex)
                {
                    const std::array<uint8_t, 16> pixels = {
                        r, g, b, 255, r, g, b, 255,
                        r, g, b, 255, r, g, b, 255
                    };
                    for (uint32_t face = 0; face < 6u; ++face)
                        fix.uploadTexture2D(tex, face, 0, pixels.data(), 8u);
                }
                return tex;
            };
            TextureHandle cubeA = makeCube(10, 20, 30, "tbl_cube_A");
            TextureHandle cubeB = makeCube(40, 50, 60, "tbl_cube_B");
            TextureHandle cubeC = makeCube(70, 80, 90, "tbl_cube_C");
            UAA_CHECK(cubeA && cubeB && cubeC, "cube texture creation failed");

            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(slotA, cubeA)), "cube slot A write failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(slotB, cubeB)), "cube slot B write failed");
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(slotC, cubeC)), "cube slot C write failed");

            BufferHandle indices = createStructuredBuffer(fix, 16, "tbl_cube_idx");
            BufferDesc outDesc;
            outDesc.setByteSize(64)
                .setStructStride(16)
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("tbl_cube_out");
            BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "cube out buffer creation failed");
            BufferHandle cbBuf = createConstantBuffer(fix, 256, "tbl_cube_cb");
            SamplerHandle sampler = dev->createSampler(
                SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
            UAA_CHECK(indices && outBuf && cbBuf && sampler, "harness creation failed");

            BindingSetHandle regularSet = dev->createBindingSet(
                BindingSetDesc()
                    .addItem(BindingSetItem::StructuredBuffer_SRV(0, indices))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(1, outBuf))
                    .addItem(BindingSetItem::Sampler(2, sampler))
                    .addItem(BindingSetItem::ConstantBuffer(3, cbBuf)),
                regularLayout);
            UAA_CHECK(regularSet != nullptr, "regular binding set creation failed");

            const std::array<uint32_t, 3> idx = { slotA, slotB, slotC };
            fix.uploadBuffer(indices, idx.data(), sizeof(idx));
            const std::array<uint32_t, 4> params = { 3u, 0u, 0u, 0u };
            fix.uploadBuffer(cbBuf, params.data(), sizeof(params));

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setComputeState(ComputeState().setPipeline(pso).addBindingSet(table).addBindingSet(regularSet));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            std::array<uint32_t, 12> out = {};
            fix.readbackBuffer(outBuf, out.data(), sizeof(out));
            // The shader writes one uint4 per element, so the components of each
            // element are the sampled colour channels.
            const uint32_t expected[3][4] = {
                { 10u, 20u, 30u, 255u },
                { 40u, 50u, 60u, 255u },
                { 70u, 80u, 90u, 255u }
            };
            for (uint32_t e = 0; e < 3u; ++e)
            {
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    UAA_CHECK_EQ(out[e * 4u + c], expected[e][c],
                        "cube slot mismatch at element " + std::to_string(e) + " channel " + std::to_string(c));
                }
            }
        }

        // A graphics pipeline whose vertex and fragment stages each read their own
        // texture array of the bindless set. The stages translate separately, so the
        // arrays take different Metal indices and each stage needs its own binding.
        static void runBindlessGraphicsStages(TestContext& ctx, bool separateSets)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            if (separateSets && fix.api() == GraphicsAPI::D3D12)
                ctx.skip("This shader uses Vulkan descriptor set indices");
            IDevice* dev = fix.device();

            const uint32_t capacity = 4u;

            ShaderHandle vs = fix.createShader("vs_bindless_stage", ShaderType::Vertex);
            ShaderHandle ps = fix.createShader(separateSets ? "ps_bindless_separate_sets" : "ps_bindless_stage", ShaderType::Pixel);
            UAA_CHECK(vs != nullptr && ps != nullptr, "shader creation failed");

            BindlessLayoutDesc bindlessDesc;
            bindlessDesc.setVisibility(ShaderType::AllGraphics)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setUpdateAfterBind(false)
                .setMaxCapacity(capacity)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0));
            if (!separateSets)
                bindlessDesc.addRegisterSpace(BindingLayoutItem::Texture_SRV(1));
            BindingLayoutHandle bindlessLayout = dev->createBindlessLayout(bindlessDesc);
            BindingLayoutHandle pixelLayout;
            if (separateSets)
            {
                bindlessDesc.setVisibility(ShaderType::Pixel).setMaxCapacity(8);
                pixelLayout = dev->createBindlessLayout(bindlessDesc);
            }
            UAA_CHECK(bindlessLayout && (!separateSets || pixelLayout), "createBindlessLayout returned null");

            BindingLayoutHandle samplerLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::AllGraphics, regularSpaceForBindless(fix), false)
                    .addItem(BindingLayoutItem::Sampler(1)));
            UAA_CHECK(samplerLayout != nullptr, "sampler layout creation failed");

            DescriptorTableHandle table = dev->createDescriptorTable(bindlessLayout);
            DescriptorTableHandle pixelTable = separateSets ? dev->createDescriptorTable(pixelLayout) : table;
            UAA_CHECK(table && pixelTable, "createDescriptorTable returned null");

            TextureDesc rtDesc;
            rtDesc.setWidth(8)
                .setHeight(8)
                .setFormat(Format::RGBA8_UNORM)
                .setIsRenderTarget(true)
                .enableAutomaticStateTracking(ResourceStates::RenderTarget)
                .setDebugName("tbl_stage_rt");
            TextureHandle rt = dev->createTexture(rtDesc);
            UAA_CHECK(rt != nullptr, "render target creation failed");
            FramebufferHandle fb = dev->createFramebuffer(FramebufferDesc().addColorAttachment(rt));
            UAA_CHECK(fb != nullptr, "framebuffer creation failed");

            GraphicsPipelineDesc psoDesc;
            psoDesc.setVertexShader(vs)
                .setPixelShader(ps)
                .addBindingLayout(bindlessLayout)
                .addBindingLayout(samplerLayout)
                .setRenderState(RenderState()
                    .setRasterState(RasterState().setCullNone())
                    .setDepthStencilState(DepthStencilState().setDepthTestEnable(false).setDepthWriteEnable(false).setStencilEnable(false)));
            if (separateSets)
                psoDesc.addBindingLayout(pixelLayout);
            GraphicsPipelineHandle pso = dev->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "createGraphicsPipeline returned null");

            TextureHandle vertTex = createSolidRgba8Texture(fix, 128, 128, 128, 255, "tbl_stage_vert");
            TextureHandle pixTex = createSolidRgba8Texture(fix, 0, 255, 0, 255, "tbl_stage_pix");
            UAA_CHECK(vertTex && pixTex, "texture creation failed");

            // The vertex stage reads slot 0 and the fragment stage reads slot 1.
            // In the shared-table case, each write reaches both register spaces.
            UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(0, vertTex)), "slot 0 write failed");
            UAA_CHECK(dev->writeDescriptorTable(pixelTable, BindingSetItem::Texture_SRV(1, pixTex)), "slot 1 write failed");

            SamplerHandle sampler = dev->createSampler(
                SamplerDesc().setAllFilters(false).setAllAddressModes(SamplerAddressMode::Clamp));
            UAA_CHECK(sampler != nullptr, "sampler creation failed");
            BindingSetHandle samplerSet = dev->createBindingSet(
                BindingSetDesc().addItem(BindingSetItem::Sampler(1, sampler)), samplerLayout);
            UAA_CHECK(samplerSet != nullptr, "sampler binding set creation failed");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->clearTextureFloat(rt, AllSubresources, Color(0.f, 0.f, 0.f, 1.f));
            GraphicsState state;
            state.setPipeline(pso).setFramebuffer(fb).addBindingSet(table).addBindingSet(samplerSet)
                .setViewport(ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport()));
            if (separateSets)
                state.addBindingSet(pixelTable);
            cmd->setGraphicsState(state);
            cmd->draw(DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(rt, 0, 0, pixels);
            UAA_CHECK_EQ(pixels.size(), size_t(8 * 8 * 4), "render target readback size mismatch");
            // Vertex colour grey times pixel colour green gives a dark green.
            for (size_t i = 0; i < pixels.size(); i += 4)
            {
                UAA_CHECK_NEAR(pixels[i], 0, 2, "stage table red channel mismatch");
                UAA_CHECK_NEAR(pixels[i + 1], 128, 3, "stage table green channel mismatch");
                UAA_CHECK_NEAR(pixels[i + 2], 0, 2, "stage table blue channel mismatch");
                UAA_CHECK_NEAR(pixels[i + 3], 255, 2, "stage table alpha channel mismatch");
            }
        }

        static void testBindlessGraphicsStages(TestContext& ctx)
        {
            runBindlessGraphicsStages(ctx, false);
        }

        static void testBindlessGraphicsMultipleLayouts(TestContext& ctx)
        {
            runBindlessGraphicsStages(ctx, true);
        }

        // Descriptor set indices outside the backend limit must be rejected.
        static void testBindlessLayoutLimit(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            if (fix.api() != GraphicsAPI::METAL)
            {
                ctx.skip("The Metal descriptor set limit is backend-specific");
            }

            IDevice* dev = fix.device();
            ShaderHandle cs = fix.createShader("cs_bindless_tex_buf", ShaderType::Compute);
            UAA_CHECK(cs != nullptr, "shader creation failed");

            BindlessLayoutDesc first;
            first.setVisibility(ShaderType::Compute)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setUpdateAfterBind(false)
                .setMaxCapacity(8u)
                .addRegisterSpace(BindingLayoutItem::Texture_SRV(0))
                .addRegisterSpace(BindingLayoutItem::StructuredBuffer_SRV(1));
            BindingLayoutHandle firstLayout = dev->createBindlessLayout(first);
            UAA_CHECK(firstLayout != nullptr, "first bindless layout returned null");

            BindlessLayoutDesc second;
            second.setVisibility(ShaderType::Compute)
                .setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setUpdateAfterBind(false)
                .setMaxCapacity(4u)
                .addRegisterSpace(BindingLayoutItem::TypedBuffer_SRV(0));
            BindingLayoutHandle secondLayout = dev->createBindlessLayout(second);
            UAA_CHECK(secondLayout != nullptr, "second bindless layout returned null");

            BindingLayoutHandle regular = dev->createBindingLayout(BindingLayoutDesc()
                .setVisibility(ShaderType::Compute).setRegisterSpaceAndDescriptorSet(c_MaxBindingLayouts - 1)
                .addItem(BindingLayoutItem::Sampler(0)));
            UAA_CHECK(regular != nullptr, "regular layout returned null");
            {
                ExpectErrorsGuard guard(fix);
                ComputePipelineHandle pso = dev->createComputePipeline(
                    ComputePipelineDesc().setComputeShader(cs)
                        .addBindingLayout(firstLayout)
                        .addBindingLayout(secondLayout)
                        .addBindingLayout(regular));
                UAA_CHECK(pso == nullptr, "descriptor sets beyond the backend limit must be rejected");
                UAA_CHECK(guard.errorCount() > 0u, "the rejection must report an error");
            }

            // A pipeline with only the first layout still works after the rejection.
            ComputePipelineHandle good = dev->createComputePipeline(
                ComputePipelineDesc().setComputeShader(cs).addBindingLayout(firstLayout));
            UAA_CHECK(good != nullptr, "pipeline creation after the rejection failed");
        }

        // An update after recording but before submission must reach the dispatch.
        static void testBindlessPreSubmitUpdate(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            if (fix.api() == GraphicsAPI::METAL)
                UAA_SKIP("Metal does not support update-after-bind");
            IDevice* dev = fix.device();
            BindlessHarness h = createBindlessHarness(ctx, fix, 8, 1, true);
            TextureHandle before = createSolidRgba8Texture(fix, 11, 22, 33, 255, "tbl_late_before");
            TextureHandle after = createSolidRgba8Texture(fix, 44, 55, 66, 255, "tbl_late_after");
            BufferHandle beforeBuf = createStructuredWord(fix, 0x1234u, "tbl_late_before_buf");
            BufferHandle afterBuf = createStructuredWord(fix, 0x5678u, "tbl_late_after_buf");
            UAA_CHECK(before && after && beforeBuf && afterBuf, "late update resources failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, before)), "initial texture write failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, beforeBuf)), "initial buffer write failed");
            const std::array<uint32_t, 4> zero = {0, 0, 0, 0};
            const std::array<uint32_t, 4> params = {1, 0, 0, 0};
            fix.uploadBuffer(h.texIndices, zero.data(), sizeof(zero));
            fix.uploadBuffer(h.bufIndices, zero.data(), sizeof(zero));
            fix.uploadBuffer(h.cbBuf, params.data(), sizeof(params));

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "late update command list failed");
            cmd->open();
            cmd->setComputeState(ComputeState().setPipeline(h.pso).addBindingSet(h.table).addBindingSet(h.regularSet));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::Texture_SRV(0, after)), "late texture write failed");
            UAA_CHECK(dev->writeDescriptorTable(h.table, BindingSetItem::StructuredBuffer_SRV(0, afterBuf)), "late buffer write failed");
            fix.submitAndWait(cmd);
            std::array<uint32_t, 2> output = {};
            fix.readbackBuffer(h.outBuf, output.data(), sizeof(output));
            UAA_CHECK_EQ(output[0], packRgba8(44, 55, 66, 255), "dispatch did not see the late texture update");
            UAA_CHECK_EQ(output[1], 0x5678u, "dispatch did not see the late buffer update");
        }

        static void testBindlessUintSparse(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            requireBindlessSupport(ctx, fix);
            IDevice* dev = fix.device();
            ShaderHandle shader = fix.createShader("cs_bindless_uint_sparse", ShaderType::Compute);
            UAA_CHECK(shader != nullptr, "integer texture shader creation failed");
            BindingLayoutHandle tableLayout = dev->createBindlessLayout(BindlessLayoutDesc()
                .setVisibility(ShaderType::Compute).setLayoutType(BindlessLayoutDesc::LayoutType::Immutable)
                .setUpdateAfterBind(false).setMaxCapacity(4).addRegisterSpace(BindingLayoutItem::Texture_SRV(0)));
            BindingLayoutHandle outputLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(ShaderType::Compute, regularSpaceForBindless(fix), false)
                    .addItem(BindingLayoutItem::StructuredBuffer_UAV(0)));
            UAA_CHECK(tableLayout && outputLayout, "integer texture layouts failed");
            DescriptorTableHandle table = dev->createDescriptorTable(tableLayout);
            ComputePipelineHandle pipeline = dev->createComputePipeline(ComputePipelineDesc().setComputeShader(shader)
                .addBindingLayout(tableLayout).addBindingLayout(outputLayout));
            UAA_CHECK(table && pipeline, "integer texture table or pipeline failed");

            std::array<TextureHandle, 2> textures;
            const std::array<std::array<uint8_t, 4>, 2> values = {{{{7, 13, 29, 255}}, {{41, 67, 101, 255}}}};
            for (uint32_t i = 0; i < textures.size(); ++i)
            {
                textures[i] = dev->createTexture(TextureDesc().setWidth(1).setHeight(1).setFormat(Format::RGBA8_UINT)
                    .setInitialState(ResourceStates::ShaderResource).setKeepInitialState(true).setDebugName("tbl_uint_sparse"));
                UAA_CHECK(textures[i] != nullptr, "integer texture creation failed");
                fix.uploadTexture2D(textures[i], 0, 0, values[i].data(), 4);
                UAA_CHECK(dev->writeDescriptorTable(table, BindingSetItem::Texture_SRV(i * 2 + 1, textures[i])),
                    "integer texture table write failed");
            }
            BufferHandle output = dev->createBuffer(BufferDesc().setByteSize(32).setStructStride(16).setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess).setKeepInitialState(true).setDebugName("tbl_uint_output"));
            UAA_CHECK(output != nullptr, "integer texture output creation failed");
            BindingSetHandle outputSet = dev->createBindingSet(
                BindingSetDesc().addItem(BindingSetItem::StructuredBuffer_UAV(0, output)), outputLayout);
            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(outputSet && cmd, "integer texture binding or command list failed");
            cmd->open();
            cmd->setComputeState(ComputeState().setPipeline(pipeline).addBindingSet(table).addBindingSet(outputSet));
            cmd->dispatch(2, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);
            std::array<uint32_t, 8> actual = {};
            fix.readbackBuffer(output, actual.data(), sizeof(actual));
            for (size_t i = 0; i < textures.size(); ++i)
                for (size_t channel = 0; channel < 4; ++channel)
                    UAA_CHECK_EQ(actual[i * 4 + channel], uint32_t(values[i][channel]), "integer texture readback mismatch");
        }

        static void testMetalUabRejected(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (fix.api() != GraphicsAPI::METAL)
                UAA_SKIP("Metal update-after-bind rejection applies only to Metal");
            ExpectErrorsGuard guard(fix);
            BindingLayoutHandle layout = makeTexBufBindlessLayout(fix, 8, true);
            UAA_CHECK(layout == nullptr, "Metal accepted an update-after-bind layout");
            UAA_CHECK(guard.errorCount() > 0, "Metal did not report the unsupported layout");
        }

        UAA_REGISTER_TEST("TBL.CreateWrite", "core", testBindlessCreateWrite);
        UAA_REGISTER_TEST("TBL.MixedSpaces", "core", testBindlessMixedSpaces);
        UAA_REGISTER_TEST("TBL.NonUniform", "core", testBindlessNonUniform);
        UAA_REGISTER_TEST("TBL.Resize", "core", testBindlessResize);
        UAA_REGISTER_TEST("TBL.Clear", "core", testBindlessClear);
        UAA_REGISTER_TEST("TBL.UpdateAfterBind", "core", testBindlessUpdateAfterBind);
        UAA_REGISTER_TEST("TBL.PendingLifetime", "core", testBindlessPendingLifetime);
        UAA_REGISTER_TEST("TBL.LargeOrdinary", "core", testBindlessLargeOrdinary);
        UAA_REGISTER_TEST("TBL.MultipleLayouts", "core", testBindlessMultipleLayouts);
        UAA_REGISTER_TEST("TBL.MultipleLayoutsExplicit", "core", testBindlessMultipleLayoutsExplicit);
        UAA_REGISTER_TEST("TBL.Samplers", "core", testBindlessSamplers);
        UAA_REGISTER_TEST("TBL.TypedUav", "core", testBindlessTypedUav);
        UAA_REGISTER_TEST("TBL.Limits", "core", testBindlessLimits);
        UAA_REGISTER_TEST("TBL.Unsupported", "core", testBindlessUnsupported);
        UAA_REGISTER_TEST("TBL.TwoTextureKinds", "core", testBindlessTwoTextureKinds);
        UAA_REGISTER_TEST("TBL.CubeSparse", "core", testBindlessCubeSparse);
        UAA_REGISTER_TEST("TBL.UintSparse", "core", testBindlessUintSparse);
        UAA_REGISTER_TEST("TBL.PreSubmitUpdate", "core", testBindlessPreSubmitUpdate);
        UAA_REGISTER_TEST("TBL.MetalUabRejected", "core,metal", testMetalUabRejected);
        UAA_REGISTER_TEST("TBL.GraphicsStages", "core", testBindlessGraphicsStages);
        UAA_REGISTER_TEST("TBL.GraphicsMultipleLayouts", "core", testBindlessGraphicsMultipleLayouts);
        UAA_REGISTER_TEST("TBL.LayoutLimit", "core", testBindlessLayoutLimit);
    } // namespace nvrhi::uaa
