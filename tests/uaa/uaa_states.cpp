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

#include <nvrhi/utils.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct CopyAddParams
        {
            uint32_t count = 0;
            uint32_t addVal = 0;
            uint32_t xorVal = 0;
            uint32_t pad = 0;
        };
        static_assert(sizeof(CopyAddParams) == 16, "CopyAddParams size mismatch");

        struct UavChainPC
        {
            uint32_t count = 0;
            uint32_t stepAdd = 0;
            uint32_t stepMul = 1;
            uint32_t pad = 0;
        };
        static_assert(sizeof(UavChainPC) == 16, "UavChainPC size mismatch");

        struct SamplerTestParams
        {
            float uvOrigin[2] = {0.0f, 0.0f};
            float uvStep[2] = {0.0f, 0.0f};
            float lod = 0.0f;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mode = 0;
        };
        static_assert(sizeof(SamplerTestParams) == 32, "SamplerTestParams size mismatch");

        struct FullscreenPC
        {
            float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            float depth = 0.5f;
            float scale = 1.0f;
            float offset[2] = {0.0f, 0.0f};
        };
        static_assert(sizeof(FullscreenPC) == 32, "FullscreenPC size mismatch");

        struct CopyAddPipelineBundle
        {
            ShaderHandle shader;
            BindingLayoutHandle layout;
            ComputePipelineHandle pipeline;
        };

        CopyAddPipelineBundle createCopyAddBundle(TestContext& ctx, TestFixture& fix)
        {
            CopyAddPipelineBundle bundle;
            bundle.shader = fix.createShader("cs_buffer_copy_add", ShaderType::Compute);
            UAA_CHECK(bundle.shader != nullptr, "cs_buffer_copy_add shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::ConstantBuffer(0))
                .addItem(BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2));
            bundle.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "createBindingLayout for cs_buffer_copy_add returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(bundle.shader)
                .addBindingLayout(bundle.layout);
            bundle.pipeline = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(bundle.pipeline != nullptr, "createComputePipeline for cs_buffer_copy_add returned null");
            return bundle;
        }

        struct SamplerPipelineBundle
        {
            ShaderHandle shader;
            BindingLayoutHandle layout;
            ComputePipelineHandle pipeline;
            SamplerHandle sampler;
        };

        SamplerPipelineBundle createSamplerBundle(TestContext& ctx, TestFixture& fix)
        {
            SamplerPipelineBundle bundle;
            bundle.shader = fix.createShader("cs_sampler_test", ShaderType::Compute);
            UAA_CHECK(bundle.shader != nullptr, "cs_sampler_test shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::Texture_SRV(0))
                .addItem(BindingLayoutItem::Sampler(1))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(2))
                .addItem(BindingLayoutItem::ConstantBuffer(3));
            bundle.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "createBindingLayout for cs_sampler_test returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(bundle.shader)
                .addBindingLayout(bundle.layout);
            bundle.pipeline = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(bundle.pipeline != nullptr, "createComputePipeline for cs_sampler_test returned null");

            SamplerDesc sampDesc;
            sampDesc.setAllFilters(false)
                .setAllAddressModes(SamplerAddressMode::ClampToEdge);
            bundle.sampler = fix.device()->createSampler(sampDesc);
            UAA_CHECK(bundle.sampler != nullptr, "createSampler returned null");
            return bundle;
        }

        bool tracksExplicitStates(GraphicsAPI api)
        {
            return api == GraphicsAPI::VULKAN || api == GraphicsAPI::D3D12;
        }

        void checkExpectedTrackedState(
            TestContext& ctx,
            GraphicsAPI api,
            ResourceStates actual,
            ResourceStates expectedOnTracked,
            const char* label)
        {
            if (tracksExplicitStates(api))
            {
                UAA_CHECK_EQ(actual, expectedOnTracked, label);
            }
            else if (api == GraphicsAPI::METAL)
            {
                UAA_CHECK_EQ(actual, ResourceStates::Unknown, label);
            }
            else if (api == GraphicsAPI::D3D11)
            {
                UAA_CHECK_EQ(actual, ResourceStates::Common, label);
            }
        }

        std::vector<uint8_t> makeSolidRgba8(uint32_t width, uint32_t height, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            std::vector<uint8_t> pixels(size_t(width) * height * 4u);
            for (size_t i = 0; i < size_t(width) * height; ++i)
            {
                pixels[i * 4u + 0u] = r;
                pixels[i * 4u + 1u] = g;
                pixels[i * 4u + 2u] = b;
                pixels[i * 4u + 3u] = a;
            }
            return pixels;
        }

        void verifySolidRgba8(
            TestContext& ctx,
            const std::vector<uint8_t>& pixels,
            uint32_t width,
            uint32_t height,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            const char* label)
        {
            UAA_CHECK_EQ(pixels.size(), size_t(width) * height * 4u, label);
            for (size_t i = 0; i < size_t(width) * height; ++i)
            {
                std::string diff;
                const bool ok = TestFixture::checkPixelRgba8(&pixels[i * 4u], r, g, b, a, 0, diff);
                if (!ok)
                {
                    UAA_CHECK(false, std::string(label) + " pixel " + std::to_string(i) + ": " + diff);
                }
            }
        }

        // Tests textures and buffers configured with keepInitialState=true versus manual beginTracking*State.
        void testStateInitial(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();
            const GraphicsAPI api = fix.api();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kTexSize = 8;
            constexpr uint32_t kAddVal = 11;
            constexpr uint32_t kXorVal = 0x5A5A5A5Au;

            const CopyAddPipelineBundle copyBundle = createCopyAddBundle(ctx, fix);
            const SamplerPipelineBundle sampBundle = createSamplerBundle(ctx, fix);

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x101u);

            // Part A: keepInitialState = true with explicit initialState.
            {
                BufferDesc inDesc;
                inDesc.setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setInitialState(ResourceStates::ShaderResource)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_in");
                BufferHandle inBuf = dev->createBuffer(inDesc);
                UAA_CHECK(inBuf != nullptr, "createBuffer keep_in returned null");

                BufferDesc outDesc;
                outDesc.setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setInitialState(ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_out");
                BufferHandle outBuf = dev->createBuffer(outDesc);
                UAA_CHECK(outBuf != nullptr, "createBuffer keep_out returned null");

                BufferDesc cbDesc;
                cbDesc.setByteSize(sizeof(CopyAddParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_cb");
                BufferHandle cb = dev->createBuffer(cbDesc);
                UAA_CHECK(cb != nullptr, "createBuffer keep_cb returned null");

                TextureDesc texDesc;
                texDesc.setWidth(kTexSize)
                    .setHeight(kTexSize)
                    .setFormat(Format::RGBA8_UNORM)
                    .setIsUAV(true)
                    .setInitialState(ResourceStates::ShaderResource)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_tex");
                TextureHandle tex = dev->createTexture(texDesc);
                UAA_CHECK(tex != nullptr, "createTexture keep_tex returned null");

                BufferDesc texOutDesc;
                texOutDesc.setByteSize(size_t(kTexSize) * kTexSize * 16u)
                    .setStructStride(16u)
                    .setCanHaveUAVs(true)
                    .setInitialState(ResourceStates::UnorderedAccess)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_tex_out");
                BufferHandle texOutBuf = dev->createBuffer(texOutDesc);
                UAA_CHECK(texOutBuf != nullptr, "createBuffer keep_tex_out returned null");

                BufferDesc sampCbDesc;
                sampCbDesc.setByteSize(sizeof(SamplerTestParams))
                    .setIsConstantBuffer(true)
                    .setInitialState(ResourceStates::ConstantBuffer)
                    .setKeepInitialState(true)
                    .setDebugName("sta_initial_keep_samp_cb");
                BufferHandle sampCb = dev->createBuffer(sampCbDesc);
                UAA_CHECK(sampCb != nullptr, "createBuffer keep_samp_cb returned null");

                BindingSetDesc copySetDesc;
                copySetDesc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf));
                BindingSetHandle copySet = dev->createBindingSet(copySetDesc, copyBundle.layout);
                UAA_CHECK(copySet != nullptr, "createBindingSet keep copySet returned null");

                BindingSetDesc sampSetDesc;
                sampSetDesc.addItem(BindingSetItem::Texture_SRV(0, tex))
                    .addItem(BindingSetItem::Sampler(1, sampBundle.sampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, texOutBuf))
                    .addItem(BindingSetItem::ConstantBuffer(3, sampCb));
                BindingSetHandle sampSet = dev->createBindingSet(sampSetDesc, sampBundle.layout);
                UAA_CHECK(sampSet != nullptr, "createBindingSet keep sampSet returned null");

                const CopyAddParams params{kCount, kAddVal, kXorVal, 0};
                SamplerTestParams sampParams{};
                sampParams.uvOrigin[0] = 0.5f / float(kTexSize);
                sampParams.uvOrigin[1] = 0.5f / float(kTexSize);
                sampParams.uvStep[0] = 1.0f / float(kTexSize);
                sampParams.uvStep[1] = 1.0f / float(kTexSize);
                sampParams.width = kTexSize;
                sampParams.height = kTexSize;

                const std::vector<uint8_t> texPixels = makeSolidRgba8(kTexSize, kTexSize, 64, 128, 192, 255);

                CommandListHandle cmd1 = dev->createCommandList();
                UAA_CHECK(cmd1 != nullptr, "createCommandList cmd1 returned null");
                cmd1->open();

                // Before first submission, an untouched texture with keepInitialState reports Common on tracked backends.
                checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(tex, 0, 0), ResourceStates::Common,
                    "initial unsubmitted texture state");
                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(inBuf), ResourceStates::Unknown,
                    "untouched buffer state before tracking");

                cmd1->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(inBuf), ResourceStates::CopyDest,
                    "buffer state after writeBuffer");

                cmd1->writeBuffer(cb, &params, sizeof(params));
                cmd1->writeBuffer(sampCb, &sampParams, sizeof(sampParams));
                cmd1->writeTexture(tex, 0, 0, texPixels.data(), size_t(kTexSize) * 4u);
                checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(tex, 0, 0), ResourceStates::CopyDest,
                    "texture state after writeTexture");

                cmd1->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(copySet));
                cmd1->dispatch(1, 1, 1);

                cmd1->setComputeState(ComputeState().setPipeline(sampBundle.pipeline).addBindingSet(sampSet));
                cmd1->dispatch(1, 1, 1);

                cmd1->close();
                fix.submitAndWait(cmd1);

                // On a subsequent command list, keepInitialState guarantees the texture starts in initialState.
                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "createCommandList cmd2 returned null");
                cmd2->open();
                checkExpectedTrackedState(ctx, api, cmd2->getTextureSubresourceState(tex, 0, 0), ResourceStates::ShaderResource,
                    "texture state on second command list after keepInitialState");
                cmd2->setTextureState(tex, AllSubresources, ResourceStates::UnorderedAccess);
                checkExpectedTrackedState(ctx, api, cmd2->getTextureSubresourceState(tex, 0, 0), ResourceStates::UnorderedAccess,
                    "texture state after explicit transition");
                cmd2->close();
                fix.submitAndWait(cmd2);

                std::vector<uint32_t> outWords(kCount, 0);
                fix.readbackBuffer(outBuf, outWords.data(), outWords.size() * sizeof(uint32_t));
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                    UAA_CHECK_EQ(outWords[i], expected, "STA.Initial keepInitialState buffer output mismatch");
                }

                std::vector<float> texOutFloats(size_t(kTexSize) * kTexSize * 4u, 0.0f);
                fix.readbackBuffer(texOutBuf, texOutFloats.data(), texOutFloats.size() * sizeof(float));
                UAA_CHECK_NEAR(texOutFloats[0], 64.0f / 255.0f, 1.0f / 255.0f, "STA.Initial keepInitialState tex R");
                UAA_CHECK_NEAR(texOutFloats[1], 128.0f / 255.0f, 1.0f / 255.0f, "STA.Initial keepInitialState tex G");
                UAA_CHECK_NEAR(texOutFloats[2], 192.0f / 255.0f, 1.0f / 255.0f, "STA.Initial keepInitialState tex B");
                UAA_CHECK_NEAR(texOutFloats[3], 1.0f, 1.0f / 255.0f, "STA.Initial keepInitialState tex A");
            }

            // Part B: keepInitialState = false with explicit beginTrackingTextureState / beginTrackingBufferState.
            {
                BufferDesc inDesc;
                inDesc.setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setKeepInitialState(false)
                    .setDebugName("sta_initial_manual_in");
                BufferHandle inBuf = dev->createBuffer(inDesc);
                UAA_CHECK(inBuf != nullptr, "createBuffer manual_in returned null");

                BufferDesc outDesc = inDesc;
                outDesc.setDebugName("sta_initial_manual_out");
                BufferHandle outBuf = dev->createBuffer(outDesc);
                UAA_CHECK(outBuf != nullptr, "createBuffer manual_out returned null");

                BufferDesc cbDesc;
                cbDesc.setByteSize(sizeof(CopyAddParams))
                    .setIsConstantBuffer(true)
                    .setKeepInitialState(false)
                    .setDebugName("sta_initial_manual_cb");
                BufferHandle cb = dev->createBuffer(cbDesc);
                UAA_CHECK(cb != nullptr, "createBuffer manual_cb returned null");

                TextureDesc texDesc;
                texDesc.setWidth(kTexSize)
                    .setHeight(kTexSize)
                    .setFormat(Format::RGBA8_UNORM)
                    .setIsUAV(true)
                    .setKeepInitialState(false)
                    .setDebugName("sta_initial_manual_tex");
                TextureHandle tex = dev->createTexture(texDesc);
                UAA_CHECK(tex != nullptr, "createTexture manual_tex returned null");

                BindingSetDesc copySetDesc;
                copySetDesc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf));
                BindingSetHandle copySet = dev->createBindingSet(copySetDesc, copyBundle.layout);
                UAA_CHECK(copySet != nullptr, "createBindingSet manual copySet returned null");

                const CopyAddParams params{kCount, kAddVal, kXorVal, 0};
                const std::vector<uint8_t> texPixels = makeSolidRgba8(kTexSize, kTexSize, 32, 96, 224, 255);

                CommandListHandle cmd1 = dev->createCommandList();
                UAA_CHECK(cmd1 != nullptr, "createCommandList manual cmd1 returned null");
                cmd1->open();

                checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(tex, 0, 0), ResourceStates::Unknown,
                    "manual texture state before beginTrackingTextureState");
                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(inBuf), ResourceStates::Unknown,
                    "manual buffer state before beginTrackingBufferState");

                cmd1->beginTrackingBufferState(inBuf, ResourceStates::Common);
                cmd1->beginTrackingBufferState(outBuf, ResourceStates::Common);
                cmd1->beginTrackingBufferState(cb, ResourceStates::Common);
                cmd1->beginTrackingTextureState(tex, AllSubresources, ResourceStates::Common);

                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(inBuf), ResourceStates::Common,
                    "manual buffer state after beginTrackingBufferState");
                checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(tex, 0, 0), ResourceStates::Common,
                    "manual texture state after beginTrackingTextureState");

                cmd1->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
                cmd1->writeBuffer(cb, &params, sizeof(params));
                cmd1->writeTexture(tex, 0, 0, texPixels.data(), size_t(kTexSize) * 4u);

                cmd1->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(copySet));
                cmd1->dispatch(1, 1, 1);
                cmd1->close();
                fix.submitAndWait(cmd1);

                // Resume tracking on a fresh command list from the states left by cmd1.
                BufferDesc rbDesc;
                rbDesc.setByteSize(kCount * sizeof(uint32_t))
                    .setCpuAccess(CpuAccessMode::Read)
                    .setDebugName("sta_initial_manual_rb");
                BufferHandle rbBuf = dev->createBuffer(rbDesc);
                UAA_CHECK(rbBuf != nullptr, "createBuffer manual_rb returned null");

                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "createCommandList manual cmd2 returned null");
                cmd2->open();
                cmd2->beginTrackingBufferState(outBuf, ResourceStates::UnorderedAccess);
                cmd2->beginTrackingTextureState(tex, AllSubresources, ResourceStates::CopyDest);
                cmd2->copyBuffer(rbBuf, 0, outBuf, 0, kCount * sizeof(uint32_t));
                cmd2->setTextureState(tex, AllSubresources, ResourceStates::CopySource);
                cmd2->close();
                fix.submitAndWait(cmd2);

                const void* mapped = dev->mapBuffer(rbBuf, CpuAccessMode::Read);
                UAA_CHECK(mapped != nullptr, "mapBuffer manual_rb returned null");
                const auto* outPtr = static_cast<const uint32_t*>(mapped);
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                    UAA_CHECK_EQ(outPtr[i], expected, "STA.Initial manual tracking buffer output mismatch");
                }
                dev->unmapBuffer(rbBuf);

                // Re-anchor texture tracking before readbackTexture2D uses it as CopySource.
                CommandListHandle cmd3 = dev->createCommandList();
                UAA_CHECK(cmd3 != nullptr, "createCommandList manual cmd3 returned null");
                cmd3->open();
                cmd3->beginTrackingTextureState(tex, AllSubresources, ResourceStates::CopySource);
                cmd3->close();
                fix.submitAndWait(cmd3);

                // Re-anchor inside a staging copy directly via beginTrackingTextureState.
                TextureDesc stgDesc;
                stgDesc.setWidth(kTexSize)
                    .setHeight(kTexSize)
                    .setFormat(Format::RGBA8_UNORM)
                    .setDebugName("sta_initial_manual_stg");
                StagingTextureHandle stg = dev->createStagingTexture(stgDesc, CpuAccessMode::Read);
                UAA_CHECK(stg != nullptr, "createStagingTexture returned null");

                CommandListHandle cmd4 = dev->createCommandList();
                UAA_CHECK(cmd4 != nullptr, "createCommandList manual cmd4 returned null");
                cmd4->open();
                cmd4->beginTrackingTextureState(tex, AllSubresources, ResourceStates::CopySource);
                cmd4->copyTexture(stg, TextureSlice().setMipLevel(0).setArraySlice(0), tex, TextureSlice().setMipLevel(0).setArraySlice(0));
                cmd4->close();
                fix.submitAndWait(cmd4);

                size_t rowPitch = 0;
                const void* texMapped = dev->mapStagingTexture(stg, TextureSlice(), CpuAccessMode::Read, &rowPitch);
                UAA_CHECK(texMapped != nullptr, "mapStagingTexture returned null");
                std::string pxDiff;
                const bool pxOk = TestFixture::checkPixelRgba8(static_cast<const uint8_t*>(texMapped), 32, 96, 224, 255, 0, pxDiff);
                dev->unmapStagingTexture(stg);
                UAA_CHECK(pxOk, "STA.Initial manual texture pixel mismatch: " + pxDiff);
            }
        }

        // Tests per-subresource tracking and transitions across a 2-slice, 3-mip 2D array texture.
        void testStatePerSubresource(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();
            const GraphicsAPI api = fix.api();

            constexpr uint32_t kWidth = 16;
            constexpr uint32_t kHeight = 16;
            constexpr uint32_t kSlices = 2;
            constexpr uint32_t kMips = 3;

            TextureDesc arrDesc;
            arrDesc.setDimension(TextureDimension::Texture2DArray)
                .setWidth(kWidth)
                .setHeight(kHeight)
                .setArraySize(kSlices)
                .setMipLevels(kMips)
                .setFormat(Format::RGBA8_UNORM)
                .setIsUAV(true)
                .setKeepInitialState(false)
                .setDebugName("sta_persubresource_arr");
            TextureHandle arrTex = dev->createTexture(arrDesc);
            UAA_CHECK(arrTex != nullptr, "createTexture Texture2DArray returned null");

            // Auxiliary source texture for copyTexture into selected subresources.
            TextureDesc srcDesc;
            srcDesc.setDimension(TextureDimension::Texture2D)
                .setWidth(kWidth)
                .setHeight(kHeight)
                .setMipLevels(kMips)
                .setFormat(Format::RGBA8_UNORM)
                .setInitialState(ResourceStates::CopySource)
                .setKeepInitialState(true)
                .setDebugName("sta_persubresource_src");
            TextureHandle srcTex = dev->createTexture(srcDesc);
            UAA_CHECK(srcTex != nullptr, "createTexture srcTex returned null");

            auto expectedColour = [](uint32_t slice, uint32_t mip) -> std::array<uint8_t, 4>
            {
                return {
                    static_cast<uint8_t>(40u + slice * 90u),
                    static_cast<uint8_t>(50u + mip * 60u),
                    static_cast<uint8_t>(100u + slice * 40u + mip * 25u),
                    255u
                };
            };

            // Populate srcTex mips 0 and 2 for copyTexture into slice 1.
            for (uint32_t mip : {0u, 2u})
            {
                const uint32_t mipW = std::max(1u, kWidth >> mip);
                const uint32_t mipH = std::max(1u, kHeight >> mip);
                const auto c = expectedColour(1u, mip);
                const std::vector<uint8_t> px = makeSolidRgba8(mipW, mipH, c[0], c[1], c[2], c[3]);
                fix.uploadTexture2D(srcTex, 0, mip, px.data(), size_t(mipW) * 4u);
            }

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();

            for (uint32_t slice = 0; slice < kSlices; ++slice)
            {
                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    cmd->beginTrackingTextureState(
                        arrTex,
                        TextureSubresourceSet(mip, 1, slice, 1),
                        ResourceStates::Common);
                }
            }

            // Transition only (mip=1, slice=0) to CopyDest and verify siblings remain unaffected.
            cmd->setTextureState(arrTex, TextureSubresourceSet(1, 1, 0, 1), ResourceStates::CopyDest);
            checkExpectedTrackedState(ctx, api, cmd->getTextureSubresourceState(arrTex, 0, 1), ResourceStates::CopyDest,
                "subresource (slice=0, mip=1) after transition");
            checkExpectedTrackedState(ctx, api, cmd->getTextureSubresourceState(arrTex, 0, 0), ResourceStates::Common,
                "sibling subresource (slice=0, mip=0) must remain Common");
            checkExpectedTrackedState(ctx, api, cmd->getTextureSubresourceState(arrTex, 1, 1), ResourceStates::Common,
                "sibling subresource (slice=1, mip=1) must remain Common");

            // Write directly to slice 0 (all mips) and (slice=1, mip=1), and copy into (slice=1, mip=0) and (slice=1, mip=2).
            for (uint32_t slice = 0; slice < kSlices; ++slice)
            {
                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    const uint32_t mipW = std::max(1u, kWidth >> mip);
                    const uint32_t mipH = std::max(1u, kHeight >> mip);
                    const auto c = expectedColour(slice, mip);

                    if (slice == 1u && (mip == 0u || mip == 2u))
                    {
                        cmd->copyTexture(
                            arrTex,
                            TextureSlice().setMipLevel(mip).setArraySlice(slice),
                            srcTex,
                            TextureSlice().setMipLevel(mip).setArraySlice(0));
                    }
                    else
                    {
                        const std::vector<uint8_t> px = makeSolidRgba8(mipW, mipH, c[0], c[1], c[2], c[3]);
                        cmd->writeTexture(arrTex, slice, mip, px.data(), size_t(mipW) * 4u);
                    }
                }
            }

            // Transition all subresources explicitly to CopySource for verification readback.
            for (uint32_t slice = 0; slice < kSlices; ++slice)
            {
                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    cmd->setTextureState(arrTex, TextureSubresourceSet(mip, 1, slice, 1), ResourceStates::CopySource);
                }
            }

            cmd->close();
            fix.submitAndWait(cmd);

            // Read back every subresource using staging texture copy with explicit beginTrackingTextureState.
            TextureDesc stgDesc;
            stgDesc.setWidth(kWidth)
                .setHeight(kHeight)
                .setMipLevels(kMips)
                .setFormat(Format::RGBA8_UNORM)
                .setDebugName("sta_persubresource_stg");
            StagingTextureHandle stg = dev->createStagingTexture(stgDesc, CpuAccessMode::Read);
            UAA_CHECK(stg != nullptr, "createStagingTexture returned null");

            CommandListHandle rbCmd = dev->createCommandList();
            UAA_CHECK(rbCmd != nullptr, "createCommandList rbCmd returned null");
            rbCmd->open();
            for (uint32_t slice = 0; slice < kSlices; ++slice)
            {
                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    rbCmd->beginTrackingTextureState(arrTex, TextureSubresourceSet(mip, 1, slice, 1), ResourceStates::CopySource);
                    rbCmd->copyTexture(
                        stg,
                        TextureSlice().setMipLevel(mip).setArraySlice(0),
                        arrTex,
                        TextureSlice().setMipLevel(mip).setArraySlice(slice));
                }
                rbCmd->close();
                fix.submitAndWait(rbCmd);

                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    const uint32_t mipW = std::max(1u, kWidth >> mip);
                    const uint32_t mipH = std::max(1u, kHeight >> mip);
                    const size_t tightRow = size_t(mipW) * 4u;
                    size_t rowPitch = 0;
                    const void* mapped = dev->mapStagingTexture(stg, TextureSlice().setMipLevel(mip).setArraySlice(0), CpuAccessMode::Read, &rowPitch);
                    UAA_CHECK(mapped != nullptr, "mapStagingTexture returned null");
                    UAA_CHECK(rowPitch >= tightRow, "staging rowPitch smaller than tight row");

                    std::vector<uint8_t> tight(tightRow * mipH);
                    const auto* srcBytes = static_cast<const uint8_t*>(mapped);
                    for (uint32_t r = 0; r < mipH; ++r)
                    {
                        std::memcpy(tight.data() + size_t(r) * tightRow, srcBytes + size_t(r) * rowPitch, tightRow);
                    }
                    dev->unmapStagingTexture(stg);

                    const auto c = expectedColour(slice, mip);
                    verifySolidRgba8(ctx, tight, mipW, mipH, c[0], c[1], c[2], c[3], "STA.PerSubresource");
                }

                if (slice + 1u < kSlices)
                {
                    rbCmd->open();
                }
            }
        }

        // Tests setResourceStatesForBindingSet, setResourceStatesForFramebuffer, automatic barriers, and explicit set*State.
        void testStateBindingTransitions(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();
            const GraphicsAPI api = fix.api();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 7;
            constexpr uint32_t kXorVal = 0x13579BDFu;
            constexpr uint32_t kRtSize = 16;

            const CopyAddPipelineBundle copyBundle = createCopyAddBundle(ctx, fix);

            BufferDesc inDesc;
            inDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setKeepInitialState(false)
                .setDebugName("sta_binding_in");
            BufferHandle inBuf = dev->createBuffer(inDesc);
            UAA_CHECK(inBuf != nullptr, "createBuffer inBuf returned null");

            BufferDesc midDesc = inDesc;
            midDesc.setDebugName("sta_binding_mid");
            BufferHandle midBuf = dev->createBuffer(midDesc);
            UAA_CHECK(midBuf != nullptr, "createBuffer midBuf returned null");

            BufferDesc dstDesc = inDesc;
            dstDesc.setDebugName("sta_binding_dst");
            BufferHandle dstBuf = dev->createBuffer(dstDesc);
            UAA_CHECK(dstBuf != nullptr, "createBuffer dstBuf returned null");

            BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(CopyAddParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("sta_binding_cb");
            BufferHandle cb = dev->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "createBuffer cb returned null");

            BindingSetDesc copySetDesc;
            copySetDesc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, midBuf));
            BindingSetHandle copySet = dev->createBindingSet(copySetDesc, copyBundle.layout);
            UAA_CHECK(copySet != nullptr, "createBindingSet returned null");

            TextureDesc rtDesc;
            rtDesc.setWidth(kRtSize)
                .setHeight(kRtSize)
                .setFormat(Format::RGBA8_UNORM)
                .setIsRenderTarget(true)
                .setInitialState(ResourceStates::RenderTarget)
                .setKeepInitialState(true)
                .setDebugName("sta_binding_rt");
            TextureHandle rtTex = dev->createTexture(rtDesc);
            UAA_CHECK(rtTex != nullptr, "createTexture rtTex returned null");

            const bool hasDepth = (dev->queryFormatSupport(Format::D32) & FormatSupport::DepthStencil) != 0;
            TextureHandle depthTex;
            if (hasDepth)
            {
                TextureDesc dDesc;
                dDesc.setWidth(kRtSize)
                    .setHeight(kRtSize)
                    .setFormat(Format::D32)
                    .setIsRenderTarget(true)
                    .setInitialState(ResourceStates::DepthWrite)
                    .setKeepInitialState(true)
                    .setDebugName("sta_binding_depth");
                depthTex = dev->createTexture(dDesc);
                UAA_CHECK(depthTex != nullptr, "createTexture depthTex returned null");
            }

            FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(rtTex);
            if (depthTex)
            {
                fbDesc.setDepthAttachment(depthTex);
            }
            FramebufferHandle fb = dev->createFramebuffer(fbDesc);
            UAA_CHECK(fb != nullptr, "createFramebuffer returned null");

            ShaderHandle vs = fix.createShader("vs_fullscreen_tri", ShaderType::Vertex);
            ShaderHandle ps = fix.createShader("ps_solid_color", ShaderType::Pixel);
            UAA_CHECK(vs != nullptr && ps != nullptr, "graphics shaders must not be null");

            BindingLayoutDesc gfxLayoutDesc = fix.makeBindingLayoutDesc(ShaderType::AllGraphics)
                .addItem(BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC)));
            BindingLayoutHandle gfxLayout = dev->createBindingLayout(gfxLayoutDesc);
            UAA_CHECK(gfxLayout != nullptr, "createBindingLayout gfxLayout returned null");

            BindingSetDesc gfxSetDesc;
            gfxSetDesc.addItem(BindingSetItem::PushConstants(0, sizeof(FullscreenPC)));
            BindingSetHandle gfxSet = dev->createBindingSet(gfxSetDesc, gfxLayout);
            UAA_CHECK(gfxSet != nullptr, "createBindingSet gfxSet returned null");

            GraphicsPipelineDesc gfxPsoDesc;
            gfxPsoDesc.setVertexShader(vs)
                .setPixelShader(ps)
                .addBindingLayout(gfxLayout);
            gfxPsoDesc.renderState.rasterState.setCullMode(RasterCullMode::None);
            gfxPsoDesc.renderState.depthStencilState.setDepthTestEnable(false).setDepthWriteEnable(false);
            GraphicsPipelineHandle gfxPso = dev->createGraphicsPipeline(gfxPsoDesc, fb->getFramebufferInfo());
            UAA_CHECK(gfxPso != nullptr, "createGraphicsPipeline returned null");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x202u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0};

            // Pass 1: explicitly verify setResourceStatesForBindingSet and setResourceStatesForFramebuffer.
            {
                CommandListHandle cmd1 = dev->createCommandList();
                UAA_CHECK(cmd1 != nullptr, "createCommandList cmd1 returned null");
                cmd1->open();

                cmd1->beginTrackingBufferState(inBuf, ResourceStates::Common);
                cmd1->beginTrackingBufferState(midBuf, ResourceStates::Common);
                cmd1->setResourceStatesForBindingSet(copySet);
                cmd1->setResourceStatesForFramebuffer(fb);
                cmd1->commitBarriers();

                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(inBuf), ResourceStates::NonPixelShaderResource,
                    "setResourceStatesForBindingSet SRV state");
                checkExpectedTrackedState(ctx, api, cmd1->getBufferState(midBuf), ResourceStates::UnorderedAccess,
                    "setResourceStatesForBindingSet UAV state");
                checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(rtTex, 0, 0), ResourceStates::RenderTarget,
                    "setResourceStatesForFramebuffer colour state");
                if (depthTex)
                {
                    checkExpectedTrackedState(ctx, api, cmd1->getTextureSubresourceState(depthTex, 0, 0), ResourceStates::DepthWrite,
                        "setResourceStatesForFramebuffer depth state");
                }

                cmd1->close();
                fix.submitAndWait(cmd1);
            }

            // Pass 2: automatic barriers across writeBuffer -> dispatch -> copyBuffer, plus graphics draw and explicit set*State.
            {
                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "createCommandList cmd2 returned null");
                cmd2->open();

                cmd2->beginTrackingBufferState(inBuf, ResourceStates::NonPixelShaderResource);
                cmd2->beginTrackingBufferState(midBuf, ResourceStates::UnorderedAccess);
                cmd2->beginTrackingBufferState(dstBuf, ResourceStates::Common);

                cmd2->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
                cmd2->writeBuffer(cb, &params, sizeof(params));

                cmd2->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(copySet));
                cmd2->dispatch(1, 1, 1);

                cmd2->copyBuffer(dstBuf, 0, midBuf, 0, kCount * sizeof(uint32_t));
                checkExpectedTrackedState(ctx, api, cmd2->getBufferState(midBuf), ResourceStates::CopySource,
                    "midBuf state after copyBuffer");
                checkExpectedTrackedState(ctx, api, cmd2->getBufferState(dstBuf), ResourceStates::CopyDest,
                    "dstBuf state after copyBuffer");

                cmd2->setBufferState(dstBuf, ResourceStates::CopySource);
                checkExpectedTrackedState(ctx, api, cmd2->getBufferState(dstBuf), ResourceStates::CopySource,
                    "dstBuf state after explicit setBufferState");

                utils::ClearColorAttachment(cmd2, fb, 0, Color(0.0f, 0.0f, 0.0f, 1.0f));
                FullscreenPC pc{};
                pc.color[0] = 0.25f;
                pc.color[1] = 0.75f;
                pc.color[2] = 0.50f;
                pc.color[3] = 1.00f;
                pc.depth = 0.5f;
                pc.scale = 1.0f;

                GraphicsState gfxState;
                gfxState.setPipeline(gfxPso)
                    .setFramebuffer(fb)
                    .setViewport(ViewportState().addViewportAndScissorRect(Viewport(float(kRtSize), float(kRtSize))))
                    .addBindingSet(gfxSet);
                cmd2->setGraphicsState(gfxState);
                cmd2->setPushConstants(&pc, sizeof(pc));
                cmd2->draw(DrawArguments().setVertexCount(3));

                cmd2->setTextureState(rtTex, AllSubresources, ResourceStates::ShaderResource);
                checkExpectedTrackedState(ctx, api, cmd2->getTextureSubresourceState(rtTex, 0, 0), ResourceStates::ShaderResource,
                    "rtTex state after explicit setTextureState");
                cmd2->commitBarriers();

                cmd2->close();
                fix.submitAndWait(cmd2);
            }

            // Re-anchor dstBuf to Common before readbackBuffer uses CopySource.
            BufferDesc rbDesc;
            rbDesc.setByteSize(kCount * sizeof(uint32_t))
                .setCpuAccess(CpuAccessMode::Read)
                .setDebugName("sta_binding_rb");
            BufferHandle rbBuf = dev->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "createBuffer rbBuf returned null");

            CommandListHandle rbCmd = dev->createCommandList();
            UAA_CHECK(rbCmd != nullptr, "createCommandList rbCmd returned null");
            rbCmd->open();
            rbCmd->beginTrackingBufferState(dstBuf, ResourceStates::CopySource);
            rbCmd->copyBuffer(rbBuf, 0, dstBuf, 0, kCount * sizeof(uint32_t));
            rbCmd->close();
            fix.submitAndWait(rbCmd);

            const void* mapped = dev->mapBuffer(rbBuf, CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer rbBuf returned null");
            const auto* outPtr = static_cast<const uint32_t*>(mapped);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(outPtr[i], expected, "STA.BindingTransitions output buffer mismatch");
            }
            dev->unmapBuffer(rbBuf);

            std::vector<uint8_t> rtPixels;
            fix.readbackTexture2D(rtTex, 0, 0, rtPixels);
            UAA_CHECK_EQ(rtPixels.size(), size_t(kRtSize) * kRtSize * 4u, "render target readback size");
            std::string pxDiff;
            const bool pxOk = TestFixture::checkPixelRgba8(rtPixels.data(), 64, 191, 128, 255, 2, pxDiff);
            UAA_CHECK(pxOk, "STA.BindingTransitions render target pixel mismatch: " + pxDiff);
        }

        // Tests UAV barriers toggled between true, false, and true with explicit setBufferState across 4 chained dispatches.
        void testStateUav(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;

            ShaderHandle shader = fix.createShader("cs_uav_chain", ShaderType::Compute);
            UAA_CHECK(shader != nullptr, "cs_uav_chain shader must not be null");

            BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(ShaderType::Compute)
                .addItem(BindingLayoutItem::PushConstants(0, sizeof(UavChainPC)))
                .addItem(BindingLayoutItem::StructuredBuffer_UAV(0));
            BindingLayoutHandle layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "createBindingLayout for cs_uav_chain returned null");

            ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(shader)
                .addBindingLayout(layout);
            ComputePipelineHandle pso = dev->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "createComputePipeline for cs_uav_chain returned null");

            BufferDesc bufDesc;
            bufDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("sta_uav_buf");
            BufferHandle buf = dev->createBuffer(bufDesc);
            UAA_CHECK(buf != nullptr, "createBuffer sta_uav_buf returned null");

            BindingSetDesc setDesc;
            setDesc.addItem(BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                .addItem(BindingSetItem::StructuredBuffer_UAV(0, buf));
            BindingSetHandle set = dev->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "createBindingSet for cs_uav_chain returned null");

            std::vector<uint32_t> initialWords(kCount);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                initialWords[i] = i + 1u;
            }

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeBuffer(buf, initialWords.data(), initialWords.size() * sizeof(uint32_t));

            const ComputeState computeState = ComputeState().setPipeline(pso).addBindingSet(set);

            // Dispatch 1 with default UAV barriers enabled.
            cmd->setEnableUavBarriersForBuffer(buf, true);
            cmd->setComputeState(computeState);
            const UavChainPC pc1{kCount, 3u, 2u, 0u};
            cmd->setPushConstants(&pc1, sizeof(pc1));
            cmd->dispatch(1, 1, 1);

            // Dispatch 2 with automatic UAV barriers enabled.
            cmd->setComputeState(computeState);
            const UavChainPC pc2{kCount, 5u, 3u, 0u};
            cmd->setPushConstants(&pc2, sizeof(pc2));
            cmd->dispatch(1, 1, 1);

            // Dispatch 3 after disabling automatic UAV barriers (first barrier after reset is still placed).
            cmd->setEnableUavBarriersForBuffer(buf, false);
            cmd->setComputeState(computeState);
            const UavChainPC pc3{kCount, 7u, 5u, 0u};
            cmd->setPushConstants(&pc3, sizeof(pc3));
            cmd->dispatch(1, 1, 1);

            // Re-enable UAV barriers and place an explicit UnorderedAccess barrier before Dispatch 4.
            cmd->setEnableUavBarriersForBuffer(buf, true);
            cmd->setBufferState(buf, ResourceStates::UnorderedAccess);
            cmd->commitBarriers();
            cmd->setComputeState(computeState);
            const UavChainPC pc4{kCount, 11u, 7u, 0u};
            cmd->setPushConstants(&pc4, sizeof(pc4));
            cmd->dispatch(1, 1, 1);

            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kCount, 0);
            fix.readbackBuffer(buf, actual.data(), actual.size() * sizeof(uint32_t));
            for (uint32_t i = 0; i < kCount; ++i)
            {
                uint32_t expected = i + 1u;
                expected = expected * 2u + 3u;
                expected = expected * 3u + 5u;
                expected = expected * 5u + 7u;
                expected = expected * 7u + 11u;
                UAA_CHECK_EQ(actual[i], expected, "STA.Uav chained value mismatch at index " + std::to_string(i));
            }
        }

        // Tests permanent buffer and texture states, compatible usage, and conflict diagnostics.
        void testStatePermanent(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();
            const GraphicsAPI api = fix.api();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kTexSize = 8;
            constexpr uint32_t kAddVal = 19;
            constexpr uint32_t kXorVal = 0x2468ACE0u;

            const CopyAddPipelineBundle copyBundle = createCopyAddBundle(ctx, fix);
            const SamplerPipelineBundle sampBundle = createSamplerBundle(ctx, fix);

            BufferDesc inDesc;
            inDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setInitialState(ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_in");
            BufferHandle inBuf = dev->createBuffer(inDesc);
            UAA_CHECK(inBuf != nullptr, "createBuffer perm_in returned null");

            BufferDesc outDesc;
            outDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_out");
            BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "createBuffer perm_out returned null");

            BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(CopyAddParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_cb");
            BufferHandle cb = dev->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "createBuffer perm_cb returned null");

            TextureDesc texDesc;
            texDesc.setWidth(kTexSize)
                .setHeight(kTexSize)
                .setFormat(Format::RGBA8_UNORM)
                .setInitialState(ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_tex");
            TextureHandle tex = dev->createTexture(texDesc);
            UAA_CHECK(tex != nullptr, "createTexture perm_tex returned null");

            BufferDesc texOutDesc;
            texOutDesc.setByteSize(size_t(kTexSize) * kTexSize * 16u)
                .setStructStride(16u)
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_tex_out");
            BufferHandle texOutBuf = dev->createBuffer(texOutDesc);
            UAA_CHECK(texOutBuf != nullptr, "createBuffer perm_tex_out returned null");

            BufferDesc sampCbDesc;
            sampCbDesc.setByteSize(sizeof(SamplerTestParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("sta_perm_samp_cb");
            BufferHandle sampCb = dev->createBuffer(sampCbDesc);
            UAA_CHECK(sampCb != nullptr, "createBuffer perm_samp_cb returned null");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x303u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0};
            SamplerTestParams sampParams{};
            sampParams.uvOrigin[0] = 0.5f / float(kTexSize);
            sampParams.uvOrigin[1] = 0.5f / float(kTexSize);
            sampParams.uvStep[0] = 1.0f / float(kTexSize);
            sampParams.uvStep[1] = 1.0f / float(kTexSize);
            sampParams.width = kTexSize;
            sampParams.height = kTexSize;
            const std::vector<uint8_t> texPixels = makeSolidRgba8(kTexSize, kTexSize, 192, 64, 32, 255);

            // Upload data and pin both resources to ShaderResource at submission.
            {
                CommandListHandle cmd1 = dev->createCommandList();
                UAA_CHECK(cmd1 != nullptr, "createCommandList perm cmd1 returned null");
                cmd1->open();
                cmd1->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
                cmd1->writeBuffer(cb, &params, sizeof(params));
                cmd1->writeTexture(tex, 0, 0, texPixels.data(), size_t(kTexSize) * 4u);
                cmd1->writeBuffer(sampCb, &sampParams, sizeof(sampParams));
                cmd1->setPermanentBufferState(inBuf, ResourceStates::ShaderResource);
                cmd1->setPermanentTextureState(tex, ResourceStates::ShaderResource);
                cmd1->close();
                fix.submitAndWait(cmd1);
            }

            // Use the permanently pinned resources in compatible SRV bindings.
            {
                BindingSetDesc copySetDesc;
                copySetDesc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf));
                BindingSetHandle copySet = dev->createBindingSet(copySetDesc, copyBundle.layout);
                UAA_CHECK(copySet != nullptr, "createBindingSet perm copySet returned null");

                BindingSetDesc sampSetDesc;
                sampSetDesc.addItem(BindingSetItem::Texture_SRV(0, tex))
                    .addItem(BindingSetItem::Sampler(1, sampBundle.sampler))
                    .addItem(BindingSetItem::StructuredBuffer_UAV(2, texOutBuf))
                    .addItem(BindingSetItem::ConstantBuffer(3, sampCb));
                BindingSetHandle sampSet = dev->createBindingSet(sampSetDesc, sampBundle.layout);
                UAA_CHECK(sampSet != nullptr, "createBindingSet perm sampSet returned null");

                CommandListHandle cmd2 = dev->createCommandList();
                UAA_CHECK(cmd2 != nullptr, "createCommandList perm cmd2 returned null");
                cmd2->open();
                cmd2->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(copySet));
                cmd2->dispatch(1, 1, 1);
                cmd2->setComputeState(ComputeState().setPipeline(sampBundle.pipeline).addBindingSet(sampSet));
                cmd2->dispatch(1, 1, 1);
                cmd2->close();
                fix.submitAndWait(cmd2);

                std::vector<uint32_t> actual(kCount, 0);
                fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                    UAA_CHECK_EQ(actual[i], expected, "STA.Permanent SRV compute output mismatch");
                }

                std::vector<float> texOutFloats(size_t(kTexSize) * kTexSize * 4u, 0.0f);
                fix.readbackBuffer(texOutBuf, texOutFloats.data(), texOutFloats.size() * sizeof(float));
                UAA_CHECK_NEAR(texOutFloats[0], 192.0f / 255.0f, 1.0f / 255.0f, "STA.Permanent tex R");
                UAA_CHECK_NEAR(texOutFloats[1], 64.0f / 255.0f, 1.0f / 255.0f, "STA.Permanent tex G");
                UAA_CHECK_NEAR(texOutFloats[2], 32.0f / 255.0f, 1.0f / 255.0f, "STA.Permanent tex B");
                UAA_CHECK_NEAR(texOutFloats[3], 1.0f, 1.0f / 255.0f, "STA.Permanent tex A");
            }

            // Conflicting use and conflicting permanent-state re-pinning under ExpectErrorsGuard.
            if (api != GraphicsAPI::D3D11)
            {
                ExpectErrorsGuard guard(fix);
                CommandListHandle cmd3 = dev->createCommandList();
                UAA_CHECK(cmd3 != nullptr, "createCommandList perm cmd3 returned null");
                cmd3->open();
                if (tracksExplicitStates(api))
                {
                    cmd3->setBufferState(inBuf, ResourceStates::CopyDest);
                    cmd3->setTextureState(tex, AllSubresources, ResourceStates::CopyDest);
                }
                cmd3->setPermanentBufferState(inBuf, ResourceStates::CopyDest);
                cmd3->setPermanentTextureState(tex, ResourceStates::CopyDest);
                cmd3->close();
                fix.submitAndWait(cmd3);
                UAA_CHECK(guard.errorCount() > 0, "Expected diagnostic error when changing or violating permanent state");
            }
        }

        // Tests empty, repeated, and cross-list commitBarriers behaviour.
        void testStateCommit(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 3;
            constexpr uint32_t kXorVal = 0x0F0F0F0Fu;

            const CopyAddPipelineBundle copyBundle = createCopyAddBundle(ctx, fix);

            BufferDesc bufDesc;
            bufDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setInitialState(ResourceStates::Common)
                .setKeepInitialState(true)
                .setDebugName("sta_commit_buf");
            BufferHandle inBuf = dev->createBuffer(bufDesc);
            BufferHandle midBuf = dev->createBuffer(bufDesc);
            BufferHandle outBuf = dev->createBuffer(bufDesc);
            UAA_CHECK(inBuf != nullptr && midBuf != nullptr && outBuf != nullptr, "createBuffer returned null");

            BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(CopyAddParams))
                .setIsConstantBuffer(true)
                .setInitialState(ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("sta_commit_cb");
            BufferHandle cb = dev->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "createBuffer cb returned null");

            BindingSetDesc set1Desc;
            set1Desc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, midBuf));
            BindingSetHandle set1 = dev->createBindingSet(set1Desc, copyBundle.layout);
            UAA_CHECK(set1 != nullptr, "createBindingSet set1 returned null");

            BindingSetDesc set2Desc;
            set2Desc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, midBuf))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf));
            BindingSetHandle set2 = dev->createBindingSet(set2Desc, copyBundle.layout);
            UAA_CHECK(set2 != nullptr, "createBindingSet set2 returned null");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x404u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0};

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");

            // First recording: empty commitBarriers on fresh list, repeated commitBarriers after transitions, submit.
            cmd->open();
            cmd->commitBarriers();
            cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            cmd->writeBuffer(cb, &params, sizeof(params));
            cmd->setBufferState(inBuf, ResourceStates::NonPixelShaderResource);
            cmd->setBufferState(midBuf, ResourceStates::UnorderedAccess);
            cmd->commitBarriers();
            cmd->commitBarriers();
            cmd->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(set1));
            cmd->dispatch(1, 1, 1);
            cmd->commitBarriers();
            cmd->close();
            fix.submitAndWait(cmd);

            // Second recording on reopened command list across list boundary.
            cmd->open();
            cmd->commitBarriers();
            cmd->setBufferState(midBuf, ResourceStates::NonPixelShaderResource);
            cmd->setBufferState(outBuf, ResourceStates::UnorderedAccess);
            cmd->commitBarriers();
            cmd->commitBarriers();
            cmd->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(set2));
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kCount, 0);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t step1 = (inputWords[i] + kAddVal) ^ kXorVal;
                const uint32_t step2 = (step1 + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(actual[i], step2, "STA.Commit output mismatch at index " + std::to_string(i));
            }
        }

        // Tests acceleration-structure state transitions when ray tracing is supported.
        void testStateRayStructures(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            if (!fix.features().supports(Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct is not supported by this backend");
            }

            IDevice* dev = fix.device();

            const float vertices[9] = {
                0.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f
            };
            BufferDesc vbDesc;
            vbDesc.setByteSize(sizeof(vertices))
                .setIsAccelStructBuildInput(true)
                .setInitialState(ResourceStates::AccelStructBuildInput)
                .setKeepInitialState(true)
                .setDebugName("sta_rt_vb");
            BufferHandle vb = dev->createBuffer(vbDesc);
            UAA_CHECK(vb != nullptr, "createBuffer sta_rt_vb returned null");
            fix.uploadBuffer(vb, vertices, sizeof(vertices));

            rt::GeometryTriangles tri;
            tri.setVertexBuffer(vb)
                .setVertexFormat(Format::RGB32_FLOAT)
                .setVertexCount(3)
                .setVertexStride(3 * sizeof(float));

            rt::GeometryDesc geomDesc;
            geomDesc.setTriangles(tri)
                .setFlags(rt::GeometryFlags::Opaque);

            rt::AccelStructDesc blasDesc;
            blasDesc.setIsTopLevel(false)
                .addBottomLevelGeometry(geomDesc)
                .setDebugName("sta_rt_blas");
            rt::AccelStructHandle blas = dev->createAccelStruct(blasDesc);
            UAA_CHECK(blas != nullptr, "createAccelStruct BLAS returned null");

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->buildBottomLevelAccelStruct(blas, &geomDesc, 1);
            cmd->setAccelStructState(blas, ResourceStates::AccelStructRead);
            cmd->commitBarriers();
            cmd->close();
            fix.submitAndWait(cmd);
        }

        // Tests full pipeline execution with automatic barriers disabled and complete manual transitions.
        void testStateDisabled(TestContext& ctx)
        {
            TestFixture& fix = *ctx.fixture;
            IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr uint32_t kAddVal = 13;
            constexpr uint32_t kXorVal = 0x55AA33CCu;
            constexpr uint32_t kTexSize = 8;

            const CopyAddPipelineBundle copyBundle = createCopyAddBundle(ctx, fix);

            BufferDesc inDesc;
            inDesc.setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setKeepInitialState(false)
                .setDebugName("sta_disabled_in");
            BufferHandle inBuf = dev->createBuffer(inDesc);
            UAA_CHECK(inBuf != nullptr, "createBuffer inBuf returned null");

            BufferDesc outDesc = inDesc;
            outDesc.setDebugName("sta_disabled_out");
            BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "createBuffer outBuf returned null");

            BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(CopyAddParams))
                .setIsConstantBuffer(true)
                .setKeepInitialState(false)
                .setDebugName("sta_disabled_cb");
            BufferHandle cb = dev->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "createBuffer cb returned null");

            BufferDesc rbDesc;
            rbDesc.setByteSize(kCount * sizeof(uint32_t))
                .setCpuAccess(CpuAccessMode::Read)
                .setDebugName("sta_disabled_rb");
            BufferHandle rbBuf = dev->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "createBuffer rbBuf returned null");

            TextureDesc texDesc;
            texDesc.setWidth(kTexSize)
                .setHeight(kTexSize)
                .setFormat(Format::RGBA8_UNORM)
                .setKeepInitialState(false)
                .setDebugName("sta_disabled_tex");
            TextureHandle tex = dev->createTexture(texDesc);
            UAA_CHECK(tex != nullptr, "createTexture tex returned null");

            BindingSetDesc setDesc;
            setDesc.addItem(BindingSetItem::ConstantBuffer(0, cb))
                .addItem(BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(BindingSetItem::StructuredBuffer_UAV(2, outBuf));
            BindingSetHandle set = dev->createBindingSet(setDesc, copyBundle.layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x505u);
            const CopyAddParams params{kCount, kAddVal, kXorVal, 0};
            const std::vector<uint8_t> texPixels = makeSolidRgba8(kTexSize, kTexSize, 10, 20, 30, 255);

            CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->setEnableAutomaticBarriers(false);

            cmd->beginTrackingBufferState(inBuf, ResourceStates::Common);
            cmd->beginTrackingBufferState(outBuf, ResourceStates::Common);
            cmd->beginTrackingBufferState(cb, ResourceStates::Common);
            cmd->beginTrackingTextureState(tex, AllSubresources, ResourceStates::Common);

            // Explicit CopyDest barriers before uploads.
            cmd->setBufferState(inBuf, ResourceStates::CopyDest);
            cmd->setBufferState(cb, ResourceStates::CopyDest);
            cmd->setTextureState(tex, AllSubresources, ResourceStates::CopyDest);
            cmd->commitBarriers();

            cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            cmd->writeBuffer(cb, &params, sizeof(params));
            cmd->writeTexture(tex, 0, 0, texPixels.data(), size_t(kTexSize) * 4u);

            // Explicit shader resource and UAV barriers before compute dispatch.
            cmd->setBufferState(inBuf, ResourceStates::NonPixelShaderResource);
            cmd->setBufferState(cb, ResourceStates::ConstantBuffer);
            cmd->setBufferState(outBuf, ResourceStates::UnorderedAccess);
            cmd->setTextureState(tex, AllSubresources, ResourceStates::ShaderResource);
            cmd->commitBarriers();

            cmd->setComputeState(ComputeState().setPipeline(copyBundle.pipeline).addBindingSet(set));
            cmd->dispatch(1, 1, 1);

            // Explicit CopySource barrier before copying to readback buffer.
            cmd->setBufferState(outBuf, ResourceStates::CopySource);
            cmd->commitBarriers();

            cmd->copyBuffer(rbBuf, 0, outBuf, 0, kCount * sizeof(uint32_t));
            cmd->close();
            fix.submitAndWait(cmd);

            const void* mapped = dev->mapBuffer(rbBuf, CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer rbBuf returned null");
            const auto* actual = static_cast<const uint32_t*>(mapped);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(actual[i], expected, "STA.Disabled output mismatch at index " + std::to_string(i));
            }
            dev->unmapBuffer(rbBuf);
        }
    } // namespace

    UAA_REGISTER_TEST("STA.Initial", "core", testStateInitial);
    UAA_REGISTER_TEST("STA.PerSubresource", "core", testStatePerSubresource);
    UAA_REGISTER_TEST("STA.BindingTransitions", "core", testStateBindingTransitions);
    UAA_REGISTER_TEST("STA.Uav", "core", testStateUav);
    UAA_REGISTER_TEST("STA.Permanent", "core", testStatePermanent);
    UAA_REGISTER_TEST("STA.Commit", "core", testStateCommit);
    UAA_REGISTER_TEST("STA.RayStructures", "core", testStateRayStructures);
    UAA_REGISTER_TEST("STA.Disabled", "core", testStateDisabled);
} // namespace nvrhi::uaa
