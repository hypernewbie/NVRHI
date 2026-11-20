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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct SamplerTestParams
        {
            float uvOrigin[2] = { 0.5f, 0.5f };
            float uvStep[2] = { 0.f, 0.f };
            float lod = 0.0f;
            uint32_t width = 1;
            uint32_t height = 1;
            uint32_t mode = 0;
        };

        struct SamplerCompareParams
        {
            float refDepth = 0.5f;
            float uStart = 0.25f;
            float uStep = 0.5f;
            float vCoord = 0.5f;
            uint32_t count = 2;
            uint32_t p0 = 0;
            uint32_t p1 = 0;
            uint32_t p2 = 0;
        };

        struct FullscreenPC
        {
            float color[4] = { 0.f, 0.f, 0.f, 1.f };
            float depth = 0.5f;
            float scale = 1.0f;
            float offset[2] = { 0.f, 0.f };
        };

        TestFixture& requireFixture(TestContext& ctx)
        {
            if (!ctx.fixture || !ctx.fixture->device())
            {
                ctx.skip("GPU fixture is not initialised");
            }
            return *ctx.fixture;
        }

        bool hasExtension(const DeviceInfo& info, const std::string& ext)
        {
            return std::find(info.enabledExtensions.begin(), info.enabledExtensions.end(), ext) != info.enabledExtensions.end();
        }

        void uploadSolidMip(const TestFixture& fix, nvrhi::ITexture* tex, uint32_t mip, uint32_t w, uint32_t h,
            uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            std::vector<uint8_t> pixels(size_t(w) * h * 4u);
            for (size_t i = 0; i < size_t(w) * h; ++i)
            {
                pixels[i * 4 + 0] = r;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = b;
                pixels[i * 4 + 3] = a;
            }
            fix.uploadTexture2D(tex, 0, mip, pixels.data(), size_t(w) * 4u);
        }

        // Tests point vs linear min/mag/mip filtering across explicit mip levels and fractional texel coordinates.
        void testSmpFilters(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle mipTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(16).setHeight(16).setMipLevels(5)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_mip_tex"));
            UAA_CHECK(mipTex != nullptr, "Failed to create multi-mip texture");

            uploadSolidMip(fix, mipTex, 0, 16, 16, 255,   0,   0, 255); // Red
            uploadSolidMip(fix, mipTex, 1,  8,  8,   0, 255,   0, 255); // Green
            uploadSolidMip(fix, mipTex, 2,  4,  4,   0,   0, 255, 255); // Blue
            uploadSolidMip(fix, mipTex, 3,  2,  2, 255, 255,   0, 255); // Yellow
            uploadSolidMip(fix, mipTex, 4,  1,  1, 255,   0, 255, 255); // Magenta

            nvrhi::TextureHandle splitTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(2).setHeight(2).setMipLevels(1)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_split_tex"));
            UAA_CHECK(splitTex != nullptr, "Failed to create 2x2 split texture");

            const uint8_t splitPixels[2 * 2 * 4] = {
                255, 0, 0, 255,   0, 255, 0, 255,
                255, 0, 0, 255,   0, 255, 0, 255
            };
            fix.uploadTexture2D(splitTex, 0, 0, splitPixels, 8u);

            nvrhi::SamplerHandle pointSamp = dev->createSampler(
                nvrhi::SamplerDesc().setAllFilters(false).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
            nvrhi::SamplerHandle linearSamp = dev->createSampler(
                nvrhi::SamplerDesc().setAllFilters(true).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(16).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_filter_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_filter_cb"));
            UAA_CHECK(pointSamp != nullptr && linearSamp != nullptr && outBuf != nullptr && cb != nullptr,
                "Failed to create samplers or buffers");

            nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(layout != nullptr, "Failed to create sampler test binding layout");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create sampler test compute pipeline");

            auto runSample = [&](nvrhi::ITexture* tex, nvrhi::ISampler* samp, float u, float v, float lod,
                                 const float expected[4], float eps, const std::string& label)
            {
                const SamplerTestParams params{ { u, v }, { 0.f, 0.f }, lod, 1u, 1u, 0u };
                fix.uploadBuffer(cb, &params, sizeof(params), 0);

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, tex))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create binding set for " + label);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float actual[4]{};
                fix.readbackBuffer(outBuf, actual, sizeof(actual), 0);
                for (int c = 0; c < 4; ++c)
                {
                    UAA_CHECK_NEAR(actual[c], expected[c], eps, label + " channel " + std::to_string(c));
                }
            };

            const float red[4]   = { 1.f, 0.f, 0.f, 1.f };
            const float green[4] = { 0.f, 1.f, 0.f, 1.f };
            const float blue[4]  = { 0.f, 0.f, 1.f, 1.f };
            const float mipMix[4] = { 0.5f, 0.5f, 0.f, 1.f };
            const float texelMix[4] = { 0.7f, 0.3f, 0.f, 1.f };

            // Explicit LODs 0, 1, 2 with point and linear samplers.
            runSample(mipTex, pointSamp, 0.5f, 0.5f, 0.0f, red,   0.01f, "point_lod0");
            runSample(mipTex, pointSamp, 0.5f, 0.5f, 1.0f, green, 0.01f, "point_lod1");
            runSample(mipTex, pointSamp, 0.5f, 0.5f, 2.0f, blue,  0.01f, "point_lod2");
            runSample(mipTex, linearSamp, 0.5f, 0.5f, 0.0f, red,   0.01f, "linear_lod0");
            runSample(mipTex, linearSamp, 0.5f, 0.5f, 1.0f, green, 0.01f, "linear_lod1");
            runSample(mipTex, linearSamp, 0.5f, 0.5f, 2.0f, blue,  0.01f, "linear_lod2");

            // Fractional mip LOD 0.5: point mip snaps to mip 0, linear mip blends mip 0 and mip 1.
            runSample(mipTex, pointSamp,  0.5f, 0.5f, 0.5f, red,    0.01f, "point_lod0_5");
            runSample(mipTex, linearSamp, 0.5f, 0.5f, 0.5f, mipMix, 0.02f, "linear_lod0_5");

            // Fractional texel coordinate u=0.4 (between center 0.25 and 0.75): point -> red, linear -> 0.7*red + 0.3*green.
            runSample(splitTex, pointSamp,  0.4f, 0.5f, 0.0f, red,      0.01f, "point_frac_uv");
            runSample(splitTex, linearSamp, 0.4f, 0.5f, 0.0f, texelMix, 0.02f, "linear_frac_uv");
        }

        // Tests Clamp, Wrap, Mirror, MirrorOnce (if supported), and Border addressing modes at UVs (-0.25,-0.25) and (1.25,1.25).
        void testSmpAddressing(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle quadTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(4).setHeight(4).setMipLevels(1)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_quad_tex"));
            UAA_CHECK(quadTex != nullptr, "Failed to create 4x4 quadrant texture");

            std::vector<uint8_t> pixels(4 * 4 * 4);
            for (uint32_t y = 0; y < 4; ++y)
            {
                for (uint32_t x = 0; x < 4; ++x)
                {
                    const size_t idx = (y * 4 + x) * 4;
                    const bool right = (x >= 2);
                    const bool bottom = (y >= 2);
                    if (!right && !bottom)
                    {
                        pixels[idx + 0] = 255; pixels[idx + 1] = 0; pixels[idx + 2] = 0; pixels[idx + 3] = 255; // Q00: Red
                    }
                    else if (right && !bottom)
                    {
                        pixels[idx + 0] = 0; pixels[idx + 1] = 255; pixels[idx + 2] = 0; pixels[idx + 3] = 255; // Q10: Green
                    }
                    else if (!right && bottom)
                    {
                        pixels[idx + 0] = 0; pixels[idx + 1] = 0; pixels[idx + 2] = 255; pixels[idx + 3] = 255; // Q01: Blue
                    }
                    else
                    {
                        pixels[idx + 0] = 255; pixels[idx + 1] = 255; pixels[idx + 2] = 0; pixels[idx + 3] = 255; // Q11: Yellow
                    }
                }
            }
            fix.uploadTexture2D(quadTex, 0, 0, pixels.data(), 16u);

            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(64).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_addr_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_addr_cb"));
            UAA_CHECK(outBuf != nullptr && cb != nullptr, "Failed to create buffers for SMP.Addressing");

            // Sample (-0.25, -0.25) at DTid=(0,0) (index 0) and (1.25, 1.25) at DTid=(1,1) (index 3) in one dispatch.
            const SamplerTestParams params{ { -0.25f, -0.25f }, { 1.5f, 1.5f }, 0.0f, 2u, 2u, 0u };
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(layout != nullptr, "Failed to create addressing binding layout");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create addressing compute pipeline");

            struct AddrCase
            {
                nvrhi::SamplerAddressMode mode;
                nvrhi::Color borderColor;
                std::array<float, 4> expectedNeg;
                std::array<float, 4> expectedPos;
                const char* name;
            };

            const std::array<float, 4> cRed    = { 1.f, 0.f, 0.f, 1.f };
            const std::array<float, 4> cYellow = { 1.f, 1.f, 0.f, 1.f };
            const std::array<float, 4> cWhite  = { 1.f, 1.f, 1.f, 1.f };
            const std::array<float, 4> cBlackOpaque = { 0.f, 0.f, 0.f, 1.f };
            const std::array<float, 4> cBlackTrans  = { 0.f, 0.f, 0.f, 0.f };

            std::vector<AddrCase> cases = {
                { nvrhi::SamplerAddressMode::Clamp,  nvrhi::Color(1.f),                cRed,         cYellow,      "Clamp" },
                { nvrhi::SamplerAddressMode::Wrap,   nvrhi::Color(1.f),                cYellow,      cRed,         "Wrap" },
                { nvrhi::SamplerAddressMode::Mirror, nvrhi::Color(1.f),                cRed,         cYellow,      "Mirror" },
                { nvrhi::SamplerAddressMode::Border, nvrhi::Color(1.f, 1.f, 1.f, 1.f), cWhite,       cWhite,       "BorderWhite" },
                { nvrhi::SamplerAddressMode::Border, nvrhi::Color(0.f, 0.f, 0.f, 1.f), cBlackOpaque, cBlackOpaque, "BorderBlackOpaque" },
                { nvrhi::SamplerAddressMode::Border, nvrhi::Color(0.f, 0.f, 0.f, 0.f), cBlackTrans,  cBlackTrans,  "BorderBlackTransparent" }
            };

            const bool mirrorOnceValid = (fix.api() != nvrhi::GraphicsAPI::VULKAN) ||
                hasExtension(fix.deviceInfo(), "VK_KHR_sampler_mirror_clamp_to_edge");
            if (mirrorOnceValid)
            {
                cases.push_back({ nvrhi::SamplerAddressMode::MirrorOnce, nvrhi::Color(1.f), cRed, cYellow, "MirrorOnce" });
            }
            else
            {
                ctx.note("Skipping MirrorOnce on Vulkan because VK_KHR_sampler_mirror_clamp_to_edge is not enabled");
            }

            for (const AddrCase& ac : cases)
            {
                nvrhi::SamplerHandle samp = dev->createSampler(
                    nvrhi::SamplerDesc()
                        .setAllFilters(false)
                        .setAllAddressModes(ac.mode)
                        .setBorderColor(ac.borderColor));
                UAA_CHECK(samp != nullptr, std::string("createSampler failed for ") + ac.name);

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, quadTex))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    layout);
                UAA_CHECK(set != nullptr, std::string("createBindingSet failed for ") + ac.name);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float results[16]{};
                fix.readbackBuffer(outBuf, results, sizeof(results), 0);
                for (int c = 0; c < 4; ++c)
                {
                    UAA_CHECK_NEAR(results[c], ac.expectedNeg[c], 0.01f,
                        std::string(ac.name) + " UV(-0.25,-0.25) ch " + std::to_string(c));
                    UAA_CHECK_NEAR(results[12 + c], ac.expectedPos[c], 0.01f,
                        std::string(ac.name) + " UV(1.25,1.25) ch " + std::to_string(c));
                }
            }
        }

        // Tests comparison sampling against a depth texture using Less and Greater compare functions.
        void testSmpCompare(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kWidth = 8;
            constexpr uint32_t kHeight = 2;

            nvrhi::TextureHandle depthTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(kWidth).setHeight(kHeight)
                    .setFormat(nvrhi::Format::D32)
                    .setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_cmp_depth"));
            nvrhi::TextureHandle dummyColor = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(kWidth).setHeight(kHeight)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_cmp_color"));
            UAA_CHECK(depthTex != nullptr && dummyColor != nullptr, "Failed to create depth/color textures for SMP.Compare");

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);

            nvrhi::BindingLayoutHandle gfxLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC))));
            nvrhi::BindingSetHandle gfxSet = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPC))),
                gfxLayout);
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(dummyColor).setDepthAttachment(depthTex));
            UAA_CHECK(gfxLayout != nullptr && gfxSet != nullptr && fb != nullptr, "Failed to create graphics FB/bindings");

            nvrhi::GraphicsPipelineDesc gfxDesc;
            gfxDesc.setVertexShader(vs).setPixelShader(ps).addBindingLayout(gfxLayout);
            gfxDesc.renderState.rasterState.setCullNone();
            gfxDesc.renderState.depthStencilState.enableDepthTest().enableDepthWrite().setDepthFunc(nvrhi::ComparisonFunc::Always);
            nvrhi::GraphicsPipelineHandle gfxPso = dev->createGraphicsPipeline(gfxDesc, fb->getFramebufferInfo());
            UAA_CHECK(gfxPso != nullptr, "Failed to create depth-fill graphics pipeline");

            // Render left half at depth 0.25 and right half at depth 0.75.
            {
                const FullscreenPC pcLeft{ { 1.f, 0.f, 0.f, 1.f }, 0.25f, 1.0f, { 0.f, 0.f } };
                const FullscreenPC pcRight{ { 0.f, 1.f, 0.f, 1.f }, 0.75f, 1.0f, { 0.f, 0.f } };

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearDepthStencilTexture(depthTex, nvrhi::AllSubresources, true, 1.0f, false, 0u);

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(gfxPso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(0.f, 4.f, 0.f, float(kHeight), 0.f, 1.f)))
                    .addBindingSet(gfxSet));
                cmd->setPushConstants(&pcLeft, sizeof(pcLeft));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(gfxPso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 8.f, 0.f, float(kHeight), 0.f, 1.f)))
                    .addBindingSet(gfxSet));
                cmd->setPushConstants(&pcRight, sizeof(pcRight));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);
            }

            nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_compare", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle csLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(csLayout != nullptr, "Failed to create compare binding layout");

            nvrhi::ComputePipelineHandle csPso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(csLayout));
            UAA_CHECK(csPso != nullptr, "Failed to create compare compute pipeline");

            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(8).setStructStride(4).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_cmp_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerCompareParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_cmp_cb"));
            UAA_CHECK(outBuf != nullptr && cb != nullptr, "Failed to create compare output/CB buffers");

            const SamplerCompareParams cParams{ 0.5f, 0.25f, 0.5f, 0.5f, 2u, 0u, 0u, 0u };
            fix.uploadBuffer(cb, &cParams, sizeof(cParams), 0);

            auto checkCompare = [&](nvrhi::ComparisonFunc func, float expLeft, float expRight, const std::string& label)
            {
                nvrhi::SamplerHandle samp = dev->createSampler(
                    nvrhi::SamplerDesc()
                        .setAllFilters(false)
                        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                        .setReductionType(nvrhi::SamplerReductionType::Comparison)
                        .setCompareFunc(func));
                UAA_CHECK(samp != nullptr, "Failed to create comparison sampler for " + label);

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, depthTex))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    csLayout);
                UAA_CHECK(set != nullptr, "Failed to create comparison binding set");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(csPso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float outVals[2]{};
                fix.readbackBuffer(outBuf, outVals, sizeof(outVals), 0);
                UAA_CHECK_NEAR(outVals[0], expLeft, 0.01f, label + " left (depth 0.25 vs ref 0.5)");
                UAA_CHECK_NEAR(outVals[1], expRight, 0.01f, label + " right (depth 0.75 vs ref 0.5)");
            };

            // SampleCmp compares the supplied reference against the sampled depth. With a
            // Less comparison, reference 0.5 is less than the right texel (0.75), while
            // Greater succeeds against the left texel (0.25).
            checkCompare(nvrhi::ComparisonFunc::Less, 0.0f, 1.0f, "ComparisonFunc::Less");
            checkCompare(nvrhi::ComparisonFunc::Greater, 1.0f, 0.0f, "ComparisonFunc::Greater");
        }

        // Tests maxAnisotropy, mipBias (via SampleGrad), and Minimum/Maximum reduction when supported.
        void testSmpOptions(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle mipTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(16).setHeight(16).setMipLevels(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_opt_mip"));
            UAA_CHECK(mipTex != nullptr, "Failed to create mip texture for SMP.Options");
            uploadSolidMip(fix, mipTex, 0, 16, 16, 255,   0,   0, 255); // Red
            uploadSolidMip(fix, mipTex, 1,  8,  8,   0, 255,   0, 255); // Green
            uploadSolidMip(fix, mipTex, 2,  4,  4,   0,   0, 255, 255); // Blue
            uploadSolidMip(fix, mipTex, 3,  2,  2, 255, 255,   0, 255); // Yellow

            nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(layout != nullptr, "Failed to create options binding layout");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create options compute pipeline");

            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(16).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_opt_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_opt_cb"));
            UAA_CHECK(outBuf != nullptr && cb != nullptr, "Failed to create buffers for SMP.Options");

            auto runGrad = [&](nvrhi::ITexture* tex, nvrhi::ISampler* samp, float uvStepX, float uvStepY, uint32_t mode,
                               const float expected[4], float eps, const std::string& label)
            {
                const SamplerTestParams params{ { 0.5f, 0.5f }, { uvStepX, uvStepY }, 0.0f, 1u, 1u, mode };
                fix.uploadBuffer(cb, &params, sizeof(params), 0);

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, tex))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create binding set for " + label);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float rgba[4]{};
                fix.readbackBuffer(outBuf, rgba, sizeof(rgba), 0);
                for (int c = 0; c < 4; ++c)
                {
                    UAA_CHECK_NEAR(rgba[c], expected[c], eps, label + " ch " + std::to_string(c));
                }
            };

            const float red[4]   = { 1.f, 0.f, 0.f, 1.f };
            const float green[4] = { 0.f, 1.f, 0.f, 1.f };

            // maxAnisotropy = 4.0f (base LOD 0 gradient).
            {
                nvrhi::SamplerHandle anisoSamp = dev->createSampler(
                    nvrhi::SamplerDesc()
                        .setAllFilters(true)
                        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                        .setMaxAnisotropy(4.0f));
                UAA_CHECK(anisoSamp != nullptr, "Failed to create anisotropic sampler");
                runGrad(mipTex, anisoSamp, 1.f / 16.f, 1.f / 16.f, 1u, red, 0.02f, "maxAnisotropy_4");
            }

            // Metal does not apply SamplerDesc::mipBias when creating its native sampler.
            // MoltenVK exposes portability_subset but does not support samplerMipLodBias.
            const bool supportsMipBias = fix.api() != nvrhi::GraphicsAPI::METAL &&
                !(fix.api() == nvrhi::GraphicsAPI::VULKAN &&
                    hasExtension(fix.deviceInfo(), "VK_KHR_portability_subset"));
            if (supportsMipBias)
            {
                nvrhi::SamplerHandle baseSamp = dev->createSampler(
                    nvrhi::SamplerDesc()
                        .setAllFilters(true)
                        .setMipFilter(false)
                        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                        .setMipBias(0.0f));
                nvrhi::SamplerHandle biasSamp = dev->createSampler(
                    nvrhi::SamplerDesc()
                        .setAllFilters(true)
                        .setMipFilter(false)
                        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                        .setMipBias(1.0f));
                UAA_CHECK(baseSamp != nullptr && biasSamp != nullptr, "Failed to create mipBias samplers");

                runGrad(mipTex, baseSamp, 1.f / 16.f, 1.f / 16.f, 1u, red,   0.02f, "mipBias_0");
                runGrad(mipTex, biasSamp, 1.f / 16.f, 1.f / 16.f, 1u, green, 0.02f, "mipBias_1");
            }
            else
            {
                ctx.note("Sampler mip LOD bias is unavailable on this device configuration");
            }

            // SamplerReductionType::Minimum and Maximum.
            if (fix.api() == nvrhi::GraphicsAPI::METAL)
            {
                ctx.note("Skipping SamplerReductionType::Minimum/Maximum: not implemented on Metal NVRHI backend");
                return;
            }

            if (fix.api() == nvrhi::GraphicsAPI::VULKAN && !hasExtension(fix.deviceInfo(), "VK_EXT_sampler_filter_minmax"))
            {
                ctx.note("VK_EXT_sampler_filter_minmax not reported in enabledExtensions; attempting Vulkan 1.2 minmax check");
            }

            if (fix.api() == nvrhi::GraphicsAPI::VULKAN && !fix.deviceInfo().vulkanSamplerFilterMinmax)
            {
                ctx.note("samplerFilterMinmax was not enabled on the Vulkan device; skipping reduction sampler checks");
                return;
            }

            nvrhi::TextureHandle minmaxTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(2).setHeight(2).setMipLevels(1)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_minmax_tex"));
            UAA_CHECK(minmaxTex != nullptr, "Failed to create 2x2 minmax texture");

            // Four texels with distinct values: 0.2 (51), 0.4 (102), 0.6 (153), 0.8 (204).
            const uint8_t mmPixels[2 * 2 * 4] = {
                 51,  51,  51, 255,   102, 102, 102, 255,
                153, 153, 153, 255,   204, 204, 204, 255
            };
            fix.uploadTexture2D(minmaxTex, 0, 0, mmPixels, 8u);

            nvrhi::SamplerHandle minSamp = dev->createSampler(
                nvrhi::SamplerDesc()
                    .setAllFilters(true)
                    .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                    .setReductionType(nvrhi::SamplerReductionType::Minimum));
            nvrhi::SamplerHandle maxSamp = dev->createSampler(
                nvrhi::SamplerDesc()
                    .setAllFilters(true)
                    .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
                    .setReductionType(nvrhi::SamplerReductionType::Maximum));
            UAA_CHECK(minSamp != nullptr && maxSamp != nullptr, "Failed to create Minimum/Maximum samplers");

            const float expMin[4] = { 51.f / 255.f, 51.f / 255.f, 51.f / 255.f, 1.f };
            const float expMax[4] = { 204.f / 255.f, 204.f / 255.f, 204.f / 255.f, 1.f };
            runGrad(minmaxTex, minSamp, 0.f, 0.f, 0u, expMin, 0.02f, "reduction_Minimum");
            runGrad(minmaxTex, maxSamp, 0.f, 0.f, 0u, expMax, 0.02f, "reduction_Maximum");
        }

        // Tests dropping the caller's sampler handle before submitting the recorded command list.
        void testSmpLifetime(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle tex = dev->createTexture(
                nvrhi::TextureDesc().setWidth(4).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("smp_life_tex"));
            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(16).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_life_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::Common)
                    .setDebugName("smp_life_cb"));
            UAA_CHECK(tex != nullptr && outBuf != nullptr && cb != nullptr, "Failed to create resources for SMP.Lifetime");

            uploadSolidMip(fix, tex, 0, 4, 4, 64, 192, 128, 255);
            const SamplerTestParams params{ { 0.5f, 0.5f }, { 0.f, 0.f }, 0.0f, 1u, 1u, 0u };
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();

            {
                nvrhi::SamplerHandle samp = dev->createSampler(
                    nvrhi::SamplerDesc().setAllFilters(false).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
                UAA_CHECK(samp != nullptr, "Failed to create sampler");

                nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                        .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
                UAA_CHECK(layout != nullptr, "Failed to create lifetime binding layout");

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, tex))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create lifetime binding set");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create lifetime compute pipeline");

                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
            }

            cmd->close();
            // Sampler, binding set, layout, shader, and PSO handles are released before submit.
            fix.submitAndWait(cmd);

            float rgba[4]{};
            fix.readbackBuffer(outBuf, rgba, sizeof(rgba), 0);
            UAA_CHECK_NEAR(rgba[0],  64.f / 255.f, 0.01f, "SMP.Lifetime R mismatch");
            UAA_CHECK_NEAR(rgba[1], 192.f / 255.f, 0.01f, "SMP.Lifetime G mismatch");
            UAA_CHECK_NEAR(rgba[2], 128.f / 255.f, 0.01f, "SMP.Lifetime B mismatch");
            UAA_CHECK_NEAR(rgba[3], 1.0f,          0.01f, "SMP.Lifetime A mismatch");
        }
    } // namespace

    UAA_REGISTER_TEST("SMP.Filters", "core", testSmpFilters);
    UAA_REGISTER_TEST("SMP.Addressing", "core", testSmpAddressing);
    UAA_REGISTER_TEST("SMP.Compare", "core", testSmpCompare);
    UAA_REGISTER_TEST("SMP.Options", "core", testSmpOptions);
    UAA_REGISTER_TEST("SMP.Lifetime", "core", testSmpLifetime);
} // namespace nvrhi::uaa
