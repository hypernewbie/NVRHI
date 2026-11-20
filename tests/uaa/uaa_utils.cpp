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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
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

        struct UavChainPC
        {
            uint32_t count = 0;
            uint32_t stepAdd = 0;
            uint32_t stepMul = 0;
            uint32_t pad = 0;
        };

        struct TexFillParams
        {
            float color[4] = {};
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t mode = 0;
            uint32_t pad = 0;
        };

        void testDescriptors(TestContext& ctx)
        {
            const nvrhi::BlendState::RenderTarget addBlend = nvrhi::utils::CreateAddBlendState(
                nvrhi::BlendFactor::SrcAlpha, nvrhi::BlendFactor::InvSrcAlpha);
            UAA_CHECK(addBlend.blendEnable, "CreateAddBlendState must enable blending");
            UAA_CHECK_EQ(addBlend.blendOp, nvrhi::BlendOp::Add, "CreateAddBlendState blendOp must be Add");
            UAA_CHECK_EQ(addBlend.srcBlend, nvrhi::BlendFactor::SrcAlpha, "CreateAddBlendState srcBlend mismatch");
            UAA_CHECK_EQ(addBlend.destBlend, nvrhi::BlendFactor::InvSrcAlpha, "CreateAddBlendState destBlend mismatch");
            UAA_CHECK_EQ(addBlend.srcBlendAlpha, nvrhi::BlendFactor::Zero, "CreateAddBlendState srcBlendAlpha mismatch");
            UAA_CHECK_EQ(addBlend.destBlendAlpha, nvrhi::BlendFactor::One, "CreateAddBlendState destBlendAlpha mismatch");

            const nvrhi::BufferDesc staticCb = nvrhi::utils::CreateStaticConstantBufferDesc(256, "uaa_static_cb");
            UAA_CHECK_EQ(staticCb.byteSize, uint64_t(256), "Static CB byteSize mismatch");
            UAA_CHECK_EQ(staticCb.debugName, std::string("uaa_static_cb"), "Static CB debugName mismatch");
            UAA_CHECK(staticCb.isConstantBuffer && !staticCb.isVolatile, "Static CB flags mismatch");

            const nvrhi::BufferDesc volatileCb = nvrhi::utils::CreateVolatileConstantBufferDesc(512, "uaa_volatile_cb", 4);
            UAA_CHECK_EQ(volatileCb.byteSize, uint64_t(512), "Volatile CB byteSize mismatch");
            UAA_CHECK_EQ(volatileCb.debugName, std::string("uaa_volatile_cb"), "Volatile CB debugName mismatch");
            UAA_CHECK(volatileCb.isConstantBuffer && volatileCb.isVolatile, "Volatile CB flags mismatch");
            UAA_CHECK_EQ(volatileCb.maxVersions, 4u, "Volatile CB maxVersions mismatch");
        }

        void testBindingSet(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            constexpr uint32_t kCount = 64;
            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0xB1D5ull);

            nvrhi::ShaderHandle cs = ctx.fixture->createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);

            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::utils::CreateStaticConstantBufferDesc(256, "uaa_utl_bs_cb")
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
            UAA_CHECK(cb != nullptr, "Failed to create CB");

            nvrhi::BufferHandle inBuf = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
                    .setDebugName("uaa_utl_bs_in"));
            UAA_CHECK(inBuf != nullptr, "Failed to create input buffer");

            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(kCount * sizeof(uint32_t))
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                    .setDebugName("uaa_utl_bs_out"));
            UAA_CHECK(outBuf != nullptr, "Failed to create output buffer");

            const CopyAddParams params{ kCount, 17u, 0x3C3C3C3Cu, 0u };
            ctx.fixture->uploadBuffer(cb, &params, sizeof(params));
            ctx.fixture->uploadBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));

            const nvrhi::BindingSetDesc setDesc = nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf));

            nvrhi::BindingLayoutHandle autoLayout;
            nvrhi::BindingSetHandle autoSet;
            const bool autoOk = nvrhi::utils::CreateBindingSetAndLayout(
                dev, nvrhi::ShaderType::Compute, 0, setDesc, autoLayout, autoSet);
            UAA_CHECK(autoOk && autoLayout != nullptr && autoSet != nullptr,
                "CreateBindingSetAndLayout failed to create auto layout and set");
            UAA_CHECK(autoLayout->getDesc() != nullptr, "Auto layout must have regular BindingLayoutDesc");
            UAA_CHECK_EQ(autoLayout->getDesc()->bindings.size(), size_t(3), "Auto layout item count mismatch");
            UAA_CHECK(autoSet->getLayout() == autoLayout.Get(), "Auto set layout pointer mismatch");

            auto runAndVerify = [&](nvrhi::IBindingLayout* layout, nvrhi::IBindingSet* set, uint32_t expectedSeedTag)
            {
                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "createComputePipeline returned null");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearBufferUInt(outBuf, expectedSeedTag);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                ctx.fixture->submitAndWait(cmd);

                std::vector<uint32_t> actual(kCount, 0u);
                ctx.fixture->readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t expected = (inputWords[i] + params.addVal) ^ params.xorVal;
                    UAA_CHECK_EQ(actual[i], expected, "GPU output mismatch at index " + std::to_string(i));
                }
            };

            if (dev->getGraphicsAPI() != nvrhi::GraphicsAPI::VULKAN)
            {
                runAndVerify(autoLayout, autoSet, 0x11111111u);
            }

            nvrhi::BindingLayoutHandle preLayout = dev->createBindingLayout(
                ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            UAA_CHECK(preLayout != nullptr, "Failed to create pre-populated binding layout");

            nvrhi::BindingSetHandle reusedSet;
            const bool reuseOk = nvrhi::utils::CreateBindingSetAndLayout(
                dev, nvrhi::ShaderType::Compute, 0, setDesc, preLayout, reusedSet);
            UAA_CHECK(reuseOk && reusedSet != nullptr, "CreateBindingSetAndLayout failed with pre-populated layout");
            UAA_CHECK(reusedSet->getLayout() == preLayout.Get(), "Reused binding set must reference pre-populated layout");
            runAndVerify(preLayout, reusedSet, 0x22222222u);

            if (ctx.fixture->isValidationActive())
            {
                ExpectErrorsGuard guard(*ctx.fixture);
                const nvrhi::BindingSetDesc badSetDesc = nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::Sampler(0, nullptr));
                nvrhi::BindingLayoutHandle badLayout;
                nvrhi::BindingSetHandle badSet;
                const bool ok = nvrhi::utils::CreateBindingSetAndLayout(
                    dev, nvrhi::ShaderType::Compute, 0, badSetDesc, badLayout, badSet);
                UAA_CHECK(!ok && badSet == nullptr, "Invalid null sampler resourceHandle binding set must fail under validation");
                UAA_CHECK(guard.errorCount() > 0, "Validation layer must report error for null sampler resourceHandle");
            }
        }

        void testClearAttachments(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            constexpr uint32_t kW = 16;
            constexpr uint32_t kH = 16;

            nvrhi::TextureHandle colorTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(kW)
                    .setHeight(kH)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
                    .setDebugName("uaa_utl_clear_color"));
            UAA_CHECK(colorTex != nullptr, "Failed to create colour attachment");

            nvrhi::TextureHandle depthTex;
            if ((dev->queryFormatSupport(nvrhi::Format::D32) & nvrhi::FormatSupport::DepthStencil) != 0u)
            {
                depthTex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setWidth(kW)
                        .setHeight(kH)
                        .setFormat(nvrhi::Format::D32)
                        .setIsRenderTarget(true)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
                        .setDebugName("uaa_utl_clear_depth"));
                UAA_CHECK(depthTex != nullptr, "Failed to create depth attachment");
            }

            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(colorTex);
            if (depthTex)
            {
                fbDesc.setDepthAttachment(depthTex);
            }
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(fbDesc);
            UAA_CHECK(fb != nullptr, "Failed to create framebuffer");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");
            cmd->open();
            nvrhi::utils::ClearColorAttachment(cmd, fb, 0, nvrhi::Color(0.25f, 0.5f, 0.75f, 1.0f));
            if (depthTex)
            {
                nvrhi::utils::ClearDepthStencilAttachment(cmd, fb, 0.5f, 0u);
            }
            cmd->close();
            ctx.fixture->submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            ctx.fixture->readbackTexture2D(colorTex, 0, 0, pixels);
            UAA_CHECK_EQ(pixels.size(), size_t(kW) * kH * 4u, "Readback pixel byte size mismatch");

            for (uint32_t i = 0; i < kW * kH; ++i)
            {
                std::string diff;
                const bool ok = TestFixture::checkPixelRgba8(
                    pixels.data() + size_t(i) * 4u, 64, 128, 191, 255, 1, diff);
                UAA_CHECK(ok, "Cleared pixel mismatch at texel " + std::to_string(i) + ": " + diff);
            }
        }

        void testBarriers(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            {
                constexpr uint32_t kCount = 64;
                std::vector<uint32_t> initial(kCount);
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    initial[i] = i + 1u;
                }

                nvrhi::ShaderHandle cs = ctx.fixture->createShader("cs_uav_chain", nvrhi::ShaderType::Compute);

                nvrhi::BufferHandle buf = dev->createBuffer(
                    nvrhi::BufferDesc()
                        .setByteSize(kCount * sizeof(uint32_t))
                        .setStructStride(sizeof(uint32_t))
                        .setCanHaveUAVs(true)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                        .setDebugName("uaa_utl_barrier_buf"));
                UAA_CHECK(buf != nullptr, "Failed to create UAV chain buffer");
                ctx.fixture->uploadBuffer(buf, initial.data(), initial.size() * sizeof(uint32_t));

                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(UavChainPC)))
                        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)));
                UAA_CHECK(layout != nullptr, "Failed to create UAV chain binding layout");

                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buf)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create UAV chain binding set");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create UAV chain compute pipeline");

                const UavChainPC pc1{ kCount, 5u, 3u, 0u };
                const UavChainPC pc2{ kCount, 7u, 2u, 0u };

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "Failed to create command list");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set));
                cmd->setPushConstants(&pc1, sizeof(pc1));
                cmd->dispatch(1, 1, 1);

                nvrhi::utils::BufferUavBarrier(cmd, buf);

                cmd->setPushConstants(&pc2, sizeof(pc2));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                ctx.fixture->submitAndWait(cmd);

                std::vector<uint32_t> actual(kCount, 0u);
                ctx.fixture->readbackBuffer(buf, actual.data(), actual.size() * sizeof(uint32_t));
                for (uint32_t i = 0; i < kCount; ++i)
                {
                    const uint32_t step1 = initial[i] * pc1.stepMul + pc1.stepAdd;
                    const uint32_t expected = step1 * pc2.stepMul + pc2.stepAdd;
                    UAA_CHECK_EQ(actual[i], expected, "BufferUavBarrier chained output mismatch at index " + std::to_string(i));
                }
            }

            {
                constexpr uint32_t kW = 8;
                constexpr uint32_t kH = 8;
                nvrhi::ShaderHandle cs = ctx.fixture->createShader("cs_texture_uav_fill", nvrhi::ShaderType::Compute);

                nvrhi::TextureHandle tex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setWidth(kW)
                        .setHeight(kH)
                        .setFormat(nvrhi::Format::RGBA32_FLOAT)
                        .setIsUAV(true)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                        .setDebugName("uaa_utl_barrier_tex"));
                UAA_CHECK(tex != nullptr, "Failed to create UAV texture");

                nvrhi::BufferHandle cb1 = dev->createBuffer(
                    nvrhi::utils::CreateStaticConstantBufferDesc(256, "uaa_utl_texfill_cb1")
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
                nvrhi::BufferHandle cb2 = dev->createBuffer(
                    nvrhi::utils::CreateStaticConstantBufferDesc(256, "uaa_utl_texfill_cb2")
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
                UAA_CHECK(cb1 != nullptr && cb2 != nullptr, "Failed to create texture fill constant buffers");

                const TexFillParams p1{ { 0.25f, 0.125f, 0.5f, 0.25f }, kW, kH, 0u, 0u };
                const TexFillParams p2{ { 0.25f, 0.375f, 0.25f, 0.75f }, kW, kH, 1u, 0u };
                ctx.fixture->uploadBuffer(cb1, &p1, sizeof(p1));
                ctx.fixture->uploadBuffer(cb2, &p2, sizeof(p2));

                nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                    ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                        .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
                        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(1)));
                UAA_CHECK(layout != nullptr, "Failed to create texture fill binding layout");

                nvrhi::BindingSetHandle set1 = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, tex))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, cb1)),
                    layout);
                nvrhi::BindingSetHandle set2 = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, tex))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, cb2)),
                    layout);
                UAA_CHECK(set1 != nullptr && set2 != nullptr, "Failed to create texture fill binding sets");

                nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
                UAA_CHECK(pso != nullptr, "Failed to create texture fill compute pipeline");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "Failed to create command list");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set1));
                cmd->dispatch(1, 1, 1);

                nvrhi::utils::TextureUavBarrier(cmd, tex);

                cmd->setComputeState(nvrhi::ComputeState().setPipeline(pso).addBindingSet(set2));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                ctx.fixture->submitAndWait(cmd);

                std::vector<uint8_t> pixels;
                ctx.fixture->readbackTexture2D(tex, 0, 0, pixels);
                constexpr float expected[] = {0.5f, 0.5f, 0.75f, 1.f};
                UAA_CHECK_EQ(pixels.size(), size_t(kW) * kH * sizeof(expected), "Texture UAV fill readback size mismatch");

                for (uint32_t i = 0; i < kW * kH; ++i)
                {
                    float actual[4];
                    std::memcpy(actual, pixels.data() + size_t(i) * sizeof(actual), sizeof(actual));
                    for (size_t channel = 0; channel < 4; ++channel)
                    {
                        UAA_CHECK_EQ(actual[channel], expected[channel],
                            "TextureUavBarrier output mismatch at texel " + std::to_string(i)
                                + ", channel " + std::to_string(channel));
                    }
                }
            }
        }

        void testRayBuild(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            if (!dev->queryFeatureSupport(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("RayTracingAccelStruct not supported");
            }

            const float vertices[9] = {
                0.0f, 0.5f, 0.0f,
                0.5f, -0.5f, 0.0f,
                -0.5f, -0.5f, 0.0f,
            };

            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(vertices))
                    .setIsAccelStructBuildInput(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput)
                    .setDebugName("uaa_utl_raybuild_vb"));
            UAA_CHECK(vb != nullptr, "Failed to create BLAS vertex buffer");
            ctx.fixture->uploadBuffer(vb, vertices, sizeof(vertices));

            nvrhi::rt::GeometryDesc geom;
            geom.setFlags(nvrhi::rt::GeometryFlags::Opaque)
                .setTriangles(
                    nvrhi::rt::GeometryTriangles()
                        .setVertexBuffer(vb)
                        .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
                        .setVertexCount(3)
                        .setVertexStride(sizeof(float) * 3));

            nvrhi::rt::AccelStructDesc asDesc;
            asDesc.addBottomLevelGeometry(geom)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_utl_blas");

            nvrhi::rt::AccelStructHandle blas = dev->createAccelStruct(asDesc);
            UAA_CHECK(blas != nullptr, "createAccelStruct returned null");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            nvrhi::utils::BuildBottomLevelAccelStruct(cmd, blas, asDesc);
            cmd->close();
            ctx.fixture->submitAndWait(cmd);
        }

        void testChooseFormat(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            const nvrhi::Format mixedCandidates[] = {
                nvrhi::Format::UNKNOWN,
                nvrhi::Format::D32,
                nvrhi::Format::RGBA8_UNORM,
                nvrhi::Format::RGBA16_FLOAT,
            };
            const nvrhi::Format pickedMixed = nvrhi::utils::ChooseFormat(
                dev,
                nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::RenderTarget,
                mixedCandidates,
                sizeof(mixedCandidates) / sizeof(mixedCandidates[0]));
            UAA_CHECK_EQ(pickedMixed, nvrhi::Format::RGBA8_UNORM, "ChooseFormat must pick first matching colour RT format");

            const nvrhi::Format orderedCandidates[] = {
                nvrhi::Format::RGBA16_FLOAT,
                nvrhi::Format::RGBA8_UNORM,
            };
            const nvrhi::Format pickedOrdered = nvrhi::utils::ChooseFormat(
                dev,
                nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::ShaderLoad,
                orderedCandidates,
                sizeof(orderedCandidates) / sizeof(orderedCandidates[0]));
            UAA_CHECK_EQ(pickedOrdered, nvrhi::Format::RGBA16_FLOAT, "ChooseFormat must return first matching candidate");

            const nvrhi::Format unsupportedCandidates[] = {
                nvrhi::Format::UNKNOWN,
                nvrhi::Format::D32,
            };
            const nvrhi::Format pickedNone = nvrhi::utils::ChooseFormat(
                dev,
                nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::RenderTarget | nvrhi::FormatSupport::Blendable,
                unsupportedCandidates,
                sizeof(unsupportedCandidates) / sizeof(unsupportedCandidates[0]));
            UAA_CHECK_EQ(pickedNone, nvrhi::Format::UNKNOWN, "All-unsupported ChooseFormat must return UNKNOWN");

            const nvrhi::Format pickedEmpty = nvrhi::utils::ChooseFormat(
                dev, nvrhi::FormatSupport::Texture, nullptr, 0);
            UAA_CHECK_EQ(pickedEmpty, nvrhi::Format::UNKNOWN, "Empty ChooseFormat list must return UNKNOWN");
        }

        void testNames(TestContext& ctx)
        {
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D11), "D3D11") == 0, "GraphicsAPIToString D3D11");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D12), "D3D12") == 0, "GraphicsAPIToString D3D12");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::VULKAN), "Vulkan") == 0, "GraphicsAPIToString Vulkan");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::METAL), "Metal") == 0, "GraphicsAPIToString Metal");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(static_cast<nvrhi::GraphicsAPI>(99)), "<UNKNOWN>") == 0, "GraphicsAPIToString unknown");

            const nvrhi::TextureDimension dims[] = {
                nvrhi::TextureDimension::Unknown,
                nvrhi::TextureDimension::Texture1D,
                nvrhi::TextureDimension::Texture1DArray,
                nvrhi::TextureDimension::Texture2D,
                nvrhi::TextureDimension::Texture2DArray,
                nvrhi::TextureDimension::TextureCube,
                nvrhi::TextureDimension::TextureCubeArray,
                nvrhi::TextureDimension::Texture2DMS,
                nvrhi::TextureDimension::Texture2DMSArray,
                nvrhi::TextureDimension::Texture3D,
            };
            for (auto d : dims)
            {
                const char* s = nvrhi::utils::TextureDimensionToString(d);
                UAA_CHECK(s != nullptr && std::strcmp(s, "<INVALID>") != 0, "TextureDimensionToString valid");
            }
            UAA_CHECK(std::strcmp(nvrhi::utils::TextureDimensionToString(static_cast<nvrhi::TextureDimension>(99)), "<INVALID>") == 0,
                "TextureDimensionToString invalid");

            UAA_CHECK(std::strcmp(nvrhi::utils::DebugNameToString(""), "<UNNAMED>") == 0, "DebugNameToString empty");
            UAA_CHECK(std::strcmp(nvrhi::utils::DebugNameToString("uaa_name"), "uaa_name") == 0, "DebugNameToString non-empty");

            const nvrhi::ShaderType stages[] = {
                nvrhi::ShaderType::None, nvrhi::ShaderType::Compute, nvrhi::ShaderType::Vertex,
                nvrhi::ShaderType::Hull, nvrhi::ShaderType::Domain, nvrhi::ShaderType::Geometry,
                nvrhi::ShaderType::Pixel, nvrhi::ShaderType::Amplification, nvrhi::ShaderType::Mesh,
                nvrhi::ShaderType::AllGraphics, nvrhi::ShaderType::RayGeneration, nvrhi::ShaderType::AnyHit,
                nvrhi::ShaderType::ClosestHit, nvrhi::ShaderType::Miss, nvrhi::ShaderType::Intersection,
                nvrhi::ShaderType::Callable, nvrhi::ShaderType::AllRayTracing, nvrhi::ShaderType::All,
            };
            for (auto st : stages)
            {
                const char* s = nvrhi::utils::ShaderStageToString(st);
                UAA_CHECK(s != nullptr && std::strcmp(s, "<INVALID>") != 0, "ShaderStageToString valid");
            }

            struct ResourceTypeExpected
            {
                nvrhi::ResourceType type;
                const char* expected;
            };
            const ResourceTypeExpected resourceTypeCases[] = {
                { nvrhi::ResourceType::None, "None" },
                { nvrhi::ResourceType::Texture_SRV, "Texture_SRV" },
                { nvrhi::ResourceType::Texture_UAV, "Texture_UAV" },
                { nvrhi::ResourceType::TypedBuffer_SRV, "Buffer_SRV" },
                { nvrhi::ResourceType::TypedBuffer_UAV, "Buffer_UAV" },
                { nvrhi::ResourceType::StructuredBuffer_SRV, "StructuredBuffer_SRV" },
                { nvrhi::ResourceType::StructuredBuffer_UAV, "StructuredBuffer_UAV" },
                { nvrhi::ResourceType::RawBuffer_SRV, "RawBuffer_SRV" },
                { nvrhi::ResourceType::RawBuffer_UAV, "RawBuffer_UAV" },
                { nvrhi::ResourceType::ConstantBuffer, "ConstantBuffer" },
                { nvrhi::ResourceType::VolatileConstantBuffer, "VolatileConstantBuffer" },
                { nvrhi::ResourceType::Sampler, "Sampler" },
                { nvrhi::ResourceType::RayTracingAccelStruct, "RayTracingAccelStruct" },
                { nvrhi::ResourceType::PushConstants, "PushConstants" },
            };
            for (const auto& entry : resourceTypeCases)
            {
                const char* s = nvrhi::utils::ResourceTypeToString(entry.type);
                UAA_CHECK(s != nullptr && std::strcmp(s, entry.expected) == 0,
                    std::string("ResourceTypeToString mismatch for ") + entry.expected);
            }
            const char* sfStr = nvrhi::utils::ResourceTypeToString(nvrhi::ResourceType::SamplerFeedbackTexture_UAV);
            UAA_CHECK(sfStr != nullptr && (std::strcmp(sfStr, "<INVALID>") == 0 || std::strcmp(sfStr, "SamplerFeedbackTexture_UAV") == 0),
                "ResourceTypeToString for SamplerFeedbackTexture_UAV");
            UAA_CHECK(std::strcmp(nvrhi::utils::ResourceTypeToString(nvrhi::ResourceType::Count), "<INVALID>") == 0,
                "ResourceTypeToString Count");

            for (size_t f = 0; f < size_t(nvrhi::Format::COUNT); ++f)
            {
                const nvrhi::Format fmt = static_cast<nvrhi::Format>(f);
                const char* s = nvrhi::utils::FormatToString(fmt);
                UAA_CHECK(s != nullptr && std::strcmp(s, nvrhi::getFormatInfo(fmt).name) == 0, "FormatToString match");
            }
            UAA_CHECK(std::strcmp(nvrhi::utils::FormatToString(static_cast<nvrhi::Format>(uint8_t(nvrhi::Format::COUNT) + 3)), "UNKNOWN") == 0,
                "FormatToString out of range");

            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Graphics), "Graphics") == 0, "CommandQueue Graphics");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Compute), "Compute") == 0, "CommandQueue Compute");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Copy), "Copy") == 0, "CommandQueue Copy");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Count), "<INVALID>") == 0, "CommandQueue Count");

            const std::string heapLocal = nvrhi::utils::GenerateHeapDebugName(
                nvrhi::HeapDesc().setType(nvrhi::HeapType::DeviceLocal).setCapacity(4096));
            const std::string heapUpload = nvrhi::utils::GenerateHeapDebugName(
                nvrhi::HeapDesc().setType(nvrhi::HeapType::Upload).setCapacity(1024));
            const std::string heapReadback = nvrhi::utils::GenerateHeapDebugName(
                nvrhi::HeapDesc().setType(nvrhi::HeapType::Readback).setCapacity(2048));
            const std::string heapInvalid = nvrhi::utils::GenerateHeapDebugName(
                nvrhi::HeapDesc().setType(static_cast<nvrhi::HeapType>(99)).setCapacity(512));
            UAA_CHECK(heapLocal.find("DeviceLocal") != std::string::npos && heapLocal.find("4096") != std::string::npos, "GenerateHeapDebugName DeviceLocal");
            UAA_CHECK(heapUpload.find("Upload") != std::string::npos && heapUpload.find("1024") != std::string::npos, "GenerateHeapDebugName Upload");
            UAA_CHECK(heapReadback.find("Readback") != std::string::npos && heapReadback.find("2048") != std::string::npos, "GenerateHeapDebugName Readback");
            UAA_CHECK(heapInvalid.find("Invalid-Type") != std::string::npos, "GenerateHeapDebugName invalid");

            const std::string texName = nvrhi::utils::GenerateTextureDebugName(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setWidth(64)
                    .setHeight(32)
                    .setArraySize(4)
                    .setMipLevels(3)
                    .setIsRenderTarget(true)
                    .setIsUAV(true)
                    .setIsTypeless(true));
            UAA_CHECK(texName.find("Texture2DArray") != std::string::npos &&
                texName.find("RGBA8_UNORM") != std::string::npos &&
                texName.find("ArraySize = 4") != std::string::npos &&
                texName.find("IsRenderTarget") != std::string::npos &&
                texName.find("IsUAV") != std::string::npos &&
                texName.find("IsTypeless") != std::string::npos,
                "GenerateTextureDebugName output mismatch");

            const std::string tex3dName = nvrhi::utils::GenerateTextureDebugName(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture3D)
                    .setFormat(nvrhi::Format::R16_FLOAT)
                    .setWidth(8)
                    .setHeight(8)
                    .setDepth(4));
            UAA_CHECK(tex3dName.find("Depth = 4") != std::string::npos, "GenerateTextureDebugName 3D depth");

            const std::string texMsName = nvrhi::utils::GenerateTextureDebugName(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DMS)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setWidth(16)
                    .setHeight(16)
                    .setSampleCount(4));
            UAA_CHECK(texMsName.find("SampleCount = 4") != std::string::npos, "GenerateTextureDebugName MS");

            const std::string bufName = nvrhi::utils::GenerateBufferDebugName(
                nvrhi::BufferDesc()
                    .setByteSize(1024)
                    .setFormat(nvrhi::Format::R32_UINT)
                    .setStructStride(16)
                    .setIsVolatile(true)
                    .setMaxVersions(3)
                    .setCanHaveUAVs(true)
                    .setCanHaveTypedViews(true)
                    .setCanHaveRawViews(true)
                    .setIsVertexBuffer(true)
                    .setIsIndexBuffer(true)
                    .setIsConstantBuffer(true)
                    .setIsDrawIndirectArgs(true)
                    .setIsAccelStructBuildInput(true)
                    .setIsAccelStructStorage(true)
                    .setIsShaderBindingTable(true));
            UAA_CHECK(bufName.find("ByteSize = 1024") != std::string::npos &&
                bufName.find("Format = R32_UINT") != std::string::npos &&
                bufName.find("StructStride = 16") != std::string::npos &&
                bufName.find("IsVolatile") != std::string::npos &&
                bufName.find("CanHaveUAVs") != std::string::npos &&
                bufName.find("IsShaderBindingTable") != std::string::npos,
                "GenerateBufferDebugName output mismatch");
        }

        void testAllocator(TestContext& ctx)
        {
            nvrhi::utils::BitSetAllocator alloc(32, false);
            UAA_CHECK_EQ(alloc.getCapacity(), size_t(32), "BitSetAllocator capacity mismatch");

            std::vector<int> seen(32, 0);
            for (int i = 0; i < 32; ++i)
            {
                const int idx = alloc.allocate();
                UAA_CHECK(idx >= 0 && idx < 32, "Allocated index out of range");
                UAA_CHECK_EQ(seen[idx], 0, "Duplicate index allocated");
                seen[idx] = 1;
            }
            UAA_CHECK_EQ(alloc.allocate(), -1, "Exhausted allocator must return -1");

            alloc.release(-1);
            alloc.release(32);
            UAA_CHECK_EQ(alloc.allocate(), -1, "Out-of-range release must not free indices");

            alloc.release(20);
            alloc.release(5);
            alloc.release(12);
            UAA_CHECK_EQ(alloc.allocate(), 5, "Reused lowest released index");
            UAA_CHECK_EQ(alloc.allocate(), 12, "Reused next released index");
            UAA_CHECK_EQ(alloc.allocate(), 20, "Reused last released index");
            UAA_CHECK_EQ(alloc.allocate(), -1, "Exhausted after reuse");

            constexpr size_t kMtCapacity = 256;
            nvrhi::utils::BitSetAllocator mtAlloc(kMtCapacity, true);
            std::array<std::atomic<int>, kMtCapacity> live{};
            for (auto& cell : live)
            {
                cell.store(0, std::memory_order_relaxed);
            }

            std::atomic<bool> collision{false};
            std::vector<std::thread> threads;
            for (int t = 0; t < 4; ++t)
            {
                threads.emplace_back([&]()
                {
                    std::vector<int> held;
                    held.reserve(32);
                    for (int iter = 0; iter < 400; ++iter)
                    {
                        for (int k = 0; k < 16; ++k)
                        {
                            const int idx = mtAlloc.allocate();
                            if (idx >= 0)
                            {
                                if (live[idx].exchange(1, std::memory_order_acq_rel) != 0)
                                {
                                    collision.store(true, std::memory_order_relaxed);
                                }
                                held.push_back(idx);
                            }
                        }
                        for (int idx : held)
                        {
                            live[idx].store(0, std::memory_order_release);
                            mtAlloc.release(idx);
                        }
                        held.clear();
                    }
                });
            }
            for (auto& th : threads)
            {
                th.join();
            }

            UAA_CHECK(!collision.load(), "Multithreaded BitSetAllocator produced duplicate live index");
            for (size_t i = 0; i < kMtCapacity; ++i)
            {
                UAA_CHECK(mtAlloc.allocate() >= 0, "Expected all slots free after thread join");
            }
            UAA_CHECK_EQ(mtAlloc.allocate(), -1, "Expected allocator full after full drain");
        }
    } // namespace

    UAA_REGISTER_TEST("UTL.Descriptors", "core", testDescriptors);
    UAA_REGISTER_TEST("UTL.BindingSet", "core", testBindingSet);
    UAA_REGISTER_TEST("UTL.ClearAttachments", "core", testClearAttachments);
    UAA_REGISTER_TEST("UTL.Barriers", "core", testBarriers);
    UAA_REGISTER_TEST("UTL.RayBuild", "core", testRayBuild);
    UAA_REGISTER_TEST("UTL.ChooseFormat", "core", testChooseFormat);
    UAA_REGISTER_TEST("UTL.Names", "core", testNames);
    UAA_REGISTER_TEST("UTL.Allocator", "core", testAllocator);
} // namespace nvrhi::uaa
