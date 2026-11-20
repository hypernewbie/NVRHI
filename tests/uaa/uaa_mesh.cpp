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

#include <cstdint>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct MeshPushColor
        {
            float color[4];
        };

        struct PsColorBias
        {
            float colorBias[4];
            float pad[60];
        };

        struct IndirectGenParams
        {
            uint32_t word0;
            uint32_t word1;
            uint32_t word2;
            uint32_t word3;
            uint32_t word4;
            uint32_t offsetWords;
            uint32_t count;
            uint32_t pad;
            uint32_t cbPad[56];
        };

        struct MeshResources
        {
            nvrhi::TextureHandle colorTarget;
            nvrhi::FramebufferHandle fb;
            nvrhi::BindingLayoutHandle layout0;
            nvrhi::BindingLayoutHandle layout1;
            nvrhi::BufferHandle psCb;
            nvrhi::BindingSetHandle set0;
            nvrhi::BindingSetHandle set1;
        };

        MeshResources createMeshResources(TestFixture& fix, TestContext& ctx, uint32_t width, uint32_t height)
        {
            MeshResources res;

            nvrhi::TextureDesc rtDesc;
            rtDesc.setWidth(width)
                .setHeight(height)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setIsRenderTarget(true)
                .setDebugName("uaa_mesh_rt");
            rtDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget);
            res.colorTarget = fix.device()->createTexture(rtDesc);
            UAA_CHECK(res.colorTarget != nullptr, "Failed to create mesh colour target");

            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(res.colorTarget);
            res.fb = fix.device()->createFramebuffer(fbDesc);
            UAA_CHECK(res.fb != nullptr, "Failed to create mesh framebuffer");

            nvrhi::BindingLayoutDesc l0Desc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics, 0, true);
            l0Desc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(MeshPushColor)));
            res.layout0 = fix.device()->createBindingLayout(l0Desc);
            UAA_CHECK(res.layout0 != nullptr, "Failed to create mesh push-constant layout0");

            nvrhi::BindingLayoutDesc l1Desc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Pixel, 1, true);
            l1Desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            res.layout1 = fix.device()->createBindingLayout(l1Desc);
            UAA_CHECK(res.layout1 != nullptr, "Failed to create mesh pixel CB layout1");

            nvrhi::BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(PsColorBias))
                .setIsConstantBuffer(true)
                .setDebugName("uaa_mesh_ps_cb");
            cbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer);
            res.psCb = fix.device()->createBuffer(cbDesc);
            UAA_CHECK(res.psCb != nullptr, "Failed to create mesh pixel constant buffer");

            PsColorBias cbData{};
            cbData.colorBias[0] = 0.0f;
            cbData.colorBias[1] = 0.0f;
            cbData.colorBias[2] = 0.0f;
            cbData.colorBias[3] = 0.0f;
            fix.uploadBuffer(res.psCb, &cbData, sizeof(cbData));

            nvrhi::BindingSetDesc s0Desc;
            s0Desc.addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(MeshPushColor)));
            res.set0 = fix.device()->createBindingSet(s0Desc, res.layout0);
            UAA_CHECK(res.set0 != nullptr, "Failed to create mesh set0");

            nvrhi::BindingSetDesc s1Desc;
            s1Desc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, res.psCb));
            res.set1 = fix.device()->createBindingSet(s1Desc, res.layout1);
            UAA_CHECK(res.set1 != nullptr, "Failed to create mesh set1");

            return res;
        }

        void verifyCenterPixelRgba(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::ITexture* tex,
            uint32_t width,
            uint32_t height,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            const std::string& label)
        {
            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(tex, 0, 0, pixels);
            UAA_CHECK_EQ(pixels.size(), size_t(width) * size_t(height) * 4u, "Unexpected readback pixel buffer size");
            const size_t centerOffset = (size_t(height / 2u) * size_t(width) + size_t(width / 2u)) * 4u;
            std::string diff;
            UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[centerOffset], r, g, b, a, 2, diff),
                label + ": " + diff);
        }

        void testMshPipeline(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets not supported");
            }

            MeshResources res = createMeshResources(fix, ctx, 64, 64);

            nvrhi::ShaderHandle ms = fix.createShader("ms_triangle", nvrhi::ShaderType::Mesh);
            nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

            nvrhi::MeshletPipelineDesc psoDesc;
            psoDesc.setMeshShader(ms)
                .setPixelShader(ps)
                .addBindingLayout(res.layout0)
                .addBindingLayout(res.layout1);

            nvrhi::MeshletPipelineHandle psoFromInfo =
                fix.device()->createMeshletPipeline(psoDesc, res.fb->getFramebufferInfo());
            UAA_CHECK(psoFromInfo != nullptr, "createMeshletPipeline(desc, fbinfo) returned nullptr");
            UAA_CHECK(psoFromInfo->getDesc().MS.Get() == ms.Get(), "Mesh shader mismatch on fbinfo pipeline");

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
            nvrhi::MeshletPipelineHandle psoFromFb =
                fix.device()->createMeshletPipeline(psoDesc, res.fb);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
            UAA_CHECK(psoFromFb != nullptr, "createMeshletPipeline(desc, fb) returned nullptr");
            UAA_CHECK(psoFromFb->getDesc().PS.Get() == ps.Get(), "Pixel shader mismatch on fb pipeline");
        }

        void testMshDirect(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets not supported");
            }

            constexpr uint32_t kSize = 64;
            MeshResources res = createMeshResources(fix, ctx, kSize, kSize);

            nvrhi::ShaderHandle ms = fix.createShader("ms_triangle", nvrhi::ShaderType::Mesh);
            nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

            nvrhi::MeshletPipelineDesc psoDesc;
            psoDesc.setMeshShader(ms)
                .setPixelShader(ps)
                .addBindingLayout(res.layout0)
                .addBindingLayout(res.layout1);
            psoDesc.renderState.rasterState.setCullNone();

            nvrhi::MeshletPipelineHandle pso =
                fix.device()->createMeshletPipeline(psoDesc, res.fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "createMeshletPipeline returned nullptr");

            const MeshPushColor pushColor{ { 0.25f, 0.50f, 0.75f, 1.0f } };

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            nvrhi::utils::ClearColorAttachment(cmd, res.fb, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));

            nvrhi::MeshletState state;
            state.setPipeline(pso)
                .setFramebuffer(res.fb)
                .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                .addBindingSet(res.set0)
                .addBindingSet(res.set1);
            cmd->setMeshletState(state);
            cmd->setPushConstants(&pushColor, sizeof(pushColor));
            cmd->dispatchMesh(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 64, 128, 191, 255, "MSH.Direct");
        }

        void testMshTaskPayload(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets not supported");
            }

            constexpr uint32_t kSize = 64;
            MeshResources res = createMeshResources(fix, ctx, kSize, kSize);

            nvrhi::ShaderHandle as = fix.createShader("as_task", nvrhi::ShaderType::Amplification);
            nvrhi::ShaderHandle ms = fix.createShader("ms_triangle_payload", nvrhi::ShaderType::Mesh);
            nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

            nvrhi::MeshletPipelineDesc psoDesc;
            psoDesc.setAmplificationShader(as)
                .setMeshShader(ms)
                .setPixelShader(ps)
                .addBindingLayout(res.layout0)
                .addBindingLayout(res.layout1);
            psoDesc.renderState.rasterState.setCullNone();

            nvrhi::MeshletPipelineHandle pso =
                fix.device()->createMeshletPipeline(psoDesc, res.fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "createMeshletPipeline with AS+MS+PS returned nullptr");

            const MeshPushColor pushColor{ { 0.80f, 0.20f, 0.40f, 1.0f } };

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            nvrhi::utils::ClearColorAttachment(cmd, res.fb, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));

            nvrhi::MeshletState state;
            state.setPipeline(pso)
                .setFramebuffer(res.fb)
                .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                .addBindingSet(res.set0)
                .addBindingSet(res.set1);
            cmd->setMeshletState(state);
            cmd->setPushConstants(&pushColor, sizeof(pushColor));
            cmd->dispatchMesh(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 204, 51, 102, 255, "MSH.TaskPayload");
        }

        void testMshIndirect(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets not supported");
            }

            constexpr uint32_t kSize = 64;
            MeshResources res = createMeshResources(fix, ctx, kSize, kSize);

            nvrhi::ShaderHandle ms = fix.createShader("ms_triangle", nvrhi::ShaderType::Mesh);
            nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

            nvrhi::MeshletPipelineDesc psoDesc;
            psoDesc.setMeshShader(ms)
                .setPixelShader(ps)
                .addBindingLayout(res.layout0)
                .addBindingLayout(res.layout1);
            psoDesc.renderState.rasterState.setCullNone();

            nvrhi::MeshletPipelineHandle pso =
                fix.device()->createMeshletPipeline(psoDesc, res.fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "createMeshletPipeline returned nullptr");

            nvrhi::ShaderHandle csGen = fix.createShader("cs_indirect_gen", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutDesc genLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute, 0, false);
            genLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1));
            nvrhi::BindingLayoutHandle genLayout = fix.device()->createBindingLayout(genLayoutDesc);
            UAA_CHECK(genLayout != nullptr, "Failed to create indirect generator binding layout");

            nvrhi::ComputePipelineDesc genPsoDesc;
            genPsoDesc.setComputeShader(csGen).addBindingLayout(genLayout);
            nvrhi::ComputePipelineHandle genPso = fix.device()->createComputePipeline(genPsoDesc);
            UAA_CHECK(genPso != nullptr, "Failed to create indirect generator compute pipeline");

            nvrhi::BufferDesc genCbDesc;
            genCbDesc.setByteSize(sizeof(IndirectGenParams))
                .setIsConstantBuffer(true)
                .setDebugName("uaa_mesh_indirect_gen_cb");
            genCbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer);
            nvrhi::BufferHandle genCb = fix.device()->createBuffer(genCbDesc);
            UAA_CHECK(genCb != nullptr, "Failed to create indirect gen CB");

            nvrhi::BufferDesc argsDesc;
            argsDesc.setByteSize(64)
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setIsDrawIndirectArgs(true)
                .setDebugName("uaa_mesh_indirect_args");
            argsDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::IndirectArgument);
            nvrhi::BufferHandle argsBuf = fix.device()->createBuffer(argsDesc);
            UAA_CHECK(argsBuf != nullptr, "Failed to create indirect args buffer");

            nvrhi::BufferDesc countDesc;
            countDesc.setByteSize(16)
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setIsDrawIndirectArgs(true)
                .setDebugName("uaa_mesh_indirect_count");
            countDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::IndirectArgument);
            nvrhi::BufferHandle countBuf = fix.device()->createBuffer(countDesc);
            UAA_CHECK(countBuf != nullptr, "Failed to create indirect count buffer");

            nvrhi::BindingSetDesc genArgsSetDesc;
            genArgsSetDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, genCb))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, argsBuf));
            nvrhi::BindingSetHandle genArgsSet = fix.device()->createBindingSet(genArgsSetDesc, genLayout);
            UAA_CHECK(genArgsSet != nullptr, "Failed to create indirect args gen binding set");

            nvrhi::BindingSetDesc genCountSetDesc;
            genCountSetDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, genCb))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, countBuf));
            nvrhi::BindingSetHandle genCountSet = fix.device()->createBindingSet(genCountSetDesc, genLayout);
            UAA_CHECK(genCountSet != nullptr, "Failed to create indirect count gen binding set");

            const MeshPushColor pushColor{ { 0.30f, 0.70f, 0.20f, 1.0f } };

            auto runIndirectDraw = [&](uint32_t groupsX, bool useCountBuffer, uint32_t drawCount)
            {
                IndirectGenParams params{};
                params.word0 = groupsX;
                params.word1 = 1;
                params.word2 = 1;
                params.offsetWords = 0;
                params.count = 3;
                fix.uploadBuffer(genCb, &params, sizeof(params));

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();

                nvrhi::ComputeState csState;
                csState.setPipeline(genPso).addBindingSet(genArgsSet);
                cmd->setComputeState(csState);
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                if (useCountBuffer)
                {
                    IndirectGenParams countParams{};
                    countParams.word0 = drawCount;
                    countParams.offsetWords = 0;
                    countParams.count = 1;
                    fix.uploadBuffer(genCb, &countParams, sizeof(countParams));

                    nvrhi::CommandListHandle cmdCount = fix.device()->createCommandList();
                    UAA_CHECK(cmdCount != nullptr, "createCommandList returned nullptr");
                    cmdCount->open();
                    nvrhi::ComputeState csCountState;
                    csCountState.setPipeline(genPso).addBindingSet(genCountSet);
                    cmdCount->setComputeState(csCountState);
                    cmdCount->dispatch(1, 1, 1);
                    cmdCount->close();
                    fix.submitAndWait(cmdCount);
                }

                nvrhi::CommandListHandle drawCmd = fix.device()->createCommandList();
                UAA_CHECK(drawCmd != nullptr, "createCommandList returned nullptr");
                drawCmd->open();
                nvrhi::utils::ClearColorAttachment(drawCmd, res.fb, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));

                nvrhi::MeshletState state;
                state.setPipeline(pso)
                    .setFramebuffer(res.fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                    .addBindingSet(res.set0)
                    .addBindingSet(res.set1)
                    .setIndirectParams(argsBuf);
                if (useCountBuffer)
                {
                    state.setIndirectCountBuffer(countBuf);
                }
                drawCmd->setMeshletState(state);
                drawCmd->setPushConstants(&pushColor, sizeof(pushColor));
                if (useCountBuffer)
                {
                    drawCmd->dispatchMeshIndirectCount(0, 0, 1);
                }
                else
                {
                    drawCmd->dispatchMeshIndirect(0, 1);
                }
                drawCmd->close();
                fix.submitAndWait(drawCmd);
            };

            // Zero threadgroups should leave cleared black background intact.
            runIndirectDraw(0, false, 0);
            verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 0, 0, 0, 255, "MSH.Indirect(0 groups)");

            // One threadgroup should draw the tinted triangle.
            runIndirectDraw(1, false, 0);
            verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 77, 179, 51, 255, "MSH.Indirect(1 group)");

            if (fix.api() == nvrhi::GraphicsAPI::VULKAN)
            {
                runIndirectDraw(1, true, 0);
                verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 0, 0, 0, 255, "MSH.IndirectCount(0)");

                runIndirectDraw(1, true, 1);
                verifyCenterPixelRgba(fix, ctx, res.colorTarget, kSize, kSize, 77, 179, 51, 255, "MSH.IndirectCount(1)");
            }
        }

        void testMshLimits(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets not supported");
            }

            if (!fix.isValidationActive())
            {
                ctx.skip("Validation layer required for MSH.Limits negative checks");
            }

            MeshResources res = createMeshResources(fix, ctx, 64, 64);
            nvrhi::ShaderHandle ms = fix.createShader("ms_triangle", nvrhi::ShaderType::Mesh);
            nvrhi::ShaderHandle ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);

            {
                ExpectErrorsGuard expectErr(fix);
                nvrhi::MeshletPipelineDesc nullFbDesc;
                nullFbDesc.setMeshShader(ms).setPixelShader(ps).addBindingLayout(res.layout0).addBindingLayout(res.layout1);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
                nvrhi::MeshletPipelineHandle badPso = fix.device()->createMeshletPipeline(nullFbDesc, static_cast<nvrhi::IFramebuffer*>(nullptr));
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
                UAA_CHECK(badPso == nullptr, "Null framebuffer must fail meshlet pipeline creation");
                UAA_CHECK(expectErr.errorCount() > 0, "Expected validation error for null framebuffer");
            }

            {
                ExpectErrorsGuard expectErr(fix);
                nvrhi::MeshletPipelineDesc wrongStageDesc;
                wrongStageDesc.setMeshShader(ps).setPixelShader(ps).addBindingLayout(res.layout0).addBindingLayout(res.layout1);
                nvrhi::MeshletPipelineHandle badPso =
                    fix.device()->createMeshletPipeline(wrongStageDesc, res.fb->getFramebufferInfo());
                UAA_CHECK(badPso == nullptr, "Pixel shader in MS slot must fail meshlet pipeline creation");
                UAA_CHECK(expectErr.errorCount() > 0, "Expected validation error for wrong shader stage in MS");
            }

            {
                ExpectErrorsGuard expectErr(fix);
                nvrhi::MeshletPipelineDesc depthMismatchDesc;
                depthMismatchDesc.setMeshShader(ms).setPixelShader(ps).addBindingLayout(res.layout0).addBindingLayout(res.layout1);
                depthMismatchDesc.renderState.depthStencilState.setDepthTestEnable(true);
                nvrhi::MeshletPipelineHandle badPso =
                    fix.device()->createMeshletPipeline(depthMismatchDesc, res.fb->getFramebufferInfo());
                UAA_CHECK(badPso == nullptr, "Depth test without depth attachment must fail meshlet pipeline creation");
                UAA_CHECK(expectErr.errorCount() > 0, "Expected validation error for incompatible framebuffer depth state");
            }
        }

        void testMshUnsupported(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (fix.features().supports(nvrhi::Feature::Meshlets))
            {
                ctx.skip("Feature::Meshlets is supported on this device");
            }

            ctx.note("Feature::Meshlets is false on this device");

            const bool vkNativeValidation = (fix.api() == nvrhi::GraphicsAPI::VULKAN)
                && (fix.deviceInfo().nativeValidation || fix.options().nativeValidation);

            nvrhi::ShaderHandle ms;
            if (vkNativeValidation)
            {
                ctx.note("Skipped tryCreateShader(ms_triangle) on Vulkan with native validation active because Khronos validation rejects Capability MeshShadingEXT when VK_EXT_mesh_shader is not enabled");
            }
            else
            {
                ExpectErrorsGuard expectErr(fix);
                ms = fix.tryCreateShader("ms_triangle", nvrhi::ShaderType::Mesh);
                ctx.note(ms ? "ms_triangle shader module loaded" : "ms_triangle shader rejected cleanly");
            }

            if (fix.api() == nvrhi::GraphicsAPI::METAL || fix.api() == nvrhi::GraphicsAPI::D3D12)
            {
                nvrhi::ShaderHandle ps = fix.tryCreateShader("ps_vertex_color", nvrhi::ShaderType::Pixel);
                nvrhi::BindingLayoutDesc l0Desc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics, 0, true);
                l0Desc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(MeshPushColor)));
                nvrhi::BindingLayoutHandle l0 = fix.device()->createBindingLayout(l0Desc);

                nvrhi::BindingLayoutDesc l1Desc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Pixel, 1, true);
                l1Desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
                nvrhi::BindingLayoutHandle l1 = fix.device()->createBindingLayout(l1Desc);

                nvrhi::FramebufferInfo fbInfo;
                fbInfo.addColorFormat(nvrhi::Format::RGBA8_UNORM);

                nvrhi::MeshletPipelineDesc psoDesc;
                psoDesc.setMeshShader(ms).setPixelShader(ps);
                if (l0) psoDesc.addBindingLayout(l0);
                if (l1) psoDesc.addBindingLayout(l1);

                ExpectErrorsGuard expectErr(fix);
                nvrhi::MeshletPipelineHandle pso = fix.device()->createMeshletPipeline(psoDesc, fbInfo);
                UAA_CHECK(pso == nullptr, "createMeshletPipeline must return nullptr when Meshlets unsupported");
                ctx.note("Verified createMeshletPipeline returns nullptr cleanly when Meshlets unsupported");
            }
            else
            {
                ctx.note("Skipped raw createMeshletPipeline call on Vulkan/D3D11 because backend asserts NotSupported() when Meshlets is false");
            }
        }
    } // namespace

    UAA_REGISTER_TEST("MSH.Pipeline", "optional", testMshPipeline);
    UAA_REGISTER_TEST("MSH.Direct", "optional", testMshDirect);
    UAA_REGISTER_TEST("MSH.TaskPayload", "optional", testMshTaskPayload);
    UAA_REGISTER_TEST("MSH.Indirect", "optional", testMshIndirect);
    UAA_REGISTER_TEST("MSH.Limits", "optional,validation", testMshLimits);
    UAA_REGISTER_TEST("MSH.Unsupported", "optional", testMshUnsupported);
} // namespace nvrhi::uaa
