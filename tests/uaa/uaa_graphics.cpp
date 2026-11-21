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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct SolidPC
        {
            float color[4];
            float depth;
            float scale;
            float offset[2];
        };

        struct MrtPC
        {
            float color0[4];
            float color1[4];
        };

        struct VsCBData
        {
            float posScale[4];
        };

        struct PsCBData
        {
            float colorBias[4];
        };

        struct VertexData
        {
            float pos[3];
            float col[4];
        };

        struct InstanceData
        {
            float offset[2];
            float tint[4];
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
        };

        TestFixture& requireFixture(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "Graphics test requires a valid TestFixture");
            return *ctx.fixture;
        }

        nvrhi::TextureHandle createColorTarget(
            TestContext& ctx,
            nvrhi::IDevice* dev,
            uint32_t width,
            uint32_t height,
            nvrhi::Format format,
            const char* debugName,
            uint32_t arraySize = 1,
            uint32_t mipLevels = 1,
            uint32_t sampleCount = 1)
        {
            nvrhi::TextureDesc desc;
            desc.setWidth(width)
                .setHeight(height)
                .setArraySize(arraySize)
                .setMipLevels(mipLevels)
                .setSampleCount(sampleCount)
                .setFormat(format)
                .setIsRenderTarget(true)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
                .setDebugName(debugName);
            if (sampleCount > 1)
            {
                desc.setDimension(arraySize > 1 ? nvrhi::TextureDimension::Texture2DMSArray : nvrhi::TextureDimension::Texture2DMS);
            }
            else if (arraySize > 1)
            {
                desc.setDimension(nvrhi::TextureDimension::Texture2DArray);
            }
            else
            {
                desc.setDimension(nvrhi::TextureDimension::Texture2D);
            }
            nvrhi::TextureHandle tex = dev->createTexture(desc);
            UAA_CHECK(tex != nullptr, std::string("Failed to create colour target: ") + debugName);
            return tex;
        }

        nvrhi::Format pickDepthFormat(TestContext& ctx, nvrhi::IDevice* dev)
        {
            const nvrhi::Format candidates[] = {nvrhi::Format::D32, nvrhi::Format::D32S8, nvrhi::Format::D24S8, nvrhi::Format::D16};
            for (nvrhi::Format fmt : candidates)
            {
                if ((dev->queryFormatSupport(fmt) & nvrhi::FormatSupport::DepthStencil) == nvrhi::FormatSupport::DepthStencil)
                {
                    return fmt;
                }
            }
            UAA_CHECK(false, "No supported depth format found");
            return nvrhi::Format::UNKNOWN;
        }

        nvrhi::Format pickDepthStencilFormat(TestContext& ctx, nvrhi::IDevice* dev)
        {
            const nvrhi::Format candidates[] = {nvrhi::Format::D32S8, nvrhi::Format::D24S8};
            for (nvrhi::Format fmt : candidates)
            {
                if ((dev->queryFormatSupport(fmt) & nvrhi::FormatSupport::DepthStencil) == nvrhi::FormatSupport::DepthStencil)
                {
                    return fmt;
                }
            }
            UAA_CHECK(false, "No supported depth-stencil format found");
            return nvrhi::Format::UNKNOWN;
        }

        nvrhi::TextureHandle createDepthTarget(
            TestContext& ctx,
            nvrhi::IDevice* dev,
            uint32_t width,
            uint32_t height,
            nvrhi::Format format,
            const char* debugName)
        {
            nvrhi::TextureDesc desc;
            desc.setWidth(width)
                .setHeight(height)
                .setFormat(format)
                .setIsRenderTarget(true)
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
                .setDebugName(debugName);
            nvrhi::TextureHandle tex = dev->createTexture(desc);
            UAA_CHECK(tex != nullptr, std::string("Failed to create depth target: ") + debugName);
            return tex;
        }

        const uint8_t* pixelAt(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t x, uint32_t y)
        {
            return &pixels[(size_t(y) * width + x) * 4u];
        }

        void checkPixel(
            TestContext& ctx,
            const std::vector<uint8_t>& pixels,
            uint32_t width,
            uint32_t x,
            uint32_t y,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            uint8_t tolerance,
            const char* label)
        {
            std::string diff;
            if (!TestFixture::checkPixelRgba8(pixelAt(pixels, width, x, y), r, g, b, a, tolerance, diff))
            {
                std::ostringstream oss;
                oss << label << " at (" << x << "," << y << "): " << diff;
                UAA_CHECK(false, oss.str());
            }
        }

        // Generates 6 clockwise vertices in screen space for an axis-aligned NDC rectangle.
        void appendQuadVertices(
            std::vector<VertexData>& out,
            float minX,
            float maxX,
            float minY,
            float maxY,
            float z,
            float r,
            float g,
            float b,
            float a,
            bool counterClockwise = false)
        {
            const VertexData tl{{minX, maxY, z}, {r, g, b, a}};
            const VertexData tr{{maxX, maxY, z}, {r, g, b, a}};
            const VertexData br{{maxX, minY, z}, {r, g, b, a}};
            const VertexData bl{{minX, minY, z}, {r, g, b, a}};
            if (!counterClockwise)
            {
                out.push_back(tl);
                out.push_back(tr);
                out.push_back(br);
                out.push_back(tl);
                out.push_back(br);
                out.push_back(bl);
            }
            else
            {
                out.push_back(tl);
                out.push_back(br);
                out.push_back(tr);
                out.push_back(tl);
                out.push_back(bl);
                out.push_back(br);
            }
        }

        struct SolidPipelineBundle
        {
            nvrhi::ShaderHandle vs;
            nvrhi::ShaderHandle ps;
            nvrhi::BindingLayoutHandle layout;
            nvrhi::BindingSetHandle bindingSet;
        };

        nvrhi::RenderState makeColorOnlyRenderState(nvrhi::RenderState state)
        {
            state.setDepthStencilState(
                nvrhi::DepthStencilState().setDepthTestEnable(false).setDepthWriteEnable(false).setStencilEnable(false));
            return state;
        }

        SolidPipelineBundle makeSolidBundle(TestContext& ctx, TestFixture& fix, size_t pushSize = sizeof(SolidPC))
        {
            nvrhi::IDevice* dev = fix.device();
            SolidPipelineBundle bundle;
            bundle.vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            bundle.ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);
            UAA_CHECK(bundle.vs != nullptr && bundle.ps != nullptr, "Failed to create solid shaders");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, pushSize));
            bundle.layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(bundle.layout != nullptr, "Failed to create solid push-constant binding layout");

            bundle.bindingSet = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, pushSize)), bundle.layout);
            UAA_CHECK(bundle.bindingSet != nullptr, "Failed to create empty binding set for solid layout");
            return bundle;
        }

        struct VertexColorBundle
        {
            nvrhi::ShaderHandle vs;
            nvrhi::ShaderHandle ps;
            nvrhi::InputLayoutHandle inputLayout;
            nvrhi::BindingLayoutHandle layout0;
            nvrhi::BindingLayoutHandle layout1;
            nvrhi::BufferHandle vsCb;
            nvrhi::BufferHandle psCb;
            nvrhi::BindingSetHandle bs0;
            nvrhi::BindingSetHandle bs1;
        };

        VertexColorBundle makeVertexColorBundle(
            TestContext& ctx,
            TestFixture& fix,
            const VsCBData& vsData,
            const PsCBData& psData)
        {
            nvrhi::IDevice* dev = fix.device();
            VertexColorBundle b;
            b.vs = fix.createShader("vs_vertex_input", nvrhi::ShaderType::Vertex);
            b.ps = fix.createShader("ps_vertex_color", nvrhi::ShaderType::Pixel);
            UAA_CHECK(b.vs != nullptr && b.ps != nullptr, "Failed to create vertex input shaders");

            const nvrhi::VertexAttributeDesc attrs[] = {
                nvrhi::VertexAttributeDesc()
                    .setName("POSITION")
                    .setFormat(nvrhi::Format::RGB32_FLOAT)
                    .setBufferIndex(0)
                    .setOffset(offsetof(VertexData, pos))
                    .setElementStride(sizeof(VertexData))
                    .setIsInstanced(false),
                nvrhi::VertexAttributeDesc()
                    .setName("COLOR0")
                    .setFormat(nvrhi::Format::RGBA32_FLOAT)
                    .setBufferIndex(0)
                    .setOffset(offsetof(VertexData, col))
                    .setElementStride(sizeof(VertexData))
                    .setIsInstanced(false),
                nvrhi::VertexAttributeDesc()
                    .setName("TEXCOORD0")
                    .setFormat(nvrhi::Format::RG32_FLOAT)
                    .setBufferIndex(1)
                    .setOffset(offsetof(InstanceData, offset))
                    .setElementStride(sizeof(InstanceData))
                    .setIsInstanced(true),
                nvrhi::VertexAttributeDesc()
                    .setName("COLOR1")
                    .setFormat(nvrhi::Format::RGBA32_FLOAT)
                    .setBufferIndex(1)
                    .setOffset(offsetof(InstanceData, tint))
                    .setElementStride(sizeof(InstanceData))
                    .setIsInstanced(true),
            };
            b.inputLayout = dev->createInputLayout(attrs, 4u, b.vs);
            UAA_CHECK(b.inputLayout != nullptr, "Failed to create input layout");

            // Metal compares layout visibility against the shader stage, so set 0 uses Vertex and set 1 uses Pixel.
            b.layout0 = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Vertex, 0, true)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0)));
            b.layout1 = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Pixel, 1, true)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0)));
            UAA_CHECK(b.layout0 != nullptr && b.layout1 != nullptr, "Failed to create vertex-color binding layouts");

            b.vsCb = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(VsCBData))
                    .setIsConstantBuffer(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
                    .setDebugName("gfx_vscb"));
            b.psCb = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(sizeof(PsCBData))
                    .setIsConstantBuffer(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
                    .setDebugName("gfx_pscb"));
            UAA_CHECK(b.vsCb != nullptr && b.psCb != nullptr, "Failed to create vertex-color CBs");

            fix.uploadBuffer(b.vsCb, &vsData, sizeof(vsData));
            fix.uploadBuffer(b.psCb, &psData, sizeof(psData));

            b.bs0 = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::ConstantBuffer(0, b.vsCb)),
                b.layout0);
            b.bs1 = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::ConstantBuffer(0, b.psCb)),
                b.layout1);
            UAA_CHECK(b.bs0 != nullptr && b.bs1 != nullptr, "Failed to create vertex-color binding sets");
            return b;
        }

        void testGfxFramebuffer(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::Format depthFmt = pickDepthFormat(ctx, dev);
            nvrhi::TextureHandle colorTex = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_fb_color");
            nvrhi::TextureHandle depthTex = createDepthTarget(ctx, dev, 64, 64, depthFmt, "gfx_fb_depth");

            nvrhi::FramebufferHandle fbColorDepth = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(colorTex).setDepthAttachment(depthTex));
            UAA_CHECK(fbColorDepth != nullptr, "Failed to create colour+depth framebuffer");

            const nvrhi::FramebufferDesc& descCD = fbColorDepth->getDesc();
            UAA_CHECK_EQ(descCD.colorAttachments.size(), size_t(1), "Colour attachment count mismatch");
            UAA_CHECK(descCD.colorAttachments[0].texture == colorTex.Get(), "Colour attachment pointer mismatch");
            UAA_CHECK(descCD.depthAttachment.texture == depthTex.Get(), "Depth attachment pointer mismatch");

            const nvrhi::FramebufferInfoEx& infoCD = fbColorDepth->getFramebufferInfo();
            UAA_CHECK_EQ(infoCD.width, 64u, "FramebufferInfo width mismatch");
            UAA_CHECK_EQ(infoCD.height, 64u, "FramebufferInfo height mismatch");
            UAA_CHECK_EQ(infoCD.arraySize, 1u, "FramebufferInfo arraySize mismatch");
            UAA_CHECK_EQ(infoCD.colorFormats.size(), size_t(1), "FramebufferInfo colorFormats size mismatch");
            UAA_CHECK(infoCD.colorFormats[0] == nvrhi::Format::RGBA8_UNORM, "FramebufferInfo colorFormat mismatch");
            UAA_CHECK(infoCD.depthFormat == depthFmt, "FramebufferInfo depthFormat mismatch");
            UAA_CHECK_EQ(infoCD.sampleCount, 1u, "FramebufferInfo sampleCount mismatch");

            nvrhi::TextureHandle mrt0 = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA8_UNORM, "gfx_fb_mrt0");
            nvrhi::TextureHandle mrt1 = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA16_FLOAT, "gfx_fb_mrt1");
            nvrhi::FramebufferHandle fbMrt = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(mrt0).addColorAttachment(mrt1));
            UAA_CHECK(fbMrt != nullptr, "Failed to create MRT framebuffer");

            const nvrhi::FramebufferInfoEx& infoMrt = fbMrt->getFramebufferInfo();
            UAA_CHECK_EQ(infoMrt.width, 32u, "MRT width mismatch");
            UAA_CHECK_EQ(infoMrt.height, 32u, "MRT height mismatch");
            UAA_CHECK_EQ(infoMrt.colorFormats.size(), size_t(2), "MRT colorFormats count mismatch");
            UAA_CHECK(infoMrt.colorFormats[0] == nvrhi::Format::RGBA8_UNORM, "MRT format 0 mismatch");
            UAA_CHECK(infoMrt.colorFormats[1] == nvrhi::Format::RGBA16_FLOAT, "MRT format 1 mismatch");
            UAA_CHECK(infoMrt.depthFormat == nvrhi::Format::UNKNOWN, "MRT depthFormat should be UNKNOWN");

            nvrhi::TextureHandle arrTex = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_fb_arr", 4, 3);
            nvrhi::FramebufferHandle fbSlice = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(arrTex, nvrhi::TextureSubresourceSet(1, 1, 2, 1)));
            UAA_CHECK(fbSlice != nullptr, "Failed to create mip/slice framebuffer");

            const nvrhi::FramebufferInfoEx& infoSlice = fbSlice->getFramebufferInfo();
            UAA_CHECK_EQ(infoSlice.width, 32u, "Mip 1 framebuffer width mismatch");
            UAA_CHECK_EQ(infoSlice.height, 32u, "Mip 1 framebuffer height mismatch");
            UAA_CHECK_EQ(infoSlice.arraySize, 1u, "Single-slice framebuffer arraySize mismatch");

            // Render to MRT framebuffer using ps_mrt to verify both targets receive independent outputs.
            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle psMrt = fix.createShader("ps_mrt", nvrhi::ShaderType::Pixel);
            UAA_CHECK(vs != nullptr && psMrt != nullptr, "Failed to create MRT shaders");
            nvrhi::BindingLayoutHandle mrtLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(MrtPC))));
            UAA_CHECK(mrtLayout != nullptr, "Failed to create MRT binding layout");
            nvrhi::BindingSetHandle mrtBS = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(MrtPC))), mrtLayout);
            UAA_CHECK(mrtBS != nullptr, "Failed to create MRT binding set");

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.setVertexShader(vs)
                .setPixelShader(psMrt)
                .addBindingLayout(mrtLayout)
                .setRenderState(makeColorOnlyRenderState(
                    nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone())));
            nvrhi::GraphicsPipelineHandle psoMrt = dev->createGraphicsPipeline(psoDesc, fbMrt->getFramebufferInfo());
            UAA_CHECK(psoMrt != nullptr, "Failed to create MRT pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");
            // The shared fullscreen VS reads the first float4 as colour and the second as depth/scale/offset.
            // Keep the second target colour's components compatible with a fullscreen triangle.
            const MrtPC pc{{1.0f, 0.25f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 0.0f}};

            cmd->open();
            cmd->clearTextureFloat(mrt0, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            cmd->clearTextureFloat(mrt1, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoMrt)
                    .setFramebuffer(fbMrt)
                    .addBindingSet(mrtBS)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbMrt->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels0;
            fix.readbackTexture2D(mrt0, 0, 0, pixels0);
            checkPixel(ctx, pixels0, 32, 16, 16, 255, 64, 0, 255, 2, "GFX.Framebuffer MRT target0");

            std::vector<uint8_t> pixels1;
            fix.readbackTexture2D(mrt1, 0, 0, pixels1);
            UAA_CHECK_EQ(pixels1.size(), size_t(32u * 32u * 4u * sizeof(uint16_t)),
                "GFX.Framebuffer MRT target1 readback size mismatch");
            const uint16_t expectedTarget1[] = { 0x0000u, 0x3C00u, 0x0000u, 0x0000u };
            for (size_t pixel = 0; pixel < 32u * 32u; ++pixel)
            {
                for (size_t c = 0; c < 4u; ++c)
                {
                    uint16_t actual = 0;
                    const size_t offset = (pixel * 4u + c) * sizeof(uint16_t);
                    std::memcpy(&actual, pixels1.data() + offset, sizeof(actual));
                    UAA_CHECK_EQ(actual, expectedTarget1[c],
                        "GFX.Framebuffer MRT target1 half-float channel mismatch at pixel " + std::to_string(pixel));
                }
            }
        }

        void testGfxClear(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::Format dsFmt = pickDepthStencilFormat(ctx, dev);
            nvrhi::TextureHandle arrTex = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA8_UNORM, "gfx_clear_arr", 3, 3);
            nvrhi::TextureHandle colorTex = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA8_UNORM, "gfx_clear_rt");
            nvrhi::TextureHandle dsTex = createDepthTarget(ctx, dev, 32, 32, dsFmt, "gfx_clear_ds");

            nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(colorTex).setDepthAttachment(dsTex));
            UAA_CHECK(fb != nullptr, "Failed to create clear test framebuffer");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);
            nvrhi::DepthStencilState dsState;
            dsState.setDepthTestEnable(true)
                .setDepthWriteEnable(false)
                .setDepthFunc(nvrhi::ComparisonFunc::Less)
                .setStencilEnable(true)
                .setStencilRefValue(0x3Cu)
                .setFrontFaceStencil(nvrhi::DepthStencilState::StencilOpDesc().setStencilFunc(nvrhi::ComparisonFunc::Equal))
                .setBackFaceStencil(nvrhi::DepthStencilState::StencilOpDesc().setStencilFunc(nvrhi::ComparisonFunc::Equal));

            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(nvrhi::RenderState()
                                        .setRasterState(nvrhi::RasterState().setCullNone())
                                        .setDepthStencilState(dsState)),
                fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create clear verification pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC pc{{0.0f, 1.0f, 0.0f, 1.0f}, 0.4f, 1.0f, {0.0f, 0.0f}};

            cmd->open();
            cmd->clearTextureFloat(arrTex, nvrhi::AllSubresources, nvrhi::Color(0.25f, 0.5f, 0.75f, 1.0f));
            cmd->clearTextureFloat(arrTex, nvrhi::TextureSubresourceSet(1, 1, 1, 1), nvrhi::Color(1.0f, 0.0f, 0.5f, 1.0f));

            cmd->clearTextureFloat(colorTex, nvrhi::AllSubresources, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
            cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 0.6f, true, 0x3Cu);

            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fb)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> clearedSlice;
            std::vector<uint8_t> untouchedSlice0;
            std::vector<uint8_t> untouchedSlice2;
            std::vector<uint8_t> untouchedMip0;
            std::vector<uint8_t> rtPixels;

            fix.readbackTexture2D(arrTex, 1, 1, clearedSlice);
            fix.readbackTexture2D(arrTex, 0, 1, untouchedSlice0);
            fix.readbackTexture2D(arrTex, 2, 1, untouchedSlice2);
            fix.readbackTexture2D(arrTex, 1, 0, untouchedMip0);
            fix.readbackTexture2D(colorTex, 0, 0, rtPixels);

            checkPixel(ctx, clearedSlice, 16, 8, 8, 255, 0, 128, 255, 2, "GFX.Clear mip=1 slice=1");
            checkPixel(ctx, untouchedSlice0, 16, 8, 8, 64, 128, 191, 255, 2, "GFX.Clear untouched slice=0");
            checkPixel(ctx, untouchedSlice2, 16, 8, 8, 64, 128, 191, 255, 2, "GFX.Clear untouched slice=2");
            checkPixel(ctx, untouchedMip0, 32, 16, 16, 64, 128, 191, 255, 2, "GFX.Clear untouched mip=0");
            checkPixel(ctx, rtPixels, 32, 16, 16, 0, 255, 0, 255, 2, "GFX.Clear depth/stencil test pass");
        }

        void testGfxDraw(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle rtA = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_draw_tri");
            nvrhi::TextureHandle rtB = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_draw_idx");
            nvrhi::FramebufferHandle fbA = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtA));
            nvrhi::FramebufferHandle fbB = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtB));
            UAA_CHECK(fbA != nullptr && fbB != nullptr, "Failed to create draw framebuffers");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);
            nvrhi::GraphicsPipelineHandle psoTri = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone()))),
                fbA->getFramebufferInfo());
            UAA_CHECK(psoTri != nullptr, "Failed to create triangle pipeline");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{1.f, 1.f, 1.f, 1.f}}, PsCBData{{0.f, 0.f, 0.f, 0.f}});
            nvrhi::GraphicsPipelineHandle psoQuad = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(vc.vs)
                    .setPixelShader(vc.ps)
                    .setInputLayout(vc.inputLayout)
                    .addBindingLayout(vc.layout0)
                    .addBindingLayout(vc.layout1)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(false)))),
                fbB->getFramebufferInfo());
            UAA_CHECK(psoQuad != nullptr, "Failed to create indexed quad pipeline");

            const VertexData verts[4] = {
                {{-0.5f,  0.5f, 0.0f}, {0.0f, 0.5f, 1.0f, 1.0f}},
                {{ 0.5f,  0.5f, 0.0f}, {0.0f, 0.5f, 1.0f, 1.0f}},
                {{ 0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 1.0f, 1.0f}},
                {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 1.0f, 1.0f}},
            };
            const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
            const InstanceData inst{{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};

            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(verts)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_draw_vb"));
            nvrhi::BufferHandle ib = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(indices)).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("gfx_draw_ib"));
            nvrhi::BufferHandle instVb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(inst)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_draw_inst"));
            UAA_CHECK(vb != nullptr && ib != nullptr && instVb != nullptr, "Failed to create draw buffers");

            fix.uploadBuffer(vb, verts, sizeof(verts));
            fix.uploadBuffer(ib, indices, sizeof(indices));
            fix.uploadBuffer(instVb, &inst, sizeof(inst));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC triPC{{1.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f}};
            const nvrhi::Color bg(10.0f / 255.0f, 20.0f / 255.0f, 30.0f / 255.0f, 1.0f);

            cmd->open();
            cmd->clearTextureFloat(rtA, nvrhi::AllSubresources, bg);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoTri)
                    .setFramebuffer(fbA)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbA->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&triPC, sizeof(triPC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

            cmd->clearTextureFloat(rtB, nvrhi::AllSubresources, bg);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoQuad)
                    .setFramebuffer(fbB)
                    .addBindingSet(vc.bs0)
                    .addBindingSet(vc.bs1)
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                    .setIndexBuffer(nvrhi::IndexBufferBinding().setBuffer(ib).setFormat(nvrhi::Format::R16_UINT).setOffset(0))
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbB->getFramebufferInfo().getViewport())));
            cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(6));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixA;
            std::vector<uint8_t> pixB;
            fix.readbackTexture2D(rtA, 0, 0, pixA);
            fix.readbackTexture2D(rtB, 0, 0, pixB);

            checkPixel(ctx, pixA, 64, 32, 32, 255, 0, 0, 255, 2, "GFX.Draw scaled triangle interior");
            checkPixel(ctx, pixA, 64, 4, 4, 10, 20, 30, 255, 2, "GFX.Draw scaled triangle border TL");
            checkPixel(ctx, pixA, 64, 4, 60, 10, 20, 30, 255, 2, "GFX.Draw scaled triangle border BL");
            checkPixel(ctx, pixA, 64, 60, 4, 10, 20, 30, 255, 2, "GFX.Draw scaled triangle border TR");

            checkPixel(ctx, pixB, 64, 32, 32, 0, 128, 255, 255, 2, "GFX.Draw indexed quad interior");
            checkPixel(ctx, pixB, 64, 4, 4, 10, 20, 30, 255, 2, "GFX.Draw indexed quad border TL");
            checkPixel(ctx, pixB, 64, 60, 4, 10, 20, 30, 255, 2, "GFX.Draw indexed quad border TR");
            checkPixel(ctx, pixB, 64, 4, 60, 10, 20, 30, 255, 2, "GFX.Draw indexed quad border BL");
            checkPixel(ctx, pixB, 64, 60, 60, 10, 20, 30, 255, 2, "GFX.Draw indexed quad border BR");
        }

        void testGfxVertexInput(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle rt = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_vi_rt");
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
            UAA_CHECK(fb != nullptr, "Failed to create vertex input framebuffer");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{0.5f, 0.5f, 1.0f, 1.0f}}, PsCBData{{0.1f, 0.0f, 0.0f, 0.0f}});
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(vc.vs)
                    .setPixelShader(vc.ps)
                    .setInputLayout(vc.inputLayout)
                    .addBindingLayout(vc.layout0)
                    .addBindingLayout(vc.layout1)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(false)))),
                fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create vertex input pipeline");

            constexpr uint32_t kVb0Offset = 16u;
            constexpr uint32_t kVb1Offset = 24u;

            std::vector<VertexData> verts;
            appendQuadVertices(verts, -0.5f, 0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.8f, 0.4f, 1.0f);

            const InstanceData instances[2] = {
                {{-0.5f, 0.0f}, {0.5f, 1.0f, 0.5f, 1.0f}},
                {{ 0.5f, 0.0f}, {1.0f, 0.25f, 1.0f, 1.0f}},
            };

            std::vector<uint8_t> vb0Bytes(kVb0Offset + verts.size() * sizeof(VertexData), TestFixture::kGuardByte);
            std::memcpy(vb0Bytes.data() + kVb0Offset, verts.data(), verts.size() * sizeof(VertexData));

            std::vector<uint8_t> vb1Bytes(kVb1Offset + sizeof(instances), TestFixture::kGuardByte);
            std::memcpy(vb1Bytes.data() + kVb1Offset, instances, sizeof(instances));

            nvrhi::BufferHandle vb0 = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(vb0Bytes.size()).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_vi_vb0"));
            nvrhi::BufferHandle vb1 = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(vb1Bytes.size()).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_vi_vb1"));
            UAA_CHECK(vb0 != nullptr && vb1 != nullptr, "Failed to create vertex input buffers");

            fix.uploadBuffer(vb0, vb0Bytes.data(), vb0Bytes.size());
            fix.uploadBuffer(vb1, vb1Bytes.data(), vb1Bytes.size());

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fb)
                    .addBindingSet(vc.bs0)
                    .addBindingSet(vc.bs1)
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb0).setSlot(0).setOffset(kVb0Offset))
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb1).setSlot(1).setOffset(kVb1Offset))
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(6).setInstanceCount(2));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(rt, 0, 0, pixels);

            checkPixel(ctx, pixels, 64, 16, 32, 26, 204, 51, 255, 2, "GFX.VertexInput instance 0");
            checkPixel(ctx, pixels, 64, 48, 32, 26, 51, 102, 255, 2, "GFX.VertexInput instance 1");
            checkPixel(ctx, pixels, 64, 32, 32, 0, 0, 0, 255, 2, "GFX.VertexInput center gap");
        }

        void testGfxDrawArguments(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle rt = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_dargs_rt");
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
            UAA_CHECK(fb != nullptr, "Failed to create DrawArguments framebuffer");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{1.f, 1.f, 1.f, 1.f}}, PsCBData{{0.f, 0.f, 0.f, 0.f}});
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(vc.vs)
                    .setPixelShader(vc.ps)
                    .setInputLayout(vc.inputLayout)
                    .addBindingLayout(vc.layout0)
                    .addBindingLayout(vc.layout1)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(false)))),
                fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create DrawArguments pipeline");

            std::vector<VertexData> verts;
            // Slots [0..5]: dummy padding vertices for startVertexLocation=6.
            appendQuadVertices(verts, -0.9f, -0.1f, 0.1f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
            // Slots [6..11]: top-left red quad for non-indexed draw(startVertexLocation=6).
            appendQuadVertices(verts, -0.9f, -0.1f, 0.1f, 0.9f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);
            // Slots [12..15]: dummy padding vertices before baseVertex=16.
            for (int i = 0; i < 4; ++i)
            {
                verts.push_back(VertexData{{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}});
            }
            // Slots [16..19]: top-right green quad (indexed vertices 0..3 with baseVertex=16).
            verts.push_back(VertexData{{0.1f, 0.9f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
            verts.push_back(VertexData{{0.9f, 0.9f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
            verts.push_back(VertexData{{0.9f, 0.1f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
            verts.push_back(VertexData{{0.1f, 0.1f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
            // Slots [20..23]: bottom-left quad (indexed vertices 4..7 with baseVertex=16).
            verts.push_back(VertexData{{-0.9f, -0.1f, 0.0f}, {0.5f, 0.5f, 1.0f, 1.0f}});
            verts.push_back(VertexData{{-0.1f, -0.1f, 0.0f}, {0.5f, 0.5f, 1.0f, 1.0f}});
            verts.push_back(VertexData{{-0.1f, -0.9f, 0.0f}, {0.5f, 0.5f, 1.0f, 1.0f}});
            verts.push_back(VertexData{{-0.9f, -0.9f, 0.0f}, {0.5f, 0.5f, 1.0f, 1.0f}});

            const uint16_t indices[18] = {
                0, 0, 0, 0, 0, 0,
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7
            };

            const InstanceData instances[4] = {
                {{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}}, // startInstance=0 (used by non-indexed draw)
                {{0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f}}, // dummy skipped when startInstanceLocation=2
                {{0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}, // instance 0 of instanced draw -> blue at BL
                {{1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}}, // instance 1 of instanced draw -> olive at BR
            };

            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(verts.size() * sizeof(VertexData)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_dargs_vb"));
            nvrhi::BufferHandle instVb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(instances)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_dargs_inst"));
            nvrhi::BufferHandle ib = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(indices)).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("gfx_dargs_ib"));
            UAA_CHECK(vb != nullptr && instVb != nullptr && ib != nullptr, "Failed to create DrawArguments buffers");

            fix.uploadBuffer(vb, verts.data(), verts.size() * sizeof(VertexData));
            fix.uploadBuffer(instVb, instances, sizeof(instances));
            fix.uploadBuffer(ib, indices, sizeof(indices));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fb)
                    .addBindingSet(vc.bs0)
                    .addBindingSet(vc.bs1)
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                    .setIndexBuffer(nvrhi::IndexBufferBinding().setBuffer(ib).setFormat(nvrhi::Format::R16_UINT).setOffset(0))
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));

            // Non-indexed draw with startVertexLocation.
            cmd->draw(nvrhi::DrawArguments().setVertexCount(6).setStartVertexLocation(6).setInstanceCount(1).setStartInstanceLocation(0));

            // Indexed draw with startIndexLocation + baseVertex (startVertexLocation).
            cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(6).setStartIndexLocation(6).setStartVertexLocation(16).setInstanceCount(1).setStartInstanceLocation(0));

            // Instanced indexed draw with startIndexLocation + baseVertex + startInstanceLocation + instanceCount=2.
            cmd->drawIndexed(nvrhi::DrawArguments().setVertexCount(6).setStartIndexLocation(12).setStartVertexLocation(16).setInstanceCount(2).setStartInstanceLocation(2));

            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(rt, 0, 0, pixels);

            checkPixel(ctx, pixels, 64, 16, 16, 255, 0, 0, 255, 2, "GFX.DrawArguments TL startVertex");
            checkPixel(ctx, pixels, 64, 48, 16, 0, 255, 0, 255, 2, "GFX.DrawArguments TR startIndex+baseVertex");
            checkPixel(ctx, pixels, 64, 16, 48, 0, 0, 255, 255, 2, "GFX.DrawArguments BL startInstance+inst0");
            checkPixel(ctx, pixels, 64, 48, 48, 128, 128, 0, 255, 2, "GFX.DrawArguments BR startInstance+inst1");
        }

        void testGfxIndirect(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::ShaderHandle csGen = fix.createShader("cs_indirect_gen", nvrhi::ShaderType::Compute);
            UAA_CHECK(csGen != nullptr, "Failed to create cs_indirect_gen");

            nvrhi::BindingLayoutHandle genLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1)));
            UAA_CHECK(genLayout != nullptr, "Failed to create genLayout");
            nvrhi::ComputePipelineHandle psoGen = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csGen).addBindingLayout(genLayout));
            UAA_CHECK(psoGen != nullptr, "Failed to create psoGen");

            nvrhi::BufferHandle indBuf = dev->createBuffer(
                nvrhi::BufferDesc()
                    .setByteSize(128)
                    .setStructStride(4)
                    .setCanHaveUAVs(true)
                    .setIsDrawIndirectArgs(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
                    .setDebugName("gfx_ind_args"));
            UAA_CHECK(indBuf != nullptr, "Failed to create indBuf");

            auto makeGenCb = [&](const IndirectGenParams& p, const char* name)
            {
                nvrhi::BufferHandle cb = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(sizeof(p)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName(name));
                UAA_CHECK(cb != nullptr, std::string("Failed to create ") + name);
                fix.uploadBuffer(cb, &p, sizeof(p));
                nvrhi::BindingSetHandle bs = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, indBuf)),
                    genLayout);
                UAA_CHECK(bs != nullptr, "Failed to create gen binding set");
                return bs;
            };

            // Offset 16 bytes (word 4): DrawIndirectArguments {3, 1, 0, 0}
            nvrhi::BindingSetHandle bsGenDraw = makeGenCb(IndirectGenParams{3u, 1u, 0u, 0u, 0u, 4u, 4u, 0u}, "gfx_ind_cb0");
            // Offset 32 bytes (word 8): zero-draw DrawIndirectArguments {0, 1, 0, 0}
            nvrhi::BindingSetHandle bsGenZero = makeGenCb(IndirectGenParams{0u, 1u, 0u, 0u, 0u, 8u, 4u, 0u}, "gfx_ind_cb1");
            // Offset 48 bytes (word 12): DrawIndexedIndirectArguments {6, 1, 0, 0, 0}
            nvrhi::BindingSetHandle bsGenIdx = makeGenCb(IndirectGenParams{6u, 1u, 0u, 0u, 0u, 12u, 5u, 0u}, "gfx_ind_cb2");

            nvrhi::TextureHandle rtA = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ind_rtA");
            nvrhi::TextureHandle rtB = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ind_rtB");
            nvrhi::TextureHandle rtC = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ind_rtC");
            nvrhi::FramebufferHandle fbA = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtA));
            nvrhi::FramebufferHandle fbB = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtB));
            nvrhi::FramebufferHandle fbC = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtC));
            UAA_CHECK(fbA != nullptr && fbB != nullptr && fbC != nullptr, "Failed to create indirect framebuffers");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);
            nvrhi::GraphicsPipelineHandle psoSolid = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone()))),
                fbA->getFramebufferInfo());
            UAA_CHECK(psoSolid != nullptr, "Failed to create solid indirect pipeline");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{1.f, 1.f, 1.f, 1.f}}, PsCBData{{0.f, 0.f, 0.f, 0.f}});
            nvrhi::GraphicsPipelineHandle psoIdx = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(vc.vs)
                    .setPixelShader(vc.ps)
                    .setInputLayout(vc.inputLayout)
                    .addBindingLayout(vc.layout0)
                    .addBindingLayout(vc.layout1)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(false)))),
                fbC->getFramebufferInfo());
            UAA_CHECK(psoIdx != nullptr, "Failed to create indexed indirect pipeline");

            std::vector<VertexData> verts;
            appendQuadVertices(verts, -0.5f, 0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f);
            // Reuse 4 vertices and 6 indices cleanly.
            const VertexData quadVerts[4] = {verts[0], verts[1], verts[2], verts[5]};
            const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
            const InstanceData inst{{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};

            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(quadVerts)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_ind_vb"));
            nvrhi::BufferHandle instVb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(inst)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_ind_inst"));
            nvrhi::BufferHandle ib = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(indices)).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("gfx_ind_ib"));
            UAA_CHECK(vb != nullptr && instVb != nullptr && ib != nullptr, "Failed to create indirect draw buffers");

            fix.uploadBuffer(vb, quadVerts, sizeof(quadVerts));
            fix.uploadBuffer(instVb, &inst, sizeof(inst));
            fix.uploadBuffer(ib, indices, sizeof(indices));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC redPC{{1.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f}};
            const nvrhi::Color bgA(10.0f / 255.0f, 20.0f / 255.0f, 30.0f / 255.0f, 1.0f);
            const nvrhi::Color bgB(40.0f / 255.0f, 50.0f / 255.0f, 60.0f / 255.0f, 1.0f);

            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGenDraw));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGenZero));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGenIdx));
            cmd->dispatch(1u, 1u, 1u);

            cmd->clearTextureFloat(rtA, nvrhi::AllSubresources, bgA);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoSolid)
                    .setFramebuffer(fbA)
                    .addBindingSet(solid.bindingSet)
                    .setIndirectParams(indBuf)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbA->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&redPC, sizeof(redPC));
            cmd->drawIndirect(16u, 1u);

            cmd->clearTextureFloat(rtB, nvrhi::AllSubresources, bgB);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoSolid)
                    .setFramebuffer(fbB)
                    .addBindingSet(solid.bindingSet)
                    .setIndirectParams(indBuf)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbB->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&redPC, sizeof(redPC));
            cmd->drawIndirect(32u, 1u);

            cmd->clearTextureFloat(rtC, nvrhi::AllSubresources, bgA);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoIdx)
                    .setFramebuffer(fbC)
                    .addBindingSet(vc.bs0)
                    .addBindingSet(vc.bs1)
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                    .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                    .setIndexBuffer(nvrhi::IndexBufferBinding().setBuffer(ib).setFormat(nvrhi::Format::R16_UINT).setOffset(0))
                    .setIndirectParams(indBuf)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbC->getFramebufferInfo().getViewport())));
            cmd->drawIndexedIndirect(48u, 1u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixA, pixB, pixC;
            fix.readbackTexture2D(rtA, 0, 0, pixA);
            fix.readbackTexture2D(rtB, 0, 0, pixB);
            fix.readbackTexture2D(rtC, 0, 0, pixC);

            checkPixel(ctx, pixA, 64, 32, 32, 255, 0, 0, 255, 2, "GFX.Indirect drawIndirect interior");
            checkPixel(ctx, pixA, 64, 4, 4, 10, 20, 30, 255, 2, "GFX.Indirect drawIndirect border");
            checkPixel(ctx, pixB, 64, 32, 32, 40, 50, 60, 255, 2, "GFX.Indirect zero-draw preserved background");
            checkPixel(ctx, pixC, 64, 32, 32, 0, 255, 0, 255, 2, "GFX.Indirect drawIndexedIndirect interior");
            checkPixel(ctx, pixC, 64, 4, 4, 10, 20, 30, 255, 2, "GFX.Indirect drawIndexedIndirect border");
        }

        void testGfxIndirectCount(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (fix.api() == nvrhi::GraphicsAPI::VULKAN && !fix.deviceInfo().vulkanDrawIndirectCount)
            {
                ctx.skip("drawIndirectCount was unavailable on the Vulkan logical device");
            }
            nvrhi::IDevice* dev = fix.device();

            nvrhi::ShaderHandle csGen = fix.createShader("cs_indirect_gen", nvrhi::ShaderType::Compute);
            UAA_CHECK(csGen != nullptr, "Failed to create cs_indirect_gen");

            nvrhi::BindingLayoutHandle genLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1)));
            UAA_CHECK(genLayout != nullptr, "Failed to create genLayout");
            nvrhi::ComputePipelineHandle psoGen = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csGen).addBindingLayout(genLayout));
            UAA_CHECK(psoGen != nullptr, "Failed to create psoGen");

            nvrhi::BufferHandle indBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(64).setStructStride(4).setCanHaveUAVs(true).setIsDrawIndirectArgs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("gfx_ic_args"));
            nvrhi::BufferHandle countBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(64).setStructStride(4).setCanHaveUAVs(true).setIsDrawIndirectArgs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess).setDebugName("gfx_ic_count"));
            UAA_CHECK(indBuf != nullptr && countBuf != nullptr, "Failed to create indirect count buffers");

            const nvrhi::DrawIndexedIndirectArguments drawArgs[2] = {
                nvrhi::DrawIndexedIndirectArguments().setIndexCount(6).setInstanceCount(1).setStartIndexLocation(0).setBaseVertexLocation(0).setStartInstanceLocation(0),
                nvrhi::DrawIndexedIndirectArguments().setIndexCount(6).setInstanceCount(1).setStartIndexLocation(6).setBaseVertexLocation(0).setStartInstanceLocation(0),
            };
            fix.uploadBuffer(indBuf, drawArgs, sizeof(drawArgs));

            auto makeCountGen = [&](uint32_t value, uint32_t offsetWords, const char* name)
            {
                const IndirectGenParams p{value, 0u, 0u, 0u, 0u, offsetWords, 1u, 0u};
                nvrhi::BufferHandle cb = dev->createBuffer(
                    nvrhi::BufferDesc().setByteSize(sizeof(p)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer).setDebugName(name));
                UAA_CHECK(cb != nullptr, std::string("Failed to create ") + name);
                fix.uploadBuffer(cb, &p, sizeof(p));
                nvrhi::BindingSetHandle bs = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, countBuf)),
                    genLayout);
                UAA_CHECK(bs != nullptr, "Failed to create count gen binding set");
                return bs;
            };

            nvrhi::BindingSetHandle bsCnt0 = makeCountGen(0u, 0u, "gfx_ic_cnt0");
            nvrhi::BindingSetHandle bsCnt1 = makeCountGen(1u, 4u, "gfx_ic_cnt1");
            nvrhi::BindingSetHandle bsCnt5 = makeCountGen(5u, 8u, "gfx_ic_cnt5");

            nvrhi::TextureHandle rt0 = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ic_rt0");
            nvrhi::TextureHandle rt1 = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ic_rt1");
            nvrhi::TextureHandle rt2 = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ic_rt2");
            nvrhi::FramebufferHandle fb0 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt0));
            nvrhi::FramebufferHandle fb1 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt1));
            nvrhi::FramebufferHandle fb2 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt2));
            UAA_CHECK(fb0 != nullptr && fb1 != nullptr && fb2 != nullptr, "Failed to create indirect count framebuffers");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{1.f, 1.f, 1.f, 1.f}}, PsCBData{{0.f, 0.f, 0.f, 0.f}});
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(vc.vs)
                    .setPixelShader(vc.ps)
                    .setInputLayout(vc.inputLayout)
                    .addBindingLayout(vc.layout0)
                    .addBindingLayout(vc.layout1)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone().setFrontCounterClockwise(false)))),
                fb0->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create indirect count pipeline");

            const VertexData verts[8] = {
                {{-0.9f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
                {{-0.1f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
                {{-0.1f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
                {{-0.9f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
                {{ 0.1f,  0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
                {{ 0.9f,  0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
                {{ 0.9f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
                {{ 0.1f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
            };
            const uint16_t indices[12] = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};
            const InstanceData inst{{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};

            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(verts)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_ic_vb"));
            nvrhi::BufferHandle instVb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(inst)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_ic_inst"));
            nvrhi::BufferHandle ib = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(indices)).setIsIndexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::IndexBuffer).setDebugName("gfx_ic_ib"));
            UAA_CHECK(vb != nullptr && instVb != nullptr && ib != nullptr, "Failed to create indirect count geometry buffers");

            fix.uploadBuffer(vb, verts, sizeof(verts));
            fix.uploadBuffer(instVb, &inst, sizeof(inst));
            fix.uploadBuffer(ib, indices, sizeof(indices));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const nvrhi::Color bg(10.0f / 255.0f, 20.0f / 255.0f, 30.0f / 255.0f, 1.0f);
            auto recordPass = [&](nvrhi::ITexture* rt, nvrhi::IFramebuffer* fb, uint32_t countOffset)
            {
                cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
                cmd->setGraphicsState(
                    nvrhi::GraphicsState()
                        .setPipeline(pso)
                        .setFramebuffer(fb)
                        .addBindingSet(vc.bs0)
                        .addBindingSet(vc.bs1)
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                        .setIndexBuffer(nvrhi::IndexBufferBinding().setBuffer(ib).setFormat(nvrhi::Format::R16_UINT).setOffset(0))
                        .setIndirectParams(indBuf)
                        .setIndirectCountBuffer(countBuf)
                        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));
                if (fix.api() == nvrhi::GraphicsAPI::METAL && !fix.isValidationActive() && countOffset == 0)
                {
                    ExpectErrorsGuard guard(fix);
                    cmd->drawIndexedIndirectCount(2u, 0u, 1u);
                    cmd->drawIndexedIndirectCount(48u, 0u, 1u);
                    cmd->drawIndexedIndirectCount(0u, 64u, 1u);
                    cmd->drawIndexedIndirectCount(0u, 0u, UINT32_MAX);
                    UAA_CHECK(guard.errorCount() == 4, "Invalid indirect ranges must be rejected");
                }
                cmd->drawIndexedIndirectCount(0u, countOffset, 2u);
            };

            cmd->open();
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsCnt0));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsCnt1));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsCnt5));
            cmd->dispatch(1u, 1u, 1u);

            recordPass(rt0, fb0, 0u);
            recordPass(rt1, fb1, 16u);
            recordPass(rt2, fb2, 32u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pix0, pix1, pix2;
            fix.readbackTexture2D(rt0, 0, 0, pix0);
            fix.readbackTexture2D(rt1, 0, 0, pix1);
            fix.readbackTexture2D(rt2, 0, 0, pix2);

            if (fix.api() == nvrhi::GraphicsAPI::D3D11)
            {
                // D3D11 uses maxDrawCount and ignores the GPU count buffer.
                for (const auto* pixels : {&pix0, &pix1, &pix2})
                {
                    checkPixel(ctx, *pixels, 64, 16, 32, 255, 0, 0, 255, 2, "GFX.IndirectCount fallback left");
                    checkPixel(ctx, *pixels, 64, 48, 32, 0, 255, 0, 255, 2, "GFX.IndirectCount fallback right");
                }
                return;
            }

            checkPixel(ctx, pix0, 64, 16, 32, 10, 20, 30, 255, 2, "GFX.IndirectCount count=0 left");
            checkPixel(ctx, pix0, 64, 48, 32, 10, 20, 30, 255, 2, "GFX.IndirectCount count=0 right");

            checkPixel(ctx, pix1, 64, 16, 32, 255, 0, 0, 255, 2, "GFX.IndirectCount count=1 left");
            checkPixel(ctx, pix1, 64, 48, 32, 10, 20, 30, 255, 2, "GFX.IndirectCount count=1 right");

            checkPixel(ctx, pix2, 64, 16, 32, 255, 0, 0, 255, 2, "GFX.IndirectCount clamped=2 left");
            checkPixel(ctx, pix2, 64, 48, 32, 0, 255, 0, 255, 2, "GFX.IndirectCount clamped=2 right");
        }

        void testGfxRasterState(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::Format depthFmt = pickDepthFormat(ctx, dev);
            nvrhi::TextureHandle rt = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_raster_rt");
            nvrhi::TextureHandle depthTex = createDepthTarget(ctx, dev, 64, 64, depthFmt, "gfx_raster_depth");
            nvrhi::FramebufferHandle fbColor = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
            nvrhi::FramebufferHandle fbDepth = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt).setDepthAttachment(depthTex));
            UAA_CHECK(fbColor != nullptr && fbDepth != nullptr, "Failed to create raster framebuffers");

            VertexColorBundle vc = makeVertexColorBundle(ctx, fix, VsCBData{{1.f, 1.f, 1.f, 1.f}}, PsCBData{{0.f, 0.f, 0.f, 0.f}});
            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);

            auto makeVcPso = [&](const nvrhi::RenderState& rs, const nvrhi::FramebufferInfo& fbInfo)
            {
                nvrhi::RenderState renderState = rs;
                if (fbInfo.depthFormat == nvrhi::Format::UNKNOWN)
                {
                    renderState = makeColorOnlyRenderState(renderState);
                }
                nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                    nvrhi::GraphicsPipelineDesc()
                        .setVertexShader(vc.vs)
                        .setPixelShader(vc.ps)
                        .setInputLayout(vc.inputLayout)
                        .addBindingLayout(vc.layout0)
                        .addBindingLayout(vc.layout1)
                        .setRenderState(renderState),
                    fbInfo);
                UAA_CHECK(pso != nullptr, "Failed to create raster vertex-color pipeline");
                return pso;
            };

            nvrhi::GraphicsPipelineHandle psoCullBackCw = makeVcPso(
                nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(false)),
                fbColor->getFramebufferInfo());
            nvrhi::GraphicsPipelineHandle psoCullFrontCw = makeVcPso(
                nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullFront().setFrontCounterClockwise(false)),
                fbColor->getFramebufferInfo());
            nvrhi::GraphicsPipelineHandle psoCullNone = makeVcPso(
                nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone()),
                fbColor->getFramebufferInfo());
            nvrhi::GraphicsPipelineHandle psoWire = makeVcPso(
                nvrhi::RenderState().setRasterState(nvrhi::RasterState().setFillWireframe().setCullNone()),
                fbColor->getFramebufferInfo());

            nvrhi::GraphicsPipelineHandle psoFccw = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullBack().setFrontCounterClockwise(true)))),
                fbColor->getFramebufferInfo());
            nvrhi::GraphicsPipelineHandle psoScissor = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone().setScissorEnable(true)))),
                fbColor->getFramebufferInfo());
            UAA_CHECK(psoFccw != nullptr && psoScissor != nullptr, "Failed to create solid raster pipelines");

            auto makeBiasPso = [&](int bias)
            {
                return makeVcPso(
                    nvrhi::RenderState()
                        .setRasterState(nvrhi::RasterState().setCullNone().setDepthBias(bias))
                        .setDepthStencilState(nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(true).setDepthFunc(nvrhi::ComparisonFunc::Less)),
                    fbDepth->getFramebufferInfo());
            };
            nvrhi::GraphicsPipelineHandle psoBiasZero = makeBiasPso(0);
            nvrhi::GraphicsPipelineHandle psoBiasNear = makeBiasPso(-1000);
            nvrhi::GraphicsPipelineHandle psoBiasFar = makeBiasPso(1000);

            std::vector<VertexData> verts;
            // [0..5]: CW red on left
            appendQuadVertices(verts, -0.9f, -0.1f, -0.8f, 0.8f, 0.5f, 1.f, 0.f, 0.f, 1.f, false);
            // [6..11]: CCW green on right
            appendQuadVertices(verts, 0.1f, 0.9f, -0.8f, 0.8f, 0.5f, 0.f, 1.f, 0.f, 1.f, true);
            // [12..17]: Full quad red for depthBias=0
            appendQuadVertices(verts, -0.8f, 0.8f, -0.8f, 0.8f, 0.5f, 1.f, 0.f, 0.f, 1.f, false);
            // [18..23]: Full quad green for depthBias=-1000
            appendQuadVertices(verts, -0.8f, 0.8f, -0.8f, 0.8f, 0.5f, 0.f, 1.f, 0.f, 1.f, false);
            // [24..29]: Full quad blue for depthBias=+1000
            appendQuadVertices(verts, -0.8f, 0.8f, -0.8f, 0.8f, 0.5f, 0.f, 0.f, 1.f, 1.f, false);

            const InstanceData inst{{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
            nvrhi::BufferHandle vb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(verts.size() * sizeof(VertexData)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_raster_vb"));
            nvrhi::BufferHandle instVb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(inst)).setIsVertexBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::VertexBuffer).setDebugName("gfx_raster_inst"));
            UAA_CHECK(vb != nullptr && instVb != nullptr, "Failed to create raster buffers");
            fix.uploadBuffer(vb, verts.data(), verts.size() * sizeof(VertexData));
            fix.uploadBuffer(instVb, &inst, sizeof(inst));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            auto runVc = [&](nvrhi::IGraphicsPipeline* pso, uint32_t startVertex)
            {
                cmd->setGraphicsState(
                    nvrhi::GraphicsState()
                        .setPipeline(pso)
                        .setFramebuffer(fbColor)
                        .addBindingSet(vc.bs0)
                        .addBindingSet(vc.bs1)
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbColor->getFramebufferInfo().getViewport())));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(6).setStartVertexLocation(startVertex));
            };

            const nvrhi::Color bg(10.0f / 255.0f, 20.0f / 255.0f, 30.0f / 255.0f, 1.0f);

            // Part 1: CullBack / CullFront / CullNone
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            runVc(psoCullBackCw, 0u);  // CW drawn
            runVc(psoCullBackCw, 6u);  // CCW culled
            cmd->close();
            fix.submitAndWait(cmd);
            std::vector<uint8_t> pix;
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 255, 0, 0, 255, 2, "GFX.RasterState CullBack CW drawn");
            checkPixel(ctx, pix, 64, 48, 32, 10, 20, 30, 255, 2, "GFX.RasterState CullBack CCW culled");

            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            runVc(psoCullFrontCw, 0u); // CW culled
            runVc(psoCullFrontCw, 6u); // CCW drawn
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 10, 20, 30, 255, 2, "GFX.RasterState CullFront CW culled");
            checkPixel(ctx, pix, 64, 48, 32, 0, 255, 0, 255, 2, "GFX.RasterState CullFront CCW drawn");

            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            runVc(psoCullNone, 0u);
            runVc(psoCullNone, 6u);
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 255, 0, 0, 255, 2, "GFX.RasterState CullNone left");
            checkPixel(ctx, pix, 64, 48, 32, 0, 255, 0, 255, 2, "GFX.RasterState CullNone right");

            // Part 2: frontCounterClockwise = true with vs_fullscreen_tri (CCW in screen space)
            const SolidPC whitePC{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 1.0f, {0.0f, 0.0f}};
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoFccw)
                    .setFramebuffer(fbColor)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbColor->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&whitePC, sizeof(whitePC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 32, 32, 255, 255, 255, 255, 2, "GFX.RasterState frontCounterClockwise=true");

            // Part 3: FillWireframe leaves quad interior untouched
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            runVc(psoWire, 12u);
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 10, 20, 30, 255, 2, "GFX.RasterState Wireframe hollow interior");

            // Part 4: depthBias
            auto runBias = [&](nvrhi::IGraphicsPipeline* pso, uint32_t startVertex)
            {
                cmd->setGraphicsState(
                    nvrhi::GraphicsState()
                        .setPipeline(pso)
                        .setFramebuffer(fbDepth)
                        .addBindingSet(vc.bs0)
                        .addBindingSet(vc.bs1)
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(vb).setSlot(0).setOffset(0))
                        .addVertexBuffer(nvrhi::VertexBufferBinding().setBuffer(instVb).setSlot(1).setOffset(0))
                        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbDepth->getFramebufferInfo().getViewport())));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(6).setStartVertexLocation(startVertex));
            };
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->clearDepthStencilTexture(depthTex, nvrhi::AllSubresources, true, 1.0f, false, 0u);
            runBias(psoBiasZero, 12u); // red
            runBias(psoBiasNear, 18u); // green (-1000 closer -> wins)
            runBias(psoBiasFar, 24u);  // blue (+1000 farther -> rejected)
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 32, 32, 0, 255, 0, 255, 2, "GFX.RasterState depthBias winner");

            // Part 5: half-screen viewport + sub-scissor rect
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoScissor)
                    .setFramebuffer(fbColor)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState()
                                     .addViewport(nvrhi::Viewport(0.f, 32.f, 0.f, 64.f, 0.f, 1.f))
                                     .addScissorRect(nvrhi::Rect(0, 32, 16, 48))));
            cmd->setPushConstants(&whitePC, sizeof(whitePC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 255, 255, 255, 255, 2, "GFX.RasterState inside viewport+scissor");
            checkPixel(ctx, pix, 64, 48, 32, 10, 20, 30, 255, 2, "GFX.RasterState outside viewport");
            checkPixel(ctx, pix, 64, 16, 8, 10, 20, 30, 255, 2, "GFX.RasterState outside scissor");
        }

        void testGfxDepthStencil(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::Format dsFmt = pickDepthStencilFormat(ctx, dev);
            nvrhi::TextureHandle rt = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_ds_rt");
            nvrhi::TextureHandle dsTex = createDepthTarget(ctx, dev, 64, 64, dsFmt, "gfx_ds_depth");

            nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(rt).setDepthAttachment(dsTex));
            nvrhi::FramebufferHandle fbReadOnly = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(rt).setDepthAttachment(
                    nvrhi::FramebufferAttachment().setTexture(dsTex).setReadOnly(true)));
            UAA_CHECK(fb != nullptr && fbReadOnly != nullptr, "Failed to create depth-stencil framebuffers");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);

            auto makeDsPso = [&](const nvrhi::DepthStencilState& ds, nvrhi::ColorMask mask = nvrhi::ColorMask::All)
            {
                nvrhi::BlendState bs;
                bs.setRenderTarget(0, nvrhi::BlendState::RenderTarget().setColorWriteMask(mask));
                nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                    nvrhi::GraphicsPipelineDesc()
                        .setVertexShader(solid.vs)
                        .setPixelShader(solid.ps)
                        .addBindingLayout(solid.layout)
                        .setRenderState(nvrhi::RenderState()
                                            .setRasterState(nvrhi::RasterState().setCullNone())
                                            .setDepthStencilState(ds)
                                            .setBlendState(bs)),
                    fb->getFramebufferInfo());
                UAA_CHECK(pso != nullptr, "Failed to create depth-stencil pipeline");
                return pso;
            };

            nvrhi::GraphicsPipelineHandle psoLessWrite = makeDsPso(
                nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(true).setDepthFunc(nvrhi::ComparisonFunc::Less));
            nvrhi::GraphicsPipelineHandle psoGreaterNoWrite = makeDsPso(
                nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(false).setDepthFunc(nvrhi::ComparisonFunc::Greater));
            nvrhi::GraphicsPipelineHandle psoGreaterWrite = makeDsPso(
                nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(true).setDepthFunc(nvrhi::ComparisonFunc::Greater));
            nvrhi::GraphicsPipelineHandle psoLessReadOnly = makeDsPso(
                nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(false).setDepthFunc(nvrhi::ComparisonFunc::Less));

            nvrhi::GraphicsPipelineHandle psoStencilWrite = makeDsPso(
                nvrhi::DepthStencilState()
                    .setDepthTestEnable(false)
                    .setDepthWriteEnable(false)
                    .setStencilEnable(true)
                    .setStencilRefValue(0x5Au)
                    .setFrontFaceStencil(nvrhi::DepthStencilState::StencilOpDesc()
                                             .setStencilFunc(nvrhi::ComparisonFunc::Always)
                                             .setPassOp(nvrhi::StencilOp::Replace))
                    .setBackFaceStencil(nvrhi::DepthStencilState::StencilOpDesc()
                                            .setStencilFunc(nvrhi::ComparisonFunc::Always)
                                            .setPassOp(nvrhi::StencilOp::Replace)),
                nvrhi::ColorMask(0));

            auto makeStencilTestPso = [&](uint8_t refVal)
            {
                return makeDsPso(
                    nvrhi::DepthStencilState()
                        .setDepthTestEnable(false)
                        .setDepthWriteEnable(false)
                        .setStencilEnable(true)
                        .setStencilRefValue(refVal)
                        .setFrontFaceStencil(nvrhi::DepthStencilState::StencilOpDesc().setStencilFunc(nvrhi::ComparisonFunc::Equal))
                        .setBackFaceStencil(nvrhi::DepthStencilState::StencilOpDesc().setStencilFunc(nvrhi::ComparisonFunc::Equal)));
            };
            nvrhi::GraphicsPipelineHandle psoStencilEq5A = makeStencilTestPso(0x5Au);
            nvrhi::GraphicsPipelineHandle psoStencilEq11 = makeStencilTestPso(0x11u);

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            auto drawTri = [&](nvrhi::IGraphicsPipeline* pso, nvrhi::IFramebuffer* targetFb, const SolidPC& pc, const nvrhi::Viewport& vp)
            {
                cmd->setGraphicsState(
                    nvrhi::GraphicsState()
                        .setPipeline(pso)
                        .setFramebuffer(targetFb)
                        .addBindingSet(solid.bindingSet)
                        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(vp)));
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            };

            const nvrhi::Viewport fullVp = fb->getFramebufferInfo().getViewport();
            const nvrhi::Viewport leftVp(0.f, 32.f, 0.f, 64.f, 0.f, 1.f);
            const nvrhi::Color bg(10.f / 255.f, 10.f / 255.f, 10.f / 255.f, 1.f);

            // Part 1: Depth comparison + depthWriteEnable
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 1.0f, true, 0u);
            drawTri(psoLessWrite, fb, SolidPC{{1.f, 0.f, 0.f, 1.f}, 0.25f, 1.f, {0.f, 0.f}}, fullVp);       // depth=0.25, red
            drawTri(psoLessWrite, fb, SolidPC{{0.f, 0.f, 1.f, 1.f}, 0.75f, 1.f, {0.f, 0.f}}, fullVp);       // rejected
            drawTri(psoGreaterNoWrite, fb, SolidPC{{0.f, 1.f, 0.f, 1.f}, 0.75f, 1.f, {0.f, 0.f}}, fullVp);  // passes, keeps depth 0.25
            drawTri(psoGreaterWrite, fb, SolidPC{{1.f, 1.f, 0.f, 1.f}, 0.50f, 1.f, {0.f, 0.f}}, fullVp);    // passes against 0.25 -> yellow
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pix;
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 32, 32, 255, 255, 0, 255, 2, "GFX.DepthStencil depth write/compare");

            // Part 2: Stencil write + test
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 1.0f, true, 0u);
            drawTri(psoStencilWrite, fb, SolidPC{{1.f, 1.f, 1.f, 1.f}, 0.5f, 1.f, {0.f, 0.f}}, leftVp);
            drawTri(psoStencilEq5A, fb, SolidPC{{0.f, 1.f, 1.f, 1.f}, 0.5f, 1.f, {0.f, 0.f}}, fullVp);
            drawTri(psoStencilEq11, fb, SolidPC{{1.f, 0.f, 1.f, 1.f}, 0.5f, 1.f, {0.f, 0.f}}, fullVp);
            cmd->close();
            fix.submitAndWait(cmd);

            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 16, 32, 0, 255, 255, 255, 2, "GFX.DepthStencil stencil match left");
            checkPixel(ctx, pix, 64, 48, 32, 10, 10, 10, 255, 2, "GFX.DepthStencil stencil reject right");

            // Part 3: Read-only DSV
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, bg);
            cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 0.5f, false, 0u);
            drawTri(psoLessReadOnly, fbReadOnly, SolidPC{{1.f, 1.f, 1.f, 1.f}, 0.25f, 1.f, {0.f, 0.f}}, fullVp);
            cmd->close();
            fix.submitAndWait(cmd);

            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 64, 32, 32, 255, 255, 255, 255, 2, "GFX.DepthStencil read-only DSV");
        }

        void testGfxBlend(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle rt = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA8_UNORM, "gfx_blend_rt");
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
            UAA_CHECK(fb != nullptr, "Failed to create blend framebuffer");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);

            auto makeBlendPso = [&](const nvrhi::BlendState::RenderTarget& rtBlend)
            {
                nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                    nvrhi::GraphicsPipelineDesc()
                        .setVertexShader(solid.vs)
                        .setPixelShader(solid.ps)
                        .addBindingLayout(solid.layout)
                        .setRenderState(makeColorOnlyRenderState(
                            nvrhi::RenderState()
                                .setRasterState(nvrhi::RasterState().setCullNone())
                                .setBlendState(nvrhi::BlendState().setRenderTarget(0, rtBlend)))),
                    fb->getFramebufferInfo());
                UAA_CHECK(pso != nullptr, "Failed to create blend pipeline");
                return pso;
            };

            nvrhi::GraphicsPipelineHandle psoOpaque = makeBlendPso(
                nvrhi::BlendState::RenderTarget().disableBlend());
            nvrhi::GraphicsPipelineHandle psoAlpha = makeBlendPso(
                nvrhi::BlendState::RenderTarget()
                    .enableBlend()
                    .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
                    .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
                    .setBlendOp(nvrhi::BlendOp::Add)
                    .setSrcBlendAlpha(nvrhi::BlendFactor::One)
                    .setDestBlendAlpha(nvrhi::BlendFactor::Zero)
                    .setBlendOpAlpha(nvrhi::BlendOp::Add));
            nvrhi::GraphicsPipelineHandle psoRevSub = makeBlendPso(
                nvrhi::BlendState::RenderTarget()
                    .enableBlend()
                    .setSrcBlend(nvrhi::BlendFactor::One)
                    .setDestBlend(nvrhi::BlendFactor::One)
                    .setBlendOp(nvrhi::BlendOp::ReverseSubtract)
                    .setSrcBlendAlpha(nvrhi::BlendFactor::One)
                    .setDestBlendAlpha(nvrhi::BlendFactor::Zero)
                    .setBlendOpAlpha(nvrhi::BlendOp::Add));
            nvrhi::GraphicsPipelineHandle psoConst = makeBlendPso(
                nvrhi::BlendState::RenderTarget()
                    .enableBlend()
                    .setSrcBlend(nvrhi::BlendFactor::ConstantColor)
                    .setDestBlend(nvrhi::BlendFactor::Zero)
                    .setSrcBlendAlpha(nvrhi::BlendFactor::ConstantColor)
                    .setDestBlendAlpha(nvrhi::BlendFactor::Zero));
            nvrhi::GraphicsPipelineHandle psoMask = makeBlendPso(
                nvrhi::BlendState::RenderTarget()
                    .disableBlend()
                    .setColorWriteMask(nvrhi::ColorMask::Red | nvrhi::ColorMask::Alpha));

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            auto runDraw = [&](nvrhi::IGraphicsPipeline* pso, const SolidPC& pc, const nvrhi::Color& blendCol = nvrhi::Color(0.f))
            {
                cmd->setGraphicsState(
                    nvrhi::GraphicsState()
                        .setPipeline(pso)
                        .setFramebuffer(fb)
                        .addBindingSet(solid.bindingSet)
                        .setBlendColor(blendCol)
                        .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            };

            std::vector<uint8_t> pix;

            // Opaque + Alpha blend
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.1f, 0.2f, 0.3f, 1.0f));
            runDraw(psoOpaque, SolidPC{{0.8f, 0.4f, 0.2f, 1.0f}, 0.f, 1.f, {0.f, 0.f}});
            runDraw(psoAlpha, SolidPC{{0.0f, 1.0f, 0.0f, 0.5f}, 0.f, 1.f, {0.f, 0.f}});
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            // The blend state uses SrcAlpha=One and DestAlpha=Zero, so the
            // output alpha is the source alpha (0.5), not the destination's
            // opaque alpha.
            checkPixel(ctx, pix, 32, 16, 16, 102, 179, 26, 128, 3, "GFX.Blend opaque + alpha");

            // ReverseSubtract + separate alpha
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(200.f / 255.f, 200.f / 255.f, 200.f / 255.f, 1.0f));
            runDraw(psoRevSub, SolidPC{{80.f / 255.f, 50.f / 255.f, 20.f / 255.f, 100.f / 255.f}, 0.f, 1.f, {0.f, 0.f}});
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 32, 16, 16, 120, 150, 180, 100, 3, "GFX.Blend reverse subtract + separate alpha");

            // ConstantColor
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 0.f));
            runDraw(psoConst, SolidPC{{1.f, 1.f, 1.f, 1.f}, 0.f, 1.f, {0.f, 0.f}}, nvrhi::Color(0.25f, 0.5f, 0.75f, 0.5f));
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 32, 16, 16, 64, 128, 191, 128, 3, "GFX.Blend ConstantColor");

            // ColorMask Red | Alpha
            cmd->open();
            cmd->clearTextureFloat(rt, nvrhi::AllSubresources, nvrhi::Color(10.f / 255.f, 120.f / 255.f, 220.f / 255.f, 50.f / 255.f));
            runDraw(psoMask, SolidPC{{1.f, 1.f, 1.f, 1.f}, 0.f, 1.f, {0.f, 0.f}});
            cmd->close();
            fix.submitAndWait(cmd);
            fix.readbackTexture2D(rt, 0, 0, pix);
            checkPixel(ctx, pix, 32, 16, 16, 255, 120, 220, 255, 2, "GFX.Blend ColorMask Red|Alpha");
        }

        void testGfxMsaa(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle msaaTex = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_msaa_src", 1, 1, 4);
            nvrhi::TextureHandle resolveTex = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_msaa_dst", 1, 1, 1);

            nvrhi::FramebufferHandle fbMsaa = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(msaaTex));
            UAA_CHECK(fbMsaa != nullptr, "Failed to create MSAA framebuffer");
            UAA_CHECK_EQ(fbMsaa->getFramebufferInfo().sampleCount, 4u, "MSAA framebuffer sampleCount mismatch");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(
                        nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone()))),
                fbMsaa->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create MSAA graphics pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC pc{{1.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f}};

            cmd->open();
            cmd->clearTextureFloat(msaaTex, nvrhi::AllSubresources, nvrhi::Color(20.f / 255.f, 40.f / 255.f, 60.f / 255.f, 1.0f));
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fbMsaa)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fbMsaa->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->resolveTexture(resolveTex, nvrhi::AllSubresources, msaaTex, nvrhi::AllSubresources);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(resolveTex, 0, 0, pixels);
            checkPixel(ctx, pixels, 64, 32, 32, 255, 0, 0, 255, 2, "GFX.MSAA resolved interior");
            checkPixel(ctx, pixels, 64, 4, 4, 20, 40, 60, 255, 2, "GFX.MSAA resolved background");
        }

        void testGfxPipelineReuse(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle rt0 = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_reuse_rt0");
            nvrhi::TextureHandle rt1 = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_reuse_rt1");
            nvrhi::TextureHandle rtCopy = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_reuse_copy");
            nvrhi::FramebufferHandle fb0 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt0));
            nvrhi::FramebufferHandle fb1 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt1));
            UAA_CHECK(fb0 != nullptr && fb1 != nullptr, "Failed to create reuse framebuffers");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);

            nvrhi::GraphicsPipelineDesc descA;
            descA.setVertexShader(solid.vs)
                .setPixelShader(solid.ps)
                .addBindingLayout(solid.layout)
                .setRenderState(makeColorOnlyRenderState(
                    nvrhi::RenderState().setRasterState(nvrhi::RasterState().setCullNone())));

            nvrhi::GraphicsPipelineDesc descB = descA;
            descB.renderState.blendState.setRenderTarget(
                0, nvrhi::BlendState::RenderTarget().disableBlend().setColorWriteMask(nvrhi::ColorMask::Green | nvrhi::ColorMask::Alpha));

            nvrhi::GraphicsPipelineHandle psoA = dev->createGraphicsPipeline(descA, fb0->getFramebufferInfo());
            UAA_CHECK(psoA != nullptr, "Failed to create psoA via FramebufferInfo");

            // Deprecated overload: backends that leave it unimplemented will fail cleanly here.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
            nvrhi::GraphicsPipelineHandle psoB = dev->createGraphicsPipeline(descB, fb1.Get());
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
            UAA_CHECK(psoB != nullptr, "createGraphicsPipeline(..., IFramebuffer*) returned null");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC redPC{{1.f, 0.f, 0.f, 1.f}, 0.f, 1.f, {0.f, 0.f}};
            const SolidPC whitePC{{1.f, 1.f, 1.f, 1.f}, 0.f, 1.f, {0.f, 0.f}};
            const SolidPC bluePC{{0.f, 0.f, 1.f, 1.f}, 0.f, 1.f, {0.f, 0.f}};

            cmd->open();
            cmd->clearTextureFloat(rt0, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            cmd->clearTextureFloat(rt1, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));

            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoA)
                    .setFramebuffer(fb0)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb0->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&redPC, sizeof(redPC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

            cmd->copyTexture(rtCopy, nvrhi::TextureSlice(), rt0, nvrhi::TextureSlice());
            cmd->clearTextureFloat(rt0, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));

            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoB)
                    .setFramebuffer(fb1)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb1->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&whitePC, sizeof(whitePC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoA)
                    .setFramebuffer(fb0)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb0->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&bluePC, sizeof(bluePC));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pix0, pix1, pixCopy;
            fix.readbackTexture2D(rt0, 0, 0, pix0);
            fix.readbackTexture2D(rt1, 0, 0, pix1);
            fix.readbackTexture2D(rtCopy, 0, 0, pixCopy);

            checkPixel(ctx, pixCopy, 64, 32, 32, 255, 0, 0, 255, 2, "GFX.PipelineReuse mid-pass copy");
            checkPixel(ctx, pix1, 64, 32, 32, 0, 255, 0, 255, 2, "GFX.PipelineReuse psoB masked green");
            checkPixel(ctx, pix0, 64, 32, 32, 0, 0, 255, 255, 2, "GFX.PipelineReuse psoA re-bound blue");
        }

        void testGfxAttachmentLifetime(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::Format depthFmt = pickDepthFormat(ctx, dev);
            nvrhi::TextureHandle rtColor = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_life_color");
            nvrhi::TextureHandle rtDepth = createDepthTarget(ctx, dev, 64, 64, depthFmt, "gfx_life_depth");
            nvrhi::TextureHandle keepTex = createColorTarget(ctx, dev, 64, 64, nvrhi::Format::RGBA8_UNORM, "gfx_life_keep");

            nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(rtColor).setDepthAttachment(rtDepth));
            UAA_CHECK(fb != nullptr, "Failed to create lifetime framebuffer");
            UAA_CHECK(rtColor->GetRefCount() > 1u, "Framebuffer did not retain reference to colour attachment");
            UAA_CHECK(rtDepth->GetRefCount() > 1u, "Framebuffer did not retain reference to depth attachment");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(nvrhi::RenderState()
                                        .setRasterState(nvrhi::RasterState().setCullNone())
                                        .setDepthStencilState(nvrhi::DepthStencilState().setDepthTestEnable(true).setDepthWriteEnable(true).setDepthFunc(nvrhi::ComparisonFunc::Less))),
                fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create lifetime pipeline");

            // Release caller texture handles before recording; the framebuffer must keep them alive.
            nvrhi::ITexture* retainedColor = fb->getDesc().colorAttachments[0].texture;
            nvrhi::ITexture* retainedDepth = fb->getDesc().depthAttachment.texture;
            rtColor = nullptr;
            rtDepth = nullptr;

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const SolidPC pc{{0.25f, 0.75f, 0.5f, 1.0f}, 0.5f, 1.0f, {0.0f, 0.0f}};

            cmd->open();
            cmd->clearTextureFloat(retainedColor, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            cmd->clearDepthStencilTexture(retainedDepth, nvrhi::AllSubresources, true, 1.0f, false, 0u);
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fb)
                    .addBindingSet(solid.bindingSet)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(fb->getFramebufferInfo().getViewport())));
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->copyTexture(keepTex, nvrhi::TextureSlice(), retainedColor, nvrhi::TextureSlice());
            cmd->close();

            // Drop the framebuffer too before execution so the command list lifetime tracker must retain it.
            fb = nullptr;

            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(keepTex, 0, 0, pixels);
            checkPixel(ctx, pixels, 64, 32, 32, 64, 191, 128, 255, 2, "GFX.AttachmentLifetime retained render output");
        }

        void testGfxInvalid(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (!fix.isValidationActive())
            {
                ctx.skip("NVRHI validation layer is not active");
            }

            nvrhi::IDevice* dev = fix.device();
            nvrhi::TextureHandle rtRgba8 = createColorTarget(ctx, dev, 32, 32, nvrhi::Format::RGBA8_UNORM, "gfx_inv_rgba8");
            nvrhi::FramebufferHandle fbRgba8 = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rtRgba8));
            UAA_CHECK(fbRgba8 != nullptr, "Failed to create invalid-test framebuffer");

            SolidPipelineBundle solid = makeSolidBundle(ctx, fix);

            // Compatible pipeline for RGBA16_FLOAT (incompatible with fbRgba8).
            nvrhi::FramebufferInfo fbInfoFloat;
            fbInfoFloat.addColorFormat(nvrhi::Format::RGBA16_FLOAT);
            nvrhi::GraphicsPipelineHandle psoFloat = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(nvrhi::RenderState())),
                fbInfoFloat);
            UAA_CHECK(psoFloat != nullptr, "Failed to create RGBA16_FLOAT pipeline");

            // Compatible pipeline for 4x MSAA (sample count mismatch with fbRgba8).
            nvrhi::FramebufferInfo fbInfoMsaa;
            fbInfoMsaa.addColorFormat(nvrhi::Format::RGBA8_UNORM).setSampleCount(4u);
            nvrhi::GraphicsPipelineHandle psoMsaa = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(nvrhi::RenderState())),
                fbInfoMsaa);
            UAA_CHECK(psoMsaa != nullptr, "Failed to create 4x MSAA pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            ExpectErrorsGuard guard(fix);

            // Depth test enabled on a FramebufferInfo with no depth format.
            nvrhi::GraphicsPipelineHandle badDepthPso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.vs)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(nvrhi::RenderState().setDepthStencilState(nvrhi::DepthStencilState().setDepthTestEnable(true))),
                fbRgba8->getFramebufferInfo());
            UAA_CHECK(badDepthPso == nullptr, "Expected null pipeline when depth test enabled without depth attachment");

            // Wrong shader stage in vertex shader slot.
            nvrhi::GraphicsPipelineHandle badStagePso = dev->createGraphicsPipeline(
                nvrhi::GraphicsPipelineDesc()
                    .setVertexShader(solid.ps)
                    .setPixelShader(solid.ps)
                    .addBindingLayout(solid.layout)
                    .setRenderState(makeColorOnlyRenderState(nvrhi::RenderState())),
                fbRgba8->getFramebufferInfo());
            UAA_CHECK(badStagePso == nullptr, "Expected null pipeline when pixel shader is bound to VS slot");

            // Missing VS/PS on backends that validate it without triggering debug asserts in vkCreateGraphicsPipelines.
            if (fix.api() != nvrhi::GraphicsAPI::VULKAN)
            {
                nvrhi::GraphicsPipelineHandle missingVsPso = dev->createGraphicsPipeline(
                    nvrhi::GraphicsPipelineDesc()
                        .setPixelShader(solid.ps)
                        .addBindingLayout(solid.layout)
                        .setRenderState(makeColorOnlyRenderState(nvrhi::RenderState())),
                    fbRgba8->getFramebufferInfo());
                UAA_CHECK(missingVsPso == nullptr, "Expected null pipeline when vertex shader is missing");
            }

            cmd->open();

            // Incompatible framebuffer format.
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoFloat)
                    .setFramebuffer(fbRgba8)
                    .addBindingSet(solid.bindingSet));

            // Sample count mismatch.
            cmd->setGraphicsState(
                nvrhi::GraphicsState()
                    .setPipeline(psoMsaa)
                    .setFramebuffer(fbRgba8)
                    .addBindingSet(solid.bindingSet));

            // Null pipeline / null framebuffer.
            cmd->setGraphicsState(nvrhi::GraphicsState());

            cmd->close();
            fix.submitAndWait(cmd);

            UAA_CHECK(guard.errorCount() >= 5u, "Expected validation errors for invalid graphics pipelines and states");
        }
    } // namespace

    UAA_REGISTER_TEST("GFX.Framebuffer", "core", testGfxFramebuffer);
    UAA_REGISTER_TEST("GFX.Clear", "core", testGfxClear);
    UAA_REGISTER_TEST("GFX.Draw", "core", testGfxDraw);
    UAA_REGISTER_TEST("GFX.VertexInput", "core", testGfxVertexInput);
    UAA_REGISTER_TEST("GFX.DrawArguments", "core", testGfxDrawArguments);
    UAA_REGISTER_TEST("GFX.Indirect", "core", testGfxIndirect);
    UAA_REGISTER_TEST("GFX.IndirectCount", "core", testGfxIndirectCount);
    UAA_REGISTER_TEST("GFX.RasterState", "core", testGfxRasterState);
    UAA_REGISTER_TEST("GFX.DepthStencil", "core", testGfxDepthStencil);
    UAA_REGISTER_TEST("GFX.Blend", "core", testGfxBlend);
    UAA_REGISTER_TEST("GFX.MSAA", "core", testGfxMsaa);
    UAA_REGISTER_TEST("GFX.PipelineReuse", "core", testGfxPipelineReuse);
    UAA_REGISTER_TEST("GFX.AttachmentLifetime", "core", testGfxAttachmentLifetime);
    UAA_REGISTER_TEST("GFX.Invalid", "core", testGfxInvalid);
} // namespace nvrhi::uaa
