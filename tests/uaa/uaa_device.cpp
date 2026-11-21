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
#include <nvrhi/utils.h>
#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
#include <nvrhi/validation.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
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

        template <typename T>
        struct GuardedFeatureInfo
        {
            uint32_t head = TestFixture::kGuardWord;
            T info{};
            uint32_t tail = TestFixture::kGuardWord;
        };

        void runCopyAddOnDevice(
            TestContext& ctx,
            nvrhi::IDevice* dev,
            const std::vector<uint8_t>& shaderBytes,
            uint64_t seed,
            uint32_t addVal,
            uint32_t xorVal)
        {
            UAA_CHECK(dev != nullptr, "runCopyAddOnDevice: null device");

            constexpr uint32_t kCount = 64;
            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kCount, seed);

            nvrhi::ShaderDesc shaderDesc;
            shaderDesc.setShaderType(nvrhi::ShaderType::Compute)
                .setEntryName("main")
                .setDebugName("uaa_cs_buffer_copy_add");
            nvrhi::ShaderHandle cs = dev->createShader(shaderDesc, shaderBytes.data(), shaderBytes.size());
            UAA_CHECK(cs != nullptr, "createShader(cs_buffer_copy_add) returned null");

            const nvrhi::BufferDesc cbDesc = nvrhi::BufferDesc()
                .setByteSize(256)
                .setIsConstantBuffer(true)
                .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                .setKeepInitialState(true)
                .setDebugName("uaa_copy_add_cb");
            nvrhi::BufferHandle cb = dev->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "Failed to create constant buffer");

            const nvrhi::BufferDesc inDesc = nvrhi::BufferDesc()
                .setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("uaa_copy_add_in");
            nvrhi::BufferHandle inBuf = dev->createBuffer(inDesc);
            UAA_CHECK(inBuf != nullptr, "Failed to create input buffer");

            const nvrhi::BufferDesc outDesc = nvrhi::BufferDesc()
                .setByteSize(kCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true)
                .setDebugName("uaa_copy_add_out");
            nvrhi::BufferHandle outBuf = dev->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "Failed to create output buffer");

            const nvrhi::BufferDesc rbDesc = nvrhi::BufferDesc()
                .setByteSize(kCount * sizeof(uint32_t))
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setDebugName("uaa_copy_add_rb");
            nvrhi::BufferHandle rbBuf = dev->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "Failed to create readback buffer");

            nvrhi::BindingLayoutDesc layoutDesc = ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create binding layout");

            const nvrhi::BindingSetDesc setDesc = nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf));
            nvrhi::BindingSetHandle set = dev->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "Failed to create binding set");

            nvrhi::ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(cs).addBindingLayout(layout);
            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            CopyAddParams params{};
            params.count = kCount;
            params.addVal = addVal;
            params.xorVal = xorVal;

            cmd->open();
            cmd->writeBuffer(cb, &params, sizeof(params));
            cmd->writeBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            cmd->clearBufferUInt(outBuf, 0u);

            nvrhi::ComputeState state;
            state.setPipeline(pso).addBindingSet(set);
            cmd->setComputeState(state);
            cmd->dispatch(1, 1, 1);
            cmd->copyBuffer(rbBuf, 0, outBuf, 0, kCount * sizeof(uint32_t));
            cmd->close();

            dev->executeCommandList(cmd);
            UAA_CHECK(dev->waitForIdle(), "waitForIdle returned false");
            dev->runGarbageCollection();

            std::vector<uint32_t> actual(kCount, 0u);
            void* mapped = dev->mapBuffer(rbBuf, nvrhi::CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer returned null for readback");
            std::memcpy(actual.data(), mapped, kCount * sizeof(uint32_t));
            dev->unmapBuffer(rbBuf);

            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = (inputWords[i] + addVal) ^ xorVal;
                UAA_CHECK_EQ(actual[i], expected, "Compute output mismatch at index " + std::to_string(i));
            }
        }

        void testCreateDestroy(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            nvrhi::IDevice* raw = ctx.fixture->rawDevice();
            UAA_CHECK(dev != nullptr, "Active device must not be null");
            UAA_CHECK(raw != nullptr, "Raw device must not be null");
            UAA_CHECK_EQ(dev->getGraphicsAPI(), raw->getGraphicsAPI(), "Active and raw device API must match");
            UAA_CHECK(dev->getMessageCallback() != nullptr, "Device message callback must not be null");

            MessageCapture secMessages;
            std::unique_ptr<IBackendContext> secBackend = ctx.fixture->backend()->createSecondaryDevice(&secMessages);
            if (!secBackend)
            {
                ctx.note("Secondary device creation not supported on this backend");
                return;
            }

            nvrhi::IDevice* secRaw = secBackend->getRawDevice();
            UAA_CHECK(secRaw != nullptr, "Secondary backend returned null raw device");
            UAA_CHECK_EQ(secRaw->getGraphicsAPI(), dev->getGraphicsAPI(), "Secondary device API mismatch");
            UAA_CHECK(secRaw->getMessageCallback() == &secMessages, "Secondary device message callback mismatch");

#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
            if (ctx.fixture->isValidationActive())
            {
                nvrhi::DeviceHandle secVal = nvrhi::validation::createValidationLayer(secRaw);
                UAA_CHECK(secVal != nullptr, "Failed to wrap secondary device with validation layer");
                UAA_CHECK_EQ(secVal->getGraphicsAPI(), secRaw->getGraphicsAPI(), "Validated secondary API mismatch");
                secVal->waitForIdle();
                secVal->runGarbageCollection();
            }
#endif

            secRaw->waitForIdle();
            secRaw->runGarbageCollection();
            secBackend.reset();
            UAA_CHECK_EQ(secMessages.errorCount(), size_t(0), "Secondary device produced unexpected errors");
        }

        void testCapabilities(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            for (nvrhi::Feature f : FeatureInventory::allFeatures())
            {
                const bool supported = dev->queryFeatureSupport(f);
                UAA_CHECK_EQ(supported, ctx.fixture->features().supports(f),
                    std::string("Feature support mismatch for ") + FeatureInventory::featureName(f));
            }

            {
                GuardedFeatureInfo<nvrhi::VariableRateShadingFeatureInfo> vrs;
                vrs.info.shadingRateImageTileSize = TestFixture::kGuardWord;
                const bool supported = dev->queryFeatureSupport(
                    nvrhi::Feature::VariableRateShading, &vrs.info, sizeof(vrs.info));
                UAA_CHECK_EQ(vrs.head, TestFixture::kGuardWord, "VRS info query corrupted leading sentinel");
                UAA_CHECK_EQ(vrs.tail, TestFixture::kGuardWord, "VRS info query corrupted trailing sentinel");
                if (supported)
                {
                    UAA_CHECK(vrs.info.shadingRateImageTileSize > 0 && vrs.info.shadingRateImageTileSize != TestFixture::kGuardWord,
                        "Supported VRS must populate valid tile size");
                }
                else
                {
                    UAA_CHECK(vrs.info.shadingRateImageTileSize == TestFixture::kGuardWord || vrs.info.shadingRateImageTileSize == 0u,
                        "Unsupported VRS must not write garbage into info struct");
                }
            }

            {
                GuardedFeatureInfo<nvrhi::WaveLaneCountMinMaxFeatureInfo> wave;
                wave.info.minWaveLaneCount = TestFixture::kGuardWord;
                wave.info.maxWaveLaneCount = TestFixture::kGuardWord;
                const bool supported = dev->queryFeatureSupport(
                    nvrhi::Feature::WaveLaneCountMinMax, &wave.info, sizeof(wave.info));
                UAA_CHECK_EQ(wave.head, TestFixture::kGuardWord, "Wave info query corrupted leading sentinel");
                UAA_CHECK_EQ(wave.tail, TestFixture::kGuardWord, "Wave info query corrupted trailing sentinel");
                if (supported)
                {
                    UAA_CHECK(wave.info.minWaveLaneCount > 0 && wave.info.minWaveLaneCount != TestFixture::kGuardWord,
                        "Supported WaveLaneCountMinMax must populate minWaveLaneCount");
                    UAA_CHECK(wave.info.maxWaveLaneCount >= wave.info.minWaveLaneCount && wave.info.maxWaveLaneCount != TestFixture::kGuardWord,
                        "Supported WaveLaneCountMinMax must populate maxWaveLaneCount >= min");
                }
                else
                {
                    UAA_CHECK_EQ(wave.info.minWaveLaneCount, TestFixture::kGuardWord,
                        "Unsupported WaveLaneCountMinMax must not overwrite minWaveLaneCount sentinel");
                    UAA_CHECK_EQ(wave.info.maxWaveLaneCount, TestFixture::kGuardWord,
                        "Unsupported WaveLaneCountMinMax must not overwrite maxWaveLaneCount sentinel");
                }
            }

            {
                GuardedFeatureInfo<nvrhi::RayTracingClustersFeatureInfo> clusters;
                clusters.info.api = static_cast<nvrhi::ClusterApi>(0xDE);
                const bool supported = dev->queryFeatureSupport(
                    nvrhi::Feature::RayTracingClusters, &clusters.info, sizeof(clusters.info));
                UAA_CHECK_EQ(clusters.head, TestFixture::kGuardWord, "Cluster info query corrupted leading sentinel");
                UAA_CHECK_EQ(clusters.tail, TestFixture::kGuardWord, "Cluster info query corrupted trailing sentinel");
                if (supported)
                {
                    UAA_CHECK(clusters.info.api == nvrhi::ClusterApi::Nv || clusters.info.api == nvrhi::ClusterApi::Dxr,
                        "Supported RayTracingClusters must report valid ClusterApi");
                }
                else
                {
                    UAA_CHECK(clusters.info.api == static_cast<nvrhi::ClusterApi>(0xDE) || clusters.info.api == nvrhi::ClusterApi::None,
                        "Unsupported RayTracingClusters must preserve sentinel or report ClusterApi::None");
                }
            }
        }

        void testFormatSupport(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            UAA_CHECK_EQ(dev->queryFormatSupport(nvrhi::Format::UNKNOWN), nvrhi::FormatSupport::None,
                "Format::UNKNOWN must return FormatSupport::None");

            for (size_t i = 1; i < size_t(nvrhi::Format::COUNT); ++i)
            {
                const nvrhi::Format fmt = static_cast<nvrhi::Format>(i);
                const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(fmt);
                const nvrhi::FormatSupport support = dev->queryFormatSupport(fmt);

                if ((support & nvrhi::FormatSupport::Texture) != 0u)
                {
                    const bool hasRenderOrDepth = (support & (nvrhi::FormatSupport::RenderTarget | nvrhi::FormatSupport::DepthStencil)) != 0u;
                    const bool hasShaderRead = (support & (nvrhi::FormatSupport::ShaderLoad | nvrhi::FormatSupport::ShaderSample)) != 0u;

                    nvrhi::TextureDesc texDesc;
                    texDesc.setWidth(4)
                        .setHeight(4)
                        .setFormat(fmt)
                        .setDimension(nvrhi::TextureDimension::Texture2D)
                        .setIsRenderTarget(hasRenderOrDepth)
                        .setDebugName(std::string("uaa_fmt_tex_") + fmtInfo.name);
                    texDesc.isShaderResource = hasShaderRead || !hasRenderOrDepth;

                    nvrhi::TextureHandle tex = dev->createTexture(texDesc);
                    UAA_CHECK(tex != nullptr,
                        std::string("createTexture failed for format advertising FormatSupport::Texture: ") + fmtInfo.name);
                    UAA_CHECK_EQ(tex->getDesc().format, fmt,
                        std::string("Created texture format mismatch for ") + fmtInfo.name);
                }

                if ((support & nvrhi::FormatSupport::Buffer) != 0u &&
                    (support & (nvrhi::FormatSupport::VertexBuffer | nvrhi::FormatSupport::IndexBuffer)) != 0u)
                {
                    nvrhi::BufferDesc bufDesc;
                    bufDesc.setByteSize(64)
                        .setFormat(fmt)
                        .setIsVertexBuffer((support & nvrhi::FormatSupport::VertexBuffer) != 0u)
                        .setIsIndexBuffer((support & nvrhi::FormatSupport::IndexBuffer) != 0u)
                        .setDebugName(std::string("uaa_fmt_buf_") + fmtInfo.name);

                    nvrhi::BufferHandle buf = dev->createBuffer(bufDesc);
                    UAA_CHECK(buf != nullptr,
                        std::string("createBuffer failed for format advertising Buffer+Vertex/Index: ") + fmtInfo.name);
                    UAA_CHECK_EQ(buf->getDesc().format, fmt,
                        std::string("Created buffer format mismatch for ") + fmtInfo.name);
                }
            }
        }

        void testMetalFormatClaims(TestContext& ctx)
        {
            if (ctx.fixture->api() != nvrhi::GraphicsAPI::METAL)
                UAA_SKIP("Metal format claims apply only to the Metal backend");

            nvrhi::IDevice* dev = ctx.fixture->device();
            using nvrhi::Format;
            using nvrhi::FormatSupport;
            const FormatSupport noCompressed = FormatSupport::Buffer | FormatSupport::RenderTarget | FormatSupport::Blendable
                | FormatSupport::DepthStencil | FormatSupport::ShaderUavLoad | FormatSupport::ShaderUavStore;
            const FormatSupport bc = dev->queryFormatSupport(Format::BC1_UNORM);
            UAA_CHECK((bc & noCompressed) == 0u, "BC1 must not advertise buffer, attachment, blend, or UAV support");

            const FormatSupport integer = dev->queryFormatSupport(Format::RGBA8_UINT);
            UAA_CHECK((integer & FormatSupport::RenderTarget) != 0u, "RGBA8_UINT must support colour attachments");
            UAA_CHECK((integer & (FormatSupport::Blendable | FormatSupport::ShaderSample)) == 0u,
                "RGBA8_UINT must not advertise blending or filtered sampling");

            const FormatSupport rgba = dev->queryFormatSupport(Format::RGBA8_UNORM);
            const FormatSupport rgbaRequired = FormatSupport::Texture | FormatSupport::RenderTarget | FormatSupport::Blendable
                | FormatSupport::ShaderSample | FormatSupport::ShaderUavStore;
            UAA_CHECK((rgba & rgbaRequired) == rgbaRequired, "RGBA8_UNORM is missing a required capability");

            const FormatSupport depth = dev->queryFormatSupport(Format::D32);
            UAA_CHECK((depth & (FormatSupport::Texture | FormatSupport::DepthStencil)) == (FormatSupport::Texture | FormatSupport::DepthStencil),
                "D32 is missing depth texture support");
            UAA_CHECK((depth & (FormatSupport::Buffer | FormatSupport::RenderTarget | FormatSupport::Blendable
                | FormatSupport::ShaderUavLoad | FormatSupport::ShaderUavStore)) == 0u,
                "D32 must not advertise buffer, colour, blend, or UAV support");
        }

        void testResourceDescriptions(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            const nvrhi::BufferDesc bufDesc = nvrhi::BufferDesc()
                .setByteSize(512)
                .setStructStride(16)
                .setCanHaveUAVs(true)
                .setCanHaveRawViews(true)
                .setDebugName("uaa_desc_buffer");
            nvrhi::BufferHandle buf = dev->createBuffer(bufDesc);
            UAA_CHECK(buf != nullptr, "createBuffer returned null");
            UAA_CHECK_EQ(buf->getDesc().byteSize, uint64_t(512), "BufferDesc byteSize mismatch");
            UAA_CHECK_EQ(buf->getDesc().structStride, 16u, "BufferDesc structStride mismatch");
            UAA_CHECK(buf->getDesc().canHaveUAVs && buf->getDesc().canHaveRawViews, "BufferDesc flags mismatch");
            UAA_CHECK_EQ(buf->getDesc().debugName, std::string("uaa_desc_buffer"), "BufferDesc debugName mismatch");

            const nvrhi::TextureDesc texDesc = nvrhi::TextureDesc()
                .setWidth(32)
                .setHeight(16)
                .setMipLevels(3)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setIsRenderTarget(true)
                .setDebugName("uaa_desc_texture");
            nvrhi::TextureHandle tex = dev->createTexture(texDesc);
            UAA_CHECK(tex != nullptr, "createTexture returned null");
            UAA_CHECK_EQ(tex->getDesc().width, 32u, "TextureDesc width mismatch");
            UAA_CHECK_EQ(tex->getDesc().height, 16u, "TextureDesc height mismatch");
            UAA_CHECK_EQ(tex->getDesc().mipLevels, 3u, "TextureDesc mipLevels mismatch");
            UAA_CHECK_EQ(tex->getDesc().format, nvrhi::Format::RGBA8_UNORM, "TextureDesc format mismatch");
            UAA_CHECK_EQ(tex->getDesc().dimension, nvrhi::TextureDimension::Texture2D, "TextureDesc dimension mismatch");
            UAA_CHECK(tex->getDesc().isRenderTarget && tex->getDesc().isShaderResource, "TextureDesc flags mismatch");
            UAA_CHECK_EQ(tex->getDesc().debugName, std::string("uaa_desc_texture"), "TextureDesc debugName mismatch");

            const auto& extensions = ctx.fixture->deviceInfo().enabledExtensions;
            const bool portabilitySubset = std::find(extensions.begin(), extensions.end(),
                "VK_KHR_portability_subset") != extensions.end();
            const float mipBias = portabilitySubset ? 0.f : 0.5f;
            const nvrhi::SamplerDesc sampDesc = nvrhi::SamplerDesc()
                .setMinFilter(false)
                .setMagFilter(true)
                .setMipFilter(false)
                .setAddressU(nvrhi::SamplerAddressMode::Wrap)
                .setAddressV(nvrhi::SamplerAddressMode::Mirror)
                .setAddressW(nvrhi::SamplerAddressMode::Clamp)
                .setMipBias(mipBias)
                .setMaxAnisotropy(4.f)
                .setBorderColor(nvrhi::Color(0.f, 0.f, 0.f, 1.f));
            nvrhi::SamplerHandle samp = dev->createSampler(sampDesc);
            UAA_CHECK(samp != nullptr, "createSampler returned null");
            UAA_CHECK(!samp->getDesc().minFilter && samp->getDesc().magFilter && !samp->getDesc().mipFilter, "Sampler filters mismatch");
            UAA_CHECK_EQ(samp->getDesc().addressU, nvrhi::SamplerAddressMode::Wrap, "Sampler addressU mismatch");
            UAA_CHECK_EQ(samp->getDesc().addressV, nvrhi::SamplerAddressMode::Mirror, "Sampler addressV mismatch");
            UAA_CHECK_NEAR(samp->getDesc().mipBias, mipBias, 1e-6, "Sampler mipBias mismatch");
            UAA_CHECK_NEAR(samp->getDesc().maxAnisotropy, 4.f, 1e-6, "Sampler maxAnisotropy mismatch");

            nvrhi::ShaderHandle vs = ctx.fixture->createShader("vs_vertex_input", nvrhi::ShaderType::Vertex);
            const nvrhi::VertexAttributeDesc attrs[] = {
                nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setBufferIndex(0).setOffset(0).setElementStride(28),
                nvrhi::VertexAttributeDesc().setName("COLOR").setFormat(nvrhi::Format::RGBA32_FLOAT).setBufferIndex(0).setOffset(12).setElementStride(28),
            };
            nvrhi::InputLayoutHandle il = dev->createInputLayout(attrs, 2, vs);
            UAA_CHECK(il != nullptr, "createInputLayout returned null");
            UAA_CHECK_EQ(il->getNumAttributes(), 2u, "InputLayout attribute count mismatch");
            const nvrhi::VertexAttributeDesc* attr0 = il->getAttributeDesc(0);
            const nvrhi::VertexAttributeDesc* attr1 = il->getAttributeDesc(1);
            UAA_CHECK(attr0 != nullptr && attr1 != nullptr, "getAttributeDesc returned null");
            UAA_CHECK_EQ(attr0->name, std::string("POSITION"), "InputLayout attr 0 name mismatch");
            UAA_CHECK_EQ(attr1->format, nvrhi::Format::RGBA32_FLOAT, "InputLayout attr 1 format mismatch");
            UAA_CHECK_EQ(attr1->offset, 12u, "InputLayout attr 1 offset mismatch");

            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(nvrhi::FramebufferAttachment().setTexture(tex).setMipLevel(1));
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(fbDesc);
            UAA_CHECK(fb != nullptr, "createFramebuffer returned null");
            UAA_CHECK_EQ(fb->getDesc().colorAttachments.size(), size_t(1), "Framebuffer attachment count mismatch");
            UAA_CHECK_EQ(fb->getFramebufferInfo().width, 16u, "Framebuffer mip 1 width mismatch");
            UAA_CHECK_EQ(fb->getFramebufferInfo().height, 8u, "Framebuffer mip 1 height mismatch");
            UAA_CHECK_EQ(fb->getFramebufferInfo().colorFormats.size(), size_t(1), "FramebufferInfo colour format count mismatch");
            UAA_CHECK_EQ(fb->getFramebufferInfo().colorFormats[0], nvrhi::Format::RGBA8_UNORM, "FramebufferInfo colour format mismatch");

            const nvrhi::BindingLayoutDesc layoutDesc = ctx.fixture->makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
                .addItem(nvrhi::BindingLayoutItem::Sampler(1));
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "createBindingLayout returned null");
            UAA_CHECK(layout->getDesc() != nullptr, "BindingLayout::getDesc() must be non-null");
            UAA_CHECK(layout->getBindlessDesc() == nullptr, "BindingLayout::getBindlessDesc() must be null");
            UAA_CHECK_EQ(layout->getDesc()->visibility, nvrhi::ShaderType::Compute, "BindingLayout visibility mismatch");
            UAA_CHECK_EQ(layout->getDesc()->bindings.size(), size_t(2), "BindingLayout item count mismatch");

            if (dev->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11)
            {
                const nvrhi::BindlessLayoutDesc bindlessDesc = nvrhi::BindlessLayoutDesc()
                    .setVisibility(nvrhi::ShaderType::Compute)
                    .setFirstSlot(0)
                    .setMaxCapacity(64)
                    .setUpdateAfterBind(false)
                    .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0));
                nvrhi::BindingLayoutHandle bindless = dev->createBindlessLayout(bindlessDesc);
                UAA_CHECK(bindless != nullptr, "createBindlessLayout returned null");
                UAA_CHECK(bindless->getDesc() == nullptr, "Bindless layout getDesc() must be null");
                UAA_CHECK(bindless->getBindlessDesc() != nullptr, "Bindless layout getBindlessDesc() must be non-null");
                UAA_CHECK_EQ(bindless->getBindlessDesc()->maxCapacity, 64u, "Bindless maxCapacity mismatch");
            }

            const nvrhi::BindingSetDesc setDesc = nvrhi::BindingSetDesc()
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buf))
                .addItem(nvrhi::BindingSetItem::Sampler(1, samp));
            nvrhi::BindingSetHandle set = dev->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "createBindingSet returned null");
            UAA_CHECK(set->getDesc() != nullptr, "BindingSet::getDesc() must be non-null");
            UAA_CHECK(set->getLayout() == layout.Get(), "BindingSet::getLayout() mismatch");
            UAA_CHECK_EQ(set->getDesc()->bindings.size(), size_t(2), "BindingSet binding count mismatch");

            if (dev->queryFeatureSupport(nvrhi::Feature::VirtualResources))
            {
                const nvrhi::HeapDesc heapDesc = nvrhi::HeapDesc()
                    .setCapacity(64 * 1024)
                    .setType(nvrhi::HeapType::DeviceLocal)
                    .setDebugName("uaa_desc_heap");
                nvrhi::HeapHandle heap = dev->createHeap(heapDesc);
                UAA_CHECK(heap != nullptr, "createHeap returned null");
                UAA_CHECK_EQ(heap->getDesc().capacity, uint64_t(64 * 1024), "Heap capacity mismatch");
                UAA_CHECK_EQ(heap->getDesc().type, nvrhi::HeapType::DeviceLocal, "Heap type mismatch");
                UAA_CHECK_EQ(heap->getDesc().debugName, std::string("uaa_desc_heap"), "Heap debugName mismatch");
            }

            NativeBufferImport nativeBuf{};
            if (ctx.fixture->backend()->createNativeImportBuffer(256, nullptr, nativeBuf))
            {
                const nvrhi::BufferDesc importBufDesc = nvrhi::BufferDesc()
                    .setByteSize(nativeBuf.byteSize)
                    .setDebugName("uaa_native_buf_import");
                nvrhi::BufferHandle imported = dev->createHandleForNativeBuffer(
                    nativeBuf.objectType, nativeBuf.handle, importBufDesc);
                UAA_CHECK(imported != nullptr, "createHandleForNativeBuffer returned null");
                UAA_CHECK_EQ(imported->getDesc().byteSize, nativeBuf.byteSize, "Imported buffer byteSize mismatch");
                UAA_CHECK_EQ(imported->getDesc().debugName, std::string("uaa_native_buf_import"), "Imported buffer debugName mismatch");
                imported = nullptr;
                if (nativeBuf.release)
                {
                    nativeBuf.release();
                }
            }

            NativeTextureImport nativeTex{};
            if (ctx.fixture->backend()->createNativeImportTexture(16, 16, nvrhi::Format::RGBA8_UNORM, nativeTex))
            {
                nativeTex.desc.setDebugName("uaa_native_tex_import");
                nvrhi::TextureHandle importedTex = dev->createHandleForNativeTexture(
                    nativeTex.objectType, nativeTex.handle, nativeTex.desc);
                UAA_CHECK(importedTex != nullptr, "createHandleForNativeTexture returned null");
                UAA_CHECK_EQ(importedTex->getDesc().width, 16u, "Imported texture width mismatch");
                UAA_CHECK_EQ(importedTex->getDesc().height, 16u, "Imported texture height mismatch");
                UAA_CHECK_EQ(importedTex->getDesc().format, nvrhi::Format::RGBA8_UNORM, "Imported texture format mismatch");
                importedTex = nullptr;
                if (nativeTex.release)
                {
                    nativeTex.release();
                }
            }
        }

        void testGarbageCollection(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            for (uint32_t iter = 0; iter < 4; ++iter)
            {
                nvrhi::BufferHandle tmpBuf = dev->createBuffer(
                    nvrhi::BufferDesc()
                        .setByteSize(1024)
                        .setCanHaveUAVs(true)
                        .setInitialState(nvrhi::ResourceStates::CopyDest)
                        .setKeepInitialState(true)
                        .setDebugName("uaa_gc_tmp_buf"));
                UAA_CHECK(tmpBuf != nullptr, "Failed to create temporary GC buffer");

                nvrhi::TextureHandle tmpTex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setWidth(16)
                        .setHeight(16)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .setIsRenderTarget(true)
                        .setInitialState(nvrhi::ResourceStates::RenderTarget)
                        .setKeepInitialState(true)
                        .setDebugName("uaa_gc_tmp_tex"));
                UAA_CHECK(tmpTex != nullptr, "Failed to create temporary GC texture");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "Failed to create temporary GC command list");
                cmd->open();
                cmd->clearBufferUInt(tmpBuf, 0x11223344u + iter);
                cmd->clearTextureFloat(tmpTex, nvrhi::AllSubresources, nvrhi::Color(0.2f, 0.4f, 0.6f, 1.f));
                cmd->close();

                ctx.fixture->submitAndWait(cmd);
                cmd = nullptr;
                tmpBuf = nullptr;
                tmpTex = nullptr;

                for (int gc = 0; gc < 3; ++gc)
                {
                    dev->runGarbageCollection();
                }
            }

            const std::vector<uint8_t> shaderBytes = ctx.fixture->loadShaderBinary("cs_buffer_copy_add");
            runCopyAddOnDevice(ctx, dev, shaderBytes, ctx.seed() ^ 0x4743ull, 13u, 0x5A5A5A5Au);
        }

        void testMultipleDevices(TestContext& ctx)
        {
            MessageCapture secMessages;
            std::unique_ptr<IBackendContext> secBackend = ctx.fixture->backend()->createSecondaryDevice(&secMessages);
            if (!secBackend)
            {
                ctx.skip("Backend does not support createSecondaryDevice");
            }

            nvrhi::IDevice* primary = ctx.fixture->device();
            nvrhi::IDevice* secondary = secBackend->getRawDevice();
            UAA_CHECK(primary != nullptr && secondary != nullptr, "Both primary and secondary devices must be non-null");

            const std::vector<uint8_t> shaderBytes = ctx.fixture->loadShaderBinary("cs_buffer_copy_add");

            runCopyAddOnDevice(ctx, secondary, shaderBytes, ctx.seed() ^ 0x1111ull, 7u, 0x13572468u);
            runCopyAddOnDevice(ctx, primary, shaderBytes, ctx.seed() ^ 0x2222ull, 19u, 0xCAFEBABEu);

            if (ctx.fixture->isValidationActive())
            {
                ctx.note("NVRHI validation layer wraps single devices independently without cross-device ownership checks; skipping unsafe cross-device handle mixing.");
            }

            secondary->waitForIdle();
            secondary->runGarbageCollection();
            secBackend.reset();
            UAA_CHECK_EQ(secMessages.errorCount(), size_t(0), "Secondary device logged unexpected errors");
        }

        void testReinitialise(TestContext& ctx)
        {
            const std::vector<uint8_t> shaderBytes = ctx.fixture->loadShaderBinary("cs_buffer_copy_add");

            for (uint32_t cycle = 0; cycle < 3; ++cycle)
            {
                MessageCapture secMessages;
                std::unique_ptr<IBackendContext> secBackend = ctx.fixture->backend()->createSecondaryDevice(&secMessages);
                if (!secBackend)
                {
                    if (cycle == 0)
                    {
                        ctx.skip("Backend does not support createSecondaryDevice");
                    }
                    UAA_CHECK(false, "Secondary device creation failed on subsequent reinitialise cycle");
                }

                nvrhi::IDevice* secRaw = secBackend->getRawDevice();
                UAA_CHECK(secRaw != nullptr, "Secondary raw device must not be null");

                runCopyAddOnDevice(ctx, secRaw, shaderBytes, ctx.seed() + cycle, 5u + cycle, 0xA5A50000u + cycle);

                secRaw->waitForIdle();
                secRaw->runGarbageCollection();
                secBackend.reset();

                UAA_CHECK_EQ(secMessages.errorCount(), size_t(0),
                    "Secondary device reported errors during cycle " + std::to_string(cycle));
            }

            runCopyAddOnDevice(ctx, ctx.fixture->device(), shaderBytes, ctx.seed() ^ 0xBEEFull, 3u, 0x0F0F0F0Fu);
        }

        void testOptionalFailures(TestContext& ctx)
        {
            nvrhi::IDevice* dev = ctx.fixture->device();
            UAA_CHECK(dev != nullptr, "Device must not be null");

            {
                ExpectErrorsGuard guard(*ctx.fixture);
                const nvrhi::BufferDesc zeroDesc = nvrhi::BufferDesc()
                    .setByteSize(0)
                    .setDebugName("uaa_optfail_zero_buf");
                nvrhi::BufferHandle zeroBuf = dev->createBuffer(zeroDesc);
                UAA_CHECK(zeroBuf == nullptr, "createBuffer with byteSize == 0 must return nullptr");
                if (dev->getGraphicsAPI() == nvrhi::GraphicsAPI::METAL)
                {
                    UAA_CHECK(guard.errorCount() > 0, "Metal must report error callback on zero-byte buffer");
                }
            }

            if (ctx.fixture->isValidationActive() || dev->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
            {
                ExpectErrorsGuard guard(*ctx.fixture);
                const nvrhi::BufferDesc badVolatile = nvrhi::BufferDesc()
                    .setByteSize(256)
                    .setIsVolatile(true)
                    .setIsConstantBuffer(false)
                    .setDebugName("uaa_optfail_bad_volatile");
                nvrhi::BufferHandle h = dev->createBuffer(badVolatile);
                UAA_CHECK(h == nullptr, "Volatile non-constant buffer must return nullptr");
                if (ctx.fixture->isValidationActive())
                {
                    UAA_CHECK(guard.errorCount() > 0, "Validation layer must report error for volatile non-CB");
                }
            }

            if (ctx.fixture->isValidationActive())
            {
                ExpectErrorsGuard guard(*ctx.fixture);
                const nvrhi::TextureDesc badTex = nvrhi::TextureDesc()
                    .setWidth(0)
                    .setHeight(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setDebugName("uaa_optfail_zero_tex");
                nvrhi::TextureHandle t = dev->createTexture(badTex);
                UAA_CHECK(t == nullptr, "createTexture with width == 0 must return nullptr under validation");
                UAA_CHECK(guard.errorCount() > 0, "Validation layer must report error for zero-width texture");
            }

            if (!dev->queryFeatureSupport(nvrhi::Feature::SamplerFeedback) &&
                (ctx.fixture->isValidationActive() || dev->getGraphicsAPI() == nvrhi::GraphicsAPI::METAL))
            {
                nvrhi::TextureHandle paired = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setWidth(16)
                        .setHeight(16)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .setDebugName("uaa_optfail_paired_tex"));
                UAA_CHECK(paired != nullptr, "Paired texture creation failed");

                ExpectErrorsGuard guard(*ctx.fixture);
                nvrhi::SamplerFeedbackTextureHandle sf = dev->createSamplerFeedbackTexture(
                    paired, nvrhi::SamplerFeedbackTextureDesc{});
                UAA_CHECK(sf == nullptr, "Unsupported createSamplerFeedbackTexture must return nullptr");
                UAA_CHECK(guard.errorCount() > 0, "Unsupported createSamplerFeedbackTexture must emit error");
            }

            if (dev->getGraphicsAPI() == nvrhi::GraphicsAPI::METAL)
            {
                ExpectErrorsGuard guard(*ctx.fixture);
                nvrhi::rt::PipelineHandle rtPso = dev->createRayTracingPipeline(nvrhi::rt::PipelineDesc{});
                UAA_CHECK(rtPso == nullptr, "Metal createRayTracingPipeline must return nullptr");
                UAA_CHECK(guard.errorCount() > 0, "Metal createRayTracingPipeline must emit error");
            }
        }
    } // namespace

    UAA_REGISTER_TEST("DEV.CreateDestroy", "core", testCreateDestroy);
    UAA_REGISTER_TEST("DEV.Capabilities", "core", testCapabilities);
    UAA_REGISTER_TEST("DEV.FormatSupport", "core", testFormatSupport);
    UAA_REGISTER_TEST("DEV.MetalFormatClaims", "core,metal", testMetalFormatClaims);
    UAA_REGISTER_TEST("DEV.ResourceDescriptions", "core", testResourceDescriptions);
    UAA_REGISTER_TEST("DEV.GarbageCollection", "core", testGarbageCollection);
    UAA_REGISTER_TEST("DEV.MultipleDevices", "core", testMultipleDevices);
    UAA_REGISTER_TEST("DEV.Reinitialise", "core", testReinitialise);
    UAA_REGISTER_TEST("DEV.OptionalFailures", "core", testOptionalFailures);
} // namespace nvrhi::uaa
