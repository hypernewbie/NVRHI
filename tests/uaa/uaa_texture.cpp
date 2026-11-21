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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct FullscreenPC
        {
            float color[4] = { 0.f, 0.f, 0.f, 1.f };
            float depth = 0.5f;
            float scale = 1.0f;
            float offset[2] = { 0.f, 0.f };
        };

        struct SamplerTestParams
        {
            float uvOrigin[2] = { 0.5f, 0.5f };
            float uvStep[2] = { 0.f, 0.f };
            float lod = 0.0f;
            uint32_t width = 1;
            uint32_t height = 1;
            uint32_t mode = 0;
        };

        struct UavFillParams
        {
            float color[4] = { 0.f, 0.f, 0.f, 1.f };
            uint32_t width = 1;
            uint32_t height = 1;
            uint32_t mode = 0;
            uint32_t pad = 0;
        };

        TestFixture& requireFixture(TestContext& ctx)
        {
            if (!ctx.fixture || !ctx.fixture->device())
            {
                ctx.skip("GPU fixture is not initialised");
            }
            return *ctx.fixture;
        }

        float srgbToLinear(float c)
        {
            if (c <= 0.04045f)
                return c / 12.92f;
            return std::pow((c + 0.055f) / 1.055f, 2.4f);
        }

        // Reads back a single 2D z-plane from a 3D texture by copying to a 2D texture first.
        void readbackTexture3DPlane(
            const TestFixture& fix,
            nvrhi::ITexture* tex3D,
            uint32_t mip,
            uint32_t zPlane,
            std::vector<uint8_t>& outTightPixels,
            TestContext& ctx)
        {
            const nvrhi::TextureDesc& desc3D = tex3D->getDesc();
            const uint32_t mipW = std::max(1u, desc3D.width >> mip);
            const uint32_t mipH = std::max(1u, desc3D.height >> mip);

            nvrhi::TextureHandle plane2D = fix.device()->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2D)
                    .setWidth(mipW)
                    .setHeight(mipH)
                    .setFormat(desc3D.format)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("tex3d_plane_readback"));
            UAA_CHECK(plane2D != nullptr, "Failed to create 2D texture for 3D plane readback");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->copyTexture(
                plane2D,
                nvrhi::TextureSlice().setOrigin(0, 0, 0).setSize(mipW, mipH, 1),
                tex3D,
                nvrhi::TextureSlice().setOrigin(0, 0, zPlane).setSize(mipW, mipH, 1).setMipLevel(mip));
            cmd->close();
            fix.submitAndWait(cmd);

            fix.readbackTexture2D(plane2D, 0, 0, outTightPixels);
        }

        // Creates all supported texture dimensions and validates getDesc() fields.
        void testTexCreate(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::TextureDesc descs[] = {
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture1D)
                    .setWidth(32).setHeight(1).setDepth(1).setArraySize(1).setMipLevels(1)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_1d"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture1DArray)
                    .setWidth(32).setHeight(1).setDepth(1).setArraySize(4).setMipLevels(1)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_1d_array"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2D)
                    .setWidth(32).setHeight(16).setDepth(1).setArraySize(1).setMipLevels(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_2d"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(16).setHeight(16).setDepth(1).setArraySize(4).setMipLevels(3)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_2d_array"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture3D)
                    .setWidth(16).setHeight(8).setDepth(4).setArraySize(1).setMipLevels(3)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_3d"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::TextureCube)
                    .setWidth(16).setHeight(16).setDepth(1).setArraySize(6).setMipLevels(3)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_cube"),
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::TextureCubeArray)
                    .setWidth(16).setHeight(16).setDepth(1).setArraySize(12).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("tex_cube_array")
            };

            for (const nvrhi::TextureDesc& expected : descs)
            {
                nvrhi::TextureHandle tex = dev->createTexture(expected);
                UAA_CHECK(tex != nullptr, "createTexture returned null for " + expected.debugName);

                const nvrhi::TextureDesc& actual = tex->getDesc();
                UAA_CHECK_EQ(actual.dimension, expected.dimension, "dimension mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.width, expected.width, "width mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.height, expected.height, "height mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.depth, expected.depth, "depth mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.arraySize, expected.arraySize, "arraySize mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.mipLevels, expected.mipLevels, "mipLevels mismatch for " + expected.debugName);
                UAA_CHECK_EQ(actual.format, expected.format, "format mismatch for " + expected.debugName);
            }
        }

        // Tests representative integer, normalised, float, sRGB, packed, depth, and block-compressed formats.
        void testTexFormats(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            struct FormatCase
            {
                nvrhi::Format format;
                bool required;
                bool isDepth;
            };

            const FormatCase cases[] = {
                { nvrhi::Format::RGBA8_UINT,         true,  false },
                { nvrhi::Format::R32_UINT,           true,  false },
                { nvrhi::Format::RGBA8_UNORM,        true,  false },
                { nvrhi::Format::BGRA8_UNORM,        true,  false },
                { nvrhi::Format::RGBA16_FLOAT,       true,  false },
                { nvrhi::Format::RGBA32_FLOAT,       true,  false },
                { nvrhi::Format::SRGBA8_UNORM,       true,  false },
                { nvrhi::Format::R10G10B10A2_UNORM,  false, false },
                { nvrhi::Format::R11G11B10_FLOAT,    false, false },
                { nvrhi::Format::BC1_UNORM,          false, false },
                { nvrhi::Format::BC7_UNORM,          false, false },
                { nvrhi::Format::D32,                true,  true  },
                { nvrhi::Format::D24S8,              false, true  },
                { nvrhi::Format::D32S8,              false, true  }
            };

            for (const FormatCase& fc : cases)
            {
                const nvrhi::FormatInfo& fmtInfo = nvrhi::getFormatInfo(fc.format);
                const nvrhi::FormatSupport sup = dev->queryFormatSupport(fc.format);
                const nvrhi::FormatSupport needed = fc.isDepth
                    ? (nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::DepthStencil)
                    : nvrhi::FormatSupport::Texture;

                if ((sup & needed) != needed)
                {
                    if (fc.required)
                    {
                        UAA_CHECK(false, std::string("Required format not supported: ") + fmtInfo.name);
                    }
                    ctx.note(std::string("Format unsupported by queryFormatSupport: ") + fmtInfo.name);
                    continue;
                }

                nvrhi::TextureDesc desc;
                desc.setWidth(8)
                    .setHeight(8)
                    .setFormat(fc.format)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName(std::string("fmt_") + fmtInfo.name);
                if (fc.isDepth)
                {
                    desc.setIsRenderTarget(true);
                }

                nvrhi::TextureHandle tex = dev->createTexture(desc);
                UAA_CHECK(tex != nullptr, std::string("createTexture failed for ") + fmtInfo.name);

                if (fc.isDepth)
                {
                    nvrhi::CommandListHandle cmd = dev->createCommandList();
                    UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                    cmd->open();
                    cmd->clearDepthStencilTexture(tex, nvrhi::AllSubresources, true, 0.5f, fmtInfo.hasStencil, 0x3Cu);
                    cmd->close();
                    fix.submitAndWait(cmd);
                    continue;
                }

                const uint32_t blockSize = std::max<uint32_t>(1u, fmtInfo.blockSize);
                const uint32_t blocksW = (8u + blockSize - 1u) / blockSize;
                const uint32_t blocksH = (8u + blockSize - 1u) / blockSize;
                const size_t rowPitch = size_t(blocksW) * fmtInfo.bytesPerBlock;
                const size_t tightBytes = rowPitch * blocksH;

                const std::vector<uint8_t> expected = TestFixture::makeDeterministicBytes(
                    tightBytes, ctx.seed() + static_cast<uint64_t>(fc.format));

                // Pad CPU staging vector to pixel-height rows for compressed formats in case a backend iterates height rows.
                std::vector<uint8_t> uploadBytes(std::max<size_t>(blocksH, 8u) * rowPitch, 0u);
                std::memcpy(uploadBytes.data(), expected.data(), tightBytes);

                fix.uploadTexture2D(tex, 0, 0, uploadBytes.data(), rowPitch);

                std::vector<uint8_t> actual;
                fix.readbackTexture2D(tex, 0, 0, actual);
                UAA_CHECK_EQ(actual.size(), tightBytes, std::string("Readback size mismatch for ") + fmtInfo.name);

                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), expected.data(), tightBytes, diff),
                    std::string("Format round-trip mismatch for ") + fmtInfo.name + ": " + diff);
            }
        }

        // Uploads distinct patterns across every mip/slice of a 2D array and every mip/z-plane of a 3D texture.
        void testTexUploadReadback(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            // 2D array with mips.
            {
                constexpr uint32_t kWidth = 16;
                constexpr uint32_t kHeight = 16;
                constexpr uint32_t kSlices = 3;
                constexpr uint32_t kMips = 4;

                nvrhi::TextureHandle arrTex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture2DArray)
                        .setWidth(kWidth)
                        .setHeight(kHeight)
                        .setArraySize(kSlices)
                        .setMipLevels(kMips)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("upload_2d_array"));
                UAA_CHECK(arrTex != nullptr, "Failed to create 2D array texture");

                std::vector<std::vector<uint8_t>> expectedAll(kSlices * kMips);
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                for (uint32_t slice = 0; slice < kSlices; ++slice)
                {
                    for (uint32_t mip = 0; mip < kMips; ++mip)
                    {
                        const uint32_t mipW = std::max(1u, kWidth >> mip);
                        const uint32_t mipH = std::max(1u, kHeight >> mip);
                        const size_t rowPitch = size_t(mipW) * 4u;
                        auto& pattern = expectedAll[slice * kMips + mip];
                        pattern = TestFixture::makeDeterministicBytes(rowPitch * mipH, ctx.seed() + slice * 100ull + mip);
                        cmd->writeTexture(arrTex, slice, mip, pattern.data(), rowPitch);
                    }
                }
                cmd->close();
                fix.submitAndWait(cmd);

                for (uint32_t slice = 0; slice < kSlices; ++slice)
                {
                    for (uint32_t mip = 0; mip < kMips; ++mip)
                    {
                        std::vector<uint8_t> actual;
                        fix.readbackTexture2D(arrTex, slice, mip, actual);
                        const auto& expected = expectedAll[slice * kMips + mip];
                        UAA_CHECK_EQ(actual.size(), expected.size(), "2D array readback size mismatch");
                        std::string diff;
                        UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), expected.data(), expected.size(), diff),
                            "2D array mismatch at slice " + std::to_string(slice) + " mip " + std::to_string(mip) + ": " + diff);
                    }
                }
            }

            // 3D texture with mips.
            {
                constexpr uint32_t kWidth = 8;
                constexpr uint32_t kHeight = 8;
                constexpr uint32_t kDepth = 4;
                constexpr uint32_t kMips = 3;

                nvrhi::TextureHandle volTex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture3D)
                        .setWidth(kWidth)
                        .setHeight(kHeight)
                        .setDepth(kDepth)
                        .setMipLevels(kMips)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("upload_3d"));
                UAA_CHECK(volTex != nullptr, "Failed to create 3D texture");

                std::vector<std::vector<uint8_t>> expectedMips(kMips);
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    const uint32_t mipW = std::max(1u, kWidth >> mip);
                    const uint32_t mipH = std::max(1u, kHeight >> mip);
                    const uint32_t mipD = std::max(1u, kDepth >> mip);
                    const size_t rowPitch = size_t(mipW) * 4u;
                    const size_t depthPitch = rowPitch * mipH;
                    expectedMips[mip] = TestFixture::makeDeterministicBytes(depthPitch * mipD, ctx.seed() + 5000ull + mip);
                    cmd->writeTexture(volTex, 0, mip, expectedMips[mip].data(), rowPitch, depthPitch);
                }
                cmd->close();
                fix.submitAndWait(cmd);

                for (uint32_t mip = 0; mip < kMips; ++mip)
                {
                    const uint32_t mipW = std::max(1u, kWidth >> mip);
                    const uint32_t mipH = std::max(1u, kHeight >> mip);
                    const uint32_t mipD = std::max(1u, kDepth >> mip);
                    const size_t planeBytes = size_t(mipW) * mipH * 4u;

                    for (uint32_t z = 0; z < mipD; ++z)
                    {
                        std::vector<uint8_t> actualPlane;
                        readbackTexture3DPlane(fix, volTex, mip, z, actualPlane, ctx);
                        UAA_CHECK_EQ(actualPlane.size(), planeBytes, "3D plane readback size mismatch");
                        std::string diff;
                        UAA_CHECK(TestFixture::checkBytesEqual(
                            actualPlane.data(),
                            expectedMips[mip].data() + z * planeBytes,
                            planeBytes,
                            diff),
                            "3D texel mismatch at mip " + std::to_string(mip) + " z " + std::to_string(z) + ": " + diff);
                    }
                }
            }
        }

        // Tests odd texture dimensions with padded rowPitch and padded depthPitch containing guard bytes.
        void testTexPitches(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            // 2D odd dimensions: 17x13 with row padding.
            {
                constexpr uint32_t kWidth = 17;
                constexpr uint32_t kHeight = 13;
                constexpr size_t kTightRow = kWidth * 4u;
                constexpr size_t kRowPitch = kTightRow + 64u;

                const std::vector<uint8_t> active = TestFixture::makeDeterministicBytes(kTightRow * kHeight, ctx.seed() ^ 0x2D2Dull);
                std::vector<uint8_t> padded(kRowPitch * kHeight, TestFixture::kGuardByte);
                for (uint32_t y = 0; y < kHeight; ++y)
                {
                    std::memcpy(padded.data() + y * kRowPitch, active.data() + y * kTightRow, kTightRow);
                }

                nvrhi::TextureHandle tex2d = dev->createTexture(
                    nvrhi::TextureDesc().setWidth(kWidth).setHeight(kHeight).setFormat(nvrhi::Format::RGBA8_UNORM).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("pitch_2d"));
                UAA_CHECK(tex2d != nullptr, "Failed to create odd-dimension 2D texture");

                fix.uploadTexture2D(tex2d, 0, 0, padded.data(), kRowPitch);
                std::vector<uint8_t> actual;
                fix.readbackTexture2D(tex2d, 0, 0, actual);
                UAA_CHECK_EQ(actual.size(), active.size(), "Odd 2D readback size mismatch");

                std::string diff;
                UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), active.data(), active.size(), diff),
                    "Odd 2D pitch upload mismatch: " + diff);
            }

            // 3D odd dimensions: 7x5x3 with row and depth padding.
            {
                constexpr uint32_t kWidth = 7;
                constexpr uint32_t kHeight = 5;
                constexpr uint32_t kDepth = 3;
                constexpr size_t kTightRow = kWidth * 4u;
                constexpr size_t kRowPitch = kTightRow + 64u;
                constexpr size_t kDepthPitch = kRowPitch * kHeight + 256u;
                constexpr size_t kPlaneBytes = kTightRow * kHeight;

                const std::vector<uint8_t> active = TestFixture::makeDeterministicBytes(kPlaneBytes * kDepth, ctx.seed() ^ 0x3D3Dull);
                std::vector<uint8_t> padded(kDepthPitch * kDepth, TestFixture::kGuardByte);
                for (uint32_t z = 0; z < kDepth; ++z)
                {
                    for (uint32_t y = 0; y < kHeight; ++y)
                    {
                        std::memcpy(
                            padded.data() + z * kDepthPitch + y * kRowPitch,
                            active.data() + z * kPlaneBytes + y * kTightRow,
                            kTightRow);
                    }
                }

                nvrhi::TextureHandle tex3d = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture3D)
                        .setWidth(kWidth)
                        .setHeight(kHeight)
                        .setDepth(kDepth)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("pitch_3d"));
                UAA_CHECK(tex3d != nullptr, "Failed to create odd-dimension 3D texture");

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->writeTexture(tex3d, 0, 0, padded.data(), kRowPitch, kDepthPitch);
                cmd->close();
                fix.submitAndWait(cmd);

                for (uint32_t z = 0; z < kDepth; ++z)
                {
                    std::vector<uint8_t> actualPlane;
                    readbackTexture3DPlane(fix, tex3d, 0, z, actualPlane, ctx);
                    UAA_CHECK_EQ(actualPlane.size(), kPlaneBytes, "Odd 3D plane readback size mismatch");

                    std::string diff;
                    UAA_CHECK(TestFixture::checkBytesEqual(
                        actualPlane.data(),
                        active.data() + z * kPlaneBytes,
                        kPlaneBytes,
                        diff),
                        "Odd 3D pitch mismatch at z=" + std::to_string(z) + ": " + diff);
                }
            }
        }

        // Copies a nonzero sub-region across mips and slices and checks surrounding background pixels remain untouched.
        void testTexCopyRegions(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle src = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(32).setHeight(32).setArraySize(3).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("copy_src"));
            nvrhi::TextureHandle dst = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(16).setHeight(16).setArraySize(2).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("copy_dst"));
            UAA_CHECK(src != nullptr && dst != nullptr, "Failed to create textures for TEX.CopyRegions");

            const std::vector<uint8_t> srcMip1Slice2 = TestFixture::makeDeterministicBytes(16 * 16 * 4, ctx.seed() + 10);
            const std::vector<uint8_t> bgSlice0 = TestFixture::makeDeterministicBytes(16 * 16 * 4, ctx.seed() + 20);
            const std::vector<uint8_t> bgSlice1 = TestFixture::makeDeterministicBytes(16 * 16 * 4, ctx.seed() + 30);

            constexpr uint32_t kSrcX = 3;
            constexpr uint32_t kSrcY = 5;
            constexpr uint32_t kDstX = 2;
            constexpr uint32_t kDstY = 4;
            constexpr uint32_t kCopyW = 7;
            constexpr uint32_t kCopyH = 6;

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->writeTexture(src, 2, 1, srcMip1Slice2.data(), 16 * 4);
            cmd->writeTexture(dst, 0, 0, bgSlice0.data(), 16 * 4);
            cmd->writeTexture(dst, 1, 0, bgSlice1.data(), 16 * 4);
            cmd->copyTexture(
                dst,
                nvrhi::TextureSlice().setOrigin(kDstX, kDstY, 0).setSize(kCopyW, kCopyH, 1).setMipLevel(0).setArraySlice(1),
                src,
                nvrhi::TextureSlice().setOrigin(kSrcX, kSrcY, 0).setSize(kCopyW, kCopyH, 1).setMipLevel(1).setArraySlice(2));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> expectedSlice1 = bgSlice1;
            for (uint32_t y = 0; y < kCopyH; ++y)
            {
                std::memcpy(
                    expectedSlice1.data() + ((kDstY + y) * 16 + kDstX) * 4,
                    srcMip1Slice2.data() + ((kSrcY + y) * 16 + kSrcX) * 4,
                    kCopyW * 4);
            }

            std::vector<uint8_t> actualSlice0;
            std::vector<uint8_t> actualSlice1;
            fix.readbackTexture2D(dst, 0, 0, actualSlice0);
            fix.readbackTexture2D(dst, 1, 0, actualSlice1);

            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(actualSlice0.data(), bgSlice0.data(), bgSlice0.size(), diff),
                "Untouched slice 0 corrupted by copyTexture: " + diff);
            UAA_CHECK(TestFixture::checkBytesEqual(actualSlice1.data(), expectedSlice1.data(), expectedSlice1.size(), diff),
                "Sub-region copy or surrounding texels mismatch on slice 1: " + diff);
        }

        // Tests CPU Write and Read staging textures across multiple mip levels.
        void testTexStaging(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kSize = 16;
            constexpr uint32_t kMips = 3;
            const nvrhi::TextureDesc desc = nvrhi::TextureDesc()
                .setWidth(kSize).setHeight(kSize).setMipLevels(kMips)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                .setDebugName("staging_tex");

            nvrhi::TextureHandle devTex = dev->createTexture(desc);
            nvrhi::StagingTextureHandle writeStaging = dev->createStagingTexture(desc, nvrhi::CpuAccessMode::Write);
            nvrhi::StagingTextureHandle readStaging = dev->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);
            UAA_CHECK(devTex != nullptr && writeStaging != nullptr && readStaging != nullptr,
                "Failed to create device or staging textures");

            std::vector<std::vector<uint8_t>> expectedMips(kMips);
            for (uint32_t mip = 0; mip < kMips; ++mip)
            {
                const uint32_t mipW = std::max(1u, kSize >> mip);
                const uint32_t mipH = std::max(1u, kSize >> mip);
                const size_t tightRow = size_t(mipW) * 4u;
                expectedMips[mip] = TestFixture::makeDeterministicBytes(tightRow * mipH, ctx.seed() + 900ull + mip);

                size_t rowPitch = 0;
                void* mapped = dev->mapStagingTexture(
                    writeStaging, nvrhi::TextureSlice().setMipLevel(mip), nvrhi::CpuAccessMode::Write, &rowPitch);
                UAA_CHECK(mapped != nullptr, "mapStagingTexture(Write) returned null at mip " + std::to_string(mip));
                UAA_CHECK(rowPitch >= tightRow, "Write staging rowPitch smaller than tight row bytes");

                auto* dstBytes = static_cast<uint8_t*>(mapped);
                for (uint32_t r = 0; r < mipH; ++r)
                {
                    std::memcpy(dstBytes + r * rowPitch, expectedMips[mip].data() + r * tightRow, tightRow);
                }
                dev->unmapStagingTexture(writeStaging);
            }

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            for (uint32_t mip = 0; mip < kMips; ++mip)
            {
                const nvrhi::TextureSlice slice = nvrhi::TextureSlice().setMipLevel(mip);
                cmd->copyTexture(devTex, slice, writeStaging, slice);
            }
            for (uint32_t mip = 0; mip < kMips; ++mip)
            {
                const nvrhi::TextureSlice slice = nvrhi::TextureSlice().setMipLevel(mip);
                cmd->copyTexture(readStaging, slice, devTex, slice);
            }
            cmd->close();
            fix.submitAndWait(cmd);

            for (uint32_t mip = 0; mip < kMips; ++mip)
            {
                const uint32_t mipW = std::max(1u, kSize >> mip);
                const uint32_t mipH = std::max(1u, kSize >> mip);
                const size_t tightRow = size_t(mipW) * 4u;

                size_t rowPitch = 0;
                void* mapped = dev->mapStagingTexture(
                    readStaging, nvrhi::TextureSlice().setMipLevel(mip), nvrhi::CpuAccessMode::Read, &rowPitch);
                UAA_CHECK(mapped != nullptr, "mapStagingTexture(Read) returned null at mip " + std::to_string(mip));
                UAA_CHECK(rowPitch >= tightRow, "Read staging rowPitch smaller than tight row bytes");

                const auto* srcBytes = static_cast<const uint8_t*>(mapped);
                std::string diff;
                for (uint32_t r = 0; r < mipH; ++r)
                {
                    const bool rowOk = TestFixture::checkBytesEqual(
                        srcBytes + r * rowPitch, expectedMips[mip].data() + r * tightRow, tightRow, diff);
                    if (!rowOk)
                    {
                        dev->unmapStagingTexture(readStaging);
                        UAA_CHECK(false, "Staging readback mismatch at mip " + std::to_string(mip) + " row " + std::to_string(r) + ": " + diff);
                    }
                }
                dev->unmapStagingTexture(readStaging);
            }
        }

        // Tests clearTextureFloat and clearTextureUInt on a single (mip=1, slice=1) subresource subset.
        void testTexClearFloatUInt(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kSize = 8;
            constexpr uint32_t kSlices = 2;
            constexpr uint32_t kMips = 2;

            auto fillAllSubresources = [&](nvrhi::ITexture* tex, uint32_t bytesPerPixel, uint64_t seedBase,
                                           std::vector<std::vector<uint8_t>>& outPrior)
            {
                outPrior.resize(kSlices * kMips);
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                for (uint32_t s = 0; s < kSlices; ++s)
                {
                    for (uint32_t m = 0; m < kMips; ++m)
                    {
                        const uint32_t mipW = std::max(1u, kSize >> m);
                        const uint32_t mipH = std::max(1u, kSize >> m);
                        auto& data = outPrior[s * kMips + m];
                        data = TestFixture::makeDeterministicBytes(size_t(mipW) * mipH * bytesPerPixel, seedBase + s * 10 + m);
                        cmd->writeTexture(tex, s, m, data.data(), size_t(mipW) * bytesPerPixel);
                    }
                }
                cmd->close();
                fix.submitAndWait(cmd);
            };

            auto checkUntouchedSubresources = [&](nvrhi::ITexture* tex, const std::vector<std::vector<uint8_t>>& prior, const std::string& label)
            {
                for (uint32_t s = 0; s < kSlices; ++s)
                {
                    for (uint32_t m = 0; m < kMips; ++m)
                    {
                        if (s == 1 && m == 1)
                            continue;
                        std::vector<uint8_t> actual;
                        fix.readbackTexture2D(tex, s, m, actual);
                        const auto& expected = prior[s * kMips + m];
                        std::string diff;
                        UAA_CHECK(TestFixture::checkBytesEqual(actual.data(), expected.data(), expected.size(), diff),
                            label + " modified untouched slice=" + std::to_string(s) + " mip=" + std::to_string(m) + ": " + diff);
                    }
                }
            };

            // RGBA8_UNORM and BGRA8_UNORM channel-order clear checks.
            for (const nvrhi::Format fmt : { nvrhi::Format::RGBA8_UNORM, nvrhi::Format::BGRA8_UNORM })
            {
                const std::string fmtName = (fmt == nvrhi::Format::BGRA8_UNORM) ? "BGRA8_UNORM" : "RGBA8_UNORM";
                nvrhi::TextureHandle tex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture2DArray)
                        .setWidth(kSize).setHeight(kSize).setArraySize(kSlices).setMipLevels(kMips)
                        .setFormat(fmt).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("clear_" + fmtName));
                UAA_CHECK(tex != nullptr, "Failed to create " + fmtName);

                std::vector<std::vector<uint8_t>> prior;
                fillAllSubresources(tex, 4u, ctx.seed() + static_cast<uint64_t>(fmt), prior);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(tex, nvrhi::TextureSubresourceSet(1, 1, 1, 1), nvrhi::Color(1.0f, 0.5f, 0.25f, 0.75f));
                cmd->close();
                fix.submitAndWait(cmd);

                checkUntouchedSubresources(tex, prior, fmtName);

                std::vector<uint8_t> cleared;
                fix.readbackTexture2D(tex, 1, 1, cleared);
                UAA_CHECK_EQ(cleared.size(), size_t(4 * 4 * 4), fmtName + " cleared mip size mismatch");

                const uint8_t exp0 = (fmt == nvrhi::Format::BGRA8_UNORM) ? 64u : 255u;
                const uint8_t exp1 = 128u;
                const uint8_t exp2 = (fmt == nvrhi::Format::BGRA8_UNORM) ? 255u : 64u;
                const uint8_t exp3 = 191u;
                std::string diff;
                for (size_t p = 0; p < 16; ++p)
                {
                    UAA_CHECK(TestFixture::checkPixelRgba8(&cleared[p * 4], exp0, exp1, exp2, exp3, 2, diff),
                        fmtName + " cleared channel order mismatch at pixel " + std::to_string(p) + ": " + diff);
                }
            }

            // RGBA32_FLOAT clear check.
            {
                nvrhi::TextureHandle fTex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture2DArray)
                        .setWidth(kSize).setHeight(kSize).setArraySize(kSlices).setMipLevels(kMips)
                        .setFormat(nvrhi::Format::RGBA32_FLOAT).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("clear_RGBA32F"));
                UAA_CHECK(fTex != nullptr, "Failed to create RGBA32_FLOAT texture");

                std::vector<std::vector<uint8_t>> prior;
                fillAllSubresources(fTex, 16u, ctx.seed() + 300ull, prior);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(fTex, nvrhi::TextureSubresourceSet(1, 1, 1, 1), nvrhi::Color(0.25f, -0.5f, 1.5f, 0.75f));
                cmd->close();
                fix.submitAndWait(cmd);

                checkUntouchedSubresources(fTex, prior, "RGBA32_FLOAT");

                std::vector<uint8_t> cleared;
                fix.readbackTexture2D(fTex, 1, 1, cleared);
                UAA_CHECK_EQ(cleared.size(), size_t(4 * 4 * 16), "RGBA32_FLOAT cleared mip size mismatch");
                const float expected[4] = { 0.25f, -0.5f, 1.5f, 0.75f };
                for (size_t p = 0; p < 16; ++p)
                {
                    float rgba[4]{};
                    std::memcpy(rgba, &cleared[p * 16], 16);
                    for (int c = 0; c < 4; ++c)
                    {
                        UAA_CHECK_NEAR(rgba[c], expected[c], 1e-5f,
                            "RGBA32_FLOAT clear mismatch at pixel " + std::to_string(p) + " ch " + std::to_string(c));
                    }
                }
            }

            // R32_UINT and RGBA8_UINT clearTextureUInt checks.
            {
                nvrhi::TextureHandle u32Tex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture2DArray)
                        .setWidth(kSize).setHeight(kSize).setArraySize(kSlices).setMipLevels(kMips)
                        .setFormat(nvrhi::Format::R32_UINT).setIsUAV(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("clear_R32U"));
                UAA_CHECK(u32Tex != nullptr, "Failed to create R32_UINT texture");

                std::vector<std::vector<uint8_t>> prior;
                fillAllSubresources(u32Tex, 4u, ctx.seed() + 400ull, prior);

                constexpr uint32_t kClearU32 = 0xCAFEBABEu;
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureUInt(u32Tex, nvrhi::TextureSubresourceSet(1, 1, 1, 1), kClearU32);
                cmd->close();
                fix.submitAndWait(cmd);

                checkUntouchedSubresources(u32Tex, prior, "R32_UINT");

                std::vector<uint8_t> cleared;
                fix.readbackTexture2D(u32Tex, 1, 1, cleared);
                UAA_CHECK_EQ(cleared.size(), size_t(4 * 4 * 4), "R32_UINT cleared size mismatch");
                for (size_t p = 0; p < 16; ++p)
                {
                    uint32_t v = 0;
                    std::memcpy(&v, &cleared[p * 4], 4);
                    UAA_CHECK_EQ(v, kClearU32, "R32_UINT clear value mismatch at pixel " + std::to_string(p));
                }
            }

            {
                nvrhi::TextureHandle u8Tex = dev->createTexture(
                    nvrhi::TextureDesc()
                        .setDimension(nvrhi::TextureDimension::Texture2DArray)
                        .setWidth(kSize).setHeight(kSize).setArraySize(kSlices).setMipLevels(kMips)
                        .setFormat(nvrhi::Format::RGBA8_UINT).setIsUAV(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("clear_RGBA8U"));
                UAA_CHECK(u8Tex != nullptr, "Failed to create RGBA8_UINT texture");

                std::vector<std::vector<uint8_t>> prior;
                fillAllSubresources(u8Tex, 4u, ctx.seed() + 500ull, prior);

                constexpr uint32_t kClearU8 = 77u;
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureUInt(u8Tex, nvrhi::TextureSubresourceSet(1, 1, 1, 1), kClearU8);
                cmd->close();
                fix.submitAndWait(cmd);

                checkUntouchedSubresources(u8Tex, prior, "RGBA8_UINT");

                std::vector<uint8_t> cleared;
                fix.readbackTexture2D(u8Tex, 1, 1, cleared);
                UAA_CHECK_EQ(cleared.size(), size_t(4 * 4 * 4), "RGBA8_UINT cleared size mismatch");
                std::string diff;
                for (size_t p = 0; p < 16; ++p)
                {
                    UAA_CHECK(TestFixture::checkPixelRgba8(&cleared[p * 4], 77u, 77u, 77u, 77u, 0, diff),
                        "RGBA8_UINT clear value mismatch at pixel " + std::to_string(p) + ": " + diff);
                }
            }
        }

        // Tests depth-only clear, combined/separate depth+stencil clears, and read-only DS framebuffer attachments.
        void testTexDepthStencil(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);

            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC))));
            UAA_CHECK(layout != nullptr, "Failed to create push-constant binding layout");

            nvrhi::BindingSetHandle set = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPC))),
                layout);
            UAA_CHECK(set != nullptr, "Failed to create push-constant binding set");

            nvrhi::TextureHandle colorTex = dev->createTexture(
                nvrhi::TextureDesc().setWidth(4).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("ds_color"));
            UAA_CHECK(colorTex != nullptr, "Failed to create DS test color texture");

            auto makePipeline = [&](const nvrhi::FramebufferInfo& fbInfo, const nvrhi::DepthStencilState& dsState) -> nvrhi::GraphicsPipelineHandle
            {
                nvrhi::GraphicsPipelineDesc desc;
                desc.setVertexShader(vs)
                    .setPixelShader(ps)
                    .addBindingLayout(layout);
                desc.renderState.rasterState.setCullNone();
                desc.renderState.depthStencilState = dsState;
                nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(desc, fbInfo);
                UAA_CHECK(pso != nullptr, "Failed to create DS test graphics pipeline");
                return pso;
            };

            auto checkColorRgba = [&](uint8_t r, uint8_t g, uint8_t b, uint8_t a, const std::string& label)
            {
                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(colorTex, 0, 0, pixels);
                std::string diff;
                for (size_t p = 0; p < 16; ++p)
                {
                    UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[p * 4], r, g, b, a, 2, diff),
                        label + " pixel " + std::to_string(p) + ": " + diff);
                }
            };

            // Part 1: D32 depth-only clear & test.
            {
                nvrhi::TextureHandle d32Tex = dev->createTexture(
                    nvrhi::TextureDesc().setWidth(4).setHeight(4).setFormat(nvrhi::Format::D32).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("ds_d32"));
                UAA_CHECK(d32Tex != nullptr, "Failed to create D32 texture");

                nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                    nvrhi::FramebufferDesc().addColorAttachment(colorTex).setDepthAttachment(d32Tex));
                UAA_CHECK(fb != nullptr, "Failed to create D32 framebuffer");

                nvrhi::DepthStencilState dsLess;
                dsLess.enableDepthTest().enableDepthWrite().setDepthFunc(nvrhi::ComparisonFunc::Less);
                nvrhi::GraphicsPipelineHandle pso = makePipeline(fb->getFramebufferInfo(), dsLess);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(colorTex, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                cmd->clearDepthStencilTexture(d32Tex, nvrhi::AllSubresources, true, 0.5f, false, 0);

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(pso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .addBindingSet(set));

                // Depth 0.75 > 0.5 -> rejected.
                const FullscreenPC pcFail{ { 1.f, 0.f, 0.f, 1.f }, 0.75f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcFail, sizeof(pcFail));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

                // Depth 0.25 < 0.5 -> accepted (green).
                const FullscreenPC pcPass{ { 0.f, 1.f, 0.f, 1.f }, 0.25f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcPass, sizeof(pcPass));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                checkColorRgba(0, 255, 0, 255, "D32 depth test");
            }

            // Part 2: D24S8 or D32S8 combined/separate clears and read-only attachment.
            nvrhi::Format dsFormat = nvrhi::Format::UNKNOWN;
            for (const nvrhi::Format candidate : { nvrhi::Format::D24S8, nvrhi::Format::D32S8 })
            {
                if ((dev->queryFormatSupport(candidate) & nvrhi::FormatSupport::DepthStencil) != 0)
                {
                    dsFormat = candidate;
                    break;
                }
            }
            UAA_CHECK(dsFormat != nvrhi::Format::UNKNOWN, "Neither D24S8 nor D32S8 is supported");

            nvrhi::TextureHandle dsTex = dev->createTexture(
                nvrhi::TextureDesc().setWidth(4).setHeight(4).setFormat(dsFormat).setIsRenderTarget(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("ds_combined"));
            UAA_CHECK(dsTex != nullptr, "Failed to create depth-stencil texture");

            nvrhi::FramebufferHandle fbRW = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(colorTex).setDepthAttachment(dsTex));
            nvrhi::FramebufferHandle fbRO = dev->createFramebuffer(
                nvrhi::FramebufferDesc()
                    .addColorAttachment(colorTex)
                    .setDepthAttachment(nvrhi::FramebufferAttachment().setTexture(dsTex).setReadOnly(true)));
            UAA_CHECK(fbRW != nullptr && fbRO != nullptr, "Failed to create RW/RO depth-stencil framebuffers");

            nvrhi::DepthStencilState dsState;
            dsState.enableDepthTest()
                .disableDepthWrite()
                .setDepthFunc(nvrhi::ComparisonFunc::Less)
                .enableStencil()
                .setStencilWriteMask(0)
                .setDynamicStencilRef(true);
            dsState.frontFaceStencil.setStencilFunc(nvrhi::ComparisonFunc::Equal);
            dsState.backFaceStencil.setStencilFunc(nvrhi::ComparisonFunc::Equal);

            nvrhi::GraphicsPipelineHandle psoRW = makePipeline(fbRW->getFramebufferInfo(), dsState);
            nvrhi::GraphicsPipelineHandle psoRO = makePipeline(fbRO->getFramebufferInfo(), dsState);

            // Step A: Clear depth=0.5 and stencil=0x2A together.
            {
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(colorTex, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 0.5f, true, 0x2Au);

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(psoRW)
                    .setFramebuffer(fbRW)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .setDynamicStencilRefValue(0x2Au)
                    .addBindingSet(set));
                const FullscreenPC pcPass{ { 0.f, 0.f, 1.f, 1.f }, 0.25f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcPass, sizeof(pcPass));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(psoRW)
                    .setFramebuffer(fbRW)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .setDynamicStencilRefValue(0x11u)
                    .addBindingSet(set));
                const FullscreenPC pcWrongStencil{ { 1.f, 0.f, 0.f, 1.f }, 0.25f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcWrongStencil, sizeof(pcWrongStencil));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                checkColorRgba(0, 0, 255, 255, "Combined depth+stencil clear");
            }

            // Step B: Clear depth=0.8 only (stencil must remain 0x2A).
            {
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(colorTex, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, true, 0.8f, false, 0u);

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(psoRW)
                    .setFramebuffer(fbRW)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .setDynamicStencilRefValue(0x2Au)
                    .addBindingSet(set));
                const FullscreenPC pcPass{ { 1.f, 1.f, 0.f, 1.f }, 0.6f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcPass, sizeof(pcPass));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                checkColorRgba(255, 255, 0, 255, "Depth-only clear preserving stencil");
            }

            // Step C: Clear stencil=0x55 only (depth must remain 0.8), then render with read-only DS attachment.
            {
                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(colorTex, nvrhi::AllSubresources, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                cmd->clearDepthStencilTexture(dsTex, nvrhi::AllSubresources, false, 0.0f, true, 0x55u);

                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(psoRO)
                    .setFramebuffer(fbRO)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                    .setDynamicStencilRefValue(0x55u)
                    .addBindingSet(set));
                // Depth 0.9 > 0.8 -> rejected if depth preserved.
                const FullscreenPC pcFail{ { 1.f, 0.f, 0.f, 1.f }, 0.9f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcFail, sizeof(pcFail));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));

                // Depth 0.65 < 0.8 and stencil == 0x55 -> accepted (cyan).
                const FullscreenPC pcPass{ { 0.f, 1.f, 1.f, 1.f }, 0.65f, 1.0f, { 0.f, 0.f } };
                cmd->setPushConstants(&pcPass, sizeof(pcPass));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                checkColorRgba(0, 255, 255, 255, "Stencil-only clear + read-only DS attachment");
            }
        }

        // Tests Texture_SRV view cache keys across format, mip, slice, default swizzle, and override swizzle.
        void testTexViews(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            const nvrhi::ComponentMapping defaultSwizzle = nvrhi::ComponentMapping()
                .setR(nvrhi::ComponentSwizzle::B)
                .setG(nvrhi::ComponentSwizzle::G)
                .setB(nvrhi::ComponentSwizzle::R)
                .setA(nvrhi::ComponentSwizzle::One);
            const nvrhi::ComponentMapping customSwizzle = nvrhi::ComponentMapping()
                .setR(nvrhi::ComponentSwizzle::G)
                .setG(nvrhi::ComponentSwizzle::R)
                .setB(nvrhi::ComponentSwizzle::A)
                .setA(nvrhi::ComponentSwizzle::Zero);

            nvrhi::TextureHandle tex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2D)
                    .setWidth(8).setHeight(8).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsTypeless(true)
                    .setDefaultComponentMapping(defaultSwizzle)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("views_tex"));
            UAA_CHECK(tex != nullptr, "Failed to create typeless swizzled texture");

            auto uploadSolid = [&](uint32_t mip, uint32_t w, uint32_t h, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
            {
                std::vector<uint8_t> pixels(w * h * 4);
                for (size_t i = 0; i < w * h; ++i)
                {
                    pixels[i * 4 + 0] = r;
                    pixels[i * 4 + 1] = g;
                    pixels[i * 4 + 2] = b;
                    pixels[i * 4 + 3] = a;
                }
                fix.uploadTexture2D(tex, 0, mip, pixels.data(), w * 4);
            };

            uploadSolid(0, 8, 8, 200, 100, 40, 128);
            uploadSolid(1, 4, 4, 40, 200, 100, 128);

            nvrhi::SamplerHandle samp = dev->createSampler(
                nvrhi::SamplerDesc().setAllFilters(false).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
            nvrhi::BufferHandle outBuf = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(16).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("views_out"));
            nvrhi::BufferHandle cb = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("views_cb"));
            UAA_CHECK(samp != nullptr && outBuf != nullptr && cb != nullptr, "Failed to create sampler/buffers for TEX.Views");

            const SamplerTestParams params{ { 0.5f, 0.5f }, { 0.f, 0.f }, 0.0f, 1u, 1u, 0u };
            fix.uploadBuffer(cb, &params, sizeof(params), 0);

            nvrhi::ShaderHandle cs = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(layout != nullptr, "Failed to create views binding layout");

            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(cs).addBindingLayout(layout));
            UAA_CHECK(pso != nullptr, "Failed to create views compute pipeline");

            // Array views must use Texture2DArray. Check this legal slice view
            // separately because the sampling shader below declares Texture2D.
            nvrhi::TextureHandle arrayTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(8).setHeight(8).setArraySize(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("views_array_tex"));
            UAA_CHECK(arrayTex != nullptr, "Failed to create array texture for view dimension check");
            nvrhi::BindingSetHandle arrayViewSet = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::Texture_SRV(0, arrayTex, nvrhi::Format::RGBA8_UNORM,
                        nvrhi::TextureSubresourceSet(0, 1, 1, 1), nvrhi::TextureDimension::Texture2DArray))
                    .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                layout);
            UAA_CHECK(arrayViewSet != nullptr, "Failed to create legal Texture2DArray slice view");

            auto sampleView = [&](
                nvrhi::Format fmt,
                uint32_t mip,
                std::optional<nvrhi::ComponentMapping> overrideMapping,
                const float expected[4],
                const std::string& label)
            {
                nvrhi::BindingSetHandle set = dev->createBindingSet(
                    nvrhi::BindingSetDesc()
                        .addItem(nvrhi::BindingSetItem::Texture_SRV(
                            0, tex, fmt, nvrhi::TextureSubresourceSet(mip, 1, 0, 1),
                            nvrhi::TextureDimension::Texture2D, overrideMapping))
                        .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf))
                        .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, cb)),
                    layout);
                UAA_CHECK(set != nullptr, "Failed to create binding set for view: " + label);

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
                    UAA_CHECK_NEAR(rgba[c], expected[c], 0.01f,
                        "TEX.Views (" + label + ") channel " + std::to_string(c) + " mismatch");
                }
            };

            const float r00 = 200.f / 255.f;
            const float g00 = 100.f / 255.f;
            const float b00 = 40.f / 255.f;
            const float a00 = 128.f / 255.f;

            const bool portabilitySubset = std::find(
                fix.deviceInfo().enabledExtensions.begin(), fix.deviceInfo().enabledExtensions.end(),
                "VK_KHR_portability_subset") != fix.deviceInfo().enabledExtensions.end();
            const bool supportsSwizzle = fix.api() != nvrhi::GraphicsAPI::D3D11 &&
                !(fix.api() == nvrhi::GraphicsAPI::VULKAN && portabilitySubset);
            if (supportsSwizzle)
            {
                // Inherit defaultComponentMapping {B, G, R, One}.
                const float expDefault[4] = { b00, g00, r00, 1.0f };
                sampleView(nvrhi::Format::RGBA8_UNORM, 0, std::nullopt, expDefault, "inherit_default_swizzle");

                // Force identity override on a texture with non-identity defaultComponentMapping.
                const float expIdentity[4] = { r00, g00, b00, a00 };
                sampleView(nvrhi::Format::RGBA8_UNORM, 0, nvrhi::ComponentMapping(), expIdentity, "force_identity_override");

                // Custom override {G, R, A, Zero}.
                const float expCustom[4] = { g00, r00, a00, 0.0f };
                sampleView(nvrhi::Format::RGBA8_UNORM, 0, customSwizzle, expCustom, "custom_override_swizzle");
            }
            else
            {
                ctx.note("Backend/device does not support non-identity ComponentMapping; skipping swizzle subchecks");
            }

            // Alternate format SRGBA8_UNORM on the same typeless texture (with identity mapping).
            const std::optional<nvrhi::ComponentMapping> idOverride = nvrhi::ComponentMapping();
            const float expSrgb[4] = { srgbToLinear(r00), srgbToLinear(g00), srgbToLinear(b00), a00 };
            sampleView(nvrhi::Format::SRGBA8_UNORM, 0, idOverride, expSrgb, "srgb_format_view");

            // Mip subset (mip=1).
            const float expMip1[4] = { 40.f / 255.f, 200.f / 255.f, 100.f / 255.f, 128.f / 255.f };
            sampleView(nvrhi::Format::RGBA8_UNORM, 1, idOverride, expMip1, "mip1_view");
        }

        // Tests resolving a 4x4 4x-MSAA texture into a selected mip/slice of a non-MSAA destination.
        void testTexResolve(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            nvrhi::TextureHandle msaaTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DMS)
                    .setWidth(4).setHeight(4).setSampleCount(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("resolve_msaa"));
            UAA_CHECK(msaaTex != nullptr, "Failed to create 4x-MSAA Texture2DMS");

            nvrhi::TextureHandle dstTex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(8).setHeight(8).setArraySize(2).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("resolve_dst"));
            UAA_CHECK(dstTex != nullptr, "Failed to create resolve destination Texture2DArray");

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);

            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC))));
            UAA_CHECK(layout != nullptr, "Failed to create resolve binding layout");

            nvrhi::BindingSetHandle set = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPC))),
                layout);
            UAA_CHECK(set != nullptr, "Failed to create resolve binding set");

            nvrhi::FramebufferHandle fb = dev->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(msaaTex));
            UAA_CHECK(fb != nullptr, "Failed to create MSAA framebuffer");

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.setVertexShader(vs).setPixelShader(ps).addBindingLayout(layout);
            psoDesc.renderState.rasterState.setCullNone();
            psoDesc.renderState.depthStencilState.disableDepthTest().disableDepthWrite();
            nvrhi::GraphicsPipelineHandle pso = dev->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "Failed to create MSAA graphics pipeline");

            const FullscreenPC pc{ { 0.25f, 0.75f, 0.5f, 1.0f }, 0.5f, 1.0f, { 0.f, 0.f } };

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->clearTextureFloat(dstTex, nvrhi::AllSubresources, nvrhi::Color(0.9f, 0.1f, 0.1f, 1.0f));
            cmd->setGraphicsState(nvrhi::GraphicsState()
                .setPipeline(pso)
                .setFramebuffer(fb)
                .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f)))
                .addBindingSet(set));
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->resolveTexture(
                dstTex, nvrhi::TextureSubresourceSet(1, 1, 1, 1),
                msaaTex, nvrhi::TextureSubresourceSet(0, 1, 0, 1));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> resolved;
            fix.readbackTexture2D(dstTex, 1, 1, resolved);
            UAA_CHECK_EQ(resolved.size(), size_t(4 * 4 * 4), "Resolved subresource size mismatch");

            std::string diff;
            for (size_t p = 0; p < 16; ++p)
            {
                UAA_CHECK(TestFixture::checkPixelRgba8(&resolved[p * 4], 64, 191, 128, 255, 2, diff),
                    "Resolved pixel mismatch at " + std::to_string(p) + ": " + diff);
            }

            std::vector<uint8_t> untouched;
            fix.readbackTexture2D(dstTex, 0, 0, untouched);
            UAA_CHECK_EQ(untouched.size(), size_t(8 * 8 * 4), "Untouched subresource size mismatch");
            for (size_t p = 0; p < 64; ++p)
            {
                UAA_CHECK(TestFixture::checkPixelRgba8(&untouched[p * 4], 230, 26, 26, 255, 2, diff),
                    "Untouched subresource corrupted at pixel " + std::to_string(p) + ": " + diff);
            }
        }

        void testTexMsaaArrayResolve(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (fix.api() != nvrhi::GraphicsAPI::METAL)
                ctx.skip("Multisampled array support is not exposed by NVRHI");
            nvrhi::IDevice* dev = fix.device();
            nvrhi::TextureHandle source = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DMSArray)
                    .setWidth(4).setHeight(4).setArraySize(3).setSampleCount(4)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("resolve_array_src"));
            nvrhi::TextureHandle dest = dev->createTexture(
                nvrhi::TextureDesc()
                    .setDimension(nvrhi::TextureDimension::Texture2DArray)
                    .setWidth(4).setHeight(4).setArraySize(3)
                    .setFormat(nvrhi::Format::RGBA8_UNORM).setIsRenderTarget(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("resolve_array_dst"));
            UAA_CHECK(source && dest, "Failed to create MSAA array resolve textures");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned null");
            cmd->open();
            cmd->clearTextureFloat(source, nvrhi::TextureSubresourceSet(0, 1, 0, 1), nvrhi::Color(1.f, 0.f, 0.f, 1.f));
            cmd->clearTextureFloat(source, nvrhi::TextureSubresourceSet(0, 1, 1, 1), nvrhi::Color(0.f, 1.f, 0.f, 1.f));
            cmd->clearTextureFloat(source, nvrhi::TextureSubresourceSet(0, 1, 2, 1), nvrhi::Color(0.f, 0.f, 1.f, 1.f));
            cmd->resolveTexture(dest, nvrhi::TextureSubresourceSet(0, 1, 2, 1),
                source, nvrhi::TextureSubresourceSet(0, 1, 1, 1));
            cmd->resolveTexture(dest, nvrhi::TextureSubresourceSet(0, 1, 0, 1),
                source, nvrhi::TextureSubresourceSet(0, 1, 2, 1));
            cmd->resolveTexture(dest, nvrhi::TextureSubresourceSet(0, 1, 1, 1),
                source, nvrhi::TextureSubresourceSet(0, 1, 0, 1));
            cmd->close();
            fix.submitAndWait(cmd);

            const uint8_t expected[3][4] = { {0, 0, 255, 255}, {255, 0, 0, 255}, {0, 255, 0, 255} };
            for (uint32_t slice = 0; slice < 3; ++slice)
            {
                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(dest, slice, 0, pixels);
                UAA_CHECK_EQ(pixels.size(), size_t(4 * 4 * 4), "Resolved array slice size mismatch");
                std::string diff;
                for (size_t p = 0; p < 16; ++p)
                    UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[p * 4], expected[slice][0], expected[slice][1],
                        expected[slice][2], expected[slice][3], 2, diff), "Resolved array slice mismatch: " + diff);
            }
        }

        // Tests upload->sample, UAV->sample, render->copy, and copy->render hazards within single command lists.
        void testTexReadWriteHazards(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kSize = 8;
            nvrhi::TextureHandle tex = dev->createTexture(
                nvrhi::TextureDesc()
                    .setWidth(kSize).setHeight(kSize)
                    .setFormat(nvrhi::Format::RGBA32_FLOAT)
                    .setIsRenderTarget(true)
                    .setIsUAV(true)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("hazard_tex"));
            nvrhi::StagingTextureHandle readStaging = dev->createStagingTexture(tex->getDesc(), nvrhi::CpuAccessMode::Read);
            nvrhi::SamplerHandle samp = dev->createSampler(
                nvrhi::SamplerDesc().setAllFilters(false).setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
            nvrhi::BufferHandle sampleOut = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(16).setStructStride(16).setCanHaveUAVs(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("hazard_sample_out"));
            nvrhi::BufferHandle sampleCB = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(SamplerTestParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("hazard_sample_cb"));
            nvrhi::BufferHandle uavCB = dev->createBuffer(
                nvrhi::BufferDesc().setByteSize(sizeof(UavFillParams)).setIsConstantBuffer(true).enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("hazard_uav_cb"));
            UAA_CHECK(tex != nullptr && readStaging != nullptr && samp != nullptr && sampleOut != nullptr && sampleCB != nullptr && uavCB != nullptr,
                "Failed to create resources for TEX.ReadWriteHazards");

            const SamplerTestParams sParams{ { 0.5f, 0.5f }, { 0.f, 0.f }, 0.0f, 1u, 1u, 0u };
            fix.uploadBuffer(sampleCB, &sParams, sizeof(sParams), 0);

            nvrhi::ShaderHandle csSample = fix.createShader("cs_sampler_test", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle sampleLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
                    .addItem(nvrhi::BindingLayoutItem::Sampler(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(3)));
            UAA_CHECK(sampleLayout != nullptr, "Failed to create sample layout");
            nvrhi::BindingSetHandle sampleSet = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::Texture_SRV(0, tex))
                    .addItem(nvrhi::BindingSetItem::Sampler(1, samp))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, sampleOut))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(3, sampleCB)),
                sampleLayout);
            nvrhi::ComputePipelineHandle samplePso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csSample).addBindingLayout(sampleLayout));
            UAA_CHECK(sampleSet != nullptr && samplePso != nullptr, "Failed to create sample pipeline");

            nvrhi::ShaderHandle csFill = fix.createShader("cs_texture_uav_fill", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutHandle fillLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::Texture_UAV(0))
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(1)));
            UAA_CHECK(fillLayout != nullptr, "Failed to create UAV fill layout");
            nvrhi::BindingSetHandle fillSet = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::Texture_UAV(0, tex))
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, uavCB)),
                fillLayout);
            nvrhi::ComputePipelineHandle fillPso = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csFill).addBindingLayout(fillLayout));
            UAA_CHECK(fillSet != nullptr && fillPso != nullptr, "Failed to create UAV fill pipeline");

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);
            nvrhi::BindingLayoutHandle gfxLayout = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPC))));
            nvrhi::BindingSetHandle gfxSet = dev->createBindingSet(
                nvrhi::BindingSetDesc().addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPC))),
                gfxLayout);
            nvrhi::FramebufferHandle fb = dev->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(tex));
            UAA_CHECK(gfxLayout != nullptr && gfxSet != nullptr && fb != nullptr, "Failed to create graphics FB/bindings");

            nvrhi::GraphicsPipelineDesc opaquePsoDesc;
            opaquePsoDesc.setVertexShader(vs).setPixelShader(ps).addBindingLayout(gfxLayout);
            opaquePsoDesc.renderState.rasterState.setCullNone();
            opaquePsoDesc.renderState.depthStencilState.disableDepthTest().disableDepthWrite();
            nvrhi::GraphicsPipelineHandle opaquePso = dev->createGraphicsPipeline(opaquePsoDesc, fb->getFramebufferInfo());

            nvrhi::GraphicsPipelineDesc addPsoDesc = opaquePsoDesc;
            addPsoDesc.renderState.blendState.targets[0]
                .enableBlend()
                .setSrcBlend(nvrhi::BlendFactor::One)
                .setDestBlend(nvrhi::BlendFactor::One)
                .setBlendOp(nvrhi::BlendOp::Add)
                .setSrcBlendAlpha(nvrhi::BlendFactor::One)
                .setDestBlendAlpha(nvrhi::BlendFactor::One)
                .setBlendOpAlpha(nvrhi::BlendOp::Add);
            nvrhi::GraphicsPipelineHandle addPso = dev->createGraphicsPipeline(addPsoDesc, fb->getFramebufferInfo());
            UAA_CHECK(opaquePso != nullptr && addPso != nullptr, "Failed to create graphics pipelines");

            // 1. Upload -> sample inside one command list.
            {
                std::vector<float> uploadData(kSize * kSize * 4);
                for (size_t p = 0; p < kSize * kSize; ++p)
                {
                    uploadData[p * 4 + 0] = 0.2f;
                    uploadData[p * 4 + 1] = 0.6f;
                    uploadData[p * 4 + 2] = 0.8f;
                    uploadData[p * 4 + 3] = 1.0f;
                }

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->writeTexture(tex, 0, 0, uploadData.data(), kSize * 4 * sizeof(float));
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(samplePso).addBindingSet(sampleSet));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float sampled[4]{};
                fix.readbackBuffer(sampleOut, sampled, sizeof(sampled), 0);
                UAA_CHECK_NEAR(sampled[0], 0.2f, 0.01f, "upload->sample R mismatch");
                UAA_CHECK_NEAR(sampled[1], 0.6f, 0.01f, "upload->sample G mismatch");
                UAA_CHECK_NEAR(sampled[2], 0.8f, 0.01f, "upload->sample B mismatch");
                UAA_CHECK_NEAR(sampled[3], 1.0f, 0.01f, "upload->sample A mismatch");
            }

            // 2. UAV -> sample inside one command list.
            {
                const UavFillParams uParams{ { 0.8f, 0.2f, 0.4f, 1.0f }, kSize, kSize, 0u, 0u };
                fix.uploadBuffer(uavCB, &uParams, sizeof(uParams), 0);

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(fillPso).addBindingSet(fillSet));
                cmd->dispatch(1, 1, 1);
                cmd->setComputeState(nvrhi::ComputeState().setPipeline(samplePso).addBindingSet(sampleSet));
                cmd->dispatch(1, 1, 1);
                cmd->close();
                fix.submitAndWait(cmd);

                float sampled[4]{};
                fix.readbackBuffer(sampleOut, sampled, sizeof(sampled), 0);
                UAA_CHECK_NEAR(sampled[0], 0.8f, 0.01f, "UAV->sample R mismatch");
                UAA_CHECK_NEAR(sampled[1], 0.2f, 0.01f, "UAV->sample G mismatch");
                UAA_CHECK_NEAR(sampled[2], 0.4f, 0.01f, "UAV->sample B mismatch");
                UAA_CHECK_NEAR(sampled[3], 1.0f, 0.01f, "UAV->sample A mismatch");
            }

            // 3. Render -> copy to staging inside one command list.
            {
                const FullscreenPC pc{ { 0.1f, 0.6f, 0.9f, 1.0f }, 0.5f, 1.0f, { 0.f, 0.f } };

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(opaquePso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                    .addBindingSet(gfxSet));
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->copyTexture(readStaging, nvrhi::TextureSlice(), tex, nvrhi::TextureSlice());
                cmd->close();
                fix.submitAndWait(cmd);

                size_t rowPitch = 0;
                void* mapped = dev->mapStagingTexture(readStaging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch);
                UAA_CHECK(mapped != nullptr, "mapStagingTexture returned null after render->copy");
                const auto* values = static_cast<const float*>(mapped);
                for (uint32_t y = 0; y < kSize; ++y)
                {
                    for (uint32_t x = 0; x < kSize; ++x)
                    {
                        const float* pixel = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(values) + y * rowPitch) + x * 4;
                        if (std::abs(pixel[0] - 0.1f) > 0.01f || std::abs(pixel[1] - 0.6f) > 0.01f ||
                            std::abs(pixel[2] - 0.9f) > 0.01f || std::abs(pixel[3] - 1.0f) > 0.01f)
                        {
                            dev->unmapStagingTexture(readStaging);
                            UAA_CHECK(false, "render->copy float mismatch at (" + std::to_string(x) + "," + std::to_string(y) + ")");
                        }
                    }
                }
                dev->unmapStagingTexture(readStaging);
            }

            // 4. Copy (clearTextureFloat) -> render with additive blend inside one command list.
            {
                const FullscreenPC pcAdd{ { 0.3f, 0.2f, 0.5f, 0.25f }, 0.5f, 1.0f, { 0.f, 0.f } };

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned null");
                cmd->open();
                cmd->clearTextureFloat(tex, nvrhi::AllSubresources, nvrhi::Color(0.2f, 0.4f, 0.0f, 0.5f));
                cmd->setGraphicsState(nvrhi::GraphicsState()
                    .setPipeline(addPso)
                    .setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                    .addBindingSet(gfxSet));
                cmd->setPushConstants(&pcAdd, sizeof(pcAdd));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(tex, 0, 0, pixels);
                UAA_CHECK_EQ(pixels.size(), size_t(kSize * kSize * 4 * sizeof(float)), "Float texture readback size mismatch");
                const float* values = reinterpret_cast<const float*>(pixels.data());
                for (size_t p = 0; p < kSize * kSize; ++p)
                {
                    UAA_CHECK_NEAR(values[p * 4 + 0], 0.5f, 0.01f, "copy->render R mismatch");
                    UAA_CHECK_NEAR(values[p * 4 + 1], 0.6f, 0.01f, "copy->render G mismatch");
                    UAA_CHECK_NEAR(values[p * 4 + 2], 0.5f, 0.01f, "copy->render B mismatch");
                    UAA_CHECK_NEAR(values[p * 4 + 3], 0.75f, 0.01f, "copy->render A mismatch");
                }
            }
        }

        // Tests invalid TextureDesc rejections under the NVRHI validation layer.
        void testTexInvalid(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (!fix.isValidationActive())
            {
                ctx.skip("Requires NVRHI validation layer");
            }

            nvrhi::IDevice* dev = fix.device();

            auto expectInvalidTex = [&](const nvrhi::TextureDesc& desc, const std::string& label)
            {
                ExpectErrorsGuard guard(fix);
                nvrhi::TextureHandle h = dev->createTexture(desc);
                UAA_CHECK(h == nullptr, "Expected null texture handle for invalid case: " + label);
                UAA_CHECK(guard.errorCount() > 0, "Expected validation error for invalid case: " + label);
            };

            expectInvalidTex(
                nvrhi::TextureDesc().setWidth(0).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("inv_zero_w"),
                "zero width");
            expectInvalidTex(
                nvrhi::TextureDesc().setWidth(4).setHeight(0).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("inv_zero_h"),
                "zero height");
            expectInvalidTex(
                nvrhi::TextureDesc().setDimension(nvrhi::TextureDimension::Texture1D).setWidth(8).setHeight(2).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("inv_1d_h"),
                "1D height != 1");
            expectInvalidTex(
                nvrhi::TextureDesc().setDimension(nvrhi::TextureDimension::TextureCube).setWidth(8).setHeight(8).setArraySize(4).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("inv_cube_arr"),
                "Cube arraySize != 6");
            expectInvalidTex(
                nvrhi::TextureDesc().setDimension(nvrhi::TextureDimension::Texture2DMS).setWidth(8).setHeight(8).setSampleCount(4).setIsUAV(true).setFormat(nvrhi::Format::RGBA8_UNORM).setDebugName("inv_msaa_uav"),
                "MSAA with isUAV");
            expectInvalidTex(
                nvrhi::TextureDesc().setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM).setInitialState(nvrhi::ResourceStates::Unknown).setKeepInitialState(true).setDebugName("inv_keep_unknown"),
                "keepInitialState with Unknown");
        }
    } // namespace

    UAA_REGISTER_TEST("TEX.Create", "core", testTexCreate);
    UAA_REGISTER_TEST("TEX.Formats", "core", testTexFormats);
    UAA_REGISTER_TEST("TEX.UploadReadback", "core", testTexUploadReadback);
    UAA_REGISTER_TEST("TEX.Pitches", "core", testTexPitches);
    UAA_REGISTER_TEST("TEX.CopyRegions", "core", testTexCopyRegions);
    UAA_REGISTER_TEST("TEX.Staging", "core", testTexStaging);
    UAA_REGISTER_TEST("TEX.ClearFloatUInt", "core", testTexClearFloatUInt);
    UAA_REGISTER_TEST("TEX.DepthStencil", "core", testTexDepthStencil);
    UAA_REGISTER_TEST("TEX.Views", "core", testTexViews);
    UAA_REGISTER_TEST("TEX.Resolve", "core", testTexResolve);
    UAA_REGISTER_TEST("TEX.MSAAArrayResolve", "core", testTexMsaaArrayResolve);
    UAA_REGISTER_TEST("TEX.ReadWriteHazards", "core", testTexReadWriteHazards);
    UAA_REGISTER_TEST("TEX.Invalid", "core,validation", testTexInvalid);
} // namespace nvrhi::uaa
