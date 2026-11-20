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

#include "uaa_runner.h"

#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/containers.h>
#include <nvrhi/common/misc.h>
#include <nvrhi/common/resource.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct ExpectedFormatInfo
        {
            nvrhi::Format format;
            const char* name;
            uint8_t bytesPerBlock;
            uint8_t blockSize;
            nvrhi::FormatKind kind;
            bool hasRed;
            bool hasGreen;
            bool hasBlue;
            bool hasAlpha;
            bool hasDepth;
            bool hasStencil;
            bool isSigned;
            bool isSRGB;
        };

        const ExpectedFormatInfo kExpectedFormatTable[] = {
            { nvrhi::Format::UNKNOWN,           "UNKNOWN",           0,  0, nvrhi::FormatKind::Integer,      false, false, false, false, false, false, false, false },
            { nvrhi::Format::R8_UINT,           "R8_UINT",           1,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, false, false },
            { nvrhi::Format::R8_SINT,           "R8_SINT",           1,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::R8_UNORM,          "R8_UNORM",          1,  1, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, false, false },
            { nvrhi::Format::R8_SNORM,          "R8_SNORM",          1,  1, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::RG8_UINT,          "RG8_UINT",          2,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::RG8_SINT,          "RG8_SINT",          2,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::RG8_UNORM,         "RG8_UNORM",         2,  1, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::RG8_SNORM,         "RG8_SNORM",         2,  1, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::R16_UINT,          "R16_UINT",          2,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, false, false },
            { nvrhi::Format::R16_SINT,          "R16_SINT",          2,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::R16_UNORM,         "R16_UNORM",         2,  1, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, false, false },
            { nvrhi::Format::R16_SNORM,         "R16_SNORM",         2,  1, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::R16_FLOAT,         "R16_FLOAT",         2,  1, nvrhi::FormatKind::Float,        true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::BGRA4_UNORM,       "BGRA4_UNORM",       2,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::B5G6R5_UNORM,      "B5G6R5_UNORM",      2,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  false, false, false, false, false },
            { nvrhi::Format::B5G5R5A1_UNORM,    "B5G5R5A1_UNORM",    2,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA8_UINT,        "RGBA8_UINT",        4,  1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA8_SINT,        "RGBA8_SINT",        4,  1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::RGBA8_UNORM,       "RGBA8_UNORM",       4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA8_SNORM,       "RGBA8_SNORM",       4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::BGRA8_UNORM,       "BGRA8_UNORM",       4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::BGRX8_UNORM,       "BGRX8_UNORM",       4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  false, false, false, false, false },
            { nvrhi::Format::SRGBA8_UNORM,      "SRGBA8_UNORM",      4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
            { nvrhi::Format::SBGRA8_UNORM,      "SBGRA8_UNORM",      4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
            { nvrhi::Format::SBGRX8_UNORM,      "SBGRX8_UNORM",      4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  false, false, false, false, true  },
            { nvrhi::Format::R10G10B10A2_UNORM, "R10G10B10A2_UNORM", 4,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::R11G11B10_FLOAT,   "R11G11B10_FLOAT",   4,  1, nvrhi::FormatKind::Float,        true,  true,  true,  false, false, false, false, false },
            { nvrhi::Format::RG16_UINT,         "RG16_UINT",         4,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::RG16_SINT,         "RG16_SINT",         4,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::RG16_UNORM,        "RG16_UNORM",        4,  1, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::RG16_SNORM,        "RG16_SNORM",        4,  1, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::RG16_FLOAT,        "RG16_FLOAT",        4,  1, nvrhi::FormatKind::Float,        true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::R32_UINT,          "R32_UINT",          4,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, false, false },
            { nvrhi::Format::R32_SINT,          "R32_SINT",          4,  1, nvrhi::FormatKind::Integer,      true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::R32_FLOAT,         "R32_FLOAT",         4,  1, nvrhi::FormatKind::Float,        true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::RGBA16_UINT,       "RGBA16_UINT",       8,  1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA16_SINT,       "RGBA16_SINT",       8,  1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::RGBA16_FLOAT,      "RGBA16_FLOAT",      8,  1, nvrhi::FormatKind::Float,        true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::RGBA16_UNORM,      "RGBA16_UNORM",      8,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA16_SNORM,      "RGBA16_SNORM",      8,  1, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::RG32_UINT,         "RG32_UINT",         8,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::RG32_SINT,         "RG32_SINT",         8,  1, nvrhi::FormatKind::Integer,      true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::RG32_FLOAT,        "RG32_FLOAT",        8,  1, nvrhi::FormatKind::Float,        true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::RGB32_UINT,        "RGB32_UINT",        12, 1, nvrhi::FormatKind::Integer,      true,  true,  true,  false, false, false, false, false },
            { nvrhi::Format::RGB32_SINT,        "RGB32_SINT",        12, 1, nvrhi::FormatKind::Integer,      true,  true,  true,  false, false, false, true,  false },
            { nvrhi::Format::RGB32_FLOAT,       "RGB32_FLOAT",       12, 1, nvrhi::FormatKind::Float,        true,  true,  true,  false, false, false, true,  false },
            { nvrhi::Format::RGBA32_UINT,       "RGBA32_UINT",       16, 1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::RGBA32_SINT,       "RGBA32_SINT",       16, 1, nvrhi::FormatKind::Integer,      true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::RGBA32_FLOAT,      "RGBA32_FLOAT",      16, 1, nvrhi::FormatKind::Float,        true,  true,  true,  true,  false, false, true,  false },
            { nvrhi::Format::D16,               "D16",               2,  1, nvrhi::FormatKind::DepthStencil, false, false, false, false, true,  false, false, false },
            { nvrhi::Format::D24S8,             "D24S8",             4,  1, nvrhi::FormatKind::DepthStencil, false, false, false, false, true,  true,  false, false },
            { nvrhi::Format::X24G8_UINT,        "X24G8_UINT",        4,  1, nvrhi::FormatKind::Integer,      false, false, false, false, false, true,  false, false },
            { nvrhi::Format::D32,               "D32",               4,  1, nvrhi::FormatKind::DepthStencil, false, false, false, false, true,  false, false, false },
            { nvrhi::Format::D32S8,             "D32S8",             8,  1, nvrhi::FormatKind::DepthStencil, false, false, false, false, true,  true,  false, false },
            { nvrhi::Format::X32G8_UINT,        "X32G8_UINT",        8,  1, nvrhi::FormatKind::Integer,      false, false, false, false, false, true,  false, false },
            { nvrhi::Format::BC1_UNORM,         "BC1_UNORM",         8,  4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::BC1_UNORM_SRGB,    "BC1_UNORM_SRGB",    8,  4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
            { nvrhi::Format::BC2_UNORM,         "BC2_UNORM",         16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::BC2_UNORM_SRGB,    "BC2_UNORM_SRGB",    16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
            { nvrhi::Format::BC3_UNORM,         "BC3_UNORM",         16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::BC3_UNORM_SRGB,    "BC3_UNORM_SRGB",    16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
            { nvrhi::Format::BC4_UNORM,         "BC4_UNORM",         8,  4, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, false, false },
            { nvrhi::Format::BC4_SNORM,         "BC4_SNORM",         8,  4, nvrhi::FormatKind::Normalized,   true,  false, false, false, false, false, true,  false },
            { nvrhi::Format::BC5_UNORM,         "BC5_UNORM",         16, 4, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, false, false },
            { nvrhi::Format::BC5_SNORM,         "BC5_SNORM",         16, 4, nvrhi::FormatKind::Normalized,   true,  true,  false, false, false, false, true,  false },
            { nvrhi::Format::BC6H_UFLOAT,       "BC6H_UFLOAT",       16, 4, nvrhi::FormatKind::Float,        true,  true,  true,  false, false, false, false, false },
            { nvrhi::Format::BC6H_SFLOAT,       "BC6H_SFLOAT",       16, 4, nvrhi::FormatKind::Float,        true,  true,  true,  false, false, false, true,  false },
            { nvrhi::Format::BC7_UNORM,         "BC7_UNORM",         16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, false },
            { nvrhi::Format::BC7_UNORM_SRGB,    "BC7_UNORM_SRGB",    16, 4, nvrhi::FormatKind::Normalized,   true,  true,  true,  true,  false, false, false, true  },
        };

        static_assert(sizeof(kExpectedFormatTable) / sizeof(kExpectedFormatTable[0]) == size_t(nvrhi::Format::COUNT),
            "Expected format table must match Format::COUNT");

        class DummyTrackableResource : public nvrhi::RefCounter<nvrhi::IResource>
        {
        public:
            explicit DummyTrackableResource(std::atomic<int>& liveCount)
                : m_LiveCount(liveCount)
            {
                ++m_LiveCount;
            }

            ~DummyTrackableResource() override
            {
                --m_LiveCount;
            }

        private:
            std::atomic<int>& m_LiveCount;
        };

        class DerivedTrackableResource : public DummyTrackableResource
        {
        public:
            using DummyTrackableResource::DummyTrackableResource;
        };

        void testFormatMetadata(TestContext& ctx)
        {
            for (size_t i = 0; i < size_t(nvrhi::Format::COUNT); ++i)
            {
                const nvrhi::Format fmt = static_cast<nvrhi::Format>(i);
                const nvrhi::FormatInfo& info = nvrhi::getFormatInfo(fmt);
                const ExpectedFormatInfo& expected = kExpectedFormatTable[i];

                UAA_CHECK_EQ(info.format, fmt, "FormatInfo::format mismatch");
                UAA_CHECK_EQ(info.format, expected.format, "Expected table format index mismatch");
                UAA_CHECK(info.name != nullptr && info.name[0] != '\0', "FormatInfo::name must be non-empty");
                UAA_CHECK(std::strcmp(info.name, expected.name) == 0, std::string("Format name mismatch for ") + expected.name);
                UAA_CHECK_EQ(info.bytesPerBlock, expected.bytesPerBlock, std::string("bytesPerBlock mismatch for ") + expected.name);
                UAA_CHECK_EQ(info.blockSize, expected.blockSize, std::string("blockSize mismatch for ") + expected.name);
                UAA_CHECK_EQ(info.kind, expected.kind, std::string("kind mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasRed), expected.hasRed, std::string("hasRed mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasGreen), expected.hasGreen, std::string("hasGreen mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasBlue), expected.hasBlue, std::string("hasBlue mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasAlpha), expected.hasAlpha, std::string("hasAlpha mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasDepth), expected.hasDepth, std::string("hasDepth mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.hasStencil), expected.hasStencil, std::string("hasStencil mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.isSigned), expected.isSigned, std::string("isSigned mismatch for ") + expected.name);
                UAA_CHECK_EQ(bool(info.isSRGB), expected.isSRGB, std::string("isSRGB mismatch for ") + expected.name);
            }
        }

        void testFormatBoundaries(TestContext& ctx)
        {
            const nvrhi::FormatInfo& unk = nvrhi::getFormatInfo(nvrhi::Format::UNKNOWN);
            UAA_CHECK_EQ(unk.format, nvrhi::Format::UNKNOWN, "UNKNOWN format mismatch");
            UAA_CHECK_EQ(unk.bytesPerBlock, uint8_t(0), "UNKNOWN bytesPerBlock should be 0");
            UAA_CHECK_EQ(unk.blockSize, uint8_t(0), "UNKNOWN blockSize should be 0");
            UAA_CHECK(!unk.hasRed && !unk.hasGreen && !unk.hasBlue && !unk.hasAlpha && !unk.hasDepth && !unk.hasStencil,
                "UNKNOWN must not advertise any channels");

            const nvrhi::Format depthStencilFormats[] = {
                nvrhi::Format::D16,
                nvrhi::Format::D24S8,
                nvrhi::Format::X24G8_UINT,
                nvrhi::Format::D32,
                nvrhi::Format::D32S8,
                nvrhi::Format::X32G8_UINT,
            };
            for (nvrhi::Format fmt : depthStencilFormats)
            {
                const nvrhi::FormatInfo& info = nvrhi::getFormatInfo(fmt);
                UAA_CHECK(info.hasDepth || info.hasStencil, "Depth/stencil format must report depth or stencil");
                UAA_CHECK(!info.hasRed && !info.hasGreen && !info.hasBlue && !info.hasAlpha,
                    "Depth/stencil format must not report colour channels");
                UAA_CHECK_EQ(info.blockSize, uint8_t(1), "Depth/stencil blockSize must be 1");
            }

            const nvrhi::Format packedFormats[] = {
                nvrhi::Format::BGRA4_UNORM,
                nvrhi::Format::B5G6R5_UNORM,
                nvrhi::Format::B5G5R5A1_UNORM,
                nvrhi::Format::R10G10B10A2_UNORM,
                nvrhi::Format::R11G11B10_FLOAT,
            };
            for (nvrhi::Format fmt : packedFormats)
            {
                const nvrhi::FormatInfo& info = nvrhi::getFormatInfo(fmt);
                UAA_CHECK_EQ(info.blockSize, uint8_t(1), "Packed format blockSize must be 1");
                UAA_CHECK(info.hasRed && info.hasGreen && info.hasBlue, "Packed format must have RGB channels");
                UAA_CHECK(!info.isSigned && !info.isSRGB, "Packed format must be unsigned non-sRGB");
            }

            for (uint32_t i = uint32_t(nvrhi::Format::BC1_UNORM); i <= uint32_t(nvrhi::Format::BC7_UNORM_SRGB); ++i)
            {
                const nvrhi::FormatInfo& info = nvrhi::getFormatInfo(static_cast<nvrhi::Format>(i));
                UAA_CHECK_EQ(info.blockSize, uint8_t(4), "BC compressed format must have blockSize 4");
                UAA_CHECK(info.bytesPerBlock == 8 || info.bytesPerBlock == 16, "BC compressed format block size must be 8 or 16");
            }

            const std::unordered_set<nvrhi::Format> expectedSrgb = {
                nvrhi::Format::SRGBA8_UNORM,
                nvrhi::Format::SBGRA8_UNORM,
                nvrhi::Format::SBGRX8_UNORM,
                nvrhi::Format::BC1_UNORM_SRGB,
                nvrhi::Format::BC2_UNORM_SRGB,
                nvrhi::Format::BC3_UNORM_SRGB,
                nvrhi::Format::BC7_UNORM_SRGB,
            };
            for (size_t i = 0; i < size_t(nvrhi::Format::COUNT); ++i)
            {
                const nvrhi::Format fmt = static_cast<nvrhi::Format>(i);
                const bool shouldBeSrgb = expectedSrgb.count(fmt) > 0;
                UAA_CHECK_EQ(bool(nvrhi::getFormatInfo(fmt).isSRGB), shouldBeSrgb, "sRGB flag mismatch");
            }

            const nvrhi::Format oob1 = static_cast<nvrhi::Format>(uint8_t(nvrhi::Format::COUNT));
            const nvrhi::Format oob2 = static_cast<nvrhi::Format>(uint8_t(nvrhi::Format::COUNT) + 5);
            UAA_CHECK_EQ(nvrhi::getFormatInfo(oob1).format, nvrhi::Format::UNKNOWN, "Format::COUNT must map safely to UNKNOWN");
            UAA_CHECK_EQ(nvrhi::getFormatInfo(oob2).format, nvrhi::Format::UNKNOWN, "Out-of-range format must map safely to UNKNOWN");
        }

        void testSubresources(TestContext& ctx)
        {
            const nvrhi::TextureSubresourceSet def;
            UAA_CHECK_EQ(def.baseMipLevel, 0u, "Default baseMipLevel");
            UAA_CHECK_EQ(def.numMipLevels, 1u, "Default numMipLevels");
            UAA_CHECK_EQ(def.baseArraySlice, 0u, "Default baseArraySlice");
            UAA_CHECK_EQ(def.numArraySlices, 1u, "Default numArraySlices");

            const nvrhi::TextureSubresourceSet fluent = nvrhi::TextureSubresourceSet()
                .setMipLevels(2, 3)
                .setArraySlices(1, 4);
            UAA_CHECK(fluent == nvrhi::TextureSubresourceSet(2, 3, 1, 4), "SubresourceSet equality");
            UAA_CHECK(fluent != def, "SubresourceSet inequality");

            const nvrhi::TextureDesc desc1D = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture1D)
                .setWidth(64)
                .setMipLevels(4);
            const nvrhi::TextureDesc desc2D = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth(64)
                .setHeight(32)
                .setMipLevels(5);
            const nvrhi::TextureDesc descCube = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::TextureCube)
                .setWidth(32)
                .setHeight(32)
                .setArraySize(6)
                .setMipLevels(3);
            const nvrhi::TextureDesc descCubeArr = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::TextureCubeArray)
                .setWidth(32)
                .setHeight(32)
                .setArraySize(12)
                .setMipLevels(4);
            const nvrhi::TextureDesc desc3D = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture3D)
                .setWidth(16)
                .setHeight(16)
                .setDepth(8)
                .setArraySize(1)
                .setMipLevels(4);

            UAA_CHECK(nvrhi::AllSubresources.isEntireTexture(desc1D), "AllSubresources covers 1D");
            UAA_CHECK(nvrhi::AllSubresources.isEntireTexture(desc2D), "AllSubresources covers 2D");
            UAA_CHECK(nvrhi::AllSubresources.isEntireTexture(descCube), "AllSubresources covers Cube");
            UAA_CHECK(nvrhi::AllSubresources.isEntireTexture(descCubeArr), "AllSubresources covers CubeArray");
            UAA_CHECK(nvrhi::AllSubresources.isEntireTexture(desc3D), "AllSubresources covers 3D");

            const auto r2d = nvrhi::AllSubresources.resolve(desc2D, false);
            UAA_CHECK(r2d == nvrhi::TextureSubresourceSet(0, 5, 0, 1), "Resolved 2D AllSubresources");
            const auto r2dSingle = nvrhi::AllSubresources.resolve(desc2D, true);
            UAA_CHECK(r2dSingle == nvrhi::TextureSubresourceSet(0, 1, 0, 1), "Resolved 2D single mip");

            const auto rCube = nvrhi::AllSubresources.resolve(descCube, false);
            UAA_CHECK(rCube == nvrhi::TextureSubresourceSet(0, 3, 0, 6), "Resolved Cube AllSubresources");
            const auto rCubeArr = nvrhi::AllSubresources.resolve(descCubeArr, false);
            UAA_CHECK(rCubeArr == nvrhi::TextureSubresourceSet(0, 4, 0, 12), "Resolved CubeArray AllSubresources");
            const auto r3d = nvrhi::AllSubresources.resolve(desc3D, false);
            UAA_CHECK(r3d == nvrhi::TextureSubresourceSet(0, 4, 0, 1), "Resolved 3D AllSubresources");

            UAA_CHECK(rCube.isEntireTexture(descCube), "Exact Cube range is entire texture");
            UAA_CHECK(!nvrhi::TextureSubresourceSet(0, 3, 0, 5).isEntireTexture(descCube), "Missing cube slice is not entire");
            UAA_CHECK(!nvrhi::TextureSubresourceSet(1, 2, 0, 6).isEntireTexture(descCube), "Nonzero base mip is not entire");
            UAA_CHECK(!nvrhi::TextureSubresourceSet(0, 2, 0, 1).isEntireTexture(desc2D), "Missing mip is not entire");

            const auto singleMipSlice = nvrhi::TextureSubresourceSet(2, 1, 3, 1).resolve(descCubeArr, false);
            UAA_CHECK(singleMipSlice == nvrhi::TextureSubresourceSet(2, 1, 3, 1), "Single mip/slice preserved");

            const auto clamped = nvrhi::TextureSubresourceSet(2, 10, 8, 10).resolve(descCubeArr, false);
            UAA_CHECK(clamped == nvrhi::TextureSubresourceSet(2, 2, 8, 4), "Partially out-of-bounds range clamped");

            const auto pastEnd = nvrhi::TextureSubresourceSet(4, 3, 12, 2).resolve(descCubeArr, false);
            UAA_CHECK_EQ(pastEnd.numMipLevels, 0u, "baseMipLevel >= mipLevels resolves to 0 mips");
            UAA_CHECK_EQ(pastEnd.numArraySlices, 0u, "baseArraySlice >= arraySize resolves to 0 slices");

            std::hash<nvrhi::TextureSubresourceSet> hasher;
            UAA_CHECK_EQ(hasher(fluent), hasher(nvrhi::TextureSubresourceSet(2, 3, 1, 4)), "Equal sets share hash");
            UAA_CHECK_NE(hasher(fluent), hasher(def), "Distinct sets should produce distinct hashes");
        }

        void testSlices(TestContext& ctx)
        {
            const nvrhi::TextureSlice def;
            UAA_CHECK_EQ(def.x, 0u, "Default slice x");
            UAA_CHECK_EQ(def.y, 0u, "Default slice y");
            UAA_CHECK_EQ(def.z, 0u, "Default slice z");
            UAA_CHECK_EQ(def.width, uint32_t(-1), "Default slice width sentinel");
            UAA_CHECK_EQ(def.height, uint32_t(-1), "Default slice height sentinel");
            UAA_CHECK_EQ(def.depth, uint32_t(-1), "Default slice depth sentinel");
            UAA_CHECK_EQ(def.mipLevel, 0u, "Default slice mipLevel");
            UAA_CHECK_EQ(def.arraySlice, 0u, "Default slice arraySlice");

            const nvrhi::TextureDesc desc2D = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2DArray)
                .setWidth(64)
                .setHeight(32)
                .setArraySize(4)
                .setMipLevels(7);

            const auto s0 = nvrhi::TextureSlice().setMipLevel(0).setArraySlice(2).resolve(desc2D);
            UAA_CHECK_EQ(s0.width, 64u, "2D mip 0 width");
            UAA_CHECK_EQ(s0.height, 32u, "2D mip 0 height");
            UAA_CHECK_EQ(s0.depth, 1u, "Non-3D depth resolves to 1");
            UAA_CHECK_EQ(s0.arraySlice, 2u, "Array slice preserved");

            const auto s2 = nvrhi::TextureSlice().setMipLevel(2).resolve(desc2D);
            UAA_CHECK_EQ(s2.width, 16u, "2D mip 2 width");
            UAA_CHECK_EQ(s2.height, 8u, "2D mip 2 height");
            UAA_CHECK_EQ(s2.depth, 1u, "2D mip 2 depth");

            const auto s6 = nvrhi::TextureSlice().setMipLevel(6).resolve(desc2D);
            UAA_CHECK_EQ(s6.width, 1u, "2D mip 6 width clamped to 1");
            UAA_CHECK_EQ(s6.height, 1u, "2D mip 6 height clamped to 1");

            const nvrhi::TextureDesc desc3D = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture3D)
                .setWidth(16)
                .setHeight(8)
                .setDepth(4)
                .setMipLevels(5);

            const auto d0 = nvrhi::TextureSlice().setMipLevel(0).resolve(desc3D);
            UAA_CHECK_EQ(d0.width, 16u, "3D mip 0 width");
            UAA_CHECK_EQ(d0.height, 8u, "3D mip 0 height");
            UAA_CHECK_EQ(d0.depth, 4u, "3D mip 0 depth");

            const auto d2 = nvrhi::TextureSlice().setMipLevel(2).resolve(desc3D);
            UAA_CHECK_EQ(d2.width, 4u, "3D mip 2 width");
            UAA_CHECK_EQ(d2.height, 2u, "3D mip 2 height");
            UAA_CHECK_EQ(d2.depth, 1u, "3D mip 2 depth");

            const auto d4 = nvrhi::TextureSlice().setMipLevel(4).resolve(desc3D);
            UAA_CHECK_EQ(d4.width, 1u, "3D mip 4 width");
            UAA_CHECK_EQ(d4.height, 1u, "3D mip 4 height");
            UAA_CHECK_EQ(d4.depth, 1u, "3D mip 4 depth clamped to 1");

            const auto custom = nvrhi::TextureSlice()
                .setOrigin(3, 4, 1)
                .setSize(7, 5, 2)
                .setMipLevel(1)
                .resolve(desc3D);
            UAA_CHECK_EQ(custom.x, 3u, "Explicit origin x");
            UAA_CHECK_EQ(custom.y, 4u, "Explicit origin y");
            UAA_CHECK_EQ(custom.z, 1u, "Explicit origin z");
            UAA_CHECK_EQ(custom.width, 7u, "Explicit width override");
            UAA_CHECK_EQ(custom.height, 5u, "Explicit height override");
            UAA_CHECK_EQ(custom.depth, 2u, "Explicit depth override");
        }

        void testBufferRanges(TestContext& ctx)
        {
            const nvrhi::BufferDesc desc = nvrhi::BufferDesc().setByteSize(256);

            const nvrhi::BufferRange entire = nvrhi::EntireBuffer.resolve(desc);
            UAA_CHECK(entire == nvrhi::BufferRange(0, 256), "EntireBuffer resolve");
            UAA_CHECK(nvrhi::EntireBuffer.isEntireBuffer(desc), "EntireBuffer.isEntireBuffer");
            UAA_CHECK(entire.isEntireBuffer(desc), "Exact range.isEntireBuffer");

            const nvrhi::BufferRange zeroRemainder(0, 0);
            UAA_CHECK(!zeroRemainder.isEntireBuffer(desc), "(0,0) is not isEntireBuffer for nonzero buffer");
            UAA_CHECK(zeroRemainder.resolve(desc) == nvrhi::BufferRange(0, 256), "(0,0) resolves to remainder");

            const nvrhi::BufferRange offsetRemainder = nvrhi::BufferRange().setByteOffset(64).setByteSize(0);
            UAA_CHECK(offsetRemainder.resolve(desc) == nvrhi::BufferRange(64, 192), "Nonzero offset remainder");
            UAA_CHECK(!offsetRemainder.isEntireBuffer(desc), "Nonzero offset is not entire buffer");

            const nvrhi::BufferRange exactEnd(64, 192);
            UAA_CHECK(exactEnd.resolve(desc) == nvrhi::BufferRange(64, 192), "Exact end preserved");

            const nvrhi::BufferRange clamped(200, 128);
            UAA_CHECK(clamped.resolve(desc) == nvrhi::BufferRange(200, 56), "Range clamped beyond end");

            const nvrhi::BufferRange pastEnd(512, 64);
            UAA_CHECK(pastEnd.resolve(desc) == nvrhi::BufferRange(256, 0), "Offset beyond end clamps to zero size");

            std::hash<nvrhi::BufferRange> hasher;
            UAA_CHECK_EQ(hasher(offsetRemainder.resolve(desc)), hasher(exactEnd), "Equal BufferRange hashes match");
            UAA_CHECK_NE(hasher(entire), hasher(exactEnd), "Distinct BufferRange hashes differ");
        }

        void testDescriptorDefaults(TestContext& ctx)
        {
            {
                const nvrhi::HeapDesc def{};
                UAA_CHECK_EQ(def.capacity, uint64_t(0), "HeapDesc default capacity");
                const auto hd = nvrhi::HeapDesc().setCapacity(4096).setType(nvrhi::HeapType::Upload).setDebugName("heap");
                UAA_CHECK_EQ(hd.capacity, uint64_t(4096), "HeapDesc capacity setter");
                UAA_CHECK_EQ(hd.type, nvrhi::HeapType::Upload, "HeapDesc type setter");
                UAA_CHECK_EQ(hd.debugName, std::string("heap"), "HeapDesc debugName setter");
            }

            {
                const nvrhi::TextureDesc def;
                UAA_CHECK_EQ(def.width, 1u, "TextureDesc default width");
                UAA_CHECK_EQ(def.height, 1u, "TextureDesc default height");
                UAA_CHECK_EQ(def.depth, 1u, "TextureDesc default depth");
                UAA_CHECK_EQ(def.arraySize, 1u, "TextureDesc default arraySize");
                UAA_CHECK_EQ(def.mipLevels, 1u, "TextureDesc default mipLevels");
                UAA_CHECK_EQ(def.sampleCount, 1u, "TextureDesc default sampleCount");
                UAA_CHECK_EQ(def.format, nvrhi::Format::UNKNOWN, "TextureDesc default format");
                UAA_CHECK_EQ(def.dimension, nvrhi::TextureDimension::Texture2D, "TextureDesc default dimension");
                UAA_CHECK(def.isShaderResource, "TextureDesc default isShaderResource");
                UAA_CHECK(!def.isRenderTarget && !def.isUAV && !def.isTypeless && !def.isVirtual, "TextureDesc default flags");
                UAA_CHECK(def.defaultComponentMapping.isIdentity(), "TextureDesc default component mapping");

                const auto td = nvrhi::TextureDesc()
                    .setWidth(128)
                    .setHeight(64)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true)
                    .setIsUAV(true)
                    .setClearValue(nvrhi::Color(1.f, 0.f, 0.f, 1.f))
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
                    .setDebugName("tex");
                UAA_CHECK_EQ(td.width, 128u, "TextureDesc width");
                UAA_CHECK_EQ(td.height, 64u, "TextureDesc height");
                UAA_CHECK(td.isRenderTarget && td.isUAV && td.useClearValue && td.keepInitialState, "TextureDesc flags");
                UAA_CHECK_EQ(td.initialState, nvrhi::ResourceStates::ShaderResource, "TextureDesc initialState");
            }

            {
                const nvrhi::BufferDesc def;
                UAA_CHECK_EQ(def.byteSize, uint64_t(0), "BufferDesc default byteSize");
                UAA_CHECK_EQ(def.structStride, 0u, "BufferDesc default structStride");
                UAA_CHECK_EQ(def.cpuAccess, nvrhi::CpuAccessMode::None, "BufferDesc default cpuAccess");
                UAA_CHECK_EQ(def.initialState, nvrhi::ResourceStates::Common, "BufferDesc default initialState");
                const auto bd = nvrhi::BufferDesc()
                    .setByteSize(1024)
                    .setStructStride(16)
                    .setCanHaveUAVs(true)
                    .setCanHaveTypedViews(true)
                    .setCanHaveRawViews(true)
                    .setIsVertexBuffer(true)
                    .setIsIndexBuffer(true)
                    .setIsConstantBuffer(true)
                    .setCpuAccess(nvrhi::CpuAccessMode::Write)
                    .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
                    .setDebugName("buf");
                UAA_CHECK_EQ(bd.byteSize, uint64_t(1024), "BufferDesc byteSize");
                UAA_CHECK_EQ(bd.structStride, 16u, "BufferDesc structStride");
                UAA_CHECK(bd.canHaveUAVs && bd.canHaveTypedViews && bd.canHaveRawViews, "BufferDesc view flags");
                UAA_CHECK(bd.isVertexBuffer && bd.isIndexBuffer && bd.isConstantBuffer && bd.keepInitialState, "BufferDesc usage flags");
            }

            {
                const nvrhi::SamplerDesc def;
                UAA_CHECK(def.minFilter && def.magFilter && def.mipFilter, "SamplerDesc default filters");
                UAA_CHECK_EQ(def.addressU, nvrhi::SamplerAddressMode::Clamp, "SamplerDesc default addressU");
                UAA_CHECK_EQ(def.reductionType, nvrhi::SamplerReductionType::Standard, "SamplerDesc default reductionType");
                UAA_CHECK_EQ(def.compareFunc, nvrhi::ComparisonFunc::Always, "SamplerDesc default compareFunc");
                const auto sd = nvrhi::SamplerDesc()
                    .setAllFilters(false)
                    .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
                    .setReductionType(nvrhi::SamplerReductionType::Comparison)
                    .setCompareFunc(nvrhi::ComparisonFunc::LessOrEqual)
                    .setMaxAnisotropy(8.f)
                    .setMipBias(-0.5f);
                UAA_CHECK(!sd.minFilter && !sd.magFilter && !sd.mipFilter, "SamplerDesc setAllFilters(false)");
                UAA_CHECK_EQ(sd.addressV, nvrhi::SamplerAddressMode::Wrap, "SamplerDesc addressV");
                UAA_CHECK_EQ(sd.compareFunc, nvrhi::ComparisonFunc::LessOrEqual, "SamplerDesc compareFunc setter");
            }

            {
                const nvrhi::FramebufferDesc fbDef;
                UAA_CHECK(fbDef.colorAttachments.empty(), "FramebufferDesc default colorAttachments empty");
                UAA_CHECK(!fbDef.depthAttachment.valid(), "FramebufferDesc default depth invalid");

                nvrhi::FramebufferInfo infoA;
                UAA_CHECK(infoA.colorFormats.empty(), "FramebufferInfo default colorFormats empty");
                UAA_CHECK_EQ(infoA.depthFormat, nvrhi::Format::UNKNOWN, "FramebufferInfo default depthFormat");
                UAA_CHECK_EQ(infoA.sampleCount, 1u, "FramebufferInfo default sampleCount");
                infoA.addColorFormat(nvrhi::Format::RGBA8_UNORM).setDepthFormat(nvrhi::Format::D32).setSampleCount(4).setSampleQuality(1);
                nvrhi::FramebufferInfo infoB = infoA;
                UAA_CHECK(infoA == infoB, "FramebufferInfo equality");
                infoB.setSampleQuality(0);
                UAA_CHECK(infoA != infoB, "FramebufferInfo inequality");

                const auto ex = nvrhi::FramebufferInfoEx().setWidth(320).setHeight(240).setArraySize(2);
                const nvrhi::Viewport vp = ex.getViewport(0.1f, 0.9f);
                UAA_CHECK(vp == nvrhi::Viewport(0.f, 320.f, 0.f, 240.f, 0.1f, 0.9f), "FramebufferInfoEx::getViewport");
            }

            {
                const nvrhi::ShaderDesc sdef;
                UAA_CHECK_EQ(sdef.shaderType, nvrhi::ShaderType::None, "ShaderDesc default type");
                UAA_CHECK_EQ(sdef.entryName, std::string("main"), "ShaderDesc default entryName");
                UAA_CHECK_EQ(sdef.hlslExtensionsUAV, -1, "ShaderDesc default hlslExtensionsUAV");
                const auto sd = nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute).setEntryName("csMain").setDebugName("cs");
                UAA_CHECK_EQ(sd.shaderType, nvrhi::ShaderType::Compute, "ShaderDesc type setter");
                UAA_CHECK_EQ(sd.entryName, std::string("csMain"), "ShaderDesc entryName setter");

                const auto su = nvrhi::ShaderSpecialization::UInt32(1, 42u);
                const auto si = nvrhi::ShaderSpecialization::Int32(2, -7);
                const auto sf = nvrhi::ShaderSpecialization::Float(3, 2.5f);
                UAA_CHECK_EQ(su.constantID, 1u, "Spec constantID UInt32");
                UAA_CHECK_EQ(su.value.u, 42u, "Spec value UInt32");
                UAA_CHECK_EQ(si.value.i, -7, "Spec value Int32");
                UAA_CHECK_NEAR(sf.value.f, 2.5f, 1e-6, "Spec value Float");
            }

            {
                const nvrhi::GraphicsPipelineDesc gp;
                UAA_CHECK_EQ(gp.primType, nvrhi::PrimitiveType::TriangleList, "GraphicsPipelineDesc default primType");
                UAA_CHECK_EQ(gp.patchControlPoints, 0u, "GraphicsPipelineDesc default patchControlPoints");
                const nvrhi::ComputePipelineDesc cp;
                UAA_CHECK(cp.CS == nullptr && cp.bindingLayouts.empty(), "ComputePipelineDesc defaults");
                const nvrhi::MeshletPipelineDesc mp;
                UAA_CHECK_EQ(mp.primType, nvrhi::PrimitiveType::TriangleList, "MeshletPipelineDesc default primType");
            }

            {
                const nvrhi::BindingLayoutDesc bld;
                UAA_CHECK_EQ(bld.visibility, nvrhi::ShaderType::None, "BindingLayoutDesc default visibility");
                UAA_CHECK_EQ(bld.registerSpace, 0u, "BindingLayoutDesc default registerSpace");
                UAA_CHECK(!bld.registerSpaceIsDescriptorSet, "BindingLayoutDesc default registerSpaceIsDescriptorSet");
                UAA_CHECK_EQ(bld.bindingOffsets.shaderResource, 0u, "VulkanBindingOffsets default SRV");
                UAA_CHECK_EQ(bld.bindingOffsets.sampler, 128u, "VulkanBindingOffsets default sampler");
                UAA_CHECK_EQ(bld.bindingOffsets.constantBuffer, 256u, "VulkanBindingOffsets default CB");
                UAA_CHECK_EQ(bld.bindingOffsets.unorderedAccess, 384u, "VulkanBindingOffsets default UAV");

                const auto bldSet = nvrhi::BindingLayoutDesc()
                    .setVisibility(nvrhi::ShaderType::All)
                    .setRegisterSpaceAndDescriptorSet(2)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, 64));
                UAA_CHECK_EQ(bldSet.registerSpace, 2u, "setRegisterSpaceAndDescriptorSet space");
                UAA_CHECK(bldSet.registerSpaceIsDescriptorSet, "setRegisterSpaceAndDescriptorSet flag");
                UAA_CHECK_EQ(bldSet.bindings[0].getArraySize(), 1u, "PushConstants getArraySize() == 1");
            }

            {
                const nvrhi::BindlessLayoutDesc bl;
                UAA_CHECK_EQ(bl.visibility, nvrhi::ShaderType::None, "BindlessLayoutDesc default visibility");
                UAA_CHECK_EQ(bl.firstSlot, 0u, "BindlessLayoutDesc default firstSlot");
                UAA_CHECK_EQ(bl.maxCapacity, 0u, "BindlessLayoutDesc default maxCapacity");
                UAA_CHECK_EQ(bl.layoutType, nvrhi::BindlessLayoutDesc::LayoutType::Immutable, "BindlessLayoutDesc default layoutType");
                UAA_CHECK(bl.updateAfterBind, "BindlessLayoutDesc default updateAfterBind must be true");

                const auto blSet = nvrhi::BindlessLayoutDesc()
                    .setVisibility(nvrhi::ShaderType::Compute)
                    .setFirstSlot(1)
                    .setMaxCapacity(1024)
                    .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv)
                    .setUpdateAfterBind(false);
                UAA_CHECK(!blSet.updateAfterBind, "BindlessLayoutDesc setUpdateAfterBind(false)");
                UAA_CHECK_EQ(blSet.maxCapacity, 1024u, "BindlessLayoutDesc maxCapacity");
            }

            {
                const nvrhi::ComponentMapping texDefault = nvrhi::ComponentMapping()
                    .setR(nvrhi::ComponentSwizzle::G)
                    .setG(nvrhi::ComponentSwizzle::R)
                    .setB(nvrhi::ComponentSwizzle::Zero)
                    .setA(nvrhi::ComponentSwizzle::One);
                UAA_CHECK(!texDefault.isIdentity(), "Custom ComponentMapping is not identity");
                UAA_CHECK_EQ(nvrhi::packComponentMapping(std::nullopt), uint16_t(0), "nullopt packs to 0");
                UAA_CHECK(nvrhi::resolveComponentMapping(uint16_t(0), texDefault) == texDefault, "0 inherits texture default");
                const uint16_t forceId = nvrhi::packComponentMapping(nvrhi::ComponentMapping{});
                UAA_CHECK(forceId & nvrhi::c_ComponentMappingExplicit, "Explicit mapping sets bit 15");
                UAA_CHECK(nvrhi::resolveComponentMapping(forceId, texDefault).isIdentity(), "Explicit identity overrides default");

                const auto noneItem = nvrhi::BindingSetItem::None(3);
                const auto srvItem = nvrhi::BindingSetItem::Texture_SRV(1, nullptr, nvrhi::Format::RGBA8_UNORM, nvrhi::AllSubresources, nvrhi::TextureDimension::Texture2D, texDefault);
                const auto uavItem = nvrhi::BindingSetItem::Texture_UAV(2, nullptr);
                const auto tbufSrv = nvrhi::BindingSetItem::TypedBuffer_SRV(3, nullptr, nvrhi::Format::R32_UINT);
                const auto tbufUav = nvrhi::BindingSetItem::TypedBuffer_UAV(4, nullptr, nvrhi::Format::R32_UINT);
                const auto cbItem = nvrhi::BindingSetItem::ConstantBuffer(5, nullptr);
                const auto sampItem = nvrhi::BindingSetItem::Sampler(6, nullptr);
                const auto rtItem = nvrhi::BindingSetItem::RayTracingAccelStruct(7, nullptr);
                const auto sbufSrv = nvrhi::BindingSetItem::StructuredBuffer_SRV(8, nullptr);
                const auto sbufUav = nvrhi::BindingSetItem::StructuredBuffer_UAV(9, nullptr);
                const auto rbufSrv = nvrhi::BindingSetItem::RawBuffer_SRV(10, nullptr);
                const auto rbufUav = nvrhi::BindingSetItem::RawBuffer_UAV(11, nullptr);
                const auto pcItem = nvrhi::BindingSetItem::PushConstants(12, 32);
                const auto sfItem = nvrhi::BindingSetItem::SamplerFeedbackTexture_UAV(13, nullptr);

                UAA_CHECK_EQ(noneItem.type, nvrhi::ResourceType::None, "None item");
                UAA_CHECK(nvrhi::resolveComponentMapping(srvItem.overrideComponentMapping, nvrhi::ComponentMapping{}) == texDefault, "Texture_SRV packed mapping");
                UAA_CHECK_EQ(uavItem.type, nvrhi::ResourceType::Texture_UAV, "Texture_UAV item");
                UAA_CHECK_EQ(tbufSrv.type, nvrhi::ResourceType::TypedBuffer_SRV, "TypedBuffer_SRV item");
                UAA_CHECK_EQ(tbufUav.type, nvrhi::ResourceType::TypedBuffer_UAV, "TypedBuffer_UAV item");
                UAA_CHECK_EQ(cbItem.type, nvrhi::ResourceType::ConstantBuffer, "ConstantBuffer item");
                UAA_CHECK_EQ(sampItem.type, nvrhi::ResourceType::Sampler, "Sampler item");
                UAA_CHECK_EQ(rtItem.type, nvrhi::ResourceType::RayTracingAccelStruct, "RayTracingAccelStruct item");
                UAA_CHECK_EQ(sbufSrv.type, nvrhi::ResourceType::StructuredBuffer_SRV, "StructuredBuffer_SRV item");
                UAA_CHECK_EQ(sbufUav.type, nvrhi::ResourceType::StructuredBuffer_UAV, "StructuredBuffer_UAV item");
                UAA_CHECK_EQ(rbufSrv.type, nvrhi::ResourceType::RawBuffer_SRV, "RawBuffer_SRV item");
                UAA_CHECK_EQ(rbufUav.type, nvrhi::ResourceType::RawBuffer_UAV, "RawBuffer_UAV item");
                UAA_CHECK_EQ(pcItem.range.byteSize, uint64_t(32), "PushConstants byteSize");
                UAA_CHECK_EQ(sfItem.type, nvrhi::ResourceType::SamplerFeedbackTexture_UAV, "SamplerFeedbackTexture_UAV item");

                nvrhi::BindingSetDesc setDesc;
                UAA_CHECK(setDesc.trackLiveness, "BindingSetDesc default trackLiveness");
                setDesc.addItem(pcItem).setTrackLiveness(false);
                UAA_CHECK(!setDesc.trackLiveness && setDesc.bindings.size() == 1, "BindingSetDesc setters");
            }

            {
                const nvrhi::VertexAttributeDesc va;
                UAA_CHECK_EQ(va.arraySize, 1u, "VertexAttributeDesc default arraySize");
                UAA_CHECK(!va.isInstanced, "VertexAttributeDesc default isInstanced");

                const nvrhi::BlendState bs;
                UAA_CHECK(!bs.alphaToCoverageEnable, "BlendState default alphaToCoverageEnable");
                UAA_CHECK(!bs.usesConstantColor(nvrhi::c_MaxRenderTargets), "Default BlendState does not use constant colour");
                const auto bsConst = nvrhi::BlendState().setRenderTarget(0,
                    nvrhi::BlendState::RenderTarget().enableBlend().setSrcBlend(nvrhi::BlendFactor::ConstantColor));
                UAA_CHECK(bsConst.usesConstantColor(1), "BlendState usesConstantColor detects ConstantColor");

                const nvrhi::RasterState rs;
                UAA_CHECK_EQ(rs.fillMode, nvrhi::RasterFillMode::Solid, "RasterState default fillMode");
                UAA_CHECK_EQ(rs.cullMode, nvrhi::RasterCullMode::Back, "RasterState default cullMode");

                const nvrhi::DepthStencilState ds;
                UAA_CHECK(ds.depthTestEnable && ds.depthWriteEnable, "DepthStencilState default depth flags");
                UAA_CHECK_EQ(ds.depthFunc, nvrhi::ComparisonFunc::Less, "DepthStencilState default depthFunc");
                UAA_CHECK_EQ(ds.stencilReadMask, uint8_t(0xff), "DepthStencilState default stencilReadMask");

                const nvrhi::Viewport vpDef;
                UAA_CHECK_NEAR(vpDef.minZ, 0.f, 1e-6, "Viewport default minZ");
                UAA_CHECK_NEAR(vpDef.maxZ, 1.f, 1e-6, "Viewport default maxZ");
                const nvrhi::Viewport vp(10.f, 110.f, 20.f, 70.f, 0.f, 1.f);
                UAA_CHECK_NEAR(vp.width(), 100.f, 1e-6, "Viewport width");
                UAA_CHECK_NEAR(vp.height(), 50.f, 1e-6, "Viewport height");
                const nvrhi::Rect rc(vp);
                UAA_CHECK_EQ(rc.width(), 100, "Rect(Viewport) width");
                UAA_CHECK_EQ(rc.height(), 50, "Rect(Viewport) height");
            }
        }

        void testRefCounter(TestContext& ctx)
        {
            std::atomic<int> liveCount{0};

            {
                auto* raw = new DummyTrackableResource(liveCount);
                UAA_CHECK_EQ(liveCount.load(), 1, "Initial liveCount after new");
                UAA_CHECK_EQ(raw->GetRefCount(), 1ul, "Initial refcount must be 1");
                UAA_CHECK_EQ(raw->AddRef(), 2ul, "AddRef increments to 2");
                UAA_CHECK_EQ(raw->Release(), 1ul, "Release decrements to 1");
                UAA_CHECK_EQ(raw->Release(), 0ul, "Final Release destroys object");
                UAA_CHECK_EQ(liveCount.load(), 0, "Object destroyed after final Release");
            }

            {
                nvrhi::RefCountPtr<DummyTrackableResource> defPtr;
                nvrhi::RefCountPtr<DummyTrackableResource> nullPtr(nullptr);
                UAA_CHECK(defPtr.Get() == nullptr && nullPtr.Get() == nullptr, "Default/nullptr RefCountPtr");

                auto p1 = nvrhi::RefCountPtr<DummyTrackableResource>::Create(new DummyTrackableResource(liveCount));
                UAA_CHECK_EQ(liveCount.load(), 1, "Create keeps refcount at 1");
                UAA_CHECK_EQ(p1->GetRefCount(), 1ul, "RefCountPtr::Create refcount");

                auto* raw2 = new DummyTrackableResource(liveCount);
                nvrhi::RefCountPtr<DummyTrackableResource> p2(raw2);
                UAA_CHECK_EQ(p2->GetRefCount(), 2ul, "Raw pointer ctor increments refcount");
                raw2->Release();
                UAA_CHECK_EQ(p2->GetRefCount(), 1ul, "Releasing extra raw ref leaves 1");

                nvrhi::RefCountPtr<DummyTrackableResource> pCopy(p1);
                UAA_CHECK_EQ(p1->GetRefCount(), 2ul, "Copy ctor increments refcount");
                pCopy = p2;
                UAA_CHECK_EQ(p1->GetRefCount(), 1ul, "Copy assign releases old");
                UAA_CHECK_EQ(p2->GetRefCount(), 2ul, "Copy assign increments new");

                nvrhi::RefCountPtr<DummyTrackableResource> pMove(std::move(pCopy));
                UAA_CHECK(pCopy.Get() == nullptr, "Moved-from RefCountPtr is null");
                UAA_CHECK_EQ(pMove->GetRefCount(), 2ul, "Move ctor preserves refcount");

                auto derived = nvrhi::RefCountPtr<DerivedTrackableResource>::Create(new DerivedTrackableResource(liveCount));
                nvrhi::RefCountPtr<nvrhi::IResource> baseCopy(derived);
                UAA_CHECK_EQ(derived->GetRefCount(), 2ul, "Convertible copy upcast increments refcount");
                nvrhi::RefCountPtr<nvrhi::IResource> baseMove(std::move(derived));
                UAA_CHECK(derived.Get() == nullptr, "Convertible move upcast nulls source");
                UAA_CHECK_EQ(baseMove->GetRefCount(), 2ul, "Convertible move upcast preserves refcount");

                std::hash<nvrhi::RefCountPtr<nvrhi::IResource>> ptrHasher;
                UAA_CHECK_EQ(ptrHasher(baseCopy), std::hash<nvrhi::IResource*>{}(baseCopy.Get()), "RefCountPtr hash matches raw pointer hash");

                p1.Swap(p2);
                UAA_CHECK(p1.Get() == raw2, "Swap exchanged pointers");

                DummyTrackableResource* detached = p1.Detach();
                UAA_CHECK(p1.Get() == nullptr && detached == raw2, "Detach clears wrapper without Release");
                p1.Attach(detached);
                UAA_CHECK(p1.Get() == detached, "Attach adopts without AddRef");

                DummyTrackableResource** addr = p1.ReleaseAndGetAddressOf();
                UAA_CHECK(*addr == nullptr, "ReleaseAndGetAddressOf releases and zeroes slot");

                UAA_CHECK_EQ(pMove.Reset(), 0ul, "Reset drops last ref to raw2");
                baseCopy.Reset();
                baseMove.Reset();
                p2.Reset();
            }

            UAA_CHECK_EQ(liveCount.load(), 0, "All tracked RefCounter objects must be destroyed");
        }

        void testContainers(TestContext& ctx)
        {
            {
                nvrhi::static_vector<int, 8> vec;
                UAA_CHECK(vec.empty() && vec.size() == 0 && vec.max_size() == 8, "Empty static_vector");

                vec.push_back(10);
                int val = 20;
                vec.push_back(val);
                int& emplaced = vec.emplace_back();
                emplaced = 30;
                UAA_CHECK_EQ(vec.size(), size_t(3), "static_vector size after pushes");
                UAA_CHECK_EQ(vec.front(), 10, "static_vector front");
                UAA_CHECK_EQ(vec.back(), 30, "static_vector back");

                vec.pop_back();
                UAA_CHECK_EQ(vec.size(), size_t(2), "static_vector size after pop_back");

                vec.resize(5);
                UAA_CHECK_EQ(vec.size(), size_t(5), "static_vector resize up");
                UAA_CHECK_EQ(vec[4], 0, "static_vector resize up zero-initialises");
                vec.resize(1);
                UAA_CHECK_EQ(vec.size(), size_t(1), "static_vector resize down");

                nvrhi::static_vector<int, 8> other{ 7, 8, 9 };
                vec.swap(other);
                UAA_CHECK_EQ(vec.size(), size_t(3), "Swap exchanged sizes");
                UAA_CHECK_EQ(other[0], 10, "Swap exchanged values");

                other.fill(42);
                UAA_CHECK_EQ(other.size(), size_t(8), "fill sets size to max_size");
                int sum = 0;
                for (auto it = other.cbegin(); it != other.cend(); ++it)
                {
                    sum += *it;
                }
                UAA_CHECK_EQ(sum, 42 * 8, "static_vector iterators cover all filled elements");
            }

            {
                nvrhi::utils::BitSetAllocator alloc(16, false);
                UAA_CHECK_EQ(alloc.getCapacity(), size_t(16), "BitSetAllocator capacity");

                std::vector<int> seen(16, 0);
                for (int i = 0; i < 16; ++i)
                {
                    const int idx = alloc.allocate();
                    UAA_CHECK(idx >= 0 && idx < 16, "Allocated index in bounds");
                    UAA_CHECK_EQ(seen[idx], 0, "Allocated index must be unique");
                    seen[idx] = 1;
                }
                UAA_CHECK_EQ(alloc.allocate(), -1, "Full BitSetAllocator returns -1");

                alloc.release(-1);
                alloc.release(16);
                alloc.release(999);
                UAA_CHECK_EQ(alloc.allocate(), -1, "Out-of-range releases must not corrupt allocator");

                alloc.release(9);
                alloc.release(3);
                UAA_CHECK_EQ(alloc.allocate(), 3, "Allocator reuses lowest released slot");
                UAA_CHECK_EQ(alloc.allocate(), 9, "Allocator reuses remaining released slot");
                UAA_CHECK_EQ(alloc.allocate(), -1, "Allocator full again");
            }

            {
                constexpr size_t kCapacity = 128;
                nvrhi::utils::BitSetAllocator mtAlloc(kCapacity, true);
                std::array<std::atomic<int>, kCapacity> inUse{};
                for (auto& cell : inUse)
                {
                    cell.store(0, std::memory_order_relaxed);
                }

                std::atomic<bool> collision{false};
                std::vector<std::thread> threads;
                for (int t = 0; t < 4; ++t)
                {
                    threads.emplace_back([&]()
                    {
                        std::vector<int> local;
                        local.reserve(16);
                        for (int iter = 0; iter < 300; ++iter)
                        {
                            for (int k = 0; k < 8; ++k)
                            {
                                const int idx = mtAlloc.allocate();
                                if (idx >= 0)
                                {
                                    if (inUse[idx].exchange(1, std::memory_order_acq_rel) != 0)
                                    {
                                        collision.store(true, std::memory_order_relaxed);
                                    }
                                    local.push_back(idx);
                                }
                            }
                            for (int idx : local)
                            {
                                inUse[idx].store(0, std::memory_order_release);
                                mtAlloc.release(idx);
                            }
                            local.clear();
                        }
                    });
                }
                for (auto& th : threads)
                {
                    th.join();
                }

                UAA_CHECK(!collision.load(), "Multithreaded BitSetAllocator must never yield overlapping live indices");
                for (size_t i = 0; i < kCapacity; ++i)
                {
                    UAA_CHECK(mtAlloc.allocate() >= 0, "All slots must be free after threads join");
                }
                UAA_CHECK_EQ(mtAlloc.allocate(), -1, "All slots consumed after drain check");
            }
        }

        void testObjectAndEnums(TestContext& ctx)
        {
            UAA_CHECK(nvrhi::verifyHeaderVersion(nvrhi::c_HeaderVersion), "Current header version must verify");
            UAA_CHECK(nvrhi::verifyHeaderVersion(), "Default header version must verify");
            UAA_CHECK(!nvrhi::verifyHeaderVersion(nvrhi::c_HeaderVersion + 1), "Mismatched header version must fail");
            UAA_CHECK(!nvrhi::verifyHeaderVersion(0), "Zero header version must fail");

            int dummy = 77;
            const nvrhi::Object intObj(uint64_t(0x123456789abcdef0ull));
            const nvrhi::Object ptrObj(&dummy);
            const nvrhi::Object nullObj(nullptr);
            UAA_CHECK_EQ(intObj.integer, uint64_t(0x123456789abcdef0ull), "Object uint64 storage");
            UAA_CHECK(ptrObj.pointer == &dummy, "Object void* storage");
            int* typedPtr = ptrObj;
            UAA_CHECK(typedPtr == &dummy && *typedPtr == 77, "Object typed pointer conversion");
            UAA_CHECK(nullObj.pointer == nullptr, "Object nullptr storage");

            const auto rs = nvrhi::ResourceStates::CopyDest | nvrhi::ResourceStates::CopySource;
            UAA_CHECK((rs & nvrhi::ResourceStates::CopyDest) != 0u, "ResourceStates operator| and operator&");
            UAA_CHECK(!nvrhi::ResourceStates::Unknown, "ResourceStates operator!");
            UAA_CHECK((~nvrhi::ResourceStates::Unknown & nvrhi::ResourceStates::Common) == uint32_t(nvrhi::ResourceStates::Common),
                "ResourceStates operator~");
            UAA_CHECK_EQ(nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::Buffer,
                uint32_t(nvrhi::FormatSupport::Texture) | uint32_t(nvrhi::FormatSupport::Buffer), "FormatSupport flags");
            UAA_CHECK((nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel) != 0u, "ShaderType flags");
            UAA_CHECK((nvrhi::ColorMask::Red | nvrhi::ColorMask::Alpha) != 0u, "ColorMask flags");
            UAA_CHECK((nvrhi::SharedResourceFlags::Shared | nvrhi::SharedResourceFlags::Shared_NTHandle) != 0u,
                "SharedResourceFlags flags");

            UAA_CHECK_EQ(nvrhi::align(0u, 256u), 0u, "align(0, 256)");
            UAA_CHECK_EQ(nvrhi::align(1u, 256u), 256u, "align(1, 256)");
            UAA_CHECK_EQ(nvrhi::align(256u, 256u), 256u, "align(256, 256)");
            UAA_CHECK_EQ(nvrhi::align(257u, 256u), 512u, "align(257, 256)");

            const std::array<int, 4> a1{ 1, 2, 3, 4 };
            const std::array<int, 4> a2{ 1, 2, 3, 4 };
            const std::array<int, 4> a3{ 1, 9, 3, 8 };
            const std::array<int, 2> a4{ 1, 2 };
            UAA_CHECK(!nvrhi::arraysAreDifferent(a1, a2), "Identical arrays");
            UAA_CHECK(nvrhi::arraysAreDifferent(a1, a3), "Modified element arrays");
            UAA_CHECK(nvrhi::arraysAreDifferent(a1, a4), "Different length arrays");
            UAA_CHECK_EQ(nvrhi::arrayDifferenceMask(a1, a2), 0u, "arrayDifferenceMask equal");
            UAA_CHECK_EQ(nvrhi::arrayDifferenceMask(a1, a3), (1u << 1) | (1u << 3), "arrayDifferenceMask bitmask");
            UAA_CHECK_EQ(nvrhi::arrayDifferenceMask(a1, a4), ~0u, "arrayDifferenceMask size mismatch");

            UAA_CHECK_EQ(nvrhi::hash_to_u32(0x0000000100000002ull), 3u, "hash_to_u32 folds high and low 32 bits");

            const nvrhi::BindingSetItem itemA = nvrhi::BindingSetItem::PushConstants(0, 16);
            const nvrhi::BindingSetItem itemB = nvrhi::BindingSetItem::PushConstants(0, 32);
            UAA_CHECK_NE(std::hash<nvrhi::BindingSetItem>{}(itemA), std::hash<nvrhi::BindingSetItem>{}(itemB),
                "BindingSetItem hash distinguishes sizes");

            const nvrhi::BindingSetDesc setA = nvrhi::BindingSetDesc().addItem(itemA);
            const nvrhi::BindingSetDesc setB = nvrhi::BindingSetDesc().addItem(itemB);
            UAA_CHECK_NE(std::hash<nvrhi::BindingSetDesc>{}(setA), std::hash<nvrhi::BindingSetDesc>{}(setB),
                "BindingSetDesc hash distinguishes items");

            const nvrhi::FramebufferInfo fbA = nvrhi::FramebufferInfo().addColorFormat(nvrhi::Format::RGBA8_UNORM);
            const nvrhi::FramebufferInfo fbB = nvrhi::FramebufferInfo().addColorFormat(nvrhi::Format::RGBA16_FLOAT);
            UAA_CHECK_NE(std::hash<nvrhi::FramebufferInfo>{}(fbA), std::hash<nvrhi::FramebufferInfo>{}(fbB),
                "FramebufferInfo hash distinguishes formats");

            const nvrhi::BlendState bsA;
            const nvrhi::BlendState bsB = nvrhi::BlendState().enableAlphaToCoverage();
            UAA_CHECK_NE(std::hash<nvrhi::BlendState>{}(bsA), std::hash<nvrhi::BlendState>{}(bsB),
                "BlendState hash distinguishes alphaToCoverage");

            const nvrhi::VariableRateShadingState vrsA;
            const nvrhi::VariableRateShadingState vrsB = nvrhi::VariableRateShadingState().setEnabled(true).setShadingRate(nvrhi::VariableShadingRate::e2x2);
            UAA_CHECK_NE(std::hash<nvrhi::VariableRateShadingState>{}(vrsA), std::hash<nvrhi::VariableRateShadingState>{}(vrsB),
                "VariableRateShadingState hash distinguishes enabled/rate");

            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D11), "D3D11") == 0, "D3D11 string");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D12), "D3D12") == 0, "D3D12 string");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::VULKAN), "Vulkan") == 0, "Vulkan string");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::METAL), "Metal") == 0, "Metal string");
            UAA_CHECK(std::strcmp(nvrhi::utils::GraphicsAPIToString(static_cast<nvrhi::GraphicsAPI>(99)), "<UNKNOWN>") == 0, "Invalid API string");

            UAA_CHECK(std::strcmp(nvrhi::utils::DebugNameToString(""), "<UNNAMED>") == 0, "Empty debug name");
            UAA_CHECK(std::strcmp(nvrhi::utils::DebugNameToString("res"), "res") == 0, "Non-empty debug name");

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
                UAA_CHECK(s != nullptr && std::strcmp(s, "<INVALID>") != 0, "Valid TextureDimensionToString");
            }
            UAA_CHECK(std::strcmp(nvrhi::utils::TextureDimensionToString(static_cast<nvrhi::TextureDimension>(99)), "<INVALID>") == 0,
                "Invalid TextureDimensionToString");

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
                UAA_CHECK(s != nullptr && std::strcmp(s, "<INVALID>") != 0, "Valid ShaderStageToString");
            }
            UAA_CHECK(std::strcmp(nvrhi::utils::ShaderStageToString(static_cast<nvrhi::ShaderType>(0x8000)), "<INVALID>") == 0,
                "Invalid ShaderStageToString");

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
                "ResourceType::Count string");

            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Graphics), "Graphics") == 0, "Graphics queue string");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Compute), "Compute") == 0, "Compute queue string");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Copy), "Copy") == 0, "Copy queue string");
            UAA_CHECK(std::strcmp(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Count), "<INVALID>") == 0, "Invalid queue string");
        }
    } // namespace

    UAA_REGISTER_TEST("CPU.FormatMetadata", "cpu", testFormatMetadata);
    UAA_REGISTER_TEST("CPU.FormatBoundaries", "cpu", testFormatBoundaries);
    UAA_REGISTER_TEST("CPU.Subresources", "cpu", testSubresources);
    UAA_REGISTER_TEST("CPU.Slices", "cpu", testSlices);
    UAA_REGISTER_TEST("CPU.BufferRanges", "cpu", testBufferRanges);
    UAA_REGISTER_TEST("CPU.DescriptorDefaults", "cpu", testDescriptorDefaults);
    UAA_REGISTER_TEST("CPU.RefCounter", "cpu", testRefCounter);
    UAA_REGISTER_TEST("CPU.Containers", "cpu", testContainers);
    UAA_REGISTER_TEST("CPU.ObjectAndEnums", "cpu", testObjectAndEnums);
} // namespace nvrhi::uaa
