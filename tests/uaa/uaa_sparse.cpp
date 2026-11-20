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
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace nvrhi::uaa
{
    namespace
    {
        constexpr uint32_t kTileSentinel = 0xCAFEBABEu;
        constexpr uint32_t kDefaultTileByteSize = 65536u;

        // Checks whether the Vulkan physical device supports sparse binding and 2D sparse residency
        // before requesting isTiled = true, avoiding a Debug assert on backends such as MoltenVK.
        bool checkVulkanSparseSupport(TestFixture& fix, std::string& outReason)
        {
            if (fix.api() != nvrhi::GraphicsAPI::VULKAN)
            {
                return true;
            }

#if defined(UAA_WITH_VULKAN) && UAA_WITH_VULKAN
            auto physDev = static_cast<VkPhysicalDevice>(
                fix.device()->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice).pointer);
            if (!physDev)
            {
                outReason = "Vulkan physical device handle unavailable";
                return false;
            }

            PFN_vkGetPhysicalDeviceFeatures fnGetFeatures = nullptr;
            PFN_vkGetPhysicalDeviceSparseImageFormatProperties fnGetSparseProps = nullptr;
#ifdef _WIN32
            HMODULE vkMod = GetModuleHandleA("vulkan-1.dll");
            if (vkMod)
            {
                fnGetFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
                    GetProcAddress(vkMod, "vkGetPhysicalDeviceFeatures"));
                fnGetSparseProps = reinterpret_cast<PFN_vkGetPhysicalDeviceSparseImageFormatProperties>(
                    GetProcAddress(vkMod, "vkGetPhysicalDeviceSparseImageFormatProperties"));
            }
#else
            void* selfMod = dlopen(nullptr, RTLD_LAZY);
            if (selfMod)
            {
                fnGetFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
                    dlsym(selfMod, "vkGetPhysicalDeviceFeatures"));
                fnGetSparseProps = reinterpret_cast<PFN_vkGetPhysicalDeviceSparseImageFormatProperties>(
                    dlsym(selfMod, "vkGetPhysicalDeviceSparseImageFormatProperties"));
            }
            if (!fnGetFeatures)
            {
#if defined(__APPLE__)
                const char* libCandidates[] = { "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib" };
#else
                const char* libCandidates[] = { "libvulkan.so.1", "libvulkan.so" };
#endif
                for (const char* libName : libCandidates)
                {
                    void* vkLib = dlopen(libName, RTLD_LAZY | RTLD_NOLOAD);
                    if (vkLib)
                    {
                        fnGetFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
                            dlsym(vkLib, "vkGetPhysicalDeviceFeatures"));
                        fnGetSparseProps = reinterpret_cast<PFN_vkGetPhysicalDeviceSparseImageFormatProperties>(
                            dlsym(vkLib, "vkGetPhysicalDeviceSparseImageFormatProperties"));
                        if (fnGetFeatures)
                        {
                            break;
                        }
                    }
                }
            }
#endif
            if (fnGetFeatures)
            {
                VkPhysicalDeviceFeatures features{};
                fnGetFeatures(physDev, &features);
                if (!features.sparseBinding || !features.sparseResidencyImage2D)
                {
                    outReason = "Vulkan physical device does not advertise sparseBinding and sparseResidencyImage2D";
                    return false;
                }

                if (fnGetSparseProps)
                {
                    uint32_t propCount = 0;
                    fnGetSparseProps(
                        physDev,
                        VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_TYPE_2D,
                        VK_SAMPLE_COUNT_1_BIT,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_TILING_OPTIMAL,
                        &propCount,
                        nullptr);
                    if (propCount == 0)
                    {
                        outReason = "Vulkan physical device reports zero sparse image format properties for RGBA8_UNORM";
                        return false;
                    }
                }
            }
            return true;
#else
            (void)outReason;
            return true;
#endif
        }

        nvrhi::TextureDesc makeTiledTextureDesc()
        {
            nvrhi::TextureDesc desc;
            desc.setWidth(512)
                .setHeight(512)
                .setMipLevels(4)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setIsUAV(true)
                .setIsRenderTarget(true)
                .setDebugName("uaa_tiled_texture");
            desc.isTiled = true;
            desc.enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            return desc;
        }

        void mapSingleTile(
            TestFixture& fix,
            nvrhi::ITexture* tex,
            const nvrhi::TileShape& tileShape,
            nvrhi::IHeap* heap,
            uint64_t heapOffsetBytes,
            uint32_t tileX,
            uint32_t tileY,
            nvrhi::CommandQueue queue)
        {
            nvrhi::TiledTextureCoordinate coord;
            coord.mipLevel = 0;
            coord.arrayLevel = 0;
            coord.x = tileX;
            coord.y = tileY;
            coord.z = 0;

            nvrhi::TiledTextureRegion region;
            region.tilesNum = 0;
            if (fix.api() == nvrhi::GraphicsAPI::D3D12)
            {
                region.width = tileShape.widthInTexels;
                region.height = tileShape.heightInTexels;
                region.depth = std::max(1u, tileShape.depthInTexels);
            }
            else
            {
                region.width = 1;
                region.height = 1;
                region.depth = 1;
            }

            uint64_t byteOffset = heapOffsetBytes;

            nvrhi::TextureTilesMapping mapping;
            mapping.tiledTextureCoordinates = &coord;
            mapping.tiledTextureRegions = &region;
            mapping.byteOffsets = heap ? &byteOffset : nullptr;
            mapping.numTextureRegions = 1;
            mapping.heap = heap;

            fix.device()->updateTextureTileMappings(tex, &mapping, 1, queue);
        }

        void testSprTiling(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (fix.api() == nvrhi::GraphicsAPI::D3D11)
            {
                ctx.skip("D3D11 does not support tiled textures or getTextureTiling");
            }

            std::string sparseSkipReason;
            if (!checkVulkanSparseSupport(fix, sparseSkipReason))
            {
                ctx.skip(sparseSkipReason);
            }

            nvrhi::TextureHandle tiledTex = fix.device()->createTexture(makeTiledTextureDesc());
            if (!tiledTex)
            {
                ctx.skip("Device returned nullptr for tiled texture (sparse textures unsupported)");
            }

            uint32_t numTiles = 0xDEAD0001u;
            nvrhi::PackedMipDesc packedMip{};
            packedMip.numStandardMips = 0xDEAD0002u;
            packedMip.numPackedMips = 0xDEAD0003u;
            nvrhi::TileShape tileShape{};
            tileShape.widthInTexels = 0xDEAD0004u;
            tileShape.heightInTexels = 0xDEAD0005u;
            tileShape.depthInTexels = 0xDEAD0006u;

            uint32_t subresourceTilingsNum = 2;
            std::vector<nvrhi::SubresourceTiling> tilings(4);
            for (auto& entry : tilings)
            {
                entry.widthInTiles = kTileSentinel;
                entry.heightInTiles = kTileSentinel;
                entry.depthInTiles = kTileSentinel;
                entry.startTileIndexInOverallResource = kTileSentinel;
            }

            fix.device()->getTextureTiling(
                tiledTex, &numTiles, &packedMip, &tileShape, &subresourceTilingsNum, tilings.data());

            if (numTiles == 0 || subresourceTilingsNum == 0)
            {
                UAA_CHECK_EQ(tilings[2].widthInTiles, kTileSentinel, "Unused tiling slot must remain untouched");
                UAA_CHECK_EQ(tilings[3].widthInTiles, kTileSentinel, "Unused tiling slot must remain untouched");
                ctx.skip("Tiled textures / sparse tiling not supported on this device (numTiles == 0)");
            }

            UAA_CHECK(subresourceTilingsNum <= 2, "getTextureTiling must respect caller subresourceTilingsNum capacity");
            UAA_CHECK_EQ(tilings[2].widthInTiles, kTileSentinel, "Overflow sentinel slot 2 must remain intact");
            UAA_CHECK_EQ(tilings[3].startTileIndexInOverallResource, kTileSentinel, "Overflow sentinel slot 3 must remain intact");

            UAA_CHECK(tileShape.widthInTexels > 0, "TileShape widthInTexels must be positive");
            UAA_CHECK(tileShape.heightInTexels > 0, "TileShape heightInTexels must be positive");
            UAA_CHECK(tileShape.depthInTexels > 0, "TileShape depthInTexels must be positive");

            uint32_t fullCount = 4;
            std::vector<nvrhi::SubresourceTiling> fullTilings(4);
            fix.device()->getTextureTiling(
                tiledTex, &numTiles, &packedMip, &tileShape, &fullCount, fullTilings.data());

            UAA_CHECK_EQ(fullCount, 4u, "Full tiling query should populate 4 mip levels");
            UAA_CHECK_EQ(packedMip.numStandardMips + packedMip.numPackedMips, 4u,
                "Standard plus packed mips must equal total mipLevels");
            UAA_CHECK(fullTilings[0].widthInTiles > 0 && fullTilings[0].heightInTiles > 0,
                "Mip 0 must span at least 1x1 tiles");

            std::ostringstream oss;
            oss << "numTiles=" << numTiles
                << ", tileShape=" << tileShape.widthInTexels << "x" << tileShape.heightInTexels << "x" << tileShape.depthInTexels
                << ", standardMips=" << packedMip.numStandardMips
                << ", packedMips=" << packedMip.numPackedMips;
            ctx.note(oss.str());
        }

        void testSprMapping(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (fix.api() == nvrhi::GraphicsAPI::D3D11)
            {
                ctx.skip("D3D11 does not support tiled textures or sparse tile mapping");
            }

            std::string sparseSkipReason;
            if (!checkVulkanSparseSupport(fix, sparseSkipReason))
            {
                ctx.skip(sparseSkipReason);
            }

            nvrhi::TextureHandle tiledTex = fix.device()->createTexture(makeTiledTextureDesc());
            if (!tiledTex)
            {
                ctx.skip("Tiled textures / sparse mapping not supported on this device");
            }

            uint32_t numTiles = 0;
            nvrhi::PackedMipDesc packedMip{};
            nvrhi::TileShape tileShape{};
            uint32_t subresourceCount = 4;
            std::vector<nvrhi::SubresourceTiling> tilings(4);
            fix.device()->getTextureTiling(
                tiledTex, &numTiles, &packedMip, &tileShape, &subresourceCount, tilings.data());

            if (numTiles == 0 || tileShape.widthInTexels == 0 || tileShape.heightInTexels == 0)
            {
                ctx.skip("Tiled textures / sparse mapping not supported on this device");
            }

            const nvrhi::MemoryRequirements memReq = fix.device()->getTextureMemoryRequirements(tiledTex);
            const uint64_t tileBytes = (memReq.alignment >= kDefaultTileByteSize) ? memReq.alignment : kDefaultTileByteSize;

            nvrhi::HeapDesc heapDesc;
            heapDesc.setCapacity(tileBytes * 4).setType(nvrhi::HeapType::DeviceLocal).setDebugName("uaa_sparse_tile_heap");
            nvrhi::HeapHandle heap = fix.device()->createHeap(heapDesc);
            UAA_CHECK(heap != nullptr, "Failed to create DeviceLocal heap for sparse tile mapping");

            const uint32_t tileW = tileShape.widthInTexels;
            const uint32_t tileH = tileShape.heightInTexels;

            nvrhi::TextureDesc tileScratchDesc;
            tileScratchDesc.setWidth(tileW)
                .setHeight(tileH)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setDebugName("uaa_tile_scratch");
            tileScratchDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            nvrhi::TextureHandle srcTileTex = fix.device()->createTexture(tileScratchDesc);
            UAA_CHECK(srcTileTex != nullptr, "Failed to create source tile texture");
            nvrhi::TextureHandle dstTileTex = fix.device()->createTexture(tileScratchDesc);
            UAA_CHECK(dstTileTex != nullptr, "Failed to create readback tile texture");

            const size_t tilePixelBytes = size_t(tileW) * size_t(tileH) * 4u;
            std::vector<uint8_t> patternA(tilePixelBytes);
            std::vector<uint8_t> patternB(tilePixelBytes);
            for (size_t i = 0; i < tilePixelBytes; i += 4)
            {
                patternA[i + 0] = 0x11;
                patternA[i + 1] = 0xAA;
                patternA[i + 2] = 0x33;
                patternA[i + 3] = 0xFF;

                patternB[i + 0] = 0xCC;
                patternB[i + 1] = 0x22;
                patternB[i + 2] = 0xDD;
                patternB[i + 3] = 0xFF;
            }

            // Map tile (0, 0) to heap offset 0 on Graphics queue, upload patternA, and read back.
            mapSingleTile(fix, tiledTex, tileShape, heap, 0, 0, 0, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed after initial tile map");

            fix.uploadTexture2D(srcTileTex, 0, 0, patternA.data(), size_t(tileW) * 4u);

            nvrhi::TextureSlice srcSlice;
            srcSlice.setWidth(tileW).setHeight(tileH).setDepth(1);
            nvrhi::TextureSlice dstSlice;
            dstSlice.setOrigin(0, 0, 0).setWidth(tileW).setHeight(tileH).setDepth(1);

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed");
                cmd->open();
                cmd->copyTexture(tiledTex, dstSlice, srcTileTex, srcSlice);
                cmd->copyTexture(dstTileTex, srcSlice, tiledTex, dstSlice);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            std::vector<uint8_t> readback;
            fix.readbackTexture2D(dstTileTex, 0, 0, readback);
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(readback.data(), patternA.data(), tilePixelBytes, diff),
                "Mapped tile patternA mismatch: " + diff);

            // Remap tile (0, 0) to heap offset tileBytes, write patternB, and verify isolation.
            mapSingleTile(fix, tiledTex, tileShape, heap, tileBytes, 0, 0, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed after tile remap");

            fix.uploadTexture2D(srcTileTex, 0, 0, patternB.data(), size_t(tileW) * 4u);
            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed");
                cmd->open();
                cmd->copyTexture(tiledTex, dstSlice, srcTileTex, srcSlice);
                cmd->copyTexture(dstTileTex, srcSlice, tiledTex, dstSlice);
                cmd->close();
                fix.submitAndWait(cmd);
            }
            fix.readbackTexture2D(dstTileTex, 0, 0, readback);
            UAA_CHECK(TestFixture::checkBytesEqual(readback.data(), patternB.data(), tilePixelBytes, diff),
                "Remapped tile patternB mismatch: " + diff);

            // Unmap tile (0, 0) cleanly, then remap back to offset 0 to confirm backing retention.
            mapSingleTile(fix, tiledTex, tileShape, nullptr, 0, 0, 0, nvrhi::CommandQueue::Graphics);
            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed after tile unmap");

            const nvrhi::CommandQueue remapQueue =
                (fix.api() == nvrhi::GraphicsAPI::D3D12 && fix.features().supports(nvrhi::Feature::CopyQueue))
                ? nvrhi::CommandQueue::Copy
                : nvrhi::CommandQueue::Graphics;
            mapSingleTile(fix, tiledTex, tileShape, heap, 0, 0, 0, remapQueue);
            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed after tile rebind");

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed");
                cmd->open();
                cmd->copyTexture(dstTileTex, srcSlice, tiledTex, dstSlice);
                cmd->close();
                fix.submitAndWait(cmd);
            }
            fix.readbackTexture2D(dstTileTex, 0, 0, readback);
            UAA_CHECK(TestFixture::checkBytesEqual(readback.data(), patternA.data(), tilePixelBytes, diff),
                "Rebound tile at offset 0 lost original patternA: " + diff);

            if (fix.api() == nvrhi::GraphicsAPI::D3D12 && fix.features().supports(nvrhi::Feature::ComputeQueue))
            {
                mapSingleTile(fix, tiledTex, tileShape, nullptr, 0, 0, 0, nvrhi::CommandQueue::Compute);
                UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed after compute queue unmap");
            }
        }

        void testSprSamplerFeedback(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::SamplerFeedback))
            {
                if (fix.isValidationActive() && fix.api() != nvrhi::GraphicsAPI::D3D12)
                {
                    nvrhi::TextureDesc pairedDesc;
                    pairedDesc.setWidth(128).setHeight(128).setMipLevels(4)
                        .setFormat(nvrhi::Format::RGBA8_UNORM)
                        .setDebugName("uaa_sf_paired_dummy");
                    pairedDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                    nvrhi::TextureHandle paired = fix.device()->createTexture(pairedDesc);
                    UAA_CHECK(paired != nullptr, "Failed to create dummy paired texture");

                    nvrhi::SamplerFeedbackTextureDesc fbDesc;
                    fbDesc.samplerFeedbackFormat = nvrhi::SamplerFeedbackFormat::MinMipOpaque;
                    fbDesc.samplerFeedbackMipRegionX = 4;
                    fbDesc.samplerFeedbackMipRegionY = 4;
                    fbDesc.samplerFeedbackMipRegionZ = 1;
                    fbDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
                    fbDesc.keepInitialState = true;

                    ExpectErrorsGuard expectErr(fix);
                    nvrhi::SamplerFeedbackTextureHandle fb =
                        fix.device()->createSamplerFeedbackTexture(paired, fbDesc);
                    UAA_CHECK(fb == nullptr, "Unsupported createSamplerFeedbackTexture must return nullptr");
                    UAA_CHECK(expectErr.errorCount() > 0,
                        "Validation layer must report error for unsupported createSamplerFeedbackTexture");
                }
                ctx.skip("Feature::SamplerFeedback not supported on this device");
            }

            nvrhi::TextureDesc pairedDesc;
            pairedDesc.setWidth(256)
                .setHeight(256)
                .setMipLevels(4)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setDebugName("uaa_sf_paired");
            pairedDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
            nvrhi::TextureHandle paired = fix.device()->createTexture(pairedDesc);
            UAA_CHECK(paired != nullptr, "Failed to create sampler feedback paired texture");

            nvrhi::SamplerFeedbackTextureDesc fbDesc;
            fbDesc.samplerFeedbackFormat = nvrhi::SamplerFeedbackFormat::MinMipOpaque;
            fbDesc.samplerFeedbackMipRegionX = 4;
            fbDesc.samplerFeedbackMipRegionY = 4;
            fbDesc.samplerFeedbackMipRegionZ = 1;
            fbDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            fbDesc.keepInitialState = true;

            nvrhi::SamplerFeedbackTextureHandle fb = fix.device()->createSamplerFeedbackTexture(paired, fbDesc);
            UAA_CHECK(fb != nullptr, "createSamplerFeedbackTexture returned nullptr despite Feature::SamplerFeedback");
            UAA_CHECK(fb->getPairedTexture().Get() == paired.Get(), "getPairedTexture mismatch");
            UAA_CHECK_EQ(static_cast<uint32_t>(fb->getDesc().samplerFeedbackFormat),
                static_cast<uint32_t>(nvrhi::SamplerFeedbackFormat::MinMipOpaque),
                "SamplerFeedback format mismatch");

            const uint32_t regionCountX = 256u / 4u;
            const uint32_t regionCountY = 256u / 4u;
            const uint32_t decodedBytes = regionCountX * regionCountY;

            nvrhi::BufferDesc decodeBufDesc;
            decodeBufDesc.setByteSize(decodedBytes)
                .setCanHaveUAVs(true)
                .setDebugName("uaa_sf_decoded");
            decodeBufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            nvrhi::BufferHandle decodeBuf = fix.device()->createBuffer(decodeBufDesc);
            UAA_CHECK(decodeBuf != nullptr, "Failed to create sampler feedback decode buffer");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList failed");
            cmd->open();
            cmd->clearSamplerFeedbackTexture(fb);
            cmd->setSamplerFeedbackTextureState(fb, nvrhi::ResourceStates::ResolveSource);
            cmd->decodeSamplerFeedbackTexture(decodeBuf, fb, nvrhi::Format::R8_UINT);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> decoded(decodedBytes, 0);
            fix.readbackBuffer(decodeBuf, decoded.data(), decoded.size());
            for (size_t i = 0; i < decoded.size(); ++i)
            {
                UAA_CHECK_EQ(decoded[i], static_cast<uint8_t>(0xFFu),
                    "Cleared MinMipOpaque sampler feedback must decode to 0xFF");
            }
        }

        void testSprNativeFeedback(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::SamplerFeedback))
            {
                ctx.skip("Feature::SamplerFeedback not supported on this device");
            }

            nvrhi::TextureDesc pairedDesc;
            pairedDesc.setWidth(128)
                .setHeight(128)
                .setMipLevels(4)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setDebugName("uaa_native_sf_paired");
            pairedDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
            nvrhi::TextureHandle paired = fix.device()->createTexture(pairedDesc);
            UAA_CHECK(paired != nullptr, "Failed to create paired texture");

            nvrhi::SamplerFeedbackTextureDesc fbDesc;
            fbDesc.samplerFeedbackFormat = nvrhi::SamplerFeedbackFormat::MinMipOpaque;
            fbDesc.samplerFeedbackMipRegionX = 4;
            fbDesc.samplerFeedbackMipRegionY = 4;
            fbDesc.samplerFeedbackMipRegionZ = 1;
            fbDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            fbDesc.keepInitialState = true;

            nvrhi::SamplerFeedbackTextureHandle baseFb = fix.device()->createSamplerFeedbackTexture(paired, fbDesc);
            UAA_CHECK(baseFb != nullptr, "createSamplerFeedbackTexture returned nullptr");

            const nvrhi::Object nativeRes = baseFb->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
            UAA_CHECK(nativeRes.pointer != nullptr, "Expected valid native D3D12 resource for sampler feedback");

            // Use rawDevice when validation is active because DeviceWrapper::createSamplerFeedbackForNativeTexture
            // recurses into itself on D3D12.
            nvrhi::IDevice* targetDevice = fix.isValidationActive() ? fix.rawDevice() : fix.device();
            if (fix.isValidationActive())
            {
                ctx.note("Bypassing validation wrapper due to D3D12 createSamplerFeedbackForNativeTexture self-recursion");
            }

            nvrhi::SamplerFeedbackTextureHandle wrapped = targetDevice->createSamplerFeedbackForNativeTexture(
                nvrhi::ObjectTypes::D3D12_Resource, nativeRes, paired);
            UAA_CHECK(wrapped != nullptr, "createSamplerFeedbackForNativeTexture returned nullptr");
            UAA_CHECK(wrapped->getPairedTexture().Get() == paired.Get(),
                "Wrapped native sampler feedback getPairedTexture mismatch");
            UAA_CHECK_EQ(wrapped->getDesc().samplerFeedbackMipRegionX, 4u, "Wrapped mip region X mismatch");
            UAA_CHECK_EQ(wrapped->getDesc().samplerFeedbackMipRegionY, 4u, "Wrapped mip region Y mismatch");
        }

        void testSprUnsupported(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool sfSupported = fix.features().supports(nvrhi::Feature::SamplerFeedback);
            ctx.note(std::string("Feature::SamplerFeedback=") + (sfSupported ? "true" : "false"));

            if (!sfSupported && fix.api() != nvrhi::GraphicsAPI::D3D12)
            {
                if (fix.api() == nvrhi::GraphicsAPI::METAL || fix.isValidationActive())
                {
                    nvrhi::TextureDesc pairedDesc;
                    pairedDesc.setWidth(64).setHeight(64).setFormat(nvrhi::Format::RGBA8_UNORM)
                        .setDebugName("uaa_sf_unsupported_paired");
                    pairedDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                    nvrhi::TextureHandle paired = fix.device()->createTexture(pairedDesc);
                    UAA_CHECK(paired != nullptr, "Failed to create paired texture");

                    ExpectErrorsGuard expectErr(fix);
                    nvrhi::SamplerFeedbackTextureHandle fb =
                        fix.device()->createSamplerFeedbackTexture(paired, nvrhi::SamplerFeedbackTextureDesc{});
                    UAA_CHECK(fb == nullptr, "Unsupported createSamplerFeedbackTexture must return nullptr");
                }
            }

            // On Metal, verify ordinary non-tiled textures report zero tiles cleanly rather than inferring sparse support.
            if (fix.api() == nvrhi::GraphicsAPI::METAL)
            {
                nvrhi::TextureDesc ordinaryDesc;
                ordinaryDesc.setWidth(64).setHeight(64).setMipLevels(2)
                    .setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setDebugName("uaa_ordinary_non_tiled");
                ordinaryDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                nvrhi::TextureHandle ordinary = fix.device()->createTexture(ordinaryDesc);
                UAA_CHECK(ordinary != nullptr, "Failed to create ordinary texture");

                uint32_t numTiles = 0x12345678u;
                nvrhi::PackedMipDesc packedMip{};
                nvrhi::TileShape tileShape{};
                uint32_t subresourceCount = 2;
                std::vector<nvrhi::SubresourceTiling> tilings(2);
                tilings[0].widthInTiles = kTileSentinel;
                tilings[1].widthInTiles = kTileSentinel;

                fix.device()->getTextureTiling(
                    ordinary, &numTiles, &packedMip, &tileShape, &subresourceCount, tilings.data());
                UAA_CHECK_EQ(numTiles, 0u, "Metal getTextureTiling must report 0 tiles");
                UAA_CHECK_EQ(subresourceCount, 0u, "Metal getTextureTiling must report 0 subresource tilings");
                UAA_CHECK_EQ(tilings[0].widthInTiles, kTileSentinel, "Metal getTextureTiling must not write output array");
            }

            ctx.note("Verified unsupported sparse/sampler feedback behaviour cleanly");
        }
    } // namespace

    UAA_REGISTER_TEST("SPR.Tiling", "optional", testSprTiling);
    UAA_REGISTER_TEST("SPR.Mapping", "optional", testSprMapping);
    UAA_REGISTER_TEST("SPR.SamplerFeedback", "optional", testSprSamplerFeedback);
    UAA_REGISTER_TEST("SPR.NativeFeedback", "optional", testSprNativeFeedback);
    UAA_REGISTER_TEST("SPR.Unsupported", "optional", testSprUnsupported);
} // namespace nvrhi::uaa
