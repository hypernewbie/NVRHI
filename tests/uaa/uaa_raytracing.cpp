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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct RayQueryParamsCB
        {
            float origins[8][4];
            float dirs[8][4];
            uint32_t rayMask;
            uint32_t rayCount;
            uint32_t p0;
            uint32_t p1;
            uint32_t cbPad[60];
        };

        struct TriangleBlasBundle
        {
            nvrhi::BufferHandle vertexBuffer;
            nvrhi::BufferHandle indexBuffer;
            nvrhi::rt::GeometryDesc geomDesc;
            nvrhi::rt::AccelStructHandle blas;
        };

        TriangleBlasBundle createAndBuildUnitTriangleBlas(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::rt::AccelStructBuildFlags buildFlags,
            nvrhi::rt::GeometryFlags geomFlags,
            bool indexed,
            const nvrhi::rt::AffineTransform* optTransform = nullptr)
        {
            TriangleBlasBundle bundle;

            const float vertices[9] = {
                -1.0f, -1.0f, 0.0f,
                 3.0f, -1.0f, 0.0f,
                -1.0f,  3.0f, 0.0f
            };

            nvrhi::BufferDesc vbDesc;
            vbDesc.setByteSize(sizeof(vertices))
                .setIsVertexBuffer(true)
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_rt_tri_vb");
            vbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            bundle.vertexBuffer = fix.device()->createBuffer(vbDesc);
            UAA_CHECK(bundle.vertexBuffer != nullptr, "Failed to create RT triangle vertex buffer");
            fix.uploadBuffer(bundle.vertexBuffer, vertices, sizeof(vertices));

            nvrhi::rt::GeometryTriangles tri;
            tri.setVertexBuffer(bundle.vertexBuffer)
                .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
                .setVertexCount(3)
                .setVertexStride(sizeof(float) * 3);

            if (indexed)
            {
                const uint32_t indices[3] = { 0u, 1u, 2u };
                nvrhi::BufferDesc ibDesc;
                ibDesc.setByteSize(sizeof(indices))
                    .setIsIndexBuffer(true)
                    .setIsAccelStructBuildInput(true)
                    .setDebugName("uaa_rt_tri_ib");
                ibDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
                bundle.indexBuffer = fix.device()->createBuffer(ibDesc);
                UAA_CHECK(bundle.indexBuffer != nullptr, "Failed to create RT triangle index buffer");
                fix.uploadBuffer(bundle.indexBuffer, indices, sizeof(indices));

                tri.setIndexBuffer(bundle.indexBuffer)
                    .setIndexFormat(nvrhi::Format::R32_UINT)
                    .setIndexCount(3);
            }

            bundle.geomDesc.setTriangles(tri).setFlags(geomFlags);
            if (optTransform)
            {
                bundle.geomDesc.setTransform(*optTransform);
            }

            nvrhi::rt::AccelStructDesc asDesc;
            asDesc.addBottomLevelGeometry(bundle.geomDesc)
                .setBuildFlags(buildFlags)
                .setDebugName("uaa_rt_tri_blas");
            bundle.blas = fix.device()->createAccelStruct(asDesc);
            UAA_CHECK(bundle.blas != nullptr, "createAccelStruct(BLAS) returned nullptr");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildBottomLevelAccelStruct(bundle.blas, &bundle.geomDesc, 1, buildFlags);
            cmd->close();
            fix.submitAndWait(cmd);

            return bundle;
        }

        void runRayQueryDispatch(
            TestFixture& fix,
            TestContext& ctx,
            nvrhi::rt::IAccelStruct* tlas,
            const RayQueryParamsCB& params,
            std::vector<uint32_t>& outWords)
        {
            nvrhi::ShaderHandle cs = fix.createShader("cs_ray_query", nvrhi::ShaderType::Compute);

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1))
                .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(2));
            nvrhi::BindingLayoutHandle layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create ray query binding layout");

            nvrhi::ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(cs).addBindingLayout(layout);
            nvrhi::ComputePipelineHandle pso = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "Failed to create ray query compute pipeline");

            const size_t outCount = 8u * 4u;
            nvrhi::BufferDesc outDesc;
            outDesc.setByteSize(outCount * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setDebugName("uaa_ray_query_out");
            outDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle outBuf = fix.device()->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "Failed to create ray query output buffer");

            nvrhi::BufferDesc cbDesc;
            cbDesc.setByteSize(sizeof(RayQueryParamsCB))
                .setIsConstantBuffer(true)
                .setDebugName("uaa_ray_query_cb");
            cbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer);
            nvrhi::BufferHandle cb = fix.device()->createBuffer(cbDesc);
            UAA_CHECK(cb != nullptr, "Failed to create ray query constant buffer");
            fix.uploadBuffer(cb, &params, sizeof(params));

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, outBuf))
                .addItem(nvrhi::BindingSetItem::ConstantBuffer(2, cb));
            nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "Failed to create ray query binding set");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->clearBufferUInt(outBuf, 0u);
            nvrhi::ComputeState state;
            state.setPipeline(pso).addBindingSet(set);
            cmd->setComputeState(state);
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            outWords.resize(outCount, 0u);
            fix.readbackBuffer(outBuf, outWords.data(), outWords.size() * sizeof(uint32_t));
        }

        float uintBitsToFloat(uint32_t bits)
        {
            float f = 0.0f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }

        uint32_t floatToUintBits(float f)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &f, sizeof(bits));
            return bits;
        }

        void testRtPrebuild(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(16)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::AllowUpdate);

            uint64_t prevResultBytes = 0;
            for (uint32_t count : { 0u, 1u, 4u, 16u })
            {
                nvrhi::rt::AccelStructPrebuildInfo info{};
                const bool ok = fix.device()->queryTopLevelAccelStructPrebuildInfo(tlasDesc, count, info);
                UAA_CHECK(ok, "queryTopLevelAccelStructPrebuildInfo should succeed for valid capacity");
                UAA_CHECK(info.resultBytes > 0, "TLAS prebuild resultBytes must be positive");
                UAA_CHECK(info.scratchBytes > 0, "TLAS prebuild scratchBytes must be positive");
                UAA_CHECK(info.updateScratchBytes > 0, "TLAS with AllowUpdate must report updateScratchBytes > 0");
                UAA_CHECK(info.resultBytes >= prevResultBytes, "TLAS resultBytes must be monotonic with instanceCount");
                prevResultBytes = info.resultBytes;
            }

            for (auto flags : {
                     nvrhi::rt::AccelStructBuildFlags::AllowEmptyInstances | nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                     nvrhi::rt::AccelStructBuildFlags::PreferFastBuild })
            {
                nvrhi::rt::AccelStructDesc variantDesc;
                variantDesc.setTopLevelMaxInstances(8).setBuildFlags(flags);
                nvrhi::rt::AccelStructPrebuildInfo info{};
                UAA_CHECK(fix.device()->queryTopLevelAccelStructPrebuildInfo(variantDesc, 8, info),
                    "Prebuild query failed for build flag variant");
                UAA_CHECK(info.resultBytes > 0 && info.scratchBytes > 0,
                    "Variant prebuild sizes must be positive");
            }

            // Over-capacity query must return false and preserve sentinel output.
            {
                nvrhi::rt::AccelStructPrebuildInfo sentinel{ 0x1111ull, 0x2222ull, 0x3333ull };
                const bool ok = fix.device()->queryTopLevelAccelStructPrebuildInfo(tlasDesc, 17, sentinel);
                UAA_CHECK(!ok, "Over-capacity TLAS prebuild query must return false");
                UAA_CHECK_EQ(sentinel.resultBytes, 0x1111ull, "Over-capacity query must preserve resultBytes");
                UAA_CHECK_EQ(sentinel.scratchBytes, 0x2222ull, "Over-capacity query must preserve scratchBytes");
                UAA_CHECK_EQ(sentinel.updateScratchBytes, 0x3333ull, "Over-capacity query must preserve updateScratchBytes");
            }

            // BLAS descriptor must return false and preserve sentinel output.
            {
                nvrhi::rt::AccelStructDesc blasDesc;
                blasDesc.setIsTopLevel(false);
                nvrhi::rt::AccelStructPrebuildInfo sentinel{ 0xAAAAull, 0xBBBBull, 0xCCCCull };
                const bool ok = fix.device()->queryTopLevelAccelStructPrebuildInfo(blasDesc, 1, sentinel);
                UAA_CHECK(!ok, "Non-top-level AS prebuild query must return false");
                UAA_CHECK_EQ(sentinel.resultBytes, 0xAAAAull, "BLAS query must preserve resultBytes");
                UAA_CHECK_EQ(sentinel.scratchBytes, 0xBBBBull, "BLAS query must preserve scratchBytes");
                UAA_CHECK_EQ(sentinel.updateScratchBytes, 0xCCCCull, "BLAS query must preserve updateScratchBytes");
            }
        }

        void testRtBottomLevel(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            const nvrhi::rt::AffineTransform shiftTransform = {
                1.0f, 0.0f, 0.0f, 1.5f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 2.0f
            };

            TriangleBlasBundle nonIndexed = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::AllowUpdate | nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                false,
                &shiftTransform);
            UAA_CHECK(!nonIndexed.blas->getDesc().isTopLevel, "Non-indexed BLAS must have isTopLevel == false");

            TriangleBlasBundle indexed = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                true,
                nullptr);
            UAA_CHECK(!indexed.blas->getDesc().isTopLevel, "Indexed BLAS must have isTopLevel == false");
            UAA_CHECK(indexed.blas->getDeviceAddress() != 0, "Indexed BLAS device address must be non-zero");
        }

        void testRtTopLevel(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                true);

            const nvrhi::rt::AffineTransform tx1 = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 5.0f
            };

            nvrhi::rt::InstanceDesc instances[2];
            instances[0].setInstanceID(10).setInstanceMask(0x01).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blas.blas);
            instances[1].setInstanceID(20).setInstanceMask(0x02).setTransform(tx1).setBLAS(blas.blas);

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(2)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_rt_tlas");
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned nullptr");
            UAA_CHECK(tlas->getDesc().isTopLevel, "TLAS must report isTopLevel == true");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildTopLevelAccelStruct(tlas, instances, 2, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            cmd->close();
            fix.submitAndWait(cmd);

            if (!fix.features().supports(nvrhi::Feature::RayQuery))
            {
                ctx.note("Feature::RayQuery not supported; verified TLAS build only");
                return;
            }

            RayQueryParamsCB params{};
            params.rayMask = 0xFFu;
            params.rayCount = 3u;
            // Ray 0 hits instance 0 at z = 0 (t = 2.0).
            params.origins[0][0] = 0.0f; params.origins[0][1] = 0.0f; params.origins[0][2] = -2.0f; params.origins[0][3] = 0.0f;
            params.dirs[0][0] = 0.0f;    params.dirs[0][1] = 0.0f;    params.dirs[0][2] = 1.0f;     params.dirs[0][3] = 0.0f;
            // Ray 1 starts past z = 0 and hits instance 1 at z = 5 (t = 3.0).
            params.origins[1][0] = 0.0f; params.origins[1][1] = 0.0f; params.origins[1][2] = 2.0f;  params.origins[1][3] = 0.0f;
            params.dirs[1][0] = 0.0f;    params.dirs[1][1] = 0.0f;    params.dirs[1][2] = 1.0f;     params.dirs[1][3] = 0.0f;
            // Ray 2 misses both instances.
            params.origins[2][0] = 20.0f; params.origins[2][1] = 20.0f; params.origins[2][2] = -2.0f; params.origins[2][3] = 0.0f;
            params.dirs[2][0] = 0.0f;     params.dirs[2][1] = 0.0f;     params.dirs[2][2] = 1.0f;     params.dirs[2][3] = 0.0f;

            std::vector<uint32_t> out;
            runRayQueryDispatch(fix, ctx, tlas, params, out);

            UAA_CHECK_EQ(out[0], 1u, "Ray 0 must hit");
            UAA_CHECK_EQ(out[1], 10u, "Ray 0 must hit instanceID 10");
            UAA_CHECK_EQ(out[2], 0u, "Ray 0 primitiveIndex must be 0");
            UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Ray 0 hit distance mismatch");

            UAA_CHECK_EQ(out[4], 1u, "Ray 1 must hit");
            UAA_CHECK_EQ(out[5], 20u, "Ray 1 must hit instanceID 20");
            UAA_CHECK_EQ(out[6], 0u, "Ray 1 primitiveIndex must be 0");
            UAA_CHECK_NEAR(uintBitsToFloat(out[7]), 3.0f, 1e-3, "Ray 1 hit distance mismatch");

            UAA_CHECK_EQ(out[8], 0u, "Ray 2 must miss");
        }

        void testRtFromBuffer(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                false);

            const uint64_t blasAddress = blas.blas->getDeviceAddress();
            UAA_CHECK(blasAddress != 0, "BLAS device address must be non-zero for buildTopLevelAccelStructFromBuffer");

            nvrhi::rt::InstanceDesc inst;
            inst.setInstanceID(42)
                .setInstanceMask(0xFF)
                .setInstanceContributionToHitGroupIndex(0)
                .setFlags(nvrhi::rt::InstanceFlags::None)
                .setTransform(nvrhi::rt::c_IdentityTransform);
            inst.blasDeviceAddress = blasAddress;

            nvrhi::BufferDesc instBufDesc;
            instBufDesc.setByteSize(sizeof(nvrhi::rt::InstanceDesc))
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_rt_instance_buf");
            instBufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle instBuf = fix.device()->createBuffer(instBufDesc);
            UAA_CHECK(instBuf != nullptr, "Failed to create instance descriptor buffer");
            fix.uploadBuffer(instBuf, &inst, sizeof(inst));

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(1)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_rt_tlas_from_buf");
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned nullptr");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildTopLevelAccelStructFromBuffer(tlas, instBuf, 0, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            cmd->close();
            fix.submitAndWait(cmd);

            if (fix.features().supports(nvrhi::Feature::RayQuery))
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 1u;
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, tlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Ray must hit TLAS built from buffer");
                UAA_CHECK_EQ(out[1], 42u, "Instance ID mismatch on TLAS built from buffer");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Hit distance mismatch on TLAS built from buffer");
            }
            else
            {
                ctx.note("Feature::RayQuery not supported; verified buildTopLevelAccelStructFromBuffer build only");
            }
        }

        void testRtUpdateRefit(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::AllowUpdate,
                nvrhi::rt::GeometryFlags::Opaque,
                false);

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(1)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::AllowUpdate)
                .setDebugName("uaa_rt_tlas_update");
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS AllowUpdate) returned nullptr");

            nvrhi::rt::InstanceDesc inst;
            inst.setInstanceID(7).setInstanceMask(0xFF).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blas.blas);

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildTopLevelAccelStruct(tlas, &inst, 1, nvrhi::rt::AccelStructBuildFlags::AllowUpdate);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            if (fix.features().supports(nvrhi::Feature::RayQuery))
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 1u;
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;
                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, tlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Initial ray must hit before refit");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Initial rayT mismatch");
            }

            // Refit BLAS vertices slightly along +Z (+1.0) and refit TLAS instance to X = 100, Z = 2.
            const float updatedVertices[9] = {
                -1.0f, -1.0f, 1.0f,
                 3.0f, -1.0f, 1.0f,
                -1.0f,  3.0f, 1.0f
            };
            fix.uploadBuffer(blas.vertexBuffer, updatedVertices, sizeof(updatedVertices));

            const nvrhi::rt::AffineTransform movedTx = {
                1.0f, 0.0f, 0.0f, 100.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 2.0f
            };
            inst.setTransform(movedTx);

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                const auto updateFlags = nvrhi::rt::AccelStructBuildFlags::AllowUpdate | nvrhi::rt::AccelStructBuildFlags::PerformUpdate;
                cmd->buildBottomLevelAccelStruct(blas.blas, &blas.geomDesc, 1, updateFlags);
                cmd->buildTopLevelAccelStruct(tlas, &inst, 1, updateFlags);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            if (fix.features().supports(nvrhi::Feature::RayQuery))
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 2u;
                // Ray 0 at old origin (0, 0, -2) must now miss.
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;
                // Ray 1 at new origin (100, 0, -2) must hit z = 2 + 1 = 3 -> t = 5.0.
                params.origins[1][0] = 100.0f;
                params.origins[1][2] = -2.0f;
                params.dirs[1][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, tlas, params, out);
                UAA_CHECK_EQ(out[0], 0u, "Ray at old position must miss after refit");
                UAA_CHECK_EQ(out[4], 1u, "Ray at refitted position must hit");
                UAA_CHECK_EQ(out[5], 7u, "Refitted instance ID mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[7]), 5.0f, 1e-3, "Refitted rayT mismatch");
            }
        }

        void testRtCopyCompact(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingAccelStruct not supported");
            }

            TriangleBlasBundle srcBundle = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::AllowCompaction,
                nvrhi::rt::GeometryFlags::Opaque,
                false);

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->compactBottomLevelAccelStructs();
                cmd->close();
                fix.submitAndWait(cmd);
            }

            const nvrhi::MemoryRequirements blasMemReq = fix.device()->getAccelStructMemoryRequirements(srcBundle.blas);
            nvrhi::rt::AccelStructHandle traceBlas = srcBundle.blas;

            if (blasMemReq.size > 0)
            {
                nvrhi::rt::AccelStructDesc dstBlasDesc;
                dstBlasDesc.addBottomLevelGeometry(srcBundle.geomDesc)
                    .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::AllowCompaction)
                    .setDebugName("uaa_rt_dst_blas");
                nvrhi::rt::AccelStructHandle dstBlas = fix.device()->createAccelStruct(dstBlasDesc);
                UAA_CHECK(dstBlas != nullptr, "createAccelStruct(dstBlas) returned nullptr");

                nvrhi::CommandListHandle copyCmd = fix.device()->createCommandList();
                UAA_CHECK(copyCmd != nullptr, "createCommandList returned nullptr");
                copyCmd->open();
                copyCmd->copyRaytracingAccelerationStructure(dstBlas, srcBundle.blas);
                copyCmd->close();
                fix.submitAndWait(copyCmd);

                // Release source BLAS before tracing the cloned BLAS.
                srcBundle.blas = nullptr;
                fix.device()->runGarbageCollection();
                traceBlas = dstBlas;
            }
            else
            {
                ctx.note("RTXMU manages BLAS storage internally; verifying compacted BLAS plus TLAS clone");
            }

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(1)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_rt_copy_src_tlas");
            nvrhi::rt::AccelStructHandle srcTlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(srcTlas != nullptr, "createAccelStruct(srcTlas) returned nullptr");

            nvrhi::rt::InstanceDesc inst;
            inst.setInstanceID(77).setInstanceMask(0xFF).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(traceBlas);

            nvrhi::rt::AccelStructDesc dstTlasDesc = tlasDesc;
            dstTlasDesc.setDebugName("uaa_rt_copy_dst_tlas");
            nvrhi::rt::AccelStructHandle dstTlas = fix.device()->createAccelStruct(dstTlasDesc);
            UAA_CHECK(dstTlas != nullptr, "createAccelStruct(dstTlas) returned nullptr");

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildTopLevelAccelStruct(srcTlas, &inst, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->copyRaytracingAccelerationStructure(dstTlas, srcTlas);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            srcTlas = nullptr;
            fix.device()->runGarbageCollection();

            if (fix.features().supports(nvrhi::Feature::RayQuery))
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 1u;
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, dstTlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Cloned TLAS/BLAS must hit");
                UAA_CHECK_EQ(out[1], 77u, "Cloned TLAS instance ID mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Cloned TLAS rayT mismatch");
            }
        }

        void testRtPlacement(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct) ||
                !fix.features().supports(nvrhi::Feature::VirtualResources))
            {
                ctx.skip("Requires both Feature::RayTracingAccelStruct and Feature::VirtualResources");
            }

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                false);

            // Virtual BLAS placement when backend manages BLAS memory directly (non-RTXMU).
            nvrhi::rt::AccelStructDesc virtBlasDesc;
            virtBlasDesc.addBottomLevelGeometry(blas.geomDesc)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setIsVirtual(true)
                .setDebugName("uaa_rt_virtual_blas");
            nvrhi::rt::AccelStructHandle virtBlas = fix.device()->createAccelStruct(virtBlasDesc);
            UAA_CHECK(virtBlas != nullptr, "createAccelStruct(virtual BLAS) returned nullptr");

            const nvrhi::MemoryRequirements blasReq = fix.device()->getAccelStructMemoryRequirements(virtBlas);
            nvrhi::HeapHandle blasHeap;
            nvrhi::rt::IAccelStruct* blasForTlas = blas.blas;
            if (blasReq.size > 0)
            {
                nvrhi::HeapDesc blasHeapDesc;
                blasHeapDesc.setCapacity(blasReq.size + blasReq.alignment)
                    .setType(nvrhi::HeapType::DeviceLocal)
                    .setDebugName("uaa_rt_virt_blas_heap");
                blasHeap = fix.device()->createHeap(blasHeapDesc);
                UAA_CHECK(blasHeap != nullptr, "Failed to create heap for virtual BLAS");
                UAA_CHECK(fix.device()->bindAccelStructMemory(virtBlas, blasHeap, 0),
                    "bindAccelStructMemory(virtual BLAS) failed");

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildBottomLevelAccelStruct(virtBlas, &blas.geomDesc, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->close();
                fix.submitAndWait(cmd);
                blasForTlas = virtBlas;
            }
            else
            {
                ctx.note("Virtual BLAS bypassed because RTXMU manages BLAS storage");
            }

            nvrhi::rt::AccelStructDesc virtTlasDesc;
            virtTlasDesc.setTopLevelMaxInstances(1)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setIsVirtual(true)
                .setDebugName("uaa_rt_virtual_tlas");
            nvrhi::rt::AccelStructHandle virtTlas = fix.device()->createAccelStruct(virtTlasDesc);
            UAA_CHECK(virtTlas != nullptr, "createAccelStruct(virtual TLAS) returned nullptr");

            const nvrhi::MemoryRequirements tlasReq = fix.device()->getAccelStructMemoryRequirements(virtTlas);
            UAA_CHECK(tlasReq.size > 0, "Virtual TLAS memory requirements must be positive");

            nvrhi::HeapDesc tlasHeapDesc;
            tlasHeapDesc.setCapacity(tlasReq.size + tlasReq.alignment)
                .setType(nvrhi::HeapType::DeviceLocal)
                .setDebugName("uaa_rt_virt_tlas_heap");
            nvrhi::HeapHandle tlasHeap = fix.device()->createHeap(tlasHeapDesc);
            UAA_CHECK(tlasHeap != nullptr, "Failed to create heap for virtual TLAS");
            UAA_CHECK(fix.device()->bindAccelStructMemory(virtTlas, tlasHeap, 0),
                "bindAccelStructMemory(virtual TLAS) failed");

            nvrhi::rt::InstanceDesc inst;
            inst.setInstanceID(88).setInstanceMask(0xFF).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blasForTlas);

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildTopLevelAccelStruct(virtTlas, &inst, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            cmd->close();
            fix.submitAndWait(cmd);

            if (fix.features().supports(nvrhi::Feature::RayQuery))
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 1u;
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, virtTlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Virtual placed TLAS must hit");
                UAA_CHECK_EQ(out[1], 88u, "Virtual placed TLAS instance ID mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Virtual placed TLAS rayT mismatch");
            }
        }

        void testRtInlineQuery(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingAccelStruct) ||
                !fix.features().supports(nvrhi::Feature::RayQuery))
            {
                ctx.skip("Feature::RayQuery or Feature::RayTracingAccelStruct not supported");
            }

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                true);

            const nvrhi::rt::AffineTransform txFar = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 4.0f
            };

            nvrhi::rt::InstanceDesc instances[2];
            instances[0].setInstanceID(11).setInstanceMask(0x01).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blas.blas);
            instances[1].setInstanceID(22).setInstanceMask(0x02).setTransform(txFar).setBLAS(blas.blas);

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(2)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_rt_inline_tlas");
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned nullptr");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildTopLevelAccelStruct(tlas, instances, 2, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            cmd->close();
            fix.submitAndWait(cmd);

            // Pass 1: mask 0xFF tests near hit, far hit, and miss.
            {
                RayQueryParamsCB params{};
                params.rayMask = 0xFFu;
                params.rayCount = 3u;
                params.origins[0][2] = -2.0f; params.dirs[0][2] = 1.0f;
                params.origins[1][2] = 1.0f;  params.dirs[1][2] = 1.0f;
                params.origins[2][0] = 50.0f; params.origins[2][1] = 50.0f; params.origins[2][2] = -2.0f; params.dirs[2][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, tlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Mask 0xFF ray 0 must hit");
                UAA_CHECK_EQ(out[1], 11u, "Mask 0xFF ray 0 instanceID mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 2.0f, 1e-3, "Mask 0xFF ray 0 distance mismatch");

                UAA_CHECK_EQ(out[4], 1u, "Mask 0xFF ray 1 must hit");
                UAA_CHECK_EQ(out[5], 22u, "Mask 0xFF ray 1 instanceID mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[7]), 3.0f, 1e-3, "Mask 0xFF ray 1 distance mismatch");

                UAA_CHECK_EQ(out[8], 0u, "Mask 0xFF ray 2 must miss");
            }

            // Pass 2: mask 0x02 skips instance 0 (mask 0x01) and hits instance 1 at z = 4 (t = 6.0).
            {
                RayQueryParamsCB params{};
                params.rayMask = 0x02u;
                params.rayCount = 1u;
                params.origins[0][2] = -2.0f;
                params.dirs[0][2] = 1.0f;

                std::vector<uint32_t> out;
                runRayQueryDispatch(fix, ctx, tlas, params, out);
                UAA_CHECK_EQ(out[0], 1u, "Mask 0x02 ray must hit second instance");
                UAA_CHECK_EQ(out[1], 22u, "Mask 0x02 ray must skip instance 11 and hit 22");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 6.0f, 1e-3, "Mask 0x02 hit distance mismatch");
            }
        }

        struct RtPipelineBundle
        {
            nvrhi::ShaderLibraryHandle lib;
            nvrhi::BindingLayoutHandle layout;
            nvrhi::rt::PipelineHandle pipeline;
        };

        RtPipelineBundle createRtLibPipeline(TestFixture& fix, TestContext& ctx)
        {
            RtPipelineBundle b;
            const std::vector<uint8_t> libBytes = fix.loadShaderBinary("rt_lib");
            b.lib = fix.device()->createShaderLibrary(libBytes.data(), libBytes.size());
            UAA_CHECK(b.lib != nullptr, "createShaderLibrary(rt_lib) returned nullptr");

            nvrhi::ShaderHandle rgs = b.lib->getShader("RayGen", nvrhi::ShaderType::RayGeneration);
            nvrhi::ShaderHandle miss = b.lib->getShader("Miss", nvrhi::ShaderType::Miss);
            nvrhi::ShaderHandle chs = b.lib->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit);
            nvrhi::ShaderHandle ahs = b.lib->getShader("AnyHit", nvrhi::ShaderType::AnyHit);
            UAA_CHECK(rgs != nullptr && miss != nullptr && chs != nullptr && ahs != nullptr,
                "Failed to extract RayGen/Miss/ClosestHit/AnyHit shaders from rt_lib");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllRayTracing, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1));
            b.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(b.layout != nullptr, "Failed to create RT global binding layout");

            nvrhi::rt::PipelineDesc psoDesc;
            psoDesc.addShader(nvrhi::rt::PipelineShaderDesc().setExportName("RayGen").setShader(rgs))
                .addShader(nvrhi::rt::PipelineShaderDesc().setExportName("Miss").setShader(miss))
                .addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
                    .setExportName("HitGroup0")
                    .setClosestHitShader(chs)
                    .setAnyHitShader(ahs))
                .addBindingLayout(b.layout)
                .setMaxPayloadSize(16)
                .setMaxAttributeSize(8)
                .setMaxRecursionDepth(1);

            b.pipeline = fix.device()->createRayTracingPipeline(psoDesc);
            UAA_CHECK(b.pipeline != nullptr, "createRayTracingPipeline returned nullptr");
            return b;
        }

        void testRtPipeline(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingPipeline) ||
                !fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingPipeline not supported");
            }

            RtPipelineBundle rt = createRtLibPipeline(fix, ctx);

            // Non-opaque BLAS so AnyHit runs and ignores instanceID == 999 while hitting instanceID == 55 behind it.
            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::None,
                true);

            const nvrhi::rt::AffineTransform txBack = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 3.0f
            };

            nvrhi::rt::InstanceDesc instances[2];
            instances[0].setInstanceID(999).setInstanceMask(0xFF)
                .setFlags(nvrhi::rt::InstanceFlags::ForceNonOpaque)
                .setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blas.blas);
            instances[1].setInstanceID(55).setInstanceMask(0xFF)
                .setFlags(nvrhi::rt::InstanceFlags::ForceNonOpaque)
                .setTransform(txBack).setBLAS(blas.blas);

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(2)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_rt_pipe_tlas");
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned nullptr");

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildTopLevelAccelStruct(tlas, instances, 2, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            nvrhi::BufferDesc outDesc;
            outDesc.setByteSize(16 * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setDebugName("uaa_rt_pipe_out");
            outDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle outBuf = fix.device()->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "Failed to create RT pipeline output buffer");

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, outBuf));
            nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, rt.layout);
            UAA_CHECK(set != nullptr, "Failed to create RT pipeline binding set");

            nvrhi::rt::ShaderTableHandle table = rt.pipeline->createShaderTable();
            UAA_CHECK(table != nullptr, "createShaderTable returned nullptr");
            table->setRayGenerationShader("RayGen");
            table->addMissShader("Miss");
            table->addHitGroup("HitGroup0");

            // Hit dispatch (Out[7] == 0 uses default ray (0,0,-2) -> (0,0,1)).
            {
                std::vector<uint32_t> initWords(16, 0u);
                fix.uploadBuffer(outBuf, initWords.data(), initWords.size() * sizeof(uint32_t));

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                nvrhi::rt::State rtState;
                rtState.setShaderTable(table).addBindingSet(set);
                cmd->setRayTracingState(rtState);
                cmd->dispatchRays(nvrhi::rt::DispatchRaysArguments().setDimensions(1, 1, 1));
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> out(16, 0u);
                fix.readbackBuffer(outBuf, out.data(), out.size() * sizeof(uint32_t));
                UAA_CHECK_EQ(out[0], 1u, "RT pipeline ray must hit instance 55 after AnyHit ignores 999");
                UAA_CHECK_EQ(out[1], 55u, "RT pipeline instance ID mismatch");
                UAA_CHECK_EQ(out[2], 0u, "RT pipeline primitive index mismatch");
                UAA_CHECK_NEAR(uintBitsToFloat(out[3]), 5.0f, 1e-3, "RT pipeline rayT mismatch");
            }

            // Miss dispatch (Out[7] == 1 reads custom ray origin/dir from Out[4..10]).
            {
                std::vector<uint32_t> initWords(16, 0u);
                initWords[4] = floatToUintBits(100.0f);
                initWords[5] = floatToUintBits(100.0f);
                initWords[6] = floatToUintBits(-2.0f);
                initWords[7] = 1u;
                initWords[8] = floatToUintBits(0.0f);
                initWords[9] = floatToUintBits(0.0f);
                initWords[10] = floatToUintBits(1.0f);
                fix.uploadBuffer(outBuf, initWords.data(), initWords.size() * sizeof(uint32_t));

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                nvrhi::rt::State rtState;
                rtState.setShaderTable(table).addBindingSet(set);
                cmd->setRayTracingState(rtState);
                cmd->dispatchRays(nvrhi::rt::DispatchRaysArguments().setDimensions(1, 1, 1));
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint32_t> out(16, 0u);
                fix.readbackBuffer(outBuf, out.data(), out.size() * sizeof(uint32_t));
                UAA_CHECK_EQ(out[0], 0u, "RT pipeline miss ray must report hit == 0");
            }
        }

        void testRtShaderTable(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingPipeline) ||
                !fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingPipeline not supported");
            }

            RtPipelineBundle rt = createRtLibPipeline(fix, ctx);

            nvrhi::rt::ShaderTableHandle table =
                rt.pipeline->createShaderTable(nvrhi::rt::ShaderTableDesc().enableCaching(8).setDebugName("uaa_rt_sbt"));
            UAA_CHECK(table != nullptr, "createShaderTable(cached) returned nullptr");
            UAA_CHECK(table->getPipeline() == rt.pipeline.Get(), "getPipeline() must match owning pipeline");
            UAA_CHECK_EQ(table->getNumEntries(), 1u, "Initial shader table should have 1 entry (RayGen slot)");

            table->setRayGenerationShader("RayGen");
            const int missIdx0 = table->addMissShader("Miss");
            const int missIdx1 = table->addMissShader("Miss");
            const int hitIdx0 = table->addHitGroup("HitGroup0");
            UAA_CHECK_EQ(missIdx0, 0, "First miss shader index must be 0");
            UAA_CHECK_EQ(missIdx1, 1, "Second miss shader index must be 1");
            UAA_CHECK_EQ(hitIdx0, 0, "First hit group index must be 0");
            UAA_CHECK_EQ(table->getNumEntries(), 4u, "Expected 4 SBT entries (1 RGS + 2 Miss + 1 HitGroup)");

            // Missing callable export must fail gracefully without adding an entry.
            {
                ExpectErrorsGuard expectErr(fix);
                const int badCallable = table->addCallableShader("MissingCallable");
                UAA_CHECK_EQ(badCallable, -1, "Missing callable shader must return -1");
                UAA_CHECK_EQ(table->getNumEntries(), 4u, "Failed addCallableShader must not alter entry count");
            }

            table->clearMissShaders();
            UAA_CHECK_EQ(table->getNumEntries(), 2u, "Entry count after clearMissShaders must be 2");
            table->clearHitShaders();
            UAA_CHECK_EQ(table->getNumEntries(), 1u, "Entry count after clearHitShaders must be 1");
            table->clearCallableShaders();
            UAA_CHECK_EQ(table->getNumEntries(), 1u, "Entry count after clearCallableShaders must remain 1");

            // Re-populate valid entries and drop caller's pipeline handle to verify ShaderTable retains it.
            table->addMissShader("Miss");
            table->addHitGroup("HitGroup0");
            UAA_CHECK_EQ(table->getNumEntries(), 3u, "Repopulated SBT must have 3 entries");

            rt.pipeline = nullptr;
            fix.device()->runGarbageCollection();
            UAA_CHECK(table->getPipeline() != nullptr, "ShaderTable must retain its owning pipeline");

            TriangleBlasBundle blas = createAndBuildUnitTriangleBlas(
                fix, ctx,
                nvrhi::rt::AccelStructBuildFlags::PreferFastTrace,
                nvrhi::rt::GeometryFlags::Opaque,
                true);

            nvrhi::rt::AccelStructDesc tlasDesc;
            tlasDesc.setTopLevelMaxInstances(1)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            nvrhi::rt::AccelStructHandle tlas = fix.device()->createAccelStruct(tlasDesc);
            UAA_CHECK(tlas != nullptr, "createAccelStruct(TLAS) returned nullptr");

            nvrhi::rt::InstanceDesc inst;
            inst.setInstanceID(33).setInstanceMask(0xFF).setTransform(nvrhi::rt::c_IdentityTransform).setBLAS(blas.blas);

            nvrhi::BufferDesc outDesc;
            outDesc.setByteSize(16 * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setDebugName("uaa_rt_sbt_out");
            outDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle outBuf = fix.device()->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "Failed to create SBT test output buffer");
            std::vector<uint32_t> zeroes(16, 0u);
            fix.uploadBuffer(outBuf, zeroes.data(), zeroes.size() * sizeof(uint32_t));

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, outBuf));
            nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, rt.layout);
            UAA_CHECK(set != nullptr, "Failed to create SBT test binding set");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildTopLevelAccelStruct(tlas, &inst, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            nvrhi::rt::State rtState;
            rtState.setShaderTable(table).addBindingSet(set);
            cmd->setRayTracingState(rtState);
            cmd->dispatchRays(nvrhi::rt::DispatchRaysArguments().setDimensions(1, 1, 1));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> out(16, 0u);
            fix.readbackBuffer(outBuf, out.data(), out.size() * sizeof(uint32_t));
            UAA_CHECK_EQ(out[0], 1u, "Cached ShaderTable dispatch after releasing pipeline handle must hit");
            UAA_CHECK_EQ(out[1], 33u, "Cached ShaderTable instanceID mismatch");
        }

        void testRtUnsupported(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasAs = fix.features().supports(nvrhi::Feature::RayTracingAccelStruct);
            const bool hasRq = fix.features().supports(nvrhi::Feature::RayQuery);
            const bool hasRtPipe = fix.features().supports(nvrhi::Feature::RayTracingPipeline);

            std::ostringstream oss;
            oss << "RayTracingAccelStruct=" << (hasAs ? "true" : "false")
                << ", RayQuery=" << (hasRq ? "true" : "false")
                << ", RayTracingPipeline=" << (hasRtPipe ? "true" : "false");
            ctx.note(oss.str());

            if (fix.api() == nvrhi::GraphicsAPI::METAL)
            {
                if (!hasAs)
                {
                    nvrhi::rt::AccelStructDesc asDesc;
                    asDesc.setTopLevelMaxInstances(1);
                    UAA_CHECK(fix.device()->createAccelStruct(asDesc) == nullptr,
                        "Unsupported createAccelStruct on Metal must return nullptr");
                }
                if (!hasRtPipe)
                {
                    ExpectErrorsGuard expectErr(fix);
                    nvrhi::rt::PipelineDesc psoDesc;
                    UAA_CHECK(fix.device()->createRayTracingPipeline(psoDesc) == nullptr,
                        "Unsupported createRayTracingPipeline on Metal must return nullptr");
                }
            }
            else if (!hasAs)
            {
                if (fix.api() == nvrhi::GraphicsAPI::VULKAN)
                {
                    ctx.note("Skipped queryTopLevelAccelStructPrebuildInfo on Vulkan when RayTracingAccelStruct is false because backend triggers utils::NotSupported() Debug assert");
                }
                else if (fix.api() != nvrhi::GraphicsAPI::D3D11)
                {
                    nvrhi::rt::AccelStructPrebuildInfo sentinel{ 0x9999ull, 0x8888ull, 0x7777ull };
                    nvrhi::rt::AccelStructDesc asDesc;
                    asDesc.setTopLevelMaxInstances(1);
                    UAA_CHECK(!fix.device()->queryTopLevelAccelStructPrebuildInfo(asDesc, 1, sentinel),
                        "Prebuild query when RayTracingAccelStruct is false must return false");
                    UAA_CHECK_EQ(sentinel.resultBytes, 0x9999ull, "Failed prebuild must preserve sentinel");
                }
            }
        }
    } // namespace

    UAA_REGISTER_TEST("RT.Prebuild", "optional", testRtPrebuild);
    UAA_REGISTER_TEST("RT.BottomLevel", "optional", testRtBottomLevel);
    UAA_REGISTER_TEST("RT.TopLevel", "optional", testRtTopLevel);
    UAA_REGISTER_TEST("RT.FromBuffer", "optional", testRtFromBuffer);
    UAA_REGISTER_TEST("RT.UpdateRefit", "optional", testRtUpdateRefit);
    UAA_REGISTER_TEST("RT.CopyCompact", "optional", testRtCopyCompact);
    UAA_REGISTER_TEST("RT.Placement", "optional", testRtPlacement);
    UAA_REGISTER_TEST("RT.InlineQuery", "optional", testRtInlineQuery);
    UAA_REGISTER_TEST("RT.Pipeline", "optional", testRtPipeline);
    UAA_REGISTER_TEST("RT.ShaderTable", "optional", testRtShaderTable);
    UAA_REGISTER_TEST("RT.Unsupported", "optional", testRtUnsupported);
} // namespace nvrhi::uaa
