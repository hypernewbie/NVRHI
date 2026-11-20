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

#include <cstdint>
#include <sstream>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct OmmTriangleDesc
        {
            uint32_t dataOffset;
            uint16_t subdivisionLevel;
            uint16_t format;
        };

        void testRtOpacityMicromap(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::RayTracingOpacityMicromap) ||
                !fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("Feature::RayTracingOpacityMicromap not supported");
            }

            // Subdivision level 1 -> 4 micro-triangles, 1 bit per micro-triangle for 2-state OMM (all opaque = 0x0F).
            const uint32_t rawOmmData = 0x0000000Fu;
            nvrhi::BufferDesc inputBufDesc;
            inputBufDesc.setByteSize(64)
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_omm_input");
            inputBufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::OpacityMicromapBuildInput);
            nvrhi::BufferHandle inputBuf = fix.device()->createBuffer(inputBufDesc);
            UAA_CHECK(inputBuf != nullptr, "Failed to create OMM input buffer");
            fix.uploadBuffer(inputBuf, &rawOmmData, sizeof(rawOmmData));

            const OmmTriangleDesc triDesc{ 0u, 1u, static_cast<uint16_t>(nvrhi::rt::OpacityMicromapFormat::OC1_2_State) };
            nvrhi::BufferDesc perOmmBufDesc;
            perOmmBufDesc.setByteSize(64)
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_omm_descs");
            perOmmBufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::OpacityMicromapBuildInput);
            nvrhi::BufferHandle perOmmBuf = fix.device()->createBuffer(perOmmBufDesc);
            UAA_CHECK(perOmmBuf != nullptr, "Failed to create OMM per-triangle desc buffer");
            fix.uploadBuffer(perOmmBuf, &triDesc, sizeof(triDesc));

            nvrhi::rt::OpacityMicromapUsageCount usage{};
            usage.count = 1;
            usage.subdivisionLevel = 1;
            usage.format = nvrhi::rt::OpacityMicromapFormat::OC1_2_State;

            nvrhi::rt::OpacityMicromapDesc ommDesc;
            ommDesc.setDebugName("uaa_omm")
                .setFlags(nvrhi::rt::OpacityMicromapBuildFlags::FastTrace)
                .setCounts({ usage })
                .setInputBuffer(inputBuf)
                .setPerOmmDescs(perOmmBuf);

            nvrhi::rt::OpacityMicromapHandle omm = fix.device()->createOpacityMicromap(ommDesc);
            UAA_CHECK(omm != nullptr, "createOpacityMicromap returned nullptr");

            const float vertices[9] = {
                -1.0f, -1.0f, 0.0f,
                 3.0f, -1.0f, 0.0f,
                -1.0f,  3.0f, 0.0f
            };
            nvrhi::BufferDesc vbDesc;
            vbDesc.setByteSize(sizeof(vertices))
                .setIsVertexBuffer(true)
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_omm_vb");
            vbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle vb = fix.device()->createBuffer(vbDesc);
            UAA_CHECK(vb != nullptr, "Failed to create OMM triangle VB");
            fix.uploadBuffer(vb, vertices, sizeof(vertices));

            const uint16_t ommIndex = 0u;
            nvrhi::BufferDesc ommIdxDesc;
            ommIdxDesc.setByteSize(16)
                .setIsAccelStructBuildInput(true)
                .setDebugName("uaa_omm_idx");
            ommIdxDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle ommIdxBuf = fix.device()->createBuffer(ommIdxDesc);
            UAA_CHECK(ommIdxBuf != nullptr, "Failed to create OMM index buffer");
            fix.uploadBuffer(ommIdxBuf, &ommIndex, sizeof(ommIndex));

            nvrhi::rt::GeometryTriangles tri;
            tri.setVertexBuffer(vb)
                .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
                .setVertexCount(3)
                .setVertexStride(sizeof(float) * 3)
                .setOpacityMicromap(omm)
                .setOmmIndexBuffer(ommIdxBuf)
                .setOmmIndexFormat(nvrhi::Format::R16_UINT)
                .setPOmmUsageCounts(&usage)
                .setNumOmmUsageCounts(1);

            nvrhi::rt::GeometryDesc geomDesc;
            geomDesc.setTriangles(tri).setFlags(nvrhi::rt::GeometryFlags::None);

            nvrhi::rt::AccelStructDesc blasDesc;
            blasDesc.addBottomLevelGeometry(geomDesc)
                .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace)
                .setDebugName("uaa_omm_blas");
            nvrhi::rt::AccelStructHandle blas = fix.device()->createAccelStruct(blasDesc);
            UAA_CHECK(blas != nullptr, "createAccelStruct(OMM BLAS) returned nullptr");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->buildOpacityMicromap(omm, ommDesc);
            cmd->buildBottomLevelAccelStruct(blas, &geomDesc, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
            cmd->close();
            fix.submitAndWait(cmd);
        }

        void testRtAdvancedGeometry(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasSpheres = fix.features().supports(nvrhi::Feature::Spheres);
            const bool hasLss = fix.features().supports(nvrhi::Feature::LinearSweptSpheres);
            const bool hasPosFetch = fix.features().supports(nvrhi::Feature::RayTracingPositionFetch);

            std::ostringstream oss;
            oss << "Spheres=" << (hasSpheres ? "true" : "false")
                << ", LinearSweptSpheres=" << (hasLss ? "true" : "false")
                << ", RayTracingPositionFetch=" << (hasPosFetch ? "true" : "false");
            ctx.note(oss.str());

            if (!hasSpheres && !hasLss && !hasPosFetch)
            {
                ctx.skip("Spheres, LinearSweptSpheres, and RayTracingPositionFetch not supported");
            }

            if (hasSpheres && fix.api() != nvrhi::GraphicsAPI::VULKAN)
            {
                const float sphereVert[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

                nvrhi::BufferDesc sphereBufDesc;
                sphereBufDesc.setByteSize(sizeof(sphereVert))
                    .setIsVertexBuffer(true)
                    .setIsAccelStructBuildInput(true)
                    .setDebugName("uaa_sphere_vb");
                sphereBufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
                nvrhi::BufferHandle sphereBuf = fix.device()->createBuffer(sphereBufDesc);
                UAA_CHECK(sphereBuf != nullptr, "Failed to create sphere vertex buffer");
                fix.uploadBuffer(sphereBuf, sphereVert, sizeof(sphereVert));

                nvrhi::rt::GeometrySpheres spheres;
                spheres.setVertexBuffer(sphereBuf)
                    .setVertexPositionFormat(nvrhi::Format::RGB32_FLOAT)
                    .setVertexPositionOffset(0)
                    .setVertexPositionStride(sizeof(float) * 4)
                    .setVertexRadiusFormat(nvrhi::Format::R32_FLOAT)
                    .setVertexRadiusOffset(sizeof(float) * 3)
                    .setVertexRadiusStride(sizeof(float) * 4)
                    .setVertexCount(1);

                nvrhi::rt::GeometryDesc geomDesc;
                geomDesc.setSpheres(spheres).setFlags(nvrhi::rt::GeometryFlags::Opaque);
                nvrhi::rt::AccelStructDesc blasDesc;
                blasDesc.addBottomLevelGeometry(geomDesc)
                    .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                nvrhi::rt::AccelStructHandle blas = fix.device()->createAccelStruct(blasDesc);
                UAA_CHECK(blas != nullptr, "createAccelStruct(Spheres) returned nullptr");

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildBottomLevelAccelStruct(blas, &geomDesc, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            if (hasLss)
            {
                const float lssVerts[8] = {
                    -1.0f, 0.0f, 0.0f, 0.5f,
                     1.0f, 0.0f, 0.0f, 0.5f
                };
                nvrhi::BufferDesc vbDesc;
                vbDesc.setByteSize(sizeof(lssVerts))
                    .setIsVertexBuffer(true)
                    .setIsAccelStructBuildInput(true)
                    .setDebugName("uaa_lss_vb");
                vbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
                nvrhi::BufferHandle vb = fix.device()->createBuffer(vbDesc);
                UAA_CHECK(vb != nullptr, "Failed to create LSS vertex buffer");
                fix.uploadBuffer(vb, lssVerts, sizeof(lssVerts));

                nvrhi::rt::GeometryLss lss;
                lss.setVertexBuffer(vb)
                    .setVertexPositionFormat(nvrhi::Format::RGB32_FLOAT)
                    .setVertexPositionOffset(0)
                    .setVertexPositionStride(sizeof(float) * 4)
                    .setVertexRadiusFormat(nvrhi::Format::R32_FLOAT)
                    .setVertexRadiusOffset(sizeof(float) * 3)
                    .setVertexRadiusStride(sizeof(float) * 4)
                    .setPrimitiveCount(1)
                    .setVertexCount(2)
                    .setPrimitiveFormat(nvrhi::rt::GeometryLssPrimitiveFormat::List)
                    .setEndcapMode(nvrhi::rt::GeometryLssEndcapMode::Chained);

                nvrhi::rt::GeometryDesc geomDesc;
                geomDesc.setLss(lss).setFlags(nvrhi::rt::GeometryFlags::Opaque);
                nvrhi::rt::AccelStructDesc blasDesc;
                blasDesc.addBottomLevelGeometry(geomDesc)
                    .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                nvrhi::rt::AccelStructHandle blas = fix.device()->createAccelStruct(blasDesc);
                UAA_CHECK(blas != nullptr, "createAccelStruct(LSS) returned nullptr");

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildBottomLevelAccelStruct(blas, &geomDesc, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            if (hasPosFetch && fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                const float vertices[9] = { -1.0f, -1.0f, 0.0f, 3.0f, -1.0f, 0.0f, -1.0f, 3.0f, 0.0f };
                nvrhi::BufferDesc vbDesc;
                vbDesc.setByteSize(sizeof(vertices))
                    .setIsVertexBuffer(true)
                    .setIsAccelStructBuildInput(true)
                    .setDebugName("uaa_posfetch_vb");
                vbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
                nvrhi::BufferHandle vb = fix.device()->createBuffer(vbDesc);
                UAA_CHECK(vb != nullptr, "Failed to create PositionFetch VB");
                fix.uploadBuffer(vb, vertices, sizeof(vertices));

                nvrhi::rt::GeometryTriangles tri;
                tri.setVertexBuffer(vb)
                    .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
                    .setVertexCount(3)
                    .setVertexStride(sizeof(float) * 3);

                nvrhi::rt::GeometryDesc geomDesc;
                geomDesc.setTriangles(tri).setFlags(nvrhi::rt::GeometryFlags::Opaque);
                nvrhi::rt::AccelStructDesc blasDesc;
                blasDesc.addBottomLevelGeometry(geomDesc)
                    .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                nvrhi::rt::AccelStructHandle blas = fix.device()->createAccelStruct(blasDesc);
                UAA_CHECK(blas != nullptr, "createAccelStruct(PositionFetch BLAS) returned nullptr");

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->buildBottomLevelAccelStruct(blas, &geomDesc, 1, nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);
                cmd->close();
                fix.submitAndWait(cmd);
            }
        }

        void testRtClusters(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            nvrhi::RayTracingClustersFeatureInfo clusterInfo{};
            clusterInfo.api = nvrhi::ClusterApi::None;
            const bool supported = fix.device()->queryFeatureSupport(
                nvrhi::Feature::RayTracingClusters, &clusterInfo, sizeof(clusterInfo));

            if (!supported)
            {
                UAA_CHECK_EQ(static_cast<uint32_t>(clusterInfo.api),
                    static_cast<uint32_t>(nvrhi::ClusterApi::None),
                    "Unsupported RayTracingClusters must report ClusterApi::None");
                UAA_CHECK_EQ(static_cast<uint32_t>(fix.features().clusterInfo.api),
                    static_cast<uint32_t>(nvrhi::ClusterApi::None),
                    "FeatureInventory clusterInfo must match ClusterApi::None");
                ctx.skip("Feature::RayTracingClusters not supported");
            }

            UAA_CHECK_NE(static_cast<uint32_t>(clusterInfo.api),
                static_cast<uint32_t>(nvrhi::ClusterApi::None),
                "Supported RayTracingClusters must report Nv or Dxr ClusterApi");

            nvrhi::rt::cluster::OperationParams params{};
            params.maxArgCount = 1;
            params.type = nvrhi::rt::cluster::OperationType::ClasBuild;
            params.mode = nvrhi::rt::cluster::OperationMode::ImplicitDestinations;
            params.flags = nvrhi::rt::cluster::OperationFlags::FastTrace;
            params.clas.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            params.clas.maxGeometryIndex = 0;
            params.clas.maxUniqueGeometryCount = 1;
            params.clas.maxTriangleCount = 1;
            params.clas.maxVertexCount = 3;
            params.clas.maxTotalTriangleCount = 1;
            params.clas.maxTotalVertexCount = 3;
            params.clas.indexFormat = nvrhi::rt::cluster::OperationIndexFormat::IndexFormat32bit;

            const nvrhi::rt::cluster::OperationSizeInfo sizeInfo =
                fix.device()->getClusterOperationSizeInfo(params);
            UAA_CHECK(sizeInfo.resultMaxSizeInBytes > 0, "Cluster operation resultMaxSizeInBytes must be positive");
            UAA_CHECK(sizeInfo.scratchSizeInBytes > 0, "Cluster operation scratchSizeInBytes must be positive");

            std::ostringstream oss;
            oss << "ClusterApi=" << (clusterInfo.api == nvrhi::ClusterApi::Nv ? "Nv" : "Dxr")
                << ", resultMaxBytes=" << sizeInfo.resultMaxSizeInBytes
                << ", scratchBytes=" << sizeInfo.scratchSizeInBytes;
            ctx.note(oss.str());

            const float vertices[9] = { -1.0f, -1.0f, 0.0f, 3.0f, -1.0f, 0.0f, -1.0f, 3.0f, 0.0f };
            const uint32_t indices[3] = { 0u, 1u, 2u };
            const uint32_t argCount = 1u;

            nvrhi::BufferDesc vbDesc;
            vbDesc.setByteSize(sizeof(vertices)).setIsAccelStructBuildInput(true).setDebugName("uaa_clas_vb");
            vbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle vb = fix.device()->createBuffer(vbDesc);
            UAA_CHECK(vb != nullptr, "Failed to create CLAS VB");
            fix.uploadBuffer(vb, vertices, sizeof(vertices));

            nvrhi::BufferDesc ibDesc;
            ibDesc.setByteSize(sizeof(indices)).setIsAccelStructBuildInput(true).setDebugName("uaa_clas_ib");
            ibDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle ib = fix.device()->createBuffer(ibDesc);
            UAA_CHECK(ib != nullptr, "Failed to create CLAS IB");
            fix.uploadBuffer(ib, indices, sizeof(indices));

            nvrhi::BufferDesc countDesc;
            countDesc.setByteSize(16).setIsAccelStructBuildInput(true).setDebugName("uaa_clas_count");
            countDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle countBuf = fix.device()->createBuffer(countDesc);
            UAA_CHECK(countBuf != nullptr, "Failed to create CLAS count buffer");
            fix.uploadBuffer(countBuf, &argCount, sizeof(argCount));

            nvrhi::BufferDesc addrDesc;
            addrDesc.setByteSize(sizeof(uint64_t)).setCanHaveUAVs(true).setDebugName("uaa_clas_addrs");
            addrDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle addrBuf = fix.device()->createBuffer(addrDesc);
            UAA_CHECK(addrBuf != nullptr, "Failed to create CLAS addresses buffer");

            nvrhi::BufferDesc sizesDesc;
            sizesDesc.setByteSize(sizeof(uint32_t)).setCanHaveUAVs(true).setDebugName("uaa_clas_sizes");
            sizesDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle sizesBuf = fix.device()->createBuffer(sizesDesc);
            UAA_CHECK(sizesBuf != nullptr, "Failed to create CLAS sizes buffer");

            nvrhi::BufferDesc outAsDesc;
            outAsDesc.setByteSize(sizeInfo.resultMaxSizeInBytes)
                .setCanHaveUAVs(true)
                .setIsAccelStructStorage(true)
                .setDebugName("uaa_clas_out");
            outAsDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructWrite);
            nvrhi::BufferHandle outAsBuf = fix.device()->createBuffer(outAsDesc);
            UAA_CHECK(outAsBuf != nullptr, "Failed to create CLAS output storage buffer");

            nvrhi::BufferDesc argsDesc;
            argsDesc.setByteSize(256).setIsAccelStructBuildInput(true).setDebugName("uaa_clas_args");
            argsDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::AccelStructBuildInput);
            nvrhi::BufferHandle argsBuf = fix.device()->createBuffer(argsDesc);
            UAA_CHECK(argsBuf != nullptr, "Failed to create CLAS indirect args buffer");

            if (clusterInfo.api == nvrhi::ClusterApi::Nv)
            {
                nvrhi::rt::cluster::nv::IndirectTriangleClasArgs nvArgs{};
                nvArgs.clusterId = 0;
                nvArgs.vertexCount = 3;
                nvArgs.triangleCount = 1;
                nvArgs.vertexBufferStride = sizeof(float) * 3;
                nvArgs.indexBufferStride = sizeof(uint32_t);
                nvArgs.indexFormat = 4;
                nvArgs.indexBuffer = ib->getGpuVirtualAddress();
                nvArgs.vertexBuffer = vb->getGpuVirtualAddress();
                fix.uploadBuffer(argsBuf, &nvArgs, sizeof(nvArgs));
            }
            else
            {
                nvrhi::rt::cluster::dxr::IndirectTriangleClasArgs dxrArgs{};
                dxrArgs.clusterId = 0;
                dxrArgs.vertexCount = 3;
                dxrArgs.triangleCount = 1;
                dxrArgs.vertexBufferStride = sizeof(float) * 3;
                dxrArgs.indexBufferStride = sizeof(uint32_t);
                dxrArgs.indexBuffer = ib->getGpuVirtualAddress();
                dxrArgs.vertexBuffer = vb->getGpuVirtualAddress();
                fix.uploadBuffer(argsBuf, &dxrArgs, sizeof(dxrArgs));
            }

            nvrhi::rt::cluster::OperationDesc opDesc{};
            opDesc.params = params;
            opDesc.scratchSizeInBytes = sizeInfo.scratchSizeInBytes;
            opDesc.inIndirectArgCountBuffer = countBuf;
            opDesc.inIndirectArgsBuffer = argsBuf;
            opDesc.inOutAddressesBuffer = addrBuf;
            opDesc.outSizesBuffer = sizesBuf;
            opDesc.outAccelerationStructuresBuffer = outAsBuf;

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->executeMultiIndirectClusterOperation(opDesc);
            cmd->close();
            fix.submitAndWait(cmd);

            uint32_t builtSize = 0;
            fix.readbackBuffer(sizesBuf, &builtSize, sizeof(builtSize));
            UAA_CHECK(builtSize > 0, "Built CLAS size must be positive");
        }

        void testRtReordering(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::ShaderExecutionReordering))
            {
                ctx.skip("Feature::ShaderExecutionReordering not supported");
            }

            ctx.note("Feature::ShaderExecutionReordering advertised as supported");

            if (!fix.features().supports(nvrhi::Feature::RayTracingPipeline) ||
                !fix.features().supports(nvrhi::Feature::RayTracingAccelStruct))
            {
                ctx.skip("SER advertised without RayTracingPipeline/AccelStruct");
            }

            const std::vector<uint8_t> libBytes = fix.loadShaderBinary("rt_lib");
            nvrhi::ShaderLibraryHandle lib = fix.device()->createShaderLibrary(libBytes.data(), libBytes.size());
            UAA_CHECK(lib != nullptr, "createShaderLibrary(rt_lib) returned nullptr");

            nvrhi::ShaderHandle rgs = lib->getShader("RayGen", nvrhi::ShaderType::RayGeneration);
            nvrhi::ShaderHandle miss = lib->getShader("Miss", nvrhi::ShaderType::Miss);
            nvrhi::ShaderHandle chs = lib->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit);
            UAA_CHECK(rgs != nullptr && miss != nullptr && chs != nullptr, "Failed to load rt_lib shaders");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllRayTracing, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::RayTracingAccelStruct(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1));
            nvrhi::BindingLayoutHandle layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create RT layout");

            nvrhi::rt::PipelineDesc psoDesc;
            psoDesc.addShader(nvrhi::rt::PipelineShaderDesc().setExportName("RayGen").setShader(rgs))
                .addShader(nvrhi::rt::PipelineShaderDesc().setExportName("Miss").setShader(miss))
                .addHitGroup(nvrhi::rt::PipelineHitGroupDesc().setExportName("HitGroup0").setClosestHitShader(chs))
                .addBindingLayout(layout)
                .setMaxPayloadSize(16)
                .setMaxAttributeSize(8)
                .setMaxRecursionDepth(1);
            nvrhi::rt::PipelineHandle pso = fix.device()->createRayTracingPipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "createRayTracingPipeline returned nullptr under SER device");
        }
    } // namespace

    UAA_REGISTER_TEST("RT.OpacityMicromap", "optional", testRtOpacityMicromap);
    UAA_REGISTER_TEST("RT.AdvancedGeometry", "optional", testRtAdvancedGeometry);
    UAA_REGISTER_TEST("RT.Clusters", "optional", testRtClusters);
    UAA_REGISTER_TEST("RT.Reordering", "optional", testRtReordering);
} // namespace nvrhi::uaa
