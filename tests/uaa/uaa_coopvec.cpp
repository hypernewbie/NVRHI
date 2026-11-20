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
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct FullscreenPushConstants
        {
            float color[4];
            float depth;
            float scale;
            float offset[2];
        };

        void testCoopCapabilities(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasInf = fix.features().supports(nvrhi::Feature::CooperativeVectorInferencing);
            const bool hasTrain = fix.features().supports(nvrhi::Feature::CooperativeVectorTraining);

            const nvrhi::coopvec::DeviceFeatures feats = fix.device()->queryCoopVecFeatures();

            const nvrhi::coopvec::MatMulFormatCombo fp16Combo{
                nvrhi::coopvec::DataType::Float16,
                nvrhi::coopvec::DataType::Float16,
                nvrhi::coopvec::DataType::Float16,
                nvrhi::coopvec::DataType::Float16,
                nvrhi::coopvec::DataType::Float16,
                false
            };
            const nvrhi::coopvec::MatMulFormatCombo int8Combo{
                nvrhi::coopvec::DataType::SInt8,
                nvrhi::coopvec::DataType::SInt8,
                nvrhi::coopvec::DataType::SInt8,
                nvrhi::coopvec::DataType::SInt32,
                nvrhi::coopvec::DataType::SInt32,
                false
            };
            const nvrhi::coopvec::MatMulFormatCombo invalidCombo{
                nvrhi::coopvec::DataType::Float64,
                nvrhi::coopvec::DataType::Float64,
                nvrhi::coopvec::DataType::Float64,
                nvrhi::coopvec::DataType::Float64,
                nvrhi::coopvec::DataType::Float64,
                false
            };

            const nvrhi::coopvec::MatMulFormatSupport fp16Sup =
                fix.device()->queryCoopVecMatMulFormatSupport(fp16Combo);
            const nvrhi::coopvec::MatMulFormatSupport int8Sup =
                fix.device()->queryCoopVecMatMulFormatSupport(int8Combo);
            const nvrhi::coopvec::MatMulFormatSupport invalidSup =
                fix.device()->queryCoopVecMatMulFormatSupport(invalidCombo);

            const nvrhi::coopvec::TrainingFormatSupport trainFp16 =
                fix.device()->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::Float16);
            const nvrhi::coopvec::TrainingFormatSupport trainFp32 =
                fix.device()->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::Float32);
            const nvrhi::coopvec::TrainingFormatSupport trainInvalid =
                fix.device()->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::UInt8);

            UAA_CHECK(!invalidSup.supported, "Float64 CoopVec matmul combination must not be supported");
            UAA_CHECK(!trainInvalid.bufferTrainingSupported, "UInt8 CoopVec training accumulation must not be supported");

            if (!hasInf)
            {
                UAA_CHECK(feats.matMulFormats.empty(), "matMulFormats must be empty when inferencing unsupported");
                UAA_CHECK(!fp16Sup.supported, "FP16 matmul must report supported == false when inferencing unsupported");
                UAA_CHECK(!int8Sup.supported, "SInt8 matmul must report supported == false when inferencing unsupported");
            }

            if (!hasTrain)
            {
                UAA_CHECK(!feats.trainingFloat16, "trainingFloat16 must be false when training unsupported");
                UAA_CHECK(!feats.trainingFloat32, "trainingFloat32 must be false when training unsupported");
                UAA_CHECK(!trainFp16.bufferTrainingSupported, "FP16 training must be false when unsupported");
                UAA_CHECK(!trainFp32.bufferTrainingSupported, "FP32 training must be false when unsupported");
            }

            std::ostringstream oss;
            oss << "Inferencing=" << (hasInf ? "true" : "false")
                << ", Training=" << (hasTrain ? "true" : "false")
                << ", matMulCombos=" << feats.matMulFormats.size();
            ctx.note(oss.str());
        }

        void testCoopMatrixSize(TestContext& ctx)
        {
            struct DataTypeExpectation
            {
                nvrhi::coopvec::DataType type;
                size_t expectedBytes;
            };

            const DataTypeExpectation kTypes[] = {
                { nvrhi::coopvec::DataType::UInt8, 1 },
                { nvrhi::coopvec::DataType::SInt8, 1 },
                { nvrhi::coopvec::DataType::UInt8Packed, 1 },
                { nvrhi::coopvec::DataType::SInt8Packed, 1 },
                { nvrhi::coopvec::DataType::UInt16, 2 },
                { nvrhi::coopvec::DataType::SInt16, 2 },
                { nvrhi::coopvec::DataType::UInt32, 4 },
                { nvrhi::coopvec::DataType::SInt32, 4 },
                { nvrhi::coopvec::DataType::UInt64, 8 },
                { nvrhi::coopvec::DataType::SInt64, 8 },
                { nvrhi::coopvec::DataType::FloatE4M3, 1 },
                { nvrhi::coopvec::DataType::FloatE5M2, 1 },
                { nvrhi::coopvec::DataType::Float16, 2 },
                { nvrhi::coopvec::DataType::BFloat16, 2 },
                { nvrhi::coopvec::DataType::Float32, 4 },
                { nvrhi::coopvec::DataType::Float64, 8 },
            };

            constexpr uint32_t kRows = 16;
            constexpr uint32_t kCols = 24;

            for (const auto& entry : kTypes)
            {
                const size_t elemSize = nvrhi::coopvec::getDataTypeSize(entry.type);
                UAA_CHECK_EQ(elemSize, entry.expectedBytes, "coopvec::getDataTypeSize mismatch");

                UAA_CHECK_EQ(
                    nvrhi::coopvec::getOptimalMatrixStride(entry.type, nvrhi::coopvec::MatrixLayout::RowMajor, kRows, kCols),
                    elemSize * kCols,
                    "RowMajor optimal stride mismatch");
                UAA_CHECK_EQ(
                    nvrhi::coopvec::getOptimalMatrixStride(entry.type, nvrhi::coopvec::MatrixLayout::ColumnMajor, kRows, kCols),
                    elemSize * kRows,
                    "ColumnMajor optimal stride mismatch");
                UAA_CHECK_EQ(
                    nvrhi::coopvec::getOptimalMatrixStride(entry.type, nvrhi::coopvec::MatrixLayout::InferencingOptimal, kRows, kCols),
                    size_t(0),
                    "InferencingOptimal stride must be 0");
                UAA_CHECK_EQ(
                    nvrhi::coopvec::getOptimalMatrixStride(entry.type, nvrhi::coopvec::MatrixLayout::TrainingOptimal, kRows, kCols),
                    size_t(0),
                    "TrainingOptimal stride must be 0");
            }

            if (!ctx.fixture)
            {
                ctx.skip("CPU CoopVec helpers verified; GPU fixture required for device->getCoopVecMatrixSize");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasInf = fix.features().supports(nvrhi::Feature::CooperativeVectorInferencing);
            for (const auto layout : {
                     nvrhi::coopvec::MatrixLayout::RowMajor,
                     nvrhi::coopvec::MatrixLayout::ColumnMajor,
                     nvrhi::coopvec::MatrixLayout::InferencingOptimal,
                     nvrhi::coopvec::MatrixLayout::TrainingOptimal })
            {
                size_t sz = 0;
                if (!hasInf && fix.isValidationActive())
                {
                    ExpectErrorsGuard expectErr(fix);
                    sz = fix.device()->getCoopVecMatrixSize(
                        nvrhi::coopvec::DataType::Float16, layout, 16, 16);
                    UAA_CHECK(expectErr.errorCount() > 0,
                        "Expected validation error for getCoopVecMatrixSize when CooperativeVectorInferencing is unsupported");
                }
                else
                {
                    sz = fix.device()->getCoopVecMatrixSize(
                        nvrhi::coopvec::DataType::Float16, layout, 16, 16);
                }

                if (!hasInf)
                {
                    UAA_CHECK_EQ(sz, size_t(0), "Unsupported getCoopVecMatrixSize must return 0");
                }
                else if (layout != nvrhi::coopvec::MatrixLayout::TrainingOptimal ||
                    fix.features().supports(nvrhi::Feature::CooperativeVectorTraining))
                {
                    UAA_CHECK(sz >= size_t(16) * size_t(16) * 2u,
                        "Supported getCoopVecMatrixSize must be at least raw matrix byte size");
                }
            }
        }

        void testCoopConvert(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::CooperativeVectorInferencing))
            {
                ctx.skip("Feature::CooperativeVectorInferencing not supported");
            }

            constexpr uint32_t kRows = 16;
            constexpr uint32_t kCols = 16;
            const size_t rowMajorStride =
                nvrhi::coopvec::getOptimalMatrixStride(nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::RowMajor, kRows, kCols);
            const size_t colMajorStride =
                nvrhi::coopvec::getOptimalMatrixStride(nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::ColumnMajor, kRows, kCols);

            const size_t rowSize = fix.device()->getCoopVecMatrixSize(
                nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::RowMajor, kRows, kCols);
            const size_t optSize = fix.device()->getCoopVecMatrixSize(
                nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::InferencingOptimal, kRows, kCols);
            const size_t colSize = fix.device()->getCoopVecMatrixSize(
                nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::ColumnMajor, kRows, kCols);
            UAA_CHECK(rowSize > 0 && optSize > 0 && colSize > 0, "Matrix sizes must be positive");

            std::vector<uint16_t> srcData(kRows * kCols);
            std::vector<uint16_t> expectedColData(kRows * kCols);
            for (uint32_t r = 0; r < kRows; ++r)
            {
                for (uint32_t c = 0; c < kCols; ++c)
                {
                    const uint16_t val = static_cast<uint16_t>(0x3C00u + r * kCols + c);
                    srcData[r * kCols + c] = val;
                    expectedColData[c * kRows + r] = val;
                }
            }

            auto makeMatrixBuf = [&](size_t bytes, const char* name)
            {
                nvrhi::BufferDesc d;
                d.setByteSize(bytes)
                    .setCanHaveUAVs(true)
                    .setDebugName(name);
                d.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
                nvrhi::BufferHandle buf = fix.device()->createBuffer(d);
                UAA_CHECK(buf != nullptr, std::string("Failed to create matrix buffer ") + name);
                return buf;
            };

            nvrhi::BufferHandle srcBuf = makeMatrixBuf(rowSize, "uaa_coop_src");
            nvrhi::BufferHandle optBuf = makeMatrixBuf(optSize, "uaa_coop_opt");
            nvrhi::BufferHandle dstRowBuf = makeMatrixBuf(rowSize, "uaa_coop_dst_row");
            nvrhi::BufferHandle dstColBuf = makeMatrixBuf(colSize, "uaa_coop_dst_col");

            fix.uploadBuffer(srcBuf, srcData.data(), srcData.size() * sizeof(uint16_t));

            nvrhi::coopvec::ConvertMatrixLayoutDesc toOpt{};
            toOpt.numRows = kRows;
            toOpt.numColumns = kCols;
            toOpt.src.buffer = srcBuf;
            toOpt.src.type = nvrhi::coopvec::DataType::Float16;
            toOpt.src.layout = nvrhi::coopvec::MatrixLayout::RowMajor;
            toOpt.src.size = rowSize;
            toOpt.src.stride = rowMajorStride;
            toOpt.dst.buffer = optBuf;
            toOpt.dst.type = nvrhi::coopvec::DataType::Float16;
            toOpt.dst.layout = nvrhi::coopvec::MatrixLayout::InferencingOptimal;
            toOpt.dst.size = optSize;
            toOpt.dst.stride = 0;

            nvrhi::coopvec::ConvertMatrixLayoutDesc toRow{};
            toRow.numRows = kRows;
            toRow.numColumns = kCols;
            toRow.src = toOpt.dst;
            toRow.dst = toOpt.src;
            toRow.dst.buffer = dstRowBuf;

            nvrhi::coopvec::ConvertMatrixLayoutDesc toCol{};
            toCol.numRows = kRows;
            toCol.numColumns = kCols;
            toCol.src = toOpt.src;
            toCol.dst.buffer = dstColBuf;
            toCol.dst.type = nvrhi::coopvec::DataType::Float16;
            toCol.dst.layout = nvrhi::coopvec::MatrixLayout::ColumnMajor;
            toCol.dst.size = colSize;
            toCol.dst.stride = colMajorStride;

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->convertCoopVecMatrices(&toOpt, 1);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->convertCoopVecMatrices(&toRow, 1);
                cmd->convertCoopVecMatrices(&toCol, 1);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            std::vector<uint16_t> outRow(kRows * kCols, 0);
            std::vector<uint16_t> outCol(kRows * kCols, 0);
            fix.readbackBuffer(dstRowBuf, outRow.data(), outRow.size() * sizeof(uint16_t));
            fix.readbackBuffer(dstColBuf, outCol.data(), outCol.size() * sizeof(uint16_t));

            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(outRow.data(), srcData.data(), outRow.size() * sizeof(uint16_t), diff),
                "CoopVec RowMajor->InferencingOptimal->RowMajor round-trip mismatch: " + diff);
            UAA_CHECK(TestFixture::checkBytesEqual(outCol.data(), expectedColData.data(), outCol.size() * sizeof(uint16_t), diff),
                "CoopVec RowMajor->ColumnMajor conversion mismatch: " + diff);
        }

        void testCoopTraining(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::CooperativeVectorTraining))
            {
                ctx.skip("Feature::CooperativeVectorTraining not supported");
            }

            const nvrhi::coopvec::TrainingFormatSupport fp16 =
                fix.device()->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::Float16);
            const nvrhi::coopvec::TrainingFormatSupport fp32 =
                fix.device()->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::Float32);
            UAA_CHECK(fp16.bufferTrainingSupported || fp32.bufferTrainingSupported,
                "CooperativeVectorTraining advertised true but neither FP16 nor FP32 buffer training is supported");

            constexpr uint32_t kRows = 16;
            constexpr uint32_t kCols = 16;
            const size_t rowStride =
                nvrhi::coopvec::getOptimalMatrixStride(nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::RowMajor, kRows, kCols);
            const size_t rowSize = fix.device()->getCoopVecMatrixSize(
                nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::RowMajor, kRows, kCols);
            const size_t trainSize = fix.device()->getCoopVecMatrixSize(
                nvrhi::coopvec::DataType::Float16, nvrhi::coopvec::MatrixLayout::TrainingOptimal, kRows, kCols);
            UAA_CHECK(rowSize > 0 && trainSize > 0, "TrainingOptimal matrix size must be positive");

            std::vector<uint16_t> srcData(kRows * kCols);
            for (size_t i = 0; i < srcData.size(); ++i)
            {
                srcData[i] = static_cast<uint16_t>(0x3C00u + i);
            }

            nvrhi::BufferDesc bufDesc;
            bufDesc.setByteSize(rowSize > trainSize ? rowSize : trainSize)
                .setCanHaveUAVs(true)
                .setDebugName("uaa_coop_train_buf");
            bufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);

            nvrhi::BufferHandle srcBuf = fix.device()->createBuffer(bufDesc);
            nvrhi::BufferHandle trainBuf = fix.device()->createBuffer(bufDesc);
            nvrhi::BufferHandle dstBuf = fix.device()->createBuffer(bufDesc);
            UAA_CHECK(srcBuf != nullptr && trainBuf != nullptr && dstBuf != nullptr,
                "Failed to create CoopVec training buffers");
            fix.uploadBuffer(srcBuf, srcData.data(), srcData.size() * sizeof(uint16_t));

            nvrhi::coopvec::ConvertMatrixLayoutDesc toTrain{};
            toTrain.numRows = kRows;
            toTrain.numColumns = kCols;
            toTrain.src.buffer = srcBuf;
            toTrain.src.type = nvrhi::coopvec::DataType::Float16;
            toTrain.src.layout = nvrhi::coopvec::MatrixLayout::RowMajor;
            toTrain.src.size = rowSize;
            toTrain.src.stride = rowStride;
            toTrain.dst.buffer = trainBuf;
            toTrain.dst.type = nvrhi::coopvec::DataType::Float16;
            toTrain.dst.layout = nvrhi::coopvec::MatrixLayout::TrainingOptimal;
            toTrain.dst.size = trainSize;
            toTrain.dst.stride = 0;

            nvrhi::coopvec::ConvertMatrixLayoutDesc toRow{};
            toRow.numRows = kRows;
            toRow.numColumns = kCols;
            toRow.src = toTrain.dst;
            toRow.dst = toTrain.src;
            toRow.dst.buffer = dstBuf;

            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->convertCoopVecMatrices(&toTrain, 1);
                cmd->close();
                fix.submitAndWait(cmd);
            }
            {
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                cmd->convertCoopVecMatrices(&toRow, 1);
                cmd->close();
                fix.submitAndWait(cmd);
            }

            std::vector<uint16_t> out(kRows * kCols, 0);
            fix.readbackBuffer(dstBuf, out.data(), out.size() * sizeof(uint16_t));
            std::string diff;
            UAA_CHECK(TestFixture::checkBytesEqual(out.data(), srcData.data(), out.size() * sizeof(uint16_t), diff),
                "CoopVec TrainingOptimal round-trip mismatch: " + diff);
        }

        void testOptVrs(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            nvrhi::VariableRateShadingFeatureInfo vrsInfo{};
            vrsInfo.shadingRateImageTileSize = TestFixture::kGuardWord;
            const bool supported = fix.device()->queryFeatureSupport(
                nvrhi::Feature::VariableRateShading, &vrsInfo, sizeof(vrsInfo));

            if (!supported)
            {
                UAA_CHECK(
                    vrsInfo.shadingRateImageTileSize == TestFixture::kGuardWord
                    || vrsInfo.shadingRateImageTileSize == 0u,
                    "Unsupported VRS feature query must preserve sentinel or zero shadingRateImageTileSize");
                if (vrsInfo.shadingRateImageTileSize == 0u)
                {
                    ctx.note("Unsupported VRS feature query zeroed shadingRateImageTileSize");
                }
                ctx.skip("Feature::VariableRateShading not supported");
            }

            UAA_CHECK(vrsInfo.shadingRateImageTileSize > 0, "VRS tile size must be positive");

            constexpr uint32_t kSize = 64;
            const uint32_t tileSize = vrsInfo.shadingRateImageTileSize;
            const uint32_t vrsW = (kSize + tileSize - 1u) / tileSize;
            const uint32_t vrsH = (kSize + tileSize - 1u) / tileSize;

            nvrhi::TextureDesc rtDesc;
            rtDesc.setWidth(kSize)
                .setHeight(kSize)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setIsRenderTarget(true)
                .setDebugName("uaa_vrs_rt");
            rtDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget);
            nvrhi::TextureHandle rt = fix.device()->createTexture(rtDesc);
            UAA_CHECK(rt != nullptr, "Failed to create VRS colour target");

            nvrhi::TextureDesc srsDesc;
            srsDesc.setWidth(vrsW)
                .setHeight(vrsH)
                .setFormat(nvrhi::Format::R8_UINT)
                .setDebugName("uaa_vrs_surface");
            srsDesc.isShadingRateSurface = true;
            srsDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShadingRateSurface);
            nvrhi::TextureHandle srs = fix.device()->createTexture(srsDesc);
            UAA_CHECK(srs != nullptr, "Failed to create VRS shading rate surface");

            // Fill VRS rate image with 0x05 (2x2 shading rate encoding: (1 << 2) | 1 = 0x5 in DX12/VK).
            std::vector<uint8_t> rateData(size_t(vrsW) * size_t(vrsH), 0x05u);
            fix.uploadTexture2D(srs, 0, 0, rateData.data(), vrsW);

            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(rt).setShadingRateAttachment(srs);
            nvrhi::FramebufferHandle fb = fix.device()->createFramebuffer(fbDesc);
            UAA_CHECK(fb != nullptr, "Failed to create VRS framebuffer");

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPushConstants)));
            nvrhi::BindingLayoutHandle layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create VRS binding layout");

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPushConstants)));
            nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "Failed to create VRS binding set");

            nvrhi::VariableRateShadingState vrsState;
            vrsState.setEnabled(true)
                .setShadingRate(nvrhi::VariableShadingRate::e2x2)
                .setPipelinePrimitiveCombiner(nvrhi::ShadingRateCombiner::Override)
                .setImageCombiner(nvrhi::ShadingRateCombiner::Max);

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.setVertexShader(vs)
                .setPixelShader(ps)
                .addBindingLayout(layout)
                .setVariableRateShadingState(vrsState);
            psoDesc.renderState.rasterState.setCullNone();
            nvrhi::GraphicsPipelineHandle pso = fix.device()->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
            UAA_CHECK(pso != nullptr, "createGraphicsPipeline with VRS state returned nullptr");

            const FullscreenPushConstants pc{ { 0.4f, 0.8f, 0.2f, 1.0f }, 0.5f, 1.0f, { 0.0f, 0.0f } };

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            nvrhi::utils::ClearColorAttachment(cmd, fb, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
            nvrhi::GraphicsState gfxState;
            gfxState.setPipeline(pso)
                .setFramebuffer(fb)
                .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                .addBindingSet(set)
                .setShadingRateState(vrsState);
            cmd->setGraphicsState(gfxState);
            cmd->setPushConstants(&pc, sizeof(pc));
            cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint8_t> pixels;
            fix.readbackTexture2D(rt, 0, 0, pixels);
            const size_t centerOffset = (size_t(kSize / 2u) * size_t(kSize) + size_t(kSize / 2u)) * 4u;
            std::string diff;
            UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[centerOffset], 102, 204, 51, 255, 2, diff),
                "OPT.VRS rendered pixel mismatch: " + diff);
        }

        void testOptRaster(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasCons = fix.features().supports(nvrhi::Feature::ConservativeRasterization);
            const bool hasFastGs = fix.features().supports(nvrhi::Feature::FastGeometryShader);
            const bool hasSps = fix.features().supports(nvrhi::Feature::SinglePassStereo);

            std::ostringstream oss;
            oss << "ConservativeRasterization=" << (hasCons ? "true" : "false")
                << ", FastGeometryShader=" << (hasFastGs ? "true" : "false")
                << ", SinglePassStereo=" << (hasSps ? "true" : "false");
            ctx.note(oss.str());

            if (!hasCons && !hasFastGs && !hasSps)
            {
                ctx.skip("ConservativeRasterization, FastGeometryShader, and SinglePassStereo not supported");
            }

            if (hasCons)
            {
                constexpr uint32_t kSize = 32;
                nvrhi::TextureDesc rtDesc;
                rtDesc.setWidth(kSize).setHeight(kSize).setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setIsRenderTarget(true).setDebugName("uaa_cons_rt");
                rtDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget);
                nvrhi::TextureHandle rt = fix.device()->createTexture(rtDesc);
                UAA_CHECK(rt != nullptr, "Failed to create conservative raster target");

                nvrhi::FramebufferHandle fb = fix.device()->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(rt));
                UAA_CHECK(fb != nullptr, "Failed to create conservative raster framebuffer");

                nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex);
                nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel);

                nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics, 0, false);
                layoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(FullscreenPushConstants)));
                nvrhi::BindingLayoutHandle layout = fix.device()->createBindingLayout(layoutDesc);
                UAA_CHECK(layout != nullptr, "Failed to create binding layout");

                nvrhi::BindingSetDesc setDesc;
                setDesc.addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(FullscreenPushConstants)));
                nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, layout);
                UAA_CHECK(set != nullptr, "Failed to create binding set");

                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.setVertexShader(vs).setPixelShader(ps).addBindingLayout(layout);
                psoDesc.renderState.rasterState.setCullNone().setConservativeRasterEnable(true);
                nvrhi::GraphicsPipelineHandle pso =
                    fix.device()->createGraphicsPipeline(psoDesc, fb->getFramebufferInfo());
                UAA_CHECK(pso != nullptr, "createGraphicsPipeline with conservativeRasterEnable returned nullptr");

                const FullscreenPushConstants pc{ { 0.25f, 0.75f, 0.50f, 1.0f }, 0.5f, 1.0f, { 0.0f, 0.0f } };
                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
                cmd->open();
                nvrhi::utils::ClearColorAttachment(cmd, fb, 0, nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f));
                nvrhi::GraphicsState gfxState;
                gfxState.setPipeline(pso).setFramebuffer(fb)
                    .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(float(kSize), float(kSize))))
                    .addBindingSet(set);
                cmd->setGraphicsState(gfxState);
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();
                fix.submitAndWait(cmd);

                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(rt, 0, 0, pixels);
                const size_t centerOffset = (size_t(kSize / 2u) * size_t(kSize) + size_t(kSize / 2u)) * 4u;
                std::string diff;
                UAA_CHECK(TestFixture::checkPixelRgba8(&pixels[centerOffset], 64, 191, 128, 255, 2, diff),
                    "ConservativeRasterization pixel mismatch: " + diff);
            }
        }

        void testOptWave(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            nvrhi::WaveLaneCountMinMaxFeatureInfo waveInfo{};
            waveInfo.minWaveLaneCount = TestFixture::kGuardWord;
            waveInfo.maxWaveLaneCount = TestFixture::kGuardWord;

            const bool supported = fix.device()->queryFeatureSupport(
                nvrhi::Feature::WaveLaneCountMinMax, &waveInfo, sizeof(waveInfo));

            if (!supported)
            {
                UAA_CHECK_EQ(waveInfo.minWaveLaneCount, TestFixture::kGuardWord,
                    "Unsupported WaveLaneCountMinMax must preserve min sentinel");
                UAA_CHECK_EQ(waveInfo.maxWaveLaneCount, TestFixture::kGuardWord,
                    "Unsupported WaveLaneCountMinMax must preserve max sentinel");
                ctx.skip("Feature::WaveLaneCountMinMax not supported");
            }

            UAA_CHECK(waveInfo.minWaveLaneCount > 0, "minWaveLaneCount must be positive");
            UAA_CHECK(waveInfo.minWaveLaneCount <= waveInfo.maxWaveLaneCount,
                "minWaveLaneCount must be <= maxWaveLaneCount");

            nvrhi::ShaderHandle cs = fix.createShader("cs_wave_lane", nvrhi::ShaderType::Compute);
            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
            nvrhi::BindingLayoutHandle layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create wave lane binding layout");

            nvrhi::ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(cs).addBindingLayout(layout);
            nvrhi::ComputePipelineHandle pso = fix.device()->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "Failed to create wave lane compute pipeline");

            nvrhi::BufferDesc outDesc;
            outDesc.setByteSize(sizeof(uint32_t) * 4)
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(true)
                .setDebugName("uaa_wave_lane_out");
            outDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess);
            nvrhi::BufferHandle outBuf = fix.device()->createBuffer(outDesc);
            UAA_CHECK(outBuf != nullptr, "Failed to create wave lane output buffer");

            nvrhi::BindingSetDesc setDesc;
            setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outBuf));
            nvrhi::BindingSetHandle set = fix.device()->createBindingSet(setDesc, layout);
            UAA_CHECK(set != nullptr, "Failed to create wave lane binding set");

            nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList returned nullptr");
            cmd->open();
            cmd->clearBufferUInt(outBuf, 0u);
            nvrhi::ComputeState csState;
            csState.setPipeline(pso).addBindingSet(set);
            cmd->setComputeState(csState);
            cmd->dispatch(1, 1, 1);
            cmd->close();
            fix.submitAndWait(cmd);

            uint32_t observed = 0;
            fix.readbackBuffer(outBuf, &observed, sizeof(observed));

            std::ostringstream oss;
            oss << "observedWaveLaneCount=" << observed
                << ", range=[" << waveInfo.minWaveLaneCount << ", " << waveInfo.maxWaveLaneCount << "]";
            ctx.note(oss.str());

            UAA_CHECK(observed >= waveInfo.minWaveLaneCount && observed <= waveInfo.maxWaveLaneCount,
                "Observed WaveGetLaneCount() outside reported [minWaveLaneCount, maxWaveLaneCount]");
        }

        void testOptOther(TestContext& ctx)
        {
            if (!ctx.fixture)
            {
                ctx.skip("GPU fixture required");
            }
            TestFixture& fix = *ctx.fixture;

            const bool hasHeapDirectlyIndexed = fix.device()->queryFeatureSupport(nvrhi::Feature::HeapDirectlyIndexed);
            const bool hasHlslExtUav = fix.device()->queryFeatureSupport(nvrhi::Feature::HlslExtensionUAV);
            const bool hasEnhancedBarriers = fix.device()->queryFeatureSupport(nvrhi::Feature::EnhancedBarriers);

            std::ostringstream oss;
            oss << "HeapDirectlyIndexed=" << (hasHeapDirectlyIndexed ? "true" : "false")
                << ", HlslExtensionUAV=" << (hasHlslExtUav ? "true" : "false")
                << ", EnhancedBarriers=" << (hasEnhancedBarriers ? "true" : "false");

            if (fix.api() != nvrhi::GraphicsAPI::D3D12)
            {
                UAA_CHECK(!hasEnhancedBarriers, "EnhancedBarriers must be false on non-D3D12 APIs");
            }

            if (hasHeapDirectlyIndexed)
            {
                nvrhi::BindlessLayoutDesc blDesc;
                blDesc.setVisibility(nvrhi::ShaderType::Compute)
                    .setFirstSlot(0)
                    .setMaxCapacity(8)
                    .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv);
                nvrhi::BindingLayoutHandle bl = fix.device()->createBindlessLayout(blDesc);
                UAA_CHECK(bl != nullptr, "createBindlessLayout(MutableSrvUavCbv) returned nullptr");

                nvrhi::DescriptorTableHandle table = fix.device()->createDescriptorTable(bl);
                UAA_CHECK(table != nullptr, "createDescriptorTable(MutableSrvUavCbv) returned nullptr");

                nvrhi::BufferDesc bufDesc;
                bufDesc.setByteSize(64)
                    .setStructStride(sizeof(uint32_t))
                    .setCanHaveUAVs(true)
                    .setDebugName("uaa_mutable_bindless_buf");
                bufDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                nvrhi::BufferHandle buf = fix.device()->createBuffer(bufDesc);
                UAA_CHECK(buf != nullptr, "Failed to create mutable bindless buffer");

                nvrhi::TextureDesc texDesc;
                texDesc.setWidth(4).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM)
                    .setDebugName("uaa_mutable_bindless_tex");
                texDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                nvrhi::TextureHandle tex = fix.device()->createTexture(texDesc);
                UAA_CHECK(tex != nullptr, "Failed to create mutable bindless texture");

                UAA_CHECK(fix.device()->writeDescriptorTable(table, nvrhi::BindingSetItem::Texture_SRV(0, tex)),
                    "writeDescriptorTable(Texture_SRV) failed on MutableSrvUavCbv");
                UAA_CHECK(fix.device()->writeDescriptorTable(table, nvrhi::BindingSetItem::StructuredBuffer_SRV(1, buf)),
                    "writeDescriptorTable(StructuredBuffer_SRV) failed on MutableSrvUavCbv");
                UAA_CHECK(fix.device()->writeDescriptorTable(table, nvrhi::BindingSetItem::StructuredBuffer_UAV(2, buf)),
                    "writeDescriptorTable(StructuredBuffer_UAV) failed on MutableSrvUavCbv");

                oss << ", MutableSrvUavCbvTableCapacity=" << table->getCapacity();
            }

            ctx.note(oss.str());
        }
    } // namespace

    UAA_REGISTER_TEST("COOP.Capabilities", "optional", testCoopCapabilities);
    UAA_REGISTER_TEST("COOP.MatrixSize", "optional", testCoopMatrixSize);
    UAA_REGISTER_TEST("COOP.Convert", "optional", testCoopConvert);
    UAA_REGISTER_TEST("COOP.Training", "optional", testCoopTraining);
    UAA_REGISTER_TEST("OPT.VRS", "optional", testOptVrs);
    UAA_REGISTER_TEST("OPT.Raster", "optional", testOptRaster);
    UAA_REGISTER_TEST("OPT.Wave", "optional", testOptWave);
    UAA_REGISTER_TEST("OPT.Other", "optional", testOptOther);
} // namespace nvrhi::uaa
