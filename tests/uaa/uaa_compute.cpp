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
#include <string>
#include <vector>

namespace nvrhi::uaa
{
    namespace
    {
        struct CopyAddParams
        {
            uint32_t count;
            uint32_t addVal;
            uint32_t xorVal;
            uint32_t pad;
        };

        struct UavChainPC
        {
            uint32_t count;
            uint32_t stepAdd;
            uint32_t stepMul;
            uint32_t pad;
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

        struct IndirectWorkPC
        {
            uint32_t baseOffset;
            uint32_t marker;
            uint32_t strideY;
            uint32_t strideZ;
        };

        TestFixture& requireFixture(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "Compute test requires a valid TestFixture");
            return *ctx.fixture;
        }

        nvrhi::BufferHandle createStructuredBuffer(
            TestContext& ctx,
            nvrhi::IDevice* dev,
            uint32_t elementCount,
            bool uav,
            bool indirectArgs,
            nvrhi::ResourceStates initialState,
            const char* debugName)
        {
            nvrhi::BufferDesc desc;
            desc.setByteSize(uint64_t(elementCount) * sizeof(uint32_t))
                .setStructStride(sizeof(uint32_t))
                .setCanHaveUAVs(uav)
                .setIsDrawIndirectArgs(indirectArgs)
                .enableAutomaticStateTracking(initialState)
                .setDebugName(debugName);
            nvrhi::BufferHandle buf = dev->createBuffer(desc);
            UAA_CHECK(buf != nullptr, std::string("Failed to create structured buffer: ") + debugName);
            return buf;
        }

        nvrhi::BufferHandle createConstantBuffer(
            TestContext& ctx,
            nvrhi::IDevice* dev,
            uint64_t byteSize,
            const char* debugName)
        {
            nvrhi::BufferDesc desc;
            desc.setByteSize(byteSize)
                .setIsConstantBuffer(true)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
                .setDebugName(debugName);
            nvrhi::BufferHandle buf = dev->createBuffer(desc);
            UAA_CHECK(buf != nullptr, std::string("Failed to create constant buffer: ") + debugName);
            return buf;
        }

        void checkWordSlice(
            TestContext& ctx,
            const std::vector<uint32_t>& actual,
            uint32_t begin,
            uint32_t end,
            uint32_t expectedConstant,
            const char* label)
        {
            for (uint32_t i = begin; i < end; ++i)
            {
                if (actual[i] != expectedConstant)
                {
                    std::ostringstream oss;
                    oss << label << " mismatch at index " << i
                        << ": actual=0x" << std::hex << actual[i]
                        << " expected=0x" << expectedConstant;
                    UAA_CHECK(false, oss.str());
                }
            }
        }

        void testComputeDirect(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kActiveCount = 137;
            constexpr uint32_t kTotalCount = 200;
            constexpr uint32_t kAddVal = 0x12345678u;
            constexpr uint32_t kXorVal = 0x9ABCDEF0u;

            nvrhi::ShaderHandle cs = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            UAA_CHECK(cs != nullptr, "Failed to create cs_buffer_copy_add shader");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layout = dev->createBindingLayout(layoutDesc);
            UAA_CHECK(layout != nullptr, "Failed to create binding layout");

            nvrhi::ComputePipelineDesc psoDesc;
            psoDesc.setComputeShader(cs).addBindingLayout(layout);
            nvrhi::ComputePipelineHandle pso = dev->createComputePipeline(psoDesc);
            UAA_CHECK(pso != nullptr, "Failed to create compute pipeline");

            nvrhi::BufferHandle cb = createConstantBuffer(ctx, dev, sizeof(CopyAddParams), "cmp_direct_cb");
            nvrhi::BufferHandle inBuf = createStructuredBuffer(ctx, dev, kTotalCount, false, false, nvrhi::ResourceStates::ShaderResource, "cmp_direct_in");
            nvrhi::BufferHandle outBuf = createStructuredBuffer(ctx, dev, kTotalCount, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_direct_out");

            const std::vector<uint32_t> inputWords = TestFixture::makeDeterministicWords(kTotalCount, ctx.seed() ^ 0x1001ull);
            const std::vector<uint32_t> sentinelWords(kTotalCount, TestFixture::kGuardWord);
            const CopyAddParams params{kActiveCount, kAddVal, kXorVal, 0u};

            fix.uploadBuffer(cb, &params, sizeof(params));
            fix.uploadBuffer(inBuf, inputWords.data(), inputWords.size() * sizeof(uint32_t));
            fix.uploadBuffer(outBuf, sentinelWords.data(), sentinelWords.size() * sizeof(uint32_t));

            nvrhi::BindingSetDesc bsDesc;
            bsDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, outBuf));
            nvrhi::BindingSetHandle bs = dev->createBindingSet(bsDesc, layout);
            UAA_CHECK(bs != nullptr, "Failed to create binding set");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            nvrhi::ComputeState state;
            state.setPipeline(pso).addBindingSet(bs);
            cmd->setComputeState(state);
            cmd->dispatch((kActiveCount + 63u) / 64u, 1u, 1u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kTotalCount, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));

            for (uint32_t i = 0; i < kActiveCount; ++i)
            {
                const uint32_t expected = (inputWords[i] + kAddVal) ^ kXorVal;
                UAA_CHECK_EQ(actual[i], expected, "CMP.Direct active element mismatch");
            }
            checkWordSlice(ctx, actual, kActiveCount, kTotalCount, TestFixture::kGuardWord, "CMP.Direct guard region");
        }

        void testComputeStateChanges(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kCount = 64;
            constexpr size_t kByteSize = kCount * sizeof(uint32_t);

            nvrhi::ShaderHandle csCopyAdd = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::ShaderHandle csChain = fix.createShader("cs_uav_chain", nvrhi::ShaderType::Compute);
            UAA_CHECK(csCopyAdd != nullptr, "Failed to create cs_buffer_copy_add");
            UAA_CHECK(csChain != nullptr, "Failed to create cs_uav_chain");

            nvrhi::BindingLayoutDesc layoutCopyDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutCopyDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            nvrhi::BindingLayoutHandle layoutCopy = dev->createBindingLayout(layoutCopyDesc);
            UAA_CHECK(layoutCopy != nullptr, "Failed to create copy_add layout");

            nvrhi::BindingLayoutDesc layoutChainDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            layoutChainDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(UavChainPC)))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
            nvrhi::BindingLayoutHandle layoutChain = dev->createBindingLayout(layoutChainDesc);
            UAA_CHECK(layoutChain != nullptr, "Failed to create uav_chain layout");

            nvrhi::ComputePipelineHandle psoCopyAdd = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csCopyAdd).addBindingLayout(layoutCopy));
            nvrhi::ComputePipelineHandle psoChain = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csChain).addBindingLayout(layoutChain));
            UAA_CHECK(psoCopyAdd != nullptr, "Failed to create psoCopyAdd");
            UAA_CHECK(psoChain != nullptr, "Failed to create psoChain");

            nvrhi::BufferHandle cbA = createConstantBuffer(ctx, dev, sizeof(CopyAddParams), "cmp_sc_cbA");
            nvrhi::BufferHandle cbB = createConstantBuffer(ctx, dev, sizeof(CopyAddParams), "cmp_sc_cbB");
            nvrhi::BufferHandle srcA = createStructuredBuffer(ctx, dev, kCount, false, false, nvrhi::ResourceStates::ShaderResource, "cmp_sc_srcA");
            nvrhi::BufferHandle srcB = createStructuredBuffer(ctx, dev, kCount, false, false, nvrhi::ResourceStates::ShaderResource, "cmp_sc_srcB");
            nvrhi::BufferHandle dstA = createStructuredBuffer(ctx, dev, kCount, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_sc_dstA");
            nvrhi::BufferHandle dstB = createStructuredBuffer(ctx, dev, kCount, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_sc_dstB");
            nvrhi::BufferHandle snapshotA = createStructuredBuffer(ctx, dev, kCount, false, false, nvrhi::ResourceStates::CopyDest, "cmp_sc_snapshotA");

            const std::vector<uint32_t> wordsA = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x2001ull);
            const std::vector<uint32_t> wordsB = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x2002ull);
            const CopyAddParams paramsA{kCount, 7u, 0x0Fu, 0u};
            const CopyAddParams paramsB{kCount, 19u, 0x33u, 0u};

            fix.uploadBuffer(cbA, &paramsA, sizeof(paramsA));
            fix.uploadBuffer(cbB, &paramsB, sizeof(paramsB));
            fix.uploadBuffer(srcA, wordsA.data(), kByteSize);
            fix.uploadBuffer(srcB, wordsB.data(), kByteSize);

            nvrhi::BindingSetHandle bsCopyA = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cbA))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srcA))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, dstA)),
                layoutCopy);
            nvrhi::BindingSetHandle bsCopyB = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cbB))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, srcB))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, dstB)),
                layoutCopy);
            nvrhi::BindingSetHandle bsChainA = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, dstA)),
                layoutChain);
            nvrhi::BindingSetHandle bsChainB = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, dstB)),
                layoutChain);
            UAA_CHECK(bsCopyA != nullptr && bsCopyB != nullptr && bsChainA != nullptr && bsChainB != nullptr,
                "Failed to create binding sets");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            const UavChainPC pcA1{kCount, 3u, 5u, 0u};
            const UavChainPC pcB1{kCount, 11u, 2u, 0u};
            const UavChainPC pcA2{kCount, 100u, 1u, 0u};

            cmd->open();
            // Switch pipelines, binding sets, and blit operations within one command list.
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoCopyAdd).addBindingSet(bsCopyA));
            cmd->dispatch(1u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoChain).addBindingSet(bsChainA));
            cmd->setPushConstants(&pcA1, sizeof(pcA1));
            cmd->dispatch(1u, 1u, 1u);

            cmd->copyBuffer(snapshotA, 0u, dstA, 0u, kByteSize);
            cmd->clearBufferUInt(dstA, 0xCAFEBABEu);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoCopyAdd).addBindingSet(bsCopyB));
            cmd->dispatch(1u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoChain).addBindingSet(bsChainB));
            cmd->setPushConstants(&pcB1, sizeof(pcB1));
            cmd->dispatch(1u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoCopyAdd).addBindingSet(bsCopyA));
            cmd->dispatch(1u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoChain).addBindingSet(bsChainA));
            cmd->setPushConstants(&pcA2, sizeof(pcA2));
            cmd->dispatch(1u, 1u, 1u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actualSnap(kCount, 0u);
            std::vector<uint32_t> actualDstA(kCount, 0u);
            std::vector<uint32_t> actualDstB(kCount, 0u);
            fix.readbackBuffer(snapshotA, actualSnap.data(), kByteSize);
            fix.readbackBuffer(dstA, actualDstA.data(), kByteSize);
            fix.readbackBuffer(dstB, actualDstB.data(), kByteSize);

            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t stepA1 = ((wordsA[i] + paramsA.addVal) ^ paramsA.xorVal) * pcA1.stepMul + pcA1.stepAdd;
                const uint32_t stepA2 = ((wordsA[i] + paramsA.addVal) ^ paramsA.xorVal) * pcA2.stepMul + pcA2.stepAdd;
                const uint32_t stepB1 = ((wordsB[i] + paramsB.addVal) ^ paramsB.xorVal) * pcB1.stepMul + pcB1.stepAdd;

                UAA_CHECK_EQ(actualSnap[i], stepA1, "CMP.StateChanges snapshotA mismatch");
                UAA_CHECK_EQ(actualDstA[i], stepA2, "CMP.StateChanges dstA final mismatch");
                UAA_CHECK_EQ(actualDstB[i], stepB1, "CMP.StateChanges dstB mismatch");
            }
        }

        void testComputeIndirect(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kOutWords = 64;

            nvrhi::ShaderHandle csGen = fix.createShader("cs_indirect_gen", nvrhi::ShaderType::Compute);
            nvrhi::ShaderHandle csWork = fix.createShader("cs_indirect_work", nvrhi::ShaderType::Compute);
            UAA_CHECK(csGen != nullptr && csWork != nullptr, "Failed to create indirect compute shaders");

            nvrhi::BindingLayoutDesc genLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            genLayoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1));
            nvrhi::BindingLayoutHandle genLayout = dev->createBindingLayout(genLayoutDesc);
            UAA_CHECK(genLayout != nullptr, "Failed to create genLayout");

            nvrhi::BindingLayoutDesc workLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            workLayoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(IndirectWorkPC)))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
            nvrhi::BindingLayoutHandle workLayout = dev->createBindingLayout(workLayoutDesc);
            UAA_CHECK(workLayout != nullptr, "Failed to create workLayout");

            nvrhi::ComputePipelineHandle psoGen = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csGen).addBindingLayout(genLayout));
            nvrhi::ComputePipelineHandle psoWork = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csWork).addBindingLayout(workLayout));
            UAA_CHECK(psoGen != nullptr && psoWork != nullptr, "Failed to create indirect pipelines");

            nvrhi::BufferHandle indBuf = createStructuredBuffer(ctx, dev, 16, true, true, nvrhi::ResourceStates::UnorderedAccess, "cmp_ind_args");
            nvrhi::BufferHandle outBuf = createStructuredBuffer(ctx, dev, kOutWords, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_ind_out");

            nvrhi::BufferHandle cb0 = createConstantBuffer(ctx, dev, sizeof(IndirectGenParams), "cmp_ind_cb0");
            nvrhi::BufferHandle cb1 = createConstantBuffer(ctx, dev, sizeof(IndirectGenParams), "cmp_ind_cb1");
            nvrhi::BufferHandle cb2 = createConstantBuffer(ctx, dev, sizeof(IndirectGenParams), "cmp_ind_cb2");

            const IndirectGenParams gen0{2u, 1u, 1u, 0u, 0u, 0u, 3u, 0u};
            const IndirectGenParams gen1{0u, 1u, 1u, 0u, 0u, 4u, 3u, 0u};
            const IndirectGenParams gen2{2u, 2u, 1u, 0u, 0u, 8u, 3u, 0u};
            fix.uploadBuffer(cb0, &gen0, sizeof(gen0));
            fix.uploadBuffer(cb1, &gen1, sizeof(gen1));
            fix.uploadBuffer(cb2, &gen2, sizeof(gen2));

            nvrhi::BindingSetHandle bsGen0 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb0))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, indBuf)),
                genLayout);
            nvrhi::BindingSetHandle bsGen1 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb1))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, indBuf)),
                genLayout);
            nvrhi::BindingSetHandle bsGen2 = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb2))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, indBuf)),
                genLayout);
            nvrhi::BindingSetHandle bsWork = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(IndirectWorkPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
                workLayout);
            UAA_CHECK(bsGen0 != nullptr && bsGen1 != nullptr && bsGen2 != nullptr && bsWork != nullptr,
                "Failed to create indirect binding sets");

            const IndirectWorkPC pc0{0u, 100u, 8u, 16u};
            const IndirectWorkPC pc1{16u, 0xBAD00000u, 8u, 16u};
            const IndirectWorkPC pc2{32u, 500u, 8u, 16u};

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            cmd->clearBufferUInt(outBuf, TestFixture::kGuardWord);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGen0));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGen1));
            cmd->dispatch(1u, 1u, 1u);
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoGen).addBindingSet(bsGen2));
            cmd->dispatch(1u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoWork).addBindingSet(bsWork).setIndirectParams(indBuf));
            cmd->setPushConstants(&pc0, sizeof(pc0));
            cmd->dispatchIndirect(0u);

            cmd->setPushConstants(&pc1, sizeof(pc1));
            cmd->dispatchIndirect(16u);

            cmd->setPushConstants(&pc2, sizeof(pc2));
            cmd->dispatchIndirect(32u);
            cmd->close();
            fix.submitAndWait(cmd);

            std::vector<uint32_t> actual(kOutWords, 0u);
            fix.readbackBuffer(outBuf, actual.data(), actual.size() * sizeof(uint32_t));

            for (uint32_t i = 0; i < 8u; ++i)
            {
                UAA_CHECK_EQ(actual[i], pc0.marker + i, "CMP.Indirect triple 0 mismatch");
            }
            checkWordSlice(ctx, actual, 8u, 32u, TestFixture::kGuardWord, "CMP.Indirect zero-count/gap guard");
            for (uint32_t i = 0; i < 16u; ++i)
            {
                UAA_CHECK_EQ(actual[32u + i], pc2.marker + i, "CMP.Indirect triple 2 mismatch");
            }
            checkWordSlice(ctx, actual, 48u, kOutWords, TestFixture::kGuardWord, "CMP.Indirect tail guard");
        }

        void testComputeIndirectValidation(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (!fix.isValidationActive())
            {
                ctx.skip("NVRHI validation layer is not active");
            }

            nvrhi::IDevice* dev = fix.device();
            nvrhi::ShaderHandle csWork = fix.createShader("cs_indirect_work", nvrhi::ShaderType::Compute);
            UAA_CHECK(csWork != nullptr, "Failed to create cs_indirect_work");

            nvrhi::BindingLayoutDesc workLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute);
            workLayoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(IndirectWorkPC)))
                .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
            nvrhi::BindingLayoutHandle workLayout = dev->createBindingLayout(workLayoutDesc);
            UAA_CHECK(workLayout != nullptr, "Failed to create workLayout");

            nvrhi::ComputePipelineHandle psoWork = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csWork).addBindingLayout(workLayout));
            UAA_CHECK(psoWork != nullptr, "Failed to create psoWork");

            nvrhi::BufferHandle outBuf = createStructuredBuffer(ctx, dev, 16, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_indval_out");
            nvrhi::BufferHandle nonIndirectBuf = createStructuredBuffer(ctx, dev, 16, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_indval_noflag");
            nvrhi::BindingSetHandle bsWork = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(IndirectWorkPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outBuf)),
                workLayout);
            UAA_CHECK(bsWork != nullptr, "Failed to create bsWork");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            ExpectErrorsGuard guard(fix);
            cmd->open();

            // Calling dispatchIndirect without compute state must report a validation error.
            cmd->dispatchIndirect(0u);

            // Setting compute state without indirectParams then calling dispatchIndirect must fail validation.
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoWork).addBindingSet(bsWork));
            cmd->dispatchIndirect(0u);

            // Binding an indirectParams buffer without isDrawIndirectArgs must fail validation in setComputeState.
            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoWork).addBindingSet(bsWork).setIndirectParams(nonIndirectBuf));

            cmd->close();
            // Validation blocked all invalid backend calls, so submitting the empty command list recycles it cleanly.
            fix.submitAndWait(cmd);

            UAA_CHECK(guard.errorCount() >= 3u, "Expected validation errors for invalid dispatchIndirect setups");
        }

        void testComputeDependencies(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            nvrhi::IDevice* dev = fix.device();

            constexpr uint32_t kCount = 128;
            constexpr size_t kByteSize = kCount * sizeof(uint32_t);
            constexpr uint32_t kAdd0 = 17u;
            constexpr uint32_t kXor0 = 0x5A5A5A5Au;

            nvrhi::ShaderHandle csCopyAdd = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::ShaderHandle csChain = fix.createShader("cs_uav_chain", nvrhi::ShaderType::Compute);
            UAA_CHECK(csCopyAdd != nullptr && csChain != nullptr, "Failed to create dependency shaders");

            nvrhi::BindingLayoutHandle layoutCopy = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            nvrhi::BindingLayoutHandle layoutChain = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)));
            UAA_CHECK(layoutCopy != nullptr && layoutChain != nullptr, "Failed to create layouts");

            nvrhi::ComputePipelineHandle psoCopy = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csCopyAdd).addBindingLayout(layoutCopy));
            nvrhi::ComputePipelineHandle psoChain = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csChain).addBindingLayout(layoutChain));
            UAA_CHECK(psoCopy != nullptr && psoChain != nullptr, "Failed to create pipelines");

            nvrhi::BufferHandle cb = createConstantBuffer(ctx, dev, sizeof(CopyAddParams), "cmp_dep_cb");
            nvrhi::BufferHandle inBuf = createStructuredBuffer(ctx, dev, kCount, false, false, nvrhi::ResourceStates::ShaderResource, "cmp_dep_in");
            nvrhi::BufferHandle workBuf = createStructuredBuffer(ctx, dev, kCount, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_dep_work");

            nvrhi::BufferDesc rbDesc;
            rbDesc.setByteSize(kByteSize)
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setDebugName("cmp_dep_rb");
            nvrhi::BufferHandle rbBuf = dev->createBuffer(rbDesc);
            UAA_CHECK(rbBuf != nullptr, "Failed to create readback buffer");

            nvrhi::BindingSetHandle bsCopy = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, workBuf)),
                layoutCopy);
            nvrhi::BindingSetHandle bsChain = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, workBuf)),
                layoutChain);
            UAA_CHECK(bsCopy != nullptr && bsChain != nullptr, "Failed to create binding sets");

            const std::vector<uint32_t> input = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x3001ull);
            const CopyAddParams params{kCount, kAdd0, kXor0, 0u};
            const UavChainPC step1{kCount, 11u, 3u, 0u};
            const UavChainPC step2{kCount, 29u, 7u, 0u};
            const UavChainPC step3{kCount, 101u, 5u, 0u};

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "Failed to create command list");

            cmd->open();
            cmd->writeBuffer(cb, &params, sizeof(params), 0u);
            cmd->writeBuffer(inBuf, input.data(), kByteSize, 0u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoCopy).addBindingSet(bsCopy));
            cmd->dispatch(kCount / 64u, 1u, 1u);

            cmd->setComputeState(nvrhi::ComputeState().setPipeline(psoChain).addBindingSet(bsChain));
            cmd->setPushConstants(&step1, sizeof(step1));
            cmd->dispatch(kCount / 64u, 1u, 1u);
            cmd->setPushConstants(&step2, sizeof(step2));
            cmd->dispatch(kCount / 64u, 1u, 1u);
            cmd->setPushConstants(&step3, sizeof(step3));
            cmd->dispatch(kCount / 64u, 1u, 1u);

            cmd->copyBuffer(rbBuf, 0u, workBuf, 0u, kByteSize);
            cmd->close();
            fix.submitAndWait(cmd);

            void* mapped = dev->mapBuffer(rbBuf, nvrhi::CpuAccessMode::Read);
            UAA_CHECK(mapped != nullptr, "mapBuffer returned null for readback buffer");
            const auto* actual = static_cast<const uint32_t*>(mapped);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                uint32_t expected = (input[i] + kAdd0) ^ kXor0;
                expected = expected * step1.stepMul + step1.stepAdd;
                expected = expected * step2.stepMul + step2.stepAdd;
                expected = expected * step3.stepMul + step3.stepAdd;
                UAA_CHECK_EQ(actual[i], expected, "CMP.Dependencies chained value mismatch");
            }
            dev->unmapBuffer(rbBuf);
        }

        void testComputeMultiQueue(TestContext& ctx)
        {
            TestFixture& fix = requireFixture(ctx);
            if (!fix.features().supports(nvrhi::Feature::ComputeQueue))
            {
                ctx.skip("ComputeQueue not supported");
            }

            nvrhi::IDevice* dev = fix.device();
            constexpr uint32_t kCount = 128;
            constexpr size_t kByteSize = kCount * sizeof(uint32_t);
            constexpr uint32_t kAdd0 = 23u;
            constexpr uint32_t kXor0 = 0x3C3C3C3Cu;

            nvrhi::ShaderHandle csCopyAdd = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute);
            nvrhi::ShaderHandle csChain = fix.createShader("cs_uav_chain", nvrhi::ShaderType::Compute);
            UAA_CHECK(csCopyAdd != nullptr && csChain != nullptr, "Failed to create multi-queue shaders");

            nvrhi::BindingLayoutHandle layoutCopy = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2)));
            nvrhi::BindingLayoutHandle layoutChain = dev->createBindingLayout(
                fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute)
                    .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)));
            UAA_CHECK(layoutCopy != nullptr && layoutChain != nullptr, "Failed to create layouts");

            nvrhi::ComputePipelineHandle psoCopy = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csCopyAdd).addBindingLayout(layoutCopy));
            nvrhi::ComputePipelineHandle psoChain = dev->createComputePipeline(
                nvrhi::ComputePipelineDesc().setComputeShader(csChain).addBindingLayout(layoutChain));
            UAA_CHECK(psoCopy != nullptr && psoChain != nullptr, "Failed to create pipelines");

            nvrhi::BufferHandle cb = createConstantBuffer(ctx, dev, sizeof(CopyAddParams), "cmp_mq_cb");
            nvrhi::BufferHandle inBuf = createStructuredBuffer(ctx, dev, kCount, false, false, nvrhi::ResourceStates::ShaderResource, "cmp_mq_in");
            nvrhi::BufferHandle workBuf = createStructuredBuffer(ctx, dev, kCount, true, false, nvrhi::ResourceStates::UnorderedAccess, "cmp_mq_work");

            const std::vector<uint32_t> input = TestFixture::makeDeterministicWords(kCount, ctx.seed() ^ 0x4001ull);
            const std::vector<uint32_t> staleWords(kCount, 0xDEAD0001u);
            const CopyAddParams params{kCount, kAdd0, kXor0, 0u};
            const UavChainPC chainPC{kCount, 77u, 9u, 0u};

            fix.uploadBuffer(cb, &params, sizeof(params));
            fix.uploadBuffer(inBuf, input.data(), kByteSize);
            fix.uploadBuffer(workBuf, staleWords.data(), kByteSize);

            nvrhi::BindingSetHandle bsCopy = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, cb))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, inBuf))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, workBuf)),
                layoutCopy);
            nvrhi::BindingSetHandle bsChain = dev->createBindingSet(
                nvrhi::BindingSetDesc()
                    .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(UavChainPC)))
                    .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, workBuf)),
                layoutChain);
            UAA_CHECK(bsCopy != nullptr && bsChain != nullptr, "Failed to create binding sets");

            nvrhi::CommandListHandle cmdCompute = dev->createCommandList(
                nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute));
            nvrhi::CommandListHandle cmdGraphics = dev->createCommandList(
                nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
            UAA_CHECK(cmdCompute != nullptr && cmdGraphics != nullptr, "Failed to create multi-queue command lists");

            cmdCompute->open();
            cmdCompute->setComputeState(nvrhi::ComputeState().setPipeline(psoCopy).addBindingSet(bsCopy));
            cmdCompute->dispatch(kCount / 64u, 1u, 1u);
            cmdCompute->close();

            cmdGraphics->open();
            cmdGraphics->setComputeState(nvrhi::ComputeState().setPipeline(psoChain).addBindingSet(bsChain));
            cmdGraphics->setPushConstants(&chainPC, sizeof(chainPC));
            cmdGraphics->dispatch(kCount / 64u, 1u, 1u);
            cmdGraphics->close();

            // Hold the producer behind a GPU gate when supported so ordering is proven even under delayed execution.
            std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
            const bool gated = gate && gate->enqueueWait(nvrhi::CommandQueue::Compute);

            const uint64_t computeSerial = dev->executeCommandList(cmdCompute, nvrhi::CommandQueue::Compute);
            dev->queueWaitForCommandList(nvrhi::CommandQueue::Graphics, nvrhi::CommandQueue::Compute, computeSerial);
            dev->executeCommandList(cmdGraphics, nvrhi::CommandQueue::Graphics);

            if (gated)
            {
                gate->signal();
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after multi-queue submission");
            dev->runGarbageCollection();

            std::vector<uint32_t> actual(kCount, 0u);
            fix.readbackBuffer(workBuf, actual.data(), kByteSize);

            for (uint32_t i = 0; i < kCount; ++i)
            {
                const uint32_t expected = ((input[i] + kAdd0) ^ kXor0) * chainPC.stepMul + chainPC.stepAdd;
                UAA_CHECK_EQ(actual[i], expected, "CMP.MultiQueue cross-queue chained result mismatch");
            }
        }
    } // namespace

    UAA_REGISTER_TEST("CMP.Direct", "core", testComputeDirect);
    UAA_REGISTER_TEST("CMP.StateChanges", "core", testComputeStateChanges);
    UAA_REGISTER_TEST("CMP.Indirect", "core", testComputeIndirect);
    UAA_REGISTER_TEST("CMP.IndirectValidation", "core", testComputeIndirectValidation);
    UAA_REGISTER_TEST("CMP.Dependencies", "core", testComputeDependencies);
    UAA_REGISTER_TEST("CMP.MultiQueue", "core", testComputeMultiQueue);
} // namespace nvrhi::uaa
