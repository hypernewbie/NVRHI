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

#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace nvrhi::uaa
{
    namespace
    {
        namespace fs = std::filesystem;

        struct CopyAddParams
        {
            uint32_t count = 0;
            uint32_t addVal = 0;
            uint32_t xorVal = 0;
            uint32_t pad = 0;
        };

        struct SolidColorPushConstants
        {
            float color[4] = { 0.f, 0.f, 0.f, 1.f };
            float depth = 0.f;
            float scale = 1.f;
            float offset[2] = { 0.f, 0.f };
        };

        struct ComputeCopyAddResources
        {
            nvrhi::ShaderHandle shader;
            nvrhi::BindingLayoutHandle layout;
            nvrhi::ComputePipelineHandle pipeline;
        };

        struct ComputeSlot
        {
            nvrhi::BufferHandle inBuf;
            nvrhi::BufferHandle outBuf;
            nvrhi::BufferHandle cbBuf;
            nvrhi::BindingSetHandle bindingSet;
            uint32_t wordCount = 0;
            uint32_t addVal = 0;
            uint32_t xorVal = 0;
            std::vector<uint32_t> inputWords;
            std::vector<uint32_t> expectedWords;
        };

        ComputeCopyAddResources createComputeCopyAddPipeline(TestContext& ctx, TestFixture& fix)
        {
            ComputeCopyAddResources res;
            res.shader = fix.createShader("cs_buffer_copy_add", nvrhi::ShaderType::Compute, "main");
            UAA_CHECK(res.shader != nullptr, "Failed to create cs_buffer_copy_add shader");

            nvrhi::BindingLayoutDesc layoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::Compute, 0, false);
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
                      .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
                      .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2));
            res.layout = fix.device()->createBindingLayout(layoutDesc);
            UAA_CHECK(res.layout != nullptr, "Failed to create cs_buffer_copy_add binding layout");

            nvrhi::ComputePipelineDesc pipeDesc;
            pipeDesc.setComputeShader(res.shader)
                    .addBindingLayout(res.layout);
            res.pipeline = fix.device()->createComputePipeline(pipeDesc);
            UAA_CHECK(res.pipeline != nullptr, "Failed to create cs_buffer_copy_add compute pipeline");

            return res;
        }

        ComputeSlot createPopulatedComputeSlot(
            TestContext& ctx,
            TestFixture& fix,
            const ComputeCopyAddResources& res,
            uint32_t wordCount,
            uint32_t addVal,
            uint32_t xorVal,
            uint64_t dataSeed,
            const std::string& debugTag)
        {
            ComputeSlot slot;
            slot.wordCount = wordCount;
            slot.addVal = addVal;
            slot.xorVal = xorVal;

            const uint64_t byteSize = uint64_t(wordCount) * sizeof(uint32_t);
            slot.inputWords = TestFixture::makeDeterministicWords(wordCount, dataSeed);
            slot.expectedWords.resize(wordCount);
            for (uint32_t i = 0; i < wordCount; ++i)
            {
                slot.expectedWords[i] = (slot.inputWords[i] + addVal) ^ xorVal;
            }

            nvrhi::BufferDesc inDesc;
            inDesc.setByteSize(byteSize)
                  .setStructStride(sizeof(uint32_t))
                  .setDebugName(debugTag + "_in")
                  .enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            slot.inBuf = fix.device()->createBuffer(inDesc);
            UAA_CHECK(slot.inBuf != nullptr, "createBuffer failed for input buffer");

            nvrhi::BufferDesc outDesc;
            outDesc.setByteSize(byteSize)
                   .setStructStride(sizeof(uint32_t))
                   .setCanHaveUAVs(true)
                   .setDebugName(debugTag + "_out")
                   .enableAutomaticStateTracking(nvrhi::ResourceStates::Common);
            slot.outBuf = fix.device()->createBuffer(outDesc);
            UAA_CHECK(slot.outBuf != nullptr, "createBuffer failed for output buffer");

            nvrhi::BufferDesc cbDesc = nvrhi::utils::CreateStaticConstantBufferDesc(
                nvrhi::c_ConstantBufferOffsetSizeAlignment,
                (debugTag + "_cb").c_str());
            cbDesc.enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer);
            slot.cbBuf = fix.device()->createBuffer(cbDesc);
            UAA_CHECK(slot.cbBuf != nullptr, "createBuffer failed for constant buffer");

            fix.uploadBuffer(slot.inBuf, slot.inputWords.data(), byteSize);

            CopyAddParams params{};
            params.count = wordCount;
            params.addVal = addVal;
            params.xorVal = xorVal;
            fix.uploadBuffer(slot.cbBuf, &params, sizeof(params));

            nvrhi::BindingSetDesc bsDesc;
            bsDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, slot.cbBuf))
                  .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, slot.inBuf))
                  .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, slot.outBuf));
            slot.bindingSet = fix.device()->createBindingSet(bsDesc, res.layout);
            UAA_CHECK(slot.bindingSet != nullptr, "createBindingSet failed");

            return slot;
        }

        void recordComputeSlot(
            nvrhi::ICommandList* cmd,
            const ComputeCopyAddResources& res,
            const ComputeSlot& slot)
        {
            nvrhi::ComputeState state;
            state.setPipeline(res.pipeline)
                 .addBindingSet(slot.bindingSet);
            cmd->setComputeState(state);
            const uint32_t groups = (slot.wordCount + 63u) / 64u;
            cmd->dispatch(groups, 1, 1);
        }

        void verifyComputeSlotReadback(
            TestContext& ctx,
            TestFixture& fix,
            const ComputeSlot& slot,
            const std::string& failurePrefix)
        {
            std::vector<uint32_t> actual(slot.wordCount, 0xDEADBEEFu);
            fix.readbackBuffer(slot.outBuf, actual.data(), size_t(slot.wordCount) * sizeof(uint32_t));

            for (uint32_t i = 0; i < slot.wordCount; ++i)
            {
                if (actual[i] != slot.expectedWords[i])
                {
                    std::ostringstream oss;
                    oss << failurePrefix
                        << " mismatch at word " << i
                        << " (count=" << slot.wordCount
                        << ", addVal=0x" << std::hex << slot.addVal
                        << ", xorVal=0x" << slot.xorVal
                        << ", input=0x" << slot.inputWords[i]
                        << ", actual=0x" << actual[i]
                        << ", expected=0x" << slot.expectedWords[i] << std::dec << ")";
                    ctx.fail(oss.str(), __FILE__, __LINE__);
                }
            }
        }

        class GpuGateGuard
        {
        public:
            explicit GpuGateGuard(IGpuGate* gate) : m_Gate(gate) {}
            ~GpuGateGuard()
            {
                release();
            }

            void arm() { m_Armed = true; }
            void release()
            {
                if (m_Gate && m_Armed)
                {
                    m_Gate->signal();
                    m_Armed = false;
                }
            }

        private:
            IGpuGate* m_Gate = nullptr;
            bool m_Armed = false;
        };

        class TempDirGuard
        {
        public:
            explicit TempDirGuard(fs::path path) : m_Path(std::move(path)) {}
            ~TempDirGuard()
            {
                if (!m_Path.empty())
                {
                    std::error_code ec;
                    fs::remove_all(m_Path, ec);
                }
            }

            const fs::path& path() const { return m_Path; }

        private:
            fs::path m_Path;
        };

        std::string readTextFile(const fs::path& path)
        {
            std::ifstream f(path, std::ios::in | std::ios::binary);
            if (!f.is_open())
            {
                return {};
            }
            std::ostringstream oss;
            oss << f.rdbuf();
            return oss.str();
        }

        bool runSubprocess(
            const std::vector<std::string>& args,
            uint32_t safetyTimeoutMs,
            const fs::path& logFilePath,
            int& outExitCode,
            double& outElapsedSec,
            bool& outTimedOut,
            std::string& outLaunchError)
        {
            outExitCode = -1;
            outElapsedSec = 0.0;
            outTimedOut = false;
            outLaunchError.clear();

            if (args.empty())
            {
                outLaunchError = "Empty subprocess argument list";
                return false;
            }

            const auto t0 = std::chrono::steady_clock::now();

#ifdef _WIN32
            auto quoteArg = [](const std::string& arg) -> std::string
            {
                if (arg.empty() || arg.find_first_of(" \t\n\v\"") != std::string::npos)
                {
                    std::string out = "\"";
                    size_t backslashes = 0;
                    for (char ch : arg)
                    {
                        if (ch == '\\')
                        {
                            ++backslashes;
                        }
                        else if (ch == '"')
                        {
                            out.append(backslashes * 2 + 1, '\\');
                            out.push_back('"');
                            backslashes = 0;
                        }
                        else
                        {
                            out.append(backslashes, '\\');
                            backslashes = 0;
                            out.push_back(ch);
                        }
                    }
                    out.append(backslashes * 2, '\\');
                    out.push_back('"');
                    return out;
                }
                return arg;
            };

            std::string cmdLine;
            for (size_t i = 0; i < args.size(); ++i)
            {
                if (i > 0) cmdLine.push_back(' ');
                cmdLine += quoteArg(args[i]);
            }

            SECURITY_ATTRIBUTES sa{};
            sa.nLength = sizeof(sa);
            sa.bInheritHandle = TRUE;

            const std::string redirectTarget = logFilePath.empty() ? std::string("NUL") : logFilePath.string();
            HANDLE logHandle = CreateFileA(
                redirectTarget.c_str(),
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                &sa,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);

            STARTUPINFOA si{};
            si.cb = sizeof(si);
            if (logHandle != INVALID_HANDLE_VALUE)
            {
                si.dwFlags = STARTF_USESTDHANDLES;
                si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
                si.hStdOutput = logHandle;
                si.hStdError = logHandle;
            }

            PROCESS_INFORMATION pi{};
            std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
            mutableCmd.push_back('\0');

            const BOOL ok = CreateProcessA(
                nullptr,
                mutableCmd.data(),
                nullptr,
                nullptr,
                logHandle != INVALID_HANDLE_VALUE ? TRUE : FALSE,
                0,
                nullptr,
                nullptr,
                &si,
                &pi);

            if (logHandle != INVALID_HANDLE_VALUE)
            {
                CloseHandle(logHandle);
            }

            if (!ok)
            {
                outLaunchError = "CreateProcessA failed with error " + std::to_string(GetLastError());
                return false;
            }

            const DWORD waitRes = WaitForSingleObject(pi.hProcess, safetyTimeoutMs);
            if (waitRes == WAIT_TIMEOUT)
            {
                outTimedOut = true;
                TerminateProcess(pi.hProcess, 124);
                WaitForSingleObject(pi.hProcess, 2000);
            }

            DWORD exitCodeDword = 0;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            outExitCode = static_cast<int>(exitCodeDword);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
#else
            std::vector<std::string> argStorage = args;
            std::vector<char*> argv;
            argv.reserve(argStorage.size() + 1);
            for (std::string& s : argStorage)
            {
                argv.push_back(const_cast<char*>(s.c_str()));
            }
            argv.push_back(nullptr);

            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            const std::string targetPath = logFilePath.empty() ? std::string("/dev/null") : logFilePath.string();
            posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, targetPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, targetPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

            pid_t pid = 0;
            const int spawnRc = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
            posix_spawn_file_actions_destroy(&actions);

            if (spawnRc != 0)
            {
                outLaunchError = std::string("posix_spawnp failed: ") + std::strerror(spawnRc);
                return false;
            }

            int status = 0;
            const auto deadline = t0 + std::chrono::milliseconds(safetyTimeoutMs);
            while (true)
            {
                const pid_t w = waitpid(pid, &status, WNOHANG);
                if (w == pid)
                {
                    break;
                }
                if (w < 0)
                {
                    outLaunchError = "waitpid failed";
                    return false;
                }
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    outTimedOut = true;
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            if (WIFEXITED(status))
            {
                outExitCode = WEXITSTATUS(status);
            }
            else if (WIFSIGNALED(status))
            {
                outExitCode = 128 + WTERMSIG(status);
            }
#endif

            const auto t1 = std::chrono::steady_clock::now();
            outElapsedSec = std::chrono::duration<double>(t1 - t0).count();
            return true;
        }

        bool fileContainsToken(const fs::path& filePath, const std::string& token)
        {
            const std::string contents = readTextFile(filePath);
            return !contents.empty() && contents.find(token) != std::string::npos;
        }

        std::string readCacheValue(const fs::path& cachePath, const std::string& keyPrefix)
        {
            std::ifstream f(cachePath);
            if (!f.is_open())
            {
                return {};
            }
            std::string line;
            while (std::getline(f, line))
            {
                if (line.rfind(keyPrefix, 0) == 0)
                {
                    return line.substr(keyPrefix.size());
                }
            }
            return {};
        }

        fs::path findNvrhiLibrary(const fs::path& installDir, const fs::path& binDir, bool isShared, const std::string& buildConfig)
        {
            std::vector<std::string> candidateNames;
#ifdef _WIN32
            candidateNames.push_back("nvrhi.lib");
#elif defined(__APPLE__)
            if (isShared)
            {
                candidateNames.push_back("libnvrhi.dylib");
            }
            else
            {
                candidateNames.push_back("libnvrhi.a");
            }
#else
            if (isShared)
            {
                candidateNames.push_back("libnvrhi.so");
            }
            else
            {
                candidateNames.push_back("libnvrhi.a");
            }
#endif

            if (!installDir.empty())
            {
                for (const auto& name : candidateNames)
                {
                    const fs::path candidate = installDir / "lib" / name;
                    if (fs::is_regular_file(candidate))
                    {
                        return candidate;
                    }
                }
                return {};
            }

            std::vector<fs::path> searchDirs;
            if (!buildConfig.empty())
            {
                searchDirs.push_back(binDir / buildConfig);
                searchDirs.push_back(binDir / "lib" / buildConfig);
            }
            searchDirs.push_back(binDir);
            searchDirs.push_back(binDir / "lib");
            searchDirs.push_back(binDir / "Release");
            searchDirs.push_back(binDir / "Debug");
            searchDirs.push_back(binDir / "RelWithDebInfo");

            for (const std::string& name : candidateNames)
            {
                for (const fs::path& dir : searchDirs)
                {
                    std::error_code ec;
                    const fs::path candidate = dir / name;
                    if (fs::is_regular_file(candidate, ec))
                    {
                        return candidate;
                    }
                }
            }

            // Shallow fallback scan inside the binary directory for multi-config layouts.
            std::error_code ec;
            for (const std::string& name : candidateNames)
            {
                fs::recursive_directory_iterator it(binDir, fs::directory_options::skip_permission_denied, ec);
                fs::recursive_directory_iterator end;
                for (; it != end && !ec; it.increment(ec))
                {
                    if (it.depth() > 3)
                    {
                        it.disable_recursion_pending();
                        continue;
                    }
                    if (it->is_regular_file(ec) && it->path().filename() == name)
                    {
                        return it->path();
                    }
                }
            }

            return {};
        }

        // STR.Repeat: deterministic create/upload/compute/readback/destroy churn across 25 cycles.
        void testStressRepeat(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "STR.Repeat requires a GPU fixture");
            TestFixture& fix = *ctx.fixture;

            ComputeCopyAddResources res = createComputeCopyAddPipeline(ctx, fix);
            std::mt19937_64 rng(ctx.seed());

            constexpr uint32_t kCycles = 25;
            for (uint32_t iter = 0; iter < kCycles; ++iter)
            {
                const uint32_t wordCount = 16u + static_cast<uint32_t>(rng() % (1024u - 16u + 1u));
                const uint32_t addVal = static_cast<uint32_t>(rng());
                const uint32_t xorVal = static_cast<uint32_t>(rng());
                const uint64_t dataSeed = ctx.seed() ^ (uint64_t(iter + 1u) * 0x9E3779B97F4A7C15ull);

                ComputeSlot slot = createPopulatedComputeSlot(
                    ctx, fix, res, wordCount, addVal, xorVal, dataSeed,
                    "str_repeat_" + std::to_string(iter));

                nvrhi::CommandListHandle cmd = fix.device()->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed in STR.Repeat");
                cmd->open();
                recordComputeSlot(cmd, res, slot);
                cmd->close();
                fix.submitAndWait(cmd);

                std::ostringstream prefix;
                prefix << "STR.Repeat (seed=0x" << std::hex << ctx.seed() << std::dec
                       << ", iter=" << iter << ", byteSize=" << (wordCount * sizeof(uint32_t)) << ")";
                verifyComputeSlotReadback(ctx, fix, slot, prefix.str());
            }
        }

        // STR.InFlight: rotate 4 independent submissions while holding slot 0 behind IGpuGate when available.
        void testStressInFlight(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "STR.InFlight requires a GPU fixture");
            TestFixture& fix = *ctx.fixture;

            ComputeCopyAddResources res = createComputeCopyAddPipeline(ctx, fix);

            constexpr uint32_t kSlotCount = 4;
            constexpr uint32_t kWordCount = 256;
            std::vector<ComputeSlot> slots(kSlotCount);
            std::vector<nvrhi::CommandListHandle> cmds(kSlotCount);

            for (uint32_t i = 0; i < kSlotCount; ++i)
            {
                const uint32_t addVal = 0x100u + i * 0x17u;
                const uint32_t xorVal = 0x5A5A0000u ^ (i * 0x1111u);
                const uint64_t dataSeed = ctx.seed() ^ (uint64_t(i + 1u) * 0xA0761D6478BD642Full);
                slots[i] = createPopulatedComputeSlot(
                    ctx, fix, res, kWordCount, addVal, xorVal, dataSeed,
                    "str_inflight_" + std::to_string(i));

                cmds[i] = fix.device()->createCommandList();
                UAA_CHECK(cmds[i] != nullptr, "createCommandList failed in STR.InFlight");
            }

            std::unique_ptr<IGpuGate> gate = fix.backend() ? fix.backend()->createGpuGate() : nullptr;
            bool gateUsed = false;
            GpuGateGuard gateGuard(gate.get());

            if (gate && gate->enqueueWait(nvrhi::CommandQueue::Graphics))
            {
                gateUsed = true;
                gateGuard.arm();
            }
            ctx.note(gateUsed
                ? "Held GPU queue behind IGpuGate across 4 in-flight submissions"
                : "IGpuGate unavailable; submitted 4 back-to-back async command lists without intermediate waitForIdle");

            for (uint32_t i = 0; i < kSlotCount; ++i)
            {
                cmds[i]->open();
                recordComputeSlot(cmds[i], res, slots[i]);
                cmds[i]->close();
                fix.device()->executeCommandList(cmds[i], nvrhi::CommandQueue::Graphics);
            }

            gateGuard.release();

            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed in STR.InFlight");
            fix.device()->runGarbageCollection();

            for (uint32_t i = 0; i < kSlotCount; ++i)
            {
                verifyComputeSlotReadback(ctx, fix, slots[i], "STR.InFlight slot " + std::to_string(i));
            }
        }

        // STR.Resize: repeatedly create, resize, shrink, and write descriptor tables and textures.
        void testStressResize(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "STR.Resize requires a GPU fixture");
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();

            const bool supportsBindless = (fix.api() != nvrhi::GraphicsAPI::D3D11);
            nvrhi::BindingLayoutHandle bindlessLayout;
            if (supportsBindless)
            {
                nvrhi::BindlessLayoutDesc bld;
                bld.setVisibility(nvrhi::ShaderType::Compute)
                   .setFirstSlot(0)
                   .setMaxCapacity(128)
                   .setUpdateAfterBind(false)
                   .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0));
                bindlessLayout = dev->createBindlessLayout(bld);
                UAA_CHECK(bindlessLayout != nullptr, "createBindlessLayout failed in STR.Resize");
            }
            else
            {
                ctx.note("D3D11 does not support bindless descriptor tables; exercising texture resize/churn cycles only");
            }

            constexpr uint32_t kCycles = 20;
            for (uint32_t cycle = 0; cycle < kCycles; ++cycle)
            {
                const uint32_t width = 16u + (cycle % 4u) * 16u;
                const uint32_t height = 16u + ((cycle + 1u) % 4u) * 16u;

                nvrhi::TextureDesc texDesc;
                texDesc.setWidth(width)
                       .setHeight(height)
                       .setFormat(nvrhi::Format::RGBA8_UNORM)
                       .setIsUAV(true)
                       .setDebugName("str_resize_tex_" + std::to_string(cycle))
                       .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource);
                nvrhi::TextureHandle tex = dev->createTexture(texDesc);
                UAA_CHECK(tex != nullptr, "createTexture failed in STR.Resize");

                if (supportsBindless)
                {
                    nvrhi::DescriptorTableHandle table = dev->createDescriptorTable(bindlessLayout);
                    UAA_CHECK(table != nullptr, "createDescriptorTable failed in STR.Resize");

                    dev->resizeDescriptorTable(table, 128, false);
                    for (uint32_t slot : { 0u, 1u, 7u, 15u, 63u, 127u })
                    {
                        const bool wrote = dev->writeDescriptorTable(table, nvrhi::BindingSetItem::Texture_SRV(slot, tex));
                        UAA_CHECK(wrote, "writeDescriptorTable failed at capacity 128");
                    }

                    dev->resizeDescriptorTable(table, 16, true);
                    for (uint32_t slot : { 0u, 3u, 15u })
                    {
                        const bool wrote = dev->writeDescriptorTable(table, nvrhi::BindingSetItem::Texture_SRV(slot, tex));
                        UAA_CHECK(wrote, "writeDescriptorTable failed after shrinking to 16");
                    }

                    dev->resizeDescriptorTable(table, 64, true);
                    for (uint32_t slot : { 0u, 31u })
                    {
                        const bool wrote = dev->writeDescriptorTable(table, nvrhi::BindingSetItem::Texture_SRV(slot, tex));
                        UAA_CHECK(wrote, "writeDescriptorTable failed after regrowing to 64");
                    }

                    table = nullptr;
                }

                tex = nullptr;
                if ((cycle % 5u) == 4u)
                {
                    UAA_CHECK(dev->waitForIdle(), "waitForIdle failed in STR.Resize loop");
                    dev->runGarbageCollection();
                }
            }

            UAA_CHECK(dev->waitForIdle(), "waitForIdle failed after STR.Resize loop");
            dev->runGarbageCollection();

            ComputeCopyAddResources res = createComputeCopyAddPipeline(ctx, fix);
            ComputeSlot verifySlot = createPopulatedComputeSlot(
                ctx, fix, res, 128, 7u, 0x12345678u, ctx.seed() ^ 0x55AAu, "str_resize_verify");

            nvrhi::CommandListHandle cmd = dev->createCommandList();
            UAA_CHECK(cmd != nullptr, "createCommandList failed in STR.Resize verification");
            cmd->open();
            recordComputeSlot(cmd, res, verifySlot);
            cmd->close();
            fix.submitAndWait(cmd);

            verifyComputeSlotReadback(ctx, fix, verifySlot, "STR.Resize post-churn compute verification");
        }

        // STR.Threads: concurrent command-list recording across worker threads when DeferredCommandLists is supported.
        void testStressThreads(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "STR.Threads requires a GPU fixture");
            TestFixture& fix = *ctx.fixture;

            if (!fix.features().supports(nvrhi::Feature::DeferredCommandLists))
            {
                ctx.note("DeferredCommandLists not supported; verifying thread-safe MessageCapture and skipping multithreaded command-list recording");

                MessageCapture localCapture;
                constexpr int kThreads = 4;
                constexpr int kMsgsPerThread = 25;
                std::vector<std::thread> workers;
                workers.reserve(kThreads);
                for (int t = 0; t < kThreads; ++t)
                {
                    workers.emplace_back([&localCapture, t]()
                    {
                        for (int m = 0; m < kMsgsPerThread; ++m)
                        {
                            localCapture.message(nvrhi::MessageSeverity::Info, ("worker_" + std::to_string(t) + "_" + std::to_string(m)).c_str());
                        }
                    });
                }
                for (auto& w : workers)
                {
                    w.join();
                }
                UAA_CHECK_EQ(localCapture.rawMessageCount(), size_t(kThreads * kMsgsPerThread),
                    "MessageCapture lost messages under concurrent writes");
                ctx.skip("DeferredCommandLists not supported");
            }

            ComputeCopyAddResources res = createComputeCopyAddPipeline(ctx, fix);

            constexpr uint32_t kNumThreads = 4;
            constexpr uint32_t kWordCount = 256;
            std::vector<ComputeSlot> slots(kNumThreads);
            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                slots[i] = createPopulatedComputeSlot(
                    ctx, fix, res, kWordCount,
                    0x20u + i * 11u,
                    0xCAFE0000u ^ (i * 0x101u),
                    ctx.seed() ^ (uint64_t(i + 1u) * 0x517CC1B727220A95ull),
                    "str_threads_" + std::to_string(i));
            }

            std::vector<nvrhi::CommandListLifetimeTrackerHandle> trackers(kNumThreads);
            std::vector<nvrhi::CommandListHandle> cmds(kNumThreads);
            std::vector<std::string> workerErrors(kNumThreads);

            std::vector<std::thread> workers;
            workers.reserve(kNumThreads);
            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                workers.emplace_back([&, i]()
                {
                    try
                    {
                        nvrhi::IDevice* dev = fix.device();
                        if (fix.api() != nvrhi::GraphicsAPI::D3D11)
                        {
                            trackers[i] = dev->createCommandListLifetimeTracker(nvrhi::CommandQueue::Graphics);
                        }

                        nvrhi::CommandListParameters clParams;
                        clParams.setEnableImmediateExecution(false)
                                .setQueueType(nvrhi::CommandQueue::Graphics)
                                .setLifetimeTracker(trackers[i].Get());

                        cmds[i] = dev->createCommandList(clParams);
                        if (!cmds[i])
                        {
                            workerErrors[i] = "createCommandList returned null";
                            return;
                        }

                        cmds[i]->open();
                        recordComputeSlot(cmds[i], res, slots[i]);
                        cmds[i]->close();
                    }
                    catch (const std::exception& ex)
                    {
                        workerErrors[i] = ex.what();
                    }
                    catch (...)
                    {
                        workerErrors[i] = "unknown exception in worker thread";
                    }
                });
            }

            for (auto& w : workers)
            {
                w.join();
            }

            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                UAA_CHECK(workerErrors[i].empty(), "Worker thread " + std::to_string(i) + " failed: " + workerErrors[i]);
                UAA_CHECK(cmds[i] != nullptr, "Worker thread " + std::to_string(i) + " produced null command list");
            }

            std::vector<nvrhi::ICommandList*> rawCmds(kNumThreads);
            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                rawCmds[i] = cmds[i].Get();
            }

            fix.device()->executeCommandLists(rawCmds.data(), rawCmds.size(), nvrhi::CommandQueue::Graphics);
            UAA_CHECK(fix.device()->waitForIdle(), "waitForIdle failed in STR.Threads");

            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                if (trackers[i])
                {
                    trackers[i]->runGarbageCollection();
                }
            }
            fix.device()->runGarbageCollection();

            for (uint32_t i = 0; i < kNumThreads; ++i)
            {
                verifyComputeSlotReadback(ctx, fix, slots[i], "STR.Threads worker " + std::to_string(i));
            }
        }

        // STR.Timeouts: spawn runner self-tests to verify fail (1), skip (77), and watchdog timeout abort (2).
        void testStressTimeouts(TestContext& ctx)
        {
            if (ctx.options.executablePath.empty())
            {
                ctx.skip("executablePath not provided by host environment");
            }

            const std::string& exe = ctx.options.executablePath;
            int exitCode = -1;
            double elapsedSec = 0.0;
            bool timedOut = false;
            std::string launchErr;

            UAA_CHECK(runSubprocess({ exe, "--self-test-fail" }, 10000, {}, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to launch --self-test-fail subprocess: " + launchErr);
            UAA_CHECK(!timedOut, "--self-test-fail subprocess unexpectedly timed out");
            UAA_CHECK_EQ(exitCode, kExitFail, "--self-test-fail must exit with code 1");

            UAA_CHECK(runSubprocess({ exe, "--self-test-skip" }, 10000, {}, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to launch --self-test-skip subprocess: " + launchErr);
            UAA_CHECK(!timedOut, "--self-test-skip subprocess unexpectedly timed out");
            UAA_CHECK_EQ(exitCode, kExitSkip, "--self-test-skip must exit with code 77");

            UAA_CHECK(runSubprocess({ exe, "--self-test-timeout", "--timeout", "1" }, 10000, {}, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to launch --self-test-timeout subprocess: " + launchErr);
            UAA_CHECK(!timedOut, "--self-test-timeout child stalled past parent safety timeout");
            UAA_CHECK_EQ(exitCode, 2, "--self-test-timeout must abort with watchdog exit code 2");
            UAA_CHECK(elapsedSec < 5.0,
                "Watchdog timeout abort took too long (" + std::to_string(elapsedSec) + "s)");
        }

        // STR.Soak: 200-iteration mixed compute + graphics soak loop gated behind --allow-destructive.
        void testStressSoak(TestContext& ctx)
        {
            UAA_CHECK(ctx.fixture != nullptr, "STR.Soak requires a GPU fixture");
            TestFixture& fix = *ctx.fixture;
            nvrhi::IDevice* dev = fix.device();

            ComputeCopyAddResources computeRes = createComputeCopyAddPipeline(ctx, fix);

            nvrhi::ShaderHandle vs = fix.createShader("vs_fullscreen_tri", nvrhi::ShaderType::Vertex, "main");
            nvrhi::ShaderHandle ps = fix.createShader("ps_solid_color", nvrhi::ShaderType::Pixel, "main");
            UAA_CHECK(vs != nullptr && ps != nullptr, "Failed to create graphics shaders in STR.Soak");

            nvrhi::BindingLayoutDesc gfxLayoutDesc = fix.makeBindingLayoutDesc(nvrhi::ShaderType::AllGraphics, 0, false);
            gfxLayoutDesc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(SolidColorPushConstants)));
            nvrhi::BindingLayoutHandle gfxLayout = dev->createBindingLayout(gfxLayoutDesc);
            UAA_CHECK(gfxLayout != nullptr, "createBindingLayout failed in STR.Soak");

            nvrhi::BindingSetDesc gfxBsDesc;
            gfxBsDesc.addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(SolidColorPushConstants)));
            nvrhi::BindingSetHandle gfxBs = dev->createBindingSet(gfxBsDesc, gfxLayout);
            UAA_CHECK(gfxBs != nullptr, "createBindingSet failed in STR.Soak");

            nvrhi::FramebufferInfo fbInfo;
            fbInfo.addColorFormat(nvrhi::Format::RGBA8_UNORM);

            nvrhi::GraphicsPipelineDesc gfxPipeDesc;
            gfxPipeDesc.setVertexShader(vs)
                       .setPixelShader(ps)
                       .addBindingLayout(gfxLayout);
            gfxPipeDesc.renderState.rasterState.setCullMode(nvrhi::RasterCullMode::None);
            gfxPipeDesc.renderState.depthStencilState.setDepthTestEnable(false).setDepthWriteEnable(false);
            nvrhi::GraphicsPipelineHandle gfxPipe = dev->createGraphicsPipeline(gfxPipeDesc, fbInfo);
            UAA_CHECK(gfxPipe != nullptr, "createGraphicsPipeline failed in STR.Soak");

            const DeviceInfo& info = fix.deviceInfo();
            ctx.note("Device=" + info.deviceName + ", Driver=" + info.driverName + " " + info.driverVersion +
                     ", OS=" + info.osName + ", Backend=" + info.backendName);

            const auto soakStart = std::chrono::steady_clock::now();
            std::mt19937_64 rng(ctx.seed());

            constexpr uint32_t kIterations = 200;
            for (uint32_t iter = 0; iter < kIterations; ++iter)
            {
                const uint32_t wordCount = 32u + static_cast<uint32_t>(rng() % 512u);
                const uint32_t addVal = static_cast<uint32_t>(rng());
                const uint32_t xorVal = static_cast<uint32_t>(rng());
                ComputeSlot slot = createPopulatedComputeSlot(
                    ctx, fix, computeRes, wordCount, addVal, xorVal,
                    ctx.seed() ^ (uint64_t(iter + 1u) * 0xD6E8FEB86659FD93ull),
                    "str_soak_" + std::to_string(iter));

                nvrhi::TextureDesc rtDesc;
                rtDesc.setWidth(16)
                      .setHeight(16)
                      .setFormat(nvrhi::Format::RGBA8_UNORM)
                      .setIsRenderTarget(true)
                      .setDebugName("str_soak_rt_" + std::to_string(iter))
                      .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget);
                nvrhi::TextureHandle rt = dev->createTexture(rtDesc);
                UAA_CHECK(rt != nullptr, "createTexture failed in STR.Soak");

                nvrhi::FramebufferDesc fbDesc;
                fbDesc.addColorAttachment(rt);
                nvrhi::FramebufferHandle fb = dev->createFramebuffer(fbDesc);
                UAA_CHECK(fb != nullptr, "createFramebuffer failed in STR.Soak");

                const uint8_t r8 = static_cast<uint8_t>(32u + (iter * 7u) % 192u);
                const uint8_t g8 = static_cast<uint8_t>(48u + (iter * 13u) % 180u);
                const uint8_t b8 = static_cast<uint8_t>(64u + (iter * 19u) % 160u);

                SolidColorPushConstants pc{};
                pc.color[0] = float(r8) / 255.f;
                pc.color[1] = float(g8) / 255.f;
                pc.color[2] = float(b8) / 255.f;
                pc.color[3] = 1.0f;
                pc.scale = 1.0f;

                nvrhi::CommandListHandle cmd = dev->createCommandList();
                UAA_CHECK(cmd != nullptr, "createCommandList failed in STR.Soak");
                cmd->open();
                recordComputeSlot(cmd, computeRes, slot);

                nvrhi::utils::ClearColorAttachment(cmd, fb, 0, nvrhi::Color(0.f, 0.f, 0.f, 1.f));
                nvrhi::GraphicsState gs;
                gs.setPipeline(gfxPipe)
                  .setFramebuffer(fb)
                  .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(nvrhi::Viewport(16.f, 16.f)))
                  .addBindingSet(gfxBs);
                cmd->setGraphicsState(gs);
                cmd->setPushConstants(&pc, sizeof(pc));
                cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
                cmd->close();

                fix.submitAndWait(cmd);

                std::ostringstream prefix;
                prefix << "STR.Soak (seed=0x" << std::hex << ctx.seed() << std::dec << ", iter=" << iter << ")";
                verifyComputeSlotReadback(ctx, fix, slot, prefix.str());

                std::vector<uint8_t> pixels;
                fix.readbackTexture2D(rt, 0, 0, pixels);
                UAA_CHECK_EQ(pixels.size(), size_t(16 * 16 * 4), "Unexpected RT pixel byte count in STR.Soak");

                std::string pixelDiff;
                const size_t centerOffset = (8u * 16u + 8u) * 4u;
                UAA_CHECK(TestFixture::checkPixelRgba8(pixels.data() + centerOffset, r8, g8, b8, 255, 2, pixelDiff),
                    prefix.str() + " graphics readback: " + pixelDiff);
            }

            const double elapsedMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - soakStart).count();
            std::ostringstream noteOss;
            noteOss << "Completed " << kIterations << " mixed compute+graphics cycles in "
                    << std::fixed << std::setprecision(1) << elapsedMs << " ms";
            ctx.note(noteOss.str());
        }

        // PKG.Build: verify UAA test targets are excluded from install manifests and build a standalone consumer.
        void testPackageBuild(TestContext& ctx)
        {
#if !defined(UAA_NVRHI_SOURCE_DIR) || !defined(UAA_NVRHI_BINARY_DIR)
            ctx.skip("UAA_NVRHI_SOURCE_DIR / UAA_NVRHI_BINARY_DIR not defined");
#else
            const fs::path nvrhiSrcDir(UAA_NVRHI_SOURCE_DIR);
            const fs::path nvrhiBinDir(UAA_NVRHI_BINARY_DIR);
            UAA_CHECK(fs::exists(nvrhiSrcDir / "include" / "nvrhi" / "nvrhi.h"),
                "UAA_NVRHI_SOURCE_DIR does not contain include/nvrhi/nvrhi.h");
            UAA_CHECK(fs::exists(nvrhiBinDir), "UAA_NVRHI_BINARY_DIR does not exist");

            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path tempRoot = fs::temp_directory_path() / ("nvrhi_uaa_pkg_" + std::to_string(nonce));
            TempDirGuard tempGuard(tempRoot);

            std::error_code ec;
            UAA_CHECK(fs::create_directories(tempRoot, ec), "Failed to create temporary packaging test directory");

            // Ensure no install script or export manifest references the UAA test executables.
            for (const fs::path& rel : {
                fs::path("cmake_install.cmake"),
                fs::path("tests/cmake_install.cmake"),
                fs::path("tests/uaa/cmake_install.cmake") })
            {
                const fs::path p = nvrhiBinDir / rel;
                if (fs::is_regular_file(p, ec))
                {
                    UAA_CHECK(!fileContainsToken(p, "nvrhi_uaa_tests"),
                        "nvrhi_uaa_tests leaked into install script: " + p.string());
                    UAA_CHECK(!fileContainsToken(p, "nvrhi_uaa_cpu"),
                        "nvrhi_uaa_cpu leaked into install script: " + p.string());
                }
            }

            {
                fs::recursive_directory_iterator it(nvrhiBinDir, fs::directory_options::skip_permission_denied, ec);
                fs::recursive_directory_iterator end;
                for (; it != end && !ec; it.increment(ec))
                {
                    if (it.depth() > 3)
                    {
                        it.disable_recursion_pending();
                        continue;
                    }
                    if (!it->is_regular_file(ec))
                    {
                        continue;
                    }
                    const std::string fname = it->path().filename().string();
                    if (fname.find("Targets") != std::string::npos && it->path().extension() == ".cmake")
                    {
                        UAA_CHECK(!fileContainsToken(it->path(), "nvrhi_uaa_tests"),
                            "nvrhi_uaa_tests leaked into CMake export manifest: " + it->path().string());
                        UAA_CHECK(!fileContainsToken(it->path(), "nvrhi_uaa_cpu"),
                            "nvrhi_uaa_cpu leaked into CMake export manifest: " + it->path().string());
                    }
                }
            }

            fs::path cachePath = nvrhiBinDir / "CMakeCache.txt";
            if (!fs::is_regular_file(cachePath, ec) && fs::is_regular_file(nvrhiBinDir.parent_path().parent_path() / "CMakeCache.txt", ec))
            {
                cachePath = nvrhiBinDir.parent_path().parent_path() / "CMakeCache.txt";
            }

            bool isShared = false;
#if defined(UAA_NVRHI_SHARED) && UAA_NVRHI_SHARED
            isShared = true;
#endif
            const std::string cacheShared = readCacheValue(cachePath, "NVRHI_BUILD_SHARED:BOOL=");
            if (cacheShared == "ON" || cacheShared == "TRUE" || cacheShared == "1")
            {
                isShared = true;
            }

            std::string buildConfig;
#ifdef UAA_BUILD_CONFIG
            buildConfig = UAA_BUILD_CONFIG;
#endif

            fs::path installDir;
            const fs::path topInstallScript = nvrhiBinDir / "cmake_install.cmake";
            const std::string cacheInstall = readCacheValue(cachePath, "NVRHI_INSTALL:BOOL=");
            if (cacheInstall == "ON" && fs::is_regular_file(topInstallScript, ec) &&
                fileContainsToken(topInstallScript, "include/nvrhi"))
            {
                const fs::path configuredPrefix = readCacheValue(cachePath, "CMAKE_INSTALL_PREFIX:PATH=");
                UAA_CHECK(configuredPrefix.is_absolute(), "CMake install prefix must be an absolute path");
#ifdef _WIN32
                // CMake cannot relocate absolute Windows destinations with DESTDIR.
                // Keep this check safe until the existing header install rule is relocatable.
                ctx.note("Installed consumer check unavailable on Windows with absolute header install destinations");
#else
                const fs::path stagingDir = tempRoot / "stage";
                installDir = stagingDir / configuredPrefix.relative_path();
                std::vector<std::string> installArgs = {
                    "cmake", "-E", "env", "DESTDIR=" + stagingDir.string(),
                    "cmake", "--install", nvrhiBinDir.string()
                };
                if (!buildConfig.empty())
                {
                    installArgs.push_back("--config");
                    installArgs.push_back(buildConfig);
                }

                int exitCode = -1;
                double elapsedSec = 0.0;
                bool timedOut = false;
                std::string launchErr;
                const fs::path installLog = tempRoot / "install.log";
                UAA_CHECK(runSubprocess(installArgs, 30000, installLog, exitCode, elapsedSec, timedOut, launchErr),
                    "Failed to run cmake --install: " + launchErr);
                UAA_CHECK(!timedOut && exitCode == 0,
                    "cmake --install failed (code " + std::to_string(exitCode) + "):\n" + readTextFile(installLog));

                UAA_CHECK(fs::exists(installDir / "include" / "nvrhi" / "nvrhi.h"),
                    "Installed tree is missing include/nvrhi/nvrhi.h");

                fs::recursive_directory_iterator it(installDir, fs::directory_options::skip_permission_denied, ec);
                fs::recursive_directory_iterator end;
                for (; it != end && !ec; it.increment(ec))
                {
                    const std::string name = it->path().filename().string();
                    UAA_CHECK(name.find("nvrhi_uaa_tests") == std::string::npos &&
                              name.find("nvrhi_uaa_cpu") == std::string::npos,
                        "UAA test binary was unexpectedly installed: " + it->path().string());
                }
#endif
            }

            const fs::path includeDir = !installDir.empty()
                ? (installDir / "include")
                : (nvrhiSrcDir / "include");
            const fs::path nvrhiLib = findNvrhiLibrary(installDir, nvrhiBinDir, isShared, buildConfig);
            UAA_CHECK(!nvrhiLib.empty(), "Could not locate built nvrhi library under " + nvrhiBinDir.string());

            const fs::path consumerSrc = tempRoot / "consumer_src";
            const fs::path consumerBuild = tempRoot / "consumer_build";
            UAA_CHECK(fs::create_directories(consumerSrc, ec), "Failed to create consumer source directory");

            {
                std::ofstream cpp(consumerSrc / "main.cpp", std::ios::trunc);
                UAA_CHECK(cpp.is_open(), "Failed to write consumer main.cpp");
                cpp << "#include <nvrhi/nvrhi.h>\n"
                    << "#include <nvrhi/utils.h>\n"
                    << "#include <iostream>\n"
                    << "int main()\n"
                    << "{\n"
                    << "    if (!nvrhi::verifyHeaderVersion())\n"
                    << "    {\n"
                    << "        std::cerr << \"verifyHeaderVersion() returned false\\n\";\n"
                    << "        return 1;\n"
                    << "    }\n"
                    << "    if (!nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::VULKAN))\n"
                    << "    {\n"
                    << "        return 2;\n"
                    << "    }\n"
                    << "    return 0;\n"
                    << "}\n";
            }

            {
                std::ofstream cm(consumerSrc / "CMakeLists.txt", std::ios::trunc);
                UAA_CHECK(cm.is_open(), "Failed to write consumer CMakeLists.txt");
                cm << "cmake_minimum_required(VERSION 3.11)\n"
                   << "project(nvrhi_uaa_consumer CXX)\n"
                   << "set(CMAKE_CXX_STANDARD 17)\n"
                   << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
                   << "add_executable(nvrhi_consumer main.cpp)\n"
                   << "target_include_directories(nvrhi_consumer PRIVATE \"" << includeDir.generic_string() << "\")\n"
                   << "target_link_libraries(nvrhi_consumer PRIVATE \"" << nvrhiLib.generic_string() << "\")\n";
                if (isShared)
                {
                    cm << "if(MSVC)\n"
                       << "    target_compile_definitions(nvrhi_consumer PRIVATE NVRHI_SHARED_LIBRARY_INCLUDE=1)\n"
                       << "endif()\n";
                }
            }

            std::vector<std::string> cfgArgs = {
                "cmake", "-S", consumerSrc.string(), "-B", consumerBuild.string()
            };
            if (!buildConfig.empty())
            {
                cfgArgs.push_back("-DCMAKE_BUILD_TYPE=" + buildConfig);
            }

            int exitCode = -1;
            double elapsedSec = 0.0;
            bool timedOut = false;
            std::string launchErr;

            const fs::path cfgLog = tempRoot / "configure.log";
            UAA_CHECK(runSubprocess(cfgArgs, 60000, cfgLog, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to configure consumer CMake project: " + launchErr);
            UAA_CHECK(!timedOut && exitCode == 0,
                "Consumer CMake configure failed (code " + std::to_string(exitCode) + "):\n" + readTextFile(cfgLog));

            std::vector<std::string> buildArgs = {
                "cmake", "--build", consumerBuild.string()
            };
            if (!buildConfig.empty())
            {
                buildArgs.push_back("--config");
                buildArgs.push_back(buildConfig);
            }

            const fs::path buildLog = tempRoot / "build.log";
            UAA_CHECK(runSubprocess(buildArgs, 60000, buildLog, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to build consumer CMake project: " + launchErr);
            UAA_CHECK(!timedOut && exitCode == 0,
                "Consumer CMake build failed (code " + std::to_string(exitCode) + "):\n" + readTextFile(buildLog));

            fs::path consumerExe;
            const std::string exeName =
#ifdef _WIN32
                "nvrhi_consumer.exe";
#else
                "nvrhi_consumer";
#endif
            for (const fs::path& cand : {
                consumerBuild / exeName,
                consumerBuild / (buildConfig.empty() ? std::string("Release") : buildConfig) / exeName,
                consumerBuild / "Debug" / exeName,
                consumerBuild / "Release" / exeName })
            {
                if (fs::is_regular_file(cand, ec))
                {
                    consumerExe = cand;
                    break;
                }
            }
            UAA_CHECK(!consumerExe.empty(), "Built consumer executable not found in " + consumerBuild.string());

#ifdef _WIN32
            if (isShared)
            {
                const fs::path dllSrc = nvrhiLib.parent_path() / "nvrhi.dll";
                if (fs::is_regular_file(dllSrc, ec))
                {
                    fs::copy_file(dllSrc, consumerExe.parent_path() / "nvrhi.dll", fs::copy_options::overwrite_existing, ec);
                }
            }
#endif

            const fs::path runLog = tempRoot / "run.log";
            UAA_CHECK(runSubprocess({ consumerExe.string() }, 15000, runLog, exitCode, elapsedSec, timedOut, launchErr),
                "Failed to execute consumer binary: " + launchErr);
            UAA_CHECK(!timedOut && exitCode == 0,
                "Consumer binary failed (code " + std::to_string(exitCode) + "):\n" + readTextFile(runLog));

            ctx.note("Verified offline consumer compile/link/verifyHeaderVersion() against " + nvrhiLib.string());
#endif
        }

        // PKG.Headers: invoke VRHI's vendored header check script when present.
        void testPackageHeaders(TestContext& ctx)
        {
#ifndef UAA_NVRHI_SOURCE_DIR
            ctx.skip("UAA_NVRHI_SOURCE_DIR not defined");
#else
            const fs::path nvrhiSrcDir(UAA_NVRHI_SOURCE_DIR);
            const fs::path scriptPath = fs::weakly_canonical(nvrhiSrcDir / ".." / ".." / "check_vendored_headers.py");
            std::error_code ec;
            if (!fs::is_regular_file(scriptPath, ec))
            {
                ctx.skip("VRHI check_vendored_headers.py not present");
            }

            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path logPath = fs::temp_directory_path() / ("nvrhi_uaa_hdr_" + std::to_string(nonce) + ".log");
            int exitCode = -1;
            double elapsedSec = 0.0;
            bool timedOut = false;
            std::string launchErr;

            bool launched = runSubprocess({ "python3", scriptPath.string() }, 30000, logPath, exitCode, elapsedSec, timedOut, launchErr);
#ifdef _WIN32
            if (!launched)
            {
                launched = runSubprocess({ "python", scriptPath.string() }, 30000, logPath, exitCode, elapsedSec, timedOut, launchErr);
            }
#endif
            const std::string output = readTextFile(logPath);
            fs::remove(logPath, ec);

            UAA_CHECK(launched, "Failed to invoke check_vendored_headers.py: " + launchErr);
            UAA_CHECK(!timedOut, "check_vendored_headers.py timed out");
            UAA_CHECK_EQ(exitCode, 0, "check_vendored_headers.py failed:\n" + output);
#endif
        }
    } // namespace
} // namespace nvrhi::uaa

UAA_REGISTER_TEST("STR.Repeat", "stress", nvrhi::uaa::testStressRepeat);
UAA_REGISTER_TEST("STR.InFlight", "stress", nvrhi::uaa::testStressInFlight);
UAA_REGISTER_TEST("STR.Resize", "stress", nvrhi::uaa::testStressResize);
UAA_REGISTER_TEST("STR.Threads", "stress", nvrhi::uaa::testStressThreads);
UAA_REGISTER_TEST("STR.Timeouts", "stress", nvrhi::uaa::testStressTimeouts);
UAA_REGISTER_TEST("STR.Soak", "stress,destructive", nvrhi::uaa::testStressSoak);
UAA_REGISTER_TEST("PKG.Build", "package", nvrhi::uaa::testPackageBuild);
UAA_REGISTER_TEST("PKG.Headers", "package", nvrhi::uaa::testPackageHeaders);
