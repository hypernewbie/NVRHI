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

#if !defined(UAA_CPU_ONLY) || !UAA_CPU_ONLY
#include "uaa_backend.h"
#include "uaa_fixture.h"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace nvrhi::uaa
{
    namespace
    {
        struct TestResultRecord
        {
            std::string id;
            std::vector<std::string> tags;
            TestStatus status = TestStatus::Pass;
            std::string reason;
            std::vector<std::string> notes;
            double durationMs = 0.0;
        };

        struct PassReport
        {
            std::string passName;
            bool nvrhiValidation = false;
            bool nativeValidation = false;
            std::string backendName;
            std::string deviceName;
            std::string driverName;
            std::string driverVersion;
            std::string osName;
            std::string buildConfig;
            std::vector<std::string> enabledExtensions;
            std::vector<std::pair<std::string, bool>> features;
            std::vector<TestResultRecord> results;
            size_t passCount = 0;
            size_t failCount = 0;
            size_t skipCount = 0;
            double durationMs = 0.0;
        };

        std::string toLowerAscii(std::string s)
        {
            for (char& ch : s)
            {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return s;
        }

        bool hasTag(const std::vector<std::string>& tags, const std::string& target)
        {
            for (const auto& t : tags)
            {
                if (t == target)
                {
                    return true;
                }
            }
            return false;
        }

        bool matchGlob(const std::string& pattern, const std::string& text)
        {
            size_t pi = 0;
            size_t ti = 0;
            size_t starIdx = std::string::npos;
            size_t matchIdx = 0;

            while (ti < text.size())
            {
                if (pi < pattern.size() && (pattern[pi] == '?' || pattern[pi] == text[ti]))
                {
                    ++pi;
                    ++ti;
                }
                else if (pi < pattern.size() && pattern[pi] == '*')
                {
                    starIdx = pi++;
                    matchIdx = ti;
                }
                else if (starIdx != std::string::npos)
                {
                    pi = starIdx + 1;
                    ti = ++matchIdx;
                }
                else
                {
                    return false;
                }
            }

            while (pi < pattern.size() && pattern[pi] == '*')
            {
                ++pi;
            }
            return pi == pattern.size();
        }

        std::string jsonEscape(const std::string& s)
        {
            std::ostringstream oss;
            for (unsigned char c : s)
            {
                switch (c)
                {
                case '"':  oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default:
                    if (c < 0x20)
                    {
                        oss << "\\u00"
                            << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(c)
                            << std::dec;
                    }
                    else
                    {
                        oss << static_cast<char>(c);
                    }
                    break;
                }
            }
            return oss.str();
        }

        const char* statusName(TestStatus status)
        {
            switch (status)
            {
            case TestStatus::Pass: return "pass";
            case TestStatus::Fail: return "fail";
            case TestStatus::Skip: return "skip";
            default: return "unknown";
            }
        }

        void printUsage(const char* argv0)
        {
            std::cout
                << "Usage: " << (argv0 ? argv0 : "nvrhi_uaa_tests") << " [options]\n"
                << "Options:\n"
                << "  --backend <cpu|vulkan|metal|d3d11|d3d12>  Select backend (default: cpu)\n"
                << "  --validation <off|on|both>                NVRHI validation pass mode (default: off)\n"
                << "  --native-validation [on|off]              Enable native API validation (default: off)\n"
                << "  --device <name_or_index>                  Select GPU adapter by substring or index\n"
                << "  --filter <glob>                           Filter test IDs by wildcard glob\n"
                << "  --tag <tag>                               Include tests with tag (repeatable/comma)\n"
                << "  --exclude-tag <tag>                       Exclude tests with tag (repeatable/comma)\n"
                << "  --seed <hex_or_dec>                       Deterministic RNG seed\n"
                << "  --timeout <seconds>                       Per-test watchdog timeout (default: 30)\n"
                << "  --shader-dir <path>                       Override compiled shader root directory\n"
                << "  --report <path>                           Write JSON test report to path\n"
                << "  --list                                    List matching tests without running\n"
                << "  --allow-destructive                       Allow tests tagged 'destructive'\n"
                << "  --self-test-fail                          Verify harness failure exit code (1)\n"
                << "  --self-test-skip                          Verify harness skip exit code (77)\n"
                << "  --self-test-timeout                       Verify harness watchdog abort exit code (2)\n"
                << "  --help, -h                                Show this help message\n";
        }

        bool parseBoolToken(const std::string& val, bool& out)
        {
            const std::string lower = toLowerAscii(val);
            if (lower == "on" || lower == "true" || lower == "1" || lower == "yes")
            {
                out = true;
                return true;
            }
            if (lower == "off" || lower == "false" || lower == "0" || lower == "no")
            {
                out = false;
                return true;
            }
            return false;
        }

        bool parseArgs(int argc, char** argv, RunnerOptions& opts, bool& outShowHelp)
        {
            outShowHelp = false;
            opts.executablePath = (argc > 0 && argv[0]) ? argv[0] : "";
#ifdef UAA_SHADER_DIR
            opts.shaderDir = UAA_SHADER_DIR;
#else
            opts.shaderDir = "shaders";
#endif

            auto requireValue = [&](int& idx, const char* flag, std::string& out) -> bool
            {
                if (idx + 1 >= argc)
                {
                    std::cerr << "Missing value for option: " << flag << "\n";
                    return false;
                }
                out = argv[++idx];
                return true;
            };

            for (int i = 1; i < argc; ++i)
            {
                const std::string arg = argv[i];
                if (arg == "--help" || arg == "-h")
                {
                    outShowHelp = true;
                    return true;
                }
                else if (arg == "--backend")
                {
                    std::string val;
                    if (!requireValue(i, "--backend", val))
                        return false;
                    opts.backend = toLowerAscii(val);
                }
                else if (arg == "--validation")
                {
                    std::string val;
                    if (!requireValue(i, "--validation", val))
                        return false;
                    opts.validation = toLowerAscii(val);
                    if (opts.validation != "off" && opts.validation != "on" && opts.validation != "both")
                    {
                        std::cerr << "Invalid --validation mode: " << val << " (expected off|on|both)\n";
                        return false;
                    }
                }
                else if (arg == "--native-validation")
                {
                    if (i + 1 < argc && argv[i + 1][0] != '-')
                    {
                        bool parsed = false;
                        if (parseBoolToken(argv[i + 1], parsed))
                        {
                            opts.nativeValidation = parsed;
                            ++i;
                            continue;
                        }
                    }
                    opts.nativeValidation = true;
                }
                else if (arg == "--device")
                {
                    if (!requireValue(i, "--device", opts.deviceSelector))
                        return false;
                }
                else if (arg == "--filter")
                {
                    if (!requireValue(i, "--filter", opts.filter))
                        return false;
                }
                else if (arg == "--tag")
                {
                    std::string val;
                    if (!requireValue(i, "--tag", val))
                        return false;
                    for (const auto& t : splitTags(val))
                    {
                        opts.includeTags.push_back(t);
                    }
                }
                else if (arg == "--exclude-tag")
                {
                    std::string val;
                    if (!requireValue(i, "--exclude-tag", val))
                        return false;
                    for (const auto& t : splitTags(val))
                    {
                        opts.excludeTags.push_back(t);
                    }
                }
                else if (arg == "--seed")
                {
                    std::string val;
                    if (!requireValue(i, "--seed", val))
                        return false;
                    try
                    {
                        opts.seed = std::stoull(val, nullptr, 0);
                    }
                    catch (...)
                    {
                        std::cerr << "Invalid --seed value: " << val << "\n";
                        return false;
                    }
                }
                else if (arg == "--timeout")
                {
                    std::string val;
                    if (!requireValue(i, "--timeout", val))
                        return false;
                    try
                    {
                        opts.timeoutSeconds = static_cast<uint32_t>(std::stoul(val, nullptr, 0));
                    }
                    catch (...)
                    {
                        std::cerr << "Invalid --timeout value: " << val << "\n";
                        return false;
                    }
                }
                else if (arg == "--shader-dir")
                {
                    if (!requireValue(i, "--shader-dir", opts.shaderDir))
                        return false;
                }
                else if (arg == "--report")
                {
                    if (!requireValue(i, "--report", opts.reportPath))
                        return false;
                }
                else if (arg == "--list")
                {
                    opts.listOnly = true;
                }
                else if (arg == "--allow-destructive")
                {
                    opts.allowDestructive = true;
                }
                else if (arg == "--self-test-fail")
                {
                    opts.selfTestFail = true;
                }
                else if (arg == "--self-test-skip")
                {
                    opts.selfTestSkip = true;
                }
                else if (arg == "--self-test-timeout")
                {
                    opts.selfTestTimeout = true;
                }
                else
                {
                    std::cerr << "Unknown option: " << arg << "\n";
                    return false;
                }
            }

            return true;
        }

        std::vector<const TestCaseDef*> selectTests(const RunnerOptions& opts)
        {
            std::vector<const TestCaseDef*> selected;
            const bool cpuMode = (opts.backend == "cpu");
#if defined(UAA_CPU_ONLY) && UAA_CPU_ONLY
            const bool cpuOnlyBuild = true;
#else
            const bool cpuOnlyBuild = false;
#endif

            for (const auto& tc : getTestRegistry())
            {
                const bool onlyCpuTag = cpuOnlyBuild || (cpuMode && !hasTag(opts.includeTags, "package"));
                if (onlyCpuTag && !hasTag(tc.tags, "cpu"))
                {
                    continue;
                }

                if (!opts.filter.empty() && !matchGlob(opts.filter, tc.id))
                {
                    continue;
                }

                if (!opts.includeTags.empty())
                {
                    bool matched = false;
                    for (const auto& inc : opts.includeTags)
                    {
                        if (hasTag(tc.tags, inc))
                        {
                            matched = true;
                            break;
                        }
                    }
                    if (!matched)
                    {
                        continue;
                    }
                }

                if (!opts.excludeTags.empty())
                {
                    bool excluded = false;
                    for (const auto& exc : opts.excludeTags)
                    {
                        if (hasTag(tc.tags, exc))
                        {
                            excluded = true;
                            break;
                        }
                    }
                    if (excluded)
                    {
                        continue;
                    }
                }

                selected.push_back(&tc);
            }

            std::sort(selected.begin(), selected.end(), [](const TestCaseDef* a, const TestCaseDef* b)
            {
                return a->id < b->id;
            });

            return selected;
        }

        class WatchdogTimer
        {
        public:
            WatchdogTimer(std::string testId, uint32_t timeoutSeconds)
                : m_TestId(std::move(testId))
                , m_TimeoutSeconds(timeoutSeconds)
            {
                if (m_TimeoutSeconds > 0)
                {
                    m_Thread = std::thread([this]()
                    {
                        std::unique_lock<std::mutex> lock(m_Mutex);
                        const bool completed = m_Cv.wait_for(
                            lock,
                            std::chrono::seconds(m_TimeoutSeconds),
                            [this]() { return m_Done; });
                        if (!completed)
                        {
                            std::fprintf(
                                stderr,
                                "\n[TIMEOUT] Test '%s' exceeded %u second watchdog timeout; aborting.\n",
                                m_TestId.c_str(),
                                m_TimeoutSeconds);
                            std::fflush(stdout);
                            std::fflush(stderr);
                            _exit(2);
                        }
                    });
                }
            }

            ~WatchdogTimer()
            {
                cancel();
            }

            void cancel()
            {
                if (m_Thread.joinable())
                {
                    {
                        std::lock_guard<std::mutex> lock(m_Mutex);
                        m_Done = true;
                    }
                    m_Cv.notify_all();
                    m_Thread.join();
                }
            }

        private:
            std::string m_TestId;
            uint32_t m_TimeoutSeconds = 0;
            std::mutex m_Mutex;
            std::condition_variable m_Cv;
            bool m_Done = false;
            std::thread m_Thread;
        };

        void writeJsonReport(
            const std::string& path,
            const RunnerOptions& opts,
            const std::vector<PassReport>& passes)
        {
            if (path.empty())
            {
                return;
            }

            std::ofstream out(path, std::ios::trunc);
            if (!out.is_open())
            {
                std::cerr << "Warning: failed to write JSON report to '" << path << "'\n";
                return;
            }

            out << "{\n";
            out << "  \"backend\": \"" << jsonEscape(opts.backend) << "\",\n";
            out << "  \"validation\": \"" << jsonEscape(opts.validation) << "\",\n";
            out << "  \"nativeValidation\": " << (opts.nativeValidation ? "true" : "false") << ",\n";
            out << "  \"seed\": \"0x" << std::hex << opts.seed << std::dec << "\",\n";
            out << "  \"passes\": [\n";

            for (size_t pi = 0; pi < passes.size(); ++pi)
            {
                const PassReport& pass = passes[pi];
                out << "    {\n";
                out << "      \"passName\": \"" << jsonEscape(pass.passName) << "\",\n";
                out << "      \"nvrhiValidation\": " << (pass.nvrhiValidation ? "true" : "false") << ",\n";
                out << "      \"nativeValidation\": " << (pass.nativeValidation ? "true" : "false") << ",\n";
                out << "      \"deviceInfo\": {\n";
                out << "        \"backendName\": \"" << jsonEscape(pass.backendName) << "\",\n";
                out << "        \"deviceName\": \"" << jsonEscape(pass.deviceName) << "\",\n";
                out << "        \"driverName\": \"" << jsonEscape(pass.driverName) << "\",\n";
                out << "        \"driverVersion\": \"" << jsonEscape(pass.driverVersion) << "\",\n";
                out << "        \"osName\": \"" << jsonEscape(pass.osName) << "\",\n";
                out << "        \"buildConfig\": \"" << jsonEscape(pass.buildConfig) << "\",\n";
                out << "        \"enabledExtensions\": [";
                for (size_t ei = 0; ei < pass.enabledExtensions.size(); ++ei)
                {
                    if (ei > 0) out << ", ";
                    out << "\"" << jsonEscape(pass.enabledExtensions[ei]) << "\"";
                }
                out << "]\n";
                out << "      },\n";
                out << "      \"features\": {\n";
                for (size_t fi = 0; fi < pass.features.size(); ++fi)
                {
                    out << "        \"" << jsonEscape(pass.features[fi].first) << "\": "
                        << (pass.features[fi].second ? "true" : "false");
                    if (fi + 1 < pass.features.size()) out << ",";
                    out << "\n";
                }
                out << "      },\n";
                out << "      \"summary\": {\n";
                out << "        \"total\": " << pass.results.size() << ",\n";
                out << "        \"pass\": " << pass.passCount << ",\n";
                out << "        \"fail\": " << pass.failCount << ",\n";
                out << "        \"skip\": " << pass.skipCount << ",\n";
                out << "        \"durationMs\": " << std::fixed << std::setprecision(3) << pass.durationMs << "\n";
                out << "      },\n";
                out << "      \"tests\": [\n";
                for (size_t ti = 0; ti < pass.results.size(); ++ti)
                {
                    const TestResultRecord& tr = pass.results[ti];
                    out << "        {\n";
                    out << "          \"id\": \"" << jsonEscape(tr.id) << "\",\n";
                    out << "          \"tags\": [";
                    for (size_t k = 0; k < tr.tags.size(); ++k)
                    {
                        if (k > 0) out << ", ";
                        out << "\"" << jsonEscape(tr.tags[k]) << "\"";
                    }
                    out << "],\n";
                    out << "          \"status\": \"" << statusName(tr.status) << "\",\n";
                    out << "          \"reason\": \"" << jsonEscape(tr.reason) << "\",\n";
                    out << "          \"notes\": [";
                    for (size_t k = 0; k < tr.notes.size(); ++k)
                    {
                        if (k > 0) out << ", ";
                        out << "\"" << jsonEscape(tr.notes[k]) << "\"";
                    }
                    out << "],\n";
                    out << "          \"durationMs\": " << std::fixed << std::setprecision(3) << tr.durationMs << "\n";
                    out << "        }" << (ti + 1 < pass.results.size() ? "," : "") << "\n";
                }
                out << "      ]\n";
                out << "    }" << (pi + 1 < passes.size() ? "," : "") << "\n";
            }

            out << "  ]\n";
            out << "}\n";
        }

        TestResultRecord executeSingleTest(
            const TestCaseDef& tc,
            const RunnerOptions& opts,
            TestFixture* fixture)
        {
            TestResultRecord record;
            record.id = tc.id;
            record.tags = tc.tags;

            if (hasTag(tc.tags, "destructive") && !opts.allowDestructive)
            {
                record.status = TestStatus::Skip;
                record.reason = "Destructive test requires --allow-destructive";
                return record;
            }

#if !defined(UAA_CPU_ONLY) || !UAA_CPU_ONLY
            if (fixture)
            {
                fixture->messages().clear();
                if (fixture->backend())
                {
                    fixture->backend()->clearNativeValidationMessages();
                }
            }
#endif

            const auto startTime = std::chrono::steady_clock::now();
            TestContext ctx(opts, fixture);

            {
                WatchdogTimer watchdog(tc.id, opts.timeoutSeconds);
                try
                {
                    tc.fn(ctx);
                    record.status = TestStatus::Pass;
                }
                catch (const TestSkipException& ex)
                {
                    record.status = TestStatus::Skip;
                    record.reason = ex.what();
                }
                catch (const TestFailureException& ex)
                {
                    record.status = TestStatus::Fail;
                    record.reason = ex.what();
                }
                catch (const std::exception& ex)
                {
                    record.status = TestStatus::Fail;
                    record.reason = std::string("Uncaught std::exception: ") + ex.what();
                }
                catch (...)
                {
                    record.status = TestStatus::Fail;
                    record.reason = "Uncaught unknown exception";
                }
                watchdog.cancel();
            }

            const auto endTime = std::chrono::steady_clock::now();
            record.durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            record.notes = std::move(ctx.notes);

#if !defined(UAA_CPU_ONLY) || !UAA_CPU_ONLY
            if (fixture && record.status != TestStatus::Skip)
            {
                auto addDiagnostic = [&](const std::string& diagnostic)
                {
                    if (!record.reason.empty())
                    {
                        record.reason += "\n";
                    }
                    record.reason += diagnostic;
                    record.status = TestStatus::Fail;
                };

                if (fixture->messages().errorCount() > 0)
                {
                    addDiagnostic("Unexpected NVRHI validation/backend errors:\n" + fixture->messages().formatAll());
                }
                if (fixture->backend())
                {
                    std::vector<std::string> nativeErrors;
                    if (fixture->backend()->hasNativeValidationErrors(nativeErrors))
                    {
                        std::ostringstream diagnostics;
                        diagnostics << "Unexpected native validation errors:";
                        for (const auto& error : nativeErrors)
                        {
                            diagnostics << "\n  " << error;
                        }
                        addDiagnostic(diagnostics.str());
                    }
                }
            }
#endif

            return record;
        }

        void printPassSummary(const PassReport& pass)
        {
            std::cout << "\n--- Summary (" << pass.passName << ") ---\n";
            std::cout << "Total: " << pass.results.size()
                      << " | Pass: " << pass.passCount
                      << " | Fail: " << pass.failCount
                      << " | Skip: " << pass.skipCount
                      << " | Time: " << std::fixed << std::setprecision(1) << pass.durationMs << " ms\n";

            if (pass.skipCount > 0)
            {
                std::cout << "Skipped tests:\n";
                for (const auto& r : pass.results)
                {
                    if (r.status == TestStatus::Skip)
                    {
                        std::cout << "  [SKIP] " << r.id << " - " << r.reason << "\n";
                    }
                }
            }

            if (pass.failCount > 0)
            {
                std::cout << "Failed tests:\n";
                for (const auto& r : pass.results)
                {
                    if (r.status == TestStatus::Fail)
                    {
                        std::cout << "  [FAIL] " << r.id << " - " << r.reason << "\n";
                    }
                }
            }
        }
    } // namespace
} // namespace nvrhi::uaa

int main(int argc, char** argv)
{
    using namespace nvrhi::uaa;

    RunnerOptions opts;
    bool showHelp = false;
    if (!parseArgs(argc, argv, opts, showHelp))
    {
        return kExitFail;
    }

    if (showHelp)
    {
        printUsage(argc > 0 ? argv[0] : nullptr);
        return kExitPass;
    }

    if (opts.selfTestFail)
    {
        std::cout << "SELF-TEST FAIL: returning failure exit code " << kExitFail << "\n";
        return kExitFail;
    }

    if (opts.selfTestSkip)
    {
        std::cout << "SELF-TEST SKIP: returning skip exit code " << kExitSkip << "\n";
        return kExitSkip;
    }

    if (opts.selfTestTimeout)
    {
        std::cout << "SELF-TEST TIMEOUT: waiting for " << opts.timeoutSeconds << "s watchdog abort...\n";
        std::cout.flush();
        WatchdogTimer watchdog("SELF_TEST.Timeout", opts.timeoutSeconds > 0 ? opts.timeoutSeconds : 1);
        std::mutex mtx;
        std::condition_variable cv;
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []() { return false; });
        return kExitFail;
    }

    // Preserve completed case output if a native API aborts or a watchdog stops the process.
    std::cout << std::unitbuf;
    const std::vector<const TestCaseDef*> selected = selectTests(opts);

    if (opts.listOnly)
    {
        for (const TestCaseDef* tc : selected)
        {
            std::cout << tc->id << " [";
            for (size_t i = 0; i < tc->tags.size(); ++i)
            {
                if (i > 0) std::cout << ",";
                std::cout << tc->tags[i];
            }
            std::cout << "]\n";
        }
        std::cout << "Total matching tests: " << selected.size() << "\n";
        return kExitPass;
    }

    std::vector<PassReport> passReports;
    size_t totalPass = 0;
    size_t totalFail = 0;
    size_t totalSkip = 0;

#if defined(UAA_CPU_ONLY) && UAA_CPU_ONLY
    const bool cpuMode = true;
    if (opts.backend != "cpu")
    {
        std::cout << "SKIP: CPU-only test binary cannot run GPU backend '" << opts.backend << "'\n";
        return kExitSkip;
    }
#else
    const bool cpuMode = (opts.backend == "cpu");
#endif

    if (cpuMode)
    {
        PassReport pass;
        pass.passName = "cpu";
        pass.backendName = "cpu";
        pass.deviceName = "CPU";
#ifdef UAA_BUILD_CONFIG
        pass.buildConfig = UAA_BUILD_CONFIG;
#endif
        std::cout << "=== NVRHI UAA Test Runner ===\n"
                  << "Backend: cpu | Seed: 0x" << std::hex << opts.seed << std::dec
                  << " | Tests: " << selected.size() << "\n";

        const auto passStart = std::chrono::steady_clock::now();
        for (const TestCaseDef* tc : selected)
        {
            TestResultRecord r = executeSingleTest(*tc, opts, nullptr);
            if (r.status == TestStatus::Pass) ++pass.passCount;
            else if (r.status == TestStatus::Fail) ++pass.failCount;
            else ++pass.skipCount;

            std::cout << (r.status == TestStatus::Pass ? "[PASS] "
                        : (r.status == TestStatus::Skip ? "[SKIP] " : "[FAIL] "))
                      << r.id << " (" << std::fixed << std::setprecision(1) << r.durationMs << " ms)";
            if (!r.reason.empty())
            {
                std::cout << " : " << r.reason;
            }
            std::cout << "\n";
            pass.results.push_back(std::move(r));
        }
        pass.durationMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - passStart).count();

        totalPass += pass.passCount;
        totalFail += pass.failCount;
        totalSkip += pass.skipCount;
        printPassSummary(pass);
        passReports.push_back(std::move(pass));
    }
#if !defined(UAA_CPU_ONLY) || !UAA_CPU_ONLY
    else
    {
        std::vector<bool> validationPasses;
        if (opts.validation == "off")
        {
            validationPasses.push_back(false);
        }
        else if (opts.validation == "on")
        {
            validationPasses.push_back(true);
        }
        else
        {
            validationPasses.push_back(false);
#if defined(UAA_WITH_VALIDATION) && UAA_WITH_VALIDATION
            validationPasses.push_back(true);
#else
            std::cout << "Note: NVRHI validation layer is not compiled; running raw pass only.\n";
#endif
        }

        for (bool enableValidation : validationPasses)
        {
            TestFixture fixture;
            std::string initMsg;
            bool unavailable = false;
            if (!fixture.init(opts, enableValidation, initMsg, unavailable))
            {
                PassReport pass;
                pass.passName = opts.backend + (enableValidation ? "_validation" : "_raw");
                pass.backendName = opts.backend;
#ifdef UAA_BUILD_CONFIG
                pass.buildConfig = UAA_BUILD_CONFIG;
#endif
                const TestStatus status = unavailable ? TestStatus::Skip : TestStatus::Fail;
                std::cout << (unavailable ? "SKIP: " : "FAIL: Backend initialisation failed: ") << initMsg << "\n";
                for (const TestCaseDef* tc : selected)
                {
                    TestResultRecord record;
                    record.id = tc->id;
                    record.tags = tc->tags;
                    record.status = status;
                    record.reason = initMsg;
                    pass.results.push_back(std::move(record));
                }
                if (unavailable)
                {
                    pass.skipCount = selected.size();
                    totalSkip += pass.skipCount;
                }
                else
                {
                    pass.failCount = selected.size();
                    totalFail += pass.failCount;
                }
                passReports.push_back(std::move(pass));
                continue;
            }

            const DeviceInfo& info = fixture.deviceInfo();
            PassReport pass;
            pass.passName = enableValidation ? (opts.backend + "_validation") : (opts.backend + "_raw");
            pass.nvrhiValidation = enableValidation;
            pass.nativeValidation = info.nativeValidation;
            pass.backendName = info.backendName;
            pass.deviceName = info.deviceName;
            pass.driverName = info.driverName;
            pass.driverVersion = info.driverVersion;
            pass.osName = info.osName;
            pass.buildConfig = info.buildConfig;
            pass.enabledExtensions = info.enabledExtensions;

            for (const auto& entry : fixture.features().entries)
            {
                pass.features.emplace_back(entry.name, entry.supported);
            }

            std::cout << "=== NVRHI UAA Test Runner ===\n"
                      << "Backend: " << info.backendName
                      << " | Device: " << info.deviceName
                      << " | Driver: " << info.driverName << " " << info.driverVersion << "\n"
                      << "OS: " << info.osName
                      << " | Build: " << info.buildConfig
                      << " | NativeVal: " << (info.nativeValidation ? "on" : "off")
                      << " | NvrhiVal: " << (info.nvrhiValidation ? "on" : "off")
                      << " | Seed: 0x" << std::hex << opts.seed << std::dec << "\n"
                      << "Features: " << fixture.features().formatSummary() << "\n";

            const auto passStart = std::chrono::steady_clock::now();
            for (const TestCaseDef* tc : selected)
            {
                TestResultRecord r = executeSingleTest(*tc, opts, &fixture);
                if (r.status == TestStatus::Pass) ++pass.passCount;
                else if (r.status == TestStatus::Fail) ++pass.failCount;
                else ++pass.skipCount;

                std::cout << (r.status == TestStatus::Pass ? "[PASS] "
                            : (r.status == TestStatus::Skip ? "[SKIP] " : "[FAIL] "))
                          << r.id << " (" << std::fixed << std::setprecision(1) << r.durationMs << " ms)";
                if (!r.reason.empty())
                {
                    std::cout << " : " << r.reason;
                }
                std::cout << "\n";
                pass.results.push_back(std::move(r));
            }
            pass.durationMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - passStart).count();

            totalPass += pass.passCount;
            totalFail += pass.failCount;
            totalSkip += pass.skipCount;
            printPassSummary(pass);
            passReports.push_back(std::move(pass));

            fixture.shutdown();
        }
    }
#endif

    writeJsonReport(opts.reportPath, opts, passReports);

    if (totalFail > 0)
    {
        return kExitFail;
    }
    if (totalPass == 0 && totalSkip > 0)
    {
        return kExitSkip;
    }
    return kExitPass;
}
