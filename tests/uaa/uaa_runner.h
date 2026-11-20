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

#pragma once

#include <nvrhi/nvrhi.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nvrhi::uaa
{
    constexpr int kExitPass = 0;
    constexpr int kExitFail = 1;
    constexpr int kExitSkip = 77;

    enum class TestStatus
    {
        Pass,
        Fail,
        Skip
    };

    class TestFixture;

    struct TestFailureException : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct TestSkipException : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct RunnerOptions
    {
        std::string backend = "cpu";
        std::string validation = "off";
        bool nativeValidation = false;
        std::string deviceSelector;
        std::string filter;
        std::vector<std::string> includeTags;
        std::vector<std::string> excludeTags;
        uint64_t seed = 0x55AA2026ull;
        uint32_t timeoutSeconds = 30;
        std::string shaderDir;
        std::string reportPath;
        std::string executablePath;
        bool listOnly = false;
        bool allowDestructive = false;
        bool selfTestFail = false;
        bool selfTestSkip = false;
        bool selfTestTimeout = false;
    };

    struct TestContext
    {
        const RunnerOptions& options;
        TestFixture* fixture = nullptr;
        std::vector<std::string> notes;

        explicit TestContext(const RunnerOptions& opts, TestFixture* fix = nullptr)
            : options(opts)
            , fixture(fix)
        {
        }

        [[noreturn]] void fail(const std::string& msg, const char* file, int line)
        {
            std::ostringstream oss;
            oss << file << ":" << line << ": " << msg;
            throw TestFailureException(oss.str());
        }

        [[noreturn]] void skip(const std::string& reason)
        {
            throw TestSkipException(reason);
        }

        void note(const std::string& msg)
        {
            notes.push_back(msg);
        }

        uint64_t seed() const
        {
            return options.seed;
        }
    };

    using TestFunc = std::function<void(TestContext&)>;

    struct TestCaseDef
    {
        std::string id;
        std::vector<std::string> tags;
        TestFunc fn;
    };

    inline std::vector<TestCaseDef>& getTestRegistry()
    {
        static std::vector<TestCaseDef> s_Registry;
        return s_Registry;
    }

    inline std::vector<std::string> splitTags(const std::string& tagsStr)
    {
        std::vector<std::string> tags;
        std::string current;
        for (char ch : tagsStr)
        {
            if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
            {
                if (!current.empty())
                {
                    tags.push_back(current);
                    current.clear();
                }
            }
            else
            {
                current.push_back(ch);
            }
        }
        if (!current.empty())
        {
            tags.push_back(current);
        }
        return tags;
    }

    struct TestRegistrar
    {
        TestRegistrar(const char* id, const char* tags, TestFunc fn)
        {
            getTestRegistry().push_back(TestCaseDef{id, splitTags(tags ? tags : ""), std::move(fn)});
        }
    };

    namespace detail
    {
        template <typename T, typename = void>
        struct IsOstreamPrintable : std::false_type {};

        template <typename T>
        struct IsOstreamPrintable<T, std::void_t<decltype(std::declval<std::ostringstream&>() << std::declval<const T&>())>>
            : std::true_type {};

        template <typename T>
        inline std::string stringifyValue(const T& value)
        {
            std::ostringstream oss;
            if constexpr (std::is_same_v<std::decay_t<T>, bool>)
            {
                oss << (value ? "true" : "false");
            }
            else if constexpr (std::is_same_v<std::decay_t<T>, uint8_t>)
            {
                oss << static_cast<uint32_t>(value);
            }
            else if constexpr (std::is_same_v<std::decay_t<T>, int8_t>)
            {
                oss << static_cast<int32_t>(value);
            }
            else if constexpr (std::is_enum_v<std::decay_t<T>>)
            {
                using Underlying = std::underlying_type_t<std::decay_t<T>>;
                oss << static_cast<long long>(static_cast<Underlying>(value));
            }
            else if constexpr (IsOstreamPrintable<T>::value)
            {
                oss << value;
            }
            else
            {
                oss << "<unprintable>";
            }
            return oss.str();
        }
    } // namespace detail
} // namespace nvrhi::uaa

#define UAA_DETAIL_CONCAT_IMPL(a, b) a##b
#define UAA_DETAIL_CONCAT(a, b) UAA_DETAIL_CONCAT_IMPL(a, b)

#define UAA_CHECK(cond, msg) \
    do \
    { \
        if (!(cond)) \
        { \
            std::ostringstream uaaMsgOss_; \
            uaaMsgOss_ << (msg) << " [condition: " << #cond << "]"; \
            ctx.fail(uaaMsgOss_.str(), __FILE__, __LINE__); \
        } \
    } while (false)

#define UAA_CHECK_EQ(a, b, msg) \
    do \
    { \
        const auto& uaaValA_ = (a); \
        const auto& uaaValB_ = (b); \
        if (!(uaaValA_ == uaaValB_)) \
        { \
            std::ostringstream uaaMsgOss_; \
            uaaMsgOss_ << (msg) << " [expected " << #a << " == " << #b \
                       << ", got " << ::nvrhi::uaa::detail::stringifyValue(uaaValA_) \
                       << " vs " << ::nvrhi::uaa::detail::stringifyValue(uaaValB_) << "]"; \
            ctx.fail(uaaMsgOss_.str(), __FILE__, __LINE__); \
        } \
    } while (false)

#define UAA_CHECK_NE(a, b, msg) \
    do \
    { \
        const auto& uaaValA_ = (a); \
        const auto& uaaValB_ = (b); \
        if (!(uaaValA_ != uaaValB_)) \
        { \
            std::ostringstream uaaMsgOss_; \
            uaaMsgOss_ << (msg) << " [expected " << #a << " != " << #b \
                       << ", both were " << ::nvrhi::uaa::detail::stringifyValue(uaaValA_) << "]"; \
            ctx.fail(uaaMsgOss_.str(), __FILE__, __LINE__); \
        } \
    } while (false)

#define UAA_CHECK_NEAR(a, b, eps, msg) \
    do \
    { \
        const double uaaValA_ = static_cast<double>(a); \
        const double uaaValB_ = static_cast<double>(b); \
        const double uaaEps_ = static_cast<double>(eps); \
        const double uaaDiff_ = std::fabs(uaaValA_ - uaaValB_); \
        if (!(uaaDiff_ <= uaaEps_) || std::isnan(uaaValA_) || std::isnan(uaaValB_)) \
        { \
            std::ostringstream uaaMsgOss_; \
            uaaMsgOss_ << (msg) << " [expected |" << #a << " - " << #b << "| <= " << uaaEps_ \
                       << ", got " << uaaValA_ << " vs " << uaaValB_ << " (diff " << uaaDiff_ << ")]"; \
            ctx.fail(uaaMsgOss_.str(), __FILE__, __LINE__); \
        } \
    } while (false)

#define UAA_SKIP(reason) \
    do \
    { \
        ctx.skip(reason); \
    } while (false)

#define UAA_REGISTER_TEST(id_str, tags_str, fn) \
    static ::nvrhi::uaa::TestRegistrar UAA_DETAIL_CONCAT(g_UaaTestRegistrar_, __COUNTER__)(id_str, tags_str, fn)
