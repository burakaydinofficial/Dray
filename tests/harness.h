// A test harness small enough to read in one sitting.
//
// WHY no external framework: the dependency budget for this project is the C++
// standard library plus vendored llama.cpp/ggml (CLAUDE.md, Toolchain). A test
// framework would be the largest third-party dependency in the tree and would
// buy us nothing that 150 lines of registry + assertions does not.
//
// WHY assertions carry both values: this project cannot casually re-run its own
// subject matter (the smallest model is 143 GB and a real run is hours), so a
// failing assertion has to be diagnosable from the ctest log alone -- there is
// no cheap "add a print and run it again" loop for the things these tests stand
// in for.
//
// Each test executable is exactly ONE translation unit that includes this
// header; main() lives here. Fixture translation units must NOT include it.
//
// Usage:
//     #include "harness.h"
//     LZ_TEST(cap_is_never_exceeded) {
//         LZ_CHECK_EQ(a.used(), 0u);
//         LZ_REQUIRE(a.reserve(Category::Misc, 8));  // fatal: aborts this case
//     }
//
// Exit code is 0 only if at least one case ran and every case passed.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dray::test {

struct Case {
    const char* name;
    void (*fn)();
};

// Function-local static so registration order across TUs is irrelevant: the
// vector is constructed on first use, before any Registrar can touch it.
inline std::vector<Case>& cases() {
    static std::vector<Case> v;
    return v;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { cases().push_back(Case{name, fn}); }
};

// Thrown by the LZ_REQUIRE family. Caught by the runner; ends the case only.
struct CaseAborted {};

struct State {
    long long checks = 0;
    long long failures = 0;
    const char* current = "";
};

inline State& state() {
    static State s;
    return s;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
    State& s = state();
    ++s.failures;
    std::fprintf(stderr, "  FAILED  %s:%d\n           %s\n", file, line, msg.c_str());
    std::fflush(stderr);
}

template <class T>
concept Streamable = requires(std::ostream& os, const T& v) { os << v; };

// Integer types that std::cmp_* accepts. bool and the character types are
// excluded because passing them to std::cmp_equal is ill-formed.
template <class T>
inline constexpr bool is_cmp_int_v =
    std::is_integral_v<std::remove_cv_t<T>> &&
    !std::is_same_v<std::remove_cv_t<T>, bool> &&
    !std::is_same_v<std::remove_cv_t<T>, char> &&
    !std::is_same_v<std::remove_cv_t<T>, wchar_t> &&
    !std::is_same_v<std::remove_cv_t<T>, char8_t> &&
    !std::is_same_v<std::remove_cv_t<T>, char16_t> &&
    !std::is_same_v<std::remove_cv_t<T>, char32_t>;

template <class T>
std::string describe(const T& v) {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_enum_v<U>) {
        return std::to_string(static_cast<long long>(v));
    } else if constexpr (std::is_integral_v<U>) {
        if constexpr (std::is_signed_v<U>) {
            return std::to_string(static_cast<long long>(v));
        } else {
            return std::to_string(static_cast<unsigned long long>(v));
        }
    } else if constexpr (std::is_floating_point_v<U>) {
        std::ostringstream os;
        os.precision(9);
        os << v;
        return os.str();
    } else if constexpr (Streamable<U>) {
        std::ostringstream os;
        os << v;
        return os.str();
    } else {
        return "<unprintable>";
    }
}

// Mixed-signedness comparison without the /W4 conversion warnings and without
// the wrong answer: std::cmp_* compares mathematical values, so
// (int)-1 == (size_t)~0ull is false here, as it should be.
template <class A, class B>
bool cmp_eq(const A& a, const B& b) {
    if constexpr (is_cmp_int_v<A> && is_cmp_int_v<B>) {
        return std::cmp_equal(a, b);
    } else {
        return a == b;
    }
}

template <class A, class B>
bool cmp_ne(const A& a, const B& b) {
    return !cmp_eq(a, b);
}

template <class A, class B>
bool cmp_lt(const A& a, const B& b) {
    if constexpr (is_cmp_int_v<A> && is_cmp_int_v<B>) {
        return std::cmp_less(a, b);
    } else {
        return a < b;
    }
}

template <class A, class B>
bool cmp_le(const A& a, const B& b) {
    if constexpr (is_cmp_int_v<A> && is_cmp_int_v<B>) {
        return std::cmp_less_equal(a, b);
    } else {
        return a <= b;
    }
}

template <class A, class B>
bool cmp_gt(const A& a, const B& b) {
    return cmp_lt(b, a);
}

template <class A, class B>
bool cmp_ge(const A& a, const B& b) {
    return cmp_le(b, a);
}

}  // namespace dray::test

#define LZ_TEST(name)                                                             \
    static void lz_case_##name();                                                 \
    static const ::dray::test::Registrar lz_reg_##name(#name, &lz_case_##name); \
    static void lz_case_##name()

#define LZ_FAIL(msg)                                                              \
    do {                                                                          \
        ::dray::test::report_failure(__FILE__, __LINE__, std::string(msg));     \
    } while (false)

#define LZ_BOOL_(expr, fatal)                                                     \
    do {                                                                          \
        ++::dray::test::state().checks;                                         \
        if (!(expr)) {                                                            \
            ::dray::test::report_failure(__FILE__, __LINE__,                    \
                                            std::string("expected true: ") + #expr); \
            if constexpr (fatal) {                                                \
                throw ::dray::test::CaseAborted{};                             \
            }                                                                     \
        }                                                                         \
    } while (false)

#define LZ_CMP_(a, b, fn, sym, fatal)                                             \
    do {                                                                          \
        ++::dray::test::state().checks;                                         \
        auto&& lz_lhs_ = (a);                                                     \
        auto&& lz_rhs_ = (b);                                                     \
        if (!::dray::test::fn(lz_lhs_, lz_rhs_)) {                             \
            ::dray::test::report_failure(                                      \
                __FILE__, __LINE__,                                               \
                std::string("expected ") + #a + " " sym " " + #b +                \
                    "\n             lhs = " + ::dray::test::describe(lz_lhs_) + \
                    "\n             rhs = " + ::dray::test::describe(lz_rhs_)); \
            if constexpr (fatal) {                                                \
                throw ::dray::test::CaseAborted{};                             \
            }                                                                     \
        }                                                                         \
    } while (false)

// Non-fatal: records the failure and keeps going, so one run reports every
// broken invariant instead of only the first.
#define LZ_CHECK(expr)      LZ_BOOL_(expr, false)
#define LZ_CHECK_EQ(a, b)   LZ_CMP_(a, b, cmp_eq, "==", false)
#define LZ_CHECK_NE(a, b)   LZ_CMP_(a, b, cmp_ne, "!=", false)
#define LZ_CHECK_LT(a, b)   LZ_CMP_(a, b, cmp_lt, "<",  false)
#define LZ_CHECK_LE(a, b)   LZ_CMP_(a, b, cmp_le, "<=", false)
#define LZ_CHECK_GT(a, b)   LZ_CMP_(a, b, cmp_gt, ">",  false)
#define LZ_CHECK_GE(a, b)   LZ_CMP_(a, b, cmp_ge, ">=", false)

// Fatal: use when continuing would dereference null or read uninitialised data.
#define LZ_REQUIRE(expr)    LZ_BOOL_(expr, true)
#define LZ_REQUIRE_EQ(a, b) LZ_CMP_(a, b, cmp_eq, "==", true)
#define LZ_REQUIRE_NE(a, b) LZ_CMP_(a, b, cmp_ne, "!=", true)
#define LZ_REQUIRE_LT(a, b) LZ_CMP_(a, b, cmp_lt, "<",  true)
#define LZ_REQUIRE_LE(a, b) LZ_CMP_(a, b, cmp_le, "<=", true)
#define LZ_REQUIRE_GT(a, b) LZ_CMP_(a, b, cmp_gt, ">",  true)
#define LZ_REQUIRE_GE(a, b) LZ_CMP_(a, b, cmp_ge, ">=", true)

// Substring assertion: used wherever a refusal string has to NAME something
// (Invariant 6 -- a refusal that does not say what would work is not a report).
#define LZ_CHECK_CONTAINS(haystack, needle)                                       \
    do {                                                                          \
        ++::dray::test::state().checks;                                         \
        const std::string lz_h_ = (haystack);                                     \
        const std::string lz_n_ = (needle);                                       \
        if (lz_h_.find(lz_n_) == std::string::npos) {                             \
            ::dray::test::report_failure(                                      \
                __FILE__, __LINE__,                                               \
                std::string("expected substring \"") + lz_n_ + "\" in:\n             " + lz_h_); \
        }                                                                         \
    } while (false)

int main(int argc, char** argv) {
    // Optional argv[1] is a substring filter, so a single case can be re-run by
    // hand without rebuilding: ctest -R accountant is coarse, this is not.
    const char* filter = (argc > 1) ? argv[1] : nullptr;

    auto& all = ::dray::test::cases();
    auto& st = ::dray::test::state();

    int ran = 0;
    int failed = 0;

    const auto suite_start = std::chrono::steady_clock::now();

    for (const ::dray::test::Case& c : all) {
        if (filter != nullptr && std::string(c.name).find(filter) == std::string::npos) {
            continue;
        }
        ++ran;
        st.current = c.name;
        const long long before = st.failures;

        std::fprintf(stdout, "[ RUN    ] %s\n", c.name);
        std::fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        try {
            c.fn();
        } catch (const ::dray::test::CaseAborted&) {
            // Already reported by the assertion that threw.
        } catch (const std::exception& e) {
            ::dray::test::report_failure(__FILE__, __LINE__,
                                            std::string("unexpected exception: ") + e.what());
        } catch (...) {
            ::dray::test::report_failure(__FILE__, __LINE__,
                                            "unexpected non-std exception");
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (st.failures > before) {
            ++failed;
            std::fprintf(stdout, "[   FAIL ] %s (%.1f ms)\n", c.name, ms);
        } else {
            std::fprintf(stdout, "[     OK ] %s (%.1f ms)\n", c.name, ms);
        }
        std::fflush(stdout);
    }

    const auto suite_end = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(suite_end - suite_start).count();

    std::fprintf(stdout, "%d case(s), %lld assertion(s), %d failed, %.1f ms\n", ran, st.checks,
                 failed, total_ms);
    std::fflush(stdout);

    if (ran == 0) {
        // An executable that ran nothing is a broken test, not a passing one --
        // a silently empty suite is exactly how coverage rots.
        std::fprintf(stderr, "no test cases ran%s%s\n", filter ? " for filter: " : "",
                     filter ? filter : "");
        return 2;
    }
    return failed == 0 ? 0 : 1;
}
