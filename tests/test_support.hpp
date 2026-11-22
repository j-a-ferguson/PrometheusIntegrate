#pragma once
//! Helpers that stand in for Rust's `unwrap`/`unwrap_err` and the `approx` crate's assertions.

#include <prometheus/integrate.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace test_support
{

namespace ti = prometheus::integrate;

/// Rust `.unwrap()`: returns the value or throws (gtest reports a thrown exception as a failure).
template <class T, class E>
T unwrap(std::expected<T, E> result)
{
    if (!result)
    {
        if constexpr (ti::Displayable<E>)
        {
            throw std::runtime_error(
                std::format("called `unwrap()` on an error: {}", result.error()));
        }
        else
        {
            throw std::runtime_error("called `unwrap()` on an error");
        }
    }
    return std::move(*result);
}

/// Rust `.unwrap_err()`.
template <class T, class E>
E unwrap_err(std::expected<T, E> result)
{
    if (result)
    {
        throw std::runtime_error("called `unwrap_err()` on a value");
    }
    return std::move(result).error();
}

/// Rust `f64::powi`, which lowers to compiler-rt's `__powidf2` (binary exponentiation).
/// `std::pow(double, int)` instead promotes to `pow(double, double)` and can differ in the last
/// ulp, so the test integrands use this to evaluate exactly as the Rust tests do.
inline double powi(double a, int b)
{
    const bool recip = b < 0;
    double r = 1.0;
    while (true)
    {
        if ((b & 1) != 0)
        {
            r *= a;
        }
        b /= 2;
        if (b == 0)
        {
            break;
        }
        a *= a;
    }
    return recip ? 1.0 / r : r;
}

inline ti::PolynomialDegree degree(std::size_t value)
{
    return unwrap(ti::PolynomialDegree::create(value));
}

inline ti::PointCount count(std::size_t value)
{
    return unwrap(ti::PointCount::create(value));
}

inline ti::Interval interval(double lower, double upper)
{
    return unwrap(ti::Interval::create(lower, upper));
}

/// `approx::relative_eq!(a, b, epsilon = eps, max_relative = max_rel)`:
/// passes if `|a - b| <= eps` *or* `|a - b| <= max_rel * max(|a|, |b|)`.
inline bool approx_relative_eq(double a,
                               double b,
                               double epsilon = std::numeric_limits<double>::epsilon(),
                               double max_relative = std::numeric_limits<double>::epsilon())
{
    if (a == b)
    {
        return true;
    }
    if (std::isinf(a) || std::isinf(b))
    {
        return false;
    }
    const double diff = std::abs(a - b);
    if (diff <= epsilon)
    {
        return true;
    }
    return diff <= std::max(std::abs(a), std::abs(b)) * max_relative;
}

/// `assert_relative_eq!(actual, expected, max_relative = rel)` as a gtest predicate formatter,
/// printing the measured relative error on failure.
inline ::testing::AssertionResult RelativeEq(const char* a_expr,
                                             const char* b_expr,
                                             const char* /*rel_expr*/,
                                             double a,
                                             double b,
                                             double max_relative)
{
    if (approx_relative_eq(a, b, std::numeric_limits<double>::epsilon(), max_relative))
    {
        return ::testing::AssertionSuccess();
    }
    const double rel = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
    return ::testing::AssertionFailure()
           << std::format("{} = {:.17g} and {} = {:.17g} differ by relative {:.3e} > {:.3e}",
                          a_expr,
                          a,
                          b_expr,
                          b,
                          rel,
                          max_relative);
}

#define EXPECT_RELATIVE_EQ(a, b, max_rel)                                                          \
    EXPECT_PRED_FORMAT3(::test_support::RelativeEq, a, b, max_rel)

inline nlohmann::json load_asset(const std::string& name)
{
    const std::filesystem::path path =
        std::filesystem::path{PROMETHEUS_INTEGRATE_ASSETS_DIR} / name;
    std::ifstream stream{path};
    if (!stream)
    {
        throw std::runtime_error("unable to read test data " + path.string());
    }
    return nlohmann::json::parse(stream);
}

} // namespace test_support
