//! Port of `tests/adaptive.rs`.

#include "test_support.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <variant>
#include <vector>

using namespace prometheus::integrate;
using test_support::degree;
using test_support::interval;
using test_support::unwrap;
using test_support::unwrap_err;

namespace
{

GaussRule rule(std::size_t d)
{
    return unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(d)));
}

TensorRule2d tensor_rule(std::size_t d)
{
    auto r = rule(d);
    return TensorRule2d{r, r};
}

Rectangle rectangle(double u_lower, double u_upper, double v_lower, double v_upper)
{
    return unwrap(Rectangle::from_bounds(u_lower, u_upper, v_lower, v_upper));
}

Tolerance absolute(double value)
{
    return unwrap(Tolerance::absolute(value));
}

/// Rust `f64::from_bits(x.to_bits() + 1)`.
double next_up(double x)
{
    return std::bit_cast<double>(std::bit_cast<std::uint64_t>(x) + 1);
}

double indicator(bool condition)
{
    return condition ? 1.0 : 0.0;
}

ConfigIssue non_increasing(RuleAxis axis, std::size_t low, std::size_t high)
{
    return config_issue::NonIncreasingRuleExactness{.axis = axis, .low = low, .high = high};
}

} // namespace

//{{{ collection: one-dimensional tests
TEST(Adaptive, OneDimensionalBuilderValidatesActualRuleExactness)
{
    const auto error = unwrap_err(
        AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(2), rule(3), absolute(1e-8))
            .build());

    ASSERT_EQ(error.issues().size(), 1U);
    EXPECT_EQ(error.issues()[0], non_increasing(RuleAxis::OneDimensional, 3, 3));
}

TEST(Adaptive, OneDimensionalFreeFunctionBuildsAndIntegratesOnce)
{
    auto builder =
        AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(3), rule(5), absolute(1e-12));

    const auto result =
        unwrap(adaptive_quad_1d([](double x) { return x * x; }, std::move(builder)));
    EXPECT_NEAR(result.integral(), 2.0 / 3.0, 1e-14);
    EXPECT_EQ(result.evaluation_count(), 5U);
}

TEST(Adaptive, OneDimensionalFreeFunctionRetainsStructuredConfigurationError)
{
    auto builder =
        AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(2), rule(3), absolute(1e-12));

    const auto options_error =
        unwrap_err(adaptive_quad_1d([](double) { return 0.0; }, std::move(builder)));
    // Rust: `let OptionsError::Config(error) = .. else { panic!() }`.
    const auto* error = std::get_if<ConfigError>(&options_error);
    ASSERT_NE(error, nullptr) << "expected structured configuration error";
    ASSERT_EQ(error->issues().size(), 1U);
    EXPECT_EQ(error->issues()[0], non_increasing(RuleAxis::OneDimensional, 3, 3));
}

TEST(Adaptive, OneDimensionalResultUsesHighRuleAndGlobalTolerance)
{
    const auto tolerance = unwrap(Tolerance::create(1e-10, 1e-10));
    const auto quadrature =
        unwrap(AdaptiveQuadrature1d::builder(interval(0.0, 30.0), rule(5), rule(11), tolerance)
                   .max_depth(RefinementDepth{20})
                   .build());

    const auto result = unwrap(quadrature.integrate([](double x) { return std::sin(x); }));
    const double expected = 1.0 - std::cos(30.0);
    const double limit =
        tolerance.absolute_value() + tolerance.relative_value() * std::abs(result.integral());
    EXPECT_NEAR(result.integral(), expected, 1e-10);
    EXPECT_LE(result.error_estimate(), limit);
    EXPECT_GT(result.terminal_region_count(), 1U);
}

TEST(Adaptive, OneDimensionalRelativeOnlyToleranceConvergesGlobally)
{
    const auto tolerance = unwrap(Tolerance::relative(1e-10));
    const auto quadrature =
        unwrap(AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(3), rule(7), tolerance)
                   .max_depth(RefinementDepth{20})
                   .build());

    const auto result = unwrap(quadrature.integrate([](double x) { return std::exp(x); }));
    const double expected = std::numbers::e - 1.0 / std::numbers::e;
    EXPECT_NEAR(result.integral(), expected, 1e-10);
    EXPECT_LE(result.error_estimate(), tolerance.relative_value() * std::abs(result.integral()));
}

TEST(Adaptive, OneDimensionalIntegrandMayMutateState)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(3), rule(5), absolute(1e-12))
                   .build());
    std::size_t calls = 0;

    const auto result = unwrap(quadrature.integrate(
        [&](double x)
        {
            ++calls;
            return x * x;
        }));

    EXPECT_NEAR(result.integral(), 2.0 / 3.0, 1e-14);
    EXPECT_EQ(calls, result.evaluation_count());
    EXPECT_EQ(result.terminal_region_count(), 1U);
    EXPECT_EQ(result.evaluation_count(), 5U);
}

TEST(Adaptive, OneDimensionalInitialSubdivisionsAreTerminalRegions)
{
    const auto quadrature = unwrap(
        unwrap(AdaptiveQuadrature1d::builder(interval(-2.0, 2.0), rule(3), rule(5), absolute(1e-12))
                   .subdivisions({-1.0, 0.0, 1.0}))
            .build());

    const auto result = unwrap(quadrature.integrate([](double x) { return x * x; }));
    EXPECT_TRUE(std::ranges::equal(quadrature.subdivision_points(), std::vector{-1.0, 0.0, 1.0}));
    EXPECT_EQ(result.terminal_region_count(), 4U);
    EXPECT_EQ(result.evaluation_count(), 20U);
}

TEST(Adaptive, OneDimensionalDepthZeroReturnsHighOrderPartialResult)
{
    const auto domain = interval(-1.0, 1.0);
    const auto low = rule(1);
    const auto high = rule(3);
    const auto quartic = [](double x) { return test_support::powi(x, 4); };
    const double expected_high =
        unwrap(FixedQuadrature1d::builder(domain, high).build().integrate(quartic));
    const auto quadrature = unwrap(AdaptiveQuadrature1d::builder(domain, low, high, absolute(1e-15))
                                       .max_depth(RefinementDepth{0})
                                       .build());

    const auto error = unwrap_err(quadrature.integrate(quartic));
    const auto* depth = std::get_if<integration_error::MaxDepthReached>(&error);
    ASSERT_NE(depth, nullptr) << "expected depth exhaustion";
    EXPECT_EQ(depth->partial_result.integral(), expected_high);
    EXPECT_GT(depth->partial_result.error_estimate(), 0.0);
    EXPECT_EQ(depth->partial_result.terminal_region_count(), 1U);
    EXPECT_EQ(depth->partial_result.evaluation_count(), 3U);
}

TEST(Adaptive, OneDimensionalNonprogressingMidpointIsReported)
{
    const double lower = 1.0;
    const double upper = next_up(lower);
    const auto domain = interval(lower, upper);
    const auto quadrature =
        unwrap(AdaptiveQuadrature1d::builder(
                   domain, rule(1), rule(3), absolute(std::numeric_limits<double>::min()))
                   .max_depth(RefinementDepth{1})
                   .build());

    const auto error =
        unwrap_err(quadrature.integrate([&](double x) { return x == upper ? 1e300 : 0.0; }));
    const auto* e = std::get_if<integration_error::NonProgressingInterval>(&error);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->interval, domain);
}

TEST(Adaptive, OneDimensionalNonfiniteIntegrandIsPropagated)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature1d::builder(interval(-1.0, 1.0), rule(1), rule(3), absolute(1e-8))
                   .build());

    const auto error = unwrap_err(
        quadrature.integrate([](double) { return std::numeric_limits<double>::quiet_NaN(); }));
    const auto* e = std::get_if<integration_error::NonFiniteIntegrand>(&error);
    ASSERT_NE(e, nullptr);
    EXPECT_TRUE(std::holds_alternative<evaluation_point::OneDimensional>(e->point));
    EXPECT_TRUE(std::isnan(e->value));
}
//}}}

//{{{ collection: two-dimensional tests
TEST(Adaptive, TwoDimensionalBuilderValidatesEachAxisExactness)
{
    const auto error = unwrap_err(AdaptiveQuadrature2d::builder(rectangle(-1.0, 1.0, -1.0, 1.0),
                                                                TensorRule2d{rule(2), rule(5)},
                                                                TensorRule2d{rule(3), rule(4)},
                                                                absolute(1e-8))
                                      .build());

    ASSERT_EQ(error.issues().size(), 2U);
    EXPECT_EQ(error.issues()[0], non_increasing(RuleAxis::U, 3, 3));
    EXPECT_EQ(error.issues()[1], non_increasing(RuleAxis::V, 5, 5));
}

TEST(Adaptive, TwoDimensionalFreeFunctionBuildsAndIntegratesOnce)
{
    auto builder = AdaptiveQuadrature2d::builder(
        rectangle(0.0, 1.0, 0.0, 1.0), tensor_rule(3), tensor_rule(5), absolute(1e-12));

    const auto result = unwrap(
        adaptive_quad_2d([](double u, double v) { return u * u + v * v; }, std::move(builder)));
    EXPECT_NEAR(result.integral(), 2.0 / 3.0, 1e-14);
    EXPECT_EQ(result.evaluation_count(), 13U);
}

TEST(Adaptive, TwoDimensionalPolynomialUsesOneTypedRegion)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature2d::builder(
                   rectangle(-1.0, 1.0, -1.0, 1.0), tensor_rule(3), tensor_rule(5), absolute(1e-12))
                   .build());
    std::size_t calls = 0;

    const auto result = unwrap(quadrature.integrate(
        [&](double u, double v)
        {
            ++calls;
            return u * u + v * v;
        }));

    EXPECT_NEAR(result.integral(), 8.0 / 3.0, 1e-14);
    EXPECT_EQ(result.terminal_region_count(), 1U);
    EXPECT_EQ(result.evaluation_count(), 13U);
    EXPECT_EQ(calls, result.evaluation_count());
}

TEST(Adaptive, TwoDimensionalSplitsBothAvailableAxes)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature2d::builder(
                   rectangle(-1.0, 1.0, -1.0, 1.0), tensor_rule(1), tensor_rule(3), absolute(1e-12))
                   .max_depth(AxisDepths::from_values(1, 1))
                   .build());

    const auto result = unwrap(
        quadrature.integrate([](double u, double v) { return indicator(u > 0.0 && v > 0.0); }));
    EXPECT_NEAR(result.integral(), 1.0, 1e-14);
    EXPECT_EQ(result.terminal_region_count(), 4U);
    EXPECT_EQ(result.evaluation_count(), 25U);
}

TEST(Adaptive, TwoDimensionalSplitsOnlyAxisWithRemainingDepth)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature2d::builder(
                   rectangle(-1.0, 1.0, -1.0, 1.0), tensor_rule(1), tensor_rule(3), absolute(1e-12))
                   .max_depth(AxisDepths::from_values(0, 1))
                   .build());

    const auto result =
        unwrap(quadrature.integrate([](double, double v) { return indicator(v > 0.0); }));
    EXPECT_NEAR(result.integral(), 2.0, 1e-14);
    EXPECT_EQ(result.terminal_region_count(), 2U);
    EXPECT_EQ(result.evaluation_count(), 15U);
}

TEST(Adaptive, TwoDimensionalDepthExhaustionContainsPartialResult)
{
    const auto quadrature =
        unwrap(AdaptiveQuadrature2d::builder(
                   rectangle(-1.0, 1.0, -1.0, 1.0), tensor_rule(1), tensor_rule(3), absolute(1e-12))
                   .max_depth(AxisDepths::from_values(0, 0))
                   .build());

    const auto error = unwrap_err(
        quadrature.integrate([](double u, double v) { return indicator(u > 0.0 && v > 0.0); }));
    const auto* depth = std::get_if<integration_error::MaxDepthReached>(&error);
    ASSERT_NE(depth, nullptr) << "expected depth exhaustion";
    EXPECT_NEAR(depth->partial_result.integral(), 1.0, 1e-14);
    EXPECT_EQ(depth->partial_result.terminal_region_count(), 1U);
    EXPECT_EQ(depth->partial_result.evaluation_count(), 5U);
}

TEST(Adaptive, TwoDimensionalNonprogressingMidpointIsReported)
{
    const double lower = 1.0;
    const double upper = next_up(lower);
    const Rectangle domain{interval(lower, upper), interval(-1.0, 1.0)};
    const auto quadrature = unwrap(
        AdaptiveQuadrature2d::builder(
            domain, tensor_rule(1), tensor_rule(3), absolute(std::numeric_limits<double>::min()))
            .max_depth(AxisDepths::from_values(1, 0))
            .build());

    const auto error = unwrap_err(
        quadrature.integrate([&](double u, double) { return u == upper ? 1e300 : 0.0; }));
    const auto* e = std::get_if<integration_error::NonProgressingRectangle>(&error);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->rectangle, domain);
}

TEST(Adaptive, TwoDimensionalInitialSubdivisionsFormAGrid)
{
    const auto quadrature = unwrap(
        unwrap(AdaptiveQuadrature2d::builder(
                   rectangle(-1.0, 1.0, -1.0, 1.0), tensor_rule(3), tensor_rule(5), absolute(1e-12))
                   .subdivisions({0.0}, {0.0}))
            .build());

    const auto result =
        unwrap(quadrature.integrate([](double u, double v) { return u * u + v * v; }));
    EXPECT_TRUE(std::ranges::equal(quadrature.u_subdivision_points(), std::vector{0.0}));
    EXPECT_TRUE(std::ranges::equal(quadrature.v_subdivision_points(), std::vector{0.0}));
    EXPECT_EQ(result.terminal_region_count(), 4U);
    EXPECT_EQ(result.evaluation_count(), 52U);
}
//}}}
