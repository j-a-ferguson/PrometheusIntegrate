//! Port of `tests/config.rs`.

#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

using namespace prometheus::integrate;
using test_support::interval;
using test_support::unwrap;
using test_support::unwrap_err;

namespace
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

template <class Issue>
bool single_issue_is(const ConfigError& error)
{
    return error.issues().size() == 1 && std::holds_alternative<Issue>(error.issues()[0]);
}
} // namespace

TEST(Config, DegreeAndPointCountValidateBoundaries)
{
    EXPECT_EQ(unwrap(PolynomialDegree::create(0)).value(), 0U);
    EXPECT_EQ(unwrap(PolynomialDegree::try_from(PolynomialDegree::MAXIMUM)).value(), 101U);
    const auto degree_error = unwrap_err(PolynomialDegree::create(102));
    ASSERT_EQ(degree_error.issues().size(), 1U);
    EXPECT_EQ(
        degree_error.issues()[0],
        (ConfigIssue{config_issue::PolynomialDegreeOutOfRange{.degree = 102, .maximum = 101}}));

    EXPECT_EQ(unwrap(PointCount::create(PointCount::MINIMUM)).value(), 1U);
    EXPECT_EQ(unwrap(PointCount::try_from(PointCount::MAXIMUM)).value(), 52U);
    for (const std::size_t invalid :
         {std::size_t{0}, std::size_t{53}, std::numeric_limits<std::size_t>::max()})
    {
        const auto error = unwrap_err(PointCount::create(invalid));
        ASSERT_EQ(error.issues().size(), 1U);
        const auto* issue = std::get_if<config_issue::PointCountOutOfRange>(&error.issues()[0]);
        ASSERT_NE(issue, nullptr);
        EXPECT_EQ(issue->point_count, invalid);
    }
}

TEST(Config, IntervalRequiresFiniteStrictlyIncreasingBounds)
{
    const auto i = interval(-2.0, 3.0);
    EXPECT_EQ(i.lower(), -2.0);
    EXPECT_EQ(i.upper(), 3.0);
    EXPECT_EQ(i.bounds(), (std::pair{-2.0, 3.0}));
    EXPECT_EQ(i.length(), 5.0);
    EXPECT_EQ(to_string(i), "[-2, 3]");

    for (const auto& [lower, upper] :
         {std::pair{NaN, 1.0}, std::pair{0.0, INF}, std::pair{-INF, INF}})
    {
        EXPECT_TRUE(single_issue_is<config_issue::NonFiniteInterval>(
            unwrap_err(Interval::create(lower, upper))));
    }

    for (const auto& [lower, upper] : {std::pair{1.0, 1.0}, std::pair{2.0, -2.0}})
    {
        EXPECT_TRUE(single_issue_is<config_issue::InvalidIntervalOrder>(
            unwrap_err(Interval::create(lower, upper))));
    }
}

TEST(Config, ConfigurationIssuesRetainOffendingValuesAndUseConciseMessages)
{
    const auto nonfinite = unwrap_err(Interval::create(NaN, 2.0));
    ASSERT_EQ(nonfinite.issues().size(), 1U);
    const auto* issue = std::get_if<config_issue::NonFiniteInterval>(&nonfinite.issues()[0]);
    ASSERT_NE(issue, nullptr);
    EXPECT_TRUE(std::isnan(issue->lower));
    EXPECT_EQ(issue->upper, 2.0);
    EXPECT_EQ(to_string(nonfinite),
              "invalid configuration; interval bounds must be finite, received (NaN, 2)");

    const auto tolerance = unwrap_err(Tolerance::create(-1.0, INF));
    ASSERT_EQ(tolerance.issues().size(), 1U);
    const auto* tol = std::get_if<config_issue::InvalidTolerance>(&tolerance.issues()[0]);
    ASSERT_NE(tol, nullptr);
    EXPECT_EQ(tol->absolute, -1.0);
    EXPECT_TRUE(std::isinf(tol->relative));
    EXPECT_FALSE(to_string(tolerance).ends_with('.'));
}

TEST(Config, RectangleContainsValidatedAxisIntervals)
{
    const auto u = interval(-1.0, 1.0);
    const auto v = interval(2.0, 5.0);
    const Rectangle rectangle{u, v};
    EXPECT_EQ(rectangle.u(), u);
    EXPECT_EQ(rectangle.v(), v);
    EXPECT_EQ(to_string(rectangle), "[-1, 1] × [2, 5]");

    const auto error = unwrap_err(Rectangle::from_bounds(1.0, 1.0, NaN, 2.0));
    ASSERT_EQ(error.issues().size(), 2U);
    EXPECT_TRUE(std::holds_alternative<config_issue::InvalidIntervalOrder>(error.issues()[0]));
    EXPECT_TRUE(std::holds_alternative<config_issue::NonFiniteInterval>(error.issues()[1]));
}

TEST(Config, ToleranceRequiresFiniteNonnegativeNonzeroComponents)
{
    const auto mixed = unwrap(Tolerance::create(1e-8, 1e-6));
    EXPECT_EQ(mixed.absolute_value(), 1e-8);
    EXPECT_EQ(mixed.relative_value(), 1e-6);

    const auto absolute = unwrap(Tolerance::absolute(1e-8));
    EXPECT_EQ(absolute.absolute_value(), 1e-8);
    EXPECT_EQ(absolute.relative_value(), 0.0);

    const auto relative = unwrap(Tolerance::relative(1e-6));
    EXPECT_EQ(relative.absolute_value(), 0.0);
    EXPECT_EQ(relative.relative_value(), 1e-6);

    for (const auto& [abs, rel] : {std::pair{0.0, 0.0},
                                   std::pair{-1.0, 0.0},
                                   std::pair{0.0, -1.0},
                                   std::pair{NaN, 1.0},
                                   std::pair{1.0, INF}})
    {
        EXPECT_TRUE(single_issue_is<config_issue::InvalidTolerance>(
            unwrap_err(Tolerance::create(abs, rel))));
    }
}

TEST(Config, RefinementDepthZeroIsValidAndDefaultsTo32)
{
    EXPECT_EQ(RefinementDepth{0}.value(), 0U);
    EXPECT_EQ(RefinementDepth{}.value(), 32U);

    const auto axes = AxisDepths::from_values(0, 7);
    EXPECT_EQ(axes.u().value(), 0U);
    EXPECT_EQ(axes.v().value(), 7U);

    const AxisDepths default_axes{};
    EXPECT_EQ(default_axes.u().value(), 32U);
    EXPECT_EQ(default_axes.v().value(), 32U);
}

TEST(Config, FixedBuilderUsesStrictSubdivisionValidation)
{
    const auto domain = interval(-1.0, 1.0);
    const auto rule =
        unwrap(GaussRule::for_degree(GaussFamily::Legendre, unwrap(PolynomialDegree::create(3))));
    const auto make = [&](std::vector<double> subdiv)
    { return FixedQuadrature1d::builder(domain, rule).subdivisions(std::move(subdiv)); };

    const auto without_subdivisions = FixedQuadrature1d::builder(domain, rule).build();
    const auto empty_subdivisions = unwrap(make({})).build();
    EXPECT_TRUE(std::ranges::equal(without_subdivisions.nodes(), empty_subdivisions.nodes()));

    for (auto invalid :
         std::vector<std::vector<double>>{{NaN}, {0.0, 0.0}, {0.5, -0.5}, {-1.0}, {1.0}})
    {
        EXPECT_FALSE(make(std::move(invalid)).has_value());
    }
}
