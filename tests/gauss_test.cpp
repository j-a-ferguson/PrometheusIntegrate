//! Port of `tests/gauss.rs`.

#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace prometheus::integrate;
using test_support::count;
using test_support::degree;
using test_support::unwrap;
using test_support::unwrap_err;

namespace
{

constexpr double MAX_REL = 1e-10;
constexpr double MAX_ABS = 1e-10;

struct ExpectedRule
{
    std::vector<double> points;
    std::vector<double> weights;
};

/// Rust `ExpectedData::load()`, parsed once and shared.
const nlohmann::json& expected_data()
{
    static const nlohmann::json data = test_support::load_asset("gauss-quad.json");
    return data;
}

ExpectedRule expected_rule(const std::string& family, const std::string& dataset)
{
    const auto& entry = expected_data().at(family).at("values").at(dataset);
    return {entry.at("points").get<std::vector<double>>(),
            entry.at("weights").get<std::vector<double>>()};
}

void assert_matches_expected(const GaussRule& actual, const ExpectedRule& expected)
{
    ASSERT_EQ(actual.point_count().value(), expected.points.size());
    for (std::size_t i = 0; i < expected.points.size(); ++i)
    {
        // `assert_relative_eq!(.., epsilon = 1e-10)` is an absolute bound.
        EXPECT_NEAR(actual.points()[i], expected.points[i], MAX_REL) << "point " << i;
        EXPECT_NEAR(actual.weights()[i], expected.weights[i], MAX_REL) << "weight " << i;
    }
}

} // namespace

//{{{ collection: legendre_test! / lobatto_test! → value-parameterised tests
class ReferenceRuleTest : public ::testing::TestWithParam<std::tuple<GaussFamily, std::size_t>>
{
};

TEST_P(ReferenceRuleTest, MatchesReferenceData)
{
    const auto [family, point_count] = GetParam();
    const auto rules = unwrap(GaussRuleSet::through_degree(family, degree(90)));
    const auto rule_ref = unwrap(rules.rule_by_point_count(count(point_count)));
    const GaussRule& rule = rule_ref.get();
    assert_matches_expected(rule,
                            expected_rule(family == GaussFamily::Legendre ? "legendre" : "lobatto",
                                          std::format("n{}", point_count)));
}

INSTANTIATE_TEST_SUITE_P(
    Gauss,
    ReferenceRuleTest,
    ::testing::Combine(::testing::Values(GaussFamily::Legendre, GaussFamily::Lobatto),
                       ::testing::Values(2U, 3U, 4U, 5U, 6U, 11U, 26U, 37U)),
    [](const auto& param_info) {
        return std::format("{}_n{}", std::get<0>(param_info.param), std::get<1>(param_info.param));
    });
//}}}

namespace
{

double integrate_monomial(const GaussRule& rule, std::size_t degree)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < rule.points().size(); ++i)
    {
        sum += rule.weights()[i] * test_support::powi(rule.points()[i], static_cast<int>(degree));
    }
    return sum;
}

double exact_monomial_integral(std::size_t degree)
{
    return degree % 2 == 0 ? 2.0 / static_cast<double>(degree + 1) : 0.0;
}

void assert_rule_structure(const GaussRule& rule)
{
    const std::size_t n = rule.point_count().value();
    ASSERT_EQ(rule.points().size(), n);
    ASSERT_EQ(rule.weights().size(), n);
    for (const double point : rule.points())
    {
        EXPECT_TRUE(std::isfinite(point));
    }
    double weight_sum = 0.0;
    for (const double weight : rule.weights())
    {
        EXPECT_TRUE(std::isfinite(weight) && weight > 0.0);
        weight_sum += weight;
    }
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        EXPECT_LT(rule.points()[i], rule.points()[i + 1]);
    }
    EXPECT_NEAR(weight_sum, 2.0, MAX_ABS);

    for (std::size_t index = 0; index < n; ++index)
    {
        const std::size_t mirror = n - index - 1;
        EXPECT_NEAR(rule.points()[index], -rule.points()[mirror], MAX_ABS);
        EXPECT_NEAR(rule.weights()[index], rule.weights()[mirror], MAX_ABS);
    }

    if (rule.family() == GaussFamily::Lobatto)
    {
        EXPECT_EQ(rule.points().front(), -1.0);
        EXPECT_EQ(rule.points().back(), 1.0);
    }
}

std::size_t family_minimum(GaussFamily family)
{
    return family == GaussFamily::Legendre ? 1 : 2;
}

} // namespace

TEST(Gauss, MinimumRulesAreSupported)
{
    const auto legendre = unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(0)));
    EXPECT_EQ(legendre.point_count(), count(1));
    EXPECT_EQ(legendre.exactness(), degree(1));
    EXPECT_EQ(std::vector<double>(legendre.points().begin(), legendre.points().end()),
              std::vector<double>{0.0});
    EXPECT_EQ(std::vector<double>(legendre.weights().begin(), legendre.weights().end()),
              std::vector<double>{2.0});

    const auto lobatto_two = unwrap(GaussRule::with_point_count(GaussFamily::Lobatto, count(2)));
    EXPECT_EQ(lobatto_two.exactness(), degree(1));
    EXPECT_EQ(std::vector<double>(lobatto_two.points().begin(), lobatto_two.points().end()),
              (std::vector<double>{-1.0, 1.0}));
    EXPECT_EQ(std::vector<double>(lobatto_two.weights().begin(), lobatto_two.weights().end()),
              (std::vector<double>{1.0, 1.0}));

    const auto lobatto_three = unwrap(GaussRule::with_point_count(GaussFamily::Lobatto, count(3)));
    EXPECT_EQ(lobatto_three.exactness(), degree(3));
    EXPECT_EQ(std::vector<double>(lobatto_three.points().begin(), lobatto_three.points().end()),
              (std::vector<double>{-1.0, 0.0, 1.0}));
    EXPECT_NEAR(lobatto_three.weights()[0], 1.0 / 3.0, MAX_ABS);
    EXPECT_NEAR(lobatto_three.weights()[1], 4.0 / 3.0, MAX_ABS);
    EXPECT_NEAR(lobatto_three.weights()[2], 1.0 / 3.0, MAX_ABS);
}

TEST(Gauss, RequestedDegreeSelectsTheMinimumRule)
{
    for (const GaussFamily family : {GaussFamily::Legendre, GaussFamily::Lobatto})
    {
        for (std::size_t d = 0; d <= 100; ++d)
        {
            const auto rule = unwrap(GaussRule::for_degree(family, degree(d)));
            EXPECT_GE(rule.exactness().value(), d);

            if (rule.point_count().value() > family_minimum(family))
            {
                const auto previous = unwrap(
                    GaussRule::with_point_count(family, count(rule.point_count().value() - 1)));
                EXPECT_LT(previous.exactness().value(), d);
            }
        }
    }
}

TEST(Gauss, RulesIntegrateMonomialsThroughTheirExactness)
{
    for (const GaussFamily family : {GaussFamily::Legendre, GaussFamily::Lobatto})
    {
        for (std::size_t point_count = family_minimum(family); point_count <= 10; ++point_count)
        {
            const auto rule = unwrap(GaussRule::with_point_count(family, count(point_count)));
            assert_rule_structure(rule);

            for (std::size_t d = 0; d <= rule.exactness().value(); ++d)
            {
                EXPECT_NEAR(integrate_monomial(rule, d), exact_monomial_integral(d), MAX_ABS)
                    << to_string(family) << " n=" << point_count << " degree " << d;
            }
        }
    }
}

TEST(Gauss, RuleSetLookupAndIterationBorrowStoredRules)
{
    const auto rules = unwrap(GaussRuleSet::through_degree(GaussFamily::Legendre, degree(10)));
    EXPECT_EQ(rules.family(), GaussFamily::Legendre);
    EXPECT_EQ(rules.maximum_degree(), degree(10));
    EXPECT_EQ(rules.minimum_point_count(), count(1));
    EXPECT_EQ(rules.maximum_point_count(), count(6));
    EXPECT_EQ(rules.size(), 6U);
    EXPECT_FALSE(rules.empty());

    std::vector<std::size_t> counts;
    for (const GaussRule& rule : rules)
    {
        counts.push_back(rule.point_count().value());
    }
    EXPECT_EQ(counts, (std::vector<std::size_t>{1, 2, 3, 4, 5, 6}));

    const auto by_degree_ref = unwrap(rules.rule_for_degree(degree(10)));
    const GaussRule& by_degree = by_degree_ref.get();
    const auto by_count_ref = unwrap(rules.rule_by_point_count(count(6)));
    const GaussRule& by_count = by_count_ref.get();
    EXPECT_EQ(&by_degree, &by_count); // Rust `ptr::eq`
}

TEST(Gauss, CachedRuleSetsAreBorrowedAndComplete)
{
    const auto legendre_ref = unwrap(legendre_rules());
    const GaussRuleSet& legendre = legendre_ref.get();
    EXPECT_EQ(&legendre, &unwrap(legendre_rules()).get());
    const auto legendre_max_ref = unwrap(legendre.rule_for_degree(degree(100)));
    const GaussRule& legendre_max = legendre_max_ref.get();
    EXPECT_EQ(legendre_max.point_count(), count(51));
    assert_rule_structure(legendre_max);
    EXPECT_NEAR(integrate_monomial(legendre_max, 100), exact_monomial_integral(100), MAX_ABS);

    const auto lobatto_ref = unwrap(lobatto_rules());
    const GaussRuleSet& lobatto = lobatto_ref.get();
    EXPECT_EQ(&lobatto, &unwrap(lobatto_rules()).get());
    const auto lobatto_max_ref = unwrap(lobatto.rule_for_degree(degree(100)));
    const GaussRule& lobatto_max = lobatto_max_ref.get();
    EXPECT_EQ(lobatto_max.point_count(), count(52));
    assert_rule_structure(lobatto_max);
    EXPECT_NEAR(integrate_monomial(lobatto_max, 100), exact_monomial_integral(100), MAX_ABS);
}

TEST(Gauss, UnsupportedDegreesAndPointCountsReturnErrors)
{
    EXPECT_EQ(
        GaussRule::for_degree(GaussFamily::Legendre, degree(101)),
        (std::unexpected{RuleError{rule_error::DegreeOutOfRange{.degree = 101, .maximum = 100}}}));

    constexpr std::size_t usize_max = std::numeric_limits<std::size_t>::max();
    for (const std::size_t invalid : {std::size_t{0}, usize_max})
    {
        const auto error = unwrap_err(PointCount::create(invalid));
        ASSERT_EQ(error.issues().size(), 1U);
        EXPECT_EQ(error.issues()[0],
                  (ConfigIssue{config_issue::PointCountOutOfRange{
                      .point_count = invalid, .minimum = 1, .maximum = 52}}));
    }

    {
        const auto error = unwrap_err(GaussRule::with_point_count(GaussFamily::Lobatto, count(1)));
        const auto* e = std::get_if<rule_error::PointCountOutOfRange>(&error);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->point_count, 1U);
        EXPECT_EQ(e->minimum, 2U);
        EXPECT_EQ(e->maximum, 52U);
    }
    {
        const auto error =
            unwrap_err(GaussRule::with_point_count(GaussFamily::Legendre, count(52)));
        const auto* e = std::get_if<rule_error::PointCountOutOfRange>(&error);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->point_count, 52U);
        EXPECT_EQ(e->minimum, 1U);
        EXPECT_EQ(e->maximum, 51U);
    }

    const auto rules = unwrap(GaussRuleSet::through_degree(GaussFamily::Lobatto, degree(8)));
    EXPECT_EQ(unwrap_err(rules.rule_for_degree(degree(9))),
              (RuleError{rule_error::DegreeOutOfRange{.degree = 9, .maximum = 8}}));
    {
        const auto error = unwrap_err(rules.rule_by_point_count(count(7)));
        const auto* e = std::get_if<rule_error::PointCountOutOfRange>(&error);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->point_count, 7U);
        EXPECT_EQ(e->minimum, 2U);
        EXPECT_EQ(e->maximum, 6U);
    }

    const auto degree_error = unwrap_err(GaussRule::for_degree(GaussFamily::Legendre, degree(101)));
    EXPECT_EQ(to_string(degree_error),
              "polynomial degree 101 exceeds the supported maximum of 100");
    const auto count_error =
        unwrap_err(GaussRule::with_point_count(GaussFamily::Lobatto, count(1)));
    EXPECT_EQ(to_string(count_error), "lobatto point count 1 is outside supported range 2..=52");
}
