//! Ports of the in-module `#[cfg(test)] mod tests` blocks from `src/config.rs` and
//! `src/gauss.rs`, which exercise crate-private functions (exposed here via `detail::`).

#include "test_support.hpp"

#include <limits>
#include <ranges>
#include <variant>
#include <vector>

using namespace prometheus::integrate;
using test_support::interval;
using test_support::unwrap;
using test_support::unwrap_err;

//{{{ collection: config.rs
TEST(ConfigInternal, SubdivisionValidationAcceptsAnyIteratorAndNormalizesEmptyInput)
{
    const auto i = interval(-1.0, 1.0);
    EXPECT_EQ(unwrap(detail::validate_subdivisions(i, {-0.5, 0.0, 0.5})),
              (std::vector{-0.5, 0.0, 0.5}));
    EXPECT_EQ(
        unwrap(detail::validate_subdivisions(i, detail::collect_points(std::views::iota(0, 0)))),
        std::vector<double>{});
    // Any input range of doubles is accepted, e.g. a lazily transformed view.
    EXPECT_EQ(unwrap(detail::validate_subdivisions(
                  i,
                  detail::collect_points(std::views::iota(1, 4) |
                                         std::views::transform([](int k) { return k * 0.25; })))),
              (std::vector{0.25, 0.5, 0.75}));
}

TEST(ConfigInternal, SubdivisionValidationCollectsAllIssues)
{
    const auto i = interval(-1.0, 1.0);
    const auto error = unwrap_err(detail::validate_subdivisions(
        i, {0.5, 0.5, std::numeric_limits<double>::quiet_NaN(), 2.0}));
    ASSERT_EQ(error.issues().size(), 3U);
    EXPECT_TRUE(
        std::holds_alternative<config_issue::SubdivisionsNotStrictlyIncreasing>(error.issues()[0]));
    EXPECT_TRUE(std::holds_alternative<config_issue::NonFiniteSubdivision>(error.issues()[1]));
    EXPECT_TRUE(
        std::holds_alternative<config_issue::SubdivisionOutsideInterval>(error.issues()[2]));
}
//}}}

//{{{ collection: gauss.rs
TEST(GaussInternal, InvalidBackendOutputIsAGenerationError)
{
    const auto error =
        unwrap_err(detail::gauss_rule_from_points_weights(GaussFamily::Legendre, {0.0}, {}));
    EXPECT_EQ(error,
              (RuleError{rule_error::GenerationFailed{
                  .family = GaussFamily::Legendre,
                  .point_count = 1,
                  .message = "backend produced 1 points but 0 weights"}}));
}

TEST(GaussInternal, InvalidEigendecompositionSizeIsAGenerationError)
{
    const auto error = unwrap_err(
        detail::golub_welsch(GaussFamily::Legendre, 1, 0, 2.0, detail::legendre_recursion_coeffs));
    EXPECT_EQ(error,
              (RuleError{rule_error::GenerationFailed{
                  .family = GaussFamily::Legendre,
                  .point_count = 1,
                  .message = "eigendecomposition matrix must be nonempty"}}));
}
//}}}
