//! Port of `tests/traits.rs`: compile-time assertions for the library's value-type policy.
//!
//! Rust trait → C++ concept used here:
//!   Clone + PartialEq  → std::copy_constructible + std::equality_comparable
//!   Copy               → std::is_trivially_copyable_v
//!   Eq + Hash          → std::equality_comparable + an enabled std::hash specialization
//!   Display / Error    → an enabled std::formatter specialization (`Displayable`)
//!   Default            → std::default_initializable
//!   Send + Sync        → no C++ equivalent; every type here is a plain value type with no
//!                        interior mutability, so concurrent `const` access is safe.

#include "test_support.hpp"

#include <concepts>
#include <functional>
#include <limits>
#include <type_traits>

using namespace prometheus::integrate;
using test_support::unwrap;

namespace
{

template <class T>
concept Hashable = requires(const T& value) {
    { std::hash<T>{}(value) } -> std::convertible_to<std::size_t>;
};

template <class T>
concept ClonePartialEq = std::copy_constructible<T> && std::equality_comparable<T>;

template <class T>
concept CopyEqHash = std::is_trivially_copyable_v<T> && std::equality_comparable<T> && Hashable<T>;

template <class T>
concept Copy = std::is_trivially_copyable_v<T>;

} // namespace

/// Discrete enums and newtypes carry the full value-type trait set.
TEST(Traits, DiscreteTypesAreCopyEqAndHash)
{
    static_assert(CopyEqHash<PolynomialDegree>);
    static_assert(CopyEqHash<PointCount>);
    static_assert(CopyEqHash<RefinementDepth>);
    static_assert(CopyEqHash<AxisDepths>);
    static_assert(CopyEqHash<RuleAxis>);
    static_assert(CopyEqHash<GaussFamily>);

    // `RuleError` is float-free but owns a diagnostic string, so it is hashable but not `Copy`.
    static_assert(std::equality_comparable<RuleError> && Hashable<RuleError>);
    static_assert(!Copy<RuleError>);
}

/// Float-bearing values, rules, configurations, and results are cloneable and comparable.
TEST(Traits, FloatBearingTypesAreCloneAndPartialEq)
{
    static_assert(ClonePartialEq<Interval>);
    static_assert(ClonePartialEq<Rectangle>);
    static_assert(ClonePartialEq<Tolerance>);
    static_assert(ClonePartialEq<ConfigIssue>);
    static_assert(ClonePartialEq<ConfigError>);
    static_assert(ClonePartialEq<GaussRule>);
    static_assert(ClonePartialEq<GaussRuleSet>);
    static_assert(ClonePartialEq<EvaluationPoint>);
    static_assert(ClonePartialEq<AdaptiveResult>);
    static_assert(ClonePartialEq<IntegrationError>);
    static_assert(ClonePartialEq<OptionsError>);
    static_assert(ClonePartialEq<FixedNode1d>);
    static_assert(ClonePartialEq<FixedNode2d>);
    static_assert(ClonePartialEq<TensorRule2d>);
    static_assert(ClonePartialEq<FixedQuadratureBuilder1d>);
    static_assert(ClonePartialEq<FixedQuadratureBuilder2d>);
    static_assert(ClonePartialEq<FixedQuadrature1d>);
    static_assert(ClonePartialEq<FixedQuadrature2d>);
    static_assert(ClonePartialEq<AdaptiveQuadratureBuilder1d>);
    static_assert(ClonePartialEq<AdaptiveQuadratureBuilder2d>);
    static_assert(ClonePartialEq<AdaptiveQuadrature1d>);
    static_assert(ClonePartialEq<AdaptiveQuadrature2d>);

    // Small, allocation-free value types are additionally `Copy`.
    static_assert(Copy<Interval>);
    static_assert(Copy<Rectangle>);
    static_assert(Copy<Tolerance>);
    static_assert(Copy<EvaluationPoint>);
    static_assert(Copy<FixedNode1d>);
    static_assert(Copy<FixedNode2d>);

    // ...and owning types are not.
    static_assert(!Copy<GaussRule>);
    static_assert(!Copy<ConfigError>);
}

/// Errors and domain values format themselves for end users.
TEST(Traits, ErrorsAndDomainValuesImplementDisplay)
{
    static_assert(Displayable<ConfigError>);
    static_assert(Displayable<RuleError>);
    static_assert(Displayable<IntegrationError>);
    static_assert(Displayable<OptionsError>);

    static_assert(Displayable<ConfigIssue>);
    static_assert(Displayable<PolynomialDegree>);
    static_assert(Displayable<PointCount>);
    static_assert(Displayable<Interval>);
    static_assert(Displayable<Rectangle>);
    static_assert(Displayable<Tolerance>);
    static_assert(Displayable<RefinementDepth>);
    static_assert(Displayable<AxisDepths>);
    static_assert(Displayable<RuleAxis>);
    static_assert(Displayable<GaussFamily>);
    static_assert(Displayable<EvaluationPoint>);

    // Not every type is printable: quadratures and builders have no `Display` in Rust either.
    static_assert(!Displayable<FixedQuadrature1d>);
}

/// `Default` is implemented only where a meaningful default exists.
TEST(Traits, OnlyRefinementDepthsImplementDefault)
{
    static_assert(std::default_initializable<RefinementDepth>);
    static_assert(std::default_initializable<AxisDepths>);

    static_assert(!std::default_initializable<PolynomialDegree>);
    static_assert(!std::default_initializable<PointCount>);
    static_assert(!std::default_initializable<Interval>);
    static_assert(!std::default_initializable<Tolerance>);
    static_assert(!std::default_initializable<GaussRule>);

    EXPECT_EQ(RefinementDepth{}.value(), RefinementDepth::DEFAULT);
    EXPECT_EQ(AxisDepths{}, AxisDepths::uniform(RefinementDepth{}));
}

TEST(Traits, DisplayOutputIsConciseAndLowercase)
{
    EXPECT_EQ(to_string(unwrap(Tolerance::create(1e-8, 1e-6))),
              "absolute 0.00000001, relative 0.000001");
    EXPECT_EQ(to_string(AxisDepths::from_values(4, 8)), "u 4, v 8");
    EXPECT_EQ(to_string(EvaluationPoint{evaluation_point::OneDimensional{0.5}}), "x = 0.5");
    EXPECT_EQ(to_string(EvaluationPoint{evaluation_point::TwoDimensional{.u = 0.5, .v = -1.5}}),
              "(u, v) = (0.5, -1.5)");
    EXPECT_EQ(to_string(IntegrationError{integration_error::NonFiniteIntegrand{
                  .point = evaluation_point::OneDimensional{0.5},
                  .value = std::numeric_limits<double>::infinity()}}),
              "integrand returned non-finite value inf at x = 0.5");
}
