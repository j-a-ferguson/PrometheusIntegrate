#pragma once
//! Gauss quadrature rules for one-dimensional real-valued integrands.
//!
//! Gauss quadrature approximates an integral with a weighted sum of function values at points
//! associated with an orthogonal-polynomial family.
//!
//! The eigendecomposition backend (Armadillo) is an implementation detail of `gauss.cpp` and
//! does not appear in this header.

//{{{ crate includes
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/detail/display.hpp"
//}}}
//{{{ std includes
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
//}}}

namespace prometheus::integrate
{

/// Rust `&'a T` inside a `Result`: `std::expected` cannot hold a reference, so borrowed results
/// are returned as non-null `std::reference_wrapper<const T>`.
template <class T>
using Ref = std::reference_wrapper<const T>;

//{{{ enum: GaussFamily
/// A supported Gauss quadrature family.
enum class GaussFamily : std::uint8_t
{
    /// Gauss-Legendre quadrature on `[-1, 1]` with unit weight.
    Legendre,
    /// Gauss-Lobatto quadrature family on `[-1, 1]`; rules include both endpoints.
    Lobatto,
};

/// Returns the reference-interval integral of the weight used by this family's node-generating
/// recurrence.
///
/// For Lobatto rules, this is the integral of the Jacobi `(1, 1)` weight used to generate the
/// interior nodes, not the sum of the final Lobatto weights.
[[nodiscard]] double weight_integral(GaussFamily family) noexcept;

/// Returns this family's reference interval.
[[nodiscard]] std::pair<double, double> range(GaussFamily family) noexcept;
//}}}

//{{{ variant: RuleError
/// Alternatives of `RuleError`.
namespace rule_error
{

/// The requested polynomial degree exceeds the supported maximum.
struct DegreeOutOfRange
{
    std::size_t degree;
    std::size_t maximum;
    bool operator==(const DegreeOutOfRange&) const = default;
};

/// The requested point count is outside the supported range.
struct PointCountOutOfRange
{
    GaussFamily family;
    std::size_t point_count;
    std::size_t minimum;
    std::size_t maximum;
    bool operator==(const PointCountOutOfRange&) const = default;
};

/// The eigendecomposition backend could not generate a rule.
struct GenerationFailed
{
    GaussFamily family;
    std::size_t point_count;
    std::string message;
    bool operator==(const GenerationFailed&) const = default;
};

} // namespace rule_error

/// An error encountered while constructing or retrieving a Gaussian rule.
using RuleError = std::variant<rule_error::DegreeOutOfRange,
                               rule_error::PointCountOutOfRange,
                               rule_error::GenerationFailed>;
//}}}

class GaussRule;

namespace detail
{
/// Rust `GaussRule::from_points_weights` (module-private; exposed for the unit tests).
std::expected<GaussRule, RuleError> gauss_rule_from_points_weights(GaussFamily family,
                                                                   std::vector<double> points,
                                                                   std::vector<double> weights);
} // namespace detail

//{{{ class: GaussRule
/// A specific Gauss quadrature rule, represented by points and associated weights.
class GaussRule
{
public:
    /// Constructs the smallest quadrature rule with at least `degree` polynomial exactness.
    ///
    /// Legendre rules support one point and Lobatto rules support two points at minimum. An even
    /// requested degree selects a rule whose actual exactness is one degree higher.
    static std::expected<GaussRule, RuleError> for_degree(GaussFamily family,
                                                          PolynomialDegree degree);

    /// Constructs a quadrature rule containing exactly `point_count` points.
    static std::expected<GaussRule, RuleError> with_point_count(GaussFamily family,
                                                                PointCount point_count);

    /// Returns the quadrature family used to construct this rule.
    [[nodiscard]] GaussFamily family() const noexcept { return family_; }
    /// Returns the rule's actual polynomial exactness.
    [[nodiscard]] PolynomialDegree exactness() const noexcept { return exactness_; }
    /// Returns the number of quadrature points and weights.
    [[nodiscard]] PointCount point_count() const noexcept
    {
        return PointCount::new_unchecked(points_.size());
    }
    /// Returns the quadrature points in ascending order.
    [[nodiscard]] std::span<const double> points() const noexcept { return points_; }
    /// Returns the weights corresponding to `points()`.
    [[nodiscard]] std::span<const double> weights() const noexcept { return weights_; }

    bool operator==(const GaussRule&) const = default;

private:
    friend std::expected<GaussRule, RuleError> detail::gauss_rule_from_points_weights(
        GaussFamily, std::vector<double>, std::vector<double>);

    GaussRule(GaussFamily family,
              PolynomialDegree exactness,
              std::vector<double> points,
              std::vector<double> weights)
        : family_{family}, exactness_{exactness}, points_{std::move(points)},
          weights_{std::move(weights)}
    {
    }

    GaussFamily family_;
    PolynomialDegree exactness_;
    std::vector<double> points_;
    std::vector<double> weights_;
};
//}}}

//{{{ class: GaussRuleSet
/// A collection of Gauss quadrature rules through a maximum requested exactness degree.
class GaussRuleSet
{
public:
    using const_iterator = std::vector<GaussRule>::const_iterator;

    /// Builds every rule for `family` through the requested polynomial exactness `degree`.
    ///
    /// Fails with `DegreeOutOfRange` when `degree` exceeds 100, or `GenerationFailed` if the
    /// eigendecomposition backend fails.
    static std::expected<GaussRuleSet, RuleError> through_degree(GaussFamily family,
                                                                 PolynomialDegree degree);

    /// Returns the smallest stored rule whose exactness is at least `degree`.
    [[nodiscard]] std::expected<Ref<GaussRule>, RuleError>
    rule_for_degree(PolynomialDegree degree) const;

    /// Returns the stored rule containing exactly `point_count` points.
    [[nodiscard]] std::expected<Ref<GaussRule>, RuleError>
    rule_by_point_count(PointCount point_count) const;

    /// Returns the quadrature family shared by every rule in this set.
    [[nodiscard]] GaussFamily family() const noexcept { return family_; }
    /// Returns the largest requested polynomial degree represented by this set.
    [[nodiscard]] PolynomialDegree maximum_degree() const noexcept { return maximum_degree_; }
    /// Returns the smallest point count represented by this set.
    [[nodiscard]] PointCount minimum_point_count() const noexcept;
    /// Returns the largest point count represented by this set.
    [[nodiscard]] PointCount maximum_point_count() const noexcept;

    /// Rust `len` / `is_empty` / `iter` → the standard container protocol.
    [[nodiscard]] std::size_t size() const noexcept { return rules_.size(); }
    [[nodiscard]] bool empty() const noexcept { return rules_.empty(); }
    [[nodiscard]] const_iterator begin() const noexcept { return rules_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return rules_.end(); }

    bool operator==(const GaussRuleSet&) const = default;

private:
    GaussRuleSet(GaussFamily family, PolynomialDegree maximum_degree, std::vector<GaussRule> rules)
        : family_{family}, maximum_degree_{maximum_degree}, rules_{std::move(rules)}
    {
    }

    [[nodiscard]] std::expected<Ref<GaussRule>, RuleError>
    rule_for_degree_value(std::size_t degree) const;
    [[nodiscard]] std::expected<Ref<GaussRule>, RuleError>
    rule_by_point_count_value(std::size_t point_count) const;

    GaussFamily family_;
    PolynomialDegree maximum_degree_;
    std::vector<GaussRule> rules_;
};
//}}}

//{{{ fun: legendre_rules / lobatto_rules
/// Returns the cached Gauss-Legendre rules through degree 100.
///
/// Repeated calls borrow the same process-wide rule set (a function-local `static`, the C++
/// analogue of `OnceLock`). The error, if any, is the deterministic rule-generation error
/// recorded while initializing the cache.
[[nodiscard]] std::expected<Ref<GaussRuleSet>, RuleError> legendre_rules();

/// Returns the cached Gauss-Lobatto rules through degree 100.
[[nodiscard]] std::expected<Ref<GaussRuleSet>, RuleError> lobatto_rules();
//}}}

namespace detail
{

/// Recurrence coefficients `(a_i, b_i, c_i)` of a three-term orthogonal-polynomial recurrence
/// `p_{i+1}(x) = (a_i x + b_i) p_i(x) - c_i p_{i-1}(x)`.
using Recurrence = std::tuple<double, double, double> (*)(std::size_t);

/// Golub-Welsch: builds the symmetric Jacobi matrix for the recurrence and returns its
/// eigenvalues (points) and the squared first eigenvector components scaled by
/// `weight_integral` (weights), sorted by point.
std::expected<std::pair<std::vector<double>, std::vector<double>>, RuleError>
golub_welsch(GaussFamily family,
             std::size_t rule_point_count,
             std::size_t matrix_size,
             double weight_integral,
             Recurrence recurrence_fcn);

std::tuple<double, double, double> legendre_recursion_coeffs(std::size_t i);
std::tuple<double, double, double> lobatto_recursion_coeffs(std::size_t i);

} // namespace detail
} // namespace prometheus::integrate

PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::GaussFamily);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::RuleError);

//{{{ collection: std::hash for RuleError alternatives (so `std::hash<RuleError>` is enabled)
template <>
struct std::hash<prometheus::integrate::rule_error::DegreeOutOfRange>
{
    std::size_t
    operator()(const prometheus::integrate::rule_error::DegreeOutOfRange& e) const noexcept
    {
        return std::hash<std::size_t>{}(e.degree) ^ (std::hash<std::size_t>{}(e.maximum) << 1U);
    }
};

template <>
struct std::hash<prometheus::integrate::rule_error::PointCountOutOfRange>
{
    std::size_t
    operator()(const prometheus::integrate::rule_error::PointCountOutOfRange& e) const noexcept
    {
        std::size_t h = std::hash<prometheus::integrate::GaussFamily>{}(e.family);
        for (const std::size_t field : {e.point_count, e.minimum, e.maximum})
        {
            h = (h << 1U) ^ std::hash<std::size_t>{}(field);
        }
        return h;
    }
};

template <>
struct std::hash<prometheus::integrate::rule_error::GenerationFailed>
{
    std::size_t
    operator()(const prometheus::integrate::rule_error::GenerationFailed& e) const noexcept
    {
        std::size_t h = std::hash<prometheus::integrate::GaussFamily>{}(e.family);
        h = (h << 1U) ^ std::hash<std::size_t>{}(e.point_count);
        return (h << 1U) ^ std::hash<std::string>{}(e.message);
    }
};
//}}}
