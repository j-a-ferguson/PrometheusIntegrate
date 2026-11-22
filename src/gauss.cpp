//! Gauss quadrature rules for one-dimensional real-valued integrands.
//!
//! Rules are generated with the Golub-Welsch algorithm; the symmetric tridiagonal (Jacobi)
//! eigenproblem is solved with Armadillo's `eig_sym` (LAPACK `dsyevd`), replacing
//! `prometheus_linalg::DMatrix::symeig`.

#include "prometheus/integrate/gauss.hpp"

#include <armadillo>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace prometheus::integrate
{

namespace
{

//{{{ static: MAX_DEGREE
/// Largest requested polynomial exactness degree included in each cached rule set.
constexpr std::size_t MAX_DEGREE = 100;
constexpr PolynomialDegree CACHED_MAXIMUM_DEGREE = PolynomialDegree::new_unchecked(MAX_DEGREE);
//}}}

//{{{ collection: GaussFamily private helpers
/// Number of points needed for a rule with at least `degree` polynomial exactness.
///
/// An `n`-point Legendre rule has exactness `2n - 1`, while an `n`-point Lobatto rule has
/// exactness `2n - 3`. Consequently, an even requested degree selects a rule whose actual
/// exactness is one degree higher.
std::size_t point_count_for_degree(GaussFamily family, std::size_t degree) noexcept
{
    switch (family)
    {
    case GaussFamily::Legendre:
        return degree / 2 + 1;
    case GaussFamily::Lobatto:
        return degree / 2 + 2;
    }
    std::unreachable();
}

/// Rust `point_count.checked_mul(2)?.checked_sub(k)`.
std::optional<std::size_t> exactness_for_point_count(GaussFamily family,
                                                     std::size_t point_count) noexcept
{
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    const auto twice_minus = [&](std::size_t k) -> std::optional<std::size_t>
    {
        if (point_count > max / 2)
        {
            return std::nullopt;
        }
        const std::size_t twice = point_count * 2;
        return twice >= k ? std::optional{twice - k} : std::nullopt;
    };
    if (family == GaussFamily::Legendre && point_count >= 1)
    {
        return twice_minus(1);
    }
    if (family == GaussFamily::Lobatto && point_count >= 2)
    {
        return twice_minus(3);
    }
    return std::nullopt;
}

/// Minimum supported point count for the quadrature family.
std::size_t minimum_point_count(GaussFamily family) noexcept
{
    switch (family)
    {
    case GaussFamily::Legendre:
        return 1;
    case GaussFamily::Lobatto:
        return 2;
    }
    std::unreachable();
}

std::size_t maximum_point_count(GaussFamily family) noexcept
{
    return point_count_for_degree(family, MAX_DEGREE);
}
//}}}

//{{{ fun: validate_degree
std::expected<void, RuleError> validate_degree(std::size_t degree, std::size_t maximum)
{
    if (degree > maximum)
    {
        return std::unexpected{rule_error::DegreeOutOfRange{.degree = degree, .maximum = maximum}};
    }
    return {};
}
//}}}

//{{{ fun: validate_point_count
std::expected<void, RuleError> validate_point_count(GaussFamily family, std::size_t point_count)
{
    const std::size_t minimum = minimum_point_count(family);
    const std::size_t maximum = maximum_point_count(family);
    if (point_count < minimum || point_count > maximum)
    {
        return std::unexpected{rule_error::PointCountOutOfRange{
            .family = family, .point_count = point_count, .minimum = minimum, .maximum = maximum}};
    }
    return {};
}
//}}}

//{{{ fun: legendre
/// Evaluates the Legendre polynomial `P_n(x)` by its three-term recurrence.
double legendre(std::size_t n, double x) noexcept
{
    double leg_n = 1.0;
    double leg_1 = 1.0;
    double leg_2 = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto ii = static_cast<double>(i);
        const double ai = (2.0 * ii + 1.0) / (ii + 1.0);
        const double ci = ii / (ii + 1.0);
        leg_n = ai * x * leg_1 - ci * leg_2;
        leg_2 = leg_1;
        leg_1 = leg_n;
    }
    return leg_n;
}
//}}}

//{{{ fun: build_gauss_rule
/// Builds a quadrature rule with exactly `point_count` points.
std::expected<GaussRule, RuleError> build_gauss_rule(GaussFamily family, std::size_t point_count)
{
    if (auto valid = validate_point_count(family, point_count); !valid)
    {
        return std::unexpected{std::move(valid).error()};
    }

    std::vector<double> points;
    std::vector<double> weights;

    switch (family)
    {
    case GaussFamily::Legendre:
    {
        auto generated = detail::golub_welsch(family,
                                              point_count,
                                              point_count,
                                              weight_integral(family),
                                              detail::legendre_recursion_coeffs);
        if (!generated)
        {
            return std::unexpected{std::move(generated).error()};
        }
        std::tie(points, weights) = std::move(*generated);
        break;
    }
    case GaussFamily::Lobatto:
    {
        const auto n = static_cast<double>(point_count);
        const double endpoint_weight = 2.0 / (n * (n - 1.0));
        points.reserve(point_count);
        weights.reserve(point_count);

        points.push_back(-1.0);
        weights.push_back(endpoint_weight);

        if (point_count > 2)
        {
            const std::size_t interior_count = point_count - 2;
            auto generated = detail::golub_welsch(family,
                                                  point_count,
                                                  interior_count,
                                                  weight_integral(family),
                                                  detail::lobatto_recursion_coeffs);
            if (!generated)
            {
                return std::unexpected{std::move(generated).error()};
            }
            for (const double point : generated->first)
            {
                const double polynomial = legendre(point_count - 1, point);
                points.push_back(point);
                weights.push_back(endpoint_weight / (polynomial * polynomial));
            }
        }

        points.push_back(1.0);
        weights.push_back(endpoint_weight);
        break;
    }
    }

    return detail::gauss_rule_from_points_weights(family, std::move(points), std::move(weights));
}
//}}}

} // namespace

//{{{ fun: GaussFamily free functions
double weight_integral(GaussFamily family) noexcept
{
    switch (family)
    {
    case GaussFamily::Legendre:
        return 2.0;
    case GaussFamily::Lobatto:
        return 4.0 / 3.0;
    }
    std::unreachable();
}

std::pair<double, double> range(GaussFamily family) noexcept
{
    switch (family)
    {
    case GaussFamily::Legendre:
    case GaussFamily::Lobatto:
        return {-1.0, 1.0};
    }
    std::unreachable();
}
//}}}

//{{{ collection: cached rule sets
std::expected<Ref<GaussRuleSet>, RuleError> legendre_rules()
{
    // Magic static: initialised exactly once, thread-safely (Rust `OnceLock::get_or_init`).
    static const auto rules =
        GaussRuleSet::through_degree(GaussFamily::Legendre, CACHED_MAXIMUM_DEGREE);
    if (!rules)
    {
        return std::unexpected{rules.error()};
    }
    return std::cref(*rules);
}

std::expected<Ref<GaussRuleSet>, RuleError> lobatto_rules()
{
    static const auto rules =
        GaussRuleSet::through_degree(GaussFamily::Lobatto, CACHED_MAXIMUM_DEGREE);
    if (!rules)
    {
        return std::unexpected{rules.error()};
    }
    return std::cref(*rules);
}
//}}}

//{{{ impl: GaussRule
std::expected<GaussRule, RuleError> GaussRule::for_degree(GaussFamily family,
                                                          PolynomialDegree degree)
{
    return validate_degree(degree.value(), MAX_DEGREE)
        .and_then(
            [&]
            { return build_gauss_rule(family, point_count_for_degree(family, degree.value())); });
}

std::expected<GaussRule, RuleError> GaussRule::with_point_count(GaussFamily family,
                                                                PointCount point_count)
{
    return validate_point_count(family, point_count.value())
        .and_then([&] { return build_gauss_rule(family, point_count.value()); });
}
//}}}

//{{{ impl: GaussRuleSet
std::expected<GaussRuleSet, RuleError> GaussRuleSet::through_degree(GaussFamily family,
                                                                    PolynomialDegree degree)
{
    if (auto valid = validate_degree(degree.value(), MAX_DEGREE); !valid)
    {
        return std::unexpected{std::move(valid).error()};
    }

    const std::size_t minimum = prometheus::integrate::minimum_point_count(family);
    const std::size_t maximum = point_count_for_degree(family, degree.value());
    std::vector<GaussRule> rules;
    rules.reserve(maximum - minimum + 1);
    for (std::size_t point_count = minimum; point_count <= maximum; ++point_count)
    {
        auto rule = build_gauss_rule(family, point_count);
        if (!rule)
        {
            return std::unexpected{std::move(rule).error()};
        }
        rules.push_back(std::move(*rule));
    }

    return GaussRuleSet{family, degree, std::move(rules)};
}

std::expected<Ref<GaussRule>, RuleError>
GaussRuleSet::rule_for_degree(PolynomialDegree degree) const
{
    return rule_for_degree_value(degree.value());
}

std::expected<Ref<GaussRule>, RuleError>
GaussRuleSet::rule_by_point_count(PointCount point_count) const
{
    return rule_by_point_count_value(point_count.value());
}

std::expected<Ref<GaussRule>, RuleError>
GaussRuleSet::rule_for_degree_value(std::size_t degree) const
{
    return validate_degree(degree, maximum_degree_.value())
        .and_then([&]
                  { return rule_by_point_count_value(point_count_for_degree(family_, degree)); });
}

std::expected<Ref<GaussRule>, RuleError>
GaussRuleSet::rule_by_point_count_value(std::size_t point_count) const
{
    const std::size_t minimum = prometheus::integrate::minimum_point_count(family_);
    const std::size_t maximum = maximum_point_count().value();
    if (point_count < minimum || point_count > maximum)
    {
        return std::unexpected{rule_error::PointCountOutOfRange{
            .family = family_, .point_count = point_count, .minimum = minimum, .maximum = maximum}};
    }
    return std::cref(rules_[point_count - minimum]);
}

PointCount GaussRuleSet::minimum_point_count() const noexcept
{
    return PointCount::new_unchecked(prometheus::integrate::minimum_point_count(family_));
}

PointCount GaussRuleSet::maximum_point_count() const noexcept
{
    return PointCount::new_unchecked(point_count_for_degree(family_, maximum_degree_.value()));
}
//}}}

namespace detail
{

//{{{ fun: gauss_rule_from_points_weights
std::expected<GaussRule, RuleError> gauss_rule_from_points_weights(GaussFamily family,
                                                                   std::vector<double> points,
                                                                   std::vector<double> weights)
{
    if (points.size() != weights.size())
    {
        return std::unexpected{rule_error::GenerationFailed{
            .family = family,
            .point_count = points.size(),
            .message = std::format(
                "backend produced {} points but {} weights", points.size(), weights.size())}};
    }
    const auto exactness = exactness_for_point_count(family, points.size());
    if (!exactness)
    {
        return std::unexpected{
            rule_error::GenerationFailed{.family = family,
                                         .point_count = points.size(),
                                         .message = "backend produced an invalid point count"}};
    }
    return GaussRule{
        family, PolynomialDegree::new_unchecked(*exactness), std::move(points), std::move(weights)};
}
//}}}

//{{{ fun: golub_welsch
/// Golub-Welsch algorithm.
///
/// Starting from the three-term recurrence `p_{i+1}(x) = (a_i x + b_i) p_i(x) - c_i p_{i-1}(x)`,
/// the roots of `p_n` (the quadrature points) are the eigenvalues of the symmetric tridiagonal
/// Jacobi matrix
///
///     J_ii = alpha_i = -b_i / a_i,   J_{i,i+1} = J_{i+1,i} = beta_i = sqrt(c_{i+1} / (a_i a_{i+1}))
///
/// and the weights are `weight_integral * q_0(t_j)^2`, the squared first component of each
/// normalised eigenvector.
std::expected<std::pair<std::vector<double>, std::vector<double>>, RuleError>
golub_welsch(GaussFamily family,
             std::size_t rule_point_count,
             std::size_t matrix_size,
             double weight_integral,
             Recurrence recurrence_fcn)
{
    if (matrix_size == 0)
    {
        return std::unexpected{
            rule_error::GenerationFailed{.family = family,
                                         .point_count = rule_point_count,
                                         .message = "eigendecomposition matrix must be nonempty"}};
    }

    const auto n = static_cast<arma::uword>(matrix_size);
    arma::mat tmat(n, n, arma::fill::zeros);

    for (arma::uword i = 0; i < n; ++i)
    {
        const auto [ai, bi, ci] = recurrence_fcn(i);
        tmat(i, i) = -(bi / ai);

        if (i + 1 < n)
        {
            const auto [aj, bj, cj] = recurrence_fcn(i + 1);
            const double beta = std::sqrt(cj / (ai * aj));
            tmat(i, i + 1) = beta;
            tmat(i + 1, i) = beta;
        }
    }

    //{{{ com: eigendecompose
    arma::vec eigvals;
    arma::mat eigvecs;
    if (!arma::eig_sym(eigvals, eigvecs, tmat))
    {
        return std::unexpected{rule_error::GenerationFailed{
            .family = family,
            .point_count = rule_point_count,
            .message = "armadillo eig_sym failed to decompose the Jacobi matrix"}};
    }
    //}}}

    //{{{ com: compute quadrature points and weights from eigenvalues and eigenvectors
    std::vector<std::size_t> order(matrix_size);
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Rust sorts by `f64::total_cmp`, which is IEEE totalOrder — `std::strong_order` on doubles.
    std::ranges::sort(order,
                      [&](std::size_t left, std::size_t right)
                      { return std::strong_order(eigvals(left), eigvals(right)) < 0; });

    std::vector<double> points;
    std::vector<double> weights;
    points.reserve(matrix_size);
    weights.reserve(matrix_size);
    for (const std::size_t j : order)
    {
        const double first_component = eigvecs(0, j);
        points.push_back(eigvals(j));
        weights.push_back(first_component * first_component * weight_integral);
    }
    //}}}

    return std::pair{std::move(points), std::move(weights)};
}
//}}}

//{{{ fun: legendre_recursion_coeffs
/// Recurrence coefficients `(a, b, c)` for the i'th Legendre polynomial.
std::tuple<double, double, double> legendre_recursion_coeffs(std::size_t i)
{
    const auto i_f64 = static_cast<double>(i);
    const double ai = (2.0 * i_f64 + 1.0) / (i_f64 + 1.0);
    const double bi = 0.0;
    const double ci = i_f64 / (i_f64 + 1.0);
    return {ai, bi, ci};
}
//}}}

//{{{ fun: lobatto_recursion_coeffs
/// Recurrence coefficients `(a, b, c)` for the Jacobi (1, 1) polynomials whose roots are the
/// interior Lobatto nodes.
std::tuple<double, double, double> lobatto_recursion_coeffs(std::size_t i)
{
    const auto i_f64 = static_cast<double>(i);
    const double ai = ((2.0 * i_f64 + 3.0) * (i_f64 + 2.0)) / ((i_f64 + 1.0) * (i_f64 + 3.0));
    const double bi = 0.0;
    const double ci = ((i_f64 + 1.0) * (i_f64 + 2.0)) / ((i_f64 + 3.0) * (i_f64 + 1.0));
    return {ai, bi, ci};
}
//}}}

} // namespace detail
} // namespace prometheus::integrate

//{{{ collection: Display
namespace ti = prometheus::integrate;
namespace re = prometheus::integrate::rule_error;

std::string ti::detail::display(const ti::GaussFamily& value)
{
    switch (value)
    {
    case ti::GaussFamily::Legendre:
        return std::format("legendre");
    case ti::GaussFamily::Lobatto:
        return std::format("lobatto");
    }
    std::unreachable();
}

std::string ti::detail::display(const ti::RuleError& value)
{
    return std::visit(
        ti::detail::overloaded{
            [&](const re::DegreeOutOfRange& e)
            {
                return std::format("polynomial degree {} exceeds the supported maximum of {}",
                                   e.degree,
                                   e.maximum);
            },
            [&](const re::PointCountOutOfRange& e)
            {
                return std::format("{} point count {} is outside supported range {}..={}",
                                   e.family,
                                   e.point_count,
                                   e.minimum,
                                   e.maximum);
            },
            [&](const re::GenerationFailed& e)
            {
                return std::format("failed to generate a {} rule with {} points: {}",
                                   e.family,
                                   e.point_count,
                                   e.message);
            },
        },
        value);
}
//}}}
