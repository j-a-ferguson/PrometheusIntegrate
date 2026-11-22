#pragma once
//! Adaptive tensor-product quadrature for two-dimensional real-valued functions.

//{{{ crate includes
#include "prometheus/integrate/adaptive/d1.hpp"
#include "prometheus/integrate/common.hpp"
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/fixed/d2.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>
#include <vector>
//}}}

namespace prometheus::integrate::adaptive::d2
{

class AdaptiveQuadrature;

namespace detail
{

using FixedQuadrature = fixed::d2::Quadrature;
using d1::detail::argmax_error;
using d1::detail::converged;
using d1::detail::midpoint;
using d1::detail::swap_remove;
using fixed::d2::TensorRule;

//{{{ struct: Region
struct Region
{
    Rectangle domain;
    std::size_t u_depth;
    std::size_t v_depth;
    double high_estimate;
    double error_estimate;
};
//}}}

//{{{ fun: evaluate_region
template <class F>
std::expected<Region, IntegrationError> evaluate_region(Rectangle domain,
                                                        std::size_t u_depth,
                                                        std::size_t v_depth,
                                                        const FixedQuadrature& low_rule,
                                                        const FixedQuadrature& high_rule,
                                                        F& f)
{
    auto low_estimate = low_rule.integrate_over(domain, f);
    if (!low_estimate)
    {
        return std::unexpected{std::move(low_estimate).error()};
    }
    auto high_estimate = high_rule.integrate_over(domain, f);
    if (!high_estimate)
    {
        return std::unexpected{std::move(high_estimate).error()};
    }
    return Region{.domain = domain,
                  .u_depth = u_depth,
                  .v_depth = v_depth,
                  .high_estimate = *high_estimate,
                  .error_estimate = std::abs(*high_estimate - *low_estimate)};
}
//}}}

/// Pushes `NonIncreasingRuleExactness` for `axis` unless `high` is strictly more exact.
void validate_rule_pair(const GaussRule& low,
                        const GaussRule& high,
                        RuleAxis axis,
                        std::vector<ConfigIssue>& issues);

/// `[lower, s_0, s_1, ..., upper]`.
[[nodiscard]] std::vector<double> axis_bounds(Interval interval,
                                              std::span<const double> subdivisions);

/// Children of `domain` when splitting each axis that has a midpoint (`u`-major order).
[[nodiscard]] std::vector<Rectangle>
child_domains(Rectangle domain, std::optional<double> u_midpoint, std::optional<double> v_midpoint);

} // namespace detail

//{{{ class: Builder
/// Consuming builder for two-dimensional adaptive quadrature.
class Builder
{
public:
    /// Sets the maximum refinement depth independently on each axis.
    [[nodiscard]] Builder max_depth(AxisDepths max_depth) &&
    {
        max_depth_ = max_depth;
        return std::move(*this);
    }

    /// Replaces the initial interior subdivision coordinates on both axes.
    template <integrate::detail::SubdivisionRange U, integrate::detail::SubdivisionRange V>
    std::expected<Builder, ConfigError> subdivisions(U&& u_points, V&& v_points) &&
    {
        return std::move(*this).set_subdivisions(
            integrate::detail::collect_points(std::forward<U>(u_points)),
            integrate::detail::collect_points(std::forward<V>(v_points)));
    }

    std::expected<Builder, ConfigError> subdivisions(std::initializer_list<double> u_points,
                                                     std::initializer_list<double> v_points) &&
    {
        return std::move(*this).set_subdivisions(std::vector<double>(u_points),
                                                 std::vector<double>(v_points));
    }

    /// Validates both axis rule pairs and constructs a reusable adaptive quadrature.
    ///
    /// Reports every axis whose high rule's actual exactness does not exceed its low rule's.
    std::expected<AdaptiveQuadrature, ConfigError> build() &&;

    bool operator==(const Builder&) const = default;

private:
    friend class AdaptiveQuadrature;
    Builder(Rectangle domain,
            detail::TensorRule low_rule,
            detail::TensorRule high_rule,
            Tolerance tolerance)
        : domain_{domain}, low_rule_{std::move(low_rule)}, high_rule_{std::move(high_rule)},
          tolerance_{tolerance}
    {
    }

    std::expected<Builder, ConfigError> set_subdivisions(std::vector<double> u,
                                                         std::vector<double> v) &&;

    Rectangle domain_;
    detail::TensorRule low_rule_;
    detail::TensorRule high_rule_;
    Tolerance tolerance_;
    AxisDepths max_depth_{};
    std::vector<double> u_subdivisions_;
    std::vector<double> v_subdivisions_;
};
//}}}

//{{{ class: AdaptiveQuadrature
/// Reusable two-dimensional adaptive tensor-product quadrature configuration.
class AdaptiveQuadrature
{
public:
    /// Starts a consuming builder with a default maximum depth of 32 on both axes.
    [[nodiscard]] static Builder builder(Rectangle domain,
                                         detail::TensorRule low_rule,
                                         detail::TensorRule high_rule,
                                         Tolerance tolerance)
    {
        return Builder{domain, std::move(low_rule), std::move(high_rule), tolerance};
    }

    /// Returns the complete integration domain.
    [[nodiscard]] Rectangle domain() const noexcept { return domain_; }
    /// Returns the low-order tensor-product rule.
    [[nodiscard]] const detail::TensorRule& low_rule() const noexcept { return low_rule_.rule(); }
    /// Returns the high-order tensor-product rule.
    [[nodiscard]] const detail::TensorRule& high_rule() const noexcept { return high_rule_.rule(); }
    /// Returns the global convergence tolerance.
    [[nodiscard]] Tolerance tolerance() const noexcept { return tolerance_; }
    /// Returns the maximum per-axis refinement depths.
    [[nodiscard]] AxisDepths max_depth() const noexcept { return max_depth_; }
    /// Returns the validated initial `u`-axis subdivision coordinates.
    [[nodiscard]] std::span<const double> u_subdivision_points() const noexcept
    {
        return u_subdivisions_;
    }
    /// Returns the validated initial `v`-axis subdivision coordinates.
    [[nodiscard]] std::span<const double> v_subdivision_points() const noexcept
    {
        return v_subdivisions_;
    }

    /// Adaptively integrates `f` using a global error estimate.
    ///
    /// Both axes are split while both remain below their limits. Once one axis reaches its
    /// maximum depth, only the other axis is split.
    template <Integrand2d F>
    std::expected<AdaptiveResult, IntegrationError> integrate(F&& f) const
    {
        using detail::Region;

        const std::size_t evaluations_per_region =
            low_rule_.point_count() + high_rule_.point_count();
        const auto u_bounds = detail::axis_bounds(domain_.u(), u_subdivisions_);
        const auto v_bounds = detail::axis_bounds(domain_.v(), v_subdivisions_);
        std::vector<Region> regions;
        regions.reserve((u_bounds.size() - 1) * (v_bounds.size() - 1));
        std::size_t evaluation_count = 0;

        //{{{ com: evaluate the initial grid of regions (Rust `windows(2)` over both axes)
        for (std::size_t i = 0; i + 1 < u_bounds.size(); ++i)
        {
            for (std::size_t j = 0; j + 1 < v_bounds.size(); ++j)
            {
                const Rectangle domain{Interval::new_unchecked(u_bounds[i], u_bounds[i + 1]),
                                       Interval::new_unchecked(v_bounds[j], v_bounds[j + 1])};
                auto region = detail::evaluate_region(domain, 0, 0, low_rule_, high_rule_, f);
                if (!region)
                {
                    return std::unexpected{std::move(region).error()};
                }
                regions.push_back(*region);
                evaluation_count += evaluations_per_region;
            }
        }
        //}}}

        double global_integral = 0.0;
        double global_error = 0.0;
        for (const Region& region : regions)
        {
            global_integral += region.high_estimate;
        }
        for (const Region& region : regions)
        {
            global_error += region.error_estimate;
        }

        while (true)
        {
            const AdaptiveResult result{
                global_integral, global_error, regions.size(), evaluation_count};
            if (detail::converged(result, tolerance_))
            {
                return result;
            }

            const auto index = detail::argmax_error(
                regions, [&](const Region& region) { return can_split(region); });
            if (!index)
            {
                return std::unexpected{integration_error::MaxDepthReached{result}};
            }

            const Region parent = regions[*index];
            const bool split_u = parent.u_depth < max_depth_.u().value();
            const bool split_v = parent.v_depth < max_depth_.v().value();
            const auto u_midpoint =
                split_u ? std::optional{detail::midpoint(parent.domain.u())} : std::nullopt;
            const auto v_midpoint =
                split_v ? std::optional{detail::midpoint(parent.domain.v())} : std::nullopt;

            const auto not_interior = [](std::optional<double> mid, Interval axis)
            { return mid.has_value() && (*mid <= axis.lower() || *mid >= axis.upper()); };
            if (not_interior(u_midpoint, parent.domain.u()) ||
                not_interior(v_midpoint, parent.domain.v()))
            {
                return std::unexpected{integration_error::NonProgressingRectangle{
                    .rectangle = parent.domain, .partial_result = result}};
            }

            const auto child_domains = detail::child_domains(parent.domain, u_midpoint, v_midpoint);
            const std::size_t u_depth = parent.u_depth + (split_u ? 1U : 0U);
            const std::size_t v_depth = parent.v_depth + (split_v ? 1U : 0U);
            const std::size_t child_count = child_domains.size();
            std::vector<Region> children;
            children.reserve(child_count);
            for (const Rectangle& domain : child_domains)
            {
                auto child =
                    detail::evaluate_region(domain, u_depth, v_depth, low_rule_, high_rule_, f);
                if (!child)
                {
                    return std::unexpected{std::move(child).error()};
                }
                children.push_back(*child);
            }

            double child_integral = 0.0;
            double child_error = 0.0;
            for (const Region& child : children)
            {
                child_integral += child.high_estimate;
            }
            for (const Region& child : children)
            {
                child_error += child.error_estimate;
            }
            evaluation_count += child_count * evaluations_per_region;
            global_integral += child_integral - parent.high_estimate;
            global_error = std::max(global_error - parent.error_estimate, 0.0) + child_error;
            detail::swap_remove(regions, *index);
            regions.insert(regions.end(), children.begin(), children.end());
        }
    }

    bool operator==(const AdaptiveQuadrature&) const = default;

private:
    friend class Builder;
    AdaptiveQuadrature(Rectangle domain,
                       detail::FixedQuadrature low_rule,
                       detail::FixedQuadrature high_rule,
                       Tolerance tolerance,
                       AxisDepths max_depth,
                       std::vector<double> u_subdivisions,
                       std::vector<double> v_subdivisions)
        : domain_{domain}, low_rule_{std::move(low_rule)}, high_rule_{std::move(high_rule)},
          tolerance_{tolerance}, max_depth_{max_depth}, u_subdivisions_{std::move(u_subdivisions)},
          v_subdivisions_{std::move(v_subdivisions)}
    {
    }

    [[nodiscard]] bool can_split(const detail::Region& region) const noexcept
    {
        return region.u_depth < max_depth_.u().value() || region.v_depth < max_depth_.v().value();
    }

    Rectangle domain_;
    detail::FixedQuadrature low_rule_;
    detail::FixedQuadrature high_rule_;
    Tolerance tolerance_;
    AxisDepths max_depth_;
    std::vector<double> u_subdivisions_;
    std::vector<double> v_subdivisions_;
};
//}}}

//{{{ fun: adaptive_quad
/// Builds a two-dimensional adaptive quadrature and integrates `f` once.
template <Integrand2d F>
std::expected<AdaptiveResult, OptionsError> adaptive_quad(F&& f, Builder builder)
{
    return std::move(builder)
        .build()
        .transform_error([](ConfigError error) { return OptionsError{std::move(error)}; })
        .and_then(
            [&](const AdaptiveQuadrature& quadrature)
            {
                return quadrature.integrate(f).transform_error(
                    [](IntegrationError error) { return OptionsError{std::move(error)}; });
            });
}
//}}}

} // namespace prometheus::integrate::adaptive::d2
