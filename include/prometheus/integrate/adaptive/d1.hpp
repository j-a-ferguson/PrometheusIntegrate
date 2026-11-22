#pragma once
//! Adaptive quadrature for one-dimensional real-valued functions.

//{{{ crate includes
#include "prometheus/integrate/common.hpp"
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/fixed/d1.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>
#include <vector>
//}}}

namespace prometheus::integrate::adaptive::d1
{

class AdaptiveQuadrature;

namespace detail
{

using FixedQuadrature = fixed::d1::Quadrature;

//{{{ struct: Region
struct Region
{
    Interval domain;
    std::size_t depth;
    double high_estimate;
    double error_estimate;
};
//}}}

//{{{ fun: evaluate_region
template <class F>
std::expected<Region, IntegrationError> evaluate_region(Interval domain,
                                                        std::size_t depth,
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
                  .depth = depth,
                  .high_estimate = *high_estimate,
                  .error_estimate = std::abs(*high_estimate - *low_estimate)};
}
//}}}

/// Global convergence test: `error <= absolute + relative * |integral|`.
[[nodiscard]] bool converged(const AdaptiveResult& result, Tolerance tolerance) noexcept;

/// Floating-point midpoint that avoids overflow for very wide intervals.
[[nodiscard]] double midpoint(Interval interval) noexcept;

/// Rust `iter().enumerate().filter(..).max_by(total_cmp)`: index of the region with the largest
/// error estimate among those passing `can_split`. Like `Iterator::max_by`, ties resolve to the
/// *last* maximal element.
template <class R, class Pred>
std::optional<std::size_t> argmax_error(const std::vector<R>& regions, Pred can_split)
{
    std::optional<std::size_t> best;
    for (std::size_t index = 0; index < regions.size(); ++index)
    {
        if (!can_split(regions[index]))
        {
            continue;
        }
        if (!best ||
            std::strong_order(regions[index].error_estimate, regions[*best].error_estimate) >= 0)
        {
            best = index;
        }
    }
    return best;
}

/// Rust `Vec::swap_remove`.
template <class T>
void swap_remove(std::vector<T>& values, std::size_t index)
{
    if (index + 1 != values.size())
    {
        values[index] = std::move(values.back());
    }
    values.pop_back();
}

} // namespace detail

//{{{ class: Builder
/// Consuming builder for one-dimensional adaptive quadrature.
class Builder
{
public:
    /// Sets the maximum number of bisections permitted from each initial region.
    [[nodiscard]] Builder max_depth(RefinementDepth max_depth) &&
    {
        max_depth_ = max_depth;
        return std::move(*this);
    }

    /// Replaces the initial interior subdivision points.
    template <integrate::detail::SubdivisionRange R>
    std::expected<Builder, ConfigError> subdivisions(R&& points) &&
    {
        return std::move(*this).set_subdivisions(
            integrate::detail::collect_points(std::forward<R>(points)));
    }

    std::expected<Builder, ConfigError> subdivisions(std::initializer_list<double> points) &&
    {
        return std::move(*this).set_subdivisions(std::vector<double>(points));
    }

    /// Validates the rule pair and constructs a reusable adaptive quadrature.
    ///
    /// Fails unless the high rule's actual exactness exceeds the low rule's.
    std::expected<AdaptiveQuadrature, ConfigError> build() &&;

    bool operator==(const Builder&) const = default;

private:
    friend class AdaptiveQuadrature;
    Builder(Interval domain, GaussRule low_rule, GaussRule high_rule, Tolerance tolerance)
        : domain_{domain}, low_rule_{std::move(low_rule)}, high_rule_{std::move(high_rule)},
          tolerance_{tolerance}
    {
    }

    std::expected<Builder, ConfigError> set_subdivisions(std::vector<double> points) &&;

    Interval domain_;
    GaussRule low_rule_;
    GaussRule high_rule_;
    Tolerance tolerance_;
    RefinementDepth max_depth_{};
    std::vector<double> subdivisions_;
};
//}}}

//{{{ class: AdaptiveQuadrature
/// Reusable one-dimensional adaptive quadrature configuration.
class AdaptiveQuadrature
{
public:
    /// Starts a consuming builder with a default maximum depth of 32.
    [[nodiscard]] static Builder
    builder(Interval domain, GaussRule low_rule, GaussRule high_rule, Tolerance tolerance)
    {
        return Builder{domain, std::move(low_rule), std::move(high_rule), tolerance};
    }

    /// Returns the complete integration domain.
    [[nodiscard]] Interval domain() const noexcept { return domain_; }
    /// Returns the low-order Gaussian rule.
    [[nodiscard]] const GaussRule& low_rule() const noexcept { return low_rule_.rule(); }
    /// Returns the high-order Gaussian rule.
    [[nodiscard]] const GaussRule& high_rule() const noexcept { return high_rule_.rule(); }
    /// Returns the global convergence tolerance.
    [[nodiscard]] Tolerance tolerance() const noexcept { return tolerance_; }
    /// Returns the maximum refinement depth.
    [[nodiscard]] RefinementDepth max_depth() const noexcept { return max_depth_; }
    /// Returns the validated initial subdivision points.
    [[nodiscard]] std::span<const double> subdivision_points() const noexcept
    {
        return subdivisions_;
    }

    /// Adaptively integrates `f` using a global error estimate.
    ///
    /// The returned integral is the sum of high-order estimates. On each iteration, the
    /// splittable terminal region with the largest estimated error is bisected.
    ///
    /// Fails with `NonFiniteIntegrand` if `f` returns NaN or infinity, `MaxDepthReached` if the
    /// tolerance cannot be met within the configured depth, or `NonProgressingInterval` if a
    /// floating-point midpoint is not strictly interior.
    template <Integrand1d F>
    std::expected<AdaptiveResult, IntegrationError> integrate(F&& f) const
    {
        using detail::Region;

        const std::size_t evaluations_per_region =
            low_rule_.point_count() + high_rule_.point_count();
        std::size_t evaluation_count = 0;
        std::vector<Region> regions;
        regions.reserve(subdivisions_.size() + 1);

        //{{{ com: evaluate the initial regions
        double lower = domain_.lower();
        auto add_initial = [&](double upper) -> std::optional<IntegrationError>
        {
            auto region = detail::evaluate_region(
                Interval::new_unchecked(lower, upper), 0, low_rule_, high_rule_, f);
            if (!region)
            {
                return std::move(region).error();
            }
            regions.push_back(*region);
            evaluation_count += evaluations_per_region;
            lower = upper;
            return std::nullopt;
        };
        for (const double upper : subdivisions_)
        {
            if (auto error = add_initial(upper))
            {
                return std::unexpected{std::move(*error)};
            }
        }
        if (auto error = add_initial(domain_.upper()))
        {
            return std::unexpected{std::move(*error)};
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
                regions, [&](const Region& region) { return region.depth < max_depth_.value(); });
            if (!index)
            {
                return std::unexpected{integration_error::MaxDepthReached{result}};
            }

            const Region parent = regions[*index];
            const double midpoint = detail::midpoint(parent.domain);
            if (midpoint <= parent.domain.lower() || midpoint >= parent.domain.upper())
            {
                return std::unexpected{integration_error::NonProgressingInterval{
                    .interval = parent.domain, .partial_result = result}};
            }

            const std::size_t child_depth = parent.depth + 1;
            auto left =
                detail::evaluate_region(Interval::new_unchecked(parent.domain.lower(), midpoint),
                                        child_depth,
                                        low_rule_,
                                        high_rule_,
                                        f);
            if (!left)
            {
                return std::unexpected{std::move(left).error()};
            }
            auto right =
                detail::evaluate_region(Interval::new_unchecked(midpoint, parent.domain.upper()),
                                        child_depth,
                                        low_rule_,
                                        high_rule_,
                                        f);
            if (!right)
            {
                return std::unexpected{std::move(right).error()};
            }

            evaluation_count += 2 * evaluations_per_region;
            global_integral += left->high_estimate + right->high_estimate - parent.high_estimate;
            global_error = std::max(global_error - parent.error_estimate, 0.0) +
                           left->error_estimate + right->error_estimate;
            detail::swap_remove(regions, *index);
            regions.push_back(*left);
            regions.push_back(*right);
        }
    }

    bool operator==(const AdaptiveQuadrature&) const = default;

private:
    friend class Builder;
    AdaptiveQuadrature(Interval domain,
                       detail::FixedQuadrature low_rule,
                       detail::FixedQuadrature high_rule,
                       Tolerance tolerance,
                       RefinementDepth max_depth,
                       std::vector<double> subdivisions)
        : domain_{domain}, low_rule_{std::move(low_rule)}, high_rule_{std::move(high_rule)},
          tolerance_{tolerance}, max_depth_{max_depth}, subdivisions_{std::move(subdivisions)}
    {
    }

    Interval domain_;
    detail::FixedQuadrature low_rule_;
    detail::FixedQuadrature high_rule_;
    Tolerance tolerance_;
    RefinementDepth max_depth_;
    std::vector<double> subdivisions_;
};
//}}}

//{{{ fun: adaptive_quad
/// Builds a one-dimensional adaptive quadrature and integrates `f` once.
///
/// Rust: `Ok(builder.build()?.integrate(f)?)`, where each `?` converts into `OptionsError` via
/// `From`. Here the same pipeline is spelled with C++23's monadic `std::expected` operations.
template <Integrand1d F>
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

} // namespace prometheus::integrate::adaptive::d1
