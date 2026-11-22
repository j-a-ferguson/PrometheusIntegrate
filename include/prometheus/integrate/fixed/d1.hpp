#pragma once
//! Fixed quadrature for one-dimensional real-valued functions.
//!
//! There are two entry points:
//!
//! - `fixed_quad`, for integrating once without keeping the mapped rule.
//! - A reusable `Quadrature`, which stores the mapped rule for use with many integrands.

//{{{ crate includes
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <cmath>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>
//}}}

namespace prometheus::integrate::fixed::d1
{

class Node;
class Builder;
class Quadrature;

namespace detail
{
/// Maps `rule` from its reference interval onto every sub-interval of `domain` delimited by
/// `subdivisions`.
std::vector<Node>
mapped_nodes(Interval domain, const GaussRule& rule, std::span<const double> subdivisions);
} // namespace detail

//{{{ class: Node
/// A mapped point and weight for one-dimensional fixed quadrature.
///
/// Standard-layout with two `double`s, matching Rust's `#[repr(C)]`.
class Node
{
public:
    /// Returns the evaluation point.
    [[nodiscard]] constexpr double point() const noexcept { return point_; }
    /// Returns the quadrature weight.
    [[nodiscard]] constexpr double weight() const noexcept { return weight_; }

    bool operator==(const Node&) const = default;

private:
    friend std::vector<Node>
    detail::mapped_nodes(Interval, const GaussRule&, std::span<const double>);
    constexpr Node(double point, double weight) noexcept : point_{point}, weight_{weight} {}

    double point_;
    double weight_;
};
//}}}

//{{{ class: Builder
/// Consuming builder for a one-dimensional fixed quadrature.
///
/// Rust's `self`-consuming methods become `&&`-qualified member functions: call them on a
/// temporary, or `std::move` a named builder.
class Builder
{
public:
    /// Replaces the interior subdivision points. Empty input means no subdivisions.
    ///
    /// Fails unless every point is finite, strictly increasing, unique, and strictly interior to
    /// the integration domain.
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

    /// Maps the reference rule onto the configured domain and subdivisions.
    [[nodiscard]] Quadrature build() &&;

    bool operator==(const Builder&) const = default;

private:
    friend class Quadrature;
    Builder(Interval domain, GaussRule rule) : domain_{domain}, rule_{std::move(rule)} {}

    std::expected<Builder, ConfigError> set_subdivisions(std::vector<double> points) &&;

    Interval domain_;
    GaussRule rule_;
    std::vector<double> subdivisions_;
};
//}}}

//{{{ class: Quadrature
/// A reusable one-dimensional fixed quadrature.
class Quadrature
{
public:
    using const_iterator = std::vector<Node>::const_iterator;

    /// Starts a consuming builder for `domain` and `rule`.
    [[nodiscard]] static Builder builder(Interval domain, GaussRule rule)
    {
        return Builder{domain, std::move(rule)};
    }

    /// Rust `impl From<GaussRule> for Quadrature`: the rule on its own reference interval.
    explicit Quadrature(GaussRule rule);

    /// Returns the configured integration domain.
    [[nodiscard]] Interval domain() const noexcept { return domain_; }
    /// Returns the reference Gaussian rule.
    [[nodiscard]] const GaussRule& rule() const noexcept { return rule_; }
    /// Returns the validated interior subdivision points.
    [[nodiscard]] std::span<const double> subdivision_points() const noexcept
    {
        return subdivisions_;
    }
    /// Returns all mapped nodes.
    [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
    /// Returns the total number of mapped points.
    [[nodiscard]] std::size_t point_count() const noexcept { return nodes_.size(); }

    /// Rust `iter()` / `IntoIterator for &Quadrature`.
    [[nodiscard]] const_iterator begin() const noexcept { return nodes_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return nodes_.end(); }

    /// Integrates over the configured domain.
    ///
    /// Fails with `NonFiniteIntegrand` if `f` returns NaN or infinity. Exceptions thrown by `f`
    /// propagate to the caller.
    template <Integrand1d F>
    std::expected<double, IntegrationError> integrate(F&& f) const
    {
        double integral = 0.0;
        for (const Node& node : nodes_)
        {
            const double value = static_cast<double>(f(node.point()));
            if (!std::isfinite(value))
            {
                return std::unexpected{integration_error::NonFiniteIntegrand{
                    .point = evaluation_point::OneDimensional{node.point()}, .value = value}};
            }
            integral += node.weight() * value;
        }
        return integral;
    }

    /// Integrates over another interval by proportionally remapping every node and subdivision.
    template <Integrand1d F>
    std::expected<double, IntegrationError> integrate_over(Interval domain, F&& f) const
    {
        const double scale = domain.length() / domain_.length();
        double integral = 0.0;
        for (const Node& node : nodes_)
        {
            const double point = domain.lower() + scale * (node.point() - domain_.lower());
            const double value = static_cast<double>(f(point));
            if (!std::isfinite(value))
            {
                return std::unexpected{integration_error::NonFiniteIntegrand{
                    .point = evaluation_point::OneDimensional{point}, .value = value}};
            }
            integral += scale * node.weight() * value;
        }
        return integral;
    }

    bool operator==(const Quadrature&) const = default;

private:
    friend class Builder;
    Quadrature(Interval domain,
               GaussRule rule,
               std::vector<double> subdivisions,
               std::vector<Node> nodes)
        : domain_{domain}, rule_{std::move(rule)}, subdivisions_{std::move(subdivisions)},
          nodes_{std::move(nodes)}
    {
    }

    Interval domain_;
    GaussRule rule_;
    std::vector<double> subdivisions_;
    std::vector<Node> nodes_;
};
//}}}

//{{{ fun: fixed_quad
/// Builds a one-dimensional fixed quadrature and integrates `f` once.
template <Integrand1d F>
std::expected<double, IntegrationError> fixed_quad(F&& f, Builder builder)
{
    return std::move(builder).build().integrate(std::forward<F>(f));
}
//}}}

} // namespace prometheus::integrate::fixed::d1
